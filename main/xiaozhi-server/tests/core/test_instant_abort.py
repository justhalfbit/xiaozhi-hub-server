"""识别中、思考中、回答中打断都立即干净：旧的一轮迟到的识别结果、模型输出、合成语音一律丢掉，上下文里不留没回答的提问。"""
import asyncio
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import core.handle.receiveAudioHandle as receive
import core.providers.asr.base as asr_base
from core.connection import ConnectionHandler
from core.handle.abortHandle import handleAbortMessage
from core.handle.sendAudioHandle import sendAudioMessage
from core.providers.asr.base import ASRProviderBase
from core.providers.tts.dto.dto import SentenceType
from core.utils import turn_gen
from core.utils.dialogue import Dialogue


def make_conn(llm=None):
    h = ConnectionHandler.__new__(ConnectionHandler)
    h.logger = MagicMock()
    h.dialogue = Dialogue()
    h.tts = MagicMock()
    h.llm = llm or MagicMock(spec=[])
    h.memory = None
    h.intent_type = "nointent"
    h.config = {}
    h.current_speaker = None
    h.introduced_speakers = set()
    h.system_introduced_speakers = set()
    h.features = {"emoji": False}
    h.client_abort = False
    h.client_is_speaking = False
    h.client_listen_mode = "auto"
    h.close_after_chat = False
    h.need_bind = False
    h.max_output_size = 0
    h.sentence_id = None
    h.session_id = "s"
    h.websocket = SimpleNamespace(send=AsyncMock())
    h.clear_queues = MagicMock()
    h.clearSpeakStatus = MagicMock()
    h.reset_audio_states = MagicMock()
    h.asr_audio_queue = queue.Queue()
    h.executor = MagicMock()
    h.voiceprint_provider = None
    h.read_config_from_api = False
    return h


def tts_texts(h):
    return [c.args[0].content_detail for c in h.tts.tts_text_queue.put.call_args_list
            if getattr(c.args[0], "content_detail", None)]


def contents(h):
    return [(m.role, m.content) for m in h.dialogue.dialogue]


def wait_for(cond, timeout=5):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "等不到"
        time.sleep(0.01)


# ---------- 识别中打断 ----------

class SlowASR(ASRProviderBase):
    """识别要一会儿：started 后等 release 才出结果。"""

    def __init__(self, text="今天天气怎么样"):
        self.text = text
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def speech_to_text_wrapper(self, pcm_data, session_id):
        self.started.set()
        await self.release.wait()
        return self.text, None

    async def speech_to_text(self, *a, **kw):
        return self.text, None


@pytest.fixture
def chats(monkeypatch):
    """记下识别完交给 startToChat 的话（不真的问模型）。"""
    asked = []

    async def fake_start(conn, text, gen=None):
        asked.append((text, gen))

    monkeypatch.setattr(asr_base, "startToChat", fake_start)
    monkeypatch.setattr(asr_base, "correct_and_record", lambda conn, t: t)
    return asked


async def test_recognition_finishing_after_abort_is_dropped(chats):
    h, asr = make_conn(), SlowASR()
    h.features = {"vad_end": True}
    task = asyncio.create_task(asr.handle_voice_stop(h, [b"\0" * 32000]))
    await asr.started.wait()
    await handleAbortMessage(h, drop_audio=True)     # 识别中喊了唤醒词
    asr.release.set()
    await task
    assert chats == []
    # 只有打断的确认，没有"没识别出字"（设备已经在听下一句了）
    sent = [json.loads(c.args[0]) for c in h.websocket.send.call_args_list]
    assert [(m["type"], m.get("state")) for m in sent] == [("tts", "stop")]


async def test_recognition_without_abort_goes_to_the_model(chats):
    h, asr = make_conn(), SlowASR()
    asr.release.set()
    await asr.handle_voice_stop(h, [b"\0" * 32000])
    assert chats == [("今天天气怎么样", turn_gen.current(h))]


async def test_device_abort_drops_audio_queued_behind_the_recognition():
    h = make_conn()
    for _ in range(5):
        h.asr_audio_queue.put(b"wake word")
    await handleAbortMessage(h, drop_audio=True)
    assert h.asr_audio_queue.empty()
    h.reset_audio_states.assert_called_once()


async def test_hub_side_abort_keeps_the_new_question_audio():
    # 新问题触发的打断（startToChat 里）不能把用户接着说的话丢掉
    h = make_conn()
    h.asr_audio_queue.put(b"next words")
    await handleAbortMessage(h)
    assert h.asr_audio_queue.qsize() == 1
    h.reset_audio_states.assert_not_called()


async def test_start_to_chat_drops_a_sentence_aborted_after_it_was_said():
    h = make_conn()
    gen = turn_gen.current(h)
    await handleAbortMessage(h, drop_audio=True)
    await receive.startToChat(h, "旧的问题", gen=gen)
    h.executor.submit.assert_not_called()
    h.websocket.send.assert_awaited_once()   # 只有打断回的 tts stop，没有 stt


async def test_start_to_chat_drops_a_sentence_aborted_during_intent(monkeypatch):
    h = make_conn()
    in_intent, release = asyncio.Event(), asyncio.Event()

    async def call_name(conn, text):
        return text

    async def slow_intent(conn, text):
        in_intent.set()
        await release.wait()
        return False

    stt = AsyncMock()
    monkeypatch.setattr(receive, "handle_call_name", call_name)
    monkeypatch.setattr(receive, "handle_user_intent", slow_intent)
    monkeypatch.setattr(receive, "send_stt_message", stt)
    task = asyncio.create_task(receive.startToChat(h, "旧的问题", gen=turn_gen.current(h)))
    await in_intent.wait()
    await handleAbortMessage(h, drop_audio=True)
    release.set()
    await task
    stt.assert_not_awaited()
    h.executor.submit.assert_not_called()


async def test_start_to_chat_hands_its_turn_to_chat(monkeypatch):
    h = make_conn()

    async def same(conn, text):
        return text

    async def no_intent(conn, text):
        return False

    monkeypatch.setattr(receive, "handle_call_name", same)
    monkeypatch.setattr(receive, "handle_user_intent", no_intent)
    monkeypatch.setattr(receive, "send_stt_message", AsyncMock())
    before = turn_gen.current(h)
    await receive.startToChat(h, "新问题", gen=before)
    args, kwargs = h.executor.submit.call_args
    assert args == (h.chat, "新问题") and kwargs["turn"] == before + 1 == turn_gen.current(h)


# ---------- 思考中打断 ----------

def test_chat_queued_before_an_abort_never_asks_the_model():
    llm = MagicMock()
    h = make_conn(llm)
    gen = turn_gen.bump(h)
    turn_gen.bump(h)                  # 线程还没接手就被打断了
    assert h.chat("旧的问题", turn=gen) is None
    llm.response.assert_not_called()
    assert contents(h) == []


class ThinkingLLM:
    """第一个问题要想很久（release 之前不出字）；记下每次收到的对话。"""

    def __init__(self):
        self.asked = []
        self.thinking = threading.Event()
        self.release = threading.Event()

    def response(self, session_id, dialogue, **kw):
        self.asked.append([(m["role"], m.get("content")) for m in dialogue])
        if dialogue[-1]["content"] == "第一个问题":
            self.thinking.set()
            self.release.wait(5)
            yield "旧的回答，不该播。"
        else:
            yield "新的回答。"


async def test_abort_while_thinking_leaves_a_clean_context():
    llm = ThinkingLLM()
    h = make_conn(llm)
    gen = turn_gen.bump(h)
    t = threading.Thread(target=h.chat, args=("第一个问题",), kwargs={"turn": gen})
    t.start()
    await asyncio.to_thread(llm.thinking.wait, 5)
    h.client_is_speaking = True                      # stt 发出后网关就当在"说话"
    await handleAbortMessage(h, played_ms=0, drop_audio=True)
    assert contents(h) == []                         # 没回答的提问当场去掉

    # 用户接着问了新问题（startToChat 把 client_abort 清回 False），这时旧的模型才想完
    h.client_abort = False
    new = turn_gen.bump(h)
    await asyncio.to_thread(h.chat, "第二个问题", 0, new)
    llm.release.set()
    await asyncio.to_thread(t.join, 5)

    assert [m for m in llm.asked[1] if m[0] != "system"] == [("user", "第二个问题")]
    assert contents(h) == [("user", "第二个问题"), ("assistant", "新的回答。")]
    assert "旧的回答，不该播。" not in tts_texts(h)


async def test_tool_call_cut_off_by_an_abort_is_not_run():
    yielded, go_on = threading.Event(), threading.Event()

    def call(name, args):
        return SimpleNamespace(index=0, id="c1", function=SimpleNamespace(name=name, arguments=args))

    class ToolLLM:
        def response_with_functions(self, session_id, dialogue, functions=None, **kw):
            yield None, [call("set_volume", '{"vol')]
            yielded.set()
            go_on.wait(5)
            yield None, [call(None, 'ume": 100}')]

    h = make_conn(ToolLLM())
    h.intent_type = "function_call"
    h.func_handler = MagicMock()
    h.func_handler.get_functions.return_value = []
    gen = turn_gen.bump(h)
    t = threading.Thread(target=h.chat, args=("把音量调到最大",), kwargs={"turn": gen})
    t.start()
    await asyncio.to_thread(yielded.wait, 5)
    h.client_is_speaking = True
    await handleAbortMessage(h, played_ms=0, drop_audio=True)
    go_on.set()
    await asyncio.to_thread(t.join, 5)
    h.func_handler.handle_llm_function_call.assert_not_called()
    assert contents(h) == []


# ---------- 回答中打断：迟到的旧语音不进新一轮 ----------

async def test_late_audio_of_the_aborted_turn_is_not_sent():
    h = make_conn()
    h.sentence_id = old = "old-turn"
    await handleAbortMessage(h, drop_audio=True)
    h.client_abort = False                          # 新的一轮开始了，chat() 还没换上自己的 sentence_id
    h.websocket.send.reset_mock()
    await sendAudioMessage(h, SentenceType.MIDDLE, [b"opus"], None, old)
    await sendAudioMessage(h, SentenceType.FIRST, None, "旧的一句", old)
    h.websocket.send.assert_not_awaited()


# ---------- 普通 API 模型：打断时立即断开流式请求 ----------

class _StallingSSE(BaseHTTPRequestHandler):
    """出一个字后就不再出字（模型在想），连接一直挂着。"""

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunk = {"id": "1", "object": "chat.completion.chunk", "created": 0, "model": "m",
                 "choices": [{"index": 0, "delta": {"content": "好"}, "finish_reason": None}]}
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.flush()
        time.sleep(10)

    def log_message(self, *a):
        pass


def test_openai_abort_cuts_a_stalled_stream_at_once(monkeypatch):
    from core.providers.llm.openai.openai import LLMProvider

    server = ThreadingHTTPServer(("127.0.0.1", 0), _StallingSSE)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        llm = LLMProvider({"model_name": "m", "api_key": "sk-test", "base_url": base_url})
        # conftest 把 httpx.Client 换成了不联网的桩；这里只连本机的假服务，用真的
        import httpx
        import httpx._client
        import openai
        monkeypatch.setattr(httpx, "Client", httpx._client.Client)
        llm.client = openai.OpenAI(api_key="sk-test", base_url=base_url, http_client=httpx.Client())
        got, ended = [], threading.Event()

        def read():
            try:
                for piece in llm.response("s", [{"role": "user", "content": "hi"}]):
                    got.append(piece)
            except Exception as e:
                got.append(type(e).__name__)
            ended.set()

        threading.Thread(target=read, daemon=True).start()
        wait_for(lambda: got)
        assert not ended.wait(0.3)                  # 卡在等下一块
        t0 = time.monotonic()
        llm.abort(SimpleNamespace(session_id="s"))
        assert ended.wait(2)
        assert time.monotonic() - t0 < 1
        assert got[0] == "好" and not llm._streams
        # 别的连接的流不受影响
        llm.abort(SimpleNamespace(session_id="other"))
    finally:
        server.shutdown()


# ---------- 聊天记录落盘：被打断丢掉的这一轮不存库 ----------

CHAT_CONFIG = {
    "LLM": {"DeepSeekLLM": {"type": "openai", "model_name": "deepseek-chat", "base_url": "https://x", "api_key": "sk-1"}},
    "chat_history": {"max_messages": 30, "max_tokens": 8000, "summary": False},
}


def stored(device="dev-1"):
    from core.agent import chat_history
    return [(r["role"], r["content"]) for r in chat_history.db().recent(device, "DeepSeekLLM", 100)]


async def test_question_interrupted_while_thinking_is_never_stored():
    from core.agent import chat_history

    llm = ThinkingLLM()
    h = make_conn(llm)
    h.config, h.device_id = CHAT_CONFIG, "dev-1"
    chat_history.load(h, h.dialogue, "DeepSeekLLM")
    gen = turn_gen.bump(h)
    t = threading.Thread(target=h.chat, args=("第一个问题",), kwargs={"turn": gen})
    t.start()
    await asyncio.to_thread(llm.thinking.wait, 5)
    h.client_is_speaking = True
    await handleAbortMessage(h, played_ms=0, drop_audio=True)
    llm.release.set()
    await asyncio.to_thread(t.join, 5)          # 旧的 chat() 结束时也会同步一次库
    h.client_abort = False
    await asyncio.to_thread(h.chat, "第二个问题", 0, turn_gen.bump(h))
    assert stored() == [("user", "第二个问题"), ("assistant", "新的回答。")]


def test_turn_already_stored_then_dropped_is_deleted_from_the_db():
    """回复整段生成完、存了库，可一个字都还没播出就被打断：库里也删掉。"""
    from core.agent import chat_history
    from core.utils import reply_progress
    from core.utils.dialogue import Message

    conn = SimpleNamespace(device_id="dev-1", session_id="s", config=CHAT_CONFIG, dialogue=Dialogue(),
                           logger=MagicMock(), hub_route=None, sentence_id="t1")
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    user = Message(role="user", content="讲个故事")
    conn.dialogue.put(user)
    reply_progress.note_turn(conn, "t1", user)
    conn.dialogue.put(Message(role="assistant", content="从前有座山。"))
    reply_progress.finish_turn(conn, "t1")
    chat_history.sync(conn)
    assert len(stored()) == 2
    reply_progress.cut(conn, played_ms=0)
    assert stored() == []
