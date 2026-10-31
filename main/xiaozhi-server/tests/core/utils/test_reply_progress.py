"""被打断的回复只把播出去的部分记进上下文；旧的一轮被打断后不再往新一轮里塞内容。"""
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.connection import ConnectionHandler
from core.utils import reply_progress
from core.utils.dialogue import Dialogue, Message

PKT = b"x"  # 一个 60ms 的 Opus 包


def make_conn(sid="t1"):
    conn = SimpleNamespace(sentence_id=sid, dialogue=Dialogue(), logger=MagicMock())
    return conn


def start_turn(conn, sid, question="讲个故事"):
    conn.sentence_id = sid
    user = Message(role="user", content=question)
    conn.dialogue.put(user)
    reply_progress.note_turn(conn, sid, user)
    return user


def speak(conn, sid, sentences, last=True):
    """sentences: [(文字, 包数)]，按顺序发给设备；last：这轮的音频到此为止。"""
    for text, packets in sentences:
        reply_progress.note_audio(conn, sid, text, None)          # FIRST：只有文字
        reply_progress.note_audio(conn, sid, None, [PKT] * packets)  # MIDDLE：音频
    if last:
        reply_progress.note_audio(conn, sid, None, [], last=True)  # LAST


def contents(conn):
    return [(m.role, m.content) for m in conn.dialogue.dialogue]


def test_heard_text_whole_sentences_then_proportional_cut():
    conn = make_conn()
    speak(conn, "t1", [("从前有座山", 10), ("山里有座庙", 10)])  # 0–600ms、600–1200ms
    assert reply_progress.heard_text(conn, 0) == ("", False)
    assert reply_progress.heard_text(conn, 600) == ("从前有座山", False)
    assert reply_progress.heard_text(conn, 840) == ("从前有座山，山里", False)  # 第二句播了 40%
    assert reply_progress.heard_text(conn, 1200) == ("从前有座山，山里有座庙", True)


def test_all_sent_audio_played_is_not_the_whole_reply():
    # 打断时后面的句子还没合成出来：已发出的都播完了，也不算播完
    conn = make_conn()
    start_turn(conn, "t1")
    speak(conn, "t1", [("好呀", 5)], last=False)
    assert reply_progress.heard_text(conn, 2000) == ("好呀", False)
    conn.dialogue.put(Message(role="assistant", content="好呀，从前有只小猫……故事讲完啦。"))
    reply_progress.finish_turn(conn, "t1")
    reply_progress.cut(conn, played_ms=2000)
    assert contents(conn)[-1] == ("assistant", "好呀……")


def test_new_turn_forgets_previous_audio():
    conn = make_conn("t1")
    speak(conn, "t1", [("旧的", 10)])
    conn.sentence_id = "t2"
    speak(conn, "t2", [("新的", 10)])
    assert reply_progress.heard_text(conn, 9999) == ("新的", True)


def test_cut_after_reply_was_stored_replaces_it():
    conn = make_conn()
    start_turn(conn, "t1")
    speak(conn, "t1", [("从前有座山", 10), ("山里有座庙", 10)])
    conn.dialogue.put(Message(role="assistant", content="从前有座山，山里有座庙。"))
    reply_progress.finish_turn(conn, "t1")   # 生成比播放快：chat() 早就结束了
    reply_progress.cut(conn, played_ms=700)
    assert contents(conn) == [("user", "讲个故事"), ("assistant", "从前有座山……")]


def test_cut_before_reply_was_stored_blocks_it_and_writes_heard_part():
    conn = make_conn()
    start_turn(conn, "t1")
    speak(conn, "t1", [("从前有座山", 10)])
    reply_progress.cut(conn, played_ms=360)
    assert not reply_progress.can_store(conn, "t1")
    reply_progress.finish_turn(conn, "t1")
    assert contents(conn) == [("user", "讲个故事"), ("assistant", "从前有……")]


def test_cut_while_thinking_keeps_question_answer_order():
    conn = make_conn()
    start_turn(conn, "t1")
    reply_progress.cut(conn, played_ms=0)
    reply_progress.finish_turn(conn, "t1")
    start_turn(conn, "t2", "算了，换个问题")
    assert contents(conn) == [("user", "讲个故事"), ("assistant", "……"), ("user", "算了，换个问题")]


def test_cut_lands_before_the_next_question():
    conn = make_conn()
    start_turn(conn, "t1")
    speak(conn, "t1", [("从前有座山", 10)])
    reply_progress.cut(conn, played_ms=600 * 2 // 3)
    start_turn(conn, "t2", "下一个问题")       # 旧的一轮还没结束时新问题就到了
    reply_progress.finish_turn(conn, "t1")
    assert contents(conn)[:3] == [("user", "讲个故事"), ("assistant", "从前有……"), ("user", "下一个问题")]


def test_tool_turn_keeps_tool_calls_and_one_reply():
    conn = make_conn()
    start_turn(conn, "t1", "调大音量再讲个笑话")
    speak(conn, "t1", [("好的", 5), ("音量调好了", 10), ("有一天", 10)])
    d = conn.dialogue
    d.put(Message(role="assistant", content="好的"))
    d.put(Message(role="assistant", tool_calls=[{"id": "c1"}]))
    d.put(Message(role="tool", tool_call_id="c1", content="ok"))
    d.put(Message(role="assistant", content="音量调好了。有一天……"))
    reply_progress.finish_turn(conn, "t1")
    reply_progress.cut(conn, played_ms=900)   # 第二句（300–900ms）刚播完
    assert [(m.role, m.content, m.tool_calls) for m in d.dialogue] == [
        ("user", "调大音量再讲个笑话", None),
        ("assistant", None, [{"id": "c1"}]),
        ("tool", "ok", None),
        ("assistant", "好的，音量调好了……", None),
    ]


def test_fully_played_reply_is_untouched():
    conn = make_conn()
    start_turn(conn, "t1")
    speak(conn, "t1", [("讲完了", 10)])
    conn.dialogue.put(Message(role="assistant", content="讲完了。"))
    reply_progress.finish_turn(conn, "t1")
    reply_progress.cut(conn, played_ms=600)
    assert contents(conn)[-1] == ("assistant", "讲完了。")
    assert reply_progress.can_store(conn, "t1")


def test_estimate_without_device_report():
    conn = make_conn()
    start_turn(conn, "t1")
    speak(conn, "t1", [("从前有座山", 10), ("山里有座庙", 10)])
    # 发出了 15 包（900ms），估算设备里还缓冲着 300ms 没播：已播 600ms
    conn.audio_flow_control = {"sentence_id": "t1", "packet_count": 15, "play_end": time.monotonic() + 0.3}
    conn.dialogue.put(Message(role="assistant", content="从前有座山，山里有座庙。"))
    reply_progress.finish_turn(conn, "t1")
    reply_progress.cut(conn)
    assert contents(conn)[-1] == ("assistant", "从前有座山……")


# ---------- ConnectionHandler.chat()：每轮一个取消令牌 ----------

def make_handler(llm):
    h = ConnectionHandler.__new__(ConnectionHandler)
    h.logger = MagicMock()
    h.dialogue = Dialogue()
    h.tts = MagicMock()
    h.llm = llm
    h.memory = None
    h.intent_type = "nointent"
    h.config = {}
    h.current_speaker = None
    h.system_introduced_speakers = set()
    h.features = {"emoji": False}
    h.client_abort = False
    h.sentence_id = None
    h.session_id = "s"
    return h


def test_superseded_turn_stops_and_does_not_store_its_reply():
    first_chunk = threading.Event()
    go_on = threading.Event()

    class SlowLLM:
        def response(self, session_id, dialogue, **kw):
            if dialogue[-1]["content"] == "第一个问题":
                yield "第一段，"
                first_chunk.set()
                go_on.wait(5)
                yield "这段不该再要。"
            else:
                yield "新的回答。"

    h = make_handler(SlowLLM())
    t = threading.Thread(target=h.chat, args=("第一个问题",))
    t.start()
    assert first_chunk.wait(5)
    # 设备打断了，一个字都还没播出去；然后用户问了新问题（startToChat 会清掉 client_abort）
    h.client_is_speaking = True
    reply_progress.cut(h, played_ms=0)
    h.client_abort = False
    h.chat("第二个问题")
    go_on.set()
    t.join(5)
    assert [(m.role, m.content) for m in h.dialogue.dialogue] == [
        ("user", "第一个问题"), ("assistant", "……"), ("user", "第二个问题"), ("assistant", "新的回答。")]
    # 旧的一轮在新问题开始后没再把字交给合成
    texts = [c.args[0].content_detail for c in h.tts.tts_text_queue.put.call_args_list
             if getattr(c.args[0], "content_detail", None)]
    assert "这段不该再要。" not in texts
