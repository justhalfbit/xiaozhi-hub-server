"""聊天记录落盘（core/agent/chat_history）和语音清除（core/agent/clear_command）。"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.agent import chat_history, clear_command
from core.agent.session_store import SessionStore
from core.utils import reply_progress
from core.utils.dialogue import Dialogue, Message

CONFIG = {
    "LLM": {
        "DeepSeekLLM": {"type": "openai", "model_name": "deepseek-chat", "base_url": "https://x", "api_key": "sk-1"},
        "ClaudeAPI": {"type": "openai", "model_name": "claude-haiku-5-5", "base_url": "https://y", "api_key": "sk-2"},
        "ClaudeCLI": {"type": "claude_cli"},
        "GrokBot_小助手": {"type": "grok_bot", "bot_id": "bot-1", "bot_name": "小助手"},
    },
    "chat_history": {"max_messages": 6, "max_tokens": 8000, "summary_min": 2},
}


def make_conn(device="dev-1", config=None):
    d = Dialogue()
    d.put(Message(role="system", content="你是语音助手"))
    d.put(Message(role="user", content="给我讲个故事吧", is_temporary=True))  # 工具调用示例
    d.put(Message(role="assistant", content="好的", is_temporary=True))
    return SimpleNamespace(device_id=device, session_id=f"s-{device}", config=config or CONFIG, dialogue=d,
                           logger=MagicMock(), hub_route=None)


def chat(conn, user, reply):
    conn.dialogue.put(Message(role="user", content=user))
    conn.dialogue.put(Message(role="assistant", content=reply))
    chat_history.sync(conn)


def sent(conn, memory=None):
    """发给模型的对话（不含系统提示和示例）。"""
    msgs = conn.dialogue.get_llm_dialogue_with_memory(memory)
    return [(m["role"], m.get("content")) for m in msgs[1:]]


def tool_turn(d, n):
    d.put(Message(role="user", content=f"查天气{n}"))
    d.put(Message(role="assistant", tool_calls=[{"id": f"c{n}", "type": "function",
                                                 "function": {"name": "get_weather", "arguments": "{}"}}]))
    d.put(Message(role="tool", tool_call_id=f"c{n}", content="晴"))
    d.put(Message(role="assistant", content=f"今天晴{n}"))


# ---------- 截断窗口 ----------

def test_window_keeps_recent_messages_by_count():
    msgs = [Message(role="user" if i % 2 == 0 else "assistant", content=str(i)) for i in range(20)]
    start = chat_history.select_window(msgs, 6, 0)
    assert start == 14 and msgs[start].role == "user"


def test_window_never_splits_tool_call_from_its_result():
    d = Dialogue()
    for n in range(3):
        tool_turn(d, n)
    # 每组 4 条：6 条只装得下最后一组，不会从工具结果中间开始
    start = chat_history.select_window(d.dialogue, 6, 0)
    assert start == 8 and d.dialogue[start].content == "查天气2"
    start = chat_history.select_window(d.dialogue, 9, 0)
    assert start == 4


def test_window_token_limit_and_last_group_always_kept():
    msgs = [Message(role="user", content="长" * 3000), Message(role="assistant", content="好"),
            Message(role="user", content="问" * 5000), Message(role="assistant", content="嗯")]
    assert chat_history.select_window(msgs, 30, 6000) == 2
    # 最新的一组自己就超了也照样带上
    assert chat_history.select_window(msgs, 30, 100) == 2
    assert chat_history.select_window(msgs, 0, 0) == 0


def test_window_starts_at_a_user_message():
    msgs = [Message(role="assistant", content="好一会儿没听到你说话"), Message(role="user", content="在吗"),
            Message(role="assistant", content="在")]
    # 开头没有用户的话的助手消息（例如告别语）不带：有的模型接口要求对话从用户开始
    assert chat_history.select_window(msgs, 2, 0) == 1
    assert chat_history.select_window(msgs, 30, 0) == 1


def test_estimate_tokens():
    assert chat_history.estimate_tokens("你好") == 2
    assert chat_history.estimate_tokens("hello world!") == 3
    assert chat_history.estimate_tokens(None) == 0


# ---------- 存、载入 ----------

def test_turns_survive_a_new_connection_and_only_recent_ones_are_sent():
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    for i in range(5):
        chat(conn, f"问{i}", f"答{i}")
    # 新连接（网关重启、断开后重连都一样）
    conn2 = make_conn()
    chat_history.load(conn2, conn2.dialogue, "DeepSeekLLM")
    assert sent(conn2) == [("user", "问2"), ("assistant", "答2"), ("user", "问3"), ("assistant", "答3"),
                           ("user", "问4"), ("assistant", "答4")]
    # 示例和系统提示没有存进库
    rows = chat_history.db().recent("dev-1", "DeepSeekLLM", 100)
    assert [r["content"] for r in rows][:2] == ["问0", "答0"] and len(rows) == 10


def test_long_connection_is_windowed_too():
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    for i in range(6):
        chat(conn, f"问{i}", f"答{i}")
    assert sent(conn)[0] == ("user", "问3") and len(sent(conn)) == 6


def test_models_and_devices_are_separate():
    a, b, c = make_conn(), make_conn(), make_conn("dev-2")
    chat_history.load(a, a.dialogue, "DeepSeekLLM")
    chat(a, "我叫小明", "你好小明")
    chat_history.load(b, b.dialogue, "ClaudeCLI")
    chat_history.load(c, c.dialogue, "DeepSeekLLM")
    assert sent(b) == [] and sent(c) == []


def test_agent_models_are_recorded_but_not_loaded():
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "ClaudeCLI")
    chat(conn, "看看日志", "好的")
    assert conn.dialogue.window is None
    conn2 = make_conn()
    chat_history.load(conn2, conn2.dialogue, "ClaudeCLI")
    assert sent(conn2) == []
    assert [m["text"] for m in chat_history.history("dev-1", 10)["ClaudeCLI"]] == ["看看日志", "好的"]


def test_no_device_id_means_nothing_is_stored():
    conn = make_conn(device=None)
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    chat(conn, "问", "答")
    assert chat_history.db().models("dev-1") == [] and len(sent(conn)) == 2


def test_sync_updates_and_deletes_and_skips_hints():
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    chat(conn, "讲个故事", "从前有座山，山里有座庙。")
    conn.dialogue.put(Message(role="user", content="[系统提示] 已达到最大工具调用次数限制"))
    extra = Message(role="assistant", content="多余的一句")
    conn.dialogue.put(extra)
    chat_history.sync(conn)
    conn.dialogue.dialogue[4].content = "从前有座山……"
    conn.dialogue.dialogue.remove(extra)
    chat_history.sync(conn)
    rows = chat_history.db().recent("dev-1", "DeepSeekLLM", 10)
    assert [(r["role"], r["content"]) for r in rows] == [("user", "讲个故事"), ("assistant", "从前有座山……")]


def test_tool_messages_round_trip():
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    tool_turn(conn.dialogue, 1)
    chat_history.sync(conn)
    conn2 = make_conn()
    chat_history.load(conn2, conn2.dialogue, "DeepSeekLLM")
    msgs = conn2.dialogue.get_llm_dialogue_with_memory(None)[1:]
    assert msgs[1]["tool_calls"][0]["id"] == "c1" and msgs[2] == {"role": "tool", "tool_call_id": "c1", "content": "晴"}


def test_cut_after_the_turn_was_saved_is_synced():
    """回复播到一半被打断：chat() 早已结束并存了整段，reply_progress 改写后库里也跟着改。"""
    conn = make_conn()
    conn.sentence_id = "t1"
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    user = Message(role="user", content="讲个故事")
    conn.dialogue.put(user)
    reply_progress.note_turn(conn, "t1", user)
    reply_progress.note_audio(conn, "t1", "从前有座山", [b"x"] * 10)
    reply_progress.note_audio(conn, "t1", "山里有座庙", [b"x"] * 10)
    reply_progress.note_audio(conn, "t1", None, [], last=True)
    conn.dialogue.put(Message(role="assistant", content="从前有座山，山里有座庙。"))
    reply_progress.finish_turn(conn, "t1")
    chat_history.sync(conn)
    reply_progress.cut(conn, played_ms=700)
    rows = chat_history.db().recent("dev-1", "DeepSeekLLM", 10)
    assert rows[-1]["content"] == "从前有座山……"
    shown = chat_history.history("dev-1", 10)["DeepSeekLLM"][-1]
    assert shown["text"] == "从前有座山" and shown["interrupted"]


# ---------- 记忆 ----------

class FakeLLM:
    def __init__(self, reply="- 用户叫小明，喜欢爬山"):
        self.reply = reply
        self.calls = []

    def response_no_stream(self, system_prompt, user_prompt, **kwargs):
        self.calls.append((system_prompt, user_prompt))
        return self.reply


def test_dropped_messages_are_summarized_into_memory():
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    chat(conn, "我叫小明", "你好小明")
    for i in range(3):
        chat(conn, f"问{i}", f"答{i}")
    cfg = chat_history.settings(CONFIG)
    rows, mem = chat_history.pending_summary("dev-1", cfg, CONFIG)
    assert [r["content"] for r in rows] == ["我叫小明", "你好小明"] and mem["summary"] == ""
    assert {r["model"] for r in rows} == {"DeepSeekLLM"}
    llm = FakeLLM()
    assert chat_history.summarize_now(llm, "dev-1", cfg, CONFIG)
    prompt = llm.calls[0][1]
    assert "用户：我叫小明" in prompt and "给我讲个故事吧" not in prompt  # 工具调用示例不会混进记忆
    assert chat_history.memory_for(conn) == "- 用户叫小明，喜欢爬山"
    # 已总结过的不再总结
    assert chat_history.pending_summary("dev-1", cfg, CONFIG)[0] == []
    # 增量：下一批带上已有记忆
    chat(conn, "问3", "答3")
    llm2 = FakeLLM("- 用户叫小明\n- 问过问0")
    chat_history.summarize_now(llm2, "dev-1", cfg, CONFIG)
    assert "【已有记忆】\n- 用户叫小明，喜欢爬山" in llm2.calls[0][1] and "用户：问0" in llm2.calls[0][1]
    assert "用户：我叫小明" not in llm2.calls[0][1]


def test_memory_goes_into_the_system_prompt():
    conn = make_conn()
    msgs = conn.dialogue.get_llm_dialogue_with_memory("- 用户叫小明")
    assert msgs[0]["content"].endswith("<memory>\n" + chat_history_intro() + "\n- 用户叫小明\n</memory>")
    assert "<memory>" not in conn.dialogue.get_llm_dialogue_with_memory("")[0]["content"]
    # 提示词里自带 <memory> 时照上游的做法替换
    conn.dialogue.update_system_message("前面<memory></memory>后面")
    assert conn.dialogue.get_llm_dialogue_with_memory("记忆")[0]["content"].startswith("前面<memory>\n记忆\n</memory>后面")


def chat_history_intro():
    from core.utils.dialogue import MEMORY_INTRO
    return MEMORY_INTRO


def test_maybe_summarize_runs_in_background(monkeypatch):
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    llm = FakeLLM()
    monkeypatch.setattr(chat_history, "_summary_llm", lambda c, name, cfg: llm)
    chat(conn, "问0", "答0")
    chat_history.maybe_summarize(conn)  # 没有截掉的：不总结
    for i in range(1, 4):
        chat(conn, f"问{i}", f"答{i}")
    chat_history.maybe_summarize(conn)
    for _ in range(100):
        if chat_history.memory_for(conn):
            break
        import time
        time.sleep(0.02)
    assert chat_history.memory_for(conn) == llm.reply and len(llm.calls) == 1


def test_no_summary_for_agents(monkeypatch):
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "ClaudeCLI")
    called = []
    monkeypatch.setattr(chat_history, "_summary_llm", lambda *a: called.append(a))
    for i in range(10):
        chat(conn, f"问{i}", f"答{i}")
    chat_history.maybe_summarize(conn)
    assert called == []


# ---------- 显示 ----------

def test_history_for_display():
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    conn.dialogue.put(Message(role="user", content=json.dumps({"speaker": "小明", "content": "今天天气"}, ensure_ascii=False)))
    tool_turn(conn.dialogue, 1)
    conn.dialogue.put(Message(role="assistant", content="😊晴天，适合出门"))
    chat_history.sync(conn)
    items = chat_history.history("dev-1", 10)["DeepSeekLLM"]
    assert [(m["role"], m["text"]) for m in items] == [
        ("user", "今天天气"), ("user", "查天气1"), ("bot", "今天晴1"), ("bot", "晴天，适合出门")]
    assert chat_history.history("dev-1", 2)["DeepSeekLLM"][0]["text"] == "今天晴1"
    assert chat_history.history("nobody", 10) == {}


# ---------- 清除 ----------

@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_history, "get_project_dir", lambda: str(tmp_path))
    return tmp_path


def test_clear_removes_records_memory_and_live_dialogues(project):
    conn = make_conn()
    other = make_conn("dev-2")
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    chat_history.load(other, other.dialogue, "DeepSeekLLM")
    for i in range(4):
        chat(conn, f"问{i}", f"答{i}")
    chat(other, "别人的", "记录")
    chat_history.db().set_memory("dev-1", "记忆", "summary")
    claude = SessionStore(project / "data" / "agent_sessions.json", "claude_cli")
    claude.touch("dev-1", "sess-1")
    agent = MagicMock()
    conn.hub_route = {"used": [agent], "dialogues": {}}

    result = chat_history.clear("dev-1", CONFIG, connections=[conn])
    assert result == {"deleted": 8, "agents": []}
    assert sent(conn) == [] and chat_history.memory_for(conn) == ""
    assert [m.role for m in conn.dialogue.dialogue] == ["system", "user", "assistant"]  # 系统提示和示例留着
    # 只清本机：Claude Code 的会话（映射、进程）不碰
    assert agent.method_calls == [] and claude.get("dev-1")["session_id"] == "sess-1"
    assert chat_history.history("dev-2", 10)
    # 清除后接着聊照常存
    chat(conn, "新问题", "新回答")
    conn2 = make_conn()
    chat_history.load(conn2, conn2.dialogue, "DeepSeekLLM")
    assert sent(conn2) == [("user", "新问题"), ("assistant", "新回答")]


def test_clear_memory_only_keeps_records_and_does_not_resummarize(project):
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    for i in range(5):
        chat(conn, f"问{i}", f"答{i}")
    cfg = chat_history.settings(CONFIG)
    chat_history.summarize_now(FakeLLM(), "dev-1", cfg, CONFIG)
    chat_history.clear("dev-1", CONFIG, memory_only=True)
    assert chat_history.memory_for(conn) == ""
    assert chat_history.pending_summary("dev-1", cfg, CONFIG)[0] == []
    assert len(chat_history.history("dev-1", 100)["DeepSeekLLM"]) == 10


def test_clear_leaves_agents_alone_and_names_them(project, monkeypatch):
    """Claude Code、Grok Bot 有自己的上下文和记忆：清除只删本机存的记录，不调它们任何东西。"""
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "GrokBot_小助手")
    chat(conn, "帮我查下", "好的")
    cli = make_conn()
    chat_history.load(cli, cli.dialogue, "ClaudeCLI")
    chat(cli, "看看日志", "好的")
    result = chat_history.clear("dev-1", CONFIG, connections=[conn, cli])
    assert result == {"deleted": 4, "agents": ["Claude", "小助手"]}
    assert chat_history.history("dev-1", 10) == {}
    text = clear_command.result_text(result)
    assert text.startswith(clear_command.DONE) and "Claude和小助手自己那边的对话没有动" in text
    assert clear_command.result_text({"deleted": 2, "agents": []}) == clear_command.DONE


# ---------- 语音指令 ----------

@pytest.mark.parametrize("text", ["清除聊天记录", "清空聊天记录。", "把聊天记录清空", "删除所有聊天记录", "帮我清除一下对话记录",
                                  "聊天记录全部删掉", "清除历史记录吧"])
def test_clear_phrases(text):
    assert clear_command.is_clear(text)


@pytest.mark.parametrize("text", ["别清除聊天记录", "不清除聊天记录", "今天天气怎么样", "清除缓存",
                                  "我想知道怎么清除微信的聊天记录才不会被恢复找回来呢"])
def test_not_clear_phrases(text):
    assert not clear_command.is_clear(text)


@pytest.mark.parametrize("text,confirm,cancel", [
    ("确定", True, False), ("确定清除", True, False), ("嗯，清除吧", True, False), ("是的", True, False),
    ("好", True, False), ("不要", False, True), ("算了", False, True), ("不清除", False, True),
    ("今天天气怎么样", False, False),
])
def test_confirm_and_cancel(text, confirm, cancel):
    assert clear_command.is_confirm(text) is confirm
    assert clear_command.is_cancel(text) is cancel


def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


@pytest.fixture
def spoken(monkeypatch):
    said = []

    async def fake_speak(conn, heard, reply):
        said.append(reply)
        return True

    async def fake_clear(conn):
        said.append("<clear>")
        return {"deleted": 3, "agents": []}

    async def fake_notify(conn):
        said.append("<notify>")

    monkeypatch.setattr(clear_command, "_speak", fake_speak)
    monkeypatch.setattr(clear_command, "_notify_cleared", fake_notify)
    monkeypatch.setattr(clear_command, "clear_now", fake_clear)
    return said


def test_voice_clear_asks_first_then_clears(spoken):
    conn = make_conn()
    assert run(clear_command.handle(conn, "清除聊天记录")) is None
    assert spoken == [clear_command.ASK]
    assert run(clear_command.handle(conn, "确定")) is None
    assert spoken[1:] == ["<clear>", "<notify>", clear_command.DONE]
    # 确认只算一次
    assert run(clear_command.handle(conn, "确定")) == "确定"


def test_voice_clear_cancel_or_other_question(spoken):
    conn = make_conn()
    run(clear_command.handle(conn, "清空聊天记录"))
    assert run(clear_command.handle(conn, "算了")) is None and spoken[-1] == clear_command.KEPT
    run(clear_command.handle(conn, "清空聊天记录"))
    # 说了别的：当没问过，照常交给模型
    assert run(clear_command.handle(conn, "今天天气怎么样")) == "今天天气怎么样"
    assert "<clear>" not in spoken


def test_voice_clear_confirmation_expires(spoken, monkeypatch):
    conn = make_conn()
    run(clear_command.handle(conn, "清除聊天记录"))
    conn.chat_clear_asked -= clear_command.CONFIRM_SECONDS + 1
    assert run(clear_command.handle(conn, "确定")) == "确定"
    assert "<clear>" not in spoken


def test_voice_clear_now_uses_online_connections(project, monkeypatch):
    from core.admin import connections

    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    chat(conn, "问", "答")
    conn.server = SimpleNamespace(config=CONFIG)
    connections.register(conn)
    try:
        result = run(clear_command.clear_now(conn))
    finally:
        connections.unregister(conn)
    assert result["deleted"] == 2 and sent(conn) == []


# ---------- 接进 ConnectionHandler.chat() ----------

def make_handler(llm, device="dev-1"):
    from core.connection import ConnectionHandler

    h = ConnectionHandler.__new__(ConnectionHandler)
    h.logger = MagicMock()
    h.dialogue = Dialogue()
    h.dialogue.put(Message(role="system", content="你是语音助手"))
    h.tts = MagicMock()
    h.llm = llm
    h.memory = None
    h.intent_type = "nointent"
    h.config = CONFIG
    h.current_speaker = None
    h.system_introduced_speakers = set()
    h.features = {"emoji": False}
    h.client_abort = False
    h.sentence_id = None
    h.session_id = f"s-{device}"
    h.device_id = device
    chat_history.load(h, h.dialogue, "DeepSeekLLM")
    return h


class EchoLLM:
    def __init__(self):
        self.seen = []

    def response(self, session_id, dialogue, **kw):
        self.seen.append(dialogue)
        yield f"收到：{dialogue[-1]['content']}"


def test_handler_saves_each_turn_and_next_connection_remembers(monkeypatch):
    monkeypatch.setattr(chat_history, "maybe_summarize", lambda conn: None)
    h = make_handler(EchoLLM())
    h.chat("我叫小明")
    h.chat("我喜欢爬山")
    # 连接断了、网关重启了：新的连接
    llm = EchoLLM()
    h2 = make_handler(llm)
    h2.chat("我刚才说了什么")
    sent_now = [(m["role"], m["content"]) for m in llm.seen[0][1:]]
    assert sent_now == [("user", "我叫小明"), ("assistant", "收到：我叫小明"), ("user", "我喜欢爬山"),
                        ("assistant", "收到：我喜欢爬山"), ("user", "我刚才说了什么")]
    assert len(chat_history.db().recent("dev-1", "DeepSeekLLM", 100)) == 6


def test_handler_puts_summary_memory_in_system_prompt():
    chat_history.db().set_memory("dev-1", "- 用户叫小明", "manual")
    llm = EchoLLM()
    h = make_handler(llm)
    h.chat("你好")
    assert "- 用户叫小明" in llm.seen[0][0]["content"]


def test_notify_cleared_message():
    sent_msgs = []

    class WS:
        async def send(self, data):
            sent_msgs.append(json.loads(data))

    conn = make_conn()
    conn.websocket = WS()
    run(clear_command._notify_cleared(conn))
    assert sent_msgs == [{"type": "chat_history", "state": "cleared", "session_id": "s-dev-1"}]


def test_summary_finished_after_a_clear_is_dropped(project):
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    for i in range(5):
        chat(conn, f"问{i}", f"答{i}")
    cfg = chat_history.settings(CONFIG)

    class ClearingLLM(FakeLLM):
        def response_no_stream(self, system_prompt, user_prompt, **kwargs):
            chat_history.clear("dev-1", CONFIG)   # 总结期间用户清除了，又接着聊
            for i in range(5):
                chat(conn, f"新{i}", f"新答{i}")
            return "- 旧记忆"

    assert not chat_history.summarize_now(ClearingLLM(), "dev-1", cfg, CONFIG)
    assert chat_history.memory_for(conn) == ""


# ---------- 记忆按设备共用 ----------

def test_memory_is_shared_by_api_models_of_a_device_but_not_agents():
    chat_history.db().set_memory("dev-1", "- 用户叫小明", "summary")
    for name, expect in (("DeepSeekLLM", "- 用户叫小明"), ("ClaudeAPI", "- 用户叫小明"), ("ClaudeCLI", ""),
                         ("GrokBot_小助手", "")):
        conn = make_conn()
        chat_history.load(conn, conn.dialogue, name)
        assert chat_history.memory_for(conn) == expect, name
    other = make_conn("dev-2")
    chat_history.load(other, other.dialogue, "DeepSeekLLM")
    assert chat_history.memory_for(other) == ""


def test_one_summary_covers_every_api_model_with_progress_per_model():
    a, b, cli = make_conn(), make_conn(), make_conn()
    chat_history.load(a, a.dialogue, "DeepSeekLLM")
    chat_history.load(b, b.dialogue, "ClaudeAPI")
    chat_history.load(cli, cli.dialogue, "ClaudeCLI")
    for i in range(4):
        chat(a, f"深{i}", f"深答{i}")
        chat(b, f"克{i}", f"克答{i}")
        chat(cli, f"命{i}", f"命答{i}")
    cfg = chat_history.settings(CONFIG)
    rows, _ = chat_history.pending_summary("dev-1", cfg, CONFIG)
    assert [r["content"] for r in rows] == ["克0", "克答0", "深0", "深答0"]   # Claude Code 的不总结
    llm = FakeLLM()
    assert chat_history.summarize_now(llm, "dev-1", cfg, CONFIG)
    assert "（和 ClaudeAPI 的对话）" in llm.calls[0][1] and "（和 DeepSeekLLM 的对话）" in llm.calls[0][1]
    progress = chat_history.db().progress("dev-1")
    assert set(progress) == {"ClaudeAPI", "DeepSeekLLM"}
    assert chat_history.pending_summary("dev-1", cfg, CONFIG)[0] == []
    # 只有一个模型又截掉了新的：只总结它的
    chat(a, "深4", "深答4")
    assert [r["content"] for r in chat_history.pending_summary("dev-1", cfg, CONFIG)[0]] == ["深1", "深答1"]


def test_summary_is_dropped_if_memory_was_edited_meanwhile():
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    for i in range(5):
        chat(conn, f"问{i}", f"答{i}")
    cfg = chat_history.settings(CONFIG)

    class EditedMeanwhile(FakeLLM):
        def response_no_stream(self, system_prompt, user_prompt, **kwargs):
            ok, _ = chat_history.set_memory("dev-1", "- 后台改的")
            assert ok
            return "- 自动总结的"

    assert not chat_history.summarize_now(EditedMeanwhile(), "dev-1", cfg, CONFIG)
    mem = chat_history.db().memory("dev-1")
    assert mem["summary"] == "- 后台改的" and mem["source"] == "manual"
    # 进度没动：下次在手改的基础上再总结这批
    llm = FakeLLM("- 合并后的")
    assert chat_history.summarize_now(llm, "dev-1", cfg, CONFIG)
    assert "【已有记忆】\n- 后台改的" in llm.calls[0][1]


def test_concurrent_summaries_of_one_device_run_one_at_a_time(monkeypatch):
    import threading
    import time as _time

    started, release = threading.Event(), threading.Event()
    calls = []

    class SlowLLM(FakeLLM):
        def response_no_stream(self, system_prompt, user_prompt, **kwargs):
            calls.append(1)
            started.set()
            release.wait(2)
            return "- 记忆"

    monkeypatch.setattr(chat_history, "_summary_llm", lambda c, name, cfg: SlowLLM())
    a, b = make_conn(), make_conn()
    chat_history.load(a, a.dialogue, "DeepSeekLLM")
    chat_history.load(b, b.dialogue, "ClaudeAPI")
    for i in range(4):
        chat(a, f"深{i}", f"深答{i}")
        chat(b, f"克{i}", f"克答{i}")
    chat_history.maybe_summarize(a)
    assert started.wait(2)
    chat_history.maybe_summarize(b)   # 同一台设备已经在总结：不再开第二个
    release.set()
    deadline = _time.monotonic() + 2
    while chat_history._summarizing and _time.monotonic() < deadline:
        _time.sleep(0.01)
    assert len(calls) == 1 and chat_history.memory_for(a) == "- 记忆"


def test_set_memory_from_admin_takes_effect_and_checks_version():
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    ok, mem = chat_history.set_memory("dev-1", "  - 用户叫小红  ")
    assert ok and mem["summary"] == "- 用户叫小红" and mem["source"] == "manual" and mem["version"] == 1
    assert "- 用户叫小红" in conn.dialogue.get_llm_dialogue_with_memory(chat_history.memory_for(conn))[0]["content"]
    ok, mem = chat_history.set_memory("dev-1", "旧版本上改的", expect_version=0)
    assert not ok and mem["summary"] == "- 用户叫小红"
    ok, mem = chat_history.set_memory("dev-1", "", expect_version=1)
    assert ok and mem["summary"] == "" and mem["source"] == "cleared"


def test_old_per_model_memories_are_merged(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    c = sqlite3.connect(str(path))
    c.executescript("""
        CREATE TABLE memories (device TEXT, model TEXT, summary TEXT, upto INTEGER, updated REAL, PRIMARY KEY (device, model));
        INSERT INTO memories VALUES ('dev-1', 'DeepSeekLLM', '- 叫小明', 12, 100), ('dev-1', 'ClaudeAPI', '- 爱爬山', 30, 200),
                                    ('dev-2', 'DeepSeekLLM', '', 5, 50);
    """)
    c.commit()
    c.close()
    store = chat_history.ChatDB(path)
    try:
        assert store.memory("dev-1")["summary"] == "- 叫小明\n- 爱爬山"
        assert store.progress("dev-1") == {"DeepSeekLLM": 12, "ClaudeAPI": 30}
        assert store.memory("dev-2")["summary"] == "" and store.progress("dev-2") == {"DeepSeekLLM": 5}
        assert not store._exec("SELECT 1 FROM sqlite_master WHERE name='memories'").fetchone()
    finally:
        store.close()
    chat_history.ChatDB(path).close()   # 再打开一次不出错


def test_clear_bumps_memory_version_so_running_summary_is_dropped(project):
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    chat_history.db().set_memory("dev-1", "- 记忆", "summary")
    before = chat_history.db().memory("dev-1")["version"]
    chat_history.clear("dev-1", CONFIG)
    mem = chat_history.db().memory("dev-1")
    assert mem["summary"] == "" and mem["version"] == before + 1 and chat_history.db().progress("dev-1") == {}
