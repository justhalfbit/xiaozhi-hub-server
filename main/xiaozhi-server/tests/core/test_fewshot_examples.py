"""工具调用示例（connection._inject_tool_call_fewshot）不能被当成真实对话。

原来示例当成几轮对话发给模型（用户"给我讲个故事吧"、"拜拜"），用户问"我刚才让你做过什么"时
DeepSeek 答"你让我讲故事、说了拜拜"。现在示例写进系统提示末尾、注明不是真实对话；也不存库、不进记忆总结。
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.agent import chat_history
from core.connection import ConnectionHandler
from core.utils.dialogue import FEWSHOT_INTRO, Dialogue, Message

CONFIG = {
    "LLM": {"DeepSeekLLM": {"type": "openai", "model_name": "deepseek-chat", "base_url": "https://x", "api_key": "sk-1"}},
    "chat_history": {"max_messages": 4, "max_tokens": 8000, "summary_min": 2},
}
EXAMPLE_WORDS = ("一公斤等于几斤", "拜拜", "再见，有事再叫我", "已直接回复", "退出意图已处理", "带伞")


def make_conn():
    d = Dialogue()
    d.put(Message(role="system", content="你是语音助手"))
    conn = SimpleNamespace(device_id="dev-1", session_id="s-1", config=CONFIG, dialogue=d, logger=MagicMock(),
                           hub_route=None, intent_type="function_call",
                           func_handler=SimpleNamespace(get_functions=lambda: [
                               {"type": "function", "function": {"name": "handle_exit_intent"}}]))
    ConnectionHandler._inject_tool_call_fewshot(conn)
    return conn


def chat(conn, user, reply):
    conn.dialogue.put(Message(role="user", content=user))
    conn.dialogue.put(Message(role="assistant", content=reply))
    chat_history.sync(conn)


def has_example(text) -> bool:
    return any(w in (text or "") for w in EXAMPLE_WORDS)


def test_examples_are_injected():
    conn = make_conn()
    assert [m.content for m in conn.dialogue.dialogue if m.is_temporary and m.role == "user"] == ["一公斤等于几斤", "拜拜"]


def test_asking_what_was_said_sends_no_example_turns():
    conn = make_conn()
    conn.dialogue.put(Message(role="user", content="今天几号"))
    conn.dialogue.put(Message(role="assistant", content="11月1日"))
    conn.dialogue.put(Message(role="user", content="我刚才说过什么"))
    msgs = conn.dialogue.get_llm_dialogue_with_memory(None)
    # 对话轮次里只有真实的话
    assert [(m["role"], m.get("content")) for m in msgs[1:]] == [
        ("user", "今天几号"), ("assistant", "11月1日"), ("user", "我刚才说过什么")]
    # 示例在系统提示里，并注明不是真实对话
    system = msgs[0]["content"]
    assert system.startswith("你是语音助手")
    block = system[system.index("<tool_call_examples>"):]
    assert FEWSHOT_INTRO in block and "不是和用户的真实对话" in block
    assert "用户：一公斤等于几斤" in block and "调用工具 direct_answer" in block
    assert "用户：拜拜" in block and "调用工具 handle_exit_intent" in block


def test_examples_come_before_memory_in_system_prompt():
    conn = make_conn()
    system = conn.dialogue.get_llm_dialogue_with_memory("- 用户叫小明")[0]["content"]
    assert system.index("</tool_call_examples>") < system.index("<memory>")
    assert system.endswith("- 用户叫小明\n</memory>")


def test_no_example_block_without_examples():
    d = Dialogue()
    d.put(Message(role="system", content="你是语音助手"))
    d.put(Message(role="user", content="你好"))
    assert "<tool_call_examples>" not in d.get_llm_dialogue_with_memory(None)[0]["content"]


def test_examples_never_go_into_chat_db():
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    chat(conn, "今天几号", "11月1日")
    rows = chat_history.db().recent("dev-1", "DeepSeekLLM", 100)
    assert [(r["role"], r["content"]) for r in rows] == [("user", "今天几号"), ("assistant", "11月1日")]
    shown = chat_history.history("dev-1", 100)["DeepSeekLLM"]
    assert not any(has_example(m.get("text")) for m in shown)
    # 下次连接从库里载入：示例还是只有新注入的那一份，载入的记录里没有
    conn2 = make_conn()
    chat_history.load(conn2, conn2.dialogue, "DeepSeekLLM")
    assert [m.content for m in conn2.dialogue.dialogue if m.role == "user" and not m.is_temporary] == ["今天几号"]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def response_no_stream(self, system_prompt, user_prompt, **kwargs):
        self.calls.append(user_prompt)
        return "- 用户叫小明"


def test_examples_never_go_into_memory_summary():
    conn = make_conn()
    chat_history.load(conn, conn.dialogue, "DeepSeekLLM")
    chat(conn, "我叫小明", "你好小明")
    for i in range(3):
        chat(conn, f"问{i}", f"答{i}")
    llm = FakeLLM()
    assert chat_history.summarize_now(llm, "dev-1", chat_history.settings(CONFIG), CONFIG)
    assert "用户：我叫小明" in llm.calls[0] and not has_example(llm.calls[0])


# ---------- 示例的内容（dialogue.fewshot_messages） ----------

def _users(names):
    from core.utils.dialogue import fewshot_messages
    return [m.content for m in fewshot_messages(names) if m.role == "user"]


def test_examples_follow_the_tools_there_are():
    assert _users({"handle_exit_intent"}) == ["一公斤等于几斤", "拜拜"]
    assert _users({"handle_exit_intent", "get_weather"}) == ["一公斤等于几斤", "明天要带伞吗", "拜拜"]


def test_no_weather_example_when_the_template_already_has_local_weather():
    """agent-base-prompt.txt 带本地 7 天天气（weather_info），那时问本地天气应该直接回答，不给调用工具的示例。"""
    from core.utils.dialogue import fewshot_messages
    users = [m.content for m in fewshot_messages({"handle_exit_intent", "get_weather"}, weather_in_context=True)
             if m.role == "user"]
    assert users == ["一公斤等于几斤", "拜拜"]


def test_examples_act_instead_of_asking_back():
    """直接回答、查本地天气不传地点：示例里的回答不以问句收尾。"""
    import json
    from core.utils.dialogue import fewshot_messages
    msgs = fewshot_messages({"handle_exit_intent", "get_weather"})
    calls = {tc["function"]["name"]: json.loads(tc["function"]["arguments"])
             for m in msgs if m.tool_calls for tc in m.tool_calls}
    assert calls["direct_answer"]["response"] == "一公斤等于两斤。"
    assert calls["get_weather"] == {"lang": "zh_CN"}
    spoken = [calls["direct_answer"]["response"]] + [m.content for m in msgs if m.role == "assistant" and m.content]
    assert spoken and not any(t.rstrip().endswith(("？", "?")) for t in spoken)
    # 每个工具调用都有对应的结果
    ids = [tc["id"] for m in msgs if m.tool_calls for tc in m.tool_calls]
    assert ids == [m.tool_call_id for m in msgs if m.role == "tool"] and len(set(ids)) == len(ids)


def test_admin_page_shows_the_same_examples():
    from core.admin.pipeline import _fewshot_text
    from core.utils.dialogue import fewshot_messages, render_fewshot
    names = {"handle_exit_intent", "get_lunar", "get_weather"}
    assert _fewshot_text(["get_weather"]) == render_fewshot(fewshot_messages(names))
    assert _fewshot_text(["get_weather"], True) == render_fewshot(fewshot_messages(names, True))


def test_template_rules():
    """默认模板：工具能答就直接调、不先问；只在真需要时反问；讲故事直接开讲；声纹的说话人说明还在。"""
    from config.config_loader import get_project_dir
    import os
    with open(os.path.join(get_project_dir(), "hub-base-prompt.txt"), encoding="utf-8") as f:
        text = f.read()
    assert "don't ask “要不要我查一下” first" in text
    assert "Ask back only when needed" in text and "just start telling a short one" in text
    assert "<speaker_recognition>" in text
    assert "Muse" not in text and "ask_muse" not in text


def test_connection_skips_the_weather_example_when_the_template_has_local_weather():
    def injected(template):
        d = Dialogue()
        conn = SimpleNamespace(device_id="dev-1", session_id="s-1", config=CONFIG, dialogue=d, logger=MagicMock(),
                               hub_route=None, intent_type="function_call",
                               prompt_manager=SimpleNamespace(base_prompt_template=template),
                               func_handler=SimpleNamespace(get_functions=lambda: [
                                   {"type": "function", "function": {"name": n}} for n in ("handle_exit_intent", "get_weather")]))
        ConnectionHandler._inject_tool_call_fewshot(conn)
        return [m.content for m in d.dialogue if m.is_temporary and m.role == "user"]

    assert injected("{{base_prompt}}") == ["一公斤等于几斤", "明天要带伞吗", "拜拜"]
    assert injected("- Local upcoming weather: {{weather_info}}") == ["一公斤等于几斤", "拜拜"]
