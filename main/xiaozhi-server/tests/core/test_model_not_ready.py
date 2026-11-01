"""[xiaozhi-hub] 模型没配置好（刚刷机、Key 还是占位文字）时马上告诉用户，不拿空 Key 去请求、等出错才说"有点忙"。"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.agent import llm_router
from core.connection import ConnectionHandler
from core.utils.dialogue import Dialogue

LLM = {
    "DeepSeekLLM": {"type": "openai", "url": "https://api.deepseek.com", "model_name": "deepseek-chat",
                    "api_key": "你的deepseek api key"},
    "KimiLLM": {"type": "openai", "base_url": "https://api.moonshot.cn/v1", "model_name": "kimi", "api_key": "sk-1"},
}


def make_handler(selected):
    h = ConnectionHandler.__new__(ConnectionHandler)
    h.logger = MagicMock()
    h.dialogue = Dialogue()
    h.tts = MagicMock()
    h.llm = MagicMock()
    h.llm.response.return_value = iter(["好的"])
    h.memory = None
    h.intent_type = "nointent"
    h.config = {"selected_module": {"LLM": selected}, "LLM": LLM}
    h.current_speaker = None
    h.system_introduced_speakers = set()
    h.features = {"emoji": False}
    h.client_abort = False
    h.sentence_id = None
    h.session_id = "s"
    return h


def spoken(h):
    return "".join(c.args[0].content_detail for c in h.tts.tts_text_queue.put.call_args_list
                   if getattr(c.args[0], "content_detail", None))


def test_placeholder_key_answers_at_once():
    h = make_handler("DeepSeekLLM")
    h._chat("今天天气怎么样")
    assert spoken(h) == llm_router.NOT_READY_REPLY
    h.llm.response.assert_not_called()
    assert not [m for m in h.dialogue.dialogue if m.role == "user"]   # 不写进对话


def test_configured_model_is_asked():
    h = make_handler("KimiLLM")
    h._chat("今天天气怎么样")
    assert spoken(h) == "好的"
    h.llm.response.assert_called_once()


def test_switched_model_counts_not_the_default():
    h = make_handler("DeepSeekLLM")
    h.hub_route = {"current": "KimiLLM", "used": [], "dialogues": {}}
    assert llm_router.not_ready_reply(h) is None
    h.hub_route["current"] = None
    assert llm_router.not_ready_reply(h) == llm_router.NOT_READY_REPLY


def test_unknown_model_is_not_blocked():
    assert llm_router.not_ready_reply(SimpleNamespace(config={})) is None


def test_key_filled_in_while_connected_gets_a_fresh_instance(monkeypatch):
    # heard the prompt, filled the key in the admin page, asked again before the connection closed:
    # the connection's instance was built with the placeholder key
    h = make_handler("DeepSeekLLM")
    stale = h.llm
    h._chat("你好")
    stale.response.assert_not_called()
    fresh = MagicMock()
    fresh.response.return_value = iter(["好的"])
    h.config["LLM"] = {**LLM, "DeepSeekLLM": {**LLM["DeepSeekLLM"], "api_key": "sk-real"}}
    monkeypatch.setattr(llm_router, "get_instance", lambda conn, name: fresh)
    h.tts.reset_mock()
    h._chat("你好")
    assert h.llm is fresh and spoken(h) == "好的"
    stale.response.assert_not_called()


def test_default_changed_to_a_configured_model_is_used(monkeypatch):
    # chose and set up another model as the default instead of filling in the first one's key
    h = make_handler("DeepSeekLLM")
    h.hub_route = {"current": "DeepSeekLLM", "used": [], "dialogues": {}}
    h._chat("你好")                                      # not configured yet
    h.config["selected_module"]["LLM"] = "KimiLLM"      # admin page: Kimi set up and made the default
    calls = []
    monkeypatch.setattr(llm_router, "switch", lambda conn, name, remember=True, live=True: calls.append(name) or True)
    assert llm_router.not_ready_reply(h) is None
    assert calls == ["KimiLLM"]


def test_failed_instance_swap_still_answers_and_retries(monkeypatch):
    h = make_handler("DeepSeekLLM")
    h._chat("你好")
    h.config["LLM"] = {**LLM, "DeepSeekLLM": {**LLM["DeepSeekLLM"], "api_key": "sk-real"}}

    def boom(conn, name):
        raise RuntimeError("create failed")
    monkeypatch.setattr(llm_router, "get_instance", boom)
    h.tts.reset_mock()
    h._chat("你好")
    assert spoken(h) == "好的"                       # asked with the old instance, as before
    assert h.hub_route["not_ready"] is True          # tried again next turn
