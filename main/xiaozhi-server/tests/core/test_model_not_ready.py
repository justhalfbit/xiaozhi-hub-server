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
