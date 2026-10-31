"""[xiaozhi-hub] 模型漏进回复文字的工具调用标记（DeepSeek DSML 等）不念、不上屏、不写进对话。"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.connection import ConnectionHandler
from core.utils.dialogue import Dialogue
from core.utils.tool_markup import ToolMarkupFilter, strip_tool_markup

# 2026-11-01 00:47 设备上 DeepSeek 讲完故事后实际念出来的那段
LEAK = "</｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>"


def stream(text, size=1):
    """按 size 个字一段喂给过滤器，模拟流式输出。"""
    f = ToolMarkupFilter()
    out = "".join(f.feed(text[i:i + size]) for i in range(0, len(text), size))
    return out + f.flush()


def test_leak_from_2026_11_01_is_removed():
    text = "要不要我再讲一个别的呀？\n" + LEAK
    assert strip_tool_markup(text) == "要不要我再讲一个别的呀？\n"
    for size in (1, 2, 3, 5, 7, 100):
        assert stream(text, size) == "要不要我再讲一个别的呀？\n"
    assert strip_tool_markup(LEAK) == ""


def test_whole_dsml_call_block_and_its_arguments_are_removed():
    text = (
        "我帮你查一下。\n"
        "<｜DSML｜function_calls>\n"
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜function_calls>\n"
        "稍等哦。"
    )
    for size in (1, 4, 1000):
        assert stream(text, size) == "我帮你查一下。\n稍等哦。"


def test_unclosed_call_drops_the_rest_and_truncated_marker_is_dropped():
    assert strip_tool_markup('好的。<｜DSML｜invoke name="x">参数') == "好的。"
    assert strip_tool_markup("好的。</｜DSML｜ param") == "好的。"


def test_old_deepseek_and_other_special_tokens():
    text = ("好的<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '{"city": "杭州"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>，马上。<｜end▁of▁sentence｜>')
    assert stream(text) == "好的，马上。"
    assert strip_tool_markup("你好<|im_end|>") == "你好"


def test_normal_text_is_untouched():
    texts = [
        "我讲一个《小橘猫的月亮灯》。",
        "3<5，a < b，x<y，<b>粗体</b>，<>",
        "竖线｜分隔｜的文字，a | b，管道 ls | grep x",
        "DSML 是一种标记语言。",
        "温度 <25 度时穿外套",
        "最后一个字是<",
        "一行\n两行\n\n三行  ",
    ]
    for text in texts:
        for size in (1, 2, 3, 1000):
            assert stream(text, size) == text, (text, size)


# ---------- ConnectionHandler.chat()：两条路径都要过滤 ----------

def make_handler(llm, intent_type):
    h = ConnectionHandler.__new__(ConnectionHandler)
    h.logger = MagicMock()
    h.dialogue = Dialogue()
    h.tts = MagicMock()
    h.llm = llm
    h.memory = None
    h.intent_type = intent_type
    h.func_handler = SimpleNamespace(get_functions=lambda: [])
    h.config = {}
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


def stored(h):
    return [m.content for m in h.dialogue.dialogue if m.role == "assistant"]


def chunks(text, size=3):
    return [text[i:i + size] for i in range(0, len(text), size)]


def test_direct_answer_path():
    """DeepSeek 讲故事走 direct_answer 的 response 参数；变形标记接口认不出，留在参数里。"""
    reply = "从前有只小猫。\n要不要我再讲一个别的呀？\n" + LEAK
    args = json.dumps({"response": reply}, ensure_ascii=False)

    def call(i, name=None, arguments=None):
        return SimpleNamespace(index=0, id="c1" if i == 0 else None,
                               function=SimpleNamespace(name=name, arguments=arguments))

    class FakeLLM:
        def response_with_functions(self, session_id, dialogue, functions=None, **kw):
            assert any(f["function"]["name"] == "direct_answer" for f in functions)
            yield None, [call(0, name="direct_answer")]
            for i, part in enumerate(chunks(args), 1):
                yield None, [call(i, arguments=part)]

    h = make_handler(FakeLLM(), "function_call")
    h.chat("讲个小猫的故事")
    assert "DSML" not in spoken(h) and "从前有只小猫" in spoken(h)
    assert stored(h) == ["从前有只小猫。\n要不要我再讲一个别的呀？"]


def test_content_path():
    """不走 direct_answer（例如工具结果之后的那轮）时，标记从正文 content 进来。"""
    reply = "今天杭州晴，二十度。\n" + LEAK

    class FakeLLM:
        def response(self, session_id, dialogue, **kw):
            yield from chunks(reply)

    h = make_handler(FakeLLM(), "nointent")
    h.chat("杭州天气")
    assert "DSML" not in spoken(h) and "今天杭州晴" in spoken(h)
    assert stored(h) == ["今天杭州晴，二十度。\n"]
