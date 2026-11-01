"""测试连接连不上时很快放弃、不重试；推理型模型想得久也等得到，读超时和连不上分开提示；
后台的同步操作用自己的线程，不和语音识别抢默认线程池。

（conftest 把 httpx 的网络层换成了假的，这里直接看测试连接怎样构造模型客户端。）
"""
import asyncio
import threading
from types import SimpleNamespace

import httpx
import openai

import core.admin.admin_handler as ah
import core.providers.llm.openai.openai as openai_llm


REQ = httpx.Request("POST", "https://zenmux.ai/api/v1")
CONFIG = {"model_name": "m", "api_key": "sk-x", "timeout": 20, "base_url": "https://zenmux.ai/api/v1"}


def fake_provider(monkeypatch, response):
    """把模型客户端换成假的：response(session_id, dialogue) 是生成器函数，按测试场景出字或抛错。"""
    made = []

    class FakeProvider:
        def __init__(self, config):
            self.config = config
            self.client = SimpleNamespace(max_retries=2)   # openai 默认失败后再试 2 次
            made.append(self)

        def response(self, session_id, dialogue):
            return response()
    monkeypatch.setattr(openai_llm, "LLMProvider", FakeProvider)
    return made


def test_unreachable_test_gives_up_fast_without_retries(monkeypatch):
    def response():
        raise openai.APIConnectionError(request=REQ)
        yield
    made = fake_provider(monkeypatch, response)
    result = ah.AdminHandler._test_llm(CONFIG)
    assert not result["ok"] and "连不上接口地址" in result["error"] and "代理" in result["error"]
    p = made[0]
    assert p.config["timeout"] == ah.TEST_TIMEOUT and ah.TEST_TIMEOUT["connect"] <= 10   # 页面传来的 20 秒被替换
    assert p.client.max_retries == 0


def test_connect_timeout_is_unreachable(monkeypatch):
    def response():
        raise openai.APITimeoutError(request=REQ) from httpx.ConnectTimeout("timed out")
        yield
    fake_provider(monkeypatch, response)
    assert "连不上接口地址" in ah.AdminHandler._test_llm(CONFIG)["error"]


def test_read_timeout_says_model_is_slow_not_unreachable(monkeypatch):
    """推理型模型先想很久：连上了、迟迟不出字，不能说成连不上。"""
    assert ah.TEST_TIMEOUT["read"] >= 60

    def before_headers():   # 等响应头时超时：openai 包成 APITimeoutError
        raise openai.APITimeoutError(request=REQ) from httpx.ReadTimeout("timed out")
        yield

    def mid_stream():       # 流读到一半超时：httpx 的原始异常
        yield ""
        raise httpx.ReadTimeout("timed out")
    for response in (before_headers, mid_stream):
        fake_provider(monkeypatch, response)
        error = ah.AdminHandler._test_llm(CONFIG)["error"]
        assert "接口连上了" in error and "60 秒" in error and "连不上" not in error


def test_stops_at_first_word_and_closes_stream(monkeypatch):
    closed = []

    def response():
        try:
            yield ""            # 推理型模型先只发思考，没有正文
            yield "好"
            raise AssertionError("拿到第一个字后不该再读")
        finally:
            closed.append(True)
    fake_provider(monkeypatch, response)
    result = ah.AdminHandler._test_llm(CONFIG)
    assert result["ok"] and result["reply"] == "好" and closed == [True]


def test_blocking_runs_on_the_admin_threads():
    name = asyncio.run(ah.AdminHandler._blocking(lambda: threading.current_thread().name))
    assert name.startswith("admin")
