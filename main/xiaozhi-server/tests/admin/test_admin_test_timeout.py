"""测试连接连不上时很快放弃、不重试；后台的同步操作用自己的线程，不和语音识别抢默认线程池。

（conftest 把 httpx 的网络层换成了假的，这里直接看测试连接怎样构造模型客户端。）
"""
import asyncio
import threading
from types import SimpleNamespace

import httpx
import openai

import core.admin.admin_handler as ah
import core.providers.llm.openai.openai as openai_llm


def test_unreachable_test_gives_up_fast_without_retries(monkeypatch):
    made = []

    class FakeProvider:
        def __init__(self, config):
            self.config = config
            self.client = SimpleNamespace(max_retries=2)   # openai 默认失败后再试 2 次
            made.append(self)

        def response_no_stream(self, *_):
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://zenmux.ai/api/v1"))
    monkeypatch.setattr(openai_llm, "LLMProvider", FakeProvider)
    result = ah.AdminHandler._test_llm({"model_name": "m", "api_key": "sk-x", "timeout": 20,
                                        "base_url": "https://zenmux.ai/api/v1"})
    assert not result["ok"] and "连不上接口地址" in result["error"] and "代理" in result["error"]
    p = made[0]
    assert p.config["timeout"] == ah.TEST_TIMEOUT and ah.TEST_TIMEOUT["connect"] <= 5   # 页面传来的 20 秒被替换
    assert p.client.max_retries == 0


def test_blocking_runs_on_the_admin_threads():
    name = asyncio.run(ah.AdminHandler._blocking(lambda: threading.current_thread().name))
    assert name.startswith("admin")
