"""联网搜索（Anthropic 协议 + 自带搜索）：地址补全、结果解析、出错原因、插件出错时的回复、后台设置和测试按钮。"""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from core.admin import admin_handler as ah
from core.admin import config_store, connections, pages, plugin_settings
from plugins_func.functions import web_search as ws

OK_BODY = {
    "content": [
        {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "今天新闻"}},
        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
            {"type": "web_search_result", "title": "第一条新闻", "url": "https://a.example/1", "page_age": "2026-11-03"},
            {"type": "web_search_result", "title": "重复的", "url": "https://a.example/1"},
            {"type": "web_search_result", "title": "第二条", "url": "https://b.example/2"},
        ]},
        {"type": "text", "text": "今天的要点是"},
        {"type": "text", "text": "某某事件。"},
    ]
}


@pytest.fixture
def fake_api(monkeypatch):
    """把 web_search 发出的请求接到假的接口上，记下请求，按 respond 回应。"""
    calls = []
    state = {"respond": lambda req: httpx.Response(200, json=OK_BODY)}

    def handler(request):
        calls.append(request)
        return state["respond"](request)

    from httpx._client import AsyncClient as real  # tests/conftest.py 把 httpx.AsyncClient 换成了连不上的桩
    monkeypatch.setattr(ws.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return SimpleNamespace(calls=calls, state=state)


SETTINGS = {"base_url": "https://api.deepseek.com/anthropic/v1", "api_key": "sk-good", "model": "deepseek-v4-flash",
            "max_uses": 3}


def test_messages_url_accepts_common_forms():
    assert ws.messages_url("https://api.deepseek.com/anthropic") == "https://api.deepseek.com/anthropic/v1/messages"
    assert ws.messages_url("https://api.deepseek.com/anthropic/v1/") == "https://api.deepseek.com/anthropic/v1/messages"
    assert ws.messages_url("https://api.anthropic.com") == "https://api.anthropic.com/v1/messages"
    assert ws.messages_url("http://192.168.1.5:8317/v1") == "http://192.168.1.5:8317/v1/messages"
    assert ws.messages_url("http://x/v1/messages") == "http://x/v1/messages"


def test_settings_defaults_and_clamp():
    s = ws.settings_of({"plugins": {"web_search": {"api_key": " k ", "max_uses": 99, "base_url": "", "model": None}}})
    assert s == {"base_url": ws.DEFAULT_BASE_URL, "api_key": "k", "model": ws.DEFAULT_MODEL, "max_uses": 10}
    assert ws.settings_of({})["max_uses"] == ws.DEFAULT_MAX_USES


async def test_search_request_and_parse(fake_api):
    r = await ws.search(SETTINGS, "今天新闻")
    req = fake_api.calls[0]
    assert str(req.url) == "https://api.deepseek.com/anthropic/v1/messages"
    assert req.headers["x-api-key"] == "sk-good" and req.headers["anthropic-version"] == ws.ANTHROPIC_VERSION
    body = json.loads(req.content)
    assert body["model"] == "deepseek-v4-flash" and body["thinking"] == {"type": "disabled"}
    assert body["tools"] == [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
    assert body["messages"] == [{"role": "user", "content": "今天新闻"}]
    assert r.summary == "今天的要点是某某事件。" and r.queries == ["今天新闻"]
    assert [x["title"] for x in r.results] == ["第一条新闻", "第二条"]  # 同一个链接只留一条
    text = ws.format_for_llm("今天新闻", r)
    assert "总结：今天的要点是某某事件。" in text and "1. 第一条新闻（2026-11-03）" in text and "https://" not in text


@pytest.mark.parametrize("respond, reason", [
    (lambda r: httpx.Response(401, json={"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}),
     "API Key 不对"),
    (lambda r: httpx.Response(404, text="Not Found"), "不是 Anthropic 协议"),
    (lambda r: httpx.Response(400, json={"error": {"message": "unknown tool type web_search_20250305"}}), "不支持自带搜索"),
    (lambda r: httpx.Response(200, text="<html>hi</html>"), "不是 Anthropic 协议"),
    (lambda r: httpx.Response(200, json={"choices": []}), "不是 Anthropic 协议"),  # OpenAI 协议的回应
    (lambda r: httpx.Response(200, json={"content": [{"type": "text", "text": "我不能上网"}]}), "没有执行搜索"),
    (lambda r: httpx.Response(200, json={"content": [
        {"type": "server_tool_use", "id": "s", "input": {"query": "q"}},
        {"type": "web_search_tool_result", "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"}}]}),
     "unavailable"),
])
async def test_search_error_reasons(fake_api, respond, reason):
    fake_api.state["respond"] = respond
    with pytest.raises(ws.SearchError, match=reason):
        await ws.search(SETTINGS, "q")


async def test_search_unreachable_and_timeout(fake_api):
    def refuse(request):
        raise httpx.ConnectError("refused")
    fake_api.state["respond"] = refuse
    with pytest.raises(ws.SearchError, match="连不上"):
        await ws.search(SETTINGS, "q")

    def slow(request):
        raise httpx.ReadTimeout("slow")
    fake_api.state["respond"] = slow
    with pytest.raises(ws.SearchError, match="没有结果"):
        await ws.search(SETTINGS, "q")
    with pytest.raises(ws.SearchError, match="API Key"):
        await ws.search({**SETTINGS, "api_key": ""}, "q")


async def test_plugin_tells_model_to_say_not_found(fake_api):
    fake_api.state["respond"] = lambda r: httpx.Response(401, json={"error": {"message": "x"}})
    conn = SimpleNamespace(config={"plugins": {"web_search": SETTINGS}})
    resp = await ws.web_search(conn, query="q")
    assert "没搜到" in resp.result and "不要编造" in resp.result
    fake_api.state["respond"] = lambda r: httpx.Response(200, json=OK_BODY)
    resp = await ws.web_search(conn, query="q")
    assert resp.result.startswith("【联网搜索结果】")


async def test_key_redacted_in_error(fake_api):
    fake_api.state["respond"] = lambda r: httpx.Response(401, json={"error": {"message": "invalid key sk-good given"}})
    with pytest.raises(ws.SearchError) as e:
        await ws.search(SETTINGS, "q")
    assert "sk-good" not in str(e.value) and "***" in str(e.value)


# ---------- 后台：联网搜索的设置和测试按钮 ----------

CUSTOM_YAML = """\
selected_module:
  LLM: ClaudeCLI
  Intent: nointent
plugins:
  web_search:
    api_key: mk-xxx
"""


@pytest.fixture
def cfg_path(tmp_path) -> Path:
    p = tmp_path / ".config.yaml"
    p.write_text(CUSTOM_YAML, encoding="utf-8")
    return p


def test_web_search_needs_config_until_key(cfg_path):
    m = config_store.load_merged(custom_path=cfg_path)
    assert next(p for p in pages.describe_intent(m)["plugins"] if p["name"] == "web_search")["needs_config"] is True
    d = plugin_settings.describe(m, "web_search")
    assert d["testable"] is True and next(f for f in d["fields"] if f["key"] == "base_url")["value"] == ws.DEFAULT_BASE_URL
    plugin_settings.save("web_search", {"api_key": "sk-good"}, m, cfg_path)
    m = config_store.load_merged(custom_path=cfg_path)
    assert next(p for p in pages.describe_intent(m)["plugins"] if p["name"] == "web_search")["needs_config"] is False


@pytest.fixture
async def client(cfg_path):
    app = web.Application()
    config = {"selected_module": {"LLM": "ClaudeCLI", "Intent": "nointent"}, "LLM": {}, "Intent": {}}
    handler = ah.AdminHandler(config, ws_server=None, config_path=cfg_path)
    handler.register(app)
    async with TestClient(TestServer(app)) as c:
        await c.post("/admin/api/setup", json={"password": "pass-1234"})
        yield c


async def test_admin_test_button(client, fake_api):
    r = await client.post("/admin/api/plugin-settings/web_search/test", json={})
    assert r.status == 400 and "API Key" in (await r.json())["error"]
    r = await (await client.post("/admin/api/plugin-settings/web_search/test", json={"api_key": "sk-good"})).json()
    assert r == {"ok": True, "seconds": r["seconds"], "count": 2, "first": "第一条新闻"}
    await client.post("/admin/api/plugin-settings/web_search/test",
                      json={"api_key": "sk-good", "base_url": "http://10.0.0.2:8317/v1"})
    assert str(fake_api.calls[-1].url) == "http://10.0.0.2:8317/v1/messages"
    fake_api.state["respond"] = lambda req: httpx.Response(401, json={"error": {"message": "bad sk-good"}})
    r = await client.post("/admin/api/plugin-settings/web_search/test", json={"api_key": "sk-good"})
    err = (await r.json())["error"]
    assert r.status == 400 and "API Key 不对" in err and "sk-good" not in err
