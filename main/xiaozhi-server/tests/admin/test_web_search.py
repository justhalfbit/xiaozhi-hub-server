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
        {"type": "text", "text": "某某事件。", "citations": [
            {"type": "web_search_result_location", "url": "https://b.example/2", "title": "第二条",
             "cited_text": "某某事件\n今天上午发生。"},
            {"type": "web_search_result_location", "url": "https://b.example/2", "cited_text": "影响很大。"},
            {"type": "web_search_result_location", "url": "https://b.example/2", "cited_text": "影响很大。"}]},
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
    # 同一个链接只留一条；有原文片段（被引用过）的排前面，片段去掉换行、重复的不拼
    assert [x["title"] for x in r.results] == ["第二条", "第一条新闻"]
    assert r.results[0]["snippet"] == "某某事件 今天上午发生。 影响很大。" and r.results[1]["snippet"] == ""
    text = ws.format_for_llm("今天新闻", r)
    assert "总结：今天的要点是某某事件。" in text and "https://" not in text
    assert "1. 第二条：某某事件 今天上午发生。 影响很大。" in text and "2. 第一条新闻（2026-11-03）" in text
    assert "以原文片段为准" in text and "一到两句" in text and "60 个字" in text


def test_snippet_and_source_limits():
    long = "字" * 400
    data = {"content": [
        {"type": "server_tool_use", "id": "s", "input": {"query": "q"}},
        {"type": "web_search_tool_result", "content": [
            {"type": "web_search_result", "title": f"t{i}", "url": f"https://x/{i}"} for i in range(9)]},
        {"type": "text", "text": "x", "citations": [{"url": "https://x/8", "cited_text": long}]},
    ]}
    r = ws._parse(data)
    assert r.results[0]["title"] == "t8" and len(r.results[0]["snippet"]) == ws.SNIPPET_LIMIT + 1  # 加省略号
    text = ws.format_for_llm("q", r)
    assert "5. t3" in text and "6." not in text
    r.results[0]["snippet"] = ""  # 没有引用的接口（如 DeepSeek）：不提原文片段
    text = ws.format_for_llm("q", r)
    assert "来源：" in text and "原文片段" not in text


C, T = ws.CONFIG, ws.TEMPORARY


@pytest.mark.parametrize("respond, reason, kind", [
    (lambda r: httpx.Response(401, json={"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}),
     "API Key 不对", C),
    (lambda r: httpx.Response(402, json={"error": {"message": "Insufficient Balance"}}), "余额不足", C),
    (lambda r: httpx.Response(400, json={"error": {"message": "Your credit balance is too low"}}), "余额不足", C),
    (lambda r: httpx.Response(429, json={"error": {"message": "rate limit"}}), "太频繁", T),
    (lambda r: httpx.Response(503, json={"error": {"message": "overloaded"}}), "接口返回错误", T),
    (lambda r: httpx.Response(502, text="Bad Gateway"), "不是 Anthropic 协议", T),
    (lambda r: httpx.Response(404, text="Not Found"), "不是 Anthropic 协议", C),
    (lambda r: httpx.Response(400, json={"error": {"message": "unknown tool type web_search_20250305"}}), "不支持自带搜索", C),
    (lambda r: httpx.Response(200, text="<html>hi</html>"), "不是 Anthropic 协议", C),
    (lambda r: httpx.Response(200, json={"choices": []}), "不是 Anthropic 协议", C),  # OpenAI 协议的回应
    (lambda r: httpx.Response(200, json={"content": [{"type": "text", "text": "我不能上网"}]}), "没有执行搜索", C),
    (lambda r: httpx.Response(200, json={"content": [
        {"type": "server_tool_use", "id": "s", "input": {"query": "q"}},
        {"type": "web_search_tool_result", "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"}}]}),
     "unavailable", T),
])
async def test_search_error_reasons(fake_api, respond, reason, kind):
    fake_api.state["respond"] = respond
    with pytest.raises(ws.SearchError, match=reason) as e:
        await ws.search(SETTINGS, "q")
    assert e.value.kind == kind and e.value.spoken


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


class FakeTTS:
    def __init__(self):
        self.said = []

    def tts_one_sentence(self, conn, content_type, content_detail=None, sentence_id=None, **kw):
        self.said.append((content_detail, sentence_id))


def fake_conn(settings=SETTINGS, **kw):
    return SimpleNamespace(config={"plugins": {"web_search": settings}}, tts=FakeTTS(), sentence_id="sid-1",
                           client_abort=False, **kw)


async def test_plugin_replies_by_error_kind(fake_api):
    # 要用户处理的：说出原因和去哪里改
    fake_api.state["respond"] = lambda r: httpx.Response(402, json={"error": {"message": "Insufficient Balance"}})
    conn = fake_conn()
    resp = await ws.web_search(conn, query="q")
    assert "余额不足" in resp.result and "管理后台" in resp.result and "不要编造" in resp.result
    # 临时问题：只说这次没搜到
    fake_api.state["respond"] = lambda r: httpx.Response(429, json={"error": {"message": "rate"}})
    resp = await ws.web_search(conn, query="q")
    assert "没搜到" in resp.result and "管理后台" not in resp.result
    # 没填 Key：说还没开通，而且不先说"我上网查一下"
    conn = fake_conn({**SETTINGS, "api_key": ""})
    resp = await ws.web_search(conn, query="q")
    assert "还没开通" in resp.result and conn.tts.said == []
    fake_api.state["respond"] = lambda r: httpx.Response(200, json=OK_BODY)
    resp = await ws.web_search(fake_conn(), query="q")
    assert resp.result.startswith("【联网搜索结果】")


async def test_plugin_says_searching_first(fake_api):
    conn = fake_conn()
    await ws.web_search(conn, query="q")
    assert conn.tts.said == [(ws.SEARCHING_REPLY, "sid-1")] and ws.SEARCHING_REPLY.endswith("。")
    conn = fake_conn()
    conn.client_abort = True  # 用户已经打断：不说
    await ws.web_search(conn, query="q")
    assert conn.tts.said == []


async def test_system_prompt_has_weekday(fake_api, monkeypatch):
    monkeypatch.setattr(ws.time, "localtime", lambda *a: SimpleNamespace(tm_wday=2))
    monkeypatch.setattr(ws.time, "strftime", lambda fmt, *a: "2026-11-04")
    await ws.search(SETTINGS, "q")
    assert "今天是 2026-11-04 星期三" in json.loads(fake_api.calls[-1].content)["system"]
    assert ws.TIMEOUT.read == 20.0 and ws.TIMEOUT.connect == 5.0


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
    assert r == {"ok": True, "seconds": r["seconds"], "count": 2, "first": "第二条"}
    await client.post("/admin/api/plugin-settings/web_search/test",
                      json={"api_key": "sk-good", "base_url": "http://10.0.0.2:8317/v1"})
    assert str(fake_api.calls[-1].url) == "http://10.0.0.2:8317/v1/messages"
    fake_api.state["respond"] = lambda req: httpx.Response(401, json={"error": {"message": "bad sk-good"}})
    r = await client.post("/admin/api/plugin-settings/web_search/test", json={"api_key": "sk-good"})
    err = (await r.json())["error"]
    assert r.status == 400 and "API Key 不对" in err and "sk-good" not in err


# ---------- 状态页：勾了联网搜索但没填 Key ----------

def _status(merged):
    from core.admin import status

    llm = (merged.get("selected_module") or {}).get("LLM")
    return status._plugin_setting_checks(merged, config_store.entry_type(merged, "LLM", llm) if llm else None)


def _merged(llm_type, intent, functions, key="", api_intent=None):
    m = {"selected_module": {"LLM": "M", "Intent": intent}, "LLM": {"M": {"type": llm_type}},
         "Intent": {intent: {"type": intent}, "function_call": {"type": "function_call", "functions": functions}},
         "plugins": {"web_search": {"api_key": key}}}
    if api_intent:
        m["admin"] = {"api_intent": api_intent}
    return m


def test_status_warns_only_when_plugins_used():
    on = ["web_search", "get_weather"]
    warn = _status(_merged("openai", "function_call", on))
    assert len(warn) == 1 and warn[0]["level"] == "warn" and "联网搜索" in warn[0]["detail"]
    assert _status(_merged("openai", "function_call", on, key="sk-x")) == []          # 填了 Key
    assert _status(_merged("openai", "function_call", ["get_weather"])) == []          # 没勾
    assert _status(_merged("openai", "nointent", on)) == []                            # API 模型不用插件
    # 默认是智能体：自己不用插件；设备切到 API 模型时会用（没选过默认用插件），选过「不用插件」就不提醒
    assert len(_status(_merged("claude_cli", "nointent", on))) == 1
    assert len(_status(_merged("grok_bot", "nointent", on, api_intent="function_call"))) == 1
    assert _status(_merged("claude_cli", "nointent", on, api_intent="nointent")) == []


# ---------- 升级：配置里还留着旧版（秘塔等）的设置 ----------

LEGACY_YAML = """\
selected_module:
  LLM: ClaudeCLI
  Intent: nointent
plugins:
  web_search:
    provider: metaso
    max_results: 5
    api_key: mk-realmetasokey12345
"""


async def test_legacy_key_never_used_or_sent(tmp_path, fake_api):
    # 插件：旧版的 Key 当成没填，不会发给新接口
    assert ws.settings_of({"plugins": {"web_search": {"provider": "metaso", "api_key": "mk-real"}}})["api_key"] == ""
    p = tmp_path / ".config.yaml"
    p.write_text(LEGACY_YAML, encoding="utf-8")
    m = config_store.load_merged(custom_path=p)
    d = plugin_settings.describe(m, "web_search")
    assert d["configured"] is False and d["legacy_note"]
    assert next(f for f in d["fields"] if f["key"] == "api_key")["set"] is False
    # 测试按钮留空 Key：不会拿旧 Key 去请求
    with pytest.raises(ValueError, match="API Key"):
        await plugin_settings.run_test("web_search", {}, m)
    assert fake_api.calls == []
    # 只改别的字段保存：旧字段和旧 Key 一起清掉
    changed = plugin_settings.save("web_search", {"max_uses": "3"}, m, p)
    assert set(changed) == {"max_uses", "api_key", "provider", "max_results"}
    saved = config_store.load_merged(custom_path=p)["plugins"]["web_search"]
    assert "provider" not in saved and "max_results" not in saved and saved["api_key"] == ""
    d = plugin_settings.describe(config_store.load_merged(custom_path=p), "web_search")
    assert d["legacy_note"] == "" and d["configured"] is False
    plugin_settings.save("web_search", {"api_key": "sk-new"}, config_store.load_merged(custom_path=p), p)
    assert plugin_settings.configured(config_store.load_merged(custom_path=p), "web_search") is True


async def test_total_timeout(fake_api, monkeypatch):
    import asyncio as aio
    monkeypatch.setattr(ws, "TOTAL_SECONDS", 0.05)

    async def slow_handler(request):
        await aio.get_running_loop().run_in_executor(None, __import__("time").sleep, 0.3)
        return httpx.Response(200, json=OK_BODY)
    from httpx._client import AsyncClient as real
    monkeypatch.setattr(ws.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(slow_handler), **kw))
    with pytest.raises(ws.SearchError, match="没有结果") as e:
        await ws.search(SETTINGS, "q")
    assert e.value.kind == ws.TEMPORARY
