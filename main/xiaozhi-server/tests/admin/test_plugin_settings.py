"""后台插件设置框架：按字段清单读写、secret 只写不读、选填字段、校验、保存后马上生效、测试接口不带出 Key。
用一个假的插件设置 demo，不依赖具体插件。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from core.admin import admin_handler as ah
from core.admin import catalog, config_store, connections, pages, plugin_settings

DEMO = {
    "note": "示例说明",
    "fields": [
        {"key": "base_url", "label": "接口地址", "type": "text", "default": "https://demo.example", "hint": "地址说明"},
        {"key": "api_key", "label": "API Key", "type": "secret", "required": True},
        {"key": "backup_key", "label": "备用 Key", "type": "secret"},  # 选填
        {"key": "city", "label": "默认城市", "type": "text", "default": "广州"},
        {"key": "count", "label": "次数", "type": "number", "default": 5, "min": 1, "max": 10},
    ],
}


@pytest.fixture(autouse=True)
def demo_plugin(monkeypatch):
    monkeypatch.setitem(catalog.PLUGIN_SETTINGS, "demo", DEMO)
    monkeypatch.setattr(catalog, "PLUGINS", catalog.PLUGINS + [
        {"name": "demo_fn", "label": "示例", "desc": "", "config": "demo_cfg", "settings": "demo"}])
    monkeypatch.setattr(catalog, "PLUGIN_NAMES", catalog.PLUGIN_NAMES | {"demo_fn"})


CUSTOM_YAML = """\
selected_module:
  LLM: ClaudeCLI
  Intent: nointent
plugins:
  demo_cfg:
    api_key: mk-xxx
"""


@pytest.fixture
def cfg_path(tmp_path) -> Path:
    p = tmp_path / ".config.yaml"
    p.write_text(CUSTOM_YAML, encoding="utf-8")
    return p


def merged(path):
    return config_store.load_merged(custom_path=path)


def demo_needs_config(path):
    return next(p for p in pages.describe_intent(merged(path))["plugins"] if p["name"] == "demo_fn")["needs_config"]


def test_describe_and_save_secret_write_only(cfg_path):
    d = plugin_settings.describe(merged(cfg_path), "demo")
    assert d["configured"] is False and d["testable"] is False and d["note"] == "示例说明"
    key = next(f for f in d["fields"] if f["key"] == "api_key")
    assert key["set"] is False and "value" not in key  # 占位 Key 算没填，也不回值
    assert next(f for f in d["fields"] if f["key"] == "base_url")["hint"] == "地址说明"
    assert demo_needs_config(cfg_path) is True

    with pytest.raises(ValueError, match="之间"):
        plugin_settings.save("demo", {"count": "0"}, merged(cfg_path), cfg_path)
    with pytest.raises(ValueError, match="整数"):
        plugin_settings.save("demo", {"count": "abc"}, merged(cfg_path), cfg_path)
    with pytest.raises(ValueError, match="http"):
        plugin_settings.save("demo", {"base_url": "demo.example"}, merged(cfg_path), cfg_path)
    with pytest.raises(ValueError, match="空格"):
        plugin_settings.save("demo", {"api_key": "a b"}, merged(cfg_path), cfg_path)

    changed = plugin_settings.save("demo", {"api_key": "sk-secret-123456789", "count": "3", "city": "北京"},
                                   merged(cfg_path), cfg_path)
    assert sorted(changed) == ["api_key", "city", "count"]
    d = plugin_settings.describe(merged(cfg_path), "demo")
    key = next(f for f in d["fields"] if f["key"] == "api_key")
    assert d["configured"] is True and key["set"] is True and "sk-secret-123456789" not in json.dumps(d)
    assert demo_needs_config(cfg_path) is False  # 选填的备用 Key 没填不影响
    # Key 留空表示不改；没提交的字段不动；没变的不算修改；填默认值和留空一样
    assert plugin_settings.save("demo", {"api_key": "", "count": "3"}, merged(cfg_path), cfg_path) == []
    assert plugin_settings.save("demo", {"base_url": "https://demo.example"}, merged(cfg_path), cfg_path) == []
    saved = merged(cfg_path)["plugins"]["demo_cfg"]
    assert saved["api_key"] == "sk-secret-123456789" and saved["city"] == "北京" and "base_url" not in saved
    # text 留空：恢复默认（存成空）
    assert plugin_settings.save("demo", {"city": ""}, merged(cfg_path), cfg_path) == ["city"]


@pytest.fixture
async def client(cfg_path):
    app = web.Application()
    config = {"selected_module": {"LLM": "ClaudeCLI", "Intent": "nointent"}, "LLM": {}, "Intent": {},
              "plugins": {"demo_cfg": {"api_key": "mk-xxx"}}}
    handler = ah.AdminHandler(config, ws_server=None, config_path=cfg_path)
    handler.register(app)
    async with TestClient(TestServer(app)) as c:
        await c.post("/admin/api/setup", json={"password": "pass-1234"})
        c.handler = handler
        yield c


async def test_http_save_applies_to_live_connections(client):
    conn = SimpleNamespace(session_id="s-demo", config={"plugins": {"demo_cfg": {"api_key": "mk-xxx"}}})
    connections.register(conn)
    try:
        assert (await client.get("/admin/api/plugin-settings/nope")).status == 404
        d = await (await client.get("/admin/api/plugin-settings/demo")).json()
        assert d["ok"] and d["configured"] is False
        r = await (await client.post("/admin/api/plugin-settings/demo", json={"api_key": "sk-good-0001"})).json()
        assert r["ok"] and r["changed"] == ["api_key"] and r["configured"] is True
        assert client.handler.config["plugins"]["demo_cfg"]["api_key"] == "sk-good-0001"
        assert conn.config["plugins"]["demo_cfg"]["api_key"] == "sk-good-0001"  # 在线连接不用重连
        r = await client.post("/admin/api/plugin-settings/demo", json={"count": "99"})
        assert r.status == 400
        assert (await client.post("/admin/api/plugin-settings/demo/test", json={})).status == 404  # 没登记测试
    finally:
        connections.unregister(conn)


async def test_http_test_uses_unsaved_values_and_hides_key(client, monkeypatch):
    seen = []

    async def fake_test(values):
        seen.append(values)
        if values["api_key"] == "sk-bad-key-9999":
            raise ValueError("Authentication Fails, your api key sk-bad-key-9999 is invalid")
        return {"count": 2, "echo": f"key={values['api_key']}"}
    monkeypatch.setitem(plugin_settings.TESTS, "demo", fake_test)

    d = await (await client.get("/admin/api/plugin-settings/demo")).json()
    assert d["testable"] is True
    r = await client.post("/admin/api/plugin-settings/demo/test", json={"api_key": "sk-bad-key-9999", "city": "上海"})
    body = await r.json()
    assert r.status == 400 and "sk-bad-key-9999" not in body["error"] and "***" in body["error"]
    assert seen[-1]["city"] == "上海"  # 测试用页面上还没保存的值
    body = await (await client.post("/admin/api/plugin-settings/demo/test", json={"api_key": "sk-good-0002"})).json()
    assert body == {"ok": True, "count": 2, "echo": "key=***"}
