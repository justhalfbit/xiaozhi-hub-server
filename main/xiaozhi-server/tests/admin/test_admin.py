"""管理后台：配置读写、密码、接口鉴权。全部使用临时配置文件，不碰 data/.config.yaml。"""

from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from core.admin import auth, config_store
from core.admin.admin_handler import AdminHandler

CUSTOM_YAML = """\
# 用户注释应被保留
selected_module:
  LLM: ClaudeAPI   # 行尾注释
LLM:
  ClaudeAPI:
    type: openai
    base_url: https://api.anthropic.com/v1/
    model_name: claude-haiku-5-5
    api_key: 你的Claude API Key
    max_tokens: 500
"""


@pytest.fixture
def cfg_path(tmp_path) -> Path:
    p = tmp_path / ".config.yaml"
    p.write_text(CUSTOM_YAML, encoding="utf-8")
    return p


def merged(cfg_path):
    return config_store.load_merged(custom_path=cfg_path)


# ---------- config_store ----------

def test_field_kind_detects_secrets():
    # max_tokens 之类的字段不能被误判成密钥
    assert config_store.field_kind("max_tokens", 500) == "number"
    assert config_store.field_kind("access_token", "x") == "secret"
    assert config_store.field_kind("token", "x") == "secret"


def test_update_writes_key_and_keeps_comments(cfg_path):
    changed = config_store.update_entry("LLM", "ClaudeAPI", {"api_key": "sk-ant-1234567890abcdef"}, merged(cfg_path), cfg_path)
    assert changed == ["api_key"]
    text = cfg_path.read_text(encoding="utf-8")
    assert "sk-ant-1234567890abcdef" in text
    assert "# 用户注释应被保留" in text and "# 行尾注释" in text
    assert config_store.mask_key(merged(cfg_path)["LLM"]["ClaudeAPI"]["api_key"]) == "sk-a…cdef"


def test_concurrent_saves_do_not_lose_updates(cfg_path, monkeypatch):
    """两个保存请求同时读改写配置文件时，后写的不能覆盖先写的修改。"""
    import threading

    real_load = config_store.load_custom
    barrier = threading.Barrier(2, timeout=0.5)

    def slow_load(path=None):
        data = real_load(path)
        try:
            barrier.wait()  # 没有锁时两个线程都会在这里读到同一份旧文件
        except threading.BrokenBarrierError:
            pass
        return data

    monkeypatch.setattr(config_store, "load_custom", slow_load)
    m = merged(cfg_path)
    jobs = [threading.Thread(target=config_store.update_entry, args=("LLM", "ClaudeAPI", {"api_key": "sk-new-1234567890"}, m, cfg_path)),
            threading.Thread(target=config_store.select_entry, args=("LLM", "ChatGLMLLM", m, cfg_path))]
    for t in jobs:
        t.start()
    for t in jobs:
        t.join()
    after = merged(cfg_path)
    assert after["LLM"]["ClaudeAPI"]["api_key"] == "sk-new-1234567890"
    assert after["selected_module"]["LLM"] == "ChatGLMLLM"


def test_update_empty_secret_means_unchanged(cfg_path):
    config_store.update_entry("LLM", "ClaudeAPI", {"api_key": "sk-old-1234567890"}, merged(cfg_path), cfg_path)
    config_store.update_entry("LLM", "ClaudeAPI", {"api_key": "", "model_name": "claude-sonnet-5-5"}, merged(cfg_path), cfg_path)
    e = merged(cfg_path)["LLM"]["ClaudeAPI"]
    assert e["api_key"] == "sk-old-1234567890" and e["model_name"] == "claude-sonnet-5-5"


def test_update_converts_types(cfg_path):
    config_store.update_entry("LLM", "ClaudeAPI", {"max_tokens": "800", "extra_body": '{"reasoning_effort": "none"}'},
                              merged(cfg_path), cfg_path)
    e = merged(cfg_path)["LLM"]["ClaudeAPI"]
    assert e["max_tokens"] == 800 and e["extra_body"] == {"reasoning_effort": "none"}
    with pytest.raises(ValueError):
        config_store.update_entry("LLM", "ClaudeAPI", {"extra_body": "[1, 2]"}, merged(cfg_path), cfg_path)
    with pytest.raises(ValueError):
        config_store.update_entry("LLM", "ClaudeAPI", {"max_tokens": "abc"}, merged(cfg_path), cfg_path)


def test_update_list_and_bool_fields(cfg_path):
    cfg_path.write_text(CUSTOM_YAML + """  ClaudeCLI:
    type: claude_cli
    allowed_tools: [Read, Grep, Glob]
    turn_timeout_seconds: 120
""", encoding="utf-8")
    config_store.update_entry("LLM", "ClaudeCLI", {"allowed_tools": "Read, Glob，Grep", "turn_timeout_seconds": "60"},
                              merged(cfg_path), cfg_path)
    e = merged(cfg_path)["LLM"]["ClaudeCLI"]
    assert e["allowed_tools"] == ["Read", "Glob", "Grep"] and e["turn_timeout_seconds"] == 60
    assert config_store.field_kind("enabled", True) == "bool"
    assert config_store._convert("bool", "true", False) is True


def test_update_ignores_unknown_fields(cfg_path):
    assert config_store.update_entry("LLM", "ClaudeAPI", {"type": "evil", "foo": "bar"}, merged(cfg_path), cfg_path) == []
    e = merged(cfg_path)["LLM"]["ClaudeAPI"]
    assert e["type"] == "openai" and "foo" not in e


def test_update_default_entry_only_overrides_changed_field(cfg_path):
    config_store.update_entry("LLM", "ChatGLMLLM", {"api_key": "glm-key-1234567890"}, merged(cfg_path), cfg_path)
    custom = config_store.load_custom(cfg_path)
    assert dict(custom["LLM"]["ChatGLMLLM"]) == {"api_key": "glm-key-1234567890"}
    assert merged(cfg_path)["LLM"]["ChatGLMLLM"]["type"] == "openai"


def test_select_entry_other_sections(cfg_path):
    config_store.select_entry("TTS", "EdgeTTS", merged(cfg_path), cfg_path)
    config_store.select_entry("Intent", "nointent", merged(cfg_path), cfg_path)
    m = merged(cfg_path)["selected_module"]
    assert m["TTS"] == "EdgeTTS" and m["Intent"] == "nointent" and m["LLM"] == "ClaudeAPI"
    with pytest.raises(KeyError):
        config_store.select_entry("LLM", "NoSuchLLM", merged(cfg_path), cfg_path)
    with pytest.raises(KeyError):
        config_store.select_entry("VAD", "SileroVAD", merged(cfg_path), cfg_path)


# ---------- auth ----------

def test_password_hash_roundtrip():
    h = auth.hash_password("correct horse")
    assert auth.verify_password("correct horse", h)
    assert not auth.verify_password("wrong", h)
    assert not auth.verify_password("x", "garbage")


def test_session_store_expiry():
    store = auth.SessionStore(ttl_seconds=-1)
    assert not store.valid(store.create())
    store = auth.SessionStore()
    t = store.create()
    assert store.valid(t)
    store.revoke(t)
    assert not store.valid(t)


# ---------- HTTP 接口 ----------

@pytest.fixture
async def client(cfg_path):
    app = web.Application()
    config = {"selected_module": {"LLM": "ClaudeAPI"}, "LLM": {}}
    AdminHandler(config, ws_server=None, config_path=cfg_path).register(app)
    async with TestClient(TestServer(app)) as c:
        yield c


async def test_setup_login_and_update_flow(client, cfg_path):
    st = await (await client.get("/admin/api/state")).json()
    assert st == {"needs_setup": True, "logged_in": False}

    r = await client.post("/admin/api/setup", json={"password": "abc12"})
    assert r.status == 400
    r = await client.post("/admin/api/setup", json={"password": "pass-1234"})
    assert r.status == 200
    assert (await (await client.get("/admin/api/state")).json())["logged_in"] is True

    r = await client.post("/admin/api/setup", json={"password": "another-pass"})
    assert r.status == 409

    assert (await client.get("/admin/api/llm")).status == 200

    await client.post("/admin/api/logout", json={})
    assert (await client.get("/admin/api/llm")).status == 401
    assert (await client.post("/admin/api/login", json={"password": "pass-1234"})).status == 200
    assert (await client.get("/admin/api/llm")).status == 200


async def test_requires_login_and_json(client):
    assert (await client.get("/admin/api/llm")).status == 401
    assert (await client.get("/admin/api/status")).status == 401
    assert (await client.post("/admin/api/llm/use", json={"vendor": "glm"})).status == 401
    r = await client.post("/admin/api/setup", data="password=pass-1234",
                          headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert r.status == 415


async def test_clash_window_requires_admin_and_same_origin(client):
    base = "/admin/api/clash-party"
    assert (await client.get(base + "/update?check=1")).status == 401
    assert (await client.post(base + "/update", json={"version": "2.0.5"})).status == 401
    assert (await client.get(base)).status == 401
    assert (await client.get(base + "/display")).status == 401
    assert (await client.get("/admin/clash-view/core/rfb.js")).status == 401
    assert (await client.post(base + "/paste", json={"text": "abc"})).status == 401
    await client.post("/admin/api/setup", json={"password": "pass-1234"})
    assert (await client.get(base)).status == 200
    for headers in ({}, {"Origin": "http://other.local"}):
        assert (await client.post(base + "/update", headers=headers, json={"version": "2.0.5"})).status == 403
        assert (await client.get(base + "/display", headers=headers)).status == 403
        assert (await client.post(base + "/paste", headers=headers, json={"text": "abc"})).status == 403


async def test_setup_allowed_from_remote(cfg_path, monkeypatch):
    """首次设置密码不限制来源：部署到云端服务器时，从自己的电脑远程打开后台也能直接设置。"""
    monkeypatch.setattr(web.BaseRequest, "remote", property(lambda self: "203.0.113.5"))  # 模拟公网上的访问者
    app = web.Application()
    AdminHandler({"selected_module": {}}, config_path=cfg_path).register(app)
    async with TestClient(TestServer(app)) as c:
        r = await c.post("/admin/api/setup", json={"password": "pass-1234"})
        assert r.status == 200
    assert config_store.get_password_hash(cfg_path)


async def test_wrong_password(client):
    await client.post("/admin/api/setup", json={"password": "pass-1234"})
    await client.post("/admin/api/logout", json={})
    r = await client.post("/admin/api/login", json={"password": "nope-nope"})
    assert r.status == 401


async def test_status_endpoint(client):
    await client.post("/admin/api/setup", json={"password": "pass-1234"})
    data = await (await client.get("/admin/api/status")).json()
    assert data["ok"] and data["addresses"]["ota"].endswith("/xiaozhi/ota/")
    names = [c["name"] for c in data["checks"]]
    assert "ffmpeg" in names and "opus" in names
    assert set(data["modules"]) == {"LLM", "ASR", "TTS", "Intent"}


def test_unchanged_fields_are_not_written(cfg_path):
    before = cfg_path.read_text(encoding="utf-8")
    changed = config_store.update_entry(
        "LLM", "ClaudeAPI",
        {"model_name": "claude-haiku-5-5", "base_url": "https://api.anthropic.com/v1/", "max_tokens": "500",
         "api_key": "", "extra_body": "{}"},
        merged(cfg_path), cfg_path)
    assert changed == []  # 值都没变，空的 extra_body 也不算新增
    assert "model_name: claude-haiku-5-5" in cfg_path.read_text(encoding="utf-8") and before != ""


async def test_devices_and_emotion_test(client):
    from types import SimpleNamespace
    from core.admin import connections

    await client.post("/admin/api/setup", json={"password": "pass-1234"})
    assert (await (await client.get("/admin/api/devices")).json())["devices"] == []
    sent = []

    class FakeWS:
        async def send(self, data):
            sent.append(data)

    fake = SimpleNamespace(session_id="s-1", device_id="dev-1", client_ip="192.168.6.211", websocket=FakeWS(),
                           hub_route={"current": "GrokBot_孙悟空"})  # 呼名切到的模型
    connections.register(fake)
    try:
        devices = (await (await client.get("/admin/api/devices")).json())["devices"]
        assert devices[0]["device_id"] == "dev-1" and devices[0]["model_label"] == "孙悟空"
        r = await client.post("/admin/api/devices/s-1/emotion-test", json={"interval": 0.5})
        data = await r.json()
        assert r.status == 200 and data == {"ok": True}
        import asyncio
        for _ in range(200):  # conftest 让 asyncio.sleep 立即返回，后台任务几轮调度即可发完
            if len(sent) == 21:
                break
            await asyncio.sleep(0)
        assert '"emotion": "funny"' in sent[0]
        assert (await client.post("/admin/api/devices/none/emotion-test", json={})).status == 404
    finally:
        connections.unregister(fake)


def test_authorization_like_fields_are_secret():
    for key in ("authorization", "api_token", "refresh_token", "cookie", "credential"):
        assert config_store.field_kind(key, "Bearer abc") == "secret", key
    assert config_store.field_kind("max_tokens", 500) == "number"
    assert config_store.field_kind("max_new_tokens", "1024") == "text"


async def test_emotion_test_rejects_bad_interval_and_survives_disconnect(client):
    from types import SimpleNamespace
    from core.admin import connections

    await client.post("/admin/api/setup", json={"password": "pass-1234"})

    class BrokenWS:
        async def send(self, data):
            raise ConnectionError("closed")

    fake = SimpleNamespace(session_id="s-2", device_id="dev-2", client_ip="x", websocket=BrokenWS())
    connections.register(fake)
    try:
        assert (await client.post("/admin/api/devices/s-2/emotion-test", json={"interval": "abc"})).status == 400
        r = await client.post("/admin/api/devices/s-2/emotion-test", json={"interval": 1})
        assert r.status == 200  # 发送失败只记日志，不产生未处理的任务异常
        import asyncio
        for _ in range(20):
            await asyncio.sleep(0)
    finally:
        connections.unregister(fake)


async def test_apply_failure_keeps_running_config(cfg_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from core.admin import admin_handler as ah

    def boom(*a, **k):
        raise RuntimeError("load failed")
    monkeypatch.setattr(ah, "initialize_modules", boom)
    ws = SimpleNamespace(config_lock=asyncio.Lock(), _llm="old-llm")
    running = {"selected_module": {"LLM": "ClaudeAPI"}, "LLM": {"ClaudeAPI": {"type": "openai"}}}
    h = ah.AdminHandler(running, ws_server=ws, config_path=cfg_path)
    config_store.select_entry("LLM", "ChatGLMLLM", config_store.load_merged(custom_path=cfg_path), cfg_path)
    result = await h._apply("LLM")
    assert result["applied"] is False
    assert running["selected_module"]["LLM"] == "ClaudeAPI" and ws._llm == "old-llm"  # 失败时服务保持原状


async def test_apply_retires_replaced_instance(cfg_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from core.admin import admin_handler as ah

    class OldLLM:
        retired = False

        def shutdown(self):
            self.retired = True

    new_llm = object()
    monkeypatch.setattr(ah, "initialize_modules", lambda *a, **k: {"llm": new_llm})
    old = OldLLM()
    ws = SimpleNamespace(config_lock=asyncio.Lock(), _llm=old)
    running = {"selected_module": {"LLM": "ClaudeAPI"}, "LLM": {"ClaudeAPI": {"type": "openai"}}}
    h = ah.AdminHandler(running, ws_server=ws, config_path=cfg_path)
    config_store.select_entry("LLM", "ChatGLMLLM", config_store.load_merged(custom_path=cfg_path), cfg_path)
    assert (await h._apply("LLM"))["applied"] is True
    assert ws._llm is new_llm and old.retired  # 替换后通知旧实例退役

    ws._llm = "plain-llm"  # 没有 shutdown 方法的实例直接替换，不报错
    assert (await h._apply("LLM"))["applied"] is True and ws._llm is new_llm


# ---------- 模块依赖检测（精简安装） ----------

def test_missing_packages_reads_module_level_imports(tmp_path, monkeypatch):
    from core.admin import deps

    src = tmp_path / "fake_provider.py"
    src.write_text(
        "import os\n"
        "import xiaozhi_missing_top\n"
        "from xiaozhi_missing_from.sub import thing\n"
        "if True:\n"
        "    import xiaozhi_missing_in_if\n"
        "try:\n"
        "    import xiaozhi_optional\n"
        "except ImportError:\n"
        "    pass\n"
        "def lazy():\n"
        "    import xiaozhi_lazy\n"
        "from core.admin import config_store\n",  # 项目自身的模块：递归检查，本身不算缺失
        encoding="utf-8",
    )
    monkeypatch.setattr(deps, "provider_file", lambda section, type_: src)
    clear_deps_cache(deps)
    try:
        assert deps.missing_packages("ASR", "fake") == (
            "xiaozhi_missing_from", "xiaozhi_missing_in_if", "xiaozhi_missing_top")
    finally:
        clear_deps_cache(deps)


def clear_deps_cache(deps):
    deps.missing_packages.cache_clear()
    deps._find_top.cache_clear()
    deps._submodule_exists.cache_clear()


def test_missing_packages_real_providers(monkeypatch):
    import importlib.machinery
    import sys
    from core.admin import deps

    # tests/conftest.py 会把 funasr 等重型包换成桩模块，这里移除桩，按真实安装情况判断
    for name in [m for m in sys.modules if m == "funasr" or m.startswith("funasr.")]:
        monkeypatch.delitem(sys.modules, name)
    clear_deps_cache(deps)
    try:
        assert deps.missing_packages("ASR", "sherpa_onnx_local") == ()  # 默认 ASR，精简安装也必须齐全
        assert deps.missing_packages("TTS", "edge") == ()
        assert deps.missing_packages("LLM", "claude_cli") == ()
        has_funasr = importlib.machinery.PathFinder.find_spec("funasr") is not None
        assert deps.missing_packages("ASR", "fun_local") == (() if has_funasr else ("funasr",))
        assert deps.missing_packages("ASR", "no_such_provider") == ()  # 找不到源文件交给原项目报错
    finally:
        clear_deps_cache(deps)


def test_status_checks_module_packages_and_sherpa_model(tmp_path, monkeypatch):
    from core.admin import deps
    from core.admin.status import collect_status

    model_dir = tmp_path / "sherpa"
    model_dir.mkdir()
    p = tmp_path / ".config.yaml"
    p.write_text(
        "selected_module:\n  ASR: SherpaASR\n  LLM: ClaudeAPI\n"
        f"ASR:\n  SherpaASR:\n    model_dir: {model_dir}\n", encoding="utf-8")

    def check(name):
        m = config_store.load_merged(custom_path=p)
        return next(c for c in collect_status(m, m)["checks"] if c["name"] == name)

    c = check("语音识别模型")
    assert c["level"] == "error" and "model.int8.onnx" in c["detail"]
    with open(model_dir / "model.int8.onnx", "wb") as f:
        f.truncate(240 * 1024 * 1024)  # 稀疏文件，不占实际空间
    c = check("语音识别模型")
    assert c["level"] == "error" and "tokens.txt" in c["detail"]
    (model_dir / "tokens.txt").write_text("x", encoding="utf-8")
    assert check("语音识别模型")["level"] == "ok"

    assert check("模块依赖")["level"] == "ok"
    monkeypatch.setattr(deps, "missing_packages",
                        lambda section, type_: ("sherpa_onnx",) if section == "ASR" else ())
    c = check("模块依赖")
    assert c["level"] == "error" and "SherpaASR 缺少 sherpa_onnx" in c["detail"]


def test_provider_file_covers_unmanaged_sections():
    from core.admin import deps

    # 后台不管理的模块也要能找到源文件，自检才能发现手动配置的模块缺依赖
    assert deps.provider_file("Memory", "nomem").is_file()
    assert deps.provider_file("Memory", "mem0ai").is_file()
    assert deps.provider_file("VAD", "silero").is_file()
    assert deps.provider_file("VLLM", "openai").is_file()


def test_status_checks_packages_of_memory_module(tmp_path, monkeypatch):
    from core.admin import deps
    from core.admin.status import collect_status

    p = tmp_path / ".config.yaml"
    p.write_text("selected_module:\n  Memory: mem0ai\n  LLM: ClaudeAPI\n", encoding="utf-8")
    monkeypatch.setattr(deps, "missing_packages",
                        lambda section, type_: ("mem0",) if section == "Memory" else ())
    m = config_store.load_merged(custom_path=p)
    c = next(c for c in collect_status(m, m)["checks"] if c["name"] == "模块依赖")
    assert c["level"] == "error" and "mem0ai 缺少 mem0" in c["detail"]
