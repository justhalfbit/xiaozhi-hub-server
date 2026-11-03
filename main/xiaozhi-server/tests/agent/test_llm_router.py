"""多模型路由的单元测试：按连接切换模型、各聊各的、记住上次的模型、连接关闭通知到每个用过的模型。"""

from types import SimpleNamespace

import pytest

from core.agent import llm_router
from core.agent.session_store import SessionStore
from core.utils import llm as llm_utils
from core.utils.dialogue import Dialogue, Message


class FakeLLM:
    def __init__(self, name, config=None):
        self.name, self.config = name, config
        self.opened, self.closed, self.retired = [], [], False

    def on_conn_open(self, conn):
        self.opened.append(conn.session_id)

    def on_conn_close(self, conn):
        self.closed.append(conn.session_id)

    def shutdown(self):
        self.retired = True


def server_config():
    return {
        "selected_module": {"LLM": "ClaudeCLI"},
        "LLM": {
            "ClaudeCLI": {"type": "claude_cli"},
            "DeepSeekLLM": {"type": "openai", "url": "https://api.deepseek.com", "model_name": "deepseek-chat",
                            "api_key": "sk-123"},
            "KimiLLM": {"type": "openai", "base_url": "https://api.moonshot.cn/v1", "model_name": "kimi",
                        "api_key": "你的api_key"},
        },
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    created = []

    def create(kind, config):
        inst = FakeLLM(kind, config)
        created.append(inst)
        return inst

    monkeypatch.setattr(llm_utils, "create_instance", create)
    monkeypatch.setattr(llm_router, "_store", SessionStore(tmp_path / "agent_sessions.json", "last_model"))
    monkeypatch.setattr(llm_router, "_pool", {})
    default = FakeLLM("default")
    server = SimpleNamespace(config=server_config(), _llm=default)
    return SimpleNamespace(server=server, default=default, created=created)


def make_conn(env, sid="s1", device="dev-1"):
    dialogue = Dialogue()
    dialogue.put(Message(role="system", content="系统提示"))
    dialogue.put(Message(role="user", content="示例", is_temporary=True))
    return SimpleNamespace(session_id=sid, device_id=device, server=env.server, llm=env.server._llm,
                           dialogue=dialogue, prompt="系统提示", config={})


def users(conn):
    return [m.content for m in conn.dialogue.dialogue if m.role == "user" and not m.is_temporary]


def test_model_ready():
    cfg = server_config()
    assert llm_router.model_ready(cfg, "ClaudeCLI")
    assert llm_router.model_ready(cfg, "DeepSeekLLM")
    assert not llm_router.model_ready(cfg, "KimiLLM")  # Key 还是占位文字
    assert not llm_router.model_ready(cfg, "Nope")


def test_switch_keeps_separate_dialogues(env):
    conn = make_conn(env)
    llm_router.on_conn_open(conn)
    assert env.default.opened == ["s1"]
    conn.dialogue.put(Message(role="user", content="问 Claude"))

    assert llm_router.switch(conn, "DeepSeekLLM")
    deepseek = env.created[0]
    assert conn.llm is deepseek and deepseek.opened == ["s1"]
    assert users(conn) == []  # 不带 Claude 的对话过去
    assert [m.content for m in conn.dialogue.dialogue if m.role == "system"] == ["系统提示"]
    assert any(m.is_temporary for m in conn.dialogue.dialogue)  # 工具调用示例保留
    conn.dialogue.put(Message(role="user", content="问 DeepSeek"))

    assert llm_router.switch(conn, "ClaudeCLI")
    assert conn.llm is env.default and users(conn) == ["问 Claude"]
    assert llm_router.switch(conn, "DeepSeekLLM")
    assert conn.llm is deepseek and users(conn) == ["问 DeepSeek"]  # 切回来还记得
    assert deepseek.opened == ["s1"]  # 同一个连接只通知一次


def test_unready_model_does_not_switch(env):
    conn = make_conn(env)
    llm_router.on_conn_open(conn)
    assert llm_router.switch(conn, "KimiLLM") is False
    assert conn.llm is env.default and env.created == []
    assert llm_router.remembered(conn) is None


def test_remembers_last_model_per_device(env):
    conn = make_conn(env)
    llm_router.on_conn_open(conn)
    llm_router.switch(conn, "DeepSeekLLM")
    assert llm_router.remembered(conn) == "DeepSeekLLM"

    again = make_conn(env, sid="s2")  # 同一台设备重连
    llm_router.on_conn_open(again)
    assert again.llm is env.created[0] and again.hub_route["current"] == "DeepSeekLLM"
    other = make_conn(env, sid="s3", device="dev-2")  # 别的设备不受影响
    llm_router.on_conn_open(other)
    assert other.llm is env.default and other.hub_route["current"] == "ClaudeCLI"


def test_remembered_model_gone_falls_back_to_default(env):
    conn = make_conn(env)
    llm_router.on_conn_open(conn)
    llm_router.switch(conn, "DeepSeekLLM")
    env.server.config["LLM"]["DeepSeekLLM"]["api_key"] = ""  # 后台把 Key 清掉了
    again = make_conn(env, sid="s2")
    llm_router.on_conn_open(again)
    assert again.llm is env.default


def test_close_notifies_every_used_model(env):
    conn = make_conn(env)
    llm_router.on_conn_open(conn)
    llm_router.switch(conn, "DeepSeekLLM")
    llm_router.switch(conn, "ClaudeCLI")
    llm_router.on_conn_close(conn)
    assert env.default.closed == ["s1"] and env.created[0].closed == ["s1"]


def test_instances_shared_and_rebuilt_when_config_changes(env):
    a, b = make_conn(env, "s1", "d1"), make_conn(env, "s2", "d2")
    for c in (a, b):
        llm_router.on_conn_open(c)
        llm_router.switch(c, "DeepSeekLLM")
    assert a.llm is b.llm and len(env.created) == 1  # 所有连接共用

    env.server.config["LLM"]["DeepSeekLLM"]["model_name"] = "deepseek-reasoner"  # 后台保存后热加载
    c = make_conn(env, "s3", "d3")
    llm_router.on_conn_open(c)
    llm_router.switch(c, "DeepSeekLLM")
    assert len(env.created) == 2 and c.llm is env.created[1]
    assert env.created[0].retired and a.llm is env.created[0]  # 在线连接继续用旧实例


def test_calling_current_model_again_uses_new_config(env):
    """已经在用某个模型，后台改了它的设置后再呼名：换成照新配置建的实例，这个模型的对话保留。"""
    conn = make_conn(env)
    llm_router.on_conn_open(conn)
    llm_router.switch(conn, "DeepSeekLLM")
    conn.dialogue.put(Message(role="user", content="问 DeepSeek"))
    old = conn.llm
    assert llm_router.switch(conn, "DeepSeekLLM") and conn.llm is old and len(env.created) == 1  # 没改：不重建

    env.server.config["LLM"]["DeepSeekLLM"]["api_key"] = "sk-new"
    assert llm_router.switch(conn, "DeepSeekLLM")
    new = env.created[1]
    assert conn.llm is new and new.config["api_key"] == "sk-new" and new.opened == ["s1"]
    assert old.retired and users(conn) == ["问 DeepSeek"]
    llm_router.on_conn_close(conn)
    assert old.closed == ["s1"] and new.closed == ["s1"]  # 新旧实例都通知到


def test_default_model_uses_server_instance(env):
    env.server.config["selected_module"]["LLM"] = "DeepSeekLLM"
    env.server._llm = FakeLLM("deepseek-default")
    conn = make_conn(env)
    conn.llm = env.server._llm
    llm_router.on_conn_open(conn)
    assert llm_router.switch(conn, "ClaudeCLI") and conn.llm is env.created[0]
    assert llm_router.switch(conn, "DeepSeekLLM") and conn.llm is env.server._llm


# ---------- 意图识别跟着这台设备的模型走 ----------

def intent_config(default="ClaudeCLI", intent="nointent", api_intent=None):
    cfg = server_config()
    cfg["selected_module"] = {"LLM": default, "Intent": intent}
    cfg["LLM"]["ClaudeAPI"] = {"type": "openai", "base_url": "https://api.anthropic.com/v1/", "model_name": "claude",
                               "api_key": "sk-ant"}
    cfg["LLM"]["GrokBot_孙悟空"] = {"type": "grok_bot", "bot_id": "bot-1"}
    cfg["Intent"] = {"nointent": {"type": "nointent"}, "function_call": {"type": "function_call", "functions": []}}
    if api_intent:
        cfg["admin"] = {"api_intent": api_intent}
    return cfg


def test_intent_mode():
    cfg = intent_config()  # 后台默认 Claude Code（意图识别只能是 nointent）
    assert llm_router.intent_mode(cfg, "ClaudeCLI") == "nointent"
    assert llm_router.intent_mode(cfg, "DeepSeekLLM") == "function_call"
    assert llm_router.intent_mode(cfg, "ClaudeAPI") == "function_call"  # 所有 API 模型都能用插件
    assert llm_router.intent_mode(intent_config(api_intent="nointent"), "DeepSeekLLM") == "nointent"  # 后台选过
    cfg = intent_config(default="DeepSeekLLM", intent="function_call")
    assert llm_router.intent_mode(cfg, "ClaudeCLI") == "nointent"
    assert llm_router.intent_mode(cfg, "GrokBot_孙悟空") == "nointent"
    assert llm_router.intent_mode(cfg, "DeepSeekLLM") == "function_call"  # 默认模型照后台的设置
    cfg = intent_config(default="DeepSeekLLM", intent="nointent", api_intent="function_call")
    assert llm_router.intent_mode(cfg, "DeepSeekLLM") == "nointent"
    cfg["Intent"]["intent_llm"] = {"type": "intent_llm"}
    cfg["selected_module"]["Intent"] = "intent_llm"
    assert llm_router.intent_mode(cfg, "ClaudeCLI") is None  # 后台用的是别的意图识别：不按设备改


def intent_conn(env, sid="s1", device="dev-1"):
    conn = make_conn(env, sid, device)
    conn.config = {"selected_module": dict(env.server.config["selected_module"]),
                   "Intent": env.server.config["Intent"]}
    conn.intent_type = "nointent"
    conn.func_handler = None
    conn.loop = None
    conn.fewshot = 0

    def inject():
        conn.fewshot += 1
        conn.dialogue.put(Message(role="user", content="工具示例", is_temporary=True))
    conn._inject_tool_call_fewshot = inject
    return conn


def test_device_intent_follows_its_model_without_touching_global(env, monkeypatch):
    import core.providers.tools.unified_tool_handler as tools

    monkeypatch.setattr(tools, "UnifiedToolHandler", lambda conn: SimpleNamespace(conn=conn))
    env.server.config = intent_config()
    conn = intent_conn(env)
    conn.dialogue.dialogue = [m for m in conn.dialogue.dialogue if not m.is_temporary]  # nointent 下没有工具示例
    llm_router.on_conn_open(conn)
    assert conn.config["selected_module"]["Intent"] == "nointent"

    assert llm_router.switch(conn, "DeepSeekLLM")
    assert conn.intent_type == "function_call" and conn.config["selected_module"]["Intent"] == "function_call"
    assert conn.func_handler is not None and conn.fewshot == 1  # 第一次用插件：补建工具处理器和工具示例
    handler = conn.func_handler
    assert llm_router.switch(conn, "ClaudeCLI") and conn.intent_type == "nointent"
    assert llm_router.switch(conn, "DeepSeekLLM") and conn.intent_type == "function_call"
    assert conn.func_handler is handler and conn.fewshot == 1  # 切回来沿用
    assert env.server.config["selected_module"] == {"LLM": "ClaudeCLI", "Intent": "nointent"}  # 全局不变

    other = intent_conn(env, "s2", "dev-2")
    llm_router.on_conn_open(other)
    assert other.config["selected_module"]["Intent"] == "nointent"  # 别的设备不受影响

    # 重连：沿用上次的模型，意图识别在组件初始化前就按它改好（_initialize_intent 照着建）
    again = intent_conn(env, "s3")
    llm_router.on_conn_open(again)
    assert again.hub_route["current"] == "DeepSeekLLM"
    assert again.config["selected_module"]["Intent"] == "function_call"
    assert again.intent_type == "nointent" and again.func_handler is None
