"""多模型路由：按称呼把对话交给不同的大模型（LLM 配置里的条目，如 ClaudeCLI、DeepSeekLLM）。

- 每个连接自己决定用哪个模型：切换时只换这个连接的 conn.llm，不影响其他设备；
- 各聊各的：每个模型在这个连接里有自己的一份对话记录（conn.dialogue 跟着换），切换时不带上下文过去；
  claude_cli 本来就按设备保存会话；API 模型的记录按设备和模型存进 data/chat.db，新连接时载入最近一段（chat_history）；
- 每台设备上次用的模型存到 data/agent_sessions.json，重连、服务重启后仍有效；
- 模型实例按条目名缓存、所有连接共用（和原来共享 ws_server._llm 一样）。后台默认模型直接用 ws_server._llm；
  其他模型第一次被叫到时才创建。条目配置变了（后台保存后热加载）就让旧实例退役，下次叫到时重建，
  在线连接继续用旧实例，重连后用新的；
- 切到别的模型后，原来那个模型不会马上释放：claude_cli 的进程等空闲超时再关，方便切回来；
  连接关闭时，这个连接用过的每个模型都会收到 on_conn_close；
- 意图识别也跟着这个连接的模型走（见 intent_mode）：Claude Code、Grok Bot 不用插件，API 模型用插件，
  只改这个连接自己的配置副本，不动后台的全局设置。
"""

import asyncio
import copy
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from config.config_loader import get_project_dir
from config.logger import setup_logging
from core.agent import chat_history
from core.agent.session_store import SessionStore

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

CLAUDE_CLI_ENTRY = "ClaudeCLI"
GROK_BOT_PREFIX = "GrokBot_"  # 每个 Grok Bot 一个条目：GrokBot_<Bot 名>，由管理后台生成
# 第一期的配置里称呼的键是 "claude"，指的就是本机 Claude Code
LEGACY_KEYS = {"claude": CLAUDE_CLI_ENTRY}

_pool = {}  # 条目名 -> (创建时的配置, 实例)
_pool_lock = threading.Lock()
_store = None


def store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore(Path(get_project_dir()) / "data" / "agent_sessions.json", "last_model")
    return _store


def entry_name(key: str) -> str:
    return LEGACY_KEYS.get(key, key)


def _server_config(conn: "ConnectionHandler") -> dict:
    server = getattr(conn, "server", None)
    return getattr(server, "config", None) or getattr(conn, "common_config", None) or conn.config


def default_name(conn: "ConnectionHandler") -> Optional[str]:
    return (_server_config(conn).get("selected_module") or {}).get("LLM")


def _placeholder(value) -> bool:
    return value is None or str(value).strip() == "" or "你" in str(value)


def model_ready(config: dict, name: str) -> bool:
    """配置里有这个条目，而且填好了：API 条目要有模型名和接口地址，有 api_key 字段时不能是占位文字。"""
    entry = (config.get("LLM") or {}).get(name)
    if not isinstance(entry, dict):
        return False
    kind = entry.get("type", name)
    if kind == "openai":
        if _placeholder(entry.get("model_name")) or _placeholder(entry.get("base_url") or entry.get("url")):
            return False
        if "api_key" in entry and _placeholder(entry.get("api_key")) and name != "CustomOpenAILLM":
            return False
    if kind == "grok_bot" and not str(entry.get("bot_id") or "").strip():
        return False
    return True


NOT_READY_REPLY = "还没配置大模型，请在电脑上打开网关后台填好 API Key。"


def not_ready_reply(conn: "ConnectionHandler") -> Optional[str]:
    """这个连接正在用的模型没填好（刚刷机、Key 还是占位文字）时，直接回给用户的话；填好了返回 None。
    不拦的话会拿空 Key 去请求，等出错才说"现在有点忙"。不知道在用哪个模型时不拦。"""
    st = _state(conn)
    name = st["current"] or default_name(conn)
    if not name:
        return None
    config = _server_config(conn)
    if not model_ready(config, name):
        default = default_name(conn)
        if default and default != name and model_ready(config, default):
            # 在后台另选了一个模型、配好后设成默认（没给原来那个填 Key）：在线的连接改用它
            try:
                if switch(conn, default, remember=False):
                    st.pop("not_ready", None)
                    return None
            except Exception as e:
                logger.bind(tag=TAG).error(f"切到默认模型 {default} 失败: {e}")
        st["not_ready"] = True
        return NOT_READY_REPLY
    if st.pop("not_ready", False):
        # 用户听到提示后在后台填好了，连接还没断：手上的实例是拿占位 Key 建的，换成按新配置建的
        try:
            instance = get_instance(conn, name)
            if instance is not conn.llm:
                _attach(conn, instance)
                conn.llm = instance
                logger.bind(tag=TAG).info(f"设备 {_device_key(conn)} 的模型 {name} 刚配置好，换用新实例")
        except Exception as e:
            st["not_ready"] = True  # 下一轮再试
            logger.bind(tag=TAG).error(f"模型 {name} 换用新实例失败，这一轮照旧请求: {e}")
    return None


INTENT_MODES = ("nointent", "function_call")
AGENT_TYPES = ("claude_cli", "grok_bot")  # 自己处理工具和对话的 Agent：不走 function_call


def _entry_type(config: dict, section: str, name: Optional[str]) -> Optional[str]:
    entry = (config.get(section) or {}).get(name)
    return entry.get("type", name) if isinstance(entry, dict) else None


def intent_mode(config: dict, name: str) -> Optional[str]:
    """一个连接用 name 这个模型时意图识别该用哪种，规则同后台切默认模型时的联动（pages.desired_intent）。

    - 用的就是后台默认模型：和后台的设置一致；
    - Claude Code、Grok Bot：nointent；
    - 其他 API 模型：后台在 API 模式下选过的，没选过时用 function_call。
    后台的意图识别不是这两种（如 intent_llm）时返回 None：不按设备改。
    """
    selected = _entry_type(config, "Intent", (config.get("selected_module") or {}).get("Intent"))
    if selected not in INTENT_MODES:
        return None
    if name == (config.get("selected_module") or {}).get("LLM"):
        return selected
    if _entry_type(config, "LLM", name) in AGENT_TYPES:
        return "nointent"
    pref = (config.get("admin") or {}).get("api_intent")
    if pref in INTENT_MODES:
        return pref
    return "function_call"


def _intent_entry(config: dict, mode: str) -> Optional[str]:
    """意图识别模式对应的条目名（上游里条目名和类型同名，用户改过名时按类型找）。"""
    intents = config.get("Intent") or {}
    if _entry_type(config, "Intent", mode) == mode:
        return mode
    return next((n for n, e in intents.items() if isinstance(e, dict) and e.get("type") == mode), None)


def _apply_intent(conn: "ConnectionHandler", name: str, live: bool = True) -> None:
    """让这个连接的意图识别跟着它的模型走。只改连接自己的配置副本（conn.config），不动全局设置。

    live=False：连接还在建立，组件还没初始化，改了配置后 _initialize_intent 会照着建好；
    live=True：已经在对话中，同时改 intent_type，第一次用上插件时补建工具处理器和工具调用示例。
    """
    mode = intent_mode(_server_config(conn), name)
    entry = _intent_entry(conn.config, mode) if mode else None
    if entry is None:
        return
    conn.config.setdefault("selected_module", {})["Intent"] = entry
    if not live or getattr(conn, "intent_type", None) == mode:
        return
    conn.intent_type = mode
    if mode == "function_call":
        conn.load_function_plugin = True
        if getattr(conn, "func_handler", None) is None:
            from core.providers.tools.unified_tool_handler import UnifiedToolHandler

            conn.func_handler = UnifiedToolHandler(conn)
            loop = getattr(conn, "loop", None)
            if loop is not None:
                asyncio.run_coroutine_threadsafe(conn.func_handler._initialize(), loop)
        if not any(m.is_temporary for m in conn.dialogue.dialogue):
            conn._inject_tool_call_fewshot()
    logger.bind(tag=TAG).info(f"设备 {_device_key(conn)} 的意图识别跟着模型改为 {mode}")


def model_label(name: str) -> str:
    from core.admin import catalog

    if name == CLAUDE_CLI_ENTRY:
        return "Claude"
    if name.startswith(GROK_BOT_PREFIX):
        return name[len(GROK_BOT_PREFIX):] or "Grok Bot"
    vendor = catalog.VENDOR_BY_ENTRY.get(name)
    return vendor["label"] if vendor else name


def _retire(instance) -> None:
    if callable(getattr(instance, "shutdown", None)):
        try:
            instance.shutdown()
        except Exception as e:
            logger.bind(tag=TAG).warning(f"旧模型实例退役出错: {e}")


def get_instance(conn: "ConnectionHandler", name: str):
    """取这个条目的模型实例：后台默认模型用 ws_server._llm，其他的从缓存取，配置变了就重建。"""
    config = _server_config(conn)
    server = getattr(conn, "server", None)
    if name == default_name(conn) and getattr(server, "_llm", None) is not None:
        return server._llm
    entry = copy.deepcopy((config.get("LLM") or {}).get(name))
    with _pool_lock:
        cached = _pool.get(name)
        if cached and cached[0] == entry:
            return cached[1]
    from core.utils import llm as llm_utils

    instance = llm_utils.create_instance(entry.get("type", name), entry)
    with _pool_lock:
        old = _pool.get(name)
        _pool[name] = (entry, instance)
    if old:
        logger.bind(tag=TAG).info(f"模型 {name} 的配置有变化，旧实例退役")
        _retire(old[1])
    logger.bind(tag=TAG).info(f"已创建模型实例 {name}")
    return instance


def _state(conn: "ConnectionHandler") -> dict:
    st = getattr(conn, "hub_route", None)
    if st is None:
        st = {"current": None, "used": [], "dialogues": {}}
        conn.hub_route = st
    return st


def _attach(conn: "ConnectionHandler", instance) -> None:
    """这个连接第一次用到某个实例时，通知它有新连接（claude_cli 按设备管理进程和会话）。"""
    st = _state(conn)
    if any(x is instance for x in st["used"]):
        return
    st["used"].append(instance)
    if callable(getattr(instance, "on_conn_open", None)):
        instance.on_conn_open(conn)


def _device_key(conn: "ConnectionHandler") -> str:
    return getattr(conn, "device_id", None) or conn.session_id


def remembered(conn: "ConnectionHandler") -> Optional[str]:
    record = store().get(_device_key(conn)) or {}
    return record.get("model")


def _dialogue_for(conn: "ConnectionHandler", name: str):
    """这个模型在本连接里的对话记录。第一次切过去时新建，带上系统提示和工具调用的示例，不带聊天内容。"""
    from core.utils.dialogue import Dialogue

    st = _state(conn)
    dialogue = st["dialogues"].get(name)
    if dialogue is None:
        dialogue = Dialogue()
        dialogue.dialogue = [m for m in conn.dialogue.dialogue if m.role == "system" or m.is_temporary]
        chat_history.load(conn, dialogue, name)
        st["dialogues"][name] = dialogue
    prompt = getattr(conn, "prompt", None)
    if prompt:
        dialogue.update_system_message(prompt)
    return dialogue


def on_conn_open(conn: "ConnectionHandler") -> None:
    """连接建立时：通知默认实例，再切到这台设备上次用的模型（还可用的话）。"""
    st = _state(conn)
    st["current"] = default_name(conn)
    chat_history.load(conn, conn.dialogue, st["current"])
    if conn.llm is not None:
        _attach(conn, conn.llm)
    name = remembered(conn)
    if name and name != st["current"] and model_ready(_server_config(conn), name):
        try:
            switch(conn, name, remember=False, live=False)
            logger.bind(tag=TAG).info(f"设备 {_device_key(conn)} 沿用上次的模型 {name}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"切换到上次的模型 {name} 失败，使用默认模型: {e}")
    if st["current"]:
        _apply_intent(conn, st["current"], live=False)
    # 上次断开后有 Agent 在外面做完了活：等问候播完念给用户
    from core.agent import call_names, reports

    reports.on_conn_open(conn)
    call_names.schedule_warm_up(conn)  # 提前准备好只喊称呼时回的"我在"


def switch(conn: "ConnectionHandler", name: str, remember: bool = True, live: bool = True) -> bool:
    """把这个连接切到指定模型，意图识别跟着改（live 见 _apply_intent）。模型没配置好时返回 False，不切换。"""
    if not model_ready(_server_config(conn), name):
        return False
    st = _state(conn)
    if st["current"] != name or conn.llm is None:
        instance = get_instance(conn, name)
        _attach(conn, instance)
        if st["current"]:
            st["dialogues"][st["current"]] = conn.dialogue
        conn.dialogue = _dialogue_for(conn, name)
        conn.llm = instance
        logger.bind(tag=TAG).info(f"设备 {_device_key(conn)} 切换模型：{st['current']} → {name}")
        st["current"] = name
        if live:
            _apply_intent(conn, name)
    elif name != default_name(conn):
        # 已经在用这个模型，但后台改过它的设置（实例要重建）：换成新实例，对话照旧。
        # 默认模型不在这里换：它由后台热加载重建，已连接的设备照旧用旧实例到重连（同 _apply 的说明）
        instance = get_instance(conn, name)
        if instance is not conn.llm:
            _attach(conn, instance)
            conn.llm = instance
            logger.bind(tag=TAG).info(f"设备 {_device_key(conn)} 的模型 {name} 配置有变化，换用新实例")
    if remember:
        store().put(_device_key(conn), {"model": name})
    return True


async def on_conn_open_async(conn: "ConnectionHandler") -> None:
    """在线程池里执行 on_conn_open：第一次创建模型实例可能要几百毫秒（claude_cli 会运行 claude --version）。"""
    await asyncio.get_running_loop().run_in_executor(None, on_conn_open, conn)


async def switch_async(conn: "ConnectionHandler", name: str) -> bool:
    return await asyncio.get_running_loop().run_in_executor(None, switch, conn, name)


def on_conn_close(conn: "ConnectionHandler") -> None:
    """连接关闭时，这个连接用过的每个模型都要通知到（claude_cli 结束对应的进程）。"""
    st = getattr(conn, "hub_route", None)
    instances = list(st["used"]) if st else []
    if conn.llm is not None and not any(x is conn.llm for x in instances):
        instances.append(conn.llm)
    for instance in instances:
        if callable(getattr(instance, "on_conn_close", None)):
            try:
                instance.on_conn_close(conn)
            except Exception as e:
                logger.bind(tag=TAG).error(f"LLM 连接关闭钩子出错: {e}")
