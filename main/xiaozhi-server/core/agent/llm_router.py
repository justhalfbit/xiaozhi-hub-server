"""多模型路由：按称呼把对话交给不同的大模型（LLM 配置里的条目，如 ClaudeCLI、DeepSeekLLM）。

- 每个连接自己决定用哪个模型：切换时只换这个连接的 conn.llm，不影响其他设备；
- 各聊各的：每个模型在这个连接里有自己的一份对话记录（conn.dialogue 跟着换），切换时不带上下文过去；
  claude_cli 本来就按设备保存会话；API 模型的记录按设备和模型存进 data/chat.db，新连接时载入最近一段（chat_history）；
- 每台设备上次用的模型存到 data/agent_sessions.json，重连、服务重启后仍有效；
- 模型实例按条目名缓存、所有连接共用（和原来共享 ws_server._llm 一样）。后台默认模型直接用 ws_server._llm；
  其他模型第一次被叫到时才创建。条目配置变了（后台保存后热加载）就让旧实例退役，下次叫到时重建，
  在线连接继续用旧实例，重连后用新的；
- 切到别的模型后，原来那个模型不会马上释放：claude_cli 的进程等空闲超时再关，方便切回来；
  连接关闭时，这个连接用过的每个模型都会收到 on_conn_close。
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
            switch(conn, name, remember=False)
            logger.bind(tag=TAG).info(f"设备 {_device_key(conn)} 沿用上次的模型 {name}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"切换到上次的模型 {name} 失败，使用默认模型: {e}")
    # 上次断开后有 Agent 在外面做完了活：等问候播完念给用户
    from core.agent import reports

    reports.on_conn_open(conn)


def switch(conn: "ConnectionHandler", name: str, remember: bool = True) -> bool:
    """把这个连接切到指定模型。模型没配置好时返回 False，不切换。"""
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
