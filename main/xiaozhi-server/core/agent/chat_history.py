"""聊天记录落盘：按设备和模型存进 data/chat.db（SQLite），网关重启、连接断开后都还在。

原来对话只在内存里（每个模型一份，见 llm_router），连接一断就没了；发给模型时整份全发，不截断。现在：

- 每轮结束、回复被打断改写（reply_progress）、连接关闭时，把对话里新增和改过的消息同步进库（sync）。
  系统提示、工具调用示例（is_temporary）不存；"[系统提示] 已达到最大工具调用次数"这种网关自己塞的话也不存；
- 普通 HTTP 模型：新连接（或切到这个模型）时从库里载入这台设备这个模型最近 max_messages 条（load），
  每次发给模型时再按"最近 max_messages 条 + 约 max_tokens 个 token"截一个窗口（select_window）。
  窗口只从用户的话开始切，工具调用、工具结果和后面的回复跟着它那句话走，不会拆开；
- 被窗口截掉的部分在后台增量总结成一段记忆（summarize），放进系统提示的 <memory>。
  只拿库里的真实对话去总结：上游 mem_local_short 把整份 dialogue 交给模型，工具调用示例（假对话）会混进记忆；
- Claude Code、Grok Bot 的上下文在它们自己的会话里（网关只发最后一句），这里只存记录给屏幕显示，
  不载入、不截断、不总结；
- 清除（clear）：删库里这台设备的记录和记忆、清在线连接的对话，Claude Code 删掉会话映射并关掉进程（下次新开会话），
  Grok Bot 调云端的"重置对话"（Bot、它的记忆和文件都保留，对话从头开始）。

配置（config.yaml 的 chat_history）：enabled、max_messages、max_tokens、summary、summary_min、summary_chars、
summary_llm（总结用哪个 LLM 条目，空则用当前模型）、display_limit。
"""

import concurrent.futures
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from config.config_loader import get_project_dir
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

DEFAULTS = {
    "enabled": True,
    "max_messages": 30,     # 每次最多带最近这么多条（用户、助手、工具消息都算）
    "max_tokens": 8000,     # 再加一道 token 上限（估算值）
    "summary": True,        # 截掉的部分总结成记忆
    "summary_min": 6,       # 截掉、还没总结的攒到这么多条才总结一次，免得每轮都调一次模型
    "summary_chars": 500,   # 记忆最多多少字
    "summary_llm": "",      # 总结用哪个 LLM 条目；空 = 当前这个模型
    "display_limit": 100,   # 给屏幕显示时每个模型最多返回多少条
}
# 上下文在自己会话里的类型：只存记录给屏幕显示
AGENT_KINDS = {"claude_cli", "grok_bot"}
HINT_PREFIX = "[系统提示]"
CUT_MARK = "……"  # reply_progress 给被打断的回复加的结尾
SUMMARY_BATCH = 200  # 一次最多总结这么多条（第一次上线时库里可能已经有很长的记录）
SUMMARY_PROMPT = """你负责给一个语音助手维护关于用户的长期记忆。下面给你【已有记忆】和【较早的对话】，这些对话之后不会再发给助手。
请把两者合并成新的记忆：保留用户的个人信息、偏好、正在做的事、约定和待办、聊天中得出的关键结论；去掉寒暄、重复和已经过时的内容。
用简洁的中文要点，每条一行，总共不超过 {chars} 字。只根据给你的对话内容写，不要编造，不要解释，只输出记忆本身；没有值得记的就输出"无"。"""

_db = None
_db_lock = threading.Lock()
_sync_lock = threading.RLock()
_summarizing = set()  # 正在总结的 (device, model)
_summarizing_lock = threading.Lock()


# ---------- 配置 ----------

def settings(config: Optional[dict]) -> dict:
    merged = dict(DEFAULTS)
    merged.update((config or {}).get("chat_history") or {})
    return merged


def _conn_config(conn) -> dict:
    server = getattr(conn, "server", None)
    return getattr(server, "config", None) or getattr(conn, "config", None) or {}


def device_of(conn) -> Optional[str]:
    """只给带 device-id 的连接存：没有的话只有一次性的 session_id，存了也没人能再取回来。"""
    return getattr(conn, "device_id", None) or None


def model_kind(config: dict, name: str) -> str:
    entry = (config.get("LLM") or {}).get(name)
    return entry.get("type", name) if isinstance(entry, dict) else str(name or "")


def is_agent(config: dict, name: str) -> bool:
    return model_kind(config, name) in AGENT_KINDS


# ---------- 数据库 ----------

class ChatDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device TEXT NOT NULL,
                model TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                created REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_device_model ON messages (device, model, id);
            CREATE TABLE IF NOT EXISTS memories (
                device TEXT NOT NULL,
                model TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                upto INTEGER NOT NULL DEFAULT 0,
                updated REAL NOT NULL,
                PRIMARY KEY (device, model)
            );
            """
        )

    def _exec(self, sql: str, args=()):
        with self._lock:
            return self._conn.execute(sql, args)

    def insert(self, device, model, role, content, tool_calls=None, tool_call_id=None, created=None) -> int:
        cur = self._exec(
            "INSERT INTO messages (device, model, role, content, tool_calls, tool_call_id, created) VALUES (?,?,?,?,?,?,?)",
            (device, model, role, content, json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
             tool_call_id, created if created is not None else time.time()))
        return cur.lastrowid

    def update_content(self, msg_id: int, content) -> None:
        self._exec("UPDATE messages SET content=? WHERE id=?", (content, msg_id))

    def delete_ids(self, ids) -> None:
        ids = list(ids)
        if ids:
            self._exec(f"DELETE FROM messages WHERE id IN ({','.join('?' * len(ids))})", ids)

    def recent(self, device, model, limit: int) -> List[dict]:
        rows = self._exec(
            "SELECT id, role, content, tool_calls, tool_call_id, created FROM messages "
            "WHERE device=? AND model=? ORDER BY id DESC LIMIT ?", (device, model, int(limit))).fetchall()
        return [_row(r) for r in reversed(rows)]

    def after(self, device, model, after_id: int, limit: int = None) -> List[dict]:
        sql = ("SELECT id, role, content, tool_calls, tool_call_id, created FROM messages "
               "WHERE device=? AND model=? AND id>? ORDER BY id")
        args = [device, model, int(after_id)]
        if limit:
            sql += " LIMIT ?"
            args.append(int(limit))
        return [_row(r) for r in self._exec(sql, args).fetchall()]

    def exists(self, msg_id: int) -> bool:
        return self._exec("SELECT 1 FROM messages WHERE id=?", (int(msg_id),)).fetchone() is not None

    def models(self, device) -> List[str]:
        return [r[0] for r in self._exec("SELECT DISTINCT model FROM messages WHERE device=?", (device,)).fetchall()]

    def memory(self, device, model) -> dict:
        r = self._exec("SELECT summary, upto FROM memories WHERE device=? AND model=?", (device, model)).fetchone()
        return {"summary": r[0], "upto": r[1]} if r else {"summary": "", "upto": 0}

    def set_memory(self, device, model, summary: str, upto: int) -> None:
        self._exec(
            "INSERT INTO memories (device, model, summary, upto, updated) VALUES (?,?,?,?,?) "
            "ON CONFLICT (device, model) DO UPDATE SET summary=excluded.summary, upto=excluded.upto, updated=excluded.updated",
            (device, model, summary, int(upto), time.time()))

    def clear(self, device) -> int:
        n = self._exec("DELETE FROM messages WHERE device=?", (device,)).rowcount
        self._exec("DELETE FROM memories WHERE device=?", (device,))
        return n

    def clear_memory(self, device) -> None:
        """只清记忆：总结清空，但记住总结到哪了，截掉的旧对话不会再被总结回来。"""
        self._exec("UPDATE memories SET summary='', updated=? WHERE device=?", (time.time(), device))

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _row(r) -> dict:
    return {"id": r[0], "role": r[1], "content": r[2], "tool_calls": json.loads(r[3]) if r[3] else None,
            "tool_call_id": r[4], "created": r[5]}


def db() -> ChatDB:
    global _db
    with _db_lock:
        if _db is None:
            _db = ChatDB(Path(get_project_dir()) / "data" / "chat.db")
        return _db


def set_db(instance: Optional[ChatDB]) -> None:
    """测试用：换成临时库。"""
    global _db
    with _db_lock:
        _db = instance


# ---------- 截断窗口 ----------

def estimate_tokens(text) -> int:
    """粗估 token：中文等非 ASCII 字符按 1 个算，ASCII 按 4 个字符 1 个算。不同模型的分词不一样，只求量级对。"""
    if not text:
        return 0
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    wide = sum(1 for c in text if ord(c) > 127)
    return wide + (len(text) - wide + 3) // 4


def _msg_tokens(m) -> int:
    get = m.get if isinstance(m, dict) else lambda k: getattr(m, k, None)
    return estimate_tokens(get("content")) + estimate_tokens(get("tool_calls")) + 4


def select_window(messages: list, max_messages: int, max_tokens: int) -> int:
    """返回窗口从第几条开始：按"一句用户的话 + 它后面的回复、工具调用和结果"成组，从新往旧装，
    超过条数或 token 上限就停。最新的一组总是带上（哪怕它自己就超了）。messages 是 Message 或 dict。"""
    if not messages:
        return 0
    role = (lambda m: m.get("role")) if isinstance(messages[0], dict) else (lambda m: m.role)
    starts = [i for i, m in enumerate(messages) if role(m) == "user"]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    start, count, tokens = len(messages), 0, 0
    for s in reversed(starts):
        group = messages[s:start]
        n, t = len(group), sum(_msg_tokens(m) for m in group)
        if start < len(messages) and ((max_messages > 0 and count + n > max_messages)
                                      or (max_tokens > 0 and tokens + t > max_tokens)):
            break
        start, count, tokens = s, count + n, tokens + t
    # 窗口开头不能是助手或工具消息（截出来的残组）：往后挪到第一句用户的话
    while start < len(messages) and role(messages[start]) != "user" and start != starts[-1]:
        nxt = next((s for s in starts if s > start), None)
        if nxt is None:
            break
        start = nxt
    return start


# ---------- 载入、同步 ----------

def _storable(m) -> bool:
    if m.role == "system" or m.is_temporary:
        return False
    if m.role == "user" and isinstance(m.content, str) and m.content.startswith(HINT_PREFIX):
        return False
    return True


def load(conn, dialogue, name: str) -> None:
    """把这台设备这个模型的记录接到对话上（新连接、第一次切到这个模型时调用）。"""
    from core.utils.dialogue import Message

    config = _conn_config(conn)
    cfg = settings(config)
    device = device_of(conn)
    if not cfg["enabled"] or not device or not name:
        return
    if getattr(dialogue, "chat_key", None) == (device, name):
        return
    dialogue.chat_key = (device, name)
    dialogue.chat_ids = set()
    if is_agent(config, name):
        return  # 只存不载入
    dialogue.window = (int(cfg["max_messages"]), int(cfg["max_tokens"]))
    try:
        rows = db().recent(device, name, max(int(cfg["max_messages"]), 1))
    except Exception as e:
        logger.bind(tag=TAG).error(f"读取聊天记录失败: {e}")
        return
    loaded = []
    for r in rows:
        m = Message(role=r["role"], content=r["content"], tool_calls=r["tool_calls"], tool_call_id=r["tool_call_id"])
        m.db_id, m.db_content = r["id"], r["content"]
        dialogue.chat_ids.add(r["id"])
        loaded.append(m)
    # 接在系统提示和示例后面（顺序其实无所谓：发给模型时系统提示、示例、对话分三段取）
    dialogue.dialogue[:] = [m for m in dialogue.dialogue if not _storable(m)] + loaded + \
        [m for m in dialogue.dialogue if _storable(m)]
    if loaded:
        logger.bind(tag=TAG).info(f"设备 {device} 模型 {name}：载入 {len(loaded)} 条聊天记录")


def sync(conn, dialogue=None) -> None:
    """把对话里新增、改过、删掉的消息同步进库。"""
    dialogue = dialogue or getattr(conn, "dialogue", None)
    key = getattr(dialogue, "chat_key", None)
    if not key:
        return
    device, name = key
    with _sync_lock:
        try:
            store = db()
            present = set()
            for m in list(dialogue.dialogue):
                if not _storable(m):
                    continue
                msg_id = getattr(m, "db_id", None)
                if msg_id is None:
                    m.db_id = store.insert(device, name, m.role, m.content, m.tool_calls, m.tool_call_id)
                    m.db_content = m.content
                    dialogue.chat_ids.add(m.db_id)
                elif m.content != getattr(m, "db_content", None):
                    store.update_content(msg_id, m.content)
                    m.db_content = m.content
                present.add(m.db_id)
            gone = dialogue.chat_ids - present
            if gone:
                store.delete_ids(gone)
                dialogue.chat_ids -= gone
        except Exception as e:
            logger.bind(tag=TAG).error(f"保存聊天记录失败: {e}")


def sync_all(conn) -> None:
    """连接关闭时：这个连接里每个模型的对话都同步一遍。"""
    st = getattr(conn, "hub_route", None) or {}
    seen = set()
    for d in [getattr(conn, "dialogue", None)] + list((st.get("dialogues") or {}).values()):
        if d is not None and id(d) not in seen:
            seen.add(id(d))
            sync(conn, d)


# ---------- 记忆 ----------

def memory_for(conn) -> str:
    key = getattr(getattr(conn, "dialogue", None), "chat_key", None)
    if not key or not settings(_conn_config(conn))["summary"]:
        return ""
    try:
        return db().memory(*key)["summary"]
    except Exception as e:
        logger.bind(tag=TAG).error(f"读取记忆失败: {e}")
        return ""


def _transcript(rows: List[dict]) -> str:
    lines = []
    for r in rows:
        content = _display_text(r["content"]) if r["role"] in ("user", "assistant") else (r["content"] or "")
        if r["role"] == "user":
            lines.append(f"用户：{content}")
        elif r["role"] == "assistant" and r["tool_calls"]:
            names = "、".join(((c.get("function") or {}).get("name") or "") for c in r["tool_calls"] if isinstance(c, dict))
            lines.append(f"助手调用了工具：{names}")
        elif r["role"] == "assistant":
            lines.append(f"助手：{content}")
        elif r["role"] == "tool":
            lines.append(f"工具结果：{content[:300]}")
    return "\n".join(lines)


def pending_summary(device: str, name: str, cfg: dict):
    """被窗口截掉、还没总结的消息：(rows, 已有记忆)。"""
    store = db()
    mem = store.memory(device, name)
    rows = store.after(device, name, mem["upto"])
    start = select_window(rows, int(cfg["max_messages"]), int(cfg["max_tokens"]))
    return rows[:start][:SUMMARY_BATCH], mem


def summarize_now(llm, device: str, name: str, cfg: dict) -> bool:
    """同步执行一次总结。返回有没有更新记忆。"""
    rows, mem = pending_summary(device, name, cfg)
    if not rows:
        return False
    prompt = (f"【已有记忆】\n{mem['summary'] or '无'}\n\n【较早的对话】\n{_transcript(rows)}")
    text = llm.response_no_stream(SUMMARY_PROMPT.format(chars=cfg["summary_chars"]), prompt)
    text = (text or "").strip()
    if not text:
        return False
    if text in ("无", "无。"):
        text = mem["summary"]
    limit = int(cfg["summary_chars"]) * 2
    store = db()
    if store.memory(device, name)["upto"] != mem["upto"]:
        return False  # 总结期间被清除了
    if not store.exists(rows[-1]["id"]):
        return False  # 这批记录已被清除
    store.set_memory(device, name, text[:limit], rows[-1]["id"])
    logger.bind(tag=TAG).info(f"设备 {device} 模型 {name}：总结了 {len(rows)} 条较早的对话，记忆 {len(text)} 字")
    return True


def maybe_summarize(conn) -> None:
    """一轮结束后：截掉的旧对话攒够了，就在后台总结一次。"""
    key = getattr(getattr(conn, "dialogue", None), "chat_key", None)
    config = _conn_config(conn)
    cfg = settings(config)
    if not key or not cfg["summary"] or is_agent(config, key[1]):
        return
    with _summarizing_lock:
        if key in _summarizing:
            return
        _summarizing.add(key)

    def run():
        try:
            rows, _ = pending_summary(key[0], key[1], cfg)
            if len(rows) < int(cfg["summary_min"]):
                return
            llm = _summary_llm(conn, key[1], cfg)
            if llm is not None:
                summarize_now(llm, key[0], key[1], cfg)
        except Exception as e:
            logger.bind(tag=TAG).error(f"总结聊天记录失败: {e}")
        finally:
            with _summarizing_lock:
                _summarizing.discard(key)

    threading.Thread(target=run, name="chat-summary", daemon=True).start()


def _summary_llm(conn, name: str, cfg: dict):
    from core.agent import llm_router

    target = cfg.get("summary_llm") or name
    config = _conn_config(conn)
    if is_agent(config, target) or not llm_router.model_ready(config, target):
        return None
    return llm_router.get_instance(conn, target)


# ---------- 显示 ----------

def _display_text(content) -> str:
    """用户的话可能是带说话人的 JSON（{"speaker", "content"}），只取正文。"""
    text = content or ""
    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "content" in data:
                return str(data["content"])
        except ValueError:
            pass
    return text


def history(device: str, limit: int, models: List[str] = None) -> Dict[str, list]:
    """给屏幕显示的记录：{模型: [{id, role, text, time, interrupted}]}，只有用户的话和助手的回复。"""
    from core.utils.textUtils import check_emoji

    store = db()
    out = {}
    for name in models or store.models(device):
        items = []
        # 多取一些：工具消息不显示
        for r in store.recent(device, name, limit * 3):
            if r["role"] not in ("user", "assistant") or r["tool_calls"]:
                continue
            text = _display_text(r["content"])
            if not text or text.startswith(HINT_PREFIX):
                continue
            interrupted = r["role"] == "assistant" and text.endswith(CUT_MARK)
            if r["role"] == "assistant":
                text = check_emoji(text).strip()
                if interrupted and text != CUT_MARK:
                    text = text[:-len(CUT_MARK)]
            items.append({"id": r["id"], "role": "user" if r["role"] == "user" else "bot", "text": text,
                          "time": r["created"], "interrupted": interrupted})
        if items:
            out[name] = items[-limit:]
    return out


# ---------- 清除 ----------

def _clear_dialogue(dialogue) -> None:
    with _sync_lock:
        dialogue.dialogue[:] = [m for m in dialogue.dialogue if not _storable(m)]
        if getattr(dialogue, "chat_key", None):
            dialogue.chat_ids = set()


def clear_dialogues(connections) -> None:
    """清在线连接里所有模型的对话（留系统提示和示例），Claude Code / Grok Bot 关掉会话、下次新开。"""
    for conn in connections:
        st = getattr(conn, "hub_route", None) or {}
        dialogues = [getattr(conn, "dialogue", None)] + list((st.get("dialogues") or {}).values())
        for d in dialogues:
            if d is not None:
                _clear_dialogue(d)
        for instance in st.get("used") or []:
            if callable(getattr(instance, "reset_session", None)):
                try:
                    instance.reset_session(conn)
                except Exception as e:
                    logger.bind(tag=TAG).error(f"重置 {type(instance).__name__} 会话失败: {e}")
        reply_turns = getattr(conn, "reply_turns", None)
        if reply_turns:
            reply_turns.clear()


def clear(device: str, config: dict, memory_only: bool = False, connections=None, grok_wait: float = 8) -> dict:
    """清除这台设备的聊天记录（所有模型）。connections：在线连接列表（core.admin.connections 的连接对象）。

    返回 {"deleted": 条数, "grok": [{"entry", "label", "reset": bool, "pending"?, "error"?}]}：
    这台设备和哪个 Grok Bot 聊过，就让它在云端重置对话（pending = 等了 grok_wait 秒还没做完，在后台接着做）。"""
    from core.agent import llm_router
    from core.agent.session_store import SessionStore

    store = db()
    if memory_only:
        store.clear_memory(device)
        logger.bind(tag=TAG).info(f"设备 {device}：已清除记忆")
        return {"deleted": 0, "grok": []}
    talked = set(store.models(device))
    deleted = store.clear(device)
    clear_dialogues(connections or [])
    # Claude Code：设备不在线时也要删掉映射，下次新开会话
    SessionStore(Path(get_project_dir()) / "data" / "agent_sessions.json", "claude_cli").forget(device)
    grok, futures = [], []
    for name in sorted(talked):
        if model_kind(config, name) != "grok_bot":
            continue
        item = {"entry": name, "label": llm_router.model_label(name), "reset": False}
        grok.append(item)
        try:
            futures.append((item, reset_grok(config, name)))
        except Exception as e:
            item["error"] = str(e)[:200]
    # 云端电脑休眠时要先唤醒，可能很久：最多等 grok_wait 秒，没做完的在后台接着做
    deadline = time.monotonic() + grok_wait
    for item, fut in futures:
        try:
            fut.result(max(deadline - time.monotonic(), 0))
            item["reset"] = True
        except concurrent.futures.TimeoutError:
            item["pending"] = True
        except Exception as e:
            item["error"] = str(e)[:200] or type(e).__name__
    for item in grok:
        if item.get("error"):
            logger.bind(tag=TAG).error(f"重置 Grok Bot {item['entry']} 的云端对话失败: {item['error']}")
    logger.bind(tag=TAG).info(f"设备 {device}：已清除 {deleted} 条聊天记录和记忆" +
                              (f"，Grok Bot：{grok}" if grok else ""))
    return {"deleted": deleted, "grok": grok}


def reset_grok(config: dict, name: str) -> "concurrent.futures.Future":
    """让这个 Grok Bot 在云端重置对话（桌面 App 的"重置对话"，方法 startFreshConversation：
    Bot、它的记忆、文件、例行任务都保留，对话历史从头开始）。返回 Future。"""
    import asyncio
    import os

    from core.agent.grokbot_client import default_credentials_path, get_hub
    from core.providers.llm.grok_bot.grok_bot import shared_loop

    entry = (config.get("LLM") or {}).get(name) or {}
    bot_id = str(entry.get("bot_id") or "").strip()
    if not bot_id:
        raise ValueError("没有 bot_id")
    raw_path = str(entry.get("credentials_path") or "").strip()
    cred = Path(os.path.expanduser(raw_path)) if raw_path else default_credentials_path()
    backend = str(entry.get("backend_url") or "").strip() or None

    async def go():
        hub = get_hub(cred, logger=logger.bind(tag=TAG), backend=backend)
        await hub.client.call("startFreshConversation", {"id": bot_id})

    return asyncio.run_coroutine_threadsafe(go(), shared_loop())
