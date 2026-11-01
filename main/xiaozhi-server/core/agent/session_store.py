"""记录"设备 → Agent 会话 ID → 最后活跃时间"，保存到 JSON 文件，服务重启后仍可接续。"""

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

# 同一个文件的所有实例共用一把锁：claude_cli、各 Grok Bot、last_model 各建各的实例，
# 都是"读整个文件 → 改自己那一块 → 整个写回"，锁不共享时并发写会互相覆盖
_locks = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = os.path.abspath(path)
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


class SessionStore:
    def __init__(self, path: Path, namespace: str):
        self.path = Path(path)
        self.namespace = namespace
        self._lock = _lock_for(self.path)

    def _read_all(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_all(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".agent_sessions.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def get(self, device_key: str) -> Optional[dict]:
        with self._lock:
            return self._read_all().get(self.namespace, {}).get(device_key)

    def touch(self, device_key: str, session_id: str, now: float = None) -> None:
        with self._lock:
            data = self._read_all()
            data.setdefault(self.namespace, {})[device_key] = {
                "session_id": session_id,
                "last_active": now if now is not None else time.time(),
            }
            self._write_all(data)

    def put(self, device_key: str, record: dict, now: float = None) -> None:
        """保存任意记录（如设备上次用的模型），自动加上 last_active。"""
        with self._lock:
            data = self._read_all()
            data.setdefault(self.namespace, {})[device_key] = {
                **record, "last_active": now if now is not None else time.time()}
            self._write_all(data)

    def forget(self, device_key: str) -> None:
        with self._lock:
            data = self._read_all()
            if data.get(self.namespace, {}).pop(device_key, None) is not None:
                self._write_all(data)


def choose_resume(record: Optional[dict], wants_resume: bool, window_minutes: float, now: float = None):
    """决定这次唤醒是接续还是新开。返回 (resume_id 或 None, 原因)。

    - 用户说了"接着上次聊"：有记录就接续，不看时间；
    - 否则距上次对话不超过接续窗口：自动接续；
    - 其他情况：新开会话。
    """
    now = now if now is not None else time.time()
    if not record or not record.get("session_id"):
        return None, "no_record" if wants_resume else "new"
    if wants_resume:
        return record["session_id"], "keyword"
    if window_minutes and now - float(record.get("last_active", 0)) <= window_minutes * 60:
        return record["session_id"], "window"
    return None, "expired"
