"""agent_sessions.json 的读写：几个命名空间各有各的实例，同一个文件要共用一把锁。"""

import threading
import time

from core.agent.session_store import SessionStore


def test_instances_of_same_file_share_lock(tmp_path):
    a = SessionStore(tmp_path / "agent_sessions.json", "claude_cli")
    b = SessionStore(tmp_path / "." / "agent_sessions.json", "last_model")
    other = SessionStore(tmp_path / "other.json", "claude_cli")
    assert a._lock is b._lock and a._lock is not other._lock


def test_concurrent_writers_in_different_namespaces_lose_nothing(tmp_path, monkeypatch):
    """claude_cli 记会话、呼名记上次的模型同时发生：两边的记录都要留下。"""
    write = SessionStore._write_all

    def slow_write(self, data):
        time.sleep(0.002)  # 拉长"读 → 写回"之间的空档，锁不共享时必然互相覆盖
        write(self, data)

    monkeypatch.setattr(SessionStore, "_write_all", slow_write)
    path = tmp_path / "agent_sessions.json"
    stores = [SessionStore(path, "claude_cli"), SessionStore(path, "last_model"), SessionStore(path, "grok_bot")]

    def work(store, n):
        for i in range(10):
            store.put(f"dev-{n}-{i}", {"model": store.namespace})

    threads = [threading.Thread(target=work, args=(s, n)) for n, s in enumerate(stores)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    data = stores[0]._read_all()
    assert {ns: len(data.get(ns, {})) for ns in ("claude_cli", "last_model", "grok_bot")} == \
        {"claude_cli": 10, "last_model": 10, "grok_bot": 10}
