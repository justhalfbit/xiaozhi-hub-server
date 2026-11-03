"""在线设备连接登记表，供管理后台查看设备、给设备发调试消息（例如表情测试）。

原项目不保存连接列表，这里由 ConnectionHandler 在建立 / 关闭连接时登记和注销。
"""

import threading
import time

_lock = threading.Lock()
_connections = {}  # session_id -> (conn, 连接时间)


def register(conn) -> None:
    with _lock:
        _connections[conn.session_id] = (conn, time.time())


def unregister(conn) -> None:
    with _lock:
        _connections.pop(getattr(conn, "session_id", None), None)


def get(session_id: str):
    with _lock:
        item = _connections.get(session_id)
    return item[0] if item else None


def all_connections() -> list:
    with _lock:
        return [conn for conn, _ in _connections.values()]


def snapshot() -> list:
    with _lock:
        items = list(_connections.items())
    return [
        {
            "session_id": sid,
            "device_id": getattr(conn, "device_id", None),
            "ip": getattr(conn, "client_ip", None),
            "connected_at": int(since),
            "model": (getattr(conn, "hub_route", None) or {}).get("current"),  # 呼名切到的模型
        }
        for sid, (conn, since) in items
    ]
