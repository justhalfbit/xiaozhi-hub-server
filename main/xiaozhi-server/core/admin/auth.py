"""管理后台的密码哈希与登录会话。"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

_SCRYPT = {"n": 2**14, "r": 8, "p": 1}
MIN_PASSWORD_LENGTH = 6


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_b64, digest_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
    except (ValueError, AttributeError):
        return False
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)
    return hmac.compare_digest(digest, expected)


class SessionStore:
    """内存中的登录会话；服务重启后需要重新登录（后台点了更新网关时除外，见 save/restore）。

    只存令牌的 SHA-256：交接文件被人看到也拿不到能用的令牌。"""

    def __init__(self, ttl_seconds: int = 12 * 3600):
        self.ttl = ttl_seconds
        self._sessions = {}  # 令牌的哈希 -> 过期时间

    @staticmethod
    def _key(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create(self) -> str:
        self._purge()
        token = secrets.token_urlsafe(32)
        self._sessions[self._key(token)] = time.time() + self.ttl
        return token

    def valid(self, token) -> bool:
        if not token:
            return False
        key = self._key(token)
        expiry = self._sessions.get(key)
        if expiry is None or expiry < time.time():
            self._sessions.pop(key, None)
            return False
        return True

    def revoke(self, token) -> None:
        if token:
            self._sessions.pop(self._key(token), None)

    def revoke_all(self) -> None:
        self._sessions.clear()

    def _purge(self) -> None:
        now = time.time()
        for key in [k for k, exp in self._sessions.items() if exp < now]:
            del self._sessions[key]

    def save(self, path: Path) -> None:
        """后台更新网关前调用：把登录会话交给重启后的新进程，更新完页面不用重新登录。"""
        self._purge()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"saved": time.time(), "sessions": self._sessions}, f)
        os.replace(tmp, path)

    def restore(self, path: Path, max_age: int = 1800) -> int:
        """启动时调用：读入 max_age 秒内交接过来的会话，读完就删掉文件。返回读入了几个。"""
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        finally:
            path.unlink(missing_ok=True)
        if not isinstance(data, dict) or time.time() - float(data.get("saved") or 0) > max_age:
            return 0
        now = time.time()
        sessions = {k: float(v) for k, v in (data.get("sessions") or {}).items()
                    if isinstance(k, str) and len(k) == 64 and isinstance(v, (int, float)) and v > now}
        self._sessions.update(sessions)
        return len(sessions)
