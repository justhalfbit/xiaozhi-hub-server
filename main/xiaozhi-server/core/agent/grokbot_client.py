"""Grok Bot 云端网关客户端（非官方接口）。

Grok Bot（xAI 与 Cursor 联合推出的云端 AI 同事）没有公开 API。每个账号在云端有一台 Linux 电脑，
Bot 跑在里面，电脑内 1340 端口有一个 HTTP + SSE 的网关；桌面 App 和手机 App 本身就是这个网关的客户端。
这里做的事就是"扮演一个 App"：

1. 用 Cursor 账号登录（PKCE：本地生成 verifier，浏览器里批准，再轮询拿 token）；
2. 向 api2.cursor.sh 的 EnsureSandBox 要到云端电脑的网关地址和令牌（休眠时先唤醒）；
3. `POST {网关}/api/<method>` 发指令，`GET {网关}/events` 收 SSE 推送。

接口随桌面 App 版本变化，2026-10-21 对 App 0.72.0 实测可用。已知坑：事件流偶尔"假活"——连接保持、
每 15 秒 `:ping` 照常、约 57 秒服务端断一次，但不推事件，所以会话层必须有补拉和轮询兜底（见 grokbot_session）。

凭证只存在本机 data/grokbot_credentials.json（0600），永远不打印、不经接口返回。
"""

import asyncio
import base64
import hashlib
import json
import os
import secrets
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlencode, urlparse

import aiohttp

from config.config_loader import get_project_dir
from core.agent.agent_session import ErrorKind

BACKEND = "https://api2.cursor.sh"
WEBSITE = "https://cursor.com"
CLIENT_ID = "KbZUR41cY7W6zRSdpSUJ7I7mLYBKOCmB"  # 桌面 App 的公开 OAuth client_id
CLIENT_VERSION = "0.75.3"  # 发给服务端的客户端版本号，跟着本机桌面 App 的版本走；实测服务端不校验客户端身份
GATEWAY_SUFFIXES = (".cursor.sh", ".cursor.com", ".cursorvm.com")
REFRESH_AHEAD = 300  # access token 到期前这么多秒就续期
WAKE_TIMEOUT = 180  # 云端电脑休眠时最多等多久醒来
WAKE_POLL = 3  # 唤醒期间每隔多久问一次就绪了没
ENSURE_TIMEOUT = 60  # 单次 EnsureSandBox 最多等多久；冷启动时会超时，WAKE_TIMEOUT 之内重试
# 刚从休眠里叫醒时 EnsureSandBox 已经报 RUNNING，里面的网关却可能一两分钟不响应（2026-11-01 真机：休眠 7 小时后
# 叫醒，读对话尾部卡住一分多钟）。设备日志统计：醒着时 EnsureSandBox 1–3 秒返回，刚醒的 4.6–10 秒，按这个分界判断
WAKE_SLOW = 4  # EnsureSandBox（含重试）用了这么久，就当云端电脑是刚叫醒的
WAKE_PROBE_MAX = 45  # 刚叫醒时最多花这么久（从醒来算起）确认网关能响应
WAKE_PROBE_TIMEOUT = 5  # 确认网关时单次请求的超时
WAKE_PROBE_PAUSE = 2  # 确认网关时两次请求的间隔
STREAM_READ_TIMEOUT = 45  # 事件流每 15 秒一个 ping，45 秒没字节就当断了
STREAM_BACKOFF = (1, 2, 4, 8, 10)
AUTH_RETRY = 60  # 登录失效时事件流多久再试一次
# 查登录状态的 /auth/poll 是长轮询：还没批准时服务端把请求挂着不回，超时只说明"还在等批准"
LOGIN_POLL_TIMEOUT = 60


class GrokBotError(Exception):
    def __init__(self, kind: ErrorKind, detail: str = ""):
        super().__init__(detail or kind.value)
        self.kind = kind
        self.detail = detail


def default_credentials_path() -> Path:
    return Path(get_project_dir()) / "data" / "grokbot_credentials.json"


async def _sleep(seconds: float) -> None:
    """不用 asyncio.sleep：测试环境会把它改成不等待的桩，重试循环会空转。"""
    if seconds <= 0:
        return
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    handle = loop.call_later(seconds, lambda: fut.done() or fut.set_result(None))
    try:
        await fut
    finally:
        handle.cancel()


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def jwt_exp(token: str) -> float:
    """JWT 的到期时间（秒）；解析不了返回 0。"""
    try:
        part = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
        return float(claims.get("exp", 0))
    except (IndexError, ValueError, TypeError):
        return 0.0


# ---------- 凭证 ----------

class Credentials:
    """本机保存的登录凭证。文件变了（后台重新登录）会自动重读。"""

    def __init__(self, path: Path = None):
        self.path = Path(path) if path else default_credentials_path()
        self._data: dict = {}
        self._mtime = None
        self.reload()

    def reload(self) -> None:
        try:
            mtime = os.stat(self.path).st_mtime_ns
        except FileNotFoundError:
            self._data, self._mtime = {}, None
            return
        if mtime == self._mtime:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            data = {}
        self._data = data if isinstance(data, dict) else {}
        self._mtime = mtime

    def save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".grokbot.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self._data = dict(data)
        self._mtime = os.stat(self.path).st_mtime_ns

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self._data, self._mtime = {}, None

    @property
    def access_token(self) -> str:
        return str(self._data.get("accessToken") or "")

    @property
    def refresh_token(self) -> str:
        return str(self._data.get("refreshToken") or "")

    @property
    def team_id(self) -> Optional[str]:
        value = self._data.get("teamId")
        return None if value in (None, "") else str(value)

    @property
    def needs_login(self) -> bool:
        """服务端要求重新登录（refresh token 失效）后标记，直到下一次登录成功。"""
        return bool(self._data.get("needsLogin"))

    @property
    def logged_in(self) -> bool:
        return bool(self.access_token) and not self.needs_login

    @property
    def expires_at(self) -> float:
        return jwt_exp(self.access_token)

    def mark_needs_login(self) -> None:
        self.reload()  # 别的进程可能刚写过新 token，别用内存里的旧数据把它盖掉
        if self._data:
            self.save({**self._data, "needsLogin": True})

    def status(self) -> dict:
        """给管理后台看的状态，不含任何令牌。"""
        self.reload()
        return {
            "logged_in": self.logged_in,
            "needs_login": self.needs_login,
            "expires_at": self.expires_at if self.access_token else 0,
            "has_refresh": bool(self.refresh_token),
            "path": str(self.path),
        }


@dataclass
class LoginFlow:
    uuid: str
    verifier: str
    url: str
    started_at: float = field(default_factory=time.time)


# ---------- 客户端 ----------

class GrokBotClient:
    """登录、续期、找云端电脑、调网关。所有方法都要在同一个事件循环里调用。"""

    def __init__(self, credentials: Credentials = None, backend: str = BACKEND, website: str = WEBSITE,
                 logger=None, client_version: str = CLIENT_VERSION):
        self.creds = credentials or Credentials()
        self.backend = backend.rstrip("/")
        self.website = website.rstrip("/")
        self.logger = logger
        self.client_version = client_version
        self.conn: Optional[dict] = None  # EnsureSandBox 的结果：gatewayUrl、gatewayToken、networkToken、runState
        self.woke_at: Optional[float] = None  # 云端电脑刚叫醒、网关还没确认能响应时，醒来的时刻（monotonic）
        self._session: Optional[aiohttp.ClientSession] = None
        self._refresh_lock: Optional[asyncio.Lock] = None
        self._ensure_lock: Optional[asyncio.Lock] = None

    # ----- 基础 -----

    def _log(self, level: str, msg: str) -> None:
        if self.logger:
            getattr(self.logger, level)(f"[grokbot] {msg}")

    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
        return self._session

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None and not session.closed:
            await session.close()

    async def _request(self, method: str, url: str, body=None, headers: dict = None, timeout: float = 30):
        """返回 (状态码, 响应头, 解析后的 JSON 或原文)。网络错误抛 GrokBotError(SERVER)，原始异常在 __cause__。"""
        try:
            async with self.session().request(
                    method, url, json=body, headers=headers or {}, allow_redirects=False,
                    timeout=aiohttp.ClientTimeout(total=timeout)) as res:
                raw = await res.read()
                try:
                    parsed = json.loads(raw) if raw else {}
                except ValueError:
                    parsed = raw.decode(errors="replace")[:300]
                return res.status, res.headers, parsed
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise GrokBotError(ErrorKind.SERVER, f"{method} {urlparse(url).path}: {e!r}") from e

    # ----- 登录 -----

    def start_login(self) -> LoginFlow:
        """生成登录地址。用户在浏览器里批准后，poll_login 才能拿到 token。地址里没有 verifier。"""
        verifier = _b64url(secrets.token_bytes(32))
        challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
        login_id = str(uuid.uuid4())
        url = self.website + "/loginDeepControl?" + urlencode({
            "challenge": challenge, "uuid": login_id, "mode": "login",
            "redirectTarget": "sand", "supportsSelectedTeamLogin": "true",
        })
        return LoginFlow(uuid=login_id, verifier=verifier, url=url)

    async def poll_login_once(self, flow: LoginFlow) -> Optional[bool]:
        """查一次登录状态：拿到 token 返回 True（已保存），还没批准返回 None，被拒绝返回 False。"""
        url = self.backend + "/auth/poll?" + urlencode({"uuid": flow.uuid, "verifier": flow.verifier})
        try:
            status, _, body = await self._request("GET", url, headers={"content-type": "application/json"},
                                                  timeout=LOGIN_POLL_TIMEOUT)
        except GrokBotError as e:
            if isinstance(e.__cause__, asyncio.TimeoutError):  # 长轮询挂满了：还没批准
                return None
            raise
        if status == 404:
            return None
        if 200 <= status < 300 and isinstance(body, dict) and body.get("accessToken"):
            data = {"accessToken": body["accessToken"], "refreshToken": body.get("refreshToken", "")}
            team = body.get("selectedTeamId")
            if team not in (None, ""):
                data["teamId"] = str(team)
            self.creds.save(data)
            self._log("info", "登录成功，access token 到期：" + time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(self.creds.expires_at)))
            return True
        if status == 403:
            return False
        raise GrokBotError(ErrorKind.SERVER, f"auth/poll HTTP {status} {body}")

    async def poll_login(self, flow: LoginFlow, timeout: float = 900) -> bool:
        """轮询直到登录成功、被拒绝或超时。间隔 1 秒起每次乘 1.2，上限 10 秒（和桌面 App 一样）。"""
        deadline = time.monotonic() + timeout
        errors = 0
        n = 0
        while time.monotonic() < deadline:
            try:
                got = await self.poll_login_once(flow)
                errors = 0
            except GrokBotError as e:
                errors += 1
                self._log("warning", f"轮询登录状态失败：{e.detail}")
                if errors >= 3:
                    raise
                got = None
            if got is not None:
                return got
            await _sleep(min(1.2 ** n, 10))
            n += 1
        return False

    # ----- 续期 -----

    async def refresh(self, force: bool = False) -> None:
        """用 refresh token 换新的 access token。服务端要求重新登录时标记 needsLogin 并抛 AUTH。

        force=False 时，token 还没到续期时间就不续（别的进程可能刚续过，先重读文件）。"""
        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()
        async with self._refresh_lock:
            self.creds.reload()
            if not force and self.creds.access_token and self.creds.expires_at - time.time() > REFRESH_AHEAD:
                return
            if not self.creds.refresh_token:
                raise GrokBotError(ErrorKind.AUTH, "没有 refresh token，需要重新登录")
            used = self.creds.refresh_token
            status, _, body = await self._request("POST", self.backend + "/oauth/token", {
                "client_id": CLIENT_ID, "grant_type": "refresh_token", "refresh_token": used,
            }, timeout=20)
            if status != 200 or not isinstance(body, dict) or not body.get("access_token"):
                # 管理后台和 provider 各有一份客户端，可能同时续期：输的一方被拒时，先看文件里是不是已经有新 token
                self.creds.reload()
                if self.creds.refresh_token != used and self.creds.expires_at - time.time() > REFRESH_AHEAD:
                    self._log("info", "续期被拒，但别处已经续成功，用新的")
                    return
                if isinstance(body, dict) and body.get("shouldLogout") or status in (400, 401, 403):
                    self.creds.mark_needs_login()
                    raise GrokBotError(ErrorKind.AUTH, f"续期被拒绝（HTTP {status}），需要重新登录")
                raise GrokBotError(ErrorKind.SERVER, f"续期失败：HTTP {status} {body}")
            data = {k: v for k, v in self.creds._data.items() if k != "needsLogin"}
            data["accessToken"] = body["access_token"]
            if body.get("refresh_token"):
                data["refreshToken"] = body["refresh_token"]
            self.creds.save(data)
            self._log("info", "access token 已续期，到期：" + time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(self.creds.expires_at)))

    async def fresh_token(self) -> str:
        self.creds.reload()
        if not self.creds.access_token:
            raise GrokBotError(ErrorKind.AUTH, "还没登录 Grok Bot")
        if self.creds.needs_login:
            raise GrokBotError(ErrorKind.AUTH, "Grok Bot 的登录已失效，需要重新登录")
        if self.creds.expires_at - time.time() < REFRESH_AHEAD:
            await self.refresh()
        return self.creds.access_token

    # ----- 云端电脑 -----

    async def _backend_headers(self) -> dict:
        h = {
            "connect-protocol-version": "1",
            "x-cursor-client-type": "sand",
            "x-cursor-client-version": self.client_version,
            "x-sand-box-namespace": "prod",
            "x-ghost-mode": "true",
            "x-request-id": str(uuid.uuid4()),
            "authorization": "Bearer " + await self.fresh_token(),
        }
        if self.creds.team_id:
            h["x-cursor-team-id"] = self.creds.team_id
        return h

    async def ensure_box(self, wake: bool = True) -> dict:
        """拿到云端电脑的网关地址和令牌；休眠时唤醒并等它就绪。"""
        if self._ensure_lock is None:
            self._ensure_lock = asyncio.Lock()
        async with self._ensure_lock:
            t0 = time.monotonic()
            deadline = t0 + WAKE_TIMEOUT
            refreshed = False
            waited = False  # 中途看到过还没就绪（或请求挂到超时）：肯定是在唤醒
            while True:
                try:
                    status, hdrs, body = await self._request(
                        "POST", self.backend + "/aiserver.v1.GrokBotService/EnsureSandBox",
                        {"wake": wake, "supportsAsleepBox": True}, await self._backend_headers(),
                        timeout=ENSURE_TIMEOUT)
                except GrokBotError as e:
                    # 冷启动时服务端会把请求挂到电脑醒来（实测 60 多秒）：超时只说明还没醒，唤醒已经在进行，再问一次
                    if not isinstance(e.__cause__, asyncio.TimeoutError) or time.monotonic() > deadline:
                        raise
                    self._log("info", f"EnsureSandBox {int(ENSURE_TIMEOUT)} 秒没返回，云端电脑可能还在醒，再问一次")
                    waited = True
                    continue
                if status == 401 and not refreshed:
                    # token 看着还没过期却被拒：强制续期一次再试；还不行就要重新登录
                    refreshed = True
                    await self.refresh(force=True)
                    continue
                if status in (401, 403):
                    self.creds.mark_needs_login()
                    raise GrokBotError(ErrorKind.AUTH, f"服务端不认这个登录（HTTP {status}），需要重新登录")
                if status != 200 or not isinstance(body, dict):
                    hint = hdrs.get("x-automation-failure-hint", "") if hdrs else ""
                    raise GrokBotError(ErrorKind.SERVER, f"EnsureSandBox HTTP {status} {body} {hint}".strip())
                if body.get("gatewayUrl") and body.get("gatewayToken"):
                    host = urlparse(body["gatewayUrl"]).hostname or ""
                    if not host.endswith(GATEWAY_SUFFIXES) and not host.startswith(("127.", "localhost")):
                        raise GrokBotError(ErrorKind.SERVER, f"网关地址不在预期域名内：{host}")
                    self.conn = body
                    took = time.monotonic() - t0
                    woke = waited or took >= WAKE_SLOW
                    if woke:
                        self.woke_at = time.monotonic()
                    self._log("info", f"云端电脑就绪 runState={body.get('runState')} 网关 {host}（{took:.1f}s"
                                      f"{'，像是刚叫醒，用之前先确认网关能响应' if woke else ''}）")
                    return body
                if time.monotonic() > deadline:
                    raise GrokBotError(ErrorKind.TIMEOUT, f"云端电脑 {WAKE_TIMEOUT} 秒内没醒（runState={body.get('runState')}）")
                self._log("info", f"云端电脑还没就绪（runState={body.get('runState')}），{WAKE_POLL} 秒后重试")
                waited = True
                await _sleep(WAKE_POLL)

    async def confirm_awake(self) -> None:
        """云端电脑刚叫醒（见 WAKE_SLOW）时，用一个很轻的请求（listAgents）确认里面的网关真能响应，
        单次 WAKE_PROBE_TIMEOUT 秒、隔 WAKE_PROBE_PAUSE 秒再试，从醒来算起最多 WAKE_PROBE_MAX 秒，至少试一次。
        确认不了抛 GrokBotError。不是刚叫醒的（或之后已经有请求成功过）什么都不做。"""
        if self.woke_at is None:
            return
        deadline = self.woke_at + WAKE_PROBE_MAX
        t0 = time.monotonic()
        tries = 0
        while True:
            tries += 1
            try:
                await self.call("listAgents", retry=False, timeout=WAKE_PROBE_TIMEOUT)
            except GrokBotError as e:
                if e.kind == ErrorKind.AUTH or time.monotonic() + WAKE_PROBE_PAUSE >= deadline:
                    raise GrokBotError(e.kind, f"云端电脑刚叫醒，网关试了 {tries} 次（{time.monotonic() - t0:.1f}s）"
                                               f"都没响应：{e.detail[:160]}")
                self._log("info", f"云端电脑刚叫醒，网关还没响应（第 {tries} 次：{e.detail[:80]}），"
                                  f"{WAKE_PROBE_PAUSE} 秒后再试")
                await _sleep(WAKE_PROBE_PAUSE)
                continue
            self._log("info", f"云端电脑刚叫醒，网关已能响应（第 {tries} 次，{time.monotonic() - t0:.1f}s）")
            return  # call() 成功时已经清掉 woke_at

    # ----- 网关 -----

    def _gateway_headers(self) -> dict:
        h = {"authorization": "Bearer " + self.conn["gatewayToken"], "x-sand-slim-avatars": "1"}
        if self.conn.get("networkToken"):
            h["x-anyrun-network-token"] = self.conn["networkToken"]
        return h

    def gateway_url(self, path: str) -> str:
        return self.conn["gatewayUrl"].rstrip("/") + path

    async def call(self, method: str, args: dict = None, retry: bool = True, timeout: float = 60):
        """`POST {网关}/api/<method>`。令牌失效或云端电脑休眠时重新 EnsureSandBox 再试一次。"""
        if self.conn is None:
            await self.ensure_box()
        try:
            status, hdrs, body = await self._request(
                "POST", self.gateway_url("/api/" + method), args or {}, self._gateway_headers(), timeout=timeout)
        except GrokBotError as e:
            # 连不上网关：云端电脑可能换了机器（地址变了）。重新问后台要一次地址再试
            if not retry:
                raise
            self._log("info", f"{method} 连不上网关（{e.detail[:80]}），重新获取网关连接")
            self.conn = None
            await self.ensure_box(wake=True)
            return await self.call(method, args, retry=False, timeout=timeout)
        asleep = status == 417 or (hdrs is not None and (hdrs.get("x-anyrun-failure-reason") or "").startswith("pod_hibernated"))
        if retry and (status in (401, 403) or asleep or status >= 500):
            self._log("info", f"{method} 返回 {status}，重新获取网关连接")
            self.conn = None
            await self.ensure_box(wake=True)
            return await self.call(method, args, retry=False, timeout=timeout)
        if not 200 <= status < 300:
            raise GrokBotError(ErrorKind.SERVER, f"{method} HTTP {status} {str(body)[:200]}")
        self.woke_at = None  # 网关响应了：刚叫醒的也不用再确认
        return body

    async def list_agents(self) -> List[dict]:
        agents = await self.call("listAgents")
        return [a for a in agents if isinstance(a, dict)] if isinstance(agents, list) else []

    async def send_prompt(self, agent_id: str, text: str, nonce: str, timeout: float = 60) -> dict:
        res = await self.call("sendPrompt", {
            "agentId": agent_id, "prompt": text, "clientNonce": nonce,
            "directAddressedAcceptance": True, "source": "desktop", "sendSource": "chat",
            "initiator": "user", "composedAtMs": int(time.time() * 1000),
        }, retry=False, timeout=timeout)
        return res if isinstance(res, dict) else {"raw": res}

    async def acceptance_status(self, agent_id: str, nonce: str, timeout: float = 60) -> dict:
        res = await self.call("promptAcceptanceStatus", {"accountSlot": "host", "agentId": agent_id, "clientNonce": nonce},
                              timeout=timeout)
        return res if isinstance(res, dict) else {"raw": res}

    async def transcript_tail(self, agent_id: str, limit: int = 10) -> List[dict]:
        """对话尾部最近 limit 条。实测 2026-10-24 云端电脑换机器后 getAgentTranscriptTail 一度返回
        500 "server transcript tail is not enabled"，所以依次试三个接口，哪个能用用哪个。"""
        last = None
        for method, args in (("getAgentTranscriptTail", {"id": agent_id, "limit": limit}),
                             ("getAgentTranscriptWindow", {"id": agent_id, "limit": limit}),
                             ("getAgentTranscript", {"id": agent_id})):
            try:
                res = await self.call(method, args)
            except GrokBotError as e:
                if e.kind == ErrorKind.AUTH:
                    raise
                last = e
                self._log("warning", f"{method} 失败，换下一个接口：{e.detail[:120]}")
                continue
            entries = res.get("entries") if isinstance(res, dict) else res
            if isinstance(entries, list):
                return [e for e in entries if isinstance(e, dict)][-limit:]
        raise last or GrokBotError(ErrorKind.SERVER, "读不到对话记录")

    async def read_events(self, on_event: Callable[[str, dict], None], on_connected: Callable[[], None] = None) -> None:
        """读一条事件流直到它断开（正常约 57 秒一次）。每帧调用 on_event(channel, payload)。"""
        if self.conn is None:
            await self.ensure_box()
        headers = {**self._gateway_headers(), "accept": "text/event-stream"}
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=20, sock_read=STREAM_READ_TIMEOUT)
        try:
            # read_bufsize 放大：一条 updated 事件带着整段长回复，超过 aiohttp 默认的行上限会断流
            async with self.session().get(self.gateway_url("/events"), headers=headers, timeout=timeout,
                                          read_bufsize=4 * 1024 * 1024) as res:
                if res.status in (401, 403, 417):
                    self.conn = None
                    raise GrokBotError(ErrorKind.SERVER, f"/events HTTP {res.status}")
                if res.status != 200:
                    raise GrokBotError(ErrorKind.SERVER, f"/events HTTP {res.status}")
                if on_connected:
                    on_connected()
                data = []
                async for raw in res.content:
                    line = raw.decode(errors="replace").rstrip("\r\n")
                    if line == "":
                        if data:
                            try:
                                ev = json.loads("\n".join(data))
                            except ValueError:
                                ev = None
                            data = []
                            if isinstance(ev, dict):
                                on_event(str(ev.get("channel") or ""), ev.get("payload") or {})
                    elif line.startswith("data:"):
                        data.append(line[5:].lstrip(" "))
                    # 以 ":" 开头的是心跳注释（:ping），retry: 是重连提示，都不用管
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self.conn = None  # 连不上或一直没字节：下次连之前重新要网关地址（云端电脑可能换了机器）
            raise GrokBotError(ErrorKind.SERVER, f"/events {e!r}")


# ---------- 事件流中枢：一个账号只保持一条事件流 ----------

class Subscription:
    def __init__(self, hub: "GrokBotHub", agent_id: str):
        self.hub = hub
        self.agent_id = agent_id
        self.queue: asyncio.Queue = asyncio.Queue()

    def close(self) -> None:
        self.hub.unsubscribe(self)


class GrokBotHub:
    """管理事件流：有人订阅就连着，断线按 1/2/4/8/10 秒退避重连，没人订阅一段时间后关闭。

    每次（重新）连上 generation 加 1，订阅方据此补拉对话尾部。reconnect() 用于事件流"假活"时强制重连。
    """

    def __init__(self, client: GrokBotClient, idle_seconds: float = 60, logger=None):
        self.client = client
        self.idle_seconds = idle_seconds
        self.logger = logger
        self.generation = 0
        self.connected = False
        self._connected_event = asyncio.Event()
        self._subs: List[Subscription] = []
        self._task: Optional[asyncio.Task] = None
        self._idle_handle: Optional[asyncio.TimerHandle] = None
        self._stream_closed: Optional[asyncio.Event] = None  # 当前这条流需要被放弃时置位

    def _set_connected(self, value: bool) -> None:
        self.connected = value
        if value:
            self._connected_event.set()
        else:
            self._connected_event.clear()

    async def wait_connected(self, timeout: float) -> bool:
        """等事件流连上（最多 timeout 秒）。发送前先等一下，免得回显等早期事件在连上之前漏掉。"""
        if self.connected:
            return True
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def _log(self, level: str, msg: str) -> None:
        if self.logger:
            getattr(self.logger, level)(f"[grokbot] {msg}")

    def subscribe(self, agent_id: str) -> Subscription:
        sub = Subscription(self, agent_id)
        self._subs.append(sub)
        if self._idle_handle:
            self._idle_handle.cancel()
            self._idle_handle = None
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self._run())
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)
        if not self._subs and self._task is not None and not self._task.done() and self._idle_handle is None:
            loop = asyncio.get_running_loop()
            self._idle_handle = loop.call_later(self.idle_seconds, self._idle_close)

    def _idle_close(self) -> None:
        self._idle_handle = None
        if not self._subs and self._task is not None:
            self._log("info", f"事件流空闲 {int(self.idle_seconds)} 秒，关闭")
            self._task.cancel()
            self._task = None
            self._set_connected(False)

    def reconnect(self) -> None:
        """放弃当前这条流（例如一直收不到事件），马上重连。"""
        if self._stream_closed is not None:
            self._stream_closed.set()

    async def close(self) -> None:
        if self._idle_handle:
            self._idle_handle.cancel()
            self._idle_handle = None
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._set_connected(False)
        await self.client.close()

    def _dispatch(self, channel: str, payload: dict) -> None:
        agent_id = None
        if channel == "transcript":
            agent_id = payload.get("agentId")
        elif channel == "agent-upserted":
            agent_id = (payload.get("agent") or {}).get("id")
        for sub in list(self._subs):
            if agent_id is None or sub.agent_id == agent_id:
                sub.queue.put_nowait((channel, payload))

    def _broadcast(self, channel: str, payload: dict) -> None:
        for sub in list(self._subs):
            sub.queue.put_nowait((channel, payload))

    async def _run(self) -> None:
        fails = 0
        while True:
            self._stream_closed = asyncio.Event()
            closed = self._stream_closed

            def connected():
                nonlocal fails
                fails = 0
                self.generation += 1
                self._set_connected(True)
                self._log("debug", f"事件流已连接（第 {self.generation} 次）")
                self._broadcast("hub-connected", {"generation": self.generation})

            reader = asyncio.ensure_future(self.client.read_events(self._dispatch, connected))
            waiter = asyncio.ensure_future(closed.wait())
            try:
                done, _ = await asyncio.wait({reader, waiter}, return_when=asyncio.FIRST_COMPLETED)
                if waiter in done:
                    self._log("info", "事件流被要求重连")
                else:
                    reader.result()  # 抛出读流时的异常
                    self._log("debug", "事件流结束，重连")
            except asyncio.CancelledError:
                raise
            except GrokBotError as e:
                self._set_connected(False)
                # 没登录、登录失效：重连也没用，等用户在后台重新登录，放慢到 1 分钟一次
                delay = AUTH_RETRY if e.kind == ErrorKind.AUTH else STREAM_BACKOFF[min(fails, len(STREAM_BACKOFF) - 1)]
                fails += 1
                self._log("warning", f"事件流：{e.detail}，{delay} 秒后重连")
                await _sleep(delay)
                continue
            except Exception as e:  # noqa: BLE001 —— 常驻任务，任何意外都重连
                self._set_connected(False)
                delay = STREAM_BACKOFF[min(fails, len(STREAM_BACKOFF) - 1)]
                fails += 1
                self._log("warning", f"事件流出错：{e!r}，{delay} 秒后重连")
                await _sleep(delay)
                continue
            finally:
                for t in (reader, waiter):
                    if not t.done():
                        t.cancel()
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass
            self._set_connected(False)


# 按凭证文件共享的中枢：同一个账号的多个 Bot（多个 provider 实例）只开一条事件流
_hubs: Dict[str, GrokBotHub] = {}


def get_hub(cred_path: Path = None, logger=None, backend: str = None) -> GrokBotHub:
    """取（或创建）这个账号的中枢。必须在要使用它的事件循环里调用。backend 只在测试时指向假服务。"""
    path = Path(cred_path) if cred_path else default_credentials_path()
    key = str(path.resolve())
    hub = _hubs.get(key)
    if hub is None:
        hub = GrokBotHub(GrokBotClient(Credentials(path), backend=backend or BACKEND, logger=logger), logger=logger)
        _hubs[key] = hub
    return hub
