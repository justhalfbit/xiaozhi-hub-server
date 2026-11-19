"""Interactive ChatGPT sign-in using the device's own Codex credential store."""
import asyncio
import json
from urllib.parse import parse_qs, urlencode, urlsplit

import aiohttp


class LoginError(ValueError):
    pass


class LoginEnded(LoginError):
    """The page needs a new authorization URL."""


def callback_parts(url):
    if not isinstance(url, str) or len(url) > 16384:
        raise LoginError("请粘贴完整的回调链接")
    parts = urlsplit(url)
    if (parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost")
            or not parts.port or parts.username is not None or parts.password is not None
            or parts.path != "/auth/callback" or parts.fragment):
        raise LoginError("回调地址与本次授权不符")
    return parts


def callback_request(auth_url, pasted):
    query = parse_qs(urlsplit(auth_url).query)
    expected = callback_parts(query["redirect_uri"][0])
    actual = callback_parts(pasted)
    params = parse_qs(actual.query, keep_blank_values=True)
    if (actual.netloc != expected.netloc or params.get("state") != query["state"]
            or len(params.get("state", [])) != 1):
        raise LoginError("回调链接与本次授权不匹配，请重新复制")
    if params.get("error"):
        raise LoginError("OpenAI 授权未完成，请重新登录")
    codes = params.get("code", [])
    if len(codes) != 1 or not codes[0]:
        raise LoginError("链接缺少有效授权码，请复制完整地址")
    return expected._replace(query=urlencode({"state": params["state"][0], "code": codes[0]})).geturl()


class CodexLogin:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.process = self.timer = self.owner = self.url = self.login_id = None

    async def _stop(self):
        if self.timer:
            self.timer.cancel()
        proc = self.process
        self.process = self.timer = self.owner = self.url = self.login_id = None
        if proc and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()

    async def close(self, app=None):
        async with self.lock:
            await self._stop()

    async def _expire(self, proc):
        async with self.lock:
            if self.process is proc:
                await self._stop()

    async def _rpc(self, method, params, request_id=None):
        msg = {"method": method, "params": params}
        if request_id is not None:
            msg["id"] = request_id
        self.process.stdin.write(json.dumps(msg).encode() + b"\n")
        await self.process.stdin.drain()
        if request_id is not None:
            return await self._read(lambda m: m.get("id") == request_id)

    async def _read(self, matches):
        while True:
            line = await self.process.stdout.readline()
            if not line:
                raise LoginError("Codex 登录进程已退出，请重新登录")
            message = json.loads(line)
            if matches(message):
                if "error" in message:
                    raise LoginError("Codex 无法发起授权，请检查 CLI 版本和登录端口")
                return message

    async def _start(self, cli):
        self.process = await asyncio.create_subprocess_exec(
            cli, "app-server", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=2**20)
        await self._rpc("initialize", {"clientInfo": {"name": "xiaozhi_hub", "version": "1"}}, 1)
        await self._rpc("initialized", {})
        reply = await self._rpc("account/login/start", {"type": "chatgpt", "useHostedLoginSuccessPage": True}, 2)
        result = reply["result"]
        url = result["authUrl"]
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != "auth.openai.com":
            raise LoginError("Codex 返回了无效的 OpenAI 授权地址")
        args = parse_qs(parsed.query)
        callback_parts(args["redirect_uri"][0])
        if len(args.get("state", [])) != 1:
            raise LoginError("Codex 返回的授权信息不完整")
        self.url, self.login_id = url, result["loginId"]

    async def start(self, owner, cli):
        async with self.lock:
            if self.process and self.process.returncode is None:
                if owner != self.owner:
                    raise LoginError("其他管理员正在授权，请稍后重试")
            else:
                await self._stop()
                try:
                    await asyncio.wait_for(self._start(cli), 20)
                    self.owner = owner
                    self.timer = asyncio.get_running_loop().call_later(
                        600, lambda p=self.process: asyncio.create_task(self._expire(p)))
                except BaseException:
                    await self._stop()
                    raise
            return {"url": self.url, "login_id": self.login_id}

    async def _complete(self, url):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as client:
            async with client.get(url, allow_redirects=False) as resp:
                if resp.status != 302:
                    # CLI error pages may contain sensitive details. Surface only recognized causes.
                    page = (await resp.content.read(65536)).lower()
                    if b"unsupported_country_region_territory" in page or b"country, region, or territory not supported" in page:
                        raise LoginError("OpenAI 拒绝了设备所在地区的登录请求（HTTP 403）")
                    if b"invalid_grant" in page:
                        raise LoginError("授权码已失效，请重新获取授权链接")
                    if b"persist_failed" in page:
                        raise LoginError("设备无法保存凭证，请检查 Codex 凭证目录的权限")
                    raise LoginError("Codex 授权失败，请检查链接有效期和设备到 OpenAI 的网络连接")
                # Older CLIs finish the RPC only after the local success page is opened.
                target = urlsplit(resp.headers.get("Location", ""))
                if target.scheme == "http" and target.netloc == urlsplit(url).netloc and target.path == "/success":
                    async with client.get(urlsplit(url)._replace(path="/success", query="").geturl(),
                                          allow_redirects=False):
                        pass
        msg = await self._read(lambda m: m.get("method") == "account/login/completed"
                               and m.get("params", {}).get("loginId") == self.login_id)
        if not msg["params"].get("success"):
            raise LoginError("Codex 未能保存登录，请重新授权")

    async def finish(self, owner, login_id, pasted=None, *, cancel=False):
        async with self.lock:
            if not self.process or owner != self.owner or login_id != self.login_id:
                raise LoginEnded("授权已失效，请重新发起登录")
            if cancel:
                await self._stop()
                return
            url = callback_request(self.url, pasted)
            try:
                await asyncio.wait_for(self._complete(url), 60)
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
                raise LoginEnded("设备连接授权服务失败或超时，请检查网络后重新登录") from None
            except LoginError as exc:
                raise LoginEnded(str(exc)) from None
            finally:
                await self._stop()
