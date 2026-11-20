"""Bridge the authenticated admin tab to the local ThinkSmart private X display."""
import asyncio
import json
import re
import os
from pathlib import Path

from aiohttp import WSMsgType, web

DISPLAY = Path("/run/thinksmart-clash")
STATE = Path("/var/lib/thinksmart-clash")
RELEASE_HELPER = Path("/opt/thinksmart/client/current/deploy/clash-party/release.py")
ASSETS = STATE / "novnc-1.5.0"


def _same_origin(request):
    if request.headers.get("Origin") != f"{request.scheme}://{request.host}":
        raise web.HTTPForbidden()


class ClashWindow:
    def __init__(self, require_login, logged_in, read_json):
        self.require_login = require_login
        self.logged_in = logged_in
        self.read_json = read_json
        self.paste_lock = asyncio.Lock()
        self.clients = set()

    def register(self, app):
        app.router.add_get("/admin/api/clash-party", self.status)
        app.router.add_get("/admin/api/clash-party/update", self.update)
        app.router.add_post("/admin/api/clash-party/update", self.update)
        app.router.add_get("/admin/api/clash-party/display", self.display)
        app.router.add_post("/admin/api/clash-party/paste", self.paste)
        app.router.add_get("/admin/clash-view/{path:.*}", self.asset)
        app.on_shutdown.append(self.shutdown)

    async def shutdown(self, app):
        await asyncio.gather(*(ws.close() for ws in list(self.clients)))

    async def status(self, request):
        self.require_login(request)
        ready = (DISPLAY / "rfb").is_socket() and (ASSETS / "core/rfb.js").is_file()
        phase = ""
        try:
            phase = (STATE / "status").read_text().strip()
        except OSError:
            pass
        message = ("请在设备的「设置 → Clash Party」中一键安装" if not phase else
                   "窗口暂未就绪，请在设备上检查 Clash Party 服务" if phase == "就绪" else phase)
        return web.json_response({"ready": ready, "message": "" if ready else message},
                                 headers={"Cache-Control": "no-store"})

    async def update(self, request):
        self.require_login(request)
        if request.method == "POST":
            _same_origin(request)
            version = (await self.read_json(request)).get("version")
            if not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) or len(version) > 64:
                raise web.HTTPBadRequest(text="版本号无效")
            args = ("queue", version)
        else:
            args = ("check",) if request.query.get("check") == "1" else ("status",)
        if not RELEASE_HELPER.is_file():
            return web.json_response({"error": "设备更新组件尚未安装"}, status=503)
        proc = await asyncio.create_subprocess_exec(
            "/usr/bin/python3", str(RELEASE_HELPER), *args, stdout=asyncio.subprocess.PIPE)
        output, _ = await proc.communicate()
        try:
            result = json.loads(output)
        except ValueError:
            return web.json_response({"error": "无法读取更新状态"}, status=503)
        return web.json_response(result, status=409 if proc.returncode else 200,
                                 headers={"Cache-Control": "no-store"})

    async def asset(self, request):
        self.require_login(request)
        path = (ASSETS / request.match_info["path"]).resolve()
        if not path.is_relative_to(ASSETS) or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    async def display(self, request):
        self.require_login(request)
        _same_origin(request)
        ws = web.WebSocketResponse(protocols=("binary",))
        if not ws.can_prepare(request).ok:
            raise web.HTTPBadRequest()
        try:
            reader, writer = await asyncio.open_unix_connection(DISPLAY / "rfb")
        except OSError:
            raise web.HTTPServiceUnavailable(text="窗口尚未启动")
        tasks = []
        try:
            with os.fdopen(os.open(DISPLAY / "show", os.O_WRONLY | os.O_NONBLOCK), "wb") as fifo:
                fifo.write(b"show\n")
            await ws.prepare(request)
            self.clients.add(ws)

            async def send():
                while data := await reader.read(65536):
                    if not self.logged_in(request):
                        break
                    await ws.send_bytes(data)

            async def receive():
                async for msg in ws:
                    if msg.type != WSMsgType.BINARY or not self.logged_in(request):
                        break
                    writer.write(msg.data)
                    await writer.drain()

            async def watch_session():
                while self.logged_in(request):
                    await asyncio.sleep(1)

            tasks = [asyncio.create_task(fn()) for fn in (send, receive, watch_session)]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            writer.close()
            self.clients.discard(ws)
            if ws.prepared:
                await ws.close()
        return ws

    async def paste(self, request):
        self.require_login(request)
        _same_origin(request)
        value = (await self.read_json(request)).get("text")
        if not isinstance(value, str) or not 0 < len(value) <= 8192:
            raise web.HTTPBadRequest(text="文本长度无效")
        if not (DISPLAY / "rfb").is_socket():
            raise web.HTTPServiceUnavailable(text="窗口尚未启动")
        env = dict(os.environ, DISPLAY=":73", XAUTHORITY=str(DISPLAY / "Xauthority"))
        async with self.paste_lock:
            for args, data in [(("xclip", "-selection", "clipboard", "-in"), value.encode()),
                               (("xdotool", "key", "--clearmodifiers", "ctrl+v"), b"")]:
                proc = await asyncio.create_subprocess_exec(
                    *args, env=env, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                try:
                    await asyncio.wait_for(proc.communicate(data), 4)
                finally:
                    if proc.returncode is None:
                        proc.kill()
                        await proc.wait()
                if proc.returncode:
                    raise web.HTTPServiceUnavailable(text="粘贴失败")
        return web.json_response({"ok": True})
