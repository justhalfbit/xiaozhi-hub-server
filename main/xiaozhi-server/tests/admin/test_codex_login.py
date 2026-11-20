import asyncio
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlencode

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from core.admin.codex_login import CodexLogin, LoginError, callback_request

CALLBACK = "http://127.0.0.1:1455/auth/callback"
AUTH = "https://auth.openai.com/oauth/authorize?" + urlencode({"redirect_uri": CALLBACK, "state": "match"})


@pytest.mark.parametrize("pasted", [
    CALLBACK + "?code=x&state=old", CALLBACK + "?code=x&state=match&code=y",
    CALLBACK + "?state=match", CALLBACK.replace("1455", "8080") + "?state=match&code=x",
    "https://example.com/auth/callback?state=match&code=x", CALLBACK + "?state=match&error=denied",
])
def test_rejects_invalid_callback(pasted):
    with pytest.raises(LoginError):
        callback_request(AUTH, pasted)


def active_login():
    flow = CodexLogin()
    flow.process = Mock(returncode=None, wait=AsyncMock())
    flow.owner, flow.login_id, flow.url = "admin", "one", AUTH
    return flow


async def test_reuse_cancel_and_expiry():
    flow = active_login()
    assert (await flow.start("admin", "codex"))["login_id"] == "one"
    with pytest.raises(LoginError):
        await flow.start("other", "codex")
    with pytest.raises(LoginError):
        await flow.finish("other", "one", cancel=True)
    with pytest.raises(LoginError):
        await flow.finish("admin", "old", cancel=True)
    proc = flow.process
    await flow.finish("admin", "one", cancel=True)
    proc.kill.assert_called_once()
    flow = active_login()
    await flow._expire(flow.process)
    with pytest.raises(LoginError):
        await flow.finish("admin", "one", cancel=True)


@pytest.mark.parametrize(("status", "body", "error"), [
    (302, "", None), (400, "", "授权失败"),
    (200, "unsupported_country_region_territory", "设备所在地区"),
    (200, "invalid_grant", "授权码已失效"),
])
async def test_callback_finishes_once(status, body, error):
    seen = []
    async def receive(request):
        seen.append(dict(request.query))
        return web.Response(status=status, text=body, headers={"Location": "https://example.invalid/success"})
    app = web.Application()
    app.router.add_get("/auth/callback", receive)
    async with TestServer(app) as srv:
        callback = str(srv.make_url("/auth/callback"))
        flow = active_login()
        flow.url = "https://auth.openai.com/oauth/authorize?" + urlencode({"redirect_uri": callback, "state": "match"})
        flow.process.stdout = asyncio.StreamReader()
        flow.process.stdout.feed_data(b'{"method":"account/login/completed","params":{"loginId":"one","success":true}}\n')
        pasted = callback + "?code=a%2Bb&state=match&unused=discard"
        if status == 302:
            await flow.finish("admin", "one", pasted)
        else:
            with pytest.raises(LoginError, match=error):
                await flow.finish("admin", "one", pasted)
        assert seen == [{"code": "a+b", "state": "match"}]
        with pytest.raises(LoginError):
            await flow.finish("admin", "one", pasted)
