"""Grok Bot 客户端、事件流中枢、会话层和 provider 的单元测试。

用一个假的"云端"（同时扮演 api2.cursor.sh 和云端电脑的网关）跑完整流程，不需要真实账号。
真实账号的端到端实测记录见工作空间的 NOTES.md。
"""

import asyncio
import threading
import base64
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from core.agent import grokbot_client as gc
from core.agent import grokbot_session as gs
from core.agent.agent_session import Activity, AgentSessionError, Done, Error, ErrorKind, TextDelta
from core.agent.grokbot_client import Credentials, GrokBotClient, GrokBotError, GrokBotHub
from core.agent.grokbot_session import GrokBotSession, GrokBotSessionConfig
from core.agent.session_store import SessionStore
from core.providers.llm.grok_bot import grok_bot as grok_mod

real_sleep = gc._sleep  # conftest 把 asyncio.sleep 改成了不等待的桩


def jwt(exp_in: float) -> str:
    claims = base64.urlsafe_b64encode(json.dumps({"exp": time.time() + exp_in}).encode()).rstrip(b"=").decode()
    return f"h.{claims}.s"


async def wait_until(cond, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("等待超时")
        await real_sleep(0.01)


# ---------- 假云端 ----------

class FakeCloud:
    def __init__(self):
        self.app = web.Application()
        self.app.add_routes([
            web.get("/auth/poll", self.auth_poll),
            web.post("/oauth/token", self.oauth_token),
            web.post("/aiserver.v1.GrokBotService/EnsureSandBox", self.ensure),
            web.post("/api/listAgents", self.list_agents),
            web.post("/api/sendPrompt", self.send_prompt),
            web.post("/api/promptAcceptanceStatus", self.acceptance_status),
            web.post("/api/getAgentTranscriptTail", self.tail),
            web.post("/api/getAgentTranscriptWindow", self.window),
            web.post("/api/getAgentTranscript", self.transcript),
            web.get("/events", self.events),
        ])
        self.url = None
        self.approved = False
        self.poll_hold = 0  # 接下来这么多次 /auth/poll 挂着不回（长轮询，还没批准）
        self.refresh_calls = 0
        self.refresh_logout = False
        self.ensure_calls = 0
        self.ensure_hold = 0  # 接下来这么多次 EnsureSandBox 挂着不回（冷启动，电脑还在醒）
        self.ensure_states = []  # 依次返回的 runState（没有 gatewayUrl）；用完后正常返回
        self.ensure_status = []  # 依次返回的 HTTP 状态码（401 等）
        self.agents = [{"id": "bot-1", "name": "孙悟空", "isRunning": False, "newestEntryId": "e0", "harness": "temporal"}]
        self.entries = [{"id": "e0", "kind": "message", "role": "user", "content": "旧消息"}]
        self.sent = []
        self.send_fail_once = False
        self.send_hold = 0.0  # sendPrompt 收到后过这么久才返回
        self.tail_hold = 0  # 接下来这么多次读尾部挂一会儿才返回
        self.tail_waiting = False
        self.acceptance = {"status": "accepted"}
        self.hibernate_once = False
        self.tail_fail = 0  # 接下来这么多次 getAgentTranscriptTail 返回 500
        self.window_fail = 0  # 同上，getAgentTranscriptWindow
        self.transcript_fail = 0  # 同上，getAgentTranscript
        self.on_refresh = None  # 续期请求到达时先调它（模拟别处同时续期成功）
        self.events_q: asyncio.Queue = asyncio.Queue()
        self.stream_dead = False  # 连上了但什么都不推
        self.streams = 0
        self.gateway_tokens = []

    async def start(self):
        self.server = TestServer(self.app)
        await self.server.start_server()
        self.url = f"http://127.0.0.1:{self.server.port}"

    async def stop(self):
        await self.server.close()

    async def push(self, channel, payload):
        await self.events_q.put((channel, payload))

    # ----- api2 -----

    async def auth_poll(self, request):
        if self.poll_hold:
            self.poll_hold -= 1
            await real_sleep(0.5)
        if not self.approved:
            return web.Response(status=404, text="Not found")
        return web.json_response({"accessToken": jwt(3600), "refreshToken": "r-1", "selectedTeamId": 0})

    async def oauth_token(self, request):
        self.refresh_calls += 1
        if self.on_refresh:
            self.on_refresh()
        if self.refresh_logout:
            return web.json_response({"shouldLogout": True}, status=401)
        return web.json_response({"access_token": jwt(3600), "refresh_token": "r-2"})

    async def ensure(self, request):
        self.ensure_calls += 1
        if self.ensure_hold:
            self.ensure_hold -= 1
            await real_sleep(0.5)
        if self.ensure_status:
            return web.json_response({"error": "x"}, status=self.ensure_status.pop(0))
        if self.ensure_states:
            return web.json_response({"runState": self.ensure_states.pop(0)})
        return web.json_response({"gatewayUrl": self.url, "gatewayToken": "gw-1", "networkToken": "nt-1",
                                  "runState": "SAND_BOX_RUN_STATE_RUNNING"})

    # ----- 网关 -----

    def _check(self, request):
        self.gateway_tokens.append(request.headers.get("authorization"))
        if self.hibernate_once:
            self.hibernate_once = False
            return web.Response(status=417, headers={"x-anyrun-failure-reason": "pod_hibernated"})
        return None

    async def list_agents(self, request):
        return self._check(request) or web.json_response(self.agents)

    async def send_prompt(self, request):
        body = await request.json()
        self.sent.append(body)
        if self.send_hold:
            await real_sleep(self.send_hold)
        if self.send_fail_once:
            self.send_fail_once = False
            return web.Response(status=500, text="boom")
        return web.json_response({"accepted": True})

    async def acceptance_status(self, request):
        return web.json_response(self.acceptance)

    async def tail(self, request):
        if self.tail_fail > 0:
            self.tail_fail -= 1
            return web.json_response({"error": "server transcript tail is not enabled"}, status=500)
        body = await request.json()
        if self.tail_hold:
            self.tail_hold -= 1
            self.tail_waiting = True
            await real_sleep(0.5)
        return web.json_response({"entries": self.entries[-int(body.get("limit", 10)):], "nextBeforeSeq": 0})

    async def window(self, request):
        if self.window_fail > 0:
            self.window_fail -= 1
            return web.Response(status=500, text="window boom")
        body = await request.json()
        return web.json_response({"entries": self.entries[-int(body.get("limit", 10)):], "nextBeforeSeq": 0, "threadCounts": {}})

    async def transcript(self, request):
        if self.transcript_fail > 0:
            self.transcript_fail -= 1
            return web.Response(status=500, text="transcript boom")
        return web.json_response(self.entries)

    async def events(self, request):
        self.streams += 1
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b"retry: 1000\n\n")
        if not self.stream_dead:
            await resp.write(b'data: {"channel":"box-disk-pressure","payload":null}\n\n')
        try:
            while True:
                try:
                    channel, payload = await asyncio.wait_for(self.events_q.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    await resp.write(b":ping\n\n")
                    continue
                if channel == "__close__":
                    break
                if self.stream_dead:
                    continue
                await resp.write(f"data: {json.dumps({'channel': channel, 'payload': payload}, ensure_ascii=False)}\n\n".encode())
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp


@pytest.fixture
async def cloud(monkeypatch):
    monkeypatch.setattr(gc, "STREAM_BACKOFF", (0.05,))
    monkeypatch.setattr(gc, "WAKE_POLL", 0.01)
    c = FakeCloud()
    await c.start()
    yield c
    await c.stop()


@pytest.fixture
def creds(tmp_path):
    c = Credentials(tmp_path / "creds.json")
    c.save({"accessToken": jwt(3600), "refreshToken": "r-1"})
    return c


@pytest.fixture
async def client(cloud, creds):
    c = GrokBotClient(creds, backend=cloud.url, website=cloud.url)
    yield c
    await c.close()


@pytest.fixture
async def hub(client):
    h = GrokBotHub(client, idle_seconds=0.1)
    yield h
    await h.close()


def cfg(**kw):
    base = dict(bot_id="bot-1", bot_name="孙悟空", echo_timeout=5, poll_interval=5, done_delay=0.15, no_reply_timeout=5)
    base.update(kw)
    return GrokBotSessionConfig(**base)


def echo_entry(cloud, text="你好"):
    return {"id": "e1", "kind": "message", "role": "user", "content": text, "clientNonce": cloud.sent[-1]["clientNonce"]}


def reply_entry(text="**收到**", eid="e2", seq=None):
    e = {"id": eid, "kind": "send-message", "message": {"type": "text", "content": text}}
    if seq is not None:
        e["seq"] = seq
    return e


async def collect(session, text="你好"):
    return [ev async for ev in session.send(text)]


def texts(evs):
    return [e.text for e in evs if isinstance(e, TextDelta)]


# ---------- 凭证 ----------

def test_credentials_file_mode_and_reload(tmp_path):
    c = Credentials(tmp_path / "c.json")
    assert not c.logged_in and c.status()["expires_at"] == 0
    c.save({"accessToken": jwt(100), "refreshToken": "r"})
    assert oct(os.stat(c.path).st_mode & 0o777) == "0o600"
    assert c.logged_in and 90 < c.expires_at - time.time() <= 100
    # 别的进程改了文件：下次读状态时重读
    other = Credentials(c.path)
    other.save({"accessToken": jwt(5000), "refreshToken": "r"})
    os.utime(c.path, ns=(os.stat(c.path).st_atime_ns, os.stat(c.path).st_mtime_ns + 1))
    assert c.status()["expires_at"] - time.time() > 4000
    c.mark_needs_login()
    assert c.needs_login and not c.logged_in
    c.clear()
    assert not c.path.exists() and c.status()["logged_in"] is False


def test_jwt_exp_tolerates_garbage():
    assert gc.jwt_exp("") == 0 and gc.jwt_exp("a.b.c") == 0


# ---------- 登录、续期、云端电脑 ----------

async def test_login_flow(cloud, tmp_path):
    client = GrokBotClient(Credentials(tmp_path / "c.json"), backend=cloud.url, website=cloud.url)
    flow = client.start_login()
    assert "/loginDeepControl?" in flow.url and "challenge=" in flow.url and flow.verifier not in flow.url
    assert await client.poll_login_once(flow) is None  # 还没批准
    cloud.approved = True
    assert await client.poll_login_once(flow) is True
    assert client.creds.logged_in and client.creds.team_id == "0"
    await client.close()


async def test_login_long_poll_timeout_means_pending(cloud, tmp_path, monkeypatch):
    """/auth/poll 还没批准时挂着不回：超时算"还在等"，连着超时多次也不放弃，批准后照样登录成功。"""
    monkeypatch.setattr(gc, "LOGIN_POLL_TIMEOUT", 0.1)

    async def no_wait(seconds):
        await real_sleep(0)
    monkeypatch.setattr(gc, "_sleep", no_wait)
    client = GrokBotClient(Credentials(tmp_path / "c.json"), backend=cloud.url, website=cloud.url)
    flow = client.start_login()
    cloud.poll_hold = 1
    assert await client.poll_login_once(flow) is None
    cloud.poll_hold, cloud.approved = 4, True  # 以前连着 3 次超时就报错放弃
    assert await client.poll_login(flow, timeout=10) is True and client.creds.logged_in
    await client.close()


async def test_refresh_when_token_expiring(cloud, tmp_path):
    creds = Credentials(tmp_path / "c.json")
    creds.save({"accessToken": jwt(100), "refreshToken": "r-1"})
    client = GrokBotClient(creds, backend=cloud.url)
    await client.fresh_token()
    assert cloud.refresh_calls == 1 and creds.refresh_token == "r-2" and creds.expires_at - time.time() > 3000
    await client.fresh_token()
    assert cloud.refresh_calls == 1  # 还没到续期时间不再续
    await client.close()


async def test_refresh_rejected_marks_needs_login(cloud, tmp_path):
    creds = Credentials(tmp_path / "c.json")
    creds.save({"accessToken": jwt(100), "refreshToken": "r-1"})
    cloud.refresh_logout = True
    client = GrokBotClient(creds, backend=cloud.url)
    with pytest.raises(GrokBotError) as e:
        await client.fresh_token()
    assert e.value.kind == ErrorKind.AUTH and creds.needs_login
    with pytest.raises(GrokBotError) as e2:
        await client.ensure_box()
    assert e2.value.kind == ErrorKind.AUTH and cloud.refresh_calls == 1  # 标记后不再反复续期
    await client.close()


async def test_not_logged_in_is_auth_error(cloud, tmp_path):
    client = GrokBotClient(Credentials(tmp_path / "none.json"), backend=cloud.url)
    with pytest.raises(GrokBotError) as e:
        await client.ensure_box()
    assert e.value.kind == ErrorKind.AUTH
    await client.close()


async def test_ensure_box_waits_for_wake(cloud, client):
    cloud.ensure_states = ["SAND_BOX_RUN_STATE_ASLEEP", "SAND_BOX_RUN_STATE_STARTING"]
    conn = await client.ensure_box()
    assert conn["gatewayUrl"] == cloud.url and cloud.ensure_calls == 3


async def test_ensure_box_401_refreshes_once_then_gives_up(cloud, client):
    cloud.ensure_status = [401]
    await client.ensure_box()
    assert cloud.refresh_calls == 1 and cloud.ensure_calls == 2
    cloud.ensure_status = [401, 401]
    client.conn = None
    with pytest.raises(GrokBotError) as e:
        await client.ensure_box()
    assert e.value.kind == ErrorKind.AUTH and client.creds.needs_login


async def test_hibernated_gateway_is_woken_and_retried(cloud, client):
    cloud.hibernate_once = True
    agents = await client.list_agents()
    assert agents[0]["name"] == "孙悟空" and cloud.ensure_calls == 2
    assert cloud.gateway_tokens[-1] == "Bearer gw-1"


# ---------- 事件流中枢 ----------

async def test_hub_reconnects_when_stream_closes(cloud, hub):
    sub = hub.subscribe("bot-1")
    await wait_until(lambda: hub.generation == 1)
    await cloud.push("__close__", {})
    await wait_until(lambda: hub.generation == 2)
    assert cloud.streams == 2
    sub.close()
    await wait_until(lambda: hub._task is None, timeout=2)  # 没人订阅后关闭


async def test_hub_dispatches_by_agent_and_broadcasts(cloud, hub):
    a, b = hub.subscribe("bot-1"), hub.subscribe("bot-2")
    await wait_until(lambda: hub.generation == 1)
    await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": {"id": "x"}})
    await cloud.push("agent-upserted", {"agent": {"id": "bot-2"}})
    ch_a, _ = await asyncio.wait_for(a.queue.get(), 2)
    while ch_a in ("hub-connected", "box-disk-pressure"):
        ch_a, _ = await asyncio.wait_for(a.queue.get(), 2)
    assert ch_a == "transcript"
    ch_b, _ = await asyncio.wait_for(b.queue.get(), 2)
    while ch_b in ("hub-connected", "box-disk-pressure"):
        ch_b, _ = await asyncio.wait_for(b.queue.get(), 2)
    assert ch_b == "agent-upserted"
    assert a.queue.empty() and b.queue.empty()


# ---------- 会话 ----------

async def test_session_normal_reply(cloud, hub):
    s = GrokBotSession(hub, cfg())
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": echo_entry(cloud)})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry()})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "updated", "entry": reply_entry()})  # 同一条更新不重播
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert texts(evs) == ["收到。"]
    assert isinstance(evs[-1], Done) and not evs[-1].interrupted and evs[-1].session_id == "bot-1"
    assert cloud.sent[0]["agentId"] == "bot-1" and cloud.sent[0]["prompt"] == "你好" and cloud.sent[0]["clientNonce"]
    assert any(isinstance(e, Activity) and e.tools_running == 1 for e in evs)
    await s.close()


async def test_session_ignores_old_entries_and_streaming(cloud, hub):
    s = GrokBotSession(hub, cfg())
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended",
                                        "entry": {"id": "e0", "kind": "send-message", "message": {"type": "text", "content": "旧的"}}})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": {**reply_entry("写到一半"), "isStreaming": True}})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "updated", "entry": reply_entry("写完了。")})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended",
                                        "entry": {"id": "e3", "kind": "tool-call", "content": "不念"}})
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert texts(evs) == ["写完了。"]
    await s.close()


async def test_session_dead_stream_falls_back_to_polling(cloud, hub):
    """事件流"假活"：连着、有 ping，但不推事件。发送后没回显就重连并补拉，之后靠轮询拿到回复和结束。"""
    cloud.stream_dead = True
    s = GrokBotSession(hub, cfg(echo_timeout=0.3, poll_interval=0.3))
    await s.start()
    gen_before = None

    async def script():
        nonlocal gen_before
        await wait_until(lambda: cloud.sent)
        gen_before = hub.generation
        cloud.entries += [echo_entry(cloud), reply_entry("还在。")]
        cloud.agents[0]["newestEntryId"] = "e2"

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert texts(evs) == ["还在。"] and isinstance(evs[-1], Done)
    assert hub.generation > gen_before and cloud.streams >= 2  # 重连过
    await s.close()


async def test_session_recovers_after_stream_closes_mid_turn(cloud, hub):
    s = GrokBotSession(hub, cfg())
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": echo_entry(cloud)})
        await cloud.push("__close__", {})  # 服务端约 57 秒断一次
        # 断线期间 Bot 回复并空闲：重连后补拉尾部拿到
        cloud.entries += [echo_entry(cloud), reply_entry("在。")]
        cloud.agents[0]["newestEntryId"] = "e2"

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert texts(evs) == ["在。"] and isinstance(evs[-1], Done) and cloud.streams >= 2
    await s.close()


async def test_send_failure_not_delivered(cloud, hub):
    cloud.send_fail_once = True
    cloud.acceptance = {"status": "not-found"}
    s = GrokBotSession(hub, cfg())
    await s.start()
    evs = await collect(s)
    assert isinstance(evs[-1], Error) and evs[-1].kind == ErrorKind.SERVER and "没收到" in evs[-1].voice_message
    await s.close()


async def test_send_failure_but_delivered_continues(cloud, hub):
    cloud.send_fail_once = True
    cloud.acceptance = {"status": "accepted", "acceptedAt": 1}
    s = GrokBotSession(hub, cfg())
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("收到。")})
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert texts(evs) == ["收到。"] and isinstance(evs[-1], Done)
    await s.close()


async def test_interrupt_stops_speaking_only(cloud, hub):
    s = GrokBotSession(hub, cfg())
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await s.interrupt()

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert isinstance(evs[-1], Done) and evs[-1].interrupted
    assert len(cloud.sent) == 1  # 没有向云端发任何停止指令
    await s.close()


async def test_waiting_for_user_in_app(cloud, hub):
    s = GrokBotSession(hub, cfg())
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False, "awaitingUserResponse": "approval"}})

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert texts(evs) == [gs.WAITING_NOTICE] and isinstance(evs[-1], Done)
    await s.close()


async def test_no_reply_notice(cloud, hub):
    s = GrokBotSession(hub, cfg(no_reply_timeout=0.3, poll_interval=0.1))
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": echo_entry(cloud)})

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert texts(evs) == [gs.NO_REPLY_NOTICE] and isinstance(evs[-1], Done)
    await s.close()


async def test_session_auth_error_on_start(cloud, tmp_path):
    client = GrokBotClient(Credentials(tmp_path / "none.json"), backend=cloud.url)
    hub = GrokBotHub(client)
    with pytest.raises(AgentSessionError) as e:
        await GrokBotSession(hub, cfg()).start()
    assert e.value.kind == ErrorKind.AUTH
    await hub.close()


def test_plain_text_and_spoken_form():
    md = "# 结论\n\n- **第一**：看了 `58` 条帖子\n- 第二：[链接](https://x.com) 里的 *重点*\n\n```py\nprint(1)\n```\n| a | b |\n|---|---|\n| 1 | 2 |"
    out = gs.plain_text(md)
    assert out.startswith("结论\n第一：看了 58 条帖子\n第二：链接 里的 重点")
    assert "**" not in out and "`" not in out and "https://" not in out and "|" not in out and "print" not in out
    assert gs.spoken_form("收到") == "收到。" and gs.spoken_form("好的！") == "好的！" and gs.spoken_form("") == ""
    assert gs.reply_text({"kind": "message", "role": "assistant", "content": "x"}) == "x"
    assert gs.reply_text({"kind": "message", "role": "user", "content": "x"}) is None
    assert gs.reply_text({"kind": "send-message", "message": {"type": "attachment"}}) is None
    assert gs.reply_text({"kind": "tool-call"}) is None


# ---------- provider ----------

class FakeGrokAgent:
    instances = []
    script = []
    start_error = None
    delay = 0.0

    def __init__(self, hub, config, logger=None, on_report=None, device_key=""):
        self.hub, self.config, self.on_report, self.device_key = hub, config, on_report, device_key
        self.session_id = config.bot_id
        self.alive = False
        self.sent, self.interrupts, self.closed, self.handed_off = [], 0, False, []
        FakeGrokAgent.instances.append(self)

    async def start(self):
        if FakeGrokAgent.start_error:
            raise FakeGrokAgent.start_error
        self.alive = True

    async def send(self, text):
        self.sent.append(text)
        if FakeGrokAgent.delay:
            await real_sleep(FakeGrokAgent.delay)
        for ev in (FakeGrokAgent.script.pop(0) if FakeGrokAgent.script else [TextDelta("收到。"), Done(session_id="bot-1")]):
            yield ev

    async def interrupt(self):
        self.interrupts += 1

    async def hand_off(self, text):
        self.sent.append(text)
        self.handed_off.append(text)

    async def close(self):
        self.closed = True
        self.alive = False


@pytest.fixture
def provider(tmp_path, monkeypatch):
    from core.agent import reports as _reports
    _reports.clear()  # 别的测试文件可能留下待汇报
    FakeGrokAgent.instances, FakeGrokAgent.script, FakeGrokAgent.start_error, FakeGrokAgent.delay = [], [], None, 0.0
    monkeypatch.setattr(grok_mod, "GrokBotSession", FakeGrokAgent)
    monkeypatch.setattr(grok_mod, "get_hub", lambda *a, **k: SimpleNamespace(name="hub"))
    p = grok_mod.LLMProvider({"bot_id": "bot-1", "bot_name": "孙悟空", "credentials_path": str(tmp_path / "c.json"),
                              "filler_delay_seconds": 0.3, "turn_timeout_seconds": 3})
    p.store = SessionStore(tmp_path / "sessions.json", "grok_bot")
    return p


def conn(sid="conn-1", device="dev-1"):
    return SimpleNamespace(session_id=sid, device_id=device, intent_type="nointent")


def run(p, c, text):
    return "".join(p.response(c.session_id, [{"role": "system", "content": "你是小智"}, {"role": "user", "content": text}]))


def test_provider_speaks_reply_and_reuses_session(provider):
    c = conn()
    provider.on_conn_open(c)
    assert run(provider, c, "第一句") == "收到。"
    assert run(provider, c, "第二句") == "收到。"
    assert len(FakeGrokAgent.instances) == 1 and FakeGrokAgent.instances[0].sent == ["第一句", "第二句"]
    assert provider.model_name == "grok_bot:孙悟空" and provider.store.get("dev-1")["session_id"] == "bot-1"


def test_provider_filler_mentions_grok(provider):
    FakeGrokAgent.delay = 0.6
    c = conn()
    provider.on_conn_open(c)
    out = run(provider, c, "慢一点")
    assert out.startswith("😶交给 Grok Bot 了。") and out.endswith("收到。")


def test_provider_auth_error_has_grok_wording(provider):
    FakeGrokAgent.start_error = AgentSessionError(ErrorKind.AUTH, "no creds")
    c = conn()
    provider.on_conn_open(c)
    assert "Grok Bot 还没登录" in run(provider, c, "你好")


def test_provider_timeout_has_grok_wording(provider, tmp_path, monkeypatch):
    monkeypatch.setattr(grok_mod, "get_hub", lambda *a, **k: None)
    p = grok_mod.LLMProvider({"bot_id": "bot-1", "bot_name": "孙悟空", "credentials_path": str(tmp_path / "c.json"),
                              "turn_timeout_seconds": 0.3, "first_reply_filler": ""})
    p.store = SessionStore(tmp_path / "s2.json", "grok_bot")
    FakeGrokAgent.script = [[Activity("agent", tools_running=1), 2.0]]

    async def slow_send(self, text):
        yield Activity("agent", tools_running=1)
        await real_sleep(2)
        yield Done()

    monkeypatch.setattr(FakeGrokAgent, "send", slow_send)
    c = conn()
    p.on_conn_open(c)
    assert "Grok Bot 这件事做得太久" in run(p, c, "你好")


def test_providers_share_one_loop(provider, tmp_path):
    p2 = grok_mod.LLMProvider({"bot_id": "bot-2", "bot_name": "林黛玉", "credentials_path": str(tmp_path / "c.json")})
    assert p2._loop is provider._loop and p2._loop.is_running()
    provider.shutdown()
    assert provider._loop.is_running()  # 共用的循环不随实例退役而停止


def test_provider_self_check_without_login(tmp_path, monkeypatch, caplog):
    p = grok_mod.LLMProvider({"bot_id": "", "bot_name": "x", "credentials_path": str(tmp_path / "c.json")})
    assert p.bot_id == "" and p.model_name == "grok_bot:x"


def test_voice_messages_cover_kinds():
    for kind in (ErrorKind.AUTH, ErrorKind.SERVER, ErrorKind.TIMEOUT, ErrorKind.STALLED, ErrorKind.PROCESS_EXITED):
        assert "Grok Bot" in gs.voice_for(kind)
    assert gs.voice_for(ErrorKind.NOT_INSTALLED) == gs.VOICE_MESSAGES[ErrorKind.UNKNOWN]


# ---------- review 后补的用例 ----------

async def test_updated_entry_growth_speaks_only_the_new_part(cloud, hub):
    s = GrokBotSession(hub, cfg())
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("第一段")})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "updated", "entry": reply_entry("第一段。第二段。")})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "updated", "entry": reply_entry("改写了")})  # 不是追加：不念
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert texts(evs) == ["第一段。", "第二段。"] and isinstance(evs[-1], Done)
    await s.close()


async def test_waiting_ends_turn_even_if_bot_still_busy(cloud, hub):
    s = GrokBotSession(hub, cfg())
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True, "awaitingUserResponse": {"kind": "question"}}})

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert texts(evs) == [gs.WAITING_NOTICE] and isinstance(evs[-1], Done)
    await s.close()


async def test_idle_before_first_reply_waits_for_late_message(cloud, hub):
    """服务端托管的 Bot 先变空闲、再追加最后一条消息：空闲后一条都没念过就多等几秒。"""
    s = GrokBotSession(hub, cfg(first_reply_grace=1.0))
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})
        await real_sleep(0.5)  # 超过 done_delay（0.15）但没超过 first_reply_grace
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("迟到的回复。")})

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert texts(evs) == ["迟到的回复。"] and isinstance(evs[-1], Done)
    await s.close()


async def test_hub_wait_connected(cloud, hub):
    assert hub.connected is False
    sub = hub.subscribe("bot-1")
    assert await hub.wait_connected(2) is True and hub.connected
    sub.close()


async def test_hub_backs_off_a_minute_when_not_logged_in(cloud, tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "AUTH_RETRY", 0.05)
    client = GrokBotClient(Credentials(tmp_path / "none.json"), backend=cloud.url)
    h = GrokBotHub(client)
    sub = h.subscribe("bot-1")
    await real_sleep(0.3)
    assert h.connected is False and h._task is not None and not h._task.done()  # 一直在重试，但没有连上
    sub.close()
    await h.close()


def test_provider_slow_start_is_not_a_stall(provider, tmp_path, monkeypatch):
    """云端电脑唤醒要几十秒：会话启动期间没有事件，不能被当成卡住。"""
    monkeypatch.setattr(grok_mod, "get_hub", lambda *a, **k: None)
    p = grok_mod.LLMProvider({"bot_id": "bot-1", "bot_name": "孙悟空", "credentials_path": str(tmp_path / "c.json"),
                              "stall_timeout_seconds": 0.3, "first_reply_filler": "", "turn_timeout_seconds": 5})
    p.store = SessionStore(tmp_path / "s3.json", "grok_bot")

    async def slow_start(self):
        await real_sleep(0.8)
        self.alive = True

    monkeypatch.setattr(FakeGrokAgent, "start", slow_start)
    c = conn()
    p.on_conn_open(c)
    assert run(p, c, "你好") == "收到。"


# ---------- 被打断后做完再汇报、热身、提示词 ----------

from core.agent import reports as reports_mod  # noqa: E402


def reporting_session(hub, **kw):
    got = []
    s = GrokBotSession(hub, cfg(**kw), on_report=got.append)
    return s, got


async def test_interrupt_then_watch_reports_when_done(cloud, hub):
    s, got = reporting_session(hub)
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await s.interrupt()

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert isinstance(evs[-1], Done) and evs[-1].interrupted and got == []
    # 打断后 Bot 继续做：回复和空闲由后台盯着的任务收，做完交给 on_report
    await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("查完了。")})
    await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("一共三条。", "e3")})
    await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})
    await wait_until(lambda: got, timeout=3)
    assert got == ["孙悟空做完了：查完了。一共三条。"]
    await s.close()


async def test_watch_reports_waiting_for_user(cloud, hub):
    s, got = reporting_session(hub)
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await s.interrupt()

    task = asyncio.ensure_future(script())
    await collect(s)
    await task
    await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False, "awaitingUserResponse": "approval"}})
    await wait_until(lambda: got, timeout=3)
    assert got == ["孙悟空在等你去 App 里处理。"]
    await s.close()


async def test_watch_survives_session_close_and_idle_stream(cloud, hub):
    """设备断开（会话关闭）后后台盯着的任务还在，事件流也因为它还有订阅而不关。"""
    s, got = reporting_session(hub)
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await s.interrupt()

    task = asyncio.ensure_future(script())
    await collect(s)
    await task
    await s.close()
    await real_sleep(0.3)  # 超过 hub 的空闲关闭时间（0.1）
    assert hub._task is not None and not hub._task.done()
    await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("好了。")})
    await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})
    await wait_until(lambda: got, timeout=3)
    assert got == ["孙悟空做完了：好了。"]
    await wait_until(lambda: hub._task is None, timeout=2)  # 汇报完没人订阅了，事件流关闭


async def test_new_send_takes_watch_prelude(cloud, hub):
    s, got = reporting_session(hub)
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await s.interrupt()

    task = asyncio.ensure_future(script())
    await collect(s)
    await task
    await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("查到一半。")})
    await wait_until(lambda: gs._watches.get(s.watch_key) is not None and gs._watches[s.watch_key].texts, timeout=3)
    # Bot 还在忙，用户又派新活：先念它刚才回的，之后的回复新的一轮自己念；后台任务停掉，不再汇报
    async def script2():
        await wait_until(lambda: len(cloud.sent) == 2)
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("新活好了。", "e9")})
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})

    task2 = asyncio.ensure_future(script2())
    evs = await collect(s, "第二件事")
    await task2
    assert texts(evs) == ["孙悟空刚才回了：查到一半。", "新活好了。"] and isinstance(evs[-1], Done)
    await real_sleep(0.3)
    assert got == [] and s.watch_key not in gs._watches
    await s.close()


async def test_no_watch_when_turn_already_finished(cloud, hub):
    s, got = reporting_session(hub)
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("收到。")})
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert texts(evs) == ["收到。"] and not evs[-1].interrupted and s.watch_key not in gs._watches
    await s.close()


async def test_voice_hint_appended_and_echo_by_content(cloud, hub):
    s = GrokBotSession(hub, cfg(voice_hint="请简短回答"))
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        sent = cloud.sent[-1]["prompt"]
        assert sent == "你好（请简短回答）"
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended",
                                        "entry": {"id": "e1", "kind": "message", "role": "user", "content": sent}})  # 没带 nonce 的回显
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("好。")})
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert texts(evs) == ["好。"]
    await s.close()


def test_provider_defaults_filler_delay_and_hint(provider):
    p0 = grok_mod.LLMProvider({"bot_id": "b", "bot_name": "x"})
    assert p0.filler_delay == 4 and p0.session_config.voice_hint == grok_mod.DEFAULT_VOICE_HINT
    p = grok_mod.LLMProvider({"bot_id": "b", "bot_name": "x", "voice_hint": "", "filler_delay_seconds": 1})
    assert p.session_config.voice_hint == "" and p.filler_delay == 1


def test_provider_warms_up_on_conn_open(provider):
    c = conn()
    provider.on_conn_open(c)
    deadline = time.time() + 2
    while time.time() < deadline and not FakeGrokAgent.instances:
        time.sleep(0.02)
    time.sleep(0.05)
    assert len(FakeGrokAgent.instances) == 1 and FakeGrokAgent.instances[0].alive and FakeGrokAgent.instances[0].sent == []
    assert run(provider, c, "你好") == "收到。"
    assert len(FakeGrokAgent.instances) == 1  # 热身的会话直接复用


def test_provider_speaks_pending_report_first(provider):
    reports_mod.clear()
    c = conn()
    provider.on_conn_open(c)
    reports_mod.add("dev-1", "bot-1", "孙悟空做完了：好。")
    assert run(provider, c, "再来一件") == "孙悟空做完了：好。收到。"
    assert not reports_mod.pending("dev-1")


def test_provider_wires_on_report_to_reports(provider):
    reports_mod.clear()
    c = conn()
    provider.on_conn_open(c)
    run(provider, c, "你好")
    FakeGrokAgent.instances[0].on_report("孙悟空做完了：嗯。")
    assert reports_mod.take_text("dev-1", "bot-1") == "孙悟空做完了：嗯。"


async def test_reported_watch_is_not_spoken_again_as_prelude(cloud, hub):
    """真实账号实测发现：后台盯着的任务已经汇报进待汇报箱后，下一轮又把同样的话当开头念了一遍。"""
    s, got = reporting_session(hub)
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await s.interrupt()

    task = asyncio.ensure_future(script())
    await collect(s)
    await task
    await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("第一。")})
    await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})
    await wait_until(lambda: got, timeout=3)
    assert got == ["孙悟空做完了：第一。"]

    async def script2():
        await wait_until(lambda: len(cloud.sent) == 2)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("好。", "e9")})
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})

    task2 = asyncio.ensure_future(script2())
    evs = await collect(s, "再来一句")
    await task2
    assert texts(evs) == ["好。"]  # 没有"刚才回了：第一。"
    await s.close()


# ---------- 独立 review 后补的用例 ----------

async def test_watch_is_picked_up_by_new_session_after_close(cloud, hub):
    """派活被打断 → 设备断开（会话关闭）→ 重连后对同一个 Bot 派新活：新会话要接上旧的后台任务，不能重复念。"""
    s1, got = reporting_session(hub)
    s1.device_key = "dev-1"
    await s1.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await s1.interrupt()

    task = asyncio.ensure_future(script())
    await collect(s1)
    await task
    await s1.close()  # 设备断开
    await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("查到一半。")})
    await wait_until(lambda: gs._watches.get(("dev-1", "bot-1")) is not None and gs._watches[("dev-1", "bot-1")].texts, timeout=3)

    s2 = GrokBotSession(hub, cfg(), on_report=got.append, device_key="dev-1")  # 重连后的新会话
    await s2.start()

    async def script2():
        await wait_until(lambda: len(cloud.sent) == 2)
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("新活好了。", "e9")})
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})

    task2 = asyncio.ensure_future(script2())
    evs = await collect(s2, "第二件事")
    await task2
    assert texts(evs) == ["孙悟空刚才回了：查到一半。", "新活好了。"]
    await real_sleep(0.3)
    assert got == [] and ("dev-1", "bot-1") not in gs._watches  # 旧任务已停，不再汇报
    await s2.close()


async def test_baseline_tail_failure_falls_back_to_other_methods(cloud, hub):
    """2026-10-24 真机：云端电脑换机器后 getAgentTranscriptTail 一度 500，换 getAgentTranscriptWindow 照常发。"""
    cloud.tail_fail = 10
    s = GrokBotSession(hub, cfg())
    await s.start()
    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("好。")})
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})
    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert texts(evs) == ["好。"] and cloud.window_fail == 0
    await s.close()


async def test_no_baseline_at_all_uses_echo_seq(cloud, hub):
    """三个读对话的接口都不通：照常发，靠回显的 seq 分新旧——排在前面的旧回复不念，后面的念。"""
    cloud.tail_fail = cloud.window_fail = cloud.transcript_fail = 10
    s = GrokBotSession(hub, cfg())
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("旧的回复。", "old1", seq=5)})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": {**echo_entry(cloud), "seq": 10}})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("更旧的。", "old2", seq=7)})
        await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("新的回复。", "new1", seq=11)})
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert cloud.sent and texts(evs) == ["新的回复。"] and isinstance(evs[-1], Done)
    await s.close()


async def test_refresh_lost_race_uses_token_written_by_other_side(cloud, tmp_path):
    creds = Credentials(tmp_path / "c.json")
    creds.save({"accessToken": jwt(100), "refreshToken": "r-old"})
    other = Credentials(tmp_path / "c.json")

    def other_refreshed_first():
        other.save({"accessToken": jwt(3600), "refreshToken": "r-new"})
        os.utime(other.path, ns=(os.stat(other.path).st_atime_ns, os.stat(other.path).st_mtime_ns + 1))

    cloud.on_refresh = other_refreshed_first
    cloud.refresh_logout = True  # 我们这边拿旧 refresh token 去换，被拒
    client = GrokBotClient(creds, backend=cloud.url)
    await client.fresh_token()  # 不抛：发现文件里已有新 token
    assert creds.refresh_token == "r-new" and not creds.needs_login
    await client.close()


def test_warm_up_then_idle_close_does_not_crash(tmp_path, monkeypatch):
    """独立 review 发现：只呼名不派活，热身后到了空闲时间关闭会话时锁还是 None，直接崩，会话永远不关。"""
    monkeypatch.setattr(grok_mod, "GrokBotSession", FakeGrokAgent)
    monkeypatch.setattr(grok_mod, "get_hub", lambda *a, **k: None)
    FakeGrokAgent.instances, FakeGrokAgent.script, FakeGrokAgent.start_error, FakeGrokAgent.delay = [], [], None, 0.0
    p = grok_mod.LLMProvider({"bot_id": "bot-1", "bot_name": "孙悟空", "credentials_path": str(tmp_path / "c.json"),
                              "idle_timeout_seconds": 0.3})
    p.store = SessionStore(tmp_path / "s4.json", "grok_bot")
    c = conn()
    p.on_conn_open(c)
    deadline = time.time() + 2
    while time.time() < deadline and not (FakeGrokAgent.instances and FakeGrokAgent.instances[0].alive):
        time.sleep(0.02)
    assert FakeGrokAgent.instances[0].alive
    time.sleep(0.8)
    assert FakeGrokAgent.instances[0].closed and p._states[c.session_id].agent is None  # 空闲后关掉了


def slow_provider(tmp_path, monkeypatch, name):
    monkeypatch.setattr(grok_mod, "GrokBotSession", FakeGrokAgent)
    monkeypatch.setattr(grok_mod, "get_hub", lambda *a, **k: None)
    FakeGrokAgent.instances, FakeGrokAgent.script, FakeGrokAgent.start_error, FakeGrokAgent.delay = [], [], None, 0.0

    async def slow_start(self):
        await real_sleep(0.6)
        self.alive = True

    monkeypatch.setattr(FakeGrokAgent, "start", slow_start)
    p = grok_mod.LLMProvider({"bot_id": "bot-1", "bot_name": "孙悟空", "credentials_path": str(tmp_path / "c.json"),
                              "first_reply_filler": ""})
    p.store = SessionStore(tmp_path / name, "grok_bot")
    return p


def test_abort_during_slow_start_still_hands_off(tmp_path, monkeypatch):
    """云端电脑唤醒期间设备打断：话照样发出去（和 Bot 醒着时发出去后再打断一样），但不等回复、不占着锁，
    直接转后台盯着，做完进待汇报箱。以前是不发，用户点了结束这件事就丢了。"""
    p = slow_provider(tmp_path, monkeypatch, "s5.json")
    c = conn()
    c.client_abort = False
    p.on_conn_open(c)
    time.sleep(0.05)
    out = []

    def consume():
        out.extend(p.response(c.session_id, [{"role": "user", "content": "派个活"}]))

    t = threading.Thread(target=consume)
    t.start()
    time.sleep(0.2)
    c.client_abort = True  # 设备打断
    p.abort(c)
    t.join(3)
    assert out == []
    deadline = time.time() + 2
    while time.time() < deadline and not any(a.handed_off for a in FakeGrokAgent.instances):
        time.sleep(0.02)
    agent = next(a for a in FakeGrokAgent.instances if a.alive)
    assert agent.handed_off == ["派个活"]  # 醒了照样发出去，转后台盯着
    c.client_abort = False
    assert run(p, c, "再来") == "收到。"  # 锁没被占着，下一句正常
    assert agent.sent == ["派个活", "再来"]


def test_close_during_slow_start_still_hands_off(tmp_path, monkeypatch):
    """用户点了结束（打断并断开），云端电脑还没醒：醒了照样发出去，做完进待汇报箱，下次连上念。"""
    p = slow_provider(tmp_path, monkeypatch, "s6.json")
    c = conn()
    c.client_abort = False
    p.on_conn_open(c)
    time.sleep(0.05)
    t = threading.Thread(target=lambda: list(p.response(c.session_id, [{"role": "user", "content": "派个活"}])))
    t.start()
    time.sleep(0.2)
    c.client_abort = True
    p.abort(c)
    p.on_conn_close(c)
    t.join(3)
    deadline = time.time() + 2
    while time.time() < deadline and not any(a.handed_off for a in FakeGrokAgent.instances):
        time.sleep(0.02)
    time.sleep(0.1)
    handed = [a for a in FakeGrokAgent.instances if a.handed_off]
    assert len(handed) == 1 and handed[0].handed_off == ["派个活"] and handed[0].closed


async def test_hand_off_sends_then_watches(cloud, hub):
    """hand_off：发出去后不等回复，直接转后台盯着，Bot 做完把回复交给 on_report。"""
    s, got = reporting_session(hub)
    s.device_key = "dev-1"
    await s.start()
    await s.hand_off("派个活")
    assert len(cloud.sent) == 1 and cloud.sent[0]["prompt"].startswith("派个活")
    assert s.watch_key in gs._watches
    await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
    await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("做完了。")})
    await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})
    await wait_until(lambda: got, timeout=3)
    assert got == ["孙悟空做完了：做完了。"]


async def test_ensure_box_retries_while_box_wakes(cloud, client, monkeypatch):
    """冷启动时 EnsureSandBox 挂到电脑醒来（实测 60 多秒），单次超时不算失败，在总时限内再问。"""
    monkeypatch.setattr(gc, "ENSURE_TIMEOUT", 0.1)
    cloud.ensure_hold = 2
    body = await client.ensure_box()
    assert body["gatewayUrl"] == cloud.url and cloud.ensure_calls == 3


# ---------- 第二位独立 reviewer 后补的用例 ----------

async def test_close_mid_turn_turns_into_watch(cloud, hub):
    """设备在 Bot 干活时断开：这一轮当作被打断，转后台盯着，做完进待汇报箱，重连后能念。"""
    s, got = reporting_session(hub)
    s.device_key = "dev-1"
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
        await s.close()  # 设备断开 → provider 关掉会话

    task = asyncio.ensure_future(script())
    evs = await collect(s)
    await task
    assert isinstance(evs[-1], Done) and evs[-1].interrupted
    await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("做完了。")})
    await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})
    await wait_until(lambda: got, timeout=3)
    assert got == ["孙悟空做完了：做完了。"]


# ---------- 设备断开的各个时间点：之后拿到的回复都要进待汇报箱 ----------

@pytest.mark.parametrize("send_hold", [0.5, 0.0], ids=["sendPrompt 还没返回", "等回显"])
async def test_close_during_send_still_reports(cloud, hub, send_hold):
    """现场 19:37：sendPrompt 还没返回设备就断开了。以前只有发送之后的断开才算打断，这一轮被当成正常的收掉，
    回复没人听也不进待汇报箱。"""
    cloud.send_hold = send_hold
    s, got = reporting_session(hub)
    s.device_key = "dev-1"
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        await s.close()  # 设备断开 → provider 关掉会话

    task = asyncio.ensure_future(script())
    evs = await collect(s, "派个活")
    await task
    assert isinstance(evs[-1], Done) and evs[-1].interrupted and texts(evs) == []
    await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": True}})
    await cloud.push("transcript", {"agentId": "bot-1", "type": "appended", "entry": reply_entry("二百四十七。")})
    await cloud.push("agent-upserted", {"agent": {"id": "bot-1", "isRunning": False}})
    await wait_until(lambda: got, timeout=3)
    assert got == ["孙悟空做完了：二百四十七。"]


async def test_close_while_re_reading_still_reports(cloud, hub):
    """发送后没回显、重连补读时设备断开：补读拿回来的回复不能当这一轮念掉（没人听），要进待汇报箱。"""
    cloud.stream_dead = True
    s, got = reporting_session(hub, echo_timeout=0.3, poll_interval=5)
    s.device_key = "dev-1"
    await s.start()

    async def script():
        await wait_until(lambda: cloud.sent)
        cloud.entries += [echo_entry(cloud), reply_entry("二百四十七。")]
        cloud.agents[0]["newestEntryId"] = "e2"
        cloud.tail_hold = 1  # 补读挂一会儿，断开发生在补读途中
        await wait_until(lambda: cloud.tail_waiting)
        await s.close()

    task = asyncio.ensure_future(script())
    evs = await collect(s, "派个活")
    await task
    assert texts(evs) == []  # 没当这一轮念掉
    await wait_until(lambda: got, timeout=5)
    assert got == ["孙悟空做完了：二百四十七。"]


async def test_gateway_connection_error_re_ensures_box(cloud, client):
    """云端电脑换了机器：旧网关地址连不上，要重新向后台要地址，而不是一直报错。"""
    await client.ensure_box()
    client.conn = {**client.conn, "gatewayUrl": "http://127.0.0.1:9"}  # 打不开的端口
    agents = await client.list_agents()
    assert agents[0]["name"] == "孙悟空" and cloud.ensure_calls == 2 and client.conn["gatewayUrl"] == cloud.url


async def test_gateway_5xx_re_ensures_box_once(cloud, client):
    cloud.tail_fail = 1
    await client.ensure_box()
    entries = await client.transcript_tail("bot-1", 5)
    assert entries and cloud.ensure_calls == 2
    cloud.tail_fail = cloud.window_fail = cloud.transcript_fail = 10
    with pytest.raises(GrokBotError):
        await client.transcript_tail("bot-1", 5)  # 每个接口只重试一次，三个都不通才放弃
    assert cloud.tail_fail == 8 and cloud.window_fail == 8 and cloud.transcript_fail == 8
