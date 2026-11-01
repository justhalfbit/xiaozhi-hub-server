"""[xiaozhi-hub] 边说边上屏：说话中途识别完的段，发给声明了 stt_partial 的客户端。"""
import asyncio
from types import SimpleNamespace

from core.providers.asr import base
from core.providers.asr.base import send_partial
from core.utils import turn_gen

from tests.hub.test_asr_tail import FRAME, FakeWS, SegASR, make_conn


def partials(conn):
    return [m["text"] for m in conn.websocket.sent if m.get("state") == "partial"]


async def settle(conn):
    """等已经送去识别的段识别完、发完（真机上帧 60 ms 一个，段早就识别完了）。
    conftest 把 asyncio.sleep 换成了不让出的空函数，这里自己让出事件循环。"""
    await asyncio.gather(*(getattr(conn, "asr_segments", None) or []))
    loop = asyncio.get_running_loop()
    for _ in range(3):
        f = loop.create_future()
        loop.call_soon(f.set_result, None)
        await f


async def feed_slowly(asr, conn, plan):
    """像真的一样一帧一帧来，帧与帧之间分段识别跑完。"""
    for k, (voiced, silent) in enumerate(plan):
        for _ in range(voiced):
            conn.client_have_voice = True
            await asr.receive_audio(conn, FRAME, True)
            await settle(conn)
        for i in range(silent):
            conn.client_voice_stop = k == len(plan) - 1 and i == silent - 1
            await asr.receive_audio(conn, FRAME, False)
            await settle(conn)


def test_each_segment_sends_everything_heard_so_far():
    asr = SegASR()
    conn = make_conn(asr, {"vad_end": True, "stt_partial": True})
    asyncio.run(feed_slowly(asr, conn, [(60, 5), (30, 20), (60, 45)]))
    assert partials(conn) == ["第1段", "第1段， 第2段"]
    assert conn.websocket.sent[0] == {"type": "stt", "state": "partial", "text": "第1段", "session_id": "s"}
    assert [m["state"] for m in conn.websocket.sent] == ["partial", "partial", "end"]   # 说完照常
    assert asr.texts == ["第1段。", "第2段。"]   # 最终结果照旧拿同样的分段，不多识别


def test_nothing_without_the_feature():
    asr = SegASR()
    conn = make_conn(asr, {"vad_end": True})
    asyncio.run(feed_slowly(asr, conn, [(60, 5), (30, 20), (60, 45)]))
    assert [m["state"] for m in conn.websocket.sent] == ["end"]
    assert len(asr.segments) == 2


def make_seg_conn(features=None):
    conn = SimpleNamespace(features=features or {"stt_partial": True}, websocket=FakeWS(), session_id="s",
                           client_voice_stop=False, asr_segments=[], client_is_speaking=False)
    return conn


async def done(text):
    f = asyncio.get_running_loop().create_future()
    f.set_result(text)
    return f


def test_a_later_segment_waits_for_the_earlier_one():
    async def run():
        conn = make_seg_conn()
        first = asyncio.get_running_loop().create_future()
        second = await done("后半句。")
        conn.asr_segments = [first, second]
        await send_partial(conn, second, 0)
        assert partials(conn) == []                 # 第一段还没出来：不能先显示后半句
        first.set_result("前半句。")
        await send_partial(conn, first, 0)
        assert partials(conn) == ["前半句， 后半句"]
    asyncio.run(run())


def test_not_sent_once_the_sentence_is_over_or_dropped():
    async def run():
        # 说完了（最终 stt / vad empty 马上就到）
        conn = make_seg_conn()
        f = await done("你好。")
        conn.asr_segments = [f]
        conn.client_voice_stop = True
        await send_partial(conn, f, 0)
        assert partials(conn) == []
        # 已经换成下一句（reset_audio_states 清掉了分段）
        conn = make_seg_conn()
        await send_partial(conn, f, 0)
        assert partials(conn) == []
        # 识别中被打断：代次变了
        conn = make_seg_conn()
        conn.asr_segments = [f]
        turn_gen.bump(conn)
        await send_partial(conn, f, 0)
        assert partials(conn) == []
        # 识别出错、什么也没认出来
        conn = make_seg_conn()
        bad = asyncio.get_running_loop().create_future()
        bad.set_exception(RuntimeError("boom"))
        empty = await done("。")
        conn.asr_segments = [bad, empty]
        await send_partial(conn, bad, 0)
        await send_partial(conn, empty, 0)
        assert partials(conn) == []
    asyncio.run(run())


def test_corrections_apply_and_echo_is_not_shown(monkeypatch):
    async def run():
        conn = make_seg_conn()
        f = await done("打开cloud code。")
        conn.asr_segments = [f]
        await send_partial(conn, f, 0)
        assert partials(conn) == ["打开Claude Code"]
        monkeypatch.setattr(base.reply_echo, "during_reply", lambda c: True)
        monkeypatch.setattr(base.reply_echo, "is_echo", lambda c, t: True)
        conn = make_seg_conn()
        conn.asr_segments = [f]
        await send_partial(conn, f, 0)
        assert partials(conn) == []
    asyncio.run(run())


def test_hello_says_the_hub_sends_partials():
    from unittest.mock import MagicMock
    from core.handle.helloHandle import handleHelloMessage
    shared = {"type": "hello", "transport": "websocket"}
    for features, expect in (({"vad_end": True, "stt_partial": True}, {"vad_end": True, "stt_partial": True}),
                             ({"stt_partial": True}, {"stt_partial": True}), ({"mcp": True}, None)):
        conn = SimpleNamespace(welcome_msg=shared, websocket=FakeWS(), logger=MagicMock(), features=None)
        asyncio.run(handleHelloMessage(conn, {"type": "hello", "features": features}))
        assert conn.websocket.sent[0].get("features") == expect
    assert "features" not in shared
