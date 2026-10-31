"""[xiaozhi-hub] 判定说完后：只把最后有声音之后 0.3 秒的静音送去识别；要 vad_end 的客户端收到通知。"""
import asyncio
import json
from types import SimpleNamespace

from core.providers.asr.base import ASR_TAIL_BYTES, ASRProviderBase
from core.providers.asr.dto.dto import InterfaceType

FRAME = b"\x01\x00" * 320  # 20 ms


class FakeASR(ASRProviderBase):
    interface_type = InterfaceType.LOCAL

    def __init__(self):
        self.got = None

    async def speech_to_text(self, *args, **kwargs):
        return "", None

    async def handle_voice_stop(self, conn, asr_audio_task, **kwargs):
        self.got = b"".join(asr_audio_task)


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, msg):
        self.sent.append(json.loads(msg))


def make_conn(asr, features):
    conn = SimpleNamespace(client_listen_mode="auto", asr_audio=[], client_have_voice=False, client_voice_stop=False,
                           asr=asr, features=features, websocket=FakeWS(), session_id="s")
    conn.reset_audio_states = lambda: (conn.asr_audio.clear(), setattr(conn, "asr_voice_end", 0))
    return conn


async def feed(asr, conn, voiced, silent):
    for _ in range(voiced):
        conn.client_have_voice = True
        await asr.receive_audio(conn, FRAME, True)
    for i in range(silent):
        conn.client_voice_stop = i == silent - 1
        await asr.receive_audio(conn, FRAME, False)


def test_trailing_silence_trimmed_and_vad_end_sent():
    asr = FakeASR()
    conn = make_conn(asr, {"vad_end": True})
    asyncio.run(feed(asr, conn, voiced=50, silent=45))   # 1 s of speech, 0.9 s of silence
    assert len(asr.got) == 50 * len(FRAME) + ASR_TAIL_BYTES
    assert conn.websocket.sent == [{"type": "vad", "state": "end", "session_id": "s"}]


def test_no_vad_end_without_the_feature():
    asr = FakeASR()
    conn = make_conn(asr, {"mcp": True})
    asyncio.run(feed(asr, conn, voiced=50, silent=10))   # silence shorter than the kept tail: all of it
    assert len(asr.got) == 60 * len(FRAME)
    assert conn.websocket.sent == []


class SegASR(FakeASR):
    """边说边识别：有 recognize_pcm、开了 segment_pause_ms 的本地识别。"""
    segment_pause_ms = 300

    def __init__(self):
        super().__init__()
        self.segments = []

    def recognize_pcm(self, pcm):
        self.segments.append(len(pcm) // len(FRAME))
        return f"第{len(self.segments)}段。"

    async def handle_voice_stop(self, conn, asr_audio_task, segments=None, segmented_bytes=0):
        self.got = b"".join(asr_audio_task)
        self.texts = [await f for f in segments]
        self.segmented = segmented_bytes // len(FRAME)


async def feed_frames(asr, conn, plan):
    """plan: [(有声帧数, 静音帧数), ...]，最后一个静音的最后一帧判定说完。"""
    for k, (voiced, silent) in enumerate(plan):
        for _ in range(voiced):
            conn.client_have_voice = True
            await asr.receive_audio(conn, FRAME, True)
        for i in range(silent):
            conn.client_voice_stop = k == len(plan) - 1 and i == silent - 1
            await asr.receive_audio(conn, FRAME, False)


def test_pauses_are_recognised_while_talking():
    asr = SegASR()
    conn = make_conn(asr, {})
    # 1.2 s 说话、0.1 s 停顿（不切）、0.6 s、0.4 s 停顿（切）、1.2 s、0.9 s 静音（0.3 s 时切，说完时什么都不剩）
    # 每次切在最后有声音之后 0.15 秒（8 帧）
    asyncio.run(feed_frames(asr, conn, [(60, 5), (30, 20), (60, 45)]))
    assert asr.segments == [60 + 5 + 30 + 8, 12 + 60 + 8]
    assert asr.texts == ["第1段。", "第2段。"]
    whole = (60 + 5 + 30 + 20 + 60 + 15) * len(FRAME)   # 整句照常交给 handle_voice_stop
    assert len(asr.got) == whole
    assert asr.segmented * len(FRAME) == whole            # 都识别过了，说完不用再识别


def test_short_pieces_are_not_cut():
    asr = SegASR()
    conn = make_conn(asr, {})
    asyncio.run(feed_frames(asr, conn, [(20, 20), (60, 45)]))   # 0.4 s 说话 + 0.3 s 停顿不到 1 秒
    assert asr.segments == [20 + 20 + 60 + 8]                   # 第一次停顿没切，最后整句一段


def test_join_segments():
    from core.providers.asr.base import join_segments
    assert join_segments(["帮我查一下明天上海的天气。", "顺便告诉我带不带伞。"]) == "帮我查一下明天上海的天气， 顺便告诉我带不带伞。"
    assert join_segments(["你好吗？", "", "我很好。"]) == "你好吗？ 我很好。"
    assert join_segments(["", "只有一段。"]) == "只有一段。"


def test_voice_stop_recognises_only_the_rest(monkeypatch):
    import core.providers.asr.base as base
    said = []

    async def to_chat(conn, text, gen=None):
        said.append(text)
    monkeypatch.setattr(base, "startToChat", to_chat)
    monkeypatch.setattr(base, "correct_and_record", lambda conn, text: text)
    monkeypatch.setattr(base, "enqueue_asr_report", lambda *a: None)
    monkeypatch.setattr(base.reply_echo, "during_reply", lambda conn: False)

    class Rest(FakeASR):
        rest = None

        async def speech_to_text_wrapper(self, pcm, session_id):
            self.rest = len(b"".join(pcm))
            return "后半句。", None
        handle_voice_stop = ASRProviderBase.handle_voice_stop
        stop_ws_connection = lambda self: None

    async def run():
        asr = Rest()
        conn = make_conn(asr, {})
        conn.voiceprint_provider = None
        first = asyncio.get_running_loop().create_future()
        first.set_result("前半句。")
        await asr.handle_voice_stop(conn, [b"\0" * 64000], segments=[first], segmented_bytes=40000)
        return asr.rest
    assert asyncio.run(run()) == 24000
    assert said == ["前半句， 后半句。"]


def test_too_short_tells_the_client_nothing_was_heard():
    asr = FakeASR()
    conn = make_conn(asr, {"vad_end": True})
    asyncio.run(feed(asr, conn, voiced=10, silent=45))   # 0.2 s 的咳嗽：太短，不识别
    assert asr.got is None
    assert [m["state"] for m in conn.websocket.sent] == ["end", "empty"]


def _patch_chat(monkeypatch, said):
    import core.providers.asr.base as base

    async def to_chat(conn, text):
        said.append(text)
    monkeypatch.setattr(base, "startToChat", to_chat)
    monkeypatch.setattr(base, "correct_and_record", lambda conn, text: text)
    monkeypatch.setattr(base, "enqueue_asr_report", lambda *a: None)
    monkeypatch.setattr(base.reply_echo, "during_reply", lambda conn: False)


class Says(FakeASR):
    def __init__(self, text):
        super().__init__()
        self.text = text

    async def speech_to_text_wrapper(self, pcm, session_id):
        return self.text, None
    handle_voice_stop = ASRProviderBase.handle_voice_stop
    stop_ws_connection = lambda self: None


def test_empty_text_tells_the_client(monkeypatch):
    said = []
    _patch_chat(monkeypatch, said)
    for text, features, sent in (("", {"vad_end": True}, ["empty"]), ("。", {"vad_end": True}, ["empty"]),
                                 ("", {"mcp": True}, []), ("你好", {"vad_end": True}, [])):
        conn = make_conn(Says(text), features)
        conn.voiceprint_provider = None
        asyncio.run(conn.asr.handle_voice_stop(conn, [b"\0" * 64000]))
        assert [m["state"] for m in conn.websocket.sent] == sent, text
    assert said == ["你好"]


def test_hello_says_the_hub_sends_vad_events():
    from unittest.mock import MagicMock
    from core.handle.helloHandle import handleHelloMessage
    shared = {"type": "hello", "transport": "websocket", "audio_params": {"sample_rate": 24000}}
    for features, expect in (({"vad_end": True}, {"vad_end": True}), ({"emoji": False}, None)):
        conn = SimpleNamespace(welcome_msg=shared, websocket=FakeWS(), logger=MagicMock(), features=None)
        asyncio.run(handleHelloMessage(conn, {"type": "hello", "features": features}))
        assert conn.websocket.sent[0].get("features") == expect
    assert "features" not in shared   # 共用的欢迎消息不被改掉
