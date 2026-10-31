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

    async def handle_voice_stop(self, conn, asr_audio_task):
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
