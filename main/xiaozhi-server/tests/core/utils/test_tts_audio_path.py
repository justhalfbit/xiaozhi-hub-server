"""[xiaozhi-hub] 合成音频的处理：进程内解码 MP3、加增益不削顶、短句缓存。"""
import shutil
import subprocess
import types

import numpy as np
import pytest

from core.providers.tts.base import TTSProviderBase
from core.utils import mp3_decode, pcm_gain, tts_phrase_cache


def _pcm(x):
    return np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16).tobytes()


def _arr(b):
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768


def test_gain_keeps_quiet_parts_exact_and_never_clips():
    sr = 16000
    t = np.arange(sr) / sr
    env = np.full(sr, 0.1)
    env[6000:7000] += 0.7 * np.hanning(1000)          # a loud syllable: ×1.5 would reach 1.2
    x = env * np.sin(2 * np.pi * 220 * t)
    y = _arr(pcm_gain.apply_gain(_pcm(x), 1.5, sr))
    assert np.abs(y).max() <= pcm_gain.CEILING + 1e-3
    # away from the loud part the gain is exactly 1.5
    np.testing.assert_allclose(y[:5000], 1.5 * x[:5000], atol=2e-4)
    np.testing.assert_allclose(y[8000:], 1.5 * x[8000:], atol=2e-4)
    # pressed down smoothly, not clipped: no sample-to-sample step bigger than the amplified input's
    assert np.abs(np.diff(y)).max() <= 1.5 * np.abs(np.diff(x)).max() + 1e-3
    # and no flat tops
    assert (np.abs(y) > pcm_gain.CEILING - 1e-3).sum() < 5


def test_gain_one_is_untouched():
    b = _pcm(np.linspace(-0.5, 0.5, 100))
    assert pcm_gain.apply_gain(b, 1.0) is b


@pytest.mark.skipif(mp3_decode._lib() is None or not shutil.which("ffmpeg"), reason="需要 libmpg123 和 ffmpeg")
def test_mp3_decode_matches_ffmpeg(tmp_path):
    sr = 24000
    t = np.arange(sr) / sr
    x = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1700 * t)).astype(np.float32)
    wav = tmp_path / "a.raw"
    wav.write_bytes(_pcm(x))
    mp3 = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", str(wav),
                          "-b:a", "48k", "-f", "mp3", "-"], capture_output=True, check=True).stdout
    ref = _arr(subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                              input=mp3, capture_output=True, check=True).stdout)
    got = _arr(mp3_decode.decode_mp3(mp3, 16000))
    assert abs(len(got) - len(ref)) <= 16
    n = min(len(got), len(ref))
    a, b = got[2000:n - 2000], ref[2000:n - 2000]
    assert 10 * np.log10(np.sum(b ** 2) / np.sum((a - b) ** 2)) > 30


def test_mp3_decode_gives_up_on_garbage():
    assert mp3_decode.decode_mp3(b"not an mp3 at all", 16000) is None


class _TTS(TTSProviderBase):
    voice = "v"

    async def text_to_speak(self, text, output_file):
        self.calls.append(text)
        return b"fake"


def test_short_phrases_are_synthesized_once(monkeypatch):
    tts = object.__new__(_TTS)
    tts.calls, tts.audio_file_type = [], "mp3"
    monkeypatch.setattr("core.providers.tts.base.decode_audio_bytes", lambda data, ft, sr: b"\x01\x00" * 10)
    monkeypatch.setattr(tts_phrase_cache, "_cache", type(tts_phrase_cache._cache)())
    assert tts._synthesize_pcm("交给 Grok Bot 了", 16000) == b"\x01\x00" * 10
    assert tts._synthesize_pcm("交给 Grok Bot 了", 16000) == b"\x01\x00" * 10
    long = "这是一段很长的正式回复，每次内容都不一样，不应该进缓存"
    tts._synthesize_pcm(long, 16000)
    tts._synthesize_pcm(long, 16000)
    assert tts.calls == ["交给 Grok Bot 了", long, long]


def test_prompt_phrases_from_config():
    cfg = {"LLM": {
        "GrokBot_a": {"type": "grok_bot", "bot_id": "x"},
        "ClaudeCLI": {"type": "claude_cli", "first_reply_filler": "稍等，我想想", "progress_reply": "还在弄"},
        "DeepSeekLLM": {"type": "openai"},
    }}
    assert tts_phrase_cache.prompt_phrases(cfg) == ["交给 Grok Bot 了", "还在处理 请稍等", "稍等 我想想", "还在弄"]
