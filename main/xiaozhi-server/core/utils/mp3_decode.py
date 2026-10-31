"""[xiaozhi-hub] 进程内解码 MP3（系统的 libmpg123，ctypes 调用），代替每段起一次 ffmpeg。

在 ThinkSmart View 这类小设备上，pydub 每段都要启动一次 ffmpeg，2 秒的语音解码要 1.3 秒，
大半是进程启动开销；一轮回复切成好几段，每段都白等这么久，句间停顿就长。
libmpg123 解同样一段只要几十毫秒。系统里没有 libmpg123、或者采样率换算不是简单整数比时返回 None，
调用方退回原来的 pydub/ffmpeg。
"""

import ctypes
import ctypes.util
import math
from functools import lru_cache
from typing import Optional

import numpy as np

from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

MPG123_OK = 0
MPG123_NEED_MORE = -10
MPG123_NEW_FORMAT = -11
MPG123_DONE = -12
MPG123_MONO = 1
MPG123_STEREO = 2
MPG123_ENC_SIGNED_16 = 0xD0
_RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)
# 重采样的上采样倍数超过这个就不在这里做（如 22050→16000 要 ×320），交给 ffmpeg
_MAX_UP = 8


@lru_cache(maxsize=1)
def _lib():
    name = ctypes.util.find_library("mpg123")
    if not name:
        return None
    try:
        lib = ctypes.CDLL(name)
        lib.mpg123_new.restype = ctypes.c_void_p
        lib.mpg123_new.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        lib.mpg123_delete.argtypes = [ctypes.c_void_p]
        lib.mpg123_open_feed.argtypes = [ctypes.c_void_p]
        lib.mpg123_format_none.argtypes = [ctypes.c_void_p]
        lib.mpg123_format.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_int]
        lib.mpg123_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                      ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        lib.mpg123_getformat.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long),
                                         ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        lib.mpg123_init()  # 1.27 起是空操作，老版本必须先调
        return lib
    except (OSError, AttributeError) as e:
        logger.bind(tag=TAG).warning(f"libmpg123 不可用，MP3 用 ffmpeg 解码: {e}")
        return None


def _decode(data: bytes):
    """MP3 → (int16 单声道样本, 采样率)；失败返回 None。"""
    lib = _lib()
    if lib is None or not data:
        return None
    err = ctypes.c_int()
    mh = lib.mpg123_new(None, ctypes.byref(err))
    if not mh:
        return None
    try:
        lib.mpg123_format_none(mh)
        for r in _RATES:
            lib.mpg123_format(mh, r, MPG123_MONO | MPG123_STEREO, MPG123_ENC_SIGNED_16)
        if lib.mpg123_open_feed(mh) != MPG123_OK:
            return None
        src = ctypes.create_string_buffer(data, len(data))
        out = ctypes.create_string_buffer(65536)
        done = ctypes.c_size_t()
        chunks = []
        rate, channels = ctypes.c_long(), ctypes.c_int()
        enc = ctypes.c_int()
        fed = False
        while True:
            if fed:
                ret = lib.mpg123_decode(mh, None, 0, out, len(out), ctypes.byref(done))
            else:
                ret = lib.mpg123_decode(mh, src, len(data), out, len(out), ctypes.byref(done))
                fed = True
            if done.value:
                chunks.append(out.raw[: done.value])
            if ret == MPG123_NEW_FORMAT:
                lib.mpg123_getformat(mh, ctypes.byref(rate), ctypes.byref(channels), ctypes.byref(enc))
            elif ret in (MPG123_NEED_MORE, MPG123_DONE):
                break
            elif ret != MPG123_OK:
                return None
        if not chunks or not rate.value:
            return None
        pcm = np.frombuffer(b"".join(chunks), dtype=np.int16)
        if channels.value == 2:
            pcm = pcm.reshape(-1, 2).mean(axis=1)
        return pcm.astype(np.float32), rate.value
    finally:
        lib.mpg123_delete(mh)


@lru_cache(maxsize=8)
def _filter(up: int, down: int, taps_per_phase: int = 24):
    """整数比重采样的低通滤波器（加窗 sinc），截止在两边较低奈奎斯特频率的 0.9 倍。"""
    n = taps_per_phase * up + 1  # 奇数长度：群延迟是整数个样本，输出和输入对齐
    cutoff = 0.9 * 0.5 / max(up, down)  # 相对上采样后的采样率
    t = np.arange(n) - (n - 1) / 2
    h = 2 * cutoff * np.sinc(2 * cutoff * t) * np.kaiser(n, 8.0)
    return (h * up / h.sum()).astype(np.float32)


def _resample(x: np.ndarray, rate_in: int, rate_out: int) -> Optional[np.ndarray]:
    if rate_in == rate_out:
        return x
    g = math.gcd(rate_in, rate_out)
    up, down = rate_out // g, rate_in // g
    if up > _MAX_UP:
        return None
    h = _filter(up, down)
    # 多相实现：第 m 个输出 = 上采样序列在 m*down 处与滤波器的卷积，只算要用的那一相
    n_out = len(x) * up // down
    delay = (len(h) - 1) // 2
    y = np.empty(n_out, dtype=np.float32)
    pos = np.arange(n_out) * down + delay  # 在上采样（补零）序列里的位置
    for p in range(up):
        hp = h[p::up]  # 这一相的系数：作用在 x[k] 上，k = (pos - p) / up - j
        sel = np.where(pos % up == p)[0]
        if not len(sel):
            continue
        full = np.convolve(x, hp)
        idx = (pos[sel] - p) // up
        ok = idx < len(full)
        y[sel[ok]] = full[idx[ok]]
        y[sel[~ok]] = 0
    return y


def decode_mp3(data: bytes, sample_rate: int) -> Optional[bytes]:
    """MP3 → 指定采样率的 16 位单声道 PCM 字节；做不到时返回 None（调用方改用 ffmpeg）。"""
    try:
        got = _decode(data)
        if got is None:
            return None
        x, rate = got
        y = _resample(x, rate, sample_rate)
        if y is None:
            return None
        return np.clip(np.round(y), -32768, 32767).astype(np.int16).tobytes()
    except Exception as e:
        logger.bind(tag=TAG).warning(f"libmpg123 解码失败，改用 ffmpeg: {e}")
        return None
