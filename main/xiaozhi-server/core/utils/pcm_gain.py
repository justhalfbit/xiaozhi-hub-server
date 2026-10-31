"""[xiaozhi-hub] 给合成好的语音加增益，峰值处平滑压下去而不是硬削顶。

Edge TTS 的 volume 是在服务端把波形整体乘上去："+50%" 时响一点的字峰值冲到满幅以上被硬削，
设备上听起来是字中间轻轻"咔"一下。改成按原音量合成、这里再乘增益：超过上限的地方，
以 4 毫秒为一块算出要压多少，前后各看两块取最小、块间线性过渡，增益变化平滑，不产生削顶的毛刺。
"""

import numpy as np

CEILING = 0.89   # 约 -1 dBFS：给 Opus 编解码留一点过冲余量
BLOCK_MS = 4
HOLD_BLOCKS = 2


def apply_gain(pcm: bytes, gain: float, sample_rate: int = 16000) -> bytes:
    """16 位单声道 PCM 乘以 gain，峰值限在 CEILING 以内。"""
    if gain == 1.0 or not pcm:
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * (gain / 32768.0)
    block = max(1, sample_rate * BLOCK_MS // 1000)
    n = -(-len(x) // block)
    padded = np.zeros(n * block, dtype=np.float32)
    padded[: len(x)] = np.abs(x)
    peak = padded.reshape(n, block).max(axis=1)
    g = np.minimum(1.0, CEILING / np.maximum(peak, 1e-9))
    if g.min() < 1.0:
        # 压峰要提前开始、晚一点结束：每块取前后 HOLD_BLOCKS 块里最小的增益
        held = np.pad(g, HOLD_BLOCKS, constant_values=1.0)
        g = np.min([held[i : i + n] for i in range(2 * HOLD_BLOCKS + 1)], axis=0)
        centers = np.arange(n) * block + block / 2
        x *= np.interp(np.arange(len(x)), centers, g).astype(np.float32)
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16).tobytes()
