"""sherpa_local：用 sherpa-onnx 在本机离线合成语音，不依赖任何网络。

为什么需要它：默认的 EdgeTTS 走微软的 speech.platform.bing.com，国内网络下经常解析失败或连接超时，
失败后要重试 4-5 次、每次间隔十几秒，一轮回复就这么丢掉了（tmp/server.log 里的"语音生成失败"）。
这里复用项目已经装好的 sherpa-onnx（语音识别用的也是它），换一个中文 VITS 模型，在进程内直接出 PCM：
不联网、没有重试、首句几百毫秒。

配置（data/.config.yaml 的 TTS 条目）：
    type: sherpa_local
    model_dir: ~/tts-models/vits-zh-hf-fanchen-C   # 模型目录（含 *.onnx、tokens.txt、lexicon.txt）
    model: ""            # 留空自动找目录里的 *.onnx
    num_threads: 2       # 别开太多：这台设备的 CPU 还要留给语音识别和浏览器
    speed: 1.0
"""

import asyncio
import os
import threading
import time
import wave
from io import BytesIO
from typing import Optional

import numpy as np
import sherpa_onnx

from config.logger import setup_logging
from core.providers.tts.base import TTSProviderBase
from core.utils import pcm_gain, tts_phrase_cache
from core.utils.util import decode_audio_bytes

TAG = __name__
logger = setup_logging()

DEFAULT_MODEL_DIR = "~/tts-models/vits-zh-hf-fanchen-C"
# 文本正规化的规则 fst（数字、日期、电话号码等），存在就用上
RULE_FSTS = ("date.fst", "number.fst", "phone.fst", "new_heteronym.fst")


def _find_onnx(model_dir: str) -> Optional[str]:
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".onnx"):
            return os.path.join(model_dir, name)
    return None


def _ints16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class TTSProvider(TTSProviderBase):
    TTS_PARAM_CONFIG = []

    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.audio_file_type = "wav"
        self.model_dir = os.path.expanduser(str(config.get("model_dir") or DEFAULT_MODEL_DIR))
        self.model_path = str(config.get("model") or "").strip() or _find_onnx(self.model_dir)
        self.num_threads = int(config.get("num_threads") or 2)
        self.speed = float(config.get("speed") or 1.0)
        self.speaker_id = int(config.get("speaker_id") or 0)
        self._lock = threading.Lock()
        self._tts = None
        self._sample_rate = 0
        self._init_tts()

    # ---------- 初始化 ----------

    def _init_tts(self) -> None:
        """加载模型。失败就让 self._tts 保持 None，调用时按"合成失败"处理，不影响服务启动。"""
        if not self.model_path or not os.path.exists(self.model_path):
            logger.bind(tag=TAG).error(f"没有找到 TTS 模型文件（目录 {self.model_dir}）")
            return
        tokens = os.path.join(self.model_dir, "tokens.txt")
        lexicon = os.path.join(self.model_dir, "lexicon.txt")
        if not os.path.exists(tokens):
            logger.bind(tag=TAG).error(f"缺少 {tokens}")
            return
        fsts = [os.path.join(self.model_dir, f) for f in RULE_FSTS
                if os.path.exists(os.path.join(self.model_dir, f))]
        t0 = time.time()
        try:
            cfg = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=self.model_path,
                        lexicon=lexicon if os.path.exists(lexicon) else "",
                        tokens=tokens,
                    ),
                    num_threads=self.num_threads,
                    provider="cpu",
                ),
                rule_fsts=",".join(fsts),
                max_num_sentences=1,
            )
            self._tts = sherpa_onnx.OfflineTts(cfg)
            self._sample_rate = self._tts.sample_rate
            # 先合成一句热身：把模型权重真正读进内存，免得第一句真实回答卡一下
            warm = self._tts.generate("你好", sid=self.speaker_id, speed=self.speed)
            logger.bind(tag=TAG).info(
                f"本地 TTS 就绪：{os.path.basename(self.model_path)}，"
                f"采样率 {self._sample_rate}，线程 {self.num_threads}，"
                f"加载+热身 {time.time() - t0:.1f}s，规则 fst {len(fsts)} 个")
            self._warm_samples = len(warm.samples) if warm else 0
        except Exception as e:  # noqa: BLE001
            self._tts = None
            logger.bind(tag=TAG).error(f"加载本地 TTS 模型失败: {e}")

    @property
    def ready(self) -> bool:
        return self._tts is not None

    # ---------- 合成 ----------

    def _pcm_of(self, text: str, sample_rate: int) -> Optional[bytes]:
        """合成一句并转成目标采样率的 16 位单声道 PCM。"""
        if not self.ready:
            raise RuntimeError("本地 TTS 模型没有加载成功")
        with self._lock:  # sherpa-onnx 的 OfflineTts 不是线程安全的
            audio = self._tts.generate(text, sid=self.speaker_id, speed=self.speed)
        if audio is None or len(audio.samples) == 0:
            return None
        samples = np.asarray(audio.samples, dtype=np.float32)
        if audio.sample_rate != sample_rate:
            from core.utils.mp3_decode import _resample  # 项目里现成的多相重采样

            resampled = _resample(samples, audio.sample_rate, sample_rate)
            if resampled is None:
                # 采样率比例太怪（不常见的模型采样率）：退回 ffmpeg 路径
                raw = _ints16(samples)
                import wave as _w

                buf = BytesIO()
                with _w.open(buf, "wb") as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(audio.sample_rate)
                    f.writeframes(raw)
                return decode_audio_bytes(buf.getvalue(), "wav", sample_rate)
            samples = resampled
        return _ints16(samples)

    def _synthesize_pcm(self, text: str, sample_rate: int) -> Optional[bytes]:
        """和基类一致：短句走缓存，之后加增益。只是合成换成本地的。"""
        k = tts_phrase_cache.key(self, text, sample_rate)
        pcm = tts_phrase_cache.get(k)
        if pcm is None:
            pcm = self._pcm_of(text, sample_rate)
            if not pcm:
                return None
            pcm = pcm_gain.apply_gain(pcm, self.post_gain, sample_rate)
            tts_phrase_cache.put(k, pcm)
        return pcm

    async def text_to_speak(self, text, output_file):
        """基类要求实现：没给输出文件时返回音频字节（wav），给了就写文件。"""
        audio = self._tts.generate(text, sid=self.speaker_id, speed=self.speed) if self.ready else None
        if audio is None or len(audio.samples) == 0:
            return None
        buf = BytesIO()
        with wave.open(buf, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(audio.sample_rate)
            f.writeframes(_ints16(np.asarray(audio.samples, dtype=np.float32)))
        data = buf.getvalue()
        if output_file:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "wb") as f:
                f.write(data)
                return None
        return data
