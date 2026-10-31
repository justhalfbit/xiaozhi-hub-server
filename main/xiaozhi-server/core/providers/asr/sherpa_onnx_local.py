import asyncio
import glob
import hashlib
import time
import wave
import os
import sys
import io
from config.logger import setup_logging
from typing import Optional, Tuple, List
from core.providers.asr.dto.dto import InterfaceType
from core.providers.asr.base import ASRProviderBase

import numpy as np
import sherpa_onnx

# [xiaozhi-hub] modelscope 改为用到时才导入（只有自动下载 SenseVoice 模型时需要）

TAG = __name__
logger = setup_logging()

# [xiaozhi-hub] 新增模型类型需要的文件：{model_type: {sherpa-onnx 参数名: model_dir 下的文件名}}
# 文件名可以带 *（如 icefall 导出的 encoder-epoch-99-avg-1.int8.onnx），有多个匹配时优先 int8
EXTRA_MODEL_FILES = {
    "fire_red_asr": {"encoder": "encoder.int8.onnx", "decoder": "decoder.int8.onnx", "tokens": "tokens.txt"},
    "fire_red_asr_ctc": {"model": "model.int8.onnx", "tokens": "tokens.txt"},
    "funasr_nano": {"encoder_adaptor": "encoder_adaptor.int8.onnx", "llm": "llm.int8.onnx",
                    "embedding": "embedding.int8.onnx", "tokenizer": "Qwen3-0.6B"},
    # qwen3_asr 和 X-ASR（transducer）需要 sherpa-onnx 1.13 以上
    "qwen3_asr": {"conv_frontend": "conv_frontend.onnx", "encoder": "encoder.int8.onnx",
                  "decoder": "decoder.int8.onnx", "tokenizer": "tokenizer"},
    "transducer": {"encoder": "encoder*.onnx", "decoder": "decoder*.onnx", "joiner": "joiner*.onnx",
                   "tokens": "tokens.txt"},
}


def find_model_file(model_dir: str, name: str):
    """[xiaozhi-hub] 在 model_dir 里找文件；name 带 * 时按通配符找，优先 int8。找不到返回 None。"""
    if "*" not in name:
        path = os.path.join(model_dir, name)
        return path if os.path.exists(path) else None
    matches = sorted(glob.glob(os.path.join(model_dir, name)), key=lambda p: (".int8." not in p, p))
    return matches[0] if matches else None


# 捕获标准输出
class CaptureOutput:
    def __enter__(self):
        self._output = io.StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = self._output

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        self.output = self._output.getvalue()
        self._output.close()

        # 将捕获到的内容通过 logger 输出
        if self.output:
            logger.bind(tag=TAG).info(self.output.strip())


class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__()
        self.interface_type = InterfaceType.LOCAL
        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_dir")
        self.model_type = config.get("model_type", "sense_voice")  # 支持 paraformer
        self.delete_audio_file = delete_audio_file
        # [xiaozhi-hub] 边说边识别：句中停顿这么久就先识别前面这段（毫秒，0 关闭），见 ASRProviderBase
        self.segment_pause_ms = int(config.get("segment_pause_ms") or 0)

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # [xiaozhi-hub] 新增的模型类型（见 EXTRA_MODEL_FILES），模型需自行下载到 model_dir
        if self.model_type in EXTRA_MODEL_FILES:
            self.model = self._create_extra_model(config)
            return

        # 初始化模型文件路径
        model_files = {
            "model.int8.onnx": os.path.join(self.model_dir, "model.int8.onnx"),
            "tokens.txt": os.path.join(self.model_dir, "tokens.txt"),
        }

        # 下载并检查模型文件
        try:
            for file_name, file_path in model_files.items():
                if not os.path.isfile(file_path):
                    logger.bind(tag=TAG).info(f"正在下载模型文件: {file_name}")
                    from modelscope.hub.file_download import model_file_download  # [xiaozhi-hub]

                    model_file_download(
                        model_id="pengzhendong/sherpa-onnx-sense-voice-zh-en-ja-ko-yue",
                        file_path=file_name,
                        local_dir=self.model_dir,
                    )

                    if not os.path.isfile(file_path):
                        raise FileNotFoundError(f"模型文件下载失败: {file_path}")

            self.model_path = model_files["model.int8.onnx"]
            self.tokens_path = model_files["tokens.txt"]

        except Exception as e:
            logger.bind(tag=TAG).error(f"模型文件处理失败: {str(e)}")
            raise

        with CaptureOutput():
            if self.model_type == "paraformer":
                self.model = sherpa_onnx.OfflineRecognizer.from_paraformer(
                    paraformer=self.model_path,
                    tokens=self.tokens_path,
                    num_threads=2,
                    sample_rate=16000,
                    feature_dim=80,
                    decoding_method="greedy_search",
                    debug=False,
                )
            else:  # sense_voice
                self.model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=self.model_path,
                    tokens=self.tokens_path,
                    num_threads=2,
                    sample_rate=16000,
                    feature_dim=80,
                    decoding_method="greedy_search",
                    debug=False,
                    use_itn=True,
                )

    def _create_extra_model(self, config: dict):
        """[xiaozhi-hub] 创建上游没有接入的 sherpa-onnx 模型。"""
        files = {}
        for key, name in EXTRA_MODEL_FILES[self.model_type].items():
            path = find_model_file(self.model_dir, name)
            if path is None:
                raise FileNotFoundError(f"缺少模型文件: {os.path.join(self.model_dir, name)}，"
                                        "请从 sherpa-onnx 的 asr-models 发布页下载并解压到 model_dir")
            files[key] = path
        num_threads = int(config.get("num_threads") or 2)
        # funasr_nano / qwen3_asr 支持热词：逗号分隔，帮助认对专有名词
        hotwords = config.get("hotwords") or ""
        if isinstance(hotwords, list):
            hotwords = ",".join(str(w) for w in hotwords)
        recognizer = sherpa_onnx.OfflineRecognizer
        with CaptureOutput():
            if self.model_type == "fire_red_asr":
                return recognizer.from_fire_red_asr(num_threads=num_threads, **files)
            if self.model_type == "fire_red_asr_ctc":
                return recognizer.from_fire_red_asr_ctc(num_threads=num_threads, **files)
            if self.model_type == "transducer":
                return recognizer.from_transducer(num_threads=num_threads, **files,
                                                  **self._transducer_hotwords(hotwords, config))
            if self.model_type == "qwen3_asr":
                if not hasattr(recognizer, "from_qwen3_asr"):
                    raise RuntimeError(f"当前 sherpa-onnx {sherpa_onnx.__version__} 不支持 Qwen3-ASR，需要 1.13 以上")
                return recognizer.from_qwen3_asr(num_threads=num_threads, hotwords=str(hotwords), **files)
            return recognizer.from_funasr_nano(num_threads=num_threads, hotwords=str(hotwords), itn=True, **files)

    def _transducer_hotwords(self, hotwords: str, config: dict) -> dict:
        """[xiaozhi-hub] transducer 的热词：解码时给热词加分（modified_beam_search），不会凭空编出词来。
        BPE 词表的模型需要 model_dir 下有 bpe.vocab（可用 sentencepiece 从 bpe.model 导出）。"""
        words = [w.strip() for w in hotwords.split(",") if w.strip()]
        if not words:
            return {}
        bpe_vocab = os.path.join(self.model_dir, "bpe.vocab")
        has_bpe = os.path.isfile(bpe_vocab) or self._export_bpe_vocab(bpe_vocab)
        content = "\n".join(words) + "\n"
        # 按内容取名：热词不变时每次启动复用同一个文件，不会在 output_dir 里越积越多
        path = os.path.join(self.output_dir, f"hotwords-{hashlib.sha1(content.encode()).hexdigest()[:10]}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {
            "decoding_method": "modified_beam_search",
            "hotwords_file": path,
            "hotwords_score": float(config.get("hotwords_score") or 1.5),
            "modeling_unit": config.get("modeling_unit") or ("bpe" if has_bpe else "cjkchar"),
            "bpe_vocab": bpe_vocab if has_bpe else "",
        }

    def _export_bpe_vocab(self, bpe_vocab: str) -> bool:
        """[xiaozhi-hub] 模型包只带 bpe.model 时，用 sentencepiece 导出 sherpa-onnx 热词需要的 bpe.vocab。"""
        bpe_model = os.path.join(self.model_dir, "bpe.model")
        if not os.path.isfile(bpe_model):
            return False
        try:
            import sentencepiece

            sp = sentencepiece.SentencePieceProcessor(model_file=bpe_model)
            tmp = bpe_vocab + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for i in range(sp.get_piece_size()):
                    f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")
            os.replace(tmp, bpe_vocab)
            logger.bind(tag=TAG).info(f"已从 bpe.model 生成 {bpe_vocab}")
            return True
        except Exception as e:
            logger.bind(tag=TAG).warning(f"生成 bpe.vocab 失败，热词可能不起作用: {e}")
            return False

    def read_wave(self, wave_filename: str) -> Tuple[np.ndarray, int]:
        """
        Args:
        wave_filename:
            Path to a wave file. It should be single channel and each sample should
            be 16-bit. Its sample rate does not need to be 16kHz.
        Returns:
        Return a tuple containing:
        - A 1-D array of dtype np.float32 containing the samples, which are
        normalized to the range [-1, 1].
        - sample rate of the wave file
        """

        with wave.open(wave_filename) as f:
            assert f.getnchannels() == 1, f.getnchannels()
            assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
            num_samples = f.getnframes()
            samples = f.readframes(num_samples)
            samples_int16 = np.frombuffer(samples, dtype=np.int16)
            samples_float32 = samples_int16.astype(np.float32)

            samples_float32 = samples_float32 / 32768
            return samples_float32, f.getframerate()

    def requires_file(self) -> bool:
        return True

    def recognize_pcm(self, pcm: bytes) -> str:
        """[xiaozhi-hub] 直接识别一段 16kHz 单声道 16 位 PCM，只返回文字（不纠错、不进对话）。
        给本机设备接口 /kiosk/asr 用；模型所有连接共用，每次新建 stream，可以在线程池里并发调用。"""
        samples = np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
        s = self.model.create_stream()
        s.accept_waveform(16000, samples)
        self.model.decode_stream(s)
        return s.result.text

    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, artifacts=None
    ) -> Tuple[Optional[str], Optional[str]]:
        """语音转文本主处理逻辑"""
        file_path = None
        try:
            if artifacts is None:
                return "", None
            file_path = artifacts.file_path

            start_time = time.time()
            s = self.model.create_stream()
            samples, sample_rate = self.read_wave(file_path)
            s.accept_waveform(sample_rate, samples)
            # [xiaozhi-hub] 放到线程里解码：长句要好几秒，期间主事件循环（其他连接、后台接口）不被卡住
            await asyncio.to_thread(self.model.decode_stream, s)
            text = s.result.text
            logger.bind(tag=TAG).debug(
                f"语音识别耗时: {time.time() - start_time:.3f}s | 结果: {text}"
            )

            return text, file_path

        except Exception as e:
            logger.bind(tag=TAG).error(f"语音识别失败: {e}", exc_info=True)
            return "", file_path
