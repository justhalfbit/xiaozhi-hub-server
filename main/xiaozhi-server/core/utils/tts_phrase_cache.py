"""[xiaozhi-hub] 短句的合成结果缓存（解码后的 PCM），以及网关启动时预先合成提示语。

"交给 Grok Bot 了""还在处理，请稍等"这类提示语每轮都一样，却每次都现合成：Edge TTS 约 1 秒，
解码再约 1 秒，而且合成线程只有一条，提示语正在合成时，模型的正式回复只能排在后面等。
缓存 PCM 而不是 Opus 包：Opus 编码器每个连接一份、带状态，用连接自己的编码器现编才能和前后接得上。
"""

import threading
import time
from collections import OrderedDict
from typing import Optional

from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

MAX_CHARS = 16      # 只缓存这么短的句子（提示语、确认语），长回复每次都不一样
MAX_ENTRIES = 64

_cache: "OrderedDict[tuple, bytes]" = OrderedDict()
_lock = threading.Lock()


def key(tts, text: str, sample_rate: int) -> Optional[tuple]:
    """同一种合成、同一个音色和语速音量音调、同一个采样率下的同一句话。太长的句子不缓存，返回 None。"""
    if not text or len(text) > MAX_CHARS:
        return None
    return (
        type(tts).__module__,
        getattr(tts, "voice", None),
        getattr(tts, "speech_rate", None),
        getattr(tts, "volume", None),
        getattr(tts, "pitch_rate", None),
        sample_rate,
        text,
    )


def get(k) -> Optional[bytes]:
    if k is None:
        return None
    with _lock:
        pcm = _cache.get(k)
        if pcm is not None:
            _cache.move_to_end(k)
        return pcm


def put(k, pcm: bytes) -> None:
    if k is None or not pcm:
        return
    with _lock:
        _cache[k] = pcm
        _cache.move_to_end(k)
        while len(_cache) > MAX_ENTRIES:
            _cache.popitem(last=False)


def prompt_phrases(config: dict) -> list:
    """配置里用到的 Agent（Claude Code、Grok Bot）的填充语和第一次进度提示，按 TTS 实际收到的样子整理。"""
    from core.providers.llm.claude_cli.claude_cli import DEFAULT_PROGRESS_REPLY, single_sentence
    from core.providers.llm.grok_bot.grok_bot import DEFAULT_FILLER
    from core.utils.textUtils import get_string_no_punctuation_or_emoji

    defaults = {"claude_cli": "好的，我看一下", "grok_bot": DEFAULT_FILLER}
    phrases = []
    for entry in (config.get("LLM") or {}).values():
        if not isinstance(entry, dict) or entry.get("type") not in defaults:
            continue
        for text in (entry.get("first_reply_filler", defaults[entry["type"]]),
                     entry.get("progress_reply") or DEFAULT_PROGRESS_REPLY):
            text = get_string_no_punctuation_or_emoji(single_sentence(text or ""))
            if text and text not in phrases:
                phrases.append(text)
    return phrases


def prewarm(config: dict, sample_rate: int = 16000) -> None:
    """后台线程里把提示语合成好放进缓存（网关启动时调用）。失败只记日志，用到时再现合成。"""

    def run():
        try:
            from core.providers.tts.dto.dto import InterfaceType
            from core.utils.modules_initialize import initialize_tts

            tts = initialize_tts(config)
            if tts.interface_type != InterfaceType.NON_STREAM or not tts.delete_audio_file or tts.audio_file_type == "p3":
                return  # 流式合成等走自己的路径，不经过这个缓存
            phrases = prompt_phrases(config)
            t0 = time.time()
            for text in phrases:
                tts.synthesize_pcm(text, sample_rate)
            logger.bind(tag=TAG).info(f"提示语已预先合成 {len(phrases)} 句，用时 {time.time() - t0:.1f}s：{phrases}")
        except Exception as e:
            logger.bind(tag=TAG).warning(f"提示语预先合成失败（用到时再合成）：{e}")

    threading.Thread(target=run, name="tts-prewarm", daemon=True).start()
