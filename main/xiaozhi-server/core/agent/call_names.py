"""呼名：句首"嗨 + 称呼"（如"嗨 Claude"）指定由哪个模型回答。

称呼对应 LLM 配置里的条目（如 ClaudeCLI、DeepSeekLLM），叫到后这台设备就切到这个模型并记住，
之后直接说话也交给它（见 llm_router）。称呼在交给模型前去掉；整句话只有称呼时，回一句固定的话，
不发给模型，也不启动 Claude Code 进程，这样下一句"接着上次聊"仍然是这个模型的第一句话，照常接续。

默认别名来自实测（工作空间 asr-test/call-names）：语音识别把 Claude 全部识别成 "cloud"，
把"嗨"识别成 嗨/嘿/hi/hey/high/hiigh。别名也可以在管理后台管理：开启"录入"后对着设备说几遍，
把语音识别实际听到的结果存成别名（Learner）。
"""

import asyncio
import hashlib
import re
import struct
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from config.config_loader import get_project_dir

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__

DEFAULT_MODELS = {"ClaudeCLI": ["claude", "cloud", "clod", "clot", "克劳德"]}
DEFAULT_REPLY = "我在"
# 发缓存的音频前等一会儿：设备收到 tts start 后要先切到说话状态（会清空解码队列），切换完成前收到的音频会被丢掉。
# 正常合成时中间有约 2 秒的合成时间，不会碰到；缓存的音频紧跟着就到，不等的话设备只显示文字、没有声音。
# 在 hello 里声明 ready_on_tts_start 的设备按顺序处理消息和音频，收到 tts start 马上就能播，不用等
CACHED_REPLY_DELAY = 0.5


def one_segment(text: str) -> str:
    """把回复整理成 TTS 只合成一段的形式：句中的标点会把一句话切成几段、逐段合成播放，听起来拖沓。"""
    return re.sub(r"\s+", " ", re.sub(r"[，,、~～；;：:。.！!？?]+", " ", text or "")).strip()

# "嘿"常被识别成 he，有时后面还带一个日文假名，偶尔只剩一个 h（实测 "He cloud."、"Heい cloud。"、"H, cloud."）
_PREFIXES = ["hiigh", "high", "hey", "hei", "hai", "hi", "he", "h", "嗨", "嘿"]
# 单个 h 后面必须是停顿，否则 "hello"、"haiku" 这类词开头的 h 会被当成招呼语
_PREFIX = "(?:" + "|".join(p if p != "h" else r"h(?=[\s,，.。!！?？、~～])" for p in _PREFIXES) + r")[\u3040-\u30ff]*"
_SEP = r"[\s,，.。!！?？、~～]*"


def _alias_pattern(alias: str) -> str:
    # 称呼后面不能紧跟（或隔着空格跟）英文字母：避免"cloudflare"、"Claude Code 怎么用"被当成称呼。
    # 中文句子里称呼后面通常是标点或中文；"Hi Claude what's up"这种不带逗号的英文句子会原样交给模型
    return re.escape(alias) + r"(?!\s*[a-z])"


def _aliases(value) -> list:
    """配置里的别名应是列表；写成单个字符串时当作一个别名，不能拆成单个字符。"""
    if isinstance(value, str):
        value = [value]
    return [str(a).strip().lower() for a in (value or []) if str(a).strip()]


def _build(models: dict) -> list:
    prefix = _PREFIX
    result = []
    for model, value in models.items():
        aliases = _aliases(value)
        if aliases:
            alternatives = "|".join(_alias_pattern(a) for a in sorted(aliases, key=len, reverse=True))
            result.append((model, re.compile(r"^" + _SEP + prefix + _SEP + "(?:" + alternatives + ")" + _SEP,
                                             re.IGNORECASE)))
    return result


def match_call_name(text: str, models: dict = None):
    """句首是称呼时返回 (模型, 去掉称呼后的文字)，文字为空表示整句话只有称呼；不是称呼返回 None。

    连说两遍称呼（实测出现过 "Hi, cloud. Hi, cloud."）也一并去掉。
    """
    patterns = _build(models or DEFAULT_MODELS)
    found = None
    rest = text or ""
    while True:
        for model, pattern in patterns:
            m = pattern.match(rest)
            if m:
                found, rest = model, rest[m.end():]
                break
        else:
            break
    if found is None:
        return None
    return found, rest.strip()


def extract_alias(text: str) -> str:
    """从一次录入的识别结果里取出称呼：去掉开头的"嗨/hi"等和标点，取到下一个停顿为止。

    例如 "嗨cloud。" → "cloud"，"Hi, cloud. Hi, cloud." → "cloud"，"克劳德" → "克劳德"。
    """
    rest = re.sub(r"^" + _SEP + _PREFIX + _SEP, "", (text or "").strip(), count=1,
                  flags=re.IGNORECASE)
    rest = re.sub(r"^" + _SEP, "", rest)
    return re.split(_SEP.replace("*", "+"), rest, maxsplit=1)[0].strip().lower()


class Learner:
    """录入模式：管理后台开启后，接下来设备说的几句话不交给模型，只记录语音识别的结果，作为候选别名。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = None

    def start(self, session_id=None, count: int = 3, timeout: float = 90, target: str = None) -> None:
        """target：给哪个模型录入（只是记下来给后台页面用）。"""
        with self._lock:
            self._state = {"session_id": session_id, "count": count, "until": time.time() + timeout, "samples": [],
                           "target": target}

    def stop(self) -> None:
        with self._lock:
            if self._state:
                self._state["until"] = 0

    def status(self) -> dict:
        with self._lock:
            st = self._state
            if not st:
                return {"active": False, "samples": [], "count": 0, "session_id": None, "seconds_left": 0,
                        "target": None}
            left = max(0, int(st["until"] - time.time()))
            active = left > 0 and len(st["samples"]) < st["count"]
            return {"active": active, "samples": list(st["samples"]), "count": st["count"],
                    "session_id": st["session_id"], "seconds_left": left if active else 0,
                    "target": st.get("target")}

    def capture(self, session_id: str, text: str):
        """录入中且是目标设备时记下这句话，返回 (第几遍, 一共几遍)；否则返回 None。"""
        with self._lock:
            st = self._state
            if not st or time.time() > st["until"] or len(st["samples"]) >= st["count"]:
                return None
            if st["session_id"] and st["session_id"] != session_id:
                return None
            st["samples"].append({"text": text, "alias": extract_alias(text)})
            return len(st["samples"]), st["count"]


learner = Learner()


# 固定回复合成好的音频：(TTS 类型, 音色, 采样率, 文字) -> opus 包。EdgeTTS 合成一句要 1～2 秒，缓存后直接发。
# 同时存到 data/call_reply_audio/，服务重启后不用重新合成；连接建立后 warm_up 会提前合成好（见 warm_up）
_audio_cache = {}
_filling = {}  # 正在合成的：key -> Task
_cache_dir = Path(get_project_dir()) / "data" / "call_reply_audio"


def _cache_key(conn: "ConnectionHandler", text: str):
    return (type(conn.tts).__name__, getattr(conn.tts, "voice", None), getattr(conn, "sample_rate", None), text)


def _cache_file(key) -> Path:
    return _cache_dir / (hashlib.sha1(repr(key).encode("utf-8")).hexdigest() + ".opus")


def _load_cached(key):
    """内存里没有时从磁盘读回（每个包前面 4 字节长度）。"""
    packets = _audio_cache.get(key)
    if packets is not None:
        return packets
    try:
        data = _cache_file(key).read_bytes()
    except OSError:
        return None
    packets, i = [], 0
    while i + 4 <= len(data):
        (n,) = struct.unpack_from(">I", data, i)
        packets.append(data[i + 4:i + 4 + n])
        i += 4 + n
    if i != len(data) or not packets:
        return None  # 文件不完整：当作没有，重新合成
    _audio_cache[key] = packets
    return packets


def _save_cached(key, packets) -> None:
    _cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_file(key)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(b"".join(struct.pack(">I", len(p)) + bytes(p) for p in packets))
    tmp.replace(path)


async def _fill_cache(conn: "ConnectionHandler", text: str) -> None:
    key = _cache_key(conn, text)
    if key in _filling:
        return await asyncio.shield(_filling[key])

    async def fill():
        try:
            packets = await asyncio.to_thread(conn.tts.to_tts, text)
            if packets:
                _audio_cache[key] = packets
                await asyncio.to_thread(_save_cached, key, packets)
        except Exception as e:
            conn.logger.bind(tag=TAG).warning(f"缓存呼名回复的音频失败: {e}")
        finally:
            _filling.pop(key, None)

    _filling[key] = asyncio.ensure_future(fill())
    await asyncio.shield(_filling[key])


def reply_text(cfg: dict) -> str:
    """只喊称呼时回的话。"""
    return one_segment((cfg or {}).get("reply") or "") or DEFAULT_REPLY


async def warm_up(conn: "ConnectionHandler") -> None:
    """连接建立后在后台准备好只喊称呼时的回复：磁盘上有就读进内存，没有就等语音合成就绪后合成一次。
    第一次只喊称呼时就不用现合成了（网关重启后的第一次也一样）。"""
    cfg = conn.config.get("call_names") or {}
    if not cfg.get("enabled", True):
        return
    deadline = time.monotonic() + 30
    while not conn.tts and time.monotonic() < deadline:
        await asyncio.sleep(0.2)
    if not conn.tts:
        return
    text = reply_text(cfg)
    if await asyncio.to_thread(_load_cached, _cache_key(conn, text)) is None:
        await _fill_cache(conn, text)


def schedule_warm_up(conn: "ConnectionHandler") -> None:
    """可以在任意线程里调用（llm_router.on_conn_open 在线程池里执行）。"""
    loop = getattr(conn, "loop", None)
    if loop is not None and loop.is_running() and getattr(conn, "config", None) is not None:
        asyncio.run_coroutine_threadsafe(warm_up(conn), loop)


async def _speak(conn: "ConnectionHandler", heard: str, reply: str, cache: bool = False) -> bool:
    """不经过大模型，直接回一句话。TTS 还没就绪时返回 False。

    cache=True 用于固定的回复：第一次照常合成，同时在后台把音频缓存起来，之后直接发缓存的音频。
    """
    from core.handle.intentHandler import speak_txt
    from core.handle.sendAudioHandle import sendAudioMessage, send_stt_message
    from core.providers.tts.dto.dto import SentenceType
    from core.utils.dialogue import Message

    # 刚连上时 TTS 可能还在初始化（和唤醒词回复一样，最多等 3 秒）
    deadline = time.monotonic() + 3
    while not conn.tts and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    if not conn.tts:
        return False
    await send_stt_message(conn, heard)
    conn.client_abort = False
    conn.sentence_id = str(uuid.uuid4().hex)
    conn.dialogue.put(Message(role="user", content=heard))
    packets = None
    if cache:
        key = _cache_key(conn, reply)
        if key in _filling:  # warm_up 正在合成：等它（比重新合成一遍快）
            try:
                await asyncio.wait_for(asyncio.shield(_filling[key]), timeout=3)
            except Exception:
                pass
        packets = _load_cached(key)
    if packets:
        # 和唤醒词回复的播放方式一样
        if not (getattr(conn, "features", None) or {}).get("ready_on_tts_start"):
            await asyncio.sleep(CACHED_REPLY_DELAY)
        await sendAudioMessage(conn, SentenceType.FIRST, packets, reply)
        await sendAudioMessage(conn, SentenceType.LAST, [], None)
        conn.dialogue.put(Message(role="assistant", content=reply))
        return True
    speak_txt(conn, reply)
    if cache:
        asyncio.create_task(_fill_cache(conn, reply))
    return True


def _is_wakeup_word(conn: "ConnectionHandler", text: str) -> bool:
    from core.utils.util import remove_punctuation_and_length

    _, filtered = remove_punctuation_and_length(text)
    return filtered in (conn.config.get("wakeup_words") or [])


async def handle_call_name(conn: "ConnectionHandler", text: str):
    """在意图识别和大模型之前处理称呼（以及管理后台开启的录入）。

    返回 None 表示这句话已经处理完（录入了一遍，或只有称呼，已回复）；否则返回要交给模型的文字。
    """
    if not text.lstrip().startswith("{") and not _is_wakeup_word(conn, text):
        got = learner.capture(getattr(conn, "session_id", None), text)
        if got:
            n, total = got
            conn.logger.bind(tag=TAG).info(f"呼名录入 第 {n}/{total} 遍：{text}")
            await _speak(conn, text, f"收到第 {n} 遍" if n < total else "录好了，请到管理后台确认")
            return None

    cfg = conn.config.get("call_names") or {}
    if not cfg.get("enabled", True):
        return text
    hit = match_call_name(text, cfg.get("models") or DEFAULT_MODELS)
    if hit is None:
        return text
    from core.agent import llm_router

    model, rest = hit
    model = llm_router.entry_name(model)
    try:
        ok = await llm_router.switch_async(conn, model)
    except Exception as e:
        conn.logger.bind(tag=TAG).error(f"呼名 {model}：切换模型失败: {e}")
        ok = False
    if not ok:
        conn.logger.bind(tag=TAG).warning(f"呼名 {model}：这个模型没有配置好或加载失败，不切换")
        if await _speak(conn, text, f"{llm_router.model_label(model)} 还没配置好，请到管理后台检查"):
            return None
        return rest or text
    if rest:
        conn.logger.bind(tag=TAG).info(f"呼名 {model}：去掉称呼后交给模型：{rest}")
        return rest

    reply = reply_text(cfg)
    # 切回一个在外面干活的 Bot（Grok Bot）：它做完的活还没念过的，跟在"我在"后面一起念
    bot_id, pending = _pending_report(conn, model)
    if pending:
        reply = f"{reply}。{pending}"
    if not await _speak(conn, text, reply, cache=not pending):
        conn.logger.bind(tag=TAG).warning("呼名：TTS 尚未就绪，无法回复，按普通对话处理")
        if pending:
            from core.agent import reports

            reports.add(reports.device_key(conn), bot_id, pending)  # 没念成，放回去下次念
        return text
    conn.logger.bind(tag=TAG).info(f"呼名 {model}：只有称呼，直接回复：{reply[:60]}")
    return None


def _pending_report(conn: "ConnectionHandler", model: str):
    """这个模型是 Grok Bot 且有做完还没念的活：返回 (bot_id, 要念的话)；否则 ("", "")。"""
    from core.agent import llm_router, reports

    entry = (llm_router._server_config(conn).get("LLM") or {}).get(model) or {}
    bot_id = str(entry.get("bot_id") or "").strip() if isinstance(entry, dict) else ""
    return bot_id, (reports.take_text(reports.device_key(conn), bot_id) if bot_id else "")
