"""回声识别过滤：丢掉把设备自己的回复识别成"用户说话"的结果。

开着回声消除的设备（hello 里带 aec）在回复播放时也开着麦克风（realtime 模式），方便用户随时插话。
残留的回声很轻，但语音识别有时仍会把它认成回复里的一小段（例如"也是。"），被当成用户插话，把回复打断。

做法：回复播放期间和播完后几秒内，识别结果里大部分字都出现在这次回复里，就当回声丢掉。
只看字，不看声音大小：回声的音量和用户离设备远近、音量设置都有关，定不出一个可靠的门限。
"""

import re
import time
from difflib import SequenceMatcher

import numpy as np

# 最后一段回复音频发出后还算"回复期间"的秒数：设备还要把缓冲播完，回声约 0.4 秒后到麦克风，
# VAD 又要等 0.8 秒安静才把这句交给识别
WINDOW_AFTER = 4.0
MIN_SHARE = 0.6  # 识别出的字里，至少这么大比例也出现在回复里，才算回声
_NON_WORD = re.compile(r"[\W_]+")


def _state(conn) -> dict:
    st = getattr(conn, "reply_echo", None)
    if st is None:
        st = {"sid": None, "text": "", "at": 0.0}
        conn.reply_echo = st
    return st


def note_reply(conn, text) -> None:
    """一句回复正在发给设备（text 为 None 表示这轮的音频发完了，只更新时间）。"""
    st = _state(conn)
    if st["sid"] != conn.sentence_id:  # 新的一轮回复：忘掉上一轮的
        st.update(sid=conn.sentence_id, text="")
    if text:
        st["text"] += _NON_WORD.sub("", str(text))
    st["at"] = time.monotonic()


def during_reply(conn) -> bool:
    st = _state(conn)
    return bool(st["text"]) and (conn.client_is_speaking or time.monotonic() - st["at"] <= WINDOW_AFTER)


def is_echo(conn, text) -> bool:
    """回复期间识别出的这句话是不是回复自己的回声。"""
    if not during_reply(conn):
        return False
    heard = _NON_WORD.sub("", str(text))
    if not heard:
        return False
    blocks = SequenceMatcher(None, heard, _state(conn)["text"], autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) / len(heard) >= MIN_SHARE


def level_db(frames) -> float:
    """这句话里最响的 60ms 的音量（16kHz PCM 帧），单位 dBFS。只用于日志，方便调参时对照。"""
    pcm = np.frombuffer(b"".join(frames), dtype="<i2").astype(np.float32) / 32768
    if not len(pcm):
        return -120.0
    step = 960
    rms = [np.sqrt(np.mean(pcm[i:i + step] ** 2)) for i in range(0, len(pcm), step)]
    return float(20 * np.log10(max(max(rms), 1e-6)))
