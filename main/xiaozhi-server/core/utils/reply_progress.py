"""被打断的回复只把播出去的部分记进上下文。

原来模型生成多少就记多少：回复一般生成得比播得快，打断时整段早已写进对话，模型以为用户全听到了。

做法：
- 每轮记下每句话在这轮音频里从第几毫秒开始（音频按每包 60ms 发，sendAudioMessage 的顺序就是播放顺序）；
- 打断时设备报告这一轮已经播了多少毫秒（abort 消息里的 played_ms）；没报告的设备按"已发出的音频 − 估算的设备缓冲"算；
- 播完的句子整句保留，正在播的那句按已播时长占这句时长的比例截字（近似按字），末尾加"……"；
- 对话里这一轮的助手回复换成这段话，一个字都没播出时记"……"，保持一问一答。

回复还没生成完就被打断时，chat() 之后才会写对话：打断时先记下，chat() 结束（finish_turn）时再换；
打断之后这一轮不再往对话里写回复（can_store），免得旧的一轮在新问题后面补上整段回复。

只对经过对话记录的模型有效。Claude Code、Grok Bot 的上下文在它们自己的会话里，管不到。
"""

import threading
import time

from core.utils.dialogue import Message

FRAME_MS = 60
KEEP_TURNS = 5
CUT_MARK = "……"
PUNCT = set("，。！？；：、,.!?;:…")

_lock = threading.Lock()


def _turns(conn) -> dict:
    turns = getattr(conn, "reply_turns", None)
    if turns is None:
        turns = conn.reply_turns = {}
    return turns


def note_turn(conn, sentence_id, user_msg) -> None:
    """chat() 开始一轮：记下这轮用户的那句话（对话里找这一轮从它开始）。"""
    with _lock:
        turns = _turns(conn)
        turns[sentence_id] = {"user": user_msg, "done": False, "cut": None}
        for old in list(turns)[:-KEEP_TURNS]:
            del turns[old]


def note_audio(conn, sentence_id, text, audios, last=False) -> None:
    """一段回复音频要发给设备了（text 不为空表示新的一句从这里开始，last 表示这轮的音频到此为止）。"""
    sid = sentence_id or conn.sentence_id
    st = getattr(conn, "reply_audio", None)
    if st is None or st["sid"] != sid:
        st = conn.reply_audio = {"sid": sid, "sentences": [], "total_ms": 0, "ended": False}
    st["ended"] = st["ended"] or last
    if text:
        st["sentences"].append([st["total_ms"], str(text)])
    if audios:
        st["total_ms"] += FRAME_MS * (1 if isinstance(audios, (bytes, bytearray)) else len(audios))


def estimate_played_ms(conn) -> float:
    """设备没报告时：这轮已发出的音频减去估算的设备缓冲。"""
    fc = getattr(conn, "audio_flow_control", None) or {}
    if fc.get("sentence_id") != conn.sentence_id:
        return 0.0
    ahead = max(fc.get("play_end", 0) - time.monotonic(), 0) * 1000
    return max(fc.get("packet_count", 0) * FRAME_MS - ahead, 0.0)


def heard_text(conn, played_ms):
    """这一轮已经播出去的文字和整轮有没有播完：(text, complete)。

    后面的句子常常还没合成出来：已发出的都播完了也不算播完，要这轮的音频发到了最后（ended）。"""
    st = getattr(conn, "reply_audio", None)
    if st is None or st["sid"] != conn.sentence_id:
        return "", False
    sentences, total = st["sentences"], st["total_ms"]
    parts = []
    for i, (start, text) in enumerate(sentences):
        end = sentences[i + 1][0] if i + 1 < len(sentences) else total
        if played_ms >= end:
            parts.append(text)
            continue
        if played_ms > start and end > start:
            n = int(len(text) * (played_ms - start) / (end - start))
            if n > 0:
                parts.append(text[:n])
        return _join(parts), False
    return _join(parts), st["ended"]


def _join(parts) -> str:
    # 合成时每句去掉了句末标点，拼回去时补一个逗号
    out = ""
    for p in parts:
        out += ("" if not out or out[-1] in PUNCT else "，") + p
    return out


def cut(conn, played_ms=None) -> None:
    """设备打断了这一轮：上下文里这轮的回复只留播出去的部分。"""
    sid = conn.sentence_id
    try:
        played = float(played_ms) if played_ms is not None else estimate_played_ms(conn)
    except (TypeError, ValueError):
        played = estimate_played_ms(conn)
    heard, complete = heard_text(conn, played)
    if complete:
        return
    heard = heard + CUT_MARK if heard else CUT_MARK
    if getattr(conn, "logger", None):
        conn.logger.bind(tag=__name__).info(f"回复被打断：已播 {played / 1000:.1f}s，上下文只记「{heard}」")
    with _lock:
        turn = _turns(conn).get(sid)
        if turn is None:
            return
        turn["cut"], turn["interrupted"] = heard, True
        if turn["done"]:
            _apply(conn, turn)


def can_store(conn, sentence_id) -> bool:
    """这一轮的回复还能不能写进对话（被打断后不能，由 cut/finish_turn 写播出去的部分）。"""
    with _lock:
        turn = _turns(conn).get(sentence_id)
        return turn is None or not turn.get("interrupted")


def finish_turn(conn, sentence_id) -> None:
    """chat() 这一轮结束了（回复都已写进对话）。"""
    with _lock:
        turn = _turns(conn).get(sentence_id)
        if turn is None:
            return
        turn["done"] = True
        if turn["cut"] is not None:
            _apply(conn, turn)


def _apply(conn, turn) -> None:
    messages = conn.dialogue.dialogue
    start = next((i for i, m in enumerate(messages) if m is turn["user"]), None)
    if start is None:
        return
    end = next((i for i in range(start + 1, len(messages)) if messages[i].role == "user"), len(messages))
    replies = [m for m in messages[start + 1:end] if m.role == "assistant" and m.tool_calls is None]
    if replies:
        for m in replies[:-1]:
            messages.remove(m)
        replies[-1].content = turn["cut"]
    else:
        messages.insert(end, Message(role="assistant", content=turn["cut"]))
    turn["cut"] = None
