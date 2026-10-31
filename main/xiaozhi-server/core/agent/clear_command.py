"""语音清除聊天记录："清除聊天记录"之类的话先问一遍，说"确定"才清（chat_history.clear，和聊天页设置里的按钮同一个接口）。

在呼名之后、意图识别和大模型之前处理，所以 Claude Code、Grok Bot 也能用。指令、确认和回复都不写进对话。
问了之后 CONFIRM_SECONDS 秒内：说确认的话就清，说否定的话就不清，说别的就当没问过、照常交给模型。
"""

import asyncio
import re
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__

CONFIRM_SECONDS = 30
ASK = "确定要清除全部聊天记录吗？清除后我会忘掉之前聊过的内容。说确定就清除"
DONE = "好的，聊天记录已经清除了"
KEPT = "好的，不清除"

_STRIP = re.compile(r"[\s，。！？、,.!?；;：:~～\"'“”‘’…]+")
_VERB = "(?:清除|清空|删除|删掉|清掉|清理|抹掉|删了|清了)"
_OBJ = "(?:全部|所有|一下|我们的|咱们的|我的|和你的|跟你的)?(?:聊天|对话|历史|会话|消息)(?:记录|历史|内容)?"
# "清除聊天记录"、"把聊天记录清空"、"聊天记录全部删掉"；前面带否定的不算（"别清除聊天记录"）
_CLEAR = re.compile(rf"(?:^|[^不别])(?:{_VERB}{_OBJ}|把?{_OBJ}(?:都|全部|全)?{_VERB})")
_CONFIRM = re.compile(r"^(?:嗯|对|是|好|行|可以|没问题|确定|确认|清除|清空|删除|删吧|清吧|好的|是的|对的|确定清除|确认清除|清除吧|删掉吧)+(?:吧|啊|呀|的)?$")
_CANCEL = re.compile(r"(?:不要|不用|别|算了|取消|不清|不删|再想想|等一下|先不)")


def normalize(text: str) -> str:
    return _STRIP.sub("", text or "").lower()


def is_clear(text: str) -> bool:
    t = normalize(text)
    return len(t) <= 20 and bool(_CLEAR.search(t))


def is_confirm(text: str) -> bool:
    t = normalize(text)
    return bool(t) and not _CANCEL.search(t) and bool(_CONFIRM.match(t))


def is_cancel(text: str) -> bool:
    return bool(_CANCEL.search(normalize(text)))


def result_text(result: dict) -> str:
    """清除结果的语音回复（Grok Bot 的云端对话另外说一句）。"""
    parts = [DONE]
    for g in result.get("grok") or []:
        if g.get("reset"):
            parts.append(f"{g['label']} 在云端的对话也重新开始了")
        elif g.get("pending"):
            parts.append(f"{g['label']} 在云端的对话正在重置")
        else:
            parts.append(f"{g['label']} 在云端的对话没能重置，还在云端")
    return "。".join(parts)


async def handle(conn: "ConnectionHandler", text: str):
    """返回 None 表示这句话已经处理完；否则返回要继续交给模型的文字。"""
    if text.lstrip().startswith("{"):
        return text
    pending = getattr(conn, "chat_clear_asked", 0)
    if pending:
        conn.chat_clear_asked = 0
        if time.monotonic() - pending < CONFIRM_SECONDS:
            if is_confirm(text):
                await _speak(conn, text, result_text(await clear_now(conn)))
                return None
            if is_cancel(text):
                await _speak(conn, text, KEPT)
                return None
    if is_clear(text):
        conn.logger.bind(tag=TAG).info(f"语音清除聊天记录，先确认：{text}")
        if await _speak(conn, text, ASK):
            conn.chat_clear_asked = time.monotonic()
            return None
    return text


async def clear_now(conn: "ConnectionHandler") -> dict:
    from core.admin import connections
    from core.agent import chat_history, llm_router

    device = chat_history.device_of(conn)
    if not device:
        # 没有 device-id 的连接没存过记录：只清这个连接的对话
        chat_history.clear_dialogues([conn])
        return {"deleted": 0, "grok": []}
    online = [c for c in (connections.get(i["session_id"]) for i in connections.snapshot()
                          if i["device_id"] == device) if c is not None]
    if conn not in online:
        online.append(conn)
    config = llm_router._server_config(conn)
    return await asyncio.get_running_loop().run_in_executor(
        None, lambda: chat_history.clear(device, config, connections=online))


async def _speak(conn: "ConnectionHandler", heard: str, reply: str) -> bool:
    """不经过大模型直接回一句，也不写进对话。TTS 还没就绪时返回 False。"""
    from core.handle.sendAudioHandle import send_stt_message
    from core.providers.tts.dto.dto import ContentType, SentenceType, TTSMessageDTO

    deadline = time.monotonic() + 3
    while not conn.tts and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    if not conn.tts:
        return False
    await send_stt_message(conn, heard)
    conn.client_abort = False
    conn.sentence_id = str(uuid.uuid4().hex)
    conn.tts.store_tts_text(conn.sentence_id, reply)
    conn.tts.tts_text_queue.put(TTSMessageDTO(sentence_id=conn.sentence_id, sentence_type=SentenceType.FIRST,
                                              content_type=ContentType.ACTION))
    conn.tts.tts_one_sentence(conn, ContentType.TEXT, content_detail=reply)
    conn.tts.tts_text_queue.put(TTSMessageDTO(sentence_id=conn.sentence_id, sentence_type=SentenceType.LAST,
                                              content_type=ContentType.ACTION))
    return True
