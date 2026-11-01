import json
import queue
import uuid
from typing import TYPE_CHECKING

from core.utils import reply_progress, turn_gen

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
TAG = __name__


async def handleAbortMessage(conn: "ConnectionHandler", played_ms=None, drop_audio=False):
    """played_ms：设备报告的这一轮回复已经播了多少毫秒（没有时按发送进度估算）。
    drop_audio：设备自己发的打断（唤醒词、按钮）：说完还没识别完的那句话、排队没处理的录音都不要了。
    服务端回声消除和新问题触发的打断不丢，那时缓冲里是用户新说的话。"""
    conn.logger.bind(tag=TAG).info("Abort message received")
    # [xiaozhi-hub] 上下文只记已经播出去的部分；要在清队列之前算（用到发送进度）
    if conn.client_is_speaking:
        reply_progress.cut(conn, played_ms)
    # [xiaozhi-hub] 旧的一轮作废：还在识别的那句、模型还在出的字、合成里的句子迟到了一律丢掉
    # （新的一轮开始时会把 client_abort 清回 False，不能只靠它）
    turn_gen.bump(conn)
    conn.sentence_id = uuid.uuid4().hex
    if drop_audio:
        _drop_pending_audio(conn)
    # 设置成打断状态，会自动打断llm、tts任务
    conn.close_after_chat = False
    conn.client_abort = True
    # [xiaozhi-hub] 让 LLM 立即停止当前回答（claude_cli 向 Claude Code 发送中断）
    if callable(getattr(conn.llm, "abort", None)):
        conn.llm.abort(conn)
    conn.clear_queues()
    # 打断客户端说话状态
    await conn.websocket.send(
        json.dumps({"type": "tts", "state": "stop", "interrupted": True, "session_id": conn.session_id})
    )
    conn.clearSpeakStatus()
    conn.logger.bind(tag=TAG).info("Abort message received-end")


def _drop_pending_audio(conn: "ConnectionHandler") -> None:
    """识别中被打断：识别线程这时卡在识别上，后面收到的录音（常常就是唤醒词）还排着队，
    不清掉就会拼进打断后的新问题。"""
    dropped = 0
    while True:
        try:
            conn.asr_audio_queue.get_nowait()
            dropped += 1
        except queue.Empty:
            break
    conn.reset_audio_states()
    if dropped:
        conn.logger.bind(tag=TAG).info(f"打断：丢掉排队未识别的录音 {dropped} 帧")
