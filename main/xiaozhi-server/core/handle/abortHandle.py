import json
from typing import TYPE_CHECKING

from core.utils import reply_progress

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
TAG = __name__


async def handleAbortMessage(conn: "ConnectionHandler", played_ms=None):
    """played_ms：设备报告的这一轮回复已经播了多少毫秒（没有时按发送进度估算）。"""
    conn.logger.bind(tag=TAG).info("Abort message received")
    # [xiaozhi-hub] 上下文只记已经播出去的部分；要在清队列之前算（用到发送进度）
    if conn.client_is_speaking:
        reply_progress.cut(conn, played_ms)
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
