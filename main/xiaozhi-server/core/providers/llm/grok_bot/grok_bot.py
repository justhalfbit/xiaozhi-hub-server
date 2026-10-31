"""grok_bot：让小智设备给一个 Grok Bot（xAI 与 Cursor 的云端 AI 同事）派活，并把它的回复念出来。

一个配置条目对应一个 Bot（bot_id、bot_name），管理后台的"Grok Bot"页勾选 Bot 后自动生成条目，
Bot 的名字自动成为称呼。同一个账号的所有 Bot 共用一条事件流（见 grokbot_client.get_hub）。

连接管理、空闲关闭、填充语、进度提示、超时都继承 claude_cli 的 provider，只换掉会话的创建方式：
- 没有本机进程，"启动会话"只是确认云端电脑醒着并订阅事件流；设备叫到这个 Bot 时就提前做（热身），
  第一句话不用再等云端电脑唤醒；
- 对话记录在云端，不需要接续逻辑；
- 被打断（含等超时）后 Bot 在云端继续做，做完的回复进待汇报箱（core/agent/reports.py），设备空闲时念；
  云端电脑还在唤醒时就被打断或断开，这句话也照样发出去，发完直接转后台盯着（hand_off）；
- 所有 grok_bot 实例共用一个后台事件循环（事件流中枢必须和会话在同一个循环里）。

凭证由管理后台登录后写到 data/grokbot_credentials.json，这里只读。
"""

import asyncio
import os
import threading
import time
from pathlib import Path

from config.config_loader import get_project_dir
from config.logger import setup_logging
from core.agent import reports
from core.agent.agent_session import Activity, AgentSessionError, Done, ErrorKind, TextDelta
from core.agent.grokbot_client import Credentials, default_credentials_path, get_hub
from core.agent.grokbot_session import GrokBotSession, GrokBotSessionConfig, voice_for
from core.agent.session_store import SessionStore
from core.providers.llm.claude_cli.claude_cli import ConnState, LLMProvider as AgentProvider, _num

TAG = __name__
logger = setup_logging()

DEFAULT_FILLER = "交给 Grok Bot 了"
DEFAULT_FILLER_DELAY = 4  # 简单问题 3–5 秒就回，等久一点再说填充语，免得每句都念一遍
# 附在每句话后面：Bot 的回复是给屏幕看的，不提示会写一大段列表。会出现在 App 的对话里，配置里填空字符串可关掉
DEFAULT_VOICE_HINT = "回复会被语音朗读：请口语化、尽量简短，不要用列表、表格和代码"

_shared = {"loop": None, "thread": None}
_shared_lock = threading.Lock()


def shared_loop() -> asyncio.AbstractEventLoop:
    """所有 grok_bot 实例共用的后台事件循环（永不停止）。"""
    with _shared_lock:
        loop = _shared["loop"]
        if loop is not None and not loop.is_closed():
            return loop
        loop = asyncio.new_event_loop()

        def run():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=run, name="grok-bot-loop", daemon=True)
        thread.start()
        _shared["loop"], _shared["thread"] = loop, thread
        return loop


class LLMProvider(AgentProvider):
    AGENT_LABEL = "Grok Bot"

    def __init__(self, config: dict):
        self.bot_id = str(config.get("bot_id") or "").strip()
        self.bot_name = str(config.get("bot_name") or "").strip() or self.bot_id[:8]
        self.model_name = f"grok_bot:{self.bot_name}"
        raw_path = str(config.get("credentials_path") or "").strip()
        self.cred_path = Path(os.path.expanduser(raw_path)) if raw_path else default_credentials_path()
        self.backend = str(config.get("backend_url") or "").strip() or None  # 只在测试时指向假服务
        self.session_config = GrokBotSessionConfig(
            bot_id=self.bot_id,
            bot_name=self.bot_name,
            voice_hint=str(config.get("voice_hint", DEFAULT_VOICE_HINT) or "").strip(),
            echo_timeout=_num(config, "echo_timeout_seconds", 10),
            poll_interval=_num(config, "poll_interval_seconds", 12),
        )
        self._init_common(config, filler_default=DEFAULT_FILLER, filler_delay_default=DEFAULT_FILLER_DELAY)
        self.store = SessionStore(Path(get_project_dir()) / "data" / "agent_sessions.json", "grok_bot")
        self._start_loop()
        self._self_check()

    # ---------- 共用的事件循环 ----------

    def _start_loop(self) -> None:
        self._loop = shared_loop()
        self._thread = _shared["thread"]

    def _maybe_stop(self) -> None:
        """共用的循环不随实例退役而停止；事件流由中枢按"有没有订阅"自己开关。"""
        return

    # ---------- 自检 ----------

    def _self_check(self) -> None:
        if not self.bot_id:
            logger.bind(tag=TAG).error("Grok Bot 条目没有 bot_id，请到管理后台的 Grok Bot 页重新勾选")
            return
        status = Credentials(self.cred_path).status()
        if not status["logged_in"]:
            logger.bind(tag=TAG).warning(
                f"Grok Bot「{self.bot_name}」：{'登录已失效' if status['needs_login'] else '还没登录'}，请到管理后台登录")
            return
        logger.bind(tag=TAG).info(
            f"Grok Bot「{self.bot_name}」：已登录，token 到期 "
            f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(status['expires_at']))}")

    def _voice_error(self, kind: ErrorKind) -> str:
        return voice_for(kind)

    # ---------- 会话 ----------

    def _new_session(self, st: ConnState) -> GrokBotSession:
        hub = get_hub(self.cred_path, logger=logger.bind(tag=TAG), backend=self.backend)
        key = st.key
        return GrokBotSession(hub, self.session_config, logger=logger.bind(tag=TAG),
                              on_report=lambda text: reports.add(key, self.bot_id, text), device_key=key)

    def abort(self, conn) -> None:
        """设备打断。会话还在启动（云端电脑唤醒要几十秒）时父类无事可做，这里记下来：启动完发了就转后台盯着。"""
        st = self._states.get(conn.session_id)
        if st is not None:
            st.abort_gen = st.turn_gen
        super().abort(conn)

    def on_conn_open(self, conn) -> None:
        """设备叫到这个 Bot（或重连时沿用它）：提前把云端电脑叫醒、连上事件流，第一句话就不用等了。"""
        super().on_conn_open(conn)
        if not self.bot_id:
            return
        st = self._state(conn.session_id)

        def kick():
            st.warm_task = asyncio.ensure_future(self._warm_up(st))

        self._loop.call_soon_threadsafe(kick)

    async def _warm_up(self, st: ConnState) -> None:
        """不拿 st.lock：热身期间第一句话到了，_ensure_agent 会等热身完再用，而不是排在锁后面干等。"""
        if st.closed or (st.agent and st.agent.alive):
            return
        agent = self._new_session(st)
        t0 = time.time()
        try:
            await agent.start()
        except AgentSessionError as e:
            logger.bind(tag=TAG).warning(f"Grok Bot「{self.bot_name}」热身失败（{e.kind.value}）：{e.detail[:200]}")
            return
        if st.closed or (st.agent and st.agent.alive):
            await agent.close()  # 连接已关闭，或这期间一轮对话自己建好了会话
            return
        st.agent = agent
        logger.bind(tag=TAG).info(f"Grok Bot「{self.bot_name}」已提前就绪（{time.time() - t0:.1f}s），设备 {st.key}")
        if st.lock is None:
            st.lock = asyncio.Lock()  # 空闲关闭要拿这把锁；还没有过任何一轮时它是 None
        if not st.lock.locked():
            self._schedule_idle_close(st)

    async def _ensure_agent(self, st: ConnState, text: str, out, system_prompt: str) -> bool:
        """确保有可用的会话。没有接续指令一说：对话在云端，永远是接着聊。

        它之前被打断的活做完了、还没念过的，先念了再发新的话。"""
        pending = reports.take_text(st.key, self.bot_id)
        if pending:
            out.put(TextDelta(pending))
        warm = getattr(st, "warm_task", None)
        if warm is not None and not warm.done():
            out.put(Activity("init", tools_running=1))  # 热身还没完：等它，期间不算卡住
            try:
                await warm
            except Exception:  # noqa: BLE001 —— 热身失败下面照常自己启动
                pass
        if not (st.agent and st.agent.alive):
            if not self.bot_id:
                raise AgentSessionError(ErrorKind.UNKNOWN, "没有配置 bot_id")
            agent = self._new_session(st)
            t0 = time.time()
            # 云端电脑休眠时唤醒要几十秒，期间没有任何事件：先报一次"还没开始干活"的 Activity（tools_running=1），
            # 否则 response() 会在 stall_timeout（默认 30 秒）后把这段等待当成卡住
            out.put(Activity("init", tools_running=1))
            await agent.start()
            if st.closed:
                # 启动期间连接已经关闭（例如用户点了结束）：话照样交出去，做完进待汇报箱，下次连上念
                logger.bind(tag=TAG).info(f"Grok Bot「{self.bot_name}」会话就绪前设备已断开，照样发出去，做完再汇报")
                await agent.hand_off(text)
                await agent.close()
                raise AgentSessionError(ErrorKind.PROCESS_EXITED, "连接已关闭")
            st.agent = agent
            logger.bind(tag=TAG).info(f"Grok Bot「{self.bot_name}」会话已就绪（{time.time() - t0:.1f}s），设备 {st.key}")
        if getattr(st, "abort_gen", None) == st.turn_gen:
            # 等启动的这段时间里设备已经打断了。和 Bot 醒着时发出去后再打断一样：话照样交给它，但不等回复
            # （没人听，也别一直占着锁），发完转后台盯着，做完进待汇报箱
            logger.bind(tag=TAG).info(f"Grok Bot「{self.bot_name}」会话就绪前设备已打断，照样发出去，做完再汇报")
            await st.agent.hand_off(text)
            out.put(Done(interrupted=True))
            return True
        return False
