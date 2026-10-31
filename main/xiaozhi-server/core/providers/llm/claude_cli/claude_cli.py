"""claude_cli：让小智设备通过本机 Claude Code（claude-agent-sdk）对话和干活。

线程模型：
- 服务端在每个连接的线程池里同步迭代 response() 生成器；
- 本 provider 自带一个后台线程运行 asyncio 事件循环，所有 SDK 调用都在这个循环里执行；
- 两边通过 queue.Queue 传递事件。

每个设备连接最多对应一个常驻的 Claude Code 进程：首句话到来时启动（顺便决定新开还是接续），
空闲超时、连接关闭或服务退出时结束。

管理后台热加载大模型时，旧实例会被调用 shutdown()：已连接的设备继续用旧实例对话，
等最后一个连接关闭、手上的任务都结束后，再停止事件循环，后台线程随之退出。
"""

import asyncio
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from config.config_loader import get_project_dir
from config.logger import setup_logging
from core.agent.agent_session import STARTING_LABELS, Activity, AgentSessionError, Done, Error, ErrorKind, TextDelta
from core.agent.claude_session import ClaudeSession, ClaudeSessionConfig
from core.agent.session_store import SessionStore, choose_resume
from core.agent.voice_commands import DEFAULT_RESUME_KEYWORDS, match_resume
from core.providers.llm.base import LLMProviderBase
from core.utils import textUtils
from core.utils.prompt_manager import EMOJI_List

TAG = __name__
logger = setup_logging()

VOICE_RULES = f"""
# 语音助手规则（最高优先级）
你现在通过一个语音设备和用户对话，你说的每个字都会被朗读出来。
1. 每次回复的第一个字符必须是一个表示心情的 emoji，并且只能从这个列表中选：{" ".join(EMOJI_List)}。列表以外的 emoji 一律不要用，回复中间和结尾也不要再出现 emoji。
2. 回答要简短口语化，一般不超过 3 句话。
3. 不要使用 Markdown、列表、编号、代码块、链接、表格等任何格式符号。
4. 需要调用工具时，调用前最多说一句过渡语；完成后用一两句话汇报结果，不要念出代码、命令、路径细节和文件内容。
5. 用户没问时，不要主动解释自己有哪些工具、权限或限制。
6. 不要用后台方式运行命令或任务（例如 run_in_background）：这一轮说完后你没法再主动开口，后台任务的结果用户永远听不到。需要等的命令就在前台运行，把超时设够；设备会陪着等，期间会自动提示"还在处理"。
"""

SENTENCE_ENDS = "。！？!?.;；~～…"
DEFAULT_PROGRESS_REPLY = "还在处理，请稍等"
PROGRESS_MAX_FACTOR = 4  # 进度提示的间隔逐次翻倍，最多拉长到 4 倍（默认 30 → 60 → 120 秒）


def elapsed_text(seconds: float) -> str:
    """把已用时间说成口语：95 秒 → "1 分半"，185 秒 → "3 分钟"。"""
    minutes, rest = divmod(int(seconds), 60)
    if minutes == 0:
        return f"{rest} 秒"
    return f"{minutes} 分半" if rest >= 30 else f"{minutes} 分钟"

COMMON_CLI_PATHS = [
    "~/.claude/local/claude",
    "~/.local/bin/claude",
    "/opt/homebrew/bin/claude",
    "/usr/local/bin/claude",
]


def single_sentence(text: str) -> str:
    """把一句话整理成 TTS 只会切出一段的形式。

    原项目为了降低首字延迟，第一句遇到逗号、顿号等就会切开，填充语会因此变成两段（两个气泡）。
    这里把句中的停顿标点换成空格，结尾统一用句号。
    """
    body = re.sub(r"[，,、~～；;：:]+", " ", text or "").strip()
    body = body.rstrip("。.！!？? ")
    return f"{body}。" if body else ""


def find_claude_cli(configured: Optional[str] = None) -> Optional[str]:
    """优先用配置的路径，其次 PATH，再其次常见安装位置（从图形界面启动时 PATH 可能不完整）。"""
    if configured:
        p = Path(os.path.expanduser(configured))
        return str(p) if p.exists() else None
    names = ["claude.exe", "claude"] if sys.platform == "win32" else ["claude"]
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    for candidate in COMMON_CLI_PATHS:
        p = Path(os.path.expanduser(candidate))
        if p.exists():
            return str(p)
    return None


def claude_cli_version(cli_path: str, timeout: float = 15) -> str:
    """运行 `claude --version`，返回第一行非空输出；失败时抛出异常。"""
    out = subprocess.run([cli_path, "--version"], capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    lines = [l.strip() for l in (out.stdout or out.stderr or "").splitlines() if l.strip()]
    return lines[0] if lines else ""


def _num(config: dict, key: str, default: float) -> float:
    """读取数字配置。只有没填（None 或空字符串）才用默认值，填 0 就是 0。"""
    value = config.get(key)
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@dataclass
class ConnState:
    key: str  # 设备 ID；没有时退回连接的 session_id
    conn: object = None
    agent: Optional[ClaudeSession] = None
    lock: Optional[asyncio.Lock] = None
    idle_handle: Optional[asyncio.TimerHandle] = None
    intent_warned: bool = False
    closed: bool = False  # 连接已关闭：不再启动或保留 Claude Code 进程
    turn_gen: int = 0  # 每开始一轮加 1，用于让过期的空闲定时器失效
    turn_task: Optional[asyncio.Task] = None  # 正在执行的一轮，卡死时要取消它，否则它一直占着锁


class LLMProvider(LLMProviderBase):
    # 日志和提示里对这个 Agent 的称呼。其他 Agent 的 provider（grok_bot）继承本类，改掉它和 _ensure_agent 即可
    AGENT_LABEL = "Claude Code"

    def __init__(self, config: dict):
        self.model_name = f"claude_cli:{config.get('model') or 'default'}"
        self.cli_path = find_claude_cli(config.get("cli_path"))
        workdir = Path(os.path.expanduser(config.get("workdir") or "~/xiaozhi-workspace"))
        workdir.mkdir(parents=True, exist_ok=True)
        self.session_config = ClaudeSessionConfig(
            cli_path=self.cli_path,
            model=config.get("model") or None,
            workdir=str(workdir),
            allowed_tools=list(config.get("allowed_tools") or ["Read", "Grep", "Glob"]),
            permission_mode=config.get("permission_mode") or "default",
            # 默认 low：语音场景以快速应答为主，实测工具类问题从约 6 秒降到约 4 秒
            effort=str(config.get("effort", "low") or "").strip() or None,
        )
        self._init_common(config)
        self.store = SessionStore(Path(get_project_dir()) / "data" / "agent_sessions.json", "claude_cli")
        self._start_loop()
        self._self_check()

    def _init_common(self, config: dict, filler_default: str = "好的，我看一下", filler_delay_default: float = 2) -> None:
        """和具体 Agent 无关的部分：超时、填充语、进度提示、连接状态表。"""
        # 超时类配置填 0 表示不限制。单轮默认 10 分钟：Claude 干长任务时设备陪着等，期间连接不会因"没人说话"断开
        self.turn_timeout = _num(config, "turn_timeout_seconds", 600) or float("inf")
        # 无响应超时：Claude Code 连续这么久没有任何事件（文字、思考、工具调用……）就判定卡住。
        # 工具执行期间（例如跑一条很慢的命令）不算，只受单轮超时限制
        self.stall_timeout = _num(config, "stall_timeout_seconds", 30) or float("inf")
        # 干活期间这么久没开口，就说一句进度，让人知道还在做；填 0 关闭。之后间隔逐渐拉长（见 _progress_gap），并报已用时间
        self.progress_interval = _num(config, "progress_interval_seconds", 30)
        self.progress_reply = single_sentence(config.get("progress_reply") or DEFAULT_PROGRESS_REPLY)
        # 第二次起的提示："还在处理，已经 3 分钟了"。自定义了提示语时用它开头
        self.progress_prefix = "还在处理" if not config.get("progress_reply") else self.progress_reply.rstrip("。")
        self.idle_timeout = _num(config, "idle_timeout_seconds", 300)  # 0：每轮结束就关闭进程
        self.resume_window = _num(config, "resume_window_minutes", 120)  # 0：不自动接续
        self.resume_keywords = list(config.get("resume_keywords") or DEFAULT_RESUME_KEYWORDS)
        self.filler = single_sentence(config.get("first_reply_filler", filler_default))
        self.filler_delay = _num(config, "filler_delay_seconds", filler_delay_default)
        self.abandon_grace = 5.0  # 发出中断后等这么久，这一轮还没停下就强制关闭进程

        self._states: Dict[str, ConnState] = {}
        self._states_lock = threading.Lock()
        self._retiring = False  # 已被新实例替换：没有连接后就停止事件循环
        self._stopping = False  # 已安排停止事件循环

    def _voice_error(self, kind: ErrorKind) -> str:
        """超时、卡住等由 provider 自己判定的错误，播报给用户的话。"""
        return Error(kind).voice_message

    def _start_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, args=(self._loop,), name="claude-cli-loop", daemon=True)
        self._thread.start()

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            loop.close()

    # ---------- 退役（管理后台热加载后调用） ----------

    def shutdown(self) -> None:
        """实例被替换后调用：不再接新连接，现有连接照常对话；没有连接时停止事件循环，后台线程退出。"""
        with self._states_lock:
            self._retiring = True
        logger.bind(tag=TAG).info(f"claude_cli 旧实例退役，剩余连接 {len(self._states)} 个，全部关闭后结束后台线程")
        self._maybe_stop()

    def _maybe_stop(self) -> None:
        with self._states_lock:
            if not self._retiring or self._stopping or self._states:
                return
            self._stopping = True
            loop = self._loop
        asyncio.run_coroutine_threadsafe(self._stop_loop(), loop)

    async def _stop_loop(self, timeout: float = 30) -> None:
        """等循环里剩下的任务（关闭进程、收尾中的一轮对话）结束，再停止循环。"""
        loop = asyncio.get_running_loop()
        pending = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        if pending:
            _, still = await asyncio.wait(pending, timeout=timeout)
            for t in still:
                t.cancel()
        loop.stop()
        logger.bind(tag=TAG).info("claude_cli 旧实例的后台线程已结束")

    # ---------- 启动自检 ----------

    def _self_check(self) -> None:
        if not self.cli_path:
            logger.bind(tag=TAG).error("未找到 claude 命令，请先安装 Claude Code 并登录，或在配置中指定 cli_path")
            return
        try:
            logger.bind(tag=TAG).info(
                f"Claude Code: {claude_cli_version(self.cli_path)} ({self.cli_path})，"
                f"工作目录 {self.session_config.workdir}，允许工具 {self.session_config.allowed_tools}，"
                f"思考力度 {self.session_config.effort or '默认'}"
            )
        except Exception as e:
            logger.bind(tag=TAG).error(f"无法运行 {self.cli_path} --version: {e}")

    # ---------- 连接钩子（由原项目代码通过 getattr 调用） ----------

    def _state(self, session_id: str) -> ConnState:
        with self._states_lock:
            if self._stopping:
                # 极少见：替换实例的瞬间刚好有设备连上旧实例，而循环已经在停止。重新开一个循环给它用，
                # 它关闭后会再次停止。
                self._stopping = False
                self._start_loop()
            st = self._states.get(session_id)
            if st is None:
                st = ConnState(key=session_id)
                self._states[session_id] = st
            return st

    def on_conn_open(self, conn) -> None:
        st = self._state(conn.session_id)
        st.conn = conn
        if getattr(conn, "device_id", None):
            st.key = conn.device_id

    def abort(self, conn) -> None:
        st = self._states.get(conn.session_id)
        if st and st.agent:
            asyncio.run_coroutine_threadsafe(st.agent.interrupt(), self._loop)
            # 卡在工具调用里时 interrupt 可能无效，兜底防止下一句话一直排队等这一轮
            asyncio.run_coroutine_threadsafe(self._abandon_if_stuck(st, st.turn_gen), self._loop)

    def reset_session(self, conn) -> None:
        """清除聊天记录：关掉这台设备的进程、忘掉会话映射，下一句话新开会话（grok_bot 继承：关掉会话，云端由调用方重置）。"""
        st = self._states.get(conn.session_id)
        if st is None:
            return
        self.store.forget(st.key)
        if st.agent:
            asyncio.run_coroutine_threadsafe(self._close_agent(st, "清除聊天记录"), self._loop)

    def on_conn_close(self, conn) -> None:
        with self._states_lock:
            st = self._states.pop(conn.session_id, None)
        if st:
            st.closed = True
            asyncio.run_coroutine_threadsafe(self._close_agent(st, "连接关闭"), self._loop)
        self._maybe_stop()

    # ---------- 后台循环中的逻辑 ----------

    async def _close_agent(self, st: ConnState, reason: str) -> None:
        if st.idle_handle:
            st.idle_handle.cancel()
            st.idle_handle = None
        agent, st.agent = st.agent, None
        if agent:
            await agent.close()
            logger.bind(tag=TAG).info(f"已关闭 {self.AGENT_LABEL} 会话（{reason}），设备 {st.key}")

    def _schedule_idle_close(self, st: ConnState) -> None:
        if st.idle_handle:
            st.idle_handle.cancel()
        gen = st.turn_gen
        st.idle_handle = self._loop.call_later(
            self.idle_timeout, lambda: asyncio.ensure_future(self._idle_close(st, gen))
        )

    async def _idle_close(self, st: ConnState, gen: int) -> None:
        async with st.lock:
            if st.turn_gen != gen:
                return  # 定时器触发前已经开始了新的一轮
            st.idle_handle = None
            await self._close_agent(st, f"空闲 {int(self.idle_timeout)} 秒")

    async def _abandon_if_stuck(self, st: ConnState, gen: int) -> None:
        """超时或设备打断后已发送中断；如果这一轮过了 abandon_grace 秒还没结束，说明 Claude Code 卡死了，直接结束进程。
        下一轮会自动重新启动进程（在接续窗口内会接续原会话）。

        关闭进程不一定能让卡住的那一轮退出（它可能还在等一个永远不来的消息），所以还要取消它，
        否则它一直占着锁，下一句话会排队等到超时。
        """
        await asyncio.sleep(self.abandon_grace)
        if st.turn_gen == gen and st.lock is not None and st.lock.locked():
            logger.bind(tag=TAG).warning(f"中断后 {self.AGENT_LABEL} 仍未结束本轮，强制关闭")
            task = st.turn_task
            await self._close_agent(st, "中断无效")
            if task is not None and not task.done() and st.turn_gen == gen:
                task.cancel()

    async def _ensure_agent(self, st: ConnState, text: str, out: queue.Queue, system_prompt: str) -> bool:
        """确保有可用的进程。返回 True 表示这句话已经作为"接续指令"处理完，不必再发给 Claude。

        接续指令后面还带着别的内容时（如"接着上次聊那个 bug"），接续后把整句话照常发给 Claude，
        不能只回一句"好的"就把后半句吞掉。
        """
        if st.agent and st.agent.alive:
            return False
        remainder = match_resume(text, self.resume_keywords)
        keyword = remainder is not None
        resume_id, reason = choose_resume(self.store.get(st.key), keyword, self.resume_window)
        cfg = ClaudeSessionConfig(**{**self.session_config.__dict__,
                                     "system_prompt_append": (system_prompt or "") + "\n" + VOICE_RULES})
        agent = ClaudeSession(cfg, resume_id=resume_id, logger=logger.bind(tag=TAG))
        t0 = time.time()
        try:
            await agent.start()
        except AgentSessionError as e:
            if e.kind != ErrorKind.SESSION_MISSING:
                raise
            logger.bind(tag=TAG).warning(f"接续会话 {resume_id} 失败，改为新开会话")
            agent = ClaudeSession(cfg, logger=logger.bind(tag=TAG))
            await agent.start()  # 新会话也启动失败时（如未登录、网络问题）保留旧记录，下次还能接续
            self.store.forget(st.key)
            if keyword:
                out.put(TextDelta("😶没有找到之前的对话，我们重新开始吧。"))
            resume_id, reason = None, "resume_failed"
        if st.closed:
            await agent.close()  # 启动期间连接已经关闭
            raise AgentSessionError(ErrorKind.PROCESS_EXITED, "连接已关闭")
        st.agent = agent
        logger.bind(tag=TAG).info(
            f"Claude Code 进程已启动（{time.time() - t0:.1f}s），设备 {st.key}，"
            f"{'接续会话 ' + resume_id if resume_id else '新会话 ' + agent.session_id}（{reason}）"
        )
        if keyword and not remainder:
            if resume_id:
                out.put(TextDelta("😉好的，我们接着上次聊。"))
            elif reason == "no_record":
                out.put(TextDelta("😶没有找到之前的对话，我们重新开始吧。"))
            out.put(Done(session_id=agent.session_id))
            return True
        if keyword and reason == "no_record":
            out.put(TextDelta("😶没有找到之前的对话，我们重新开始吧。"))
        return False

    async def _run_turn(self, st: ConnState, text: str, system_prompt: str, out: queue.Queue) -> None:
        if st.lock is None:
            st.lock = asyncio.Lock()
        async with st.lock:
            st.turn_gen += 1
            st.turn_task = asyncio.current_task()
            if st.idle_handle:
                st.idle_handle.cancel()
                st.idle_handle = None
            try:
                if await self._ensure_agent(st, text, out, system_prompt):
                    if st.agent:  # grok_bot 在连接已关闭时把话交出去后也返回 True，那时没有会话
                        self.store.touch(st.key, st.agent.session_id)
                    return
                agent, recorded, working, failed, previous = st.agent, False, False, False, None
                async for ev in agent.send(text):
                    if isinstance(ev, Error) and not working and ev.kind != ErrorKind.PROCESS_EXITED:
                        failed = True  # 还没开始干活就出错：没登录、额度用完、限流或过载重试用完等
                    elif not failed and isinstance(ev, (TextDelta, Activity)):
                        if not recorded:
                            # 会话已存在于本机（init 时用户的话就已存进去）。马上记下来：这一轮如果卡死（例如和 Anthropic 的
                            # 网络卡住，只收到 init）、进程被强制关闭，下一句话启动新进程时也能接回这个会话
                            previous = self.store.get(st.key)
                            self.store.touch(st.key, agent.session_id)
                            recorded = True
                        working = working or not (isinstance(ev, Activity) and ev.label in STARTING_LABELS)
                    out.put(ev)
                    if isinstance(ev, Error) and ev.kind == ErrorKind.PROCESS_EXITED:
                        st.agent = None
                if failed:
                    # 还没开始干活就出错：恢复原来的记录，免得新会话覆盖之前能接续的会话（包括最后活跃时间）
                    if recorded:
                        self._restore_record(st.key, previous)
                elif recorded and st.agent:
                    self.store.touch(st.key, st.agent.session_id)  # 更新最后活跃时间
            except AgentSessionError as e:
                out.put(Error(e.kind, e.detail, voice=self._voice_error(e.kind)))
            except asyncio.CancelledError:
                # Claude Code 卡死被强制关闭（_abandon_if_stuck 取消了这一轮）：告诉还在等的一方这一轮结束了
                out.put(Done(interrupted=True))
                raise
            except Exception as e:
                logger.bind(tag=TAG).exception(f"{self.AGENT_LABEL} 处理出错: {e}")
                out.put(Error(ErrorKind.UNKNOWN, str(e), voice=self._voice_error(ErrorKind.UNKNOWN)))
            finally:
                if st.agent and not st.closed:
                    self._schedule_idle_close(st)

    # ---------- 服务端调用的同步接口 ----------

    def response(self, session_id, dialogue, **kwargs):
        text = next((m.get("content") for m in reversed(dialogue) if m.get("role") == "user"), "") or ""
        system_prompt = dialogue[0].get("content", "") if dialogue and dialogue[0].get("role") == "system" else ""
        st = self._state(session_id)
        self._warn_intent(st)

        out: queue.Queue = queue.Queue()
        asyncio.run_coroutine_threadsafe(self._run_turn(st, text, system_prompt, out), self._loop)

        t0 = time.time()
        last_event_at = t0
        last_spoken_at = t0       # 上一次说话（填充语、正式回复、进度提示）的时间
        tail = ""                 # 已说出的最后一个字，判断能不能在这里插一句进度
        tools_running = 0
        progress_count = 0        # 这一轮已经说了几次进度
        first_at = None
        filler_sent = False
        emotion_buf = ""          # 填充语之后，正式回复开头的几个字，用来找表情
        emotion_resent = False
        finished = False
        try:
            while True:
                if self._aborted(st):
                    # 设备打断了（abort() 已经让 Claude 中断）：这一轮到此为止，不再保活、不再说进度。
                    # 不能等 Claude 自己结束：它卡住不理中断时，这里会一直等到单轮超时（最长 10 分钟），
                    # 期间连接一直不会因为没人说话断开；用户开口说下一句后，打断标志被清掉，
                    # 这里还会接着吐出"还在处理""做得太久"
                    finished = True
                    return
                try:
                    ev = out.get(timeout=0.2)
                except queue.Empty:
                    self._keep_alive(st)
                    elapsed = time.time() - t0
                    if first_at is None and self.filler and not filler_sent and elapsed >= self.filler_delay:
                        filler_sent = True
                        last_spoken_at, tail = time.time(), self.filler[-1]
                        yield f"😶{self.filler}"
                    elif self._progress_due(last_spoken_at, tail, first_at is not None or filler_sent, progress_count):
                        text = self.progress_reply if progress_count == 0 else \
                            f"{self.progress_prefix}，已经 {elapsed_text(time.time() - t0)}了。"
                        notice = self._before_notice(tail, first_at is not None or filler_sent) + text
                        progress_count += 1
                        last_spoken_at, tail = time.time(), text[-1]
                        yield notice
                    stalled = not tools_running and time.time() - last_event_at >= self.stall_timeout
                    if elapsed >= self.turn_timeout or stalled:
                        kind = ErrorKind.STALLED if stalled else ErrorKind.TIMEOUT
                        logger.bind(tag=TAG).warning(
                            f"已中断本轮：{'连续 %d 秒没有任何响应' % int(self.stall_timeout) if stalled else '超过 %d 秒' % int(self.turn_timeout)}"
                        )
                        if st.agent:
                            asyncio.run_coroutine_threadsafe(st.agent.interrupt(), self._loop)
                            asyncio.run_coroutine_threadsafe(self._abandon_if_stuck(st, st.turn_gen), self._loop)
                        yield self._before_notice(tail, first_at is not None or filler_sent) + self._voice_error(kind)
                        finished = True
                        return
                    continue
                last_event_at = time.time()
                self._keep_alive(st)
                if isinstance(ev, Activity):
                    if ev.tools_running >= 0:
                        tools_running = ev.tools_running
                    continue
                if isinstance(ev, TextDelta):
                    if first_at is None:
                        first_at = time.time() - t0
                    if filler_sent and not emotion_resent:
                        # 原项目每轮只按第一段文字设一次表情，填充语会把表情锁成 😶。
                        # 正式回复开头带了表情时，再补发一次情绪消息，让设备换成真正的表情。
                        emotion_buf += ev.text
                        emoji = next((ch for ch in emotion_buf[:8] if ch in textUtils.EMOJI_MAP), None)
                        if emoji or len(emotion_buf) >= 8:
                            emotion_resent = True
                            if emoji:
                                self._send_emotion(st, emoji)
                    if ev.text.strip():
                        last_spoken_at, tail = time.time(), ev.text.strip()[-1]
                    yield ev.text
                elif isinstance(ev, Error):
                    logger.bind(tag=TAG).error(f"{self.AGENT_LABEL} 错误 [{ev.kind.value}]: {ev.detail[:500]}")
                    yield self._before_notice(tail, first_at is not None or filler_sent) + ev.voice_message
                    finished = True
                    return
                elif isinstance(ev, Done):
                    finished = True
                    logger.bind(tag=TAG).info(
                        f"{self.AGENT_LABEL} 本轮完成：首字 {first_at if first_at is None else round(first_at, 2)}s，"
                        f"总耗时 {time.time() - t0:.1f}s{'，已被打断' if ev.interrupted else ''}"
                    )
                    return
        finally:
            if not finished and st.agent:
                # 生成器被提前关闭（例如设备打断后服务端停止迭代），让 Claude 也停下来
                asyncio.run_coroutine_threadsafe(st.agent.interrupt(), self._loop)

    @staticmethod
    def _aborted(st: ConnState) -> bool:
        """设备是否打断了这一轮。打断时服务端会把连接的 client_abort 置为 True，新的一句话开始时清掉。"""
        return bool(getattr(st.conn, "client_abort", False))

    def _restore_record(self, key: str, previous: Optional[dict]) -> None:
        """把这台设备的会话记录恢复成这一轮开始前的样子；原来没有记录就删掉。"""
        if previous and previous.get("session_id"):
            self.store.touch(key, previous["session_id"], now=previous.get("last_active"))
        else:
            self.store.forget(key)

    @staticmethod
    def _keep_alive(st: ConnState) -> None:
        """Claude 干活期间刷新连接的活动时间。

        服务端在"没有任何动静"一段时间后会兜底断开连接（core/connection.py 的 _check_timeout，
        默认 180 秒）。干活时设备在等、服务端也可能很久不说话，不刷新的话长任务做到一半连接就断了，
        Claude Code 进程也会随之关闭。
        """
        if st.conn is not None:
            st.conn.last_activity_time = time.time() * 1000

    @staticmethod
    def _before_notice(tail: str, spoke: bool) -> str:
        """服务端自己插一句话（超时、出错、进度提示）前要加的东西。

        还没说过话：加一个表情，设备才会换脸；已经说了半句（结尾不是句末标点）：先补个句号把那半句收住，
        否则 TTS 会把两句粘成一句念，比如"我先看一下文件这件事做得太久"。
        """
        if not spoke:
            return "😶"
        return "" if not tail or tail in SENTENCE_ENDS or tail in textUtils.EMOJI_MAP else "。"

    def _progress_gap(self, count: int) -> float:
        """第 count+1 次进度提示前要等多久：每次翻倍，最多拉长到 PROGRESS_MAX_FACTOR 倍。
        默认依次在第 30、90、210、330… 秒提示，10 分钟的任务说 6 次，而不是每 30 秒一次说 20 次。"""
        return self.progress_interval * min(2 ** count, PROGRESS_MAX_FACTOR)

    def _progress_due(self, last_spoken_at: float, tail: str, spoke: bool, count: int = 0) -> bool:
        """该不该说一句进度：超过间隔没开口，而且上一句已经说完（不在半句话中间插话）。"""
        if not self.progress_reply or not self.progress_interval:
            return False
        if time.time() - last_spoken_at < self._progress_gap(count):
            return False
        return not spoke or not tail or tail in SENTENCE_ENDS or tail in textUtils.EMOJI_MAP

    def _send_emotion(self, st: ConnState, emoji: str) -> None:
        conn = st.conn
        loop = getattr(conn, "loop", None)
        if conn is None or loop is None or not (getattr(conn, "features", None) or {}).get("emoji", True):
            return
        asyncio.run_coroutine_threadsafe(textUtils.get_emotion(conn, emoji), loop)
        logger.bind(tag=TAG).debug(f"填充语之后补发表情 {emoji}")

    def response_no_stream(self, system_prompt, user_prompt, **kwargs):
        logger.bind(tag=TAG).warning(f"{self.AGENT_LABEL} 不支持非流式调用（意图识别、记忆总结等），请为这些模块单独配置 API 模型")
        return ""

    def _warn_intent(self, st: ConnState) -> None:
        if st.intent_warned or st.conn is None:
            return
        intent_type = getattr(st.conn, "intent_type", None)
        if intent_type and intent_type != "nointent":
            logger.bind(tag=TAG).warning(f"使用 {self.AGENT_LABEL} 时 Intent 应设为 nointent，当前为 {intent_type}")
        st.intent_warned = True
