"""pi_cli：让小智设备通过本机的 pi（pi coding agent，Node 写的 CLI）对话和干活。

连接管理、空闲关闭、填充语、进度提示、超时、后台任务、接续记录都继承 claude_cli 的 provider，
只换掉会话的创建方式（core/agent/pi_session.py）：pi 没有 Python SDK，靠 RPC 模式的子进程通信。

和 Claude Code 的差别：
- 需要 Node（pi 是 Node 包，要求 >= 22.19.0），没装时自检报错、播报"没找到 pi"；
- 用哪个模型由 pi 自己的 provider 决定，所以配置里要填 provider/model，API Key 通过环境变量交给 pi；
- 工具在 RPC 模式下直接执行、不弹审批，权限只靠启动时的 --tools 白名单。
"""

import asyncio
import os
import time
from pathlib import Path

from config.config_loader import get_project_dir
from config.logger import setup_logging
from core.agent.agent_session import AgentSessionError, Done, ErrorKind, TextDelta
from core.agent.pi_session import PiSession, PiSessionConfig, build_env, find_pi_cli, pi_version
from core.agent.session_store import SessionStore, choose_resume
from core.agent.voice_commands import match_resume
from core.providers.llm.claude_cli.claude_cli import (
    VOICE_RULES,
    ConnState,
    LLMProvider as AgentProvider,
    _num,
)

TAG = __name__
logger = setup_logging()

DEFAULT_FILLER = "好的，我看一下"
DEFAULT_FILLER_DELAY = 2
# 语音场景默认给"读文件 + 跑命令"：和 Claude Code 的"只读"档对齐，写文件要显式放开
DEFAULT_TOOLS = ["read", "bash"]

# 播报给用户的话。agent_session 里的默认文案说的是 Claude，这里换成 pi
VOICE_MESSAGES = {
    ErrorKind.NOT_INSTALLED: "这台设备上没有找到 pi，或者是 pi 缺少 Node 运行环境，请先装好再试。",
    ErrorKind.AUTH: "pi 还没有登录，或者 API Key 不对，请在后台检查。",
    ErrorKind.RATE_LIMIT: "pi 用的模型现在用量到上限了，稍后再试吧。",
    ErrorKind.BILLING: "pi 用的模型账户额度不够了，请检查一下账户。",
    ErrorKind.SERVER: "模型服务那边暂时有点问题，稍后再试吧。",
    ErrorKind.PROCESS_EXITED: "pi 意外退出了，再说一次试试。",
    ErrorKind.TIMEOUT: "这件事做得太久，我先停下了。想接着做的话，跟我说继续就行。",
    ErrorKind.STALLED: "pi 那边一直没有回应，我先停下了。稍后跟我说继续，就能接着做。",
}


class LLMProvider(AgentProvider):
    AGENT_LABEL = "pi"
    BACKGROUND_TURNS = True
    REPORT_ID = "pi_cli"

    def __init__(self, config: dict):
        self.model_name = f"pi_cli:{config.get('model') or 'default'}"
        self.cli_path = find_pi_cli(config.get("cli_path"))
        workdir = Path(os.path.expanduser(config.get("workdir") or "~/xiaozhi-workspace"))
        workdir.mkdir(parents=True, exist_ok=True)
        tools = config.get("allowed_tools")
        if tools is None or (isinstance(tools, str) and not tools.strip()):
            tools = DEFAULT_TOOLS
        if isinstance(tools, str):
            tools = [t.strip() for t in tools.split(",") if t.strip()]
        self.session_config = PiSessionConfig(
            cli_path=self.cli_path or "pi",
            provider=str(config.get("provider") or "").strip(),
            model=str(config.get("model") or "").strip(),
            api_key=str(config.get("api_key") or "").strip(),
            workdir=str(workdir),
            allowed_tools=list(tools),
            path_prefix=str(config.get("path_prefix") or "").strip(),
            no_extensions=bool(config.get("no_extensions", True)),
            no_skills=bool(config.get("no_skills", True)),
        )
        self._init_common(config, filler_default=DEFAULT_FILLER, filler_delay_default=DEFAULT_FILLER_DELAY)
        self.store = SessionStore(Path(get_project_dir()) / "data" / "agent_sessions.json", "pi_cli")
        # 预热：呼名到 pi 时先把进程拉起来，把 2-3 秒的 Node 冷启动藏进用户说话这段时间
        self.warm_up = bool(config.get("warm_up", True))
        self._prompts = {}  # 设备 -> 上次用过的系统提示词（预热时本轮的系统提示词可能还拿不到）
        self._start_loop()
        self._self_check()

    # ---------- 自检 ----------

    def _self_check(self) -> None:
        if not self.cli_path:
            logger.bind(tag=TAG).error(
                "未找到 pi 命令，请先安装 pi（需要 Node >= 22.19.0），或在配置中指定 cli_path")
            return
        try:
            logger.bind(tag=TAG).info(
                f"pi: {pi_version(self.cli_path, env=build_env(self.session_config))} ({self.cli_path})，"
                f"provider {self.session_config.provider or '默认'}，模型 {self.session_config.model or '默认'}，"
                f"工作目录 {self.session_config.workdir}，允许工具 {self.session_config.allowed_tools}")
        except Exception as e:  # noqa: BLE001
            logger.bind(tag=TAG).error(f"无法运行 {self.cli_path} --version: {e}")

    def _voice_error(self, kind: ErrorKind) -> str:
        return VOICE_MESSAGES.get(kind) or super()._voice_error(kind)

    # ---------- 会话 ----------

    def _new_session(self, st: ConnState, system_prompt: str, resume_id: str = None) -> PiSession:
        cfg = PiSessionConfig(**{**self.session_config.__dict__,
                                 "system_prompt_append": (system_prompt or "") + "\n" + VOICE_RULES})
        return PiSession(cfg, resume_id=resume_id, logger=logger.bind(tag=TAG))

    # ---------- 预热 ----------

    @staticmethod
    def _system_prompt_of(conn) -> str:
        """从连接的对话里取系统提示词（preheat 要用它，此刻本轮 dialogue 还没传进来）。"""
        messages = getattr(getattr(conn, "dialogue", None), "dialogue", None) or []
        for m in messages:
            if getattr(m, "role", None) == "system":
                return getattr(m, "content", "") or ""
        return ""

    def on_conn_open(self, conn) -> None:
        """设备连上就预热：下一句很可能就是叫 pi（设备会沿用上次用的模型）。"""
        super().on_conn_open(conn)
        if not self.warm_up:
            return
        st = self._state(conn.session_id)
        prompt = self._system_prompt_of(conn) or self._prompts.get(st.key, "")
        if prompt:
            self._prompts[st.key] = prompt

        def kick():
            st.warm_task = asyncio.ensure_future(self._warm_up(st, prompt))

        self._loop.call_soon_threadsafe(kick)

    async def _warm_up(self, st: ConnState, system_prompt: str) -> None:
        """不拿 st.lock：预热期间第一句话到了，_ensure_agent 会等预热完再用，而不是排在锁后面干等。"""
        if st.closed or (st.agent and st.agent.alive):
            return
        agent = self._new_session(st, system_prompt)
        t0 = time.time()
        try:
            await agent.start()
        except AgentSessionError as e:
            logger.bind(tag=TAG).warning(f"pi 预热失败（{e.kind.value}）：{e.detail[:200]}")
            return
        if st.closed or (st.agent and st.agent.alive):
            await agent.close()  # 连接已关，或这期间一轮对话自己建好了会话
            return
        st.agent = agent
        logger.bind(tag=TAG).info(f"pi 已提前就绪（{time.time() - t0:.1f}s），设备 {st.key}")
        if st.lock is None:
            st.lock = asyncio.Lock()
        if not st.lock.locked():
            self._schedule_idle_close(st)

    async def _ensure_agent(self, st: ConnState, text: str, out, system_prompt: str) -> bool:
        """确保有可用的进程；和父类（claude_cli）一样带接续：叫名字时接上次的会话，而不是新开一个。"""
        self._prompts[st.key] = system_prompt or ""
        warm = getattr(st, "warm_task", None)
        if warm is not None and not warm.done():
            try:
                await warm  # 预热还没完：等它，别重复启一个
            except Exception:  # noqa: BLE001 —— 预热失败下面照常自己启动
                pass
        if st.agent and st.agent.alive:
            return False
        remainder = match_resume(text, self.resume_keywords)
        keyword = remainder is not None
        resume_id, reason = choose_resume(self.store.get(st.key), keyword, self.resume_window)
        agent = self._new_session(st, system_prompt, resume_id)
        t0 = time.time()
        try:
            await agent.start()
        except AgentSessionError as e:
            if e.kind != ErrorKind.SESSION_MISSING:
                raise
            # pi 的 --session-id 找不到会新建，理论上不会走到这里；兜底保持和 Claude 一样的行为
            logger.bind(tag=TAG).warning(f"接续会话 {resume_id} 失败，改为新开会话")
            agent = self._new_session(st, system_prompt)
            await agent.start()
            self.store.forget(st.key)
            if keyword:
                out.put(TextDelta("😶没有找到之前的对话，我们重新开始吧。"))
            resume_id, reason = None, "resume_failed"
        if st.closed:
            await agent.close()
            raise AgentSessionError(ErrorKind.PROCESS_EXITED, "连接已关闭")
        st.agent = agent
        logger.bind(tag=TAG).info(
            f"pi 进程已启动（{time.time() - t0:.1f}s），设备 {st.key}，"
            f"{'接续会话 ' + resume_id if resume_id else '新会话 ' + (agent.session_id or '')}（{reason}）")
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
