"""基于 pi 的 RPC 模式（--mode rpc）的会话：一个常驻子进程，JSONL 收发，多轮对话、可打断、可接续。

pi 是 Node 写的 CLI，没有 Python SDK，所以这里自己管子进程和记录（协议见 pi 自带的 docs/rpc.md、docs/json.md）：
- 往 stdin 写命令：prompt、abort、get_state ……
- 从 stdout 读三类记录：response（对某条命令的应答）、session 事件（message_update、tool_execution_* ……）、
  扩展的 UI 请求（本项目启动时关掉了扩展，收到就丢弃）。

和 claude_session 的几处差别：
- 一轮的结束信号是 agent_settled，不是 agent_end：agent_end 只表示一次底层 run 结束，之后 pi 还可能自动重试、
  压缩上下文、接着做排队的消息，念早了一轮会被截断；
- 工具在 RPC 模式下直接执行、不弹审批，所以权限改成在启动时用 --tools 白名单限定；
- 接续用 --session-id：pi 找不到这个 ID 会新建，所以不会出现"会话不存在"的错误。
"""

import asyncio
import json
import os
import re
import shutil
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

from core.agent.agent_session import (
    Activity,
    AgentEvent,
    AgentSession,
    AgentSessionError,
    Done,
    Error,
    ErrorKind,
    TextDelta,
)

# 各家 provider 读哪个环境变量拿 Key（pi 的 docs/providers.md）。表里没有的按 <PROVIDER>_API_KEY 猜
API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GEMINI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
    "zhipu": "ZHIPU_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
}

# 从图形界面或 systemd 启动时 PATH 可能不完整，按这几处找 pi
COMMON_PI_PATHS = [
    "~/.local/bin/pi",
    "~/.npm-global/bin/pi",
    "/usr/local/bin/pi",
    "/opt/homebrew/bin/pi",
]

# 出错时按关键字归类，播报不同的话（VOICE_ERROR_MESSAGES 见 agent_session）
ERROR_PATTERNS = [
    (ErrorKind.AUTH, re.compile(r"unauthor|authenticat|invalid api key|no api key|not logged in|401", re.I)),
    (ErrorKind.BILLING, re.compile(r"insufficient|quota|credit|billing|payment|402", re.I)),
    (ErrorKind.RATE_LIMIT, re.compile(r"rate limit|too many requests|429|overloaded|capacity", re.I)),
    (ErrorKind.SERVER, re.compile(r"internal server|bad gateway|service unavailable|timeout|timed out|5\d\d", re.I)),
]

# 工具的返回值可能是给屏幕看的，只留前若干个字符进日志
LOG_SLICE = 160

# 句末标点：一段话说到这些字符，TTS 才会马上念；否则要等下一段，期间插不进进度提示
SENTENCE_END = set("。！？!?.;；~～…\n")


def build_env(config: "PiSessionConfig") -> Dict[str, str]:
    """准备子进程的环境变量：PATH 前缀（pi 是 npm 装的，要靠它找到 node）+ provider 的 API Key。

    self_check 也要用同一套环境，否则会报"找不到 node"其实只是自检没带 PATH。
    """
    env = dict(os.environ)
    prefix = [p.strip() for p in re.split(r"[:;]", config.path_prefix or "") if p.strip()]
    if prefix:
        env["PATH"] = ":".join(prefix + [env.get("PATH", "")])
    if config.api_key:
        env[api_key_env(config.provider)] = config.api_key
    return env


def find_pi_cli(configured: Optional[str] = None) -> Optional[str]:
    """优先用配置的路径，其次 PATH，再其次常见安装位置。"""
    if configured:
        p = Path(os.path.expanduser(configured))
        return str(p) if p.exists() else None
    names = ["pi.exe", "pi"] if sys.platform == "win32" else ["pi"]
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    for candidate in COMMON_PI_PATHS:
        p = Path(os.path.expanduser(candidate))
        if p.exists():
            return str(p)
    return None


def pi_version(cli_path: str, timeout: float = 20, env: Optional[Dict[str, str]] = None) -> str:
    """运行 `pi --version`，返回第一行非空输出；失败时抛出异常。"""
    import subprocess

    out = subprocess.run([cli_path, "--version"], capture_output=True, text=True, encoding="utf-8",
                         timeout=timeout, env=env)
    lines = [l.strip() for l in (out.stdout or out.stderr or "").splitlines() if l.strip()]
    return lines[0] if lines else ""


def api_key_env(provider: str) -> str:
    name = (provider or "").strip().lower()
    if name in API_KEY_ENV:
        return API_KEY_ENV[name]
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_") + "_API_KEY"


def classify_error(text: str) -> ErrorKind:
    for kind, pattern in ERROR_PATTERNS:
        if pattern.search(text or ""):
            return kind
    return ErrorKind.UNKNOWN


@dataclass
class PiSessionConfig:
    cli_path: Optional[str] = None
    provider: str = ""  # pi 的 provider 名，例如 deepseek、anthropic
    model: str = ""  # 模型 ID 或匹配模式，例如 deepseek-flash
    api_key: str = ""  # 留空则用 pi 自己的登录状态（auth.json）
    workdir: Optional[str] = None
    # 工具白名单：语音场景默认只给读文件和跑命令，不给写文件
    allowed_tools: List[str] = field(default_factory=lambda: ["read", "bash"])
    system_prompt_append: str = ""
    # 关掉扩展、技能、提示模板的发现：语音设备上行为要可预期，启动也快
    no_extensions: bool = True
    no_skills: bool = True
    # 需要额外加进 PATH 的目录（冒号或分号分隔）。pi 是 npm 安装的，启动脚本要能找到 node
    path_prefix: str = ""
    extra_args: List[str] = field(default_factory=list)


class PiSession(AgentSession):
    def __init__(self, config: PiSessionConfig, resume_id: Optional[str] = None, logger=None):
        self.config = config
        self.resume_id = resume_id
        self.logger = logger
        self._session_id = resume_id or str(uuid.uuid4())
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._events: asyncio.Queue = asyncio.Queue()
        self._pending: Dict[str, asyncio.Future] = {}
        self._seq = 0
        self._stderr_tail = deque(maxlen=20)
        self._tasks: List[asyncio.Task] = []
        self._closed = False
        # 一轮里的临时状态
        self._tools_running = 0
        self._text_emitted: Dict[int, int] = {}
        self._last_char = ""  # 上一个说出去的字，用于判断要不要补句号
        self._text_started = False  # 当前这条 assistant 消息是否已经开过口
        self._aborted = False

    # ---------- 基本状态 ----------

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    # ---------- 生命周期 ----------

    def _command(self) -> List[str]:
        cmd = [self.config.cli_path or "pi", "--mode", "rpc", "--session-id", self._session_id]
        if self.config.provider:
            cmd += ["--provider", self.config.provider]
        if self.config.model:
            cmd += ["--model", self.config.model]
        if self.config.allowed_tools:
            cmd += ["--tools", ",".join(self.config.allowed_tools)]
        else:
            cmd += ["--no-tools"]
        if self.config.system_prompt_append:
            cmd += ["--append-system-prompt", self.config.system_prompt_append]
        if self.config.no_extensions:
            cmd += ["--no-extensions"]
        if self.config.no_skills:
            cmd += ["--no-skills"]
        cmd += list(self.config.extra_args)
        return cmd

    def _env(self) -> Dict[str, str]:
        return build_env(self.config)

    def _exit_detail(self, error: str = "") -> str:
        """进程意外退出时拼给上层看的说明：把 stderr 尾部带上，常见的是 node 没装、Key 没填。"""
        parts = [p for p in (error, self.stderr_tail) if p]
        return "\n".join(parts) or "pi 进程已退出"

    async def start(self) -> None:
        cmd = self._command()
        if not self.config.cli_path:
            raise AgentSessionError(ErrorKind.NOT_INSTALLED, "没有找到 pi 命令")
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.workdir or None,
                env=self._env(),
                limit=8 * 1024 * 1024,  # 工具返回大块内容时单行可能很长
            )
        except (FileNotFoundError, PermissionError) as e:
            raise AgentSessionError(ErrorKind.NOT_INSTALLED, str(e))
        self._tasks = [
            asyncio.ensure_future(self._read_stdout()),
            asyncio.ensure_future(self._read_stderr()),
        ]
        # 握一次手：确认进程能回话，顺便拿到 pi 实际使用的会话 ID（--session-id 找不到时会新建，ID 不变）
        try:
            state = await self._request({"type": "get_state"}, timeout=25)
        except AgentSessionError as e:
            await self.close()
            raise AgentSessionError(e.kind, self._exit_detail(e.detail))
        actual = ((state or {}).get("data") or {}).get("sessionId")
        if actual:
            self._session_id = actual
        self._log("info", f"pi 已启动，会话 {self._session_id[:8]}，模型 {(state or {}).get('data', {}).get('model', {}).get('id', '?')}")

    async def close(self) -> None:
        self._closed = True
        proc, self._proc = self._proc, None
        for task in self._tasks:
            task.cancel()
        self._tasks = []
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        if proc is None:
            return
        try:
            if proc.stdin is not None and not proc.stdin.is_closing():
                proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        except ProcessLookupError:
            pass
        except Exception as e:  # noqa: BLE001
            self._log("warning", f"关闭 pi 进程时出错: {e}")

    async def interrupt(self) -> None:
        if not self.alive:
            return
        self._aborted = True
        try:
            await self._send({"type": "abort"})
        except Exception as e:  # noqa: BLE001
            self._log("warning", f"中断 pi 失败: {e}")

    # ---------- 收发 ----------

    def _log(self, level: str, message: str) -> None:
        if self.logger:
            getattr(self.logger, level)(f"[{(self._session_id or '')[:8]}] {message}")

    async def _send(self, record: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise AgentSessionError(ErrorKind.PROCESS_EXITED, self._exit_detail())
        try:
            proc.stdin.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, RuntimeError) as e:
            # pi 已经退出（没装 node、Key 不对、崩溃）：别把裸异常抛给上层，否则只会播报"出了点小问题"
            raise AgentSessionError(ErrorKind.PROCESS_EXITED, self._exit_detail(str(e)))

    async def _request(self, record: dict, timeout: float = 15) -> dict:
        """发一条命令并等它自己的 response。事件记录不受影响，照常进队列。"""
        self._seq += 1
        req_id = f"hub-{self._seq}"
        record = {**record, "id": req_id}
        fut = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self._send(record)
            resp = await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            raise AgentSessionError(ErrorKind.STALLED, f"pi 对 {record.get('type')} 没有应答（{timeout} 秒）")
        finally:
            self._pending.pop(req_id, None)
        if not resp.get("success", False):
            detail = json.dumps(resp.get("error") or resp.get("data") or resp, ensure_ascii=False)
            raise AgentSessionError(classify_error(detail), detail)
        return resp

    async def _read_stdout(self) -> None:
        """读完 stdout 所有记录，直到进程退出。response 交给等待的命令，其余进事件队列。"""
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    self._log("debug", f"忽略非 JSON 输出: {text[:200]}")
                    continue
                if record.get("type") == "response":
                    fut = self._pending.get(record.get("id") or "")
                    if fut is not None and not fut.done():
                        fut.set_result(record)
                    continue
                await self._events.put(record)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self._log("warning", f"读取 pi 输出出错: {e}")
        finally:
            await self._events.put({"type": "_eof"})

    async def _read_stderr(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").rstrip()
                if text:
                    self._stderr_tail.append(text)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            pass

    async def _next_event(self) -> dict:
        record = await self._events.get()
        if record.get("type") == "_eof":
            raise AgentSessionError(ErrorKind.PROCESS_EXITED, self._exit_detail())
        return record

    # ---------- 一轮对话 ----------

    def _reset_turn(self) -> None:
        self._tools_running = 0
        self._text_emitted.clear()
        self._last_char = ""
        self._text_started = False
        self._aborted = False
        while not self._events.empty():
            self._events.get_nowait()

    @staticmethod
    def _assistant_event(record: dict) -> dict:
        return record.get("assistantMessageEvent") or {}

    async def send(self, text: str) -> AsyncIterator[AgentEvent]:
        if not self.alive:
            raise AgentSessionError(ErrorKind.PROCESS_EXITED, self.stderr_tail or "pi 进程已退出")
        self._reset_turn()
        try:
            await self._request({"type": "prompt", "message": text}, timeout=20)
        except AgentSessionError:
            raise
        yield Activity("init", tools_running=-1)  # 已受理，还没开始干活
        while True:
            try:
                record = await self._next_event()
            except AgentSessionError as e:
                yield Error(e.kind, e.detail, voice=Error(e.kind).voice_message)
                return
            kind = record.get("type")

            if kind == "agent_start":
                yield Activity("", tools_running=self._tools_running)
                continue
            if kind == "message_start":
                # 一条新的 assistant 消息开始：计数器重置，下一段文字到时要判断该不该补句号
                if ((record.get("message") or {}).get("role")) == "assistant":
                    self._text_started = False
                continue
            if kind == "message_update":
                async for ev in self._message_update(record):
                    yield ev
                continue
            if kind == "tool_execution_start":
                self._tools_running += 1
                name = record.get("toolName") or "工具"
                self._log("info", f"开始执行 {name}")
                yield Activity(name, tools_running=self._tools_running)
                continue
            if kind == "tool_execution_update":
                yield Activity(record.get("toolName") or "工具", tools_running=self._tools_running)
                continue
            if kind == "tool_execution_end":
                self._tools_running = max(0, self._tools_running - 1)
                name = record.get("toolName") or "工具"
                ok = not record.get("isError")
                self._log("info", f"{name} 结束（{'成功' if ok else '失败'}，{record.get('durationMs') or '?'} 毫秒）")
                yield Activity(name, tools_running=self._tools_running)
                continue
            if kind == "auto_retry_start":
                self._log("warning", f"pi 正在重试（第 {record.get('attempt')} 次）：{record.get('errorMessage')}")
                yield Activity("api_retry", tools_running=self._tools_running)
                continue
            if kind == "auto_retry_end" and not record.get("success", True):
                detail = str(record.get("finalError") or "重试用完")
                yield Error(classify_error(detail), detail, voice=Error(classify_error(detail)).voice_message)
                return
            if kind == "agent_end":
                if record.get("willRetry"):
                    yield Activity("api_retry", tools_running=self._tools_running)
                continue
            if kind == "agent_settled":
                yield Done(session_id=self._session_id, interrupted=bool(record.get("aborted")))
                return
            # 其余（turn_start/turn_end/message_start/message_end/queue_update/compaction_* 等）不播报

    async def _message_update(self, record: dict) -> AsyncIterator[AgentEvent]:
        ev = self._assistant_event(record)
        etype = ev.get("type")
        if etype == "text_delta":
            delta = ev.get("delta") or ""
            if delta:
                if not self._text_started and self._last_char and self._last_char not in SENTENCE_END:
                    # 上一段话没说完（典型是说完"我跑一下这个命令："就去调工具了）：先补个句号，TTS 才会马上念
                    yield TextDelta("。")
                    self._last_char = "。"
                self._text_started = True
                index = ev.get("contentIndex", 0)
                self._text_emitted[index] = self._text_emitted.get(index, 0) + len(delta)
                self._last_char = delta[-1]
                yield TextDelta(delta)
            return
        if etype == "text_end":
            # 模型没流式吐字时，只在结束时才有内容：补上
            index = ev.get("contentIndex", 0)
            content = ev.get("content") or ""
            if content and not self._text_emitted.get(index):
                self._text_emitted[index] = len(content)
                self._last_char = content[-1]
                self._text_started = True
                yield TextDelta(content)
            return
        if etype in ("thinking_start", "thinking_delta"):
            yield Activity("thinking", tools_running=self._tools_running)
            return
        if etype == "toolcall_start":
            if self._last_char and self._last_char not in SENTENCE_END:
                # 过渡语没说完就要调工具：补句号让它先念出来，否则要等工具跑完
                self._last_char = "。"
                yield TextDelta("。")
            yield Activity(ev.get("toolName") or "工具", tools_running=self._tools_running)
            return
        if etype == "error":
            reason = str(ev.get("reason") or "")
            detail = str(ev.get("error") or reason)
            self._log("warning", f"pi 报错: {detail[:LOG_SLICE]}")
            kind = classify_error(detail)
            yield Error(kind, detail, voice=Error(kind).voice_message)
            return
