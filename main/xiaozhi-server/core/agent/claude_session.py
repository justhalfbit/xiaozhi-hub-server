"""基于 claude-agent-sdk 的 Claude Code 会话：一个常驻 CLI 进程，多轮对话、可打断、可接续。

注意：SDK 固定为 0.1.81。更新版本依赖 mcp>=1.23，而原项目锁定 mcp==1.22.0，
升级 mcp 到 2.x 会让原项目的 MCP 客户端导入失败。
"""

import asyncio
from core.agent import cli_proxy
import json
import os
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    CLINotFoundError,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    RateLimitEvent,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk import ClaudeSDKError

from core.agent.agent_session import (
    STARTING_LABELS,
    Activity,
    AgentEvent,
    AgentSession,
    AgentSessionError,
    Done,
    Error,
    ErrorKind,
    TextDelta,
)

SENTENCE_END = set("。！？!?；;…~～.\n")

# 只读的工具：Auto 模式下即使走到人工审批这一步也可以放行
READ_ONLY_TOOLS = {"Read", "Grep", "Glob", "WebSearch", "WebFetch", "TodoWrite"}
# 工具参数里表示文件或目录的字段
PATH_KEYS = ("file_path", "path", "notebook_path")


GLOB_META = re.compile(r"[*?\[{]")


def _escapes_root(root: str, value: str) -> bool:
    target = os.path.realpath(os.path.join(root, os.path.expanduser(value)))
    try:
        return os.path.commonpath([root, target]) != root
    except ValueError:  # Windows 上两个路径不在同一个盘
        return True


def _glob_escapes(root: str, pattern: str) -> bool:
    """Glob 的 pattern 本身也能指定位置：绝对路径、~、..，或藏在大括号里的绝对路径（如 {/etc,.}/*）。"""
    if re.search(r"(^|[/\\])\.\.([/\\]|$)", pattern) or re.search(r"[{,]\s*[/\\~]", pattern):
        return True
    if not (os.path.isabs(pattern) or pattern.startswith("~")):
        return False
    m = GLOB_META.search(pattern)
    fixed = pattern[:m.start()] if m else pattern
    return _escapes_root(root, os.path.dirname(fixed) if m else fixed)


def outside_workdir(tool_input: dict, workdir: Optional[str], tool_name: Optional[str] = None) -> Optional[str]:
    """工具要访问的路径在工作目录之外时，返回这个路径；否则返回 None。"""
    if not workdir or not isinstance(tool_input, dict):
        return None
    root = os.path.realpath(os.path.expanduser(workdir))
    for key in PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip() and _escapes_root(root, value):
            return value
    # Grep 的 pattern 是正则表达式，不是路径，只检查 Glob
    pattern = tool_input.get("pattern")
    if tool_name == "Glob" and isinstance(pattern, str) and _glob_escapes(root, pattern.strip()):
        return pattern
    return None

ASSISTANT_ERROR_KINDS = {
    "authentication_failed": ErrorKind.AUTH,
    "billing_error": ErrorKind.BILLING,
    "rate_limit": ErrorKind.RATE_LIMIT,
    "server_error": ErrorKind.SERVER,
}


@dataclass
class ClaudeSessionConfig:
    cli_path: Optional[str] = None
    model: Optional[str] = None
    workdir: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=lambda: ["Read", "Grep", "Glob"])
    permission_mode: str = "default"
    effort: Optional[str] = None  # 思考力度：low / medium / high / xhigh / max，None 表示用 Claude Code 默认值
    system_prompt_append: str = ""


class ClaudeSession(AgentSession):
    def __init__(self, config: ClaudeSessionConfig, resume_id: Optional[str] = None, logger=None):
        self.config = config
        self.resume_id = resume_id
        self.logger = logger
        self._session_id = resume_id or str(uuid.uuid4())
        self._client: Optional[ClaudeSDKClient] = None
        self._interrupting = False
        self._stderr_tail = deque(maxlen=20)

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    # ---------- 生命周期 ----------

    def _options(self) -> ClaudeAgentOptions:
        allowed = list(self.config.allowed_tools)

        mode = self.config.permission_mode
        workdir = self.config.workdir

        def deny_reason(name, tool_input):
            if name not in allowed:
                return "语音模式下不允许使用这个工具"
            path = outside_workdir(tool_input, workdir, name)
            if path:
                return f"语音模式下只能访问工作目录里的文件，不能访问 {path}"
            return None

        async def pre_tool_use(input_data, tool_use_id, context):
            # 每次调用工具前都会执行，不管权限模式是什么。实测 Claude Code 读工作目录外的文件、
            # 以及 Auto 模式放行的操作都不会经过 can_use_tool，所以硬性限制放在这里
            reason = deny_reason(input_data.get("tool_name"), input_data.get("tool_input"))
            if reason:
                return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                               "permissionDecision": "deny", "permissionDecisionReason": reason}}
            return {}

        async def can_use_tool(name, tool_input, context):
            # Claude Code 需要人工审批时才会走到这里。语音场景没有人能点"同意"：
            # 工具列表之外、工作目录之外的一律拒绝；Auto 模式下它自己没有直接放行的操作（只读工具除外）也拒绝
            reason = deny_reason(name, tool_input)
            if reason:
                return PermissionResultDeny(message=reason)
            if mode == "auto" and name not in READ_ONLY_TOOLS:
                return PermissionResultDeny(message="这个操作需要人工确认，语音模式下不执行")
            return PermissionResultAllow()

        kwargs = dict(
            env=cli_proxy.overrides(),
            cli_path=self.config.cli_path,
            cwd=self.config.workdir,
            model=self.config.model or None,
            tools=allowed,
            allowed_tools=allowed,
            permission_mode=self.config.permission_mode,
            can_use_tool=can_use_tool,
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[pre_tool_use])]},
            strict_mcp_config=True,  # 不加载账号里的 claude.ai 连接器（Gmail、Drive 等）和其他 MCP 服务器
            include_partial_messages=True,
            setting_sources=[],  # 不加载用户的 hooks、插件等设置，行为可预期
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": self.config.system_prompt_append,
            },
            stderr=self._stderr_tail.append,
        )
        if self.config.effort:
            kwargs["effort"] = self.config.effort
        if self.resume_id:
            kwargs["resume"] = self.resume_id
        else:
            kwargs["session_id"] = self._session_id
        return ClaudeAgentOptions(**kwargs)

    async def start(self) -> None:
        client = ClaudeSDKClient(options=self._options())
        try:
            await client.connect()
        except CLINotFoundError as e:
            raise AgentSessionError(ErrorKind.NOT_INSTALLED, str(e))
        except ClaudeSDKError as e:
            detail = f"{e}\n{self.stderr_tail}".strip()
            if self.resume_id:
                raise AgentSessionError(ErrorKind.SESSION_MISSING, detail)
            raise AgentSessionError(ErrorKind.PROCESS_EXITED, detail)
        self._client = client

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        try:
            await asyncio.wait_for(client.disconnect(), timeout=5)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"关闭 Claude Code 进程时出错: {e}")

    async def interrupt(self) -> None:
        if self._client is None:
            return
        self._interrupting = True
        try:
            await self._client.interrupt()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"中断 Claude Code 失败: {e}")

    @property
    def alive(self) -> bool:
        return self._client is not None

    # ---------- 日志 ----------

    def _log(self, level: str, message: str) -> None:
        if self.logger:
            getattr(self.logger, level)(f"[{self._session_id[:8]}] {message}")

    def _log_message(self, msg) -> None:
        """把 SDK 的非文字事件记到日志里，方便排查卡顿、限流、工具出错等问题。"""
        if isinstance(msg, AssistantMessage):
            prefix = "子代理" if msg.parent_tool_use_id else ""
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    brief = json.dumps(block.input, ensure_ascii=False)[:160]
                    self._log("info", f"{prefix}调用工具 {block.name} {brief}")
                elif isinstance(block, ThinkingBlock):
                    self._log("debug", f"{prefix}思考 {len(block.thinking or '')} 字")
            if msg.error:
                self._log("warning", f"assistant 错误 {msg.error}")
        elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ToolResultBlock) and block.is_error:
                    self._log("warning", f"工具返回错误：{str(block.content)[:200]}")
        elif isinstance(msg, RateLimitEvent):
            info = msg.rate_limit_info
            self._log("warning" if getattr(info, "status", "") not in ("allowed", None) else "info",
                      f"限流状态 status={getattr(info, 'status', None)} type={getattr(info, 'rate_limit_type', None)} "
                      f"utilization={getattr(info, 'utilization', None)} resets_at={getattr(info, 'resets_at', None)}")
        elif isinstance(msg, SystemMessage):
            data = msg.data or {}
            if msg.subtype == "init":
                self._log("debug", f"system init model={data.get('model')} tools={data.get('tools')}")
            elif msg.subtype in ("status", "thinking_tokens"):
                self._log("debug", f"system {msg.subtype} {data.get('status') or data.get('estimated_tokens')}")
            else:
                self._log("info", f"system {msg.subtype} {json.dumps(data, ensure_ascii=False, default=str)[:200]}")
        elif isinstance(msg, ResultMessage):
            self._log("info" if not msg.is_error else "warning",
                      f"本轮结果 {msg.subtype} 轮数={msg.num_turns} API耗时={msg.duration_api_ms}ms "
                      f"费用={msg.total_cost_usd} 出错={msg.is_error}")

    # ---------- 一轮对话 ----------

    async def send(self, text: str) -> AsyncIterator[AgentEvent]:
        if self._client is None:
            yield Error(ErrorKind.PROCESS_EXITED, "session not started")
            return
        self._interrupting = False
        streamed = False  # 当前这条 assistant 消息是否已经通过逐字流输出过
        last_char = ""
        last_activity = 0.0
        pending_tools = set()  # 已调用、还没返回结果的工具（只看主对话，子代理的工具包在 Task 工具里）
        try:
            await self._client.query(text)
            async for msg in self._client.receive_response():
                self._log_message(msg)
                before = len(pending_tools)
                if isinstance(msg, AssistantMessage) and msg.parent_tool_use_id is None:
                    pending_tools.update(b.id for b in msg.content if isinstance(b, ToolUseBlock))
                elif isinstance(msg, UserMessage) and msg.parent_tool_use_id is None and isinstance(msg.content, list):
                    pending_tools.difference_update(b.tool_use_id for b in msg.content if isinstance(b, ToolResultBlock))
                is_text = isinstance(msg, StreamEvent) and ((msg.event or {}).get("delta") or {}).get("type") == "text_delta"
                # 初始化单独报一次 label 为 init 的 Activity：这时会话已经建好、用户的话已经存进去，但 Claude 还没开始干活。
                # 实测没登录时是 init → 伪装成 assistant 的错误 → result；和 Anthropic 的网络卡住时只有 init，之后没有动静。
                # 上层靠 init 之后来的是错误还是别的事件来区分这两种情况（见 claude_cli 的 _run_turn）。
                # 限流（429）、过载（529）时实测是 init → api_retry × N → 伪装成 assistant 的错误 → result：重试也还没开始干活，
                # 同样单独报（label 为 api_retry，不节流）。重试间隔会越来越长，照常报 Activity 才不会被当成卡住
                if isinstance(msg, SystemMessage) and msg.subtype in STARTING_LABELS:
                    last_activity = time.monotonic()
                    yield Activity(msg.subtype, tools_running=len(pending_tools))
                    continue
                # 不算在干活：限流信息、收尾的结果，以及伪装成 assistant 消息的错误（没登录、额度用完等），不报 Activity
                not_work = isinstance(msg, (RateLimitEvent, ResultMessage)) or (
                    isinstance(msg, AssistantMessage) and bool(msg.error or msg.model == "<synthetic>"))
                # 节流：每秒最多一次；正在执行的工具数变了要马上告诉调用方
                if not not_work and ((not is_text and time.monotonic() - last_activity >= 1)
                                     or len(pending_tools) != before):
                    last_activity = time.monotonic()
                    yield Activity(type(msg).__name__, tools_running=len(pending_tools))
                if isinstance(msg, StreamEvent):
                    if msg.parent_tool_use_id is not None:
                        continue  # 子代理的输出不读
                    ev = msg.event or {}
                    delta = ev.get("delta") or {}
                    if ev.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                        piece = delta.get("text") or ""
                        if piece:
                            if not streamed and last_char and last_char not in SENTENCE_END:
                                yield TextDelta("。")  # 两段话之间补一个句号，方便 TTS 断句
                            streamed = True
                            last_char = piece[-1]
                            yield TextDelta(piece)
                elif isinstance(msg, AssistantMessage):
                    if msg.parent_tool_use_id is not None:
                        continue
                    if msg.error or msg.model == "<synthetic>":
                        # 额度、鉴权等错误会伪装成一条 assistant 消息，不能把原文念出来
                        detail = " ".join(b.text for b in msg.content if isinstance(b, TextBlock))
                        yield Error(ASSISTANT_ERROR_KINDS.get(msg.error or "", ErrorKind.UNKNOWN), detail)
                        streamed = False
                        continue
                    if not streamed:
                        # 没有收到逐字流时，才使用整条消息的文字，避免同一段话念两遍
                        for block in msg.content:
                            if isinstance(block, TextBlock) and block.text:
                                if last_char and last_char not in SENTENCE_END:
                                    yield TextDelta("。")
                                last_char = block.text[-1]
                                yield TextDelta(block.text)
                    if last_char and last_char not in SENTENCE_END and any(
                            isinstance(b, ToolUseBlock) for b in msg.content):
                        # 过渡语没有句末标点（如"我先看一下文件""我跑一下这个命令："）就去调用工具了：补一个句号，
                        # TTS 才会马上念出来；否则要等工具跑完、下一段话来了才念，期间也插不进进度提示
                        last_char = "。"
                        yield TextDelta("。")
                    streamed = False
                elif isinstance(msg, ResultMessage):
                    if msg.session_id:
                        self._session_id = msg.session_id
                    if msg.is_error and not self._interrupting:
                        yield Error(ErrorKind.UNKNOWN, f"{msg.subtype}: {msg.errors or msg.result or ''}")
                    else:
                        yield Done(session_id=self._session_id, interrupted=self._interrupting)
                    return
            yield Done(session_id=self._session_id, interrupted=self._interrupting)
        except ClaudeSDKError as e:
            # 有些错误（如输出解析失败）发生时进程还活着，必须关掉，否则会留下孤儿进程
            await self.close()
            yield Error(ErrorKind.PROCESS_EXITED, f"{e}\n{self.stderr_tail}".strip())
        finally:
            self._interrupting = False
