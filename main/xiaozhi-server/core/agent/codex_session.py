"""Non-interactive Codex CLI transport for the Hub agent lifecycle."""
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from core.agent.agent_session import Activity, AgentSession, AgentSessionError, Done, Error, ErrorKind, TextDelta


def permission_level(entry):
    level = entry.get("level") or ("auto" if entry.get("enabled") else "read")
    if level not in ("read", "edit", "auto"):
        raise ValueError("不支持的 Codex 权限级别")
    return level


def installation(workdir, *, check_login=True, cli_path="", level="auto"):
    cli = shutil.which("codex") or shutil.which("codex", path=str(Path.home() / ".local/bin"))
    if cli_path:
        cli = shutil.which(str(Path(cli_path).expanduser()))
    logged_in = False
    if cli and check_login:
        try:
            logged_in = subprocess.run([cli, "login", "status"], capture_output=True, timeout=5).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            pass
    directory = Path(workdir).expanduser()
    return {"path": cli, "installed": bool(cli), "logged_in": logged_in,
            "workspace_ready": directory.is_dir() and os.access(directory, (os.R_OK if level == "read" else os.W_OK | os.R_OK) | os.X_OK)}


def voice_error(kind):
    return Error(kind).voice_message.replace("Claude Code", "Codex").replace("Claude", "Codex")


class CodexSession(AgentSession):
    def __init__(self, cli, workdir, model, instructions, resume_id=None, *, level="auto", effort="", web=False):
        self.level = level
        self.effort = effort
        self.web = web
        self.cli = cli
        self.workdir = workdir
        self.model = model
        self.instructions = instructions
        self._id = resume_id
        self.process = None
        self.alive = False

    @property
    def session_id(self):
        return self._id

    async def start(self):
        if not self.cli:
            raise AgentSessionError(ErrorKind.NOT_INSTALLED)
        self.alive = True

    async def send(self, text):
        if self.level == "auto":
            async for event in self._execute(text, self.workdir):
                yield event
        else:
            # Keep project-local Codex config, hooks and skills out of restricted sessions.
            with tempfile.TemporaryDirectory(prefix="hub-codex-") as neutral:
                async for event in self._execute(text, neutral):
                    yield event

    async def _execute(self, text, cwd):
        args = [self.cli, "exec", "--json", "--skip-git-repo-check"]
        settings = {"approval_policy": "never", "web_search": "live" if self.web else "disabled",
                    "developer_instructions": self.instructions}
        if self.effort:
            settings["model_reasoning_effort"] = self.effort
        if self.level == "auto":
            settings["sandbox_mode"] = "workspace-write"
        elif self.level in ("read", "edit"):
            args.extend(["--ignore-user-config", "--ignore-rules"])
            for feature in ("shell_tool", "code_mode", "apps", "plugins", "hooks", "computer_use",
                            "multi_agent", "multi_agent_v2", "image_generation"):
                args.extend(["--disable", feature])
            args.extend(["-c", 'permissions.hub_files.filesystem={ ":minimal" = "read" }'])
            settings.update({
                "default_permissions": "hub_files",
                "permissions.hub_files.network.enabled": False,
                "mcp_servers.workspace.command": sys.executable,
                "mcp_servers.workspace.args": ["-I", str(Path(__file__).with_name("codex_files.py").resolve()),
                                               str(self.workdir), self.level],
                "mcp_servers.workspace.required": True,
                "mcp_servers.workspace.default_tools_approval_mode": "approve",
                "developer_instructions": self.instructions + "\n请使用 workspace 文件工具访问项目。",
            })
        else:
            raise ValueError("不支持的 Codex 权限级别")
        for key, value in settings.items():
            args.extend(["-c", key + "=" + json.dumps(value, ensure_ascii=False)])
        if self.model:
            args.extend(["--model", self.model])
        if self._id:
            args.extend(["resume", self._id])
        args.append("-")
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=cwd, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            start_new_session=os.name == "posix", limit=8 * 1024 * 1024)
        self.process = proc
        running = set()
        completed = False
        try:
            proc.stdin.write(text.encode())
            await proc.stdin.drain()
            proc.stdin.close()
            async for line in proc.stdout:
                event = json.loads(line)
                typ = event.get("type")
                item = event.get("item") or {}
                if typ == "thread.started":
                    self._id = event["thread_id"]
                    yield Activity("init")
                elif typ == "turn.failed":
                    detail = str((event.get("error") or {}).get("message", "")).lower()
                    auth = any(word in detail for word in ("401", "unauthorized", "login", "authentication"))
                    kind = ErrorKind.AUTH if auth else ErrorKind.SERVER
                    yield Error(kind, voice=voice_error(kind))
                    return
                elif typ == "turn.completed":
                    completed = True
                elif item.get("type") == "agent_message" and typ == "item.completed":
                    yield TextDelta(item.get("text", ""))
                elif item.get("type") in {"command_execution", "file_change", "mcp_tool_call", "web_search"}:
                    if typ == "item.started":
                        running.add(item["id"])
                    elif typ == "item.completed":
                        running.discard(item["id"])
                    yield Activity("tool", len(running))
                elif typ == "error":
                    yield Activity("api_retry")
            if await proc.wait() == 0 and completed:
                yield Done(self._id)
            else:
                yield Error(ErrorKind.PROCESS_EXITED, voice=voice_error(ErrorKind.PROCESS_EXITED))
        finally:
            await self.interrupt()
            self.process = None

    async def interrupt(self):
        proc = self.process
        if proc is None or proc.returncode is not None:
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, sig)
                else:
                    proc.kill()
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(proc.wait(), 2)
                return
            except asyncio.TimeoutError:
                continue

    async def close(self):
        self.alive = False
        await self.interrupt()
