"""Run Codex in the configured workspace using the local CLI agent lifecycle."""
import asyncio
from pathlib import Path

from config.config_loader import get_project_dir
from core.agent.agent_session import AgentSessionError, ErrorKind
from core.agent.codex_session import CodexSession, installation, voice_error
from core.agent.session_store import SessionStore
from core.providers.llm.claude_cli.claude_cli import LLMProvider as LocalAgentProvider, VOICE_RULES


class LLMProvider(LocalAgentProvider):
    AGENT_LABEL = SPOKEN_NAME = "Codex"
    REPORT_ID = "codex_cli"

    def __init__(self, config):
        self.model = config.get("model") or ""
        self.model_name = "codex_cli:" + (self.model or "default")
        self.workdir = Path(config.get("workdir") or "~/xiaozhi-workspace").expanduser().resolve()
        self.cli_path = None
        self._init_common(config)
        self.store = SessionStore(Path(get_project_dir()) / "data" / "agent_sessions.json",
                                  f"codex_cli:{self.workdir}")
        self._start_loop()

    async def _ensure_agent(self, st, text, out, system_prompt):
        if not st.agent or not st.agent.alive:
            status = await asyncio.to_thread(installation, self.workdir)
            if not status["installed"]:
                raise AgentSessionError(ErrorKind.NOT_INSTALLED)
            if not status["logged_in"]:
                raise AgentSessionError(ErrorKind.AUTH)
            if not status["workspace_ready"]:
                raise ValueError("Codex 工作目录不可用")
            self.cli_path = status["path"]
        return await super()._ensure_agent(st, text, out, system_prompt)

    def _make_session(self, system_prompt, resume_id=None):
        return CodexSession(self.cli_path, self.workdir, self.model, (system_prompt or "") + "\n" + VOICE_RULES,
                            resume_id)

    def _voice_error(self, kind):
        return voice_error(kind)
