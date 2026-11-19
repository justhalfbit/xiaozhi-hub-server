import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from core.agent.agent_session import Activity, AgentSessionError, Done, Error, ErrorKind, TextDelta
from core.agent.codex_session import CodexSession
from core.providers.llm.codex_cli import codex_cli


def fake_cli(directory, script):
    cli = directory / "codex"
    cli.write_text(f"#!{sys.executable}\n" + script)
    cli.chmod(0o700)
    return CodexSession(str(cli), directory, "", "voice instructions")


@pytest.mark.asyncio
async def test_turn_and_followup_use_the_same_thread(tmp_path):
    session = fake_cli(tmp_path, """
import json, sys
from pathlib import Path
with Path('calls').open('a') as f:
    print(json.dumps([sys.argv[1:], sys.stdin.read()]), file=f)
for event in [
    {'type':'thread.started', 'thread_id':'test-thread'},
    {'type':'item.started', 'item':{'id':'c','type':'command_execution'}},
    {'type':'item.completed', 'item':{'id':'c','type':'command_execution','aggregated_output':'private output'}},
    {'type':'item.completed', 'item':{'id':'m','type':'agent_message','text':'完成'}},
    {'type':'turn.completed'},
]:
    print(json.dumps(event))
""")
    await session.start()
    for prompt in ["检查项目", "继续"]:
        events = [e async for e in session.send(prompt)]
        assert events == [Activity("init"), Activity("tool", 1), Activity("tool", 0), TextDelta("完成"), Done("test-thread")]
    first, followup = map(json.loads, (tmp_path / "calls").read_text().splitlines())
    assert "resume" not in first[0]
    assert followup[0][-3:] == ["resume", "test-thread", "-"]
    assert followup[1] == "继续"
    for args, _ in [first, followup]:
        assert 'sandbox_mode="workspace-write"' in args
        assert 'approval_policy="never"' in args


@pytest.mark.asyncio
@pytest.mark.parametrize("script,expected", [
    ("print(" + repr(json.dumps({"type": "turn.failed", "error": {"message": "401 Unauthorized"}})) + ")", ErrorKind.AUTH),
    ("raise SystemExit(7)", ErrorKind.PROCESS_EXITED),
])
async def test_failure(tmp_path, script, expected):
    session = fake_cli(tmp_path, "import sys\nsys.stdin.read()\n" + script)
    events = [e async for e in session.send("测试")]
    assert isinstance(events[-1], Error)
    assert events[-1].kind == expected


@pytest.mark.asyncio
async def test_interrupt_reaps_process(tmp_path):
    session = fake_cli(tmp_path, """import sys,time
sys.stdin.read()
print('{"type":"thread.started","thread_id":"x"}',flush=True)
time.sleep(20)
""")
    started = asyncio.Event()

    async def collect():
        async for _ in session.send("等待"):
            started.set()

    task = asyncio.create_task(collect())
    await asyncio.wait_for(started.wait(), 5)
    await session.interrupt()
    await asyncio.wait_for(task, 5)
    assert session.process is None


@pytest.mark.asyncio
async def test_saved_default_can_load_before_login(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_cli.LLMProvider, "_start_loop", lambda self: None)
    monkeypatch.setattr(codex_cli, "installation", lambda _: {"installed": True, "logged_in": False})
    provider = codex_cli.LLMProvider({"workdir": str(tmp_path)})
    with pytest.raises(AgentSessionError) as failure:
        await provider._ensure_agent(SimpleNamespace(agent=None), "你好", None, "")
    assert failure.value.kind == ErrorKind.AUTH
