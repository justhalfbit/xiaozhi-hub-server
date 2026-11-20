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
    monkeypatch.setattr(codex_cli, "installation", lambda _, **kw: {"installed": True, "logged_in": False})
    provider = codex_cli.LLMProvider({"workdir": str(tmp_path)})
    with pytest.raises(AgentSessionError) as failure:
        await provider._ensure_agent(SimpleNamespace(agent=None), "你好", None, "")
    assert failure.value.kind == ErrorKind.AUTH


@pytest.mark.parametrize("level", ["read", "edit"])
def test_workspace_files_exclude_links_and_private_paths(tmp_path, level):
    from core.agent.codex_files import Workspace
    import os

    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside")
    (project / "link").symlink_to(outside)
    (project / "parent").symlink_to(tmp_path, target_is_directory=True)
    os.link(outside, project / "hard")
    workspace = Workspace(project)
    try:
        for path in ["../outside", str(outside), "link", "parent/outside", "hard", ".git/config"]:
            with pytest.raises((OSError, ValueError)):
                workspace.read_file(path)
            if level == "edit":
                with pytest.raises((OSError, ValueError)):
                    workspace.write_file(path, "changed")
        (project / "plain").write_text("one")
        assert workspace.read_file("plain") == "one"
        assert workspace.search_files("one") == ["plain:1: one"]
        if level == "edit":
            workspace.write_file("plain", "two")
            assert (project / "plain").read_text() == "two"
        assert outside.read_text() == "outside"
    finally:
        os.close(workspace.root)


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["read", "edit", "auto"])
async def test_each_invocation_enforces_selected_policy(tmp_path, level):
    path = tmp_path / "args"
    session = fake_cli(tmp_path, f"""
import json, sys
sys.stdin.read()
with open({str(path)!r}, 'a') as f: print(json.dumps(sys.argv),file=f)
print('{{"type":"thread.started","thread_id":"t"}}')
print('{{"type":"turn.completed"}}')
""")
    session.level = level
    session.web = level == "edit"
    session.effort = "low"
    for prompt in ("first", "followup"):
        assert isinstance([ev async for ev in session.send(prompt)][-1], Done)
    for line in path.read_text().splitlines():
        argv = json.loads(line)
        assert 'model_reasoning_effort="low"' in argv
        assert ('web_search="live"' if session.web else 'web_search="disabled"') in argv
        if level == "auto":
            assert 'sandbox_mode="workspace-write"' in argv
        else:
            assert "shell_tool" in argv and "--ignore-user-config" in argv
            assert 'default_permissions="hub_files"' in argv
            args = next(arg for arg in argv if arg.startswith("mcp_servers.workspace.args="))
            assert json.loads(args.split("=", 1)[1])[-2:] == [str(tmp_path), level]
