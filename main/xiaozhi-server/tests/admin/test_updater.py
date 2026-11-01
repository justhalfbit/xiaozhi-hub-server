"""后台「更新网关」：检查新版本、在后台跑 scripts/update.sh、失败退回原版本、重启后沿用登录。

用临时 git 仓库模拟：remote.git 是 GitHub，install 是装好的网关。update.sh 用仓库里真的那份，
install.sh、autostart.sh 换成桩：把调用记到 HUB_TEST_CALLS，按环境变量模拟失败。
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from core.admin import auth, updater
from core.admin.admin_handler import AdminHandler

REAL_UPDATE_SH = Path(updater.__file__).resolve().parents[4] / "scripts" / "update.sh"

INSTALL_STUB = """#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBJECT="$(git -C "$ROOT" log -1 --format=%s)"
echo "install $SUBJECT HUB_UPDATING=${HUB_UPDATING:-}" >>"$HUB_TEST_CALLS"
# HUB_TEST_INSTALL_FAIL：提交说明里带这段文字的版本装不上；all 是哪个版本都装不上
if [[ -n "${HUB_TEST_INSTALL_FAIL:-}" && ( "$HUB_TEST_INSTALL_FAIL" == all || "$SUBJECT" == *"$HUB_TEST_INSTALL_FAIL"* ) ]]; then printf "\\n\\033[31m✗ 模拟：依赖下载失败\\033[0m\\n" >&2; exit 1; fi
: >"$ROOT/main/xiaozhi-server/data/.install-complete"
"""

# 提交说明里带 broken 的版本"起不来"：restart 返回失败
AUTOSTART_STUB = """#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$1" in
  status) [[ -n "${HUB_TEST_AUTOSTART:-}" ]] ;;
  restart)
    echo "restart $(git -C "$ROOT" log -1 --format=%s) timeout=${HUB_READY_TIMEOUT:-}" >>"$HUB_TEST_CALLS"
    [[ "$(git -C "$ROOT" log -1 --format=%s)" != *broken* ]] || { echo "✗ 服务 300 秒内没有启动成功" >&2; exit 1; } ;;
esac
"""


def git(cwd, *args) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repos(tmp_path, monkeypatch):
    """返回 (seed, install, calls)：seed 里提交并推送就是"GitHub 上出了新版本"。"""
    for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("INVOCATION_ID", raising=False)  # 不走 systemd-run
    calls = tmp_path / "calls.txt"
    calls.touch()
    monkeypatch.setenv("HUB_TEST_CALLS", str(calls))
    monkeypatch.setenv("HUB_TEST_AUTOSTART", "1")
    for k in ("HUB_TEST_INSTALL_FAIL", "HUB_READY_TIMEOUT"):
        monkeypatch.delenv(k, raising=False)

    seed = tmp_path / "seed"
    (seed / "scripts").mkdir(parents=True)
    (seed / "main" / "xiaozhi-server").mkdir(parents=True)
    shutil.copy(REAL_UPDATE_SH, seed / "scripts" / "update.sh")
    (seed / "scripts" / "install.sh").write_text(INSTALL_STUB)
    (seed / "scripts" / "autostart.sh").write_text(AUTOSTART_STUB)
    (seed / "main" / "xiaozhi-server" / "requirements-hub.txt").write_text("aiohttp\n")
    (seed / "README.md").write_text("v1\n")
    (seed / ".gitignore").write_text("main/xiaozhi-server/data/\nmain/xiaozhi-server/tmp/\n")
    git(seed, "init", "-q", "-b", "main")
    git(seed, "add", "-A")
    git(seed, "commit", "-qm", "v1")
    remote = tmp_path / "remote.git"
    git(tmp_path, "clone", "-q", "--bare", str(seed), str(remote))
    git(seed, "remote", "add", "gh", str(remote))

    install = tmp_path / "install"
    git(tmp_path, "clone", "-q", str(remote), str(install))
    (install / "main" / "xiaozhi-server" / "data").mkdir()
    (install / "main" / "xiaozhi-server" / "data" / ".install-complete").touch()
    return seed, install, calls


def publish(seed: Path, subject: str, files: dict = None) -> str:
    """在"GitHub"上发一个新提交，返回提交号。"""
    for name, text in (files or {"README.md": subject + "\n"}).items():
        (seed / name).write_text(text)
    git(seed, "commit", "-qam", subject)
    git(seed, "push", "-q", "gh", "main")
    return git(seed, "rev-parse", "HEAD")


def make_updater(install: Path) -> updater.Updater:
    return updater.Updater(install, state_dir=install / "main" / "xiaozhi-server" / "tmp")


def wait_done(u: updater.Updater, timeout=30) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = u.status()
        if st["state"] != "running":
            return st
        time.sleep(0.1)
    raise AssertionError("更新没有结束：" + "\n".join(u.status()["log"]))


# ---------- 检查更新 ----------

def test_version_and_latest(repos):
    _, install, _ = repos
    u = make_updater(install)
    v = u.version()
    assert v["unsupported"] == "" and v["branch"] == "main"
    assert v["current"]["subject"] == "v1" and v["source"]["remote"] == "origin" and v["source"]["branch"] == "main"
    c = u.check()
    assert c["state"] == "latest" and c["behind"] == 0 and c["checked_at"]
    assert u.last_check is c


def test_check_lists_new_commits_without_touching_code(repos):
    seed, install, _ = repos
    head = git(install, "rev-parse", "HEAD")
    publish(seed, "修复 A")
    new = publish(seed, "新功能 B")
    c = make_updater(install).check()
    assert c["state"] == "available" and c["behind"] == 2
    assert [x["subject"] for x in c["commits"]] == ["新功能 B", "修复 A"]
    assert c["latest"]["commit"] == new
    assert git(install, "rev-parse", "HEAD") == head  # 只检查，不动代码


def test_check_follows_branch_upstream(repos, tmp_path):
    """分支设了上游就从上游更新（设备上用测试 remote 模拟新版本就靠它）。"""
    seed, install, _ = repos
    other = tmp_path / "other.git"
    git(tmp_path, "clone", "-q", "--bare", str(seed), str(other))
    git(install, "remote", "add", "test", str(other))
    git(install, "fetch", "-q", "test")
    git(install, "checkout", "-q", "-b", "dev", "--track", "test/main")
    git(seed, "remote", "add", "other", str(other))
    (seed / "README.md").write_text("only on other\n")
    git(seed, "commit", "-qam", "只在 other 上")
    git(seed, "push", "-q", "other", "main")
    c = make_updater(install).check()
    assert c["source"]["remote"] == "test" and c["source"]["branch"] == "main"
    assert c["state"] == "available" and c["commits"][0]["subject"] == "只在 other 上"


def test_check_dirty_and_start_refuses(repos):
    seed, install, _ = repos
    publish(seed, "v2")
    (install / "README.md").write_text("我改过\n")
    u = make_updater(install)
    c = u.check()
    assert c["state"] == "dirty" and c["files"] == ["README.md"] and c["behind"] == 1
    with pytest.raises(updater.UpdateError, match="README.md"):
        u.start()
    assert u.status() == {"state": "idle"}


def test_check_diverged(repos):
    seed, install, _ = repos
    publish(seed, "v2")
    (install / "README.md").write_text("local\n")
    git(install, "commit", "-qam", "本地提交")
    c = make_updater(install).check()
    assert c["state"] == "diverged" and "分叉" in c["message"]
    git(install, "reset", "-q", "--hard", "origin/main")
    git(install, "fetch", "-q")
    git(install, "reset", "-q", "--hard", "FETCH_HEAD")
    (install / "README.md").write_text("newer\n")
    git(install, "commit", "-qam", "比远端还新")
    c = make_updater(install).check()
    assert c["state"] == "diverged" and "比远端还新" in c["message"]


def test_check_network_errors(repos, monkeypatch):
    _, install, _ = repos
    git(install, "remote", "set-url", "origin", str(install.parent / "missing.git"))
    c = make_updater(install).check()
    assert c["state"] == "error" and c["message"]

    git(install, "remote", "set-url", "origin", str(install.parent / "remote.git"))
    git(install, "config", "branch.main.merge", "refs/heads/nope")
    c = make_updater(install).check()
    assert c["state"] == "error" and "没有 nope 分支" in c["message"]

    u = make_updater(install)
    real = u._git

    def slow(*args, timeout=15):
        if "fetch" in args:
            raise subprocess.TimeoutExpired(args, timeout)
        return real(*args, timeout=timeout)

    monkeypatch.setattr(u, "_git", slow)
    c = u.check(timeout=3)
    assert c["state"] == "error" and "超时（3 秒）" in c["message"] and "http.proxy" in c["message"]


def test_unsupported_without_git(tmp_path):
    u = updater.Updater(tmp_path, state_dir=tmp_path / "tmp")
    assert u.version()["unsupported"]
    assert u.check()["state"] == "unsupported"
    with pytest.raises(updater.UpdateError):
        u.start()


# ---------- 更新 ----------

def test_update_success_skips_install_when_deps_unchanged(repos):
    seed, install, calls = repos
    old = git(install, "rev-parse", "HEAD")
    new = publish(seed, "v2 文档")
    u = make_updater(install)
    first = u.start()
    assert first["state"] in ("running", "done")
    st = wait_done(u)
    assert st["state"] == "done" and st["changed"], st["log"]
    assert st["from"]["commit"] == old and st["to"]["commit"] == new
    assert git(install, "rev-parse", "HEAD") == new
    assert "依赖没有变化，跳过安装" in "\n".join(st["log"])
    assert calls.read_text().splitlines() == [f"restart v2 文档 timeout={updater.READY_TIMEOUT}"]
    # 服务被重启过：新进程里看，不用再提示重启
    assert st["restart_needed"] is True
    assert make_updater(install).status()["restart_needed"] is False


def test_update_installs_when_requirements_change(repos):
    seed, install, calls = repos
    new = publish(seed, "加依赖", {"main/xiaozhi-server/requirements-hub.txt": "aiohttp\nrequests\n"})
    u = make_updater(install)
    u.start()
    st = wait_done(u)
    assert st["state"] == "done" and git(install, "rev-parse", "HEAD") == new
    assert calls.read_text().splitlines() == ["install 加依赖 HUB_UPDATING=1", f"restart 加依赖 timeout={updater.READY_TIMEOUT}"]


def test_update_install_failure_keeps_old_version(repos, monkeypatch):
    seed, install, calls = repos
    monkeypatch.setenv("HUB_TEST_INSTALL_FAIL", "加依赖")
    old = git(install, "rev-parse", "HEAD")
    publish(seed, "加依赖", {"main/xiaozhi-server/requirements-hub.txt": "aiohttp\nrequests\n"})
    u = make_updater(install)
    u.start()
    st = wait_done(u)
    assert st["state"] == "failed" and not st["changed"]
    assert "已退回原来的版本" in st["error"]
    assert git(install, "rev-parse", "HEAD") == old
    assert git(install, "status", "--porcelain", "--untracked-files=no") == ""
    assert (install / "main" / "xiaozhi-server" / "data" / ".install-complete").exists()
    # 退回代码后在原来的版本上再装一次，把依赖装回去；没有重启
    assert calls.read_text().splitlines() == ["install 加依赖 HUB_UPDATING=1", "install v1 HUB_UPDATING=1"]


def test_update_install_failure_and_deps_restore_failure(repos, monkeypatch):
    """依赖也没能装回去：代码照样退回，不留完成标记（下次 update.sh 会重新补装），报错说清楚。"""
    seed, install, calls = repos
    monkeypatch.setenv("HUB_TEST_INSTALL_FAIL", "all")
    old = git(install, "rev-parse", "HEAD")
    publish(seed, "加依赖", {"main/xiaozhi-server/requirements-hub.txt": "aiohttp\nrequests\n"})
    u = make_updater(install)
    u.start()
    st = wait_done(u)
    assert st["state"] == "failed" and not st["changed"]
    assert "依赖没能装回原来的版本" in st["error"] and "install.sh" in st["error"]
    assert git(install, "rev-parse", "HEAD") == old
    assert not (install / "main" / "xiaozhi-server" / "data" / ".install-complete").exists()
    assert calls.read_text().splitlines() == ["install 加依赖 HUB_UPDATING=1", "install v1 HUB_UPDATING=1"]


def test_update_rolls_back_when_new_version_does_not_start(repos):
    seed, install, calls = repos
    old = git(install, "rev-parse", "HEAD")
    publish(seed, "broken 版本")
    u = make_updater(install)
    u.start()
    st = wait_done(u)
    assert st["state"] == "failed" and not st["changed"]
    assert "新版本没有启动成功，已退回原来的版本" in st["error"]
    assert git(install, "rev-parse", "HEAD") == old
    assert calls.read_text().splitlines() == [f"restart broken 版本 timeout={updater.READY_TIMEOUT}",
                                              f"restart v1 timeout={updater.READY_TIMEOUT}"]


def test_update_rollback_restores_deps_when_new_version_does_not_start(repos):
    """新版本补装过依赖又起不来：退回代码后先把依赖装回原来的版本，再重启。"""
    seed, install, calls = repos
    old = git(install, "rev-parse", "HEAD")
    publish(seed, "broken 加依赖", {"main/xiaozhi-server/requirements-hub.txt": "aiohttp\nrequests\n"})
    u = make_updater(install)
    u.start()
    st = wait_done(u)
    assert st["state"] == "failed" and not st["changed"]
    assert "新版本没有启动成功，已退回原来的版本" in st["error"]
    assert git(install, "rev-parse", "HEAD") == old
    assert (install / "main" / "xiaozhi-server" / "data" / ".install-complete").exists()
    assert calls.read_text().splitlines() == ["install broken 加依赖 HUB_UPDATING=1",
                                              f"restart broken 加依赖 timeout={updater.READY_TIMEOUT}",
                                              "install v1 HUB_UPDATING=1",
                                              f"restart v1 timeout={updater.READY_TIMEOUT}"]


def test_update_rollback_deps_restore_failure_skips_restart(repos, monkeypatch):
    """依赖没能装回去就不再重启（旧代码配新依赖多半也起不来），报错让用户装回依赖。"""
    seed, install, calls = repos
    monkeypatch.setenv("HUB_TEST_INSTALL_FAIL", "v1")
    publish(seed, "broken 加依赖", {"main/xiaozhi-server/requirements-hub.txt": "aiohttp\nrequests\n"})
    u = make_updater(install)
    u.start()
    st = wait_done(u)
    assert st["state"] == "failed" and "依赖没能装回原来的版本" in st["error"]
    assert not (install / "main" / "xiaozhi-server" / "data" / ".install-complete").exists()
    assert calls.read_text().splitlines()[-1] == "install v1 HUB_UPDATING=1"


def test_update_when_already_latest(repos):
    _, install, calls = repos
    u = make_updater(install)
    u.start()
    st = wait_done(u)
    assert st["state"] == "done" and not st["changed"] and not st["restart_needed"]
    assert calls.read_text() == ""


def test_update_without_autostart_needs_manual_restart(repos, monkeypatch):
    seed, install, calls = repos
    monkeypatch.delenv("HUB_TEST_AUTOSTART")
    publish(seed, "v2")
    u = make_updater(install)
    u.start()
    st = wait_done(u)
    assert st["state"] == "done" and st["changed"] and st["restart_needed"]
    assert calls.read_text() == ""


def test_shallow_clone_update(repos, tmp_path):
    """一键安装是浅克隆（--depth 1 --filter=blob:none）：也能列出新提交、更新过去。"""
    seed, _, calls = repos
    shallow = tmp_path / "shallow"
    git(tmp_path, "clone", "-q", "--depth", "1", "--filter=blob:none", (tmp_path / "remote.git").as_uri(), str(shallow))
    (shallow / "main" / "xiaozhi-server" / "data").mkdir()
    (shallow / "main" / "xiaozhi-server" / "data" / ".install-complete").touch()
    publish(seed, "v2")
    new = publish(seed, "v3")
    u = make_updater(shallow)
    c = u.check()
    assert c["state"] == "available" and c["behind"] == 2
    u.start()
    st = wait_done(u)
    assert st["state"] == "done" and git(shallow, "rev-parse", "HEAD") == new, st["log"]


def shallow_with_local_commit(tmp_path, monkeypatch, name="shallow") -> tuple:
    """浅克隆后在本地提交一次，返回 (仓库, 本地提交)。

    本地提交比和远端共同的那个提交晚（实际总是这样）：按当前版本的时间取远端提交时取不到共同的提交，
    历史接不上，曾被误判成有新版本、更新时直接切过去把本地提交丢掉。提交时间写死晚 1 分钟，
    不靠落在不同的秒里碰运气（原来同一秒内做完就碰巧能过）。
    """
    shallow = tmp_path / name
    git(tmp_path, "clone", "-q", "--depth", "1", (tmp_path / "remote.git").as_uri(), str(shallow))
    (shallow / "main" / "xiaozhi-server" / "data").mkdir()
    (shallow / "main" / "xiaozhi-server" / "data" / ".install-complete").touch()
    (shallow / "README.md").write_text("local\n")
    base_time = int(git(shallow, "log", "-1", "--format=%ct"))
    monkeypatch.setenv("GIT_COMMITTER_DATE", f"@{base_time + 60} +0800")
    git(shallow, "commit", "-qam", "本地提交")
    monkeypatch.delenv("GIT_COMMITTER_DATE")
    return shallow, git(shallow, "rev-parse", "HEAD")


def test_shallow_clone_with_local_commit_is_diverged(repos, tmp_path, monkeypatch):
    """浅克隆但接得上历史时照样能看出分叉，后台的「立即更新」按钮不会出现。"""
    seed, _, calls = repos
    shallow, local = shallow_with_local_commit(tmp_path, monkeypatch)
    publish(seed, "v2")
    c = make_updater(shallow).check()
    assert c["state"] == "diverged" and "本地有自己的提交" in c["message"], c


def test_shallow_clone_with_local_commit_update_refuses(repos, tmp_path, monkeypatch):
    """没先检查、直接跑 update.sh（终端里更新）也要拒绝，不能切到新版本把本地提交丢掉。"""
    seed, _, calls = repos
    shallow, local = shallow_with_local_commit(tmp_path, monkeypatch)
    publish(seed, "v2")
    u = make_updater(shallow)
    u.start()
    st = wait_done(u)
    assert st["state"] == "failed" and "本地有自己的提交" in st["error"], st["log"]
    assert git(shallow, "rev-parse", "HEAD") == local and calls.read_text() == ""


def test_start_refuses_while_running_and_detects_interrupted(repos):
    _, install, _ = repos
    u = make_updater(install)
    u.state_dir.mkdir(parents=True)
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        u._write_state({"started": time.time(), "from": git(install, "rev-parse", "HEAD"), "pid": sleeper.pid})
        assert u.running() and u.status()["state"] == "running"
        with pytest.raises(updater.UpdateError, match="已经在更新"):
            u.start()
        assert u.check()["state"] == "error"
    finally:
        sleeper.kill()
        sleeper.wait()
    st = u.status()
    assert st["state"] == "interrupted" and st["error"]


def test_status_not_interrupted_when_update_ends_while_checking(repos, monkeypatch):
    """读退出码时还没写、查进程时已经结束：是正常结束，不是中断，报错读的是最后一行。"""
    _, install, _ = repos
    u = make_updater(install)
    u.state_dir.mkdir(parents=True)
    u._write_state({"started": time.time(), "from": git(install, "rev-parse", "HEAD"), "pid": 999999})
    u.log_file.write_text("==> 补装依赖和模型\n✗ 安装步骤失败，已退回原来的版本\n", encoding="utf-8")

    def ends_now(pid):
        u.exit_file.write_text("1")
        return False

    monkeypatch.setattr(u, "_alive", ends_now)
    st = u.status()
    assert st["state"] == "failed" and st["error"] == "安装步骤失败，已退回原来的版本"


# ---------- 登录交接 ----------

def test_session_handover(tmp_path):
    path = tmp_path / ".admin-sessions.json"
    old = auth.SessionStore()
    token = old.create()
    old.save(path)
    assert token not in path.read_text()  # 只存哈希
    assert (path.stat().st_mode & 0o777) == 0o600
    new = auth.SessionStore()
    assert new.restore(path) == 1 and new.valid(token)
    assert not path.exists()  # 读完就删

    old.save(path)
    data = json.loads(path.read_text())
    data["saved"] -= 3600
    path.write_text(json.dumps(data))
    assert auth.SessionStore().restore(path) == 0  # 太久以前的不认


# ---------- HTTP 接口 ----------

class FakeUpdater:
    def __init__(self):
        self.last_check = None
        self.started = False
        self.refuse = ""

    def version(self):
        return {"current": {"short": "abc1234", "commit": "a" * 40}, "running": {}, "branch": "main",
                "source": {"remote": "origin", "branch": "main", "url": "https://github.com/x/y.git"}, "unsupported": ""}

    def status(self):
        return {"state": "running", "step": "检查新版本"} if self.started else {"state": "idle"}

    def check(self):
        self.last_check = {"state": "available", "behind": 1, "commits": [{"short": "def5678", "subject": "新"}],
                           "checked_at": 1, "current": self.version()["current"]}
        return self.last_check

    def start(self):
        if self.refuse:
            raise updater.UpdateError(self.refuse)
        self.started = True
        return self.status()


@pytest.fixture
async def update_client(tmp_path):
    cfg = tmp_path / "data" / ".config.yaml"
    cfg.parent.mkdir()
    cfg.write_text("{}\n", encoding="utf-8")
    fake = FakeUpdater()
    app = web.Application()
    AdminHandler({"selected_module": {}}, config_path=cfg, gateway_updater=fake).register(app)
    async with TestClient(TestServer(app)) as c:
        yield c, fake, cfg.parent / ".admin-sessions.json"


async def test_update_api_requires_login(update_client):
    c, fake, _ = update_client
    assert (await c.get("/admin/api/update")).status == 401
    assert (await c.post("/admin/api/update/check", json={})).status == 401
    assert (await c.post("/admin/api/update/start", json={})).status == 401
    assert not fake.started


async def test_update_api_flow(update_client):
    c, fake, handover = update_client
    await c.post("/admin/api/setup", json={"password": "pass-1234"})
    d = await (await c.get("/admin/api/update")).json()
    assert d["ok"] and d["current"]["short"] == "abc1234" and d["check"] is None and d["update"]["state"] == "idle"
    # 写操作只收 JSON（防跨站提交）
    r = await c.post("/admin/api/update/start", data="x=1", headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert r.status == 415 and not fake.started

    d = await (await c.post("/admin/api/update/check", json={})).json()
    assert d["check"]["state"] == "available"
    assert (await (await c.get("/admin/api/update")).json())["check"]["behind"] == 1
    fake.last_check["current"] = {"short": "0000000", "commit": "0" * 40}  # 检查之后代码变了：结果作废
    assert (await (await c.get("/admin/api/update")).json())["check"] is None
    fake.check()

    fake.refuse = "这些文件被改过"
    r = await c.post("/admin/api/update/start", json={})
    assert r.status == 409 and "改过" in (await r.json())["error"] and not handover.exists()

    fake.refuse = ""
    d = await (await c.post("/admin/api/update/start", json={})).json()
    assert d["ok"] and d["update"]["state"] == "running"
    assert handover.exists()  # 重启后的新进程据此沿用登录

    # 模拟网关重启：新的处理器读入交接文件，原来的 cookie 仍然有效
    app = web.Application()
    AdminHandler({"selected_module": {}}, config_path=handover.parent / ".config.yaml",
                 gateway_updater=FakeUpdater()).register(app)
    cookie = c.session.cookie_jar.filter_cookies(c.make_url("/admin"))["xz_admin"].value
    async with TestClient(TestServer(app)) as c2:
        r = await c2.get("/admin/api/update", cookies={"xz_admin": cookie})
        assert r.status == 200
    assert not handover.exists()
