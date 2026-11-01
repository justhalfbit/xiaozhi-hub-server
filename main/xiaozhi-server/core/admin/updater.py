"""管理后台的「更新网关」：查看当前版本、和远端比较有没有新版本、在后台运行 scripts/update.sh。

只在用户点了才检查、才更新，不做自动更新。更新本身全部交给 update.sh（和命令行更新同一套逻辑）：
快进代码、依赖有变化才补装、重启后台服务；补装失败或新版本起不来时退回原来的版本。

update.sh 会重启网关，不能作为网关的普通子进程跑（重启时会被一起杀掉）：Linux 上网关由 systemd 管着时，
用 systemd-run --user --scope 放进单独的 scope；其他情况新开一个会话（setsid），launchd 只杀同一进程组。
进度通过文件传递，网关重启后新进程照样读得到（都在 tmp/ 下）：
- update-state.json：开始时间、更新前的提交、进程号
- update.log：update.sh 的输出
- update.exit：update.sh 结束时写入的退出码
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = SERVER_DIR.parents[1]
CHECK_TIMEOUT = 30          # 检查更新时连远端最多等多久（秒）
RUNNING_LIMIT = 3600        # 超过这么久还没结束的更新按中断处理（进程号可能已被别的进程复用）
MAX_COMMITS = 30            # 更新说明最多列多少条
READY_TIMEOUT = 300         # 重启后等新版本起来的时间，超时就退回原版本（设备上启动要加载语音识别模型）
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# 关掉凭据询问（没有终端会一直卡住）；限速：连上了但 15 秒内几乎没有数据就放弃
_GIT_ENV = {"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "", "SSH_ASKPASS": "", "LC_ALL": "C"}
_GIT_NET = ["-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=15"]


class UpdateError(Exception):
    """不能开始更新（本地有改动、正在更新等），消息直接给用户看。"""


class Updater:
    def __init__(self, root: Path = REPO_ROOT, state_dir: Path = None):
        self.root = Path(root)
        self.state_dir = Path(state_dir) if state_dir else SERVER_DIR / "tmp"
        self.state_file = self.state_dir / "update-state.json"
        self.log_file = self.state_dir / "update.log"
        self.exit_file = self.state_dir / "update.exit"
        self.last_check = None
        self._lock = threading.Lock()  # 检查和开始更新不能同时动 git
        # 网关启动时的版本：磁盘上的代码更新了、服务还没重启时，运行中的仍是它
        self.running_commit = self._head()

    # ---------- git ----------

    def _git(self, *args, timeout=15) -> subprocess.CompletedProcess:
        """运行 git；超时时连同它启动的 git-remote-https 一起结束。"""
        proc = subprocess.Popen(["git", "-C", str(self.root), *args], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ, **_GIT_ENV},
                                start_new_session=os.name != "nt")
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL) if os.name != "nt" else proc.kill()
            except OSError:
                pass
            proc.communicate()
            raise
        return subprocess.CompletedProcess(proc.args, proc.returncode, out.decode("utf-8", "replace"),
                                           err.decode("utf-8", "replace"))

    def _out(self, *args, timeout=15) -> str:
        """成功时返回输出，失败返回空字符串。"""
        try:
            r = self._git(*args, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return r.stdout.strip() if r.returncode == 0 else ""

    def _head(self) -> str:
        return self._out("rev-parse", "HEAD")

    def _commit(self, rev: str) -> dict:
        out = self._out("log", "-1", "--format=%H%x1f%h%x1f%cI%x1f%s", rev)
        if not out:
            return {}
        full, short, date, subject = (out.split("\x1f") + ["", "", "", ""])[:4]
        return {"commit": full, "short": short, "date": date, "subject": subject}

    def _commits(self, rev_range: str) -> list:
        out = self._out("log", f"-{MAX_COMMITS}", "--format=%h%x1f%cI%x1f%s", rev_range)
        return [dict(zip(("short", "date", "subject"), line.split("\x1f"))) for line in out.splitlines() if line]

    def _source(self, branch: str) -> dict:
        """从哪里更新，和 update.sh 一样：分支设了上游就用上游，否则 origin 上的同名分支。"""
        remote = self._out("config", f"branch.{branch}.remote") or "origin"
        merge = self._out("config", f"branch.{branch}.merge") or branch
        url = self._out("remote", "get-url", remote)
        url = re.sub(r"//[^/@]+@", "//", url)  # 地址里带的账号密码不显示
        return {"remote": remote, "branch": merge.removeprefix("refs/heads/"), "url": url}

    def _dirty_files(self) -> list:
        # 只看受 Git 管理的文件；配置、模型、虚拟环境都被 .gitignore 忽略，不算改动（同 update.sh）
        try:
            r = self._git("status", "--porcelain", "--untracked-files=no")
        except (OSError, subprocess.TimeoutExpired):
            return []
        return [line[3:] for line in r.stdout.splitlines() if line.strip()] if r.returncode == 0 else []

    # ---------- 版本与检查 ----------

    def version(self) -> dict:
        """当前版本、从哪里更新、能不能在后台更新（unsupported 非空时不能，说明原因）。"""
        info = {"current": {}, "running": {}, "branch": "", "source": {}, "unsupported": ""}
        if os.name == "nt":
            info["unsupported"] = "Windows 上请按教程手动更新：在项目目录运行 git pull，再运行一次 install.ps1"
        if not (self.root / ".git").exists():
            info["unsupported"] = "网关不是用 git 下载的，没法在后台更新。请重新运行一键安装命令，或手动下载新版本"
            return info
        info["current"] = self._commit("HEAD")
        if self.running_commit and self.running_commit != info["current"].get("commit"):
            info["running"] = self._commit(self.running_commit)
        info["branch"] = self._out("symbolic-ref", "--short", "-q", "HEAD")
        if not info["branch"]:
            info["unsupported"] = info["unsupported"] or "当前不在任何分支上，没法自动更新"
            return info
        info["source"] = self._source(info["branch"])
        return info

    def check(self, timeout: int = CHECK_TIMEOUT) -> dict:
        """从远端取最新版本和当前版本比较，不改动代码。

        state：latest 已是最新；available 有新版本；dirty 有新版本但本地改过文件；
        diverged 本地有远端没有的提交；error 连不上远端等；unsupported 不能在后台更新。
        """
        with self._lock:
            result = self._check(timeout)
        result["checked_at"] = int(time.time())
        self.last_check = result
        return result

    def _check(self, timeout: int) -> dict:
        info = self.version()
        base = {"current": info["current"], "source": info["source"], "latest": {}, "behind": 0, "commits": []}
        if info["unsupported"]:
            return {**base, "state": "unsupported", "message": info["unsupported"]}
        if self.running():
            return {**base, "state": "error", "message": "正在更新，请等更新结束再检查"}
        src = info["source"]
        shallow = self._out("rev-parse", "--is-shallow-repository") == "true"
        fetch = [*_GIT_NET, "fetch", "--no-tags", "--quiet"]
        # 一键安装是只含最新一次提交的浅克隆：取浅克隆边界之后的提交，接得上历史才列得出更新说明、看得出分叉
        attempts = [["--shallow-since=" + self._shallow_since(info["current"]["date"])], ["--depth=1"]] \
            if shallow else [[]]
        started = time.monotonic()
        for extra in attempts:
            left = timeout - (time.monotonic() - started)
            try:
                r = self._git(*fetch, *extra, src["remote"], src["branch"], timeout=max(left, 1))
            except subprocess.TimeoutExpired:
                return {**base, "state": "error", "message": self._timeout_message(timeout)}
            except OSError as e:
                return {**base, "state": "error", "message": f"运行 git 失败：{e}"}
            if r.returncode == 0:
                break
        else:
            return {**base, "state": "error", "message": self._fetch_error(r.stderr, src)}

        head, new = info["current"]["commit"], self._out("rev-parse", "FETCH_HEAD")
        base["latest"] = self._commit(new)
        if new == head:
            return {**base, "state": "latest", "message": "已经是最新版本"}
        if self._is_ancestor(head, new):
            base["behind"] = int(self._out("rev-list", "--count", f"{head}..{new}") or 0)
            base["commits"] = self._commits(f"{head}..{new}")
        elif shallow and not self._out("merge-base", head, new):
            # 浅克隆接不上历史，判断不了先后，同 update.sh 直接换到新版本；只列得出最新一次提交
            base["behind"], base["commits"] = None, [{k: base["latest"][k] for k in ("short", "date", "subject")}]
        elif self._is_ancestor(new, head):
            return {**base, "state": "diverged",
                    "message": "本地的版本比远端还新（有远端没有的提交），不用更新。要以远端为准请手动处理"}
        else:
            return {**base, "state": "diverged",
                    "message": f"本地有自己的提交，和远端分叉了，没法直接更新。请手动合并：git merge {src['remote']}/{src['branch']}"}
        files = self._dirty_files()
        if files:
            return {**base, "state": "dirty", "files": files,
                    "message": "有新版本，但下面这些文件被改过，更新会和它们冲突。请先撤销（git checkout -- <文件>）或备份后再更新"}
        return {**base, "state": "available", "message": "有新版本"}

    def _shallow_since(self, fallback: str) -> str:
        """浅克隆本地最早那个提交（边界）的时间（同 update.sh）。

        不能用当前版本的时间：本地有自己的提交时它比和远端共同的那个提交新，按它取回来的历史接不上，
        分叉会被当成「接不上历史」，判成有新版本，更新时直接切过去把本地提交丢掉。
        """
        roots = self._out("rev-list", "--max-parents=0", "HEAD").split()
        out = self._out("log", "--no-walk", "--format=%ct %cI", *roots) if roots else ""
        dates = sorted((int(t), d) for t, _, d in (line.partition(" ") for line in out.splitlines()) if t.isdigit())
        return dates[0][1] if dates else fallback

    def _is_ancestor(self, a: str, b: str) -> bool:
        try:
            return self._git("merge-base", "--is-ancestor", a, b).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _timeout_message(self, timeout: int) -> str:
        return (f"连接更新服务器超时（{timeout} 秒）。国内网络可能连不上 GitHub，可以稍后再试，"
                f"或者给网关的仓库设置代理：git -C {self.root} config http.proxy http://代理地址:端口")

    def _fetch_error(self, stderr: str, src: dict) -> str:
        text = stderr.strip()
        last = text.splitlines()[-1] if text else ""
        if "couldn't find remote ref" in text or "no such ref" in text.lower():
            return f"远端 {src['remote']}（{src['url']}）上没有 {src['branch']} 分支"
        if "Could not resolve host" in text or "Failed to connect" in text or "Connection timed out" in text \
                or "Operation timed out" in text or "Connection refused" in text or "too slow" in text:
            return (f"连不上更新服务器（{last}）。国内网络可能连不上 GitHub，可以稍后再试，"
                    f"或者给网关的仓库设置代理：git -C {self.root} config http.proxy http://代理地址:端口")
        if "Authentication failed" in text or "could not read Username" in text:
            return f"远端要求登录（{src['url']}），后台没法输入账号密码。请确认仓库地址是公开的"
        return f"检查更新失败：{last or '未知错误'}"

    # ---------- 更新 ----------

    def _state(self) -> dict:
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _alive(pid) -> bool:
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _exit_code(self):
        try:
            return int(self.exit_file.read_text().strip())
        except (OSError, ValueError):
            return None

    def running(self) -> bool:
        state = self._state()
        return bool(state) and self._exit_code() is None and self._alive(state.get("pid")) \
            and time.time() - state.get("started", 0) < RUNNING_LIMIT

    def start(self) -> dict:
        """在后台开始更新；不能开始时抛 UpdateError。马上返回，进度用 status() 查。"""
        with self._lock:
            if self.running():
                raise UpdateError("已经在更新了，请等它结束")
            info = self.version()
            if info["unsupported"]:
                raise UpdateError(info["unsupported"])
            files = self._dirty_files()
            if files:
                raise UpdateError("这些文件被改过，更新会和它们冲突：" + "、".join(files[:10]) +
                                  "。请先撤销（git checkout -- <文件>）或备份后再更新")
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.exit_file.unlink(missing_ok=True)
            with open(self.log_file, "wb") as log:
                proc = subprocess.Popen(self._command(), cwd=str(self.root), stdin=subprocess.DEVNULL, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        env={**os.environ, "HUB_READY_TIMEOUT": str(READY_TIMEOUT)})
            self._write_state({"started": time.time(), "from": info["current"].get("commit", ""), "pid": proc.pid})
            # 等它结束（网关被它重启时这个线程也就没了）；没写退回码就结束的（如 systemd-run 起不来）记成失败
            threading.Thread(target=self._reap, args=(proc,), daemon=True).start()
        return self.status()

    def _command(self) -> list:
        script = 'bash "$1"; code=$?; echo "$code" >"$2.part" && mv "$2.part" "$2"'
        cmd = ["bash", "-c", script, "hub-update", str(self.root / "scripts" / "update.sh"), str(self.exit_file)]
        if sys.platform.startswith("linux") and os.environ.get("INVOCATION_ID") and shutil.which("systemd-run"):
            # 网关是 systemd 服务：重启时会结束服务 cgroup 里的所有进程，更新进程要放到服务外面
            cmd = ["systemd-run", "--user", "--scope", "--quiet", "--collect",
                   f"--unit=xiaozhi-hub-update-{int(time.time())}", *cmd]
        return cmd

    def _write_state(self, state: dict) -> None:
        tmp = self.state_file.with_suffix(".part")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(self.state_file)

    def _reap(self, proc) -> None:
        code = proc.wait()
        if self._exit_code() is None:
            self.exit_file.write_text(str(code or 1))

    def status(self) -> dict:
        """最近一次更新的情况。state：idle 没更新过；running；done；failed；interrupted 没结束就中断了（如断电）。"""
        state = self._state()
        if not state:
            return {"state": "idle"}
        code = self._exit_code()
        if code is None:
            alive = self._alive(state.get("pid")) and time.time() - state.get("started", 0) < RUNNING_LIMIT
            # 读退出码和查进程之间它可能刚好结束：再读一次，免得把正常结束当成中断
            code = None if alive else self._exit_code()
            status = "running" if alive else "interrupted" if code is None else "done" if code == 0 else "failed"
        else:
            status = "done" if code == 0 else "failed"
        # 日志在定下状态之后读：结束了就一定读得到最后的报错
        try:
            lines = [_ANSI.sub("", line).rstrip() for line in
                     self.log_file.read_text(encoding="utf-8", errors="replace").splitlines()]
        except OSError:
            lines = []
        steps = [line[4:].strip() for line in lines if line.startswith("==> ")]
        errors = [line.lstrip("✗ ").strip() for line in lines if line.lstrip().startswith("✗")]
        head = self._head()
        result = {"state": status, "started": int(state.get("started", 0)), "step": steps[-1] if steps else "",
                  "from": self._commit(state["from"]) if state.get("from") else {}, "to": self._commit("HEAD"),
                  "changed": bool(head) and head != state.get("from"), "log": lines[-80:]}
        if code is not None:
            try:
                result["finished"] = int(self.exit_file.stat().st_mtime)
            except OSError:
                pass
        if status == "failed":
            result["error"] = errors[-1] if errors else (lines[-1] if lines else "更新失败，没有输出")
        elif status == "interrupted":
            result["error"] = "更新没有正常结束（中途断电或进程被结束）。可以再点一次更新，会接着把它做完"
        # 代码换了但服务没重启（手动启动的服务，update.sh 不会替它重启）
        result["restart_needed"] = status == "done" and bool(head) and head != self.running_commit
        return result
