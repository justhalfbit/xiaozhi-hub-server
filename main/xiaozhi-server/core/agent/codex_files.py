"""Workspace-only stdio tools for Codex read/edit sessions on Linux and macOS."""
import os
import secrets
import stat
import sys
from contextlib import contextmanager
from pathlib import PurePosixPath

from mcp.server.fastmcp import FastMCP

LIMIT = 512 * 1024
PROTECTED = {".git", ".agents", ".codex"}
DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW if os.name == "posix" else None


class Workspace:
    def __init__(self, root):
        if DIRECTORY is None:
            raise ValueError("受限文件工具需要 POSIX 目录权限支持")
        self.root = os.open(root, DIRECTORY)

    @contextmanager
    def parent(self, path):
        parts = PurePosixPath(path).parts
        if not parts or path.startswith("/") or any(p.lower() in PROTECTED or p == ".." for p in parts):
            raise ValueError("需要工作目录内的相对路径")
        fd = os.dup(self.root)
        try:
            for part in parts[:-1]:
                child = os.open(part, DIRECTORY, dir_fd=fd)
                os.close(fd)
                fd = child
            yield fd, parts[-1]
        finally:
            os.close(fd)

    def list_files(self, directory: str = ".") -> list[str]:
        """List one directory. Directory names end with a slash."""
        with self.parent(str(PurePosixPath(directory) / "_")) as (fd, _):
            with os.scandir(fd) as entries:
                names = []
                for item in entries:
                    if item.name.lower() not in PROTECTED and not item.is_symlink():
                        names.append(item.name + ("/" if item.is_dir(follow_symlinks=False) else ""))
                    if len(names) == 2000:
                        break
                return sorted(names)

    def read_file(self, path: str) -> str:
        """Read a UTF-8 text file relative to the workspace."""
        with self.parent(path) as (parent, name):
            fd = os.open(name, os.O_NOFOLLOW | os.O_RDONLY | os.O_NONBLOCK, dir_fd=parent)
            with os.fdopen(fd, "rb") as file:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > LIMIT:
                    raise ValueError("只支持普通文本文件（最多 512 KiB）")
                data = file.read(LIMIT + 1)
                if len(data) > LIMIT:
                    raise ValueError("文件过大")
                return data.decode()

    def search_files(self, text: str, directory: str = ".") -> list[str]:
        """Find literal text in workspace files. Returns at most 100 matching lines."""
        pending = [directory]
        matches = []
        for _ in range(2000):
            if not pending or len(matches) >= 100:
                break
            path = pending.pop()
            try:
                if path == directory or path.endswith("/"):
                    pending.extend(str(PurePosixPath(path) / n) + ("/" if n.endswith("/") else "")
                                   for n in self.list_files(path))
                else:
                    for index, line in enumerate(self.read_file(path).splitlines(), 1):
                        if text in line:
                            matches.append(f"{path}:{index}: {line[:240]}")
                            if len(matches) >= 100:
                                break
            except (OSError, ValueError):
                continue
        return matches

    def write_file(self, path: str, content: str) -> str:
        """Create or replace a UTF-8 text file in an existing workspace directory."""
        data = content.encode()
        if len(data) > LIMIT:
            raise ValueError("文件过大")
        with self.parent(path) as (parent, name):
            mode = 0o600
            try:
                info = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("只允许替换普通文件")
                mode = info.st_mode & 0o777
            except FileNotFoundError:
                pass
            tmp = ".codex-edit-" + secrets.token_hex(12)
            fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=parent)
            try:
                with os.fdopen(fd, "wb") as file:
                    file.write(data)
                    os.fchmod(file.fileno(), mode)
                os.replace(tmp, name, src_dir_fd=parent, dst_dir_fd=parent)
            finally:
                try:
                    os.unlink(tmp, dir_fd=parent)
                except FileNotFoundError:
                    pass
        return path


if __name__ == "__main__":
    root, level = sys.argv[1:]
    files = Workspace(root)
    server = FastMCP("workspace")
    for tool in (files.list_files, files.read_file, files.search_files):
        server.tool()(tool)
    if level == "edit":
        server.tool()(files.write_file)
    server.run()
