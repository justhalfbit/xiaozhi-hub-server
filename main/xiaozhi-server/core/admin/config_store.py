"""读写 data/.config.yaml。

写入使用 ruamel.yaml 往返模式，保留用户文件里的注释和格式；
读取合并后的配置时绕过 load_config 的缓存，保证拿到磁盘上的最新值。

模块配置（LLM / ASR / TTS / Intent）用一套通用规则编辑：
- 只能修改该条目在合并配置里已有的字段（少数允许新增的字段见 ADDABLE_FIELDS）；
- 字段类型沿用原值：文本、数字、开关、字符串列表、JSON 对象；
- 名字像密钥的字段只写不读，提交空值表示不修改。
"""

import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from config.config_loader import get_project_dir, merge_configs, read_config

SECTIONS = ("LLM", "ASR", "TTS", "Intent")
SECRET_FIELD = re.compile(
    r"(^token$|_token$|api_?key|access_?token|access_?key|secret|password|passwd|private_?key|app_?key"
    r"|authorization|bearer|cookie|credential)",
    re.I,
)
# 条目里原本没有、但允许在后台新增的字段：{条目 type: {字段: 类型}}
ADDABLE_FIELDS = {
    "openai": {"extra_body": "json"},
    # 后来新增的配置项：老用户的配置文件里没有，也要能在后台改
    "claude_cli": {"progress_interval_seconds": "number", "progress_reply": "text",
                   "background_max_seconds": "number", "background_after_seconds": "number"},
}


def default_config_path() -> Path:
    return Path(get_project_dir()) / "config.yaml"


def custom_config_path() -> Path:
    return Path(get_project_dir()) / "data" / ".config.yaml"


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096
    return y


def load_custom(path: Path = None) -> CommentedMap:
    path = path or custom_config_path()
    if not path.exists():
        return CommentedMap()
    with open(path, "r", encoding="utf-8") as f:
        data = _yaml().load(f)
    return data if isinstance(data, CommentedMap) else CommentedMap()


# 后台的每个保存请求在线程池里各自执行；读改写整个过程要串行，否则两个请求同时保存时后写的会覆盖先写的
_WRITE_LOCK = threading.RLock()


@contextmanager
def edit_custom(path: Path = None):
    """读取 data/.config.yaml 交给调用方修改，正常结束时写回；中途出错不写入。"""
    with _WRITE_LOCK:
        data = load_custom(path)
        yield data
        save_custom(data, path)


def save_custom(data: CommentedMap, path: Path = None) -> None:
    """原子写入：先写临时文件再替换，避免写一半时服务读到损坏的配置。"""
    path = path or custom_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            _yaml().dump(data, f)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_merged(default_path: Path = None, custom_path: Path = None) -> dict:
    default = read_config(str(default_path or default_config_path())) or {}
    custom = read_config(str(custom_path or custom_config_path())) or {}
    return merge_configs(default, custom)


def is_placeholder(value) -> bool:
    """原项目用含"你"的占位文本表示未填写，例如"你的api_key"。"""
    return value is None or str(value).strip() == "" or "你" in str(value)


def mask_key(value) -> str:
    if is_placeholder(value):
        return ""
    value = str(value)
    if len(value) <= 12:
        return "已设置"
    return f"{value[:4]}…{value[-4:]}"


# ---------- 字段描述与类型转换 ----------

def field_kind(name: str, value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str) or value is None:
        return "secret" if SECRET_FIELD.search(name) else "text"
    if isinstance(value, list) and all(isinstance(v, (str, int, float)) for v in value):
        return "list"
    if isinstance(value, dict):
        return "json"
    return None


def _editable_fields(entry: dict) -> dict:
    """{字段: 类型}，包括允许新增的字段。"""
    fields = {}
    for name, value in entry.items():
        if name == "type":
            continue
        kind = field_kind(name, value)
        if kind:
            fields[name] = kind
    for name, kind in ADDABLE_FIELDS.get(entry.get("type"), {}).items():
        fields.setdefault(name, kind)
    return fields


def _convert(kind, raw, original):
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if kind == "number":
        text = str(raw).strip()
        if text == "":
            raise ValueError("数字不能为空")
        return int(text) if isinstance(original, int) and not isinstance(original, bool) and re.fullmatch(r"-?\d+", text) else float(text)
    if kind == "list":
        items = raw if isinstance(raw, list) else re.split(r"[,，\n]", str(raw))
        return CommentedSeq([str(i).strip() for i in items if str(i).strip()])
    if kind == "json":
        value = raw if isinstance(raw, dict) else json.loads(str(raw).strip() or "{}")
        if not isinstance(value, dict):
            raise ValueError("需要 JSON 对象，例如 {\"key\": \"value\"}")
        return value
    return "" if raw is None else str(raw).strip()


def update_entry(section: str, name: str, payload: dict, merged: dict, path: Path = None) -> list:
    """把字段写入 data/.config.yaml 的 <section>.<name>，返回实际修改的字段名。

    字段校验失败时抛出 ValueError（带字段名），整个请求不写入。
    """
    if section not in SECTIONS:
        raise KeyError(section)
    entry = (merged.get(section) or {}).get(name)
    if not isinstance(entry, dict):
        raise KeyError(name)
    kinds = _editable_fields(entry)
    changes = {}
    for field, raw in (payload or {}).items():
        kind = kinds.get(field)
        if kind is None:
            continue
        if kind == "secret" and (raw is None or str(raw).strip() == ""):
            continue  # 密钥留空表示不修改
        if field not in entry and (raw is None or str(raw).strip() == ""):
            continue  # 允许新增的字段留空表示不设置（数字留空也不报错）
        try:
            value = _convert(kind, raw, entry.get(field))
        except (ValueError, TypeError) as e:
            raise ValueError(f"{field}: {e}")
        original = entry.get(field)
        if kind == "list":
            same = [str(v) for v in (original or [])] == list(value)
        elif kind == "text":
            same = value == ("" if is_placeholder(original) else str(original))
        else:
            same = value == original and field in entry
        if field not in entry and value in ({}, [], ""):
            same = True  # 允许新增的字段留空时不写入
        if not same:
            changes[field] = value
    if not changes:
        return []
    with edit_custom(path) as data:
        target = data.setdefault(section, CommentedMap()).setdefault(name, CommentedMap())
        for field, value in changes.items():
            target[field] = value
    return list(changes)


def select_entry(section: str, name: str, merged: dict, path: Path = None) -> None:
    if section not in SECTIONS:
        raise KeyError(section)
    if name not in (merged.get(section) or {}):
        raise KeyError(name)
    with edit_custom(path) as data:
        data.setdefault("selected_module", CommentedMap())[section] = name


def entry_type(merged: dict, section: str, name: str = None) -> str:
    name = name or (merged.get("selected_module") or {}).get(section)
    entry = (merged.get(section) or {}).get(name) or {}
    return entry.get("type", name)


# ---------- 管理密码 ----------

def get_password_hash(path: Path = None):
    admin = load_custom(path).get("admin") or {}
    return admin.get("password_hash")


def set_password_hash(password_hash: str, path: Path = None) -> None:
    with edit_custom(path) as data:
        data.setdefault("admin", CommentedMap())["password_hash"] = password_hash
