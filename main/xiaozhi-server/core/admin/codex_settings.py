"""Codex-specific form values and validation."""
import json
import math
import os
import re
import subprocess
from pathlib import Path

from core.admin import config_store
from core.agent.codex_session import installation, permission_level

NUMBERS = ("turn_timeout_seconds", "stall_timeout_seconds", "idle_timeout_seconds", "resume_window_minutes",
           "background_max_seconds", "background_after_seconds", "filler_delay_seconds", "progress_interval_seconds")
FIELDS = ("level", "effort", "web", "workdir", "model", "first_reply_filler", "progress_reply", "resume_keywords",
          "cli_path", *NUMBERS)
EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")


def status(entry, *, check_login=True):
    return installation(entry.get("workdir") or "~/xiaozhi-workspace", cli_path=entry.get("cli_path") or "",
                        level=permission_level(entry), check_login=check_login)


def model_catalog(cli):
    if not cli:
        return []
    try:
        result = subprocess.run([cli, "debug", "models", "--bundled"], timeout=5, capture_output=True, check=True)
        return [{"id": m["slug"], "label": m.get("display_name") or m["slug"],
                 "efforts": [r["effort"] for r in m["supported_reasoning_levels"] if r["effort"] in EFFORTS]}
                for m in json.loads(result.stdout)["models"] if m.get("visibility") == "list"]
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError):
        return []


def describe(entry):
    info = status(entry)
    values = {key: entry.get(key) for key in FIELDS}
    values["level"] = permission_level(entry)
    return {**values, **info, "models": model_catalog(info["path"]), "efforts": EFFORTS}


def save(body, merged, path):
    entry = merged["LLM"]["CodexCLI"]
    values = {key: body[key] for key in FIELDS if key in body}
    values.setdefault("level", permission_level(entry))
    if values["level"] not in ("read", "edit", "auto"):
        raise ValueError("权限级别无效")
    if "effort" in values and values["effort"] not in ("", *EFFORTS):
        raise ValueError("思考级别无效")
    if "web" in values and not isinstance(values["web"], bool):
        raise ValueError("联网搜索需要布尔值")
    for key in NUMBERS:
        if key in values:
            try:
                number = float(values[key])
                if isinstance(values[key], bool) or not math.isfinite(number) or number < 0:
                    raise ValueError
            except (ValueError, TypeError):
                raise ValueError(f"{key} 需要非负数") from None
    for key in ("workdir", "cli_path"):
        if key in values:
            if key == "cli_path" and values[key] == "":
                continue
            try:
                target = Path(values[key]).expanduser()
                valid = target.is_dir() if key == "workdir" else target.is_file()
                if not target.is_absolute() or not valid or (key == "cli_path" and not os.access(target, os.X_OK)):
                    raise ValueError
            except (RuntimeError, TypeError, ValueError, OSError):
                raise ValueError(f"{key} 需要有效的绝对路径") from None
            values[key] = str(target)
    if "model" in values:
        model = values["model"]
        if not isinstance(model, str) or (model and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]*", model)):
            raise ValueError("模型名格式无效")
    if any(key in values for key in ("model", "effort")):
        settings = {**entry, **values}
        model, effort = settings.get("model"), settings.get("effort")
        if model and effort:
            for item in model_catalog(status(settings, check_login=False)["path"]):
                if item["id"] == model and effort not in item["efforts"]:
                    raise ValueError(f"{model} 不支持该思考级别")
    for key in ("first_reply_filler", "progress_reply"):
        if key in values and not isinstance(values[key], str):
            raise ValueError(f"{key} 需要文本")
    if "resume_keywords" in values and not (isinstance(values["resume_keywords"], list) and
                                            all(isinstance(k, str) for k in values["resume_keywords"])):
        raise ValueError("接续关键字需要文本列表")
    values["enabled"] = True
    return config_store.update_entry("LLM", "CodexCLI", values, merged, path, literal_text=True)
