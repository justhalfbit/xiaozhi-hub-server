"""插件设置：后台「意图识别」页插件旁边的「设置」。

字段清单在 catalog.PLUGIN_SETTINGS（按 id），PLUGINS 条目用 "settings" 指向它，值存在 plugins.<config> 下。
这里只读写配置文件；保存后同步到运行中的服务和在线连接由 AdminHandler 负责。
要加「测试」按钮：在 TESTS 里登记 async 函数 (values) -> dict，失败时抛 ValueError（message 给用户看）。
"""

from pathlib import Path

from ruamel.yaml.comments import CommentedMap

from core.admin import catalog, config_store


def _spec(settings_id: str) -> dict:
    spec = catalog.PLUGIN_SETTINGS.get(settings_id)
    if not spec:
        raise KeyError(settings_id)
    return spec


def config_key(settings_id: str) -> str:
    """设置存在 plugins 下的哪个键（几个插件可以共用一份，例如智能家居的三个）。"""
    _spec(settings_id)
    return next(p["config"] for p in catalog.PLUGINS if p.get("settings") == settings_id)


def _saved(merged: dict, settings_id: str) -> dict:
    cfg = (merged.get("plugins") or {}).get(config_key(settings_id))
    return cfg if isinstance(cfg, dict) else {}


def _filled(value) -> bool:
    """上游的占位值（"你的…"、"mk-xxx"）算没填。"""
    return not config_store.is_placeholder(value) and not str(value).strip().lower().endswith("-xxx")


def configured(merged: dict, settings_id: str) -> bool:
    saved = _saved(merged, settings_id)
    return all(_filled(saved.get(f["key"])) for f in _spec(settings_id)["fields"] if f.get("required"))


def describe(merged: dict, settings_id: str) -> dict:
    spec, saved = _spec(settings_id), _saved(merged, settings_id)
    fields = []
    for f in spec["fields"]:
        item = {k: v for k, v in f.items()}
        raw = saved.get(f["key"])
        if f["type"] == "secret":
            item["set"] = _filled(raw)
            item["hint_set"] = config_store.mask_key(raw) if item["set"] else ""
        else:
            item["value"] = "" if raw is None or not _filled(raw) else raw
        fields.append(item)
    return {"id": settings_id, "note": spec.get("note", ""), "fields": fields,
            "configured": configured(merged, settings_id), "testable": settings_id in TESTS}


def _effective(f: dict, value):
    """字段实际生效的值：没填（或占位值）用默认值。"""
    if value is None or str(value).strip() == "" or not _filled(value):
        return f.get("default", "")
    return value


def _values(settings_id: str, payload: dict, merged: dict) -> dict:
    """校验页面提交的值，返回完整的设置：secret 留空用已保存的，其他留空表示用默认值（存成空），
    页面没提交的字段用已保存的。"""
    payload = payload or {}
    saved = _saved(merged, settings_id)
    values = {}
    for f in _spec(settings_id)["fields"]:
        key, label = f["key"], f["label"]
        if key not in payload:
            values[key] = saved.get(key) if _filled(saved.get(key)) else ""
            continue
        raw = payload.get(key)
        text = "" if raw is None else str(raw).strip()
        if f["type"] == "secret":
            if any(c.isspace() for c in text):
                raise ValueError(f"{label} 里不能有空格或换行")
            values[key] = text or (str(saved.get(key)).strip() if _filled(saved.get(key)) else "")
        elif f["type"] == "number":
            if text == "":
                values[key] = ""
                continue
            try:
                number = int(text)
            except ValueError:
                raise ValueError(f"{label} 要填整数")
            if ("min" in f and number < f["min"]) or ("max" in f and number > f["max"]):
                raise ValueError(f"{label} 要在 {f.get('min')} 到 {f.get('max')} 之间")
            values[key] = number
        else:
            if "\n" in text:
                raise ValueError(f"{label} 不能换行")
            if key.endswith("url") and text and not text.startswith(("http://", "https://")):
                raise ValueError(f"{label} 要以 http:// 或 https:// 开头")
            values[key] = text
    return values


def save(settings_id: str, payload: dict, merged: dict, path: Path = None) -> list:
    """保存设置，返回修改的字段名。"""
    values = _values(settings_id, payload, merged)
    saved = _saved(merged, settings_id)
    fields = {f["key"]: f for f in _spec(settings_id)["fields"]}
    changes = {k: v for k, v in values.items()
               if str(_effective(fields[k], v)).strip() != str(_effective(fields[k], saved.get(k))).strip()}
    if not changes:
        return []
    with config_store.edit_custom(path) as data:
        target = data.setdefault("plugins", CommentedMap()).setdefault(config_key(settings_id), CommentedMap())
        for key, value in changes.items():
            target[key] = value
    return list(changes)


def test_values(settings_id: str, payload: dict, merged: dict) -> dict:
    """测试用的设置：页面上还没保存的值优先。"""
    return _values(settings_id, payload, merged)


# ---------- 测试函数 ----------
# settings_id -> async (values) -> dict；values 是 test_values 整理好的完整设置，失败时抛 ValueError

async def _test_web_search(values: dict) -> dict:
    from plugins_func.functions import web_search

    settings = web_search.settings_of({"plugins": {"web_search": values}})
    try:
        result = await web_search.search(settings, "今天的重要新闻")
    except web_search.SearchError as e:
        raise ValueError(str(e))
    if not result.results:
        raise ValueError("接口执行了搜索，但没有返回结果")
    return {"seconds": result.seconds, "count": len(result.results), "first": result.results[0]["title"]}


TESTS = {"web_search": _test_web_search}


def _redact(message: str, values: dict, settings_id: str) -> str:
    """报错信息里不带出密钥（有的接口会把收到的 Key 原样写进错误信息）。"""
    for f in _spec(settings_id)["fields"]:
        secret = str(values.get(f["key"]) or "") if f["type"] == "secret" else ""
        if len(secret) >= 4:
            message = message.replace(secret, "***")
    return message


async def run_test(settings_id: str, payload: dict, merged: dict) -> dict:
    if settings_id not in TESTS:
        raise KeyError(settings_id)
    values = test_values(settings_id, payload, merged)
    try:
        result = await TESTS[settings_id](values)
    except ValueError as e:
        raise ValueError(_redact(str(e), values, settings_id))
    return {k: _redact(v, values, settings_id) if isinstance(v, str) else v for k, v in result.items()}
