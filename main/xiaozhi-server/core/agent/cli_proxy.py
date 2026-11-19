"""Shared proxy settings for locally launched LLM command-line clients."""
import os
import socket
from urllib.parse import urlsplit


DEFAULT_URL = "http://127.0.0.1:7890"


def settings(config):
    value = config.get("cli_proxy") or {}
    return {"enabled": value.get("enabled") is True, "url": value.get("url") or DEFAULT_URL}


def save(body, path):
    from core.admin import config_store

    url = body.get("url")
    if not isinstance(body.get("enabled"), bool) or not isinstance(url, str):
        raise ValueError("请填写代理地址和开关状态")
    url = url.strip()
    if "://" not in url:
        url = "http://" + url
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme in ("http", "https") and parsed.hostname and parsed.port
                 and parsed.path in ("", "/") and not parsed.query and not parsed.fragment
                 and parsed.username is None and not any(c.isspace() for c in url))
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("代理地址需要 HTTP(S) 主机和端口，例如 127.0.0.1:7890")
    value = {"enabled": body["enabled"], "url": url.rstrip("/")}
    with config_store.edit_custom(path) as config:
        config["cli_proxy"] = value
    return value, connection_warning(value)


def connection_warning(value):
    if value["enabled"]:
        parsed = urlsplit(value["url"])
        try:
            with socket.create_connection((parsed.hostname, parsed.port), timeout=2):
                pass
        except OSError:
            return f"无法连接 {value['url']}，请检查 Clash Party 的运行状态和端口"
    return ""


def overrides():
    from core.admin import config_store

    value = settings(config_store.load_custom())
    if not value["enabled"]:
        return {}
    bypass = ",".join(filter(None, (os.getenv("NO_PROXY"), os.getenv("no_proxy"), "localhost,127.0.0.1,::1")))
    env = {}
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        env[key] = env[key.lower()] = bypass if key == "NO_PROXY" else value["url"]
    return env


def child_env():
    return {**os.environ, **overrides()}
