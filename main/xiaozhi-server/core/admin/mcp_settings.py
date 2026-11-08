"""Atomic MCP configuration editing; never return environment or header secrets."""
import json
import math
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit

from core.admin.config_store import _WRITE_LOCK


def load(path):
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {'mcpServers': {}}
    if not isinstance(data, dict) or not isinstance(data.get('mcpServers', {}), dict):
        raise ValueError('MCP 設定必須包含 mcpServers 物件')
    if not all(isinstance(config, dict) for config in data.get('mcpServers', {}).values()):
        raise ValueError('每個 MCP 伺服器設定都必須是 JSON 物件')
    return data


def describe(path):
    servers = {}
    for name, config in load(path).get('mcpServers', {}).items():
        visible = {k: v for k, v in config.items() if k not in ('env', 'headers', 'API_ACCESS_TOKEN')}
        for field in ('env', 'headers'):
            if field in config:
                visible[field] = {k: '' for k in config[field]}
        if 'API_ACCESS_TOKEN' in config:
            visible['API_ACCESS_TOKEN'] = ''
        servers[name] = visible
    return {'servers': servers, 'note': '儲存後，裝置重新連線時生效。API 模型需啟用工具呼叫；Claude Code 使用自己的 MCP 設定。'}


def validate(config, old):
    if not isinstance(config, dict):
        raise ValueError('伺服器設定必須是 JSON 物件')
    allowed = {'command', 'args', 'env', 'url', 'transport', 'headers', 'API_ACCESS_TOKEN',
               'timeout', 'sse_read_timeout', 'terminate_on_close', 'des', 'link'}
    if set(config) - allowed:
        raise ValueError('包含不支援的 MCP 設定欄位')
    result = dict(config)
    if bool(result.get('command')) == bool(result.get('url')):
        raise ValueError('請設定 command 或 url，兩者擇一')
    if result.get('command'):
        if not isinstance(result['command'], str) or not result['command'].strip():
            raise ValueError('command 必須是非空白字串')
        if not isinstance(result.get('args', []), list) or not all(isinstance(a, str) for a in result.get('args', [])):
            raise ValueError('args 必須是字串陣列')
    else:
        result.pop('command', None)  # SDK selects stdio by key presence, including an empty string.
        if not isinstance(result['url'], str):
            raise ValueError('url 必須是字串')
        url = urlsplit(result['url'])
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('請使用 HTTP(S) 網址；權杖請放在 headers，不要放在網址')
        if result.get('transport', 'sse') not in ('sse', 'http', 'streamable-http'):
            raise ValueError('傳輸類型必須是 sse 或 streamable-http')
    for field in ('env', 'headers'):
        if field not in result:
            continue
        values = result[field]
        if not isinstance(values, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in values.items()):
            raise ValueError(f'{field} 必須是字串對應表')
        result[field] = {k: (v if v else (old.get(field) or {}).get(k, '')) for k, v in values.items()}
    if result.get('API_ACCESS_TOKEN') == '':
        result['API_ACCESS_TOKEN'] = old.get('API_ACCESS_TOKEN', '')
    for field in ('timeout', 'sse_read_timeout'):
        if field in result:
            value = result[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 3600:
                raise ValueError(f'{field} 必須介於 0 和 3600 秒之間')
    if 'terminate_on_close' in result and not isinstance(result['terminate_on_close'], bool):
        raise ValueError('terminate_on_close 必須是布林值')
    return result


def save(path, name, config=None, delete=False):
    if not isinstance(name, str) or not re.fullmatch(r'[\w .-]{1,80}', name) or not name.strip():
        raise ValueError('請填寫伺服器名稱（最多 80 字元）')
    path = Path(path)
    with _WRITE_LOCK:
        data = load(path)
        servers = data.setdefault('mcpServers', {})
        if delete:
            servers.pop(name, None)
        else:
            servers[name] = validate(config, servers.get(name) or {})
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.mcp.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
