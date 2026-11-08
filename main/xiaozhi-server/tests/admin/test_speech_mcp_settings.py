"""Speech/MCP configuration must preserve secrets and reject invalid edits atomically."""
import json
import stat

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from core.admin import config_store, mcp_settings, pages, speech_settings
from core.admin.admin_handler import AdminHandler


@pytest.mark.parametrize('section', ['ASR', 'TTS'])
def test_speech_keys_roundtrip_without_readback(tmp_path, section):
    path = tmp_path / '.config.yaml'
    path.write_text('# keep this comment\n{}\n')
    merged = lambda: config_store.load_merged(custom_path=path)
    speech_settings.save(section, {'base_url': 'https://speech.example/v1/', 'model': 'my-model',
                                  'api_key': 'speech-secret-abcdefghijk'}, merged(), path)
    spec = speech_settings.SPEECH[section]
    entry = merged()[section][spec['entry']]
    assert entry[spec['url_field']] == 'https://speech.example/v1' + spec['endpoint']
    assert entry[spec['model_field']] == 'my-model'
    visible = speech_settings.describe(section, merged())
    assert visible['configured'] and visible['base_url'] == 'https://speech.example/v1'
    assert 'speech-secret-abcdefghijk' not in json.dumps(visible)
    speech_settings.save(section, {'api_key': '', 'model': 'next-model'}, merged(), path)
    assert merged()[section][spec['entry']]['api_key'] == 'speech-secret-abcdefghijk'
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert '# keep this comment' in path.read_text()
    config_store.select_entry(section, spec['entry'], merged(), path)
    if section == 'ASR':
        assert pages.describe_asr(merged())['choice'] == 'openai'
    else:
        assert pages.describe_tts(merged())['type'] == 'openai'


@pytest.mark.parametrize('payload', [
    {'base_url': 'file:///tmp/private'}, {'base_url': 'https://user:secret@example.com'},
    {'base_url': 'https://example.com?token=secret'}, {'model': ''},
    {'api_key': 'with whitespace'}, {'speed': 'nan'}, {'speed': '4.1'}, {'voice': ''},
])
def test_invalid_speech_edit_is_atomic(tmp_path, payload):
    path = tmp_path / '.config.yaml'
    path.write_text('{}\n')
    original = path.read_bytes()
    with pytest.raises(ValueError):
        speech_settings.save('TTS', {'api_key': 'test-key', **payload},
                             config_store.load_merged(custom_path=path), path)
    assert path.read_bytes() == original


def test_asr_language_and_prompt(tmp_path):
    path = tmp_path / '.config.yaml'
    path.write_text('{}\n')
    merged = config_store.load_merged(custom_path=path)
    speech_settings.save('ASR', {'api_key': 'test-key', 'language': 'zh',
                               'prompt': '請以繁體中文轉寫。'}, merged, path)
    entry = config_store.load_merged(custom_path=path)['ASR']['OpenAIASR']
    assert entry['language'] == 'zh' and entry['prompt'] == '請以繁體中文轉寫。'
    with pytest.raises(ValueError):
        speech_settings.save('ASR', {'api_key': 'test-key', 'language': 'zh-TW'}, merged, path)


def test_mcp_secrets_kept_on_edit_and_other_servers_preserved(tmp_path):
    path = tmp_path / '.mcp_server_settings.json'
    mcp_settings.save(path, 'remote', {'url': 'https://mcp.example/mcp', 'transport': 'streamable-http',
                                     'headers': {'Authorization': 'Bearer private-token'}})
    mcp_settings.save(path, 'local', {'command': 'python', 'args': ['server.py'], 'env': {'API_KEY': 'private-env'}})
    visible = mcp_settings.describe(path)
    assert 'private-token' not in json.dumps(visible) and 'private-env' not in json.dumps(visible)
    config = visible['servers']['remote']
    config['timeout'] = 20
    mcp_settings.save(path, 'remote', config)
    saved = mcp_settings.load(path)['mcpServers']
    assert saved['remote']['headers']['Authorization'] == 'Bearer private-token'
    assert saved['local']['env']['API_KEY'] == 'private-env'
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    mcp_settings.save(path, 'remote', delete=True)
    assert set(mcp_settings.load(path)['mcpServers']) == {'local'}


def test_remote_mcp_drops_empty_command_so_sdk_uses_http(tmp_path):
    path = tmp_path / 'mcp.json'
    mcp_settings.save(path, 'remote', {'command': '', 'url': 'http://localhost:8000/mcp', 'transport': 'http'})
    assert 'command' not in mcp_settings.load(path)['mcpServers']['remote']


@pytest.mark.parametrize('config', [
    {'url': 'file:///tmp/a'}, {'url': 'https://a/mcp?key=secret'},
    {'url': 'https://a/mcp', 'command': 'sh'}, {'command': 'sh', 'args': 'run'},
    {'url': 'https://a/mcp', 'transport': 'invalid'},
    {'url': 'https://a/mcp', 'headers': {'Authorization': 123}},
    {'url': 'https://a/mcp', 'timeout': float('inf')},
])
def test_invalid_mcp_does_not_create_file(tmp_path, config):
    path = tmp_path / 'mcp.json'
    with pytest.raises(ValueError):
        mcp_settings.save(path, 'bad', config)
    assert not path.exists()


@pytest.mark.asyncio
async def test_new_admin_routes_require_login(tmp_path):
    h = AdminHandler({}, config_path=tmp_path / '.config.yaml')
    app = web.Application(); h.register(app)
    async with TestClient(TestServer(app)) as c:
        for path in ('mcp', 'speech/asr', 'speech/tts', 'tts/use', 'mcp/delete', 'language'):
            r = await c.post('/admin/api/' + path, json={})
            assert r.status == 401
        assert (await c.get('/admin/api/mcp')).status == 401
        assert (await c.get('/admin/i18n/runtime.js')).status == 200
        assert (await c.get('/admin/i18n/unknown.js')).status == 404


def test_reply_language_preserves_original_prompt():
    from core.agent.reply_language import apply
    prompt = 'Original prompt with code and 簡體／繁體 references.'
    assert apply(prompt, '') == prompt
    assert apply(prompt, 'zh-Hant').startswith(prompt)
    assert 'Traditional Chinese' in apply(prompt, 'zh-Hant')
    assert 'Simplified Chinese' in apply(prompt, 'zh-Hans')


@pytest.mark.asyncio
async def test_authenticated_speech_switch_and_mcp_roundtrip(tmp_path):
    path = tmp_path / '.config.yaml'; path.write_text('{}\n')
    h = AdminHandler(config_store.load_merged(custom_path=path), config_path=path)
    app = web.Application(); h.register(app)
    async with TestClient(TestServer(app)) as c:
        assert (await c.post('/admin/api/setup', json={'password': 'test-password'})).status == 200
        for section in ('asr', 'tts'):
            r = await c.post('/admin/api/speech/' + section,
                             json={'api_key': section + '-test-key', 'base_url': 'https://speech.example/v1', 'model': 'test-model'})
            assert r.status == 200
            r = await c.post('/admin/api/' + section + '/use', json={'choice': 'openai'})
            assert r.status == 200 and (await r.json())['ok']
            visible = await (await c.get('/admin/api/' + section)).json()
            assert section + '-test-key' not in json.dumps(visible)
        r = await c.post('/admin/api/mcp', json={'name': 'remote', 'config': {'url': 'https://mcp.example/mcp',
                            'transport': 'streamable-http', 'headers': {'Authorization': 'Bearer test-secret'}}})
        assert r.status == 200
        assert 'test-secret' not in await r.text()
        assert (await c.post('/admin/api/language', json={'language': 'zh-Hant'})).status == 200
        assert config_store.load_merged(custom_path=path)['reply_language'] == 'zh-Hant'
