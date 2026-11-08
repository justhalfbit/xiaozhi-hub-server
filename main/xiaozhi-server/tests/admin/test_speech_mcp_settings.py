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
        for path in ('mcp', 'speech/asr', 'speech/tts', 'speech/asr/test', 'speech/tts/test', 'tts/use', 'mcp/delete', 'language'):
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


def test_spoken_languages_have_reply_rules_localized_prompts_and_separate_voice_cache():
    from core.agent import reply_language
    config = {'selected_module': {'TTS': 'OpenAITTS'}, 'TTS': {'OpenAITTS': {'type': 'openai', 'voice': 'Leda'}}}
    revisions = set()
    for language in reply_language.LANGUAGES:
        profile = speech_settings.tts_profile({**config, 'reply_language': language})
        revisions.add(profile['revision'])
        assert profile['wake_text'] == reply_language.phrase('wake', language)
        assert reply_language.phrase('test', language)
        if language:
            assert reply_language.apply('Original prompt', language).startswith('Original prompt')
            assert len(reply_language.apply('', language)) > 20
    assert len(revisions) == len(reply_language.LANGUAGES)
    assert reply_language.phrase('wake', 'yue-HK') == '喺度'
    assert reply_language.phrase('wake', 'en') == "I'm here."


@pytest.mark.asyncio
async def test_users_choose_language_without_changing_provider_or_interface(tmp_path):
    import copy
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from core.admin import connections
    from core.admin.admin_handler import COOKIE_NAME
    path = tmp_path / '.config.yaml'; path.write_text('reply_language: zh-Hant\n')
    config = config_store.load_merged(custom_path=path)
    selected = copy.deepcopy(config['selected_module'])
    h = AdminHandler(config, ws_server=object(), config_path=path)
    conn = SimpleNamespace(session_id='language-selection-test', common_config=config,
                           config=copy.deepcopy(config), websocket=SimpleNamespace(close=AsyncMock()))
    connections.register(conn)
    cookie = {COOKIE_NAME: h.sessions.create()}
    app = web.Application(); h.register(app)
    try:
        async with TestClient(TestServer(app)) as c:
            before = path.read_bytes()
            d = await (await c.get('/admin/api/tts', cookies=cookie)).json()
            assert d['reply_language'] == 'zh-Hant'
            assert {'yue-HK', 'en', 'ja', 'ko', 'es', 'fr'} <= {o['value'] for o in d['language_options']}
            assert path.read_bytes() == before  # opening the menu never chooses for the user
            r = await c.post('/admin/api/language', json={'language': 'unsupported'}, cookies=cookie)
            assert r.status == 400 and path.read_bytes() == before
            r = await c.post('/admin/api/language', json={'language': 'yue-HK'}, cookies=cookie)
            assert r.status == 200
            assert config_store.load_custom(path)['reply_language'] == 'yue-HK'
            assert config['reply_language'] == conn.config['reply_language'] == 'yue-HK'
            conn.websocket.close.assert_awaited_once()
            memory = await (await c.get('/admin/api/memory', cookies=cookie)).json()
            assert memory['reply_language'] == 'yue-HK'
            assert memory['language_options'] == d['language_options']
            assert config['selected_module'] == selected
    finally:
        connections.unregister(conn)


def test_tts_profile_tracks_voice_model_and_speed_without_credentials():
    import copy
    config = {'selected_module': {'TTS': 'OpenAITTS'}, 'TTS': {'OpenAITTS': {
        'type': 'openai', 'model': 'gemini-tts', 'voice': 'Leda', 'speed': 1,
        'api_key': 'never-return-this-key', 'api_url': 'https://speech.example/v1/audio/speech'}}}
    profile = speech_settings.tts_profile(config)
    assert profile['voice'] == 'Leda' and 'never-return-this-key' not in json.dumps(profile)
    assert 'speech.example' not in json.dumps(profile)
    for field, value in [('voice', 'Kore'), ('model', 'another-model'), ('speed', 1.25), ('api_url', 'https://another.example/speech')]:
        changed = copy.deepcopy(config); changed['TTS']['OpenAITTS'][field] = value
        assert speech_settings.tts_profile(changed)['revision'] != profile['revision']


@pytest.mark.asyncio
async def test_tts_profile_is_local_only_and_reports_actual_instance(monkeypatch):
    from types import SimpleNamespace
    from core.admin import connections
    config = {'selected_module': {'TTS': 'OpenAITTS'}, 'TTS': {'OpenAITTS': {'model': 'new', 'voice': 'Leda'}}}
    h = AdminHandler(config)
    tts = SimpleNamespace(model='actual-model', voice='actual-voice', last_request={'model': 'actual-model', 'voice': 'actual-voice'})
    conn = SimpleNamespace(session_id='session', common_config=config, tts=tts)
    monkeypatch.setattr(connections, 'all_connections', lambda: [conn])
    denied = await h.handle_kiosk_tts_profile(SimpleNamespace(remote='192.0.2.5', headers={}))
    assert denied.status == 403
    response = await h.handle_kiosk_tts_profile(SimpleNamespace(remote='127.0.0.1', headers={}))
    result = json.loads(response.text)
    assert result['voice'] == 'Leda' and result['active'][0]['voice'] == 'actual-voice'
    assert result['active'][0]['last_request']['voice'] == 'actual-voice'


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


@pytest.mark.parametrize('section', ['ASR', 'TTS'])
def test_zenmux_settings_and_validation(tmp_path, section):
    path = tmp_path / 'config.yaml'; path.write_text('{}\n')
    merged = config_store.load_merged(custom_path=path)
    speech_settings.save(section, {'api_key': 'key', 'api_protocol': 'zenmux', 'format': 'pcm',
                                   'pcm_sample_rate': 24000}, merged, path)
    merged = config_store.load_merged(custom_path=path)
    assert speech_settings.describe(section, merged)['api_protocol'] == 'zenmux'
    original = path.read_bytes()
    for bad in ({'api_protocol': 'other'}, *([{'format': 'bad'}, {'pcm_sample_rate': 1}] if section == 'TTS' else [])):
        with pytest.raises(ValueError):
            speech_settings.save(section, bad, merged, path)
        assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_speech_apply_closes_only_this_gateways_stale_sessions(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from core.admin import connections
    from core.admin import admin_handler as ah
    path = tmp_path / '.config.yaml'; path.write_text('{}\n')
    config = config_store.load_merged(custom_path=path)
    ws = SimpleNamespace(config_lock=asyncio.Lock(), _asr=object())
    handler = AdminHandler(config, ws_server=ws, config_path=path)
    mine = SimpleNamespace(common_config=config, websocket=SimpleNamespace(close=AsyncMock()))
    other = SimpleNamespace(common_config={}, websocket=SimpleNamespace(close=AsyncMock()))
    monkeypatch.setattr(connections, 'all_connections', lambda: [mine, other])
    monkeypatch.setattr(ah, 'initialize_modules', lambda *a, **k: {'asr': object()})
    for section in ('ASR', 'TTS'):
        mine.websocket.close.reset_mock()
        result = await handler._apply(section)
        assert result['sessions_closed'] == 1
        mine.websocket.close.assert_awaited_once_with(code=1012, reason='Speech settings changed')
        other.websocket.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_asr_apply_does_not_close_sessions(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from core.admin import admin_handler as ah, connections
    path = tmp_path / '.config.yaml'; path.write_text('{}\n')
    config = config_store.load_merged(custom_path=path)
    conn = SimpleNamespace(common_config=config, websocket=SimpleNamespace(close=AsyncMock()))
    monkeypatch.setattr(connections, 'all_connections', lambda: [conn])
    def fail(*a, **kw):
        raise RuntimeError('model unavailable')
    monkeypatch.setattr(ah, 'initialize_modules', fail)
    h = AdminHandler(config, ws_server=SimpleNamespace(config_lock=asyncio.Lock(), _asr=object()), config_path=path)
    assert (await h._apply('ASR'))['failed']
    conn.websocket.close.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('language, phrase', [('en', 'Hello, this is an English speech test.'),
                                           ('yue-HK', '你好，呢個係廣東話語音測試。')])
async def test_speech_probes_use_saved_protocol_and_separate_credentials(tmp_path, monkeypatch, language, phrase):
    import base64
    import io
    import wave
    from types import SimpleNamespace
    from core.admin.admin_handler import COOKIE_NAME
    from core.providers.asr import openai as provider
    path = tmp_path / '.config.yaml'; path.write_text('reply_language: ' + language + '\n')
    for section in ('ASR', 'TTS'):
        speech_settings.save(section, {'api_key': section + '-private-key', 'api_protocol': 'zenmux',
                                       'voice': 'Kore', 'format': 'pcm'},
                             config_store.load_merged(custom_path=path), path)
    config = config_store.load_merged(custom_path=path)
    selected = dict(config['selected_module'])
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(b'\0\0'*1600)
    wav_bytes = buffer.getvalue()
    def post(url, **kw):
        section = 'TTS' if url.endswith('/speech') else 'ASR'
        assert kw['headers']['Authorization'] == 'Bearer ' + section + '-private-key'
        if section == 'ASR':
            assert base64.b64decode(kw['json']['input_audio']['data']) == wav_bytes
            return SimpleNamespace(status_code=200, json=lambda: {'text': '轉寫成功'})
        assert kw['json']['response_format'] == 'pcm'
        assert kw['json']['input'] == phrase  # never put language instructions in the spoken transcript
        return SimpleNamespace(status_code=200, content=b'json',
                               json=lambda: {'audio': base64.b64encode(wav_bytes).decode(), 'mime_type': 'audio/wav'})
    monkeypatch.setattr(provider.requests, 'post', post)
    h = AdminHandler(config, config_path=path)
    cookie = {COOKIE_NAME: h.sessions.create()}
    app = web.Application(); h.register(app)
    async with TestClient(TestServer(app)) as c:
        r = await c.post('/admin/api/speech/tts/test', json={}, cookies=cookie)
        assert r.status == 200
        result = await r.json()
        assert result['language'] == language and result['text'] == phrase
        assert base64.b64decode(result['audio']) == wav_bytes and result['mime_type'] == 'audio/wav'
        r = await c.post('/admin/api/speech/asr/test', json={'audio': base64.b64encode(wav_bytes).decode()}, cookies=cookie)
        assert r.status == 200 and (await r.json())['text'] == '轉寫成功'
        r = await c.post('/admin/api/speech/asr/test', json={'audio': 'invalid-base64'}, cookies=cookie)
        assert r.status == 400
        def error(*a, **kw):
            return SimpleNamespace(status_code=400, json=lambda: {'error': {'message': 'Unsupported voice TTS-private-key'}})
        monkeypatch.setattr(provider.requests, 'post', error)
        r = await c.post('/admin/api/speech/tts/test', json={}, cookies=cookie)
        assert r.status == 400
        message = await r.text()
        assert 'Unsupported voice' in message and 'TTS-private-key' not in message
    assert config_store.load_merged(custom_path=path)['selected_module'] == selected
