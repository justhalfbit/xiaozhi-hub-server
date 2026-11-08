"""Speech API requests carry separate credentials and do not block the event loop."""
import asyncio
import base64
import io
import threading
import wave
from types import SimpleNamespace

import pytest
from core.providers.asr import openai as asr_provider
from core.providers.tts import openai as tts_provider


@pytest.mark.asyncio
async def test_transcription_form_and_worker_thread(tmp_path, monkeypatch):
    provider = asr_provider
    file = tmp_path / 'speech.wav'; file.write_bytes(b'RIFF-audio')
    main_thread = threading.get_ident()
    def post(url, **kwargs):
        assert threading.get_ident() != main_thread
        assert url == 'https://asr.example/v1/audio/transcriptions'
        assert kwargs['headers']['Authorization'] == 'Bearer asr-test-key'
        assert kwargs['data'] == {'model': 'transcribe-model', 'language': 'zh', 'prompt': '請用繁體中文'}
        assert kwargs['timeout'] == 60
        assert kwargs['files']['file'].read() == b'RIFF-audio'
        return SimpleNamespace(status_code=200, json=lambda: {'text': '測試轉寫'})
    monkeypatch.setattr(provider.requests, 'post', post)
    asr = provider.ASRProvider({'api_key': 'asr-test-key', 'base_url': 'https://asr.example/v1/audio/transcriptions',
                               'model_name': 'transcribe-model', 'language': 'zh', 'prompt': '請用繁體中文',
                               'output_dir': str(tmp_path)}, False)
    assert await asr.speech_to_text([], 'test', SimpleNamespace(file_path=str(file))) == ('測試轉寫', str(file))


@pytest.mark.asyncio
async def test_cancelled_transcription_keeps_file_alive_until_worker_finishes(tmp_path, monkeypatch):
    provider = asr_provider
    file = tmp_path / 'speech.wav'; file.write_bytes(b'audio')
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    open_state = []
    def post(url, **kwargs):
        started.set()
        release.wait(2)
        open_state.append(not kwargs['files']['file'].closed)
        finished.set()
        return SimpleNamespace(status_code=200, json=lambda: {'text': 'ok'})
    monkeypatch.setattr(provider.requests, 'post', post)
    asr = provider.ASRProvider({'api_key': 'test', 'base_url': 'https://asr.example', 'model_name': 'test',
                               'output_dir': str(tmp_path)}, False)
    task = asyncio.create_task(asr.speech_to_text([], 'test', SimpleNamespace(file_path=str(file))))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await asyncio.to_thread(finished.wait, 2)
    assert open_state == [True]


@pytest.mark.asyncio
async def test_speech_output_and_separate_key(tmp_path, monkeypatch):
    provider = tts_provider
    main_thread = threading.get_ident()
    def post(url, **kwargs):
        assert threading.get_ident() != main_thread
        assert url == 'https://tts.example/v1/audio/speech'
        assert kwargs['headers']['Authorization'] == 'Bearer tts-test-key'
        assert kwargs['json'] == {'model': 'tts-model', 'input': '你好', 'voice': 'voice-id',
                                  'response_format': 'wav', 'speed': 1.25}
        assert kwargs['timeout'] > 0
        return SimpleNamespace(status_code=200, content=b'RIFF-tts-output')
    monkeypatch.setattr(provider.requests, 'post', post)
    tts = provider.TTSProvider({'api_key': 'tts-test-key', 'api_url': 'https://tts.example/v1/audio/speech',
                               'model': 'tts-model', 'voice': 'voice-id', 'speed': 1.25}, False)
    output = tmp_path / 'reply.wav'
    await tts.text_to_speak('你好', str(output))
    assert output.read_bytes() == b'RIFF-tts-output'


@pytest.mark.asyncio
async def test_zenmux_transcription_uses_json_input_audio(tmp_path, monkeypatch):
    file = tmp_path / 'speech.wav'; file.write_bytes(b'RIFF-audio')
    def post(url, **kw):
        assert 'files' not in kw and 'data' not in kw
        assert kw['json'] == {'model': 'qwen/asr', 'language': 'zh', 'prompt': '繁體中文',
                              'input_audio': {'data': base64.b64encode(b'RIFF-audio').decode(), 'format': 'wav'}}
        return SimpleNamespace(status_code=200, json=lambda: {'text': '你好'})
    monkeypatch.setattr(asr_provider.requests, 'post', post)
    asr = asr_provider.ASRProvider({'api_protocol': 'zenmux', 'api_key': 'asr-key', 'model_name': 'qwen/asr',
                                  'base_url': 'https://zenmux.example/api/v1/audio/transcriptions',
                                  'language': 'zh', 'prompt': '繁體中文', 'output_dir': str(tmp_path)}, False)
    assert await asr.transcribe_file(file) == '你好'


@pytest.mark.asyncio
@pytest.mark.parametrize('container', [False, True])
async def test_zenmux_base64_audio_and_pcm_playback(monkeypatch, container):
    raw = b'\x01\x00' * 2400
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000); wav.writeframes(raw)
    payload = buffer.getvalue() if container else raw
    def post(url, **kw):
        assert kw['json']['response_format'] == 'pcm' and kw['json']['voice'] == 'Kore'
        return SimpleNamespace(status_code=200, content=b'JSON, not audio',
                               json=lambda: {'audio': base64.b64encode(payload).decode(), 'mime_type': 'audio/wav'})
    monkeypatch.setattr(tts_provider.requests, 'post', post)
    provider = tts_provider.TTSProvider({'api_protocol': 'zenmux', 'api_key': 'tts-key',
                                       'format': 'pcm', 'voice': 'Kore'}, True)
    audio = await provider.text_to_speak('測試', None)
    assert provider.audio_file_type == 'wav'
    with wave.open(io.BytesIO(audio), 'rb') as wav:
        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == 24000
        assert wav.readframes(wav.getnframes()) == raw  # no JSON, Base64 or duplicate WAV header


@pytest.mark.asyncio
async def test_openai_raw_pcm_is_wrapped_for_decoder(monkeypatch):
    monkeypatch.setattr(tts_provider.requests, 'post', lambda *a, **kw:
                        SimpleNamespace(status_code=200, content=b'\0\0' * 1600))
    provider = tts_provider.TTSProvider({'api_key': 'key', 'format': 'pcm', 'pcm_sample_rate': 16000}, True)
    with wave.open(io.BytesIO(await provider.text_to_speak('test', None)), 'rb') as wav:
        assert wav.getframerate() == 16000 and wav.getnframes() == 1600


@pytest.mark.asyncio
async def test_provider_error_is_useful_without_exposing_key(monkeypatch):
    response = SimpleNamespace(status_code=400, json=lambda: {'error': {'message': 'Unsupported format: wav; secret-key-123'}})
    monkeypatch.setattr(tts_provider.requests, 'post', lambda *a, **kw: response)
    provider = tts_provider.TTSProvider({'api_key': 'secret-key-123'}, True)
    with pytest.raises(RuntimeError, match='HTTP 400.*Unsupported format') as error:
        await provider.text_to_speak('test', None)
    assert 'secret-key-123' not in str(error.value)
