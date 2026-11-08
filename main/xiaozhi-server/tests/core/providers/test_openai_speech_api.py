"""Speech API requests carry separate credentials and do not block the event loop."""
import asyncio
import threading
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
