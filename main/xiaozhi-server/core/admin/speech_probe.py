"""Explicit, authenticated speech tests use the saved provider credentials."""
import base64
import binascii
import io
import tempfile
import wave
from pathlib import Path

from core.admin import speech_settings
from core.utils.modules_initialize import initialize_asr, initialize_tts


async def test(section, body, merged):
    details = speech_settings.describe(section, merged)
    if not details['configured']:
        raise ValueError('請先儲存語音 API 設定')
    config = {**merged, 'selected_module': {**merged.get('selected_module', {}), section: details['entry']}}
    if section == 'TTS':
        provider = initialize_tts(config)
        audio = await provider.text_to_speak('你好，這是語音測試。', None)
        return {'audio': base64.b64encode(audio).decode('ascii'),
                'mime_type': 'audio/mpeg' if provider.audio_file_type == 'mp3' else 'audio/' + provider.audio_file_type}
    try:
        audio = base64.b64decode(body.get('audio', ''), validate=True)
    except (ValueError, TypeError, binascii.Error):
        raise ValueError('請上傳有效的 WAV 音檔') from None
    if not audio or len(audio) > 700_000:
        raise ValueError('請上傳小於 700 KB 的 WAV 音檔')
    try:
        with wave.open(io.BytesIO(audio), 'rb') as wav:
            if not 0 < wav.getnframes() / wav.getframerate() <= 30:
                raise ValueError('測試音檔長度須介於 0 和 30 秒之間')
    except (wave.Error, EOFError):
        raise ValueError('請上傳有效的 WAV 音檔') from None
    with tempfile.TemporaryDirectory(prefix='speech-probe-') as tmp:
        file = Path(tmp) / 'test.wav'
        file.write_bytes(audio)
        provider = initialize_asr(config)
        return {'text': await provider.transcribe_file(str(file))}
