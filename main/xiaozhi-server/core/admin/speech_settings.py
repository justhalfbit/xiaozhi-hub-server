"""OpenAI-compatible speech settings; credentials remain write-only."""
import math
import re
from urllib.parse import urlsplit

from ruamel.yaml.comments import CommentedMap

from core.admin import config_store

SPEECH = {
    'ASR': {'entry': 'OpenAIASR', 'endpoint': '/audio/transcriptions', 'model_field': 'model_name',
            'url_field': 'base_url', 'model': 'whisper-1'},
    'TTS': {'entry': 'OpenAITTS', 'endpoint': '/audio/speech', 'model_field': 'model',
            'url_field': 'api_url', 'model': 'tts-1'},
}


def describe(section, merged):
    spec = SPEECH[section]
    entry = (merged.get(section) or {}).get(spec['entry']) or {}
    endpoint = str(entry.get(spec['url_field']) or 'https://api.openai.com/v1' + spec['endpoint'])
    key = entry.get('api_key')
    result = {'entry': spec['entry'], 'base_url': endpoint.removesuffix(spec['endpoint']),
              'model': entry.get(spec['model_field']) or spec['model'],
              'key_set': not config_store.is_placeholder(key), 'key_hint': config_store.mask_key(key),
              'conflict': bool(entry) and entry.get('type') != 'openai'}
    result['api_protocol'] = entry.get('api_protocol') or 'openai'
    result['configured'] = result['key_set'] and not result['conflict']
    if section == 'ASR':
        result.update(language=entry.get('language') or '', prompt=entry.get('prompt') or '')
    else:
        result.update(voice=entry.get('private_voice') or entry.get('voice') or 'alloy',
                      speed=entry.get('speed') or 1, format=entry.get('format') or 'wav',
                      pcm_sample_rate=entry.get('pcm_sample_rate') or 24000)
    return result


def save(section, payload, merged, path=None):
    spec = SPEECH[section]
    old = (merged.get(section) or {}).get(spec['entry']) or {}
    if old and old.get('type') != 'openai':
        raise ValueError(f"{spec['entry']} 已存在，但不是 OpenAI 相容類型")
    defaults = describe(section, merged)
    protocol = payload.get('api_protocol', defaults['api_protocol'])
    if protocol not in ('openai', 'zenmux'):
        raise ValueError('請選擇 OpenAI 或 ZenMux 語音協定')
    base = str(payload.get('base_url', defaults['base_url'])).strip().rstrip('/')
    url = urlsplit(base)
    if (url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password
            or url.query or url.fragment or any(c.isspace() for c in base)):
        raise ValueError('Base URL 必須是完整的 HTTP(S) 網址，不可含帳密、查詢參數或空白')
    model = str(payload.get('model', defaults['model'])).strip()
    if not model or len(model) > 200:
        raise ValueError('請填寫模型名稱（最多 200 字元）')
    key = str(payload.get('api_key') or '').strip()
    if any(c.isspace() for c in key):
        raise ValueError('API Key 不可含空白或換行')
    if not key and config_store.is_placeholder(old.get('api_key')):
        raise ValueError('請填寫 API Key')
    values = {'type': 'openai', 'api_protocol': protocol,
              spec['url_field']: base + spec['endpoint'], spec['model_field']: model}
    if key:
        values['api_key'] = key
    if not old:
        values['output_dir'] = 'tmp/'
    if section == 'ASR':
        language = str(payload.get('language', defaults['language'])).strip()
        if language and not re.fullmatch(r'[a-z]{2,3}', language):
            raise ValueError('辨識語言請使用 ISO 639 代碼，例如 zh 或 en；留空則自動偵測')
        prompt = str(payload.get('prompt', defaults['prompt'])).strip()
        if len(prompt) > 4000:
            raise ValueError('轉寫提示最多 4000 字元')
        values.update(language=language, prompt=prompt)
    else:
        voice = str(payload.get('voice', defaults['voice'])).strip()
        try:
            speed = float(payload.get('speed', defaults['speed']))
        except (TypeError, ValueError):
            raise ValueError('語速必須介於 0.25 和 4 之間')
        if not voice or len(voice) > 200 or not math.isfinite(speed) or not .25 <= speed <= 4:
            raise ValueError('請填寫音色，語速必須介於 0.25 和 4 之間')
        values.update(voice=voice, speed=speed)
        audio_format = payload.get('format', defaults['format'])
        if audio_format not in ('wav', 'mp3', 'pcm', 'opus', 'aac', 'flac'):
            raise ValueError('不支援的音訊格式')
        rate = payload.get('pcm_sample_rate', defaults['pcm_sample_rate'])
        if str(rate) not in ('8000', '16000', '22050', '24000', '32000', '44100', '48000'):
            raise ValueError('請選擇有效的 PCM 取樣率')
        values.update(format=audio_format, pcm_sample_rate=int(rate))
        if old.get('private_voice'):
            values['private_voice'] = ''
    changes = {k: v for k, v in values.items() if old.get(k) != v}
    if changes:
        with config_store.edit_custom(path) as data:
            data.setdefault(section, CommentedMap()).setdefault(spec['entry'], CommentedMap()).update(changes)
    return list(changes)
