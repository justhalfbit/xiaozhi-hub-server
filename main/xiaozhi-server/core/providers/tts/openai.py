import requests
import asyncio
import base64
import binascii
import io
import wave
from core.utils.util import check_model_key
from core.providers.tts.base import TTSProviderBase
from config.logger import setup_logging
from core.utils.speech_api import response_error

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    TTS_PARAM_CONFIG = [
        ("ttsRate", "speed", 0.25, 4, 1, lambda v: round(float(v), 2)),
    ]

    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.api_key = config.get("api_key")
        self.api_url = config.get("api_url", "https://api.openai.com/v1/audio/speech")
        self.model = config.get("model", "tts-1")
        if config.get("private_voice"):
            self.voice = config.get("private_voice")
        else:
            self.voice = config.get("voice", "alloy")
        self.response_format = config.get('format') or 'wav'
        self.pcm_sample_rate = int(config.get('pcm_sample_rate') or 24000)
        self.audio_file_type = 'wav' if self.response_format == 'pcm' else self.response_format
        self.api_protocol = config.get('api_protocol', 'openai')

        # 处理空字符串的情况
        speed = config.get("speed", "1.0")
        self.speed = float(speed) if speed else 1.0

        # 应用百分比调整（如果存在），否则使用公有化配置
        self._apply_percentage_params(config)

        self.output_file = config.get("output_dir", "tmp/")
        model_key_msg = check_model_key("TTS", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)

    async def text_to_speak(self, text, output_file):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": self.response_format,
            "speed": self.speed,
        }
        response = await asyncio.to_thread(requests.post,
            self.api_url, json=data, headers=headers, timeout=self.tts_timeout
        )
        if response.status_code == 200:
            audio = response.content
            if self.api_protocol == 'zenmux':
                try:
                    body = response.json()
                    audio = base64.b64decode(body['audio'], validate=True)
                except (ValueError, KeyError, TypeError, binascii.Error):
                    raise RuntimeError('ZenMux TTS 回應缺少有效的 Base64 audio 欄位') from None
            if not audio:
                raise RuntimeError('TTS 回應沒有音訊')
            # ZenMux can return a WAV container even when response_format=pcm.
            if self.response_format == 'pcm' and not (audio[:4] == b'RIFF' and audio[8:12] == b'WAVE'):
                if len(audio) % 2:
                    raise RuntimeError('PCM 音訊長度不符合 16-bit 格式')
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(self.pcm_sample_rate)
                    wav.writeframes(audio)
                audio = buffer.getvalue()
            if output_file:
                with open(output_file, "wb") as audio_file:
                    audio_file.write(audio)
            else:
                return audio
        else:
            raise response_error(response, self.api_key, 'TTS')
