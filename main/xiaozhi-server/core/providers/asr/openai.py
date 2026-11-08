import time
import asyncio
import os
from config.logger import setup_logging
from typing import Optional, Tuple, List
from core.providers.asr.dto.dto import InterfaceType
from core.providers.asr.base import ASRProviderBase

import requests

TAG = __name__
logger = setup_logging()

class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        self.interface_type = InterfaceType.NON_STREAM
        self.api_key = config.get("api_key")
        self.api_url = config.get("base_url")
        self.model = config.get("model_name")        
        self.output_dir = config.get("output_dir") or 'tmp/'
        self.language = config.get('language') or ''
        self.prompt = config.get('prompt') or ''
        self.timeout = float(config.get('timeout') or 60)
        self.delete_audio_file = delete_audio_file

        os.makedirs(self.output_dir, exist_ok=True)

    def requires_file(self) -> bool:
        return True

    async def speech_to_text(self, opus_data: List[bytes], session_id: str, artifacts=None) -> Tuple[Optional[str], Optional[str]]:
        file_path = None
        try:
            if artifacts is None:
                return "", None
            file_path = artifacts.file_path
                
            logger.bind(tag=TAG).info(f"file path: {file_path}")
            start_time = time.time()
            response = await asyncio.to_thread(self._transcribe, file_path)
            logger.bind(tag=TAG).debug(f"语音识别耗时: {time.time() - start_time:.3f}s")

            if response.status_code == 200:
                text = response.json().get("text", "")
                return text, file_path
            else:
                raise Exception(f"API请求失败: HTTP {response.status_code}")
                
        except Exception as e:
            logger.bind(tag=TAG).error(f"语音识别失败: {e}")
            return "", None
        

    def _transcribe(self, file_path):
        # Keep the file alive in the worker even if the awaiting task is cancelled.
        data = {"model": self.model}
        if self.language:
            data["language"] = self.language
        if self.prompt:
            data["prompt"] = self.prompt
        with open(file_path, "rb") as audio_file:
            return requests.post(self.api_url, files={"file": audio_file}, data=data,
                                 headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout)
