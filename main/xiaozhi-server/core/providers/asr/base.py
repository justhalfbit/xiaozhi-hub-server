import os
import io
import wave
import uuid
import json
import time
import queue
import shutil
import asyncio
import tempfile
import traceback
import threading

from abc import ABC, abstractmethod
from config.logger import setup_logging
from core.providers.asr.dto.dto import InterfaceType
from core.handle.receiveAudioHandle import startToChat
from core.handle.reportHandle import enqueue_asr_report
from core.utils.util import remove_punctuation_and_length
from core.handle.receiveAudioHandle import handleAudioMessage
from core.agent.asr_text import correct_and_record  # [xiaozhi-hub]
from core.utils import reply_echo  # [xiaozhi-hub]
from typing import Optional, Tuple, List, NamedTuple, TYPE_CHECKING


if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

ASR_TAIL_BYTES = 16000 * 2 * 3 // 10  # [xiaozhi-hub] 送去识别的尾部静音：0.3 秒 16kHz 16 位
SEGMENT_MIN_BYTES = 16000 * 2  # [xiaozhi-hub] 边说边识别：不到 1 秒的段不单独识别（太短没有上下文）
# [xiaozhi-hub] 切在最后有声音之后 0.15 秒：停顿确认时（静音满 segment_pause_ms）下一个字可能已经开头了，
# VAD 判定有声音要滞后约 0.1 秒，切在当下会把它的开头切进上一段（"我， 我要"）
SEGMENT_CUT_BYTES = 16000 * 2 * 15 // 100


async def send_vad_event(conn: "ConnectionHandler", state: str):
    """[xiaozhi-hub] 给 hello 里声明了 features.vad_end 的客户端发 {"type": "vad", "state": ...}：
    end = 判定说完、开始识别（界面显示"正在识别"）；empty = 这句什么也没识别出来（咳嗽、太短、只有回声、
    识别出错），客户端不用再等文字，马上提示"没听清"。"""
    if not (conn.features or {}).get("vad_end"):
        return
    try:
        await conn.websocket.send(json.dumps({"type": "vad", "state": state, "session_id": conn.session_id}))
    except Exception:
        pass


def join_segments(texts: List[str]) -> str:
    """[xiaozhi-hub] 拼接分段识别的文字：前一段末尾的句号换成逗号（停顿处多半是一句话中间）。"""
    out = ""
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        if out:
            out = out.rstrip("。.")
            if out and out[-1] not in "，,？?！!；;、":
                out += "，"
            out += " "
        out += t
    return out


class ASRProviderBase(ABC):
    def __init__(self):
        pass

    # 打开音频通道
    async def open_audio_channels(self, conn: "ConnectionHandler"):
        conn.asr_priority_thread = threading.Thread(
            target=self.asr_text_priority_thread, args=(conn,), daemon=True
        )
        conn.asr_priority_thread.start()

    # 有序处理ASR音频
    def asr_text_priority_thread(self, conn: "ConnectionHandler"):
        while not conn.stop_event.is_set():
            try:
                message = conn.asr_audio_queue.get(timeout=1)
                future = asyncio.run_coroutine_threadsafe(
                    handleAudioMessage(conn, message),
                    conn.loop,
                )
                future.result()
            except queue.Empty:
                continue
            except Exception as e:
                logger.bind(tag=TAG).error(
                    f"处理ASR文本失败: {str(e)}, 类型: {type(e).__name__}, 堆栈: {traceback.format_exc()}"
                )
                continue

    # 接收音频
    async def receive_audio(self, conn: "ConnectionHandler", pcm_frame, audio_have_voice):
        if conn.client_listen_mode == "manual":
            # 手动模式：缓存音频用于ASR识别
            conn.asr_audio.append(pcm_frame)
        else:
            # 自动/实时模式：使用VAD检测
            conn.asr_audio.append(pcm_frame)
            if audio_have_voice:
                conn.asr_voice_end = len(conn.asr_audio)  # [xiaozhi-hub] 最后一帧有声音的位置

            # 如果没有语音，且之前也没有声音，缓存部分音频
            if not audio_have_voice and not conn.client_have_voice:
                conn.asr_audio = conn.asr_audio[-10:]
                return

            if conn.asr.interface_type != InterfaceType.STREAM and not conn.client_voice_stop:
                self._maybe_decode_segment(conn)

            # 自动模式下通过VAD检测到语音停止时触发识别
            if conn.asr.interface_type != InterfaceType.STREAM and conn.client_voice_stop:
                await send_vad_event(conn, "end")  # [xiaozhi-hub]
                # [xiaozhi-hub] 判定说完要等约 0.8 秒静音，这段静音不用识别：只留最后有声音之后的 0.3 秒
                # （识别耗时和音频长短成正比，板子上约省 0.25 秒；0.3 秒以内结尾的字不受影响）
                voice_end = getattr(conn, "asr_voice_end", 0) or len(conn.asr_audio)
                pcm_bytes = b"".join(conn.asr_audio[:voice_end]) + b"".join(conn.asr_audio[voice_end:])[:ASR_TAIL_BYTES]
                # 检查是否有足够的音频数据（每帧1920字节，15帧约28800字节）
                if len(pcm_bytes) > 1920 * 15:
                    segments = getattr(conn, "asr_segments", None) or []
                    start = getattr(conn, "asr_segment_start", 0)
                    # 最后一段切完后没再说话：剩下的只是静音，不用再识别
                    done = len(pcm_bytes) if segments and voice_end <= start else \
                        sum(len(f) for f in conn.asr_audio[:start])
                    await self.handle_voice_stop(conn, [pcm_bytes], segments=segments, segmented_bytes=done)
                else:
                    await send_vad_event(conn, "empty")  # [xiaozhi-hub] 太短（咳一声）：不识别
                conn.reset_audio_states()

    def _maybe_decode_segment(self, conn: "ConnectionHandler"):
        """[xiaozhi-hub] 边说边识别：句中停顿够长（segment_pause_ms）就把这一段先送去识别，
        说完时只剩最后一段要识别。识别耗时和音频长短成正比，长句说完后快 1.5～3 秒；
        代价是跨段没有上下文，偶尔认错同音字（如"她"认成"他"）。只用于能直接识别 PCM 的本地模型。"""
        pause_ms = getattr(self, "segment_pause_ms", 0)
        if not pause_ms or not hasattr(self, "recognize_pcm") or not conn.client_have_voice:
            return
        start = getattr(conn, "asr_segment_start", 0)
        voice_end = getattr(conn, "asr_voice_end", 0)
        if voice_end <= start:
            return  # 这次停顿已经切过了，或者上次切完还没再说话
        silent = sum(len(f) for f in conn.asr_audio[voice_end:])
        if silent < pause_ms * 32:
            return
        end, kept = voice_end, 0
        while end < len(conn.asr_audio) and kept < SEGMENT_CUT_BYTES:
            kept += len(conn.asr_audio[end])
            end += 1
        pcm = b"".join(conn.asr_audio[start:end])
        if len(pcm) < SEGMENT_MIN_BYTES:
            return
        future = asyncio.get_running_loop().run_in_executor(None, self.recognize_pcm, pcm)
        conn.asr_segments = (getattr(conn, "asr_segments", None) or []) + [future]
        conn.asr_segment_start = end
        logger.bind(tag=TAG).debug(f"边说边识别：第 {len(conn.asr_segments)} 段 {len(pcm) / 32000:.1f} 秒")

    # 处理语音停止
    async def handle_voice_stop(self, conn: "ConnectionHandler", asr_audio_task: List[bytes],
                                segments=None, segmented_bytes: int = 0):
        """并行处理ASR和声纹识别

        [xiaozhi-hub] segments：说话期间已经送去识别的前几段（future），asr_audio_task 的前 segmented_bytes
        字节就是它们；这里只识别剩下的部分，再按顺序拼起来。"""
        try:
            total_start_time = time.monotonic()

            # 数据已经是PCM直接使用
            pcm_data = asr_audio_task
            combined_pcm_data = b"".join(pcm_data)
            if segments:
                asr_audio_task = [combined_pcm_data[segmented_bytes:]]

            # 预先准备WAV数据
            wav_data = None
            if conn.voiceprint_provider and combined_pcm_data:
                wav_data = self._pcm_to_wav(combined_pcm_data)

            # 定义ASR任务
            if segments and len(asr_audio_task[0]) < 3200:
                # [xiaozhi-hub] 最后的停顿处已经切过段：全都识别过了
                async def nothing():
                    return "", None
                asr_task = nothing()
            else:
                asr_task = self.speech_to_text_wrapper(
                    asr_audio_task, conn.session_id
                )

            if conn.voiceprint_provider and wav_data:
                voiceprint_task = conn.voiceprint_provider.identify_speaker(
                    wav_data, conn.session_id
                )
                # 并发等待两个结果
                asr_result, voiceprint_result = await asyncio.gather(
                    asr_task, voiceprint_task, return_exceptions=True
                )
            else:
                asr_result = await asr_task
                voiceprint_result = None

            # 记录识别结果 - 检查是否为异常
            if isinstance(asr_result, Exception):
                logger.bind(tag=TAG).error(f"ASR识别失败: {asr_result}")
                raw_text = ""
            else:
                raw_text, _ = asr_result
            if segments:
                earlier = await asyncio.gather(*segments, return_exceptions=True)
                texts = ["" if isinstance(t, Exception) else t for t in earlier]
                raw_text = join_segments(texts + [raw_text or ""])
                logger.bind(tag=TAG).info(
                    f"边说边识别：{len(segments)} 段说话时已识别，说完只识别最后 "
                    f"{len(asr_audio_task[0]) / 32000:.1f} 秒，用时 {time.monotonic() - total_start_time:.2f} 秒")
            # [xiaozhi-hub] 识别后纠错替换（如 cloud code → Claude Code），并记一条最近识别记录供管理后台查看
            raw_text = correct_and_record(conn, raw_text)

            if isinstance(voiceprint_result, Exception):
                logger.bind(tag=TAG).error(f"声纹识别失败: {voiceprint_result}")
                speaker_name = ""
            else:
                speaker_name = voiceprint_result

            # 判断 ASR 结果类型
            if isinstance(raw_text, dict):
                # FunASR 返回的 dict 格式
                if speaker_name:
                    raw_text["speaker"] = speaker_name

                # 记录识别结果
                if raw_text.get("language"):
                    logger.bind(tag=TAG).info(f"识别语言: {raw_text['language']}")
                if raw_text.get("emotion"):
                    logger.bind(tag=TAG).info(f"识别情绪: {raw_text['emotion']}")
                if raw_text.get("content"):
                    logger.bind(tag=TAG).info(f"识别文本: {raw_text['content']}")
                if speaker_name:
                    logger.bind(tag=TAG).info(f"识别说话人: {speaker_name}")

                # 转换为 JSON 字符串用于下游
                enhanced_text = json.dumps(raw_text, ensure_ascii=False)
                content_for_length_check = raw_text.get("content", "")
            else:
                # 其他 ASR 返回的纯文本
                if raw_text:
                    logger.bind(tag=TAG).info(f"识别文本: {raw_text}")
                if speaker_name:
                    logger.bind(tag=TAG).info(f"识别说话人: {speaker_name}")

                # 构建包含说话人信息的JSON字符串
                enhanced_text = self._build_enhanced_text(raw_text, speaker_name)
                content_for_length_check = raw_text

            # 性能监控
            total_time = time.monotonic() - total_start_time
            logger.bind(tag=TAG).debug(f"总处理耗时: {total_time:.3f}s")

            # 检查文本长度
            text_len, _ = remove_punctuation_and_length(content_for_length_check)
            self.stop_ws_connection()

            # [xiaozhi-hub] 回复期间识别出的是回复自己的回声（开着回声消除时麦克风不关）就丢掉
            if text_len > 0 and reply_echo.during_reply(conn):
                logger.bind(tag=TAG).info(
                    f"回复期间识别: {content_for_length_check} | 峰值 {reply_echo.level_db(pcm_data):.1f} dBFS")
                if reply_echo.is_echo(conn, content_for_length_check):
                    logger.bind(tag=TAG).info(f"忽略回声识别: {content_for_length_check}")
                    text_len = 0

            if text_len > 0:
                audio_snapshot = pcm_data.copy()
                enqueue_asr_report(conn, enhanced_text, audio_snapshot)
                # 使用自定义模块进行上报
                await startToChat(conn, enhanced_text)
            else:
                await send_vad_event(conn, "empty")  # [xiaozhi-hub] 没识别出字，或只是回复的回声
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理语音停止失败: {e}")
            await send_vad_event(conn, "empty")  # [xiaozhi-hub]
            import traceback

            logger.bind(tag=TAG).debug(f"异常详情: {traceback.format_exc()}")

    def _build_enhanced_text(self, text: str, speaker_name: Optional[str]) -> str:
        """构建包含说话人信息的文本（仅用于纯文本ASR）"""
        if speaker_name and speaker_name.strip():
            return json.dumps(
                {"speaker": speaker_name, "content": text}, ensure_ascii=False
            )
        else:
            return text

    def _pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """将PCM数据转换为WAV格式"""
        if len(pcm_data) == 0:
            logger.bind(tag=TAG).warning("PCM数据为空，无法转换WAV")
            return b""

        # 确保数据长度是偶数（16位音频）
        if len(pcm_data) % 2 != 0:
            pcm_data = pcm_data[:-1]

        # 创建WAV文件头
        wav_buffer = io.BytesIO()
        try:
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16位
                wav_file.setframerate(16000)  # 16kHz采样率
                wav_file.writeframes(pcm_data)

            wav_buffer.seek(0)
            wav_data = wav_buffer.read()

            return wav_data
        except Exception as e:
            logger.bind(tag=TAG).error(f"WAV转换失败: {e}")
            return b""

    def stop_ws_connection(self):
        pass

    async def close(self):
        pass

    class AudioArtifacts(NamedTuple):
        pcm_frames: List[bytes]
        """PCM音频帧列表"""
        pcm_bytes: bytes
        """合并后的PCM音频字节数据"""
        file_path: Optional[str]
        """WAV文件路径"""
        temp_path: Optional[str]
        """临时WAV文件路径"""

    def get_current_artifacts(self) -> Optional["ASRProviderBase.AudioArtifacts"]:
        return self._current_artifacts

    def requires_file(self) -> bool:
        """是否需要文件输入"""
        return False

    def prefers_temp_file(self) -> bool:
        """是否优先使用临时文件"""
        return False

    def build_temp_file(self, pcm_bytes: bytes) -> Optional[str]:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            with wave.open(temp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(pcm_bytes)
            return temp_path
        except Exception as e:
            logger.bind(tag=TAG).error(f"临时音频文件生成失败: {e}")
            return None

    def save_audio_to_file(self, pcm_data: List[bytes], session_id: str) -> str:
        """PCM数据保存为WAV文件"""
        module_name = __name__.split(".")[-1]
        file_name = f"asr_{module_name}_{session_id}_{uuid.uuid4()}.wav"
        file_path = os.path.join(self.output_dir, file_name)

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"".join(pcm_data))

        return file_path

    async def speech_to_text_wrapper(
        self, pcm_data: List[bytes], session_id: str
    ) -> Tuple[Optional[str], Optional[str]]:
        file_path = None
        temp_path = None
        try:
            combined_pcm_data = b"".join(pcm_data)

            free_space = shutil.disk_usage(self.output_dir).free
            if free_space < len(combined_pcm_data) * 2:
                raise OSError("磁盘空间不足")

            if self.requires_file() and self.prefers_temp_file():
                temp_path = self.build_temp_file(combined_pcm_data)

            if (hasattr(self, "delete_audio_file") and not self.delete_audio_file) or (
                self.requires_file() and not self.prefers_temp_file()
            ):
                file_path = self.save_audio_to_file(pcm_data, session_id)

            if len(combined_pcm_data) == 0:
                artifacts = None
            else:
                artifacts = ASRProviderBase.AudioArtifacts(
                    pcm_frames=pcm_data,
                    pcm_bytes=combined_pcm_data,
                    file_path=file_path,
                    temp_path=temp_path,
                )

            text, _ = await self.speech_to_text(
                pcm_data, session_id, artifacts
            )
            return text, file_path
        except OSError as e:
            logger.bind(tag=TAG).error(f"文件操作错误: {e}")
            return None, None
        except Exception as e:
            logger.bind(tag=TAG).error(f"语音识别失败: {e}")
            return None, None
        finally:
            try:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                if (
                    hasattr(self, "delete_audio_file")
                    and self.delete_audio_file
                    and file_path
                    and os.path.exists(file_path)
                ):
                    os.remove(file_path)
            except Exception as e:
                logger.bind(tag=TAG).error(f"文件清理失败: {e}")

    @abstractmethod
    async def speech_to_text(
        self,
        opus_data: List[bytes],
        session_id: str,
        artifacts: Optional[AudioArtifacts] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """将语音数据转换为文本

        :param opus_data: 输入的Opus音频数据
        :param session_id: 会话ID
        :param artifacts: 音频工件，包含PCM数据、文件路径等
        :return: 识别结果文本和文件路径（如果有）
        """
        pass
