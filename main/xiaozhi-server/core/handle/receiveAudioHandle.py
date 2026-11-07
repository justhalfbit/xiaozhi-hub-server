import time
import json
import uuid
import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
from core.utils.util import audio_to_data
from core.agent import clear_command
from core.agent.call_names import handle_call_name
from core.handle.abortHandle import handleAbortMessage
from core.handle.intentHandler import handle_user_intent, speak_txt
from core.utils.output_counter import check_device_output_limit
from core.utils import turn_gen
from core.handle.sendAudioHandle import send_stt_message, SentenceType

TAG = __name__

# 唤醒/文字提问后忽略音频的时长（秒），见 mark_just_woken_up
JUST_WOKEN_UP_SECONDS = 2


def mark_just_woken_up(conn: "ConnectionHandler"):
    """唤醒后短暂忽略收到的音频（防问候语回声），从现在起 JUST_WOKEN_UP_SECONDS 秒内有效。"""
    conn.just_woken_up = True
    conn.just_woken_up_at = time.monotonic()


async def handleAudioMessage(conn: "ConnectionHandler", pcm_frame):
    # 当前片段是否有人说话
    have_voice = conn.vad.is_vad(conn, pcm_frame)
    # 如果设备刚刚被唤醒，短暂忽略VAD检测。
    # 窗口从打上标记时算起（JUST_WOKEN_UP_SECONDS），不是从第一帧音频算起：
    # 文字提问后设备可能很久不送音频，旧写法会让标记一直留着，
    # 下次语音唤醒时头 2 秒被扔掉，第一句识别为空。
    if getattr(conn, "just_woken_up", False):
        if (
            time.monotonic() - getattr(conn, "just_woken_up_at", 0.0)
            < JUST_WOKEN_UP_SECONDS
        ):
            return
        conn.just_woken_up = False
    # 服务端AEC功能需要实时触发打断
    if conn.client_aec and have_voice:
        if conn.client_is_speaking and conn.client_listen_mode != "manual":
            await handleAbortMessage(conn)
    # 设备长时间空闲检测，用于say goodbye
    await no_voice_close_connect(conn, have_voice)
    # 接收音频
    await conn.asr.receive_audio(conn, pcm_frame, have_voice)


async def startToChat(conn: "ConnectionHandler", text, gen=None):
    """gen：这句话说完时的轮次代次（turn_gen）；之后设备打断过就不再回答它。"""
    if gen is not None and turn_gen.stale(conn, gen):
        conn.logger.bind(tag=TAG).info(f"这句话说完后被打断过，不再回答: {text}")
        return
    # 检查输入是否是JSON格式（包含说话人信息）
    speaker_name = None
    actual_text = text

    try:
        # 尝试解析JSON格式的输入
        if text.strip().startswith("{") and text.strip().endswith("}"):
            data = json.loads(text)
            if "speaker" in data and "content" in data:
                speaker_name = data["speaker"]
                actual_content = data["content"]
                conn.logger.bind(tag=TAG).info(f"解析到说话人信息: {speaker_name}")

                # 仅在该说话人首次出现时保留 {"speaker":...} JSON，让模型自然称呼一次；
                # 后续轮降为纯文本，避免每轮重复出现名字诱导模型反复称呼
                if speaker_name not in conn.introduced_speakers:
                    conn.introduced_speakers.add(speaker_name)
                    actual_text = text
                else:
                    actual_text = actual_content
            elif "content" in data:
                # FunASR 等返回 {"content","language","emotion"}：只把识别正文交给 LLM，
                # 否则整段 JSON 会进入对话历史
                actual_text = data["content"]
    except (json.JSONDecodeError, KeyError):
        # 如果解析失败，继续使用原始文本
        pass

    # 保存说话人信息到连接对象
    if speaker_name:
        conn.current_speaker = speaker_name
    else:
        conn.current_speaker = None

    if conn.need_bind:
        await check_bind_device(conn)
        return

    # 如果当日的输出字数大于限定的字数
    if conn.max_output_size > 0:
        if check_device_output_limit(
            conn.headers.get("device-id"), conn.max_output_size
        ):
            await max_out_size(conn)
            return

    # manual 模式下不打断正在播放的内容
    if conn.client_is_speaking and conn.client_listen_mode != "manual":
        await handleAbortMessage(conn)

    # [xiaozhi-hub] 新的一轮从这里开始：之前的轮次作废；下面呼名、意图识别、发 stt 都要等，
    # 期间设备打断了就不再回答
    gen = turn_gen.bump(conn)

    # [xiaozhi-hub] 呼名：句首"嗨 Claude"先去掉称呼；只有称呼时直接回复，不发给模型
    actual_text = await handle_call_name(conn, actual_text)
    if actual_text is None:
        return

    # [xiaozhi-hub] "清除聊天记录"：先问一遍，说确定才清（不交给模型，各种模型都能用）
    actual_text = await clear_command.handle(conn, actual_text)
    if actual_text is None:
        return

    # 首先进行意图分析，使用实际文本内容
    intent_handled = await handle_user_intent(conn, actual_text)

    if intent_handled:
        # 如果意图已被处理，不再进行聊天
        return

    if turn_gen.stale(conn, gen):
        conn.logger.bind(tag=TAG).info(f"开始回答前被打断，不再回答: {actual_text}")
        return

    # 意图未被处理，继续常规聊天流程，使用实际文本内容
    await send_stt_message(conn, actual_text)

    # 准备开始新会话
    conn.client_abort = False

    conn.executor.submit(conn.chat, actual_text, turn=gen)


async def no_voice_close_connect(conn: "ConnectionHandler", have_voice):
    if have_voice:
        conn.last_activity_time = time.time() * 1000
        return
    # 只有在已经初始化过时间戳的情况下才进行超时检查
    if conn.last_activity_time > 0.0:
        no_voice_time = time.time() * 1000 - conn.last_activity_time
        close_connection_no_voice_time = int(
            conn.config.get("close_connection_no_voice_time", 120)
        )
        if (
            not conn.close_after_chat
            and no_voice_time > 1000 * close_connection_no_voice_time
        ):
            conn.close_after_chat = True
            conn.client_abort = False
            end_prompt = conn.config.get("end_prompt", {})
            if end_prompt and end_prompt.get("enable", True) is False:
                conn.logger.bind(tag=TAG).info("结束对话，无需发送结束提示语")
                await conn.close()
                return
            # [xiaozhi-hub] 配置了固定告别语（end_prompt.reply）时直接说这句再断开，不发给大模型：
            # 接 Claude Code 时，"请你以时间过得真快开头……"这句假的用户消息会被永久写进对话记录
            farewell = end_prompt.get("reply")
            if farewell:
                conn.sentence_id = uuid.uuid4().hex
                speak_txt(conn, farewell)
                return
            prompt = end_prompt.get("prompt")
            if not prompt:
                prompt = "请你以```时间过得真快```未来头，用富有感情、依依不舍的话来结束这场对话吧。！"
            await startToChat(conn, prompt)


async def max_out_size(conn: "ConnectionHandler"):
    # 播放超出最大输出字数的提示
    conn.client_abort = False
    text = "不好意思，我现在有点事情要忙，明天这个时候我们再聊，约好了哦！明天不见不散，拜拜！"
    await send_stt_message(conn, text)
    file_path = "config/assets/max_output_size.wav"
    opus_packets = await audio_to_data(file_path)
    conn.tts.tts_audio_queue.put((SentenceType.LAST, opus_packets, text))
    conn.close_after_chat = True


async def check_bind_device(conn: "ConnectionHandler"):
    if conn.bind_code:
        # 确保bind_code是6位数字
        if len(conn.bind_code) != 6:
            conn.logger.bind(tag=TAG).error(f"无效的绑定码格式: {conn.bind_code}")
            text = "绑定码格式错误，请检查配置。"
            await send_stt_message(conn, text)
            return

        text = f"请登录控制面板，输入{conn.bind_code}，绑定设备。"
        await send_stt_message(conn, text)

        # 播放提示音
        music_path = "config/assets/bind_code.wav"
        opus_packets = await audio_to_data(music_path)
        conn.tts.tts_audio_queue.put((SentenceType.FIRST, opus_packets, text))

        # 逐个播放数字
        for i in range(6):  # 确保只播放6位数字
            try:
                digit = conn.bind_code[i]
                num_path = f"config/assets/bind_code/{digit}.wav"
                num_packets = await audio_to_data(num_path)
                conn.tts.tts_audio_queue.put((SentenceType.MIDDLE, num_packets, None))
            except Exception as e:
                conn.logger.bind(tag=TAG).error(f"播放数字音频失败: {e}")
                continue
        conn.tts.tts_audio_queue.put((SentenceType.LAST, [], None))
    else:
        # 播放未绑定提示
        conn.client_abort = False
        text = f"没有找到该设备的版本信息，请正确配置 OTA地址，然后重新编译固件。"
        await send_stt_message(conn, text)
        music_path = "config/assets/bind_not_found.wav"
        opus_packets = await audio_to_data(music_path)
        conn.tts.tts_audio_queue.put((SentenceType.LAST, opus_packets, text))
