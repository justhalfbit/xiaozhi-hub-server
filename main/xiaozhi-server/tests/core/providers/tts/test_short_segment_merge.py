"""[xiaozhi-hub] 少于 MIN_SEGMENT_CHARS 个字的段并到下一段，不单独合成（每段合成都有固定开销，段越碎停顿越多）。"""
import pytest

from core.providers.tts.base import TTSProviderBase


class _TTS(TTSProviderBase):
    async def text_to_speak(self, text, output_file):
        return None


def _segments(text, step, first_sentence=True, stop=True):
    tts = object.__new__(_TTS)
    tts.punctuations = ("。", "？", "?", "！", "!", "；", ";", "：")
    tts.first_sentence_punctuations = ("，", "~", "、", ",") + tts.punctuations
    tts.tts_text_buff, tts.processed_chars = [], 0
    tts.is_first_sentence, tts.tts_stop_request = first_sentence, False
    out = []
    for i in range(0, len(text), step):
        tts.tts_text_buff.append(text[i:i + step])
        segment = tts._get_segment_text()
        if segment:
            out.append(segment)
    if stop:
        tts.tts_stop_request = True
        rest = tts._get_segment_text()
        if rest:
            out.append(rest)
    return out


@pytest.mark.parametrize("step", [1, 3, 50])
def test_short_first_clause_joins_the_next(step):
    # 原来首句遇逗号就切，"好"单独合成一段
    assert _segments("好，我来给你讲五个月球冷知识。月球一直只把同一面朝向地球。", step) == [
        "好，我来给你讲五个月球冷知识",
        "月球一直只把同一面朝向地球",
    ]


@pytest.mark.parametrize("step", [1, 4])
def test_short_sentence_joins_the_next(step):
    assert _segments("嗯。这个问题其实不难。对！答案就是四十二。", step, first_sentence=False) == [
        "嗯。这个问题其实不难",
        "对！答案就是四十二",
    ]


def test_long_enough_segment_is_cut_at_once():
    # 提示语（填充语）不能等后面的文字：够长就马上切出来
    assert _segments("😶交给 Grok Bot 了。", 1, stop=False) == ["交给 Grok Bot 了"]
    assert _segments("还在处理，请稍等。", 1, first_sentence=False, stop=False) == ["还在处理，请稍等"]


def test_short_text_waits_and_is_flushed_at_the_end():
    assert _segments("好的。", 1, stop=False) == []
    assert _segments("好的。", 1) == ["好的。"]
