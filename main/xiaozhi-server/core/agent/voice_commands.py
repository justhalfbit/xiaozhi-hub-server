"""语音控制指令的关键字匹配。"""

import re

DEFAULT_RESUME_KEYWORDS = ["接着上次聊", "继续上次的话题", "继续刚才的话题", "接着刚才聊"]

_STRIP = re.compile(r"[\s，。！？、,.!?；;：:~～\"'“”‘’…]+")


def normalize(text: str) -> str:
    return _STRIP.sub("", text or "")


NEGATIONS = ("不要", "不用", "别", "不想", "不必", "甭")


# 关键字至少这么长时，允许错 1 个字：语音识别常把个别字认错（实测"接着上次聊"被识别成"接着上次量"）
FUZZY_MIN_LEN = 5
# 只在整句话基本就是这条指令时才允许错字：命中接续指令后这句话不会发给 Claude，
# 长句里相近的片段（如"接着上次买的那个东西怎么用"）多半是真问题，误判会把问题吞掉
_LEADING_FILLERS = re.compile(r"^(?:嗯|啊|呃|哦|喂|那|好的|好|那么|我们|咱们|帮我|请|麻烦)+")
_TRAILING_FILLERS = re.compile(r"(?:吧|啊|呀|呢|啦|了|嘛|吗|好吗|行吗|可以吗)+$")


def _negated(t: str, start: int) -> bool:
    return any(t[:start].endswith(n) for n in NEGATIONS)


def _remainder(t: str, start: int, end: int) -> str:
    """关键字以外的内容（去掉首尾语气词）。只剩 1 个字以内时视为没有，比如"接着上次聊嘛"。"""
    rest = _LEADING_FILLERS.sub("", t[:start]) + _TRAILING_FILLERS.sub("", t[end:])
    return rest if len(rest) > 1 else ""


def match_resume(text: str, keywords=None):
    """判断是不是接续指令。不是时返回 None；是时返回关键字以外的内容，空字符串表示整句话就是这条指令。

    去掉空白和标点后，包含任一关键字就算命中；关键字前面紧跟否定词（如"别接着上次聊"）则不算。
    容忍语音识别的错字：5 个字及以上的关键字，在整句话（去掉首尾语气词后）最多比关键字多 1 个字时，
    允许有 1 个字不同（同样长度，不容忍多字、少字）。
    """
    t = normalize(text)
    keys = [k for k in (normalize(k) for k in (keywords or DEFAULT_RESUME_KEYWORDS)) if k]
    for k in keys:
        start = t.find(k)
        while start != -1:
            if not _negated(t, start):
                return _remainder(t, start, start + len(k))
            start = t.find(k, start + 1)
    core = _TRAILING_FILLERS.sub("", _LEADING_FILLERS.sub("", t))
    for k in keys:
        if len(k) < FUZZY_MIN_LEN or len(core) > len(k) + 1:
            continue
        for start in range(len(core) - len(k) + 1):
            window = core[start:start + len(k)]
            if sum(a != b for a, b in zip(window, k)) <= 1 and not _negated(core, start):
                return _remainder(core, start, start + len(k))
    return None


# 让后台在做的事停下：整句话基本就是这条指令时才算（"别停下"这类带否定的不算）
STOP_WORDS = ("停下", "停下来", "停止", "停", "别做了", "不用做了", "不做了", "不要做了", "取消", "算了", "停掉")
_STOP_PREFIX = re.compile(r"^(?:让它|叫它|你|给我|先|赶紧|快)+")


def match_stop(text: str) -> bool:
    core = _STOP_PREFIX.sub("", _LEADING_FILLERS.sub("", normalize(text)))
    # "算了""别做了"本身以语气词结尾：去掉结尾语气词前后各比一次
    return core in STOP_WORDS or _TRAILING_FILLERS.sub("", core) in STOP_WORDS
