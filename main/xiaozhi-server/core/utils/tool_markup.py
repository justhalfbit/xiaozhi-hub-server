"""[xiaozhi-hub] 过滤模型漏进回复文字里的工具调用标记，不让它被念出来、显示在聊天页、写进对话。

DeepSeek（V3.2 起）的工具调用在模型内部是一段 DSML 标记：
    <｜DSML｜function_calls>
    <｜DSML｜invoke name="direct_answer">
    <｜DSML｜parameter name="response" string="true">回复的文字</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
接口按标准写法把它解析成 tool_calls。模型偶尔会生成变形的标记（2026-11-01 遇到的是
「</｜｜DSML｜｜ parameter>」「</｜｜DSML｜｜ calls>」：竖线成对、名字前有空格），接口认不出，
就当成参数里的文字原样交出来；网关把 direct_answer 的 response 参数当回复念，于是念了出来。
更早的 DeepSeek 用 <｜tool▁calls▁begin｜> 这类特殊记号，其他模型也有 <|im_end|> 之类，一并处理。

规则：
- 标记本身一律去掉。
- 开始标记（DSML 的非闭合标签、tool▁calls▁begin / tool▁call▁begin）到结束标记（DSML 的 …calls
  闭合标签、tool▁calls▁end / tool▁call▁end）之间是调用参数，也去掉；有开始没结束就丢到这轮结束。
- 只认「尖括号 + 竖线」包着的记号（竖线全角半角都算），正常文字不会这样写。
流式输入时标记可能被切在两段之间：末尾像标记开头的一小段先扣住，等下一段再判断。
"""

import re

_BAR = r"[|｜]"
_MARKER = re.compile(
    rf"<\s*(?P<close>/)?\s*{_BAR}+\s*DSML\s*{_BAR}+\s*(?P<tag>[A-Za-z_]*)[^<>]*>"
    rf"|<{_BAR}(?P<token>[^<>|｜\s]{{1,48}}){_BAR}>"
)
_TRAILING_BREAK = re.compile(r"[ \t]*\r?\n?")  # 标记独占一行时连同换行一起去掉
# 可能还没收全的标记：「<」「</」，或者「<」后面跟着竖线、还没等到「>」
_PARTIAL = re.compile(rf"<\s*/?\s*(?:{_BAR}[^<>]*)?")
_PARTIAL_WITH_BAR = re.compile(rf"<\s*/?\s*{_BAR}[^<>]*")
_MAX_MARKER = 160  # 比这长还没等到「>」就不是标记


def _kind(m):
    """标记的作用：'begin' 进入调用参数，'end' 离开，'strip' 只去掉自己。"""
    if m.group("token") is not None:
        name = m.group("token").replace("▁", "_").lower()
        if "call" in name and "begin" in name:
            return "begin"
        if "call" in name and "end" in name:
            return "end"
        return "strip"
    if not m.group("close"):
        return "begin"
    return "end" if "calls" in m.group("tag").lower() else "strip"


class ToolMarkupFilter:
    """流式过滤：一轮回复一个实例，feed 每段文字，最后 flush。"""

    def __init__(self):
        self._buf = ""
        self._in_call = False
        self._skip_break = False  # 上一个标记后面的换行还没等到
        self.removed = 0  # 去掉的标记个数，用于记日志

    def feed(self, text):
        if not text:
            return ""
        buf = self._buf + text
        out = []
        pos = self._eat_break(buf, 0) if self._skip_break else 0
        while True:
            m = _MARKER.search(buf, pos)
            if not m:
                break
            if not self._in_call:
                out.append(buf[pos:m.start()])
            self.removed += 1
            kind = _kind(m)
            if kind == "begin":
                self._in_call = True
            elif kind == "end":
                self._in_call = False
            pos = self._eat_break(buf, m.end())
        rest = buf[pos:]
        self._buf = ""
        i = rest.rfind("<")
        if i >= 0 and len(rest) - i <= _MAX_MARKER and _PARTIAL.fullmatch(rest, i):
            rest, self._buf = rest[:i], rest[i:]
        if not self._in_call:
            out.append(rest)
        return "".join(out)

    def _eat_break(self, buf, pos):
        end = _TRAILING_BREAK.match(buf, pos).end()
        self._skip_break = end == len(buf) and not buf[pos:end].endswith("\n")
        return end

    def flush(self):
        rest, self._buf = self._buf, ""
        if self._in_call or _PARTIAL_WITH_BAR.fullmatch(rest):
            return ""  # 调用参数，或者没收全的标记
        return rest


def strip_tool_markup(text):
    """一次性过滤整段文字。"""
    f = ToolMarkupFilter()
    return f.feed(text) + f.flush()
