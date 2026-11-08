"""管理后台"工作原理"页：一句话经过的五步、"怎么跟它说话"速查，以及发给大模型的提示词。

面向正在用设备的人，只讲"说了什么会怎样"；为什么这样设计、每一步的实现细节在 hub-docs/design.md。
只读：不修改配置，也不影响运行中的服务。页面上的值都从配置读取，切换模块后跟着变。
提示词按原项目的拼装方式（core/utils/prompt_manager.py、core/utils/dialogue.py）重新渲染一份预览，
模板变量用标记包起来，前端据此高亮显示"这里填进了什么"。
"""

import os
from datetime import datetime

from jinja2 import Template

from config.config_loader import get_project_dir
from core.admin import catalog, connections, pages

# 变量标记：\x01变量名\x02值\x03，前端换成高亮
MARK_START, MARK_SEP, MARK_END = "\x01", "\x02", "\x03"

VAR_LABELS = {
    "base_prompt": "人设",
    "current_time": "当前时间",
    "today_date": "日期",
    "today_weekday": "星期",
    "lunar_date": "农历",
    "emojiList": "表情列表",
    "dynamic_context": "动态上下文",
    "language": "回复语言",
}

def _fewshot_text(tool_names=None, weather_in_context=False) -> str:
    """和 connection._inject_tool_call_fewshot 注入的示例一致，按 dialogue.render_fewshot 写进系统提示的样子。
    tool_names：勾选的插件加上总会加载的（catalog.ALWAYS_ON_PLUGINS）；后台看不到在线设备的工具，示例只和网关这边的有关。
    weather_in_context：模板里有没有本地天气（weather_info）。"""
    from core.utils.dialogue import fewshot_messages, render_fewshot

    tool_names = set(tool_names or ()) | {p["name"] for p in catalog.ALWAYS_ON_PLUGINS}
    return render_fewshot(fewshot_messages(tool_names, weather_in_context))


def _mark(name: str, value) -> str:
    return f"{MARK_START}{name}{MARK_SEP}{value}{MARK_END}"


def _entry(merged: dict, section: str) -> tuple:
    name = (merged.get("selected_module") or {}).get(section)
    entry = (merged.get(section) or {}).get(name) or {}
    return name, entry if isinstance(entry, dict) else {}


def _online_conn():
    """任意一台在线设备的连接（用来取它实际使用的提示词）。"""
    for item in connections.snapshot():
        conn = connections.get(item["session_id"])
        if conn is not None:
            return conn
    return None


def _system_prompt_of(conn) -> str:
    dialogue = getattr(getattr(conn, "dialogue", None), "dialogue", None) or []
    for msg in dialogue:
        if getattr(msg, "role", None) == "system":
            return str(getattr(msg, "content", "") or "")
    return ""


# ---------- 流程 ----------

DESIGN_DOC_URL = "https://github.com/justhalfbit/xiaozhi-hub-server/blob/main/hub-docs/design.md"

def _voice_name(voice: str) -> str:
    info = next((v for v in catalog.EDGE_VOICES if v["value"] == voice), None)
    return f"{info['name']}（{info['group']}）" if info else voice


def _llm_label(name: str, entry: dict) -> str:
    if entry.get("type") == "claude_cli":
        return "Claude Code"
    vendor = catalog.VENDOR_BY_ENTRY.get(name)
    return vendor["label"] if vendor else name


def _asr_label(name: str, entry: dict) -> str:
    if entry.get("type") == "doubao_stream":
        return "豆包流式识别"
    if entry.get("type") == "sherpa_onnx_local":
        if "x-asr" in str(entry.get("model_dir") or "").lower():
            return "本地 X-ASR"
        model_type = entry.get("model_type") or "sense_voice"
        return "本地 " + pages.SHERPA_MODEL_LABELS.get(model_type, model_type)
    return name or "未设置"


def _claude_model_label(model) -> str:
    option = next((o for o in catalog.CLAUDE_MODELS if o["value"] == model), None)
    return option["label"] if option else (str(model) if model else "默认模型")


def _effort_label(effort) -> str:
    option = next((o for o in catalog.CLAUDE_EFFORTS if o["value"] == effort), None)
    return option["label"] if option else "默认"


def describe_stages(merged: dict) -> list:
    """一句话经过的五步：每步只给当前用的模块和一个关键设置，点击跳到对应的设置页（page 为 None 表示后台没有设置页）。"""
    _, vad = _entry(merged, "VAD")
    asr_name, asr = _entry(merged, "ASR")
    llm_name, llm = _entry(merged, "LLM")
    tts_name, tts = _entry(merged, "TTS")
    silence = vad.get("min_silence_duration_ms", 1000)
    prebuffer = int(merged.get("tts_audio_prebuffer_count") or 5)

    if llm.get("type") == "claude_cli":
        llm_value = f"Claude Code · {_claude_model_label(llm.get('model'))}"
        llm_hint = f"思考力度：{_effort_label(llm.get('effort'))}"
    else:
        llm_value = _llm_label(llm_name, llm)
        llm_hint = str(llm.get("model_name") or "")
    if tts.get("type") == "edge":
        tts_value, tts_hint = f"EdgeTTS · {_voice_name(str(tts.get('voice') or ''))}", "微软在线语音，需要联网"
    else:
        tts_value, tts_hint = tts_name or "未设置", str(tts.get("type") or "")
    local_asr = asr.get("type") == "sherpa_onnx_local"
    return [
        {"id": "listen", "icon": "🎙", "title": "收音", "value": f"停顿 {silence}ms 算说完",
         "hint": "说话中间停顿太久会被截成两句", "page": "asr"},
        {"id": "asr", "icon": "📝", "title": "识别", "value": _asr_label(asr_name, asr),
         "hint": "在本机识别，不联网" if local_asr else "云端识别，需要联网", "page": "asr"},
        {"id": "llm", "icon": "✦", "title": "大模型", "value": llm_value, "hint": llm_hint, "page": "llm"},
        {"id": "tts", "icon": "🔊", "title": "合成", "value": tts_value, "hint": tts_hint, "page": "tts"},
        {"id": "play", "icon": "📢", "title": "播放", "value": f"缓冲约 {prebuffer * 60 / 1000:.1f} 秒",
         "hint": "说话断续时，在配置文件里调大", "page": None},
    ]


def _duration(seconds) -> str:
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return str(seconds)
    if seconds >= 60 and seconds % 60 == 0:
        minutes = int(seconds // 60)
        return f"{minutes // 60} 小时" if minutes % 60 == 0 else f"{minutes} 分钟"
    return f"{int(seconds)} 秒"


def _setting(entry: dict, key: str, default: float) -> float:
    """和 claude_cli 读配置的规则一样：没填（None 或空字符串）用默认值，填 0 就是 0。"""
    from core.providers.llm.claude_cli.claude_cli import _num

    return _num(entry, key, default)


def _quoted(words, sep: str = "") -> str:
    return sep.join(f"“{w}”" for w in words)


def describe_tips(merged: dict) -> list:
    """"你这样说 → 会这样"的速查卡片，值都从当前配置读取。"""
    calls = pages.describe_call_names(merged)
    llms = merged.get("LLM") or {}
    default_name = (merged.get("selected_module") or {}).get("LLM")
    called = [m["entry"] for m in calls["models"]] if calls["enabled"] else []
    names = [default_name, *called]
    cli = next((llms[n] for n in names if isinstance(llms.get(n), dict) and llms[n].get("type") == "claude_cli"), None)
    has_api = any(isinstance(llms.get(n), dict) and llms[n].get("type") != "claude_cli" for n in names)

    tips = [{"title": "唤醒", "say": "说唤醒词（StackChan 是“Hi, StackChan”），或按一下按键、点一下屏幕",
             "result": "设备开始听你说话。交给这台设备上次用的模型，第一次用时是 "
                       f"{_llm_label(default_name, llms.get(default_name) or {})}"}]
    if calls["enabled"]:
        aliases = "；".join(f"{m['label']}：{'、'.join(m['aliases'])}" for m in calls["models"])
        tips.append({"title": "换模型", "say": "句首说“嗨 + 称呼”，后面可以直接接内容",
                     "result": f"换成对应的模型回答，之后一直用它；只说称呼时回一声“{calls['reply']}”。"
                               + (f"现在的称呼 —— {aliases}" if aliases else "还没有配置称呼")})
    tips.append({"title": "打断", "say": "设备说话时按一下按键（StackChan 是点一下屏幕上的脸），或者喊唤醒词",
                 "result": "马上停下，然后可以说下一句。设备说话时不收音，不能直接插话"})
    if cli:
        progress = _setting(cli, "progress_interval_seconds", 30)
        turn = _setting(cli, "turn_timeout_seconds", 600)
        tips.append({"title": "长任务", "say": "让 Claude 做要几分钟的事",
                     "result": ("设备陪着等，不限时长" if not turn else f"设备陪着等，最长 {_duration(turn)}")
                               + (f"；{_duration(progress)}没开口就提示一次进度" if progress else "")
                               + ("。中途打断、结束或等满了，Claude 不停，后台接着做，做完了告诉你；想让它停下就说“停下”"
                                  if _setting(cli, "background_max_seconds", 7200) else "。被停下或打断后，说“继续”接着做")})
        window = _setting(cli, "resume_window_minutes", 120)
        keywords = cli.get("resume_keywords") or ["接着上次聊"]
        said = _quoted(keywords[:2], "或")
        result = (f"{_duration(window * 60)}内再唤醒，Claude 自动接上上次的对话；隔得更久的，说{said}也能接上"
                  if window else f"说{said}接上上次的对话")
        if has_api:
            result += "。API 模型断开后就不记得了"
        tips.append({"title": "接着上次聊", "say": "过一会儿再唤醒设备", "result": result})
    elif has_api:
        tips.append({"title": "接着上次聊", "say": "过一会儿再唤醒设备",
                     "result": "API 模型只记得这次连接里的对话，断开后从头开始"})

    first = int(merged.get("close_connection_no_voice_time", 120) or 120)
    end_prompt = merged.get("end_prompt") or {}
    if end_prompt.get("enable", True) is False:
        farewell = "直接断开"
    elif end_prompt.get("reply"):
        farewell = f"说一句“{end_prompt['reply']}”后断开"
    else:
        farewell = "说一句告别语后断开"
    exits = merged.get("exit_commands") or []
    tips.append({"title": "结束", "say": f"说{_quoted(exits)}" if exits else "不再说话",
                 "result": ("立即断开。" if exits else "") + f"{_duration(first)}没人说话时，{farewell}"})
    return tips


def describe_flow(merged: dict) -> dict:
    return {"stages": describe_stages(merged), "tips": describe_tips(merged)}


# ---------- 提示词拆解 ----------

def _template_name(merged: dict) -> str:
    return str(merged.get("prompt_template") or "hub-base-prompt.txt")


def _template_text(merged: dict) -> str:
    path = os.path.join(get_project_dir(), _template_name(merged))
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _context_values(merged: dict) -> dict:
    from core.utils.current_time import get_current_date, get_current_lunar_date, get_current_weekday
    from core.utils.prompt_manager import EMOJI_List

    _, tts = _entry(merged, "TTS")
    return {
        "current_time": datetime.now().strftime("%H:%M"),
        "today_date": get_current_date(),
        "today_weekday": get_current_weekday(),
        "lunar_date": get_current_lunar_date(),
        "emojiList": EMOJI_List,
        "dynamic_context": "",
        "language": tts.get("language") or "中文",
    }


def render_base_prompt(merged: dict) -> str:
    """按服务端的方式渲染基础模板，变量值用标记包起来。"""
    template = _template_text(merged)
    persona = str(merged.get("prompt") or "")
    if not template:
        return _mark("base_prompt", persona)
    values = _context_values(merged)
    marked = {k: _mark(k, v) for k, v in values.items()}
    return Template(template).render(base_prompt=_mark("base_prompt", persona), emoji_enabled=True,
                                     device_id="", client_ip="", **marked)


def describe_prompt(merged: dict) -> dict:
    llm_name, llm = _entry(merged, "LLM")
    _, intent = _entry(merged, "Intent")
    is_cli = llm.get("type") == "claude_cli"
    conn = _online_conn()
    base = render_base_prompt(merged)

    segments = []
    if is_cli:
        segments.append({
            "id": "claude_preset", "title": "Claude Code 自带的系统提示", "source": "Claude Code", "when": "进程启动时",
            "text": "", "collapsed": True,
            "note": "Claude Code 内置的说明，介绍它自己有哪些工具、工作目录在哪、怎么回答编程问题等。"
                    "这段由 Claude Code 程序自己生成，不在我们的配置里，这里无法显示原文。我们的内容追加在它后面。",
        })
    segments.append({
        "id": "base", "title": "基础模板（含人设）", "source": "服务端", "when":
            "设备连接时生成；Claude 进程启动时固定" if is_cli else "设备连接时生成；当前时间每轮更新",
        "text": base, "collapsed": False,
        "note": f"来自 {_template_name(merged)}。高亮的部分是填进去的值：人设来自配置里的 prompt，"
                "日期、农历在设备连接时生成，当前时间" +
                ("在 Claude 进程启动时取一次。" if is_cli else "每轮发送前更新。"),
    })
    if is_cli:
        from core.providers.llm.claude_cli.claude_cli import VOICE_RULES

        segments.append({
            "id": "voice_rules", "title": "语音规则", "source": "服务端", "when": "进程启动时",
            "text": VOICE_RULES.strip(), "collapsed": False,
            "note": "我们为 Claude Code 加的规则：开头一个 emoji、简短口语、不用 Markdown。写在 claude_cli.py 的 VOICE_RULES。",
        })
        segments.append({
            "id": "turn", "title": "每轮发送的内容", "source": "服务端", "when": "每轮",
            "text": "（这一轮识别出的文字，例如：帮我看看工作目录里有哪些文件）", "collapsed": False,
            "note": "只发这一轮的话，之前的对话保存在 Claude Code 会话里。第一次识别到说话人时，会发 "
                    "{\"speaker\": \"名字\", \"content\": \"文字\"} 这样的 JSON（声纹识别目前没开）。",
        })
    else:
        if intent.get("type") == "function_call":
            segments.append({
                "id": "fewshot", "title": "工具调用示例", "source": "服务端", "when": "每轮",
                "text": _fewshot_text(intent.get("functions") or [], "weather_info" in _template_text(merged)),
                "collapsed": True,
                "note": "追加在系统提示词末尾的一段示例，教模型什么时候直接回答、什么时候调用工具。不会显示给用户。"
                        "原项目把它们当成几轮对话发，模型会当成用户真说过（问「我刚才让你做过什么」答「讲故事、说拜拜」），"
                        "所以改成写在系统提示里并注明不是真实对话。",
            })
            plugins = [p["label"] for p in catalog.PLUGINS if p["name"] in (intent.get("functions") or [])]
            segments.append({
                "id": "tools", "title": "工具清单", "source": "服务端", "when": "每轮",
                "text": "、".join(["直接回答（direct_answer）", *[p["label"] for p in catalog.ALWAYS_ON_PLUGINS], *plugins]),
                "collapsed": False,
                "note": "随请求一起发给模型的工具（OpenAI 协议的 tools 参数），每个工具带名字、用途和参数说明。",
            })
        segments.append({
            "id": "history", "title": "历史对话 + 这一轮的话", "source": "服务端", "when": "每轮",
            "text": "[user] 上一轮说的话\n[assistant] 上一轮的回复\n……\n[user] 这一轮识别出的文字", "collapsed": False,
            "note": "每轮都把这次连接以来的全部对话一起发送，没有截断；设备断开后清空（没有开启记忆）。",
        })

    warnings = []
    if is_cli:
        idle = _setting(llm, "idle_timeout_seconds", 300)
        if idle:  # 填 0 时每轮结束就关进程，下一轮重新传入，日期时间总是新的
            warnings.append(f"提示词在 Claude Code 进程启动时一次性传入，进程存活期间（空闲 {_duration(idle)}才关闭）"
                            "里面的日期、时间不会更新。")

    actual = _system_prompt_of(conn) if conn is not None else ""
    return {
        "llm": _llm_label(llm_name, llm),
        "segments": segments,
        "warnings": warnings,
        "actual": {"device_id": getattr(conn, "device_id", None), "text": actual} if actual else None,
        "marks": {"start": MARK_START, "sep": MARK_SEP, "end": MARK_END},
        "var_labels": VAR_LABELS,
    }


def describe(merged: dict) -> dict:
    return {"flow": describe_flow(merged), "prompt": describe_prompt(merged), "design_url": DESIGN_DOC_URL}
