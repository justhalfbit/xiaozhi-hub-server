"""管理后台用到的固定清单：大模型厂商预设、语音识别选项、EdgeTTS 中文音色、插件说明、Claude CLI 选项。"""

# ---------- 大模型：API 厂商（统一走 OpenAI 兼容协议） ----------
# entry：保存到配置里的条目名。已有的上游条目直接复用（保留用户填过的 Key），其余是新条目。
# models：模型名的候选，第一个是默认值；页面上仍可手填其他模型。
LLM_VENDORS = [
    {"id": "claude", "label": "Claude", "entry": "ClaudeAPI",
     "base_url": "https://api.anthropic.com/v1/",
     "models": ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]},
    {"id": "openai", "label": "OpenAI", "entry": "OpenAILLM",
     "base_url": "https://api.openai.com/v1",
     "models": ["gpt-5-mini", "gpt-5"]},
    {"id": "gemini", "label": "Gemini", "entry": "GeminiAPI",
     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
     "models": ["gemini-2.5-flash", "gemini-2.5-pro"]},
    {"id": "deepseek", "label": "DeepSeek", "entry": "DeepSeekLLM",
     "base_url": "https://api.deepseek.com",
     "models": ["deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner"]},
    {"id": "qwen", "label": "通义千问", "entry": "AliLLM",
     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "models": ["qwen-flash", "qwen-plus", "qwen-max"]},
    {"id": "doubao", "label": "豆包", "entry": "DoubaoLLM",
     "base_url": "https://ark.cn-beijing.volces.com/api/v3",
     "models": ["doubao-seed-2-0-lite-260215"]},
    {"id": "glm", "label": "智谱 GLM", "entry": "ChatGLMLLM",
     "base_url": "https://open.bigmodel.cn/api/paas/v4/",
     "models": ["glm-4-flash", "glm-4-plus"]},
    {"id": "kimi", "label": "Kimi", "entry": "KimiLLM",
     "base_url": "https://api.moonshot.cn/v1",
     "models": ["kimi-latest", "moonshot-v1-8k"]},
    {"id": "custom", "label": "自定义", "entry": "CustomOpenAILLM",
     "base_url": "", "models": [],
     "hint": "其他 OpenAI 兼容服务，例如本机代理、LM Studio（http://localhost:1234/v1）、Ollama（http://localhost:11434/v1）"},
]
VENDOR_BY_ID = {v["id"]: v for v in LLM_VENDORS}
VENDOR_BY_ENTRY = {v["entry"]: v for v in LLM_VENDORS}

# ---------- 大模型：Claude Code CLI ----------
CLAUDE_MODELS = [
    {"value": "haiku", "label": "Haiku", "hint": "最快，适合语音"},
    {"value": "sonnet", "label": "Sonnet", "hint": "更聪明"},
    {"value": "opus", "label": "Opus", "hint": "很强，较慢"},
    {"value": "fable", "label": "Fable", "hint": "最强，最慢"},
]
CLAUDE_EFFORTS = [
    {"value": "low", "label": "低"},
    {"value": "medium", "label": "中"},
    {"value": "high", "label": "高"},
    {"value": "xhigh", "label": "超高", "slow": True},
    {"value": "max", "label": "最高", "slow": True},
]
# 权限级别：决定开放哪些工具，以及 Claude Code 的权限模式
CLAUDE_READ_TOOLS = ["Read", "Grep", "Glob"]
CLAUDE_WEB_TOOLS = ["WebSearch", "WebFetch"]
CLAUDE_LEVELS = [
    {"value": "read", "label": "只读", "mode": "default", "tools": CLAUDE_READ_TOOLS,
     "desc": "只能查看、搜索工作目录里的文件"},
    {"value": "edit", "label": "可编辑", "mode": "acceptEdits", "tools": CLAUDE_READ_TOOLS + ["Write", "Edit"],
     "desc": "还能在工作目录里新建、修改文件，自动通过"},
    {"value": "auto", "label": "Auto", "mode": "auto",
     "tools": CLAUDE_READ_TOOLS + ["Write", "Edit", "Bash", "TodoWrite"],
     "desc": "还能执行命令等，由 Claude Code 自己判断：低风险的自动放行，需要确认的一律不执行"},
]
CLAUDE_LEVEL_BY_VALUE = {lv["value"]: lv for lv in CLAUDE_LEVELS}
CLAUDE_TOOLS = [
    {"name": "Read", "label": "读文件", "risky": False},
    {"name": "Grep", "label": "搜内容", "risky": False},
    {"name": "Glob", "label": "找文件", "risky": False},
    {"name": "WebSearch", "label": "联网搜索", "risky": False},
    {"name": "WebFetch", "label": "打开网页", "risky": False},
    {"name": "Bash", "label": "执行命令", "risky": True},
    {"name": "Write", "label": "写文件", "risky": True},
    {"name": "Edit", "label": "改文件", "risky": True},
    {"name": "TodoWrite", "label": "记待办", "risky": False},
]

# ---------- 语音识别：后台可以选的两种 ----------
# 本地只开放默认的 X-ASR（安装脚本已下载）；其他本地模型选中时会自动下载，不放进后台
LOCAL_ASR_ENTRY = "SherpaXASR"
DOUBAO_ASR_ENTRY = "DoubaoStreamASRV2"
DOUBAO_RESOURCES = [
    {"value": "volc.seedasr.sauc.duration", "label": "2.0 小时版", "hint": "按使用时长计费"},
    {"value": "volc.seedasr.sauc.concurrent", "label": "2.0 并发版", "hint": "按并发路数计费"},
    {"value": "volc.bigasr.sauc.duration", "label": "1.0 小时版", "hint": "上一代模型"},
    {"value": "volc.bigasr.sauc.concurrent", "label": "1.0 并发版", "hint": "上一代模型"},
]
DOUBAO_RESOURCE_VALUES = {r["value"] for r in DOUBAO_RESOURCES}

# ---------- 语音合成：EdgeTTS 中文音色 ----------
EDGE_VOICES = [
    {"value": "zh-CN-XiaoxiaoNeural", "name": "晓晓", "gender": "女", "style": "温暖", "group": "普通话"},
    {"value": "zh-CN-XiaoyiNeural", "name": "晓伊", "gender": "女", "style": "活泼", "group": "普通话"},
    {"value": "zh-CN-YunxiNeural", "name": "云希", "gender": "男", "style": "活泼、阳光", "group": "普通话"},
    {"value": "zh-CN-YunxiaNeural", "name": "云夏", "gender": "男", "style": "可爱，偏童声", "group": "普通话"},
    {"value": "zh-CN-YunjianNeural", "name": "云健", "gender": "男", "style": "激情，偏解说", "group": "普通话"},
    {"value": "zh-CN-YunyangNeural", "name": "云扬", "gender": "男", "style": "专业稳重，偏播报", "group": "普通话"},
    {"value": "zh-CN-liaoning-XiaobeiNeural", "name": "晓北", "gender": "女", "style": "幽默", "group": "方言：东北话"},
    {"value": "zh-CN-shaanxi-XiaoniNeural", "name": "晓妮", "gender": "女", "style": "明亮", "group": "方言：陕西话"},
    {"value": "zh-HK-HiuGaaiNeural", "name": "曉佳", "gender": "女", "style": "友好", "group": "粤语"},
    {"value": "zh-HK-HiuMaanNeural", "name": "曉曼", "gender": "女", "style": "友好", "group": "粤语"},
    {"value": "zh-HK-WanLungNeural", "name": "雲龍", "gender": "男", "style": "友好", "group": "粤语"},
    {"value": "zh-TW-HsiaoChenNeural", "name": "曉臻", "gender": "女", "style": "友好", "group": "台湾"},
    {"value": "zh-TW-HsiaoYuNeural", "name": "曉雨", "gender": "女", "style": "友好", "group": "台湾"},
    {"value": "zh-TW-YunJheNeural", "name": "雲哲", "gender": "男", "style": "友好", "group": "台湾"},
]
EDGE_VOICE_VALUES = {v["value"] for v in EDGE_VOICES}
VOICE_SAMPLE = {
    "普通话": "你好，今天想聊点什么？",
    "粤语": "你好，今日想傾啲乜嘢？",
}


def voice_sample(group: str) -> str:
    return VOICE_SAMPLE.get(group, VOICE_SAMPLE["普通话"])


# ---------- 意图识别：插件 ----------
# 原项目总会加载"退出对话"和"查日期"，不需要勾选
ALWAYS_ON_PLUGINS = [
    {"name": "handle_exit_intent", "label": "退出对话", "desc": "说“再见”“退下吧”时结束对话"},
    {"name": "get_lunar", "label": "查日期", "desc": "今天几号、星期几、农历和节气"},
]
PLUGINS = [
    {"name": "get_weather", "label": "查天气", "desc": "实时天气和预报（使用项目共用的天气 Key，用多了可能被限制）"},
    {"name": "get_news_from_newsnow", "label": "热点新闻", "desc": "澎湃新闻、百度热搜、财联社等热榜"},
    {"name": "get_news_from_chinanews", "label": "中新网新闻", "desc": "中国新闻网的社会、国际、财经新闻"},
    {"name": "play_music", "label": "放音乐", "desc": "播放电脑上 music 目录里的歌曲"},
    {"name": "web_search", "label": "联网搜索", "desc": "用 Anthropic 协议接口自带的搜索查实时信息，点「设置」填接口和 Key",
     "config": "web_search", "settings": "web_search"},
    {"name": "change_role", "label": "切换角色", "desc": "换人设和说话风格"},
    {"name": "hass_get_state", "label": "查智能家居状态", "desc": "需要在配置文件的 plugins.home_assistant 里填写地址和令牌",
     "config": "home_assistant"},
    {"name": "hass_set_state", "label": "控制智能家居", "desc": "需要在配置文件的 plugins.home_assistant 里填写地址和令牌",
     "config": "home_assistant"},
    {"name": "hass_play_music", "label": "智能音箱放歌", "desc": "需要在配置文件的 plugins.home_assistant 里填写地址和令牌",
     "config": "home_assistant"},
    {"name": "search_from_ragflow", "label": "查知识库", "desc": "需要在配置文件的 plugins.search_from_ragflow 里填写 RAGFlow 地址和 Key",
     "config": "search_from_ragflow"},
]
PLUGIN_NAMES = {p["name"] for p in PLUGINS}

# 插件设置（后台插件旁边的「设置」）：按字段清单自动生成表单，值存在 plugins.<config> 下。
# PLUGINS 条目加 "settings": <id> 指向这里。每份设置 {"note": 整体说明, "fields": [...]}，字段：
#   key、label；type：text / secret（只写不读，留空表示不改）/ number（可带 min、max）；
#   required：没填时插件显示"还没配置"（不写就是选填）；default：留空时实际用的值，页面上当占位提示；
#   hint：字段旁边的说明文字。
# legacy_keys：旧版插件才有的字段，配置里还有它们时，已保存的密钥当成没填（是给别的服务的），
#   保存时删掉这些字段并清掉旧密钥；legacy_note 是页面上的提示。
# 要有「测试」按钮，在 core/admin/plugin_settings.py 的 TESTS 里登记测试函数。
PLUGIN_SETTINGS = {
    "web_search": {
        "note": "需要 Anthropic 协议、支持自带搜索（web_search）的接口，比如 DeepSeek、Anthropic 官方。"
                "聊天模型用哪家都可以，和这里无关。注意：网关装在设备上时，127.0.0.1 指设备自己。",
        "fields": [
            {"key": "base_url", "label": "接口地址", "type": "text", "default": "https://api.deepseek.com/anthropic/v1",
             "hint": "写到 /anthropic 或 /anthropic/v1 都可以；Anthropic 官方填 https://api.anthropic.com"},
            {"key": "api_key", "label": "API Key", "type": "secret", "required": True},
            {"key": "model", "label": "模型", "type": "text", "default": "deepseek-v4-flash",
             "hint": "用来搜索和总结的模型，比如 deepseek-v4-flash、claude-haiku-5-5"},
            {"key": "max_uses", "label": "单次最多搜索次数", "type": "number", "default": 5, "min": 1, "max": 10,
             "hint": "一个问题里模型最多搜几轮；越多越全，也越慢"},
        ],
        # 旧版联网搜索（秘塔、Tavily、Serply）的字段：那时的 Key 不能发给新接口
        "legacy_keys": ["provider", "max_results"],
        "legacy_note": "配置里还留着旧版联网搜索（秘塔、Tavily、Serply）的设置，那时填的 Key 不能用在新接口上，"
                       "新版不会使用它。请重新填写 API Key 并保存，旧设置会一起清掉。",
    },
}
INTENT_MODES = ("nointent", "function_call")

# ---------- 情绪（表情测试） ----------
EMOTION_LABELS = {
    "neutral": "平静", "happy": "开心", "laughing": "大笑", "funny": "搞笑", "sad": "难过", "angry": "生气",
    "crying": "哭泣", "loving": "喜爱", "embarrassed": "尴尬", "surprised": "惊讶", "shocked": "震惊",
    "thinking": "思考", "winking": "眨眼", "cool": "酷", "relaxed": "放松", "delicious": "好吃",
    "kissy": "亲亲", "confident": "自信", "sleepy": "困", "silly": "傻乐", "confused": "困惑",
}
