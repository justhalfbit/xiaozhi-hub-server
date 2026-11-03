"""联网搜索：向 Anthropic 协议的接口发一次请求，带上服务端自带的搜索工具 web_search_20250305，
让那个模型去搜，拿回总结和搜索结果（标题、日期），再交给聊天模型回答。

和 DeepSeek Harness 的做法一致：不接专门的搜索服务，用支持自带搜索的 Anthropic 协议接口
（DeepSeek 的 https://api.deepseek.com/anthropic/v1、Anthropic 官方等）。聊天模型用哪家都可以，和这里无关。
"""

import time
from dataclasses import dataclass, field

import httpx
from config.logger import setup_logging
from core.providers.tts.dto.dto import ContentType
from plugins_func.register import (
    register_function,
    ToolType,
    ActionResponse,
    Action,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic/v1"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_USES = 5
MAX_USES_LIMIT = 10
# 语音场景用户在干等：连不上 5 秒就放弃，整次搜索（可能搜好几轮）最多 20 秒（实测一般 3 到 5 秒）
TIMEOUT = httpx.Timeout(20.0, connect=5.0)
WEEKDAYS = "一二三四五六日"
# 模型决定搜索时先播的一句，免得搜索那几秒没有声音
SEARCHING_REPLY = "我上网查一下。"  # 句号不能省：合成按标点切句，没有就会和后面的回答连成一句
RESULT_LIMIT = 5  # 交给聊天模型的来源条数上限
SNIPPET_LIMIT = 150  # 每条来源的原文片段最多多少字
ANTHROPIC_VERSION = "2023-06-01"

_DEFAULT_DESCRIPTION = (
    "联网搜索工具。用户问到实时信息、最近发生的事、新闻、价格、比分，或者你不确定、需要查证的事实时使用。"
)

WEB_SEARCH_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": _DEFAULT_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要搜索的问题或关键词，写完整，例如带上地点和时间",
                }
            },
            "required": ["query"],
        },
    },
}

_SEARCH_SYSTEM = (
    "你是联网搜索助手。必须先用 web_search 工具搜索，再根据搜索结果用中文总结要点，"
    "300 字以内，只写和问题直接相关的内容，保留关键的数字、日期和来源名称；"
    "用纯文本，不要用 Markdown 标题、列表符号或加粗（结果会交给语音助手念出来）；"
    "不要编造搜索结果里没有的内容。"
    "今天是 {today}。"
)

# 失败分两类：config 要用户去后台处理（设备要说出原因），temporary 是临时问题（只说这次没搜到）
CONFIG, TEMPORARY = "config", "temporary"


class SearchError(Exception):
    """搜索失败。message 是详细原因（后台测试按钮直接显示）；spoken 是设备上能说给用户听的简短原因。"""

    def __init__(self, message: str, kind: str = TEMPORARY, spoken: str = ""):
        super().__init__(message)
        self.kind = kind
        self.spoken = spoken or message


def today_text() -> str:
    return time.strftime("%Y-%m-%d") + " 星期" + WEEKDAYS[time.localtime().tm_wday]


@dataclass
class SearchResult:
    summary: str = ""
    results: list = field(default_factory=list)  # [{"title", "url", "page_age", "snippet"}]
    queries: list = field(default_factory=list)  # 模型实际搜的词
    seconds: float = 0.0


def messages_url(base_url: str) -> str:
    """接口地址补成 .../v1/messages：/anthropic、/anthropic/v1、.../v1/messages 三种写法都认。"""
    url = str(base_url or "").strip().rstrip("/")
    if url.endswith("/messages"):
        return url
    if url.endswith("/v1"):
        return url + "/messages"
    return url + "/v1/messages"


def settings_of(config: dict) -> dict:
    """从 plugins.web_search 取出整理好的四项设置（缺的用默认值）。"""
    cfg = (config.get("plugins") or {}).get("web_search") or {}
    try:
        max_uses = int(cfg.get("max_uses") or DEFAULT_MAX_USES)
    except (TypeError, ValueError):
        max_uses = DEFAULT_MAX_USES
    return {
        "base_url": str(cfg.get("base_url") or "").strip() or DEFAULT_BASE_URL,
        "api_key": str(cfg.get("api_key") or "").strip(),
        "model": str(cfg.get("model") or "").strip() or DEFAULT_MODEL,
        "max_uses": min(max(max_uses, 1), MAX_USES_LIMIT),
    }


def _api_error(response: httpx.Response, api_key: str = "") -> SearchError:
    """把接口返回的错误翻译成能看懂的原因并分类（有的接口会把收到的 Key 写进错误信息，换成 ***）。"""
    try:
        data = response.json()
    except ValueError:
        data = None
    message = ""
    if isinstance(data, dict):
        err = data.get("error")
        message = str(err.get("message") or "") if isinstance(err, dict) else str(err or data.get("message") or "")
    if api_key:
        message = message.replace(api_key, "***")
    status = response.status_code
    detail = f"{status}{'：' + message if message else ''}"
    low = message.lower()
    if status == 402 or "insufficient balance" in low or "余额" in message or "credit balance" in low:
        return SearchError(f"账户余额不足（{detail}）", CONFIG, "联网搜索的账户余额不足，需要去充值")
    if status in (401, 403):
        return SearchError(f"API Key 不对或没有权限（{detail}）", CONFIG, "联网搜索的 API Key 不对")
    if status == 404:
        return SearchError(f"接口地址不对，或者不是 Anthropic 协议的接口（{detail}）", CONFIG,
                           "联网搜索的接口地址不对")
    if status == 429:
        return SearchError(f"请求太频繁，稍后再试（{detail}）", TEMPORARY)
    if data is None:
        return SearchError(f"接口返回的不是 Anthropic 协议的格式（HTTP {status}）",
                           TEMPORARY if status >= 500 else CONFIG, "联网搜索的接口地址不对")
    if status in (400, 422) and ("web_search" in low or "tool" in low or "server" in low):
        return SearchError(f"这个接口或模型不支持自带搜索（{detail}）", CONFIG, "联网搜索设置的接口或模型不支持搜索")
    return SearchError(f"接口返回错误（{detail}）", TEMPORARY)


def _citation_snippets(blocks: list) -> dict:
    """url -> 原文片段：搜索结果本身不带正文，只有模型回答时引用的原文（citations[].cited_text）。
    同一来源被引用多次时拼起来，最多 SNIPPET_LIMIT 字。"""
    snippets = {}
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        for cite in block.get("citations") or []:
            if not isinstance(cite, dict):
                continue
            url, text = str(cite.get("url") or ""), " ".join(str(cite.get("cited_text") or "").split())
            if not url or not text:
                continue
            old = snippets.get(url, "")
            if text in old or len(old) >= SNIPPET_LIMIT:
                continue
            joined = f"{old} {text}".strip()
            snippets[url] = joined if len(joined) <= SNIPPET_LIMIT else joined[:SNIPPET_LIMIT] + "…"
    return snippets


def _parse(data) -> SearchResult:
    if not isinstance(data, dict) or not isinstance(data.get("content"), list):
        raise SearchError("接口返回的不是 Anthropic 协议的格式", CONFIG, "联网搜索的接口地址不对")
    result = SearchResult()
    texts, searched, errors, seen = [], False, [], set()
    snippets = _citation_snippets(data["content"])
    for block in data["content"]:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and block.get("text"):
            texts.append(str(block["text"]))
        elif kind == "server_tool_use":
            searched = True
            query = (block.get("input") or {}).get("query")
            if query:
                result.queries.append(str(query))
        elif kind == "web_search_tool_result":
            searched = True
            content = block.get("content")
            if isinstance(content, dict):  # web_search_tool_result_error
                errors.append(str(content.get("error_code") or "unknown"))
                continue
            for item in content or []:
                if not isinstance(item, dict) or item.get("type") != "web_search_result":
                    continue
                url = str(item.get("url") or "")
                if url in seen:
                    continue
                seen.add(url)
                result.results.append({"title": str(item.get("title") or "无标题"), "url": url,
                                       "page_age": str(item.get("page_age") or ""),
                                       "snippet": snippets.get(url, "")})
    # 有原文片段的（总结里引用过的）排前面，交给聊天模型时只留前几条
    result.results.sort(key=lambda r: not r["snippet"])
    result.summary = "".join(texts).strip()
    if not searched:
        raise SearchError("接口没有执行搜索，这个接口或模型可能不支持自带搜索", CONFIG,
                          "联网搜索设置的接口或模型不支持搜索")
    if not result.results and errors:
        raise SearchError(f"搜索没有成功（{', '.join(errors)}）")
    return result


async def search(settings: dict, query: str) -> SearchResult:
    """搜一次。失败时抛 SearchError，原因写在 message 里。"""
    if not settings.get("api_key"):
        raise SearchError("还没有填写 API Key", CONFIG, "联网搜索还没开通，需要先填 API Key")
    url = messages_url(settings["base_url"])
    if not url.startswith(("http://", "https://")):
        raise SearchError("接口地址要以 http:// 或 https:// 开头", CONFIG, "联网搜索的接口地址不对")
    payload = {
        "model": settings["model"],
        "max_tokens": 1024,
        "thinking": {"type": "disabled"},
        "system": _SEARCH_SYSTEM.format(today=today_text()),
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": settings["max_uses"]}],
        "messages": [{"role": "user", "content": query}],
    }
    headers = {"x-api-key": settings["api_key"], "anthropic-version": ANTHROPIC_VERSION,
               "content-type": "application/json"}
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException:
        raise SearchError(f"等了 {time.monotonic() - started:.0f} 秒没有结果，接口太慢或地址不通")
    except httpx.HTTPError as e:
        raise SearchError(f"连不上接口地址（{type(e).__name__}：{e}）")
    if response.status_code != 200:
        raise _api_error(response, settings["api_key"])
    try:
        data = response.json()
    except ValueError:
        raise SearchError("接口返回的不是 Anthropic 协议的格式", CONFIG, "联网搜索的接口地址不对")
    result = _parse(data)
    result.seconds = round(time.monotonic() - started, 1)
    return result


def format_for_llm(query: str, result: SearchResult) -> str:
    """交给聊天模型：搜索模型的短总结，加前几条来源的标题、日期和原文片段（不给网址，语音用不上）。"""
    lines = [f"【联网搜索结果】搜索：{query}"]
    if result.summary:
        lines.append(f"总结：{result.summary}")
    # 有的接口（如 DeepSeek）回答时不带引用，就只有标题和日期
    has_snippet = any(item["snippet"] for item in result.results[:RESULT_LIMIT])
    if result.results:
        lines.append("来源和原文片段：" if has_snippet else "来源：")
        for i, item in enumerate(result.results[:RESULT_LIMIT], 1):
            age = f"（{item['page_age']}）" if item["page_age"] else ""
            snippet = f"：{item['snippet']}" if item["snippet"] else ""
            lines.append(f"{i}. {item['title']}{age}{snippet}")
    lines.append("回答要求：" + ("总结和原文片段对不上时，以原文片段为准；" if has_snippet else "")
                 + "结果里没有的内容不要编。用口语先说一两句最重要的结论，不要一条条全念，也不要念网址；"
                 "内容多的话，最后问一句用户还想听哪方面。")
    return "\n".join(lines)


def _say_searching(conn) -> None:
    """先播一句"我上网查一下"，接在这一轮回答的前面（同一个 sentence_id）。用户已经打断时不说。"""
    tts, sentence_id = getattr(conn, "tts", None), getattr(conn, "sentence_id", None)
    if tts is None or not sentence_id or getattr(conn, "client_abort", False):
        return
    try:
        tts.tts_one_sentence(conn, ContentType.TEXT, content_detail=SEARCHING_REPLY, sentence_id=sentence_id)
    except Exception as e:
        logger.bind(tag=TAG).warning(f"播搜索提示失败: {e}")


@register_function("web_search", WEB_SEARCH_FUNCTION_DESC, ToolType.SYSTEM_CTL)
async def web_search(conn: "ConnectionHandler", query: str = None):
    logger.bind(tag=TAG).info(f"web_search 被调用 | query={query}")
    if not query:
        return ActionResponse(Action.REQLLM, "请提供搜索关键词。", None)

    settings = settings_of(conn.config)
    logger.bind(tag=TAG).info(
        f"web_search 配置 | url={messages_url(settings['base_url'])} | model={settings['model']} "
        f"| max_uses={settings['max_uses']} | key={'已配置' if settings['api_key'] else '未配置'}")
    if settings["api_key"]:
        _say_searching(conn)
    try:
        result = await search(settings, query)
    except SearchError as e:
        logger.bind(tag=TAG).error(f"联网搜索失败（{e.kind}）: {e}")
        if e.kind == CONFIG:
            reply = (f"联网搜索用不了：{e.spoken}。请用一句话告诉用户这个原因，"
                     "并说明要到网关管理后台「意图识别」页、联网搜索旁边的「设置」里处理；不要编造答案。")
        else:
            reply = "联网搜索这次没成功（临时问题）。请用一句话告诉用户这次没搜到，稍后再试；不要编造答案。"
        return ActionResponse(Action.REQLLM, reply, None)
    except Exception as e:
        logger.bind(tag=TAG).error(f"联网搜索异常: {type(e).__name__}: {e}")
        return ActionResponse(Action.REQLLM, "联网搜索出现异常。请用一句话告诉用户这次没搜到，不要编造答案。", None)

    text = format_for_llm(query, result)
    logger.bind(tag=TAG).info(
        f"联网搜索完成 | {result.seconds} 秒 | 搜索词 {result.queries} | {len(result.results)} 条结果:\n{text}")
    return ActionResponse(Action.REQLLM, text, None)
