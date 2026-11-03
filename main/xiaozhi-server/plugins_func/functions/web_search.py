"""联网搜索：向 Anthropic 协议的接口发一次请求，带上服务端自带的搜索工具 web_search_20250305，
让那个模型去搜，拿回总结和搜索结果（标题、日期），再交给聊天模型回答。

和 DeepSeek Harness 的做法一致：不接专门的搜索服务，用支持自带搜索的 Anthropic 协议接口
（DeepSeek 的 https://api.deepseek.com/anthropic/v1、Anthropic 官方等）。聊天模型用哪家都可以，和这里无关。
"""

import time
from dataclasses import dataclass, field

import httpx
from config.logger import setup_logging
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
# 语音场景用户在干等：连不上 5 秒就放弃，整次搜索（可能搜好几轮）最多 30 秒
TIMEOUT = httpx.Timeout(30.0, connect=5.0)
RESULT_LIMIT = 8  # 交给聊天模型的搜索结果条数上限
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


class SearchError(Exception):
    """搜索失败，message 是给人看的原因（后台测试按钮直接显示）。"""


@dataclass
class SearchResult:
    summary: str = ""
    results: list = field(default_factory=list)  # [{"title", "url", "page_age"}]
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


def _api_error(response: httpx.Response, api_key: str = "") -> str:
    """把接口返回的错误翻译成能看懂的原因（有的接口会把收到的 Key 写进错误信息，换成 ***）。"""
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
    if status in (401, 403):
        return f"API Key 不对或没有权限（{status}{'：' + message if message else ''}）"
    if status == 404:
        return f"接口地址不对，或者不是 Anthropic 协议的接口（404{'：' + message if message else ''}）"
    if data is None:
        return f"接口返回的不是 Anthropic 协议的格式（HTTP {status}）"
    low = message.lower()
    if status in (400, 422) and ("web_search" in low or "tool" in low or "server" in low):
        return f"这个接口或模型不支持自带搜索（{status}：{message}）"
    return f"接口返回错误（{status}{'：' + message if message else ''}）"


def _parse(data) -> SearchResult:
    if not isinstance(data, dict) or not isinstance(data.get("content"), list):
        raise SearchError("接口返回的不是 Anthropic 协议的格式")
    result = SearchResult()
    texts, searched, errors, seen = [], False, [], set()
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
                                       "page_age": str(item.get("page_age") or "")})
    result.summary = "".join(texts).strip()
    if not searched:
        raise SearchError("接口没有执行搜索，这个接口或模型可能不支持自带搜索")
    if not result.results and errors:
        raise SearchError(f"搜索没有成功（{', '.join(errors)}）")
    return result


async def search(settings: dict, query: str) -> SearchResult:
    """搜一次。失败时抛 SearchError，原因写在 message 里。"""
    if not settings.get("api_key"):
        raise SearchError("还没有填写 API Key")
    url = messages_url(settings["base_url"])
    if not url.startswith(("http://", "https://")):
        raise SearchError("接口地址要以 http:// 或 https:// 开头")
    payload = {
        "model": settings["model"],
        "max_tokens": 1024,
        "thinking": {"type": "disabled"},
        "system": _SEARCH_SYSTEM.format(today=time.strftime("%Y-%m-%d")),
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
        raise SearchError(_api_error(response, settings["api_key"]))
    try:
        data = response.json()
    except ValueError:
        raise SearchError("接口返回的不是 Anthropic 协议的格式")
    result = _parse(data)
    result.seconds = round(time.monotonic() - started, 1)
    return result


def format_for_llm(query: str, result: SearchResult) -> str:
    lines = [f"【联网搜索结果】搜索：{query}"]
    if result.summary:
        lines.append(f"总结：{result.summary}")
    if result.results:
        lines.append("来源：")
        for i, item in enumerate(result.results[:RESULT_LIMIT], 1):
            age = f"（{item['page_age']}）" if item["page_age"] else ""
            lines.append(f"{i}. {item['title']}{age}")
    lines.append("请根据以上结果用口语简短回答，不要念网址；结果里没有的内容不要编。")
    return "\n".join(lines)


@register_function("web_search", WEB_SEARCH_FUNCTION_DESC, ToolType.SYSTEM_CTL)
async def web_search(conn: "ConnectionHandler", query: str = None):
    logger.bind(tag=TAG).info(f"web_search 被调用 | query={query}")
    if not query:
        return ActionResponse(Action.REQLLM, "请提供搜索关键词。", None)

    settings = settings_of(conn.config)
    logger.bind(tag=TAG).info(
        f"web_search 配置 | url={messages_url(settings['base_url'])} | model={settings['model']} "
        f"| max_uses={settings['max_uses']} | key={'已配置' if settings['api_key'] else '未配置'}")
    try:
        result = await search(settings, query)
    except SearchError as e:
        logger.bind(tag=TAG).error(f"联网搜索失败: {e}")
        return ActionResponse(
            Action.REQLLM,
            f"联网搜索失败：{e}。请用一句话告诉用户这次没搜到，不要编造答案。",
            None,
        )
    except Exception as e:
        logger.bind(tag=TAG).error(f"联网搜索异常: {type(e).__name__}: {e}")
        return ActionResponse(Action.REQLLM, "联网搜索出现异常。请用一句话告诉用户这次没搜到，不要编造答案。", None)

    text = format_for_llm(query, result)
    logger.bind(tag=TAG).info(
        f"联网搜索完成 | {result.seconds} 秒 | 搜索词 {result.queries} | {len(result.results)} 条结果:\n{text}")
    return ActionResponse(Action.REQLLM, text, None)
