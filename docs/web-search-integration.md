# 联网搜索插件

## 功能简介

联网搜索插件 `web_search` 让大模型在对话中查实时信息（新闻、天气以外的最新消息、价格、比分等）。

做法和 DeepSeek Harness 一致：不接专门的搜索服务，而是向一个 **Anthropic 协议** 的接口发请求，带上接口自带的搜索工具 `web_search_20250305`，让那个模型去搜，拿回总结和搜索结果（标题、日期），再交给聊天模型回答。

- 需要支持自带搜索的 Anthropic 协议接口，例如 DeepSeek（`https://api.deepseek.com/anthropic/v1`，用 DeepSeek 平台的 Key，见 [DeepSeek 文档](https://api-docs.deepseek.com/guides/anthropic_api)）、Anthropic 官方（`https://api.anthropic.com`）。
- 聊天模型和这里无关，用哪家都可以；搜索接口单独设置。
- 搜不到或出错时，插件会让模型告诉用户"这次没搜到"，不会卡住。整次搜索最多等 20 秒（实测一般 3 到 5 秒）。

## 在管理后台设置（推荐）

1. 打开管理后台的「意图识别」页，选「用插件」。
2. 在「联网搜索」旁边点「设置」，填写：
   - 接口地址：默认 `https://api.deepseek.com/anthropic/v1`（和 DeepSeek Harness 一致），写到 `/anthropic` 或 `/anthropic/v1` 都可以；
   - API Key：只写不读，填过后显示"已配置"，留空表示不改；
   - 模型：默认 `deepseek-v4-flash`；
   - 单次最多搜索次数：默认 5（1–10）。
3. 点「测试」会真搜一次：成功显示搜到几条和第一条标题，失败说明原因（Key 不对、地址不通、不是 Anthropic 协议、不支持自带搜索等）。
4. 点「保存设置」后马上生效，不用重启；再勾选「联网搜索」并保存意图识别。

> 网关装在设备上时，接口地址里的 `127.0.0.1` 指设备自己。要用另一台电脑上的接口，填那台电脑的局域网地址，并让那个服务监听局域网。

## 在配置文件里设置

在 `data/.config.yaml` 中：

```yaml
plugins:
  web_search:
    base_url: "https://api.deepseek.com/anthropic/v1"
    api_key: "你的API Key"
    model: "deepseek-v4-flash"
    max_uses: 5
    # 可选：工具描述，告诉大模型什么时候该搜
    # description: "联网搜索工具。……"
Intent:
  function_call:
    functions:
      - web_search
```

改配置文件后重启服务生效。

旧版本的 `provider`（秘塔、Tavily、Serply）和 `max_results` 已不再使用。配置里还留着 `provider` 时，原来填的 Key 是给那些服务的，新版不会使用、也不会发出去，插件当成还没配置；在后台重新填 Key 并保存，会一起清掉旧设置。
