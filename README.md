<p align="center">
  <picture>
    <source srcset="hub-docs/images/logo-dark.svg" media="(prefers-color-scheme: dark)">
    <source srcset="hub-docs/images/logo-light.svg" media="(prefers-color-scheme: light)">
    <img src="hub-docs/images/logo-light.svg" alt="xiaozhi-hub-server">
  </picture>
</p>
<p align="center">让小智设备成为 AI 的语音中枢：叫谁的名字，就交给谁干活。</p>
<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/justhalfbit/xiaozhi-hub-server?style=flat-square" /></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10-3776ab?style=flat-square" />
  <img alt="Platform" src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-64748b?style=flat-square" />
  <a href="https://github.com/justhalfbit/xiaozhi-hub-server/stargazers"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/justhalfbit/xiaozhi-hub-server?style=flat-square" /></a>
  <a href="https://x.com/justhalfbit"><img alt="X (Twitter)" src="https://img.shields.io/badge/X-%40justhalfbit-000000?style=flat-square&logo=x&logoColor=white" /></a>
</p>
<p align="center">
  <a href="README.md">简体中文</a> |
  <a href="README.en.md">English</a>
</p>
<p align="center">
  <a href="hub-docs/guide.md">使用说明</a> · <a href="hub-docs/design.md">设计细节</a>
</p>

[![管理后台](hub-docs/images/admin.png)](hub-docs/guide.md)

---

小智设备（M5Stack StackChan、ESP32 开发板等）的**中枢网关**。设备负责听和说；网关负责识别和合成语音，再按你叫的名字，把话交给对应的 agent 或大模型去干活。

- **部署在自己的电脑上**：可以直接让本机的 Claude Code 干活，也可以用各家大模型的 API。
- **部署在云端服务器上**：用不了本机的 Claude Code，改用大模型的 API。

```
小智设备   ⇄   xiaozhi-hub-server        ⇄   Claude Code / 大模型 API / …
听和说          识别、按称呼分派、合成语音       思考和干活
```

> 兼容小智协议的社区项目，不是小智官方项目。

## 功能

**一个设备，多个 AI**

- **叫名字换 AI**：说"嗨 Claude"交给 Claude Code，说"嗨 DeepSeek"交给 DeepSeek，称呼在管理后台配。每个 AI 各聊各的，每台设备记得上次用的是哪个。
- **Claude Code**：用你在本机登录的账号干活，比如"把这周的笔记整理成一份清单"。
- **大模型 API**：Claude、OpenAI、Gemini、DeepSeek、通义千问、豆包、智谱 GLM、Kimi，以及任意 OpenAI 兼容接口。
- **Grok Bot**：给 xAI 的云端 AI 同事派活。在后台用 Cursor 账号登录、勾选 Bot、配上称呼，说"嗨 孙悟空，帮我把这周的 X 热点整理一下"，它的回复会念出来。长任务可以打断不等，它在云端做完后设备会主动汇报。走的是 App 用的内部接口，不是官方 API，App 升级后可能失效。
- **更多 agent**：Codex 在规划中。网关里的 agent 都走同一套会话接口，接入新的 agent 主要就是实现这套接口。

**长任务也能陪着做**

- **陪着等**：让 Claude Code 做几分钟的任务，设备一直陪着，隔一会儿报一下进度，做完当场说结果。
- **随时打断**：设备说话时按一下设备上的按键（StackChan 是点一下屏幕上的脸），或者喊唤醒词，它马上停下。
- **接着上次聊**：断线重连自动接上；隔得久了，说"接着上次聊"也能接回去（Claude Code）。

**听得准**

- 中英夹杂也能认对：默认用离线运行的 X-ASR，"帮我跑一下 npm install"也能认准；认错的词可以加进热词和纠错表。也可以换成豆包流式识别。

**管理后台**

- 在网页上配置大模型和 API Key、Claude Code 的权限、称呼、插件、热词纠错和音色，查看在线设备和环境自检。

## 快速开始

开始前：电脑和设备连在同一个局域网，电脑上装好 git。要用 Claude Code 的话，先装好它并在终端运行一次 `claude` 登录；没装时会改用 Claude API，到管理后台填 API Key 就行。

### 1. 安装

**macOS / Linux**，在终端运行：

```bash
curl -fsSL https://raw.githubusercontent.com/justhalfbit/xiaozhi-hub-server/main/scripts/bootstrap.sh | bash
```

代码装到 `~/xiaozhi-hub-server`，结束时询问是否开机自启（直接回车就是开启）。macOS 缺 ffmpeg、opus 时会用 [Homebrew](https://brew.sh) 安装，这种情况要先装好 Homebrew。

**Windows 10 / 11**，在 PowerShell 里运行：

```powershell
git clone --depth 1 https://github.com/justhalfbit/xiaozhi-hub-server.git
cd xiaozhi-hub-server
powershell -ExecutionPolicy Bypass -File scripts\install.ps1   # 安装
powershell -ExecutionPolicy Bypass -File scripts\start.ps1     # 启动
```

第一次启动时 Windows 防火墙会询问，请勾选"专用网络"并允许，否则设备连不上。Windows 暂时没有开机自启，每次开机后运行一次 `start.ps1`。

不想用 `curl | bash`、下载太慢或者想看装了什么，见[使用说明 · 安装](hub-docs/guide.md#1-安装)。

### 2. 设置管理密码

装好后尽快在这台电脑的浏览器里打开 `http://127.0.0.1:16688/admin`（其他电脑用 `http://<电脑的局域网 IP>:16688/admin`），第一次使用时先设置管理密码。在你设置之前，局域网里谁先打开后台谁就能设置。

### 3. 连接设备

设备配网时，在配网页的"高级选项"里填 OTA 地址：

```
http://<电脑的局域网 IP>:16688/xiaozhi/ota/
```

StackChan 的配网步骤、怎么跟设备说话，见[使用说明](hub-docs/guide.md#3-连接设备)。

**部署到云端服务器**：只建议在局域网里用；要装在云服务器上，请让设备和服务器走 Tailscale、WireGuard 这类私人组网，不要把端口放行到公网，见[使用说明 · 部署到云端服务器](hub-docs/guide.md#部署到云端服务器)。

## 更新与卸载

| | macOS / Linux | Windows |
| --- | --- | --- |
| 更新 | `bash ~/xiaozhi-hub-server/scripts/update.sh` | 在项目目录里 `git pull`，再运行一次 `install.ps1` |
| 卸载 | `bash ~/xiaozhi-hub-server/scripts/uninstall.sh` | 运行 `scripts\uninstall.ps1`，注意事项见[使用说明](hub-docs/guide.md#卸载) |

手动下载的，把 `~/xiaozhi-hub-server` 换成自己的目录。开机自启的开关、卸载会删什么，见[使用说明 · 开机自启、更新与卸载](hub-docs/guide.md#7-开机自启更新与卸载)。

## 安全提示

- 只在可信的局域网内使用，不要把 `16688`、`16689` 端口暴露到公网；要远程用（包括部署到云端服务器），走 Tailscale、WireGuard 这类私人组网。
- 语音连接（`16689` 端口）实际上不认证，只认设备自己报的设备号。同一局域网的人不需要设备就能冒充设备号连上来：使用你的模型、读写这台设备的聊天记录和记忆、指挥 Claude Code 和 Grok Bot。
- 服务启动后尽快打开管理后台，设置足够强的管理密码；在你设置之前，局域网里谁先打开后台谁就能设置。
- Claude Code 默认只能读取工作目录 `~/xiaozhi-workspace` 里的文件。在管理后台放开写文件、执行命令后，语音指令就能修改文件、执行命令，请谨慎；其中执行命令（Auto 级别）不受工作目录限制。

## 关于本仓库

本项目只用到 `main/xiaozhi-server`、`scripts/` 和 `hub-docs/`。其余目录是上游 [xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server) 的原样代码，本项目不使用，保留它们是为了定期合并上游时少出冲突，好跟上小智固件的更新；一键安装不会下载这些目录。

想了解一句话从唤醒到回答经过了什么，看[设计细节](hub-docs/design.md)，或者管理后台的"工作原理"页。

## 许可

[MIT License](LICENSE)。基于 [xinnan-tech/xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server)（MIT）改造，第三方组件的致谢见[使用说明](hub-docs/guide.md#致谢与许可)。
