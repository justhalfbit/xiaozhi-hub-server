# xiaozhi-hub-server 使用说明

[返回 README](../README.md) · [设计细节](design.md)

xiaozhi-hub-server 是小智设备（M5Stack StackChan、ESP32 开发板等）的**本地服务端**。设备负责听和说；服务端运行在你自己的电脑上，把你说的话交给本机的 Claude Code，或者你选的大模型。

它是兼容小智协议的社区项目，不是小智官方项目，与官方没有关联。

从安装到第一次对话，按顺序看前四节就够了：

1. [安装](#1-安装)
2. [首次配置](#2-首次配置)
3. [连接设备](#3-连接设备)
4. [跟设备说话](#4-跟设备说话)
5. [管理后台](#5-管理后台)
6. [常见问题](#6-常见问题)
7. [开机自启、更新与卸载](#7-开机自启更新与卸载)
8. [安装细节](#8-安装细节)

---

## 1. 安装

需要：

- macOS、Linux 或 Windows 10/11；
- 能访问网络；
- 已安装 git（macOS 没有时运行 `xcode-select --install`；Windows 没有时在 PowerShell 里运行 `winget install --id Git.Git -e`）。

下面是 macOS / Linux 的步骤，Windows 见本节最后的 [Windows 安装](#windows-安装)。

**一键安装**：在终端运行下面这一行。代码会下载到 `~/xiaozhi-hub-server`，然后自动安装；安装结束时会询问是否开机自启，直接回车就是开启（见 [§7](#7-开机自启更新与卸载)）。

```bash
curl -fsSL https://raw.githubusercontent.com/justhalfbit/xiaozhi-hub-server/main/scripts/bootstrap.sh | bash
```

**手动安装**：

```bash
git clone --depth 1 https://github.com/justhalfbit/xiaozhi-hub-server.git
cd xiaozhi-hub-server
bash scripts/install.sh   # 安装
bash scripts/start.sh     # 启动（没有开启开机自启时）
```

安装脚本会准备好运行需要的一切，缺什么会先问你再装：Python 3.10 和依赖（装在项目自己的虚拟环境里，不影响系统的 Python）、opus 和 ffmpeg、语音识别模型，以及配置文件。电脑上装了 Claude Code 时，直接用它当大模型；没装时用 Claude API，需要到管理后台填 API Key（也可以换成别的厂商）。

一共下载约 370MB，装好后占用约 800MB 硬盘，运行时内存约 0.6GB。国内网络慢的话，先看 [§8 安装细节](#8-安装细节) 里的镜像设置。

安装结束和每次启动时都会运行自检，并显示管理后台和 OTA 的地址。

### Windows 安装

在 PowerShell 里运行：

```powershell
git clone --depth 1 https://github.com/justhalfbit/xiaozhi-hub-server.git
cd xiaozhi-hub-server
powershell -ExecutionPolicy Bypass -File scripts\install.ps1   # 安装
powershell -ExecutionPolicy Bypass -File scripts\start.ps1     # 启动
```

- 安装脚本做的事和 macOS / Linux 一样：uv、Python 3.10、依赖、语音识别模型、配置文件。ffmpeg 用 winget 安装（`Gyan.FFmpeg`）；缺 opus 时自动下载 `opus.dll` 放进虚拟环境（见 [§8](#8-安装细节)）。
- 第一次启动时，Windows 防火墙会询问是否允许 Python 访问网络，请勾选"专用网络"并允许，否则设备连不上。
- Windows 目前没有开机自启，每次开机后运行一次 `start.ps1`。
- 更新：在项目目录里运行 `git pull`，再运行一次 `install.ps1` 补装新增的依赖。
- 卸载：先运行 `cd ~` 离开项目目录（Windows 删不掉正在使用的目录），再运行 `powershell -ExecutionPolicy Bypass -File xiaozhi-hub-server\scripts\uninstall.ps1`（项目不在用户目录下的，换成实际路径）。

### 部署到云端服务器

Linux 服务器的安装方法和上面一样。云服务器上一般没有 Claude Code，安装脚本会改用 Claude API，到管理后台填 API Key 就行（也可以换成别的厂商）。装好后还要做这几步：

1. **放行端口**：在云服务器的安全组或防火墙里放行 `16688`（OTA 接口和管理后台）和 `16689`（设备的语音连接），两个都要开。设备每次开机都会先访问 OTA 接口，所以 `16688` 不能只对自己的 IP 开放。
2. **写明公网地址**：编辑 `main/xiaozhi-server/data/.config.yaml`，在已有的 `server:` 下面加一行 `websocket`：

   ```yaml
   server:
     websocket: ws://<服务器的公网 IP 或域名>:16689/xiaozhi/v1/
   ```

   设备会从 OTA 接口拿到这个地址再去连接。不写的话，下发的是服务器自己检测到的 IP，云服务器上通常是内网 IP，设备连不上。改完重启服务（开了开机自启的，运行 `bash scripts/autostart.sh restart`）。
3. **设置管理密码**：服务启动后，尽快在浏览器里打开 `http://<公网 IP>:16688/admin` 设置管理密码。在你设置之前，谁先打开后台谁就能设置。
4. **连接设备**：配网时 OTA 地址填 `http://<公网 IP>:16688/xiaozhi/ota/`。

- 想让服务器开机就启动、不用登录，运行 `sudo loginctl enable-linger $USER`，见 [§7](#7-开机自启更新与卸载)。
- 管理后台走的是 HTTP，登录时密码是明文传输的。

## 2. 首次配置

1. **设置管理密码**：用浏览器打开管理后台，第一次打开时设置管理密码（至少 6 位）。在运行服务的这台电脑上打开 `http://127.0.0.1:16688/admin`，其他电脑用 `http://<服务所在电脑的 IP>:16688/admin`。
2. **准备 Claude Code**（要用 Claude Code 时）：在终端运行一次 `claude` 完成登录。Claude Code 默认只能读取工作目录 `~/xiaozhi-workspace` 里的文件；模型、思考级别和权限级别可以在管理后台的"大模型"页调整。
3. **配置别的大模型**（可选）：在"大模型"页选一个厂商，填接口地址、模型和 API Key，可以先点"测试连接"试一下。

所有设置都保存在 `main/xiaozhi-server/data/.config.yaml`，这个文件不会提交到 Git。API Key 在里面是明文，管理密码只保存哈希值。

## 3. 连接设备

设备配网时，在配网页的"高级选项"里填写 OTA 地址：

```
http://<电脑的局域网 IP>:16688/xiaozhi/ota/
```

| 设备 | 方法 |
| --- | --- |
| 官方小智固件 1.6.1 及以上的开发板 | 进入配网模式，在配网页的"高级选项"里填写 OTA 地址 |
| M5Stack StackChan（官方固件，实测 1.5.1） | 在 App 中解绑设备 → 设置 → 系统 → Factory Reset → 开机引导页选 Skip → 进入 AI Agent → 手机连接设备热点打开配网页 → 在"高级选项"里填写 OTA 地址 → 填写 2.4GHz Wi-Fi |

- 设备同一时间只连一台电脑。换一台电脑运行服务时，要重新配网，填新电脑的 OTA 地址。
- 建议在路由器里给这台电脑设置固定 IP，否则 IP 变了设备就连不上。
- StackChan 待机时会随机摇头，可以在设备的"设置 → AI.Agent → General → Idle movement"里关闭。

## 4. 跟设备说话

下面的时长都是默认值，可以在管理后台改。后台的"工作原理"页会按你当前的设置列出同样的内容。

**唤醒**：说"Hi, StackChan"，设备回一句问候后开始听；也可以点一下屏幕，设备直接开始听（没有问候）。

**换模型（呼名）**：句子开头说"嗨 + 称呼"（说"Hi""Hey"也行），就换成对应的模型回答。比如"嗨 Claude"交给本机 Claude Code，"嗨 DeepSeek"交给 DeepSeek。
- 称呼后面可以直接接内容：说"嗨 Claude，看看工作目录里有什么"，称呼会去掉，后半句交给 Claude。
- 只说"嗨 Claude"时，设备回一声"我在"，等你说下一句。
- 换过一次后，这台设备之后都交给这个模型，直到你叫别的称呼；断线重连、服务重启后也记得。
- 每个模型只记得跟自己的对话，切换时不带上下文过去。
- 默认只能叫 Claude。给其他模型配称呼，或者按你的发音录入称呼，在后台的"呼名"页（见 [§5](#5-管理后台)）。

**给 Grok Bot 派活**：Grok Bot 是 xAI 的云端 AI 同事，每个 Bot 有自己的名字和云端电脑。在后台的"Grok Bot"页用 Cursor 账号登录（给出一个登录地址，在浏览器里批准即可，不用在后台输密码）、勾选要叫的 Bot，它的名字会自动成为称呼（想改写法或按你的发音录入，到"呼名"页），之后说"嗨 孙悟空，帮我把这周的 X 热点整理一下"就行。
- 消息会出现在 Grok Bot App 里该 Bot 的主对话中，和你在 App 里发的一样；它的回复一条一条念出来，简单问题几秒，派活要几分钟。每句话后面会附一句"请口语化简短回答"的提示（App 里能看到），不想要可以在配置里把 `voice_hint` 填空。
- 叫到它的名字时服务端就提前把它的云端电脑叫醒，第一句话不用等。回复慢时（4 秒）设备先说"交给 Grok Bot 了"，之后隔一会儿报一次进度，最长等 10 分钟。
- **不用陪着等**：按键打断、或等到超时，它在云端继续做。做完后设备会主动说"孙悟空做完了：……"：设备在线且空闲就马上说，设备在休息就下次唤醒时先说，你说"嗨 孙悟空"切回去时也会先说。服务重启后还没念的就没了。
- Bot 需要你审批或回答问题时，设备会说"Grok Bot 在等你去 App 里处理"。
- 一个 Bot 的对话只有一份：同一时间只从一台设备（或 App）跟它说话，两边同时发，设备会把两边的回复都念出来。设备在它干活时断开也没关系，做完会在下次连上时汇报。
- 走的是 Grok Bot 桌面 App 用的内部接口，不是官方公开的 API：App 升级后可能失效，使用频率请和手动聊天相当。登录凭证只保存在本机。

**打断**：设备说话时**不收音**，不能直接插话（这是固件的收音方式）。要打断，按一下设备上的按键（开发板上通常是 BOOT 键；StackChan 是点一下屏幕上的脸），它马上停下，然后你接着说。喊唤醒词也能打断，但设备音量大时容易听不到。

**长任务**：让 Claude 做要几分钟的事（比如整理一批文件）时，设备会陪着等，最长 10 分钟，做完当场说结果。
- 30 秒还没开口，设备会说"还在处理，请稍等"；之后报已用时间，间隔逐渐拉长到 2 分钟一次。
- 等待期间想说话，先按键或点屏幕打断。
- 超时被停下、Claude 没有回应、或者你打断之后，说"继续"它会接着做。

**接着上次聊**（使用 Claude Code 时）：
- 两小时内再次唤醒，自动接上上次的对话。
- 隔得更久的，说"接着上次聊"或"继续刚才的话题"也能接上；说"嗨 Claude，接着上次聊"同样可以。
- 对话记录按设备、按电脑分别保存。
- 用其他大模型（API）时，对话只在这次连接里有效，设备断开后就不记得了。

**结束**：说"退出"或"关闭"，马上断开。2 分钟没人说话时，设备说一句"好一会儿没听到你说话，我先休息啦"后断开。

想知道连接什么时候断、Claude Code 进程什么时候启动和关闭、哪些记录会保存，见 [设计细节](design.md)。

## 5. 管理后台

在本机打开 `http://127.0.0.1:16688/admin`；局域网里的其他电脑用 `http://<电脑的局域网 IP>:16688/admin`。

| 页面 | 能做什么 | 什么时候生效 |
| --- | --- | --- |
| 状态 | 在线设备、连接地址、环境自检；给设备发表情，看固件支持哪些表情 | — |
| 工作原理 | 按当前设置列出一句话经过的每一步、说什么会怎样，以及发给大模型的提示词 | — |
| 大模型 | 在本机 Claude Code 和各家 API 之间选一个默认模型（设备刚连上时用它；配了称呼的模型随时能叫到，页面上会标出每个模型的称呼）。Claude Code 可以选模型、思考级别、权限级别（只读 / 可编辑 / Auto）、是否联网，以及超时、进度提示等高级设置 | 设备重新连接后 |
| 呼名 | 给每个模型配称呼，改只说称呼时的回复。"用设备录入"：唤醒设备说 3 遍"嗨 + 称呼"，把识别到的写法勾选加入称呼 | 马上 |
| Grok Bot | 用 Cursor 账号登录 Grok Bot（浏览器里批准），从云端读取 Bot 列表并勾选要在设备上呼叫的，Bot 的名字自动成为称呼；退出登录会删掉本机凭证 | 马上 |
| 语音识别 | 在本地 X-ASR 和豆包流式识别之间切换（豆包填 API Key，可以先测试连接）；说完停顿多久算一句；热词；识别后纠错；最近识别的句子，认错的词一键加进热词或纠错表 | 停顿时长、纠错：马上；切换识别、热词：设备重新连接后 |
| 语音合成 | 选 EdgeTTS 的音色并试听 | 设备重新连接后 |
| 意图识别 | 用 API 模型时，选择要不要让模型调用插件（查天气、听新闻、放音乐等）。用 Claude Code 时固定为不用插件 | 设备重新连接后 |

"设备重新连接"：等设备因为没人说话自动断开，下次唤醒时就用上了；着急的话可以重启设备。

**热词和纠错怎么选**：某个词经常被认错时，先把正确的写法加进热词，让识别更偏向它；加了热词还是认错（比如 "Claude Code" 总被认成 "cloud code"），就在纠错表里写"认错的写法 → 正确写法"，识别出文字后直接替换。

### 本机设备接口

设备客户端和服务端装在同一台机器上时（例如带屏幕的语音终端），客户端可以用下面的接口在自己的屏幕上切换模型、提示待汇报，不用登录后台。只接受本机直接发来的请求：局域网里的其他电脑、经过反向代理转发来的请求（带 `X-Forwarded-For` 或 `Forwarded` 头）一律返回 403。

| 接口 | 请求 | 返回 |
| --- | --- | --- |
| `GET /kiosk/llm` | 可选 `?device=<device-id>` | `options`：可选的模型（装了 Claude Code 时排第一，之后是填好的 API 厂商），每项有 `kind`（`cli` / `api`）、`vendor`、`label`、`entry`、`model`；`default`：后台默认模型；`selected`：带了 device 时是这台设备实际会用的模型（它上次切到的，还可用的话），否则同 `default` |
| `POST /kiosk/llm/use` | `{"kind": "cli"}` 或 `{"kind": "api", "vendor": "deepseek"}`，可带 `"device"` | 和后台"保存并设为默认"一样切换默认模型（意图识别跟着切）；在线的设备马上换到新模型（从新对话开始，同呼名切换）；带了 device 时这台设备记住这个模型 |
| `POST /kiosk/reports` | `{"device": "...", "quiet": false, "play": false}` | `count`：这台设备有几条待汇报；`playing`：是否已安排念。`quiet` 为 true 表示设备在睡眠时段，期间有待汇报也不主动念（每次查询都要带上）；`play` 为 true 表示用户点了要听，马上在设备的连接上念 |

## 6. 常见问题

**连接**

- **设备显示"检查新版本失败"**：确认电脑和设备在同一个局域网，OTA 地址的 IP、端口正确。用手机浏览器打开 OTA 地址，能看到"OTA接口运行正常"，就说明网络是通的。
- **电脑 IP 变了，设备连不上**：在路由器里给电脑设置固定 IP，然后重新配网。
- **Windows 防火墙**：需要放行 TCP 16688 和 16689 端口（仅限专用网络）。

**听和说**

- **话说到一半就被截断**：在"语音识别"页把"说完停顿多久算一句"调大，一般 800～1200 毫秒比较合适。
- **某个词总是认错**：用"语音识别"页的热词和纠错，见 [§5](#5-管理后台) 末尾。
- **叫"嗨 Claude"没反应**：在"语音识别"页的"最近识别"里看你的发音被认成了什么，再到"呼名"页用"用设备录入"把这种写法加进称呼。
- **设备说话断续**：调大 `data/.config.yaml` 里的 `tts_audio_prebuffer_count`（默认 15，约 0.9 秒），合理范围是 10～25。调大后，打断时要多播一会儿才停。
- **用豆包时设备说"语音识别连不上"**：到"语音识别"页检查豆包的 API Key，点"测试连接"看具体原因。

**Claude Code**

- **Claude Code 没有反应**：先在"状态"页确认能找到 Claude Code。状态页只检查有没有安装，不检查有没有登录；没登录时，在终端运行一次 `claude` 登录。然后查看 `main/xiaozhi-server/tmp/server.log`，里面记录了工具调用、限流状态和每轮的结果。
- **回答太慢**：在"大模型"页把模型换成 Haiku，思考级别选"低"。
- **它说没有权限做某件事**：默认只能读文件。要让它改文件或执行命令，在"大模型"页把权限级别改成"可编辑"或"Auto"，改之前先看 [安全提示](#安全提示)。

**安装**

- **自检提示"模块依赖"缺包**：重新运行一次安装脚本补装依赖。
- **下载模型很慢或失败**：设置 Hugging Face 镜像后重新运行安装脚本，见 [§8](#8-安装细节)；下载支持断点续传。

## 7. 开机自启、更新与卸载

### 开机自启

macOS 和 Linux 安装结束时会询问是否开机自启，默认开启（直接回车）；Windows 目前没有开机自启。非交互安装（`ASSUME_YES=1` 或没有终端）也会开启。开启后，服务在后台运行，不用开着终端；异常退出时 30 秒后自动重启。以后也可以随时开关：

```bash
bash scripts/autostart.sh enable    # 开启，并马上在后台启动
bash scripts/autostart.sh disable   # 关闭，并停止后台服务
bash scripts/autostart.sh status    # 查看是否在运行
bash scripts/autostart.sh restart   # 重启后台服务
```

- **macOS**：用 launchd（`~/Library/LaunchAgents/com.xiaozhi-hub.server.plist`），**登录这个用户后**自动启动。Claude Code 的登录信息存在用户的钥匙串里，所以不能做成登录前就启动的系统服务。想开机就启动，在"系统设置 → 用户与群组"里开启自动登录（开了 FileVault 的电脑不能自动登录）。安装目录不要放在"文稿""桌面""下载"里，后台服务没有权限读这些目录。
- **Linux**：用 systemd 用户服务（`~/.config/systemd/user/xiaozhi-hub.service`），默认登录后启动。想开机就启动（不用登录），运行 `sudo loginctl enable-linger $USER`。
- 日志照常在 `main/xiaozhi-server/tmp/server.log`。服务启动失败时，macOS 的报错在 `tmp/autostart-error.log`，Linux 用 `journalctl --user -u xiaozhi-hub` 查看。
- 开启时会记下当前终端的 PATH，让后台服务能找到 ffmpeg 和 claude。以后 Claude Code 换了安装位置，重新运行一次 `enable` 就行。
- 开着自启时，不要再手动运行 `start.sh`，否则端口会冲突。想在终端里看实时输出调试，先 `disable`，调试完再 `enable`。
- 一个用户只能有一份安装开机自启。装了好几份（例如换了安装目录）时，开机自启属于最后一次开启它的那一份；在另一份里安装或运行 `enable`，会询问后改成指向那一份，并停掉原来那份的后台服务。`disable`、更新和卸载不会动不属于自己的开机自启。

### 更新

```bash
bash ~/xiaozhi-hub-server/scripts/update.sh
```

手动下载的，换成自己的目录。更新脚本会：

1. 从 GitHub 拉取最新代码。配置 `data/.config.yaml`、语音识别模型和虚拟环境都不在 Git 里，不受影响；
2. 重新运行安装脚本，补装新版本新增的依赖和模型，已有的直接跳过；
3. 开了开机自启的话，自动重启后台服务。手动启动的服务，在它的终端按 Ctrl+C 停止，再运行 `start.sh`。

如果改过仓库里的文件（例如直接改了 `config.yaml`），更新脚本会列出这些文件并停下，不会覆盖。自己的配置请在管理后台修改，或者写在 `data/.config.yaml` 里。

再次运行一键安装命令也可以：检测到已经安装过，会改为运行更新脚本。

### 卸载

```bash
bash ~/xiaozhi-hub-server/scripts/uninstall.sh
```

Windows：先运行 `cd ~` 离开项目目录（Windows 删不掉正在使用的目录），再运行 `powershell -ExecutionPolicy Bypass -File xiaozhi-hub-server\scripts\uninstall.ps1`（项目不在用户目录下的，换成实际路径）。卸载脚本会先列出要删的内容，回答 `y` 才执行：

1. 取消开机自启，停止后台服务（macOS 删掉 LaunchAgent，Linux 删掉 systemd 用户服务）；
2. 删除整个安装目录，包括虚拟环境、语音识别模型、配置文件和里面的 API Key；
3. 询问是否删除 Claude Code 的工作目录 `~/xiaozhi-workspace`，默认保留。

- 用 `start.sh` 手动启动的服务还在运行时，卸载脚本会停下，请先按 Ctrl+C 停止服务。
- 完整 clone 的 git 仓库（不是一键安装下载的）可能有没推送的改动，卸载脚本只取消开机自启，不删目录。
- uv、Python 3.10、ffmpeg、opus、Claude Code 可能被别的程序共用，不会删除。卸载结束时会列出手动删除的方法，例如 `uv cache clean` 删掉约 500MB 的下载缓存。

## 8. 安装细节

### 装了什么

- uv（Python 包管理工具）、Python 3.10、opus 和 ffmpeg：电脑上已有就直接用，缺少时先询问再安装：
  - macOS：用 Homebrew 安装 ffmpeg 和 opus；
  - Linux：用 apt、dnf 或 pacman 安装（需要 sudo）；
  - Windows：ffmpeg 用 winget 安装（`Gyan.FFmpeg`）；opus 自动下载 discord.py 官方安装包里的 Windows 版 libopus 1.3.1（约 1MB，固定版本并校验哈希），放到 `main/xiaozhi-server/.venv/Scripts/opus.dll`。ARM 版 Windows 没有现成的 opus.dll，需要手动准备；
- Python 依赖：全部装在项目自己的虚拟环境 `main/xiaozhi-server/.venv` 里；
- 语音识别模型：X-ASR 中英混合模型（sherpa-onnx int8 版），从 Hugging Face 下载；
- 配置文件 `main/xiaozhi-server/data/.config.yaml`：已经存在时保持不变；
- 一键安装只下载运行需要的 `main/xiaozhi-server`、`scripts`、`hub-docs`，不下载上游的智控台、数字人等目录和提交历史。

### 下载量和内存

2026-10-20 在 macOS 上实测：

| 项目 | 下载 | 装好后占用硬盘 |
| --- | --- | --- |
| 代码（只含运行需要的目录） | 约 21MB | 约 50MB |
| Python 3.10 | 约 25MB | 约 66MB |
| Python 依赖（87 个包，最大的是 claude-agent-sdk，约 61MB） | 约 146MB | 约 500MB |
| 语音识别模型 X-ASR | 约 175MB | 约 175MB |
| **合计** | **约 370MB** | **约 800MB** |

- 电脑上没有 uv、ffmpeg、opus 时，还要额外下载它们，不算在上表里。
- uv 会在 `~/.cache/uv` 里留一份 Python 包的缓存（约 500MB），下次安装、更新时直接用，不用重新下载。不需要的话可以运行 `uv cache clean` 删掉。
- 运行时内存：服务本身约 0.6GB，大部分是语音识别模型。用 Claude Code 对话时，每台连着的设备多一个 Claude Code 进程，约 0.26GB；对话结束、空闲超时后这个进程会退出。

### 环境变量

| 变量 | 作用 |
| --- | --- |
| `UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/` | 使用国内 Python 包镜像 |
| `HF_ENDPOINT=https://hf-mirror.com` | 使用国内 Hugging Face 镜像下载语音识别模型 |
| `ASSUME_YES=1` | 非交互安装，所有询问都按默认回答 |
| `SKIP_DOCTOR=1` | 启动时跳过自检 |
| `XIAOZHI_HUB_DIR` | 一键安装时的安装目录，默认 `~/xiaozhi-hub-server` |

一键安装时，环境变量要写在 `bash` 前面，例如：

```bash
curl -fsSL https://raw.githubusercontent.com/justhalfbit/xiaozhi-hub-server/main/scripts/bootstrap.sh | HF_ENDPOINT=https://hf-mirror.com bash
```

## 安全提示

- 服务**只应在局域网内使用**，不要暴露到公网。
- 管理后台一定要设置密码；API Key 不要提交到 Git。
- Claude Code 默认只开放只读工具。在后台放开写文件、执行命令后，语音指令就能修改文件、在电脑上执行命令，请谨慎。读写文件限制在工作目录里，但执行的命令（Auto 级别）不受这个限制。

## 致谢与许可

本项目使用 [MIT License](../LICENSE)。

- **原项目**：基于 [xinnan-tech/xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server) 改造，MIT License，Copyright (c) 2025 xinnan-tech。
- **设备端协议与固件**：[78/xiaozhi-esp32](https://github.com/78/xiaozhi-esp32)。
- **语音识别模型 X-ASR**（默认）：来自 [Gilgamesh-J/X-ASR](https://github.com/Gilgamesh-J/X-ASR)，Apache-2.0。使用的 int8 版由 [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 转换（Apache-2.0），从 Hugging Face 的 `csukuangfj2/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03` 下载。
- **语音识别模型 SenseVoiceSmall**（可选）：来自 [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)（阿里巴巴通义实验室），按 [FunASR 模型开源许可协议](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE) 使用。int8 版由 sherpa-onnx 转换，从 ModelScope 的 `pengzhendong/sherpa-onnx-sense-voice-zh-en-ja-ko-yue` 下载；只在把语音识别切换为 SherpaASR 时才会下载。
- 模型都由安装脚本或程序下载，本仓库不包含模型权重。
- **语音活动检测 silero-vad**：[snakers4/silero-vad](https://github.com/snakers4/silero-vad)，MIT License。
- **Claude Code 接入**：使用 Anthropic 的 [claude-agent-sdk](https://pypi.org/project/claude-agent-sdk/)。需要用户自行安装 Claude Code 并登录。
- **测试页里的 Live2D 核心库**：`main/digital-human/js/live2d/live2dcubismcore.min.js`，属于 Live2D Inc. 的专有许可，随原项目一起提供，本项目没有修改。
