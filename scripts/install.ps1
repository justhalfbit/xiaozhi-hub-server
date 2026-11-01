# xiaozhi-hub-server 一键安装（Windows PowerShell）
# 用法（在项目根目录）：powershell -ExecutionPolicy Bypass -File scripts\install.ps1
#   只装默认组合需要的依赖（不含 torch），语音识别使用 sherpa-onnx 版 X-ASR 中英混合模型（约 175MB）
# 可选环境变量：$env:UV_INDEX_URL（Python 包镜像），$env:HF_ENDPOINT（Hugging Face 镜像，如 https://hf-mirror.com），$env:ASSUME_YES = "1"（非交互）
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Server = Join-Path $Root "main\xiaozhi-server"
$Venv = Join-Path $Server ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
# 默认的语音识别模型：sherpa-onnx 版 X-ASR（int8），目录与 config.yaml 中 SherpaXASR 的 model_dir 一致
$XasrName = "sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03"
$XasrDir = Join-Path $Server "models\$XasrName"
$HfEndpoint = if ($env:HF_ENDPOINT) { $env:HF_ENDPOINT } else { "https://huggingface.co" }
$XasrUrl = "$HfEndpoint/csukuangfj2/$XasrName/resolve/main"
$Config = Join-Path $Server "data\.config.yaml"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Info($msg) { Write-Host "    $msg" }
function Fail($msg) { Write-Host "`n✗ $msg" -ForegroundColor Red; exit 1 }
function Ask($question, $default = "N") {
    if ($env:ASSUME_YES -eq "1") { $answer = $default }
    else {
        $hint = if ($default -eq "Y") { "[Y/n]" } else { "[y/N]" }
        $answer = Read-Host "    $question $hint"
        if (-not $answer) { $answer = $default }
    }
    return $answer -match "^[Yy]"
}
# PowerShell 5.1 在 $ErrorActionPreference = "Stop" 时，会把原生命令写到 stderr 的内容（即使 2>$null）当成错误直接终止脚本。
# 预期可能失败、只看退出码的命令用它运行
function Quiet([scriptblock]$block) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & $block 2>$null } finally { $ErrorActionPreference = $old }
}
function Has($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
# 断点续传下载，下载完校验大小
function Download($url, $dest, $minBytes) {
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -ge $minBytes)) { Info "已存在，跳过：$dest"; return }
    New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
    $part = "$dest.part"
    if (Has "curl.exe") { curl.exe -L --fail --retry 3 -C - -o $part $url }
    else { Invoke-WebRequest -Uri $url -OutFile $part }
    if (-not (Test-Path $part) -or (Get-Item $part).Length -lt $minBytes) { Fail "下载失败或文件大小异常，请重新运行本脚本：$url" }
    Move-Item -Force $part $dest
    Info "已下载：$dest"
}

# ---------- 1. uv ----------
Step "检查 uv（用于管理 Python 版本和虚拟环境）"
if (-not (Has "uv")) {
    $userBin = Join-Path $env:USERPROFILE ".local\bin"
    if (Test-Path (Join-Path $userBin "uv.exe")) { $env:Path = "$userBin;$env:Path" }
}
if (-not (Has "uv")) {
    if (Ask "未找到 uv，是否用官方脚本安装？" "Y") {
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        $env:Path = "$(Join-Path $env:USERPROFILE '.local\bin');$env:Path"
    } else { Fail "请先安装 uv：https://docs.astral.sh/uv/getting-started/installation/" }
}
Info (uv --version)

# ---------- 2. ffmpeg ----------
Step "检查 ffmpeg"
if (Has "ffmpeg") { Info "ffmpeg 已安装" }
elseif (Has "winget") {
    if (Ask "未找到 ffmpeg，是否用 winget 安装（Gyan.FFmpeg）？" "Y") {
        winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
        # winget 把 ffmpeg 加进了系统和用户的 PATH，但当前窗口的 PATH 还是旧的，重新读一次
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User") + ";" + $env:Path
        if (Has "ffmpeg") { Info "ffmpeg 已安装" } else { Info "ffmpeg 装好了但当前窗口还找不到，请关掉 PowerShell 重新打开后再运行本脚本" }
    } else { Fail "请手动安装 ffmpeg 并加入 PATH 后重新运行本脚本" }
} else { Fail "未找到 ffmpeg 和 winget，请手动安装 ffmpeg 并加入 PATH：https://www.gyan.dev/ffmpeg/builds/" }

# ---------- 3. Python 3.10 虚拟环境 ----------
Step "准备 Python 3.10 虚拟环境"
if ((Test-Path $Py) -and ((& $Py -c "import sys; print(sys.version_info[:2] == (3, 10))") -eq "True")) {
    Info "已存在：$Venv"
} else { uv venv --python 3.10 $Venv }

# ---------- 4. Python 依赖 ----------
Step "安装 Python 依赖"
& $Py (Join-Path $Root "scripts\install_deps.py") --python $Py
if ($LASTEXITCODE -ne 0) { Fail "Python 依赖安装失败" }

# ---------- 5. opus ----------
Step "检查 opus"
# opuslib_next 在 PATH 里找 opus.dll。我们把它放在虚拟环境的 Scripts 目录里，安装和启动（start.ps1）时都把这个目录加进 PATH
$VenvScripts = Join-Path $Venv "Scripts"
$env:Path = "$VenvScripts;$env:Path"
function Show-OpusHelp {
    Info "请手动准备 opus.dll，任选一种："
    Info "  1) 安装 Anaconda/Miniconda 后执行：conda install -c conda-forge libopus，再把 opus.dll 复制到 $VenvScripts"
    Info "  2) 下载 Windows 版 opus.dll，放到 $VenvScripts"
    Info "处理后重新运行本脚本，或直接运行 scripts\start.ps1 查看自检结果"
}
Quiet { & $Py -c "import opuslib_next" }
if ($LASTEXITCODE -eq 0) { Info "opus 可用" }
else {
    # discord.py 的官方安装包里带了 Windows 版 libopus 1.3.1（64 位和 32 位，BSD 许可）。按哈希固定版本下载，只取出 DLL
    $arch = (& $Py -c "import platform; print(platform.machine())" | Out-String).Trim()
    $target = @{ "AMD64" = "x64"; "x86" = "x86" }[$arch]
    if (-not $target) { Info "未找到 opus.dll，当前 Python 是 $arch 版本，没有现成的 opus.dll 可用。"; Show-OpusHelp }
    elseif (Ask "未找到 opus.dll，是否自动下载一份（取自 discord.py 官方安装包，约 1MB，会校验文件哈希）？" "Y") {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("xiaozhi-opus-" + [guid]::NewGuid())
        $req = "$tmp.txt"
        Set-Content -Path $req -Value "discord.py==2.7.1 --hash=sha256:849dca2c63b171146f3a7f3f8acc04248098e9e6203412ce3cf2745f284f7439" -Encoding ASCII
        $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        uv pip install --python $Py --no-deps --require-hashes --target $tmp -r $req
        $ErrorActionPreference = $old
        $dll = Join-Path $tmp "discord\bin\libopus-0.$target.dll"
        if (Test-Path $dll) {
            Copy-Item $dll (Join-Path $VenvScripts "opus.dll") -Force
            Copy-Item (Join-Path $tmp "discord\bin\COPYING") (Join-Path $VenvScripts "opus-COPYING.txt") -Force
        }
        Remove-Item -Recurse -Force $tmp, $req -ErrorAction SilentlyContinue
        Quiet { & $Py -c "import opuslib_next" }
        if ($LASTEXITCODE -eq 0) { Info "opus 可用（$VenvScripts\opus.dll）" }
        else { Info "opus.dll 自动下载失败。如果是网络问题，可以设置 `$env:UV_INDEX_URL 使用国内镜像后重新运行。"; Show-OpusHelp }
    } else { Show-OpusHelp }
}

# ---------- 6. 语音识别模型 ----------
Step "下载语音识别模型 X-ASR（sherpa-onnx int8 版，约 175MB；国内下载慢可设置 `$env:HF_ENDPOINT = 'https://hf-mirror.com'）"
Download "$XasrUrl/encoder-epoch-99-avg-1.int8.onnx" (Join-Path $XasrDir "encoder-epoch-99-avg-1.int8.onnx") 150000000
Download "$XasrUrl/decoder-epoch-99-avg-1.onnx" (Join-Path $XasrDir "decoder-epoch-99-avg-1.onnx") 10000000
Download "$XasrUrl/joiner-epoch-99-avg-1.int8.onnx" (Join-Path $XasrDir "joiner-epoch-99-avg-1.int8.onnx") 2000000
Download "$XasrUrl/tokens.txt" (Join-Path $XasrDir "tokens.txt") 50000
Download "$XasrUrl/bpe.model" (Join-Path $XasrDir "bpe.model") 100000

# ---------- 7. 配置文件 ----------
Step "生成配置文件"
$claude = Get-Command claude.exe -ErrorAction SilentlyContinue
if (-not $claude) { $claude = Get-Command claude -ErrorAction SilentlyContinue }
if (-not $claude) { $claude = Get-Command (Join-Path $env:USERPROFILE ".local\bin\claude.exe") -ErrorAction SilentlyContinue }
if (Test-Path $Config) { Info "已存在，保持不变：$Config" }
else {
    New-Item -ItemType Directory -Force -Path (Split-Path $Config) | Out-Null
    $text = Get-Content (Join-Path $Root "scripts\config.template.yaml") -Raw -Encoding UTF8
    # 本机装了 Claude Code 就直接用它；没装时用管理后台厂商列表里的第一个：Claude API，需要填 API Key
    if ($claude) { Info "大模型：本机 Claude Code" }
    else {
        $text = $text -replace "(\n  LLM: )ClaudeCLI[^\r\n]*", '${1}ClaudeAPI          # 在管理后台填写 API Key'
        Info "大模型：Claude API（没找到本机的 Claude Code）。请在管理后台填写 Claude API Key，或者换成别的厂商"
    }
    [System.IO.File]::WriteAllText($Config, $text, (New-Object System.Text.UTF8Encoding $false))
    Info "已生成：$Config"
}

# 旧配置里语音识别是 FunASR 或 SenseVoice 时，询问是否切换到默认的 X-ASR（中英混合更准，速度一样）
Push-Location $Server
$asrNow = Quiet { & $Py -c "from core.admin import config_store; m = config_store.load_merged(); print(m['selected_module']['ASR'], config_store.entry_type(m, 'ASR'))" }
if ($asrNow) {
    $asrName, $asrType = $asrNow -split " ", 2
    if (($asrType -eq "fun_local" -or $asrName -eq "SherpaASR") -and (Ask "当前的语音识别是 $asrName。切换到 X-ASR（中英混合识别更准，速度一样）？" "Y")) {
        Quiet { & $Py -c "from core.admin import config_store; config_store.select_entry('ASR', 'SherpaXASR', config_store.load_merged())" }
        if ($LASTEXITCODE -eq 0) { Info "语音识别已切换为 SherpaXASR" } else { Info "切换失败，请手动把 data\.config.yaml 里的 selected_module.ASR 改成 SherpaXASR" }
    }
}
Pop-Location

# ---------- 8. Claude Code ----------
Step "检查 Claude Code"
if ($claude) {
    Info "$(& $claude.Source --version)（$($claude.Source)）"
    Info "如果还没登录，请在终端运行 claude 并按提示登录（安装脚本不会代替你登录）"
} else { Info "未找到 Claude Code。如需使用，请先安装并登录，再在管理后台切换到 ClaudeCLI" }
New-Item -ItemType Directory -Force -Path (Join-Path $env:USERPROFILE "xiaozhi-workspace") | Out-Null

# ---------- 9. 自检 ----------
Step "环境自检"
& $Py (Join-Path $Root "scripts\doctor.py") --skip-ports
if ($LASTEXITCODE -ne 0) { Info "自检发现异常，请按上面的提示处理后再启动" }

$HttpPort = 8003; $WsPort = 8000
foreach ($f in @((Join-Path $Server "config.yaml"), $Config)) {
    $t = Get-Content $f -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
    if ($t -match "(?m)^\s+http_port:\s*(\d+)") { $HttpPort = $Matches[1] }
    if ($t -match "(?m)^server:[\s\S]*?^\s+port:\s*(\d+)") { $WsPort = $Matches[1] }
}

# 用的是 API 模型、还没填 Key 时，提醒去后台填
# 注意：PowerShell 把中文弯引号（“”‘’）当成普通引号，写在字符串里会把字符串截断、整个脚本解析失败，引用文字用「」
Push-Location $Server
# Python 只输出模型名（ASCII），中文提示在这里拼，避免控制台编码不同导致乱码
$NoKey = Quiet { & $Py -c "from core.admin import config_store as c; m = c.load_merged(); n = m['selected_module']['LLM']; e = m['LLM'][n]; print(n if c.entry_type(m, 'LLM') != 'claude_cli' and any(c.is_placeholder(e.get(k)) for k in e if c.SECRET_FIELD.search(k)) else '')" }
Pop-Location
$NoKey = ($NoKey | Out-String).Trim()
$LlmHint = if ($NoKey) { "  大模型 $NoKey 还没填 API Key：打开管理后台的「大模型」页填写后才能对话。" } else { "" }

Write-Host @"

安装完成。

  启动服务：  powershell -ExecutionPolicy Bypass -File scripts\start.ps1
  卸载：      先 cd ~ 离开项目目录，再运行 powershell -ExecutionPolicy Bypass -File "$Root\scripts\uninstall.ps1"
  请尽快打开管理后台设置密码：本机用 http://127.0.0.1:$HttpPort/admin，其他电脑用 http://<这台电脑的 IP>:$HttpPort/admin。
  在你设置之前，局域网里谁先打开后台谁就能设置。
$LlmHint

提醒：
  - 请在路由器里给这台电脑设置固定 IP，否则 IP 变了设备会连不上。
  - 请在 Windows 防火墙中放行 TCP 端口 ${WsPort} 和 ${HttpPort}（仅限专用网络）。
  - 服务只应在可信的局域网内使用，不要暴露到公网；远程使用请走 Tailscale、WireGuard 这类私人组网。
    语音连接不认证，同一局域网的人能冒充设备使用你的模型、读写聊天记录和记忆、指挥 Claude Code 和 Grok Bot。
"@
