#!/usr/bin/env bash
# xiaozhi-hub-server 一键安装（macOS / Linux）
# 用法：bash scripts/install.sh
#   只装默认组合需要的依赖（不含 torch），语音识别使用 sherpa-onnx 版 X-ASR 中英混合模型（约 175MB）
# 可选环境变量：
#   UV_INDEX_URL   Python 包镜像，例如 https://mirrors.aliyun.com/pypi/simple/
#   HF_ENDPOINT    Hugging Face 镜像（下载语音识别模型用），例如 https://hf-mirror.com
#   ASSUME_YES=1   所有询问都按默认回答（非交互安装）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="$ROOT/main/xiaozhi-server"
VENV="$SERVER/.venv"
PY="$VENV/bin/python"
# 默认的语音识别模型：sherpa-onnx 版 X-ASR（int8），目录与 config.yaml 中 SherpaXASR 的 model_dir 一致
XASR_NAME="sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03"
XASR_DIR="$SERVER/models/$XASR_NAME"
XASR_URL="${HF_ENDPOINT:-https://huggingface.co}/csukuangfj2/$XASR_NAME/resolve/main"
CONFIG="$SERVER/data/.config.yaml"

step() { printf "\n\033[1m==> %s\033[0m\n" "$*"; }
info() { printf "    %s\n" "$*"; }
fail() { printf "\n\033[31m✗ %s\033[0m\n" "$*" >&2; exit 1; }
# ask "问题" Y|N：返回 0 表示"是"；ASSUME_YES=1 时直接用默认值
ask() {
  local default="${2:-N}" hint="[y/N]" answer
  [[ "$default" == "Y" ]] && hint="[Y/n]"
  if [[ "${ASSUME_YES:-0}" == "1" || ! -t 0 ]]; then answer="$default"; else read -r -p "    $1 $hint " answer; fi
  answer="${answer:-$default}"
  [[ "$answer" =~ ^[Yy] ]]
}
# download <地址> <保存路径> <最小字节数>：断点续传，下载完校验大小
download() {
  local url="$1" dest="$2" min="$3"
  if (( $(file_size "$dest") >= min )); then info "已存在，跳过：$dest"; return; fi
  mkdir -p "$(dirname "$dest")"
  curl -L --fail --retry 3 -C - -o "$dest.part" "$url" || fail "下载失败，可重新运行本脚本断点续传：$url"
  (( $(file_size "$dest.part") >= min )) || fail "文件大小异常，请删除 $dest.part 后重试"
  mv "$dest.part" "$dest"
  info "已下载：$dest"
}
file_size() { "$PY" -c 'import os,sys; print(os.path.getsize(sys.argv[1]) if os.path.exists(sys.argv[1]) else 0)' "$1"; }

[[ $# -eq 0 ]] || fail "不需要参数：${*}"

OS="$(uname -s)"
[[ "$OS" == "Darwin" || "$OS" == "Linux" ]] || fail "不支持的系统：${OS}（Windows 请使用 scripts/install.ps1）"
# Claude Code 的登录信息、工作目录、开机自启都跟着当前用户走，用 root 装会和平时用的账号对不上
if [[ "$(id -u)" == "0" ]]; then
  info "⚠️  正在以 root 运行。一般请用平时登录的账号运行，不要加 sudo（需要时脚本会自己用 sudo 装系统依赖）"
fi

# ---------- 1. uv ----------
step "检查 uv（用于管理 Python 版本和虚拟环境）"
if ! command -v uv >/dev/null 2>&1; then
  [[ -x "$HOME/.local/bin/uv" ]] && export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  if ask "未找到 uv，是否用官方脚本安装？（curl -LsSf https://astral.sh/uv/install.sh | sh）" Y; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  else
    fail "请先安装 uv：https://docs.astral.sh/uv/getting-started/installation/"
  fi
fi
info "$(uv --version)"

# ---------- 2. 系统依赖：opus、ffmpeg ----------
step "检查系统依赖（opus、ffmpeg）"
if [[ "$OS" == "Darwin" ]]; then
  # 只看命令和库是否可用，不看由哪个 formula 提供（例如已装 ffmpeg-full 时不能再装 ffmpeg）
  need=()
  command -v ffmpeg >/dev/null 2>&1 || need+=(ffmpeg)
  brew_prefix="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
  [[ -n "$(ls "$brew_prefix"/lib/libopus*.dylib /usr/local/lib/libopus*.dylib 2>/dev/null)" ]] || need+=(opus)
  if (( ${#need[@]} == 0 )); then info "opus、ffmpeg 已可用"; else
    command -v brew >/dev/null 2>&1 || fail "缺少 ${need[*]}，且未找到 Homebrew。请先安装 Homebrew（https://brew.sh）或手动安装 ${need[*]}"
    if ask "需要安装：${need[*]}，执行 brew install ${need[*]}？" Y; then brew install "${need[@]}"; else
      fail "请手动安装 ${need[*]} 后重新运行本脚本"
    fi
  fi
else
  # root 下（例如容器里）通常没有 sudo，直接运行
  SUDO=sudo; [[ "$(id -u)" == "0" ]] && SUDO=""
  need=()
  command -v ffmpeg >/dev/null 2>&1 || need+=(ffmpeg)
  # Debian/Ubuntu 的普通用户 PATH 里没有 /sbin，直接调 ldconfig 会找不到命令
  # 不用 grep -q：它找到就提前退出，开着 pipefail 时 ldconfig 收到 SIGPIPE，会被误判成没装
  if ! ({ ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null; } | grep libopus >/dev/null); then need+=(opus); fi
  if (( ${#need[@]} == 0 )); then info "opus、ffmpeg 已安装"; else
    if command -v apt-get >/dev/null 2>&1; then
      pkgs=(); for p in "${need[@]}"; do [[ "$p" == opus ]] && pkgs+=(libopus0) || pkgs+=(ffmpeg); done
      # 新装的系统可能还没有软件包列表，先 update，否则会报"找不到软件包"
      cmd=($SUDO sh -c "apt-get update && apt-get install -y ${pkgs[*]}")
    elif command -v dnf >/dev/null 2>&1; then
      pkgs=(); for p in "${need[@]}"; do [[ "$p" == opus ]] && pkgs+=(opus) || pkgs+=(ffmpeg); done
      cmd=($SUDO dnf install -y "${pkgs[@]}")
    elif command -v pacman >/dev/null 2>&1; then
      cmd=($SUDO pacman -S --noconfirm "${need[@]}")
    else
      fail "无法自动安装 ${need[*]}，请用系统的包管理器安装后重新运行本脚本"
    fi
    info "需要安装：${need[*]}"
    if ask "执行 ${cmd[*]}？" Y; then "${cmd[@]}"; else fail "请手动安装 ${need[*]} 后重新运行本脚本"; fi
  fi
fi

# ---------- 3. Python 3.10 虚拟环境 ----------
step "准备 Python 3.10 虚拟环境"
if [[ -x "$PY" ]] && "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)'; then
  info "已存在：$VENV"
else
  uv venv --python 3.10 "$VENV"
fi

# ---------- 4. Python 依赖 ----------
step "安装 Python 依赖"
"$PY" "$ROOT/scripts/install_deps.py" --python "$PY" || fail "Python 依赖安装失败"

# ---------- 5. 语音识别模型 ----------
step "下载语音识别模型 X-ASR（sherpa-onnx int8 版，约 175MB；国内下载慢可设置 HF_ENDPOINT=https://hf-mirror.com）"
download "$XASR_URL/encoder-epoch-99-avg-1.int8.onnx" "$XASR_DIR/encoder-epoch-99-avg-1.int8.onnx" 150000000
download "$XASR_URL/decoder-epoch-99-avg-1.onnx" "$XASR_DIR/decoder-epoch-99-avg-1.onnx" 10000000
download "$XASR_URL/joiner-epoch-99-avg-1.int8.onnx" "$XASR_DIR/joiner-epoch-99-avg-1.int8.onnx" 2000000
download "$XASR_URL/tokens.txt" "$XASR_DIR/tokens.txt" 50000
download "$XASR_URL/bpe.model" "$XASR_DIR/bpe.model" 100000

# ---------- 6. 配置文件 ----------
step "生成配置文件"
CLAUDE_BIN="$(command -v claude || true)"
[[ -z "$CLAUDE_BIN" && -x "$HOME/.local/bin/claude" ]] && CLAUDE_BIN="$HOME/.local/bin/claude"
[[ -z "$CLAUDE_BIN" && -x "$HOME/.claude/local/claude" ]] && CLAUDE_BIN="$HOME/.claude/local/claude"
if [[ -f "$CONFIG" ]]; then
  info "已存在，保持不变：$CONFIG"
else
  mkdir -p "$(dirname "$CONFIG")"
  cp "$ROOT/scripts/config.template.yaml" "$CONFIG"
  chmod 600 "$CONFIG"
  # 本机装了 Claude Code 就直接用它；没装时用管理后台厂商列表里的第一个：Claude API，需要填 API Key
  if [[ -n "$CLAUDE_BIN" ]]; then
    info "大模型：本机 Claude Code"
  else
    "$PY" - "$CONFIG" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
s = re.sub(r"(\n  LLM: )ClaudeCLI[^\n]*", r"\1ClaudeAPI          # 在管理后台填写 API Key", s, count=1)
open(p, "w", encoding="utf-8").write(s)
PYEOF
    info "大模型：Claude API（没找到本机的 Claude Code）。请在管理后台填写 Claude API Key，或者换成别的厂商"
  fi
  info "已生成：$CONFIG"
fi

# 旧配置里语音识别是 FunASR 或 SenseVoice 时，询问是否切换到默认的 X-ASR（中英混合更准，速度一样）
ASR_NOW="$(cd "$SERVER" && "$PY" -c 'from core.admin import config_store; m = config_store.load_merged(); print(m["selected_module"]["ASR"], config_store.entry_type(m, "ASR"))' 2>/dev/null || true)"
if [[ "$ASR_NOW" == *" fun_local" || "$ASR_NOW" == "SherpaASR "* ]] \
  && ask "当前的语音识别是 ${ASR_NOW%% *}。切换到 X-ASR（中英混合识别更准，速度一样）？" Y; then
  (cd "$SERVER" && "$PY" -c 'from core.admin import config_store; config_store.select_entry("ASR", "SherpaXASR", config_store.load_merged())' 2>/dev/null) \
    && info "语音识别已切换为 SherpaXASR" || info "切换失败，请手动把 data/.config.yaml 里的 selected_module.ASR 改成 SherpaXASR"
fi

# ---------- 7. Claude Code ----------
step "检查 Claude Code"
if [[ -n "$CLAUDE_BIN" ]]; then
  info "$("$CLAUDE_BIN" --version 2>/dev/null || echo '无法获取版本')（${CLAUDE_BIN}）"
  info "如果还没登录，请在终端运行 claude 并按提示登录（安装脚本不会代替你登录）"
else
  info "未找到 Claude Code。如需使用，请先安装：https://docs.claude.com/claude-code ，安装并登录后在管理后台切换到 ClaudeCLI"
fi
mkdir -p "$HOME/xiaozhi-workspace"

# ---------- 8. 自检 ----------
step "环境自检"
"$PY" "$ROOT/scripts/doctor.py" --skip-ports 2>/dev/null || info "自检发现异常，请按上面的提示处理后再启动"

HTTP_PORT="$("$PY" - "$SERVER" <<'PYEOF'
import sys, yaml
from pathlib import Path
d = Path(sys.argv[1])
port = 8003
for f in (d / "config.yaml", d / "data" / ".config.yaml"):
    try:
        port = ((yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("server") or {}).get("http_port", port)
    except Exception:
        pass
print(port)
PYEOF
)"

# 安装步骤全部做完，留个标记：update.sh 看到没有这个文件时，会接着把安装做完
: >"$SERVER/data/.install-complete"

# update.sh 调用时到这里就结束：重启服务由 update.sh 处理
[[ "${HUB_UPDATING:-0}" == "1" ]] && exit 0

# ---------- 9. 开机自启 ----------
step "开机自启"
AUTOSTART=0
enable_autostart() {
  if bash "$ROOT/scripts/autostart.sh" enable; then AUTOSTART=1
  elif bash "$ROOT/scripts/autostart.sh" status >/dev/null 2>&1; then
    # 已经开启，只是这次没启动成功（会每 30 秒自动重试）
    AUTOSTART=1
    info "开机自启已开启，但服务暂时没有启动成功，请按上面的报错处理"
  else
    info "设置失败，可以稍后运行 bash \"$ROOT/scripts/autostart.sh\" enable 重试"
  fi
}
# status 的退出码：0 开着且指向这份安装；1 没开；3 开着但指向另一份安装（例如以前装在别的目录）
AUTOSTART_STATE=0
bash "$ROOT/scripts/autostart.sh" status >/dev/null 2>&1 || AUTOSTART_STATE=$?
if [[ "$AUTOSTART_STATE" == "0" ]]; then
  # 重新运行安装脚本（例如补装依赖）后，让后台服务用上新装的依赖
  AUTOSTART=1
  if bash "$ROOT/scripts/autostart.sh" restart >/dev/null; then info "已开启开机自启，后台服务已重启"; else
    info "已开启开机自启，但后台服务重启失败，请运行 bash \"$ROOT/scripts/autostart.sh\" status 查看"
  fi
elif [[ "$AUTOSTART_STATE" == "3" ]]; then
  info "$(bash "$ROOT/scripts/autostart.sh" status 2>/dev/null || true)"
  if ask "开机自启只能有一份。是否改成这一份？（会停掉那份的后台服务）" Y; then enable_autostart
  else info "保持不变。以后想改成这一份：bash \"$ROOT/scripts/autostart.sh\" enable"; fi
elif ask "是否设置开机自启？（服务在后台运行，开机登录后自动启动，不用每次手动运行 start.sh）" Y; then
  enable_autostart
else
  info "没有设置。以后想开启：bash \"$ROOT/scripts/autostart.sh\" enable"
fi

# 用的是 API 模型、还没填 Key 时，提醒去后台填
LLM_HINT="$(cd "$SERVER" && "$PY" -c 'from core.admin import config_store as c; m = c.load_merged(); n = m["selected_module"]["LLM"]; e = m["LLM"][n]; print(f"\n  大模型 {n} 还没填 API Key：打开管理后台的\"大模型\"页填写后才能对话。" if c.entry_type(m, "LLM") != "claude_cli" and any(c.is_placeholder(e.get(k)) for k in e if c.SECRET_FIELD.search(k)) else "")' 2>/dev/null || true)"

if [[ "$AUTOSTART" == "1" ]]; then
  START_HINT="服务在后台运行，开机登录后会自动启动。查看状态：bash \"$ROOT/scripts/autostart.sh\" status"
else
  START_HINT="启动服务：bash \"$ROOT/scripts/start.sh\""
fi

cat <<EOF

安装完成。安装目录：$ROOT

  ${START_HINT}
  请尽快打开管理后台设置密码：本机用 http://127.0.0.1:${HTTP_PORT}/admin，其他电脑用 http://<这台电脑的 IP>:${HTTP_PORT}/admin。
  在你设置之前，局域网里谁先打开后台谁就能设置。${LLM_HINT}
  以后更新到最新版本：bash "$ROOT/scripts/update.sh"
  卸载（会一起取消开机自启）：bash "$ROOT/scripts/uninstall.sh"

提醒：
  - 请在路由器里给这台电脑设置固定 IP，否则 IP 变了设备会连不上。
  - 服务只应在可信的局域网内使用，不要暴露到公网；远程使用请走 Tailscale、WireGuard 这类私人组网。
    语音连接不认证，同一局域网的人能冒充设备使用你的模型、读写聊天记录和记忆、指挥 Claude Code 和 Grok Bot。
EOF
