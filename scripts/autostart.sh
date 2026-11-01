#!/usr/bin/env bash
# xiaozhi-hub-server 开机自启（macOS / Linux）
# 用法：bash scripts/autostart.sh enable|disable|restart|status
#   enable   设置开机自启，并马上在后台启动服务
#   disable  取消开机自启，并停止后台服务
#   restart  重启后台服务（更新后、改了需要重启的配置后）
#   status   查看是否已开启、是否在运行。退出码：0 开着且指向这份安装；1 没开；3 开着但指向另一份安装
# 一个用户只有一个自启文件。装了好几份时，它只属于最后一次运行 enable 的那份；
# 别的那份运行 enable 会改成指向自己，disable、restart 不会动不属于自己的自启。
# macOS 用 launchd（~/Library/LaunchAgents），登录后启动；Linux 用 systemd 用户服务。
# 服务的日志照常写在 main/xiaozhi-server/tmp/server.log。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="$ROOT/main/xiaozhi-server"
LABEL="com.xiaozhi-hub.server"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UNIT="xiaozhi-hub.service"
UNIT_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/$UNIT"
# 启动失败时的报错（日志系统起来之前的异常）；正常运行的日志在 tmp/server.log
ERR_LOG="$SERVER/tmp/autostart-error.log"

info() { printf "    %s\n" "$*"; }
fail() { printf "\033[31m✗ %s\033[0m\n" "$*" >&2; exit 1; }

# 服务的 HTTP 端口（管理后台、OTA 用的那个）
hub_port() {
  (cd "$SERVER" && .venv/bin/python -c 'from core.admin import config_store; print(int((config_store.load_merged().get("server") or {}).get("http_port", 8003)))' 2>/dev/null) || echo 8003
}
ota_up() { curl -s -m 2 -o /dev/null "http://127.0.0.1:$1/xiaozhi/ota/"; }

# 等服务真正能访问（启动时要加载语音识别模型，一般十几秒），起不来就把报错显示出来。
# 最多等 HUB_READY_TIMEOUT 秒（默认 90；update.sh 等得更久，起不来要退回原版本）
wait_ready() {
  local port="$1" i limit="${HUB_READY_TIMEOUT:-90}"
  command -v curl >/dev/null 2>&1 || return 0
  info "等待服务启动 ..."
  for ((i = 0; i < limit / 2; i++)); do
    ota_up "$port" && { info "服务已启动：http://127.0.0.1:$port/admin"; return 0; }
    sleep 2
  done
  printf "\033[31m✗ 服务 ${limit} 秒内没有启动成功。\033[0m 最近的报错：\n" >&2
  if [[ "$P" == mac ]]; then tail -n 15 "$ERR_LOG" >&2 2>/dev/null || true
  else journalctl --user -u "$UNIT" -n 15 --no-pager >&2 2>/dev/null || true; fi
  info "完整日志：$SERVER/tmp/server.log" >&2
  return 1
}

xml_escape() { local s="${1//&/&amp;}"; s="${s//</&lt;}"; printf '%s' "${s//>/&gt;}"; }

# 自启文件指向的安装目录（从里面写的 start.sh 路径取出来）
mac_owner() {
  local s
  s="$(sed -n 's#.*<string>\(.*\)/scripts/start\.sh</string>.*#\1#p' "$PLIST" 2>/dev/null | head -n 1 || true)"
  s="${s//&lt;/<}"; s="${s//&gt;/>}"; printf '%s' "${s//&amp;/&}"
}
linux_owner() { sed -n 's#^ExecStart=/bin/bash "\(.*\)/scripts/start\.sh"$#\1#p' "$UNIT_FILE" 2>/dev/null | head -n 1 || true; }
# 比较前换成真实路径，符号链接、/tmp 和 /private/tmp 这类写法不同也算同一个目录
canon() { (cd "$1" 2>/dev/null && pwd -P) || printf '%s' "$1"; }
# 开着自启，而且指向的就是这份安装
mine() { "${P}_enabled" && [[ "$(canon "$("${P}_owner")")" == "$(canon "$ROOT")" ]]; }

# ---------- macOS：launchd ----------
mac_enabled() { [[ -f "$PLIST" ]]; }
mac_pid() { launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | awk '$1 == "pid" {print $3; exit}'; }

mac_enable() {
  mkdir -p "$(dirname "$PLIST")" "$SERVER/tmp"
  # launchd 的 PATH 只有系统目录，找不到 Homebrew 的 ffmpeg 和 ~/.local/bin 里的 claude，所以带上当前终端的 PATH
  cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$(xml_escape "$ROOT/scripts/start.sh")</string>
  </array>
  <key>WorkingDirectory</key><string>$(xml_escape "$SERVER")</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>$(xml_escape "$PATH")</string>
    <key>SKIP_DOCTOR</key><string>1</string>
    <key>HUB_AUTOSTART</key><string>1</string>
    <key>PYTHONUTF8</key><string>1</string>
    <key>AUTOSTART_ERR_LOG</key><string>$(xml_escape "$ERR_LOG")</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <!-- 异常退出时自动重启，间隔至少 30 秒 -->
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>/dev/null</string>
  <key>StandardErrorPath</key><string>$(xml_escape "$ERR_LOG")</string>
</dict>
</plist>
EOF
  mac_bootout
  launchctl enable "gui/$(id -u)/$LABEL"
  # 刚移除的同名服务偶尔还没完全清掉，注册会报 "Bootstrap failed: 5"，稍等重试
  local i
  for ((i = 0; i < 5; i++)); do
    launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null && return 0
    sleep 1
  done
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
}

# 移除服务并等它真正消失（launchd 是异步移除的，马上再注册同名服务会失败）
mac_bootout() {
  local i
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  for ((i = 0; i < 20; i++)); do
    launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || return 0
    sleep 0.5
  done
}

mac_disable() {
  mac_bootout
  rm -f "$PLIST"
}

mac_restart() { launchctl kickstart -k "gui/$(id -u)/$LABEL"; }

# ---------- Linux：systemd 用户服务 ----------
linux_check() {
  command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1 \
    || fail "这台电脑没有可用的 systemd 用户服务，无法自动设置开机自启。可以用 crontab 的 @reboot 或其他方式运行 bash \"$ROOT/scripts/start.sh\""
}
linux_enabled() { [[ -f "$UNIT_FILE" ]]; }
linux_pid() { local p; p="$(systemctl --user show -p MainPID --value "$UNIT" 2>/dev/null || true)"; [[ "$p" != "0" ]] && echo "$p"; }

linux_enable() {
  linux_check
  mkdir -p "$(dirname "$UNIT_FILE")"
  # systemd 的值里 % 要写成 %%；路径里有空格时整体加引号
  local path_env="${PATH//%/%%}"
  cat >"$UNIT_FILE" <<EOF
[Unit]
Description=xiaozhi-hub-server
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$SERVER
Environment="PATH=$path_env" SKIP_DOCTOR=1 HUB_AUTOSTART=1 PYTHONUTF8=1
ExecStart=/bin/bash "$ROOT/scripts/start.sh"
Restart=on-failure
RestartSec=30
StandardOutput=null

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now "$UNIT"
  # 不开 linger 的话，用户登录后服务才会启动、退出登录后停止
  local user="${USER:-$(id -un)}"
  if [[ "$(loginctl show-user "$user" -p Linger --value 2>/dev/null || true)" != "yes" ]]; then
    info "提示：现在要登录这个用户后服务才会启动。想开机就启动（不用登录），请运行："
    info "  sudo loginctl enable-linger $user"
  fi
}

linux_disable() {
  if command -v systemctl >/dev/null 2>&1; then systemctl --user disable --now "$UNIT" 2>/dev/null || true; fi
  rm -f "$UNIT_FILE"
  if command -v systemctl >/dev/null 2>&1; then systemctl --user daemon-reload 2>/dev/null || true; fi
}

linux_restart() { systemctl --user restart "$UNIT"; }

# ---------- 命令 ----------
case "$(uname -s)" in
  Darwin) P=mac ;;
  Linux) P=linux ;;
  *) fail "不支持的系统：$(uname -s)" ;;
esac

case "${1:-}" in
  enable)
    [[ -x "$SERVER/.venv/bin/python" ]] || fail "还没安装，请先运行：bash \"$ROOT/scripts/install.sh\""
    # macOS 的隐私保护默认不让后台服务读"文稿""桌面""下载"和 iCloud 里的文件，放在这些目录里很可能启动失败
    if [[ "$P" == mac ]]; then
      case "$ROOT/" in
        "$HOME/Documents/"* | "$HOME/Desktop/"* | "$HOME/Downloads/"* | "$HOME/Library/Mobile Documents/"*)
          info "⚠️  安装目录在\"文稿/桌面/下载/iCloud\"里，macOS 默认不允许后台服务读取这些目录，服务很可能启动失败。"
          info "    启动失败的话，请把 $ROOT 移到别处（例如 ~/xiaozhi-hub-server）后重新开启" ;;
      esac
    fi
    port="$(hub_port)"
    # 已经开启过的先停掉（包括指向另一份安装的），再看端口：还被占用说明有手动启动的服务在运行，两个会抢同一个端口
    if "${P}_enabled"; then
      mine || info "开机自启原来指向另一份安装（$("${P}_owner")），改成指向这一份，并停掉那份的后台服务"
      "${P}_disable"
    fi
    if ota_up "$port"; then
      fail "端口 $port 已被占用，可能有用 start.sh 手动启动的服务在运行。请先在它的终端按 Ctrl+C 停止，再开启开机自启"
    fi
    "${P}_enable"
    info "已设置开机自启"
    [[ "$P" == mac ]] && info "（登录这个 macOS 用户后自动启动；想开机就启动，请在 系统设置 → 用户与群组 里开启自动登录）"
    info "取消：bash \"$ROOT/scripts/autostart.sh\" disable"
    wait_ready "$port"
    ;;
  disable)
    if "${P}_enabled" && ! mine; then
      fail "开机自启属于另一份安装（$("${P}_owner")），没有改动。要取消，请在那份安装里运行 autostart.sh disable"
    fi
    "${P}_disable"
    info "已取消开机自启，后台服务已停止"
    ;;
  restart)
    "${P}_enabled" || fail "没有开启开机自启。手动启动的服务请在它的终端按 Ctrl+C 停止，再运行 bash \"$ROOT/scripts/start.sh\""
    mine || fail "开机自启属于另一份安装（$("${P}_owner")），没有重启。要改成这一份，请运行 bash \"$ROOT/scripts/autostart.sh\" enable"
    "${P}_restart"
    sleep 3  # 让旧进程先退出，免得下面连到的还是旧进程
    wait_ready "$(hub_port)"
    info "后台服务已重启"
    ;;
  status)
    if ! "${P}_enabled"; then echo "未开启开机自启"; exit 1; fi
    pid="$("${P}_pid" || true)"
    if ! mine; then
      if [[ -n "$pid" ]]; then echo "开机自启指向另一份安装（$("${P}_owner")），它的服务正在运行（进程 ${pid}）"
      else echo "开机自启指向另一份安装（$("${P}_owner")），它的服务没有在运行"; fi
      exit 3
    fi
    if [[ -n "$pid" ]]; then echo "已开启开机自启，服务正在运行（进程 ${pid}）"
    else echo "已开启开机自启，但服务没有在运行。启动失败的原因见 $ERR_LOG 和 $SERVER/tmp/server.log"; fi
    ;;
  *)
    echo "用法：bash scripts/autostart.sh enable|disable|restart|status" >&2
    exit 2
    ;;
esac
