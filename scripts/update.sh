#!/usr/bin/env bash
# xiaozhi-hub-server 更新到最新版本（macOS / Linux）
# 用法：bash scripts/update.sh
#   1. 从 GitHub 拉取最新代码（配置 data/.config.yaml、模型、虚拟环境都不在 Git 里，不受影响）
#   2. 依赖有变化时重新运行 install.sh 补装新增的依赖和模型（已有的会跳过）
#   3. 开了开机自启时重启后台服务；手动启动的服务需要自己重启
# 补装依赖失败、或新版本重启后起不来时，自动退回原来的版本（开着自启的话再重启一次），网关不会停在坏掉的状态。
# 从哪里更新：当前分支设了上游（git branch --set-upstream-to）就用上游，否则用 origin 上的同名分支。
# 管理后台的「立即更新」也是在后台运行这个脚本。
set -euo pipefail

# 整个脚本放在函数里：git 更新会改写本文件，bash 边读边执行会读到新旧混杂的内容。
# 拉取完代码后用 exec 换成新版本的本脚本继续后面的步骤
main() {
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  # install.sh 全部步骤做完时写下这个文件；没有它说明上次安装或更新中途失败了
  INSTALLED="$ROOT/main/xiaozhi-server/data/.install-complete"
  cd "$ROOT"
  if [[ "${1:-}" == "--after-pull" ]]; then after_pull "${2:-}"; return; fi

  git rev-parse --git-dir >/dev/null 2>&1 || fail "${ROOT} 不是用 git 下载的，无法自动更新。请重新运行一键安装命令，或手动下载新版本"
  local branch remote rbranch old new
  branch="$(git symbolic-ref --short -q HEAD)" || fail "当前不在任何分支上，无法自动更新"
  remote="$(git config "branch.${branch}.remote" || echo origin)"
  rbranch="$(git config "branch.${branch}.merge" || echo "$branch")"
  rbranch="${rbranch#refs/heads/}"

  # 只看受 Git 管理的文件；配置、模型、虚拟环境都被 .gitignore 忽略，不算改动
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    git status --short --untracked-files=no
    fail "上面这些文件被改过，更新会和它们冲突。请先撤销（git checkout -- <文件>）或备份后再更新"
  fi

  step "检查新版本"
  old="$(git rev-parse HEAD)"
  local shallow
  shallow="$(git rev-parse --is-shallow-repository)"
  if [[ "$shallow" == "true" ]]; then
    # 一键安装下载的是只含最新一次提交的浅克隆：只取当前版本之后的提交（接得上历史，才列得出更新内容、
    # 判断得了是不是快进）；远端不支持时退回只取最新一次提交
    git fetch --shallow-since="$(git log -1 --format=%cI)" "$remote" "$rbranch" 2>/dev/null \
      || git fetch --depth 1 "$remote" "$rbranch"
  else
    git fetch "$remote" "$rbranch"
  fi
  new="$(git rev-parse FETCH_HEAD)"
  if [[ "$old" == "$new" ]]; then
    info "已经是最新版本（$(git log -1 --format='%h，%cd' --date=short)）"
    if [[ ! -f "$INSTALLED" ]]; then
      step "补全安装（上次安装可能没有完成，或者是用早期版本的脚本装的）"
      bash "$ROOT/scripts/install.sh"
    fi
    return
  fi

  step "下载新版本"
  if [[ "$shallow" == "true" ]] && ! git merge-base HEAD FETCH_HEAD >/dev/null 2>&1; then
    # 浅克隆接不上新旧版本之间的历史，不能快进合并，直接切到新版本（上面已确认没有本地改动）
    git reset --keep FETCH_HEAD
  else
    git merge --ff-only FETCH_HEAD || fail "本地有自己的提交，无法直接更新。请手动合并：git merge ${remote}/${rbranch}"
  fi
  if [[ "$(git rev-parse HEAD)" == "$old" ]]; then
    info "本地的版本比远端还新（有远端没有的提交），不用更新"
    return
  fi
  exec bash "$ROOT/scripts/update.sh" --after-pull "$old"
}

after_pull() {
  local old="$1"
  # 只检出运行需要的目录；和 bootstrap.sh 里的列表保持一致。以后新增目录时，更新会把它检出来
  local SPARSE_PATHS=(main/xiaozhi-server scripts hub-docs)
  if [[ "$(git config --bool core.sparseCheckout || true)" == "true" ]]; then
    git sparse-checkout set "${SPARSE_PATHS[@]}"
  fi

  if [[ -n "$old" ]] && git cat-file -e "${old}^{commit}" 2>/dev/null && git merge-base --is-ancestor "$old" HEAD 2>/dev/null; then
    info "更新内容："
    # 用 -20 而不是 | head：开着 pipefail 时 head 提前退出会让 git 收到 SIGPIPE，整个脚本跟着退出
    git log -20 --format='      %h %s' "$old..HEAD"
  else
    # 浅克隆只有最新一次提交，列不出中间的改动，给出 GitHub 上的对比页
    info "已更新到：$(git log -1 --format='%h %s（%cd）' --date=short)"
    local url
    url="$(git remote get-url origin 2>/dev/null | sed -E 's#^git@github.com:#https://github.com/#; s#\.git$##' || true)"
    [[ -n "$old" && "$url" == https://github.com/* ]] && info "全部改动：$url/compare/${old:0:12}...$(git rev-parse --short=12 HEAD)"
  fi

  if [[ ! -f "$INSTALLED" ]]; then
    # 第一次安装就没装完：按完整安装走，包括最后询问是否开机自启
    step "补全安装（上次安装可能没有完成，或者是用早期版本的脚本装的）"
    bash "$ROOT/scripts/install.sh" || fail "安装没有完成。按上面的提示处理后，重新运行 bash \"$ROOT/scripts/update.sh\" 即可接着装"
    return
  fi
  local reinstalled=0
  if deps_changed "$old"; then
    reinstalled=1
    step "补装依赖和模型"
    # 先删掉完成标记：这一步失败又没能退回原版本时，下次运行 update.sh 会发现并重新补装
    rm -f "$INSTALLED"
    if ! HUB_UPDATING=1 bash "$ROOT/scripts/install.sh"; then
      roll_back "$old" || fail "安装步骤失败，退回原来的版本也没有成功。按上面的提示处理后，重新运行 bash \"$ROOT/scripts/update.sh\" 即可补装"
      restore_deps || fail "安装步骤失败，已退回原来的版本，但依赖没能装回原来的版本，服务没有重启（正在运行的不受影响，下次重启可能起不来）。按上面的提示处理后，运行 bash \"$ROOT/scripts/install.sh\" 装回依赖"
      fail "安装步骤失败，已退回原来的版本（$(git log -1 --format='%h，%cd' --date=short)），服务没有重启。按上面的提示处理后再更新"
    fi
  else
    info "依赖没有变化，跳过安装"
  fi

  if bash "$ROOT/scripts/autostart.sh" status >/dev/null 2>&1; then
    step "重启后台服务"
    # 设备上启动要加载语音识别模型，慢的一两分钟；等足够久再判断新版本起不来
    if ! HUB_READY_TIMEOUT="${HUB_READY_TIMEOUT:-300}" bash "$ROOT/scripts/autostart.sh" restart; then
      step "新版本没有启动成功，退回原来的版本"
      roll_back "$old" || fail "退回原来的版本也没有成功，后台服务现在可能起不来。请按上面的报错处理"
      if [[ "$reinstalled" == 1 ]]; then
        restore_deps || fail "已退回原来的版本，但依赖没能装回原来的版本，后台服务现在可能起不来。按上面的提示处理后，运行 bash \"$ROOT/scripts/install.sh\" 装回依赖"
      fi
      HUB_READY_TIMEOUT="${HUB_READY_TIMEOUT:-300}" bash "$ROOT/scripts/autostart.sh" restart \
        || fail "已退回原来的版本，但后台服务仍然没有启动成功。请按上面的报错处理"
      fail "新版本没有启动成功，已退回原来的版本（$(git log -1 --format='%h，%cd' --date=short)）并重启。报错见上面"
    fi
    printf "\n更新完成，后台服务已重启。\n"
  else
    printf "\n更新完成。如果服务正在运行，请在它的终端按 Ctrl+C 停止，再运行 bash \"%s\" 启动新版本。\n" "$ROOT/scripts/start.sh"
  fi
}

# 这次更新有没有改到安装相关的文件（依赖清单、安装脚本、模型下载）。判断不了时按有变化处理
deps_changed() {
  local old="$1" changed
  [[ -n "$old" ]] && git cat-file -e "${old}^{commit}" 2>/dev/null || return 0
  # --no-renames：只比较文件名，不用下载文件内容（一键安装是不带文件内容的浅克隆）
  changed="$(git diff --no-renames --name-only "$old" HEAD -- \
    'main/xiaozhi-server/requirements*.txt' scripts/install.sh scripts/install_deps.py)" || return 0
  [[ -n "$changed" ]]
}

# 代码退回更新前的版本（只动受 Git 管理的文件，配置、模型、虚拟环境不受影响）
roll_back() {
  local old="$1"
  [[ -n "$old" ]] && git cat-file -e "${old}^{commit}" 2>/dev/null || return 1
  git reset -q --keep "$old"
}

# 依赖清单是写死版本的，新版本补装时可能把某些包升级了，旧代码配新依赖不一定跑得起来：
# 退回代码后在原来的版本上再补装一次，把虚拟环境装回原来的版本（uv 缓存里有原来的包，一般不用联网）
restore_deps() {
  step "把依赖装回原来的版本"
  # 装回失败时不留完成标记，下次运行 update.sh 会发现并重新补装
  rm -f "$INSTALLED"
  HUB_UPDATING=1 bash "$ROOT/scripts/install.sh"
}

step() { printf "\n\033[1m==> %s\033[0m\n" "$*"; }
info() { printf "    %s\n" "$*"; }
fail() { printf "\n\033[31m✗ %s\033[0m\n" "$*" >&2; exit 1; }

main "$@"
