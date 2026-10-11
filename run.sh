#!/usr/bin/env bash
# PhotoLab — Linux / WSL 启动脚本（run.bat 的 Linux 版本）
#
# 用法：
#   ./run.sh                 启动（自动建虚拟环境 + 装依赖）
#   ./run.sh 图片路径        启动并直接打开图片
#   ./run.sh --raw           同时安装 rawpy（RAW 支持，可选）
#   ./run.sh --deps          自动安装缺失的系统依赖（需要 sudo）
#   ./run.sh --mirror         pip 换成国内源（清华，写入 venv 后长期生效）
#   ./run.sh --mirror aliyun 指定国内源：tsinghua|aliyun|ustc|tencent|official
#   ./run.sh --help          显示帮助
#
# WSL 提示：项目在 /mnt/d/workspace/photolab（注意 Linux 区分大小写，是小写 d）；
#           无图形环境（WSL 未开 WSLg）时自动切换到 offscreen，仍可处理/保存图片。

set -euo pipefail
cd "$(dirname "$0")"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[1;33m[提示] %s\033[0m\n' "$1"; }
die()  { printf '\033[1;31m[错误] %s\033[0m\n' "$1" >&2; exit 1; }

mirror_url() {
    case "$1" in
        tsinghua) echo "https://pypi.tuna.tsinghua.edu.cn/simple" ;;
        aliyun)   echo "https://mirrors.aliyun.com/pypi/simple" ;;
        ustc)     echo "https://pypi.mirrors.ustc.edu.cn/simple" ;;
        tencent)  echo "https://mirrors.cloud.tencent.com/pypi/simple" ;;
        official) echo "https://pypi.org/simple" ;;
    esac
}

PY="${PYTHON:-python3}"
VENV=".venv"
WANT_RAW=0
AUTO_DEPS=0
MIRROR=""
ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --raw)      WANT_RAW=1 ;;
        --deps)     AUTO_DEPS=1 ;;
        --mirror)
            MIRROR="${2:-tsinghua}"
            case "$MIRROR" in
                tsinghua|aliyun|ustc|tencent|official) ;;
                *) die "未知镜像源：$MIRROR（可选 tsinghua|aliyun|ustc|tencent|official）" ;;
            esac
            [ $# -ge 2 ] && shift || true
            ;;
        -h|--help)
            sed -n '2,/^[[:space:]]*$/p' "$0" | sed -e '${/^$/d}' -e 's/^# \{0,1\}//'
            exit 0
            ;;
        *)          ARGS+=("$1") ;;
    esac
    shift
done

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[1;33m[提示] %s\033[0m\n' "$1"; }
die()  { printf '\033[1;31m[错误] %s\033[0m\n' "$1" >&2; exit 1; }

mirror_url() {
    case "$1" in
        tsinghua) echo "https://pypi.tuna.tsinghua.edu.cn/simple" ;;
        aliyun)   echo "https://mirrors.aliyun.com/pypi/simple" ;;
        ustc)     echo "https://pypi.mirrors.ustc.edu.cn/simple" ;;
        tencent)  echo "https://mirrors.cloud.tencent.com/pypi/simple" ;;
        official) echo "https://pypi.org/simple" ;;
    esac
}

# ---------------------------------------------------------------- 1. Python
step "检查 Python"
command -v "$PY" >/dev/null 2>&1 || die "未找到 $PY，请先安装：sudo apt install python3 python3-pip python3-venv"
"$PY" --version

"$PY" -c "import venv" >/dev/null 2>&1 \
    || die "缺少 venv 模块，请执行：sudo apt install python3-venv"

# ---------------------------------------------------------------- 2. 虚拟环境
step "准备虚拟环境 ($VENV)"
if [ ! -x "$VENV/bin/python" ]; then
    "$PY" -m venv "$VENV" || die "创建虚拟环境失败（可尝试：sudo apt install python3-venv）"
    echo "已创建 $VENV"
else
    echo "已存在 $VENV"
fi

# ---------------------------------------------------------------- 3. pip 镜像源
if [ -n "$MIRROR" ]; then
    step "配置 pip 镜像源：$MIRROR"
    URL="$(mirror_url "$MIRROR")"
    # 写入 venv 内的 pip 配置，之后不带 --mirror 也依然生效
    "$VENV/bin/python" -m pip config set global.index-url "$URL" >/dev/null
    "$VENV/bin/python" -m pip config set global.trusted-host \
        "$(echo "$URL" | sed -E 's#^https?://([^/]+)/.*#\1#')" >/dev/null
    echo "index-url = $URL（已写入 $VENV/pip.conf）"
else
    "$VENV/bin/python" -m pip config unset global.index-url >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------- 4. Python 依赖
step "检查 Python 依赖"
REQS=(requirements.txt)
[ "$WANT_RAW" = "1" ] && REQS+=(rawpy)

if ! "$VENV/bin/python" -c "import PySide6, numpy, cv2" >/dev/null 2>&1; then
    echo "安装依赖中（首次需要几分钟）..."
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install "${REQS[@]}" || die "依赖安装失败（可加 --mirror 换国内源重试）"
else
    echo "Python 依赖已就绪"
fi

if [ "$WANT_RAW" = "1" ] && ! "$VENV/bin/python" -c "import rawpy" >/dev/null 2>&1; then
    "$VENV/bin/python" -m pip install rawpy || warn "rawpy 安装失败（RAW 将不可用）"
fi

# ---------------------------------------------------------------- 5. 系统库
step "检查 Qt 所需系统库"
MISSING=()
for lib in libGL.so.1 libEGL.so.1 libxkbcommon.so.0 libdbus-1.so.3 \
           libfontconfig.so.1 libglib-2.0.so.0; do
    ldconfig -p 2>/dev/null | grep -q "$lib" || MISSING+=("$lib")
done

if [ "${#MISSING[@]}" -gt 0 ]; then
    warn "缺少 Qt 运行库：${MISSING[*]}"
    echo "     Debian/Ubuntu 安装："
    echo "       sudo apt update && sudo apt install -y \\"
    echo "         libgl1 libegl1 libxkbcommon0 libdbus-1-3 libfontconfig1 libglib2.0-0"
    if [ "$AUTO_DEPS" = "1" ]; then
        sudo apt update
        sudo apt install -y libgl1 libegl1 libxkbcommon0 libdbus-1-3 \
            libfontconfig1 libglib2.0-0 || die "系统依赖安装失败"
    else
        warn "加 --deps 参数可自动安装（需要 sudo）"
    fi
else
    echo "系统库齐全"
fi

# ---------------------------------------------------------------- 6. 中文字体
step "检查中文字体"
if command -v fc-list >/dev/null 2>&1; then
    if fc-list 2>/dev/null | grep -qiE "noto.*cjk|wqy|source han|droid.*fallback"; then
        echo "已安装中文字体"
    else
        warn "未找到中文字体，界面中文会显示为方块"
        echo "     Debian/Ubuntu 安装：sudo apt install -y fonts-noto-cjk"
        [ "$AUTO_DEPS" = "1" ] && sudo apt install -y fonts-noto-cjk
    fi
else
    warn "未安装 fontconfig，Qt 可能取不到字体（中文显示为方块）"
    echo "     Debian/Ubuntu 安装：sudo apt install -y fontconfig fonts-noto-cjk"
    [ "$AUTO_DEPS" = "1" ] && sudo apt install -y fontconfig fonts-noto-cjk
fi

# ---------------------------------------------------------------- 7. 显示环境
step "启动 PhotoLab"
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
    warn "未检测到图形显示（DISPLAY/WAYLAND_DISPLAY 均为空）"
    echo "     已切换到 offscreen 模式：界面不显示，但打开/调色/保存仍可用。"
    echo "     WSL 用户可开启 WSLg，或设置 DISPLAY 后重试。"
fi

exec "$VENV/bin/python" main.py ${ARGS[@]+"${ARGS[@]}"}
