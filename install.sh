#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${CKPT_VIEWER_VENV:-"$PROJECT_DIR/.venv"}"
BIN_DIR="${CKPT_VIEWER_BIN_DIR:-"$HOME/.local/bin"}"
USE_SYSTEM_SITE="${CKPT_VIEWER_VENV_SYSTEM_SITE:-1}"

log() {
  printf '\033[1;34m[ckpt-viewer]\033[0m %s\n' "$1"
}

die() {
  printf '\033[1;31m[ckpt-viewer]\033[0m %s\n' "$1" >&2
  exit 1
}

find_python() {
  if [[ -n "${CKPT_VIEWER_PYTHON:-}" ]]; then
    command -v "$CKPT_VIEWER_PYTHON" >/dev/null 2>&1 || die "CKPT_VIEWER_PYTHON 不存在：$CKPT_VIEWER_PYTHON"
    printf '%s\n' "$CKPT_VIEWER_PYTHON"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi

  die "未找到 Python。请安装 Python >= 3.10，或设置 CKPT_VIEWER_PYTHON=/path/to/python 后重试。"
}

PYTHON_BIN="$(find_python)"

log "项目目录：$PROJECT_DIR"
log "Python：$PYTHON_BIN"

"$PYTHON_BIN" - <<'PY' || die "Python 版本过低。Ckpt_viewer 需要 Python >= 3.10。"
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY

log "创建/复用虚拟环境：$VENV_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  VENV_ARGS=()
  if [[ "$USE_SYSTEM_SITE" == "1" ]]; then
    VENV_ARGS+=("--system-site-packages")
  fi
  VENV_ARGS+=("$VENV_DIR")
  "$PYTHON_BIN" -m venv "${VENV_ARGS[@]}" || die "创建虚拟环境失败。请确认 python-venv/ensurepip 可用。"
fi

VENV_PY="$VENV_DIR/bin/python"
log "升级 pip/setuptools/wheel"
"$VENV_PY" -m pip install --upgrade pip setuptools wheel

log "安装 Ckpt_viewer 运行依赖"
"$VENV_PY" -m pip install -e "$PROJECT_DIR"

log "写入用户级启动器：$BIN_DIR"
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/ckpt-viewer" <<EOF
#!/usr/bin/env bash
exec "$VENV_PY" -m ckpt_viewer.cli "\$@"
EOF
cat > "$BIN_DIR/pth-inspector" <<EOF
#!/usr/bin/env bash
exec "$VENV_PY" -m ckpt_viewer.cli "\$@"
EOF
chmod +x "$BIN_DIR/ckpt-viewer" "$BIN_DIR/pth-inspector"

ensure_path_in_file() {
  local file="$1"
  local marker_begin="# >>> ckpt-viewer PATH >>>"
  local marker_end="# <<< ckpt-viewer PATH <<<"
  mkdir -p "$(dirname "$file")"
  touch "$file"
  if ! grep -Fq "$marker_begin" "$file"; then
    {
      printf '\n%s\n' "$marker_begin"
      printf 'export PATH="%s:$PATH"\n' "$BIN_DIR"
      printf '%s\n' "$marker_end"
    } >> "$file"
  fi
}

if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  log "将 $BIN_DIR 加入用户 PATH 配置"
  ensure_path_in_file "$HOME/.profile"
  if [[ -f "$HOME/.bashrc" || "${SHELL:-}" == *"bash"* ]]; then
    ensure_path_in_file "$HOME/.bashrc"
  fi
  if [[ -f "$HOME/.zshrc" || "${SHELL:-}" == *"zsh"* ]]; then
    ensure_path_in_file "$HOME/.zshrc"
  fi
  export PATH="$BIN_DIR:$PATH"
fi

log "验证安装"
"$BIN_DIR/ckpt-viewer" --version

cat <<EOF

安装完成。

当前终端可直接运行：
  ckpt-viewer
  ckpt-viewer analyze /path/to/model.pth --out report

新打开任意终端也可以直接运行：
  ckpt-viewer

如果当前终端找不到命令，请执行：
  export PATH="$BIN_DIR:\$PATH"
EOF
