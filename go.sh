#!/bin/bash
# ╔══════════════════════════════════════════╗
# ║  TERMINAL ELITE v3.0 — SMART LAUNCHER   ║
# ║  Works from ZIP extract OR git clone    ║
# ╚══════════════════════════════════════════╝
set -e

# ── Detect execution directory (NO clone, NO duplicate) ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "▶ Working directory: $SCRIPT_DIR"

# ── Smart update (only if git repo) ──
VERSION_FILE="$SCRIPT_DIR/.version"
REMOTE_VER_URL="https://raw.githubusercontent.com/imampersegam00-afk/crypto-forex-dashboard/main/.version"
APP_URL="https://raw.githubusercontent.com/imampersegam00-afk/crypto-forex-dashboard/main/app.py"
HTML_URL="https://raw.githubusercontent.com/imampersegam00-afk/crypto-forex-dashboard/main/templates/index.html"

update_file(){
  local url="$1" dest="$2" label="$3"
  local tmp="/tmp/_elite_update_$(basename $dest)"
  if curl -sfL "$url" -o "$tmp" 2>/dev/null; then
    # checksum compare — skip if identical
    local old_sum new_sum
    old_sum=$(md5sum "$dest" 2>/dev/null | awk '{print $1}' || echo "none")
    new_sum=$(md5sum "$tmp"  2>/dev/null | awk '{print $1}')
    if [ "$old_sum" = "$new_sum" ]; then
      echo "  ✓ $label unchanged — skip"
    else
      cp "$tmp" "$dest"
      echo "  ✅ $label updated"
    fi
  else
    echo "  ⚠ $label update skipped (offline?)"
  fi
  rm -f "$tmp"
}

# Only update if online
if curl -sf --max-time 3 "https://github.com" > /dev/null 2>&1; then
  echo "▶ Checking for updates..."
  mkdir -p "$SCRIPT_DIR/templates"
  update_file "$APP_URL"  "$SCRIPT_DIR/app.py"              "app.py"
  update_file "$HTML_URL" "$SCRIPT_DIR/templates/index.html" "index.html"
else
  echo "▶ Offline — using local files"
fi

# ── Setup Python environment ──
echo "▶ Setting up Python..."
PY=""

# Try existing venv
if [ -f "$SCRIPT_DIR/venv/bin/python3" ]; then
  PY="$SCRIPT_DIR/venv/bin/python3"
  echo "  ✓ Existing venv found"
fi

# Create venv if needed
if [ -z "$PY" ]; then
  echo "  ⚙ Creating virtual environment..."
  python3 -m venv "$SCRIPT_DIR/venv" 2>/dev/null || true
  if [ -f "$SCRIPT_DIR/venv/bin/python3" ]; then
    PY="$SCRIPT_DIR/venv/bin/python3"
    echo "  ✅ venv created"
  fi
fi

# Install dependencies
DEPS="flask yfinance ta pandas numpy requests feedparser scikit-learn"
if [ -n "$PY" ] && [ -f "$SCRIPT_DIR/venv/bin/pip" ]; then
  echo "▶ Installing dependencies..."
  "$SCRIPT_DIR/venv/bin/pip" install -q $DEPS 2>&1 | grep -E "Successfully|already|ERROR" || true
else
  echo "▶ Falling back to system pip..."
  pip3 install $DEPS --break-system-packages -q 2>/dev/null || \
  pip3 install $DEPS -q 2>/dev/null || true
  PY="python3"
fi

# Verify flask
echo "▶ Verifying Flask..."
if ! $PY -c "import flask" 2>/dev/null; then
  echo "  ❌ Flask not found — trying pip install again..."
  $PY -m pip install flask -q || pip3 install flask --break-system-packages -q
fi
$PY -c "import flask; print('  ✅ Flask', flask.__version__)"

# ── Kill existing instance ──
pkill -f "python.*app.py" 2>/dev/null || true
sleep 1

# ── Launch ──
echo ""
echo "╔══════════════════════════════════════╗"
echo "║   TERMINAL ELITE v3.0 STARTING       ║"
echo "║   http://localhost:8080               ║"
echo "╚══════════════════════════════════════╝"
echo ""

cd "$SCRIPT_DIR"
$PY app.py
