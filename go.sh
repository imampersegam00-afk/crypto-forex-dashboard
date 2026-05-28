#!/bin/bash
# TERMINAL ELITE — ONE CLICK RUN
DIR="$HOME/crypto-forex-dashboard"
REPO="https://raw.githubusercontent.com/imampersegam00-afk/crypto-forex-dashboard/main"

mkdir -p "$DIR/templates"
cd "$DIR"

echo ">>> Download terbaru dari GitHub..."
curl -sL "$REPO/app.py" -o app.py
curl -sL "$REPO/templates/index.html" -o templates/index.html

echo ">>> Cek Python..."
PY=""

# Coba venv dulu
if [ -f "$DIR/venv/bin/python3" ]; then
    "$DIR/venv/bin/pip" install flask -q 2>/dev/null
    PY="$DIR/venv/bin/python3"
fi

# Kalau venv gagal, buat ulang
if [ -z "$PY" ]; then
    echo ">>> Buat venv baru..."
    python3 -m venv "$DIR/venv" 2>/dev/null
    if [ -f "$DIR/venv/bin/python3" ]; then
        PY="$DIR/venv/bin/python3"
    fi
fi

# Install semua deps ke venv
if [ -n "$PY" ] && [ -f "$DIR/venv/bin/pip" ]; then
    echo ">>> Install dependencies ke venv..."
    "$DIR/venv/bin/pip" install -q flask yfinance ta pandas numpy requests feedparser scikit-learn 2>&1 | tail -2
else
    # Fallback: install langsung ke sistem
    echo ">>> Install ke sistem Python..."
    pip3 install flask yfinance ta pandas numpy requests feedparser scikit-learn \
        --break-system-packages -q 2>/dev/null || \
    pip3 install flask yfinance ta pandas numpy requests feedparser scikit-learn -q 2>/dev/null
    PY="python3"
fi

echo ">>> Python: $PY"
echo ">>> Test import flask..."
$PY -c "import flask; print('flask OK:', flask.__version__)"

pkill -f "python.*app.py" 2>/dev/null
sleep 1

echo ""
echo "================================"
echo " Dashboard → http://localhost:8080"
echo "================================"
cd "$DIR"
$PY app.py
