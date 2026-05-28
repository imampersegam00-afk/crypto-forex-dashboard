#!/bin/bash
set -e
DIR="$HOME/crypto-forex-dashboard"
REPO="https://raw.githubusercontent.com/imampersegam00-afk/crypto-forex-dashboard/main"

echo "================================"
echo "  TERMINAL ELITE — INSTALLER"
echo "================================"

# Buat folder
mkdir -p "$DIR/templates"
cd "$DIR"

# Download files
echo "[1/4] Download files..."
curl -sL "$REPO/app.py" -o app.py && echo "  OK app.py"
curl -sL "$REPO/templates/index.html" -o templates/index.html && echo "  OK index.html"

# Install python3-venv kalau belum ada
echo "[2/4] Setup Python..."
if ! python3 -m venv --help > /dev/null 2>&1; then
    sudo apt install python3-venv -y 2>/dev/null || apt install python3-venv -y 2>/dev/null || true
fi

# Buat venv
if [ ! -d "$DIR/venv" ]; then
    python3 -m venv "$DIR/venv" 2>/dev/null || true
fi

# Install deps — coba venv dulu, fallback ke --break-system-packages
echo "[3/4] Install dependencies..."
if [ -f "$DIR/venv/bin/pip" ]; then
    "$DIR/venv/bin/pip" install -q flask yfinance ta pandas numpy requests feedparser scikit-learn
    PYTHON="$DIR/venv/bin/python3"
else
    pip3 install flask yfinance ta pandas numpy requests feedparser scikit-learn --break-system-packages -q 2>/dev/null || \
    pip3 install flask yfinance ta pandas numpy requests feedparser scikit-learn -q
    PYTHON="python3"
fi

# Simpan cara run
echo "#!/bin/bash
cd $DIR
$PYTHON app.py" > "$DIR/run.sh"
chmod +x "$DIR/run.sh"

echo "[4/4] Selesai!"
echo ""
echo "================================"
echo "  DASHBOARD → localhost:8080"
echo "================================"
echo ""

# Langsung jalankan
pkill -f "app.py" 2>/dev/null; sleep 1
cd "$DIR"
$PYTHON app.py
