#!/bin/bash
# ═══════════════════════════════════════════════════
#  TERMINAL ELITE — Auto Update Script
#  Jalankan: bash update.sh
# ═══════════════════════════════════════════════════

DIR="$HOME/crypto-forex-dashboard"
echo "🔄 Updating TERMINAL ELITE dashboard..."

# Buat folder jika belum ada
mkdir -p "$DIR/templates"

# Download file terbaru dari GitHub
BASE="https://raw.githubusercontent.com/imampersegam00-afk/crypto-forex-dashboard/main"
curl -sL "$BASE/app.py"               -o "$DIR/app.py"        && echo "✅ app.py"
curl -sL "$BASE/templates/index.html" -o "$DIR/templates/index.html" && echo "✅ index.html"
curl -sL "$BASE/requirements.txt"     -o "$DIR/requirements.txt" && echo "✅ requirements.txt"

# Setup venv jika belum ada
if [ ! -d "$DIR/venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv "$DIR/venv"
fi

# Install dependencies
echo "📦 Installing dependencies..."
"$DIR/venv/bin/pip" install -q flask yfinance ta pandas numpy requests feedparser scikit-learn

# Stop existing instance
pkill -f "python3 app.py" 2>/dev/null
sleep 1

# Run dashboard
echo ""
echo "🚀 Starting dashboard at http://localhost:8080"
echo "   Press Ctrl+C to stop"
echo ""
cd "$DIR"
"$DIR/venv/bin/python3" app.py
