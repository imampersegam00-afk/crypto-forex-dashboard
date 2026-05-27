#!/bin/bash
# TERMINAL ELITE — Update & Run
# Usage: bash update.sh

DIR="$HOME/crypto-forex-dashboard"
BASE="https://raw.githubusercontent.com/imampersegam00-afk/crypto-forex-dashboard/main"

mkdir -p "$DIR/templates"

echo "⬇ Downloading..."
curl -sL "$BASE/app.py"               -o "$DIR/app.py"                      && echo "  ✅ app.py"
curl -sL "$BASE/templates/index.html" -o "$DIR/templates/index.html"        && echo "  ✅ index.html"
curl -sL "$BASE/requirements.txt"     -o "$DIR/requirements.txt"            && echo "  ✅ requirements.txt"

[ ! -d "$DIR/venv" ] && echo "📦 Creating venv..." && python3 -m venv "$DIR/venv"

echo "📦 Installing deps..."
"$DIR/venv/bin/pip" install -q flask yfinance ta pandas numpy requests feedparser scikit-learn

pkill -f "python3 app.py" 2>/dev/null; sleep 1

echo ""
echo "🚀 Dashboard running → http://localhost:8080"
echo "   Ctrl+C to stop"
echo ""
cd "$DIR" && "$DIR/venv/bin/python3" app.py
