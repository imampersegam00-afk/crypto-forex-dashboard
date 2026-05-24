#!/bin/bash

clear

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏦  CRYPTO FOREX NEWS HUB — ELITE AI"
echo "📡  Starting Ultimate Institutional Dashboard"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "🚀 Starting dashboard on port 8080..."
echo ""

# Start app
python3 app.py &
APP_PID=$!

sleep 4

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ LOCAL   : http://127.0.0.1:8080"
echo ""

# Cloudflare tunnel (if available)
if command -v cloudflared &> /dev/null; then
    echo "🌐 Starting Cloudflare tunnel..."
    cloudflared tunnel --url http://127.0.0.1:8080 &
    echo "🌐 GLOBAL  : Check terminal output above for public URL"
else
    echo "💡 For public URL: install cloudflared then re-run"
    echo "   brew install cloudflare/cloudflare/cloudflared"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧠 AI SIGNALS: BTC · ETH · GOLD · NASDAQ · DXY"
echo "📊 INDICATORS: RSI · EMA · MACD · BB · Stoch · ATR"
echo "🎯 FEATURES  : Sniper Entry · TP/SL · Smart Money"
echo "📰 NEWS FEED : ForexLive Live Updates"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

wait $APP_PID
