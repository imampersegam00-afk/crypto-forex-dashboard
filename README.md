# 🏦 Crypto Forex News Hub — Elite AI Dashboard

> Institutional-grade trading signal dashboard built with Python Flask + yfinance + TA-Lib

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📊 Features

- **5 Assets** — BTC, ETH, XAU/USD (Gold), NASDAQ, DXY
- **6 Technical Indicators** — RSI, EMA 20/50/200, MACD, Bollinger Bands, Stochastic, ATR
- **AI Scoring System** — Multi-factor scoring for BUY/SELL/HOLD signals
- **Sniper Entry** — Smart money confirmation before entry
- **TP/SL Calculator** — Auto risk/reward 1:2 based on ATR
- **Bollinger Band Position Bar** — Visual price position inside BB
- **Live News Feed** — ForexLive RSS
- **Market Sentiment** — Overall bullish/bearish/neutral
- **Auto Refresh** — Every 60 seconds
- **Cache System** — Prevents excessive API calls
- **REST API** — `/api/data` and `/api/news` endpoints

## 🚀 Quick Start

```bash
git clone https://github.com/imampersegam00-afk/crypto-forex-dashboard
cd crypto-forex-dashboard
pip install -r requirements.txt
python3 app.py
```

Open: [http://localhost:8080](http://localhost:8080)

## 📁 Structure

```
├── app.py                  # Main Flask app + signal engine
├── templates/
│   └── index.html          # Professional dark UI
├── requirements.txt
└── start.sh                # One-click start script
```

## ⚠️ Disclaimer

This dashboard is for **informational purposes only**. Not financial advice. Always use proper risk management.

---

Built with ❤️ using Python · Flask · yfinance · TA-Lib
