from flask import Flask, render_template, jsonify
import yfinance as yf
import ta
import pandas as pd
import numpy as np
import feedparser
from datetime import datetime
import threading
import time

app = Flask(__name__)

# ─────────────────────────────────────────────
#  ASSETS
# ─────────────────────────────────────────────
ASSETS = {
    "BTC/USD":  {"ticker": "BTC-USD",  "icon": "₿",  "type": "crypto"},
    "ETH/USD":  {"ticker": "ETH-USD",  "icon": "⟠",  "type": "crypto"},
    "XAU/USD":  {"ticker": "GC=F",     "icon": "🥇", "type": "commodity"},
    "NASDAQ":   {"ticker": "^IXIC",    "icon": "📈", "type": "index"},
    "DXY":      {"ticker": "DX-Y.NYB", "icon": "💵", "type": "forex"},
}

# Cache
_cache = {}
_cache_time = {}
CACHE_TTL = 60  # seconds

# ─────────────────────────────────────────────
#  NEWS
# ─────────────────────────────────────────────
def get_news():
    try:
        feed = feedparser.parse("https://www.forexlive.com/feed/news")
        news = []
        for entry in feed.entries[:8]:
            news.append({
                "title": entry.title,
                "link": entry.get("link", "#"),
                "time": entry.get("published", "")
            })
        return news
    except:
        return [{"title": "News unavailable", "link": "#", "time": ""}]

# ─────────────────────────────────────────────
#  SIGNAL ENGINE
# ─────────────────────────────────────────────
def get_signal(name, ticker):
    now = time.time()
    if ticker in _cache and now - _cache_time.get(ticker, 0) < CACHE_TTL:
        return _cache[ticker]

    try:
        data = yf.download(ticker, period="30d", interval="15m", progress=False)
        if data.empty:
            raise ValueError("No data")

        close  = data["Close"].squeeze()
        high   = data["High"].squeeze()
        low    = data["Low"].squeeze()
        volume = data["Volume"].squeeze()

        price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        change_pct = ((price - prev_price) / prev_price) * 100

        # Indicators
        rsi    = float(ta.momentum.RSIIndicator(close).rsi().iloc[-1])
        ema20  = float(ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1])
        ema50  = float(ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1])
        ema200 = float(ta.trend.EMAIndicator(close, window=200).ema_indicator().iloc[-1])

        atr_series = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        atr = float(atr_series.iloc[-1])
        atr_avg = float(atr_series.tail(20).mean())

        macd_obj = ta.trend.MACD(close)
        macd_line = float(macd_obj.macd().iloc[-1])
        macd_signal = float(macd_obj.macd_signal().iloc[-1])
        macd_hist = float(macd_obj.macd_diff().iloc[-1])

        bb = ta.volatility.BollingerBands(close)
        bb_upper = float(bb.bollinger_hband().iloc[-1])
        bb_lower = float(bb.bollinger_lband().iloc[-1])
        bb_mid   = float(bb.bollinger_mavg().iloc[-1])

        stoch = ta.momentum.StochasticOscillator(high, low, close)
        stoch_k = float(stoch.stoch().iloc[-1])
        stoch_d = float(stoch.stoch_signal().iloc[-1])

        last_volume = float(volume.iloc[-1])
        avg_volume  = float(volume.tail(20).mean())

        # ── Scoring System ──
        score = 0
        reasons = []

        # Trend
        if ema20 > ema50 > ema200:
            score += 30
            reasons.append("EMA alignment bullish")
            trend = "STRONG UPTREND"
        elif ema20 > ema50:
            score += 20
            reasons.append("EMA20 > EMA50")
            trend = "UPTREND"
        elif ema20 < ema50 < ema200:
            score -= 30
            reasons.append("EMA alignment bearish")
            trend = "STRONG DOWNTREND"
        elif ema20 < ema50:
            score -= 20
            reasons.append("EMA20 < EMA50")
            trend = "DOWNTREND"
        else:
            trend = "RANGING"

        # RSI
        if rsi > 60:
            score += 20
            reasons.append(f"RSI bullish ({round(rsi,1)})")
        elif rsi < 40:
            score -= 20
            reasons.append(f"RSI bearish ({round(rsi,1)})")
        elif 45 < rsi < 55:
            reasons.append(f"RSI neutral ({round(rsi,1)})")

        # MACD
        if macd_line > macd_signal and macd_hist > 0:
            score += 15
            reasons.append("MACD bullish crossover")
        elif macd_line < macd_signal and macd_hist < 0:
            score -= 15
            reasons.append("MACD bearish crossover")

        # Stochastic
        if stoch_k > stoch_d and stoch_k < 80:
            score += 10
            reasons.append("Stoch bullish")
        elif stoch_k < stoch_d and stoch_k > 20:
            score -= 10
            reasons.append("Stoch bearish")

        # Volume / Smart Money
        if last_volume > avg_volume * 1.5:
            smart_money = "🔥 VERY ACTIVE"
            score += 15
        elif last_volume > avg_volume:
            smart_money = "🟢 ACTIVE"
            score += 8
        else:
            smart_money = "🔴 WEAK"

        # Volatility
        if atr > atr_avg * 1.3:
            volatility = "🔥 HIGH"
        elif atr < atr_avg * 0.7:
            volatility = "🟡 LOW"
        else:
            volatility = "🟢 NORMAL"

        # BB position
        bb_pos = (price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) != 0 else 50

        # ── Final Signal ──
        if score >= 60:
            signal = "STRONG BUY"
            signal_class = "strong-buy"
            signal_icon = "🚀"
        elif score >= 25:
            signal = "BUY"
            signal_class = "buy"
            signal_icon = "📈"
        elif score <= -60:
            signal = "STRONG SELL"
            signal_class = "strong-sell"
            signal_icon = "💥"
        elif score <= -25:
            signal = "SELL"
            signal_class = "sell"
            signal_icon = "📉"
        else:
            signal = "HOLD"
            signal_class = "hold"
            signal_icon = "⏸️"

        # Sniper Entry
        sniper = "WAIT"
        if signal in ["BUY", "STRONG BUY"] and smart_money != "🔴 WEAK" and abs(price - ema20) < atr:
            sniper = "🎯 LONG READY"
        elif signal in ["SELL", "STRONG SELL"] and smart_money != "🔴 WEAK" and abs(price - ema20) < atr:
            sniper = "🎯 SHORT READY"

        # Risk/Reward
        sl_long  = round(price - (atr * 1.5), 4)
        tp_long  = round(price + (atr * 3.0), 4)
        sl_short = round(price + (atr * 1.5), 4)
        tp_short = round(price - (atr * 3.0), 4)

        confidence = min(abs(score), 99)

        result = {
            "name": name,
            "ticker": ticker,
            "price": round(price, 4),
            "change_pct": round(change_pct, 2),
            "signal": signal,
            "signal_class": signal_class,
            "signal_icon": signal_icon,
            "confidence": confidence,
            "score": score,
            "trend": trend,
            "rsi": round(rsi, 2),
            "ema20": round(ema20, 4),
            "ema50": round(ema50, 4),
            "ema200": round(ema200, 4),
            "macd": round(macd_hist, 4),
            "stoch_k": round(stoch_k, 2),
            "stoch_d": round(stoch_d, 2),
            "atr": round(atr, 4),
            "bb_upper": round(bb_upper, 4),
            "bb_lower": round(bb_lower, 4),
            "bb_mid": round(bb_mid, 4),
            "bb_pos": round(bb_pos, 1),
            "smart_money": smart_money,
            "volatility": volatility,
            "volume": round(last_volume, 0),
            "avg_volume": round(avg_volume, 0),
            "sniper": sniper,
            "sl_long": sl_long,
            "tp_long": tp_long,
            "sl_short": sl_short,
            "tp_short": tp_short,
            "reasons": reasons,
            "update_time": datetime.now().strftime("%H:%M:%S"),
        }

        _cache[ticker] = result
        _cache_time[ticker] = now
        return result

    except Exception as e:
        return {
            "name": name,
            "ticker": ticker,
            "price": 0,
            "change_pct": 0,
            "signal": "ERROR",
            "signal_class": "hold",
            "signal_icon": "⚠️",
            "confidence": 0,
            "score": 0,
            "trend": "N/A",
            "rsi": 0,
            "ema20": 0, "ema50": 0, "ema200": 0,
            "macd": 0, "stoch_k": 0, "stoch_d": 0,
            "atr": 0, "bb_upper": 0, "bb_lower": 0, "bb_mid": 0, "bb_pos": 0,
            "smart_money": "N/A",
            "volatility": "N/A",
            "volume": 0, "avg_volume": 0,
            "sniper": "N/A",
            "sl_long": 0, "tp_long": 0, "sl_short": 0, "tp_short": 0,
            "reasons": [str(e)],
            "update_time": datetime.now().strftime("%H:%M:%S"),
        }

# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    market_data = {}
    for name, info in ASSETS.items():
        market_data[name] = get_signal(name, info["ticker"])
        market_data[name]["icon"] = info["icon"]
        market_data[name]["type"] = info["type"]

    news_data = get_news()
    now = datetime.now().strftime("%A, %d %B %Y  %H:%M:%S")

    buy_count  = sum(1 for d in market_data.values() if "BUY"  in d["signal"])
    sell_count = sum(1 for d in market_data.values() if "SELL" in d["signal"])
    hold_count = sum(1 for d in market_data.values() if d["signal"] == "HOLD")

    return render_template(
        "index.html",
        data=market_data,
        news=news_data,
        now=now,
        buy_count=buy_count,
        sell_count=sell_count,
        hold_count=hold_count,
    )

@app.route("/api/data")
def api_data():
    market_data = {}
    for name, info in ASSETS.items():
        market_data[name] = get_signal(name, info["ticker"])
        market_data[name]["icon"] = info["icon"]
    return jsonify(market_data)

@app.route("/api/news")
def api_news():
    return jsonify(get_news())

# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
