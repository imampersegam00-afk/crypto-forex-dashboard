from flask import Flask, render_template, jsonify
import yfinance as yf
import ta
import pandas as pd
import numpy as np
import feedparser
import requests
from datetime import datetime
import time

app = Flask(__name__)

ASSETS = {
    "XAU/USD":  {"ticker": "GC=F",     "icon": "xauusd", "tv": "XAUUSD",  "type": "commodity"},
    "BTC/USD":  {"ticker": "BTC-USD",  "icon": "btcusd", "tv": "BTCUSD",  "type": "crypto"},
    "ETH/USD":  {"ticker": "ETH-USD",  "icon": "ethusd", "tv": "ETHUSD",  "type": "crypto"},
    "NASDAQ":   {"ticker": "^IXIC",    "icon": "nas100", "tv": "NASDAQ:NDX", "type": "index"},
    "DXY":      {"ticker": "DX-Y.NYB", "icon": "dxy",    "tv": "TVC:DXY", "type": "forex"},
    "EUR/USD":  {"ticker": "EURUSD=X", "icon": "eurusd", "tv": "EURUSD",  "type": "forex"},
}

_cache = {}
_cache_time = {}
CACHE_TTL = 90

NEWS_SOURCES = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD,GC=F&region=US&lang=en-US",
    "https://www.forexlive.com/feed/news",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://rss.cnn.com/rss/money_news_international.rss",
]

def get_news():
    news = []
    seen = set()
    for url in NEWS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:4]:
                title = entry.get("title","").strip()
                if title and title not in seen:
                    seen.add(title)
                    news.append({
                        "title": title,
                        "link": entry.get("link","#"),
                        "time": entry.get("published","")[:16] if entry.get("published") else "",
                        "source": "CryptoForexNewsHub"
                    })
        except:
            pass
        if len(news) >= 12:
            break
    return news[:12]

def compute_signal(name, ticker):
    now = time.time()
    if ticker in _cache and now - _cache_time.get(ticker,0) < CACHE_TTL:
        return _cache[ticker]
    try:
        data = yf.download(ticker, period="30d", interval="15m", progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError("No data")

        close  = data["Close"].squeeze()
        high   = data["High"].squeeze()
        low    = data["Low"].squeeze()
        volume = data["Volume"].squeeze()

        price      = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        change_pct = round((price - prev_price) / prev_price * 100, 2)

        rsi   = float(ta.momentum.RSIIndicator(close).rsi().iloc[-1])
        ema20 = float(ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1])
        ema50 = float(ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1])
        ema200= float(ta.trend.EMAIndicator(close, window=200).ema_indicator().iloc[-1])

        atr_ser = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        atr     = float(atr_ser.iloc[-1])

        macd_obj  = ta.trend.MACD(close)
        macd_hist = float(macd_obj.macd_diff().iloc[-1])
        macd_line = float(macd_obj.macd().iloc[-1])
        macd_sig  = float(macd_obj.macd_signal().iloc[-1])

        bb       = ta.volatility.BollingerBands(close)
        bb_upper = float(bb.bollinger_hband().iloc[-1])
        bb_lower = float(bb.bollinger_lband().iloc[-1])
        bb_mid   = float(bb.bollinger_mavg().iloc[-1])
        bb_pos   = round((price - bb_lower) / (bb_upper - bb_lower) * 100, 1) if (bb_upper - bb_lower) != 0 else 50

        stoch   = ta.momentum.StochasticOscillator(high, low, close)
        stoch_k = float(stoch.stoch().iloc[-1])
        stoch_d = float(stoch.stoch_signal().iloc[-1])

        last_vol = float(volume.iloc[-1])
        avg_vol  = float(volume.tail(20).mean())

        # Fibonacci levels
        recent_high = float(high.tail(96).max())
        recent_low  = float(low.tail(96).min())
        fib_range   = recent_high - recent_low
        fib_levels  = {
            "0.0":   round(recent_low, 4),
            "0.236": round(recent_low + fib_range * 0.236, 4),
            "0.382": round(recent_low + fib_range * 0.382, 4),
            "0.5":   round(recent_low + fib_range * 0.5, 4),
            "0.618": round(recent_low + fib_range * 0.618, 4),
            "0.786": round(recent_low + fib_range * 0.786, 4),
            "1.0":   round(recent_high, 4),
        }

        # Support / Resistance
        resistance1 = round(recent_high, 3)
        resistance2 = round(recent_high + atr * 1.5, 3)
        support1    = round(recent_low + atr, 3)
        support2    = round(recent_low, 3)

        # Smart Money
        delta_vol = round(last_vol - avg_vol, 0)
        if last_vol > avg_vol * 1.5:
            smart_money_pct = 78
            smart_money     = "BULLISH" if ema20 > ema50 else "BEARISH"
            institutional   = 78
            retail_sent     = 22
        elif last_vol > avg_vol:
            smart_money_pct = 55
            smart_money     = "NEUTRAL"
            institutional   = 55
            retail_sent     = 45
        else:
            smart_money_pct = 30
            smart_money     = "WEAK"
            institutional   = 30
            retail_sent     = 70

        # Scoring
        score = 0
        if ema20 > ema50 > ema200:  score += 30
        elif ema20 > ema50:         score += 18
        elif ema20 < ema50 < ema200:score -= 30
        elif ema20 < ema50:         score -= 18

        if rsi > 60:   score += 20
        elif rsi < 40: score -= 20

        if macd_line > macd_sig and macd_hist > 0: score += 15
        elif macd_line < macd_sig and macd_hist < 0: score -= 15

        if stoch_k > stoch_d and stoch_k < 80: score += 10
        elif stoch_k < stoch_d and stoch_k > 20: score -= 10

        if last_vol > avg_vol: score += 8

        if score >= 60:      signal, signal_class, signal_icon = "STRONG BUY",  "strong-buy",  "LONG"
        elif score >= 25:    signal, signal_class, signal_icon = "BUY",          "buy",         "LONG"
        elif score <= -60:   signal, signal_class, signal_icon = "STRONG SELL",  "strong-sell", "SHORT"
        elif score <= -25:   signal, signal_class, signal_icon = "SELL",         "sell",        "SHORT"
        else:                signal, signal_class, signal_icon = "NEUTRAL",      "hold",        "WAIT"

        # Bias / Structure
        if ema20 > ema50:
            bias      = "LONG"
            structure = "BULLISH"
            trend_str = "UPTREND (M15)"
            orderflow = "BUYING PRESSURE"
        else:
            bias      = "SHORT"
            structure = "BEARISH"
            trend_str = "DOWNTREND (M15)"
            orderflow = "SELLING PRESSURE"

        confidence = min(abs(score), 99)

        # TP / SL
        sl_long  = round(price - atr * 1.5, 3)
        tp1_long = round(price + atr * 2.0, 3)
        tp2_long = round(price + atr * 3.0, 3)
        tp3_long = round(price + atr * 4.5, 3)

        sl_short  = round(price + atr * 1.5, 3)
        tp1_short = round(price - atr * 2.0, 3)
        tp2_short = round(price - atr * 3.0, 3)
        tp3_short = round(price - atr * 4.5, 3)

        # Spread (simulated)
        spread = round(atr * 0.02, 2)

        # Sniper / Demand / Supply zones
        demand_zone_h = round(support1 + atr * 0.3, 3)
        demand_zone_l = round(support2, 3)
        supply_zone_h = round(resistance2, 3)
        supply_zone_l = round(resistance1, 3)

        result = {
            "name": name,
            "ticker": ticker,
            "price": round(price, 3),
            "prev_price": round(prev_price, 3),
            "change_pct": change_pct,
            "change_abs": round(price - prev_price, 3),
            "signal": signal,
            "signal_class": signal_class,
            "signal_icon": signal_icon,
            "confidence": confidence,
            "score": score,
            "bias": bias,
            "structure": structure,
            "trend_str": trend_str,
            "orderflow": orderflow,
            "rsi": round(rsi, 2),
            "ema20": round(ema20, 3),
            "ema50": round(ema50, 3),
            "ema200": round(ema200, 3),
            "macd": round(macd_hist, 4),
            "atr": round(atr, 3),
            "spread": spread,
            "bb_upper": round(bb_upper, 3),
            "bb_lower": round(bb_lower, 3),
            "bb_mid": round(bb_mid, 3),
            "bb_pos": bb_pos,
            "stoch_k": round(stoch_k, 1),
            "stoch_d": round(stoch_d, 1),
            "smart_money": smart_money,
            "smart_money_pct": smart_money_pct,
            "institutional": institutional,
            "retail_sent": retail_sent,
            "delta_vol": int(delta_vol),
            "fib": fib_levels,
            "resistance1": resistance1,
            "resistance2": resistance2,
            "support1": support1,
            "support2": support2,
            "demand_zone_h": demand_zone_h,
            "demand_zone_l": demand_zone_l,
            "supply_zone_h": supply_zone_h,
            "supply_zone_l": supply_zone_l,
            "sl_long": sl_long, "tp1_long": tp1_long,
            "tp2_long": tp2_long, "tp3_long": tp3_long,
            "sl_short": sl_short, "tp1_short": tp1_short,
            "tp2_short": tp2_short, "tp3_short": tp3_short,
            "update_time": datetime.now().strftime("%H:%M:%S"),
        }
        _cache[ticker] = result
        _cache_time[ticker] = now
        return result

    except Exception as e:
        return {"name": name, "ticker": ticker, "price": 0, "change_pct": 0,
                "signal": "ERROR", "signal_class": "hold", "signal_icon": "WAIT",
                "confidence": 0, "score": 0, "bias": "N/A", "structure": "N/A",
                "trend_str": "N/A", "orderflow": "N/A", "rsi": 0,
                "ema20": 0, "ema50": 0, "ema200": 0, "macd": 0, "atr": 0, "spread": 0,
                "bb_upper": 0, "bb_lower": 0, "bb_mid": 0, "bb_pos": 50,
                "stoch_k": 0, "stoch_d": 0, "smart_money": "N/A",
                "smart_money_pct": 0, "institutional": 0, "retail_sent": 0, "delta_vol": 0,
                "fib": {}, "resistance1": 0, "resistance2": 0, "support1": 0, "support2": 0,
                "demand_zone_h": 0, "demand_zone_l": 0, "supply_zone_h": 0, "supply_zone_l": 0,
                "sl_long": 0, "tp1_long": 0, "tp2_long": 0, "tp3_long": 0,
                "sl_short": 0, "tp1_short": 0, "tp2_short": 0, "tp3_short": 0,
                "change_abs": 0, "update_time": "N/A"}

@app.route("/")
def home():
    market_data = {}
    for name, info in ASSETS.items():
        d = compute_signal(name, info["ticker"])
        d["tv"]   = info["tv"]
        d["type"] = info["type"]
        market_data[name] = d

    gold = market_data.get("XAU/USD", {})
    news_data = get_news()
    now_str   = datetime.now().strftime("%H:%M:%S (UTC+7)")
    date_str  = datetime.now().strftime("%d %b %Y").upper()
    buy_count  = sum(1 for d in market_data.values() if "BUY"  in d["signal"])
    sell_count = sum(1 for d in market_data.values() if "SELL" in d["signal"])
    hold_count = sum(1 for d in market_data.values() if d["signal"] in ["NEUTRAL","ERROR"])

    return render_template("index.html",
        data=market_data, gold=gold, news=news_data,
        now=now_str, date=date_str,
        buy_count=buy_count, sell_count=sell_count, hold_count=hold_count)

@app.route("/api/data")
def api_data():
    out = {}
    for name, info in ASSETS.items():
        d = compute_signal(name, info["ticker"])
        d["tv"] = info["tv"]
        out[name] = d
    return jsonify(out)

@app.route("/api/news")
def api_news():
    return jsonify(get_news())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
