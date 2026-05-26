from flask import Flask, render_template, jsonify
import yfinance as yf
import ta
import pandas as pd
import numpy as np
import feedparser
import time
from datetime import datetime

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

app = Flask(__name__)

ASSETS = {
    "XAU/USD":  {"ticker": "GC=F",     "tv": "XAUUSD",     "type": "commodity"},
    "BTC/USD":  {"ticker": "BTC-USD",  "tv": "BTCUSD",     "type": "crypto"},
    "ETH/USD":  {"ticker": "ETH-USD",  "tv": "ETHUSD",     "type": "crypto"},
    "EUR/USD":  {"ticker": "EURUSD=X", "tv": "EURUSD",     "type": "forex"},
    "NASDAQ":   {"ticker": "^IXIC",    "tv": "NASDAQ:NDX", "type": "index"},
    "DXY":      {"ticker": "DX-Y.NYB", "tv": "TVC:DXY",   "type": "forex"},
    "OIL":      {"ticker": "CL=F",     "tv": "USOIL",      "type": "commodity"},
}

CORR_PAIRS = [
    ("BTC/USD", "NASDAQ"),
    ("XAU/USD", "DXY"),
    ("OIL",     "NASDAQ"),
    ("ETH/USD", "BTC/USD"),
]

_cache      = {}
_cache_time = {}
_corr_cache = {}
_corr_time  = 0
CACHE_TTL   = 90

NEWS_SOURCES = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD,GC=F&region=US&lang=en-US",
    "https://www.forexlive.com/feed/news",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://rss.cnn.com/rss/money_news_international.rss",
    "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
]

# ─────────────────────────────────────────────
#  NEWS HUB (real_news_hub logic)
# ─────────────────────────────────────────────
def get_news():
    news, seen = [], set()
    for url in NEWS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:4]:
                title = e.get("title", "").strip()
                if title and title not in seen:
                    seen.add(title)
                    news.append({
                        "title":  title,
                        "link":   e.get("link", "#"),
                        "time":   e.get("published", "")[:16],
                        "source": "CryptoForexNewsHub"
                    })
        except:
            pass
        if len(news) >= 12:
            break
    return news[:12]

# ─────────────────────────────────────────────
#  ML ENGINE (machine_learning_ai + hedgefund_mode)
# ─────────────────────────────────────────────
def ml_predict(ticker):
    if not ML_AVAILABLE:
        return {"ml_signal": "N/A", "ml_accuracy": 0, "ml_confidence": 50}
    try:
        data = yf.download(ticker, period="60d", interval="15m", progress=False, auto_adjust=True)
        if len(data) < 100:
            raise ValueError("Not enough data")
        close = data["Close"].squeeze()
        high  = data["High"].squeeze()
        low   = data["Low"].squeeze()
        data["RSI"]    = ta.momentum.RSIIndicator(close).rsi()
        data["EMA20"]  = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        data["EMA50"]  = ta.trend.EMAIndicator(close, window=50).ema_indicator()
        data["ATR"]    = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        data["MACD"]   = ta.trend.MACD(close).macd_diff()
        data["TARGET"] = np.where(close.shift(-1) > close, 1, 0)
        data = data.dropna()
        X = data[["RSI","EMA20","EMA50","ATR","MACD"]]
        y = data["TARGET"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        acc   = round(float(accuracy_score(y_test, model.predict(X_test))) * 100, 1)
        pred  = int(model.predict(X.iloc[[-1]])[0])
        proba = float(model.predict_proba(X.iloc[[-1]])[0][pred]) * 100
        signal = "🟢 ML BUY" if pred == 1 else "🔴 ML SELL"
        return {"ml_signal": signal, "ml_accuracy": acc, "ml_confidence": round(proba, 1)}
    except:
        return {"ml_signal": "N/A", "ml_accuracy": 0, "ml_confidence": 50}

# ─────────────────────────────────────────────
#  MULTI TIMEFRAME (multi_timeframe_ai)
# ─────────────────────────────────────────────
def multi_tf(ticker):
    result = {}
    bullish = bearish = 0
    for label, tf in [("5m","5m"),("15m","15m"),("1h","60m")]:
        try:
            d = yf.download(ticker, period="5d", interval=tf, progress=False, auto_adjust=True)
            c = d["Close"].squeeze()
            v = d["Volume"].squeeze()
            rsi   = float(ta.momentum.RSIIndicator(c).rsi().iloc[-1])
            ema20 = float(ta.trend.EMAIndicator(c, window=20).ema_indicator().iloc[-1])
            ema50 = float(ta.trend.EMAIndicator(c, window=50).ema_indicator().iloc[-1])
            lv    = float(v.iloc[-1])
            av    = float(v.tail(20).mean())
            score = 0
            if rsi > 55:   score += 30
            if ema20>ema50: score += 40
            if lv > av:    score += 30
            sig = "BULLISH" if score >= 70 else ("BEARISH" if score <= 30 else "NEUTRAL")
            if sig == "BULLISH": bullish += 1
            elif sig == "BEARISH": bearish += 1
            result[label] = {"signal": sig, "score": score, "rsi": round(rsi, 1)}
        except:
            result[label] = {"signal": "N/A", "score": 0, "rsi": 0}
    if bullish >= 2:   mtf_bias = "MTF BULLISH"
    elif bearish >= 2: mtf_bias = "MTF BEARISH"
    else:              mtf_bias = "MTF MIXED"
    result["bias"] = mtf_bias
    return result

# ─────────────────────────────────────────────
#  WHALE TRACKER (whale_tracker)
# ─────────────────────────────────────────────
def whale_track(ticker):
    try:
        d = yf.download(ticker, period="2d", interval="5m", progress=False, auto_adjust=True)
        v    = d["Volume"].squeeze()
        lv   = float(v.iloc[-1])
        av   = float(v.tail(20).mean())
        ratio = round(lv / av, 2) if av > 0 else 1
        if ratio >= 2.0:   whale = "🐋 ACCUMULATION"
        elif ratio <= 0.5: whale = "🐋 DISTRIBUTION"
        else:              whale = "NORMAL"
        conf = min(round(ratio * 50, 1), 99)
        return {"whale": whale, "vol_ratio": ratio, "whale_conf": conf}
    except:
        return {"whale": "N/A", "vol_ratio": 1, "whale_conf": 0}

# ─────────────────────────────────────────────
#  CORRELATION ENGINE (correlation_engine)
# ─────────────────────────────────────────────
def get_correlations(market_data):
    global _corr_cache, _corr_time
    now = time.time()
    if _corr_cache and now - _corr_time < 300:
        return _corr_cache
    result = []
    for a1, a2 in CORR_PAIRS:
        try:
            t1 = ASSETS[a1]["ticker"]
            t2 = ASSETS[a2]["ticker"]
            d1 = yf.download(t1, period="5d", interval="15m", progress=False, auto_adjust=True)["Close"].squeeze()
            d2 = yf.download(t2, period="5d", interval="15m", progress=False, auto_adjust=True)["Close"].squeeze()
            corr   = round(float(d1.corr(d2)), 2)
            strength = abs(corr)
            if corr >= 0.7:   status = "STRONG POSITIVE"
            elif corr >= 0.4: status = "POSITIVE"
            elif corr <= -0.7:status = "STRONG NEGATIVE"
            elif corr <= -0.4:status = "NEGATIVE"
            else:             status = "WEAK"
            result.append({"pair": f"{a1} ↔ {a2}", "corr": corr, "status": status,
                           "strength": round(strength * 100), "positive": corr >= 0})
        except:
            result.append({"pair": f"{a1} ↔ {a2}", "corr": 0, "status": "N/A", "strength": 0, "positive": True})
    _corr_cache = result
    _corr_time  = now
    return result

# ─────────────────────────────────────────────
#  NARRATIVE AI (narrative_ai)
# ─────────────────────────────────────────────
def get_narrative(score, rsi, ema20, ema50, lv, av, price):
    s = 0
    if rsi > 60:       s += 25
    if ema20 > ema50:  s += 35
    if lv > av:        s += 25
    if price > ema20:  s += 15
    if s >= 80:   return "STRONG BULLISH NARRATIVE"
    elif s >= 60: return "BULLISH FLOW BUILDING"
    elif s >= 40: return "MARKET CONSOLIDATING"
    elif s >= 20: return "BEARISH PRESSURE"
    else:         return "BEARISH DISTRIBUTION"

# ─────────────────────────────────────────────
#  MACRO ENGINE (macro_engine)
# ─────────────────────────────────────────────
def get_macro(rsi, ema20, ema50, lv, av):
    s = 0
    if rsi > 55:  s += 30
    if ema20>ema50: s += 40
    if lv > av:   s += 20
    if s >= 70:   return "RISK ON 🟢"
    elif s <= 30: return "RISK OFF 🔴"
    else:         return "NEUTRAL 🟡"

# ─────────────────────────────────────────────
#  ORDER BLOCK ENGINE (orderblock_engine)
# ─────────────────────────────────────────────
def get_orderblock(close_last, open_last, ema20, lv, av, high_tail, low_tail):
    candle_body = abs(close_last - open_last)
    avg_range   = high_tail - low_tail
    if close_last > ema20 and lv > av and candle_body > avg_range * 0.5:
        return "BULLISH ORDER BLOCK 🟢"
    elif close_last < ema20 and lv > av and candle_body > avg_range * 0.5:
        return "BEARISH ORDER BLOCK 🔴"
    return "NO CLEAR BLOCK 🟡"

# ─────────────────────────────────────────────
#  SCALPING ENGINE (scalping_engine)
# ─────────────────────────────────────────────
def get_scalp(rsi, ema20, ema50, lv, av):
    s = 0
    if rsi > 55:  s += 30
    if ema20>ema50: s += 40
    if lv > av:   s += 30
    if s >= 70:   return {"scalp": "SCALP LONG 🟢", "scalp_score": s}
    elif s <= 30: return {"scalp": "SCALP SHORT 🔴", "scalp_score": s}
    return {"scalp": "NO SCALP 🟡", "scalp_score": s}

# ─────────────────────────────────────────────
#  LIQUIDITY ENGINE (liquidity_engine)
# ─────────────────────────────────────────────
def get_liquidity(price, recent_high, recent_low, lv, av):
    s = 0
    sig = "NO SWEEP 🟡"
    if price >= recent_high:
        sig = "BUY SIDE LIQUIDITY TAKEN 🟢"; s += 50
    elif price <= recent_low:
        sig = "SELL SIDE LIQUIDITY TAKEN 🔴"; s += 50
    if lv / av > 1.5 if av > 0 else False:
        s += 40
    return {"liquidity": sig, "liq_score": min(s, 99)}

# ─────────────────────────────────────────────
#  HEDGEFUND SCORE (hedgefund_mode + central_ai_brain)
# ─────────────────────────────────────────────
def get_hedgefund_score(rsi, ema20, ema50, lv, av, price, candle_body, avg_range):
    s = 0
    trend     = "BULLISH 🟢" if ema20 > ema50 else "BEARISH 🔴"
    if ema20 > ema50: s += 25
    whale     = "ACCUMULATION 🟢" if lv > av else "DISTRIBUTION 🔴"
    if lv > av: s += 20
    liquidity = "BUY SIDE 🟢" if price > ema20 else "SELL SIDE 🔴"
    if price > ema20: s += 15
    smartmoney= "ACTIVE 🟢" if candle_body > avg_range * 0.5 else "WEAK 🔴"
    if candle_body > avg_range * 0.5: s += 20
    macro     = "RISK ON 🟢" if rsi > 55 else "RISK OFF 🔴"
    if rsi > 55: s += 20
    hf_signal = "STRONG BUY ✅" if s >= 80 else ("BUY 🟢" if s >= 60 else ("SELL 🔴" if s <= 30 else "NEUTRAL 🟡"))
    return {
        "hf_score": s, "hf_signal": hf_signal,
        "hf_trend": trend, "hf_whale": whale,
        "hf_liquidity": liquidity, "hf_smartmoney": smartmoney,
        "hf_macro": macro
    }

# ─────────────────────────────────────────────
#  WAR ROOM GLOBAL BIAS (warroom_engine)
# ─────────────────────────────────────────────
def get_warroom_bias(market_data):
    bullish = sum(1 for d in market_data.values() if "BUY" in d.get("signal",""))
    bearish = sum(1 for d in market_data.values() if "SELL" in d.get("signal",""))
    if bullish > bearish:   return "GLOBAL RISK ON 🟢", bullish, bearish
    elif bearish > bullish: return "GLOBAL RISK OFF 🔴", bullish, bearish
    else:                   return "MIXED MARKET 🟡", bullish, bearish

# ─────────────────────────────────────────────
#  PORTFOLIO MANAGER
# ─────────────────────────────────────────────
def get_portfolio(market_data, balance=10000, risk_pct=0.01):
    portfolio = []
    for name, d in market_data.items():
        if d.get("atr", 0) > 0 and d.get("price", 0) > 0:
            risk_amount = balance * risk_pct
            sl_dist     = d["atr"] * 1.5
            lot_size    = round(risk_amount / sl_dist, 4) if sl_dist > 0 else 0
            portfolio.append({
                "name":     name,
                "signal":   d.get("signal","N/A"),
                "price":    d.get("price", 0),
                "lot":      lot_size,
                "sl":       d.get("sl_long", 0) if "BUY" in d.get("signal","") else d.get("sl_short", 0),
                "tp":       d.get("tp2_long", 0) if "BUY" in d.get("signal","") else d.get("tp2_short", 0),
                "risk_usd": round(risk_amount, 2),
            })
    return portfolio

# ─────────────────────────────────────────────
#  MAIN SIGNAL ENGINE (all engines combined)
# ─────────────────────────────────────────────
def compute_signal(name, ticker):
    now = time.time()
    if ticker in _cache and now - _cache_time.get(ticker, 0) < CACHE_TTL:
        return _cache[ticker]
    try:
        data = yf.download(ticker, period="30d", interval="15m", progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError("No data")

        close  = data["Close"].squeeze()
        high   = data["High"].squeeze()
        low    = data["Low"].squeeze()
        volume = data["Volume"].squeeze()
        openp  = data["Open"].squeeze()

        price      = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        change_pct = round((price - prev_price) / prev_price * 100, 2)
        change_abs = round(price - prev_price, 3)

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

        recent_high = float(high.tail(96).max())
        recent_low  = float(low.tail(96).min())
        fib_range   = recent_high - recent_low

        candle_body = abs(float(close.iloc[-1]) - float(openp.iloc[-1]))
        avg_range   = float(high.tail(20).mean()) - float(low.tail(20).mean())

        # ── Scoring (ai_signal_engine) ──
        score = 0
        if rsi > 60:    score += 20
        elif rsi < 40:  score -= 20
        if ema20 > ema50 > ema200:  score += 30
        elif ema20 > ema50:         score += 18
        elif ema20 < ema50 < ema200:score -= 30
        elif ema20 < ema50:         score -= 18
        if macd_line > macd_sig and macd_hist > 0: score += 15
        elif macd_line < macd_sig and macd_hist < 0: score -= 15
        if stoch_k > stoch_d and stoch_k < 80: score += 10
        elif stoch_k < stoch_d and stoch_k > 20: score -= 10
        if last_vol > avg_vol: score += 8

        if score >= 60:    signal, signal_class = "STRONG BUY",  "strong-buy"
        elif score >= 25:  signal, signal_class = "BUY",          "buy"
        elif score <= -60: signal, signal_class = "STRONG SELL",  "strong-sell"
        elif score <= -25: signal, signal_class = "SELL",         "sell"
        else:              signal, signal_class = "NEUTRAL",      "hold"

        bias      = "LONG"  if ema20 > ema50 else "SHORT"
        structure = "BULLISH" if ema20 > ema50 else "BEARISH"
        trend_str = "UPTREND (M15)" if ema20 > ema50 else "DOWNTREND (M15)"
        orderflow = "BUYING PRESSURE" if ema20 > ema50 else "SELLING PRESSURE"
        confidence = min(abs(score), 99)

        # ── Sub-engines ──
        whale_data  = whale_track(ticker)
        scalp_data  = get_scalp(rsi, ema20, ema50, last_vol, avg_vol)
        liq_data    = get_liquidity(price, recent_high, recent_low, last_vol, avg_vol)
        ob          = get_orderblock(float(close.iloc[-1]), float(openp.iloc[-1]),
                                     ema20, last_vol, avg_vol,
                                     float(high.tail(20).mean()), float(low.tail(20).mean()))
        hf          = get_hedgefund_score(rsi, ema20, ema50, last_vol, avg_vol,
                                          price, candle_body, avg_range)
        narrative   = get_narrative(score, rsi, ema20, ema50, last_vol, avg_vol, price)
        macro       = get_macro(rsi, ema20, ema50, last_vol, avg_vol)

        # Smart Money (smartmoney_engine)
        if last_vol > avg_vol * 1.5:
            sm_pct, sm_label, institutional, retail = 78, "BULLISH" if ema20>ema50 else "BEARISH", 78, 22
        elif last_vol > avg_vol:
            sm_pct, sm_label, institutional, retail = 55, "NEUTRAL", 55, 45
        else:
            sm_pct, sm_label, institutional, retail = 30, "WEAK", 30, 70
        delta_vol = int(last_vol - avg_vol)

        # Key Levels
        resistance1 = round(recent_high, 3)
        resistance2 = round(recent_high + atr * 1.5, 3)
        support1    = round(recent_low + atr, 3)
        support2    = round(recent_low, 3)
        spread      = round(atr * 0.02, 2)

        # TP/SL
        sl_long   = round(price - atr * 1.5, 3)
        tp1_long  = round(price + atr * 2.0, 3)
        tp2_long  = round(price + atr * 3.0, 3)
        tp3_long  = round(price + atr * 4.5, 3)
        sl_short  = round(price + atr * 1.5, 3)
        tp1_short = round(price - atr * 2.0, 3)
        tp2_short = round(price - atr * 3.0, 3)
        tp3_short = round(price - atr * 4.5, 3)

        result = {
            "name": name, "ticker": ticker,
            "price": round(price, 3), "prev_price": round(prev_price, 3),
            "change_pct": change_pct, "change_abs": change_abs,
            "signal": signal, "signal_class": signal_class,
            "confidence": confidence, "score": score,
            "bias": bias, "structure": structure, "trend_str": trend_str, "orderflow": orderflow,
            "rsi": round(rsi, 2), "ema20": round(ema20, 3), "ema50": round(ema50, 3), "ema200": round(ema200, 3),
            "macd": round(macd_hist, 4), "macd_line": round(macd_line, 4), "macd_sig": round(macd_sig, 4),
            "atr": round(atr, 3), "spread": spread,
            "bb_upper": round(bb_upper, 3), "bb_lower": round(bb_lower, 3),
            "bb_mid": round(bb_mid, 3), "bb_pos": bb_pos,
            "stoch_k": round(stoch_k, 1), "stoch_d": round(stoch_d, 1),
            "smart_money_pct": sm_pct, "smart_money": sm_label,
            "institutional": institutional, "retail_sent": retail, "delta_vol": delta_vol,
            "resistance1": resistance1, "resistance2": resistance2,
            "support1": support1, "support2": support2,
            "recent_high": round(recent_high, 3), "recent_low": round(recent_low, 3),
            "sl_long": sl_long, "tp1_long": tp1_long, "tp2_long": tp2_long, "tp3_long": tp3_long,
            "sl_short": sl_short, "tp1_short": tp1_short, "tp2_short": tp2_short, "tp3_short": tp3_short,
            "narrative": narrative, "macro": macro, "orderblock": ob,
            **whale_data, **scalp_data, **liq_data, **hf,
            "update_time": datetime.now().strftime("%H:%M:%S"),
        }
        _cache[ticker]      = result
        _cache_time[ticker] = now
        return result

    except Exception as e:
        return {
            "name": name, "ticker": ticker, "price": 0, "prev_price": 0,
            "change_pct": 0, "change_abs": 0, "signal": "ERROR", "signal_class": "hold",
            "confidence": 0, "score": 0, "bias": "N/A", "structure": "N/A",
            "trend_str": "N/A", "orderflow": "N/A", "rsi": 0,
            "ema20": 0, "ema50": 0, "ema200": 0, "macd": 0, "macd_line": 0, "macd_sig": 0,
            "atr": 0, "spread": 0, "bb_upper": 0, "bb_lower": 0, "bb_mid": 0, "bb_pos": 50,
            "stoch_k": 0, "stoch_d": 0, "smart_money_pct": 0, "smart_money": "N/A",
            "institutional": 0, "retail_sent": 0, "delta_vol": 0,
            "resistance1": 0, "resistance2": 0, "support1": 0, "support2": 0,
            "recent_high": 0, "recent_low": 0,
            "sl_long": 0, "tp1_long": 0, "tp2_long": 0, "tp3_long": 0,
            "sl_short": 0, "tp1_short": 0, "tp2_short": 0, "tp3_short": 0,
            "narrative": "N/A", "macro": "N/A", "orderblock": "N/A",
            "whale": "N/A", "vol_ratio": 1, "whale_conf": 0,
            "scalp": "N/A", "scalp_score": 0,
            "liquidity": "N/A", "liq_score": 0,
            "hf_score": 0, "hf_signal": "N/A", "hf_trend": "N/A",
            "hf_whale": "N/A", "hf_liquidity": "N/A", "hf_smartmoney": "N/A", "hf_macro": "N/A",
            "update_time": "N/A",
        }

# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    market_data = {}
    for name, info in ASSETS.items():
        d = compute_signal(name, info["ticker"])
        d["tv"]   = info["tv"]
        d["type"] = info["type"]
        market_data[name] = d

    gold         = market_data.get("XAU/USD", {})
    news_data    = get_news()
    corr_data    = get_correlations(market_data)
    portfolio    = get_portfolio(market_data)
    warroom, bulls, bears = get_warroom_bias(market_data)
    buy_count    = sum(1 for d in market_data.values() if "BUY"  in d["signal"])
    sell_count   = sum(1 for d in market_data.values() if "SELL" in d["signal"])
    hold_count   = len(market_data) - buy_count - sell_count
    now_str      = datetime.now().strftime("%H:%M:%S")
    date_str     = datetime.now().strftime("%A, %d %B %Y").upper()

    return render_template("index.html",
        data=market_data, gold=gold,
        news=news_data, corr=corr_data,
        portfolio=portfolio, warroom=warroom,
        bulls=bulls, bears=bears,
        buy_count=buy_count, sell_count=sell_count, hold_count=hold_count,
        now=now_str, date=date_str,
    )

@app.route("/api/data")
def api_data():
    out = {}
    for name, info in ASSETS.items():
        d = compute_signal(name, info["ticker"])
        d["tv"] = info["tv"]
        out[name] = d
    return jsonify(out)


@app.route("/api/asset/<path:key>")
def api_asset(key):
    key = key.replace("_", "/").upper()
    info = ASSETS.get(key)
    if not info:
        return jsonify({"error": "not found"}), 404
    d = compute_signal(key, info["ticker"])
    d["tv"] = info["tv"]
    return jsonify(d)

@app.route("/api/news")
def api_news():
    return jsonify(get_news())

@app.route("/api/warroom")
def api_warroom():
    out = {}
    for name, info in ASSETS.items():
        d = compute_signal(name, info["ticker"])
        out[name] = {k: d[k] for k in ["signal","score","hf_signal","hf_score","narrative","macro"]}
    warroom, bulls, bears = get_warroom_bias(out)
    return jsonify({"warroom": warroom, "bulls": bulls, "bears": bears, "assets": out})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
