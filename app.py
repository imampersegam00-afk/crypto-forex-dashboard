from flask import Flask, render_template, jsonify
import yfinance as yf
import ta
import pandas as pd
import numpy as np
import feedparser
import requests
import zipfile
import io
import csv
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

# ── DEFAULT FOCUS: GOLD (XAU/USD) ─────────────────────────────
ASSETS = {
    "XAU/USD":  {"ticker": "GC=F",     "tv": "XAUUSD",     "type": "commodity"},
    "BTC/USD":  {"ticker": "BTC-USD",  "tv": "BTCUSD",     "type": "crypto"},
    "ETH/USD":  {"ticker": "ETH-USD",  "tv": "ETHUSD",     "type": "crypto"},
    "EUR/USD":  {"ticker": "EURUSD=X", "tv": "EURUSD",     "type": "forex"},
    "NASDAQ":   {"ticker": "^IXIC",    "tv": "NASDAQ:NDX", "type": "index"},
    "DXY":      {"ticker": "DX-Y.NYB", "tv": "TVC:DXY",   "type": "forex"},
    "OIL":      {"ticker": "CL=F",     "tv": "USOIL",      "type": "commodity"},
}
DEFAULT_PAIR = "XAU/USD"

CORR_PAIRS = [
    ("XAU/USD", "DXY"),
    ("XAU/USD", "BTC/USD"),
    ("BTC/USD", "NASDAQ"),
    ("OIL",     "NASDAQ"),
]

_cache      = {}
_cache_time = {}
_corr_cache = {}
_corr_time  = 0
CACHE_TTL   = 90

NEWS_SOURCES = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F,XAUUSD&region=US&lang=en-US",
    "https://www.forexlive.com/feed/news",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://rss.cnn.com/rss/money_news_international.rss",
    "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
]

_news_cache = {"data": [], "ts": 0}

# ── CME GROUP FUTURES DATA ─────────────────────────────────────
CME_FUTURES = {
    "GC=F":  {"name": "Gold",          "label": "XAU FUTURES",   "icon": "🥇"},
    "SI=F":  {"name": "Silver",        "label": "XAG FUTURES",   "icon": "🪙"},
    "CL=F":  {"name": "Crude Oil",     "label": "WTI FUTURES",   "icon": "🛢"},
    "ES=F":  {"name": "S&P500 E-mini", "label": "ES FUTURES",    "icon": "📈"},
    "NQ=F":  {"name": "Nasdaq E-mini", "label": "NQ FUTURES",    "icon": "💻"},
    "BTC=F": {"name": "Bitcoin CME",   "label": "BTC FUTURES",   "icon": "₿"},
    "ETH=F": {"name": "Ethereum CME",  "label": "ETH FUTURES",   "icon": "⟠"},
    "6E=F":  {"name": "EUR/USD",       "label": "EUR FUTURES",   "icon": "€"},
    "ZB=F":  {"name": "30Y Bond",      "label": "BOND FUTURES",  "icon": "🏦"},
}
_cme_cache = {"data": [], "ts": 0}

# ── CFTC COT REPORT (Gold COMEX - Official Data) ───────────────
_cot_cache = {"data": {}, "ts": 0}
COT_TTL = 3600 * 6  # refresh every 6 hours

def get_cot_gold():
    """Fetch CFTC Commitments of Traders Report for Gold COMEX (100 troy oz)"""
    now = time.time()
    if now - _cot_cache["ts"] < COT_TTL and _cot_cache["data"]:
        return _cot_cache["data"]
    try:
        url = "https://www.cftc.gov/files/dea/history/deacot2026.zip"
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open(z.namelist()[0]) as f:
            reader = csv.DictReader(io.TextIOWrapper(f))
            rows = list(reader)

        # Filter GOLD COMEX 100 troy oz (code 088691)
        gold_rows = [
            row for row in rows
            if row.get("Market and Exchange Names", "").strip() == "GOLD - COMMODITY EXCHANGE INC."
        ]
        gold_rows.sort(key=lambda x: x.get("As of Date in Form YYYY-MM-DD", ""))

        if not gold_rows:
            return {}

        latest = gold_rows[-1]

        def iv(k): 
            try: return int(latest.get(k, "0").replace(",", "").strip())
            except: return 0
        def fv(k):
            try: return float(latest.get(k, "0").replace(",", "").strip())
            except: return 0.0

        oi        = iv("Open Interest (All)")
        nc_long   = iv("Noncommercial Positions-Long (All)")
        nc_short  = iv("Noncommercial Positions-Short (All)")
        nc_spread = iv("Noncommercial Positions-Spreading (All)")
        com_long  = iv("Commercial Positions-Long (All)")
        com_short = iv("Commercial Positions-Short (All)")
        ret_long  = iv("Nonreportable Positions-Long (All)")
        ret_short = iv("Nonreportable Positions-Short (All)")
        chg_oi    = iv("Change in Open Interest (All)")
        chg_nc_l  = iv("Change in Noncommercial-Long (All)")
        chg_nc_s  = iv("Change in Noncommercial-Short (All)")
        traders   = iv("Traders-Total (All)")
        traders_l = iv("Traders-Noncommercial-Long (All)")
        traders_s = iv("Traders-Noncommercial-Short (All)")

        # Net positions
        nc_net    = nc_long - nc_short
        com_net   = com_long - com_short

        # Long/Short ratio
        total_long  = nc_long + com_long + ret_long
        total_short = nc_short + com_short + ret_short
        ls_ratio    = round(total_long / total_short, 2) if total_short > 0 else 0

        # Sentiment
        if nc_net > 50000:  sentiment = "STRONGLY BULLISH"
        elif nc_net > 20000: sentiment = "BULLISH"
        elif nc_net < -50000: sentiment = "STRONGLY BEARISH"
        elif nc_net < -20000: sentiment = "BEARISH"
        else: sentiment = "NEUTRAL"

        data = {
            "report_date":  latest.get("As of Date in Form YYYY-MM-DD", ""),
            "source":       "CFTC COMEX (100 troy oz)",
            "oi":           oi,
            "chg_oi":       chg_oi,
            "nc_long":      nc_long,
            "nc_short":     nc_short,
            "nc_spread":    nc_spread,
            "nc_net":       nc_net,
            "chg_nc_long":  chg_nc_l,
            "chg_nc_short": chg_nc_s,
            "com_long":     com_long,
            "com_short":    com_short,
            "com_net":      com_net,
            "ret_long":     ret_long,
            "ret_short":    ret_short,
            "total_long":   total_long,
            "total_short":  total_short,
            "ls_ratio":     ls_ratio,
            "pct_nc_long":  round(fv("% of OI-Noncommercial-Long (All)"), 1),
            "pct_nc_short": round(fv("% of OI-Noncommercial-Short (All)"), 1),
            "pct_com_long": round(fv("% of OI-Commercial-Long (All)"), 1),
            "pct_com_short":round(fv("% of OI-Commercial-Short (All)"), 1),
            "traders_total":traders,
            "traders_long": traders_l,
            "traders_short":traders_s,
            "sentiment":    sentiment,
            "contract_unit":"100 troy oz per contract",
        }
        _cot_cache["data"] = data
        _cot_cache["ts"]   = now
        return data
    except Exception as e:
        return {"error": str(e)}

# ── GOLD CME LIVE DATA (Open Interest, Volume, Bid/Ask) ────────
_gold_cme_cache = {"data": {}, "ts": 0}

def get_gold_cme_live():
    now = time.time()
    if now - _gold_cme_cache["ts"] < 120 and _gold_cme_cache["data"]:
        return _gold_cme_cache["data"]
    try:
        gc = yf.Ticker("GC=F")
        info = gc.info
        fi   = gc.fast_info
        data = {
            "price":       round(fi.last_price, 2) if hasattr(fi, "last_price") else "N/A",
            "prev_close":  round(fi.previous_close, 2) if hasattr(fi, "previous_close") else "N/A",
            "open_interest": info.get("openInterest", "N/A"),
            "volume":      info.get("volume", "N/A"),
            "bid":         info.get("bid", "N/A"),
            "ask":         info.get("ask", "N/A"),
            "week52_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "week52_low":  info.get("fiftyTwoWeekLow", "N/A"),
            "exchange":    info.get("exchange", "CMX"),
            "contract":    "100 Troy Oz",
            "currency":    "USD",
        }
        if data["price"] != "N/A" and data["prev_close"] != "N/A":
            chg = round(data["price"] - data["prev_close"], 2)
            pct = round((chg / data["prev_close"]) * 100, 2)
            data["chg"]  = chg
            data["pct"]  = pct
            data["dir"]  = "▲" if chg >= 0 else "▼"
            data["color"]= "#00ff88" if chg >= 0 else "#ff3366"
        _gold_cme_cache["data"] = data
        _gold_cme_cache["ts"]   = now
        return data
    except Exception as e:
        return {"error": str(e)}

def get_news():
    now = time.time()
    if now - _news_cache["ts"] < 300 and _news_cache["data"]:
        return _news_cache["data"]
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
    _news_cache["data"] = news[:12]
    _news_cache["ts"]   = now
    return _news_cache["data"]

def get_cme_futures():
    now = time.time()
    if now - _cme_cache["ts"] < 120:
        return _cme_cache["data"]
    result = []
    for sym, info in CME_FUTURES.items():
        try:
            t  = yf.Ticker(sym)
            fi = t.fast_info
            price = fi.last_price
            prev  = fi.previous_close
            if price and prev and prev != 0:
                chg = round(price - prev, 4)
                pct = round((chg / prev) * 100, 2)
                result.append({
                    "sym":   sym,
                    "name":  info["name"],
                    "label": info["label"],
                    "icon":  info["icon"],
                    "price": round(price, 4),
                    "chg":   chg,
                    "pct":   pct,
                    "dir":   "▲" if chg >= 0 else "▼",
                    "color": "var(--green)" if chg >= 0 else "var(--red)",
                })
        except:
            pass
    _cme_cache["data"] = result
    _cme_cache["ts"]   = now
    return result

def ml_predict(ticker):
    if not ML_AVAILABLE:
        return {"ml_signal": "N/A", "ml_accuracy": 0, "ml_confidence": 50}
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df is None or len(df) < 60:
            return {"ml_signal": "N/A", "ml_accuracy": 0, "ml_confidence": 50}
        df["ret"]    = df["Close"].pct_change()
        df["ma5"]    = df["Close"].rolling(5).mean()
        df["ma20"]   = df["Close"].rolling(20).mean()
        df["vol_ma"] = df["Volume"].rolling(10).mean()
        df["target"] = (df["ret"].shift(-1) > 0).astype(int)
        df.dropna(inplace=True)
        if len(df) < 40:
            return {"ml_signal": "N/A", "ml_accuracy": 0, "ml_confidence": 50}
        feats = ["ret","ma5","ma20","vol_ma"]
        X = df[feats].values
        y = df["target"].values
        split = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_tr, y_tr)
        acc  = round(accuracy_score(y_te, clf.predict(X_te)) * 100, 1)
        prob = clf.predict_proba(X[-1:].reshape(1,-1))[0]
        pred = int(clf.predict(X[-1:].reshape(1,-1))[0])
        conf = round(max(prob) * 100, 1)
        return {
            "ml_signal":     "BUY" if pred == 1 else "SELL",
            "ml_accuracy":   acc,
            "ml_confidence": conf,
        }
    except:
        return {"ml_signal": "N/A", "ml_accuracy": 0, "ml_confidence": 50}

def multi_tf(ticker):
    tfs = {}
    for tf, period in [("15m","5d"),("1h","30d"),("4h","60d"),("1d","180d")]:
        try:
            df = yf.download(ticker, period=period, interval=tf, progress=False)
            if df is None or len(df) < 20:
                tfs[tf] = "N/A"; continue
            close = df["Close"].values.flatten()
            rsi   = float(ta.momentum.RSIIndicator(pd.Series(close)).rsi().iloc[-1])
            ema20 = float(ta.trend.EMAIndicator(pd.Series(close),20).ema_indicator().iloc[-1])
            if   rsi > 65 and close[-1] > ema20: tfs[tf] = "BUY"
            elif rsi < 35 and close[-1] < ema20: tfs[tf] = "SELL"
            else:                                 tfs[tf] = "NEUTRAL"
        except:
            tfs[tf] = "N/A"
    return tfs

def whale_track(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="1h", progress=False)
        if df is None or len(df) < 10:
            return {"whale_signal": "N/A", "whale_volume": "N/A"}
        vol    = df["Volume"].values.flatten()
        avg_v  = float(np.mean(vol[-20:])) if len(vol)>=20 else float(np.mean(vol))
        last_v = float(vol[-1])
        ratio  = round(last_v / avg_v, 2) if avg_v > 0 else 1
        close  = df["Close"].values.flatten()
        chg    = (close[-1] - close[-2]) / close[-2] if len(close) >= 2 else 0
        if   ratio > 2.0 and chg > 0:  sig = "WHALE BUY 🐋"
        elif ratio > 2.0 and chg < 0:  sig = "WHALE SELL 🐋"
        elif ratio > 1.5:               sig = "HIGH VOLUME"
        else:                           sig = "NORMAL"
        return {"whale_signal": sig, "whale_volume": ratio}
    except:
        return {"whale_signal": "N/A", "whale_volume": "N/A"}

def get_correlations(market_data):
    now = time.time()
    global _corr_cache, _corr_time
    if now - _corr_time < 300 and _corr_cache:
        return _corr_cache
    corr = []
    for p1, p2 in CORR_PAIRS:
        try:
            t1 = ASSETS[p1]["ticker"]
            t2 = ASSETS[p2]["ticker"]
            d1 = yf.download(t1, period="30d", interval="1d", progress=False)["Close"].pct_change().dropna()
            d2 = yf.download(t2, period="30d", interval="1d", progress=False)["Close"].pct_change().dropna()
            mn = min(len(d1), len(d2))
            if mn < 10:
                continue
            c = float(np.corrcoef(d1.values[-mn:].flatten(), d2.values[-mn:].flatten())[0,1])
            c = round(c, 2)
            if   c >  0.7: rel = "STRONG +"
            elif c >  0.3: rel = "MODERATE +"
            elif c < -0.7: rel = "STRONG -"
            elif c < -0.3: rel = "MODERATE -"
            else:           rel = "WEAK"
            corr.append({"pair1": p1, "pair2": p2, "corr": c, "rel": rel})
        except:
            pass
    _corr_cache = corr
    _corr_time  = now
    return corr

def get_narrative(score, rsi, ema20, ema50, lv, av, price):
    if   score >= 4: return "STRONG BULLISH momentum — institutional accumulation detected"
    elif score >= 2: return "BULLISH bias — trend continuation expected"
    elif score <= -4: return "STRONG BEARISH pressure — smart money distribution"
    elif score <= -2: return "BEARISH bias — downside pressure building"
    else:             return "CONSOLIDATION — market seeking direction"

def get_macro(rsi, ema20, ema50, lv, av):
    if   rsi > 60 and ema20 > ema50: return "RISK-ON"
    elif rsi < 40 and ema20 < ema50: return "RISK-OFF"
    else:                             return "NEUTRAL"

def get_orderblock(close_last, open_last, ema20, lv, av, high_tail, low_tail):
    body = abs(close_last - open_last)
    if body > 0.001 * close_last and lv > av * 1.5:
        return "BULLISH OB" if close_last > open_last else "BEARISH OB"
    return "NO OB"

def get_scalp(rsi, ema20, ema50, lv, av):
    if   rsi < 30 and lv > av: return "SCALP BUY"
    elif rsi > 70 and lv > av: return "SCALP SELL"
    else:                       return "WAIT"

def get_liquidity(price, recent_high, recent_low, lv, av):
    range_ = recent_high - recent_low
    if   price > recent_high * 0.995 and lv > av * 1.5: return "RESISTANCE SWEEP"
    elif price < recent_low  * 1.005 and lv > av * 1.5: return "SUPPORT SWEEP"
    else:                                                 return "MID RANGE"

def get_hedgefund_score(rsi, ema20, ema50, lv, av, price, candle_body, avg_range):
    score = 0
    if price > ema20:  score += 1
    if price > ema50:  score += 1
    if rsi > 50:       score += 1
    if lv > av * 1.2:  score += 1
    if candle_body > avg_range * 0.6: score += 1
    if price < ema20:  score -= 1
    if price < ema50:  score -= 1
    if rsi < 50:       score -= 1
    hf_sig = "STRONG BUY" if score >= 4 else "BUY" if score >= 2 else \
             "STRONG SELL" if score <= -4 else "SELL" if score <= -2 else "NEUTRAL"
    conf = min(100, abs(score) * 20 + 40)
    return {"hf_signal": hf_sig, "hf_score": score, "hf_conf": conf}

def get_warroom_bias(market_data):
    bulls, bears = 0, 0
    for k, d in market_data.items():
        s = d.get("signal","")
        if "BUY" in s:  bulls += 1
        if "SELL" in s: bears += 1
    total = bulls + bears
    if total == 0: return "NEUTRAL", bulls, bears
    pct = bulls / total
    if   pct >= 0.7: return "RISK-ON 🟢", bulls, bears
    elif pct <= 0.3: return "RISK-OFF 🔴", bulls, bears
    else:            return "MIXED ⚪", bulls, bears

def get_portfolio(market_data, balance=10000, risk_pct=0.01):
    positions = []
    for name, d in market_data.items():
        sig   = d.get("signal", "")
        score = d.get("score",  0)
        price = d.get("price",  0)
        if "BUY" in sig and price > 0:
            risk_amt = balance * risk_pct
            sl_dist  = price * 0.01
            size     = round(risk_amt / sl_dist, 4) if sl_dist > 0 else 0
            positions.append({"pair": name, "side": "BUY", "size": size, "price": price, "score": score})
        elif "SELL" in sig and price > 0:
            risk_amt = balance * risk_pct
            sl_dist  = price * 0.01
            size     = round(risk_amt / sl_dist, 4) if sl_dist > 0 else 0
            positions.append({"pair": name, "side": "SELL", "size": size, "price": price, "score": score})
    return sorted(positions, key=lambda x: abs(x["score"]), reverse=True)[:5]

def compute_signal(name, ticker):
    now = time.time()
    if name in _cache and now - _cache_time.get(name, 0) < CACHE_TTL:
        return _cache[name]
    try:
        df = yf.download(ticker, period="5d", interval="15m", progress=False)
        if df is None or len(df) < 30:
            return _cache.get(name, {"signal":"N/A","score":0,"price":0,"name":name})

        close = df["Close"].values.flatten().astype(float)
        high  = df["High"].values.flatten().astype(float)
        low   = df["Low"].values.flatten().astype(float)
        vol   = df["Volume"].values.flatten().astype(float)
        open_ = df["Open"].values.flatten().astype(float)

        s  = pd.Series(close)
        sv = pd.Series(vol)

        rsi   = float(ta.momentum.RSIIndicator(s).rsi().iloc[-1])
        ema20 = float(ta.trend.EMAIndicator(s,20).ema_indicator().iloc[-1])
        ema50 = float(ta.trend.EMAIndicator(s,50).ema_indicator().iloc[-1])
        macd_obj = ta.trend.MACD(s)
        macd  = float(macd_obj.macd().iloc[-1])
        macd_sig = float(macd_obj.macd_signal().iloc[-1])
        bb    = ta.volatility.BollingerBands(s)
        bb_h  = float(bb.bollinger_hband().iloc[-1])
        bb_l  = float(bb.bollinger_lband().iloc[-1])
        bb_m  = float(bb.bollinger_mavg().iloc[-1])
        stoch = ta.momentum.StochasticOscillator(pd.Series(high), pd.Series(low), s)
        stoch_k = float(stoch.stoch().iloc[-1])
        stoch_d = float(stoch.stoch_signal().iloc[-1])
        atr   = float(ta.volatility.AverageTrueRange(pd.Series(high), pd.Series(low), s).average_true_range().iloc[-1])

        price = float(close[-1])
        lv    = float(vol[-1])
        av    = float(np.mean(vol[-20:]))
        recent_high = float(np.max(high[-20:]))
        recent_low  = float(np.min(low[-20:]))
        candle_body = abs(float(close[-1]) - float(open_[-1]))
        avg_range   = float(np.mean(np.abs(close[-20:] - open_[-20:])))

        # TP/SL
        if price > ema20:
            tp1 = round(price + atr * 1.5, 2)
            tp2 = round(price + atr * 3.0, 2)
            tp3 = round(price + atr * 5.0, 2)
            sl  = round(price - atr * 1.0, 2)
        else:
            tp1 = round(price - atr * 1.5, 2)
            tp2 = round(price - atr * 3.0, 2)
            tp3 = round(price - atr * 5.0, 2)
            sl  = round(price + atr * 1.0, 2)

        score = 0
        if rsi < 35:   score += 2
        elif rsi < 45: score += 1
        elif rsi > 65: score -= 2
        elif rsi > 55: score -= 1
        if price > ema20: score += 1
        if price > ema50: score += 1
        if macd > macd_sig: score += 1
        else:               score -= 1
        if price < bb_l: score += 1
        if price > bb_h: score -= 1
        if stoch_k < 20: score += 1
        if stoch_k > 80: score -= 1

        if   score >= 4: signal = "STRONG BUY ⚡"
        elif score >= 2: signal = "BUY 📈"
        elif score <= -4:signal = "STRONG SELL ⚡"
        elif score <= -2:signal = "SELL 📉"
        else:            signal = "NEUTRAL ↔"

        hf     = get_hedgefund_score(rsi, ema20, ema50, lv, av, price, candle_body, avg_range)
        ml     = ml_predict(ticker)
        whale  = whale_track(ticker)
        mtf    = multi_tf(ticker)
        narrative = get_narrative(score, rsi, ema20, ema50, lv, av, price)
        macro  = get_macro(rsi, ema20, ema50, lv, av)
        ob     = get_orderblock(close[-1], open_[-1], ema20, lv, av, high[-1]-close[-1], close[-1]-low[-1])
        scalp  = get_scalp(rsi, ema20, ema50, lv, av)
        liq    = get_liquidity(price, recent_high, recent_low, lv, av)

        prev_close = float(close[-2]) if len(close) >= 2 else price
        chg_pct = round((price - prev_close) / prev_close * 100, 2) if prev_close > 0 else 0

        result = {
            "name":       name,
            "signal":     signal,
            "score":      score,
            "price":      round(price, 4),
            "chg_pct":    chg_pct,
            "rsi":        round(rsi, 1),
            "ema20":      round(ema20, 4),
            "ema50":      round(ema50, 4),
            "macd":       round(macd, 4),
            "macd_sig":   round(macd_sig, 4),
            "bb_h":       round(bb_h, 4),
            "bb_l":       round(bb_l, 4),
            "bb_m":       round(bb_m, 4),
            "stoch_k":    round(stoch_k, 1),
            "stoch_d":    round(stoch_d, 1),
            "atr":        round(atr, 4),
            "volume":     int(lv),
            "avg_volume": int(av),
            "tp1": tp1, "tp2": tp2, "tp3": tp3, "sl": sl,
            "recent_high": round(recent_high, 4),
            "recent_low":  round(recent_low, 4),
            "hf_signal":   hf["hf_signal"],
            "hf_score":    hf["hf_score"],
            "hf_conf":     hf["hf_conf"],
            "ml_signal":   ml["ml_signal"],
            "ml_accuracy": ml["ml_accuracy"],
            "ml_confidence": ml["ml_confidence"],
            "whale_signal":  whale["whale_signal"],
            "whale_volume":  whale["whale_volume"],
            "multi_tf":    mtf,
            "narrative":   narrative,
            "macro":       macro,
            "orderblock":  ob,
            "scalp":       scalp,
            "liquidity":   liq,
        }
        _cache[name]          = result
        _cache_time[name]     = now
        return result
    except Exception as e:
        return _cache.get(name, {"signal":"N/A","score":0,"price":0,"name":name,"error":str(e)})

# ══════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════
@app.route("/")
def home():
    d   = compute_signal(DEFAULT_PAIR, ASSETS[DEFAULT_PAIR]["ticker"])
    cme = get_cme_futures()
    cot = get_cot_gold()
    gold_cme = get_gold_cme_live()
    return render_template("index.html",
        d=d, cme=cme, cot=cot, gold_cme=gold_cme,
        assets=list(ASSETS.keys()),
        default_pair=DEFAULT_PAIR,
        default_tv=ASSETS[DEFAULT_PAIR]["tv"])

@app.route("/api/data")
def api_data():
    out = {}
    for name, info in ASSETS.items():
        out[name] = compute_signal(name, info["ticker"])
    return out

@app.route("/api/asset/<path:key>")
def api_asset(key):
    key  = key.replace("_", "/").upper()
    info = ASSETS.get(key)
    if not info:
        from flask import jsonify
        return jsonify({"error": "not found"}), 404
    d = compute_signal(key, info["ticker"])
    d["tv"] = info["tv"]
    from flask import jsonify
    return jsonify(d)

@app.route("/api/cme")
def api_cme():
    from flask import jsonify
    return jsonify(get_cme_futures())

@app.route("/api/cot")
def api_cot():
    from flask import jsonify
    return jsonify(get_cot_gold())

@app.route("/api/gold")
def api_gold():
    from flask import jsonify
    return jsonify(get_gold_cme_live())

@app.route("/api/news")
def api_news():
    from flask import jsonify
    return jsonify(get_news())

@app.route("/api/warroom")
def api_warroom():
    from flask import jsonify
    out = {}
    for name, info in ASSETS.items():
        d = compute_signal(name, info["ticker"])
        out[name] = {k: d[k] for k in ["signal","score","hf_signal","hf_score","narrative","macro"]}
    warroom, bulls, bears = get_warroom_bias(out)
    return jsonify({"bias": warroom, "bulls": bulls, "bears": bears, "detail": out})

@app.route("/api/portfolio")
def api_portfolio():
    from flask import jsonify
    out = {}
    for name, info in ASSETS.items():
        out[name] = compute_signal(name, info["ticker"])
    return jsonify(get_portfolio(out))

@app.route("/api/correlation")
def api_correlation():
    from flask import jsonify
    return jsonify(get_correlations({}))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
