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

ASSETS = {
    "XAU/USD": {"ticker": "GC=F",     "tv": "XAUUSD"},
    "BTC/USD": {"ticker": "BTC-USD",  "tv": "BTCUSD"},
    "ETH/USD": {"ticker": "ETH-USD",  "tv": "ETHUSD"},
    "EUR/USD": {"ticker": "EURUSD=X", "tv": "EURUSD"},
    "NASDAQ":  {"ticker": "^IXIC",    "tv": "NASDAQ:NDX"},
    "DXY":     {"ticker": "DX-Y.NYB", "tv": "TVC:DXY"},
    "OIL":     {"ticker": "CL=F",     "tv": "USOIL"},
}
DEFAULT_PAIR = "XAU/USD"

_cache      = {}
_cache_time = {}
CACHE_TTL   = 90

_news_cache   = {"data": [], "ts": 0}
_cme_cache    = {"data": [], "ts": 0}
_cot_cache    = {"data": {}, "ts": 0}
_goldlv_cache = {"data": {}, "ts": 0}
_corr_cache   = {"data": [], "ts": 0}

NEWS_SOURCES = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US",
    "https://www.forexlive.com/feed/news",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://rss.cnn.com/rss/money_news_international.rss",
]

CME_FUTURES = {
    "GC=F":  {"name": "Gold Futures",    "icon": "🥇"},
    "SI=F":  {"name": "Silver Futures",  "icon": "🪙"},
    "CL=F":  {"name": "Crude Oil WTI",   "icon": "🛢"},
    "ES=F":  {"name": "S&P500 E-mini",   "icon": "📈"},
    "NQ=F":  {"name": "Nasdaq E-mini",   "icon": "💻"},
    "BTC=F": {"name": "Bitcoin CME",     "icon": "₿"},
    "6E=F":  {"name": "EUR/USD Futures", "icon": "€"},
    "ZB=F":  {"name": "30Y Bond",        "icon": "🏦"},
}

# ── helpers ──────────────────────────────────────────────────
def _iv(d, k):
    try: return int(str(d.get(k,"0")).replace(",","").strip())
    except: return 0

def _fv(d, k):
    try: return float(str(d.get(k,"0")).replace(",","").strip())
    except: return 0.0

def _fmt(n):
    if n is None or n == "N/A": return "N/A"
    try:
        n = float(n)
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
        if n >= 1_000:     return f"{n/1_000:.1f}K"
        return str(int(n))
    except: return str(n)

# ── COT Report (CFTC) ────────────────────────────────────────
def get_cot():
    now = time.time()
    if now - _cot_cache["ts"] < 21600 and _cot_cache["data"]:
        return _cot_cache["data"]
    try:
        r = requests.get("https://www.cftc.gov/files/dea/history/deacot2026.zip",
                         timeout=15, headers={"User-Agent":"Mozilla/5.0"})
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open(z.namelist()[0]) as f:
            rows = list(csv.DictReader(io.TextIOWrapper(f)))
        gold = sorted(
            [row for row in rows if row.get("Market and Exchange Names","").strip()
             == "GOLD - COMMODITY EXCHANGE INC."],
            key=lambda x: x.get("As of Date in Form YYYY-MM-DD","")
        )
        if not gold:
            return {}
        lat = gold[-1]
        nc_l  = _iv(lat,"Noncommercial Positions-Long (All)")
        nc_s  = _iv(lat,"Noncommercial Positions-Short (All)")
        com_l = _iv(lat,"Commercial Positions-Long (All)")
        com_s = _iv(lat,"Commercial Positions-Short (All)")
        ret_l = _iv(lat,"Nonreportable Positions-Long (All)")
        ret_s = _iv(lat,"Nonreportable Positions-Short (All)")
        oi    = _iv(lat,"Open Interest (All)")
        chg   = _iv(lat,"Change in Open Interest (All)")
        nc_net= nc_l - nc_s
        if   nc_net >  100000: sent = "STRONGLY BULLISH"
        elif nc_net >   30000: sent = "BULLISH"
        elif nc_net < -100000: sent = "STRONGLY BEARISH"
        elif nc_net <  -30000: sent = "BEARISH"
        else:                  sent = "NEUTRAL"
        data = {
            "date":      lat.get("As of Date in Form YYYY-MM-DD",""),
            "oi":        oi,  "oi_fmt":  _fmt(oi),
            "chg_oi":    chg, "chg_oi_fmt": ("+" if chg>=0 else "")+_fmt(chg),
            "nc_long":   nc_l,  "nc_long_fmt":  _fmt(nc_l),
            "nc_short":  nc_s,  "nc_short_fmt": _fmt(nc_s),
            "nc_net":    nc_net,"nc_net_fmt":   ("+" if nc_net>=0 else "")+_fmt(nc_net),
            "com_long":  com_l, "com_long_fmt":  _fmt(com_l),
            "com_short": com_s, "com_short_fmt": _fmt(com_s),
            "ret_long":  ret_l, "ret_long_fmt":  _fmt(ret_l),
            "ret_short": ret_s, "ret_short_fmt": _fmt(ret_s),
            "traders":   _iv(lat,"Traders-Total (All)"),
            "pct_nc_long":  _fv(lat,"% of OI-Noncommercial-Long (All)"),
            "pct_nc_short": _fv(lat,"% of OI-Noncommercial-Short (All)"),
            "pct_com_long": _fv(lat,"% of OI-Commercial-Long (All)"),
            "pct_com_short":_fv(lat,"% of OI-Commercial-Short (All)"),
            "sentiment": sent,
        }
        _cot_cache.update({"data": data, "ts": now})
        return data
    except Exception as e:
        return {"error": str(e)}

# ── Gold CME Live (COMEX) ────────────────────────────────────
def get_gold_live():
    now = time.time()
    if now - _goldlv_cache["ts"] < 120 and _goldlv_cache["data"]:
        return _goldlv_cache["data"]
    try:
        t  = yf.Ticker("GC=F")
        fi = t.fast_info
        info = t.info
        price = round(fi.last_price, 2)
        prev  = round(fi.previous_close, 2)
        chg   = round(price - prev, 2)
        pct   = round(chg / prev * 100, 2) if prev else 0
        data  = {
            "price":    price,
            "chg":      chg,
            "pct":      pct,
            "dir":      "▲" if chg >= 0 else "▼",
            "color":    "#00ff88" if chg >= 0 else "#ff3366",
            "oi":       info.get("openInterest","N/A"),
            "oi_fmt":   _fmt(info.get("openInterest","N/A")),
            "volume":   info.get("volume","N/A"),
            "vol_fmt":  _fmt(info.get("volume","N/A")),
            "bid":      info.get("bid","N/A"),
            "ask":      info.get("ask","N/A"),
            "high52":   info.get("fiftyTwoWeekHigh","N/A"),
            "low52":    info.get("fiftyTwoWeekLow","N/A"),
            "exchange": "COMEX/CME",
            "contract": "100 Troy Oz",
        }
        _goldlv_cache.update({"data": data, "ts": now})
        return data
    except Exception as e:
        return {"error": str(e)}

# ── CME Futures strip ────────────────────────────────────────
def get_cme():
    now = time.time()
    if now - _cme_cache["ts"] < 120 and _cme_cache["data"]:
        return _cme_cache["data"]
    result = []
    for sym, info in CME_FUTURES.items():
        try:
            fi = yf.Ticker(sym).fast_info
            p  = fi.last_price
            pv = fi.previous_close
            if p and pv and pv != 0:
                chg = round(p - pv, 4)
                pct = round(chg / pv * 100, 2)
                result.append({
                    "sym":   sym,
                    "name":  info["name"],
                    "icon":  info["icon"],
                    "price": round(p, 2),
                    "chg":   chg,
                    "pct":   pct,
                    "dir":   "▲" if chg >= 0 else "▼",
                    "color": "#00ff88" if chg >= 0 else "#ff3366",
                })
        except:
            pass
    _cme_cache.update({"data": result, "ts": now})
    return result

# ── News ─────────────────────────────────────────────────────
def get_news():
    now = time.time()
    if now - _news_cache["ts"] < 300 and _news_cache["data"]:
        return _news_cache["data"]
    news, seen = [], set()
    for url in NEWS_SOURCES:
        try:
            for e in feedparser.parse(url).entries[:4]:
                t = e.get("title","").strip()
                if t and t not in seen:
                    seen.add(t)
                    news.append({"title": t, "link": e.get("link","#"),
                                 "source": "CryptoForexNewsHub"})
        except: pass
        if len(news) >= 10: break
    _news_cache.update({"data": news[:10], "ts": now})
    return _news_cache["data"]

# ── ML predict ───────────────────────────────────────────────
def ml_predict(ticker):
    if not ML_AVAILABLE:
        return {"signal":"N/A","accuracy":0,"confidence":50}
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df is None or len(df) < 60:
            return {"signal":"N/A","accuracy":0,"confidence":50}
        df["ret"]  = df["Close"].pct_change()
        df["ma5"]  = df["Close"].rolling(5).mean()
        df["ma20"] = df["Close"].rolling(20).mean()
        df["volma"]= df["Volume"].rolling(10).mean()
        df["tgt"]  = (df["ret"].shift(-1) > 0).astype(int)
        df.dropna(inplace=True)
        if len(df) < 40:
            return {"signal":"N/A","accuracy":0,"confidence":50}
        X = df[["ret","ma5","ma20","volma"]].values
        y = df["tgt"].values
        sp = int(len(X)*0.8)
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X[:sp], y[:sp])
        acc  = round(accuracy_score(y[sp:], clf.predict(X[sp:]))*100,1)
        prob = clf.predict_proba(X[-1:].reshape(1,-1))[0]
        pred = int(clf.predict(X[-1:].reshape(1,-1))[0])
        return {"signal":"BUY" if pred==1 else "SELL",
                "accuracy":acc,"confidence":round(max(prob)*100,1)}
    except:
        return {"signal":"N/A","accuracy":0,"confidence":50}

# ── Multi TF ────────────────────────────────────────────────
def multi_tf(ticker):
    out = {}
    for tf,period in [("15m","5d"),("1h","30d"),("4h","60d"),("1d","180d")]:
        try:
            df = yf.download(ticker,period=period,interval=tf,progress=False)
            if df is None or len(df)<20: out[tf]="N/A"; continue
            c = df["Close"].values.flatten()
            rsi   = float(ta.momentum.RSIIndicator(pd.Series(c)).rsi().iloc[-1])
            ema20 = float(ta.trend.EMAIndicator(pd.Series(c),20).ema_indicator().iloc[-1])
            if   rsi>65 and c[-1]>ema20: out[tf]="BUY"
            elif rsi<35 and c[-1]<ema20: out[tf]="SELL"
            else: out[tf]="NEUTRAL"
        except: out[tf]="N/A"
    return out

# ── Main signal engine ───────────────────────────────────────
def compute(name, ticker):
    now = time.time()
    if name in _cache and now - _cache_time.get(name,0) < CACHE_TTL:
        return _cache[name]
    try:
        df = yf.download(ticker, period="5d", interval="15m", progress=False)
        if df is None or len(df) < 30:
            raise ValueError("not enough data")

        close = df["Close"].values.flatten().astype(float)
        high  = df["High"].values.flatten().astype(float)
        low   = df["Low"].values.flatten().astype(float)
        vol   = df["Volume"].values.flatten().astype(float)
        open_ = df["Open"].values.flatten().astype(float)

        s  = pd.Series(close)
        rsi    = float(ta.momentum.RSIIndicator(s).rsi().iloc[-1])
        ema20  = float(ta.trend.EMAIndicator(s,20).ema_indicator().iloc[-1])
        ema50  = float(ta.trend.EMAIndicator(s,50).ema_indicator().iloc[-1])
        macd_o = ta.trend.MACD(s)
        macd   = float(macd_o.macd().iloc[-1])
        macd_s = float(macd_o.macd_signal().iloc[-1])
        bb     = ta.volatility.BollingerBands(s)
        bb_h   = float(bb.bollinger_hband().iloc[-1])
        bb_l   = float(bb.bollinger_lband().iloc[-1])
        stoch  = ta.momentum.StochasticOscillator(pd.Series(high),pd.Series(low),s)
        stoch_k= float(stoch.stoch().iloc[-1])
        atr    = float(ta.volatility.AverageTrueRange(pd.Series(high),pd.Series(low),s).average_true_range().iloc[-1])

        price  = float(close[-1])
        lv     = float(vol[-1])
        av     = float(np.mean(vol[-20:]))
        rh     = float(np.max(high[-20:]))
        rl     = float(np.min(low[-20:]))
        prev   = float(close[-2]) if len(close)>=2 else price

        score = 0
        if rsi<35:    score+=2
        elif rsi<45:  score+=1
        elif rsi>65:  score-=2
        elif rsi>55:  score-=1
        if price>ema20: score+=1
        if price>ema50: score+=1
        if macd>macd_s: score+=1
        else:           score-=1
        if price<bb_l:  score+=1
        if price>bb_h:  score-=1
        if stoch_k<20:  score+=1
        if stoch_k>80:  score-=1

        if   score>=4:  signal="STRONG BUY ⚡"
        elif score>=2:  signal="BUY 📈"
        elif score<=-4: signal="STRONG SELL ⚡"
        elif score<=-2: signal="SELL 📉"
        else:           signal="NEUTRAL ↔"

        if price>ema20:
            tp1=round(price+atr*1.5,2); tp2=round(price+atr*3,2); tp3=round(price+atr*5,2)
            sl =round(price-atr*1.0,2)
        else:
            tp1=round(price-atr*1.5,2); tp2=round(price-atr*3,2); tp3=round(price-atr*5,2)
            sl =round(price+atr*1.0,2)

        hf_sig = ("STRONG BUY" if score>=4 else "BUY" if score>=2 else
                  "STRONG SELL" if score<=-4 else "SELL" if score<=-2 else "NEUTRAL")
        hf_conf= min(100, abs(score)*20+40)

        ml  = ml_predict(ticker)
        mtf = multi_tf(ticker)

        chg_pct = round((price-prev)/prev*100,2) if prev else 0

        result = {
            "name":    name,
            "ticker":  ticker,
            "signal":  signal,
            "score":   score,
            "price":   round(price,4),
            "chg_pct": chg_pct,
            "rsi":     round(rsi,1),
            "ema20":   round(ema20,4),
            "ema50":   round(ema50,4),
            "macd":    round(macd,4),
            "macd_sig":round(macd_s,4),
            "bb_h":    round(bb_h,4),
            "bb_l":    round(bb_l,4),
            "stoch_k": round(stoch_k,1),
            "atr":     round(atr,4),
            "volume":  int(lv),
            "avg_vol": int(av),
            "vol_ratio": round(lv/av,2) if av>0 else 1,
            "tp1":tp1,"tp2":tp2,"tp3":tp3,"sl":sl,
            "recent_high": round(rh,4),
            "recent_low":  round(rl,4),
            "hf_signal":   hf_sig,
            "hf_conf":     hf_conf,
            "ml_signal":   ml["signal"],
            "ml_accuracy": ml["accuracy"],
            "ml_conf":     ml["confidence"],
            "multi_tf":    mtf,
            "whale":       "WHALE 🐋" if lv>av*2 else "NORMAL",
            "scalp":       ("SCALP BUY" if rsi<30 and lv>av else
                            "SCALP SELL" if rsi>70 and lv>av else "WAIT"),
            "macro":       ("RISK-ON" if rsi>60 and ema20>ema50 else
                            "RISK-OFF" if rsi<40 and ema20<ema50 else "NEUTRAL"),
            "liquidity":   ("RESISTANCE SWEEP" if price>rh*0.995 and lv>av*1.5 else
                            "SUPPORT SWEEP" if price<rl*1.005 and lv>av*1.5 else "MID RANGE"),
        }
        _cache[name]      = result
        _cache_time[name] = now
        return result
    except Exception as e:
        blank = {k:v for k,v in {
            "name":name,"ticker":ticker,"signal":"N/A","score":0,"price":0,
            "chg_pct":0,"rsi":0,"ema20":0,"ema50":0,"macd":0,"macd_sig":0,
            "bb_h":0,"bb_l":0,"stoch_k":0,"atr":0,"volume":0,"avg_vol":0,
            "vol_ratio":0,"tp1":0,"tp2":0,"tp3":0,"sl":0,"recent_high":0,
            "recent_low":0,"hf_signal":"N/A","hf_conf":0,"ml_signal":"N/A",
            "ml_accuracy":0,"ml_conf":0,"multi_tf":{},"whale":"N/A",
            "scalp":"N/A","macro":"N/A","liquidity":"N/A","error":str(e)
        }.items()}
        return _cache.get(name, blank)

# ── Correlation ──────────────────────────────────────────────
def get_corr():
    now = time.time()
    if now - _corr_cache["ts"] < 300 and _corr_cache["data"]:
        return _corr_cache["data"]
    pairs = [("XAU/USD","DXY"),("XAU/USD","BTC/USD"),("BTC/USD","NASDAQ"),("OIL","NASDAQ")]
    result = []
    for p1,p2 in pairs:
        try:
            d1 = yf.download(ASSETS[p1]["ticker"],period="30d",interval="1d",progress=False)["Close"].pct_change().dropna()
            d2 = yf.download(ASSETS[p2]["ticker"],period="30d",interval="1d",progress=False)["Close"].pct_change().dropna()
            mn = min(len(d1),len(d2))
            if mn<10: continue
            c = float(np.corrcoef(d1.values[-mn:].flatten(),d2.values[-mn:].flatten())[0,1])
            c = round(c,2)
            result.append({"p1":p1,"p2":p2,"corr":c,
                "rel":("STRONG +" if c>.7 else "MOD +" if c>.3 else
                       "STRONG -" if c<-.7 else "MOD -" if c<-.3 else "WEAK")})
        except: pass
    _corr_cache.update({"data":result,"ts":now})
    return result

# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def home():
    d       = compute(DEFAULT_PAIR, ASSETS[DEFAULT_PAIR]["ticker"])
    cot     = get_cot()
    gold    = get_gold_live()
    cme     = get_cme()
    news    = get_news()
    corr    = get_corr()
    assets  = list(ASSETS.keys())
    return render_template("index.html",
        d=d, cot=cot, gold=gold, cme=cme, news=news, corr=corr,
        assets=assets, default_pair=DEFAULT_PAIR,
        default_tv=ASSETS[DEFAULT_PAIR]["tv"])

@app.route("/api/asset/<path:key>")
def api_asset(key):
    key  = key.replace("_","/").upper()
    info = ASSETS.get(key)
    if not info:
        return jsonify({"error":"not found"}), 404
    d = compute(key, info["ticker"])
    d["tv"] = info["tv"]
    return jsonify(d)

@app.route("/api/cot")
def api_cot():    return jsonify(get_cot())

@app.route("/api/gold")
def api_gold():   return jsonify(get_gold_live())

@app.route("/api/cme")
def api_cme():    return jsonify(get_cme())

@app.route("/api/news")
def api_news():   return jsonify(get_news())

@app.route("/api/corr")
def api_corr():   return jsonify(get_corr())

@app.route("/api/all")
def api_all():
    return jsonify({n: compute(n, ASSETS[n]["ticker"]) for n in ASSETS})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
