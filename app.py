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

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════
#  ASSETS  (key → ticker/tv)
# ═══════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════
#  CACHE
# ═══════════════════════════════════════════════════════════════
_signal_cache = {}       # {pair: result}
_signal_ts    = {}       # {pair: timestamp}
SIGNAL_TTL    = 90

_cot_cache  = {"data": {}, "ts": 0}
_gold_cache = {"data": {}, "ts": 0}
_cme_cache  = {"data": [], "ts": 0}
_news_cache = {"data": [], "ts": 0}
_corr_cache = {"data": [], "ts": 0}

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def _s(v):
    """safe float from string"""
    try: return float(str(v).replace(",","").strip())
    except: return 0.0

def _i(v):
    try: return int(str(v).replace(",","").strip())
    except: return 0

def _fmt(n):
    try:
        n = float(n)
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
        if n >= 1_000:     return f"{n/1_000:.1f}K"
        return f"{int(n):,}"
    except: return "N/A"

def _chgsign(n):
    try: return ("+" if float(n)>=0 else "")+_fmt(n)
    except: return "N/A"

# ═══════════════════════════════════════════════════════════════
#  COT REPORT — CFTC (Gold COMEX 100 oz, code 088691)
# ═══════════════════════════════════════════════════════════════
def get_cot():
    now = time.time()
    if now - _cot_cache["ts"] < 21600 and _cot_cache["data"]:
        return _cot_cache["data"]
    try:
        r = requests.get(
            "https://www.cftc.gov/files/dea/history/deacot2026.zip",
            timeout=15, headers={"User-Agent": "Mozilla/5.0"}
        )
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open(z.namelist()[0]) as f:
            rows = list(csv.DictReader(io.TextIOWrapper(f)))

        gold = [row for row in rows
                if row.get("Market and Exchange Names","").strip()
                == "GOLD - COMMODITY EXCHANGE INC."]
        gold.sort(key=lambda x: x.get("As of Date in Form YYYY-MM-DD",""))
        if not gold: return {}

        lat   = gold[-1]
        nc_l  = _i(lat.get("Noncommercial Positions-Long (All)","0"))
        nc_s  = _i(lat.get("Noncommercial Positions-Short (All)","0"))
        com_l = _i(lat.get("Commercial Positions-Long (All)","0"))
        com_s = _i(lat.get("Commercial Positions-Short (All)","0"))
        ret_l = _i(lat.get("Nonreportable Positions-Long (All)","0"))
        ret_s = _i(lat.get("Nonreportable Positions-Short (All)","0"))
        oi    = _i(lat.get("Open Interest (All)","0"))
        chg   = _i(lat.get("Change in Open Interest (All)","0"))
        traders = _i(lat.get("Traders-Total (All)","0"))
        nc_net  = nc_l - nc_s

        if   nc_net >  100000: sent = "STRONGLY BULLISH"
        elif nc_net >   30000: sent = "BULLISH"
        elif nc_net < -100000: sent = "STRONGLY BEARISH"
        elif nc_net <  -30000: sent = "BEARISH"
        else:                  sent = "NEUTRAL"

        data = {
            "date": lat.get("As of Date in Form YYYY-MM-DD",""),
            # raw
            "nc_long": nc_l, "nc_short": nc_s,
            "com_long": com_l, "com_short": com_s,
            "ret_long": ret_l, "ret_short": ret_s,
            "oi": oi, "chg_oi": chg,
            "nc_net": nc_net, "traders": traders,
            # formatted
            "nc_long_f":  _fmt(nc_l),  "nc_short_f":  _fmt(nc_s),
            "com_long_f": _fmt(com_l), "com_short_f": _fmt(com_s),
            "ret_long_f": _fmt(ret_l), "ret_short_f": _fmt(ret_s),
            "oi_f":       _fmt(oi),    "chg_oi_f":    _chgsign(chg),
            "nc_net_f":   _chgsign(nc_net), "traders_f": _fmt(traders),
            "sentiment":  sent,
            # bar percentages (0-100)
            "nc_pct_long":  round(nc_l/(nc_l+nc_s)*100,1) if (nc_l+nc_s)>0 else 50,
            "com_pct_long": round(com_l/(com_l+com_s)*100,1) if (com_l+com_s)>0 else 50,
            "ret_pct_long": round(ret_l/(ret_l+ret_s)*100,1) if (ret_l+ret_s)>0 else 50,
        }
        _cot_cache.update({"data": data, "ts": now})
        return data
    except Exception as e:
        return {"error": str(e)}

# ═══════════════════════════════════════════════════════════════
#  GOLD CME LIVE (COMEX — yfinance GC=F)
# ═══════════════════════════════════════════════════════════════
def get_gold_live():
    now = time.time()
    if now - _gold_cache["ts"] < 120 and _gold_cache["data"]:
        return _gold_cache["data"]
    try:
        t    = yf.Ticker("GC=F")
        fi   = t.fast_info
        info = t.info
        price = round(fi.last_price, 2)
        prev  = round(fi.previous_close, 2)
        chg   = round(price - prev, 2)
        pct   = round(chg/prev*100, 2) if prev else 0
        oi    = info.get("openInterest", 0)
        vol   = info.get("volume", 0)
        data  = {
            "price": price, "prev": prev,
            "chg": chg, "pct": pct,
            "dir": "▲" if chg>=0 else "▼",
            "color": "#00ff88" if chg>=0 else "#ff3366",
            "oi": oi,   "oi_f":  _fmt(oi),
            "vol": vol, "vol_f": _fmt(vol),
            "bid":   info.get("bid","N/A"),
            "ask":   info.get("ask","N/A"),
            "high52":info.get("fiftyTwoWeekHigh","N/A"),
            "low52": info.get("fiftyTwoWeekLow","N/A"),
            "exchange": "COMEX/CME",
            "contract": "100 Troy Oz",
        }
        _gold_cache.update({"data": data, "ts": now})
        return data
    except Exception as e:
        return {"error": str(e)}

# ═══════════════════════════════════════════════════════════════
#  CME FUTURES STRIP
# ═══════════════════════════════════════════════════════════════
CME_LIST = [
    ("GC=F","Gold Futures","🥇"),
    ("SI=F","Silver","🪙"),
    ("CL=F","Crude Oil WTI","🛢"),
    ("ES=F","S&P500 E-mini","📈"),
    ("NQ=F","Nasdaq E-mini","💻"),
    ("BTC=F","Bitcoin CME","₿"),
    ("6E=F","EUR/USD Fut","€"),
    ("ZB=F","30Y Bond","🏦"),
]

def get_cme():
    now = time.time()
    if now - _cme_cache["ts"] < 120 and _cme_cache["data"]:
        return _cme_cache["data"]
    result = []
    for sym, name, icon in CME_LIST:
        try:
            fi  = yf.Ticker(sym).fast_info
            p   = fi.last_price
            pv  = fi.previous_close
            if p and pv and pv != 0:
                chg = round(p-pv, 4)
                pct = round(chg/pv*100, 2)
                result.append({
                    "sym": sym, "name": name, "icon": icon,
                    "price": round(p,2), "chg": chg, "pct": pct,
                    "dir":   "▲" if chg>=0 else "▼",
                    "color": "#00ff88" if chg>=0 else "#ff3366",
                })
        except: pass
    _cme_cache.update({"data": result, "ts": now})
    return result

# ═══════════════════════════════════════════════════════════════
#  NEWS
# ═══════════════════════════════════════════════════════════════
NEWS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US",
    "https://www.forexlive.com/feed/news",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://rss.cnn.com/rss/money_news_international.rss",
]

def get_news():
    now = time.time()
    if now - _news_cache["ts"] < 300 and _news_cache["data"]:
        return _news_cache["data"]
    items, seen = [], set()
    for url in NEWS_FEEDS:
        try:
            for e in feedparser.parse(url).entries[:4]:
                t = e.get("title","").strip()
                if t and t not in seen:
                    seen.add(t)
                    items.append({"title": t, "link": e.get("link","#"),
                                  "source": "CryptoForexNewsHub"})
        except: pass
        if len(items) >= 10: break
    _news_cache.update({"data": items[:10], "ts": now})
    return _news_cache["data"]

# ═══════════════════════════════════════════════════════════════
#  ENGINE FUNCTIONS (from AI_HEDGEFUND_SYSTEM logic)
# ═══════════════════════════════════════════════════════════════

def _engine_hedgefund(close, high, low, openp, volume):
    """hedgefund_mode.py — RF on RSI/EMA/ATR features"""
    if not ML_AVAILABLE: return {"hf_signal":"N/A","hf_score":0,"hf_conf":0}
    try:
        s = pd.Series(close)
        df = pd.DataFrame()
        df["RSI"]   = ta.momentum.RSIIndicator(s).rsi()
        df["EMA20"] = ta.trend.EMAIndicator(s,20).ema_indicator()
        df["EMA50"] = ta.trend.EMAIndicator(s,50).ema_indicator()
        df["ATR"]   = ta.volatility.AverageTrueRange(
            pd.Series(high),pd.Series(low),s).average_true_range()
        df["TARGET"]= np.where(s.shift(-1) > s, 1, 0)
        df.dropna(inplace=True)
        if len(df) < 40: return {"hf_signal":"N/A","hf_score":0,"hf_conf":0}
        X = df[["RSI","EMA20","EMA50","ATR"]].values
        y = df["TARGET"].values
        sp = int(len(X)*0.8)
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X[:sp], y[:sp])
        acc  = round(accuracy_score(y[sp:], clf.predict(X[sp:]))*100,1)
        pred = int(clf.predict(X[-1:].reshape(1,-1))[0])
        prob = clf.predict_proba(X[-1:].reshape(1,-1))[0]
        conf = round(max(prob)*100,1)
        # score logic from hedgefund_mode.py
        rsi   = float(df["RSI"].iloc[-1])
        ema20 = float(df["EMA20"].iloc[-1])
        ema50 = float(df["EMA50"].iloc[-1])
        lv    = float(volume[-1])
        av    = float(np.mean(volume[-20:]))
        cb    = abs(float(close[-1]) - float(openp[-1]))
        ar    = float(np.mean(high[-20:]) - np.mean(low[-20:]))
        sc    = 0
        trend = "BULLISH" if ema20>ema50 else "BEARISH"
        if ema20>ema50:       sc+=20
        if lv>av:             sc+=20
        if cb>ar*0.5:         sc+=20
        if rsi>55:            sc+=20
        if pred==1:           sc+=20
        if   sc>=85: sig="STRONG BUY"
        elif sc>=65: sig="BUY"
        elif sc<=30: sig="STRONG SELL"
        elif sc<=50: sig="SELL"
        else:        sig="NEUTRAL"
        return {"hf_signal":sig,"hf_score":sc,"hf_conf":conf,
                "hf_accuracy":acc,"hf_trend":trend}
    except:
        return {"hf_signal":"N/A","hf_score":0,"hf_conf":0}

def _engine_ml(ticker):
    """machine_learning_ai.py — 60d/15m RF predict"""
    if not ML_AVAILABLE: return {"ml_signal":"N/A","ml_accuracy":0,"ml_conf":50}
    try:
        df = yf.download(ticker, period="60d", interval="15m", progress=False)
        if df is None or len(df) < 60:
            return {"ml_signal":"N/A","ml_accuracy":0,"ml_conf":50}
        close = df["Close"].squeeze()
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()
        vol   = df["Volume"].squeeze()
        data  = pd.DataFrame()
        data["RSI"]   = ta.momentum.RSIIndicator(close).rsi()
        data["EMA20"] = ta.trend.EMAIndicator(close,20).ema_indicator()
        data["EMA50"] = ta.trend.EMAIndicator(close,50).ema_indicator()
        data["ATR"]   = ta.volatility.AverageTrueRange(high,low,close).average_true_range()
        data["TARGET"]= np.where(close.shift(-1)>close,1,0)
        data.dropna(inplace=True)
        if len(data)<40: return {"ml_signal":"N/A","ml_accuracy":0,"ml_conf":50}
        X  = data[["RSI","EMA20","EMA50","ATR"]].values
        y  = data["TARGET"].values
        sp = int(len(X)*0.8)
        clf= RandomForestClassifier(n_estimators=200,random_state=42)
        clf.fit(X[:sp],y[:sp])
        acc  = round(accuracy_score(y[sp:],clf.predict(X[sp:]))*100,1)
        pred = int(clf.predict(X[-1:].reshape(1,-1))[0])
        prob = clf.predict_proba(X[-1:].reshape(1,-1))[0]
        return {"ml_signal":"BUY" if pred==1 else "SELL",
                "ml_accuracy":acc,"ml_conf":round(max(prob)*100,1)}
    except:
        return {"ml_signal":"N/A","ml_accuracy":0,"ml_conf":50}

def _engine_whale(volume):
    """whale_tracker.py — volume ratio"""
    try:
        lv  = float(volume[-1])
        av  = float(np.mean(volume[-20:]))
        rat = round(lv/av,2) if av>0 else 1
        if   rat>=2.0: status="WHALE ACCUMULATION 🐋"
        elif rat<=0.5: status="WHALE DISTRIBUTION 🐋"
        else:          status="NORMAL"
        conf = min(round(rat*50,1),99)
        return {"whale":status,"whale_ratio":rat,"whale_conf":conf}
    except:
        return {"whale":"N/A","whale_ratio":0,"whale_conf":0}

def _engine_orderblock(close, openp, high, low, volume):
    """orderblock_engine.py"""
    try:
        s    = pd.Series(close)
        ema20= float(ta.trend.EMAIndicator(s,20).ema_indicator().iloc[-1])
        lv   = float(volume[-1])
        av   = float(np.mean(volume[-20:]))
        cb   = abs(float(close[-1])-float(openp[-1]))
        ar   = float(np.mean(high[-20:])-np.mean(low[-20:]))
        ob   = "NO ORDER BLOCK"
        sc   = 0
        if close[-1]>ema20 and lv>av and cb>ar*0.5:
            ob="BULLISH ORDER BLOCK 📦"; sc=80
        elif close[-1]<ema20 and lv>av and cb>ar*0.5:
            ob="BEARISH ORDER BLOCK 📦"; sc=80
        return {"orderblock":ob,"ob_conf":min(sc,99)}
    except:
        return {"orderblock":"N/A","ob_conf":0}

def _engine_sniper(close, high, low, volume):
    """sniper_engine.py — 5m ultra-precision"""
    try:
        s    = pd.Series(close)
        ph   = pd.Series(high)
        pl   = pd.Series(low)
        rsi  = float(ta.momentum.RSIIndicator(s,14).rsi().iloc[-1])
        ema20= float(ta.trend.EMAIndicator(s,20).ema_indicator().iloc[-1])
        ema50= float(ta.trend.EMAIndicator(s,50).ema_indicator().iloc[-1])
        atr_s= ta.volatility.AverageTrueRange(ph,pl,s).average_true_range()
        catr = float(atr_s.iloc[-1])
        aatr = float(atr_s.tail(20).mean())
        lv   = float(volume[-1])
        av   = float(np.mean(volume[-20:]))
        price= float(close[-1])
        sc   = 0
        if rsi>60:        sc+=20
        if ema20>ema50:   sc+=30
        if lv>av:         sc+=25
        if catr>aatr:     sc+=15
        if price>ema20:   sc+=10
        if   sc>=85: sig="SNIPER BUY 🎯"
        elif sc<=30: sig="SNIPER SELL 🎯"
        else:        sig="WAIT"
        tp1 = round(price+catr*1.5,2)
        tp2 = round(price+catr*3.0,2)
        sl  = round(price-catr,2)
        return {"sniper":sig,"sniper_conf":min(sc,99),"sniper_tp1":tp1,"sniper_tp2":tp2,"sniper_sl":sl}
    except:
        return {"sniper":"N/A","sniper_conf":0,"sniper_tp1":0,"sniper_tp2":0,"sniper_sl":0}

def _engine_sentiment(close, volume):
    """sentiment_engine.py"""
    try:
        s    = pd.Series(close)
        rsi  = float(ta.momentum.RSIIndicator(s).rsi().iloc[-1])
        ema20= float(ta.trend.EMAIndicator(s,20).ema_indicator().iloc[-1])
        ema50= float(ta.trend.EMAIndicator(s,50).ema_indicator().iloc[-1])
        lv   = float(volume[-1])
        av   = float(np.mean(volume[-20:]))
        sc   = 0
        if rsi>55:  sc+=1
        if ema20>ema50: sc+=1
        if lv>av:   sc+=1
        sent = "BULLISH" if sc>=3 else "BEARISH" if sc<=1 else "NEUTRAL"
        conf = round(sc/3*100,1)
        return {"sentiment":sent,"sentiment_conf":conf}
    except:
        return {"sentiment":"N/A","sentiment_conf":0}

def _engine_volatility(close, high, low, volume):
    """volatility_engine.py"""
    try:
        s    = pd.Series(close)
        ph   = pd.Series(high)
        pl   = pd.Series(low)
        atr_s= ta.volatility.AverageTrueRange(ph,pl,s).average_true_range()
        catr = float(atr_s.iloc[-1])
        aatr = float(atr_s.tail(20).mean())
        rat  = round(catr/aatr,2) if aatr>0 else 1
        lv   = float(volume[-1])
        av   = float(np.mean(volume[-20:]))
        sc   = 0
        if   rat>=1.5: status="VOLATILITY EXPLOSION ⚡"; sc+=60
        elif rat<=0.7: status="LOW VOLATILITY";          sc+=30
        else:          status="NORMAL VOLATILITY"
        if lv>av: sc+=30
        return {"volatility":status,"atr_ratio":rat,"vol_conf":min(sc,99)}
    except:
        return {"volatility":"N/A","atr_ratio":0,"vol_conf":0}

def _engine_scalping(close, high, low, volume):
    """scalping_engine.py — RSI7/EMA9/21"""
    try:
        s    = pd.Series(close)
        ph   = pd.Series(high)
        pl   = pd.Series(low)
        rsi  = float(ta.momentum.RSIIndicator(s,7).rsi().iloc[-1])
        ema9 = float(ta.trend.EMAIndicator(s,9).ema_indicator().iloc[-1])
        ema21= float(ta.trend.EMAIndicator(s,21).ema_indicator().iloc[-1])
        atr_s= ta.volatility.AverageTrueRange(ph,pl,s,7).average_true_range()
        catr = float(atr_s.iloc[-1])
        aatr = float(atr_s.tail(20).mean())
        lv   = float(volume[-1])
        av   = float(np.mean(volume[-20:]))
        price= float(close[-1])
        sc   = 0
        if rsi>55:   sc+=25
        if ema9>ema21:sc+=35
        if lv>av:    sc+=25
        if catr>aatr:sc+=15
        if   sc>=80: sig="SCALP BUY ⚡"
        elif sc<=30: sig="SCALP SELL ⚡"
        else:        sig="WAIT"
        tp  = round(price+catr*1.5,2)
        sl  = round(price-catr*0.8,2)
        return {"scalp":sig,"scalp_conf":min(sc,99),"scalp_tp":tp,"scalp_sl":sl}
    except:
        return {"scalp":"N/A","scalp_conf":0,"scalp_tp":0,"scalp_sl":0}

def _engine_smartmoney(close, high, low, openp, volume):
    """smartmoney_engine.py"""
    try:
        s    = pd.Series(close)
        ema20= float(ta.trend.EMAIndicator(s,20).ema_indicator().iloc[-1])
        ema50= float(ta.trend.EMAIndicator(s,50).ema_indicator().iloc[-1])
        rsi  = float(ta.momentum.RSIIndicator(s).rsi().iloc[-1])
        lv   = float(volume[-1])
        av   = float(np.mean(volume[-20:]))
        cb   = abs(float(close[-1])-float(openp[-1]))
        ar   = float(np.mean(high[-20:])-np.mean(low[-20:]))
        sc   = 0
        if ema20>ema50: sc+=30
        if lv>av:       sc+=30
        if cb>ar*0.5:   sc+=20
        if rsi>55:      sc+=20
        if   sc>=80: sm="SMART MONEY BULLISH 🏦"
        elif sc<=30: sm="SMART MONEY BEARISH 🏦"
        else:        sm="NEUTRAL"
        return {"smartmoney":sm,"sm_conf":min(sc,99)}
    except:
        return {"smartmoney":"N/A","sm_conf":0}

def _engine_narrative(close, volume, rsi, ema20, ema50):
    """narrative_ai.py"""
    try:
        lv = float(volume[-1])
        av = float(np.mean(volume[-20:]))
        sc = 0
        if rsi>60:      sc+=25
        if ema20>ema50: sc+=35
        if lv>av:       sc+=20
        if   sc>=70: nar="INSTITUTIONAL ACCUMULATION — STRONG BULLISH NARRATIVE"
        elif sc>=50: nar="MILD BULLISH NARRATIVE — TREND CONTINUATION"
        elif sc<=20: nar="DISTRIBUTION PHASE — BEARISH NARRATIVE"
        elif sc<=35: nar="RISK-OFF NARRATIVE — MARKET CORRECTION"
        else:        nar="MARKET WAITING — NO CLEAR NARRATIVE"
        return {"narrative":nar}
    except:
        return {"narrative":"N/A"}

def _engine_macro(rsi, ema20, ema50, lv, av):
    """macro_engine.py"""
    sc = 0
    if rsi>55:      sc+=30
    if ema20>ema50: sc+=40
    if lv>av:       sc+=30
    if   sc>=70: macro="RISK-ON 🟢"
    elif sc<=30: macro="RISK-OFF 🔴"
    else:        macro="NEUTRAL ⚪"
    return {"macro":macro}

def _engine_liquidity(price, high, low, volume):
    """liquidity_engine.py"""
    try:
        rh = float(np.max(high[-20:]))
        rl = float(np.min(low[-20:]))
        lv = float(volume[-1])
        av = float(np.mean(volume[-20:]))
        if   price>rh*0.995 and lv>av*1.5: liq="RESISTANCE SWEEP 🎯"
        elif price<rl*1.005 and lv>av*1.5: liq="SUPPORT SWEEP 🎯"
        elif price>(rh+rl)/2:              liq="UPPER RANGE"
        else:                               liq="LOWER RANGE"
        return {"liquidity":liq,"recent_high":round(rh,4),"recent_low":round(rl,4)}
    except:
        return {"liquidity":"N/A","recent_high":0,"recent_low":0}

def _engine_risk(price, atr, score):
    """risk_management_ai.py"""
    try:
        if price>0 and atr>0:
            if score>0:
                tp1=round(price+atr*1.5,2); tp2=round(price+atr*3,2); tp3=round(price+atr*5,2)
                sl =round(price-atr*1.0,2)
            else:
                tp1=round(price-atr*1.5,2); tp2=round(price-atr*3,2); tp3=round(price-atr*5,2)
                sl =round(price+atr*1.0,2)
            rr = round(abs(tp1-price)/abs(price-sl),2) if abs(price-sl)>0 else 0
            return {"tp1":tp1,"tp2":tp2,"tp3":tp3,"sl":sl,"rr":rr}
        return {"tp1":0,"tp2":0,"tp3":0,"sl":0,"rr":0}
    except:
        return {"tp1":0,"tp2":0,"tp3":0,"sl":0,"rr":0}

def _engine_multi_tf(ticker):
    """multi_timeframe_ai.py"""
    out = {}
    for tf, period in [("15m","5d"),("1h","30d"),("4h","60d"),("1d","180d")]:
        try:
            df = yf.download(ticker, period=period, interval=tf, progress=False)
            if df is None or len(df)<20: out[tf]="N/A"; continue
            c    = df["Close"].squeeze()
            rsi  = float(ta.momentum.RSIIndicator(c).rsi().iloc[-1])
            ema20= float(ta.trend.EMAIndicator(c,20).ema_indicator().iloc[-1])
            macd = ta.trend.MACD(c)
            m    = float(macd.macd().iloc[-1])
            ms   = float(macd.macd_signal().iloc[-1])
            sc   = 0
            if rsi>60:          sc+=1
            if c.iloc[-1]>ema20:sc+=1
            if m>ms:            sc+=1
            if   sc>=3: out[tf]="BUY"
            elif sc<=1: out[tf]="SELL"
            else:       out[tf]="NEUTRAL"
        except: out[tf]="N/A"
    return out

# ═══════════════════════════════════════════════════════════════
#  MASTER SIGNAL COMPUTE — orchestrates ALL engines
# ═══════════════════════════════════════════════════════════════
def compute(pair):
    now = time.time()
    if pair in _signal_cache and now - _signal_ts.get(pair,0) < SIGNAL_TTL:
        return _signal_cache[pair]

    ticker = ASSETS[pair]["ticker"]
    tv     = ASSETS[pair]["tv"]

    try:
        df = yf.download(ticker, period="5d", interval="15m", progress=False)
        if df is None or len(df) < 30:
            raise ValueError("not enough data")

        close  = df["Close"].squeeze().values.astype(float)
        high   = df["High"].squeeze().values.astype(float)
        low    = df["Low"].squeeze().values.astype(float)
        openp  = df["Open"].squeeze().values.astype(float)
        volume = df["Volume"].squeeze().values.astype(float)

        s = pd.Series(close)

        # Core indicators
        rsi    = float(ta.momentum.RSIIndicator(s,14).rsi().iloc[-1])
        ema20  = float(ta.trend.EMAIndicator(s,20).ema_indicator().iloc[-1])
        ema50  = float(ta.trend.EMAIndicator(s,50).ema_indicator().iloc[-1])
        macd_o = ta.trend.MACD(s)
        macd   = float(macd_o.macd().iloc[-1])
        macd_s = float(macd_o.macd_signal().iloc[-1])
        bb     = ta.volatility.BollingerBands(s)
        bb_h   = float(bb.bollinger_hband().iloc[-1])
        bb_l   = float(bb.bollinger_lband().iloc[-1])
        bb_m   = float(bb.bollinger_mavg().iloc[-1])
        stoch  = ta.momentum.StochasticOscillator(pd.Series(high),pd.Series(low),s)
        stoch_k= float(stoch.stoch().iloc[-1])
        atr_s  = ta.volatility.AverageTrueRange(pd.Series(high),pd.Series(low),s)
        atr    = float(atr_s.average_true_range().iloc[-1])

        price  = float(close[-1])
        prev   = float(close[-2]) if len(close)>=2 else price
        lv     = float(volume[-1])
        av     = float(np.mean(volume[-20:]))
        chg_pct= round((price-prev)/prev*100,2) if prev else 0
        vol_ratio= round(lv/av,2) if av>0 else 1

        # Master score (ai_signal_engine.py logic)
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

        # Run all sub-engines
        hf   = _engine_hedgefund(close,high,low,openp,volume)
        ml   = _engine_ml(ticker)
        wh   = _engine_whale(volume)
        ob   = _engine_orderblock(close,openp,high,low,volume)
        sn   = _engine_sniper(close,high,low,volume)
        sent = _engine_sentiment(close,volume)
        vl   = _engine_volatility(close,high,low,volume)
        sc_e = _engine_scalping(close,high,low,volume)
        sm   = _engine_smartmoney(close,high,low,openp,volume)
        nar  = _engine_narrative(close,volume,rsi,ema20,ema50)
        mac  = _engine_macro(rsi,ema20,ema50,lv,av)
        liq  = _engine_liquidity(price,high,low,volume)
        risk = _engine_risk(price,atr,score)
        mtf  = _engine_multi_tf(ticker)

        result = {
            # Identity
            "pair": pair, "ticker": ticker, "tv": tv,
            # Price
            "price": round(price,4), "prev": round(prev,4),
            "chg_pct": chg_pct,
            "vol_ratio": vol_ratio,
            # Master signal
            "signal": signal, "score": score,
            # Core indicators
            "rsi": round(rsi,1), "ema20": round(ema20,4), "ema50": round(ema50,4),
            "macd": round(macd,4), "macd_sig": round(macd_s,4),
            "bb_h": round(bb_h,4), "bb_l": round(bb_l,4), "bb_m": round(bb_m,4),
            "stoch_k": round(stoch_k,1), "atr": round(atr,4),
            "volume": int(lv), "avg_vol": int(av),
            # TP/SL
            "tp1": risk["tp1"], "tp2": risk["tp2"], "tp3": risk["tp3"],
            "sl": risk["sl"], "rr": risk["rr"],
            # Engines
            **hf, **ml, **wh, **ob, **sn, **sent,
            **vl, **sc_e, **sm, **nar, **mac, **liq,
            "recent_high": liq["recent_high"],
            "recent_low":  liq["recent_low"],
            "multi_tf": mtf,
        }
        _signal_cache[pair] = result
        _signal_ts[pair]    = now
        return result

    except Exception as e:
        empty = {
            "pair":pair,"ticker":ticker,"tv":tv,
            "price":0,"prev":0,"chg_pct":0,"vol_ratio":0,
            "signal":"N/A","score":0,
            "rsi":0,"ema20":0,"ema50":0,"macd":0,"macd_sig":0,
            "bb_h":0,"bb_l":0,"bb_m":0,"stoch_k":0,"atr":0,
            "volume":0,"avg_vol":0,"tp1":0,"tp2":0,"tp3":0,"sl":0,"rr":0,
            "hf_signal":"N/A","hf_score":0,"hf_conf":0,"hf_accuracy":0,"hf_trend":"N/A",
            "ml_signal":"N/A","ml_accuracy":0,"ml_conf":0,
            "whale":"N/A","whale_ratio":0,"whale_conf":0,
            "orderblock":"N/A","ob_conf":0,
            "sniper":"N/A","sniper_conf":0,"sniper_tp1":0,"sniper_tp2":0,"sniper_sl":0,
            "sentiment":"N/A","sentiment_conf":0,
            "volatility":"N/A","atr_ratio":0,"vol_conf":0,
            "scalp":"N/A","scalp_conf":0,"scalp_tp":0,"scalp_sl":0,
            "smartmoney":"N/A","sm_conf":0,
            "narrative":"N/A","macro":"N/A",
            "liquidity":"N/A","recent_high":0,"recent_low":0,
            "multi_tf":{},"error":str(e)
        }
        return _signal_cache.get(pair, empty)

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
                "rel":"STRONG +" if c>.7 else "MOD +" if c>.3 else
                      "STRONG -" if c<-.7 else "MOD -" if c<-.3 else "WEAK"})
        except: pass
    _corr_cache.update({"data":result,"ts":now})
    return result

# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════
@app.route("/")
def home():
    d    = compute(DEFAULT_PAIR)
    cot  = get_cot()
    gold = get_gold_live()
    cme  = get_cme()
    news = get_news()
    corr = get_corr()
    return render_template("index.html",
        d=d, cot=cot, gold=gold, cme=cme, news=news, corr=corr,
        assets=list(ASSETS.keys()),
        asset_tv={k:v["tv"] for k,v in ASSETS.items()},
        default_pair=DEFAULT_PAIR)

@app.route("/api/asset/<path:key>")
def api_asset(key):
    key = key.replace("_","/").upper()
    if key not in ASSETS:
        return jsonify({"error":"not found"}), 404
    return jsonify(compute(key))

@app.route("/api/cot")
def api_cot():   return jsonify(get_cot())

@app.route("/api/gold")
def api_gold():  return jsonify(get_gold_live())

@app.route("/api/cme")
def api_cme():   return jsonify(get_cme())

@app.route("/api/news")
def api_news():  return jsonify(get_news())

@app.route("/api/corr")
def api_corr():  return jsonify(get_corr())

@app.route("/api/all")
def api_all():
    return jsonify({p: compute(p) for p in ASSETS})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
