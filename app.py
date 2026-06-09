"""
TERMINAL ELITE — Institutional-Grade Crypto/Forex Dashboard
Version: 3.0 — Full Architecture Overhaul
"""
import os, time, threading, hashlib, json
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import requests
import feedparser
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════
#  ASSET REGISTRY — Single Source of Truth
#  ticker  = yfinance symbol
#  tv      = TradingView symbol
#  cat     = category
#  type    = asset type
#  drivers = key market drivers
# ══════════════════════════════════════════════════════════════
ASSETS = {
    # FOREX
    "EUR/USD": {"ticker":"EURUSD=X",  "tv":"EURUSD",   "cat":"FOREX",  "type":"Major Pair",  "base":"EUR","quote":"USD","drivers":"ECB policy, USD flow, Risk sentiment"},
    "GBP/USD": {"ticker":"GBPUSD=X",  "tv":"GBPUSD",   "cat":"FOREX",  "type":"Major Pair",  "base":"GBP","quote":"USD","drivers":"BoE policy, Brexit, USD flow"},
    "USD/JPY": {"ticker":"JPY=X",     "tv":"USDJPY",   "cat":"FOREX",  "type":"Major Pair",  "base":"USD","quote":"JPY","drivers":"BoJ policy, risk sentiment, yields"},
    "AUD/USD": {"ticker":"AUDUSD=X",  "tv":"AUDUSD",   "cat":"FOREX",  "type":"Major Pair",  "base":"AUD","quote":"USD","drivers":"Commodities, RBA, China"},
    "USD/CHF": {"ticker":"CHF=X",     "tv":"USDCHF",   "cat":"FOREX",  "type":"Major Pair",  "base":"USD","quote":"CHF","drivers":"Safe haven, SNB, risk-off"},
    "USD/CAD": {"ticker":"CAD=X",     "tv":"USDCAD",   "cat":"FOREX",  "type":"Major Pair",  "base":"USD","quote":"CAD","drivers":"Oil price, BoC, trade balance"},
    "NZD/USD": {"ticker":"NZDUSD=X",  "tv":"NZDUSD",   "cat":"FOREX",  "type":"Major Pair",  "base":"NZD","quote":"USD","drivers":"RBNZ, dairy, China trade"},
    # METALS
    "XAU/USD": {"ticker":"GC=F",      "tv":"XAUUSD",   "cat":"METALS", "type":"Precious Metal","base":"XAU","quote":"USD","drivers":"USD strength, real yields, risk-off"},
    "XAG/USD": {"ticker":"SI=F",      "tv":"XAGUSD",   "cat":"METALS", "type":"Precious Metal","base":"XAG","quote":"USD","drivers":"Industrial demand, USD, gold correlation"},
    # CRYPTO
    "BTC/USD": {"ticker":"BTC-USD",   "tv":"BTCUSDT",  "cat":"CRYPTO", "type":"Digital Asset","base":"BTC","quote":"USD","drivers":"Risk appetite, liquidity, macro"},
    "ETH/USD": {"ticker":"ETH-USD",   "tv":"ETHUSDT",  "cat":"CRYPTO", "type":"Digital Asset","base":"ETH","quote":"USD","drivers":"DeFi, network upgrades, BTC correlation"},
    "SOL/USD": {"ticker":"SOL-USD",   "tv":"SOLUSDT",  "cat":"CRYPTO", "type":"Digital Asset","base":"SOL","quote":"USD","drivers":"DeFi, network activity, risk appetite"},
    "BNB/USD": {"ticker":"BNB-USD",   "tv":"BNBUSDT",  "cat":"CRYPTO", "type":"Digital Asset","base":"BNB","quote":"USD","drivers":"Binance ecosystem, BTC correlation"},
    # INDICES
    "NASDAQ":  {"ticker":"NQ=F",      "tv":"NAS100",   "cat":"INDEX",  "type":"Equity Index", "base":"NAS","quote":"USD","drivers":"Tech earnings, Fed rates, growth"},
    "SP500":   {"ticker":"ES=F",      "tv":"SPX500",   "cat":"INDEX",  "type":"Equity Index", "base":"SP5","quote":"USD","drivers":"Corporate earnings, Fed, macro"},
    "DXY":     {"ticker":"DX-Y.NYB",  "tv":"DXY",      "cat":"INDEX",  "type":"Currency Index","base":"DXY","quote":"","drivers":"Fed policy, US macro, safe haven"},
    "DJI":     {"ticker":"YM=F",      "tv":"DJ30",     "cat":"INDEX",  "type":"Equity Index", "base":"DJI","quote":"USD","drivers":"Blue chip earnings, macro"},
    # COMMODITIES
    "OIL":     {"ticker":"CL=F",      "tv":"USOIL",    "cat":"CMDTY",  "type":"Commodity",    "base":"WTI","quote":"USD","drivers":"OPEC, supply/demand, geopolitics"},
    "BRENT":   {"ticker":"BZ=F",      "tv":"UKOIL",    "cat":"CMDTY",  "type":"Commodity",    "base":"BRT","quote":"USD","drivers":"OPEC+, global demand, refinery"},
    "NATGAS":  {"ticker":"NG=F",      "tv":"NATURALGAS","cat":"CMDTY", "type":"Commodity",    "base":"NG", "quote":"USD","drivers":"Weather, storage, LNG export"},
}

CATEGORIES = {"FOREX":[], "METALS":[], "CRYPTO":[], "INDEX":[], "CMDTY":[]}
for k,v in ASSETS.items():
    CATEGORIES[v["cat"]].append(k)

DEFAULT_PAIR  = "XAU/USD"
CAT_DEFAULT   = {"FOREX":"EUR/USD","METALS":"XAU/USD","CRYPTO":"BTC/USD","INDEX":"NASDAQ","CMDTY":"OIL"}

# TradingView map for JS
TV_MAP = {k: v["tv"] for k,v in ASSETS.items()}

# ══════════════════════════════════════════════════════════════
#  CACHE SYSTEM — TTL-based per asset
# ══════════════════════════════════════════════════════════════
_sig_cache   = {}
_sig_ts      = {}
SIG_TTL      = 60   # seconds

_news_cache  = None
_news_ts     = 0
NEWS_TTL     = 300

_cot_cache   = None
_cot_ts      = 0
COT_TTL      = 21600

_gold_cache  = None
_gold_ts     = 0
GOLD_TTL     = 60

_cme_cache   = None
_cme_ts      = 0
CME_TTL      = 120

_corr_cache  = None
_corr_ts     = 0
CORR_TTL     = 120

def _cached_asset(pair):
    now = time.time()
    if pair in _sig_cache and (now - _sig_ts.get(pair,0)) < SIG_TTL:
        return _sig_cache[pair]
    return None

def _set_asset_cache(pair, data):
    _sig_cache[pair] = data
    _sig_ts[pair]    = time.time()

# ══════════════════════════════════════════════════════════════
#  CME FUTURES STRIP
# ══════════════════════════════════════════════════════════════
CME_LIST = [
    ("GC=F","Gold","🥇"),("SI=F","Silver","🪙"),("CL=F","WTI Oil","🛢"),
    ("ES=F","S&P500","📊"),("NQ=F","NASDAQ","💻"),("BTC=F","BTC Fut","₿"),
    ("ETH=F","ETH Fut","⟠"),("6E=F","EUR Fut","€"),("ZB=F","30Y Bond","📜"),
]

# ══════════════════════════════════════════════════════════════
#  NEWS FEEDS — multi-source institutional
# ══════════════════════════════════════════════════════════════
NEWS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=EURUSD%3DX&region=US&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?region=US&lang=en-US",
    "https://www.forexlive.com/feed/news",
    "https://www.investing.com/rss/news_301.rss",
    "https://www.fxstreet.com/rss/news",
]

# Asset keyword mapping for news relevance
ASSET_NEWS_MAP = {
    "gold":"XAU/USD","silver":"XAG/USD","xau":"XAU/USD","xag":"XAG/USD",
    "bitcoin":"BTC/USD","btc":"BTC/USD","ethereum":"ETH/USD","eth":"ETH/USD",
    "fed":"DXY","dollar":"DXY","dxy":"DXY","usd":"DXY",
    "oil":"OIL","crude":"OIL","opec":"OIL","brent":"BRENT",
    "nasdaq":"NASDAQ","tech":"NASDAQ","sp500":"SP500","s&p":"SP500",
    "euro":"EUR/USD","eur":"EUR/USD","gbp":"GBP/USD","pound":"GBP/USD",
    "yen":"USD/JPY","jpy":"USD/JPY","aud":"AUD/USD","cad":"USD/CAD",
}

SENTIMENT_KEYWORDS = {
    "bullish":["rally","surge","jump","gain","rise","strong","bullish","higher","beat","soar","breakout"],
    "bearish":["drop","fall","decline","weak","bearish","lower","miss","crash","selloff","breakdown"],
    "high_impact":["fed","fomc","inflation","gdp","nfp","jobs","opec","war","crisis","rate","decision"],
}

def _news_sentiment(title):
    t = title.lower()
    if any(w in t for w in SENTIMENT_KEYWORDS["bearish"]): return "BEARISH","🔴"
    if any(w in t for w in SENTIMENT_KEYWORDS["bullish"]): return "BULLISH","🟢"
    return "NEUTRAL","⚪"

def _news_impact(title):
    t = title.lower()
    if any(w in t for w in SENTIMENT_KEYWORDS["high_impact"]): return "HIGH","🔴"
    return "MEDIUM","🟡"

def _news_affected_assets(title):
    t = title.lower()
    assets = []
    for kw, pair in ASSET_NEWS_MAP.items():
        if kw in t and pair not in assets:
            assets.append(pair)
    return assets[:4] if assets else ["General"]

def _news_summary(title):
    """Generate institutional 2-line summary from headline"""
    t = title.lower()
    sent, _ = _news_sentiment(title)
    impact, _ = _news_impact(title)
    assets = _news_affected_assets(title)

    # Context sentences
    if "fed" in t or "fomc" in t or "rate" in t:
        if sent=="BULLISH":
            return "Fed signals dovish stance, supporting risk assets. Expect USD weakness, Gold and equities may rally."
        return "Fed hawkish signals boost USD. Pressure on Gold, crypto, and risk assets. DXY strength expected."
    if "inflation" in t or "cpi" in t:
        if sent=="BULLISH":
            return "Inflation data cools, raising rate cut expectations. Risk-on environment supports equities and Gold."
        return "Inflation beats expectations, keeping Fed restrictive. USD strengthens, equities face pressure."
    if "oil" in t or "crude" in t or "opec" in t:
        if sent=="BULLISH":
            return "Oil supply constraints support energy prices. CAD may strengthen, commodity currencies supported."
        return "Oil weakness signals demand slowdown. CAD under pressure, risk-off tone for commodity currencies."
    if "bitcoin" in t or "btc" in t or "crypto" in t:
        if sent=="BULLISH":
            return "Crypto market shows bullish momentum. Risk-on appetite supports BTC, ETH, and altcoins broadly."
        return "Crypto sentiment weakens. Risk-off pressure may spread to broader risk assets."
    if "gold" in t or "xau" in t:
        if sent=="BULLISH":
            return "Gold supported by safe-haven demand or USD weakness. Watch yield levels for confirmation."
        return "Gold faces headwinds from USD strength or risk appetite. Watch support levels closely."
    if sent=="BULLISH":
        return f"Positive development for {', '.join(assets[:2])}. Market participants may increase risk exposure."
    if sent=="BEARISH":
        return f"Risk factors emerging for {', '.join(assets[:2])}. Defensive positioning may increase."
    return f"Market monitoring {', '.join(assets[:2])} for directional cues. Await confirmation signals."

def get_news():
    global _news_cache, _news_ts
    now = time.time()
    if _news_cache and (now - _news_ts) < NEWS_TTL:
        return _news_cache
    items = []
    for url in NEWS_FEEDS[:4]:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:3]:
                title = e.get("title","")[:120]
                if not title: continue
                sent, sent_icon = _news_sentiment(title)
                impact, impact_icon = _news_impact(title)
                affected = _news_affected_assets(title)
                summary  = _news_summary(title)
                items.append({
                    "title":    title,
                    "summary":  summary,
                    "sentiment":sent,
                    "sent_icon":sent_icon,
                    "impact":   impact,
                    "imp_icon": impact_icon,
                    "assets":   affected,
                    "time":     e.get("published","")[:16],
                    "link":     e.get("link","#"),
                })
                if len(items) >= 15: break
        except: continue
    if not items:
        items = [{"title":"Markets steady — monitoring key levels","summary":"No major catalyst detected. Markets range-bound. Monitor technical levels for breakout cues.",
                  "sentiment":"NEUTRAL","sent_icon":"⚪","impact":"LOW","imp_icon":"⚪","assets":["General"],"time":"","link":"#"}]
    _news_cache = items[:12]
    _news_ts    = now
    return _news_cache

# ══════════════════════════════════════════════════════════════
#  COT REPORT
# ══════════════════════════════════════════════════════════════
def get_cot():
    global _cot_cache, _cot_ts
    now = time.time()
    if _cot_cache and (now - _cot_ts) < COT_TTL:
        return _cot_cache
    try:
        url = "https://www.cftc.gov/dea/futures/deacomesf.htm"
        headers = {"User-Agent":"Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=8)
        lines = r.text.split('\n')
        gold_line = next((l for l in lines if 'GOLD' in l.upper() and ',' in l), None)
        if gold_line:
            parts = gold_line.split(',')
            if len(parts) > 10:
                def fmt(v):
                    try: return f"{int(v.strip()):,}"
                    except: return v.strip()
                nc_long  = int(parts[7].strip().replace('"',''))
                nc_short = int(parts[8].strip().replace('"',''))
                oi       = int(parts[3].strip().replace('"',''))
                chg_oi   = int(parts[4].strip().replace('"','')) if len(parts)>4 else 0
                sentiment= "STRONGLY BULLISH" if nc_long > nc_short*3 else "BULLISH" if nc_long > nc_short else "BEARISH"
                _cot_cache = {
                    "date":parts[2].strip().replace('"',''),
                    "nc_long_f":f"{nc_long/1000:.1f}K","nc_short_f":f"{nc_short/1000:.1f}K",
                    "oi_f":f"{oi/1000:.1f}K","chg_oi":chg_oi,"chg_oi_f":f"{chg_oi:+,}",
                    "traders_f":parts[11].strip() if len(parts)>11 else "N/A",
                    "com_long":parts[5].strip() if len(parts)>5 else "0",
                    "sentiment":sentiment,
                    "nc_long":nc_long,"nc_short":nc_short,
                }
                _cot_ts = now
                return _cot_cache
    except: pass
    try:
        df = yf.download("GC=F", period="5d", interval="1d", progress=False)
        price = float(df["Close"].squeeze().iloc[-1]) if df is not None and len(df)>0 else 0
        _cot_cache = {"date":"N/A","nc_long_f":"N/A","nc_short_f":"N/A","oi_f":"N/A",
                      "chg_oi":0,"chg_oi_f":"N/A","traders_f":"N/A","com_long":"0",
                      "sentiment":"N/A","nc_long":0,"nc_short":0}
    except: pass
    if not _cot_cache: _cot_cache = {"date":"N/A","nc_long_f":"N/A","nc_short_f":"N/A","oi_f":"N/A",
                                      "chg_oi":0,"chg_oi_f":"N/A","traders_f":"N/A","com_long":"0",
                                      "sentiment":"N/A","nc_long":0,"nc_short":0}
    _cot_ts = now
    return _cot_cache

# ══════════════════════════════════════════════════════════════
#  GOLD CME LIVE
# ══════════════════════════════════════════════════════════════
def get_gold_live():
    global _gold_cache, _gold_ts
    now = time.time()
    if _gold_cache and (now - _gold_ts) < GOLD_TTL:
        return _gold_cache
    try:
        t = yf.Ticker("GC=F")
        fi = t.fast_info
        info = {}
        try: info = t.info
        except: pass
        price   = float(getattr(fi,"last_price",0) or info.get("regularMarketPrice",0))
        prev    = float(getattr(fi,"previous_close",price) or price)
        chg     = round(price - prev, 2)
        chg_pct = round((chg/prev*100), 2) if prev else 0
        ask     = float(getattr(fi,"ask",0) or info.get("ask",0) or round(price+0.3,1))
        bid     = float(getattr(fi,"bid",0) or info.get("bid",0) or round(price-0.3,1))
        oi      = int(info.get("openInterest",0) or 0)
        vol     = int(getattr(fi,"three_month_average_volume",0) or info.get("volume",0) or 0)
        h52     = float(getattr(fi,"year_high",0) or info.get("fiftyTwoWeekHigh",0))
        l52     = float(getattr(fi,"year_low",0) or info.get("fiftyTwoWeekLow",0))
        _gold_cache = {
            "price":round(price,1),"chg":chg,"chg_pct":chg_pct,
            "bid":round(bid,1),"ask":round(ask,1),
            "oi_f":f"{oi/1000:.1f}K" if oi else "N/A",
            "vol_f":f"{vol/1000:.1f}K" if vol else "N/A",
            "h52":round(h52,1),"l52":round(l52,1),
        }
        _gold_ts = now
        return _gold_cache
    except Exception as e:
        return {"price":0,"chg":0,"chg_pct":0,"bid":0,"ask":0,"oi_f":"N/A","vol_f":"N/A","h52":0,"l52":0}

# ══════════════════════════════════════════════════════════════
#  CME FUTURES STRIP
# ══════════════════════════════════════════════════════════════
def get_cme():
    global _cme_cache, _cme_ts
    now = time.time()
    if _cme_cache and (now - _cme_ts) < CME_TTL:
        return _cme_cache
    out = []
    for sym, name, icon in CME_LIST:
        try:
            df = yf.download(sym, period="2d", interval="1d", progress=False)
            if df is None or len(df)<1: out.append({"sym":sym,"name":name,"icon":icon,"price":"N/A","chg":"—","pct":"—"}); continue
            c = df["Close"].squeeze()
            price = float(c.iloc[-1])
            prev  = float(c.iloc[-2]) if len(c)>=2 else price
            chg   = round(price-prev,2); pct=round((chg/prev*100),2) if prev else 0
            out.append({"sym":sym,"name":name,"icon":icon,
                        "price":f"{price:,.2f}","chg":f"{chg:+.2f}","pct":f"{pct:+.2f}%",
                        "up":chg>=0})
        except: out.append({"sym":sym,"name":name,"icon":icon,"price":"N/A","chg":"—","pct":"—"})
    _cme_cache = out
    _cme_ts    = now
    return _cme_cache

# ══════════════════════════════════════════════════════════════
#  CORRELATION ENGINE
# ══════════════════════════════════════════════════════════════
def get_corr():
    global _corr_cache, _corr_ts
    now = time.time()
    if _corr_cache and (now - _corr_ts) < CORR_TTL:
        return _corr_cache
    pairs_corr = [("XAU/USD","GC=F"),("BTC/USD","BTC-USD"),("EUR/USD","EURUSD=X"),
                  ("OIL","CL=F"),("NASDAQ","NQ=F")]
    try:
        data = {}
        for name, sym in pairs_corr:
            df = yf.download(sym, period="20d", interval="1d", progress=False)
            if df is not None and len(df)>10:
                data[name] = df["Close"].squeeze().pct_change().dropna()
        out = []
        ref = "XAU/USD"
        if ref in data:
            for pair, _ in pairs_corr:
                if pair == ref: continue
                if pair in data:
                    s1 = data[ref]; s2 = data[pair]
                    mn = min(len(s1),len(s2))
                    c  = round(float(np.corrcoef(s1.values[-mn:],s2.values[-mn:])[0,1]),2)
                    tag = "🟢 Strong" if abs(c)>0.7 else "🟡 Moderate" if abs(c)>0.4 else "🔴 Weak"
                    out.append({"pair":pair,"corr":c,"tag":tag,"up":c>=0})
        _corr_cache = out
        _corr_ts    = now
        return _corr_cache
    except: return []

# ══════════════════════════════════════════════════════════════
#  CURRENCY STRENGTH ENGINE (Forex flow)
# ══════════════════════════════════════════════════════════════
def get_currency_strength():
    """Calculate relative currency strength index 0-100"""
    pairs_map = {
        "EUR":["EURUSD=X","EURGBP=X","EURJPY=X"],
        "USD":["EURUSD=X","GBPUSD=X","AUDUSD=X"],
        "GBP":["GBPUSD=X","EURGBP=X","GBPJPY=X"],
        "JPY":["JPY=X","GBPJPY=X","EURJPY=X"],
        "AUD":["AUDUSD=X","AUDCAD=X"],
        "CAD":["CAD=X","AUDCAD=X"],
        "CHF":["CHF=X","EURCHF=X"],
        "NZD":["NZDUSD=X"],
    }
    strength = {}
    try:
        all_syms = ["EURUSD=X","GBPUSD=X","AUDUSD=X","JPY=X","CHF=X","CAD=X","NZDUSD=X"]
        for sym in all_syms:
            df = yf.download(sym, period="5d", interval="1h", progress=False)
            if df is None or len(df)<10: continue
            c = df["Close"].squeeze()
            chg = float((c.iloc[-1] - c.iloc[0]) / c.iloc[0] * 100)
            # Map to currencies
            if "EUR" in sym: strength["EUR"] = strength.get("EUR",0) + chg
            if "GBP" in sym: strength["GBP"] = strength.get("GBP",0) + chg
            if "AUD" in sym: strength["AUD"] = strength.get("AUD",0) + chg
            if "NZD" in sym: strength["NZD"] = strength.get("NZD",0) + chg
            if sym == "JPY=X": strength["USD"] = strength.get("USD",0) + chg; strength["JPY"] = strength.get("JPY",0) - chg
            if sym == "CHF=X": strength["USD"] = strength.get("USD",0) + chg; strength["CHF"] = strength.get("CHF",0) - chg
            if sym == "CAD=X": strength["USD"] = strength.get("USD",0) + chg; strength["CAD"] = strength.get("CAD",0) - chg
    except: pass
    # Normalize to 0-100
    if strength:
        mn = min(strength.values()); mx = max(strength.values())
        rng = mx - mn if mx != mn else 1
        strength = {k: round((v-mn)/rng*100,1) for k,v in strength.items()}
    return strength

# ══════════════════════════════════════════════════════════════
#  ANALYTICAL ENGINES
# ══════════════════════════════════════════════════════════════

def _eng_hf(close, high, low, openp, volume):
    try:
        s = pd.Series(close); ph = pd.Series(high); pl = pd.Series(low)
        score = 0
        rsi  = float(ta.momentum.RSIIndicator(s,14).rsi().iloc[-1])
        ema9 = float(ta.trend.EMAIndicator(s,9).ema_indicator().iloc[-1])
        ema21= float(ta.trend.EMAIndicator(s,21).ema_indicator().iloc[-1])
        macd = ta.trend.MACD(s); mh = float(macd.macd_histogram().iloc[-1])
        atr  = float(ta.volatility.AverageTrueRange(ph,pl,s).average_true_range().iloc[-1])
        price= float(close[-1])
        if rsi < 40: score += 2
        elif rsi > 60: score -= 2
        if ema9 > ema21: score += 2
        else: score -= 2
        if mh > 0: score += 1
        else: score -= 1
        if price > ema21: score += 1
        else: score -= 1
        sig = "STRONG BUY" if score>=5 else "BUY" if score>=2 else "STRONG SELL" if score<=-5 else "SELL" if score<=-2 else "HOLD"
        return {"hf_signal":sig,"hf_score":min(abs(score)*12+35,100)}
    except: return {"hf_signal":"N/A","hf_score":0}

def _eng_ml(ticker):
    try:
        df = yf.download(ticker, period="60d", interval="1h", progress=False)
        if df is None or len(df)<50: return {"ml_signal":"N/A","ml_accuracy":0}
        c = df["Close"].squeeze(); h = df["High"].squeeze(); l = df["Low"].squeeze()
        s = pd.Series(c.values)
        f1 = ta.momentum.RSIIndicator(s,14).rsi().values
        f2 = ta.trend.EMAIndicator(s,20).ema_indicator().values
        f3 = ta.trend.EMAIndicator(s,50).ema_indicator().values
        f4 = ta.trend.MACD(s).macd_histogram().values
        feats = np.column_stack([f1,f2,f3,f4])
        target= (s.shift(-3).values > s.values).astype(int)
        valid = ~(np.isnan(feats).any(axis=1) | np.isnan(target))
        feats = feats[valid]; target = target[valid]
        if len(feats)<30: return {"ml_signal":"N/A","ml_accuracy":0}
        X_tr,X_te,y_tr,y_te = train_test_split(feats,target,test_size=0.2,random_state=42)
        clf = RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=1)
        clf.fit(X_tr,y_tr)
        acc  = round(clf.score(X_te,y_te)*100,1)
        pred = clf.predict([feats[-1]])[0]
        sig  = "BUY" if pred==1 else "SELL"
        return {"ml_signal":sig,"ml_accuracy":acc}
    except: return {"ml_signal":"N/A","ml_accuracy":0}

def _eng_whale(volume):
    try:
        v = volume[-1]; av = float(np.mean(volume[-20:]))
        ratio = v/av if av>0 else 1
        if ratio > 3.0: return {"whale":"WHALE ALERT 🐋"}
        if ratio > 2.0: return {"whale":"HIGH VOLUME ⚠️"}
        if ratio > 1.5: return {"whale":"ELEVATED VOL"}
        return {"whale":"NORMAL"}
    except: return {"whale":"N/A"}

def _eng_orderblock(close, openp, high, low, volume):
    try:
        for i in range(len(close)-3, max(0,len(close)-20), -1):
            body = abs(close[i]-openp[i])
            rng  = high[i]-low[i]
            if rng>0 and body/rng>0.7 and volume[i]>np.mean(volume[-20:])*1.5:
                if close[i] > openp[i]: return {"orderblock":"BULL ORDER BLOCK 🟢"}
                else: return {"orderblock":"BEAR ORDER BLOCK 🔴"}
        return {"orderblock":"NO ORDER BLOCK"}
    except: return {"orderblock":"N/A"}

def _eng_sniper(close, high, low, volume):
    try:
        s=pd.Series(close); ph=pd.Series(high); pl=pd.Series(low)
        rsi = float(ta.momentum.RSIIndicator(s,14).rsi().iloc[-1])
        atr = float(ta.volatility.AverageTrueRange(ph,pl,s).average_true_range().iloc[-1])
        price=float(close[-1]); ph20=float(np.max(high[-20:])); pl20=float(np.min(low[-20:]))
        tp1_b=round(price+atr*2,4); tp1_s=round(price-atr*2,4)
        if rsi<35 and price<(ph20+pl20)/2:
            return {"sniper":"SNIPER BUY 🎯","sniper_tp1":tp1_b}
        if rsi>65 and price>(ph20+pl20)/2:
            return {"sniper":"SNIPER SELL 🎯","sniper_tp1":tp1_s}
        return {"sniper":"STANDBY","sniper_tp1":0}
    except: return {"sniper":"N/A","sniper_tp1":0}

def _eng_sentiment(close, volume):
    try:
        c=pd.Series(close); ema=float(ta.trend.EMAIndicator(c,20).ema_indicator().iloc[-1])
        price=float(close[-1]); lv=float(volume[-1]); av=float(np.mean(volume[-20:]))
        if price>ema and lv>av: return {"sentiment":"BULLISH"}
        if price<ema and lv>av: return {"sentiment":"BEARISH"}
        return {"sentiment":"NEUTRAL"}
    except: return {"sentiment":"N/A"}

def _eng_volatility(close, high, low, volume):
    try:
        s=pd.Series(close); ph=pd.Series(high); pl=pd.Series(low)
        atr=float(ta.volatility.AverageTrueRange(ph,pl,s).average_true_range().iloc[-1])
        atr_avg=float(ta.volatility.AverageTrueRange(ph,pl,s).average_true_range().tail(20).mean())
        ratio=atr/atr_avg if atr_avg>0 else 1
        if ratio>1.5:   return {"volatility":"HIGH VOL 🔥","atr_ratio":round(ratio,2)}
        if ratio>1.1:   return {"volatility":"ELEVATED VOL","atr_ratio":round(ratio,2)}
        if ratio<0.7:   return {"volatility":"LOW VOL 😴","atr_ratio":round(ratio,2)}
        return {"volatility":"NORMAL VOL","atr_ratio":round(ratio,2)}
    except: return {"volatility":"N/A","atr_ratio":1}

def _eng_scalping(close, high, low, volume):
    try:
        s=pd.Series(close); ph=pd.Series(high); pl=pd.Series(low)
        rsi=float(ta.momentum.RSIIndicator(s,5).rsi().iloc[-1])
        atr=float(ta.volatility.AverageTrueRange(ph,pl,s).average_true_range().iloc[-1])
        price=float(close[-1])
        tp=round(price+atr*0.8,4) if rsi<40 else round(price-atr*0.8,4)
        if rsi<35: return {"scalp":"SCALP BUY ⚡","scalp_tp":tp}
        if rsi>65: return {"scalp":"SCALP SELL ⚡","scalp_tp":tp}
        return {"scalp":"NO SCALP","scalp_tp":0}
    except: return {"scalp":"N/A","scalp_tp":0}

def _eng_smartmoney(close, high, low, openp, volume):
    try:
        s=pd.Series(close)
        ema20=float(ta.trend.EMAIndicator(s,20).ema_indicator().iloc[-1])
        ema50=float(ta.trend.EMAIndicator(s,50).ema_indicator().iloc[-1])
        lv=float(volume[-1]); av=float(np.mean(volume[-20:]))
        price=float(close[-1])
        if price>ema20>ema50 and lv>av*1.2: return {"smartmoney":"SMART MONEY BULL 🏦"}
        if price<ema20<ema50 and lv>av*1.2: return {"smartmoney":"SMART MONEY BEAR 🏦"}
        if price>ema50: return {"smartmoney":"ACCUMULATION ZONE"}
        return {"smartmoney":"DISTRIBUTION ZONE"}
    except: return {"smartmoney":"N/A"}

def _eng_narrative(close, volume, rsi, ema20, ema50, pair="", cat=""):
    """AI Narrative — institutional market commentary per asset"""
    try:
        price=float(close[-1]); lv=float(volume[-1]); av=float(np.mean(volume[-20:]))
        chg1 = round((close[-1]-close[-4])/close[-4]*100,2) if len(close)>=4 else 0
        above_ema = price > ema20 > ema50
        below_ema = price < ema20 < ema50
        vol_surge = lv > av*1.5
        vol_weak  = lv < av*0.7
        # Directional first
        if rsi > 72:
            n = "OVERBOUGHT EXTREME — PROFIT TAKING EXPECTED"
        elif rsi < 28:
            n = "OVERSOLD EXTREME — POTENTIAL REVERSAL ZONE"
        elif above_ema and vol_surge and rsi > 58:
            n = "INSTITUTIONAL ACCUMULATION — BULLISH CONTINUATION"
        elif above_ema and rsi < 48:
            n = "PULLBACK IN UPTREND — BUY DIP OPPORTUNITY"
        elif above_ema and chg1 > 0.3:
            n = "BULLISH MOMENTUM — UPTREND CONTINUATION"
        elif below_ema and vol_surge and rsi < 42:
            n = "INSTITUTIONAL DISTRIBUTION — BEARISH CONTINUATION"
        elif below_ema and rsi > 52:
            n = "DEAD CAT BOUNCE — SELL RALLY ZONE"
        elif below_ema and chg1 < -0.3:
            n = "BEARISH MOMENTUM — DOWNTREND CONTINUATION"
        elif vol_weak and abs(chg1) < 0.1:
            n = "LOW VOLUME CONSOLIDATION — BREAKOUT IMMINENT"
        elif rsi > 60:
            n = "BULLISH BIAS — MOMENTUM BUILDING"
        elif rsi < 40:
            n = "BEARISH BIAS — SELLERS IN CONTROL"
        else:
            n = "RANGE CONSOLIDATION — AWAIT DIRECTIONAL CATALYST"
        # Append asset context
        if cat == "FOREX":
            n += " | USD FLOW DRIVEN"
        elif cat == "METALS":
            n += " | YIELD & SAFE HAVEN"
        elif cat == "CRYPTO":
            n += " | RISK APPETITE"
        elif cat == "INDEX":
            n += " | EQUITY SENTIMENT"
        elif cat == "CMDTY":
            n += " | SUPPLY/DEMAND"
        return {"narrative": n}
    except: return {"narrative":"N/A"}

def _eng_macro(rsi, ema20, ema50, lv, av):
    try:
        vol_high = lv > av*1.3
        if ema20>ema50 and vol_high and rsi>50: return {"macro":"RISK-ON 🟢"}
        if ema20<ema50 and vol_high and rsi<50: return {"macro":"RISK-OFF 🔴"}
        if ema20>ema50: return {"macro":"RISK-ON 🟢"}
        return {"macro":"RISK-OFF 🔴"}
    except: return {"macro":"N/A"}

def _eng_liquidity(price, high, low, volume):
    try:
        rh=float(np.max(high[-20:])); rl=float(np.min(low[-20:]))
        mid=(rh+rl)/2
        near_high=abs(price-rh)/rh<0.002 if rh else False
        near_low=abs(price-rl)/rl<0.002 if rl else False
        if near_high: liq="UPPER LIQUIDITY ⚠️"
        elif near_low: liq="LOWER LIQUIDITY ⚠️"
        elif price>mid: liq="UPPER RANGE"
        else: liq="LOWER RANGE"
        return {"liquidity":liq,"recent_high":round(rh,4),"recent_low":round(rl,4)}
    except: return {"liquidity":"N/A","recent_high":0,"recent_low":0}

def _eng_risk(price, atr, score, swing_high=0, swing_low=0):
    """Swing-based TP/SL — institutional structure"""
    try:
        if price<=0 or atr<=0: return {"tp1":0,"tp2":0,"tp3":0,"sl":0,"rr":0}
        buf = round(atr*0.3, 4)
        if score>0:  # BUY
            sl  = round((swing_low-buf) if swing_low>0 else (price-atr*1.2), 4)
            tp1 = round(price+atr*1.5, 4)
            tp2 = round(price+atr*3.0, 4)
            tp3 = round(price+atr*5.0, 4)
        else:  # SELL / NEUTRAL
            sl  = round((swing_high+buf) if swing_high>0 else (price+atr*1.2), 4)
            tp1 = round(price-atr*1.5, 4)
            tp2 = round(price-atr*3.0, 4)
            tp3 = round(price-atr*5.0, 4)
        sl_dist = abs(price-sl)
        rr = round(abs(tp1-price)/sl_dist, 2) if sl_dist>0 else 0
        return {"tp1":tp1,"tp2":tp2,"tp3":tp3,"sl":sl,"rr":rr}
    except: return {"tp1":0,"tp2":0,"tp3":0,"sl":0,"rr":0}

def _eng_mtf(ticker):
    """Multi-timeframe analysis"""
    tfs = {"M1":"1m","M5":"5m","M15":"15m","H1":"1h","H4":"4h","D1":"1d"}
    periods = {"M1":"1d","M5":"2d","M15":"5d","H1":"5d","H4":"20d","D1":"60d"}
    result = {}
    for label, interval in tfs.items():
        try:
            df = yf.download(ticker, period=periods[label], interval=interval, progress=False)
            if df is None or len(df)<20:
                result[label]="N/A"; continue
            c = df["Close"].squeeze()
            s = pd.Series(c.values)
            rsi = float(ta.momentum.RSIIndicator(s,14).rsi().iloc[-1])
            e20 = float(ta.trend.EMAIndicator(s,min(20,len(s)-1)).ema_indicator().iloc[-1])
            e50 = float(ta.trend.EMAIndicator(s,min(50,len(s)-1)).ema_indicator().iloc[-1])
            price = float(c.iloc[-1])
            sc = 0
            if rsi<45: sc+=1
            elif rsi>55: sc-=1
            if price>e20: sc+=1
            else: sc-=1
            if price>e50: sc+=1
            else: sc-=1
            result[label] = "BUY" if sc>=2 else "SELL" if sc<=-2 else "NEUTRAL"
        except: result[label]="N/A"
    return result

def _eng_ema200(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df is None or len(df)<50: return {"ema200":0,"ema200_status":"N/A"}
        c = df["Close"].squeeze()
        win = min(200, len(c)-1)
        ema200 = float(ta.trend.EMAIndicator(c, win).ema_indicator().iloc[-1])
        price  = float(c.iloc[-1])
        status = "ABOVE EMA200 📈" if price>ema200 else "BELOW EMA200 📉"
        return {"ema200":round(ema200,4),"ema200_status":status}
    except: return {"ema200":0,"ema200_status":"N/A"}

# ══════════════════════════════════════════════════════════════
#  ASSET INTELLIGENCE — Classification + Flow engine
# ══════════════════════════════════════════════════════════════
def _asset_intelligence(pair, close, rsi, ema20, ema50, atr):
    """Returns asset classification + flow commentary"""
    info = ASSETS.get(pair, {})
    cat  = info.get("cat","")
    typ  = info.get("type","")
    base = info.get("base","")
    quot = info.get("quote","")
    drv  = info.get("drivers","")
    price= float(close[-1])

    # Flow bias
    chg5 = round((price - close[-5])/close[-5]*100, 2) if len(close)>=5 else 0
    bias = "BULLISH" if chg5>0.1 else "BEARISH" if chg5<-0.1 else "NEUTRAL"
    bias_icon = "📈" if bias=="BULLISH" else "📉" if bias=="BEARISH" else "↔️"

    # Category-specific commentary
    if cat=="FOREX":
        flow_txt = f"{base} Flow {chg5:+.2f}% vs {quot}"
    elif cat=="METALS":
        flow_txt = f"Metals flow {chg5:+.2f}% | USD & Yield driven"
    elif cat=="CRYPTO":
        flow_txt = f"Crypto flow {chg5:+.2f}% | Risk appetite"
    elif cat=="INDEX":
        flow_txt = f"Equity flow {chg5:+.2f}% | Growth sentiment"
    else:
        flow_txt = f"Commodity flow {chg5:+.2f}%"

    return {
        "asset_cat":cat,"asset_type":typ,"asset_base":base,"asset_quote":quot,
        "asset_drivers":drv,"flow_bias":bias,"flow_icon":bias_icon,
        "flow_txt":flow_txt,"chg5":chg5,
    }

# ══════════════════════════════════════════════════════════════
#  MASTER COMPUTE — Fully dynamic, zero hardcoded Gold
# ══════════════════════════════════════════════════════════════
def compute(pair):
    cached = _cached_asset(pair)
    if cached: return cached

    info = ASSETS.get(pair)
    if not info:
        return {"pair":pair,"signal":"N/A","price":0,"error":"unknown pair"}

    ticker = info["ticker"]
    tv     = info["tv"]
    cat    = info["cat"]

    try:
        df = yf.download(ticker, period="5d", interval="15m", progress=False)
        if df is None or len(df)<30: raise ValueError("insufficient data")

        close  = df["Close"].squeeze().values.astype(float)
        high   = df["High"].squeeze().values.astype(float)
        low    = df["Low"].squeeze().values.astype(float)
        openp  = df["Open"].squeeze().values.astype(float)
        volume = df["Volume"].squeeze().values.astype(float)

        s = pd.Series(close); ph = pd.Series(high); pl = pd.Series(low)

        # Core indicators
        rsi    = float(ta.momentum.RSIIndicator(s,14).rsi().iloc[-1])
        ema20  = float(ta.trend.EMAIndicator(s,20).ema_indicator().iloc[-1])
        ema50  = float(ta.trend.EMAIndicator(s,50).ema_indicator().iloc[-1])
        macd_o = ta.trend.MACD(s)
        macd   = float(macd_o.macd().iloc[-1])
        macd_sig=float(macd_o.macd_signal().iloc[-1])
        bb     = ta.volatility.BollingerBands(s)
        bb_h   = float(bb.bollinger_hband().iloc[-1])
        bb_l   = float(bb.bollinger_lband().iloc[-1])
        bb_m   = float(bb.bollinger_mavg().iloc[-1])
        stoch  = ta.momentum.StochasticOscillator(ph,pl,s)
        stoch_k= float(stoch.stoch().iloc[-1])
        atr_i  = ta.volatility.AverageTrueRange(ph,pl,s)
        atr    = float(atr_i.average_true_range().iloc[-1])

        price  = float(close[-1])
        prev   = float(close[-2]) if len(close)>=2 else price
        lv     = float(volume[-1])
        av     = float(np.mean(volume[-20:]))
        chg_pct= round((price-prev)/prev*100,4) if prev else 0
        vol_ratio=round(lv/av,2) if av>0 else 1
        spread = round(atr*0.05,4)

        # Master score
        score = 0
        if rsi<35: score+=2
        elif rsi<45: score+=1
        elif rsi>65: score-=2
        elif rsi>55: score-=1
        if price>ema20: score+=1
        if price>ema50: score+=1
        if macd>macd_sig: score+=1
        else: score-=1
        if price<bb_l: score+=1
        if price>bb_h: score-=1
        if stoch_k<20: score+=1
        if stoch_k>80: score-=1

        if score>=4:    signal="STRONG BUY ⚡"
        elif score>=2:  signal="BUY 📈"
        elif score<=-4: signal="STRONG SELL ⚡"
        elif score<=-2: signal="SELL 📉"
        else:           signal="NEUTRAL ↔"

        confidence   = min(abs(score)*15+40, 95)
        trend_status = "UPTREND 📈" if ema20>ema50 else "DOWNTREND 📉"
        momentum     = "STRONG" if abs(macd-macd_sig)>atr*0.1 else "WEAK"

        # Swing structure — defined BEFORE _eng_risk
        rh20 = float(np.max(high[-20:])); rl20 = float(np.min(low[-20:]))
        rh10 = float(np.max(high[-10:])); rl10 = float(np.min(low[-10:]))

        # Sub-engines
        hf   = _eng_hf(close,high,low,openp,volume)
        ml   = _eng_ml(ticker)
        wh   = _eng_whale(volume)
        ob   = _eng_orderblock(close,openp,high,low,volume)
        sn   = _eng_sniper(close,high,low,volume)
        sent = _eng_sentiment(close,volume)
        vl   = _eng_volatility(close,high,low,volume)
        sc_e = _eng_scalping(close,high,low,volume)
        sm   = _eng_smartmoney(close,high,low,openp,volume)
        nar  = _eng_narrative(close,volume,rsi,ema20,ema50,pair=pair,cat=cat)
        mac  = _eng_macro(rsi,ema20,ema50,lv,av)
        liq  = _eng_liquidity(price,high,low,volume)
        risk = _eng_risk(price,atr,score,swing_high=rh20,swing_low=rl20)
        mtf  = _eng_mtf(ticker)
        e200 = _eng_ema200(ticker)
        intel= _asset_intelligence(pair,close,rsi,ema20,ema50,atr)

        # SMC
        bos   = "BOS BULLISH 🔼" if price>rh20 else "BOS BEARISH 🔽" if price<rl20 else "NO BOS"
        choch = "CHoCH DETECTED ⚡" if (price>rh10 and ema20<ema50) or (price<rl10 and ema20>ema50) else "NO CHoCH"
        fvg   = "FVG BULLISH" if price<bb_m and rsi<45 else "FVG BEARISH" if price>bb_m and rsi>55 else "NO FVG"

        result = {
            "pair":pair,"ticker":ticker,"tv":tv,"cat":cat,
            "signal":signal,"price":round(price,4),"chg_pct":chg_pct,
            "confidence":confidence,"trend_status":trend_status,"momentum":momentum,
            "rsi":round(rsi,1),"ema20":round(ema20,4),"ema50":round(ema50,4),
            "macd":round(macd,4),"bb_h":round(bb_h,4),"bb_l":round(bb_l,4),
            "stoch_k":round(stoch_k,1),"atr":round(atr,4),"spread":spread,
            "vol_ratio":vol_ratio,"recent_high":rh20,"recent_low":rl20,
            "bos":bos,"choch":choch,"fvg":fvg,
            **hf,**ml,**wh,**ob,**sn,**sent,**vl,**sc_e,**sm,**nar,**mac,**liq,**risk,**e200,**intel,
            "multi_tf":mtf,
        }
        _set_asset_cache(pair, result)
        return result

    except Exception as e:
        return {"pair":pair,"signal":"N/A","price":0,"error":str(e),
                "ticker":ticker,"tv":tv,"cat":cat,
                "tp1":0,"tp2":0,"tp3":0,"sl":0,"rr":0,"multi_tf":{},
                "recent_high":0,"recent_low":0,"bos":"N/A","choch":"N/A","fvg":"N/A",
                "asset_cat":cat,"asset_type":"","asset_drivers":"","flow_bias":"NEUTRAL",
                "flow_icon":"↔️","flow_txt":"","chg5":0}

# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    d    = compute(DEFAULT_PAIR)
    cot  = get_cot()
    gold = get_gold_live()
    cme  = get_cme()
    news = get_news()
    corr = get_corr()
    return render_template("index.html",
        d=d, cot=cot, gold=gold, cme=cme, news=news, corr=corr,
        assets=ASSETS, categories=CATEGORIES,
        asset_tv={k:v["tv"] for k,v in ASSETS.items()},
        cat_default=CAT_DEFAULT,
        default_pair=DEFAULT_PAIR,
        tv_map=json.dumps(TV_MAP),
    )

@app.route("/api/asset/<path:key>")
def api_asset(key):
    key = key.replace("_","/").upper()
    if key not in ASSETS: return jsonify({"error":"not found"}),404
    return jsonify(compute(key))

@app.route("/api/cot")
def api_cot(): return jsonify(get_cot())

@app.route("/api/gold")
def api_gold(): return jsonify(get_gold_live())

@app.route("/api/cme")
def api_cme(): return jsonify(get_cme())

@app.route("/api/news")
def api_news(): return jsonify(get_news())

@app.route("/api/corr")
def api_corr(): return jsonify(get_corr())

@app.route("/api/all")
def api_all():
    return jsonify({p: _cached_asset(p) or {} for p in ASSETS})

@app.route("/api/currency_strength")
def api_currency_strength():
    return jsonify(get_currency_strength())

@app.route("/api/risk")
def api_risk():
    try:
        balance  = float(request.args.get("balance",10000))
        risk_pct = float(request.args.get("risk",1))
        entry    = float(request.args.get("entry",0))
        sl_price = float(request.args.get("sl",0))
        leverage = float(request.args.get("leverage",1))
        if entry<=0 or sl_price<=0: return jsonify({"error":"invalid params"})
        sl_dist  = abs(entry - sl_price)
        max_loss = round(balance * risk_pct/100, 2)
        lot_size = round(max_loss / sl_dist, 4) if sl_dist>0 else 0
        margin   = round((lot_size * entry)/leverage, 2) if leverage>0 else 0
        pip_val  = round(sl_dist/entry*100, 4) if entry>0 else 0
        return jsonify({"max_loss":max_loss,"lot_size":lot_size,"margin":margin,"pip":pip_val,"sl_dist":round(sl_dist,4)})
    except Exception as e: return jsonify({"error":str(e)})

@app.route("/api/scan")
def api_scan():
    """Quick market scan — one-glance overview all pairs"""
    out = {}
    for pair in ASSETS:
        c = _cached_asset(pair)
        if c:
            out[pair] = {
                "signal":c.get("signal","N/A"),
                "price":c.get("price",0),
                "chg_pct":c.get("chg_pct",0),
                "confidence":c.get("confidence",0),
                "cat":c.get("cat",""),
                "trend":c.get("trend_status","N/A"),
                "flow_bias":c.get("flow_bias","N/A"),
            }
    return jsonify(out)

@app.route("/health")
def health(): return jsonify({"status":"ok","pairs":len(ASSETS),"cached":len(_sig_cache)})

# ══════════════════════════════════════════════════════════════
#  STARTUP — background preload all pairs
# ══════════════════════════════════════════════════════════════
def _preload_all():
    def worker():
        time.sleep(2)
        print("[ELITE TERMINAL] Preloading all pairs...")
        for pair in ASSETS:
            try:
                data = compute(pair)
                print(f"  ✅ {pair} — {data.get('signal','?')} @ {data.get('price',0)}")
            except Exception as e:
                print(f"  ❌ {pair} — {e}")
        print("[ELITE TERMINAL] All pairs cached.")
    threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    print("╔══════════════════════════════════════╗")
    print("║   TERMINAL ELITE v3.0 — STARTING     ║")
    print("║   http://localhost:8080               ║")
    print("╚══════════════════════════════════════╝")
    _preload_all()
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
