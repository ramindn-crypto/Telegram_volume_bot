#!/usr/bin/env python3
"""
Telegram crypto screener & alert bot (CoinEx via CCXT) — FUTURES ONLY

Features:
- Shows 3 priorities of coins (P1, P2, P3) using CoinEx volumes (Spot volume shown, but signals are FUTURES-only)
- Table: SYM | F | S | %24H | %4H | %1H
- Email alerts when there are strong BUY/SELL signals (trigger 1h, confirm 15m, trend confirm 4h)
- Additional alert: coins with F volume > 1M and 24h change > +10% (24h movers)
- Email alerts only 12:30–01:00 (Australia/Melbourne), no Sundays
- Max 1 email per 15 minutes
- NO EMAIL if:
    * the set of ALL symbols (recs + movers) is the same as in the last email
- Performance tracking (SQLite):
    * Stores signals (entry/sl/tp/confidence/priority)
    * Evaluates TOUCH-based TP/SL using 15m candles up to 24h
    * /report, /report30, /last10, /stats BTC
    * Weekly report auto (Mon 09:10–09:15 Melbourne): performance + top/worst coins + high-conf wins/losses
- Commands: /start /screen /notify_on /notify_off /notify /diag /report /report30 /last10 /stats
- Typing a symbol (e.g. PYTH) gives its row
"""

import asyncio
import logging
import os
import time
import re
import ssl
import smtplib
import math
import sqlite3
from pathlib import Path
from statistics import mean, pstdev
from email.message import EmailMessage
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
from zoneinfo import ZoneInfo

import ccxt
from tabulate import tabulate
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import Conflict
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ================== CONFIG ==================

EXCHANGE_ID = "coinex"
TOKEN = os.environ.get("TELEGRAM_TOKEN")

# Priority thresholds (USD notional)
P1_SPOT_MIN = 500_000
P1_FUT_MIN = 5_000_000
P2_FUT_MIN = 2_000_000
P3_SPOT_MIN = 3_000_000

TOP_N_P1 = 10
TOP_N_P2 = 15
TOP_N_P3 = 10

STABLES = {"USD", "USDT", "USDC", "TUSD", "FDUSD", "USDD", "USDE", "DAI", "PYUSD"}

# Pinned coins (only appear in P3)
PINNED_P3 = ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "PEPE", "LINK"]
PINNED_SET = set(PINNED_P3)

# Email config (set in env)
EMAIL_ENABLED_DEFAULT = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST = os.environ.get("EMAIL_HOST", "")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "465"))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", EMAIL_USER)
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_USER)

# Scheduler / alerts
CHECK_INTERVAL_MIN = 5  # run every 5 minutes
EMAIL_MIN_INTERVAL_SEC = 15 * 60  # one email per 15 minutes

# DB
DB_PATH = os.environ.get("BOT_DB_PATH", "signals.db")
SIGNAL_EVAL_INTERVAL_MIN = 15   # evaluate open signals every 15 min
SIGNAL_MAX_AGE_HOURS = 24       # evaluate up to 24h
EVAL_TF = "15m"                 # touch-check timeframe

# Indicators (Trigger/Confirm)
IND_TREND_TF = "1h"      # trigger timeframe
IND_CONFIRM_TF = "15m"   # confirmation timeframe
IND_HTF = "4h"           # higher timeframe trend confirm

EMA_FAST = 20
EMA_SLOW = 50
RSI_LEN = 14
ATR_LEN = 14
ADX_LEN = 14
BREAKOUT_N = 20
VOL_Z_N = 20

ADX_MIN = 18.0
VOL_Z_MIN_TRIGGER = 1.3
VOL_Z_MIN_CONFIRM = 1.5
ATR_PCT_MAX = 6.0
RSI_LONG_MIN, RSI_LONG_MAX = 50.0, 72.0
RSI_SHORT_MIN, RSI_SHORT_MAX = 28.0, 50.0

SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.6

MAX_WICK_BODY_RATIO = 2.5  # wick/body > this => reject breakout

# Cooldown / anti-overtrade
SYMBOL_COOLDOWN_SEC = 90 * 60  # 90 minutes

# Caches to reduce API calls
OHLCV_CACHE: Dict[Tuple[str, str], Tuple[float, List[List[float]]]] = {}  # key=(symbol,timeframe)
OHLCV_TTL_SEC = 60  # 1 minute

PCT4H_CACHE: Dict[str, float] = {}
PCT1H_CACHE: Dict[str, float] = {}

# Globals
LAST_EMAIL_TS: float = 0.0
LAST_ALL_SYMBOLS: Set[str] = set()
LAST_ERROR: Optional[str] = None
NOTIFY_ON: bool = EMAIL_ENABLED_DEFAULT

LAST_SIGNAL_TS: Dict[str, float] = {}  # cooldown by base symbol

LAST_CHAT_ID: Optional[int] = None
LAST_WEEKLY_SENT_YMD: Optional[str] = None


# ================== DATA STRUCTURES ==================

@dataclass
class MarketVol:
    symbol: str
    base: str
    quote: str
    last: float
    open: float
    percentage: float
    base_vol: float
    quote_vol: float
    vwap: float


# ================== BASIC HELPERS ==================

def safe_split_symbol(sym: Optional[str]):
    if not sym:
        return None
    pair = sym.split(":")[0]
    if "/" not in pair:
        return None
    return tuple(pair.split("/", 1))


def to_mv(t: dict) -> Optional[MarketVol]:
    sym = t.get("symbol")
    sp = safe_split_symbol(sym)
    if not sp:
        return None
    base, quote = sp
    return MarketVol(
        symbol=sym,
        base=base,
        quote=quote,
        last=float(t.get("last") or 0.0),
        open=float(t.get("open") or 0.0),
        percentage=float(t.get("percentage") or 0.0),
        base_vol=float(t.get("baseVolume") or 0.0),
        quote_vol=float(t.get("quoteVolume") or 0.0),
        vwap=float(t.get("vwap") or 0.0),
    )


def usd_notional(mv: Optional[MarketVol]) -> float:
    """Return 24h notional volume in USD."""
    if not mv:
        return 0.0
    if mv.quote in STABLES and mv.quote_vol:
        return mv.quote_vol
    price = mv.vwap if mv.vwap else mv.last
    if not price or not mv.base_vol:
        return 0.0
    return mv.base_vol * price


def pct_change_24h(mv_spot: Optional[MarketVol], mv_fut: Optional[MarketVol]) -> float:
    """24h % change (prefer ticker.percentage)."""
    for mv in (mv_fut, mv_spot):
        if mv and mv.percentage:
            return float(mv.percentage)
    mv = mv_fut or mv_spot
    if mv and mv.open:
        return (mv.last - mv.open) / mv.open * 100.0
    return 0.0


def pct_with_emoji(p: float) -> str:
    val = round(p)
    if val >= 3:
        emo = "🟢"
    elif val <= -3:
        emo = "🔴"
    else:
        emo = "🟡"
    return f"{val:+d}% {emo}"


def m_dollars(x: float) -> str:
    return str(round(x / 1_000_000))


def build_exchange(default_type: str):
    klass = ccxt.__dict__[EXCHANGE_ID]
    return klass(
        {
            "enableRateLimit": True,
            "timeout": 20000,
            "options": {"defaultType": default_type},
        }
    )


def safe_fetch_tickers(ex):
    """Fetch tickers; on error, log and return {}."""
    try:
        ex.load_markets()
        return ex.fetch_tickers()
    except Exception as e:
        global LAST_ERROR
        LAST_ERROR = f"{type(e).__name__}: {e}"
        logging.exception("fetch_tickers failed")
        return {}


# ================== 1h % change helpers (FUTURES ONLY) ==================

def compute_pct_for_symbol(symbol: str, hours: int) -> float:
    """
    Compute % change over the last N completed hours using 1h candles (FUTURES ONLY).
    Cache results per symbol+hours.
    """
    cache = PCT4H_CACHE if hours == 4 else PCT1H_CACHE
    if symbol in cache:
        return cache[symbol]
    try:
        ex = build_exchange("swap")
        ex.load_markets()
        candles = ex.fetch_ohlcv(symbol, timeframe="1h", limit=hours + 1)
        if not candles or len(candles) <= hours:
            cache[symbol] = 0.0
            return 0.0
        closes = [c[4] for c in candles][- (hours + 1):]
        if not closes or not closes[0]:
            pct = 0.0
        else:
            pct = (closes[-1] - closes[0]) / closes[0] * 100.0
        cache[symbol] = pct
        return pct
    except Exception:
        logging.exception("compute_pct_for_symbol failed: %dh %s", hours, symbol)
        cache[symbol] = 0.0
        return 0.0


# ================== INDICATORS (FUTURES ONLY) ==================

def _ema(values: List[float], length: int) -> List[float]:
    if not values or length <= 1:
        return values[:]
    k = 2 / (length + 1)
    out = []
    ema = values[0]
    out.append(ema)
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
        out.append(ema)
    return out


def _rsi(closes: List[float], length: int = 14) -> float:
    if len(closes) < length + 2:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(-length, 0):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains += change
        else:
            losses += abs(change)
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def _true_ranges(highs: List[float], lows: List[float], closes: List[float]) -> List[float]:
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    return trs


def _atr(highs: List[float], lows: List[float], closes: List[float], length: int = 14) -> float:
    if len(closes) < length + 2:
        return 0.0
    trs = _true_ranges(highs, lows, closes)
    if len(trs) < length:
        return 0.0
    return sum(trs[-length:]) / length


def _adx(highs: List[float], lows: List[float], closes: List[float], length: int = 14) -> float:
    """
    Lightweight ADX-ish filter (DX proxy) for gating trendiness.
    """
    if len(closes) < length + 2:
        return 10.0

    plus_dm = []
    minus_dm = []

    for i in range(1, len(closes)):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        pdm = up_move if (up_move > down_move and up_move > 0) else 0.0
        mdm = down_move if (down_move > up_move and down_move > 0) else 0.0

        plus_dm.append(pdm)
        minus_dm.append(mdm)

    trs = _true_ranges(highs, lows, closes)
    if len(trs) < length:
        return 10.0

    tr_n = sum(trs[-length:])
    if tr_n == 0:
        return 10.0

    pdi = 100.0 * (sum(plus_dm[-length:]) / tr_n)
    mdi = 100.0 * (sum(minus_dm[-length:]) / tr_n)
    denom = (pdi + mdi)
    if denom == 0:
        return 10.0

    dx = 100.0 * abs(pdi - mdi) / denom
    return dx


def _vol_z(volumes: List[float], n: int = 20) -> float:
    if len(volumes) < n + 2:
        return 0.0
    window = volumes[-n-1:-1]
    mu = mean(window)
    sd = pstdev(window) if len(window) > 1 else 0.0
    if sd == 0:
        return 0.0
    return (volumes[-1] - mu) / sd


def wick_body_ratio(candle):
    """
    candle: [ts, open, high, low, close, vol]
    returns max(wick/body) ratio
    """
    o, h, l, c = candle[1], candle[2], candle[3], candle[4]
    body = abs(c - o)
    if body == 0:
        return 999.0
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return max(upper_wick, lower_wick) / body


def fetch_ohlcv_cached_futures(symbol: str, timeframe: str, limit: int) -> List[List[float]]:
    """
    Cached OHLCV fetch — FUTURES ONLY.
    key=(symbol,timeframe)
    """
    key = (symbol, timeframe)
    now = time.time()
    if key in OHLCV_CACHE:
        ts, data = OHLCV_CACHE[key]
        if now - ts < OHLCV_TTL_SEC and data:
            return data

    try:
        ex = build_exchange("swap")
        ex.load_markets()
        data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if data:
            OHLCV_CACHE[key] = (now, data)
            return data
    except Exception:
        logging.exception("fetch_ohlcv_cached_futures failed: %s %s", symbol, timeframe)

    return []


def compute_indicators_futures(market_symbol: str) -> Dict[str, float]:
    """
    Compute:
      - Trigger indicators on 1h (ema20/50, rsi, atr, adx, atr%, vol_z_1h)
      - Confirm indicators on 15m (breakout up/down, vol_z_15m, wick_ok)
      - HTF trend confirmation on 4h (trend4h_up/dn)
    """
    out = {
        "ema20": 0.0, "ema50": 0.0, "rsi": 50.0,
        "atr": 0.0, "atr_pct": 0.0, "adx": 10.0,
        "vol_z_1h": 0.0,
        "vol_z_15m": 0.0, "brk_up_15m": 0.0, "brk_dn_15m": 0.0, "wick_ok_15m": 0.0,
        "trend4h_up": 0.0, "trend4h_dn": 0.0,
        "ema20_4h": 0.0, "ema50_4h": 0.0,
    }

    # --- 1h trigger data ---
    ohlcv_1h = fetch_ohlcv_cached_futures(
        market_symbol,
        IND_TREND_TF,
        limit=max(EMA_SLOW, VOL_Z_N, ATR_LEN, ADX_LEN) + 10
    )
    if ohlcv_1h and len(ohlcv_1h) > EMA_SLOW + 2:
        highs = [c[2] for c in ohlcv_1h]
        lows = [c[3] for c in ohlcv_1h]
        closes = [c[4] for c in ohlcv_1h]
        vols = [c[5] for c in ohlcv_1h]

        ema20 = _ema(closes, EMA_FAST)[-1]
        ema50 = _ema(closes, EMA_SLOW)[-1]
        rsi = _rsi(closes, RSI_LEN)
        atr = _atr(highs, lows, closes, ATR_LEN)
        adx = _adx(highs, lows, closes, ADX_LEN)
        atr_pct = (atr / closes[-1] * 100.0) if closes[-1] else 0.0
        vz1 = _vol_z(vols, VOL_Z_N)

        out.update({"ema20": ema20, "ema50": ema50, "rsi": rsi, "atr": atr, "adx": adx, "atr_pct": atr_pct, "vol_z_1h": vz1})

    # --- 15m confirm data ---
    ohlcv_15m = fetch_ohlcv_cached_futures(
        market_symbol,
        IND_CONFIRM_TF,
        limit=BREAKOUT_N + VOL_Z_N + 20
    )
    if ohlcv_15m and len(ohlcv_15m) > BREAKOUT_N + 2:
        highs15 = [c[2] for c in ohlcv_15m]
        lows15 = [c[3] for c in ohlcv_15m]
        closes15 = [c[4] for c in ohlcv_15m]
        vols15 = [c[5] for c in ohlcv_15m]

        prev_high = max(highs15[-(BREAKOUT_N + 1):-1])
        prev_low = min(lows15[-(BREAKOUT_N + 1):-1])
        last_close = closes15[-1]

        brk_up = 1.0 if last_close > prev_high else 0.0
        brk_dn = 1.0 if last_close < prev_low else 0.0
        vz15 = _vol_z(vols15, VOL_Z_N)

        last_candle = ohlcv_15m[-1]
        wick_ratio = wick_body_ratio(last_candle)
        wick_ok = 1.0 if wick_ratio <= MAX_WICK_BODY_RATIO else 0.0

        out.update({"brk_up_15m": brk_up, "brk_dn_15m": brk_dn, "vol_z_15m": vz15, "wick_ok_15m": wick_ok})

    # --- 4h HTF trend confirm ---
    ohlcv_4h = fetch_ohlcv_cached_futures(
        market_symbol,
        IND_HTF,
        limit=max(EMA_SLOW, 60)
    )
    if ohlcv_4h and len(ohlcv_4h) > EMA_SLOW + 2:
        closes4 = [c[4] for c in ohlcv_4h]
        ema20_4h = _ema(closes4, EMA_FAST)[-1]
        ema50_4h = _ema(closes4, EMA_SLOW)[-1]
        last4 = closes4[-1]
        trend4h_up = 1.0 if (ema20_4h > ema50_4h and last4 > ema50_4h) else 0.0
        trend4h_dn = 1.0 if (ema20_4h < ema50_4h and last4 < ema50_4h) else 0.0
        out.update({"ema20_4h": ema20_4h, "ema50_4h": ema50_4h, "trend4h_up": trend4h_up, "trend4h_dn": trend4h_dn})

    return out


# ================== PRIORITIES ==================

def load_best():
    """
    Load top spot & futures markets per BASE symbol from CoinEx.
    NOTE: Signals are FUTURES ONLY, but we still show spot volume as context.
    """
    ex_spot = build_exchange("spot")
    ex_fut = build_exchange("swap")
    spot_tickers = safe_fetch_tickers(ex_spot)
    fut_tickers = safe_fetch_tickers(ex_fut)

    best_spot: Dict[str, MarketVol] = {}
    best_fut: Dict[str, MarketVol] = {}

    for t in spot_tickers.values():
        mv = to_mv(t)
        if mv and mv.quote in STABLES:
            if mv.base not in best_spot or usd_notional(mv) > usd_notional(best_spot[mv.base]):
                best_spot[mv.base] = mv

    for t in fut_tickers.values():
        mv = to_mv(t)
        if mv:
            if mv.base not in best_fut or usd_notional(mv) > usd_notional(best_fut[mv.base]):
                best_fut[mv.base] = mv

    return best_spot, best_fut, len(spot_tickers), len(fut_tickers)


def build_priorities(best_spot: Dict[str, MarketVol], best_fut: Dict[str, MarketVol]):
    """
    Build P1, P2, P3 lists.
    Each row: [BASE, fut_usd, spot_usd, pct24, pct4, pct1, last_price, fut_market_symbol]
    FUTURES ONLY rows (must have futures market).
    """
    p1: List[List] = []
    p2: List[List] = []
    p3: List[List] = []
    used = set()

    # P1: F≥5M & S≥0.5M (pinned excluded) — require futures
    for base in set(best_spot) & set(best_fut):
        if base in PINNED_SET:
            continue
        s = best_spot[base]
        f = best_fut[base]
        fut_usd = usd_notional(f)
        spot_usd = usd_notional(s)
        if fut_usd >= P1_FUT_MIN and spot_usd >= P1_SPOT_MIN:
            pct24 = pct_change_24h(s, f)
            pct4 = compute_pct_for_symbol(f.symbol, 4)
            pct1 = compute_pct_for_symbol(f.symbol, 1)
            last_price = f.last or s.last or 0.0
            p1.append([base, fut_usd, spot_usd, pct24, pct4, pct1, last_price, f.symbol])

    p1.sort(key=lambda x: x[1], reverse=True)
    p1 = p1[:TOP_N_P1]
    used |= {r[0] for r in p1}

    # P2: F≥2M (pinned & already used excluded) — require futures
    for base, f in best_fut.items():
        if base in used or base in PINNED_SET:
            continue
        fut_usd = usd_notional(f)
        if fut_usd >= P2_FUT_MIN:
            s = best_spot.get(base)
            spot_usd = usd_notional(s) if s else 0.0
            pct24 = pct_change_24h(s, f)
            pct4 = compute_pct_for_symbol(f.symbol, 4)
            pct1 = compute_pct_for_symbol(f.symbol, 1)
            last_price = f.last or (s.last if s else 0.0)
            p2.append([base, fut_usd, spot_usd, pct24, pct4, pct1, last_price, f.symbol])

    p2.sort(key=lambda x: x[1], reverse=True)
    p2 = p2[:TOP_N_P2]
    used |= {r[0] for r in p2}

    # P3: pinned + others with Spot≥3M, but require futures market for each row
    tmp: Dict[str, List] = {}

    # pinned first (only if futures exists)
    for base in PINNED_P3:
        f = best_fut.get(base)
        s = best_spot.get(base)
        if not f:
            continue
        fut_usd = usd_notional(f)
        spot_usd = usd_notional(s) if s else 0.0
        pct24 = pct_change_24h(s, f)
        pct4 = compute_pct_for_symbol(f.symbol, 4)
        pct1 = compute_pct_for_symbol(f.symbol, 1)
        last_price = f.last or (s.last if s else 0.0)
        tmp[base] = [base, fut_usd, spot_usd, pct24, pct4, pct1, last_price, f.symbol]

    # non-pinned, not used, Spot≥3M — require futures
    for base, s in best_spot.items():
        if base in used or base in PINNED_SET:
            continue
        f = best_fut.get(base)
        if not f:
            continue
        spot_usd = usd_notional(s)
        if spot_usd >= P3_SPOT_MIN:
            fut_usd = usd_notional(f)
            pct24 = pct_change_24h(s, f)
            pct4 = compute_pct_for_symbol(f.symbol, 4)
            pct1 = compute_pct_for_symbol(f.symbol, 1)
            last_price = f.last or s.last
            tmp[base] = [base, fut_usd, spot_usd, pct24, pct4, pct1, last_price, f.symbol]

    all_rows = list(tmp.values())
    pinned_rows = [r for r in all_rows if r[0] in PINNED_SET]
    other_rows = [r for r in all_rows if r[0] not in PINNED_SET]

    pinned_rows.sort(key=lambda r: r[2], reverse=True)
    other_rows.sort(key=lambda r: r[2], reverse=True)
    p3 = (pinned_rows + other_rows)[:TOP_N_P3]

    return p1, p2, p3


# ================== SCORING / CLASSIFICATION ==================

def score_long(row7: List) -> float:
    """
    row7: [sym, fut_usd, spot_usd, pct24, pct4, pct1, last_price]
    """
    _, fut_usd, _, pct24, pct4, pct1, _ = row7
    score = 0.0
    if pct24 > 0:
        score += min(pct24 / 5.0, 3.0)
    if pct4 > 0:
        score += min(pct4 / 2.0, 3.0)
    if pct1 > 0:
        score += min(pct1 / 1.0, 2.0)

    if fut_usd > 10_000_000:
        score += 2.0
    elif fut_usd > 5_000_000:
        score += 1.0

    return max(score, 0.0)


def score_short(row7: List) -> float:
    """
    row7: [sym, fut_usd, spot_usd, pct24, pct4, pct1, last_price]
    """
    _, fut_usd, _, pct24, pct4, pct1, _ = row7
    score = 0.0
    if pct24 < 0:
        score += min(abs(pct24) / 5.0, 3.0)
    if pct4 < 0:
        score += min(abs(pct4) / 2.0, 3.0)
    if pct1 < 0:
        score += min(abs(pct1) / 1.0, 2.0)

    if fut_usd > 10_000_000:
        score += 2.0
    elif fut_usd > 5_000_000:
        score += 1.0

    return max(score, 0.0)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def confidence_score(ind: Dict[str, float], last_price: float) -> int:
    """
    Score 0..100
    """
    score = 0.0

    trend4h_up = ind.get("trend4h_up", 0.0)
    trend4h_dn = ind.get("trend4h_dn", 0.0)
    ema20 = ind.get("ema20", 0.0)
    ema50 = ind.get("ema50", 0.0)

    trend1h_up = 1.0 if (ema20 > ema50 and last_price > ema50) else 0.0
    trend1h_dn = 1.0 if (ema20 < ema50 and last_price < ema50) else 0.0

    trend_align = 1.0 if ((trend1h_up and trend4h_up) or (trend1h_dn and trend4h_dn)) else 0.0
    score += 25.0 * trend_align

    adx = ind.get("adx", 10.0)
    score += 15.0 * clamp((adx - ADX_MIN) / 20.0, 0.0, 1.0)

    vz1 = ind.get("vol_z_1h", 0.0)
    score += 15.0 * clamp(vz1 / 3.0, 0.0, 1.0)

    vz15 = ind.get("vol_z_15m", 0.0)
    score += 15.0 * clamp(vz15 / 3.0, 0.0, 1.0)

    brk = max(ind.get("brk_up_15m", 0.0), ind.get("brk_dn_15m", 0.0))
    wick_ok = ind.get("wick_ok_15m", 0.0)
    score += 10.0 * clamp(brk * wick_ok, 0.0, 1.0)

    rsi = ind.get("rsi", 50.0)
    dist = min(abs(rsi - 60.0), abs(rsi - 40.0))
    score += 10.0 * clamp(1.0 - dist / 20.0, 0.0, 1.0)

    atr_pct = ind.get("atr_pct", 0.0)
    score += 10.0 * clamp(1.0 - (atr_pct / ATR_PCT_MAX), 0.0, 1.0)

    return int(round(clamp(score, 0.0, 100.0)))


def classify_row(row: List) -> Tuple[bool, bool, Dict[str, float]]:
    """
    row: [base, fut_usd, spot_usd, pct24, pct4, pct1, last_price, market_symbol]
    FUTURES ONLY.
    """
    _, _, _, pct24, pct4, pct1, last_price, mkt_symbol = row
    if not mkt_symbol or not last_price:
        return (False, False, {})

    ind = compute_indicators_futures(mkt_symbol)

    ema20 = ind["ema20"]
    ema50 = ind["ema50"]
    rsi = ind["rsi"]
    adx = ind["adx"]
    atr_pct = ind["atr_pct"]
    vz1 = ind["vol_z_1h"]
    vz15 = ind["vol_z_15m"]
    brk_up = ind["brk_up_15m"]
    brk_dn = ind["brk_dn_15m"]
    wick_ok = ind.get("wick_ok_15m", 0.0)

    trend4h_up = ind.get("trend4h_up", 0.0)
    trend4h_dn = ind.get("trend4h_dn", 0.0)

    # No-trade zones
    if adx < ADX_MIN:
        return (False, False, ind)
    if atr_pct > ATR_PCT_MAX:
        return (False, False, ind)

    # 1H trend
    trend1h_up = (ema20 > ema50) and (last_price > ema50)
    trend1h_dn = (ema20 < ema50) and (last_price < ema50)

    # Multi-timeframe trend (1h + 4h)
    trend_up = trend1h_up and (trend4h_up > 0.5)
    trend_dn = trend1h_dn and (trend4h_dn > 0.5)

    # Momentum trigger (1h)
    long_mom_ok = pct1 >= 1.5 and pct4 >= 0.0
    short_mom_ok = pct1 <= -1.5 and pct4 <= 0.0

    # Volume confirm: 1h abnormal + 15m abnormal
    vol_ok = (vz1 >= VOL_Z_MIN_TRIGGER) and (vz15 >= VOL_Z_MIN_CONFIRM)

    # RSI zone
    rsi_long_ok = RSI_LONG_MIN <= rsi <= RSI_LONG_MAX
    rsi_short_ok = RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX

    # Confirm: breakout + wick filter on 15m
    confirm_long = (brk_up > 0.5) and (wick_ok > 0.5)
    confirm_short = (brk_dn > 0.5) and (wick_ok > 0.5)

    # 24h soft bias
    bias_long = pct24 > -4.0
    bias_short = pct24 < +4.0

    long_ok = trend_up and long_mom_ok and vol_ok and rsi_long_ok and confirm_long and bias_long
    short_ok = trend_dn and short_mom_ok and vol_ok and rsi_short_ok and confirm_short and bias_short

    return (long_ok, short_ok, ind)


# ================== DB (SQLite) ==================

def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_init():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with db_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            market_symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry REAL NOT NULL,
            sl REAL NOT NULL,
            tp REAL NOT NULL,
            confidence INTEGER NOT NULL,
            priority TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'OPEN',
            closed_ts INTEGER,
            exit_price REAL,
            mfe REAL,
            mae REAL
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_status_ts ON signals(status, ts);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_sym_side_ts ON signals(symbol, side, ts);")
        conn.commit()


def db_recent_open_exists(symbol: str, side: str, within_minutes: int = 60) -> bool:
    since_ts = int(time.time()) - within_minutes * 60
    with db_conn() as conn:
        r = conn.execute("""
            SELECT 1 FROM signals
            WHERE symbol=? AND side=? AND status='OPEN' AND ts >= ?
            LIMIT 1
        """, (symbol.upper(), side.upper(), since_ts)).fetchone()
    return r is not None


def db_insert_signal(ts: int, symbol: str, market_symbol: str, side: str, entry: float, sl: float, tp: float,
                     confidence: int, priority: str):
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO signals (ts, symbol, market_symbol, side, entry, sl, tp, confidence, priority, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
        """, (ts, symbol.upper(), market_symbol, side.upper(), entry, sl, tp, int(confidence), priority))
        conn.commit()


def db_close_signal(sig_id: int, status: str, closed_ts: int, exit_price: float, mfe: float, mae: float):
    with db_conn() as conn:
        conn.execute("""
            UPDATE signals
            SET status=?, closed_ts=?, exit_price=?, mfe=?, mae=?
            WHERE id=?
        """, (status, closed_ts, exit_price, mfe, mae, sig_id))
        conn.commit()


def db_fetch_open_signals(limit: int = 200):
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM signals
            WHERE status='OPEN'
            ORDER BY ts ASC
            LIMIT ?
        """, (limit,)).fetchall()
    return rows


def db_fetch_last(limit: int = 10):
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM signals
            ORDER BY ts DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return rows


def db_fetch_since(days: int):
    since_ts = int(time.time()) - days * 24 * 3600
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM signals
            WHERE ts >= ?
        """, (since_ts,)).fetchall()
    return rows


def db_fetch_symbol_stats(symbol: str, days: int = 30):
    since_ts = int(time.time()) - days * 24 * 3600
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM signals
            WHERE symbol = ? AND ts >= ?
        """, (symbol.upper(), since_ts)).fetchall()
    return rows


# ================== RECOMMENDATIONS ==================

def conf_label(conf: int) -> str:
    if conf >= 80:
        return "HIGH"
    if conf >= 65:
        return "MED"
    return "LOW"


def pick_best_trades(p1: List[List], p2: List[List], p3: List[List]):
    """
    Return TOP 3 trades overall (BUY or SELL).
    Each candidate:
      (side, sym, entry, tp, sl, score, priority_label, conf)
    Also persists new OPEN signals to DB (with anti-dup + cooldown).
    """
    rows_with_label: List[Tuple[List, str]] = []
    rows_with_label += [(r, "P1") for r in p1]
    rows_with_label += [(r, "P2") for r in p2]
    rows_with_label += [(r, "P3") for r in p3]

    candidates: List[Tuple[str, str, float, float, float, float, str, int]] = []

    now = time.time()
    now_ts = int(now)

    for r, label in rows_with_label:
        sym, fut_usd, _, pct24, pct4, pct1, last_price, mkt_symbol = r
        if not last_price or last_price <= 0 or not mkt_symbol:
            continue

        # cooldown per symbol
        last_sig = LAST_SIGNAL_TS.get(sym, 0.0)
        if now - last_sig < SYMBOL_COOLDOWN_SEC:
            continue

        long_ok, short_ok, ind = classify_row(r)
        atr = float(ind.get("atr", 0.0) or 0.0)
        adx = float(ind.get("adx", 10.0) or 10.0)
        vz1 = float(ind.get("vol_z_1h", 0.0) or 0.0)
        vz15 = float(ind.get("vol_z_15m", 0.0) or 0.0)

        if atr <= 0:
            atr = last_price * 0.02  # fallback 2%

        row7 = r[:-1]  # drop market_symbol
        base_long = score_long(row7)
        base_short = score_short(row7)

        quality = 0.0
        quality += min(max((adx - ADX_MIN) / 10.0, 0.0), 2.0)
        quality += min(max(vz1 / 2.0, 0.0), 2.0)
        quality += min(max(vz15 / 2.0, 0.0), 2.0)

        pr_boost = 0.6 if label == "P1" else (0.3 if label == "P2" else 0.0)

        conf = confidence_score(ind, last_price)

        if long_ok:
            entry = last_price
            sl = entry - SL_ATR_MULT * atr
            tp = entry + TP_ATR_MULT * atr
            score = base_long + quality + pr_boost

            candidates.append(("BUY", sym, entry, tp, sl, score, label, conf))

            if not db_recent_open_exists(sym, "BUY", within_minutes=60):
                db_insert_signal(
                    ts=now_ts, symbol=sym, market_symbol=mkt_symbol, side="BUY",
                    entry=entry, sl=sl, tp=tp, confidence=conf, priority=label
                )
                LAST_SIGNAL_TS[sym] = now

        if short_ok:
            entry = last_price
            sl = entry + SL_ATR_MULT * atr
            tp = entry - TP_ATR_MULT * atr
            score = base_short + quality + pr_boost

            candidates.append(("SELL", sym, entry, tp, sl, score, label, conf))

            if not db_recent_open_exists(sym, "SELL", within_minutes=60):
                db_insert_signal(
                    ts=now_ts, symbol=sym, market_symbol=mkt_symbol, side="SELL",
                    entry=entry, sl=sl, tp=tp, confidence=conf, priority=label
                )
                LAST_SIGNAL_TS[sym] = now

    candidates.sort(key=lambda x: x[5], reverse=True)
    return candidates[:3]


def format_recommended_trades(recs: List[Tuple]) -> str:
    """
    rec: (side, sym, entry, tp, sl, score, label, conf)
    """
    if not recs:
        return "_No strong recommendations right now._"

    sections: Dict[str, List[str]] = {"P1": [], "P2": [], "P3": []}
    for side, sym, entry, exit_px, sl, score, label, conf in recs:
        tag = conf_label(int(conf))
        sections.setdefault(label, []).append(
            f"{side} {sym} — Conf {conf}/100 ({tag}) — Entry {entry:.6g} — TP {exit_px:.6g} — SL {sl:.6g} (score {score:.1f})"
        )

    lines = ["*Top 3 Recommendations:*"]
    for label in ("P1", "P2", "P3"):
        if sections.get(label):
            lines.append(f"\n_{label}_:")
            lines.extend(sections[label])

    return "\n".join(lines)


def scan_big_movers(best_spot: Dict[str, MarketVol], best_fut: Dict[str, MarketVol]) -> List[str]:
    """
    Find coins with:
      - futures notional volume > 1M (24h)
      - 24h % change > +10%
    Return list of BASE symbols.
    """
    movers: List[str] = []
    for base, f in best_fut.items():
        fut_usd = usd_notional(f)
        if fut_usd < 1_000_000:
            continue
        s = best_spot.get(base)
        pct24 = pct_change_24h(s, f)
        if pct24 > 10.0:
            movers.append(base)
    return sorted(set(movers))


# ================== FORMATTING ==================

def fmt_table(rows: List[List], title: str) -> str:
    if not rows:
        return f"*{title}*: _None_\n"
    pretty = [
        [
            r[0],
            m_dollars(r[1]),
            m_dollars(r[2]),
            pct_with_emoji(r[3]),
            pct_with_emoji(r[4]),
            pct_with_emoji(r[5]),
        ]
        for r in rows
    ]
    return (
        f"*{title}*:\n"
        "```\n"
        + tabulate(pretty, headers=["SYM", "F", "S", "%24H", "%4H", "%1H"], tablefmt="github")
        + "\n```\n"
    )


def fmt_single(sym: str, fusd: float, susd: float, p24: float, p4: float, p1: float) -> str:
    row = [[sym, m_dollars(fusd), m_dollars(susd),
            pct_with_emoji(p24), pct_with_emoji(p4), pct_with_emoji(p1)]]
    return (
        "```\n"
        + tabulate(row, headers=["SYM", "F", "S", "%24H", "%4H", "%1H"], tablefmt="github")
        + "\n```"
    )


# ================== EMAIL HELPERS ==================

def email_config_ok() -> bool:
    return all([EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FROM, EMAIL_TO])


def send_email(subject: str, body: str) -> bool:
    if not email_config_ok():
        logging.warning("Email not configured.")
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        msg.set_content(body)

        if EMAIL_PORT == 465:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=ctx) as s:
                s.login(EMAIL_USER, EMAIL_PASS)
                s.send_message(msg)
        else:
            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as s:
                s.starttls()
                s.login(EMAIL_USER, EMAIL_PASS)
                s.send_message(msg)
        return True
    except Exception as e:
        logging.exception("send_email failed: %s", e)
        return False


def melbourne_ok() -> bool:
    """
    Return True only if local time is between 12:30–01:00 in Melbourne,
    and it is NOT Sunday (local Melbourne date).
    """
    try:
        now_mel = datetime.now(ZoneInfo("Australia/Melbourne"))
        dow = now_mel.weekday()  # Monday=0 ... Sunday=6
        if dow == 6:
            return False

        h = now_mel.hour
        m = now_mel.minute

        after_1230 = (h > 12) or (h == 12 and m >= 30)
        before_1am = h < 1

        return after_1230 or before_1am
    except Exception:
        logging.exception("Melbourne time failed; defaulting to allowed.")
        return True


# ================== PERFORMANCE REPORT HELPERS ==================

def trade_return_pct(r) -> float:
    entry = float(r["entry"] or 0.0)
    exit_px = float(r["exit_price"] or 0.0)
    if entry <= 0 or exit_px <= 0:
        return 0.0
    side = (r["side"] or "").upper()
    if side == "BUY":
        return (exit_px - entry) / entry * 100.0
    else:
        return (entry - exit_px) / entry * 100.0


def summarize_rows(rows: List[sqlite3.Row]) -> str:
    total = len(rows)
    if total == 0:
        return "No data."

    wins = sum(1 for r in rows if r["status"] == "WIN")
    loss = sum(1 for r in rows if r["status"] == "LOSS")
    exp = sum(1 for r in rows if r["status"] == "EXPIRED")
    open_ = sum(1 for r in rows if r["status"] == "OPEN")

    closed = [r for r in rows if r["status"] in ("WIN", "LOSS", "EXPIRED")]
    avg_mfe = mean([r["mfe"] for r in closed if r["mfe"] is not None] or [0.0])
    avg_mae = mean([r["mae"] for r in closed if r["mae"] is not None] or [0.0])

    wr = (wins / (wins + loss) * 100.0) if (wins + loss) > 0 else 0.0

    return (
        f"Signals: {total}\n"
        f"WIN: {wins} | LOSS: {loss} | EXPIRED: {exp} | OPEN: {open_}\n"
        f"Win rate (W/(W+L)): {wr:.1f}%\n"
        f"Avg MFE: {avg_mfe:.2f}% | Avg MAE: {avg_mae:.2f}%\n"
        f"(Touch-based over {SIGNAL_MAX_AGE_HOURS}h using {EVAL_TF} candles)"
    )


def top_coins_summary(rows, top_n: int = 5, min_trades: int = 3) -> str:
    closed = [r for r in rows if r["status"] in ("WIN", "LOSS", "EXPIRED")]
    if not closed:
        return "No closed signals yet."

    agg = {}
    for r in closed:
        sym = r["symbol"]
        agg.setdefault(sym, {"n": 0, "w": 0, "sum_ret": 0.0})
        agg[sym]["n"] += 1
        if r["status"] == "WIN":
            agg[sym]["w"] += 1
        agg[sym]["sum_ret"] += trade_return_pct(r)

    items = [(sym, v) for sym, v in agg.items() if v["n"] >= min_trades]
    if not items:
        items = list(agg.items())

    scored = []
    for sym, v in items:
        n = v["n"]
        wr = (v["w"] / n * 100.0) if n else 0.0
        total = v["sum_ret"]
        avg = (v["sum_ret"] / n) if n else 0.0
        scored.append((sym, n, wr, avg, total))

    scored.sort(key=lambda x: x[4], reverse=True)
    best = scored[:top_n]
    worst = list(reversed(scored[-top_n:]))

    def fmt_list(lst):
        lines = []
        for sym, n, wr, avg, total in lst:
            lines.append(f"{sym:6} | n={n:2d} | WR={wr:5.1f}% | Avg={avg:6.2f}% | Total={total:7.2f}%")
        return "\n".join(lines)

    out = []
    out.append("Top 5 Best Coins (by Total Return%):")
    out.append(fmt_list(best) if best else "None")
    out.append("")
    out.append("Top 5 Worst Coins (by Total Return%):")
    out.append(fmt_list(worst) if worst else "None")
    return "\n".join(out)


def conf_wins_losses_summary(rows, top_n: int = 5, loss_conf_min: int = 80) -> str:
    closed = [r for r in rows if r["status"] in ("WIN", "LOSS")]
    if not closed:
        return "No WIN/LOSS trades yet."

    wins = [r for r in closed if r["status"] == "WIN"]
    losses = [r for r in closed if r["status"] == "LOSS" and int(r["confidence"] or 0) >= loss_conf_min]

    wins_sorted = sorted(
        wins,
        key=lambda r: (int(r["confidence"] or 0), trade_return_pct(r)),
        reverse=True
    )[:top_n]

    losses_sorted = sorted(losses, key=lambda r: trade_return_pct(r))[:top_n]  # most negative first

    def fmt_trade(r):
        sym = r["symbol"]
        side = r["side"]
        conf = int(r["confidence"] or 0)
        ret = trade_return_pct(r)
        entry = float(r["entry"] or 0.0)
        exit_px = float(r["exit_price"] or 0.0)
        return f"{sym:6} {side:4} | conf={conf:3d} | ret={ret:6.2f}% | entry={entry:.6g} exit={exit_px:.6g}"

    out = []
    out.append(f"Top {top_n} Highest-Confidence WINs:")
    out.append("\n".join(fmt_trade(r) for r in wins_sorted) if wins_sorted else "None")
    out.append("")
    out.append(f"Worst {top_n} High-Confidence LOSSes (conf>={loss_conf_min}):")
    out.append("\n".join(fmt_trade(r) for r in losses_sorted) if losses_sorted else "None")
    return "\n".join(out)


def weekly_report_text(days: int = 7) -> str:
    rows = db_fetch_since(days)
    base = summarize_rows(rows)
    tops = top_coins_summary(rows, top_n=5, min_trades=3)
    confs = conf_wins_losses_summary(rows, top_n=5, loss_conf_min=80)
    return base + "\n\n" + tops + "\n\n" + confs


# ================== PERFORMANCE EVALUATION (Touch-based TP/SL) ==================

def eval_signal_touch(sig_row: sqlite3.Row) -> Optional[Tuple[str, float, float, float]]:
    """
    Returns (status, exit_price, mfe_pct, mae_pct) or None if still OPEN.
    status: WIN/LOSS/EXPIRED
    Uses 15m OHLCV up to 24h from signal time.
    Conservative: if both SL and TP touched inside a candle => LOSS.
    """
    sig_ts = int(sig_row["ts"])
    now_ts = int(time.time())
    age_sec = now_ts - sig_ts

    if age_sec > SIGNAL_MAX_AGE_HOURS * 3600:
        ohlcv = fetch_ohlcv_cached_futures(sig_row["market_symbol"], EVAL_TF, limit=200)
        exit_px = ohlcv[-1][4] if ohlcv else float(sig_row["entry"])
        return ("EXPIRED", float(exit_px), 0.0, 0.0)

    limit = 140  # ~35 hours of 15m, enough buffer
    ohlcv = fetch_ohlcv_cached_futures(sig_row["market_symbol"], EVAL_TF, limit=limit)
    if not ohlcv:
        return None

    entry = float(sig_row["entry"])
    tp = float(sig_row["tp"])
    sl = float(sig_row["sl"])
    side = (sig_row["side"] or "").upper()

    candles = [c for c in ohlcv if c[0] >= sig_ts * 1000]
    if not candles:
        return None

    best_fav = 0.0
    worst_adv = 0.0

    for c in candles:
        _, o, h, l, close, vol = c

        if side == "BUY":
            fav = (h - entry) / entry * 100.0
            adv = (l - entry) / entry * 100.0
            best_fav = max(best_fav, fav)
            worst_adv = min(worst_adv, adv)

            hit_sl = l <= sl
            hit_tp = h >= tp

            if hit_sl and hit_tp:
                return ("LOSS", float(sl), float(best_fav), float(worst_adv))
            if hit_sl:
                return ("LOSS", float(sl), float(best_fav), float(worst_adv))
            if hit_tp:
                return ("WIN", float(tp), float(best_fav), float(worst_adv))

        else:  # SELL
            fav = (entry - l) / entry * 100.0
            adv = (entry - h) / entry * 100.0
            best_fav = max(best_fav, fav)
            worst_adv = min(worst_adv, -abs(adv))  # keep negative

            hit_sl = h >= sl
            hit_tp = l <= tp

            if hit_sl and hit_tp:
                return ("LOSS", float(sl), float(best_fav), float(worst_adv))
            if hit_sl:
                return ("LOSS", float(sl), float(best_fav), float(worst_adv))
            if hit_tp:
                return ("WIN", float(tp), float(best_fav), float(worst_adv))

    return None


async def performance_job(context: ContextTypes.DEFAULT_TYPE):
    try:
        opens = await asyncio.to_thread(db_fetch_open_signals)
        if not opens:
            return

        for sig in opens:
            res = eval_signal_touch(sig)
            if not res:
                continue
            status, exit_price, mfe, mae = res
            await asyncio.to_thread(
                db_close_signal,
                sig["id"],
                status,
                int(time.time()),
                float(exit_price),
                float(mfe),
                float(mae),
            )
    except Exception:
        logging.exception("performance_job failed")


# ================== WEEKLY REPORT JOB ==================

async def weekly_report_job(context: ContextTypes.DEFAULT_TYPE):
    global LAST_WEEKLY_SENT_YMD
    try:
        now_mel = datetime.now(ZoneInfo("Australia/Melbourne"))

        # Monday only
        if now_mel.weekday() != 0:
            return

        # Send between 09:10–09:15
        if not (now_mel.hour == 9 and 10 <= now_mel.minute <= 15):
            return

        ymd = now_mel.strftime("%Y-%m-%d")
        if LAST_WEEKLY_SENT_YMD == ymd:
            return

        txt_body = await asyncio.to_thread(weekly_report_text, 7)
        msg = "*Weekly Performance (7D) + Top/Worst + High-Conf*\n" + "```\n" + txt_body + "\n```"

        if LAST_CHAT_ID is not None:
            await context.bot.send_message(chat_id=LAST_CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)

        if email_config_ok():
            send_email("Weekly Performance (7D) + Top/Worst + High-Conf", txt_body)

        LAST_WEEKLY_SENT_YMD = ymd

    except Exception:
        logging.exception("weekly_report_job failed")


# ================== TELEGRAM HANDLERS ==================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_CHAT_ID
    LAST_CHAT_ID = update.effective_chat.id
    await update.message.reply_text(
        "Commands:\n"
        "• /screen — show P1, P2, P3 + recommendations\n"
        "• /notify_on /notify_off /notify\n"
        "• /diag — short diagnostics\n"
        "• /report — performance 7D\n"
        "• /report30 — performance 30D\n"
        "• /last10 — last 10 signals\n"
        "• /stats BTC — symbol stats (30D)\n"
        "• Type a symbol (e.g. PYTH) for its row"
    )


async def screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_CHAT_ID
    LAST_CHAT_ID = update.effective_chat.id
    try:
        # clear short-lived caches
        PCT4H_CACHE.clear()
        PCT1H_CACHE.clear()

        best_spot, best_fut, raw_spot, raw_fut = await asyncio.to_thread(load_best)
        p1, p2, p3 = await asyncio.to_thread(build_priorities, best_spot, best_fut)

        recs = pick_best_trades(p1, p2, p3)
        rec_text = format_recommended_trades(recs)

        big_movers = scan_big_movers(best_spot, best_fut)
        movers_text = ""
        if big_movers:
            movers_text = "\n\n*24h +10% movers (F vol >1M):*\n" + ", ".join(big_movers)

        msg = (
            fmt_table(p1, "P1 (F≥5M & S≥0.5M — pinned excluded)")
            + fmt_table(p2, "P2 (F≥2M — pinned excluded)")
            + fmt_table(p3, "P3 (Pinned + S≥3M)")
            + f"tickers: spot={raw_spot}, fut={raw_fut}\n\n"
            + "*Recommended trades (Futures only):*\n"
            + rec_text
            + movers_text
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logging.exception("screen error")
        await update.message.reply_text(f"Error: {e}")


async def diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"*Diag*\n"
        f"- signals: FUTURES only\n"
        f"- trigger/confirm: {IND_TREND_TF}/{IND_CONFIRM_TF} + HTF {IND_HTF}\n"
        f"- email window: 12:30–01:00 Melbourne (no Sundays)\n"
        f"- scan interval: {CHECK_INTERVAL_MIN} min\n"
        f"- perf eval: every {SIGNAL_EVAL_INTERVAL_MIN} min (touch-based {SIGNAL_MAX_AGE_HOURS}h)\n"
        f"- email: {'ON' if NOTIFY_ON else 'OFF'} → {EMAIL_TO or 'n/a'}"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def notify_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global NOTIFY_ON
    NOTIFY_ON = True
    await update.message.reply_text("Email alerts: ON")


async def notify_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global NOTIFY_ON
    NOTIFY_ON = False
    await update.message.reply_text("Email alerts: OFF")


async def notify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Email alerts: {'ON' if NOTIFY_ON else 'OFF'} → {EMAIL_TO or 'n/a'}"
    )


async def report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = await asyncio.to_thread(db_fetch_since, 7)
    txt = "*Performance (7D)*\n" + "```\n" + summarize_rows(rows) + "\n```"
    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)


async def report30(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = await asyncio.to_thread(db_fetch_since, 30)
    txt = "*Performance (30D)*\n" + "```\n" + summarize_rows(rows) + "\n```"
    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)


async def last10(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = await asyncio.to_thread(db_fetch_last, 10)
    if not rows:
        await update.message.reply_text("No signals yet.")
        return

    lines = []
    for r in rows:
        t = datetime.fromtimestamp(r["ts"], ZoneInfo("Australia/Melbourne")).strftime("%m-%d %H:%M")
        lines.append(
            f"{t} | {r['side']} {r['symbol']} | conf {r['confidence']} | {r['status']} | entry {float(r['entry']):.6g}"
        )
    txt = "*Last 10 signals*\n" + "```\n" + "\n".join(lines) + "\n```"
    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /stats BTC")
        return
    sym = re.sub(r"[^A-Za-z$]", "", context.args[0]).upper().lstrip("$")
    rows = await asyncio.to_thread(db_fetch_symbol_stats, sym, 30)
    txt = f"*Stats {sym} (30D)*\n" + "```\n" + summarize_rows(rows) + "\n```"
    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)


async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    token = re.sub(r"[^A-Za-z$]", "", text).upper().lstrip("$")
    if len(token) < 2:
        return
    try:
        best_spot, best_fut, _, _ = await asyncio.to_thread(load_best)
        s = best_spot.get(token)
        f = best_fut.get(token)
        fusd = usd_notional(f) if f else 0.0
        susd = usd_notional(s) if s else 0.0
        if fusd == 0.0 and susd == 0.0:
            await update.message.reply_text("Symbol not found.")
            return
        pct24 = pct_change_24h(s, f)
        if not f:
            await update.message.reply_text("No futures market found for this symbol on CoinEx.")
            return
        pct4 = compute_pct_for_symbol(f.symbol, 4)
        pct1 = compute_pct_for_symbol(f.symbol, 1)
        msg = fmt_single(token, fusd, susd, pct24, pct4, pct1)
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logging.exception("text_router error")
        await update.message.reply_text(f"Error: {e}")


# ================== ALERT JOB ==================

async def alert_job(context: ContextTypes.DEFAULT_TYPE):
    global LAST_EMAIL_TS, LAST_ALL_SYMBOLS
    try:
        if not NOTIFY_ON:
            return
        if not melbourne_ok():
            return
        if not email_config_ok():
            return

        now = time.time()
        if now - LAST_EMAIL_TS < EMAIL_MIN_INTERVAL_SEC:
            return

        # Clear short-lived caches
        PCT4H_CACHE.clear()
        PCT1H_CACHE.clear()

        best_spot, best_fut, _, _ = await asyncio.to_thread(load_best)
        p1, p2, p3 = await asyncio.to_thread(build_priorities, best_spot, best_fut)

        recs = pick_best_trades(p1, p2, p3)
        big_movers = scan_big_movers(best_spot, best_fut)

        if not recs and not big_movers:
            return

        rec_symbols: Set[str] = {sym for _, sym, *_ in recs}
        mover_symbols: Set[str] = set(big_movers)
        current_symbols: Set[str] = rec_symbols | mover_symbols

        if current_symbols and current_symbols == LAST_ALL_SYMBOLS:
            return

        parts = []
        if recs:
            parts.append(format_recommended_trades(recs))
        if big_movers:
            parts.append("*24h +10% movers (F vol >1M):*\n" + ", ".join(sorted(mover_symbols)))

        body = "\n\n".join(parts)

        subject = "Crypto Alert: Futures Signals & 24h Movers"
        if big_movers and not recs:
            subject = "Crypto Alert: 24h +10% Movers"

        if send_email(subject, body):
            LAST_EMAIL_TS = now
            LAST_ALL_SYMBOLS = current_symbols

    except Exception as e:
        logging.exception("alert_job error: %s", e)


# ================== ERROR HANDLER ==================

async def log_err(update: object, context: ContextTypes.DEFAULT_TYPE):
    try:
        raise context.error
    except Conflict:
        logging.warning("Conflict: another instance already polling.")
    except Exception as e:
        logging.exception("Unhandled error: %s", e)


# ================== MAIN ==================

def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN missing")

    logging.basicConfig(level=logging.INFO)

    # Init DB
    db_init()

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("screen", screen))
    app.add_handler(CommandHandler("diag", diag))
    app.add_handler(CommandHandler("notify_on", notify_on))
    app.add_handler(CommandHandler("notify_off", notify_off))
    app.add_handler(CommandHandler("notify", notify))

    app.add_handler(CommandHandler("report", report))
    app.add_handler(CommandHandler("report30", report30))
    app.add_handler(CommandHandler("last10", last10))
    app.add_handler(CommandHandler("stats", stats))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))
    app.add_error_handler(log_err)

    if getattr(app, "job_queue", None):
        app.job_queue.run_repeating(alert_job, interval=CHECK_INTERVAL_MIN * 60, first=10)
        app.job_queue.run_repeating(performance_job, interval=SIGNAL_EVAL_INTERVAL_MIN * 60, first=30)
        # weekly report checker runs every 60s but only sends in the scheduled window
        app.job_queue.run_repeating(weekly_report_job, interval=60, first=20)
    else:
        logging.warning(
            "JobQueue not available. Make sure you installed "
            '"python-telegram-bot[job-queue]" in requirements.txt.'
        )

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
