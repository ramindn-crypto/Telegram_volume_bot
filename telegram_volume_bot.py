#!/usr/bin/env python3
"""
Telegram crypto screener & alert bot (CoinEx via CCXT)

Features:
- Uses CoinEx via CCXT for 24h volume & price
- Priorities:
    P1: F≥5M & S≥0.5M (pinned excluded)
    P2: F≥2M (pinned excluded)
    P3: Pinned + S≥3M
- Table columns: SYM | F | S | %24H | %4H | %1H | %15M
- Intraday alert rules (base logic):
    BUY (neutral 24h):  1h ≥ +2% AND 15m ≥ +2%
    SELL (neutral 24h): 1h ≤ -2% AND 15m ≤ -2%
- 24h trend filter (Interpretation B):
    If 24h > +5%  → SELL disabled (but BUY still needs confirmation)
    If 24h < -5%  → BUY disabled (but SELL still needs confirmation)

Extra filters:
- 4h 200 EMA trend:
    If last_price > EMA200(4h) → only BUY allowed
    If last_price < EMA200(4h) → only SELL allowed
- Coinalyze OI filter hook:
    get_oi_change_4h_from_coinalyze(base_symbol) → if implemented and ≤ 0, disables signals
    (currently stubbed to return None so it never blocks)

Signals:
- BUY & SELL are mutually exclusive per symbol.
- Only symbols that pass ALL filters are considered for alerts & recommendations.

Emails:
- Email alerts ANYTIME (no time-of-day limit)
- Max 1 email every 15 minutes (global)
- Email sent only if:
    * At least one signal (after all filters), AND
    * The set of BUY/SELL symbols changed since last email

Recommendations:
- From symbols that pass alert rules:
    → Score BUY/SELL candidates
    → Pick TOP 2 recommendations (BUY or SELL)
    → Provide Entry / Exit / SL

Telegram:
/start
/screen   → ALWAYS show full P1/P2/P3, but "Recommended trades" only if there is at least 1 signal
/notify_on /notify_off /notify
/diag
Typing a symbol (e.g. PYTH) → one-row table for that symbol
"""

import asyncio
import logging
import os
import time
import re
import ssl
import smtplib
from email.message import EmailMessage
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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

# Email config (set in Render env)
EMAIL_ENABLED_DEFAULT = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST = os.environ.get("EMAIL_HOST", "")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "465"))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", EMAIL_USER)
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_USER)

# Scheduler / alerts
CHECK_INTERVAL_MIN = 5  # run alert job every 5 minutes

# Intraday alert thresholds
BUY_PCT_1H_MIN = 2.0
BUY_PCT_15M_MIN = 2.0
SELL_PCT_1H_MAX = -2.0
SELL_PCT_15M_MAX = -2.0

# 24h trend filter (Interpretation B)
TREND_LONG_THRESHOLD = 5.0     # > +5% → disable SELL
TREND_SHORT_THRESHOLD = -5.0   # < -5% → disable BUY

# Per-symbol alert cooldown
ALERT_THROTTLE_SEC = 15 * 60  # 15 minutes per (priority,side,symbol)

# Global email gap
EMAIL_MIN_GAP_SEC = 15 * 60  # 1 email every 15 minutes (no daily cap)

# Globals
LAST_ERROR: Optional[str] = None
NOTIFY_ON: bool = EMAIL_ENABLED_DEFAULT

PCT4H_CACHE: Dict[Tuple[str, str, int], float] = {}
PCT1H_CACHE: Dict[Tuple[str, str, int], float] = {}
PCT15M_CACHE: Dict[Tuple[str, str, int], float] = {}

EMA4H_CACHE: Dict[str, float] = {}
OI4H_CACHE: Dict[str, Optional[float]] = {}

ALERT_SENT_CACHE: Dict[Tuple[str, str], float] = {}  # (priority_side, symbol) -> last_sent_time
EMAIL_SEND_LOG: List[float] = []  # timestamps of sent emails

# For "only send if symbols/directions changed"
LAST_SIGNAL_SIGNATURE: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None


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


# ================== HELPERS ==================

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
    for mv in (mv_spot, mv_fut):
        if mv and mv.percentage:
            return float(mv.percentage)
    mv = mv_spot or mv_fut
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
    """Return millions as a rounded integer string."""
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


# ================== PERCENTAGES (4H / 1H / 15M) ==================

def compute_pct_for_symbol_1h(symbol: str, hours: int, prefer_swap: bool = True) -> float:
    """
    Compute % change over the last N hours using 1h candles.
    Used for %4H (hours=4) and %1H (hours=1).
    """
    try_order = ["swap", "spot"] if prefer_swap else ["spot", "swap"]
    cache = PCT4H_CACHE if hours == 4 else PCT1H_CACHE

    for dtype in try_order:
        cache_key = (dtype, symbol, hours)
        if cache_key in cache:
            return cache[cache_key]
        try:
            ex = build_exchange(dtype)
            ex.load_markets()
            candles = ex.fetch_ohlcv(symbol, timeframe="1h", limit=hours + 1)
            if not candles or len(candles) <= hours:
                continue
            closes = [c[4] for c in candles][- (hours + 1):]
            if not closes or not closes[0]:
                pct = 0.0
            else:
                pct = (closes[-1] - closes[0]) / closes[0] * 100.0
            cache[cache_key] = pct
            return pct
        except Exception:
            logging.exception(
                "compute_pct_for_symbol_1h: %dh failed for %s (%s)", hours, symbol, dtype
            )
            continue
    return 0.0


def compute_pct_for_symbol_15m(symbol: str, minutes: int = 15, prefer_swap: bool = True) -> float:
    """
    Compute % change over the last N minutes using 15m candles.
    For 15 min window we only need 2 candles (last & previous).
    """
    try_order = ["swap", "spot"] if prefer_swap else ["spot", "swap"]
    cache = PCT15M_CACHE

    for dtype in try_order:
        cache_key = (dtype, symbol, minutes)
        if cache_key in cache:
            return cache[cache_key]
        try:
            ex = build_exchange(dtype)
            ex.load_markets()
            candles = ex.fetch_ohlcv(symbol, timeframe="15m", limit=2)
            if not candles or len(candles) < 2:
                continue
            c0, c1 = candles[-2], candles[-1]
            if not c0[4]:
                pct = 0.0
            else:
                pct = (c1[4] - c0[4]) / c0[4] * 100.0
            cache[cache_key] = pct
            return pct
        except Exception:
            logging.exception(
                "compute_pct_for_symbol_15m: %dmin failed for %s (%s)", minutes, symbol, dtype
            )
            continue
    return 0.0


# ================== EMA + OI HELPERS ==================

def get_ema_4h_200(symbol: str) -> float:
    """
    Compute 200 EMA on 4h candles for the given trading symbol.
    We prefer swap markets for futures symbols like 'BTC/USDT:USDT'.
    Cached per symbol.
    """
    if not symbol:
        return 0.0
    if symbol in EMA4H_CACHE:
        return EMA4H_CACHE[symbol]

    try:
        # crude detection: if it has ':', assume futures (swap)
        default_type = "swap" if ":" in symbol else "spot"
        ex = build_exchange(default_type)
        ex.load_markets()
        candles = ex.fetch_ohlcv(symbol, timeframe="4h", limit=210)
        if not candles or len(candles) < 200:
            EMA4H_CACHE[symbol] = 0.0
            return 0.0

        closes = [c[4] for c in candles]
        # standard EMA
        k = 2 / (200 + 1)
        ema = closes[0]
        for price in closes[1:]:
            ema = price * k + ema * (1 - k)

        EMA4H_CACHE[symbol] = float(ema)
        return float(ema)
    except Exception:
        logging.exception("get_ema_4h_200 failed for %s", symbol)
        EMA4H_CACHE[symbol] = 0.0
        return 0.0


def get_oi_change_4h_from_coinalyze(base_symbol: str) -> Optional[float]:
    """
    Placeholder for Coinalyze OI 4h change.
    Return percentage OI change over last 4h or None if unavailable.

    IMPORTANT:
      - Currently returns None so it DOES NOT block any signals.
      - When you have Coinalyze API access, implement the HTTP call here:
          1) Build URL with your API key & symbol
          2) Fetch OI history
          3) Compute % change over 4h
          4) Cache result in OI4H_CACHE[base_symbol]

    Example logic (pseudo):
        if base_symbol in OI4H_CACHE:
            return OI4H_CACHE[base_symbol]
        ...
        OI4H_CACHE[base_symbol] = oi_pct_change
        return oi_pct_change
    """
    return OI4H_CACHE.get(base_symbol, None)


# ================== PRIORITIES ==================

def load_best():
    """Load top spot & futures markets per BASE symbol from CoinEx."""
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
    Each row: [SYM, FUSD, SUSD, %24H, %4H, %1H, %15M, LASTPRICE, MKTSYM]
      - MKTSYM is the futures (swap) symbol if exists, else spot symbol
      - LASTPRICE is internal, not shown in table.
    """
    p1: List[List] = []
    p2: List[List] = []
    p3: List[List] = []
    used = set()

    # ---- P1: F≥5M & S≥500k (pinned excluded) ----
    for base in set(best_spot) & set(best_fut):
        if base in PINNED_SET:
            continue
        s = best_spot[base]
        f = best_fut[base]
        fut_usd = usd_notional(f)
        spot_usd = usd_notional(s)
        if fut_usd >= P1_FUT_MIN and spot_usd >= P1_SPOT_MIN:
            pct24 = pct_change_24h(s, f)
            symbol_for_pct = f.symbol
            pct4 = compute_pct_for_symbol_1h(symbol_for_pct, 4)
            pct1 = compute_pct_for_symbol_1h(symbol_for_pct, 1)
            pct15 = compute_pct_for_symbol_15m(symbol_for_pct, 15)
            last_price = f.last or s.last or 0.0
            p1.append(
                [base, fut_usd, spot_usd, pct24, pct4, pct1, pct15, last_price, symbol_for_pct]
            )

    p1.sort(key=lambda x: x[1], reverse=True)
    p1 = p1[:TOP_N_P1]
    used |= {r[0] for r in p1}

    # ---- P2: F≥2M (pinned & already used excluded) ----
    for base, f in best_fut.items():
        if base in used or base in PINNED_SET:
            continue
        fut_usd = usd_notional(f)
        if fut_usd >= P2_FUT_MIN:
            s = best_spot.get(base)
            spot_usd = usd_notional(s) if s else 0.0
            pct24 = pct_change_24h(s, f)
            symbol_for_pct = f.symbol
            pct4 = compute_pct_for_symbol_1h(symbol_for_pct, 4)
            pct1 = compute_pct_for_symbol_1h(symbol_for_pct, 1)
            pct15 = compute_pct_for_symbol_15m(symbol_for_pct, 15)
            last_price = f.last or (s.last if s else 0.0)
            p2.append(
                [base, fut_usd, spot_usd, pct24, pct4, pct1, pct15, last_price, symbol_for_pct]
            )

    p2.sort(key=lambda x: x[1], reverse=True)
    p2 = p2[:TOP_N_P2]
    used |= {r[0] for r in p2}

    # ---- P3: pinned + others with Spot≥3M (pinned only in P3) ----
    tmp: Dict[str, List] = {}

    # pinned first
    for base in PINNED_P3:
        s = best_spot.get(base)
        f = best_fut.get(base)
        if not s and not f:
            continue
        fut_usd = usd_notional(f) if f else 0.0
        spot_usd = usd_notional(s) if s else 0.0
        pct24 = pct_change_24h(s, f)
        symbol_for_pct = f.symbol if f else (s.symbol if s else "")
        pct4 = compute_pct_for_symbol_1h(symbol_for_pct, 4) if symbol_for_pct else 0.0
        pct1 = compute_pct_for_symbol_1h(symbol_for_pct, 1) if symbol_for_pct else 0.0
        pct15 = compute_pct_for_symbol_15m(symbol_for_pct, 15) if symbol_for_pct else 0.0
        last_price = (f.last if f else (s.last if s else 0.0))
        tmp[base] = [
            base,
            fut_usd,
            spot_usd,
            pct24,
            pct4,
            pct1,
            pct15,
            last_price,
            symbol_for_pct,
        ]

    # non-pinned, not used, Spot≥3M
    for base, s in best_spot.items():
        if base in used or base in PINNED_SET:
            continue
        spot_usd = usd_notional(s)
        if spot_usd >= P3_SPOT_MIN:
            f = best_fut.get(base)
            fut_usd = usd_notional(f) if f else 0.0
            pct24 = pct_change_24h(s, f)
            symbol_for_pct = f.symbol if f else s.symbol
            pct4 = compute_pct_for_symbol_1h(symbol_for_pct, 4)
            pct1 = compute_pct_for_symbol_1h(symbol_for_pct, 1)
            pct15 = compute_pct_for_symbol_15m(symbol_for_pct, 15)
            last_price = (f.last if f else s.last)
            tmp[base] = [
                base,
                fut_usd,
                spot_usd,
                pct24,
                pct4,
                pct1,
                pct15,
                last_price,
                symbol_for_pct,
            ]

    all_rows = list(tmp.values())
    pinned_rows = [r for r in all_rows if r[0] in PINNED_SET]
    other_rows = [r for r in all_rows if r[0] not in PINNED_SET]

    pinned_rows.sort(key=lambda r: r[2], reverse=True)
    other_rows.sort(key=lambda r: r[2], reverse=True)
    p3 = (pinned_rows + other_rows)[:TOP_N_P3]

    return p1, p2, p3


# ================== SCORING & RECOMMENDATIONS ==================

def score_long(row: List) -> float:
    """
    Long (BUY) score based on:
    - Positive 4h, 1h, 15m momentum
    - Futures liquidity
    """
    _, fut_usd, _, pct24, pct4, pct1, pct15, _, _ = row
    score = 0.0

    if pct4 > 0:
        score += min(pct4 / 2.0, 3.0)
    if pct1 > 0:
        score += min(pct1 / 2.0, 3.0)
    if pct15 > 0:
        score += min(pct15 / 2.0, 2.0)

    if fut_usd > 10_000_000:
        score += 2.0
    elif fut_usd > 5_000_000:
        score += 1.0

    return max(score, 0.0)


def score_short(row: List) -> float:
    """
    Short (SELL) score based on:
    - Negative 4h, 1h, 15m momentum
    - Futures liquidity
    """
    _, fut_usd, _, pct24, pct4, pct1, pct15, _, _ = row
    score = 0.0

    if pct4 < 0:
        score += min(abs(pct4) / 2.0, 3.0)
    if pct1 < 0:
        score += min(abs(pct1) / 2.0, 3.0)
    if pct15 < 0:
        score += min(abs(pct15) / 2.0, 2.0)

    if fut_usd > 10_000_000:
        score += 2.0
    elif fut_usd > 5_000_000:
        score += 1.0

    return max(score, 0.0)


def pick_best_trades(p1: List[List], p2: List[List], p3: List[List]) -> List[Tuple]:
    """
    Score BUY/SELL and return ONLY TOP 2 recommendations.
    Assumes rows already passed the alert filters.
    Uses 24h trend to resolve conflicts.
    """
    rows = p1 + p2 + p3
    scored: List[Tuple[str, str, float, float, float, float]] = []

    for r in rows:
        sym, _, _, pct24, pct4, pct1, pct15, last_price, _ = r
        if not last_price or last_price <= 0:
            continue

        long_s = score_long(r)
        short_s = score_short(r)

        # 24h filter for recommendations
        if pct24 > TREND_LONG_THRESHOLD:
            short_s = 0.0
        elif pct24 < TREND_SHORT_THRESHOLD:
            long_s = 0.0

        side: Optional[str] = None
        score: float = 0.0

        if long_s > 0 and short_s > 0:
            if pct24 > 0:
                side = "BUY"
                score = long_s
            elif pct24 < 0:
                side = "SELL"
                score = short_s
            else:
                if long_s >= short_s:
                    side = "BUY"
                    score = long_s
                else:
                    side = "SELL"
                    score = short_s
        elif long_s > 0:
            side = "BUY"
            score = long_s
        elif short_s > 0:
            side = "SELL"
            score = short_s

        if side is None:
            continue

        entry = last_price
        if side == "BUY":
            sl = entry * 0.96
            tp = entry * 1.08
        else:
            sl = entry * 1.04
            tp = entry * 0.94

        scored.append((side, sym, entry, tp, sl, score))

    scored.sort(key=lambda x: x[5], reverse=True)
    return scored[:2]


def format_recommended_trades(recs: List[Tuple]) -> str:
    if not recs:
        return "_No strong recommendations right now._"

    lines = ["*Top 2 Recommendations:*"]
    for side, sym, entry, exit_price, sl, score in recs:
        lines.append(
            f"{side} {sym} — Entry {entry:.6g} — Exit {exit_price:.6g} — SL {sl:.6g} (score {score:.1f})"
        )
    return "\n".join(lines)


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
            pct_with_emoji(r[6]),
        ]
        for r in rows
    ]
    return (
        f"*{title}*:\n"
        "```\n"
        + tabulate(
            pretty,
            headers=["SYM", "F", "S", "%24H", "%4H", "%1H", "%15M"],
            tablefmt="github",
        )
        + "\n```\n"
    )


def fmt_single(sym: str, fusd: float, susd: float, p24: float, p4: float, p1: float, p15: float) -> str:
    row = [[sym,
            m_dollars(fusd),
            m_dollars(susd),
            pct_with_emoji(p24),
            pct_with_emoji(p4),
            pct_with_emoji(p1),
            pct_with_emoji(p15)]]
    return (
        "```\n"
        + tabulate(row, headers=["SYM", "F", "S", "%24H", "%4H", "%1H", "%15M"], tablefmt="github")
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


def email_quota_ok() -> bool:
    """
    Global email limit:
    - Max 1 email every 15 minutes (no daily cap, no time-of-day limit)
    """
    now = time.time()
    if EMAIL_SEND_LOG:
        last_sent = EMAIL_SEND_LOG[-1]
        if now - last_sent < EMAIL_MIN_GAP_SEC:
            return False
    return True


def record_email():
    EMAIL_SEND_LOG.append(time.time())


def should_alert(priority: str, symbol: str, side: str) -> bool:
    """Per (priority,side,symbol) cool-down."""
    now = time.time()
    key = (f"{priority}_{side}", symbol)
    last = ALERT_SENT_CACHE.get(key, 0.0)
    if now - last >= ALERT_THROTTLE_SEC:
        ALERT_SENT_CACHE[key] = now
        return True
    return False


# ================== SIGNAL DETECTION ==================

def detect_signals(p1: List[List], p2: List[List], p3: List[List]):
    """
    Detect BUY/SELL signals based on:
      - intraday rules (1h & 15m)
      - 24h trend filter
      - 4h EMA200 trend filter
      - (optional) Coinalyze OI 4h filter
      - mutual exclusivity per symbol

    Returns:
      {
        "P1": {"BUY": set(...), "SELL": set(...)},
        "P2": {"BUY": ...},
        "P3": ...
      }
    """
    signals = {
        "P1": {"BUY": set(), "SELL": set()},
        "P2": {"BUY": set(), "SELL": set()},
        "P3": {"BUY": set(), "SELL": set()},
    }

    priority_map = [("P1", p1), ("P2", p2), ("P3", p3)]

    for label, rows in priority_map:
        for r in rows:
            sym, _, _, pct24, pct4, pct1, pct15, last_price, mkt_symbol = r

            buy_ok = False
            sell_ok = False

            # Intraday conditions
            if pct1 >= BUY_PCT_1H_MIN and pct15 >= BUY_PCT_15M_MIN:
                buy_ok = True
            if pct1 <= SELL_PCT_1H_MAX and pct15 <= SELL_PCT_15M_MAX:
                sell_ok = True

            # 24h trend filter
            if pct24 > TREND_LONG_THRESHOLD:
                sell_ok = False
            elif pct24 < TREND_SHORT_THRESHOLD:
                buy_ok = False

            # 4h EMA200 trend filter
            ema_4h = get_ema_4h_200(mkt_symbol)
            if ema_4h > 0 and last_price > 0:
                if last_price > ema_4h:
                    # Uptrend → don't short
                    sell_ok = False
                elif last_price < ema_4h:
                    # Downtrend → don't long
                    buy_ok = False

            # Coinalyze OI 4h filter (if available)
            oi_change_4h = get_oi_change_4h_from_coinalyze(sym)
            if oi_change_4h is not None and oi_change_4h <= 0:
                # If OI not increasing, skip both sides
                buy_ok = False
                sell_ok = False

            # Mutual exclusivity resolution
            if buy_ok and sell_ok:
                if pct24 > 0:
                    sell_ok = False
                elif pct24 < 0:
                    buy_ok = False
                else:
                    if abs(pct1) >= abs(pct15):
                        if pct1 >= 0:
                            sell_ok = False
                        else:
                            buy_ok = False
                    else:
                        if pct15 >= 0:
                            sell_ok = False
                        else:
                            buy_ok = False

            if buy_ok:
                signals[label]["BUY"].add(sym)
            if sell_ok:
                signals[label]["SELL"].add(sym)

    return signals


def scan_for_alerts(p1: List[List], p2: List[List], p3: List[List], rec_text: str) -> Optional[str]:
    """
    Use detect_signals() to find coins that meet alert rules.
    Then apply:
      - per-symbol cooldown (should_alert)
      - global signature change (only send if BUY/SELL sets changed)
    """
    global LAST_SIGNAL_SIGNATURE

    overall_buy: set[str] = set()
    overall_sell: set[str] = set()
    lines: List[str] = []

    signals = detect_signals(p1, p2, p3)
    priority_map = [("P1", p1), ("P2", p2), ("P3", p3)]

    for label, rows in priority_map:
        buy_hits: List[str] = []
        sell_hits: List[str] = []

        for sym in signals[label]["BUY"]:
            if should_alert(label, sym, "BUY"):
                buy_hits.append(sym)
        for sym in signals[label]["SELL"]:
            if should_alert(label, sym, "SELL"):
                sell_hits.append(sym)

        if buy_hits or sell_hits:
            parts = []
            if buy_hits:
                overall_buy.update(buy_hits)
                parts.append("BUY: " + ", ".join(sorted(buy_hits)))
            if sell_hits:
                overall_sell.update(sell_hits)
                parts.append("SELL: " + ", ".join(sorted(sell_hits)))
            lines.append(f"{label}: " + " | ".join(parts))

    if not lines:
        return None

    signature = (tuple(sorted(overall_buy)), tuple(sorted(overall_sell)))
    if LAST_SIGNAL_SIGNATURE == signature:
        return None
    LAST_SIGNAL_SIGNATURE = signature

    body = "Signals (24h filter + EMA + 1h & 15m):\n\n" + "\n".join(lines)
    if rec_text:
        body += "\n\nRecommended trades:\n" + rec_text

    return body


# ================== TELEGRAM HANDLERS ==================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n"
        "• /screen — show P1, P2, P3 (full tables) and ONLY show recommendations if alert rules are met\n"
        "• /notify_on /notify_off /notify\n"
        "• /diag — short diagnostics\n"
        "• Type a symbol (e.g. PYTH) for its row"
    )


async def screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /screen:
    - Load P1/P2/P3
    - ALWAYS show full tables (all rows that passed volume filters)
    - Then compute signals
    - Only if at least one symbol meets alert rules, show "Recommended trades" at the end.
      If no symbol matches, no recommendations section is printed.
    """
    try:
        PCT4H_CACHE.clear()
        PCT1H_CACHE.clear()
        PCT15M_CACHE.clear()
        EMA4H_CACHE.clear()
        # OI4H_CACHE is not cleared; it's conceptually slower-changing (stub anyway).

        best_spot, best_fut, raw_spot, raw_fut = await asyncio.to_thread(load_best)
        p1, p2, p3 = await asyncio.to_thread(build_priorities, best_spot, best_fut)

        msg = (
            fmt_table(p1, "P1 (F≥5M & S≥0.5M — pinned excluded)")
            + fmt_table(p2, "P2 (F≥2M — pinned excluded)")
            + fmt_table(p3, "P3 (Pinned + S≥3M)")
            + f"tickers: spot={raw_spot}, fut={raw_fut}\n\n"
        )

        # ---- Detect signals & build recommendations ONLY if there are signals ----
        signals = detect_signals(p1, p2, p3)

        def filter_rows(rows: List[List], label: str) -> List[List]:
            active_syms = signals[label]["BUY"] | signals[label]["SELL"]
            return [r for r in rows if r[0] in active_syms]

        p1_sig = filter_rows(p1, "P1")
        p2_sig = filter_rows(p2, "P2")
        p3_sig = filter_rows(p3, "P3")

        if p1_sig or p2_sig or p3_sig:
            recs = pick_best_trades(p1_sig, p2_sig, p3_sig)
            if recs:
                rec_text = format_recommended_trades(recs)
                msg += "*Recommended trades:*\n" + rec_text

        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logging.exception("screen error")
        await update.message.reply_text(f"Error: {e}")


async def diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"*Diag*\n"
        f"- BUY (neutral 24h): 1h ≥ +{BUY_PCT_1H_MIN:.0f}% & 15m ≥ +{BUY_PCT_15M_MIN:.0f}%\n"
        f"- SELL (neutral 24h): 1h ≤ {SELL_PCT_1H_MAX:.0f}% & 15m ≤ {SELL_PCT_15M_MAX:.0f}%\n"
        f"- 24h filter: > +{TREND_LONG_THRESHOLD:.0f}% → disable SELL, < {TREND_SHORT_THRESHOLD:.0f}% → disable BUY\n"
        f"- EMA filter: 4h 200 EMA (long above, short below)\n"
        f"- OI filter: Coinalyze hook (currently inactive / stub)\n"
        f"- interval: {CHECK_INTERVAL_MIN} min\n"
        f"- email: {'ON' if NOTIFY_ON else 'OFF'}\n"
        f"- global gap: 1 email / 15 min"
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
        symbol_for_pct = f.symbol if f else (s.symbol if s else "")
        pct4 = compute_pct_for_symbol_1h(symbol_for_pct, 4) if symbol_for_pct else 0.0
        pct1 = compute_pct_for_symbol_1h(symbol_for_pct, 1) if symbol_for_pct else 0.0
        pct15 = compute_pct_for_symbol_15m(symbol_for_pct, 15) if symbol_for_pct else 0.0

        msg = fmt_single(token, fusd, susd, pct24, pct4, pct1, pct15)
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logging.exception("text_router error")
        await update.message.reply_text(f"Error: {e}")


# ================== ALERT JOB ==================

async def alert_job(context: ContextTypes.DEFAULT_TYPE):
    try:
        if not NOTIFY_ON:
            return
        if not email_config_ok():
            return
        if not email_quota_ok():
            return

        PCT4H_CACHE.clear()
        PCT1H_CACHE.clear()
        PCT15M_CACHE.clear()
        EMA4H_CACHE.clear()

        best_spot, best_fut, _, _ = await asyncio.to_thread(load_best)
        p1, p2, p3 = await asyncio.to_thread(build_priorities, best_spot, best_fut)

        # For emails, recommendations only among coins that meet alert rules
        signals = detect_signals(p1, p2, p3)

        def filter_rows(rows: List[List], label: str) -> List[List]:
            active_syms = signals[label]["BUY"] | signals[label]["SELL"]
            return [r for r in rows if r[0] in active_syms]

        p1_sig = filter_rows(p1, "P1")
        p2_sig = filter_rows(p2, "P2")
        p3_sig = filter_rows(p3, "P3")

        recs = pick_best_trades(p1_sig, p2_sig, p3_sig)
        rec_text = format_recommended_trades(recs) if recs else ""

        body = scan_for_alerts(p1, p2, p3, rec_text)
        if not body:
            return

        if send_email("Crypto Alert: EMA + 24h + intraday", body):
            record_email()
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

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("screen", screen))
    app.add_handler(CommandHandler("diag", diag))
    app.add_handler(CommandHandler("notify_on", notify_on))
    app.add_handler(CommandHandler("notify_off", notify_off))
    app.add_handler(CommandHandler("notify", notify))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    app.add_error_handler(log_err)

    if getattr(app, "job_queue", None):
        app.job_queue.run_repeating(
            alert_job,
            interval=CHECK_INTERVAL_MIN * 60,
            first=10,
        )
    else:
        logging.warning(
            "JobQueue not available. Make sure you installed "
            '"python-telegram-bot[job-queue]" in requirements.txt.'
        )

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
