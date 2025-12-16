#!/usr/bin/env python3
"""
PulseFutures — Telegram crypto futures screener & alert bot (CoinEx via CCXT)

Core goals:
- Clean /screen for mobile (portrait-friendly)
- High-confidence day-trade setups (1H trigger + 15m confirm)
- Simple risk sizing + trade ledger (/risk, /open, /closepnl, /open positions)
- Email alerts with non-repetitive symbols and readable formatting

Important notes:
- FUTURES ONLY (CoinEx swap via CCXT)
- This bot is an assistant, NOT financial advice.
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
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, date
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

# ----------- Email config (Render env) -----------
EMAIL_ENABLED_DEFAULT = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST = os.environ.get("EMAIL_HOST", "")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "465"))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", EMAIL_USER)
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_USER)

# ----------- Scheduling -----------
CHECK_INTERVAL_MIN = 5
EMAIL_MIN_INTERVAL_SEC = 60 * 60  # 60 minutes between emails
LAST_EMAIL_TS: float = 0.0
LAST_ALL_SYMBOLS: Set[str] = set()

# ----------- Trading window (default NY + London, but you can tune) -----------
# We'll keep the same "Melbourne time window guard" style; update as needed.
# User previously asked for NY+London default; here we enforce a Melbourne window
# and show a warning if outside (email must be blocked outside).
EMAIL_WINDOW_START_HHMM = (8, 0)   # 08:00 Melbourne
EMAIL_WINDOW_END_HHMM = (23, 0)    # 23:00 Melbourne

# Sunday emails?
# Your preference: usually NO emails on Sunday reduces noise.
# Set to True if you want Sunday emails.
SEND_EMAIL_ON_SUNDAYS = False

# ----------- Signal / thresholds -----------
CONF_THRESHOLD_EMAIL = 75  # only email setups with Conf >= 75
MAX_SETUPS_PER_EMAIL = 2   # keep emails short; can raise later

# 1H trigger + 15m confirm logic parameters (simple & robust)
TRIGGER_1H_MIN_ABS = 2.0   # abs % move threshold for trigger (1H)
CONFIRM_15M_MIN_ABS = 0.8  # abs % move for confirm (15m)

# SL/TP defaults (tight day-trade)
DEFAULT_SL_PCT = 0.03  # 3%
DEFAULT_TP_R = 2.0     # TP2 at ~2R; TP1 at 1R

# Multi-TP only if Conf >= 75
MULTI_TP_CONF = 75

# ----------- Risk management defaults -----------
DEFAULT_EQUITY = 1000.0
DEFAULT_RISK_PCT = 1.0  # 1% per trade
DEFAULT_DAILY_RISK_PCT = 3.0  # 3% per day max
DEFAULT_OPEN_RISK_PCT = 2.0   # 2% max open risk at once
DEFAULT_MAX_TRADES_PER_DAY = 3

# ----------- Internal caches -----------
LAST_ERROR: Optional[str] = None
NOTIFY_ON: bool = EMAIL_ENABLED_DEFAULT

PCT_CACHE: Dict[Tuple[str, str, str], float] = {}  # (symbol, timeframe, key) -> pct
LATEST_SETUPS: Dict[str, dict] = {}  # base -> setup dict (latest)
LATEST_SETUPS_TS: float = 0.0

# user state (in-memory)
USER_EQUITY: Dict[int, float] = {}
USER_RISK_PCT: Dict[int, float] = {}
USER_DAILY_RISK_PCT: Dict[int, float] = {}
USER_OPEN_RISK_PCT: Dict[int, float] = {}
USER_MAX_TRADES_PER_DAY: Dict[int, int] = {}

# ledger
OPEN_POSITIONS: Dict[int, List[dict]] = {}  # user_id -> list of open positions
DAILY_RISK_USED: Dict[int, Dict[str, float]] = {}  # user_id -> {YYYY-MM-DD: risk_used_usd}
DAILY_TRADES_COUNT: Dict[int, Dict[str, int]] = {}  # user_id -> {YYYY-MM-DD: count}

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

STABLES = {"USD", "USDT", "USDC", "TUSD", "FDUSD", "USDD", "USDE", "DAI", "PYUSD"}


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
    try:
        ex.load_markets()
        return ex.fetch_tickers()
    except Exception as e:
        global LAST_ERROR
        LAST_ERROR = f"{type(e).__name__}: {e}"
        logging.exception("fetch_tickers failed")
        return {}


def usd_notional(mv: Optional[MarketVol]) -> float:
    """Return 24h notional volume in USD (best-effort)."""
    if not mv:
        return 0.0
    if mv.quote in STABLES and mv.quote_vol:
        return mv.quote_vol
    price = mv.vwap if mv.vwap else mv.last
    if not price or not mv.base_vol:
        return 0.0
    return mv.base_vol * price


def pct_change(mv: Optional[MarketVol]) -> float:
    """Prefer ticker.percentage; fallback to open/last."""
    if not mv:
        return 0.0
    if mv.percentage:
        return float(mv.percentage)
    if mv.open:
        return (mv.last - mv.open) / mv.open * 100.0
    return 0.0


def pct_emoji(p: float) -> str:
    val = round(p)
    if val >= 3:
        emo = "🟢"
    elif val <= -3:
        emo = "🔴"
    else:
        emo = "🟡"
    return f"{val:+d}% {emo}"


def bias_letter(x: str) -> str:
    """Map bias to short letter."""
    x = (x or "").upper()
    if x.startswith("LONG"):
        return "L"
    if x.startswith("SHORT"):
        return "S"
    return "N"


def compact_usd(x: float) -> str:
    """Compact notional display: 12M, 1.2B, 950K."""
    try:
        x = float(x)
    except Exception:
        return "0"
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"{x/1_000_000_000:.1f}B".replace(".0B", "B")
    if ax >= 1_000_000:
        return f"{x/1_000_000:.0f}M"
    if ax >= 1_000:
        return f"{x/1_000:.0f}K"
    return f"{x:.0f}"


def fmt_price(x: float) -> str:
    """Adaptive decimals: avoid too many digits."""
    if x == 0 or x is None:
        return "0"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:.2f}"
    if ax >= 100:
        return f"{x:.3f}"
    if ax >= 1:
        return f"{x:.4f}"
    if ax >= 0.1:
        return f"{x:.5f}"
    if ax >= 0.01:
        return f"{x:.6f}"
    return f"{x:.8f}"


def today_key_melbourne() -> str:
    now_mel = datetime.now(ZoneInfo("Australia/Melbourne"))
    return now_mel.date().isoformat()


def melbourne_in_email_window() -> bool:
    """
    Email allowed only within [START..END] Melbourne time.
    Blocks Sunday if SEND_EMAIL_ON_SUNDAYS is False.
    """
    try:
        now_mel = datetime.now(ZoneInfo("Australia/Melbourne"))
        dow = now_mel.weekday()  # Mon=0 ... Sun=6
        if dow == 6 and not SEND_EMAIL_ON_SUNDAYS:
            return False

        sh, sm = EMAIL_WINDOW_START_HHMM
        eh, em = EMAIL_WINDOW_END_HHMM

        start = now_mel.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end = now_mel.replace(hour=eh, minute=em, second=0, microsecond=0)

        # same-day window
        return start <= now_mel <= end
    except Exception:
        logging.exception("Time window check failed; default allow.")
        return True


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


# ================== MARKET / SIGNAL LOGIC (FUTURES ONLY) ==================

def fetch_futures_best() -> Tuple[Dict[str, MarketVol], int]:
    """
    Fetch CoinEx swap tickers and keep best USD-quoted market per BASE.
    Returns: best_fut_by_base, raw_count
    """
    ex = build_exchange("swap")
    tickers = safe_fetch_tickers(ex)
    best: Dict[str, MarketVol] = {}

    for t in tickers.values():
        mv = to_mv(t)
        if not mv:
            continue
        # Futures often not stable quoted; still compute notional
        if mv.base not in best or usd_notional(mv) > usd_notional(best[mv.base]):
            best[mv.base] = mv

    return best, len(tickers)


def fetch_pct(symbol: str, timeframe: str, hours: int) -> float:
    """
    Compute % change over last N hours using 1h candles (for 1H/4H) or 15m candles for 15m confirm.
    Caches results.
    """
    cache_key = (symbol, timeframe, str(hours))
    if cache_key in PCT_CACHE:
        return PCT_CACHE[cache_key]

    try:
        ex = build_exchange("swap")
        ex.load_markets()
        if timeframe == "1h":
            candles = ex.fetch_ohlcv(symbol, timeframe="1h", limit=hours + 1)
        else:
            # 15m confirm: use 15m candles, last 4 = 1 hour; last 2 = 30m etc.
            # We'll treat `hours` as number of 15m blocks here if timeframe == "15m"
            candles = ex.fetch_ohlcv(symbol, timeframe="15m", limit=hours + 1)
        if not candles or len(candles) <= hours:
            PCT_CACHE[cache_key] = 0.0
            return 0.0
        closes = [c[4] for c in candles][- (hours + 1):]
        if not closes or not closes[0]:
            pct = 0.0
        else:
            pct = (closes[-1] - closes[0]) / closes[0] * 100.0
        PCT_CACHE[cache_key] = pct
        return pct
    except Exception:
        logging.exception("fetch_pct failed for %s %s", symbol, timeframe)
        PCT_CACHE[cache_key] = 0.0
        return 0.0


def determine_bias(p24: float, p4: float) -> str:
    """
    Simple bias:
    - LONG if 24H>+5 and 4H>=0
    - SHORT if 24H<-5 and 4H<=0
    - else NEUTRAL
    """
    if p24 > 5 and p4 >= 0:
        return "LONG"
    if p24 < -5 and p4 <= 0:
        return "SHORT"
    return "NEUTRAL"


def compute_confidence(p24: float, p4: float, p1: float, p15: float, fut_usd: float, bias: str) -> int:
    """
    Produce a 0..100 confidence score.
    """
    score = 50.0

    # momentum alignment
    score += min(abs(p24), 20) * 0.6
    score += min(abs(p4), 15) * 0.8
    score += min(abs(p1), 8) * 1.0
    score += min(abs(p15), 4) * 2.0

    # liquidity bonus
    if fut_usd >= 15_000_000:
        score += 8
    elif fut_usd >= 6_000_000:
        score += 5
    elif fut_usd >= 2_000_000:
        score += 2

    # bias penalty if unclear
    if bias == "NEUTRAL":
        score -= 6

    # clamp
    score = max(0, min(100, score))
    return int(round(score))


def make_trade_plan(entry: float, side: str, sl_pct: float = DEFAULT_SL_PCT) -> Tuple[float, float, float, float, float]:
    """
    Returns: sl, tp1, tp2, tp1_pct, tp2_pct (relative)
    TP1 at 1R, TP2 at DEFAULT_TP_R * R.
    """
    if entry <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    if side == "BUY":
        sl = entry * (1 - sl_pct)
        r = entry - sl
        tp1 = entry + r
        tp2 = entry + (DEFAULT_TP_R * r)
    else:
        sl = entry * (1 + sl_pct)
        r = sl - entry
        tp1 = entry - r
        tp2 = entry - (DEFAULT_TP_R * r)

    tp1_pct = (tp1 - entry) / entry * 100.0
    tp2_pct = (tp2 - entry) / entry * 100.0
    return sl, tp1, tp2, tp1_pct, tp2_pct


def build_market_leaders(best_fut: Dict[str, MarketVol], top_n: int = 10) -> List[dict]:
    """
    Market leaders: top N by futures notional volume.
    Each row includes symbol base, fut_usd, 24H, 4H, 1H, bias letter.
    """
    rows: List[dict] = []
    for base, mv in best_fut.items():
        fut_usd = usd_notional(mv)
        p24 = pct_change(mv)
        p4 = fetch_pct(mv.symbol, "1h", 4)
        p1 = fetch_pct(mv.symbol, "1h", 1)
        bias = determine_bias(p24, p4)
        rows.append(
            {
                "base": base,
                "symbol": mv.symbol,
                "fut_usd": fut_usd,
                "p24": p24,
                "p4": p4,
                "p1": p1,
                "bias": bias,
                "last": mv.last or 0.0,
            }
        )
    rows.sort(key=lambda r: r["fut_usd"], reverse=True)
    return rows[:top_n]


def build_strong_movers(best_fut: Dict[str, MarketVol], top_n: int = 10) -> Tuple[List[dict], List[dict]]:
    """
    Strong movers (24H): top + and top - by 24h % among sufficiently liquid markets.
    """
    rows: List[dict] = []
    for base, mv in best_fut.items():
        fut_usd = usd_notional(mv)
        if fut_usd < 1_000_000:
            continue
        p24 = pct_change(mv)
        p4 = fetch_pct(mv.symbol, "1h", 4)
        p1 = fetch_pct(mv.symbol, "1h", 1)
        bias = determine_bias(p24, p4)
        rows.append(
            {
                "base": base,
                "symbol": mv.symbol,
                "fut_usd": fut_usd,
                "p24": p24,
                "p4": p4,
                "p1": p1,
                "bias": bias,
                "last": mv.last or 0.0,
            }
        )

    gainers = sorted(rows, key=lambda r: r["p24"], reverse=True)[:top_n]
    losers = sorted(rows, key=lambda r: r["p24"])[:top_n]
    return gainers, losers


def generate_setups(best_fut: Dict[str, MarketVol]) -> List[dict]:
    """
    Produce candidate setups based on:
    - 1H trigger (abs move >= TRIGGER_1H_MIN_ABS)
    - 15m confirm (abs move >= CONFIRM_15M_MIN_ABS)
    - direction derived from 24H/4H bias
    - confidence computed
    """
    setups: List[dict] = []

    for base, mv in best_fut.items():
        fut_usd = usd_notional(mv)
        if fut_usd < 1_500_000:
            continue

        p24 = pct_change(mv)
        p4 = fetch_pct(mv.symbol, "1h", 4)
        p1 = fetch_pct(mv.symbol, "1h", 1)
        # 15m confirm: use last 4 candles = 1 hour? We'll use last 2 candles = 30m as "confirm"
        p15 = fetch_pct(mv.symbol, "15m", 2)

        bias = determine_bias(p24, p4)

        # Trigger + confirm
        if abs(p1) < TRIGGER_1H_MIN_ABS:
            continue
        if abs(p15) < CONFIRM_15M_MIN_ABS:
            continue

        # side by bias + short-term direction
        side = None
        if bias == "LONG" and p15 > 0:
            side = "BUY"
        elif bias == "SHORT" and p15 < 0:
            side = "SELL"
        else:
            # if neutral, allow direction by p15 but lower confidence
            side = "BUY" if p15 > 0 else "SELL"

        conf = compute_confidence(p24, p4, p1, p15, fut_usd, bias)

        entry = mv.last or 0.0
        sl, tp1, tp2, tp1pct, tp2pct = make_trade_plan(entry, side, DEFAULT_SL_PCT)

        setups.append(
            {
                "base": base,
                "symbol": mv.symbol,
                "side": side,
                "conf": conf,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp1pct": tp1pct,
                "tp2pct": tp2pct,
                "fut_usd": fut_usd,
                "p24": p24,
                "p4": p4,
                "p1": p1,
                "bias": bias,
                "p15": p15,
                "ts": time.time(),
            }
        )

    # sort by confidence then liquidity
    setups.sort(key=lambda s: (s["conf"], s["fut_usd"]), reverse=True)
    return setups


def refresh_latest_setups(setups: List[dict]):
    """Cache latest setups by base symbol."""
    global LATEST_SETUPS, LATEST_SETUPS_TS
    LATEST_SETUPS = {s["base"]: s for s in setups}
    LATEST_SETUPS_TS = time.time()


# ================== FORMATTING ==================

def fmt_leaders_table(rows: List[dict], title: str) -> str:
    if not rows:
        return f"*{title}*: _None_\n"
    pretty = []
    for r in rows:
        pretty.append([
            f"**{r['base']}**",
            compact_usd(r["fut_usd"]),
            pct_emoji(r["p24"]),
            pct_emoji(r["p4"]),
            pct_emoji(r["p1"]),
            bias_letter(r["bias"]),
        ])
    return (
        f"*{title}*\n"
        "```\n"
        + tabulate(pretty, headers=["SYM", "F", "24H", "4H", "1H", "B"], tablefmt="github")
        + "\n```\n"
    )


def fmt_setup_email(setup: dict, idx: int) -> str:
    base = setup["base"]
    side = setup["side"]
    conf = setup["conf"]
    entry = setup["entry"]
    sl = setup["sl"]
    tp1 = setup["tp1"]
    tp2 = setup["tp2"]
    fut_usd = setup["fut_usd"]
    p24, p4, p1 = setup["p24"], setup["p4"], setup["p1"]
    bias = setup["bias"]

    lines = []
    lines.append(f"Setup #{idx}: {side} [{base}] — Confidence {conf}/100")
    lines.append(f"Entry: {fmt_price(entry)}")
    lines.append(f"SL: {fmt_price(sl)}")

    # Multi-TP details (always shown; percent allocations if Conf >= threshold)
    if conf >= MULTI_TP_CONF:
        lines.append("TP Plan:")
        lines.append(f"- TP1 (40%): {fmt_price(tp1)}")
        lines.append(f"- TP2 (40%): {fmt_price(tp2)}")
        lines.append("- Runner (20%): trail EMA20 (1H) or last 15m swing (direction-based)")
    else:
        lines.append("TP Plan:")
        lines.append(f"- TP1: {fmt_price(tp1)}")
        lines.append(f"- TP2: {fmt_price(tp2)}")

    lines.append(f"Snapshot: F~{compact_usd(fut_usd)} | 24H {pct_emoji(p24)} | 4H {pct_emoji(p4)} | 1H {pct_emoji(p1)} | {bias}")
    return "\n".join(lines)


def fmt_setup_telegram(setup: dict, idx: int) -> str:
    base = setup["base"]
    side = setup["side"]
    conf = setup["conf"]
    entry = setup["entry"]
    sl = setup["sl"]
    tp1 = setup["tp1"]
    tp2 = setup["tp2"]
    fut_usd = setup["fut_usd"]
    p24, p4, p1 = setup["p24"], setup["p4"], setup["p1"]
    bias = setup["bias"]

    lines = []
    lines.append(f"*Setup #{idx}* — {side} **{base}** — *Confidence {conf}/100*")
    lines.append(f"Entry {fmt_price(entry)} | SL {fmt_price(sl)} | TP {fmt_price(tp2)}")
    lines.append(f"Market Snapshot: F~{compact_usd(fut_usd)} | 24H {pct_emoji(p24)} | 4H {pct_emoji(p4)} | 1H {pct_emoji(p1)} | {bias}")
    if conf >= MULTI_TP_CONF:
        lines.append("Multi-TP: TP1 40% | TP2 40% | Runner 20%")
        lines.append(f"TP1 {fmt_price(tp1)} | TP2 {fmt_price(tp2)}")
        lines.append("Runner: trail EMA20 (1H) or last 15m swing (direction-based)")
    return "\n".join(lines)


# ================== RISK MGMT ==================

def get_user_setting(d: Dict[int, float], user_id: int, default: float) -> float:
    return float(d.get(user_id, default))


def get_open_risk_usd(user_id: int) -> float:
    """Sum of risk USD of currently open positions."""
    total = 0.0
    for p in OPEN_POSITIONS.get(user_id, []):
        total += float(p.get("risk_usd", 0.0))
    return total


def get_daily_risk_used_usd(user_id: int, day_key: Optional[str] = None) -> float:
    """Sum of risk USD of trades OPENED today (does not decrease when positions close)."""
    day_key = day_key or today_key_melbourne()
    return float(DAILY_RISK_USED.get(user_id, {}).get(day_key, 0.0))


def inc_daily_risk_used(user_id: int, risk_usd: float, day_key: Optional[str] = None):
    day_key = day_key or today_key_melbourne()
    DAILY_RISK_USED.setdefault(user_id, {})
    DAILY_RISK_USED[user_id][day_key] = get_daily_risk_used_usd(user_id, day_key) + float(risk_usd)


def inc_daily_trade_count(user_id: int, day_key: Optional[str] = None):
    day_key = day_key or today_key_melbourne()
    DAILY_TRADES_COUNT.setdefault(user_id, {})
    DAILY_TRADES_COUNT[user_id][day_key] = int(DAILY_TRADES_COUNT[user_id].get(day_key, 0)) + 1


def get_daily_trade_count(user_id: int, day_key: Optional[str] = None) -> int:
    day_key = day_key or today_key_melbourne()
    return int(DAILY_TRADES_COUNT.get(user_id, {}).get(day_key, 0))


def risk_limits_ok(user_id: int, new_risk: float) -> Tuple[bool, str]:
    eq = get_user_setting(USER_EQUITY, user_id, DEFAULT_EQUITY)
    daily_risk_pct = get_user_setting(USER_DAILY_RISK_PCT, user_id, DEFAULT_DAILY_RISK_PCT)
    open_risk_pct = get_user_setting(USER_OPEN_RISK_PCT, user_id, DEFAULT_OPEN_RISK_PCT)
    max_trades = int(USER_MAX_TRADES_PER_DAY.get(user_id, DEFAULT_MAX_TRADES_PER_DAY))

    day_key = today_key_melbourne()

    daily_used = get_daily_risk_used_usd(user_id, day_key)
    open_used = get_open_risk_usd(user_id)
    trades_today = get_daily_trade_count(user_id, day_key)

    daily_cap = eq * (daily_risk_pct / 100.0)
    open_cap = eq * (open_risk_pct / 100.0)

    if trades_today >= max_trades:
        return False, f"Max trades per day reached ({max_trades})."
    if daily_used + new_risk > daily_cap:
        return False, f"Daily risk cap exceeded. Used ${daily_used:.2f} / Cap ${daily_cap:.2f}."
    if open_used + new_risk > open_cap:
        return False, f"Open risk cap exceeded. Open ${open_used:.2f} / Cap ${open_cap:.2f}."
    return True, "OK"


def calc_position_size(entry: float, sl: float, risk_usd: float) -> Tuple[float, float]:
    """
    Return (qty, notional_usd) using risk = abs(entry-sl)*qty
    """
    dist = abs(entry - sl)
    if entry <= 0 or dist <= 0 or risk_usd <= 0:
        return 0.0, 0.0
    qty = risk_usd / dist
    notional = qty * entry
    return qty, notional


# ================== TELEGRAM COMMANDS ==================

HELP_TEXT = """\
*PulseFutures — Quick Guide (English)*

*What this bot does*
- Scans CoinEx FUTURES markets and posts a clean market view (/screen).
- Sends high-confidence *day-trade setups* via email (if enabled).
- Helps you size positions using risk-based sizing (/risk), track open positions (/open, /openpositions), and close with PnL (/closepnl).

*Key commands*
- /start — show basics
- /help — show this guide
- /screen — Market Leaders + Strong Movers + top setups (compact for mobile)
- /risk <SYMBOL> <RISK_USD> — calculates position size. If SYMBOL matches a bot setup, it auto-uses that setup. If not, it uses a default SL% and warns it’s manual.
- /open <SYMBOL> <RISK_USD> — records a new open position (auto-detects bot setup if available). This is the fastest workflow after you receive an email.
- /openpositions — list your open positions (with spacing & numbering)
- /closepnl <SYMBOL> <PNL_USD> — closes the latest open position for SYMBOL and updates equity (equity += pnl)
- /equity <USD> — set your account equity
- /limits <max_trades_per_day> <daily_risk_pct> <open_risk_pct> — set daily/open risk controls
- /notify_on /notify_off /notify — email alert toggle/status

*Daily Risk Used vs Open Risk (Important)*
- *Open Risk* = sum of risk (USD) of *currently open* positions. If all open trades hit SL, this is your total loss.
- *Daily Risk Used* = sum of risk (USD) of trades *opened today*. It does NOT decrease when you close positions. This prevents overtrading.

*TP1 / TP2 / Runner (Multi-TP)*
- If Confidence >= 75, the bot uses a Multi-TP plan:
  - TP1 (40%): first target (locks partial profit)
  - TP2 (40%): second target
  - Runner (20%): optional final portion, trailed using EMA20 (1H) or last 15m swing
- If Confidence < 75, TP targets are still provided but without the 40/40/20 plan emphasis.

*Notes & limits*
- Futures ONLY. This is not financial advice.
"""

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "PulseFutures is running.\n"
        "Try: /screen, /help\n"
        "Fast workflow after email: /risk SYMBOL RISKUSD then /open SYMBOL RISKUSD",
        parse_mode=ParseMode.MARKDOWN
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)


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


async def equity_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    if not args:
        eq = get_user_setting(USER_EQUITY, user_id, DEFAULT_EQUITY)
        await update.message.reply_text(f"Equity: ${eq:.2f}")
        return
    try:
        eq = float(args[0])
        USER_EQUITY[user_id] = max(0.0, eq)
        await update.message.reply_text(f"Equity set to ${USER_EQUITY[user_id]:.2f}")
    except Exception:
        await update.message.reply_text("Usage: /equity 1000")


async def limits_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    if len(args) != 3:
        mt = int(USER_MAX_TRADES_PER_DAY.get(user_id, DEFAULT_MAX_TRADES_PER_DAY))
        dr = get_user_setting(USER_DAILY_RISK_PCT, user_id, DEFAULT_DAILY_RISK_PCT)
        orp = get_user_setting(USER_OPEN_RISK_PCT, user_id, DEFAULT_OPEN_RISK_PCT)
        await update.message.reply_text(
            f"Limits:\n- max_trades_per_day: {mt}\n- daily_risk_pct: {dr}%\n- open_risk_pct: {orp}%\n"
            "Set: /limits 3 3 2"
        )
        return
    try:
        USER_MAX_TRADES_PER_DAY[user_id] = int(args[0])
        USER_DAILY_RISK_PCT[user_id] = float(args[1])
        USER_OPEN_RISK_PCT[user_id] = float(args[2])
        await update.message.reply_text(
            f"Updated limits:\n- max_trades_per_day: {USER_MAX_TRADES_PER_DAY[user_id]}\n"
            f"- daily_risk_pct: {USER_DAILY_RISK_PCT[user_id]}%\n"
            f"- open_risk_pct: {USER_OPEN_RISK_PCT[user_id]}%"
        )
    except Exception:
        await update.message.reply_text("Usage: /limits 3 3 2")


def normalize_symbol_token(txt: str) -> str:
    token = re.sub(r"[^A-Za-z$]", "", txt or "").upper().lstrip("$")
    return token


async def screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        PCT_CACHE.clear()
        best_fut, _raw = await asyncio.to_thread(fetch_futures_best)

        leaders = await asyncio.to_thread(build_market_leaders, best_fut, 10)
        gainers, losers = await asyncio.to_thread(build_strong_movers, best_fut, 10)

        setups = await asyncio.to_thread(generate_setups, best_fut)
        refresh_latest_setups(setups)

        msg_parts = []
        msg_parts.append(fmt_leaders_table(leaders, "Market Leaders (Top 10 by futures volume)"))

        # space after leaders
        msg_parts.append("\n")

        msg_parts.append(fmt_leaders_table(gainers, "Strong Movers +24H (Top 10)"))
        # space after + movers
        msg_parts.append("\n")
        msg_parts.append(fmt_leaders_table(losers, "Strong Movers -24H (Top 10)"))

        # space after - movers
        msg_parts.append("\n")

        # top setups (keep short)
        top_setups = [s for s in setups if s["conf"] >= CONF_THRESHOLD_EMAIL][:3]
        if top_setups:
            msg_parts.append("*Top Setups (Now)*\n")
            for i, s in enumerate(top_setups, 1):
                msg_parts.append(fmt_setup_telegram(s, i))
                msg_parts.append("\n---\n")
        else:
            msg_parts.append("*Top Setups (Now)*\n_No strong recommendation right now._\n")

        # remove confusing tickers count line (requested)
        await update.message.reply_text("\n".join(msg_parts), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logging.exception("screen error")
        await update.message.reply_text(f"Error: {e}")


async def risk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /risk SYMBOL RISKUSD
    - If SYMBOL is in latest setups cache: use its entry/sl.
    - Else: manual sizing using current last price (if available) and default SL%.
    """
    user_id = update.effective_user.id
    args = context.args
    if len(args) != 2:
        await update.message.reply_text("Usage: /risk SYMBOL RISKUSD\nExample: /risk SUI 20")
        return

    sym = normalize_symbol_token(args[0])
    try:
        risk_usd = float(args[1])
    except Exception:
        await update.message.reply_text("RiskUSD must be a number. Example: /risk SUI 20")
        return

    ok, why = risk_limits_ok(user_id, risk_usd)
    if not ok:
        await update.message.reply_text(f"❌ {why}")
        return

    # Determine if it's a bot setup
    setup = LATEST_SETUPS.get(sym)

    if setup:
        entry = float(setup["entry"])
        sl = float(setup["sl"])
        side = setup["side"]
        conf = setup["conf"]
        qty, notional = calc_position_size(entry, sl, risk_usd)
        await update.message.reply_text(
            f"✅ Bot Setup detected for **{sym}** ({side}, Conf {conf}/100)\n"
            f"Entry: {fmt_price(entry)} | SL: {fmt_price(sl)}\n"
            f"Risk: ${risk_usd:.2f}\n"
            f"Qty: {qty:.6g}\n"
            f"Notional: ~${notional:.2f}\n\n"
            f"To record this trade: /open {sym} {risk_usd}",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    # Manual sizing:
    # We use current market last price if possible; otherwise explain.
    try:
        best_fut, _ = await asyncio.to_thread(fetch_futures_best)
        mv = best_fut.get(sym)
    except Exception:
        mv = None

    if not mv:
        await update.message.reply_text(
            f"⚠️ **{sym}** not found in current futures tickers.\n"
            f"I can still help with risk sizing if you provide an approximate entry.\n"
            f"(For now, try a symbol that exists or use /screen first.)",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    entry = float(mv.last or 0.0)
    # default SL distance for sizing
    sl = entry * (1 - DEFAULT_SL_PCT)
    qty, notional = calc_position_size(entry, sl, risk_usd)

    await update.message.reply_text(
        f"⚠️ Manual sizing for **{sym}** (NOT a bot setup)\n"
        f"Using default SL distance: {DEFAULT_SL_PCT*100:.1f}%\n"
        f"Entry (last): {fmt_price(entry)} | SL (default): {fmt_price(sl)}\n"
        f"Risk: ${risk_usd:.2f}\n"
        f"Qty: {qty:.6g}\n"
        f"Notional: ~${notional:.2f}\n\n"
        f"To record this trade: /open {sym} {risk_usd}\n"
        f"Note: You are responsible for this manual trade.",
        parse_mode=ParseMode.MARKDOWN
    )


async def open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /open SYMBOL RISKUSD
    Records a new open position.
    If symbol is in latest setups cache, uses that setup's side/entry/sl/tp.
    If not, records as manual using current last price + default SL%.
    """
    user_id = update.effective_user.id
    args = context.args
    if len(args) != 2:
        await update.message.reply_text("Usage: /open SYMBOL RISKUSD\nExample: /open SUI 20")
        return

    sym = normalize_symbol_token(args[0])
    try:
        risk_usd = float(args[1])
    except Exception:
        await update.message.reply_text("RiskUSD must be a number. Example: /open SUI 20")
        return

    ok, why = risk_limits_ok(user_id, risk_usd)
    if not ok:
        await update.message.reply_text(f"❌ {why}")
        return

    day_key = today_key_melbourne()

    setup = LATEST_SETUPS.get(sym)
    position = {
        "symbol": sym,
        "opened_at": datetime.now(ZoneInfo("Australia/Melbourne")).isoformat(timespec="seconds"),
        "risk_usd": risk_usd,
        "is_bot_setup": bool(setup),
        "day_key": day_key,
    }

    if setup:
        position.update({
            "side": setup["side"],
            "entry": float(setup["entry"]),
            "sl": float(setup["sl"]),
            "tp1": float(setup["tp1"]),
            "tp2": float(setup["tp2"]),
            "conf": int(setup["conf"]),
        })
    else:
        # manual: use current last price (if possible)
        best_fut, _ = await asyncio.to_thread(fetch_futures_best)
        mv = best_fut.get(sym)
        if not mv or not mv.last:
            await update.message.reply_text(
                f"⚠️ Could not fetch last price for **{sym}**. Try again later or use /screen first.",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        entry = float(mv.last)
        sl = entry * (1 - DEFAULT_SL_PCT)
        sl, tp1, tp2, _, _ = make_trade_plan(entry, "BUY", DEFAULT_SL_PCT)  # direction-agnostic sizing, but keep numbers
        position.update({
            "side": "MANUAL",
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "conf": None,
        })

    OPEN_POSITIONS.setdefault(user_id, [])
    OPEN_POSITIONS[user_id].append(position)

    # Update daily counters
    inc_daily_risk_used(user_id, risk_usd, day_key)
    inc_daily_trade_count(user_id, day_key)

    open_risk = get_open_risk_usd(user_id)
    daily_used = get_daily_risk_used_usd(user_id, day_key)

    await update.message.reply_text(
        f"✅ Opened position recorded: **{sym}**\n"
        f"Risk: ${risk_usd:.2f}\n"
        f"Open Risk (now): ${open_risk:.2f}\n"
        f"Daily Risk Used (today): ${daily_used:.2f}\n\n"
        f"Close it later with: /closepnl {sym} +12.5 (or -7.2)",
        parse_mode=ParseMode.MARKDOWN
    )


async def openpositions_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    pos = OPEN_POSITIONS.get(user_id, [])
    if not pos:
        await update.message.reply_text("No open positions.")
        return

    lines = ["*Open Positions*"]
    for i, p in enumerate(pos, 1):
        sym = p["symbol"]
        side = p.get("side", "?")
        risk_usd = float(p.get("risk_usd", 0.0))
        entry = p.get("entry", None)
        sl = p.get("sl", None)
        is_bot = p.get("is_bot_setup", False)
        conf = p.get("conf", None)

        lines.append(f"\n#{i} — **{sym}** — {side} — Risk ${risk_usd:.2f}")
        if entry and sl:
            lines.append(f"Entry {fmt_price(float(entry))} | SL {fmt_price(float(sl))}")
        if is_bot and conf is not None:
            lines.append(f"Bot setup: YES (Conf {conf}/100)")
        elif not is_bot:
            lines.append("Bot setup: NO (manual)")
        lines.append("—")  # simple separator

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def closepnl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /closepnl SYMBOL PNL_USD
    Closes the latest open position matching SYMBOL and updates equity.
    """
    user_id = update.effective_user.id
    args = context.args
    if len(args) != 2:
        await update.message.reply_text("Usage: /closepnl SYMBOL PNL_USD\nExample: /closepnl SUI +25.4")
        return

    sym = normalize_symbol_token(args[0])
    try:
        pnl = float(args[1])
    except Exception:
        await update.message.reply_text("PNL must be a number. Example: /closepnl SUI +25.4")
        return

    pos = OPEN_POSITIONS.get(user_id, [])
    idx = None
    for i in range(len(pos) - 1, -1, -1):
        if pos[i].get("symbol") == sym:
            idx = i
            break

    if idx is None:
        await update.message.reply_text(f"No open position found for {sym}.")
        return

    closed = pos.pop(idx)

    # Update equity
    eq = get_user_setting(USER_EQUITY, user_id, DEFAULT_EQUITY)
    eq_new = eq + pnl
    USER_EQUITY[user_id] = max(0.0, eq_new)

    open_risk = get_open_risk_usd(user_id)
    daily_used = get_daily_risk_used_usd(user_id, today_key_melbourne())

    await update.message.reply_text(
        f"✅ Closed **{sym}** | PnL: ${pnl:.2f}\n"
        f"Equity updated: ${USER_EQUITY[user_id]:.2f}\n"
        f"Open Risk (now): ${open_risk:.2f}\n"
        f"Daily Risk Used (today): ${daily_used:.2f}",
        parse_mode=ParseMode.MARKDOWN
    )


# ================== EMAIL ALERT JOB ==================

async def alert_job(context: ContextTypes.DEFAULT_TYPE):
    global LAST_EMAIL_TS, LAST_ALL_SYMBOLS
    try:
        if not NOTIFY_ON:
            return
        if not email_config_ok():
            return
        if not melbourne_in_email_window():
            return

        now = time.time()
        if now - LAST_EMAIL_TS < EMAIL_MIN_INTERVAL_SEC:
            return

        PCT_CACHE.clear()
        best_fut, _ = await asyncio.to_thread(fetch_futures_best)

        setups = await asyncio.to_thread(generate_setups, best_fut)
        refresh_latest_setups(setups)

        # Keep only high-confidence setups for email
        setups = [s for s in setups if s["conf"] >= CONF_THRESHOLD_EMAIL]
        if not setups:
            return

        setups = setups[:MAX_SETUPS_PER_EMAIL]

        current_symbols = {s["base"] for s in setups}
        if current_symbols == LAST_ALL_SYMBOLS:
            return

        body_lines = []
        for i, s in enumerate(setups, 1):
            body_lines.append(fmt_setup_email(s, i))
            body_lines.append("")  # blank line between setups

        subject = "PulseFutures Alert: Setups"

        if send_email(subject, "\n".join(body_lines).strip()):
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

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("screen", screen))

    app.add_handler(CommandHandler("notify_on", notify_on))
    app.add_handler(CommandHandler("notify_off", notify_off))
    app.add_handler(CommandHandler("notify", notify))

    app.add_handler(CommandHandler("equity", equity_cmd))
    app.add_handler(CommandHandler("limits", limits_cmd))
    app.add_handler(CommandHandler("risk", risk_cmd))
    app.add_handler(CommandHandler("open", open_cmd))
    app.add_handler(CommandHandler("openpositions", openpositions_cmd))
    app.add_handler(CommandHandler("closepnl", closepnl_cmd))

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
