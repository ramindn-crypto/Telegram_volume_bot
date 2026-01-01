#!/usr/bin/env python3
"""
PulseFutures — Bybit Futures (Swap) Screener + Signals Email + Risk Manager + Trade Journal (Telegram)
"""

import asyncio

import os
import sys
import logging

logger = logging.getLogger(__name__)

def _render_single_instance_guard() -> None:
    """
    Prevent overlapping instances (Render redeploy overlap protection).
    """
    instance_id = os.environ.get("RENDER_INSTANCE_ID")
    if instance_id is not None and instance_id != "0":
        logger.warning(
            "Secondary Render instance detected (RENDER_INSTANCE_ID=%s) — exiting",
            instance_id,
        )
        raise SystemExit(0)

import re
import ssl
import smtplib
import sqlite3
import time
import json
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import Counter, defaultdict

import ccxt
from tabulate import tabulate
from telegram import Update
import asyncio

async def _pre_polling_cleanup(application) -> None:
    # Remove any leftover webhook to avoid polling conflicts
    await application.bot.delete_webhook(drop_pending_updates=True)

    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# =========================================================
# CONFIG
# =========================================================
EXCHANGE_ID = "bybit"
TOKEN = os.environ.get("TELEGRAM_TOKEN")

DEFAULT_TYPE = "swap"  # bybit futures
DB_PATH = os.environ.get("DB_PATH", "pulsefutures.db")

CHECK_INTERVAL_MIN = int(os.environ.get("CHECK_INTERVAL_MIN", "5"))

# -------------------------
# ✅ Admin / Visibility (anti-copy)
# -------------------------
# Put your Telegram user id(s) here, comma-separated (example: "123,456")
ADMIN_USER_IDS = set(
    int(x.strip()) for x in os.environ.get("ADMIN_USER_IDS", "").split(",") if x.strip().isdigit()
)

# IMPORTANT CHANGE:
# System sends signals only. It does NOT show reject reasons to public users.
# Admin can still see internals if you set PUBLIC_DIAGNOSTICS_MODE=admin
PUBLIC_DIAGNOSTICS_MODE = os.environ.get("PUBLIC_DIAGNOSTICS_MODE", "off").strip().lower()
# values:
# - "off"    => hide reject diagnostics completely for non-admin
# - "admin"  => only admins can see diagnostics (recommended)
# Admin always sees full if needed.

# -------------------------
# Screen output sizes
# -------------------------
LEADERS_N = 10


# -------------------------
# Screen output sizes
# -------------------------
LEADERS_N = 10

# ✅ More setups on /screen (UX), while email stays strict
SETUPS_N = 6
EMAIL_SETUPS_N = 3

# ✅ /screen scan breadth + loosened trigger only for screen (NOT email)
SCREEN_UNIVERSE_N = 70          # was effectively 35 (inside pick_setups)
SCREEN_TRIGGER_LOOSEN = 0.85    # 15% easier trigger on /screen only
SCREEN_WAITING_NEAR_PCT = 0.75  # near-miss threshold for "Waiting for Trigger"
SCREEN_WAITING_N = 10



# Directional Leaders/Losers thresholds
MOVER_VOL_USD_MIN = 5_000_000
MOVER_UP_24H_MIN = 10.0
MOVER_DN_24H_MAX = -10.0

# =========================================================
# ✅ ENGINES
# =========================================================
ENGINE_A_PULLBACK_ENABLED = True     # pullback / mean-reversion near adaptive EMA
ENGINE_B_MOMENTUM_ENABLED = True     # pump / expansion

# =========================================================
# ✅ 1H MOMENTUM INTENSITY (SESSION-DYNAMIC)
# =========================================================
# Base floor (still used), but we now scale it per session:
TRIGGER_1H_ABS_MIN_BASE = 1.2        # global floor
CONFIRM_15M_ABS_MIN = 0.45
ALIGN_4H_MIN = 0.0

# "EARLY" filler (email only)
EARLY_1H_ABS_MIN = 2.8
EARLY_CONF_PENALTY = 6
EARLY_EMAIL_EXTRA_CONF = 4
EARLY_EMAIL_MAX_FILL = 1

TREND_24H_TOL = 0.5

# ✅ Session-based 1H strictness:
# ASIA = tightest, LON = medium, NY = loosest
SESSION_1H_BASE_MULT = {
    "NY": 0.85,
    "LON": 1.00,
    "ASIA": 1.15,
}


# =========================================================
# ✅ ENGINE B (MOMENTUM / EXPANSION) SETTINGS (for pumps)
# =========================================================
MOMENTUM_MIN_CH1 = 1.8               # pump gate for 1H (easier than before)
MOMENTUM_MIN_24H = 10.0              # must be moving
MOMENTUM_VOL_MULT = 1.2              # volume spike vs mover min
MOMENTUM_ATR_BODY_MULT = 0.95        # expansion vs ATR% (easier)
MOMENTUM_MAX_ADAPTIVE_EMA_DIST = 7.5 # allow being far from EMA (pumps)

# Higher TP behavior for Engine B (pumps)
ENGINE_B_TP_CAP_BONUS_PCT = 4.0      # adds to TP cap %
ENGINE_B_RR_BONUS = 0.35             # adds to RR target (TP3)

# =========================================================
# RISK DEFAULTS
# =========================================================
DEFAULT_EQUITY = 0.0
DEFAULT_RISK_MODE = "PCT"
DEFAULT_RISK_VALUE = 1.5
DEFAULT_DAILY_CAP_MODE = "PCT"
DEFAULT_DAILY_CAP_VALUE = 5.0
DEFAULT_MAX_TRADES_DAY = 5
DEFAULT_MIN_EMAIL_GAP_MIN = 60
DEFAULT_MAX_EMAILS_PER_SESSION = 4
DEFAULT_MAX_EMAILS_PER_DAY = 4

DEFAULT_MAX_RISK_PCT_PER_TRADE = 2.0
WARN_RISK_PCT_PER_TRADE = 2.0

SYMBOL_COOLDOWN_HOURS = 18

# Multi-TP
ATR_PERIOD = 14
ATR_MIN_PCT = 1.0
ATR_MAX_PCT = 8.0
MULTI_TP_MIN_CONF = 78
TP_ALLOCS = (40, 40, 20)

TP_R_MULTS_DEFAULT = (1.0, 1.7, 2.4)

# Dynamic TP cap: normal vs hot coins
TP_MAX_PCT_NORMAL = 12.0
TP_MAX_PCT_HOT = 15.0
HOT_VOL_USD = 50_000_000
HOT_CH24_ABS = 15.0

# =========================================================
# ✅ ADAPTIVE EMA SUPPORT (global)
# =========================================================
# Minimum/maximum adaptive EMA period bounds
ADAPTIVE_EMA_MIN = 8
ADAPTIVE_EMA_MAX = 21

# Proximity threshold (derived from ATR%), clamped
EMA_SUPPORT_MAX_DIST_ATR_MULT = 0.7
EMA_SUPPORT_MAX_DIST_PCT_MIN = 0.25
EMA_SUPPORT_MAX_DIST_PCT_MAX = 1.8

# Sharp 1H move gating
SHARP_1H_MOVE_PCT = 20.0




# Email
EMAIL_ENABLED = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST = os.environ.get("EMAIL_HOST", "")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "465"))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", EMAIL_USER)
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_USER)

WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "").strip()
TELEGRAM_BOT_URL = os.environ.get("TELEGRAM_BOT_URL", "https://t.me/PulseFuturesBot").strip()

# Caching for speed
TICKERS_TTL_SEC = 45
OHLCV_TTL_SEC = 60

# =========================================================
# DEBUG / REJECT REASONS (INTERNAL ONLY)
# =========================================================
# Keep internal reject stats for you, but DO NOT show to public
DEBUG_REJECTS = os.environ.get("DEBUG_REJECTS", "false").lower() == "true"
REJECT_TOP_N = int(os.environ.get("REJECT_TOP_N", "12"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pulsefutures")

STABLES = {"USDT", "USDC", "USD", "TUSD", "FDUSD", "DAI", "PYUSD"}

HDR = "━━━━━━━━━━━━━━━━━━━━"
SEP = "────────────────────"

ALERT_LOCK = asyncio.Lock()






# Sessions defined in UTC windows
SESSIONS_UTC = {
    "ASIA": {"start": "00:00", "end": "06:00"},
    "LON":  {"start": "07:00", "end": "12:00"},
    "NY":   {"start": "13:00", "end": "20:00"},
}

SESSION_PRIORITY = ["NY", "LON", "ASIA"]

SESSION_MIN_CONF = {
    "NY": 72,
    "LON": 78,
    "ASIA": 82,
}

SESSION_MIN_RR_TP3 = {
    "NY": 1.8,
    "LON": 2.0,
    "ASIA": 2.2,
}

SESSION_EMA_PROX_MULT = {
    "NY": 1.20,
    "LON": 1.00,
    "ASIA": 0.85,
}

SESSION_EMA_REACTION_LOOKBACK = {
    "NY": 9,
    "LON": 7,
    "ASIA": 6,
}

# ✅ 1H trigger loosened per session (overall easier)
SESSION_TRIGGER_ATR_MULT = {
    "NY": 0.65,
    "LON": 0.85,
    "ASIA": 1.00,
}

def session_knobs(session_name: str) -> dict:
    s = (session_name or "LON").upper()
    if s not in SESSIONS_UTC:
        s = "LON"
    return {
        "name": s,
        "ema_prox_mult": float(SESSION_EMA_PROX_MULT.get(s, 1.0)),
        "ema_reaction_lookback": int(SESSION_EMA_REACTION_LOOKBACK.get(s, 7)),
        "trigger_atr_mult": float(SESSION_TRIGGER_ATR_MULT.get(s, 0.85)),
        "min_conf": int(SESSION_MIN_CONF.get(s, 78)),
        "min_rr_tp3": float(SESSION_MIN_RR_TP3.get(s, 2.0)),
    }




def trigger_1h_abs_min_atr_adaptive(atr_pct: float, session_name: str) -> float:
    """
    Session-dynamic 1H trigger:
    - ATR-adaptive (existing)
    - PLUS per-session strictness multiplier:
        ASIA tighter > LON > NY looser
    """
    knobs = session_knobs(session_name)
    mult_atr = float(knobs["trigger_atr_mult"])

    # ATR-based dynamic trigger
    dyn = clamp(float(atr_pct) * float(mult_atr), 1.0, 4.5)

    # Session strictness multiplier applied to BASE floor
    sess = knobs["name"]
    base_mult = float(SESSION_1H_BASE_MULT.get(sess, 1.0))
    base_floor = float(TRIGGER_1H_ABS_MIN_BASE) * base_mult

    return max(float(base_floor), float(dyn))





# =========================================================
# DATA STRUCTURES
# =========================================================
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


@dataclass
class Setup:
    setup_id: str
    symbol: str
    market_symbol: str
    side: str
    conf: int
    entry: float
    sl: float
    tp1: Optional[float]
    tp2: Optional[float]
    tp3: float
    fut_vol_usd: float
    ch24: float
    ch4: float
    ch1: float
    ch15: float

    # ✅ adaptive EMA support (15m, volatility-based)
    ema_support_period: int
    ema_support_dist_pct: float

    # ✅ NEW: pullback EMA selection (7/14/21) + status
    pullback_ema_period: int
    pullback_ema_dist_pct: float
    pullback_ready: bool          # True => pullback happened / entry quality
    pullback_bypass_hot: bool     # True => volume>=50M bypass (no pullback needed)

    # ✅ engine label (A pullback / B momentum)
    engine: str

    is_trailing_tp3: bool
    created_ts: float



# =========================================================
# SIMPLE TTL CACHE (in-memory)
# =========================================================
_CACHE: Dict[str, Tuple[float, Any]] = {}

def cache_get(key: str) -> Optional[Any]:
    v = _CACHE.get(key)
    if not v:
        return None
    ts, obj = v
    return obj

def cache_set(key: str, obj: Any):
    _CACHE[key] = (time.time(), obj)

def cache_valid(key: str, ttl: int) -> bool:
    v = _CACHE.get(key)
    if not v:
        return False
    ts, _ = v
    return (time.time() - ts) <= ttl

# =========================================================
# MATH / INDICATORS HELPERS (MISSING FIX)
# =========================================================
def clamp(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, x))


def ema(values: List[float], period: int) -> float:
    """
    Returns the LAST EMA value (not full series).
    Safe for short lists.
    """
    if not values or period <= 0:
        return 0.0
    if len(values) < 2:
        return float(values[-1])

    k = 2.0 / (period + 1.0)
    e = float(values[0])
    for v in values[1:]:
        e = (float(v) * k) + (e * (1.0 - k))
    return float(e)


def compute_atr_from_ohlcv(ohlcv: List[List[float]], period: int = 14) -> float:
    """
    ATR (Wilder's smoothing) from OHLCV.
    ohlcv rows: [ts, open, high, low, close, vol]
    Returns last ATR value.
    """
    if not ohlcv or period <= 0:
        return 0.0

    # need at least period+1 candles for a stable ATR
    if len(ohlcv) < period + 1:
        return 0.0

    highs = []
    lows = []
    closes = []
    for c in ohlcv:
        try:
            highs.append(float(c[2]))
            lows.append(float(c[3]))
            closes.append(float(c[4]))
        except Exception:
            return 0.0

    trs: List[float] = []
    for i in range(1, len(ohlcv)):
        h = highs[i]
        l = lows[i]
        pc = closes[i - 1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(float(tr))

    if len(trs) < period:
        return 0.0

    # Wilder: first ATR = SMA(TR, period), then recursive smoothing
    atr = sum(trs[:period]) / float(period)
    for tr in trs[period:]:
        atr = ((atr * (period - 1)) + tr) / float(period)

    return float(atr)


def is_hot_coin(fut_vol_usd: float, ch24: float) -> bool:
    """
    Hot coin definition for TP cap logic.
    """
    try:
        return (float(fut_vol_usd) >= float(HOT_VOL_USD)) or (abs(float(ch24)) >= float(HOT_CH24_ABS))
    except Exception:
        return False


def tp_cap_pct_for_coin(fut_vol_usd: float, ch24: float) -> float:
    """
    Dynamic TP3 cap % based on coin hotness.
    """
    return float(TP_MAX_PCT_HOT) if is_hot_coin(fut_vol_usd, ch24) else float(TP_MAX_PCT_NORMAL)


# =========================================================
# ADAPTIVE EMA SUPPORT
# =========================================================
def adaptive_ema_period(atr_pct: float) -> int:
    """
    Volatility-aware EMA period:
    Higher ATR% => faster EMA (smaller period)
    Lower ATR%  => slower EMA (larger period)
    """
    try:
        a = float(atr_pct)
    except Exception:
        a = 2.0

    if a >= 6.0:
        p = 8
    elif a >= 4.0:
        p = 10
    elif a >= 2.0:
        p = 12
    else:
        p = 16

    return int(clamp(p, ADAPTIVE_EMA_MIN, ADAPTIVE_EMA_MAX))

def adaptive_ema_value(closes: List[float], atr_pct: float) -> Tuple[float, int]:
    """
    Returns: (ema_value, period_used)
    """
    if not closes:
        return 0.0, adaptive_ema_period(atr_pct)
    p = adaptive_ema_period(atr_pct)
    lookback = min(len(closes), p + 30)
    return ema(closes[-lookback:], p), p

def best_pullback_ema_15m(closes_15: List[float], entry: float) -> Tuple[float, int, float]:
    """
    Pick the best EMA among (7,14,21) based on which is closest to price.
    Returns: (ema_value, period, dist_pct)
    """
    if not closes_15 or entry <= 0:
        return 0.0, 14, 999.0

    candidates = [7, 14, 21]
    best = (0.0, 14, 999.0)

    for p in candidates:
        lookback = min(len(closes_15), p + 60)
        e = ema(closes_15[-lookback:], p)
        if e <= 0:
            continue
        dist_pct = abs(entry - e) / entry * 100.0
        if dist_pct < best[2]:
            best = (float(e), int(p), float(dist_pct))

    return best




# =========================================================
# REJECT TRACKER (in-memory per run) — COMPLETE
# =========================================================
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional



# Track counts + (optional) sample lines per reject reason
_REJECT_STATS = Counter()
_REJECT_SAMPLES: Dict[str, List[str]] = {}


# ✅ NEW: per-symbol reject reason (last one) for /screen
_REJECT_BY_SYMBOL: Dict[str, str] = {}  # base -> reason_key

# ✅ NEW: "Waiting for Trigger" (near-miss candidates)
# base -> {"side": "BUY"/"SELL", "ch1": float, "trig": float, "need": float}
_WAITING_TRIGGER: Dict[str, Dict[str, Any]] = {}

# ✅ NEW: last email skip reasons (per user) for /health transparency
_LAST_EMAIL_DECISION: Dict[int, Dict[str, Any]] = {}
_EMAIL_SKIP_COUNTERS: Dict[int, Counter] = defaultdict(Counter)

# ✅ NEW: last SMTP error (per run) to debug "email didn't arrive"
_LAST_SMTP_ERROR: Dict[int, str] = {}  # user_id -> last error text

# ✅ NEW: per-user diagnostics preference (runtime)
# - admin always "full"
# - non-admin default based on PUBLIC_DIAGNOSTICS_MODE: "friendly" or "off"
_USER_DIAG_MODE: Dict[int, str] = {}  # user_id -> "full" | "friendly" | "off"





# -------------------------
# Friendly reject titles (NO thresholds / params shown)
# IMPORTANT: keep this map COMPLETE for all _rej() keys used in make_setup/pick_setups
# -------------------------


REJECT_FRIENDLY_EN = {
    # global / admin gating (legacy key kept for compatibility)
    "melbourne_blackout_10_12": "⛔️ Signals are disabled between 10:00–12:00 (Melbourne time).",

    # data / market availability
    "no_fut_vol": "📉 Insufficient futures trading volume.",
    "bad_entry": "⚠️ Invalid or unreliable entry price data.",
    "ohlcv_missing_or_insufficient": "⚠️ Not enough candle data available (try again later).",

    # primary gates
    "ch1_below_trigger": "🧊 1H momentum is not strong enough yet.",
    "4h_not_aligned_for_long": "↔️ 4H trend is not aligned with LONG direction.",
    "4h_not_aligned_for_short": "↔️ 4H trend is not aligned with SHORT direction.",

    # engines / EMA logic
    "price_not_near_ema12_15m": "📍 Price is not close enough to Adaptive EMA (15m) for a quality entry.",
    "no_engine_passed": "🚫 Setup failed both engines (pullback + momentum filters).",
    "sharp_1h_no_ema_reaction": "⚡️ Strong 1H move detected, but no EMA reaction confirmation.",

    # SL/TP validity
    "bad_sl_tp_or_atr": "⚠️ Could not compute SL/TP reliably (ATR/price issue).",

    # direction / bias filters
    "24h_contradiction_for_long": "🚫 24H trend contradicts LONG bias.",
    "24h_contradiction_for_short": "🚫 24H trend contradicts SHORT bias.",

    # micro confirmation (email strictness)
    "15m_weak_and_not_early": "🟡 15m confirmation is weak and the setup is not strong enough to qualify as early.",

    # fallback / unknown
    "unknown": "❓ Filtered by strategy rules (details hidden).",
}



def is_admin_user(user_id: int) -> bool:
    return int(user_id) in ADMIN_USER_IDS if ADMIN_USER_IDS else False


def user_diag_mode(user_id: int) -> str:
    """
    Returns: "full" | "friendly" | "off"
    Admin always full.
    """
    uid = int(user_id)
    if is_admin_user(uid):
        return "full"
    if uid in _USER_DIAG_MODE:
        return _USER_DIAG_MODE[uid]
    if PUBLIC_DIAGNOSTICS_MODE in {"off", "none"}:
        return "off"
    return "friendly"


def fmt_price(x: float) -> str:
    ax = abs(float(x))
    if ax >= 100:
        return f"{x:.2f}"
    if ax >= 1:
        return f"{x:.4f}"
    if ax >= 0.1:
        return f"{x:.5f}"
    return f"{x:.6f}"


def reset_reject_tracker() -> None:
    """Call at the start of each scan so stats are per /screen run."""
    global _REJECT_STATS, _REJECT_SAMPLES, _REJECT_BY_SYMBOL
    _REJECT_STATS = Counter()
    _REJECT_SAMPLES = {}
    _REJECT_BY_SYMBOL = {}


def _rej(reason: str, base: str, mv: "MarketVol", extra: str = "") -> None:
    """
    Records reject stats + optional samples for DEBUG_REJECTS.

    - reason: stable key (do not put dynamic numbers in the key)
    - extra: can include numbers/thresholds (admin-only, stored only if DEBUG_REJECTS)
    """
    global _REJECT_STATS, _REJECT_SAMPLES, _REJECT_BY_SYMBOL

    reason = (reason or "unknown").strip()
    _REJECT_STATS[reason] += 1

    # ✅ store LAST reject reason per symbol (we show friendly label later)
    _REJECT_BY_SYMBOL[str(base)] = reason

    if not DEBUG_REJECTS:
        return

    try:
        last = fmt_price(float(getattr(mv, "last", 0.0) or 0.0))
    except Exception:
        last = str(getattr(mv, "last", "-"))

    sym = getattr(mv, "symbol", "") or ""
    line = f"{base} | {sym} | last={last}"
    if extra:
        line = f"{line} | {extra}"

    xs = _REJECT_SAMPLES.get(reason, [])
    if len(xs) < REJECT_TOP_N:
        xs.append(line)
        _REJECT_SAMPLES[reason] = xs



def _reject_report(diag_mode: str = "friendly") -> str:
    """
    diag_mode:
      - "full": show ALL technical keys + counts (+ samples if DEBUG_REJECTS)
      - "friendly": friendly titles + counts (no thresholds/params) [top 10 only]
      - "off": ""
    """
    diag_mode = (diag_mode or "friendly").strip().lower()
    if diag_mode == "off":
        return ""
    if not _REJECT_STATS:
        return ""

    parts: List[str] = []
    parts.append("🧩 Reject Diagnostics")
    parts.append(SEP)

    items = _REJECT_STATS.most_common() if diag_mode == "full" else _REJECT_STATS.most_common(10)

    for reason, cnt in items:
        if diag_mode == "full":
            parts.append(f"- {reason}: {cnt}")
            if DEBUG_REJECTS and (reason in _REJECT_SAMPLES) and _REJECT_SAMPLES[reason]:
                for s in _REJECT_SAMPLES[reason][:3]:
                    parts.append(f"    • {s}")
        else:
            title = REJECT_FRIENDLY_EN.get(reason, REJECT_FRIENDLY_EN.get("unknown", "Filtered by strategy rules."))
            parts.append(f"- {title}  (×{cnt})")

    if diag_mode != "full":
        parts.append("")
        parts.append("🔒 Technical details are hidden to protect the strategy.")

    return "\n".join(parts).strip()





# =========================================================
# DB
# =========================================================
def db_connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def db_init():
    con = db_connect()
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        tz TEXT NOT NULL,
        equity REAL NOT NULL,
        risk_mode TEXT NOT NULL,
        risk_value REAL NOT NULL,
        daily_cap_mode TEXT NOT NULL,
        daily_cap_value REAL NOT NULL,
        max_trades_day INTEGER NOT NULL,
        notify_on INTEGER NOT NULL,
        sessions_enabled TEXT NOT NULL,
        max_emails_per_session INTEGER NOT NULL,
        email_gap_min INTEGER NOT NULL,
        max_emails_per_day INTEGER NOT NULL,
        day_trade_date TEXT NOT NULL,
        day_trade_count INTEGER NOT NULL
    )
    """)

    cur.execute("PRAGMA table_info(users)")
    cols = {r[1] for r in cur.fetchall()}
    if "max_emails_per_day" not in cols:
        cur.execute(f"ALTER TABLE users ADD COLUMN max_emails_per_day INTEGER NOT NULL DEFAULT {int(DEFAULT_MAX_EMAILS_PER_DAY)}")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        entry REAL NOT NULL,
        sl REAL NOT NULL,
        risk_usd REAL NOT NULL,
        qty REAL NOT NULL,
        opened_ts REAL NOT NULL,
        closed_ts REAL,
        pnl REAL,
        r_mult REAL,
        note TEXT,
        signal_id TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        setup_id TEXT PRIMARY KEY,
        created_ts REAL NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        conf INTEGER NOT NULL,
        entry REAL NOT NULL,
        sl REAL NOT NULL,
        tp1 REAL,
        tp2 REAL,
        tp3 REAL NOT NULL,
        fut_vol_usd REAL NOT NULL,
        ch24 REAL NOT NULL,
        ch4 REAL NOT NULL,
        ch1 REAL NOT NULL,
        ch15 REAL NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS email_state (
        user_id INTEGER PRIMARY KEY,
        session_key TEXT NOT NULL,
        sent_count INTEGER NOT NULL,
        last_email_ts REAL NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS emailed_symbols (
        user_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        emailed_ts REAL NOT NULL,
        PRIMARY KEY (user_id, symbol)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS email_daily (
        user_id INTEGER NOT NULL,
        day_local TEXT NOT NULL,
        sent_count INTEGER NOT NULL,
        PRIMARY KEY (user_id, day_local)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS risk_daily (
        user_id INTEGER NOT NULL,
        day_local TEXT NOT NULL,
        used_risk_usd REAL NOT NULL,
        PRIMARY KEY (user_id, day_local)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS setup_counter (
        day_yyyymmdd TEXT PRIMARY KEY,
        seq INTEGER NOT NULL
    )
    """)

    con.commit()
    con.close()

def _default_sessions_for_tz(tz_name: str) -> List[str]:
    s = tz_name.lower()
    if "america" in s or s.startswith("us/") or "new_york" in s:
        return ["NY"]
    if "europe" in s or "london" in s or "africa" in s:
        return ["LON"]
    return ["ASIA"]

def _order_sessions(xs: List[str]) -> List[str]:
    cleaned = []
    for x in xs:
        x = str(x).strip().upper()
        if x in SESSIONS_UTC and x not in cleaned:
            cleaned.append(x)
    return [s for s in SESSION_PRIORITY if s in cleaned]

def get_user(user_id: int) -> dict:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()

    if not row:
        tz_name = "Australia/Melbourne"
        sessions = _order_sessions(_default_sessions_for_tz(tz_name)) or _default_sessions_for_tz(tz_name)
        now_local = datetime.now(ZoneInfo(tz_name)).date().isoformat()
        cur.execute("""
            INSERT INTO users (
                user_id, tz, equity, risk_mode, risk_value,
                daily_cap_mode, daily_cap_value,
                max_trades_day, notify_on,
                sessions_enabled, max_emails_per_session, email_gap_min,
                max_emails_per_day,
                day_trade_date, day_trade_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            tz_name,
            float(DEFAULT_EQUITY),
            DEFAULT_RISK_MODE,
            float(DEFAULT_RISK_VALUE),
            DEFAULT_DAILY_CAP_MODE,
            float(DEFAULT_DAILY_CAP_VALUE),
            int(DEFAULT_MAX_TRADES_DAY),
            1 if EMAIL_ENABLED else 0,
            json.dumps(sessions),
            int(DEFAULT_MAX_EMAILS_PER_SESSION),
            int(DEFAULT_MIN_EMAIL_GAP_MIN),
            int(DEFAULT_MAX_EMAILS_PER_DAY),
            now_local,
            0
        ))
        con.commit()
        cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        row = cur.fetchone()

    con.close()
    return dict(row)

def update_user(user_id: int, **kwargs):
    if not kwargs:
        return
    con = db_connect()
    cur = con.cursor()
    sets = ", ".join([f"{k}=?" for k in kwargs.keys()])
    vals = list(kwargs.values()) + [user_id]
    cur.execute(f"UPDATE users SET {sets} WHERE user_id=?", vals)
    con.commit()
    con.close()

def reset_daily_if_needed(user: dict) -> dict:
    tz = ZoneInfo(user["tz"])
    today = datetime.now(tz).date().isoformat()
    if user["day_trade_date"] != today:
        update_user(user["user_id"], day_trade_date=today, day_trade_count=0)
        user = get_user(user["user_id"])
    return user

def list_users_notify_on() -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE notify_on=1")
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]

def email_state_get(user_id: int) -> dict:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM email_state WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    if not row:
        cur.execute("INSERT INTO email_state (user_id, session_key, sent_count, last_email_ts) VALUES (?, ?, ?, ?)",
                    (user_id, "NONE", 0, 0.0))
        con.commit()
        cur.execute("SELECT * FROM email_state WHERE user_id=?", (user_id,))
        row = cur.fetchone()
    con.close()
    return dict(row)

def email_state_set(user_id: int, **kwargs):
    if not kwargs:
        return
    con = db_connect()
    cur = con.cursor()
    sets = ", ".join([f"{k}=?" for k in kwargs.keys()])
    vals = list(kwargs.values()) + [user_id]
    cur.execute(f"UPDATE email_state SET {sets} WHERE user_id=?", vals)
    con.commit()
    con.close()

def mark_symbol_emailed(user_id: int, symbol: str):
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO emailed_symbols (user_id, symbol, emailed_ts)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, symbol) DO UPDATE SET emailed_ts=excluded.emailed_ts
    """, (user_id, symbol.upper(), time.time()))
    con.commit()
    con.close()

def symbol_recently_emailed(user_id: int, symbol: str, cooldown_hours: float) -> bool:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT emailed_ts FROM emailed_symbols WHERE user_id=? AND symbol=?", (user_id, symbol.upper()))
    row = cur.fetchone()
    con.close()
    if not row:
        return False
    return (time.time() - float(row["emailed_ts"])) < (cooldown_hours * 3600)

def _email_daily_get(user_id: int, day_local: str) -> int:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT sent_count FROM email_daily WHERE user_id=? AND day_local=?", (user_id, day_local))
    row = cur.fetchone()
    if not row:
        cur.execute("INSERT INTO email_daily (user_id, day_local, sent_count) VALUES (?, ?, ?)", (user_id, day_local, 0))
        con.commit()
        con.close()
        return 0
    con.close()
    return int(row["sent_count"])

def _email_daily_inc(user_id: int, day_local: str, inc: int = 1):
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO email_daily (user_id, day_local, sent_count)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, day_local) DO UPDATE SET sent_count = sent_count + excluded.sent_count
    """, (user_id, day_local, int(inc)))
    con.commit()
    con.close()

def _risk_daily_get(user_id: int, day_local: str) -> float:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT used_risk_usd FROM risk_daily WHERE user_id=? AND day_local=?", (user_id, day_local))
    row = cur.fetchone()
    if not row:
        cur.execute("INSERT INTO risk_daily (user_id, day_local, used_risk_usd) VALUES (?, ?, ?)", (user_id, day_local, 0.0))
        con.commit()
        con.close()
        return 0.0
    con.close()
    return float(row["used_risk_usd"])

def _risk_daily_inc(user_id: int, day_local: str, inc_usd: float):
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO risk_daily (user_id, day_local, used_risk_usd)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, day_local) DO UPDATE SET used_risk_usd = used_risk_usd + excluded.used_risk_usd
    """, (user_id, day_local, float(inc_usd)))
    con.commit()
    con.close()

def _user_day_local(user: dict) -> str:
    tz = ZoneInfo(user["tz"])
    return datetime.now(tz).date().isoformat()

def db_insert_signal(s: Setup):
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO signals (
            setup_id, created_ts, symbol, side, conf, entry, sl, tp1, tp2, tp3,
            fut_vol_usd, ch24, ch4, ch1, ch15
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        s.setup_id, s.created_ts, s.symbol, s.side, s.conf, s.entry, s.sl,
        s.tp1, s.tp2, s.tp3, s.fut_vol_usd, s.ch24, s.ch4, s.ch1, s.ch15
    ))
    con.commit()
    con.close()

def db_get_signal(setup_id: str) -> Optional[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM signals WHERE setup_id=?", (setup_id.strip(),))
    row = cur.fetchone()
    con.close()
    return dict(row) if row else None

def db_list_signals_since(ts_from: float) -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM signals WHERE created_ts>=? ORDER BY created_ts ASC", (ts_from,))
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]

def db_trade_open(user_id: int, symbol: str, side: str, entry: float, sl: float,
                  risk_usd: float, qty: float, note: str = "", signal_id: str = "") -> int:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO trades (user_id, symbol, side, entry, sl, risk_usd, qty, opened_ts, note, signal_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, symbol.upper(), side.upper(), float(entry), float(sl),
        float(risk_usd), float(qty), time.time(), note, signal_id
    ))
    trade_id = cur.lastrowid
    con.commit()
    con.close()
    return int(trade_id)

def db_trade_close(user_id: int, trade_id: int, pnl: float) -> Optional[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM trades WHERE id=? AND user_id=? AND closed_ts IS NULL", (trade_id, user_id))
    row = cur.fetchone()
    if not row:
        con.close()
        return None
    t = dict(row)

    risk = float(t["risk_usd"]) if float(t["risk_usd"]) != 0 else 0.0
    r_mult = (float(pnl) / risk) if risk > 0 else None

    cur.execute("""
        UPDATE trades
        SET closed_ts=?, pnl=?, r_mult=?
        WHERE id=? AND user_id=?
    """, (time.time(), float(pnl), r_mult, trade_id, user_id))
    con.commit()
    con.close()

    t["pnl"] = float(pnl)
    t["r_mult"] = r_mult
    return t

def db_open_trades(user_id: int) -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM trades WHERE user_id=? AND closed_ts IS NULL ORDER BY opened_ts ASC", (user_id,))
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]

def db_trades_since(user_id: int, ts_from: float) -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        SELECT * FROM trades
        WHERE user_id=? AND opened_ts>=?
        ORDER BY opened_ts ASC
    """, (user_id, ts_from))
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


# =========================================================
# EXCHANGE HELPERS (FAST: singleton exchange + no repeated load_markets)
# =========================================================
import threading

_EX: Optional[ccxt.Exchange] = None
_EX_LOCK = threading.Lock()
_EX_MARKETS_LOADED = False

def get_exchange() -> ccxt.Exchange:
    """
    ✅ SINGLETON exchange instance
    - Avoids building a new exchange per request
    - Avoids calling load_markets() repeatedly
    """
    global _EX, _EX_MARKETS_LOADED
    with _EX_LOCK:
        if _EX is None:
            klass = ccxt.__dict__[EXCHANGE_ID]
            _EX = klass({
                "enableRateLimit": True,
                "timeout": 20000,
                "options": {"defaultType": DEFAULT_TYPE},
            })
            _EX_MARKETS_LOADED = False

        if not _EX_MARKETS_LOADED:
            _EX.load_markets()
            _EX_MARKETS_LOADED = True

        return _EX

def safe_split_symbol(sym: Optional[str]) -> Optional[Tuple[str, str]]:
    if not sym:
        return None
    pair = sym.split(":")[0]
    if "/" not in pair:
        return None
    return tuple(pair.split("/", 1))

def usd_notional(mv: MarketVol) -> float:
    if mv.quote in STABLES and mv.quote_vol:
        return float(mv.quote_vol)
    price = mv.vwap if mv.vwap else mv.last
    if not price or not mv.base_vol:
        return 0.0
    return float(mv.base_vol) * float(price)

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

def fetch_futures_tickers() -> Dict[str, MarketVol]:
    """
    ✅ Uses singleton exchange (no repeated load_markets)
    """
    if cache_valid("tickers_best_fut", TICKERS_TTL_SEC):
        return cache_get("tickers_best_fut")

    ex = get_exchange()
    tickers = ex.fetch_tickers()

    best: Dict[str, MarketVol] = {}
    for t in tickers.values():
        mv = to_mv(t)
        if not mv:
            continue
        if mv.quote not in STABLES:
            continue
        if mv.base not in best or usd_notional(mv) > usd_notional(best[mv.base]):
            best[mv.base] = mv

    cache_set("tickers_best_fut", best)
    return best

def fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> List[List[float]]:
    """
    ✅ Uses singleton exchange (no repeated build + load_markets)
    ✅ TTL cache already exists
    """
    key = f"ohlcv:{symbol}:{timeframe}:{limit}"
    if cache_valid(key, OHLCV_TTL_SEC):
        return cache_get(key)

    ex = get_exchange()
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit) or []
    cache_set(key, data)
    return data


# =========================================================
# ✅ Session resolution helpers (for /screen + engine)
# =========================================================
def parse_hhmm(s: str) -> Tuple[int, int]:
    m = re.match(r"^(\d{2}):(\d{2})$", s.strip())
    if not m:
        raise ValueError("bad time")
    hh = int(m.group(1))
    mm = int(m.group(2))
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        raise ValueError("bad time")
    return hh, mm


def current_session_utc(now_utc: Optional[datetime] = None) -> str:
    """
    24H coverage (no gaps) in UTC:
    - ASIA: 20:00–06:00
    - LON : 06:00–13:00
    - NY  : 13:00–20:00

    Can be called with no args safely.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    h = now_utc.hour

    # ASIA crosses midnight
    if (h >= 20) or (h < 6):
        return "ASIA"
    if 6 <= h < 13:
        return "LON"
    if 13 <= h < 20:
        return "NY"

    # Should never happen, but safe fallback
    return "ASIA"






def ema_support_proximity_ok(entry: float, ema_val: float, atr_1h: float, session_name: str):
    """
    Adaptive EMA proximity check.
    Returns: (ok, dist_pct, thr_pct, atr_pct)
    """
    if entry <= 0 or ema_val <= 0:
        return False, 999.0, 0.0, 0.0

    dist = abs(entry - ema_val)
    dist_pct = (dist / entry) * 100.0

    atr_pct = (atr_1h / entry) * 100.0 if (atr_1h and entry) else 0.0

    knobs = session_knobs(session_name)
    prox_mult = knobs["ema_prox_mult"]

    thr_pct = clamp(
        (EMA_SUPPORT_MAX_DIST_ATR_MULT * prox_mult) * atr_pct,
        EMA_SUPPORT_MAX_DIST_PCT_MIN,
        EMA_SUPPORT_MAX_DIST_PCT_MAX,
    )

    ok = dist_pct <= thr_pct
    return ok, dist_pct, thr_pct, atr_pct

def ema_support_reaction_ok_15m(c15: List[List[float]], ema_val: float, side: str, session_name: str) -> bool:
    """
    Adaptive EMA reaction:
    - BUY: touched/broke below EMA and closed back above
    - SELL: touched/broke above EMA and closed back below
    """
    if not c15 or len(c15) < 10 or ema_val <= 0:
        return False

    lookback_n = int(session_knobs(session_name)["ema_reaction_lookback"])
    lookback_n = int(clamp(lookback_n, 4, 12))
    lookback = c15[-lookback_n:]

    for c in lookback:
        h = float(c[2]); l = float(c[3]); cl = float(c[4])
        if side == "BUY":
            if (l <= ema_val) and (cl > ema_val):
                return True
        else:
            if (h >= ema_val) and (cl < ema_val):
                return True
    return False







def metrics_from_candles_1h_15m(market_symbol: str) -> Tuple[float, float, float, float, float, int, List[List[float]]]:
    """
    returns: ch1, ch4, ch15, atr_1h, ema_support_15m, ema_support_period, c15
    """
    need_1h = max(ATR_PERIOD + 6, 35)
    c1 = fetch_ohlcv(market_symbol, "1h", limit=need_1h)
    if not c1 or len(c1) < 6:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, []

    closes_1h = [float(x[4]) for x in c1]
    c_last = closes_1h[-1]
    c_prev1 = closes_1h[-2]
    c_prev4 = closes_1h[-5] if len(closes_1h) >= 5 else closes_1h[0]

    ch1 = ((c_last - c_prev1) / c_prev1) * 100.0 if c_prev1 else 0.0
    ch4 = ((c_last - c_prev4) / c_prev4) * 100.0 if c_prev4 else 0.0
    atr_1h = compute_atr_from_ohlcv(c1, ATR_PERIOD)

    c15 = fetch_ohlcv(market_symbol, "15m", limit=80)
    if not c15 or len(c15) < 20:
        return ch1, ch4, 0.0, atr_1h, 0.0, 0, []

    closes_15 = [float(x[4]) for x in c15]
    entry_proxy = float(closes_15[-1]) if closes_15 else 0.0
    atr_pct_proxy = (atr_1h / entry_proxy) * 100.0 if (atr_1h and entry_proxy) else 2.0

    ema_support_15m, ema_period = adaptive_ema_value(closes_15, atr_pct_proxy)

    c15_last = float(c15[-1][4])
    c15_prev = float(c15[-2][4])
    ch15 = ((c15_last - c15_prev) / c15_prev) * 100.0 if c15_prev else 0.0

    return ch1, ch4, ch15, atr_1h, ema_support_15m, int(ema_period), c15



# =========================================================
# FORMATTING
# =========================================================
def fmt_money(x: float) -> str:
    ax = abs(x)
    if ax >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if ax >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:.0f}"

def pct_with_emoji(p: float) -> str:
    val = int(round(p))
    if val >= 3:
        emo = "🟢"
    elif val <= -3:
        emo = "🔴"
    else:
        emo = "🟡"
    return f"{val:+d}% {emo}"

def tv_chart_url(symbol_base: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol=BYBIT:{symbol_base.upper()}USDT.P"

def table_md(rows: List[List[Any]], headers: List[str]) -> str:
    return "```\n" + tabulate(rows, headers=headers, tablefmt="github") + "\n```"

# Email-specific price formatting (less noisy)
def fmt_price_email(x: float) -> str:
    ax = abs(float(x))
    if ax >= 100:
        return f"{x:.2f}"
    if ax >= 1:
        return f"{x:.3f}"
    if ax >= 0.1:
        return f"{x:.4f}"
    if ax >= 0.01:
        return f"{x:.5f}"
    return f"{x:.6f}"


# =========================================================
# TELEGRAM SAFE SEND (chunking + markdown fallback)
# =========================================================
SAFE_CHUNK = 3500

async def send_long_message(update: Update, text: str, parse_mode: Optional[str] = None,
                            disable_web_page_preview: bool = True, reply_markup=None):
    if not update or not update.message:
        return

    chunks = []
    s = text or ""
    while s:
        chunks.append(s[:SAFE_CHUNK])
        s = s[SAFE_CHUNK:]

    first = True
    for ch in chunks:
        try:
            await update.message.reply_text(
                ch,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
                reply_markup=reply_markup if first else None,
            )
        except Exception as e:
            logger.exception("send_long_message markdown failed, fallback plain. err=%s", e)
            await update.message.reply_text(
                ch,
                disable_web_page_preview=disable_web_page_preview,
                reply_markup=reply_markup if first else None,
            )
        first = False


# =========================================================
# SIGNAL IDs
# =========================================================
def next_setup_id() -> str:
    today = datetime.utcnow().strftime("%Y%m%d")
    con = db_connect()
    cur = con.cursor()
    try:
        cur.execute("BEGIN IMMEDIATE")
        cur.execute("""
            INSERT INTO setup_counter (day_yyyymmdd, seq)
            VALUES (?, 1)
            ON CONFLICT(day_yyyymmdd) DO UPDATE SET seq = seq + 1
        """, (today,))
        cur.execute("SELECT seq FROM setup_counter WHERE day_yyyymmdd=?", (today,))
        n = int(cur.fetchone()["seq"])
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    return f"PF-{today}-{n:04d}"


# =========================================================
# SL/TP ENGINE (closer + dynamic cap + ✅ confidence-weighted TP scaling)
# =========================================================
def sl_mult_from_conf(conf: int) -> float:
    # slightly tighter than old premium; tuned for higher win-rate with closer TPs
    if conf >= 88:
        return 2.2
    if conf >= 80:
        return 1.95
    if conf >= 70:
        return 1.70
    return 1.55

def tp3_rr_target_from_conf(conf: int) -> float:
    """
    ✅ TP scaling: confidence-weighted RR target for TP3.
    """
    if conf >= 88:
        return 2.6
    if conf >= 80:
        return 2.4
    if conf >= 72:
        return 2.2
    # below 72 we still may keep for /screen (awareness), but email floors will filter
    return 2.0

def tp_r_mults_from_conf(conf: int) -> Tuple[float, float, float]:
    """
    Returns (tp1_rr, tp2_rr, tp3_rr) with tp3 driven by confidence.
    """
    tp3 = tp3_rr_target_from_conf(conf)
    tp1 = 1.0
    tp2 = max(1.5, min(1.9, tp3 * 0.70))
    if tp2 >= tp3:
        tp2 = max(1.6, tp3 - 0.3)
    return (tp1, tp2, tp3)



def compute_sl_tp(
    entry: float,
    side: str,
    atr: float,
    conf: int,
    tp_cap_pct: float,
    rr_bonus: float = 0.0,
    tp_cap_bonus_pct: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Returns (sl, tp3, R)
    """
    if entry <= 0 or atr <= 0:
        return 0.0, 0.0, 0.0

    sl_dist = sl_mult_from_conf(conf) * atr

    min_dist = (ATR_MIN_PCT / 100.0) * entry
    max_dist = (ATR_MAX_PCT / 100.0) * entry
    sl_dist = clamp(sl_dist, min_dist, max_dist)

    R = sl_dist

    # ✅ confidence-weighted RR + engine bonus
    rr_target = tp3_rr_target_from_conf(conf) + rr_bonus
    tp_dist = rr_target * R

    # ✅ TP cap with engine bonus
    tp_cap = ((tp_cap_pct + tp_cap_bonus_pct) / 100.0) * entry
    tp_dist = min(tp_dist, tp_cap)

    if side == "BUY":
        sl = entry - sl_dist
        tp3 = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp3 = entry - tp_dist

    return sl, tp3, R




def _distinctify(a: float, b: float, c: float, entry: float, side: str) -> Tuple[float, float, float]:
    if entry <= 0:
        return a, b, c
    eps = max(entry * 1e-6, 1e-8)
    if side == "BUY":
        if a >= b: a = b - eps
        if b >= c: b = c - eps
        if a >= b: a = b - eps
    else:
        if a <= b: a = b + eps
        if b <= c: b = c + eps
        if a <= b: a = b + eps
    if abs(a - b) < eps: a = a - eps if side == "BUY" else a + eps
    if abs(b - c) < eps: b = b - eps if side == "BUY" else b + eps
    if abs(a - c) < eps: a = a - 2*eps if side == "BUY" else a + 2*eps
    return a, b, c


def multi_tp(
    entry: float,
    side: str,
    R: float,
    tp_cap_pct: float,
    conf: int,
    rr_bonus: float = 0.0,
    tp_cap_bonus_pct: float = 0.0,
) -> Tuple[float, float, float]:
    if entry <= 0 or R <= 0:
        return 0.0, 0.0, 0.0

    r1, r2, r3 = tp_r_mults_from_conf(conf)
    r3 += rr_bonus  # ✅ Engine B bonus

    maxd = ((tp_cap_pct + tp_cap_bonus_pct) / 100.0) * entry

    d3 = min(r3 * R, maxd)
    d2 = min(r2 * R, d3 * (r2 / r3 if r3 > 0 else 0.7))
    d1 = min(r1 * R, d3 * (r1 / r3 if r3 > 0 else 0.4))

    if side == "BUY":
        tp1, tp2, tp3 = (entry + d1, entry + d2, entry + d3)
    else:
        tp1, tp2, tp3 = (entry - d1, entry - d2, entry - d3)

    return _distinctify(tp1, tp2, tp3, entry, side)





def rr_to_tp(entry: float, sl: float, tp: float) -> float:
    d_sl = abs(entry - sl)
    d_tp = abs(tp - entry)
    if d_sl <= 0:
        return 0.0
    return d_tp / d_sl


# =========================================================
# SETUP ENGINE
# =========================================================
def compute_confidence(side: str, ch24: float, ch4: float, ch1: float, ch15: float, fut_vol_usd: float) -> int:
    score = 50.0
    is_long = (side == "BUY")

    def align(x: float, w: float):
        nonlocal score
        score += w if ((x > 0) == is_long) else -w

    align(ch24, 12)
    align(ch4, 10)
    align(ch1, 9)
    align(ch15, 6)

    def signed_mag(x: float, k: float) -> float:
        return abs(x) * k if ((x > 0) == is_long) else -abs(x) * k

    mag = (
        signed_mag(ch24, 0.7) +
        signed_mag(ch4,  1.1) +
        signed_mag(ch1,  1.6) +
        signed_mag(ch15, 1.2)
    )
    mag = clamp(mag, -22.0, 22.0)
    score += mag

    if fut_vol_usd >= 20_000_000:
        score += 9
    elif fut_vol_usd >= 8_000_000:
        score += 7
    elif fut_vol_usd >= 3_000_000:
        score += 5

    score = clamp(score, 0, 100)
    return int(round(score))

def build_leaders_table(best_fut: Dict[str, MarketVol]) -> str:
    leaders = sorted(best_fut.items(), key=lambda kv: usd_notional(kv[1]), reverse=True)[:LEADERS_N]
    rows = []
    for base, mv in leaders:
        rows.append([base, fmt_money(usd_notional(mv)), pct_with_emoji(float(mv.percentage or 0.0)), fmt_price(float(mv.last or 0.0))])
    return "*Market Leaders (Top 10 by Futures Volume)*\n" + table_md(rows, ["SYM", "F Vol", "24H", "Last"])

def compute_directional_lists(best_fut: Dict[str, MarketVol]) -> Tuple[List[Tuple], List[Tuple]]:
    """
    ✅ FAST version:
    - For leaders/losers we only need: volume, 24H move, and 4H alignment
    - So we fetch ONLY 4H candles (limit=2) instead of full 1H+15m metrics
    """
    up, dn = [], []

    for base, mv in best_fut.items():
        vol = usd_notional(mv)
        if vol < MOVER_VOL_USD_MIN:
            continue

        ch24 = float(mv.percentage or 0.0)
        if ch24 < MOVER_UP_24H_MIN and ch24 > MOVER_DN_24H_MAX:
            continue

        # ✅ 4H alignment using real 4H candles (very light)
        c4h = fetch_ohlcv(mv.symbol, "4h", limit=2)
        if not c4h or len(c4h) < 2:
            continue

        c_last = float(c4h[-1][4])
        c_prev = float(c4h[-2][4])
        if c_prev <= 0:
            continue
        ch4 = ((c_last - c_prev) / c_prev) * 100.0

        if ch24 >= MOVER_UP_24H_MIN and ch4 > 0:
            up.append((base, vol, ch24, ch4, float(mv.last or 0.0)))
        if ch24 <= MOVER_DN_24H_MAX and ch4 < 0:
            dn.append((base, vol, ch24, ch4, float(mv.last or 0.0)))

    up.sort(key=lambda x: (x[2], x[1]), reverse=True)
    dn.sort(key=lambda x: (x[2], x[1]))
    return up, dn


def movers_tables(best_fut: Dict[str, MarketVol]) -> Tuple[str, str]:
    up, dn = compute_directional_lists(best_fut)
    up_rows = [[b, fmt_money(v), pct_with_emoji(c24), pct_with_emoji(c4), fmt_price(px)] for b, v, c24, c4, px in up[:10]]
    dn_rows = [[b, fmt_money(v), pct_with_emoji(c24), pct_with_emoji(c4), fmt_price(px)] for b, v, c24, c4, px in dn[:10]]
    up_txt = "*Directional Leaders (24H ≥ +10%, F vol ≥ 5M, 4H aligned)*\n" + (table_md(up_rows, ["SYM", "F Vol", "24H", "4H", "Last"]) if up_rows else "_None_")
    dn_txt = "*Directional Losers (24H ≤ -10%, F vol ≥ 5M, 4H aligned)*\n" + (table_md(dn_rows, ["SYM", "F Vol", "24H", "4H", "Last"]) if dn_rows else "_None_")
    return up_txt, dn_txt



def make_setup(
    base: str,
    mv: MarketVol,
    strict_15m: bool = True,
    session_name: str = "LON",
    allow_no_pullback: bool = True,     # ✅ /screen True, Email False (except HOT bypass)
    hot_vol_usd: float = HOT_VOL_USD,   # 50M
    trigger_loosen_mult: float = 1.0,
    waiting_near_pct: float = SCREEN_WAITING_NEAR_PCT,
) -> Optional[Setup]:

    fut_vol = usd_notional(mv)
    if fut_vol <= 0:
        _rej("no_fut_vol", base, mv)
        return None

    entry = float(mv.last or 0.0)
    if entry <= 0:
        _rej("bad_entry", base, mv)
        return None

    ch24 = float(mv.percentage or 0.0)

    ch1, ch4, ch15, atr_1h, ema_support_15m, ema_period, c15 = metrics_from_candles_1h_15m(mv.symbol)
    if (ch1 == 0.0 and ch4 == 0.0 and ch15 == 0.0 and atr_1h == 0.0) or (not c15) or (ema_support_15m == 0.0):
        _rej("ohlcv_missing_or_insufficient", base, mv, "metrics/ema missing")
        return None

    # --------- SESSION-DYNAMIC 1H TRIGGER ----------
    atr_pct_now = (atr_1h / entry) * 100.0 if (atr_1h and entry) else 0.0
    trig_min_raw = trigger_1h_abs_min_atr_adaptive(atr_pct_now, session_name)

    trig_min = max(0.4, float(trig_min_raw) * float(trigger_loosen_mult))

    if abs(ch1) < trig_min:
        # "Waiting for Trigger" capture
        if trig_min > 0 and abs(ch1) >= (float(waiting_near_pct) * trig_min):
            side_guess = "BUY" if ch1 > 0 else "SELL"
            _WAITING_TRIGGER[str(base)] = {
                "side": side_guess,
                "ch1": float(ch1),
                "trig": float(trig_min),
                "need": float(trig_min - abs(ch1)),
            }
        _rej("ch1_below_trigger", base, mv, f"ch1={ch1:+.2f}% < {trig_min:.2f}% (ATR%={atr_pct_now:.2f})")
        return None

    side = "BUY" if ch1 > 0 else "SELL"

    # 4H alignment
    if side == "BUY" and ch4 < ALIGN_4H_MIN:
        _rej("4h_not_aligned_for_long", base, mv, f"side=BUY ch4={ch4:+.2f}%")
        return None
    if side == "SELL" and ch4 > -ALIGN_4H_MIN:
        _rej("4h_not_aligned_for_short", base, mv, f"side=SELL ch4={ch4:+.2f}%")
        return None

    # =========================================================
    # ✅ PULLBACK EMA (7/14/21) selection (15m)
    # =========================================================
    closes_15 = [float(x[4]) for x in c15]
    pb_ema_val, pb_ema_p, pb_dist_pct = best_pullback_ema_15m(closes_15, entry)

    # "HOT" bypass: if futures 24h vol >= 50M, no pullback required
    pullback_bypass_hot = (float(fut_vol) >= float(hot_vol_usd))

    # Proximity threshold uses session knobs + ATR% (same idea as your adaptive EMA gate)
    pb_ok = False
    if pb_ema_val > 0:
        pb_ok, _, _, _ = ema_support_proximity_ok(entry, pb_ema_val, atr_1h, session_name)

    pullback_ready = bool(pb_ok or pullback_bypass_hot)

    # =========================================================
    # ENGINE A (Pullback) vs ENGINE B (Momentum)
    # =========================================================
    # Engine A requires pullback-ready (or hot bypass)
    engine_a_ok = bool(ENGINE_A_PULLBACK_ENABLED and pullback_ready)

    # Engine B = momentum/expansion (can be far from EMA)
    engine_b_ok = False
    if ENGINE_B_MOMENTUM_ENABLED:
        if abs(ch1) >= MOMENTUM_MIN_CH1 and abs(ch24) >= MOMENTUM_MIN_24H:
            if fut_vol >= (MOVER_VOL_USD_MIN * MOMENTUM_VOL_MULT):
                body_pct = abs(ch1)
                if atr_pct_now > 0 and body_pct >= (MOMENTUM_ATR_BODY_MULT * atr_pct_now):
                    # distance vs ADAPTIVE EMA (existing pump tolerance)
                    ema_ok, dist_pct, _, _ = ema_support_proximity_ok(entry, ema_support_15m, atr_1h, session_name)
                    if dist_pct <= MOMENTUM_MAX_ADAPTIVE_EMA_DIST:
                        engine_b_ok = True

    # ---------------------------------------------------------
    # ✅ Two-case logic AFTER 1H criteria:
    # A) Pullback-ready => valid
    # B) Not pullback-ready => only allowed on /screen (allow_no_pullback=True),
    #    BUT emails must not use this (except hot bypass already handled above).
    # ---------------------------------------------------------
    if not engine_a_ok and not engine_b_ok:
        _rej("no_engine_passed", base, mv, f"ch1={ch1:.2f} ch24={ch24:.2f} pb_dist={pb_dist_pct:.2f}")
        return None

    # If pullback not ready:
    # - allow only if allow_no_pullback True AND engine_b_ok True (screen awareness)
    if not pullback_ready:
        if not (allow_no_pullback and engine_b_ok):
            _rej("price_not_near_ema12_15m", base, mv, f"pullback_not_ready pb_dist={pb_dist_pct:.2f}")
            return None

    # Prefer Engine A when pullback-ready; otherwise engine B
    engine = "A" if pullback_ready else "B"

    # Sharp 1H move gating:
    # - if we are relying on pullback (engine A) then require EMA reaction
    if engine == "A" and abs(float(ch1)) >= float(SHARP_1H_MOVE_PCT):
        if not ema_support_reaction_ok_15m(c15, pb_ema_val, side, session_name):
            _rej("sharp_1h_no_ema_reaction", base, mv, f"ch1={ch1:+.2f}% needs EMA reaction")
            return None

    # Soft 24H contradiction gate
    thr = clamp(max(12.0, 2.5 * ((atr_1h / entry) * 100.0 if (atr_1h and entry) else 0.0)), 12.0, 22.0)
    if side == "BUY" and ch24 <= -thr:
        _rej("24h_contradiction_for_long", base, mv, f"ch24={ch24:+.1f}% <= -{thr:.1f}%")
        return None
    if side == "SELL" and ch24 >= +thr:
        _rej("24h_contradiction_for_short", base, mv, f"ch24={ch24:+.1f}% >= +{thr:.1f}%")
        return None

    # 15m confirm logic (email strictness only)
    is_confirm_15m = abs(ch15) >= CONFIRM_15M_ABS_MIN
    is_early_allowed = (abs(ch1) >= EARLY_1H_ABS_MIN)

    if strict_15m:
        if (not is_confirm_15m) and (not is_early_allowed):
            _rej("15m_weak_and_not_early", base, mv, f"ch15={ch15:+.2f}% ch1={ch1:+.2f}%")
            return None

    conf = compute_confidence(side, ch24, ch4, ch1, ch15, fut_vol)

    if strict_15m and (not is_confirm_15m):
        conf = max(0, int(conf) - int(EARLY_CONF_PENALTY))

    tp_cap_pct = tp_cap_pct_for_coin(fut_vol, ch24)

    # Engine B gets higher TP expectation ONLY when we actually tag engine B
    rr_bonus = ENGINE_B_RR_BONUS if engine == "B" else 0.0
    tp_cap_bonus = ENGINE_B_TP_CAP_BONUS_PCT if engine == "B" else 0.0

    sl, tp3_single, R = compute_sl_tp(entry, side, atr_1h, conf, tp_cap_pct,
                                      rr_bonus=rr_bonus, tp_cap_bonus_pct=tp_cap_bonus)
    if sl <= 0 or tp3_single <= 0 or R <= 0:
        _rej("bad_sl_tp_or_atr", base, mv, f"atr={atr_1h:.6g} entry={entry:.6g}")
        return None

    tp1 = tp2 = None
    tp3 = tp3_single
    if conf >= MULTI_TP_MIN_CONF:
        _tp1, _tp2, _tp3 = multi_tp(entry, side, R, tp_cap_pct, conf,
                                    rr_bonus=rr_bonus, tp_cap_bonus_pct=tp_cap_bonus)
        if _tp1 and _tp2 and _tp3:
            tp1, tp2, tp3 = _tp1, _tp2, _tp3

    sid = next_setup_id()
    hot = is_hot_coin(fut_vol, ch24)

    return Setup(
        setup_id=sid,
        symbol=base,
        market_symbol=mv.symbol,
        side=side,
        conf=int(conf),
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        fut_vol_usd=fut_vol,
        ch24=ch24,
        ch4=ch4,
        ch1=ch1,
        ch15=ch15,
        ema_support_period=int(ema_period),
        ema_support_dist_pct=float(abs(entry - float(ema_support_15m)) / entry * 100.0 if entry > 0 else 999.0),
        pullback_ema_period=int(pb_ema_p),
        pullback_ema_dist_pct=float(pb_dist_pct),
        pullback_ready=bool(pullback_ready),
        pullback_bypass_hot=bool(pullback_bypass_hot),
        engine=str(engine),
        is_trailing_tp3=bool(hot),
        created_ts=time.time(),
    )








def pick_setups(
    best_fut: Dict[str, MarketVol],
    n: int,
    strict_15m: bool = True,
    session_name: str = "LON",
    universe_n: int = 35,
    trigger_loosen_mult: float = 1.0,
    waiting_near_pct: float = SCREEN_WAITING_NEAR_PCT,
    allow_no_pullback: bool = True,   # ✅ screen True, email False (except HOT bypass inside make_setup)
) -> List[Setup]:
    global _REJECT_STATS, _REJECT_SAMPLES, _REJECT_BY_SYMBOL, _WAITING_TRIGGER
    _REJECT_STATS = Counter()
    _REJECT_SAMPLES = {}
    _REJECT_BY_SYMBOL = {}
    _WAITING_TRIGGER = {}

    universe_n = int(max(10, universe_n))
    universe = sorted(best_fut.items(), key=lambda kv: usd_notional(kv[1]), reverse=True)[:universe_n]

    setups: List[Setup] = []
    for base, mv in universe:
        s = make_setup(
            base,
            mv,
            strict_15m=strict_15m,
            session_name=session_name,
            allow_no_pullback=allow_no_pullback,
            trigger_loosen_mult=float(trigger_loosen_mult),
            waiting_near_pct=float(waiting_near_pct),
        )
        if s:
            setups.append(s)

    setups.sort(key=lambda x: (x.conf, x.fut_vol_usd), reverse=True)
    return setups[:n]




# =========================================================
# EMAIL
# =========================================================
def email_config_ok() -> bool:
    return all([EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FROM, EMAIL_TO])

def send_email(subject: str, body: str, user_id_for_debug: Optional[int] = None) -> bool:
    """
    Sends email and stores last SMTP error per user (for /health).
    """
    if not email_config_ok():
        logger.warning("Email not configured.")
        if user_id_for_debug is not None:
            _LAST_SMTP_ERROR[int(user_id_for_debug)] = "Email not configured (missing env vars)."
        return False

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        msg.set_content(body)

        if EMAIL_PORT == 465:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=ctx, timeout=30) as s:
                s.login(EMAIL_USER, EMAIL_PASS)
                s.send_message(msg)
        else:
            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT, timeout=30) as s:
                s.ehlo()
                s.starttls()
                s.ehlo()
                s.login(EMAIL_USER, EMAIL_PASS)
                s.send_message(msg)

        if user_id_for_debug is not None:
            _LAST_SMTP_ERROR.pop(int(user_id_for_debug), None)
        return True

    except Exception as e:
        logger.exception("send_email failed: %s", e)
        if user_id_for_debug is not None:
            _LAST_SMTP_ERROR[int(user_id_for_debug)] = f"{type(e).__name__}: {str(e)}"
        return False

async def email_test_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /email_test
    Sends a test email NOW (admin only).
    """
    uid = update.effective_user.id
    if not is_admin_user(uid):
        await update.message.reply_text("⛔️ این دستور فقط برای Admin فعال است.")
        return

    if not EMAIL_ENABLED:
        await update.message.reply_text("⚠️ EMAIL_ENABLED=false است. اول فعالش کن.")
        return

    if not email_config_ok():
        await update.message.reply_text("⚠️ Email ENV ناقص است. EMAIL_HOST/PORT/USER/PASS/FROM/TO را چک کن.")
        return

    now = datetime.now(ZoneInfo("Australia/Melbourne"))
    subject = f"PulseFutures • EMAIL TEST • {now.strftime('%Y-%m-%d %H:%M')}"
    body = "This is a test email from PulseFutures.\n\nIf you received this, SMTP is OK."

    ok = await asyncio.to_thread(send_email, subject, body, uid)
    if ok:
        await update.message.reply_text("✅ Test email sent. Inbox/Spam را چک کن.")
    else:
        err = _LAST_SMTP_ERROR.get(uid, "unknown")
        await update.message.reply_text(f"❌ Test email failed.\nError: {err}")


# =========================================================
# SESSIONS (user)
# =========================================================
def user_enabled_sessions(user: dict) -> List[str]:
    try:
        xs = json.loads(user["sessions_enabled"])
        if isinstance(xs, list) and xs:
            ordered = _order_sessions(xs)
            return ordered or _default_sessions_for_tz(user["tz"])
    except Exception:
        return _default_sessions_for_tz(user["tz"])

def in_session_now(user: dict) -> Optional[dict]:
    tz = ZoneInfo(user["tz"])
    now_local = datetime.now(tz)
    now_utc = now_local.astimezone(timezone.utc)

    enabled = user_enabled_sessions(user)
    for name in enabled:
        w = SESSIONS_UTC[name]
        sh, sm = parse_hhmm(w["start"])
        eh, em = parse_hhmm(w["end"])
        start_utc = now_utc.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end_utc = now_utc.replace(hour=eh, minute=em, second=0, microsecond=0)

        if end_utc <= start_utc:
            end_utc += timedelta(days=1)

        if now_utc < start_utc and (start_utc - now_utc) > timedelta(hours=12):
            start_utc -= timedelta(days=1)
            end_utc -= timedelta(days=1)

        if start_utc <= now_utc <= end_utc:
            session_key = f"{start_utc.strftime('%Y-%m-%d')}_{name}"
            return {
                "name": name,
                "session_key": session_key,
                "start_utc": start_utc,
                "end_utc": end_utc,
                "now_local": now_local,
            }

    return None


# =========================================================
# RISK
# =========================================================
def compute_risk_usd(user: dict, mode: str, value: float) -> float:
    mode = mode.upper()
    if mode == "USD":
        return max(0.0, float(value))
    eq = float(user["equity"])
    if eq <= 0:
        return 0.0
    return max(0.0, eq * (float(value) / 100.0))

def calc_qty(entry: float, sl: float, risk_usd: float) -> float:
    d = abs(entry - sl)
    if entry <= 0 or d <= 0 or risk_usd <= 0:
        return 0.0
    return risk_usd / d

def daily_cap_usd(user: dict) -> float:
    mode = user["daily_cap_mode"].upper()
    val = float(user["daily_cap_value"])
    if mode == "USD":
        return max(0.0, val)
    eq = float(user["equity"])
    if eq <= 0:
        return 0.0
    return max(0.0, eq * (val / 100.0))


# =========================================================
# REPORTING / ADVICE
# =========================================================
def _stats_from_trades(trades: List[dict]) -> dict:
    closed = [t for t in trades if t.get("closed_ts") is not None and t.get("pnl") is not None]
    wins = [t for t in closed if float(t["pnl"]) > 0]
    losses = [t for t in closed if float(t["pnl"]) < 0]
    net = sum(float(t["pnl"]) for t in closed) if closed else 0.0
    win_rate = (len(wins) / len(closed) * 100.0) if closed else 0.0
    avg_r = None
    r_vals = [float(t["r_mult"]) for t in closed if t.get("r_mult") is not None]
    if r_vals:
        avg_r = sum(r_vals) / len(r_vals)
    biggest_loss = min((float(t["pnl"]) for t in closed), default=0.0)
    biggest_win = max((float(t["pnl"]) for t in closed), default=0.0)
    return {
        "closed_n": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "net": net,
        "win_rate": win_rate,
        "avg_r": avg_r,
        "biggest_loss": biggest_loss,
        "biggest_win": biggest_win,
    }

def _advice(user: dict, stats: dict) -> List[str]:
    adv = []
    if stats["closed_n"] < 5:
        adv.append("📌 Not enough data yet: close at least 5–10 trades for meaningful statistics.")
    if stats["win_rate"] < 45 and stats["closed_n"] >= 5:
        adv.append("⚠️ Low win rate detected: reduce trade frequency and only take high-confidence setups.")
    if stats["avg_r"] is not None and stats["avg_r"] < 0.2 and stats["closed_n"] >= 5:
        adv.append("⚠️ Low R-multiple: stop-loss may be too tight or profits are taken too early.")
    if stats["biggest_loss"] < -0.9 * max(1.0, float(user["equity"]) * 0.01):
        adv.append("🛑 A large loss detected: review stop-loss execution and slippage/re-entry discipline.")
    if int(user["max_trades_day"]) <= 5:
        adv.append("✅ Daily trade limits help prevent overtrading. Focus only on top-quality setups.")
    return adv[:6]



# =========================================================
# HELP TEXT
# =========================================================
HELP_TEXT = """\
PulseFutures — Commands (Telegram)

1) Market Scan
- /screen
  Shows:
  • Top Trade Setups (best quality)
  • Directional Leaders/Losers (24H ±10% with futures vol >= $5M)
  • Market Leaders by futures volume
  • Reject Diagnostics (professional explanation)

2) Position Sizing (Risk + SL => Qty)
- /size <SYMBOL> <long|short> sl <STOP> [risk <usd|pct> <VALUE>] [entry <ENTRY>]

Examples:
- /size BTC long sl 42000
  → Default risk = 2% of Equity (safer). If your equity is 0, set it first:
    /equity 1000

- /size BTC long risk usd 40 sl 42000
  → Bot uses current Bybit futures price as Entry and returns Qty for $40 risk.

- /size ETH short risk pct 2.5 sl 2480
  → Uses Equity. If your equity is 0, set it first:
    /equity 1000

Manual entry examples:
- /size BTC long sl 42000 entry 43000
- /size BTC long risk usd 50 sl 42000 entry 43000

Notes:
- If you do NOT specify "risk", the bot uses 2% of your Equity by default.
- If you specify risk above 2% of Equity, the bot will warn you.
- pct uses your Equity
- Qty = RiskUSD / |Entry - SL|
- This command does NOT open a trade.

3) Trade Journal (Open / Close) + Equity auto-update
Set equity:
- /equity 1000

Open trade:
- /trade_open <SYMBOL> <long|short> entry <ENTRY> sl <SL> risk <usd|pct> <VALUE> [note "..."] [sig <SETUP_ID>]

Close trade:
- /trade_close <TRADE_ID> pnl <PNL>

Equity behavior:
- Equity updates ONLY when trades are closed
- Persistent until reset:
  /equity_reset

4) Status
- /status

5) Risk Settings
- /riskmode pct 2.5
- /riskmode usd 25
- /dailycap pct 5
- /dailycap usd 60
- /limits maxtrades 5
- /limits emailcap 4        (0 = unlimited)
- /limits emailgap 60
- /limits emaildaycap 4     (0 = unlimited)

6) Sessions (Emails by session)
Default by timezone:
- Americas → NY
- Europe/Africa → LON
- Asia/Oceania → ASIA

Session priority:
NY > LON > ASIA

Commands:
- /sessions
- /sessions_on NY
- /sessions_off LON

7) Email Alerts
- /notify_on
- /notify_off

Email rules:
- Sent only during enabled sessions
- Session-based quality filters
- No same symbol for 18h
- Daily email cap supported

8) Performance Reports
- /report_daily
- /report_weekly

9) Signal Reports
- /signals_daily
- /signals_weekly

10) Health (Transparent)
- /health
Shows engine layer status, current session knobs, and last email decision.

Not financial advice.
"""

# =========================================================
# TELEGRAM COMMANDS
# =========================================================
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)

    sess_now = in_session_now(user)
    sess_name = sess_now["name"] if sess_now else current_session_utc()
    knobs = session_knobs(sess_name)

    last_dec = _LAST_EMAIL_DECISION.get(uid, {})
    diag_mode = user_diag_mode(uid)

    last_lines = []
    if last_dec:
        last_lines.append(f"Last email decision: {last_dec.get('status','-')}")
        if last_dec.get("status") == "SKIP":
            reasons = last_dec.get("reasons", [])
            if reasons:
                last_lines.append("Top skip reasons:")
                for r in reasons[:6]:
                    last_lines.append(f"- {r}")
        if last_dec.get("when"):
            last_lines.append(f"When: {last_dec.get('when')}")

    prox_mode = f"Session-adaptive ({knobs['name']}: mult={knobs['ema_prox_mult']:.2f})"
    react_mode = f"Session-adaptive ({knobs['name']}: lookback={knobs['ema_reaction_lookback']})"
    trig_mode = f"ATR-adaptive ({knobs['name']}: mult={knobs['trigger_atr_mult']:.2f})"

    rr_floor = SESSION_MIN_RR_TP3.get(knobs["name"], 2.0)
    conf_floor = SESSION_MIN_CONF.get(knobs["name"], 78)

    smtp_err = _LAST_SMTP_ERROR.get(uid, "")

    msg = [
        "🫀 PulseFutures Health Check",
        HDR,
        f"User TZ: {user['tz']}",
        f"Session (user-enabled): {sess_name}",
        f"Diagnostics mode: {('FULL' if is_admin_user(uid) else diag_mode)}",
        "",
        "Layer / Status",
        SEP,
        f"EMA12 Proximity: ✅ {prox_mode}",
        f"EMA12 Reaction: ✅ {react_mode}",
        f"1H Trigger: ✅ {trig_mode}",
        f"Confidence floors: ✅ Session-based (min={conf_floor})",
        f"RR floors: ✅ Session-based (minRR={rr_floor:.2f})",
        f"TP scaling: ✅ Confidence-weighted",
        "",
        f"Email Engine: {'ACTIVE' if EMAIL_ENABLED and email_config_ok() else 'OFF/NOT CONFIGURED'}",
        f"SMTP last error: {smtp_err if smtp_err else '-'}",
        f"Scanner cache: tickers_ttl={TICKERS_TTL_SEC}s ohlcv_ttl={OHLCV_TTL_SEC}s",
        f"Reject Diagnostics: {'ON' if diag_mode != 'off' or is_admin_user(uid) else 'OFF'} | Samples: {'ON' if DEBUG_REJECTS else 'OFF'}",
        HDR,
    ]
    if last_lines:
        msg.append("Last Email Details")
        msg.append(SEP)
        msg.extend(last_lines)
        msg.append(HDR)

    await update.message.reply_text("\n".join(msg).strip())


async def diag_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /diag_on [friendly|off]
    Non-admin can only choose friendly/off. Admin always full.
    """
    uid = update.effective_user.id
    if is_admin_user(uid):
        _USER_DIAG_MODE[uid] = "full"
        await update.message.reply_text("✅ Diagnostics: FULL (admin).")
        return

    mode = (context.args[0].strip().lower() if context.args else "friendly")
    if mode not in {"friendly", "off"}:
        await update.message.reply_text("Usage: /diag_on friendly  OR  /diag_on off")
        return
    _USER_DIAG_MODE[uid] = mode
    await update.message.reply_text(f"✅ Diagnostics set to: {mode}")

async def diag_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /diag_off  -> hides diagnostics for this user (non-admin).
    """
    uid = update.effective_user.id
    if is_admin_user(uid):
        await update.message.reply_text("✅ Admin diagnostics همیشه FULL است.")
        return
    _USER_DIAG_MODE[uid] = "off"
    await update.message.reply_text("✅ Diagnostics: OFF")

async def tz_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if not context.args:
        await update.message.reply_text(f"Your TZ: {user['tz']}\nSet: /tz Australia/Melbourne")
        return
    tz_name = " ".join(context.args).strip()
    try:
        ZoneInfo(tz_name)
    except Exception:
        await update.message.reply_text("Invalid TZ. Example: /tz Australia/Melbourne  or  /tz America/New_York")
        return
    sessions = _order_sessions(_default_sessions_for_tz(tz_name)) or _default_sessions_for_tz(tz_name)
    update_user(uid, tz=tz_name, sessions_enabled=json.dumps(sessions))
    await update.message.reply_text(f"✅ TZ set to {tz_name}\nDefault sessions updated. Use /sessions to view.")

async def equity_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if not context.args:
        await update.message.reply_text(f"Equity: ${float(user['equity']):.2f}")
        return
    try:
        eq = float(context.args[0])
        if eq < 0:
            raise ValueError()
        update_user(uid, equity=eq)
        await update.message.reply_text(f"✅ Equity set: ${eq:.2f}")
    except Exception:
        await update.message.reply_text("Usage: /equity 1000")

async def equity_reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    update_user(uid, equity=0.0)
    await update.message.reply_text("✅ Equity reset to $0.00")

async def riskmode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if len(context.args) != 2:
        await update.message.reply_text(f"Current: {user['risk_mode']} {float(user['risk_value']):.2f}\nUsage: /riskmode pct 2.5  OR  /riskmode usd 25")
        return
    mode = context.args[0].strip().upper()
    try:
        val = float(context.args[1])
    except Exception:
        await update.message.reply_text("Usage: /riskmode pct 2.5  OR  /riskmode usd 25")
        return
    if mode not in {"PCT", "USD"}:
        await update.message.reply_text("Mode must be pct or usd")
        return
    if mode == "PCT" and not (0.1 <= val <= 10):
        await update.message.reply_text("pct value should be between 0.1 and 10")
        return
    if mode == "USD" and val <= 0:
        await update.message.reply_text("usd value must be > 0")
        return
    update_user(uid, risk_mode=mode, risk_value=val)
    await update.message.reply_text(f"✅ Risk mode updated: {mode} {val:.2f}")

async def dailycap_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if len(context.args) != 2:
        cap = daily_cap_usd(user)
        await update.message.reply_text(f"Current: {user['daily_cap_mode']} {float(user['daily_cap_value']):.2f} (≈ ${cap:.2f})\nUsage: /dailycap pct 5  OR  /dailycap usd 60")
        return
    mode = context.args[0].strip().upper()
    try:
        val = float(context.args[1])
    except Exception:
        await update.message.reply_text("Usage: /dailycap pct 5  OR  /dailycap usd 60")
        return
    if mode not in {"PCT", "USD"}:
        await update.message.reply_text("Mode must be pct or usd")
        return
    if mode == "PCT" and not (0.5 <= val <= 30):
        await update.message.reply_text("pct/day should be between 0.5 and 30")
        return
    if mode == "USD" and val < 0:
        await update.message.reply_text("usd/day must be >= 0")
        return
    update_user(uid, daily_cap_mode=mode, daily_cap_value=val)
    user = get_user(uid)
    await update.message.reply_text(f"✅ Daily cap updated. (≈ ${daily_cap_usd(user):.2f})")

async def limits_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)

    if len(context.args) < 2:
        await update.message.reply_text(
            f"Limits:\n"
            f"- maxtrades: {int(user['max_trades_day'])}\n"
            f"- emailcap: {int(user['max_emails_per_session'])} (0 = unlimited)\n"
            f"- emailgap: {int(user['email_gap_min'])} min\n"
            f"- emaildaycap: {int(user.get('max_emails_per_day', DEFAULT_MAX_EMAILS_PER_DAY))} (0 = unlimited)\n\n"
            f"Set examples:\n"
            f"/limits maxtrades 5\n"
            f"/limits emailcap 0\n"
            f"/limits emailgap 60\n"
            f"/limits emaildaycap 0\n"
        )
        return

    key = context.args[0].strip().lower()
    try:
        val = int(float(context.args[1]))
    except Exception:
        await update.message.reply_text("Value must be a number.")
        return

    if key == "maxtrades":
        if not (1 <= val <= 50):
            await update.message.reply_text("maxtrades must be 1..50")
            return
        update_user(uid, max_trades_day=val)
        await update.message.reply_text(f"✅ maxtrades/day set to {val}")

    elif key == "emailcap":
        if not (0 <= val <= 50):
            await update.message.reply_text("emailcap must be 0..50 (0 = unlimited)")
            return
        update_user(uid, max_emails_per_session=val)
        await update.message.reply_text(f"✅ emailcap/session set to {val} (0 = unlimited)")

    elif key == "emailgap":
        if not (0 <= val <= 360):
            await update.message.reply_text("emailgap must be 0..360 minutes")
            return
        update_user(uid, email_gap_min=val)
        await update.message.reply_text(f"✅ emailgap set to {val} minutes")

    elif key in {"emaildaycap", "dailyemailcap", "emailcapday"}:
        if not (0 <= val <= 50):
            await update.message.reply_text("emaildaycap must be 0..50 (0 = unlimited)")
            return
        update_user(uid, max_emails_per_day=val)
        await update.message.reply_text(f"✅ daily email cap set to {val} (0 = unlimited)")

    else:
        await update.message.reply_text("Unknown key. Use: maxtrades | emailcap | emailgap | emaildaycap")

async def sessions_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    enabled = user_enabled_sessions(user)
    now_s = in_session_now(user)
    now_txt = f"Now in session: {now_s['name']}" if now_s else "Now in session: NONE"
    await update.message.reply_text(
        f"Your TZ: {user['tz']}\n"
        f"{now_txt}\n\n"
        f"Enabled sessions (priority): {', '.join(enabled)}\n"
        f"Available: ASIA, LON, NY\n\n"
        f"Enable: /sessions_on NY\n"
        f"Disable: /sessions_off LON"
    )

async def sessions_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if not context.args:
        await update.message.reply_text("Usage: /sessions_on NY")
        return
    name = context.args[0].strip().upper()
    if name not in SESSIONS_UTC:
        await update.message.reply_text("Session must be one of: ASIA, LON, NY")
        return

    enabled = user_enabled_sessions(user)
    if name not in enabled:
        enabled.append(name)

    enabled = _order_sessions(enabled) or enabled
    update_user(uid, sessions_enabled=json.dumps(enabled))
    await update.message.reply_text(f"✅ Enabled sessions: {', '.join(enabled)}")

async def sessions_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if not context.args:
        await update.message.reply_text("Usage: /sessions_off LON")
        return
    name = context.args[0].strip().upper()

    enabled = [s for s in user_enabled_sessions(user) if s != name]
    if not enabled:
        enabled = _default_sessions_for_tz(user["tz"])

    enabled = _order_sessions(enabled) or enabled
    update_user(uid, sessions_enabled=json.dumps(enabled))
    await update.message.reply_text(f"✅ Enabled sessions: {', '.join(enabled)}")

async def notify_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    update_user(uid, notify_on=1)
    await update.message.reply_text("✅ Email alerts: ON")

async def notify_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    update_user(uid, notify_on=0)
    await update.message.reply_text("✅ Email alerts: OFF")

def _equity_risk_pct_from_usd(user: dict, risk_usd: float) -> Optional[float]:
    eq = float(user.get("equity", 0.0) or 0.0)
    if eq <= 0:
        return None
    return (float(risk_usd) / eq) * 100.0

async def size_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)

    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text("Usage: /size BTC long sl 42000  (optional: risk pct 2 | risk usd 40 | entry 43000)")
        return

    tokens = raw.split()
    if len(tokens) < 4:
        await update.message.reply_text("Usage: /size BTC long sl 42000  (optional: risk pct 2 | risk usd 40 | entry 43000)")
        return

    sym = re.sub(r"[^A-Za-z0-9]", "", tokens[0]).upper()
    direction = tokens[1].lower()
    if direction not in {"long", "short"}:
        await update.message.reply_text("Second arg must be long or short.")
        return
    side = "BUY" if direction == "long" else "SELL"

    if "sl" not in tokens:
        await update.message.reply_text("Missing SL. Example: /size BTC long sl 42000")
        return
    try:
        sl_i = tokens.index("sl")
        sl = float(tokens[sl_i + 1])
    except Exception:
        await update.message.reply_text("Bad SL format. Example: /size BTC long sl 42000")
        return

    entry = None
    if "entry" in tokens:
        try:
            e_i = tokens.index("entry")
            entry = float(tokens[e_i + 1])
        except Exception:
            await update.message.reply_text("Bad entry format. Example: entry 43000")
            return

    risk_mode = None
    risk_val = None
    if "risk" in tokens:
        try:
            r_i = tokens.index("risk")
            risk_mode = tokens[r_i + 1].upper()
            risk_val = float(tokens[r_i + 2])
        except Exception:
            await update.message.reply_text("Bad risk format. Example: risk pct 2  OR  risk usd 40")
            return
        if risk_mode not in {"USD", "PCT"}:
            await update.message.reply_text("risk mode must be usd or pct")
            return
    else:
        # Use user's configured defaults
        risk_mode = str(user.get("risk_mode", "PCT")).upper()
        risk_val = float(user.get("risk_value", DEFAULT_RISK_VALUE))

    if entry is None:
        best = await asyncio.to_thread(fetch_futures_tickers)
        mv = best.get(sym)
        if not mv or float(mv.last or 0) <= 0:
            await update.message.reply_text(f"Could not fetch price for {sym}. Provide entry manually: entry 43000")
            return
        entry = float(mv.last)

    if entry <= 0 or sl <= 0 or entry == sl:
        await update.message.reply_text("Entry/SL invalid. Ensure entry and sl are positive and not equal.")
        return

    if side == "BUY" and sl >= entry:
        await update.message.reply_text("For LONG, SL should be BELOW entry.")
        return
    if side == "SELL" and sl <= entry:
        await update.message.reply_text("For SHORT, SL should be ABOVE entry.")
        return

    risk_usd = compute_risk_usd(user, risk_mode, risk_val)

    if risk_usd <= 0:
        if risk_mode == "PCT":
            await update.message.reply_text("Risk computed as $0. Set your equity first: /equity 1000")
            return
        await update.message.reply_text("Risk USD computed as 0. Check your risk value.")
        return

    qty = calc_qty(entry, sl, risk_usd)
    if qty <= 0:
        await update.message.reply_text("Could not compute qty. Check entry/sl/risk.")
        return

    warn = ""
    if risk_mode == "PCT":
        if float(risk_val) > float(WARN_RISK_PCT_PER_TRADE):
            warn = (
                f"⚠️ Warning: You are risking {float(risk_val):.2f}% of equity.\n"
                f"Recommended max risk per trade is {WARN_RISK_PCT_PER_TRADE:.2f}%."
            )
    else:
        pct_equiv = _equity_risk_pct_from_usd(user, risk_usd)
        if pct_equiv is not None and pct_equiv > float(WARN_RISK_PCT_PER_TRADE):
            warn = (
                f"⚠️ Warning: This equals ~{pct_equiv:.2f}% of equity.\n"
                f"Recommended max risk per trade is {WARN_RISK_PCT_PER_TRADE:.2f}%."
            )

    msg = (
        f"✅ Position Size\n"
        f"- Symbol: {sym}\n"
        f"- Side: {side}\n"
        f"- Entry: {fmt_price(entry)}\n"
        f"- SL: {fmt_price(sl)}\n"
        f"- Risk: ${risk_usd:.2f}\n"
        f"- Qty: {qty:.6g}\n"
    )
    if warn:
        msg += "\n" + warn + "\n"

    msg += f"\nChart: {tv_chart_url(sym)}"
    await update.message.reply_text(msg)

async def trade_open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = reset_daily_if_needed(get_user(uid))

    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text("Usage: /trade_open BTC long entry 43000 sl 42000 risk usd 40 [note ...] [sig PF-...]")
        return

    tokens = raw.split()
    if len(tokens) < 9:
        await update.message.reply_text("Usage: /trade_open BTC long entry 43000 sl 42000 risk usd 40 [note ...] [sig PF-...]")
        return

    sym = re.sub(r"[^A-Za-z0-9]", "", tokens[0]).upper()
    direction = tokens[1].lower()
    if direction not in {"long", "short"}:
        await update.message.reply_text("Second arg must be long or short.")
        return
    side = "BUY" if direction == "long" else "SELL"

    if int(user["day_trade_count"]) >= int(user["max_trades_day"]):
        await update.message.reply_text(f"⚠️ Max trades/day reached ({user['max_trades_day']}). If you continue, you are overtrading.")
        return

    try:
        def idx(x): return tokens.index(x)
        entry = float(tokens[idx("entry")+1])
        sl = float(tokens[idx("sl")+1])

        r_i = idx("risk")
        risk_mode = tokens[r_i+1].upper()
        risk_val = float(tokens[r_i+2])

        note = ""
        signal_id = ""
        if "note" in tokens:
            n_i = idx("note")
            note = " ".join(tokens[n_i+1:])
        if "sig" in tokens:
            s_i = idx("sig")
            signal_id = tokens[s_i+1].strip()
            if " sig " in f" {note} ":
                note = note.split(" sig ")[0].strip()
    except Exception:
        await update.message.reply_text("Bad format. Example: /trade_open BTC long entry 43000 sl 42000 risk usd 40 note breakout sig PF-20251219-0007")
        return

    if risk_mode not in {"USD", "PCT"}:
        await update.message.reply_text("risk mode must be usd or pct")
        return

    if side == "BUY" and sl >= entry:
        await update.message.reply_text("For LONG, SL should be BELOW entry.")
        return
    if side == "SELL" and sl <= entry:
        await update.message.reply_text("For SHORT, SL should be ABOVE entry.")
        return

    risk_usd = compute_risk_usd(user, risk_mode, risk_val)
    if risk_usd <= 0:
        await update.message.reply_text("Risk USD computed as 0. If you used pct, set your equity first: /equity 1000")
        return

    qty = calc_qty(entry, sl, risk_usd)
    if qty <= 0:
        await update.message.reply_text("Could not compute qty. Check entry/sl/risk.")
        return

    cap = daily_cap_usd(user)
    day_local = _user_day_local(user)
    used_before = _risk_daily_get(uid, day_local)
    remaining_before = (cap - used_before) if cap > 0 else float("inf")

    warn_daily = ""
    if cap > 0 and float(risk_usd) > max(0.0, remaining_before) + 1e-9:
        warn_daily = (
            f"⚠️ Daily Risk Warning: "
            f"This position (${risk_usd:.2f}) exceeds today's remaining risk "
            f"(${max(0.0, remaining_before):.2f}).\n"
        )

    warn_trade_vs_cap = ""
    if cap > 0 and float(risk_usd) > float(cap) + 1e-9:
        warn_trade_vs_cap = f"⚠️ Note: This trade risk (${risk_usd:.2f}) exceeds the total Daily Risk Cap (${cap:.2f}).\n"

    tid = db_trade_open(uid, sym, side, entry, sl, risk_usd, qty, note=note, signal_id=signal_id)
    _risk_daily_inc(uid, day_local, float(risk_usd))

    used_after = _risk_daily_get(uid, day_local)
    remaining_after = (cap - used_after) if cap > 0 else float("inf")

    update_user(uid, day_trade_count=int(user["day_trade_count"]) + 1)
    user = get_user(uid)

    daily_risk_line = (
        f"- Daily risk limit: {user['daily_cap_mode']} {float(user['daily_cap_value']):.2f} (≈ ${cap:.2f})\n"
        f"- Used today: ${used_after:.2f}\n"
        f"- Remaining today: ${max(0.0, remaining_after):.2f}" if cap > 0 else
        f"- Daily risk limit: {user['daily_cap_mode']} {float(user['daily_cap_value']):.2f} (≈ ${cap:.2f})\n"
        f"- Used today: ${used_after:.2f}\n"
        f"- Remaining today: ∞"
    )

    await update.message.reply_text(
        f"✅ Trade OPENED (Journal)\n"
        f"- ID: {tid}\n"
        f"- {sym} {side}\n"
        f"- Entry: {fmt_price(entry)}\n"
        f"- SL: {fmt_price(sl)}\n"
        f"- Risk: ${risk_usd:.2f}\n"
        f"- Qty: {qty:.6g}\n"
        f"{warn_daily}{warn_trade_vs_cap}"
        f"{daily_risk_line}\n"
        f"- Trades today: {int(user['day_trade_count'])}/{int(user['max_trades_day'])}\n"
        f"- Equity: ${float(user['equity']):.2f}\n"
        f"- Signal: {signal_id if signal_id else '-'}\n"
        f"- Note: {note if note else '-'}"
    )

async def trade_close_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text("Usage: /trade_close <TRADE_ID> pnl <PNL>")
        return

    tokens = raw.split()
    if len(tokens) < 3:
        await update.message.reply_text("Usage: /trade_close <TRADE_ID> pnl <PNL>")
        return

    try:
        trade_id = int(tokens[0])
        if tokens[1].lower() != "pnl":
            raise ValueError()
        pnl = float(tokens[2])
    except Exception:
        await update.message.reply_text("Usage: /trade_close <TRADE_ID> pnl <PNL>  (example: /trade_close 12 pnl +85.5)")
        return

    t = db_trade_close(uid, trade_id, pnl)
    if not t:
        await update.message.reply_text("Trade not found or already closed.")
        return

    new_eq = float(user["equity"]) + float(pnl)
    update_user(uid, equity=new_eq)
    user = get_user(uid)

    r_mult = t.get("r_mult")
    r_txt = f"{r_mult:+.2f}R" if r_mult is not None else "-"

    await update.message.reply_text(
        f"✅ Trade CLOSED\n"
        f"- ID: {trade_id}\n"
        f"- PnL: {float(pnl):+.2f}\n"
        f"- R: {r_txt}\n"
        f"- New Equity: ${float(user['equity']):.2f}"
    )

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = reset_daily_if_needed(get_user(uid))
    opens = db_open_trades(uid)

    cap = daily_cap_usd(user)
    day_local = _user_day_local(user)
    used_today = _risk_daily_get(uid, day_local)
    remaining_today = (cap - used_today) if cap > 0 else float("inf")

    enabled = user_enabled_sessions(user)
    now_s = in_session_now(user)
    now_txt = now_s["name"] if now_s else "NONE"

    lines = []
    lines.append("📌 Status")
    lines.append(HDR)
    lines.append(f"Equity: ${float(user['equity']):.2f}")
    lines.append(f"Trades today: {int(user['day_trade_count'])}/{int(user['max_trades_day'])}")

    lines.append(f"Daily cap: {user['daily_cap_mode']} {float(user['daily_cap_value']):.2f} (≈ ${cap:.2f})")
    lines.append(f"Daily risk used: ${used_today:.2f}")
    lines.append(f"Daily risk remaining: ${max(0.0, remaining_today):.2f}" if cap > 0 else "Daily risk remaining: ∞")

    lines.append(f"Email alerts: {'ON' if int(user['notify_on'])==1 else 'OFF'}")
    lines.append(f"Sessions enabled: {', '.join(enabled)} | Now: {now_txt}")
    lines.append(f"Email caps: session={int(user['max_emails_per_session'])} (0=∞), day={int(user.get('max_emails_per_day', DEFAULT_MAX_EMAILS_PER_DAY))} (0=∞), gap={int(user['email_gap_min'])}m")
    lines.append(HDR)

    if not opens:
        lines.append("Open trades: None")
        await update.message.reply_text("\n".join(lines))
        return

    lines.append("Open trades:")
    for t in opens:
        lines.append(
            f"- ID {t['id']} | {t['symbol']} {t['side']} | Entry {fmt_price(float(t['entry']))} | "
            f"SL {fmt_price(float(t['sl']))} | Risk ${float(t['risk_usd']):.2f} | Qty {float(t['qty']):.6g}"
        )
    await update.message.reply_text("\n".join(lines))

async def report_daily_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    tz = ZoneInfo(user["tz"])
    start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    trades = db_trades_since(uid, start)
    stats = _stats_from_trades(trades)
    adv = _advice(user, stats)

    msg = [
        "📊 Daily Report",
        HDR,
        f"Closed: {stats['closed_n']} | Wins: {stats['wins']} | Losses: {stats['losses']}",
        f"Win rate: {stats['win_rate']:.1f}%",
        f"Net PnL: {stats['net']:+.2f}",
        f"Avg R: {stats['avg_r']:+.2f}" if stats["avg_r"] is not None else "Avg R: -",
        f"Best: {stats['biggest_win']:+.2f} | Worst: {stats['biggest_loss']:+.2f}",
        HDR,
    ]
    if adv:
        msg.append("🧠 Advice:")
        msg.extend([f"- {x}" for x in adv])

    await update.message.reply_text("\n".join(msg))

async def report_weekly_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    tz = ZoneInfo(user["tz"])
    now = datetime.now(tz)
    start_dt = now - timedelta(days=7)
    ts_from = start_dt.timestamp()

    trades = db_trades_since(uid, ts_from)
    stats = _stats_from_trades(trades)
    adv = _advice(user, stats)

    msg = [
        "📊 Weekly Report (last 7 days)",
        HDR,
        f"Closed: {stats['closed_n']} | Wins: {stats['wins']} | Losses: {stats['losses']}",
        f"Win rate: {stats['win_rate']:.1f}%",
        f"Net PnL: {stats['net']:+.2f}",
        f"Avg R: {stats['avg_r']:+.2f}" if stats["avg_r"] is not None else "Avg R: -",
        f"Best: {stats['biggest_win']:+.2f} | Worst: {stats['biggest_loss']:+.2f}",
        HDR,
    ]
    if adv:
        msg.append("🧠 Advice:")
        msg.extend([f"- {x}" for x in adv])

    await update.message.reply_text("\n".join(msg))

async def signals_daily_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    tz = ZoneInfo(user["tz"])
    start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc).timestamp()
    sigs = db_list_signals_since(start)

    msg = ["📮 Signals Daily Summary (generated)"]
    msg.append(HDR)
    msg.append(f"Total setups generated today: {len(sigs)}")
    msg.append(HDR)

    trades = db_trades_since(uid, start)
    linked = [t for t in trades if t.get("signal_id")]
    closed_linked = [t for t in linked if t.get("closed_ts") and t.get("pnl") is not None]

    if closed_linked:
        win = len([t for t in closed_linked if float(t["pnl"]) > 0])
        loss = len([t for t in closed_linked if float(t["pnl"]) < 0])
        net = sum(float(t["pnl"]) for t in closed_linked)
        msg.append(f"Your linked signal trades closed today: {len(closed_linked)} | W {win} / L {loss} | Net {net:+.2f}")
    else:
        msg.append("Your linked signal trades closed today: 0")

    await update.message.reply_text("\n".join(msg))

async def signals_weekly_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    tz = ZoneInfo(user["tz"])
    start_dt = datetime.now(tz) - timedelta(days=7)
    start = start_dt.astimezone(timezone.utc).timestamp()

    sigs = db_list_signals_since(start)
    msg = ["📮 Signals Weekly Summary (last 7 days)"]
    msg.append(HDR)
    msg.append(f"Total setups generated: {len(sigs)}")
    msg.append(HDR)

    trades = db_trades_since(uid, start)
    linked = [t for t in trades if t.get("signal_id")]
    closed_linked = [t for t in linked if t.get("closed_ts") and t.get("pnl") is not None]

    if closed_linked:
        win = len([t for t in closed_linked if float(t["pnl"]) > 0])
        loss = len([t for t in closed_linked if float(t["pnl"]) < 0])
        net = sum(float(t["pnl"]) for t in closed_linked)
        wr = (win / len(closed_linked) * 100.0) if closed_linked else 0.0
        msg.append(f"Your linked signal trades closed: {len(closed_linked)} | WinRate {wr:.1f}% | Net {net:+.2f}")
    else:
        msg.append("Your linked signal trades closed: 0")

    await update.message.reply_text("\n".join(msg))



# =========================================================
# TREND ENGINE — ADAPTIVE EMA (NO FIXED EMA7/21/50)
# =========================================================

TREND_EMA_PULLBACK_ATR = 1.2
TREND_MAX_CONFIDENCE = 90

def ema_series(values, period):
    if not values or len(values) < period:
        return []
    k = 2.0 / (period + 1.0)
    ema_vals = [values[0]]
    for v in values[1:]:
        ema_vals.append(v * k + ema_vals[-1] * (1 - k))
    return ema_vals

def ema_slope(series, n=3):
    if len(series) < n + 1:
        return 0.0
    return series[-1] - series[-1 - n]

def trend_dynamic_confidence(base_conf, price, ema_fast_last, atr_15m, ch24_pct, reclaimed):
    score = base_conf
    if reclaimed:
        score += 8

    dist_atr = abs(price - ema_fast_last) / atr_15m
    if dist_atr < 0.4:
        score += 8
    elif dist_atr < 0.8:
        score += 4

    score += min(10, abs(ch24_pct) / 2)
    return int(min(score, TREND_MAX_CONFIDENCE))

def trend_watch_for_symbol(base, mv, session_name):
    try:
        c4h = fetch_ohlcv(mv.symbol, "4h", 60)
        c15 = fetch_ohlcv(mv.symbol, "15m", 140)
        if not c4h or not c15 or len(c15) < 40:
            return None

        closes_4h = [float(x[4]) for x in c4h]
        closes_15 = [float(x[4]) for x in c15]

        ch24 = float(mv.percentage or 0.0)
        side = "BUY" if ch24 > 0 else "SELL"

        # volatility proxy for adaptive EMA
        atr_15m = compute_atr_from_ohlcv(c15, ATR_PERIOD)
        price = float(mv.last or closes_15[-1])
        atr_pct = (atr_15m / price) * 100.0 if (atr_15m and price) else 2.0

        # adaptive EMA periods
        p_fast = adaptive_ema_period(atr_pct)          # 8..21
        p_slow = int(clamp(p_fast * 2, 16, 55))        # adaptive slow

        ema_fast = ema_series(closes_15[-120:], p_fast)
        ema_slow = ema_series(closes_15[-120:], p_slow)
        if not ema_fast or not ema_slow:
            return None

        if side == "BUY":
            if not (ema_fast[-1] > ema_slow[-1]):
                return None
            if ema_slope(ema_fast, 3) <= 0:
                return None
        else:
            if not (ema_fast[-1] < ema_slow[-1]):
                return None
            if ema_slope(ema_fast, 3) >= 0:
                return None

        dist_atr = abs(price - ema_fast[-1]) / atr_15m if atr_15m > 0 else 999
        if dist_atr > TREND_EMA_PULLBACK_ATR:
            return None

        reclaimed = False
        if side == "BUY":
            reclaimed = closes_15[-2] < ema_fast[-2] and price > ema_fast[-1]
        else:
            reclaimed = closes_15[-2] > ema_fast[-2] and price < ema_fast[-1]

        fut_vol = usd_notional(mv)
        base_conf = 55 + (15 if fut_vol >= MOVER_VOL_USD_MIN else 0)

        conf = trend_dynamic_confidence(base_conf, price, ema_fast[-1], atr_15m, ch24, reclaimed)

        return {"symbol": base, "side": side, "confidence": conf, "ch24": ch24}

    except Exception:
        return None



# =========================================================
# /screen — Premium Telegram UI
# =========================================================
async def screen_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # quick UX response
        await update.message.reply_text("⏳ Scanning market… Please wait")

        # Reset per-run trackers
        reset_reject_tracker()

        best_fut = await asyncio.to_thread(fetch_futures_tickers)
        if not best_fut:
            await update.message.reply_text("❌ Failed to fetch futures data.")
            return

        # -------------------------------------------------
        # Header / session
        # -------------------------------------------------
        uid = update.effective_user.id
        user = get_user(uid)

        session = current_session_utc()
        now_mel = datetime.now(ZoneInfo("Australia/Melbourne")).strftime("%Y-%m-%d %H:%M")

        header = (
            f"✨ *PulseFutures — Market Scan*\n"
            f"{HDR}\n"
            f"🧠 *Session:* `{session}`   |   🕒 *Melbourne:* `{now_mel}`\n"
            f"📦 *Universe:* `{len(best_fut)}` tickers\n"
            f"{HDR}"
        )

        # -------------------------------------------------
        # Main setups (screen-loosened rules)
        # -------------------------------------------------
        setups = await asyncio.to_thread(
            pick_setups,
            best_fut,
            SETUPS_N,
            False,                 # strict_15m = False for screen UX
            session,
            SCREEN_UNIVERSE_N,
            SCREEN_TRIGGER_LOOSEN,
            SCREEN_WAITING_NEAR_PCT,
            True,                  # allow_no_pullback for screen awareness
        )

        for s in setups:
            db_insert_signal(s)

        # -------------------------------------------------
        # Setup cards
        # -------------------------------------------------
        if setups:
            cards = []
            for i, s in enumerate(setups, 1):
                side_emoji = "🟢" if s.side == "BUY" else "🔴"
                engine_tag = "⚡️ Momentum" if s.engine == "B" else "🎯 Pullback"
                rr3 = rr_to_tp(s.entry, s.sl, s.tp3)

                pullback = (
                    "🔥 Bypass (Hot)"
                    if s.pullback_bypass_hot
                    else f"EMA{s.pullback_ema_period} ({s.pullback_ema_dist_pct:.2f}%)"
                    if s.pullback_ready
                    else f"⏳ Waiting EMA{s.pullback_ema_period}"
                )

                tp_line = (
                    f"*TP1:* `{fmt_price(s.tp1)}`  |  "
                    f"*TP2:* `{fmt_price(s.tp2)}`  |  "
                    f"*TP3:* `{fmt_price(s.tp3)}`"
                    if s.tp1 and s.tp2
                    else f"*TP:* `{fmt_price(s.tp3)}`"
                )

                cards.append(
                    f"╭──────────────╮\n"
                    f"*#{i}* {side_emoji} *{s.side}* — *{s.symbol}*\n"
                    f"╰──────────────╯\n"
                    f"🆔 `{s.setup_id}`  |  *Conf:* `{s.conf}/100`\n"
                    f"{engine_tag}  |  *RR(TP3):* `{rr3:.2f}`\n"
                    f"🧲 *Pullback:* {pullback}\n"
                    f"💰 *Entry:* `{fmt_price(s.entry)}`   |   🛑 *SL:* `{fmt_price(s.sl)}`\n"
                    f"🎯 {tp_line}\n"
                    f"📈 *Moves:* 24H {pct_with_emoji(s.ch24)}  •  "
                    f"4H {pct_with_emoji(s.ch4)}  •  "
                    f"1H {pct_with_emoji(s.ch1)}  •  "
                    f"15m {pct_with_emoji(s.ch15)}\n"
                    f"💧 *Volume:* `~{fmt_money(s.fut_vol_usd)}`\n"
                    f"🔗 *Chart:* {tv_chart_url(s.symbol)}"
                )

            setups_txt = "\n\n".join(cards)
        else:
            setups_txt = "_No high-quality setups right now._"

        # -------------------------------------------------
        # Waiting for trigger (near-miss)
        # -------------------------------------------------
        waiting_txt = ""
        if _WAITING_TRIGGER:
            lines = ["⏳ *Waiting for Trigger (near-miss)*", SEP]
            for base, d in list(_WAITING_TRIGGER.items())[:SCREEN_WAITING_N]:
                side_emoji = "🟢" if d.get("side") == "BUY" else "🔴"
                lines.append(
                    f"• *{base}* {side_emoji} `{d['side']}` | "
                    f"1H `{d['ch1']:+.2f}%` → need `{d['need']:.2f}%` (trigger `{d['trig']:.2f}%`)"
                )
            waiting_txt = "\n".join(lines)

        # -------------------------------------------------
        # Rejection summary (per-user visibility)
        # -------------------------------------------------
        diag_mode = "full" if is_admin_user(uid) else user_diag_mode(uid)
        reject_txt = _reject_report(diag_mode)

        # -------------------------------------------------
        # Trend continuation watch (adaptive EMA)
        # -------------------------------------------------
        trend_txt = ""
        trend_watch = []
        up_list, dn_list = compute_directional_lists(best_fut)

        # take a small watchlist from leaders/losers (already filtered for vol+move+4h align)
        watch = [b for b, *_ in up_list[:6]] + [b for b, *_ in dn_list[:6]]

        for base in dict.fromkeys(watch):
            mv = best_fut.get(base)
            if not mv:
                continue
            r = await asyncio.to_thread(trend_watch_for_symbol, base, mv, session)
            if r:
                trend_watch.append(r)

        if trend_watch:
            lines = ["📊 *Trend Continuation Watch*", SEP]
            for t in sorted(trend_watch, key=lambda x: x["confidence"], reverse=True)[:6]:
                side_emoji = "🟢" if t["side"] == "BUY" else "🔴"
                lines.append(
                    f"• *{t['symbol']}* {side_emoji} `{t['side']}`  |  "
                    f"Conf `{t['confidence']}/100`  |  24H {pct_with_emoji(t['ch24'])}"
                )
            trend_txt = "\n".join(lines)

        # -------------------------------------------------
        # Market context tables (leaders/losers + top volume)
        # -------------------------------------------------
        leaders_txt = await asyncio.to_thread(build_leaders_table, best_fut)
        up_txt, dn_txt = await asyncio.to_thread(movers_tables, best_fut)

        # -------------------------------------------------
        # Final assembly
        # -------------------------------------------------
        blocks = [
            header,
            "",
            "🔥 *Top Trade Setups*",
            SEP,
            setups_txt,
        ]

        if waiting_txt:
            blocks.extend(["", waiting_txt])

        if trend_txt:
            blocks.extend(["", trend_txt])

        if reject_txt:
            blocks.extend(["", reject_txt])

        blocks.extend([
            "",
            "📌 *Directional Leaders / Losers*",
            SEP,
            up_txt,
            "",
            dn_txt,
            "",
            "🏦 *Market Leaders*",
            SEP,
            leaders_txt,
        ])

        msg = "\n".join([b for b in blocks if b is not None]).strip()

        # Inline buttons for quick charts (top setups only)
        keyboard = [
            [InlineKeyboardButton(text=f"📈 {s.symbol} • {s.setup_id}", url=tv_chart_url(s.symbol))]
            for s in (setups or [])
        ]

        await send_long_message(
            update,
            msg,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None,
        )

    except Exception as e:
        logger.exception("screen_cmd failed")
        await update.message.reply_text(f"⚠️ /screen failed: {e}")







# =========================================================
# TEXT ROUTER (Signal ID lookup)
# =========================================================
async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if text.startswith("PF-"):
        sig = db_get_signal(text)
        if not sig:
            await update.message.reply_text("Signal ID not found.")
            return

        tps = f"TP {fmt_price(float(sig['tp3']))}"
        if sig.get("tp1") and sig.get("tp2") and int(sig["conf"]) >= MULTI_TP_MIN_CONF:
            tps = f"TP1 {fmt_price(float(sig['tp1']))} | TP2 {fmt_price(float(sig['tp2']))} | TP3 {fmt_price(float(sig['tp3']))}"

        rr3 = rr_to_tp(float(sig["entry"]), float(sig["sl"]), float(sig["tp3"]))

        await update.message.reply_text(
            f"🔎 Signal {sig['setup_id']}\n"
            f"{sig['side']} {sig['symbol']} — Conf {sig['conf']}/100\n"
            f"Entry {fmt_price(float(sig['entry']))} | SL {fmt_price(float(sig['sl']))}\n"
            f"{tps}\n"
            f"RR(TP3): {rr3:.2f}\n"
            f"Chart: {tv_chart_url(sig['symbol'])}\n\n"
            f"To journal it:\n"
            f"/trade_open {sig['symbol']} {'long' if sig['side']=='BUY' else 'short'} entry {sig['entry']} sl {sig['sl']} risk usd 40 sig {sig['setup_id']}"
        )
        return


# =========================================================
# EMAIL BODY
# =========================================================
def _email_body_pretty(session_name: str, now_local: datetime, user_tz: str, setups: List[Setup], best_fut: Dict[str, MarketVol]) -> str:
    parts = []
    parts.append(HDR)
    parts.append(f"📩 PulseFutures • {session_name} Session • {now_local.strftime('%Y-%m-%d %H:%M')} ({user_tz})")
    parts.append(HDR)
    parts.append("")

    up, dn = compute_directional_lists(best_fut)
    leaders = sorted(best_fut.items(), key=lambda kv: usd_notional(kv[1]), reverse=True)[:5]
    parts.append("Market Snapshot")
    parts.append(SEP)

    if leaders:
        topv = ", ".join([f"{b}({pct_with_emoji(float(mv.percentage or 0.0))})" for b, mv in leaders])
        parts.append(f"Top Volume: {topv}")

    if up:
        parts.append("Leaders: " + ", ".join([f"{b}({pct_with_emoji(c24)})" for b, v, c24, c4, px in up[:3]]))
    if dn:
        parts.append("Losers:  " + ", ".join([f"{b}({pct_with_emoji(c24)})" for b, v, c24, c4, px in dn[:3]]))
    parts.append("")

    parts.append("Top Setups")
    parts.append(SEP)
    parts.append("")

    for i, s in enumerate(setups, 1):
        rr3 = rr_to_tp(s.entry, s.sl, s.tp3)
        parts.append(f"{i}) {s.setup_id} — {s.side} {s.symbol} — Conf {s.conf}/100")
        parts.append(f"   Entry: {fmt_price_email(s.entry)} | SL: {fmt_price_email(s.sl)} | RR(TP3): {rr3:.2f}")

        if s.tp1 and s.tp2 and s.conf >= MULTI_TP_MIN_CONF:
            parts.append(
                f"   TP1: {fmt_price_email(s.tp1)} ({TP_ALLOCS[0]}%) | "
                f"TP2: {fmt_price_email(s.tp2)} ({TP_ALLOCS[1]}%) | "
                f"TP3: {fmt_price_email(s.tp3)} ({TP_ALLOCS[2]}%)"
            )
        else:
            parts.append(f"   TP: {fmt_price_email(s.tp3)}")

        if s.is_trailing_tp3:
            parts.append("   TP3 Mode: Trailing (hot coin)")

        parts.append(f"   24H {pct_with_emoji(s.ch24)} | 4H {pct_with_emoji(s.ch4)} | 1H {pct_with_emoji(s.ch1)} | 15m {pct_with_emoji(s.ch15)} | Vol~{fmt_money(s.fut_vol_usd)}")
        parts.append(f"   Chart: {tv_chart_url(s.symbol)}")
        parts.append("")

    parts.append(HDR)
    parts.append("🤖 Position Sizing")
    parts.append("Use the PulseFutures Telegram bot to calculate safe position size based on your Stop Loss.")
    parts.append(f"👉 {TELEGRAM_BOT_URL}")
    parts.append(HDR)
    parts.append("Not financial advice.")
    parts.append("PulseFutures")
    parts.append(HDR)
    return "\n".join(parts).strip()

# =========================================================
# EMAIL JOB
# =========================================================
async def alert_job(context: ContextTypes.DEFAULT_TYPE):
    if ALERT_LOCK.locked():
        return
    async with ALERT_LOCK:
        # Store decisions per user for /health transparency
        if not EMAIL_ENABLED:
            # no users loop; but keep bot quiet
            return
        if not email_config_ok():
            return

        users = list_users_notify_on()
        if not users:
            return

        best_fut = await asyncio.to_thread(fetch_futures_tickers)
        if not best_fut:
            return

        # ✅ Build setups per session (session-aware engine knobs)
        setups_by_session: Dict[str, List[Setup]] = {}
        for sess_name in ["NY", "LON", "ASIA"]:
            setups = await asyncio.to_thread(
                pick_setups,
                best_fut,
                max(EMAIL_SETUPS_N * 3, 9),
                True,
                sess_name,
                35,
                1.0,
                SCREEN_WAITING_NEAR_PCT,
                False,   # ✅ allow_no_pullback = False for EMAIL (option A only, HOT bypass handled inside)
            )

          
            setups_by_session[sess_name] = setups
            for s in setups:
                db_insert_signal(s)

        for user in users:
            uid = int(user["user_id"])
            tz = ZoneInfo(user["tz"])

            # session gate
            sess = in_session_now(user)
            if not sess:
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reasons": ["not_in_enabled_session"],
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
                continue

            setups_all = setups_by_session.get(sess["name"], [])
            if not setups_all:
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reasons": [f"no_setups_generated_for_session ({sess['name']})"],
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
                continue

            st = email_state_get(uid)
            if st["session_key"] != sess["session_key"]:
                email_state_set(uid, session_key=sess["session_key"], sent_count=0, last_email_ts=0.0)
                st = email_state_get(uid)

            max_emails = int(user["max_emails_per_session"])
            gap_min = int(user["email_gap_min"])
            gap_sec = gap_min * 60

            day_local = datetime.now(tz).date().isoformat()
            sent_today = _email_daily_get(uid, day_local)
            day_cap = int(user.get("max_emails_per_day", DEFAULT_MAX_EMAILS_PER_DAY))

            if day_cap > 0 and sent_today >= day_cap:
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reasons": [f"daily_email_cap_reached ({sent_today}/{day_cap})"],
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
                continue

            if max_emails > 0 and int(st["sent_count"]) >= max_emails:
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reasons": [f"session_email_cap_reached ({int(st['sent_count'])}/{max_emails})"],
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
                continue

            now_ts = time.time()
            if gap_sec > 0 and (now_ts - float(st["last_email_ts"])) < gap_sec:
                remain = int(gap_sec - (now_ts - float(st["last_email_ts"])))
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reasons": [f"email_gap_active (remain {remain}s)"],
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
                continue

            min_conf = SESSION_MIN_CONF.get(sess["name"], 78)
            min_rr = SESSION_MIN_RR_TP3.get(sess["name"], 2.0)

            confirmed: List[Setup] = []
            early: List[Setup] = []

            skip_reasons_counter = Counter()

            for s in setups_all:
                if s.conf < min_conf:
                    skip_reasons_counter["below_session_conf_floor"] += 1
                    continue

                # Trend-follow filter
                if s.side == "BUY" and float(s.ch24) < float(TREND_24H_TOL):
                    skip_reasons_counter["trend_filter_buy_24h_too_low"] += 1
                    continue
                if s.side == "SELL" and float(s.ch24) > -float(TREND_24H_TOL):
                    skip_reasons_counter["trend_filter_sell_24h_too_high"] += 1
                    continue

                # RR floor by session (TP3)
                rr3 = rr_to_tp(s.entry, s.sl, s.tp3)
                if rr3 < min_rr:
                    skip_reasons_counter["below_session_rr_floor"] += 1
                    continue

                # Symbol cooldown
                if symbol_recently_emailed(uid, s.symbol, SYMBOL_COOLDOWN_HOURS):
                    skip_reasons_counter["symbol_cooldown_18h"] += 1
                    continue

                is_confirm_15m = abs(float(s.ch15)) >= CONFIRM_15M_ABS_MIN

                if is_confirm_15m:
                    confirmed.append(s)
                else:
                    if abs(float(s.ch1)) < EARLY_1H_ABS_MIN:
                        skip_reasons_counter["early_gate_ch1_not_strong"] += 1
                        continue
                    if s.conf < (min_conf + EARLY_EMAIL_EXTRA_CONF):
                        skip_reasons_counter["early_gate_conf_not_high_enough"] += 1
                        continue
                    early.append(s)

                if len(confirmed) >= EMAIL_SETUPS_N:
                    break

            filtered: List[Setup] = confirmed[:EMAIL_SETUPS_N]
            if len(filtered) < EMAIL_SETUPS_N and early:
                need = EMAIL_SETUPS_N - len(filtered)
                filtered.extend(early[:min(need, EARLY_EMAIL_MAX_FILL)])

            if not filtered:
                top_reasons = [f"{k}: {v}" for k, v in skip_reasons_counter.most_common(6)]
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reasons": ["no_setups_after_filters"] + (top_reasons if top_reasons else []),
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
                continue

            now_local = datetime.now(tz)
            body = _email_body_pretty(sess["name"], now_local, user["tz"], filtered, best_fut)
            subject = f"PulseFutures • {sess['name']} • Premium Setups ({int(st['sent_count'])+1})"

            ok = await asyncio.to_thread(send_email, subject, body, uid)
            if ok:
                email_state_set(uid, sent_count=int(st["sent_count"]) + 1, last_email_ts=now_ts)
                _email_daily_inc(uid, day_local, 1)
                for s in filtered:
                    mark_symbol_emailed(uid, s.symbol)

                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SENT",
                    "reasons": [f"sent {len(filtered)} setups"],
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
            else:
                err = _LAST_SMTP_ERROR.get(uid, "unknown")
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reasons": [f"send_email_failed ({err})"],
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }

# =========================================================
# /health (transparent system health)
# =========================================================
async def health_sys_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Transparent health check:
    - DB reachable
    - Exchange tickers reachable
    - Email configured/enabled status
    - Cache stats
    - Blackout status
    """
    # DB check
    db_ok = True
    db_err = ""
    try:
        con = db_connect()
        con.execute("SELECT 1")
        con.close()
    except Exception as e:
        db_ok = False
        db_err = str(e)

    # Exchange check (quick)
    ex_ok = True
    ex_err = ""
    tickers_n = 0
    t0 = time.time()
    try:
        best = await asyncio.to_thread(fetch_futures_tickers)
        tickers_n = len(best or {})
    except Exception as e:
        ex_ok = False
        ex_err = str(e)
    dt_ms = int((time.time() - t0) * 1000)

    # Cache stats
    cache_items = len(_CACHE)

    # Email status
    email_cfg = email_config_ok()
    email_on = EMAIL_ENABLED

    # Sessions
    uid = update.effective_user.id
    user = get_user(uid)
    enabled = user_enabled_sessions(user)
    sess = in_session_now(user)
    now_s = sess["name"] if sess else "NONE"

    # Blackout

    msg = [
        "🩺 PulseFutures • Health",
        HDR,
        f"DB: {'OK' if db_ok else 'FAIL'}" + (f" | {db_err}" if (not db_ok and db_err) else ""),
        f"Bybit/CCXT: {'OK' if ex_ok else 'FAIL'} | tickers={tickers_n} | {dt_ms}ms" + (f" | {ex_err}" if (not ex_ok and ex_err) else ""),
        f"Email: enabled={email_on} | configured={email_cfg}",
        f"Cache: items={cache_items} | tickersTTL={TICKERS_TTL_SEC}s | ohlcvTTL={OHLCV_TTL_SEC}s",
        HDR,
        f"Your TZ: {user['tz']}",
        f"Sessions enabled: {', '.join(enabled)} | Now: {now_s}",
        f"Limits: emailcap/session={int(user['max_emails_per_session'])} (0=∞), emaildaycap={int(user.get('max_emails_per_day', DEFAULT_MAX_EMAILS_PER_DAY))} (0=∞), gap={int(user['email_gap_min'])}m",
    ]
    await update.message.reply_text("\n".join(msg))


# =========================================================
# MAIN (Background Worker = POLLING)
# =========================================================

async def _post_init(app: Application):
    # اگر قبلاً webhook ست شده بوده، پاکش کن تا polling گیر نکنه
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
    except Exception as e:
        logger.warning("delete_webhook failed (ignored): %s", e)

def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN missing")

    db_init()

    app = Application.builder().token(TOKEN).post_init(_post_init).build()

    # ================= Handlers =================
    app.add_handler(CommandHandler(["help", "start"], cmd_help))
    app.add_handler(CommandHandler("tz", tz_cmd))
    app.add_handler(CommandHandler("screen", screen_cmd))

    app.add_handler(CommandHandler("equity", equity_cmd))
    app.add_handler(CommandHandler("equity_reset", equity_reset_cmd))
    app.add_handler(CommandHandler("riskmode", riskmode_cmd))
    app.add_handler(CommandHandler("dailycap", dailycap_cmd))
    app.add_handler(CommandHandler("limits", limits_cmd))

    app.add_handler(CommandHandler("sessions", sessions_cmd))
    app.add_handler(CommandHandler("sessions_on", sessions_on_cmd))
    app.add_handler(CommandHandler("sessions_off", sessions_off_cmd))

    app.add_handler(CommandHandler("notify_on", notify_on))
    app.add_handler(CommandHandler("notify_off", notify_off))

    app.add_handler(CommandHandler("size", size_cmd))
    app.add_handler(CommandHandler("trade_open", trade_open_cmd))
    app.add_handler(CommandHandler("trade_close", trade_close_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    app.add_handler(CommandHandler("report_daily", report_daily_cmd))
    app.add_handler(CommandHandler("report_weekly", report_weekly_cmd))
    app.add_handler(CommandHandler("signals_daily", signals_daily_cmd))
    app.add_handler(CommandHandler("signals_weekly", signals_weekly_cmd))

    app.add_handler(CommandHandler("health", health_cmd))
    app.add_handler(CommandHandler("health_sys", health_sys_cmd))
    app.add_handler(CommandHandler("diag_on", diag_on_cmd))
    app.add_handler(CommandHandler("diag_off", diag_off_cmd))
    app.add_handler(CommandHandler("email_test", email_test_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    # ================= JobQueue =================
    if app.job_queue:
        app.job_queue.run_repeating(
            alert_job,
            interval=CHECK_INTERVAL_MIN * 60,
            first=30,
            name="alert_job"
        )
    else:
        logger.error("JobQueue NOT available – install python-telegram-bot[job-queue]")

    logger.info("Starting Telegram bot in POLLING mode (Background Worker) ...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()

