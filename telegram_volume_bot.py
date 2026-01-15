#!/usr/bin/env python3
"""
PulseFutures — Bybit Futures (Swap) Screener + Signals Email + Risk Manager + Trade Journal (Telegram)
"""

import os
import sys
import time
import logging

logger = logging.getLogger(__name__)

import json
import re
import ssl
import smtplib
import sqlite3
import asyncio

from dataclasses import dataclass
from email.message import EmailMessage
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import Counter, defaultdict

import ccxt
from tabulate import tabulate

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)




def render_primary_only() -> None:
    """
    Render-safe single instance guard.

    IMPORTANT:
    - On Render, RENDER_INSTANCE_ID is often a string like "srv-...".
      That does NOT mean "secondary".
    - We only exit if Render provides a *numeric replica index* AND it's not 0.
    """
    # Some Render setups expose an index/number; only trust numeric ones.
    candidates = [
        os.environ.get("RENDER_INSTANCE_NUMBER"),
        os.environ.get("RENDER_INSTANCE_INDEX"),
        os.environ.get("RENDER_REPLICA_NUMBER"),
    ]

    for val in candidates:
        if val is None:
            continue
        # Only treat as authoritative if it's purely numeric
        if val.isdigit():
            if val != "0":
                logging.warning(f"Secondary Render replica ({val}) exiting.")
                raise SystemExit(0)
            return  # primary replica

    # Fallback: DO NOT exit just because RENDER_INSTANCE_ID exists (it's usually non-numeric).
    instance_id = os.environ.get("RENDER_INSTANCE_ID")
    logging.info(f"Render instance id: {instance_id} (no numeric replica index found; continuing)")




# =========================================================
# CONFIG
# =========================================================
EXCHANGE_ID = "bybit"
TOKEN = os.environ.get("TELEGRAM_TOKEN")

DEFAULT_TYPE = "swap"  # bybit futures
DB_PATH = os.environ.get("DB_PATH", "/var/data/pulsefutures.db")

CHECK_INTERVAL_MIN = int(os.environ.get("CHECK_INTERVAL_MIN", "5"))

# -------------------------
# LOGGING: redact secrets + quiet noisy libs (Render-safe)
# -------------------------
class RedactSecretsFilter(logging.Filter):
    def __init__(self, secrets: List[str]):
        super().__init__()
        self.secrets = [s for s in (secrets or []) if s]

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True

        for s in self.secrets:
            if s and s in msg:
                msg = msg.replace(s, "[REDACTED]")

        # Replace rendered message safely
        record.msg = msg
        record.args = ()
        return True


def setup_logging():
    # Let Render control via LOG_LEVEL, default INFO
    lvl = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, lvl, logging.INFO))

    # Quiet the libs that spam request URLs (and may leak token)
    for noisy in ("httpx", "telegram", "telegram.ext", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Redact secrets from ALL logs
    secrets = [
        os.environ.get("TELEGRAM_TOKEN", ""),
        os.environ.get("EMAIL_PASS", ""),
    ]
    logging.getLogger().addFilter(RedactSecretsFilter(secrets))


# Call once at import time (after TOKEN/envs exist)
setup_logging()

# Recreate your app logger after setup_logging so it inherits config
logger = logging.getLogger("pulsefutures")




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

# =========================================================
# ✅ COOLDOWNS (email anti-spam)
# =========================================================
# Session-aware cooldown hours (recommended)
SESSION_SYMBOL_COOLDOWN_HOURS = {
    "NY": 2,     # more active
    "LON": 3,    # balanced
    "ASIA": 4,   # more strict
}

# Fallback if session unknown
SYMBOL_COOLDOWN_HOURS = 4

def cooldown_hours_for_session(session_name: str) -> int:
    s = (session_name or "").strip().upper()
    return int(SESSION_SYMBOL_COOLDOWN_HOURS.get(s, SYMBOL_COOLDOWN_HOURS))

def max_cooldown_hours() -> int:
    try:
        return int(max(list(SESSION_SYMBOL_COOLDOWN_HOURS.values()) + [SYMBOL_COOLDOWN_HOURS]))
    except Exception:
        return int(SYMBOL_COOLDOWN_HOURS)


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




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pulsefutures")

STABLES = {"USDT", "USDC", "USD", "TUSD", "FDUSD", "DAI", "PYUSD"}

HDR = "━━━━━━━━━━━━━━━━━━━━"
SEP = "────────────────────"

ALERT_LOCK = asyncio.Lock()

# =========================================================
# ✅ DB BACKUP/RESET + EMAIL TRADE WINDOW (per-user)
# =========================================================
DB_BACKUP_PATH = os.environ.get("DB_BACKUP_PATH", DB_PATH + ".bak")

DB_FILE_LOCK = asyncio.Lock()



# Sessions defined in UTC windows (market convention, with overlaps)
# ASIA: 00:00–09:00 UTC
# LON : 07:00–16:00 UTC
# NY  : 13:00–22:00 UTC
# Priority resolves overlaps: NY > LON > ASIA
SESSIONS_UTC = {
    "ASIA": {"start": "00:00", "end": "09:00"},
    "LON":  {"start": "07:00", "end": "16:00"},
    "NY":   {"start": "13:00", "end": "22:00"},
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


# ✅ NEW: "Waiting for Trigger" (near-miss candidates)

_WAITING_TRIGGER: Dict[str, Dict[str, Any]] = {}


def is_admin_user(user_id: int) -> bool:
    return int(user_id) in ADMIN_USER_IDS if ADMIN_USER_IDS else False



# =========================================================
# GLOBAL STATE
# =========================================================

# Stores last SMTP error per user for /health and /email_test
_LAST_SMTP_ERROR: Dict[int, str] = {}

# Per-user diagnostics preference (non-admin). Admin always full.
_USER_DIAG_MODE: Dict[int, str] = {}

# ✅ NEW: Stores last email decision per user (SENT / SKIP + reasons)
_LAST_EMAIL_DECISION: Dict[int, Dict[str, Any]] = {}


def user_diag_mode(user_id: int) -> str:
    """
    Returns diagnostic visibility mode for this user:
    - admin => 'full'
    - non-admin => 'friendly' or 'off'
    Enforces PUBLIC_DIAGNOSTICS_MODE:
      - 'off'   => always off for non-admin
      - 'admin' => allow per-user friendly/off
    """
    uid = int(user_id)

    if is_admin_user(uid):
        return "full"

    # hard lock for public users
    if PUBLIC_DIAGNOSTICS_MODE == "off":
        return "off"

    # admin-only diagnostics policy (recommended):
    # non-admin can choose friendly/off locally
    mode = (_USER_DIAG_MODE.get(uid) or "off").strip().lower()
    if mode not in {"friendly", "off"}:
        mode = "off"
    return mode

def reset_reject_tracker() -> None:
    return

def _rej(reason: str, base: str, mv: "MarketVol", extra: str = "") -> None:
    return

def _reject_report(diag_mode: str = "off") -> str:
    return ""


# =========================================================
# DB
# =========================================================
def db_connect() -> sqlite3.Connection:
    # ensure directory exists
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    # safer sqlite on hosted envs
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA busy_timeout=5000;")
        con.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass

    return con


def db_backup_file() -> Tuple[bool, str]:
    """
    Copies DB_PATH -> DB_BACKUP_PATH
    """
    try:
        # Ensure folder exists
        os.makedirs(os.path.dirname(DB_BACKUP_PATH) or ".", exist_ok=True)

        if not os.path.exists(DB_PATH):
            return False, f"DB file not found: {DB_PATH}"

        # Copy bytes
        with open(DB_PATH, "rb") as src, open(DB_BACKUP_PATH, "wb") as dst:
            dst.write(src.read())

        return True, f"Backup created: {DB_BACKUP_PATH}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def db_restore_file() -> Tuple[bool, str]:
    """
    Copies DB_BACKUP_PATH -> DB_PATH
    """
    try:
        if not os.path.exists(DB_BACKUP_PATH):
            return False, f"No backup found: {DB_BACKUP_PATH}"

        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

        with open(DB_BACKUP_PATH, "rb") as src, open(DB_PATH, "wb") as dst:
            dst.write(src.read())

        return True, f"Restored from backup: {DB_BACKUP_PATH}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def db_wipe_all_data_keep_schema() -> None:
    """
    Deletes ALL rows from core tables, keeps schema.
    Also resets in-memory trackers.
    """
    con = db_connect()
    cur = con.cursor()

    # Order matters if FK ever added later. We currently don’t have FK refs, but still safe.
    tables = [
        "trades",
        "signals",
        "emailed_symbols",
        "email_state",
        "email_daily",
        "risk_daily",
        "setup_counter",
        # users is kept (preferences) OR can be wiped too. You requested "clean database":
        # wiping users means everyone will be re-created on next /start.
        "users",
    ]

    for t in tables:
        try:
            cur.execute(f"DELETE FROM {t}")
        except Exception:
            pass

    con.commit()
    con.close()

    # Also clear runtime trackers
    _LAST_SMTP_ERROR.clear()
    _USER_DIAG_MODE.clear()
    _LAST_EMAIL_DECISION.clear()


async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /reset
    Admin-only:
    1) Backup DB file
    2) Wipe tables (clean DB)
    """
    uid = update.effective_user.id
    if not is_admin_user(uid):
        await update.message.reply_text("⛔️ Admin only.")
        return

    async with DB_FILE_LOCK:
        ok, msg = db_backup_file()
        if not ok:
            await update.message.reply_text(f"❌ Backup failed.\n{msg}")
            return

        try:
            db_wipe_all_data_keep_schema()
        except Exception as e:
            await update.message.reply_text(f"❌ Reset failed.\n{type(e).__name__}: {e}")
            return

    await update.message.reply_text(
        "✅ Database RESET completed.\n"
        f"{msg}\n\n"
        "Use /restore to revert to the backup."
    )


async def restore_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /restore
    Admin-only:
    Restores DB from DB_BACKUP_PATH
    """
    uid = update.effective_user.id
    if not is_admin_user(uid):
        await update.message.reply_text("⛔️ Admin only.")
        return

    async with DB_FILE_LOCK:
        ok, msg = db_restore_file()
        if not ok:
            await update.message.reply_text(f"❌ Restore failed.\n{msg}")
            return

        # After restore, ensure schema migrations still applied
        try:
            db_init()
        except Exception:
            pass

    await update.message.reply_text(f"✅ Database RESTORED.\n{msg}")




def db_init():
    # ensures folder exists before creating DB file
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
        
    con = db_connect()
    cur = con.cursor()

    # =========================================================
    # ✅ Cooldown table (v2): direction-aware + optional session stamp
    # - old: PRIMARY KEY(user_id, symbol)
    # - new: PRIMARY KEY(user_id, symbol, side)
    # =========================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS emailed_symbols (
        user_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        session TEXT NOT NULL DEFAULT '',
        emailed_ts REAL NOT NULL,
        PRIMARY KEY (user_id, symbol, side)
    )
    """)

    # --- Migration from old schema (if needed) ---
    # If table exists but missing "side", then it's old schema.
    try:
        cur.execute("PRAGMA table_info(emailed_symbols)")
        cols = [r[1] for r in cur.fetchall()]
        if "side" not in cols:
            # Create new table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS emailed_symbols_v2 (
                    user_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    session TEXT NOT NULL DEFAULT '',
                    emailed_ts REAL NOT NULL,
                    PRIMARY KEY (user_id, symbol, side)
                )
            """)

            # Copy old rows -> assume side=ANY (we duplicate into BUY and SELL to be safe)
            cur.execute("SELECT user_id, symbol, emailed_ts FROM emailed_symbols")
            old_rows = cur.fetchall() or []
            for r in old_rows:
                uid0 = int(r[0])
                sym0 = str(r[1]).upper()
                ts0 = float(r[2])
                # Duplicate into BUY and SELL so cooldown still applies in both directions initially.
                cur.execute("""
                    INSERT OR REPLACE INTO emailed_symbols_v2 (user_id, symbol, side, session, emailed_ts)
                    VALUES (?, ?, ?, ?, ?)
                """, (uid0, sym0, "BUY", "", ts0))
                cur.execute("""
                    INSERT OR REPLACE INTO emailed_symbols_v2 (user_id, symbol, side, session, emailed_ts)
                    VALUES (?, ?, ?, ?, ?)
                """, (uid0, sym0, "SELL", "", ts0))

            # Swap tables
            cur.execute("DROP TABLE emailed_symbols")
            cur.execute("ALTER TABLE emailed_symbols_v2 RENAME TO emailed_symbols")
            con.commit()
    except Exception:
        # Don't block startup if migration fails; worst case cooldown table resets.
        pass


    cur.execute("PRAGMA table_info(users)")
    cols = {r[1] for r in cur.fetchall()}

    # ✅ Trade window columns (local time HH:MM), empty = disabled
    if "trade_window_start" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN trade_window_start TEXT NOT NULL DEFAULT ''")
    if "trade_window_end" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN trade_window_end TEXT NOT NULL DEFAULT ''")

    
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

def set_user_email(uid: int, email: str) -> None:
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        # ensure row exists
        cur.execute(
            "INSERT OR IGNORE INTO users(user_id) VALUES(?)",
            (uid,)
        )
        cur.execute(
            "UPDATE users SET email_to = ? WHERE user_id = ?",
            (email, uid)
        )
        con.commit()

def ensure_email_column():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        try:
            cur.execute("ALTER TABLE users ADD COLUMN email_to TEXT")
            con.commit()
            logger.info("Added email_to column to users table")
        except sqlite3.OperationalError:
            # Column already exists
            pass



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

def mark_symbol_emailed(user_id: int, symbol: str, side: str, session_name: str = ""):
    """
    ✅ Direction-aware cooldown: stored per (symbol, side)
    Optionally stores session name (NY/LON/ASIA) for audit.
    """
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO emailed_symbols (user_id, symbol, side, session, emailed_ts)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id, symbol, side) DO UPDATE SET
            session=excluded.session,
            emailed_ts=excluded.emailed_ts
    """, (int(user_id), str(symbol).upper(), str(side).upper(), str(session_name or ""), time.time()))
    con.commit()
    con.close()


def symbol_recently_emailed(
    user_id: int,
    symbol: str,
    side: str,
    session_name: str,
) -> bool:
    """
    ✅ Session-aware + direction-aware cooldown check.
    Cooldown hours depend on CURRENT session (NY/LON/ASIA).
    """
    cooldown_hours = float(cooldown_hours_for_session(session_name))

    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        SELECT emailed_ts
        FROM emailed_symbols
        WHERE user_id=? AND symbol=? AND side=?
    """, (int(user_id), str(symbol).upper(), str(side).upper()))
    row = cur.fetchone()
    con.close()
    if not row:
        return False

    last_ts = float(row["emailed_ts"])
    return (time.time() - last_ts) < (cooldown_hours * 3600.0)


def list_cooldowns(user_id: int) -> List[dict]:
    """
    Returns list of cooldown stamps for user:
    [{symbol, side, session, emailed_ts}, ...]
    """
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        SELECT symbol, side, session, emailed_ts
        FROM emailed_symbols
        WHERE user_id=?
        ORDER BY emailed_ts DESC
    """, (int(user_id),))
    rows = cur.fetchall() or []
    con.close()
    return [dict(r) for r in rows]


def _fmt_dur(seconds: float) -> str:
    s = int(max(0, seconds))
    h = s // 3600
    m = (s % 3600) // 60
    if h <= 0:
        return f"{m}m"
    return f"{h}h {m}m"


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


def db_trades_all(user_id: int) -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        SELECT * FROM trades
        WHERE user_id=?
        ORDER BY opened_ts ASC
    """, (int(user_id),))
    rows = cur.fetchall() or []
    con.close()
    return [dict(r) for r in rows]


def _profit_factor(trades: List[dict]) -> Optional[float]:
    closed = [t for t in trades if t.get("closed_ts") is not None and t.get("pnl") is not None]
    if not closed:
        return None
    gp = sum(float(t["pnl"]) for t in closed if float(t["pnl"]) > 0)
    gl = abs(sum(float(t["pnl"]) for t in closed if float(t["pnl"]) < 0))
    if gl <= 0:
        return None if gp <= 0 else float("inf")
    return gp / gl


def _expectancy_r(trades: List[dict]) -> Optional[float]:
    closed = [t for t in trades if t.get("closed_ts") is not None and t.get("r_mult") is not None]
    if not closed:
        return None
    rs = [float(t["r_mult"]) for t in closed if t.get("r_mult") is not None]
    return (sum(rs) / len(rs)) if rs else None


async def report_overall_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)

    trades = db_trades_all(uid)
    stats = _stats_from_trades(trades)

    pf = _profit_factor(trades)
    exp_r = _expectancy_r(trades)

    msg = [
        "📊 Overall Report (ALL TIME)",
        HDR,
        f"Closed: {stats['closed_n']} | Wins: {stats['wins']} | Losses: {stats['losses']}",
        f"Win rate: {stats['win_rate']:.1f}%",
        f"Net PnL: {stats['net']:+.2f}",
        f"Avg R: {stats['avg_r']:+.2f}" if stats["avg_r"] is not None else "Avg R: -",
        f"Expectancy (R): {exp_r:+.2f}" if exp_r is not None else "Expectancy (R): -",
        f"Profit Factor: {pf:.2f}" if (pf is not None and pf != float('inf')) else ("Profit Factor: ∞" if pf == float('inf') else "Profit Factor: -"),
        f"Best: {stats['biggest_win']:+.2f} | Worst: {stats['biggest_loss']:+.2f}",
        HDR,
        f"Equity (current): ${float(user['equity']):.2f}",
    ]

    await update.message.reply_text("\n".join(msg))


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
    Market sessions in UTC (with overlaps handled by priority):
    - NY  : 13:00–22:00
    - LON : 07:00–16:00
    - ASIA: 00:00–09:00
    Priority: NY > LON > ASIA
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    h = now_utc.hour

    # Priority first (overlaps intentionally resolved)
    if 13 <= h < 22:
        return "NY"
    if 7 <= h < 16:
        return "LON"
    if 0 <= h < 9:
        return "ASIA"

    # Outside the three main windows: treat as NY tail/transition
    # (or return "ASIA"/"OFF" if you prefer)
    return "NY"




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
def fmt_price(x: float) -> str:
    """
    Telegram-friendly price formatting (less noisy than raw float).
    """
    try:
        x = float(x)
    except Exception:
        return "0"

    ax = abs(x)
    if ax >= 1000:
        return f"{x:.2f}"
    if ax >= 100:
        return f"{x:.2f}"
    if ax >= 1:
        return f"{x:.3f}"
    if ax >= 0.1:
        return f"{x:.4f}"
    if ax >= 0.01:
        return f"{x:.5f}"
    return f"{x:.6f}"

def fmt_price_email(x: float) -> str:
    return fmt_price(x)


# =========================================================
# TELEGRAM SAFE SEND (chunking + markdown fallback)
# =========================================================

from telegram.error import BadRequest, TimedOut, NetworkError, RetryAfter

SAFE_CHUNK = 3500

async def send_long_message(
    update: Update,
    text: str,
    parse_mode: Optional[str] = None,
    disable_web_page_preview: bool = True,
    reply_markup=None,
):
    if not update or not update.message:
        return

    s = text or ""
    chunks = [s[i:i+SAFE_CHUNK] for i in range(0, len(s), SAFE_CHUNK)]

    first = True
    for ch in chunks:
        try:
            await update.message.reply_text(
                ch,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
                reply_markup=reply_markup if first else None,
            )

        except RetryAfter as e:
            # Telegram rate limit: wait then retry once
            await asyncio.sleep(int(e.retry_after) + 1)
            await update.message.reply_text(
                ch,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
                reply_markup=reply_markup if first else None,
            )

        except BadRequest:
            # Usually markdown issue -> fallback to plain text
            await update.message.reply_text(
                ch,
                parse_mode=None,
                disable_web_page_preview=disable_web_page_preview,
                reply_markup=reply_markup if first else None,
            )

        except (TimedOut, NetworkError) as e:
            logger.warning("Telegram send failed (network): %s", e)
            # optionally: retry once
            await asyncio.sleep(2)

        except Exception as e:
            logger.exception("Telegram send failed: %s", e)

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
        # ✅ 4H change from real 4h candles (light: limit=2)
        ch4 = 0.0
        try:
            c4h = fetch_ohlcv(mv.symbol, "4h", limit=2)
            if c4h and len(c4h) >= 2:
                c_last = float(c4h[-1][4])
                c_prev = float(c4h[-2][4])
                if c_prev > 0:
                    ch4 = ((c_last - c_prev) / c_prev) * 100.0
        except Exception:
            ch4 = 0.0

        rows.append([
            base,
            fmt_money(usd_notional(mv)),
            pct_with_emoji(float(mv.percentage or 0.0)),  # 24H
            pct_with_emoji(float(ch4)),                  # 4H
        ])

    return "*Market Leaders (Top 10 by Futures Volume)*\n" + table_md(rows, ["SYM", "F Vol", "24H", "4H"])

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
    up_rows = [[b, fmt_money(v), pct_with_emoji(c24), pct_with_emoji(c4)] for b, v, c24, c4, px in up[:10]]
    dn_rows = [[b, fmt_money(v), pct_with_emoji(c24), pct_with_emoji(c4)] for b, v, c24, c4, px in dn[:10]]
    up_txt = "*Directional Leaders (24H ≥ +10%, F vol ≥ 5M, 4H aligned)*\n" + (table_md(up_rows, ["SYM", "F Vol", "24H", "4H"]) if up_rows else "_None_")
    dn_txt = "*Directional Losers (24H ≤ -10%, F vol ≥ 5M, 4H aligned)*\n" + (table_md(dn_rows, ["SYM", "F Vol", "24H", "4H"]) if dn_rows else "_None_")
    return up_txt, dn_txt



# =========================================================
# ✅ MODIFIED: make_setup (trailing TP3 only when it's a trailing-needed setup)
# Rule applied: trailing only for HOT coins AND Engine B (Momentum)
# =========================================================
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
        # Waiting for Trigger (near-miss) — store ONLY side + a color dot (no numbers)
        if trig_min > 0:
            ratio = abs(ch1) / trig_min  # internal only
            if ratio >= float(waiting_near_pct):
                # color indicates "how close" WITHOUT revealing thresholds
                if ratio >= 0.92:
                    dot = "🟢"
                elif ratio >= 0.82:
                    dot = "🟡"
                else:
                    dot = "🔴"

                side_guess = "BUY" if ch1 > 0 else "SELL"
                _WAITING_TRIGGER[str(base)] = {"side": side_guess, "dot": dot}

        _rej("ch1_below_trigger", base, mv)
        return None

    # ✅ IMPORTANT: this must be OUTSIDE the if block (no extra indent)
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

    pb_ok = False
    if pb_ema_val > 0:
        pb_ok, _, _, _ = ema_support_proximity_ok(entry, pb_ema_val, atr_1h, session_name)

    pullback_ready = bool(pb_ok or pullback_bypass_hot)

    # =========================================================
    # ENGINE A (Mean-Reversion) vs ENGINE B (Momentum)
    # =========================================================
    engine_a_ok = bool(ENGINE_A_PULLBACK_ENABLED and pullback_ready)

    engine_b_ok = False
    if ENGINE_B_MOMENTUM_ENABLED:
        if abs(ch1) >= MOMENTUM_MIN_CH1 and abs(ch24) >= MOMENTUM_MIN_24H:
            if fut_vol >= (MOVER_VOL_USD_MIN * MOMENTUM_VOL_MULT):
                body_pct = abs(ch1)
                if atr_pct_now > 0 and body_pct >= (MOMENTUM_ATR_BODY_MULT * atr_pct_now):
                    ema_ok, dist_pct, _, _ = ema_support_proximity_ok(entry, ema_support_15m, atr_1h, session_name)
                    if dist_pct <= MOMENTUM_MAX_ADAPTIVE_EMA_DIST:
                        engine_b_ok = True

    if not engine_a_ok and not engine_b_ok:
        _rej("no_engine_passed", base, mv, f"ch1={ch1:.2f} ch24={ch24:.2f} pb_dist={pb_dist_pct:.2f}")
        return None

    if not pullback_ready:
        if not (allow_no_pullback and engine_b_ok):
            _rej("price_not_near_ema12_15m", base, mv, f"pullback_not_ready pb_dist={pb_dist_pct:.2f}")
            return None

    engine = "A" if pullback_ready else "B"

    if engine == "A" and abs(float(ch1)) >= float(SHARP_1H_MOVE_PCT):
        if not ema_support_reaction_ok_15m(c15, pb_ema_val, side, session_name):
            _rej("sharp_1h_no_ema_reaction", base, mv, f"ch1={ch1:+.2f}% needs EMA reaction")
            return None

    thr = clamp(max(12.0, 2.5 * ((atr_1h / entry) * 100.0 if (atr_1h and entry) else 0.0)), 12.0, 22.0)
    if side == "BUY" and ch24 <= -thr:
        _rej("24h_contradiction_for_long", base, mv, f"ch24={ch24:+.1f}% <= -{thr:.1f}%")
        return None
    if side == "SELL" and ch24 >= +thr:
        _rej("24h_contradiction_for_short", base, mv, f"ch24={ch24:+.1f}% >= +{thr:.1f}%")
        return None

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

    # ✅ trailing only for the setups that need it (Momentum + Hot)
    trailing_tp3 = bool(hot and engine == "B")

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
        is_trailing_tp3=trailing_tp3,
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

    # --- Trade window enforcement ---
    if not in_trade_window_now(user, now_local):
        _LAST_EMAIL_DECISION[user_id] = {
            "status": "SKIP",
            "reason": "outside_trade_window",
            "ts": time.time(),
        }
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

    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = email_to
    msg["Subject"] = subject

    if body_html:
        msg.set_content(body_text)
        msg.add_alternative(body_html, subtype="html")
    else:
        msg.set_content(body_text)

    try:
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)

        _LAST_EMAIL_DECISION[user_id] = {
            "status": "SENT",
            "reason": "ok",
            "ts": time.time(),
        }
        return True

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        _LAST_SMTP_ERROR[user_id] = err
        _LAST_EMAIL_DECISION[user_id] = {
            "status": "FAIL",
            "reason": err,
            "ts": time.time(),
        }
        logger.error(f"Email send failed for user {user_id}: {err}")
        return False


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

async def email_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id

    if not context.args:
        await update.message.reply_text("Usage: /email your@email.com\nExample: /email ramin@gmail.com")
        return

    email = context.args[0].strip()
    if not EMAIL_RE.match(email):
        await update.message.reply_text("❌ Invalid email format.\nExample: /email ramin@gmail.com")
        return

    try:
        # Save per-user email (key you already read in email_test_cmd)
        set_user_email(uid, email)

        await update.message.reply_text(
            f"✅ Recipient email saved:\n{email}\n\nNow run: /email_test"
        )
    except Exception as e:
        logger.exception("email_cmd failed")
        await update.message.reply_text(f"❌ Failed to save email: {type(e).__name__}: {e}")



async def email_test_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)

    # Always respond immediately
    await update.message.reply_text("📧 Running email test…")

    # Show config status clearly
    if not EMAIL_ENABLED:
        await update.message.reply_text("❌ Email is disabled (EMAIL_ENABLED=False). Turn it on in your config/env.")
        return

    if not email_config_ok():
        # If you have a helper that checks env vars, mention it
        await update.message.reply_text(
            "❌ Email config is NOT OK.\n"
            "Check env vars like EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FROM (and anything your code expects).\n"
            "Tip: run /health_sys and verify Email: enabled/configured."
        )
        return

    # Determine recipient
    # (Your bot earlier used a per-user email field; adjust key name if yours differs)
    to_email = (user.get("email_to") or user.get("email") or "").strip()
    if not to_email:
        await update.message.reply_text(
            "❌ No recipient email found for your user.\n"
            "Set it first (whatever your bot uses), e.g. /email your@email.com"
        )
        return

    # Build a short test message
    tz_name = str(user.get("tz") or "UTC")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
        tz_name = "UTC"

    now_local = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    subject = "PulseFutures — Email Test ✅"
    body = (
        f"{HDR}\n"
        f"PulseFutures Email Test\n"
        f"{HDR}\n\n"
        f"User ID: {uid}\n"
        f"Recipient: {to_email}\n"
        f"Time: {now_local} ({tz_name})\n\n"
        f"If you received this, SMTP is working.\n"
        f"{HDR}\n"
    )

    # Send
    try:
        ok = await asyncio.to_thread(send_email, subject, body, uid)
    except Exception as e:
        logger.exception("email_test_cmd failed")
        await update.message.reply_text(f"❌ Test email crashed: {type(e).__name__}: {e}")
        return

    if ok:
        await update.message.reply_text(f"✅ Test email SENT to: {to_email}")
    else:
        err = _LAST_SMTP_ERROR.get(uid, "unknown_error")
        await update.message.reply_text(
            "❌ Test email FAILED.\n"
            f"Reason: {err}\n\n"
            "Common causes:\n"
            "- Gmail: app password / 2FA issues\n"
            "- Wrong EMAIL_HOST/PORT (465 SSL vs 587 STARTTLS)\n"
            "- EMAIL_FROM not matching account\n"
        )




def _parse_hhmm_local(s: str) -> Tuple[int, int]:
    s = (s or "").strip()
    m = re.match(r"^(\d{2}):(\d{2})$", s)
    if not m:
        raise ValueError("bad time")
    hh = int(m.group(1)); mm = int(m.group(2))
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        raise ValueError("bad time")
    return hh, mm


def in_trade_window_now(user: dict, now_local: Optional[datetime] = None) -> bool:
    """
    If user trade_window_start/end are empty -> allowed (no restriction).
    Otherwise checks local time window (supports overnight windows).
    """
    start_s = str(user.get("trade_window_start") or "").strip()
    end_s = str(user.get("trade_window_end") or "").strip()
    if not start_s or not end_s:
        return True  # disabled => allow

    tz = ZoneInfo(user["tz"])
    if now_local is None:
        now_local = datetime.now(tz)

    sh, sm = _parse_hhmm_local(start_s)
    eh, em = _parse_hhmm_local(end_s)

    start_dt = now_local.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end_dt = now_local.replace(hour=eh, minute=em, second=0, microsecond=0)

    # Overnight window support, e.g. 22:00 -> 06:00
    if end_dt <= start_dt:
        # window crosses midnight
        if now_local >= start_dt:
            return True
        # else compare with "yesterday start"
        start_dt = start_dt - timedelta(days=1)
        end_dt = end_dt + timedelta(days=1)

    return start_dt <= now_local <= end_dt


async def trade_window_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /trade_window
    View: /trade_window
    Set : /trade_window 09:00 17:30
    Off : /trade_window off
    """
    uid = update.effective_user.id
    user = get_user(uid)

    if not context.args:
        cur_s = (user.get("trade_window_start") or "").strip()
        cur_e = (user.get("trade_window_end") or "").strip()
        if not cur_s or not cur_e:
            await update.message.reply_text(
                "🕒 Trade Window (Email Signals)\n"
                f"{HDR}\n"
                "Current: OFF (no time restriction)\n\n"
                "Set: /trade_window 09:00 17:30\n"
                "Off: /trade_window off"
            )
        else:
            await update.message.reply_text(
                "🕒 Trade Window (Email Signals)\n"
                f"{HDR}\n"
                f"Current: {cur_s} → {cur_e} (local)\n\n"
                "Change: /trade_window 09:00 17:30\n"
                "Off: /trade_window off"
            )
        return

    if len(context.args) == 1 and context.args[0].strip().lower() in {"off", "disable", "none"}:
        update_user(uid, trade_window_start="", trade_window_end="")
        await update.message.reply_text("✅ Trade window is now OFF (emails allowed anytime).")
        return

    if len(context.args) != 2:
        await update.message.reply_text("Usage: /trade_window 09:00 17:30  OR  /trade_window off")
        return

    start_s = context.args[0].strip()
    end_s = context.args[1].strip()

    try:
        _parse_hhmm_local(start_s)
        _parse_hhmm_local(end_s)
    except Exception:
        await update.message.reply_text("Invalid time format. Use HH:MM (24h). Example: /trade_window 09:00 17:30")
        return

    update_user(uid, trade_window_start=start_s, trade_window_end=end_s)
    await update.message.reply_text(f"✅ Trade window set: {start_s} → {end_s} (local time).")


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
/help

────────────────────
1) Market Scan
────────────────────
/screen

Shows a real-time market snapshot:
• Top Trade Setups (highest quality)
• Waiting for Trigger (near-miss candidates)
• Trend Continuation Watch (adaptive EMA logic)
• Directional Leaders
• Directional Losers
• Market Leaders (Top by Futures Volume)

────────────────────
2) Position Sizing (NO trade opened)
────────────────────
/size <SYMBOL> <long|short> sl <STOP>
     [risk <usd|pct> <VALUE>]
     [entry <ENTRY>]

Purpose:
• Calculates correct position size from Risk + SL
• Does NOT open a trade

Examples:
• /size BTC long sl 42000
  → Uses default /riskmode

• /size BTC long risk usd 40 sl 42000
  → Uses current Bybit futures price as Entry

• /size ETH short risk pct 2.5 sl 2480
  → Uses your Equity

• /size BTC long sl 42000 entry 43000
  → Manual entry price

Notes:
• If risk is omitted → /riskmode is used
• pct risk uses Equity
• Qty = Risk / |Entry − SL|

────────────────────
3) Trade Journal & Equity Tracking
────────────────────
Set equity:
• /equity 1000
Reset equity:
• /equity_reset

Open trade:
• /trade_open <SYMBOL> <long|short>
              entry <ENTRY> sl <SL>
              risk <usd|pct> <VALUE>
              [note "..."] [sig <SETUP_ID>]

Manage open trade:
• /trade_sl <TRADE_ID> <NEW_SL>
  → Updates SL and recalculates trade risk
  → Warns if risk increases
  → Adjusts today’s used risk

• /trade_rf <TRADE_ID>
  → Moves SL to Entry (Risk-Free)
  → Sets trade risk to 0
  → Releases today’s used risk immediately

Close trade:
• /trade_close <TRADE_ID> pnl <PNL>

Equity behavior:
• Equity updates ONLY when trades are closed
• If trade closes in profit → its risk is released
• Trade journal is persistent (stored in DB)

────────────────────
4) Status Dashboard
────────────────────
/status

Shows:
• Equity
• Trades today (count)
• Daily risk cap + used / remaining risk
• Active sessions + current session
• Email alert status + email limits
• Open trades list
• Active symbol cooldowns (current session)

────────────────────
5) Risk Settings
────────────────────
Risk per trade:
• /riskmode pct 2.5
• /riskmode usd 25

Daily risk cap:
• /dailycap pct 5
• /dailycap usd 60

────────────────────
6) Limits (Discipline Controls)
────────────────────
/limits maxtrades 5
/limits emailcap 4        (0 = unlimited per session)
/limits emailgap 60       (minutes between emails)
/limits emaildaycap 4     (0 = unlimited per day)

Limits protect against:
• Overtrading
• Email spam
• Emotional decision-making

────────────────────
7) Sessions (Email Delivery Windows)
────────────────────
View sessions:
• /sessions

Enable / disable:
• /sessions_on NY
• /sessions_off LON

Defaults by timezone:
• Americas → NY
• Europe/Africa → LON
• Asia/Oceania → ASIA

Session priority:
NY > LON > ASIA

Emails are sent ONLY during enabled sessions.

────────────────────
8) Email Alerts (On/Off + Time Window)
────────────────────
Enable / disable:
• /notify_on
• /notify_off

Set allowed daily email window (local time):
• /trade_window <START_HH:MM> <END_HH:MM>
  Example:
  • /trade_window 09:00 17:30
  Notes:
  • Signals/emails will only be sent inside this window
  • Uses your /tz local time

Test + diagnostics:
• /email_test
  → Sends a test email (confirms SMTP works)

• /email_decision
  → Shows the last email send/skip decision + reasons

────────────────────
9) Symbol Cooldowns (Anti-Spam Logic)
────────────────────
View cooldowns:
• /cooldowns
  → Shows remaining cooldown time for NY / LON / ASIA for each symbol + direction

Query a single symbol:
• /cooldown <SYMBOL> <long|short>
  → Shows remaining cooldown time for NY / LON / ASIA for that exact symbol+direction

Admin-only reset:
• /cooldown_clear <SYMBOL> <long|short>
• /cooldown_clear_all

────────────────────
10) Reports
────────────────────
Performance:
• /report_daily
• /report_weekly
• /report_overall
  → All-time performance (win rate, net PnL, avg R, expectancy, profit factor)

Signal summaries:
• /signals_daily
• /signals_weekly

────────────────────
11) System Health
────────────────────
• /health
  → Quick bot health check

• /health_sys
  → DB status
  → Bybit/CCXT connectivity
  → Cache stats
  → Session state
  → Email configuration

────────────────────
12) Data Maintenance (Admin)
────────────────────
• /reset
  → Resets data / clean database (admin only)

• /restore
  → Restores previously removed data (admin only)

────────────────────
Final Notes
────────────────────
• PulseFutures does NOT auto-trade
• PulseFutures does NOT promise profits
• PulseFutures enforces discipline, risk control, and session awareness

Trade less. Trade better. Risk first.
PulseFutures

"""

# =========================================================
# TELEGRAM COMMANDS
# =========================================================
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # HELP_TEXT is long -> must be chunked
    await send_long_message(
        update,
        HELP_TEXT,
        parse_mode=None,  # keep plain text (safe)
        disable_web_page_preview=True,
    )

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_help(update, context)


async def health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)

    smtp_err = _LAST_SMTP_ERROR.get(uid, "")
    msg = [
        "🫀 PulseFutures Health",
        HDR,
        f"User TZ: {user['tz']}",
        f"Email Engine: {'ACTIVE' if EMAIL_ENABLED and email_config_ok() else 'OFF/NOT CONFIGURED'}",
        f"SMTP last error: {smtp_err if smtp_err else '-'}",
        HDR,
    ]
    await update.message.reply_text("\n".join(msg).strip())


def db_backup_file() -> Tuple[bool, str]:
    """
    Copies DB_PATH -> DB_BACKUP_PATH
    """
    try:
        # Ensure folder exists
        os.makedirs(os.path.dirname(DB_BACKUP_PATH) or ".", exist_ok=True)

        if not os.path.exists(DB_PATH):
            return False, f"DB file not found: {DB_PATH}"

        # Copy bytes
        with open(DB_PATH, "rb") as src, open(DB_BACKUP_PATH, "wb") as dst:
            dst.write(src.read())

        return True, f"Backup created: {DB_BACKUP_PATH}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def db_restore_file() -> Tuple[bool, str]:
    """
    Copies DB_BACKUP_PATH -> DB_PATH
    """
    try:
        if not os.path.exists(DB_BACKUP_PATH):
            return False, f"No backup found: {DB_BACKUP_PATH}"

        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

        with open(DB_BACKUP_PATH, "rb") as src, open(DB_PATH, "wb") as dst:
            dst.write(src.read())

        return True, f"Restored from backup: {DB_BACKUP_PATH}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def db_wipe_all_data_keep_schema() -> None:
    """
    Deletes ALL rows from core tables, keeps schema.
    Also resets in-memory trackers.
    """
    con = db_connect()
    cur = con.cursor()

    # Order matters if FK ever added later. We currently don’t have FK refs, but still safe.
    tables = [
        "trades",
        "signals",
        "emailed_symbols",
        "email_state",
        "email_daily",
        "risk_daily",
        "setup_counter",
        # users is kept (preferences) OR can be wiped too. You requested "clean database":
        # wiping users means everyone will be re-created on next /start.
        "users",
    ]

    for t in tables:
        try:
            cur.execute(f"DELETE FROM {t}")
        except Exception:
            pass

    con.commit()
    con.close()

    # Also clear runtime trackers
    _LAST_SMTP_ERROR.clear()
    _USER_DIAG_MODE.clear()
    _LAST_EMAIL_DECISION.clear()



async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /reset
    Admin-only:
    1) Backup DB file
    2) Wipe tables (clean DB)
    """
    uid = update.effective_user.id
    if not is_admin_user(uid):
        await update.message.reply_text("⛔️ Admin only.")
        return

    async with DB_FILE_LOCK:
        ok, msg = db_backup_file()
        if not ok:
            await update.message.reply_text(f"❌ Backup failed.\n{msg}")
            return

        try:
            db_wipe_all_data_keep_schema()
        except Exception as e:
            await update.message.reply_text(f"❌ Reset failed.\n{type(e).__name__}: {e}")
            return

    await update.message.reply_text(
        "✅ Database RESET completed.\n"
        f"{msg}\n\n"
        "Use /restore to revert to the backup."
    )


async def restore_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /restore
    Admin-only:
    Restores DB from DB_BACKUP_PATH
    """
    uid = update.effective_user.id
    if not is_admin_user(uid):
        await update.message.reply_text("⛔️ Admin only.")
        return

    async with DB_FILE_LOCK:
        ok, msg = db_restore_file()
        if not ok:
            await update.message.reply_text(f"❌ Restore failed.\n{msg}")
            return

        # After restore, ensure schema migrations still applied
        try:
            db_init()
        except Exception:
            pass

    await update.message.reply_text(f"✅ Database RESTORED.\n{msg}")





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
    
    # ✅ release unused risk if profitable
    if pnl > 0:
        day_local = _user_day_local(user)
        _risk_daily_inc(uid, day_local, -float(t["risk_usd"]))
    
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



async def trade_sl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    tokens = context.args
    if len(tokens) != 2:
        await update.message.reply_text("Usage: /trade_sl <TRADE_ID> <NEW_SL>")
        return

    try:
        trade_id = int(tokens[0])
        new_sl = float(tokens[1])
    except Exception:
        await update.message.reply_text("Invalid arguments.")
        return

    con = db_connect()
    cur = con.cursor()
    cur.execute(
        "SELECT * FROM trades WHERE id=? AND user_id=? AND closed_ts IS NULL",
        (trade_id, uid),
    )
    row = cur.fetchone()
    if not row:
        con.close()
        await update.message.reply_text("Open trade not found.")
        return

    t = dict(row)
    entry = float(t["entry"])
    side = str(t["side"]).upper()

    # validate new SL direction
    if side == "BUY" and new_sl >= entry:
        con.close()
        await update.message.reply_text("For BUY, SL must be below entry.")
        return
    if side == "SELL" and new_sl <= entry:
        con.close()
        await update.message.reply_text("For SELL, SL must be above entry.")
        return

    old_risk = float(t["risk_usd"])
    qty = float(t["qty"])
    new_risk = abs(entry - new_sl) * qty

    # update trade
    cur.execute(
        "UPDATE trades SET sl=?, risk_usd=? WHERE id=? AND user_id=?",
        (float(new_sl), float(new_risk), trade_id, uid),
    )
    con.commit()
    con.close()

    # update daily used risk by delta
    user = get_user(uid)
    day_local = _user_day_local(user)
    delta = float(new_risk) - float(old_risk)

    if abs(delta) > 1e-9:
        _risk_daily_inc(uid, day_local, delta)

        # clamp to 0 if negative due to edge cases
        used_now = _risk_daily_get(uid, day_local)
        if used_now < 0:
            con2 = db_connect()
            cur2 = con2.cursor()
            cur2.execute(
                "UPDATE risk_daily SET used_risk_usd=? WHERE user_id=? AND day_local=?",
                (0.0, uid, day_local),
            )
            con2.commit()
            con2.close()

    cap = daily_cap_usd(user)
    used_today = _risk_daily_get(uid, day_local)
    remaining_today = (cap - used_today) if cap > 0 else float("inf")

    warn = ""
    if new_risk > old_risk + 1e-9:
        warn = "⚠️ Risk increased!"
    elif new_risk < old_risk - 1e-9:
        warn = "✅ Risk reduced (released daily risk)."

    await update.message.reply_text(
        f"✅ Stop Loss UPDATED\n"
        f"- Trade ID: {trade_id}\n"
        f"- Side: {side}\n"
        f"- Entry: {fmt_price(entry)}\n"
        f"- New SL: {fmt_price(new_sl)}\n"
        f"- Old Risk: ${old_risk:.2f}\n"
        f"- New Risk: ${new_risk:.2f}\n"
        f"{warn}\n\n"
        f"📌 Daily Risk\n"
        f"- Cap: ≈ ${cap:.2f}\n"
        f"- Used today: ${used_today:.2f}\n"
        f"- Remaining today: ${max(0.0, remaining_today):.2f}" if cap > 0 else
        f"📌 Daily Risk\n"
        f"- Cap: ≈ ${cap:.2f}\n"
        f"- Used today: ${used_today:.2f}\n"
        f"- Remaining today: ∞"
    )




async def trade_rf_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Usage: /trade_rf <TRADE_ID>")
        return

    trade_id = int(context.args[0])

    con = db_connect()
    cur = con.cursor()
    cur.execute(
        "SELECT * FROM trades WHERE id=? AND user_id=? AND closed_ts IS NULL",
        (trade_id, uid),
    )
    row = cur.fetchone()
    if not row:
        con.close()
        await update.message.reply_text("Open trade not found.")
        return

    t = dict(row)
    entry = float(t["entry"])
    old_risk = float(t["risk_usd"])

    cur.execute(
        "UPDATE trades SET sl=?, risk_usd=? WHERE id=?",
        (entry, 0.0, trade_id),
    )
    con.commit()
    con.close()

    # ✅ Risk Free of the Day
    day_local = _user_day_local(get_user(uid))
    _risk_daily_inc(uid, day_local, -old_risk)

    await update.message.reply_text(
        f"🟢 Trade Risk-Free\n"
        f"- Trade ID: {trade_id}\n"
        f"- SL moved to Entry\n"
        f"- Released Risk: ${old_risk:.2f}"
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
    
    # ✅ Cooldowns (active ones for current session)
    rows = list_cooldowns(uid)
    now_ts = time.time()
    sess_for_cd = now_txt if now_txt != "NONE" else current_session_utc()
    cd_hours = cooldown_hours_for_session(sess_for_cd)
    cd_sec = cd_hours * 3600

    active = []
    seen = set()
    for r in rows:
        sym = str(r["symbol"]).upper()
        side = str(r["side"]).upper()
        key = (sym, side)
        if key in seen:
            continue
        seen.add(key)

        last_ts = float(r["emailed_ts"])
        ago = now_ts - last_ts
        if ago < cd_sec:
            remain = cd_sec - ago
            active.append((sym, side, remain))

    active.sort(key=lambda x: x[2])  # soonest to expire first
    if active:
        lines.append(f"Cooldowns active (session={sess_for_cd}, {cd_hours}h):")
        for sym, side, rem in active[:10]:
            lines.append(f"- {sym} {side} | remaining {_fmt_dur(rem)}")
    else:
        lines.append(f"Cooldowns active (session={sess_for_cd}, {cd_hours}h): none")

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


async def cooldowns_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)

    now_sess = in_session_now(user)
    current_session = (now_sess["name"] if now_sess else current_session_utc())
    rows = list_cooldowns(uid)

    if not rows:
        await update.message.reply_text(
            "⏱ Cooldowns\n"
            f"{HDR}\n"
            "No cooldowns recorded yet."
        )
        return

    # We show remaining cooldown for ALL sessions (NY/LON/ASIA) per (symbol, side)
    now_ts = time.time()
    out = []
    out.append("⏱ Cooldowns (All Sessions)")
    out.append(HDR)
    out.append(f"Current session: {current_session}")
    out.append(f"Policy: NY={cooldown_hours_for_session('NY')}h | LON={cooldown_hours_for_session('LON')}h | ASIA={cooldown_hours_for_session('ASIA')}h")
    out.append(SEP)

    # keep only most recent per (symbol, side)
    seen = set()
    compact = []
    for r in rows:
        key = (str(r["symbol"]).upper(), str(r["side"]).upper())
        if key in seen:
            continue
        seen.add(key)
        compact.append(r)

    # show top 20 recent
    compact = compact[:20]

    for r in compact:
        sym = str(r["symbol"]).upper()
        side = str(r["side"]).upper()
        last_ts = float(r["emailed_ts"])
        ago = now_ts - last_ts

        rem_ny = max(0.0, cooldown_hours_for_session("NY") * 3600 - ago)
        rem_lon = max(0.0, cooldown_hours_for_session("LON") * 3600 - ago)
        rem_asia = max(0.0, cooldown_hours_for_session("ASIA") * 3600 - ago)

        def tag(rem):
            return "✅" if rem <= 0 else "⛔️"

        out.append(
            f"- {sym} {side} | "
            f"NY: {tag(rem_ny)} {_fmt_dur(rem_ny)} | "
            f"LON: {tag(rem_lon)} {_fmt_dur(rem_lon)} | "
            f"ASIA: {tag(rem_asia)} {_fmt_dur(rem_asia)}"
        )

    await update.message.reply_text("\n".join(out))


def clear_cooldown(user_id: int, symbol: str, side: str) -> int:
    """
    Deletes ONE cooldown row for (user, symbol, side).
    Returns deleted row count.
    """
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        DELETE FROM emailed_symbols
        WHERE user_id=? AND symbol=? AND side=?
    """, (int(user_id), str(symbol).upper(), str(side).upper()))
    n = cur.rowcount
    con.commit()
    con.close()
    return int(n)


def clear_all_cooldowns(user_id: int) -> int:
    """
    Deletes ALL cooldown rows for this user.
    Returns deleted row count.
    """
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        DELETE FROM emailed_symbols
        WHERE user_id=?
    """, (int(user_id),))
    n = cur.rowcount
    con.commit()
    con.close()
    return int(n)


async def cooldown_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /cooldown <SYMBOL> <long|short>
    Show remaining cooldown times for NY/LON/ASIA for that symbol+direction.
    """
    uid = update.effective_user.id
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /cooldown <SYMBOL> <long|short>\nExample: /cooldown BTC long")
        return

    sym = re.sub(r"[^A-Za-z0-9]", "", context.args[0]).upper()
    direction = context.args[1].strip().lower()
    if direction not in {"long", "short"}:
        await update.message.reply_text("Second arg must be long or short.\nExample: /cooldown BTC long")
        return

    side = "BUY" if direction == "long" else "SELL"

    # find latest cooldown row for this (sym, side)
    rows = list_cooldowns(uid)
    match = None
    for r in rows:
        if str(r["symbol"]).upper() == sym and str(r["side"]).upper() == side:
            match = r
            break

    if not match:
        await update.message.reply_text(
            "⏱ Cooldown\n"
            f"{HDR}\n"
            f"{sym} {side}\n"
            "No cooldown recorded yet (✅ available)."
        )
        return

    now_ts = time.time()
    last_ts = float(match["emailed_ts"])
    ago = now_ts - last_ts

    rem_ny = max(0.0, cooldown_hours_for_session("NY") * 3600 - ago)
    rem_lon = max(0.0, cooldown_hours_for_session("LON") * 3600 - ago)
    rem_asia = max(0.0, cooldown_hours_for_session("ASIA") * 3600 - ago)

    def tag(rem): return "✅" if rem <= 0 else "⛔️"

    await update.message.reply_text(
        "⏱ Cooldown (per session policy)\n"
        f"{HDR}\n"
        f"{sym} {side}\n"
        f"NY: {tag(rem_ny)} {_fmt_dur(rem_ny)}\n"
        f"LON: {tag(rem_lon)} {_fmt_dur(rem_lon)}\n"
        f"ASIA: {tag(rem_asia)} {_fmt_dur(rem_asia)}"
    )


async def cooldown_clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /cooldown_clear <SYMBOL> <long|short>  (admin only)
    Clears one cooldown row.
    """
    uid = update.effective_user.id
    if not is_admin_user(uid):
        await update.message.reply_text("⛔️ Admin only.")
        return

    if len(context.args) < 2:
        await update.message.reply_text("Usage: /cooldown_clear <SYMBOL> <long|short>\nExample: /cooldown_clear BTC long")
        return

    sym = re.sub(r"[^A-Za-z0-9]", "", context.args[0]).upper()
    direction = context.args[1].strip().lower()
    if direction not in {"long", "short"}:
        await update.message.reply_text("Second arg must be long or short.")
        return

    side = "BUY" if direction == "long" else "SELL"
    n = clear_cooldown(uid, sym, side)

    if n > 0:
        await update.message.reply_text(f"✅ Cooldown cleared: {sym} {side}")
    else:
        await update.message.reply_text(f"ℹ️ No cooldown found for: {sym} {side}")


async def cooldown_clear_all_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /cooldown_clear_all  (admin only)
    Clears all cooldown rows for the admin user.
    """
    uid = update.effective_user.id
    if not is_admin_user(uid):
        await update.message.reply_text("⛔️ Admin only.")
        return

    n = clear_all_cooldowns(uid)
    await update.message.reply_text(f"✅ Cleared {n} cooldown record(s).")



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


def db_trades_all(user_id: int) -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        SELECT * FROM trades
        WHERE user_id=?
        ORDER BY opened_ts ASC
    """, (int(user_id),))
    rows = cur.fetchall() or []
    con.close()
    return [dict(r) for r in rows]


def _profit_factor(trades: List[dict]) -> Optional[float]:
    closed = [t for t in trades if t.get("closed_ts") is not None and t.get("pnl") is not None]
    if not closed:
        return None
    gp = sum(float(t["pnl"]) for t in closed if float(t["pnl"]) > 0)
    gl = abs(sum(float(t["pnl"]) for t in closed if float(t["pnl"]) < 0))
    if gl <= 0:
        return None if gp <= 0 else float("inf")
    return gp / gl


def _expectancy_r(trades: List[dict]) -> Optional[float]:
    closed = [t for t in trades if t.get("closed_ts") is not None and t.get("r_mult") is not None]
    if not closed:
        return None
    rs = [float(t["r_mult"]) for t in closed if t.get("r_mult") is not None]
    return (sum(rs) / len(rs)) if rs else None


async def report_overall_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)

    trades = db_trades_all(uid)
    stats = _stats_from_trades(trades)

    pf = _profit_factor(trades)
    exp_r = _expectancy_r(trades)

    msg = [
        "📊 Overall Report (ALL TIME)",
        HDR,
        f"Closed: {stats['closed_n']} | Wins: {stats['wins']} | Losses: {stats['losses']}",
        f"Win rate: {stats['win_rate']:.1f}%",
        f"Net PnL: {stats['net']:+.2f}",
        f"Avg R: {stats['avg_r']:+.2f}" if stats["avg_r"] is not None else "Avg R: -",
        f"Expectancy (R): {exp_r:+.2f}" if exp_r is not None else "Expectancy (R): -",
        f"Profit Factor: {pf:.2f}" if (pf is not None and pf != float('inf')) else ("Profit Factor: ∞" if pf == float('inf') else "Profit Factor: -"),
        f"Best: {stats['biggest_win']:+.2f} | Worst: {stats['biggest_loss']:+.2f}",
        HDR,
        f"Equity (current): ${float(user['equity']):.2f}",
    ]

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
# User location/time helpers (TZ -> City/Country label)
# =========================================================
def tz_location_label(tz_name: str) -> str:
    """
    Converts IANA TZ like 'Australia/Melbourne' -> 'Melbourne (Australia)'
    """
    tz_name = (tz_name or "").strip()
    if not tz_name or "/" not in tz_name:
        return tz_name or "Unknown"

    parts = tz_name.split("/")
    region = (parts[0] or "").replace("_", " ")
    city = (parts[-1] or "").replace("_", " ")
    if region and city:
        return f"{city} ({region})"
    return city or region or tz_name

def user_location_and_time(user: dict) -> Tuple[str, str]:
    """
    Returns: (location_label, time_str) based on user's tz
    """
    tz_name = str(user.get("tz") or "UTC")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
        tz_name = "UTC"

    now_local = datetime.now(tz)
    loc = tz_location_label(tz_name)
    return loc, now_local.strftime("%Y-%m-%d %H:%M")


# =========================================================
# movers_tables (remove brackets/thresholds from titles)
# =========================================================
def movers_tables(best_fut: Dict[str, MarketVol]) -> Tuple[str, str]:
    up, dn = compute_directional_lists(best_fut)
    up_rows = [[b, fmt_money(v), pct_with_emoji(c24), pct_with_emoji(c4)] for b, v, c24, c4, px in up[:10]]
    dn_rows = [[b, fmt_money(v), pct_with_emoji(c24), pct_with_emoji(c4)] for b, v, c24, c4, px in dn[:10]]

    up_txt = "*Directional Leaders*\n" + (table_md(up_rows, ["SYM", "F Vol", "24H", "4H"]) if up_rows else "_None_")
    dn_txt = "*Directional Losers*\n" + (table_md(dn_rows, ["SYM", "F Vol", "24H", "4H"]) if dn_rows else "_None_")
    return up_txt, dn_txt



# =========================================================
# ✅ MODIFIED: /screen — Premium Telegram UI
# Changes:
# 1) Header shows USER city/location + time (not Melbourne hardcoded)
# 2) Under each signal: remove "Pullback" word + remove pullback line
# 3) Directional Leaders/Losers titles handled in movers_tables() above
# =========================================================
async def screen_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("⏳ Scanning market… Please wait")

        reset_reject_tracker()

        best_fut = await asyncio.to_thread(fetch_futures_tickers)
        if not best_fut:
            await update.message.reply_text("❌ Failed to fetch futures data.")
            return

        # -------------------------------------------------
        # Header / session + user location/time
        # -------------------------------------------------
        uid = update.effective_user.id
        user = get_user(uid)

        session = current_session_utc()

        loc_label, loc_time = user_location_and_time(user)
        header = (
            f"✨ *PulseFutures — Market Scan*\n"
            f"{HDR}\n"
            f"🧠 *Session:* `{session}`   |   📍 *{loc_label}:* `{loc_time}`\n"
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

                # ✅ remove "Pullback" word
                engine_tag = "⚡️ Momentum" if s.engine == "B" else "🎯 Mean-Reversion"

                rr3 = rr_to_tp(s.entry, s.sl, s.tp3)

                tp_line = (
                    f"*TP1:* `{fmt_price(s.tp1)}`  |  "
                    f"*TP2:* `{fmt_price(s.tp2)}`  |  "
                    f"*TP3:* `{fmt_price(s.tp3)}`"
                    if s.tp1 and s.tp2
                    else f"*TP:* `{fmt_price(s.tp3)}`"
                )

                # ✅ remove pullback block entirely
                cards.append(
                    f"╭──────────────╮\n"
                    f"*#{i}* {side_emoji} *{s.side}* — *{s.symbol}*\n"
                    f"╰──────────────╯\n"
                    f"🆔 `{s.setup_id}`  |  *Conf:* `{s.conf}`\n"
                    f"{engine_tag}  |  *RR(TP3):* `{rr3:.2f}`\n"
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
                dot = d.get("dot", "🟡")
                side = d.get("side", "BUY")
                lines.append(f"• *{base}* {dot} `{side}`")
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
                    f"Conf `{t['confidence']}`  |  24H {pct_with_emoji(t['ch24'])}"
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
# ✅ MODIFIED: EMAIL BODY
# Changes:
# 4) TP3 Mode: Trailing only when setup.is_trailing_tp3 True (now engine B + hot)
# 5) Header includes user City/Country + time (same as /screen)
# 6) Setup ID is shown as "ID-<id>"
# =========================================================
def _email_body_pretty(
    session_name: str,
    now_local: datetime,
    user_tz: str,
    setups: List[Setup],
    best_fut: Dict[str, MarketVol],
) -> str:
    loc_label = tz_location_label(user_tz)
    when_str = now_local.strftime("%Y-%m-%d %H:%M")

    parts = []
    parts.append(HDR)
    parts.append(f"📩 PulseFutures • {session_name} • {loc_label}: {when_str} ({user_tz})")
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

        # ✅ "ID-" prefix
        parts.append(f"{i}) ID-{s.setup_id} — {s.side} {s.symbol} — Conf {s.conf}")
        parts.append(f"   Entry: {fmt_price_email(s.entry)} | SL: {fmt_price_email(s.sl)} | RR(TP3): {rr3:.2f}")

        if s.tp1 and s.tp2 and s.conf >= MULTI_TP_MIN_CONF:
            parts.append(
                f"   TP1: {fmt_price_email(s.tp1)} ({TP_ALLOCS[0]}%) | "
                f"TP2: {fmt_price_email(s.tp2)} ({TP_ALLOCS[1]}%) | "
                f"TP3: {fmt_price_email(s.tp3)} ({TP_ALLOCS[2]}%)"
            )
        else:
            parts.append(f"   TP: {fmt_price_email(s.tp3)}")

        # ✅ only for trailing-needed setups
        if s.is_trailing_tp3:
            parts.append("   TP3 Mode: Trailing")

        parts.append(
            f"   24H {pct_with_emoji(s.ch24)} | 4H {pct_with_emoji(s.ch4)} | "
            f"1H {pct_with_emoji(s.ch1)} | 15m {pct_with_emoji(s.ch15)} | Vol~{fmt_money(s.fut_vol_usd)}"
        )
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
    # Prevent overlapping runs (JobQueue can overlap if a run is slow)
    if ALERT_LOCK.locked():
        return

    async with ALERT_LOCK:
        # Keep it quiet if email is off or not configured
        if not EMAIL_ENABLED:
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
                True,                 # strict_15m for email
                sess_name,
                35,                   # email universe
                1.0,                  # no trigger loosen for email
                SCREEN_WAITING_NEAR_PCT,
                False,                # ✅ allow_no_pullback = False for EMAIL
            )

            setups_by_session[sess_name] = setups
            for s in setups:
                db_insert_signal(s)

        # Per-user send / skip logic
        for user in users:
            uid = int(user["user_id"])
            tz = ZoneInfo(user["tz"])

            sess = in_session_now(user)

            # ✅ If user is NOT in an enabled session right now, skip
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

                # ✅ Symbol cooldown (direction-aware + session-aware)
                if symbol_recently_emailed(uid, s.symbol, s.side, sess["name"]):
                    skip_reasons_counter["symbol_cooldown_active"] += 1
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
            subject = f"PulseFutures • {sess['name']} • Premium Setups ({int(st['sent_count']) + 1})"

            ok = await asyncio.to_thread(send_email, subject, body, uid)
            if ok:
                email_state_set(uid, sent_count=int(st["sent_count"]) + 1, last_email_ts=now_ts)
                _email_daily_inc(uid, day_local, 1)

                for s in filtered:
                    mark_symbol_emailed(uid, s.symbol, s.side, sess["name"])

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


async def email_decision_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    d = _LAST_EMAIL_DECISION.get(uid)
    if not d:
        await update.message.reply_text("No email decision recorded yet.")
        return
    await update.message.reply_text(
        "📧 Last Email Decision\n"
        f"Status: {d.get('status')}\n"
        f"When: {d.get('when')}\n"
        f"Reasons:\n- " + "\n- ".join(d.get("reasons", []))
    )



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



async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    # Always log
    logger.exception("Telegram error", exc_info=context.error)

    # Also tell the user (otherwise it looks like "nothing happened")
    try:
        if update and getattr(update, "effective_message", None):
            msg = "⚠️ Something crashed while handling your request."
            # If admin, show the exception text as well
            uid = getattr(getattr(update, "effective_user", None), "id", None)
            if uid is not None and is_admin_user(int(uid)):
                msg += f"\n\nError: {type(context.error).__name__}: {context.error}"
            await update.effective_message.reply_text(msg)
    except Exception:
        # never let error handler crash
        pass



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
    # Hard guard: Background Worker ONLY
    if os.environ.get("RENDER_SERVICE_TYPE") == "web":
        raise SystemExit("Web service detected — polling disabled.")

    # Single instance guard (Render overlap protection)
    render_primary_only()

    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN missing")

    db_init()
    ensure_email_column()
    
    app = Application.builder().token(TOKEN).post_init(_post_init).build()
    app.add_error_handler(error_handler)

    # ================= Handlers =================
    app.add_handler(CommandHandler(["help", "start"], cmd_help))
    app.add_handler(CommandHandler("tz", tz_cmd))
    app.add_handler(CommandHandler("screen", screen_cmd))

    app.add_handler(CommandHandler("equity", equity_cmd))
    app.add_handler(CommandHandler("equity_reset", equity_reset_cmd))
    app.add_handler(CommandHandler("riskmode", riskmode_cmd))
    app.add_handler(CommandHandler("dailycap", dailycap_cmd))
    app.add_handler(CommandHandler("limits", limits_cmd))

    app.add_handler(CommandHandler("trade_sl", trade_sl_cmd))
    app.add_handler(CommandHandler("trade_rf", trade_rf_cmd))
   
    app.add_handler(CommandHandler("sessions", sessions_cmd))
    app.add_handler(CommandHandler("sessions_on", sessions_on_cmd))
    app.add_handler(CommandHandler("sessions_off", sessions_off_cmd))

    app.add_handler(CommandHandler("notify_on", notify_on))
    app.add_handler(CommandHandler("notify_off", notify_off))

    app.add_handler(CommandHandler("size", size_cmd))
    app.add_handler(CommandHandler("trade_open", trade_open_cmd))
    app.add_handler(CommandHandler("trade_close", trade_close_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    app.add_handler(CommandHandler("cooldowns", cooldowns_cmd))
    app.add_handler(CommandHandler("cooldown", cooldown_cmd))
    app.add_handler(CommandHandler("cooldown_clear", cooldown_clear_cmd))
    app.add_handler(CommandHandler("cooldown_clear_all", cooldown_clear_all_cmd))
 
    app.add_handler(CommandHandler("report_daily", report_daily_cmd))
    app.add_handler(CommandHandler("report_overall", report_overall_cmd))  
    app.add_handler(CommandHandler("report_weekly", report_weekly_cmd))
    app.add_handler(CommandHandler("signals_daily", signals_daily_cmd))
    app.add_handler(CommandHandler("signals_weekly", signals_weekly_cmd))

    app.add_handler(CommandHandler("health", health_cmd))

    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(CommandHandler("restore", restore_cmd))
    
    app.add_handler(CommandHandler("health_sys", health_sys_cmd))
    app.add_handler(CommandHandler("trade_window", trade_window_cmd))
    app.add_handler(CommandHandler("email", email_cmd))   
    app.add_handler(CommandHandler("email_test", email_test_cmd))  
    app.add_handler(CommandHandler("email_decision", email_decision_cmd))
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    # ================= JobQueue ================= #
    if app.job_queue:
        app.job_queue.run_repeating(
            alert_job,
            interval=CHECK_INTERVAL_MIN * 60,
            first=30,
            name="alert_job",
        )
    else:
        logger.error("JobQueue NOT available – install python-telegram-bot[job-queue]")

    logger.info("Starting Telegram bot in POLLING mode (Background Worker) ...")

    # Optional: if any other poller exists, don't crash-restart; just sleep.
    from telegram.error import Conflict
    try:
        app.run_polling(
            drop_pending_updates=True,
            close_loop=False,
            allowed_updates=Update.ALL_TYPES,
        )
    except Conflict:
        logger.error("Another instance is polling. Sleeping forever.")
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
