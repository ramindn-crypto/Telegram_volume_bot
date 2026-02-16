_LAST_SCAN_UNIVERSE = []  # bases used for setups in last scan (for /why + filtering)
#!/usr/bin/env python3
"""
PulseFutures — Bybit Futures (Swap) Screener + Signals Email + Risk Manager + Trade Journal (Telegram)
"""

# =========================================================
# IMPORTS
# =========================================================

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
import contextvars

from dataclasses import dataclass
from email.message import EmailMessage
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import Counter, defaultdict

import ccxt
from tabulate import tabulate

import html
import textwrap
from telegram.constants import ParseMode

import difflib

import base64
import tempfile

import os
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)


from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    ApplicationHandlerStop,
)


import re
import sqlite3

TXID_REGEX = re.compile(r"^[A-Fa-f0-9]{64}$")

def is_valid_txid(txid: str) -> bool:
    return bool(TXID_REGEX.match(txid))

def usdt_txid_exists(txid: str) -> bool:
    conn = sqlite3.connect("bot.db")
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM usdt_payments WHERE txid = ?", (txid,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists

def save_usdt_payment(user_id, username, txid, plan):
    conn = sqlite3.connect("bot.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO usdt_payments (telegram_id, username, txid, plan, status)
        VALUES (?, ?, ?, ?, 'PENDING')
    """, (user_id, username, txid, plan))
    conn.commit()
    conn.close()



# =========================================================
# SAAS / STRIPE CONFIG
# =========================================================
import stripe
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

FREE_TRIAL_DAYS = 7

VALID_LICENSE_PREFIX = {
    "PF-STD": "standard",
    "PF-PRO": "pro",
}

# Reuse your existing admin system
ADMIN_IDS = {74935310}  # <-- PUT YOUR TELEGRAM USER ID HERE

ALLOWED_WHEN_LOCKED = {
    "start",
    "help",
    "billing",
    "manage",
    "myplan",
    "license",
    "support",
    "support_status",
}


# =========================================================
# ACCESS / TRIAL / PRO FEATURE GATING
# =========================================================

def _now_ts() -> float:
    return time.time()

def _plan_valid_until_ok(user: dict) -> bool:
    try:
        exp = float(user.get("plan_expires") or 0)
    except Exception:
        exp = 0.0
    return (exp <= 0.0) or (_now_ts() <= exp)

def effective_plan(user_id: int, user: Optional[dict]) -> str:
    """Effective plan with admin override + trial handling."""
    if is_admin_user(int(user_id)):
        return "pro"

    u = user or {}
    plan = str(u.get("plan") or "free").strip().lower()

    if plan in ("pro", "standard") and _plan_valid_until_ok(u):
        return plan

    try:
        until = float(u.get("trial_until") or 0.0)
    except Exception:
        until = 0.0
    if plan == "trial" and until > 0 and _now_ts() <= until:
        return "trial"

    return "free"

def user_has_pro(user_id: int, user: Optional[dict] = None) -> bool:
    u = user if user is not None else (get_user(user_id) or {})
    return effective_plan(int(user_id), u) in ("pro", "trial")

def has_active_access(user_id: int, user: Optional[dict] = None) -> bool:
    u = user if user is not None else (get_user(user_id) or {})
    return effective_plan(int(user_id), u) in ("standard", "pro", "trial")

def ensure_trial_started(user_id: int, user: Optional[dict], force: bool = False) -> None:
    """Starts the 7-day FULL Pro trial ONCE when user runs /start."""
    if is_admin_user(int(user_id)):
        return
    u = user or {}
    plan = str(u.get("plan") or "free").strip().lower()
    try:
        ts_start = float(u.get("trial_start_ts") or 0.0)
    except Exception:
        ts_start = 0.0

    if plan != "free":
        return
    if ts_start > 0:
        return
    if not force:
        return

    now = _now_ts()
    update_user(
        int(user_id),
        plan="trial",
        trial_start_ts=now,
        trial_until=now + float(TRIAL_DAYS) * 86400.0,
        access_source="trial",
        access_ref="start",
        access_updated_ts=now,
    )

PRO_ONLY_COMMANDS = {
    # Email + notifications
    "email", "email_on", "email_off", "email_on_off", "email_test", "email_decision", "notify_on", "notify_off",
    # Unlimited sessions
    "sessions_on_unlimited", "sessions_off_unlimited", "sessions_unlimited_on", "sessions_unlimited_off",
    # Big move alerts
    "bigmove_alert",
    # Advanced reports
    "report_daily", "report_weekly", "report_overall",
}

def enforce_access_or_block(update: Update, command: str) -> bool:
    uid = update.effective_user.id
    if is_admin_user(uid):
        return True

    user = get_user(uid) or {}
    if has_active_access(uid, user):
        return True

    if command in ALLOWED_WHEN_LOCKED:
        return True

    # Non-blocking reply: never stall the event loop for access messages
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(update.message.reply_text(
            "⛔ Access locked.\n\n"
            "Your 7-day free trial has ended.\n\n"
            "To continue, choose a plan:\n"
            "• Standard — $49/month\n"
            "• Pro — $99/month\n\n"
            "👉 /billing"
        ))
    except Exception:
        try:
            # Fallback (best effort)
            update.message.reply_text(
                "⛔ Access locked.\n\n"
                "Your 7-day free trial has ended.\n\n"
                "To continue, choose a plan:\n"
                "• Standard — $49/month\n"
                "• Pro — $99/month\n\n"
                "👉 /billing"
            )
        except Exception:
            pass
    return False

async def _command_guard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Global guard: locks access after trial + gates Pro-only commands."""
    try:
        if not getattr(update, "message", None):
            return
        txt = (update.message.text or "").strip()
        if not txt.startswith("/"):
            return
        cmd = txt.split()[0][1:].split("@")[0].strip().lower()

        # 1) Trial/access lock
        if not enforce_access_or_block(update, cmd):
            raise ApplicationHandlerStop

        # 2) Pro-only commands
        if cmd in PRO_ONLY_COMMANDS:
            uid = update.effective_user.id
            if not user_has_pro(uid):
                
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(update.message.reply_text(
                        "🚀 Pro feature.\n\n"
                        "This command is available in **Pro** (and during your 7-day trial).\n\n"
                        "👉 /billing",
                        parse_mode="Markdown"
                    ))
                except Exception:
                    try:
                        update.message.reply_text(
                            "🚀 Pro feature.\n\n"
                            "This command is available in **Pro** (and during your 7-day trial).\n\n"
                            "👉 /billing",
                            parse_mode="Markdown"
                        )
                    except Exception:
                        pass
                raise ApplicationHandlerStop
    except ApplicationHandlerStop:
        raise
    except Exception:
        return

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

STRIPE_PRICE_TO_PLAN = {
    os.getenv("STRIPE_PRICE_STANDARD"): "standard",
    os.getenv("STRIPE_PRICE_PRO"): "pro",
}

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
TRIGGER_1H_ABS_MIN_BASE = 0.15        # global floor
CONFIRM_15M_ABS_MIN = 0.25
ALIGN_4H_MIN = 0.0
ALIGN_4H_NEUTRAL_ZONE = 0.35  # if |4H| < this, treat regime as neutral (avoid blocking leaders)

# "EARLY" filler (email only)
EARLY_1H_ABS_MIN = 1.8
EARLY_CONF_PENALTY = 4
EARLY_EMAIL_EXTRA_CONF = 4
EARLY_EMAIL_MAX_FILL = 1

# =========================================================
# EMAIL QUALITY GATES (STRICT)
# =========================================================
# Rule 1) Minimum momentum gate (prevents "15m=0%" emails via EARLY path)
EMAIL_EARLY_MIN_CH15_ABS = 0.20   # require at least this abs(15m %) for EARLY emails

# Rule 2) Volume relative filter (killer rule)
EMAIL_ABS_VOL_USD_MIN = 1_000_000     # hard floor (was too strict)
EMAIL_REL_VOL_MIN_MULT = 0.6         # require vol >= 0.6x median (was 1.25x median)

# Rule 3) Priority override (Directional Leaders/Losers first)
EMAIL_PRIORITY_OVERRIDE_ON = True

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
MOMENTUM_MIN_CH1 = 1.3               # pump gate for 1H (easier than before)
MOMENTUM_MIN_24H = 8.0              # must be moving
MOMENTUM_VOL_MULT = 1.2              # volume spike vs mover min
MOMENTUM_ATR_BODY_MULT = 0.95        # expansion vs ATR% (easier)
# ✅ MUCH STRICTER: avoid pump / mid-wave momentum entries
MOMENTUM_MAX_ADAPTIVE_EMA_DIST = 3.5   # percent, was 7.5


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
DEFAULT_MIN_EMAIL_GAP_MIN = 30

# Backward-compat alias
DEFAULT_EMAIL_GAP_MIN = DEFAULT_MIN_EMAIL_GAP_MIN
DEFAULT_MAX_EMAILS_PER_SESSION = 5
DEFAULT_MAX_EMAILS_PER_DAY = 10

DEFAULT_MAX_RISK_PCT_PER_TRADE = 2.0

# =========================================================
# SCAN PROFILES
# =========================================================
# standard = fewer, higher quality
# aggressive = more setups on /screen and more candidates for email filters (still risk-gated)
DEFAULT_SCAN_PROFILE = "standard"
SCAN_PROFILES = {"standard", "aggressive"}

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

# =========================================================
# WIN-RATE FILTERS (stricter = fewer signals, higher quality)
# =========================================================

# 1) Flip-Guard: prevents opposite-direction alerts for same symbol for a period
# Example: SELL then BUY within 2 hours => BUY is blocked (and vice-versa)
FLIP_GUARD_ENABLED = True
FLIP_GUARD_MULT = 1.0  # 1.0 = same as session cooldown hours; try 1.5–2.0 to be stricter

# 2) Higher-TF alignment: require 1H + 4H momentum to agree with the signal direction
TF_ALIGN_ENABLED = True
TF_ALIGN_1H_MIN_ABS = 0.5   # percent
TF_ALIGN_4H_MIN_ABS = 0.5   # percent


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
SCAN_LOCK = asyncio.Lock()  # prevents /screen from blocking other commands under load


# =========================================================
# ✅ PRICE-ACTION CONFIRMATION (stricter = fewer, better signals)
# =========================================================

# Require a clear 15m rejection/reclaim candle at the chosen pullback EMA
REQUIRE_15M_EMA_REJECTION = False  # loosened: more setups

# Candle must close in top/bottom portion of its range (shows strength)
REJECTION_CLOSE_POS_MIN = 0.60   # BUY: close in top 35% of candle
REJECTION_CLOSE_POS_MAX = 0.40   # SELL: close in bottom 35% of candle

# Strong reversal exception (ONLY if you want rare counter-trend calls)
ALLOW_STRONG_REVERSAL_EXCEPTION = True
REVERSAL_CH24_ABS_MIN = 25.0
REVERSAL_CH4_ABS_MIN  = 1.2
REVERSAL_CH1_ABS_MIN  = 0.8



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
    # Non-overlapping UTC windows (DST-proof for Melbourne display)
    # ASIA: 00:00–08:00 UTC
    # LON : 08:00–17:00 UTC
    # NY  : 13:00–22:00 UTC
    "ASIA": {"start": "00:00", "end": "08:00"},
    "LON":  {"start": "08:00", "end": "17:00"},
    "NY":   {"start": "13:00", "end": "22:00"},
}

SESSION_PRIORITY = ["NY", "LON", "ASIA"]

SESSION_MIN_CONF = {
    "NY": 78,
    "LON": 80,
    "ASIA": 82,
}

SESSION_MIN_RR_TP3 = {
    "NY": 1.7,
    "LON": 1.8,
    "ASIA": 1.9,
}

SESSION_EMA_PROX_MULT = {
    "NY": 1.60,
    "LON": 1.40,
    "ASIA": 1.20,
}

SESSION_EMA_REACTION_LOOKBACK = {
    "NY": 9,
    "LON": 7,
    "ASIA": 6,
}

# ✅ 1H trigger loosened per session (overall easier)
SESSION_TRIGGER_ATR_MULT = {
    "NY": 0.55,
    "LON": 0.70,
    "ASIA": 0.85,
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
    Session‑dynamic 1H trigger (loosened so signals actually fire).

    Goal:
    - Still ATR-adaptive (avoid tiny noise when ATR is high)
    - But DO NOT impose huge hard floors that suppress all setups
    """
    knobs = session_knobs(session_name)
    mult_atr = float(knobs["trigger_atr_mult"])

    # ATR-adaptive component: scaled by session knob, gently clamped
    dyn = clamp(float(atr_pct) * float(mult_atr), 0.10, 3.5)

    sess = knobs["name"]
    base_mult = float(SESSION_1H_BASE_MULT.get(sess, 1.0))

    # Global base floor (kept modest)
    base_floor = float(TRIGGER_1H_ABS_MIN_BASE) * base_mult

    # Final trigger is the max of base floor and ATR-adaptive requirement
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

def best_pullback_ema_15m(
    closes_15: List[float],
    c15: List[List[float]],
    entry: float,
    side: str,
    session_name: str,
    atr_1h: float
) -> Tuple[float, int, float]:
    """
    Pick the best EMA among (7,14,21) based on SR behavior (support/resistance),
    NOT simply which is closest.

    Returns: (ema_value, period, dist_pct)

    Scoring features:
    - EMA slope aligned with side (trend rail)
    - Touch+rejection count in recent candles (SR respect)
    - Penalize "cross-through" candles (EMA not acting as SR)
    - Prefer closer EMA but not at the expense of SR quality
    """
    if not closes_15 or not c15 or entry <= 0:
        return 0.0, 14, 999.0

    knobs = session_knobs(session_name)
    lookback_n = int(knobs.get("ema_reaction_lookback", 7))
    lookback_n = int(clamp(lookback_n, 5, 16))

    candidates = [7, 14, 21]
    best_val = 0.0
    best_p = 14
    best_dist = 999.0
    best_score = -1e9

    # Precompute candle slice (most recent lookback window)
    recent = c15[-lookback_n:] if len(c15) >= lookback_n else c15[:]

    for p in candidates:
        lookback = min(len(closes_15), p + 80)
        ema_series_src = closes_15[-lookback:]
        e_now = float(ema(ema_series_src, p) or 0.0)
        if e_now <= 0:
            continue

        # EMA slope: compare current EMA with EMA a bit earlier
        # (cheap slope approximation without building full series)
        mid_cut = max(10, int(p * 2))
        if len(ema_series_src) > mid_cut:
            e_prev = float(ema(ema_series_src[:-mid_cut], p) or e_now)
        else:
            e_prev = e_now

        slope = (e_now - e_prev)  # absolute slope
        slope_ok = (slope > 0) if side == "BUY" else (slope < 0)

        # Distance (still matters)
        dist_pct = abs(entry - e_now) / entry * 100.0

        # Dynamic near threshold based on ATR + session knobs
        # Use your existing proximity function to get threshold behavior consistent
        pb_ok, pb_dist2, pb_thr, _ = ema_support_proximity_ok(entry, e_now, atr_1h, session_name)

        # Count touches + "rejection direction" + cross-through
        touch = 0
        reject = 0
        cross = 0

        for k in recent:
            o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4])
            # touch = candle range intersects EMA
            if l <= e_now <= h:
                touch += 1

                # reject = candle closes on the "right side" of EMA for continuation
                if side == "BUY":
                    if c > e_now and c > o:
                        reject += 1
                else:
                    if c < e_now and c < o:
                        reject += 1

                # cross-through = opens one side closes other side (EMA not respected)
                if (o - e_now) * (c - e_now) < 0:
                    cross += 1

        # Score weights (tuned for "fewer, better")
        score = 0.0

        # Must be meaningfully near; if not near, heavily penalize
        if not pb_ok:
            score -= 25.0
        else:
            score += 10.0

        # Reward SR respect: touches + rejection, penalize cross-through
        score += (touch * 2.0)
        score += (reject * 4.0)
        score -= (cross * 5.0)

        # Reward slope alignment (trend rail)
        score += 8.0 if slope_ok else -10.0

        # Prefer closer EMA slightly (but not dominant)
        score -= (dist_pct * 1.5)

        # Mild preference for slower EMA in trends (21 > 14 > 7) IF slope aligns
        if slope_ok:
            if p == 21:
                score += 2.5
            elif p == 14:
                score += 1.2

        # Pick best
        if score > best_score:
            best_score = score
            best_val = e_now
            best_p = int(p)
            best_dist = float(dist_pct)

    if best_val <= 0:
        return 0.0, 14, 999.0

    return float(best_val), int(best_p), float(best_dist)


# ✅ NEW: "Waiting for Trigger" (near-miss candidates)

_WAITING_TRIGGER: Dict[str, Dict[str, Any]] = {}


def is_admin_user(user_id: int) -> bool:
    """Admin check (supports list + single-id env fallbacks)."""
    try:
        uid = int(user_id)
    except Exception:
        return False

    for k in ("ADMIN_TELEGRAM_ID", "OWNER_USER_ID", "ADMIN_ID"):
        v = os.environ.get(k)
        if v:
            try:
                if uid == int(str(v).strip()):
                    return True
            except Exception:
                pass

    return (uid in ADMIN_USER_IDS) if ADMIN_USER_IDS else False


def effective_plan(user_id: int, user: Optional[dict]) -> str:
    """Effective plan, forcing admin to PRO."""
    if is_admin_user(user_id):
        return "pro"
    return str((user or {}).get("plan") or "free").strip().lower()


def is_unlimited_admin(user_id: int) -> bool:
    return is_admin_user(user_id)

# =========================================================
# GLOBAL STATE
# =========================================================
# Stores last SMTP error per user for /health and /email_test
_LAST_SMTP_ERROR: Dict[int, str] = {}

# Per-user diagnostics preference (non-admin). Admin always full.
_USER_DIAG_MODE: Dict[int, str] = {}

# ✅ NEW: Stores last email decision per user (SENT / SKIP + reasons)
_LAST_EMAIL_DECISION: Dict[int, Dict[str, Any]] = {}

_LAST_BIGMOVE_DECISION: Dict[int, dict] = {}

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


# =========================================================
# DIAGNOSTICS (reject reasons)
# =========================================================
_REJECT_CTX = contextvars.ContextVar("pf_reject_ctx", default=None)
_LAST_REJECTS = {}  # uid -> {"ts": float, "counts": { "A:no_trigger": 12, ... }}
_GLOBAL_REJECT_CTX = None  # fallback reject ctx when contextvars don't propagate across nested threads

def _rej(reason: str, base: str, mv: "MarketVol", extra: str = "") -> None:
    """Record reject reasons for diagnostics.

    - Aggregate counters: ctx["<BASE>:<REASON>"] += 1
    - Per-symbol last-known reason: ctx["__per__"][BASE] = {"reason": REASON, "n": count}
    - If ctx["__allow__"] exists, only record for bases in that set (keeps /why focused).
    """
    ctx = _REJECT_CTX.get()
    if not isinstance(ctx, dict):
        # Fallback: nested asyncio.to_thread() inside our thread runner can drop contextvars.
        # Use the global ctx for this scan if available.
        global _GLOBAL_REJECT_CTX
        if isinstance(_GLOBAL_REJECT_CTX, dict):
            ctx = _GLOBAL_REJECT_CTX
        else:
            return

    b = str(base or "").upper().strip()
    if not b:
        return

    allow = ctx.get("__allow__")
    try:
        # If allow-list is present but empty, treat it as disabled (avoid silencing /why)
        if allow is not None:
            _allow_set = set(allow)
            if len(_allow_set) > 0 and b not in _allow_set:
                return
    except Exception:
        pass

    r = str(reason or "").strip()
    if not r:
        r = "unknown_reject"

    key = f"{b}:{r}"
    ctx[key] = int(ctx.get(key, 0)) + 1

    try:
        per = ctx.get("__per__")
        if not isinstance(per, dict):
            per = {}
            ctx["__per__"] = per
        per_item = per.get(b) or {}
        n = int(per_item.get("n") or 0) + 1
        per[b] = {"reason": r, "n": n}
    except Exception:
        pass
    return



def _note_status(status: str, base: str, mv: "MarketVol", extra: str = "") -> None:
    """Record non-reject per-symbol status for diagnostics (/why).

    This uses the same reject context plumbing as _rej(), but does NOT increment aggregate reject counters.
    It only sets ctx["__per__"][BASE] so we don't show '(not evaluated / no decision)' for symbols that actually
    passed gates or produced a setup.
    """
    ctx = _REJECT_CTX.get()
    if not isinstance(ctx, dict):
        global _GLOBAL_REJECT_CTX
        if isinstance(_GLOBAL_REJECT_CTX, dict):
            ctx = _GLOBAL_REJECT_CTX
        else:
            return
    b = str(base or "").upper().strip()
    if not b:
        return
    allow = ctx.get("__allow__")
    try:
        if allow is not None:
            _allow_set = set(allow)
            if len(_allow_set) > 0 and b not in _allow_set:
                return
    except Exception:
        pass
    st = str(status or "").strip() or "status"
    try:
        per = ctx.get("__per__")
        if not isinstance(per, dict):
            per = {}
            ctx["__per__"] = per
        per_item = per.get(b) or {}
        n = int(per_item.get("n") or 0) + 1
        # keep 'reason' key for backward compatibility in /why renderer
        per[b] = {"reason": st, "n": n}
    except Exception:
        pass
    return

def _reject_report_for_uid(uid: int, top_n: int = 12) -> str:
    """Explain why setups were rejected in the *last* scan for this user.

    Output is intentionally compact:
    - Shows how many symbols were in-scope (leaders/losers + optionally market leaders)
    - Shows how many had recorded reject reasons
    - Shows top reject reasons (aggregate)
    - Shows a per-symbol last-known reason list (limited)
    """
    rec = _LAST_REJECTS.get(int(uid)) or {}
    counts = rec.get("counts") or {}
    allow = rec.get("allow") or []
    per_sym = rec.get("per_symbol") or {}  # base -> {"reason": str, "n": int}

    if not allow and not counts:
        return "No reject stats recorded yet. Run /screen once."

    allow_set = [str(x).upper() for x in (allow or []) if str(x).strip()]
    allow_set_unique = []
    seen = set()
    for b in allow_set:
        if b not in seen:
            seen.add(b)
            allow_set_unique.append(b)

    # Filter counts to allow-set if we have it (keeps /why focused).
    filtered_counts = {}
    if allow_set_unique:
        allow_s = set(allow_set_unique)
        for k, v in (counts or {}).items():
            base = str(k).split(":", 1)[0].upper().strip()
            if base in allow_s:
                filtered_counts[k] = v
    else:
        filtered_counts = dict(counts or {})

    # Aggregate top reject keys
    items = sorted(filtered_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:int(top_n)]

    # Per-symbol summary (show all if small; otherwise truncate)
    per_lines = []
    allow_s = set(allow_set_unique) if allow_set_unique else None
    if allow_set_unique:
        bases = allow_set_unique
    else:
        # fallback: infer bases from counts
        bases = sorted(list({str(k).split(":", 1)[0].upper().strip() for k in (filtered_counts or {}).keys()}))

    for b in bases:
        info = per_sym.get(b) or per_sym.get(b.upper()) or None
        if info and isinstance(info, dict):
            r = str(info.get("reason") or "").strip()
            n = int(info.get("n") or 0)
            if r:
                per_lines.append(f"• {b}: {r} ({n})")
                continue
        # No recorded reject for this symbol in the last scan
        per_lines.append(f"• {b}: (not evaluated / no decision)")

    # If the universe is large, keep per-symbol section readable
    max_per = 20
    per_tail = ""
    if len(per_lines) > max_per:
        per_tail = f"… (+{len(per_lines) - max_per} more)"
        per_lines = per_lines[:max_per]

    lines = []
    lines.append("🧩 Last Scan Reject Reasons")
    if allow_set_unique:
        lines.append(f"Universe (leaders/losers/market leaders): {len(allow_set_unique)} symbols")
        lines.append("Symbols: " + ", ".join(allow_set_unique[:30]) + ("…" if len(allow_set_unique) > 30 else ""))
    lines.append(f"Recorded reject keys: {len(filtered_counts)}")
    if items:
        lines.append("")
        lines.append("Top reasons (aggregate):")
        for k, v in items:
            lines.append(f"• {k} = {v}")
    lines.append("")
    lines.append("Per-symbol (last known):")
    lines.extend(per_lines)
    if per_tail:
        lines.append(per_tail)
    return "\n".join(lines)



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
    # ✅ Users table (core) — created if missing (fresh installs)
    # =========================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        tz TEXT DEFAULT 'UTC',
        scan_profile TEXT DEFAULT 'default',
        equity REAL DEFAULT 1000.0,
        risk_mode TEXT DEFAULT 'percent',
        risk_value REAL DEFAULT 1.0,
        daily_cap_mode TEXT DEFAULT 'percent',
        daily_cap_value REAL DEFAULT 3.0,
        max_trades_day INTEGER DEFAULT 3,
        notify_on INTEGER DEFAULT 0,

        sessions_enabled TEXT DEFAULT '',
        max_emails_per_session INTEGER DEFAULT 5,
        email_gap_min INTEGER DEFAULT 30,
        max_emails_per_day INTEGER DEFAULT 10,

        day_trade_date TEXT DEFAULT '',
        day_trade_count INTEGER DEFAULT 0,

        -- Email columns
        email_to TEXT,
        email_alerts_enabled INTEGER DEFAULT 1,

        -- Billing / access columns (backward compatible)
        plan TEXT DEFAULT 'free',
        trial_start_ts REAL DEFAULT 0,
        trial_until REAL DEFAULT 0,
        plan_expires REAL DEFAULT 0,
        access_source TEXT,
        access_ref TEXT,
        access_updated_ts REAL DEFAULT 0
    )
    """)

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
    
    # Trade window columns (local time HH:MM), empty = disabled
    if "trade_window_start" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN trade_window_start TEXT NOT NULL DEFAULT ''")
    if "trade_window_end" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN trade_window_end TEXT NOT NULL DEFAULT ''")
    
    
    # Scan profile (standard/aggressive)
    if "scan_profile" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN scan_profile TEXT NOT NULL DEFAULT 'standard'")
# Daily email cap
    if "max_emails_per_day" not in cols:
        cur.execute(
            f"ALTER TABLE users ADD COLUMN max_emails_per_day INTEGER NOT NULL DEFAULT {int(DEFAULT_MAX_EMAILS_PER_DAY)}"
        )
    
    # NEW: Unlimited session emailing (24h)
    if "sessions_unlimited" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN sessions_unlimited INTEGER NOT NULL DEFAULT 0")
    
    # NEW: Big-move alert emails (even if not a valid setup)
    if "bigmove_alert_on" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN bigmove_alert_on INTEGER NOT NULL DEFAULT 1")
    
    # Aligned defaults (per your request): 24H >= 40 OR 4H >= 15
    # Aligned defaults: 4H >= 20 OR 1H >= 10
    if "bigmove_alert_4h" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN bigmove_alert_4h REAL NOT NULL DEFAULT 20")
    if "bigmove_alert_1h" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN bigmove_alert_1h REAL NOT NULL DEFAULT 10")

    # NEW: Spike Reversal Alerts
    if "spike_alert_on" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN spike_alert_on INTEGER NOT NULL DEFAULT 1")

    # default volume gate: 15M
    if "spike_min_vol_usd" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN spike_min_vol_usd REAL NOT NULL DEFAULT 10000000")

    # wick ratio threshold (0.55 means wick is 55%+ of candle range)
    if "spike_wick_ratio" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN spike_wick_ratio REAL NOT NULL DEFAULT 0.45")

    # spike size must be >= ATR * this multiplier
    if "spike_atr_mult" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN spike_atr_mult REAL NOT NULL DEFAULT 1.00")

    # NEW: Early Warning (Possible Reversal Zones) email alerts
    # Default OFF to avoid inbox noise; users can enable explicitly.

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

    cur.execute("""
        CREATE TABLE IF NOT EXISTS emailed_bigmoves (
            user_id     INTEGER NOT NULL,
            symbol      TEXT    NOT NULL,
            direction   TEXT    NOT NULL,   -- "UP" or "DOWN"
            emailed_ts  REAL    NOT NULL,
            PRIMARY KEY (user_id, symbol, direction)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS emailed_spikes (
            user_id     INTEGER NOT NULL,
            symbol      TEXT    NOT NULL,
            direction   TEXT    NOT NULL,   -- "UP" or "DOWN"
            emailed_ts  REAL    NOT NULL,
            PRIMARY KEY (user_id, symbol, direction)
        )
    """)


    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS emailed_earlywarnings (
            user_id     INTEGER NOT NULL,
            symbol      TEXT    NOT NULL,
            side        TEXT    NOT NULL,   -- "BUY" or "SELL"
            emailed_ts  REAL    NOT NULL,
            PRIMARY KEY (user_id, symbol, side)
        )
    """
    )

    # =========================================================
    # USDT payments + unified access/payments ledger
    # =========================================================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS usdt_payments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        telegram_id INTEGER NOT NULL,
        username TEXT,
        txid TEXT UNIQUE NOT NULL,
        plan TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PENDING',   -- PENDING / APPROVED / REJECTED
        created_ts REAL NOT NULL,
        decided_ts REAL,
        decided_by INTEGER,
        note TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS payments_ledger (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        source TEXT NOT NULL,         -- 'stripe' | 'usdt' | 'manual'
        ref TEXT NOT NULL,            -- stripe invoice/sub id OR txid OR manual note
        plan TEXT NOT NULL,           -- free | standard | pro
        amount REAL NOT NULL DEFAULT 0,
        currency TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL,         -- paid | refunded | void | pending
        created_ts REAL NOT NULL
    )
    """)

    # =========================================================
    # Optional: add access tracking columns to users table
    # =========================================================
    try:
        cur.execute("PRAGMA table_info(users)")
        user_cols = {r[1] for r in cur.fetchall()}

        if "access_source" not in user_cols:
            cur.execute(
                "ALTER TABLE users ADD COLUMN access_source TEXT NOT NULL DEFAULT ''"
            )

        if "access_ref" not in user_cols:
            cur.execute(
                "ALTER TABLE users ADD COLUMN access_ref TEXT NOT NULL DEFAULT ''"
            )

        if "access_updated_ts" not in user_cols:
            cur.execute(
                "ALTER TABLE users ADD COLUMN access_updated_ts REAL NOT NULL DEFAULT 0"
            )

    except Exception:
        # Do not block startup if ALTER TABLE fails
        pass
   
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
        tz_name = os.environ.get("DEFAULT_USER_TZ", "UTC")
        sessions = ['NY']
        now_local = datetime.now(ZoneInfo(tz_name)).date().isoformat()
        cur.execute("""
            INSERT INTO users (
                user_id, tz, scan_profile, equity, risk_mode, risk_value,
                daily_cap_mode, daily_cap_value,
                max_trades_day, notify_on,
                sessions_enabled, max_emails_per_session, email_gap_min,
                max_emails_per_day,
                day_trade_date, day_trade_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            tz_name,
            str(DEFAULT_SCAN_PROFILE),
            float(DEFAULT_EQUITY),
            DEFAULT_RISK_MODE,
            float(DEFAULT_RISK_VALUE),
            DEFAULT_DAILY_CAP_MODE,
            float(DEFAULT_DAILY_CAP_VALUE),
            int(DEFAULT_MAX_TRADES_DAY),
            
            # ✅ FIX: default notify_on should NOT depend on EMAIL_ENABLED
            1,
            
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

# =========================================================
# ACCESS CONTROL (FREE TRIAL + PAYWALL)
# =========================================================
def trial_expired(user: dict) -> bool:
    try:
        created = datetime.fromisoformat(user["created_at"])
    except Exception:
        return True
    return datetime.utcnow() > created + timedelta(days=FREE_TRIAL_DAYS)


def has_active_access(user: dict) -> bool:
    if user.get("plan") in ("standard", "pro"):
        return True
    if user.get("plan") == "free" and not trial_expired(user):
        return True
    return False


def enforce_access_or_block(update: Update, command: str) -> bool:
    uid = update.effective_user.id
    if is_admin_user(uid):
        return True

    user = get_user(uid)

    if has_active_access(user, uid):
        return True

    if command in ALLOWED_WHEN_LOCKED:
        return True

    update.message.reply_text(
        "⛔ Access locked.\n\n"
        "Your 7-day free trial has ended.\n\n"
        "Plans:\n"
        "• Standard — $50/month\n"
        "• Pro — $99/month\n\n"
        "👉 /billing"
    )
    return False

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
    """Backward-compatible email + access columns migration.

    Also ensures billing/access columns exist (plan/trial/etc), because many commands
    (e.g. /status, /screen gating) depend on them.
    """
    # Ensure billing/access columns exist first
    ensure_billing_columns()

    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()

        # email address
        try:
            cur.execute("ALTER TABLE users ADD COLUMN email_to TEXT")
            con.commit()
            logger.info("Added email_to column to users table")
        except sqlite3.OperationalError:
            pass

        # per-user email preferences
        try:
            cur.execute("ALTER TABLE users ADD COLUMN email_enabled INTEGER DEFAULT 1")
            con.commit()
            logger.info("Added email_enabled column to users table")
        except sqlite3.OperationalError:
            pass

        try:
            cur.execute("ALTER TABLE users ADD COLUMN notify_on INTEGER DEFAULT 1")
            con.commit()
            logger.info("Added notify_on column to users table")
        except sqlite3.OperationalError:
            pass

        try:
            cur.execute("ALTER TABLE users ADD COLUMN bigmove_alert INTEGER DEFAULT 1")
            con.commit()
            logger.info("Added bigmove_alert column to users table")
        except sqlite3.OperationalError:
            pass

        try:
            cur.execute("ALTER TABLE users ADD COLUMN bigmove_min_usd REAL DEFAULT 5000000")
            con.commit()
            logger.info("Added bigmove_min_usd column to users table")
        except sqlite3.OperationalError:
            pass

        try:
            cur.execute("ALTER TABLE users ADD COLUMN sessions_unlimited INTEGER DEFAULT 0")
            con.commit()
            logger.info("Added sessions_unlimited column to users table")
        except sqlite3.OperationalError:
            pass

def ensure_billing_columns():
    """Ensure billing / plan / trial columns exist on users table (backward compatible migrations)."""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()

        # Plan / trial columns
        for ddl, label in [
            ("ALTER TABLE users ADD COLUMN plan TEXT DEFAULT 'free'", "plan"),
            ("ALTER TABLE users ADD COLUMN trial_start_ts REAL DEFAULT 0", "trial_start_ts"),
            ("ALTER TABLE users ADD COLUMN trial_until REAL DEFAULT 0", "trial_until"),
            ("ALTER TABLE users ADD COLUMN plan_expires REAL DEFAULT 0", "plan_expires"),
            ("ALTER TABLE users ADD COLUMN access_source TEXT", "access_source"),
            ("ALTER TABLE users ADD COLUMN access_ref TEXT", "access_ref"),
            ("ALTER TABLE users ADD COLUMN access_updated_ts REAL DEFAULT 0", "access_updated_ts"),
        ]:
            try:
                cur.execute(ddl)
                con.commit()
                logger.info("Added %s column to users table", label)
            except sqlite3.OperationalError:
                # column already exists (or older sqlite limitation)
                pass

def reset_daily_if_needed(user: dict) -> dict:
    tz = ZoneInfo(user["tz"])
    today = datetime.now(tz).date().isoformat()
    if user["day_trade_date"] != today:
        update_user(user["user_id"], day_trade_date=today, day_trade_count=0)
        user = get_user(user["user_id"])
    return user

def list_users_notify_on() -> List[dict]:
    """Users who should receive scan emails. Admins are always included."""
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE notify_on=1")
    rows = [dict(r) for r in cur.fetchall()]

    # Pro-only: scan emails are available only for Pro users (and active Trial)
    rows = [r for r in rows if effective_plan(int(r.get("user_id") or 0), r) in ("pro", "trial")]

    # Add admins with an email address saved, even if notify_on=0
    try:
        if ADMIN_USER_IDS:
            placeholders = ",".join(["?"] * len(ADMIN_USER_IDS))
            cur.execute(
                f"SELECT * FROM users WHERE user_id IN ({placeholders}) AND ( (email IS NOT NULL AND TRIM(email)!='') OR (email_to IS NOT NULL AND TRIM(email_to)!='') )",
                [int(x) for x in ADMIN_USER_IDS],
            )
            for r in cur.fetchall():
                d = dict(r)
                if not any(int(x.get("user_id") or 0) == int(d.get("user_id") or 0) for x in rows):
                    rows.append(d)
    except Exception:
        pass

    con.close()
    return rows


def list_users_with_email() -> List[dict]:
    """
    Users eligible for email sends (have a saved recipient email).
    This is used for Big-Move Alerts so it works even if notify_on=0.

    IMPORTANT:
    Some DB versions don't have users.email (only users.email_to).
    This function auto-detects columns and builds a safe query.
    """
    con = db_connect()
    try:
        # Ensure rows are dict-convertible
        try:
            import sqlite3
            con.row_factory = sqlite3.Row
        except Exception:
            pass

        cur = con.cursor()

        # Detect available columns in users table
        try:
            cur.execute("PRAGMA table_info(users)")
            cols = {str(r[1]).lower() for r in cur.fetchall()}  # r[1] is column name
        except Exception:
            cols = set()

        has_email_to = ("email_to" in cols)
        has_email = ("email" in cols)
        has_email_enabled = ("email_alerts_enabled" in cols)

        # Build safe WHERE clause
        where_parts = []
        if has_email_to:
            where_parts.append("(email_to IS NOT NULL AND TRIM(email_to) != '')")
        if has_email:
            where_parts.append("(email IS NOT NULL AND TRIM(email) != '')")

        # If neither column exists, no one is eligible
        if not where_parts:
            return []

        where_sql = " OR ".join(where_parts)
        if has_email_enabled:
            where_sql = f"({where_sql}) AND (COALESCE(email_alerts_enabled, 1) = 1)"

        cur.execute(f"""
            SELECT *
            FROM users
            WHERE {where_sql}
        """)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        try:
            con.close()
        except Exception:
            pass


    
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

def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s == "-" or s.lower() == "none":
            return default
        return float(s)
    except Exception:
        return default



BIGMOVE_COOLDOWN_SEC = 60 * 60 * 3  # 3 hours

def bigmove_recently_emailed(uid: int, symbol: str, direction: str) -> bool:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT emailed_ts FROM emailed_bigmoves WHERE user_id=? AND symbol=? AND direction=?",
                (int(uid), str(symbol), str(direction)),
            )
            row = cur.fetchone()
            if not row:
                return False
            last_ts = float(row[0] or 0.0)
            return (time.time() - last_ts) < BIGMOVE_COOLDOWN_SEC
    except Exception:
        return False

def mark_bigmove_emailed(uid: int, symbol: str, direction: str) -> None:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO emailed_bigmoves(user_id, symbol, direction, emailed_ts) VALUES(?,?,?,?)",
                (int(uid), str(symbol), str(direction), time.time()),
            )
            conn.commit()
    except Exception:
        pass

SPIKE_COOLDOWN_SEC = 60 * 60 * 3  # 3 hours

def spike_recently_emailed(uid: int, symbol: str, direction: str) -> bool:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT emailed_ts FROM emailed_spikes WHERE user_id=? AND symbol=? AND direction=?",
                (int(uid), str(symbol), str(direction)),
            )
            row = cur.fetchone()
            if not row:
                return False
            last_ts = float(row[0] or 0.0)
            return (time.time() - last_ts) < SPIKE_COOLDOWN_SEC
    except Exception:
        return False

def mark_spike_emailed(uid: int, symbol: str, direction: str) -> None:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO emailed_spikes(user_id, symbol, direction, emailed_ts) VALUES(?,?,?,?)",
                (int(uid), str(symbol), str(direction), time.time()),
            )
            conn.commit()
    except Exception:
        pass


EARLYWARN_COOLDOWN_SEC = 60 * 60 * 3  # 3 hours

def earlywarn_recently_emailed(uid: int, symbol: str, side: str) -> bool:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT emailed_ts FROM emailed_earlywarnings WHERE user_id=? AND symbol=? AND side=?",
                (int(uid), str(symbol), str(side).upper()),
            )
            row = cur.fetchone()
            if not row:
                return False
            last_ts = float(row[0] or 0.0)
            return (time.time() - last_ts) < EARLYWARN_COOLDOWN_SEC
    except Exception:
        return False

def mark_earlywarn_emailed(uid: int, symbol: str, side: str) -> None:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO emailed_earlywarnings(user_id, symbol, side, emailed_ts) VALUES(?,?,?,?)",
                (int(uid), str(symbol), str(side).upper(), time.time()),
            )
            conn.commit()
    except Exception:
        pass

def _bigmove_candidates(best_fut: dict, p4: float, p1: float, min_vol_usd: float = 0.0, max_items: int = 12) -> list:
    """
    Returns list of dicts: {symbol, ch4, ch1, vol, direction, score}

    direction:
      - "UP"   → strong positive move
      - "DOWN" → strong negative move

    Triggers:
      - UP   if ch4 >= +p4 OR ch1 >= +p1
      - DOWN if ch4 <= -p4 OR ch1 <= -p1
    """

    def _pick_pct(mv, keys) -> float:
        for k in keys:
            try:
                v = getattr(mv, k, None)
                if v is None:
                    continue
                return float(v or 0.0)
            except Exception:
                continue
        return 0.0

    out = []

    for sym, mv in (best_fut or {}).items():
        try:
            # Try multiple possible field names (fixes "different field names" issue)
            ch4 = _pick_pct(mv, ["ch4", "pct_4h", "change_4h", "chg_4h", "percentage_4h", "p4", "h4"])
            ch1 = _pick_pct(mv, ["ch1", "pct_1h", "change_1h", "chg_1h", "percentage_1h", "p1", "h1"])

            # ✅ FIX: correct 24H USD volume (MarketVol does NOT have fut_vol_usd)
            vol = float(usd_notional(mv) or 0.0)

            # If still missing, compute from 1h candles (same logic as compute_metrics-style)
            if (abs(ch4) < 1e-9 and abs(ch1) < 1e-9) and getattr(mv, "symbol", None):
                try:
                    c1 = fetch_ohlcv(mv.symbol, "1h", 6)
                    if c1 and len(c1) >= 2:
                        closes_1h = [float(x[4]) for x in c1]
                        c_last = closes_1h[-1]
                        c_prev1 = closes_1h[-2]
                        c_prev4 = closes_1h[-5] if len(closes_1h) >= 5 else closes_1h[0]
                        ch1 = ((c_last - c_prev1) / c_prev1) * 100.0 if c_prev1 else 0.0
                        ch4 = ((c_last - c_prev4) / c_prev4) * 100.0 if c_prev4 else 0.0
                except Exception:
                    pass

        except Exception:
            continue

        up_hit = (ch4 >= float(p4)) or (ch1 >= float(p1))
        down_hit = (ch4 <= -float(p4)) or (ch1 <= -float(p1))

        if not (up_hit or down_hit):
            continue

        # ✅ NEW: volume gate (skip low-volume coins completely)
        if float(min_vol_usd or 0.0) > 0.0 and vol < float(min_vol_usd):
            continue

        if down_hit and not up_hit:
            direction = "DOWN"
        elif up_hit and not down_hit:
            direction = "UP"
        else:
            direction = "UP" if ch1 >= 0 else "DOWN"

        score_up = max(
            (abs(ch4) / max(p4, 1e-9)) if ch4 > 0 else 0.0,
            (abs(ch1) / max(p1, 1e-9)) if ch1 > 0 else 0.0,
        )
        score_dn = max(
            (abs(ch4) / max(p4, 1e-9)) if ch4 < 0 else 0.0,
            (abs(ch1) / max(p1, 1e-9)) if ch1 < 0 else 0.0,
        )
        score = max(score_up, score_dn)

        out.append({
            "symbol": sym,
            "ch4": ch4,
            "ch1": ch1,
            "vol": vol,
            "direction": direction,
            "score": score,
        })

    out.sort(key=lambda x: (x["score"], x["vol"]), reverse=True)
    return out[:max_items]


def _spike_reversal_candidates(
    best_fut: Dict[str, Any],
    min_vol_usd: float = 15_000_000.0,
    wick_ratio_min: float = 0.55,
    atr_mult_min: float = 1.20,
    max_items: int = 10,
) -> List[dict]:
    """
    Detects wick-spike rejection in the direction opposite to the spike, aligned with the bigger trend.
    Downtrend: upper-wick spike rejection -> SELL
    Uptrend: lower-wick spike rejection -> BUY
    """
    out: List[dict] = []
    if not best_fut:
        return out

    for base, mv in (best_fut or {}).items():
        try:
            sym_base = str(base).upper()
            market_symbol = str(getattr(mv, "symbol", "") or "")
            vol24 = float(getattr(mv, "fut_vol_usd", 0.0) or 0.0)
            if vol24 < float(min_vol_usd):
                continue
            if not market_symbol:
                continue

            # 1H candles for spike detection + ATR + trend
            c1 = fetch_ohlcv(market_symbol, "1h", limit=max(ATR_PERIOD + 10, 220))
            if not c1 or len(c1) < (ATR_PERIOD + 50):
                continue

            atr_1h = compute_atr_from_ohlcv(c1, ATR_PERIOD)
            if atr_1h <= 0:
                continue

            closes_1h = [float(x[4]) for x in c1]
            ema50 = float(ema(closes_1h[-200:], 50) or 0.0)
            ema200 = float(ema(closes_1h[-220:], 200) or 0.0)
            if ema50 <= 0 or ema200 <= 0:
                continue

            uptrend = (ema50 > ema200)
            downtrend = (ema50 < ema200)
            if not (uptrend or downtrend):
                continue

            # Spike candle = most recent closed 1H candle
            o = float(c1[-1][1]); h = float(c1[-1][2]); l = float(c1[-1][3]); c = float(c1[-1][4])
            rng = max(1e-12, h - l)

            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            upper_ratio = upper_wick / rng
            lower_ratio = lower_wick / rng

            big_range = (rng >= (atr_1h * float(atr_mult_min)))

            
            # 15m confirmation (stronger): last 15m close must confirm rejection
            # SELL: close below spike mid (and preferably below spike open)
            # BUY : close above spike mid (and preferably above spike open)
            c15 = fetch_ohlcv(market_symbol, "15m", limit=60)
            if not c15 or len(c15) < 12:
                continue
            c15_last_close = float(c15[-1][4])

            spike_mid = (h + l) / 2.0
            sell_confirm = (c15_last_close < spike_mid)
            buy_confirm  = (c15_last_close > spike_mid)

            # Downtrend: spike UP into resistance -> SELL reversal
            if downtrend and big_range and upper_ratio >= float(wick_ratio_min) and c < o:
                weak_confirm = (not sell_confirm)
                # allow weak confirmations with lower confidence

                entry = float(l)  # break of spike candle low
                sl = float(h) + (0.10 * atr_1h)
                r = max(1e-12, sl - entry)

                tp1 = entry - 1.0 * r
                tp2 = entry - 1.6 * r
                tp3 = entry - 2.3 * r

                conf = 82
                if weak_confirm:
                    conf -= 6
                if (c - l) / rng < 0.40:
                    conf += 4

                out.append({
                    "symbol": sym_base,
                    "market_symbol": market_symbol,
                    "side": "SELL",
                    "direction": "DOWN",
                    "conf": int(clamp(conf, 1, 99)),
                    "entry": float(entry),
                    "sl": float(sl),
                    "tp1": float(tp1),
                    "tp2": float(tp2),
                    "tp3": float(tp3),
                    "vol": float(vol24),
                    "why": f"Downtrend (EMA50<EMA200). 1H upper-wick spike rejection (wick={upper_ratio:.2f}, range={rng/atr_1h:.2f} ATR). " + ("15m confirm: WEAK" if weak_confirm else "15m confirm: OK"),
                })
                continue

            # Uptrend: spike DOWN into support -> BUY reversal
            if uptrend and big_range and lower_ratio >= float(wick_ratio_min) and c > o:
                weak_confirm = (not buy_confirm)
                # allow weak confirmations with lower confidence

                entry = float(h)  # break of spike candle high
                sl = float(l) - (0.10 * atr_1h)
                r = max(1e-12, entry - sl)

                tp1 = entry + 1.0 * r
                tp2 = entry + 1.6 * r
                tp3 = entry + 2.3 * r

                conf = 82
                if weak_confirm:
                    conf -= 6
                if (h - c) / rng < 0.40:
                    conf += 4

                out.append({
                    "symbol": sym_base,
                    "market_symbol": market_symbol,
                    "side": "BUY",
                    "direction": "UP",
                    "conf": int(clamp(conf, 1, 99)),
                    "entry": float(entry),
                    "sl": float(sl),
                    "tp1": float(tp1),
                    "tp2": float(tp2),
                    "tp3": float(tp3),
                    "vol": float(vol24),
                    "why": f"Uptrend (EMA50>EMA200). 1H lower-wick spike rejection (wick={lower_ratio:.2f}, range={rng/atr_1h:.2f} ATR). " + ("15m confirm: WEAK" if weak_confirm else "15m confirm: OK"),
                })
                continue

        except Exception:
            continue

    out = sorted(out, key=lambda x: (int(x.get("conf", 0)), float(x.get("vol", 0.0))), reverse=True)
    return out[:max_items]



def _spike_reversal_warnings(
    best_fut: Dict[str, Any],
    min_vol_usd: float = 15_000_000.0,
    atr_mult_min: float = 1.05,
    body_ratio_min: float = 0.50,
    lookback_1h: int = 8,
    retrace_min: float = 0.22,
    max_items: int = 8,
) -> List[dict]:
    """
    EARLY WARNING (non-trade):
    Flags possible reversal zones after a recent 1H impulse (spike) when price starts
    retracing and momentum cools, but BEFORE strict wick-rejection + 15m-confirm rules.

    Output dict: {symbol, side, conf, vol, why}
    """
    out: List[dict] = []
    if not best_fut:
        return out

    for base, mv in (best_fut or {}).items():
        try:
            sym_base = str(base).upper()
            market_symbol = str(getattr(mv, "symbol", "") or "")
            vol24 = float(getattr(mv, "fut_vol_usd", 0.0) or 0.0)
            if vol24 < float(min_vol_usd):
                continue
            if not market_symbol:
                continue

            c1 = fetch_ohlcv(market_symbol, "1h", limit=max(ATR_PERIOD + 10, 240))
            if not c1 or len(c1) < (ATR_PERIOD + 60):
                continue

            atr_1h = compute_atr_from_ohlcv(c1, ATR_PERIOD)
            if atr_1h <= 0:
                continue

            closes_1h = [float(x[4]) for x in c1]
            ema50 = float(ema(closes_1h[-200:], 50) or 0.0)
            ema200 = float(ema(closes_1h[-220:], 200) or 0.0)
            if ema50 <= 0 or ema200 <= 0:
                continue

            uptrend = (ema50 > ema200)
            downtrend = (ema50 < ema200)

            # find most recent impulse candle in lookback window
            imp = None
            for i in range(1, int(lookback_1h) + 1):
                o = float(c1[-i][1]); h = float(c1[-i][2]); l = float(c1[-i][3]); c = float(c1[-i][4])
                rng = max(1e-12, h - l)
                body = abs(c - o)
                if rng >= (atr_1h * float(atr_mult_min)) and (body / rng) >= float(body_ratio_min):
                    imp = (i, o, h, l, c, rng, body)
                    break
            if not imp:
                continue

            i, oI, hI, lI, cI, rngI, bodyI = imp
            impulse_up = (cI > oI)
            impulse_dn = (cI < oI)
            if not (impulse_up or impulse_dn):
                continue

            c0 = float(c1[-1][4])

            # momentum cooling on 1H closes (soft)
            cooling_dn = (float(c1[-1][4]) <= float(c1[-2][4]) <= float(c1[-3][4])) if len(c1) >= 3 else True
            cooling_up = (float(c1[-1][4]) >= float(c1[-2][4]) >= float(c1[-3][4])) if len(c1) >= 3 else True

            if impulse_up:
                retrace = (hI - c0) / max(1e-12, rngI)
                if retrace < float(retrace_min) or not cooling_dn:
                    continue
                trend_note = "Downtrend context. " if downtrend else ("Uptrend context. " if uptrend else "")
                conf = 72 + int(clamp(retrace * 20.0, 0, 15))
                out.append({
                    "symbol": sym_base,
                    "market_symbol": market_symbol,
                    "side": "SELL",
                    "conf": int(clamp(conf, 1, 99)),
                    "vol": float(vol24),
                    "why": f"{trend_note}Recent 1H impulse UP (range={rngI/atr_1h:.2f} ATR, body={bodyI/rngI:.2f}). Retrace={retrace:.2f}; cooling closes.",
                })
                continue

            if impulse_dn:
                retrace = (c0 - lI) / max(1e-12, rngI)
                if retrace < float(retrace_min) or not cooling_up:
                    continue
                trend_note = "Uptrend context. " if uptrend else ("Downtrend context. " if downtrend else "")
                conf = 72 + int(clamp(retrace * 20.0, 0, 15))
                out.append({
                    "symbol": sym_base,
                    "market_symbol": market_symbol,
                    "side": "BUY",
                    "conf": int(clamp(conf, 1, 99)),
                    "vol": float(vol24),
                    "why": f"{trend_note}Recent 1H impulse DOWN (range={rngI/atr_1h:.2f} ATR, body={bodyI/rngI:.2f}). Retrace={retrace:.2f}; cooling closes.",
                })
                continue

        except Exception:
            continue

    out = sorted(out, key=lambda x: (int(x.get("conf", 0)), float(x.get("vol", 0.0))), reverse=True)
    return out[:max_items]

def symbol_flip_guard_active(
    user_id: int,
    symbol: str,
    side: str,
    session_name: str,
) -> bool:
    """
    Blocks opposite-direction alerts for the same symbol for a period.
    Uses (session cooldown hours * FLIP_GUARD_MULT).
    """
    if not FLIP_GUARD_ENABLED:
        return False

    cooldown_hours = float(cooldown_hours_for_session(session_name)) * float(FLIP_GUARD_MULT)
    opposite = "SELL" if str(side).upper() == "BUY" else "BUY"

    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        SELECT emailed_ts
        FROM emailed_symbols
        WHERE user_id=? AND symbol=? AND side=?
    """, (int(user_id), str(symbol).upper(), opposite))
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

    if not has_active_access(user, uid):
        await update.message.reply_text(
            "⛔️ Trial finished.\n\n"
            "Your 7-day trial is over — you need to pay to keep using PulseFutures.\n\n"
            "👉 /billing"
        )
        return

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
    Market sessions in UTC.

    NOTE:
    - We use non-overlapping UTC windows to avoid ambiguous session labels around overlaps.
    - This also fixes Melbourne display so ~6:00 PM Melbourne remains ASIA until 7:00 PM (AEDT).

    Windows:
    - NY  : 13:00–22:00 UTC
    - LON : 08:00–17:00 UTC
    - ASIA: 00:00–08:00 UTC
    Priority: NY > LON > ASIA
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    h = now_utc.hour

    if 13 <= h < 22:
        return "NY"
    if 8 <= h < 17:
        return "LON"
    if 0 <= h < 8:
        return "ASIA"

    # Outside the three main windows: treat as NY tail/transition
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
    Adaptive EMA proximity + structure check.
    Returns: (ok, dist_pct, thr_pct, atr_pct)

    Rules:
    - Must be near EMA (ATR-adaptive)
    - Price must be on the correct side of EMA
      BUY  → entry >= EMA
      SELL → entry <= EMA
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

    # Near enough?
    near_ok = dist_pct <= thr_pct

    # Structure: entry must be on the correct side
    # (side check is done outside, so infer by relative position)
    structure_ok = (
        (entry >= ema_val) or  # BUY-side structure
        (entry <= ema_val)     # SELL-side structure
    )

    ok = bool(near_ok and structure_ok)
    return ok, dist_pct, thr_pct, atr_pct


def ema_support_reaction_ok_15m(c15: List[List[float]], ema_val: float, side: str, session_name: str) -> bool:
    """
    Strict EMA reaction confirmation (15m):

    BUY:
      - Candle touches/breaks EMA
      - Closes back ABOVE EMA
      - Bullish candle
      - Close in top part of range
      - Rejects EMA (not crossing through)

    SELL: mirrored logic
    """
    if not c15 or len(c15) < 10 or ema_val <= 0:
        return False

    lookback_n = int(session_knobs(session_name)["ema_reaction_lookback"])
    lookback_n = int(clamp(lookback_n, 4, 12))
    lookback = c15[-lookback_n:]

    reject_count = 0
    cross_count = 0

    for c in lookback:
        o = float(c[1])
        h = float(c[2])
        l = float(c[3])
        cl = float(c[4])

        rng = max(1e-9, h - l)
        close_pos = (cl - l) / rng  # 0..1

        touched = (l <= ema_val <= h)

        crossed = ((o - ema_val) * (cl - ema_val)) < 0
        if crossed:
            cross_count += 1

        if not touched:
            continue

        if side == "BUY":
            if cl > ema_val and cl > o and close_pos >= 0.65:
                reject_count += 1
        else:
            if cl < ema_val and cl < o and close_pos <= 0.35:
                reject_count += 1

    # ❌ EMA chop → invalid
    if cross_count >= 2:
        return False

    # ✅ Need at least ONE strong rejection
    return reject_count >= 1


# =========================================================
# Optional pullback policy (do NOT hard-reject signals)
# =========================================================

PULLBACK_OPTIONAL_DEFAULT = True
PULLBACK_MISS_PENALTY_CONF = 6.0

def apply_pullback_policy(
    *,
    require_pullback: bool,
    pullback_ok: bool,
    pullback_price: float | None,
    confidence: float,
    notes: list[str],
):
    if pullback_ok:
        return True, confidence

    if require_pullback:
        return False, confidence

    confidence = max(0.0, confidence - float(PULLBACK_MISS_PENALTY_CONF))

    if pullback_price and pullback_price > 0:
        notes.append(f"📝 Optional: wait for pullback entry near {pullback_price:.6g}.")
    else:
        notes.append("📝 Optional: consider waiting for a pullback before entering.")

    return True, confidence


def ema_rejection_candle_ok_15m(c15: List[List[float]], ema_val: float, side: str) -> bool:
    """
    Strict confirmation on the MOST RECENT 15m candle:

    BUY (support reclaim):
      - low <= EMA
      - close > EMA
      - bullish candle
      - close in top part of range (strength)

    SELL (resistance reject):
      - high >= EMA
      - close < EMA
      - bearish candle
      - close in bottom part of range (strength)
    """
    if not c15 or len(c15) < 2 or ema_val <= 0:
        return False

    o = float(c15[-1][1])
    h = float(c15[-1][2])
    l = float(c15[-1][3])
    c = float(c15[-1][4])

    rng = max(1e-9, (h - l))
    close_pos = (c - l) / rng  # 0..1

    if side == "BUY":
        if l <= ema_val and c > ema_val and c > o and close_pos >= float(REJECTION_CLOSE_POS_MIN):
            return True
        return False

    # SELL
    if h >= ema_val and c < ema_val and c < o and close_pos <= float(REJECTION_CLOSE_POS_MAX):
        return True
    return False


def strong_reversal_exception_ok(side: str, ch24: float, ch4: float, ch1: float) -> bool:
    """
    Rare override: only allow counter-trend if the move is strong across TFs.
    """
    if not ALLOW_STRONG_REVERSAL_EXCEPTION:
        return False
    try:
        return (abs(float(ch24)) >= float(REVERSAL_CH24_ABS_MIN)
                and abs(float(ch4)) >= float(REVERSAL_CH4_ABS_MIN)
                and abs(float(ch1)) >= float(REVERSAL_CH1_ABS_MIN))
    except Exception:
        return False


def metrics_from_candles_1h_15m(market_symbol: str) -> Tuple[float, float, float, float, float, int, List[List[float]], List[List[float]]]:
    """
    returns: ch1, ch4, ch15, atr_1h, ema_support_15m, ema_support_period, c15, c1
    """
    need_1h = max(ATR_PERIOD + 6, 35)
    c1 = fetch_ohlcv(market_symbol, "1h", limit=need_1h)
    if not c1 or len(c1) < 25:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, [], []

    closes_1h = [float(x[4]) for x in c1]
    c_last = closes_1h[-1]
    c_prev1 = closes_1h[-2]
    c_prev4 = closes_1h[-5] if len(closes_1h) >= 5 else closes_1h[0]

    ch1 = ((c_last - c_prev1) / c_prev1) * 100.0 if c_prev1 else 0.0
    ch4 = ((c_last - c_prev4) / c_prev4) * 100.0 if c_prev4 else 0.0
    atr_1h = compute_atr_from_ohlcv(c1, ATR_PERIOD)

    c15 = fetch_ohlcv(market_symbol, "15m", limit=80)
    if not c15 or len(c15) < 20:
        return ch1, ch4, 0.0, atr_1h, 0.0, 0, [], c1

    closes_15 = [float(x[4]) for x in c15]
    entry_proxy = float(closes_15[-1]) if closes_15 else 0.0
    atr_pct_proxy = (atr_1h / entry_proxy) * 100.0 if (atr_1h and entry_proxy) else 2.0

    ema_support_15m, ema_period = adaptive_ema_value(closes_15, atr_pct_proxy)

    c15_last = float(c15[-1][4])
    c15_prev = float(c15[-2][4])
    ch15 = ((c15_last - c15_prev) / c15_prev) * 100.0 if c15_prev else 0.0

    return ch1, ch4, ch15, atr_1h, ema_support_15m, int(ema_period), c15, c1


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

def _fmt_when(ts) -> str:
    """Best-effort timestamp formatter for decision debug commands."""
    try:
        if ts is None:
            return ""
        # If already an ISO-ish string, return as-is
        if isinstance(ts, str):
            s = ts.strip()
            if s:
                return s
            return ""
        # Unix timestamp (sec)
        import datetime as _dt
        return _dt.datetime.fromtimestamp(float(ts), tz=_dt.timezone.utc).isoformat(timespec="seconds")
    except Exception:
        try:
            return str(ts)
        except Exception:
            return ""



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

SAFE_CHUNK = 3800

async def send_long_message(
    update: Update,
    text: str,
    parse_mode: Optional[str] = None,
    disable_web_page_preview: bool = True,
    reply_markup=None,
):
    """
    Sends long messages safely by chunking and retrying.

    Notes:
    - Telegram max message length is ~4096 chars. We stay below that.
    - If parse_mode causes BadRequest (HTML/Markdown formatting), we retry as plain text.
    - Retries on RetryAfter and transient network errors to avoid losing chunks.
    """
    if not update or not getattr(update, "message", None):
        return

    s = text or ""

    # Be safe if SAFE_CHUNK is missing or too large
    max_len = 3800
    try:
        max_len = int(globals().get("SAFE_CHUNK", 3800))
    except Exception:
        max_len = 3800
    if max_len > 3900:
        max_len = 3900
    if max_len < 500:
        max_len = 2000

    chunks = [s[i : i + max_len] for i in range(0, len(s), max_len)]

    first = True
    for ch in chunks:

        async def _send(pm: Optional[str]):
            await update.message.reply_text(
                ch,
                parse_mode=pm,
                disable_web_page_preview=disable_web_page_preview,
                reply_markup=reply_markup if first else None,
            )

        # Try with requested parse_mode first (or None)
        try:
            await _send(parse_mode)

        except RetryAfter as e:
            # Telegram rate limit: wait then retry once
            wait_s = int(getattr(e, "retry_after", 1)) + 1
            await asyncio.sleep(wait_s)
            await _send(parse_mode)

        except (TimedOut, NetworkError) as e:
            # transient network: retry once
            logger.warning("Telegram send failed (network): %s", e)
            await asyncio.sleep(2)
            try:
                await _send(parse_mode)
            except Exception as e2:
                logger.warning("Telegram retry failed (network): %s", e2)

        except BadRequest as e:
            # Usually formatting issue -> fallback to plain text
            logger.warning("Telegram BadRequest (parse_mode=%s): %s | Falling back to plain text.", parse_mode, e)
            try:
                await _send(None)
            except Exception as e2:
                logger.warning("Telegram fallback send failed: %s", e2)

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
    """
    Stricter confidence designed for:
    - Fewer, higher quality continuation signals
    - Avoiding mid-wave / pump entries
    - Preferring "pullback then continuation" (15m should be mild or slightly counter-trend)
    """
    is_long = (side == "BUY")

    # Base score starts lower to reduce inflated confidences
    score = 42.0

    # -----------------------------
    # 1) Higher-TF alignment first
    # -----------------------------
    def aligned(x: float) -> bool:
        return (x > 0) if is_long else (x < 0)

    a24 = aligned(ch24)
    a4  = aligned(ch4)
    a1  = aligned(ch1)
    a15 = aligned(ch15)

    align_count = int(a24) + int(a4) + int(a1) + int(a15)

    # Stronger reward for 4H/1H alignment (regime + execution TF)
    score += 14.0 if a4 else -18.0
    score += 12.0 if a1 else -16.0

    # 24H matters, but less than 4H/1H for trade execution quality
    score += 8.0 if a24 else -10.0

    # 15m alignment is NOT always good for entries (pumps are bad entries)
    # We'll handle 15m separately as "pullback quality"
    # (so do not add big alignment reward here)
    score += 2.0 if a15 else -2.0

    # Hard cap if the structure is weak:
    # If 4H+1H disagree with your side, confidence should never be high.
    if (not a4) and (not a1):
        return int(55)  # keep it low; you can also return 0 if you want total block

    # If at least 3 TFs align (24/4/1/15), small boost
    if align_count >= 3:
        score += 6.0
    elif align_count <= 1:
        score -= 10.0

    # -----------------------------
    # 2) Magnitude scoring (capped)
    # -----------------------------
    def signed_mag(x: float, k: float) -> float:
        return (abs(x) * k) if aligned(x) else (-abs(x) * k)

    mag = (
        signed_mag(ch24, 0.5) +
        signed_mag(ch4,  0.9) +
        signed_mag(ch1,  1.2) +
        signed_mag(ch15, 0.6)
    )
    mag = clamp(mag, -18.0, 18.0)
    score += mag

    # -----------------------------
    # 3) Anti-pump + pullback preference (15m behavior)
    # -----------------------------
    # For continuation entries:
    # - We *don't* want 15m strongly in-trend (often means mid-wave / extended).
    # - We *do* like mild pullback against the trend (e.g. -0.3% to -1.2% in BUY).
    ch15_abs = abs(float(ch15))

    if is_long:
        # Mid-wave pump penalty: 15m strongly positive
        if ch15 >= 0.8:
            score -= 12.0
        elif ch15 >= 0.4:
            score -= 6.0

        # Preferred pullback zone (slight red 15m)
        if -1.2 <= ch15 <= -0.25:
            score += 10.0
        elif -2.2 <= ch15 < -1.2:
            score += 4.0

        # Too much dump into support = unstable
        if ch15 <= -2.8:
            score -= 10.0
    else:
        # Mid-wave dump penalty: 15m strongly negative
        if ch15 <= -0.8:
            score -= 12.0
        elif ch15 <= -0.4:
            score -= 6.0

        # Preferred pullback zone (slight green 15m)
        if 0.25 <= ch15 <= 1.2:
            score += 10.0
        elif 1.2 < ch15 <= 2.2:
            score += 4.0

        # Too much pump into resistance = unstable
        if ch15 >= 2.8:
            score -= 10.0

    # -----------------------------
    # 4) Volume bonus (smaller + only when structure is already good)
    # -----------------------------
    # Volume should NOT rescue a bad setup.
    # Only add if score already indicates decent structure.
    if score >= 60.0:
        if fut_vol_usd >= 40_000_000:
            score += 5.0
        elif fut_vol_usd >= 15_000_000:
            score += 3.0
        elif fut_vol_usd >= 6_000_000:
            score += 1.5

    score = clamp(score, 0.0, 100.0)
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
# make_setup
# =========================================================
def make_setup(
    base: str,
    mv: MarketVol,
    strict_15m: bool = True,
    session_name: str = "LON",
    allow_no_pullback: bool = True,     # /screen True, Email False (except HOT bypass)
    hot_vol_usd: float = HOT_VOL_USD,   # 50M
    trigger_loosen_mult: float = 1.0,
    waiting_near_pct: float = SCREEN_WAITING_NEAR_PCT,
    scan_profile: str = DEFAULT_SCAN_PROFILE,
) -> Optional[Setup]:

    fut_vol = usd_notional(mv)
    if fut_vol <= 0:
        _rej("no_futures_volume", base, mv)
        return None

    entry = float(mv.last or 0.0)
    if entry <= 0:
        _rej("bad_entry", base, mv)
        return None

    ch24 = float(mv.percentage or 0.0)

    _note_status("evaluated", base, mv)


    notes = []  # collect internal notes for pullback policy / diagnostics
    try:

        ch1, ch4, ch15, atr_1h, ema_support_15m, ema_period, c15, c1 = metrics_from_candles_1h_15m(mv.symbol)

    except Exception:

        _rej("ohlcv_missing_or_insufficient", base, mv)

        return None
    # Use true 4H change from 4H candles (more stable than 1H*4 approximation)
    try:
        ch4_exact = 0.0
        try:
            c4h = fetch_ohlcv(mv.symbol, "4h", limit=6)
            if c4h and len(c4h) >= 2:
                c_last_4h = float(c4h[-1][4])
                c_prev_4h = float(c4h[-2][4])
                ch4_exact = ((c_last_4h - c_prev_4h) / c_prev_4h) * 100.0 if c_prev_4h else 0.0
        except Exception:
            ch4_exact = 0.0

        ch4_used = ch4_exact if abs(ch4_exact) > 0.0001 else ch4

        if (ch1 == 0.0 and ch4 == 0.0 and ch15 == 0.0 and atr_1h == 0.0) or (not c15) or (ema_support_15m == 0.0):
            _rej("ohlcv_missing_or_insufficient", base, mv, "metrics/ema missing")
            return None

        # Scan profile tuning
        prof = str(scan_profile or DEFAULT_SCAN_PROFILE).strip().lower()
        if prof not in SCAN_PROFILES:
            prof = DEFAULT_SCAN_PROFILE
        aggressive_screen = (prof == "aggressive" and (not strict_15m))

        # --------- SESSION-DYNAMIC 1H TRIGGER ----------
        atr_pct_now = (atr_1h / entry) * 100.0 if (atr_1h and entry) else 0.0
        trig_min_raw = trigger_1h_abs_min_atr_adaptive(atr_pct_now, session_name)

        floor_min = 0.025 if aggressive_screen else 0.05  # further loosened: allow setups in quiet 1H candles
        trig_min = max(float(floor_min), float(trig_min_raw) * float(trigger_loosen_mult))

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
                        dot = "🟠"

                    side_guess = ("BUY" if ch4 >= 0 else "SELL") if abs(ch4) >= 0.25 else ("BUY" if ch1 > 0 else "SELL")
                    _WAITING_TRIGGER[str(base)] = {"side": side_guess, "dot": dot}

            # Balanced breakout override: even if 1H change is small, allow true breakouts
            # Additional overrides: allow setups during quiet 1H candles if 4H/24H move is strong and 15m confirms.
            override_4h = (abs(float(ch4_used or 0.0)) >= max(0.9, float(trig_min) * 1.6)) and (abs(float(ch15 or 0.0)) >= 0.10)
            override_24h = (abs(float(ch24 or 0.0)) >= 10.0) and (abs(float(ch15 or 0.0)) >= 0.08)
            # Extra override: if 24H is very strong AND 4H aligns, don't require 15m confirmation (quiet consolidation after a big move)
            override_24h_strong = (abs(float(ch24 or 0.0)) >= 18.0) and (abs(float(ch4_used or 0.0)) >= 0.60) and (float(fut_vol or 0.0) >= 5_000_000.0)
            breakout_override = False
            # Trend override: if 4H regime move is meaningful, allow even if this 1H is quiet.
            try:
                if abs(float(ch4_used or 0.0)) >= max(0.75, float(trig_min) * 3.0) and float(fut_vol or 0.0) >= 5_000_000.0:
                    breakout_override = True
            except Exception:
                pass
            try:
                if c1 and len(c1) >= 25 and abs(ch24) >= 6.0:
                    highs_1h = [float(x[2]) for x in c1]
                    lows_1h  = [float(x[3]) for x in c1]
                    closes_1h = [float(x[4]) for x in c1]
                    vols_1h  = [float(x[5]) for x in c1]
                    last_close = float(closes_1h[-1])
                    last_high  = float(highs_1h[-1])
                    last_low   = float(lows_1h[-1])
                    hh20 = max(highs_1h[-21:-1])
                    ll20 = min(lows_1h[-21:-1])
                    vnow = float(vols_1h[-1])
                    vavg = (sum(vols_1h[-21:-1]) / 20.0) if vols_1h[-21:-1] else 0.0

                    vol_mult = 1.08  # balanced default
                    if (ch24 >= 6.0) and (last_high > hh20 or last_close > hh20) and ((vavg > 0 and vnow >= vavg * vol_mult) or (vavg <= 0 and vnow > 0)):
                        breakout_override = True
                    if (ch24 <= -6.0) and (last_low < ll20 or last_close < ll20) and ((vavg > 0 and vnow >= vavg * vol_mult) or (vavg <= 0 and vnow > 0)):
                        breakout_override = True
            except Exception:
                breakout_override = False

            # Soft override: if we are close to the trigger and the higher timeframe move supports direction,
            # allow a lower-confidence setup instead of rejecting everything on a quiet 1H candle.
            try:
                ratio = (abs(float(ch1)) / float(trig_min)) if float(trig_min) > 0 else 0.0
            except Exception:
                ratio = 0.0
            soft_override = (ratio >= 0.78) and (float(fut_vol or 0.0) >= 5_000_000.0) and (
                abs(float(ch4_used or 0.0)) >= 0.60 or abs(float(ch24 or 0.0)) >= 14.0
            )

            if not (breakout_override or override_4h or override_24h or override_24h_strong or soft_override):
                _rej("ch1_below_trigger", base, mv, f"ch1={ch1:+.2f}% trig={trig_min:.2f}% ch4={ch4:+.2f}% ch24={ch24:+.2f}% ch15={ch15:+.2f}%")
                return None


        # ✅ IMPORTANT: this must be OUTSIDE the if block (no extra indent)
        side = ("BUY" if ch4 >= 0 else "SELL") if abs(ch4) >= 0.40 else ("BUY" if ch1 > 0 else "SELL")  # trend-side gating: prefer 4H regime over 1H noise

        # 4H alignment
        # If 4H is basically flat, don't block (leaders often show ch4 ~ 0 while 24H is huge).
        if abs(ch4) >= float(ALIGN_4H_NEUTRAL_ZONE):
            if side == "BUY" and ch4 < ALIGN_4H_MIN:
                _rej("4h_not_aligned_for_long", base, mv, f"side=BUY ch4={ch4:+.2f}%")
                return None
            if side == "SELL" and ch4 > -ALIGN_4H_MIN:
                _rej("4h_not_aligned_for_short", base, mv, f"side=SELL ch4={ch4:+.2f}%")
                return None

        # ✅ HARD regime gate: don't fight the 4H direction (unless "strong reversal exception")
        if TF_ALIGN_ENABLED and abs(ch4) >= float(ALIGN_4H_NEUTRAL_ZONE):
            if side == "BUY" and ch4 < 0:
                if not strong_reversal_exception_ok(side, ch24, ch4, ch1):
                    _rej("4h_bear_regime_blocks_long", base, mv, f"ch4={ch4:+.2f}%")
                    return None
            if side == "SELL" and ch4 > 0:
                if not strong_reversal_exception_ok(side, ch24, ch4, ch1):
                    _rej("4h_bull_regime_blocks_short", base, mv, f"ch4={ch4:+.2f}%")
                    return None

        # =========================================================
        # ✅ PULLBACK EMA (7/14/21) selection (15m)
        # =========================================================
        closes_15 = [float(x[4]) for x in c15]
        pb_ema_val, pb_ema_p, pb_dist_pct = best_pullback_ema_15m(closes_15, c15, entry, side, session_name, atr_1h)

        # ✅ No HOT bypass: pullback is ALWAYS required for high-quality continuation entries
        pullback_bypass_hot = False

        pb_ok = False
        pb_thr_pct = 0.0
        pb_dist_pct2 = 999.0

        if pb_ema_val > 0:
            pb_ok, pb_dist_pct2, pb_thr_pct, _ = ema_support_proximity_ok(entry, pb_ema_val, atr_1h, session_name)

        pullback_ready = bool(pb_ok)

        # Aggressive /screen: allow "near-EMA" pullback (slightly looser proximity)
        if (not pullback_ready) and aggressive_screen and (pb_ema_val > 0) and (pb_thr_pct > 0):
            try:
                if float(pb_dist_pct2) <= float(pb_thr_pct) * 1.35:
                    pullback_ready = True
            except Exception:
                pass

        # 15m rejection candle requirement (skip in Aggressive /screen)
        require_rejection = bool(REQUIRE_15M_EMA_REJECTION and (not aggressive_screen))

        # ✅ Strict continuation entry: must show EMA interaction + strong 15m rejection/reclaim
        if pullback_ready and require_rejection:
            if (not c15) or (pb_ema_val <= 0):
                _rej("no_ema_touch_reclaim_recent", base, mv,
                     f"ema{pb_ema_p} pb_dist={pb_dist_pct2:.2f}% thr={pb_thr_pct:.2f}%")
                return None

            if not ema_rejection_candle_ok_15m(c15, pb_ema_val, side):
                _rej("no_strong_rejection_candle_15m", base, mv, f"ema{pb_ema_p} pb_dist={pb_dist_pct2:.2f}%")
                return None

        # =========================================================
        # ENGINE A (Mean-Reversion) vs ENGINE B (Momentum)
        # =========================================================
        engine_a_ok = bool(ENGINE_A_PULLBACK_ENABLED and pullback_ready)

        engine_b_ok = False

        if ENGINE_B_MOMENTUM_ENABLED:
            # Aggressive /screen: slightly looser momentum requirements
            mom_min_ch1 = float(MOMENTUM_MIN_CH1) * (0.75 if aggressive_screen else 1.0)
            mom_min_24h = float(MOMENTUM_MIN_24H) * (0.75 if aggressive_screen else 1.0)
            mom_body_mult = float(MOMENTUM_ATR_BODY_MULT) * (0.85 if aggressive_screen else 1.0)
            mom_max_ema_dist = float(MOMENTUM_MAX_ADAPTIVE_EMA_DIST) * ((1.60 if (not strict_15m) else 1.0)) * (1.30 if aggressive_screen else 1.0)

            # ------------------------------------------------------------------
            # B1) Momentum continuation (existing)
            # ------------------------------------------------------------------
            if abs(ch1) >= mom_min_ch1 and abs(ch24) >= mom_min_24h:
                if fut_vol >= (MOVER_VOL_USD_MIN * MOMENTUM_VOL_MULT):
                    body_pct = abs(ch1)
                    if atr_pct_now > 0 and body_pct >= (mom_body_mult * atr_pct_now):
                        _, dist_pct, _, _ = ema_support_proximity_ok(entry, ema_support_15m, atr_1h, session_name)
                        if dist_pct <= mom_max_ema_dist:
                            engine_b_ok = True

            # ------------------------------------------------------------------
            # B2) Balanced Breakout continuation (NEW)
            # Purpose: avoid "leaders full, setups empty" during expansion phases.
            # Uses the already-fetched 1H candles (c1) to avoid extra API calls / rate limits.
            # ------------------------------------------------------------------
            try:
                ch24_thr = 4.0 if aggressive_screen else 5.0
                vol_mult = 1.05 if aggressive_screen else 1.08

                # Use 4H/side gating already computed as the trend regime.
                uptrend = (ch4_used >= 0)
                downtrend = (ch4 < 0)

                if abs(ch24) >= ch24_thr and fut_vol >= max(5_000_000.0, float(MOVER_VOL_USD_MIN) * 0.70) and c1 and len(c1) >= 25:
                    highs_1h = [float(x[2]) for x in c1]
                    lows_1h  = [float(x[3]) for x in c1]
                    closes_1h = [float(x[4]) for x in c1]
                    vols_1h  = [float(x[5]) for x in c1]

                    last_close = float(closes_1h[-1])
                    last_high = float(highs_1h[-1])
                    last_low  = float(lows_1h[-1])
                    vnow = float(vols_1h[-1])
                    vavg = (sum(vols_1h[-21:-1]) / 20.0) if vols_1h[-21:-1] else 0.0

                    # Prior 20-candle extremes (exclude the current candle)
                    hh20 = max(highs_1h[-21:-1])
                    ll20 = min(lows_1h[-21:-1])

                    # Breakout BUY
                    if uptrend and ch24 >= ch24_thr and (last_high > hh20 or last_close > hh20) and ((vavg > 0 and vnow >= vavg * vol_mult) or (vavg <= 0 and vnow > 0)):
                        engine_b_ok = True
                        mv._pf_breakout_hint = "BUY"

                    # Breakdown SELL
                    if downtrend and ch24 <= -ch24_thr and (last_low < ll20 or last_close < ll20) and ((vavg > 0 and vnow >= vavg * vol_mult) or (vavg <= 0 and vnow > 0)):
                        engine_b_ok = True
                        mv._pf_breakout_hint = "SELL"
            except Exception:
                pass
    
        # If breakout engine fired, force side to match the breakout direction (prevents green BUY / red mismatch)
        try:
            hint = getattr(mv, "_pf_breakout_hint", None)
            if engine_b_ok and hint in ("BUY", "SELL"):
                side = hint
        except Exception:
            pass

        # If breakout engine fired, force side to match the breakout direction (prevents green BUY / red mismatch)
        try:
            hint = getattr(mv, "_pf_breakout_hint", None)
            if engine_b_ok and hint in ("BUY", "SELL"):
                side = hint
        except Exception:
            pass

        if not engine_a_ok and not engine_b_ok:
            _rej("no_engine_passed", base, mv, f"ch1={ch1:.2f} ch24={ch24:.2f} pb_dist={pb_dist_pct:.2f}")
            return None

        # ---------------------------------------------------------
        # Pullback policy (optional for Engine B)
        # ---------------------------------------------------------
        engine = "A" if engine_a_ok else "B"
        require_pullback = (not bool(allow_no_pullback))

        # Compute confidence BEFORE applying optional pullback penalty
        conf = compute_confidence(side, ch24, ch4, ch1, ch15, fut_vol)

        # Engine A already required pullback_ready; Engine B does not require pullback
        pullback_ok_local = True if engine == "A" else True

        keep, conf2 = apply_pullback_policy(
            require_pullback=require_pullback,
            pullback_ok=pullback_ok_local,
            pullback_price=(pb_ema_val if pb_ema_val > 0 else None),
            confidence=float(conf),
            notes=notes,
        )

        conf = conf2

        if not keep:
            _rej("pullback_required_not_met", base, mv, "require_pullback=1")
            return None

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

        if strict_15m and (not is_confirm_15m):
            conf = max(0, int(conf) - int(EARLY_CONF_PENALTY))

        tp_cap_pct = tp_cap_pct_for_coin(fut_vol, ch24)

        rr_bonus = ENGINE_B_RR_BONUS if engine_b_ok else 0.0
        tp_cap_bonus = ENGINE_B_TP_CAP_BONUS_PCT if engine_b_ok else 0.0

        sl, tp3_single, R = compute_sl_tp(
            entry, side, atr_1h, conf, tp_cap_pct,
            rr_bonus=rr_bonus, tp_cap_bonus_pct=tp_cap_bonus
        )
        if sl <= 0 or tp3_single <= 0 or R <= 0:
            _rej("bad_sl_tp_or_atr", base, mv, f"atr={atr_1h:.6g} entry={entry:.6g}")
            return None

        tp1 = tp2 = None
        tp3 = tp3_single
        if conf >= MULTI_TP_MIN_CONF:
            _tp1, _tp2, _tp3 = multi_tp(
                entry, side, R, tp_cap_pct, conf,
                rr_bonus=rr_bonus, tp_cap_bonus_pct=tp_cap_bonus
            )
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
            pullback_ema_dist_pct=float(pb_dist_pct2),
            pullback_ready=bool(pullback_ready),
            pullback_bypass_hot=bool(pullback_bypass_hot),
            engine=str(engine),
            is_trailing_tp3=trailing_tp3,
            created_ts=time.time(),
        )
    except Exception as e:
        try:
            _rej("exception_in_make_setup", base, mv, f"{type(e).__name__}: {e}")
        except Exception:
            pass
        return None




def make_breakout_setup(
    base: str,
    mv: MarketVol,
    session_name: str = "LON",
    scan_profile: str = DEFAULT_SCAN_PROFILE,
) -> Optional[Setup]:
    """
    Engine B: Momentum Breakout/Breakdown setups.
    Uses already-fetched 1H candles from metrics_from_candles_1h_15m().
    """
    fut_vol = usd_notional(mv)
    if fut_vol <= 0:
        _rej("no_futures_volume", base, mv)
        return None

    entry = float(mv.last or 0.0)
    if entry <= 0:
        _rej("bad_entry", base, mv)
        return None

    ch24 = float(mv.percentage or 0.0)

    try:
        ch1, ch4, ch15, atr_1h, ema_support_15m, ema_period, c15, c1 = metrics_from_candles_1h_15m(mv.symbol)
    except Exception:
        _rej("ohlcv_missing_or_insufficient", base, mv)
        return None

    # Align naming with make_setup()
    ch4_used = float(ch4 or 0.0)

    if not c1 or len(c1) < 25 or atr_1h <= 0:
        _rej("no_candles_breakout", base, mv)
        return None

    # Prefer trend regime from 4H
    trend_up = ch4_used > 0
    trend_dn = ch4_used < 0

    highs = [float(x[2]) for x in c1 if x and len(x) >= 6][-25:]
    lows  = [float(x[3]) for x in c1 if x and len(x) >= 6][-25:]
    closes= [float(x[4]) for x in c1 if x and len(x) >= 6][-25:]
    vols  = [float(x[5]) for x in c1 if x and len(x) >= 6][-25:]

    if len(highs) < 22 or len(vols) < 22:
        _rej("insufficient_1h_window", base, mv)
        return None

    # prior window excludes the last candle to avoid self-referencing
    prior_high20 = max(highs[-21:-1])
    prior_low20  = min(lows[-21:-1])

    last_high = highs[-1]
    last_low  = lows[-1]
    last_close = closes[-1]

    vol_avg20 = sum(vols[-21:-1]) / 20.0
    vol_now = vols[-1]
    vol_ok = (vol_now > 0) and ((vol_avg20 <= 0) or (vol_now >= vol_avg20 * 1.00))  # looser

    # Balanced thresholds
    ch24_buy_min = 6.0
    ch24_sell_max = -6.0

    # Breakout/Breakdown by wick OR close
    is_breakout = (last_high > prior_high20) or (last_close > prior_high20)
    is_breakdown = (last_low < prior_low20) or (last_close < prior_low20)

    # If 4H is strongly up/down, allow direction to follow regime
    if trend_up and is_breakout and vol_ok and ch24 >= ch24_buy_min:
        side = "BUY"
    elif trend_dn and is_breakdown and vol_ok and ch24 <= ch24_sell_max:
        side = "SELL"
    else:
        # As a fallback, allow if 4H is neutral but 24H is strong
        if is_breakout and vol_ok and ch24 >= max(10.0, ch24_buy_min) and (ch4 >= 0):
            side = "BUY"
        elif is_breakdown and vol_ok and ch24 <= min(-10.0, ch24_sell_max) and (ch4_used <= 0):
            side = "SELL"
        else:
            # Balanced fallback: strong momentum continuation (not a fresh HH/LL break)
            if vol_ok:
                if (ch4_used >= 4.0 and ch24 >= 12.0):
                    side = "BUY"
                elif (ch4_used <= -4.0 and ch24 <= -12.0):
                    side = "SELL"
                else:
                    _rej("no_breakout_trigger", base, mv)
                    return None
            else:
                _rej("no_breakout_trigger", base, mv)
                return None


    # SL/TP using ATR (simple + robust)
    sl_atr = 1.35
    rr_target = 2.0  # TP3 RR target for momentum
    if side == "BUY":
        sl = entry - (atr_1h * sl_atr)
        tp3 = entry + (entry - sl) * rr_target
    else:
        sl = entry + (atr_1h * sl_atr)
        tp3 = entry - (sl - entry) * rr_target

    if sl <= 0 or tp3 <= 0:
        _rej("bad_sl_tp", base, mv)
        return None

    # Confidence scoring (balanced)
    conf = 80
    if abs(ch4_used) >= 5:
        conf += 2
    if abs(ch24) >= 20:
        conf += 2
    if vol_avg20 > 0 and vol_now >= vol_avg20 * 1.25:
        conf += 2
    conf = int(min(conf, 90))

    setup_id = make_setup_id(base, side)
    return Setup(
        setup_id=setup_id,
        symbol=base,
        market_symbol=mv.symbol,
        side=side,
        conf=conf,
        entry=float(entry),
        sl=float(sl),
        tp1=None,
        tp2=None,
        tp3=float(tp3),
        fut_vol_usd=float(fut_vol),
        ch24=float(ch24),
        ch4=float(ch4_used),
        ch1=float(ch1),
        ch15=float(ch15),
        ema_support_period=int(ema_period or 0),
        ema_support_dist_pct=float(0.0),
        pullback_ema_period=int(0),
        pullback_ema_dist_pct=float(0.0),
        pullback_ready=False,
        pullback_bypass_hot=False,
        engine="B",
        is_trailing_tp3=False,
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
    allow_no_pullback: bool = False,   # /screen True, email False (except HOT bypass inside make_setup)
    scan_profile: str = DEFAULT_SCAN_PROFILE,
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
        try:
            s = make_setup(
                base,
                mv,
                strict_15m=strict_15m,
                session_name=session_name,
                allow_no_pullback=allow_no_pullback,
                trigger_loosen_mult=float(trigger_loosen_mult),
                waiting_near_pct=float(waiting_near_pct),
                scan_profile=str(scan_profile or DEFAULT_SCAN_PROFILE),
            )
        except Exception:
            try:
                _rej("make_setup_exception", base, mv)
            except Exception:
                pass
            s = None

        if s:
            try:
                _note_status("setup_generated", base, mv)
            except Exception:
                pass
            setups.append(s)
        else:
            # If make_setup returns None without recording a reject, add a generic reject
            # so /why is never empty and tuning is possible.
            try:
                ctx = _REJECT_CTX.get()
                b = str(base or "").upper().strip()
                per = (ctx or {}).get("__per__") if isinstance(ctx, dict) else None
                if b and isinstance(ctx, dict) and (not isinstance(per, dict) or b not in per):
                    _rej("no_setup_candidate", base, mv, "make_setup_returned_none")
            except Exception:
                pass

    setups.sort(key=lambda x: (x.conf, x.fut_vol_usd), reverse=True)
    return setups[:n]


# =========================================================
# EMAIL
# =========================================================

def pick_breakout_setups(
    best_fut: Dict[str, MarketVol],
    n: int,
    session_name: str,
    universe_cap: int,
    scan_profile: str = DEFAULT_SCAN_PROFILE,
) -> List[Setup]:
    """
    Engine B selector: iterate symbols and build momentum breakout/breakdown setups.
    """
    items = list((best_fut or {}).items())
    # simple volume-based ordering
    items.sort(key=lambda kv: usd_notional(kv[1] or MarketVol()), reverse=True)
    if universe_cap and universe_cap > 0:
        items = items[:universe_cap]

    out: List[Setup] = []
    for base, mv in items:
        try:
            s = make_breakout_setup(base, mv, session_name=session_name, scan_profile=scan_profile)
            if s:
                out.append(s)
        except Exception:
            continue

    # order by confidence then volume
    out.sort(key=lambda s: (s.conf, s.fut_vol_usd), reverse=True)
    return out[: max(0, int(n)) ]




def user_email_alerts_enabled(user: dict) -> bool:
    # Admin always receives emails
    try:
        uid = int((user or {}).get("user_id") or 0)
        if uid and is_admin_user(uid):
            return True
    except Exception:
        pass
    try:
        return int((user or {}).get("email_alerts_enabled", 1)) == 1
    except Exception:
        return True


def set_user_email_alerts_enabled(uid: int, enabled: bool):
    update_user(int(uid), email_alerts_enabled=(1 if enabled else 0))

async def email_on_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """User command to enable/disable ALL email alerts without removing the saved email address."""
    uid = update.effective_user.id
    user = get_user(uid) or {}

    arg = (context.args[0].strip().lower() if context.args else "")
    if arg not in ("on", "off", "enable", "disable"):
        cur = "ON" if user_email_alerts_enabled(user) else "OFF"
        await update.message.reply_text(
            "📧 Email alerts (master switch)\n"
            "────────────────────\n"
            f"Current: {cur}\n\n"
            "Usage:\n"
            "/email_on_off on\n"
            "/email_on_off off\n\n"
            "Note: This does not remove your saved email address."
        )
        return

    enabled = arg in ("on", "enable")
    set_user_email_alerts_enabled(uid, enabled)
    await update.message.reply_text(
        "✅ Email alerts updated\n"
        "────────────────────\n"
        f"Now: {'ON' if enabled else 'OFF'}"
    )

def email_config_ok() -> bool:
    """
    Only checks SMTP sender config.
    Recipient is per-user (users.email_to OR users.email) OR fallback EMAIL_TO.
    """
    return all([EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FROM])


def send_email(
    subject: str,
    body: str,
    user_id_for_debug: Optional[int] = None,
    enforce_trade_window: bool = True
) -> bool:
    """
    Sends an email.

    Recipient resolution:
    - If user_id_for_debug is provided: uses users.email_to OR users.email
    - Otherwise falls back to EMAIL_TO (if set)

    Trade window:
    - Enforced only if enforce_trade_window=True
      (use False for system alerts like big-move alerts)

    Tracks last SMTP error + last email decision for /health and /email_decision.
    """
    global _SMTP_CONN, _SMTP_CONN_IS_SSL, _SMTP_CONN_TS

    # Per-user master email alerts switch
    if user_id_for_debug is not None:
        u = get_user(int(user_id_for_debug))
        if u and not user_email_alerts_enabled(u):
            _EMAIL_LAST_DECISION["reason"] = "user_email_alerts_disabled"
            _EMAIL_LAST_DECISION["ts"] = _now()
            return False

    uid = int(user_id_for_debug) if user_id_for_debug is not None else None

    # --- SMTP config check (sender side) ---
    if not email_config_ok():
        msg = "Email not configured (missing SMTP env vars)."
        logger.warning(msg)
        if uid is not None:
            _LAST_SMTP_ERROR[uid] = msg
            _LAST_EMAIL_DECISION[uid] = {
                "status": "FAIL",
                "reason": msg,
                "ts": time.time(),
            }
        return False

    # --- Resolve recipient ---
    to_email = ""
    user = None
    now_local = None

    if uid is not None:
        user = get_user(uid) or {}
        to_email = str(user.get("email_to") or user.get("email") or "").strip()

        if not to_email:
            _LAST_EMAIL_DECISION[uid] = {
                "status": "FAIL",
                "reasons": ["no_recipient_email_set"],
                "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            return False

        if enforce_trade_window:
            try:
                try:
                    tz = ZoneInfo(str(user.get("tz") or "UTC"))
                except Exception:
                    tz = timezone.utc
                now_local = datetime.now(tz)

                if not in_trade_window_now(user, now_local):
                    _LAST_EMAIL_DECISION[uid] = {
                        "status": "SKIP",
                        "reason": "outside_trade_window",
                        "ts": time.time(),
                    }
                    return False
            except Exception:
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reason": "trade_window_check_failed",
                    "ts": time.time(),
                }
                return False
    else:
        to_email = str(EMAIL_TO or "").strip()
        if not to_email:
            logger.warning("No recipient email (EMAIL_TO empty and no user_id provided).")
            return False

    # --- Build message ---
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = to_email
    msg.set_content(body)



      
    def _close_cached():
        global _SMTP_CONN, _SMTP_CONN_IS_SSL, _SMTP_CONN_TS
        try:
            if _SMTP_CONN is not None:
                try:
                    _SMTP_CONN.quit()
                except Exception:
                    try:
                        _SMTP_CONN.close()
                    except Exception:
                        pass
        finally:
            _SMTP_CONN = None
            _SMTP_CONN_IS_SSL = None
            _SMTP_CONN_TS = 0.0

    def _need_new_conn(is_ssl: bool) -> bool:
        global _SMTP_CONN, _SMTP_CONN_IS_SSL, _SMTP_CONN_TS
        if _SMTP_CONN is None:
            return True
        if _SMTP_CONN_IS_SSL is None or _SMTP_CONN_IS_SSL != is_ssl:
            return True
        if (time.time() - float(_SMTP_CONN_TS or 0.0)) > float(SMTP_REUSE_TTL_SEC or 0.0):
            return True
        return False


    def _connect_and_login(is_ssl: bool) -> smtplib.SMTP:
        timeout = float(EMAIL_SEND_TIMEOUT_SEC)
        if is_ssl:
            ctx = ssl.create_default_context()
            s = smtplib.SMTP_SSL(EMAIL_HOST, int(EMAIL_PORT), context=ctx, timeout=timeout)
        else:
            s = smtplib.SMTP(EMAIL_HOST, int(EMAIL_PORT), timeout=timeout)
            s.ehlo()
            s.starttls(context=ssl.create_default_context())
            s.ehlo()

        # login can be slow; keep under socket timeout
        s.login(EMAIL_USER, EMAIL_PASS)
        return s

    # --- Send (with connection reuse + lock) ---
    try:
        is_ssl = (int(EMAIL_PORT) == 465)

        with _SMTP_LOCK:
            # Refresh connection if needed
            if _need_new_conn(is_ssl):
                _close_cached()
                _SMTP_CONN = _connect_and_login(is_ssl)
                _SMTP_CONN_IS_SSL = is_ssl
                _SMTP_CONN_TS = time.time()
            else:
                # Light keepalive; if dropped, reconnect
                try:
                    _SMTP_CONN.noop()
                except Exception:
                    _close_cached()
                    _SMTP_CONN = _connect_and_login(is_ssl)
                    _SMTP_CONN_IS_SSL = is_ssl
                    _SMTP_CONN_TS = time.time()

            # Actual send
            _SMTP_CONN.send_message(msg)
            _SMTP_CONN_TS = time.time()

        if uid is not None:
            _LAST_SMTP_ERROR.pop(uid, None)
            _LAST_EMAIL_DECISION[uid] = {
                "status": "SENT",
                "reason": "ok",
                "ts": time.time(),
            }
        return True

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.exception("send_email failed: %s", err)

        # On any send failure, drop cached connection (often becomes poisoned)
        try:
            with _SMTP_LOCK:
                _close_cached()
        except Exception:
            pass

        if uid is not None:
            _LAST_SMTP_ERROR[uid] = err
            _LAST_EMAIL_DECISION[uid] = {
                "status": "FAIL",
                "reason": err,
                "ts": time.time(),
            }
        return False


# =========================================================
# STRIPE CHECKOUT / CUSTOMER PORTAL
# =========================================================
def create_checkout_session(email: str, plan: str) -> str:
    price_id = (
        os.environ.get("STRIPE_PRICE_STANDARD")
        if plan == "standard"
        else os.environ.get("STRIPE_PRICE_PRO")
    )

    session = stripe.checkout.Session.create(
        mode="subscription",
        customer_email=email,
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=TELEGRAM_BOT_URL,
        cancel_url=TELEGRAM_BOT_URL,
        metadata={"plan": plan},
    )
    return session.url


def create_customer_portal(email: str) -> str:
    customer = stripe.Customer.list(email=email, limit=1).data[0]
    portal = stripe.billing_portal.Session.create(
        customer=customer.id,
        return_url=TELEGRAM_BOT_URL,
    )
    return portal.url


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

async def email_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid) or {}

    if not context.args:
        cur = "ON" if user_email_alerts_enabled(user) else "OFF"
        saved = (user.get("email_to") or user.get("email") or "").strip()
        await update.message.reply_text(
            "📧 Email Settings\n"
            "────────────────────\n"
            f"Alerts: {cur}\n"
            f"Saved: {saved if saved else '(none)'}\n\n"
            "Set email: /email you@example.com\n"
            "Turn off: /email off\n"
            "Turn on: /email on"
        )
        return

    arg = context.args[0].strip().lower()

    # Support /email off|on
    if arg in ("off", "0", "disable", "disabled"):
        set_user_email_alerts_enabled(uid, False)
        await update.message.reply_text("✅ Email alerts: OFF")
        return

    if arg in ("on", "1", "enable", "enabled"):
        set_user_email_alerts_enabled(uid, True)
        await update.message.reply_text("✅ Email alerts: ON")
        return

    # Otherwise treat as an email address
    email = context.args[0].strip()
    if not EMAIL_RE.match(email):
        await update.message.reply_text(
            "❌ Invalid email format.\n"
            "Example: /email you@example.com\n"
            "Or: /email off"
        )
        return

    try:
        set_user_email(uid, email)
        set_user_email_alerts_enabled(uid, True)

        await update.message.reply_text(
            f"✅ Recipient email saved:\n{email}\n\n"
            "Email alerts are ON.\n"
            "Test now: /email_test\n"
            "Turn off any time: /email off"
        )
    except Exception as e:
        logger.exception("email_cmd failed")
        await update.message.reply_text(f"❌ Failed to save email: {type(e).__name__}: {e}")


async def email_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Alias: /email_on
    context.args = ["on"]
    await email_cmd(update, context)

async def email_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Alias: /email_off
    context.args = ["off"]
    await email_cmd(update, context)

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

    
    # Send (run in worker thread so it never blocks other Telegram commands)
    status_msg = await update.message.reply_text("📤 Sending test email…")
    try:
        ok = await _send_email_async(int(EMAIL_SEND_TIMEOUT_SEC), subject, body, uid, False)
    except Exception as e:
        logger.exception("email_test_cmd failed")
        try:
            await status_msg.edit_text(f"❌ Test email crashed: {type(e).__name__}: {e}")
        except Exception:
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
            return ordered or ['NY']
    except Exception:
        return ['NY']

def _guess_session_name_utc(now_utc: datetime) -> str:
    """
    Returns NY/LON/ASIA if within their UTC windows (priority NY > LON > ASIA).
    If we're in the 22:00–24:00 UTC gap, default to NY (keep UI consistent with /screen).
    """
    for name in SESSION_PRIORITY:  # ["NY","LON","ASIA"]
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
            return name
    return "NY"


def in_session_now(user: dict) -> Optional[dict]:
    tz = ZoneInfo(user["tz"])
    now_local = datetime.now(tz)
    now_utc = now_local.astimezone(timezone.utc)

    # NEW: unlimited mode => always return a session (no more NONE)
    if int(user.get("sessions_unlimited", 0) or 0) == 1:
        name = _guess_session_name_utc(now_utc)
        session_key = f"{now_utc.strftime('%Y-%m-%d')}_{name}_UNL"
        return {
            "name": name,
            "session_key": session_key,
            "start_utc": now_utc,
            "end_utc": now_utc,
            "now_local": now_local,
        }

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



import difflib

import base64
import tempfile

# =========================================================
# UNKNOWN COMMAND + "DID YOU MEAN" SUGGESTION (ALL USERS)
# =========================================================

KNOWN_COMMANDS = sorted(set([
    # Help
    "help", "help_admin",

    # Market scan
    "screen",

    # Position sizing / risk
    "size", "riskmode", "dailycap",

    # Trade journal & equity
    "equity", "equity_reset",
    "trade_open", "trade_sl", "trade_rf", "trade_close",

    # Status
    "status",

    # Limits
    "limits",

    # Sessions
    "sessions", "sessions_on", "sessions_off", "sessions_on_unlimited", "sessions_off_unlimited",

    # Email alerts
    "notify_on", "notify_off",
    "trade_window",
    "email_test", "email_decision",
    "bigmove_alert",

    # Cooldowns (user)
    "cooldowns", "cooldown",

    # Reports
    "report_daily", "report_weekly", "report_overall",
    "signals_daily", "signals_weekly",

    # System health
    "health", "health_sys",

    # Timezone
    "tz",

    # Billing / plan / support
    "myplan", "billing", "support",

    # USDT user
    "usdt", "usdt_paid",

    # Admin cooldown controls
    "cooldown_clear", "cooldown_clear_all",

    # Admin data/recovery
    "reset", "restore",

    # USDT admin
    "usdt_pending", "usdt_approve", "usdt_reject",

    # Payments & access admin
    "admin_user", "admin_users", "admin_payments",
    "admin_grant", "admin_revoke",
]))

def _normalize_cmd(text: str) -> str:
    s = (text or "").strip()
    if not s.startswith("/"):
        return ""
    s = s[1:]  # remove "/"
    if " " in s:
        s = s.split(" ", 1)[0]
    if "@" in s:
        s = s.split("@", 1)[0]
    return s.lower()

async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update or not update.message:
        return

    raw = update.message.text or ""
    cmd = _normalize_cmd(raw)
    if not cmd:
        return

    # Suggest closest command
    matches = difflib.get_close_matches(cmd, KNOWN_COMMANDS, n=1, cutoff=0.62)
    suggestion = matches[0] if matches else ""

    if suggestion:
        msg = (
            "❌ *Unknown command*\n\n"
            f"Did you mean: `/{suggestion}` ?\n\n"
            "Type /help to see all commands."
        )
    else:
        msg = (
            "❌ *Unknown command*\n\n"
            "Please check the spelling.\n"
            "Type /help to see all commands."
        )

    try:
        await update.message.reply_text(
            msg,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
    except Exception:
        # plain text fallback
        if suggestion:
            await update.message.reply_text(
                f"❌ Unknown command\n\nDid you mean: /{suggestion} ?\n\nType /help to see all commands."
            )
        else:
            await update.message.reply_text(
                "❌ Unknown command\n\nPlease check the spelling.\nType /help to see all commands."
            )



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

def entry_sanity_check(entry: float, current: float) -> Tuple[Optional[str], bool]:
    """
    Returns (warning_message, severe_flag).
    - warning_message is None if entry looks normal.
    - severe_flag True means it's extremely off and should be blocked unless user confirms.
    """
    try:
        e = float(entry)
        c = float(current)
    except Exception:
        return ("⚠️ Entry sanity check: could not compare entry vs current price.", False)

    if c <= 0 or e <= 0:
        return ("⚠️ Entry sanity check: entry/current must be positive.", True)

    diff_pct = abs(e - c) / c * 100.0
    ratio = e / c

    # Detect common decimal-shift mistakes (x10 or /10)
    dec_shift = ""
    if 8.0 <= ratio <= 12.5:
        dec_shift = " (looks like ~x10 higher — possible decimal mistake)"
    elif 0.08 <= ratio <= 0.125:
        dec_shift = " (looks like ~x10 lower — possible decimal mistake)"

    # Tune thresholds here
    WARN_PCT = 15.0
    SEVERE_PCT = 40.0

    if diff_pct < WARN_PCT:
        return (None, False)

    if diff_pct >= SEVERE_PCT:
        msg = (
            f"🚨 ENTRY PRICE LOOKS WRONG\n"
            f"- Current: {fmt_price(c)}\n"
            f"- Entered: {fmt_price(e)}\n"
            f"- Difference: {diff_pct:.1f}%{dec_shift}\n"
            f"Double-check your entry value."
        )
        return (msg, True)

    msg = (
        f"⚠️ Entry looks unusual vs current price\n"
        f"- Current: {fmt_price(c)}\n"
        f"- Entered: {fmt_price(e)}\n"
        f"- Difference: {diff_pct:.1f}%{dec_shift}\n"
        f"Please double-check."
    )
    return (msg, False)


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
# USDT Payment
# =========================================================

async def usdt_paid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2:
        await update.message.reply_text(
            "Usage:\n/usdt_paid <TXID> <standard|pro>"
        )
        return

    txid, plan = context.args
    plan = plan.lower()

    if plan not in ("standard", "pro"):
        await update.message.reply_text("Plan must be 'standard' or 'pro'")
        return

    if not is_valid_txid(txid):
        await update.message.reply_text("❌ Invalid TXID format.")
        return

    if usdt_txid_exists(txid):
        await update.message.reply_text("⚠️ This TXID has already been used.")
        return

    save_usdt_payment(
        update.effective_user.id,
        update.effective_user.username,
        txid,
        plan
    )

    await update.message.reply_text(
        "✅ Payment submitted.\n"
        "Status: Pending admin approval."
    )

    await context.bot.send_message(
        chat_id=int(os.getenv("USDT_ADMIN_CHAT_ID")),
        text=(
            "🧾 New USDT payment request\n\n"
            f"User: @{update.effective_user.username}\n"
            f"Plan: {plan.upper()}\n"
            f"TXID: {txid}\n\n"
            f"Approve with:\n"
            f"/usdt_approve {txid}"
        )
    )

def approve_usdt(txid: str):
    conn = sqlite3.connect("bot.db")
    cur = conn.cursor()
    cur.execute("""
        UPDATE usdt_payments
        SET status = 'APPROVED'
        WHERE txid = ?
    """, (txid,))
    conn.commit()
    conn.close()

async def usdt_approve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != int(os.getenv("USDT_ADMIN_CHAT_ID")):
        await update.message.reply_text("❌ Admin only.")
        return

    if len(context.args) != 1:
        await update.message.reply_text("Usage: /usdt_approve <TXID>")
        return

    txid = context.args[0]

    approve_usdt(txid)

    await update.message.reply_text("✅ USDT payment approved.")

    # 🔑 ACCESS GRANT HOOK
    # TODO:
    # grant_standard_access(user_id)
    # grant_pro_access(user_id)



# =========================================================
# FULL USER GUIDE (EMBEDDED DOCX)
# =========================================================
GUIDE_FULL_DOCX_NAME = "PulseFutures_User_Guide.docx"
GUIDE_FULL_DOCX_B64 = """UEsDBBQABgAIAAAAIQDKFrjBuAEAAF4KAAATAAgCW0NvbnRlbnRfVHlwZXNdLnhtbCCiBAIooAACAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADElt1OwkAQhe9NfIdmbw1d1MQYQ/HCn0slER9g2Z3C
avcnu4PA2zul0BgDFIWGG5Iyc875Zrdppnc/N0XyBSFqZzN2mXZZAlY6pe04Y+/D584tSyIKq0ThLGRsAZHd98/PesOFh5iQ2saM
TRD9HedRTsCImDoPliq5C0YgPYYx90J+ijHwq273hktnESx2sPRg/d4j5GJaYPI0p78rkg8PY5Y8VI1lVsa0KQ2WBb5RE6CIvzTC
+0JLgVTnX1b9IuusqFJSLnviRPt4QQ1bEsrK9oCV7pWOM2gFyUAEfBGGuvjMBcWVk1NDynS3zQZOl+daQq0v3XxwEmKkezJFWleM
0HbNv5XDTs0IAimPD1JbN0JEXBQQj09Q+TbHAyIJ2gBYOTcizGD01hrFD/NGkNw5tA7buI3auhECrGqJYe3ciDABoSBcHp+gMt4z
/+pk+eVltTJ/Zbxnfgvz75lfHdP1ic+/hfy9z5/yxKiANghW1o0QSCsEVL+Hv4lLm12R1DkIzkdaScI/xl7vD6W6QwN7CKh3f2nq
RLI+eD4oVxMF6q/ZchrRmYPjK5sN4Xy5Hfa/AQAA//8DAFBLAwQUAAYACAAAACEAmVV+Bf4AAADhAgAACwAIAl9yZWxzLy5yZWxz
IKIEAiigAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKySTUsD
MRCG74L/Icy9O9sqItLdXkToTWT9AUMy+4GbD5Kptv/eKIou1LWHHjN558kzQ9abvR3VK8c0eFfBsihBsdPeDK6r4Ll5WNyCSkLO
0OgdV3DgBJv68mL9xCNJbkr9EJLKFJcq6EXCHWLSPVtKhQ/s8k3royXJx9hhIP1CHeOqLG8w/mZAPWGqrakgbs0VqOYQ+BS2b9tB
873XO8tOjjyBvBd2hs0ixNwfZcjTqIZix1KB8foxlxNSCEVGAx43Wp1u9Pe0aFnIkBBqH3ne5yMxJ7Q854qmiR+bNx8Nmq/ynM31
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
AGkAcwBwAGwAYQB5ACAAUAAzbWx1YwAAAAAAAAABAAAADGVuVVMAAAA0AAAAHABDAG8AcAB5AHIAaQBnAGgAdAAgAEEAcABwAGwA
ZQAgAEkAbgBjAC4ALAAgADIAMAAyADJYWVogAAAAAAAA9tUAAQAAAADTLFhZWiAAAAAAAACD3wAAPb////+7WFlaIAAAAAAAAEq/
AACxNwAACrlYWVogAAAAAAAAKDgAABELAADIuXBhcmEAAAAAAAMAAAACZmYAAPKnAAANWQAAE9AAAApbc2YzMgAAAAAAAQxCAAAF
3v//8yYAAAeTAAD9kP//+6L///2jAAAD3AAAwG7/wAARCAQ7BQQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQF
BgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJico
KSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5
usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QA
tREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5
OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbH
yMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9sAQwADAwMDAwMEAwMEBgQEBAYIBgYGBggKCAgICAgKDQoKCgoKCg0NDQ0N
DQ0NDw8PDw8PEhISEhIUFBQUFBQUFBQU/9sAQwEDAwMFBQUJBQUJFQ4MDhUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUVFRUV
FRUVFRUVFRUVFRUVFRUVFRUV/90ABABR/9oADAMBAAIRAxEAPwD85aKKK6znCiiimgCiiiiwBRRRRYAooopAFFFFABRRRQAUUUUA
FFFFABRRRQAUUUUAFFFFABRRRQBLRRRTYBRRRSAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAcOlLSDpS1QBRR
RQAUUUUAFFFFABRRRQAUUUUAOHSikBGKXipaAaetFB60VQCilpBS0mAUUUUgCiiigAooooAKKKKACiiigAooooAfSjrSUU2A+iky
KOKRLQtFJxRxQFhCRmk4pCDmkwaBpjqUdaADigA5oC46iik4oJFopOKOKAFooooAKKKKAHDpRSAjFLxQ0AUUUUAFFFFABRRRQAUU
UUAFFFFABRRRQAUUUUAOHSlpB0pabAKKKKQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAOFLSCjimwFopOKOKQCN1pK
U9aSmwHClpBS0MAooopAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFNgFFFFIAooooAKKKKAP/0PzlooorrOcKKKKACiiigAoo
opoAoooosAUUUUWAKKKKQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAEtFICMUvFDQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRR
QAUUUUAFFFFADh0paQEYo4qgFopOKOKAFooooAKKKKACiiigAooooAKKKKACiiigBRS0gpeKloAooooAKKKKACiiigAooooAKKKK
ACiiigAooooAfRSAjFLxQ0AUUcUcU7AFKOtJSjrSAdRRRQSwooooENbrSUrdaSgsKUdaSlHWgB1FFFBAUUUUAFFFFADh0opARil4
oaAKKOKOKdgCiiikAUUUUAFFFFABRRRQAUUUUAOHSlpARil4psAooopAFFFHFABRRxRxQAUUUUAFFFHFABRRRQAUUUUAFFFFABRR
RQAUUUUAFFFFABRRRQA4UtIKOKbAWiiikAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFNgFFFFIAooooA/9H85aKKK6zn
CiiigAooooAKKKKACiiigAooooAKKKKbAKKKKQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAEtFIOlLTYBRRRSAKKKKACiiigAoooo
AKKKKACiiigAooooAKKKKACiiigAooooAcOlLSDpS1QBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFACilpBS0mAUUUUgCiiigAooooA
KKKKACiiigAooooAKKKKACiiigB9KOtJRTYD6KKKRLQUUUUBYa3WkpT1pKCgpR1pKUdaAHUUUUEBRRRQAUUUUAFFFFABRRRQA4dK
KQEYpeKbAKKKKQBRRRQAUUUUAFFFFABRRRQA4dKWkBGKXimwGnrSUp60lDAKKKKQDhS0gpeKAEPSkHWlJGKQdaaAdRRRSAKKKKAF
BxRk00nFGTTsA4nNJSA5paQBRRRQAUUUUAFFFFABRRRQA4UtIKOKbAWik4o4pALRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQA
UUUU2AUUUUgP/9L85aKKK6znCiiigAooooAKKKKACiiigAooooAKKKKACiiimwCiiikAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQB
LRSDpS02AUUUUgCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAcOlLSDpS1QBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAC
ilpBS0mAUUUUgEPWlPSkPWlPSqAQUtIKWkwCiiikAUUUUAFFFFABRRRQAUUUUAPooopsAooo4pAFFFFABRRRQAUo60lKOtADqKKK
CAooooAKKKKACiiigAooooAKKKKAHDpRQOlFNgFFFFIAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAHClpBS02AUUUUg
CiiigAooooAKKKKACiiigAooooAKKKKACiiigAooop3AUU6minUMAooopAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFO4H//T
/OWiiius5wooooAKKKKACiiigAooooAKKKKACiiigAooopsAooopAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAEtFFFNgFFF
FIAooooAKKKKACiiigAooooAKKKKACiiigAooooAcOlLSDpS1QBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFACilpBS0mAUUUUgCiii
gAooooAKKKKACiiigAooooAKKKKACiiigB9FIOlLTYCN1ptKetJSAfRRRTYBRS4NGDSASlHWjBowaAHUUUUEBRRRQAUUUUAFFFFA
BRRRQAUUUUAOHSigdKKbAKKKKQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAOFLSL0pabAKKKKQBRRRQAUUUUAFFFFA
BRRRQAUUUUAFFFFABRRRQAUUUUAKKdTRTqbAKKKKQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQB//9T85aKKK6znCiiigAoo
ooAKKKKACiiigAooooAKKKKACiiimwCiiikAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAS0Ug6UtNgFFFFIAooooAKKKKACo
z1NB6mnr0FAAvQUtFFABRRRQAUUUUAFFFFABRRRQA4dKWkHSlqgCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAUUtIKW
kwCiiikAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQA+lHWmjpS02A+iiikQFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRT1
AcOlFA6UUMAooopAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQA4UtIvSlpsAooopAFFFFABRRRQAUUUUAFFFFABRRRQ
AUUUUAFFFFACinU0U6mwGt1pKVutJSAUU6minUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAf/9X85aKKK6znCiiigAooooAKKKKA
CiiigAooooAKKKKACiiincAoooouAUUUUXAKKKKGAUUUUgCiiigAooooAKKKKACiiigAyaMmiigCQdKWkHSlpsAooopAFFFFABRR
RQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQA4dKWkHSlqgCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAdRRRSYBRRRSA
KKKKACiiigAooooAKKKKACiiigApR1pKUdaAHUUUU7kDh0paQdKWkAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQA4dKKB0opsAooop
AFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRTuAUUUUXAcKWkXpS0MAooopAFFFFABRRRQAUUUUAFF
FFABRRRQAUUUUAFFFFABRRRTuAUUUUXAUU6minUMAooopAFFFFABRRRQB//W/OWiiius5wooooAKKKKACiiigAooooAKKKKACiii
gAooooAKKKKACiiigAooopsAooopAFFFFABRRRQAUUUUAFFFFABRRRQADinMaQgACkHWqAMmjJoPBNFS2AZNGTRRTuBIvQUtIOgp
aGAUUUUgCiiigAooooAKKKKACiiigAooooAKKKKAHDpS0g6UtUAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAOooHSikwDvRS
Z5pT0pgFFFFJgFFFFIAooooAKKKKACiiigApR1pKUdaAHUUUUEscOlLSDpS0CCiiigAooooAKKKKACiiigAooooAKKKKACiiigAo
oop6gOHSigdKKGAUUUUgCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKdwHClpBS0MAooopAF
FFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAop1NFOpsAooopAFFFFAH//1/zlooorrOcKKKKACiiigAooooAKKKKACiii
gAooooAKKKKACiiigAooooAKKKKbAKKKKTuAUUUU2AUUUUgCiiigAooooAKKKKACiiigAooooAKKKKAJB0FLSDoKWmwCiiikAUUU
UAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAOHSlpB0paoAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAHUUUUmAUUUUgC
iiigAooooAKKKKAClHWkooAfRRRQSxw6UtIOlLQIKKKKACiiigAooooAKKKKACiiigAooooAKKKKAHDpRQOlFNgNPWig9aKLgOHS
igdKKGAUUUUgCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAcKWkXpS02AUUUUgCiiigAooooAKKK
KACiiigAooooAKKKKACiiigAooooAUU6minU2AUUUUgCiiigD//Q/OWiiius5wooooAKKKKACiiigAooooAKKKKACiiigAooooAK
KKKACiiigAooop3AKKKKGAUUUUgCiiigAooooAKKKKACiiigAooooAKKKKACiiigCQdBS0g6ClpsAooopAFFFFABRRRQAUUUUAFF
FFABRRRQAUUUUAFFFFADh0paQdKWqAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAHUUUUmAUUUUgCiiigAooooAKKKKACiiig
AooooAKKKKAHDpS0g6UtNgFFFFIAooooAfRRRQQFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRTuA4dKKB0ooYBRRRSAKK
KKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAHClpF6UtNgFFFFIAooooAKKKKACiiigAooooAKKKKACiiigAo
oooAKKKKAFFOpop1NgFFFFID/9H85aKKK6znCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooop3AKKKK
GAUUUUWAKKKKQBRRRQAUUUUAFFFFABRRRQAUUUUAGTRk0UUAGTRk0UU7gSL0FLSDoKWhgFFFFIAooooAKKKKACiiigAooooAKKKK
ACiiigBw6UtIOlLVAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAOooHSikwCiiikAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUA
FFFFO4Dh0paQdKWhgPooopEBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAOHSigdKKbAKKKKQBRRRQAUUUUAFFFFABRR
RQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFO4DhS0gpaGAUUUUgCiiigAooooAKKKKACiiigAooooAKKKKAFFOpop1N
gFFFFID/0vziooors1OcKKKKSYD6KKKACiiigAooooAKKKKACiiigBlOptOpsBaKKKQBRRRQAUUUUAFFFFABRRRTuAUUUUMAooop
AFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABk0ZNFFAEi9BS0g6ClpsAooopAFFFFABRRRQAUUUUAFFFFADh0paQdKWqAaetJ
SnrSUmA4UtIKWmAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQA6iiikwCiiikAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAOHSl
pB0pabAKKKKQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQA+iiiggKKKKACiiiqAKKKKAHUUUVIBRRRQAUUUUAFFFFABRRRQA
UUUUAFFFFABRRRQAUUUUAFFFFABRRRQA4UtIKWmwCiiikAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAop1NFOpsAooopAf//T/OWi
iius5xlFFFNgFFFFFwCiiii4BRRRRcB9FFFIAooooAKKKKAGnrS0tNPWgB1FJS0AFFFFABRRRQAUUUUAFFFFNgFFFFIAooooAKKK
KACiiigAooooAKKKKACiiigAooooAKKKKAJB0FLSDoKWmwCiiikAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAOFLSClqgCii
igAooooAKKKKACiiigAooooAdRQOlFJgI3WkoPWimA6igdKKTAKKKKQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRT1AcOlLSD
pS0MAopcGjBpAJRS4NGDQAlFFLg0AJRS4NGDTsAlFFFIAooooAfRRRQQFFFFADT1pKUg5owaCkKKWkAOKXBqhMD1ooPWipbEKKWk
FLTYBRRRSAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAcKWkFLTYBRRRSAKKKKACiiigBDQKDQKAFooooAUHFG
aaTijNOwDic0lIDmlpAFFFFO4H//1Pziooorsuc4UUUUgCiiigAooooAKKKKACiiincAoooouA6lpKWkAUUUUAFFFFABRRRQAUUU
UAFFFFABRRRTuAUUUUagFFFFIAooooAKKKKACiiigAooooAKKKKACiiigAooooAMmjJoooAMmjJoop6gSL0FLSDoKWhgFFFFIAoo
ooAKKKKACiiigAooooAKKKKACiiigAooooAcKWkFLVAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAOooopMAooopAFFFFABRRRQAUU
UUAFFFFABRRRQA4AYpeKQdKWgAooooAU9aSg96dQQAOaMig470nB707BYXANLSZFH1pAJjByDgmkOfuDj3rT07R9R1i5jtNMt3nn
kOFVecmvpTw/+yr491LRJ9c1iJtLgiTeBKh5GM8FTSdSxahc+ViSeoxSjpVzU7JtOv57JjuMLFCfXBxVMdKd7gLRRRQA+iiiggKK
KKACiiigAooooAKKKKACiiigBRS0gpabAKKKKQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFADhS0gpaACiiigA
ooooAKKKKACiiigAo4oooAKKKKACiiigD//V/OKiiius5xMGjBp2DRg1QCUUUVIBRRRQAUUUUAFFFFABRRRQA6lpKWmwCiiikAUU
UUAFFFFABRRRQAUUUUAFFFFABRRRTYBRRRSAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigCQdBS0g6ClpsAooopAFFFF
ABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFADhRxQKWqAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAHUUDpRSYBRRRSAKKKKACi
iigAooooAKKKKACiiigAoopcGgBR0paQdKWmwFA5JJpCTmkOSc04EjoMmhuxCTGkkuAO3UetBGDgAqD2J5ru/CXw58Y+M51g8P6T
Pe+YwDNGAdue/UV95fDH9iNilvqfjW4wOGa2G9HHsTyKxdSxqkfnxoHhTXfE1ytpollLcyOQAFAPWvtT4ZfsX6/rIhvvF7/YIThj
CysGI+qkiv0c8J/C/wAHeDbZINH02JfLAG91VmP/AAIjNehogEeAu0DoBUOo2WoHjXgL4FeAvAMEcWnaessyAfPKFkOfUFhmuw8d
xRw+EdSjRVVRC2AoAH3T2FdsMjknH1rjfiA2PCWpk94W5/A1N7lKyP59PFII8Q6hn/ns/wD6Eawh0rf8ULjxDqGTnMz/APoRrnz1
rqWxj1HUUg6UtAh9FFFBAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAopaQUtNgFFFFIAooooAKKKKACiiigAooooAKKKKACiiigAo
oooAKKKKACiiigBwpaQUtACN1ptObrTaAHClpBS02AUUUUgCiiigAooooAKKKKAP/9b84qKKK6znCiiihMB3FHFLRTuAyiiihgFF
FFIB3FHFLRTuAnFHFLRRcAooopAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFO4BRRRSAKKKKACiiigAooooAKKKKACiiigAoo
ooAKKKKACiiigCQdBS0g6ClpsAooopAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQA4UtIKWqAKKKKACiiigAooooAKK
KKACiiigAooooAKKKKAHUUUUmAUUUUgCiiigAooooAKKKKACiiigB4YjGeRSAgMSDhT60byB8oyauWNjfajKttZW7TSucAKM0nIS
gU+OCp605EaWQRxjLnoPWvqL4d/srfELxjJFcXVs2n2UmCXlQ4I9ipr7++G37JPgTwd5d3qUZv71cEsxLJn/AHWFS6hagfmJ4C+B
PxB8fSqLDTZYbaQjbO65THrwc198/C39irQtEWK/8aSm/uRhtkTMig+6sCK+6NO0bTtMhENhawwKowBGiqP/AB0CtLacYbn6Vm5X
NjmdA8H+HfDVtHaaRp8FukYADLGob8WABrqlGOPSmr06Yp2ahgOwKO1IDkUtSwGEc15/8TpPK8Gamen7o/yNegZFeX/F+cQeBdUb
0jNXAln4HeJSTrt8fWVv/QjWJWxr7h9YvGHeVv8A0I1j10mTCiiigQUUUUAFFFFABRRRQAUUUUAFFFFABRRRQBIvSl4pF6UtO5AU
UUUgCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigBwpaQUtACN1ptObrTaAHClpBS02AhptONNpAOFLSClo
AKKKKACiiigD/9f84qKKK6znCiiigBcmjJpKKACiiigAooooAXJoyaSigBcmjJpKKAFyaMmkooAdS0lLQAUUUUAFFFFABRRRQAUU
UUAFFFFABRRRQAUUUU2AUUUUgCiiigAooooAKKKKACiiigAooooAKKKKACiiigAyaMmiigCRegpaQdBS02AUUUUgCiiigAooooAK
KKKACiiigAooooAKKUAmnCNz0FLmS0bE2luMopSCKSmu499RwpaQdKWmgCiiimAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQA6igdK
KTAKKKKQBRRRQAUUUUAFFFFABRRRQB2PgSy8Pah4htrfxLci1sHbDuzFQCSMcgGv2V+Dfwh+FGlaLBqfhyKLUTOobzZCJR6cbhxX
4csCwAHVTnjrxXvPwi+P/jD4Y30aJO0+lsw3xPliAOPlLHA4qJIpM/d2C3gtUEVtEsaDgKgAA/AVMQR92vDPhN8cfCvxN0yKaynW
K7ZQXhZgWBx6CvclYsM9M1zyTNUyQHI9KWmgjHBp1QmxhSYFLRWiYCr0paRelLSkAw9q8c+O8oi+G+rvnGIxzXsZOK8G/aOlMPwu
1lhx8g/nVQRLPwr1Rt2oXDert/M1Qqzdtuupj6sf51WrpMmFFFFAgooooAKKKKACiiigAooooAKKXBowaAEopcGjBoAWloooIFFL
SCloAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigBwpaQUtNgFFFFIAooooA/
/9D84qKKK6znCiiigAooooAKKKKACiiigAooooAKKKKACiiigBcmjJpKKAHUtJS0AFFFFABRRRQAUUUUAFFFFABRRRQAUUUU7gFF
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
kEdxVP4uftQeKPiDZjTbeT7LasuGVQAWyOeVIr5PYs7tI5LM5ySeck9aqMAlMaCGOc5J6/WnU0DBxTq06kBRRRSAKKKKbAKKKKQB
RRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUASDpS0g6UtNgFFFFIAooooAKKKKACiilHWmht6XPQfhz/yGh9D/ACr6Mxya
+cfh0f8Aicj6N/Kvo4dTXwed/wC8N+R/SPh9rlq9Tzr4hpnSifTH86+dQOD9a+lPHqhtElb0Ir5tHIx9a+hyR/uNT8t8REljm12A
9aB1pCeaK9qXQ+BUfdbFPWkpT1pKCEFFFFAwooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigBwpaQUtUAUUUUAFFF
FABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAKKWkFLSYBRRRSAKKKKACiiigD//0PzdooorrOQK
KKKACiiigAooooAKKKKDYKKKKACiiigxHUUUU2WFFFFIAooooAKKKKACiiigAooooAKKKKACiiigB9FFFNgFFFFIBB3oNBpB1qgF
HSlooqQCiiigAooooAKKKKACiiigAooooAcOlLSDpS02AUUUUgCiiigAooooAKKKKACiiigBMcYpaKKAEIzS0UUAFFFFABRRRTuA
UUUUgCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAAHFLk0lFAC5NGTSUUASL0FLSDoKWgAooooAcenFIvWjnpQB
jrTUnawfDTcUzpfCmpDTNVjllO1G43H3r6Ut9Tt5oBJFKrKwzmvkYk9jggirianexgCOdyAMYDED+deRjcoWIdz7zh3jWeX0XTse
ueP9fhWAWMDiQv8AexzjBrxc5BwelDyTSHe7M7N/eJJ/WlBGQh4x1ruw2H9mlFdD5fOM1ljKjqPqNIwfWhaADkig8V0s81SvHlFP
WkoopEJW0CiiigYUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFO4BRRRRcBwpaQUtMAooooAKKKKAC
iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigBRS0gpaTAKKKKQH/0fzdooorrOQKKKKACiiigAooooAKKKKD
YKKKKACiiigxHUUUU2WFFFFIAooooAKKKKACiiigAooooAKKKKACiiigB9FFFNgFFFFIAooooAKKKKACiiigAooooAKKKKACiiig
AooooAKKKKAHDpS0g6UtNgFFFFIAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKbAKKKKQBRRRQAUUUUAFFFFABRRRQAUU
UUAFFFFABRRRQAUUUUAFFFFABRRRQAA4pcmkooAkXoKWkHQUtACg4pCc0pPFHU07qWgR095oaDmnDA4FGMc0gGaLSjsVJRlrewpP
OR1pp5570vfFPwOuaLijNR0G5oNJRSbuJx15kFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUU
AFFFFABRRRQA4UtIKWqAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAUUtIKWkwCiiikB//
0vzdooorrMgooooAKKKKACiiigAooooAKKKKACiiigAooooAdRRRQAUUUUCYUUUUEhRRRQAUUUUAFFFFABRRRQA+iiimwCiiikAU
UUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABS5NJRQA4dKWkHSloAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiii
mwCiiikAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAEg6ClpB0FLTYDCTmkyaGpKQEgORS0g6Clp
sBjE5pMmlbrTaQEi9BS0g6CloAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiig
AooooAKKKKdwHClpBS0wGt1pKVutJUtgOFLSClqgCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD//0/zd
ooorrMgooooAKKKKACiiigAooooAKKKKACiiigAooooAMmjJoooAdRRRQJsKKKKCQooooAKKXBpKACiiigAooooAKKKKdwH0UUUg
CiiigAooooAKKKKACiiigAooooAKKKKAEOelKCRSEAclsZpoePpup3AkyTzijJphdB0NKhVjjdTAWilIAOAcikqQClyaSigBw6Ut
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
AUMRSHnrzRRTuAdvajcfWiikADjpRnvRRTYC7jgjPWkzwB6UUUgDJoJJx7UUUAKCRn3pCc5zzmiigBScgD0pCSSCeooop3AXJpM4
oopALuOc5oyaSigAooooAM8YoJJoooACcgA9BQxLEE84oooAXc3rQWJ70lFAC5PPvSZPB9KKKAAkmgEiiigABIJI70UUUAJjnNOJ
JGD0pKKAF3Hj2oJJ6mkooAASDnvR3J7nrRRTuAUZNFFIAJzRRRQAAkUuTSUUAPUkjmnU1elOpsAooopAFFFFABRRRQAUUUUAFFFF
ABRRRQAUUUUAISRjFLTW7U6gB2BRgUtFUB//1fzaooorrMgooooAKKKKACiiigAooooAKKKKACiiigAooooAXJoyaSigB9FFFABR
RRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQYhRRRQAZNGTRRQA6iiigsOKOKKKACiiigAooooAOKOKKKACiiigAooooAKKKKAFya
Mmkop3AKKKKQDh0paQdKWmwCiiikAUUUUAFLk0lFABRRRQAUUUUAFLk0lFAC5NGTSUU7gPopKWkAUUUUAFFFFABRRRQAUUUUAFFF
FABRRRQAUUUUAFFFFO4BRRRSAKKKKACiiincAooopAFFFFDYBRRRTuAUUUUgCiiigAooooAKKKKACiiigAooooAKKKKACiiigAoo
ooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAAEilyaSigB6kkc06mr0p1NgMYkdKTJpWptIB6kkc06mp
0NOpsAooopAFFFFABRRRQAUUUUAGM0UUUALk0ZNJRTuB/9b82qKKK6zIKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigBc
mjJpKKAH0UUUAFFFFADSTmjJoPWkoAfRRRQAUUUUAFFFFABRRRQYhRRRQAUUUUAGTRk0UUAOHSigdKKCxCTmkyaD1ooJbHDpRQOl
FBQUUUUAFFFFABRRRQAUUUUAFFFFABRRRQA4dKWkHSlpsAooopAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUALk0ZNJRTuAuTRk0lF
FwFyaMmkoouAuTRk0lFFwH0UlLSAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooop3AKKKKQBRRRTuAUUUUgCiiinc
AooopAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAACRS5NJRQ
A4c9adhfSkToadTuAw8dKTJpWptFwFyaMmkoouA9SSOadTU6GnUMAooopANYkDihSSOaUgGmtwOKAH0U1SSOadQAUUUUAf/X/Nqi
iiusyCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAFyaMmkooAXJoyaSigB3FHFLRQAUUUUAFFFFABRRRQAUUUUG
IUUUUAFFFFACgDFLgUUUFWG5xxRk0HrRTYmwooopCHDpRQOlFNlhRRRSAKKKKACiiigAooooAKKKKACiiigBw6UtIOlLTYBRRRSA
KKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAFBxSk4pMGkHNOwDgc0tA4opAFFFFABRRRQAUUUUAFFFFABR
RRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQA
UUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAAJFLk0lFAASTRRRQAUUUUAAJFLk0lFAD1JPWnU1Ohp1NgFIQDS0UgEAApaKKACi
iigD/9D82qKKK6zIKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigB9FFFABRRRQAUUUUAFFFFABRRR
QAUUUUCaCiiigyDJoyaKKACiiigAooooAMmjJoooAcOlFA6UUFhRRRQAUUUUAFFFFABRRRQAUUUUALk0ZNJRTuAuTRk0lFFwFyaM
mkoouAuTRk0lFFwHDpS0gpaGAUUUUgCiiigAooooAKKKKACiiigAooooAKKKKACiiigBcmjJpKKAH0UlLQAUUUUAFFFFABRRRQAU
UUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRTuAUUUUXAKKKKLgFFFFDAKKKKQBRRRQAUUUUAFF
FFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAAJFLk0lFAD1JPWnU1Ohp1NgFFFFIAooooA/9H8
2qKKK6zIKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigBcmjJpKKAH0UUUAFFFFABRRRQAUUUUAFFF
FABRRRQYhRRRQAUUUUAFFFFABRRRQAZNGTRRTuA4dKKB0opFhRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAKDijJpKKA
FyaMmkop3AcDmlpBS0MAooopAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAOpaSlpsAooopAFFFFABRRRQAUUUUAFFFFABRRRQAUUU
UAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRTuAUUUUXAKKKKGAUUUUgCiiigAooooAKKKKACiiig
AooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigB6dDTqanQ06mwCiiikAUUUUAf//S/NqiiiusyCiiigAooooAKKKK
ACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAFyaMmkooAfRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUU
CsFFFFBkKAMUuBQOlFDZSQUUUUDCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigBwpaQUtNgFFFFIAooooA
KKKKACiiigAooooAKKKKACiiigBcmjJpKKdwFyaMmkoouA+ikpaQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFA
BRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUU7gFFFFIAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAoo
ooAKKKKACiiigAooooAKKKKAFBIo3GkooAeCT1p1NToadTYBRRRSA//T/NqiiiusyCiiigAooooAKKKKACiiigAooooAKKKKACii
igAooooAKKKKACiiigAooooAKKKKACiiigBcmjJpKKAH0UUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFBiGTRk0UU7gOHSi
gdKKRYUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAKDijJpKKAH0UUUAFFFFABRRRQAUUUUAFFFFABRRRQ
AUUUUAFFFFABRRRQA6lpKWgBpJzRk0HrSUAOHSlpB0paAGknNGTQetJQA4dKWkHSloAKKKKACiiigAooooAKKKKACiiigAooooAK
KKKACiiigAooooAKKKKACiiigAooooAKKKKdwCiiihgFFFFIAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooo
oAKKKKACiiigBygHrTsL6UidDTqAGMAOlNp79BTKbAenQ06mp0NOoYBRRRSA/9T82qKKK6zC4UUUUFBRRRQAUUUUAFFFFABRRRQA
UUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQA+iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooMQooooAMmj
Joop3AMmjJooouO44dKKB0opFBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAD6KKKbAKKKKQBRRRQAUUUUAFFFF
ABRRRQAUUUUAFFFFABRRRQAuTRk0lFABRRRQAuTRk0lFO4BRRRSAXJoyaSincBw6UtIOlLSAKKKKACiiigAooooAKKKKACiiigAo
oooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKbAKKKKQBRRRTYBRRRSAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiii
gAooooAKKKKAFBIo3GkooACSaKKKAHp0NOpqdDTqbAKKKKQH/9X82qKKK6znCiiigAooooHcKKKKAuFFFFAXCiiigoKKKKACiiig
AooooAKKKKACiiigAooooAKKKKACiiigAooooAfRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUGIUUUUAFFFFABRRRQA4d
KKB0opssKKKKQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUALk0ZNJRTuAuTRk0lFFwFyaMmkoouAuTRk0lFFwFyaMmk
oouA+iiikAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAC5NGTSUU7gOHSlpB0paQBRRRQAUUUUAFFFFABRRRQAUUUUAF
FFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFO4BRRRSAKKKKbAKKKKQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFF
FABRRRQAUUUUAFFFFACgkUbjSUUALuNG40lFO4H/1vzaooorrOcKKKKACiiigAooooAKKKKACiiigdwooooC4UUUUBcKKKKCgooo
oAKKKKACiiigAooooAKKKKACiiigBcmjJpKKAFyaMmkooAfRRRQA0kg0ZNB60lAD6KSloAKKKKACiiigAooooAKKKKBWCiiigyCi
iigAyaMmiincBw6UUDpRSLCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAcB3owKWigBMCjApaKdw
CiiikAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAuTRk0lFO4C5NGTSUUXAcOlLSDpS0gCiiigAooooAKKKKAC
iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKdwCiiihgFFFFIAooooAKKKKbAKKKKQBRRRQAUUUUAFFFFABRRRQAUU
UUAFFFFABRRRQAUUUUAFFFFABRRRQB//1/zaooorrOcKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigsKKKKACiiigAooo
oAKKKKACiiigAooooAKKKKACiiigB9FFFADT1pKU9aSgB1LSUtABRRRQAUUUUAFFFFABRRRQAUUUUGTCiiigQoAxS4FA6UUNlJCE
9qTJoPWincTY4dKKB0opFCEnNJk0HrRQS2OHSigdKKCgooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAH0UUU2AhOKTJpW
602kA+iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAHDpS0g6UtNgFFFFIAooooAKKKKACiii
gAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiincAoooouAUUUUgCiiincAooooYBRRRSAKKKKACiiigAooooAKKKK
ACiiigAooooAKKKKACiiigAooooA/9D82qKKK6znCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooLCiiigAooooAKKKKAC
iiigAooooAKKKKACiiigAooooAXJoyaSigB3Bo4opaAGk9qMmg9aSgB9FJS0AFFFFABRRRQAUUUUAFFFFBkwooooEGTRk0UU7gFF
FFIBw6UUDpRTZYYFGBRRSAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAXJoyaSincAzmiiikA+iiimwCiiikAU
UUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAOHSlpB0pabAKKKKQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFA
BRRRQAUUUUAFFFFABRRRQAUUUUAFFFFO4BRRRQwCiiikAUUUU7gFFFFIAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAC
iiigAooooA//0fzaooorrOcKKKKACiiigAooooAKKKKACiiigAooooAKKKKAEJwKAcig9DSL0oLHUUUUCvcKKKKBhRRRQAUUUUAF
FFFABRRRQA7AowKbRQK4p60lFFAwooooAXJoyaSigB2M80YFFLQAUUUUAFFFFABRRRQAUUUUAFFFFBkwooooEFFFFABRRRQAZNGT
RRTuAZNGTRRRcdxw6UUDpRSKCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigB9FFFNgFFFFIAooooAKKKKA
CiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAXJoyaSincBw6UtIOlLSAKKKKACiiigAooooAKKKKACiiigAooooAKKKKA
CiiigAooooAKKKKACiiigAooop3AKKKKQBRRRTuAUUUUMAooopAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAF
FFFAH//S/Nqiiius5wpD0paQ9KBobRRRUFBRRRQA+iiirICiiigAooooAKKKKACgDsKKKACiiigAooooLCiiigAooooAKKKKACii
igAooooAKKKKACiiigAooooAXJoyaSigBcmjJpKKAH0UlLQAUUUUAFFFFABRRRQKwUUUUBYKKKKAsFFFFAWCiiigyCiiigAyaMmi
incBw6UUDpRSLEJOaTJoPWiglscDmikFLQUFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAC5NGTSUU7gLk0ZNJRRcB9FFFIAoo
ooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAXJoyaSincBw6UtIOlLSAaSc0ZNB60lADh0paQdKWgAooooAKK
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
vmP+N9bs9KHz8n8AAAD//wMAUEsDBBQABgAIAAAAIQDvfVxakREAAFMRAgASAAAAd29yZC9udW1iZXJpbmcueG1s7J3dbuO4Fcfv
C/QdggC93DG/RFLBzi5sy2632F0U2Cl67djKxFh/wXYyk14V+yi96SP0ffoC+wolJVuxY5kjUbJHTv4344wlHZN/8pDndySR337/
eTq5eoyXq/F89v6aviPXV/FsOB+NZx/fX//9Q/8bfX21Wg9mo8FkPovfXz/Fq+vvv/vjH779dDN7mN7GS3PilbExW918WgzfX9+v
14ubVms1vI+ng9W76Xi4nK/md+t3w/m0Nb+7Gw/j1qf5ctRihJLkr8VyPoxXK2OnO5g9DlbXG3PDz8WsjZaDT+Zia1C0hveD5Tr+
/GyDljYStMKWPjTEPAyZGjJ6aIqXNiVbtlQHhoSXIVOqA0uBn6Wcykk/S+zQkvKzxA8taT9LB91petjB54t4Zg7ezZfTwdr8d/mx
NR0sf31YfGMMLwbr8e14Ml4/GZtEbs0MxrNfPUpkrsosTPmotAXVms5H8YSPtlbm768flrObzfXfZNfbot+k128+siviSbGfNT8X
tuLP68lqvb12WUS79PJoPnyYxrN1olprGU+MjvPZ6n68yEaHqa81c/B+a+TRJcDjdLI979OCFnS1Y0NblDbDs8Eixd+03XSSltxt
kZICrWlNZFcUKcL+b25LMjU9+PmHvaTZEZcWHHy2BtiBATmMC04WWxt6Y6M1fPZua2dc0K22dtJWsXbGz8LSgmPgy8LsGBg9lDLB
+LYc9sNevmNrNVqP7suZ27ZRy147WA/uB6vMaVKLdwUHgq1FsWMx7WCT+TAbz6zNuJxoQWbwabrThouP1Rz1z8v5w+LZ2riatR+e
h+xPNnoqYWvj8LuD0KpaYX65HyzMSD4d3vzwcTZfDm4npkTGfa+MB14lLWD/NR3ZfiR/xp+T723/2fxxN7F/jB6u7JB4/Z2JAge3
q/VyMFz//DC92vvfD8aVTDRpjN8sYxNCLu2XacDYvlvHy84yHvxqT7FWZiv7szePA9OtTMk1l732dcsemT5M1uMf48d48uFpEW/P
Sb6d2G/Ts9bTxWR7rNcjpr+pbnpk8mgPjM3H9reSsmxPpulZJpbtT7Mvbx8mk3idXf/BTGTbQ7//9t/s+78Ot99O4rvN6Yu/LZPy
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"""

# =========================================================
# HELP TEXT (USER)
# =========================================================

HELP_TEXT = """\
🚀 PulseFutures — Trading System in Telegram

PulseFutures is NOT a signal spam bot.
It’s a full trading assistant that helps you trade with discipline.

────────────────────
🔍 Core Commands
────────────────────
/screen
• Scan the market for high-quality setups

/size <symbol> <entry> <sl>
• Position sizing based on your risk rules

/status
• Your plan, trial status & enabled features

/commands
• Full command guide + examples

/guide_full
• Download the full user guide (DOCX)

────────────────────
⏰ Timezone (Email + Headers)
────────────────────
/tz <Region/City>
• Sets your local timezone so emails show your local time
• Examples:
  - Australia (Melbourne): /tz Australia/Melbourne
  - UAE (Dubai): /tz Asia/Dubai
  - UK (London): /tz Europe/London
  - USA (New York): /tz America/New_York

────────────────────
⚠️ Alerts & Context
────────────────────
/bigmove_alert on|off
• Major market moves (📧 Pro/Trial only)

• Possible reversal zones (📧 Pro/Trial only)

────────────────────
💎 Plans
────────────────────
🟢 Standard — Telegram only
🔵 Pro — Telegram + Email alerts
🎁 New users get a 7-day Pro trial automatically.

🤖 Bot: @PulseFuturesBot
📢 Updates: @PulseFutures
🆘 Support: @PulseFuturesSupport
"""\

COMMANDS_TEXT = """\
📘 PulseFutures — Command Guide & Examples

PulseFutures is a full trading system inside Telegram.
Below are the key commands with simple examples.

────────────────────
🔍 MARKET SCAN
────────────────────
/screen
• Scans the market for high-quality setups
• Sections you may see:
  - Top Trade Setups (ready)
  - Waiting for Trigger (near-miss)
  - Trend Continuation Watch
  - Spike Reversal Alerts
  - Leaders/Losers + Market Leaders

Example:
/screen

────────────────────
⚖️ RISK & POSITION SIZING
────────────────────
/size <symbol> <side> <entry> <sl>
• Calculates position size based on your risk rules

Examples:
/size BTC long 42000 41000
/size ELSA short 0.09087 0.09671

────────────────────
🕒 SESSION CONTROL
────────────────────
/sessions
• View your session settings

/sessions_on <ASIA|LON|NY>
/sessions_off <ASIA|LON|NY>
• Enable/disable sessions

/sessions_on_unlimited
/sessions_off_unlimited
• 24-hour mode for scans (if enabled in your build)

Example:
/sessions_on NY

────────────────────
⏰ TIMEZONE (LOCAL TIME IN EMAILS)
────────────────────
/tz
• Show your current timezone

/tz <Region/City>
• Set your timezone so headers + emails show your local time
• Use IANA format: Region/City

Examples:
/tz Australia/Melbourne
/tz Asia/Dubai   (UAE)
/tz Europe/London
/tz America/New_York

────────────────────
⚠️ ALERTS & EMAILS
────────────────────
/bigmove_alert on|off [4H%] [1H%]
• Big move alerts in either direction (UP or DOWN)
• 📧 Email alerts are Pro/Trial only

• Possible reversal zones (context, not an entry)
• 📧 Email alerts are Pro/Trial only

/email you@gmail.com
• Set your email for alerts

/email_test
• Send a test email to confirm delivery

/email off
• Disable email

Examples:
/bigmove_alert on 30 12
/email you@example.com

────────────────────
📊 PLAN & STATUS
────────────────────
/status
• Shows your plan (Trial/Standard/Pro), trial days remaining, and enabled features

────────────────────
🆘 HELP & SUPPORT
────────────────────
/help
• Quick overview

/commands
• Full guide (this)

/guide_full
• Download the full user guide (DOCX)

Support: @PulseFuturesSupport
Updates: @PulseFutures
"""\



# =========================================================
# HELP TEXT (ADMIN)
# =========================================================

HELP_TEXT_ADMIN = """\
🛠 PulseFutures — Admin Command Guide

Admin commands are powerful. Use carefully.
Not financial advice.

────────────────────
👤 USERS & ACCESS
────────────────────
/admin_user <user_id>
• View full user record (plan, trial, alerts)

/admin_users
• List users (overview)

/admin_grant <user_id> <standard|pro>
• Grant or change user plan

/admin_revoke <user_id>
• Revoke paid access (sets to standard)

/myplan
• View your own plan status (admins too)

────────────────────
💳 PAYMENTS (USDT)
────────────────────
/admin_payments
• View payments ledger

/usdt_pending
• Show pending USDT requests

/usdt_approve <TXID>
• Approve payment (grants access + writes ledger)

/usdt_reject <TXID> <reason>
• Reject payment

────────────────────
⏱️ COOLDOWNS
────────────────────
/cooldown
/cooldowns
• View cooldowns

/cooldown_clear <SYMBOL> <long|short>
• Clear cooldown for one symbol + side

/cooldown_clear_all
• Clear all cooldowns (global)

────────────────────
⚙️ DATA / RECOVERY
────────────────────
/reset
• Reset user data / clean DB (⚠️ DANGEROUS)

/restore
• Restore previously removed data (if backup exists)

────────────────────
🧪 DIAGNOSTICS
────────────────────
/why
• Last /screen reject summary (why no setups)

/email_decision
• Last email decision (why email sent/skipped/error)

/health_sys
• System health (DB, exchange, email)

────────────────────
📢 Channels
────────────────────
Updates: @PulseFutures
Support: @PulseFuturesSupport
"""\


# =========================================================
# PERFORMANCE HELPERS
# =========================================================

def _run_coro_in_thread(coro):
    """Run an async coroutine to completion in a worker thread."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # Fallback for edge cases where a loop is already running in that thread
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.close()
            except Exception:
                pass

# =========================================================
# BILLING COMMANDS (Stripe Payment Links + USDT)
# =========================================================

def _env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return (v or default).strip()

def _mask_addr(addr: str) -> str:
    a = (addr or "").strip()
    if len(a) <= 14:
        return a
    return f"{a[:6]}…{a[-6:]}"

async def billing_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Billing menu: Stripe Payment Links + USDT (MANUAL activation for Stripe).
    - Safe: uses env vars only for Stripe links (no Stripe API calls required)
    - USDT address works with either USDT_ADDRESS or USDT_RECEIVE_ADDRESS
    - Does NOT require email to show billing
    """
    user = update.effective_user
    uid = user.id if user else None

    # Stripe payment links
    stripe_standard_url = _env("STRIPE_STANDARD_URL")
    stripe_pro_url = _env("STRIPE_PRO_URL")

    # USDT config (support BOTH possible env names)
    usdt_network = _env("USDT_NETWORK", "TRC20")
    usdt_address = _env("USDT_ADDRESS") or _env("USDT_RECEIVE_ADDRESS")
    usdt_note = _env("USDT_NOTE")  # optional

    # Prices (display only)
    usdt_standard_price = _env("USDT_STANDARD_PRICE", "45")
    usdt_pro_price = _env("USDT_PRO_PRICE", "99")

    # Support
    support_handle = _env("BILLING_SUPPORT_HANDLE", "@PulseFuturesSupport")

    # Reference (helps you match tickets/payments)
    ref = f"PF-{uid}" if uid else "PF-UNKNOWN"

    lines = []
    lines.append("💳 PulseFutures — Billing & Upgrade")
    lines.append("")
    lines.append("Choose your payment method below.")
    lines.append(f"Reference (important): {ref}")
    lines.append("")
    lines.append("────────────────────")
    lines.append("1) Stripe (Card / Apple Pay / Google Pay)")
    lines.append("────────────────────")
    if stripe_standard_url or stripe_pro_url:
        lines.append("Tap a plan button to pay securely via Stripe.")
        lines.append("✅ Activation is MANUAL for now (fast). After paying, send:")
        lines.append(f"• Reference: {ref}")
        lines.append(f"• Telegram ID: {uid if uid else 'unknown'}")
        lines.append("• Plan: Standard or Pro")
        lines.append("• Stripe email used")
        lines.append(f"Support: {support_handle}")
    else:
        lines.append("Stripe is not configured yet.")
        lines.append("Admin: set STRIPE_STANDARD_URL / STRIPE_PRO_URL in Render env vars.")

    lines.append("")
    lines.append("────────────────────")
    lines.append("2) USDT")
    lines.append("────────────────────")
    if usdt_address:
        lines.append(f"Network: USDT ({usdt_network})")
        lines.append(f"Address: {usdt_address}")
        lines.append(f"(Short: {_mask_addr(usdt_address)})")
        lines.append(f"Prices: Standard {usdt_standard_price} USDT • Pro {usdt_pro_price} USDT")
        if usdt_note:
            lines.append(f"Note: {usdt_note.replace('<REF>', ref)}")
        else:
            lines.append(f"Note: Use reference '{ref}' in your message to support if needed.")
        lines.append("After paying, submit:")
        lines.append("/usdt_paid <TXID> <standard|pro>")
    else:
        lines.append("USDT is not configured yet.")
        lines.append("Admin: set USDT_ADDRESS or USDT_RECEIVE_ADDRESS in Render env vars.")
        lines.append("Optional: set USDT_NETWORK (default TRC20).")

    lines.append("")
    lines.append("────────────────────")
    lines.append("After payment (Activation)")
    lines.append("────────────────────")
    lines.append("✅ Stripe:")
    lines.append(f"Send to Support: {support_handle}")
    lines.append(f"1) Reference: {ref}")
    lines.append(f"2) Telegram ID: {uid if uid else 'unknown'}")
    lines.append("3) Plan: Standard or Pro")
    lines.append("4) Stripe email used")
    lines.append("")
    lines.append("✅ USDT:")
    lines.append("Submit: /usdt_paid <TXID> <standard|pro>")
    lines.append(f"Support: {support_handle}")
    lines.append(f"Reference: {ref}")

    msg = "\n".join(lines)

    # Buttons (Stripe)
    buttons = []
    stripe_row = []
    if stripe_standard_url:
        stripe_row.append(InlineKeyboardButton("✅ Stripe — Standard", url=stripe_standard_url))
    if stripe_pro_url:
        stripe_row.append(InlineKeyboardButton("🚀 Stripe — Pro", url=stripe_pro_url))
    if stripe_row:
        buttons.append(stripe_row)

    howto_url = _env("BILLING_HOWTO_URL")
    if howto_url:
        buttons.append([InlineKeyboardButton("ℹ️ Payment Instructions", url=howto_url)])

    reply_markup = InlineKeyboardMarkup(buttons) if buttons else None

    await send_long_message(
        update,
        msg,
        parse_mode=None,
        disable_web_page_preview=True,
        reply_markup=reply_markup,
    )

# =========================================================
# Table Format
# =========================================================

def _format_help_table(rows, col_cmd=16, col_what=46, col_ex=28) -> str:
    """
    Monospace table with wrapping; optional Example column.
    rows: [{"cmd": str, "what": str, "ex": str|""}, ...]
    - If ex == cmd -> omit example.
    - If no examples in a section -> omit Example column entirely.
    """
    clean_rows = []
    any_ex = False

    for r in rows:
        cmd = (r.get("cmd") or "").strip()
        what = (r.get("what") or "").strip()
        ex = (r.get("ex") or "").strip()

        if ex == cmd:
            ex = ""
        if ex:
            any_ex = True

        clean_rows.append({"cmd": cmd, "what": what, "ex": ex})

    if any_ex:
        header = f"{'Command':<{col_cmd}}  {'What it does':<{col_what}}  {'Example':<{col_ex}}"
        sep    = f"{'-'*col_cmd}  {'-'*col_what}  {'-'*col_ex}"
    else:
        header = f"{'Command':<{col_cmd}}  {'What it does':<{col_what}}"
        sep    = f"{'-'*col_cmd}  {'-'*col_what}"

    lines = [header, sep]

    for r in clean_rows:
        cmd = r["cmd"][:col_cmd]
        what_lines = textwrap.wrap(r["what"], width=col_what) or [""]
        if any_ex:
            ex_lines = textwrap.wrap(r["ex"], width=col_ex) or [""]
            n = max(len(what_lines), len(ex_lines))
            for i in range(n):
                c = cmd if i == 0 else ""
                w = what_lines[i] if i < len(what_lines) else ""
                e = ex_lines[i] if i < len(ex_lines) else ""
                lines.append(f"{c:<{col_cmd}}  {w:<{col_what}}  {e:<{col_ex}}")
        else:
            for i, w in enumerate(what_lines):
                c = cmd if i == 0 else ""
                lines.append(f"{c:<{col_cmd}}  {w:<{col_what}}")

    return "\n".join(lines)


def build_help_html(title: str, sections: list) -> str:
    """
    sections: [{"name": str, "rows": [...]}, ...]
    Produces HTML message with <pre> monospace tables.
    """
    parts = [f"📘 <b>{html.escape(title)}</b>"]
    for sec in sections:
        parts.append(f"\n<b>• {html.escape(sec['name'])}</b>")
        table = _format_help_table(sec["rows"])
        parts.append(f"<pre>{html.escape(table)}</pre>")
    return "\n".join(parts)


# =========================================================
# TELEGRAM COMMANDS
# =========================================================
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Plain text help (no tables, no HTML builder)
    await send_long_message(
        update,
        HELP_TEXT,
        parse_mode=None,
        disable_web_page_preview=True,
    )

async def commands_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Full command guide with examples
    await send_long_message(
        update,
        COMMANDS_TEXT,
        parse_mode=None,
        disable_web_page_preview=True,
    )




async def guide_full_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Sends the full PulseFutures User Guide as a PDF document in Telegram.
    """
    try:
        # The PDF should be committed to the repo alongside the bot code.
        pdf_name = "PulseFutures_User_Guide.pdf"
        pdf_path = pdf_name
        if not os.path.exists(pdf_path):
            # Fallback to the directory of this script (Render runs from /opt/render/project/src)
            try:
                pdf_path = os.path.join(os.path.dirname(__file__), pdf_name)
            except Exception:
                pdf_path = pdf_name

        if not os.path.exists(pdf_path):
            await update.message.reply_text("❌ PDF guide file is not available right now.")
            return

        caption = "📘 PulseFutures — Full User Guide (PDF)"
        with open(pdf_path, "rb") as fh:
            await context.bot.send_document(
                chat_id=update.effective_chat.id,
                document=fh,
                filename=pdf_name,
                caption=caption,
            )
    except Exception:
        await update.message.reply_text("❌ Could not send the guide. Please try again.")

async def cmd_help_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("Admin only.")
        return

    await send_long_message(
        update,
        HELP_TEXT_ADMIN,
        parse_mode=None,
        disable_web_page_preview=True,
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid) or {}

    # Start 7-day FULL Pro trial on first /start only
    ensure_trial_started(uid, user, force=True)

    await cmd_help(update, context)

# =========================================================
# SUPPORT SYSTEM
# =========================================================
async def support_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args:
        await update.message.reply_text(
            "Usage:\n/support <your issue>"
        )
        return

    issue = " ".join(context.args)
    ticket_id = f"TKT-{uid}-{int(time.time())}"

    msg = (
        f"🆘 Support Ticket {ticket_id}\n\n"
        f"User: {uid}\n"
        f"Message:\n{issue}"
    )

    for admin in ADMIN_IDS:
        await context.bot.send_message(admin, msg)

    await update.message.reply_text(
        f"✅ Ticket created: {ticket_id}\n"
        "Use /support_status to check."
    )


async def support_status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📨 Your latest support ticket is being reviewed.\n"
        "Resolved tickets are auto-closed."
    )

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
        await update.message.reply_text(
            f"Your TZ: {user['tz']}\n\n"
            "Set your timezone (IANA format Region/City):\n"
            "• /tz Australia/Melbourne\n"
            "• /tz Asia/Dubai (UAE)\n"
            "• /tz Europe/London\n"
            "• /tz America/New_York"
        )
        return
    tz_name = " ".join(context.args).strip()
    try:
        ZoneInfo(tz_name)
    except Exception:
        await update.message.reply_text(
            "Invalid timezone. Use Region/City, for example:\n"
            "• /tz Australia/Melbourne\n"
            "• /tz Asia/Dubai (UAE)\n"
            "• /tz Europe/London\n"
            "• /tz America/New_York"
        )
        return
    sessions = ['NY']
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
        await update.message.reply_text("Usage: /sessions_off ASIA")
        return

    name = context.args[0].strip().upper()
    if name not in SESSIONS_UTC:
        await update.message.reply_text("Session must be one of: ASIA, LON, NY")
        return

    enabled = [s for s in user_enabled_sessions(user) if s != name]

    # Never allow "no sessions" — fall back to sensible defaults
    if not enabled:
        enabled = _default_sessions_for_tz(user["tz"])
        enabled = _order_sessions(enabled) or enabled

    update_user(uid, sessions_enabled=json.dumps(enabled))
    await update.message.reply_text(
        f"✅ Disabled: {name}\n"
        f"Enabled sessions: {', '.join(enabled)}"
    )


async def sessions_on_unlimited_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    update_user(uid, sessions_unlimited=1)
    await update.message.reply_text("✅ Sessions: UNLIMITED (24h emailing enabled).")

async def sessions_off_unlimited_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    update_user(uid, sessions_unlimited=0)
    await update.message.reply_text("✅ Sessions: back to normal (enabled sessions only).")

async def sessions_unlimited_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Alias: /sessions_unlimited_on
    await sessions_on_unlimited_cmd(update, context)

async def sessions_unlimited_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Alias: /sessions_unlimited_off
    await sessions_off_unlimited_cmd(update, context)

async def bigmove_alert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)

    if not has_active_access(user, uid):
        await update.message.reply_text(
            "⛔️ Trial finished.\n\n"
            "Your 7-day trial is over — you need to pay to keep using PulseFutures.\n\n"
            "👉 /billing"
        )
        return   
        
    if not context.args:
        on = int(user.get("bigmove_alert_on", 1) or 0)
        p4 = float(user.get("bigmove_alert_4h", 20) or 20)
        p1 = float(user.get("bigmove_alert_1h", 10) or 10)
        await update.message.reply_text(
            "📣 Big-Move Alert Emails\n"
            f"{HDR}\n"
            f"Status: {'ON' if on else 'OFF'}\n"
            f"Thresholds: |4H| ≥ {p4:.0f}% OR |1H| ≥ {p1:.0f}% (both directions)\n\n"
            "Set: /bigmove_alert on 20 10\n"
            "Off: /bigmove_alert off"
        )
        return

    mode = context.args[0].strip().lower()

    if mode in {"off", "0", "disable"}:
        update_user(uid, bigmove_alert_on=0)
        await update.message.reply_text("✅ Big-move alert emails: OFF")
        return

    if mode in {"on", "1", "enable"}:
        p4 = 20.0
        p1 = 10.0
        if len(context.args) >= 3:
            try:
                p4 = float(context.args[1])
                p1 = float(context.args[2])
            except Exception:
                await update.message.reply_text("Usage: /bigmove_alert on <4H%> <1H%>  (e.g., /bigmove_alert on 20 10)")
                return
        update_user(uid, bigmove_alert_on=1, bigmove_alert_4h=p4, bigmove_alert_1h=p1)
        await update.message.reply_text(f"✅ Big-move alert emails: ON (4H≥{p4:.0f}% OR 1H≥{p1:.0f}%)")
        return

    await update.message.reply_text("Usage: /bigmove_alert on <4H%> <1H%>  (e.g., /bigmove_alert on 20 10)  OR  /bigmove_alert off")


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


def _find_unknown_tokens(tokens: list) -> list:
    """
    Very strict validation:
    - allowed keywords: symbol, side, sl, entry, risk, usd, pct
    - side values: long/short/buy/sell
    - everything else should be numeric (like prices/values)
    Any unknown non-numeric token => error.
    """
    allowed = {
        "sl", "stop",
        "entry", "ent",
        "risk",
        "usd", "pct",
        "long", "short", "buy", "sell",
    }

    unknown = []
    for t in tokens:
        tt = str(t).strip().lower()
        if not tt:
            continue
        if tt in allowed:
            continue
        # numeric is OK
        try:
            float(tt)
            continue
        except Exception:
            unknown.append(tt)
    return unknown

def _typo_hint_for_token(tok: str) -> str:
    t = (tok or "").lower()
    # common "entry" typos
    if t.startswith("entr") and t not in ("entry", "ent"):
        return "Did you mean `entry`?"
    if t in ("enrty", "etry", "etryy", "enter", "entery", "enrt"):
        return "Did you mean `entry`?"
    return ""


async def size_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)

    if not has_active_access(user, uid):
        await update.message.reply_text(
            "⛔️ Trial finished.\n\n"
            "Your 7-day trial is over — you need to pay to keep using PulseFutures.\n\n"
            "👉 /billing"
        )
        return   
        
    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text("Usage: /size BTC long sl 42000  (optional: risk pct 2 | risk usd 40 | entry 43000)")
        return

    tokens = raw.split()
    
    # -------------------------------------------------
    # REQUIRED POSITIONAL ARGS
    # -------------------------------------------------
    if len(tokens) < 4:
        await update.message.reply_text(
            "Usage:\n"
            "/size <SYMBOL> <long|short> entry <PRICE> sl <STOP>\n"
            "Optional: risk <usd|pct> <VALUE> (default: 1.5%)"
        )
        return
    
    symbol = re.sub(r"[^A-Za-z0-9]", "", tokens[0]).upper()
    direction = tokens[1].lower()
    
    if direction not in ("long", "short"):
        await update.message.reply_text("Second argument must be long or short.")
        return
    
    # Strip positional args BEFORE keyword parsing
    tokens = tokens[2:]

    # STRICT: reject unknown tokens (prevents silent fallback to live price on typos like "entrt")
    allowed = {"sl", "entry", "risk", "usd", "pct"}
    unknown = []
    for t in tokens:
        tt = str(t).strip().lower()
        if not tt:
            continue
        if tt in allowed:
            continue
        # numeric token is OK
        try:
            float(tt)
            continue
        except Exception:
            unknown.append(tt)

    if unknown:
        hint = ""
        u0 = unknown[0]
        if u0.startswith("entr") and u0 != "entry":
            hint = "\nDid you mean `entry`?"
        await update.message.reply_text(
            "❌ Invalid /size syntax.\n\n"
            f"Unknown keyword(s): {', '.join(unknown)}{hint}\n\n"
            "Use:\n/size <SYMBOL> <long|short> entry <ENTRY> sl <STOP> [risk <usd|pct> <VALUE>]\n"
            "Example:\n/size BTC long entry 43000 sl 42000 risk usd 40"
        )
        return

    # -------------------------------------------------
    # KEYWORD PARSING (entry + sl REQUIRED)
    # -------------------------------------------------
    side = "BUY" if direction == "long" else "SELL"
    sym = symbol  # <- use the original parsed symbol (positional)

    entry = None
    sl = None

    risk_mode = "PCT"
    risk_val = 1.5  # <- DEFAULT: 1.5% of equity if user does not provide risk

    i = 0
    while i < len(tokens):
        t = tokens[i].lower()

        if t == "entry" and i + 1 < len(tokens):
            try:
                entry = float(tokens[i + 1])
            except Exception:
                await update.message.reply_text("Bad entry format. Example: entry 43000")
                return
            i += 2
            continue

        if t == "sl" and i + 1 < len(tokens):
            try:
                sl = float(tokens[i + 1])
            except Exception:
                await update.message.reply_text("Bad SL format. Example: sl 42000")
                return
            i += 2
            continue

        if t == "risk" and i + 2 < len(tokens):
            rm = tokens[i + 1].lower()
            if rm not in ("usd", "pct"):
                await update.message.reply_text("Bad risk format. Use: risk pct 2  OR  risk usd 40")
                return
            try:
                rv = float(tokens[i + 2])
            except Exception:
                await update.message.reply_text("Bad risk format. Use: risk pct 2  OR  risk usd 40")
                return
            risk_mode = rm.upper()
            risk_val = rv
            i += 3
            continue

        await update.message.reply_text(f"❌ Invalid keyword: {tokens[i]}")
        return

    # entry + sl are mandatory (as you requested)
    if entry is None or sl is None:
        await update.message.reply_text(
            "❌ Missing entry or SL.\n"
            "Use:\n/size <SYMBOL> <long|short> entry <ENTRY> sl <STOP>\n"
            "Example:\n/size EUL long entry 2.46 sl 2.26"
        )
        return

    # --- Entry sanity check vs live price (warn only; does not block /size) ---
    try:
        best_now = await asyncio.to_thread(fetch_futures_tickers)
        mv_now = best_now.get(sym)
        cur_px = float(mv_now.last) if mv_now and float(mv_now.last or 0) > 0 else 0.0
    except Exception:
        cur_px = 0.0
    
    entry_warn = ""
    if cur_px > 0:
        w, severe = entry_sanity_check(entry, cur_px)
        if w:
            # /size: warn only (no blocking), but make severe warnings very visible
            entry_warn = ("\n" + w + "\n")
    # --------------------------------------------------------------

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
    if entry_warn:
        msg += entry_warn
    msg += f"\nChart: {tv_chart_url(sym)}"

    await update.message.reply_text(msg)

async def trade_open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = reset_daily_if_needed(get_user(uid))


    if not has_active_access(user, uid):
        await update.message.reply_text(
            "⛔️ Trial finished.\n\n"
            "Your 7-day trial is over — you need to pay to keep using PulseFutures.\n\n"
            "👉 /billing"
        )
        return   
        
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

    # =========================================================
    # ENTRY PRICE SANITY CHECK (BLOCK if severe unless "force")
    # =========================================================
    force = ("force" in tokens)  # user can add: force
    
    try:
        best_now = await asyncio.to_thread(fetch_futures_tickers)
        mv_now = best_now.get(sym)
        cur_px = float(mv_now.last) if mv_now and float(mv_now.last or 0) > 0 else 0.0
    except Exception:
        cur_px = 0.0
    
    if cur_px > 0:
        w, severe = entry_sanity_check(entry, cur_px)
        if w and severe and not force:
            await update.message.reply_text(
                w
                + "\n\n❌ Trade NOT opened to protect you from a bad entry.\n"
                  "If you are 100% sure, re-run the command and add: force"
            )
            return
        elif w and not severe:
            await update.message.reply_text(w)
    # =========================================================
    
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

    if not has_active_access(user, uid):
        await update.message.reply_text(
            "⛔️ Trial finished.\n\n"
            "Your 7-day trial is over — you need to pay to keep using PulseFutures.\n\n"
            "👉 /billing"
        )
        return   
    
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

    # Ensure daily reset happens before any risk math
    user = reset_daily_if_needed(get_user(uid))

    if not context.args:
        await update.message.reply_text("Usage: /trade_rf <TRADE_ID>")
        return

    try:
        trade_id = int(context.args[0])
    except Exception:
        await update.message.reply_text("Usage: /trade_rf <TRADE_ID>")
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
    old_risk = float(t["risk_usd"])

    # If already risk-free, don't release again
    if old_risk <= 1e-9:
        con.close()
        cap = daily_cap_usd(user)
        day_local = _user_day_local(user)
        used_today = _risk_daily_get(uid, day_local)
        remaining_today = (cap - used_today) if cap > 0 else float("inf")

        await update.message.reply_text(
            f"✅ Trade is already Risk-Free\n"
            f"- Trade ID: {trade_id}\n"
            f"- SL is already at Entry\n\n"
            f"📌 Daily Risk\n"
            f"- Cap: ≈ ${cap:.2f}\n"
            f"- Used today: ${used_today:.2f}\n"
            f"- Remaining today: ${max(0.0, remaining_today):.2f}" if cap > 0 else
            f"📌 Daily Risk\n"
            f"- Cap: ≈ ${cap:.2f}\n"
            f"- Used today: ${used_today:.2f}\n"
            f"- Remaining today: ∞"
        )
        return

    # Move SL to entry and set trade risk to 0
    cur.execute(
        "UPDATE trades SET sl=?, risk_usd=? WHERE id=? AND user_id=?",
        (entry, 0.0, trade_id, uid),
    )
    con.commit()
    con.close()

    # Release today's risk immediately
    day_local = _user_day_local(user)
    _risk_daily_inc(uid, day_local, -old_risk)

    # Clamp used risk to 0 if it went negative (edge cases)
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
        used_now = 0.0

    cap = daily_cap_usd(user)
    remaining_today = (cap - used_now) if cap > 0 else float("inf")

    await update.message.reply_text(
        f"✅ Trade Risk-Free\n"
        f"- Trade ID: {trade_id}\n"
        f"- SL moved to Entry\n"
        f"- Released Risk: ${old_risk:.2f}\n\n"
        f"📌 Daily Risk (updated)\n"
        f"- Cap: ≈ ${cap:.2f}\n"
        f"- Used today: ${used_now:.2f}\n"
        f"- Remaining today: ${max(0.0, remaining_today):.2f}" if cap > 0 else
        f"📌 Daily Risk (updated)\n"
        f"- Cap: ≈ ${cap:.2f}\n"
        f"- Used today: ${used_now:.2f}\n"
        f"- Remaining today: ∞"
    )


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = reset_daily_if_needed(get_user(uid))

    if not has_active_access(user, uid):
        await update.message.reply_text(
            "⛔️ Trial finished.\n\n"
            "Your 7-day trial is over — you need to pay to keep using PulseFutures.\n\n"
            "👉 /billing"
        )
        return

    opens = db_open_trades(uid)

    plan = str(effective_plan(uid, user)).upper()
    equity = float((user or {}).get("equity") or 0.0)

    cap = daily_cap_usd(user)
    day_local = _user_day_local(user)
    used_today = _risk_daily_get(uid, day_local)
    remaining_today = (cap - used_today) if cap > 0 else float("inf")

    enabled = user_enabled_sessions(user)
    now_s = in_session_now(user)
    now_txt = now_s["name"] if now_s else "NONE"

    # Email caps (show infinite as 0=∞ to match UI)
    cap_sess = int((user or {}).get("max_emails_per_session", DEFAULT_MAX_EMAILS_PER_SESSION) or DEFAULT_MAX_EMAILS_PER_SESSION)
    cap_day = int((user or {}).get("max_emails_per_day", DEFAULT_MAX_EMAILS_PER_DAY) or DEFAULT_MAX_EMAILS_PER_DAY)
    gap_m = int((user or {}).get("email_gap_min", DEFAULT_EMAIL_GAP_MIN) or DEFAULT_EMAIL_GAP_MIN)

    # Big-move status
    bm_on = int((user or {}).get("bigmove_alert_on", 1) or 0)
    bm_4h = float((user or {}).get("bigmove_alert_4h", 20) or 20)
    bm_1h = float((user or {}).get("bigmove_alert_1h", 10) or 10)

    lines = []
    lines.append("📌 Status")
    lines.append(f"Plan: {plan}")
    lines.append(f"Equity: ${equity:.2f}")
    lines.append(f"Trades today: {int(user.get('day_trade_count',0))}/{int(user.get('max_trades_day',0))}")
    lines.append(f"Daily cap: {user.get('daily_cap_mode','PCT')} {float(user.get('daily_cap_value',0.0)):.2f} (≈ ${cap:.2f})")
    lines.append(f"Daily risk used: ${used_today:.2f}")
    lines.append(f"Daily risk remaining: ${max(0.0, remaining_today):.2f}" if cap > 0 else "Daily risk remaining: ∞")
    lines.append(f"Email alerts: {'ON' if int(user.get('notify_on',1))==1 else 'OFF'}")
    lines.append(f"Sessions enabled: {' | '.join(enabled)} | Now: {now_txt}")
    lines.append(f"Email caps: session={cap_sess} (0=∞), day={cap_day} (0=∞), gap={gap_m}m")
    lines.append(f"Big-move alert emails: {'ON' if bm_on else 'OFF'} (4H≥{bm_4h:.0f}% OR 1H≥{bm_1h:.0f}%)")
    lines.append(HDR)

    if not opens:
        lines.append("Open trades: None")
        await update.message.reply_text("\n".join(lines))
        return

    lines.append("Open trades:")
    for t in opens:
        try:
            entry = float(t.get("entry") or 0.0)
            sl = float(t.get("sl") or 0.0)
            qty = float(t.get("qty") or 0.0)
            risk = float(t.get("risk_usd") or 0.0)
            lines.append(
                f"- ID {t.get('id')} | {t.get('symbol')} {t.get('side')} | "
                f"Entry {fmt_price(entry)} | SL {fmt_price(sl)} | Risk ${risk:.2f} | Qty {qty:.6g}"
            )
        except Exception:
            continue

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
    
    if not has_active_access(user, uid):
        await update.message.reply_text(
            "⛔️ Trial finished.\n\n"
            "Your 7-day trial is over — you need to pay to keep using PulseFutures.\n\n"
            "👉 /billing"
        )
        return   
    
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

    if not has_active_access(user, uid):
        await update.message.reply_text(
            "⛔️ Trial finished.\n\n"
            "Your 7-day trial is over — you need to pay to keep using PulseFutures.\n\n"
            "👉 /billing"
        )
        return    

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

    if not has_active_access(user, uid):
        await update.message.reply_text(
            "⛔️ Trial finished.\n\n"
            "Your 7-day trial is over — you need to pay to keep using PulseFutures.\n\n"
            "👉 /billing"
        )
        return   
    
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

    if not has_active_access(user, uid):
        await update.message.reply_text(
            "⛔️ Trial finished.\n\n"
            "Your 7-day trial is over — you need to pay to keep using PulseFutures.\n\n"
            "👉 /billing"
        )
        return   
    
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

    if not has_active_access(user, uid):
        await update.message.reply_text(
            "⛔️ Trial finished.\n\n"
            "Your 7-day trial is over — you need to pay to keep using PulseFutures.\n\n"
            "👉 /billing"
        )
        return   
    
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


# =====================================================================
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

            # pullback touch or near-touch
            last = closes_15[-1]
            near = abs(last - ema_fast[-1]) / last * 100.0
            if near > 1.2:
                return None

        else:
            if not (ema_fast[-1] < ema_slow[-1]):
                return None
            if ema_slope(ema_fast, 3) >= 0:
                return None

            last = closes_15[-1]
            near = abs(last - ema_fast[-1]) / last * 100.0
            if near > 1.2:
                return None

        conf = 80
        # session boost (optional)
        if session_name == "NY":
            conf += 5
        elif session_name == "LON":
            conf += 2

        return {
            "symbol": base,
            "side": side,
            "conf": int(clamp(conf, 1, 99)),
            "ch24": float(ch24),
        }

    except Exception:
        return None


# =========================================================
# PRIORITY SIGNAL POOL (USED BY /screen AND EMAIL)
# Directional Leaders/Losers → Trend Continuation Watch → Waiting for Trigger → Market Leaders
# =========================================================

def _bases_from_directional(rows, n):
    out = []
    for r in (rows or [])[:n]:
        try:
            out.append(str(r[0]).upper())
        except Exception:
            continue
    return out

def _subset_best(best_fut: dict, bases: list) -> dict:
    s = set([str(b).upper() for b in (bases or [])])
    return {k: v for k, v in (best_fut or {}).items() if str(k).upper() in s}

def _market_leader_bases(best_fut: dict, n: int) -> list: 
    
    try:
        items = [(b, mv) for b, mv in (best_fut or {}).items()]
        items = sorted(items, key=lambda x: float(getattr(x[1], "fut_vol_usd", 0.0) or 0.0), reverse=True)
        return [str(b).upper() for b, _ in items[:n]]
    except Exception:
        return []

def _norm_sym(s: str) -> str:
    s = (s or "").upper().strip()
    # common variants -> base
    s = s.replace("USDT.P", "").replace("USDT", "")
    s = s.replace("-PERP", "").replace("PERP", "")
    s = s.replace("/", "").replace("-", "").replace("_", "")
    return s

def _best_fut_vol_usd(best_fut: dict, symbol: str) -> float:
    """
    Returns futures volume USD from best_fut dict using robust symbol matching.
    Handles keys like BTC, BTCUSDT, BTCUSDT.P, BTC/USDT, etc.
    """
    if not best_fut:
        return 0.0

    sym_raw = str(symbol or "").upper().strip()
    sym_n = _norm_sym(sym_raw)

    # direct hits first
    mv = (best_fut or {}).get(sym_raw)
    if mv is None:
        mv = (best_fut or {}).get(sym_n)

    # brute search by normalized keys
    if mv is None:
        for k, v in (best_fut or {}).items():
            if _norm_sym(str(k)) == sym_n:
                mv = v
                break

    if mv is None:
        return 0.0

    try:
        return float(getattr(mv, "fut_vol_usd", 0.0) or 0.0)
    except Exception:
        return 0.0


def _median(values: list) -> float:
    try:
        vals = sorted([float(x) for x in values if float(x) > 0])
        if not vals:
            return 0.0
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            return float(vals[mid])
        return float((vals[mid - 1] + vals[mid]) / 2.0)
    except Exception:
        return 0.0

def _email_priority_bases(best_fut: dict, directional_take: int = 12) -> set:
    # Directional Leaders/Losers = priority symbols (email override)
    try:
        up_list, dn_list = compute_directional_lists(best_fut)
        leaders = _bases_from_directional(up_list, directional_take)
        losers  = _bases_from_directional(dn_list, directional_take)
        return set([str(x).upper() for x in (leaders + losers)])
    except Exception:
        return set()



def _fallback_setups_from_universe(best_fut: dict, leaders: list, losers: list, market_bases: list, session_name: str, max_items: int = 4) -> list:
    """Last-resort generator to avoid empty Top Trade Setups.

    Creates simple ATR-based setups for the most active symbols in the
    directional leaders/losers + market leaders universe.
    """
    bases = []
    for b in (leaders or [])[:2]:
        bases.append((str(b).upper(), "BUY"))
    for b in (losers or [])[:2]:
        bases.append((str(b).upper(), "SELL"))
    for b in (market_bases or [])[:2]:
        bb = str(b).upper()
        if all(bb != x[0] for x in bases):
            # Decide side by 4H sign if available later; default BUY
            bases.append((bb, "BUY"))

    setups = []
    for base, side in bases:
        mv = (best_fut or {}).get(base)
        if not mv:
            continue
        market_symbol = str(getattr(mv, "symbol", base))
        entry = float(getattr(mv, "last", 0.0) or 0.0)
        if entry <= 0:
            continue

        try:
            c1 = fetch_ohlcv(market_symbol, "1h", limit=max(ATR_PERIOD + 10, 80))
            atr_1h = compute_atr_from_ohlcv(c1, ATR_PERIOD) if c1 else 0.0
        except Exception:
            atr_1h = 0.0

        if atr_1h <= 0:
            # small default: 1% of price
            atr_1h = max(entry * 0.01, 0.0000001)

        sl_dist = 1.4 * atr_1h
        tp_dist = 2.2 * atr_1h

        if side == "BUY":
            sl = max(entry - sl_dist, entry * 0.001)
            tp3 = entry + tp_dist
            tp1 = entry + tp_dist * 0.6
            tp2 = entry + tp_dist * 0.85
        else:
            sl = entry + sl_dist
            tp3 = max(entry - tp_dist, entry * 0.001)
            tp1 = max(entry - tp_dist * 0.6, entry * 0.001)
            tp2 = max(entry - tp_dist * 0.85, entry * 0.001)

        fut_vol = _fut_vol_usd_from_best(best_fut, base)

        setups.append(Setup(
            setup_id=_new_setup_id(),
            symbol=base,
            market_symbol=market_symbol,
            side=side,
            conf=70,
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            fut_vol_usd=float(fut_vol or 0.0),
            ch24=float(getattr(mv, "percentage", 0.0) or 0.0),
            ch4=0.0,
            ch1=0.0,
            ch15=0.0,
            ema_support_period=0,
            ema_support_dist_pct=0.0,
            pullback_ema_period=0,
            pullback_ema_dist_pct=0.0,
            pullback_ready=True,
            pullback_bypass_hot=True,
            engine="F",
            is_trailing_tp3=False,
            created_ts=time.time(),
        ))

        if len(setups) >= int(max_items):
            break
    return setups

async def build_priority_pool(best_fut: dict, session_name: str, mode: str, scan_profile: str = DEFAULT_SCAN_PROFILE, uid: int | None = None) -> dict:
    """
    mode: "screen" or "email"
    returns: {
        "setups": [Setup...],
        "waiting": [(base, info_dict)...],
        "trend_watch": [dict...],
        "spikes": [dict...],   # NEW
        "spike_warnings": [dict...],  # NEW (Early Warning, non-trade)
    }
    """

    # Diagnostics: collect reject reasons for this scan
    # Keep a stable structure so /why is always meaningful.
    _rej_ctx = {
        "__agg__": {},          # aggregate counters per reason
        "__per__": {},          # per-symbol last reason/status
        "__allow__": set(),     # universe allow-list (uppercased bases)
    }

    # Fallback for nested threads that may lose contextvars
    global _GLOBAL_REJECT_CTX
    _GLOBAL_REJECT_CTX = _rej_ctx
    _rej_token = _REJECT_CTX.set(_rej_ctx)

    # Reset near-miss list for this scan (so /screen doesn't show stale symbols)
    try:
        _WAITING_TRIGGER.clear()
    except Exception:
        pass

    # knobs
    if mode == "screen":
        n_target = int(SETUPS_N)
        strict_15m = False
        universe_cap = int(SCREEN_UNIVERSE_N)
        trigger_loosen = float(SCREEN_TRIGGER_LOOSEN)
        waiting_near = float(SCREEN_WAITING_NEAR_PCT)
        allow_no_pullback = True
        scan_multiplier = 10
        directional_take = 12
        market_take = 15
        trend_take = 12
    else:  # email
        n_target = int(max(EMAIL_SETUPS_N * 3, 9))
        strict_15m = True
        universe_cap = 35
        trigger_loosen = 1.0
        waiting_near = float(SCREEN_WAITING_NEAR_PCT)
        allow_no_pullback = True
        scan_multiplier = 10
        directional_take = 12
        market_take = 15
        trend_take = 12

    # Aggressive profile overrides
    prof = str(scan_profile or DEFAULT_SCAN_PROFILE).strip().lower()
    if prof not in SCAN_PROFILES:
        prof = DEFAULT_SCAN_PROFILE

    if prof == "aggressive":
        if mode == "screen":
            n_target = int(max(SETUPS_N, 8))
            universe_cap = int(max(SCREEN_UNIVERSE_N, 110))
            trigger_loosen = float(min(0.80, SCREEN_TRIGGER_LOOSEN))
            waiting_near = float(min(0.70, SCREEN_WAITING_NEAR_PCT))
            scan_multiplier = 12
            directional_take = 16
            market_take = 18
            trend_take = 16
            strict_15m = False
            allow_no_pullback = True
        else:
            # Email pool becomes broader, but final email gates still apply
            n_target = int(max(EMAIL_SETUPS_N * 4, 12))
            universe_cap = int(max(60, universe_cap))
            trigger_loosen = 0.95
            waiting_near = float(SCREEN_WAITING_NEAR_PCT)
            strict_15m = True
            allow_no_pullback = True
            scan_multiplier = 12
            directional_take = 14
            market_take = 16
            trend_take = 14

    # 1) Directional leaders / losers (priority #1)
    up_list, dn_list = compute_directional_lists(best_fut)

    directional_table_n = 10
    leaders = [str(t[0]).upper() for t in (up_list or [])[:directional_table_n]]
    losers  = [str(t[0]).upper() for t in (dn_list or [])[:directional_table_n]]

    # Market Leaders (Top by Futures Volume)
    market_bases = _market_leader_bases(best_fut, market_take)

    # ✅ Setup universe: ONLY what /screen shows — Directional Leaders + Directional Losers + Market Leaders.
    universe_bases = list(dict.fromkeys([b.upper() for b in (leaders + losers + (market_bases or []))]))
    universe_best = _subset_best(best_fut, universe_bases) if universe_bases else {}

    # Diagnostics: keep /why focused on this scan universe
    try:
        _rej_ctx["__allow__"] = set(str(x).upper() for x in (universe_bases or []) if x)
        _rej_ctx["__per__"] = {b: {"reason": "not_evaluated", "n": 0} for b in (_rej_ctx["__allow__"] or set())}

        try:
            global _LAST_SCAN_UNIVERSE
            _LAST_SCAN_UNIVERSE = list(universe_bases or [])
        except Exception:
            pass
    except Exception:
        pass

    priority_setups = []

    # Leaders pass
    if leaders:
        sub = _subset_best(best_fut, leaders)
        try:
            tmp = pick_setups(
                sub,
                n_target * scan_multiplier,
                strict_15m,
                session_name,
                min(int(universe_cap), len(sub) if sub else universe_cap),
                trigger_loosen,
                waiting_near,
                allow_no_pullback,
                scan_profile=prof,
            )
        except Exception:
            tmp = []
        for s in (tmp or []):
            if s.side == "BUY":
                priority_setups.append(s)

    # Losers pass
    if losers:
        sub = _subset_best(best_fut, losers)
        try:
            tmp = pick_setups(
                sub,
                n_target * scan_multiplier,
                strict_15m,
                session_name,
                min(int(universe_cap), len(sub) if sub else universe_cap),
                trigger_loosen,
                waiting_near,
                allow_no_pullback,
                scan_profile=prof,
            )
        except Exception:
            tmp = []
        for s in (tmp or []):
            if s.side == "SELL":
                priority_setups.append(s)

    # 1b) Full universe pass (leaders + losers + market leaders)
    try:
        if universe_best:
            tmp = pick_setups(
                universe_best,
                n_target * scan_multiplier,
                strict_15m,
                session_name,
                min(int(universe_cap), len(universe_best) if universe_best else universe_cap),
                trigger_loosen,
                waiting_near,
                allow_no_pullback,
                scan_profile=prof,
            )
        else:
            tmp = []
    except Exception:
        tmp = []
    for s in (tmp or []):
        priority_setups.append(s)

    # ------------------------------------------------
    # Engine B: Momentum Breakout Setups (Balanced)
    # ------------------------------------------------
    breakout_setups = []
    try:
        bases_for_breakout = list(dict.fromkeys([b.upper() for b in (leaders + losers)]))
        if bases_for_breakout:
            sub = _subset_best(best_fut, bases_for_breakout)
            breakout_setups = pick_breakout_setups(
                sub,
                int(max(6, n_target)),   # allow a handful
                session_name,
                min(int(universe_cap), len(sub) if sub else universe_cap),
                scan_profile=prof,
            )
    except Exception:
        breakout_setups = []

    if breakout_setups:
        priority_setups.extend(breakout_setups)

    # 2) Trend continuation watch (priority #2)
    watch_bases = []
    watch_bases.extend([b for b in leaders[:6]])
    watch_bases.extend([b for b in losers[:6]])
    watch_bases = list(dict.fromkeys([b.upper() for b in watch_bases]))

    trend_watch = []
    trend_bases = []
    for base in watch_bases:
        mv = (best_fut or {}).get(base)
        if not mv:
            continue
        r = await asyncio.to_thread(trend_watch_for_symbol, base, mv, session_name)
        if r:
            trend_watch.append(r)
            trend_bases.append(base)

    if trend_bases:
        sub = _subset_best(best_fut, trend_bases[:trend_take])
        try:
            tmp = pick_setups(
                sub,
                n_target * scan_multiplier,
                strict_15m,
                session_name,
                min(int(universe_cap), len(sub) if sub else universe_cap),
                trigger_loosen,
                waiting_near,
                True,  # allow trend continuation to pass even on email
                scan_profile=prof,
            )
        except Exception:
            tmp = []

        side_map = {str(t.get("symbol")).upper(): str(t.get("side")) for t in (trend_watch or []) if t.get("symbol")}
        for s in (tmp or []):
            want = side_map.get(str(s.symbol).upper())
            if want and s.side == want:
                priority_setups.append(s)

    # 3) Waiting for Trigger (priority #3) is produced by pick_setups via _WAITING_TRIGGER
    waiting_items = []
    try:
        if _WAITING_TRIGGER:
            allow_set = set(str(x).upper() for x in (_LAST_SCAN_UNIVERSE or []))
            waiting_items = [
                (b, o)
                for (b, o) in list(_WAITING_TRIGGER.items())
                if (not allow_set) or (str(b).upper() in allow_set)
            ][:SCREEN_WAITING_N]
    except Exception:
        waiting_items = []

    # 4) Market leaders fallback (priority #4)
    if len(priority_setups) < (n_target * 2):
        if market_bases:
            sub = _subset_best(best_fut, market_bases)
            try:
                tmp = pick_setups(
                    sub,
                    n_target * scan_multiplier,
                    strict_15m,
                    session_name,
                    min(int(universe_cap), len(sub) if sub else universe_cap),
                    trigger_loosen,
                    waiting_near,
                    allow_no_pullback,
                    scan_profile=prof,
                )
            except Exception:
                tmp = []
            priority_setups.extend(tmp or [])

    # 4B) Momentum Breakout setups (Engine B) — Balanced
    try:
        mom_n = max(6, int(n_target) * 2)
        mom = pick_breakout_setups(
            universe_best,  # ✅ restricted universe
            mom_n,
            session_name,
            int(universe_cap),
            scan_profile=prof,
        )
    except Exception:
        mom = []
    if mom:
        priority_setups.extend(mom)

    # de-dupe by (symbol, side, engine) keeping highest conf, preserving priority order
    best = {}
    for s in priority_setups:
        k = (str(s.symbol).upper(), str(s.side), str(getattr(s, "engine", "")))
        if k not in best or int(s.conf) > int(best[k].conf):
            best[k] = s

    ordered = []
    seen = set()
    for s in priority_setups:
        k = (str(s.symbol).upper(), str(s.side), str(getattr(s, "engine", "")))
        if k in seen:
            continue
        if k in best:
            ordered.append(best[k])
            seen.add(k)

    # If still empty, create a small fallback set so /screen isn't blank.
    if not ordered and mode == "screen":
        try:
            ordered = _fallback_setups_from_universe(best_fut, leaders, losers, market_bases, session_name, max_items=max(4, n_target))
        except Exception:
            pass

    # -----------------------------------------------------
    # NEW: Spike Reversal candidates (15M+ Vol) — for /screen only
    # -----------------------------------------------------
    spike_candidates = []
    if mode == "screen":
        try:
            spike_candidates = await asyncio.to_thread(
                _spike_reversal_candidates,
                universe_best,
                10_000_000.0,  # min_vol_usd
                0.55,          # wick_ratio_min
                1.20,          # atr_mult_min
                6,             # max_items
            )
        except Exception:
            spike_candidates = []

    # -----------------------------------------------------
    # NEW: Early Warning — Possible Reversal Zones (non-trade)
    # -----------------------------------------------------
    spike_warnings = []
    if mode == "screen":
        try:
            spike_warnings = await asyncio.to_thread(
                _spike_reversal_warnings,
                universe_best,
                10_000_000.0,  # min_vol_usd
                1.15,          # atr_mult_min
                0.60,          # body_ratio_min
                8,             # lookback_1h
                0.30,          # retrace_min
                6,             # max_items
            )
        except Exception:
            spike_warnings = []

    # Store diagnostics for /why, then ALWAYS reset context
    try:
        if uid is not None:
            _LAST_REJECTS[int(uid)] = {
                "ts": time.time(),
                "allow": list(_rej_ctx.get("__allow__") or []),
                "per_symbol": dict((_rej_ctx.get("__per__") or {})),
            }
    finally:
        try:
            _REJECT_CTX.reset(_rej_token)
        except Exception:
            pass
        try:
            _GLOBAL_REJECT_CTX = None
        except Exception:
            pass

    return {
        "setups": ordered,
        "waiting": waiting_items,
        "trend_watch": trend_watch,
        "spikes": spike_candidates,
        "spike_warnings": spike_warnings,
    }



def user_location_and_time(user: dict):
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

    # Build location label safely
    if "/" in tz_name:
        parts = tz_name.split("/")
        region = parts[0].replace("_", " ")
        city = parts[-1].replace("_", " ")
        loc = f"{city} ({region})"
    else:
        loc = tz_name

    return loc, now_local.strftime("%Y-%m-%d %H:%M")

# =========================================================
# User location/time helpers
# =========================================================
def user_location_and_time(user: dict):
    """
    Returns: (location_label, time_str) based on user's tz
    """
    tz_name = str((user or {}).get("tz") or "UTC")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
        tz_name = "UTC"

    now_local = datetime.now(tz)

    if "/" in tz_name:
        parts = tz_name.split("/")
        region = parts[0].replace("_", " ")
        city = parts[-1].replace("_", " ")
        loc = f"{city} ({region})"
    else:
        loc = tz_name

    return loc, now_local.strftime("%Y-%m-%d %H:%M")


# =========================================================
# /screen fast cache (per-instance)
# =========================================================
SCREEN_CACHE_TTL_SEC = 20  # seconds
_SCREEN_CACHE = {
    "ts": 0.0,
    "body": "",
    "kb": [],
}
_SCREEN_LOCK = asyncio.Lock()


# =========================================================
# /screen
# =========================================================



def _build_screen_body_and_kb(best_fut: dict, session: str, uid: int):
    """Heavy /screen builder (runs in a worker thread).

    Returns:
        body (str): cached body (header is built in the async handler)
        kb (list[tuple[str,str]]): [(SYMBOL, SETUP_ID), ...] for TradingView buttons
    """
    # Build pool (coroutine) in this worker thread (isolated event loop)
    pool = _run_coro_in_thread(
        build_priority_pool(
            best_fut,
            session,
            mode="screen",
            scan_profile=str(DEFAULT_SCAN_PROFILE),
            uid=uid,
        )
    )

    # Other heavy helpers are sync; run them here too.
    leaders_txt = build_leaders_table(best_fut)
    up_txt, dn_txt = movers_tables(best_fut)

    # Hard cap: never show more than 3 top setups on /screen
    try:
        setups = (pool.get("setups") or [])[:min(int(SETUPS_N), 3)]
    except Exception:
        setups = (pool.get("setups") or [])[:3]

    # Safety: if engine produced no setups, generate fallback ATR-based setups
    if not setups:
        try:
            up_list, dn_list = compute_directional_lists(best_fut)
            leaders_bases = [str(t[0]).upper() for t in (up_list or [])[:10]]
            losers_bases  = [str(t[0]).upper() for t in (dn_list or [])[:10]]
            market_bases  = _market_leader_bases(best_fut)[:10]
            setups = _fallback_setups_from_universe(
                best_fut,
                leaders_bases,
                losers_bases,
                market_bases,
                session,
                max_items=max(4, int(SETUPS_N or 4)),
            )[:int(SETUPS_N or 4)]
        except Exception:
            setups = []

    # Ensure conf exists + persist signal cards to DB (used by Signal ID lookup)
    try:
        for s in (setups or []):
            if not hasattr(s, "conf") or s.conf is None:
                s.conf = 0
            db_insert_signal(s)
    except Exception:
        pass

    # Setup cards -> combined text (single "Top Trade Setups" section)
    combined_setups_txt = "_No high-quality setups right now._"
    if setups:
        def _mv_dot(p: float) -> str:
            try:
                p = float(p or 0.0)
            except Exception:
                p = 0.0
            if abs(p) < 2.0:
                return "🟡"
            return "🟢" if p >= 0 else "🔴"

        def _engine_label(e: str) -> str:
            ee = str(e or "").strip().upper()
            if ee == "A":
                return "Pullback"
            if ee == "B":
                return "Momentum Breakout"
            if ee == "F":
                return "Fallback"
            return "Setup"

        lines2 = []
        for s in setups:
            try:
                sym = str(getattr(s, "symbol", "")).upper()
                sid = str(getattr(s, "setup_id", "") or "")
                side = str(getattr(s, "side", "") or "").upper()
                conf = int(getattr(s, "conf", 0) or 0)

                entry = float(getattr(s, "entry", 0.0) or 0.0)
                sl = float(getattr(s, "sl", 0.0) or 0.0)
                tp1 = getattr(s, "tp1", None)
                tp2 = getattr(s, "tp2", None)
                tp3 = float(getattr(s, "tp3", 0.0) or 0.0)
                vol = float(getattr(s, "fut_vol_usd", 0.0) or 0.0)

                ch24 = float(getattr(s, "ch24", 0.0) or 0.0)
                ch4 = float(getattr(s, "ch4", 0.0) or 0.0)
                ch1 = float(getattr(s, "ch1", 0.0) or 0.0)
                ch15 = float(getattr(s, "ch15", 0.0) or 0.0)

                rr_den = abs(entry - sl)
                rr1 = (abs(float(tp1) - entry) / rr_den) if (rr_den > 0 and tp1 not in (None, 0, 0.0)) else 0.0
                rr2 = (abs(float(tp2) - entry) / rr_den) if (rr_den > 0 and tp2 not in (None, 0, 0.0)) else 0.0
                rr3 = (abs(tp3 - entry) / rr_den) if rr_den > 0 else 0.0

                pos_word = "long" if side == "BUY" else "short"
                size_cmd = f"/size {sym} {pos_word} entry {entry:.6g} sl {sl:.6g}"

                emoji = "🟢" if side == "BUY" else "🔴"
                typ = _engine_label(getattr(s, "engine", ""))

                # Card-style formatting (same as previous detailed preview) + /size command
                block = []
                block.append(f"{emoji} *{side} — {sym}*")
                block.append(f"`{sid}` | Conf: `{conf}`")
                block.append(f"Type: {typ} | RR(TP1): `{rr1:.2f}` | RR(TP2): `{rr2:.2f}` | RR(TP3): `{rr3:.2f}`")
                block.append(f"Entry: `{fmt_price(entry)}` | SL: `{fmt_price(sl)}`")
                if tp1 not in (None, 0, 0.0) and tp2 not in (None, 0, 0.0):
                    block.append(f"TP1: `{fmt_price(float(tp1))}` | TP2: `{fmt_price(float(tp2))}` | TP3: `{fmt_price(tp3)}`")
                else:
                    block.append(f"TP: `{fmt_price(tp3)}`")
                block.append(
                    f"Moves: 24H {ch24:+.0f}% {_mv_dot(ch24)} • 4H {ch4:+.0f}% {_mv_dot(ch4)} • "
                    f"1H {ch1:+.0f}% {_mv_dot(ch1)} • 15m {ch15:+.0f}% {_mv_dot(ch15)}"
                )
                block.append(f"Volume: ~{vol/1e6:.1f}M")
                block.append(f"Chart: {tv_chart_url(sym)}")
                block.append(f"`{size_cmd}`")
                lines2.append("\n".join(block))
            except Exception:
                continue

        combined_setups_txt = ("\n\n".join(lines2)).strip() if lines2 else "_No high-quality setups right now._"

    # Waiting for Trigger (near-miss)
    waiting_txt = ""
    waiting_items = pool.get("waiting") or []
    if not waiting_items and _WAITING_TRIGGER:
        try:
            waiting_items = list(_WAITING_TRIGGER.items())[:SCREEN_WAITING_N]
        except Exception:
            waiting_items = []

    if waiting_items:
        lines = ["*Waiting for Trigger (near-miss)*", SEP]
        for item in waiting_items[:SCREEN_WAITING_N]:
            try:
                if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], dict):
                    base, d = item
                    raw_side = str(d.get("side", "BUY") or "BUY").strip().upper()
                    if raw_side in ("LONG",):
                        side = "BUY"
                    elif raw_side in ("SHORT",):
                        side = "SELL"
                    elif raw_side in ("BUY", "SELL"):
                        side = raw_side
                    else:
                        side = raw_side
                    dot = "🟢" if side == "BUY" else ("🔴" if side == "SELL" else "🟡")
                    lines.append(f"• *{base}* {dot} `{side}`")
                else:
                    lines.append(f"• `{str(item)}`")
            except Exception:
                continue
        waiting_txt = "\n".join(lines)

    # Trend continuation watch
    trend_txt = ""
    trend_watch = pool.get("trend_watch") or []
    if trend_watch:
        lines = ["*Trend Continuation Watch*", SEP]
        trend_watch_sorted = sorted(
            trend_watch,
            key=lambda x: int(x.get("confidence", x.get("conf", 0)) or 0),
            reverse=True
        )[:6]
        for t in trend_watch_sorted:
            side = str(t.get("side", "BUY"))
            side_emoji = "🟢" if side == "BUY" else "🔴"
            conf_val = int(t.get("confidence", t.get("conf", 0)) or 0)
            sym = str(t.get("symbol", "")).upper()
            ch24 = float(t.get("ch24", 0.0) or 0.0)
            lines.append(f"• *{sym}* {side_emoji} `{side}` | Conf `{conf_val}` | 24H {pct_with_emoji(ch24)}")
        trend_txt = "\n".join(lines)

    # Early Warning
    warning_txt = ""
    warnings = pool.get("spike_warnings") or []
    if warnings:
        lines = ["*Early Warning (Possible Reversal Zones)*", SEP]
        for w in warnings[:6]:
            try:
                sym = str(w.get("symbol", "")).upper()
                side = str(w.get("side", "SELL")).upper()
                conf = int(w.get("conf", 0) or 0)
                vol = float(w.get("vol", 0.0) or 0.0)
                side_emoji = "🟢" if side == "BUY" else "🔴"
                lines.append(f"• *{sym}* {side_emoji} `{side}` | Conf `{conf}` | Vol~`{vol/1e6:.1f}M`")
            except Exception:
                continue
        warning_txt = "\n".join(lines)

    # Spike Reversal Alerts (10M+ Vol) — includes /size line
    spike_txt = ""
    spikes = pool.get("spikes") or []
    if spikes:
        lines = ["*Spike Reversal Alerts (10M+ Vol)*", SEP]
        for c in spikes[:6]:
            try:
                sym = str(c.get("symbol", "")).upper()
                side = str(c.get("side", "SELL")).upper()
                conf = int(c.get("conf", 0) or 0)
                entry = float(c.get("entry", 0.0) or 0.0)
                sl = float(c.get("sl", 0.0) or 0.0)
                tp3 = float(c.get("tp3", 0.0) or 0.0)
                vol = float(c.get("vol", 0.0) or 0.0)

                rr_den = abs(entry - sl)
                rr3 = (abs(tp3 - entry) / rr_den) if rr_den > 0 else 0.0
                pos_word = "long" if side == "BUY" else "short"
                size_cmd = f"/size {sym} {pos_word} entry {entry:.6g} sl {sl:.6g}"

                side_emoji = "🟢" if side == "BUY" else "🔴"
                lines.append(f"• *{sym}* {side_emoji} `{side}` | Conf `{conf}` | RR(TP3) `{rr3:.2f}` | Vol~`{vol/1e6:.1f}M`")
                lines.append(f"  `{size_cmd}`")
            except Exception:
                continue
        spike_txt = "\n".join(lines)

    # Assemble body (cache THIS, header stays live)
    def _is_empty(txt: str) -> bool:
        t = (txt or "").strip()
        if not t:
            return True
        return t.startswith("_No ") or t.startswith("No ") or t.endswith("right now._")

    blocks = []
    blocks.extend(["", "*Top Trade Setups*", SEP, combined_setups_txt])

    if waiting_txt and (not _is_empty(waiting_txt)):
        blocks.extend(["", waiting_txt])

    if trend_txt and (not _is_empty(trend_txt)):
        blocks.extend(["", trend_txt])

    if warning_txt and (not _is_empty(warning_txt)):
        blocks.extend(["", warning_txt])

    if spike_txt and (not _is_empty(spike_txt)):
        blocks.extend(["", spike_txt])

    if up_txt and (not _is_empty(up_txt)) and ("|" in up_txt):
        blocks.extend(["", "*Directional Leaders / Losers*", SEP, up_txt])

    if dn_txt and (not _is_empty(dn_txt)) and ("|" in dn_txt):
        blocks.extend(["", dn_txt])

    if leaders_txt and (not _is_empty(leaders_txt)) and ("|" in leaders_txt):
        blocks.extend(["", "*Market Leaders*", SEP, leaders_txt])

    body = "\n".join([b for b in blocks if b is not None]).strip()

    kb = []
    try:
        kb = [(s.symbol, s.setup_id) for s in (setups or [])]
    except Exception:
        kb = []
    return body, kb
async def screen_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):

    uid = update.effective_user.id
    user = get_user(uid)

    if not has_active_access(user, uid):
        await update.message.reply_text(
            "⛔️ Trial finished.\n\n"
            "Your 7-day trial is over — you need to pay to keep using PulseFutures.\n\n"
            "👉 /billing"
        )
        return

    # Avoid long scans blocking other commands on small instances
    if SCAN_LOCK.locked():
        await update.message.reply_text("⏳ Scan is running… please try /screen again in a moment.")
        return

    await SCAN_LOCK.acquire()
    try:
        # Send immediate response (fast perceived UX)
        status_msg = await update.message.reply_text("🔎 Scanning market… Please wait")

        reset_reject_tracker()

        best_fut = await asyncio.to_thread(fetch_futures_tickers)
        if not best_fut:
            await status_msg.edit_text("❌ Failed to fetch futures data.")
            return

        # Header (always fresh)
        uid = update.effective_user.id
        user = get_user(uid)
        session = current_session_utc()
        loc_label, loc_time = user_location_and_time(user)

        header = (
            f"*PulseFutures — Market Scan*\n"
            f"{HDR}\n"
            f"*Session:* `{session}` | *{loc_label}:* `{loc_time}`\n"
        )

        now_ts = time.time()

        # ------------- FAST PATH (cache hit) -------------
        cached_body = ""
        cached_kb = []
        if (_SCREEN_CACHE.get("body") and (now_ts - float(_SCREEN_CACHE.get("ts", 0.0)) <= float(SCREEN_CACHE_TTL_SEC))):
            cached_body = str(_SCREEN_CACHE.get("body") or "")
            cached_kb = list(_SCREEN_CACHE.get("kb") or [])

            msg = (header + "\n" + cached_body).strip()

            keyboard = [
                [InlineKeyboardButton(text=f"📈 {sym} • {sid}", url=tv_chart_url(sym))]
                for (sym, sid) in (cached_kb or [])
            ]

            try:
                await status_msg.delete()
            except Exception:
                pass

            await send_long_message(
                update,
                msg,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
                reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None,
            )
            return

        # ------------- SLOW PATH (build once under lock) -------------
        async with _SCREEN_LOCK:
            # Re-check cache after waiting for lock (another request may have filled it)
            now_ts = time.time()
            if (_SCREEN_CACHE.get("body") and (now_ts - float(_SCREEN_CACHE.get("ts", 0.0)) <= float(SCREEN_CACHE_TTL_SEC))):
                cached_body = str(_SCREEN_CACHE.get("body") or "")
                cached_kb = list(_SCREEN_CACHE.get("kb") or [])

                msg = (header + "\n" + cached_body).strip()
                keyboard = [
                    [InlineKeyboardButton(text=f"📈 {sym} • {sid}", url=tv_chart_url(sym))]
                    for (sym, sid) in (cached_kb or [])
                ]

                try:
                    await status_msg.delete()
                except Exception:
                    pass

                await send_long_message(
                    update,
                    msg,
                    parse_mode=ParseMode.MARKDOWN,
                    disable_web_page_preview=True,
                    reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None,
                )
                return


            # Heavy build MUST NOT run on the asyncio event loop.
            # Only /screen is allowed to take longer; everything here runs in a worker thread.
            body, kb = await asyncio.to_thread(
                _build_screen_body_and_kb,
                best_fut,
                session,
                int(update.effective_user.id),
            )

            # Cache for fast subsequent /screen calls
            _SCREEN_CACHE["ts"] = time.time()
            _SCREEN_CACHE["body"] = body
            _SCREEN_CACHE["kb"] = list(kb or [])


        # Send final
        msg = (header + "\n" + str(_SCREEN_CACHE.get("body") or "")).strip()
        keyboard = [
            [InlineKeyboardButton(text=f"📈 {sym} • {sid}", url=tv_chart_url(sym))]
            for (sym, sid) in (_SCREEN_CACHE.get("kb") or [])
        ]

        try:
            await status_msg.delete()
        except Exception:
            pass

        await send_long_message(
            update,
            msg,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None,
        )

    except Exception as e:
        logger.exception("screen_cmd failed")
        try:
            await update.message.reply_text(f"❌ /screen failed: {e}")
        except Exception:
            pass



# =========================================================
# TEXT ROUTER (Signal ID lookup)
# =========================================================
    finally:
        try:
            if SCAN_LOCK.locked():
                SCAN_LOCK.release()
        except Exception:
            pass


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

def tz_location_label(tz_name: str) -> str:
    """Return a friendly location label from an IANA tz like 'Australia/Sydney'."""
    try:
        tz_name = str(tz_name or '').strip()
    except Exception:
        tz_name = ''
    if not tz_name:
        return 'Local time'
    parts = tz_name.split('/')
    region = parts[0] if parts else tz_name
    city = parts[-1] if parts else tz_name
    try:
        city = city.replace('_', ' ').strip()
    except Exception:
        pass
    region_label = region
    # Small UX mappings (keep it simple + safe)
    if region == 'America':
        region_label = 'USA'
    elif region == 'Etc':
        region_label = 'UTC'
    return f"{city} ({region_label})" if city else str(region_label)

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
        try:
            _pos = "long" if str(getattr(s, "side", "")).upper() == "BUY" else "short"
            parts.append(f"   /size {str(getattr(s, 'symbol', ''))} {_pos} entry {float(getattr(s, 'entry', 0.0) or 0.0):.6g} sl {float(getattr(s, 'sl', 0.0) or 0.0):.6g}")
        except Exception:
            pass
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

def send_email_alert_multi(user: dict, sess: dict, setups: List[Setup], best_fut) -> bool:
    """
    One email containing multiple setups.
    Requires _email_body_pretty(...) and send_email(...).
    """
    if not setups:
        return False

    uid = int(user["user_id"])
    user_tz = str(user.get("tz") or "UTC")
    try:
        tz = ZoneInfo(user_tz)
    except Exception:
        tz = timezone.utc
        user_tz = "UTC"

    now_local = datetime.now(tz)

    first = setups[0]
    subject = f"PulseFutures • {sess['name']} • {first.side} {first.symbol}"
    if len(setups) > 1:
        subject += f" (+{len(setups)-1} more)"

    body = _email_body_pretty(
        session_name=str(sess["name"]),
        now_local=now_local,
        user_tz=user_tz,
        setups=setups,
        best_fut=best_fut,
    )

    return send_email(subject, body, user_id_for_debug=uid)



# =========================================================
# STRIPE WEBHOOK SERVER
# =========================================================
class StripeWebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = self.rfile.read(int(self.headers["Content-Length"]))
        sig = self.headers.get("Stripe-Signature")

        try:
            event = stripe.Webhook.construct_event(
                payload,
                sig,
                os.environ.get("STRIPE_WEBHOOK_SECRET"),
            )
        except Exception:
            self.send_response(400)
            self.end_headers()
            return

        data = event["data"]["object"]

        if event["type"] in ("checkout.session.completed", "invoice.paid"):
            email = data.get("customer_email")
        
            # get price + plan
            line = data["lines"]["data"][0]
            price_id = line["price"]["id"]
            plan = STRIPE_PRICE_TO_PLAN.get(price_id)
        
            # reference + amount
            ref = data.get("id") or event.get("id")
            amount = (line.get("amount_total") or data.get("amount_paid") or 0) / 100
            currency = (data.get("currency") or "usd").upper()
        
            if email and plan:
                activate_user_with_ledger_by_email(
                    email=email,
                    plan=plan,
                    ref=ref,
                    amount=amount,
                    currency=currency,
                )

        if event["type"] == "customer.subscription.deleted":
            email = data.get("customer_email")
            ref = data.get("id") or event.get("id")
            if email:
                downgrade_user_with_ledger_by_email(email, ref=ref)

        self.send_response(200)
        self.end_headers()

def start_stripe_webhook():
    HTTPServer(("0.0.0.0", 4242), StripeWebhookHandler).serve_forever()

# =========================================================
# MY PLAN & BILLING
# =========================================================

async def myplan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    user = get_user(user_id)  # however you fetch user data

    plan = effective_plan(user_id, user)
    expires = (user or {}).get("plan_expires")

    msg = f"""\
📦 Your Plan

• Plan: {plan.upper()}
• Status: {'Admin (Unlimited)' if is_admin_user(user_id) else ('Active' if plan != 'free' else 'Free user')}
"""

    if expires:
        msg += f"• Expires: {expires}\n"

    await update.message.reply_text(msg)


def _env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return (v or default).strip()

def _mask_addr(addr: str) -> str:
    a = (addr or "").strip()
    if len(a) <= 12:
        return a
    return f"{a[:6]}…{a[-6:]}"

async def _billing_cmd_unused(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Billing menu: Stripe (Payment Links) + USDT.
    - Uses env vars only (safe on Render)
    - Does NOT call Stripe API (no stripe.api_key needed)
    - Never crashes if not configured
    """
    user = update.effective_user
    uid = user.id if user else None

    # Stripe Payment Links
    stripe_standard_url = _env("STRIPE_STANDARD_URL")
    stripe_pro_url = _env("STRIPE_PRO_URL")

    # USDT
    usdt_network = _env("USDT_NETWORK", "TRC20")
    usdt_address = _env("USDT_ADDRESS")
    usdt_note = _env("USDT_NOTE")

    # Support
    support_handle = _env("BILLING_SUPPORT_HANDLE", "@PulseFuturesSupport")

    # Reference for manual matching (USDT payments, or any support ticket)
    ref = f"PF-{uid}" if uid else "PF-UNKNOWN"

    lines = []
    lines.append("💳 PulseFutures — Billing & Upgrade")
    lines.append("")
    lines.append("Choose your payment method below.")
    lines.append(f"Reference (important): {ref}")
    lines.append("")
    lines.append("────────────────────")
    lines.append("1) Stripe (Card / Apple Pay / Google Pay)")
    lines.append("────────────────────")
    if stripe_standard_url or stripe_pro_url:
        lines.append("Tap a plan button to pay securely via Stripe.")
    else:
        lines.append("Stripe is not configured yet.")
        lines.append("Admin: set STRIPE_STANDARD_URL / STRIPE_PRO_URL in Render env vars.")

    lines.append("")
    lines.append("────────────────────")
    lines.append("2) USDT")
    lines.append("────────────────────")
    if usdt_address:
        lines.append(f"Network: {usdt_network}")
        lines.append(f"Address: {usdt_address}")
        lines.append(f"(Short: {_mask_addr(usdt_address)})")
        if usdt_note:
            lines.append(f"Note: {usdt_note.replace('<REF>', ref)}")
        else:
            lines.append(f"Note: After sending, message support with TXID + reference '{ref}'.")
    else:
        lines.append("USDT is not configured yet.")
        lines.append("Admin: set USDT_ADDRESS (+ optional USDT_NETWORK) in Render env vars.")

    lines.append("")
    lines.append("────────────────────")
    lines.append("After payment")
    lines.append("────────────────────")
    lines.append("1) If Stripe: you’ll be activated after confirmation (or contact support if needed).")
    lines.append("2) If USDT: submit TXID using /usdt_paid <TXID> <standard|pro>.")
    lines.append(f"Support: {support_handle}")
    lines.append(f"Reference: {ref}")

    msg = "\n".join(lines)

    # Buttons
    buttons = []
    stripe_row = []
    if stripe_standard_url:
        stripe_row.append(InlineKeyboardButton("✅ Stripe — Standard", url=stripe_standard_url))
    if stripe_pro_url:
        stripe_row.append(InlineKeyboardButton("🚀 Stripe — Pro", url=stripe_pro_url))
    if stripe_row:
        buttons.append(stripe_row)

    howto_url = _env("BILLING_HOWTO_URL")
    if howto_url:
        buttons.append([InlineKeyboardButton("ℹ️ Payment Instructions", url=howto_url)])

    reply_markup = InlineKeyboardMarkup(buttons) if buttons else None

    await send_long_message(
        update,
        msg,
        parse_mode=None,
        disable_web_page_preview=True,
        reply_markup=reply_markup,
    )

def activate_user_with_ledger_by_email(email: str, plan: str, ref: str, amount: float = 0, currency: str = "USD"):
    """
    Activates user by email AND records payment in ledger.
    """
    user = get_user_by_email(email)
    if not user:
        return

    user_id = user["user_id"]

    # 1) grant access (existing behavior)
    activate_user_by_email(email, plan)

    # 2) write unified access state
    _set_user_access(user_id, plan, "stripe", ref)

    # 3) write payment ledger
    _ledger_add(
        user_id=user_id,
        source="stripe",
        ref=ref,
        plan=plan,
        amount=amount,
        currency=currency,
        status="paid",
    )

def downgrade_user_with_ledger_by_email(email: str, ref: str = "stripe_cancel"):
    user = get_user_by_email(email)
    if not user:
        return

    user_id = user["user_id"]

    downgrade_user_by_email(email)
    _set_user_access(user_id, "free", "stripe", ref)



# =========================================================
# EMAIL JOB
# =========================================================

EMAIL_FETCH_TIMEOUT_SEC = int(os.environ.get("EMAIL_FETCH_TIMEOUT_SEC", "60"))
EMAIL_BUILD_POOL_TIMEOUT_SEC = int(os.environ.get("EMAIL_BUILD_POOL_TIMEOUT_SEC", "60"))
EMAIL_SEND_TIMEOUT_SEC = int(os.environ.get("EMAIL_SEND_TIMEOUT_SEC", "60"))

# SMTP connection reuse (Render speed fix)
SMTP_REUSE_TTL_SEC = int(os.environ.get("SMTP_REUSE_TTL_SEC", "240"))  # 4 minutes

_SMTP_LOCK = threading.RLock()
_SMTP_CONN = None          # cached SMTP connection
_SMTP_CONN_IS_SSL = None   # bool
_SMTP_CONN_TS = 0.0        # last-used timestamp


async def _to_thread_with_timeout(fn, timeout_sec: int, *args, **kwargs):
    return await asyncio.wait_for(asyncio.to_thread(fn, *args, **kwargs), timeout=timeout_sec)

async def _send_email_async(timeout_sec: int, *args, **kwargs) -> bool:
    """
    Runs send_email() in a worker thread with a hard timeout so SMTP/network stalls
    can't block the Telegram event loop (Render lag fix).
    """
    try:
        return bool(await _to_thread_with_timeout(send_email, timeout_sec, *args, **kwargs))
    except asyncio.TimeoutError:
        return False
    except Exception:
        return False

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

        # Trade-signal emails may be notify_on-gated,
        # but Big-Move Alerts should go to anyone who has an email saved.

        try:
            users_notify = list_users_notify_on()
        except Exception as e:
            logger.exception("list_users_notify_on failed: %s", e)
            users_notify = []

        try:
            users_bigmove = list_users_with_email()
        except Exception as e:
            logger.exception("list_users_with_email failed: %s", e)
            users_bigmove = []

        if not users_notify and not users_bigmove:
            return

        # TIMEOUT-PROTECTED fetch (prevents lock being held forever)
        try:
            best_fut = await _to_thread_with_timeout(fetch_futures_tickers, EMAIL_FETCH_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            return
        except Exception:
            return

        if not best_fut:
            return

        # -----------------------------------------------------
        # Rule 2: Volume relative filter base line (killer rule)
        # -----------------------------------------------------
        _all_vols = []
        for _b, _mv in (best_fut or {}).items():
            try:
                _all_vols.append(float(getattr(_mv, "fut_vol_usd", 0.0) or 0.0))
            except Exception:
                pass
        MARKET_VOL_MEDIAN_USD = _median(_all_vols)

        # -----------------------------------------------------
        # Big-Move Alert Emails (independent of full trade setups)
        # Trigger: |4H| >= user.bigmove_alert_4h  OR  |1H| >= user.bigmove_alert_1h
        # Volume gate: vol24 >= user.bigmove_min_vol_usd (default 10M)
        # Defaults: 1H=7.5%, 4H=15%
        # -----------------------------------------------------
        for u in (users_bigmove or []):
            # Pro/trial-only email features
            try:
                uid = int(u.get("user_id") or u.get("id") or 0)
            except Exception:
                uid = 0
            if uid and (not user_has_pro(uid)):
                continue

            tz = timezone.utc
            uid = 0
            try:
                try:
                    uid = int(u.get("user_id") or u.get("id") or 0)
                except Exception:
                    uid = 0
                if not uid:
                    continue

                uu = get_user(uid) or {}

                tz_name = str((uu or {}).get("tz") or "UTC")
                try:
                    tz = ZoneInfo(tz_name)
                except Exception:
                    tz = timezone.utc

                # Respect per-user ON/OFF
                on = int(uu.get("bigmove_alert_on", 1) or 0)
                if not on:
                    _LAST_BIGMOVE_DECISION[uid] = {
                        "status": "SKIP",
                        "when": datetime.now(tz).isoformat(timespec="seconds"),
                        "reasons": ["bigmove_alert_off"],
                    }
                    continue

                try:
                    p4 = float(uu.get("bigmove_alert_4h", 15.0) or 15.0)
                    p1 = float(uu.get("bigmove_alert_1h", 7.5) or 7.5)
                except Exception:
                    p4, p1 = 15.0, 7.5

                try:
                    min_vol = float(uu.get("bigmove_min_vol_usd", 10_000_000) or 10_000_000)
                except Exception:
                    min_vol = 10_000_000.0

                candidates = _bigmove_candidates(best_fut, p4=p4, p1=p1, min_vol_usd=min_vol, max_items=12)

                # Debug counts from the SAME dataset used for bigmove (same field-name logic)
                def _pick_pct(_mv, _keys) -> float:
                    for _k in _keys:
                        try:
                            _v = getattr(_mv, _k, None)
                            if _v is None:
                                continue
                            return float(_v or 0.0)
                        except Exception:
                            continue
                    return 0.0

                try:
                    bm_any_4h = sum(
                        1 for _sym, _mv in (best_fut or {}).items()
                        if abs(_pick_pct(_mv, ["ch4", "pct_4h", "change_4h", "chg_4h", "percentage_4h", "p4", "h4"])) >= float(p4)
                    )
                except Exception:
                    bm_any_4h = -1

                try:
                    bm_any_1h = sum(
                        1 for _sym, _mv in (best_fut or {}).items()
                        if abs(_pick_pct(_mv, ["ch1", "pct_1h", "change_1h", "chg_1h", "percentage_1h", "p1", "h1"])) >= float(p1)
                    )
                except Exception:
                    bm_any_1h = -1

                if not candidates:
                    _LAST_BIGMOVE_DECISION[uid] = {
                        "status": "SKIP",
                        "when": datetime.now(tz).isoformat(timespec="seconds"),
                        "reasons": [
                            f"no_candidates (p4={p4}, p1={p1})",
                            f"debug_raw_hits:4h={bm_any_4h},1h={bm_any_1h}",
                        ],
                    }
                    continue

                # Volume gate (default 10M) + remove ones emailed recently (per symbol + direction)
                filtered = []
                for c in candidates:
                    try:
                        vol = float(c.get("vol", 0.0) or 0.0)
                    except Exception:
                        vol = 0.0

                    if vol > 0.0 and vol < float(min_vol):
                        continue

                    try:
                        if not bigmove_recently_emailed(uid, c["symbol"], c["direction"]):
                            filtered.append(c)
                    except Exception:
                        filtered.append(c)

                if not filtered:
                    _LAST_BIGMOVE_DECISION[uid] = {
                        "status": "SKIP",
                        "when": datetime.now(tz).isoformat(timespec="seconds"),
                        "reasons": [f"no_candidates_after_volume_or_cooldown (min_vol={min_vol/1e6:.1f}M)"],
                    }
                    continue

                # Build email body
                lines = []
                lines.append("⚡ PulseFutures — BIG MOVE ALERT")
                lines.append(HDR)
                lines.append(f"Triggers: |4H| ≥ {p4:.1f}%  OR  |1H| ≥ {p1:.1f}%")
                lines.append(f"Min Vol (24H): {min_vol/1e6:.1f}M")
                lines.append("")

                top = filtered[0]
                top_sym = top["symbol"]
                top_dir = "UP" if top.get("direction") == "UP" else "DOWN"
                top_move = top["ch4"] if abs(top.get("ch4", 0.0)) >= abs(top.get("ch1", 0.0)) else top.get("ch1", 0.0)
                top_tf = "4H" if abs(top.get("ch4", 0.0)) >= abs(top.get("ch1", 0.0)) else "1H"

                subject = f"⚡ Big Move Alert • {top_sym} {top_dir} • {top_tf} {top_move:+.0f}%"
                if len(filtered) > 1:
                    subject += f" (+{len(filtered)-1} more)"

                for c in filtered[:8]:
                    sym = c["symbol"]
                    ch4 = float(c.get("ch4", 0.0) or 0.0)
                    ch1 = float(c.get("ch1", 0.0) or 0.0)
                    vol = float(c.get("vol", 0.0) or 0.0)
                    arrow = "🟢" if c.get("direction") == "UP" else "🔴"

                    lines.append(f"{arrow} {sym}: 4H {ch4:+.0f}% | 1H {ch1:+.0f}% | Vol ~{vol/1e6:.1f}M")
                    lines.append(f"Chart: https://www.tradingview.com/chart/?symbol=BYBIT:{sym}USDT.P")
                    lines.append("")

                body = "\n".join(lines).strip()

                _LAST_BIGMOVE_DECISION[uid] = {
                    "status": "TRY_SEND",
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                    "reasons": [
                        f"candidates={len(filtered)}",
                        f"p4={p4}",
                        f"p1={p1}",
                        f"min_vol={min_vol}",
                    ],
                }

                ok = await _send_email_async(
                    EMAIL_SEND_TIMEOUT_SEC,
                    subject,
                    body,
                    user_id_for_debug=uid,
                    enforce_trade_window=False
                )


                _LAST_BIGMOVE_DECISION[uid] = {
                    "status": "SENT" if ok else "FAIL",
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                    "reasons": ["ok"] if ok else [_LAST_SMTP_ERROR.get(uid, "send_email_failed")],
                }

                if ok:
                    for c in filtered[:8]:
                        try:
                            mark_bigmove_emailed(uid, c["symbol"], c["direction"])
                        except Exception:
                            pass

            except Exception as e:
                logger.exception("Big-move alert failed for uid=%s: %s", uid, e)
                _LAST_BIGMOVE_DECISION[uid] = {
                    "status": "ERROR",
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                    "reasons": [f"{type(e).__name__}: {e}"],
                }
                continue
        

        # -----------------------------------------------------
        setups_by_session: Dict[str, List[Setup]] = {}
        for sess_name in ["NY", "LON", "ASIA"]:
            try:
                pool = await asyncio.wait_for(asyncio.to_thread(_run_coro_in_thread, build_priority_pool(best_fut, sess_name, mode="email", scan_profile=str(DEFAULT_SCAN_PROFILE), uid=uid)), timeout=EMAIL_BUILD_POOL_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                pool = {"setups": []}
            except Exception:
                pool = {"setups": []}

            setups = (pool.get("setups", []) or [])[:max(EMAIL_SETUPS_N * 3, 9)]

            # Rule 3: Priority override (Directional Leaders/Losers first)
            if EMAIL_PRIORITY_OVERRIDE_ON:
                pri = _email_priority_bases(best_fut, directional_take=12)
                setups = sorted(
                    setups,
                    key=lambda s: (0 if str(getattr(s, "symbol", "")).upper() in pri else 1)
                )

            setups_by_session[sess_name] = setups

            for s in setups:
                try:
                    db_insert_signal(s)
                except Exception:
                    pass

        # -----------------------------------------------------
        # Per-user send / skip logic
        # -----------------------------------------------------
        for user in (users_notify or []):
            # ✅ Robust uid resolution (supports either user_id or id)
            try:
                uid = int(user.get("user_id") or user.get("id") or 0)
            except Exception:
                uid = 0
            if not uid:
                continue

            # ✅ Robust tz resolution (never crash job)
            tz_name = str(user.get("tz") or "UTC")
            try:
                tz = ZoneInfo(tz_name)
            except Exception:
                tz = timezone.utc
                tz_name = "UTC"

            # ✅ Ensure session logic never crashes job
            try:
                sess = in_session_now(user)
            except Exception as e:
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reasons": [f"in_session_now_failed ({type(e).__name__})"],
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
                continue

            # If user is NOT in an enabled session right now, skip
            if not sess:
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reasons": ["not_in_enabled_session"],
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
                continue

            setups_all = setups_by_session.get(str(sess.get("name") or ""), []) or []

            if not setups_all:
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reasons": [f"no_setups_generated_for_session ({sess.get('name')})"],
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
                continue

            # state init must not crash job
            try:
                st = email_state_get(uid)
            except Exception as e:
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "ERROR",
                    "reasons": [f"email_state_get_failed ({type(e).__name__})"],
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
                continue

            # Reset session state if session_key changed
            try:
                if str(st.get("session_key")) != str(sess.get("session_key")):
                    email_state_set(uid, session_key=str(sess.get("session_key")), sent_count=0, last_email_ts=0.0)
                    st = email_state_get(uid)
            except Exception:
                pass

            # Safe defaults (never KeyError)
            try:
                max_emails = int(user.get("max_emails_per_session", DEFAULT_MAX_EMAILS_PER_SESSION))
            except Exception:
                max_emails = int(DEFAULT_MAX_EMAILS_PER_SESSION)

            try:
                gap_min = int(user.get("email_gap_min", DEFAULT_MIN_EMAIL_GAP_MIN))
            except Exception:
                gap_min = int(DEFAULT_MIN_EMAIL_GAP_MIN)

            gap_sec = max(0, gap_min) * 60

            # Daily cap
            day_local = datetime.now(tz).date().isoformat()
            try:
                sent_today = _email_daily_get(uid, day_local)
            except Exception:
                sent_today = 0

            try:
                day_cap = int(user.get("max_emails_per_day", DEFAULT_MAX_EMAILS_PER_DAY))
            except Exception:
                day_cap = int(DEFAULT_MAX_EMAILS_PER_DAY)

            if day_cap > 0 and sent_today >= day_cap:
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reasons": [f"daily_email_cap_reached ({sent_today}/{day_cap})"],
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
                continue

            # Session cap
            try:
                sent_in_session = int(st.get("sent_count", 0) or 0)
            except Exception:
                sent_in_session = 0

            if max_emails > 0 and sent_in_session >= max_emails:
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reasons": [f"session_email_cap_reached ({sent_in_session}/{max_emails})"],
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
                continue

            # Gap
            now_ts = time.time()
            try:
                last_ts = float(st.get("last_email_ts", 0.0) or 0.0)
            except Exception:
                last_ts = 0.0

            if gap_sec > 0 and (now_ts - last_ts) < gap_sec:
                remain = int(gap_sec - (now_ts - last_ts))
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reasons": [f"email_gap_active (remain {remain}s)"],
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
                continue

            # ---------------------------
            # Your existing filter + pick logic
            # ---------------------------
            min_conf = SESSION_MIN_CONF.get(sess["name"], 78)
            min_rr = SESSION_MIN_RR_TP3.get(sess["name"], 2.2)

            # -------------------------------------------------
            # Per-user EMAIL filter parameters (safe defaults)
            # -------------------------------------------------
            try:
                email_abs_vol_min = float((user.get("email_abs_vol_min_usd") if isinstance(user, dict) else None) or EMAIL_ABS_VOL_USD_MIN)
            except Exception:
                email_abs_vol_min = float(EMAIL_ABS_VOL_USD_MIN)

            try:
                email_rel_vol_min_mult = float((user.get("email_rel_vol_min_mult") if isinstance(user, dict) else None) or EMAIL_REL_VOL_MIN_MULT)
            except Exception:
                email_rel_vol_min_mult = float(EMAIL_REL_VOL_MIN_MULT)

            try:
                confirm_15m_abs_min = float((user.get("confirm_15m_abs_min") if isinstance(user, dict) else None) or CONFIRM_15M_ABS_MIN)
            except Exception:
                confirm_15m_abs_min = float(CONFIRM_15M_ABS_MIN)

            try:
                early_1h_abs_min = float((user.get("early_1h_abs_min") if isinstance(user, dict) else None) or EARLY_1H_ABS_MIN)
            except Exception:
                early_1h_abs_min = float(EARLY_1H_ABS_MIN)

            # Extra strictness for EMAIL (but not "never send"): allow user override, else keep conservative defaults.
            try:
                email_early_min_ch15_abs = float((user.get("email_early_min_ch15_abs") if isinstance(user, dict) else None) or EMAIL_EARLY_MIN_CH15_ABS)
            except Exception:
                email_early_min_ch15_abs = float(EMAIL_EARLY_MIN_CH15_ABS)

            confirmed: List[Setup] = []
            early: List[Setup] = []
            skip_reasons_counter = Counter()

            for s in setups_all:
                base = str(getattr(s, "symbol", "") or "").upper().strip()

                # For /screen, be slightly more permissive so the bot doesn't feel "dead" in slow hours.
                # Email stays at the stricter session floors.
                eff_min_conf = int(min_conf)
                eff_min_rr = float(min_rr)
                if s.conf < eff_min_conf:
                    skip_reasons_counter["below_session_conf_floor"] += 1
                    try:
                        mv = (best_fut or {}).get(base)
                        if mv is not None:
                            _rej("below_session_conf_floor", base, mv, f"conf={int(s.conf)} min={int(eff_min_conf)}")
                    except Exception:
                        pass
                    continue

                rr3 = rr_to_tp(float(s.entry), float(s.sl), float(s.tp3))
                if float(rr3) < float(eff_min_rr):
                    skip_reasons_counter["below_session_rr_floor"] += 1
                    try:
                        mv = (best_fut or {}).get(base)
                        if mv is not None:
                            _rej("below_session_rr_floor", base, mv, f"rr3={float(rr3):.2f} min={float(eff_min_rr):.2f}")
                    except Exception:
                        pass
                    continue

                # =========================================================
                # Rule 2: Volume relative filter (killer rule)
                # =========================================================
                vol_usd = 0.0
                try:
                    vol_usd = float(_best_fut_vol_usd(best_fut, getattr(s, "symbol", "")) or 0.0)
                except Exception:
                    vol_usd = 0.0

                if vol_usd <= 0.0:
                    try:
                        vol_usd = float(getattr(s, "fut_vol_usd", 0.0) or 0.0)
                    except Exception:
                        vol_usd = 0.0

                # Per-user email volume floors (stricter than /screen, but configurable)
                try:
                    abs_min = float(user.get("email_abs_vol_min_usd", user.get("email_abs_vol_min", 5_000_000)) or 5_000_000)
                except Exception:
                    abs_min = 5_000_000.0

                try:
                    rel_mult = float(user.get("email_rel_vol_min_mult", user.get("email_rel_vol_mult", 0.80)) or 0.80)
                except Exception:
                    rel_mult = 0.80

                if vol_usd > 0.0 and vol_usd < abs_min:
                    skip_reasons_counter["email_vol_abs_too_low"] += 1
                    continue

                if vol_usd > 0.0 and float(MARKET_VOL_MEDIAN_USD or 0.0) > 0:
                    rel = vol_usd / float(MARKET_VOL_MEDIAN_USD)
                    if rel < float(rel_mult):
                        skip_reasons_counter["email_vol_rel_too_low"] += 1
                        continue

                # =========================================================
                # Rule 1: Minimum momentum gate for EMAILS
                # =========================================================
                ch15 = _safe_float(getattr(s, "ch15", 0.0), 0.0)
                is_confirm_15m = abs(float(ch15)) >= float(confirm_15m_abs_min)

                if is_confirm_15m:
                    confirmed.append(s)
                else:
                    if abs(float(s.ch1)) < float(early_1h_abs_min):
                        skip_reasons_counter["early_gate_ch1_not_strong"] += 1
                        continue
                    if abs(float(ch15)) < float(email_early_min_ch15_abs):
                        skip_reasons_counter["early_gate_15m_too_weak"] += 1
                        continue
                    if s.conf < (min_conf + EARLY_EMAIL_EXTRA_CONF):
                        skip_reasons_counter["early_gate_conf_not_high_enough"] += 1
                        continue
                    early.append(s)

            confirmed = sorted(confirmed, key=lambda x: x.conf, reverse=True)
            early = sorted(early, key=lambda x: x.conf, reverse=True)

            picks: List[Setup] = []
            for s in confirmed:
                if len(picks) >= int(EMAIL_SETUPS_N):
                    break
                picks.append(s)

            fill_left = int(EMAIL_SETUPS_N) - len(picks)
            if fill_left > 0 and int(EARLY_EMAIL_MAX_FILL) > 0:
                allow_early = min(int(EARLY_EMAIL_MAX_FILL), fill_left)
                picks.extend(early[:allow_early])

            if not picks:
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reasons": ["no_setups_after_filters"] + (
                        [f"top_reasons={dict(skip_reasons_counter.most_common(5))}"]
                        if skip_reasons_counter else []
                    ),
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
                continue

            chosen_list: List[Setup] = []
            cooldown_blocked = 0
            flip_blocked = 0

            for s in picks:
                if len(chosen_list) >= int(EMAIL_SETUPS_N):
                    break

                sym = str(getattr(s, "symbol", "")).upper()
                side = str(getattr(s, "side", "")).upper()
                sess_name = str(sess["name"])

                if symbol_flip_guard_active(uid, sym, side, sess_name):
                    flip_blocked += 1
                    continue

                if symbol_recently_emailed(uid, sym, side, sess_name):
                    cooldown_blocked += 1
                    continue

                chosen_list.append(s)

            if not chosen_list:
                reasons = []
                if cooldown_blocked:
                    reasons.append(f"cooldown_blocked={cooldown_blocked}")
                if flip_blocked:
                    reasons.append(f"flip_guard_blocked={flip_blocked}")
                reasons.append("all_candidates_blocked")

                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SKIP",
                    "reasons": reasons,
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }
                continue

            # Send ONE email containing MULTIPLE setups (timeout-protected)
            try:
                ok = await asyncio.wait_for(
                    asyncio.to_thread(send_email_alert_multi, user, sess, chosen_list, best_fut),
                    timeout=EMAIL_SEND_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                ok = False
                try:
                    _LAST_SMTP_ERROR[uid] = f"timeout_after_{int(EMAIL_SEND_TIMEOUT_SEC)}s"
                except Exception:
                    pass
            except Exception as e:
                ok = False
                logger.exception("send_email_alert_multi failed for uid=%s: %s", uid, e)
                try:
                    _LAST_SMTP_ERROR[uid] = f"{type(e).__name__}: {e}"
                except Exception:
                    pass

            if ok:
                try:
                    email_state_set(uid, last_email_ts=time.time(), sent_count=int(st.get("sent_count", 0) or 0) + 1)
                except Exception:
                    pass

                try:
                    _email_daily_inc(uid, day_local)
                except Exception:
                    pass

                for s in chosen_list:
                    try:
                        mark_symbol_emailed(uid, s.symbol, s.side, sess["name"])
                    except Exception:
                        pass

                _LAST_EMAIL_DECISION[uid] = {
                    "status": "SENT",
                    "picked": ", ".join([f"{s.side} {s.symbol} conf={s.conf}" for s in chosen_list]),
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                    "reasons": ["passed_filters_multi"],
                }
            else:
                _LAST_EMAIL_DECISION[uid] = {
                    "status": "ERROR",
                    "reasons": ["send_email_failed_or_timeout", _LAST_SMTP_ERROR.get(uid, "unknown_error")],
                    "when": datetime.now(tz).isoformat(timespec="seconds"),
                }

async def email_decision_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id

    scan = _LAST_EMAIL_DECISION.get(uid) or {}
    bigm = _LAST_BIGMOVE_DECISION.get(uid) or {}

    # If nothing recorded
    if not scan and not bigm:
        await update.message.reply_text("No email decision recorded yet.")
        return

    lines = []
    lines.append("📧 Email Decisions")

    if bigm:
        lines.append("")
        lines.append("⚡ Big-Move Alert Decision")
        lines.append(f"Status: {bigm.get('status')}")
        lines.append(f"When: {bigm.get('when') or _fmt_when(bigm.get('ts'))}")
        rs = bigm.get("reasons") or []
        if rs:
            lines.append("Reasons:\n- " + "\n- ".join(rs))

    if scan:
        lines.append("")
        lines.append("🧠 Market Scan Decision")
        lines.append(f"Status: {scan.get('status')}")
        lines.append(f"When: {scan.get('when') or _fmt_when(scan.get('ts'))}")
        rs = scan.get("reasons") or scan.get("reason")
        if isinstance(rs, list):
            lines.append("Reasons:\n- " + "\n- ".join(rs))
        elif rs:
            lines.append(f"Reason: {rs}")

    await update.message.reply_text("\n".join(lines).strip())


# =========================================================
# /why (debug why no setups)
# =========================================================
async def why_no_setups_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    txt = _reject_report_for_uid(uid)
    await update.message.reply_text(txt)



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

        

# =========================================================
# USDT (SEMI-AUTO) + ADMIN PAYMENTS/ACCESS MANAGEMENT
# =========================================================
import re
import time

TXID_REGEX = re.compile(r"^[A-Fa-f0-9]{64}$")

USDT_NETWORK = os.getenv("USDT_NETWORK", "TRC20").strip().upper()
USDT_RECEIVE_ADDRESS = os.getenv("USDT_RECEIVE_ADDRESS", "").strip()

USDT_STANDARD_PRICE = float(os.getenv("USDT_STANDARD_PRICE", "45"))
USDT_PRO_PRICE = float(os.getenv("USDT_PRO_PRICE", "99"))

def _now() -> float:
    return float(time.time())

def _is_admin(update: Update) -> bool:
    try:
        return is_admin_user(update.effective_user.id)
    except Exception:
        return False

def _db():
    return sqlite3.connect(DB_PATH)

def _user_ident_str(uid: int) -> str:
    return f"uid={uid}"

def _ledger_add(user_id: int, source: str, ref: str, plan: str, amount: float, currency: str, status: str):
    with _db() as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO payments_ledger (user_id, source, ref, plan, amount, currency, status, created_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (int(user_id), source, str(ref), str(plan), float(amount), str(currency), str(status), _now()))
        con.commit()

def _set_user_access(user_id: int, plan: str, source: str, ref: str):
    update_user(
        int(user_id),
        plan=str(plan),
        access_source=str(source),
        access_ref=str(ref),
        access_updated_ts=_now(),
    )


def _ensure_trial_state(user: dict, uid: Optional[int] = None) -> dict:
    """Ensure the user has a 7-day trial recorded, and downgrade to free (locked) when expired."""
    try:
        if uid is not None and is_admin_user(int(uid)):
            return user
    except Exception:
        pass

    if not user:
        return user

    now = time.time()
    plan = str(user.get("plan") or "free").strip().lower()
    trial_until = float(user.get("trial_until", 0) or 0.0)
    trial_start = float(user.get("trial_start_ts", 0) or 0.0)

    # Initialize trial if user has never had one
    if trial_until <= 0:
        trial_start = now
        trial_until = now + (7 * 86400)
        update_user(int(user.get("user_id") or uid or 0), plan="trial", trial_start_ts=trial_start, trial_until=trial_until)
        user["plan"] = "trial"
        user["trial_start_ts"] = trial_start
        user["trial_until"] = trial_until
        return user

    # Keep trial active while inside the window
    if now <= trial_until:
        if plan != "trial":
            update_user(int(user.get("user_id") or uid or 0), plan="trial")
            user["plan"] = "trial"
        return user

    # Trial expired => lock user (free = locked)
    if plan != "free":
        update_user(int(user.get("user_id") or uid or 0), plan="free")
        user["plan"] = "free"
    return user


def has_active_access(user: dict, uid: Optional[int] = None) -> bool:
    """Access rules:
    - Admin: always allowed
    - Trial: allowed for 7 days from first seen
    - Paid plans (standard/pro): allowed (optionally with plan_expires if you set it)
    - Free: LOCKED (after trial)
    """
    try:
        if uid is not None and is_admin_user(int(uid)):
            return True
    except Exception:
        pass

    if not user:
        return False

    # Ensure trial is initialized / downgraded when needed
    user = _ensure_trial_state(user, uid=uid)

    now = time.time()
    plan = str(user.get("plan") or "free").strip().lower()

    if plan in ("standard", "pro"):
        # If you use plan_expires, enforce it (0/None => no expiry)
        try:
            exp = float(user.get("plan_expires", 0) or 0.0)
            if exp > 0 and now > exp:
                update_user(int(user.get("user_id") or uid or 0), plan="free")
                return False
        except Exception:
            pass
        return True

    if plan == "trial":
        try:
            return now <= float(user.get("trial_until", 0) or 0.0)
        except Exception:
            return False

    # free = locked
    return False

def _usdt_payment_row_by_txid(txid: str):
    with _db() as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM usdt_payments WHERE txid = ?", (txid,))
        r = cur.fetchone()
        return dict(r) if r else None

def _usdt_insert_pending(user_id: int, username: str, txid: str, plan: str):
    with _db() as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO usdt_payments (telegram_id, username, txid, plan, status, created_ts)
            VALUES (?, ?, ?, ?, 'PENDING', ?)
        """, (int(user_id), (username or ""), txid, plan, _now()))
        con.commit()

def _usdt_set_status(txid: str, status: str, decided_by: int, note: str = ""):
    with _db() as con:
        cur = con.cursor()
        cur.execute("""
            UPDATE usdt_payments
            SET status = ?, decided_ts = ?, decided_by = ?, note = ?
            WHERE txid = ?
        """, (status, _now(), int(decided_by), (note or ""), txid))
        con.commit()

# ---------------- USER COMMANDS ----------------

async def usdt_info_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    addr = USDT_RECEIVE_ADDRESS or "(not set)"
    await update.message.reply_text(
        "USDT Payment (Semi-Auto)\n\n"
        f"Network: USDT ({USDT_NETWORK})\n"
        f"Address: {addr}\n\n"
        f"Standard: {USDT_STANDARD_PRICE:.0f} USDT\n"
        f"Pro: {USDT_PRO_PRICE:.0f} USDT\n\n"
        "After payment submit:\n"
        "/usdt_paid <TXID> <standard|pro>\n\n"
        "Example:\n"
        "/usdt_paid 7f3a...c9 standard\n\n"
        "Note: USDT payments are final. Access after admin approval."
    )

async def usdt_paid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("Usage:\n/usdt_paid <TXID> <standard|pro>")
        return

    txid = context.args[0].strip()
    plan = context.args[1].strip().lower()

    if plan not in ("standard", "pro"):
        await update.message.reply_text("❌ Plan must be: standard or pro\nExample: /usdt_paid <TXID> standard")
        return

    if not TXID_REGEX.match(txid):
        await update.message.reply_text("❌ TXID looks invalid.\nIt must be a 64-char hex hash.")
        return

    # prevent duplicates
    existing = _usdt_payment_row_by_txid(txid)
    if existing:
        await update.message.reply_text(f"⚠️ This TXID is already in the system (status: {existing['status']}).")
        return

    uid = int(update.effective_user.id)
    uname = update.effective_user.username or ""
    try:
        _usdt_insert_pending(uid, uname, txid, plan)
    except sqlite3.IntegrityError:
        await update.message.reply_text("⚠️ This TXID is already used.")
        return
    except Exception as e:
        logger.exception("usdt_paid_cmd failed")
        await update.message.reply_text(f"❌ Failed to save. {type(e).__name__}: {e}")
        return

    await update.message.reply_text(
        "✅ USDT payment submitted.\n"
        "Status: PENDING admin approval.\n\n"
        "If you made a mistake, contact /support."
    )

    # Notify admins (send to each admin user id)
    try:
        for admin_id in (ADMIN_USER_IDS or set()):
            await context.bot.send_message(
                chat_id=int(admin_id),
                text=(
                    "🧾 New USDT Payment Request\n\n"
                    f"User: @{uname or '(no username)'} ({uid})\n"
                    f"Plan: {plan.upper()}\n"
                    f"TXID: {txid}\n\n"
                    f"Approve: /usdt_approve {txid}\n"
                    f"Reject:  /usdt_reject {txid} <reason>"
                )
            )
    except Exception:
        # don't block user success if admin notify fails
        logger.exception("Failed to notify admins about USDT request")

# ---------------- ADMIN COMMANDS ----------------

async def usdt_pending_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        await update.message.reply_text("❌ Admin only.")
        return

    with _db() as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("""
            SELECT txid, telegram_id, username, plan, created_ts
            FROM usdt_payments
            WHERE status='PENDING'
            ORDER BY created_ts DESC
            LIMIT 20
        """)
        rows = cur.fetchall()

    if not rows:
        await update.message.reply_text("✅ No pending USDT requests.")
        return

    lines = ["Pending USDT requests (latest 20):\n"]
    for r in rows:
        lines.append(f"- {r['plan'].upper()} | @{r['username'] or 'n/a'} ({r['telegram_id']}) | {r['txid']}")
    lines.append("\nApprove with: /usdt_approve <TXID>")
    await update.message.reply_text("\n".join(lines))

async def usdt_approve_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        await update.message.reply_text("❌ Admin only.")
        return

    if not context.args:
        await update.message.reply_text("Usage: /usdt_approve <TXID>")
        return

    txid = context.args[0].strip()
    if not TXID_REGEX.match(txid):
        await update.message.reply_text("❌ Invalid TXID format (must be 64-char hex).")
        return

    row = _usdt_payment_row_by_txid(txid)
    if not row:
        await update.message.reply_text("❌ TXID not found in DB. Ask user to re-submit with /usdt_paid.")
        return

    if row["status"] == "APPROVED":
        await update.message.reply_text("⚠️ Already approved.")
        return
    if row["status"] == "REJECTED":
        await update.message.reply_text("⚠️ This TXID was rejected earlier.")
        return

    plan = row["plan"].lower()
    uid = int(row["telegram_id"])

    # Grant access + ledger
    amount = USDT_STANDARD_PRICE if plan == "standard" else USDT_PRO_PRICE
    _set_user_access(uid, plan, "usdt", txid)
    _ledger_add(uid, "usdt", txid, plan, amount, "USDT", "paid")
    _usdt_set_status(txid, "APPROVED", update.effective_user.id, note="approved")

    await update.message.reply_text(f"✅ Approved. Access granted: {plan.upper()} to user {uid}.")
    try:
        await context.bot.send_message(chat_id=uid, text=f"✅ Payment approved. Your plan is now: {plan.upper()}")
    except Exception:
        pass

async def usdt_reject_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        await update.message.reply_text("❌ Admin only.")
        return

    if not context.args:
        await update.message.reply_text("Usage: /usdt_reject <TXID> <reason(optional)>")
        return

    txid = context.args[0].strip()
    reason = " ".join(context.args[1:]).strip() if len(context.args) > 1 else "rejected"
    if not TXID_REGEX.match(txid):
        await update.message.reply_text("❌ Invalid TXID format.")
        return

    row = _usdt_payment_row_by_txid(txid)
    if not row:
        await update.message.reply_text("❌ TXID not found.")
        return

    if row["status"] != "PENDING":
        await update.message.reply_text(f"⚠️ Cannot reject (current status: {row['status']}).")
        return

    _usdt_set_status(txid, "REJECTED", update.effective_user.id, note=reason)
    await update.message.reply_text("✅ Rejected.")
    try:
        await context.bot.send_message(chat_id=int(row["telegram_id"]), text=f"❌ USDT payment rejected: {reason}")
    except Exception:
        pass

# ---------------- ADMIN: USERS / ACCESS / PAYMENTS ----------------

async def admin_user_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        await update.message.reply_text("❌ Admin only.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /admin_user <telegram_id>")
        return
    try:
        uid = int(context.args[0])
    except Exception:
        await update.message.reply_text("❌ telegram_id must be a number.")
        return

    user = get_user(uid)
    plan = user.get("plan")
    src = user.get("access_source", "")
    ref = user.get("access_ref", "")
    upd = user.get("access_updated_ts", 0)

    # latest payments
    with _db() as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("""
            SELECT source, ref, plan, amount, currency, status, created_ts
            FROM payments_ledger
            WHERE user_id = ?
            ORDER BY created_ts DESC
            LIMIT 5
        """, (uid,))
        pays = cur.fetchall()

    lines = [
        f"User: {uid}",
        f"Plan: {plan}",
        f"Access source/ref: {src} / {ref}",
        f"Access updated: {datetime.utcfromtimestamp(upd).isoformat()+'Z' if upd else 'n/a'}",
        f"Email: {user.get('email_to','') or ''}",
        "",
        "Last payments:"
    ]
    if pays:
        for p in pays:
            lines.append(f"- {p['source']} | {p['plan']} | {p['amount']} {p['currency']} | {p['status']} | {p['ref']}")
    else:
        lines.append("- (none)")
    await update.message.reply_text("\n".join(lines))

def _table_columns(conn, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(r[1]) for r in rows}

def _first_existing_col(cols: set[str], candidates: list[str]):
    for c in candidates:
        if c in cols:
            return c
    return None

async def admin_users_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_admin_user(uid):
        return

    conn = _db()
    cols = _table_columns(conn, "users")

    id_col = _first_existing_col(cols, ["user_id", "id"]) or "id"
    plan_col = _first_existing_col(cols, ["plan", "tier", "subscription_plan"])
    email_col = _first_existing_col(cols, ["email", "user_email"])
    tz_col = _first_existing_col(cols, ["tz", "timezone"])

    sql = f"""
        SELECT
            {id_col} AS uid,
            {plan_col if plan_col else 'NULL'} AS plan,
            {email_col if email_col else 'NULL'} AS email,
            {tz_col if tz_col else 'NULL'} AS tz
        FROM users
        ORDER BY uid DESC
        LIMIT 50
    """

    rows = conn.execute(sql).fetchall()

    lines = ["👤 Admin — Users", HDR]
    for uid, plan, email, tz in rows:
        lines.append(
            f"• {uid}"
            f" | {plan or ''}"
            f" | {email or ''}"
            f" | tz:{tz or ''}"
        )

    await update.message.reply_text("\n".join(lines))


async def admin_revoke_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        await update.message.reply_text("❌ Admin only.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /admin_revoke <telegram_id>")
        return
    uid = int(context.args[0])
    _set_user_access(uid, "free", "manual", "revoked")
    await update.message.reply_text(f"✅ Access revoked. User {uid} set to FREE.")

async def admin_payments_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        await update.message.reply_text("❌ Admin only.")
        return
    n = 20
    if context.args and context.args[0].isdigit():
        n = max(1, min(50, int(context.args[0])))

    with _db() as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("""
            SELECT user_id, source, plan, amount, currency, status, ref, created_ts
            FROM payments_ledger
            ORDER BY created_ts DESC
            LIMIT ?
        """, (n,))
        rows = cur.fetchall()

    if not rows:
        await update.message.reply_text("No payments in ledger.")
        return

    lines = [f"Latest payments (max {n}):"]
    for r in rows:
        ts = datetime.utcfromtimestamp(r["created_ts"]).isoformat() + "Z"
        lines.append(f"- {ts} | uid={r['user_id']} | {r['source']} | {r['plan']} | {r['amount']} {r['currency']} | {r['status']} | {r['ref']}")
    await update.message.reply_text("\n".join(lines))

async def manage_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Stripe customer portal link (if you have it).
    If you are using Payment Links only and not portal, keep as-is or disable this command.
    """
    user = get_user(update.effective_user.id)
    if user.get("plan") not in ("standard", "pro"):
        await update.message.reply_text("No active subscription.")
        return

    # If you still use Stripe Customer Portal:
    if user.get("email_to"):
        await update.message.reply_text(create_customer_portal(user["email_to"]))
        return

    await update.message.reply_text("❌ No email on file for subscription management.")


async def upgrade_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Upgrade opens the billing menu
    return await billing_cmd(update, context)


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
    
    app = Application.builder().token(TOKEN).post_init(_post_init).concurrent_updates(True).build()

    # Global access + Pro gating (runs before any other command handler)
    app.add_handler(MessageHandler(filters.COMMAND, _command_guard), group=-1)
    app.add_error_handler(error_handler)

    # ================= Handlers =================
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("commands", commands_cmd))
    app.add_handler(CommandHandler("guide_full", guide_full_cmd))
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help_admin", cmd_help_admin))
    app.add_handler(CommandHandler("manage", manage_cmd))
    app.add_handler(CommandHandler("myplan", myplan_cmd))
    app.add_handler(CommandHandler("support", support_cmd))
    app.add_handler(CommandHandler("support_status", support_status_cmd))
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
    app.add_handler(CommandHandler("sessions_on_unlimited", sessions_on_unlimited_cmd))
    app.add_handler(CommandHandler("sessions_unlimited_on", sessions_unlimited_on_cmd))
    app.add_handler(CommandHandler("sessions_off_unlimited", sessions_off_unlimited_cmd))
    app.add_handler(CommandHandler("sessions_unlimited_off", sessions_unlimited_off_cmd))

    app.add_handler(CommandHandler("bigmove_alert", bigmove_alert_cmd))
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
    app.add_handler(CommandHandler("billing", billing_cmd))
    app.add_handler(CommandHandler("email_on_off", email_on_off_cmd))
    app.add_handler(CommandHandler("upgrade", upgrade_cmd))
    app.add_handler(CommandHandler("trade_window", trade_window_cmd))
    app.add_handler(CommandHandler("email", email_cmd))
    app.add_handler(CommandHandler("email_on", email_on_cmd))
    app.add_handler(CommandHandler("email_off", email_off_cmd))   
    app.add_handler(CommandHandler("email_test", email_test_cmd))  
    app.add_handler(CommandHandler("email_decision", email_decision_cmd))

    app.add_handler(CommandHandler("why", why_no_setups_cmd))    
    # ================= USDT (semi-auto) =================
    app.add_handler(CommandHandler("usdt", usdt_info_cmd))
    app.add_handler(CommandHandler("usdt_paid", usdt_paid_cmd))
    app.add_handler(CommandHandler("usdt_pending", usdt_pending_cmd))
    app.add_handler(CommandHandler("usdt_approve", usdt_approve_cmd))
    app.add_handler(CommandHandler("usdt_reject", usdt_reject_cmd))
    
    # ================= Admin: access & payments =================
    
    app.add_handler(CommandHandler("admin_user", admin_user_cmd))
    app.add_handler(CommandHandler("admin_users", admin_users_cmd))
    app.add_handler(CommandHandler("admin_grant", admin_user_cmd))
    app.add_handler(CommandHandler("admin_revoke", admin_revoke_cmd)) 
    app.add_handler(CommandHandler("admin_payments", admin_payments_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))
    
    # Catch-all for unknown /commands (MUST be after all CommandHandlers)
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    if app.job_queue:
        interval_sec = int(CHECK_INTERVAL_MIN * 60)
    
        app.job_queue.run_repeating(
            alert_job,
            interval=interval_sec,
            first=90,
            name="alert_job",
            job_kwargs={
                "max_instances": 1,
                "coalesce": True,
                "misfire_grace_time": 60,
            },
        )
    else:
        logger.error("JobQueue NOT available – install python-telegram-bot[job-queue]")



    # ================= Stripe Webhook ================= #
    threading.Thread(
        target=start_stripe_webhook,
        daemon=True
    ).start()

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



# ===============================
# TRIAL + STATUS (ADDED)
# ===============================

TRIAL_DAYS = 7

def _ensure_trial(user):
    if not user:
        return
    if user.get("plan"):
        return
    start = user.get("trial_start_ts")
    now = time.time()
    if not start:
        update_user(user["user_id"], plan="trial", trial_start_ts=now, trial_until=now + TRIAL_DAYS*86400)
    elif now <= float(user.get("trial_until", 0)):
        update_user(user["user_id"], plan="trial")
    else:
        update_user(user["user_id"], plan="standard")

def user_has_pro(uid: int) -> bool:
    # Admin is always Pro/Unlimited
    try:
        if is_admin_user(int(uid)):
            return True
    except Exception:
        pass

    u = get_user(uid)
    if not u:
        return False

    _ensure_trial(u)

    # Use effective plan (covers legacy DBs + admin override)
    try:
        plan = str(effective_plan(u, int(uid))).strip().lower()
    except Exception:
        plan = str(u.get("plan") or "free").strip().lower()

    if plan == "pro":
        return True
    if plan == "trial" and time.time() <= float(u.get("trial_until", 0) or 0):
        return True
    return False
    _ensure_trial(u)
    if u.get("plan") == "pro":
        return True
    if u.get("plan") == "trial" and time.time() <= float(u.get("trial_until", 0)):
        return True
    return False

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = reset_daily_if_needed(get_user(uid))

    if not has_active_access(user, uid):
        await update.message.reply_text(
            "⛔️ Trial finished.\n\n"
            "Your 7-day trial is over — you need to pay to keep using PulseFutures.\n\n"
            "👉 /billing"
        )
        return

    opens = db_open_trades(uid)

    plan = str((user or {}).get("plan") or "free").upper()
    equity = float((user or {}).get("equity") or 0.0)

    cap = daily_cap_usd(user)
    day_local = _user_day_local(user)
    used_today = _risk_daily_get(uid, day_local)
    remaining_today = (cap - used_today) if cap > 0 else float("inf")

    enabled = user_enabled_sessions(user)
    now_s = in_session_now(user)
    now_txt = now_s["name"] if now_s else "NONE"

    # Email caps (show infinite as 0=∞ to match UI)
    cap_sess = int((user or {}).get("max_emails_per_session", DEFAULT_MAX_EMAILS_PER_SESSION) or DEFAULT_MAX_EMAILS_PER_SESSION)
    cap_day = int((user or {}).get("max_emails_per_day", DEFAULT_MAX_EMAILS_PER_DAY) or DEFAULT_MAX_EMAILS_PER_DAY)
    gap_m = int((user or {}).get("email_gap_min", DEFAULT_EMAIL_GAP_MIN) or DEFAULT_EMAIL_GAP_MIN)

    # Big-move status
    bm_on = int((user or {}).get("bigmove_alert_on", 1) or 0)
    bm_4h = float((user or {}).get("bigmove_alert_4h", 20) or 20)
    bm_1h = float((user or {}).get("bigmove_alert_1h", 10) or 10)

    lines = []
    lines.append("📌 Status")
    lines.append(f"Plan: {plan}")
    lines.append(f"Equity: ${equity:.2f}")
    lines.append(f"Trades today: {int(user.get('day_trade_count',0))}/{int(user.get('max_trades_day',0))}")
    lines.append(f"Daily cap: {user.get('daily_cap_mode','PCT')} {float(user.get('daily_cap_value',0.0)):.2f} (≈ ${cap:.2f})")
    lines.append(f"Daily risk used: ${used_today:.2f}")
    lines.append(f"Daily risk remaining: ${max(0.0, remaining_today):.2f}" if cap > 0 else "Daily risk remaining: ∞")
    lines.append(f"Email alerts: {'ON' if int(user.get('notify_on',1))==1 else 'OFF'}")
    lines.append(f"Sessions enabled: {' | '.join(enabled)} | Now: {now_txt}")
    lines.append(f"Email caps: session={cap_sess} (0=∞), day={cap_day} (0=∞), gap={gap_m}m")
    lines.append(f"Big-move alert emails: {'ON' if bm_on else 'OFF'} (4H≥{bm_4h:.0f}% OR 1H≥{bm_1h:.0f}%)")
    lines.append(HDR)

    if not opens:
        lines.append("Open trades: None")
        await update.message.reply_text("\n".join(lines))
        return

    lines.append("Open trades:")
    for t in opens:
        try:
            entry = float(t.get("entry") or 0.0)
            sl = float(t.get("sl") or 0.0)
            qty = float(t.get("qty") or 0.0)
            risk = float(t.get("risk_usd") or 0.0)
            lines.append(
                f"- ID {t.get('id')} | {t.get('symbol')} {t.get('side')} | "
                f"Entry {fmt_price(entry)} | SL {fmt_price(sl)} | Risk ${risk:.2f} | Qty {qty:.6g}"
            )
        except Exception:
            continue

    await update.message.reply_text("\n".join(lines))

if __name__ == "__main__":
    main()
