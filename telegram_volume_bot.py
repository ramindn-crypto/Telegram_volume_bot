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

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Bot,
    BotCommand,
    MenuButtonCommands,
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

def enforce_access_or_block_legacy(update: Update, command: str) -> bool:
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


# =========================================================
# CHANNEL SUBSCRIPTION GATE (optional)
# =========================================================
REQUIRED_CHANNEL = os.getenv("REQUIRED_CHANNEL", "").strip()  # e.g. "@PulseFutures" or "-1001234567890"
REQUIRED_CHANNEL_JOIN_URL = os.getenv("REQUIRED_CHANNEL_JOIN_URL", "").strip()  # e.g. "https://t.me/PulseFutures"

async def _is_user_subscribed(bot: Bot, user_id: int) -> bool:
    """Returns True if user is a member of REQUIRED_CHANNEL (member/admin/creator).
    NOTE: Bot must be an admin in the channel to reliably check membership.
    """
    if not REQUIRED_CHANNEL:
        return True
    try:
        cm = await bot.get_chat_member(chat_id=REQUIRED_CHANNEL, user_id=int(user_id))
        status = str(getattr(cm, "status", "") or "").lower()
        return status in {"member", "administrator", "creator"}
    except Exception:
        return False

async def _reply_subscribe_required(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        join_url = REQUIRED_CHANNEL_JOIN_URL or (f"https://t.me/{REQUIRED_CHANNEL.lstrip('@')}" if REQUIRED_CHANNEL.startswith("@") else "")
        kb = None
        if join_url:
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("📢 Join Channel", url=join_url)]])
        await update.message.reply_text(
            "📢 To use PulseFutures, you must join our channel first.\n\n"
            f"Channel: {REQUIRED_CHANNEL or '@PulseFutures'}\n\n"
            "After joining, come back and press /start.",
            reply_markup=kb,
            disable_web_page_preview=True,
        )
    except Exception:
        pass

async def _command_guard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Global guard: locks access after trial + gates Pro-only commands."""
    try:
        if not getattr(update, "message", None):
            return

        txt = (update.message.text or "").strip()
        if not txt.startswith("/"):
            return

        cmd = txt.split()[0][1:].split("@")[0].strip().lower()

        # 0) Channel subscription gate (optional)
        if REQUIRED_CHANNEL:
            ok = await _is_user_subscribed(
                context.bot, int(update.effective_user.id)
            )
            if not ok:
                if cmd not in {"start", "help", "commands", "billing", "guide_full"}:
                    await _reply_subscribe_required(update, context)
                    raise ApplicationHandlerStop

                if cmd == "start":
                    await _reply_subscribe_required(update, context)
                    raise ApplicationHandlerStop

        # 1) Trial/access lock
        if not enforce_access_or_block(update, cmd):
            raise ApplicationHandlerStop

        # 2) Pro-only commands
        if cmd in PRO_ONLY_COMMANDS:
            uid = update.effective_user.id
            if not user_has_pro(uid):
                try:
                    await update.message.reply_text(
                        "🚀 Pro feature.\n\n"
                        "This command is available in *Pro* (and during your 7-day trial).\n\n"
                        "👉 /billing",
                        parse_mode="Markdown",
                    )
                except Exception:
                    pass
                raise ApplicationHandlerStop

    except ApplicationHandlerStop:
        raise
    except Exception:
        return

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
def trial_expired_legacy(user: dict) -> bool:
    try:
        created = datetime.fromisoformat(user["created_at"])
    except Exception:
        return True
    return datetime.utcnow() > created + timedelta(days=FREE_TRIAL_DAYS)


def has_active_access_legacy(user: dict) -> bool:
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
    tz_name = str(user.get("tz") or user.get("timezone") or "UTC").strip()
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")
    today = datetime.now(tz).date().isoformat()

    if user["day_trade_date"] != today:
        update_user(
            user["user_id"],
            day_trade_date=today,
            day_trade_count=0
        )
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
    tz_name = str(user.get("tz") or user.get("timezone") or "UTC").strip()
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")
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


def get_cached_futures_tickers() -> Dict[str, MarketVol]:
    """Fast, no-network accessor for last known futures tickers.

    Used by instant commands like /size and /status to avoid blocking on CCXT calls.
    Returns {} if nothing has been cached yet.
    """
    try:
        obj = cache_get("tickers_best_fut")
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

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
# HELP TEXT (USER)
# =========================================================

HELP_TEXT = """\
🚀 PulseFutures — Trading System in Telegram
────────────────────
Core Commands
────────────────────
/screen
• Scan the market for high-quality setups

/status
• Your plan, trial status & enabled features

/commands
• Full command guide + examples

/guide_full
• Download the full user guide (PDF)

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
Core Commands
────────────────────
/status
• Shows your plan (Trial/Standard/Pro) & enabled features

/health
• Bot & data health check 

────────────────────
Market & Signals 
────────────────────
/screen
• Scans the market for high-quality setups

────────────────────
⚖️ RISK & POSITION SIZING
────────────────────
/equity
• Set your equity

/riskmode
• Set your risk per trade

/size <symbol> <side> <entry> <sl>
• Calculates position size based on your risk rules

────────────────────
Trade Journal
────────────────────
/trade_open
• Log an opned position

/trade_sl
• Update Stop Loss

/trade_rf
• Risk-Free a position

/trade_close
• Log a closed position

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
• 24-hour mode for scans

/trade_window
• Set allowed trading time window 

────────────────────
⚠️ EMAILS & ALERTS
────────────────────

/email you@gmail.com
• Set your email for alerts

/email_test
• Send a test email to confirm delivery

/email on
• Enable email

/email off
• Disable email

/limits emailcap 
• Set number of emails per session

/limits emailgap
• Set min gap between emails 

/limits emaildaycap 
• Set max number of emails per day

/bigmove_alert on|off [4H%] [1H%]
• Big move alerts in either direction (UP or DOWN)

────────────────────
⏰ TIMEZONE (LOCAL TIME IN EMAILS)
────────────────────
/tz
• Show your current timezone

/tz <Region/City>
• Set your timezone so emails show your local time
• Use IANA format: Region/City

Examples:
/tz Australia/Melbourne
/tz Asia/Dubai   
/tz Europe/London
/tz America/New_York

────────────────────
Reports 
────────────────────
/report_daily 
• Daily performance report 

/report_weekly 
• Weekly performance report 

/report_overall 
• All-time performance report 

────────────────────
🆘 HELP & SUPPORT
────────────────────
/help
• Quick overview

/commands
• Full guide (this)

/guide_full
• Download the full user guide (PDF)

/Support
• Submit your support request

────────────────────
📢 Channels
────────────────────
Channel: @PulseFutures
Support: @PulseFuturesSupport
YouTube: @PulseFutures
Website: https://pulsefutures.com/

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
Channel: @PulseFutures
Support: @PulseFuturesSupport
YouTube: @PulseFutures
Website: https://pulsefutures.com/


────────────────────
🆘 SUPPORT
────────────────────
/support_open
• List open support tickets (admin)

/support_close <TICKET_ID>
• Close a support ticket

(Users)
 /support <issue>
 /support_status

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

# Support notifications:
# - Admin IDs always receive tickets
# - Optionally forward to a dedicated support group/channel where the bot is admin:
#     Set SUPPORT_CHAT_ID="-1001234567890"
SUPPORT_CHAT_ID = os.getenv("SUPPORT_CHAT_ID", "").strip()

def _admin_ids_all() -> List[int]:
    ids = set()
    # ADMIN_IDS is used in many places
    try:
        for x in (ADMIN_IDS or []):
            try:
                ids.add(int(x))
            except Exception:
                pass
    except Exception:
        pass
    # ADMIN_USER_IDS (legacy)
    try:
        for x in (ADMIN_USER_IDS or []):
            try:
                ids.add(int(x))
            except Exception:
                pass
    except Exception:
        pass
    # Single env fallbacks
    for k in ("ADMIN_TELEGRAM_ID", "OWNER_USER_ID", "ADMIN_ID"):
        v = os.getenv(k, "").strip()
        if v:
            try:
                ids.add(int(v))
            except Exception:
                pass
    return sorted(ids)

def _support_db_init() -> None:
    con = None
    try:
        con = db_connect()
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS support_tickets (
                ticket_id TEXT PRIMARY KEY,
                user_id INTEGER,
                username TEXT,
                message TEXT,
                status TEXT,
                created_ts REAL,
                updated_ts REAL
            )
        """)
        con.commit()
    except Exception:
        try:
            if con:
                con.rollback()
        except Exception:
            pass
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

def _support_ticket_create(ticket_id: str, user_id: int, username: str, message: str):
    _support_db_init()
    con = db_connect()
    cur = con.cursor()
    ts = time.time()
    cur.execute("""
        INSERT OR REPLACE INTO support_tickets (ticket_id, user_id, username, message, status, created_ts, updated_ts)
        VALUES (?, ?, ?, ?, 'OPEN', ?, ?)
    """, (str(ticket_id), int(user_id), str(username or ""), str(message or ""), float(ts), float(ts)))
    con.commit()
    con.close()

def _support_ticket_latest_for_user(user_id: int) -> Optional[dict]:
    _support_db_init()
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM support_tickets WHERE user_id=? ORDER BY created_ts DESC LIMIT 1", (int(user_id),))
    r = cur.fetchone()
    con.close()
    return dict(r) if r else None

def _support_ticket_list_open(limit: int = 20) -> List[dict]:
    _support_db_init()
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM support_tickets WHERE status='OPEN' ORDER BY created_ts DESC LIMIT ?", (int(limit),))
    rows = cur.fetchall() or []
    con.close()
    return [dict(x) for x in rows]

def _support_ticket_set_status(ticket_id: str, status: str):
    _support_db_init()
    con = db_connect()
    cur = con.cursor()
    ts = time.time()
    cur.execute("UPDATE support_tickets SET status=?, updated_ts=? WHERE ticket_id=?", (str(status), float(ts), str(ticket_id)))
    con.commit()
    con.close()

async def support_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = int(update.effective_user.id)

    if not context.args:
        await update.message.reply_text(
            "Usage:\n/support <your issue>"
        )
        return

    issue = " ".join(context.args).strip()
    if not issue:
        await update.message.reply_text(
            "Usage:\n/support <your issue>"
        )
        return

    ticket_id = f"TKT-{uid}-{int(time.time())}"
    username = getattr(update.effective_user, "username", "") or ""

    _support_ticket_create(ticket_id, uid, username, issue)

    msg = (
        f"🆘 Support Ticket {ticket_id}\n\n"
        f"User: {uid} @{username}\n"
        f"Message:\n{issue}"
    )

    # Notify admins
    for admin in _admin_ids_all():
        try:
            await context.bot.send_message(admin, msg)
        except Exception:
            pass

    # Optional: forward to a dedicated support group/channel
    if SUPPORT_CHAT_ID:
        try:
            await context.bot.send_message(chat_id=SUPPORT_CHAT_ID, text=msg)
        except Exception:
            pass

    await update.message.reply_text(
        f"✅ Ticket created: {ticket_id}\n"
        "Use /support_status to check progress."
    )


async def support_status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = int(update.effective_user.id)
    t = _support_ticket_latest_for_user(uid)

    if not t:
        await update.message.reply_text(
            "📨 You have no support tickets yet. Use /support <your issue>."
        )
        return

    status = str(t.get("status") or "OPEN").upper()
    tid = str(t.get("ticket_id") or "")

    await update.message.reply_text(
        f"📨 Latest ticket: {tid}\n"
        f"Status: {status}\n\n"
        "If you need to add more info, create a new ticket with /support <your issue>."
    )



async def admin_support_open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = int(update.effective_user.id)
    if not is_admin_user(uid):
        await update.message.reply_text("Admin only.")
        return

    rows = _support_ticket_list_open(limit=30)
    if not rows:
        await update.message.reply_text("✅ No open support tickets.")
        return

    lines = ["🧾 Open support tickets", HDR]
    for r in rows:
        tid = r.get("ticket_id")
        u = r.get("user_id")
        un = r.get("username") or ""
        msg = (r.get("message") or "").strip().replace("\n", " ")
        if len(msg) > 80:
            msg = msg[:77] + "..."
        lines.append(f"- {tid} | {u} @{un} | {msg}")
    lines.append(HDR)
    lines.append("Close: /support_close <TICKET_ID>")
    await send_long_message(update, "\n".join(lines), parse_mode=None, disable_web_page_preview=True)

async def admin_support_close_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = int(update.effective_user.id)
    if not is_admin_user(uid):
        await update.message.reply_text("Admin only.")
        return
    if not context.args:
        await update.message.reply_text("Usage:\n/support_close <TICKET_ID>")
        return
    tid = str(context.args[0]).strip()
    _support_ticket_set_status(tid, "CLOSED")
    await update.message.reply_text(f"✅ Closed: {tid}")


async def health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

    # --- Entry sanity check vs last known price (warn only; NEVER blocks /size) ---
    try:
        best_now = get_cached_futures_tickers()
        mv_now = best_now.get(sym) if isinstance(best_now, dict) else None
        cur_px = float(mv_now.last) if mv_now and float(getattr(mv_now, "last", 0) or 0) > 0 else 0.0
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
        best_now = get_cached_futures_tickers()
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
    # If webhook was set previously, remove it so polling starts cleanly
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
    except Exception as e:
        logger.warning("delete_webhook failed (ignored): %s", e)

    # Bot menu + command list (Telegram "Menu" button)
    try:
        cmds = [
            BotCommand("start", "Start / restart (starts 7-day trial after joining channel)"),
            BotCommand("screen", "Scan market and show top setups"),
            BotCommand("status", "Your status (plan, caps, sessions, open trades)"),
            BotCommand("size", "Position size calculator"),
            BotCommand("equity", "Set equity / account size"),
            BotCommand("risk_mode", "Set risk mode (USD / PCT)"),
            BotCommand("risk", "Set risk per trade"),
            BotCommand("limits", "Set daily caps / limits"),
            BotCommand("billing", "Subscription & payment info"),
            BotCommand("support", "Open support ticket: /support <issue>"),
            BotCommand("support_status", "Check your latest support ticket"),
            BotCommand("help", "Help"),
            BotCommand("commands", "Command list"),
        ]
        await app.bot.set_my_commands(cmds)
        # Ensure menu button is enabled for private chats
        try:
            await app.bot.set_chat_menu_button(menu_button=MenuButtonCommands())
        except Exception:
            pass
    except Exception as e:
        logger.warning("set_my_commands failed (ignored): %s", e)



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
    
    app = Application.builder().token(TOKEN).post_init(_post_init).concurrent_updates(32).build()

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
    app.add_handler(CommandHandler("support_open", admin_support_open_cmd))
    app.add_handler(CommandHandler("support_close", admin_support_close_cmd))
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
