
#!/usr/bin/env python3
"""
PulseFutures — Bybit Futures (Swap) Screener + Signals Email + Risk Manager + Trade Journal (Telegram)

✅ What you asked (implemented):
1) Position sizing from Risk + StopLoss:
   - /size BTC long risk usd 40 sl 42000
   - /size BTC short risk pct 2.5 sl 43210
   Bot fetches current futures price as Entry (unless you provide entry manually).
   It returns Qty (contracts/base units) so user can open the position.

2) Full Trade Journal (open + close) and Equity auto-update:
   - /trade_open BTC long entry 43000 sl 42000 risk usd 40 note "breakout" sig PF-20251219-0007
   - /trade_close 12 pnl +85.5
   Equity updates ONLY when you close trades, and stays persistent until /equity_reset.

3) Emails per user timezone: 3–4 emails per active session (min 60m gap), best setups only.
   Default enabled session is chosen based on user's TZ (Americas→NY, Europe/Africa→LON, Asia/Oceania→ASIA).
   User can enable other sessions:
   - /sessions
   - /sessions_on NY
   - /sessions_on LON
   - /sessions_on ASIA
   - /sessions_off LON

4) Daily/Weekly user performance summary + advice:
   - /report_daily
   - /report_weekly
   Includes win/loss, net PnL, win rate, avg R, biggest loss, and practical advice.

5) Unique Setup IDs for emailed signals + signal summaries:
   - Every emailed setup has a unique ID like PF-YYYYMMDD-0001
   - Users can reference it in your Telegram channel, or attach it to trades
   - Signal summary (sent signals, and performance if users link trades):
     - /signals_daily
     - /signals_weekly

6) Speed + Render stability:
   - Heavy CCXT + OHLCV work runs in asyncio.to_thread()
   - In-memory TTL caching for tickers and OHLCV
   - Alert job lock prevents overlapping runs
   - Optional WEBHOOK mode (recommended on Render) to avoid 409 conflicts

✅ New additions (per your latest requests, WITHOUT removing anything):
7) Fixed session priority ordering (professional):
   - Priority: NY > LON > ASIA (no accidental alphabetic sorting)
   - Used consistently when enabling/disabling sessions and when listing enabled sessions

8) Professional definition of "best signals with session priority":
   - Different strictness per session using MIN confidence:
     NY: easier (more signals)  | LON: stricter | ASIA: strictest (highest quality)
   - The bot still scans the FULL session window continuously (not only first 3 hours)

9) Optional "unlimited email cap" support:
   - /limits emailcap 0  => unlimited emails per session
   - This keeps your minimum gap rule (emailgap) and cooldown (no repeating same symbol within 18h)

✅ Premium signal upgrades (added WITHOUT removing anything):
10) Premium Risk/Reward gating + wider TP/SL (now tuned closer per your request):
   - TP ladder tuned closer (RR target ~2.0–2.5)
   - TP3 dynamic cap (normal vs hot coins)
   - Email includes: "No chase" style via entry zone REMOVED (per request)

11) 15m logic (Model A) — UPDATED to Soft Confirm (no removal):
   - /screen remains SOFT on 15m (does NOT hard-reject weak 15m)
   - Email becomes SOFT on 15m:
       • CONFIRMED if 15m meets threshold
       • EARLY if 15m is weak BUT 1H is very strong (extra gates apply)

12) Quality-only delivery + “2–3 trades/day style” control:
   - Added a DAILY EMAIL CAP (per user, across ALL sessions) default = 4
   - User-controllable via /limits emaildaycap <N> (0=unlimited)

13) Trend-follow Email filter (avoid counter-trend reversals):
   - Emails only when 24H supports the trade direction:
       BUY  => 24H >= +0.5%
       SELL => 24H <= -0.5%
   - /screen stays unchanged (still shows setups for awareness)

14) Reject Diagnostics (why only 1 setup?)
   - Optional DEBUG_REJECTS=true to include sample rejects
   - Always counts reject reasons per /screen run (in-memory)
   - /screen appends a short Reject Diagnostics block (counts + samples if enabled)

15) ✅ NO MORE duplicate Setup IDs:
   - SQLite daily counter table with atomic increment (setup_counter)

16) ✅ Daily Email Cap is now USER-CONTROLLABLE via Telegram:
   - /limits emaildaycap 4
   - /limits emaildaycap 0

17) ✅ Daily Risk Used / Remaining (Telegram):
   - On /trade_open and /status

✅ NEW (your 29 Dec changes — implemented):
18) EMA12(15m) mandatory filter:
   - Signals only if price is near EMA12 on 15m (BUY/SELL)
19) Melbourne "no-signal" window:
   - Between 10:00 and 12:00 Melbourne, no signals (/screen + emails)
20) Sharp 1H move gating:
   - If abs(1H change) >= 20%, signal ONLY if EMA12 reaction is detected (not immediately after spike)
21) TP/SL tuned closer + dynamic:
   - TP3 cap: Normal ~12%, Hot ~15%
   - Allocations: 40/40/20
   - RR gate: TP3 RR >= 2.0
22) Email cleanup:
   - Removed Entry Zone
   - Removed "Rule: after TP1..."
   - More sensible decimals in email
23) Debug extras (requested):
   - /screen shows EMA12 distance (%) for each setup
   - HOT coins flagged with TP3 TRAILING (informational)

IMPORTANT (Render):
- If you see: "Conflict: terminated by other getUpdates request"
  it means multiple instances are polling. Use WEBHOOK mode or ensure only 1 instance.

ENV:
- TELEGRAM_TOKEN (required)

Email ENV (required to actually send):
- EMAIL_ENABLED=true
- EMAIL_HOST (e.g., smtp.gmail.com)
- EMAIL_PORT (465 or 587)
- EMAIL_USER
- EMAIL_PASS (app password)
- EMAIL_FROM
- EMAIL_TO (can be same as EMAIL_USER or comma separated list)

Optional:
- CHECK_INTERVAL_MIN=5
- DB_PATH=pulsefutures.db
- WEBHOOK_URL=https://your-service.onrender.com   (recommended on Render)
- PORT=10000 (Render sets it)

Reject diagnostics optional:
- DEBUG_REJECTS=true
- REJECT_TOP_N=12

Optional (email footer link):
- TELEGRAM_BOT_URL=https://t.me/PulseFuturesBot
"""

import asyncio
import logging
import os
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
from collections import Counter

import ccxt
from tabulate import tabulate
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
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

# Screen output sizes
LEADERS_N = 10
SETUPS_N = 3
EMAIL_SETUPS_N = 3

# Directional Leaders/Losers thresholds
MOVER_VOL_USD_MIN = 5_000_000
MOVER_UP_24H_MIN = 10.0
MOVER_DN_24H_MAX = -10.0

# Setup engine thresholds (tune later)
TRIGGER_1H_ABS_MIN = 2.0
CONFIRM_15M_ABS_MIN = 0.6
ALIGN_4H_MIN = 0.0  # require same direction on 4H

# ✅ Soft-confirm 15m for EMAIL (without removing anything)
EARLY_1H_ABS_MIN = 3.8         # "very strong" 1H momentum gate
EARLY_CONF_PENALTY = 6         # reduce conf a bit if 15m is weak (still eligible if very strong)
EARLY_EMAIL_EXTRA_CONF = 4     # EARLY requires higher confidence than session min_conf
EARLY_EMAIL_MAX_FILL = 1       # at most 1 EARLY setup used to fill the email (keeps quality)

# ✅ Trend-follow Email filter (avoid counter-trend reversals)
TREND_24H_TOL = 0.5  # percent; BUY needs >= +0.5%, SELL needs <= -0.5%

# Risk defaults (user controlled; equity starts at 0 by design)
DEFAULT_EQUITY = 0.0
DEFAULT_RISK_MODE = "PCT"     # PCT or USD
DEFAULT_RISK_VALUE = 1.5      # 1.5% by default (only used if user doesn't override)
DEFAULT_DAILY_CAP_MODE = "PCT"
DEFAULT_DAILY_CAP_VALUE = 5.0  # ✅ default 5% equity/day risk cap
DEFAULT_MAX_TRADES_DAY = 5
DEFAULT_MIN_EMAIL_GAP_MIN = 60
DEFAULT_MAX_EMAILS_PER_SESSION = 4  # user can set 0 for unlimited

# ✅ DAILY email cap per user across ALL sessions (user-controllable)
DEFAULT_MAX_EMAILS_PER_DAY = 4

# ✅ /size default risk ceiling
DEFAULT_MAX_RISK_PCT_PER_TRADE = 2.0
WARN_RISK_PCT_PER_TRADE = 2.0

# Cooldown
SYMBOL_COOLDOWN_HOURS = 18  # do not repeat same symbol in emails for that user within 18h

# Multi-TP
ATR_PERIOD = 14

ATR_MIN_PCT = 1.0
ATR_MAX_PCT = 8.0

MULTI_TP_MIN_CONF = 78

# ✅ allocations requested
TP_ALLOCS = (40, 40, 20)

# ✅ TP ladder tuned closer (RR target ~2.0–2.5)
TP_R_MULTS = (1.0, 1.7, 2.4)

# ✅ Gate low RR setups from EMAIL (TP3)
MIN_RR_TP3 = 2.0

# ✅ Dynamic TP cap: normal vs hot coins (TP3 about 15% for hot)
TP_MAX_PCT_NORMAL = 12.0
TP_MAX_PCT_HOT = 15.0
HOT_VOL_USD = 50_000_000
HOT_CH24_ABS = 15.0

# =========================================================
# ✅ EMA12 (15m) proximity + sharp move gating
# =========================================================
EMA12_PERIOD = 12

# maximum allowed distance from EMA12 (derived from ATR%), clamped
EMA12_MAX_DIST_ATR_MULT = 0.35  # threshold_pct = (0.35*ATR/entry)*100
EMA12_MAX_DIST_PCT_MIN = 0.15   # percent
EMA12_MAX_DIST_PCT_MAX = 0.90   # percent

# Sharp 1H move gating
SHARP_1H_MOVE_PCT = 20.0

# Melbourne blackout window
BLACKOUT_TZ = "Australia/Melbourne"
BLACKOUT_START_HH = 10
BLACKOUT_END_HH = 12

# Email
EMAIL_ENABLED = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST = os.environ.get("EMAIL_HOST", "")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "465"))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", EMAIL_USER)
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_USER)  # comma-separated ok

# Render stability
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "").strip()

# Email footer / CTA
TELEGRAM_BOT_URL = os.environ.get("TELEGRAM_BOT_URL", "https://t.me/PulseFuturesBot").strip()

# Caching for speed
TICKERS_TTL_SEC = 45
OHLCV_TTL_SEC = 60

# =========================================================
# DEBUG / REJECT REASONS
# =========================================================
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
    "ASIA": {"start": "00:00", "end": "06:00"},  # 00-06 UTC
    "LON":  {"start": "07:00", "end": "12:00"},  # 07-12 UTC
    "NY":   {"start": "13:00", "end": "20:00"},  # 13-20 UTC
}

# Priority order for sessions
SESSION_PRIORITY = ["NY", "LON", "ASIA"]

# Stricter filters by session (NY easiest, ASIA strictest)
SESSION_MIN_CONF = {
    "NY": 72,
    "LON": 78,
    "ASIA": 82,
}

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
    ema12_dist_pct: float          # ✅ NEW: debug for /screen
    is_trailing_tp3: bool          # ✅ NEW: hot coin info
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
# REJECT TRACKER (in-memory per run)
# =========================================================
_REJECT_STATS = Counter()
_REJECT_SAMPLES: Dict[str, List[str]] = {}

def fmt_price(x: float) -> str:
    ax = abs(x)
    if ax >= 100:
        return f"{x:.2f}"
    if ax >= 1:
        return f"{x:.4f}"
    if ax >= 0.1:
        return f"{x:.5f}"
    return f"{x:.6f}"

def _rej(reason: str, base: str, mv: "MarketVol", extra: str = "") -> None:
    global _REJECT_STATS, _REJECT_SAMPLES
    _REJECT_STATS[reason] += 1
    if DEBUG_REJECTS:
        try:
            last = fmt_price(float(mv.last or 0.0))
        except Exception:
            last = str(mv.last)
        line = f"{base} | {mv.symbol} | last={last} | {extra}".strip()
        xs = _REJECT_SAMPLES.get(reason, [])
        if len(xs) < REJECT_TOP_N:
            xs.append(line)
            _REJECT_SAMPLES[reason] = xs

def _reject_report() -> str:
    if not _REJECT_STATS:
        return ""
    parts = []
    parts.append("🧩 Reject Diagnostics (why setups were filtered)")
    parts.append(SEP)
    top = _REJECT_STATS.most_common(10)
    for reason, cnt in top:
        parts.append(f"- {reason}: {cnt}")
        if DEBUG_REJECTS and reason in _REJECT_SAMPLES and _REJECT_SAMPLES[reason]:
            smp = _REJECT_SAMPLES[reason][:3]
            for s in smp:
                parts.append(f"    • {s}")
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
# EXCHANGE HELPERS
# =========================================================
def build_exchange():
    klass = ccxt.__dict__[EXCHANGE_ID]
    return klass({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": DEFAULT_TYPE},
    })

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
    if cache_valid("tickers_best_fut", TICKERS_TTL_SEC):
        return cache_get("tickers_best_fut")

    ex = build_exchange()
    ex.load_markets()
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
    key = f"ohlcv:{symbol}:{timeframe}:{limit}"
    if cache_valid(key, OHLCV_TTL_SEC):
        return cache_get(key)

    ex = build_exchange()
    ex.load_markets()
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit) or []
    cache_set(key, data)
    return data

def compute_atr_from_ohlcv(candles: List[List[float]], period: int) -> float:
    if not candles or len(candles) < period + 2:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        prev_close = float(candles[i - 1][4])
        high = float(candles[i][2])
        low = float(candles[i][3])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if len(trs) < period:
        return 0.0
    return sum(trs[-period:]) / period

def ema(values: List[float], period: int) -> float:
    if not values or len(values) < period:
        return 0.0
    k = 2.0 / (period + 1.0)
    e = float(values[0])
    for v in values[1:]:
        e = (float(v) * k) + (e * (1.0 - k))
    return float(e)

def is_blackout_melbourne_now() -> bool:
    tz = ZoneInfo(BLACKOUT_TZ)
    now = datetime.now(tz)
    return (BLACKOUT_START_HH <= now.hour < BLACKOUT_END_HH)

def is_hot_coin(fut_vol_usd: float, ch24: float) -> bool:
    return (float(fut_vol_usd) >= float(HOT_VOL_USD)) and (abs(float(ch24)) >= float(HOT_CH24_ABS))

def tp_cap_pct_for_coin(fut_vol_usd: float, ch24: float) -> float:
    return TP_MAX_PCT_HOT if is_hot_coin(fut_vol_usd, ch24) else TP_MAX_PCT_NORMAL

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def ema12_proximity_ok(entry: float, ema12_val: float, atr_1h: float) -> bool:
    """
    Require entry to be near EMA12(15m).
    Threshold is derived from ATR% (1H ATR proxy) and clamped to sensible bounds.
    """
    if entry <= 0 or ema12_val <= 0:
        return False
    dist = abs(entry - ema12_val)
    pct_dist = (dist / entry) * 100.0

    # derive threshold from ATR%, then clamp
    atr_pct = (atr_1h / entry) * 100.0 if (atr_1h and entry) else 0.0
    thr_pct = clamp(EMA12_MAX_DIST_ATR_MULT * atr_pct, EMA12_MAX_DIST_PCT_MIN, EMA12_MAX_DIST_PCT_MAX)

    return pct_dist <= thr_pct

def ema12_reaction_ok_15m(c15: List[List[float]], ema12_val: float, side: str) -> bool:
    """
    Reaction definition:
    - BUY: recent candle touched/broke below EMA12 and closed back above EMA12
    - SELL: recent candle touched/broke above EMA12 and closed back below EMA12
    Checks last ~6 candles.
    """
    if not c15 or len(c15) < 8 or ema12_val <= 0:
        return False

    lookback = c15[-6:]
    for c in lookback:
        h = float(c[2]); l = float(c[3]); cl = float(c[4])
        if side == "BUY":
            if (l <= ema12_val) and (cl > ema12_val):
                return True
        else:
            if (h >= ema12_val) and (cl < ema12_val):
                return True
    return False

def metrics_from_candles_1h_15m(market_symbol: str) -> Tuple[float, float, float, float, float, List[List[float]]]:
    """
    returns: ch1, ch4, ch15, atr_1h, ema12_15m, c15
    """
    need_1h = max(ATR_PERIOD + 6, 35)
    c1 = fetch_ohlcv(market_symbol, "1h", limit=need_1h)
    if not c1 or len(c1) < 6:
        return 0.0, 0.0, 0.0, 0.0, 0.0, []

    closes_1h = [float(x[4]) for x in c1]
    c_last = closes_1h[-1]
    c_prev1 = closes_1h[-2]
    c_prev4 = closes_1h[-5] if len(closes_1h) >= 5 else closes_1h[0]

    ch1 = ((c_last - c_prev1) / c_prev1) * 100.0 if c_prev1 else 0.0
    ch4 = ((c_last - c_prev4) / c_prev4) * 100.0 if c_prev4 else 0.0
    atr_1h = compute_atr_from_ohlcv(c1, ATR_PERIOD)

    c15 = fetch_ohlcv(market_symbol, "15m", limit=60)
    if not c15 or len(c15) < 15:
        return ch1, ch4, 0.0, atr_1h, 0.0, []

    closes_15 = [float(x[4]) for x in c15]
    ema12_15m = ema(closes_15[-(EMA12_PERIOD + 20):], EMA12_PERIOD)

    c15_last = float(c15[-1][4])
    c15_prev = float(c15[-2][4])
    ch15 = ((c15_last - c15_prev) / c15_prev) * 100.0 if c15_prev else 0.0

    return ch1, ch4, ch15, atr_1h, ema12_15m, c15


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
TELEGRAM_MAX = 4096
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
# SL/TP ENGINE (closer + dynamic cap)
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

def compute_sl_tp(entry: float, side: str, atr: float, conf: int, tp_cap_pct: float) -> Tuple[float, float, float]:
    # returns (sl, tp3, R)
    if entry <= 0 or atr <= 0:
        return 0.0, 0.0, 0.0

    sl_dist = sl_mult_from_conf(conf) * atr
    min_dist = (ATR_MIN_PCT / 100.0) * entry
    max_dist = (ATR_MAX_PCT / 100.0) * entry
    sl_dist = clamp(sl_dist, min_dist, max_dist)

    R = sl_dist

    # target RR ~2.2 then cap
    tp_dist = 2.2 * R
    tp_cap = (float(tp_cap_pct) / 100.0) * entry
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

def multi_tp(entry: float, side: str, R: float, tp_cap_pct: float) -> Tuple[float, float, float]:
    if entry <= 0 or R <= 0:
        return 0.0, 0.0, 0.0
    r1, r2, r3 = TP_R_MULTS
    maxd = (float(tp_cap_pct) / 100.0) * entry

    d3 = min(r3 * R, maxd)
    d2 = min(r2 * R, d3 * (r2 / r3))
    d1 = min(r1 * R, d3 * (r1 / r3))

    if side == "BUY":
        tp1, tp2, tp3 = (entry + d1, entry + d2, entry + d3)
    else:
        tp1, tp2, tp3 = (entry - d1, entry - d2, entry - d3)

    tp1, tp2, tp3 = _distinctify(tp1, tp2, tp3, entry, side)
    return tp1, tp2, tp3

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
    up, dn = [], []
    for base, mv in best_fut.items():
        vol = usd_notional(mv)
        if vol < MOVER_VOL_USD_MIN:
            continue
        ch24 = float(mv.percentage or 0.0)
        if ch24 < MOVER_UP_24H_MIN and ch24 > MOVER_DN_24H_MAX:
            continue

        ch1, ch4, ch15, atr_1h, ema12_15m, c15 = metrics_from_candles_1h_15m(mv.symbol)

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

def make_setup(base: str, mv: MarketVol, strict_15m: bool = True) -> Optional[Setup]:
    # ✅ Melbourne blackout applies to all signal generation
    if is_blackout_melbourne_now():
        _rej("melbourne_blackout_10_12", base, mv, "No signals 10:00–12:00 Melbourne")
        return None

    fut_vol = usd_notional(mv)
    if fut_vol <= 0:
        _rej("no_fut_vol", base, mv)
        return None

    entry = float(mv.last or 0.0)
    if entry <= 0:
        _rej("bad_entry", base, mv)
        return None

    ch24 = float(mv.percentage or 0.0)
    ch1, ch4, ch15, atr_1h, ema12_15m, c15 = metrics_from_candles_1h_15m(mv.symbol)

    if (ch1 == 0.0 and ch4 == 0.0 and ch15 == 0.0 and atr_1h == 0.0) or (not c15) or (ema12_15m == 0.0):
        _rej("ohlcv_missing_or_insufficient", base, mv, "metrics/ema missing")
        return None

    # Trigger on 1H momentum
    if abs(ch1) < TRIGGER_1H_ABS_MIN:
        _rej("ch1_below_trigger", base, mv, f"ch1={ch1:+.2f}% < {TRIGGER_1H_ABS_MIN:.2f}%")
        return None

    side = "BUY" if ch1 > 0 else "SELL"

    # 4H alignment
    if side == "BUY" and ch4 < ALIGN_4H_MIN:
        _rej("4h_not_aligned_for_long", base, mv, f"side=BUY ch4={ch4:+.2f}% < {ALIGN_4H_MIN:.2f}%")
        return None
    if side == "SELL" and ch4 > -ALIGN_4H_MIN:
        _rej("4h_not_aligned_for_short", base, mv, f"side=SELL ch4={ch4:+.2f}% > {-ALIGN_4H_MIN:.2f}%")
        return None

    # EMA12 proximity mandatory (15m)
    if not ema12_proximity_ok(entry, ema12_15m, atr_1h):
        d_pct = abs(entry - ema12_15m) / entry * 100.0 if ema12_15m > 0 else 999.0
        _rej("price_not_near_ema12_15m", base, mv, f"dist={d_pct:.2f}%")
        return None

    # Sharp 1H move gating: must show EMA reaction, not immediate spike-chase
    if abs(float(ch1)) >= float(SHARP_1H_MOVE_PCT):
        if not ema12_reaction_ok_15m(c15, ema12_15m, side):
            _rej("sharp_1h_no_ema_reaction", base, mv, f"ch1={ch1:+.2f}% needs EMA12 reaction")
            return None

    # Soft 24H contradiction gate (dynamic by ATR%)
    atr_pct = (atr_1h / entry) * 100.0 if (atr_1h and entry) else 0.0
    thr = clamp(max(12.0, 2.5 * atr_pct), 12.0, 22.0)

    if side == "BUY" and ch24 <= -thr:
        _rej("24h_contradiction_for_long", base, mv, f"ch24={ch24:+.1f}% <= -{thr:.1f}% (atr%={atr_pct:.2f})")
        return None
    if side == "SELL" and ch24 >= +thr:
        _rej("24h_contradiction_for_short", base, mv, f"ch24={ch24:+.1f}% >= +{thr:.1f}% (atr%={atr_pct:.2f})")
        return None

    # 15m confirm logic (soft confirm)
    is_confirm_15m = abs(ch15) >= CONFIRM_15M_ABS_MIN
    is_early_allowed = (abs(ch1) >= EARLY_1H_ABS_MIN)

    if strict_15m:
        if (not is_confirm_15m) and (not is_early_allowed):
            _rej("15m_weak_and_not_early", base, mv, f"ch15={ch15:+.2f}% < {CONFIRM_15M_ABS_MIN:.2f}% and ch1={ch1:+.2f}% < {EARLY_1H_ABS_MIN:.2f}%")
            return None

    conf = compute_confidence(side, ch24, ch4, ch1, ch15, fut_vol)

    if strict_15m and (not is_confirm_15m):
        conf = max(0, int(conf) - int(EARLY_CONF_PENALTY))

    # Dynamic TP cap by coin "hotness"
    tp_cap_pct = tp_cap_pct_for_coin(fut_vol, ch24)
    sl, tp3_single, R = compute_sl_tp(entry, side, atr_1h, conf, tp_cap_pct)
    if sl <= 0 or tp3_single <= 0 or R <= 0:
        _rej("bad_sl_tp_or_atr", base, mv, f"atr={atr_1h:.6g} entry={entry:.6g}")
        return None

    tp1 = tp2 = None
    tp3 = tp3_single
    if conf >= MULTI_TP_MIN_CONF:
        _tp1, _tp2, _tp3 = multi_tp(entry, side, R, tp_cap_pct)
        if _tp1 and _tp2 and _tp3:
            tp1, tp2, tp3 = _tp1, _tp2, _tp3

    sid = next_setup_id()
    ema12_dist_pct = abs(entry - ema12_15m) / entry * 100.0 if ema12_15m > 0 else 999.0
    hot = is_hot_coin(fut_vol, ch24)

    s = Setup(
        setup_id=sid,
        symbol=base,
        market_symbol=mv.symbol,
        side=side,
        conf=conf,
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
        ema12_dist_pct=ema12_dist_pct,
        is_trailing_tp3=bool(hot),   # informational
        created_ts=time.time(),
    )
    return s

def pick_setups(best_fut: Dict[str, MarketVol], n: int, strict_15m: bool = True) -> List[Setup]:
    global _REJECT_STATS, _REJECT_SAMPLES
    _REJECT_STATS = Counter()
    _REJECT_SAMPLES = {}

    universe = sorted(best_fut.items(), key=lambda kv: usd_notional(kv[1]), reverse=True)[:35]
    setups: List[Setup] = []
    for base, mv in universe:
        s = make_setup(base, mv, strict_15m=strict_15m)
        if s:
            setups.append(s)

    setups.sort(key=lambda x: (x.conf, x.fut_vol_usd), reverse=True)
    return setups[:n]


# =========================================================
# EMAIL
# =========================================================
def email_config_ok() -> bool:
    return all([EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FROM, EMAIL_TO])

def send_email(subject: str, body: str) -> bool:
    if not email_config_ok():
        logger.warning("Email not configured.")
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
        logger.exception("send_email failed: %s", e)
        return False


# =========================================================
# SESSIONS
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
        adv.append("📌 Data کم است: حداقل 5–10 ترید ببند تا آمار معنی‌دار شود.")
    if stats["win_rate"] < 45 and stats["closed_n"] >= 5:
        adv.append("⚠️ وین‌ریت پایین است: تعداد ستاپ‌ها را کمتر کن و فقط Conf بالا را ترید کن.")
    if stats["avg_r"] is not None and stats["avg_r"] < 0.2 and stats["closed_n"] >= 5:
        adv.append("⚠️ R پایین است: یا استاپ خیلی نزدیک است یا تی‌پی‌ها زود بسته می‌شوند.")
    if stats["biggest_loss"] < -0.9 * max(1.0, float(user["equity"]) * 0.01):
        adv.append("🛑 یک باخت بزرگ داری: روی اجرای حد ضرر و اسلیپیج/ری‌انتر کنترل بگذار.")
    if int(user["max_trades_day"]) <= 5:
        adv.append("✅ محدودیت ترید روزانه کمک می‌کند overtrade نکنی. فقط بهترین‌ها را بگیر.")
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

Examples:
- /trade_open BTC long entry 43000 sl 42000 risk usd 40 note breakout sig PF-20251219-0007
- /trade_open ETH short entry 2520 sl 2575 risk pct 2 note trend

Close trade:
- /trade_close <TRADE_ID> pnl <PNL>

Equity behavior:
- Equity updates ONLY when trades are closed
- Persistent until reset:
  /equity_reset

4) Status
- /status
Shows:
• Open trades
• Daily trade count
• Daily risk used & remaining
• Equity

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

Not financial advice.
"""

# =========================================================
# TELEGRAM COMMANDS
# =========================================================
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

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
        risk_mode = "PCT"
        risk_val = float(DEFAULT_MAX_RISK_PCT_PER_TRADE)

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
            f"⚠️ Daily Risk Warning: این پوزیشن (${risk_usd:.2f}) از ریسک باقیمانده امروز (${max(0.0, remaining_before):.2f}) بیشتره.\n"
        )

    warn_trade_vs_cap = ""
    if cap > 0 and float(risk_usd) > float(cap) + 1e-9:
        warn_trade_vs_cap = f"⚠️ Note: Risk این ترید (${risk_usd:.2f}) از کل Daily Cap (${cap:.2f}) بیشتره.\n"

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
# /screen
# =========================================================
async def screen_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("⏳ Scanning market…")

        best_fut = await asyncio.to_thread(fetch_futures_tickers)
        if not best_fut:
            await update.message.reply_text("Error: could not fetch futures tickers.")
            return

        leaders_txt = await asyncio.to_thread(build_leaders_table, best_fut)
        up_txt, dn_txt = await asyncio.to_thread(movers_tables, best_fut)

        setups = await asyncio.to_thread(pick_setups, best_fut, SETUPS_N, False)
        for s in setups:
            db_insert_signal(s)

        if setups:
            setup_blocks = []
            for i, s in enumerate(setups, 1):
                tps = f"TP {fmt_price(s.tp3)}"
                if s.tp1 and s.tp2 and s.conf >= MULTI_TP_MIN_CONF:
                    tps = f"TP1 {fmt_price(s.tp1)} | TP2 {fmt_price(s.tp2)} | TP3 {fmt_price(s.tp3)}"
                rr3 = rr_to_tp(s.entry, s.sl, s.tp3)
                tp3_mode = "TRAILING (hot coin)" if s.is_trailing_tp3 else "FIXED"
                setup_blocks.append(
                    f"🔥 Setup #{i} — {s.side} {s.symbol} — Conf {s.conf}/100\n"
                    f"ID: {s.setup_id}\n"
                    f"Entry {fmt_price(s.entry)} | SL {fmt_price(s.sl)}\n"
                    f"{tps}\n"
                    f"RR(TP3): {rr3:.2f}\n"
                    f"EMA12 dist (15m): {s.ema12_dist_pct:.2f}%\n"
                    f"TP3 Mode: {tp3_mode}\n"
                    f"24H {pct_with_emoji(s.ch24)} | 4H {pct_with_emoji(s.ch4)} | 1H {pct_with_emoji(s.ch1)} | 15m {pct_with_emoji(s.ch15)} | Vol~{fmt_money(s.fut_vol_usd)}\n"
                    f"Chart: {tv_chart_url(s.symbol)}"
                )
            setups_txt = "\n\n".join(setup_blocks)
        else:
            setups_txt = "No high-quality setups right now."

        diag_txt = _reject_report()
        if diag_txt:
            setups_txt = setups_txt + "\n\n" + diag_txt

        kb = []
        for s in setups:
            kb.append([InlineKeyboardButton(text=f"📈 {s.symbol} ({s.setup_id})", url=tv_chart_url(s.symbol))])

        msg = (
            f"✨ PulseFutures — Market Scan\n"
            f"{HDR}\n"
            f"— Top Trade Setups\n"
            f"{setups_txt}\n\n"
            f"— Directional Leaders / Losers\n"
            f"{up_txt}\n\n{dn_txt}\n\n"
            f"— Market Leaders\n"
            f"{leaders_txt}"
        )

        await send_long_message(
            update,
            msg,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup(kb) if kb else None,
        )

    except Exception as e:
        logger.exception("screen_cmd failed: %s", e)
        try:
            await update.message.reply_text(f"⚠️ /screen failed: {e}")
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
        if not EMAIL_ENABLED:
            return
        if not email_config_ok():
            return

        # ✅ Melbourne blackout window: no emails either
        if is_blackout_melbourne_now():
            return

        users = list_users_notify_on()
        if not users:
            return

        best_fut = await asyncio.to_thread(fetch_futures_tickers)
        if not best_fut:
            return

        setups_all = await asyncio.to_thread(pick_setups, best_fut, max(EMAIL_SETUPS_N * 3, 9), True)

        for s in setups_all:
            db_insert_signal(s)

        for user in users:
            uid = int(user["user_id"])
            sess = in_session_now(user)
            if not sess:
                continue

            st = email_state_get(uid)

            if st["session_key"] != sess["session_key"]:
                email_state_set(uid, session_key=sess["session_key"], sent_count=0, last_email_ts=0.0)
                st = email_state_get(uid)

            max_emails = int(user["max_emails_per_session"])
            gap_min = int(user["email_gap_min"])
            gap_sec = gap_min * 60

            tz = ZoneInfo(user["tz"])
            day_local = datetime.now(tz).date().isoformat()
            sent_today = _email_daily_get(uid, day_local)
            day_cap = int(user.get("max_emails_per_day", DEFAULT_MAX_EMAILS_PER_DAY))
            if day_cap > 0 and sent_today >= day_cap:
                continue

            if max_emails > 0 and int(st["sent_count"]) >= max_emails:
                continue

            now_ts = time.time()
            if gap_sec > 0 and (now_ts - float(st["last_email_ts"])) < gap_sec:
                continue

            min_conf = SESSION_MIN_CONF.get(sess["name"], 78)

            confirmed: List[Setup] = []
            early: List[Setup] = []

            for s in setups_all:
                if s.conf < min_conf:
                    continue

                # Trend-follow filter
                if s.side == "BUY" and float(s.ch24) < float(TREND_24H_TOL):
                    continue
                if s.side == "SELL" and float(s.ch24) > -float(TREND_24H_TOL):
                    continue

                rr3 = rr_to_tp(s.entry, s.sl, s.tp3)
                if rr3 < MIN_RR_TP3:
                    continue

                if symbol_recently_emailed(uid, s.symbol, SYMBOL_COOLDOWN_HOURS):
                    continue

                is_confirm_15m = abs(float(s.ch15)) >= CONFIRM_15M_ABS_MIN

                if is_confirm_15m:
                    confirmed.append(s)
                else:
                    if abs(float(s.ch1)) < EARLY_1H_ABS_MIN:
                        continue
                    if s.conf < (min_conf + EARLY_EMAIL_EXTRA_CONF):
                        continue
                    early.append(s)

                if len(confirmed) >= EMAIL_SETUPS_N:
                    break

            filtered: List[Setup] = confirmed[:EMAIL_SETUPS_N]
            if len(filtered) < EMAIL_SETUPS_N and early:
                need = EMAIL_SETUPS_N - len(filtered)
                filtered.extend(early[:min(need, EARLY_EMAIL_MAX_FILL)])

            if not filtered:
                continue

            now_local = datetime.now(tz)
            body = _email_body_pretty(sess["name"], now_local, user["tz"], filtered, best_fut)

            subject = f"PulseFutures • {sess['name']} • Premium Setups ({int(st['sent_count'])+1})"

            ok = await asyncio.to_thread(send_email, subject, body)
            if ok:
                email_state_set(uid, sent_count=int(st["sent_count"]) + 1, last_email_ts=now_ts)
                _email_daily_inc(uid, day_local, 1)
                for s in filtered:
                    mark_symbol_emailed(uid, s.symbol)


# =========================================================
# MAIN (Polling or Webhook)
# =========================================================
def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN missing")

    db_init()

    app = Application.builder().token(TOKEN).build()

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

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    if getattr(app, "job_queue", None):
        app.job_queue.run_repeating(alert_job, interval=CHECK_INTERVAL_MIN * 60, first=15, name="alert_job")
    else:
        logger.warning('JobQueue not available. Install: python-telegram-bot[job-queue]')

    port = int(os.environ.get("PORT", "10000"))

    if WEBHOOK_URL:
        if not WEBHOOK_URL.startswith("https://"):
            raise RuntimeError("WEBHOOK_URL must start with https://")
        path = f"/telegram/{TOKEN[:12]}"
        webhook_full = WEBHOOK_URL.rstrip("/") + path
        logger.info("Starting WEBHOOK mode: %s", webhook_full)
        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=path.lstrip("/"),
            webhook_url=webhook_full,
            drop_pending_updates=True,
        )
    else:
        logger.info("Starting POLLING mode (ensure ONLY ONE instance running).")
        app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
