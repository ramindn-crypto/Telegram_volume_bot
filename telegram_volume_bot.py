#!/usr/bin/env python3
"""
PulseFutures — Bybit Futures Screener + Risk Manager + Trading Journal + Email Signal Alerts

================================================================================
WHAT THIS BOT DOES
================================================================================
1) Telegram:
   - /screen  : fast market scan (leaders + directional leaders/losers + top setups)
   - /size    : position sizing using ONLY (risk + stoploss) and live price
   - Journal  : user records trades (open/close), equity auto-updates with PnL
   - /status  : full risk dashboard + open trades + today's trades count
   - Reports  : daily/weekly summaries + advice based on user's trades
   - Signal report: daily/weekly summary of emailed setups with unique Setup IDs

2) Email (core feature):
   - Sends 3–4 emails per enabled session (NY + London enabled by default)
   - Minimum 60 minutes gap between emails
   - No repeated symbol for a user within cooldown window (default 18 hours)
   - Each emailed setup has a unique SetupID so you can reference it in your channel

================================================================================
RENDER DEPLOYMENT (IMPORTANT)
================================================================================
To avoid Telegram "409 Conflict terminated by other getUpdates request":
- Use WEBHOOK mode on Render (recommended).
- Set env:
  WEBHOOK_URL=https://<your-service>.onrender.com
  PORT is provided by Render automatically.

If WEBHOOK_URL is not set, bot falls back to polling (works only if ONE instance runs).

================================================================================
ENV VARS
================================================================================
Required:
- TELEGRAM_TOKEN

Optional Email:
- EMAIL_ENABLED=true/false   (default true if email config exists)
- EMAIL_HOST, EMAIL_PORT (465 SSL or 587 TLS)
- EMAIL_USER, EMAIL_PASS
- EMAIL_FROM, EMAIL_TO (comma-separated allowed)

Optional Behavior:
- CHECK_INTERVAL_MIN=5
- SYMBOL_COOLDOWN_HOURS=18
- DEFAULT_EMAILS_PER_SESSION=4
- EMAIL_GAP_MIN=60
- USE_WEBHOOK=true/false  (auto if WEBHOOK_URL exists)
- DB_PATH=pulsefutures.db

Not financial advice.
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
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Dict, List, Optional, Tuple

from zoneinfo import ZoneInfo

import ccxt
from tabulate import tabulate
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# =========================
# CONFIG
# =========================
EXCHANGE_ID = "bybit"
DEFAULT_TYPE = "swap"

TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
DB_PATH = os.environ.get("DB_PATH", "pulsefutures.db")

CHECK_INTERVAL_MIN = int(os.environ.get("CHECK_INTERVAL_MIN", "5"))

# Email
EMAIL_HOST = os.environ.get("EMAIL_HOST", "").strip()
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "465"))
EMAIL_USER = os.environ.get("EMAIL_USER", "").strip()
EMAIL_PASS = os.environ.get("EMAIL_PASS", "").strip()
EMAIL_FROM = os.environ.get("EMAIL_FROM", EMAIL_USER).strip()
EMAIL_TO_RAW = os.environ.get("EMAIL_TO", EMAIL_USER).strip()
EMAIL_TO = [x.strip() for x in EMAIL_TO_RAW.split(",") if x.strip()]

EMAIL_CONFIG_OK = bool(EMAIL_HOST and EMAIL_PORT and EMAIL_USER and EMAIL_PASS and EMAIL_FROM and EMAIL_TO)
EMAIL_ENABLED = os.environ.get("EMAIL_ENABLED", "true").lower().strip() == "true" and EMAIL_CONFIG_OK

# Email pacing & cooldown
DEFAULT_EMAILS_PER_SESSION = int(os.environ.get("DEFAULT_EMAILS_PER_SESSION", "4"))  # 3-4 recommended
EMAIL_GAP_MIN = int(os.environ.get("EMAIL_GAP_MIN", "60"))
SYMBOL_COOLDOWN_HOURS = int(os.environ.get("SYMBOL_COOLDOWN_HOURS", "18"))

# Signals content
LEADERS_N = 10
DIR_N = 10
SETUPS_N = 3
MOVER_VOL_USD_MIN = 5_000_000  # min futures vol to be considered

# Setup engine thresholds (tunable)
TRIGGER_1H_ABS_MIN = 2.0
CONFIRM_15M_ABS_MIN = 0.6
ALIGN_4H_MIN = 0.0

# ATR & TP logic
ATR_PERIOD = 14
ATR_MIN_PCT = 0.7
ATR_MAX_PCT = 6.0
TP_MAX_PCT = 5.0
MULTI_TP_MIN_CONF = 75
TP_ALLOCS = (40, 40, 20)
TP_R_MULTS = (0.8, 1.4, 2.0)

# Limits
DEFAULT_MAX_TRADES_DAY = 5
DEFAULT_DAILY_CAP_MODE = "PCT"    # PCT or USD
DEFAULT_DAILY_CAP_VALUE = 5.0     # % per day
DEFAULT_RISK_MODE = "PCT"         # PCT or USD
DEFAULT_RISK_VALUE = 1.5          # % per trade

# Sessions are defined in UTC (stable globally), converted per user TZ:
# Approx:
# - Asia session: 00:00-03:00 UTC
# - London session: 07:00-10:00 UTC
# - NY session: 13:00-17:00 UTC
SESSION_DEFS_UTC = {
    "ASIA":   {"start": "00:00", "end": "03:00"},
    "LONDON": {"start": "07:00", "end": "10:00"},
    "NY":     {"start": "13:00", "end": "17:00"},
}
DEFAULT_ENABLED_SESSIONS = ["NY", "LONDON"]  # per your preference

STABLES = {"USDT", "USDC", "USD", "TUSD", "FDUSD", "DAI", "PYUSD"}

# Speed locks
ALERT_LOCK = asyncio.Lock()
SCREEN_LOCK = asyncio.Lock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pulsefutures")


# =========================
# DATA STRUCTURES
# =========================
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
    symbol: str          # base e.g. BTC
    market_symbol: str   # e.g. BTC/USDT:USDT
    side: str            # BUY/SELL
    conf: int
    entry: float
    sl: float
    tp: float
    tp1: Optional[float]
    tp2: Optional[float]
    fut_vol_usd: float
    ch24: float
    ch4: float
    ch1: float
    ch15: float
    created_ts: float


# =========================
# DB
# =========================
def db_connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def db_init():
    con = db_connect()
    cur = con.cursor()

    # users
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        tz TEXT NOT NULL,
        email_on INTEGER NOT NULL DEFAULT 1,
        equity REAL NOT NULL DEFAULT 0,
        risk_mode TEXT NOT NULL DEFAULT 'PCT',
        risk_value REAL NOT NULL DEFAULT 1.5,
        daily_cap_mode TEXT NOT NULL DEFAULT 'PCT',
        daily_cap_value REAL NOT NULL DEFAULT 5.0,
        max_trades_day INTEGER NOT NULL DEFAULT 5,
        day_date TEXT NOT NULL DEFAULT '',
        day_trade_count INTEGER NOT NULL DEFAULT 0,
        daily_risk_used REAL NOT NULL DEFAULT 0,
        enabled_sessions TEXT NOT NULL DEFAULT '["NY","LONDON"]',
        emails_per_session INTEGER NOT NULL DEFAULT 4,
        email_gap_min INTEGER NOT NULL DEFAULT 60
    )
    """)

    # trades journal
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        entry REAL NOT NULL,
        sl REAL NOT NULL,
        risk_usd REAL NOT NULL,
        opened_ts REAL NOT NULL,
        status TEXT NOT NULL DEFAULT 'OPEN',
        close_ts REAL,
        pnl REAL,
        notes TEXT
    )
    """)

    # email state (per user per session)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS email_state (
        user_id INTEGER NOT NULL,
        session_name TEXT NOT NULL,
        session_key TEXT NOT NULL,
        sent_count INTEGER NOT NULL DEFAULT 0,
        last_email_ts REAL NOT NULL DEFAULT 0,
        PRIMARY KEY (user_id, session_name)
    )
    """)

    # symbol cooldown per user
    cur.execute("""
    CREATE TABLE IF NOT EXISTS symbol_cooldown (
        user_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        last_sent_ts REAL NOT NULL,
        PRIMARY KEY (user_id, symbol)
    )
    """)

    # setup unique seq per day (global)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS setup_seq (
        day TEXT PRIMARY KEY,
        seq INTEGER NOT NULL
    )
    """)

    # setups sent (global storage for referencing in channel)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS setups_sent (
        setup_id TEXT PRIMARY KEY,
        sent_ts REAL NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        conf INTEGER NOT NULL,
        entry REAL NOT NULL,
        sl REAL NOT NULL,
        tp REAL NOT NULL,
        tp1 REAL,
        tp2 REAL,
        ch24 REAL NOT NULL,
        ch4 REAL NOT NULL,
        ch1 REAL NOT NULL,
        ch15 REAL NOT NULL,
        fut_vol_usd REAL NOT NULL
    )
    """)

    # mapping: which user got which setup (for per-user signal report)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_setups (
        user_id INTEGER NOT NULL,
        setup_id TEXT NOT NULL,
        sent_ts REAL NOT NULL,
        session_name TEXT NOT NULL,
        PRIMARY KEY (user_id, setup_id)
    )
    """)

    con.commit()
    con.close()


def get_user(user_id: int) -> dict:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    if not row:
        # default TZ from Telegram may not exist; start with UTC
        tz = "UTC"
        today = datetime.now(timezone.utc).date().isoformat()
        cur.execute("""
            INSERT INTO users (
                user_id, tz, email_on, equity, risk_mode, risk_value,
                daily_cap_mode, daily_cap_value, max_trades_day,
                day_date, day_trade_count, daily_risk_used,
                enabled_sessions, emails_per_session, email_gap_min
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            tz,
            1 if EMAIL_ENABLED else 0,
            0.0,  # IMPORTANT: no default equity
            DEFAULT_RISK_MODE,
            float(DEFAULT_RISK_VALUE),
            DEFAULT_DAILY_CAP_MODE,
            float(DEFAULT_DAILY_CAP_VALUE),
            int(DEFAULT_MAX_TRADES_DAY),
            today,
            0,
            0.0,
            json.dumps(DEFAULT_ENABLED_SESSIONS),
            int(DEFAULT_EMAILS_PER_SESSION),
            int(EMAIL_GAP_MIN),
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
    tz = ZoneInfo(user.get("tz") or "UTC")
    today = datetime.now(tz).date().isoformat()
    if user.get("day_date") != today:
        update_user(user["user_id"], day_date=today, day_trade_count=0, daily_risk_used=0.0)
        user = get_user(user["user_id"])
    return user


def list_users_email_on() -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE email_on=1")
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


def trade_open_db(user_id: int, symbol: str, side: str, entry: float, sl: float, risk_usd: float, notes: str):
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO trades (user_id, symbol, side, entry, sl, risk_usd, opened_ts, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
    """, (user_id, symbol, side, entry, sl, risk_usd, time.time(), notes))
    con.commit()
    con.close()


def trade_close_db(user_id: int, symbol: str, pnl: float) -> Optional[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        SELECT * FROM trades
        WHERE user_id=? AND status='OPEN' AND symbol=?
        ORDER BY opened_ts DESC
        LIMIT 1
    """, (user_id, symbol))
    row = cur.fetchone()
    if not row:
        con.close()
        return None
    trade = dict(row)
    cur.execute("""
        UPDATE trades SET status='CLOSED', pnl=?, close_ts=?
        WHERE id=?
    """, (pnl, time.time(), trade["id"]))
    con.commit()
    con.close()
    trade["pnl"] = pnl
    return trade


def get_open_trades(user_id: int) -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        SELECT * FROM trades
        WHERE user_id=? AND status='OPEN'
        ORDER BY opened_ts ASC
    """, (user_id,))
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_trades_between(user_id: int, ts_from: float, ts_to: float) -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        SELECT * FROM trades
        WHERE user_id=? AND opened_ts BETWEEN ? AND ?
        ORDER BY opened_ts ASC
    """, (user_id, ts_from, ts_to))
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


def email_state_get(user_id: int, session_name: str) -> dict:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM email_state WHERE user_id=? AND session_name=?", (user_id, session_name))
    row = cur.fetchone()
    if not row:
        cur.execute("""
            INSERT INTO email_state (user_id, session_name, session_key, sent_count, last_email_ts)
            VALUES (?, ?, ?, 0, 0)
        """, (user_id, session_name, ""))
        con.commit()
        cur.execute("SELECT * FROM email_state WHERE user_id=? AND session_name=?", (user_id, session_name))
        row = cur.fetchone()
    con.close()
    return dict(row)


def email_state_set(user_id: int, session_name: str, **kwargs):
    if not kwargs:
        return
    con = db_connect()
    cur = con.cursor()
    sets = ", ".join([f"{k}=?" for k in kwargs.keys()])
    vals = list(kwargs.values()) + [user_id, session_name]
    cur.execute(f"UPDATE email_state SET {sets} WHERE user_id=? AND session_name=?", vals)
    con.commit()
    con.close()


def cooldown_ok(user_id: int, symbol: str, now_ts: float) -> bool:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT last_sent_ts FROM symbol_cooldown WHERE user_id=? AND symbol=?", (user_id, symbol))
    row = cur.fetchone()
    con.close()
    if not row:
        return True
    last_ts = float(row["last_sent_ts"])
    return (now_ts - last_ts) >= (SYMBOL_COOLDOWN_HOURS * 3600)


def cooldown_mark(user_id: int, symbol: str, now_ts: float):
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO symbol_cooldown (user_id, symbol, last_sent_ts)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, symbol) DO UPDATE SET last_sent_ts=excluded.last_sent_ts
    """, (user_id, symbol, now_ts))
    con.commit()
    con.close()


def next_setup_id(now_utc: datetime) -> str:
    day = now_utc.strftime("%Y%m%d")
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT seq FROM setup_seq WHERE day=?", (day,))
    row = cur.fetchone()
    if not row:
        seq = 1
        cur.execute("INSERT INTO setup_seq (day, seq) VALUES (?, ?)", (day, seq))
    else:
        seq = int(row["seq"]) + 1
        cur.execute("UPDATE setup_seq SET seq=? WHERE day=?", (seq, day))
    con.commit()
    con.close()
    return f"PF-{day}-{seq:04d}"


def store_setup_sent(setup: Setup):
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO setups_sent (
            setup_id, sent_ts, symbol, side, conf, entry, sl, tp, tp1, tp2,
            ch24, ch4, ch1, ch15, fut_vol_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        setup.setup_id,
        setup.created_ts,
        setup.symbol,
        setup.side,
        setup.conf,
        setup.entry,
        setup.sl,
        setup.tp,
        setup.tp1,
        setup.tp2,
        setup.ch24,
        setup.ch4,
        setup.ch1,
        setup.ch15,
        setup.fut_vol_usd,
    ))
    con.commit()
    con.close()


def store_user_setup(user_id: int, setup_id: str, sent_ts: float, session_name: str):
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO user_setups (user_id, setup_id, sent_ts, session_name)
        VALUES (?, ?, ?, ?)
    """, (user_id, setup_id, sent_ts, session_name))
    con.commit()
    con.close()


def get_user_setups_between(user_id: int, ts_from: float, ts_to: float) -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        SELECT us.setup_id, us.sent_ts, us.session_name, ss.symbol, ss.side, ss.conf, ss.entry, ss.sl, ss.tp, ss.tp1, ss.tp2
        FROM user_setups us
        JOIN setups_sent ss ON ss.setup_id = us.setup_id
        WHERE us.user_id=? AND us.sent_ts BETWEEN ? AND ?
        ORDER BY us.sent_ts ASC
    """, (user_id, ts_from, ts_to))
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


# =========================
# EXCHANGE HELPERS (FAST)
# =========================
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


def usd_notional(mv: MarketVol) -> float:
    if mv.quote in STABLES and mv.quote_vol:
        return float(mv.quote_vol)
    px = mv.vwap if mv.vwap else mv.last
    if not px or not mv.base_vol:
        return 0.0
    return float(mv.base_vol) * float(px)


def fetch_futures_tickers() -> Dict[str, MarketVol]:
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
    return best


def fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> List[List[float]]:
    ex = build_exchange()
    ex.load_markets()
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit) or []


def compute_atr(candles: List[List[float]], period: int) -> float:
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


def metrics_1h_4h_15m_atr(market_symbol: str) -> Tuple[float, float, float, float]:
    need_1h = max(ATR_PERIOD + 10, 40)
    c1 = fetch_ohlcv(market_symbol, "1h", need_1h)
    if len(c1) < 6:
        return 0.0, 0.0, 0.0, 0.0
    closes = [float(x[4]) for x in c1]
    last = closes[-1]
    prev1 = closes[-2]
    prev4 = closes[-5] if len(closes) >= 5 else closes[0]
    ch1 = ((last - prev1) / prev1) * 100 if prev1 else 0.0
    ch4 = ((last - prev4) / prev4) * 100 if prev4 else 0.0
    atr = compute_atr(c1, ATR_PERIOD)

    c15 = fetch_ohlcv(market_symbol, "15m", 3)
    if len(c15) < 2:
        ch15 = 0.0
    else:
        c15_last = float(c15[-1][4])
        c15_prev = float(c15[-2][4])
        ch15 = ((c15_last - c15_prev) / c15_prev) * 100 if c15_prev else 0.0

    return ch1, ch4, ch15, atr


# =========================
# SETUP ENGINE
# =========================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_conf(side: str, ch24: float, ch4: float, ch1: float, ch15: float, vol_usd: float) -> int:
    score = 50.0
    is_long = (side == "BUY")

    def align(x: float, w: float):
        nonlocal score
        if is_long:
            score += w if x > 0 else -w
        else:
            score += w if x < 0 else -w

    align(ch24, 12)
    align(ch4, 10)
    align(ch1, 8)
    align(ch15, 6)

    mag = min(abs(ch24) / 2 + abs(ch4) + abs(ch1) * 2 + abs(ch15) * 2, 18)
    score += mag

    if vol_usd >= 15_000_000:
        score += 8
    elif vol_usd >= 6_000_000:
        score += 6
    elif vol_usd >= 2_000_000:
        score += 4

    return int(round(clamp(score, 0, 100)))


def sl_mult_from_conf(conf: int) -> float:
    if conf >= 85:
        return 2.0
    if conf >= 75:
        return 1.7
    if conf >= 65:
        return 1.45
    return 1.25


def sl_tp_from_atr(entry: float, side: str, atr: float, conf: int) -> Tuple[float, float, float]:
    if entry <= 0 or atr <= 0:
        return 0.0, 0.0, 0.0

    sl_dist = sl_mult_from_conf(conf) * atr
    sl_dist = clamp(sl_dist, (ATR_MIN_PCT / 100) * entry, (ATR_MAX_PCT / 100) * entry)
    r = sl_dist

    tp_dist = 1.5 * r
    tp_dist = min(tp_dist, (TP_MAX_PCT / 100) * entry)

    if side == "BUY":
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist

    return sl, tp, r


def ensure_strict_tps(entry: float, side: str, tp1: float, tp2: float, tp3: float) -> Tuple[float, float, float]:
    # Fix equal/overlapping TP values robustly.
    # Step is a tiny fraction of price (or R) to break ties.
    eps = max(entry * 0.0002, 1e-8)  # 0.02%
    if side == "BUY":
        a, b, c = sorted([tp1, tp2, tp3])
        if b <= a:
            b = a + eps
        if c <= b:
            c = b + eps
        return a, b, c
    else:
        a, b, c = sorted([tp1, tp2, tp3], reverse=True)
        if b >= a:
            b = a - eps
        if c >= b:
            c = b - eps
        return a, b, c


def multi_tp(entry: float, side: str, r: float) -> Tuple[float, float, float]:
    r1, r2, r3 = TP_R_MULTS
    maxd = (TP_MAX_PCT / 100) * entry
    d3 = min(r3 * r, maxd)
    d1 = d3 * (r1 / r3)
    d2 = d3 * (r2 / r3)

    if side == "BUY":
        tp1, tp2, tp3 = entry + d1, entry + d2, entry + d3
    else:
        tp1, tp2, tp3 = entry - d1, entry - d2, entry - d3

    return ensure_strict_tps(entry, side, tp1, tp2, tp3)


def make_setup(now_utc: datetime, base: str, mv: MarketVol) -> Optional[Setup]:
    vol = usd_notional(mv)
    if vol <= 0:
        return None

    entry = float(mv.last or 0.0)
    if entry <= 0:
        return None

    ch24 = float(mv.percentage or 0.0)
    ch1, ch4, ch15, atr = metrics_1h_4h_15m_atr(mv.symbol)

    # momentum gate
    if abs(ch1) < TRIGGER_1H_ABS_MIN or abs(ch15) < CONFIRM_15M_ABS_MIN:
        return None

    side = "BUY" if ch1 > 0 else "SELL"

    # trend alignment gate
    if side == "BUY" and ch4 < ALIGN_4H_MIN:
        return None
    if side == "SELL" and ch4 > -ALIGN_4H_MIN:
        return None

    conf = compute_conf(side, ch24, ch4, ch1, ch15, vol)
    sl, tp_single, r = sl_tp_from_atr(entry, side, atr, conf)
    if sl <= 0 or tp_single <= 0 or r <= 0:
        return None

    tp1 = tp2 = None
    tp = tp_single
    if conf >= MULTI_TP_MIN_CONF:
        t1, t2, t3 = multi_tp(entry, side, r)
        tp1, tp2, tp = t1, t2, t3

    setup_id = next_setup_id(now_utc)
    return Setup(
        setup_id=setup_id,
        symbol=base,
        market_symbol=mv.symbol,
        side=side,
        conf=conf,
        entry=entry,
        sl=sl,
        tp=tp,
        tp1=tp1,
        tp2=tp2,
        fut_vol_usd=vol,
        ch24=ch24,
        ch4=ch4,
        ch1=ch1,
        ch15=ch15,
        created_ts=time.time(),
    )


def pick_setups(now_utc: datetime, best: Dict[str, MarketVol]) -> List[Setup]:
    # speed: scan top 30 by volume only
    top = sorted(best.items(), key=lambda kv: usd_notional(kv[1]), reverse=True)[:30]
    out = []
    for base, mv in top:
        s = make_setup(now_utc, base, mv)
        if s:
            out.append(s)
    out.sort(key=lambda x: (x.conf, x.fut_vol_usd), reverse=True)
    return out[:SETUPS_N]


def directional_lists(best: Dict[str, MarketVol]) -> Tuple[List[Tuple], List[Tuple]]:
    up, dn = [], []
    for base, mv in best.items():
        vol = usd_notional(mv)
        if vol < MOVER_VOL_USD_MIN:
            continue
        ch24 = float(mv.percentage or 0.0)
        if ch24 >= 10:
            ch1, ch4, ch15, atr = metrics_1h_4h_15m_atr(mv.symbol)
            up.append((base, vol, ch24, ch4, float(mv.last or 0.0)))
        elif ch24 <= -10:
            ch1, ch4, ch15, atr = metrics_1h_4h_15m_atr(mv.symbol)
            dn.append((base, vol, ch24, ch4, float(mv.last or 0.0)))
    up.sort(key=lambda x: (x[2], x[1]), reverse=True)
    dn.sort(key=lambda x: (x[2], x[1]))
    return up[:DIR_N], dn[:DIR_N]


# =========================
# FORMAT
# =========================
def fmt_price(x: float) -> str:
    ax = abs(x)
    if ax >= 100:
        return f"{x:.2f}"
    if ax >= 1:
        return f"{x:.4f}"
    if ax >= 0.1:
        return f"{x:.5f}"
    return f"{x:.6f}"


def fmt_money(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:.0f}"


def pct_emoji(p: float) -> str:
    v = int(round(p))
    if v >= 3:
        e = "🟢"
    elif v <= -3:
        e = "🔴"
    else:
        e = "🟡"
    return f"{v:+d}% {e}"


def tv_url(base: str) -> str:
    # bybit perpetual style
    return f"https://www.tradingview.com/chart/?symbol=BYBIT:{base.upper()}USDT.P"


def table_md(rows: List[List], headers: List[str]) -> str:
    return "```\n" + tabulate(rows, headers=headers, tablefmt="github") + "\n```"


def setup_block_md(s: Setup, idx: int) -> str:
    if s.tp1 and s.tp2 and s.conf >= MULTI_TP_MIN_CONF:
        tp_line = f"TP1 {fmt_price(s.tp1)} ({TP_ALLOCS[0]}%) | TP2 {fmt_price(s.tp2)} ({TP_ALLOCS[1]}%) | TP3 {fmt_price(s.tp)} ({TP_ALLOCS[2]}%)"
        rule = "After TP1 → move SL to BE"
    else:
        tp_line = f"TP {fmt_price(s.tp)}"
        rule = ""

    lines = [
        f"🔥 *Setup #{idx}*  |  *{s.setup_id}*",
        f"*{s.side} {s.symbol}*  • Conf *{s.conf}/100*",
        f"Entry {fmt_price(s.entry)} | SL {fmt_price(s.sl)}",
        tp_line,
        f"24H {pct_emoji(s.ch24)} | 4H {pct_emoji(s.ch4)} | 1H {pct_emoji(s.ch1)} | F~{fmt_money(s.fut_vol_usd)}",
    ]
    if rule:
        lines.append(f"🧠 {rule}")
    lines.append(f"📈 {tv_url(s.symbol)}")
    return "\n".join(lines)


# =========================
# RISK / SIZING
# =========================
def parse_risk_value(risk_str: str, equity: float) -> Tuple[float, str]:
    # returns (risk_usd, label)
    rs = risk_str.strip().upper()
    if rs.endswith("%"):
        if equity <= 0:
            raise ValueError("Equity is 0. Set equity first: /equity 1000")
        pct = float(rs[:-1])
        if pct <= 0:
            raise ValueError("Risk% must be > 0")
        return equity * (pct / 100.0), f"{pct:.2f}%"
    # assume USD
    val = float(rs)
    if val <= 0:
        raise ValueError("Risk USD must be > 0")
    return val, f"${val:.2f}"


def qty_from_risk(entry: float, sl: float, risk_usd: float) -> float:
    d = abs(entry - sl)
    if d <= 0:
        return 0.0
    return risk_usd / d


def daily_cap_usd(user: dict) -> float:
    mode = (user.get("daily_cap_mode") or DEFAULT_DAILY_CAP_MODE).upper()
    val = float(user.get("daily_cap_value") or DEFAULT_DAILY_CAP_VALUE)
    eq = float(user.get("equity") or 0.0)
    if mode == "USD":
        return max(0.0, val)
    return max(0.0, eq * (val / 100.0))


# =========================
# SESSIONS (UTC -> USER LOCAL)
# =========================
def parse_hhmm(s: str) -> Tuple[int, int]:
    m = re.match(r"^(\d{1,2}):(\d{2})$", s.strip())
    if not m:
        raise ValueError("Bad time format")
    hh = int(m.group(1))
    mm = int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError("Bad time range")
    return hh, mm


def enabled_sessions(user: dict) -> List[str]:
    try:
        raw = user.get("enabled_sessions") or "[]"
        arr = json.loads(raw)
        out = []
        for x in arr:
            t = str(x).strip().upper()
            if t in SESSION_DEFS_UTC:
                out.append(t)
        return out or DEFAULT_ENABLED_SESSIONS
    except Exception:
        return DEFAULT_ENABLED_SESSIONS


def session_window_local(user_tz: ZoneInfo, session_name: str, now_utc: datetime) -> Tuple[datetime, datetime]:
    spec = SESSION_DEFS_UTC[session_name]
    sh, sm = parse_hhmm(spec["start"])
    eh, em = parse_hhmm(spec["end"])

    # anchor on today's UTC date
    start_utc = now_utc.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end_utc = now_utc.replace(hour=eh, minute=em, second=0, microsecond=0)

    if end_utc <= start_utc:
        end_utc += timedelta(days=1)

    # convert to user's local tz
    return start_utc.astimezone(user_tz), end_utc.astimezone(user_tz)


def session_key_for_day(session_name: str, now_utc: datetime) -> str:
    # key per UTC day + session name
    return f"{now_utc.strftime('%Y-%m-%d')}_{session_name}"


def is_in_session(user: dict, session_name: str, now_utc: datetime) -> bool:
    tz = ZoneInfo(user.get("tz") or "UTC")
    start_l, end_l = session_window_local(tz, session_name, now_utc)
    now_l = now_utc.astimezone(tz)
    return start_l <= now_l <= end_l


def sessions_status_text(user: dict) -> str:
    tz = ZoneInfo(user.get("tz") or "UTC")
    now_utc = datetime.now(timezone.utc)
    en = enabled_sessions(user)

    lines = [f"TZ: {user.get('tz')} | Enabled: {', '.join(en)}"]
    for name in ["ASIA", "LONDON", "NY"]:
        start_l, end_l = session_window_local(tz, name, now_utc)
        active = "✅ ACTIVE" if is_in_session(user, name, now_utc) else "—"
        lines.append(f"- {name:<6} {start_l.strftime('%H:%M')}–{end_l.strftime('%H:%M')}  {active}")
    return "\n".join(lines)


# =========================
# EMAIL
# =========================
def send_email(subject: str, body: str) -> bool:
    if not EMAIL_ENABLED:
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = ", ".join(EMAIL_TO)
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


def email_pretty(now_local_str: str, user_tz: str, session_name: str, email_no: int, email_max: int,
                 setups: List[Setup], dir_up: List[Tuple], dir_dn: List[Tuple]) -> Tuple[str, str]:
    HDR = "━━━━━━━━━━━━━━━━━━━━"
    SEP = "────────────────────"

    lines = []
    lines.append(HDR)
    lines.append(f"PulseFutures • {session_name} Session • {now_local_str} ({user_tz})")
    lines.append(f"Email {email_no}/{email_max} • Cooldown {SYMBOL_COOLDOWN_HOURS}h • Gap {EMAIL_GAP_MIN}m")
    lines.append(HDR)
    lines.append("")
    lines.append("🔥 TOP TRADE SETUPS")
    lines.append(SEP)
    lines.append("")

    for i, s in enumerate(setups, 1):
        lines.append(f"{i}) {s.setup_id} | {s.side} {s.symbol} | Conf {s.conf}/100")
        lines.append(f"Entry {fmt_price(s.entry)} | SL {fmt_price(s.sl)}")
        if s.tp1 and s.tp2 and s.conf >= MULTI_TP_MIN_CONF:
            lines.append(f"TP1 {fmt_price(s.tp1)} ({TP_ALLOCS[0]}%) | TP2 {fmt_price(s.tp2)} ({TP_ALLOCS[1]}%) | TP3 {fmt_price(s.tp)} ({TP_ALLOCS[2]}%)")
            lines.append("Rule: after TP1 → move SL to BE")
        else:
            lines.append(f"TP {fmt_price(s.tp)}")
        lines.append(f"24H {pct_emoji(s.ch24)} | 4H {pct_emoji(s.ch4)} | 1H {pct_emoji(s.ch1)} | F~{fmt_money(s.fut_vol_usd)}")
        lines.append(f"Chart: {tv_url(s.symbol)}")
        lines.append("")
        lines.append(SEP)

    lines.append("")
    lines.append("📈 DIRECTIONAL LEADERS (24H ≥ +10%, F vol ≥ 5M)")
    lines.append(SEP)
    if dir_up:
        for b, v, c24, c4, px in dir_up[:5]:
            lines.append(f"{b:<6} {pct_emoji(c24)} | 4H {pct_emoji(c4)} | F~{fmt_money(v)} | Last {fmt_price(px)}")
    else:
        lines.append("None")

    lines.append("")
    lines.append("📉 DIRECTIONAL LOSERS (24H ≤ -10%, F vol ≥ 5M)")
    lines.append(SEP)
    if dir_dn:
        for b, v, c24, c4, px in dir_dn[:5]:
            lines.append(f"{b:<6} {pct_emoji(c24)} | 4H {pct_emoji(c4)} | F~{fmt_money(v)} | Last {fmt_price(px)}")
    else:
        lines.append("None")

    lines.append("")
    lines.append(HDR)
    lines.append("Not financial advice.")
    lines.append("PulseFutures")
    lines.append(HDR)

    subject = f"PulseFutures • {session_name} • Setups ({now_local_str})"
    body = "\n".join(lines)
    return subject, body


# =========================
# HELP TEXT (PLAIN)
# =========================
HELP_TEXT = r"""
PulseFutures — Commands

1) Market Scan
- /screen
  Fast scan: Leaders + Directional Leaders/Losers + Top Setups
  Tip: Bot replies instantly with "Scanning..." and updates the same message.

2) Risk & Position Sizing (NO trade is auto-opened)
- /size <SYMBOL> <RISK> <SL>
  RISK can be:
    - USD number: 40
    - Percent: 2%
  SL is a PRICE (not percent).

  Examples:
   /size BTC 40 42150
     -> uses live price as Entry, risk $40, SL=42150, returns Qty
   /size ETH 2% 2190
     -> uses your Equity (must be set), risk=2% of equity, returns Qty

- /riskmode pct 2.5    (default per-trade sizing mode suggestion)
- /riskmode usd 40
- /dailycap pct 5      (max daily risk)
- /dailycap usd 150
- /limits 5            (max trades/day)

3) Equity & Journal (Professional Trading Journal)
- /equity <amount>
  Sets your equity (one-time or whenever you want). Equity will then AUTO-UPDATE
  when you close trades with PnL.

- /equity_reset <amount>
  Resets equity intentionally.

Open/Close trades:
- /trade_open <SYMBOL> <SIDE> <ENTRY> <SL> <RISK> [note]
  SIDE: buy or sell
  RISK: 40  OR  2%
  Example:
   /trade_open BTC buy 42800 42150 40 scalp-A
   /trade_open ETH sell 2350 2386 1.5% breakdown

- /trade_close <SYMBOL> <PNL>
  PNL: +25.5 or -18
  Example:
   /trade_close BTC +32.0

Status:
- /status
  Shows:
   - Equity
   - Daily cap + remaining
   - Daily trades count (limit)
   - Daily risk used
   - Open trades list (journal)
   - Session status (NY/London/Asia in your local TZ)

4) Sessions & Email Alerts (Global)
Sessions are defined in UTC and converted to your local timezone:
- NY     : 13:00–17:00 UTC
- LONDON : 07:00–10:00 UTC
- ASIA   : 00:00–03:00 UTC

Defaults:
- NY + London enabled for every user
- 3–4 emails per session, minimum 60 minutes gap
- No repeated symbol for 18 hours

Commands:
- /sessions
  Shows your local session times and which session is active now.

- /sessions_on NY
- /sessions_on LONDON
- /sessions_on ASIA
- /sessions_off ASIA
  You can enable extra sessions if you want more coverage.

- /email_on
- /email_off

5) Reports (Daily/Weekly)
- /report_day
- /report_week
  Shows:
   - total trades, wins/losses, winrate
   - net PnL
   - best advice based on your history

6) Signal Reports (What was emailed to you)
- /signal_report_day
- /signal_report_week
  Lists SetupIDs and summary of emailed signals.
  (SetupID helps you reference a signal in your PulseFutures channel.)

Notes:
- This bot is a risk manager + journal + signal alert tool.
- Not financial advice.
""".strip()


# =========================
# TELEGRAM COMMANDS
# =========================
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)


async def tz_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args:
        u = get_user(uid)
        await update.message.reply_text(f"Your TZ: {u.get('tz')}\nSet: /tz Australia/Melbourne")
        return
    tz = " ".join(context.args).strip()
    try:
        ZoneInfo(tz)
    except Exception:
        await update.message.reply_text("Invalid timezone. Example: /tz Australia/Melbourne or /tz Europe/London")
        return
    update_user(uid, tz=tz)
    await update.message.reply_text(f"✅ TZ set to: {tz}")


async def equity_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = get_user(uid)
    if not context.args:
        await update.message.reply_text(f"Equity: ${float(u.get('equity') or 0):.2f}\nSet: /equity 1000")
        return
    try:
        val = float(context.args[0])
        if val < 0:
            raise ValueError()
        update_user(uid, equity=val)
        await update.message.reply_text(f"✅ Equity set: ${val:.2f}")
    except Exception:
        await update.message.reply_text("Usage: /equity 1000")


async def equity_reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Usage: /equity_reset 1000")
        return
    try:
        val = float(context.args[0])
        if val < 0:
            raise ValueError()
        update_user(uid, equity=val, daily_risk_used=0.0, day_trade_count=0)
        await update.message.reply_text(f"✅ Equity RESET: ${val:.2f}")
    except Exception:
        await update.message.reply_text("Usage: /equity_reset 1000")


async def riskmode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = get_user(uid)
    if not context.args:
        await update.message.reply_text(f"Risk mode: {u['risk_mode']} {float(u['risk_value']):.2f}\nSet: /riskmode pct 2.5  OR  /riskmode usd 40")
        return
    if len(context.args) != 2:
        await update.message.reply_text("Usage: /riskmode pct 2.5  OR  /riskmode usd 40")
        return
    mode = context.args[0].strip().upper()
    try:
        val = float(context.args[1])
    except Exception:
        await update.message.reply_text("Invalid value. Example: /riskmode pct 2.5")
        return
    if mode not in {"PCT", "USD"}:
        await update.message.reply_text("Mode must be pct or usd.")
        return
    if val <= 0:
        await update.message.reply_text("Value must be > 0")
        return
    update_user(uid, risk_mode=mode, risk_value=val)
    await update.message.reply_text(f"✅ Risk mode set: {mode} {val:.2f}")


async def dailycap_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = get_user(uid)
    if not context.args:
        cap = daily_cap_usd(u)
        await update.message.reply_text(f"Daily cap: {u['daily_cap_mode']} {float(u['daily_cap_value']):.2f} (≈ ${cap:.2f})\nSet: /dailycap pct 5  OR  /dailycap usd 150")
        return
    if len(context.args) != 2:
        await update.message.reply_text("Usage: /dailycap pct 5  OR  /dailycap usd 150")
        return
    mode = context.args[0].strip().upper()
    try:
        val = float(context.args[1])
    except Exception:
        await update.message.reply_text("Invalid value.")
        return
    if mode not in {"PCT", "USD"}:
        await update.message.reply_text("Mode must be pct or usd.")
        return
    if val < 0:
        await update.message.reply_text("Value must be >= 0")
        return
    update_user(uid, daily_cap_mode=mode, daily_cap_value=val)
    u = get_user(uid)
    cap = daily_cap_usd(u)
    await update.message.reply_text(f"✅ Daily cap set: {mode} {val:.2f} (≈ ${cap:.2f})")


async def limits_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = get_user(uid)
    if not context.args:
        await update.message.reply_text(f"Max trades/day: {int(u.get('max_trades_day') or DEFAULT_MAX_TRADES_DAY)}\nSet: /limits 5")
        return
    try:
        n = int(context.args[0])
        if n < 1 or n > 50:
            raise ValueError()
        update_user(uid, max_trades_day=n)
        await update.message.reply_text(f"✅ Max trades/day set to {n}")
    except Exception:
        await update.message.reply_text("Usage: /limits 5  (1..50)")


async def email_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not EMAIL_CONFIG_OK:
        await update.message.reply_text("❌ Email is not configured on server (missing EMAIL_* env).")
        return
    update_user(uid, email_on=1)
    await update.message.reply_text("✅ Email alerts ON")


async def email_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    update_user(uid, email_on=0)
    await update.message.reply_text("✅ Email alerts OFF")


async def sessions_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = get_user(uid)
    await update.message.reply_text(sessions_status_text(u))


async def sessions_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = get_user(uid)
    if not context.args:
        await update.message.reply_text("Usage: /sessions_on NY  OR /sessions_on LONDON  OR /sessions_on ASIA")
        return
    name = context.args[0].strip().upper()
    if name not in SESSION_DEFS_UTC:
        await update.message.reply_text("Session must be NY, LONDON, or ASIA")
        return
    cur = enabled_sessions(u)
    if name not in cur:
        cur.append(name)
    update_user(uid, enabled_sessions=json.dumps(cur))
    await update.message.reply_text(f"✅ Enabled sessions: {', '.join(cur)}")


async def sessions_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = get_user(uid)
    if not context.args:
        await update.message.reply_text("Usage: /sessions_off ASIA (or NY/LONDON)")
        return
    name = context.args[0].strip().upper()
    cur = enabled_sessions(u)
    cur = [x for x in cur if x != name]
    if not cur:
        cur = DEFAULT_ENABLED_SESSIONS[:]  # never allow empty; keep core
    update_user(uid, enabled_sessions=json.dumps(cur))
    await update.message.reply_text(f"✅ Enabled sessions: {', '.join(cur)}")


async def size_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = get_user(uid)

    if len(context.args) < 3:
        await update.message.reply_text("Usage: /size BTC 40 42150   OR   /size ETH 2% 2190")
        return

    sym = re.sub(r"[^A-Za-z0-9]", "", context.args[0]).upper().lstrip("$")
    risk_str = context.args[1].strip()
    sl_str = context.args[2].strip()

    try:
        sl = float(sl_str)
        if sl <= 0:
            raise ValueError()
    except Exception:
        await update.message.reply_text("SL must be a PRICE number. Example: /size BTC 40 42150")
        return

    equity = float(u.get("equity") or 0.0)
    try:
        risk_usd, risk_label = parse_risk_value(risk_str, equity)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")
        return

    # live price
    best = await asyncio.to_thread(fetch_futures_tickers)
    mv = best.get(sym)
    if not mv or float(mv.last or 0.0) <= 0:
        await update.message.reply_text(f"Could not find {sym} on Bybit futures.")
        return
    entry = float(mv.last)

    qty = qty_from_risk(entry, sl, risk_usd)
    if qty <= 0:
        await update.message.reply_text("Could not compute qty (check SL vs Entry).")
        return

    msg = (
        f"📏 *Position Size*\n"
        f"Symbol: *{sym}*\n"
        f"Entry (live): {fmt_price(entry)}\n"
        f"SL: {fmt_price(sl)}\n"
        f"Risk: {risk_label} (≈ ${risk_usd:.2f})\n"
        f"✅ Qty: *{qty:.6g}*\n"
        f"📈 {tv_url(sym)}"
    )
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("📈 Chart (TV)", url=tv_url(sym))]])
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=kb, disable_web_page_preview=True)


async def trade_open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = reset_daily_if_needed(get_user(uid))

    if len(context.args) < 5:
        await update.message.reply_text("Usage: /trade_open BTC buy 42800 42150 40 [note]\nOr: /trade_open ETH sell 2350 2386 1.5% breakdown")
        return

    sym = re.sub(r"[^A-Za-z0-9]", "", context.args[0]).upper().lstrip("$")
    side_raw = context.args[1].strip().upper()
    side = "BUY" if side_raw in {"BUY", "LONG"} else "SELL" if side_raw in {"SELL", "SHORT"} else ""
    if not side:
        await update.message.reply_text("SIDE must be buy/sell (long/short).")
        return

    try:
        entry = float(context.args[2])
        sl = float(context.args[3])
        if entry <= 0 or sl <= 0 or entry == sl:
            raise ValueError()
    except Exception:
        await update.message.reply_text("Bad entry/sl. Example: /trade_open BTC buy 42800 42150 40")
        return

    risk_str = context.args[4].strip()
    notes = " ".join(context.args[5:]).strip() if len(context.args) > 5 else ""

    equity = float(u.get("equity") or 0.0)
    try:
        risk_usd, risk_label = parse_risk_value(risk_str, equity)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")
        return

    # limits check
    max_trades = int(u.get("max_trades_day") or DEFAULT_MAX_TRADES_DAY)
    if int(u.get("day_trade_count") or 0) >= max_trades:
        await update.message.reply_text(f"⚠️ You already reached max trades/day ({max_trades}).")
        return

    cap = daily_cap_usd(u)
    used = float(u.get("daily_risk_used") or 0.0)
    remaining = cap - used
    if cap > 0 and risk_usd > remaining:
        await update.message.reply_text(f"⚠️ Daily risk cap exceeded.\nCap ≈ ${cap:.2f}\nUsed ${used:.2f}\nRemaining ${remaining:.2f}")
        return

    trade_open_db(uid, sym, side, entry, sl, risk_usd, notes)
    update_user(uid, day_trade_count=int(u.get("day_trade_count") or 0) + 1, daily_risk_used=used + risk_usd)

    msg = (
        f"✅ *Trade OPENED (Journal)*\n"
        f"{sym} {side}\n"
        f"Entry {fmt_price(entry)} | SL {fmt_price(sl)}\n"
        f"Risk {risk_label} (≈ ${risk_usd:.2f})\n"
        f"Notes: {notes or '-'}"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def trade_close_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = get_user(uid)

    if len(context.args) < 2:
        await update.message.reply_text("Usage: /trade_close BTC +32.5")
        return

    sym = re.sub(r"[^A-Za-z0-9]", "", context.args[0]).upper().lstrip("$")
    try:
        pnl = float(context.args[1])
    except Exception:
        await update.message.reply_text("PNL must be a number like +32.5 or -18")
        return

    tr = trade_close_db(uid, sym, pnl)
    if not tr:
        await update.message.reply_text(f"No OPEN trade found for {sym}.")
        return

    equity = float(u.get("equity") or 0.0)
    new_eq = equity + pnl
    update_user(uid, equity=new_eq)

    msg = (
        f"✅ *Trade CLOSED*\n"
        f"{sym} {tr['side']}\n"
        f"PnL: {pnl:+.2f}\n"
        f"Equity: ${equity:.2f} → ${new_eq:.2f}"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = reset_daily_if_needed(get_user(uid))

    equity = float(u.get("equity") or 0.0)
    cap = daily_cap_usd(u)
    used = float(u.get("daily_risk_used") or 0.0)
    remaining = cap - used if cap > 0 else 0.0

    max_trades = int(u.get("max_trades_day") or DEFAULT_MAX_TRADES_DAY)
    day_trades = int(u.get("day_trade_count") or 0)

    opens = get_open_trades(uid)

    lines = []
    lines.append("📌 *Status*")
    lines.append(f"Equity: ${equity:.2f}")
    lines.append(f"Daily cap: {u['daily_cap_mode']} {float(u['daily_cap_value']):.2f} (≈ ${cap:.2f})")
    lines.append(f"Daily risk used: ${used:.2f} | Remaining: ${remaining:.2f}")
    lines.append(f"Trades today: {day_trades}/{max_trades}")
    if day_trades >= max_trades:
        lines.append("⚠️ You are above/at your daily trade limit.")

    lines.append("")
    lines.append("🕒 *Sessions*")
    lines.append(sessions_status_text(u))

    lines.append("")
    lines.append("📓 *Open Trades*")
    if not opens:
        lines.append("None")
    else:
        for i, t in enumerate(opens, 1):
            lines.append(f"#{i} {t['symbol']} {t['side']} | Entry {fmt_price(float(t['entry']))} | SL {fmt_price(float(t['sl']))} | Risk ${float(t['risk_usd']):.2f} | Note {t.get('notes') or '-'}")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


def advice_from_trades(trades: List[dict]) -> str:
    # Simple, practical advice engine
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("pnl") is not None]
    if len(closed) < 5:
        return "Advice: Keep journaling. After 10+ trades, advice becomes more meaningful."

    wins = [t for t in closed if float(t["pnl"]) > 0]
    losses = [t for t in closed if float(t["pnl"]) < 0]
    winrate = (len(wins) / len(closed)) * 100 if closed else 0.0
    net = sum(float(t["pnl"]) for t in closed)

    adv = []
    if winrate < 50:
        adv.append("Winrate < 50% → reduce risk per trade (e.g., -30%) and trade only highest-quality setups.")
    if net < 0:
        adv.append("Net negative → add a strict rule: no trade without clear invalidation (SL) and pre-defined TP.")
    if len(losses) >= 3 and all(float(x["pnl"]) < 0 for x in losses[-3:]):
        adv.append("3 losses in a row → stop trading for the day (cooldown).")
    if not adv:
        adv.append("Good stability → keep risk consistent and avoid overtrading.")

    return "Advice:\n- " + "\n- ".join(adv)


async def report_day_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = get_user(uid)
    tz = ZoneInfo(u.get("tz") or "UTC")

    now = datetime.now(tz)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    end = now.replace(hour=23, minute=59, second=59, microsecond=0).timestamp()

    trades = get_trades_between(uid, start, end)
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("pnl") is not None]
    wins = [t for t in closed if float(t["pnl"]) > 0]
    losses = [t for t in closed if float(t["pnl"]) < 0]
    winrate = (len(wins) / len(closed)) * 100 if closed else 0.0
    net = sum(float(t["pnl"]) for t in closed)

    msg = (
        f"📊 *Daily Report* ({now.strftime('%Y-%m-%d')})\n"
        f"Trades opened: {len(trades)}\n"
        f"Closed: {len(closed)} | Wins: {len(wins)} | Losses: {len(losses)} | Winrate: {winrate:.1f}%\n"
        f"Net PnL: {net:+.2f}\n\n"
        f"{advice_from_trades(trades)}"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def report_week_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = get_user(uid)
    tz = ZoneInfo(u.get("tz") or "UTC")

    now = datetime.now(tz)
    start_dt = now - timedelta(days=7)
    start = start_dt.timestamp()
    end = now.timestamp()

    trades = get_trades_between(uid, start, end)
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("pnl") is not None]
    wins = [t for t in closed if float(t["pnl"]) > 0]
    losses = [t for t in closed if float(t["pnl"]) < 0]
    winrate = (len(wins) / len(closed)) * 100 if closed else 0.0
    net = sum(float(t["pnl"]) for t in closed)

    msg = (
        f"📈 *Weekly Report* (last 7 days)\n"
        f"Trades opened: {len(trades)}\n"
        f"Closed: {len(closed)} | Wins: {len(wins)} | Losses: {len(losses)} | Winrate: {winrate:.1f}%\n"
        f"Net PnL: {net:+.2f}\n\n"
        f"{advice_from_trades(trades)}"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def signal_report_day_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = get_user(uid)
    tz = ZoneInfo(u.get("tz") or "UTC")

    now = datetime.now(tz)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc).timestamp()
    end = now.replace(hour=23, minute=59, second=59, microsecond=0).astimezone(timezone.utc).timestamp()

    rows = get_user_setups_between(uid, start, end)
    if not rows:
        await update.message.reply_text("No emailed setups today.")
        return

    lines = [f"📨 *Signal Report — Today* ({now.strftime('%Y-%m-%d')})", f"Count: {len(rows)}", ""]
    for r in rows[:30]:
        t_local = datetime.fromtimestamp(float(r["sent_ts"]), tz=timezone.utc).astimezone(tz).strftime("%H:%M")
        lines.append(f"- {t_local} | {r['session_name']} | {r['setup_id']} | {r['side']} {r['symbol']} | Conf {r['conf']}")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def signal_report_week_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = get_user(uid)
    tz = ZoneInfo(u.get("tz") or "UTC")

    now = datetime.now(tz)
    start_dt = (now - timedelta(days=7)).astimezone(timezone.utc)
    start = start_dt.timestamp()
    end = datetime.now(timezone.utc).timestamp()

    rows = get_user_setups_between(uid, start, end)
    if not rows:
        await update.message.reply_text("No emailed setups in last 7 days.")
        return

    by_session = {}
    for r in rows:
        by_session.setdefault(r["session_name"], 0)
        by_session[r["session_name"]] += 1

    lines = [f"📨 *Signal Report — Week* (last 7 days)", f"Total setups: {len(rows)}", ""]
    lines.append("By session:")
    for k in sorted(by_session.keys()):
        lines.append(f"- {k}: {by_session[k]}")
    lines.append("")
    lines.append("Recent SetupIDs:")
    for r in rows[-20:]:
        t_local = datetime.fromtimestamp(float(r["sent_ts"]), tz=timezone.utc).astimezone(tz).strftime("%m-%d %H:%M")
        lines.append(f"- {t_local} | {r['setup_id']} | {r['side']} {r['symbol']} | Conf {r['conf']}")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


# =========================
# /screen (FAST + EDIT MESSAGE)
# =========================
async def screen_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # prevent overlap if user spams /screen
    if SCREEN_LOCK.locked():
        await update.message.reply_text("⏳ Scan already running…")
        return

    async with SCREEN_LOCK:
        chat_id = update.effective_chat.id
        msg = await update.message.reply_text("⏳ Scanning market… (this message will update)")
        msg_id = msg.message_id

        try:
            # heavy work in thread
            best = await asyncio.to_thread(fetch_futures_tickers)
            if not best:
                await context.bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text="Error: could not fetch futures tickers.")
                return

            now_utc = datetime.now(timezone.utc)
            leaders = sorted(best.items(), key=lambda kv: usd_notional(kv[1]), reverse=True)[:LEADERS_N]
            leaders_rows = [[b, fmt_money(usd_notional(mv)), pct_emoji(float(mv.percentage or 0.0)), fmt_price(float(mv.last or 0.0))] for b, mv in leaders]

            dir_up, dir_dn = await asyncio.to_thread(directional_lists, best)
            setups = await asyncio.to_thread(pick_setups, now_utc, best)

            parts = []
            parts.append("✨ *PulseFutures — Market Scan*")
            parts.append("")
            parts.append("🔥 *Top Setups*")
            if not setups:
                parts.append("_No high-quality setup right now._")
            else:
                for i, s in enumerate(setups, 1):
                    parts.append("")
                    parts.append(setup_block_md(s, i))

            parts.append("")
            parts.append("📈 *Directional Leaders*")
            if dir_up:
                rows = [[b, fmt_money(v), pct_emoji(c24), pct_emoji(c4), fmt_price(px)] for b, v, c24, c4, px in dir_up]
                parts.append(table_md(rows, ["SYM", "F Vol", "24H", "4H", "Last"]))
            else:
                parts.append("_None_")

            parts.append("")
            parts.append("📉 *Directional Losers*")
            if dir_dn:
                rows = [[b, fmt_money(v), pct_emoji(c24), pct_emoji(c4), fmt_price(px)] for b, v, c24, c4, px in dir_dn]
                parts.append(table_md(rows, ["SYM", "F Vol", "24H", "4H", "Last"]))
            else:
                parts.append("_None_")

            parts.append("")
            parts.append("🏆 *Market Leaders (Volume)*")
            parts.append(table_md(leaders_rows, ["SYM", "F Vol", "24H", "Last"]))

            text = "\n".join(parts).strip()

            kb = []
            for s in setups:
                kb.append([InlineKeyboardButton(text=f"📈 {s.symbol} Chart", url=tv_url(s.symbol))])

            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup(kb) if kb else None,
                disable_web_page_preview=True,
            )

        except BadRequest:
            # fallback: remove markdown
            await context.bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text="Scan done (formatting removed due to Telegram markdown limits).")


# =========================
# EMAIL ALERT JOB
# =========================
async def alert_job(context: ContextTypes.DEFAULT_TYPE):
    if ALERT_LOCK.locked():
        return
    async with ALERT_LOCK:
        try:
            if not EMAIL_ENABLED:
                return

            users = list_users_email_on()
            if not users:
                return

            now_utc = datetime.now(timezone.utc)
            best = await asyncio.to_thread(fetch_futures_tickers)
            if not best:
                return

            dir_up, dir_dn = await asyncio.to_thread(directional_lists, best)
            setups_base = await asyncio.to_thread(pick_setups, now_utc, best)
            if not setups_base:
                return

            for user in users:
                uid = int(user["user_id"])
                user = reset_daily_if_needed(user)

                if not user.get("email_on"):
                    continue

                tz = ZoneInfo(user.get("tz") or "UTC")
                en_sessions = enabled_sessions(user)

                for session_name in en_sessions:
                    if not is_in_session(user, session_name, now_utc):
                        continue

                    # state per session
                    st = email_state_get(uid, session_name)
                    sess_key = session_key_for_day(session_name, now_utc)
                    if st.get("session_key") != sess_key:
                        email_state_set(uid, session_name, session_key=sess_key, sent_count=0, last_email_ts=0.0)
                        st = email_state_get(uid, session_name)

                    max_em = int(user.get("emails_per_session") or DEFAULT_EMAILS_PER_SESSION)
                    gap_min = int(user.get("email_gap_min") or EMAIL_GAP_MIN)
                    gap_sec = max(0, gap_min) * 60

                    if int(st.get("sent_count") or 0) >= max_em:
                        continue
                    if gap_sec > 0 and (time.time() - float(st.get("last_email_ts") or 0.0)) < gap_sec:
                        continue

                    # apply cooldown: no repeated symbol within 18 hours
                    filtered: List[Setup] = []
                    now_ts = time.time()
                    for s in setups_base:
                        if cooldown_ok(uid, s.symbol, now_ts):
                            filtered.append(s)

                    if not filtered:
                        continue

                    # we may send up to SETUPS_N setups; assign unique SetupIDs and store
                    setups_to_send = []
                    for s in filtered[:SETUPS_N]:
                        # ensure unique id per setup sent
                        s2 = Setup(**{**s.__dict__})
                        # Setup ID already created in make_setup; but to guarantee uniqueness across runs, regenerate here
                        s2.setup_id = next_setup_id(now_utc)
                        s2.created_ts = now_ts
                        store_setup_sent(s2)
                        store_user_setup(uid, s2.setup_id, now_ts, session_name)
                        cooldown_mark(uid, s2.symbol, now_ts)
                        setups_to_send.append(s2)

                    if not setups_to_send:
                        continue

                    now_local = now_utc.astimezone(tz).strftime("%Y-%m-%d %H:%M")
                    subject, body = email_pretty(
                        now_local_str=now_local,
                        user_tz=user.get("tz") or "UTC",
                        session_name=session_name,
                        email_no=int(st.get("sent_count") or 0) + 1,
                        email_max=max_em,
                        setups=setups_to_send,
                        dir_up=dir_up,
                        dir_dn=dir_dn,
                    )

                    ok = await asyncio.to_thread(send_email, subject, body)
                    if ok:
                        email_state_set(
                            uid,
                            session_name,
                            sent_count=int(st.get("sent_count") or 0) + 1,
                            last_email_ts=time.time(),
                        )

        except Exception as e:
            logger.exception("alert_job error: %s", e)


# =========================
# TEXT ROUTER (optional)
# =========================
async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Keep it minimal to avoid accidental parsing delays
    return


# =========================
# GLOBAL ERROR HANDLER
# =========================
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled exception", exc_info=context.error)


# =========================
# MAIN (WEBHOOK preferred for Render)
# =========================
def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN missing")

    db_init()

    app = Application.builder().token(TOKEN).build()
    app.add_error_handler(on_error)

    # Commands
    app.add_handler(CommandHandler(["help", "start"], cmd_help))
    app.add_handler(CommandHandler("tz", tz_cmd))

    app.add_handler(CommandHandler("screen", screen_cmd))

    app.add_handler(CommandHandler("equity", equity_cmd))
    app.add_handler(CommandHandler("equity_reset", equity_reset_cmd))
    app.add_handler(CommandHandler("riskmode", riskmode_cmd))
    app.add_handler(CommandHandler("dailycap", dailycap_cmd))
    app.add_handler(CommandHandler("limits", limits_cmd))

    app.add_handler(CommandHandler("size", size_cmd))

    app.add_handler(CommandHandler("trade_open", trade_open_cmd))
    app.add_handler(CommandHandler("trade_close", trade_close_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    app.add_handler(CommandHandler("report_day", report_day_cmd))
    app.add_handler(CommandHandler("report_week", report_week_cmd))

    app.add_handler(CommandHandler("signal_report_day", signal_report_day_cmd))
    app.add_handler(CommandHandler("signal_report_week", signal_report_week_cmd))

    app.add_handler(CommandHandler("sessions", sessions_cmd))
    app.add_handler(CommandHandler("sessions_on", sessions_on_cmd))
    app.add_handler(CommandHandler("sessions_off", sessions_off_cmd))

    app.add_handler(CommandHandler("email_on", email_on_cmd))
    app.add_handler(CommandHandler("email_off", email_off_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    if getattr(app, "job_queue", None):
        app.job_queue.run_repeating(alert_job, interval=CHECK_INTERVAL_MIN * 60, first=10, name="alert_job")
    else:
        logger.warning('JobQueue not available. Install "python-telegram-bot[job-queue]>=20.7,<22.0"')

    webhook_url = os.environ.get("WEBHOOK_URL", "").strip()
    port = int(os.environ.get("PORT", "10000"))
    listen = "0.0.0.0"

    if webhook_url:
        # Webhook mode (recommended on Render)
        if not webhook_url.startswith("https://"):
            raise RuntimeError("WEBHOOK_URL must start with https://")
        path = f"/telegram/{TOKEN[:12]}"
        full_url = webhook_url.rstrip("/") + path
        logger.info("Starting WEBHOOK mode: %s", full_url)
        app.run_webhook(
            listen=listen,
            port=port,
            url_path=path.lstrip("/"),
            webhook_url=full_url,
            drop_pending_updates=True,
        )
    else:
        # Polling mode (ONLY safe if one instance is running)
        logger.info("Starting POLLING mode")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
