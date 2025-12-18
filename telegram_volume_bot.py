#!/usr/bin/env python3
"""
PulseFutures — Bybit Futures (Swap) Screener + TradeSetup + Risk Ledger

✅ Latest fixes in this version:
- /help and /start no longer use Telegram MARKDOWN (fixes: Can't parse entities)
- Added global Telegram error handler (so bot won't die on unexpected errors)
- Session-based email cap + min gap
- Symbol cooldown PER SESSION (no repeat until next session)
- Movers/Losers min volume = $5M
- Multi-TP fixed (TP1/TP2/TP3 won't collapse)
- /screen faster (2 OHLCV calls per symbol: 1h + 15m)
- Optional charts: quickchart png + TradingView link

ENV:
- TELEGRAM_TOKEN (required)
Email ENV (if you want emails):
- EMAIL_ENABLED=true/false
- EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FROM, EMAIL_TO
Optional:
- CHECK_INTERVAL_MIN=5
- CHART_IMG_ENABLED=true/false

Not financial advice.
"""

import logging
import os
import re
import ssl
import smtplib
import sqlite3
import time
import json
import urllib.request
import urllib.parse
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from io import BytesIO

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
TOKEN = os.environ.get("TELEGRAM_TOKEN")
DEFAULT_TYPE = "swap"

LEADERS_N = 10
MOVERS_N = 10
SETUPS_N = 3

# Movers thresholds
MOVER_VOL_USD_MIN = 5_000_000
MOVER_UP_24H_MIN = 10.0
MOVER_DN_24H_MAX = -10.0

# 4H alignment for Directional Leaders/Losers
MOVER_4H_TREND_FILTER = os.environ.get("MOVER_4H_TREND_FILTER", "true").lower() == "true"
MOVER_4H_ALIGN_MIN = float(os.environ.get("MOVER_4H_ALIGN_MIN", "1.0"))

# Setups thresholds
TRIGGER_1H_ABS_MIN = 2.0
CONFIRM_15M_ABS_MIN = 0.6
ALIGN_4H_MIN = 0.0

# Risk defaults
DEFAULT_EQUITY = 1000.0
DEFAULT_MAX_TRADES_DAY = 3
DEFAULT_OPEN_RISK_CAP_USD = 75.0

# Risk controls
DEFAULT_RISK_MODE = "PCT"       # PCT or USD
DEFAULT_RISK_VALUE = 1.5        # if PCT -> % per trade, if USD -> $ value
DEFAULT_DAILY_CAP_MODE = "PCT"  # PCT or USD
DEFAULT_DAILY_CAP_VALUE = 4.5   # if PCT -> % per day, if USD -> $ value
DEFAULT_RISK_AUTO = 0           # OFF by default

# Time-stop
SETUP_TIME_STOP_HOURS = int(os.environ.get("SETUP_TIME_STOP_HOURS", "18"))

# ATR-based SL/TP
USE_ATR_SLTP = os.environ.get("USE_ATR_SLTP", "true").lower() == "true"
ATR_PERIOD = int(os.environ.get("ATR_PERIOD", "14"))
ATR_MIN_PCT = float(os.environ.get("ATR_MIN_PCT", "0.6"))
ATR_MAX_PCT = float(os.environ.get("ATR_MAX_PCT", "6.0"))
TP_MAX_PCT = float(os.environ.get("TP_MAX_PCT", "4.5"))

MULTI_TP_MIN_CONF = int(os.environ.get("MULTI_TP_MIN_CONF", "75"))
TP_ALLOCS = (40, 40, 20)

try:
    _rm = os.environ.get("TP_R_MULTS", "0.8,1.4,2.0")
    TP_R_MULTS = tuple(float(x.strip()) for x in _rm.split(","))
    if len(TP_R_MULTS) != 3:
        TP_R_MULTS = (0.8, 1.4, 2.0)
except Exception:
    TP_R_MULTS = (0.8, 1.4, 2.0)

# Email
EMAIL_ENABLED_DEFAULT = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST = os.environ.get("EMAIL_HOST", "")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "465"))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", EMAIL_USER)
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_USER)

# Email content
INCLUDE_MOVERS_IN_EMAIL = True
EMAIL_MOVERS_TOP_N = 5

# Per-user email session defaults
DEFAULT_SESSIONS = [{"start": "18:00", "end": "23:30"}]  # user local time
DEFAULT_MAX_EMAILS_PER_SESSION = int(os.environ.get("DEFAULT_MAX_EMAILS_PER_SESSION", "3"))
DEFAULT_EMAIL_GAP_MIN = int(os.environ.get("DEFAULT_EMAIL_GAP_MIN", "60"))  # minutes

# Symbol cooldown PER SESSION (no repeats inside same session)
SYMBOL_COOLDOWN_PER_SESSION = True

# Market bias
BIAS_UNIVERSE_N = int(os.environ.get("BIAS_UNIVERSE_N", "20"))
BIAS_STRONG_TH = float(os.environ.get("BIAS_STRONG_TH", "0.6"))

# Bot runtime
CHECK_INTERVAL_MIN = int(os.environ.get("CHECK_INTERVAL_MIN", "5"))

# DB
DB_PATH = os.environ.get("DB_PATH", "pulsefutures.db")

# Charts (QuickChart)
CHART_IMG_ENABLED = os.environ.get("CHART_IMG_ENABLED", "true").lower() == "true"
CHART_TIMEFRAME = os.environ.get("CHART_TIMEFRAME", "1h")
CHART_BARS = int(os.environ.get("CHART_BARS", "120"))
CHART_WIDTH = int(os.environ.get("CHART_WIDTH", "900"))
CHART_HEIGHT = int(os.environ.get("CHART_HEIGHT", "500"))

HDR = "━━━━━━━━━━━━━━━━━━━━"
SEP = "────────────────────"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pulsefutures")

STABLES = {"USDT", "USDC", "USD", "TUSD", "FDUSD", "DAI", "PYUSD"}


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
    symbol: str
    market_symbol: str
    side: str
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
    regime: str
    created_ts: float


# =========================
# DB HELPERS
# =========================
def db_connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def db_init():
    con = db_connect()
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            equity REAL NOT NULL,
            risk_pct REAL NOT NULL,
            max_trades_day INTEGER NOT NULL,
            daily_risk_cap REAL NOT NULL,
            open_risk_cap REAL NOT NULL,
            notify_on INTEGER NOT NULL,
            tz TEXT NOT NULL,
            day_trade_count INTEGER NOT NULL,
            day_trade_date TEXT NOT NULL,
            daily_risk_used REAL NOT NULL,
            risk_mode TEXT,
            risk_value REAL,
            daily_cap_mode TEXT,
            daily_cap_value REAL,
            risk_auto INTEGER,
            trade_sessions TEXT,
            max_emails_per_session INTEGER,
            email_gap_min INTEGER
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry REAL NOT NULL,
            sl REAL NOT NULL,
            tp REAL NOT NULL,
            risk_usd REAL NOT NULL,
            qty REAL NOT NULL,
            notional REAL NOT NULL,
            conf INTEGER,
            created_ts REAL NOT NULL,
            status TEXT NOT NULL,
            pnl REAL,
            closed_ts REAL,
            notes TEXT,
            time_stop_ts REAL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signals (
            symbol TEXT PRIMARY KEY,
            market_symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            conf INTEGER NOT NULL,
            entry REAL NOT NULL,
            sl REAL NOT NULL,
            tp REAL NOT NULL,
            tp1 REAL,
            tp2 REAL,
            fut_vol_usd REAL NOT NULL,
            ch24 REAL NOT NULL,
            ch4 REAL NOT NULL,
            ch1 REAL NOT NULL,
            ch15 REAL NOT NULL,
            regime TEXT NOT NULL,
            created_ts REAL NOT NULL
        )
        """
    )

    # Per-user email state (session cap + gap)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS email_state (
            user_id INTEGER PRIMARY KEY,
            session_key TEXT NOT NULL,
            sent_count INTEGER NOT NULL,
            last_email_ts REAL NOT NULL,
            last_symbols TEXT NOT NULL
        )
        """
    )

    # Symbol cooldown PER SESSION
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS emailed_symbols (
            user_id INTEGER NOT NULL,
            session_key TEXT NOT NULL,
            symbol TEXT NOT NULL,
            PRIMARY KEY (user_id, session_key, symbol)
        )
        """
    )

    con.commit()
    con.close()


def _normalize_user_row(d: dict) -> dict:
    if not d.get("risk_mode"):
        d["risk_mode"] = DEFAULT_RISK_MODE
    if d.get("risk_value") is None:
        d["risk_value"] = float(DEFAULT_RISK_VALUE)
    if not d.get("daily_cap_mode"):
        d["daily_cap_mode"] = DEFAULT_DAILY_CAP_MODE
    if d.get("daily_cap_value") is None:
        d["daily_cap_value"] = float(DEFAULT_DAILY_CAP_VALUE)
    if d.get("risk_auto") is None:
        d["risk_auto"] = int(DEFAULT_RISK_AUTO)

    if not d.get("trade_sessions"):
        d["trade_sessions"] = json.dumps(DEFAULT_SESSIONS)
    if d.get("max_emails_per_session") is None:
        d["max_emails_per_session"] = int(DEFAULT_MAX_EMAILS_PER_SESSION)
    if d.get("email_gap_min") is None:
        d["email_gap_min"] = int(DEFAULT_EMAIL_GAP_MIN)

    return d


def get_user(user_id: int) -> dict:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()

    if not row:
        now_mel = datetime.now(ZoneInfo("Australia/Melbourne"))
        cur.execute(
            """
            INSERT INTO users (
                user_id, equity, risk_pct, max_trades_day, daily_risk_cap, open_risk_cap,
                notify_on, tz, day_trade_count, day_trade_date, daily_risk_used,
                risk_mode, risk_value, daily_cap_mode, daily_cap_value, risk_auto,
                trade_sessions, max_emails_per_session, email_gap_min
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                DEFAULT_EQUITY,
                1.0,
                DEFAULT_MAX_TRADES_DAY,
                50.0,  # legacy
                DEFAULT_OPEN_RISK_CAP_USD,
                1 if EMAIL_ENABLED_DEFAULT else 0,
                "Australia/Melbourne",
                0,
                now_mel.date().isoformat(),
                0.0,
                DEFAULT_RISK_MODE,
                float(DEFAULT_RISK_VALUE),
                DEFAULT_DAILY_CAP_MODE,
                float(DEFAULT_DAILY_CAP_VALUE),
                int(DEFAULT_RISK_AUTO),
                json.dumps(DEFAULT_SESSIONS),
                int(DEFAULT_MAX_EMAILS_PER_SESSION),
                int(DEFAULT_EMAIL_GAP_MIN),
            ),
        )
        con.commit()
        cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        row = cur.fetchone()

    con.close()
    return _normalize_user_row(dict(row))


def list_users_notify_on() -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE notify_on=1")
    rows = cur.fetchall()
    con.close()
    return [_normalize_user_row(dict(r)) for r in rows]


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


def reset_daily_counters_if_needed(user: dict) -> dict:
    tz = ZoneInfo(user.get("tz") or "Australia/Melbourne")
    today = datetime.now(tz).date().isoformat()
    if user.get("day_trade_date") != today:
        update_user(user["user_id"], day_trade_count=0, day_trade_date=today, daily_risk_used=0.0)
        user = get_user(user["user_id"])
    return user


def db_upsert_signal(setup: Setup):
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO signals (
            symbol, market_symbol, side, conf, entry, sl, tp, tp1, tp2,
            fut_vol_usd, ch24, ch4, ch1, ch15, regime, created_ts
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            market_symbol=excluded.market_symbol,
            side=excluded.side,
            conf=excluded.conf,
            entry=excluded.entry,
            sl=excluded.sl,
            tp=excluded.tp,
            tp1=excluded.tp1,
            tp2=excluded.tp2,
            fut_vol_usd=excluded.fut_vol_usd,
            ch24=excluded.ch24,
            ch4=excluded.ch4,
            ch1=excluded.ch1,
            ch15=excluded.ch15,
            regime=excluded.regime,
            created_ts=excluded.created_ts
        """,
        (
            setup.symbol,
            setup.market_symbol,
            setup.side,
            setup.conf,
            setup.entry,
            setup.sl,
            setup.tp,
            setup.tp1,
            setup.tp2,
            setup.fut_vol_usd,
            setup.ch24,
            setup.ch4,
            setup.ch1,
            setup.ch15,
            setup.regime,
            setup.created_ts,
        ),
    )
    con.commit()
    con.close()


def db_get_signal(symbol: str, ttl_sec: int = 24 * 3600) -> Optional[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM signals WHERE symbol=?", (symbol.upper(),))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    s = dict(row)
    if time.time() - float(s["created_ts"]) > ttl_sec:
        con = db_connect()
        cur = con.cursor()
        cur.execute("DELETE FROM signals WHERE symbol=?", (symbol.upper(),))
        con.commit()
        con.close()
        return None
    return s


def db_add_position(
    user_id: int,
    symbol: str,
    side: str,
    entry: float,
    sl: float,
    tp: float,
    risk_usd: float,
    qty: float,
    notional: float,
    conf: Optional[int],
    time_stop_ts: Optional[float] = None,
    notes: str = "",
):
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO positions (
            user_id, symbol, side, entry, sl, tp, risk_usd, qty, notional, conf,
            created_ts, status, time_stop_ts, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
        """,
        (
            user_id,
            symbol,
            side,
            entry,
            sl,
            tp,
            risk_usd,
            qty,
            notional,
            conf,
            time.time(),
            time_stop_ts,
            notes,
        ),
    )
    con.commit()
    con.close()


def db_get_open_positions(user_id: int) -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT * FROM positions
        WHERE user_id=? AND status='OPEN'
        ORDER BY created_ts ASC
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


def db_close_position(user_id: int, symbol: str, pnl: float) -> Optional[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT * FROM positions
        WHERE user_id=? AND status='OPEN' AND symbol=?
        ORDER BY created_ts ASC
        LIMIT 1
        """,
        (user_id, symbol.upper()),
    )
    row = cur.fetchone()
    if not row:
        con.close()
        return None
    pos = dict(row)
    cur.execute(
        """
        UPDATE positions
        SET status='CLOSED', pnl=?, closed_ts=?
        WHERE id=?
        """,
        (pnl, time.time(), pos["id"]),
    )
    con.commit()
    con.close()
    pos["pnl"] = pnl
    return pos


# ---------- Email State / Symbol Cooldown (PER SESSION) ----------
def email_state_get(user_id: int) -> dict:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM email_state WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    if not row:
        cur.execute(
            "INSERT INTO email_state (user_id, session_key, sent_count, last_email_ts, last_symbols) VALUES (?, ?, ?, ?, ?)",
            (user_id, "NONE", 0, 0.0, ""),
        )
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


def has_emailed_symbol_in_session(user_id: int, session_key: str, symbol: str) -> bool:
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        "SELECT 1 FROM emailed_symbols WHERE user_id=? AND session_key=? AND symbol=?",
        (user_id, session_key, symbol.upper()),
    )
    row = cur.fetchone()
    con.close()
    return bool(row)


def mark_emailed_symbols_in_session(user_id: int, session_key: str, symbols: List[str]):
    if not symbols:
        return
    con = db_connect()
    cur = con.cursor()
    for sym in symbols:
        cur.execute(
            "INSERT OR IGNORE INTO emailed_symbols (user_id, session_key, symbol) VALUES (?, ?, ?)",
            (user_id, session_key, sym.upper()),
        )
    con.commit()
    con.close()


# =========================
# EXCHANGE HELPERS
# =========================
def build_exchange():
    klass = ccxt.__dict__[EXCHANGE_ID]
    return klass(
        {
            "enableRateLimit": True,
            "timeout": 20000,
            "options": {"defaultType": DEFAULT_TYPE},
        }
    )


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


def usd_notional(mv: Optional[MarketVol]) -> float:
    if not mv:
        return 0.0
    if mv.quote in STABLES and mv.quote_vol:
        return float(mv.quote_vol)
    price = mv.vwap if mv.vwap else mv.last
    if not price or not mv.base_vol:
        return 0.0
    return float(mv.base_vol) * float(price)


def fetch_futures_tickers() -> Dict[str, MarketVol]:
    ex = build_exchange()
    try:
        ex.load_markets()
        tickers = ex.fetch_tickers()
    except Exception as e:
        logger.exception("fetch_tickers failed: %s", e)
        return {}

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


def fetch_ohlcv(symbol: str, timeframe: str, limit: int):
    ex = build_exchange()
    ex.load_markets()
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit) or []


def compute_atr_from_ohlcv(candles: List[List[float]], period: int) -> float:
    if not candles or len(candles) < period + 1:
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


# =========================
# SPEED-UP METRICS (2 calls only)
# =========================
def metrics_from_candles_1h_15m(market_symbol: str) -> Tuple[float, float, float, float]:
    """
    Returns (ch1, ch4, ch15, atr) using only 2 fetches:
      - 1h candles -> ch1, ch4, atr
      - 15m candles -> ch15
    """
    need_1h = max(ATR_PERIOD + 5, 30)
    c1 = fetch_ohlcv(market_symbol, "1h", limit=need_1h)
    if not c1 or len(c1) < 6:
        return 0.0, 0.0, 0.0, 0.0

    closes_1h = [float(x[4]) for x in c1]
    c_last = closes_1h[-1]
    c_prev1 = closes_1h[-2]
    c_prev4 = closes_1h[-5] if len(closes_1h) >= 5 else closes_1h[0]

    ch1 = ((c_last - c_prev1) / c_prev1) * 100.0 if c_prev1 else 0.0
    ch4 = ((c_last - c_prev4) / c_prev4) * 100.0 if c_prev4 else 0.0
    atr = compute_atr_from_ohlcv(c1, ATR_PERIOD)

    c15 = fetch_ohlcv(market_symbol, "15m", limit=3)
    if not c15 or len(c15) < 2:
        ch15 = 0.0
    else:
        c15_last = float(c15[-1][4])
        c15_prev = float(c15[-2][4])
        ch15 = ((c15_last - c15_prev) / c15_prev) * 100.0 if c15_prev else 0.0

    return ch1, ch4, ch15, atr


# =========================
# SL/TP + MULTI TP (FIXED)
# =========================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sl_mult_from_conf(conf: int) -> float:
    if conf > 80:
        return 1.9
    if conf >= 70:
        return 1.6
    return 1.3


def sl_tp_from_atr(entry: float, side: str, atr: float, conf: int) -> Tuple[float, float, float]:
    """
    returns (sl, tp_single, r_abs)
    """
    if entry <= 0 or atr <= 0:
        return 0.0, 0.0, 0.0

    sl_dist = sl_mult_from_conf(conf) * atr
    min_dist = (ATR_MIN_PCT / 100.0) * entry
    max_dist = (ATR_MAX_PCT / 100.0) * entry
    sl_dist = clamp(sl_dist, min_dist, max_dist)

    r = sl_dist
    tp_dist = 1.4 * r
    tp_max_dist = (TP_MAX_PCT / 100.0) * entry
    tp_dist = min(tp_dist, tp_max_dist)

    if side == "BUY":
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist

    return sl, tp, r


def multi_tp_from_r(entry: float, side: str, r: float) -> Tuple[float, float, float]:
    """
    Prevent TP1/TP2/TP3 collapsing to same value when TP cap hits.
    - cap only TP3 distance
    - scale TP1/TP2 relative to capped TP3
    """
    if r <= 0 or entry <= 0:
        return 0.0, 0.0, 0.0

    r1, r2, r3 = TP_R_MULTS
    if r3 <= 0:
        r1, r2, r3 = (0.8, 1.4, 2.0)

    maxd = (TP_MAX_PCT / 100.0) * entry

    d3_raw = r3 * r
    d3 = min(d3_raw, maxd)

    d1 = d3 * (r1 / r3)
    d2 = d3 * (r2 / r3)

    if d1 >= d2:
        d1 = d2 * 0.98
    if d2 >= d3:
        d2 = d3 * 0.98

    if side == "BUY":
        return (entry + d1, entry + d2, entry + d3)
    else:
        return (entry - d1, entry - d2, entry - d3)


# =========================
# CHARTS
# =========================
def tv_chart_url(symbol_base: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol=BYBIT:{symbol_base.upper()}USDT.P"


def quickchart_png_bytes(title: str, labels: List[str], closes: List[float]) -> Optional[bytes]:
    chart_cfg = {
        "type": "line",
        "data": {"labels": labels, "datasets": [{"label": title, "data": closes, "fill": False, "pointRadius": 0, "borderWidth": 2, "tension": 0.25}]},
        "options": {"legend": {"display": False}, "scales": {"xAxes": [{"display": False}], "yAxes": [{"ticks": {"maxTicksLimit": 6}}]}},
    }
    qs = urllib.parse.quote(json.dumps(chart_cfg), safe="")
    url = f"https://quickchart.io/chart?c={qs}&format=png&width={CHART_WIDTH}&height={CHART_HEIGHT}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.read()
    except Exception:
        logger.exception("quickchart fetch failed")
        return None


def build_chart_png_for_market(symbol_base: str, market_symbol: str) -> Optional[bytes]:
    if not CHART_IMG_ENABLED:
        return None
    try:
        candles = fetch_ohlcv(market_symbol, CHART_TIMEFRAME, CHART_BARS)
        if len(candles) < 10:
            return None
        labels, closes = [], []
        for ts, o, h, l, c, v in candles:
            dt = datetime.fromtimestamp(ts / 1000, tz=ZoneInfo("UTC"))
            labels.append(dt.strftime("%m-%d %H:%M"))
            closes.append(float(c))
        title = f"{symbol_base.upper()} ({CHART_TIMEFRAME})"
        return quickchart_png_bytes(title, labels, closes)
    except Exception:
        logger.exception("build_chart_png_for_market failed")
        return None


# =========================
# FORMATTING HELPERS
# =========================
def fmt_price(x: float) -> str:
    ax = abs(x)
    if ax >= 100:
        return f"{x:.2f}"
    if ax >= 1:
        return f"{x:.3f}"
    if ax >= 0.1:
        return f"{x:.4f}"
    return f"{x:.5f}"


def fmt_money(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
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


def regime_label(ch24: float, ch4: float) -> str:
    if ch24 >= 3 and ch4 >= 2:
        return "LONG 🟢"
    if ch24 <= -3 and ch4 <= -2:
        return "SHORT 🔴"
    return "NEUTRAL 🟡"


def fmt_table(rows: List[List], headers: List[str]) -> str:
    return "```\n" + tabulate(rows, headers=headers, tablefmt="github") + "\n```"


def fmt_header(title: str) -> str:
    return f"✨ *{title}*"


def fmt_section(title: str) -> str:
    return f"\n— *{title}*"


def fmt_kv(key: str, val: str) -> str:
    return f"*{key}:* {val}"


def fmt_inline_setup_card(i: int, s: Setup) -> str:
    if s.tp1 and s.tp2 and s.conf >= MULTI_TP_MIN_CONF:
        tp_line = f"TP1 {fmt_price(s.tp1)} ({TP_ALLOCS[0]}%) | TP2 {fmt_price(s.tp2)} ({TP_ALLOCS[1]}%) | TP3 {fmt_price(s.tp)} ({TP_ALLOCS[2]}%)"
        rule = "Rule: after TP1 → SL to BE"
    else:
        tp_line = f"TP {fmt_price(s.tp)}"
        rule = ""

    base = (
        f"🔥 *Setup #{i}*  •  *{s.side} {s.symbol}*  •  Conf *{s.conf}/100*\n"
        f"{fmt_kv('Entry', fmt_price(s.entry))}  |  {fmt_kv('SL', fmt_price(s.sl))}\n"
        f"{tp_line}\n"
        f"{fmt_kv('24H', pct_with_emoji(s.ch24))}  |  {fmt_kv('4H', pct_with_emoji(s.ch4))}  |  "
        f"{fmt_kv('1H', pct_with_emoji(s.ch1))}  |  {fmt_kv('F Vol', '≈'+fmt_money(s.fut_vol_usd))}\n"
        f"🧭 {s.regime}"
    )
    if rule:
        base += f"\n🧠 {rule}"
    return base


# =========================
# MARKET BIAS
# =========================
def market_bias(best_fut: Dict[str, MarketVol]) -> str:
    top = sorted(best_fut.items(), key=lambda kv: usd_notional(kv[1]), reverse=True)[:BIAS_UNIVERSE_N]
    if not top:
        return "Market Bias: UNKNOWN"

    w_sum = 0.0
    score = 0.0
    long_n = 0
    short_n = 0

    for base, mv in top:
        vol = usd_notional(mv)
        if vol <= 0:
            continue

        ch1, ch4, ch15, atr = metrics_from_candles_1h_15m(mv.symbol)
        ch24 = float(mv.percentage or 0.0)

        if ch4 > 0:
            s = 1.0
            long_n += 1
        elif ch4 < 0:
            s = -1.0
            short_n += 1
        else:
            s = 0.0

        agree = 1.15 if (s > 0 and ch24 > 0) or (s < 0 and ch24 < 0) else 1.0
        w = vol * agree
        score += s * w
        w_sum += w

    if w_sum <= 0:
        return "Market Bias: UNKNOWN"

    norm = score / w_sum
    if norm >= BIAS_STRONG_TH:
        label = "LONG 🟢"
    elif norm <= -BIAS_STRONG_TH:
        label = "SHORT 🔴"
    else:
        label = "MIXED 🟡"

    return f"Market Bias: {label} | Top{BIAS_UNIVERSE_N}: Long {long_n} / Short {short_n}"


def bias_tag(bias_line: str) -> str:
    if "LONG 🟢" in bias_line:
        return "LONG"
    if "SHORT 🔴" in bias_line:
        return "SHORT"
    if "MIXED 🟡" in bias_line:
        return "MIXED"
    return "UNKNOWN"


# =========================
# SETUP ENGINE (FAST)
# =========================
def compute_confidence(side: str, ch24: float, ch4: float, ch1: float, ch15: float, fut_vol_usd: float) -> int:
    score = 50.0

    def add_align(x: float, is_long: bool, w: float):
        nonlocal score
        if is_long:
            score += w if x > 0 else -w
        else:
            score += w if x < 0 else -w

    is_long = (side == "BUY")
    add_align(ch24, is_long, 12.0)
    add_align(ch4, is_long, 10.0)
    add_align(ch1, is_long, 8.0)
    add_align(ch15, is_long, 6.0)

    mag = min(abs(ch24) / 2.0 + abs(ch4) + abs(ch1) * 2.0 + abs(ch15) * 2.0, 18.0)
    score += mag

    if fut_vol_usd >= 15_000_000:
        score += 8
    elif fut_vol_usd >= 6_000_000:
        score += 6
    elif fut_vol_usd >= 2_000_000:
        score += 4
    elif fut_vol_usd >= 1_000_000:
        score += 2

    score = max(0.0, min(100.0, score))
    return int(round(score))


def make_setup(base: str, mv: MarketVol) -> Optional[Setup]:
    fut_vol = usd_notional(mv)
    if fut_vol <= 0:
        return None

    ch24 = float(mv.percentage or 0.0)
    ch1, ch4, ch15, atr = metrics_from_candles_1h_15m(mv.symbol)

    if abs(ch1) < TRIGGER_1H_ABS_MIN:
        return None
    if abs(ch15) < CONFIRM_15M_ABS_MIN:
        return None

    side = "BUY" if ch1 > 0 else "SELL"

    if side == "BUY" and ch4 < ALIGN_4H_MIN:
        return None
    if side == "SELL" and ch4 > -ALIGN_4H_MIN:
        return None

    entry = float(mv.last or 0.0)
    if entry <= 0:
        return None

    conf = compute_confidence(side, ch24, ch4, ch1, ch15, fut_vol)
    reg = regime_label(ch24, ch4)

    if USE_ATR_SLTP:
        sl, tp_single, r = sl_tp_from_atr(entry, side, atr, conf)
        if sl <= 0 or tp_single <= 0 or r <= 0:
            return None

        tp1 = tp2 = None
        tp = tp_single

        if conf >= MULTI_TP_MIN_CONF:
            _tp1, _tp2, _tp3 = multi_tp_from_r(entry, side, r)
            if _tp1 > 0 and _tp2 > 0 and _tp3 > 0:
                tp1, tp2, tp = _tp1, _tp2, _tp3
    else:
        if side == "BUY":
            sl = entry * 0.97
            r = entry - sl
            tp = entry + 1.4 * r
        else:
            sl = entry * 1.03
            r = sl - entry
            tp = entry - 1.4 * r
        tp1 = tp2 = None

    return Setup(
        symbol=base,
        market_symbol=mv.symbol,
        side=side,
        conf=conf,
        entry=entry,
        sl=sl,
        tp=tp,
        tp1=tp1,
        tp2=tp2,
        fut_vol_usd=fut_vol,
        ch24=ch24,
        ch4=ch4,
        ch1=ch1,
        ch15=ch15,
        regime=reg,
        created_ts=time.time(),
    )


def pick_setups(best_fut: Dict[str, MarketVol]) -> List[Setup]:
    universe = sorted(best_fut.items(), key=lambda kv: usd_notional(kv[1]), reverse=True)[:30]
    setups: List[Setup] = []
    for base, mv in universe:
        s = make_setup(base, mv)
        if s:
            setups.append(s)
    setups.sort(key=lambda s: (s.conf, s.fut_vol_usd), reverse=True)
    return setups[:SETUPS_N]


# =========================
# DIRECTIONAL LISTS
# =========================
def compute_directional_lists(best_fut: Dict[str, MarketVol]) -> Tuple[List[Tuple], List[Tuple]]:
    up = []
    dn = []
    for base, mv in best_fut.items():
        vol = usd_notional(mv)
        if vol < MOVER_VOL_USD_MIN:
            continue

        ch24 = float(mv.percentage or 0.0)
        if ch24 < MOVER_UP_24H_MIN and ch24 > MOVER_DN_24H_MAX:
            continue

        ch1, ch4, ch15, atr = metrics_from_candles_1h_15m(mv.symbol)

        if ch24 >= MOVER_UP_24H_MIN:
            if (not MOVER_4H_TREND_FILTER) or (ch4 >= MOVER_4H_ALIGN_MIN):
                up.append((base, vol, ch24, ch4, float(mv.last or 0.0)))
        if ch24 <= MOVER_DN_24H_MAX:
            if (not MOVER_4H_TREND_FILTER) or (ch4 <= -MOVER_4H_ALIGN_MIN):
                dn.append((base, vol, ch24, ch4, float(mv.last or 0.0)))

    up.sort(key=lambda x: (x[2], x[1]), reverse=True)
    dn.sort(key=lambda x: (x[2], x[1]))
    return up, dn


# =========================
# SCREEN FORMATTERS
# =========================
def build_leaders_table(best_fut: Dict[str, MarketVol]) -> str:
    leaders = sorted(best_fut.items(), key=lambda kv: usd_notional(kv[1]), reverse=True)[:LEADERS_N]
    rows = []
    for base, mv in leaders:
        rows.append([base, fmt_money(usd_notional(mv)), pct_with_emoji(float(mv.percentage or 0.0)), fmt_price(float(mv.last or 0.0))])
    return "*Market Leaders (Top 10 by Futures Volume)*\n" + fmt_table(rows, ["SYM", "F Vol", "24H", "Last"])


def build_movers_tables(best_fut: Dict[str, MarketVol]) -> Tuple[str, str]:
    up, dn = compute_directional_lists(best_fut)

    up_rows = [[b, fmt_money(v), pct_with_emoji(c24), pct_with_emoji(c4), fmt_price(px)] for b, v, c24, c4, px in up[:MOVERS_N]]
    dn_rows = [[b, fmt_money(v), pct_with_emoji(c24), pct_with_emoji(c4), fmt_price(px)] for b, v, c24, c4, px in dn[:MOVERS_N]]

    up_txt = "*Directional Leaders (24H ≥ +10%, F vol ≥ 5M, 4H aligned)*\n" + (fmt_table(up_rows, ["SYM", "F Vol", "24H", "4H", "Last"]) if up_rows else "_None_")
    dn_txt = "*Directional Losers (24H ≤ -10%, F vol ≥ 5M, 4H aligned)*\n" + (fmt_table(dn_rows, ["SYM", "F Vol", "24H", "4H", "Last"]) if dn_rows else "_None_")
    return up_txt, dn_txt


def format_setups_for_screen_cards(setups: List[Setup]) -> str:
    if not setups:
        return "No high-quality setup right now."
    out = []
    for i, s in enumerate(setups, 1):
        out.append(fmt_inline_setup_card(i, s))
        out.append("")
    return "\n".join(out).strip()


# =========================
# RISK LOGIC
# =========================
def compute_trade_risk_usd(user: dict, equity: float, bias_line: str, conf: int) -> float:
    risk_mode = (user.get("risk_mode") or DEFAULT_RISK_MODE).upper()
    risk_value = float(user.get("risk_value") or DEFAULT_RISK_VALUE)

    if risk_mode == "USD":
        base = risk_value
    else:
        base = equity * (risk_value / 100.0)

    if int(user.get("risk_auto") or 0) == 1:
        tag = bias_tag(bias_line)
        factor = 1.0
        if tag == "MIXED":
            factor *= 0.7
        if conf >= 85:
            factor *= 1.1
        factor = clamp(factor, 0.5, 1.2)
        base *= factor

    return max(0.0, base)


def compute_daily_cap_usd(user: dict, equity: float) -> float:
    mode = (user.get("daily_cap_mode") or DEFAULT_DAILY_CAP_MODE).upper()
    val = float(user.get("daily_cap_value") or DEFAULT_DAILY_CAP_VALUE)
    if mode == "USD":
        return max(0.0, val)
    return max(0.0, equity * (val / 100.0))


def calc_position_size(entry: float, sl: float, risk_usd: float) -> Tuple[float, float]:
    risk_per_unit = abs(entry - sl)
    if risk_per_unit <= 0:
        return 0.0, 0.0
    qty = risk_usd / risk_per_unit
    notional = qty * entry
    return qty, notional


def suggested_leverage(notional: float, equity: float) -> int:
    if equity <= 0:
        return 1
    x = notional / equity
    if x <= 2.5:
        return 3
    if x <= 4.5:
        return 5
    return 10


# =========================
# SESSION TIME HELPERS
# =========================
def parse_hhmm(s: str) -> Tuple[int, int]:
    m = re.match(r"^(\d{1,2}):(\d{2})$", s.strip())
    if not m:
        raise ValueError("bad time")
    hh = int(m.group(1))
    mm = int(m.group(2))
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        raise ValueError("bad time")
    return hh, mm


def load_sessions(user: dict) -> List[dict]:
    try:
        data = user.get("trade_sessions") or json.dumps(DEFAULT_SESSIONS)
        ses = json.loads(data)
        if not isinstance(ses, list) or not ses:
            return DEFAULT_SESSIONS
        out = []
        for x in ses:
            if not isinstance(x, dict):
                continue
            st = str(x.get("start", "")).strip()
            en = str(x.get("end", "")).strip()
            parse_hhmm(st)
            parse_hhmm(en)
            out.append({"start": st, "end": en})
        return out or DEFAULT_SESSIONS
    except Exception:
        return DEFAULT_SESSIONS


def active_session_for_user(user: dict) -> Optional[dict]:
    """
    Returns dict: {start_dt, end_dt, start_str, end_str, session_key}
    or None if now is outside all sessions.
    Supports overnight sessions (end <= start means ends next day).
    """
    tz = ZoneInfo(user.get("tz") or "Australia/Melbourne")
    now = datetime.now(tz)
    sessions = load_sessions(user)

    for s in sessions:
        sh, sm = parse_hhmm(s["start"])
        eh, em = parse_hhmm(s["end"])

        start_dt = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end_dt = now.replace(hour=eh, minute=em, second=0, microsecond=0)

        if end_dt <= start_dt:
            end_dt = end_dt + timedelta(days=1)
            if now < start_dt:
                start_dt = start_dt - timedelta(days=1)
                end_dt = end_dt - timedelta(days=1)

        if start_dt <= now <= end_dt:
            session_key = f"{start_dt.strftime('%Y-%m-%d')}_{s['start']}"
            return {
                "start_dt": start_dt,
                "end_dt": end_dt,
                "start_str": s["start"],
                "end_str": s["end"],
                "session_key": session_key,
            }

    return None


# =========================
# EMAIL HELPERS
# =========================
def email_config_ok() -> bool:
    return all([EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FROM, EMAIL_TO])


def send_email(subject: str, body: str, attachments: Optional[List[Tuple[str, bytes, str]]] = None) -> bool:
    if not email_config_ok():
        logger.warning("Email not configured.")
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        msg.set_content(body)

        if attachments:
            for fn, data, mime in attachments:
                maintype, subtype = mime.split("/", 1)
                msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=fn)

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


def format_setup_email_block_pretty(s: Setup) -> str:
    lines = []
    lines.append(f"{s.side} {s.symbol} — Confidence {s.conf}/100")
    lines.append(f"Entry: {fmt_price(s.entry)}")
    lines.append(f"SL:    {fmt_price(s.sl)}")

    if s.tp1 and s.tp2 and s.conf >= MULTI_TP_MIN_CONF:
        lines.append(f"TP1:   {fmt_price(s.tp1)} ({TP_ALLOCS[0]}%)")
        lines.append(f"TP2:   {fmt_price(s.tp2)} ({TP_ALLOCS[1]}%)")
        lines.append(f"TP3:   {fmt_price(s.tp)} ({TP_ALLOCS[2]}%)")
        lines.append("Rule: After TP1 → move SL to BE")
    else:
        lines.append(f"TP:    {fmt_price(s.tp)}")

    lines.append("")
    lines.append(f"24H {pct_with_emoji(s.ch24)} | 4H {pct_with_emoji(s.ch4)} | 1H {pct_with_emoji(s.ch1)} | F~{fmt_money(s.fut_vol_usd)}")
    lines.append(f"Chart: {tv_chart_url(s.symbol)}")
    return "\n".join(lines)


# =========================
# HELP (PLAIN TEXT — NO MARKDOWN)
# =========================
HELP_TEXT = """\
PulseFutures — Commands

Market
- /screen  => Market Leaders + Directional Leaders/Losers + Top Setups

Email Session Controls (per user)
- /sessions
- /sessions_set 18:00-23:30,07:00-09:00
- /emailcap 3
- /emailgap 60

Risk & Trading
- /equity 1000
- /riskmode pct 2.5
- /riskmode usd 25
- /dailycap pct 5
- /dailycap usd 60
- /riskauto on|off
- /tradesetup BTC
- /risk BTC
- /open
- /closepnl BTC +23.5

Alerts
- /notify_on
- /notify_off
- /diag

Notes
- Emails only send inside YOUR sessions.
- A symbol emailed once in a session will NOT be emailed again until the NEXT session.
- Not financial advice.
"""


# =========================
# TELEGRAM HANDLERS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("START/HELP called by user_id=%s", update.effective_user.id if update.effective_user else None)
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return
    # ✅ send as plain text to avoid Markdown parse errors
    try:
        await context.bot.send_message(chat_id=chat_id, text=HELP_TEXT)
    except BadRequest:
        await context.bot.send_message(chat_id=chat_id, text=re.sub(r"[*_`]", "", HELP_TEXT))


async def diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = reset_daily_counters_if_needed(get_user(update.effective_user.id))
    notify = "ON" if user["notify_on"] else "OFF"

    equity = float(user["equity"])
    daily_cap = compute_daily_cap_usd(user, equity)

    risk_mode = (user.get("risk_mode") or DEFAULT_RISK_MODE).upper()
    risk_value = float(user.get("risk_value") or DEFAULT_RISK_VALUE)
    risk_desc = f"${risk_value:.2f} / trade" if risk_mode == "USD" else f"{risk_value:.2f}% / trade"

    dc_mode = (user.get("daily_cap_mode") or DEFAULT_DAILY_CAP_MODE).upper()
    dc_value = float(user.get("daily_cap_value") or DEFAULT_DAILY_CAP_VALUE)
    dc_desc = f"${dc_value:.2f} / day" if dc_mode == "USD" else f"{dc_value:.2f}% / day"

    ses = load_sessions(user)
    ses_txt = ", ".join([f"{x['start']}-{x['end']}" for x in ses])

    msg = (
        f"*Diag*\n"
        f"- Futures: Bybit swap\n"
        f"- Email notify: {notify}\n"
        f"- Your TZ: {user.get('tz')}\n"
        f"- Sessions: {ses_txt}\n"
        f"- Email cap/session: {int(user.get('max_emails_per_session') or DEFAULT_MAX_EMAILS_PER_SESSION)}\n"
        f"- Email gap: {int(user.get('email_gap_min') or DEFAULT_EMAIL_GAP_MIN)} min\n"
        f"- Equity: ${equity:.2f}\n"
        f"- Risk per trade: {risk_desc}\n"
        f"- Daily cap: {dc_desc} (≈ ${daily_cap:.2f})\n"
        f"- Risk auto: {'ON' if int(user.get('risk_auto') or 0)==1 else 'OFF'}\n"
        f"- Max trades/day: {user['max_trades_day']}\n"
        f"- Daily risk used: ${user['daily_risk_used']:.2f}\n"
        f"- Open risk cap (USD): ${user['open_risk_cap']:.2f}\n"
        f"- Time-stop: {SETUP_TIME_STOP_HOURS}h\n"
        f"- TP cap: {TP_MAX_PCT}%\n"
        f"- Multi-TP: Conf >= {MULTI_TP_MIN_CONF} | TP R-mults {TP_R_MULTS}\n"
        f"- Symbol cooldown: {'PER SESSION' if SYMBOL_COOLDOWN_PER_SESSION else 'OFF'}\n"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def notify_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    update_user(update.effective_user.id, notify_on=1)
    await update.message.reply_text("Email alerts: ON")


async def notify_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    update_user(update.effective_user.id, notify_on=0)
    await update.message.reply_text("Email alerts: OFF")


async def sessions_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = get_user(update.effective_user.id)
    ses = load_sessions(user)
    ses_txt = "\n".join([f"- {x['start']}-{x['end']}" for x in ses])
    await update.message.reply_text(
        f"Your sessions (TZ: {user.get('tz')}):\n{ses_txt}\n\nSet: /sessions_set 18:00-23:30,07:00-09:00"
    )


async def sessions_set_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Usage: /sessions_set 18:00-23:30,07:00-09:00")
        return
    raw = " ".join(context.args).strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    sessions = []
    try:
        for p in parts:
            if "-" not in p:
                raise ValueError()
            st, en = [x.strip() for x in p.split("-", 1)]
            parse_hhmm(st)
            parse_hhmm(en)
            sessions.append({"start": st, "end": en})
        if not sessions:
            raise ValueError()
    except Exception:
        await update.message.reply_text("Bad format. Example: /sessions_set 18:00-23:30,07:00-09:00")
        return

    update_user(uid, trade_sessions=json.dumps(sessions))
    await update.message.reply_text(f"✅ Sessions saved: {', '.join([f\"{x['start']}-{x['end']}\" for x in sessions])}")


async def emailcap_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if not context.args:
        await update.message.reply_text(
            f"Email cap/session: {int(user.get('max_emails_per_session') or DEFAULT_MAX_EMAILS_PER_SESSION)}\nSet: /emailcap 3"
        )
        return
    try:
        n = int(context.args[0])
        if n < 1 or n > 20:
            raise ValueError()
        update_user(uid, max_emails_per_session=n)
        await update.message.reply_text(f"✅ Email cap/session set to {n}")
    except Exception:
        await update.message.reply_text("Usage: /emailcap 3  (1..20)")


async def emailgap_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if not context.args:
        await update.message.reply_text(
            f"Email min gap: {int(user.get('email_gap_min') or DEFAULT_EMAIL_GAP_MIN)} minutes\nSet: /emailgap 60"
        )
        return
    try:
        m = int(context.args[0])
        if m < 0 or m > 360:
            raise ValueError()
        update_user(uid, email_gap_min=m)
        await update.message.reply_text(f"✅ Email min gap set to {m} minutes")
    except Exception:
        await update.message.reply_text("Usage: /emailgap 60  (0..360)")


async def equity_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args:
        user = get_user(uid)
        await update.message.reply_text(f"Equity: ${float(user['equity']):.2f}")
        return
    try:
        eq = float(context.args[0])
        if eq <= 0:
            raise ValueError()
        update_user(uid, equity=eq)
        await update.message.reply_text(f"Equity updated: ${eq:.2f}")
    except Exception:
        await update.message.reply_text("Usage: /equity 1000")


async def riskmode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)

    if not context.args:
        mode = (user.get("risk_mode") or DEFAULT_RISK_MODE).upper()
        val = float(user.get("risk_value") or DEFAULT_RISK_VALUE)
        if mode == "USD":
            await update.message.reply_text(f"Risk per trade: USD ${val:.2f}\nSet: /riskmode usd 25")
        else:
            await update.message.reply_text(f"Risk per trade: PCT {val:.2f}%\nSet: /riskmode pct 2.5")
        return

    if len(context.args) != 2:
        await update.message.reply_text("Usage: /riskmode pct 2.5  OR  /riskmode usd 25")
        return

    mode = context.args[0].strip().upper()
    try:
        val = float(context.args[1])
    except Exception:
        await update.message.reply_text("Usage: /riskmode pct 2.5  OR  /riskmode usd 25")
        return

    if mode not in {"PCT", "USD"}:
        await update.message.reply_text("Mode must be pct or usd. Example: /riskmode pct 2.5")
        return

    if mode == "PCT" and not (0.1 <= val <= 10.0):
        await update.message.reply_text("For pct, choose 0.5–5.0. Example: /riskmode pct 2.5")
        return
    if mode == "USD" and val <= 0:
        await update.message.reply_text("For usd, amount must be > 0. Example: /riskmode usd 25")
        return

    update_user(uid, risk_mode=mode, risk_value=val)
    await update.message.reply_text(f"✅ Risk per trade set: {mode} {val:.2f}")


async def dailycap_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    equity = float(user["equity"])

    if not context.args:
        mode = (user.get("daily_cap_mode") or DEFAULT_DAILY_CAP_MODE).upper()
        val = float(user.get("daily_cap_value") or DEFAULT_DAILY_CAP_VALUE)
        cap_usd = compute_daily_cap_usd(user, equity)
        if mode == "USD":
            await update.message.reply_text(f"Daily cap: USD ${val:.2f} (≈ ${cap_usd:.2f})\nSet: /dailycap usd 60")
        else:
            await update.message.reply_text(f"Daily cap: PCT {val:.2f}% (≈ ${cap_usd:.2f})\nSet: /dailycap pct 5")
        return

    if len(context.args) != 2:
        await update.message.reply_text("Usage: /dailycap pct 5  OR  /dailycap usd 60")
        return

    mode = context.args[0].strip().upper()
    try:
        val = float(context.args[1])
    except Exception:
        await update.message.reply_text("Usage: /dailycap pct 5  OR  /dailycap usd 60")
        return

    if mode not in {"PCT", "USD"}:
        await update.message.reply_text("Mode must be pct or usd. Example: /dailycap pct 5")
        return

    if mode == "PCT" and not (0.5 <= val <= 20.0):
        await update.message.reply_text("For pct/day, choose 3–8. Example: /dailycap pct 5")
        return
    if mode == "USD" and val < 0:
        await update.message.reply_text("For usd/day, amount must be >= 0. Example: /dailycap usd 60")
        return

    update_user(uid, daily_cap_mode=mode, daily_cap_value=val)
    user = get_user(uid)
    cap_usd = compute_daily_cap_usd(user, float(user["equity"]))
    await update.message.reply_text(f"✅ Daily cap set: {mode} {val:.2f} (≈ ${cap_usd:.2f})")


async def riskauto_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args or context.args[0].strip().lower() not in {"on", "off"}:
        user = get_user(uid)
        cur = "ON" if int(user.get("risk_auto") or 0) == 1 else "OFF"
        await update.message.reply_text(f"Risk auto is {cur}\nUsage: /riskauto on  OR  /riskauto off")
        return
    val = 1 if context.args[0].strip().lower() == "on" else 0
    update_user(uid, risk_auto=val)
    await update.message.reply_text(f"✅ Risk auto: {'ON' if val==1 else 'OFF'}")


async def screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    best_fut = fetch_futures_tickers()
    if not best_fut:
        await update.message.reply_text("Error: could not fetch futures tickers.")
        return

    bias_line = market_bias(best_fut)

    leaders_txt = build_leaders_table(best_fut)
    dir_up_txt, dir_dn_txt = build_movers_tables(best_fut)

    setups = pick_setups(best_fut)
    for s in setups:
        db_upsert_signal(s)

    parts = []
    parts.append(fmt_header("PulseFutures — Market Scan"))
    parts.append(f"🧠 *{bias_line}*")
    parts.append(fmt_section("Top Trade Setups"))
    parts.append(format_setups_for_screen_cards(setups))
    parts.append(fmt_section("Directional Leaders / Losers"))
    parts.append(dir_up_txt)
    parts.append("")
    parts.append(dir_dn_txt)
    parts.append(fmt_section("Market Leaders (Volume)"))
    parts.append(leaders_txt)

    msg = "\n".join(parts).strip()

    keyboard = []
    for s in setups:
        keyboard.append([InlineKeyboardButton(text=f"📈 {s.symbol} Chart", url=tv_chart_url(s.symbol))])

    await update.message.reply_text(
        msg,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None,
        disable_web_page_preview=True,
    )

    for s in setups:
        png = build_chart_png_for_market(s.symbol, s.market_symbol)
        if png:
            cap = f"{s.symbol} • {s.side} • Conf {s.conf}/100 • {CHART_TIMEFRAME}"
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=BytesIO(png), caption=cap)


async def tradesetup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = reset_daily_counters_if_needed(get_user(uid))

    if not context.args:
        await update.message.reply_text("Usage: /tradesetup BTC")
        return

    sym = re.sub(r"[^A-Za-z0-9]", "", context.args[0]).upper().lstrip("$")
    if not sym:
        await update.message.reply_text("Usage: /tradesetup BTC")
        return

    sig = db_get_signal(sym)
    warning = ""
    if not sig:
        warning = f"⚠️ {sym} is not in current PulseFutures signals. You can still use TradeSetup for sizing, but responsibility is yours.\n\n"

    best_fut = fetch_futures_tickers()
    mv = best_fut.get(sym)
    if not mv:
        await update.message.reply_text(f"{warning}Could not find {sym} on Bybit futures.")
        return

    bias_line = market_bias(best_fut)

    ch24 = float(mv.percentage or 0.0)
    ch1, ch4, ch15, atr = metrics_from_candles_1h_15m(mv.symbol)

    side = "BUY" if ch1 >= 0 else "SELL"

    entry = float(mv.last or 0.0)
    if entry <= 0:
        await update.message.reply_text(f"{warning}Invalid price for {sym}.")
        return

    fut_vol = usd_notional(mv)
    conf = compute_confidence(side, ch24, ch4, ch1, ch15, fut_vol)

    sl, tp_single, r = sl_tp_from_atr(entry, side, atr, conf)
    if sl <= 0 or tp_single <= 0 or r <= 0:
        await update.message.reply_text(f"{warning}Could not compute ATR-based SL/TP for {sym}.")
        return

    tp1 = tp2 = None
    tp = tp_single
    if conf >= MULTI_TP_MIN_CONF:
        _tp1, _tp2, _tp3 = multi_tp_from_r(entry, side, r)
        if _tp1 > 0 and _tp2 > 0 and _tp3 > 0:
            tp1, tp2, tp = _tp1, _tp2, _tp3

    open_positions = db_get_open_positions(uid)
    open_risk = sum(float(p["risk_usd"]) for p in open_positions)
    if open_risk >= float(user["open_risk_cap"]):
        await update.message.reply_text("❌ Open risk cap reached. Close some positions or increase your open risk cap.")
        return

    if int(user["day_trade_count"]) >= int(user["max_trades_day"]):
        await update.message.reply_text("❌ Max trades per day reached. Try again tomorrow or increase your daily limit.")
        return

    equity = float(user["equity"])
    daily_cap_usd = compute_daily_cap_usd(user, equity)
    daily_remaining = daily_cap_usd - float(user["daily_risk_used"])
    open_remaining = float(user["open_risk_cap"]) - open_risk

    trade_risk_target = compute_trade_risk_usd(user, equity, bias_line, conf)
    risk_usd = min(trade_risk_target, daily_remaining, open_remaining)

    if risk_usd <= 0:
        await update.message.reply_text("❌ No risk budget remaining (daily cap or open cap).")
        return

    qty, notional = calc_position_size(entry, sl, risk_usd)
    if qty <= 0 or notional <= 0:
        await update.message.reply_text("Could not compute position size (invalid entry/SL).")
        return

    lev = suggested_leverage(notional, equity)
    notes = "PulseFutures signal" if db_get_signal(sym) else "Manual (not in signals)"
    time_stop_ts = time.time() + SETUP_TIME_STOP_HOURS * 3600

    db_add_position(uid, sym, side, entry, sl, tp, risk_usd, qty, notional, conf, time_stop_ts=time_stop_ts, notes=notes)
    update_user(uid, day_trade_count=int(user["day_trade_count"]) + 1, daily_risk_used=float(user["daily_risk_used"]) + risk_usd)

    tz = ZoneInfo(user.get("tz") or "Australia/Melbourne")
    deadline = datetime.fromtimestamp(time_stop_ts, tz=tz).strftime("%Y-%m-%d %H:%M")

    if tp1 and tp2 and conf >= MULTI_TP_MIN_CONF:
        tp_line = f"TP1 {fmt_price(tp1)} ({TP_ALLOCS[0]}%) | TP2 {fmt_price(tp2)} ({TP_ALLOCS[1]}%) | TP3 {fmt_price(tp)} ({TP_ALLOCS[2]}%)\n🧠 Rule: after TP1 → SL to BE"
    else:
        tp_line = f"TP {fmt_price(tp)}"

    msg = (
        warning
        + f"✅ *TradeSetup* — *{side} {sym}*  •  Conf *{conf}/100*\n"
        + f"{fmt_kv('Entry', fmt_price(entry))}  |  {fmt_kv('SL', fmt_price(sl))}\n"
        + f"{tp_line}\n"
        + f"{fmt_kv('Risk', f'${risk_usd:.2f}')}  |  {fmt_kv('Qty', f'{qty:.6g}')}  |  {fmt_kv('Notional', f'${notional:.2f}')}  |  {fmt_kv('Lev', f'{lev}x')}\n"
        + f"🧠 {bias_line}\n"
        + f"⏱️ Time-Stop: {SETUP_TIME_STOP_HOURS}h (Deadline: {deadline})\n"
        + f"📈 Chart: {tv_chart_url(sym)}"
    )

    kb = InlineKeyboardMarkup([[InlineKeyboardButton("📈 Chart (TV)", url=tv_chart_url(sym))]])
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=kb, disable_web_page_preview=True)

    png = build_chart_png_for_market(sym, mv.symbol)
    if png:
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=BytesIO(png), caption=f"{sym} • {CHART_TIMEFRAME}")


async def risk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await tradesetup(update, context)


async def open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = reset_daily_counters_if_needed(get_user(uid))
    pos = db_get_open_positions(uid)
    if not pos:
        await update.message.reply_text("No open positions.")
        return

    lines = [f"*Open Positions* (Equity ${float(user['equity']):.2f})\n"]
    now_ts = time.time()

    for i, p in enumerate(pos, 1):
        expired_tag = ""
        tst = p.get("time_stop_ts")
        if tst and now_ts > float(tst):
            expired_tag = " ⏱️EXPIRED"

        lines.append(
            f"#{i} {p['symbol']} {p['side']}{expired_tag} | Entry {fmt_price(float(p['entry']))} | "
            f"SL {fmt_price(float(p['sl']))} | TP {fmt_price(float(p['tp']))} | Conf {p.get('conf') or '-'}"
        )
        lines.append(
            f"Risk ${float(p['risk_usd']):.2f} | Qty {float(p['qty']):.6g} | "
            f"Notional ${float(p['notional']):.2f} | {p.get('notes') or ''}"
        )
        lines.append("")

    await update.message.reply_text("\n".join(lines).strip(), parse_mode=ParseMode.MARKDOWN)


async def closepnl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /closepnl BTC +23.5")
        return
    sym = re.sub(r"[^A-Za-z0-9]", "", context.args[0]).upper().lstrip("$")
    if not sym:
        await update.message.reply_text("Usage: /closepnl BTC +23.5")
        return
    try:
        pnl = float(context.args[1])
    except Exception:
        await update.message.reply_text("Usage: /closepnl BTC +23.5")
        return

    pos = db_close_position(uid, sym, pnl)
    if not pos:
        await update.message.reply_text(f"No open position found for {sym}.")
        return

    user = get_user(uid)
    new_eq = float(user["equity"]) + pnl
    update_user(uid, equity=new_eq)

    await update.message.reply_text(f"✅ Closed {sym} ({pos['side']}). PnL: {pnl:+.2f}\nEquity updated: ${new_eq:.2f}")


async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    token = re.sub(r"[^A-Za-z0-9$]", "", text).upper().lstrip("$")
    if len(token) < 2:
        return

    sig = db_get_signal(token)
    if sig:
        tp1 = sig.get("tp1")
        tp2 = sig.get("tp2")
        if tp1 and tp2 and int(sig.get("conf") or 0) >= MULTI_TP_MIN_CONF:
            tp_line = f"TP1 {fmt_price(float(tp1))} ({TP_ALLOCS[0]}%) | TP2 {fmt_price(float(tp2))} ({TP_ALLOCS[1]}%) | TP3 {fmt_price(float(sig['tp']))} ({TP_ALLOCS[2]}%)"
        else:
            tp_line = f"TP {fmt_price(float(sig['tp']))}"

        msg = (
            f"🔎 *{token}* in PulseFutures signals\n"
            f"*{sig['side']}* — Conf *{sig['conf']}/100*\n"
            f"{fmt_kv('Entry', fmt_price(float(sig['entry'])))}  |  {fmt_kv('SL', fmt_price(float(sig['sl'])))}\n"
            f"{tp_line}\n"
            f"{fmt_kv('F Vol', '≈'+fmt_money(float(sig['fut_vol_usd'])))}  |  "
            f"{fmt_kv('24H', pct_with_emoji(float(sig['ch24'])))}  |  {fmt_kv('4H', pct_with_emoji(float(sig['ch4'])))}\n"
            f"📈 {tv_chart_url(token)}\n\n"
            f"TradeSetup: /tradesetup {token}"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)
    else:
        await update.message.reply_text(f"{token} not found in current PulseFutures signals.\nYou can still size risk: /tradesetup {token}")


# =========================
# EMAIL JOB (PER USER, PER SESSION) + SYMBOL COOLDOWN PER SESSION
# =========================
async def alert_job(context: ContextTypes.DEFAULT_TYPE):
    try:
        if not EMAIL_ENABLED_DEFAULT:
            return
        if not email_config_ok():
            return

        users = list_users_notify_on()
        if not users:
            return

        best_fut = fetch_futures_tickers()
        if not best_fut:
            return

        bias_line = market_bias(best_fut)
        dir_up, dir_dn = compute_directional_lists(best_fut)
        setups = pick_setups(best_fut)
        for s in setups:
            db_upsert_signal(s)

        for user in users:
            uid = int(user["user_id"])
            sess = active_session_for_user(user)
            if not sess:
                continue

            st = email_state_get(uid)

            if st["session_key"] != sess["session_key"]:
                email_state_set(uid, session_key=sess["session_key"], sent_count=0, last_email_ts=0.0, last_symbols="")
                st = email_state_get(uid)

            max_emails = int(user.get("max_emails_per_session") or DEFAULT_MAX_EMAILS_PER_SESSION)
            gap_min = int(user.get("email_gap_min") or DEFAULT_EMAIL_GAP_MIN)
            gap_sec = max(0, gap_min) * 60

            if int(st["sent_count"]) >= max_emails:
                continue

            now_ts = time.time()
            if gap_sec > 0 and (now_ts - float(st["last_email_ts"] or 0.0)) < gap_sec:
                continue

            session_key = sess["session_key"]
            filtered_setups: List[Setup] = []
            for s in setups:
                if SYMBOL_COOLDOWN_PER_SESSION and has_emailed_symbol_in_session(uid, session_key, s.symbol):
                    continue
                filtered_setups.append(s)

            if not filtered_setups:
                continue

            last_syms = set([x for x in (st["last_symbols"] or "").split(",") if x])
            cur_syms = set([s.symbol for s in filtered_setups])
            if cur_syms and cur_syms == last_syms:
                continue

            attachments = []
            for s in filtered_setups:
                png = build_chart_png_for_market(s.symbol, s.market_symbol)
                if png:
                    attachments.append((f"chart_{s.symbol}_{CHART_TIMEFRAME}.png", png, "image/png"))

            tz = ZoneInfo(user.get("tz") or "Australia/Melbourne")
            now_local = datetime.now(tz).strftime("%Y-%m-%d %H:%M")

            parts: List[str] = []
            parts.append(HDR)
            parts.append(f"📊 PulseFutures — Session Update ({now_local} {user.get('tz')})")
            parts.append(HDR)
            parts.append("")
            parts.append(bias_line)
            parts.append(f"Session: {sess['start_str']}-{sess['end_str']} | Email {int(st['sent_count'])+1}/{max_emails} | Gap {gap_min}m")
            parts.append("")

            parts.append(SEP)
            parts.append("🔥 Top Trade Setups")
            parts.append(SEP)
            parts.append("")

            for i, s in enumerate(filtered_setups, 1):
                parts.append(f"{i}) {format_setup_email_block_pretty(s)}")
                parts.append("")
                parts.append(SEP)

            if INCLUDE_MOVERS_IN_EMAIL:
                parts.append("")
                parts.append("📈 Directional Leaders (Top 5)")
                parts.append(SEP)
                if dir_up:
                    for b, v, c24, c4, _ in dir_up[:EMAIL_MOVERS_TOP_N]:
                        parts.append(f"{b:<6} {pct_with_emoji(c24)} | 4H {pct_with_emoji(c4)} | F~{fmt_money(v)}")
                else:
                    parts.append("None")

                parts.append("")
                parts.append("📉 Directional Losers (Top 5)")
                parts.append(SEP)
                if dir_dn:
                    for b, v, c24, c4, _ in dir_dn[:EMAIL_MOVERS_TOP_N]:
                        parts.append(f"{b:<6} {pct_with_emoji(c24)} | 4H {pct_with_emoji(c4)} | F~{fmt_money(v)}")
                else:
                    parts.append("None")

            parts.append("")
            parts.append(HDR)
            parts.append("This is not financial advice.")
            parts.append("PulseFutures • Bybit Futures")
            parts.append(HDR)

            body = "\n".join(parts).strip()
            subject = f"PulseFutures • Setups ({sess['start_str']}-{sess['end_str']})"

            if send_email(subject, body, attachments=attachments):
                email_state_set(
                    uid,
                    sent_count=int(st["sent_count"]) + 1,
                    last_email_ts=now_ts,
                    last_symbols=",".join(sorted(cur_syms)),
                )

                if SYMBOL_COOLDOWN_PER_SESSION:
                    mark_emailed_symbols_in_session(uid, session_key, [s.symbol for s in filtered_setups])

    except Exception as e:
        logger.exception("alert_job error: %s", e)


# =========================
# GLOBAL ERROR HANDLER
# =========================
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled exception in Telegram handler", exc_info=context.error)
    # (optional) If you want to notify admin, you can send message here.


# =========================
# MAIN
# =========================
def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN missing")

    db_init()

    app = Application.builder().token(TOKEN).build()

    app.add_error_handler(on_error)

    app.add_handler(CommandHandler(["help", "start"], start))
    app.add_handler(CommandHandler("screen", screen))
    app.add_handler(CommandHandler("diag", diag))
    app.add_handler(CommandHandler("notify_on", notify_on))
    app.add_handler(CommandHandler("notify_off", notify_off))

    app.add_handler(CommandHandler("sessions", sessions_cmd))
    app.add_handler(CommandHandler("sessions_set", sessions_set_cmd))
    app.add_handler(CommandHandler("emailcap", emailcap_cmd))
    app.add_handler(CommandHandler("emailgap", emailgap_cmd))

    app.add_handler(CommandHandler("equity", equity_cmd))
    app.add_handler(CommandHandler("riskmode", riskmode_cmd))
    app.add_handler(CommandHandler("dailycap", dailycap_cmd))
    app.add_handler(CommandHandler("riskauto", riskauto_cmd))

    app.add_handler(CommandHandler("tradesetup", tradesetup))
    app.add_handler(CommandHandler("risk", risk_cmd))
    app.add_handler(CommandHandler("open", open_cmd))
    app.add_handler(CommandHandler("closepnl", closepnl_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    if getattr(app, "job_queue", None):
        app.job_queue.run_repeating(alert_job, interval=CHECK_INTERVAL_MIN * 60, first=10)
    else:
        logger.warning('JobQueue not available. Install "python-telegram-bot[job-queue]>=20.7,<22.0"')

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
