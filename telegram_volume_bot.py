#!/usr/bin/env python3
"""
PulseFutures — Bybit Futures (Swap) Screener + Risk Manager + Pro Trading Journal + Email Signal Alerts
International-ready (per-user timezone), designed for day traders.

GOALS (your spec)
- Telegram: fast /screen, risk settings, journal (open/close trades), status dashboard
- Email: beautiful plain-text (clean), includes:
    * best signals (high probability)
    * quick market summary
    * TradingView link for each signal
- Sessions: New York (best) -> London -> Asia
  Per-user: user selects timezone; sessions are evaluated in that timezone
- Per session: 3–4 emails, minimum 60 minutes gap
- No repeated symbol inside the SAME session
- Also: optional "global cooldown" (18h) supported via env (default ON)

IMPORTANT NOTE ABOUT 80%+ WIN RATE
- Code can filter aggressively (higher confidence threshold, multi-timeframe alignment, volume filter),
  but "80%+" depends on market regime & execution; the bot is tuned to prefer fewer, higher-quality setups.

ENV REQUIRED
- TELEGRAM_TOKEN

EMAIL ENV (required for email to actually send)
- EMAIL_ENABLED=true
- EMAIL_HOST
- EMAIL_PORT (465 recommended)
- EMAIL_USER
- EMAIL_PASS
- EMAIL_FROM
- EMAIL_TO

OPTIONAL ENV
- CHECK_INTERVAL_MIN=5
- DB_PATH=pulsefutures.db
- DEFAULT_EMAIL_GAP_MIN=60
- MAX_EMAILS_PER_SESSION=4
- SYMBOL_EMAIL_COOLDOWN_HOURS=18        (global cooldown across sessions; set 0 to disable)
- SIGNAL_MIN_CONF=78                    (raise to reduce quantity, improve quality)
- USE_4H_ALIGN=true                     (stronger directional filter)
- USE_24H_ALIGN=true                    (stronger directional filter)
- ATR_MIN_PCT=0.9                       (wider SL -> fewer quick stopouts)
- ATR_MAX_PCT=7.0
- TP_MAX_PCT=6.0

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
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

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

# =========================
# CONFIG
# =========================
EXCHANGE_ID = "bybit"
DEFAULT_TYPE = "swap"
TOKEN = os.environ.get("TELEGRAM_TOKEN")

DB_PATH = os.environ.get("DB_PATH", "pulsefutures.db")
CHECK_INTERVAL_MIN = int(os.environ.get("CHECK_INTERVAL_MIN", "5"))

# Email
EMAIL_ENABLED = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST = os.environ.get("EMAIL_HOST", "")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "465"))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", EMAIL_USER)
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_USER)

DEFAULT_EMAIL_GAP_MIN = int(os.environ.get("DEFAULT_EMAIL_GAP_MIN", "60"))
MAX_EMAILS_PER_SESSION = int(os.environ.get("MAX_EMAILS_PER_SESSION", "4"))

SYMBOL_EMAIL_COOLDOWN_HOURS = int(os.environ.get("SYMBOL_EMAIL_COOLDOWN_HOURS", "18"))
SYMBOL_EMAIL_COOLDOWN_SEC = max(0, SYMBOL_EMAIL_COOLDOWN_HOURS) * 3600

# Sessions priority order (NY best -> London -> Asia)
# Sessions are evaluated in USER timezone. These are "local" trading windows.
# If user sets tz=America/New_York, then NY session naturally matches. If tz differs, they still get these windows in their local time.
SESSIONS = [
    {"name": "NEW_YORK", "start": "20:00", "end": "23:59"},  # local time window (tuned for typical evening in many regions)
    {"name": "LONDON",   "start": "15:00", "end": "18:00"},
    {"name": "ASIA",     "start": "09:00", "end": "12:00"},
]

# Scan sizes
LEADERS_N = 10
DIR_N = 10
SETUPS_N = 3  # how many signals to email/show

# Filters / quality knobs
SIGNAL_MIN_CONF = int(os.environ.get("SIGNAL_MIN_CONF", "78"))  # raise to get fewer but stronger signals
USE_4H_ALIGN = os.environ.get("USE_4H_ALIGN", "true").lower() == "true"
USE_24H_ALIGN = os.environ.get("USE_24H_ALIGN", "true").lower() == "true"

# Directional movers
MOVER_VOL_USD_MIN = 5_000_000
MOVER_UP_24H_MIN = 10.0
MOVER_DN_24H_MAX = -10.0
MOVER_4H_ALIGN_MIN = 1.0

# Setup thresholds
TRIGGER_1H_ABS_MIN = 2.0
CONFIRM_15M_ABS_MIN = 0.6

# ATR SL/TP
ATR_PERIOD = 14
ATR_MIN_PCT = float(os.environ.get("ATR_MIN_PCT", "0.9"))
ATR_MAX_PCT = float(os.environ.get("ATR_MAX_PCT", "7.0"))
TP_MAX_PCT = float(os.environ.get("TP_MAX_PCT", "6.0"))

MULTI_TP_MIN_CONF = 78
TP_ALLOCS = (40, 35, 25)
TP_R_MULTS = (0.9, 1.6, 2.4)

# Risk defaults (NO default equity -> 0)
DEFAULT_TZ = "UTC"
DEFAULT_EQUITY = 0.0
DEFAULT_RISK_MODE = "USD"  # safer for new users (works even if equity=0)
DEFAULT_RISK_VALUE = 40.0
DEFAULT_DAILY_CAP_MODE = "USD"
DEFAULT_DAILY_CAP_VALUE = 200.0
DEFAULT_MAX_TRADES_DAY = 5

HDR = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
SEP = "────────────────────────────────────"
STABLES = {"USDT", "USDC", "USD", "TUSD", "FDUSD", "DAI", "PYUSD"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pulsefutures")

ALERT_LOCK = asyncio.Lock()

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
    symbol: str          # base e.g. BTC
    market_symbol: str   # ccxt symbol
    side: str            # BUY / SELL
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

    cur.execute(
        """
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
            day_trade_date TEXT NOT NULL,
            day_trade_count INTEGER NOT NULL,
            daily_risk_used REAL NOT NULL
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
            tp1 REAL,
            tp2 REAL,
            tp3 REAL NOT NULL,
            fut_vol_usd REAL NOT NULL,
            ch24 REAL NOT NULL,
            ch4 REAL NOT NULL,
            ch1 REAL NOT NULL,
            ch15 REAL NOT NULL,
            created_ts REAL NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS email_state (
            user_id INTEGER PRIMARY KEY,
            session_key TEXT NOT NULL,
            sent_count INTEGER NOT NULL,
            last_email_ts REAL NOT NULL
        )
        """
    )

    # no repeats inside session
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS emailed_symbols_session (
            user_id INTEGER NOT NULL,
            session_key TEXT NOT NULL,
            symbol TEXT NOT NULL,
            PRIMARY KEY (user_id, session_key, symbol)
        )
        """
    )

    # optional global cooldown across sessions
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS emailed_symbols_global (
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            last_sent_ts REAL NOT NULL,
            PRIMARY KEY (user_id, symbol)
        )
        """
    )

    # journal
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry REAL NOT NULL,
            sl REAL,
            tp REAL,
            risk_usd REAL NOT NULL,
            opened_ts REAL NOT NULL,
            status TEXT NOT NULL,
            exit REAL,
            pnl REAL,
            closed_ts REAL
        )
        """
    )

    con.commit()
    con.close()


def get_user(user_id: int) -> dict:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()

    if not row:
        today = datetime.now(ZoneInfo(DEFAULT_TZ)).date().isoformat()
        cur.execute(
            """
            INSERT INTO users (
                user_id, tz, equity, risk_mode, risk_value, daily_cap_mode, daily_cap_value,
                max_trades_day, notify_on, day_trade_date, day_trade_count, daily_risk_used
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, DEFAULT_TZ, float(DEFAULT_EQUITY),
                DEFAULT_RISK_MODE, float(DEFAULT_RISK_VALUE),
                DEFAULT_DAILY_CAP_MODE, float(DEFAULT_DAILY_CAP_VALUE),
                int(DEFAULT_MAX_TRADES_DAY),
                1 if EMAIL_ENABLED else 0,
                today, 0, 0.0
            ),
        )
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
    tz = ZoneInfo(user.get("tz") or DEFAULT_TZ)
    today = datetime.now(tz).date().isoformat()
    if user.get("day_trade_date") != today:
        update_user(user["user_id"], day_trade_date=today, day_trade_count=0, daily_risk_used=0.0)
        user = get_user(user["user_id"])
    return user


def db_upsert_signal(s: Setup):
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO signals (
            symbol, market_symbol, side, conf, entry, sl, tp1, tp2, tp3,
            fut_vol_usd, ch24, ch4, ch1, ch15, created_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            market_symbol=excluded.market_symbol,
            side=excluded.side,
            conf=excluded.conf,
            entry=excluded.entry,
            sl=excluded.sl,
            tp1=excluded.tp1,
            tp2=excluded.tp2,
            tp3=excluded.tp3,
            fut_vol_usd=excluded.fut_vol_usd,
            ch24=excluded.ch24,
            ch4=excluded.ch4,
            ch1=excluded.ch1,
            ch15=excluded.ch15,
            created_ts=excluded.created_ts
        """,
        (
            s.symbol, s.market_symbol, s.side, s.conf, s.entry, s.sl, s.tp1, s.tp2, s.tp3,
            s.fut_vol_usd, s.ch24, s.ch4, s.ch1, s.ch15, s.created_ts
        )
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
    d = dict(row)
    if time.time() - float(d["created_ts"]) > ttl_sec:
        con = db_connect()
        cur = con.cursor()
        cur.execute("DELETE FROM signals WHERE symbol=?", (symbol.upper(),))
        con.commit()
        con.close()
        return None
    return d


def email_state_get(user_id: int) -> dict:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM email_state WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    if not row:
        cur.execute(
            "INSERT INTO email_state (user_id, session_key, sent_count, last_email_ts) VALUES (?, ?, ?, ?)",
            (user_id, "NONE", 0, 0.0),
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


def session_symbol_sent(user_id: int, session_key: str, symbol: str) -> bool:
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        "SELECT 1 FROM emailed_symbols_session WHERE user_id=? AND session_key=? AND symbol=?",
        (user_id, session_key, symbol.upper()),
    )
    row = cur.fetchone()
    con.close()
    return bool(row)


def mark_session_symbols(user_id: int, session_key: str, symbols: List[str]):
    if not symbols:
        return
    con = db_connect()
    cur = con.cursor()
    for sym in symbols:
        cur.execute(
            "INSERT OR IGNORE INTO emailed_symbols_session (user_id, session_key, symbol) VALUES (?, ?, ?)",
            (user_id, session_key, sym.upper()),
        )
    con.commit()
    con.close()


def emailed_recently_global(user_id: int, symbol: str, cooldown_sec: int) -> bool:
    if cooldown_sec <= 0:
        return False
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        "SELECT last_sent_ts FROM emailed_symbols_global WHERE user_id=? AND symbol=?",
        (user_id, symbol.upper()),
    )
    row = cur.fetchone()
    con.close()
    if not row:
        return False
    return (time.time() - float(row["last_sent_ts"])) < cooldown_sec


def mark_global_symbols(user_id: int, symbols: List[str]):
    now = time.time()
    con = db_connect()
    cur = con.cursor()
    for sym in symbols:
        cur.execute(
            """
            INSERT INTO emailed_symbols_global (user_id, symbol, last_sent_ts)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, symbol) DO UPDATE SET last_sent_ts=excluded.last_sent_ts
            """,
            (user_id, sym.upper(), now),
        )
    con.commit()
    con.close()


# Journal
def journal_open_trade(user_id: int, symbol: str, side: str, entry: float, sl: Optional[float], tp: Optional[float], risk_usd: float):
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO trades (user_id, symbol, side, entry, sl, tp, risk_usd, opened_ts, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
        """,
        (user_id, symbol.upper(), side.upper(), float(entry),
         float(sl) if sl is not None else None,
         float(tp) if tp is not None else None,
         float(risk_usd), time.time())
    )
    con.commit()
    con.close()


def journal_close_trade(user_id: int, symbol: str, exit_price: Optional[float], pnl: float) -> Optional[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT * FROM trades
        WHERE user_id=? AND status='OPEN' AND symbol=?
        ORDER BY opened_ts ASC
        LIMIT 1
        """,
        (user_id, symbol.upper())
    )
    row = cur.fetchone()
    if not row:
        con.close()
        return None
    t = dict(row)
    cur.execute(
        """
        UPDATE trades
        SET status='CLOSED', exit=?, pnl=?, closed_ts=?
        WHERE id=?
        """,
        (float(exit_price) if exit_price is not None else None, float(pnl), time.time(), t["id"])
    )
    con.commit()
    con.close()
    t["exit"] = exit_price
    t["pnl"] = pnl
    return t


def journal_open_trades(user_id: int) -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        "SELECT * FROM trades WHERE user_id=? AND status='OPEN' ORDER BY opened_ts ASC",
        (user_id,)
    )
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


# =========================
# EXCHANGE
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


def usd_notional_mv(mv: MarketVol) -> float:
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
        if mv.base not in best or usd_notional_mv(mv) > usd_notional_mv(best[mv.base]):
            best[mv.base] = mv
    return best


def fetch_ohlcv(symbol: str, timeframe: str, limit: int):
    ex = build_exchange()
    ex.load_markets()
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit) or []


def compute_atr(candles: List[List[float]], period: int) -> float:
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


def metrics_1h_4h_15m(market_symbol: str) -> Tuple[float, float, float, float]:
    need_1h = max(ATR_PERIOD + 6, 40)
    c1 = fetch_ohlcv(market_symbol, "1h", limit=need_1h)
    if not c1 or len(c1) < 8:
        return 0.0, 0.0, 0.0, 0.0

    closes_1h = [float(x[4]) for x in c1]
    last = closes_1h[-1]
    prev1 = closes_1h[-2]
    prev4 = closes_1h[-5] if len(closes_1h) >= 5 else closes_1h[0]

    ch1 = ((last - prev1) / prev1) * 100.0 if prev1 else 0.0
    ch4 = ((last - prev4) / prev4) * 100.0 if prev4 else 0.0
    atr = compute_atr(c1, ATR_PERIOD)

    c15 = fetch_ohlcv(market_symbol, "15m", limit=3)
    if not c15 or len(c15) < 2:
        ch15 = 0.0
    else:
        c15_last = float(c15[-1][4])
        c15_prev = float(c15[-2][4])
        ch15 = ((c15_last - c15_prev) / c15_prev) * 100.0 if c15_prev else 0.0

    return ch1, ch4, ch15, atr


# =========================
# PRICE / SLTP
# =========================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def price_tick(x: float) -> float:
    ax = abs(x)
    if ax >= 1000:
        return 0.5
    if ax >= 100:
        return 0.05
    if ax >= 1:
        return 0.001
    if ax >= 0.1:
        return 0.0001
    return 0.00001


def quantize(x: float) -> float:
    t = price_tick(x)
    return round(x / t) * t


def sl_mult_from_conf(conf: int) -> float:
    if conf >= 90:
        return 2.4
    if conf >= 82:
        return 2.1
    if conf >= 78:
        return 1.9
    if conf >= 70:
        return 1.6
    return 1.4


def sl_tp_from_atr(entry: float, side: str, atr: float, conf: int) -> Tuple[float, float, float]:
    if entry <= 0 or atr <= 0:
        return 0.0, 0.0, 0.0

    sl_dist = sl_mult_from_conf(conf) * atr
    min_dist = (ATR_MIN_PCT / 100.0) * entry
    max_dist = (ATR_MAX_PCT / 100.0) * entry
    sl_dist = clamp(sl_dist, min_dist, max_dist)

    r = sl_dist
    tp_dist = 1.55 * r
    tp_cap = (TP_MAX_PCT / 100.0) * entry
    tp_dist = min(tp_dist, tp_cap)

    if side == "BUY":
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist

    sl = quantize(sl)
    tp = quantize(tp)
    return sl, tp, r


def multi_tp(entry: float, side: str, r: float) -> Tuple[float, float, float]:
    if r <= 0 or entry <= 0:
        return 0.0, 0.0, 0.0
    r1, r2, r3 = TP_R_MULTS

    cap = (TP_MAX_PCT / 100.0) * entry
    d3 = min(r3 * r, cap)
    d2 = min(r2 * r, d3 * 0.995)
    d1 = min(r1 * r, d2 * 0.995)

    tick = price_tick(entry)
    d1 = max(d1, tick)
    d2 = max(d2, d1 + tick)
    d3 = max(d3, d2 + tick)

    if side == "BUY":
        tp1 = entry + d1
        tp2 = entry + d2
        tp3 = entry + d3
    else:
        tp1 = entry - d1
        tp2 = entry - d2
        tp3 = entry - d3

    tp1, tp2, tp3 = quantize(tp1), quantize(tp2), quantize(tp3)

    # strict monotonic after quantize
    if side == "BUY":
        if tp1 >= tp2:
            tp1 = quantize(tp2 - tick)
        if tp2 >= tp3:
            tp2 = quantize(tp3 - tick)
    else:
        if tp1 <= tp2:
            tp1 = quantize(tp2 + tick)
        if tp2 <= tp3:
            tp2 = quantize(tp3 + tick)

    return tp1, tp2, tp3


# =========================
# FORMATTING
# =========================
def fmt_price(x: float) -> str:
    ax = abs(x)
    if ax >= 100:
        return f"{x:.2f}"
    if ax >= 1:
        return f"{x:.3f}"
    if ax >= 0.1:
        return f"{x:.4f}"
    return f"{x:.6f}"


def fmt_money(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:.0f}"


def pct_emo(p: float) -> str:
    v = int(round(p))
    if v >= 3:
        e = "🟢"
    elif v <= -3:
        e = "🔴"
    else:
        e = "🟡"
    return f"{v:+d}% {e}"


def tv_url(base: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol=BYBIT:{base.upper()}USDT.P"


def fmt_table(rows: List[List], headers: List[str]) -> str:
    return "```\n" + tabulate(rows, headers=headers, tablefmt="github") + "\n```"


# =========================
# CONFIDENCE / SETUPS
# =========================
def confidence(side: str, ch24: float, ch4: float, ch1: float, ch15: float, vol: float) -> int:
    s = 50.0
    is_long = (side == "BUY")

    def align(x: float, w: float):
        nonlocal s
        if is_long:
            s += w if x > 0 else -w
        else:
            s += w if x < 0 else -w

    align(ch24, 12.0)
    align(ch4, 10.0)
    align(ch1, 8.0)
    align(ch15, 6.0)

    mag = min(abs(ch24) / 2.0 + abs(ch4) + abs(ch1) * 2.0 + abs(ch15) * 2.0, 20.0)
    s += mag

    if vol >= 20_000_000:
        s += 9
    elif vol >= 10_000_000:
        s += 7
    elif vol >= 5_000_000:
        s += 5
    elif vol >= 2_000_000:
        s += 3

    return int(round(clamp(s, 0, 100)))


def make_setup(base: str, mv: MarketVol) -> Optional[Setup]:
    vol = usd_notional_mv(mv)
    if vol <= 0:
        return None

    ch24 = float(mv.percentage or 0.0)
    ch1, ch4, ch15, atr = metrics_1h_4h_15m(mv.symbol)

    if abs(ch1) < TRIGGER_1H_ABS_MIN:
        return None
    if abs(ch15) < CONFIRM_15M_ABS_MIN:
        return None

    side = "BUY" if ch1 > 0 else "SELL"

    # directional align filters (to avoid long in downtrend, etc.)
    if USE_4H_ALIGN:
        if side == "BUY" and ch4 < 0:
            return None
        if side == "SELL" and ch4 > 0:
            return None
    if USE_24H_ALIGN:
        if side == "BUY" and ch24 < 0:
            return None
        if side == "SELL" and ch24 > 0:
            return None

    entry = float(mv.last or 0.0)
    if entry <= 0:
        return None

    conf = confidence(side, ch24, ch4, ch1, ch15, vol)
    if conf < SIGNAL_MIN_CONF:
        return None

    sl, tp_single, r = sl_tp_from_atr(entry, side, atr, conf)
    if sl <= 0 or tp_single <= 0 or r <= 0:
        return None

    tp1 = tp2 = None
    tp3 = tp_single
    if conf >= MULTI_TP_MIN_CONF:
        tp1, tp2, tp3 = multi_tp(entry, side, r)

    return Setup(
        symbol=base,
        market_symbol=mv.symbol,
        side=side,
        conf=conf,
        entry=quantize(entry),
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        fut_vol_usd=vol,
        ch24=ch24,
        ch4=ch4,
        ch1=ch1,
        ch15=ch15,
        created_ts=time.time(),
    )


def pick_setups(best: Dict[str, MarketVol]) -> List[Setup]:
    universe = sorted(best.items(), key=lambda kv: usd_notional_mv(kv[1]), reverse=True)[:40]
    out: List[Setup] = []
    for base, mv in universe:
        s = make_setup(base, mv)
        if s:
            out.append(s)
    out.sort(key=lambda x: (x.conf, x.fut_vol_usd), reverse=True)
    return out[:SETUPS_N]


def directional_lists(best: Dict[str, MarketVol]) -> Tuple[List[Tuple], List[Tuple]]:
    up, dn = [], []
    for base, mv in best.items():
        vol = usd_notional_mv(mv)
        if vol < MOVER_VOL_USD_MIN:
            continue
        ch24 = float(mv.percentage or 0.0)
        if ch24 < MOVER_UP_24H_MIN and ch24 > MOVER_DN_24H_MAX:
            continue
        ch1, ch4, ch15, atr = metrics_1h_4h_15m(mv.symbol)

        if ch24 >= MOVER_UP_24H_MIN and ch4 >= MOVER_4H_ALIGN_MIN:
            up.append((base, vol, ch24, ch4, float(mv.last or 0.0)))
        if ch24 <= MOVER_DN_24H_MAX and ch4 <= -MOVER_4H_ALIGN_MIN:
            dn.append((base, vol, ch24, ch4, float(mv.last or 0.0)))

    up.sort(key=lambda x: (x[2], x[1]), reverse=True)
    dn.sort(key=lambda x: (x[2], x[1]))
    return up[:DIR_N], dn[:DIR_N]


def leaders_table(best: Dict[str, MarketVol]) -> str:
    leaders = sorted(best.items(), key=lambda kv: usd_notional_mv(kv[1]), reverse=True)[:LEADERS_N]
    rows = []
    for base, mv in leaders:
        rows.append([base, fmt_money(usd_notional_mv(mv)), pct_emo(float(mv.percentage or 0.0)), fmt_price(float(mv.last or 0.0))])
    return "*Market Leaders (Top by Futures Volume)*\n" + fmt_table(rows, ["SYM", "F Vol", "24H", "Last"])


def movers_tables(best: Dict[str, MarketVol]) -> Tuple[str, str]:
    up, dn = directional_lists(best)
    up_rows = [[b, fmt_money(v), pct_emo(c24), pct_emo(c4), fmt_price(px)] for b, v, c24, c4, px in up]
    dn_rows = [[b, fmt_money(v), pct_emo(c24), pct_emo(c4), fmt_price(px)] for b, v, c24, c4, px in dn]
    up_txt = "*Directional Leaders (24H ≥ +10%, 4H aligned)*\n" + (fmt_table(up_rows, ["SYM", "F Vol", "24H", "4H", "Last"]) if up_rows else "_None_")
    dn_txt = "*Directional Losers (24H ≤ -10%, 4H aligned)*\n" + (fmt_table(dn_rows, ["SYM", "F Vol", "24H", "4H", "Last"]) if dn_rows else "_None_")
    return up_txt, dn_txt


# =========================
# SESSIONS (PER USER TZ)
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


def active_session(user: dict) -> Optional[dict]:
    tz = ZoneInfo(user.get("tz") or DEFAULT_TZ)
    now = datetime.now(tz)

    for s in SESSIONS:
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
            session_key = f"{start_dt.strftime('%Y-%m-%d')}_{s['name']}"
            return {
                "name": s["name"],
                "start_dt": start_dt,
                "end_dt": end_dt,
                "start_str": s["start"],
                "end_str": s["end"],
                "session_key": session_key,
            }
    return None


# =========================
# EMAIL
# =========================
def email_ok() -> bool:
    return all([EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FROM, EMAIL_TO])


def send_email(subject: str, body: str) -> bool:
    if not email_ok():
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


def email_body(setups: List[Setup], dir_up: List[Tuple], dir_dn: List[Tuple], best: Dict[str, MarketVol], user: dict, sess: dict) -> Tuple[str, str]:
    tz = ZoneInfo(user.get("tz") or DEFAULT_TZ)
    now_local = datetime.now(tz).strftime("%a %d %b %Y • %H:%M")
    subject = f"PulseFutures • {sess['name']} • Best Signals ({now_local})"

    # Market snapshot: take top 5 leaders and compute "bias" simple (count up vs down)
    leaders = sorted(best.items(), key=lambda kv: usd_notional_mv(kv[1]), reverse=True)[:10]
    up_cnt = sum(1 for _, mv in leaders if float(mv.percentage or 0.0) > 0)
    dn_cnt = sum(1 for _, mv in leaders if float(mv.percentage or 0.0) < 0)
    if up_cnt >= 7:
        bias = "BULLISH 🟢"
    elif dn_cnt >= 7:
        bias = "BEARISH 🔴"
    else:
        bias = "MIXED 🟡"

    lines: List[str] = []
    lines.append(HDR)
    lines.append("PulseFutures — Email Signal Alert (Bybit Futures)")
    lines.append(f"Time: {now_local}  |  TZ: {user.get('tz')}")
    lines.append(f"Session: {sess['name']}  ({sess['start_str']} → {sess['end_str']})")
    lines.append(HDR)
    lines.append("")
    lines.append("MARKET SUMMARY")
    lines.append(SEP)
    lines.append(f"Bias (Top10 volume leaders): {bias}  (Up {up_cnt} / Down {dn_cnt})")
    lines.append("Top Leaders (by futures volume):")
    for base, mv in leaders[:5]:
        lines.append(f"- {base:<6} {pct_emo(float(mv.percentage or 0.0))} | F~{fmt_money(usd_notional_mv(mv))} | Last {fmt_price(float(mv.last or 0.0))}")
    lines.append("")
    lines.append("BEST SIGNALS (high probability)")
    lines.append(SEP)

    for i, s in enumerate(setups, 1):
        lines.append(f"{i}) {s.side} {s.symbol}  •  Confidence {s.conf}/100")
        lines.append(f"   Entry: {fmt_price(s.entry)}")
        lines.append(f"   SL:    {fmt_price(s.sl)}")
        if s.tp1 and s.tp2 and s.conf >= MULTI_TP_MIN_CONF:
            lines.append(f"   TP1:   {fmt_price(s.tp1)}   ({TP_ALLOCS[0]}%)")
            lines.append(f"   TP2:   {fmt_price(s.tp2)}   ({TP_ALLOCS[1]}%)")
            lines.append(f"   TP3:   {fmt_price(s.tp3)}   ({TP_ALLOCS[2]}%)")
            lines.append("   Rule: After TP1 → move SL to BreakEven")
        else:
            lines.append(f"   TP:    {fmt_price(s.tp3)}")
        lines.append(f"   24H {pct_emo(s.ch24)} | 4H {pct_emo(s.ch4)} | 1H {pct_emo(s.ch1)} | 15m {pct_emo(s.ch15)} | F~{fmt_money(s.fut_vol_usd)}")
        lines.append(f"   Chart: {tv_url(s.symbol)}")
        lines.append("")

    lines.append("DIRECTIONAL MOVERS (quick glance)")
    lines.append(SEP)
    lines.append("Leaders (24H strong up & 4H aligned):")
    if dir_up:
        for b, v, c24, c4, px in dir_up[:5]:
            lines.append(f"- {b:<6} {pct_emo(c24)} | 4H {pct_emo(c4)} | F~{fmt_money(v)}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Losers (24H strong down & 4H aligned):")
    if dir_dn:
        for b, v, c24, c4, px in dir_dn[:5]:
            lines.append(f"- {b:<6} {pct_emo(c24)} | 4H {pct_emo(c4)} | F~{fmt_money(v)}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append(HDR)
    lines.append("Note: Not financial advice. Use your own risk rules.")
    lines.append(HDR)

    return subject, "\n".join(lines)


# =========================
# RISK SETTINGS
# =========================
def daily_cap_usd(user: dict) -> float:
    mode = (user.get("daily_cap_mode") or "USD").upper()
    val = float(user.get("daily_cap_value") or 0.0)
    equity = float(user.get("equity") or 0.0)
    if mode == "USD":
        return max(0.0, val)
    if equity <= 0:
        return 0.0
    return max(0.0, equity * (val / 100.0))


def risk_per_trade_usd(user: dict) -> float:
    mode = (user.get("risk_mode") or "USD").upper()
    val = float(user.get("risk_value") or 0.0)
    equity = float(user.get("equity") or 0.0)
    if mode == "USD":
        return max(0.0, val)
    if equity <= 0:
        return 0.0
    return max(0.0, equity * (val / 100.0))


def position_size(entry: float, sl: float, risk_usd: float) -> float:
    rpu = abs(entry - sl)
    if rpu <= 0:
        return 0.0
    return risk_usd / rpu


# =========================
# TELEGRAM TEXT
# =========================
HELP_TEXT = """\
PulseFutures — International Day Trader Bot

Market
- /screen

Timezone (IMPORTANT for sessions)
- /tz America/New_York
- /tz Europe/London
- /tz Asia/Singapore
(Any IANA timezone)

Risk Settings
- /equity 1500
- /riskmode usd 40
- /riskmode pct 2
- /dailycap usd 200
- /dailycap pct 6
- /limits 5

Trade Setup (suggestion only)
- /tradesetup BTC

Journal (log your real trades)
- /trade_open BTC buy 65000 64000 67500 40
- /trade_close BTC +120
- /status

Email
- /notify_on /notify_off

Notes
- Emails only inside your active session window (based on your timezone).
- 3–4 emails per session with >= 60 min gap.
- No repeated symbol inside the same session.
"""


# =========================
# TELEGRAM HANDLERS
# =========================
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)


async def tz_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args:
        user = get_user(uid)
        await update.message.reply_text(f"Your timezone: {user.get('tz')}\nSet: /tz America/New_York")
        return
    tzname = " ".join(context.args).strip()
    try:
        ZoneInfo(tzname)
    except Exception:
        await update.message.reply_text("Invalid timezone. Example: /tz America/New_York")
        return
    update_user(uid, tz=tzname)
    await update.message.reply_text(f"✅ Timezone set: {tzname}")


async def notify_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    update_user(uid, notify_on=1)
    await update.message.reply_text("✅ Email alerts: ON")


async def notify_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    update_user(uid, notify_on=0)
    await update.message.reply_text("🛑 Email alerts: OFF")


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
        await update.message.reply_text("Usage: /equity 1500")


async def riskmode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if not context.args:
        await update.message.reply_text(f"Risk mode: {user['risk_mode']} {float(user['risk_value']):.2f}\nSet: /riskmode usd 40  OR  /riskmode pct 2")
        return
    if len(context.args) != 2:
        await update.message.reply_text("Usage: /riskmode usd 40  OR  /riskmode pct 2")
        return
    mode = context.args[0].strip().upper()
    try:
        val = float(context.args[1])
    except Exception:
        await update.message.reply_text("Usage: /riskmode usd 40  OR  /riskmode pct 2")
        return
    if mode not in {"USD", "PCT"}:
        await update.message.reply_text("Mode must be usd or pct.")
        return
    if val <= 0:
        await update.message.reply_text("Value must be > 0")
        return
    update_user(uid, risk_mode=mode, risk_value=val)
    await update.message.reply_text(f"✅ Risk per trade: {mode} {val:.2f}")


async def dailycap_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if not context.args:
        cap = daily_cap_usd(user)
        await update.message.reply_text(f"Daily cap: {user['daily_cap_mode']} {float(user['daily_cap_value']):.2f} (≈ ${cap:.2f})\nSet: /dailycap usd 200  OR  /dailycap pct 6")
        return
    if len(context.args) != 2:
        await update.message.reply_text("Usage: /dailycap usd 200  OR  /dailycap pct 6")
        return
    mode = context.args[0].strip().upper()
    try:
        val = float(context.args[1])
    except Exception:
        await update.message.reply_text("Usage: /dailycap usd 200  OR  /dailycap pct 6")
        return
    if mode not in {"USD", "PCT"}:
        await update.message.reply_text("Mode must be usd or pct.")
        return
    if val < 0:
        await update.message.reply_text("Value must be >= 0")
        return
    update_user(uid, daily_cap_mode=mode, daily_cap_value=val)
    user = get_user(uid)
    cap = daily_cap_usd(user)
    await update.message.reply_text(f"✅ Daily cap set: {mode} {val:.2f} (≈ ${cap:.2f})")


async def limits_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if not context.args:
        await update.message.reply_text(f"Max trades/day: {int(user['max_trades_day'])}\nSet: /limits 5")
        return
    try:
        n = int(context.args[0])
        if n < 1 or n > 30:
            raise ValueError()
        update_user(uid, max_trades_day=n)
        await update.message.reply_text(f"✅ Max trades/day set: {n}")
    except Exception:
        await update.message.reply_text("Usage: /limits 5")


async def screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    best = await asyncio.to_thread(fetch_futures_tickers)
    if not best:
        await update.message.reply_text("❌ Could not fetch futures tickers.")
        return

    setups = await asyncio.to_thread(pick_setups, best)
    for s in setups:
        db_upsert_signal(s)

    up_txt, dn_txt = await asyncio.to_thread(movers_tables, best)
    leaders_txt = await asyncio.to_thread(leaders_table, best)

    parts = []
    parts.append("✨ *PulseFutures — Market Scan*")
    parts.append("")
    parts.append("— *Top Trade Setups (High confidence)*")
    if not setups:
        parts.append("_No high-quality setup now (filters are strict)._")
    else:
        for i, s in enumerate(setups, 1):
            if s.tp1 and s.tp2 and s.conf >= MULTI_TP_MIN_CONF:
                tp_line = f"TP1 {fmt_price(s.tp1)} | TP2 {fmt_price(s.tp2)} | TP3 {fmt_price(s.tp3)}"
            else:
                tp_line = f"TP {fmt_price(s.tp3)}"
            parts.append(
                f"🔥 *#{i}*  *{s.side} {s.symbol}*  •  Conf *{s.conf}/100*\n"
                f"Entry {fmt_price(s.entry)} | SL {fmt_price(s.sl)}\n"
                f"{tp_line}\n"
                f"24H {pct_emo(s.ch24)} | 4H {pct_emo(s.ch4)} | 1H {pct_emo(s.ch1)} | 15m {pct_emo(s.ch15)} | F~{fmt_money(s.fut_vol_usd)}\n"
                f"📈 {tv_url(s.symbol)}"
            )
            parts.append("")

    parts.append("— *Directional Leaders / Losers*")
    parts.append(up_txt)
    parts.append("")
    parts.append(dn_txt)
    parts.append("")
    parts.append("— *Market Leaders (Volume)*")
    parts.append(leaders_txt)

    msg = "\n".join(parts).strip()

    keyboard = []
    for s in setups:
        keyboard.append([InlineKeyboardButton(text=f"📈 {s.symbol} Chart", url=tv_url(s.symbol))])

    await update.message.reply_text(
        msg,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None,
        disable_web_page_preview=True,
    )


async def tradesetup_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = reset_daily_if_needed(get_user(uid))

    if not context.args:
        await update.message.reply_text("Usage: /tradesetup BTC")
        return

    sym = re.sub(r"[^A-Za-z0-9]", "", context.args[0]).upper().lstrip("$")
    best = await asyncio.to_thread(fetch_futures_tickers)
    mv = best.get(sym)
    if not mv:
        await update.message.reply_text(f"❌ {sym} not found on Bybit futures.")
        return

    ch24 = float(mv.percentage or 0.0)
    ch1, ch4, ch15, atr = await asyncio.to_thread(metrics_1h_4h_15m, mv.symbol)
    side = "BUY" if ch1 > 0 else "SELL"
    vol = usd_notional_mv(mv)
    conf = confidence(side, ch24, ch4, ch1, ch15, vol)

    sl, tp_single, r = sl_tp_from_atr(float(mv.last or 0.0), side, atr, conf)
    if sl <= 0:
        await update.message.reply_text("❌ Not enough data to compute ATR SL/TP.")
        return

    entry = quantize(float(mv.last or 0.0))
    tp1 = tp2 = None
    tp3 = tp_single
    if conf >= MULTI_TP_MIN_CONF:
        tp1, tp2, tp3 = multi_tp(entry, side, r)

    risk_usd = risk_per_trade_usd(user)
    if risk_usd <= 0:
        await update.message.reply_text("⚠️ Your risk per trade is 0.\nSet it: /riskmode usd 40\n(Equity optional if using USD mode)")
        return

    qty = position_size(entry, sl, risk_usd)
    if qty <= 0:
        await update.message.reply_text("❌ Invalid sizing (entry and SL too close).")
        return

    if tp1 and tp2 and conf >= MULTI_TP_MIN_CONF:
        tp_line = f"TP1 {fmt_price(tp1)} ({TP_ALLOCS[0]}%) | TP2 {fmt_price(tp2)} ({TP_ALLOCS[1]}%) | TP3 {fmt_price(tp3)} ({TP_ALLOCS[2]}%)"
        rule = "Rule: after TP1 → move SL to BE"
    else:
        tp_line = f"TP {fmt_price(tp3)}"
        rule = ""

    msg = (
        f"✅ *TradeSetup (Suggestion)*\n"
        f"*{side} {sym}*  •  Conf *{conf}/100*\n"
        f"Entry {fmt_price(entry)} | SL {fmt_price(sl)}\n"
        f"{tp_line}\n"
        f"Risk/trade: *${risk_usd:.2f}* → Suggested Qty: *{qty:.6g}*\n"
        f"24H {pct_emo(ch24)} | 4H {pct_emo(ch4)} | 1H {pct_emo(ch1)} | 15m {pct_emo(ch15)} | F~{fmt_money(vol)}\n"
        f"📈 {tv_url(sym)}"
    )
    if rule:
        msg += f"\n🧠 {rule}"

    kb = InlineKeyboardMarkup([[InlineKeyboardButton("📈 Chart (TV)", url=tv_url(sym))]])
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=kb, disable_web_page_preview=True)


async def trade_open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /trade_open BTC buy entry sl tp risk_usd
    uid = update.effective_user.id
    user = reset_daily_if_needed(get_user(uid))

    if len(context.args) < 6:
        await update.message.reply_text("Usage: /trade_open BTC buy entry sl tp risk_usd\nExample: /trade_open BTC buy 65000 64000 67500 40")
        return

    sym = re.sub(r"[^A-Za-z0-9]", "", context.args[0]).upper().lstrip("$")
    side_raw = context.args[1].strip().upper()
    side = "BUY" if side_raw in {"BUY", "LONG"} else "SELL" if side_raw in {"SELL", "SHORT"} else ""
    if not side:
        await update.message.reply_text("Side must be buy/sell (or long/short).")
        return

    try:
        entry = float(context.args[2])
        sl = float(context.args[3])
        tp = float(context.args[4])
        risk_usd = float(context.args[5])
    except Exception:
        await update.message.reply_text("Bad numbers. Example: /trade_open BTC buy 65000 64000 67500 40")
        return
    if risk_usd <= 0:
        await update.message.reply_text("risk_usd must be > 0")
        return

    max_trades = int(user.get("max_trades_day") or DEFAULT_MAX_TRADES_DAY)
    if int(user.get("day_trade_count") or 0) >= max_trades:
        await update.message.reply_text(f"⚠️ Max trades/day reached ({max_trades}).")
        return

    cap = daily_cap_usd(user)
    used = float(user.get("daily_risk_used") or 0.0)
    remaining = cap - used if cap > 0 else 0.0

    journal_open_trade(uid, sym, side, entry, sl, tp, risk_usd)
    update_user(uid,
                day_trade_count=int(user.get("day_trade_count") or 0) + 1,
                daily_risk_used=float(user.get("daily_risk_used") or 0.0) + risk_usd)

    warn = ""
    if cap > 0 and risk_usd > remaining:
        warn = f"\n⚠️ You exceeded daily remaining risk (${remaining:.2f})."

    await update.message.reply_text(
        f"✅ Trade OPEN logged: {sym} {side}\nEntry {fmt_price(entry)} | SL {fmt_price(sl)} | TP {fmt_price(tp)}\nRisk ${risk_usd:.2f}{warn}"
    )


async def trade_close_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /trade_close BTC +120
    uid = update.effective_user.id
    user = reset_daily_if_needed(get_user(uid))

    if len(context.args) < 2:
        await update.message.reply_text("Usage: /trade_close BTC +120")
        return
    sym = re.sub(r"[^A-Za-z0-9]", "", context.args[0]).upper().lstrip("$")
    try:
        pnl = float(context.args[1])
    except Exception:
        await update.message.reply_text("Usage: /trade_close BTC +120")
        return

    t = journal_close_trade(uid, sym, exit_price=None, pnl=pnl)
    if not t:
        await update.message.reply_text(f"❌ No OPEN trade found for {sym}.")
        return

    eq = float(user.get("equity") or 0.0)
    eq_line = ""
    if eq > 0:
        new_eq = eq + pnl
        update_user(uid, equity=new_eq)
        eq_line = f"\nEquity updated: ${new_eq:.2f}"

    await update.message.reply_text(f"✅ Trade CLOSED: {sym} {t['side']}\nPnL: {pnl:+.2f}{eq_line}")


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = reset_daily_if_needed(get_user(uid))

    cap = daily_cap_usd(user)
    used = float(user.get("daily_risk_used") or 0.0)
    remaining = max(0.0, cap - used) if cap > 0 else 0.0

    open_trades = journal_open_trades(uid)
    max_trades = int(user.get("max_trades_day") or DEFAULT_MAX_TRADES_DAY)
    trade_count = int(user.get("day_trade_count") or 0)

    sess = active_session(user)
    sess_line = f"{sess['name']} ({sess['start_str']}-{sess['end_str']}) ✅" if sess else "Outside sessions ⛔"

    lines = []
    lines.append("📌 *Status — Risk & Journal*")
    lines.append(f"Timezone: *{user.get('tz')}*")
    lines.append(f"Active session now: *{sess_line}*")
    lines.append("")
    lines.append(f"Equity: *${float(user.get('equity') or 0.0):.2f}*")
    lines.append(f"Risk/trade: *{user['risk_mode']} {float(user['risk_value']):.2f}*  (≈ ${risk_per_trade_usd(user):.2f})")
    lines.append(f"Daily cap: *{user['daily_cap_mode']} {float(user['daily_cap_value']):.2f}*  (≈ ${cap:.2f})")
    lines.append(f"Daily risk used: *${used:.2f}* | Remaining: *${remaining:.2f}*")
    lines.append(f"Trades today: *{trade_count}/{max_trades}*")
    if trade_count >= max_trades:
        lines.append("⚠️ You reached your daily trade limit.")

    lines.append("")
    lines.append("— *Open Trades (Journal)*")
    if not open_trades:
        lines.append("_None_")
    else:
        for i, t in enumerate(open_trades, 1):
            lines.append(
                f"#{i} *{t['symbol']}* {t['side']} | Entry {fmt_price(float(t['entry']))}"
                f" | SL {fmt_price(float(t['sl'])) if t['sl'] is not None else '-'}"
                f" | TP {fmt_price(float(t['tp'])) if t['tp'] is not None else '-'}"
                f" | Risk ${float(t['risk_usd']):.2f}"
            )

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (update.message.text or "").strip()
    token = re.sub(r"[^A-Za-z0-9$]", "", txt).upper().lstrip("$")
    if len(token) < 2:
        return
    sig = db_get_signal(token)
    if not sig:
        await update.message.reply_text(f"{token} not in current signals.\nTry: /tradesetup {token}")
        return

    tp1 = sig.get("tp1")
    tp2 = sig.get("tp2")
    if tp1 and tp2 and int(sig.get("conf") or 0) >= MULTI_TP_MIN_CONF:
        tp_line = f"TP1 {fmt_price(float(tp1))} | TP2 {fmt_price(float(tp2))} | TP3 {fmt_price(float(sig['tp3']))}"
    else:
        tp_line = f"TP {fmt_price(float(sig['tp3']))}"

    msg = (
        f"🔎 *{token}* (cached signal)\n"
        f"*{sig['side']}* — Conf *{sig['conf']}/100*\n"
        f"Entry {fmt_price(float(sig['entry']))} | SL {fmt_price(float(sig['sl']))}\n"
        f"{tp_line}\n"
        f"24H {pct_emo(float(sig['ch24']))} | 4H {pct_emo(float(sig['ch4']))} | F~{fmt_money(float(sig['fut_vol_usd']))}\n"
        f"📈 {tv_url(token)}\n\n"
        f"TradeSetup: /tradesetup {token}"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)


# =========================
# EMAIL JOB
# =========================
def list_notify_users() -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE notify_on=1")
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


async def alert_job(context: ContextTypes.DEFAULT_TYPE):
    if ALERT_LOCK.locked():
        return

    async with ALERT_LOCK:
        try:
            if not EMAIL_ENABLED:
                return
            if not email_ok():
                logger.warning("EMAIL_ENABLED=true but missing env (EMAIL_HOST/USER/PASS/etc).")
                return

            users = list_notify_users()
            if not users:
                return

            best = await asyncio.to_thread(fetch_futures_tickers)
            if not best:
                return

            setups = await asyncio.to_thread(pick_setups, best)
            for s in setups:
                db_upsert_signal(s)

            dir_up, dir_dn = await asyncio.to_thread(directional_lists, best)

            for user in users:
                user = reset_daily_if_needed(user)
                uid = int(user["user_id"])

                sess = active_session(user)
                if not sess:
                    continue

                st = email_state_get(uid)
                if st["session_key"] != sess["session_key"]:
                    # new session => reset counts
                    email_state_set(uid, session_key=sess["session_key"], sent_count=0, last_email_ts=0.0)
                    st = email_state_get(uid)

                sent_count = int(st["sent_count"])
                if sent_count >= MAX_EMAILS_PER_SESSION:
                    continue

                gap_sec = DEFAULT_EMAIL_GAP_MIN * 60
                if gap_sec > 0 and (time.time() - float(st["last_email_ts"] or 0.0)) < gap_sec:
                    continue

                # filter: no repeated symbol in session + optional global cooldown
                session_key = sess["session_key"]
                filtered: List[Setup] = []
                for s in setups:
                    if session_symbol_sent(uid, session_key, s.symbol):
                        continue
                    if emailed_recently_global(uid, s.symbol, SYMBOL_EMAIL_COOLDOWN_SEC):
                        continue
                    filtered.append(s)

                if not filtered:
                    continue

                filtered = filtered[:SETUPS_N]
                subject, body = email_body(filtered, dir_up, dir_dn, best, user, sess)

                ok = await asyncio.to_thread(send_email, subject, body)
                if ok:
                    email_state_set(uid, sent_count=sent_count + 1, last_email_ts=time.time())
                    mark_session_symbols(uid, session_key, [s.symbol for s in filtered])
                    mark_global_symbols(uid, [s.symbol for s in filtered])
                    logger.info("Email sent user_id=%s session=%s (%s/%s)", uid, sess["name"], sent_count + 1, MAX_EMAILS_PER_SESSION)

        except Exception as e:
            logger.exception("alert_job error: %s", e)


# =========================
# MAIN
# =========================
def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN missing")

    db_init()

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler(["help", "start"], cmd_help))
    app.add_handler(CommandHandler("tz", tz_cmd))
    app.add_handler(CommandHandler("screen", screen))

    app.add_handler(CommandHandler("notify_on", notify_on))
    app.add_handler(CommandHandler("notify_off", notify_off))

    app.add_handler(CommandHandler("equity", equity_cmd))
    app.add_handler(CommandHandler("riskmode", riskmode_cmd))
    app.add_handler(CommandHandler("dailycap", dailycap_cmd))
    app.add_handler(CommandHandler("limits", limits_cmd))

    app.add_handler(CommandHandler("tradesetup", tradesetup_cmd))
    app.add_handler(CommandHandler("trade_open", trade_open_cmd))
    app.add_handler(CommandHandler("trade_close", trade_close_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    if getattr(app, "job_queue", None):
        app.job_queue.run_repeating(alert_job, interval=CHECK_INTERVAL_MIN * 60, first=10, name="alert_job")
    else:
        logger.warning('JobQueue not available. Install "python-telegram-bot[job-queue]>=20.7,<22.0"')

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
