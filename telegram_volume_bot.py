#!/usr/bin/env python3
"""
Telegram Crypto Signals + Risk Ledger Bot (CoinEx Futures via CCXT)

Main UX (/screen):
- 🔥 Market Leaders (Liquidity + Momentum) + Bias hint
- 🚀 Strong Movers (24h)
- 🎯 Top 3 Trade Setups

Core principles:
- Futures-only (swap)
- Universe Filter (no P1/P2/P3)
- Trading Window Guard (default London+NY in UTC), with /session control
- Risk Ledger: /equity, /limits, /risk, /open, /closepnl, /status
- Multi-TP for high-confidence setups (Conf >= 75): TP1/TP2 + Runner suggestion
- Optional email alerts (/notify_on /notify_off) + scheduled job via JobQueue

Required env vars:
- TELEGRAM_TOKEN

Optional email env vars:
- EMAIL_ENABLED=true/false
- EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FROM, EMAIL_TO

Install:
- python-telegram-bot[job-queue]>=20.7,<22.0
- ccxt>=4.0.0
- tabulate>=0.9.0
"""

import asyncio
import logging
import math
import os
import re
import ssl
import smtplib
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Dict, List, Tuple, Optional, Set

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

# =========================================================
# CONFIG
# =========================================================

EXCHANGE_ID = "coinex"
TOKEN = os.environ.get("TELEGRAM_TOKEN")

# Universe Filter (Futures only)
UNIVERSE_MIN_FUT_USD = 2_000_000
LEADERS_MIN_FUT_USD = 5_000_000
LEADERS_TOP_N = 7
MOVERS_TOP_N = 10
SETUPS_TOP_N = 3

# Multi-TP for high-confidence
CONF_MULTI_TP_MIN = 75

# Trading Sessions (UTC)
SESSION_LONDON = (7, 16)   # 07:00–16:00 UTC
SESSION_NY = (13, 22)      # 13:00–22:00 UTC
DEFAULT_SESSION_MODE = "both"  # both | london | ny | off

# Scheduler
CHECK_INTERVAL_MIN = 5

# Email config
EMAIL_ENABLED_DEFAULT = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST = os.environ.get("EMAIL_HOST", "")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "465"))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", EMAIL_USER)
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_USER)

EMAIL_MIN_INTERVAL_SEC = 15 * 60
LAST_EMAIL_TS: float = 0.0
LAST_EMAIL_KEYS: Set[str] = set()

# Globals
LAST_ERROR: Optional[str] = None
NOTIFY_ON: bool = EMAIL_ENABLED_DEFAULT

# Cache for OHLC % changes (keep small)
PCT_CACHE: Dict[Tuple[str, str, str], float] = {}  # (dtype, symbol, tf) -> pct

# DB path
DB_PATH = os.environ.get("DB_PATH", "bot.db")

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


# =========================================================
# DB HELPERS
# =========================================================

def db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

def db_init():
    with db_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
          key TEXT PRIMARY KEY,
          value TEXT
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts INTEGER NOT NULL,
          symbol TEXT NOT NULL,
          side TEXT NOT NULL,
          entry REAL,
          sl REAL,
          tp REAL,
          confidence INTEGER,
          score REAL,
          status TEXT NOT NULL,          -- SIGNAL | OPEN | CLOSED
          risk_usd REAL,
          opened_ts INTEGER,
          closed_ts INTEGER,
          close_result TEXT,
          pnl_usd REAL
        );
        """)
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_signals_symbol_status_ts
        ON signals(symbol, status, ts DESC);
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS executions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts INTEGER NOT NULL,
          symbol TEXT NOT NULL,
          side TEXT NOT NULL,
          risk_usd REAL NOT NULL
        );
        """)
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_exec_ts
        ON executions(ts DESC);
        """)
        conn.commit()

def db_get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    with db_conn() as conn:
        r = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return (r["value"] if r else default)

def db_set_setting(key: str, value: str):
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value))
        )
        conn.commit()

def get_equity() -> Optional[float]:
    v = db_get_setting("equity")
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None

def set_equity(x: float):
    db_set_setting("equity", str(float(x)))

def get_limits() -> Tuple[int, float, float]:
    """
    (max_trades_per_day, max_daily_risk_usd, max_open_risk_usd)
    """
    mt = db_get_setting("max_trades_day", "5")
    md = db_get_setting("max_daily_risk_usd", "200")
    mo = db_get_setting("max_open_risk_usd", "300")
    try:
        return int(mt), float(md), float(mo)
    except Exception:
        return 5, 200.0, 300.0

def set_limits(max_trades: int, max_daily: float, max_open: float):
    db_set_setting("max_trades_day", str(int(max_trades)))
    db_set_setting("max_daily_risk_usd", str(float(max_daily)))
    db_set_setting("max_open_risk_usd", str(float(max_open)))

def mel_day_start_ts() -> int:
    """
    Start of 'today' in Australia/Melbourne as unix ts.
    """
    tz = ZoneInfo("Australia/Melbourne")
    now = datetime.now(tz)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(start.timestamp())

def db_daily_trade_count(day_start_ts: int) -> int:
    with db_conn() as conn:
        r = conn.execute("SELECT COUNT(*) AS c FROM executions WHERE ts>=?", (day_start_ts,)).fetchone()
        return int(r["c"] or 0)

def db_daily_used_risk(day_start_ts: int) -> float:
    with db_conn() as conn:
        r = conn.execute("SELECT COALESCE(SUM(risk_usd),0) AS s FROM executions WHERE ts>=?", (day_start_ts,)).fetchone()
        return float(r["s"] or 0.0)

def db_open_used_risk() -> float:
    with db_conn() as conn:
        r = conn.execute("SELECT COALESCE(SUM(risk_usd),0) AS s FROM signals WHERE status='OPEN'", ()).fetchone()
        return float(r["s"] or 0.0)

def db_open_count() -> int:
    with db_conn() as conn:
        r = conn.execute("SELECT COUNT(*) AS c FROM signals WHERE status='OPEN'", ()).fetchone()
        return int(r["c"] or 0)

def db_apply_pnl_to_equity(pnl_usd: float) -> float:
    with db_conn() as conn:
        r = conn.execute("SELECT value FROM settings WHERE key='equity'").fetchone()
        if not r or r["value"] is None:
            raise RuntimeError("Equity not set")
        eq = float(r["value"])
        new_eq = eq + float(pnl_usd)
        conn.execute(
            "INSERT INTO settings(key,value) VALUES('equity',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(new_eq),)
        )
        conn.commit()
        return new_eq


# =========================================================
# SESSION MODE (Trading Window Guard)
# =========================================================

def get_session_mode() -> str:
    return db_get_setting("session_mode", DEFAULT_SESSION_MODE) or DEFAULT_SESSION_MODE

def set_session_mode(mode: str):
    mode = (mode or "").strip().lower()
    if mode not in ("both", "london", "ny", "off"):
        raise ValueError("invalid session mode")
    db_set_setting("session_mode", mode)

def is_in_trading_window(session_mode: str) -> bool:
    """
    session_mode: both | london | ny | off
    all in UTC
    """
    if session_mode == "off":
        return True

    now_utc = datetime.utcnow()
    h = now_utc.hour + now_utc.minute / 60.0

    def in_range(start, end):
        return start <= h < end

    in_london = in_range(*SESSION_LONDON)
    in_ny = in_range(*SESSION_NY)

    if session_mode == "london":
        return in_london
    if session_mode == "ny":
        return in_ny
    return in_london or in_ny  # both

def trading_window_warning() -> Optional[str]:
    mode = get_session_mode()
    if mode == "off":
        return None
    if not is_in_trading_window(mode):
        return "⚠️ Outside optimal trading window (London + New York)\nSignals may be limited or skipped.\n\n"
    return None


# =========================================================
# EXCHANGE HELPERS (Futures-only)
# =========================================================

def build_exchange_swap():
    klass = ccxt.__dict__[EXCHANGE_ID]
    return klass({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap"},
    })

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
    if not mv:
        return 0.0
    price = mv.vwap if mv.vwap else mv.last
    if not price or not mv.base_vol:
        return 0.0
    return mv.base_vol * price

def pct_change_24h(mv_spot: Optional[MarketVol], mv_fut: Optional[MarketVol]) -> float:
    # futures only here; keep signature for compatibility
    mv = mv_fut or mv_spot
    if not mv:
        return 0.0
    if mv.percentage:
        return float(mv.percentage)
    if mv.open:
        return (mv.last - mv.open) / mv.open * 100.0
    return 0.0

def safe_fetch_tickers_swap(ex):
    try:
        ex.load_markets()
        return ex.fetch_tickers()
    except Exception as e:
        global LAST_ERROR
        LAST_ERROR = f"{type(e).__name__}: {e}"
        logging.exception("fetch_tickers failed")
        return {}

def load_best_futures():
    """
    Load best futures market per BASE from CoinEx swap.
    Returns best_fut dict + raw count.
    """
    ex = build_exchange_swap()
    fut_tickers = safe_fetch_tickers_swap(ex)

    best_fut: Dict[str, MarketVol] = {}
    for t in fut_tickers.values():
        mv = to_mv(t)
        if not mv:
            continue
        if mv.base not in best_fut or usd_notional(mv) > usd_notional(best_fut[mv.base]):
            best_fut[mv.base] = mv

    return best_fut, len(fut_tickers)

def compute_pct_for_symbol(symbol: str, hours: int, prefer_swap: bool = True) -> float:
    """
    % change over last N hours using 1h candles.
    """
    dtype = "swap"
    tf_key = f"{hours}h"
    cache_key = (dtype, symbol, tf_key)
    if cache_key in PCT_CACHE:
        return PCT_CACHE[cache_key]

    try:
        ex = build_exchange_swap()
        ex.load_markets()
        candles = ex.fetch_ohlcv(symbol, timeframe="1h", limit=hours + 1)
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
        logging.exception("compute_pct_for_symbol failed for %s (%dh)", symbol, hours)
        PCT_CACHE[cache_key] = 0.0
        return 0.0


# =========================================================
# UNIVERSE / LEADERS / MOVERS / SETUPS
# =========================================================

def build_universe(best_fut: Dict[str, MarketVol]) -> List[List]:
    """
    Futures-only eligible universe
    row: [BASE, fut_usd, pct24, pct4, pct1, last_price, fut_symbol]
    """
    rows: List[List] = []
    for base, f in best_fut.items():
        fut_usd = usd_notional(f)
        if fut_usd < UNIVERSE_MIN_FUT_USD:
            continue

        pct24 = pct_change_24h(None, f)
        pct4 = compute_pct_for_symbol(f.symbol, 4, prefer_swap=True)
        pct1 = compute_pct_for_symbol(f.symbol, 1, prefer_swap=True)

        last_price = f.last or 0.0
        rows.append([base, fut_usd, pct24, pct4, pct1, last_price, f.symbol])

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows

def side_hint(pct4: float, pct1: float) -> str:
    if pct1 >= 2.0 and pct4 >= 0.0:
        return "LONG 🟢"
    if pct1 <= -2.0 and pct4 <= 0.0:
        return "SHORT 🔴"
    return "NEUTRAL 🟡"

def leader_rank_score_u(row: List) -> float:
    _, fut_usd, pct24, pct4, pct1, *_ = row
    liq = math.log10(max(fut_usd, 1.0))
    mom = (abs(pct1) * 1.2) + (abs(pct4) * 1.0) + (abs(pct24) * 0.4)
    return liq + mom

def build_market_leaders_u(universe: List[List]) -> List[List]:
    leaders = [r for r in universe if r[1] >= LEADERS_MIN_FUT_USD]
    leaders.sort(key=leader_rank_score_u, reverse=True)
    return leaders[:LEADERS_TOP_N]

def scan_strong_movers_fut(best_fut: Dict[str, MarketVol]) -> List[Tuple[str, float, float]]:
    out = []
    for base, f in best_fut.items():
        fut_usd = usd_notional(f)
        if fut_usd < 1_000_000:
            continue
        pct24 = pct_change_24h(None, f)
        if pct24 > 10.0:
            out.append((base, fut_usd, pct24))
    out.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return out[:MOVERS_TOP_N]

def score_long_u(row: List) -> float:
    _, fut_usd, pct24, pct4, pct1, *_ = row
    score = 0.0
    if pct24 > 0: score += min(pct24 / 5.0, 3.0)
    if pct4 > 0:  score += min(pct4 / 2.0, 3.0)
    if pct1 > 0:  score += min(pct1 / 1.0, 2.0)
    if fut_usd > 10_000_000: score += 2.0
    elif fut_usd > 5_000_000: score += 1.0
    return max(score, 0.0)

def score_short_u(row: List) -> float:
    _, fut_usd, pct24, pct4, pct1, *_ = row
    score = 0.0
    if pct24 < 0: score += min(abs(pct24) / 5.0, 3.0)
    if pct4 < 0:  score += min(abs(pct4) / 2.0, 3.0)
    if pct1 < 0:  score += min(abs(pct1) / 1.0, 2.0)
    if fut_usd > 10_000_000: score += 2.0
    elif fut_usd > 5_000_000: score += 1.0
    return max(score, 0.0)

def classify_row_u(row: List) -> Tuple[bool, bool]:
    _, _, pct24, pct4, pct1, *_ = row
    long_ok = short_ok = False

    if pct24 > 5.0:
        long_ok = True
    elif pct24 < -5.0:
        short_ok = True
    else:
        if pct1 >= 2.0 and pct4 >= 0.0:
            long_ok = True
        if pct1 <= -2.0 and pct4 <= 0.0:
            short_ok = True

    if long_ok and short_ok:
        if pct24 > 0:
            short_ok = False
        elif pct24 < 0:
            long_ok = False
        else:
            short_ok = False if pct1 >= 0 else short_ok

    return long_ok, short_ok

def confidence_from_score(score: float) -> int:
    """
    Map internal score (~0..10) to 0..100
    """
    s = max(0.0, min(score, 10.0))
    return int(round((s / 10.0) * 100))

def multi_tp_plan(entry: float, sl: float, side: str, confidence: int):
    """
    If confidence >= 75:
      TP1 = 1R, TP2 = 2R, Runner suggestion
    else:
      None
    """
    if confidence < CONF_MULTI_TP_MIN:
        return None

    try:
        entry = float(entry); sl = float(sl)
    except Exception:
        return None

    R = abs(entry - sl)
    if R <= 0:
        return None

    side = (side or "").upper()
    if side == "BUY":
        tp1 = entry + 1.0 * R
        tp2 = entry + 2.0 * R
        runner = "Runner (20%) — trail EMA20 (1H) or last 15m swing-low"
    else:
        tp1 = entry - 1.0 * R
        tp2 = entry - 2.0 * R
        runner = "Runner (20%) — trail EMA20 (1H) or last 15m swing-high"

    return {
        "R": R,
        "tp1": tp1,
        "tp2": tp2,
        "runner_text": runner,
        "weights": "TP1 40% | TP2 40% | Runner 20%",
    }

def pick_best_trades_u(universe: List[List], top_n: int = SETUPS_TOP_N):
    """
    Return top N trade setups from Universe.
    Each candidate: (side, sym, entry, tp, sl, score, confidence)
    """
    candidates = []
    for r in universe:
        base, _, pct24, pct4, pct1, last_price, _ = r
        if not last_price or last_price <= 0:
            continue

        long_ok, short_ok = classify_row_u(r)

        if long_ok:
            sc = score_long_u(r)
            if sc > 0:
                entry = last_price
                sl = entry * 0.94
                tp = entry * 1.12
                conf = confidence_from_score(sc)
                plan = multi_tp_plan(entry, sl, "BUY", conf)
                if plan:
                    tp = plan["tp2"]  # store TP2 as main TP
                candidates.append(("BUY", base, entry, tp, sl, sc, conf))

        if short_ok:
            sc = score_short_u(r)
            if sc > 0:
                entry = last_price
                sl = entry * 1.06
                tp = entry * 0.91
                conf = confidence_from_score(sc)
                plan = multi_tp_plan(entry, sl, "SELL", conf)
                if plan:
                    tp = plan["tp2"]
                candidates.append(("SELL", base, entry, tp, sl, sc, conf))

    candidates.sort(key=lambda x: x[5], reverse=True)
    return candidates[:top_n]


# =========================================================
# FORMATTING
# =========================================================

def pct_with_emoji(p: float) -> str:
    val = int(round(p))
    if val >= 3:
        emo = "🟢"
    elif val <= -3:
        emo = "🔴"
    else:
        emo = "🟡"
    return f"{val:+d}% {emo}"

def m_dollars(x: float) -> str:
    return str(round(x / 1_000_000))

def fmt_leaders_u(rows: List[List]) -> str:
    if not rows:
        return "*🔥 Market Leaders*: _None_\n"

    pretty = []
    for r in rows:
        base, fut_usd, pct24, pct4, pct1, *_ = r
        pretty.append([
            base,
            m_dollars(fut_usd),
            side_hint(pct4, pct1),
            pct_with_emoji(pct24),
            pct_with_emoji(pct4),
            pct_with_emoji(pct1),
        ])

    return (
        "*🔥 Market Leaders (Liquidity + Momentum)*:\n"
        "```\n"
        + tabulate(pretty, headers=["SYM", "F", "BIAS", "%24H", "%4H", "%1H"], tablefmt="github")
        + "\n```\n"
    )

def fmt_movers(movers: List[Tuple[str, float, float]]) -> str:
    if not movers:
        return "*🚀 Strong Movers (24h)*: _None_\n"

    pretty = []
    for base, fut_usd, pct24 in movers:
        pretty.append([base, m_dollars(fut_usd), pct_with_emoji(pct24)])

    return (
        "*🚀 Strong Movers (24h, F vol > $1M & +10%)*:\n"
        "```\n"
        + tabulate(pretty, headers=["SYM", "F", "%24H"], tablefmt="github")
        + "\n```\n"
    )

def format_trade_setups(recs: List[Tuple]) -> str:
    """
    recs: (side, sym, entry, tp, sl, score, conf)
    show Multi-TP if conf>=75
    """
    if not recs:
        return "_No strong setups right now._"

    lines = []
    for side, sym, entry, tp, sl, score, conf in recs:
        plan = multi_tp_plan(entry, sl, side, conf)
        if plan:
            lines.append(
                f"{side} {sym} — Conf {conf}/100 🔥 — score {score:.1f}\n"
                f"Entry {entry:.6g} | SL {sl:.6g}\n"
                f"TP plan: {plan['weights']}\n"
                f"TP1 {plan['tp1']:.6g} | TP2 {plan['tp2']:.6g} (stored TP)\n"
                f"{plan['runner_text']}\n"
            )
        else:
            lines.append(
                f"{side} {sym} — Conf {conf}/100 — score {score:.1f}\n"
                f"Entry {entry:.6g} | SL {sl:.6g} | TP {tp:.6g}\n"
            )
    return "\n".join(lines).strip()


# =========================================================
# SIGNAL STORAGE (so /risk can reference latest)
# =========================================================

def db_store_setups_as_signals(recs: List[Tuple]):
    """
    recs: (side, sym, entry, tp, sl, score, conf)
    Store as status=SIGNAL
    """
    now_ts = int(time.time())
    with db_conn() as conn:
        for side, sym, entry, tp, sl, score, conf in recs:
            conn.execute("""
                INSERT INTO signals(ts, symbol, side, entry, sl, tp, confidence, score, status)
                VALUES(?,?,?,?,?,?,?,?, 'SIGNAL')
            """, (now_ts, sym, side, float(entry), float(sl), float(tp), int(conf), float(score)))
        conn.commit()

def db_get_latest_signal(sym: str) -> Optional[sqlite3.Row]:
    """
    Find latest SIGNAL for symbol (any side), else latest OPEN.
    """
    with db_conn() as conn:
        r = conn.execute("""
            SELECT *
            FROM signals
            WHERE symbol=? AND status='SIGNAL'
            ORDER BY ts DESC
            LIMIT 1
        """, (sym,)).fetchone()
        if r:
            return r
        r2 = conn.execute("""
            SELECT *
            FROM signals
            WHERE symbol=? AND status='OPEN'
            ORDER BY opened_ts DESC, ts DESC
            LIMIT 1
        """, (sym,)).fetchone()
        return r2

def db_open_from_signal(sig_id: int, risk_usd: float) -> int:
    """
    Convert SIGNAL->OPEN and attach risk_usd/opened_ts.
    returns id
    """
    now_ts = int(time.time())
    with db_conn() as conn:
        conn.execute("""
            UPDATE signals
            SET status='OPEN', risk_usd=?, opened_ts=?
            WHERE id=? AND status='SIGNAL'
        """, (float(risk_usd), now_ts, int(sig_id)))
        conn.commit()
    return sig_id

def db_close_open_symbol(sym: str, result: str) -> bool:
    """
    Close latest OPEN for symbol without pnl.
    """
    now_ts = int(time.time())
    with db_conn() as conn:
        r = conn.execute("""
            SELECT id FROM signals
            WHERE symbol=? AND status='OPEN'
            ORDER BY opened_ts DESC, ts DESC
            LIMIT 1
        """, (sym,)).fetchone()
        if not r:
            return False
        conn.execute("""
            UPDATE signals
            SET status='CLOSED', closed_ts=?, close_result=?
            WHERE id=?
        """, (now_ts, result, int(r["id"])))
        conn.commit()
        return True

def db_closepnl_open_symbol(sym: str, pnl: float) -> Optional[dict]:
    """
    Close latest OPEN for symbol with pnl and equity update.
    returns details dict
    """
    now_ts = int(time.time())
    result = "win" if pnl > 0 else ("loss" if pnl < 0 else "flat")

    with db_conn() as conn:
        eq_row = conn.execute("SELECT value FROM settings WHERE key='equity'").fetchone()
        if not eq_row or eq_row["value"] is None:
            return {"error": "NO_EQUITY"}

        old_eq = float(eq_row["value"])

        sig = conn.execute("""
            SELECT id, side, entry, sl, tp, confidence, score, risk_usd
            FROM signals
            WHERE symbol=? AND status='OPEN'
            ORDER BY opened_ts DESC, ts DESC
            LIMIT 1
        """, (sym,)).fetchone()
        if not sig:
            return {"error": "NO_OPEN"}

        conn.execute("""
            UPDATE signals
            SET status='CLOSED', closed_ts=?, close_result=?, pnl_usd=?
            WHERE id=?
        """, (now_ts, result, float(pnl), int(sig["id"])))

        new_eq = old_eq + float(pnl)
        conn.execute(
            "INSERT INTO settings(key,value) VALUES('equity',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(new_eq),)
        )
        conn.commit()

        return {
            "error": None,
            "symbol": sym,
            "side": sig["side"],
            "entry": float(sig["entry"] or 0),
            "sl": float(sig["sl"] or 0),
            "tp": float(sig["tp"] or 0),
            "confidence": int(sig["confidence"] or 0),
            "score": float(sig["score"] or 0),
            "risk_usd": float(sig["risk_usd"] or 0),
            "old_eq": old_eq,
            "new_eq": new_eq,
            "pnl": float(pnl),
            "result": result,
        }


# =========================================================
# EMAIL
# =========================================================

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


# =========================================================
# TELEGRAM COMMANDS
# =========================================================

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Commands:\n"
        "• /screen — Market Leaders + Movers + Top Setups\n"
        "• /session [both|london|ny|off]\n"
        "• /equity 1000\n"
        "• /limits 5 200 300   (maxTrades/day, maxDailyRisk$, maxOpenRisk$)\n"
        "• /risk BTC 1%   OR   /risk BTC 10\n"
        "• /open — open positions ledger\n"
        "• /close BTC win|loss|flat\n"
        "• /closepnl BTC +23.5   (updates equity)\n"
        "• /status\n"
        "• /notify_on /notify_off /notify\n"
        "• /diag\n"
    )
    await update.message.reply_text(text)

async def diag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode = get_session_mode()
    mt, md, mo = get_limits()
    eq = get_equity()
    msg = (
        f"*Diag*\n"
        f"- session_mode: `{mode}`\n"
        f"- in_window_now: `{is_in_trading_window(mode)}`\n"
        f"- equity: `{eq if eq is not None else 'not set'}`\n"
        f"- limits: trades/day={mt}, daily_risk=${md:.2f}, open_risk=${mo:.2f}\n"
        f"- email notify: `{'ON' if NOTIFY_ON else 'OFF'}` to `{EMAIL_TO or 'n/a'}`\n"
        f"- last_error: `{LAST_ERROR or 'none'}`\n"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def session_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        mode = get_session_mode()
        await update.message.reply_text(
            "Trading Session Mode:\n"
            f"- Current: {mode}\n\n"
            "Set:\n"
            "/session both    (London + New York)\n"
            "/session london\n"
            "/session ny\n"
            "/session off     (always on – advanced)"
        )
        return

    mode = context.args[0].lower().strip()
    try:
        await asyncio.to_thread(set_session_mode, mode)
        await update.message.reply_text(f"✅ Trading session set to: {mode}")
    except Exception:
        await update.message.reply_text("Usage: /session both | london | ny | off")

async def equity_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        eq = get_equity()
        await update.message.reply_text(f"Equity: {eq if eq is not None else 'not set'}")
        return
    try:
        eq = float(context.args[0].replace(",", ""))
        await asyncio.to_thread(set_equity, eq)
        await update.message.reply_text(f"✅ Equity set to ${eq:,.2f}")
    except Exception:
        await update.message.reply_text("Usage: /equity 1000")

async def limits_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        mt, md, mo = get_limits()
        await update.message.reply_text(
            f"Limits:\n"
            f"- max trades/day: {mt}\n"
            f"- max daily risk: ${md:,.2f}\n"
            f"- max open risk:  ${mo:,.2f}\n\n"
            f"Set: /limits 5 200 300"
        )
        return
    if len(context.args) < 3:
        await update.message.reply_text("Usage: /limits <maxTradesDay> <maxDailyRiskUSD> <maxOpenRiskUSD>")
        return
    try:
        mt = int(context.args[0])
        md = float(context.args[1].replace(",", ""))
        mo = float(context.args[2].replace(",", ""))
        await asyncio.to_thread(set_limits, mt, md, mo)
        await update.message.reply_text(f"✅ Limits set: trades/day={mt}, daily=${md:,.2f}, open=${mo:,.2f}")
    except Exception:
        await update.message.reply_text("Usage: /limits 5 200 300")

async def notify_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global NOTIFY_ON
    NOTIFY_ON = True
    await update.message.reply_text("Email alerts: ON")

async def notify_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global NOTIFY_ON
    NOTIFY_ON = False
    await update.message.reply_text("Email alerts: OFF")

async def notify_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Email alerts: {'ON' if NOTIFY_ON else 'OFF'} → {EMAIL_TO or 'n/a'}"
    )

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        eq = get_equity()
        mt, md, mo = get_limits()
        mode = get_session_mode()

        day_start = mel_day_start_ts()
        daily_used = await asyncio.to_thread(db_daily_used_risk, day_start)
        trades_today = await asyncio.to_thread(db_daily_trade_count, day_start)
        open_used = await asyncio.to_thread(db_open_used_risk)
        open_count = await asyncio.to_thread(db_open_count)

        warn = trading_window_warning() or ""

        lines = []
        lines.append("STATUS (Today)")
        lines.append("-----------------------------")
        lines.append(f"Session mode:     {mode}")
        lines.append(f"Equity:           {('$' + format(eq, ',.2f')) if eq is not None else 'Not set'}")
        lines.append(f"Trades today:     {trades_today}/{mt}")
        lines.append(f"Daily risk used:  ${daily_used:,.2f}/${md:,.2f}")
        lines.append(f"Open risk used:   ${open_used:,.2f}/${mo:,.2f}")
        lines.append(f"Open positions:   {open_count}")

        alerts = []
        if trades_today >= mt:
            alerts.append("⚠️ Max trades/day reached")
        if daily_used >= md * 0.8:
            alerts.append("⚠️ Daily risk near cap (80%+)")
        if open_used >= mo * 0.8:
            alerts.append("⚠️ Open risk near cap (80%+)")

        text = "\n".join(lines)
        if alerts:
            text += "\n\nWarnings:\n- " + "\n- ".join(alerts)

        msg = (warn or "") + "*Status*\n" + "```\n" + text + "\n```"
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logging.exception("status_cmd failed")
        await update.message.reply_text(f"Status error: {e}")

async def open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        eq = get_equity()
        mt, md, mo = get_limits()

        def _fetch_open():
            with db_conn() as conn:
                return conn.execute("""
                    SELECT symbol, side, entry, sl, tp, confidence, score, risk_usd, opened_ts, ts
                    FROM signals
                    WHERE status='OPEN'
                    ORDER BY COALESCE(opened_ts, ts) DESC
                """).fetchall()

        rows = await asyncio.to_thread(_fetch_open)
        if not rows:
            await update.message.reply_text("No OPEN positions in the bot ledger.")
            return

        now_ts = int(time.time())
        total_risk = 0.0
        lines = []
        lines.append("OPEN POSITIONS (ledger)")
        lines.append("--------------------------------------------------")

        for r in rows:
            sym = r["symbol"]
            side = r["side"]
            entry = float(r["entry"] or 0)
            sl = float(r["sl"] or 0)
            tp = float(r["tp"] or 0)
            conf = int(r["confidence"] or 0)
            score = float(r["score"] or 0)
            risk = float(r["risk_usd"] or 0.0)
            opened = int(r["opened_ts"] or r["ts"] or now_ts)
            age_hr = max(0.0, (now_ts - opened) / 3600.0)

            total_risk += risk
            risk_pct = ""
            if eq and eq > 0 and risk > 0:
                risk_pct = f" ({(risk / eq) * 100.0:.2f}%)"

            plan = multi_tp_plan(entry, sl, side, conf)

            if plan:
                tp_block = f"TP1 {plan['tp1']:.6g} | TP2 {plan['tp2']:.6g} | Runner"
            else:
                tp_block = f"TP {tp:.6g}"

            lines.append(
                f"{sym:6} {side:4} | Conf {conf:3d} | score {score:4.1f} | "
                f"E:{entry:.6g} SL:{sl:.6g} {tp_block} | "
                f"Risk:${risk:,.2f}{risk_pct} | Age:{age_hr:.1f}h"
            )

        lines.append("--------------------------------------------------")
        lines.append(f"Open count: {len(rows)}")
        lines.append(f"Open risk:  ${total_risk:,.2f} / ${mo:,.2f}")
        lines.append("")
        lines.append("Close with: /closepnl SYMBOL +23.5   (or -10)")
        lines.append("Fallback:   /close SYMBOL win|loss|flat")

        msg = "*Open*\n" + "```\n" + "\n".join(lines) + "\n```"
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logging.exception("open_cmd failed")
        await update.message.reply_text(f"Open error: {e}")

async def close_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /close BTC win|loss|flat")
        return
    sym = re.sub(r"[^A-Za-z$]", "", context.args[0]).upper().lstrip("$")
    result = context.args[1].lower().strip()
    if result not in ("win", "loss", "flat"):
        await update.message.reply_text("Usage: /close BTC win|loss|flat")
        return
    ok = await asyncio.to_thread(db_close_open_symbol, sym, result)
    if ok:
        await update.message.reply_text(f"✅ CLOSED {sym} ({result}). Tip: use /closepnl {sym} +23.5 for accurate tracking.")
    else:
        await update.message.reply_text(f"No OPEN position found for {sym}.")

async def closepnl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /closepnl BTC +23.5   OR   /closepnl BTC -10")
        return
    sym = re.sub(r"[^A-Za-z$]", "", context.args[0]).upper().lstrip("$")
    pnl_txt = context.args[1].replace(",", "").strip()
    try:
        pnl = float(pnl_txt)
    except Exception:
        await update.message.reply_text("PNL must be a number. Example: /closepnl BTC -10")
        return

    details = await asyncio.to_thread(db_closepnl_open_symbol, sym, pnl)
    if details.get("error") == "NO_EQUITY":
        await update.message.reply_text("Equity is not set. Set it first: /equity 1000")
        return
    if details.get("error") == "NO_OPEN":
        await update.message.reply_text(f"No OPEN position found for {sym}.")
        return

    day_start = mel_day_start_ts()
    daily_used = await asyncio.to_thread(db_daily_used_risk, day_start)
    trades_today = await asyncio.to_thread(db_daily_trade_count, day_start)
    open_used = await asyncio.to_thread(db_open_used_risk)
    mt, md, mo = get_limits()

    text = (
        f"CLOSED {sym} ({details['side']}) — {details['result'].upper()}\n"
        f"PNL: ${details['pnl']:,.2f}\n"
        f"Equity: ${details['old_eq']:,.2f} → ${details['new_eq']:,.2f}\n"
        f"Reserved risk (was): ${details['risk_usd']:,.2f}\n\n"
        f"Limits snapshot (today):\n"
        f"- Trades today: {trades_today}/{mt}\n"
        f"- Daily risk used: ${daily_used:,.2f}/${md:,.2f}\n"
        f"- Open risk used:  ${open_used:,.2f}/${mo:,.2f}\n"
    )
    await update.message.reply_text("*Close PnL*\n```\\\n" + text + "\n```", parse_mode=ParseMode.MARKDOWN)

async def risk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /risk BTC 1%
    /risk BTC 10
    Uses latest stored SIGNAL for symbol (from /screen), converts to OPEN and stores risk.
    """
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /risk BTC 1%   OR   /risk BTC 10")
        return

    sym = re.sub(r"[^A-Za-z$]", "", context.args[0]).upper().lstrip("$")
    risk_txt = context.args[1].replace(",", "").strip()

    eq = get_equity()
    if eq is None or eq <= 0:
        await update.message.reply_text("Equity is not set. Set it first: /equity 1000")
        return

    # Parse risk
    risk_usd = None
    if risk_txt.endswith("%"):
        try:
            pct = float(risk_txt[:-1])
            risk_usd = eq * (pct / 100.0)
        except Exception:
            pass
    else:
        try:
            risk_usd = float(risk_txt)
        except Exception:
            pass

    if risk_usd is None or risk_usd <= 0:
        await update.message.reply_text("Invalid risk. Examples: /risk BTC 1%   or  /risk BTC 10")
        return

    # Limits checks
    mt, md, mo = get_limits()
    day_start = mel_day_start_ts()
    daily_used = await asyncio.to_thread(db_daily_used_risk, day_start)
    trades_today = await asyncio.to_thread(db_daily_trade_count, day_start)
    open_used = await asyncio.to_thread(db_open_used_risk)

    if trades_today + 1 > mt:
        await update.message.reply_text("❌ Max trades/day reached.")
        return
    if daily_used + risk_usd > md:
        await update.message.reply_text("❌ Daily risk cap exceeded.")
        return
    if open_used + risk_usd > mo:
        await update.message.reply_text("❌ Open risk cap exceeded.")
        return

    sig = await asyncio.to_thread(db_get_latest_signal, sym)
    if not sig:
        await update.message.reply_text(f"No stored signal found for {sym}. Run /screen first.")
        return

    entry = float(sig["entry"] or 0)
    sl = float(sig["sl"] or 0)
    tp = float(sig["tp"] or 0)
    side = (sig["side"] or "").upper()
    conf = int(sig["confidence"] or 0)
    score = float(sig["score"] or 0)

    if not entry or not sl or entry <= 0 or sl <= 0:
        await update.message.reply_text(f"Signal for {sym} is missing entry/SL. Run /screen again.")
        return

    stop_dist = abs(entry - sl)
    if stop_dist <= 0:
        await update.message.reply_text("Invalid SL distance.")
        return

    qty = risk_usd / stop_dist
    notional = qty * entry

    # Convert SIGNAL -> OPEN if possible; otherwise if already OPEN just record execution
    sig_id = int(sig["id"])
    if sig["status"] == "SIGNAL":
        await asyncio.to_thread(db_open_from_signal, sig_id, risk_usd)
    else:
        # If latest is OPEN, do NOT open again
        await update.message.reply_text(f"{sym} already has an OPEN position in ledger. Use /open or /closepnl.")
        return

    # Record execution for daily risk + trades/day
    now_ts = int(time.time())
    def _ins_exec():
        with db_conn() as conn:
            conn.execute(
                "INSERT INTO executions(ts, symbol, side, risk_usd) VALUES(?,?,?,?)",
                (now_ts, sym, side, float(risk_usd))
            )
            conn.commit()
    await asyncio.to_thread(_ins_exec)

    plan = multi_tp_plan(entry, sl, side, conf)
    if plan:
        tp_block = (
            f"TP plan: {plan['weights']}\n"
            f"TP1:   {plan['tp1']:.6g}\n"
            f"TP2:   {plan['tp2']:.6g} (stored TP)\n"
            f"Runner: {plan['runner_text']}\n"
        )
    else:
        tp_block = f"TP:    {tp:.6g}\n"

    msg = (
        f"*Risk Plan — {sym}*\n"
        f"```\n"
        f"Side:  {side}\n"
        f"Conf:  {conf}/100 | score {score:.1f}\n"
        f"Entry: {entry:.6g}\n"
        f"SL:    {sl:.6g}\n"
        f"{tp_block}"
        f"Risk:  ${risk_usd:,.2f} ({(risk_usd/eq)*100:.2f}% of equity)\n"
        f"Stop distance: {stop_dist:.6g}\n"
        f"Position size (qty): {qty:.6g}\n"
        f"Notional: ${notional:,.2f}\n"
        f"\n"
        f"Close with: /closepnl {sym} +23.5 (or -10)\n"
        f"```\n"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def screen_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /screen shows:
    - Leaders
    - Movers
    - Setups (Top 3)
    Also stores setups into DB as SIGNAL, but only if inside trading window.
    """
    try:
        PCT_CACHE.clear()

        best_fut, raw_fut = await asyncio.to_thread(load_best_futures)
        universe = await asyncio.to_thread(build_universe, best_fut)
        leaders = await asyncio.to_thread(build_market_leaders_u, universe)
        movers = await asyncio.to_thread(scan_strong_movers_fut, best_fut)
        recs = await asyncio.to_thread(pick_best_trades_u, universe, SETUPS_TOP_N)

        warn = trading_window_warning() or ""

        # Store signals only if inside trading window (guard)
        mode = get_session_mode()
        if is_in_trading_window(mode) and recs:
            await asyncio.to_thread(db_store_setups_as_signals, recs)

        rec_text = format_trade_setups(recs)

        msg = (
            (warn or "")
            + fmt_leaders_u(leaders)
            + fmt_movers(movers)
            + "*🎯 Trade Setups (Top 3)*:\n"
            + rec_text
            + f"\n\n`tickers: fut={raw_fut}`"
        )

        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logging.exception("screen error")
        await update.message.reply_text(f"Error: {e}")

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    If user types a symbol like BTC -> show latest stored SIGNAL/OPEN details.
    """
    text = (update.message.text or "").strip()
    token = re.sub(r"[^A-Za-z$]", "", text).upper().lstrip("$")
    if len(token) < 2:
        return

    sig = await asyncio.to_thread(db_get_latest_signal, token)
    if not sig:
        await update.message.reply_text("No stored signal found. Run /screen first.")
        return

    sym = sig["symbol"]
    side = sig["side"]
    entry = float(sig["entry"] or 0)
    sl = float(sig["sl"] or 0)
    tp = float(sig["tp"] or 0)
    conf = int(sig["confidence"] or 0)
    score = float(sig["score"] or 0)
    status = sig["status"]

    plan = multi_tp_plan(entry, sl, side, conf)
    if plan:
        tp_line = f"TP1 {plan['tp1']:.6g} | TP2 {plan['tp2']:.6g} | Runner"
    else:
        tp_line = f"TP {tp:.6g}"

    out = (
        "```\n"
        f"{sym} ({status})\n"
        f"Side: {side} | Conf {conf}/100 | score {score:.1f}\n"
        f"Entry: {entry:.6g}\n"
        f"SL:    {sl:.6g}\n"
        f"{tp_line}\n"
        "```"
    )
    await update.message.reply_text(out, parse_mode=ParseMode.MARKDOWN)


# =========================================================
# ALERT JOB (Optional Email)
# =========================================================

async def alert_job(context: ContextTypes.DEFAULT_TYPE):
    global LAST_EMAIL_TS, LAST_EMAIL_KEYS
    try:
        if not NOTIFY_ON:
            return
        if not email_config_ok():
            return

        mode = get_session_mode()
        if not is_in_trading_window(mode):
            return

        now = time.time()
        if now - LAST_EMAIL_TS < EMAIL_MIN_INTERVAL_SEC:
            return

        PCT_CACHE.clear()

        best_fut, _ = await asyncio.to_thread(load_best_futures)
        universe = await asyncio.to_thread(build_universe, best_fut)
        recs = await asyncio.to_thread(pick_best_trades_u, universe, SETUPS_TOP_N)
        movers = await asyncio.to_thread(scan_strong_movers_fut, best_fut)

        if not recs and not movers:
            return

        # Dedup: keys based on symbols + side for recs and symbols for movers
        rec_keys = {f"{side}:{sym}" for side, sym, *_ in recs}
        mov_keys = {f"M:{sym}" for sym, *_ in movers}
        keys = rec_keys | mov_keys
        if keys and keys == LAST_EMAIL_KEYS:
            return

        # store setups as signals (so /risk can use)
        if recs:
            await asyncio.to_thread(db_store_setups_as_signals, recs)

        body_parts = []
        if recs:
            body_parts.append("Top Trade Setups:\n" + format_trade_setups(recs))
        if movers:
            mv_lines = "\n".join([f"{sym} | F~{m_dollars(fusd)}M | {pct_with_emoji(p24)}" for sym, fusd, p24 in movers])
            body_parts.append("Strong Movers (24h):\n" + mv_lines)

        body = "\n\n".join(body_parts)
        subject = "Crypto Alert: Setups & Movers"

        if send_email(subject, body):
            LAST_EMAIL_TS = now
            LAST_EMAIL_KEYS = keys

    except Exception as e:
        logging.exception("alert_job error: %s", e)


# =========================================================
# ERROR HANDLER
# =========================================================

async def log_err(update: object, context: ContextTypes.DEFAULT_TYPE):
    try:
        raise context.error
    except Conflict:
        logging.warning("Conflict: another instance already polling.")
    except Exception as e:
        logging.exception("Unhandled error: %s", e)


# =========================================================
# MAIN
# =========================================================

def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN missing")

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

    db_init()

    app = Application.builder().token(TOKEN).build()

    # core
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("screen", screen_cmd))
    app.add_handler(CommandHandler("session", session_cmd))
    app.add_handler(CommandHandler("diag", diag_cmd))

    # risk ledger
    app.add_handler(CommandHandler("equity", equity_cmd))
    app.add_handler(CommandHandler("limits", limits_cmd))
    app.add_handler(CommandHandler("risk", risk_cmd))
    app.add_handler(CommandHandler("open", open_cmd))
    app.add_handler(CommandHandler("close", close_cmd))
    app.add_handler(CommandHandler("closepnl", closepnl_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    # email
    app.add_handler(CommandHandler("notify_on", notify_on_cmd))
    app.add_handler(CommandHandler("notify_off", notify_off_cmd))
    app.add_handler(CommandHandler("notify", notify_cmd))

    # symbol lookup
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    app.add_error_handler(log_err)

    # JobQueue for email alerts
    if getattr(app, "job_queue", None):
        app.job_queue.run_repeating(
            alert_job,
            interval=CHECK_INTERVAL_MIN * 60,
            first=10,
        )
    else:
        logging.warning('JobQueue not available. Install "python-telegram-bot[job-queue]".')

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
