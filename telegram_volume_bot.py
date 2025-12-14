#!/usr/bin/env python3
"""
PulseFutures — Telegram Futures Signals Bot + Stripe Paywall (Single Service, Render-friendly)

✅ Render/Python 3.11 safe:
- Telegram bot runs in a background thread with its own asyncio event loop
- run_polling(stop_signals=None) to avoid signal-handler crash in non-main thread
- FastAPI/Uvicorn runs in main thread (for /health + Stripe webhooks)

✅ Your requested changes included:
1) User Guide: ENGLISH ONLY (no Farsi anywhere)
2) TEST MODE: unlock EVERYTHING for you during testing
   - set env: TEST_MODE=true
3) No email on Sundays (Melbourne time)
4) Remove "score" from user-facing outputs (only Confidence shown)
5) Dynamic price decimals (no ridiculous long decimals)
"""

import asyncio
import logging
import math
import os
import re
import ssl
import smtplib
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from typing import Dict, List, Tuple, Optional, Set
from zoneinfo import ZoneInfo

import ccxt
import stripe
from tabulate import tabulate

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, PlainTextResponse

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

BOT_NAME = "PulseFutures"
EXCHANGE_ID = "coinex"

TOKEN = os.environ.get("TELEGRAM_TOKEN")
DB_PATH = os.environ.get("DB_PATH", "bot.db")

# ✅ Test Mode (unlock everything)
TEST_MODE = os.environ.get("TEST_MODE", "false").lower() == "true"

# Stripe / Paywall
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
PUBLIC_BASE_URL = (os.environ.get("PUBLIC_BASE_URL", "") or "").rstrip("/")
PRICE_PRO = os.environ.get("STRIPE_PRICE_PRO", "")
PRICE_ELITE = os.environ.get("STRIPE_PRICE_ELITE", "")

# Web server port (Render provides PORT)
PORT = int(os.environ.get("PORT", "10000"))

# Universe Filter (Futures-only)
UNIVERSE_MIN_FUT_USD = 2_000_000
LEADERS_MIN_FUT_USD = 5_000_000
LEADERS_TOP_N = 7
MOVERS_TOP_N = 10
SETUPS_TOP_N = 3

# Multi-TP threshold (confidence)
CONF_MULTI_TP_MIN = 75

# Trading Sessions in UTC
SESSION_LONDON = (7, 16)   # 07:00–16:00 UTC
SESSION_NY = (13, 22)      # 13:00–22:00 UTC
DEFAULT_SESSION_MODE = "both"  # both | london | ny | off

# Scheduler
CHECK_INTERVAL_MIN = 5

# Email (optional)
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

NOTIFY_ON: bool = EMAIL_ENABLED_DEFAULT
LAST_ERROR: Optional[str] = None

# Cache for OHLC % changes
PCT_CACHE: Dict[Tuple[str, str, str], float] = {}  # (dtype, symbol, tf) -> pct


# =========================================================
# DATA
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
# UTILS: formatting
# =========================================================

def fmt_price(px: float) -> str:
    """Dynamic decimals for price-like numbers (trader-friendly)."""
    try:
        px = float(px)
    except Exception:
        return str(px)
    if px >= 1000:
        return f"{px:.1f}"
    elif px >= 100:
        return f"{px:.2f}"
    elif px >= 1:
        return f"{px:.3f}"
    elif px >= 0.1:
        return f"{px:.4f}"
    elif px >= 0.01:
        return f"{px:.5f}"
    else:
        return f"{px:.6f}"

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
    return str(round(float(x) / 1_000_000))


# =========================================================
# DB
# =========================================================

def db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
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

        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
          tg_user_id INTEGER PRIMARY KEY,
          plan TEXT NOT NULL DEFAULT 'free',   -- free | pro | elite
          expires_ts INTEGER NOT NULL DEFAULT 0,
          email TEXT,
          created_ts INTEGER NOT NULL
        );
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

# -------- Users / Paywall --------

def ensure_user(tg_user_id: int):
    now_ts = int(time.time())
    with db_conn() as conn:
        r = conn.execute("SELECT tg_user_id FROM users WHERE tg_user_id=?", (tg_user_id,)).fetchone()
        if not r:
            conn.execute(
                "INSERT INTO users(tg_user_id, plan, expires_ts, created_ts) VALUES(?, 'free', 0, ?)",
                (tg_user_id, now_ts)
            )
            conn.commit()

def get_user_plan(tg_user_id: int) -> Tuple[str, int]:
    ensure_user(tg_user_id)
    with db_conn() as conn:
        r = conn.execute("SELECT plan, expires_ts FROM users WHERE tg_user_id=?", (tg_user_id,)).fetchone()
        if not r:
            return "free", 0
        return (r["plan"] or "free"), int(r["expires_ts"] or 0)

def is_active_paid(plan: str, expires_ts: int) -> bool:
    if plan not in ("pro", "elite"):
        return False
    return int(time.time()) < int(expires_ts)

def set_user_plan(tg_user_id: int, plan: str, expires_ts: int, email: Optional[str] = None):
    ensure_user(tg_user_id)
    with db_conn() as conn:
        conn.execute("""
          UPDATE users
          SET plan=?, expires_ts=?, email=COALESCE(?, email)
          WHERE tg_user_id=?
        """, (plan, int(expires_ts), email, tg_user_id))
        conn.commit()

def require_pro(update: Update) -> Tuple[bool, str]:
    # ✅ Test Mode bypass: unlock everything for you while testing
    if TEST_MODE:
        return True, ""

    uid = update.effective_user.id if update.effective_user else 0
    plan, exp = get_user_plan(uid)
    if is_active_paid(plan, exp):
        return True, ""

    if PUBLIC_BASE_URL:
        return False, (
            "🔒 Pro feature.\n\n"
            f"Upgrade:\n"
            f"- Pro:   {PUBLIC_BASE_URL}/pay?plan=pro&tg_user_id={uid}\n"
            f"- Elite: {PUBLIC_BASE_URL}/pay?plan=elite&tg_user_id={uid}\n"
        )
    return False, "🔒 Pro feature. (Server missing PUBLIC_BASE_URL env var.)"


# -------- Risk Ledger --------

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


# =========================================================
# Trading Window Guard
# =========================================================

def get_session_mode() -> str:
    return db_get_setting("session_mode", DEFAULT_SESSION_MODE) or DEFAULT_SESSION_MODE

def set_session_mode(mode: str):
    mode = (mode or "").strip().lower()
    if mode not in ("both", "london", "ny", "off"):
        raise ValueError("invalid session mode")
    db_set_setting("session_mode", mode)

def is_in_trading_window(session_mode: str) -> bool:
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
    return in_london or in_ny

def trading_window_warning() -> Optional[str]:
    mode = get_session_mode()
    if mode == "off":
        return None
    if not is_in_trading_window(mode):
        return "⚠️ Outside optimal trading window (London + New York)\nSignals may be limited.\n\n"
    return None


# =========================================================
# Exchange Helpers (Futures-only)
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

def pct_change_24h(mv_fut: Optional[MarketVol]) -> float:
    if not mv_fut:
        return 0.0
    if mv_fut.percentage:
        return float(mv_fut.percentage)
    if mv_fut.open:
        return (mv_fut.last - mv_fut.open) / mv_fut.open * 100.0
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

def compute_pct_for_symbol(symbol: str, hours: int) -> float:
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
# Universe / Leaders / Movers / Setups
# =========================================================

def build_universe(best_fut: Dict[str, MarketVol]) -> List[List]:
    """
    row: [BASE, fut_usd, pct24, pct4, pct1, last_price, fut_symbol]
    """
    rows: List[List] = []
    for base, f in best_fut.items():
        fut_usd = usd_notional(f)
        if fut_usd < UNIVERSE_MIN_FUT_USD:
            continue

        pct24 = pct_change_24h(f)
        pct4 = compute_pct_for_symbol(f.symbol, 4)
        pct1 = compute_pct_for_symbol(f.symbol, 1)
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
        pct24 = pct_change_24h(f)
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
    s = max(0.0, min(score, 10.0))
    return int(round((s / 10.0) * 100))

def multi_tp_plan(entry: float, sl: float, side: str, confidence: int):
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
        "tp1": tp1,
        "tp2": tp2,
        "runner_text": runner,
        "weights": "TP1 40% | TP2 40% | Runner 20%",
    }

def pick_best_trades_u(universe: List[List], top_n: int = SETUPS_TOP_N):
    candidates = []
    for r in universe:
        base, _, _, pct4, pct1, last_price, _ = r
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
                    tp = plan["tp2"]
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
# Formatting blocks
# =========================================================

def fmt_leaders_u(rows: List[List]) -> str:
    if not rows:
        return "*🔥 Market Leaders*: _None_\n"
    pretty = []
    for r in rows:
        base, fut_usd, pct24, pct4, pct1, *_ = r
        pretty.append([base, m_dollars(fut_usd), side_hint(pct4, pct1),
                       pct_with_emoji(pct24), pct_with_emoji(pct4), pct_with_emoji(pct1)])
    return (
        "*🔥 Market Leaders (Liquidity + Momentum)*:\n"
        "```\n"
        + tabulate(pretty, headers=["SYM", "F(M)", "BIAS", "%24H", "%4H", "%1H"], tablefmt="github")
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
        + tabulate(pretty, headers=["SYM", "F(M)", "%24H"], tablefmt="github")
        + "\n```\n"
    )

def format_trade_setups(recs: List[Tuple]) -> str:
    if not recs:
        return "_No strong setups right now._"

    lines = []
    for side, sym, entry, tp, sl, _score, conf in recs:
        plan = multi_tp_plan(entry, sl, side, conf)
        if plan:
            lines.append(
                f"{side} {sym} — Confidence {conf}/100 🔥\n"
                f"Entry {fmt_price(entry)} | SL {fmt_price(sl)}\n"
                f"TP plan: {plan['weights']}\n"
                f"TP1 {fmt_price(plan['tp1'])} | TP2 {fmt_price(plan['tp2'])}\n"
                f"{plan['runner_text']}\n"
            )
        else:
            lines.append(
                f"{side} {sym} — Confidence {conf}/100\n"
                f"Entry {fmt_price(entry)} | SL {fmt_price(sl)} | TP {fmt_price(tp)}\n"
            )
    return "\n".join(lines).strip()


# =========================================================
# Signal storage for Risk Ledger
# =========================================================

def db_store_setups_as_signals(recs: List[Tuple]):
    now_ts = int(time.time())
    with db_conn() as conn:
        for side, sym, entry, tp, sl, score, conf in recs:
            conn.execute("""
                INSERT INTO signals(ts, symbol, side, entry, sl, tp, confidence, score, status)
                VALUES(?,?,?,?,?,?,?,?, 'SIGNAL')
            """, (now_ts, sym, side, float(entry), float(sl), float(tp), int(conf), float(score)))
        conn.commit()

def db_get_latest_signal(sym: str) -> Optional[sqlite3.Row]:
    with db_conn() as conn:
        r = conn.execute("""
            SELECT * FROM signals
            WHERE symbol=? AND status='SIGNAL'
            ORDER BY ts DESC LIMIT 1
        """, (sym,)).fetchone()
        if r:
            return r
        r2 = conn.execute("""
            SELECT * FROM signals
            WHERE symbol=? AND status='OPEN'
            ORDER BY opened_ts DESC, ts DESC LIMIT 1
        """, (sym,)).fetchone()
        return r2

def db_open_from_signal(sig_id: int, risk_usd: float):
    now_ts = int(time.time())
    with db_conn() as conn:
        conn.execute("""
            UPDATE signals
            SET status='OPEN', risk_usd=?, opened_ts=?
            WHERE id=? AND status='SIGNAL'
        """, (float(risk_usd), now_ts, int(sig_id)))
        conn.commit()

def db_closepnl_open_symbol(sym: str, pnl: float) -> Optional[dict]:
    now_ts = int(time.time())
    result = "win" if pnl > 0 else ("loss" if pnl < 0 else "flat")

    with db_conn() as conn:
        eq_row = conn.execute("SELECT value FROM settings WHERE key='equity'").fetchone()
        if not eq_row or eq_row["value"] is None:
            return {"error": "NO_EQUITY"}

        old_eq = float(eq_row["value"])
        sig = conn.execute("""
            SELECT id FROM signals
            WHERE symbol=? AND status='OPEN'
            ORDER BY opened_ts DESC, ts DESC LIMIT 1
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

        return {"error": None, "old_eq": old_eq, "new_eq": new_eq, "pnl": float(pnl), "result": result}


# =========================================================
# Email (optional) helpers
# =========================================================

def email_config_ok() -> bool:
    return all([EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FROM, EMAIL_TO])

def melbourne_is_sunday() -> bool:
    now = datetime.now(ZoneInfo("Australia/Melbourne"))
    return now.weekday() == 6  # Monday=0 ... Sunday=6

def send_email(subject: str, body: str) -> bool:
    if not email_config_ok():
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
    except Exception:
        logging.exception("send_email failed")
        return False


# =========================================================
# Bot Guide (ENGLISH ONLY)
# =========================================================

HELP_TEXT = f"""📘 {BOT_NAME} — User Guide (English)

What this bot is:
{BOT_NAME} is a futures-only crypto scanner + trade setup assistant for day traders.
It focuses on liquidity + momentum, and shows a small number of high-confidence setups.

What you get:
• Market Leaders: top liquid contracts with momentum snapshot
• Strong Movers (24h): coins with strong 24h moves (filtered by futures liquidity)
• Trade Setups (Top {SETUPS_TOP_N}): entry / SL / TP ideas with a Confidence score
• Multi-TP (Confidence ≥ {CONF_MULTI_TP_MIN}): TP1 + TP2 + Runner guidance
• Trading Window Guard: warns when outside London/NY optimal window
• Risk Tools: equity, risk limits, position sizing
• Manual Trade Ledger: /risk to “open”, /closepnl to “close” and update equity

What this bot does NOT do:
• No auto-trading (it does not place trades)
• No exchange account connection
• No guaranteed profit (no hype, no promises)

Recommended workflow:
1) Run /screen
   - Review Leaders + Movers + Setups
2) Set your equity (once):
   - /equity 1000
3) Set your limits (once):
   - /limits 5 200 300
4) When you like a setup, size it:
   - /risk BTC 1%   (or /risk BTC 10)
5) Track open positions:
   - /open
6) Close and update equity:
   - /closepnl BTC +23.5   (or /closepnl BTC -10)

Commands:
• /start
  Quick intro + your current access status.

• /help
  This guide.

• /screen
  Shows Leaders + Movers + Setups.
  In paid mode (or TEST MODE), setups are stored so you can use /risk later.

• /session [both|london|ny|off]
  Controls the trading window guard.

• /diag
  Diagnostics (plan/session/last error/config flags).

Risk & Journal:
• /equity <amount>
  Set your account equity (used for % risk sizing).

• /limits <maxTradesDay> <maxDailyRiskUSD> <maxOpenRiskUSD>
  Set risk caps to prevent over-trading.

• /status
  Shows daily risk used, open risk used, and your open positions count.

• /risk <SYMBOL> <RISK>
  Calculates position size using the latest stored signal for that symbol.
  Example: /risk BTC 1%   OR   /risk BTC 10

• /open
  Lists OPEN positions in your ledger.

• /closepnl <SYMBOL> <PNL>
  Closes a position and updates your equity.
  Example: /closepnl BTC +23.5
"""

# =========================================================
# Telegram Commands
# =========================================================

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    ensure_user(uid)
    plan, exp = get_user_plan(uid)
    active = is_active_paid(plan, exp) or TEST_MODE

    msg = (
        f"Welcome to {BOT_NAME} 🚀\n\n"
        "Futures-only market leaders, movers, and high-confidence day-trade setups.\n\n"
        "Commands:\n"
        "• /screen — Leaders + Movers + Setups\n"
        "• /help\n"
        "• /session [both|london|ny|off]\n"
        "• /diag\n\n"
        "Risk & journal:\n"
        "• /equity 1000\n"
        "• /limits 5 200 300\n"
        "• /risk BTC 1%\n"
        "• /open\n"
        "• /closepnl BTC +23.5\n"
        "• /status\n"
    )

    if TEST_MODE:
        msg += "\n✅ TEST MODE is ON: all features unlocked."
    else:
        msg += f"\nYour plan: {plan} ({'ACTIVE' if active else 'inactive/free'})"

    await update.message.reply_text(msg)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def upgrade_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if TEST_MODE:
        await update.message.reply_text("✅ TEST MODE is ON: paywall is bypassed for testing.")
        return

    uid = update.effective_user.id
    ensure_user(uid)
    if not PUBLIC_BASE_URL:
        await update.message.reply_text("Server is missing PUBLIC_BASE_URL. Set it in Render env.")
        return
    msg = (
        "💳 Upgrade PulseFutures\n\n"
        f"Pro:   {PUBLIC_BASE_URL}/pay?plan=pro&tg_user_id={uid}\n"
        f"Elite: {PUBLIC_BASE_URL}/pay?plan=elite&tg_user_id={uid}\n\n"
        "After payment, come back and run /start or /screen."
    )
    await update.message.reply_text(msg)

async def diag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    plan, exp = get_user_plan(uid)
    active = is_active_paid(plan, exp) or TEST_MODE
    msg = (
        f"*Diag*\n"
        f"- test_mode: `{TEST_MODE}`\n"
        f"- plan: `{plan}` active=`{active}`\n"
        f"- expires_ts: `{exp}`\n"
        f"- session_mode: `{get_session_mode()}` in_window=`{is_in_trading_window(get_session_mode())}`\n"
        f"- email_enabled: `{EMAIL_ENABLED_DEFAULT}` notify_on=`{NOTIFY_ON}` sunday_block=`{melbourne_is_sunday()}`\n"
        f"- stripe: secret_key=`{bool(STRIPE_SECRET_KEY)}` webhook_secret=`{bool(STRIPE_WEBHOOK_SECRET)}`\n"
        f"- public_base_url: `{PUBLIC_BASE_URL or 'not set'}`\n"
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
            "/session both\n/session london\n/session ny\n/session off"
        )
        return
    mode = context.args[0].lower().strip()
    try:
        set_session_mode(mode)
        await update.message.reply_text(f"✅ session set to: {mode}")
    except Exception:
        await update.message.reply_text("Usage: /session both|london|ny|off")

# --- Pro-only (bypassed in TEST_MODE via require_pro) ---

async def equity_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, msg = require_pro(update)
    if not ok:
        await update.message.reply_text(msg)
        return
    if not context.args:
        eq = get_equity()
        await update.message.reply_text(f"Equity: {eq if eq is not None else 'not set'}")
        return
    try:
        eq = float(context.args[0].replace(",", ""))
        set_equity(eq)
        await update.message.reply_text(f"✅ Equity set to ${eq:,.2f}")
    except Exception:
        await update.message.reply_text("Usage: /equity 1000")

async def limits_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, msg = require_pro(update)
    if not ok:
        await update.message.reply_text(msg)
        return
    if not context.args:
        mt, md, mo = get_limits()
        await update.message.reply_text(f"Limits: trades/day={mt}, daily=${md:,.2f}, open=${mo:,.2f}\nSet: /limits 5 200 300")
        return
    if len(context.args) < 3:
        await update.message.reply_text("Usage: /limits <maxTradesDay> <maxDailyRiskUSD> <maxOpenRiskUSD>")
        return
    try:
        mt = int(context.args[0])
        md = float(context.args[1].replace(",", ""))
        mo = float(context.args[2].replace(",", ""))
        set_limits(mt, md, mo)
        await update.message.reply_text(f"✅ Limits set: trades/day={mt}, daily=${md:,.2f}, open=${mo:,.2f}")
    except Exception:
        await update.message.reply_text("Usage: /limits 5 200 300")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, msg = require_pro(update)
    if not ok:
        await update.message.reply_text(msg)
        return
    eq = get_equity()
    mt, md, mo = get_limits()
    day_start = mel_day_start_ts()
    daily_used = db_daily_used_risk(day_start)
    trades_today = db_daily_trade_count(day_start)
    open_used = db_open_used_risk()
    open_count = db_open_count()
    warn = trading_window_warning() or ""

    text = (
        f"Session: {get_session_mode()}\n"
        f"Equity: {('$'+format(eq, ',.2f')) if eq else 'not set'}\n"
        f"Trades today: {trades_today}/{mt}\n"
        f"Daily risk used: ${daily_used:,.2f}/${md:,.2f}\n"
        f"Open risk used:  ${open_used:,.2f}/${mo:,.2f}\n"
        f"Open positions: {open_count}\n"
    )
    await update.message.reply_text((warn or "") + "```\n" + text + "```", parse_mode=ParseMode.MARKDOWN)

async def open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, msg = require_pro(update)
    if not ok:
        await update.message.reply_text(msg)
        return
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT symbol, side, entry, sl, tp, confidence, risk_usd, opened_ts, ts
            FROM signals WHERE status='OPEN'
            ORDER BY COALESCE(opened_ts, ts) DESC
        """).fetchall()
    if not rows:
        await update.message.reply_text("No OPEN positions in ledger.")
        return
    lines = ["OPEN POSITIONS", "----------------------------"]
    now_ts = int(time.time())
    for r in rows:
        opened = int(r["opened_ts"] or r["ts"] or now_ts)
        age_hr = (now_ts - opened) / 3600.0
        lines.append(
            f"{r['symbol']:6} {r['side']:4} conf {int(r['confidence'] or 0):3d} "
            f"E:{fmt_price(float(r['entry'] or 0))} SL:{fmt_price(float(r['sl'] or 0))} "
            f"TP:{fmt_price(float(r['tp'] or 0))} risk:${float(r['risk_usd'] or 0):,.2f} "
            f"age:{age_hr:.1f}h"
        )
    lines.append("")
    lines.append("Close: /closepnl BTC +23.5   (or -10)")
    await update.message.reply_text("```\n" + "\n".join(lines) + "\n```", parse_mode=ParseMode.MARKDOWN)

async def closepnl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, msg = require_pro(update)
    if not ok:
        await update.message.reply_text(msg)
        return
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /closepnl BTC +23.5")
        return
    sym = re.sub(r"[^A-Za-z$]", "", context.args[0]).upper().lstrip("$")
    try:
        pnl = float(context.args[1].replace(",", ""))
    except Exception:
        await update.message.reply_text("PNL must be a number.")
        return
    details = db_closepnl_open_symbol(sym, pnl)
    if details.get("error") == "NO_EQUITY":
        await update.message.reply_text("Set equity first: /equity 1000")
        return
    if details.get("error") == "NO_OPEN":
        await update.message.reply_text(f"No OPEN position for {sym}.")
        return
    await update.message.reply_text(
        f"✅ CLOSED {sym} ({details['result']}) | PnL ${details['pnl']:,.2f}\n"
        f"Equity: ${details['old_eq']:,.2f} → ${details['new_eq']:,.2f}"
    )

async def risk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, msg = require_pro(update)
    if not ok:
        await update.message.reply_text(msg)
        return

    if len(context.args) < 2:
        await update.message.reply_text("Usage: /risk BTC 1%   OR   /risk BTC 10")
        return

    sym = re.sub(r"[^A-Za-z$]", "", context.args[0]).upper().lstrip("$")
    risk_txt = context.args[1].replace(",", "").strip()

    eq = get_equity()
    if eq is None or eq <= 0:
        await update.message.reply_text("Set equity first: /equity 1000")
        return

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
        await update.message.reply_text("Invalid risk amount.")
        return

    mt, md, mo = get_limits()
    day_start = mel_day_start_ts()
    daily_used = db_daily_used_risk(day_start)
    trades_today = db_daily_trade_count(day_start)
    open_used = db_open_used_risk()

    if trades_today + 1 > mt:
        await update.message.reply_text("❌ Max trades/day reached.")
        return
    if daily_used + risk_usd > md:
        await update.message.reply_text("❌ Daily risk cap exceeded.")
        return
    if open_used + risk_usd > mo:
        await update.message.reply_text("❌ Open risk cap exceeded.")
        return

    sig = db_get_latest_signal(sym)
    if not sig or sig["status"] != "SIGNAL":
        await update.message.reply_text(f"No stored SIGNAL for {sym}. Run /screen first.")
        return

    entry = float(sig["entry"] or 0)
    sl = float(sig["sl"] or 0)
    if not entry or not sl:
        await update.message.reply_text("Signal missing entry/SL. Run /screen again.")
        return

    stop_dist = abs(entry - sl)
    if stop_dist <= 0:
        await update.message.reply_text("Invalid SL distance.")
        return

    qty = risk_usd / stop_dist
    notional = qty * entry

    db_open_from_signal(int(sig["id"]), risk_usd)

    # record execution
    now_ts = int(time.time())
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO executions(ts, symbol, side, risk_usd) VALUES(?,?,?,?)",
            (now_ts, sym, sig["side"], float(risk_usd))
        )
        conn.commit()

    await update.message.reply_text(
        "```\n"
        f"{sym} {sig['side']} | Entry {fmt_price(entry)} | SL {fmt_price(sl)}\n"
        f"Risk ${risk_usd:,.2f} -> Qty {qty:.6g} | Notional ${notional:,.2f}\n"
        "Close with /closepnl SYMBOL +/-PnL\n"
        "```",
        parse_mode=ParseMode.MARKDOWN
    )


# =========================================================
# /screen (TEST MODE unlock + later paywall-ready)
# =========================================================

async def screen_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    ensure_user(uid)
    plan, exp = get_user_plan(uid)
    active = is_active_paid(plan, exp) or TEST_MODE  # ✅ unlock in TEST_MODE

    PCT_CACHE.clear()
    best_fut, raw_fut = await asyncio.to_thread(load_best_futures)
    universe = await asyncio.to_thread(build_universe, best_fut)
    leaders = await asyncio.to_thread(build_market_leaders_u, universe)
    movers = await asyncio.to_thread(scan_strong_movers_fut, best_fut)

    warn = trading_window_warning() or ""
    msg = (warn or "") + fmt_leaders_u(leaders) + fmt_movers(movers)

    # If later paywall is enabled (TEST_MODE off), we can lock setups here.
    if not active:
        if PUBLIC_BASE_URL:
            msg += (
                f"*🎯 Trade Setups (Top {SETUPS_TOP_N})*:\n"
                "🔒 Locked on Free plan.\n\n"
                "Upgrade to Pro:\n"
                f"{PUBLIC_BASE_URL}/pay?plan=pro&tg_user_id={uid}\n\n"
                f"`tickers: fut={raw_fut}`"
            )
        else:
            msg += f"*🎯 Trade Setups (Top {SETUPS_TOP_N})*:\n🔒 Locked on Free plan.\n(Server missing PUBLIC_BASE_URL)\n"
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        return

    recs = await asyncio.to_thread(pick_best_trades_u, universe, SETUPS_TOP_N)

    mode = get_session_mode()
    if is_in_trading_window(mode) and recs:
        await asyncio.to_thread(db_store_setups_as_signals, recs)

    msg += f"*🎯 Trade Setups (Top {SETUPS_TOP_N})*:\n" + format_trade_setups(recs)
    msg += f"\n\n`tickers: fut={raw_fut}`"
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


# =========================================================
# Text router: show latest stored signal (if any)
# =========================================================

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    token = re.sub(r"[^A-Za-z$]", "", text).upper().lstrip("$")
    if len(token) < 2:
        return
    sig = await asyncio.to_thread(db_get_latest_signal, token)
    if not sig:
        await update.message.reply_text("No stored signal. Run /screen first.")
        return
    await update.message.reply_text(
        "```\n"
        f"{sig['symbol']} ({sig['status']}) {sig['side']}\n"
        f"Entry {fmt_price(float(sig['entry'] or 0))} | SL {fmt_price(float(sig['sl'] or 0))} | TP {fmt_price(float(sig['tp'] or 0))}\n"
        f"Confidence {int(sig['confidence'] or 0)}/100\n"
        "```",
        parse_mode=ParseMode.MARKDOWN
    )


# =========================================================
# Email Alert Job (optional) — No Sundays (Melbourne)
# =========================================================

async def alert_job(context: ContextTypes.DEFAULT_TYPE):
    global LAST_EMAIL_TS, LAST_EMAIL_KEYS
    try:
        if not NOTIFY_ON:
            return
        if not email_config_ok():
            return

        # ✅ No emails on Sundays (Melbourne)
        if melbourne_is_sunday():
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

        rec_keys = {f"{side}:{sym}" for side, sym, *_ in recs}
        mov_keys = {f"M:{sym}" for sym, *_ in movers}
        keys = rec_keys | mov_keys
        if keys and keys == LAST_EMAIL_KEYS:
            return

        body_parts = []
        if recs:
            # ✅ user-facing: Confidence only (no score)
            body_parts.append(f"Top Trade Setups (Confidence-based):\n{format_trade_setups(recs)}")
        if movers:
            mv_lines = "\n".join(
                [f"{sym} | F~{m_dollars(fusd)}M | {pct_with_emoji(p24)}"
                 for sym, fusd, p24 in movers]
            )
            body_parts.append("Strong Movers (24h):\n" + mv_lines)

        body = "\n\n".join(body_parts)
        subject = f"{BOT_NAME} Alert: Setups & Movers"

        if send_email(subject, body):
            LAST_EMAIL_TS = now
            LAST_EMAIL_KEYS = keys

    except Exception:
        logging.exception("alert_job error")


# =========================================================
# Error handler
# =========================================================

async def log_err(update: object, context: ContextTypes.DEFAULT_TYPE):
    try:
        raise context.error
    except Conflict:
        logging.warning("Conflict: another instance already polling.")
    except Exception:
        logging.exception("Unhandled error")


# =========================================================
# FastAPI (Stripe Paywall)
# =========================================================

app_api = FastAPI()

@app_api.get("/health")
async def health():
    return {"ok": True, "name": BOT_NAME, "test_mode": TEST_MODE}

@app_api.get("/pay")
async def pay(plan: str, tg_user_id: int):
    if TEST_MODE:
        return PlainTextResponse("TEST_MODE is ON. Paywall is bypassed.", status_code=200)

    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="STRIPE_SECRET_KEY not set")
    if not PUBLIC_BASE_URL:
        raise HTTPException(status_code=500, detail="PUBLIC_BASE_URL not set")
    if plan not in ("pro", "elite"):
        raise HTTPException(status_code=400, detail="Invalid plan")

    price_id = PRICE_PRO if plan == "pro" else PRICE_ELITE
    if not price_id:
        raise HTTPException(status_code=500, detail="Missing STRIPE_PRICE_PRO/ELITE env var")

    stripe.api_key = STRIPE_SECRET_KEY
    ensure_user(int(tg_user_id))

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=f"{PUBLIC_BASE_URL}/success?tg_user_id={tg_user_id}&plan={plan}",
        cancel_url=f"{PUBLIC_BASE_URL}/cancel?tg_user_id={tg_user_id}&plan={plan}",
        metadata={"tg_user_id": str(tg_user_id), "plan": plan},
    )
    return RedirectResponse(url=session.url, status_code=303)

@app_api.get("/success")
async def success(tg_user_id: int, plan: str):
    return PlainTextResponse(
        "✅ Payment successful.\n\n"
        "Go back to Telegram and run /start or /screen.\n"
        "If access doesn’t unlock within 30 seconds, run /start again."
    )

@app_api.get("/cancel")
async def cancel(tg_user_id: int, plan: str):
    return PlainTextResponse("Payment cancelled. You can retry from Telegram using /upgrade.")

@app_api.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    if TEST_MODE:
        return PlainTextResponse("TEST_MODE is ON. Webhook ignored.", status_code=200)

    if not STRIPE_WEBHOOK_SECRET or not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe webhook not configured")

    stripe.api_key = STRIPE_SECRET_KEY
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        logging.warning("Webhook signature verification failed: %s", e)
        return PlainTextResponse("bad signature", status_code=400)

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        md = session.get("metadata") or {}
        tg_user_id = int(md.get("tg_user_id") or 0)
        plan = (md.get("plan") or "pro").lower()

        sub_id = session.get("subscription")
        expires_ts = int(time.time()) + 30 * 24 * 3600  # fallback
        try:
            if sub_id:
                sub = stripe.Subscription.retrieve(sub_id)
                expires_ts = int(sub["current_period_end"])
        except Exception:
            logging.exception("Could not retrieve subscription for expiry; using fallback")

        customer_details = session.get("customer_details") or {}
        email = customer_details.get("email")

        if tg_user_id > 0:
            set_user_plan(tg_user_id, plan, expires_ts, email=email)
            logging.info("Activated user %s plan=%s exp=%s", tg_user_id, plan, expires_ts)

    return PlainTextResponse("ok", status_code=200)

def run_api_server():
    import uvicorn
    uvicorn.run(app_api, host="0.0.0.0", port=PORT, log_level="info")


# =========================================================
# Bot runner (Thread-safe for Py3.11)
# =========================================================

def run_bot():
    # ✅ Create a dedicated event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("upgrade", upgrade_cmd))
    app.add_handler(CommandHandler("screen", screen_cmd))
    app.add_handler(CommandHandler("session", session_cmd))
    app.add_handler(CommandHandler("diag", diag_cmd))

    # Risk tools / ledger
    app.add_handler(CommandHandler("equity", equity_cmd))
    app.add_handler(CommandHandler("limits", limits_cmd))
    app.add_handler(CommandHandler("risk", risk_cmd))
    app.add_handler(CommandHandler("open", open_cmd))
    app.add_handler(CommandHandler("closepnl", closepnl_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))
    app.add_error_handler(log_err)

    if getattr(app, "job_queue", None):
        app.job_queue.run_repeating(alert_job, interval=CHECK_INTERVAL_MIN * 60, first=10)

    # ✅ IMPORTANT: disable stop_signals in non-main thread
    app.run_polling(drop_pending_updates=True, stop_signals=None)


# =========================================================
# MAIN
# =========================================================

def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN missing")

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

    db_init()

    # ✅ Start Telegram bot in background thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    # ✅ Run API in main thread (Render-friendly)
    run_api_server()

if __name__ == "__main__":
    main()
