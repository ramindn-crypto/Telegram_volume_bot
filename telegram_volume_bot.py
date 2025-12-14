#!/usr/bin/env python3
"""
PulseFutures — Futures-only Telegram Scanner + Trade Setups + Risk Ledger
Render-friendly single service (Telegram bot + FastAPI health endpoint)

✅ What’s included (as per everything we agreed):
- Futures ONLY (CoinEx swap via CCXT)
- /screen redesigned: Market Leaders (Top 10) + Strong Movers (24h) + Trade Setups (Top 3)
- Market Leaders = 10 (fixed)
- Strong Movers shown for context (not signals)
- Trade Setups are the only actionable ideas (entry/SL/TP)
- Trigger logic: 1h momentum; (basic confirmation in logic via 4h bias)
- Confidence score shown (NO “Score” in outputs)
- Multi-TP only for Confidence >= 75
- Dynamic decimals for prices (no long ugly decimals)
- Emails:
  - Optional (if EMAIL_* configured)
  - NO Sundays (Melbourne)
  - No duplicate spam (same keys -> skip)
- Paywall OFF for now (no Locked on Free Plan anywhere)
  - require_pro always returns True
- English-only User Guide (/help)
- Render thread-safe: bot runs in background thread with its own event loop
  - run_polling(stop_signals=None) avoids signal handler crash in non-main thread
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
from tabulate import tabulate

from fastapi import FastAPI
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
PORT = int(os.environ.get("PORT", "10000"))

# Universe Filter (Futures-only)
UNIVERSE_MIN_FUT_USD = 2_000_000

# Display sizes (fixed)
LEADERS_MIN_FUT_USD = 5_000_000
LEADERS_TOP_N = 10
MOVERS_TOP_N = 10
SETUPS_TOP_N = 3

# Multi-TP threshold (Confidence)
CONF_MULTI_TP_MIN = 75

# Trading Sessions in UTC (guard/warning only)
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
# Formatting Helpers
# =========================================================

def fmt_price(px: float) -> str:
    """Dynamic decimals for trader-friendly price display."""
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
          status TEXT NOT NULL,      -- SIGNAL | OPEN | CLOSED
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

# ---- Paywall is OFF for now ----
def require_pro(update: Update) -> Tuple[bool, str]:
    return True, ""


# =========================================================
# Risk / Ledger
# =========================================================

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
# Trading Window Guard (warning only)
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
        return "⚠️ Outside optimal trading window (London + New York). Signals may be limited.\n\n"
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
    # Trigger: 1h; Confirm: 4h
    if pct1 >= 2.0 and pct4 >= 0.0:
        return "LONG 🟢"
    if pct1 <= -2.0 and pct4 <= 0.0:
        return "SHORT 🔴"
    return "NEUTRAL 🟡"

def leader_rank_score_u(row: List) -> float:
    # Liquidity + magnitude of momentum (context leaders)
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
    # Trigger: 1h; Confirm: 4h; Bias: 24h
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
    # Normalize to 0..100 for user-friendly confidence
    s = max(0.0, min(score, 10.0))
    return int(round((s / 10.0) * 100))

def multi_tp_plan(entry: float, sl: float, side: str, confidence: int):
    if confidence < CONF_MULTI_TP_MIN:
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
    """
    Return top N setups. Tuple:
      (side, sym, entry, tp, sl, confidence)
    NO user-facing score.
    """
    candidates = []
    for r in universe:
        base, _, _, pct4, pct1, last_price, _ = r
        if not last_price or last_price <= 0:
            continue

        long_ok, short_ok = classify_row_u(r)

        if long_ok:
            sc = score_long_u(r)
            if sc > 0:
                entry = float(last_price)
                sl = entry * 0.94
                conf = confidence_from_score(sc)
                plan = multi_tp_plan(entry, sl, "BUY", conf)
                tp = (plan["tp2"] if plan else (entry * 1.12))
                candidates.append(("BUY", base, entry, tp, sl, conf))

        if short_ok:
            sc = score_short_u(r)
            if sc > 0:
                entry = float(last_price)
                sl = entry * 1.06
                conf = confidence_from_score(sc)
                plan = multi_tp_plan(entry, sl, "SELL", conf)
                tp = (plan["tp2"] if plan else (entry * 0.91))
                candidates.append(("SELL", base, entry, tp, sl, conf))

    # Sort by confidence desc (and then by tighter stop? not needed now)
    candidates.sort(key=lambda x: x[5], reverse=True)
    return candidates[:top_n]


# =========================================================
# Output Blocks (tables + setups)
# =========================================================

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
        "*🔥 Market Leaders (Top 10)*:\n"
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
        "*🚀 Strong Movers (24h, context)*:\n"
        "```\n"
        + tabulate(pretty, headers=["SYM", "F(M)", "%24H"], tablefmt="github")
        + "\n```\n"
    )

def format_trade_setups(recs: List[Tuple]) -> str:
    if not recs:
        return "_No strong setups right now._"

    lines = []
    for idx, (side, sym, entry, tp, sl, conf) in enumerate(recs, start=1):
        plan = multi_tp_plan(float(entry), float(sl), side, int(conf))
        if plan:
            lines.append(
                f"*Setup #{idx}* — {side} {sym} — Confidence {int(conf)}/100 🔥\n"
                f"Entry {fmt_price(entry)} | SL {fmt_price(sl)}\n"
                f"TP plan: {plan['weights']}\n"
                f"TP1 {fmt_price(plan['tp1'])} | TP2 {fmt_price(plan['tp2'])}\n"
                f"{plan['runner_text']}\n"
            )
        else:
            lines.append(
                f"*Setup #{idx}* — {side} {sym} — Confidence {int(conf)}/100\n"
                f"Entry {fmt_price(entry)} | SL {fmt_price(sl)} | TP {fmt_price(tp)}\n"
            )
    return "\n".join(lines).strip()


# =========================================================
# Store setups for /risk
# =========================================================

def db_store_setups_as_signals(recs: List[Tuple]):
    now_ts = int(time.time())
    with db_conn() as conn:
        for side, sym, entry, tp, sl, conf in recs:
            conn.execute("""
                INSERT INTO signals(ts, symbol, side, entry, sl, tp, confidence, status)
                VALUES(?,?,?,?,?,?,?, 'SIGNAL')
            """, (now_ts, sym, side, float(entry), float(sl), float(tp), int(conf)))
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
# Email (optional)
# =========================================================

def email_config_ok() -> bool:
    return all([EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FROM, EMAIL_TO])

def melbourne_is_sunday() -> bool:
    now = datetime.now(ZoneInfo("Australia/Melbourne"))
    return now.weekday() == 6

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
# English-only User Guide
# =========================================================

HELP_TEXT = f"""📘 {BOT_NAME} — User Guide (English)

What this bot is:
{BOT_NAME} is a futures-only crypto scanner + trade setup assistant for day traders.
It highlights liquidity + momentum, and shows a small set of high-confidence trade setups.

What you get:
• Market Leaders (Top {LEADERS_TOP_N}): where liquidity + momentum concentrate right now
• Strong Movers (24h, Top {MOVERS_TOP_N}): context only — not trade signals
• Trade Setups (Top {SETUPS_TOP_N}): actionable ideas with Entry / SL / TP
• Multi-TP (Confidence ≥ {CONF_MULTI_TP_MIN}): TP1 + TP2 + Runner guidance
• Trading Window Guard: warning when outside London/NY
• Risk Tools: equity, limits, position sizing
• Manual Ledger: /risk (open) + /closepnl (close & update equity)

What this bot does NOT do:
• No auto-trading (does not place trades)
• No exchange account connection
• No guaranteed profits

Recommended workflow:
1) /screen
2) /equity 1000
3) /limits 5 200 300
4) /risk BTC 1%   (or /risk BTC 10)
5) /open
6) /closepnl BTC +23.5

Commands:
• /start — quick intro
• /help — this guide
• /screen — Leaders + Movers + Setups
• /session both|london|ny|off — trading window warning
• /diag — diagnostics

Risk & Journal:
• /equity <amount>
• /limits <maxTradesDay> <maxDailyRiskUSD> <maxOpenRiskUSD>
• /status
• /risk <SYMBOL> <RISK>   (e.g., /risk BTC 1% or /risk BTC 10)
• /open
• /closepnl <SYMBOL> <PNL> (e.g., /closepnl BTC -10)
"""


# =========================================================
# Telegram Commands
# =========================================================

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"Welcome to {BOT_NAME} 🚀\n\n"
        "Futures-only market leaders, movers (context), and high-confidence day-trade setups.\n\n"
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
    await update.message.reply_text(msg)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def diag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"*Diag*\n"
        f"- session_mode: `{get_session_mode()}` in_window=`{is_in_trading_window(get_session_mode())}`\n"
        f"- email_enabled: `{EMAIL_ENABLED_DEFAULT}` notify_on=`{NOTIFY_ON}` sunday_block=`{melbourne_is_sunday()}`\n"
        f"- leaders_top_n: `{LEADERS_TOP_N}` movers_top_n: `{MOVERS_TOP_N}` setups_top_n: `{SETUPS_TOP_N}`\n"
        f"- universe_min_fut_usd: `{UNIVERSE_MIN_FUT_USD}` leaders_min_fut_usd: `{LEADERS_MIN_FUT_USD}`\n"
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

    sig = await asyncio.to_thread(db_get_latest_signal, sym)
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

    await asyncio.to_thread(db_open_from_signal, int(sig["id"]), float(risk_usd))

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

async def screen_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    PCT_CACHE.clear()

    best_fut, raw_fut = await asyncio.to_thread(load_best_futures)
    universe = await asyncio.to_thread(build_universe, best_fut)
    leaders = await asyncio.to_thread(build_market_leaders_u, universe)
    movers = await asyncio.to_thread(scan_strong_movers_fut, best_fut)
    recs = await asyncio.to_thread(pick_best_trades_u, universe, SETUPS_TOP_N)

    # store setups for /risk usage (only when in session window)
    mode = get_session_mode()
    if is_in_trading_window(mode) and recs:
        await asyncio.to_thread(db_store_setups_as_signals, recs)

    warn = trading_window_warning() or ""

    msg = (
        (warn or "")
        + "ℹ️ Market Leaders show where liquidity and momentum are concentrated right now.\n\n"
        + fmt_leaders_u(leaders)
        + "ℹ️ Strong Movers are for market context only — not trade signals.\n\n"
        + fmt_movers(movers)
        + "ℹ️ Trade Setups are the only actionable ideas (entry, SL, TP).\n\n"
        + f"*🎯 Trade Setups (Top {SETUPS_TOP_N})*:\n"
        + format_trade_setups(recs)
        + f"\n\n`tickers: fut={raw_fut}`"
    )

    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

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
# Email Alert Job (optional)
# =========================================================

async def alert_job(context: ContextTypes.DEFAULT_TYPE):
    global LAST_EMAIL_TS, LAST_EMAIL_KEYS
    try:
        if not NOTIFY_ON:
            return
        if not email_config_ok():
            return

        # NO emails on Sundays (Melbourne)
        if melbourne_is_sunday():
            return

        # optional: only during London/NY
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

        rec_keys = {f"{side}:{sym}:{conf}" for side, sym, *_rest, conf in recs}
        mov_keys = {f"M:{sym}:{int(p24)}" for sym, _fusd, p24 in movers}
        keys = rec_keys | mov_keys
        if keys and keys == LAST_EMAIL_KEYS:
            return

        body_parts = []
        if recs:
            body_parts.append("Trade Setups (Confidence-based):\n" + re.sub(r"\*","",format_trade_setups(recs)))
        if movers:
            mv_lines = "\n".join(
                [f"{sym} | F~{m_dollars(fusd)}M | {pct_with_emoji(p24)}" for sym, fusd, p24 in movers]
            )
            body_parts.append("Strong Movers (24h, context):\n" + mv_lines)

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
# FastAPI (health only for now)
# =========================================================

app_api = FastAPI()

@app_api.get("/health")
async def health():
    return {"ok": True, "name": BOT_NAME}

def run_api_server():
    import uvicorn
    uvicorn.run(app_api, host="0.0.0.0", port=PORT, log_level="info")


# =========================================================
# Bot runner (Render/Py3.11 safe)
# =========================================================

def run_bot():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("screen", screen_cmd))
    app.add_handler(CommandHandler("session", session_cmd))
    app.add_handler(CommandHandler("diag", diag_cmd))

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

    # Start Telegram bot in background thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    # Run FastAPI in main thread
    run_api_server()

if __name__ == "__main__":
    main()
