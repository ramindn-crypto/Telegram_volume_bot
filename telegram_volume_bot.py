#!/usr/bin/env python3
"""
PulseFutures — Futures-only Telegram scanner + day-trader setups + risk tools (Render-friendly)

Key changes (Day Trader optimized):
- SL/TP smaller: SL=3%, TP=6%
- Signal expiry (time-stop): SIGNAL older than 24h -> EXPIRED (not actionable)
- Email:
  - Min interval: 60 minutes
  - Only if at least one setup with Confidence >= 75
  - Per-symbol cooldown: 6 hours (avoid fatigue)
  - Persist last email + symbol timestamps in SQLite (no duplicates after restart)
  - Only inside London/NY sessions (unless session_mode=off)
  - Email sends Top 2 setups (screen can still show Top 3)

Risk tools:
- /equity /limits /status /risk /open /closepnl
- /risk supports manual sizing with warning: /risk BTC 1% 65000 64000

Optional charts:
- CHARTS_ENABLED=true (requires matplotlib in requirements)
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

# Optional charts
CHARTS_ENABLED = os.environ.get("CHARTS_ENABLED", "false").lower() == "true"
try:
    if CHARTS_ENABLED:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
except Exception:
    CHARTS_ENABLED = False


# =========================================================
# CONFIG
# =========================================================

BOT_NAME = "PulseFutures"
EXCHANGE_ID = "coinex"

TOKEN = os.environ.get("TELEGRAM_TOKEN")
DB_PATH = os.environ.get("DB_PATH", "bot.db")
PORT = int(os.environ.get("PORT", "10000"))

# Futures-only Universe / Display sizes
UNIVERSE_MIN_FUT_USD = 2_000_000
LEADERS_MIN_FUT_USD = 5_000_000

LEADERS_TOP_N = 10
MOVERS_TOP_N = 10
SETUPS_TOP_N_SCREEN = 3
SETUPS_TOP_N_EMAIL = 2

# Trigger/Confirm logic
TRIGGER_1H_PCT = 2.0
CONFIRM_15M_SIGN = True

# Confidence thresholds
CONF_MULTI_TP_MIN = 75
EMAIL_CONF_GATE = 75

# Day-trader SL/TP (%)
SL_PCT = 0.03
TP_PCT = 0.06

# Signal expiry (time-stop) for signals not opened
SIGNAL_EXPIRY_HOURS = 24

# Trading sessions in UTC
SESSION_LONDON = (7, 16)    # 07:00–16:00 UTC
SESSION_NY = (13, 22)       # 13:00–22:00 UTC
DEFAULT_SESSION_MODE = "both"  # both | london | ny | off

# Scheduler
CHECK_INTERVAL_MIN = 5

# Email
EMAIL_ENABLED_DEFAULT = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST = os.environ.get("EMAIL_HOST", "")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "465"))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", EMAIL_USER)
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_USER)

# ✅ 60 minutes interval
EMAIL_MIN_INTERVAL_SEC = 60 * 60

# ✅ Per-symbol cooldown (seconds)
EMAIL_SYMBOL_COOLDOWN_SEC = 6 * 60 * 60

NOTIFY_ON: bool = EMAIL_ENABLED_DEFAULT
LAST_ERROR: Optional[str] = None

# OHLC % cache
PCT_CACHE: Dict[Tuple[str, str, str], float] = {}

# Leverage guide thresholds
EFFECTIVE_LEV_WARN = 5.0


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
# UTILS
# =========================================================

def fmt_price(px: float) -> str:
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

def safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z$]", "", s).upper().lstrip("$")

def now_ts() -> int:
    return int(time.time())

def leverage_guide_block(notional: float, equity: float) -> str:
    if equity <= 0:
        return ""
    eff = notional / equity
    levels = [3, 5, 10, 20]
    margins = [f"{L}x≈${(notional / L):,.2f}" for L in levels]
    warn = ""
    if eff >= EFFECTIVE_LEV_WARN:
        warn = f"⚠️ Effective leverage is high ({eff:.1f}x). Consider smaller Qty or wider SL.\n"
    return (
        f"{warn}"
        f"Effective Leverage: {eff:.2f}x  (Notional/Equity)\n"
        f"Margin guide: " + " | ".join(margins)
    )


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
          status TEXT NOT NULL,          -- SIGNAL | OPEN | CLOSED | EXPIRED
          risk_usd REAL,
          opened_ts INTEGER,
          closed_ts INTEGER,
          close_result TEXT,
          pnl_usd REAL,
          origin TEXT                   -- BOT | MANUAL
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
        conn.execute("UPDATE signals SET origin='BOT' WHERE origin IS NULL;")
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

# ---------- Email persistence ----------
def db_get_last_email_ts() -> int:
    v = db_get_setting("email_last_ts", "0")
    try:
        return int(float(v))
    except Exception:
        return 0

def db_set_last_email_ts(ts: int):
    db_set_setting("email_last_ts", str(int(ts)))

def db_get_last_email_keys() -> Set[str]:
    v = db_get_setting("email_last_keys", "")
    if not v:
        return set()
    return set([x for x in v.split(",") if x.strip()])

def db_set_last_email_keys(keys: Set[str]):
    db_set_setting("email_last_keys", ",".join(sorted(keys)))

def db_get_symbol_last_emailed(sym: str) -> int:
    v = db_get_setting(f"email_sym_ts_{sym}", "0")
    try:
        return int(float(v))
    except Exception:
        return 0

def db_set_symbol_last_emailed(sym: str, ts: int):
    db_set_setting(f"email_sym_ts_{sym}", str(int(ts)))

# -------- Risk ledger --------

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

def db_expire_old_signals():
    """Mark SIGNAL rows older than SIGNAL_EXPIRY_HOURS as EXPIRED."""
    cutoff = now_ts() - SIGNAL_EXPIRY_HOURS * 3600
    with db_conn() as conn:
        conn.execute("""
            UPDATE signals
            SET status='EXPIRED'
            WHERE status='SIGNAL' AND ts < ?
        """, (cutoff,))
        conn.commit()

def db_store_setups_as_signals(recs: List[Tuple]):
    ts = now_ts()
    with db_conn() as conn:
        for side, sym, entry, tp, sl, conf in recs:
            conn.execute("""
                INSERT INTO signals(ts, symbol, side, entry, sl, tp, confidence, status, origin)
                VALUES(?,?,?,?,?,?,?, 'SIGNAL', 'BOT')
            """, (ts, sym, side, float(entry), float(sl), float(tp), int(conf)))
        conn.commit()

def db_get_latest_signal(sym: str) -> Optional[sqlite3.Row]:
    with db_conn() as conn:
        # Prefer latest SIGNAL that is not expired
        r = conn.execute("""
            SELECT * FROM signals
            WHERE symbol=? AND status='SIGNAL'
            ORDER BY ts DESC LIMIT 1
        """, (sym,)).fetchone()
        if r:
            return r
        # Otherwise latest OPEN
        r2 = conn.execute("""
            SELECT * FROM signals
            WHERE symbol=? AND status='OPEN'
            ORDER BY opened_ts DESC, ts DESC LIMIT 1
        """, (sym,)).fetchone()
        return r2

def db_open_from_signal(sig_id: int, risk_usd: float):
    ts = now_ts()
    with db_conn() as conn:
        conn.execute("""
            UPDATE signals
            SET status='OPEN', risk_usd=?, opened_ts=?
            WHERE id=? AND status='SIGNAL'
        """, (float(risk_usd), ts, int(sig_id)))
        conn.commit()

def db_open_manual(sym: str, side: str, entry: float, sl: float, tp: float, risk_usd: float):
    ts = now_ts()
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO signals(ts, symbol, side, entry, sl, tp, confidence, status, risk_usd, opened_ts, origin)
            VALUES(?,?,?,?,?,?,?, 'OPEN', ?, ?, 'MANUAL')
        """, (ts, sym, side, float(entry), float(sl), float(tp), 0, float(risk_usd), ts))
        conn.commit()

def db_closepnl_open_symbol(sym: str, pnl: float) -> Optional[dict]:
    ts = now_ts()
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
        """, (ts, result, float(pnl), int(sig["id"])))

        new_eq = old_eq + float(pnl)
        conn.execute(
            "INSERT INTO settings(key,value) VALUES('equity',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(new_eq),)
        )
        conn.commit()

        return {"error": None, "old_eq": old_eq, "new_eq": new_eq, "pnl": float(pnl), "result": result}


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
        return "⚠️ Outside optimal trading window (London + New York). Signals may be weaker.\n\n"
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

def compute_pct(symbol: str, timeframe: str, bars: int) -> float:
    dtype = "swap"
    cache_key = (dtype, symbol, f"{timeframe}:{bars}")
    if cache_key in PCT_CACHE:
        return PCT_CACHE[cache_key]

    try:
        ex = build_exchange_swap()
        ex.load_markets()
        candles = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=bars + 1)
        if not candles or len(candles) <= bars:
            PCT_CACHE[cache_key] = 0.0
            return 0.0

        closes = [c[4] for c in candles][- (bars + 1):]
        if not closes or not closes[0]:
            pct = 0.0
        else:
            pct = (closes[-1] - closes[0]) / closes[0] * 100.0

        PCT_CACHE[cache_key] = pct
        return pct
    except Exception:
        logging.exception("compute_pct failed for %s %s", symbol, timeframe)
        PCT_CACHE[cache_key] = 0.0
        return 0.0


# =========================================================
# Charts (optional)
# =========================================================

def fetch_ohlcv_for_chart(fut_symbol: str, timeframe: str = "1h", limit: int = 48) -> List[List]:
    ex = build_exchange_swap()
    ex.load_markets()
    return ex.fetch_ohlcv(fut_symbol, timeframe=timeframe, limit=limit)

def make_chart_png(fut_symbol: str, title: str, out_path: str, timeframe: str = "1h", limit: int = 48):
    if not CHARTS_ENABLED:
        return
    candles = fetch_ohlcv_for_chart(fut_symbol, timeframe=timeframe, limit=limit)
    if not candles:
        return
    xs = [c[0] for c in candles]
    closes = [c[4] for c in candles]

    plt.figure(figsize=(5.2, 2.2), dpi=160)
    plt.plot(xs, closes, linewidth=1.2)
    plt.title(title, fontsize=9)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# =========================================================
# Universe / Leaders / Movers / Setups
# =========================================================

def build_universe(best_fut: Dict[str, MarketVol]) -> List[List]:
    """
    row: [BASE, fut_usd, pct24, pct4h, pct1h, pct15m, last_price, fut_symbol]
    """
    rows: List[List] = []
    for base, f in best_fut.items():
        fut_usd = usd_notional(f)
        if fut_usd < UNIVERSE_MIN_FUT_USD:
            continue

        pct24 = pct_change_24h(f)
        pct4h = compute_pct(f.symbol, "1h", 4)
        pct1h = compute_pct(f.symbol, "1h", 1)
        pct15m = compute_pct(f.symbol, "15m", 1)
        last_price = f.last or 0.0

        rows.append([base, fut_usd, pct24, pct4h, pct1h, pct15m, last_price, f.symbol])

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows

def side_hint(pct4h: float, pct1h: float) -> str:
    if pct1h >= TRIGGER_1H_PCT and pct4h >= 0.0:
        return "LONG 🟢"
    if pct1h <= -TRIGGER_1H_PCT and pct4h <= 0.0:
        return "SHORT 🔴"
    return "NEUTRAL 🟡"

def leader_rank_score(row: List) -> float:
    _, fut_usd, pct24, pct4h, pct1h, *_ = row
    liq = math.log10(max(fut_usd, 1.0))
    mom = (abs(pct1h) * 1.2) + (abs(pct4h) * 1.0) + (abs(pct24) * 0.4)
    return liq + mom

def build_market_leaders(universe: List[List]) -> List[List]:
    leaders = [r for r in universe if r[1] >= LEADERS_MIN_FUT_USD]
    leaders.sort(key=leader_rank_score, reverse=True)
    return leaders[:LEADERS_TOP_N]

def scan_strong_movers(best_fut: Dict[str, MarketVol]) -> List[Tuple[str, float, float]]:
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

def score_long(row: List) -> float:
    _, fut_usd, pct24, pct4h, pct1h, pct15m, *_ = row
    score = 0.0
    if pct24 > 0: score += min(pct24 / 5.0, 3.0)
    if pct4h > 0: score += min(pct4h / 2.0, 3.0)
    if pct1h > 0: score += min(pct1h / 1.0, 2.0)
    if fut_usd > 10_000_000: score += 2.0
    elif fut_usd > 5_000_000: score += 1.0
    if pct15m > 0: score += 0.5
    return max(score, 0.0)

def score_short(row: List) -> float:
    _, fut_usd, pct24, pct4h, pct1h, pct15m, *_ = row
    score = 0.0
    if pct24 < 0: score += min(abs(pct24) / 5.0, 3.0)
    if pct4h < 0: score += min(abs(pct4h) / 2.0, 3.0)
    if pct1h < 0: score += min(abs(pct1h) / 1.0, 2.0)
    if fut_usd > 10_000_000: score += 2.0
    elif fut_usd > 5_000_000: score += 1.0
    if pct15m < 0: score += 0.5
    return max(score, 0.0)

def confidence_from_score(score: float) -> int:
    s = max(0.0, min(score, 10.0))
    return int(round((s / 10.0) * 100))

def classify_row(row: List) -> Tuple[bool, bool]:
    _, _, pct24, pct4h, pct1h, pct15m, *_ = row
    long_ok = False
    short_ok = False

    # Bias from 24h
    if pct24 > 5.0:
        long_ok = True
    elif pct24 < -5.0:
        short_ok = True
    else:
        if pct1h >= TRIGGER_1H_PCT and pct4h >= 0.0:
            long_ok = True
        if pct1h <= -TRIGGER_1H_PCT and pct4h <= 0.0:
            short_ok = True

    # 15m confirmation
    if CONFIRM_15M_SIGN:
        if long_ok and pct15m < 0:
            long_ok = False
        if short_ok and pct15m > 0:
            short_ok = False

    return long_ok, short_ok

def multi_tp_plan(entry: float, sl: float, side: str, confidence: int):
    if confidence < CONF_MULTI_TP_MIN:
        return None
    entry = float(entry); sl = float(sl)
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

def pick_best_trades(universe: List[List], top_n: int):
    candidates = []
    for r in universe:
        sym, _, _, _, _, _, last_price, _ = r
        if not last_price or last_price <= 0:
            continue

        long_ok, short_ok = classify_row(r)

        if long_ok:
            sc = score_long(r)
            conf = confidence_from_score(sc)
            if sc > 0 and conf >= 70:  # mild gate for signal quality
                entry = last_price
                sl = entry * (1.0 - SL_PCT)
                plan = multi_tp_plan(entry, sl, "BUY", conf)
                tp = (plan["tp2"] if plan else entry * (1.0 + TP_PCT))
                candidates.append(("BUY", sym, entry, tp, sl, conf))

        if short_ok:
            sc = score_short(r)
            conf = confidence_from_score(sc)
            if sc > 0 and conf >= 70:
                entry = last_price
                sl = entry * (1.0 + SL_PCT)
                plan = multi_tp_plan(entry, sl, "SELL", conf)
                tp = (plan["tp2"] if plan else entry * (1.0 - TP_PCT))
                candidates.append(("SELL", sym, entry, tp, sl, conf))

    candidates.sort(key=lambda x: x[5], reverse=True)
    return candidates[:top_n]


# =========================================================
# Formatting blocks
# =========================================================

def fmt_leaders(rows: List[List]) -> str:
    if not rows:
        return "*🔥 Market Leaders*: _None_\n"
    pretty = []
    for r in rows:
        base, fut_usd, pct24, pct4h, pct1h, *_ = r
        pretty.append([base, m_dollars(fut_usd), side_hint(pct4h, pct1h),
                       pct_with_emoji(pct24), pct_with_emoji(pct4h), pct_with_emoji(pct1h)])
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
        "*🚀 Strong Movers (24h — context only)*:\n"
        "```\n"
        + tabulate(pretty, headers=["SYM", "F(M)", "%24H"], tablefmt="github")
        + "\n```\n"
    )

def format_trade_setups(recs: List[Tuple]) -> str:
    if not recs:
        return "_No strong setups right now._"

    lines: List[str] = []
    for idx, (side, sym, entry, tp, sl, conf) in enumerate(recs, start=1):
        plan = multi_tp_plan(entry, sl, side, conf)
        if plan:
            lines.append(
                f"*Setup #{idx}* — {side} {sym} — Confidence {conf}/100 🔥\n"
                f"Entry {fmt_price(entry)} | SL {fmt_price(sl)}\n"
                f"TP plan: {plan['weights']}\n"
                f"TP1 {fmt_price(plan['tp1'])} | TP2 {fmt_price(plan['tp2'])}\n"
                f"{plan['runner_text']}\n"
            )
        else:
            lines.append(
                f"*Setup #{idx}* — {side} {sym} — Confidence {conf}/100\n"
                f"Entry {fmt_price(entry)} | SL {fmt_price(sl)} | TP {fmt_price(tp)}\n"
            )
    return "\n".join(lines).strip()

def snapshot_for_symbol(universe: List[List], sym: str) -> Optional[str]:
    for r in universe:
        if r[0] == sym:
            _, fut_usd, pct24, pct4h, pct1h, *_ = r
            bias = side_hint(pct4h, pct1h)
            return (
                f"Market Snapshot: F~{m_dollars(fut_usd)}M | "
                f"24H {pct_with_emoji(pct24)} | 4H {pct_with_emoji(pct4h)} | 1H {pct_with_emoji(pct1h)} | "
                f"{bias}"
            )
    return None

def fut_symbol_for_base(universe: List[List], sym: str) -> Optional[str]:
    for r in universe:
        if r[0] == sym:
            return r[7]
    return None


# =========================================================
# Email helpers
# =========================================================

def email_config_ok() -> bool:
    return all([EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FROM, EMAIL_TO])

def melbourne_is_sunday() -> bool:
    now = datetime.now(ZoneInfo("Australia/Melbourne"))
    return now.weekday() == 6

def send_email(subject: str, text_body: str, html_body: Optional[str] = None, inline_images: Optional[List[Tuple[str, str]]] = None) -> bool:
    if not email_config_ok():
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO

        msg.set_content(text_body)

        if html_body:
            msg.add_alternative(html_body, subtype="html")
            if inline_images:
                for cid, path in inline_images:
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                        msg.get_payload()[-1].add_related(
                            data,
                            maintype="image",
                            subtype="png",
                            cid=f"<{cid}>",
                            filename=os.path.basename(path),
                        )
                    except Exception:
                        logging.exception("inline image attach failed")

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
# User Guide (EN only)
# =========================================================

HELP_TEXT = f"""📘 {BOT_NAME} — User Guide (English)

This is a futures-only crypto scanner + day-trade setup assistant.

Day-trader design:
• Trigger: 1H momentum
• Confirmation: 15m direction confirmation
• Targets: Smaller SL/TP for faster resolution (aiming for < 24h)
• Signals expire after {SIGNAL_EXPIRY_HOURS}h if not opened

Leverage & Notional:
• Qty = how many coins/contracts you trade
• Notional = Qty × Entry (position size in USD)
• Effective Leverage = Notional / Equity
• Margin guide: Margin ≈ Notional / Leverage
If Effective Leverage is high (> {EFFECTIVE_LEV_WARN:.0f}x), you get a warning.

What this bot does NOT do:
• No auto-trading, no exchange connection, no profit guarantees.

Commands:
• /start
• /help
• /screen — Leaders + Movers + Setups
• /session [both|london|ny|off]
• /diag
• /notify_on /notify_off /notify

Risk & Journal:
• /equity <amount>
• /limits <maxTradesDay> <maxDailyRiskUSD> <maxOpenRiskUSD>
• /status
• /risk <SYMBOL> <RISK> [ENTRY SL]
  - Bot signal sizing: /risk BTC 1%
  - Manual sizing (warning): /risk BTC 1% 65000 64000
• /open
• /closepnl <SYMBOL> <PNL>
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
        "• /limits 3 150 200   (example)\n"
        "• /risk BTC 1%\n"
        "• /risk BTC 1% 65000 64000 (manual)\n"
        "• /open\n"
        "• /closepnl BTC +23.5\n"
        "• /status\n\n"
        "Email alerts (optional): /notify_on /notify_off /notify\n"
    )
    await update.message.reply_text(msg)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def diag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    last_ts = db_get_last_email_ts()
    last_keys = db_get_last_email_keys()
    last_ts_str = datetime.fromtimestamp(last_ts, tz=ZoneInfo("Australia/Melbourne")).isoformat() if last_ts else "none"
    msg = (
        f"*Diag*\n"
        f"- session_mode: `{get_session_mode()}` in_window=`{is_in_trading_window(get_session_mode())}`\n"
        f"- interval: `{CHECK_INTERVAL_MIN}m` email_min_interval: `{EMAIL_MIN_INTERVAL_SEC//60}m`\n"
        f"- email_configured: `{email_config_ok()}` notify_on=`{NOTIFY_ON}` sunday_block=`{melbourne_is_sunday()}`\n"
        f"- email_conf_gate: `{EMAIL_CONF_GATE}` symbol_cooldown_h: `{EMAIL_SYMBOL_COOLDOWN_SEC/3600:.0f}`\n"
        f"- last_email_ts: `{last_ts_str}`\n"
        f"- last_email_keys_count: `{len(last_keys)}`\n"
        f"- charts_enabled: `{CHARTS_ENABLED}`\n"
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
        f"Email alerts: {'ON' if NOTIFY_ON else 'OFF'} | configured={'YES' if email_config_ok() else 'NO'} | min_interval={EMAIL_MIN_INTERVAL_SEC//60}m"
    )

async def equity_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    if not context.args:
        mt, md, mo = get_limits()
        await update.message.reply_text(
            f"Limits: trades/day={mt}, daily=${md:,.2f}, open=${mo:,.2f}\nSet: /limits 3 150 200"
        )
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
        await update.message.reply_text("Usage: /limits 3 150 200")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        f"Signal expiry: {SIGNAL_EXPIRY_HOURS}h\n"
    )
    await update.message.reply_text((warn or "") + "```\n" + text + "```", parse_mode=ParseMode.MARKDOWN)

async def open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT symbol, side, entry, sl, tp, confidence, risk_usd, opened_ts, ts, origin
            FROM signals WHERE status='OPEN'
            ORDER BY COALESCE(opened_ts, ts) DESC
        """).fetchall()
    if not rows:
        await update.message.reply_text("No OPEN positions in ledger.")
        return

    lines = ["OPEN POSITIONS", "----------------------------"]
    t = now_ts()
    for r in rows:
        opened = int(r["opened_ts"] or r["ts"] or t)
        age_hr = (t - opened) / 3600.0
        origin = (r["origin"] or "BOT").upper()
        tag = "MANUAL" if origin == "MANUAL" else "BOT"
        note = "⚠️ aged>24h" if age_hr >= SIGNAL_EXPIRY_HOURS else ""
        lines.append(
            f"{r['symbol']:6} {r['side']:4} [{tag}] "
            f"conf {int(r['confidence'] or 0):3d} "
            f"E:{fmt_price(float(r['entry'] or 0))} SL:{fmt_price(float(r['sl'] or 0))} "
            f"TP:{fmt_price(float(r['tp'] or 0))} risk:${float(r['risk_usd'] or 0):,.2f} "
            f"age:{age_hr:.1f}h {note}"
        )
    lines.append("")
    lines.append("Close: /closepnl BTC +23.5   (or -10)")
    await update.message.reply_text("```\n" + "\n".join(lines) + "\n```", parse_mode=ParseMode.MARKDOWN)

async def closepnl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /closepnl BTC +23.5")
        return
    sym = safe_token(context.args[0])
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
    """
    /risk BTC 1%            -> sizes latest BOT SIGNAL (if exists)
    /risk BTC 1% 65000 64000 -> manual sizing (warning)
    """
    if len(context.args) < 2:
        await update.message.reply_text("Usage:\n/risk BTC 1%\n/risk BTC 1% 65000 64000 (manual)")
        return

    sym = safe_token(context.args[0])
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

    # limits
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

    # manual if no SIGNAL exists
    if (not sig) or (sig["status"] != "SIGNAL"):
        if len(context.args) < 4:
            await update.message.reply_text(
                f"⚠️ {sym} is NOT a {BOT_NAME} signal.\n"
                f"Risk management only — trade responsibility is yours.\n\n"
                f"To size it manually, provide ENTRY and SL:\n"
                f"/risk {sym} <RISK> <ENTRY> <SL>\n"
                f"Example: /risk {sym} 1% 65000 64000"
            )
            return

        try:
            entry = float(context.args[2].replace(",", ""))
            sl = float(context.args[3].replace(",", ""))
        except Exception:
            await update.message.reply_text("ENTRY and SL must be numbers. Example: /risk BTC 1% 65000 64000")
            return

        if entry <= 0 or sl <= 0 or entry == sl:
            await update.message.reply_text("Invalid ENTRY/SL.")
            return

        side = "BUY" if sl < entry else "SELL"
        tp = entry * (1.0 + TP_PCT) if side == "BUY" else entry * (1.0 - TP_PCT)

        stop_dist = abs(entry - sl)
        qty = risk_usd / stop_dist
        notional = qty * entry

        await asyncio.to_thread(db_open_manual, sym, side, entry, sl, tp, risk_usd)

        with db_conn() as conn:
            conn.execute(
                "INSERT INTO executions(ts, symbol, side, risk_usd) VALUES(?,?,?,?)",
                (now_ts(), sym, side, float(risk_usd))
            )
            conn.commit()

        lev_block = leverage_guide_block(notional, eq)

        await update.message.reply_text(
            "⚠️ Manual position (NOT a PulseFutures signal)\n"
            "Risk management only — trade responsibility is yours.\n\n"
            "```\n"
            f"{sym} {side} [MANUAL]\n"
            f"Entry {fmt_price(entry)} | SL {fmt_price(sl)} | TP {fmt_price(tp)}\n"
            f"Risk ${risk_usd:,.2f} -> Qty {qty:.6g} | Notional ${notional:,.2f}\n"
            f"{lev_block}\n"
            "Close with /closepnl SYMBOL +/-PnL\n"
            "```",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    # open from BOT signal (not expired)
    entry = float(sig["entry"] or 0)
    sl = float(sig["sl"] or 0)
    if not entry or not sl or entry == sl:
        await update.message.reply_text("Signal missing entry/SL. Run /screen again.")
        return

    stop_dist = abs(entry - sl)
    qty = risk_usd / stop_dist
    notional = qty * entry

    await asyncio.to_thread(db_open_from_signal, int(sig["id"]), risk_usd)

    with db_conn() as conn:
        conn.execute(
            "INSERT INTO executions(ts, symbol, side, risk_usd) VALUES(?,?,?,?)",
            (now_ts(), sym, sig["side"], float(risk_usd))
        )
        conn.commit()

    lev_block = leverage_guide_block(notional, eq)

    await update.message.reply_text(
        "```\n"
        f"{sym} {sig['side']} [BOT]\n"
        f"Entry {fmt_price(entry)} | SL {fmt_price(sl)} | TP {fmt_price(float(sig['tp'] or 0))}\n"
        f"Risk ${risk_usd:,.2f} -> Qty {qty:.6g} | Notional ${notional:,.2f}\n"
        f"{lev_block}\n"
        "Close with /closepnl SYMBOL +/-PnL\n"
        "```",
        parse_mode=ParseMode.MARKDOWN
    )

async def screen_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # expire old signals before new ones
        await asyncio.to_thread(db_expire_old_signals)

        PCT_CACHE.clear()
        best_fut, raw_fut = await asyncio.to_thread(load_best_futures)
        universe = await asyncio.to_thread(build_universe, best_fut)

        leaders = await asyncio.to_thread(build_market_leaders, universe)
        movers = await asyncio.to_thread(scan_strong_movers, best_fut)
        recs = await asyncio.to_thread(pick_best_trades, universe, SETUPS_TOP_N_SCREEN)

        # store setups only if in window
        if is_in_trading_window(get_session_mode()) and recs:
            await asyncio.to_thread(db_store_setups_as_signals, recs)

        warn = trading_window_warning() or ""

        msg = ""
        msg += warn or ""
        msg += "ℹ️ Market Leaders show where liquidity and momentum are concentrated right now.\n\n"
        msg += fmt_leaders(leaders)
        msg += "\n"

        msg += "ℹ️ Strong Movers are for market context only — not trade signals.\n\n"
        msg += fmt_movers(movers)
        msg += "\n"

        msg += "ℹ️ Trade Setups are the only actionable ideas (entry, SL, TP).\n\n"
        msg += f"*🎯 Trade Setups (Top {SETUPS_TOP_N_SCREEN})*\n\n"
        msg += format_trade_setups(recs)

        msg += f"\n\n`tickers: fut={raw_fut}`"

        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

        # optional charts to telegram
        if CHARTS_ENABLED and recs:
            for idx, (side, sym, entry, tp, sl, conf) in enumerate(recs, start=1):
                fut_sym = fut_symbol_for_base(universe, sym)
                if not fut_sym:
                    continue
                out_path = f"/tmp/{BOT_NAME}_{sym}_{idx}.png"
                title = f"{sym} {side} | Conf {conf}/100 | 1H snapshot"
                try:
                    await asyncio.to_thread(make_chart_png, fut_sym, title, out_path, "1h", 48)
                    if os.path.exists(out_path):
                        await update.message.reply_photo(
                            photo=open(out_path, "rb"),
                            caption=f"Setup #{idx}: {side} {sym} — Conf {conf}/100",
                        )
                except Exception:
                    logging.exception("chart send failed")

    except Exception as e:
        logging.exception("screen_cmd error")
        await update.message.reply_text(f"Error: {e}")

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    token = safe_token(text)
    if len(token) < 2:
        return
    sig = await asyncio.to_thread(db_get_latest_signal, token)
    if not sig:
        await update.message.reply_text("No stored signal/position. Run /screen first (or use /risk SYMBOL RISK ENTRY SL).")
        return

    origin = (sig["origin"] or "BOT").upper()
    tag = "MANUAL" if origin == "MANUAL" else "BOT"

    await update.message.reply_text(
        "```\n"
        f"{sig['symbol']} ({sig['status']}) {sig['side']} [{tag}]\n"
        f"Entry {fmt_price(float(sig['entry'] or 0))} | SL {fmt_price(float(sig['sl'] or 0))} | TP {fmt_price(float(sig['tp'] or 0))}\n"
        f"Confidence {int(sig['confidence'] or 0)}/100\n"
        "```",
        parse_mode=ParseMode.MARKDOWN
    )


# =========================================================
# Email Alert Job (optimized)
# =========================================================

async def alert_job(context: ContextTypes.DEFAULT_TYPE):
    try:
        if not NOTIFY_ON:
            return
        if not email_config_ok():
            return

        # no emails on Sundays (Melbourne)
        if melbourne_is_sunday():
            return

        # Only in trading window
        if not is_in_trading_window(get_session_mode()):
            return

        # Expire old signals
        await asyncio.to_thread(db_expire_old_signals)

        now = now_ts()

        # DB-persisted min interval
        last_ts = db_get_last_email_ts()
        if now - last_ts < EMAIL_MIN_INTERVAL_SEC:
            return

        PCT_CACHE.clear()

        best_fut, _ = await asyncio.to_thread(load_best_futures)
        universe = await asyncio.to_thread(build_universe, best_fut)

        recs = await asyncio.to_thread(pick_best_trades, universe, SETUPS_TOP_N_EMAIL)
        movers = await asyncio.to_thread(scan_strong_movers, best_fut)

        # Gate: at least one setup with high confidence
        strong_recs = [r for r in recs if int(r[5]) >= EMAIL_CONF_GATE]
        if not strong_recs:
            return

        # Per-symbol cooldown
        filtered_recs = []
        for side, sym, entry, tp, sl, conf in strong_recs:
            last_sym_ts = db_get_symbol_last_emailed(sym)
            if now - last_sym_ts < EMAIL_SYMBOL_COOLDOWN_SEC:
                continue
            filtered_recs.append((side, sym, entry, tp, sl, conf))

        if not filtered_recs:
            return

        # Newness check vs last email keys (persisted)
        keys = set()
        keys |= {f"S:{side}:{sym}:{int(conf)}" for side, sym, _, _, _, conf in filtered_recs}
        # movers are context; include only symbols (not required to gate)
        keys |= {f"M:{sym}" for sym, *_ in movers[:MOVERS_TOP_N]}

        last_keys = db_get_last_email_keys()
        if keys == last_keys:
            return

        # Build email body with snapshots
        text_lines = [f"{BOT_NAME} Alert", "", "Trade Setups (Confidence-based):"]
        html_parts = [f"<h2>{BOT_NAME} Alert</h2>", "<h3>Trade Setups</h3>"]
        inline_images: List[Tuple[str, str]] = []

        for idx, (side, sym, entry, tp, sl, conf) in enumerate(filtered_recs, start=1):
            text_lines.append(f"\nSetup #{idx}: {side} {sym} — Confidence {conf}/100")
            text_lines.append(f"Entry {fmt_price(entry)} | SL {fmt_price(sl)} | TP {fmt_price(tp)}")

            snap = snapshot_for_symbol(universe, sym)
            if snap:
                text_lines.append(snap)

            plan = multi_tp_plan(entry, sl, side, conf)
            if plan:
                text_lines.append(f"Multi-TP: {plan['weights']}")
                text_lines.append(f"TP1 {fmt_price(plan['tp1'])} | TP2 {fmt_price(plan['tp2'])}")
                text_lines.append(plan["runner_text"])

            html_block = f"""
            <div style="margin-bottom:14px;padding:10px;border:1px solid #ddd;border-radius:10px;">
              <b>Setup #{idx}</b>: {side} {sym} — <b>Confidence {conf}/100</b><br/>
              Entry <b>{fmt_price(entry)}</b> | SL <b>{fmt_price(sl)}</b> | TP <b>{fmt_price(tp)}</b><br/>
            """
            if snap:
                html_block += f"<div style='margin-top:6px;font-size:12px;color:#333;'>{snap}</div>"

            if plan:
                html_block += f"<div style='margin-top:6px;font-size:12px;'>Multi-TP: {plan['weights']}<br/>TP1 {fmt_price(plan['tp1'])} | TP2 {fmt_price(plan['tp2'])}<br/>{plan['runner_text']}</div>"

            if CHARTS_ENABLED:
                fut_sym = fut_symbol_for_base(universe, sym)
                if fut_sym:
                    out_path = f"/tmp/{BOT_NAME}_email_{sym}_{idx}.png"
                    cid = f"chart_{sym}_{idx}"
                    try:
                        await asyncio.to_thread(make_chart_png, fut_sym, f"{sym} 1H snapshot", out_path, "1h", 48)
                        if os.path.exists(out_path):
                            inline_images.append((cid, out_path))
                            html_block += f"<div style='margin-top:10px;'><img src='cid:{cid}' style='max-width:520px;border-radius:10px;'/></div>"
                    except Exception:
                        logging.exception("email chart failed")

            html_block += "</div>"
            html_parts.append(html_block)

        # Context movers (optional section, no gating)
        if movers:
            mv_lines = "\n".join([f"{sym} | F~{m_dollars(fusd)}M | {pct_with_emoji(p24)}"
                                  for sym, fusd, p24 in movers[:MOVERS_TOP_N]])
            text_lines.append("\nStrong Movers (24h — context only):")
            text_lines.append(mv_lines)
            html_parts.append("<h3>Strong Movers (24h — context only)</h3>")
            html_parts.append("<pre style='background:#f6f6f6;padding:10px;border-radius:10px;'>"
                              + mv_lines + "</pre>")

        text_body = "\n".join(text_lines).strip()
        html_body = "\n".join(html_parts).strip()

        subject = f"{BOT_NAME} Alert: High-Confidence Setups"

        ok = send_email(subject, text_body, html_body=html_body, inline_images=inline_images)
        if ok:
            # persist state
            db_set_last_email_ts(now)
            db_set_last_email_keys(keys)
            # update per-symbol timestamps
            for _, sym, *_ in filtered_recs:
                db_set_symbol_last_emailed(sym, now)

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
# FastAPI health
# =========================================================

app_api = FastAPI()

@app_api.get("/health")
async def health():
    return {"ok": True, "name": BOT_NAME}

def run_api_server():
    import uvicorn
    uvicorn.run(app_api, host="0.0.0.0", port=PORT, log_level="info")


# =========================================================
# Bot runner (Render + Py3.11 thread safe)
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

    app.add_handler(CommandHandler("notify_on", notify_on_cmd))
    app.add_handler(CommandHandler("notify_off", notify_off_cmd))
    app.add_handler(CommandHandler("notify", notify_cmd))

    app.add_handler(CommandHandler("equity", equity_cmd))
    app.add_handler(CommandHandler("limits", limits_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("risk", risk_cmd))
    app.add_handler(CommandHandler("open", open_cmd))
    app.add_handler(CommandHandler("closepnl", closepnl_cmd))

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

    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    run_api_server()


if __name__ == "__main__":
    main()
