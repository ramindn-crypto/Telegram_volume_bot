#!/usr/bin/env python3
"""
PulseFutures — Bybit Futures (Swap) Screener + TradeSetup + Risk Ledger

(Identical to your current working version, with ONLY ONE change:
Exchange switched from CoinEx to Bybit.)

Env vars required:
- TELEGRAM_TOKEN

Optional email env vars:
- EMAIL_ENABLED=true/false (default false)
- EMAIL_HOST, EMAIL_PORT (465), EMAIL_USER, EMAIL_PASS, EMAIL_FROM, EMAIL_TO
Optional guard:
- SUNDAY_EMAILS=true/false (default false)
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
from email.message import EmailMessage
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
from zoneinfo import ZoneInfo

import ccxt
from tabulate import tabulate
from telegram import Update
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

# ✅ ONLY CHANGE: CoinEx -> Bybit
EXCHANGE_ID = "bybit"

TOKEN = os.environ.get("TELEGRAM_TOKEN")

# Futures-only
DEFAULT_TYPE = "swap"

# Universe / Display
LEADERS_N = 10
MOVERS_N = 10
SETUPS_N = 3

# Movers thresholds
MOVER_VOL_USD_MIN = 1_000_000
MOVER_UP_24H_MIN = 10.0
MOVER_DN_24H_MAX = -10.0

# Setups thresholds (tunable later)
TRIGGER_1H_ABS_MIN = 2.0          # trigger when |1H| >= 2%
CONFIRM_15M_ABS_MIN = 0.6         # confirm when |15m| >= 0.6%
ALIGN_4H_MIN = 0.0                # prefer alignment with 4H (>=0 for long, <=0 for short)

# Risk defaults
DEFAULT_EQUITY = 1000.0
DEFAULT_RISK_PCT = 1.0            # % of equity per trade risk
DEFAULT_MAX_TRADES_DAY = 3
DEFAULT_DAILY_RISK_CAP = 50.0     # USD
DEFAULT_OPEN_RISK_CAP = 75.0      # USD

# Stop/TP sizing (percent-based for now; can move to ATR later)
SL_PCT = 3.0
TP_PCT = 6.0

# Multi-TP (only when Conf >= 75)
MULTI_TP_MIN_CONF = 75
TP1_PCT_ALLOC = 40
TP2_PCT_ALLOC = 40
RUNNER_PCT_ALLOC = 20

# Email / Alerts
EMAIL_ENABLED_DEFAULT = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST = os.environ.get("EMAIL_HOST", "")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "465"))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", EMAIL_USER)
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_USER)

NOTIFY_ON_DEFAULT = EMAIL_ENABLED_DEFAULT
EMAIL_MIN_INTERVAL_SEC = 60 * 60  # 60 minutes

# Trading window guard (default: London + New York sessions)
GUARD_ENABLED = True
SUNDAY_EMAILS = os.environ.get("SUNDAY_EMAILS", "false").lower() == "true"

LONDON_TZ = ZoneInfo("Europe/London")
NY_TZ = ZoneInfo("America/New_York")

# Rough session windows (local times)
LONDON_START = (7, 0)
LONDON_END = (16, 0)
NY_START = (9, 0)
NY_END = (17, 0)

# Bot runtime
CHECK_INTERVAL_MIN = 5

# DB
DB_PATH = os.environ.get("DB_PATH", "pulsefutures.db")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pulsefutures")

# =========================
# DATA STRUCTURES
# =========================

STABLES = {"USDT", "USDC", "USD", "TUSD", "FDUSD", "DAI", "PYUSD"}

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
    symbol: str         # base symbol (e.g., "BTC")
    market_symbol: str  # ccxt symbol (e.g., "BTC/USDT:USDT")
    side: str           # "BUY" or "SELL"
    conf: int           # 0-100
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
    regime: str         # "LONG"/"SHORT"/"NEUTRAL"
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

    cur.execute("""
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
        daily_risk_used REAL NOT NULL
    )
    """)

    cur.execute("""
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
        status TEXT NOT NULL,      -- OPEN/CLOSED
        pnl REAL,
        closed_ts REAL,
        notes TEXT
    )
    """)

    cur.execute("""
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
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS meta (
        k TEXT PRIMARY KEY,
        v TEXT NOT NULL
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
        now_mel = datetime.now(ZoneInfo("Australia/Melbourne"))
        cur.execute("""
            INSERT INTO users (user_id, equity, risk_pct, max_trades_day, daily_risk_cap, open_risk_cap,
                              notify_on, tz, day_trade_count, day_trade_date, daily_risk_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, DEFAULT_EQUITY, DEFAULT_RISK_PCT, DEFAULT_MAX_TRADES_DAY,
            DEFAULT_DAILY_RISK_CAP, DEFAULT_OPEN_RISK_CAP,
            1 if NOTIFY_ON_DEFAULT else 0,
            "Australia/Melbourne",
            0, now_mel.date().isoformat(), 0.0
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

def reset_daily_counters_if_needed(user: dict) -> dict:
    tz = ZoneInfo(user.get("tz") or "Australia/Melbourne")
    today = datetime.now(tz).date().isoformat()
    if user.get("day_trade_date") != today:
        update_user(user["user_id"], day_trade_count=0, day_trade_date=today, daily_risk_used=0.0)
        user = get_user(user["user_id"])
    return user

def db_set_meta(k: str, v: str):
    con = db_connect()
    cur = con.cursor()
    cur.execute("INSERT INTO meta (k,v) VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))
    con.commit()
    con.close()

def db_get_meta(k: str, default: str = "") -> str:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT v FROM meta WHERE k=?", (k,))
    row = cur.fetchone()
    con.close()
    return row["v"] if row else default

def db_upsert_signal(setup: Setup):
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
    INSERT INTO signals (symbol, market_symbol, side, conf, entry, sl, tp, tp1, tp2,
                         fut_vol_usd, ch24, ch4, ch1, ch15, regime, created_ts)
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
    """, (
        setup.symbol, setup.market_symbol, setup.side, setup.conf,
        setup.entry, setup.sl, setup.tp, setup.tp1, setup.tp2,
        setup.fut_vol_usd, setup.ch24, setup.ch4, setup.ch1, setup.ch15,
        setup.regime, setup.created_ts
    ))
    con.commit()
    con.close()

def db_get_signal(symbol: str, ttl_sec: int = 24*3600) -> Optional[dict]:
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

def db_add_position(user_id: int, symbol: str, side: str, entry: float, sl: float, tp: float,
                    risk_usd: float, qty: float, notional: float, conf: Optional[int], notes: str = ""):
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
    INSERT INTO positions (user_id, symbol, side, entry, sl, tp, risk_usd, qty, notional, conf,
                           created_ts, status, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
    """, (user_id, symbol, side, entry, sl, tp, risk_usd, qty, notional, conf, time.time(), notes))
    con.commit()
    con.close()

def db_get_open_positions(user_id: int) -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
    SELECT * FROM positions
    WHERE user_id=? AND status='OPEN'
    ORDER BY created_ts ASC
    """, (user_id,))
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]

def db_close_position(user_id: int, symbol: str, pnl: float) -> Optional[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
    SELECT * FROM positions
    WHERE user_id=? AND status='OPEN' AND symbol=?
    ORDER BY created_ts ASC
    LIMIT 1
    """, (user_id, symbol.upper()))
    row = cur.fetchone()
    if not row:
        con.close()
        return None
    pos = dict(row)
    cur.execute("""
    UPDATE positions
    SET status='CLOSED', pnl=?, closed_ts=?
    WHERE id=?
    """, (pnl, time.time(), pos["id"]))
    con.commit()
    con.close()
    pos["pnl"] = pnl
    return pos

# =========================
# EXCHANGE HELPERS
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

def fetch_ohlcv_pct(symbol: str, timeframe: str, bars: int) -> float:
    ex = build_exchange()
    try:
        ex.load_markets()
        candles = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=bars + 1)
        if not candles or len(candles) < bars + 1:
            return 0.0
        closes = [c[4] for c in candles][- (bars + 1):]
        if not closes or not closes[0]:
            return 0.0
        return (closes[-1] - closes[0]) / closes[0] * 100.0
    except Exception:
        logger.exception("fetch_ohlcv_pct failed: %s %s", symbol, timeframe)
        return 0.0

def pct_4h_from_1h(symbol: str) -> float:
    return fetch_ohlcv_pct(symbol, "1h", 4)

def pct_1h_from_1h(symbol: str) -> float:
    return fetch_ohlcv_pct(symbol, "1h", 1)

def pct_15m_from_15m(symbol: str) -> float:
    return fetch_ohlcv_pct(symbol, "15m", 1)

# =========================
# FORMATTING HELPERS
# =========================

def fmt_price(x: float) -> str:
    ax = abs(x)
    if ax >= 1000:
        return f"{x:.2f}"
    if ax >= 100:
        return f"{x:.3f}"
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

def pct_with_emoji(p: float) -> str:
    val = round(p)
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

# =========================
# SETUP ENGINE
# =========================

def compute_confidence(side: str, ch24: float, ch4: float, ch1: float, ch15: float, fut_vol_usd: float) -> int:
    score = 50.0

    def add_align(x: float, long: bool, w: float):
        nonlocal score
        if long:
            score += w if x > 0 else -w
        else:
            score += w if x < 0 else -w

    is_long = (side == "BUY")
    add_align(ch24, is_long, 12.0)
    add_align(ch4, is_long, 10.0)
    add_align(ch1, is_long, 8.0)
    add_align(ch15, is_long, 6.0)

    mag = min(abs(ch24)/2.0 + abs(ch4) + abs(ch1)*2.0 + abs(ch15)*2.0, 18.0)
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
    ch4 = pct_4h_from_1h(mv.symbol)
    ch1 = pct_1h_from_1h(mv.symbol)
    ch15 = pct_15m_from_15m(mv.symbol)

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

    if side == "BUY":
        sl = entry * (1 - SL_PCT/100.0)
        tp = entry * (1 + TP_PCT/100.0)
        tp1 = entry * (1 + (TP_PCT/2)/100.0)
        tp2 = tp
    else:
        sl = entry * (1 + SL_PCT/100.0)
        tp = entry * (1 - TP_PCT/100.0)
        tp1 = entry * (1 - (TP_PCT/2)/100.0)
        tp2 = tp

    conf = compute_confidence(side, ch24, ch4, ch1, ch15, fut_vol)
    reg = regime_label(ch24, ch4)

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
        created_ts=time.time()
    )

def pick_setups(best_fut: Dict[str, MarketVol]) -> List[Setup]:
    universe = sorted(best_fut.items(), key=lambda kv: usd_notional(kv[1]), reverse=True)[: max(50, LEADERS_N)]
    setups: List[Setup] = []
    for base, mv in universe:
        s = make_setup(base, mv)
        if not s:
            continue
        setups.append(s)
    setups.sort(key=lambda s: (s.conf, s.fut_vol_usd), reverse=True)
    return setups[:SETUPS_N]

# =========================
# TRADING WINDOW GUARD
# =========================

def _in_window_local(tz: ZoneInfo, start: Tuple[int,int], end: Tuple[int,int]) -> bool:
    now = datetime.now(tz)
    if now.weekday() == 6 and not SUNDAY_EMAILS:
        return False

    sh, sm = start
    eh, em = end
    start_m = sh*60 + sm
    end_m = eh*60 + em
    now_m = now.hour*60 + now.minute
    return start_m <= now_m <= end_m

def window_ok() -> bool:
    if not GUARD_ENABLED:
        return True
    return _in_window_local(LONDON_TZ, LONDON_START, LONDON_END) or _in_window_local(NY_TZ, NY_START, NY_END)

# =========================
# EMAIL HELPERS
# =========================

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

def format_setup_email_block(s: Setup) -> str:
    snapshot = (
        f"Market Snapshot: F~{fmt_money(s.fut_vol_usd)} | "
        f"24H {pct_with_emoji(s.ch24)} | 4H {pct_with_emoji(s.ch4)} | 1H {pct_with_emoji(s.ch1)} | {s.regime}"
    )

    lines = []
    lines.append(f"Setup: {s.side} {s.symbol} — Confidence {s.conf}/100")
    lines.append(f"Entry {fmt_price(s.entry)} | SL {fmt_price(s.sl)} | TP {fmt_price(s.tp)}")
    lines.append(snapshot)

    if s.conf >= MULTI_TP_MIN_CONF and s.tp1 and s.tp2:
        if s.side == "BUY":
            runner_hint = "trail EMA20 (1H) or last 15m swing-low"
        else:
            runner_hint = "trail EMA20 (1H) or last 15m swing-high"

        lines.append(
            f"TP1 ({TP1_PCT_ALLOC}%) {fmt_price(s.tp1)} | "
            f"TP2 ({TP2_PCT_ALLOC}%) {fmt_price(s.tp2)} | "
            f"Runner ({RUNNER_PCT_ALLOC}%) — {runner_hint}"
        )
    return "\n".join(lines)

# =========================
# RISK / POSITION SIZING
# =========================

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
# BOT TEXTS
# =========================

HELP_TEXT = """\
**PulseFutures — Commands**

**Market**
- /screen — Market Leaders + Movers + Strong Movers + Top Setups

**Risk & Trading**
- /equity <amount> — Set your equity (e.g., /equity 1000)
- /limits <maxTradesPerDay> <dailyRiskCapUSD> <openRiskCapUSD> — e.g. /limits 3 50 75
- /tradesetup <SYMBOL> — Create a TradeSetup using PulseFutures signal if available (no need to run /screen)
- /risk <SYMBOL> — Same as /tradesetup
- /open — Show open positions
- /closepnl <SYMBOL> <pnlUSD> — Close oldest open position for SYMBOL and update equity (e.g. /closepnl BTC +23.5)

**Alerts**
- /notify_on /notify_off — Email alerts (if configured)
- /diag — Diagnostics

**Notes**
- Futures-only setups (Bybit swap).
- If you TradeSetup a symbol that is not in current PulseFutures signals, you will get a warning (you can still use the risk sizing).
- This is not financial advice.
"""

# =========================
# SCREEN FORMATTERS
# =========================

def build_leaders_table(best_fut: Dict[str, MarketVol]) -> str:
    leaders = sorted(best_fut.items(), key=lambda kv: usd_notional(kv[1]), reverse=True)[:LEADERS_N]
    rows = []
    for base, mv in leaders:
        rows.append([
            base,
            fmt_money(usd_notional(mv)),
            pct_with_emoji(float(mv.percentage or 0.0)),
            fmt_price(float(mv.last or 0.0)),
        ])
    return "*Market Leaders (Top 10 by Futures Volume)*\n" + fmt_table(rows, ["SYM", "F Vol", "24H", "Last"])

def build_movers_tables(best_fut: Dict[str, MarketVol]) -> Tuple[str, str]:
    up = []
    dn = []
    for base, mv in best_fut.items():
        vol = usd_notional(mv)
        if vol < MOVER_VOL_USD_MIN:
            continue
        ch24 = float(mv.percentage or 0.0)
        if ch24 >= MOVER_UP_24H_MIN:
            up.append((base, vol, ch24, float(mv.last or 0.0)))
        if ch24 <= MOVER_DN_24H_MAX:
            dn.append((base, vol, ch24, float(mv.last or 0.0)))

    up.sort(key=lambda x: (x[2], x[1]), reverse=True)
    dn.sort(key=lambda x: (x[2], x[1]))

    up_rows = [[b, fmt_money(v), pct_with_emoji(c), fmt_price(px)] for b, v, c, px in up[:MOVERS_N]]
    dn_rows = [[b, fmt_money(v), pct_with_emoji(c), fmt_price(px)] for b, v, c, px in dn[:MOVERS_N]]

    up_txt = "*Movers (24H ≥ +10%, F vol ≥ 1M)*\n" + (fmt_table(up_rows, ["SYM", "F Vol", "24H", "Last"]) if up_rows else "_None_")
    dn_txt = "*Strong Movers (24H ≤ -10%, F vol ≥ 1M)*\n" + (fmt_table(dn_rows, ["SYM", "F Vol", "24H", "Last"]) if dn_rows else "_None_")
    return up_txt, dn_txt

def format_setups_for_screen(setups: List[Setup]) -> str:
    if not setups:
        return "*Top Setups (Trigger 1H, Confirm 15m)*\n_No strong setup right now._"
    lines = ["*Top Setups (Trigger 1H, Confirm 15m)*"]
    for i, s in enumerate(setups, 1):
        snapshot = (
            f"F~{fmt_money(s.fut_vol_usd)} | "
            f"24H {pct_with_emoji(s.ch24)} | 4H {pct_with_emoji(s.ch4)} | 1H {pct_with_emoji(s.ch1)} | {s.regime}"
        )
        lines.append(
            f"Setup #{i}: {s.side} {s.symbol} — Confidence {s.conf}/100\n"
            f"Entry {fmt_price(s.entry)} | SL {fmt_price(s.sl)} | TP {fmt_price(s.tp)}\n"
            f"{snapshot}"
        )
        if s.conf >= MULTI_TP_MIN_CONF and s.tp1 and s.tp2:
            if s.side == "BUY":
                runner_hint = "trail EMA20 (1H) or last 15m swing-low"
            else:
                runner_hint = "trail EMA20 (1H) or last 15m swing-high"
            lines.append(
                f"TP1 ({TP1_PCT_ALLOC}%) {fmt_price(s.tp1)} | "
                f"TP2 ({TP2_PCT_ALLOC}%) {fmt_price(s.tp2)} | "
                f"Runner ({RUNNER_PCT_ALLOC}%) — {runner_hint}"
            )
        lines.append("")
    return "\n".join(lines).strip()

# =========================
# TELEGRAM HANDLERS
# =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)

async def diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = get_user(update.effective_user.id)
    user = reset_daily_counters_if_needed(user)
    notify = "ON" if user["notify_on"] else "OFF"
    msg = (
        f"*Diag*\n"
        f"- Futures: Bybit swap\n"
        f"- Email notify: {notify}\n"
        f"- Guard: {'ON' if GUARD_ENABLED else 'OFF'} (London+NY)\n"
        f"- Sunday emails: {'YES' if SUNDAY_EMAILS else 'NO'}\n"
        f"- Email interval: {EMAIL_MIN_INTERVAL_SEC//60} min\n"
        f"- Equity: ${user['equity']:.2f}\n"
        f"- Risk%: {user['risk_pct']:.2f}%\n"
        f"- Max trades/day: {user['max_trades_day']}\n"
        f"- Daily risk used/cap: ${user['daily_risk_used']:.2f}/${user['daily_risk_cap']:.2f}\n"
        f"- Open risk cap: ${user['open_risk_cap']:.2f}\n"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def notify_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    update_user(uid, notify_on=1)
    await update.message.reply_text("Email alerts: ON")

async def notify_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    update_user(uid, notify_on=0)
    await update.message.reply_text("Email alerts: OFF")

async def equity_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args:
        user = get_user(uid)
        await update.message.reply_text(f"Equity: ${user['equity']:.2f}")
        return
    try:
        eq = float(context.args[0])
        if eq <= 0:
            raise ValueError()
        update_user(uid, equity=eq)
        await update.message.reply_text(f"Equity updated: ${eq:.2f}")
    except Exception:
        await update.message.reply_text("Usage: /equity 1000")

async def limits_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if len(context.args) != 3:
        user = get_user(uid)
        await update.message.reply_text(
            f"Usage: /limits <maxTradesDay> <dailyRiskCapUSD> <openRiskCapUSD>\n"
            f"Current: {user['max_trades_day']} | ${user['daily_risk_cap']:.2f} | ${user['open_risk_cap']:.2f}"
        )
        return
    try:
        max_trades = int(context.args[0])
        daily_cap = float(context.args[1])
        open_cap = float(context.args[2])
        if max_trades < 1 or daily_cap < 0 or open_cap < 0:
            raise ValueError()
        update_user(uid, max_trades_day=max_trades, daily_risk_cap=daily_cap, open_risk_cap=open_cap)
        await update.message.reply_text(f"Limits updated: max/day={max_trades}, daily cap=${daily_cap:.2f}, open cap=${open_cap:.2f}")
    except Exception:
        await update.message.reply_text("Usage: /limits 3 50 75")

async def screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    best_fut = await asyncio.to_thread(fetch_futures_tickers)
    if not best_fut:
        await update.message.reply_text("Error: could not fetch futures tickers.")
        return

    leaders_txt = build_leaders_table(best_fut)
    movers_up, movers_dn = build_movers_tables(best_fut)

    setups = await asyncio.to_thread(pick_setups, best_fut)
    for s in setups:
        db_upsert_signal(s)

    setups_txt = format_setups_for_screen(setups)

    msg = (
        leaders_txt
        + "\n\n"
        + movers_up
        + "\n\n"
        + movers_dn
        + "\n\n"
        + setups_txt
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

def parse_symbol_from_args(args: List[str]) -> Optional[str]:
    if not args:
        return None
    token = re.sub(r"[^A-Za-z0-9]", "", args[0]).upper().lstrip("$")
    return token if token else None

async def tradesetup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    user = reset_daily_counters_if_needed(user)

    sym = parse_symbol_from_args(context.args)
    if not sym:
        await update.message.reply_text("Usage: /tradesetup BTC")
        return

    sig = db_get_signal(sym)

    warning = ""
    if not sig:
        warning = f"⚠️ {sym} is not in current PulseFutures signals. You can still use TradeSetup for risk sizing, but responsibility is yours.\n\n"
        best_fut = await asyncio.to_thread(fetch_futures_tickers)
        mv = best_fut.get(sym)
        if not mv:
            await update.message.reply_text(f"{warning}Could not find {sym} on Bybit futures.")
            return
        ch24 = float(mv.percentage or 0.0)
        ch4 = await asyncio.to_thread(pct_4h_from_1h, mv.symbol)
        ch1 = await asyncio.to_thread(pct_1h_from_1h, mv.symbol)
        ch15 = await asyncio.to_thread(pct_15m_from_15m, mv.symbol)
        side = "BUY" if ch1 >= 0 else "SELL"
        entry = float(mv.last or 0.0)
        if entry <= 0:
            await update.message.reply_text(f"{warning}Invalid price for {sym}.")
            return
        if side == "BUY":
            sl = entry * (1 - SL_PCT/100.0)
            tp = entry * (1 + TP_PCT/100.0)
        else:
            sl = entry * (1 + SL_PCT/100.0)
            tp = entry * (1 - TP_PCT/100.0)
        conf = compute_confidence(side, ch24, ch4, ch1, ch15, usd_notional(mv))
        sig = {
            "symbol": sym,
            "market_symbol": mv.symbol,
            "side": side,
            "conf": conf,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "tp1": None,
            "tp2": None,
            "fut_vol_usd": usd_notional(mv),
            "ch24": ch24,
            "ch4": ch4,
            "ch1": ch1,
            "ch15": ch15,
            "regime": regime_label(ch24, ch4),
        }

    open_positions = db_get_open_positions(uid)
    open_risk = sum(float(p["risk_usd"]) for p in open_positions)
    if open_risk >= float(user["open_risk_cap"]):
        await update.message.reply_text("❌ Open risk cap reached. Close some positions or increase your open risk cap.")
        return

    if int(user["day_trade_count"]) >= int(user["max_trades_day"]):
        await update.message.reply_text("❌ Max trades per day reached. Try again tomorrow or increase your daily limit.")
        return

    equity = float(user["equity"])
    risk_pct = float(user["risk_pct"])
    risk_by_pct = equity * (risk_pct / 100.0)

    daily_remaining = float(user["daily_risk_cap"]) - float(user["daily_risk_used"])
    open_remaining = float(user["open_risk_cap"]) - open_risk

    risk_usd = min(risk_by_pct, daily_remaining, open_remaining)
    if risk_usd <= 0:
        await update.message.reply_text("❌ No risk budget remaining (daily cap or open cap).")
        return

    entry = float(sig["entry"])
    sl = float(sig["sl"])
    tp = float(sig["tp"])
    side = str(sig["side"])
    conf = int(sig.get("conf") or 0)

    qty, notional = calc_position_size(entry, sl, risk_usd)
    if qty <= 0 or notional <= 0:
        await update.message.reply_text("Could not compute position size (invalid entry/SL).")
        return

    lev = suggested_leverage(notional, equity)

    notes = "PulseFutures signal" if db_get_signal(sym) else "Manual (not in signals)"
    db_add_position(uid, sym, side, entry, sl, tp, risk_usd, qty, notional, conf, notes=notes)

    update_user(uid,
                day_trade_count=int(user["day_trade_count"]) + 1,
                daily_risk_used=float(user["daily_risk_used"]) + risk_usd)

    snapshot = (
        f"F~{fmt_money(float(sig.get('fut_vol_usd', 0)))} | "
        f"24H {pct_with_emoji(float(sig.get('ch24', 0)))} | "
        f"4H {pct_with_emoji(float(sig.get('ch4', 0)))} | "
        f"1H {pct_with_emoji(float(sig.get('ch1', 0)))} | "
        f"15m {pct_with_emoji(float(sig.get('ch15', 0)))} | {sig.get('regime','')}"
    )

    msg = (
        warning
        + f"✅ TradeSetup: {side} {sym} — Confidence {conf}/100\n"
        + f"Entry {fmt_price(entry)} | SL {fmt_price(sl)} | TP {fmt_price(tp)}\n"
        + f"{snapshot}\n\n"
        + f"Risk: ${risk_usd:.2f} ({risk_pct:.2f}% of equity cap)\n"
        + f"Qty: {qty:.6g} | Notional: ${notional:.2f}\n"
        + f"Suggested Leverage: {lev}x\n\n"
        + f"Use /open to view positions. Close with: /closepnl {sym} +10 (or -10)"
    )

    await update.message.reply_text(msg)

async def risk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await tradesetup(update, context)

async def open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    user = reset_daily_counters_if_needed(user)

    pos = db_get_open_positions(uid)
    if not pos:
        await update.message.reply_text("No open positions.")
        return

    lines = [f"*Open Positions* (Equity ${user['equity']:.2f})\n"]
    for i, p in enumerate(pos, 1):
        lines.append(
            f"#{i} {p['symbol']} {p['side']} | Entry {fmt_price(float(p['entry']))} | "
            f"SL {fmt_price(float(p['sl']))} | TP {fmt_price(float(p['tp']))} | Conf {p.get('conf') or '-'}"
        )
        lines.append(
            f"Risk ${float(p['risk_usd']):.2f} | Qty {float(p['qty']):.6g} | Notional ${float(p['notional']):.2f} | {p.get('notes') or ''}"
        )
        lines.append("")

    await update.message.reply_text("\n".join(lines).strip(), parse_mode=ParseMode.MARKDOWN)

async def closepnl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /closepnl BTC +23.5")
        return
    sym = parse_symbol_from_args([context.args[0]])
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

    await update.message.reply_text(
        f"✅ Closed {sym} ({pos['side']}). PnL: {pnl:+.2f}\nEquity updated: ${new_eq:.2f}"
    )

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    token = re.sub(r"[^A-Za-z0-9$]", "", text).upper().lstrip("$")
    if len(token) < 2:
        return

    sig = db_get_signal(token)
    if sig:
        msg = (
            f"{token} in PulseFutures signals:\n"
            f"{sig['side']} — Confidence {sig['conf']}/100\n"
            f"Entry {fmt_price(float(sig['entry']))} | SL {fmt_price(float(sig['sl']))} | TP {fmt_price(float(sig['tp']))}\n"
            f"Snapshot: F~{fmt_money(float(sig['fut_vol_usd']))} | "
            f"24H {pct_with_emoji(float(sig['ch24']))} | 4H {pct_with_emoji(float(sig['ch4']))} | "
            f"1H {pct_with_emoji(float(sig['ch1']))} | 15m {pct_with_emoji(float(sig['ch15']))} | {sig['regime']}\n\n"
            f"TradeSetup: /tradesetup {token}"
        )
        await update.message.reply_text(msg)
    else:
        await update.message.reply_text(
            f"{token} not found in current PulseFutures signals.\n"
            f"You can still size risk: /tradesetup {token}"
        )

# =========================
# ALERT JOB (EMAIL)
# =========================

async def alert_job(context: ContextTypes.DEFAULT_TYPE):
    try:
        last_ts = float(db_get_meta("last_email_ts", "0") or "0")
        last_syms = set((db_get_meta("last_email_symbols", "") or "").split(",")) if db_get_meta("last_email_symbols", "") else set()

        if not EMAIL_ENABLED_DEFAULT:
            return
        if not email_config_ok():
            return
        if not window_ok():
            return

        now = time.time()
        if now - last_ts < EMAIL_MIN_INTERVAL_SEC:
            return

        best_fut = await asyncio.to_thread(fetch_futures_tickers)
        if not best_fut:
            return

        setups = await asyncio.to_thread(pick_setups, best_fut)
        for s in setups:
            db_upsert_signal(s)

        movers_up, movers_dn = build_movers_tables(best_fut)

        setup_syms = {s.symbol for s in setups}
        mover_syms = set()
        for base, mv in best_fut.items():
            vol = usd_notional(mv)
            if vol < MOVER_VOL_USD_MIN:
                continue
            ch24 = float(mv.percentage or 0.0)
            if ch24 >= MOVER_UP_24H_MIN or ch24 <= MOVER_DN_24H_MAX:
                mover_syms.add(base)

        cur_syms = setup_syms | mover_syms
        cur_syms.discard("")

        if cur_syms and cur_syms == last_syms:
            return

        parts: List[str] = []
        if setups:
            for i, s in enumerate(setups, 1):
                parts.append(f"Setup #{i}:\n" + format_setup_email_block(s))
                parts.append("")
        else:
            parts.append("No strong setup right now.")
            parts.append("")

        parts.append("")
        parts.append("Movers:")
        parts.append(movers_up.replace("```", "").replace("*", ""))
        parts.append("")
        parts.append("Strong Movers:")
        parts.append(movers_dn.replace("```", "").replace("*", ""))

        body = "\n".join(parts).strip()
        subject = "PulseFutures: Setups + Movers"

        if send_email(subject, body):
            db_set_meta("last_email_ts", str(now))
            db_set_meta("last_email_symbols", ",".join(sorted(cur_syms)))
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

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("screen", screen))
    app.add_handler(CommandHandler("diag", diag))

    app.add_handler(CommandHandler("notify_on", notify_on))
    app.add_handler(CommandHandler("notify_off", notify_off))

    app.add_handler(CommandHandler("equity", equity_cmd))
    app.add_handler(CommandHandler("limits", limits_cmd))

    app.add_handler(CommandHandler("tradesetup", tradesetup))
    app.add_handler(CommandHandler("risk", risk_cmd))

    app.add_handler(CommandHandler("open", open_cmd))
    app.add_handler(CommandHandler("closepnl", closepnl_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    if getattr(app, "job_queue", None):
        app.job_queue.run_repeating(alert_job, interval=CHECK_INTERVAL_MIN * 60, first=10)
    else:
        logger.warning(
            "JobQueue not available. Ensure requirements include "
            '"python-telegram-bot[job-queue]>=20.7,<22.0"'
        )

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
