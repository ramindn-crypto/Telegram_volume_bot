#!/usr/bin/env python3
"""
PulseFutures — Bybit Futures (Swap) Screener + TradeSetup + FULL Trade Journal + Risk Manager (Telegram)

GOAL:
✅ Telegram becomes a professional trading journal:
- Trader can OPEN any trade (manual) and CLOSE it later
- Risk per trade is tracked, open risk is tracked, daily risk budget is tracked
- /status shows: equity, daily cap, daily used, remaining, open risk, trades today, all OPEN trades
- Max trades/day enforced (default 5). Warn/deny when exceeded.
- Trades are stored in SQLite (persistent if you mount a disk on Render)

✅ TradeSetup is ONLY a suggestion (does NOT auto-open a trade).
- /tradesetup BTC -> sizing + SL/TP suggestion based on risk settings
- user must /open to record it as a real trade in journal

✅ Speed:
- All heavy CCXT/market work is moved to background threads (asyncio.to_thread)

✅ Fixes:
- Multi-TP equality issues fixed by enforcing strict separation
- Equity “reset” usually is non-persistent disk / redeploy -> use persistent disk and DB_PATH.

ENV (required):
- TELEGRAM_TOKEN

Optional:
- DB_PATH=/path/to/persistent/pulsefutures.db
- CHECK_INTERVAL_MIN=0 (default off)
- EXCHANGE_TIMEOUT_MS=20000

NOT FINANCIAL ADVICE.
"""

import asyncio
import logging
import os
import re
import sqlite3
import time
import math
import html
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

import ccxt
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

# =========================
# CONFIG
# =========================
EXCHANGE_ID = "bybit"
TOKEN = os.environ.get("TELEGRAM_TOKEN")

DEFAULT_TYPE = "swap"
EXCHANGE_TIMEOUT_MS = int(os.environ.get("EXCHANGE_TIMEOUT_MS", "20000"))

CHECK_INTERVAL_MIN = int(os.environ.get("CHECK_INTERVAL_MIN", "0"))  # 0 = no background job

DB_PATH = os.environ.get("DB_PATH", "pulsefutures.db")

# Screener sizes
LEADERS_N = 10
MOVERS_N = 10
SETUPS_N = 3

# Directional leaders/losers thresholds
MOVER_VOL_USD_MIN = 5_000_000
MOVER_UP_24H_MIN = 10.0
MOVER_DN_24H_MAX = -10.0

# Setup thresholds
TRIGGER_1H_ABS_MIN = 2.0
CONFIRM_15M_ABS_MIN = 0.6
ALIGN_4H_MIN = 0.0

# ATR-based SL/TP
USE_ATR_SLTP = True
ATR_PERIOD = 14
ATR_MIN_PCT = 0.8
ATR_MAX_PCT = 8.0
TP_MAX_PCT = 8.0

MULTI_TP_MIN_CONF = 75
TP_ALLOCS = (40, 40, 20)
TP_R_MULTS = (0.9, 1.6, 2.4)  # TP1/TP2/TP3 in R

# Risk defaults (IMPORTANT: equity default must be ZERO)
DEFAULT_EQUITY = 0.0
DEFAULT_RISK_MODE = "USD"     # USD or PCT
DEFAULT_RISK_VALUE = 40.0     # if USD -> $40, if PCT -> %
DEFAULT_DAILY_CAP_MODE = "PCT"
DEFAULT_DAILY_CAP_VALUE = 5.0
DEFAULT_OPEN_RISK_CAP_USD = 200.0
DEFAULT_MAX_TRADES_DAY = 5

# Market bias
BIAS_UNIVERSE_N = 20
BIAS_STRONG_TH = 0.6

STABLES = {"USDT", "USDC", "USD", "TUSD", "FDUSD", "DAI", "PYUSD"}

HDR = "━━━━━━━━━━━━━━━━━━━━"
SEP = "────────────────────"

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
    symbol: str
    market_symbol: str
    side: str  # BUY/SELL
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
    con = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")
    except Exception:
        pass
    return con


def db_init():
    con = db_connect()
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            equity REAL NOT NULL,
            tz TEXT NOT NULL,

            risk_mode TEXT NOT NULL,
            risk_value REAL NOT NULL,

            daily_cap_mode TEXT NOT NULL,
            daily_cap_value REAL NOT NULL,

            open_risk_cap REAL NOT NULL,
            max_trades_day INTEGER NOT NULL,

            day_trade_date TEXT NOT NULL,
            day_trade_count INTEGER NOT NULL,
            daily_risk_used REAL NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,

            symbol TEXT NOT NULL,
            side TEXT NOT NULL,               -- LONG/SHORT
            entry REAL NOT NULL,
            sl REAL NOT NULL,

            tp1 REAL,
            tp2 REAL,
            tp3 REAL,

            risk_usd REAL NOT NULL,
            qty REAL NOT NULL,
            notes TEXT,
            tags TEXT,

            status TEXT NOT NULL,             -- OPEN/CLOSED
            opened_ts REAL NOT NULL,
            closed_ts REAL,
            exit_price REAL,
            pnl_usd REAL
        )
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_trades_user_status
        ON trades(user_id, status);
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

    con.commit()
    con.close()


def get_user(user_id: int) -> dict:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()

    if not row:
        today = date.today().isoformat()
        cur.execute(
            """
            INSERT INTO users (
                user_id, equity, tz,
                risk_mode, risk_value,
                daily_cap_mode, daily_cap_value,
                open_risk_cap, max_trades_day,
                day_trade_date, day_trade_count, daily_risk_used
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                float(DEFAULT_EQUITY),
                "Australia/Melbourne",
                DEFAULT_RISK_MODE,
                float(DEFAULT_RISK_VALUE),
                DEFAULT_DAILY_CAP_MODE,
                float(DEFAULT_DAILY_CAP_VALUE),
                float(DEFAULT_OPEN_RISK_CAP_USD),
                int(DEFAULT_MAX_TRADES_DAY),
                today,
                0,
                0.0,
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
    today = date.today().isoformat()
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
            s.symbol.upper(),
            s.market_symbol,
            s.side,
            s.conf,
            s.entry,
            s.sl,
            s.tp,
            s.tp1,
            s.tp2,
            s.fut_vol_usd,
            s.ch24,
            s.ch4,
            s.ch1,
            s.ch15,
            s.regime,
            s.created_ts,
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


def db_open_trade(
    user_id: int,
    symbol: str,
    side: str,
    entry: float,
    sl: float,
    tp1: Optional[float],
    tp2: Optional[float],
    tp3: Optional[float],
    risk_usd: float,
    qty: float,
    notes: str = "",
    tags: str = "",
) -> int:
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO trades (
            user_id, symbol, side, entry, sl, tp1, tp2, tp3,
            risk_usd, qty, notes, tags,
            status, opened_ts
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
        """,
        (
            user_id,
            symbol.upper(),
            side.upper(),
            float(entry),
            float(sl),
            float(tp1) if tp1 is not None else None,
            float(tp2) if tp2 is not None else None,
            float(tp3) if tp3 is not None else None,
            float(risk_usd),
            float(qty),
            notes,
            tags,
            time.time(),
        ),
    )
    trade_id = cur.lastrowid
    con.commit()
    con.close()
    return int(trade_id)


def db_get_open_trades(user_id: int) -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT * FROM trades
        WHERE user_id=? AND status='OPEN'
        ORDER BY opened_ts ASC
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


def db_get_trade(user_id: int, trade_id: int) -> Optional[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM trades WHERE user_id=? AND id=?", (user_id, trade_id))
    row = cur.fetchone()
    con.close()
    return dict(row) if row else None


def db_close_trade(
    user_id: int,
    trade_id: int,
    exit_price: Optional[float],
    pnl_usd: float,
) -> Optional[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        "SELECT * FROM trades WHERE user_id=? AND id=? AND status='OPEN'",
        (user_id, trade_id),
    )
    row = cur.fetchone()
    if not row:
        con.close()
        return None
    t = dict(row)
    cur.execute(
        """
        UPDATE trades
        SET status='CLOSED', closed_ts=?, exit_price=?, pnl_usd=?
        WHERE user_id=? AND id=?
        """,
        (
            time.time(),
            float(exit_price) if exit_price is not None else None,
            float(pnl_usd),
            user_id,
            trade_id,
        ),
    )
    con.commit()
    con.close()
    t["exit_price"] = exit_price
    t["pnl_usd"] = pnl_usd
    t["status"] = "CLOSED"
    return t


def db_journal_last(user_id: int, limit: int = 20) -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT * FROM trades
        WHERE user_id=?
        ORDER BY opened_ts DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


def db_journal_range(user_id: int, start_ts: float, end_ts: float, limit: int = 200) -> List[dict]:
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT * FROM trades
        WHERE user_id=?
          AND opened_ts BETWEEN ? AND ?
        ORDER BY opened_ts DESC
        LIMIT ?
        """,
        (user_id, start_ts, end_ts, limit),
    )
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


# =========================
# EXCHANGE HELPERS
# =========================
def build_exchange():
    klass = ccxt.__dict__[EXCHANGE_ID]
    return klass(
        {
            "enableRateLimit": True,
            "timeout": EXCHANGE_TIMEOUT_MS,
            "options": {"defaultType": DEFAULT_TYPE},
        }
    )


def safe_split_symbol(sym: Optional[str]) -> Optional[Tuple[str, str]]:
    if not sym:
        return None
    pair = sym.split(":")[0]
    if "/" not in pair:
        return None
    base, quote = pair.split("/", 1)
    return base, quote


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


def metrics_from_candles_1h_15m(market_symbol: str) -> Tuple[float, float, float, float]:
    need_1h = max(ATR_PERIOD + 8, 35)
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
# FORMAT / UTILS
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


def pct_emoji(p: float) -> str:
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
        return "BULL 🟢"
    if ch24 <= -3 and ch4 <= -2:
        return "BEAR 🔴"
    return "NEUTRAL 🟡"


def parse_kv_args(text: str) -> dict:
    out = {}
    for part in text.split():
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


def parse_side(s: str) -> Optional[str]:
    s = s.strip().lower()
    if s in {"long", "buy"}:
        return "LONG"
    if s in {"short", "sell"}:
        return "SHORT"
    return None


def parse_float(x: str) -> float:
    x = x.strip().replace(",", "")
    return float(x)


def parse_symbol(x: str) -> str:
    x = re.sub(r"[^A-Za-z0-9]", "", x).upper().lstrip("$")
    return x


# =========================
# MARKET BIAS + SETUP MATH
# =========================
def market_bias(best_fut: Dict[str, MarketVol]) -> str:
    top = sorted(best_fut.items(), key=lambda kv: usd_notional(kv[1]), reverse=True)[:BIAS_UNIVERSE_N]
    if not top:
        return "Market Bias: UNKNOWN"

    w_sum = 0.0
    score = 0.0
    long_n = 0
    short_n = 0

    for _base, mv in top:
        vol = usd_notional(mv)
        if vol <= 0:
            continue

        _ch1, ch4, _ch15, _atr = metrics_from_candles_1h_15m(mv.symbol)
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


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sl_mult_from_conf(conf: int) -> float:
    if conf >= 85:
        return 2.2
    if conf >= 75:
        return 1.9
    if conf >= 65:
        return 1.6
    return 1.35


def sl_tp_from_atr(entry: float, side: str, atr: float, conf: int) -> Tuple[float, float, float]:
    if entry <= 0 or atr <= 0:
        return 0.0, 0.0, 0.0

    sl_dist = sl_mult_from_conf(conf) * atr
    min_dist = (ATR_MIN_PCT / 100.0) * entry
    max_dist = (ATR_MAX_PCT / 100.0) * entry
    sl_dist = clamp(sl_dist, min_dist, max_dist)

    r = sl_dist
    tp_dist = 1.6 * r
    tp_max_dist = (TP_MAX_PCT / 100.0) * entry
    tp_dist = min(tp_dist, tp_max_dist)

    if side == "BUY":
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist

    return sl, tp, r


def enforce_strict_levels(entry: float, side: str, tps: Tuple[float, float, float]) -> Tuple[float, float, float]:
    tp1, tp2, tp3 = tps
    eps = max(entry * 0.0002, 1e-9)  # 0.02%

    if side == "BUY":
        tp1 = max(tp1, entry + eps)
        tp2 = max(tp2, tp1 + eps)
        tp3 = max(tp3, tp2 + eps)
    else:
        tp1 = min(tp1, entry - eps)
        tp2 = min(tp2, tp1 - eps)
        tp3 = min(tp3, tp2 - eps)

    return tp1, tp2, tp3


def multi_tp_from_r(entry: float, side: str, r: float) -> Tuple[float, float, float]:
    if r <= 0 or entry <= 0:
        return 0.0, 0.0, 0.0

    r1, r2, r3 = TP_R_MULTS
    if not (r1 > 0 and r2 > 0 and r3 > 0 and r1 < r2 < r3):
        r1, r2, r3 = (0.9, 1.6, 2.4)

    maxd = (TP_MAX_PCT / 100.0) * entry
    d3 = min(r3 * r, maxd)
    d1 = d3 * (r1 / r3)
    d2 = d3 * (r2 / r3)

    if side == "BUY":
        tp1, tp2, tp3 = (entry + d1, entry + d2, entry + d3)
    else:
        tp1, tp2, tp3 = (entry - d1, entry - d2, entry - d3)

    return enforce_strict_levels(entry, side, (tp1, tp2, tp3))


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

    mag = min(abs(ch24) / 2.0 + abs(ch4) + abs(ch1) * 2.0 + abs(ch15) * 2.0, 20.0)
    score += mag

    if fut_vol_usd >= 20_000_000:
        score += 10
    elif fut_vol_usd >= 10_000_000:
        score += 8
    elif fut_vol_usd >= 5_000_000:
        score += 6
    elif fut_vol_usd >= 2_000_000:
        score += 3

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
    if side == "BUY" and ch4 < 0:
        return None
    if side == "SELL" and ch4 > 0:
        return None

    entry = float(mv.last or 0.0)
    if entry <= 0:
        return None

    conf = compute_confidence(side, ch24, ch4, ch1, ch15, fut_vol)
    reg = regime_label(ch24, ch4)

    sl, tp_single, r = sl_tp_from_atr(entry, side, atr, conf)
    if sl <= 0 or tp_single <= 0 or r <= 0:
        return None

    tp1 = tp2 = None
    tp3 = tp_single
    if conf >= MULTI_TP_MIN_CONF:
        tp1, tp2, tp3 = multi_tp_from_r(entry, side, r)

    return Setup(
        symbol=base.upper(),
        market_symbol=mv.symbol,
        side=side,
        conf=conf,
        entry=entry,
        sl=sl,
        tp=tp3,
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
    universe = sorted(best_fut.items(), key=lambda kv: usd_notional(kv[1]), reverse=True)[:35]
    out: List[Setup] = []
    for base, mv in universe:
        s = make_setup(base, mv)
        if s:
            out.append(s)
    out.sort(key=lambda s: (s.conf, s.fut_vol_usd), reverse=True)
    return out[:SETUPS_N]


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
        _ch1, ch4, _ch15, _atr = metrics_from_candles_1h_15m(mv.symbol)

        if ch24 >= MOVER_UP_24H_MIN and ch4 >= 0:
            up.append((base.upper(), vol, ch24, ch4, float(mv.last or 0.0)))
        if ch24 <= MOVER_DN_24H_MAX and ch4 <= 0:
            dn.append((base.upper(), vol, ch24, ch4, float(mv.last or 0.0)))

    up.sort(key=lambda x: (x[2], x[1]), reverse=True)
    dn.sort(key=lambda x: (x[2], x[1]))
    return up, dn


# =========================
# RISK
# =========================
def compute_trade_risk_usd(user: dict) -> float:
    equity = float(user.get("equity") or 0.0)
    mode = (user.get("risk_mode") or DEFAULT_RISK_MODE).upper()
    val = float(user.get("risk_value") or 0.0)
    if mode == "PCT":
        return max(0.0, equity * (val / 100.0))
    return max(0.0, val)


def compute_daily_cap_usd(user: dict) -> float:
    equity = float(user.get("equity") or 0.0)
    mode = (user.get("daily_cap_mode") or DEFAULT_DAILY_CAP_MODE).upper()
    val = float(user.get("daily_cap_value") or 0.0)
    if mode == "PCT":
        return max(0.0, equity * (val / 100.0))
    return max(0.0, val)


def calc_qty_from_risk(entry: float, sl: float, risk_usd: float) -> float:
    dist = abs(entry - sl)
    if dist <= 0:
        return 0.0
    return risk_usd / dist


def open_risk_usd(open_trades: List[dict]) -> float:
    return float(sum(float(t["risk_usd"]) for t in open_trades))


# =========================
# HELP TEXT
# =========================
HELP_TEXT = """\
PulseFutures — Journal + Risk Manager

Core
- /help
- /equity 5000
- /status

Risk settings
- /riskmode usd 40
- /riskmode pct 2.0
- /dailycap usd 200
- /dailycap pct 5
- /opencap 200
- /maxtrades 5

Journal
- /open BTC long entry=98100 sl=97200 risk=40 notes=breakout
- /open BTC long entry=98100 sl=97200   (risk from your /riskmode)
- /close 17 exit=98950
- /close 17 pnl=+65
- /close 17 r=+1.2
- /journal today
- /journal week
- /journal last 20

Market
- /screen
- /tradesetup BTC   (suggestion only)

Notes
- Equity starts at 0. Set it once using /equity.
- If equity “resets”, your DB is not persistent. Mount a persistent disk and set DB_PATH there.
- Not financial advice.
"""


# =========================
# COMMANDS
# =========================
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)


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
        await update.message.reply_text(f"✅ Equity updated: ${eq:.2f}")
    except Exception:
        await update.message.reply_text("Usage: /equity 5000")


async def riskmode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if not context.args:
        await update.message.reply_text(
            f"Risk per trade: {user['risk_mode']} {float(user['risk_value']):.2f}\n"
            f"Set: /riskmode usd 40  OR  /riskmode pct 2.0"
        )
        return
    if len(context.args) != 2:
        await update.message.reply_text("Usage: /riskmode usd 40  OR  /riskmode pct 2.0")
        return
    mode = context.args[0].strip().upper()
    try:
        val = float(context.args[1])
    except Exception:
        await update.message.reply_text("Usage: /riskmode usd 40  OR  /riskmode pct 2.0")
        return
    if mode not in {"USD", "PCT"}:
        await update.message.reply_text("Mode must be usd or pct.")
        return
    if mode == "USD" and val <= 0:
        await update.message.reply_text("For USD, value must be > 0")
        return
    if mode == "PCT" and not (0.1 <= val <= 20.0):
        await update.message.reply_text("For PCT, choose 0.1..20.0")
        return
    update_user(uid, risk_mode=mode, risk_value=val)
    await update.message.reply_text(f"✅ Risk per trade set: {mode} {val:.2f}")


async def dailycap_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if not context.args:
        cap = compute_daily_cap_usd(user)
        await update.message.reply_text(
            f"Daily cap: {user['daily_cap_mode']} {float(user['daily_cap_value']):.2f} (≈ ${cap:.2f})\n"
            f"Set: /dailycap usd 200  OR  /dailycap pct 5"
        )
        return
    if len(context.args) != 2:
        await update.message.reply_text("Usage: /dailycap usd 200  OR  /dailycap pct 5")
        return
    mode = context.args[0].strip().upper()
    try:
        val = float(context.args[1])
    except Exception:
        await update.message.reply_text("Usage: /dailycap usd 200  OR  /dailycap pct 5")
        return
    if mode not in {"USD", "PCT"}:
        await update.message.reply_text("Mode must be usd or pct.")
        return
    if mode == "USD" and val < 0:
        await update.message.reply_text("For USD, value must be >= 0")
        return
    if mode == "PCT" and not (0.0 <= val <= 50.0):
        await update.message.reply_text("For PCT, choose 0..50")
        return
    update_user(uid, daily_cap_mode=mode, daily_cap_value=val)
    user = get_user(uid)
    cap = compute_daily_cap_usd(user)
    await update.message.reply_text(f"✅ Daily cap set: {mode} {val:.2f} (≈ ${cap:.2f})")


async def opencap_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if not context.args:
        await update.message.reply_text(f"Open risk cap: ${float(user['open_risk_cap']):.2f}\nSet: /opencap 200")
        return
    try:
        v = float(context.args[0])
        if v < 0:
            raise ValueError()
        update_user(uid, open_risk_cap=v)
        await update.message.reply_text(f"✅ Open risk cap set: ${v:.2f}")
    except Exception:
        await update.message.reply_text("Usage: /opencap 200")


async def maxtrades_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if not context.args:
        await update.message.reply_text(f"Max trades/day: {int(user['max_trades_day'])}\nSet: /maxtrades 5")
        return
    try:
        n = int(context.args[0])
        if n < 1 or n > 50:
            raise ValueError()
        update_user(uid, max_trades_day=n)
        await update.message.reply_text(f"✅ Max trades/day set: {n}")
    except Exception:
        await update.message.reply_text("Usage: /maxtrades 5  (1..50)")


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = reset_daily_if_needed(get_user(uid))

    open_trades = db_get_open_trades(uid)
    o_risk = open_risk_usd(open_trades)

    equity = float(user["equity"])
    daily_cap = compute_daily_cap_usd(user)
    daily_used = float(user["daily_risk_used"])
    daily_rem = max(0.0, daily_cap - daily_used)

    max_trades = int(user["max_trades_day"])
    trades_today = int(user["day_trade_count"])

    warn = []
    if trades_today >= max_trades:
        warn.append("⚠️ Max trades/day reached.")
    if o_risk >= float(user["open_risk_cap"]):
        warn.append("⚠️ Open risk cap reached.")
    if daily_cap > 0 and daily_used >= daily_cap:
        warn.append("⚠️ Daily risk cap reached.")

    lines = []
    lines.append(HDR)
    lines.append("📒 PulseFutures — Status")
    lines.append(HDR)
    lines.append(f"Equity: ${equity:.2f}")
    lines.append(f"Risk/Trade: {user['risk_mode']} {float(user['risk_value']):.2f} (≈ ${compute_trade_risk_usd(user):.2f})")
    lines.append(f"Daily Cap: {user['daily_cap_mode']} {float(user['daily_cap_value']):.2f} (≈ ${daily_cap:.2f})")
    lines.append(f"Daily Used: ${daily_used:.2f} | Remaining: ${daily_rem:.2f}")
    lines.append(f"Open Risk: ${o_risk:.2f} / ${float(user['open_risk_cap']):.2f}")
    lines.append(f"Trades Today: {trades_today} / {max_trades}")
    if warn:
        lines.append("")
        lines.extend(warn)

    lines.append("")
    lines.append(SEP)
    lines.append("OPEN Trades")
    lines.append(SEP)
    if not open_trades:
        lines.append("None")
    else:
        now_ts = time.time()
        for t in open_trades:
            age_min = int((now_ts - float(t["opened_ts"])) / 60)
            tp_line = ""
            if t.get("tp1") or t.get("tp2") or t.get("tp3"):
                tp_line = f" | TP: {fmt_price(float(t.get('tp1') or 0))}/{fmt_price(float(t.get('tp2') or 0))}/{fmt_price(float(t.get('tp3') or 0))}"
            lines.append(f"#{t['id']} {t['symbol']} {t['side']} | Entry {fmt_price(float(t['entry']))} | SL {fmt_price(float(t['sl']))}{tp_line}")
            lines.append(f"Risk ${float(t['risk_usd']):.2f} | Qty {float(t['qty']):.6g} | Age {age_min}m")
            if t.get("notes"):
                lines.append(f"Notes: {t['notes']}")
            if t.get("tags"):
                lines.append(f"Tags: {t['tags']}")
            lines.append("")

    await update.message.reply_text("\n".join(lines).strip())


async def open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = reset_daily_if_needed(get_user(uid))

    if len(context.args) < 3:
        await update.message.reply_text(
            "Usage:\n/open BTC long entry=98100 sl=97200 risk=40 notes=breakout\n"
            "Or:\n/open BTC long entry=98100 sl=97200  (risk from your /riskmode)"
        )
        return

    sym = parse_symbol(context.args[0])
    side = parse_side(context.args[1])
    if not sym or not side:
        await update.message.reply_text("Usage: /open BTC long entry=... sl=... risk=...")
        return

    kv = parse_kv_args(" ".join(context.args[2:]))

    try:
        entry = parse_float(kv.get("entry", ""))
        sl = parse_float(kv.get("sl", ""))
    except Exception:
        await update.message.reply_text("Missing/invalid entry or sl. Example: entry=98100 sl=97200")
        return

    if float(user["equity"]) <= 0:
        await update.message.reply_text("❌ Equity is 0. Set it first: /equity 5000")
        return

    if entry <= 0 or sl <= 0 or entry == sl:
        await update.message.reply_text("Invalid entry/SL.")
        return

    open_trades = db_get_open_trades(uid)
    o_risk = open_risk_usd(open_trades)

    trades_today = int(user["day_trade_count"])
    max_trades = int(user["max_trades_day"])
    if trades_today >= max_trades:
        await update.message.reply_text(f"❌ You already opened {trades_today}/{max_trades} trades today.")
        return

    daily_cap = compute_daily_cap_usd(user)
    daily_used = float(user["daily_risk_used"])
    daily_remaining = max(0.0, daily_cap - daily_used) if daily_cap > 0 else float("inf")
    open_remaining = max(0.0, float(user["open_risk_cap"]) - o_risk)

    risk_usd = None
    if "risk" in kv:
        try:
            risk_usd = float(parse_float(kv["risk"]))
        except Exception:
            risk_usd = None
    if risk_usd is None:
        risk_usd = compute_trade_risk_usd(user)

    if risk_usd <= 0:
        await update.message.reply_text("❌ Risk per trade is 0. Set it: /riskmode usd 40 OR /riskmode pct 2")
        return

    risk_usd = min(risk_usd, daily_remaining, open_remaining)
    if risk_usd <= 0:
        await update.message.reply_text("❌ No risk budget remaining (daily cap or open cap).")
        return

    qty = None
    if "qty" in kv:
        try:
            qty = float(parse_float(kv["qty"]))
        except Exception:
            qty = None
    if qty is None or qty <= 0:
        qty = calc_qty_from_risk(entry, sl, risk_usd)

    def opt_f(k: str) -> Optional[float]:
        if k not in kv:
            return None
        try:
            v = float(parse_float(kv[k]))
            return v if v > 0 else None
        except Exception:
            return None

    tp1 = opt_f("tp1")
    tp2 = opt_f("tp2")
    tp3 = opt_f("tp3")
    notes = kv.get("notes", "")
    tags = kv.get("tags", "")

    trade_id = db_open_trade(uid, sym, side, entry, sl, tp1, tp2, tp3, risk_usd, qty, notes=notes, tags=tags)
    update_user(uid, day_trade_count=trades_today + 1, daily_risk_used=float(user["daily_risk_used"]) + float(risk_usd))

    msg = (
        f"✅ Trade OPENED (Journal)\n"
        f"ID: #{trade_id}\n"
        f"{sym} {side}\n"
        f"Entry: {fmt_price(entry)}\n"
        f"SL:    {fmt_price(sl)}\n"
        f"Risk:  ${risk_usd:.2f}\n"
        f"Qty:   {qty:.6g}\n"
        f"Trades Today: {get_user(uid)['day_trade_count']} / {get_user(uid)['max_trades_day']}"
    )
    await update.message.reply_text(msg)


async def close_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = reset_daily_if_needed(get_user(uid))

    if len(context.args) < 2:
        await update.message.reply_text("Usage: /close 17 exit=98950  OR  /close 17 pnl=+65  OR  /close 17 r=+1.2")
        return

    try:
        trade_id = int(context.args[0])
    except Exception:
        await update.message.reply_text("Trade ID must be an integer. Example: /close 17 pnl=+65")
        return

    t = db_get_trade(uid, trade_id)
    if not t or t.get("status") != "OPEN":
        await update.message.reply_text("Trade not found or not OPEN.")
        return

    kv = parse_kv_args(" ".join(context.args[1:]))
    exit_price = None
    pnl = None

    if "pnl" in kv:
        try:
            pnl = float(parse_float(kv["pnl"]))
        except Exception:
            pnl = None

    if pnl is None and "exit" in kv:
        try:
            exit_price = float(parse_float(kv["exit"]))
        except Exception:
            exit_price = None
        if exit_price and exit_price > 0:
            entry = float(t["entry"])
            qty = float(t["qty"])
            if (t["side"] or "").upper() == "LONG":
                pnl = (exit_price - entry) * qty
            else:
                pnl = (entry - exit_price) * qty

    if pnl is None and "r" in kv:
        try:
            r = float(parse_float(kv["r"]))
            pnl = r * float(t["risk_usd"])
        except Exception:
            pnl = None

    if pnl is None:
        await update.message.reply_text("Could not compute pnl. Use one of: pnl=... OR exit=... OR r=...")
        return

    closed = db_close_trade(uid, trade_id, exit_price, pnl)
    if not closed:
        await update.message.reply_text("Trade not found or already closed.")
        return

    new_eq = float(user["equity"]) + float(pnl)
    update_user(uid, equity=new_eq)

    await update.message.reply_text(
        f"✅ Trade CLOSED\nID: #{trade_id} ({t['symbol']} {t['side']})\nPnL: {pnl:+.2f}\nEquity: ${new_eq:.2f}"
    )


async def journal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id

    arg = (context.args[0].lower() if context.args else "last")
    now = datetime.utcnow()

    if arg == "today":
        start = datetime(now.year, now.month, now.day)
        end = start + timedelta(days=1)
        rows = db_journal_range(uid, start.timestamp(), end.timestamp(), limit=200)
        title = "Journal — Today"
    elif arg == "week":
        start = now - timedelta(days=7)
        end = now
        rows = db_journal_range(uid, start.timestamp(), end.timestamp(), limit=200)
        title = "Journal — Last 7 Days"
    else:
        n = 20
        if len(context.args) >= 2:
            try:
                n = int(context.args[1])
                n = max(1, min(200, n))
            except Exception:
                n = 20
        rows = db_journal_last(uid, limit=n)
        title = f"Journal — Last {min(len(rows), n)}"

    if not rows:
        await update.message.reply_text(f"{title}\n\nNone")
        return

    closed = [r for r in rows if r.get("status") == "CLOSED" and r.get("pnl_usd") is not None]
    pnl_sum = sum(float(r["pnl_usd"]) for r in closed) if closed else 0.0
    wins = sum(1 for r in closed if float(r["pnl_usd"]) > 0)
    losses = sum(1 for r in closed if float(r["pnl_usd"]) < 0)

    lines = [HDR, f"📒 {title}", HDR]
    if closed:
        lines.append(f"Closed: {len(closed)} | Wins: {wins} | Losses: {losses} | PnL: {pnl_sum:+.2f}")
    lines.append(SEP)

    for r in rows[:50]:
        st = "🟩" if r["status"] == "OPEN" else ("🟢" if (r.get("pnl_usd") or 0) > 0 else ("🔴" if (r.get("pnl_usd") or 0) < 0 else "⚪"))
        opened = datetime.utcfromtimestamp(float(r["opened_ts"])).strftime("%m-%d %H:%M")
        line = f"{st} #{r['id']} {r['symbol']} {r['side']} | Risk ${float(r['risk_usd']):.2f} | {opened}"
        if r["status"] == "CLOSED":
            line += f" | PnL {float(r['pnl_usd']):+.2f}"
        lines.append(line)

    await update.message.reply_text("\n".join(lines).strip())


async def screen_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        best_fut = await asyncio.to_thread(fetch_futures_tickers)
    except Exception as e:
        logger.exception("fetch tickers failed: %s", e)
        await update.message.reply_text("Error: could not fetch futures tickers.")
        return

    if not best_fut:
        await update.message.reply_text("No data.")
        return

    bias_line = await asyncio.to_thread(market_bias, best_fut)
    up, dn = await asyncio.to_thread(compute_directional_lists, best_fut)
    setups = await asyncio.to_thread(pick_setups, best_fut)
    for s in setups:
        db_upsert_signal(s)

    lines = [HDR, "📊 PulseFutures — Market Scan", HDR, bias_line, ""]
    lines += [SEP, "🔥 Top Setups (Suggestion)", SEP]
    if not setups:
        lines.append("None")
    else:
        for i, s in enumerate(setups, 1):
            if s.tp1 and s.tp2 and s.conf >= MULTI_TP_MIN_CONF:
                tps = f"TP1 {fmt_price(s.tp1)} | TP2 {fmt_price(s.tp2)} | TP3 {fmt_price(s.tp)}"
            else:
                tps = f"TP {fmt_price(s.tp)}"
            lines.append(f"{i}) {s.side} {s.symbol} | Conf {s.conf}/100 | Entry {fmt_price(s.entry)} | SL {fmt_price(s.sl)} | {tps}")
            lines.append(f"   24H {pct_emoji(s.ch24)} | 4H {pct_emoji(s.ch4)} | 1H {pct_emoji(s.ch1)} | F~{fmt_money(s.fut_vol_usd)}")
            lines.append(f"   Tap button or use: /tradesetup {s.symbol}")
            lines.append("")

    lines += ["", SEP, "📈 Directional Leaders (Top 10)", SEP]
    lines += [f"{b:<6} 24H {pct_emoji(c24)} | 4H {pct_emoji(c4)} | F~{fmt_money(v)} | Last {fmt_price(px)}" for b, v, c24, c4, px in up[:MOVERS_N]] or ["None"]

    lines += ["", SEP, "📉 Directional Losers (Top 10)", SEP]
    lines += [f"{b:<6} 24H {pct_emoji(c24)} | 4H {pct_emoji(c4)} | F~{fmt_money(v)} | Last {fmt_price(px)}" for b, v, c24, c4, px in dn[:MOVERS_N]] or ["None"]

    kb_rows = [[InlineKeyboardButton(text=f"⚙️ TradeSetup {s.symbol}", callback_data=f"ts:{s.symbol}")] for s in setups[:SETUPS_N]]
    await update.message.reply_text("\n".join(lines).strip(), reply_markup=InlineKeyboardMarkup(kb_rows) if kb_rows else None)


async def tradesetup_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = reset_daily_if_needed(get_user(uid))

    if not context.args:
        await update.message.reply_text("Usage: /tradesetup BTC")
        return

    sym = parse_symbol(context.args[0])
    if not sym:
        await update.message.reply_text("Usage: /tradesetup BTC")
        return

    if float(user["equity"]) <= 0:
        await update.message.reply_text("❌ Equity is 0. Set it first: /equity 5000")
        return

    sig = db_get_signal(sym)
    if sig:
        entry = float(sig["entry"])
        sl = float(sig["sl"])
        side = "LONG" if sig["side"] == "BUY" else "SHORT"
        tp1 = sig.get("tp1")
        tp2 = sig.get("tp2")
        tp3 = sig.get("tp")
        conf = int(sig.get("conf") or 0)
        note_sig = "✅ From latest signal cache"
    else:
        best_fut = await asyncio.to_thread(fetch_futures_tickers)
        mv = best_fut.get(sym)
        if not mv:
            await update.message.reply_text(f"Could not find {sym} on Bybit futures.")
            return
        bias_line = await asyncio.to_thread(market_bias, best_fut)
        ch24 = float(mv.percentage or 0.0)
        ch1, ch4, ch15, atr = await asyncio.to_thread(metrics_from_candles_1h_15m, mv.symbol)
        side_buy = (ch1 > 0 and ch4 >= 0)
        side_sell = (ch1 < 0 and ch4 <= 0)
        if not (side_buy or side_sell):
            await update.message.reply_text(f"{sym}: No clean direction (trend conflict). Try later.")
            return
        buy_side = "BUY" if side_buy else "SELL"
        entry = float(mv.last or 0.0)
        conf = compute_confidence(buy_side, ch24, ch4, ch1, ch15, usd_notional(mv))
        sl, tp_single, r = sl_tp_from_atr(entry, buy_side, atr, conf)
        tp1 = tp2 = None
        tp3 = tp_single
        if conf >= MULTI_TP_MIN_CONF:
            tp1, tp2, tp3 = multi_tp_from_r(entry, buy_side, r)
        side = "LONG" if buy_side == "BUY" else "SHORT"
        note_sig = f"⚠️ Live compute (not cached). {bias_line}"

    open_trades = db_get_open_trades(uid)
    o_risk = open_risk_usd(open_trades)
    daily_cap = compute_daily_cap_usd(user)
    daily_used = float(user["daily_risk_used"])
    daily_rem = max(0.0, daily_cap - daily_used) if daily_cap > 0 else float("inf")
    open_rem = max(0.0, float(user["open_risk_cap"]) - o_risk)

    risk_target = compute_trade_risk_usd(user)
    risk_usd = min(risk_target, daily_rem, open_rem)

    if risk_usd <= 0:
        await update.message.reply_text("❌ No risk budget remaining (daily cap or open cap).")
        return

    qty = calc_qty_from_risk(entry, sl, risk_usd)
    if qty <= 0:
        await update.message.reply_text("Could not compute qty (invalid entry/SL).")
        return

    if tp1 and tp2 and conf >= MULTI_TP_MIN_CONF:
        tps = f"TP1 {fmt_price(float(tp1))} ({TP_ALLOCS[0]}%) | TP2 {fmt_price(float(tp2))} ({TP_ALLOCS[1]}%) | TP3 {fmt_price(float(tp3))} ({TP_ALLOCS[2]}%)\nRule: after TP1 → move SL to BE"
    else:
        tps = f"TP {fmt_price(float(tp3))}"

    msg = (
        f"{HDR}\n"
        f"⚙️ TradeSetup (Suggestion Only)\n"
        f"{HDR}\n"
        f"{sym} {side} | Confidence {conf}/100\n"
        f"{note_sig}\n\n"
        f"Entry: {fmt_price(entry)}\n"
        f"SL:    {fmt_price(sl)}\n"
        f"{tps}\n\n"
        f"Risk used: ${risk_usd:.2f}\n"
        f"Qty:      {qty:.6g}\n\n"
        f"To journal this trade:\n"
        f"/open {sym} {'long' if side=='LONG' else 'short'} entry={fmt_price(entry)} sl={fmt_price(sl)} risk={risk_usd:.2f}\n"
    )
    await update.message.reply_text(msg)


async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = query.data or ""
    if data.startswith("ts:"):
        sym = data.split(":", 1)[1].strip()
        context.args = [sym]
        # Use the message object as target to reply
        fake_update = Update(update.update_id, message=query.message)
        await tradesetup_cmd(fake_update, context)


async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    token = parse_symbol((update.message.text or "").strip())
    if len(token) < 2:
        return
    sig = db_get_signal(token)
    if not sig:
        await update.message.reply_text(f"{token}: not in current signals.\nUse: /tradesetup {token}")
        return
    side = "LONG" if sig["side"] == "BUY" else "SHORT"
    conf = int(sig["conf"])
    entry = float(sig["entry"])
    sl = float(sig["sl"])
    tp = float(sig["tp"])
    tp1 = sig.get("tp1")
    tp2 = sig.get("tp2")
    if tp1 and tp2 and conf >= MULTI_TP_MIN_CONF:
        tps = f"TP1 {fmt_price(float(tp1))} | TP2 {fmt_price(float(tp2))} | TP3 {fmt_price(tp)}"
    else:
        tps = f"TP {fmt_price(tp)}"
    await update.message.reply_text(
        f"{token} in signals:\n{side} | Conf {conf}/100\nEntry {fmt_price(entry)} | SL {fmt_price(sl)} | {tps}\nUse: /tradesetup {token}"
    )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled exception", exc_info=context.error)


async def background_job(context: ContextTypes.DEFAULT_TYPE):
    if ALERT_LOCK.locked():
        return
    async with ALERT_LOCK:
        try:
            best_fut = await asyncio.to_thread(fetch_futures_tickers)
            if not best_fut:
                return
            setups = await asyncio.to_thread(pick_setups, best_fut)
            for s in setups:
                db_upsert_signal(s)
            logger.info("background_job: refreshed %d setups", len(setups))
        except Exception:
            logger.exception("background_job failed")


def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN missing")

    db_init()

    app = Application.builder().token(TOKEN).build()
    app.add_error_handler(on_error)

    app.add_handler(CommandHandler(["help", "start"], help_cmd))
    app.add_handler(CommandHandler("equity", equity_cmd))
    app.add_handler(CommandHandler("riskmode", riskmode_cmd))
    app.add_handler(CommandHandler("dailycap", dailycap_cmd))
    app.add_handler(CommandHandler("opencap", opencap_cmd))
    app.add_handler(CommandHandler("maxtrades", maxtrades_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("open", open_cmd))
    app.add_handler(CommandHandler("close", close_cmd))
    app.add_handler(CommandHandler("journal", journal_cmd))
    app.add_handler(CommandHandler("screen", screen_cmd))
    app.add_handler(CommandHandler("tradesetup", tradesetup_cmd))

    # ✅ Correct callback handler
    app.add_handler(CallbackQueryHandler(callback_router))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    if CHECK_INTERVAL_MIN > 0 and getattr(app, "job_queue", None):
        app.job_queue.run_repeating(background_job, interval=CHECK_INTERVAL_MIN * 60, first=10, name="background_job")
    elif CHECK_INTERVAL_MIN > 0:
        logger.warning('JobQueue not available. Install "python-telegram-bot[job-queue]>=20.7,<22.0"')

    logger.info("Starting bot. DB_PATH=%s", DB_PATH)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
