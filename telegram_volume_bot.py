#!/usr/bin/env python3
"""
CoinEx screener bot
Table columns (Telegram):
  SYM | FUT | SPOT | % | FUT(4h) | %(4h)
- FUT/SPOT are 24h notionals in $M (integers)
- % and %(4h) are rounded integers with emoji
  🟢 >= +5%, 🟡 between -4.99%..+4.99%, 🔴 <= -5%

Features:
- /screen → P1: 10 rows, P2: 5 rows, P3: 5 rows
- Send a coin ticker (e.g., PYTH) → same table just for that coin (ignores exclusions)
- /excel → Excel .xlsx (priority,symbol,usd_24h)
- /diag  → diagnostics

Lists EXCLUDE bases: BTC, ETH, XRP, SOL, DOGE, ADA, PEPE, LINK
Thresholds:
  P1: Futures ≥ $5M & Spot ≥ $500k
  P2: Futures ≥ $2M
  P3: Spot   ≥ $3M
"""

import asyncio, logging, os, time, io, traceback, re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import ccxt  # type: ignore
from tabulate import tabulate  # type: ignore
from openpyxl import Workbook  # type: ignore
from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# ---- Config / thresholds ----
P1_SPOT_MIN = 500_000
P1_FUT_MIN  = 5_000_000
P2_FUT_MIN  = 2_000_000
P3_SPOT_MIN = 3_000_000

# Rows per priority
TOP_N_P1    = 10
TOP_N_P2    = 5
TOP_N_P3    = 5

EXCHANGE_ID = "coinex"
TOKEN = os.environ.get("TELEGRAM_TOKEN")
STABLES = {"USD","USDT","USDC","TUSD","FDUSD","USDD","USDE","DAI","PYUSD"}

# Exclusions (apply to lists only — NOT to direct symbol lookup)
EXCLUDE_BASES = {"BTC","ETH","XRP","SOL","DOGE","ADA","PEPE","LINK"}

LAST_ERROR: Optional[str] = None

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

def safe_split_symbol(sym: Optional[str]):
    if not sym: return None
    pair = sym.split(":")[0]
    if "/" not in pair: return None
    return tuple(pair.split("/", 1))

def to_mv(t: dict) -> Optional[MarketVol]:
    sym = t.get("symbol")
    split = safe_split_symbol(sym)
    if not split: return None
    base, quote = split
    last = float(t.get("last") or t.get("close") or 0.0)
    open_ = float(t.get("open") or 0.0)
    percentage = float(t.get("percentage") or 0.0)
    base_vol = float(t.get("baseVolume") or 0.0)
    quote_vol = float(t.get("quoteVolume") or 0.0)
    vwap = float(t.get("vwap") or 0.0)
    return MarketVol(
        symbol=sym, base=base, quote=quote,
        last=last, open=open_, percentage=percentage,
        base_vol=base_vol, quote_vol=quote_vol, vwap=vwap
    )

def usd_notional(mv: Optional[MarketVol]) -> float:
    """USD-like 24h notional (prefer quoteVolume if USD-quoted; else baseVolume * (vwap or last))."""
    if not mv: return 0.0
    if mv.quote in STABLES and mv.quote_vol and mv.quote_vol > 0:
        return mv.quote_vol
    price = mv.vwap if mv.vwap and mv.vwap > 0 else mv.last
    return mv.base_vol * price if price and mv.base_vol else 0.0

def pct_change(mv_spot: Optional[MarketVol], mv_fut: Optional[MarketVol]) -> float:
    """Prefer exchange-provided 24h percentage; else compute from open/last."""
    for mv in (mv_spot, mv_fut):
        if mv and mv.percentage:
            return float(mv.percentage)
    mv = mv_spot or mv_fut
    if mv and mv.open and mv.open > 0 and mv.last:
        return (mv.last - mv.open) / mv.open * 100.0
    return 0.0

# --- Display helpers (rounded integers & emoji thresholds) ---
def pct_with_emoji(p: float) -> str:
    pr = round(p)
    if pr <= -5: emoji = "🔴"
    elif pr >= 5: emoji = "🟢"
    else: emoji = "🟡"
    return f"{pr:+d}% {emoji}"

def m_dollars_int(x: float) -> str:
    """Return millions as integer (rounded)."""
    return str(round(x / 1_000_000.0))

# --- CCXT wrappers ---
def build_exchange(default_type: str):
    klass = ccxt.__dict__[EXCHANGE_ID]
    return klass({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": default_type},
    })

def safe_fetch_tickers(ex: ccxt.Exchange) -> Dict[str, dict]:
    try:
        ex.load_markets()
        return ex.fetch_tickers()
    except Exception as e:
        global LAST_ERROR
        LAST_ERROR = f"{type(e).__name__}: {e}"
        logging.exception("fetch_tickers failed")
        return {}

def load_best(apply_exclusions: bool = True) -> Tuple[Dict[str, MarketVol], Dict[str, MarketVol], int, int]:
    """Return 'best' spot/fut ticker per BASE (highest USD notional).
       If apply_exclusions=False, EXCLUDE_BASES is ignored (for direct symbol lookup)."""
    # SPOT
    ex_spot = build_exchange("spot")
    spot_tickers = safe_fetch_tickers(ex_spot)
    best_spot: Dict[str, MarketVol] = {}
    for _, t in spot_tickers.items():
        mv = to_mv(t)
        if not mv: continue
        if mv.quote not in STABLES: continue
        if apply_exclusions and mv.base in EXCLUDE_BASES: continue
        prev = best_spot.get(mv.base)
        if prev is None or usd_notional(mv) > usd_notional(prev):
            best_spot[mv.base] = mv

    # FUTURES
    ex_fut = build_exchange("swap")
    fut_tickers = safe_fetch_tickers(ex_fut)
    best_fut: Dict[str, MarketVol] = {}
    for _, t in fut_tickers.items():
        mv = to_mv(t)
        if not mv: continue
        if apply_exclusions and mv.base in EXCLUDE_BASES: continue
        prev = best_fut.get(mv.base)
        if prev is None or usd_notional(mv) > usd_notional(prev):
            best_fut[mv.base] = mv

    return best_spot, best_fut, len(spot_tickers), len(fut_tickers)

# --- 4h futures metrics (volume + % change) ---
def fut_4h_metrics(ex_fut: ccxt.Exchange, mv_fut: Optional[MarketVol], fut_usd_24h: float) -> Tuple[float, float]:
    """
    For the futures symbol, pull 1h candles (last 4).
    Returns (usd_volume_4h, pct_change_4h).
    Candle volume is often in CONTRACTS for swaps; convert using market.contractSize when contract==True.
    We also cap 4h volume at the 24h futures notional for sanity.
    """
    if not mv_fut:
        return 0.0, 0.0
    symbol = mv_fut.symbol  # e.g., "AAVE/USDT:USDT"
    try:
        ex_fut.load_markets()
        market = ex_fut.market(symbol)
        is_contract = bool(market.get("contract"))
        contract_size = float(market.get("contractSize") or 1.0)

        ohlcv = ex_fut.fetch_ohlcv(symbol, timeframe="1h", limit=4)
        if not ohlcv:
            return 0.0, 0.0

        usd_vol = 0.0
        first_open = None
        last_close = None
        for i, c in enumerate(ohlcv):
            ts, o, h, l, cl, vol = c
            vol = float(vol or 0.0)
            # If contract market, 'vol' is contracts → convert to base units
            base_qty = vol * contract_size if is_contract else vol
            typical = (float(h or 0.0) + float(l or 0.0) + float(cl or 0.0)) / 3.0
            usd_vol += base_qty * typical
            if i == 0:
                first_open = float(o or 0.0)
            last_close = float(cl or 0.0)

        pct4 = 0.0
        if first_open and first_open > 0 and last_close is not None:
            pct4 = (last_close - first_open) / first_open * 100.0

        # Sanity: 4h must not exceed 24h
        if fut_usd_24h and fut_usd_24h > 0:
            usd_vol = min(usd_vol, fut_usd_24h)

        return usd_vol, pct4
    except Exception as e:
        logging.exception(f"4h metrics failed for {symbol}: {e}")
