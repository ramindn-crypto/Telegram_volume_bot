#!/usr/bin/env python3
"""
CoinEx screener bot
Table columns (Telegram):
  SYM | F | S | % | %4H
  - F, S are *million USD*, rounded to integers
  - % and %4H are integers with emoji (🟢/🟡/🔴)

Features:
- /screen → P1 (10 rows), P2 (5), P3 (5)
- Type a ticker (e.g., PYTH or $PYTH) → one-row table for that coin (ignores exclusions)
- /excel  → Excel .xlsx (legacy 3-col export to keep compatibility)
- /diag   → diagnostics

Exclusions for lists only: BTC, ETH, XRP, SOL, DOGE, ADA, PEPE, LINK
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

# Exclusions for lists (NOT for direct symbol query)
EXCLUDE_BASES = {"BTC","ETH","XRP","SOL","DOGE","ADA","PEPE","LINK"}

LAST_ERROR: Optional[str] = None

# cache for 4H % to avoid repeated OHLCV calls in one run
PCT4H_CACHE: Dict[Tuple[str,str], float] = {}  # (defaultType, symbol) -> pct4h

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
    """24h %: prefer ticker's percentage (spot then fut), else compute from open/last."""
    for mv in (mv_spot, mv_fut):
        if mv and mv.percentage:
            return float(mv.percentage)
    mv = mv_spot or mv_fut
    if mv and mv.open and mv.open > 0 and mv.last:
        return (mv.last - mv.open) / mv.open * 100.0
    return 0.0

def pct_with_emoji(p: float) -> str:
    p_rounded = round(p)  # integer only
    if p_rounded <= -3: emoji = "🔴"
    elif p_rounded >= 3: emoji = "🟢"
    else: emoji = "🟡"
    return f"{p_rounded:+d}% {emoji}"

def m_dollars_int(x: float) -> str:
    """Return millions as integer (rounded)."""
    return str(round(x / 1_000_000.0))

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
    """Return best spot/fut tickers per BASE."""
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

# ---- 4H % from OHLCV(1h) ----
def compute_pct4h_for_symbol(market_symbol: str, prefer_swap: bool = True) -> float:
    """
    Compute % change over the last 4 completed hours using 1h candles.
    Prefer futures ('swap') symbol; fall back to spot if needed.
    """
    try_order = ["swap", "spot"] if prefer_swap else ["spot", "swap"]
    for dtype in try_order:
        cache_key = (dtype, market_symbol)
        if cache_key in PCT4H_CACHE:
            return PCT4H_CACHE[cache_key]
        try:
            ex = build_exchange(dtype)
            ex.load_markets()
            candles = ex.fetch_ohlcv(market_symbol, timeframe="1h", limit=5)
            if not candles or len(candles) < 5:
                PCT4H_CACHE[cache_key] = 0.0
                continue
            closes = [c[4] for c in candles]
            close_now = closes[-1]
            close_4h_ago = closes[-5]
            pct4h = ((close_now - close_4h_ago) / close_4h_ago * 100.0) if close_4h_ago else 0.0
            PCT4H_CACHE[cache_key] = pct4h
            return pct4h
        except Exception:
            logging.exception("compute_pct4h_for_symbol failed for %s (%s)", market_symbol, dtype)
            PCT4H_CACHE[cache_key] = 0.0
            continue
    return 0.0

# ---- Priority builders ----
def build_priorities(best_spot: Dict[str,MarketVol], best_fut: Dict[str,MarketVol]):
    """
    Rows are [base, fut_usd, spot_usd, pct_24h, pct_4h]
    Sorting: P1 & P2 by fut_usd desc; P3 by spot_usd desc
    """
    p1_full, p2_full, p3_full = [], [], []

    # P1
    for base in set(best_spot) & set(best_fut):
        if base in EXCLUDE_BASES: continue
        s, f = best_spot[base], best_fut[base]
        fut_usd, spot_usd = usd_notional(f), usd_notional(s)
        if fut_usd >= P1_FUT_MIN and spot_usd >= P1_SPOT_MIN:
            pct4h = compute_pct4h_for_symbol(f.symbol, prefer_swap=True)
            p1_full.append([base, fut_usd, spot_usd, pct_change(s, f), pct4h])
    p1_full.sort(key=lambda r: r[1], reverse=True)
    p1 = p1_full[:TOP_N_P1]
    used = {r[0] for r in p1}

    # P2
    for base, f in best_fut.items():
        if base in used or base in EXCLUDE_BASES: continue
        fut_usd = usd_notional(
