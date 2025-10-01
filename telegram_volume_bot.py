#!/usr/bin/env python3
"""
CoinEx screener bot
Table columns (Telegram):
  SYM | F | S | % | %4H
  - F, S are *million USD*, rounded to integers
  - % and %4H are integers with emoji (🟢/🟡/🔴)

Features:
- /screen → P1 (10 rows), P2 (5), P3 (10; P3 always includes pinned: BTC,ETH,XRP,SOL,DOGE,ADA,PEPE,LINK)
- Type a ticker (e.g., PYTH or $PYTH) → one-row table for that coin
- /excel  → Excel .xlsx (legacy 3-col export kept)
- /diag   → diagnostics

Priority rules:
  P1: Futures ≥ $10M (EXCLUDES pinned)
  P2: Futures ≥ $2M (EXCLUDES pinned)
  P3: Always include pinned + Spot ≥ $3M (pinned first), TOTAL 10 rows
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
P1_FUT_MIN  = 10_000_000   # <-- updated
P2_FUT_MIN  = 2_000_000
P3_SPOT_MIN = 3_000_000

TOP_N_P1 = 10
TOP_N_P2 = 5
TOP_N_P3 = 10

EXCHANGE_ID = "coinex"
TOKEN = os.environ.get("TELEGRAM_TOKEN")
STABLES = {"USD","USDT","USDC","TUSD","FDUSD","USDD","USDE","DAI","PYUSD"}

# Pinned coins: must appear only in P3 (never in P1/P2)
PINNED_P3 = ["BTC","ETH","XRP","SOL","DOGE","ADA","PEPE","LINK"]
PINNED_SET = set(PINNED_P3)

LAST_ERROR: Optional[str] = None
PCT4H_CACHE: Dict[Tuple[str,str], float] = {}

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
    return MarketVol(sym, base, quote, last, open_, percentage, base_vol, quote_vol, vwap)

def usd_notional(mv: Optional[MarketVol]) -> float:
    if not mv: return 0.0
    if mv.quote in STABLES and mv.quote_vol and mv.quote_vol > 0:
        return mv.quote_vol
    price = mv.vwap if mv.vwap and mv.vwap > 0 else mv.last
    return mv.base_vol * price if price and mv.base_vol else 0.0

def pct_change(mv_spot: Optional[MarketVol], mv_fut: Optional[MarketVol]) -> float:
    for mv in (mv_spot, mv_fut):
        if mv and mv.percentage:
            return float(mv.percentage)
    mv = mv_spot or mv_fut
    if mv and mv.open and mv.open > 0 and mv.last:
        return (mv.last - mv.open) / mv.open * 100.0
    return 0.0

def pct_with_emoji(p: float) -> str:
    p_rounded = round(p)
    if p_rounded <= -3: emoji = "🔴"
    elif p_rounded >= 3: emoji = "🟢"
    else: emoji = "🟡"
    return f"{p_rounded:+d}% {emoji}"

def m_dollars_int(x: float) -> str:
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

def load_best() -> Tuple[Dict[str, MarketVol], Dict[str, MarketVol], int, int]:
    ex_spot = build_exchange("spot")
    spot_tickers = safe_fetch_tickers(ex_spot)
    best_spot: Dict[str, MarketVol] = {}
    for _, t in spot_tickers.items():
        mv = to_mv(t)
        if not mv: continue
        if mv.quote not in STABLES: continue
        if mv.base not in best_spot or usd_notional(mv) > usd_notional(best_spot[mv.base]):
            best_spot[mv.base] = mv

    ex_fut = build_exchange("swap")
    fut_tickers = safe_fetch_tickers(ex_fut)
    best_fut: Dict[str, MarketVol] = {}
    for _, t in fut_tickers.items():
        mv = to_mv(t)
        if not mv: continue
        if mv.base not in best_fut or usd_notional(mv) > usd_notional(best_fut[mv.base]):
            best_fut[mv.base] = mv

    return best_spot, best_fut, len(spot_tickers), len(fut_tickers)

def compute_pct4h_for_symbol(market_symbol: str, prefer_swap: bool = True) -> float:
    cache_key = ("swap" if prefer_swap else "spot", market_symbol)
    if cache_key in PCT4H_CACHE:
        return PCT4H_CACHE[cache_key]
    try_order = ["swap", "spot"] if prefer_swap else ["spot", "swap"]
    for dtype in try_order:
        ck = (dtype, market_symbol)
        if ck in PCT4H_CACHE: return PCT4H_CACHE[ck]
        try:
            ex = build_exchange(dtype)
            ex.load_markets()
            candles = ex.fetch_ohlcv(market_symbol, timeframe="1h", limit=5)
            if not candles or len(candles) < 5:
                PCT4H_CACHE[ck] = 0.0
                continue
            closes = [c[4] for c in candles]
            pct4h = ((closes[-1] - closes[0]) / closes[0] * 100.0) if closes[0] else 0.0
            PCT4H_CACHE[ck] = pct4h
            return pct4h
        except Exception:
            logging.exception("compute_pct4h_for_symbol failed for %s (%s)", market_symbol, dtype)
            PCT4H_CACHE[ck] = 0.0
            continue
    return 0.0

# ---- Priorities ----
def build_priorities(best_spot: Dict[str,MarketVol], best_fut: Dict[str,MarketVol]):
    p1_full, p2_full = [], []
    used = set()

    # --- P1: Fut ≥ $10M (exclude pinned) ---
    for base, f in best_fut.items():
        if base in PINNED_SET: continue
        fut_usd = usd_notional(f)
        if fut_usd >= P1_FUT_MIN:
            s = best_spot.get(base)
            pct4h = compute_pct4h_for_symbol(f.symbol, True)
            p1_full.append([base, fut_usd, usd_notional(s) if s else 0.0, pct_change(s, f), pct4h])
    p1_full.sort(key=lambda r: r[1], reverse=True)
    p1 = [row for row in p1_full if row[0] not in PINNED_SET][:TOP_N_P1]
    used.update({r[0] for r in p1})

    # --- P2: Fut ≥ $2M (exclude pinned, not already used) ---
    for base, f in best_fut.items():
        if base in used or base in PINNED_SET: continue
        fut_usd = usd_notional(f)
        if fut_usd >= P2_FUT_MIN:
            s = best_spot.get(base)
            pct4h = compute_pct4h_for_symbol(f.symbol, True)
            p2_full.append([base, fut_usd, usd_notional(s) if s else 0.0, pct_change(s, f), pct4h])
    p2_full.sort(key=lambda r: r[1], reverse=True)
    p2 = [row for row in p2_full if row[0] not in PINNED_SET][:TOP_N_P2]
    used.update({r[0] for r in p2})

    # --- P3: Always include pinned + Spot ≥ $3M ---
    p3_dict: Dict[str, List] = {}
    for base in PINNED_P3:
        s, f = best_spot.get(base), best_fut.get(base)
        if not s and not f: continue
        fut_usd = usd_notional(f) if f else 0.0
        spot_usd = usd_notional(s) if s else 0.0
        pct = pct_change(s, f)
        pct4h = compute_pct4h_for_symbol(f.symbol, True) if f else (compute_pct4h_for_symbol(s.symbol, False) if s else 0.0)
        p3_dict[base] = [base, fut_usd, spot_usd, pct, pct4h]
    for base, s in best_spot.items():
        if base in used or base in PINNED_SET: continue
        spot_usd = usd_notional(s)
        if spot_usd >= P3_SPOT_MIN:
            f = best_fut.get(base)
            pct4h = compute_pct4h_for_symbol(f.symbol, True) if f else compute_pct4h_for_symbol(s.symbol, False)
            p3_dict[base] = [base, usd_notional(f) if f else 0.0, spot_usd, pct_change(s, f), pct4h]
    pinned_rows = [r for r in p3_dict.values() if r[0] in PINNED_SET]
    other_rows  = [r for r in p3_dict.values() if r[0] not in PINNED_SET]
    pinned_rows.sort(key=lambda r: r[2], reverse=True)
    other_rows.sort(key=lambda r: r[2], reverse=True)
    p3 = (pinned_rows + other_rows)[:TOP_N_P3]

    return p1, p2, p3

# ---- Formatting & Telegram handlers (unchanged) ----
# ... keep the rest of the code exactly as in my last version ...
