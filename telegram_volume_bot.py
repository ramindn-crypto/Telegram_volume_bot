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

TOP_N_P1 = 10
TOP_N_P2 = 5
TOP_N_P3 = 5

EXCHANGE_ID = "coinex"
TOKEN = os.environ.get("TELEGRAM_TOKEN")
STABLES = {"USD","USDT","USDC","TUSD","FDUSD","USDD","USDE","DAI","PYUSD"}

EXCLUDE_BASES = {"BTC","ETH","XRP","SOL","DOGE","ADA","PEPE","LINK"}

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

def load_best(apply_exclusions: bool = True) -> Tuple[Dict[str, MarketVol], Dict[str, MarketVol], int, int]:
    ex_spot = build_exchange("spot")
    spot_tickers = safe_fetch_tickers(ex_spot)
    best_spot: Dict[str, MarketVol] = {}
    for _, t in spot_tickers.items():
        mv = to_mv(t)
        if not mv: continue
        if mv.quote not in STABLES: continue
        if apply_exclusions and mv.base in EXCLUDE_BASES: continue
        if mv.base not in best_spot or usd_notional(mv) > usd_notional(best_spot[mv.base]):
            best_spot[mv.base] = mv

    ex_fut = build_exchange("swap")
    fut_tickers = safe_fetch_tickers(ex_fut)
    best_fut: Dict[str, MarketVol] = {}
    for _, t in fut_tickers.items():
        mv = to_mv(t)
        if not mv: continue
        if apply_exclusions and mv.base in EXCLUDE_BASES: continue
        if mv.base not in best_fut or usd_notional(mv) > usd_notional(best_fut[mv.base]):
            best_fut[mv.base] = mv

    return best_spot, best_fut, len(spot_tickers), len(fut_tickers)

# ---- 4H % ----
def compute_pct4h_for_symbol(market_symbol: str, prefer_swap: bool = True) -> float:
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
            pct4h = ((closes[-1] - closes[0]) / closes[0] * 100.0) if closes[0] else 0.0
            PCT4H_CACHE[cache_key] = pct4h
            return pct4h
        except Exception:
            logging.exception("compute_pct4h_for_symbol failed for %s (%s)", market_symbol, dtype)
            PCT4H_CACHE[cache_key] = 0.0
            continue
    return 0.0

# ---- Priorities ----
def build_priorities(best_spot: Dict[str,MarketVol], best_fut: Dict[str,MarketVol]):
    p1_full, p2_full, p3_full = [], [], []

    for base in set(best_spot) & set(best_fut):
        if base in EXCLUDE_BASES: continue
        s, f = best_spot[base], best_fut[base]
        fut_usd, spot_usd = usd_notional(f), usd_notional(s)
        if fut_usd >= P1_FUT_MIN and spot_usd >= P1_SPOT_MIN:
            pct4h = compute_pct4h_for_symbol(f.symbol, True)
            p1_full.append([base, fut_usd, spot_usd, pct_change(s, f), pct4h])
    p1_full.sort(key=lambda r: r[1], reverse=True)
    p1 = p1_full[:TOP_N_P1]
    used = {r[0] for r in p1}

    for base, f in best_fut.items():
        if base in used or base in EXCLUDE_BASES: continue
        fut_usd = usd_notional(f)
        if fut_usd >= P2_FUT_MIN:
            s = best_spot.get(base)
            pct4h = compute_pct4h_for_symbol(f.symbol, True)
            p2_full.append([base, fut_usd, usd_notional(s) if s else 0.0, pct_change(s, f), pct4h])
    p2_full.sort(key=lambda r: r[1], reverse=True)
    p2 = p2_full[:TOP_N_P2]
    used.update({r[0] for r in p2})

    for base, s in best_spot.items():
        if base in used or base in EXCLUDE_BASES: continue
        spot_usd = usd_notional(s)
        if spot_usd >= P3_SPOT_MIN:
            f = best_fut.get(base)
            if f:
                pct4h = compute_pct4h_for_symbol(f.symbol, True)
            else:
                pct4h = compute_pct4h_for_symbol(s.symbol, False)
            p3_full.append([base, usd_notional(f) if f else 0.0, spot_usd, pct_change(s, f), pct4h])
    p3_full.sort(key=lambda r: r[2], reverse=True)
    p3 = p3_full[:TOP_N_P3]

    return p1, p2, p3

# ---- Formatting ----
def fmt_table(rows: List[List], title: str) -> str:
    if not rows: return f"*{title}*: _None_\n"
    pretty = [[r[0], m_dollars_int(r[1]), m_dollars_int(r[2]), pct_with_emoji(r[3]), pct_with_emoji(r[4])] for r in rows]
    return f"*{title}*:\n```\n" + tabulate(pretty, headers=["SYM","F","S","%","%4H"], tablefmt="github") + "\n```\n"

def fmt_table_single(sym: str, fut_usd: float, spot_usd: float, pct: float, pct4h: float, title: str) -> str:
    row = [[sym.upper(), m_dollars_int(fut_usd), m_dollars_int(spot_usd), pct_with_emoji(pct), pct_with_emoji(pct4h)]]
    return f"*{title}*:\n```\n" + tabulate(row, headers=["SYM","F","S","%","%4H"], tablefmt="github") + "\n```\n"

# ---- Telegram handlers ----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Commands:\n• /screen → P1(10), P2(5), P3(5)\n• /excel  → Excel file\n• /diag   → diagnostics\nTip: Send a ticker (e.g., PYTH) to get info.")

async def screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_ERROR, PCT4H_CACHE
    LAST_ERROR = None
    PCT4H_CACHE = {}
    try:
        t0 = time.time()
        best_spot, best_fut, raw_spot_count, raw_fut_count = await asyncio.to_thread(load_best, True)
        p1, p2, p3 = await asyncio.to_thread(build_priorities, best_spot, best_fut)
        dt = time.time() - t0
        text = fmt_table(p1, f"Priority 1 (F≥$5M & S≥$500k) — Top {TOP_N_P1}") + \
               fmt_table(p2, f"Priority 2 (F≥$2M) — Top {TOP_N_P2}") + \
               fmt_table(p3, f"Priority 3 (S≥$3M) — Top {TOP_N_P3}") + \
               f"⏱️ {dt:.1f}s • CoinEx via CCXT • tickers: spot={raw_spot_count}, fut={raw_fut_count}"
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        LAST_ERROR = f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=3)
        logging.exception("screen error")
        await update.message.reply_text(f"Error: {LAST_ERROR}")

async def excel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        best_spot, best_fut, *_ = await asyncio.to_thread(load_best, True)
        p1, p2, p3 = await asyncio.to_thread(build_priorities, best_spot, best_fut)
        wb = Workbook(); ws = wb.active; ws.title = "Screener"
        ws.append(["priority","symbol","usd_24h"])
        for sym, fut_usd, spot_usd, _, _ in p1: ws.append(["P1", sym, fut_usd])
        for sym, fut_usd, spot_usd, _, _ in p2: ws.append(["P2", sym, fut_usd])
        for sym, fut_usd, spot_usd, _, _ in p3: ws.append(["P3", sym, spot_usd])
        buf = io.BytesIO(); wb.save(buf); buf.seek(0)
        await update.message.reply_document(document=InputFile(buf, filename="screener.xlsx"), caption="Excel export")
    except Exception as e:
        logging.exception("excel error")
        await update.message.reply_text(f"Error: {e}")

async def diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        best_spot, best_fut, raw_spot_count, raw_fut_count = await asyncio.to_thread(load_best, True)
        msg = f"*Diag*\n- thresholds: P1 F≥{P1_FUT_MIN:,}, S≥{P1_SPOT_MIN:,} | P2 F≥{P2_FUT_MIN:,} | P3 S≥{P3_SPOT_MIN:,}\n- P1 rows: {TOP_N_P1}, P2: {TOP_N_P2}, P3: {TOP_N_P3}\n- excludes: {', '.join(sorted(EXCLUDE_BASES))}\n- tickers fetched: spot={raw_spot_count}, fut={raw_fut_count}\n- kept: spot={len(best_spot)}, fut={len(best_fut)}\n- last_error: {LAST_ERROR or '_None_'}"
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"Diag error: {e}")

# --- Symbol lookup ---
def normalize_symbol_text(text: str) -> Optional[str]:
    if not text: return None
    s = text.strip()
    candidates = re.findall(r"[A-Za-z$]{2,10}", s)
    if not candidates: return None
    token = candidates[0].upper().lstrip("$")
    return token if 2 <= len(token) <= 10 else None

async def coin_query(update: Update, symbol_text: str):
    global PCT4H_CACHE
    try:
        base = normalize_symbol_text(symbol_text)
        if not base:
            await update.message.reply_text("Please provide a ticker, e.g. `PYTH`.", parse_mode=ParseMode.MARKDOWN)
            return
        PCT4H_CACHE = {}
        best_spot, best_fut, *_ = await asyncio.to_thread(load_best, False)
        s, f = best_spot.get(base), best_fut.get(base)
        fut_usd, spot_usd = usd_notional(f) if f else 0.0, usd_notional(s) if s else 0.0
        pct = pct_change(s, f)
        pct4h = 0.0
        if f: pct4h = await asyncio.to_thread(compute_pct4h_for_symbol, f.symbol, True)
        elif s: pct4h = await asyncio.to_thread(compute_pct4h_for_symbol, s.symbol, False)
        if fut_usd == 0.0 and spot_usd == 0.0:
            await update.message.reply_text(f"Couldn't find data for `{base}`.", parse_mode=ParseMode.MARKDOWN)
            return
        text = fmt_table_single(base, fut_usd, spot_usd, pct, pct4h, f"{base} (24h / 4h)")
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logging.exception("coin query error")
        await update.message.reply_text(f"Error: {e}")

async def coin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    arg = " ".join(context.args) if context.args else ""
    await coin_query(update, arg)

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await coin_query(update, update.message.text or "")

def main():
    if not TOKEN: raise RuntimeError("
