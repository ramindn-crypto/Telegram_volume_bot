#!/usr/bin/env python3
"""
CoinEx screener bot
Table columns (Telegram):
  SYM | F | S | % | %4H
  - F, S are *million USD*, rounded to integers
  - % and %4H are integers with emoji (🟢/🟡/🔴)

Features:
- /screen → P1 (10 rows), P2 (15), P3 (15; P3 always includes pinned: BTC,ETH,XRP,SOL,DOGE,ADA,PEPE,LINK)
- Type a ticker (e.g., PYTH or $PYTH) → one-row table for that coin
- /excel  → Excel .xlsx (legacy 3-col export kept)
- /diag   → diagnostics

Priority rules:
  P1: Futures ≥ $5M (EXCLUDES pinned; pinned can NEVER appear in P1)
  P2: Futures ≥ $2M             (EXCLUDES pinned; pinned can NEVER appear in P2)
  P3: Always include pinned + Spot ≥ $2M (pinned first), TOTAL 15 rows
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
P1_FUT_MIN  = 10_000_000
P2_FUT_MIN  = 2_000_000
P3_SPOT_MIN = 2_000_000

TOP_N_P1 = 10
TOP_N_P2 = 15
TOP_N_P3 = 15

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
    """24h notional in USD terms. Prefer quoteVolume if USD-quoted; else baseVolume * (vwap or last)."""
    if not mv: return 0.0
    if mv.quote in STABLES and mv.quote_vol and mv.quote_vol > 0:
        return mv.quote_vol
    price = mv.vwap if mv.vwap and mv.vwap > 0 else mv.last
    return mv.base_vol * price if price and mv.base_vol else 0.0

def pct_change(mv_spot: Optional[MarketVol], mv_fut: Optional[MarketVol]) -> float:
    """24h % change: prefer ticker 'percentage' (spot then futures); else compute from open/last."""
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
    """Return millions as an integer (rounded)."""
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
    """Return best spot/futures tickers per BASE (no exclusions)."""
    # SPOT
    ex_spot = build_exchange("spot")
    spot_tickers = safe_fetch_tickers(ex_spot)
    best_spot: Dict[str, MarketVol] = {}
    for _, t in spot_tickers.items():
        mv = to_mv(t)
        if not mv: continue
        if mv.quote not in STABLES: continue
        prev = best_spot.get(mv.base)
        if prev is None or usd_notional(mv) > usd_notional(prev):
            best_spot[mv.base] = mv

    # FUTURES (swap)
    ex_fut = build_exchange("swap")
    fut_tickers = safe_fetch_tickers(ex_fut)
    best_fut: Dict[str, MarketVol] = {}
    for _, t in fut_tickers.items():
        mv = to_mv(t)
        if not mv: continue
        prev = best_fut.get(mv.base)
        if prev is None or usd_notional(mv) > usd_notional(prev):
            best_fut[mv.base] = mv

    return best_spot, best_fut, len(spot_tickers), len(fut_tickers)

# ---- 4H % from 1h OHLCV ----
def compute_pct4h_for_symbol(market_symbol: str, prefer_swap: bool = True) -> float:
    """
    Compute % change over the last 4 completed hours using 1h candles.
    Prefer futures ('swap') series; fall back to spot if needed.
    """
    cache_key = ("swap" if prefer_swap else "spot", market_symbol)
    if cache_key in PCT4H_CACHE:
        return PCT4H_CACHE[cache_key]

    try_order = ["swap", "spot"] if prefer_swap else ["spot", "swap"]
    for dtype in try_order:
        ck = (dtype, market_symbol)
        if ck in PCT4H_CACHE:
            return PCT4H_CACHE[ck]
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
    """
    Returns:
      p1, p2, p3  where each row = [base, fut_usd, spot_usd, pct_24h, pct_4h]
    Sorting:
      P1 & P2 by FUT USD desc (EXCLUDE pinned).
      P3 always includes pinned coins + others with SPOT ≥ $3M; pinned first; cap to 10.
    """
    p1_full, p2_full = [], []
    used = set()  # bases already placed in P1 or P2

    # --- P1: Fut≥5M (EXCLUDING pinned) ---
    for base in set(best_spot) & set(best_fut):
        if base in PINNED_SET:
            continue  # hard exclude pinned from P1
        s, f = best_fut[base]
        fut_usd = usd_notional(f)
        if fut_usd >= P1_FUT_MIN:
            pct4h = compute_pct4h_for_symbol(f.symbol, True)
            p1_full.append([base, fut_usd, pct_change(f), pct4h])

    # Sort and slice
    p1_full.sort(key=lambda r: r[1], reverse=True)
    p1 = p1_full[:TOP_N_P1]
    # Safety filter: ensure no pinned in final P1
    p1 = [row for row in p1 if row[0] not in PINNED_SET]
    used.update({r[0] for r in p1})

    # --- P2: Fut≥2M (EXCLUDING pinned and already used) ---
    for base, f in best_fut.items():
        if base in used or base in PINNED_SET:
            continue  # hard exclude pinned from P2
        fut_usd = usd_notional(f)
        if fut_usd >= P2_FUT_MIN:
            s = best_spot.get(base)
            pct4h = compute_pct4h_for_symbol(f.symbol, True)
            p2_full.append([base, fut_usd, usd_notional(s) if s else 0.0, pct_change(s, f), pct4h])

    p2_full.sort(key=lambda r: r[1], reverse=True)
    p2 = p2_full[:TOP_N_P2]
    # Safety filter: ensure no pinned in final P2
    p2 = [row for row in p2 if row[0] not in PINNED_SET]
    used.update({r[0] for r in p2})

    # --- P3: Always include pinned + Spot≥3M others (not already used), pinned first ---
    p3_dict: Dict[str, List] = {}

    # Add pinned coins (even if they don't meet P3_SPOT_MIN)
    for base in PINNED_P3:
        s = best_spot.get(base)
        f = best_fut.get(base)
        if not s and not f:
            continue  # no data available
        fut_usd = usd_notional(f) if f else 0.0
        spot_usd = usd_notional(s) if s else 0.0
        pct = pct_change(s, f)
        pct4h = compute_pct4h_for_symbol(f.symbol, True) if f else (compute_pct4h_for_symbol(s.symbol, False) if s else 0.0)
        p3_dict[base] = [base, fut_usd, spot_usd, pct, pct4h]

    # Add non-pinned others meeting Spot≥3M (not already used)
    for base, s in best_spot.items():
        if base in used or base in PINNED_SET:
            continue
        spot_usd = usd_notional(s)
        if spot_usd >= P3_SPOT_MIN:
            f = best_fut.get(base)
            pct4h = compute_pct4h_for_symbol(f.symbol, True) if f else compute_pct4h_for_symbol(s.symbol, False)
            p3_dict[base] = [base, usd_notional(f) if f else 0.0, spot_usd, pct_change(s, f), pct4h]

    # Sort: pinned first by spot desc, then others by spot desc; cap to TOP_N_P3
    all_rows = list(p3_dict.values())
    pinned_rows = [r for r in all_rows if r[0] in PINNED_SET]
    other_rows  = [r for r in all_rows if r[0] not in PINNED_SET]
    pinned_rows.sort(key=lambda r: r[2], reverse=True)
    other_rows.sort(key=lambda r: r[2], reverse=True)
    p3 = (pinned_rows + other_rows)[:TOP_N_P3]

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
    await update.message.reply_text(
        "👋 Commands:\n"
        "• /screen → P1(10), P2(5), P3(10) with columns: SYM | F | S | % | %4H\n"
        "• /excel  → Excel file (.xlsx)\n"
        "• /diag   → diagnostics\n"
        "Tip: Send a ticker (e.g., PYTH) to get a one-row table for that coin."
    )

async def screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_ERROR, PCT4H_CACHE
    LAST_ERROR = None
    PCT4H_CACHE = {}
    try:
        t0 = time.time()
        best_spot, best_fut, raw_spot_count, raw_fut_count = await asyncio.to_thread(load_best)
        p1, p2, p3 = await asyncio.to_thread(build_priorities, best_spot, best_fut)
        dt = time.time() - t0
        text = (
            fmt_table(p1, f"Priority 1 (F≥$5M & S≥$500k — pinned excluded) — Top {TOP_N_P1}") +
            fmt_table(p2, f"Priority 2 (F≥$2M — pinned excluded) — Top {TOP_N_P2}") +
            fmt_table(p3, f"Priority 3 (Pinned + S≥$3M) — Top {TOP_N_P3}") +
            f"⏱️ {dt:.1f}s • CoinEx via CCXT • tickers: spot={raw_spot_count}, fut={raw_fut_count}"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        LAST_ERROR = f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=3)
        logging.exception("screen error")
        await update.message.reply_text(f"Error: {LAST_ERROR}")

async def excel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        best_spot, best_fut, *_ = await asyncio.to_thread(load_best)
        p1, p2, p3 = await asyncio.to_thread(build_priorities, best_spot, best_fut)

        wb = Workbook()
        ws = wb.active
        ws.title = "Screener"
        # Keep legacy schema (priority,symbol,usd_24h) for compatibility
        ws.append(["priority","symbol","usd_24h"])
        for sym, fut_usd, spot_usd, _, _ in p1: ws.append(["P1", sym, fut_usd])
        for sym, fut_usd, spot_usd, _, _ in p2: ws.append(["P2", sym, fut_usd])
        for sym, fut_usd, spot_usd, _, _ in p3: ws.append(["P3", sym, spot_usd])

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        await update.message.reply_document(
            document=InputFile(buf, filename="screener.xlsx"),
            caption="Excel export (priority,symbol,usd_24h)"
        )
    except Exception as e:
        logging.exception("excel error")
        await update.message.reply_text(f"Error: {e}")

async def diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        best_spot, best_fut, raw_spot_count, raw_fut_count = await asyncio.to_thread(load_best)
        msg = (
            "*Diag*\n"
            f"- thresholds: P1 F≥${P1_FUT_MIN:,} & S≥${P1_SPOT_MIN:,} | "
            f"P2 F≥${P2_FUT_MIN:,} | P3 S≥${P3_SPOT_MIN:,} (+ pinned)\n"
            f"- rows: P1={TOP_N_P1}, P2={TOP_N_P2}, P3={TOP_N_P3}\n"
            f"- tickers fetched: spot={raw_spot_count}, fut={raw_fut_count}\n"
            f"- kept bases: spot={len(best_spot)}, fut={len(best_fut)}\n"
            f"- last_error: {LAST_ERROR or '_None_'}"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"Diag error: {type(e).__name__}: {e}")

# --- Symbol lookup (no exclusions) ---
def normalize_symbol_text(text: str) -> Optional[str]:
    if not text: return None
    s = text.strip()
    candidates = re.findall(r"[A-Za-z$]{2,10}", s)
    if not candidates: return None
    token = candidates[0].upper().lstrip("$")
    token = token.replace(".", "").replace(",", "")
    return token if 2 <= len(token) <= 10 else None

async def coin_query(update: Update, symbol_text: str):
    global PCT4H_CACHE
    try:
        base = normalize_symbol_text(symbol_text)
        if not base:
            await update.message.reply_text("Please provide a ticker, e.g. `PYTH`.", parse_mode=ParseMode.MARKDOWN)
            return
        PCT4H_CACHE = {}
        best_spot, best_fut, *_ = await asyncio.to_thread(load_best)
        s, f = best_spot.get(base), best_fut.get(base)
        fut_usd = usd_notional(f) if f else 0.0
        spot_usd = usd_notional(s) if s else 0.0
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
        await update.message.reply_text(f"Error: {type(e).__name__}: {e}")

async def coin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    arg = " ".join(context.args) if context.args else ""
    await coin_query(update, arg or "")

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await coin_query(update, update.message.text or "")

def main():
    if not TOKEN:
        raise RuntimeError("Set TELEGRAM_TOKEN env var")
    logging.basicConfig(level=logging.INFO)
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("screen", screen))
    app.add_handler(CommandHandler("excel", excel_cmd))
    app.add_handler(CommandHandler("diag", diag))
    app.add_handler(CommandHandler("coin", coin_cmd))  # /coin PYTH

    # Plain-text symbol lookups (must be after commands)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
