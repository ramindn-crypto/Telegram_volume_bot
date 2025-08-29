#!/usr/bin/env python3
"""
Telegram bot: CoinEx screener with 3 priorities
- Excludes: BTC, ETH, XRP, SOL, DOGE
- /screen → shows tables in chat (SYMBOL + USD amount, max 5 rows per priority)
- /excel  → sends an Excel .xlsx file (priority,symbol,usd_24h)

Priorities:
  P1: Futures >= $5,000,000 AND Spot >= $500,000
  P2: Futures >= $2,000,000  (ignore spot)
  P3: Spot    >= $1,000,000  (ignore futures)
"""

import asyncio, logging, os, time, io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import ccxt  # type: ignore
from tabulate import tabulate  # type: ignore
from openpyxl import Workbook  # type: ignore
from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# Thresholds
P1_SPOT_MIN = 500_000
P1_FUT_MIN  = 5_000_000
P2_FUT_MIN  = 2_000_000
P3_SPOT_MIN = 1_000_000
TOP_N       = 5  # << limit to 5 rows per priority

TOKEN = os.environ.get("TELEGRAM_TOKEN")
EXCHANGE_ID = "coinex"
STABLES = {"USD", "USDT", "USDC", "TUSD", "FDUSD", "USDD", "USDE", "DAI", "PYUSD"}

# Exclusions (bases not to show anywhere)
EXCLUDE_BASES = {"BTC", "ETH", "XRP", "SOL", "DOGE"}

@dataclass
class MarketVol:
    symbol: str
    base: str
    quote: str
    last: float
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
    return MarketVol(
        symbol=sym, base=base, quote=quote,
        last=float(t.get("last") or t.get("close") or 0.0),
        base_vol=float(t.get("baseVolume") or 0.0),
        quote_vol=float(t.get("quoteVolume") or 0.0),
        vwap=float(t.get("vwap") or 0.0),
    )

def usd_notional(mv: Optional[MarketVol]) -> float:
    if not mv: return 0.0
    if mv.quote in STABLES and mv.quote_vol > 0:
        return mv.quote_vol
    price = mv.vwap if mv.vwap > 0 else mv.last
    return mv.base_vol * price if price and mv.base_vol else 0.0

def load_best() -> Tuple[Dict[str, MarketVol], Dict[str, MarketVol]]:
    # SPOT
    ex_spot = ccxt.__dict__[EXCHANGE_ID]({"enableRateLimit": True,"options":{"defaultType":"spot"}})
    ex_spot.load_markets()
    spot_tickers = ex_spot.fetch_tickers()
    best_spot: Dict[str, MarketVol] = {}
    for _, t in spot_tickers.items():
        mv = to_mv(t)
        if not mv or mv.quote not in STABLES: 
            continue
        if mv.base in EXCLUDE_BASES: 
            continue
        if mv.base not in best_spot or usd_notional(mv) > usd_notional(best_spot[mv.base]):
            best_spot[mv.base] = mv

    # FUTURES (swap)
    ex_fut = ccxt.__dict__[EXCHANGE_ID]({"enableRateLimit": True,"options":{"defaultType":"swap"}})
    ex_fut.load_markets()
    fut_tickers = ex_fut.fetch_tickers()
    best_fut: Dict[str, MarketVol] = {}
    for _, t in fut_tickers.items():
        mv = to_mv(t)
        if not mv: 
            continue
        if mv.base in EXCLUDE_BASES: 
            continue
        if mv.base not in best_fut or usd_notional(mv) > usd_notional(best_fut[mv.base]):
            best_fut[mv.base] = mv

    return best_spot, best_fut

def build_priorities(best_spot, best_fut):
    # Build full lists
    p1_full, p2_full, p3_full = [], [], []

    # P1 (both thresholds)
    for base in set(best_spot) & set(best_fut):
        if base in EXCLUDE_BASES: 
            continue
        s, f = best_spot[base], best_fut[base]
        fut_usd, spot_usd = usd_notional(f), usd_notional(s)
        if fut_usd >= P1_FUT_MIN and spot_usd >= P1_SPOT_MIN:
            p1_full.append([base, fut_usd])
    p1_full.sort(key=lambda r: r[1], reverse=True)
    p1 = p1_full[:TOP_N]
    used = {r[0] for r in p1}  # exclude only what we actually show

    # P2 (futures-only, excluding shown P1)
    for base, f in best_fut.items():
        if base in used or base in EXCLUDE_BASES: 
            continue
        fut_usd = usd_notional(f)
        if fut_usd >= P2_FUT_MIN:
            p2_full.append([base, fut_usd])
    p2_full.sort(key=lambda r: r[1], reverse=True)
    p2 = p2_full[:TOP_N]
    used.update({r[0] for r in p2})

    # P3 (spot-only, excluding shown P1/P2)
    for base, s in best_spot.items():
        if base in used or base in EXCLUDE_BASES: 
            continue
        spot_usd = usd_notional(s)
        if spot_usd >= P3_SPOT_MIN:
            p3_full.append([base, spot_usd])
    p3_full.sort(key=lambda r: r[1], reverse=True)
    p3 = p3_full[:TOP_N]

    return p1, p2, p3

def fmt_table(rows: List[List], title: str) -> str:
    if not rows: return f"*{title}*: _None_\n"
    pretty = [[r[0], f"${r[1]:,.0f}"]]  # seed header fix
    pretty = [[r[0], f"${r[1]:,.0f}"] for r in rows]
    return f"*{title}*:\n```\n" + tabulate(pretty, headers=["SYMBOL","USD 24h"], tablefmt="github") + "\n```\n"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Commands:\n"
        "• /screen → 3 lists (SYMBOL + USD amount, 5 rows each, excludes BTC/ETH/XRP/SOL/DOGE)\n"
        "• /excel  → Excel file (.xlsx) of the same data"
    )

async def screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        t0 = time.time()
        best_spot, best_fut = await asyncio.to_thread(load_best)
        p1, p2, p3 = await asyncio.to_thread(build_priorities, best_spot, best_fut)
        dt = time.time() - t0
        text = (
            fmt_table(p1, "Priority 1 (Fut≥$5M & Spot≥$500k)") +
            fmt_table(p2, "Priority 2 (Fut≥$2M)") +
            fmt_table(p3, "Priority 3 (Spot≥$1M)") +
            f"⏱️ {dt:.1f}s • CoinEx via CCXT"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logging.exception("screen error")
        await update.message.reply_text(f"Error: {e}")

async def excel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        best_spot, best_fut = await asyncio.to_thread(load_best)
        p1, p2, p3 = await asyncio.to_thread(build_priorities, best_spot, best_fut)

        wb = Workbook()
        ws = wb.active
        ws.title = "Screener"
        ws.append(["priority","symbol","usd_24h"])
        for sym, usd in p1: ws.append(["P1", sym, usd])
        for sym, usd in p2: ws.append(["P2", sym, usd])
        for sym, usd in p3: ws.append(["P3", sym, usd])

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)

        await update.message.reply_document(
            document=InputFile(buf, filename="screener.xlsx"),
            caption="Excel export: priority,symbol,usd_24h (excludes BTC/ETH/XRP/SOL/DOGE)"
        )
    except Exception as e:
        logging.exception("excel error")
        await update.message.reply_text(f"Error: {e}")

def main():
    if not TOKEN: raise RuntimeError("Set TELEGRAM_TOKEN env var")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("screen", screen))
    app.add_handler(CommandHandler("excel", excel_cmd))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
