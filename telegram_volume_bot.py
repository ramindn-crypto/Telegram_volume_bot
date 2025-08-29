#!/usr/bin/env python3
"""
CoinEx screener bot (vertical view)
- /screen → vertical blocks with: sym, fut($M), spot($M), %24h (top 5 per priority)
- /excel  → Excel .xlsx (priority,symbol,usd_24h)
- /diag   → diagnostics

Rules:
  P1: Futures >= $5,000,000 AND Spot >= $500,000   (sorted by futures)
  P2: Futures >= $2,000,000                        (sorted by futures)
  P3: Spot    >= $3,000,000                        (sorted by spot)
Excludes: BTC, ETH, XRP, SOL, DOGE, ADA, PEPE, LINK
"""

import asyncio, logging, os, time, io, traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import ccxt  # type: ignore
from openpyxl import Workbook  # type: ignore
from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ---- Config / thresholds ----
P1_SPOT_MIN = 500_000
P1_FUT_MIN  = 5_000_000
P2_FUT_MIN  = 2_000_000
P3_SPOT_MIN = 3_000_000
TOP_N       = 5

EXCHANGE_ID = "coinex"
TOKEN = os.environ.get("TELEGRAM_TOKEN")
STABLES = {"USD","USDT","USDC","TUSD","FDUSD","USDD","USDE","DAI","PYUSD"}
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

def m_dollars(x: float) -> str:
    m = x / 1_000_000.0
    return f"{m:.0f}" if abs(m) >= 10 else f"{m:.1f}"

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
    # SPOT
    ex_spot = build_exchange("spot")
    spot_tickers = safe_fetch_tickers(ex_spot)
    best_spot: Dict[str, MarketVol] = {}
    for _, t in spot_tickers.items():
        mv = to_mv(t)
        if not mv: continue
        if mv.quote not in STABLES: continue
        if mv.base in EXCLUDE_BASES: continue
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
        if mv.base in EXCLUDE_BASES: continue
        prev = best_fut.get(mv.base)
        if prev is None or usd_notional(mv) > usd_notional(prev):
            best_fut[mv.base] = mv

    return best_spot, best_fut, len(spot_tickers), len(fut_tickers)

def build_priorities(best_spot, best_fut):
    """
    Returns rows as [base, fut_usd, spot_usd, pct]
    P1/P2 sorted by fut_usd desc; P3 by spot_usd desc
    """
    p1_full, p2_full, p3_full = [], [], []

    # P1
    for base in set(best_spot) & set(best_fut):
        if base in EXCLUDE_BASES: continue
        s, f = best_spot[base], best_fut[base]
        fut_usd, spot_usd = usd_notional(f), usd_notional(s)
        if fut_usd >= P1_FUT_MIN and spot_usd >= P1_SPOT_MIN:
            p1_full.append([base, fut_usd, spot_usd, pct_change(s, f)])
    p1_full.sort(key=lambda r: r[1], reverse=True)
    p1 = p1_full[:TOP_N]
    used = {r[0] for r in p1}

    # P2
    for base, f in best_fut.items():
        if base in used or base in EXCLUDE_BASES: continue
        fut_usd = usd_notional(f)
        if fut_usd >= P2_FUT_MIN:
            s = best_spot.get(base)
            p2_full.append([base, fut_usd, usd_notional(s) if s else 0.0, pct_change(s, f)])
    p2_full.sort(key=lambda r: r[1], reverse=True)
    p2 = p2_full[:TOP_N]
    used.update({r[0] for r in p2})

    # P3
    for base, s in best_spot.items():
        if base in used or base in EXCLUDE_BASES: continue
        spot_usd = usd_notional(s)
        if spot_usd >= P3_SPOT_MIN:
            f = best_fut.get(base)
            p3_full.append([base, usd_notional(f) if f else 0.0, spot_usd, pct_change(s, f)])
    p3_full.sort(key=lambda r: r[2], reverse=True)
    p3 = p3_full[:TOP_N]

    return p1, p2, p3

# ---- pretty printing (VERTICAL) ----
def fmt_vertical(rows: List[List]) -> str:
    """Return a compact vertical block for each row."""
    if not rows:
        return "_None_\n"
    parts = []
    for base, fut_usd, spot_usd, pct in rows:
        parts.append(
            f"sym: {base}\n"
            f"fut($M): {m_dollars(fut_usd)}\n"
            f"spot($M): {m_dollars(spot_usd)}\n"
            f"%24h: {pct:+.1f}%"
        )
    return "```\n" + "\n\n".join(parts) + "\n```"

# ---- Telegram handlers ----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 /screen shows vertical blocks per coin with: sym, fut($M), spot($M), %24h\n"
        "Top 5 per priority • Excludes BTC/ETH/XRP/SOL/DOGE/ADA/PEPE/LINK\n"
        "Other cmds: /excel (xlsx), /diag (diagnostics)"
    )

async def screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_ERROR
    LAST_ERROR = None
    try:
        t0 = time.time()
        best_spot, best_fut, raw_spot_count, raw_fut_count = await asyncio.to_thread(load_best)
        p1, p2, p3 = await asyncio.to_thread(build_priorities, best_spot, best_fut)
        dt = time.time() - t0
        text = (
            "*Priority 1* (Fut≥$5M & Spot≥$500k)\n" + fmt_vertical(p1) + "\n" +
            "*Priority 2* (Fut≥$2M)\n" + fmt_vertical(p2) + "\n" +
            "*Priority 3* (Spot≥$3M)\n" + fmt_vertical(p3) + "\n" +
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
        ws.append(["priority","symbol","usd_24h"])
        for sym, fut_usd, spot_usd, _ in p1: ws.append(["P1", sym, fut_usd])
        for sym, fut_usd, spot_usd, _ in p2: ws.append(["P2", sym, fut_usd])
        for sym, fut_usd, spot_usd, _ in p3: ws.append(["P3", sym, spot_usd])

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)

        await update.message.reply_document(
            document=InputFile(buf, filename="screener.xlsx"),
            caption="Excel export (priority,symbol,usd_24h) — excludes BTC/ETH/XRP/SOL/DOGE/ADA/PEPE/LINK"
        )
    except Exception as e:
        logging.exception("excel error")
        await update.message.reply_text(f"Error: {e}")

async def diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        best_spot, best_fut, raw_spot_count, raw_fut_count = await asyncio.to_thread(load_best)
        msg = (
            "*Diag*\n"
            f"- thresholds: P1 Fut≥${P1_FUT_MIN:,} & Spot≥${P1_SPOT_MIN:,} | "
            f"P2 Fut≥${P2_FUT_MIN:,} | P3 Spot≥${P3_SPOT_MIN:,}\n"
            f"- excludes: {', '.join(sorted(EXCLUDE_BASES))}\n"
            f"- tickers fetched: spot={raw_spot_count}, fut={raw_fut_count}\n"
            f"- bases kept: spot={len(best_spot)}, fut={len(best_fut)}\n"
            f"- last_error: {LAST_ERROR or '_None_'}"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"Diag error: {type(e).__name__}: {e}")

def main():
    if not TOKEN: raise RuntimeError("Set TELEGRAM_TOKEN env var")
    logging.basicConfig(level=logging.INFO)
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("screen", screen))
    app.add_handler(CommandHandler("excel", excel_cmd))
    app.add_handler(CommandHandler("diag", diag))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
