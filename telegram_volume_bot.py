#!/usr/bin/env python3
"""
Telegram bot: CoinEx screener with 3 priorities
Simplified output: only SYMBOL + USD AMOUNT
Max 10 rows per priority
"""

import asyncio, logging, os, re, time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import ccxt  # type: ignore
from tabulate import tabulate  # type: ignore
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# Thresholds
P1_SPOT_MIN = 500_000      # Priority 1 spot
P1_FUT_MIN  = 5_000_000    # Priority 1 futures
P2_FUT_MIN  = 2_000_000    # Priority 2 futures
P3_SPOT_MIN = 1_000_000    # Priority 3 spot
TOP_N       = 15

TOKEN = os.environ.get("TELEGRAM_TOKEN")
EXCHANGE_ID = "coinex"
STABLES = {"USD", "USDT", "USDC", "TUSD", "FDUSD", "USDD", "USDE", "DAI", "PYUSD"}

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
    ex_spot = ccxt.__dict__[EXCHANGE_ID]({"enableRateLimit": True,"options":{"defaultType":"spot"}})
    ex_spot.load_markets()
    spot_tickers = ex_spot.fetch_tickers()
    best_spot = {}
    for _,t in spot_tickers.items():
        mv=to_mv(t)
        if not mv or mv.quote not in STABLES: continue
        if mv.base not in best_spot or usd_notional(mv)>usd_notional(best_spot[mv.base]):
            best_spot[mv.base]=mv

    ex_fut = ccxt.__dict__[EXCHANGE_ID]({"enableRateLimit": True,"options":{"defaultType":"swap"}})
    ex_fut.load_markets()
    fut_tickers = ex_fut.fetch_tickers()
    best_fut = {}
    for _,t in fut_tickers.items():
        mv=to_mv(t)
        if not mv: continue
        if mv.base not in best_fut or usd_notional(mv)>usd_notional(best_fut[mv.base]):
            best_fut[mv.base]=mv

    return best_spot,best_fut

def build_priorities(best_spot,best_fut):
    p1,p2,p3=[],[],[]
    # P1
    for base in set(best_spot)&set(best_fut):
        s,f=best_spot[base],best_fut[base]
        if usd_notional(f)>=P1_FUT_MIN and usd_notional(s)>=P1_SPOT_MIN:
            p1.append([base,usd_notional(f)])
    p1.sort(key=lambda r:r[1],reverse=True)
    used={r[0] for r in p1}

    # P2
    for base,f in best_fut.items():
        if base in used: continue
        if usd_notional(f)>=P2_FUT_MIN:
            p2.append([base,usd_notional(f)])
    p2.sort(key=lambda r:r[1],reverse=True)
    used.update({r[0] for r in p2})

    # P3
    for base,s in best_spot.items():
        if base in used: continue
        if usd_notional(s)>=P3_SPOT_MIN:
            p3.append([base,usd_notional(s)])
    p3.sort(key=lambda r:r[1],reverse=True)

    return p1[:TOP_N],p2[:TOP_N],p3[:TOP_N]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 /screen will show:\n"
        "• Priority 1: Futures ≥ $5M AND Spot ≥ $500k\n"
        "• Priority 2: Futures ≥ $2M\n"
        "• Priority 3: Spot ≥ $1M\n"
        "Only 10 rows each, with SYMBOL + USD amount."
    )

async def screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        t0=time.time()
        best_spot,best_fut=await asyncio.to_thread(load_best)
        p1,p2,p3=build_priorities(best_spot,best_fut)
        dt=time.time()-t0

        def fmt(rows,title):
            if not rows: return f"*{title}*: _None_\n"
            pretty=[[r[0],f"${r[1]:,.0f}"] for r in rows]
            return f"*{title}*:\n```\n"+tabulate(pretty,headers=["SYMBOL","USD 24h"],tablefmt="github")+"\n```\n"

        text=(
            fmt(p1,"Priority 1 (Fut≥$5M & Spot≥$500k)")+
            fmt(p2,"Priority 2 (Fut≥$2M)")+
            fmt(p3,"Priority 3 (Spot≥$1M)")+
            f"⏱️ {dt:.1f}s • CoinEx via CCXT"
        )
        await update.message.reply_text(text,parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logging.exception("screen error")
        await update.message.reply_text(f"Error: {e}")

def main():
    if not TOKEN: raise RuntimeError("Set TELEGRAM_TOKEN env var")
    app=Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start",start))
    app.add_handler(CommandHandler("screen",screen))
    app.run_polling(drop_pending_updates=True)

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    main()
