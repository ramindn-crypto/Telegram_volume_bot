#!/usr/bin/env python3
"""
Telegram bot: CoinEx spot+futures USD-volume screener with 3 priorities

Priorities (all returned in one /screen):
  P1: Futures >= $5,000,000 AND Spot >= $500,000   (sorted by Futures USD vol)
  P2: Futures >= $2,000,000                        (sorted by Futures USD vol) [excluding P1 coins]
  P3: Spot    >= $1,000,000                        (sorted by Spot USD vol)    [excluding P1 & P2 coins]

Commands:
  /start
  /screen
  /screen spot=500000 fut=5000000 fut2=2000000  (optional overrides)
"""

from __future__ import annotations
import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from tabulate import tabulate  # type: ignore
import ccxt  # type: ignore
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ---- Defaults (USD) ----
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "coinex").lower()
P1_SPOT_MIN = float(os.getenv("SPOT_MIN_USD", 500_000))   # P1 spot threshold
P1_FUT_MIN  = float(os.getenv("FUTURES_MIN_USD", 5_000_000))  # P1 futures threshold
P2_FUT_MIN  = float(os.getenv("FUTURES_MIN_USD_P2", 2_000_000))  # P2 futures-only threshold
P3_SPOT_MIN = float(os.getenv("SPOT_MIN_USD_P3", 1_000_000))     # P3 spot-only threshold

TOKEN = os.environ.get("TELEGRAM_TOKEN")

STABLES = {"USD", "USDT", "USDC", "TUSD", "FDUSD", "USDD", "USDE", "DAI", "PYUSD"}

@dataclass
class MarketVol:
    symbol: str
    base: str
    quote: str
    last: float
    base_vol: float
    quote_vol: float  # in quote units; for stable quotes, treat as USD

def safe_split_symbol(sym: Optional[str]) -> Optional[Tuple[str, str]]:
    """Return (base, quote) if symbol looks like 'BASE/QUOTE[:QUOTE]'; else None."""
    if not sym:
        return None
    pair = sym.split(":")[0]  # drop suffix like ':USDT'
    if "/" not in pair:
        return None
    base, quote = pair.split("/", 1)
    if not base or not quote:
        return None
    return base, quote

def to_marketvol(t: dict) -> Optional[MarketVol]:
    sym = t.get("symbol")
    split = safe_split_symbol(sym)
    if not split:
        return None
    base, quote = split
    last = float(t.get("last") or t.get("close") or 0.0)
    base_vol = float(t.get("baseVolume") or 0.0)
    quote_vol = float(t.get("quoteVolume") or 0.0)
    return MarketVol(symbol=sym, base=base, quote=quote, last=last, base_vol=base_vol, quote_vol=quote_vol)

def fmt_money(x: float) -> str:
    try:
        return f"{x:,.0f}"
    except Exception:
        return str(x)

def load_best_by_base() -> Tuple[Dict[str, MarketVol], Dict[str, MarketVol]]:
    """Return (best_spot_by_base, best_futures_by_base), keeping highest quote_vol per base."""
    # Spot
    spot = ccxt.__dict__[EXCHANGE_ID]({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    spot.load_markets()
    spot_tickers = spot.fetch_tickers()

    best_spot: Dict[str, MarketVol] = {}
    for _, t in spot_tickers.items():
        try:
            mv = to_marketvol(t)
            if not mv:
                continue
            if mv.quote not in STABLES:  # keep USD-like quotes only
                continue
            prev = best_spot.get(mv.base)
            if prev is None or mv.quote_vol > prev.quote_vol:
                best_spot[mv.base] = mv
        except Exception:
            continue

    # Futures (swap)
    swap = ccxt.__dict__[EXCHANGE_ID]({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    swap.load_markets()
    swap_tickers = swap.fetch_tickers()

    best_fut: Dict[str, MarketVol] = {}
    for _, t in swap_tickers.items():
        try:
            mv = to_marketvol(t)
            if not mv:
                continue
            if mv.quote not in STABLES:
                continue
            prev = best_fut.get(mv.base)
            if prev is None or mv.quote_vol > prev.quote_vol:
                best_fut[mv.base] = mv
        except Exception:
            continue

    return best_spot, best_fut

def build_priorities(
    best_spot: Dict[str, MarketVol],
    best_fut: Dict[str, MarketVol],
    p1_spot_min: float,
    p1_fut_min: float,
    p2_fut_min: float,
    p3_spot_min: float,
):
    # Priority 1: both thresholds
    p1 = []
    for base in sorted(set(best_spot.keys()) & set(best_fut.keys())):
        s = best_spot[base]
        f = best_fut[base]
        if f.quote_vol >= p1_fut_min and s.quote_vol >= p1_spot_min:
            p1.append([base, f.symbol, f.quote_vol, s.symbol, s.quote_vol, f.last or s.last])
    # sort by futures USD volume desc
    p1.sort(key=lambda r: r[2], reverse=True)

    used_bases = {row[0] for row in p1}

    # Priority 2: futures-only >= p2_fut_min (exclude P1 bases)
    p2 = []
    for base, f in best_fut.items():
        if base in used_bases:
            continue
        if f.quote_vol >= p2_fut_min:
            s = best_spot.get(base)
            p2.append([base, f.symbol, f.quote_vol, s.symbol if s else "-", s.quote_vol if s else 0.0, f.last])
    p2.sort(key=lambda r: r[2], reverse=True)
    used_bases.update({row[0] for row in p2})

    # Priority 3: spot-only >= p3_spot_min (exclude P1 & P2 bases)
    p3 = []
    for base, s in best_spot.items():
        if base in used_bases:
            continue
        if s.quote_vol >= p3_spot_min:
            f = best_fut.get(base)
            p3.append([base, f.symbol if f else "-", f.quote_vol if f else 0.0, s.symbol, s.quote_vol, s.last])
    # sort by spot USD volume desc (index 4)
    p3.sort(key=lambda r: r[4], reverse=True)

    return p1, p2, p3

# ---- Telegram handlers ----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "👋 Ready!\n"
        "• Priority 1: Futures ≥ $5M AND Spot ≥ $500k\n"
        "• Priority 2: Futures ≥ $2M (ignore spot)\n"
        "• Priority 3: Spot ≥ $1M (ignore futures)\n\n"
        "Use /screen to get all three, sorted by USD volume.\n"
        "You can override thresholds e.g. `/screen spot=600000 fut=7000000 fut2=3000000`"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

def parse_overrides(args: List[str]) -> Tuple[float, float, float, float]:
    """Allow optional overrides via /screen spot=... fut=... fut2=... spot3=..."""
    p1_spot = P1_SPOT_MIN
    p1_fut  = P1_FUT_MIN
    p2_fut  = P2_FUT_MIN
    p3_spot = P3_SPOT_MIN
    text = " ".join(args or [])
    m = re.search(r"spot=(\d+(?:\.\d+)?)", text)
    if m: p1_spot = float(m.group(1))
    m = re.search(r"fut=(\d+(?:\.\d+)?)", text)
    if m: p1_fut = float(m.group(1))
    m = re.search(r"fut2=(\d+(?:\.\d+)?)", text)
    if m: p2_fut = float(m.group(1))
    m = re.search(r"spot3=(\d+(?:\.\d+)?)", text)
    if m: p3_spot = float(m.group(1))
    return p1_spot, p1_fut, p2_fut, p3_spot

async def screen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        p1_spot_min, p1_fut_min, p2_fut_min, p3_spot_min = parse_overrides(context.args)
        t0 = time.time()
        best_spot, best_fut = await asyncio.to_thread(load_best_by_base)
        p1, p2, p3 = build_priorities(best_spot, best_fut, p1_spot_min, p1_fut_min, p2_fut_min, p3_spot_min)
        dt = time.time() - t0

        def fmt_table(rows: List[List], headers: List[str]) -> str:
            if not rows:
                return "_None_"
            pretty = []
            for r in rows:
                # r = [base, fut_sym, fut_usd, spot_sym, spot_usd, last]
                pretty.append([r[0], r[1], f"${fmt_money(r[2])}", r[3], f"${fmt_money(r[4])}", r[5]])
            return "```\n" + tabulate(pretty, headers=headers, tablefmt="github") + "\n```"

        txt = (
            f"*Priority 1* (Fut ≥ ${fmt_money(p1_fut_min)} AND Spot ≥ ${fmt_money(p1_spot_min)})\n" +
            fmt_table(p1, ["BASE","FUT SYMBOL","FUT 24h USD","SPOT SYMBOL","SPOT 24h USD","LAST"]) + "\n\n" +
            f"*Priority 2* (Fut ≥ ${fmt_money(p2_fut_min)})\n" +
            fmt_table(p2, ["BASE","FUT SYMBOL","FUT 24h USD","SPOT SYMBOL","SPOT 24h USD","LAST"]) + "\n\n" +
            f"*Priority 3* (Spot ≥ ${fmt_money(p3_spot_min)})\n" +
            fmt_table(p3, ["BASE","FUT SYMBOL","FUT 24h USD","SPOT SYMBOL","SPOT 24h USD","LAST"]) + "\n\n" +
            f"⏱️ {dt:.1f}s • Source: CoinEx via CCXT"
        )
        await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logging.exception("screen error")
        await update.message.reply_text(f"Error: {e}")

def main() -> None:
    if not TOKEN:
        raise RuntimeError("Set TELEGRAM_TOKEN env var on the server")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("screen", screen))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
