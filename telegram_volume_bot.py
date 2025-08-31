#!/usr/bin/env python3
"""
CoinEx screener bot
Output columns (Telegram /screen or symbol lookup):
  SYM | FUT | SPOT | % (with emoji, integers only)

Features:
- /screen → P1: 10 rows, P2: 5 rows, P3: 5 rows
- Type a coin ticker/name (e.g., "PYTH" or "$PYTH") → same table for that coin (ignores exclusions)
- /excel  → Excel .xlsx (priority,symbol,usd_24h)
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

# Exclusions (lists only — NOT applied to direct symbol query)
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
    """Return best spot/fut tickers per BASE.
       If apply_exclusions=False, EXCLUDE_BASES is ignored (used for direct symbol query)."""
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

def build_priorities(best_spot, best_fut):
    """Return rows as [base, fut_usd, spot_usd, pct_24h]."""
    p1_full, p2_full, p3_full = [], [], []

    # P1
    for base in set(best_spot) & set(best_fut):
        if base in EXCLUDE_BASES: continue
        s, f = best_spot[base], best_fut[base]
        fut_usd, spot_usd = usd_notional(f), usd_notional(s)
        if fut_usd >= P1_FUT_MIN and spot_usd >= P1_SPOT_MIN:
            p1_full.append([base, fut_usd, spot_usd, pct_change(s, f)])
    p1_full.sort(key=lambda r: r[1], reverse=True)
    p1 = p1_full[:TOP_N_P1]
    used = {r[0] for r in p1}

    # P2
    for base, f in best_fut.items():
        if base in used or base in EXCLUDE_BASES: continue
        fut_usd = usd_notional(f)
        if fut_usd >= P2_FUT_MIN:
            s = best_spot.get(base)
            p2_full.append([base, fut_usd, usd_notional(s) if s else 0.0, pct_change(s, f)])
    p2_full.sort(key=lambda r: r[1], reverse=True)
    p2 = p2_full[:TOP_N_P2]
    used.update({r[0] for r in p2})

    # P3
    for base, s in best_spot.items():
        if base in used or base in EXCLUDE_BASES: continue
        spot_usd = usd_notional(s)
        if spot_usd >= P3_SPOT_MIN:
            f = best_fut.get(base)
            p3_full.append([base, usd_notional(f) if f else 0.0, spot_usd, pct_change(s, f)])
    p3_full.sort(key=lambda r: r[2], reverse=True)
    p3 = p3_full[:TOP_N_P3]

    return p1, p2, p3

def fmt_table(rows: List[List], title: str) -> str:
    if not rows: return f"*{title}*: _None_\n"
    pretty = [[r[0], m_dollars_int(r[1]), m_dollars_int(r[2]), pct_with_emoji(r[3])] for r in rows]
    return (
        f"*{title}*:\n"
        "```\n" + tabulate(pretty,
            headers=["SYM","FUT","SPOT","%"],
            tablefmt="github"
        ) + "\n```\n"
    )

def fmt_table_single(sym: str, fut_usd: float, spot_usd: float, pct: float, title: str) -> str:
    rows = [[sym.upper(), m_dollars_int(fut_usd), m_dollars_int(spot_usd), pct_with_emoji(pct)]]
    return (
        f"*{title}*:\n"
        "```\n" + tabulate(rows, headers=["SYM","FUT","SPOT","%"], tablefmt="github") + "\n```\n"
    )

# ---- Telegram handlers ----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Commands:\n"
        "• /screen → P1(10), P2(5), P3(5) lists\n"
        "• /excel  → Excel file (.xlsx)\n"
        "• /diag   → diagnostics\n"
        "Tip: Send a coin ticker (e.g., PYTH or $PYTH) to get its table."
    )

async def screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_ERROR
    LAST_ERROR = None
    try:
        t0 = time.time()
        best_spot, best_fut, raw_spot_count, raw_fut_count = await asyncio.to_thread(load_best, True)
        p1, p2, p3 = await asyncio.to_thread(build_priorities, best_spot, best_fut)
        dt = time.time() - t0
        text = (
            fmt_table(p1, f"Priority 1 (Fut≥$5M & Spot≥$500k) — Top {TOP_N_P1}") +
            fmt_table(p2, f"Priority 2 (Fut≥$2M) — Top {TOP_N_P2}") +
            fmt_table(p3, f"Priority 3 (Spot≥$3M) — Top {TOP_N_P3}") +
            f"⏱️ {dt:.1f}s • CoinEx via CCXT • tickers: spot={raw_spot_count}, fut={raw_fut_count}"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        LAST_ERROR = f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=3)
        logging.exception("screen error")
        await update.message.reply_text(f"Error: {LAST_ERROR}")

async def excel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        best_spot, best_fut, *_ = await asyncio.to_thread(load_best, True)
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
            caption="Excel export (priority,symbol,usd_24h)"
        )
    except Exception as e:
        logging.exception("excel error")
        await update.message.reply_text(f"Error: {e}")

async def diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        best_spot, best_fut, raw_spot_count, raw_fut_count = await asyncio.to_thread(load_best, True)
        msg = (
            "*Diag*\n"
            f"- thresholds: P1 Fut≥${P1_FUT_MIN:,} & Spot≥${P1_SPOT_MIN:,} | "
            f"P2 Fut≥${P2_FUT_MIN:,} | P3 Spot≥${P3_SPOT_MIN:,}\n"
            f"- P1 rows: {TOP_N_P1}, P2 rows: {TOP_N_P2}, P3 rows: {TOP_N_P3}\n"
            f"- excludes (lists): {', '.join(sorted(EXCLUDE_BASES))}\n"
            f"- tickers fetched: spot={raw_spot_count}, fut={raw_fut_count}\n"
            f"- bases kept: spot={len(best_spot)}, fut={len(best_fut)}\n"
            f"- last_error: {LAST_ERROR or '_None_'}"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"Diag error: {type(e).__name__}: {e}")

# --- Symbol lookup (free text and /coin) ---
def normalize_symbol_text(text: str) -> Optional[str]:
    # Accept "$PYTH", "pyth", "PYTH.", "pyth/usdt", etc. → "PYTH"
    if not text: return None
    s = text.strip()
    # If user pasted a sentence, take the first token that looks like a ticker
    candidates = re.findall(r"[A-Za-z$]{2,10}", s)
    if not candidates: return None
    token = candidates[0].upper().lstrip("$")
    # strip common suffixes
    token = token.replace(".", "").replace(",", "")
    return token if 2 <= len(token) <= 10 else None

async def coin_query(update: Update, symbol_text: str):
    try:
        base = normalize_symbol_text(symbol_text)
        if not base:
            await update.message.reply_text("Please provide a coin ticker, e.g. `PYTH`.", parse_mode=ParseMode.MARKDOWN)
            return

        # Load without exclusions so you can query BTC/ETH/etc.
        best_spot, best_fut, raw_spot_count, raw_fut_count = await asyncio.to_thread(load_best, False)
        s = best_spot.get(base)
        f = best_fut.get(base)

        # If not found in best maps (because no USD-quoted spot etc.), try a fallback scan:
        if not s or not f:
            # quick pass through markets to find any symbol that matches base
            # This keeps it lightweight; main path already covers most cases.
            pass  # keeping it simple; best_* maps usually have the base if listed

        fut_usd = usd_notional(f) if f else 0.0
        spot_usd = usd_notional(s) if s else 0.0
        pct = pct_change(s, f)

        if fut_usd == 0.0 and spot_usd == 0.0:
            await update.message.reply_text(f"Couldn't find data for `{base}`.", parse_mode=ParseMode.MARKDOWN)
            return

        title = f"{base} (24h)"
        text = fmt_table_single(base, fut_usd, spot_usd, pct, title)
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logging.exception("coin query error")
        await update.message.reply_text(f"Error: {type(e).__name__}: {e}")

async def coin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    arg = " ".join(context.args) if context.args else ""
    await coin_query(update, arg or "")

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # If it's plain text (not a command), treat it as a coin query
    await coin_query(update, update.message.text or "")

def main():
    if not TOKEN: raise RuntimeError("Set TELEGRAM_TOKEN env var")
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
