#!/usr/bin/env python3
"""
CoinEx screener bot → Telegram
Also appends /screen results to Google Sheets (if enabled).

Columns:
  SYM | F | S | % | %4H
  - F, S: million USD (rounded ints)
  - % and %4H: integer with emoji (🟢/🟡/🔴)

Priorities:
  P1 (Top 10, non-pinned only):
     (A) Futures ≥ $5M AND Spot ≥ $500k
  OR (B) max(F, S) ≥ $500k AND %4H ≥ +10%
  P2 (Top 5, non-pinned): Futures ≥ $2M
  P3 (Top 10): Always include pinned [BTC, ETH, XRP, SOL, DOGE, ADA, PEPE, LINK] + others Spot ≥ $3M (pinned first)

Notes:
- Pinned coins NEVER appear in P1/P2. They’re forced to P3 only.
- /excel export keeps legacy 3-col schema.
"""

import asyncio, logging, os, time, io, traceback, re, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import ccxt  # type: ignore
from tabulate import tabulate  # type: ignore
from openpyxl import Workbook  # type: ignore
from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# ----- Optional Google Sheets integration -----
ENABLE_SHEET_APPEND = os.environ.get("ENABLE_SHEET_APPEND", "0") == "1"
GSHEET_ID = os.environ.get("GSHEET_ID", "")
GSHEET_TAB = os.environ.get("GSHEET_TAB", "Journal")
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")

if ENABLE_SHEET_APPEND:
    import gspread  # type: ignore
    from google.oauth2.service_account import Credentials  # type: ignore

# ---- Thresholds / settings ----
P1_SPOT_MIN = 500_000
P1_FUT_MIN  = 5_000_000
P1_ALT_MIN  = 500_000     # your new rule: >= $500k (either F or S) AND %4H >= +10%
P1_ALT_PCT4H_MIN = 10.0

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
    # SPOT
    ex_spot = build_exchange("spot")
    spot_tickers = safe_fetch_tickers(ex_spot)
    best_spot: Dict[str, MarketVol] = {}
    for _, t in spot_tickers.items():
        mv = to_mv(t)
        if not mv: continue
        if mv.quote not in STABLES: continue
        if mv.base not in best_spot or usd_notional(mv) > usd_notional(best_spot[mv.base]):
            best_spot[mv.base] = mv

    # FUTURES
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
      p1, p2, p3 with rows: [base, fut_usd, spot_usd, pct_24h, pct_4h]
    """
    p1_candidates = []
    used = set()

    # --- Build P1 candidates (exclude pinned) ---
    for base in set(best_spot) | set(best_fut):
        if base in PINNED_SET:
            continue
        s = best_spot.get(base)
        f = best_fut.get(base)
        if not s and not f:
            continue

        fut_usd = usd_notional(f) if f else 0.0
        spot_usd = usd_notional(s) if s else 0.0
        pct24 = pct_change(s, f)

        # %4H from futures if available else from spot
        pct4h = 0.0
        if f:
            pct4h = compute_pct4h_for_symbol(f.symbol, True)
        elif s:
            pct4h = compute_pct4h_for_symbol(s.symbol, False)

        # Rule A (original): F ≥ $5M AND S ≥ $500k
        rule_a = (fut_usd >= P1_FUT_MIN and spot_usd >= P1_SPOT_MIN)
        # Rule B (your new rule): max(F,S) ≥ $500k AND %4H ≥ +10
        rule_b = (max(fut_usd, spot_usd) >= P1_ALT_MIN and pct4h >= P1_ALT_PCT4H_MIN)

        if rule_a or rule_b:
            p1_candidates.append([base, fut_usd, spot_usd, pct24, pct4h])

    # Sort by F USD (desc) to keep consistency; cap to TOP_N_P1
    p1_candidates.sort(key=lambda r: r[1], reverse=True)
    p1 = p1_candidates[:TOP_N_P1]
    used.update({r[0] for r in p1})

    # --- P2 (exclude pinned + used): F ≥ $2M ---
    p2_candidates = []
    for base, f in best_fut.items():
        if base in used or base in PINNED_SET:
            continue
        fut_usd = usd_notional(f)
        if fut_usd >= P2_FUT_MIN:
            s = best_spot.get(base)
            pct24 = pct_change(s, f)
            pct4h = compute_pct4h_for_symbol(f.symbol, True)
            p2_candidates.append([base, fut_usd, usd_notional(s) if s else 0.0, pct24, pct4h])
    p2_candidates.sort(key=lambda r: r[1], reverse=True)
    p2 = p2_candidates[:TOP_N_P2]
    used.update({r[0] for r in p2})

    # --- P3: pinned first (always included if data exists), then others Spot ≥ $3M (not used) ---
    p3_dict: Dict[str, List] = {}

    # Pinned
    for base in PINNED_P3:
        s = best_spot.get(base)
        f = best_fut.get(base)
        if not s and not f:
            continue
        fut_usd = usd_notional(f) if f else 0.0
        spot_usd = usd_notional(s) if s else 0.0
        pct24 = pct_change(s, f)
        pct4h = compute_pct4h_for_symbol(f.symbol, True) if f else (compute_pct4h_for_symbol(s.symbol, False) if s else 0.0)
        p3_dict[base] = [base, fut_usd, spot_usd, pct24, pct4h]

    # Others
    for base, s in best_spot.items():
        if base in used or base in PINNED_SET:
            continue
        spot_usd = usd_notional(s)
        if spot_usd >= P3_SPOT_MIN:
            f = best_fut.get(base)
            pct24 = pct_change(s, f)
            pct4h = compute_pct4h_for_symbol(f.symbol, True) if f else compute_pct4h_for_symbol(s.symbol, False)
            p3_dict[base] = [base, usd_notional(f) if f else 0.0, spot_usd, pct24, pct4h]

    rows = list(p3_dict.values())
    pinned_rows = [r for r in rows if r[0] in PINNED_SET]
    other_rows  = [r for r in rows if r[0] not in PINNED_SET]
    pinned_rows.sort(key=lambda r: r[2], reverse=True)  # by Spot desc
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

# ---- Google Sheets append ----
def sheet_append_rows(flat_rows: List[List[str]]) -> None:
    if not ENABLE_SHEET_APPEND:
        return
    try:
        sa_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(GSHEET_ID)
        ws = sh.worksheet(GSHEET_TAB)
        ws.append_rows(flat_rows, value_input_option="USER_ENTERED")
    except Exception:
        logging.exception("Google Sheets append failed")

def build_journal_rows(now_iso: str, p1, p2, p3) -> List[List[str]]:
    # columns: timestamp, priority, symbol, F_usd, S_usd, pct_24h, pct_4h
    out: List[List[str]] = []
    for sym, fut_usd, spot_usd, pct, pct4h in p1:
        out.append([now_iso, "P1", sym, str(int(round(fut_usd))), str(int(round(spot_usd))), str(round(pct)), str(round(pct4h))])
    for sym, fut_usd, spot_usd, pct, pct4h in p2:
        out.append([now_iso, "P2", sym, str(int(round(fut_usd))), str(int(round(spot_usd))), str(round(pct)), str(round(pct4h))])
    for sym, fut_usd, spot_usd, pct, pct4h in p3:
        out.append([now_iso, "P3", sym, str(int(round(fut_usd))), str(int(round(spot_usd))), str(round(pct)), str(round(pct4h))])
    return out

# ---- Telegram handlers ----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Commands:\n"
        "• /screen → P1(10), P2(5), P3(10) | Columns: SYM | F | S | % | %4H\n"
        "• /excel  → Excel export (.xlsx)\n"
        "• /diag   → diagnostics\n"
        "Tip: Send a ticker (e.g., PYTH) to get a one-row table."
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
            fmt_table(p1, f"Priority 1 — Top {TOP_N_P1}  (A: F≥$5M & S≥$500k  OR  B: max(F,S)≥$500k & %4H≥+10%)") +
            fmt_table(p2, f"Priority 2 — Top {TOP_N_P2}  (F≥$2M)") +
            fmt_table(p3, f"Priority 3 — Top {TOP_N_P3}  (Pinned + S≥$3M)") +
            f"⏱️ {dt:.1f}s • CoinEx via CCXT • tickers: spot={raw_spot_count}, fut={raw_fut_count}"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

        # Append to Google Sheet (journal)
        now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        rows = build_journal_rows(now_iso, p1, p2, p3)
        await asyncio.to_thread(sheet_append_rows, rows)

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
            f"- P1 rules: (A) F≥${P1_FUT_MIN:,} & S≥${P1_SPOT_MIN:,}  OR  (B) max(F,S)≥${P1_ALT_MIN:,} & %4H≥{int(P1_ALT_PCT4H_MIN)}\n"
            f"- P2 rule: F≥${P2_FUT_MIN:,}\n"
            f"- P3 rule: pinned + S≥${P3_SPOT_MIN:,}\n"
            f"- rows: P1={TOP_N_P1}, P2={TOP_N_P2}, P3={TOP_N_P3}\n"
            f"- pinned excluded from P1/P2: True\n"
            f"- tickers fetched: spot={raw_spot_count}, fut={raw_fut_count}\n"
            f"- last_error: {LAST_ERROR or '_None_'}\n"
            f"- sheet_append: {'ON' if ENABLE_SHEET_APPEND else 'OFF'}"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"Diag error: {type(e).__name__}: {e}")

# --- Symbol lookup (free text and /coin) ---
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
