#!/usr/bin/env python3
"""
Telegram crypto screener & alert bot (CoinEx via CCXT)

Features:
- Shows 3 priorities of coins (P1, P2, P3) using CoinEx volumes
- Table: SYM | F | S | %24H | %4H | %1H
- Email alerts when there are strong BUY/SELL signals (based on 24h, 4h, 1h momentum)
- Email alerts only 12:30–01:00 (Australia/Melbourne), no Sundays
- Max 1 email per 15 minutes (no daily limit)
- Commands: /start /screen /notify_on /notify_off /notify /diag
- Typing a symbol (e.g. PYTH) gives its row
"""

import asyncio
import logging
import os
import time
import re
import ssl
import smtplib
from email.message import EmailMessage
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

import ccxt
from tabulate import tabulate
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import Conflict
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ================== CONFIG ==================

EXCHANGE_ID = "coinex"
TOKEN = os.environ.get("TELEGRAM_TOKEN")

# Priority thresholds (USD notional)
P1_SPOT_MIN = 500_000
P1_FUT_MIN = 5_000_000
P2_FUT_MIN = 2_000_000
P3_SPOT_MIN = 3_000_000

TOP_N_P1 = 10
TOP_N_P2 = 15
TOP_N_P3 = 10

STABLES = {"USD", "USDT", "USDC", "TUSD", "FDUSD", "USDD", "USDE", "DAI", "PYUSD"}

# Pinned coins (only appear in P3)
PINNED_P3 = ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "PEPE", "LINK"]
PINNED_SET = set(PINNED_P3)

# Email config (set in Render env)
EMAIL_ENABLED_DEFAULT = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST = os.environ.get("EMAIL_HOST", "")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "465"))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", EMAIL_USER)
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_USER)

# Scheduler / alerts
CHECK_INTERVAL_MIN = 5  # run every 5 minutes

# One email every 15 minutes (no daily limit)
EMAIL_MIN_INTERVAL_SEC = 15 * 60
LAST_EMAIL_TS: float = 0.0

# Globals
LAST_ERROR: Optional[str] = None
NOTIFY_ON: bool = EMAIL_ENABLED_DEFAULT

PCT4H_CACHE: Dict[Tuple[str, str], float] = {}
PCT1H_CACHE: Dict[Tuple[str, str], float] = {}

# ================== DATA STRUCTURES ==================

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


# ================== HELPERS ==================

def safe_split_symbol(sym: Optional[str]):
    if not sym:
        return None
    pair = sym.split(":")[0]
    if "/" not in pair:
        return None
    return tuple(pair.split("/", 1))


def to_mv(t: dict) -> Optional[MarketVol]:
    sym = t.get("symbol")
    sp = safe_split_symbol(sym)
    if not sp:
        return None
    base, quote = sp
    return MarketVol(
        symbol=sym,
        base=base,
        quote=quote,
        last=float(t.get("last") or 0.0),
        open=float(t.get("open") or 0.0),
        percentage=float(t.get("percentage") or 0.0),
        base_vol=float(t.get("baseVolume") or 0.0),
        quote_vol=float(t.get("quoteVolume") or 0.0),
        vwap=float(t.get("vwap") or 0.0),
    )


def usd_notional(mv: Optional[MarketVol]) -> float:
    """Return 24h notional volume in USD."""
    if not mv:
        return 0.0
    if mv.quote in STABLES and mv.quote_vol:
        return mv.quote_vol
    price = mv.vwap if mv.vwap else mv.last
    if not price or not mv.base_vol:
        return 0.0
    return mv.base_vol * price


def pct_change_24h(mv_spot: Optional[MarketVol], mv_fut: Optional[MarketVol]) -> float:
    """24h % change (prefer ticker.percentage)."""
    for mv in (mv_spot, mv_fut):
        if mv and mv.percentage:
            return float(mv.percentage)
    mv = mv_spot or mv_fut
    if mv and mv.open:
        return (mv.last - mv.open) / mv.open * 100.0
    return 0.0


def pct_with_emoji(p: float) -> str:
    val = round(p)
    if val >= 3:
        emo = "🟢"
    elif val <= -3:
        emo = "🔴"
    else:
        emo = "🟡"
    return f"{val:+d}% {emo}"


def m_dollars(x: float) -> str:
    """Return millions as a rounded integer string."""
    return str(round(x / 1_000_000))


def build_exchange(default_type: str):
    klass = ccxt.__dict__[EXCHANGE_ID]
    return klass(
        {
            "enableRateLimit": True,
            "timeout": 20000,
            "options": {"defaultType": default_type},
        }
    )


def safe_fetch_tickers(ex):
    """Fetch tickers; on error, log and return {}."""
    try:
        ex.load_markets()
        return ex.fetch_tickers()
    except Exception as e:
        global LAST_ERROR
        LAST_ERROR = f"{type(e).__name__}: {e}"
        logging.exception("fetch_tickers failed")
        return {}


# ================== PERCENTAGE (4H / 1H) ==================

def compute_pct_for_symbol(symbol: str, hours: int, prefer_swap: bool = True) -> float:
    """
    Compute % change over the last N completed hours using 1h candles.
    Cache results per (dtype, symbol).
    """
    try_order = ["swap", "spot"] if prefer_swap else ["spot", "swap"]
    cache = PCT4H_CACHE if hours == 4 else PCT1H_CACHE
    for dtype in try_order:
        cache_key = (dtype, symbol)
        if cache_key in cache:
            return cache[cache_key]
        try:
            ex = build_exchange(dtype)
            ex.load_markets()
            candles = ex.fetch_ohlcv(symbol, timeframe="1h", limit=hours + 1)
            if not candles or len(candles) <= hours:
                continue
            closes = [c[4] for c in candles][- (hours + 1):]
            if not closes or not closes[0]:
                pct = 0.0
            else:
                pct = (closes[-1] - closes[0]) / closes[0] * 100.0
            cache[cache_key] = pct
            return pct
        except Exception:
            logging.exception("compute_pct_for_symbol: %dh failed for %s (%s)", hours, symbol, dtype)
            continue
    return 0.0


# ================== PRIORITIES ==================

def load_best():
    """Load top spot & futures markets per BASE symbol from CoinEx."""
    ex_spot = build_exchange("spot")
    ex_fut = build_exchange("swap")
    spot_tickers = safe_fetch_tickers(ex_spot)
    fut_tickers = safe_fetch_tickers(ex_fut)

    best_spot: Dict[str, MarketVol] = {}
    best_fut: Dict[str, MarketVol] = {}

    for t in spot_tickers.values():
        mv = to_mv(t)
        if mv and mv.quote in STABLES:
            if mv.base not in best_spot or usd_notional(mv) > usd_notional(best_spot[mv.base]):
                best_spot[mv.base] = mv

    for t in fut_tickers.values():
        mv = to_mv(t)
        if mv:
            if mv.base not in best_fut or usd_notional(mv) > usd_notional(best_fut[mv.base]):
                best_fut[mv.base] = mv

    return best_spot, best_fut, len(spot_tickers), len(fut_tickers)


def build_priorities(best_spot: Dict[str, MarketVol], best_fut: Dict[str, MarketVol]):
    """
    Build P1, P2, P3 lists.
    Each row: [SYM, FUSD, SUSD, %24H, %4H, %1H, LASTPRICE]
      - LASTPRICE is internal, not shown in table.
    """
    p1: List[List] = []
    p2: List[List] = []
    p3: List[List] = []
    used = set()

    # ---- P1: F≥5M & S≥500k (pinned excluded) ----
    for base in set(best_spot) & set(best_fut):
        if base in PINNED_SET:
            continue
        s = best_spot[base]
        f = best_fut[base]
        fut_usd = usd_notional(f)
        spot_usd = usd_notional(s)
        if fut_usd >= P1_FUT_MIN and spot_usd >= P1_SPOT_MIN:
            pct24 = pct_change_24h(s, f)
            pct4 = compute_pct_for_symbol(f.symbol, 4)
            pct1 = compute_pct_for_symbol(f.symbol, 1)
            last_price = f.last or s.last or 0.0
            p1.append([base, fut_usd, spot_usd, pct24, pct4, pct1, last_price])

    p1.sort(key=lambda x: x[1], reverse=True)
    p1 = p1[:TOP_N_P1]
    used |= {r[0] for r in p1}

    # ---- P2: F≥2M (pinned & already used excluded) ----
    for base, f in best_fut.items():
        if base in used or base in PINNED_SET:
            continue
        fut_usd = usd_notional(f)
        if fut_usd >= P2_FUT_MIN:
            s = best_spot.get(base)
            spot_usd = usd_notional(s) if s else 0.0
            pct24 = pct_change_24h(s, f)
            pct4 = compute_pct_for_symbol(f.symbol, 4)
            pct1 = compute_pct_for_symbol(f.symbol, 1)
            last_price = f.last or (s.last if s else 0.0)
            p2.append([base, fut_usd, spot_usd, pct24, pct4, pct1, last_price])

    p2.sort(key=lambda x: x[1], reverse=True)
    p2 = p2[:TOP_N_P2]
    used |= {r[0] for r in p2}

    # ---- P3: pinned + others with Spot≥3M (pinned only in P3) ----
    tmp: Dict[str, List] = {}

    # pinned first
    for base in PINNED_P3:
        s = best_spot.get(base)
        f = best_fut.get(base)
        if not s and not f:
            continue
        fut_usd = usd_notional(f) if f else 0.0
        spot_usd = usd_notional(s) if s else 0.0
        pct24 = pct_change_24h(s, f)
        symbol_for_pct = f.symbol if f else (s.symbol if s else "")
        pct4 = compute_pct_for_symbol(symbol_for_pct, 4) if symbol_for_pct else 0.0
        pct1 = compute_pct_for_symbol(symbol_for_pct, 1) if symbol_for_pct else 0.0
        last_price = (f.last if f else (s.last if s else 0.0))
        tmp[base] = [base, fut_usd, spot_usd, pct24, pct4, pct1, last_price]

    # non-pinned, not used, Spot≥3M
    for base, s in best_spot.items():
        if base in used or base in PINNED_SET:
            continue
        spot_usd = usd_notional(s)
        if spot_usd >= P3_SPOT_MIN:
            f = best_fut.get(base)
            fut_usd = usd_notional(f) if f else 0.0
            pct24 = pct_change_24h(s, f)
            symbol_for_pct = f.symbol if f else s.symbol
            pct4 = compute_pct_for_symbol(symbol_for_pct, 4)
            pct1 = compute_pct_for_symbol(symbol_for_pct, 1)
            last_price = (f.last if f else s.last)
            tmp[base] = [base, fut_usd, spot_usd, pct24, pct4, pct1, last_price]

    all_rows = list(tmp.values())
    pinned_rows = [r for r in all_rows if r[0] in PINNED_SET]
    other_rows = [r for r in all_rows if r[0] not in PINNED_SET]

    pinned_rows.sort(key=lambda r: r[2], reverse=True)
    other_rows.sort(key=lambda r: r[2], reverse=True)
    p3 = (pinned_rows + other_rows)[:TOP_N_P3]

    return p1, p2, p3


# ================== SCORING & RECOMMENDATIONS ==================

def score_long(row: List) -> float:
    """
    Long (BUY) score based on:
    - Positive 24h, 4h, 1h momentum
    - Futures liquidity
    """
    _, fut_usd, _, pct24, pct4, pct1, _ = row
    score = 0.0

    if pct24 > 0:
        score += min(pct24 / 5.0, 3.0)
    if pct4 > 0:
        score += min(pct4 / 2.0, 3.0)
    if pct1 > 0:
        score += min(pct1 / 1.0, 2.0)

    if fut_usd > 10_000_000:
        score += 2.0
    elif fut_usd > 5_000_000:
        score += 1.0

    return max(score, 0.0)


def score_short(row: List) -> float:
    """
    Short (SELL) score based on:
    - Negative 24h, 4h, 1h momentum
    - Futures liquidity
    """
    _, fut_usd, _, pct24, pct4, pct1, _ = row
    score = 0.0

    if pct24 < 0:
        score += min(abs(pct24) / 5.0, 3.0)
    if pct4 < 0:
        score += min(abs(pct4) / 2.0, 3.0)
    if pct1 < 0:
        score += min(abs(pct1) / 1.0, 2.0)

    if fut_usd > 10_000_000:
        score += 2.0
    elif fut_usd > 5_000_000:
        score += 1.0

    return max(score, 0.0)


def classify_row(row: List) -> Tuple[bool, bool]:
    """
    Decide if a row is a LONG or SHORT candidate based on 24h, 4h, 1h.

    - Basic thresholds:
        LONG: 1h >= +2 and 4h >= 0
        SHORT: 1h <= -2 and 4h <= 0
    - 24h bias:
        If 24h > +5 → only LONG allowed
        If 24h < -5 → only SHORT allowed
    - If both would be allowed → tie-break using 24h then 1h.
    """
    _, _, _, pct24, pct4, pct1, _ = row
    long_ok = False
    short_ok = False

    # 24h bias override
    if pct24 > 5.0:
        long_ok = True
        short_ok = False
    elif pct24 < -5.0:
        short_ok = True
        long_ok = False
    else:
        # Normal 1h / 4h rules
        if pct1 >= 2.0 and pct4 >= 0.0:
            long_ok = True
        if pct1 <= -2.0 and pct4 <= 0.0:
            short_ok = True

    # If still both, resolve conflict
    if long_ok and short_ok:
        if pct24 > 0:
            short_ok = False
        elif pct24 < 0:
            long_ok = False
        else:
            if pct1 >= 0:
                short_ok = False
            else:
                long_ok = False

    return long_ok, short_ok


def pick_best_trades(p1: List[List], p2: List[List], p3: List[List]):
    """
    Return TOP 2 trades overall (BUY or SELL).
    Uses classify_row() to decide LONG/SHORT candidates and score_long/score_short for ranking.

    Returns list of up to 2 items:
        [(side, sym, entry, exit, sl, score), ...]
    """

    rows = p1 + p2 + p3
    candidates: List[Tuple[str, str, float, float, float, float]] = []

    for r in rows:
        sym, _, _, pct24, pct4, pct1, last_price = r
        if not last_price or last_price <= 0:
            continue

        long_ok, short_ok = classify_row(r)

        if long_ok:
            score = score_long(r)
            if score > 0:
                entry = last_price
                sl = entry * 0.94   # 6% SL
                tp = entry * 1.12   # 12% TP
                candidates.append(("BUY", sym, entry, tp, sl, score))

        if short_ok:
            score = score_short(r)
            if score > 0:
                entry = last_price
                sl = entry * 1.06   # 6% SL
                tp = entry * 0.91   # 9% TP
                candidates.append(("SELL", sym, entry, tp, sl, score))

    candidates.sort(key=lambda x: x[5], reverse=True)
    return candidates[:2]  # top 2 recommendations


def format_recommended_trades(recs: List[Tuple]) -> str:
    if not recs:
        return "_No strong recommendations right now._"
    lines = ["*Top Recommendations:*"]
    for side, sym, entry, exit_px, sl, score in recs:
        lines.append(
            f"{side} {sym} — Entry {entry:.6g} — Exit {exit_px:.6g} — SL {sl:.6g} (score {score:.1f})"
        )
    return "\n".join(lines)


# ================== FORMATTING ==================

def fmt_table(rows: List[List], title: str) -> str:
    if not rows:
        return f"*{title}*: _None_\n"
    pretty = [
        [
            r[0],
            m_dollars(r[1]),
            m_dollars(r[2]),
            pct_with_emoji(r[3]),
            pct_with_emoji(r[4]),
            pct_with_emoji(r[5]),
        ]
        for r in rows
    ]
    return (
        f"*{title}*:\n"
        "```\n"
        + tabulate(pretty, headers=["SYM", "F", "S", "%24H", "%4H", "%1H"], tablefmt="github")
        + "\n```\n"
    )


def fmt_single(sym: str, fusd: float, susd: float, p24: float, p4: float, p1: float) -> str:
    row = [[sym, m_dollars(fusd), m_dollars(susd),
            pct_with_emoji(p24), pct_with_emoji(p4), pct_with_emoji(p1)]]
    return (
        "```\n"
        + tabulate(row, headers=["SYM", "F", "S", "%24H", "%4H", "%1H"], tablefmt="github")
        + "\n```"
    )


# ================== EMAIL HELPERS ==================

def email_config_ok() -> bool:
    return all([EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FROM, EMAIL_TO])


def send_email(subject: str, body: str) -> bool:
    if not email_config_ok():
        logging.warning("Email not configured.")
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        msg.set_content(body)

        if EMAIL_PORT == 465:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=ctx) as s:
                s.login(EMAIL_USER, EMAIL_PASS)
                s.send_message(msg)
        else:
            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as s:
                s.starttls()
                s.login(EMAIL_USER, EMAIL_PASS)
                s.send_message(msg)
        return True
    except Exception as e:
        logging.exception("send_email failed: %s", e)
        return False


def melbourne_ok() -> bool:
    """
    Return True only if local time is between 12:30–01:00 in Melbourne,
    and it is NOT Sunday (local Melbourne date).
    """
    try:
        now_mel = datetime.now(ZoneInfo("Australia/Melbourne"))
        dow = now_mel.weekday()  # Monday=0 ... Sunday=6
        if dow == 6:  # Sunday
            return False

        h = now_mel.hour
        m = now_mel.minute

        after_1230 = (h > 12) or (h == 12 and m >= 30)
        before_1am = h < 1

        return after_1230 or before_1am
    except Exception:
        logging.exception("Melbourne time failed; defaulting to allowed.")
        return True


# ================== TELEGRAM HANDLERS ==================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n"
        "• /screen — show P1, P2, P3 + recommendations\n"
        "• /notify_on /notify_off /notify\n"
        "• /diag — short diagnostics\n"
        "• Type a symbol (e.g. PYTH) for its row"
    )


async def screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        PCT4H_CACHE.clear()
        PCT1H_CACHE.clear()
        best_spot, best_fut, raw_spot, raw_fut = await asyncio.to_thread(load_best)
        p1, p2, p3 = await asyncio.to_thread(build_priorities, best_spot, best_fut)

        recs = pick_best_trades(p1, p2, p3)
        rec_text = format_recommended_trades(recs)

        msg = (
            fmt_table(p1, "P1 (F≥5M & S≥0.5M — pinned excluded)")
            + fmt_table(p2, "P2 (F≥2M — pinned excluded)")
            + fmt_table(p3, "P3 (Pinned + S≥3M)")
            + f"tickers: spot={raw_spot}, fut={raw_fut}\n\n"
            + "*Recommended trades:*\n"
            + rec_text
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logging.exception("screen error")
        await update.message.reply_text(f"Error: {e}")


async def diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"*Diag*\n"
        f"- time window: 12:30–01:00 Melbourne (no Sundays)\n"
        f"- interval: {CHECK_INTERVAL_MIN} min\n"
        f"- email: {'ON' if NOTIFY_ON else 'OFF'} → {EMAIL_TO or 'n/a'}"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def notify_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global NOTIFY_ON
    NOTIFY_ON = True
    await update.message.reply_text("Email alerts: ON")


async def notify_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global NOTIFY_ON
    NOTIFY_ON = False
    await update.message.reply_text("Email alerts: OFF")


async def notify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Email alerts: {'ON' if NOTIFY_ON else 'OFF'} → {EMAIL_TO or 'n/a'}"
    )


async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    token = re.sub(r"[^A-Za-z$]", "", text).upper().lstrip("$")
    if len(token) < 2:
        return
    try:
        best_spot, best_fut, _, _ = await asyncio.to_thread(load_best)
        s = best_spot.get(token)
        f = best_fut.get(token)
        fusd = usd_notional(f) if f else 0.0
        susd = usd_notional(s) if s else 0.0
        if fusd == 0.0 and susd == 0.0:
            await update.message.reply_text("Symbol not found.")
            return
        pct24 = pct_change_24h(s, f)
        symbol_for_pct = f.symbol if f else (s.symbol if s else "")
        pct4 = compute_pct_for_symbol(symbol_for_pct, 4) if symbol_for_pct else 0.0
        pct1 = compute_pct_for_symbol(symbol_for_pct, 1) if symbol_for_pct else 0.0
        msg = fmt_single(token, fusd, susd, pct24, pct4, pct1)
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logging.exception("text_router error")
        await update.message.reply_text(f"Error: {e}")


# ================== ALERT JOB ==================

async def alert_job(context: ContextTypes.DEFAULT_TYPE):
    global LAST_EMAIL_TS
    try:
        if not NOTIFY_ON:
            return
        if not melbourne_ok():
            return
        if not email_config_ok():
            return

        now = time.time()
        if now - LAST_EMAIL_TS < EMAIL_MIN_INTERVAL_SEC:
            return

        PCT4H_CACHE.clear()
        PCT1H_CACHE.clear()

        best_spot, best_fut, _, _ = await asyncio.to_thread(load_best)
        p1, p2, p3 = await asyncio.to_thread(build_priorities, best_spot, best_fut)
        recs = pick_best_trades(p1, p2, p3)

        if not recs:
            return

        rec_text = format_recommended_trades(recs)
        body = "Recommended trades:\n\n" + rec_text

        if send_email("Crypto Alert: Momentum Signals", body):
            LAST_EMAIL_TS = now
    except Exception as e:
        logging.exception("alert_job error: %s", e)


# ================== ERROR HANDLER ==================

async def log_err(update: object, context: ContextTypes.DEFAULT_TYPE):
    try:
        raise context.error
    except Conflict:
        logging.warning("Conflict: another instance already polling.")
    except Exception as e:
        logging.exception("Unhandled error: %s", e)


# ================== MAIN ==================

def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN missing")
    logging.basicConfig(level=logging.INFO)

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("screen", screen))
    app.add_handler(CommandHandler("diag", diag))
    app.add_handler(CommandHandler("notify_on", notify_on))
    app.add_handler(CommandHandler("notify_off", notify_off))
    app.add_handler(CommandHandler("notify", notify))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    app.add_error_handler(log_err)

    if getattr(app, "job_queue", None):
        app.job_queue.run_repeating(
            alert_job,
            interval=CHECK_INTERVAL_MIN * 60,
            first=10,
        )
    else:
        logging.warning(
            "JobQueue not available. Make sure you installed "
            '"python-telegram-bot[job-queue]" in requirements.txt.'
        )

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
