#!/usr/bin/env python3
"""
Telegram crypto screener & alert bot (CoinEx via CCXT)

Features:
- Shows 3 priorities of coins (P1, P2, P3) using CoinEx volumes
- Table: SYM | F | S | %24H | %4H | %1H
- Email alerts if (4h >= +5%) AND (1h >= +5%)
- Emails only 07:00–23:00 (Australia/Melbourne)
- Max 4 emails/hour, 20 emails/day, 15-min cooldown per (priority,symbol)
- Check interval = 5 min
- Commands: /start /screen /notify_on /notify_off /notify /diag
- Typing a symbol (e.g. PYTH) gives a one-row table

This version also:
- Calculates simple LONG and SHORT scores based on %24h, %4h, %1h, and futures volume
- Picks 1 best BUY and 1 best SELL from all P1/P2/P3 rows
- Generates Entry / Exit / Stop Loss for both, shown at bottom of /screen and in email alerts
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
ALERT_PCT_4H_MIN = 5.0
ALERT_PCT_1H_MIN = 5.0
ALERT_THROTTLE_SEC = 15 * 60  # 15 minutes per (priority,symbol)

EMAIL_DAILY_LIMIT = 20
EMAIL_HOURLY_LIMIT = 4
EMAIL_DAILY_WINDOW_SEC = 24 * 60 * 60
EMAIL_HOURLY_WINDOW_SEC = 60 * 60

# Globals
LAST_ERROR: Optional[str] = None
NOTIFY_ON: bool = EMAIL_ENABLED_DEFAULT

PCT4H_CACHE: Dict[Tuple[str, str], float] = {}
PCT1H_CACHE: Dict[Tuple[str, str], float] = {}

ALERT_SENT_CACHE: Dict[Tuple[str, str], float] = {}  # (priority, symbol) -> last_sent_time
EMAIL_SEND_LOG: List[float] = []  # timestamps of sent emails


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

    # Momentum
    if pct24 > 0:
        score += min(pct24 / 5.0, 3.0)
    if pct4 > 0:
        score += min(pct4 / 2.0, 3.0)
    if pct1 > 0:
        score += min(pct1 / 1.0, 2.0)

    # Volume bonus
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

    # Downside momentum
    if pct24 < 0:
        score += min(abs(pct24) / 5.0, 3.0)
    if pct4 < 0:
        score += min(abs(pct4) / 2.0, 3.0)
    if pct1 < 0:
        score += min(abs(pct1) / 1.0, 2.0)

    # Volume bonus
    if fut_usd > 10_000_000:
        score += 2.0
    elif fut_usd > 5_000_000:
        score += 1.0

    return max(score, 0.0)

def pick_best_trades(p1: List[List], p2: List[List], p3: List[List]):
    """
    Instead of 1 BUY + 1 SELL, return TOP 4 trades overall (BUY or SELL).
    Scoring:
      - long score → BUY
      - short score → SELL

    Returns a list of 4 items:
        [(side, sym, entry, exit, sl, score), ...]
    """

    rows = p1 + p2 + p3
    scored = []

    for r in rows:
        sym, _, _, _, _, _, last_price = r
        if not last_price or last_price <= 0:
            continue

        long_s = score_long(r)
        short_s = score_short(r)

        # BUY candidate
        if long_s > 0:
            score = long_s
            entry = last_price
            sl = entry * 0.96          # 4% SL
            exit = entry * 1.08         # 8% TP
            scored.append(("BUY", sym, entry, exit, sl, score))

        # SELL candidate
        if short_s > 0:
            score = short_s
            entry = last_price
            sl = entry * 1.04          # 4% SL
            exit = entry * 0.94         # 6% TP
            scored.append(("SELL", sym, entry, exit, sl, score))

    # Sort by score descending, pick top 4
    scored.sort(key=lambda x: x[5], reverse=True)
    return scored[:4]    # top 4 recommendations


def format_recommended_trades(recs: List[Tuple]) -> str:
    if not recs:
        return "_No strong recommendations right now._"

    lines = ["*Top 4 Recommendations:*"]
    for side, sym, entry, exit, sl, score in recs:
        lines.append(
            f"{side} {sym} — Entry {entry:.6g} — Exit {exit:.6g} — SL {sl:.6g} (score {score:.1f})"
        )
    return "\n".join(lines)


# ================== FORMATTING ==================

def fmt_table(rows: List[List], title: str) -> str:
    if not rows:
        return f"*{title}*: _None_\n"
    # Only show first 6 columns (SYM, F, S, %24H, %4H, %1H), skip LASTPRICE
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
    """Return True only if local time is between 11:00–23:00 in Melbourne."""
    try:
        hr = datetime.now(ZoneInfo("Australia/Melbourne")).hour
        return 11 <= hr < 23
    except Exception:
        # If timezone fails, default to allow
        logging.exception("Melbourne time failed; defaulting to allowed.")
        return True


def email_quota_ok() -> bool:
    """Check daily + hourly global email caps."""
    now = time.time()
    recent = [x for x in EMAIL_SEND_LOG if now - x < EMAIL_DAILY_WINDOW_SEC]
    EMAIL_SEND_LOG[:] = recent

    if len(EMAIL_SEND_LOG) >= EMAIL_DAILY_LIMIT:
        return False

    hourly = sum(1 for x in EMAIL_SEND_LOG if now - x < EMAIL_HOURLY_WINDOW_SEC)
    if hourly >= EMAIL_HOURLY_LIMIT:
        return False

    return True


def record_email():
    EMAIL_SEND_LOG.append(time.time())


def should_alert(priority: str, symbol: str) -> bool:
    """Per (priority,symbol) cool-down."""
    now = time.time()
    key = (priority, symbol)
    last = ALERT_SENT_CACHE.get(key, 0)
    if now - last >= ALERT_THROTTLE_SEC:
        ALERT_SENT_CACHE[key] = now
        return True
    return False


def scan_for_alerts(p1: List[List], p2: List[List], p3: List[List], rec_text: str) -> Optional[str]:
    """
    Build email body if we have coins with:
      4h >= ALERT_PCT_4H_MIN AND 1h >= ALERT_PCT_1H_MIN
    Also append recommended trades at the end.
    """
    lines: List[str] = []
    for label, rows in (("P1", p1), ("P2", p2), ("P3", p3)):
        hits: List[str] = []
        for r in rows:
            sym, _, _, _, pct4, pct1, _ = r
            if pct4 >= ALERT_PCT_4H_MIN and pct1 >= ALERT_PCT_1H_MIN:
                if should_alert(label, sym):
                    hits.append(sym)
        if hits:
            lines.append(f"{label}: " + ", ".join(sorted(set(hits))))
    if not lines and not rec_text:
        return None
    body_parts = []
    if lines:
        body_parts.append("Coins +5% (4h & 1h):\n\n" + "\n".join(lines))
    if rec_text:
        if lines:
            body_parts.append("\n\nRecommended trades:\n" + rec_text)
        else:
            body_parts.append("Recommended trades:\n" + rec_text)
    return "".join(body_parts) if body_parts else None


# ================== TELEGRAM HANDLERS ==================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n"
        "• /screen — show P1, P2, P3 + recommended trades\n"
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
        f"- rule: +5% (4h & 1h)\n"
        f"- interval: {CHECK_INTERVAL_MIN} min\n"
        f"- email: {'ON' if NOTIFY_ON else 'OFF'}"
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
    try:
        if not NOTIFY_ON:
            return
        if not melbourne_ok():
            return
        if not email_config_ok():
            return
        if not email_quota_ok():
            return

        PCT4H_CACHE.clear()
        PCT1H_CACHE.clear()

        best_spot, best_fut, _, _ = await asyncio.to_thread(load_best)
        p1, p2, p3 = await asyncio.to_thread(build_priorities, best_spot, best_fut)
        buy_trade, sell_trade = pick_best_trades(p1, p2, p3)
        rec_text = format_recommended_trades(buy_trade, sell_trade)

        body = scan_for_alerts(p1, p2, p3, rec_text)
        if not body:
            return

        if send_email("Crypto Alert: +5% (4h) & +5% (1h)", body):
            record_email()
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

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("screen", screen))
    app.add_handler(CommandHandler("diag", diag))
    app.add_handler(CommandHandler("notify_on", notify_on))
    app.add_handler(CommandHandler("notify_off", notify_off))
    app.add_handler(CommandHandler("notify", notify))

    # Text symbol lookup
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    # Error handler
    app.add_error_handler(log_err)

    # Job queue for alerts
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
