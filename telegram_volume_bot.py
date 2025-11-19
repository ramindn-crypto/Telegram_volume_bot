#!/usr/bin/env python3
"""
Telegram crypto screener & alert bot (Final Version)

KEY FEATURES:
- P1 / P2 / P3 (CoinEx)
- %24h, %4h, %1h change
- Futures & Spot USD volume
- Binance Futures OI + Funding (Coinalyze-style scoring)
- Top 4 BUY/SELL mixed recommendations
- Entry, Exit (TP), Stop-Loss generated for all 4 trades
- Email alerts when 4h>=5% & 1h>=5%
- Email alerts ONLY sent if recommended trades (symbol+side) DIFFER
  from last email
- Time window: 09:00 → 21:00 Melbourne only
- P2 list expanded to 10 rows
- 2 emails per hour, 20 per day
- Symbol query via text (e.g., "PYTH")
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

COINEX_ID = "coinex"
BINANCE_ID = "binance"
TOKEN = os.environ.get("TELEGRAM_TOKEN")

# Priority thresholds (USD notional)
P1_SPOT_MIN = 500_000
P1_FUT_MIN = 5_000_000
P2_FUT_MIN = 2_000_000
P3_SPOT_MIN = 3_000_000

TOP_N_P1 = 10
TOP_N_P2 = 10   # ← updated from 5 to 10
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
CHECK_INTERVAL_MIN = 5
ALERT_PCT_4H_MIN = 5.0
ALERT_PCT_1H_MIN = 5.0
ALERT_THROTTLE_SEC = 15 * 60  # (only for priority matches, not recommendations)

EMAIL_DAILY_LIMIT = 20
EMAIL_HOURLY_LIMIT = 2
EMAIL_DAILY_WINDOW_SEC = 24 * 60 * 60
EMAIL_HOURLY_WINDOW_SEC = 60 * 60

# Globals
LAST_ERROR: Optional[str] = None
NOTIFY_ON: bool = EMAIL_ENABLED_DEFAULT

PCT4H_CACHE: Dict[Tuple[str, str], float] = {}
PCT1H_CACHE: Dict[Tuple[str, str], float] = {}

ALERT_SENT_CACHE: Dict[Tuple[str, str], float] = {}
EMAIL_SEND_LOG: List[float] = []

# NEW: store last 4 recommended signals (symbol + BUY/SELL)
LAST_REC_SET: Optional[List[Tuple[str, str]]] = None  # List of (SYM, SIDE)
# =========================
#    UTILITIES
# =========================

def now_ts() -> float:
    return time.time()


def melbourne_time() -> datetime:
    return datetime.now(ZoneInfo("Australia/Melbourne"))


def melbourne_ok() -> bool:
    """Only send alerts between 09:00 and 21:00 Melbourne."""
    hr = melbourne_time().hour
    return 9 <= hr < 21


def round_int(x: float) -> int:
    try:
        return int(round(x))
    except:
        return 0


# =========================
#      EXCHANGES
# =========================

coinex = ccxt.coinex({"enableRateLimit": True})
binance = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
binance_spot = ccxt.binance({"enableRateLimit": True})

# =========================
#  OHLCV HELPERS (%1h / %4h)
# =========================

async def fetch_coinex_tickers():
    """
    Fetch CoinEx spot & futures tickers in a safe way:

    - load_markets() to get metadata (spot / swap)
    - fetch_tickers() once (all markets)
    - split into spot/futures based on market['spot'] / market['swap']
    """
    try:
        # markets: e.g. {"BTC/USDT": {"spot": True}, "BTC/USDT:USDT": {"swap": True}, ...}
        markets = coinex.load_markets()
        all_tickers = coinex.fetch_tickers()
    except Exception as e:
        logging.warning(f"CoinEx fetch_markets/fetch_tickers failed: {type(e).__name__}: {e}")
        return {}, {}

    spot = {}
    fut = {}

    for sym, tk in all_tickers.items():
        m = markets.get(sym)
        if not m:
            continue

        # ccxt marks them like: m['spot'] == True or m['swap'] == True
        if m.get("swap"):
            fut[sym] = tk
        elif m.get("spot"):
            spot[sym] = tk

    return spot, fut



# =========================
#      BINANCE DATA
#   (Funding, OI etc.)
# =========================

async def fetch_binance_funding(symbol: str) -> float:
    """Fetch funding rate on Binance futures for BASE/USDT."""
    try:
        mk = f"{symbol}/USDT"
        fr = binance.fapiPublic_get_premiumindex({"symbol": symbol + "USDT"})
        return float(fr["lastFundingRate"]) * 100
    except:
        return 0.0


async def fetch_binance_oi(symbol: str) -> float:
    """
    Fetch open interest in USD via binance.fapiPublic_get_openinterest.
    Returns raw USD OI.
    """
    try:
        resp = binance.fapiPublic_get_openinterest({"symbol": symbol + "USDT"})
        return float(resp["openInterest"])  # contracts, often equal to USD
    except:
        return 0.0


async def fetch_binance_oi_change_24h(symbol: str) -> float:
    """Approximate 24h OI change by comparing last candle's OI vs earlier."""
    try:
        # Use open interest history if available:
        oi_now = await fetch_binance_oi(symbol)
        candles = binance.fetch_ohlcv(symbol + "/USDT", "1h", limit=25)
        # VERY rough OI proxy: use price levels
        if candles:
            old_price = candles[0][1]
            new_price = candles[-1][1]
            return ((new_price - old_price) / old_price) * 100
        return 0.0
    except:
        return 0.0

# =========================
#     COINEX DATA
#  (Spot & Futures)
# =========================

async def fetch_coinex_tickers():
    """
    Fetch all CoinEx spot & futures markets.
    If one side fails, return {} for that side but do NOT crash.
    Log as WARNING instead of ERROR.
    """
    spot = {}
    fut = {}

    # Spot tickers
    try:
        spot = coinex.fetch_tickers()
    except Exception as e:
        logging.warning(f"CoinEx spot fetch_tickers failed: {type(e).__name__}: {e}")

    # Futures (swap) tickers
    try:
        fut = coinex.fetch_tickers({"market": "swap"})
    except Exception as e:
        logging.warning(f"CoinEx futures fetch_tickers failed: {type(e).__name__}: {e}")

    return spot, fut


def parse_symbol(sym: str) -> Optional[Tuple[str, str]]:
    """
    Handle CoinEx / ccxt symbols in forms like:
    - "BTC/USDT"
    - "BTC/USDT:USDT"
    - "BTCUSDT"
    and return (BASE, QUOTE) if quote is a stable.
    """
    # Case 1: "BTC/USDT" or "BTC/USDT:USDT"
    if "/" in sym:
        base, quote = sym.split("/", 1)
        # e.g. "USDT:USDT"
        if ":" in quote:
            quote = quote.split(":", 1)[0]
        if quote in STABLES:
            return base, quote

    # Case 2: fallback like "BTCUSDT"
    for st in STABLES:
        if sym.endswith(st):
            base = sym[: -len(st)]
            if base:
                return base, st

    return None


# =========================
#   BUILD PRIORITY LISTS
# =========================

async def build_priorities():
    """
    Build P1 / P2 / P3 using BINANCE data instead of CoinEx.

    - Futures: from binance (USDT-margined futures)
    - Spot: from binance_spot
    - Same thresholds:
        P1: fut >= 5M AND spot >= 500k
        P2: fut >= 2M
        P3: pinned coins (BTC, ETH, etc) with spot >= 3M
    """

    # --- get tickers from binance ---
    try:
        spot_tk = binance_spot.fetch_tickers()
    except Exception as e:
        logging.warning(f"Binance spot fetch_tickers failed: {type(e).__name__}: {e}")
        spot_tk = {}

    try:
        fut_tk = binance.fetch_tickers()
    except Exception as e:
        logging.warning(f"Binance futures fetch_tickers failed: {type(e).__name__}: {e}")
        fut_tk = {}

    rows_p1 = []
    rows_p2 = []
    rows_p3 = []

    for sym, ft in fut_tk.items():
        parsed = parse_symbol(sym)
        if not parsed:
            continue

        base, quote = parsed
        if quote not in STABLES:
            continue

        # Spot symbol key on binance is usually "BASE/QUOTE"
        spot_sym = f"{base}/{quote}"
        spot = spot_tk.get(spot_sym)

        fut_usd = usd_notional(ft)
        spot_usd = usd_notional(spot) if spot else 0.0

        # % changes from futures ticker
        try:
            pct24 = float(ft.get("percentage", 0.0))
        except Exception:
            pct24 = 0.0

        # use binance futures candles for 4h & 1h change
        pct4 = await pct_change(binance, sym, base, quote, 4)
        pct1 = await pct_change(binance, sym, base, quote, 1)

        # P1
        if fut_usd >= P1_FUT_MIN and spot_usd >= P1_SPOT_MIN:
            rows_p1.append((base, fut_usd, spot_usd, pct24, pct4, pct1))

        # P2
        if fut_usd >= P2_FUT_MIN:
            rows_p2.append((base, fut_usd, spot_usd, pct24, pct4, pct1))

        # P3 (pinned coins — BTC, ETH, etc — with big spot volume)
        if base in PINNED_SET and spot_usd >= P3_SPOT_MIN:
            rows_p3.append((base, fut_usd, spot_usd, pct24, pct4, pct1))

    # Sort by futures notional
    rows_p1.sort(key=lambda x: x[1], reverse=True)
    rows_p2.sort(key=lambda x: x[1], reverse=True)
    rows_p3.sort(key=lambda x: x[1], reverse=True)

    return (
        rows_p1[:TOP_N_P1],
        rows_p2[:TOP_N_P2],
        rows_p3[:TOP_N_P3],
    )

# =========================
#   SCORING ENGINE
# =========================

async def compute_score(base: str, fut_usd: float, spot_usd: float,
                        pct24: float, pct4: float, pct1: float) -> Tuple[float, float]:
    """
    Returns (long_score, short_score)
    Using Coinalyze-style proxies via Binance futures:
    - Funding
    - OI level
    - OI 24h change
    - Price % changes
    - Volume weights
    """

    # Funding rate
    funding = await fetch_binance_funding(base)

    # Open interest level (raw)
    oi = await fetch_binance_oi(base)

    # OI change proxy
    oi_ch = await fetch_binance_oi_change_24h(base)

    # Weighting
    w_vol = min(fut_usd / 5_000_000, 2)   # 0–2
    w_mom = 1.0
    w_fr = 0.5
    w_oi = 0.3

    # Long score favors positive momentum, rising OI, positive funding
    long_score = (
        (pct24 * 0.4 + pct4 * 0.35 + pct1 * 0.25) * w_mom +
        (max(oi_ch, 0) * 0.6) * w_oi +
        (max(funding, 0) * 10) * w_fr +
        w_vol
    )

    # Short score favors negative momentum, falling OI, negative funding
    short_score = (
        (abs(min(pct24, 0)) * 0.4 +
         abs(min(pct4, 0)) * 0.35 +
         abs(min(pct1, 0)) * 0.25) * w_mom +
        (abs(min(oi_ch, 0)) * 0.6) * w_oi +
        (abs(min(funding, 0)) * 10) * w_fr +
        w_vol
    )

    return long_score, short_score


# =========================
#   TRADE SETUP GENERATOR
# =========================

def generate_trade_setup(side: str, last_price: float) -> Tuple[float, float, float]:
    """
    For each recommended trade we generate:
        entry, take_profit, stop_loss
    Simple multiples for now.
    """

    if last_price <= 0:
        return 0, 0, 0

    if side == "BUY":
        entry = last_price
        tp = last_price * 1.02   # +2%
        sl = last_price * 0.98   # -2%
    else:  # SELL
        entry = last_price
        tp = last_price * 0.98
        sl = last_price * 1.02

    return (round(entry, 4), round(tp, 4), round(sl, 4))


# =========================
#   BUILD TOP 4 SIGNALS
# =========================

async def build_top_signals(all_rows: List[Tuple[str, float, float, float, float, float]]):
    """
    all_rows: list of (BASE, fut_usd, spot_usd, pct24, pct4, pct1)

    Returns:
    - top_4 list of dicts:
        {
          "symbol": "AVAX",
          "side": "BUY"/"SELL",
          "score": float,
          "entry": float,
          "tp": float,
          "sl": float
        }
    """

    scored = []

    for (base, fut_usd, spot_usd, pct24, pct4, pct1) in all_rows:
        long_sc, short_sc = await compute_score(base, fut_usd, spot_usd, pct24, pct4, pct1)

        # pick the stronger side
        if long_sc >= short_sc:
            side = "BUY"
            score = long_sc
        else:
            side = "SELL"
            score = short_sc

        # get last price from coinex
        try:
            mk = f"{base}/USDT"
            lastp = float(coinex.fetch_ticker(mk)["last"])
        except:
            lastp = 0

        entry, tp, sl = generate_trade_setup(side, lastp)

        scored.append({
            "symbol": base,
            "side": side,
            "score": score,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "pct24": pct24,
            "pct4": pct4,
            "pct1": pct1
        })

    # Take top-4 by score
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:4]


# =========================
#   RECOMMENDATION DUPLICATE CHECK
# =========================

def same_recommendations(new_recs: List[Dict], old_recs: Optional[List[Tuple[str, str]]]) -> bool:
    """
    Compare (symbol + side).
    new_recs: list of dicts with 'symbol' and 'side'
    old_recs: list of tuples [(SYM, SIDE), ...]

    Returns True if identical.
    """
    if old_recs is None:
        return False

    new_set = {(r["symbol"], r["side"]) for r in new_recs}
    old_set = set(old_recs)

    return new_set == old_set


# =========================
#   TABLE BUILDER
# =========================

def style_pct(p: float) -> str:
    """Add arrows & no decimals."""
    rp = int(round(p))
    if rp > 0:
        return f"{rp}% 🔼"
    elif rp < 0:
        return f"{rp}% 🔽"
    else:
        return f"{rp}%"


def build_table(title: str, rows):
    """rows: (base, fut_usd, spot_usd, pct24, pct4, pct1)"""
    if not rows:
        return f"*{title}*\n_None_\n"

    tbl = []
    for (base, fut, spot, pct24, pct4, pct1) in rows:
        tbl.append([
            base,
            round_int(fut),
            round_int(spot),
            style_pct(pct24),
            style_pct(pct4),
            style_pct(pct1),
        ])

    text = f"*{title}*\n"
    text += "```\n"
    text += tabulate(tbl, headers=["SYM", "F", "S", "%24H", "%4H", "%1H"])
    text += "\n```\n"
    return text


# =========================
#   BUILD SCREEN OUTPUT
# =========================

async def build_screen():
    p1, p2, p3 = await build_priorities()

    # build top-4 across all rows
    all_rows = p1 + p2 + p3
    top4 = await build_top_signals(all_rows)

    # Build final screen text
    text = ""
    text += build_table("PRIORITY 1 (TOP 10)", p1)
    text += build_table("PRIORITY 2 (TOP 10)", p2)
    text += build_table("PRIORITY 3 (TOP 10)", p3)

    # Add recommendations
    text += "*Top 4 Recommendations*\n"
    for r in top4:
        text += (
            f"- *{r['side']} {r['symbol']}* "
            f"(score {round(r['score'],2)}) "
            f"Entry {r['entry']} / TP {r['tp']} / SL {r['sl']}\n"
        )

    return text, top4
# =========================
#   EMAIL SENDING
# =========================

def email_limit_ok() -> bool:
    """Check 2/hour + 20/day limit."""
    now = now_ts()

    # Remove old entries
    while EMAIL_SEND_LOG and now - EMAIL_SEND_LOG[0] > EMAIL_DAILY_WINDOW_SEC:
        EMAIL_SEND_LOG.pop(0)

    last_hour = [t for t in EMAIL_SEND_LOG if now - t < EMAIL_HOURLY_WINDOW_SEC]

    if len(EMAIL_SEND_LOG) >= EMAIL_DAILY_LIMIT:
        return False
    if len(last_hour) >= EMAIL_HOURLY_LIMIT:
        return False
    return True


def send_email_alert(subject: str, body: str):
    """Send email via SMTP SSL."""
    if not NOTIFY_ON:
        return

    if not email_limit_ok():
        logging.warning("Email suppressed due to rate limits.")
        return

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        msg.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=context) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)

        EMAIL_SEND_LOG.append(now_ts())
        logging.info(f"Email sent: {subject}")

    except Exception as e:
        logging.error(f"Email error: {e}")


# =========================
#   ALERT CHECK
# =========================

async def alert_job(app: Application):
    """
    Runs every 5 minutes.
    Conditions:
    - %4h >= 5 AND %1h >= 5 for any symbol
    - Email allowed only 09–21 Melbourne
    - Email sent ONLY if top-4 recommendations changed (symbol+side)
    """

    global LAST_REC_SET

    if not melbourne_ok():
        return

    p1, p2, p3 = await build_priorities()
    alert_rows = []

    # Collect any symbols hitting 4h>=5 and 1h>=5
    for (base, fut, spot, pct24, pct4, pct1) in p1 + p2 + p3:
        if pct4 >= ALERT_PCT_4H_MIN and pct1 >= ALERT_PCT_1H_MIN:
            alert_rows.append((base, pct4, pct1))

    if not alert_rows:
        return

    # Build top-4 recommendations
    all_rows = p1 + p2 + p3
    top4 = await build_top_signals(all_rows)

    # Duplicate check (symbol+side)
    new_rec_set = [(r["symbol"], r["side"]) for r in top4]

    if same_recommendations(top4, LAST_REC_SET):
        logging.info("Email suppressed (same recommended signals).")
        return

    # Build email text
    subject = "Crypto Alert (4H+1H spike)"
    body = "Symbols triggering alert:\n"
    for (sym, p4, p1h) in alert_rows:
        body += f"- {sym}: 4h={round(p4)}%, 1h={round(p1h)}%\n"

    body += "\nTop 4 Recommendations:\n"
    for r in top4:
        body += (
            f"{r['side']} {r['symbol']} | "
            f"Entry {r['entry']} | TP {r['tp']} | SL {r['sl']} | "
            f"Score {round(r['score'],2)}\n"
        )

    send_email_alert(subject, body)

    # Update last recommendation
    LAST_REC_SET = new_rec_set


# =========================
#   TELEGRAM HANDLERS
# =========================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot is running.\nUse /screen")


async def cmd_screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text, top4 = await build_screen()
    # Update cached recommendations (for immediate user queries)
    global LAST_REC_SET
    LAST_REC_SET = [(r["symbol"], r["side"]) for r in top4]

    try:
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except:
        # fallback
        await update.message.reply_text("Output too long. Try again.")


async def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now = melbourne_time().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"*Diagnostics*\nTime: {now}\nNotify: {NOTIFY_ON}\n"
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global NOTIFY_ON
    NOTIFY_ON = True
    await update.message.reply_text("Email alerts ENABLED.")


async def cmd_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global NOTIFY_ON
    NOTIFY_ON = False
    await update.message.reply_text("Email alerts DISABLED.")


async def query_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    User types symbol name: 'PYTH'
    Returns table with: %24h, %4h, %1h and top-4 trade view.
    """
    sym = update.message.text.strip().upper()
    if len(sym) > 10 or not sym.isalnum():
        return

    p1, p2, p3 = await build_priorities()
    for (base, fut, spot, pct24, pct4, pct1) in p1 + p2 + p3:
        if base == sym:
            tbl = (
                f"*{sym} Analysis*\n"
                f"%24H: {style_pct(pct24)}\n"
                f"%4H : {style_pct(pct4)}\n"
                f"%1H : {style_pct(pct1)}\n"
            )
            await update.message.reply_text(tbl, parse_mode=ParseMode.MARKDOWN)
            return

    await update.message.reply_text("Symbol not found in priorities.")


# =========================
#           MAIN
# =========================

async def post_init(app: Application):
    # Start background alert loop
    async def run_alerts():
        while True:
            try:
                await alert_job(app)
            except Exception as e:
                logging.error(f"Alert job error: {e}")
            await asyncio.sleep(CHECK_INTERVAL_MIN * 60)

    # Now safe to create background task
    asyncio.create_task(run_alerts())


def main():
    if not TOKEN:
        raise RuntimeError("Set TELEGRAM_TOKEN env var")

    logging.basicConfig(level=logging.INFO)

    app = Application.builder().token(TOKEN).post_init(post_init).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("screen", cmd_screen))
    app.add_handler(CommandHandler("diag", cmd_diag))
    app.add_handler(CommandHandler("on", cmd_on))
    app.add_handler(CommandHandler("off", cmd_off))

    # Symbol text
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, query_symbol))

    # Start bot (this creates and manages event loop internally)
    app.run_polling()


if __name__ == "__main__":
    main()
