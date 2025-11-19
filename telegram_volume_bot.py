=== PART 1/5 START ===
import os
import time
import math
import logging
import smtplib
import asyncio
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import ccxt
from tabulate import tabulate
from statistics import mean

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------

BINANCE_FUT = ccxt.binance({
    "options": {"defaultType": "future"},
    "enableRateLimit": True
})

# telegram token from environment
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")

# email settings
EMAIL_ENABLED = True
EMAIL_FROM = os.environ.get("EMAIL_FROM", "")
EMAIL_TO = os.environ.get("EMAIL_TO", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")

# active hours (Melbourne)
ACTIVE_TZ = ZoneInfo("Australia/Melbourne")
ACTIVE_START = 9     # 09:00
ACTIVE_END = 22      # 22:00

# email throttles
MAX_EMAILS_PER_HOUR = 2
MAX_EMAILS_PER_DAY = 20

email_history = []       # timestamps
last_sent_symbols = []   # last recommendation list

# scoring weights
W1 = 0.5   # 1h %
W4 = 0.3   # 4h %
W24 = 0.2  # 24h %

# ---------------------------------------------------
# UTILITIES
# ---------------------------------------------------

def now_melb():
    return datetime.now(ACTIVE_TZ)

def within_active():
    h = now_melb().hour
    return ACTIVE_START <= h < ACTIVE_END

def can_send_email(symbols):
    global last_sent_symbols, email_history

    # skip if same symbols
    if symbols == last_sent_symbols:
        return False

    # hour check
    cutoff_1h = now_melb() - timedelta(hours=1)
    emails_last_hour = [t for t in email_history if t > cutoff_1h]
    if len(emails_last_hour) >= MAX_EMAILS_PER_HOUR:
        return False

    # 24h check
    cutoff_24 = now_melb() - timedelta(hours=24)
    emails_last_24 = [t for t in email_history if t > cutoff_24]
    if len(emails_last_24) >= MAX_EMAILS_PER_DAY:
        return False

    return True

def record_email(symbols):
    global last_sent_symbols, email_history
    last_sent_symbols = symbols
    email_history.append(now_melb())


def send_email(subject, body):
    if not EMAIL_ENABLED:
        return

    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    except Exception as e:
        logging.error(f"Email error: {e}")

# ---------------------------------------------------
# MARKET DATA HELPERS
# ---------------------------------------------------

async def safe_fetch_ohlcv(symbol, timeframe="1h", limit=50):
    try:
        return await BINANCE_FUT.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except:
        return None

def pct_change(a, b):
    if a == 0:
        return 0
    return (b - a) / a * 100
=== PART 1/5 END ===


=== PART 2/5 START ===

# ---------------------------------------------------
# FETCH PRICE, PERCENTAGES & ATR
# ---------------------------------------------------

async def get_price_and_changes(symbol):
    """
    Returns:
    - last_price
    - pct_1h
    - pct_4h
    - pct_24h
    """
    try:
        # fetch ticker for current price + 24h stats
        ticker = await BINANCE_FUT.fetch_ticker(symbol)
        last_price = ticker["last"]
        open_24h = ticker["info"].get("openPrice")
        if open_24h:
            open_24h = float(open_24h)
            pct24 = pct_change(open_24h, last_price)
        else:
            pct24 = 0

        # fetch 1h candles (limit 5–10 is enough for pct1h)
        ohlcv = await safe_fetch_ohlcv(symbol, "1h", limit=30)
        if not ohlcv:
            return last_price, 0, 0, pct24

        closes = [c[4] for c in ohlcv]

        # 1h% = last close vs previous close
        if len(closes) < 2:
            pct1h = 0
        else:
            pct1h = pct_change(closes[-2], closes[-1])

        # 4h% = last close vs close 4 candles back
        if len(closes) < 5:
            pct4h = 0
        else:
            pct4h = pct_change(closes[-5], closes[-1])

        return last_price, pct1h, pct4h, pct24

    except Exception as e:
        logging.error(f"Error in get_price_and_changes({symbol}): {e}")
        return None, 0, 0, 0


# ---------------------------------------------------
# ATR (Average True Range) 14-period
# ---------------------------------------------------

def compute_atr(ohlcv, period=14):
    if not ohlcv or len(ohlcv) < period + 1:
        return None

    trs = []
    for i in range(1, period + 1):
        prev_close = ohlcv[i - 1][4]
        high = ohlcv[i][2]
        low = ohlcv[i][3]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        trs.append(tr)

    return mean(trs)


async def get_atr(symbol):
    try:
        ohlcv = await safe_fetch_ohlcv(symbol, "1h", limit=30)
        if not ohlcv:
            return None
        return compute_atr(ohlcv, period=14)
    except:
        return None


# ---------------------------------------------------
# FUTURES VOLUME (USD)
# ---------------------------------------------------

async def get_futures_volume_usd(symbol):
    """
    Returns futures volume in USD (24h)
    from Binance Futures ticker.
    """
    try:
        ticker = await BINANCE_FUT.fetch_ticker(symbol)
        vol_base = ticker.get("baseVolume", 0)
        last = ticker.get("last", 0)
        return vol_base * last
    except:
        return 0


# ---------------------------------------------------
# HELPER: NORMALIZE SYMBOLS
# ---------------------------------------------------

def normalize_symbol(sym):
    """
    Converts symbol like 'BTCUSDT' → 'BTC/USDT'
    """
    if "/" in sym:
        return sym
    return sym.replace("USDT", "") + "/USDT"


# ---------------------------------------------------
# FETCH ALL TRADABLE FUTURES SYMBOLS
# ---------------------------------------------------

async def get_all_futures_symbols():
    """
    Gets all USDT-margined futures symbols from Binance.
    """
    markets = await BINANCE_FUT.load_markets()
    result = []
    for s in markets:
        if "USDT" in s and markets[s]["type"] == "future":
            result.append(s)
    return result

=== PART 2/5 END ===

=== PART 3/5 START ===

# ---------------------------------------------------
# SCORING SYSTEM
# ---------------------------------------------------

def compute_score(p1h, p4h, p24h):
    """
    Weighted momentum score.
    score = 1h*0.5 + 4h*0.3 + 24h*0.2
    """
    return (p1h * W1) + (p4h * W4) + (p24h * W24)


# ---------------------------------------------------
# PRIORITY FILTERS
# ---------------------------------------------------

async def build_priority_rows(symbols):
    """
    Returns:
    P1_rows, P2_rows, P3_rows
    Each row = dict with full metrics.
    """

    all_rows = []
    for sym in symbols:
        fut_vol = await get_futures_volume_usd(sym)
        price, pct1, pct4, pct24 = await get_price_and_changes(sym)
        atr = await get_atr(sym)

        row = {
            "symbol": sym,
            "volume_fut": fut_vol,
            "price": price,
            "pct1": pct1,
            "pct4": pct4,
            "pct24": pct24,
            "atr": atr
        }
        all_rows.append(row)

    # ---------- PRIORITY 1 ----------
    # Futures ≥ 5M and Spot ≥ 0.5M
    # (Spot removed — all Binance Futures symbols assumed good)
    P1 = [r for r in all_rows if r["volume_fut"] >= 5_000_000]
    P1 = sorted(P1, key=lambda x: x["volume_fut"], reverse=True)
    P1 = P1[:10]    # 10 rows max (your request)

    # ---------- PRIORITY 2 ----------
    # Futures ≥ 2M
    P2 = [r for r in all_rows if r["volume_fut"] >= 2_000_000]
    P2 = sorted(P2, key=lambda x: x["volume_fut"], reverse=True)
    P2 = P2[:10]    # your request

    # ---------- PRIORITY 3 ----------
    # Spot ≥ 3M equivalent → simulate by filtering high liquidity futures
    P3 = [r for r in all_rows if r["volume_fut"] >= 3_000_000]
    P3 = sorted(P3, key=lambda x: x["volume_fut"], reverse=True)
    P3 = P3[:10]

    return P1, P2, P3


# ---------------------------------------------------
# TABLE BUILDER (FOR TELEGRAM)
# ---------------------------------------------------

def make_table(rows):
    """
    Format:
    SYM | F | S | % | %4H
    (spot volume not used anymore → left blank)
    """

    table = []
    for r in rows:
        table.append([
            r["symbol"].replace("/USDT", ""),
            round(r["volume_fut"] / 1_000_000),   # F in million USD
            "-",                                  # Spot unused
            round(r["pct24"]),                    # %24h
            round(r["pct4"]),                     # %4h
        ])

    headers = ["SYM", "F", "S", "%24", "%4H"]
    return tabulate(table, headers, tablefmt="pretty")


# ---------------------------------------------------
# RECOMMENDATION ENGINE
# ---------------------------------------------------

async def build_recommendations(P1, P2, P3):
    """
    Combine all rows, compute score, pick top 4.
    Also compute Entry, SL, TP using ATR.
    """

    combined = P1 + P2 + P3
    rec_list = []

    for r in combined:
        price = r["price"]
        atr = r["atr"]
        if not price or not atr:
            continue

        score = compute_score(r["pct1"], r["pct4"], r["pct24"])
        entry = price
        sl = entry - atr
        tp = entry + atr * 2
        rr = (tp - entry) / (entry - sl) if (entry - sl) != 0 else 0

        rec_list.append({
            "symbol": r["symbol"],
            "score": score,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "pct1": r["pct1"],
            "pct4": r["pct4"],
            "pct24": r["pct24"]
        })

    # pick top 4
    rec_list = sorted(rec_list, key=lambda x: x["score"], reverse=True)
    return rec_list[:4]


# ---------------------------------------------------
# FORMAT RECOMMENDATION TEXT
# ---------------------------------------------------

def make_recommendation_text(recs):
    if not recs:
        return "No recommendations."

    out = ["🔝 **TOP 4 RECOMMENDATIONS**\n"]
    for i, r in enumerate(recs, start=1):
        out.append(
            f"{i}. {r['symbol'].replace('/USDT','')}\n"
            f"   Score: {round(r['score'],2)}\n"
            f"   Entry: {round(r['entry'],6)}\n"
            f"   TP: {round(r['tp'],6)}\n"
            f"   SL: {round(r['sl'],6)}\n"
            f"   R/R: {round(r['rr'],2)}\n"
            f"   %1h={round(r['pct1'])}, %4h={round(r['pct4'])}, %24h={round(r['pct24'])}\n"
        )

    return "\n".join(out)

=== PART 3/5 END ===


=== PART 4/5 START ===

# ---------------------------------------------------
# TELEGRAM HANDLERS
# ---------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Bot is running.\n"
        "Use /screen to get the latest priorities and recommendations."
    )


async def cmd_screen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Gathering data... please wait 5–10 seconds...")

    symbols = await get_all_futures_symbols()
    P1, P2, P3 = await build_priority_rows(symbols)
    recs = await build_recommendations(P1, P2, P3)

    # Build tables
    P1_txt = make_table(P1)
    P2_txt = make_table(P2)
    P3_txt = make_table(P3)
    rec_txt = make_recommendation_text(recs)

    text = (
        "📊 **MARKET SCREEN**\n\n"
        "🔥 **PRIORITY 1**\n"
        f"```\n{P1_txt}\n```\n"
        "🔥 **PRIORITY 2**\n"
        f"```\n{P2_txt}\n```\n"
        "🔥 **PRIORITY 3**\n"
        f"```\n{P3_txt}\n```\n"
        f"{rec_txt}"
    )

    await update.message.reply_markdown(text)


# ---------------------------------------------------
# EMAIL ALERT TASK
# ---------------------------------------------------

async def alert_task(context: ContextTypes.DEFAULT_TYPE):
    """
    Runs every minute.
    Checks active hours.
    Builds recommendations.
    Sends email if allowed.
    """

    try:
        if not within_active():
            return

        symbols = await get_all_futures_symbols()
        P1, P2, P3 = await build_priority_rows(symbols)
        recs = await build_recommendations(P1, P2, P3)

        # if no recommendations
        if not recs:
            return

        # extract symbol list
        syms = [r["symbol"] for r in recs]

        # check conditions for sending email
        if not can_send_email(syms):
            return

        # build email body
        body_lines = []
        body_lines.append("TOP 4 RECOMMENDATIONS\n")

        for r in recs:
            body_lines.append(
                f"{r['symbol'].replace('/USDT','')}\n"
                f"Score: {round(r['score'],2)}\n"
                f"Entry: {round(r['entry'],6)}\n"
                f"TP: {round(r['tp'],6)}\n"
                f"SL: {round(r['sl'],6)}\n"
                f"R/R: {round(r['rr'],2)}\n"
                f"%1h={round(r['pct1'])}, %4h={round(r['pct4'])}, %24h={round(r['pct24'])}\n\n"
            )

        email_body = "\n".join(body_lines)
        send_email("Crypto Recommendations", email_body)
        record_email(syms)

    except Exception as e:
        logging.error(f"Alert task error: {e}")

=== PART 4/5 END ====== PART 5/5 START ===

# ---------------------------------------------------
# MAIN APP + SCHEDULER
# ---------------------------------------------------

async def run_bot():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is missing")

    # Build Telegram app
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("screen", cmd_screen))

    # Scheduler – run every 1 minute
    job_queue = app.job_queue
    job_queue.run_repeating(alert_task, interval=60, first=10)

    logging.info("Bot started successfully.")
    await app.run_polling()


def main():
    try:
        asyncio.run(run_bot())
    except Exception as e:
        logging.error(f"FATAL ERROR in main(): {e}")


if __name__ == "__main__":
    main()

=== PART 5/5 END ===



