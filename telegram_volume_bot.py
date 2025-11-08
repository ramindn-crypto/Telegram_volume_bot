#!/usr/bin/env python3
"""
Telegram crypto screener & alert bot (CoinEx via CCXT)

🧩 Features:
- Shows 3 priorities of coins (P1, P2, P3)
- Table columns: SYM | F | S | %24H | %4H | %1H
- Email alerts if 4h ≥ +5% AND 1h ≥ +5%
- Sends emails only 11 AM – 11 PM (Melbourne)
- Max 2 emails/hour, 20 emails/day, 15 min cool-down
- Check interval = 5 min
- Commands: /screen /notify_on /notify_off /notify /diag
- You can also just type a coin name (e.g. PYTH)

Pinned coins (BTC ETH XRP SOL DOGE ADA PEPE LINK) appear ONLY in P3.
"""

import asyncio, logging, os, time, traceback, re, ssl, smtplib, io
from email.message import EmailMessage
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

import ccxt
from tabulate import tabulate
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes,
    MessageHandler, filters, ErrorHandler
)
from telegram.error import Conflict

# ================== CONFIG ==================

EXCHANGE_ID = "coinex"
TOKEN = os.environ.get("TELEGRAM_TOKEN")

# --- thresholds ---
P1_SPOT_MIN = 500_000
P1_FUT_MIN  = 5_000_000
P2_FUT_MIN  = 2_000_000
P3_SPOT_MIN = 3_000_000

TOP_N_P1 = 10
TOP_N_P2 = 5
TOP_N_P3 = 10

STABLES = {"USD","USDT","USDC","TUSD","FDUSD","USDD","USDE","DAI","PYUSD"}

# --- pinned coins (only in P3) ---
PINNED_P3 = ["BTC","ETH","XRP","SOL","DOGE","ADA","PEPE","LINK"]
PINNED_SET = set(PINNED_P3)

# --- email configuration (set in Render env vars) ---
EMAIL_ENABLED_DEFAULT = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST = os.environ.get("EMAIL_HOST", "")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "465"))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", EMAIL_USER)
EMAIL_TO   = os.environ.get("EMAIL_TO", EMAIL_USER)

# --- scheduler / alerts ---
CHECK_INTERVAL_MIN = 5                   # run every 5 min
ALERT_PCT_4H_MIN = 5.0
ALERT_PCT_1H_MIN = 5.0
ALERT_THROTTLE_SEC = 15 * 60             # 15 min cool-down per symbol & priority
EMAIL_DAILY_LIMIT = 20
EMAIL_HOURLY_LIMIT = 2
EMAIL_DAILY_WINDOW_SEC = 24 * 60 * 60
EMAIL_HOURLY_WINDOW_SEC = 60 * 60

# --- globals ---
LAST_ERROR: Optional[str] = None
NOTIFY_ON: bool = EMAIL_ENABLED_DEFAULT
PCT4H_CACHE: Dict[Tuple[str,str], float] = {}
PCT1H_CACHE: Dict[Tuple[str,str], float] = {}
ALERT_SENT_CACHE: Dict[Tuple[str,str], float] = {}
EMAIL_SEND_LOG: List[float] = []

# ================== HELPERS ==================

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
    sp = safe_split_symbol(sym)
    if not sp: return None
    base, quote = sp
    return MarketVol(
        sym, base, quote,
        float(t.get("last") or 0.0),
        float(t.get("open") or 0.0),
        float(t.get("percentage") or 0.0),
        float(t.get("baseVolume") or 0.0),
        float(t.get("quoteVolume") or 0.0),
        float(t.get("vwap") or 0.0),
    )

def usd_notional(mv: Optional[MarketVol]) -> float:
    """Return 24h notional volume in USD."""
    if not mv: return 0.0
    if mv.quote in STABLES and mv.quote_vol:
        return mv.quote_vol
    price = mv.vwap if mv.vwap else mv.last
    return mv.base_vol * price if mv.base_vol else 0.0

def pct_change(mv_spot: Optional[MarketVol], mv_fut: Optional[MarketVol]) -> float:
    """24h % change (prefer ticker.percentage)."""
    for mv in (mv_spot, mv_fut):
        if mv and mv.percentage:
            return mv.percentage
    mv = mv_spot or mv_fut
    if mv and mv.open:
        return (mv.last - mv.open) / mv.open * 100
    return 0.0

def pct_with_emoji(p: float) -> str:
    val = round(p)
    if val >= 3: emo = "🟢"
    elif val <= -3: emo = "🔴"
    else: emo = "🟡"
    return f"{val:+d}% {emo}"

def m_dollars(x: float) -> str:
    return str(round(x / 1_000_000))

def build_exchange(default_type: str):
    klass = ccxt.__dict__[EXCHANGE_ID]
    return klass({"enableRateLimit": True, "timeout": 20000,
                  "options": {"defaultType": default_type}})

def safe_fetch_tickers(ex):
    try:
        ex.load_markets()
        return ex.fetch_tickers()
    except Exception as e:
        global LAST_ERROR
        LAST_ERROR = f"{type(e).__name__}: {e}"
        logging.exception("fetch_tickers failed")
        return {}

# ================== PERCENTAGE FETCH (4H / 1H) ==================

def compute_pct_for_symbol(symbol: str, hours: int, prefer_swap=True) -> float:
    """
    Compute % change over the last N completed hours using 1h candles.
    """
    try_order = ["swap", "spot"] if prefer_swap else ["spot", "swap"]
    for dtype in try_order:
        ck = (dtype, symbol, hours)
        if hours == 4 and ck in PCT4H_CACHE:
            return PCT4H_CACHE[ck]
        if hours == 1 and ck in PCT1H_CACHE:
            return PCT1H_CACHE[ck]
        try:
            ex = build_exchange(dtype)
            ex.load_markets()
            candles = ex.fetch_ohlcv(symbol, timeframe="1h", limit=hours + 1)
            if not candles or len(candles) <= hours:
                continue
            closes = [c[4] for c in candles][- (hours + 1) :]
            pct = ((closes[-1] - closes[0]) / closes[0] * 100.0) if closes[0] else 0.0
            if hours == 4:
                PCT4H_CACHE[ck] = pct
            else:
                PCT1H_CACHE[ck] = pct
            return pct
        except Exception:
            logging.exception("pct_%dh failed for %s (%s)", hours, symbol, dtype)
            continue
    return 0.0

# ================== BUILD PRIORITIES ==================

def load_best():
    """Load top tickers for spot and futures (swap)."""
    ex_spot = build_exchange("spot")
    ex_fut = build_exchange("swap")
    spot_tickers = safe_fetch_tickers(ex_spot)
    fut_tickers = safe_fetch_tickers(ex_fut)

    best_spot, best_fut = {}, {}
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

def build_priorities(best_spot, best_fut):
    """
    Build P1, P2, P3 lists.
    Each row: [SYM, FUSD, SUSD, %24H, %4H, %1H]
    """
    p1, p2, p3 = [], [], []
    used = set()

    # P1
    for b in set(best_spot) & set(best_fut):
        if b in PINNED_SET: continue
        s, f = best_spot[b], best_fut[b]
        fut, spot = usd_notional(f), usd_notional(s)
        if fut >= P1_FUT_MIN and spot >= P1_SPOT_MIN:
            p1.append([b, fut, spot, pct_change(s,f),
                       compute_pct_for_symbol(f.symbol,4),
                       compute_pct_for_symbol(f.symbol,1)])
    p1.sort(key=lambda x: x[1], reverse=True)
    p1 = p1[:TOP_N_P1]
    used |= {r[0] for r in p1}

    # P2
    for b,f in best_fut.items():
        if b in used or b in PINNED_SET: continue
        fut = usd_notional(f)
        if fut >= P2_FUT_MIN:
            s = best_spot.get(b)
            p2.append([b, fut, usd_notional(s) if s else 0,
                       pct_change(s,f),
                       compute_pct_for_symbol(f.symbol,4),
                       compute_pct_for_symbol(f.symbol,1)])
    p2.sort(key=lambda x:x[1], reverse=True)
    p2 = p2[:TOP_N_P2]
    used |= {r[0] for r in p2}

    # P3: pinned + others by spot≥3M
    temp = {}
    for b in PINNED_P3:
        s,f = best_spot.get(b), best_fut.get(b)
        if not s and not f: continue
        temp[b] = [b,
                   usd_notional(f) if f else 0,
                   usd_notional(s) if s else 0,
                   pct_change(s,f),
                   compute_pct_for_symbol(f.symbol if f else s.symbol,4),
                   compute_pct_for_symbol(f.symbol if f else s.symbol,1)]
    for b,s in best_spot.items():
        if b in used or b in PINNED_SET: continue
        if usd_notional(s)>=P3_SPOT_MIN:
            f=best_fut.get(b)
            temp[b]=[b,usd_notional(f) if f else 0,usd_notional(s),
                     pct_change(s,f),
                     compute_pct_for_symbol(f.symbol if f else s.symbol,4),
                     compute_pct_for_symbol(f.symbol if f else s.symbol,1)]
    allr=list(temp.values())
    pins=[r for r in allr if r[0] in PINNED_SET]
    others=[r for r in allr if r[0] not in PINNED_SET]
    pins.sort(key=lambda r:r[2],reverse=True)
    others.sort(key=lambda r:r[2],reverse=True)
    p3=(pins+others)[:TOP_N_P3]
    return p1,p2,p3

# ================== FORMATTING ==================

def fmt_table(rows,title):
    if not rows: return f"*{title}*: _None_\n"
    pretty=[[r[0],m_dollars(r[1]),m_dollars(r[2]),
             pct_with_emoji(r[3]),pct_with_emoji(r[4]),pct_with_emoji(r[5])] for r in rows]
    return f"*{title}*:\n```\n"+tabulate(pretty,headers=["SYM","F","S","%24H","%4H","%1H"],tablefmt="github")+"\n```\n"

def fmt_single(sym,fusd,susd,p24,p4,p1):
    row=[[sym,m_dollars(fusd),m_dollars(susd),
          pct_with_emoji(p24),pct_with_emoji(p4),pct_with_emoji(p1)]]
    return "```\n"+tabulate(row,headers=["SYM","F","S","%24H","%4H","%1H"],tablefmt="github")+"\n```"

# ================== EMAIL HELPERS ==================

def email_config_ok():
    return all([EMAIL_HOST,EMAIL_PORT,EMAIL_USER,EMAIL_PASS,EMAIL_FROM,EMAIL_TO])

def send_email(subject,body):
    if not email_config_ok():
        logging.warning("Email not configured.")
        return False
    try:
        msg=EmailMessage();msg["Subject"]=subject;msg["From"]=EMAIL_FROM;msg["To"]=EMAIL_TO
        msg.set_content(body)
        if EMAIL_PORT==465:
            ctx=ssl.create_default_context()
            with smtplib.SMTP_SSL(EMAIL_HOST,EMAIL_PORT,context=ctx) as s:
                s.login(EMAIL_USER,EMAIL_PASS);s.send_message(msg)
        else:
            with smtplib.SMTP(EMAIL_HOST,EMAIL_PORT) as s:
                s.starttls();s.login(EMAIL_USER,EMAIL_PASS);s.send_message(msg)
        return True
    except Exception as e:
        logging.exception("send_email failed: %s",e)
        return False

def melbourne_ok():
    try:
        hr=datetime.now(ZoneInfo("Australia/Melbourne")).hour
        return 11<=hr<23
    except Exception:
        return True

def email_quota_ok():
    now=time.time()
    EMAIL_SEND_LOG[:]=[x for x in EMAIL_SEND_LOG if now-x<EMAIL_DAILY_WINDOW_SEC]
    if len(EMAIL_SEND_LOG)>=EMAIL_DAILY_LIMIT: return False
    if sum(1 for x in EMAIL_SEND_LOG if now-x<EMAIL_HOURLY_WINDOW_SEC)>=EMAIL_HOURLY_LIMIT: return False
    return True

def record_email(): EMAIL_SEND_LOG.append(time.time())

def should_alert(prio,sym):
    now=time.time();key=(prio,sym)
    last=ALERT_SENT_CACHE.get(key,0)
    if now-last>=ALERT_THROTTLE_SEC:
        ALERT_SENT_CACHE[key]=now;return True
    return False

def scan_for_alerts(p1,p2,p3):
    lines=[]
    for lbl,rows in (("P1",p1),("P2",p2),("P3",p3)):
        hit=[]
        for s,_,_,_,p4,p1h in rows:
            if p4>=ALERT_PCT_4H_MIN and p1h>=ALERT_PCT_1H_MIN and should_alert(lbl,s):
                hit.append(s)
        if hit: lines.append(f"{lbl}: "+", ".join(hit))
    return None if not lines else "Coins +5% (4h & 1h):\n\n"+"\n".join(lines)

# ================== TELEGRAM HANDLERS ==================

async def start(u,c): await u.message.reply_text("Use /screen • /notify_on • /notify_off • /notify • /diag\nType a symbol (e.g. PYTH) to get its data.")

async def screen(u,c):
    try:
        PCT4H_CACHE.clear();PCT1H_CACHE.clear()
        bs,bf,_,_=await asyncio.to_thread(load_best)
        p1,p2,p3=await asyncio.to_thread(build_priorities,bs,bf)
        msg=fmt_table(p1,"P1 (F≥5M,S≥0.5M)")+fmt_table(p2,"P2 (F≥2M)")+fmt_table(p3,"P3 (Pinned+S≥3M)")
        await u.message.reply_text(msg,parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logging.exception("screen");await u.message.reply_text(f"Error: {e}")

async def diag(u,c):
    msg=f"*Diag*  Rule: +5% (4h & 1h)  Interval: {CHECK_INTERVAL_MIN}min\nEmail: {'ON' if NOTIFY_ON else 'OFF'}"
    await u.message.reply_text(msg,parse_mode=ParseMode.MARKDOWN)

async def notify_on(u,c): 
    global NOTIFY_ON;NOTIFY_ON=True;await u.message.reply_text("Email alerts: ON")
async def notify_off(u,c):
    global NOTIFY_ON;NOTIFY_ON=False;await u.message.reply_text("Email alerts: OFF")
async def notify(u,c):
    await u.message.reply_text(f"Email alerts: {'ON' if NOTIFY_ON else 'OFF'} → {EMAIL_TO}")

async def text_router(u,c):
    sym=re.sub(r'[^A-Za-z$]','',u.message.text or '').upper().lstrip('$')
    if len(sym)<2: return
    bs,bf,_,_=await asyncio.to_thread(load_best)
    s,f=bs.get(sym),bf.get(sym)
    fusd,susd=usd_notional(f) if f else 0,usd_notional(s) if s else 0
    if not fusd and not susd:
        await u.message.reply_text("Not found.");return
    msg=fmt_single(sym,fusd,susd,
                   pct_change(s,f),
                   compute_pct_for_symbol(f.symbol if f else s.symbol,4),
                   compute_pct_for_symbol(f.symbol if f else s.symbol,1))
    await u.message.reply_text(msg,parse_mode=ParseMode.MARKDOWN)

# ================== ALERT JOB ==================

async def alert_job(c):
    if not NOTIFY_ON or not melbourne_ok() or not email_config_ok(): return
    if not email_quota_ok(): return
    bs,bf,_,_=await asyncio.to_thread(load_best)
    p1,p2,p3=await asyncio.to_thread(build_priorities,bs,bf)
    body=scan_for_alerts(p1,p2,p3)
    if body and send_email("Crypto Alert: +5% (4h) & +5% (1h)",body):
        record_email()

# ================== MAIN ==================

async def log_err(u,c):
    if isinstance(c.error,Conflict):
        logging.warning("Conflict: duplicate polling.");return
    logging.exception("Error: %s",c.error)

def main():
    if not TOKEN: raise RuntimeError("TELEGRAM_TOKEN missing")
    logging.basicConfig(level=logging.INFO)
    app=Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start",start))
    app.add_handler(CommandHandler("screen",screen))
    app.add_handler(CommandHandler("diag",diag))
    app.add_handler(CommandHandler("notify_on",notify_on))
    app.add_handler(CommandHandler("notify_off",notify_off))
    app.add_handler(CommandHandler("notify",notify))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,text_router))
    app.add_error_handler(ErrorHandler(log_err))
    if getattr(app,"job_queue",None):
        app.job_queue.run_repeating(alert_job,interval=CHECK_INTERVAL_MIN*60,first=10)
    app.run_polling(drop_pending_updates=True)

if __name__=="__main__": main()

