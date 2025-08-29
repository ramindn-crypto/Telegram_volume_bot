#!/usr/bin/env python3
"""
CoinEx screener bot
Output columns (Telegram /screen):
  SYM | FUT | SPOT | % (with emoji, integers only)

- Excludes: BTC, ETH, XRP, SOL, DOGE, ADA, PEPE, LINK
- /screen → P1: 10 rows, P2: 5 rows, P3: 5 rows
- /excel  → Excel .xlsx (priority,symbol,usd_24h)
- /diag   → diagnostics
"""

import asyncio, logging, os, time, io, traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import ccxt  # type: ignore
from tabulate import tabulate  # type: ignore
from openpyxl import Workbook  # type: ignore
from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ---- Config / thresholds ----
P1_SPOT_MIN = 500_000
P1_FUT_MIN  = 5_000_000
P2_FUT_MIN  = 2_000_000
P3_SPOT_MIN = 3_000_000   # Spot threshold

# Rows per priority
TOP_N_P1    = 10
TOP_N_P2    = 5
TOP_N_P3    = 5

EXCHANGE_ID = "coinex"
TOKEN = os.environ.get("TELEGRAM_TOKEN")
STABLES = {"USD","USDT","USDC","TUSD","FDUSD","USDD","USDE","DAI","PYUSD"}

# Exclusions
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
    quo
