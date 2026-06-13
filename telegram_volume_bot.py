#!/usr/bin/env python3
"""
PulseFutures emergency Telegram connectivity test.
This file is intentionally NOT the trading bot.
Purpose: prove whether Render + TELEGRAM_TOKEN + Telegram polling can reply at all.
If this replies to /start, the freeze is inside the full PulseFutures bot startup/jobs.
If this does not reply, the issue is Render deployment/env/token/polling conflict/wrong service.
"""
import os
import sys
import time
import signal
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("pulsefutures.pingtest")

TOKEN = os.getenv("TELEGRAM_TOKEN")
BOOT_TS = time.time()


def start_keepalive_http_server():
    try:
        port = int(os.getenv("PORT", "10000") or 10000)
    except Exception:
        port = 10000

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                body = f"ok pulsefutures ping-test uptime={time.time()-BOOT_TS:.1f}s\n".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                pass

        def log_message(self, fmt, *args):
            return

    try:
        srv = HTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        logger.info("HTTP keepalive bound on 0.0.0.0:%s", port)
    except Exception as exc:
        logger.exception("HTTP keepalive failed: %s", exc)


async def post_init(app: Application):
    try:
        me = await app.bot.get_me()
        logger.info("Telegram get_me OK: id=%s username=@%s name=%s", me.id, me.username, me.first_name)
    except Exception as exc:
        logger.exception("Telegram get_me FAILED: %s", exc)


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else 0
    chat_id = update.effective_chat.id if update.effective_chat else 0
    await update.effective_message.reply_text(
        "✅ PulseFutures ping-test is alive.\n"
        f"User ID: {uid}\n"
        f"Chat ID: {chat_id}\n"
        f"Uptime: {time.time()-BOOT_TS:.1f}s\n"
        "This proves Telegram polling is working."
    )


async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(f"pong ✅ uptime={time.time()-BOOT_TS:.1f}s")


async def whoami_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat = update.effective_chat
    await update.effective_message.reply_text(
        f"user_id={getattr(user, 'id', 0)}\n"
        f"username=@{getattr(user, 'username', '') or '-'}\n"
        f"chat_id={getattr(chat, 'id', 0)}\n"
        f"chat_type={getattr(chat, 'type', '')}"
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Telegram handler error update=%r", update, exc_info=context.error)


def main():
    logger.info("Starting PulseFutures Telegram ping-test file=%s", __file__)
    if not TOKEN:
        logger.error("TELEGRAM_TOKEN is missing. Render env var must be TELEGRAM_TOKEN.")
        raise SystemExit(2)
    start_keepalive_http_server()

    app = (
        Application.builder()
        .token(TOKEN)
        .post_init(post_init)
        .concurrent_updates(4)
        .connect_timeout(10)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(10)
        .build()
    )
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("whoami", whoami_cmd))
    app.add_error_handler(error_handler)

    logger.info("Calling run_polling(drop_pending_updates=True). Stop every other Render service using same TELEGRAM_TOKEN first.")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )


if __name__ == "__main__":
    main()
