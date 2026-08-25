"""
Loads secrets from a local .env file (never from chat, never hardcoded, never
committed -- see .gitignore). If a required key is missing, functions that
need it fail with a clear message rather than silently limping along.
"""

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET")


def require_groq_key():
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )
    return GROQ_API_KEY


def require_razorpay_keys():
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise RuntimeError(
            "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are not set. "
            "Copy .env.example to .env and fill in your test-mode keys."
        )
    # Razorpay key IDs are always prefixed rzp_test_ or rzp_live_ -- this
    # project's whole premise (see integrations/razorpay_client.py's module
    # docstring, and the "TEST MODE" pill the dashboard always shows) is
    # that it only ever creates test-mode orders. Nothing before this point
    # actually checked that the configured key is a test key, so pasting a
    # live key into .env by mistake -- an easy slip, since Razorpay's
    # dashboard shows both key sets side by side -- would authenticate
    # against Razorpay's live API and create real Order records, with the
    # UI still confidently labeled "TEST MODE" the whole time.
    if not RAZORPAY_KEY_ID.startswith("rzp_test_"):
        raise RuntimeError(
            "RAZORPAY_KEY_ID does not look like a test-mode key (expected it to "
            "start with 'rzp_test_'). RiskLens is a demo that only ever creates "
            "test-mode orders -- refusing to run against what looks like a live "
            "key. Double-check RAZORPAY_KEY_ID in your .env file."
        )
    return RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET
