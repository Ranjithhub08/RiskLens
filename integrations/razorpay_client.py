"""
Real (test-mode) Razorpay API integration.

This is a genuine, authenticated call to Razorpay's actual API server --
not a mock, not a hand-rolled fixture. In test mode, Order creation is the
right integration point for a server-side agent: capturing a live payment
requires the customer-facing checkout flow (by design, no payment gateway
lets a backend charge a card without that flow), but creating an Order is
a full round trip to Razorpay's real infrastructure with a real Order ID
coming back, and it's exactly the first API call a real merchant
integration makes before checkout.

RiskLens uses these live orders as the "transaction" half of its input --
amount, currency, receipt, timestamp, status. The merchant *history*
(KYC status, chargeback rate, account age, etc. -- the fields no public or
sandbox API can give a student project) stays simulated and is clearly
labeled as such throughout. See docs/ARCHITECTURE.md for the full
reasoning.
"""

import razorpay

from config import require_razorpay_keys


def get_client() -> razorpay.Client:
    key_id, key_secret = require_razorpay_keys()
    client = razorpay.Client(auth=(key_id, key_secret))
    return client


# Razorpay's hard limit on the `receipt` field. The Live Agent's Merchant ID
# text input has no length limit of its own, so a merchant ID longer than
# ~31 characters (entirely realistic -- a human easily types more than
# that) previously produced a receipt over 40 characters, and Razorpay's
# API rejected the ENTIRE order.create call with a BadRequestError. That
# turned a cosmetic display concern for one field into a hard failure of
# the whole "Run investigation" flow, on every single click, for any such
# merchant ID.
RECEIPT_MAX_LENGTH = 40


def build_receipt(merchant_id: str) -> str:
    """Builds the `receipt` value for order.create, truncated to Razorpay's limit."""
    return f"risklens-{merchant_id}"[:RECEIPT_MAX_LENGTH]


def create_test_order(amount_rupees: float, merchant_id: str, receipt: str = None) -> dict:
    """
    Creates a real Order against Razorpay's test-mode API. Returns the raw
    API response (a dict) -- includes a genuine Razorpay order id, amount
    (in paise), currency, status, and created_at timestamp.
    """
    client = get_client()
    amount_paise = int(round(amount_rupees * 100))
    order = client.order.create(
        {
            "amount": amount_paise,
            "currency": "INR",
            "receipt": (receipt or build_receipt(merchant_id))[:RECEIPT_MAX_LENGTH],
            "notes": {"merchant_id": merchant_id, "source": "RiskLens demo"},
        }
    )
    return order


def fetch_recent_orders(count: int = 10) -> list:
    """Returns the most recent real Orders from this account's test mode."""
    client = get_client()
    response = client.order.all({"count": count})
    return response.get("items", [])


def order_to_transaction_fields(order: dict) -> dict:
    """
    Map a real Razorpay Order response onto the transaction-level fields
    RiskLens's feature pipeline understands (see features.RAW_REQUIRED_COLUMNS).
    Only the fields Razorpay's API can actually give us are set here; the
    rest (merchant history) come from agent.tools.get_merchant_context.
    """
    return {
        "razorpay_order_id": order.get("id"),
        "daily_txn_volume": order.get("amount", 0) / 100.0,  # paise -> rupees
        "currency": order.get("currency"),
        "order_status": order.get("status"),
        "created_at_epoch": order.get("created_at"),
        "merchant_id": (order.get("notes") or {}).get("merchant_id"),
    }
