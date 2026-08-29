"""
Tests for integrations/razorpay_client.py's receipt construction.

Razorpay's order.create API hard-rejects any `receipt` value longer than 40
characters with a BadRequestError -- and the Live Agent's Merchant ID text
input has no length limit of its own, so this is easily reachable by typing
a realistic (if long) merchant ID.
"""

from integrations.razorpay_client import RECEIPT_MAX_LENGTH, build_receipt


def test_build_receipt_for_a_short_merchant_id():
    assert build_receipt("merchant_1001") == "risklens-merchant_1001"


def test_build_receipt_truncates_a_long_merchant_id_to_the_razorpay_limit():
    long_merchant_id = "m" * 60
    receipt = build_receipt(long_merchant_id)
    assert len(receipt) == RECEIPT_MAX_LENGTH
    assert receipt.startswith("risklens-")


def test_build_receipt_never_exceeds_the_razorpay_limit_regardless_of_input_length():
    for length in (0, 1, 30, 31, 32, 40, 100, 1000):
        receipt = build_receipt("m" * length)
        assert len(receipt) <= RECEIPT_MAX_LENGTH
