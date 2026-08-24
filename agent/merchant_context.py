"""
Simulated merchant history.

Razorpay's API (test mode or otherwise) has no endpoint that hands a
student project a merchant's KYC status or 30-day chargeback rate -- that
data is internal and sensitive for good reason. So this module deterministically
derives a plausible merchant profile from the merchant_id (same id always
produces the same profile, via a seeded hash -- not random each call), which
plugs into the same feature schema real Razorpay data would use.

This is the one part of the pipeline that is unambiguously simulated, and
it's kept in its own file specifically so that's easy to point to and be
transparent about -- see docs/ARCHITECTURE.md's Limitations section.
"""

import hashlib

from features.features import BUSINESS_CATEGORIES


def _seeded_rng(merchant_id: str):
    seed = int(hashlib.sha256(str(merchant_id).encode()).hexdigest(), 16) % (2**32)
    import random

    rng = random.Random(seed)
    return rng


def get_merchant_context(merchant_id: str) -> dict:
    """Deterministic, simulated merchant profile for a given merchant_id."""
    rng = _seeded_rng(merchant_id)
    return {
        "merchant_id": merchant_id,
        "account_age_days": rng.randint(5, 2000),
        "kyc_status": rng.choices(["complete", "incomplete"], weights=[0.85, 0.15])[0],
        "business_category": rng.choice(BUSINESS_CATEGORIES),
        "avg_30d_txn_volume": round(rng.lognormvariate(9.5, 1.1), 2),
        "total_txns_30d": rng.randint(5, 3000),
        "chargebacks_30d": rng.randint(0, 15),
        "refunds_30d": rng.randint(0, 60),
        "avg_ticket_size": round(rng.uniform(50, 5000), 2),
    }
