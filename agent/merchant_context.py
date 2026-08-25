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
import math

from features.features import BUSINESS_CATEGORIES

# Same per-transaction rates data/raw/generate_data.py uses to derive
# chargebacks_30d/refunds_30d from total_txns_30d -- see _poisson's docstring
# for why this module needs to match that generative assumption, not just
# its rough scale.
_CHARGEBACK_RATE = 0.004
_REFUND_RATE = 0.02


def _seeded_rng(merchant_id: str):
    seed = int(hashlib.sha256(str(merchant_id).encode()).hexdigest(), 16) % (2**32)
    import random

    rng = random.Random(seed)
    return rng


def _poisson(rng, lam: float) -> int:
    """Knuth's algorithm: an exact Poisson(lam) draw using only rng.random().

    Used instead of numpy's rng.poisson so this module doesn't have to pull
    in numpy just for one distribution, while still sampling from the same
    distribution family data/raw/generate_data.py's own chargebacks_30d/
    refunds_30d columns come from (see that file's rng.poisson(lam=...)
    calls) -- see get_merchant_context's docstring for why matching the
    distribution, not just picking numbers that "look" similarly sized,
    is what actually matters here.
    """
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= l:
            return k - 1


def get_merchant_context(merchant_id: str) -> dict:
    """Deterministic, simulated merchant profile for a given merchant_id."""
    rng = _seeded_rng(merchant_id)
    account_age_days = rng.randint(5, 2000)
    kyc_status = rng.choices(["complete", "incomplete"], weights=[0.85, 0.15])[0]
    business_category = rng.choice(BUSINESS_CATEGORIES)
    avg_30d_txn_volume = round(rng.lognormvariate(9.5, 1.1), 2)
    total_txns_30d = rng.randint(5, 3000)
    # chargebacks_30d/refunds_30d must scale with total_txns_30d, not be
    # drawn independently of it -- data/raw/generate_data.py's training data
    # derives them exactly this way (Poisson, rate proportional to volume),
    # which keeps chargeback_rate/refund_rate realistic (well under 30% in
    # the actual training data -- see transform_features/chargeback_rate).
    # An earlier version here drew chargebacks_30d ~ randint(0, 15) and
    # refunds_30d ~ randint(0, 60) with no relation to total_txns_30d at
    # all: for a merchant with few transactions that could (and, sampled
    # across enough merchant IDs, regularly did -- roughly 1 in 100) put
    # refunds_30d or chargebacks_30d ABOVE total_txns_30d -- a refund_rate
    # over 100%, a value that never occurs anywhere in training and that
    # the model has no reliable basis to score. That's the exact feature
    # this demo's "Try a different simulated merchant" button invites a
    # live audience to sample at random, so it needed to actually match the
    # distribution the model was trained on, not just resemble it.
    chargebacks_30d = _poisson(rng, total_txns_30d * _CHARGEBACK_RATE)
    refunds_30d = _poisson(rng, total_txns_30d * _REFUND_RATE)
    avg_ticket_size = round(rng.uniform(50, 5000), 2)
    return {
        "merchant_id": merchant_id,
        "account_age_days": account_age_days,
        "kyc_status": kyc_status,
        "business_category": business_category,
        "avg_30d_txn_volume": avg_30d_txn_volume,
        "total_txns_30d": total_txns_30d,
        "chargebacks_30d": chargebacks_30d,
        "refunds_30d": refunds_30d,
        "avg_ticket_size": avg_ticket_size,
    }
