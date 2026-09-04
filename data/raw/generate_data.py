"""
Synthetic merchant risk dataset generator.

We don't have access to Razorpay's real transaction/KYC data, so this script
generates a realistic stand-in: one row per merchant snapshot, with the kind
of raw signals a real risk system would see (volume, chargebacks, refunds,
KYC status, account age), plus a label ("is_risky") produced by a rule +
noise process so the dataset has genuine, learnable signal without being
trivially perfect.

Run:
    python3 data/raw/generate_data.py
Produces:
    data/raw/merchant_snapshots.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
N_MERCHANTS = 6000
START_DATE = datetime(2026, 1, 1)
DATE_SPREAD_DAYS = 180  # snapshots spread across ~6 months, for a time-based split later

BUSINESS_CATEGORIES = ["electronics", "fashion", "services", "travel", "subscriptions", "other"]
# Rough relative "riskiness" weight per category, used only to shape realistic label noise
CATEGORY_RISK_WEIGHT = {
    "electronics": 0.15,
    "fashion": 0.05,
    "services": 0.03,
    "travel": 0.10,
    "subscriptions": 0.08,
    "other": 0.05,
}


def generate(n_merchants: int = N_MERCHANTS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    merchant_id = np.arange(1, n_merchants + 1)
    account_age_days = rng.integers(5, 2000, size=n_merchants)
    kyc_status = rng.choice(["complete", "incomplete"], size=n_merchants, p=[0.85, 0.15])
    business_category = rng.choice(BUSINESS_CATEGORIES, size=n_merchants)

    # Baseline daily volume varies a lot by merchant size (log-normal is realistic for money amounts)
    avg_30d_txn_volume = rng.lognormal(mean=9.5, sigma=1.1, size=n_merchants)  # ~ INR thousands to lakhs
    total_txns_30d = rng.integers(5, 3000, size=n_merchants)

    # Most merchants have volume close to their own average; a genuine risky
    # subset spikes hard -- this comment described the intent from the
    # start, but the code below it didn't implement it: EVERY merchant's
    # multiplier used to be drawn from one narrow Normal(1.0, 0.15)
    # distribution, whose tail practically never reaches 1.8x across 6000
    # draws (confirmed empirically: the generated dataset's
    # volume_change_pct topped out at 0.499 -- never once crossing the 0.8
    # threshold the label rule below treats as a real risk signal). That
    # meant `3.0 * (volume_change_pct > 0.8)` in risk_score_latent fired
    # ZERO times during training: the model never saw a single genuine
    # volume-spike example and had no way to learn that a spike matters, no
    # matter how clearly the label-generation logic intended it to (found
    # live: a real 20x demo spike scored as slightly RISK-REDUCING instead
    # of raising the score, because it fell in a region of feature space
    # the model was never trained on). A separate, independently-drawn
    # spiking subset now actually produces daily volumes 1.8x-6x a
    # merchant's own average, so the >0.8 branch has real, learnable
    # examples on both sides of it.
    is_spiking = rng.random(n_merchants) < 0.08
    normal_multiplier = rng.normal(loc=1.0, scale=0.15, size=n_merchants)
    spike_multiplier = rng.uniform(1.8, 6.0, size=n_merchants)
    volume_multiplier = np.where(is_spiking, spike_multiplier, normal_multiplier)
    daily_txn_volume = avg_30d_txn_volume * np.clip(volume_multiplier, 0.3, None)

    chargebacks_30d = rng.poisson(lam=total_txns_30d * 0.004)
    refunds_30d = rng.poisson(lam=total_txns_30d * 0.02)
    avg_ticket_size = daily_txn_volume / np.maximum(total_txns_30d / 30, 1)

    snapshot_offsets = rng.integers(0, DATE_SPREAD_DAYS, size=n_merchants)
    snapshot_date = [START_DATE + timedelta(days=int(d)) for d in snapshot_offsets]

    df = pd.DataFrame(
        {
            "merchant_id": merchant_id,
            "snapshot_date": snapshot_date,
            "account_age_days": account_age_days,
            "kyc_status": kyc_status,
            "business_category": business_category,
            "daily_txn_volume": daily_txn_volume,
            "avg_30d_txn_volume": avg_30d_txn_volume,
            "total_txns_30d": total_txns_30d,
            "chargebacks_30d": chargebacks_30d,
            "refunds_30d": refunds_30d,
            "avg_ticket_size": avg_ticket_size,
        }
    )

    # --- Label generation: rule-based ground truth + noise -----------------
    volume_change_pct = (df["daily_txn_volume"] - df["avg_30d_txn_volume"]) / df["avg_30d_txn_volume"]
    chargeback_rate = df["chargebacks_30d"] / df["total_txns_30d"].clip(lower=1)
    refund_rate = df["refunds_30d"] / df["total_txns_30d"].clip(lower=1)
    category_weight = df["business_category"].map(CATEGORY_RISK_WEIGHT)

    risk_score_latent = (
        3.0 * (volume_change_pct > 0.8).astype(float)
        + 2.5 * (chargeback_rate > 0.02).astype(float)
        + 1.5 * (refund_rate > 0.08).astype(float)
        + 2.0 * (df["kyc_status"] == "incomplete").astype(float)
        + 1.0 * (df["account_age_days"] < 60).astype(float)
        + rng.normal(0, 0.5, size=n_merchants)
        + category_weight
    )

    # Convert latent score to a probability via logistic function, then sample the label.
    # This keeps the dataset "noisy" and realistic instead of a deterministic rule a model
    # could memorize perfectly.
    prob_risky = 1 / (1 + np.exp(-(risk_score_latent - 3.0)))
    is_risky = rng.binomial(1, prob_risky)

    df["is_risky"] = is_risky
    df = df.sort_values("snapshot_date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "data/raw/merchant_snapshots.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Positive rate (is_risky=1): {df['is_risky'].mean():.3f}")
