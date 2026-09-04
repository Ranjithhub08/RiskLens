"""
Tests for data/raw/generate_data.py's synthetic dataset generation.

This file previously had no coverage at all -- which is exactly how a real
bug went unnoticed for six rounds of code review: the label-generation
logic explicitly treats a volume_change_pct above 0.8 as a strong risk
signal (found live, via a real demo run: a genuine 20x volume spike scored
as slightly RISK-REDUCING instead of raising the score), but the
`daily_txn_volume` generator drew EVERY merchant's daily/average ratio from
one narrow Normal(1.0, 0.15) distribution whose tail never actually reaches
that threshold across a realistic sample size -- so the >0.8 branch fired
zero times during training, and the model never saw a single genuine
volume-spike example to learn from. These tests assert the generator
actually produces the risk signal the label rule depends on, so that gap
can't silently reopen.
"""

import pandas as pd

from data.raw.generate_data import generate


def test_generated_data_contains_genuine_volume_spikes():
    df = generate(n_merchants=6000, seed=42)
    volume_change_pct = (df["daily_txn_volume"] - df["avg_30d_txn_volume"]) / df["avg_30d_txn_volume"]

    # The label rule (risk_score_latent) treats volume_change_pct > 0.8 as a
    # real risk signal -- if this fires zero times, that branch is dead
    # code and the model has nothing to learn from it, however clearly the
    # label logic intends it to matter.
    spiking = (volume_change_pct > 0.8).sum()
    assert spiking > 0, (
        "No generated merchant has a volume_change_pct above 0.8 -- the "
        "label rule's volume-spike risk signal would be dead code, and a "
        "retrained model would have zero real spike examples to learn from."
    )
    # Not a token handful either -- a meaningful, learnable subset.
    assert spiking / len(df) > 0.02


def test_spiking_merchants_are_meaningfully_riskier_than_non_spiking_ones():
    # Regression test: it's not enough for a spike to merely be POSSIBLE --
    # it must actually correlate with the is_risky label the label rule
    # says it should, or a model trained on it still has nothing real to
    # learn (e.g. if the spiking subset happened to be generated
    # independently of is_risky by a wiring mistake).
    df = generate(n_merchants=6000, seed=42)
    volume_change_pct = (df["daily_txn_volume"] - df["avg_30d_txn_volume"]) / df["avg_30d_txn_volume"]

    spiking_risky_rate = df.loc[volume_change_pct > 0.8, "is_risky"].mean()
    non_spiking_risky_rate = df.loc[volume_change_pct <= 0.8, "is_risky"].mean()

    assert spiking_risky_rate > non_spiking_risky_rate * 2


def test_generate_is_deterministic_for_a_fixed_seed():
    # Reproducibility matters for a demo -- the same seed must always
    # produce the same dataset, not just the same shape.
    df1 = generate(n_merchants=500, seed=7)
    df2 = generate(n_merchants=500, seed=7)
    pd.testing.assert_frame_equal(df1, df2)
