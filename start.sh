#!/usr/bin/env bash
# One-command startup for RiskLens.
#
# Does whatever hasn't been done yet (install deps, generate the synthetic
# dataset, train the model) and then launches the dashboard -- safe to
# re-run any time, since each step is skipped if its output already exists.
#
# Usage:
#   ./start.sh

set -e

cd "$(dirname "$0")"

# Always run this (not just when streamlit is missing entirely): pip still
# checks the version constraints in requirements.txt and upgrades anything
# that's present but too old, which a plain "is it importable" check would
# silently skip -- exactly how an outdated pre-installed streamlit (missing
# the `width=` argument dashboard.py relies on) got past this script once
# already. Fast and a no-op when everything already satisfies the pins.
echo "==> Checking dependencies..."
pip install -q -r requirements.txt

if [ ! -f "data/raw/merchant_snapshots.csv" ]; then
    echo "==> Generating synthetic dataset (first run only)..."
    python3 data/raw/generate_data.py
fi

if [ ! -f "model/artifacts/xgb_model.joblib" ]; then
    echo "==> Training model (first run only)..."
    python3 model/train.py
fi

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "==> No .env found -- copying .env.example."
    echo "    Overview / Investigations / Batch Scoring / Models / Audit Trail all work with no keys."
    echo "    Fill in GROQ_API_KEY and your Razorpay TEST-mode keys in .env to use the Live Agent tab."
    cp .env.example .env
fi

echo "==> Launching dashboard at http://localhost:8501"
streamlit run app/dashboard.py
