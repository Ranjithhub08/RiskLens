"""
Optional API layer -- wraps the same pipeline used by the dashboard behind
a `/score` endpoint, so the project can be described (and demoed) as
something that could plug into a real service rather than only existing as
a Streamlit app. Not required by the buildathon; included because it costs
little and strengthens the "how would this actually integrate" story in a
panel interview.

Run:
    uvicorn api.main:app --reload --port 8000

Then:
    curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{
        "merchant_id": "M123",
        "account_age_days": 40,
        "kyc_status": "incomplete",
        "business_category": "electronics",
        "daily_txn_volume": 50000,
        "avg_30d_txn_volume": 8000,
        "total_txns_30d": 100,
        "chargebacks_30d": 6,
        "refunds_30d": 12,
        "avg_ticket_size": 80
    }'
"""

import os
import sys
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit.audit_log import get_connection
from explainability.explain import RiskExplainer
from pipeline import load_model, score_record

app = FastAPI(title="RiskLens API", description="Explainable merchant risk scoring")

_model = None
_explainer = None


class MerchantRecord(BaseModel):
    merchant_id: Optional[str] = None
    account_age_days: Optional[float] = None
    kyc_status: Optional[str] = None
    business_category: Optional[str] = None
    daily_txn_volume: Optional[float] = None
    avg_30d_txn_volume: Optional[float] = None
    total_txns_30d: Optional[float] = None
    chargebacks_30d: Optional[float] = None
    refunds_30d: Optional[float] = None
    avg_ticket_size: Optional[float] = None


@app.on_event("startup")
def load_resources():
    global _model, _explainer
    _model = load_model()
    _explainer = RiskExplainer(_model)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/score")
def score(record: MerchantRecord):
    # A fresh connection per request, used only by the thread handling that
    # request and closed when it's done -- NOT a single connection object
    # shared across every request. FastAPI dispatches this sync `def`
    # handler to a thread pool, so concurrent /score calls really do run on
    # different OS threads; a single sqlite3.Connection reused across all of
    # them (the previous behavior, via a startup-time global) let two
    # concurrent writers interleave INSERT/commit calls on the very same
    # connection object. Reproduced under load: intermittent
    # "cannot start a transaction within a transaction" / "cannot commit --
    # no transaction is active" errors, and worse, HTTP 200 responses whose
    # audit row silently never made it into audit_events at all -- a real
    # hole in "every scoring event is written here, append-only" under
    # nothing more exotic than ordinary concurrent traffic. SQLite itself
    # safely supports multiple connections to the same file as long as each
    # is only ever touched by one thread, which get_connection()'s per-call
    # schema check/migration is cheap enough to make the right default here.
    conn = get_connection()
    try:
        result = score_record(record.model_dump(), _model, _explainer, conn)
    finally:
        conn.close()
    return result
