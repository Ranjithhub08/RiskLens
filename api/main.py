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
_conn = None


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
    global _model, _explainer, _conn
    _model = load_model()
    _explainer = RiskExplainer(_model)
    _conn = get_connection()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/score")
def score(record: MerchantRecord):
    result = score_record(record.model_dump(), _model, _explainer, _conn)
    return result
