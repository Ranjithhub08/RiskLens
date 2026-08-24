"""
Tools the risk agent is allowed to call.

Each tool wraps a piece of RiskLens that was already built and tested
before the agent existed (the model, the explainer, the audit log) -- the
agent doesn't reimplement any of that logic, it just decides *when* to call
it and reasons over the results. This is deliberate: the agent's job is
orchestration and judgment, not scoring math.
"""

import pandas as pd

from agent.merchant_context import get_merchant_context as _get_merchant_context
from audit.audit_log import get_events_for_merchant
from explainability.explain import RiskExplainer
from features.features import transform_features

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_merchant_context",
            "description": (
                "Look up this merchant's history: account age, KYC status, business "
                "category, and 30-day transaction/chargeback/refund stats. Call this "
                "first for any merchant you haven't already looked up in this conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {"merchant_id": {"type": "string"}},
                "required": ["merchant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_transaction_risk",
            "description": (
                "Run the trained risk model on a transaction plus merchant context and "
                "get back a risk probability (0-1). Call get_merchant_context first so "
                "you have the merchant fields to pass in."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant_id": {"type": "string"},
                    "account_age_days": {"type": "number"},
                    "kyc_status": {"type": "string", "enum": ["complete", "incomplete"]},
                    "business_category": {"type": "string"},
                    "daily_txn_volume": {"type": "number"},
                    "avg_30d_txn_volume": {"type": "number"},
                    "total_txns_30d": {"type": "number"},
                    "chargebacks_30d": {"type": "number"},
                    "refunds_30d": {"type": "number"},
                    "avg_ticket_size": {"type": "number"},
                },
                "required": [
                    "merchant_id", "account_age_days", "kyc_status", "business_category",
                    "daily_txn_volume", "avg_30d_txn_volume", "total_txns_30d",
                    "chargebacks_30d", "refunds_30d", "avg_ticket_size",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_transaction_risk",
            "description": (
                "Get a plain-language explanation of which factors drove a risk score. "
                "Takes the same fields as score_transaction_risk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant_id": {"type": "string"},
                    "account_age_days": {"type": "number"},
                    "kyc_status": {"type": "string", "enum": ["complete", "incomplete"]},
                    "business_category": {"type": "string"},
                    "daily_txn_volume": {"type": "number"},
                    "avg_30d_txn_volume": {"type": "number"},
                    "total_txns_30d": {"type": "number"},
                    "chargebacks_30d": {"type": "number"},
                    "refunds_30d": {"type": "number"},
                    "avg_ticket_size": {"type": "number"},
                },
                "required": [
                    "merchant_id", "account_age_days", "kyc_status", "business_category",
                    "daily_txn_volume", "avg_30d_txn_volume", "total_txns_30d",
                    "chargebacks_30d", "refunds_30d", "avg_ticket_size",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_audit_history",
            "description": "Look up this merchant's past RiskLens decisions, if any, for context.",
            "parameters": {
                "type": "object",
                "properties": {"merchant_id": {"type": "string"}},
                "required": ["merchant_id"],
            },
        },
    },
]


class RiskAgentTools:
    """Binds the tool functions to a specific model/explainer/audit connection."""

    def __init__(self, model, explainer: RiskExplainer, conn):
        self.model = model
        self.explainer = explainer
        self.conn = conn

    def get_merchant_context(self, merchant_id: str) -> dict:
        return _get_merchant_context(merchant_id)

    def _row_from_args(self, args: dict) -> pd.DataFrame:
        return pd.DataFrame([{k: v for k, v in args.items() if k != "merchant_id"}])

    def score_transaction_risk(self, **args) -> dict:
        X = transform_features(self._row_from_args(args))
        score = float(self.model.predict_proba(X)[:, 1][0])
        return {"risk_score": score}

    def explain_transaction_risk(self, **args) -> dict:
        X = transform_features(self._row_from_args(args))
        return self.explainer.explain_row(X)

    def get_recent_audit_history(self, merchant_id: str) -> dict:
        events = get_events_for_merchant(self.conn, merchant_id, limit=5)
        return {
            "past_decisions": [
                {"timestamp": e["timestamp_utc"], "decision": e["decision"], "risk_score": e["risk_score"]}
                for e in events
            ]
        }

    def call(self, name: str, arguments: dict):
        dispatch = {
            "get_merchant_context": lambda a: self.get_merchant_context(**a),
            "score_transaction_risk": lambda a: self.score_transaction_risk(**a),
            "explain_transaction_risk": lambda a: self.explain_transaction_risk(**a),
            "get_recent_audit_history": lambda a: self.get_recent_audit_history(**a),
        }
        if name not in dispatch:
            raise ValueError(f"Unknown tool: {name}")
        return dispatch[name](arguments)
