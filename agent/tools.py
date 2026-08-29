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
from audit.audit_log import get_all_events, get_events_for_merchant
from explainability.explain import RiskExplainer
from features.features import find_missing_or_invalid, transform_features

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
    {
        "type": "function",
        "function": {
            "name": "get_similar_past_cases",
            "description": (
                "Look up how RiskLens has decided past cases for OTHER merchants in the "
                "same business category, so you can compare this case against precedent "
                "instead of judging it in isolation. This complements get_recent_audit_history, "
                "which only looks at THIS merchant's own past decisions -- use this one to see "
                "how similar merchants elsewhere were treated. Optional but recommended before "
                "you submit your decision."
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
            "name": "submit_decision",
            "description": (
                "Call this exactly once, as your last action, after you have called "
                "score_transaction_risk and explain_transaction_risk, to submit your final "
                "recommendation. This does not take any action by itself -- a separate fixed "
                "safety system independently re-checks the risk score before anything is final."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recommended_decision": {
                        "type": "string",
                        "enum": ["clear", "escalate", "flag_for_compliance_review"],
                    },
                    "reasoning": {"type": "string", "description": "Your reasoning, in plain language."},
                },
                "required": ["recommended_decision", "reasoning"],
            },
        },
    },
]

# Not dispatched through RiskAgentTools.call() like the investigative tools above --
# the agent loop in risk_agent.py intercepts a call to this name directly, since it's
# how the model reports its own answer rather than something for RiskLens to compute.
FINAL_ANSWER_TOOL = "submit_decision"


class RiskAgentTools:
    """Binds the tool functions to a specific model/explainer/audit connection."""

    def __init__(self, model, explainer: RiskExplainer, conn):
        self.model = model
        self.explainer = explainer
        self.conn = conn

    def get_merchant_context(self, merchant_id: str) -> dict:
        return _get_merchant_context(merchant_id)

    def _row_from_args(self, args: dict) -> pd.DataFrame:
        # dtype=object: pandas' default type-inference on a plain dict can
        # raise OverflowError itself (e.g. a hallucinated 300+-digit tool
        # argument) before _validated_row_from_args's find_missing_or_invalid
        # call ever gets a chance to reject it safely -- see pipeline.py's
        # score_record, which has the identical guard for the same reason.
        return pd.DataFrame([{k: v for k, v in args.items() if k != "merchant_id"}], dtype=object)

    def _validated_row_from_args(self, args: dict):
        """Returns (X, error_dict) -- exactly one of the two is not None.

        score_transaction_risk/explain_transaction_risk's tool schemas
        declare kyc_status as a fixed enum, but that's a hint to the model,
        not a server-side guarantee every provider enforces, and
        business_category isn't even constrained to an enum there at all --
        it's a plain string. Without this check, a hallucinated or
        mistyped category (e.g. "retail", which isn't one of
        features.BUSINESS_CATEGORIES) would reach transform_features
        directly, which one-hot-encodes anything it doesn't recognize as
        all-zero rather than raising -- silently scoring as if the merchant
        had no business category at all, and returning a normal-looking
        risk_score with no error, instead of the same fail-safe
        needs_manual_review outcome the exact same record would get through
        pipeline.py's score_record (which does run this check). Reusing
        find_missing_or_invalid here closes that gap instead of leaving the
        agent's own tools as a second, less-validated path to the model.
        """
        row = self._row_from_args(args)
        problems = find_missing_or_invalid(row)
        if problems:
            return None, {"error": f"Cannot score: missing or invalid field(s): {problems}."}
        return transform_features(row), None

    def score_transaction_risk(self, **args) -> dict:
        X, error = self._validated_row_from_args(args)
        if error:
            return error
        score = float(self.model.predict_proba(X)[:, 1][0])
        return {"risk_score": score}

    def explain_transaction_risk(self, **args) -> dict:
        X, error = self._validated_row_from_args(args)
        if error:
            return error
        return self.explainer.explain_row(X)

    def get_recent_audit_history(self, merchant_id: str) -> dict:
        events = get_events_for_merchant(self.conn, merchant_id, limit=5)
        return {
            "past_decisions": [
                {"timestamp": e["timestamp_utc"], "decision": e["decision"], "risk_score": e["risk_score"]}
                for e in events
            ]
        }

    def get_similar_past_cases(self, merchant_id: str, limit: int = 3) -> dict:
        """
        Cross-merchant precedent, not this merchant's own history (that's
        get_recent_audit_history). Ranked by: same business category first
        (merchant_context.py derives a category deterministically from any
        merchant_id, so this works even for merchants seen only once before),
        then most recent. Deliberately excludes this merchant's own past
        events so the agent is comparing against *other* cases, not itself.
        """
        current_category = _get_merchant_context(merchant_id)["business_category"]
        events = get_all_events(self.conn, limit=500)  # already ordered most-recent-first

        same_category, other_category = [], []
        for event in events:
            past_merchant = event.get("merchant_id")
            if not past_merchant or str(past_merchant) == str(merchant_id):
                continue
            if event.get("risk_score") is None:
                continue
            past_category = _get_merchant_context(past_merchant)["business_category"]
            entry = {
                "merchant_id": past_merchant,
                "business_category": past_category,
                "risk_score": round(event["risk_score"], 4),
                "decision": event.get("decision"),
                "explanation": (event.get("explanation") or "")[:160],
                "timestamp": event.get("timestamp_utc"),
            }
            (same_category if past_category == current_category else other_category).append(entry)

        picked = (same_category + other_category)[:limit]
        return {
            "current_business_category": current_category,
            "similar_cases": picked,
            "note": None if picked else "No past cases exist yet for any other merchant.",
        }

    def call(self, name: str, arguments: dict):
        dispatch = {
            "get_merchant_context": lambda a: self.get_merchant_context(**a),
            "score_transaction_risk": lambda a: self.score_transaction_risk(**a),
            "explain_transaction_risk": lambda a: self.explain_transaction_risk(**a),
            "get_recent_audit_history": lambda a: self.get_recent_audit_history(**a),
            "get_similar_past_cases": lambda a: self.get_similar_past_cases(**a),
        }
        if name not in dispatch:
            raise ValueError(f"Unknown tool: {name}")
        return dispatch[name](arguments)
