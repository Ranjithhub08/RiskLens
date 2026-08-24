"""
Tests for the agent reasoning loop. These use a FAKE Groq client (a scripted
sequence of responses) rather than calling the real Groq API -- so the test
suite runs offline, doesn't need a real API key, and doesn't cost anyone
money or depend on network access. The point being tested isn't "does Groq
work" (that's Groq's own test suite's job); it's "does our loop correctly
call tools, and does the deterministic gate correctly override the agent
when they disagree."
"""

import json
import os
from types import SimpleNamespace

import pytest

from agent.risk_agent import run_risk_agent
from agent.tools import RiskAgentTools
from audit.audit_log import get_connection
from explainability.explain import RiskExplainer
from pipeline import load_model

MODEL_PATH = "model/artifacts/xgb_model.joblib"


def _tool_call(call_id, name, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _tool_message(tool_calls):
    return SimpleNamespace(content=None, tool_calls=tool_calls)


def _final_message(content: str):
    return SimpleNamespace(content=content, tool_calls=None)


def _response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ScriptedGroqClient:
    """Returns a pre-scripted sequence of responses, one per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model artifact not found -- run model/train.py first")
    return load_model()


@pytest.fixture(scope="module")
def explainer(model):
    return RiskExplainer(model)


@pytest.fixture
def tools(model, explainer, tmp_path):
    conn = get_connection(str(tmp_path / "test_audit.db"))
    yield RiskAgentTools(model, explainer, conn)
    conn.close()


MERCHANT_ARGS = {
    "merchant_id": "AGT1",
    "account_age_days": 900,
    "kyc_status": "complete",
    "business_category": "services",
    "daily_txn_volume": 9000,
    "avg_30d_txn_volume": 9000,
    "total_txns_30d": 500,
    "chargebacks_30d": 0,
    "refunds_30d": 2,
    "avg_ticket_size": 18,
}


def test_agent_calls_tools_in_sequence_and_returns_gated_decision(tools):
    scripted = ScriptedGroqClient(
        [
            _response(_tool_message([_tool_call("1", "get_merchant_context", {"merchant_id": "AGT1"})])),
            _response(_tool_message([_tool_call("2", "score_transaction_risk", MERCHANT_ARGS)])),
            _response(_tool_message([_tool_call("3", "explain_transaction_risk", MERCHANT_ARGS)])),
            _response(_final_message(json.dumps({"recommended_decision": "clear", "reasoning": "Low risk."}))),
        ]
    )
    result = run_risk_agent({"merchant_id": "AGT1", "daily_txn_volume": 9000}, tools, groq_client=scripted)

    assert result["risk_score"] is not None
    assert len(result["trace"]) == 3
    assert result["trace"][0]["tool"] == "get_merchant_context"
    assert result["trace"][1]["tool"] == "score_transaction_risk"
    assert result["gated_decision"] in {"clear", "escalate", "flag_for_compliance_review", "needs_manual_review"}


def test_gate_overrides_agent_when_they_disagree(tools):
    """
    The agent claims "clear" no matter what -- but we feed it inputs that
    produce a high risk score. The gate must win: the final gated_decision
    should NOT be "clear" just because the agent said so.
    """
    high_risk_args = dict(MERCHANT_ARGS, kyc_status="incomplete", chargebacks_30d=12, avg_30d_txn_volume=2000)
    scripted = ScriptedGroqClient(
        [
            _response(_tool_message([_tool_call("1", "score_transaction_risk", high_risk_args)])),
            _response(_tool_message([_tool_call("2", "explain_transaction_risk", high_risk_args)])),
            # Agent insists everything is fine regardless of the score it just saw.
            _response(_final_message(json.dumps({"recommended_decision": "clear", "reasoning": "Looks fine to me."}))),
        ]
    )
    result = run_risk_agent({"merchant_id": "AGT2", "daily_txn_volume": 9000}, tools, groq_client=scripted)

    assert result["agent_proposal"]["recommended_decision"] == "clear"
    # The deterministic gate is computed independently from the actual risk score,
    # so if the score is genuinely high, the gate must not just parrot the agent.
    if result["risk_score"] is not None and result["risk_score"] > 0.75:
        assert result["gated_decision"] != "clear"
        assert result["agent_and_gate_agree"] is False


def test_malformed_final_response_does_not_crash(tools):
    scripted = ScriptedGroqClient(
        [
            _response(_tool_message([_tool_call("1", "score_transaction_risk", MERCHANT_ARGS)])),
            _response(_final_message("this is not json at all")),
        ]
    )
    result = run_risk_agent({"merchant_id": "AGT3", "daily_txn_volume": 9000}, tools, groq_client=scripted)

    assert result["agent_proposal"] is None
    assert result["agent_and_gate_agree"] is False
    # Gate still produces a valid decision from the risk score that WAS computed.
    assert result["gated_decision"] in {"clear", "escalate", "flag_for_compliance_review", "needs_manual_review"}


def test_running_out_of_turns_fails_safe_to_manual_review(tools):
    # Every single call is a tool call, never a final answer -- simulates an
    # agent stuck in a loop.
    endless_tool_call = _response(
        _tool_message([_tool_call("x", "get_merchant_context", {"merchant_id": "AGT4"})])
    )
    scripted = ScriptedGroqClient([endless_tool_call] * 10)
    result = run_risk_agent({"merchant_id": "AGT4", "daily_txn_volume": 9000}, tools, groq_client=scripted)

    assert result["risk_score"] is None
    assert result["gated_decision"] == "needs_manual_review"


def test_tool_error_is_captured_not_raised(tools):
    # score_transaction_risk called with missing required fields -- transform_features
    # will raise; the loop must catch it and continue rather than crash.
    scripted = ScriptedGroqClient(
        [
            _response(_tool_message([_tool_call("1", "score_transaction_risk", {"merchant_id": "AGT5"})])),
            _response(_final_message(json.dumps({"recommended_decision": "escalate", "reasoning": "Incomplete data."}))),
        ]
    )
    result = run_risk_agent({"merchant_id": "AGT5", "daily_txn_volume": 9000}, tools, groq_client=scripted)

    assert "error" in result["trace"][0]["result"]
    assert result["risk_score"] is None
    assert result["gated_decision"] == "needs_manual_review"
