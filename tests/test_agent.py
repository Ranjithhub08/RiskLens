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

from agent.merchant_context import get_merchant_context
from agent.risk_agent import _finalize, run_risk_agent
from agent.tools import RiskAgentTools
from audit.audit_log import get_connection, log_event
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


class FlakyGroqClient:
    """Like ScriptedGroqClient, but the given `fail_on_call` (1-indexed) call
    raises instead of returning a scripted response -- simulates a
    transient LLM-API failure (rate limit, network timeout, ...) partway
    through the loop."""

    def __init__(self, responses, fail_on_call: int):
        self._responses = list(responses)
        self._fail_on_call = fail_on_call
        self.call_count = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.call_count += 1
        if self.call_count == self._fail_on_call:
            raise RuntimeError("simulated transient Groq API failure")
        response = self._responses[self.call_count - 1]
        return response


def test_mid_loop_api_failure_preserves_already_computed_score_instead_of_losing_it(tools):
    # Regression test: turn 1 successfully computes a real risk score and
    # explanation (score_transaction_risk, explain_transaction_risk). Turn
    # 2's LLM API call then fails (a transient rate limit/timeout/malformed
    # response -- a real, common failure mode, not a rare edge case). This
    # used to propagate the raw exception straight out of run_risk_agent,
    # discarding computed_risk_score/explanation/top_factors/trace entirely
    # even though they were already legitimately computed -- in contrast to
    # the "ran out of turns" path (tested above), which deliberately
    # preserves and gates the same kind of partial progress instead of
    # throwing it away.
    scripted_responses = [
        _response(_tool_message([_tool_call("1", "score_transaction_risk", MERCHANT_ARGS)])),
        _response(_tool_message([_tool_call("2", "explain_transaction_risk", MERCHANT_ARGS)])),
    ]
    flaky = FlakyGroqClient(scripted_responses, fail_on_call=3)

    result = run_risk_agent({"merchant_id": "AGT5", "daily_txn_volume": 9000}, tools, groq_client=flaky)

    # Must not raise, and must preserve what turn 1 already computed rather
    # than discarding it.
    assert result["risk_score"] is not None
    assert result["explanation"] is not None
    assert len(result["trace"]) == 2
    assert result["gated_decision"] in {"clear", "escalate", "flag_for_compliance_review", "needs_manual_review"}
    # The gate's real decision (derived from the real score), not a blind
    # "needs_manual_review" that throws away the evidence already gathered.
    assert result["gated_reason"] != "The agent's reasoning could not be completed (a language-model API call failed) -- routed for manual review."


def test_api_failure_before_any_score_still_fails_safe_without_crashing(tools):
    # If the very first call fails (no score ever computed), there's
    # nothing to gate on -- must still fail safe to manual review, not
    # crash, and not claim a score exists.
    flaky = FlakyGroqClient([], fail_on_call=1)

    result = run_risk_agent({"merchant_id": "AGT6", "daily_txn_volume": 9000}, tools, groq_client=flaky)

    assert result["risk_score"] is None
    assert result["gated_decision"] == "needs_manual_review"
    assert result["trace"] == []


def test_final_answer_via_submit_decision_tool_call(tools):
    """
    Groq's real tool-use models -- unlike our scripted fake client in every other
    test here -- tend to submit structured final output as an actual tool call
    rather than plain text, even when asked to reply with bare JSON. Before
    submit_decision existed as a real, declared tool, the model would invent an
    undeclared one (observed in production: a call to a tool literally named
    "json"), which the Groq API rejects outright with a 400 tool-validation
    error and the whole investigation would fail. This test locks in the fix:
    a final answer delivered via submit_decision must parse exactly like one
    delivered as plain-text JSON used to.
    """
    scripted = ScriptedGroqClient(
        [
            _response(_tool_message([_tool_call("1", "get_merchant_context", {"merchant_id": "AGT6"})])),
            _response(_tool_message([_tool_call("2", "score_transaction_risk", MERCHANT_ARGS)])),
            _response(_tool_message([_tool_call("3", "explain_transaction_risk", MERCHANT_ARGS)])),
            _response(
                _tool_message(
                    [_tool_call("4", "submit_decision", {"recommended_decision": "clear", "reasoning": "Low risk."})]
                )
            ),
        ]
    )
    result = run_risk_agent({"merchant_id": "AGT6", "daily_txn_volume": 9000}, tools, groq_client=scripted)

    assert result["risk_score"] is not None
    # Only the three real investigative calls should land in the trace --
    # submit_decision itself isn't a RiskLens tool and shouldn't appear there.
    assert len(result["trace"]) == 3
    assert result["agent_proposal"] == {"recommended_decision": "clear", "reasoning": "Low risk."}
    assert result["gated_decision"] in {"clear", "escalate", "flag_for_compliance_review", "needs_manual_review"}


def test_on_step_callback_fires_live_progress_events(tools):
    """
    The Live Agent page's "watch it think" feed depends on on_step firing in
    real time as the investigation proceeds, not just once at the very end.
    This locks in the event sequence a UI can rely on: one "thinking" event
    per model call, one "tool_call" event per REAL tool dispatched (submit_decision
    itself must not produce one, matching how it's excluded from result["trace"]),
    and exactly one "final" event carrying the agent's proposed decision.
    """
    scripted = ScriptedGroqClient(
        [
            _response(_tool_message([_tool_call("1", "get_merchant_context", {"merchant_id": "AGT7"})])),
            _response(_tool_message([_tool_call("2", "score_transaction_risk", MERCHANT_ARGS)])),
            _response(_tool_message([_tool_call("3", "explain_transaction_risk", MERCHANT_ARGS)])),
            _response(
                _tool_message(
                    [_tool_call("4", "submit_decision", {"recommended_decision": "clear", "reasoning": "Low risk."})]
                )
            ),
        ]
    )
    events = []
    result = run_risk_agent(
        {"merchant_id": "AGT7", "daily_txn_volume": 9000}, tools, groq_client=scripted, on_step=events.append
    )

    thinking_events = [e for e in events if e["type"] == "thinking"]
    tool_call_events = [e for e in events if e["type"] == "tool_call"]
    final_events = [e for e in events if e["type"] == "final"]

    assert len(thinking_events) == 4  # one per model call, including the submit_decision turn
    assert [e["tool"] for e in tool_call_events] == [
        "get_merchant_context",
        "score_transaction_risk",
        "explain_transaction_risk",
    ]
    assert all(e["status"] == "success" for e in tool_call_events)
    assert len(final_events) == 1
    assert final_events[0]["decision"] == "clear"
    assert final_events[0]["gated_decision"] == result["gated_decision"]


def test_get_similar_past_cases_prioritizes_same_business_category(tools):
    """
    get_similar_past_cases is cross-merchant precedent, not this merchant's
    own history (that's get_recent_audit_history's job) -- so this locks in
    three things: the subject merchant's own past event must never come
    back, a past case in the SAME business category must outrank one in a
    different category, and the reported category must match what
    merchant_context.py would derive for the subject merchant.
    """
    subject_id = "SIM-SUBJECT"
    subject_category = get_merchant_context(subject_id)["business_category"]

    # business_category is deterministic per merchant_id (seeded hash), not
    # something a test can assign directly -- so scan for one candidate that
    # lands in the same category and one that doesn't.
    same_category_id, other_category_id = None, None
    for i in range(500):
        candidate = f"SIM-CANDIDATE-{i}"
        category = get_merchant_context(candidate)["business_category"]
        if category == subject_category and same_category_id is None:
            same_category_id = candidate
        elif category != subject_category and other_category_id is None:
            other_category_id = candidate
        if same_category_id and other_category_id:
            break
    assert same_category_id and other_category_id  # sanity: more than one category exists

    log_event(
        tools.conn, merchant_id=same_category_id, input_snapshot={}, risk_score=0.2,
        top_factors=None, explanation="Same-category past case.", decision="clear", decision_reason="ok",
    )
    log_event(
        tools.conn, merchant_id=other_category_id, input_snapshot={}, risk_score=0.9,
        top_factors=None, explanation="Other-category past case.", decision="flag_for_compliance_review", decision_reason="risky",
    )
    log_event(
        tools.conn, merchant_id=subject_id, input_snapshot={}, risk_score=0.5,
        top_factors=None, explanation="Subject's own past case.", decision="escalate", decision_reason="own history",
    )

    result = tools.get_similar_past_cases(subject_id)

    returned_ids = [c["merchant_id"] for c in result["similar_cases"]]
    assert subject_id not in returned_ids
    assert same_category_id in returned_ids
    assert returned_ids.index(same_category_id) < returned_ids.index(other_category_id)
    assert result["current_business_category"] == subject_category


def test_agent_can_call_get_similar_past_cases_tool(tools):
    scripted = ScriptedGroqClient(
        [
            _response(_tool_message([_tool_call("1", "get_merchant_context", {"merchant_id": "AGT8"})])),
            _response(_tool_message([_tool_call("2", "score_transaction_risk", MERCHANT_ARGS)])),
            _response(_tool_message([_tool_call("3", "explain_transaction_risk", MERCHANT_ARGS)])),
            _response(_tool_message([_tool_call("4", "get_similar_past_cases", {"merchant_id": "AGT8"})])),
            _response(
                _tool_message(
                    [_tool_call(
                        "5", "submit_decision",
                        {"recommended_decision": "clear", "reasoning": "Low risk, consistent with precedent."},
                    )]
                )
            ),
        ]
    )
    result = run_risk_agent({"merchant_id": "AGT8", "daily_txn_volume": 9000}, tools, groq_client=scripted)

    assert result["risk_score"] is not None
    assert len(result["trace"]) == 4
    assert result["trace"][3]["tool"] == "get_similar_past_cases"
    assert "similar_cases" in result["trace"][3]["result"]


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


def test_score_and_explain_reject_unrecognized_business_category(tools):
    """Regression test: RiskAgentTools.score_transaction_risk/explain_transaction_risk
    must fail safe (an error dict) on a category the model doesn't recognize,
    the same way pipeline.py's score_record already does via
    find_missing_or_invalid -- not silently one-hot-encode it as "no
    category" and return a normal-looking score. The tool schema's
    business_category field has no enum constraint (only kyc_status does),
    and even an enum is only ever a hint to the model, not something every
    provider enforces -- so this has to be checked in the tool itself."""
    bad_args = {k: v for k, v in MERCHANT_ARGS.items() if k != "merchant_id"}
    bad_args["business_category"] = "retail"  # not one of features.BUSINESS_CATEGORIES

    score_result = tools.score_transaction_risk(**bad_args)
    assert "error" in score_result
    assert "risk_score" not in score_result

    explain_result = tools.explain_transaction_risk(**bad_args)
    assert "error" in explain_result

    # A recognized category must still score normally.
    good_args = dict(bad_args, business_category="services")
    good_result = tools.score_transaction_risk(**good_args)
    assert "risk_score" in good_result
    assert isinstance(good_result["risk_score"], float)


def test_simulated_merchants_never_have_impossible_chargeback_or_refund_rates():
    """Regression test: get_merchant_context's chargebacks_30d/refunds_30d
    must scale with total_txns_30d (as data/raw/generate_data.py's actual
    training data does), not be drawn independently of it. An earlier
    version drew them independently, which meant roughly 1% of merchant IDs
    got a chargeback_rate or refund_rate above 100% -- more chargebacks/
    refunds than transactions, a value the model never sees in training --
    reachable through the Live Agent's "Try a different simulated merchant"
    button, i.e. through completely ordinary use of the most visible,
    judge-facing part of the app."""
    for i in range(2000):
        ctx = get_merchant_context(f"regression-check-{i}")
        total = ctx["total_txns_30d"]
        assert ctx["chargebacks_30d"] / total <= 1.0, ctx
        assert ctx["refunds_30d"] / total <= 1.0, ctx

    # Same merchant_id must always produce the same profile (determinism is
    # the whole point -- demos need to be reproducible).
    assert get_merchant_context("joy123") == get_merchant_context("joy123")


def test_finalize_treats_non_dict_agent_proposal_as_no_proposal():
    """Regression test: _parse_final_json only guarantees valid JSON, not a
    JSON *object* -- a model's final plain-text reply of "42", "true", or
    "[1,2]" is all valid JSON that isn't a dict. _finalize used to call
    agent_proposal.get(...) unconditionally whenever agent_proposal wasn't
    None, so a non-dict value raised AttributeError -- and because that
    crash happened before agent_pipeline.py's log_event call, the case
    never reached the audit log at all instead of failing safe to
    needs_manual_review like every other failure mode in this module."""
    result = _finalize(
        agent_proposal=42,
        agent_raw_response="42",
        computed_risk_score=0.55,
        explanation="some explanation",
        top_factors=None,
        trace=[],
    )
    assert result["agent_proposal"] is None
    # The independent gate still produces a real decision from the score --
    # a malformed agent answer must not block the fail-safe outcome.
    assert result["gated_decision"] is not None
    assert result["agent_and_gate_agree"] is False


def test_finalize_still_handles_a_well_formed_dict_proposal():
    result = _finalize(
        agent_proposal={"recommended_decision": "escalate", "reasoning": "high chargeback rate"},
        agent_raw_response='{"recommended_decision": "escalate", "reasoning": "high chargeback rate"}',
        computed_risk_score=0.55,
        explanation="some explanation",
        top_factors=None,
        trace=[],
    )
    assert result["agent_proposal"]["recommended_decision"] == "escalate"
