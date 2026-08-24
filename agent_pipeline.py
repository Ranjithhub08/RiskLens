"""
End-to-end AGENTIC pipeline: a live Razorpay test-mode order in, an
audited, gated decision out -- with an LLM reasoning loop in between
instead of a fixed script.

This sits alongside pipeline.py (the original deterministic path), rather
than replacing it: pipeline.py is still what the dashboard's "Score a case"
tab uses for instant, deterministic scoring, and remains fully covered by
its own tests. This module is the new agent-driven path, used by the
dashboard's "Live agent" tab.
"""

from agent.risk_agent import run_risk_agent
from agent.tools import RiskAgentTools
from audit.audit_log import log_event
from integrations.razorpay_client import create_test_order, order_to_transaction_fields


def run_agentic_scoring(merchant_id: str, amount_rupees: float, model, explainer, conn, groq_client=None) -> dict:
    """
    Creates a real Razorpay test-mode order for the given amount, then runs
    the risk agent over it (merchant_id ties the live order to a simulated
    merchant history -- see agent/merchant_context.py), and logs the full
    result including the agent's reasoning trace.
    """
    order = create_test_order(amount_rupees, merchant_id)
    txn_fields = order_to_transaction_fields(order)

    tools = RiskAgentTools(model, explainer, conn)
    transaction = {
        "merchant_id": merchant_id,
        "daily_txn_volume": txn_fields["daily_txn_volume"],
        "razorpay_order_id": txn_fields["razorpay_order_id"],
    }

    agent_result = run_risk_agent(transaction, tools, groq_client=groq_client)

    event_id = log_event(
        conn,
        merchant_id=merchant_id,
        input_snapshot={"transaction": transaction, "razorpay_order": order},
        risk_score=agent_result["risk_score"],
        top_factors=agent_result["top_factors"],
        explanation=agent_result["explanation"],
        decision=agent_result["gated_decision"],
        decision_reason=agent_result["gated_reason"],
        source="agent_pipeline",
        agent_proposal=agent_result["agent_proposal"],
        agent_trace=agent_result["trace"],
    )

    return {
        "event_id": event_id,
        "razorpay_order_id": txn_fields["razorpay_order_id"],
        "razorpay_order_amount": txn_fields["daily_txn_volume"],
        "razorpay_order_currency": txn_fields["currency"],
        "razorpay_order_status": txn_fields["order_status"],
        "razorpay_order_created_at_epoch": txn_fields["created_at_epoch"],
        "merchant_id": merchant_id,
        **agent_result,
    }
