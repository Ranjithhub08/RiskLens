"""
Tests for agent_pipeline.py's end-to-end orchestration.

run_agentic_scoring() previously had no try/except around create_test_order
(a real, unauthenticated-on-failure network call to Razorpay's API) or the
agent loop itself -- a failure there propagated straight out with NOTHING
logged to the audit trail, not even a needs_manual_review fallback like
every other failure mode in the agent path gets. This directly contradicts
the project's own repeated invariant that every scoring event is written to
the audit log.
"""

from unittest.mock import patch

import pytest

from audit.audit_log import get_connection, get_events_for_merchant
from gating.decision_engine import DECISION_MANUAL_REVIEW

import agent_pipeline


@pytest.fixture
def conn(tmp_path):
    connection = get_connection(str(tmp_path / "test_agent_pipeline_audit.db"))
    yield connection
    connection.close()


def test_razorpay_order_creation_failure_still_logs_a_manual_review_event(conn):
    with patch.object(agent_pipeline, "create_test_order", side_effect=RuntimeError("Razorpay unreachable")):
        with pytest.raises(RuntimeError, match="Razorpay unreachable"):
            agent_pipeline.run_agentic_scoring("merchant-xyz", 500.0, model=None, explainer=None, conn=conn)

    events = get_events_for_merchant(conn, "merchant-xyz")
    assert len(events) == 1
    assert events[0]["decision"] == DECISION_MANUAL_REVIEW
    assert events[0]["source"] == "agent_pipeline"
    assert events[0]["risk_score"] is None
