import os
import tempfile

import pytest

from audit.audit_log import get_all_events, get_connection, get_events_for_merchant, log_event


@pytest.fixture
def temp_conn():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = get_connection(path)
    yield conn
    conn.close()
    os.remove(path)


def test_log_event_returns_an_id(temp_conn):
    event_id = log_event(
        temp_conn,
        merchant_id="M1",
        input_snapshot={"a": 1},
        risk_score=0.5,
        top_factors=[{"feature": "x", "shap_value": 0.1}],
        explanation="test explanation",
        decision="clear",
        decision_reason="test reason",
    )
    assert event_id is not None
    assert len(event_id) > 0


def test_logged_event_is_retrievable(temp_conn):
    log_event(
        temp_conn,
        merchant_id="M2",
        input_snapshot={"a": 2},
        risk_score=0.8,
        top_factors=None,
        explanation="explanation 2",
        decision="flag_for_compliance_review",
        decision_reason="reason 2",
    )
    events = get_all_events(temp_conn)
    assert len(events) == 1
    assert events[0]["merchant_id"] == "M2"
    assert events[0]["decision"] == "flag_for_compliance_review"


def test_get_events_for_merchant_filters_correctly(temp_conn):
    log_event(temp_conn, "M1", {"a": 1}, 0.1, None, "e1", "clear", "r1")
    log_event(temp_conn, "M2", {"a": 2}, 0.9, None, "e2", "flag_for_compliance_review", "r2")
    log_event(temp_conn, "M1", {"a": 3}, 0.2, None, "e3", "clear", "r3")

    m1_events = get_events_for_merchant(temp_conn, "M1")
    assert len(m1_events) == 2
    assert all(e["merchant_id"] == "M1" for e in m1_events)


def test_log_is_append_only_no_update_or_delete_helpers_exist():
    import audit.audit_log as audit_log_module

    module_functions = dir(audit_log_module)
    assert not any("update" in f.lower() for f in module_functions)
    assert not any("delete" in f.lower() for f in module_functions)


def test_multiple_events_ordered_most_recent_first(temp_conn):
    log_event(temp_conn, "M1", {"a": 1}, 0.1, None, "e1", "clear", "r1")
    log_event(temp_conn, "M1", {"a": 2}, 0.2, None, "e2", "clear", "r2")
    events = get_all_events(temp_conn)
    assert len(events) == 2
    # most recent first
    assert events[0]["input_snapshot"] == '{"a": 2}'
