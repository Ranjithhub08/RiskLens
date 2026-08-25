import os
import tempfile

import pytest

from audit.audit_log import (
    get_all_events,
    get_all_overrides,
    get_connection,
    get_events_for_merchant,
    get_overrides_for_event,
    log_event,
    log_override,
)


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


def test_log_override_does_not_touch_the_original_event(temp_conn):
    """
    A human override must never mutate the original decision -- it's a
    separate, additional record layered on top (see audit_log.py's
    OVERRIDES_SCHEMA docstring for why). This is the test that would fail
    first if someone "simplified" log_override into an UPDATE statement.
    """
    event_id = log_event(temp_conn, "M1", {"a": 1}, 0.85, None, "explanation", "flag_for_compliance_review", "high risk")
    original_event = get_all_events(temp_conn)[0]

    log_override(
        temp_conn, event_id=event_id, original_decision="flag_for_compliance_review",
        overridden_decision="clear", reason="Confirmed legitimate with the merchant directly.",
    )

    # The original audit_events row is byte-for-byte unchanged.
    unchanged_event = get_all_events(temp_conn)[0]
    assert unchanged_event == original_event
    assert unchanged_event["decision"] == "flag_for_compliance_review"


def test_get_overrides_for_event_filters_and_orders_correctly(temp_conn):
    event_id = log_event(temp_conn, "M1", {"a": 1}, 0.85, None, "e", "escalate", "r")
    other_event_id = log_event(temp_conn, "M2", {"a": 2}, 0.85, None, "e", "escalate", "r")
    log_override(temp_conn, event_id, "escalate", "clear", "first correction")
    log_override(temp_conn, event_id, "clear", "flag_for_compliance_review", "second correction, reviewer changed their mind")
    log_override(temp_conn, other_event_id, "escalate", "clear", "unrelated case's override")

    overrides = get_overrides_for_event(temp_conn, event_id)
    assert len(overrides) == 2
    assert all(o["event_id"] == event_id for o in overrides)
    # Most recent first.
    assert overrides[0]["reason"] == "second correction, reviewer changed their mind"
    assert overrides[0]["original_decision"] == "clear"
    assert overrides[0]["overridden_decision"] == "flag_for_compliance_review"


def test_get_overrides_for_event_breaks_timestamp_ties_by_insertion_order(temp_conn):
    """Regression test: get_overrides_for_event/get_all_overrides/
    get_all_events/get_events_for_merchant all order by timestamp_utc DESC
    alone used to have no tiebreaker. Two overrides landing in the same
    microsecond (a scripted burst, or two rapid clicks) get an identical
    ISO timestamp string, and SQLite then returns tied rows in their
    on-disk (insertion) order under DESC -- the OLDER of the two ties would
    sort FIRST, which model/feedback.py's override-dedup fix and the
    dashboard's override-history display both silently depend on being
    false. Insert two overrides with an identical timestamp_utc directly
    (bypassing log_override's real-clock timestamp so the tie is
    deterministic) and confirm the later INSERT still sorts first."""
    event_id = log_event(temp_conn, "M1", {"a": 1}, 0.85, None, "e", "escalate", "r")
    tied_ts = "2026-01-01T00:00:00+00:00"
    temp_conn.execute(
        "INSERT INTO human_overrides (override_id, event_id, timestamp_utc, original_decision, overridden_decision, reason) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("override-1-first-mistaken", event_id, tied_ts, "escalate", "escalate", "first, mistaken"),
    )
    temp_conn.execute(
        "INSERT INTO human_overrides (override_id, event_id, timestamp_utc, original_decision, overridden_decision, reason) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("override-2-correction", event_id, tied_ts, "escalate", "clear", "self-corrected"),
    )
    temp_conn.commit()

    overrides = get_overrides_for_event(temp_conn, event_id)
    assert len(overrides) == 2
    assert overrides[0]["override_id"] == "override-2-correction"


def test_get_all_overrides_returns_every_override(temp_conn):
    e1 = log_event(temp_conn, "M1", {"a": 1}, 0.85, None, "e", "escalate", "r")
    e2 = log_event(temp_conn, "M2", {"a": 2}, 0.2, None, "e", "clear", "r")
    log_override(temp_conn, e1, "escalate", "clear", "reason 1")
    log_override(temp_conn, e2, "clear", "escalate", "reason 2")

    all_overrides = get_all_overrides(temp_conn)
    assert len(all_overrides) == 2
    assert {o["event_id"] for o in all_overrides} == {e1, e2}
