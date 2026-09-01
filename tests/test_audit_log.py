import json
import os
import sqlite3
import tempfile
import threading

import pytest

from audit.audit_log import (
    _MIGRATION_COLUMNS,
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


def test_log_event_persists_gate_version_and_thresholds_used(temp_conn):
    # Regression test: gating.decision_engine.GATE_VERSION/thresholds_used
    # used to be computed on every decision but never written anywhere --
    # the audit trail's own docstring promise ("any past decision can be
    # replayed exactly") was false, and it had already gone stale once
    # (thresholds changed in commit bf2d34c with nothing logged either
    # way). A row must carry the exact gate identity that produced it.
    event_id = log_event(
        temp_conn, "M1", {"a": 1}, 0.55, None, "e", "escalate", "r",
        gate_version="DETERMINISTIC-GATE-01",
        thresholds_used={"escalate_threshold": 0.50, "flag_threshold": 0.62},
    )
    events = get_all_events(temp_conn)
    event = next(e for e in events if e["event_id"] == event_id)
    assert event["gate_version"] == "DETERMINISTIC-GATE-01"
    assert json.loads(event["thresholds_used"]) == {"escalate_threshold": 0.50, "flag_threshold": 0.62}


def test_log_event_without_gate_version_leaves_it_null_not_a_crash(temp_conn):
    # Existing callers that haven't been updated to pass gate_version/
    # thresholds_used must still work unchanged.
    event_id = log_event(temp_conn, "M1", {"a": 1}, 0.1, None, "e", "clear", "r")
    events = get_all_events(temp_conn)
    event = next(e for e in events if e["event_id"] == event_id)
    assert event["gate_version"] is None
    assert event["thresholds_used"] is None


def test_concurrent_writes_from_multiple_threads_on_one_shared_connection_do_not_corrupt_or_lose_rows(temp_conn):
    # Regression test: the dashboard caches ONE sqlite3.Connection
    # (check_same_thread=False) shared across every Streamlit session's
    # worker thread. Before _WRITE_LOCK existed, two threads' execute()/
    # commit() calls could interleave on that same connection object --
    # exactly the mechanism commit a423fa2 already proved causes
    # "cannot start a transaction within a transaction" errors and rows
    # that silently never land in the table, for api/main.py's now-fixed
    # per-request-connection pattern. That fix never touched this
    # still-shared dashboard connection. Hammer the same connection from
    # many threads at once and confirm every single row survives.
    n_threads = 20
    writes_per_thread = 5
    errors = []

    def write_many(thread_idx):
        try:
            for i in range(writes_per_thread):
                log_event(
                    temp_conn, f"M{thread_idx}", {"i": i}, 0.5, None, "e", "clear",
                    f"thread {thread_idx} write {i}",
                )
        except Exception as e:  # pragma: no cover - failure path under test
            errors.append(e)

    threads = [threading.Thread(target=write_many, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    events = get_all_events(temp_conn, limit=n_threads * writes_per_thread + 10)
    assert len(events) == n_threads * writes_per_thread


def _flaky_connection_factory(operational_error_message):
    """A sqlite3.Connection subclass whose first ALTER TABLE ADD COLUMN
    raises the given OperationalError message once, then behaves normally --
    simulating a concurrent connection having just won a migration race."""

    class FlakyConnection(sqlite3.Connection):
        _raised = False

        def execute(self, sql, *args, **kwargs):
            if "ALTER TABLE" in sql and "ADD COLUMN" in sql and not FlakyConnection._raised:
                FlakyConnection._raised = True
                raise sqlite3.OperationalError(operational_error_message)
            return super().execute(sql, *args, **kwargs)

    return FlakyConnection


def test_get_connection_tolerates_a_concurrent_duplicate_column_migration_race(monkeypatch, tmp_path):
    # Regression test: get_connection() reads existing_cols, then runs
    # ALTER TABLE for any column it thinks is missing. Two connections
    # opened against the same on-disk database at nearly the same moment
    # can both read existing_cols BEFORE either has run its ALTER TABLE --
    # so even though THIS connection correctly saw the column as missing
    # (a genuine TOCTOU race, not a bug in its own bookkeeping), a
    # concurrent connection can add it first, and this connection's own
    # ALTER TABLE then raises "duplicate column name" and used to crash
    # get_connection() (and whatever request/page load triggered it) for
    # no real reason -- the column ends up present either way.
    db_path = str(tmp_path / "race.db")
    real_connect = sqlite3.connect
    factory = _flaky_connection_factory(f"duplicate column name: {_MIGRATION_COLUMNS[0][0]}")
    monkeypatch.setattr(
        "audit.audit_log.sqlite3.connect",
        lambda database, **kwargs: real_connect(database, factory=factory, **kwargs),
    )

    conn = get_connection(db_path)  # must not raise
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(audit_events)")}
    # The second migration column, unaffected by the simulated race,
    # was still added normally by this same call.
    assert _MIGRATION_COLUMNS[1][0] in existing_cols
    conn.close()


def test_get_connection_still_raises_a_genuinely_different_operational_error(monkeypatch, tmp_path):
    # Only "duplicate column name" is a known-harmless race; any other
    # ALTER TABLE failure is a real problem and must still surface rather
    # than being silently swallowed alongside it.
    db_path = str(tmp_path / "real_error.db")
    real_connect = sqlite3.connect
    factory = _flaky_connection_factory("disk I/O error")
    monkeypatch.setattr(
        "audit.audit_log.sqlite3.connect",
        lambda database, **kwargs: real_connect(database, factory=factory, **kwargs),
    )

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        get_connection(db_path)
