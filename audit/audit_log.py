"""
Append-only audit log.

Every scoring event -- whatever the outcome -- is written here before it's
returned to a caller. Rows are never updated or deleted (no UPDATE/DELETE
statements exist in this module on purpose), so any past decision can be
replayed exactly: what data went in, what the model said, why, and what
the gating engine decided.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

DEFAULT_DB_PATH = "audit/audit_log.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    event_id TEXT PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    merchant_id TEXT,
    input_snapshot TEXT NOT NULL,
    risk_score REAL,
    top_factors TEXT,
    explanation TEXT,
    decision TEXT NOT NULL,
    decision_reason TEXT NOT NULL,
    source TEXT,
    agent_proposal TEXT,
    agent_trace TEXT
);
"""

# A human override is deliberately its OWN table, not a column bolted onto
# audit_events -- the original scoring event (what the model/gate decided,
# and why) must stay exactly as it was decided, immutable, forever. An
# override is a separate fact layered on top of it later ("a reviewer later
# disagreed and changed the outcome"), not an edit to history. This also
# means an event can carry a full override history over time, not just one.
OVERRIDES_SCHEMA = """
CREATE TABLE IF NOT EXISTS human_overrides (
    override_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    original_decision TEXT NOT NULL,
    overridden_decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    reviewer TEXT
);
"""

# Columns added after the original schema shipped. Kept as an explicit
# migration list (rather than DROP/recreate) so an existing audit_log.db
# from before the agent existed keeps its history instead of losing it.
_MIGRATION_COLUMNS = [
    ("source", "TEXT"),
    ("agent_proposal", "TEXT"),
    ("agent_trace", "TEXT"),
]


def get_connection(db_path: str = DEFAULT_DB_PATH):
    # check_same_thread=False: the dashboard caches this connection as a
    # resource (st.cache_resource) but Streamlit can rerun the script body
    # on a different worker thread per session, which would otherwise raise
    # "SQLite objects created in a thread can only be used in that same
    # thread." Writes here are small and infrequent (one row per scoring
    # event), so sharing the connection across threads is safe for this
    # single-process demo app.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(SCHEMA)
    conn.execute(OVERRIDES_SCHEMA)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(audit_events)")}
    for col_name, col_type in _MIGRATION_COLUMNS:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE audit_events ADD COLUMN {col_name} {col_type}")
    conn.commit()
    return conn


def log_event(
    conn,
    merchant_id,
    input_snapshot: dict,
    risk_score,
    top_factors,
    explanation: str,
    decision: str,
    decision_reason: str,
    source: str = "rule_pipeline",
    agent_proposal: dict = None,
    agent_trace: list = None,
) -> str:
    """Write one immutable audit record. Returns the generated event_id."""
    event_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO audit_events
            (event_id, timestamp_utc, merchant_id, input_snapshot, risk_score,
             top_factors, explanation, decision, decision_reason, source,
             agent_proposal, agent_trace)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            datetime.now(timezone.utc).isoformat(),
            str(merchant_id) if merchant_id is not None else None,
            json.dumps(input_snapshot, default=str),
            risk_score,
            json.dumps(top_factors, default=str) if top_factors is not None else None,
            explanation,
            decision,
            decision_reason,
            source,
            json.dumps(agent_proposal, default=str) if agent_proposal is not None else None,
            json.dumps(agent_trace, default=str) if agent_trace is not None else None,
        ),
    )
    conn.commit()
    return event_id


def get_all_events(conn, limit: int = 500):
    cur = conn.execute(
        "SELECT * FROM audit_events ORDER BY timestamp_utc DESC LIMIT ?", (limit,)
    )
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_event_by_id(conn, event_id: str):
    """Single event lookup, e.g. to trace a human_overrides row back to the
    original scoring event it corrected (see model/feedback.py)."""
    cur = conn.execute("SELECT * FROM audit_events WHERE event_id = ?", (str(event_id),))
    row = cur.fetchone()
    if row is None:
        return None
    columns = [d[0] for d in cur.description]
    return dict(zip(columns, row))


def get_events_for_merchant(conn, merchant_id, limit: int = 100):
    cur = conn.execute(
        "SELECT * FROM audit_events WHERE merchant_id = ? ORDER BY timestamp_utc DESC LIMIT ?",
        (str(merchant_id), limit),
    )
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def log_override(conn, event_id: str, original_decision: str, overridden_decision: str, reason: str, reviewer: str = None) -> str:
    """
    Record a human reviewer correcting a past decision. This never touches
    the original audit_events row (see OVERRIDES_SCHEMA's docstring) -- it's
    a new, separate, equally immutable record: "on this date, a reviewer
    changed this case's outcome from X to Y, and here's why." That reason
    text is exactly the kind of labeled correction a future retrain would
    want to learn from -- see get_all_overrides.
    """
    override_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO human_overrides
            (override_id, event_id, timestamp_utc, original_decision, overridden_decision, reason, reviewer)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            override_id,
            str(event_id),
            datetime.now(timezone.utc).isoformat(),
            original_decision,
            overridden_decision,
            reason,
            reviewer,
        ),
    )
    conn.commit()
    return override_id


def get_overrides_for_event(conn, event_id: str, limit: int = 20):
    cur = conn.execute(
        "SELECT * FROM human_overrides WHERE event_id = ? ORDER BY timestamp_utc DESC LIMIT ?",
        (str(event_id), limit),
    )
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_all_overrides(conn, limit: int = 1000):
    """Every override ever recorded, most recent first -- the raw material for
    a feedback-driven retrain: each row says what the model/gate originally
    decided, what a human corrected it to, and why."""
    cur = conn.execute("SELECT * FROM human_overrides ORDER BY timestamp_utc DESC LIMIT ?", (limit,))
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]
