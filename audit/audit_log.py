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
import threading
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
    # Which gate rule set (GATE_VERSION) and threshold values actually
    # produced this row's decision, at the time it was produced -- without
    # these, "any past decision can be replayed exactly" (this module's own
    # docstring) was false: gating/decision_engine.py's ESCALATE_THRESHOLD/
    # FLAG_THRESHOLD have already changed once (commit bf2d34c) without
    # GATE_VERSION being bumped or anything logged either way, so an old row
    # and a new row were indistinguishable, and the dashboard's case-detail
    # "Final authority" display always showed the CURRENT constant for
    # every case, including ones scored under different thresholds.
    ("gate_version", "TEXT"),
    ("thresholds_used", "TEXT"),
]


# check_same_thread=False (below) lets the SAME sqlite3.Connection object be
# called from more than one OS thread -- which the dashboard actually does,
# since it caches one connection as a resource (st.cache_resource) but
# Streamlit dispatches different user sessions to different worker threads.
# But check_same_thread=False only lifts sqlite3's thread-affinity check; it
# does NOT make concurrent use of one connection safe on its own -- two
# threads' execute()/commit() calls can still interleave on the same
# connection object and corrupt the same logical transaction. That's exactly
# what commit a423fa2 found and fixed for api/main.py (each request there
# now gets its own connection) -- but a connection created here can still be
# shared across threads by a caller (as the dashboard does), so every WRITE
# in this module is serialized through _WRITE_LOCK below rather than
# trusting each caller to avoid the same mistake api/main.py already made
# once.
_WRITE_LOCK = threading.Lock()


def get_connection(db_path: str = DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(SCHEMA)
    conn.execute(OVERRIDES_SCHEMA)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(audit_events)")}
    for col_name, col_type in _MIGRATION_COLUMNS:
        if col_name not in existing_cols:
            try:
                conn.execute(f"ALTER TABLE audit_events ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError as e:
                # TOCTOU: get_connection() can be called concurrently against
                # the same on-disk database from more than one connection
                # (e.g. api/main.py opens a fresh one per request while the
                # dashboard holds its own long-lived one open) -- two
                # connections can both read existing_cols BEFORE either has
                # run its ALTER TABLE, and the second one then raises
                # "duplicate column name", crashing get_connection() (and
                # whatever request/page load triggered it) for no reason,
                # since the column ends up present either way. Only swallow
                # that specific, harmless race; any other ALTER TABLE
                # failure is a real problem and should still surface.
                if "duplicate column name" not in str(e):
                    raise
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
    gate_version: str = None,
    thresholds_used: dict = None,
) -> str:
    """Write one immutable audit record. Returns the generated event_id.

    gate_version/thresholds_used: the gating.decision_engine.GATE_VERSION
    identifier and GatingResult.thresholds_used dict that actually produced
    `decision` for THIS event -- captured at write time so a later change to
    the live thresholds/gate implementation can never make an old row's
    displayed authority/thresholds silently wrong (see the migration
    comment on _MIGRATION_COLUMNS above). Optional so existing callers that
    haven't been updated yet still work; such rows simply have no persisted
    gate identity, same as any row written before this column existed.
    """
    event_id = str(uuid.uuid4())
    with _WRITE_LOCK:
        conn.execute(
            """
            INSERT INTO audit_events
                (event_id, timestamp_utc, merchant_id, input_snapshot, risk_score,
                 top_factors, explanation, decision, decision_reason, source,
                 agent_proposal, agent_trace, gate_version, thresholds_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                gate_version,
                json.dumps(thresholds_used, default=str) if thresholds_used is not None else None,
            ),
        )
        conn.commit()
    return event_id


# Every "most recent first" query in this module orders by timestamp_utc
# DESC with rowid DESC as a tiebreaker. timestamp_utc alone isn't a strict
# ordering: two writes landing in the same microsecond (a scripted burst, or
# two rapid override clicks) get an identical ISO string, and SQLite then
# falls back to returning tied rows in their on-disk (insertion) order under
# DESC -- the OLDER of the two ties would sort first, backwards from what
# every caller of these functions assumes ("most recent" for the dashboard's
# override-history display, "latest verdict" for model/feedback.py's
# override dedup). rowid strictly increases with insertion order and is
# never reused (these tables use a TEXT PRIMARY KEY, not
# `INTEGER PRIMARY KEY`, so it doesn't get aliased away), so ordering by it
# as a tiebreaker resolves ties in true insertion order regardless of
# timestamp resolution.
def get_all_events(conn, limit: int = 500):
    cur = conn.execute(
        "SELECT * FROM audit_events ORDER BY timestamp_utc DESC, rowid DESC LIMIT ?", (limit,)
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
        "SELECT * FROM audit_events WHERE merchant_id = ? ORDER BY timestamp_utc DESC, rowid DESC LIMIT ?",
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
    with _WRITE_LOCK:
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
        "SELECT * FROM human_overrides WHERE event_id = ? ORDER BY timestamp_utc DESC, rowid DESC LIMIT ?",
        (str(event_id), limit),
    )
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_all_overrides(conn, limit: int = 1000):
    """Every override ever recorded, most recent first -- the raw material for
    a feedback-driven retrain: each row says what the model/gate originally
    decided, what a human corrected it to, and why."""
    cur = conn.execute("SELECT * FROM human_overrides ORDER BY timestamp_utc DESC, rowid DESC LIMIT ?", (limit,))
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]
