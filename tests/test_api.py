"""
Tests for api/main.py's /score endpoint under concurrent load.

The API used to open ONE sqlite3.Connection at startup and store it in a
module-level global, reused by every request. FastAPI dispatches sync `def`
route handlers (like `score()`) to a thread pool, so concurrent /score
calls really do run on different OS threads -- all sharing that single
connection object. Under real concurrent traffic this produced intermittent
"cannot start a transaction within a transaction" errors and, worse, HTTP
200 responses whose audit row silently never made it into audit_events at
all: a real hole in "every scoring event is written here, append-only"
under nothing more exotic than ordinary concurrent requests.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from audit.audit_log import get_all_events, get_connection

VALID_RECORD = {
    "merchant_id": "M-CONCURRENT-TEST",
    "account_age_days": 400,
    "kyc_status": "complete",
    "business_category": "services",
    "daily_txn_volume": 1000,
    "avg_30d_txn_volume": 1000,
    "total_txns_30d": 100,
    "chargebacks_30d": 1,
    "refunds_30d": 2,
    "avg_ticket_size": 50,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_api_audit.db")
    monkeypatch.setattr(api_main, "get_connection", lambda: get_connection(db_path))
    api_main.load_resources()
    return TestClient(api_main.app), db_path


def test_concurrent_score_requests_all_land_in_the_audit_log(client):
    """Regression test: every successful /score response must correspond to
    exactly one durable row in audit_events -- no request should get a
    real-looking event_id back for a row that silently never got written."""
    # This is a genuine race condition against the old shared-connection
    # code -- reproducing it isn't 100% deterministic on every run (it took
    # 30 concurrent requests / 10 workers roughly 1-in-5 runs to trip during
    # development), so this uses more requests and more workers than the
    # minimum that showed the bug, to make a regression here reliably
    # reproducible rather than an occasional flake.
    test_client, db_path = client
    n_requests = 80

    def make_request(_i):
        response = test_client.post("/score", json=VALID_RECORD)
        return response.status_code, response.json().get("event_id") if response.status_code == 200 else None

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(make_request, range(n_requests)))

    statuses = [r[0] for r in results]
    assert all(s == 200 for s in statuses), f"non-200 responses: {statuses}"

    returned_event_ids = {r[1] for r in results}
    assert len(returned_event_ids) == n_requests  # every response got a distinct event_id

    conn = get_connection(db_path)
    try:
        durable_events = get_all_events(conn, limit=1000)
    finally:
        conn.close()
    durable_event_ids = {e["event_id"] for e in durable_events if e["merchant_id"] == "M-CONCURRENT-TEST"}
    assert durable_event_ids == returned_event_ids
