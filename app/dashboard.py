"""
RiskLens -- Merchant Risk Intelligence Platform.

Five pages, routed with Streamlit's native sidebar navigation (st.navigation)
rather than top tabs, matching a real desktop fintech application shell:
  1. Overview        -- command center: KPIs, risk activity/distribution,
                         recent investigations, system health.
  2. Investigations  -- searchable/filterable case table with a full case
                         detail panel (merchant context, risk assessment,
                         SHAP, decision control, audit reference).
  3. Live Agent       -- creates a REAL Razorpay test-mode order and runs the
                         LLM-driven risk agent, in a three-column
                         transaction / execution / decision layout.
  4. Models           -- held-out test-set performance with real, interactive
                         (Altair) ROC, confusion-matrix, and SHAP charts.
  5. Audit Trail      -- every decision this session has made, filterable
                         and individually inspectable.

Run:
    streamlit run app/dashboard.py
"""

import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_pipeline import run_agentic_scoring
from app.theme import (
    STATUS_ACTIVE,
    STATUS_CONNECTED,
    STATUS_OFFLINE,
    STATUS_ONLINE,
    authority_strip_html,
    case_id_from_event,
    compare_panel_html,
    confusion_matrix_chart,
    decision_badge_html,
    empty_state_html,
    html_block,
    inject_theme,
    kpi_html,
    model_comparison_table_html,
    order_panel_html,
    render_sidebar_status,
    render_top_bar,
    risk_activity_chart,
    risk_distribution_chart,
    risk_label_for_score,
    risk_scale_html,
    roc_chart,
    shap_bars_html,
    shap_global_chart,
    status_row_html,
    workflow_html,
)
from audit.audit_log import get_all_events, get_connection
from explainability.explain import RiskExplainer
from features.features import BUSINESS_CATEGORIES
from gating.decision_engine import GATE_VERSION
from pipeline import load_model, score_record

MODEL_PATH = "model/artifacts/xgb_model.joblib"
METRICS_PATH = "model/artifacts/metrics.json"
CHART_DATA_PATH = "model/artifacts/chart_data.json"
RAW_DATA_PATH = "data/raw/merchant_snapshots.csv"

INVESTIGATION_STEPS = ["Merchant data", "Risk model", "SHAP explanation", "Deterministic gate", "Final decision", "Audit event"]
AGENT_STEPS = ["Order created", "Agent investigation", "Agent proposal", "Deterministic gate", "Final decision", "Audit committed"]
REVIEW_DECISIONS = {"escalate", "flag_for_compliance_review", "needs_manual_review"}
HIGH_SEVERITY_DECISIONS = {"flag_for_compliance_review", "needs_manual_review"}

st.set_page_config(page_title="RiskLens", layout="wide", page_icon=None)
inject_theme()


@st.cache_resource
def get_model_and_explainer():
    model = load_model(MODEL_PATH)
    explainer = RiskExplainer(model)
    return model, explainer


@st.cache_resource
def get_db_connection():
    return get_connection("audit/audit_log.db")


@st.cache_data
def load_sample_merchants(n: int = 25):
    if not os.path.exists(RAW_DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(RAW_DATA_PATH)
    return df.sample(n=min(n, len(df)), random_state=7).reset_index(drop=True)


if not os.path.exists(MODEL_PATH):
    st.error(
        "No trained model found at `model/artifacts/xgb_model.joblib`. "
        "Run `python3 data/raw/generate_data.py` then `python3 model/train.py` first."
    )
    st.stop()

model, explainer = get_model_and_explainer()
conn = get_db_connection()

GROQ_CONFIGURED = bool(os.environ.get("GROQ_API_KEY"))
RAZORPAY_CONFIGURED = bool(os.environ.get("RAZORPAY_KEY_ID")) and bool(os.environ.get("RAZORPAY_KEY_SECRET"))


def relative_time(ts_str: str) -> str:
    try:
        ts = datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return "—"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - ts
    seconds = delta.total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def extract_case_view(event: dict) -> dict:
    """Normalizes a rule_pipeline or agent_pipeline audit event into one
    common shape for the Investigations table/detail panel. Every field
    either comes straight off the event or, for agent-pipeline events, is
    recovered from the recorded tool trace -- nothing is guessed or
    invented; unavailable fields show as None and render as a dash."""
    source = event.get("source") or "rule_pipeline"
    snapshot = event.get("input_snapshot")
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except json.JSONDecodeError:
            snapshot = {}
    snapshot = snapshot or {}

    view = {
        "event_id": event.get("event_id"),
        "timestamp_utc": event.get("timestamp_utc"),
        "merchant_id": event.get("merchant_id"),
        "source": source,
        "risk_score": event.get("risk_score"),
        "decision": event.get("decision"),
        "decision_reason": event.get("decision_reason"),
        "explanation": event.get("explanation"),
        "order_id": None,
        "account_age_days": None, "kyc_status": None, "business_category": None,
        "daily_txn_volume": None, "avg_30d_txn_volume": None, "total_txns_30d": None,
        "chargebacks_30d": None, "refunds_30d": None, "avg_ticket_size": None,
        "agent_proposal": None, "agent_trace": None,
    }

    top_factors = event.get("top_factors")
    if isinstance(top_factors, str):
        try:
            top_factors = json.loads(top_factors)
        except json.JSONDecodeError:
            top_factors = None
    view["top_factors"] = top_factors

    if source == "rule_pipeline":
        for k in ("account_age_days", "kyc_status", "business_category", "daily_txn_volume",
                   "avg_30d_txn_volume", "total_txns_30d", "chargebacks_30d", "refunds_30d", "avg_ticket_size"):
            view[k] = snapshot.get(k)
    else:
        txn = snapshot.get("transaction", {})
        view["order_id"] = txn.get("razorpay_order_id")
        view["daily_txn_volume"] = txn.get("daily_txn_volume")

        agent_proposal = event.get("agent_proposal")
        if isinstance(agent_proposal, str):
            try:
                agent_proposal = json.loads(agent_proposal)
            except json.JSONDecodeError:
                agent_proposal = None
        view["agent_proposal"] = agent_proposal

        agent_trace = event.get("agent_trace")
        if isinstance(agent_trace, str):
            try:
                agent_trace = json.loads(agent_trace)
            except json.JSONDecodeError:
                agent_trace = None
        view["agent_trace"] = agent_trace

        if agent_trace:
            for step in agent_trace:
                if step.get("tool") == "get_merchant_context" and isinstance(step.get("result"), dict):
                    ctx = step["result"]
                    view["account_age_days"] = ctx.get("account_age_days")
                    view["kyc_status"] = ctx.get("kyc_status")
                    view["business_category"] = ctx.get("business_category")
                    view["avg_30d_txn_volume"] = ctx.get("avg_30d_txn_volume")
                    view["total_txns_30d"] = ctx.get("total_txns_30d")
                    view["chargebacks_30d"] = ctx.get("chargebacks_30d")
                    view["refunds_30d"] = ctx.get("refunds_30d")
                    view["avg_ticket_size"] = ctx.get("avg_ticket_size")
                    break
    return view


def render_case_detail(view: dict):
    risk_label, risk_color = risk_label_for_score(view["risk_score"])
    score_display = f"{view['risk_score']:.2f}" if view["risk_score"] is not None else "--"

    html_block(
        f"""
        <div class="rl-panel">
            <div class="rl-panel-label">Merchant context</div>
            <div class="rl-kv-grid">
                <div><div class="rl-kv-label">Merchant</div><div class="rl-kv-value">{view['merchant_id'] or '—'}</div></div>
                <div><div class="rl-kv-label">Account age</div><div class="rl-kv-value">{f"{view['account_age_days']:.0f} days" if view['account_age_days'] is not None else '—'}</div></div>
                <div><div class="rl-kv-label">KYC</div><div class="rl-kv-value">{(view['kyc_status'] or '—').title()}</div></div>
                <div><div class="rl-kv-label">Category</div><div class="rl-kv-value">{(view['business_category'] or '—').title()}</div></div>
                <div><div class="rl-kv-label">Txn volume</div><div class="rl-kv-value">{f"₹{view['daily_txn_volume']:,.2f}" if view['daily_txn_volume'] is not None else '—'}</div></div>
                <div><div class="rl-kv-label">30d average</div><div class="rl-kv-value">{f"₹{view['avg_30d_txn_volume']:,.2f}" if view['avg_30d_txn_volume'] is not None else '—'}</div></div>
                <div><div class="rl-kv-label">Chargebacks</div><div class="rl-kv-value">{view['chargebacks_30d'] if view['chargebacks_30d'] is not None else '—'}</div></div>
                <div><div class="rl-kv-label">Refunds</div><div class="rl-kv-value">{view['refunds_30d'] if view['refunds_30d'] is not None else '—'}</div></div>
                <div><div class="rl-kv-label">Avg ticket</div><div class="rl-kv-value">{f"₹{view['avg_ticket_size']:,.2f}" if view['avg_ticket_size'] is not None else '—'}</div></div>
            </div>
        </div>
        <div class="rl-panel">
            <div class="rl-panel-label">Risk assessment</div>
            <div class="rl-case-id">CASE #{case_id_from_event(view['event_id'])}{f" &middot; {view['order_id']}" if view['order_id'] else ""}</div>
            <div class="rl-score-row">
                <span class="rl-score-value">{score_display}</span>
                <span class="rl-score-tag" style="color:{risk_color}; background:{risk_color}1A;">{risk_label}</span>
            </div>
            {risk_scale_html(view["risk_score"])}
            <div style="margin-top:18px;">
                <div class="rl-panel-label" style="margin-bottom:7px;">Final decision</div>
                {decision_badge_html(view["decision"])}
            </div>
            {authority_strip_html(view["decision_reason"] or "")}
        </div>
        """
    )

    if view["explanation"] or view["top_factors"]:
        html_block(
            f"""
            <div class="rl-panel">
                <div class="rl-panel-label">Why this score?</div>
                {shap_bars_html(view["top_factors"])}
                <p style="margin-top:10px; color:var(--rl-text-dim); font-size:0.86rem; line-height:1.55;">{view['explanation'] or ''}</p>
            </div>
            """
        )

    if view["source"] == "agent_pipeline":
        agree = view["agent_proposal"] is not None and view["agent_proposal"].get("recommended_decision") == view["decision"]
        html_block(
            f"""
            <div class="rl-panel">
                <div class="rl-panel-label">Decision control</div>
                {compare_panel_html(view["agent_proposal"], view["decision"], view["decision_reason"] or "", agree if view["agent_proposal"] else None)}
            </div>
            """
        )

    st.caption(f"Audit event ID: `{view['event_id']}`")


# =============================================================================
# PAGE: Overview
# =============================================================================
def page_overview():
    render_top_bar(
        "Merchant Risk Intelligence",
        "Monitor merchant risk, investigate anomalies, and keep every AI-assisted decision accountable.",
        RAZORPAY_CONFIGURED,
    )

    events = get_all_events(conn, limit=1000)
    active_cases = len({e["merchant_id"] for e in events if e.get("merchant_id")})
    high_risk_merchants = len({e["merchant_id"] for e in events if e.get("merchant_id") and e.get("decision") in HIGH_SEVERITY_DECISIONS})
    review_queue = sum(1 for e in events if e.get("decision") in REVIEW_DECISIONS)
    agent_investigations = sum(1 for e in events if (e.get("source") or "rule_pipeline") == "agent_pipeline")

    html_block(
        f"""
        <div class="rl-kpi-row">
            {kpi_html("Active cases", active_cases, "Distinct merchants scored this session")}
            {kpi_html("High-risk merchants", high_risk_merchants, "Flagged or sent to manual review", accent=high_risk_merchants > 0)}
            {kpi_html("Review queue", review_queue, "Events awaiting or requiring review")}
            {kpi_html("Agent investigations", agent_investigations, "Runs via the LLM agent pipeline")}
            {kpi_html("Audit events", len(events), "Total logged decisions")}
        </div>
        """
    )

    scored = [e for e in events if e.get("risk_score") is not None and e.get("timestamp_utc")]
    col_main, col_dist = st.columns([1.6, 1])
    with col_main:
        with st.container(border=True, key="panel_risk_activity"):
            html_block('<div class="rl-panel-label">Risk activity</div>')
            if len(scored) >= 3:
                df = pd.DataFrame(scored)[["timestamp_utc", "risk_score", "decision"]]
                df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
                st.altair_chart(risk_activity_chart(df), use_container_width=True)
            else:
                st.caption(
                    f"Not enough scored events yet to plot risk over time (need at least 3, have {len(scored)}). "
                    "Run a few investigations to populate this chart -- shown below instead is the real decision "
                    "breakdown so far."
                )
    with col_dist:
        counts = {}
        for e in events:
            d = e.get("decision")
            if d:
                counts[d] = counts.get(d, 0) + 1
        with st.container(border=True, key="panel_risk_distribution"):
            html_block('<div class="rl-panel-label">Risk distribution</div>')
            st.altair_chart(risk_distribution_chart(counts), use_container_width=True)

    with st.container(border=True, key="panel_recent_investigations"):
        html_block('<div class="rl-panel-label">Recent investigations</div>')
        if not events:
            html_block(empty_state_html("No investigations yet.", "Run your first merchant risk assessment in Investigations or Live Agent to begin."))
        else:
            recent = sorted(events, key=lambda e: e.get("timestamp_utc") or "", reverse=True)[:8]
            rows = []
            for e in recent:
                view = extract_case_view(e)
                agent_dec = (view["agent_proposal"] or {}).get("recommended_decision", "—") if view["source"] == "agent_pipeline" else "—"
                rows.append(
                    {
                        "Case": case_id_from_event(view["event_id"]),
                        "Merchant": view["merchant_id"] or "—",
                        "Risk": f"{view['risk_score']:.2f}" if view["risk_score"] is not None else "—",
                        "Decision": (view["decision"] or "—").replace("_", " ").title(),
                        "Agent": agent_dec.replace("_", " ").title() if agent_dec != "—" else "—",
                        "Gate": (view["decision"] or "—").replace("_", " ").title(),
                        "Status": "Verified" if view["decision"] else "—",
                        "Time": relative_time(view["timestamp_utc"]),
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    col_health, col_story = st.columns([1, 1.3])
    with col_health:
        html_block(
            f"""
            <div class="rl-panel">
                <div class="rl-panel-label">System health</div>
                {status_row_html([
                    ("Risk engine", STATUS_ONLINE),
                    ("Model service", STATUS_ONLINE),
                    ("Agent service (Groq)", STATUS_ONLINE if GROQ_CONFIGURED else STATUS_OFFLINE),
                    ("Razorpay test API", STATUS_CONNECTED if RAZORPAY_CONFIGURED else STATUS_OFFLINE),
                    ("Audit log", STATUS_ACTIVE),
                ])}
            </div>
            """
        )
    with col_story:
        html_block(
            f"""
            <div class="rl-panel">
                <div class="rl-panel-label">How a decision is made</div>
                {workflow_html(["Transaction", "Merchant context", "Risk model", "Explainability", "AI investigation", "Deterministic gate", "Audit trail"], 7)}
                <p style="color:var(--rl-text-dim); font-size:0.85rem; line-height:1.55; margin-top:2px;">
                    An AI agent may investigate, score, and recommend a decision on any merchant transaction. It never has
                    final authority. Every recommendation is checked against <b style="color:var(--rl-text);">{GATE_VERSION}</b>
                    before it counts as a decision, and every decision is written to an append-only audit trail.
                </p>
            </div>
            """
        )


# =============================================================================
# PAGE: Investigations
# =============================================================================
def page_investigations():
    render_top_bar("Investigations", "Review merchant risk cases and understand why each decision was made.", RAZORPAY_CONFIGURED)

    tab_cases, tab_new = st.tabs(["Case table", "New investigation"])

    with tab_new:
        st.subheader("Score a merchant snapshot")
        html_block(workflow_html(INVESTIGATION_STEPS, 0))

        sample_df = load_sample_merchants()
        use_sample = st.checkbox("Load a merchant snapshot from the dataset", value=True)
        if use_sample and not sample_df.empty:
            idx = st.selectbox(
                "Select a merchant snapshot", options=sample_df.index,
                format_func=lambda i: f"Merchant {sample_df.loc[i, 'merchant_id']} (recorded outcome: {'risky' if sample_df.loc[i, 'is_risky'] == 1 else 'not risky'})",
            )
            defaults = sample_df.loc[idx].to_dict()
        else:
            defaults = {
                "merchant_id": "manual-entry", "account_age_days": 365, "kyc_status": "complete",
                "business_category": "services", "daily_txn_volume": 10000.0, "avg_30d_txn_volume": 10000.0,
                "total_txns_30d": 200, "chargebacks_30d": 1, "refunds_30d": 5, "avg_ticket_size": 50.0,
            }

        c1, c2, c3 = st.columns(3)
        with c1:
            account_age_days = st.number_input("Account age (days)", value=float(defaults["account_age_days"]), min_value=0.0)
            kyc_status = st.selectbox("KYC status", ["complete", "incomplete"], index=0 if defaults["kyc_status"] == "complete" else 1)
            business_category = st.selectbox("Business category", BUSINESS_CATEGORIES, index=BUSINESS_CATEGORIES.index(defaults["business_category"]) if defaults["business_category"] in BUSINESS_CATEGORIES else 0)
        with c2:
            daily_txn_volume = st.number_input("Today's transaction volume", value=float(defaults["daily_txn_volume"]))
            avg_30d_txn_volume = st.number_input("30-day average volume", value=float(defaults["avg_30d_txn_volume"]))
            total_txns_30d = st.number_input("Total transactions (30d)", value=float(defaults["total_txns_30d"]), min_value=1.0)
        with c3:
            chargebacks_30d = st.number_input("Chargebacks (30d)", value=float(defaults["chargebacks_30d"]), min_value=0.0)
            refunds_30d = st.number_input("Refunds (30d)", value=float(defaults["refunds_30d"]), min_value=0.0)
            avg_ticket_size = st.number_input("Average ticket size", value=float(defaults["avg_ticket_size"]))

        if st.button("Run investigation", type="primary"):
            record = {
                "merchant_id": defaults.get("merchant_id", "manual-entry"), "account_age_days": account_age_days,
                "kyc_status": kyc_status, "business_category": business_category, "daily_txn_volume": daily_txn_volume,
                "avg_30d_txn_volume": avg_30d_txn_volume, "total_txns_30d": total_txns_30d,
                "chargebacks_30d": chargebacks_30d, "refunds_30d": refunds_30d, "avg_ticket_size": avg_ticket_size,
            }
            result = score_record(record, model, explainer, conn)
            html_block(workflow_html(INVESTIGATION_STEPS, 6 if result["risk_score"] is not None else 4))
            st.success(f"Case #{case_id_from_event(result['event_id'])} scored -- see it in the Case table tab.")
            st.session_state["_last_case_event_id"] = result["event_id"]

    with tab_cases:
        events = get_all_events(conn, limit=500)
        if not events:
            html_block(empty_state_html("No investigations yet.", "Run your first merchant risk assessment in the New Investigation tab to begin."))
            return

        views = [extract_case_view(e) for e in events]

        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            search = st.text_input("Search case, merchant, or order ID")
        with fcol2:
            decision_filter = st.multiselect("Decision", sorted({v["decision"] for v in views if v["decision"]}))
        with fcol3:
            risk_filter = st.multiselect("Risk level", ["Low risk", "Elevated risk", "High risk", "Unscored"])
        with fcol4:
            source_filter = st.multiselect("Pipeline", sorted({v["source"] for v in views}))

        def matches(v):
            if search:
                needle = search.lower()
                haystack = " ".join(str(x) for x in [case_id_from_event(v["event_id"]), v["merchant_id"], v["order_id"], v["event_id"]] if x).lower()
                if needle not in haystack:
                    return False
            if decision_filter and v["decision"] not in decision_filter:
                return False
            if risk_filter and risk_label_for_score(v["risk_score"])[0] not in risk_filter:
                return False
            if source_filter and v["source"] not in source_filter:
                return False
            return True

        filtered = [v for v in views if matches(v)]
        filtered.sort(key=lambda v: v["timestamp_utc"] or "", reverse=True)

        table_rows = []
        for v in filtered:
            agent_dec = (v["agent_proposal"] or {}).get("recommended_decision") if v["source"] == "agent_pipeline" else None
            primary_driver = "—"
            if v["top_factors"]:
                primary_driver = max(v["top_factors"], key=lambda f: abs(f["shap_value"]))["feature"]
            table_rows.append(
                {
                    "Case ID": case_id_from_event(v["event_id"]),
                    "Merchant": v["merchant_id"] or "—",
                    "Order": v["order_id"] or "—",
                    "Risk score": v["risk_score"] if v["risk_score"] is not None else None,
                    "Risk level": risk_label_for_score(v["risk_score"])[0],
                    "Primary driver": primary_driver,
                    "Agent": (agent_dec or "—").replace("_", " ").title() if agent_dec else "—",
                    "Gate": (v["decision"] or "—").replace("_", " ").title(),
                    "Final decision": (v["decision"] or "—").replace("_", " ").title(),
                    "Updated": relative_time(v["timestamp_utc"]),
                }
            )

        st.caption(f"{len(filtered)} of {len(views)} cases shown.")
        table_df = pd.DataFrame(table_rows)
        event = st.dataframe(
            table_df, hide_index=True, width="stretch",
            on_select="rerun", selection_mode="single-row", key="cases_table",
        )

        selected_rows = event.selection.rows if event and event.selection else []
        if selected_rows:
            selected_view = filtered[selected_rows[0]]
            st.divider()
            st.markdown(f"#### Case #{case_id_from_event(selected_view['event_id'])}")
            render_case_detail(selected_view)
        else:
            st.caption("Select a row above to open the full case detail.")


# =============================================================================
# PAGE: Live Agent
# =============================================================================
def page_live_agent():
    render_top_bar("Live risk investigation", "Razorpay test environment -- a real order, a real LLM-driven agent, a bounded final decision.", RAZORPAY_CONFIGURED)

    missing_config = []
    if not GROQ_CONFIGURED:
        missing_config.append("GROQ_API_KEY")
    if not RAZORPAY_CONFIGURED:
        missing_config.append("RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET")

    if missing_config:
        html_block(
            empty_state_html(
                "Agent service unavailable",
                f"Missing configuration: {', '.join(missing_config)}. Copy <code>.env.example</code> to <code>.env</code> "
                "and set your own keys to enable live investigations. Overview, Investigations, Models, and Audit Trail all work without it.",
            )
        )
        return

    c1, c2 = st.columns(2)
    with c1:
        agent_merchant_id = st.text_input("Merchant ID", value="live-demo-merchant-1")
    with c2:
        agent_amount = st.number_input("Transaction amount (INR)", value=5000.0, min_value=1.0)

    if st.button("Run investigation", type="primary", key="agent_run_btn"):
        result, error_message = None, None
        with st.status("Creating Razorpay test-mode order and running the risk agent...", expanded=True) as status_box:
            try:
                result = run_agentic_scoring(agent_merchant_id, agent_amount, model, explainer, conn)
            except Exception as exc:  # noqa: BLE001 -- surface any failure to the console, don't crash the app
                error_message = str(exc)
                status_box.update(label="Investigation failed", state="error")
            else:
                status_box.write(f"Order created: `{result['razorpay_order_id']}`")
                status_box.write(f"Agent completed {len(result['trace'])} tool call(s).")
                status_box.write(f"Gate decision: {result['gated_decision']}")
                status_box.update(label="Investigation complete", state="complete")
        st.session_state["_last_agent_result"] = result
        st.session_state["_last_agent_error"] = error_message

    result = st.session_state.get("_last_agent_result")
    error_message = st.session_state.get("_last_agent_error")

    if error_message:
        html_block(empty_state_html("Razorpay API connection failed", f"The test order could not be created or the investigation failed to complete.<br><code>{error_message}</code>"))

    if result:
        html_block(workflow_html(AGENT_STEPS, 6))
        left, center, right = st.columns([1, 1.3, 1.1])

        with left:
            html_block(
                f"""
                <div class="rl-panel">
                    <div class="rl-panel-label">Transaction</div>
                    {order_panel_html(result)}
                </div>
                <div class="rl-panel">
                    <div class="rl-panel-label">Merchant</div>
                    <div class="rl-kv-grid">
                        <div><div class="rl-kv-label">Merchant ID</div><div class="rl-kv-value">{result.get('merchant_id') or '—'}</div></div>
                    </div>
                </div>
                """
            )

        with center:
            steps_html = []
            for step in result["trace"]:
                status = step.get("status", "success")
                dot_color = "#1E9E5A" if status == "success" else "#D93025"
                ts = step.get("timestamp")
                time_label = ts.split("T")[1][:8] if isinstance(ts, str) and "T" in ts else "--:--:--"
                duration = step.get("duration_ms")
                duration_label = f" &middot; {duration:.0f}ms" if isinstance(duration, (int, float)) else ""
                tool = step.get("tool", "unknown")
                res = step.get("result", {})
                if isinstance(res, dict) and "error" in res:
                    detail = f"error: {res['error']}"
                elif tool == "score_transaction_risk":
                    detail = f"risk_score = {res.get('risk_score'):.4f}" if res.get("risk_score") is not None else "no score returned"
                elif tool == "explain_transaction_risk":
                    detail = (res.get("explanation") or "")[:120]
                elif tool == "get_merchant_context":
                    detail = f"kyc={res.get('kyc_status')}, age={res.get('account_age_days')}d"
                elif tool == "get_recent_audit_history":
                    detail = f"{len(res.get('past_decisions', []))} prior decision(s)"
                else:
                    detail = str(res)[:120]
                steps_html.append(
                    f'<div class="rl-tl-step"><div class="rl-tl-rail"><div class="rl-tl-dot" style="background:{dot_color};"></div><div class="rl-tl-line"></div></div>'
                    f'<div class="rl-tl-body"><div class="rl-tl-title">{tool}</div><div class="rl-tl-detail">{detail}</div>'
                    f'<div class="rl-tl-time">{time_label}{duration_label}</div></div></div>'
                )
            html_block(
                f"""
                <div class="rl-panel">
                    <div class="rl-panel-label">Agent investigation</div>
                    <div class="rl-timeline">{''.join(steps_html)}</div>
                </div>
                """
            )
            with st.expander("Raw trace (full tool arguments and results)"):
                for i, step in enumerate(result["trace"], 1):
                    st.write(f"**Step {i}: `{step['tool']}`**")
                    st.json({"arguments": step["arguments"], "result": step["result"]})

        with right:
            risk_label, risk_color = risk_label_for_score(result["risk_score"])
            score_display = f"{result['risk_score']:.2f}" if result["risk_score"] is not None else "--"
            html_block(
                f"""
                <div class="rl-panel">
                    <div class="rl-panel-label">Decision control</div>
                    <div class="rl-score-row"><span class="rl-score-value">{score_display}</span>
                        <span class="rl-score-tag" style="color:{risk_color}; background:{risk_color}1A;">{risk_label}</span></div>
                    {risk_scale_html(result["risk_score"])}
                    {compare_panel_html(result["agent_proposal"], result["gated_decision"], result["gated_reason"], result["agent_and_gate_agree"])}
                </div>
                <div class="rl-panel">
                    <div class="rl-panel-label">Final decision</div>
                    {decision_badge_html(result["gated_decision"])}
                    {authority_strip_html(result["gated_reason"])}
                    <p style="margin-top:12px; font-size:0.8rem; color:var(--rl-text-dim);">&#10003; Verified by deterministic gate &nbsp;&#183;&nbsp; &#10003; Audit event committed</p>
                </div>
                """
            )
            if result["explanation"] or result["top_factors"]:
                html_block(
                    f"""
                    <div class="rl-panel">
                        <div class="rl-panel-label">Why this score?</div>
                        {shap_bars_html(result["top_factors"])}
                        <p style="margin-top:10px; color:var(--rl-text-dim); font-size:0.85rem;">{result['explanation']}</p>
                    </div>
                    """
                )
        st.caption(f"Audit event ID: `{result['event_id']}`")


# =============================================================================
# PAGE: Models
# =============================================================================
def page_models():
    render_top_bar("Model performance", "XGBoost vs. the logistic-regression baseline, evaluated on a held-out test set.", RAZORPAY_CONFIGURED)

    if not (os.path.exists(METRICS_PATH) and os.path.exists(CHART_DATA_PATH)):
        html_block(empty_state_html("No metrics found.", "Run <code>python3 model/train.py</code> to generate model metrics and charts."))
        return

    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    with open(CHART_DATA_PATH) as f:
        chart_data = json.load(f)

    xgb, base = metrics["xgboost"], metrics["baseline_logistic_regression"]

    html_block(f'<div class="rl-panel"><div class="rl-panel-label">XGBoost vs. Logistic Regression</div>{model_comparison_table_html(xgb, base)}</div>')
    st.caption(f"Test set: {metrics['test_rows']} rows, {metrics['test_positive_rate']:.1%} positive (risky) rate. Thresholds tuned on a separate validation split.")

    col_roc, col_cm = st.columns(2)
    with col_roc:
        with st.container(border=True, key="panel_roc_curve"):
            html_block('<div class="rl-panel-label">ROC curve</div>')
            st.altair_chart(roc_chart(chart_data["roc_curve"]), use_container_width=True)
    with col_cm:
        with st.container(border=True, key="panel_confusion_matrix"):
            html_block(f'<div class="rl-panel-label">Confusion matrix &middot; XGBoost (threshold {xgb["threshold"]:.2f})</div>')
            st.altair_chart(confusion_matrix_chart(chart_data["confusion_matrix"]), use_container_width=True)

    with st.container(border=True, key="panel_shap_global"):
        html_block('<div class="rl-panel-label">Global feature importance (SHAP, test set)</div>')
        st.altair_chart(shap_global_chart(chart_data["shap_global_importance"]), use_container_width=True)


# =============================================================================
# PAGE: Audit Trail
# =============================================================================
def page_audit_trail():
    render_top_bar("Audit trail", "Every AI action, gate decision, and final outcome is traceable.", RAZORPAY_CONFIGURED)

    events = get_all_events(conn, limit=500)
    if not events:
        html_block(empty_state_html("No audit events recorded.", "Run an investigation in Investigations or Live Agent to populate the audit trail."))
        return

    df_events = pd.DataFrame(events)
    if "source" not in df_events.columns:
        df_events["source"] = "rule_pipeline"
    df_events["source"] = df_events["source"].fillna("rule_pipeline")

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        decision_filter = st.multiselect("Filter by decision", sorted(df_events["decision"].dropna().unique().tolist()))
    with fcol2:
        source_filter = st.multiselect("Filter by pipeline", sorted(df_events["source"].dropna().unique().tolist()))
    with fcol3:
        merchant_search = st.text_input("Search merchant ID")

    filtered = df_events
    if decision_filter:
        filtered = filtered[filtered["decision"].isin(decision_filter)]
    if source_filter:
        filtered = filtered[filtered["source"].isin(source_filter)]
    if merchant_search:
        filtered = filtered[filtered["merchant_id"].astype(str).str.contains(merchant_search, case=False, na=False)]

    display_cols = ["timestamp_utc", "source", "merchant_id", "risk_score", "decision", "decision_reason"]
    st.dataframe(filtered[display_cols], hide_index=True, width="stretch")
    st.caption(f"{len(filtered)} of {len(df_events)} audit events shown. Every row is written once and never edited or deleted.")

    with st.expander("Inspect a single event's full record"):
        if not filtered.empty:
            selected_id = st.selectbox("Event ID", filtered["event_id"].tolist(), format_func=lambda e: f"{case_id_from_event(e)} ({e})")
            full_event = filtered[filtered["event_id"] == selected_id].iloc[0].to_dict()
            for json_field in ("input_snapshot", "top_factors", "agent_proposal", "agent_trace"):
                if full_event.get(json_field):
                    try:
                        full_event[json_field] = json.loads(full_event[json_field])
                    except (TypeError, json.JSONDecodeError):
                        pass
            st.json(full_event)
        else:
            st.caption("No events match the current filters.")


# =============================================================================
# Sidebar + routing
# =============================================================================
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "risklens_logo.svg")
if os.path.exists(LOGO_PATH):
    st.logo(LOGO_PATH, size="large")

pages = [
    st.Page(page_overview, title="Overview", icon=":material/dashboard:", default=True),
    st.Page(page_investigations, title="Investigations", icon=":material/search:"),
    st.Page(page_live_agent, title="Live Agent", icon=":material/bolt:"),
    st.Page(page_models, title="Models", icon=":material/insights:"),
    st.Page(page_audit_trail, title="Audit Trail", icon=":material/fact_check:"),
]
pg = st.navigation(pages, position="sidebar")

render_sidebar_status(
    [
        ("Risk engine", STATUS_ONLINE),
        ("Agent service", STATUS_ONLINE if GROQ_CONFIGURED else STATUS_OFFLINE),
        ("Razorpay API", STATUS_CONNECTED if RAZORPAY_CONFIGURED else STATUS_OFFLINE),
        ("Audit log", STATUS_ACTIVE),
    ]
)

pg.run()
