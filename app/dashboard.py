"""
RiskLens -- Merchant Risk Intelligence Platform.

Six pages, routed with Streamlit's native sidebar navigation (st.navigation)
rather than top tabs, matching a real desktop fintech application shell:
  1. Overview        -- command center: KPIs, risk activity/distribution,
                         recent investigations, system health.
  2. Investigations  -- searchable/filterable case table with a full case
                         detail panel (merchant context, risk assessment,
                         SHAP, decision control, audit reference).
  3. Batch Scoring    -- upload a CSV of many merchants (or sample from the
                         dataset) and score the whole portfolio in one pass,
                         through the exact same pipeline as a single case.
  4. Live Agent       -- creates a REAL Razorpay test-mode order and runs the
                         LLM-driven risk agent, in a three-column
                         transaction / execution / decision layout.
  5. Models           -- held-out test-set performance with real, interactive
                         (Altair) ROC, confusion-matrix, and SHAP charts.
  6. Audit Trail      -- every decision this session has made, filterable
                         and individually inspectable.

Run:
    streamlit run app/dashboard.py
"""

import html
import json
import os
import secrets
import sys
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.case_qa import answer_case_question
from agent.risk_agent import summarize_tool_result
from agent_pipeline import run_agentic_scoring
from app.theme import (
    DECISION_STYLE,
    STATUS_ACTIVE,
    STATUS_CONNECTED,
    STATUS_OFFLINE,
    STATUS_ONLINE,
    agent_recommendation_card_html,
    authority_strip_html,
    case_id_from_event,
    compare_panel_html,
    confusion_matrix_chart,
    decision_badge_html,
    decision_volume_chart,
    empty_state_html,
    gate_decision_card_html,
    html_block,
    inject_theme,
    kpi_html,
    model_comparison_table_html,
    order_panel_html,
    override_banner_html,
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
    verdict_banner_html,
    workflow_html,
)
from audit.audit_log import get_all_events, get_all_overrides, get_connection, get_overrides_for_event, log_override
from explainability.explain import RiskExplainer
from features.features import BUSINESS_CATEGORIES, RAW_REQUIRED_COLUMNS, transform_features
from gating.decision_engine import DECISION_CLEAR, DECISION_ESCALATE, DECISION_FLAG, DECISION_MANUAL_REVIEW, GATE_VERSION
from model.feedback import TEST_SNAPSHOT_PATH, promote_candidate, train_candidate_with_feedback
from model.train import load_and_split
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


def _safe_metric_html(value, formatter, default="—"):
    """Format+escape a case-detail field that may hold unvalidated data.

    Batch Scoring lets someone upload an arbitrary CSV straight into
    score_record(); a row that fails find_missing_or_invalid isn't
    discarded -- it's still logged and still routed to a viewable case
    (that's what "needs_manual_review" means), so account_age_days /
    daily_txn_volume / avg_30d_txn_volume / avg_ticket_size can legitimately
    be a non-numeric string by the time a reviewer opens this panel. Without
    this guard, `f"{value:.0f}"` on such a row raises ValueError and takes
    down the entire case-detail render. html.escape() on the fallback path
    also closes the same stored-XSS hole merchant_id was fixed for below --
    a non-numeric value here is exactly as attacker-reachable as merchant_id.
    """
    if value is None:
        return default
    try:
        return formatter(value)
    except (ValueError, TypeError):
        return html.escape(str(value))


def merchant_context_display_values(view: dict) -> dict:
    """Escaped/formatted strings for the case-detail Merchant context panel.

    Pulled out of render_case_detail as a plain function (no Streamlit call
    in it) specifically so this escaping logic is unit-testable directly --
    the earlier stored-XSS fixes in this file had no test coverage at all,
    which is exactly how the sibling gap this function closes (kyc_status/
    business_category/chargebacks_30d/refunds_30d never being escaped, only
    merchant_id) went unnoticed for as long as it did.

    merchant_id is free text a user typed (New investigation's manual entry,
    a batch CSV upload, or the Live Agent page) and is stored verbatim in
    the audit log -- it's never validated against a fixed format the way
    kyc_status/business_category are. render_case_detail renders this via
    html_block(unsafe_allow_html=True), so without escaping, a merchant_id
    like <img src=x onerror=...> would execute as live HTML/JS for every
    future reviewer who opens this case -- a stored XSS via the most
    ordinary possible input.

    kyc_status/business_category/chargebacks_30d/refunds_30d are exactly as
    reachable via the same batch-CSV path as merchant_id -- a row whose
    business_category fails the allow-list check in find_missing_or_invalid
    is still logged and still displayed here, not discarded. chargebacks_30d
    /refunds_30d are range-checked (non-negative, finite, and each no
    greater than total_txns_30d) but that's a numeric-plausibility check,
    not an HTML-safety one -- str(2) is exactly as safe to interpolate raw
    as str("<img src=x onerror=...>"), so both still need the same escaping
    here even though the former "looks" harmless.
    """
    return {
        "merchant": html.escape(str(view['merchant_id'])) if view['merchant_id'] else '—',
        "kyc": html.escape(str(view['kyc_status']).title()) if view['kyc_status'] else '—',
        "category": html.escape(str(view['business_category']).title()) if view['business_category'] else '—',
        "chargebacks": html.escape(str(view['chargebacks_30d'])) if view['chargebacks_30d'] is not None else '—',
        "refunds": html.escape(str(view['refunds_30d'])) if view['refunds_30d'] is not None else '—',
        "account_age": _safe_metric_html(view['account_age_days'], lambda v: f"{v:.0f} days"),
        "daily_txn": _safe_metric_html(view['daily_txn_volume'], lambda v: f"₹{v:,.2f}"),
        "avg_30d": _safe_metric_html(view['avg_30d_txn_volume'], lambda v: f"₹{v:,.2f}"),
        "avg_ticket": _safe_metric_html(view['avg_ticket_size'], lambda v: f"₹{v:,.2f}"),
    }


def render_case_detail(view: dict):
    risk_label, risk_color = risk_label_for_score(view["risk_score"])
    score_display = f"{view['risk_score']:.2f}" if view["risk_score"] is not None else "--"
    ctx = merchant_context_display_values(view)

    html_block(
        f"""
        <div class="rl-panel">
            <div class="rl-panel-label">Merchant context</div>
            <div class="rl-kv-grid">
                <div><div class="rl-kv-label">Merchant</div><div class="rl-kv-value">{ctx['merchant']}</div></div>
                <div><div class="rl-kv-label">Account age</div><div class="rl-kv-value">{ctx['account_age']}</div></div>
                <div><div class="rl-kv-label">KYC</div><div class="rl-kv-value">{ctx['kyc']}</div></div>
                <div><div class="rl-kv-label">Category</div><div class="rl-kv-value">{ctx['category']}</div></div>
                <div><div class="rl-kv-label">Txn volume</div><div class="rl-kv-value">{ctx['daily_txn']}</div></div>
                <div><div class="rl-kv-label">30d average</div><div class="rl-kv-value">{ctx['avg_30d']}</div></div>
                <div><div class="rl-kv-label">Chargebacks</div><div class="rl-kv-value">{ctx['chargebacks']}</div></div>
                <div><div class="rl-kv-label">Refunds</div><div class="rl-kv-value">{ctx['refunds']}</div></div>
                <div><div class="rl-kv-label">Avg ticket</div><div class="rl-kv-value">{ctx['avg_ticket']}</div></div>
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

    render_override_section(view)

    report_overrides = get_overrides_for_event(conn, view["event_id"])
    render_case_qa_section(view, report_overrides)

    st.download_button(
        "Download case report",
        data=case_report_text(view, report_overrides),
        file_name=f"{case_id_from_event(view['event_id'])}_report.txt",
        mime="text/plain",
        key=f"download_report_{view['event_id']}",
    )
    st.caption(f"Audit event ID: `{view['event_id']}`")


def render_case_qa_section(view: dict, overrides: list):
    """
    Read-only natural-language Q&A grounded in this one case's own recorded
    data (see agent/case_qa.py's module docstring for why this needs no
    gate in front of it, unlike the Live Agent's proposal: it has no tools
    and cannot take or recommend any action, so there is nothing here for a
    deterministic check to need to catch -- the worst outcome is a badly
    phrased explanation, never an unauthorized decision).
    """
    event_id = view["event_id"]
    with st.expander("Ask about this case"):
        if not GROQ_CONFIGURED:
            st.caption(
                "Needs GROQ_API_KEY configured (see .env.example) to answer questions -- every "
                "other part of RiskLens works fully without it; this one feature just needs an LLM."
            )
            return

        st.caption(
            "Ask in plain language -- answers are grounded only in this case's own recorded data "
            "above, and this cannot override or change anything (use the control above for that)."
        )

        history_key = f"_case_qa_history_{event_id}"
        history = st.session_state.setdefault(history_key, [])

        for turn in history:
            with st.chat_message(turn["role"]):
                st.write(turn["content"])

        question = st.chat_input("e.g. Why was this flagged?", key=f"qa_input_{event_id}")
        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = answer_case_question(question, case_report_text(view, overrides), history=history)
                    except Exception as exc:  # noqa: BLE001 -- surface the failure, don't crash the page
                        answer = f"Couldn't get an answer: {exc}"
                st.write(answer)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})


def case_report_text(view: dict, overrides: list) -> str:
    """
    A plain-text, self-contained export of one case -- everything shown on
    the case detail panel, in one file a reviewer can save, email, or attach
    to a compliance ticket without needing to open RiskLens itself. Exists
    because the audit trail being real and queryable (Section 4.6) is only
    useful to a reviewer if a single case can also leave the app in a form
    someone outside it can read.
    """
    risk_label, _ = risk_label_for_score(view["risk_score"])
    score_display = f"{view['risk_score']:.3f}" if view["risk_score"] is not None else "not scored"
    # Same unvalidated-batch-upload data render_case_detail's Merchant context
    # panel guards against with _safe_metric_html: account_age_days can be a
    # non-numeric value on a row that failed find_missing_or_invalid but was
    # still logged (needs_manual_review, not discarded). This report is
    # plain text, not HTML, so there's no XSS angle here -- but the bare
    # f"{value:.0f}" format spec still raises ValueError on a non-numeric
    # value and used to crash the whole report/download, which is exactly
    # the class of bug _safe_metric_html was added to prevent, just missed
    # in this sibling function.
    if view["account_age_days"] is None:
        account_age_display = "unknown"
    else:
        try:
            account_age_display = f"{view['account_age_days']:.0f} days"
        except (ValueError, TypeError):
            account_age_display = str(view["account_age_days"])
    lines = [
        "RISKLENS CASE REPORT",
        "=====================",
        f"Case ID:      {case_id_from_event(view['event_id'])}",
        f"Event ID:     {view['event_id']}",
        f"Generated:    {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Recorded:     {view['timestamp_utc'] or 'unknown'}",
        f"Pipeline:     {view['source']}",
        "",
        "MERCHANT",
        "--------",
        f"Merchant ID:        {view['merchant_id'] or 'unknown'}",
        f"Account age:        {account_age_display}",
        f"KYC status:         {view['kyc_status'] or 'unknown'}",
        f"Business category:  {view['business_category'] or 'unknown'}",
        f"Daily txn volume:   {view['daily_txn_volume'] if view['daily_txn_volume'] is not None else 'unknown'}",
        f"30-day average:     {view['avg_30d_txn_volume'] if view['avg_30d_txn_volume'] is not None else 'unknown'}",
        f"Chargebacks (30d):  {view['chargebacks_30d'] if view['chargebacks_30d'] is not None else 'unknown'}",
        f"Refunds (30d):      {view['refunds_30d'] if view['refunds_30d'] is not None else 'unknown'}",
        f"Avg ticket size:    {view['avg_ticket_size'] if view['avg_ticket_size'] is not None else 'unknown'}",
        "",
        "RISK ASSESSMENT",
        "---------------",
        f"Risk score:     {score_display} ({risk_label})",
        f"Final decision: {(view['decision'] or 'unknown').replace('_', ' ').title()}",
        f"Reason:         {view['decision_reason'] or 'n/a'}",
        "",
    ]
    if view.get("top_factors"):
        lines.append("Top contributing factors:")
        for f in view["top_factors"]:
            lines.append(f"  - {f.get('feature')}: SHAP {f.get('shap_value'):+.4f}")
        lines.append("")
    if view.get("explanation"):
        lines += ["Explanation:", f"  {view['explanation']}", ""]

    if view["source"] == "agent_pipeline" and view.get("agent_proposal"):
        proposal = view["agent_proposal"]
        agree = proposal.get("recommended_decision") == view["decision"]
        lines += [
            "AGENT PROPOSAL",
            "--------------",
            f"Recommended decision: {(proposal.get('recommended_decision') or 'unknown').replace('_', ' ').title()}",
            f"Agreed with gate:     {'yes' if agree else 'no'}",
            f"Reasoning:            {proposal.get('reasoning', 'n/a')}",
            "",
        ]

    lines += ["HUMAN OVERRIDE HISTORY", "----------------------"]
    if not overrides:
        lines.append("No overrides recorded for this case.")
    else:
        for o in overrides:
            lines += [
                f"[{o['timestamp_utc']}] {o['original_decision']} -> {o['overridden_decision']}"
                + (f"  (reviewer: {o['reviewer']})" if o.get("reviewer") else ""),
                f"  Reason: {o['reason']}",
            ]
    lines += [
        "",
        "---",
        "This report is a point-in-time export of an append-only audit record.",
        "The decision above cannot be edited -- only overridden with a new, separately logged record.",
    ]
    return "\n".join(lines)


def render_override_section(view: dict):
    """
    Lets a human reviewer correct a case's outcome after the fact -- the
    deterministic gate is authoritative at decision time, but a reviewer who
    later spots something the gate/agent missed (or a false positive) should
    be able to say so, on the record. The original decision is never edited;
    this only ever adds a new human_overrides row (see audit_log.py), so the
    case detail below always shows both "what the system decided" and "what
    a human later corrected it to, and why" -- and that correction is exactly
    the labeled signal a future model retrain would want to learn from.
    """
    event_id = view["event_id"]
    overrides = get_overrides_for_event(conn, event_id)
    if overrides:
        html_block(override_banner_html(overrides))

    decision_options = [DECISION_CLEAR, DECISION_ESCALATE, DECISION_FLAG, DECISION_MANUAL_REVIEW]
    current_decision = overrides[0]["overridden_decision"] if overrides else view["decision"]
    default_index = decision_options.index(current_decision) if current_decision in decision_options else 0

    with st.expander("Override this decision" if not overrides else "Record another override"):
        st.caption(
            "For a reviewer who disagrees with the outcome above after reviewing the full case. "
            "This doesn't erase or edit the original decision -- it's logged alongside it, and "
            "becomes labeled feedback future retraining can learn from."
        )
        with st.form(key=f"override_form_{event_id}"):
            new_decision = st.selectbox(
                "Corrected decision",
                decision_options,
                index=default_index,
                format_func=lambda d: DECISION_STYLE.get(d, {"label": d})["label"],
                key=f"override_decision_{event_id}",
            )
            reason = st.text_area(
                "Reason for this override", key=f"override_reason_{event_id}",
                placeholder="e.g. Confirmed with the merchant directly -- the large transaction was a legitimate one-off invoice.",
            )
            submitted = st.form_submit_button("Submit override")
            if submitted:
                if not reason.strip():
                    st.error("Please give a reason -- it's what makes this useful as future training signal, not just a status flip.")
                else:
                    log_override(
                        conn,
                        event_id=event_id,
                        original_decision=current_decision,
                        overridden_decision=new_decision,
                        reason=reason.strip(),
                    )
                    st.success("Override recorded.")
                    st.rerun()


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
                st.altair_chart(risk_activity_chart(df), width="stretch")
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
            st.altair_chart(risk_distribution_chart(counts), width="stretch")

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
            # Jump the Case table's search box straight to this new case, so
            # "see it in the Case table tab" is actually true instead of
            # leaving the reviewer to scroll/search for it themselves. Safe
            # to set here (rather than raising the same StreamlitAPIException
            # fixed earlier on the Live Agent page) because this tab's code
            # runs BEFORE tab_cases's search widget is instantiated below, in
            # this same script run -- not after, like that earlier bug.
            st.session_state["investigations_search"] = case_id_from_event(result["event_id"])

    with tab_cases:
        events = get_all_events(conn, limit=500)
        if not events:
            html_block(empty_state_html("No investigations yet.", "Run your first merchant risk assessment in the New Investigation tab to begin."))
            return

        views = [extract_case_view(e) for e in events]

        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            search = st.text_input("Search case, merchant, or order ID", key="investigations_search")
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
# PAGE: Batch Scoring
# =============================================================================
BATCH_REQUIRED_COLUMNS = ["merchant_id"] + list(RAW_REQUIRED_COLUMNS)


def _batch_template_csv() -> str:
    example = {
        "merchant_id": "merchant_1001", "account_age_days": 400, "kyc_status": "complete",
        "business_category": "services", "daily_txn_volume": 12000, "avg_30d_txn_volume": 10000,
        "total_txns_30d": 300, "chargebacks_30d": 1, "refunds_30d": 4, "avg_ticket_size": 40,
    }
    return pd.DataFrame([example])[BATCH_REQUIRED_COLUMNS].to_csv(index=False)


def page_batch_scoring():
    render_top_bar(
        "Batch scoring",
        "Score an entire portfolio of merchants in one pass -- upload a CSV, get every merchant risk-scored, gated, and ranked.",
        RAZORPAY_CONFIGURED,
    )

    html_block(
        """
        <div class="rl-panel">
            <div class="rl-panel-label">How this works</div>
            <p style="color:var(--rl-text-dim); font-size:0.86rem; line-height:1.6; margin:0;">
                Each row runs through the exact same pipeline as a single investigation -- the same
                model, the same SHAP explanation, the same deterministic gate -- and is written to the
                audit trail exactly like any other case. A batch run is never a shortcut around
                accountability; it's the same scoring, just applied to an entire portfolio at once
                (e.g. every merchant onboarded this week) instead of one at a time.
            </p>
        </div>
        """
    )

    c1, c2 = st.columns([3, 2])
    with c1:
        uploaded = st.file_uploader(
            "Merchant snapshot CSV",
            type=["csv"],
            help=f"Required columns: {', '.join(BATCH_REQUIRED_COLUMNS)}",
        )
    with c2:
        st.download_button(
            "Download CSV template",
            data=_batch_template_csv(),
            file_name="risklens_batch_template.csv",
            mime="text/csv",
        )
        demo_n = st.number_input(
            "...or score N sample merchants (no file needed)",
            min_value=0, max_value=100, value=0, step=5,
        )

    batch_df = None
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that CSV: {e}")
        else:
            missing_cols = [c for c in BATCH_REQUIRED_COLUMNS if c not in batch_df.columns]
            if missing_cols:
                st.error(
                    f"Missing required column(s): {', '.join(missing_cols)}. "
                    "Download the template above to see the expected format."
                )
                batch_df = None
    elif demo_n:
        sample = load_sample_merchants(n=int(demo_n))
        if not sample.empty:
            batch_df = sample[BATCH_REQUIRED_COLUMNS].copy()

    if batch_df is None or batch_df.empty:
        html_block(empty_state_html("No batch loaded yet", "Upload a CSV or pick a sample size above, then run the batch below."))
        return

    st.caption(f"{len(batch_df)} merchant(s) ready to score.")
    if st.button(f"Score all {len(batch_df)} merchants", type="primary"):
        progress = st.progress(0.0, text="Scoring...")
        results = []
        records = batch_df.to_dict("records")
        for i, record in enumerate(records):
            outcome = score_record(record, model, explainer, conn)
            primary_driver = "—"
            if outcome["top_factors"]:
                primary_driver = max(outcome["top_factors"], key=lambda f: abs(f["shap_value"]))["feature"]
            results.append(
                {
                    "Case ID": case_id_from_event(outcome["event_id"]),
                    "Merchant": record.get("merchant_id", "—"),
                    "Risk score": outcome["risk_score"],
                    "Risk level": risk_label_for_score(outcome["risk_score"])[0],
                    "Primary driver": primary_driver,
                    "Decision": (outcome["decision"] or "—").replace("_", " ").title(),
                    "Reason": outcome["decision_reason"],
                }
            )
            progress.progress((i + 1) / len(records), text=f"Scored {i + 1}/{len(records)}")
        progress.empty()
        st.session_state["_batch_results"] = pd.DataFrame(results)
        st.success(f"Scored {len(results)} merchants -- every one was also logged to the audit trail.")

    results_df = st.session_state.get("_batch_results")
    if results_df is not None and not results_df.empty:
        st.divider()
        st.markdown("#### Batch report")

        decision_counts = results_df["Decision"].value_counts().to_dict()
        kcols = st.columns(4)
        with kcols[0]:
            html_block(kpi_html("Merchants scored", len(results_df)))
        with kcols[1]:
            avg_score = results_df["Risk score"].dropna().mean()
            html_block(kpi_html("Average risk score", f"{avg_score:.2f}" if pd.notna(avg_score) else "--"))
        with kcols[2]:
            needs_review = sum(v for k, v in decision_counts.items() if k != "Clear")
            html_block(kpi_html("Need review or higher", needs_review, accent=needs_review > 0))
        with kcols[3]:
            top_decision = max(decision_counts, key=decision_counts.get) if decision_counts else "—"
            html_block(kpi_html("Most common outcome", top_decision))

        sort_desc = st.checkbox("Sort by risk score (highest first)", value=True)
        display_df = results_df.sort_values("Risk score", ascending=not sort_desc, na_position="last")
        st.dataframe(display_df, hide_index=True, width="stretch")

        st.download_button(
            "Download full batch report (CSV)",
            data=results_df.to_csv(index=False),
            file_name="risklens_batch_report.csv",
            mime="text/csv",
        )


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

    # The simulated merchant history (agent/merchant_context.py) is
    # deterministic on purpose -- same Merchant ID -> same simulated
    # profile every time, so a demo is reproducible. That's easy to
    # mistake for a bug if the field is left on its default and "Run
    # investigation" is clicked more than once: same ID in, same
    # everything out, including the agent's own reasoning text.
    # A fresh random-looking default per page visit, plus an explicit
    # "different merchant" button, make that obvious instead of
    # surprising.
    if "agent_merchant_id_input" not in st.session_state:
        st.session_state["agent_merchant_id_input"] = f"live-demo-merchant-{secrets.token_hex(3)}"

    def _randomize_merchant_id():
        # Streamlit forbids writing to a widget's own session_state key
        # after that widget has already been instantiated in the current
        # script run (raises StreamlitAPIException) -- a plain `if
        # st.button(...): st.session_state[key] = ...` block runs AFTER
        # the text_input above it has already been drawn, so it hit that
        # exact error. A button's on_click callback runs BEFORE the script
        # reruns and redraws any widgets, so updating the value there is
        # the correct, safe way to do this.
        st.session_state["agent_merchant_id_input"] = f"live-demo-merchant-{secrets.token_hex(3)}"

    c1, c2 = st.columns(2)
    with c1:
        agent_merchant_id = st.text_input("Merchant ID", key="agent_merchant_id_input")
        st.button("🎲 Try a different simulated merchant", on_click=_randomize_merchant_id)
    with c2:
        agent_amount = st.number_input("Transaction amount (INR)", value=5000.0, min_value=1.0)
    st.caption(
        "The transaction is a real Razorpay test-mode order, but the merchant's history (account age, "
        "chargebacks, KYC, etc.) is simulated -- and simulated **from the Merchant ID itself**, so the "
        "same ID always produces the same simulated profile, score, and reasoning on purpose (this makes "
        "demos reproducible). To see a different outcome, change the Merchant ID -- or click the button "
        "above for a fresh one -- before running again."
    )

    if st.button("Run investigation", type="primary", key="agent_run_btn"):
        result, error_message = None, None
        with st.status("Creating Razorpay test-mode order and running the risk agent...", expanded=True) as status_box:
            # Live reasoning feed: on_step fires in real time as the agent
            # investigates, so the reviewer watches it think turn by turn
            # instead of staring at a spinner until everything is done at once.
            def _on_step(event, _box=status_box):
                etype = event.get("type")
                if etype == "thinking":
                    _box.write(f"Turn {event['turn']}: agent is deciding its next move...")
                elif etype == "tool_call":
                    icon = "Error" if event.get("status") == "error" else "Done"
                    _box.write(f"[{icon}] `{event['tool']}` -- {event.get('summary', '')}")
                elif etype == "final":
                    decision_label = (event.get("decision") or "no proposal").replace("_", " ")
                    _box.write(f"Agent submitted its recommendation: **{decision_label}**")
                elif etype == "timeout":
                    _box.write("Agent did not finish within the allotted turns -- falling back to manual review.")

            try:
                result = run_agentic_scoring(agent_merchant_id, agent_amount, model, explainer, conn, on_step=_on_step)
            except Exception as exc:  # noqa: BLE001 -- surface any failure to the console, don't crash the app
                error_message = str(exc)
                status_box.update(label="Investigation failed", state="error")
            else:
                status_box.write(f"Order created: `{result['razorpay_order_id']}`")
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

        # The outcome, shown once, immediately, full-width -- not buried at the
        # bottom of whichever of several side-by-side columns happens to run
        # longest (see verdict_banner_html's docstring for why).
        html_block(verdict_banner_html(result))

        steps_html = []
        raw_trace_steps_html = []
        for i, step in enumerate(result["trace"], 1):
            status = step.get("status", "success")
            dot_color = "#1E9E5A" if status == "success" else "#D93025"
            ts = step.get("timestamp")
            time_label = ts.split("T")[1][:8] if isinstance(ts, str) and "T" in ts else "--:--:--"
            duration = step.get("duration_ms")
            duration_label = f" &middot; {duration:.0f}ms" if isinstance(duration, (int, float)) else ""
            tool = step.get("tool", "unknown")
            res = step.get("result", {})
            # Same one-line summary the live on_step feed showed while the
            # agent was still running -- kept in one place (risk_agent.py)
            # so the live view and this post-hoc view can never drift apart.
            detail = summarize_tool_result(tool, res)
            # `tool` is the LLM's own tool_call.function.name, taken verbatim
            # -- a hallucinated call to an undeclared name reaches here
            # unmodified (agent/tools.py raises ValueError("Unknown tool: "
            # + name) for it, which summarize_tool_result's first branch
            # then echoes straight into `detail` too, via the "error: ..."
            # prefix). The raw-trace <details> block just below this one
            # already escapes this exact same `tool` value -- it was only
            # ever missing here, in the always-visible timeline.
            tool_display = html.escape(str(tool))
            detail_display = html.escape(str(detail))
            steps_html.append(
                f'<div class="rl-tl-step"><div class="rl-tl-rail"><div class="rl-tl-dot" style="background:{dot_color};"></div><div class="rl-tl-line"></div></div>'
                f'<div class="rl-tl-body"><div class="rl-tl-title">{tool_display}</div><div class="rl-tl-detail">{detail_display}</div>'
                f'<div class="rl-tl-time">{time_label}{duration_label}</div></div></div>'
            )
            # st.expander/st.json can't live inside a CSS multi-column masonry
            # flow (a native Streamlit widget always renders as its own
            # sibling element, outside any surrounding markdown's HTML -- the
            # same reason the panel-wrapping bug existed elsewhere in this
            # app). A plain <details> disclosure with escaped, preformatted
            # JSON gets the same "click to inspect" behavior without leaving
            # the masonry grid.
            step_json = json.dumps({"arguments": step["arguments"], "result": step["result"]}, indent=2, default=str)
            raw_trace_steps_html.append(
                f'<div class="rl-details-step"><div class="rl-details-step-title">Step {i}: '
                f'<code>{html.escape(tool)}</code></div><pre>{html.escape(step_json)}</pre></div>'
            )

        why_this_score_card = ""
        if result["explanation"] or result["top_factors"]:
            why_this_score_card = f"""
            <div class="rl-panel">
                <div class="rl-panel-label">Why this score?</div>
                {shap_bars_html(result["top_factors"])}
                <p style="margin-top:10px; color:var(--rl-text-dim); font-size:0.85rem;">{result['explanation']}</p>
            </div>
            """

        # One continuous HTML fragment, not several separate st.markdown calls --
        # CSS multi-column layout needs every card as a sibling inside the SAME
        # container to balance their heights across columns; splitting it across
        # calls would reintroduce the "div never actually wraps its contents"
        # bug fixed elsewhere in this app.
        html_block(
            f"""
            <div class="rl-masonry">
                <div class="rl-panel">
                    <div class="rl-panel-label">Transaction</div>
                    {order_panel_html(result)}
                </div>
                <div class="rl-panel">
                    <div class="rl-panel-label">Merchant</div>
                    <div class="rl-kv-grid">
                        <div><div class="rl-kv-label">Merchant ID</div><div class="rl-kv-value">{html.escape(str(result.get('merchant_id'))) if result.get('merchant_id') else '—'}</div></div>
                    </div>
                </div>
                <div class="rl-panel">
                    <div class="rl-panel-label">Agent investigation</div>
                    <div class="rl-timeline">{''.join(steps_html)}</div>
                </div>
                <div class="rl-panel">
                    {agent_recommendation_card_html(result["agent_proposal"])}
                </div>
                <div class="rl-panel">
                    {gate_decision_card_html(result["gated_decision"], result["gated_reason"])}
                </div>
                <div class="rl-panel">
                    <div class="rl-panel-label">Final decision</div>
                    {decision_badge_html(result["gated_decision"])}
                    {authority_strip_html(result["gated_reason"])}
                    <p style="margin-top:12px; font-size:0.8rem; color:var(--rl-text-dim);">&#10003; Verified by deterministic gate &nbsp;&#183;&nbsp; &#10003; Audit event committed</p>
                </div>
                {why_this_score_card}
                <details class="rl-details">
                    <summary>Raw trace (full tool arguments and results)</summary>
                    {''.join(raw_trace_steps_html)}
                </details>
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
            st.altair_chart(roc_chart(chart_data["roc_curve"]), width="stretch")
    with col_cm:
        with st.container(border=True, key="panel_confusion_matrix"):
            html_block(f'<div class="rl-panel-label">Confusion matrix &middot; XGBoost (threshold {xgb["threshold"]:.2f})</div>')
            st.altair_chart(confusion_matrix_chart(chart_data["confusion_matrix"]), width="stretch")

    with st.container(border=True, key="panel_shap_global"):
        html_block('<div class="rl-panel-label">Global feature importance (SHAP, test set)</div>')
        st.altair_chart(shap_global_chart(chart_data["shap_global_importance"]), width="stretch")

    render_threshold_explorer(xgb["threshold"])
    render_retrain_from_feedback_section()


def _test_set_predictions():
    """Fresh predict_proba on the held-out test split for whatever model is
    currently loaded. Deliberately NOT cached -- if a candidate was just
    promoted (see render_retrain_from_feedback_section), get_model_and_explainer
    is cleared and `model` is reloaded from disk on the rerun, so this should
    always reflect what's actually deployed right now, not a stale snapshot.

    Prefers TEST_SNAPSHOT_PATH (the exact raw rows promote_candidate last
    evaluated the deployed model on) over re-deriving a split from
    data/raw/merchant_snapshots.csv via load_and_split(). Those two stop
    being the same test set the moment any feedback-retrained model gets
    promoted: combining in human-override rows and re-sorting by date shifts
    where the train/val/test boundaries fall, so load_and_split() alone
    would silently score the deployed model against a different set of rows
    (and a different row count) than the one its own metrics.json/
    chart_data.json were computed from -- two panels on this same page
    disagreeing about the same model at the same threshold. Falling back to
    load_and_split() when no snapshot file exists yet is still exactly
    correct for a model that's never been retrained with feedback, since
    that's precisely the split model/train.py used to produce it."""
    if os.path.exists(TEST_SNAPSHOT_PATH):
        test_df = pd.read_csv(TEST_SNAPSHOT_PATH)
        X_test = transform_features(test_df)
        y_test = test_df["is_risky"].values
    else:
        splits = load_and_split()
        X_test, y_test = splits["test"]
    probs = model.predict_proba(X_test)[:, 1]
    return y_test, probs


def render_threshold_explorer(default_threshold: float):
    """
    A what-if simulator, not a live control: moving this slider never touches
    the actual gating rules (gating/decision_engine.py) that decide real
    cases -- those stay fixed, versioned, and reviewed separately. What this
    shows is the real tradeoff a threshold choice makes on the held-out test
    set: raise it and you catch less real risk but bother fewer legitimate
    merchants; lower it and the reverse. Useful for judges/reviewers to see
    that ESCALATE_THRESHOLD/FLAG_THRESHOLD in config were not just guessed.
    """
    st.divider()
    st.markdown("#### Threshold explorer")
    st.caption(
        "A what-if simulator on the held-out test set -- moving this slider does not change the live "
        "gating rules. It shows the real precision/recall tradeoff behind a threshold choice: catch "
        "more real risk and you will also flag more legitimate merchants by mistake."
    )

    y_test, probs = _test_set_predictions()

    threshold = st.slider(
        "Decision threshold (a risk score at or above this counts as \"risky\")",
        min_value=0.0, max_value=1.0, value=float(default_threshold), step=0.01,
    )

    y_pred = (probs >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    kcols = st.columns(4)
    with kcols[0]:
        html_block(kpi_html("Precision", f"{precision:.2f}", "of flagged merchants are truly risky"))
    with kcols[1]:
        html_block(kpi_html("Recall", f"{recall:.2f}", "of truly risky merchants get caught"))
    with kcols[2]:
        html_block(kpi_html("F1", f"{f1:.2f}"))
    with kcols[3]:
        html_block(kpi_html("Flagged", tp + fp, f"of {len(y_test)} test merchants"))

    st.markdown(
        f"At this threshold: **{tp}** truly risky merchants are correctly caught, **{fn}** risky "
        f"merchant(s) slip through undetected (false negatives), and **{fp}** legitimate merchant(s) "
        f"are wrongly flagged for review (false positives) -- out of {len(y_test)} merchants in the "
        "held-out test set."
    )

    cm_data = {"labels": ["Not risky", "Risky"], "matrix": [[tn, fp], [fn, tp]]}
    with st.container(border=True, key="panel_threshold_cm"):
        html_block(f'<div class="rl-panel-label">Confusion matrix at threshold {threshold:.2f}</div>')
        st.altair_chart(confusion_matrix_chart(cm_data), width="stretch")


def render_retrain_from_feedback_section():
    """
    Closes the loop the human-override feature opens: a reviewer's
    corrections (Investigations page) are labeled data sitting in the audit
    log, and this is where that data actually gets used. Deliberately a
    two-step, human-gated flow -- "train a candidate and show its impact"
    is always safe to do on demand, but "replace the live model" only ever
    happens if a person looks at the before/after numbers here and decides
    to promote it. Same propose-then-a-human-decides pattern as everywhere
    else in RiskLens.
    """
    st.divider()
    st.markdown("#### Retrain from feedback")

    overrides = get_all_overrides(conn)
    if not overrides:
        html_block(
            empty_state_html(
                "No feedback to train on yet",
                "Once a reviewer overrides a decision on the Investigations page, it shows up here as "
                "training data. Retraining is always optional and never happens automatically -- this "
                "section only runs when you click the button below.",
            )
        )
        return

    st.caption(
        f"{len(overrides)} human override(s) recorded. Training a candidate model never changes what's "
        "live -- it only happens if you review the numbers below and choose to promote it."
    )

    if st.button("Train candidate model with feedback", key="train_candidate_btn"):
        with st.spinner("Retraining a candidate model on the original data plus your feedback..."):
            try:
                st.session_state["_retrain_result"] = train_candidate_with_feedback(conn)
            except Exception as exc:  # noqa: BLE001 -- surface any failure, don't crash the page
                st.session_state["_retrain_result"] = None
                st.error(f"Retraining failed: {exc}")

    result = st.session_state.get("_retrain_result")
    if not result:
        return

    skipped = result["total_overrides"] - result["feedback_rows_used"]
    st.caption(
        f"Trained on {result['combined_rows']} rows: the original training set plus "
        f"{result['feedback_rows_used']} usable override(s)"
        + (f" ({skipped} skipped -- incomplete feature data, or superseded by a later override on the same case)." if skipped else ".")
    )
    html_block(
        f"""
        <div class="rl-panel">
            <div class="rl-panel-label">Candidate (+ feedback) vs. currently deployed</div>
            {model_comparison_table_html(result["candidate_metrics"], result["current_metrics"], "Candidate (+feedback)", "Current (deployed)")}
        </div>
        """
    )

    candidate_better = result["candidate_metrics"]["f1"] >= result["current_metrics"]["f1"]
    if not candidate_better:
        st.caption("The candidate doesn't beat the deployed model's F1 on this test split -- promoting it anyway is your call, not a recommendation.")

    if st.button("Promote candidate to production", key="promote_candidate_btn", type="primary"):
        promote_candidate(
            result["candidate_model"],
            result["candidate_threshold"],
            result["candidate_metrics"],
            result["candidate_artifacts"],
            candidate_test_df=result["candidate_test_df"],
        )
        get_model_and_explainer.clear()  # so the app picks up the new model immediately, not just after a restart
        st.session_state["_retrain_result"] = None
        st.success("Candidate promoted -- RiskLens is now scoring with the retrained model.")
        st.rerun()


# =============================================================================
# PAGE: Audit Trail
# =============================================================================
def render_monitoring_section(events: list):
    """
    Portfolio-level health, not a single case's story -- this is deliberately
    different from the Overview page, which shows a snapshot of what's
    happening right now. This asks "is the system behaving well over time?"
    via two numbers that matter for an accountable AI system: how often a
    human ends up correcting the gate (a rising override rate is an early
    warning worth investigating before it becomes a support complaint), and
    how often the agent's own recommendation actually agrees with what the
    deterministic gate decides (an agent that never disagrees with the gate
    isn't adding independent judgment; one that never agrees suggests its
    reasoning has drifted from the rules actually in force).
    """
    # Overrides are fetched with their own limit (1000) that doesn't match
    # `events`'s own cap (page_audit_trail passes get_all_events(limit=500))
    # -- so once the audit log holds more than 500 events, an override
    # recorded on an older event outside that 500-row window would still
    # count toward this rate's numerator without that event being part of
    # the denominator at all. Restricting to overrides whose event_id is
    # actually IN `events` keeps numerator and denominator counting the
    # same population, so this can never exceed 100% or attribute an
    # override to a case this page isn't even showing.
    event_ids_in_view = {e["event_id"] for e in events}
    overrides = get_all_overrides(conn, limit=1000)
    overridden_ids = {o["event_id"] for o in overrides if o["event_id"] in event_ids_in_view}
    override_rate = len(overridden_ids) / len(events) if events else 0.0

    agent_events = [e for e in events if (e.get("source") or "rule_pipeline") == "agent_pipeline"]
    agree_count = 0
    for e in agent_events:
        proposal = e.get("agent_proposal")
        if isinstance(proposal, str):
            try:
                proposal = json.loads(proposal)
            except (TypeError, json.JSONDecodeError):
                proposal = None
        if proposal and proposal.get("recommended_decision") == e.get("decision"):
            agree_count += 1
    agreement_rate = (agree_count / len(agent_events)) if agent_events else None

    scored = [e["risk_score"] for e in events if e.get("risk_score") is not None]
    avg_score = (sum(scored) / len(scored)) if scored else None

    st.markdown("#### System monitoring")
    st.caption("How the system is behaving over time -- not just what happened in any single case.")

    kcols = st.columns(4)
    with kcols[0]:
        html_block(kpi_html("Total decisions", len(events)))
    with kcols[1]:
        html_block(
            kpi_html(
                "Human override rate", f"{override_rate:.1%}",
                f"{len(overridden_ids)} of {len(events)} decisions later corrected",
                accent=override_rate > 0.15,
            )
        )
    with kcols[2]:
        html_block(
            kpi_html(
                "Agent-gate agreement",
                f"{agreement_rate:.1%}" if agreement_rate is not None else "—",
                f"{agree_count} of {len(agent_events)} agent runs" if agent_events else "No agent runs yet",
            )
        )
    with kcols[3]:
        html_block(kpi_html("Average risk score", f"{avg_score:.2f}" if avg_score is not None else "—"))

    dated = [e for e in events if e.get("timestamp_utc") and e.get("decision")]
    if len(dated) >= 3:
        vol_df = pd.DataFrame(dated)[["timestamp_utc", "decision"]]
        vol_df["timestamp_utc"] = pd.to_datetime(vol_df["timestamp_utc"], utc=True, errors="coerce")
        with st.container(border=True, key="panel_decision_volume"):
            html_block('<div class="rl-panel-label">Decision volume over time</div>')
            st.altair_chart(decision_volume_chart(vol_df), width="stretch")
    else:
        st.caption(f"Not enough events yet to plot volume over time (need at least 3, have {len(dated)}).")

    st.divider()


def page_audit_trail():
    render_top_bar("Audit trail", "Every AI action, gate decision, and final outcome is traceable.", RAZORPAY_CONFIGURED)

    events = get_all_events(conn, limit=500)
    if not events:
        html_block(empty_state_html("No audit events recorded.", "Run an investigation in Investigations or Live Agent to populate the audit trail."))
        return

    render_monitoring_section(events)

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
        # regex=False: this is a plain-text search box (the placeholder and
        # sibling search on the Investigations page both imply literal
        # substring matching), and str.contains defaults to regex=True --
        # an unbalanced parenthesis or other regex metacharacter in a
        # merchant ID a reviewer types (e.g. "test(store)") would otherwise
        # raise and crash the whole page instead of just finding no match.
        filtered = filtered[filtered["merchant_id"].astype(str).str.contains(merchant_search, case=False, na=False, regex=False)]

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

    st.divider()
    st.markdown("#### Human feedback for retraining")
    overrides = get_all_overrides(conn, limit=1000)
    if not overrides:
        html_block(
            empty_state_html(
                "No reviewer overrides yet",
                "When a reviewer corrects a decision (see &ldquo;Override this decision&rdquo; on any case in "
                "Investigations), it's logged here as labeled feedback &mdash; exactly the kind of signal a future "
                "model retrain would use to learn from real corrections instead of just the original training data.",
            )
        )
    else:
        overrides_df = pd.DataFrame(overrides)[
            ["timestamp_utc", "event_id", "original_decision", "overridden_decision", "reason", "reviewer"]
        ]
        st.caption(
            f"{len(overrides)} reviewer correction(s) recorded so far -- this table is the training signal "
            "a feedback-driven retrain would use."
        )
        st.dataframe(overrides_df, hide_index=True, width="stretch")
        st.download_button(
            "Download feedback as CSV",
            data=overrides_df.to_csv(index=False),
            file_name="risklens_human_feedback.csv",
            mime="text/csv",
        )


# =============================================================================
# Sidebar + routing
# =============================================================================
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "risklens_logo.svg")
if os.path.exists(LOGO_PATH):
    st.logo(LOGO_PATH, size="large")

pages = [
    st.Page(page_overview, title="Overview", icon=":material/dashboard:", default=True),
    st.Page(page_investigations, title="Investigations", icon=":material/search:"),
    st.Page(page_batch_scoring, title="Batch Scoring", icon=":material/upload_file:"),
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
