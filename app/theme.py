"""
Visual system for the RiskLens console -- light, premium-fintech palette.

Kept separate from dashboard.py on purpose: this file is pure presentation
(CSS + small HTML/Altair-rendering helpers), so it can be reworked without
touching any of the scoring/agent/gating logic it sits on top of.

Every color, threshold, and label here is either a fixed system constant
(gate thresholds, gate version) or is populated at render time from a real
computed result -- nothing in this module invents data for visual effect.
"""

import html
import math

import altair as alt
import pandas as pd
import streamlit as st

from gating.decision_engine import (
    DECISION_CLEAR,
    DECISION_ESCALATE,
    DECISION_FLAG,
    DECISION_MANUAL_REVIEW,
    ESCALATE_THRESHOLD,
    FLAG_THRESHOLD,
    GATE_VERSION,
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG = "#F7F8FA"
SURFACE = "#FFFFFF"
SURFACE_2 = "#F3F4F6"
BORDER = "#E6E8EB"
TEXT = "#17191C"
TEXT_DIM = "#66707A"
TEXT_MUTED = "#98A0A8"
ACCENT = "#E53935"

GREEN = "#1E9E5A"
AMBER = "#D97C0A"
ORANGE = "#E0631C"
RED = "#D93025"
BLUE = "#2E6FCC"

DECISION_STYLE = {
    DECISION_CLEAR: {"color": GREEN, "bg": "rgba(30,158,90,0.10)", "label": "Clear"},
    DECISION_ESCALATE: {"color": AMBER, "bg": "rgba(217,124,10,0.10)", "label": "Escalate"},
    DECISION_FLAG: {"color": ORANGE, "bg": "rgba(224,99,28,0.10)", "label": "Compliance review"},
    DECISION_MANUAL_REVIEW: {"color": RED, "bg": "rgba(217,48,37,0.10)", "label": "Manual review"},
}

STATUS_ONLINE = {"color": GREEN, "label": "Online"}
STATUS_CONNECTED = {"color": GREEN, "label": "Connected"}
STATUS_ACTIVE = {"color": GREEN, "label": "Active"}
STATUS_OFFLINE = {"color": TEXT_MUTED, "label": "Not configured"}


def html_block(html: str):
    """Render a raw HTML string via st.markdown, safely.

    Streamlit's markdown parser follows normal Markdown rules: a line
    indented 4+ spaces is read as a fenced code block, which Python
    f-strings indented to match surrounding code trip constantly. The fix
    is to flatten the whole block to one line -- but joining with an empty
    string glues words together when prose wraps across source lines
    ("authority.Every" instead of "authority. Every"), so lines are joined
    with a single space instead: harmless between HTML tags, and it keeps
    prose readable. Every HTML block in this app goes through this helper,
    not st.markdown directly, so neither bug can regress.
    """
    flattened = " ".join(line.strip() for line in html.strip().splitlines())
    st.markdown(flattened, unsafe_allow_html=True)


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {{
    --rl-bg: {BG}; --rl-surface: {SURFACE}; --rl-surface-2: {SURFACE_2}; --rl-border: {BORDER};
    --rl-text: {TEXT}; --rl-text-dim: {TEXT_DIM}; --rl-text-muted: {TEXT_MUTED}; --rl-accent: {ACCENT};
}}

html {{
    color-scheme: light only;
}}
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: var(--rl-bg) !important;
    font-family: 'Inter', -apple-system, sans-serif;
    color: var(--rl-text);
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stAppViewContainer"] > .main .block-container {{ padding-top: 1.2rem; max-width: 1320px; }}
h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; font-weight: 600 !important; color: var(--rl-text); }}
p, span, div, label {{ font-family: 'Inter', sans-serif; }}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {{
    background: var(--rl-surface) !important; border-right: 1px solid var(--rl-border);
    min-width: 252px !important; max-width: 260px !important;
}}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 0.6rem; }}
[data-testid="stSidebarNav"] {{ padding-top: 4px; }}
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNav"] a * {{
    color: var(--rl-text-dim) !important; opacity: 1 !important;
}}
[data-testid="stSidebarNav"] a {{
    border-radius: 8px !important; font-weight: 500 !important; font-size: 0.88rem !important;
    padding: 8px 12px !important; margin: 1px 10px !important; gap: 10px;
}}
[data-testid="stSidebarNav"] a:hover, [data-testid="stSidebarNav"] a:hover * {{
    background: var(--rl-surface-2) !important; color: var(--rl-text) !important;
}}
[data-testid="stSidebarNav"] a[aria-current="page"], [data-testid="stSidebarNav"] a[aria-current="page"] * {{
    background: rgba(229,57,53,0.08) !important; color: var(--rl-accent) !important; font-weight: 600 !important;
}}
[data-testid="stSidebarUserContent"] {{ padding-top: 0; }}

/* Sidebar brand block */
.rl-sidebar-brand {{ padding: 10px 18px 16px 18px; border-bottom: 1px solid var(--rl-border); margin-bottom: 6px; }}
.rl-sidebar-name {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.12rem; color: var(--rl-text); }}
.rl-sidebar-sub {{ font-size: 0.70rem; color: var(--rl-text-muted); letter-spacing: 0.04em; margin-top: 1px; }}

/* Sidebar status footer */
.rl-sidebar-status {{ padding: 12px 18px 4px 18px; border-top: 1px solid var(--rl-border); margin-top: 10px; }}
.rl-sidebar-status-title {{ font-size: 0.66rem; color: var(--rl-text-muted); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; margin-bottom: 8px; }}
.rl-sb-row {{ display: flex; align-items: center; gap: 8px; font-size: 0.78rem; color: var(--rl-text-dim); padding: 3px 0; }}
.rl-sb-dot {{ width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }}

/* ---- Top bar (per-page header) ---- */
.rl-topbar {{ display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 4px; flex-wrap: wrap; gap: 12px; }}
.rl-topbar h1 {{ font-size: 1.7rem; margin: 0; }}
.rl-topbar-sub {{ color: var(--rl-text-dim); font-size: 0.92rem; margin-top: 6px; max-width: 62ch; }}
.rl-env-pill {{
    display: flex; flex-direction: column; align-items: flex-end; gap: 2px; font-size: 0.72rem;
    color: var(--rl-text-muted); letter-spacing: 0.04em; white-space: nowrap; padding-top: 2px;
}}
.rl-env-pill b {{ color: var(--rl-text); font-family: 'Space Grotesk', sans-serif; letter-spacing: 0.06em; font-size: 0.74rem; }}
.rl-env-live {{ display: flex; align-items: center; gap: 6px; font-weight: 600; }}
.rl-env-dot {{ width: 6px; height: 6px; border-radius: 50%; }}

/* ---- Panels ---- */
.rl-panel {{ background: var(--rl-surface); border: 1px solid var(--rl-border); border-radius: 10px; padding: 20px 22px; margin: 12px 0; }}
.rl-panel-label {{ margin: 0 0 14px 0; color: var(--rl-text-muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }}

/* Panels that must contain a live Streamlit widget (a chart, dataframe) use a
   real st.container(border=True, key="panel_...") instead of a raw HTML div,
   because a native widget renders as a sibling element and a hand-written
   <div> opened in one st.markdown call can never actually wrap it -- the
   browser closes the unclosed tag at that markdown call's own boundary, so
   the "card" background/border only ever wrapped the label, not the widget
   below it. This selector re-skins Streamlit's own bordered container to
   look exactly like .rl-panel instead. */
div.stVerticalBlock[class*="st-key-panel_"] {{
    background: var(--rl-surface) !important;
    border: 1px solid var(--rl-border) !important;
    border-radius: 10px !important;
    padding: 20px 22px !important;
    margin: 12px 0 !important;
    gap: 0.5rem !important;
}}
.rl-panel-title {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.02rem; color: var(--rl-text); margin-bottom: 2px; }}

/* ---- Live Agent verdict strip: the outcome, shown immediately, full-width,
   instead of buried at the bottom of the tallest of several side-by-side
   columns. ---- */
.rl-verdict-strip {{
    display: flex; flex-wrap: wrap; align-items: center; gap: 28px;
    background: var(--rl-surface); border: 1px solid var(--rl-border); border-left: 5px solid var(--rl-border);
    border-radius: 10px; padding: 18px 24px; margin: 16px 0 4px 0;
}}
.rl-verdict-case-label, .rl-verdict-score-value + .rl-verdict-score-tag, .rl-verdict-decision-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--rl-text-muted); font-weight: 700; }}
.rl-verdict-case-label {{ margin-bottom: 3px; }}
.rl-verdict-case-id {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.05rem; color: var(--rl-text); }}
.rl-verdict-score {{ display: flex; align-items: baseline; gap: 8px; }}
.rl-verdict-score-value {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.9rem; color: var(--rl-text); }}
.rl-verdict-score-tag {{ font-size: 0.72rem; font-weight: 700; padding: 3px 9px; border-radius: 999px; }}
.rl-verdict-decision-label {{ margin-bottom: 3px; }}
.rl-verdict-decision-value {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.3rem; }}
.rl-verdict-note {{ margin-left: auto; text-align: right; font-size: 0.85rem; font-weight: 600; }}
.rl-verdict-authority {{ margin-top: 3px; font-size: 0.72rem; font-weight: 500; color: var(--rl-text-muted); }}
@media (max-width: 900px) {{ .rl-verdict-note {{ margin-left: 0; text-align: left; }} }}

/* ---- Balanced masonry grid for the supporting detail cards below the
   verdict strip. A plain flex/grid row of unequal-height columns leaves the
   tallest column cascading down alone with empty space beside it (the exact
   bug this replaced); CSS multi-column layout instead measures total content
   height up front and distributes it evenly across columns, so no single
   column ever visibly straggles below the others. break-inside: avoid keeps
   each card intact rather than splitting one across two columns. ---- */
.rl-masonry {{ column-count: 3; column-gap: 20px; margin-top: 18px; }}
.rl-masonry .rl-panel {{ break-inside: avoid; -webkit-column-break-inside: avoid; width: 100%; display: inline-block; }}
@media (max-width: 1100px) {{ .rl-masonry {{ column-count: 2; }} }}
@media (max-width: 700px) {{ .rl-masonry {{ column-count: 1; }} }}

/* Native <details> used for the raw trace disclosure inside the masonry grid
   -- a real Streamlit widget (st.expander) can't live inside a CSS
   multi-column flow, so this is styled to match .rl-panel exactly. */
.rl-details {{ background: var(--rl-surface); border: 1px solid var(--rl-border); border-radius: 10px; padding: 16px 20px; margin: 12px 0; break-inside: avoid; }}
.rl-details summary {{ cursor: pointer; color: var(--rl-text-muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; list-style: none; }}
.rl-details summary::-webkit-details-marker {{ display: none; }}
.rl-details summary::before {{ content: "▸ "; }}
.rl-details[open] summary::before {{ content: "▾ "; }}
.rl-details-step {{ margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--rl-border); }}
.rl-details-step:first-of-type {{ border-top: none; }}
.rl-details-step-title {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.85rem; color: var(--rl-text); margin-bottom: 6px; }}
.rl-details pre {{ background: var(--rl-surface-2); border-radius: 6px; padding: 10px 12px; font-size: 0.75rem; color: var(--rl-text-dim); overflow-x: auto; white-space: pre-wrap; word-break: break-word; margin: 0; }}

/* ---- KPI tiles ---- */
.rl-kpi-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 16px 0; }}
.rl-kpi {{ flex: 1; min-width: 168px; background: var(--rl-surface); border: 1px solid var(--rl-border); border-radius: 10px; padding: 16px 18px; }}
.rl-kpi-label {{ font-size: 0.72rem; color: var(--rl-text-muted); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}
.rl-kpi-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.85rem; font-weight: 700; color: var(--rl-text); margin-top: 6px; font-variant-numeric: tabular-nums; }}
.rl-kpi-context {{ font-size: 0.76rem; color: var(--rl-text-muted); margin-top: 3px; }}
.rl-kpi-accent {{ border-left: 3px solid var(--rl-accent); }}

/* ---- System status ---- */
.rl-status-list {{ display: flex; flex-direction: column; gap: 10px; }}
.rl-status-row {{ display: flex; align-items: center; gap: 9px; font-size: 0.87rem; }}
.rl-status-dot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }}
.rl-status-value {{ margin-left: auto; font-size: 0.76rem; font-weight: 600; }}

/* ---- Decision badge ---- */
.rl-badge {{ display: inline-flex; align-items: center; gap: 6px; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 0.86rem; padding: 6px 13px; border-radius: 7px; }}

/* ---- Workflow tracker ---- */
.rl-workflow {{ display: flex; align-items: center; gap: 0; margin: 4px 0 16px 0; flex-wrap: wrap; }}
.rl-workflow-step {{ font-size: 0.68rem; font-weight: 600; letter-spacing: 0.04em; padding: 5px 11px; border-radius: 6px; border: 1px solid var(--rl-border); color: var(--rl-text-muted); background: var(--rl-surface-2); }}
.rl-workflow-step.done {{ color: var(--rl-accent); border-color: var(--rl-accent); background: rgba(229,57,53,0.06); }}
.rl-workflow-arrow {{ color: var(--rl-text-muted); padding: 0 6px; font-size: 0.75rem; }}

/* ---- Case / score ---- */
.rl-case-id {{ font-family: 'Space Grotesk', sans-serif; font-size: 0.8rem; color: var(--rl-text-muted); }}
.rl-score-row {{ display: flex; align-items: baseline; gap: 12px; margin: 6px 0 2px 0; }}
.rl-score-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; font-weight: 700; color: var(--rl-text); font-variant-numeric: tabular-nums; }}
.rl-score-tag {{ font-size: 0.74rem; font-weight: 700; padding: 3px 9px; border-radius: 5px; }}

/* ---- Risk scale ---- */
.rl-scale-wrap {{ margin: 14px 0 6px 0; }}
.rl-scale-track {{ position: relative; height: 6px; border-radius: 3px; width: 100%;
    background: linear-gradient(to right, {GREEN} 0%, {GREEN} {ESCALATE_THRESHOLD * 100:.0f}%, {AMBER} {ESCALATE_THRESHOLD * 100:.0f}%, {AMBER} {FLAG_THRESHOLD * 100:.0f}%, {ORANGE} {FLAG_THRESHOLD * 100:.0f}%, {ORANGE} 100%); }}
.rl-scale-marker {{ position: absolute; top: -6px; width: 2px; height: 18px; background: var(--rl-text); }}
.rl-scale-marker::after {{ content: attr(data-score); position: absolute; top: -22px; left: 50%; transform: translateX(-50%);
    font-family: 'Space Grotesk', sans-serif; font-size: 0.7rem; font-weight: 700; color: var(--rl-text); white-space: nowrap; }}
.rl-scale-labels {{ display: flex; justify-content: space-between; margin-top: 6px; font-size: 0.65rem; color: var(--rl-text-muted); }}
.rl-scale-note {{ margin-top: 4px; font-size: 0.65rem; color: var(--rl-text-muted); line-height: 1.4; }}

/* ---- Authority strip ---- */
.rl-authority {{ display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;
    margin-top: 15px; padding-top: 13px; border-top: 1px solid var(--rl-border); font-size: 0.74rem; color: var(--rl-text-muted); }}
.rl-authority b {{ color: var(--rl-text); font-weight: 700; }}

/* ---- Human override ---- */
.rl-override-banner {{ margin-top: 14px; padding: 12px 16px; border-radius: 8px; border-left: 4px solid {BLUE};
    background: rgba(46,111,204,0.07); font-size: 0.82rem; color: var(--rl-text); line-height: 1.5; }}
.rl-override-banner b {{ color: {BLUE}; }}
.rl-override-meta {{ margin-top: 5px; font-size: 0.72rem; color: var(--rl-text-muted); }}

/* ---- KV grid ---- */
.rl-kv-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 13px; }}
.rl-kv-label {{ font-size: 0.66rem; color: var(--rl-text-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }}
.rl-kv-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem; color: var(--rl-text); margin-top: 2px; font-variant-numeric: tabular-nums; }}

/* ---- SHAP bars ---- */
.rl-shap-row {{ margin-bottom: 11px; }}
.rl-shap-label {{ font-size: 0.74rem; color: var(--rl-text-dim); font-weight: 600; margin-bottom: 4px; }}
.rl-shap-bar-track {{ display: flex; align-items: center; height: 16px; }}
.rl-shap-bar {{ height: 7px; border-radius: 3px; }}
.rl-shap-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 0.75rem; font-weight: 700; margin-left: 8px; }}

/* ---- Timeline (Live Agent center panel) ---- */
.rl-timeline {{ display: flex; flex-direction: column; }}
.rl-tl-step {{ display: flex; gap: 12px; padding: 0 0 16px 0; position: relative; }}
.rl-tl-rail {{ display: flex; flex-direction: column; align-items: center; width: 14px; flex-shrink: 0; }}
.rl-tl-dot {{ width: 9px; height: 9px; border-radius: 50%; margin-top: 4px; flex-shrink: 0; }}
.rl-tl-line {{ flex: 1; width: 1px; background: var(--rl-border); margin-top: 2px; }}
.rl-tl-body {{ flex: 1; padding-bottom: 2px; }}
.rl-tl-title {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.85rem; color: var(--rl-text); }}
.rl-tl-detail {{ font-size: 0.78rem; color: var(--rl-text-dim); margin-top: 2px; }}
.rl-tl-time {{ font-size: 0.68rem; color: var(--rl-text-muted); margin-top: 2px; font-variant-numeric: tabular-nums; }}

/* ---- Compare / decision control ---- */
.rl-compare-col {{ background: var(--rl-surface-2); border: 1px solid var(--rl-border); border-radius: 9px; padding: 14px 16px; margin-bottom: 10px; }}
.rl-compare-title {{ font-size: 0.66rem; color: var(--rl-text-muted); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700; margin-bottom: 7px; }}
.rl-compare-decision {{ font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 700; }}
.rl-compare-sub {{ font-size: 0.78rem; color: var(--rl-text-dim); margin-top: 5px; }}
.rl-verdict-banner {{ margin-top: 10px; padding: 10px 14px; border-radius: 8px; font-size: 0.8rem; font-weight: 600; }}

/* ---- Empty / error state ---- */
.rl-state-card {{ background: var(--rl-surface); border: 1px dashed var(--rl-border); border-radius: 10px; padding: 26px 24px; margin: 12px 0; text-align: left; }}
.rl-state-title {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 0.95rem; color: var(--rl-text); margin-bottom: 6px; }}
.rl-state-body {{ font-size: 0.84rem; color: var(--rl-text-dim); line-height: 1.55; }}
.rl-state-body code {{ background: var(--rl-surface-2); border: 1px solid var(--rl-border); border-radius: 4px; padding: 1px 6px; }}

/* ---- Buttons ---- */
.stButton > button {{
    background: var(--rl-accent) !important; color: white !important; border: none !important;
    border-radius: 7px !important; font-weight: 600 !important; font-size: 0.86rem !important;
    padding: 0.55rem 1.3rem !important; box-shadow: none !important;
}}
.stButton > button:hover {{ filter: brightness(1.08); }}
.stButton > button[kind="secondary"] {{ background: var(--rl-surface) !important; color: var(--rl-text) !important; border: 1px solid var(--rl-border) !important; }}

/* ---- Metrics / dataframe / expander ---- */
[data-testid="stMetricValue"] {{ font-family: 'Space Grotesk', sans-serif; color: var(--rl-text); font-size: 1.4rem; }}
[data-testid="stMetricLabel"] {{ color: var(--rl-text-muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--rl-border); border-radius: 8px; overflow: hidden; }}
[data-testid="stExpander"] {{ border: 1px solid var(--rl-border) !important; border-radius: 8px !important; background: var(--rl-surface); }}
input, textarea, select {{ font-family: 'Inter', sans-serif !important; }}
</style>
"""


def inject_theme():
    st.markdown(CSS, unsafe_allow_html=True)
    _register_altair_theme()


def _register_altair_theme():
    def rl_theme():
        return {
            "config": {
                "background": SURFACE,
                "font": "Inter",
                "axis": {
                    "labelFont": "Inter", "labelColor": TEXT_DIM, "labelFontSize": 10.5,
                    "titleFont": "Inter", "titleColor": TEXT_MUTED, "titleFontSize": 10.5, "titleFontWeight": 600,
                    "gridColor": BORDER, "gridOpacity": 0.7, "domainColor": BORDER, "tickColor": BORDER,
                },
                "legend": {"labelFont": "Inter", "labelColor": TEXT_DIM, "labelFontSize": 10.5, "titleColor": TEXT_MUTED, "titleFontSize": 10.5},
                "view": {"stroke": "transparent"},
            }
        }

    alt.themes.register("risklens", rl_theme)
    alt.themes.enable("risklens")


def render_sidebar_status(rows: list):
    """rows: list of (label, status_dict)."""
    items = "".join(
        f'<div class="rl-sb-row"><span class="rl-sb-dot" style="background:{s["color"]};"></span>{label}</div>'
        for label, s in rows
    )
    with st.sidebar:
        html_block(
            f"""
            <div class="rl-sidebar-status">
                <div class="rl-sidebar-status-title">System</div>
                {items}
            </div>
            """
        )


def render_top_bar(title: str, subtitle: str, razorpay_connected: bool):
    dot_color = GREEN if razorpay_connected else TEXT_MUTED
    conn_label = "Connected" if razorpay_connected else "Not configured"
    html_block(
        f"""
        <div class="rl-topbar">
            <div>
                <h1>{title}</h1>
                <div class="rl-topbar-sub">{subtitle}</div>
            </div>
            <div class="rl-env-pill">
                <b>RAZORPAY &middot; TEST MODE</b>
                <span class="rl-env-live"><span class="rl-env-dot" style="background:{dot_color};"></span>{conn_label}</span>
                <span>Built for the Razorpay AI Risk Manager Challenge</span>
            </div>
        </div>
        """
    )


def status_row_html(rows: list) -> str:
    items = "".join(
        f'<div class="rl-status-row"><span class="rl-status-dot" style="background:{s["color"]};"></span>'
        f'<span>{label}</span><span class="rl-status-value" style="color:{s["color"]};">{s["label"]}</span></div>'
        for label, s in rows
    )
    return f'<div class="rl-status-list">{items}</div>'


def kpi_html(label: str, value, context: str = "", accent: bool = False) -> str:
    cls = "rl-kpi rl-kpi-accent" if accent else "rl-kpi"
    ctx = f'<div class="rl-kpi-context">{context}</div>' if context else ""
    return f'<div class="{cls}"><div class="rl-kpi-label">{label}</div><div class="rl-kpi-value">{value}</div>{ctx}</div>'


def decision_badge_html(decision: str) -> str:
    style = DECISION_STYLE.get(decision, {"color": TEXT_MUTED, "bg": "rgba(152,160,168,0.10)", "label": decision})
    return f'<span class="rl-badge" style="background:{style["bg"]}; color:{style["color"]};">{style["label"]}</span>'


def risk_label_for_score(score):
    # gating/decision_engine.decide_from_score treats a NaN/inf score the
    # same as a missing one (routed to needs_manual_review with an honest
    # reason, not silently compared against a threshold) -- this display
    # function used to only guard against None, so every comparison below
    # being False for NaN meant a NaN score fell through to the final
    # `return "High risk"` branch. A case the gate correctly flagged as
    # "couldn't be scored" would then visually present to a reviewer as
    # confirmed maximum risk instead of unscored.
    if score is None or not math.isfinite(score):
        return "Unscored", TEXT_MUTED
    if score < ESCALATE_THRESHOLD:
        return "Low risk", GREEN
    if score <= FLAG_THRESHOLD:
        return "Elevated risk", AMBER
    return "High risk", ORANGE


def workflow_html(steps: list, completed: int) -> str:
    parts = []
    for i, step in enumerate(steps):
        cls = "rl-workflow-step done" if i < completed else "rl-workflow-step"
        parts.append(f'<span class="{cls}">{step}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="rl-workflow-arrow">&#8594;</span>')
    return f'<div class="rl-workflow">{"".join(parts)}</div>'


def case_id_from_event(event_id) -> str:
    if not event_id:
        return "RL-PENDING"
    return f"RL-{str(event_id).split('-')[0][:6].upper()}"


def risk_scale_html(score) -> str:
    # Same NaN/inf guard as risk_label_for_score: min(1.0, float('nan'))
    # returns 1.0 in Python, so without this check a NaN score used to draw
    # the marker at the far right of the scale (100%, "maximum risk") --
    # the visual opposite of the gate's own "couldn't be scored, needs
    # manual review" outcome for that same NaN score.
    if score is None or not math.isfinite(score):
        marker = ""
    else:
        pct = round(max(0.0, min(1.0, score)) * 100, 2)
        marker = f'<div class="rl-scale-marker" style="left:calc({pct}% - 1px);" data-score="{score:.2f}"></div>'
    return f"""
    <div class="rl-scale-wrap">
        <div class="rl-scale-track">{marker}</div>
        <div class="rl-scale-labels">
            <span>0.00 Clear</span><span>{ESCALATE_THRESHOLD:.2f} Escalate</span>
            <span>{FLAG_THRESHOLD:.2f} Flag for compliance review</span><span>1.00</span>
        </div>
        <div class="rl-scale-note">Needs manual review instead, regardless of score: input was missing/invalid, or the score was too close to {ESCALATE_THRESHOLD:.2f} to call automatically.</div>
    </div>
    """


def authority_strip_html(gate_reason: str) -> str:
    return f"""
    <div class="rl-authority">
        <span>{gate_reason}</span>
        <span>AI does not control this decision &nbsp;&#183;&nbsp; Final authority: <b>{GATE_VERSION}</b></span>
    </div>
    """


def override_banner_html(overrides: list) -> str:
    """
    Shown on a case that a human reviewer has since corrected. The gate's
    original decision is never erased or edited (see audit_log.OVERRIDES_SCHEMA)
    -- this banner sits alongside it, making clear a person, not the system,
    changed the effective outcome, and why. If a case has been overridden more
    than once, only the most recent correction is the one that currently applies.
    """
    if not overrides:
        return ""
    latest = overrides[0]  # get_overrides_for_event returns most-recent-first
    original_style = DECISION_STYLE.get(latest["original_decision"], {"label": latest["original_decision"]})
    new_style = DECISION_STYLE.get(latest["overridden_decision"], {"label": latest["overridden_decision"]})
    when = (latest.get("timestamp_utc") or "")[:19].replace("T", " ")
    # reviewer and reason are free text a human typed into the override form
    # (see app/dashboard.py's render_override_section) -- html.escape() here
    # for the same reason the raw trace JSON is escaped in the Live Agent
    # view: this whole app renders its HTML with unsafe_allow_html=True, so
    # unescaped user-typed text would be interpreted as live HTML/JS by
    # anyone who later opens this case, not just displayed as text.
    reviewer_label = f" by {html.escape(str(latest['reviewer']))}" if latest.get("reviewer") else ""
    history_note = (
        f'<div class="rl-override-meta">{len(overrides)} override(s) recorded for this case &mdash; showing the most recent.</div>'
        if len(overrides) > 1
        else ""
    )
    return f"""
    <div class="rl-override-banner">
        <b>&#9998; Reviewer override{reviewer_label}</b> &mdash; changed from
        <b>{original_style['label']}</b> to <b>{new_style['label']}</b>.
        <div class="rl-override-meta">&ldquo;{html.escape(str(latest['reason']))}&rdquo; &nbsp;&#183;&nbsp; {when} UTC</div>
        {history_note}
    </div>
    """


_FACTOR_LABELS = {
    "account_age_days": "Account age",
    "kyc_complete": "KYC status",
    "volume_change_pct": "Transaction volume",
    "chargeback_rate": "Chargeback rate",
    "refund_rate": "Refund rate",
    "avg_ticket_size": "Average ticket size",
}


def _factor_label(feature: str) -> str:
    if feature in _FACTOR_LABELS:
        return _FACTOR_LABELS[feature]
    if feature.startswith("category_"):
        return f"Business category ({feature.replace('category_', '')})"
    return feature


def shap_bars_html(top_factors: list) -> str:
    if not top_factors:
        return '<p style="color:var(--rl-text-muted); font-size:0.85rem;">No contributing factors available.</p>'
    max_abs = max(abs(f["shap_value"]) for f in top_factors) or 1.0
    rows = []
    for f in top_factors:
        val = f["shap_value"]
        width_pct = max(4, round(abs(val) / max_abs * 100))
        color = ORANGE if val > 0 else GREEN
        sign = "+" if val > 0 else ""
        rows.append(
            f'<div class="rl-shap-row"><div class="rl-shap-label">{_factor_label(f["feature"])}</div>'
            f'<div class="rl-shap-bar-track"><div class="rl-shap-bar" style="width:{width_pct}%; background:{color};"></div>'
            f'<span class="rl-shap-value" style="color:{color};">{sign}{val:.3f}</span></div></div>'
        )
    return "".join(rows)


def _agreement_banner_html(agent_proposal, gate_style, agree) -> str:
    if agree is True:
        return f'<div class="rl-verdict-banner" style="background:rgba(30,158,90,0.08); color:{GREEN};">&#10003; Agreement &mdash; the agent\'s recommendation matches the deterministic gate.</div>'
    if agree is False:
        return (
            f'<div class="rl-verdict-banner" style="background:rgba(217,48,37,0.08); color:{RED};">'
            f"&#9888; Gate override &mdash; the gate did not accept the agent's recommendation. "
            f"Final decision: <b>{gate_style['label']}</b>, authority: {GATE_VERSION}.</div>"
        )
    return f'<div class="rl-verdict-banner" style="background:rgba(152,160,168,0.08); color:{TEXT_MUTED};">No comparable agent proposal was produced &mdash; the gate decision stands alone.</div>'


def agent_recommendation_card_html(agent_proposal) -> str:
    """Just the AI-side half of the comparison, as its own standalone card --
    used where compare_panel_html's combined block would be too large a
    single unit for a balanced layout (e.g. a CSS masonry grid)."""
    agent_decision = (agent_proposal or {}).get("recommended_decision")
    # submit_decision's tool schema declares recommended_decision as a fixed
    # enum (agent/tools.py), but that enum is a hint to the model, not a
    # server-side guarantee every provider enforces -- so when agent_decision
    # isn't one of the 4 known decisions, this label falls back to whatever
    # string the LLM actually returned instead of always being a value this
    # app controls. Same reasoning as the escaping below: escape it too,
    # rather than assume tool-call output is as trustworthy as the gate's own
    # decision strings (which unlike this one really are always one of the
    # 4 fixed constants, since Python code produces them, not an LLM).
    fallback_label = html.escape(str(agent_decision)) if agent_decision else "No proposal"
    agent_style = DECISION_STYLE.get(agent_decision, {"color": TEXT_MUTED, "label": fallback_label})
    # The agent's reasoning is LLM-generated text, not a fixed template like
    # the gate's reason strings -- and the transaction it reasons about
    # includes the merchant_id a user typed in, unconstrained, on the Live
    # Agent page. Escaped for the same reason merchant_id itself is escaped
    # in dashboard.py: this whole app renders HTML with unsafe_allow_html=True,
    # so unescaped text here would be interpreted as live markup rather than
    # displayed as the words it actually is.
    reasoning = html.escape(str((agent_proposal or {}).get("reasoning", "No reasoning returned.")))
    return f"""
    <div class="rl-compare-col">
        <div class="rl-compare-title">AI agent recommendation</div>
        <div class="rl-compare-decision" style="color:{agent_style['color']};">{agent_style['label']}</div>
        <div class="rl-compare-sub">{reasoning}</div>
    </div>
    """


def gate_decision_card_html(gated_decision, gate_reason) -> str:
    """The gate-side half of the comparison, as its own standalone card -- see
    agent_recommendation_card_html."""
    gate_style = DECISION_STYLE.get(gated_decision, {"color": TEXT_MUTED, "label": gated_decision})
    return f"""
    <div class="rl-compare-col">
        <div class="rl-compare-title">Deterministic gate &middot; {GATE_VERSION}</div>
        <div class="rl-compare-decision" style="color:{gate_style['color']};">{gate_style['label']}</div>
        <div class="rl-compare-sub">{gate_reason}</div>
    </div>
    """


def compare_panel_html(agent_proposal, gated_decision, gate_reason, agree) -> str:
    gate_style = DECISION_STYLE.get(gated_decision, {"color": TEXT_MUTED, "label": gated_decision})
    banner = _agreement_banner_html(agent_proposal, gate_style, agree)
    return (
        agent_recommendation_card_html(agent_proposal)
        + gate_decision_card_html(gated_decision, gate_reason)
        + banner
    )


def verdict_banner_html(result: dict) -> str:
    """
    The Live Agent's outcome, shown once, full-width, immediately after the
    workflow tracker -- instead of only appearing at the bottom of whichever
    of three side-by-side columns happens to be tallest. Reuses the same
    agree/override language as compare_panel_html so the two never disagree
    with each other on the same page.
    """
    score = result.get("risk_score")
    risk_label, risk_color = risk_label_for_score(score)
    score_display = f"{score:.2f}" if score is not None else "--"

    gated_decision = result.get("gated_decision")
    gate_style = DECISION_STYLE.get(gated_decision, {"color": TEXT_MUTED, "label": gated_decision or "—"})
    agree = result.get("agent_and_gate_agree")
    agent_label = DECISION_STYLE.get(
        (result.get("agent_proposal") or {}).get("recommended_decision"), {}
    ).get("label", "no proposal")

    if agree is True:
        note = f'<span style="color:{GREEN};">&#10003; Agent and gate agree</span>'
    elif agree is False:
        note = f'<span style="color:{RED};">&#9888; Gate overrode the agent\'s &ldquo;{agent_label}&rdquo; recommendation</span>'
    else:
        note = f'<span style="color:{TEXT_MUTED};">No comparable agent proposal was produced</span>'

    case_id = case_id_from_event(result.get("event_id")) if result.get("event_id") else "—"

    return f"""
    <div class="rl-verdict-strip" style="border-left-color:{gate_style['color']};">
        <div>
            <div class="rl-verdict-case-label">Case</div>
            <div class="rl-verdict-case-id">{case_id}</div>
        </div>
        <div class="rl-verdict-score">
            <div class="rl-verdict-score-value">{score_display}</div>
            <div class="rl-verdict-score-tag" style="color:{risk_color}; background:{risk_color}1A;">{risk_label}</div>
        </div>
        <div>
            <div class="rl-verdict-decision-label">Final decision</div>
            <div class="rl-verdict-decision-value" style="color:{gate_style['color']};">{gate_style['label']}</div>
        </div>
        <div class="rl-verdict-note">
            {note}
            <div class="rl-verdict-authority">Final authority: {GATE_VERSION}</div>
        </div>
    </div>
    """


def empty_state_html(title: str, body: str) -> str:
    return f'<div class="rl-state-card"><div class="rl-state-title">{title}</div><div class="rl-state-body">{body}</div></div>'


def order_panel_html(result: dict) -> str:
    from datetime import datetime, timezone

    def cell(label, value):
        display = value if value not in (None, "") else "—"
        return f'<div><div class="rl-kv-label">{label}</div><div class="rl-kv-value">{display}</div></div>'

    amount = result.get("razorpay_order_amount")
    created_epoch = result.get("razorpay_order_created_at_epoch")
    created_display = None
    if created_epoch:
        try:
            created_display = datetime.fromtimestamp(int(created_epoch), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        except (ValueError, OSError, TypeError):
            created_display = str(created_epoch)

    cells = "".join(
        [
            cell("Order ID", result.get("razorpay_order_id")),
            cell("Amount", f"₹{amount:,.2f}" if amount is not None else None),
            cell("Currency", result.get("razorpay_order_currency")),
            cell("Status", (result.get("razorpay_order_status") or "").upper() or None),
            cell("Created", created_display),
        ]
    )
    return f'<div class="rl-kv-grid">{cells}</div>'


# ---------------------------------------------------------------------------
# Altair chart builders -- all consume real data passed in by the caller.
# ---------------------------------------------------------------------------

def roc_chart(roc_data: dict):
    frames = []
    series_order = []
    for series_name, key in [("XGBoost", "xgboost"), ("Logistic Regression", "baseline_logistic_regression")]:
        d = roc_data[key]
        df = pd.DataFrame({"fpr": d["fpr"], "tpr": d["tpr"]})
        label = f"{series_name} (AUC {d['auc']:.3f})"
        df["model"] = label
        series_order.append(label)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    diag = pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})

    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("fpr:Q", title="False positive rate", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("tpr:Q", title="True positive rate", scale=alt.Scale(domain=[0, 1])),
            # Explicit domain (not Altair's default alphabetical) so XGBoost --
            # the production model -- always gets the accent color regardless
            # of which model name happens to sort first.
            color=alt.Color("model:N", title=None, scale=alt.Scale(domain=series_order, range=[ACCENT, TEXT_MUTED])),
            tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("fpr:Q", format=".2f"), alt.Tooltip("tpr:Q", format=".2f")],
        )
    )
    baseline = alt.Chart(diag).mark_line(strokeDash=[4, 4], color=BORDER).encode(x="fpr:Q", y="tpr:Q")
    return (line + baseline).properties(
        height=300, autosize=alt.AutoSizeParams(type="fit-x", contains="padding")
    ).interactive()


def confusion_matrix_chart(cm_data: dict):
    labels = cm_data["labels"]
    matrix = cm_data["matrix"]
    rows = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            rows.append({"actual": actual, "predicted": predicted, "count": matrix[i][j]})
    df = pd.DataFrame(rows)

    heat = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("predicted:N", title="Predicted label", sort=labels),
            y=alt.Y("actual:N", title="Actual label", sort=list(reversed(labels))),
            color=alt.Color("count:Q", scale=alt.Scale(range=[SURFACE_2, ACCENT]), legend=None),
            tooltip=[alt.Tooltip("actual:N"), alt.Tooltip("predicted:N"), alt.Tooltip("count:Q")],
        )
    )
    text = alt.Chart(df).mark_text(fontSize=15, fontWeight="bold").encode(
        x=alt.X("predicted:N", sort=labels), y=alt.Y("actual:N", sort=list(reversed(labels))),
        text="count:Q", color=alt.condition(alt.datum.count > (max(m for row in matrix for m in row) / 2), alt.value("white"), alt.value(TEXT)),
    )
    return (heat + text).properties(
        height=240, autosize=alt.AutoSizeParams(type="fit-x", contains="padding")
    )


def shap_global_chart(shap_global: list, top_n: int = 8):
    df = pd.DataFrame(shap_global[:top_n])
    df["feature"] = df["feature"].apply(_factor_label)
    chart = (
        alt.Chart(df)
        .mark_bar(color=ACCENT, cornerRadiusEnd=3)
        .encode(
            x=alt.X("mean_abs_shap:Q", title="Mean |SHAP value| (test set)"),
            y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=160, labelPadding=4)),
            tooltip=[alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("mean_abs_shap:Q", title="Mean |SHAP|", format=".4f")],
        )
        .properties(height=28 * len(df) + 20, autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))
    )
    return chart


def risk_activity_chart(df: pd.DataFrame):
    """df: columns timestamp_utc (datetime), risk_score (float), decision (str).

    Tick format and count adapt to how much real time the data actually
    spans -- a demo session where several cases were scored seconds apart
    needs second-level ticks, not the day/month ticks that would suit a
    dataset spanning weeks. Without this, Vega-Lite's default temporal
    axis can pick a tick unit finer than a second and render fractional-
    second labels like ".500", which reads as broken rather than precise.
    """
    color_scale = alt.Scale(
        domain=[DECISION_STYLE[k]["label"] for k in [DECISION_CLEAR, DECISION_ESCALATE, DECISION_FLAG, DECISION_MANUAL_REVIEW]],
        range=[GREEN, AMBER, ORANGE, RED],
    )
    df = df.copy()
    df["decision_label"] = df["decision"].map(lambda d: DECISION_STYLE.get(d, {}).get("label", d))

    span_seconds = (df["timestamp_utc"].max() - df["timestamp_utc"].min()).total_seconds()
    if span_seconds < 3600:
        time_format, tooltip_format = "%H:%M:%S", "%H:%M:%S"
    elif span_seconds < 86400:
        time_format, tooltip_format = "%H:%M", "%H:%M:%S"
    else:
        time_format, tooltip_format = "%b %d", "%b %d, %H:%M"
    tick_count = min(len(df), 8)

    points = (
        alt.Chart(df)
        .mark_circle(size=90, opacity=0.85)
        .encode(
            x=alt.X("timestamp_utc:T", title=None, axis=alt.Axis(format=time_format, labelAngle=0, tickCount=tick_count)),
            y=alt.Y("risk_score:Q", title="Risk score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("decision_label:N", title="Decision", scale=color_scale),
            tooltip=[alt.Tooltip("timestamp_utc:T", title="Time", format=tooltip_format), alt.Tooltip("risk_score:Q", format=".3f"), alt.Tooltip("decision_label:N", title="Decision")],
        )
    )
    line = alt.Chart(df).mark_line(color=BORDER, strokeWidth=1).encode(x="timestamp_utc:T", y="risk_score:Q")
    escalate_rule = alt.Chart(pd.DataFrame({"y": [ESCALATE_THRESHOLD]})).mark_rule(strokeDash=[3, 3], color=AMBER).encode(y="y:Q")
    flag_rule = alt.Chart(pd.DataFrame({"y": [FLAG_THRESHOLD]})).mark_rule(strokeDash=[3, 3], color=ORANGE).encode(y="y:Q")
    return (line + points + escalate_rule + flag_rule).properties(
        height=260, autosize=alt.AutoSizeParams(type="fit-x", contains="padding")
    ).interactive()


def risk_distribution_chart(counts: dict):
    order = [DECISION_CLEAR, DECISION_ESCALATE, DECISION_FLAG, DECISION_MANUAL_REVIEW]
    df = pd.DataFrame(
        [{"decision": DECISION_STYLE[k]["label"], "count": counts.get(k, 0), "color": DECISION_STYLE[k]["color"]} for k in order]
    )
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3, height=22)
        .encode(
            x=alt.X("count:Q", title="Audit events"),
            y=alt.Y("decision:N", sort=None, title=None, axis=alt.Axis(labelLimit=140, labelPadding=4)),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[alt.Tooltip("decision:N"), alt.Tooltip("count:Q")],
        )
        .properties(height=160, autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))
    )
    return chart


def decision_volume_chart(df: pd.DataFrame):
    """df: columns timestamp_utc (datetime), decision (str).

    Stacked bar of decision volume over time, bucketed adaptively (like
    risk_activity_chart) so a demo session spanning a few minutes doesn't
    collapse into a single bar the way a fixed daily bucket would.
    """
    df = df.copy()
    df["decision_label"] = df["decision"].map(lambda d: DECISION_STYLE.get(d, {}).get("label", d or "Unknown"))

    span_seconds = (df["timestamp_utc"].max() - df["timestamp_utc"].min()).total_seconds()
    if span_seconds < 3600:
        freq, time_format = "1min", "%H:%M"
    elif span_seconds < 86400:
        freq, time_format = "15min", "%H:%M"
    else:
        freq, time_format = "1D", "%b %d"

    df["bucket"] = df["timestamp_utc"].dt.floor(freq)
    grouped = df.groupby(["bucket", "decision_label"]).size().reset_index(name="count")

    color_scale = alt.Scale(
        domain=[DECISION_STYLE[k]["label"] for k in [DECISION_CLEAR, DECISION_ESCALATE, DECISION_FLAG, DECISION_MANUAL_REVIEW]],
        range=[GREEN, AMBER, ORANGE, RED],
    )
    return (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=alt.X("bucket:T", title=None, axis=alt.Axis(format=time_format, labelAngle=0)),
            y=alt.Y("count:Q", title="Decisions"),
            color=alt.Color("decision_label:N", title="Decision", scale=color_scale),
            tooltip=[
                alt.Tooltip("bucket:T", title="Time", format=time_format),
                alt.Tooltip("decision_label:N", title="Decision"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(height=260, autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))
    )


def model_comparison_table_html(xgb: dict, base: dict, left_label: str = "XGBoost", right_label: str = "Logistic Regression") -> str:
    # "Higher wins" only makes sense for metrics where bigger is genuinely
    # better. Decision threshold isn't one of those -- it's just each
    # model's own tuned operating point, not a competition score -- so it's
    # rendered plainly below, with no winner dot or highlight color (a
    # previous version compared it like the others, which meant "XGBoost
    # wins" purely because its threshold happened to be a bigger number).
    comparable_rows = [
        ("Precision", xgb["precision"], base["precision"]),
        ("Recall", xgb["recall"], base["recall"]),
        ("F1", xgb["f1"], base["f1"]),
        ("ROC-AUC", xgb["roc_auc"], base["roc_auc"]),
    ]
    body_rows = "".join(
        f'<tr><td style="padding:9px 14px 9px 0; color:var(--rl-text-dim); border-top:1px solid var(--rl-border);">{label}</td>'
        f'<td style="padding:9px 14px; text-align:right; font-family:\'Space Grotesk\',sans-serif; font-weight:600; border-top:1px solid var(--rl-border); color:{"var(--rl-text)" if xg >= bl else "var(--rl-text-dim)"};">{xg:.3f}{" &#9679;" if xg >= bl else ""}</td>'
        f'<td style="padding:9px 0 9px 14px; text-align:right; font-family:\'Space Grotesk\',sans-serif; font-weight:600; border-top:1px solid var(--rl-border); color:{"var(--rl-text)" if bl > xg else "var(--rl-text-dim)"};">{bl:.3f}{" &#9679;" if bl > xg else ""}</td></tr>'
        for label, xg, bl in comparable_rows
    )
    body_rows += (
        f'<tr><td style="padding:9px 14px 9px 0; color:var(--rl-text-dim); border-top:1px solid var(--rl-border);">Decision threshold '
        f'<span style="color:var(--rl-text-muted); font-size:0.68rem;">(own tuned point, not a competition score)</span></td>'
        f'<td style="padding:9px 14px; text-align:right; font-family:\'Space Grotesk\',sans-serif; font-weight:600; border-top:1px solid var(--rl-border); color:var(--rl-text-dim);">{xgb["threshold"]:.3f}</td>'
        f'<td style="padding:9px 0 9px 14px; text-align:right; font-family:\'Space Grotesk\',sans-serif; font-weight:600; border-top:1px solid var(--rl-border); color:var(--rl-text-dim);">{base["threshold"]:.3f}</td></tr>'
    )
    return f"""
    <table style="width:100%; border-collapse:collapse; font-size:0.86rem;">
        <thead><tr>
            <th style="text-align:left; padding:0 14px 8px 0; color:var(--rl-text-muted); font-size:0.68rem; text-transform:uppercase; letter-spacing:0.05em;">Metric</th>
            <th style="text-align:right; padding:0 14px 8px; color:var(--rl-text-muted); font-size:0.68rem; text-transform:uppercase; letter-spacing:0.05em;">{left_label}</th>
            <th style="text-align:right; padding:0 0 8px 14px; color:var(--rl-text-muted); font-size:0.68rem; text-transform:uppercase; letter-spacing:0.05em;">{right_label}</th>
        </tr></thead>
        <tbody>{body_rows}</tbody>
    </table>
    <p style="font-size:0.72rem; color:var(--rl-text-muted); margin-top:8px;">&#9679; marks the stronger model on that metric.</p>
    """
