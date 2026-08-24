"""
The risk agent's reasoning loop.

This is the actual "agentic" part: instead of a fixed script, an LLM
(via Groq) is handed a transaction and a set of tools, and decides for
itself which tools to call and in what order before proposing a decision.

Important design choice, stated plainly: the agent PROPOSES a decision and
explains its reasoning, but it does not have final authority. The risk
score it computes (via the score_transaction_risk tool) is always re-checked
against the same deterministic gating engine (gating/decision_engine.py)
used everywhere else in RiskLens, and the deterministic gate's decision is
what's treated as ground truth -- not the agent's opinion. If the agent's
recommendation and the gate's decision disagree, both are logged, and the
gate wins.

This is deliberate, not a limitation: an LLM can misread a threshold,
hallucinate a number, or be prompt-injected via unusual transaction data.
Letting an LLM freely decide account actions with no independent check
would violate the buildathon's own "bounded and gated" requirement. Letting
it reason and explain, while a separate deterministic layer enforces the
actual boundaries, satisfies both "agentic" and "bounded and gated" at once.
"""

import json
import time
from datetime import datetime, timezone

from groq import Groq

from agent.tools import TOOL_SCHEMAS, RiskAgentTools
from config import require_groq_key
from gating.decision_engine import decide_from_score

# Groq's model lineup changes over time -- llama-3.3-70b-versatile was
# deprecated from their production lineup after this was first written.
# openai/gpt-oss-120b is Groq's current flagship production model with
# tool-use support; if Groq's lineup changes again, check
# https://console.groq.com/docs/models for the current model list.
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_TURNS = 6

SYSTEM_PROMPT = """You are RiskLens, a risk-review agent for a payments platform.

You will be given a transaction (a merchant_id and a transaction amount).
Your job: investigate using your tools, then propose a decision.

Required steps, in order:
1. Call get_merchant_context to learn the merchant's history.
2. Call score_transaction_risk with the merchant's context plus the transaction amount
   (use the transaction amount as daily_txn_volume).
3. Call explain_transaction_risk with the same fields, to get the reasoning behind the score.
4. Optionally call get_recent_audit_history if you want to see this merchant's past decisions.

You do NOT have the authority to freeze, block, or approve anything directly -- a separate
fixed safety system makes the final call using the risk score you compute. Your job is to
investigate thoroughly and propose a well-reasoned recommendation.

Once you have called score_transaction_risk and explain_transaction_risk, respond with ONLY
a JSON object (no other text) in this exact shape:
{"recommended_decision": "clear" | "escalate" | "flag_for_compliance_review", "reasoning": "<your reasoning in plain language>"}
"""


class AgentFailure(Exception):
    pass


def run_risk_agent(transaction: dict, tools: RiskAgentTools, groq_client=None, model: str = DEFAULT_MODEL) -> dict:
    """
    transaction: {"merchant_id": ..., "daily_txn_volume": ...} at minimum --
    this is the "live" half of the input (e.g. from a real Razorpay test order).

    Returns a dict with the agent's proposed decision, the final GATED decision
    (which is what actually counts), and a full trace of every tool call made,
    for the audit log.
    """
    client = groq_client or Groq(api_key=require_groq_key())

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(transaction)},
    ]

    trace = []
    computed_risk_score = None
    explanation = None
    top_factors = None

    for turn in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        if message.tool_calls:
            messages.append({"role": "assistant", "content": message.content or "", "tool_calls": message.tool_calls})
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                step_started = time.perf_counter()
                timestamp = datetime.now(timezone.utc).isoformat()
                try:
                    result = tools.call(name, args)
                    status = "error" if isinstance(result, dict) and "error" in result else "success"
                except Exception as exc:  # noqa: BLE001 -- tool failures must not crash the loop
                    result = {"error": str(exc)}
                    status = "error"
                duration_ms = round((time.perf_counter() - step_started) * 1000, 1)

                if name == "score_transaction_risk" and "risk_score" in result:
                    computed_risk_score = result["risk_score"]
                if name == "explain_transaction_risk" and "explanation" in result:
                    explanation = result["explanation"]
                    top_factors = result.get("top_factors")

                trace.append(
                    {
                        "tool": name,
                        "arguments": args,
                        "result": result,
                        "timestamp": timestamp,
                        "duration_ms": duration_ms,
                        "status": status,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str),
                    }
                )
            continue

        # No tool calls -- the agent believes it's ready to give a final answer.
        final_text = (message.content or "").strip()
        agent_proposal = _parse_final_json(final_text)
        gating_result = decide_from_score(computed_risk_score)

        agreement = (
            agent_proposal is not None
            and agent_proposal.get("recommended_decision") == gating_result.decision
        )

        return {
            "agent_proposal": agent_proposal,
            "agent_raw_response": final_text,
            "gated_decision": gating_result.decision,
            "gated_reason": gating_result.reason,
            "risk_score": computed_risk_score,
            "explanation": explanation,
            "top_factors": top_factors,
            "agent_and_gate_agree": agreement,
            "trace": trace,
        }

    # Ran out of turns without a final answer -- fail safe rather than guess.
    gating_result = decide_from_score(computed_risk_score)
    return {
        "agent_proposal": None,
        "agent_raw_response": None,
        "gated_decision": gating_result.decision if computed_risk_score is not None else "needs_manual_review",
        "gated_reason": (
            gating_result.reason
            if computed_risk_score is not None
            else "Agent did not complete its investigation within the allotted turns."
        ),
        "risk_score": computed_risk_score,
        "explanation": explanation,
        "top_factors": top_factors,
        "agent_and_gate_agree": None,
        "trace": trace,
    }


def _parse_final_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Agent didn't follow the format -- don't guess at its intent.
        return None
