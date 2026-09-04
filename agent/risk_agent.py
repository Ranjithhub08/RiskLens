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

from agent.tools import FINAL_ANSWER_TOOL, TOOL_SCHEMAS, RiskAgentTools
from config import require_groq_key
from gating.decision_engine import decide_from_score

# Groq's model lineup changes over time -- llama-3.3-70b-versatile was
# deprecated from their production lineup after this was first written.
# openai/gpt-oss-120b is Groq's current flagship production model with
# tool-use support; if Groq's lineup changes again, check
# https://console.groq.com/docs/models for the current model list.
DEFAULT_MODEL = "openai/gpt-oss-120b"
# Found live: every real Live Agent demo run observed so far ended with "No
# proposal" -- the agent completed its investigation (up to 5 tool calls:
# get_merchant_context, score_transaction_risk, explain_transaction_risk,
# plus the two optional lookups the system prompt below invites) but never
# actually called submit_decision, so run_risk_agent hit its "ran out of
# turns" fail-safe every single time. At the old MAX_TURNS=6, a thorough
# investigation that used both optional tools consumed all 6 turns on
# investigation alone, leaving zero turns for the model to ever submit a
# final answer -- the turn budget itself made the flagship "agent proposes,
# gate decides" comparison structurally unable to produce a proposal on
# exactly the most thorough (and most interesting to watch) runs. Raised to
# 8 for headroom, and paired with forcing tool_choice to submit_decision on
# the actual final turn (see the loop below) so running out of turns without
# a proposal should now require the model to keep calling investigative
# tools even when forced to answer -- not just an ordinary thorough
# investigation.
MAX_TURNS = 8

SYSTEM_PROMPT = """You are RiskLens, a risk-review agent for a payments platform operating
in India on Razorpay. Every transaction amount you see is in Indian Rupees (INR) -- when
you refer to an amount in your reasoning, write it in Rupees (e.g. "Rs 4,99,999" or
"499999 INR"), never in dollars, and never assume or imply a currency other than INR.

You will be given a transaction (a merchant_id and a transaction amount).
Your job: investigate using your tools, then propose a decision.

Required steps, in order:
1. Call get_merchant_context to learn the merchant's history.
2. Call score_transaction_risk with the merchant's context plus the transaction amount
   (use the transaction amount as daily_txn_volume).
3. Call explain_transaction_risk with the same fields, to get the reasoning behind the score.
4. Optionally call get_recent_audit_history if you want to see this merchant's own past decisions,
   and/or get_similar_past_cases to see how OTHER merchants in the same business category were
   decided -- if you use it, briefly note in your reasoning whether this case is consistent with
   that precedent or why it differs.

You do NOT have the authority to freeze, block, or approve anything directly -- a separate
fixed safety system makes the final call using the risk score you compute. Your job is to
investigate thoroughly and propose a well-reasoned recommendation.

Once you have called score_transaction_risk and explain_transaction_risk, call the
submit_decision tool exactly once with your recommended_decision and reasoning. That is
the only way to give your final answer -- do not write the JSON out as plain text instead.
"""


class AgentFailure(Exception):
    pass


def _emit(on_step, event: dict) -> None:
    """Fire a progress event to the live UI, if anyone is listening.

    on_step is an optional callback -- when the agent is being watched live
    (e.g. the Live Agent page), it gets called immediately as each thing
    happens (about to call the model, a tool call just finished, the agent
    submitted its answer), so the UI can render the investigation as it
    unfolds instead of only after run_risk_agent returns. Every existing
    caller that doesn't pass on_step is unaffected -- this is a no-op then.

    A bug in on_step itself (a UI-side rendering mistake, an assumption
    about the event dict that later drifts, a Streamlit widget-state
    exception) must never be allowed to propagate out of here: this is a
    best-effort progress notification, not part of the investigation
    itself, and letting it raise used to discard whatever score/
    explanation/trace had already been legitimately computed by that
    point -- exactly the same "already-computed result thrown away by an
    unrelated failure" class of bug already fixed for a mid-loop LLM-API
    error (see the try/except around client.chat.completions.create below).
    A broken listener should never silence or crash the speaker.
    """
    if on_step is not None:
        try:
            on_step(event)
        except Exception:
            pass


def summarize_tool_result(tool: str, result) -> str:
    """
    One-line human summary of a tool's result. Shared between the live
    on_step callback (shown while the agent is still investigating) and the
    post-hoc timeline/raw-trace rendered in the UI afterwards, so the two
    views never describe the same step differently.
    """
    res = result if isinstance(result, dict) else {}
    if isinstance(result, dict) and "error" in result:
        return f"error: {result['error']}"
    if tool == "score_transaction_risk":
        return f"risk_score = {res.get('risk_score'):.4f}" if res.get("risk_score") is not None else "no score returned"
    if tool == "explain_transaction_risk":
        return (res.get("explanation") or "")[:120]
    if tool == "get_merchant_context":
        return f"kyc={res.get('kyc_status')}, age={res.get('account_age_days')}d"
    if tool == "get_recent_audit_history":
        return f"{len(res.get('past_decisions', []))} prior decision(s)"
    if tool == "get_similar_past_cases":
        return f"{len(res.get('similar_cases', []))} similar case(s) in {res.get('current_business_category', 'this category')}"
    return str(result)[:120]


def run_risk_agent(
    transaction: dict,
    tools: RiskAgentTools,
    groq_client=None,
    model: str = DEFAULT_MODEL,
    on_step=None,
) -> dict:
    """
    transaction: {"merchant_id": ..., "daily_txn_volume": ...} at minimum --
    this is the "live" half of the input (e.g. from a real Razorpay test order).

    on_step: optional callback(event: dict) invoked in real time as the
    investigation progresses -- event["type"] is one of "thinking" (about to
    call the model), "tool_call" (a tool just returned), "final" (the agent
    submitted its recommendation), or "timeout" (ran out of turns). Lets a UI
    show the agent's reasoning live instead of only once this function returns.

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
        _emit(on_step, {"type": "thinking", "turn": turn + 1})
        # On the last available turn, stop letting the model choose freely
        # ("auto" -- which is exactly how it kept choosing to investigate
        # further, or to reply in plain prose, right up until the turn
        # budget ran out with no proposal ever submitted) and force it to
        # call submit_decision specifically. This still respects "the
        # score/decide_from_score gate is the real authority, not the
        # agent" -- forcing WHICH tool is called on the final turn doesn't
        # change what the gate does with the score, it just guarantees the
        # comparison this feature exists to show (agent's proposal vs. the
        # gate's decision) actually has an agent proposal to compare, on a
        # turn where the alternative was certain to end in "ran out of
        # turns, no proposal" anyway.
        is_last_turn = turn == MAX_TURNS - 1
        tool_choice = (
            {"type": "function", "function": {"name": FINAL_ANSWER_TOOL}} if is_last_turn else "auto"
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice=tool_choice,
            )
            message = response.choices[0].message
        except Exception:
            # A transient LLM-API failure (rate limit, network timeout, an
            # unexpected/empty response shape) on turn 2+ used to propagate
            # straight out of this function, DISCARDING whatever had
            # already been legitimately computed on earlier turns --
            # computed_risk_score, explanation, top_factors, and the full
            # tool-call trace -- even though every other "agent didn't
            # finish normally" exit path (the loop simply running out of
            # turns, two lines below the loop) deliberately preserves and
            # gates that same data instead of throwing it away. The net
            # effect was an opaque, evidence-free manual-review record with
            # no diagnostic trail, in place of a real gated decision the
            # already-computed score could have produced. Fail safe using
            # whatever is already known, the same way running out of turns
            # does, instead of letting a mid-loop API hiccup erase it.
            return _fail_safe_result(
                computed_risk_score, explanation, top_factors, trace, on_step,
                event_type="api_error",
                reason_without_score="The agent's reasoning could not be completed (a language-model API call failed) -- routed for manual review.",
            )

        if message.tool_calls:
            messages.append({"role": "assistant", "content": message.content or "", "tool_calls": message.tool_calls})
            submitted_proposal = None
            submitted_raw = None
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    # tool_call.function.arguments is documented/typical as
                    # a JSON string, but a provider/SDK anomaly (a
                    # malformed or truncated tool-call response) can hand
                    # back None instead of "{}" -- json.loads(None) raises
                    # TypeError, a different exception class than the
                    # JSONDecodeError this used to only guard against, so
                    # that case fell straight through uncaught and crashed
                    # the whole reasoning loop, discarding any
                    # already-computed score exactly like the other
                    # fail-safe gaps fixed in this file.
                    args = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}

                # submit_decision is how the model reports ITS OWN final answer, not
                # something for RiskLens to compute -- handle it separately from the
                # investigative tools (get_merchant_context, score_transaction_risk, ...)
                # dispatched through tools.call() below. Groq's own tool-use models
                # reliably call a real tool for structured final output; giving them
                # this one to call, instead of asking for a bare JSON text reply,
                # avoids them inventing an undeclared tool (e.g. a "json" tool) to do
                # the same thing, which the API then rejects outright.
                if name == FINAL_ANSWER_TOOL:
                    submitted_proposal = args
                    submitted_raw = tool_call.function.arguments
                    # Still needs a "tool" role reply so the message thread stays
                    # valid, in case the model isn't actually done (e.g. it called
                    # this alongside other tools this turn) and we loop again below.
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps({"received": True})}
                    )
                    continue

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
                _emit(
                    on_step,
                    {
                        "type": "tool_call",
                        "tool": name,
                        "status": status,
                        "summary": summarize_tool_result(name, result),
                        "duration_ms": duration_ms,
                    },
                )

            if submitted_proposal is not None:
                return _finalize(
                    submitted_proposal, submitted_raw, computed_risk_score, explanation, top_factors, trace, on_step
                )
            continue

        # No tool calls -- fall back to parsing the final answer as plain-text JSON,
        # in case the model ignores submit_decision and replies with bare text anyway.
        final_text = (message.content or "").strip()
        agent_proposal = _parse_final_json(final_text)
        return _finalize(agent_proposal, final_text, computed_risk_score, explanation, top_factors, trace, on_step)

    # Ran out of turns without a final answer -- fail safe rather than guess.
    return _fail_safe_result(
        computed_risk_score, explanation, top_factors, trace, on_step,
        event_type="timeout",
        reason_without_score="Agent did not complete its investigation within the allotted turns.",
    )


def _fail_safe_result(computed_risk_score, explanation, top_factors, trace, on_step, event_type: str, reason_without_score: str) -> dict:
    """
    Shared by every "the agent didn't reach a normal submit_decision call"
    exit path (running out of turns, and a mid-loop LLM API failure) --
    whatever was already legitimately computed (a risk score from
    score_transaction_risk, an explanation, and the tool-call trace so far)
    is preserved and gated for real, exactly as if the agent had reached
    that point and simply failed to submit a final answer, rather than
    being silently discarded. Only when NO score was ever computed does
    this fall back to a plain needs_manual_review with no comparison to
    make.
    """
    gating_result = decide_from_score(computed_risk_score)
    gated_decision = gating_result.decision if computed_risk_score is not None else "needs_manual_review"
    _emit(on_step, {"type": event_type, "gated_decision": gated_decision})
    return {
        "agent_proposal": None,
        "agent_raw_response": None,
        "gated_decision": gated_decision,
        "gated_reason": gating_result.reason if computed_risk_score is not None else reason_without_score,
        "risk_score": computed_risk_score,
        "explanation": explanation,
        "top_factors": top_factors,
        "agent_and_gate_agree": None,
        "trace": trace,
        "thresholds_used": gating_result.thresholds_used,
    }


def _finalize(agent_proposal, agent_raw_response, computed_risk_score, explanation, top_factors, trace, on_step=None) -> dict:
    # agent_proposal is untrusted: it's either the model's raw submit_decision
    # tool-call arguments, or the result of parsing its plain-text final reply
    # as JSON (see _parse_final_json). Neither path guarantees a dict -- a
    # provider could in principle return non-object tool-call arguments, and
    # a plain-text reply can be valid-but-non-object JSON like "42" or
    # "true". Treating anything that isn't a dict as "no proposal" (the same
    # outcome as the model not answering at all) means a malformed final
    # answer fails safe to the gate's own decision instead of crashing this
    # function on `.get()` -- which used to happen here, and worse, happened
    # before log_event runs upstream in agent_pipeline.py, so the case
    # silently never reached the audit log at all instead of landing in
    # needs_manual_review like every other failure mode in this module.
    if not isinstance(agent_proposal, dict):
        agent_proposal = None

    gating_result = decide_from_score(computed_risk_score)
    agreement = (
        agent_proposal is not None
        and agent_proposal.get("recommended_decision") == gating_result.decision
    )
    _emit(
        on_step,
        {
            "type": "final",
            "decision": agent_proposal.get("recommended_decision") if agent_proposal else None,
            "gated_decision": gating_result.decision,
            "agree": agreement,
        },
    )
    return {
        "agent_proposal": agent_proposal,
        "agent_raw_response": agent_raw_response,
        "gated_decision": gating_result.decision,
        "gated_reason": gating_result.reason,
        "risk_score": computed_risk_score,
        "explanation": explanation,
        "top_factors": top_factors,
        "agent_and_gate_agree": agreement,
        "thresholds_used": gating_result.thresholds_used,
        "trace": trace,
    }


def _parse_final_json(text: str):
    """
    Best-effort parse of a plain-text final answer into the same
    {recommended_decision, reasoning} shape submit_decision expects.

    Only reached when the model ignores submit_decision entirely and
    replies in plain text instead (the tool-call path above is what
    normally happens, and is what forcing tool_choice on the last turn is
    meant to make even more reliable) -- but plain-text replies observed
    in practice are usually "almost" valid JSON, not unstructured prose: a
    markdown code fence around an otherwise-valid object (` ```json
    {...} ``` `), or a JSON object with a sentence of commentary before or
    after it. Recovering those two specific, superficial formatting
    patterns is worthwhile; anything else correctly still falls through to
    None -- this must never GUESS at intent for text with no embedded JSON
    object at all, only strip formatting around JSON that's really there.
    """
    candidates = [text]

    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence line (```` ``` `` or ```` ```json ``` ``)
        # and the closing ```` ``` ````, if present.
        after_open = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        candidates.append(after_open.rsplit("```", 1)[0].strip())

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    # Agent didn't follow the format and no embedded JSON object was
    # recoverable -- don't guess at its intent.
    return None
