"""Semantic state of native replay, including what the sampled assistant actually saw."""

import hashlib
import json
from pathlib import Path

from rl.native_grpo_rollout import native_history


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def portable(value):
    if isinstance(value, dict):
        return {
            k: portable(v)
            for k, v in value.items()
            if k
            not in {
                "session_id",
                "updated_at",
                "created_at",
                "downloaded_at",
                "translated_at",
                "last_active_at",
                "timestamp",
            }
        }
    if isinstance(value, list):
        return [portable(v) for v in value]
    return value


def receipt(text):
    try:
        return portable(json.loads(text))
    except (ValueError, TypeError):
        return text


def canonical_visible(text):
    """Keep formatting and JSON field order unless lifecycle metadata needs normalization."""
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return text
    clean = portable(obj)
    return text if clean == obj else json.dumps(clean, ensure_ascii=False)


def environment(agent):
    store = agent.side_effects.store.capture_state()
    session = store["_sessions"].get(agent.session_id)

    def assets(key):
        values = {}
        for paper, item in store[key].items():
            fields = portable(item.model_dump(mode="json"))
            for k, v in fields.items():
                if k.endswith("_path") and v:
                    fields[k] = Path(v.replace(chr(92), "/")).name
            values[paper] = fields
        return values

    jobs = [
        portable(job.model_dump(mode="json", exclude={"task_id", "input_pdf_path", "output_pdf_path"}))
        for job in store["_tasks"].values()
    ]
    return dict(
        papers=[p.model_dump(mode="json") for p in session.papers] if session else [],
        active=session.last_active_paper_id if session else None,
        pdf=assets("_pdf_assets"),
        translation=assets("_translate_assets"),
        jobs=jobs,
        downloaded=sorted(
            p for sid, p in agent.env.capture_state()["_offline_downloaded"] if sid == agent.session_id
        ),
        status=dict(agent._translation_state),
        pending_checks=sorted(agent._pending_cache_checks),
    )


def fingerprints(rollout, task, sample, root, *, guard_states=None):
    """Replay recorded actions (no model inference), rejecting mismatched evidence.

    Assistant tokenization comes from the actual sampled IDs. Tool feedback is the frozen
    native template applied to its receipt. Nonsemantic lifecycle fields are canonicalized.
    """
    ep = rollout.episode(task, root)
    msgs = sample["native_messages"]
    n = next((i for i, m in enumerate(msgs) if m["role"] == "assistant"), len(msgs))

    # Setup text includes receipt JSON; compare after replacing only the verified receipts.
    def prompt(messages, receipts):
        out = [dict(m) for m in messages]
        for item in receipts:
            raw = item["observation"]
            normalized = canonical_visible(raw)
            for m in out:
                m["content"] = m["content"].replace(raw, normalized)
        return out

    visible = prompt(msgs[:n], sample["setup_receipts"])
    if ep.stop or visible != prompt(ep.prompt, ep.setup_receipts):
        raise ValueError("setup context mismatch")
    if native_history(msgs[n:], ep.setup_receipts) != sample["history"]:
        raise ValueError("native history/receipt mismatch")
    raw, spans = sample["raw_assistant_turns"], sample["turn_token_spans"]
    if len(raw) != len(spans) or len(spans) != len(sample["history"]):
        raise ValueError("assistant span count mismatch")
    context = digest([visible, ep.tools])
    states = []
    for i, (turn, span, step) in enumerate(zip(raw, spans, sample["history"], strict=True)):
        env_state = environment(ep.agent)
        states.append(digest([context, env_state]))
        if guard_states is not None:
            from rl.repair_credit import guard_state
            guard_states.append(guard_state(ep.agent, env_state, step))
        if i + 1 == len(raw):
            break
        if not ep.add_text(turn):
            raise ValueError("recorded continuation after terminal state")
        actual = ep.messages[-1]
        if receipt(actual["content"]) != receipt(step["observation"]) or actual.get("tool_issue") != step.get(
            "tool_issue"
        ):
            raise ValueError("replayed tool receipt differs")
        start, end = span
        feedback = sample["completion_ids"][end : spans[i + 1][0]]
        recorded = step["observation"]
        tokenizer = getattr(rollout, "audit_tokenizer", None)
        if tokenizer is not None:
            actual["content"] = recorded
            if rollout.feedback_ids(tokenizer, ep) != feedback:
                raise ValueError("stored context tokens differ from native feedback")
            actual["content"] = canonical_visible(recorded)
            if actual["content"] != recorded:
                feedback = rollout.feedback_ids(tokenizer, ep)
        context = digest([context, sample["completion_ids"][start:end], feedback])
    return states
