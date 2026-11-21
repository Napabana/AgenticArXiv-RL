"""Diagnostics for the paired ReAct and native tool-calling benchmark."""

import json
import re
from collections import Counter, defaultdict
from typing import Any

from agents.base_agent import BaseAgent, is_terminal_action
from benchmark.metrics import _count_snapshot_coverage_errors, _count_tool_failures, _parse_tool_action, is_strict_success
from rl.reward import RewardCalculator
from tools.tool_registry import registry
from utils.llm_client import parse_tool_response


PROTOCOL_AGENTS = ("regex", "native_tool_calling")
_REACT_ACTION = re.compile(r"Action:\s*(.*?)(?=\nObservation:|$)", re.DOTALL)


def _parse_react_choice(choice: dict) -> dict | None:
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("missing_assistant_message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("missing_action")
    match = _REACT_ACTION.search(content)
    if match is None:
        raise ValueError("missing_action")
    action_text = match[1].strip()
    if is_terminal_action(action_text):
        return None
    action_json = re.search(r"({.*})", action_text, re.DOTALL)
    if action_json is None:
        raise ValueError("missing_action_json")
    try:
        action = json.loads(action_json[1])
    except (ValueError, RecursionError) as exc:
        raise ValueError("invalid_action_json") from exc
    if not isinstance(action, dict) or not isinstance(action.get("name"), str):
        raise ValueError("invalid_action")
    if is_terminal_action(action["name"]):
        return None
    if not action["name"].strip() or not isinstance(action.get("args"), dict):
        raise ValueError("invalid_action")
    return {"name": action["name"], "args": action["args"]}


def _audit_turn(response: dict, agent_type: str) -> dict:
    if not isinstance(response, dict):
        response = {}
    choices = response.get("choices") or []
    choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice, dict):
        choice = {}
    turn = {
        "model": response.get("model"),
        "response": response,
        "choice": choice,
        "format_ok": False,
        "schema_ok": None,
        "action": None,
        "issue": "",
    }
    try:
        if agent_type == "regex":
            action = _parse_react_choice(choice)
        else:
            _, _, action = parse_tool_response(choice)
    except (ValueError, RecursionError) as exc:
        turn["issue"] = f"format:{exc}"
        return turn
    turn["format_ok"] = True
    turn["action"] = action
    if action is not None:
        if agent_type == "native_tool_calling":
            tool = registry.model_tool(action["name"])
            normalized = registry.normalize_arguments(action["name"], action["args"])
            turn["normalized_args"] = normalized
            action = {**action, "args": normalized}
        else:
            tool = registry.get_tool(action["name"])
        valid, reason = (
            BaseAgent._validate_routed_action(action, tool) if tool else (False, "unknown_tool")
        )
        turn["schema_ok"] = valid
        if not valid:
            turn["issue"] = f"schema:{reason}"
    return turn


def score_protocol_trial(
    task: dict, raw: dict, agent_type: str, trial: int, session_id: str, responses: list[dict],
):
    """Keep raw protocol evidence and apply the existing scorer to a compatible history."""
    turns = [_audit_turn(response, agent_type) for response in responses]
    history = raw["history"]
    for turn, step in zip(turns, history):
        turn["runtime_action"] = step["action"]
        turn["recovered"] = not turn["format_ok"] and _parse_tool_action(step["action"]) is not None
        if not turn["format_ok"] and step["action"] == "ERROR":
            step["parse_failed"] = True

    if agent_type == "regex" and turns and history and history[-1]["action"] == "FINISH":
        if not turns[-1]["format_ok"]:
            raw["runtime_terminal_step"] = history[-1]
            history = [*history[:-1], {
                **history[-1],
                "action": "ERROR",
                "parse_failed": True,
                "observation": f"无法解析 ReAct 动作: {turns[-1]['issue']}",
            }]
            raw["history"] = history
            raw["final_observation"] = history[-1]["observation"]

    dispatched = []
    for step in history:
        action = _parse_tool_action(step["action"])
        if action and isinstance(action.get("name"), str) and registry.get_tool(action["name"]):
            dispatched.append(step)
    def signature(name, args):
        return name, json.dumps(registry.effective_arguments(name, args), sort_keys=True, ensure_ascii=False)

    expected_calls = Counter(
        signature(name, args or {}) for name, args in zip(
            task.get("expected_tools") or [], task.get("expected_tool_args") or [],
        )
    )
    actual_calls = Counter()
    pending = False
    for step in dispatched:
        action = _parse_tool_action(step["action"])
        name = action["name"]
        actual_calls[signature(name, action.get("args") or {})] += 1
        if name == "translate_arxiv_pdf" and not step.get("tool_issue"):
            pending |= "PENDING" in str(step.get("observation", ""))
    repeated = sum(max(0, count - max(1, expected_calls[key])) for key, count in actual_calls.items())
    raw["protocol"] = {
        "turns": turns,
        "unexpected_repeat_calls": repeated,
        "pending_submission": pending,
        "pending_finished": pending and bool(history) and history[-1]["action"] == "FINISH",
        "pending_guard_errors": sum(step.get("tool_issue") == "pending_guard" for step in dispatched),
        "tool_calls_executed": len(dispatched),
        "tool_execution_errors": _count_tool_failures(dispatched),
        "snapshot_coverage_errors": _count_snapshot_coverage_errors(dispatched),
        "request_error": any(
            step["action"] == "ERROR" and step["thought"] == "LLM调用失败" for step in history
        ),
    }
    breakdown, metrics = RewardCalculator().compute_reward_breakdown(
        task, raw, agent_type, trial, session_id,
    )
    raw["reward"] = breakdown.to_dict()
    return metrics


def _rate(count: int, total: int) -> dict:
    return {"count": count, "total": total, "rate": count / total if total else None}


def summarize_protocol_results(results: list) -> dict[str, Any]:
    """Report format/dispatch counts and strict outcomes for planned paired trials."""
    grouped = defaultdict(list)
    paired = defaultdict(dict)
    for result in results:
        if result.agent_type in PROTOCOL_AGENTS:
            grouped[result.agent_type].append(result)
            paired[result.task_id, result.trial][result.agent_type] = bool(
                result.metrics and is_strict_success(result.metrics)
            )
    agents = {}
    for name, runs in grouped.items():
        protocols = [run.raw_result.get("protocol", {}) for run in runs]
        turns = [turn for protocol in protocols for turn in protocol.get("turns", [])]
        metrics = [run.metrics for run in runs if run.metrics is not None]
        arg_scores = [metric.arg_score for metric in metrics if metric.arg_applicable]
        rewards = [run.raw_result["reward"]["total"] for run in runs if "reward" in run.raw_result]
        calls = sum(protocol.get("tool_calls_executed", 0) for protocol in protocols)
        agents[name] = {
            "format_parse_success": _rate(sum(turn["format_ok"] for turn in turns), len(turns)),
            "schema_compliance": _rate(
                sum(turn["schema_ok"] is True for turn in turns),
                sum(turn["schema_ok"] is not None for turn in turns),
            ),
            "tool_accuracy": _rate(sum(metric.tool_call_accurate for metric in metrics), len(metrics)),
            "argument_accuracy": {"score_sum": sum(arg_scores), "total": len(arg_scores),
                                  "rate": sum(arg_scores) / len(arg_scores) if arg_scores else None},
            "tool_error_rate": _rate(sum(p.get("tool_execution_errors", 0) for p in protocols), calls),
            "snapshot_coverage_errors": sum(p.get("snapshot_coverage_errors", 0) for p in protocols),
            "setup_failures": sum(m.setup_failures for m in metrics),
            "unexpected_repeat_calls": sum(p.get("unexpected_repeat_calls", 0) for p in protocols),
            "pending_guard_errors": sum(p.get("pending_guard_errors", 0) for p in protocols),
            "pending_finished": _rate(sum(p.get("pending_finished", False) for p in protocols),
                                      sum(p.get("pending_submission", False) for p in protocols)),
            "explained_refusal": _rate(sum(m.terminal_semantics_accurate for m in metrics
                                             if m.terminal_semantics_applicable),
                                       sum(m.terminal_semantics_applicable for m in metrics)),
            "strict_success": _rate(sum(is_strict_success(metric) for metric in metrics), len(runs)),
            "avg_executed_tool_calls": calls / len(runs),
            "avg_prompt_tokens": sum(metric.prompt_tokens for metric in metrics) / len(runs),
            "avg_completion_tokens": sum(metric.completion_tokens for metric in metrics) / len(runs),
            "avg_total_tokens": sum(metric.total_tokens for metric in metrics) / len(runs),
            "avg_llm_ms": sum(metric.total_llm_ms for metric in metrics) / len(runs),
            "avg_total_ms": sum(metric.total_time_ms for metric in metrics) / len(runs),
            "avg_reward": sum(rewards) / len(rewards) if rewards else None,
            "benchmark_errors": sum(bool(run.error) for run in runs),
            "request_errors": sum(p.get("request_error", False) for p in protocols),
            "error_terminations": sum(metric.termination_type == "ERROR" for metric in metrics),
            "issues": dict(Counter(turn["issue"] for turn in turns if turn["issue"])),
            "recovered_format_errors": sum(turn.get("recovered", False) for turn in turns),
            "returned_models": sorted({turn["model"] for turn in turns if turn["model"]}),
        }
        generation_stops = Counter(
            reason for turn in turns if (reason := turn["choice"].get("generation_stop_reason"))
        )
        if generation_stops:
            agents[name]["generation_stop_reasons"] = dict(generation_stops)
    complete_pairs = [pair for pair in paired.values() if all(name in pair for name in PROTOCOL_AGENTS)]
    return {
        "agents": agents,
        "paired_strict": {
            "total": len(complete_pairs),
            "fail_to_pass": sum(not p["regex"] and p["native_tool_calling"] for p in complete_pairs),
            "pass_to_fail": sum(p["regex"] and not p["native_tool_calling"] for p in complete_pairs),
        },
    }
