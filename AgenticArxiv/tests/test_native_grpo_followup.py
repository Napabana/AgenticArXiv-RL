"""The follow-up pilot requires model-generated, target-specific training signal."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_native_grpo_followup_probe import audit
from build_native_grpo_tasks import CACHE, SEARCH
from tools.bootstrap import require_all_tools


def test_followup_gate_requires_both_regressions_and_reward_ordering():
    require_all_tools("test")
    a = {"id": "wide", "decision_focus": "search_scope", "expected_tool_args": [{"aspect": "*"}]}
    b = {"id": "pair", "decision_focus": "two_paper_cache", "expected_paper_ids": ["p1", "p2"]}

    def group(task, name, args, obs, index):
        sample = {"termination_reason": "finish", "history": [
            {"action": json.dumps({"name": name, "args": args}), "observation": json.dumps(obs)}]}
        return {"task_id": task["id"], "group": index, "seed": index, "step_credit": {"mode": "reward"},
                "strict": [True, True, True, False], "trajectories": [None, None, None, sample],
                "reward_breakdowns": [{"total": total} for total in (1, 1, 1, .75)]}

    groups = [group(a, SEARCH, {"aspect": "CL"}, {}, i) for i in (0, 1)]
    groups += [group(b, CACHE, {"ref": 2}, {"paper_id": "p1"}, i) for i in (2, 3)]
    def decide():
        return audit({"tasks": [a, b]}, {"status": "GPU_PASS", "selected_task_ids": ["wide", "pair"]},
                     groups)["decision"]

    assert decide() == "run_dev_pilot"
    groups[-1]["reward_breakdowns"][-1]["total"] = 1
    assert decide() == "stop_before_training"
    groups[-1]["reward_breakdowns"][-1]["total"] = .75
    groups[-1]["trajectories"][-1]["termination_reason"] = "token_budget"
    assert decide() == "stop_before_training"

