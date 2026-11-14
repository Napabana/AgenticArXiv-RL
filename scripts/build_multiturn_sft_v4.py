#!/usr/bin/env python3
"""多轮 SFT 数据 v4：同任务决策链 + 快照 Observation，对齐 runner 推理分布。

根因 #2（plan.md S3 复盘）：v3 及之前的轨迹是「单轮决策」样本
（user=任务+手册，assistant=一次 Thought/Action），训练出的 27B 不会
「读 Observation 后继续/修正」，benchmark 一半任务迭代耗尽。

本脚本把同一派生任务的多条决策行串成链：

1. 按 (source_task_id, 语言视图) 分组、按 trajectory_step 排序；
2. 依次用 MockArxivEnv（离线快照回放）执行每步 Action，捕获确定性 Observation；
3. 每一步 k 生成一条训练样本：user = runner 同款 build_messages(task, 手册,
   历史(步骤 1..k-1 含 Observation))，assistant = 该步的 gold 决策——
   **训练前缀与推理前缀逐字节同分布**（含会话状态块）；
4. 快照缺失等 env 错误 → 链在该步截断（与 benchmark 行为一致）。

用法::

    python scripts/build_multiturn_sft_v4.py --output data/sft/sft_v4_multiturn.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))

import os as _os

_os.environ.setdefault("STORE_BACKEND", "memory")


class GoldDecisionParseError(ValueError):
    """A gold SFT decision is neither a valid tool call nor a terminal action."""


class SnapshotMissError(RuntimeError):
    """The replay snapshot does not contain a required tool result."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_declared_path(value: str, declaring_manifest: Path) -> Path:
    """Resolve a manifest path without depending on the caller's cwd."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate
    return declaring_manifest.parent / candidate


def resolve_task_manifest_path(
    seed_data_path: Path,
    task_manifest_path: Path | None = None,
) -> Path:
    """Locate the expert-seed manifest that owns derived task setup metadata."""
    if task_manifest_path is not None:
        return Path(task_manifest_path)

    linguistic_manifest_path = seed_data_path.with_suffix(
        seed_data_path.suffix + ".manifest.json"
    )
    if not linguistic_manifest_path.exists():
        raise FileNotFoundError(
            f"缺少语言扩增 manifest: {linguistic_manifest_path}"
        )
    linguistic_manifest = json.loads(
        linguistic_manifest_path.read_text(encoding="utf-8")
    )
    declared_path = linguistic_manifest.get("input_manifest")
    if not declared_path:
        raise ValueError(
            f"语言扩增 manifest 未声明 input_manifest: {linguistic_manifest_path}"
        )
    resolved = _resolve_declared_path(str(declared_path), linguistic_manifest_path)
    if not resolved.exists():
        raise FileNotFoundError(f"派生任务 manifest 不存在: {resolved}")

    expected_hash = linguistic_manifest.get("input_manifest_sha256")
    if expected_hash and sha256_file(resolved) != expected_hash:
        raise ValueError(
            f"派生任务 manifest SHA256 不匹配: {resolved}"
        )
    return resolved


def load_task_map(
    seed_data_path: Path,
    task_manifest_path: Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Load the derived-task declarations used to rebuild setup state."""
    manifest_path = resolve_task_manifest_path(seed_data_path, task_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"派生任务 manifest 缺少 tasks: {manifest_path}")

    task_map: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        task_id = str(task.get("derived_task_id") or "")
        if not task_id:
            raise ValueError(f"派生任务缺少 derived_task_id: {task!r}")
        if task_id in task_map:
            raise ValueError(f"派生任务 id 重复: {task_id}")
        setup = task.get("setup") or []
        if not isinstance(setup, list):
            raise ValueError(f"派生任务 setup 不是列表: {task_id}")
        task_map[task_id] = dict(task)
    return task_map


def _execute_agent_action(agent: Any, action: Dict[str, Any], session_id: str) -> Any:
    """执行动作并把 Agent 返回的失败 Observation 恢复成异常。"""
    agent.session_id = session_id
    observation = agent._execute_with_side_effects(action)
    if isinstance(observation, str) and (
        "[MockArxivEnv] replay 模式下快照缺失:" in observation
    ):
        raise SnapshotMissError(observation)
    if isinstance(observation, str) and observation.startswith(
        ("工具执行失败:", "错误:")
    ):
        raise RuntimeError(observation)
    return observation


def prepare_setup(agent: Any, task: Dict[str, Any], session_id: str) -> str:
    """Apply setup to the chain session and return its policy-visible context."""
    from agents.prompt_templates import build_visible_setup_context

    task_id = str(task.get("derived_task_id") or task.get("id") or "<unknown>")
    for action in task.get("setup") or []:
        name = str(action.get("name") or "")
        args = dict(action.get("args") or {})
        try:
            _execute_agent_action(
                agent,
                {"name": name, "args": args},
                session_id,
            )
        except Exception as exc:  # noqa: BLE001 - preserve setup context and cause
            raise RuntimeError(
                f"任务 {task_id} 的 setup 执行失败: "
                f"tool={name}, args={args}, error={exc}"
            ) from exc
    return build_visible_setup_context({"setup": task.get("setup") or []})


def advance_gold_decision(
    agent: Any,
    decision: str,
    session_id: str,
) -> tuple[str, Any | None]:
    """Execute one gold call, while treating FINISH as normal termination."""
    from rl.grpo_reward import parse_react_action

    kind, action = parse_react_action(decision)
    if kind == "parse_error":
        raise GoldDecisionParseError(
            f"无法解析 gold decision: {decision[:200]!r}"
        )
    if kind == "finish":
        return kind, None

    observation = _execute_agent_action(agent, action, session_id)
    return kind, observation


def main() -> None:
    parser = argparse.ArgumentParser(description="多轮 SFT v4 数据构建")
    parser.add_argument("--seed-data", default="data/sft/sft_v3_t5_parametric_linguistic.jsonl")
    parser.add_argument(
        "--task-manifest",
        default=None,
        help="派生任务 seed manifest；默认从语言扩增 manifest 的 input_manifest 解析",
    )
    parser.add_argument("--snapshot", default="data/mock_arxiv_snapshot.json")
    parser.add_argument("--output", default="data/sft/sft_v4_multiturn.jsonl")
    args = parser.parse_args()

    from tools.bootstrap import register_all_tools
    from tools.tool_registry import registry
    from agents.prompt_templates import format_tool_description

    register_all_tools()
    canonical_desc = format_tool_description(registry.list_tools())

    from agents.agent_engine import ReActAgent
    from agents.side_effects import LocalSideEffectManager
    from utils.llm_client import LLMClient
    from rl.env import MockArxivEnv
    class _Fake(LLMClient):
        def __init__(self):
            pass

        def chat_completions(self, model, messages, **kw):
            return {"choices": [{"message": {"role": "assistant", "content": ""}}], "usage": {}}

    agent = ReActAgent(_Fake(), side_effect_mgr=LocalSideEffectManager(), env=None)

    rows = []
    with Path(args.seed_data).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    seed_data_path = Path(args.seed_data)
    task_manifest_path = resolve_task_manifest_path(
        seed_data_path,
        Path(args.task_manifest) if args.task_manifest else None,
    )
    task_map = load_task_map(seed_data_path, task_manifest_path)

    # 分组键：派生血缘（同一任务同一语言视图 = 一条链）
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        aug = row.get("augmentation")
        aug_key = json.dumps(aug, ensure_ascii=False, sort_keys=True) if aug else ""
        key = (row.get("source_task_id") or row.get("parent_task_id") or row["sample_sha256"][:16], aug_key)
        groups[key].append(row)

    snapshot_path = Path(args.snapshot)
    chains_total = steps_total = 0
    completed = 0
    truncated = 0
    failure_counts: Counter[str] = Counter()
    out_rows: List[Dict[str, Any]] = []
    task_re = __import__("re").compile(r"当前任务：(.+)")

    for gi, (key, group) in enumerate(sorted(groups.items())):
        group.sort(key=lambda r: r.get("trajectory_step") or 0)
        # 任务文本就在行内（「当前任务：…」行），无需查任务集
        first_user = [m for m in group[0]["messages"] if m["role"] == "user"][0]["content"]
        m_task = task_re.search(first_user)
        if not m_task:
            raise ValueError(f"链 {key!r} 的 user prompt 缺少“当前任务”字段")
        task_text = m_task.group(1).strip()

        derived_task_id = str(
            group[0].get("source_task_id")
            or group[0].get("derived_task_id")
            or ""
        )
        if not derived_task_id or derived_task_id not in task_map:
            raise KeyError(
                f"链 {key!r} 无法映射到派生任务 manifest: {derived_task_id!r}"
            )
        spec = task_map[derived_task_id]

        env = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")
        session_id = f"v4_{gi}"
        agent.session_id = session_id
        agent.env = env
        # setup 既要真实铺设 session 状态，也要按 benchmark 契约渲染为可见上下文。
        history_text = prepare_setup(agent, spec, session_id)
        chain_failure = None
        terminated = False
        for row_index, row in enumerate(group):
            assistant = [m for m in row["messages"] if m["role"] == "assistant"]
            if not assistant:
                continue
            decision = assistant[-1]["content"]
            # 每步一条训练样本：历史(不含本步) + 本步 gold 决策
            msgs, _extra = agent.build_messages(task_text, canonical_desc, history_text)
            out_rows.append({
                "messages": [{"role": "user", "content": msgs[0]["content"]},
                             {"role": "assistant", "content": decision}],
                "source": "v4_multiturn",
                "derived_task_id": row.get("derived_task_id"),
                "trajectory_step": row.get("trajectory_step"),
            })
            steps_total += 1

            # 推进链：FINISH 是正常终止；解析/快照/工具错误分别记录。
            try:
                kind, obs = advance_gold_decision(agent, decision, session_id)
            except GoldDecisionParseError as exc:
                chain_failure = "parse_error"
                print(
                    f"  [WARN] chain={key!r} step={row_index} parse_error={exc}",
                    flush=True,
                )
                break
            except SnapshotMissError as exc:
                chain_failure = "snapshot_miss"
                print(
                    f"  [WARN] chain={key!r} step={row_index} snapshot_miss={exc}",
                    flush=True,
                )
                break
            except Exception as exc:  # noqa: BLE001 - classified and kept auditable
                chain_failure = "tool_error"
                print(
                    f"  [WARN] chain={key!r} step={row_index} tool_error={exc}",
                    flush=True,
                )
                break

            if kind == "finish":
                terminated = True
                if row_index != len(group) - 1:
                    chain_failure = "premature_finish"
                break

            obs_text = obs if isinstance(obs, str) else json.dumps(obs, ensure_ascii=False, default=str)
            history_text = (
                f"{history_text}\n\n{decision}\nObservation: {obs_text}".strip()
                if history_text else f"{decision}\nObservation: {obs_text}"
            )
        if chain_failure is None and not terminated:
            chain_failure = "missing_finish"
        if chain_failure is not None:
            truncated += 1
            failure_counts[chain_failure] += 1
        else:
            completed += 1
        chains_total += 1
        if (gi + 1) % 25 == 0:
            print(f"  [{gi+1}/{len(groups)}] 链 {chains_total} | 样本 {steps_total} | 截断链 {truncated}", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "kind": "sft_v4_multiturn",
        "version": 2,
        "seed_data": Path(args.seed_data).as_posix(),
        "seed_sha256": sha256_file(Path(args.seed_data)),
        "task_manifest": task_manifest_path.as_posix(),
        "task_manifest_sha256": sha256_file(task_manifest_path),
        "snapshot": Path(args.snapshot).as_posix(),
        "snapshot_sha256": sha256_file(snapshot_path),
        "output": output_path.as_posix(),
        "output_sha256": sha256_file(output_path),
        "summary": {
            "chains": chains_total,
            "rows": len(out_rows),
            "completed_chains": completed,
            "truncated_chains": truncated,
            "failure_counts": dict(sorted(failure_counts.items())),
            "note": "user 内容 = runner 同款 build_messages（含会话状态块与 Observation 历史），与推理逐字节同分布",
        },
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SFT_V4_MULTITURN_OK")
    print(
        f"chains: {chains_total} | rows: {len(out_rows)} | "
        f"completed: {completed} | truncated: {truncated}"
    )
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
