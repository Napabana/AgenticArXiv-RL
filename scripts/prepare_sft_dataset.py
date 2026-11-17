#!/usr/bin/env python3
"""Prepare a length-filtered, auditable SFT dataset.

- Preserve the original JSONL.
- Count tokens with the exact project chat-template path.
- Exclude samples whose full rendered conversation exceeds max_length.
- Preserve excluded samples separately.
- Generate a new manifest compatible with train_sft.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
import sys
from pathlib import Path

# 确保仓库根目录在 Python 模块搜索路径中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from AgenticArxiv.rl.model_loading import (
    apply_non_thinking_chat_template,
    assert_default_template_is_non_thinking,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}: {exc}"
                ) from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def token_length(tokenizer, messages: list[dict]) -> int:
    # 与 train_sft.py 的 _token_length 保持一致：
    # 先使用项目模板渲染，再通过 tokenizer 计算长度。
    text = apply_non_thinking_chat_template(
        tokenizer,
        messages,
        tokenize=False,
    )
    return len(tokenizer(text)["input_ids"])


def percentile(values: list[int], p: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[min(int(len(ordered) * p), len(ordered) - 1)]


def validate_row(row: dict, index: int) -> None:
    messages = row.get("messages")

    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"Row {index}: expected exactly 2 messages")

    if not all(isinstance(message, dict) for message in messages):
        raise ValueError(f"Row {index}: messages must contain objects")

    roles = [m.get("role") for m in messages]

    if roles != ["user", "assistant"]:
        raise ValueError(f"Row {index}: invalid roles {roles}")

    if not isinstance(messages[0].get("content"), str):
        raise ValueError(f"Row {index}: invalid user content")

    if not isinstance(messages[1].get("content"), str):
        raise ValueError(f"Row {index}: invalid assistant content")

    if not messages[1]["content"].strip():
        raise ValueError(f"Row {index}: empty assistant completion")

    derived_task_id = row.get("derived_task_id")
    if not isinstance(derived_task_id, str) or not derived_task_id.strip():
        raise ValueError(f"Row {index}: missing or invalid derived_task_id")

    if row.get("source") != "v4_multiturn":
        raise ValueError(f"Row {index}: unexpected source")

    chain_id = row.get("chain_id")
    if not isinstance(chain_id, str) or not chain_id.strip():
        raise ValueError(f"Row {index}: missing or invalid chain_id")

    trajectory_step = row.get("trajectory_step")
    if type(trajectory_step) is not int or trajectory_step < 0:
        raise ValueError(f"Row {index}: trajectory_step must be a non-negative integer")


def validate_source_manifest(manifest: dict) -> dict:
    """Accept only audited builder-v2 sources with complete gold trajectories."""
    if manifest.get("kind") != "sft_v4_multiturn" or manifest.get("version") != 2:
        raise ValueError(
            "旧版 SFT v4 Manifest 不可用于重新训练；请用修复后的 "
            "build_multiturn_sft_v4.py 生成 version=2 Manifest"
        )
    summary = manifest.get("summary") or {}
    required = ("chains", "completed_chains", "truncated_chains", "failure_counts", "unique_chain_ids")
    missing = [key for key in required if key not in summary]
    if missing:
        raise ValueError(f"SFT v4 source Manifest 缺少审计字段: {missing}")
    if summary["completed_chains"] != summary["chains"]:
        raise ValueError(
            "SFT v4 source 存在未完成轨迹: "
            f"chains={summary['chains']}, completed={summary['completed_chains']}"
        )
    if summary["truncated_chains"] != 0 or summary["failure_counts"]:
        raise ValueError(
            "SFT v4 source 含截断/失败轨迹，禁止进入长度过滤和训练: "
            f"truncated={summary['truncated_chains']}, "
            f"failure_counts={summary['failure_counts']}"
        )
    if summary["unique_chain_ids"] != summary["chains"]:
        raise ValueError(
            "SFT v4 source chain_id 数量与 chains 不一致: "
            f"unique_chain_ids={summary['unique_chain_ids']}, chains={summary['chains']}"
        )
    return summary


def _is_finish_decision(content: str) -> bool:
    match = re.search(r"Action:\s*(.*?)(?=\nObservation:|$)", content, re.DOTALL)
    if not match:
        return False
    action_text = match.group(1).strip()
    if action_text.upper() == "FINISH":
        return True
    try:
        action = json.loads(action_text)
    except json.JSONDecodeError:
        return False
    return isinstance(action, dict) and str(action.get("name", "")).upper() == "FINISH"


def validate_complete_chains(rows: list[dict]) -> dict:
    """Require contiguous decision steps and a terminal FINISH in every chain."""
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["chain_id"]].append((index, row))
    for chain_id, entries in grouped.items():
        entries.sort(key=lambda item: item[1]["trajectory_step"])
        steps = [row["trajectory_step"] for _, row in entries]
        if steps != list(range(len(entries))):
            raise ValueError(
                f"Chain {chain_id}: trajectory_step 不连续或重复: {steps}"
            )
        task_ids = {row["derived_task_id"] for _, row in entries}
        if len(task_ids) != 1:
            raise ValueError(f"Chain {chain_id}: derived_task_id 不一致: {task_ids}")
        final_content = entries[-1][1]["messages"][1]["content"]
        if not _is_finish_decision(final_content):
            raise ValueError(f"Chain {chain_id}: 最后一步不是 Action: FINISH")
    return {"original_chains": len(grouped)}


def filter_complete_chains(
    rows: list[dict],
    audit_rows: list[dict],
    max_length: int,
) -> tuple[list[dict], list[dict], dict]:
    """Drop a whole chain if any one of its rendered samples exceeds max_length."""
    if len(rows) != len(audit_rows):
        raise ValueError("rows 与 audit_rows 数量不一致")
    validate_complete_chains(rows)
    grouped_indices = defaultdict(list)
    for index, row in enumerate(rows):
        grouped_indices[row["chain_id"]].append(index)

    excluded_chain_ids = {
        chain_id
        for chain_id, indices in grouped_indices.items()
        if any(audit_rows[index]["token_length"] > max_length for index in indices)
    }
    kept = []
    excluded = []
    for index, row in enumerate(rows):
        chain_excluded = row["chain_id"] in excluded_chain_ids
        audit_rows[index]["chain_excluded"] = chain_excluded
        audit_rows[index]["filter_reason"] = (
            "chain_has_overlength_step" if chain_excluded else None
        )
        (excluded if chain_excluded else kept).append(row)

    chain_stats = {
        "original_chains": len(grouped_indices),
        "kept_chains": len(grouped_indices) - len(excluded_chain_ids),
        "excluded_chains": len(excluded_chain_ids),
    }
    return kept, excluded, chain_stats


def build_filtered_manifest(
    original_manifest: dict,
    *,
    input_path: Path,
    output_path: Path,
    input_hash: str,
    output_hash: str,
    original_rows: list[dict],
    kept_rows: list[dict],
    excluded_rows: list[dict],
    chain_stats: dict,
    max_length: int,
    tokenizer_name: str,
    excluded_path: Path,
    audit_path: Path,
) -> dict:
    """Build a manifest for a length filter that preserves whole trajectories."""
    source_summary = dict(original_manifest.get("summary") or {})
    kept_task_ids = {row["derived_task_id"] for row in kept_rows}
    original_task_ids = {row["derived_task_id"] for row in original_rows}
    kept_chain_ids = {row["chain_id"] for row in kept_rows}
    excluded_chain_ids = {row["chain_id"] for row in excluded_rows}

    summary = dict(source_summary)
    summary.update({
        "source_chains": source_summary.get("chains"),
        "chains": chain_stats["kept_chains"],
        "completed_chains": chain_stats["kept_chains"],
        "truncated_chains": 0,
        "failure_counts": {},
        "rows": len(kept_rows),
        "original_rows": len(original_rows),
        "excluded_rows": len(excluded_rows),
        "unique_derived_task_ids": len(kept_task_ids),
        "original_unique_derived_task_ids": len(original_task_ids),
        "unique_chain_ids": len(kept_chain_ids),
        "kept_chains": chain_stats["kept_chains"],
        "excluded_chains": chain_stats["excluded_chains"],
        "source_completed_chains": source_summary.get("completed_chains"),
        "source_truncated_chains_reported": source_summary.get("truncated_chains"),
        "length_filter_policy": "drop_entire_chain_if_any_row_exceeds_max_length",
    })

    manifest = dict(original_manifest)
    manifest.update({
        "kind": "sft_v4_multiturn",
        "version": 3,
        "parent_dataset": input_path.as_posix(),
        "parent_sha256": input_hash,
        "output": output_path.as_posix(),
        "output_sha256": output_hash,
        "summary": summary,
        "transformation": {
            "name": "chain_aware_token_length_filter",
            "max_length": max_length,
            "tokenizer": tokenizer_name,
            "token_count_method": (
                "apply_non_thinking_chat_template followed by tokenizer"
            ),
            "original_rows": len(original_rows),
            "kept_rows": len(kept_rows),
            "excluded_rows": len(excluded_rows),
            "original_chains": chain_stats["original_chains"],
            "kept_chains": chain_stats["kept_chains"],
            "excluded_chains": chain_stats["excluded_chains"],
            "chain_filter_policy": "drop_entire_chain_if_any_row_exceeds_max_length",
            "excluded_file": excluded_path.as_posix(),
            "length_audit_file": audit_path.as_posix(),
        },
    })
    return manifest


def main():
    # Keep manifest/schema helpers importable in the lightweight CPU data
    # environment. Tokenizer dependencies are needed only for the CLI path.
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/sft/sft_v4_multiturn.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/sft/sft_v4_multiturn_6144.jsonl",
    )
    parser.add_argument(
        "--model",
        default="/mnt/workspace/models/Qwen3.5-2B",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=6144,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(
        str(input_path) + ".manifest.json"
    )
    output_manifest_path = Path(
        str(output_path) + ".manifest.json"
    )

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    print("===== Input Audit =====")

    original_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    source_summary = validate_source_manifest(original_manifest)

    input_hash = sha256_file(input_path)

    if input_hash != original_manifest.get("output_sha256"):
        raise ValueError("Original dataset SHA256 mismatch")

    rows = read_jsonl(input_path)

    expected_rows = (
        original_manifest.get("summary") or {}
    ).get("rows")

    if expected_rows != len(rows):
        raise ValueError(
            f"Original row count mismatch: "
            f"manifest={expected_rows}, actual={len(rows)}"
        )

    if len(rows) != source_summary["rows"]:
        raise ValueError(
            f"Source rows 与 Manifest 不一致: manifest={source_summary['rows']}, actual={len(rows)}"
        )
    validate_complete_chains(rows)

    print("Original rows:", len(rows))
    print("Original chains:", source_summary["chains"])
    print("Original SHA256:", input_hash)

    print("\n===== Load Tokenizer =====")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    assert_default_template_is_non_thinking(tokenizer)

    print("Tokenizer loaded:", args.model)

    print("\n===== Calculate Token Lengths =====")

    audit_rows = []

    for index, row in enumerate(rows, 1):
        validate_row(row, index)

        length = token_length(
            tokenizer,
            row["messages"],
        )

        audit = {
            "index": index,
            "derived_task_id": row["derived_task_id"],
            "chain_id": row["chain_id"],
            "trajectory_step": row["trajectory_step"],
            "token_length": length,
            "over_length": length > args.max_length,
        }
        audit_rows.append(audit)

        if index % 250 == 0:
            print(f"Processed {index}/{len(rows)}")

    kept, excluded, chain_stats = filter_complete_chains(
        rows,
        audit_rows,
        args.max_length,
    )
    lengths = [audit["token_length"] for audit in audit_rows]
    kept_lengths = [
        audit["token_length"] for audit in audit_rows
        if not audit["chain_excluded"]
    ]
    excluded_lengths = [
        audit["token_length"] for audit in audit_rows
        if audit["chain_excluded"]
    ]
    overlength_rows = sum(1 for audit in audit_rows if audit["over_length"])

    if not kept:
        raise ValueError("No samples remain after filtering")

    print("\n===== Filtering Summary =====")

    print("Max length:", args.max_length)
    print("Original rows:", len(rows))
    print("Kept rows:", len(kept))
    print("Excluded rows (whole chains):", len(excluded))
    print("Original chains:", chain_stats["original_chains"])
    print("Kept chains:", chain_stats["kept_chains"])
    print("Excluded chains:", chain_stats["excluded_chains"])
    print("Over-length individual rows:", overlength_rows)
    print("Keep ratio:", f"{len(kept)/len(rows):.2%}")

    print("\n===== Original Length Distribution =====")

    print("Median:", percentile(lengths, 0.5))
    print("P90:", percentile(lengths, 0.9))
    print("Max:", max(lengths))

    print("\n===== Kept Length Distribution =====")

    print("Median:", percentile(kept_lengths, 0.5))
    print("P90:", percentile(kept_lengths, 0.9))
    print("Max:", max(kept_lengths))

    print("\n===== Excluded Samples =====")

    for audit in audit_rows:
        if audit["chain_excluded"]:
            print(audit)

    print("\n===== Trajectory Step Distribution =====")

    original_steps = Counter(
        r["trajectory_step"] for r in rows
    )
    kept_steps = Counter(
        r["trajectory_step"] for r in kept
    )

    for step in sorted(original_steps):
        print(
            f"step={step}: "
            f"original={original_steps[step]}, "
            f"kept={kept_steps[step]}, "
            f"excluded={original_steps[step]-kept_steps[step]}"
        )

    print("\n===== Save Artifacts =====")

    write_jsonl(output_path, kept)

    excluded_path = Path(
        str(output_path) + ".excluded.jsonl"
    )
    write_jsonl(excluded_path, excluded)

    audit_path = Path(
        str(output_path) + ".lengths.jsonl"
    )
    write_jsonl(audit_path, audit_rows)

    output_hash = sha256_file(output_path)

    output_manifest = build_filtered_manifest(
        original_manifest,
        input_path=input_path,
        output_path=output_path,
        input_hash=input_hash,
        output_hash=output_hash,
        original_rows=rows,
        kept_rows=kept,
        excluded_rows=excluded,
        max_length=args.max_length,
        tokenizer_name=args.model,
        excluded_path=excluded_path,
        audit_path=audit_path,
        chain_stats=chain_stats,
    )

    output_manifest_path.write_text(
        json.dumps(
            output_manifest,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n===== Final Verification =====")

    saved_rows = read_jsonl(output_path)
    saved_hash = sha256_file(output_path)

    assert len(saved_rows) == len(kept)
    assert saved_hash == output_hash
    assert len(saved_rows) + len(excluded) == len(rows)

    assert all(
        token_length(tokenizer, r["messages"]) <= args.max_length
        for r in saved_rows
    )

    print("Output rows:", len(saved_rows))
    print("Output SHA256:", saved_hash)
    print("Manifest:", output_manifest_path)
    print("Excluded:", excluded_path)
    print("Length audit:", audit_path)

    print("\nSFT_DATA_PREPARATION_OK")


if __name__ == "__main__":
    main()
