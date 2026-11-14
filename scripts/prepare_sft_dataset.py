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
import statistics
from collections import Counter, defaultdict
import sys
from pathlib import Path

# 确保仓库根目录在 Python 模块搜索路径中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer

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

    trajectory_step = row.get("trajectory_step")
    if type(trajectory_step) is not int or trajectory_step < 0:
        raise ValueError(f"Row {index}: trajectory_step must be a non-negative integer")



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
    max_length: int,
    tokenizer_name: str,
    excluded_path: Path,
    audit_path: Path,
) -> dict:
    """Build a provenance-preserving manifest for a length-filtered dataset.

    summary.chains describes source trajectories, not the number of unique
    derived task IDs represented by the retained rows. Filtering individual
    trajectory steps must therefore never overwrite that source-level count.
    """
    source_summary = dict(original_manifest.get("summary") or {})
    source_truncated = source_summary.get(
        "source_truncated_chains_reported",
        source_summary.get("truncated_chains"),
    )
    kept_task_ids = {row["derived_task_id"] for row in kept_rows}
    original_task_ids = {row["derived_task_id"] for row in original_rows}

    summary = dict(source_summary)
    summary.update({
        "rows": len(kept_rows),
        "original_rows": len(original_rows),
        "excluded_rows": len(excluded_rows),
        "unique_derived_task_ids": len(kept_task_ids),
        "original_unique_derived_task_ids": len(original_task_ids),
        # Length filtering removes rows; it does not truncate trajectories.
        "truncated_chains": None,
        "source_truncated_chains_reported": source_truncated,
    })

    manifest = dict(original_manifest)
    manifest.update({
        "kind": "sft_v4_multiturn",
        "version": 2,
        "parent_dataset": input_path.as_posix(),
        "parent_sha256": input_hash,
        "output": output_path.as_posix(),
        "output_sha256": output_hash,
        "summary": summary,
        "transformation": {
            "name": "token_length_filter",
            "max_length": max_length,
            "tokenizer": tokenizer_name,
            "token_count_method": (
                "apply_non_thinking_chat_template followed by tokenizer"
            ),
            "original_rows": len(original_rows),
            "kept_rows": len(kept_rows),
            "excluded_rows": len(excluded_rows),
            "excluded_file": excluded_path.as_posix(),
            "length_audit_file": audit_path.as_posix(),
        },
    })
    return manifest


def main():
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

    if original_manifest.get("kind") != "sft_v4_multiturn":
        raise ValueError(
            f"Unexpected manifest kind: {original_manifest.get('kind')}"
        )

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

    print("Original rows:", len(rows))
    print("Original SHA256:", input_hash)

    print("\n===== Load Tokenizer =====")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    assert_default_template_is_non_thinking(tokenizer)

    print("Tokenizer loaded:", args.model)

    print("\n===== Calculate Token Lengths =====")

    kept = []
    excluded = []
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
            "trajectory_step": row["trajectory_step"],
            "token_length": length,
        }

        audit_rows.append(audit)

        if length <= args.max_length:
            kept.append(row)
        else:
            excluded.append(row)

        if index % 250 == 0:
            print(f"Processed {index}/{len(rows)}")

    lengths = [r["token_length"] for r in audit_rows]
    kept_lengths = [
        r["token_length"]
        for r in audit_rows
        if r["token_length"] <= args.max_length
    ]

    excluded_lengths = [
        r["token_length"]
        for r in audit_rows
        if r["token_length"] > args.max_length
    ]

    if not kept:
        raise ValueError("No samples remain after filtering")

    print("\n===== Filtering Summary =====")

    print("Max length:", args.max_length)
    print("Original:", len(rows))
    print("Kept:", len(kept))
    print("Excluded:", len(excluded))
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
        if audit["token_length"] > args.max_length:
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
