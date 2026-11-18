"""CPU-only tests for chain-preserving SFT length filtering."""

import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_sft_dataset import (
    build_filtered_manifest,
    filter_complete_chains,
    validate_source_manifest,
)


def _row(chain_id, task_id, step, action):
    return {
        "messages": [
            {"role": "user", "content": f"task {task_id}, step {step}"},
            {"role": "assistant", "content": f"Thought: step {step}\nAction: {action}"},
        ],
        "derived_task_id": task_id,
        "chain_id": chain_id,
        "source": "v4_multiturn",
        "trajectory_step": step,
    }


class FilteredSftManifestTest(unittest.TestCase):
    def test_source_manifest_requires_audited_complete_builder_v2(self):
        valid = {
            "kind": "sft_v4_multiturn",
            "version": 2,
            "summary": {
                "chains": 2,
                "completed_chains": 2,
                "truncated_chains": 0,
                "failure_counts": {},
                "unique_chain_ids": 2,
                "rows": 4,
            },
        }
        self.assertEqual(validate_source_manifest(valid)["chains"], 2)

        legacy = {
            "kind": "sft_v4_multiturn",
            "version": 1,
            "summary": {"chains": 1032, "truncated_chains": 1032},
        }
        with self.assertRaisesRegex(ValueError, "version=2"):
            validate_source_manifest(legacy)

    def test_source_manifest_rejects_truncated_or_failed_chains(self):
        invalid = {
            "kind": "sft_v4_multiturn",
            "version": 2,
            "summary": {
                "chains": 2,
                "completed_chains": 1,
                "truncated_chains": 1,
                "failure_counts": {"tool_error": 1},
                "unique_chain_ids": 2,
                "rows": 3,
            },
        }
        with self.assertRaisesRegex(ValueError, "未完成轨迹|截断/失败轨迹"):
            validate_source_manifest(invalid)

    def test_length_filter_drops_entire_chain_if_any_step_is_over_limit(self):
        rows = [
            _row("chain-a", "task-a", 0, '{"name":"search_arxiv_papers","args":{}}'),
            _row("chain-a", "task-a", 1, "FINISH"),
            _row("chain-b", "task-b", 0, '{"name":"search_arxiv_papers","args":{}}'),
            _row("chain-b", "task-b", 1, "FINISH"),
        ]
        audits = [
            {"chain_id": "chain-a", "token_length": 120},
            {"chain_id": "chain-a", "token_length": 7000},
            {"chain_id": "chain-b", "token_length": 300},
            {"chain_id": "chain-b", "token_length": 400},
        ]

        kept, excluded, stats = filter_complete_chains(rows, audits, 6144)

        self.assertEqual([row["chain_id"] for row in kept], ["chain-b", "chain-b"])
        self.assertEqual([row["chain_id"] for row in excluded], ["chain-a", "chain-a"])
        self.assertEqual(stats, {
            "original_chains": 2,
            "kept_chains": 1,
            "excluded_chains": 1,
        })
        self.assertTrue(all(audit["chain_excluded"] for audit in audits[:2]))
        self.assertTrue(all(not audit["chain_excluded"] for audit in audits[2:]))

    def test_filter_rejects_non_contiguous_or_non_terminal_chains(self):
        non_contiguous = [
            _row("chain-a", "task-a", 0, '{"name":"search_arxiv_papers","args":{}}'),
            _row("chain-a", "task-a", 2, "FINISH"),
        ]
        audits = [
            {"chain_id": "chain-a", "token_length": 100},
            {"chain_id": "chain-a", "token_length": 100},
        ]
        with self.assertRaisesRegex(ValueError, "不连续"):
            filter_complete_chains(non_contiguous, audits, 6144)

        no_finish = [
            _row("chain-b", "task-b", 0, '{"name":"search_arxiv_papers","args":{}}'),
            _row("chain-b", "task-b", 1, '{"name":"get_paper_cache_status","args":{}}'),
        ]
        audits = [
            {"chain_id": "chain-b", "token_length": 100},
            {"chain_id": "chain-b", "token_length": 100},
        ]
        with self.assertRaisesRegex(ValueError, "FINISH"):
            filter_complete_chains(no_finish, audits, 6144)

    def test_filtered_manifest_records_chain_level_exclusions(self):
        original_rows = [
            _row("chain-a", "task-a", 0, '{"name":"search_arxiv_papers","args":{}}'),
            _row("chain-a", "task-a", 1, "FINISH"),
            _row("chain-b", "task-b", 0, '{"name":"search_arxiv_papers","args":{}}'),
            _row("chain-b", "task-b", 1, "FINISH"),
        ]
        kept_rows = original_rows[2:]
        excluded_rows = original_rows[:2]
        source = {
            "kind": "sft_v4_multiturn",
            "version": 2,
            "summary": {
                "chains": 2,
                "completed_chains": 2,
                "truncated_chains": 0,
                "failure_counts": {},
                "unique_chain_ids": 2,
                "rows": 4,
            },
        }
        manifest = build_filtered_manifest(
            source,
            input_path=Path("input.jsonl"),
            output_path=Path("filtered.jsonl"),
            input_hash="parent-hash",
            output_hash="output-hash",
            original_rows=original_rows,
            kept_rows=kept_rows,
            excluded_rows=excluded_rows,
            chain_stats={
                "original_chains": 2,
                "kept_chains": 1,
                "excluded_chains": 1,
            },
            max_length=6144,
            tokenizer_name="Qwen3.5-2B",
            excluded_path=Path("filtered.excluded.jsonl"),
            audit_path=Path("filtered.lengths.jsonl"),
        )

        summary = manifest["summary"]
        self.assertEqual(manifest["version"], 3)
        self.assertEqual(summary["chains"], 1)
        self.assertEqual(summary["source_chains"], 2)
        self.assertEqual(summary["completed_chains"], 1)
        self.assertEqual(summary["truncated_chains"], 0)
        self.assertEqual(summary["failure_counts"], {})
        self.assertEqual(summary["unique_chain_ids"], 1)
        self.assertEqual(summary["excluded_chains"], 1)
        self.assertEqual(summary["original_rows"], 4)
        self.assertEqual(summary["excluded_rows"], 2)
        self.assertEqual(
            manifest["transformation"]["chain_filter_policy"],
            "drop_entire_chain_if_any_row_exceeds_max_length",
        )

    def test_does_not_mutate_source_manifest(self):
        source = {
            "kind": "sft_v4_multiturn",
            "version": 2,
            "summary": {"chains": 2, "completed_chains": 2, "truncated_chains": 0},
        }
        before = {"kind": source["kind"], "version": source["version"],
                  "summary": dict(source["summary"])}
        build_filtered_manifest(
            source,
            input_path=Path("input.jsonl"),
            output_path=Path("filtered.jsonl"),
            input_hash="parent-hash",
            output_hash="output-hash",
            original_rows=[
                _row("chain-a", "task-a", 0, "FINISH"),
                _row("chain-b", "task-b", 0, "FINISH"),
            ],
            kept_rows=[_row("chain-b", "task-b", 0, "FINISH")],
            excluded_rows=[_row("chain-a", "task-a", 0, "FINISH")],
            chain_stats={"original_chains": 2, "kept_chains": 1, "excluded_chains": 1},
            max_length=6144,
            tokenizer_name="Qwen3.5-2B",
            excluded_path=Path("filtered.excluded.jsonl"),
            audit_path=Path("filtered.lengths.jsonl"),
        )
        self.assertEqual(source, before)


if __name__ == "__main__":
    unittest.main()
