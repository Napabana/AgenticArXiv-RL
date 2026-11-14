"""CPU-only tests for filtered SFT manifest provenance."""

import unittest
from pathlib import Path

from scripts.prepare_sft_dataset import build_filtered_manifest


class FilteredSftManifestTest(unittest.TestCase):
    def _build(self, source_summary):
        original_rows = [
            {"derived_task_id": "task-a"},
            {"derived_task_id": "task-a"},
            {"derived_task_id": "task-b"},
        ]
        kept_rows = original_rows[:2]
        excluded_rows = original_rows[2:]
        return build_filtered_manifest(
            {"kind": "sft_v4_multiturn", "summary": source_summary},
            input_path=Path("input.jsonl"),
            output_path=Path("filtered.jsonl"),
            input_hash="parent-hash",
            output_hash="output-hash",
            original_rows=original_rows,
            kept_rows=kept_rows,
            excluded_rows=excluded_rows,
            max_length=6144,
            tokenizer_name="Qwen3.5-2B",
            excluded_path=Path("filtered.excluded.jsonl"),
            audit_path=Path("filtered.lengths.jsonl"),
        )

    def test_preserves_source_chain_count_and_separates_unique_task_ids(self):
        manifest = self._build({
            "chains": 1032,
            "rows": 3,
            "truncated_chains": 1032,
            "note": "same runner build_messages distribution",
        })
        summary = manifest["summary"]
        self.assertEqual(summary["chains"], 1032)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["unique_derived_task_ids"], 1)
        self.assertEqual(summary["original_unique_derived_task_ids"], 2)
        self.assertIsNone(summary["truncated_chains"])
        self.assertEqual(summary["source_truncated_chains_reported"], 1032)
        self.assertEqual(summary["note"], "same runner build_messages distribution")

    def test_preserves_previously_corrected_source_truncation_metadata(self):
        manifest = self._build({
            "chains": 1032,
            "rows": 3,
            "truncated_chains": None,
            "source_truncated_chains_reported": 1032,
        })
        summary = manifest["summary"]
        self.assertEqual(summary["chains"], 1032)
        self.assertIsNone(summary["truncated_chains"])
        self.assertEqual(summary["source_truncated_chains_reported"], 1032)

    def test_does_not_mutate_source_manifest(self):
        source = {"kind": "sft_v4_multiturn", "summary": {"chains": 1032}}
        before = {"kind": source["kind"], "summary": dict(source["summary"])}
        self._build(source["summary"])
        self.assertEqual(source, before)


if __name__ == "__main__":
    unittest.main()
