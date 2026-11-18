"""Validate chain-aware v4 filtered-data manifests before SFT."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rl.train_sft import _verify_data_manifest


class SftV4ManifestValidationTest(unittest.TestCase):
    def _write_case(self, summary, *, rows=None, version=3):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        data = Path(temp.name) / "train.jsonl"
        if rows is None:
            rows = [
                {
                    "messages": [
                        {"role": "user", "content": "task one"},
                        {"role": "assistant", "content": 'Thought: search\nAction: {"name":"search_arxiv_papers","args":{}}'},
                    ],
                    "derived_task_id": "derived-a",
                    "chain_id": "chain-a",
                    "source": "v4_multiturn",
                    "trajectory_step": 0,
                },
                {
                    "messages": [
                        {"role": "user", "content": "task two"},
                        {"role": "assistant", "content": "Thought: done\nAction: FINISH"},
                    ],
                    "derived_task_id": "derived-a",
                    "chain_id": "chain-a",
                    "source": "v4_multiturn",
                    "trajectory_step": 1,
                },
            ]
        payload = "".join(json.dumps(row) + "\n" for row in rows)
        data.write_text(payload, encoding="utf-8")
        manifest_path = Path(str(data) + ".manifest.json")
        manifest_path.write_text(json.dumps({
            "kind": "sft_v4_multiturn",
            "version": version,
            "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "summary": {"rows": len(rows), **summary},
            "transformation": {
                "chain_filter_policy": "drop_entire_chain_if_any_row_exceeds_max_length"
            },
        }), encoding="utf-8")
        return data

    def _valid_summary(self):
        return {
            "chains": 1,
            "completed_chains": 1,
            "truncated_chains": 0,
            "failure_counts": {},
            "unique_chain_ids": 1,
            "kept_chains": 1,
            "excluded_chains": 0,
            "unique_derived_task_ids": 1,
            "original_rows": 2,
            "excluded_rows": 0,
            "source_truncated_chains_reported": 0,
        }

    def test_accepts_complete_chain_aware_manifest(self):
        data = self._write_case(self._valid_summary())
        manifest = _verify_data_manifest(data)
        self.assertEqual(manifest["summary"]["chains"], 1)

    def test_rejects_legacy_v1_manifest(self):
        data = self._write_case(
            {"chains": 1032, "truncated_chains": 1032},
            version=1,
        )
        with self.assertRaisesRegex(SystemExit, "chain-aware version=3"):
            _verify_data_manifest(data)

    def test_rejects_missing_chain_id(self):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "Thought: done\nAction: FINISH"},
                ],
                "derived_task_id": "derived-a",
                "source": "v4_multiturn",
                "trajectory_step": 0,
            }
        ]
        data = self._write_case(self._valid_summary(), rows=rows)
        with self.assertRaisesRegex(SystemExit, "chain_id"):
            _verify_data_manifest(data)

    def test_rejects_non_contiguous_chain_steps(self):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "task one"},
                    {"role": "assistant", "content": 'Thought: search\nAction: {"name":"search_arxiv_papers","args":{}}'},
                ],
                "derived_task_id": "derived-a",
                "chain_id": "chain-a",
                "source": "v4_multiturn",
                "trajectory_step": 0,
            },
            {
                "messages": [
                    {"role": "user", "content": "task two"},
                    {"role": "assistant", "content": "Thought: done\nAction: FINISH"},
                ],
                "derived_task_id": "derived-a",
                "chain_id": "chain-a",
                "source": "v4_multiturn",
                "trajectory_step": 2,
            },
        ]
        data = self._write_case(self._valid_summary(), rows=rows)
        with self.assertRaisesRegex(SystemExit, "不连续"):
            _verify_data_manifest(data)

    def test_rejects_chain_without_terminal_finish(self):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "task one"},
                    {"role": "assistant", "content": 'Thought: search\nAction: {"name":"search_arxiv_papers","args":{}}'},
                ],
                "derived_task_id": "derived-a",
                "chain_id": "chain-a",
                "source": "v4_multiturn",
                "trajectory_step": 0,
            }
        ]
        data = self._write_case(self._valid_summary(), rows=rows)
        with self.assertRaisesRegex(SystemExit, "FINISH"):
            _verify_data_manifest(data)

    def test_rejects_nonzero_truncation_count(self):
        summary = self._valid_summary()
        summary["truncated_chains"] = 1
        data = self._write_case(summary)
        with self.assertRaisesRegex(SystemExit, "未完成/失败轨迹"):
            _verify_data_manifest(data)


if __name__ == "__main__":
    unittest.main()
