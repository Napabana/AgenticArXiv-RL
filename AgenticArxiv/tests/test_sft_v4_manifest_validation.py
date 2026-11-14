"""Validate v4 filtered-data manifest semantics before SFT."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rl.train_sft import _verify_data_manifest


class SftV4ManifestValidationTest(unittest.TestCase):
    def _write_case(self, summary):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        data = Path(temp.name) / "train.jsonl"
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "task one"},
                    {"role": "assistant", "content": "action one"},
                ],
                "derived_task_id": "derived-a",
                "source": "v4_multiturn",
                "trajectory_step": 0,
            },
            {
                "messages": [
                    {"role": "user", "content": "task two"},
                    {"role": "assistant", "content": "action two"},
                ],
                "derived_task_id": "derived-a",
                "source": "v4_multiturn",
                "trajectory_step": 1,
            },
        ]
        payload = "".join(json.dumps(row) + "\n" for row in rows)
        data.write_text(payload, encoding="utf-8")
        manifest_path = Path(str(data) + ".manifest.json")
        manifest_path.write_text(json.dumps({
            "kind": "sft_v4_multiturn",
            "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "summary": {"rows": 2, **summary},
        }), encoding="utf-8")
        return data

    def test_source_chain_count_is_not_compared_to_unique_task_ids(self):
        data = self._write_case({
            "chains": 1032,
            "unique_derived_task_ids": 1,
            "original_rows": 3,
            "excluded_rows": 1,
            "truncated_chains": None,
            "source_truncated_chains_reported": 1032,
        })
        manifest = _verify_data_manifest(data)
        self.assertEqual(manifest["summary"]["chains"], 1032)

    def test_rejects_wrong_unique_task_id_count(self):
        data = self._write_case({"unique_derived_task_ids": 2})
        with self.assertRaisesRegex(SystemExit, "unique_derived_task_ids"):
            _verify_data_manifest(data)

    def test_rejects_inconsistent_filter_row_accounting(self):
        data = self._write_case({"original_rows": 4, "excluded_rows": 1})
        with self.assertRaisesRegex(SystemExit, "过滤行数"):
            _verify_data_manifest(data)


if __name__ == "__main__":
    unittest.main()
