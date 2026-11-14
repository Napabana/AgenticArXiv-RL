"""Regression tests for the audited multi-turn SFT v4 builder."""

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts.build_multiturn_sft_v4 import (
    SnapshotMissError,
    advance_gold_decision,
    load_task_map,
    main,
    prepare_setup,
)


class RecordingAgent:
    def __init__(self, *, observation="ok"):
        self.calls = []
        self.observation = observation
        self.session_id = ""

    def _execute_with_side_effects(self, action):
        self.calls.append((self.session_id, dict(action)))
        return self.observation


class TaskManifestMappingTest(unittest.TestCase):
    def test_loads_derived_task_setup_from_seed_manifest_lineage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_data = root / "linguistic.jsonl"
            seed_data.write_text("", encoding="utf-8")
            task_manifest = root / "seed.manifest.json"
            task_payload = {
                "kind": "train_only_parametric_expert_seed",
                "tasks": [
                    {
                        "derived_task_id": "derived-1",
                        "task": "download the prior paper",
                        "setup": [
                            {
                                "name": "get_recently_submitted_cs_papers",
                                "args": {"aspect": "AI", "max_results": 5},
                            }
                        ],
                    }
                ],
            }
            task_manifest.write_text(
                json.dumps(task_payload), encoding="utf-8"
            )
            linguistic_manifest = Path(str(seed_data) + ".manifest.json")
            linguistic_manifest.write_text(
                json.dumps(
                    {
                        "input_manifest": task_manifest.name,
                        "input_manifest_sha256": hashlib.sha256(
                            task_manifest.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )

            task_map = load_task_map(seed_data)

        self.assertEqual(
            task_map["derived-1"]["setup"][0]["name"],
            "get_recently_submitted_cs_papers",
        )


class SetupReplayTest(unittest.TestCase):
    def test_setup_updates_the_session_and_returns_visible_context(self):
        agent = RecordingAgent()
        task = {
            "derived_task_id": "derived-1",
            "setup": [
                {
                    "name": "get_recently_submitted_cs_papers",
                    "args": {"aspect": "AI", "days": 7, "max_results": 5},
                }
            ],
        }

        visible_context = prepare_setup(agent, task, "session-1")

        self.assertEqual(
            agent.calls,
            [
                (
                    "session-1",
                    {
                        "name": "get_recently_submitted_cs_papers",
                        "args": {"aspect": "AI", "days": 7, "max_results": 5},
                    },
                )
            ],
        )
        self.assertIn("任务开始前的会话状态", visible_context)
        self.assertIn("此前已检索", visible_context)

    def test_setup_failure_is_not_replaced_with_empty_history(self):
        agent = RecordingAgent(
            observation="工具执行失败: missing snapshot key"
        )
        task = {
            "derived_task_id": "derived-1",
            "setup": [{"name": "search_arxiv_papers", "args": {"query": "x"}}],
        }

        with self.assertRaisesRegex(RuntimeError, "derived-1.*setup 执行失败"):
            prepare_setup(agent, task, "session-1")


class GoldDecisionTest(unittest.TestCase):
    def test_normal_finish_does_not_execute_a_tool(self):
        agent = RecordingAgent()

        kind, observation = advance_gold_decision(
            agent,
            "Thought: all required work is complete\nAction: FINISH",
            "session-1",
        )

        self.assertEqual(kind, "finish")
        self.assertIsNone(observation)
        self.assertEqual(agent.calls, [])

    def test_tool_action_uses_the_chain_session(self):
        agent = RecordingAgent(observation={"ok": True})

        kind, observation = advance_gold_decision(
            agent,
            'Thought: search first\nAction: {"name":"search_arxiv_papers",'
            '"args":{"query":"agent"}}',
            "session-1",
        )

        self.assertEqual(kind, "call")
        self.assertEqual(observation["ok"], True)
        self.assertEqual(
            agent.calls,
            [
                (
                    "session-1",
                    {
                        "name": "search_arxiv_papers",
                        "args": {"query": "agent"},
                    },
                )
            ],
        )

    def test_invalid_gold_action_fails_loudly(self):
        agent = RecordingAgent()
        with self.assertRaisesRegex(ValueError, "无法解析 gold decision"):
            advance_gold_decision(agent, "Thought: malformed", "session-1")

    def test_snapshot_miss_is_distinct_from_other_tool_failures(self):
        agent = RecordingAgent(
            observation=(
                "工具执行失败: [MockArxivEnv] replay 模式下快照缺失: "
                "tool=search_arxiv_papers key={}"
            )
        )

        with self.assertRaises(SnapshotMissError):
            advance_gold_decision(
                agent,
                'Thought: search first\nAction: {"name":"search_arxiv_papers",'
                '"args":{"query":"agent"}}',
                "session-1",
            )


class BuilderIntegrationTest(unittest.TestCase):
    def test_finish_only_chain_is_completed_not_truncated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_data = root / "linguistic.jsonl"
            seed_data.write_text(
                json.dumps(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "工具说明\n当前任务：无需调用工具，直接结束",
                            },
                            {
                                "role": "assistant",
                                "content": "Thought: 任务无需工具\nAction: FINISH",
                            },
                        ],
                        "source_task_id": "derived-finish",
                        "derived_task_id": "derived-finish",
                        "trajectory_step": 0,
                        "sample_sha256": "fixture",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            task_manifest = root / "seed.manifest.json"
            task_manifest.write_text(
                json.dumps(
                    {
                        "kind": "train_only_parametric_expert_seed",
                        "tasks": [
                            {
                                "derived_task_id": "derived-finish",
                                "task": "无需调用工具，直接结束",
                                "setup": [],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            Path(str(seed_data) + ".manifest.json").write_text(
                json.dumps(
                    {
                        "input_manifest": task_manifest.name,
                        "input_manifest_sha256": hashlib.sha256(
                            task_manifest.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            snapshot = root / "snapshot.json"
            snapshot.write_text("{}", encoding="utf-8")
            output = root / "output.jsonl"

            argv = [
                "build_multiturn_sft_v4.py",
                "--seed-data",
                str(seed_data),
                "--snapshot",
                str(snapshot),
                "--output",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                main()

            manifest = json.loads(
                Path(str(output) + ".manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["summary"]["chains"], 1)
        self.assertEqual(manifest["summary"]["completed_chains"], 1)
        self.assertEqual(manifest["summary"]["truncated_chains"], 0)
        self.assertEqual(manifest["summary"]["failure_counts"], {})


if __name__ == "__main__":
    unittest.main()
