"""Snapshot defaults must match training paths regardless of the caller's cwd."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from rl import build_snapshot  # noqa: E402



class SnapshotReferenceTimeoutTest(unittest.TestCase):
    def test_pinned_reference_lookup_uses_arxiv_network_timeout(self):
        ai_original = [
            {"id": f"ai-existing-{i}", "title": f"AI {i}"} for i in range(5)
        ]
        cv_original = [
            {"id": f"cv-existing-{i}", "title": f"CV {i}"} for i in range(5)
        ]
        env = SimpleNamespace(snapshot={
            "get_recently_submitted_cs_papers": {
                "ai": {"args": {"aspect": "AI"}, "result": ai_original},
                "cv": {"args": {"aspect": "CV"}, "result": cv_original},
            }
        })
        wanted = [
            "2608.14539v1", "2608.14530v1", "2608.14528v1",
            "2608.14546v1", "2608.14543v1",
        ]
        timeout_context = mock.MagicMock()
        with mock.patch(
            "tools.arxiv_tool._network_timeout", return_value=timeout_context
        ) as network_timeout, mock.patch("arxiv.Client") as client_cls, mock.patch(
            "tools.arxiv_tool._paper_info",
            side_effect=[{"id": paper_id, "title": paper_id} for paper_id in wanted],
        ):
            client_cls.return_value.results.return_value = [object() for _ in wanted]
            build_snapshot._pin_reference_papers(env)

        network_timeout.assert_called_once_with()
        timeout_context.__enter__.assert_called_once()
        timeout_context.__exit__.assert_called_once()
        self.assertEqual(
            [paper["id"] for paper in env.snapshot["get_recently_submitted_cs_papers"]["ai"]["result"][:3]],
            wanted[:3],
        )
        self.assertEqual(
            [paper["id"] for paper in env.snapshot["get_recently_submitted_cs_papers"]["cv"]["result"][:3]],
            [wanted[3], wanted[4], wanted[0]],
        )


class SnapshotPathTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        self.expected = REPO_ROOT / "data" / "mock_arxiv_snapshot.json"

    def test_defaults_are_independent_of_working_directory(self):
        for cwd in (REPO_ROOT, PACKAGE_ROOT, Path(self.tmpdir.name)):
            with self.subTest(cwd=cwd), \
                 mock.patch.object(build_snapshot, "MockArxivEnv") as env_cls, \
                 redirect_stdout(io.StringIO()):
                os.chdir(cwd)
                build_snapshot.build(pin_references=False)
                self.assertEqual(
                    env_cls.call_args.kwargs["snapshot_path"].resolve(),
                    self.expected,
                )
                with mock.patch.object(sys, "argv", ["build_snapshot", "--no-pin-references"]):
                    build_snapshot.main()
                self.assertEqual(
                    env_cls.call_args.kwargs["snapshot_path"].resolve(),
                    self.expected,
                )

    def test_cli_explicit_paths_are_preserved(self):
        os.chdir(self.tmpdir.name)
        for path in ("custom/snapshot.json", str(Path.cwd() / "absolute.json")):
            with self.subTest(path=path), \
                 mock.patch.object(sys, "argv", ["build_snapshot", "--snapshot", path]), \
                 mock.patch.object(build_snapshot, "build") as build:
                build_snapshot.main()
                self.assertEqual(build.call_args.kwargs["snapshot_path"], path)

    def test_translated_pages_are_recorded_by_default(self):
        """读译文任务引用每池第 1 篇；默认快照不录它们，这些任务就无法离线回放。"""
        with mock.patch.object(sys, "argv", ["build_snapshot"]), \
             mock.patch.object(build_snapshot, "build") as build:
            build_snapshot.main()
        self.assertEqual(build.call_args.kwargs["translate_max_ref"], 1)


    def test_pool_ref_parser(self):
        self.assertEqual(
            build_snapshot._parse_pool_refs(["AI=2", "CV=5", "LG=10"]),
            {"AI": 2, "CV": 5, "LG": 10},
        )
        self.assertIsNone(build_snapshot._parse_pool_refs(None))
        for invalid in (["AI"], ["AI=0"], ["AI=-1"], ["AI=x"], ["AI=1", "AI=2"]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_snapshot._parse_pool_refs(invalid)

    def test_targeted_pool_selection_only_keeps_requested_prefixes(self):
        env = SimpleNamespace(snapshot={
            "get_recently_submitted_cs_papers": {
                "ai": {
                    "args": {"aspect": "AI"},
                    "result": [
                        {"id": "ai-1", "title": "AI 1"},
                        {"id": "ai-2", "title": "AI 2"},
                        {"id": "ai-3", "title": "AI 3"},
                    ],
                },
                "cv": {
                    "args": {"aspect": "CV"},
                    "result": [
                        {"id": "cv-1", "title": "CV 1"},
                        {"id": "cv-2", "title": "CV 2"},
                    ],
                },
                "keyword": {
                    "args": {"query": "large language model"},
                    "result": [{"id": "kw-1", "title": "Keyword result"}],
                },
            }
        })
        selected = build_snapshot._unique_snapshot_papers_for_pools(
            env, {"AI": 2, "CV": 1}
        )
        self.assertEqual(list(selected), ["ai-1", "ai-2", "cv-1"])

    def test_targeted_pool_selection_rejects_insufficient_results(self):
        env = SimpleNamespace(snapshot={
            "get_recently_submitted_cs_papers": {
                "ai": {
                    "args": {"aspect": "AI"},
                    "result": [{"id": "ai-1", "title": "AI 1"}],
                }
            }
        })
        with self.assertRaisesRegex(RuntimeError, "requested_refs=2"):
            build_snapshot._unique_snapshot_papers_for_pools(env, {"AI": 2})

    def test_cli_passes_targeted_pool_ref_maps(self):
        argv = [
            "build_snapshot",
            "--content-pool-refs", "AI=4", "CV=5", "CL=3",
            "--translate-pool-refs", "AI=2", "LG=10",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            build_snapshot, "build"
        ) as build:
            build_snapshot.main()
        self.assertEqual(
            build.call_args.kwargs["content_pool_refs"],
            {"AI": 4, "CV": 5, "CL": 3},
        )
        self.assertEqual(
            build.call_args.kwargs["translate_pool_refs"],
            {"AI": 2, "LG": 10},
        )


if __name__ == "__main__":
    unittest.main()
