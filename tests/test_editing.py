from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import json
import sys

from deepseek_harness.editing import ProposalStore


class ProposalStoreTests(TestCase):
    def test_requires_approval_before_writing(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            source = workspace / "app.py"
            source.write_text("old\n", encoding="utf-8")
            store = ProposalStore(root / "proposals", workspace)
            proposal = store.create(path="app.py", new_content="new\n", reason="update")
            self.assertEqual(source.read_text(encoding="utf-8"), "old\n")
            self.assertIn("-old", proposal["diff"])
            store.approve(proposal["id"])
            self.assertEqual(source.read_text(encoding="utf-8"), "new\n")

    def test_refuses_to_overwrite_changed_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            source = workspace / "app.py"
            source.write_text("old", encoding="utf-8")
            store = ProposalStore(root / "proposals", workspace)
            proposal = store.create(path="app.py", new_content="new", reason="update")
            source.write_text("user change", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "文件已发生变化"):
                store.approve(proposal["id"])

    def test_only_latest_pending_proposal_for_same_file_is_kept(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            source = workspace / "app.py"
            source.write_text("original\n", encoding="utf-8")
            store = ProposalStore(root / "proposals", workspace)

            first = store.create(path="app.py", new_content="version 1\n", reason="第一次")
            second = store.create(path="app.py", new_content="version 2\n", reason="最终版本")

            visible = store.list()
            self.assertEqual([item["id"] for item in visible], [second["id"]])
            self.assertEqual(store.get(first["id"])["status"], "superseded")

            store.approve(second["id"])
            self.assertEqual(source.read_text(encoding="utf-8"), "version 2\n")
            self.assertTrue((root / "proposals" / f"{second['id']}.bak").is_file())

    def test_only_latest_record_for_each_file_is_listed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            source = workspace / "app.py"
            source.write_text("original\n", encoding="utf-8")
            store = ProposalStore(root / "proposals", workspace)

            first = store.create(path="app.py", new_content="version 1\n", reason="旧版本")
            store.approve(first["id"])
            second = store.create(path="app.py", new_content="version 2\n", reason="新版本")

            self.assertEqual([item["id"] for item in store.list()], [second["id"]])

    def test_proposal_records_originating_run(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "app.py").write_text("old\n", encoding="utf-8")
            store = ProposalStore(root / "proposals", workspace, run_id="run-123")

            proposal = store.create(path="app.py", new_content="new\n", reason="更新")

            self.assertEqual(proposal["run_id"], "run-123")

    def test_approve_and_validate_rolls_back_when_test_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            source = workspace / "app.py"
            source.write_text("old\n", encoding="utf-8")
            (workspace / ".harness.json").write_text(json.dumps({"allowed_commands": [[sys.executable, "-c", "raise SystemExit(1)"]]}), encoding="utf-8")
            store = ProposalStore(root / "proposals", workspace)
            proposal = store.create(path="app.py", new_content="broken\n", reason="测试回滚")

            result = store.approve_and_validate(proposal["id"])

            self.assertEqual(result["status"], "rolled_back")
            self.assertEqual(source.read_text(encoding="utf-8"), "old\n")

    def test_approve_and_validate_keeps_change_when_test_passes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            source = workspace / "app.py"
            source.write_text("old\n", encoding="utf-8")
            (workspace / ".harness.json").write_text(json.dumps({"allowed_commands": [[sys.executable, "-c", "pass"]]}), encoding="utf-8")
            store = ProposalStore(root / "proposals", workspace)
            proposal = store.create(path="app.py", new_content="new\n", reason="测试通过")

            result = store.approve_and_validate(proposal["id"])

            self.assertEqual(result["status"], "applied")
            self.assertEqual(source.read_text(encoding="utf-8"), "new\n")

    def test_zero_tests_is_not_considered_success(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            source = workspace / "app.py"
            source.write_text("old\n", encoding="utf-8")
            (workspace / ".harness.json").write_text(json.dumps({"allowed_commands": [[sys.executable, "-c", "print('NO TESTS RAN')"]]}), encoding="utf-8")
            store = ProposalStore(root / "proposals", workspace)
            proposal = store.create(path="app.py", new_content="new\n", reason="测试数量校验")

            result = store.approve_and_validate(proposal["id"])

            self.assertEqual(result["status"], "rolled_back")
            self.assertEqual(source.read_text(encoding="utf-8"), "old\n")

    def test_batch_approval_applies_dependent_files_before_testing(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            source = workspace / "value.txt"
            expected = workspace / "expected.txt"
            source.write_text("old", encoding="utf-8")
            expected.write_text("old", encoding="utf-8")
            command = [
                sys.executable, "-c",
                "from pathlib import Path; raise SystemExit(0 if Path('value.txt').read_text() == Path('expected.txt').read_text() == 'new' else 1)",
            ]
            (workspace / ".harness.json").write_text(
                json.dumps({"allowed_commands": [command]}), encoding="utf-8"
            )
            store = ProposalStore(root / "proposals", workspace, run_id="run-1")
            store.create(path="value.txt", new_content="new", reason="修改代码")
            store.create(path="expected.txt", new_content="new", reason="修改测试")

            result = store.approve_run_and_validate("run-1")

            self.assertEqual(result["status"], "applied")
            self.assertEqual(result["proposal_count"], 2)
            self.assertEqual(source.read_text(encoding="utf-8"), "new")
            self.assertEqual(expected.read_text(encoding="utf-8"), "new")

    def test_batch_approval_rolls_back_every_file_on_failure(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            first = workspace / "first.txt"
            second = workspace / "second.txt"
            first.write_text("one", encoding="utf-8")
            second.write_text("two", encoding="utf-8")
            (workspace / ".harness.json").write_text(
                json.dumps({"allowed_commands": [[sys.executable, "-c", "raise SystemExit(1)"]]}),
                encoding="utf-8",
            )
            store = ProposalStore(root / "proposals", workspace, run_id="run-2")
            proposals = [
                store.create(path="first.txt", new_content="changed one", reason="一"),
                store.create(path="second.txt", new_content="changed two", reason="二"),
            ]

            result = store.approve_run_and_validate("run-2")

            self.assertEqual(result["status"], "rolled_back")
            self.assertEqual(first.read_text(encoding="utf-8"), "one")
            self.assertEqual(second.read_text(encoding="utf-8"), "two")
            self.assertTrue(all(store.get(item["id"])["status"] == "rolled_back" for item in proposals))
