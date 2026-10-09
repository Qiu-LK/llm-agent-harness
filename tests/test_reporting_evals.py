from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from deepseek_harness.evals import evaluate_answer, load_cases
from deepseek_harness.reporting import write_report


class ReportingAndEvalTests(TestCase):
    def test_writes_html_report(self) -> None:
        with TemporaryDirectory() as directory:
            trace = Path(directory) / "run.jsonl"
            trace.write_text(json.dumps({"event": "run_finished", "answer": "完成 <ok>"}, ensure_ascii=False) + "\n", encoding="utf-8")
            body = write_report(trace).read_text(encoding="utf-8")
            self.assertIn("run_finished", body)
            self.assertIn("&lt;ok&gt;", body)

    def test_evaluates_answer_and_tools(self) -> None:
        events = [{"event": "tool_started", "tool": "list_files", "step": 1}, {"event": "run_finished", "answer": "README.md", "step": 2}]
        failures = evaluate_answer({"id": "case", "expected_contains": ["README"], "expected_tools": ["list_files"]}, "README.md", events)
        self.assertEqual(failures, [])

    def test_cases_require_prompt(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases.jsonl"
            cases.write_text('{"id":"missing"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "id and prompt"):
                load_cases(cases)

    def test_enhanced_assertions(self) -> None:
        events = [
            {"event": "tool_started", "tool": "list_files"},
            {"event": "tool_started", "tool": "read_file"},
            {"event": "tool_finished", "tool": "read_file", "result": "ok"},
        ]
        case = {"assertions": {"contains_any": [["入口", "main.py"]], "matches_regex": [r"main\.py"], "tool_order": ["list_files", "read_file"], "max_tokens": 10}}
        self.assertEqual(evaluate_answer(case, "项目入口是 main.py", events), [])
