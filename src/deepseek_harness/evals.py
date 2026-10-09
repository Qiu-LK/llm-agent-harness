from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import Settings
from .runtime import build_runtime


@dataclass(slots=True)
class EvalResult:
    case_id: str
    category: str
    passed: bool
    failures: list[str]
    answer: str = ""
    trace: str = ""
    duration_ms: float = 0
    steps: int = 0
    total_tokens: int = 0


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid case JSON on line {line_number}") from exc
            if not case.get("id") or not case.get("prompt"):
                raise ValueError(f"Case on line {line_number} needs id and prompt")
            cases.append(case)
    return cases


def evaluate_answer(case: dict[str, Any], answer: str, events: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    tools = [str(e.get("tool")) for e in events if e.get("event") == "tool_started"]
    assertions = case.get("assertions") or {}
    contains_all = list(case.get("expected_contains", [])) + list(assertions.get("contains_all", []))
    for expected in contains_all:
        if str(expected).casefold() not in answer.casefold():
            failures.append(f"answer does not contain {expected!r}")
    for alternatives in assertions.get("contains_any", []):
        options = alternatives if isinstance(alternatives, list) else [alternatives]
        if not any(str(option).casefold() in answer.casefold() for option in options):
            failures.append(f"answer contains none of {options!r}")
    for pattern in assertions.get("matches_regex", []):
        try:
            matched = re.search(str(pattern), answer, re.IGNORECASE | re.MULTILINE)
        except re.error as exc:
            failures.append(f"invalid regex {pattern!r}: {exc}")
            continue
        if not matched:
            failures.append(f"answer does not match regex {pattern!r}")
    for forbidden in list(case.get("forbidden_contains", [])) + list(assertions.get("forbidden_contains", [])):
        if str(forbidden).casefold() in answer.casefold():
            failures.append(f"answer contains forbidden text {forbidden!r}")
    for tool in list(case.get("expected_tools", [])) + list(assertions.get("tools_used", [])):
        if tool not in tools:
            failures.append(f"tool {tool!r} was not used")
    if case.get("no_tool_errors") or assertions.get("no_tool_errors"):
        if any(str(e.get("result", "")).startswith("ERROR:") for e in events if e.get("event") == "tool_finished"):
            failures.append("one or more tools returned errors")
    expected_order = assertions.get("tool_order", [])
    positions = [tools.index(name) for name in expected_order if name in tools]
    if len(positions) != len(expected_order) or positions != sorted(positions):
        failures.append(f"tool order did not match {expected_order!r}")
    return failures


async def run_case(case: dict[str, Any], settings: Settings, base_dir: Path) -> EvalResult:
    case_id = str(case["id"])
    workspace_value = case.get("workspace")
    workspace = (base_dir / workspace_value).resolve() if workspace_value else None
    events: list[dict[str, Any]] = []

    class Collector:
        def record(self, event: str, **data: Any) -> None:
            events.append({"event": event, **data})

    safe_run_id = "eval-" + hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16]
    runtime = build_runtime(settings, workspace=workspace, extra_tracer=Collector(), run_id=safe_run_id)
    started = time.monotonic()
    failures: list[str] = []
    answer = ""
    try:
        answer = await runtime.harness.run(str(case["prompt"]))
        failures.extend(evaluate_answer(case, answer, events))
    except Exception as exc:
        failures.append(f"run failed: {exc}")
    finally:
        await runtime.adapter.aclose()
    finished = next((e for e in reversed(events) if e.get("event") == "run_finished"), {})
    steps = max((int(e.get("step", 0)) for e in events), default=0)
    max_steps = case.get("max_steps", assertions.get("max_steps"))
    if max_steps is not None and steps > int(max_steps):
        failures.append(f"used {steps} steps; limit is {max_steps}")
    max_tokens = assertions.get("max_tokens")
    if max_tokens is not None and int(finished.get("total_tokens", 0)) > int(max_tokens):
        failures.append(f"used {finished.get('total_tokens', 0)} tokens; limit is {max_tokens}")
    max_duration = assertions.get("max_duration_ms")
    duration_ms = (time.monotonic() - started) * 1000
    if max_duration is not None and duration_ms > float(max_duration):
        failures.append(f"used {duration_ms:.0f} ms; limit is {max_duration} ms")
    return EvalResult(
        case_id=case_id, category=str(case.get("category", "未分类")),
        passed=not failures, failures=failures, answer=answer,
        trace=str(runtime.trace_path), duration_ms=round((time.monotonic() - started) * 1000, 2),
        steps=steps, total_tokens=int(finished.get("total_tokens", 0)),
    )


async def run_suite(cases_path: Path, settings: Settings, output_dir: Path) -> list[EvalResult]:
    from .reporting import write_eval_report

    output_dir.mkdir(parents=True, exist_ok=True)
    suite_settings = replace(settings, trace_dir=output_dir / "traces")
    results = []
    for case in load_cases(cases_path):
        results.append(await run_case(case, suite_settings, cases_path.parent))
    (output_dir / "results.json").write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    write_eval_report(results, output_dir / "report.html")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an end-to-end Harness evaluation suite")
    parser.add_argument("cases", type=Path)
    parser.add_argument("--output", type=Path, default=Path("eval-results/latest"))
    args = parser.parse_args()
    load_dotenv()
    try:
        settings = Settings.from_env()
        results = asyncio.run(run_suite(args.cases, settings, args.output))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    passed = sum(result.passed for result in results)
    for result in results:
        print(("PASS" if result.passed else "FAIL"), result.case_id, "; ".join(result.failures))
    print(f"\n{passed}/{len(results)} passed; report: {(args.output / 'report.html').resolve()}")
    raise SystemExit(0 if passed == len(results) else 1)
