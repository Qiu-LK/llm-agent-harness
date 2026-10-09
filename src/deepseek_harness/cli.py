from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import Settings
from .models import ModelError
from .runtime import build_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DeepSeek agent harness")
    parser.add_argument("prompt", help="Task or question for the agent")
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Optional directory exposed through read-only file tools",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--timeout", type=float, help="Per-request timeout in seconds")
    parser.add_argument("--task-timeout", type=float, help="Whole-task timeout in seconds")
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--allow-edit-proposals", action="store_true")
    return parser


async def async_main(args: argparse.Namespace) -> int:
    load_dotenv()
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    max_steps = settings.max_steps if args.max_steps is None else args.max_steps
    if max_steps < 1:
        print("--max-steps must be at least 1", file=sys.stderr)
        return 2

    if args.timeout is not None and args.timeout <= 0:
        print("--timeout must be greater than 0", file=sys.stderr)
        return 2
    if args.task_timeout is not None and args.task_timeout <= 0:
        print("--task-timeout must be greater than 0", file=sys.stderr)
        return 2

    settings = Settings(
        provider=settings.provider, api_key=settings.api_key, base_url=settings.base_url,
        model=settings.model, request_timeout_seconds=args.timeout if args.timeout is not None else settings.request_timeout_seconds,
        max_retries=settings.max_retries, max_steps=max_steps,
        task_timeout_seconds=args.task_timeout if args.task_timeout is not None else settings.task_timeout_seconds,
        max_repeated_tool_calls=settings.max_repeated_tool_calls,
        trace_dir=args.trace_dir or settings.trace_dir, proposal_dir=settings.proposal_dir,
        database_path=settings.database_path,
        max_total_tokens=settings.max_total_tokens,
        max_model_requests=settings.max_model_requests,
    )
    try:
        runtime = build_runtime(settings, workspace=args.workspace, allow_edit_proposals=args.allow_edit_proposals)
    except (OSError, ValueError) as exc:
        print(f"Invalid workspace or configuration: {exc}", file=sys.stderr)
        return 2

    try:
        answer = await runtime.harness.run(args.prompt)
    except (ModelError, RuntimeError) as exc:
        runtime.harness.tracer.record("run_error", error=str(exc))
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Trace: {runtime.trace_path.resolve()}", file=sys.stderr)
        return 1
    finally:
        await runtime.adapter.aclose()

    print(answer)
    print(f"\n[trace: {runtime.trace_path.resolve()}]", file=sys.stderr)
    return 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(async_main(args)))
