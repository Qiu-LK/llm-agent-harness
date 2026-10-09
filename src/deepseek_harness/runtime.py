from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import AgentHarness, HarnessConfig
from .config import Settings
from .editing import ProposalStore, editable_registry
from .models import CompatibleChatAdapter, create_model_adapter
from .tools import ToolRegistry, WorkspaceTools
from .tracing import JsonlTracer, MultiTracer


@dataclass(slots=True)
class Runtime:
    run_id: str
    harness: AgentHarness
    adapter: CompatibleChatAdapter
    trace_path: Path
    proposal_store: ProposalStore | None = None


def build_runtime(
    settings: Settings,
    *,
    workspace: Path | None = None,
    allow_edit_proposals: bool = False,
    extra_tracer: Any | None = None,
    run_id: str | None = None,
) -> Runtime:
    actual_run_id = run_id or uuid.uuid4().hex
    trace_path = settings.trace_dir / f"{actual_run_id}.jsonl"
    file_tracer = JsonlTracer(trace_path, run_id=actual_run_id)
    tracer = MultiTracer(file_tracer, extra_tracer) if extra_tracer else file_tracer
    proposal_store = None
    if workspace and allow_edit_proposals:
        tools, proposal_store = editable_registry(
            workspace, settings.proposal_dir, run_id=actual_run_id
        )
    elif workspace:
        tools = WorkspaceTools(workspace).registry()
    else:
        tools = ToolRegistry()
    adapter = create_model_adapter(
        provider=settings.provider, api_key=settings.api_key, model=settings.model,
        base_url=settings.base_url, timeout_seconds=settings.request_timeout_seconds,
        max_retries=settings.max_retries,
    )
    harness = AgentHarness(
        model=adapter, tools=tools, tracer=tracer,
        config=HarnessConfig(
            max_steps=settings.max_steps,
            task_timeout_seconds=settings.task_timeout_seconds,
            max_repeated_tool_calls=settings.max_repeated_tool_calls,
            max_total_tokens=settings.max_total_tokens,
            max_model_requests=settings.max_model_requests,
        ),
    )
    tracer.record("run_manifest", provider=settings.provider, model=settings.model, workspace=str(workspace.resolve()) if workspace else None, editable=allow_edit_proposals)
    return Runtime(actual_run_id, harness, adapter, trace_path, proposal_store)
