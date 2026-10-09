from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Settings
from .editing import ProposalStore
from .runtime import build_runtime


WEB_ROOT = Path(__file__).with_name("web_assets")


@dataclass
class RunState:
    id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    done: bool = False
    messages: list[dict[str, Any]] = field(default_factory=list)
    workspace: str | None = None
    allow_edit_proposals: bool = False
    task: asyncio.Task | None = None
    cancelled: bool = False

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        for queue in self.subscribers:
            queue.put_nowait(event)


class RunRequest(BaseModel):
    prompt: str
    workspace: str | None = None
    allow_edit_proposals: bool = False


class FollowupRequest(BaseModel):
    prompt: str


def create_app(settings: Settings | None = None) -> FastAPI:
    load_dotenv()
    config = settings or Settings.from_env()
    app = FastAPI(title="LLM Agent Harness", version="0.6.0")
    app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")
    runs: dict[str, RunState] = {}

    @app.get("/", response_class=FileResponse)
    async def index() -> Path:
        return WEB_ROOT / "templates" / "index.html"

    async def execute(state: RunState, prompt: str, *, reset: bool = False) -> None:
        class EventSink:
            def record(self, event: str, **data: Any) -> None:
                state.publish({"event": event, **data})

        runtime = None
        try:
            workspace = Path(state.workspace).expanduser() if state.workspace else None
            runtime = build_runtime(
                config,
                workspace=workspace,
                allow_edit_proposals=state.allow_edit_proposals,
                extra_tracer=EventSink(),
                run_id=state.id,
            )
            if reset or not state.messages:
                state.messages = [
                    {"role": "system", "content": runtime.harness.config.system_prompt}
                ]
            _, state.messages = await runtime.harness.run_turn(state.messages, prompt)
        except Exception as exc:
            state.publish({"event": "run_error", "error": str(exc)})
        finally:
            if runtime:
                await runtime.adapter.aclose()
            state.done = True
            state.publish({"event": "stream_end"})

    @app.post("/api/runs")
    async def start_run(request: RunRequest) -> dict[str, str]:
        if not request.prompt.strip():
            raise HTTPException(422, "任务不能为空")
        run_id = uuid.uuid4().hex
        state = RunState(
            run_id,
            workspace=request.workspace,
            allow_edit_proposals=request.allow_edit_proposals,
        )
        runs[run_id] = state
        state.task = asyncio.create_task(execute(state, request.prompt, reset=True))
        return {"run_id": run_id}

    @app.post("/api/runs/{run_id}/followup")
    async def followup(run_id: str, request: FollowupRequest) -> dict[str, str]:
        state = runs.get(run_id)
        if not state:
            raise HTTPException(404, "会话不存在")
        if not request.prompt.strip():
            raise HTTPException(422, "追问不能为空")
        state.done = False
        state.task = asyncio.create_task(execute(state, request.prompt))
        return {"run_id": run_id}

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel(run_id: str) -> dict[str, str]:
        state = runs.get(run_id)
        if not state:
            raise HTTPException(404, "会话不存在")
        if state.task and not state.task.done():
            state.cancelled = True
            state.task.cancel()
            state.publish({"event": "run_cancelled", "error": "任务已由用户停止"})
            state.done = True
            state.publish({"event": "stream_end"})
        return {"run_id": run_id, "status": "cancelled"}

    @app.get("/api/runs/{run_id}/events")
    async def stream_events(run_id: str) -> StreamingResponse:
        state = runs.get(run_id)
        if not state:
            raise HTTPException(404, "会话不存在")

        async def stream():
            queue: asyncio.Queue = asyncio.Queue()
            state.subscribers.append(queue)
            try:
                for event in state.events:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if state.done:
                    return
                while True:
                    event = await queue.get()
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event.get("event") == "stream_end":
                        break
            finally:
                if queue in state.subscribers:
                    state.subscribers.remove(queue)

        return StreamingResponse(stream(), media_type="text/event-stream")

    def proposal_store(workspace: str) -> ProposalStore:
        return ProposalStore(
            config.proposal_dir, Path(workspace).expanduser().resolve(strict=True)
        )

    @app.get("/api/proposals")
    async def proposals(workspace: str, run_id: str | None = None) -> list[dict]:
        try:
            store = proposal_store(workspace)
            records = store.list()
            if run_id:
                records = [record for record in records if record.get("run_id") == run_id]
            return records
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/proposals/history")
    async def proposal_history(workspace: str) -> list[dict]:
        try:
            return proposal_store(workspace).list_all()
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/proposals/{proposal_id}/approve")
    async def approve(proposal_id: str, workspace: str) -> dict:
        try:
            return proposal_store(workspace).approve_and_validate(proposal_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/runs/{run_id}/proposals/approve-all")
    async def approve_all(run_id: str, workspace: str) -> dict:
        try:
            return proposal_store(workspace).approve_run_and_validate(run_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/proposals/{proposal_id}/reject")
    async def reject(proposal_id: str, workspace: str) -> dict:
        try:
            return proposal_store(workspace).reject(proposal_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 Harness 演示网页")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
