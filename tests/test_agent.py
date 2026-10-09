from __future__ import annotations

from collections import deque
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from deepseek_harness.agent import AgentHarness, HarnessConfig
from deepseek_harness.models import ModelResponse, ToolCall
from deepseek_harness.tools import ToolRegistry, WorkspaceTools
from deepseek_harness.tracing import NullTracer


class FakeModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = deque(responses)
        self.seen_messages: list[list[dict]] = []

    async def generate(self, messages: list[dict], tools=None) -> ModelResponse:
        self.seen_messages.append(list(messages))
        return self.responses.popleft()


class SlowModel:
    async def generate(self, messages: list[dict], tools=None) -> ModelResponse:
        await asyncio.sleep(0.1)
        return ModelResponse(content="too late")


class AgentHarnessTests(IsolatedAsyncioTestCase):
    async def test_returns_direct_answer(self) -> None:
        model = FakeModel([ModelResponse(content="完成")])
        harness = AgentHarness(
            model=model,
            tools=ToolRegistry(),
            tracer=NullTracer(),
        )

        self.assertEqual(await harness.run("测试"), "完成")

    async def test_tool_result_is_returned_to_model(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "hello.txt").write_text("第一行\n第二行", encoding="utf-8")
            model = FakeModel(
                [
                    ModelResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="call-1",
                                name="read_file",
                                arguments={"path": "hello.txt", "max_lines": 10},
                            )
                        ],
                    ),
                    ModelResponse(content="文件有两行"),
                ]
            )
            harness = AgentHarness(
                model=model,
                tools=WorkspaceTools(workspace).registry(),
                tracer=NullTracer(),
            )

            self.assertEqual(await harness.run("读取文件"), "文件有两行")
            second_request = model.seen_messages[1]
            self.assertEqual(second_request[-1]["role"], "tool")
            self.assertIn("1: 第一行", second_request[-1]["content"])

    async def test_workspace_cannot_escape(self) -> None:
        with TemporaryDirectory() as directory:
            registry = WorkspaceTools(Path(directory)).registry()
            result = await registry.execute("read_file", {"path": "../secret.txt"})
            self.assertTrue(result.startswith("ERROR:"))
            self.assertIn("escapes", result)

    async def test_lists_workspace_files(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "main.py").write_text("print('ok')", encoding="utf-8")
            (workspace / "notes.txt").write_text("note", encoding="utf-8")
            registry = WorkspaceTools(workspace).registry()
            result = await registry.execute("list_files", {"pattern": "*.py"})
            self.assertIn("main.py", result)
            self.assertNotIn("notes.txt", result)

    async def test_max_steps_stops_loop(self) -> None:
        model = FakeModel(
            [
                ModelResponse(
                    content=None,
                    tool_calls=[ToolCall(id="1", name="missing", arguments={})],
                )
            ]
        )
        harness = AgentHarness(
            model=model,
            tools=ToolRegistry(),
            tracer=NullTracer(),
            config=HarnessConfig(max_steps=1),
        )

        with self.assertRaisesRegex(RuntimeError, "maximum"):
            await harness.run("循环")

    async def test_repeated_tool_call_is_stopped(self) -> None:
        repeated = ModelResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="missing", arguments={"x": 1})],
        )
        model = FakeModel([repeated, repeated])
        harness = AgentHarness(
            model=model,
            tools=ToolRegistry(),
            tracer=NullTracer(),
            config=HarnessConfig(max_steps=3, max_repeated_tool_calls=1),
        )
        with self.assertRaisesRegex(RuntimeError, "repeated"):
            await harness.run("循环")

    async def test_task_timeout(self) -> None:
        harness = AgentHarness(
            model=SlowModel(),
            tools=ToolRegistry(),
            tracer=NullTracer(),
            config=HarnessConfig(task_timeout_seconds=0.01),
        )
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            await harness.run("慢任务")
