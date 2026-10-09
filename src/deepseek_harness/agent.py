from __future__ import annotations

import json
import asyncio
import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .models import Message, ModelAdapter
from .tools import ToolRegistry


class Tracer(Protocol):
    def record(self, event: str, **data: Any) -> None: ...


@dataclass(slots=True)
class HarnessConfig:
    max_steps: int = 8
    task_timeout_seconds: float = 300.0
    max_repeated_tool_calls: int = 2
    max_total_tokens: int = 30_000
    max_model_requests: int = 20
    system_prompt: str = (
        "你是一个谨慎、可观察的代码项目助手。"
        "请始终使用简体中文回答用户，包括分析、解释、过程性说明、修改建议和提案原因。"
        "代码、文件名、路径、变量名、函数名和必要的英文技术术语可以保留原样。"
        "需要时使用工具；只有工具确实返回过文件内容，才能声称你读过该文件。"
        "不要编造工具结果。"
    )


class AgentHarness:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        tracer: Tracer,
        config: HarnessConfig | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.tracer = tracer
        self.config = config or HarnessConfig()

    async def run(self, user_prompt: str) -> str:
        try:
            async with asyncio.timeout(self.config.task_timeout_seconds):
                answer, _ = await self._run_loop(user_prompt)
                return answer
        except TimeoutError as exc:
            self.tracer.record(
                "run_failed",
                reason="task_timeout",
                timeout_seconds=self.config.task_timeout_seconds,
            )
            raise RuntimeError(
                f"Agent exceeded the task timeout of {self.config.task_timeout_seconds:g} seconds"
            ) from exc

    async def run_turn(self, messages: list[Message], user_prompt: str) -> tuple[str, list[Message]]:
        try:
            async with asyncio.timeout(self.config.task_timeout_seconds):
                return await self._run_loop(user_prompt, messages)
        except TimeoutError as exc:
            self.tracer.record(
                "run_failed",
                reason="task_timeout",
                timeout_seconds=self.config.task_timeout_seconds,
            )
            raise RuntimeError(
                f"Agent exceeded the task timeout of {self.config.task_timeout_seconds:g} seconds"
            ) from exc

    async def _run_loop(self, user_prompt: str, messages: list[Message] | None = None) -> tuple[str, list[Message]]:
        messages = list(messages or [
            {"role": "system", "content": self.config.system_prompt},
        ])
        messages.append({"role": "user", "content": user_prompt})
        self.tracer.record("run_started", prompt=user_prompt)
        repeated_calls: dict[str, int] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        model_requests = 0

        for step in range(1, self.config.max_steps + 1):
            model_requests += 1
            if model_requests > self.config.max_model_requests:
                self.tracer.record("run_failed", reason="model_request_budget_exceeded")
                raise RuntimeError("超出模型请求次数预算")
            self.tracer.record("model_request", step=step, message_count=len(messages))
            response = await self.model.generate(messages, self.tools.schemas())
            self.tracer.record(
                "model_response",
                step=step,
                content=response.content,
                tool_calls=[asdict(call) for call in response.tool_calls],
                finish_reason=response.finish_reason,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            total_input_tokens += response.input_tokens or 0
            total_output_tokens += response.output_tokens or 0
            if total_input_tokens + total_output_tokens > self.config.max_total_tokens:
                self.tracer.record(
                    "run_failed", reason="token_budget_exceeded",
                    total_tokens=total_input_tokens + total_output_tokens,
                )
                raise RuntimeError("超出任务 Token 预算")

            assistant_message: Message = {
                "role": "assistant",
                "content": response.content,
            }
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in response.tool_calls
                ]
            messages.append(assistant_message)

            if not response.tool_calls:
                answer = response.content or ""
                self.tracer.record(
                    "run_finished",
                    step=step,
                    answer=answer,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    total_tokens=total_input_tokens + total_output_tokens,
                )
                return answer, messages

            for call in response.tool_calls:
                signature_source = json.dumps(
                    [call.name, call.arguments], ensure_ascii=False, sort_keys=True
                )
                signature = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()
                repeated_calls[signature] = repeated_calls.get(signature, 0) + 1
                if repeated_calls[signature] > self.config.max_repeated_tool_calls:
                    self.tracer.record(
                        "run_failed",
                        reason="repeated_tool_call",
                        tool=call.name,
                        arguments=call.arguments,
                    )
                    raise RuntimeError(
                        f"Agent repeated the same {call.name!r} tool call too many times"
                    )
                self.tracer.record(
                    "tool_started", step=step, tool=call.name, arguments=call.arguments
                )
                result = await self.tools.execute(call.name, call.arguments)
                self.tracer.record(
                    "tool_finished", step=step, tool=call.name, result=result
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

        self.tracer.record("run_failed", reason="max_steps_exceeded")
        raise RuntimeError(f"Agent exceeded the maximum of {self.config.max_steps} steps")
