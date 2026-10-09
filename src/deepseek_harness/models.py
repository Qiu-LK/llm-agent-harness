from __future__ import annotations

import json
import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


Message = dict[str, Any]


class ModelError(RuntimeError):
    """Raised when the provider returns an unusable response."""


class AuthenticationError(ModelError):
    pass


class RateLimitError(ModelError):
    pass


class ProviderServerError(ModelError):
    pass


class RequestTimeoutError(ModelError):
    pass


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ModelResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw_message: Message = field(default_factory=dict)


class ModelAdapter(Protocol):
    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse: ...


class CompatibleChatAdapter:
    """Direct DeepSeek Chat Completions adapter using HTTP, without an SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
        require_api_key: bool = True,
        endpoint: str = "chat/completions",
    ) -> None:
        if require_api_key and not api_key:
            raise ValueError("api_key cannot be empty")
        self.model = model
        self.endpoint = endpoint
        self.max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={**({"Authorization": f"Bearer {api_key}"} if api_key else {}), "Content-Type": "application/json"},
            timeout=httpx.Timeout(timeout_seconds),
            transport=httpx.AsyncHTTPTransport(retries=0),
        )

    async def __aenter__(self) -> CompatibleChatAdapter:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(self.endpoint, json=payload)
            except httpx.TimeoutException as exc:
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                raise RequestTimeoutError("DeepSeek request timed out") from exc
            except httpx.HTTPError as exc:
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                raise ModelError(f"DeepSeek network request failed: {exc}") from exc

            if response.status_code == 429 or response.status_code >= 500:
                if attempt < self.max_retries:
                    await asyncio.sleep(self._retry_delay(response, attempt))
                    continue
                body = response.text[:1000]
                error_type = RateLimitError if response.status_code == 429 else ProviderServerError
                raise error_type(
                    f"DeepSeek API returned HTTP {response.status_code}: {body}"
                )
            if response.status_code in {401, 403}:
                raise AuthenticationError(
                    f"DeepSeek authentication failed with HTTP {response.status_code}"
                )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ModelError(
                    f"DeepSeek API returned HTTP {response.status_code}: {response.text[:1000]}"
                ) from exc
            break

        if response is None:  # pragma: no cover - defensive guard
            raise ModelError("DeepSeek request produced no response")
        try:
            data = response.json()
        except ValueError as exc:
            raise ModelError("DeepSeek returned invalid JSON") from exc

        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError("DeepSeek returned an unexpected response shape") from exc

        tool_calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = (
                    raw_arguments
                    if isinstance(raw_arguments, dict)
                    else json.loads(raw_arguments)
                )
            except json.JSONDecodeError as exc:
                raise ModelError(
                    f"Model produced invalid tool arguments for {function.get('name')!r}"
                ) from exc
            if not isinstance(arguments, dict):
                raise ModelError("Tool arguments must decode to a JSON object")
            tool_calls.append(
                ToolCall(
                    id=str(item["id"]),
                    name=str(function["name"]),
                    arguments=arguments,
                )
            )

        usage = data.get("usage") or {}
        return ModelResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            raw_message=message,
        )

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(8.0, 0.5 * (2**attempt)) + random.uniform(0.0, 0.25)

    @classmethod
    def _retry_delay(cls, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(30.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
        return cls._backoff(attempt)


class DeepSeekAdapter(CompatibleChatAdapter):
    """DeepSeek-specific defaults over the compatible Chat Completions protocol."""


class OllamaAdapter(CompatibleChatAdapter):
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        max_retries: int = 1,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key="",
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            client=client,
            require_api_key=False,
            endpoint="v1/chat/completions",
        )


def create_model_adapter(
    *, provider: str, api_key: str, model: str, base_url: str,
    timeout_seconds: float, max_retries: int,
) -> CompatibleChatAdapter:
    if provider == "ollama":
        return OllamaAdapter(model=model, base_url=base_url, timeout_seconds=timeout_seconds, max_retries=max_retries)
    adapter_type = DeepSeekAdapter if provider == "deepseek" else CompatibleChatAdapter
    return adapter_type(api_key=api_key, model=model, base_url=base_url, timeout_seconds=timeout_seconds, max_retries=max_retries)
