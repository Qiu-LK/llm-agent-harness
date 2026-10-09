from __future__ import annotations

from unittest import IsolatedAsyncioTestCase

import httpx

from deepseek_harness.models import AuthenticationError, DeepSeekAdapter, RateLimitError


class ModelAdapterTests(IsolatedAsyncioTestCase):
    async def test_compatible_adapter_preserves_base_path(self) -> None:
        seen: list[str] = []
        async def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]})
        client = httpx.AsyncClient(base_url="https://example.test/v1/", transport=httpx.MockTransport(handler))
        adapter = DeepSeekAdapter(api_key="test", client=client)
        await adapter.generate([{"role": "user", "content": "test"}])
        await client.aclose()
        self.assertEqual(seen, ["https://example.test/v1/chat/completions"])

    async def test_parses_tool_call(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "list_files", "arguments": "{\"path\":\".\"}"}}]}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

        client = httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(handler))
        adapter = DeepSeekAdapter(api_key="test", client=client)
        response = await adapter.generate([{"role": "user", "content": "test"}], [])
        await client.aclose()
        self.assertEqual(response.tool_calls[0].name, "list_files")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "."})
        self.assertEqual(response.input_tokens, 10)

    async def test_authentication_error_is_classified(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(401, text="bad key"))
        client = httpx.AsyncClient(base_url="https://example.test", transport=transport)
        adapter = DeepSeekAdapter(api_key="test", client=client, max_retries=0)
        with self.assertRaises(AuthenticationError):
            await adapter.generate([{"role": "user", "content": "test"}])
        await client.aclose()

    async def test_rate_limit_error_is_classified(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(429, text="slow down"))
        client = httpx.AsyncClient(base_url="https://example.test", transport=transport)
        adapter = DeepSeekAdapter(api_key="test", client=client, max_retries=0)
        with self.assertRaises(RateLimitError):
            await adapter.generate([{"role": "user", "content": "test"}])
        await client.aclose()
