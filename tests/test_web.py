from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from deepseek_harness.config import Settings


@skipUnless(HAS_FASTAPI, "FastAPI dependencies are not installed in this interpreter")
class WebTests(TestCase):
    def test_home_page_loads(self) -> None:
        with TemporaryDirectory() as directory:
            settings = Settings(
                provider="ollama", api_key="", base_url="http://localhost:11434", model="test",
                request_timeout_seconds=1, max_retries=0, max_steps=2,
                task_timeout_seconds=1, max_repeated_tool_calls=1,
                trace_dir=Path(directory) / "runs", proposal_dir=Path(directory) / "proposals",
            )
            from deepseek_harness.web import create_app
            client = TestClient(create_app(settings))
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("LLM Agent Harness", response.text)
