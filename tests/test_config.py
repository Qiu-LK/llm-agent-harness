from __future__ import annotations

import os
from unittest import TestCase, mock

from deepseek_harness.config import Settings


class ConfigTests(TestCase):
    def test_ollama_does_not_require_api_key(self) -> None:
        with mock.patch.dict(os.environ, {"MODEL_PROVIDER": "ollama", "OLLAMA_MODEL": "test"}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.provider, "ollama")
        self.assertEqual(settings.model, "test")

    def test_compatible_requires_generic_key(self) -> None:
        with mock.patch.dict(os.environ, {"MODEL_PROVIDER": "compatible"}, clear=True):
            with self.assertRaisesRegex(ValueError, "MODEL_API_KEY"):
                Settings.from_env()
