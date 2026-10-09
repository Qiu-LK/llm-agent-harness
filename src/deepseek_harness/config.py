from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    provider: str
    api_key: str
    base_url: str
    model: str
    request_timeout_seconds: float
    max_retries: int
    max_steps: int
    task_timeout_seconds: float
    max_repeated_tool_calls: int
    trace_dir: Path
    proposal_dir: Path
    database_path: Path = Path("harness.db")
    max_total_tokens: int = 30_000
    max_model_requests: int = 20

    @classmethod
    def from_env(cls, *, require_api_key: bool = True) -> Settings:
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        provider = os.getenv("MODEL_PROVIDER", "deepseek").strip().lower()
        if provider not in {"deepseek", "compatible", "ollama"}:
            raise ValueError("MODEL_PROVIDER must be deepseek, compatible, or ollama")
        generic_key = os.getenv("MODEL_API_KEY", "").strip()
        if require_api_key and provider == "deepseek" and not api_key:
            raise ValueError(
                "Missing DEEPSEEK_API_KEY. Copy .env.example to .env and fill in your key."
            )
        if require_api_key and provider == "compatible" and not generic_key:
            raise ValueError("Missing MODEL_API_KEY for compatible provider")
        return cls(
            provider=provider,
            api_key=(generic_key if provider == "compatible" else api_key),
            base_url=(
                os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                if provider == "ollama"
                else os.getenv("MODEL_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            ).rstrip("/"),
            model=(
                os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
                if provider == "ollama"
                else os.getenv("MODEL_NAME") or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            ),
            request_timeout_seconds=_float_env(
                "DEEPSEEK_TIMEOUT_SECONDS", 60.0, minimum=0.1
            ),
            max_retries=_int_env("DEEPSEEK_MAX_RETRIES", 3),
            max_steps=_int_env("HARNESS_MAX_STEPS", 8, minimum=1),
            task_timeout_seconds=_float_env(
                "HARNESS_TASK_TIMEOUT_SECONDS", 300.0, minimum=0.1
            ),
            max_repeated_tool_calls=_int_env(
                "HARNESS_MAX_REPEATED_TOOL_CALLS", 2, minimum=1
            ),
            trace_dir=Path(os.getenv("HARNESS_TRACE_DIR", "runs")),
            proposal_dir=Path(os.getenv("HARNESS_PROPOSAL_DIR", "proposals")),
            database_path=Path(os.getenv("HARNESS_DATABASE_PATH", "harness.db")),
            max_total_tokens=_int_env("HARNESS_MAX_TOTAL_TOKENS", 30_000, minimum=1),
            max_model_requests=_int_env("HARNESS_MAX_MODEL_REQUESTS", 20, minimum=1),
        )
