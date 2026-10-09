from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ToolExecutionError(RuntimeError):
    pass


class ReadFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="相对于工作区根目录的文件路径")
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(default=200, ge=1, le=500)


class SearchTextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=300)
    path: str = Field(default=".", description="相对于工作区根目录的目录路径")
    max_results: int = Field(default=50, ge=1, le=200)


class ListFilesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(default=".", description="相对于工作区根目录的目录路径")
    pattern: str = Field(default="*", description="文件匹配模式，例如 *.py")
    max_results: int = Field(default=200, ge=1, le=1000)


ToolHandler = Callable[[BaseModel], Awaitable[str]]


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self, *, max_output_chars: int = 20_000) -> None:
        self._tools: dict[str, Tool] = {}
        self.max_output_chars = max_output_chars

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool {name!r}"
        try:
            validated = tool.input_model.model_validate(arguments)
            output = await tool.handler(validated)
        except (ValidationError, ToolExecutionError, OSError, UnicodeError) as exc:
            return f"ERROR: {exc}"
        if len(output) > self.max_output_chars:
            return output[: self.max_output_chars] + "\n...[tool output truncated]"
        return output


class WorkspaceTools:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError(f"Workspace is not a directory: {self.root}")

    def _resolve(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ToolExecutionError("path escapes the workspace root") from exc
        return candidate

    async def read_file(self, raw: BaseModel) -> str:
        args = ReadFileInput.model_validate(raw.model_dump())
        path = self._resolve(args.path)
        if not path.is_file():
            raise ToolExecutionError(f"file does not exist: {args.path}")

        def read() -> str:
            lines = path.read_text(encoding="utf-8").splitlines()
            selected = lines[args.start_line - 1 : args.start_line - 1 + args.max_lines]
            return "\n".join(
                f"{number}: {line}"
                for number, line in enumerate(selected, start=args.start_line)
            )

        return await asyncio.to_thread(read)

    async def search_text(self, raw: BaseModel) -> str:
        args = SearchTextInput.model_validate(raw.model_dump())
        base = self._resolve(args.path)
        if not base.exists():
            raise ToolExecutionError(f"path does not exist: {args.path}")

        def search() -> str:
            paths = [base] if base.is_file() else base.rglob("*")
            matches: list[str] = []
            for path in paths:
                if not path.is_file() or any(part.startswith(".") for part in path.relative_to(self.root).parts):
                    continue
                try:
                    if path.stat().st_size > 1_000_000:
                        continue
                    for line_number, line in enumerate(
                        path.read_text(encoding="utf-8").splitlines(), start=1
                    ):
                        if args.query.casefold() in line.casefold():
                            relative = path.relative_to(self.root)
                            matches.append(f"{relative}:{line_number}: {line}")
                            if len(matches) >= args.max_results:
                                return "\n".join(matches)
                except (OSError, UnicodeError):
                    continue
            return "\n".join(matches) if matches else "No matches found."

        return await asyncio.to_thread(search)

    async def list_files(self, raw: BaseModel) -> str:
        args = ListFilesInput.model_validate(raw.model_dump())
        base = self._resolve(args.path)
        if not base.is_dir():
            raise ToolExecutionError(f"directory does not exist: {args.path}")

        def list_paths() -> str:
            results: list[str] = []
            for path in base.rglob(args.pattern):
                relative = path.relative_to(self.root)
                if any(part.startswith(".") for part in relative.parts):
                    continue
                suffix = "/" if path.is_dir() else ""
                results.append(relative.as_posix() + suffix)
                if len(results) >= args.max_results:
                    break
            return "\n".join(sorted(results)) if results else "No files found."

        return await asyncio.to_thread(list_paths)

    def registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="list_files",
                description="使用文件匹配模式列出工作区中的文件和目录。",
                input_model=ListFilesInput,
                handler=self.list_files,
            )
        )
        registry.register(
            Tool(
                name="read_file",
                description="读取工作区内的 UTF-8 文本文件，并返回带行号的内容。",
                input_model=ReadFileInput,
                handler=self.read_file,
            )
        )
        registry.register(
            Tool(
                name="search_text",
                description="在工作区内的 UTF-8 文本文件中搜索指定字符串。",
                input_model=SearchTextInput,
                handler=self.search_text,
            )
        )
        return registry
