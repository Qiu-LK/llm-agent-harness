from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import uuid
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .tools import Tool, ToolExecutionError, ToolRegistry, WorkspaceTools


class ProposeEditInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Existing UTF-8 file relative to the workspace")
    new_content: str = Field(description="Complete proposed replacement content")
    reason: str = Field(min_length=1, max_length=1000, description="请用简体中文说明修改原因")


@dataclass(slots=True)
class ProposalStore:
    root: Path
    workspace: Path
    run_id: str | None = None

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace = self.workspace.resolve(strict=True)

    def _resolve_workspace_file(self, relative_path: str) -> Path:
        target = (self.workspace / relative_path).resolve(strict=False)
        try:
            target.relative_to(self.workspace)
        except ValueError as exc:
            raise ToolExecutionError("path escapes the workspace root") from exc
        if not target.is_file():
            raise ToolExecutionError("only existing files can be edited")
        return target

    def _proposal_path(self, proposal_id: str) -> Path:
        if not proposal_id.isalnum():
            raise ValueError("invalid proposal id")
        return self.root / f"{proposal_id}.json"

    def create(self, *, path: str, new_content: str, reason: str) -> dict:
        target = self._resolve_workspace_file(path)
        original = target.read_text(encoding="utf-8")
        if original == new_content:
            raise ToolExecutionError("proposed content is identical to the current file")
        relative = target.relative_to(self.workspace).as_posix()
        # A model may emit several incremental edits for one file in one turn.
        # Keep only the newest pending version so approvals do not conflict.
        for previous in self.list():
            if previous.get("status") == "pending" and previous.get("path") == relative:
                previous["status"] = "superseded"
                previous["superseded_at"] = datetime.now(timezone.utc).isoformat()
                self._proposal_path(previous["id"]).write_text(
                    json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        proposal_id = uuid.uuid4().hex
        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True), new_content.splitlines(keepends=True),
            fromfile=f"a/{relative}", tofile=f"b/{relative}",
        ))
        record = {
            "id": proposal_id,
            "path": relative,
            "reason": reason,
            "diff": diff,
            "new_content": new_content,
            "original_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
            "status": "pending",
            "workspace": str(self.workspace),
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._proposal_path(proposal_id).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    def get(self, proposal_id: str) -> dict:
        record = json.loads(self._proposal_path(proposal_id).read_text(encoding="utf-8"))
        if Path(record.get("workspace", "")).resolve() != self.workspace:
            raise ValueError("proposal belongs to a different workspace")
        return record

    def list(self) -> list[dict]:
        records = self._list_all()
        records.sort(key=lambda item: item.get("created_at", ""), reverse=True)

        # Migrate older proposal data lazily: only the newest pending proposal
        # for a file remains actionable and visible.
        newest_pending_paths: set[str] = set()
        for record in records:
            path = str(record.get("path", ""))
            if record.get("status") != "pending":
                continue
            if path in newest_pending_paths:
                record["status"] = "superseded"
                record["superseded_at"] = datetime.now(timezone.utc).isoformat()
                self._proposal_path(record["id"]).write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            else:
                newest_pending_paths.add(path)
        latest_by_path: dict[str, dict] = {}
        for record in records:
            if record.get("status") == "superseded":
                continue
            path = str(record.get("path", ""))
            if path not in latest_by_path:
                latest_by_path[path] = record
        return list(latest_by_path.values())

    def list_all(self) -> list[dict]:
        return sorted(self._list_all(), key=lambda item: item.get("created_at", ""), reverse=True)

    def _list_all(self) -> list[dict]:
        records = []
        for path in self.root.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if Path(record.get("workspace", "")).resolve() == self.workspace:
                    records.append(record)
            except (OSError, json.JSONDecodeError):
                continue
        return records

    def approve(self, proposal_id: str) -> dict:
        record = self.get(proposal_id)
        if record["status"] != "pending":
            raise ValueError("这条修改建议已经处理，不能重复应用")
        target = self._resolve_workspace_file(record["path"])
        current = target.read_text(encoding="utf-8")
        current_hash = hashlib.sha256(current.encode("utf-8")).hexdigest()
        if current_hash != record["original_sha256"]:
            raise ValueError("生成建议后文件已发生变化。为避免覆盖新内容，请让模型基于当前文件重新生成修改建议")
        self._proposal_path(proposal_id).with_suffix(".bak").write_text(
            current, encoding="utf-8"
        )
        target.write_text(record["new_content"], encoding="utf-8")
        record["status"] = "applied"
        record["applied_at"] = datetime.now(timezone.utc).isoformat()
        self._proposal_path(proposal_id).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        self._supersede_other_pending(record)
        return record

    def _supersede_other_pending(self, approved: dict) -> None:
        for path in self.root.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                record.get("id") != approved.get("id")
                and record.get("workspace") == approved.get("workspace")
                and record.get("path") == approved.get("path")
                and record.get("status") == "pending"
            ):
                record["status"] = "superseded"
                record["superseded_at"] = datetime.now(timezone.utc).isoformat()
                path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )

    def reject(self, proposal_id: str) -> dict:
        record = self.get(proposal_id)
        if record["status"] != "pending":
            raise ValueError("这条修改建议已经处理，不能重复拒绝")
        record["status"] = "rejected"
        self._proposal_path(proposal_id).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    def approve_and_validate(self, proposal_id: str) -> dict:
        record = self.get(proposal_id)
        if record["status"] != "pending":
            raise ValueError("这条修改建议已经处理，不能重复应用")
        config_path = self.workspace / ".harness.json"
        commands: list[list[str]] = []
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                commands = config.get("allowed_commands", [])
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"无法读取 .harness.json: {exc}") from exc
            if not isinstance(commands, list):
                raise ValueError("allowed_commands 必须是字符串数组的数组")
            for command in commands:
                if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
                    raise ValueError("allowed_commands 必须是字符串数组的数组")
        # Validate all commands before changing the target file.
        target = self._resolve_workspace_file(record["path"])
        current = target.read_text(encoding="utf-8")
        if hashlib.sha256(current.encode("utf-8")).hexdigest() != record["original_sha256"]:
            raise ValueError("生成建议后文件已发生变化。请基于当前文件重新生成修改建议")
        backup_path = self._proposal_path(proposal_id).with_suffix(".bak")
        backup_path.write_text(current, encoding="utf-8")
        target.write_text(record["new_content"], encoding="utf-8")
        record["status"] = "applied"
        record["applied_at"] = datetime.now(timezone.utc).isoformat()
        if not commands:
            record["validation"] = "skipped_no_allowed_commands"
            return self._save(record)
        results = []
        for command in commands:
            try:
                completed = subprocess.run(command, cwd=self.workspace, capture_output=True, text=True, timeout=120, shell=False)
            except subprocess.TimeoutExpired as exc:
                results.append({"command": command, "returncode": None, "stdout": str(exc.stdout or "")[-4000:], "stderr": "测试命令超时"})
                completed = None
            output_text = f"{completed.stdout}\n{completed.stderr}" if completed is not None else ""
            no_tests_ran = "NO TESTS RAN" in output_text.upper()
            if completed is not None:
                results.append({
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                    "no_tests_ran": no_tests_ran,
                })
            if completed is None or completed.returncode != 0 or no_tests_ran:
                target.write_text(current, encoding="utf-8")
                record["status"] = "rolled_back"
                record["validation"] = results
                record["validation_summary"] = "测试未通过，已自动恢复原文件"
                record["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
                return self._save(record)
        record["validation"] = results
        record["validation_summary"] = "测试通过，修改已保留"
        self._supersede_other_pending(record)
        return self._save(record)

    def approve_run_and_validate(self, run_id: str) -> dict:
        """Apply every pending proposal from one run as a single transaction."""
        if not run_id:
            raise ValueError("缺少会话 ID")
        records = [
            record for record in self.list_all()
            if record.get("run_id") == run_id and record.get("status") == "pending"
        ]
        if not records:
            raise ValueError("当前会话没有待审批的修改建议")

        # Keep the newest pending proposal for every file.
        records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        latest_by_path: dict[str, dict] = {}
        for record in records:
            latest_by_path.setdefault(str(record["path"]), record)
        records = list(latest_by_path.values())
        commands = self._load_allowed_commands()

        prepared: list[tuple[dict, Path, str]] = []
        for record in records:
            target = self._resolve_workspace_file(record["path"])
            current = target.read_text(encoding="utf-8")
            current_hash = hashlib.sha256(current.encode("utf-8")).hexdigest()
            if current_hash != record["original_sha256"]:
                raise ValueError(
                    f"文件 {record['path']} 在提案生成后发生变化，批量审批已取消"
                )
            prepared.append((record, target, current))

        batch_id = uuid.uuid4().hex
        applied_at = datetime.now(timezone.utc).isoformat()
        for record, target, current in prepared:
            self._proposal_path(record["id"]).with_suffix(".bak").write_text(
                current, encoding="utf-8"
            )
            target.write_text(record["new_content"], encoding="utf-8")
            record["batch_id"] = batch_id
            record["applied_at"] = applied_at

        results, passed = self._run_commands(commands)
        if not passed:
            rolled_back_at = datetime.now(timezone.utc).isoformat()
            for record, target, current in prepared:
                target.write_text(current, encoding="utf-8")
                record["status"] = "rolled_back"
                record["validation"] = results
                record["validation_summary"] = "批量测试未通过，所有文件已自动恢复"
                record["rolled_back_at"] = rolled_back_at
                self._save(record)
            return {
                "batch_id": batch_id,
                "status": "rolled_back",
                "proposal_count": len(records),
                "proposals": records,
                "validation": results,
            }

        for record, _, _ in prepared:
            record["status"] = "applied"
            record["validation"] = results if commands else "skipped_no_allowed_commands"
            record["validation_summary"] = (
                "批量测试通过，所有修改已保留"
                if commands else "未配置测试命令，所有修改已保留"
            )
            self._supersede_other_pending(record)
            self._save(record)
        return {
            "batch_id": batch_id,
            "status": "applied",
            "proposal_count": len(records),
            "proposals": records,
            "validation": results,
        }

    def _load_allowed_commands(self) -> list[list[str]]:
        config_path = self.workspace / ".harness.json"
        if not config_path.is_file():
            return []
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取 .harness.json: {exc}") from exc
        commands = config.get("allowed_commands", [])
        if not isinstance(commands, list):
            raise ValueError("allowed_commands 必须是字符串数组的数组")
        for command in commands:
            if (
                not isinstance(command, list)
                or not command
                or not all(isinstance(item, str) for item in command)
            ):
                raise ValueError("allowed_commands 必须是字符串数组的数组")
        return commands

    def _run_commands(self, commands: list[list[str]]) -> tuple[list[dict], bool]:
        results: list[dict] = []
        for command in commands:
            try:
                completed = subprocess.run(
                    command, cwd=self.workspace, capture_output=True,
                    text=True, timeout=120, shell=False,
                )
                output_text = f"{completed.stdout}\n{completed.stderr}"
                no_tests_ran = "NO TESTS RAN" in output_text.upper()
                results.append({
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                    "no_tests_ran": no_tests_ran,
                })
                if completed.returncode != 0 or no_tests_ran:
                    return results, False
            except subprocess.TimeoutExpired as exc:
                results.append({
                    "command": command,
                    "returncode": None,
                    "stdout": str(exc.stdout or "")[-4000:],
                    "stderr": "测试命令超时",
                    "no_tests_ran": False,
                })
                return results, False
        return results, True

    def _read_original_from_diff(self, record: dict) -> str:
        # The approval record keeps a backup next to the proposal for reliable rollback.
        backup = self._proposal_path(record["id"]).with_suffix(".bak")
        if not backup.is_file():
            raise ValueError("缺少修改前备份，无法回滚")
        return backup.read_text(encoding="utf-8")

    def _save(self, record: dict) -> dict:
        self._proposal_path(record["id"]).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record


def editable_registry(
    workspace: Path, proposal_dir: Path, *, run_id: str | None = None
) -> tuple[ToolRegistry, ProposalStore]:
    workspace_tools = WorkspaceTools(workspace)
    registry = workspace_tools.registry()
    store = ProposalStore(proposal_dir, workspace_tools.root, run_id=run_id)

    async def propose(raw: BaseModel) -> str:
        args = ProposeEditInput.model_validate(raw.model_dump())
        record = await asyncio.to_thread(store.create, path=args.path, new_content=args.new_content, reason=args.reason)
        return f"已创建修改提案 {record['id']}（文件：{record['path']}）。需要你明确批准后才会写入。\n{record['diff']}"

    registry.register(Tool(
        name="propose_edit",
        description="提出替换已有文件的修改建议。只生成 diff，必须经过用户批准才会写入；请用简体中文说明修改原因。",
        input_model=ProposeEditInput,
        handler=propose,
    ))
    return registry, store
