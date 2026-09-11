# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Adapters for subscription-authenticated coding-agent CLIs."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pydantic import Field

from artemis.llm.codex_cli import ChatCodexCLI


class ChatAgentCLI(ChatCodexCLI):
    """Use a coding-agent CLI as Artemis's structured chat backend."""

    cli_name: str = Field(description="CLI executable name")

    def _run_codex(self, prompt: str, images: list[bytes]) -> dict[str, Any]:
        properties: dict[str, Any] = {"content": {"type": "string"}}
        required = ["content"]
        if self.bound_tools:
            properties["tool_calls"] = self._tool_calls_schema()
            required.append("tool_calls")
        schema = {"type": "object", "additionalProperties": False, "required": required, "properties": properties}
        with tempfile.TemporaryDirectory(prefix="artemis-agent-cli-") as tmp:
            directory = Path(tmp)
            schema_path = directory / "schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            if self.cli_name == "claude":
                command = [
                    "claude", "--print", "--output-format", "json", "--json-schema", json.dumps(schema),
                    "--no-session-persistence", "--restricted", "--strict-mcp-config", "--tools", "", "--", prompt,
                ]
            elif self.cli_name == "prime-agent":
                command = ["prime-agent", "--print", "--mode", "json", "--no-tools", "--no-session", prompt]
            elif self.cli_name == "pi":
                command = ["pi", "--print", prompt]
            elif self.cli_name == "omp":
                command = ["omp", "--print", prompt]
            else:
                raise RuntimeError(f"Unsupported agent CLI: {self.cli_name}")
            if self.model_name and self.model_name != "default" and self.cli_name in {"pi", "omp", "prime-agent"}:
                command.extend(["--model", self.model_name])
            result = subprocess.run(command, text=True, capture_output=True, timeout=self.timeout_seconds, check=False, cwd=directory)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()[-1000:]
                raise RuntimeError(f"{self.cli_name} CLI failed ({result.returncode}): {detail}")
            try:
                payload = json.loads(result.stdout)
                structured = payload.get("structured_output") if isinstance(payload, dict) else None
                if isinstance(structured, dict):
                    return structured
                if isinstance(payload, dict) and isinstance(payload.get("result"), str):
                    return json.loads(payload["result"])
                return payload
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{self.cli_name} CLI did not return JSON output") from exc


_STATIC_MODELS = {
    "codex": ["gpt-5.1", "gpt-5.2-codex", "gpt-5.3-codex", "gpt-5.4", "gpt-5.5"],
    "claude_cli": ["opus", "sonnet", "haiku"],
    "pi_cli": [],
    "omp_cli": [],
    "prime_agent": [],
}

_CLI_MODEL_SOURCES = {"codex": "openai-codex", "claude_cli": "anthropic"}


def get_agent_cli_models(provider: str) -> list[str]:
    """List selectable models for a subscription CLI, newest catalog first."""
    models: list[str] = []
    if shutil.which("prime-agent"):
        try:
            result = subprocess.run(
                ["prime-agent", "model", "list"], text=True, capture_output=True, timeout=20, check=False
            )
            if result.returncode == 0:
                wanted = _CLI_MODEL_SOURCES.get(provider)
                # The catalog table is printed on stderr by prime-agent.
                listing = f"{result.stdout}\n{result.stderr}"
                for line in listing.splitlines():
                    if line.startswith("provider"):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    source, model = parts[0], parts[1]
                    if wanted is None or source == wanted:
                        models.append(model)
        except (OSError, subprocess.TimeoutExpired):
            models = []
    if not models:
        models = list(_STATIC_MODELS.get(provider, []))
    return ["default", *dict.fromkeys(models)]
