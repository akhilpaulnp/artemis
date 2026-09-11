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
    if provider in _LOCAL_RUNTIMES:
        return ["default", *(_local_runtime_models(_LOCAL_RUNTIMES[provider][0]) or [])]

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


_LOCAL_RUNTIMES = {
    "ollama": ("http://127.0.0.1:11434/v1/models", "Ollama", "ollama serve"),
    "lmstudio": ("http://127.0.0.1:1234/v1/models", "LM Studio", "lms server start"),
}


def _local_runtime_models(url: str) -> list[str] | None:
    """Return model ids from an OpenAI-compatible local server, or None if it is down."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None
    return [entry["id"] for entry in payload.get("data", []) if entry.get("id")]


def get_local_runtime_statuses() -> dict[str, dict[str, str | bool]]:
    """Report whether local model runtimes are installed and serving."""
    commands = {"ollama": "ollama", "lmstudio": "lms"}
    statuses: dict[str, dict[str, str | bool]] = {}
    for provider, (url, label, start_hint) in _LOCAL_RUNTIMES.items():
        installed = shutil.which(commands[provider]) is not None
        models = _local_runtime_models(url) if installed else None
        if not installed:
            message = f"{label} is not installed."
        elif models is None:
            message = f"{label} is installed but not serving. Run `{start_hint}`."
        elif not models:
            message = f"{label} is running but has no models. Download one first."
        else:
            message = f"{label} is running with {len(models)} model(s)."
        statuses[provider] = {
            "installed": installed,
            "authenticated": bool(models),
            "message": message,
        }
    return statuses


_PROVIDER_GUIDANCE: dict[str, dict[str, str | None]] = {
    "codex": {"recommended": "gpt-5.5", "warning": None},
    "claude_cli": {"recommended": "sonnet", "warning": "Every Artemis step uses your Claude subscription quota."},
    "pi_cli": {
        "recommended": None,
        "warning": "Experimental: the Pi CLI has no structured-output flag, so tool calls can fail.",
    },
    "omp_cli": {
        "recommended": None,
        "warning": "Experimental: the OMP CLI has no structured-output flag, so tool calls can fail.",
    },
    "prime_agent": {
        "recommended": None,
        "warning": "Experimental: Prime Agent JSON mode is not schema-guaranteed, so tool calls can fail.",
    },
    "ollama": {
        "recommended": None,
        "warning": "Local models are weak at pixel grounding. Use a vision model, prefer Pro profile, and expect lower accuracy.",
    },
    "lmstudio": {
        "recommended": None,
        "warning": "Local models are weak at pixel grounding. Load a vision model and expect lower accuracy.",
    },
}


def _recommended_ollama_model() -> str | None:
    """Prefer an installed Ollama model that can see screenshots and call tools."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None
    for entry in payload.get("models", []):
        capabilities = set(entry.get("capabilities") or [])
        if {"vision", "tools"} <= capabilities:
            return entry.get("name")
    return None


def get_provider_guidance(provider: str) -> dict[str, str | None]:
    """Return the recommended model and usage warning for one provider."""
    guidance = dict(_PROVIDER_GUIDANCE.get(provider, {"recommended": None, "warning": None}))
    if provider == "ollama":
        guidance["recommended"] = _recommended_ollama_model()
    available = get_agent_cli_models(provider)
    recommended = guidance["recommended"]
    if recommended and recommended not in available:
        # CLI aliases such as "sonnet" resolve to a full catalog entry.
        guidance["recommended"] = next((m for m in available if recommended in m), None)
    return guidance
