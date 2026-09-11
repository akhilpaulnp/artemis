# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Experimental LangChain adapter for a ChatGPT-authenticated Codex CLI."""

import base64
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field


def get_codex_cli_status() -> dict[str, str | bool]:
    """Return whether the official Codex CLI is installed and logged in."""
    executable = shutil.which("codex")
    if not executable:
        return {
            "installed": False,
            "authenticated": False,
            "message": "Codex CLI is not installed. Install it, then run `codex login`.",
        }
    try:
        result = subprocess.run(
            [executable, "login", "status"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"installed": True, "authenticated": False, "message": str(exc)}

    message = (result.stdout or result.stderr).strip()
    return {
        "installed": True,
        "authenticated": result.returncode == 0,
        "message": message or "Codex CLI login status is unavailable.",
    }


def get_agent_cli_statuses() -> dict[str, dict[str, str | bool]]:
    """Return safe installation/login state for optional agent CLIs."""
    statuses = {"codex": get_codex_cli_status()}
    for provider, command in (("claude_cli", "claude"), ("pi_cli", "pi"), ("omp_cli", "omp"), ("prime_agent", "prime-agent")):
        executable = shutil.which(command)
        if not executable:
            statuses[provider] = {
                "installed": False,
                "authenticated": False,
                "message": f"{command} CLI is not installed.",
            }
            continue
        statuses[provider] = {
            "installed": True,
            "authenticated": False,
            "message": f"{command} CLI is installed.",
        }

    claude = statuses["claude_cli"]
    if claude["installed"]:
        try:
            result = subprocess.run(["claude", "auth", "status"], text=True, capture_output=True, timeout=10, check=False)
            claude["authenticated"] = result.returncode == 0
            claude["message"] = "Claude Code is logged in." if result.returncode == 0 else "Run `claude auth login` to sign in."
        except (OSError, subprocess.TimeoutExpired):
            claude["message"] = "Could not check Claude Code login."
    return statuses


class ChatCodexCLI(BaseChatModel):
    """Use the logged-in Codex CLI as a structured, read-only chat backend.

    This is deliberately an adapter, not an OAuth-token bridge. Authentication
    remains owned by the official ``codex`` executable and its ChatGPT login.
    """

    model_name: str | None = None
    timeout_seconds: float = 180.0
    bound_tools: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "codex-cli"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ChatCodexCLI":
        return self.model_copy(
            update={"bound_tools": [convert_to_openai_tool(tool) for tool in tools]}
        )

    def _message_text_and_images(self, messages: list[BaseMessage]) -> tuple[str, list[bytes]]:
        transcript: list[str] = []
        images: list[bytes] = []
        for message in messages:
            content = message.content
            if isinstance(content, str):
                rendered = content
            elif isinstance(content, list):
                text_parts: list[str] = []
                for part in content:
                    if not isinstance(part, dict):
                        text_parts.append(str(part))
                        continue
                    if part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                    elif part.get("type") in ("image", "image_url"):
                        value = part.get("image_url", part.get("image", ""))
                        url = value.get("url", "") if isinstance(value, dict) else str(value)
                        if url.startswith("data:image/") and "," in url:
                            images.append(base64.b64decode(url.split(",", 1)[1]))
                        else:
                            text_parts.append("[Image attachment unavailable to Codex CLI]")
                    else:
                        text_parts.append(json.dumps(part, default=str))
                rendered = "\n".join(text_parts)
            else:
                rendered = str(content)
            transcript.append(f"[{message.type}]\n{rendered}")
        return "\n\n".join(transcript), images

    def _prompt(self, transcript: str) -> str:
        tool_names = [tool["function"]["name"] for tool in self.bound_tools]
        return f"""You are a structured response backend for ARTEMIS mobile automation.
Do not run shell commands, modify files, browse, or use your own tools. Analyze only the supplied conversation and images.
Return exactly one JSON object matching the supplied output schema.
For each tool call, encode its argument object in the `arguments_json` string field.
Use only these Artemis tool names when an action is required: {tool_names}.

Conversation:
{transcript}"""

    def _tool_calls_schema(self) -> dict[str, Any]:
        # Codex structured output rejects oneOf inside array items. Encode each
        # tool's argument object as JSON, then decode it before returning the
        # LangChain AIMessage tool call.
        return {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "arguments_json"],
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [tool["function"]["name"] for tool in self.bound_tools],
                    },
                    "arguments_json": {"type": "string"},
                },
            },
        }

    def _run_codex(self, prompt: str, images: list[bytes]) -> dict[str, Any]:
        properties: dict[str, Any] = {"content": {"type": "string"}}
        required = ["content"]
        if self.bound_tools:
            properties["tool_calls"] = self._tool_calls_schema()
            required.append("tool_calls")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }
        with tempfile.TemporaryDirectory(prefix="artemis-codex-") as tmp:
            directory = Path(tmp)
            schema_path = directory / "schema.json"
            output_path = directory / "output.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            command = [
                "codex", "exec", "--ignore-user-config", "--ephemeral", "--ignore-rules", "--skip-git-repo-check",
                "--sandbox", "read-only", "--output-schema", str(schema_path),
                "--output-last-message", str(output_path), "--cd", str(directory),
            ]
            if self.model_name and self.model_name != "default":
                command.extend(["--model", self.model_name])
            for index, image in enumerate(images):
                image_path = directory / f"image-{index}.png"
                image_path.write_bytes(image)
                command.extend(["--image", str(image_path)])
            result = subprocess.run(
                command + ["--", prompt], text=True, capture_output=True, timeout=self.timeout_seconds,
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()[-1000:]
                raise RuntimeError(f"Codex CLI failed ({result.returncode}): {detail}")
            try:
                return json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("Codex CLI did not return schema-valid JSON") from exc

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        transcript, images = self._message_text_and_images(messages)
        payload = self._run_codex(self._prompt(transcript), images)
        calls = [
            {
                "name": call["name"],
                "args": json.loads(call["arguments_json"]),
                "id": f"codex-{uuid.uuid4().hex[:12]}",
            }
            for call in payload.get("tool_calls", [])
        ]
        message = AIMessage(content=payload["content"], tool_calls=calls)
        return ChatResult(generations=[ChatGeneration(message=message)])
