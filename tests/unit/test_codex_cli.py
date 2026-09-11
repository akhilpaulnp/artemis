from pathlib import Path
import subprocess

from langchain_core.messages import HumanMessage

from artemis.llm.codex_cli import ChatCodexCLI


def test_codex_cli_adapter_runs_ephemeral_read_only_and_parses_actions(monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(
            '{"content":"Tap Settings.","tool_calls":[{"name":"tap","arguments_json":"{\\"x\\":1}"}]}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("artemis.llm.codex_cli.subprocess.run", fake_run)
    model = ChatCodexCLI(model_name="default", timeout_seconds=12).bind_tools(
        [{"type": "function", "function": {"name": "tap", "parameters": {"type": "object"}}}]
    )

    response = model.invoke([HumanMessage(content="Tap Settings")])

    assert response.content == "Tap Settings."
    assert response.tool_calls[0]["name"] == "tap"
    assert response.tool_calls[0]["args"] == {"x": 1}
    command = commands[0]
    assert "--ignore-user-config" in command
    assert "--ephemeral" in command
    assert "--ignore-rules" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--model" not in command
    assert command[-2] == "--"
