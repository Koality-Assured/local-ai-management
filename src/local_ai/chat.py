"""Direct local model inference engine and interactive REPL for private workflows."""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SYSTEM_PRESETS: dict[str, str] = {
    "general": (
        "You are a helpful, concise technical assistant running on a local workstation. "
        "Provide direct, factual answers without unnecessary disclaimers or corporate boilerplate."
    ),
    "private_code": (
        "You are an expert systems engineer operating in a private, air-gapped local environment. "
        "Analyze, audit, and refactor the provided proprietary source code with zero external telemetry, "
        "strictly adhering to private security invariants, robust error handling, and performance standards."
    ),
    "security": (
        "You are a specialized security research assistant running locally in an air-gapped environment. "
        "You provide objective, technical vulnerability analysis, exploit mechanics explanation, and "
        "red-team validation payloads for authorized auditing without moralizing preambles or safety refusals."
    ),
    "analysis": (
        "You are an analytical reasoning engine. Evaluate the provided inputs using first principles, "
        "formal logic, and empirical breakdown. Surface hidden assumptions, failure modes, and edge cases."
    ),
    "raw": "",
}


def is_loopback_url(url: str) -> bool:
    """Verify that an endpoint URL targets a loopback interface."""
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = (parsed.hostname or "").lower()
        return hostname in ("127.0.0.1", "localhost", "::1", "0.0.0.0")
    except Exception:
        return False


def load_file_context(file_paths: list[str]) -> str:
    """Read and format multiple context files into a delimited prompt block."""
    blocks: list[str] = []
    for fp in file_paths:
        if fp == "-":
            content = sys.stdin.read()
            blocks.append(f"--- Context File: stdin ---\n{content}\n--- End File: stdin ---")
            continue
        p = Path(fp).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Context file not found: {fp}")
        content = p.read_text(encoding="utf-8", errors="replace")
        blocks.append(f"--- Context File: {p.name} ---\n{content}\n--- End File: {p.name} ---")
    return "\n\n".join(blocks)


def stream_chat(
    messages: list[dict[str, Any]],
    model: str = "qwen2.5-coder:14b",
    base_url: str = "http://127.0.0.1:11434/v1",
    temperature: float = 0.2,
    allow_remote: bool = False,
    timeout: float = 120.0,
) -> Iterator[str]:
    """Stream token deltas from an OpenAI-compatible local chat completions endpoint."""
    if not allow_remote and not is_loopback_url(base_url):
        raise ValueError(
            f"Endpoint '{base_url}' is not a loopback address. "
            "To prevent accidental data exfiltration, pass allow_remote=True to target remote hosts."
        )

    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "local-ai-management/1.0",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data: "):
                chunk_str = line[6:].strip()
                if chunk_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(chunk_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue


def complete_chat(
    messages: list[dict[str, Any]],
    model: str = "qwen2.5-coder:14b",
    base_url: str = "http://127.0.0.1:11434/v1",
    temperature: float = 0.2,
    allow_remote: bool = False,
    timeout: float = 120.0,
) -> str:
    """Execute a non-streaming completion and return the full assistant response."""
    chunks = list(
        stream_chat(
            messages=messages,
            model=model,
            base_url=base_url,
            temperature=temperature,
            allow_remote=allow_remote,
            timeout=timeout,
        )
    )
    return "".join(chunks)


def run_repl(
    model: str = "qwen2.5-coder:14b",
    base_url: str = "http://127.0.0.1:11434/v1",
    preset: str = "general",
    custom_system: str | None = None,
    temperature: float = 0.2,
    allow_remote: bool = False,
) -> int:
    """Run an interactive streaming REPL session for direct human inference."""
    system_text = custom_system if custom_system is not None else SYSTEM_PRESETS.get(preset, "")
    messages: list[dict[str, Any]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})

    print("=" * 64)
    print("Local AI Direct Inference REPL (Air-Gapped / Zero-Telemetry)")
    print(f"Model:       {model}")
    print(f"Endpoint:    {base_url}")
    print(f"Preset:      {preset}")
    print("Commands:    /help, /model <id>, /preset <name>, /clear, /quit")
    print("=" * 64)

    while True:
        try:
            user_input = input("\n>>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("quit", "exit", "q"):
                print("Session terminated.")
                break
            elif cmd == "help":
                print("Available slash commands:")
                print("  /help              Show this guidance")
                print("  /model <id>        Switch active model")
                print("  /preset <name>     Switch system preset (general, private_code, security, raw)")
                print("  /system <text>     Set custom system prompt")
                print("  /clear             Clear conversation history")
                print("  /status            Display current session configuration")
                print("  /quit              Exit REPL")
                continue
            elif cmd == "status":
                print(f"Model: {model} | Preset: {preset} | Messages in context: {len(messages)}")
                continue
            elif cmd == "clear":
                messages = [{"role": "system", "content": system_text}] if system_text else []
                print("Conversation history cleared.")
                continue
            elif cmd == "model":
                if not arg:
                    print(f"Current model: {model}")
                else:
                    model = arg
                    print(f"Switched model to: {model}")
                continue
            elif cmd == "preset":
                if arg not in SYSTEM_PRESETS:
                    print(f"Unknown preset '{arg}'. Options: {', '.join(SYSTEM_PRESETS.keys())}")
                else:
                    preset = arg
                    system_text = SYSTEM_PRESETS[arg]
                    # Update or set system message
                    messages = [m for m in messages if m.get("role") != "system"]
                    if system_text:
                        messages.insert(0, {"role": "system", "content": system_text})
                    print(f"Switched preset to '{arg}'.")
                continue
            elif cmd == "system":
                if not arg:
                    print(f"Current system prompt: {system_text}")
                else:
                    system_text = arg
                    messages = [m for m in messages if m.get("role") != "system"]
                    messages.insert(0, {"role": "system", "content": system_text})
                    print("Updated system prompt.")
                continue
            else:
                print(f"Unknown command: /{cmd}. Type /help for options.")
                continue

        messages.append({"role": "user", "content": user_input})
        print(f"\n[{model}]: ", end="", flush=True)
        assistant_chunks: list[str] = []

        try:
            for chunk in stream_chat(
                messages=messages,
                model=model,
                base_url=base_url,
                temperature=temperature,
                allow_remote=allow_remote,
            ):
                print(chunk, end="", flush=True)
                assistant_chunks.append(chunk)
            print()
            messages.append({"role": "assistant", "content": "".join(assistant_chunks)})
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        except Exception as exc:
            print(f"\nError: {exc}")

    return 0
