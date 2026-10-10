"""Modelfile and startup script synthesis engine."""

from __future__ import annotations


def generate_ollama_modelfile(
    base_model: str,
    context_tokens: int = 32768,
    temperature: float = 0.2,
    top_p: float = 0.95,
    repeat_penalty: float = 1.05,
    system_prompt: str = "",
) -> str:
    """Generate an optimized Ollama Modelfile."""
    lines = [
        f"FROM {base_model}",
        "",
        "# Runtime Context and Generation Parameters",
        f"PARAMETER num_ctx {context_tokens}",
        f"PARAMETER temperature {temperature}",
        f"PARAMETER top_p {top_p}",
        f"PARAMETER repeat_penalty {repeat_penalty}",
    ]
    if system_prompt:
        lines.append("")
        lines.append("# Custom Agent System Prompt")
        lines.append(f'SYSTEM """{system_prompt.strip()}"""')
    lines.append("")
    return "\n".join(lines)


def generate_llama_server_command(
    gguf_path: str,
    gpu_layers: int = 99,
    context_tokens: int = 32768,
    cpu_threads: int = 16,
    quantize_kv: bool = True,
    port: int = 8080,
    host: str = "127.0.0.1",
) -> str:
    """Generate CLI command for running llama-server with FlashAttention and quantized KV."""
    flags = [
        f"-m {gguf_path}",
        f"-ngl {gpu_layers}",
        f"-c {context_tokens}",
        "-fa",
        f"--port {port}",
        f"--host {host}",
    ]
    if cpu_threads > 0 and gpu_layers < 99:
        flags.append(f"-t {cpu_threads}")
    if quantize_kv:
        flags.append("-ctk q8_0 -ctv q8_0")
    return "./llama-server " + " \\\n  ".join(flags)
