# Direct Local Model Inference Operational Guide

Operational handbook for human operators running direct, air-gapped, and restricted inference workloads on local models.

---

## 1. Executive Summary & Rationale

Frontier commercial AI models (Claude 3.7/3.5, GPT-4o, Gemini 1.5/2.0) operate under rigid constraints that preclude certain classes of high-value engineering and security work:

1. **Safety Refusals & Moralizing Preambles**: Frontier providers enforce commercial alignment filters that frequently refuse legitimate vulnerability reproduction, exploit payload analysis, shellcode inspection, and red-team auditing.
2. **Confidentiality & WAN Exfiltration Risk**: Proprietary codebases, internal trade secrets, client NDA data, API keys, and corporate credentials cannot be transmitted over public networks or ingested into vendor training pools.
3. **Air-Gapped & Offline Continuity**: Critical infrastructure audits and field security assessments must function without continuous WAN connectivity.
4. **Zero-Cost Unmetered Batch Work**: Bulk automated code transformation, extensive multi-candidate fuzzing analysis, and dense local classification run free of token meters.

`local-ai-management` provides direct, terminal-native inference tooling designed specifically for the human operator to interact with local System 2 models (e.g. `qwen2.5-coder:14b`, `deepseek-r1:14b`) and System 1 decision models without intermediaries or cloud leaks.

---

## 2. Architectural Boundary: Public Tooling vs Private Repositories

When conducting sensitive or restricted work, strict architectural separation is required between public tooling and private intellectual property:

```
+-----------------------------------------------------------------------+
|  Public Open-Source Tier (Host-Agnostic, Zero Secrets)               |
|                                                                       |
|  - Koality-Assured/local-ai-management (Catalog, Benchmarks, REPL)    |
|  - ai-router harness (`harness chat --provider local`)                |
|  - Ollama / llama.cpp / vLLM runtime binaries                         |
+-----------------------------------------------------------------------+
                                   |
                                   | (Invocations via standard CLI)
                                   v
+-----------------------------------------------------------------------+
|  Private Workstation Tier (Air-Gapped / Private Git Repository)       |
|                                                                       |
|  Repo: `Koality-Assured/private-workstation-ai` (Private)             |
|                                                                       |
|  - Internal exploit research, PoC scripts, and red-team payloads      |
|  - Proprietary corporate source code & NDA documentation              |
|  - Domain-specific system prompts and prompt libraries                |
|  - Custom fine-tuned LoRA weights & private Modelfiles                |
|  - Local session notes & scratch analysis                             |
+-----------------------------------------------------------------------+
```

### Guidance on Private Koality-Assured Repository
If you are storing:
- Real vulnerability exploits, private CVE PoCs, or offensive security playbooks.
- Proprietary customer IP or internal confidential infrastructure manifests.
- Specialized system prompts tuned for internal business logic.

**Recommendation**: Create a private repository under Koality-Assured (e.g. `Koality-Assured/private-workstation-ai`). Keep the execution tooling (`local-ai` CLI and `harness`) completely generic and public, while pointing them at private target files and custom system prompt files stored inside your private repository.

---

## 3. Direct Inference Modes

The `local-ai chat` CLI supports multiple modes of interaction.

### 3.1 Interactive Terminal REPL

Launch an interactive multi-turn session with streaming response:

```bash
# Standard interactive coding session
local-ai chat --model qwen2.5-coder:14b

# Security audit session with offensive security preset
local-ai chat --model qwen2.5-coder:14b --preset security

# Private code refactoring session
local-ai chat --model qwen2.5-coder:14b --preset private_code
```

#### In-Session Slash Commands
During interactive sessions, use slash commands to steer execution:
- `/help`: Display available commands.
- `/model <model_id>`: Dynamically switch models mid-conversation (e.g. `/model deepseek-r1:14b`).
- `/preset <name>`: Switch system preset (`general`, `private_code`, `security`, `analysis`, `raw`).
- `/system <prompt>`: Override the system prompt with custom instructions.
- `/status`: Show active model, preset, and context length.
- `/clear`: Flush conversation memory and start clean.
- `/quit` or `Ctrl+C`: Terminate session.

---

### 3.2 One-Shot Queries & Streaming CLI

Execute quick queries directly from the shell without entering the REPL:

```bash
# Direct one-shot query
local-ai chat -m qwen2.5-coder:14b -q "Explain the difference between mutexes and spinlocks in Linux kernel modules"

# Buffer output and print once (ideal for terminal scripts)
local-ai chat -m qwen2.5-coder:14b -q "Write a Python script to parse ELF headers" --no-stream

# Machine-readable JSON output
local-ai chat -m qwen2.5-coder:14b -q "List 3 common TOCTOU race conditions" --json
```

---

### 3.3 File Context Ingestion

Directly inject local files into the model context without copy-pasting:

```bash
# Single file audit
local-ai chat -m qwen2.5-coder:14b -f ./kernel_patch.c -q "Audit this patch for memory leaks or null dereferences"

# Multiple context files
local-ai chat -m qwen2.5-coder:14b \
  -f ./src/auth.c \
  -f ./include/auth.h \
  --preset security \
  -q "Identify authorization bypass vectors in these components"
```

The CLI automatically formats files into delimited blocks:
```
--- Context File: auth.c ---
[file content]
--- End File: auth.c ---
```

---

### 3.4 Piped STDIN Ingestion

Pipe command outputs, logs, diffs, or decompiled code directly into `local-ai chat`:

```bash
# Analyze a git diff before committing
git diff HEAD~1 | local-ai chat -m qwen2.5-coder:14b -q "Review this diff for security regressions"

# Audit access logs for anomalous behavior
cat /var/log/nginx/access.log | local-ai chat -m qwen2.5-coder:14b --preset security -q "Identify potential SQLi or traversal patterns"

# Direct prompt piping without -q flag
echo "Explain format string exploitation in C" | local-ai chat -m qwen2.5-coder:14b --preset security
```

---

## 4. System Prompt Presets

`local-ai chat` includes pre-calibrated system prompts tailored for restricted tasks:

| Preset Name | Target Objective | Core Behavioral Invariant |
|:---|:---|:---|
| `general` | Daily technical tasks | Direct, concise, zero corporate fluff or preambles. |
| `private_code` | Proprietary code refactoring | Air-gapped focus, zero external telemetry, strict adherence to security invariants. |
| `security` | Exploit analysis & auditing | Objective technical analysis, exploit reproduction mechanics, zero moralizing refusals. |
| `analysis` | Logic verification & architecture | First-principles breakdown, edge case mapping, formal verification. |
| `raw` | Pure model defaults | Empty system prompt; direct unconditioned weight inference. |

You can also pass custom system prompts via `--system "<string>"` or `--system-file <path>`.

---

## 5. Working with Uncensored & Custom GGUF Models

When analyzing malware, reverse engineering, or offensive payloads, standard open-source weights may occasionally include mild fine-tuning guardrails. To achieve 100% unconstrained analysis:

### 5.1 Using Uncensored/Abliterated Models
Download or compile an unaligned/ablated model (e.g. Qwen2.5-Coder abliterated or DeepSeek-R1 unaligned):

```bash
# Pull an unaligned model in Ollama
ollama pull huihui-ai/qwen2.5-coder-abliterated:14b

# Run directly via local-ai
local-ai chat -m huihui-ai/qwen2.5-coder-abliterated:14b --preset security
```

### 5.2 Building Custom Modelfiles
Synthesize custom Modelfiles configured for maximum context and zero refusal:

```bash
# Generate Modelfile template
local-ai modelfile --model qwen2.5-coder:14b --context 32768 --temperature 0.2 > Modelfile.local

# Build customized local model
ollama create my-local-coder:14b -f Modelfile.local

# Run customized model
local-ai chat -m my-local-coder:14b
```

---

## 6. Air-Gap & Exfiltration Safety Guarantees

`local-ai` enforces strict networking controls:
1. **Loopback Protection**: By default, `local-ai chat` validates that the target endpoint resolves strictly to loopback (`127.0.0.1`, `localhost`, `::1`, `0.0.0.0`). Non-loopback URLs are rejected with an explicit error to prevent accidental packet transmission across the LAN or WAN.
2. **Explicit Override**: Remote endpoints require the explicit `--allow-remote` flag.
3. **Zero Telemetry**: No usage metrics, prompts, or hashes are sent upstream.
