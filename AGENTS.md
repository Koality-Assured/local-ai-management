# Agent Guidelines — local-ai-management

Canonical operational rules for AI agents working in this repository.

## Purpose & Scope

`local-ai-management` is a public satellite repository providing host-local LLM serving architecture, dynamic hardware profiling, empirical benchmarking suites, Modelfile templates, and AI harness integration tools.

## Critical Normative Directives

### 1. Privacy & Zero Host PII Leaks (Strict MUST)
- **Zero Personal Identifiers**: Never commit personal user names, private machine paths (`C:\Users\<Name>\`), personal IP addresses, or host-specific serial numbers.
- **Hardware Profile Generalization**: All hardware analyses must be expressed in terms of standardized **Hardware Profile Tiers (Tier A through Tier E)** rather than specific personal workstation setups.
- **Dynamic Detection**: Code must dynamically inspect the executing environment at runtime (`local-ai profile` / `detect_hardware.py`); never hardcode or assume single-machine specifications.

### 2. Human Approval Gate for Environment Mutations (Strict MUST)
- **Zero Automatic Downloads**: Agents and scripts **MUST NOT** automatically download multi-gigabyte model weights or install background daemons (Ollama / llama-server) without explicit human confirmation.
- **Optional Local AI**: Local model execution is strictly optional. Workstation onboarding workflows must always remain functional even when local AI is disabled or skipped.
- **Explicit Flags**: All mutating onboarding commands require an explicit confirmation flag (e.g. `--approve`).

### 3. Empirical Grounding & Benchmarking Standards
- All model performance claims (latency, throughput, memory consumption, benchmark accuracy) must be validated with reproducible empirical measurements or cited from authoritative primary model releases (Qwen, DeepSeek, Meta AI).
- Benchmarking scripts must support a `--mock` mode to enable fast, deterministic unit testing in CI environments without live GPU daemons.

### 4. Code & Architecture Standards
- **Python Standard Library First**: Core CLI and utility scripts should minimize external dependencies to allow execution on any developer machine out-of-the-box.
- **Cross-Platform Compatibility**: Code and documentation must support Windows 11, macOS (Apple Silicon Metal), and Linux (CUDA/ROCm/CPU).
- **Conventional Commits**: All commit messages must strictly adhere to the Conventional Commits specification (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).

## Directory Structure

```text
local-ai-management/
├── docs/                     # Architectural documentation & specifications
│   ├── hardware-profiles.md  # Tier A-E hardware definitions & VRAM budget math
│   ├── serving-engines.md    # Ollama, llama.cpp, vLLM setup recipes & tuning
│   ├── frontier-deferral.md  # Frontier-to-local task delegation taxonomy
│   └── benchmarking-methodology.md # Standardized latency & throughput evaluation
├── models/                   # Optimized Ollama Modelfiles
│   └── modelfiles/           # Optimized Ollama Modelfiles
├── src/                      # Python package source
│   └── local_ai/             # CLI, hardware probe, benchmarking, and onboarding
│       ├── data/             # Packaged resources
│       │   └── catalog.json  # Canonical registry of evaluated local models
│       ├── __init__.py
│       ├── hardware.py       # Dynamic hardware detection & tier classification
│       ├── catalog.py        # Model catalog loading and querying
│       ├── benchmark.py      # Latency & throughput benchmarking suite
│       ├── onboard.py        # Human-approved onboarding workflow
│       ├── modelfile.py      # Modelfile synthesis generator
│       └── cli.py            # Unified local-ai CLI interface
└── tests/                    # Comprehensive unit tests
```
