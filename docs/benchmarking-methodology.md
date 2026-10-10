# Local Model Benchmarking Methodology

Standardized empirical protocol for measuring latency, generation throughput, memory consumption, and task completion quality of locally hosted language models.

---

## 1. Key Evaluation Metrics

1. **Time-to-First-Token (TTFT)**:
   - Measures prompt evaluation latency from request dispatch until the first token is received via Server-Sent Events (SSE).
   - Crucial for interactive agent responsiveness. Target: $<150\text{ ms}$ on local GPU acceleration.
2. **Sustained Generation Speed (tokens/second)**:
   - Measures output token emission rate after TTFT until completion.
   - Target: $>70\text{ tok/s}$ for 14B models; $>110\text{ tok/s}$ for 7B models.
3. **Memory Footprint**:
   - VRAM allocated by weights and KV cache at designated context lengths (2k, 8k, 16k, 32k).
4. **Task Correctness & Quality**:
   - Evaluates whether the model adheres to instructions, generates valid syntax, passes test assertions, and produces compliant structured JSON outputs.

---

## 2. Standardized Evaluation Suites

The benchmarking suite tests four representative task profiles:

| Suite Name | Prompt Type | Evaluation Criteria |
| :--- | :--- | :--- |
| `code_gen` | Algorithmic implementation (e.g. merge intervals, LRU cache) | Syntactic correctness, algorithmic complexity, boundary condition handling. |
| `structured_json` | Extracting entities and metrics into strict JSON schemas | Schema compliance, valid JSON parsing, correct field types. |
| `refactor` | Fixing lint violations and typing in an existing code snippet | Lint clean, type check passes, behavioral invariance. |
| `reasoning` | Multi-step mathematical or architectural deduction | Step-by-step logical soundess, final answer accuracy. |

---

## 3. Running Benchmarks

### 3.1 Live Benchmark Execution

Run against a local serving engine (Ollama or llama-server):

```bash
# Benchmark default recommended model
local-ai benchmark --model qwen2.5-coder:14b

# Run across multiple models
local-ai benchmark --models qwen2.5-coder:14b,qwen2.5-coder:7b --runs 3

# Export results as JSON
local-ai benchmark --model qwen2.5-coder:14b --json > results/benchmark_run.json
```

### 3.2 Offline / Mock Testing

To validate benchmarking logic without a running GPU daemon (e.g. in CI pipelines):

```bash
local-ai benchmark --mock --model qwen2.5-coder:14b
```
