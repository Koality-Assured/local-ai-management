# Frontier-to-Local Deferral Architecture & Harness Integration

Architectural specification for enabling frontier coding agents (Gemini 3 Flash/Pro, Claude 3.7 Sonnet, GPT-4o) to defer deterministic, mechanical, or high-volume tasks to locally served models via AI harness control planes.

---

## 1. Problem Statement & Economics

Frontier model APIs (Claude 3.7 Sonnet, GPT-4o, Gemini 3 Pro) are expensive ($2.50–$15.00 per 1M output tokens) and subject to rolling window rate limits (`RESOURCE_EXHAUSTED` / 429 errors). Autonomous agent sessions frequently expend thousands of frontier tokens on mechanical or bulky operations:
- Fixing formatting and linter violations across dozens of files.
- Generating unit test skeletons and mock boilerplate.
- Synthesizing commit messages conforming to Conventional Commits.
- Distilling 500-line test failure outputs or security scan logs into structured facts.

Expending frontier reasoning budgets on these tasks creates unnecessary API spend and exhausts token windows. Deferring this work to local models running on workstation hardware (75–140+ tok/s at **$0.00 marginal cost**) preserves cloud quotas for high-leverage architectural orchestration.

---

## 2. Work Taxonomy & Deferral Suitability Matrix

| Work Category | Typical Tasks | Optimal Execution Model | Recommended Model |
| :--- | :--- | :--- | :--- |
| **Category 1: Frontier Orchestration** *(Do Not Defer)* | System design, cross-repo planning, security threat modeling (STRIDE), ambiguous requirement alignment, complex PR synthesis. | **Frontier Agent** (Parent orchestrator) | Claude 3.7 Sonnet / Gemini 3.1 Pro / GPT-5 |
| **Category 2: Bounded Autonomous Coding** *(Optional Deferral)* | Function implementation from detailed spec, single-file refactoring, writing complex algorithms, AST structure queries. | **Frontier OR Local Workhorse** | Qwen 2.5 Coder 14B (Q4_K_M) or Frontier Standard |
| **Category 3: Mechanical & High-Volume** *(Mandatory Deferral)* | Markdown linting fixes, docstring generation, boilerplate test cases, log summarization, JSON schema extraction, commit message drafting. | **Local Model** (Zero-cost worker) | Qwen 2.5 Coder 7B / 14B |
| **Category 4: Quota Fallback** *(Resilience Deferral)* | Secondary model exhaustion (429 recovery), offline air-gapped development, network degradation. | **Local Model** (Automatic failover) | Qwen 2.5 Coder 14B / 32B |

---

## 3. Harness Integration Pattern

The harness control plane provides first-class support for routing requests to local OpenAI-compatible endpoints:

```text
+-------------------------------------------------------------------------------+
|                       FRONTIER ORCHESTRATOR (Agent)                          |
|         Claude 3.7 Sonnet / Gemini 3.1 Pro / GPT-4o (Antigravity/Cursor)      |
+-------------------------------------------------------------------------------+
                                     |
                       Delegates Category 3/4 Task
                                     v
+-------------------------------------------------------------------------------+
|                        HARNESS CLI CONTROL PLANE                              |
|          harness chat --provider local --model qwen2.5-coder:14b -q "..."     |
+-------------------------------------------------------------------------------+
                                     |
                          Loopback HTTP Request
                                     v
+-------------------------------------------------------------------------------+
|                         LOCAL SERVING RUNTIME                                 |
|          Ollama / llama-server @ http://127.0.0.1:11434/v1                    |
|          Accelerated via Dedicated GPU VRAM / Apple Silicon Metal             |
+-------------------------------------------------------------------------------+
```

### 3.1 CLI Invocation Examples

```bash
# Verify local runtime status
harness local status

# Run mechanical linting or boilerplate generation locally
harness chat --provider local --model qwen2.5-coder:14b -q "Generate pytest fixtures for user authentication service"

# Check hardware onboarding status
harness local onboarding --profile-only
```

---

## 4. Empirical ROI & Token Savings

Across simulated 100-turn autonomous coding workloads, deferring Category 3 tasks to local models achieves:
- **40%–60% reduction in cloud API token consumption**.
- **Zero cloud API spend** on boilerplate and mechanical lint fixes.
- **Sub-150ms Time-to-First-Token (TTFT)** eliminating network latency bottlenecks.
- Complete resilience against cloud API 429 rate limit triggers.
