"""Empirical benchmarking suite for local models."""

from __future__ import annotations

import dataclasses
import json
import time
import urllib.parse
import urllib.request
from typing import Any, Callable

STANDARD_BENCHMARKS = [
    {
        "id": "code_gen",
        "name": "Algorithmic Code Generation",
        "system": "You are an expert Python engineer. Provide only clean, working Python code.",
        "prompt": "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges overlapping intervals. Include doctests.",
        "validator": lambda text: "def merge_intervals" in text and "return" in text,
    },
    {
        "id": "structured_json",
        "name": "Strict JSON Extraction",
        "system": "You are a data extraction engine. Return ONLY valid JSON adhering to the schema.",
        "prompt": "Extract entities into JSON with keys 'project', 'version', 'features' (list of strings): 'Antigravity AI Harness v2.4 features sandboxed worktrees and local LLM deferral.'",
        "validator": lambda text: _validate_json_extraction(text),
    },
    {
        "id": "refactor",
        "name": "Code Refactoring & Typing",
        "system": "You are a senior code reviewer. Fix violations and add full PEP 484 type annotations.",
        "prompt": "Refactor this snippet with type hints and docstrings: `def process_batch(items, threshold=10): return [i for i in items if i > threshold]`",
        "validator": lambda text: "def process_batch" in text and "list[" in text.lower(),
    },
    {
        "id": "reasoning",
        "name": "Logical & Mathematical Deduction",
        "system": "You are an analytical reasoning assistant. Show step-by-step logic and the final answer.",
        "prompt": "A train leaves Station A traveling at 60 mph. Another train leaves Station B 30 miles away traveling towards Station A at 90 mph. In how many minutes will they meet?",
        "validator": lambda text: "12" in text or "twelve" in text.lower(),
    },
]

SYSTEM_ONE_BENCHMARKS = [
    {
        "id": "sys1_choice_routing",
        "name": "System 1 Skill Dispatch (Choice)",
        "system": "You are a System 1 decision engine. Return typed choice: {'answer': str, 'confidence': float}.",
        "prompt": "State: User requests refactoring of auth middleware to use JWT verification. Questions: {'skill': {'type': 'choice', 'criteria': {'refactor': 'code cleanup', 'testing': 'test writing', 'deploy': 'deployment'}}}",
        "validator": lambda text: "refactor" in text.lower(),
        "is_system_one": True,
    },
    {
        "id": "sys1_mutation_gating",
        "name": "System 1 Mutation Gating (Noul)",
        "system": "You are a System 1 decision engine. Return calibrated boolean: {'answer': 'yes'|'no', 'p_true': float}.",
        "prompt": "State: git checkout -b feature/auth && git commit -m 'feat: add jwt'. Questions: {'is_mutating': {'type': 'noul', 'instructions': 'Does this change repository state?'}}",
        "validator": lambda text: "yes" in text.lower(),
        "is_system_one": True,
    },
    {
        "id": "sys1_complexity_scoring",
        "name": "System 1 Complexity Rubric (Score)",
        "system": "You are a System 1 decision engine. Return expected rubric score: {'score': float, 'variance': float}.",
        "prompt": "State: Replace print statement with logger.debug in helper.py. Questions: {'complexity': {'type': 'score', 'instructions': 'Score 1 to 5'}}",
        "validator": lambda text: "score" in text.lower(),
        "is_system_one": True,
    },
]


def _validate_json_extraction(text: str) -> bool:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    try:
        data = json.loads(cleaned.strip())
        return isinstance(data, dict) and "project" in data and "version" in data
    except Exception:
        return False


@dataclasses.dataclass
class BenchmarkRunResult:
    """Individual benchmark run metrics."""

    task_id: str
    task_name: str
    model_id: str
    ttft_ms: float
    total_duration_sec: float
    output_tokens: int
    tokens_per_sec: float
    passed_validation: bool
    output_preview: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "model_id": self.model_id,
            "ttft_ms": round(self.ttft_ms, 2),
            "total_duration_sec": round(self.total_duration_sec, 3),
            "output_tokens": self.output_tokens,
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "passed_validation": self.passed_validation,
            "output_preview": self.output_preview[:120] + "..." if len(self.output_preview) > 120 else self.output_preview,
        }


def run_single_benchmark(
    base_url: str,
    model_id: str,
    benchmark_task: dict[str, Any],
    mock: bool = False,
) -> BenchmarkRunResult:
    """Execute a single benchmark task against local endpoint or mock generator."""
    task_id = benchmark_task["id"]
    task_name = benchmark_task["name"]
    system_prompt = benchmark_task["system"]
    prompt = benchmark_task["prompt"]
    validator: Callable[[str], bool] = benchmark_task.get("validator", lambda _: True)

    if mock:
        is_sys1 = benchmark_task.get("is_system_one", False)
        if is_sys1:
            time.sleep(0.01)
            ttft_ms = 14.8
            mock_output = _get_mock_output(task_id)
            tokens = len(mock_output.split()) * 2
            gen_time = 0.015
            passed = validator(mock_output)
            return BenchmarkRunResult(
                task_id=task_id,
                task_name=task_name,
                model_id=model_id,
                ttft_ms=ttft_ms,
                total_duration_sec=0.01 + gen_time,
                output_tokens=tokens,
                tokens_per_sec=tokens / gen_time if gen_time > 0 else 0.0,
                passed_validation=passed,
                output_preview=mock_output,
            )

        # Simulate realistic local streaming latency and token generation
        time.sleep(0.05)  # 50ms TTFT
        ttft_ms = 48.5
        mock_output = _get_mock_output(task_id)
        tokens = len(mock_output.split()) * 2
        gen_time = 0.15  # ~100 tok/s
        time.sleep(0.02)
        passed = validator(mock_output)
        return BenchmarkRunResult(
            task_id=task_id,
            task_name=task_name,
            model_id=model_id,
            ttft_ms=ttft_ms,
            total_duration_sec=0.05 + gen_time,
            output_tokens=tokens,
            tokens_per_sec=tokens / gen_time if gen_time > 0 else 0.0,
            passed_validation=passed,
            output_preview=mock_output,
        )

    # Real HTTP SSE streaming execution
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "stream": True,
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body_bytes,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )

    start_time = time.perf_counter()
    first_token_time: float | None = None
    chunks: list[str] = []

    with urllib.request.urlopen(req, timeout=30) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if not line_str.startswith("data:"):
                continue
            data_part = line_str[5:].strip()
            if data_part == "[DONE]":
                break
            try:
                chunk_obj = json.loads(data_part)
                choices = chunk_obj.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {}).get("content", "")
                    if delta:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        chunks.append(delta)
            except Exception:
                pass

    end_time = time.perf_counter()
    total_sec = end_time - start_time
    ttft_ms = ((first_token_time - start_time) * 1000.0) if first_token_time else (total_sec * 1000.0)

    full_text = "".join(chunks)
    output_tokens = len(full_text.split()) * 2  # rough token approximation if not emitted in SSE
    gen_duration = total_sec - (ttft_ms / 1000.0)
    tok_per_sec = (output_tokens / gen_duration) if gen_duration > 0 else 0.0
    passed = validator(full_text)

    return BenchmarkRunResult(
        task_id=task_id,
        task_name=task_name,
        model_id=model_id,
        ttft_ms=ttft_ms,
        total_duration_sec=total_sec,
        output_tokens=output_tokens,
        tokens_per_sec=tok_per_sec,
        passed_validation=passed,
        output_preview=full_text,
    )


def _get_mock_output(task_id: str) -> str:
    if task_id == "code_gen":
        return """def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    \"\"\"Merge overlapping intervals.
    >>> merge_intervals([[1, 3], [2, 6], [8, 10]])
    [[1, 6], [8, 10]]
    \"\"\"
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for cur in intervals[1:]:
        prev = merged[-1]
        if cur[0] <= prev[1]:
            prev[1] = max(prev[1], cur[1])
        else:
            merged.append(cur)
    return merged"""
    if task_id == "structured_json":
        return '{"project": "Antigravity AI Harness", "version": "v2.4", "features": ["sandboxed worktrees", "local LLM deferral"]}'
    if task_id == "refactor":
        return """def process_batch(items: list[int], threshold: int = 10) -> list[int]:
    \"\"\"Filter items exceeding given threshold.\"\"\"
    return [i for i in items if i > threshold]"""
    if task_id == "reasoning":
        return "Combined speed is 60 + 90 = 150 mph. Time to meet = 30 / 150 = 0.2 hours = 12 minutes."
    if task_id == "sys1_choice_routing":
        return '{"skill": {"answer": "refactor", "confidence": 0.962}}'
    if task_id == "sys1_mutation_gating":
        return '{"is_mutating": {"answer": "yes", "p_true": 0.995, "confidence": 0.995}}'
    if task_id == "sys1_complexity_scoring":
        return '{"complexity": {"score": 1.2, "variance": 0.08}}'
    return "Mock model output"


def run_benchmark_suite(
    model_ids: list[str],
    base_url: str = "http://127.0.0.1:11434/v1",
    tasks: list[dict[str, Any]] | None = None,
    mock: bool = False,
    suite: str = "standard",
) -> list[BenchmarkRunResult]:
    """Execute benchmark suite across multiple models."""
    if tasks is not None:
        selected_tasks = tasks
    elif suite == "system_one":
        selected_tasks = SYSTEM_ONE_BENCHMARKS
    elif suite == "all":
        selected_tasks = STANDARD_BENCHMARKS + SYSTEM_ONE_BENCHMARKS
    else:
        selected_tasks = STANDARD_BENCHMARKS

    results: list[BenchmarkRunResult] = []
    for mid in model_ids:
        for task in selected_tasks:
            res = run_single_benchmark(base_url, mid, task, mock=mock)
            results.append(res)
    return results
