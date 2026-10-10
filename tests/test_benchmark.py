"""Unit tests for benchmarking harness."""

from __future__ import annotations

import unittest

from local_ai.benchmark import (
    STANDARD_BENCHMARKS,
    _validate_json_extraction,
    run_benchmark_suite,
    run_single_benchmark,
)


class TestBenchmarkSuite(unittest.TestCase):
    """Test empirical benchmark runner and validation logic."""

    def test_json_extraction_validator(self) -> None:
        valid_json = '{"project": "local-ai", "version": "1.0", "features": ["bench"]}'
        self.assertTrue(_validate_json_extraction(valid_json))

        valid_markdown_fenced = '```json\n{"project": "local-ai", "version": "1.0"}\n```'
        self.assertTrue(_validate_json_extraction(valid_markdown_fenced))

        invalid_json = "This is not json"
        self.assertFalse(_validate_json_extraction(invalid_json))

    def test_run_single_benchmark_mock(self) -> None:
        task = STANDARD_BENCHMARKS[0]  # code_gen
        res = run_single_benchmark(
            base_url="http://127.0.0.1:11434/v1",
            model_id="qwen2.5-coder:14b",
            benchmark_task=task,
            mock=True,
        )
        self.assertEqual(res.task_id, "code_gen")
        self.assertEqual(res.model_id, "qwen2.5-coder:14b")
        self.assertGreater(res.ttft_ms, 0)
        self.assertGreater(res.tokens_per_sec, 0)
        self.assertTrue(res.passed_validation)
        self.assertIn("def merge_intervals", res.output_preview)

    def test_run_benchmark_suite_mock(self) -> None:
        results = run_benchmark_suite(
            model_ids=["qwen2.5-coder:14b", "qwen2.5-coder:7b"],
            mock=True,
        )
        self.assertEqual(len(results), len(STANDARD_BENCHMARKS) * 2)
        for r in results:
            self.assertTrue(r.passed_validation)
            as_dict = r.to_dict()
            self.assertIn("ttft_ms", as_dict)
            self.assertIn("tokens_per_sec", as_dict)


if __name__ == "__main__":
    unittest.main()
