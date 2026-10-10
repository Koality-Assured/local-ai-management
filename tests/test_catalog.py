"""Unit tests for model catalog queries."""

from __future__ import annotations

import unittest

from local_ai.catalog import filter_by_task, filter_by_tier, get_model, load_catalog


class TestModelCatalog(unittest.TestCase):
    """Test model catalog loading and search functions."""

    def test_load_catalog(self) -> None:
        models = load_catalog()
        self.assertGreaterEqual(len(models), 6)
        ids = [m["id"] for m in models]
        self.assertIn("qwen2.5-coder:14b", ids)
        self.assertIn("qwen2.5-coder:7b", ids)
        self.assertIn("deepseek-r1:14b", ids)

    def test_get_model(self) -> None:
        m = get_model("qwen2.5-coder:14b")
        self.assertIsNotNone(m)
        self.assertEqual(m["parameters_billion"], 14.7)
        self.assertEqual(m["default_quantization"], "Q4_K_M")

    def test_filter_by_tier(self) -> None:
        tier_a_models = filter_by_tier("Tier A")
        ids = [m["id"] for m in tier_a_models]
        self.assertIn("qwen2.5-coder:14b", ids)
        self.assertIn("deepseek-r1:14b", ids)

        tier_e_models = filter_by_tier("Tier E")
        tier_e_ids = [m["id"] for m in tier_e_models]
        self.assertIn("qwen2.5-coder:1.5b", tier_e_ids)

    def test_filter_by_task(self) -> None:
        refactor_models = filter_by_task("refactoring")
        ids = [m["id"] for m in refactor_models]
        self.assertIn("qwen2.5-coder:14b", ids)


if __name__ == "__main__":
    unittest.main()
