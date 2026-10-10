"""Unit tests for local-ai CLI commands."""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from local_ai.cli import main


class TestCli(unittest.TestCase):
    """Test CLI dispatch and output formatting."""

    def test_cli_profile_json(self) -> None:
        out = io.StringIO()
        with patch("sys.stdout", out):
            code = main(["profile", "--json"])
        self.assertEqual(code, 0)
        data = json.loads(out.getvalue())
        self.assertIn("tier", data)
        self.assertIn("recommended_models", data)

    def test_cli_catalog_json(self) -> None:
        out = io.StringIO()
        with patch("sys.stdout", out):
            code = main(["catalog", "--tier", "Tier A", "--json"])
        self.assertEqual(code, 0)
        models = json.loads(out.getvalue())
        self.assertIsInstance(models, list)
        self.assertTrue(any(m["id"] == "qwen2.5-coder:14b" for m in models))

    def test_cli_catalog_single_id(self) -> None:
        out = io.StringIO()
        with patch("sys.stdout", out):
            code = main(["catalog", "--id", "qwen2.5-coder:14b", "--json"])
        self.assertEqual(code, 0)
        m = json.loads(out.getvalue())
        self.assertEqual(m["id"], "qwen2.5-coder:14b")

    def test_cli_benchmark_mock(self) -> None:
        out = io.StringIO()
        with patch("sys.stdout", out):
            code = main(["benchmark", "--model", "qwen2.5-coder:14b", "--mock", "--json"])
        self.assertEqual(code, 0)
        results = json.loads(out.getvalue())
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertTrue(r["passed_validation"])

    def test_cli_onboard_unapproved(self) -> None:
        out = io.StringIO()
        with patch("sys.stdout", out):
            code = main(["onboard", "--json"])
        self.assertEqual(code, 0)
        res = json.loads(out.getvalue())
        self.assertEqual(res["status"], "approval_required")
        self.assertFalse(res["human_approved"])

    def test_cli_onboard_approved(self) -> None:
        out = io.StringIO()
        with patch("sys.stdout", out):
            code = main(["onboard", "--approve", "--json"])
        self.assertEqual(code, 0)
        res = json.loads(out.getvalue())
        self.assertEqual(res["status"], "approved")
        self.assertTrue(res["human_approved"])

    def test_cli_modelfile(self) -> None:
        out = io.StringIO()
        with patch("sys.stdout", out):
            code = main(["modelfile", "--model", "qwen2.5-coder:14b"])
        self.assertEqual(code, 0)
        self.assertIn("FROM qwen2.5-coder:14b", out.getvalue())


if __name__ == "__main__":
    unittest.main()
