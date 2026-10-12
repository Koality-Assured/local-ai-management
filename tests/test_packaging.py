"""Packaging and wheel distribution verification tests."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import zipfile
from importlib.resources import files
from pathlib import Path

from local_ai.catalog import load_catalog


class TestPackaging(unittest.TestCase):
    """Verify package data inclusion and wheel build integrity."""

    def test_catalog_resource_exists(self) -> None:
        """Verify catalog.json is accessible via importlib.resources."""
        catalog_resource = files("local_ai.data").joinpath("catalog.json")
        self.assertTrue(catalog_resource.is_file())
        models = load_catalog()
        self.assertGreaterEqual(len(models), 6)

    def test_wheel_contains_catalog_json(self) -> None:
        """Verify building a wheel includes local_ai/data/catalog.json."""
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(
                [sys.executable, "-m", "build", "--wheel", "--outdir", td, str(repo_root)],
                check=True,
                capture_output=True,
                text=True,
            )
            wheels = list(Path(td).glob("*.whl"))
            self.assertEqual(len(wheels), 1)
            wheel_path = wheels[0]
            with zipfile.ZipFile(wheel_path, "r") as zf:
                namelist = zf.namelist()
                matching = [n for n in namelist if n.endswith("local_ai/data/catalog.json")]
                self.assertTrue(
                    len(matching) > 0,
                    f"Catalog JSON not found in wheel archive: {namelist}",
                )


if __name__ == "__main__":
    unittest.main()
