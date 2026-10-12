"""Unit tests for hardware profiling and tier classification."""

from __future__ import annotations

import unittest

from local_ai.hardware import classify_hardware_tier, detect_hardware


class TestHardwareProfiling(unittest.TestCase):
    """Test hardware detection and tier classification."""

    def test_classify_tier_a(self) -> None:
        tier, desc, models = classify_hardware_tier(vram_gb=16.0, system_ram_gb=128.0)
        self.assertEqual(tier, "Tier A")
        self.assertIn("16 GB+", desc)
        model_ids = [m["id"] for m in models]
        self.assertIn("qwen2.5-coder:14b", model_ids)
        self.assertIn("deepseek-r1:14b", model_ids)

    def test_classify_tier_b(self) -> None:
        tier, desc, models = classify_hardware_tier(vram_gb=11.5, system_ram_gb=64.0)
        self.assertEqual(tier, "Tier B")
        self.assertIn("10-12 GB", desc)
        model_ids = [m["id"] for m in models]
        self.assertIn("qwen2.5-coder:7b", model_ids)

    def test_classify_tier_c(self) -> None:
        tier, desc, models = classify_hardware_tier(vram_gb=7.8, system_ram_gb=32.0)
        self.assertEqual(tier, "Tier C")
        self.assertIn("6-8 GB", desc)
        model_ids = [m["id"] for m in models]
        self.assertIn("qwen2.5-coder:7b", model_ids)
        self.assertIn("qwen2.5-coder:3b", model_ids)

    def test_classify_tier_d_apple_silicon(self) -> None:
        tier, desc, models = classify_hardware_tier(
            vram_gb=None,
            system_ram_gb=64.0,
            os_name="Darwin",
            arch="arm64",
        )
        self.assertEqual(tier, "Tier D")
        self.assertIn("Apple Silicon", desc)
        model_ids = [m["id"] for m in models]
        self.assertIn("qwen2.5-coder:14b", model_ids)

    def test_classify_tier_e_cpu_only(self) -> None:
        tier, desc, models = classify_hardware_tier(
            vram_gb=None,
            system_ram_gb=32.0,
            os_name="Linux",
            arch="x86_64",
        )
        self.assertEqual(tier, "Tier E")
        self.assertIn("CPU-Only", desc)
        model_ids = [m["id"] for m in models]
        self.assertIn("qwen2.5-coder:1.5b", model_ids)

    def test_detect_hardware_overrides(self) -> None:
        profile = detect_hardware(override_vram_gb=16.0, override_ram_gb=64.0)
        self.assertEqual(profile.tier, "Tier A")
        self.assertEqual(profile.vram_gb, 16.0)
        self.assertEqual(profile.system_ram_gb, 64.0)
        as_dict = profile.to_dict()
        self.assertEqual(as_dict["tier"], "Tier A")
        self.assertIn("recommended_models", as_dict)


if __name__ == "__main__":
    unittest.main()
