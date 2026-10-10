"""Unit tests for human-approved onboarding logic."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from local_ai.onboard import check_endpoint_health, run_onboarding


class TestOnboarding(unittest.TestCase):
    """Test onboarding workflow and approval enforcement."""

    def test_onboarding_profile_only(self) -> None:
        res = run_onboarding(profile_only=True)
        self.assertEqual(res["status"], "profile_inspected")
        self.assertTrue(res["profile_only"])
        self.assertIn("hardware", res)

    def test_onboarding_requires_approval(self) -> None:
        res = run_onboarding(approved=False)
        self.assertEqual(res["status"], "approval_required")
        self.assertFalse(res["human_approved"])
        self.assertIn("requires explicit human approval", res["message"])

    def test_onboarding_approved_offline(self) -> None:
        with patch("local_ai.onboard.check_endpoint_health", return_value=(False, [])):
            res = run_onboarding(approved=True)
            self.assertEqual(res["status"], "approved")
            self.assertTrue(res["human_approved"])
            self.assertFalse(res["endpoint_online"])
            self.assertIn("offline", res["message"])

    def test_onboarding_approved_online(self) -> None:
        with patch("local_ai.onboard.check_endpoint_health", return_value=(True, ["qwen2.5-coder:14b"])):
            res = run_onboarding(approved=True)
            self.assertEqual(res["status"], "approved")
            self.assertTrue(res["human_approved"])
            self.assertTrue(res["endpoint_online"])
            self.assertIn("online", res["message"])


if __name__ == "__main__":
    unittest.main()
