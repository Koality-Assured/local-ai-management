"""Human-approved onboarding orchestration for local model environments."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from local_ai.hardware import HardwareProfile, detect_hardware


def check_endpoint_health(base_url: str = "http://127.0.0.1:11434/v1") -> tuple[bool, list[str]]:
    """Probe OpenAI-compatible endpoint health and list loaded models."""
    models_url = base_url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(models_url, headers={"User-Agent": "local-ai-management"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data") or data.get("models") or []
            names: list[str] = [
                str(entry_id)
                for m in models
                if isinstance(m, dict) and (entry_id := (m.get("id") or m.get("name")))
            ]
            return True, names
    except Exception:
        return False, []


def run_onboarding(
    approved: bool = False,
    profile_only: bool = False,
    base_url: str = "http://127.0.0.1:11434/v1",
) -> dict[str, Any]:
    """Execute onboarding assessment with human-approval verification."""
    profile: HardwareProfile = detect_hardware()
    is_online, loaded_models = check_endpoint_health(base_url)

    result = {
        "hardware": profile.to_dict(),
        "endpoint": base_url,
        "endpoint_online": is_online,
        "loaded_models": loaded_models,
        "human_approved": approved,
        "profile_only": profile_only,
    }

    if profile_only:
        result["status"] = "profile_inspected"
        result["message"] = "Profile inspected successfully. No modifications made."
        return result

    if not approved:
        result["status"] = "approval_required"
        result["message"] = (
            "Local AI model execution is optional and requires explicit human approval "
            "in this environment before installing runtimes or downloading weights. "
            "Re-run with --approve to confirm."
        )
        return result

    result["status"] = "approved"
    if is_online:
        result["message"] = f"Local AI engine is online at {base_url} and ready for inference."
    else:
        result["message"] = "Local AI engine is offline. Start Ollama ('ollama serve') to complete setup."
    return result
