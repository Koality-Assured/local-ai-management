"""Model catalog loader and query interface."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any


def get_catalog_path() -> Path:
    """Return path to packaged catalog.json."""
    ref = files("local_ai.data").joinpath("catalog.json")
    try:
        candidate = Path(str(ref))
        if candidate.is_file():
            return candidate
    except Exception:
        pass
    fallback = Path(__file__).resolve().parent / "data" / "catalog.json"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError("Model catalog resource 'data/catalog.json' not found in package 'local_ai'")


def load_catalog(custom_path: Path | None = None) -> list[dict[str, Any]]:
    """Load model entries from the catalog.

    Raises:
        FileNotFoundError: If the catalog file or resource cannot be found.
        json.JSONDecodeError: If the catalog is not valid JSON.
    """
    if custom_path is not None:
        if not custom_path.is_file():
            raise FileNotFoundError(f"Model catalog not found at custom path: {custom_path}")
        with open(custom_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("models", [])

    ref = files("local_ai.data").joinpath("catalog.json")
    try:
        content = ref.read_text(encoding="utf-8")
        data = json.loads(content)
        return data.get("models", [])
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    fallback = Path(__file__).resolve().parent / "data" / "catalog.json"
    if fallback.is_file():
        with open(fallback, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("models", [])

    raise FileNotFoundError("Model catalog resource 'data/catalog.json' not found in package 'local_ai'")


def get_model(model_id: str, models: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """Find model by exact id or name."""
    entries = models if models is not None else load_catalog()
    target = model_id.strip().lower()
    for m in entries:
        if m.get("id", "").lower() == target or m.get("name", "").lower() == target:
            return m
    return None


def filter_by_tier(tier: str, models: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Filter models suitable for a specific hardware tier."""
    entries = models if models is not None else load_catalog()
    target_tier = tier.strip()
    return [m for m in entries if target_tier in m.get("recommended_tiers", [])]


def filter_by_task(task: str, models: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Filter models matching a task capability."""
    entries = models if models is not None else load_catalog()
    target_task = task.strip().lower()
    return [m for m in entries if target_task in [t.lower() for t in m.get("task_suitability", [])]]


def filter_system_one(models: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Filter models categorized as System 1 decision models."""
    entries = models if models is not None else load_catalog()
    return [m for m in entries if m.get("category") == "system_one" or m.get("is_system_one", False)]
