"""Model catalog loader and query interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_catalog_path() -> Path:
    """Return absolute path to models/catalog.json."""
    repo_path = Path(__file__).resolve().parents[2] / "models" / "catalog.json"
    if repo_path.exists():
        return repo_path
    cwd_path = Path.cwd() / "models" / "catalog.json"
    if cwd_path.exists():
        return cwd_path
    pkg_path = Path(__file__).resolve().parent / "models" / "catalog.json"
    if pkg_path.exists():
        return pkg_path
    return repo_path


def load_catalog(custom_path: Path | None = None) -> list[dict[str, Any]]:
    """Load model entries from the catalog."""
    catalog_file = custom_path or get_catalog_path()
    if not catalog_file.exists():
        return []
    with open(catalog_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("models", [])


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

