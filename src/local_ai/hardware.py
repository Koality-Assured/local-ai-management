"""Hardware inspection and profile tier classification."""

from __future__ import annotations

import dataclasses
import os
import platform
import shutil
import subprocess
from typing import Any


@dataclasses.dataclass
class HardwareProfile:
    """Detected compute hardware and mapped profile tier."""

    os_name: str
    cpu_cores: int
    system_ram_gb: float | None
    gpu_name: str | None
    vram_gb: float | None
    tier: str
    tier_description: str
    recommended_models: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "os_name": self.os_name,
            "cpu_cores": self.cpu_cores,
            "system_ram_gb": self.system_ram_gb,
            "gpu_name": self.gpu_name,
            "vram_gb": self.vram_gb,
            "tier": self.tier,
            "tier_description": self.tier_description,
            "recommended_models": self.recommended_models,
        }


def classify_hardware_tier(
    vram_gb: float | None,
    system_ram_gb: float | None,
    os_name: str | None = None,
    arch: str | None = None,
) -> tuple[str, str, list[dict[str, str]]]:
    """Classify hardware into standard Tier A through Tier E profiles."""
    current_os = (os_name or platform.system()).lower()
    current_arch = (arch or platform.machine()).lower()

    if vram_gb is not None and vram_gb >= 14.0:
        tier = "Tier A"
        desc = "High-Performance GPU Workstation (16 GB+ VRAM)"
        models = [
            {"id": "qwen2.5-coder:14b", "role": "Primary coding workhorse (100% in VRAM)"},
            {"id": "qwen2.5-coder:7b", "role": "High-velocity subagent deferral"},
            {"id": "deepseek-r1:14b", "role": "Local reasoning specialist"},
            {"id": "nomic-embed-text", "role": "Local embeddings"},
        ]
    elif vram_gb is not None and vram_gb >= 10.0:
        tier = "Tier B"
        desc = "Standard GPU Workstation (10-12 GB VRAM)"
        models = [
            {"id": "qwen2.5-coder:14b", "role": "Primary coding workhorse (bounded context 8k-16k)"},
            {"id": "qwen2.5-coder:7b", "role": "High-velocity worker (100% in VRAM)"},
            {"id": "nomic-embed-text", "role": "Local embeddings"},
        ]
    elif vram_gb is not None and vram_gb >= 6.0:
        tier = "Tier C"
        desc = "Entry / Mobile GPU Workstation (6-8 GB VRAM)"
        models = [
            {"id": "qwen2.5-coder:7b", "role": "Primary coding workhorse (100% in VRAM)"},
            {"id": "qwen2.5-coder:3b", "role": "Lightweight subagent worker"},
            {"id": "nomic-embed-text", "role": "Local embeddings"},
        ]
    elif "darwin" in current_os and current_arch in ("arm64", "aarch64"):
        tier = "Tier D"
        desc = "Unified Memory / Apple Silicon Workstation"
        models = [
            {"id": "qwen2.5-coder:14b", "role": "Primary coding workhorse (Metal accelerated)"},
            {"id": "qwen2.5-coder:7b", "role": "High-velocity subagent worker"},
            {"id": "deepseek-r1:14b", "role": "Local reasoning specialist"},
            {"id": "nomic-embed-text", "role": "Local embeddings"},
        ]
    else:
        tier = "Tier E"
        desc = "CPU-Only / Vector Acceleration Workstation"
        models = [
            {"id": "qwen2.5-coder:1.5b", "role": "High-speed CPU coding model"},
            {"id": "qwen2.5-coder:7b", "role": "Standard CPU coding model (AVX2/AVX-512)"},
            {"id": "nomic-embed-text", "role": "Local embeddings"},
        ]

    return tier, desc, models


def detect_hardware(
    override_vram_gb: float | None = None,
    override_ram_gb: float | None = None,
) -> HardwareProfile:
    """Inspect system hardware and return structured profile."""
    os_desc = f"{platform.system()} {platform.release()} ({platform.machine()})"
    cpu_cores = os.cpu_count() or 1

    # Detect System RAM
    ram_gb: float | None = override_ram_gb
    if ram_gb is None:
        try:
            import psutil  # type: ignore

            ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        except Exception:
            pass

    # Detect GPU & VRAM
    gpu_name: str | None = None
    vram_gb: float | None = override_vram_gb

    if vram_gb is None:
        nvsmi = shutil.which("nvidia-smi")
        if nvsmi:
            try:
                proc = subprocess.run(
                    [nvsmi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
                    if len(parts) >= 2:
                        gpu_name = parts[0]
                        vram_gb = round(float(parts[1]) / 1024.0, 1)
            except Exception:
                pass

    tier, tier_desc, models = classify_hardware_tier(
        vram_gb=vram_gb,
        system_ram_gb=ram_gb,
        os_name=platform.system(),
        arch=platform.machine(),
    )

    return HardwareProfile(
        os_name=os_desc,
        cpu_cores=cpu_cores,
        system_ram_gb=ram_gb,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        tier=tier,
        tier_description=tier_desc,
        recommended_models=models,
    )
