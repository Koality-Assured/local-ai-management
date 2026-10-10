"""Unified CLI control plane for local-ai-management."""

from __future__ import annotations

import argparse
import json
import sys

from local_ai.benchmark import STANDARD_BENCHMARKS, SYSTEM_ONE_BENCHMARKS, run_benchmark_suite
from local_ai.catalog import filter_by_task, filter_by_tier, filter_system_one, get_model, load_catalog
from local_ai.hardware import detect_hardware
from local_ai.modelfile import generate_llama_server_command, generate_ollama_modelfile
from local_ai.onboard import run_onboarding


def build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--json", action="store_true", help="Format output as JSON")

    parser = argparse.ArgumentParser(
        prog="local-ai",
        description="Local AI Management: Hardware profiling, model catalog, benchmarking, and onboarding.",
        parents=[shared],
    )
    sub = parser.add_subparsers(dest="cmd")

    # profile
    sub.add_parser("profile", help="Inspect compute hardware and classify hardware profile tier", parents=[shared])

    # catalog
    p_cat = sub.add_parser("catalog", help="Inspect machine-readable model catalog", parents=[shared])
    p_cat.add_argument("--tier", help="Filter models by hardware tier (e.g. 'Tier A')")
    p_cat.add_argument("--task", help="Filter models by task suitability")
    p_cat.add_argument("--system-one", action="store_true", help="Filter for System 1 decision models")
    p_cat.add_argument("--id", help="Inspect detailed specifications for a specific model id")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run empirical benchmarks across latency and quality", parents=[shared])
    p_bench.add_argument("--model", default="qwen2.5-coder:14b", help="Model ID to benchmark")
    p_bench.add_argument("--models", help="Comma-separated model IDs to benchmark")
    p_bench.add_argument("--suite", choices=["standard", "system_one", "all"], default="standard", help="Benchmark suite to execute")
    p_bench.add_argument("--base-url", default="http://127.0.0.1:11434/v1", help="OpenAI-compatible base URL")
    p_bench.add_argument("--mock", action="store_true", help="Run in mock/offline mode for CI/testing")

    # onboard
    p_onboard = sub.add_parser("onboard", help="Probe hardware and guide human-approved onboarding", parents=[shared])
    p_onboard.add_argument("--approve", action="store_true", help="Explicit human approval for local AI onboarding")
    p_onboard.add_argument("--profile-only", action="store_true", help="Only inspect hardware capabilities")
    p_onboard.add_argument("--base-url", default="http://127.0.0.1:11434/v1", help="Target local endpoint")

    # modelfile
    p_model = sub.add_parser("modelfile", help="Synthesize Ollama Modelfile or llama-server command", parents=[shared])
    p_model.add_argument("--model", required=True, help="Base model identifier or GGUF path")
    p_model.add_argument("--context", type=int, default=32768, help="Context window token count")
    p_model.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    p_model.add_argument("--engine", choices=["ollama", "llama-server"], default="ollama", help="Target engine")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.cmd:
        parser.print_help()
        return 0

    if args.cmd == "profile":
        hw = detect_hardware()
        if args.json:
            print(json.dumps(hw.to_dict(), indent=2))
            return 0
        print("=== Hardware Capability & Profile Classification ===")
        print(f"OS:               {hw.os_name}")
        print(f"CPU Cores:        {hw.cpu_cores}")
        print(f"System RAM:       {hw.system_ram_gb} GB" if hw.system_ram_gb else "System RAM:       Unknown")
        print(f"GPU:              {hw.gpu_name or 'None (CPU fallback)'}")
        print(f"VRAM:             {hw.vram_gb} GB" if hw.vram_gb else "VRAM:             None")
        print(f"Hardware Tier:    {hw.tier} - {hw.tier_description}")
        print("\nRecommended Model Fleet:")
        for m in hw.recommended_models:
            print(f"  - {m['id']:<18} ({m['role']})")
        return 0

    if args.cmd == "catalog":
        models = load_catalog()
        if args.id:
            m = get_model(args.id, models)
            if not m:
                print(f"Model '{args.id}' not found in catalog.", file=sys.stderr)
                return 1
            if args.json:
                print(json.dumps(m, indent=2))
                return 0
            print(f"=== {m['name']} ({m['id']}) ===")
            print(f"Parameters:       {m['parameters_billion']}B")
            print(f"Quantization:     {m['default_quantization']}")
            print(f"Disk Size:        {m['size_gb']} GB")
            print(f"Minimum VRAM:     {m['minimum_vram_gb']} GB")
            print(f"Recommended Tiers: {', '.join(m['recommended_tiers'])}")
            print(f"Primary Role:     {m['primary_role']}")
            print(f"Pull Command:     {m['ollama_pull_command']}")
            print(f"Notes:            {m['notes']}")
            return 0

        if args.tier:
            models = filter_by_tier(args.tier, models)
        if args.task:
            models = filter_by_task(args.task, models)
        if getattr(args, "system_one", False):
            models = filter_system_one(models)

        if args.json:
            print(json.dumps(models, indent=2))
            return 0

        print(f"=== Model Catalog ({len(models)} models) ===")
        for m in models:
            tiers = ", ".join(m.get("recommended_tiers", []))
            print(f"- {m['id']:<32} | {m['size_gb']:>4} GB | Tiers: [{tiers}] | {m['primary_role']}")
        return 0

    if args.cmd == "benchmark":
        model_list = [m.strip() for m in args.models.split(",")] if args.models else [args.model]
        suite_name = getattr(args, "suite", "standard")
        results = run_benchmark_suite(model_list, base_url=args.base_url, mock=args.mock, suite=suite_name)
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
            return 0

        print(f"=== Local Model Benchmark Results: Suite '{suite_name}' ({len(results)} runs) ===")
        print(f"{'Task':<26} {'Model':<18} {'TTFT':<10} {'Tok/s':<10} {'Passed':<8}")
        print("-" * 76)
        for r in results:
            passed_str = "YES" if r.passed_validation else "NO"
            print(f"{r.task_id:<26} {r.model_id:<18} {r.ttft_ms:>6.1f} ms {r.tokens_per_sec:>7.1f} t/s {passed_str:<8}")
        return 0

    if args.cmd == "onboard":
        res = run_onboarding(approved=args.approve, profile_only=args.profile_only, base_url=args.base_url)
        if args.json:
            print(json.dumps(res, indent=2))
            return 0

        hw = res["hardware"]
        print("=== Local AI Workstation Onboarding ===")
        print(f"Detected Profile: {hw['tier']} - {hw['tier_description']}")
        print(f"Endpoint:         {res['endpoint']} ({'ONLINE' if res['endpoint_online'] else 'OFFLINE'})")

        if res["profile_only"]:
            print(f"\nStatus: {res['status']}")
            print(res["message"])
            return 0

        if not res["human_approved"]:
            print("\n" + "=" * 60)
            print("[HUMAN APPROVAL REQUIRED]")
            print(res["message"])
            print("=" * 60)
            return 0

        print(f"\nStatus: APPROVED")
        print(res["message"])
        return 0

    if args.cmd == "modelfile":
        if args.engine == "ollama":
            content = generate_ollama_modelfile(
                base_model=args.model,
                context_tokens=args.context,
                temperature=args.temperature,
            )
        else:
            content = generate_llama_server_command(
                gguf_path=args.model,
                context_tokens=args.context,
            )
        print(content)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
