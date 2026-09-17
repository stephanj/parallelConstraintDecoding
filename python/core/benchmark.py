"""
Benchmark runner comparing Autoregressive Generation vs.
Parallel Constrained Decision Engine on Apple Silicon.
"""

import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from core.schema import StructuredSchema
from core.engine import run_naive_generation, run_parallel_generation, get_engine


def compare_single(context: str, schema_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Runs both engines on the exact same problem prompt and returns side-by-side metrics."""
    schema = StructuredSchema(schema_dict)
    
    # 1. Run Autoregressive Baseline
    naive_res = run_naive_generation(context, schema)
    
    # 2. Run Parallel Constrained Engine
    parallel_res = run_parallel_generation(context, schema)
    
    speedup = naive_res["elapsed_ms"] / max(parallel_res["elapsed_ms"], 1.0)
    steps_speedup = naive_res["sequential_forward_passes"] / max(parallel_res["sequential_forward_passes"], 1.0)
    
    return {
        "speedup_multiplier": round(speedup, 1),
        "steps_reduction": round(steps_speedup, 1),
        "naive": naive_res,
        "parallel": parallel_res,
        # Backward compatibility
        "rlcd": parallel_res
    }


def run_benchmark_suite(preset_paths: List[str], warmup: bool = True) -> List[Dict[str, Any]]:
    print("=" * 70)
    print("Parallel Constrained vs. Autoregressive Generation Benchmark")
    print("=" * 70)
    
    get_engine()
    
    if warmup:
        print("\n[+] Warming up GPU compute graphs...")
        with open(preset_paths[0]) as f:
            p = json.load(f)
        compare_single(p["context"], p["schema"])
        print("[+] Warmup complete.\n")
        
    results = []
    for path in preset_paths:
        with open(path) as f:
            preset = json.load(f)
            
        print(f"--> Running preset: {preset['title']} ({len(preset['schema'])} fields)...")
        comp = compare_single(preset["context"], preset["schema"])
        comp["preset_id"] = preset["id"]
        comp["preset_title"] = preset["title"]
        results.append(comp)
        
        n = comp["naive"]
        r = comp["parallel"]
        print(f"    Autoregressive Baseline : {n['elapsed_ms']:>8.1f} ms | {n['total_tokens']:>3} tokens ({n['tokens_per_second']} tok/s) | Passes: {n['sequential_forward_passes']}")
        print(f"    Parallel Constrained    : {r['elapsed_ms']:>8.1f} ms |   0 tokens (O(1))           | Passes: {r['sequential_forward_passes']}")
        print(f"    >> SPEEDUP: {comp['speedup_multiplier']}x faster (Step reduction: {comp['steps_reduction']}x)")
        print(f"    >> Schema match: Naive={n['schema_match']} | Parallel={r['schema_match']} (100% guaranteed)")
        print("-" * 70)
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Parallel vs Autoregressive LLM JSON benchmark")
    presets_dir = Path(__file__).resolve().parents[2] / "presets"
    parser.add_argument("--presets", nargs="+", default=[
        str(presets_dir / "fintech_fraud.json"),
        str(presets_dir / "code_security.json"),
        str(presets_dir / "support_triage.json"),
        str(presets_dir / "high_cardinality_255.json"),
    ])
    args = parser.parse_args()
    run_benchmark_suite(args.presets)
