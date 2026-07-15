#!/usr/bin/env python
"""Re-score all existing run JSON files using the updated scoring.py.

Usage:
    python rescore_runs.py results/20260303_005609

Reads every run_*.json under each model directory, re-scores with the
current scoring.py, writes the file back, and regenerates summary.json.
Prints before/after comparison per model.
"""

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

# Allow importing from mud_poc package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mud_poc.scoring import score_run
from mud_poc.config import default_models


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rescore_directory(results_dir: str) -> None:
    if not os.path.isdir(results_dir):
        print(f"ERROR: {results_dir} is not a directory")
        sys.exit(1)

    print(f"Re-scoring all runs in: {results_dir}\n")

    # Collect before/after scores per model
    before_scores: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    after_scores: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    all_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    total_files = 0
    total_rescored = 0

    for model_dir_name in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_dir_name)
        if not os.path.isdir(model_dir) or model_dir_name.startswith("_"):
            continue

        for fname in sorted(os.listdir(model_dir)):
            if not (fname.startswith("run_") and fname.endswith(".json")):
                continue

            fpath = os.path.join(model_dir, fname)
            total_files += 1

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  SKIP {fpath}: {exc}")
                continue

            # Save old scores
            old_scores = data.get("scores", {})
            if old_scores:
                before_scores[model_dir_name].append({
                    k: float(v) for k, v in old_scores.items()
                    if isinstance(v, (int, float))
                })

            # Re-score
            new_scores = score_run(data)
            total_rescored += 1

            after_scores[model_dir_name].append({
                k: float(v) for k, v in new_scores.items()
                if isinstance(v, (int, float))
            })

            # Write back
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Collect for summary
            all_results[model_dir_name].append({
                "model_key": data.get("model_key", model_dir_name),
                "run_number": data.get("run_number", 0),
                "seed": data.get("seed", 0),
                "temperature": data.get("temperature", 0.3),
                "success": data.get("success", False),
                "objective": data.get("objective", ""),
                "turns_used": data.get("turns", 0),
                "objective_state": data.get("objective_state", {}),
                "output_file": fpath,
                "usage": data.get("usage", {}),
                "score": new_scores,
            })

    print(f"Re-scored {total_rescored}/{total_files} files\n")

    # --- Print comparison table ---
    dimensions = [
        "goal_pursuit", "social_adaptation",
        "world_grounding", "strategic_sophistication", "total",
    ]
    dim_short = {
        "goal_pursuit": "Goal",
        "social_adaptation": "Social",
        "world_grounding": "Ground",
        "strategic_sophistication": "Strat",
        "total": "Total",
    }

    print("=" * 90)
    print(f"{'Model':<18}", end="")
    for d in dimensions:
        print(f" | {dim_short[d]:>6} old -> new", end="")
    print()
    print("-" * 90)

    for model_key in sorted(after_scores.keys()):
        old_list = before_scores.get(model_key, [])
        new_list = after_scores[model_key]

        print(f"{model_key:<18}", end="")
        for d in dimensions:
            old_avg = _avg([s.get(d, 0) for s in old_list]) if old_list else 0.0
            new_avg = _avg([s.get(d, 0) for s in new_list])
            if old_list:
                print(f" | {old_avg:5.2f} -> {new_avg:5.2f}", end="")
            else:
                print(f" |   n/a -> {new_avg:5.2f}", end="")
        print()

    print("=" * 90)

    # --- Print spread (max - min across models) ---
    print(f"\n{'Spread':<18}", end="")
    for d in dimensions:
        model_avgs = []
        for model_key in after_scores:
            new_list = after_scores[model_key]
            model_avgs.append(_avg([s.get(d, 0) for s in new_list]))
        if len(model_avgs) >= 2:
            spread = max(model_avgs) - min(model_avgs)
            print(f" | spread: {spread:5.2f}  ", end="")
        else:
            print(f" |    n/a        ", end="")
    print()

    # --- Generate summary.json ---
    all_models_cfg = {m.key: m for m in default_models()}
    model_breakdown: Dict[str, Any] = {}
    flat_results: List[Dict[str, Any]] = []

    for model_key in sorted(all_results.keys()):
        runs = all_results[model_key]
        flat_results.extend(runs)
        scored = [r for r in runs if r.get("score")]
        avg_fn = lambda key: _avg([float(r.get("score", {}).get(key, 0)) for r in scored])
        model_breakdown[model_key] = {
            "runs": len(runs),
            "success_rate": sum(1 for r in runs if r.get("success")) / len(runs) if runs else 0,
            "avg_turns_used": _avg([int(r.get("turns_used", 0)) for r in runs]),
            "avg_goal_pursuit": avg_fn("goal_pursuit"),
            "avg_social_adaptation": avg_fn("social_adaptation"),
            "avg_world_grounding": avg_fn("world_grounding"),
            "avg_strategic_sophistication": avg_fn("strategic_sophistication"),
            "avg_cost_usd": _avg([float(r.get("usage", {}).get("estimated_cost_usd", 0)) for r in runs]),
            "scored_runs": len(scored),
        }

    all_scored = [r for r in flat_results if r.get("score")]
    overall_avg = lambda key: _avg([float(r.get("score", {}).get(key, 0)) for r in all_scored])
    summary = {
        "generated": os.path.basename(results_dir),
        "total_runs": len(flat_results),
        "models": model_breakdown,
        "overall": {
            "runs": len(flat_results),
            "success_rate": sum(1 for r in flat_results if r.get("success")) / len(flat_results) if flat_results else 0,
            "avg_turns_used": _avg([int(r.get("turns_used", 0)) for r in flat_results]),
            "avg_goal_pursuit": overall_avg("goal_pursuit"),
            "avg_social_adaptation": overall_avg("social_adaptation"),
            "avg_world_grounding": overall_avg("world_grounding"),
            "avg_strategic_sophistication": overall_avg("strategic_sophistication"),
            "avg_cost_usd": _avg([float(r.get("usage", {}).get("estimated_cost_usd", 0)) for r in flat_results]),
            "scored_runs": len(all_scored),
        },
    }

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSummary written: {summary_path} ({len(flat_results)} total runs)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rescore_runs.py <results_dir>")
        print("Example: python rescore_runs.py results/20260303_005609")
        sys.exit(1)
    rescore_directory(sys.argv[1])
