"""
Episodic Ablation Analysis -- CrucibleBench A2 Reviewer Response

Compares model performance under two conditions:
  (A) Full persistence (original design)
  (B) Episodic reset every N turns (world state reset)

Reports:
  (i)   Spearman rank correlation between conditions
  (ii)  Between-model eta-squared under each condition (Kruskal-Wallis)
  (iii) Persistence x Model interaction per dimension (two-way ANOVA
        via aligned-rank transform, at the scenario-cell level)

Usage:
  python analysis_episodic_ablation.py \
    --persistent-dir Run_Data \
    --episodic-dir results/episodic_ablation \
    --models gpt_5_4,claude_sonnet,deepseek_r1,grok_4,olmo_3_1

  If --episodic-dir is omitted, the script runs a self-test against the
  persistent data only (useful for verifying the pipeline works before
  you have Condition B data).

Changes from v1:
  - Fix #37: Exact permutation p-value for Spearman rho at small n
    (scipy's asymptotic approximation returns spurious p~0 for rho=1.0
    with n=5; true exact p = 0.017)
  - Fix #3: Interaction tests now operate on scenario-cell means
    (10 per model = 5 seeds x 2 objectives) instead of raw runs,
    avoiding pseudoreplication from treating correlated within-cell
    repetitions as independent observations
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from itertools import permutations
from math import factorial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMENSIONS = [
    "goal_pursuit",
    "social_adaptation",
    "world_grounding",
    "strategic_sophistication",
    "total",
]

DEFAULT_MODELS = [
    "gpt_5_4",
    "claude_sonnet",
    "deepseek_r1",
    "grok_4",
    "olmo_3_1",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_runs(base_dir: str, models: List[str]) -> Dict[str, List[dict]]:
    """Load run JSONs for each model.

    Searches recursively under base_dir for directories matching each
    model name, then collects all .json files in those directories.
    """
    result: Dict[str, List[dict]] = {}
    base = Path(base_dir)

    for model in models:
        # Find all directories matching the model name anywhere under base
        model_dirs = list(base.rglob(model))
        runs = []
        for md in model_dirs:
            if not md.is_dir():
                continue
            for jf in sorted(md.glob("*.json")):
                try:
                    with open(jf, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    # Must have scores dict
                    if "scores" in data:
                        runs.append(data)
                except (json.JSONDecodeError, KeyError):
                    continue
        result[model] = runs

    return result


def _extract_dim_score(run: dict, dim: str) -> float:
    """Pull a single dimension score from a run JSON."""
    scores = run.get("scores", {})
    if dim == "total":
        # Total is average of the four dimensions
        vals = []
        for d in DIMENSIONS[:-1]:
            v = scores.get(d, {})
            if isinstance(v, dict):
                v = v.get("rubric_score", v.get("score", None))
            if v is not None:
                vals.append(float(v))
        return float(np.mean(vals)) if vals else np.nan
    else:
        v = scores.get(dim, {})
        if isinstance(v, dict):
            v = v.get("rubric_score", v.get("score", None))
        return float(v) if v is not None else np.nan


def extract_scores(
    runs: Dict[str, List[dict]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Convert run dicts to {model: {dim: array_of_scores}}."""
    result = {}
    for model, run_list in runs.items():
        dim_scores: Dict[str, list] = {d: [] for d in DIMENSIONS}
        for run in run_list:
            for dim in DIMENSIONS:
                dim_scores[dim].append(_extract_dim_score(run, dim))
        result[model] = {d: np.array(v) for d, v in dim_scores.items()}
    return result


def extract_cell_scores(
    runs: Dict[str, List[dict]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Aggregate run-level scores to scenario-cell means.

    A scenario cell is defined by (seed, objective). Each cell typically
    has 5 repetitions. Returns {model: {dim: array_of_cell_means}}, where
    each array has length = number of unique cells (expected: 10 per model
    = 5 seeds x 2 objectives).

    This is the correct unit of analysis for interaction tests, avoiding
    pseudoreplication from treating correlated within-cell reps as
    independent (Fix #3).
    """
    result = {}
    for model, run_list in runs.items():
        # Group runs by (seed, objective)
        cells: Dict[Tuple, List[dict]] = defaultdict(list)
        for run in run_list:
            seed = run.get("seed", run.get("scenario_seed", "unknown"))
            obj = run.get("objective", run.get("objective_key", "unknown"))
            cells[(seed, obj)].append(run)

        dim_means: Dict[str, list] = {d: [] for d in DIMENSIONS}
        for cell_key, cell_runs in sorted(cells.items()):
            for dim in DIMENSIONS:
                scores = [_extract_dim_score(r, dim) for r in cell_runs]
                scores = [s for s in scores if not np.isnan(s)]
                if scores:
                    dim_means[dim].append(float(np.mean(scores)))

        result[model] = {d: np.array(v) for d, v in dim_means.items()}
    return result


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _spearman_exact_p(rho: float, n: int) -> float:
    """Exact two-tailed p-value for Spearman rho via permutation enumeration.

    For small n (<=8), enumerates all n\! permutations. For larger n,
    falls back to scipy's asymptotic approximation.  This fixes the
    known issue where scipy returns p~0 for rho=1.0 at small n (#37).
    """
    if n > 8:
        # Asymptotic is adequate for n > 8
        _, p = stats.spearmanr(list(range(n)), list(range(n)))
        return float(p)

    # Enumerate all permutations of ranks 0..n-1
    ref = np.arange(n, dtype=float)
    count_ge = 0
    total = factorial(n)
    for perm in permutations(range(n)):
        perm_arr = np.array(perm, dtype=float)
        r, _ = stats.spearmanr(ref, perm_arr)
        if abs(r) >= abs(rho) - 1e-12:
            count_ge += 1

    return count_ge / total


def rank_models(
    scores: Dict[str, Dict[str, np.ndarray]],
    dim: str = "total",
) -> List[Tuple[str, float]]:
    """Return models sorted by mean score on `dim`, descending."""
    means = []
    for model, dims in scores.items():
        arr = dims.get(dim, np.array([]))
        if len(arr) > 0:
            means.append((model, float(np.nanmean(arr))))
    return sorted(means, key=lambda x: x[1], reverse=True)


def ranking_comparison(
    scores_a: Dict[str, Dict[str, np.ndarray]],
    scores_b: Dict[str, Dict[str, np.ndarray]],
    dim: str = "total",
) -> dict:
    """Spearman rank correlation between two conditions on total score."""
    rank_a = rank_models(scores_a, dim)
    rank_b = rank_models(scores_b, dim)

    # Only compare models present in both
    models_a = {m for m, _ in rank_a}
    models_b = {m for m, _ in rank_b}
    shared = sorted(models_a & models_b)

    if len(shared) < 3:
        return {"error": f"only {len(shared)} shared models, need >=3"}

    # Assign ranks (1 = best)
    rank_a_map = {m: i + 1 for i, (m, _) in enumerate(rank_a) if m in shared}
    rank_b_map = {m: i + 1 for i, (m, _) in enumerate(rank_b) if m in shared}

    ranks_a_vec = [rank_a_map[m] for m in shared]
    ranks_b_vec = [rank_b_map[m] for m in shared]

    rho, _ = stats.spearmanr(ranks_a_vec, ranks_b_vec)
    # Use exact permutation test for small n
    p = _spearman_exact_p(float(rho), len(shared))

    return {
        "dimension": dim,
        "n_models": len(shared),
        "ranking_persistent": [(m, s) for m, s in rank_a if m in shared],
        "ranking_episodic": [(m, s) for m, s in rank_b if m in shared],
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "p_method": "exact_permutation" if len(shared) <= 8 else "asymptotic",
    }


def kruskal_eta2(groups: List[np.ndarray]) -> dict:
    """Kruskal-Wallis H-test with classical eta-squared."""
    clean = [g[~np.isnan(g)] for g in groups]
    clean = [g for g in clean if len(g) > 0]
    if len(clean) < 2:
        return {"H": np.nan, "p": np.nan, "eta2": np.nan, "n": 0, "k": 0}

    H, p = stats.kruskal(*clean)
    N = sum(len(g) for g in clean)
    k = len(clean)
    eta2 = (H - k + 1) / (N - k) if N > k else np.nan
    return {"H": float(H), "p": float(p), "eta2": float(eta2),
            "n": int(N), "k": int(k)}


def effect_size_comparison(
    scores_a: Dict[str, Dict[str, np.ndarray]],
    scores_b: Dict[str, Dict[str, np.ndarray]],
) -> dict:
    """Between-model eta-sq under each condition, per dimension."""
    result = {}
    for dim in DIMENSIONS:
        groups_a = [scores_a[m][dim] for m in scores_a if dim in scores_a[m]]
        groups_b = [scores_b[m][dim] for m in scores_b if dim in scores_b[m]]
        kw_a = kruskal_eta2(groups_a)
        kw_b = kruskal_eta2(groups_b)
        result[dim] = {
            "persistent": kw_a,
            "episodic": kw_b,
            "eta2_delta": kw_a["eta2"] - kw_b["eta2"],
        }
    return result


def interaction_test(
    scores_a: Dict[str, Dict[str, np.ndarray]],
    scores_b: Dict[str, Dict[str, np.ndarray]],
    dim: str,
) -> dict:
    """Aligned-rank-transform interaction test (Model x Condition).

    Expects CELL-LEVEL data (10 per model = 5 seeds x 2 objectives),
    not raw runs, to avoid pseudoreplication (Fix #3).
    """
    models = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if not models:
        return {"error": "no shared models"}

    # Build long-form arrays
    y_vals, model_codes, cond_codes = [], [], []
    per_model_deltas = {}

    for i, m in enumerate(models):
        arr_a = scores_a[m].get(dim, np.array([]))
        arr_b = scores_b[m].get(dim, np.array([]))
        arr_a = arr_a[~np.isnan(arr_a)]
        arr_b = arr_b[~np.isnan(arr_b)]

        per_model_deltas[m] = {
            "persistent_mean": float(np.mean(arr_a)) if len(arr_a) > 0 else np.nan,
            "episodic_mean": float(np.mean(arr_b)) if len(arr_b) > 0 else np.nan,
            "delta": float(np.mean(arr_a) - np.mean(arr_b)) if len(arr_a) > 0 and len(arr_b) > 0 else np.nan,
        }

        for v in arr_a:
            y_vals.append(v)
            model_codes.append(i)
            cond_codes.append(0)
        for v in arr_b:
            y_vals.append(v)
            model_codes.append(i)
            cond_codes.append(1)

    y = np.array(y_vals)
    model_f = np.array(model_codes)
    cond_f = np.array(cond_codes)
    n = len(y)
    n_models = len(models)

    # -- Aligned Rank Transform for interaction --
    # 1. Compute cell means for main effects
    cell_means_model = np.zeros(n)
    cell_means_cond = np.zeros(n)
    grand_mean = np.mean(y)

    for mi in range(n_models):
        mask = model_f == mi
        cell_means_model[mask] = np.mean(y[mask])
    for ci in [0, 1]:
        mask = cond_f == ci
        cell_means_cond[mask] = np.mean(y[mask])

    # For model x cond cells
    cell_means_mc = np.zeros(n)
    for mi in range(n_models):
        for ci in [0, 1]:
            mask = (model_f == mi) & (cond_f == ci)
            if np.any(mask):
                cell_means_mc[mask] = np.mean(y[mask])

    # 2. Aligned values for interaction: subtract main effects, keep interaction
    aligned_int = y - cell_means_model - cell_means_cond + grand_mean
    ranks_int = stats.rankdata(aligned_int)

    # 3. F-test on ranked aligned values for interaction
    # Two-way ANOVA: compute SS for interaction
    grand_rank_mean = np.mean(ranks_int)
    ss_interaction = 0.0
    for mi in range(n_models):
        for ci in [0, 1]:
            mask = (model_f == mi) & (cond_f == ci)
            if np.any(mask):
                cell_mean = np.mean(ranks_int[mask])
                row_mean = np.mean(ranks_int[model_f == mi])
                col_mean = np.mean(ranks_int[cond_f == ci])
                n_cell = np.sum(mask)
                ss_interaction += n_cell * (cell_mean - row_mean - col_mean + grand_rank_mean) ** 2

    ss_within = np.sum((ranks_int - np.array([
        np.mean(ranks_int[(model_f == mi) & (cond_f == ci)])
        for mi, ci in zip(model_f, cond_f)
    ])) ** 2)

    df_interaction = (n_models - 1) * (2 - 1)  # (a-1)(b-1)
    df_within = n - n_models * 2

    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 1

    F_int = ms_interaction / ms_within if ms_within > 0 else 0
    p_int = 1.0 - stats.f.cdf(F_int, df_interaction, df_within)

    # -- Raw F for condition main effect --
    ss_cond = sum(
        np.sum(cond_f == ci) * (np.mean(y[cond_f == ci]) - grand_mean) ** 2
        for ci in [0, 1]
    )
    ss_total = np.sum((y - grand_mean) ** 2)
    ss_within_raw = ss_total - ss_cond - sum(
        np.sum(model_f == mi) * (np.mean(y[model_f == mi]) - grand_mean) ** 2
        for mi in range(n_models)
    ) - ss_interaction  # approximate
    df_cond = 1
    ms_cond = ss_cond / df_cond
    ms_within_raw = ss_within_raw / df_within if df_within > 0 else 1
    F_cond = ms_cond / ms_within_raw if ms_within_raw > 0 else 0
    p_cond = 1.0 - stats.f.cdf(F_cond, df_cond, df_within)

    return {
        "dim": dim,
        "n_obs": int(n),
        "n_models": n_models,
        "F_interaction_ART": float(F_int),
        "p_interaction_ART": float(p_int),
        "df_interaction": int(df_interaction),
        "df_within": int(df_within),
        "F_condition_raw": float(F_cond),
        "p_condition_raw": float(p_cond),
        "condition_means": {
            "persistent": float(np.mean(y[cond_f == 0])),
            "episodic": float(np.mean(y[cond_f == 1])),
            "delta": float(np.mean(y[cond_f == 0]) - np.mean(y[cond_f == 1])),
        },
        "per_model_deltas": per_model_deltas,
    }


# ---------------------------------------------------------------------------
# Self-test helpers
# ---------------------------------------------------------------------------

def self_test_split(
    runs: Dict[str, List[dict]],
) -> Tuple[Dict[str, List[dict]], Dict[str, List[dict]]]:
    """Split runs into odd/even halves for self-testing."""
    a, b = {}, {}
    for model, run_list in runs.items():
        a[model] = [r for i, r in enumerate(run_list) if i % 2 == 0]
        b[model] = [r for i, r in enumerate(run_list) if i % 2 == 1]
    return a, b


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(ranking, effect_sizes, interactions, self_test=False):
    mode = "SELF-TEST" if self_test else "ABLATION"
    print(f"\n{'='*72}")
    print(f"  EPISODIC ABLATION -- {mode} ANALYSIS")
    print(f"{'='*72}")

    # (i) Rankings
    rho = ranking.get("spearman_rho", float("nan"))
    p_rho = ranking["spearman_p"]
    p_method = ranking.get("p_method", "unknown")

    print(f"\n(i) MODEL RANKINGS: persistent vs episodic\n")
    print(f"  {'Model':<18s} {'persistent mean':>14s}  {'rank':>4s}   {'episodic mean':>14s}  {'rank':>4s}")
    print(f"  {'-'*18} {'-'*14} {'-'*5}   {'-'*14} {'-'*5}")
    rank_p = ranking.get("ranking_persistent", [])
    rank_e = ranking.get("ranking_episodic", [])
    rank_e_map = {m: (i+1, s) for i, (m, s) in enumerate(rank_e)}
    for i, (m, s) in enumerate(rank_p):
        e_rank, e_score = rank_e_map.get(m, (0, 0))
        print(f"  {m:<18s} {s:>14.3f} {i+1:>5d}   {e_score:>14.3f} {e_rank:>5d}")

    print(f"\n  Spearman rho = {rho:.3f}  (p = {p_rho:.4f}, n = {ranking['n_models']}, {p_method})")
    if rho > 0.9:
        print("  -> Rankings are highly stable across conditions.")

    # (ii) Effect sizes
    print(f"\n{'-'*72}")
    print("(ii) BETWEEN-MODEL EFFECT SIZE (eta-sq) BY CONDITION\n")
    print(f"  {'Dimension':<28s} {'eta2_persistent':>15s} {'eta2_episodic':>12s} {'delta':>10s}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*10}")
    for dim in DIMENSIONS:
        es = effect_sizes[dim]
        eta_p = es["persistent"]["eta2"]
        eta_e = es["episodic"]["eta2"]
        delta = es["eta2_delta"]
        flag = " ***" if abs(delta) > 0.10 else ""
        print(f"  {dim:<28s} {eta_p:>12.3f} {eta_e:>12.3f}     {delta:>+.3f}{flag}")
    print(f"\n  *** = delta eta-sq > 0.10 (large practical difference)")

    # (iii) Interactions
    print(f"\n{'-'*72}")
    print("(iii) PERSISTENCE x MODEL INTERACTION (ART, scenario-cell level)\n")
    print(f"  {'Dimension':<28s} {'F_interaction':>14s} {'p':>10s} {'df':>8s}  {'Cond delta':>10s}")
    print(f"  {'-'*28} {'-'*14} {'-'*10} {'-'*8}  {'-'*10}")
    for dim in DIMENSIONS:
        it = interactions[dim]
        F_val = it["F_interaction_ART"]
        p_val = it["p_interaction_ART"]
        df_str = f"{it['df_interaction']},{it['df_within']}"
        delta = it["condition_means"]["delta"]
        sig = " *" if p_val < 0.05 else ""
        print(f"  {dim:<28s} {F_val:>14.2f} {p_val:>10.4f} {df_str:>8s}  {delta:>+10.3f}{sig}")

    # Social adaptation detail
    sa = interactions.get("social_adaptation", {})
    if sa and sa.get("p_interaction_ART", 0) > 0.05:
        print(f"\n  Social Adaptation: per-model persistence effect (persistent mean - episodic mean)")
        for m, d in sorted(sa.get("per_model_deltas", {}).items(),
                           key=lambda x: x[1].get("delta", 0), reverse=True):
            print(f"    {m:<18s} persistent={d['persistent_mean']:.3f}  "
                  f"episodic={d['episodic_mean']:.3f}  delta={d['delta']:+.3f}")

    # Summary
    print(f"\n{'-'*72}")
    print("SUMMARY VERDICT\n")

    eta2_total = effect_sizes["total"]
    sa_p = interactions.get("social_adaptation", {}).get("p_interaction_ART", 1.0)

    if rho > 0.9 and abs(eta2_total["eta2_delta"]) < 0.05:
        print("  Rankings and effect sizes are stable across conditions.")
        print("  Persistence does NOT appear to change model ordering.")
        if sa_p < 0.05:
            print(f"  However, Social Adaptation shows a significant "
                  f"persistence x model interaction (p={sa_p:.4f}).")
            print("  -> Partial vindication: persistence matters for the "
                  "social rubric but not overall rankings.")
        else:
            print("  -> Consider reframing: persistence is an environment "
                  "feature, not a demonstrated evaluative feature.")
    elif rho < 0.7 or abs(eta2_total["eta2_delta"]) > 0.10:
        print("  Rankings or effect sizes differ meaningfully.")
        print("  Persistence IS producing evaluative signal.")
        print("  -> Paper's persistence claim is empirically supported.")
    else:
        print("  Mixed signal. Rankings moderately stable but some "
              "dimensions show persistence effects.")
        print("  -> Report dimension-level results honestly; persistence "
              "matters for some aspects but not all.")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Episodic ablation analysis for CrucibleBench A2"
    )
    parser.add_argument(
        "--persistent-dir", required=True,
        help="Directory with persistent-condition run JSONs (e.g., Run_Data/)"
    )
    parser.add_argument(
        "--episodic-dir", default=None,
        help="Directory with episodic-condition run JSONs. "
             "If omitted, runs a self-test by splitting persistent data."
    )
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS),
        help="Comma-separated model keys to include"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save JSON results (default: print only)"
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    self_test = args.episodic_dir is None

    print(f"Models: {models}")
    print(f"Persistent dir: {args.persistent_dir}")

    if self_test:
        print("Episodic dir: NOT PROVIDED -- running self-test mode")
        print("  (splitting persistent runs into odd/even halves)\n")
    else:
        print(f"Episodic dir: {args.episodic_dir}\n")

    # Load data
    print("Loading persistent-condition runs...")
    runs_a = load_runs(args.persistent_dir, models)
    for m, rs in runs_a.items():
        print(f"  {m}: {len(rs)} runs")

    if self_test:
        runs_a_split, runs_b_split = self_test_split(runs_a)
        scores_a = extract_scores(runs_a_split)
        scores_b = extract_scores(runs_b_split)
        cell_scores_a = extract_cell_scores(runs_a_split)
        cell_scores_b = extract_cell_scores(runs_b_split)
    else:
        print("\nLoading episodic-condition runs...")
        runs_b = load_runs(args.episodic_dir, models)
        for m, rs in runs_b.items():
            print(f"  {m}: {len(rs)} runs")
        scores_a = extract_scores(runs_a)
        scores_b = extract_scores(runs_b)
        # Cell-level aggregation for interaction tests (fixes #3)
        print("\nAggregating to scenario-cell means (5 seeds x 2 objectives)...")
        cell_scores_a = extract_cell_scores(runs_a)
        cell_scores_b = extract_cell_scores(runs_b)
        for m in models:
            n_a = len(cell_scores_a.get(m, {}).get("total", []))
            n_b = len(cell_scores_b.get(m, {}).get("total", []))
            print(f"  {m}: {n_a} persistent cells, {n_b} episodic cells")

    # (i) Ranking comparison (run-level means are fine here)
    ranking = ranking_comparison(scores_a, scores_b, dim="total")

    # (ii) Effect sizes (Kruskal-Wallis on cell means)
    effect_sizes = effect_size_comparison(cell_scores_a, cell_scores_b)

    # (iii) Interactions per dimension (cell-level ART)
    interactions = {}
    for dim in DIMENSIONS:
        interactions[dim] = interaction_test(cell_scores_a, cell_scores_b, dim)

    # Print report
    print_report(ranking, effect_sizes, interactions, self_test=self_test)

    # Save JSON
    output = {
        "mode": "self_test" if self_test else "ablation",
        "models": models,
        "ranking_comparison": ranking,
        "effect_size_comparison": effect_sizes,
        "interaction_tests": interactions,
    }

    # Clean numpy types for JSON serialization
    def _clean(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        return obj

    output = _clean(output)

    out_path = args.output
    if out_path is None:
        # Default output path next to the script
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "episodic_ablation_results.json"
        )

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
