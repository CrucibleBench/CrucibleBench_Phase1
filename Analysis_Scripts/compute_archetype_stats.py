"""
Opening Action Archetype Analysis for CrucibleBench (Section 9.6)

Classifies models into opening archetypes based on first-5-action distributions,
computes group means, and runs Kruskal-Wallis and Mann-Whitney tests.

Reads:
  - review_stats.json  (first_5_actions from compute_review_stats.py)
  - scores.csv         (per-run scoring data)

Outputs statistics for Section 9.6 of the white paper.

Usage:
    python compute_archetype_stats.py
"""

import json
import pandas as pd
import numpy as np
from scipy import stats


# ── Configuration ────────────────────────────────────────────────────────────

# Archetype assignments based on first-5-action distributions.
# Paper criteria (Sec 9.6):
#   navigate-first: "go" dominates at ~100/250
#   examine-first:  "examine" rivals or exceeds "go"
#   observe-first:  "look" dominates
#
# Qwen 3.5: 103 examine vs 100 go → examine-first (paper: "leads with 103/250 examine")
# Grok 4:   97 examine vs 101 go  → examine-first (paper: "97/250 examine vs 101/250 go")
# OLMo 3.1: 80 look vs 50 go     → observe-first (paper: "look dominates at 80/250")
# All others: go >= 99, examine < go → navigate-first

EXAMINE_FIRST = ["qwen_3_5", "grok_4"]
OBSERVE_FIRST = ["olmo_3_1"]
# All remaining models are navigate-first


def classify_archetypes(first_5_actions: dict) -> dict:
    """Assign each model to an archetype. Returns {model: archetype}."""
    archetypes = {}
    for model in first_5_actions:
        if model in EXAMINE_FIRST:
            archetypes[model] = "examine-first"
        elif model in OBSERVE_FIRST:
            archetypes[model] = "observe-first"
        else:
            archetypes[model] = "navigate-first"
    return archetypes


def main():
    # ── Load data ────────────────────────────────────────────────────────
    with open("review_stats.json") as f:
        review = json.load(f)
    scores = pd.read_csv("scores.csv")

    first_5 = review["first_5_actions"]
    archetypes = classify_archetypes(first_5)

    # ── Print first-5-action table ───────────────────────────────────────
    print("=" * 75)
    print("FIRST-5 ACTION COUNTS BY MODEL")
    print("=" * 75)
    print(f"{'Model':<20} {'go':>5} {'examine':>8} {'look':>5} {'talk':>5} {'take':>5}  Archetype")
    print("-" * 75)
    for m in sorted(first_5.keys()):
        a = first_5[m]
        arch = archetypes[m]
        print(
            f"{m:<20} {a.get('go',0):>5} {a.get('examine',0):>8} "
            f"{a.get('look',0):>5} {a.get('talk',0):>5} {a.get('take',0):>5}  {arch}"
        )

    # ── Build groups ─────────────────────────────────────────────────────
    groups = {}
    for m, arch in archetypes.items():
        groups.setdefault(arch, []).append(m)

    group_order = ["navigate-first", "examine-first", "observe-first"]
    group_data = {}
    for g in group_order:
        models = groups.get(g, [])
        data = scores[scores["model"].isin(models)]["total"].values
        group_data[g] = data

    # ── Group summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("GROUP SUMMARY")
    print(f"{'=' * 75}")
    print(f"{'Group':<20} {'Models':>7} {'Runs':>6} {'Mean':>8} {'SD':>8}")
    print("-" * 55)
    for g in group_order:
        models = groups.get(g, [])
        d = group_data[g]
        print(f"{g:<20} {len(models):>7} {len(d):>6} {d.mean():>8.4f} {d.std():>8.4f}")

    # ── Kruskal-Wallis (3 groups, all runs) ──────────────────────────────
    print(f"\n{'=' * 75}")
    print("KRUSKAL-WALLIS TEST (3 groups, run-level)")
    print(f"{'=' * 75}")
    kw_arrays = [group_data[g] for g in group_order]
    H, p = stats.kruskal(*kw_arrays)
    N = sum(len(d) for d in kw_arrays)
    k = len(kw_arrays)
    eta_sq = (H - k + 1) / (N - k)
    print(f"H = {H:.2f}, p = {p:.2e}, eta_sq = {eta_sq:.4f}")
    print(f"N = {N}, k = {k}")

    # ── Mann-Whitney (navigate vs examine, excluding OLMo) ───────────────
    print(f"\n{'=' * 75}")
    print("MANN-WHITNEY U (navigate-first vs examine-first, excluding OLMo)")
    print(f"{'=' * 75}")
    nav = group_data["navigate-first"]
    exam = group_data["examine-first"]
    U, p_mw = stats.mannwhitneyu(nav, exam, alternative="two-sided")
    n1, n2 = len(nav), len(exam)
    # Effect size r = |Z| / sqrt(N)
    z = (U - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    r = abs(z) / np.sqrt(n1 + n2)
    print(f"U = {U:.1f}, p = {p_mw:.6f}, Z = {z:.2f}, r = {r:.4f}")
    print(f"Navigate: n={n1}, mean={nav.mean():.4f}")
    print(f"Examine:  n={n2}, mean={exam.mean():.4f}")

    # ── Per-model breakdown within each group ────────────────────────────
    print(f"\n{'=' * 75}")
    print("PER-MODEL MEANS WITHIN GROUPS")
    print(f"{'=' * 75}")
    model_means = scores.groupby("model")["total"].mean()
    for g in group_order:
        models = sorted(groups.get(g, []))
        print(f"\n  {g}:")
        for m in models:
            print(f"    {m:<20} {model_means[m]:.4f}")

if __name__ == "__main__":
    main()
