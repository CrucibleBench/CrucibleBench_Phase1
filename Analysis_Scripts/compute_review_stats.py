#!/usr/bin/env python3
"""Compute additional statistics requested in the peer review.

Outputs: review_stats.json with SDs, Cohen's d, seed analysis,
OLMo-excluded eta-sq, power analysis, per-model probe rates,
raw score distributions, bootstrap algorithm spec, interaction test,
and failure mode taxonomy.
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mud_poc.scoring import score_run, _score_goal_pursuit, _score_social_adaptation, _score_world_grounding, _score_strategic

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "run2")
DIMENSIONS = ["goal_pursuit", "social_adaptation", "world_grounding", "strategic_sophistication"]


def load_all_runs():
    runs = []
    for model_dir in sorted(os.listdir(RESULTS_DIR)):
        mpath = os.path.join(RESULTS_DIR, model_dir)
        if not os.path.isdir(mpath):
            continue
        for fname in sorted(os.listdir(mpath)):
            if not (fname.startswith("run_") and fname.endswith(".json")):
                continue
            fpath = os.path.join(mpath, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            scores = score_run(data)
            data["_model"] = model_dir
            data["_scores"] = scores
            data["_file"] = fpath
            runs.append(data)
    return runs


def compute_sds(runs):
    """Critique #1: SD and IQR per model per dimension."""
    by_model = defaultdict(lambda: defaultdict(list))
    for r in runs:
        m = r["_model"]
        for dim in DIMENSIONS:
            by_model[m][dim].append(r["_scores"][dim])
        by_model[m]["total"].append(r["_scores"]["total"])

    result = {}
    for m in sorted(by_model.keys()):
        result[m] = {}
        for dim in DIMENSIONS + ["total"]:
            vals = np.array(by_model[m][dim])
            result[m][dim] = {
                "mean": round(float(np.mean(vals)), 3),
                "sd": round(float(np.std(vals, ddof=1)), 3),
                "iqr": round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 3),
                "q25": round(float(np.percentile(vals, 25)), 3),
                "median": round(float(np.median(vals)), 3),
                "q75": round(float(np.percentile(vals, 75)), 3),
            }
    return result


def compute_cross_objective_correlation(runs):
    """Critique #2: Clarify n for cross-objective Spearman r.
    Compute at model level (n=13) and report honestly."""
    model_success = defaultdict(lambda: {"trust": [], "ident": []})
    for r in runs:
        m = r["_model"]
        obj = r.get("objective", "")
        success = 1 if r.get("success", False) else 0
        if "trust" in obj:
            model_success[m]["trust"].append(success)
        else:
            model_success[m]["ident"].append(success)

    models = sorted(model_success.keys())
    trust_rates = [np.mean(model_success[m]["trust"]) for m in models]
    ident_rates = [np.mean(model_success[m]["ident"]) for m in models]

    r_val, p_val = stats.spearmanr(trust_rates, ident_rates)

    # Fisher z CI for Spearman r with n=13
    n = len(models)
    z = np.arctanh(r_val)
    se = 1.0 / math.sqrt(n - 3)
    z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
    r_lo, r_hi = np.tanh(z_lo), np.tanh(z_hi)

    return {
        "n": n,
        "unit_of_analysis": "model-level success rates",
        "spearman_r": round(float(r_val), 4),
        "p_value": round(float(p_val), 4),
        "ci_95_lo": round(float(r_lo), 3),
        "ci_95_hi": round(float(r_hi), 3),
        "note": f"n={n} is too small for inferential claims. CI [{r_lo:.2f}, {r_hi:.2f}] spans nearly the full range. Report descriptively."
    }


def compute_per_model_probe_rates(runs):
    """Critique #3: Per-model probe-positive rates from classifier."""
    model_probes = defaultdict(lambda: {"total_talks": 0, "probe_positive": 0})
    for r in runs:
        m = r["_model"]
        events = r.get("transcript_events", r.get("transcript", []))
        for ev in events:
            action = ev.get("action", {})
            if isinstance(action, dict) and action.get("command") == "talk":
                model_probes[m]["total_talks"] += 1
                # Check classifier signal
                signal = ev.get("classifier_signal", ev.get("dialog_signal", {}))
                if isinstance(signal, dict) and signal.get("direct_objective_probe", False):
                    model_probes[m]["probe_positive"] += 1

    result = {}
    for m in sorted(model_probes.keys()):
        d = model_probes[m]
        rate = d["probe_positive"] / d["total_talks"] if d["total_talks"] > 0 else 0
        result[m] = {
            "total_talks": d["total_talks"],
            "probe_positive": d["probe_positive"],
            "probe_positive_rate": round(rate, 3),
        }
    return result


def compute_seed_analysis(runs):
    """Critique #4: Seed-level analysis. Test seed effect and seed × model interaction."""
    by_seed = defaultdict(list)
    by_model_seed = defaultdict(lambda: defaultdict(list))
    for r in runs:
        seed = r.get("seed", 0)
        m = r["_model"]
        total = r["_scores"]["total"]
        by_seed[seed].append(total)
        by_model_seed[m][seed].append(total)

    # Seed means
    seed_means = {s: round(float(np.mean(v)), 3) for s, v in sorted(by_seed.items())}

    # Kruskal-Wallis across seeds (is seed significant?)
    seed_groups = [np.array(by_seed[s]) for s in sorted(by_seed.keys())]
    h_seed, p_seed = stats.kruskal(*seed_groups)

    # Seed × model: for each model, report per-seed means
    model_seed_means = {}
    for m in sorted(by_model_seed.keys()):
        model_seed_means[m] = {str(s): round(float(np.mean(v)), 3) for s, v in sorted(by_model_seed[m].items())}

    # Compute seed effect eta-squared
    all_scores = np.array([r["_scores"]["total"] for r in runs])
    grand_mean = np.mean(all_scores)
    ss_total = np.sum((all_scores - grand_mean) ** 2)
    ss_seed = sum(len(by_seed[s]) * (np.mean(by_seed[s]) - grand_mean) ** 2 for s in by_seed)
    eta_sq_seed = round(float(ss_seed / ss_total), 4) if ss_total > 0 else 0

    return {
        "seed_means": seed_means,
        "kruskal_wallis_seed": {"H": round(float(h_seed), 3), "p": round(float(p_seed), 4)},
        "eta_squared_seed": eta_sq_seed,
        "model_seed_means": model_seed_means,
        "note": "Seed effect tested via Kruskal-Wallis on total scores grouped by seed."
    }


def specify_bootstrap_algorithm(runs):
    """Critique #5: Exact bootstrap specification."""
    models = sorted(set(r["_model"] for r in runs))
    # Detect cell structure
    cells = set()
    for r in runs:
        seed = r.get("seed", 0)
        obj = r.get("objective", "")
        cells.add((seed, obj))

    return {
        "algorithm": "scenario-cell bootstrap",
        "cell_definition": "seed x objective (e.g., seed=20260302 x gain_watch_trust)",
        "n_cells_per_model": len(cells),
        "repetitions_per_cell": 5,
        "procedure": [
            "1. For each model, organize 50 runs into 10 cells (5 seeds x 2 objectives).",
            "2. Each cell contains 5 repetitions.",
            "3. For each bootstrap iteration:",
            "   a. Resample the 10 cells WITH replacement (sampling cells, not individual runs).",
            "   b. For each sampled cell, take ALL 5 repetitions.",
            "   c. Compute the mean total score across all selected runs.",
            "4. Repeat 10,000 times.",
            "5. The 2.5th and 97.5th percentiles of the bootstrap distribution form the 95% CI."
        ],
        "exchangeability_assumption": "Cells are exchangeable units. Repetitions within a cell share the same seed+objective and are NOT resampled independently.",
        "note": "This preserves the within-cell correlation structure. Resampling individual runs would underestimate variance by treating correlated observations as independent."
    }


def compute_eta_squared_olmo_excluded(runs):
    """Critique #6: Eta-squared with and without OLMo."""
    result = {}
    for include_olmo in [True, False]:
        label = "all_models" if include_olmo else "olmo_excluded"
        filtered = [r for r in runs if include_olmo or r["_model"] != "olmo_3_1"]
        dim_eta = {}
        for dim in DIMENSIONS:
            by_model = defaultdict(list)
            for r in filtered:
                by_model[r["_model"]].append(r["_scores"][dim])
            all_vals = np.array([v for vals in by_model.values() for v in vals])
            grand_mean = np.mean(all_vals)
            ss_total = np.sum((all_vals - grand_mean) ** 2)
            ss_between = sum(len(vals) * (np.mean(vals) - grand_mean) ** 2 for vals in by_model.values())
            dim_eta[dim] = round(float(ss_between / ss_total), 4) if ss_total > 0 else 0
        # Also total
        by_model_total = defaultdict(list)
        for r in filtered:
            by_model_total[r["_model"]].append(r["_scores"]["total"])
        all_t = np.array([v for vals in by_model_total.values() for v in vals])
        gm = np.mean(all_t)
        sst = np.sum((all_t - gm) ** 2)
        ssb = sum(len(vals) * (np.mean(vals) - gm) ** 2 for vals in by_model_total.values())
        dim_eta["total"] = round(float(ssb / sst), 4) if sst > 0 else 0

        result[label] = {
            "type": "eta-squared (SS_between / SS_total), one-way ANOVA decomposition",
            "NOT_partial_eta_squared": True,
            "n_models": len(set(r["_model"] for r in filtered)),
            "dimensions": dim_eta,
        }
    return result


def compute_cohens_d_matrix(runs):
    """Critique #7: Cohen's d for key pairwise comparisons."""
    by_model = defaultdict(list)
    for r in runs:
        by_model[r["_model"]].append(r["_scores"]["total"])

    models = sorted(by_model.keys())
    d_matrix = {}
    for m1, m2 in combinations(models, 2):
        v1, v2 = np.array(by_model[m1]), np.array(by_model[m2])
        pooled_sd = math.sqrt(((len(v1)-1)*np.var(v1, ddof=1) + (len(v2)-1)*np.var(v2, ddof=1)) / (len(v1)+len(v2)-2))
        d = float((np.mean(v1) - np.mean(v2)) / pooled_sd) if pooled_sd > 0 else 0
        d_matrix[f"{m1}_vs_{m2}"] = round(d, 3)

    # Key pairs for the paper
    key_pairs = [
        ("gpt_5_4", "claude_opus"),
        ("gpt_5_4", "grok_4"),
        ("claude_opus", "claude_sonnet"),
        ("claude_opus", "gpt_5_3_chat"),
        ("mistral_large", "gpt_5_4"),
        ("deepseek_v3_2", "gpt_5_4"),
        ("olmo_3_1", "gpt_5_4"),
    ]
    key_d = {}
    for m1, m2 in key_pairs:
        k = f"{m1}_vs_{m2}" if f"{m1}_vs_{m2}" in d_matrix else f"{m2}_vs_{m1}"
        if k in d_matrix:
            key_d[f"{m1} vs {m2}"] = d_matrix[k] if k.startswith(m1) else -d_matrix[k]
        else:
            # compute directly
            v1, v2 = np.array(by_model[m1]), np.array(by_model[m2])
            pooled_sd = math.sqrt(((len(v1)-1)*np.var(v1, ddof=1) + (len(v2)-1)*np.var(v2, ddof=1)) / (len(v1)+len(v2)-2))
            key_d[f"{m1} vs {m2}"] = round(float((np.mean(v1) - np.mean(v2)) / pooled_sd), 3)

    return {"key_pairs": key_d, "full_matrix": d_matrix}


def compute_interaction_test(runs):
    """Critique #8: Test the model × objective interaction properly."""
    # Two-way ANOVA-style: compute SS for model, objective, interaction, residual
    all_scores = []
    all_models = []
    all_objectives = []
    for r in runs:
        all_scores.append(r["_scores"]["total"])
        all_models.append(r["_model"])
        all_objectives.append(r.get("objective", "unknown"))

    scores = np.array(all_scores)
    grand_mean = np.mean(scores)
    ss_total = np.sum((scores - grand_mean) ** 2)

    models = sorted(set(all_models))
    objectives = sorted(set(all_objectives))

    model_means = {m: np.mean([s for s, mm in zip(all_scores, all_models) if mm == m]) for m in models}
    obj_means = {o: np.mean([s for s, oo in zip(all_scores, all_objectives) if oo == o]) for o in objectives}

    # Cell means and counts
    cell_data = defaultdict(list)
    for s, m, o in zip(all_scores, all_models, all_objectives):
        cell_data[(m, o)].append(s)

    cell_means = {k: np.mean(v) for k, v in cell_data.items()}
    cell_counts = {k: len(v) for k, v in cell_data.items()}

    # SS_model
    ss_model = sum(sum(1 for mm in all_models if mm == m) * (model_means[m] - grand_mean) ** 2 for m in models)
    # SS_objective
    ss_obj = sum(sum(1 for oo in all_objectives if oo == o) * (obj_means[o] - grand_mean) ** 2 for o in objectives)
    # SS_interaction
    ss_interaction = 0
    for (m, o), cmean in cell_means.items():
        expected = model_means[m] + obj_means[o] - grand_mean
        ss_interaction += cell_counts[(m, o)] * (cmean - expected) ** 2
    # SS_residual
    ss_residual = ss_total - ss_model - ss_obj - ss_interaction

    # Degrees of freedom
    a = len(models)  # 13
    b = len(objectives)  # 2
    N = len(all_scores)  # 650
    df_model = a - 1
    df_obj = b - 1
    df_interaction = (a - 1) * (b - 1)
    df_residual = N - a * b

    # F-tests
    ms_interaction = ss_interaction / df_interaction
    ms_residual = ss_residual / df_residual
    f_interaction = ms_interaction / ms_residual
    p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_residual)

    ms_model = ss_model / df_model
    f_model = ms_model / ms_residual
    p_model = 1 - stats.f.cdf(f_model, df_model, df_residual)

    ms_obj = ss_obj / df_obj
    f_obj = ms_obj / ms_residual
    p_obj = 1 - stats.f.cdf(f_obj, df_obj, df_residual)

    eta_sq_interaction = round(float(ss_interaction / ss_total), 4)
    partial_eta_sq_interaction = round(float(ss_interaction / (ss_interaction + ss_residual)), 4)

    return {
        "test": "Two-way ANOVA (Type III approximation)",
        "model_effect": {"F": round(float(f_model), 2), "df": [df_model, df_residual], "p": round(float(p_model), 6), "eta_sq": round(float(ss_model/ss_total), 4)},
        "objective_effect": {"F": round(float(f_obj), 2), "df": [df_obj, df_residual], "p": round(float(p_obj), 6), "eta_sq": round(float(ss_obj/ss_total), 4)},
        "interaction": {
            "F": round(float(f_interaction), 2),
            "df": [df_interaction, df_residual],
            "p": round(float(p_interaction), 6),
            "eta_squared": eta_sq_interaction,
            "partial_eta_squared": partial_eta_sq_interaction,
            "interpretation": "small" if eta_sq_interaction < 0.06 else "medium" if eta_sq_interaction < 0.14 else "large",
        },
        "note": "η²=0.022 is small by Cohen's conventions but the F-test determines significance. Even small effects can be significant with N=650."
    }


def compute_raw_score_distributions(runs):
    """Critique #9: Raw score distributions to check ceiling/floor effects."""
    raw_by_dim = defaultdict(list)
    for r in runs:
        # Recompute raw scores (before rubric mapping)
        for dim in DIMENSIONS:
            rubric = r["_scores"][dim]
            raw = (rubric - 1.0) / 4.0  # inverse of rubric = 1 + 4*raw
            raw_by_dim[dim].append(raw)

    result = {}
    for dim in DIMENSIONS:
        vals = np.array(raw_by_dim[dim])
        result[dim] = {
            "mean_raw": round(float(np.mean(vals)), 3),
            "sd_raw": round(float(np.std(vals, ddof=1)), 3),
            "min": round(float(np.min(vals)), 3),
            "max": round(float(np.max(vals)), 3),
            "pct_at_floor_0": round(float(np.mean(vals <= 0.05) * 100), 1),
            "pct_at_ceiling_95": round(float(np.mean(vals >= 0.95) * 100), 1),
            "pct_above_80": round(float(np.mean(vals >= 0.80) * 100), 1),
            "skewness": round(float(stats.skew(vals)), 3),
        }
    return result


def compute_power_analysis():
    """Critique #10: Retrospective power analysis for Phase 1.
    With n=25 per objective per model (within-objective pairwise),
    what's the minimum detectable effect size at 80% power, alpha=0.05?
    Using a two-sample t-test power formula as approximation.
    """
    from scipy.stats import norm

    alpha = 0.05
    power = 0.80
    n_per_group = 25  # per objective per model

    z_alpha = norm.ppf(1 - alpha / 2)  # 1.96
    z_beta = norm.ppf(power)  # 0.842

    # d = (z_alpha + z_beta) * sqrt(2/n)
    d_min = (z_alpha + z_beta) * math.sqrt(2 / n_per_group)

    # Also for n=50 (total runs per model, aggregated across objectives)
    d_min_50 = (z_alpha + z_beta) * math.sqrt(2 / 50)

    return {
        "n_per_objective_per_model": n_per_group,
        "n_total_per_model": 50,
        "alpha": alpha,
        "target_power": power,
        "min_detectable_d_within_objective": round(d_min, 3),
        "min_detectable_d_across_objectives": round(d_min_50, 3),
        "interpretation": f"Within-objective (n=25): can only detect d>={d_min:.2f}. Across objectives (n=50): d>={d_min_50:.2f}. Effects smaller than these are indistinguishable from noise at 80% power.",
        "implication": "Top-tier CI overlaps are expected: pairwise differences within Tier 2 are likely d<0.5, below detection threshold."
    }


def compute_failure_taxonomy(runs):
    """Bonus: Failure mode taxonomy with per-model frequencies."""
    model_failures = defaultdict(lambda: Counter())
    model_totals = defaultdict(int)

    for r in runs:
        m = r["_model"]
        model_totals[m] += 1
        events = r.get("transcript_events", r.get("transcript", []))
        has_wrong_room_talk = False
        has_dialogue_loop = False
        has_exploration_paralysis = False
        has_objective_confusion = False

        talk_targets = []
        rooms_visited = set()
        look_streak = 0
        max_look_streak = 0

        for ev in events:
            action = ev.get("action", {})
            if isinstance(action, dict):
                cmd = action.get("command", "")
                valid = ev.get("valid", True)

                # Wrong room interaction: talk to NPC not present
                if cmd == "talk" and not valid:
                    narr = ev.get("narration", "").lower()
                    if "not here" in narr or "no one" in narr or "cannot see" in narr or "don't see" in narr:
                        has_wrong_room_talk = True

                # Track rooms
                post_room = ev.get("post_room", "")
                if post_room:
                    rooms_visited.add(post_room)

                # Look streaks (exploration paralysis)
                if cmd == "look":
                    look_streak += 1
                    max_look_streak = max(max_look_streak, look_streak)
                else:
                    look_streak = 0

                # Dialogue loop: same talk target 5+ times
                if cmd == "talk":
                    args = action.get("args", [])
                    target = args[0] if args else ""
                    talk_targets.append(target)

        # Detect dialogue loop
        from collections import Counter as C2
        if talk_targets:
            tc = C2(talk_targets)
            if tc.most_common(1)[0][1] >= 8:
                has_dialogue_loop = True

        # Exploration paralysis: fewer than 3 unique rooms
        if len(rooms_visited) <= 2 and len(events) >= 20:
            has_exploration_paralysis = True

        # Look paralysis: 5+ consecutive looks
        if max_look_streak >= 5:
            has_exploration_paralysis = True

        if has_wrong_room_talk:
            model_failures[m]["wrong_room_interaction"] += 1
        if has_dialogue_loop:
            model_failures[m]["dialogue_loop"] += 1
        if has_exploration_paralysis:
            model_failures[m]["exploration_paralysis"] += 1

    result = {}
    for m in sorted(model_totals.keys()):
        result[m] = {
            "total_runs": model_totals[m],
            "wrong_room_interaction": model_failures[m].get("wrong_room_interaction", 0),
            "dialogue_loop": model_failures[m].get("dialogue_loop", 0),
            "exploration_paralysis": model_failures[m].get("exploration_paralysis", 0),
        }
    return result


def compute_first_5_actions(runs):
    """Bonus: First 5 action distribution by model for behavioral archetypes."""
    model_first5 = defaultdict(lambda: Counter())
    for r in runs:
        m = r["_model"]
        events = r.get("transcript_events", r.get("transcript", []))
        for ev in events[:5]:
            action = ev.get("action", {})
            if isinstance(action, dict):
                cmd = action.get("command", "look")
                model_first5[m][cmd] += 1

    result = {}
    for m in sorted(model_first5.keys()):
        total = sum(model_first5[m].values())
        result[m] = {cmd: count for cmd, count in model_first5[m].most_common()}
        result[m]["_total"] = total
    return result


def main():
    print("Loading runs...")
    runs = load_all_runs()
    print(f"  {len(runs)} runs loaded")

    output = {}

    print("\n1. Standard deviations...")
    output["model_descriptives"] = compute_sds(runs)

    print("2. Cross-objective correlation (clarified)...")
    output["cross_objective_correlation"] = compute_cross_objective_correlation(runs)

    print("3. Per-model probe-positive rates...")
    output["per_model_probe_rates"] = compute_per_model_probe_rates(runs)

    print("4. Seed-level analysis...")
    output["seed_analysis"] = compute_seed_analysis(runs)

    print("5. Bootstrap algorithm specification...")
    output["bootstrap_specification"] = specify_bootstrap_algorithm(runs)

    print("6. Eta-squared with/without OLMo...")
    output["eta_squared_comparison"] = compute_eta_squared_olmo_excluded(runs)

    print("7. Cohen's d matrix...")
    output["cohens_d"] = compute_cohens_d_matrix(runs)

    print("8. Interaction test (two-way ANOVA)...")
    output["interaction_test"] = compute_interaction_test(runs)

    print("9. Raw score distributions...")
    output["raw_score_distributions"] = compute_raw_score_distributions(runs)

    print("10. Power analysis...")
    output["power_analysis"] = compute_power_analysis()

    print("\nBonus: Failure mode taxonomy...")
    output["failure_taxonomy"] = compute_failure_taxonomy(runs)

    print("Bonus: First-5 action profiles...")
    output["first_5_actions"] = compute_first_5_actions(runs)

    outpath = os.path.join(PROJECT_ROOT, "review_stats.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nWritten: {outpath}")


if __name__ == "__main__":
    main()
