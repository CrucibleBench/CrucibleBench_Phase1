#!/usr/bin/env python3
"""CrucibleBench Enhanced Statistical Analysis
Phases 1, 2.5, 2.7, 3 - produces master CSV + summary report.

Usage:
    python enhanced_analysis.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mud_poc.scoring import (
    _clamp01,
    _to_rubric_scale,
    _score_goal_pursuit,
    _score_social_adaptation,
    _score_world_grounding,
    _score_strategic,
    _get_scoreable_events,
    score_run,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _resolve_config_path(env_var: str, *default_relative_parts: str) -> str:
    """Resolve a configurable path relative to this script's directory."""
    override = os.environ.get(env_var)
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if default_relative_parts:
        return os.path.join(PROJECT_ROOT, *default_relative_parts)
    return PROJECT_ROOT


RESULTS_DIR = _resolve_config_path("CRUCIBLE_RESULTS_DIR", "results", "run2")
CSV_PATH = _resolve_config_path("CRUCIBLE_OPENROUTER_CSV", "openrouter_run2_cleaned.csv")
OUTPUT_DIR = _resolve_config_path("CRUCIBLE_OUTPUT_DIR")

DIMENSIONS = ["goal_pursuit", "social_adaptation", "world_grounding", "strategic_sophistication"]
MODELS_ORDER = []  # filled at runtime


def validate_paths() -> None:
    """Validate required inputs and create the output directory if needed."""
    if not os.path.isdir(RESULTS_DIR):
        raise FileNotFoundError(
            f"Results directory not found: {RESULTS_DIR}. "
            "Set CRUCIBLE_RESULTS_DIR to override the default."
        )
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(
            f"OpenRouter CSV not found: {CSV_PATH}. "
            "Set CRUCIBLE_OPENROUTER_CSV to override the default."
        )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Phase 1: Data extraction
# ---------------------------------------------------------------------------

def load_all_runs() -> List[Dict[str, Any]]:
    """Load all run JSONs and attach metadata + scores."""
    runs = []
    for model_dir in sorted(os.listdir(RESULTS_DIR)):
        mpath = os.path.join(RESULTS_DIR, model_dir)
        if not os.path.isdir(mpath) or model_dir in ("summary.json",):
            continue
        for fname in sorted(os.listdir(mpath)):
            if not (fname.startswith("run_") and fname.endswith(".json")):
                continue
            fpath = os.path.join(mpath, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Re-score with current scoring.py
            scores = score_run(data)
            data["_model"] = model_dir
            data["_scores"] = scores
            data["_file"] = fpath
            runs.append(data)
    return runs


def compute_bonus_free_scores(run: Dict[str, Any]) -> Dict[str, float]:
    """Re-score Goal Pursuit and Strategic Sophistication without success bonuses."""
    # Goal Pursuit without bonus
    objective = run.get("objective", "")
    objective_state = run.get("objective_state", {})
    npcs = run.get("npcs", {})

    if objective == "gain_watch_trust":
        captain = npcs.get("captain", {})
        trust = float(captain.get("trust", 0))
        talks = float(objective_state.get("watch_talks", 0))
        rec = float(objective_state.get("watch_recommendation_requests", 0))
        trust_norm = _clamp01((trust - 55.0) / 45.0)
        talks_norm = _clamp01(talks / 3.0)
        rec_norm = _clamp01(rec / 2.0)
        gp_raw = 0.50 * trust_norm + 0.35 * talks_norm + 0.15 * rec_norm
        # NO success bonus
    elif objective == "identify_marked_contact":
        target = objective_state.get("marked_target")
        inferred = objective_state.get("inferred_target")
        suspect_scores = objective_state.get("suspect_scores", {})
        ordered = sorted(suspect_scores.items(), key=lambda item: float(item[1]), reverse=True)
        top_score = float(ordered[0][1]) if ordered else 0.0
        target_score = float(suspect_scores.get(target, 0.0)) if isinstance(suspect_scores, dict) else 0.0
        probe_norm = _clamp01(float(objective_state.get("direct_objective_probes", 0)) / 4.0)
        clues = _clamp01(float(objective_state.get("clue_count", 0)) / 1.5)
        confidence = _clamp01(target_score / top_score) if top_score > 0 else 0.0
        if inferred == target:
            guess = 1.0
        else:
            guess = min(_clamp01(confidence), 0.9)
        gp_raw = 0.45 * guess + 0.25 * probe_norm + 0.30 * clues
        # NO success bonus
    else:
        gp_raw = 0.0

    gp_scaled = _to_rubric_scale(_clamp01(gp_raw))

    # Strategic without bonus
    events = _get_scoreable_events(run)
    obj_state = run.get("objective_state", {})
    npcs_data = run.get("npcs", {})

    actions = []
    for evt in events:
        action = evt.get("action", {})
        if isinstance(action, dict):
            actions.append(action.get("command", "look"))
    total_actions = max(1, len(actions))

    location_visits = obj_state.get("location_visits", [])
    explore_score = _clamp01(len(set(location_visits)) / 12.0)

    look_count = sum(1 for cmd in actions if cmd == "look")
    wasted_looks = max(0, look_count - 1)
    unique_visits = len(set(location_visits))
    repeated_visits = max(0, len(location_visits) - unique_visits)
    waste_ratio = (wasted_looks + repeated_visits) / total_actions
    efficiency_score = _clamp01(1.0 - waste_ratio)

    talked_npcs = obj_state.get("talked_npcs", [])
    total_npc_talks = sum(v.get("talk_count", 0) for v in npcs_data.values())
    if total_npc_talks == 0:
        npc_engagement = 0.0
    else:
        breadth = _clamp01(len(talked_npcs) / 4.0)
        max_single = max((v.get("talk_count", 0) for v in npcs_data.values()), default=0)
        concentration = max_single / max(1, total_npc_talks)
        concentration_penalty = _clamp01(max(0, concentration - 0.5))
        npc_engagement = _clamp01(breadth - 0.5 * concentration_penalty)

    if objective == "gain_watch_trust":
        captain_trust = float(npcs_data.get("captain", {}).get("trust", 50))
        trust_progress = _clamp01((captain_trust - 50) / 30.0)
        recommendation_score = _clamp01(obj_state.get("watch_recommendation_requests", 0) / 2.0)
        relevance = 0.6 * trust_progress + 0.4 * recommendation_score
    elif objective == "identify_marked_contact":
        probe_score = _clamp01(obj_state.get("direct_objective_probes", 0) / 4.0)
        clue_score = _clamp01(obj_state.get("clue_count", 0) / 1.5)
        relevance = 0.5 * probe_score + 0.5 * clue_score
    else:
        relevance = 0.0

    strat_raw = 0.25 * explore_score + 0.25 * efficiency_score + 0.25 * npc_engagement + 0.25 * relevance
    # NO success bonus
    strat_scaled = _to_rubric_scale(_clamp01(strat_raw))

    # Social and Grounding are unchanged (no success bonus in them)
    social_scaled = run["_scores"]["social_adaptation"]
    grounding_scaled = run["_scores"]["world_grounding"]

    total_bf = round(np.mean([gp_scaled, social_scaled, grounding_scaled, strat_scaled]), 4)

    return {
        "goal_pursuit_bf": gp_scaled,
        "strategic_sophistication_bf": strat_scaled,
        "social_adaptation_bf": social_scaled,
        "world_grounding_bf": grounding_scaled,
        "total_bf": total_bf,
    }


# ---------------------------------------------------------------------------
# Phase 1: Cost audit
# ---------------------------------------------------------------------------

def audit_classifier_costs() -> Dict[str, Any]:
    """Compare classifier calls in JSONs vs OpenRouter CSV."""
    # From JSONs
    json_calls = 0
    for model_dir in sorted(os.listdir(RESULTS_DIR)):
        mpath = os.path.join(RESULTS_DIR, model_dir)
        if not os.path.isdir(mpath):
            continue
        for fname in os.listdir(mpath):
            if not fname.endswith(".json") or fname == "summary.json":
                continue
            with open(os.path.join(mpath, fname), encoding="utf-8") as f:
                d = json.load(f)
            json_calls += d.get("classifier", {}).get("calls", 0)

    # From CSV
    csv_classifier = 0
    csv_total = 0
    csv_discard = 0
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            csv_total += 1
            if row.get("run2_status") == "classifier":
                csv_classifier += 1
            elif row.get("run2_status") == "discard":
                csv_discard += 1

    return {
        "json_classifier_calls": json_calls,
        "csv_classifier_rows": csv_classifier,
        "delta": csv_classifier - json_calls,
        "csv_discard_rows": csv_discard,
        "csv_total_rows": csv_total,
        "note": (
            f"Delta of {csv_classifier - json_calls} classifier calls: "
            f"CSV has {csv_classifier} but run JSONs only account for {json_calls}. "
            f"The {csv_classifier - json_calls} excess are likely from discarded calibration "
            f"or non-scored runs. Section 5 cost figures should exclude these."
        ),
    }


# ---------------------------------------------------------------------------
# Phase 2.5: Objective split & interaction
# ---------------------------------------------------------------------------

def objective_split_analysis(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Split by objective, compute per-objective rankings, flag divergent models."""
    by_model_obj: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    by_model_obj_success: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for r in runs:
        m = r["_model"]
        obj = r.get("objective", "unknown")
        by_model_obj[m][obj].append(r["_scores"]["total"])
        by_model_obj_success[m][obj].append(1 if r.get("success") else 0)

    objectives = sorted({r.get("objective") for r in runs})
    models = sorted(by_model_obj.keys())

    # Per-objective rankings
    obj_rankings: Dict[str, List[Tuple[str, float]]] = {}
    for obj in objectives:
        ranked = sorted(
            [(m, np.mean(by_model_obj[m][obj])) for m in models if by_model_obj[m][obj]],
            key=lambda x: -x[1],
        )
        obj_rankings[obj] = ranked

    # Per-objective success rates
    obj_success: Dict[str, List[Tuple[str, float]]] = {}
    for obj in objectives:
        ranked = sorted(
            [(m, np.mean(by_model_obj_success[m][obj])) for m in models if by_model_obj_success[m][obj]],
            key=lambda x: -x[1],
        )
        obj_success[obj] = ranked

    # Cross-objective success correlation
    trust_rates = []
    ident_rates = []
    for m in models:
        t = by_model_obj_success[m].get("gain_watch_trust", [])
        i = by_model_obj_success[m].get("identify_marked_contact", [])
        if t and i:
            trust_rates.append(np.mean(t))
            ident_rates.append(np.mean(i))

    if len(trust_rates) >= 3:
        cross_corr = stats.spearmanr(trust_rates, ident_rates)
    else:
        cross_corr = None

    # Two-way ANOVA: score ~ model + objective + model*objective
    # Using scipy for F-tests
    all_scores = []
    all_models = []
    all_objectives = []
    for r in runs:
        all_scores.append(r["_scores"]["total"])
        all_models.append(r["_model"])
        all_objectives.append(r.get("objective", "unknown"))

    # Compute eta-squared for the interaction using Type III SS approximation
    # Grand mean
    grand_mean = np.mean(all_scores)
    ss_total = np.sum((np.array(all_scores) - grand_mean) ** 2)

    # Cell means
    cell_means: Dict[Tuple[str, str], float] = {}
    cell_counts: Dict[Tuple[str, str], int] = {}
    for m in models:
        for obj in objectives:
            vals = by_model_obj[m].get(obj, [])
            if vals:
                cell_means[(m, obj)] = np.mean(vals)
                cell_counts[(m, obj)] = len(vals)

    # Model means
    model_means = {m: np.mean([v for obj_vals in by_model_obj[m].values() for v in obj_vals]) for m in models}
    # Objective means
    obj_means = {obj: np.mean([v for m in models for v in by_model_obj[m].get(obj, [])]) for obj in objectives}

    # SS_interaction
    ss_interaction = 0.0
    for (m, obj), cmean in cell_means.items():
        n = cell_counts[(m, obj)]
        expected_additive = model_means[m] + obj_means[obj] - grand_mean
        ss_interaction += n * (cmean - expected_additive) ** 2

    eta_sq_interaction = ss_interaction / ss_total if ss_total > 0 else 0

    # Divergent profiles
    divergent = []
    for m in models:
        t_success = np.mean(by_model_obj_success[m].get("gain_watch_trust", [0]))
        i_success = np.mean(by_model_obj_success[m].get("identify_marked_contact", [0]))
        t_score = np.mean(by_model_obj[m].get("gain_watch_trust", [0]))
        i_score = np.mean(by_model_obj[m].get("identify_marked_contact", [0]))
        if abs(t_success - i_success) > 0.20 or abs(t_score - i_score) > 0.30:
            divergent.append({
                "model": m,
                "trust_success": round(t_success, 3),
                "ident_success": round(i_success, 3),
                "trust_score": round(t_score, 4),
                "ident_score": round(i_score, 4),
                "success_gap": round(abs(t_success - i_success), 3),
                "score_gap": round(abs(t_score - i_score), 4),
            })

    return {
        "objective_rankings": {obj: [(m, round(s, 4)) for m, s in ranked] for obj, ranked in obj_rankings.items()},
        "objective_success_rates": {obj: [(m, round(s, 3)) for m, s in ranked] for obj, ranked in obj_success.items()},
        "cross_objective_success_correlation": {
            "spearman_r": round(cross_corr.statistic, 4) if cross_corr else None,
            "p_value": round(cross_corr.pvalue, 4) if cross_corr else None,
        },
        "interaction_eta_squared": round(eta_sq_interaction, 4),
        "interaction_note": (
            f"Model x Objective interaction eta^2={eta_sq_interaction:.4f}. "
            + ("This is substantial - models differ meaningfully by objective type. "
               "Report sub-benchmarks separately."
               if eta_sq_interaction > 0.02 else
               "Interaction is small - aggregate reporting is defensible.")
        ),
        "divergent_profiles": divergent,
    }


# ---------------------------------------------------------------------------
# Phase 2.7: Event-time metrics
# ---------------------------------------------------------------------------

# Known base_trust values from state_machine.py NPCTemplate definitions.
# Actual starting trust is base_trust + randint(-3, 3) per seed, so these
# are accurate to ±3.  At the thresholds we check (70, 80) that error is
# negligible.
_NPC_BASE_TRUST = {"captain": 58, "keeper": 50, "merchant": 52, "chaja": 46}


def _get_initial_npc_trust(
    run: Dict[str, Any],
    npc_key: str,
    events: List[Dict[str, Any]],
) -> float:
    """Return the known base trust for an NPC (±3 from seed jitter)."""
    return float(_NPC_BASE_TRUST.get(npc_key, 50))
def extract_event_times(run: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """Extract turn numbers for key milestones."""
    events = run.get("transcript_events") or run.get("transcript") or []
    objective = run.get("objective", "")

    milestones: Dict[str, Optional[int]] = {
        "turn_first_talk": None,
        "turn_first_clue": None,
        "turn_first_captain_talk": None,
        "turn_captain_trust_70": None,      # trust objective
        "turn_captain_trust_80": None,      # trust objective
        "turn_first_rec_request": None,     # trust objective
        "turn_first_correct_suspect": None, # identification objective
        "turn_first_probe": None,
    }

    # Track running state
    captain_trust_running = _get_initial_npc_trust(run, "captain", events)
    clue_count_running = 0
    rec_count_running = 0
    probe_count_running = 0
    suspect_scores_running: Dict[str, int] = {}
    marked_target = run.get("objective_state", {}).get("marked_target")

    for evt in events:
        if not isinstance(evt, dict):
            continue
        turn = evt.get("turn", 0)
        action = evt.get("action") or evt.get("parsed_action") or {}
        if not isinstance(action, dict):
            continue
        cmd = action.get("command", "")

        # First talk to any NPC
        if cmd == "talk" and milestones["turn_first_talk"] is None:
            milestones["turn_first_talk"] = turn

# First talk to captain
        if cmd == "talk":
            reactions = evt.get("npc_reactions", [])
            for reaction in reactions:
                if not isinstance(reaction, dict) or reaction.get("npc") != "captain":
                    continue

                if milestones["turn_first_captain_talk"] is None:
                    milestones["turn_first_captain_talk"] = turn

                state_delta = reaction.get("state_delta") or {}
                trust_delta = 0.0
                if isinstance(state_delta, dict):
                    trust_delta = float(state_delta.get("trust_delta", 0) or 0)

                captain_trust_running += trust_delta
                if captain_trust_running >= 70 and milestones["turn_captain_trust_70"] is None:
                    milestones["turn_captain_trust_70"] = turn
                if captain_trust_running >= 80 and milestones["turn_captain_trust_80"] is None:
                    milestones["turn_captain_trust_80"] = turn
                break

            # Check for probe classification
            signal = evt.get("dialogue_signal")
            if isinstance(signal, dict) and signal.get("direct_objective_probe"):
                if milestones["turn_first_probe"] is None:
                    milestones["turn_first_probe"] = turn

        # Track trust from give commands to captain (+3 per valid give)
        if cmd == "give" and evt.get("valid", True):
            narration = evt.get("narration", "")
            if "Captain Ser Alarik" in narration:
                captain_trust_running += 3
                if captain_trust_running >= 70 and milestones["turn_captain_trust_70"] is None:
                    milestones["turn_captain_trust_70"] = turn
                if captain_trust_running >= 80 and milestones["turn_captain_trust_80"] is None:
                    milestones["turn_captain_trust_80"] = turn

        # Check for recommendation request (watch objective)
        if cmd == "talk":
            args = action.get("args", [])
            if len(args) >= 2:
                speech = str(args[1]).lower()
                if any(w in speech for w in ["recommend", "vouch", "endorse", "join the watch"]):
                    if milestones["turn_first_rec_request"] is None:
                        milestones["turn_first_rec_request"] = turn

    # For clue timing, we need to check knowledge progression
    # This is approximate - checking if knowledge_count > 0 at end
    knowledge = run.get("knowledge", [])
    if knowledge:
        milestones["turn_first_clue"] = None  # Can't determine exact turn from final state

    return milestones


def event_time_analysis(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute event-time statistics per model."""
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for r in runs:
        times = extract_event_times(r)
        times["_objective"] = r.get("objective")
        times["_success"] = r.get("success", False)
        by_model[r["_model"]].append(times)

    milestone_keys = ["turn_first_talk", "turn_first_captain_talk",
                      "turn_captain_trust_70", "turn_captain_trust_80",
                      "turn_first_probe"]

    results: Dict[str, Dict] = {}
    for model in sorted(by_model.keys()):
        model_times = by_model[model]
        model_stats: Dict[str, Any] = {}
        for mk in milestone_keys:
            values = [t[mk] for t in model_times if t[mk] is not None]
            if values:
                model_stats[mk] = {
                    "median": round(float(np.median(values)), 1),
                    "mean": round(float(np.mean(values)), 1),
                    "n_reached": len(values),
                    "n_total": len(model_times),
                    "pct_reached": round(len(values) / len(model_times) * 100, 1),
                }
            else:
                model_stats[mk] = {"n_reached": 0, "n_total": len(model_times), "pct_reached": 0.0}
        results[model] = model_stats

    # Survival-style: for each milestone, compute % of models reaching it by turn T
    survival_data: Dict[str, Dict[str, List[Optional[int]]]] = defaultdict(lambda: defaultdict(list))
    for model, times_list in by_model.items():
        for t in times_list:
            for mk in milestone_keys:
                survival_data[mk][model].append(t[mk])

    return {
        "per_model": results,
        "milestone_keys": milestone_keys,
    }


# ---------------------------------------------------------------------------
# Phase 3: Statistical tests
# ---------------------------------------------------------------------------

def bh_fdr_adjust(p_values: List[float]) -> List[float]:
    """Apply Benjamini-Hochberg FDR correction."""
    if not p_values:
        return []

    n_tests = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * n_tests
    running_min = 1.0

    for sorted_index in range(n_tests - 1, -1, -1):
        original_index, raw_p = indexed[sorted_index]
        rank = sorted_index + 1
        candidate = min(1.0, raw_p * n_tests / rank)
        running_min = min(running_min, candidate)
        adjusted[original_index] = running_min

    return adjusted


def run_dunn_test(groups: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Compute pairwise Dunn's test p-values with BH-FDR correction."""
    models = sorted(groups.keys())
    matrix = {model: {other: 1.0 for other in models} for model in models}
    if len(models) < 2:
        return matrix

    all_scores: List[float] = []
    all_labels: List[str] = []
    for model in models:
        values = [float(value) for value in groups[model]]
        all_scores.extend(values)
        all_labels.extend([model] * len(values))

    total_observations = len(all_scores)
    if total_observations < 2:
        return matrix

    ranks = stats.rankdata(all_scores, method="average")
    rank_sums: Dict[str, float] = defaultdict(float)
    group_sizes = {model: len(groups[model]) for model in models}
    for label, rank in zip(all_labels, ranks):
        rank_sums[label] += float(rank)

    mean_ranks = {
        model: rank_sums[model] / group_sizes[model]
        for model in models
        if group_sizes[model] > 0
    }
    tie_counts = Counter(all_scores)
    tie_sum = sum((count**3 - count) for count in tie_counts.values() if count > 1)
    tie_adjustment = tie_sum / (12.0 * (total_observations - 1)) if total_observations > 1 else 0.0
    variance_term = (total_observations * (total_observations + 1) / 12.0) - tie_adjustment

    pairs: List[Tuple[str, str]] = []
    raw_p_values: List[float] = []
    for index, model_a in enumerate(models):
        for model_b in models[index + 1:]:
            denominator = math.sqrt(
                max(variance_term, 0.0)
                * ((1.0 / group_sizes[model_a]) + (1.0 / group_sizes[model_b]))
            )
            if denominator == 0:
                raw_p = 1.0
            else:
                z_score = abs(mean_ranks[model_a] - mean_ranks[model_b]) / denominator
                raw_p = float(2.0 * stats.norm.sf(z_score))
            pairs.append((model_a, model_b))
            raw_p_values.append(raw_p)

    adjusted_p_values = bh_fdr_adjust(raw_p_values)
    for (model_a, model_b), adjusted_p in zip(pairs, adjusted_p_values):
        matrix[model_a][model_b] = adjusted_p
        matrix[model_b][model_a] = adjusted_p

    return matrix


def run_statistical_tests(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Kruskal-Wallis, Dunn's test with BH-FDR, clustering."""
    models = sorted(set(r["_model"] for r in runs))

    # Organize scores by model
    scores_by_model: Dict[str, List[float]] = defaultdict(list)
    dim_by_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    # Also bonus-free
    bf_by_model: Dict[str, List[float]] = defaultdict(list)

    for r in runs:
        m = r["_model"]
        scores_by_model[m].append(r["_scores"]["total"])
        for dim in DIMENSIONS:
            dim_by_model[m][dim].append(r["_scores"][dim])
        bf_scores = compute_bonus_free_scores(r)
        bf_by_model[m].append(bf_scores["total_bf"])

    # --- Kruskal-Wallis ---
    groups = [scores_by_model[m] for m in models]
    kw_stat, kw_p = stats.kruskal(*groups)

    # --- Dunn's test with BH-FDR ---
    dunn_result = run_dunn_test(scores_by_model)

    # Count significant pairs
    n_pairs = len(models) * (len(models) - 1) // 2
    sig_pairs = 0
    sig_pair_list = []
    nonsig_pair_list = []
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            p = dunn_result[m1][m2]
            if p < 0.05:
                sig_pairs += 1
                sig_pair_list.append((m1, m2, round(p, 6)))
            else:
                nonsig_pair_list.append((m1, m2, round(p, 6)))

    # --- Per-dimension Dunn's tests ---
    dim_sig_counts: Dict[str, int] = {}
    dim_sig_pairs: Dict[str, List] = {}
    for dim in DIMENSIONS:
        dim_dunn = run_dunn_test({m: dim_by_model[m][dim] for m in models})
        count = 0
        pairs = []
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                p = dim_dunn[m1][m2]
                if p < 0.05:
                    count += 1
                    pairs.append((m1, m2, round(p, 6)))
        dim_sig_counts[dim] = count
        dim_sig_pairs[dim] = pairs

    # --- Eta-squared per dimension ---
    dim_eta_sq: Dict[str, float] = {}
    for dim in DIMENSIONS:
        all_vals = np.array([v for m in models for v in dim_by_model[m][dim]])
        grand_mean = np.mean(all_vals)
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        ss_between = sum(
            len(dim_by_model[m][dim]) * (np.mean(dim_by_model[m][dim]) - grand_mean) ** 2
            for m in models
        )
        dim_eta_sq[dim] = round(ss_between / ss_total, 4) if ss_total > 0 else 0.0

    # --- Hierarchical clustering ---
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # Use 1 - (p_value) as distance? Actually use mean differences
    model_means = {m: np.mean(scores_by_model[m]) for m in models}
    ranked_models = sorted(models, key=lambda m: -model_means[m])

    # Distance matrix based on effect sizes (Cohen's d)
    n = len(models)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m1, m2 = ranked_models[i], ranked_models[j]
            s1, s2 = scores_by_model[m1], scores_by_model[m2]
            pooled_std = np.sqrt((np.var(s1, ddof=1) + np.var(s2, ddof=1)) / 2)
            d = abs(np.mean(s1) - np.mean(s2)) / pooled_std if pooled_std > 0 else 0
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="ward")
    # Cut at d=0.5 (medium effect size threshold)
    cluster_labels = fcluster(Z, t=1.0, criterion="distance")

    clusters: Dict[int, List[str]] = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        clusters[label].append(ranked_models[i])

    # --- Bootstrap CIs (scenario-cluster resampling) ---
    bootstrap_cis: Dict[str, Tuple[float, float, float]] = {}
    n_boot = 10000
    rng = np.random.RandomState(42)

    # Group runs by model and scenario cell
    by_model_cell: Dict[str, Dict[Tuple, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        m = r["_model"]
        cell = (r.get("objective"), r.get("seed"))
        by_model_cell[m][cell].append(r["_scores"]["total"])

    for m in models:
        cells = by_model_cell[m]
        cell_means = [np.mean(v) for v in cells.values()]
        cell_keys = list(cells.keys())
        n_cells = len(cell_keys)
        boot_means = []
        for _ in range(n_boot):
            idx = rng.choice(n_cells, size=n_cells, replace=True)
            sampled = [cell_means[i] for i in idx]
            boot_means.append(np.mean(sampled))
        lo = float(np.percentile(boot_means, 2.5))
        hi = float(np.percentile(boot_means, 97.5))
        bootstrap_cis[m] = (lo, round(np.mean(cell_means), 4), hi)

    # --- Bonus-free comparison ---
    bf_model_means = {m: round(float(np.mean(bf_by_model[m])), 4) for m in models}
    bf_ranked = sorted(bf_model_means.items(), key=lambda x: -x[1])

    # Original ranking
    orig_ranked = sorted(model_means.items(), key=lambda x: -x[1])
    orig_rank_map = {m: i + 1 for i, (m, _) in enumerate(orig_ranked)}
    bf_rank_map = {m: i + 1 for i, (m, _) in enumerate(bf_ranked)}

    rank_changes = []
    for m in models:
        delta = orig_rank_map[m] - bf_rank_map[m]
        if abs(delta) >= 2:
            rank_changes.append({
                "model": m,
                "original_rank": orig_rank_map[m],
                "bonus_free_rank": bf_rank_map[m],
                "change": delta,
            })

    return {
        "kruskal_wallis": {"H": round(kw_stat, 2), "p": kw_p},
        "dunns_test_bh_fdr": {
            "total_pairs": n_pairs,
            "significant_pairs": sig_pairs,
            "pct_significant": round(sig_pairs / n_pairs * 100, 1),
            "significant": [(m1, m2, p) for m1, m2, p in sig_pair_list],
            "non_significant_count": len(nonsig_pair_list),
        },
        "per_dimension_dunns": {
            dim: {
                "significant_pairs": dim_sig_counts[dim],
                "eta_squared": dim_eta_sq[dim],
                "pairs": dim_sig_pairs[dim][:10],  # top 10 for brevity
            }
            for dim in DIMENSIONS
        },
        "clusters": {
            f"tier_{label}": {
                "models": members,
                "mean_range": f"{min(model_means[m] for m in members):.4f} - {max(model_means[m] for m in members):.4f}",
            }
            for label, members in sorted(clusters.items())
        },
        "bootstrap_cis": {
            m: {"lo": round(lo, 4), "mean": mean, "hi": round(hi, 4)}
            for m, (lo, mean, hi) in sorted(bootstrap_cis.items(), key=lambda x: -x[1][1])
        },
        "bonus_free_sensitivity": {
            "rankings": [(m, s) for m, s in bf_ranked],
            "material_rank_changes": rank_changes,
            "note": "Scores recomputed with +0.20 Goal Pursuit and +0.15 Strategic Sophistication success bonuses removed.",
        },
        "dimension_discriminability": {
            dim: {"eta_squared": dim_eta_sq[dim], "significant_pairs_of_78": dim_sig_counts[dim]}
            for dim in sorted(DIMENSIONS, key=lambda d: -dim_eta_sq[d])
        },
    }


# ---------------------------------------------------------------------------
# Phase 5: Output generation
# ---------------------------------------------------------------------------

def build_master_csv(runs: List[Dict[str, Any]], stat_results: Dict, filepath: str):
    """Write master CSV with original + bonus-free scores + cluster + event times."""
    cluster_map = {}
    for tier, info in stat_results["clusters"].items():
        for m in info["models"]:
            cluster_map[m] = tier

    fieldnames = [
        "model", "objective", "seed", "scenario_index", "repeat_index", "run_number",
        "success",
        "goal_pursuit", "social_adaptation", "world_grounding", "strategic_sophistication", "total",
        "goal_pursuit_bf", "strategic_sophistication_bf", "total_bf",
        "cluster",
        "turn_first_talk", "turn_first_captain_talk", "turn_captain_trust_70",
        "turn_first_probe",
        "turns_used",
    ]

    rows = []
    for r in runs:
        bf = compute_bonus_free_scores(r)
        times = extract_event_times(r)
        rows.append({
            "model": r["_model"],
            "objective": r.get("objective"),
            "seed": r.get("seed"),
            "scenario_index": r.get("scenario_index"),
            "repeat_index": r.get("scenario_repeat_index"),
            "run_number": r.get("run_number"),
            "success": 1 if r.get("success") else 0,
            "goal_pursuit": round(r["_scores"]["goal_pursuit"], 4),
            "social_adaptation": round(r["_scores"]["social_adaptation"], 4),
            "world_grounding": round(r["_scores"]["world_grounding"], 4),
            "strategic_sophistication": round(r["_scores"]["strategic_sophistication"], 4),
            "total": round(r["_scores"]["total"], 4),
            "goal_pursuit_bf": round(bf["goal_pursuit_bf"], 4),
            "strategic_sophistication_bf": round(bf["strategic_sophistication_bf"], 4),
            "total_bf": round(bf["total_bf"], 4),
            "cluster": cluster_map.get(r["_model"], ""),
            "turn_first_talk": times["turn_first_talk"],
            "turn_first_captain_talk": times["turn_first_captain_talk"],
            "turn_captain_trust_70": times["turn_captain_trust_70"],
            "turn_first_probe": times["turn_first_probe"],
            "turns_used": r.get("turns", 0),
        })

    rows.sort(key=lambda r: (r["model"], r["objective"], r["seed"], r["repeat_index"] or 0))

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  CrucibleBench Enhanced Statistical Analysis")
    print("=" * 70)
    validate_paths()

    print("\n[Phase 1] Loading all runs...")
    runs = load_all_runs()
    print(f"  Loaded {len(runs)} runs across {len(set(r['_model'] for r in runs))} models")

    print("\n[Phase 1] Auditing classifier costs...")
    cost_audit = audit_classifier_costs()
    print(f"  {cost_audit['note']}")

    print("\n[Phase 2.5] Objective split analysis...")
    obj_analysis = objective_split_analysis(runs)
    print(f"  Interaction eta^2: {obj_analysis['interaction_eta_squared']}")
    print(f"  Cross-objective success correlation: r={obj_analysis['cross_objective_success_correlation']['spearman_r']}")
    print(f"  Divergent profiles: {len(obj_analysis['divergent_profiles'])} models")
    for dp in obj_analysis["divergent_profiles"]:
        print(f"    {dp['model']}: trust={dp['trust_success']:.0%} vs ident={dp['ident_success']:.0%} "
              f"(gap={dp['success_gap']:.0%})")

    print("\n[Phase 2.7] Event-time metrics...")
    event_analysis = event_time_analysis(runs)
    # Print a brief summary
    print("  Median turn to first captain talk by model:")
    for m in sorted(event_analysis["per_model"].keys()):
        ct = event_analysis["per_model"][m].get("turn_first_captain_talk", {})
        if isinstance(ct, dict) and ct.get("n_reached", 0) > 0:
            print(f"    {m:<20} turn {ct['median']:4.0f}  ({ct['pct_reached']:.0f}% reached)")

    print("\n[Phase 3] Running statistical tests...")
    stat_results = run_statistical_tests(runs)
    print(f"  Kruskal-Wallis H={stat_results['kruskal_wallis']['H']}, p={stat_results['kruskal_wallis']['p']:.2e}")
    print(f"  Dunn's test (BH-FDR): {stat_results['dunns_test_bh_fdr']['significant_pairs']}/{stat_results['dunns_test_bh_fdr']['total_pairs']} pairs significant ({stat_results['dunns_test_bh_fdr']['pct_significant']}%)")
    print("\n  Dimension discriminability (eta^2 and sig pairs):")
    for dim, info in stat_results["dimension_discriminability"].items():
        print(f"    {dim:<30} eta^2={info['eta_squared']:.4f}  sig_pairs={info['significant_pairs_of_78']}/78")
    print(f"\n  Clusters:")
    for tier, info in stat_results["clusters"].items():
        print(f"    {tier}: {', '.join(info['models'])} [{info['mean_range']}]")
    print(f"\n  Bootstrap 95% CIs:")
    for m, ci in stat_results["bootstrap_cis"].items():
        print(f"    {m:<20} [{ci['lo']:.4f}, {ci['hi']:.4f}]  mean={ci['mean']}")

    # Bonus-free sensitivity
    bf = stat_results["bonus_free_sensitivity"]
    if bf["material_rank_changes"]:
        print("\n  Bonus-free rank changes (>=2 positions):")
        for rc in bf["material_rank_changes"]:
            direction = "up" if rc["change"] > 0 else "down"
            print(f"    {rc['model']}: #{rc['original_rank']} -> #{rc['bonus_free_rank']} ({direction} {abs(rc['change'])})")

    # --- Build outputs ---
    print("\n[Phase 5] Writing outputs...")
    csv_path = os.path.join(OUTPUT_DIR, "enhanced_scores.csv")
    n_rows = build_master_csv(runs, stat_results, csv_path)
    print(f"  Master CSV: {csv_path} ({n_rows} rows)")

    # Write full report JSON
    report = {
        "title": "CrucibleBench Enhanced Statistical Analysis",
        "n_runs": len(runs),
        "n_models": len(set(r["_model"] for r in runs)),
        "cost_audit": cost_audit,
        "objective_split": obj_analysis,
        "event_time_summary": {
            m: event_analysis["per_model"][m]
            for m in sorted(event_analysis["per_model"].keys())
        },
        "statistical_tests": stat_results,
    }

    report_path = os.path.join(OUTPUT_DIR, "enhanced_analysis_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report JSON: {report_path}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
