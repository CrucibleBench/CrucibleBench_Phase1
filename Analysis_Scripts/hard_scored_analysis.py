"""CrucibleBench hard-scored subtotal analysis for Run 2.

This script recomputes the World Grounding + Social Adaptation subtotal
("hard-scored") directly from the 650 Run 2 JSONs in ``results/run2``.
It compares hard-scored rankings and tiers against the full four-dimension
total using the current scoring pipeline in ``mud_poc.scoring`` and writes a
per-model CSV summary.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "run2"
VALIDATION_PATH = RESULTS_DIR / "validation_classifier.json"
OUTPUT_CSV_PATH = RESULTS_DIR / "hard_scored_analysis.csv"
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260313
EXPECTED_MODELS = 13
EXPECTED_RUNS_PER_MODEL = 50
EXPECTED_CELLS_PER_MODEL = 10
EXPECTED_RUNS_PER_CELL = 5

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mud_poc.scoring import score_run


MODEL_LABELS: Dict[str, str] = {
    "claude_haiku": "Claude Haiku 4.5",
    "claude_opus": "Claude Opus 4.6",
    "claude_sonnet": "Claude Sonnet 4.6",
    "deepseek_r1": "DeepSeek R1",
    "deepseek_v3_2": "DeepSeek V3.2",
    "gemini_3_1_pro": "Gemini 3.1 Pro",
    "gpt_5_2": "GPT-5.2",
    "gpt_5_3_chat": "GPT-5.3 Chat",
    "gpt_5_4": "GPT-5.4",
    "grok_4": "Grok 4",
    "mistral_large": "Mistral Large 3",
    "olmo_3_1": "OLMo 3.1 32B",
    "qwen_3_5": "Qwen 3.5 397B",
}


def _display_name(model_key: str) -> str:
    return MODEL_LABELS.get(model_key, model_key.replace("_", " "))


def _bh_fdr_adjust(p_values: Sequence[float]) -> List[float]:
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


def _run_dunn_test(groups: Mapping[str, Sequence[float]]) -> Dict[str, Dict[str, float]]:
    """Compute pairwise Dunn test adjusted p-values."""
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
        for model_b in models[index + 1 :]:
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

    adjusted_p_values = _bh_fdr_adjust(raw_p_values)
    for (model_a, model_b), adjusted_p in zip(pairs, adjusted_p_values):
        matrix[model_a][model_b] = adjusted_p
        matrix[model_b][model_a] = adjusted_p

    return matrix


def _eta_squared(groups: Iterable[Sequence[float]]) -> float:
    """Compute eta-squared from one-way between-group variance."""
    arrays = [np.asarray(group, dtype=float) for group in groups]
    all_values = np.concatenate(arrays) if arrays else np.array([], dtype=float)
    if all_values.size == 0:
        return 0.0

    grand_mean = float(np.mean(all_values))
    ss_total = float(np.sum((all_values - grand_mean) ** 2))
    if ss_total == 0:
        return 0.0

    ss_between = 0.0
    for group in arrays:
        if group.size == 0:
            continue
        ss_between += float(group.size * (np.mean(group) - grand_mean) ** 2)
    return ss_between / ss_total


def _load_validation_probe_agreement() -> Dict[str, float]:
    if not VALIDATION_PATH.exists():
        return {}

    with VALIDATION_PATH.open(encoding="utf-8") as handle:
        report = json.load(handle)

    per_model = report.get("per_model", {})
    return {
        model_key: float(info.get("probe_agreement", 0.0))
        for model_key, info in per_model.items()
        if isinstance(info, dict)
    }


def _load_run_frame() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(f"Run directory not found: {RESULTS_DIR}")

    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        model_key = model_dir.name
        for run_path in sorted(model_dir.glob("run_*.json")):
            with run_path.open(encoding="utf-8") as handle:
                run_data = json.load(handle)

            scores = score_run(run_data)
            hard_score = (scores["world_grounding"] + scores["social_adaptation"]) / 2.0
            cell_key = f"{run_data.get('objective', 'unknown')}::{run_data.get('seed', 'unknown')}"
            rows.append(
                {
                    "model_key": model_key,
                    "model_label": _display_name(model_key),
                    "objective": run_data.get("objective", "unknown"),
                    "seed": str(run_data.get("seed", "unknown")),
                    "cell_key": cell_key,
                    "run_number": int(run_data.get("run_number", 0)),
                    "success": bool(run_data.get("success", False)),
                    "goal": float(scores["goal_pursuit"]),
                    "social": float(scores["social_adaptation"]),
                    "ground": float(scores["world_grounding"]),
                    "strat": float(scores["strategic_sophistication"]),
                    "hard": float(hard_score),
                    "full": float(scores["total"]),
                }
            )

    if not rows:
        raise RuntimeError(f"No Run 2 JSONs found in {RESULTS_DIR}")

    runs_df = pd.DataFrame(rows)
    _validate_run_frame(runs_df)
    return runs_df


def _validate_run_frame(runs_df: pd.DataFrame) -> None:
    total_runs = len(runs_df)
    if total_runs != EXPECTED_MODELS * EXPECTED_RUNS_PER_MODEL:
        raise RuntimeError(
            f"Expected {EXPECTED_MODELS * EXPECTED_RUNS_PER_MODEL} runs, found {total_runs}."
        )

    per_model_counts = runs_df.groupby("model_key").size()
    if len(per_model_counts) != EXPECTED_MODELS:
        raise RuntimeError(f"Expected {EXPECTED_MODELS} models, found {len(per_model_counts)}.")
    if not (per_model_counts == EXPECTED_RUNS_PER_MODEL).all():
        raise RuntimeError(
            f"Each model must have {EXPECTED_RUNS_PER_MODEL} runs; got {per_model_counts.to_dict()}."
        )

    cell_sizes = runs_df.groupby(["model_key", "cell_key"]).size()
    cells_per_model = cell_sizes.groupby(level=0).size()
    if not (cells_per_model == EXPECTED_CELLS_PER_MODEL).all():
        raise RuntimeError(
            f"Each model must have {EXPECTED_CELLS_PER_MODEL} cells; got {cells_per_model.to_dict()}."
        )
    if not (cell_sizes == EXPECTED_RUNS_PER_CELL).all():
        raise RuntimeError(
            f"Each model/cell must have {EXPECTED_RUNS_PER_CELL} runs; got {cell_sizes.to_dict()}."
        )


def _build_ranking_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        runs_df.groupby(["model_key", "model_label"], as_index=False)
        .agg(
            WG=("ground", "mean"),
            SA=("social", "mean"),
            GP=("goal", "mean"),
            SS=("strat", "mean"),
            Hard=("hard", "mean"),
            Full=("full", "mean"),
        )
    )

    full_sorted = summary.sort_values(["Full", "model_label"], ascending=[False, True]).reset_index(drop=True)
    full_ranks = {row.model_key: index + 1 for index, row in full_sorted.iterrows()}

    hard_sorted = summary.sort_values(["Hard", "model_label"], ascending=[False, True]).reset_index(drop=True)
    hard_ranks = {row.model_key: index + 1 for index, row in hard_sorted.iterrows()}

    summary["Full Rank"] = summary["model_key"].map(full_ranks)
    summary["Hard Rank"] = summary["model_key"].map(hard_ranks)
    summary["Rank Delta"] = summary["Full Rank"] - summary["Hard Rank"]
    return summary.sort_values("Full Rank").reset_index(drop=True)


def _build_cell_frame(runs_df: pd.DataFrame) -> pd.DataFrame:
    cell_df = (
        runs_df.groupby(
            ["model_key", "model_label", "objective", "seed", "cell_key"],
            as_index=False,
        )
        .agg(
            run_count=("run_number", "size"),
            success_rate=("success", "mean"),
            goal=("goal", "mean"),
            social=("social", "mean"),
            ground=("ground", "mean"),
            strat=("strat", "mean"),
            hard=("hard", "mean"),
            full=("full", "mean"),
        )
    )

    expected_cells = EXPECTED_MODELS * EXPECTED_CELLS_PER_MODEL
    if len(cell_df) != expected_cells:
        raise RuntimeError(f"Expected {expected_cells} aggregated cells, found {len(cell_df)}.")

    per_model_counts = cell_df.groupby("model_key").size()
    if not (per_model_counts == EXPECTED_CELLS_PER_MODEL).all():
        raise RuntimeError(
            f"Each model must have {EXPECTED_CELLS_PER_MODEL} aggregated cells; "
            f"got {per_model_counts.to_dict()}."
        )
    if not (cell_df["run_count"] == EXPECTED_RUNS_PER_CELL).all():
        raise RuntimeError(
            "Each aggregated cell must contain "
            f"{EXPECTED_RUNS_PER_CELL} runs; got {cell_df['run_count'].tolist()}."
        )

    return cell_df


def _collect_score_groups(
    frame: pd.DataFrame,
    score_column: str,
    model_order: Sequence[str],
) -> Dict[str, np.ndarray]:
    return {
        model_key: frame.loc[frame["model_key"] == model_key, score_column].to_numpy(dtype=float)
        for model_key in model_order
    }


def _summarize_statistical_tests(
    groups: Mapping[str, Sequence[float]],
    model_order: Sequence[str],
) -> Dict[str, object]:
    kruskal_result = stats.kruskal(*groups.values())
    dunn_matrix = _run_dunn_test(groups)
    total_pairs = len(model_order) * (len(model_order) - 1) // 2
    observations_per_model = len(next(iter(groups.values()))) if groups else 0
    return {
        "kruskal": kruskal_result,
        "eta_squared": _eta_squared(groups.values()),
        "dunn_matrix": dunn_matrix,
        "significant_pairs": _count_significant_pairs(dunn_matrix, model_order),
        "total_pairs": total_pairs,
        "observations_per_model": observations_per_model,
    }


def _compare_significance_overlap(
    hard_matrix: Mapping[str, Mapping[str, float]],
    full_matrix: Mapping[str, Mapping[str, float]],
    model_order: Sequence[str],
) -> Dict[str, int]:
    overlap = {
        "both": 0,
        "hard_only": 0,
        "full_only": 0,
    }
    for index, model_a in enumerate(model_order):
        for model_b in model_order[index + 1 :]:
            hard_sig = hard_matrix[model_a][model_b] < 0.05
            full_sig = full_matrix[model_a][model_b] < 0.05
            if hard_sig and full_sig:
                overlap["both"] += 1
            elif hard_sig:
                overlap["hard_only"] += 1
            elif full_sig:
                overlap["full_only"] += 1

    return overlap


def _build_results_export(
    ranking_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    tiers_hard: Mapping[str, str],
    tiers_full: Mapping[str, str],
    boot_cis: Mapping[str, Tuple[float, float, float]],
    validation_probe_agreement: Mapping[str, float],
) -> pd.DataFrame:
    run_counts = runs_df.groupby("model_key").size()
    cell_counts = runs_df.groupby("model_key")["cell_key"].nunique()
    success_rates = runs_df.groupby("model_key")["success"].mean()

    export_df = ranking_df.rename(
        columns={
            "WG": "wg_mean",
            "SA": "sa_mean",
            "GP": "gp_mean",
            "SS": "ss_mean",
            "Hard": "hard_mean",
            "Full": "full_mean",
            "Hard Rank": "hard_rank",
            "Full Rank": "full_rank",
            "Rank Delta": "rank_delta",
        }
    ).copy()
    export_df["run_count"] = export_df["model_key"].map(run_counts)
    export_df["cell_count"] = export_df["model_key"].map(cell_counts)
    export_df["success_rate"] = export_df["model_key"].map(success_rates)
    export_df["hard_tier"] = export_df["model_key"].map(tiers_hard)
    export_df["full_tier"] = export_df["model_key"].map(tiers_full)
    export_df["tier_changed"] = export_df["hard_tier"] != export_df["full_tier"]
    export_df["hard_ci_lower"] = export_df["model_key"].map(lambda model_key: boot_cis[model_key][0])
    export_df["hard_ci_mean"] = export_df["model_key"].map(lambda model_key: boot_cis[model_key][1])
    export_df["hard_ci_upper"] = export_df["model_key"].map(lambda model_key: boot_cis[model_key][2])
    export_df["hard_ci_width"] = export_df["hard_ci_upper"] - export_df["hard_ci_lower"]
    export_df["probe_agreement"] = export_df["model_key"].map(validation_probe_agreement)

    return export_df[
        [
            "model_key",
            "model_label",
            "run_count",
            "cell_count",
            "success_rate",
            "wg_mean",
            "sa_mean",
            "gp_mean",
            "ss_mean",
            "hard_mean",
            "hard_rank",
            "full_mean",
            "full_rank",
            "rank_delta",
            "hard_tier",
            "full_tier",
            "tier_changed",
            "hard_ci_lower",
            "hard_ci_mean",
            "hard_ci_upper",
            "hard_ci_width",
            "probe_agreement",
        ]
    ]


def _write_results_csv(results_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False, float_format="%.6f")


def _build_gemini_interpretation(gemini_row: pd.Series) -> str:
    hard_rank = int(gemini_row["Hard Rank"])
    full_rank = int(gemini_row["Full Rank"])
    rank_delta = int(gemini_row["Rank Delta"])

    if rank_delta < 0:
        return (
            f"Gemini's hard-scored rank ({hard_rank}) lands below its full-score rank ({full_rank}), "
            "which is consistent with classifier-sensitive dimensions contributing more to its "
            "full-score placement."
        )
    if rank_delta > 0:
        return (
            f"Gemini's hard-scored rank ({hard_rank}) lands above its full-score rank ({full_rank}), "
            "so classifier-sensitive dimensions do not appear to be propping up its placement in "
            "this run."
        )
    return (
        f"Gemini's hard-scored rank ({hard_rank}) matches its full-score rank ({full_rank}), "
        "so removing classifier-sensitive dimensions does not change its placement."
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV_PATH,
        help="Path for the per-model CSV summary (default: results/run2/hard_scored_analysis.csv).",
    )
    return parser.parse_args(argv)


def _count_significant_pairs(
    dunn_matrix: Mapping[str, Mapping[str, float]],
    model_order: Sequence[str],
) -> int:
    significant = 0
    for index, model_a in enumerate(model_order):
        for model_b in model_order[index + 1 :]:
            if dunn_matrix[model_a][model_b] < 0.05:
                significant += 1
    return significant


def _compute_tiers(model_means: pd.Series) -> Dict[str, str]:
    model_keys = list(model_means.index)
    if len(model_keys) == 1:
        return {model_keys[0]: "Top"}

    features = model_means.to_numpy(dtype=float).reshape(-1, 1)
    linkage_matrix = linkage(features, method="ward")
    clusters = fcluster(linkage_matrix, t=min(3, len(model_keys)), criterion="maxclust")

    cluster_mean_scores: Dict[int, float] = {}
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        cluster_mean_scores[int(cluster_id)] = float(np.mean(features[mask]))

    tier_names = ["Top", "Mid", "Floor"]
    tier_map: Dict[int, str] = {}
    for index, (cluster_id, _) in enumerate(
        sorted(cluster_mean_scores.items(), key=lambda item: item[1], reverse=True)
    ):
        tier_map[cluster_id] = tier_names[index] if index < len(tier_names) else f"Tier {index + 1}"

    return {model_key: tier_map[int(cluster)] for model_key, cluster in zip(model_keys, clusters)}


def _bootstrap_confidence_intervals(
    runs_df: pd.DataFrame,
    score_column: str,
    iterations: int,
    seed: int,
) -> Dict[str, Tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    boot_cis: Dict[str, Tuple[float, float, float]] = {}

    for model_key, model_runs in runs_df.groupby("model_key"):
        cell_means = model_runs.groupby("cell_key")[score_column].mean().to_numpy(dtype=float)
        bootstrap_means = np.empty(iterations, dtype=float)
        for index in range(iterations):
            sampled = rng.choice(cell_means, size=len(cell_means), replace=True)
            bootstrap_means[index] = float(np.mean(sampled))

        lower = float(np.percentile(bootstrap_means, 2.5))
        upper = float(np.percentile(bootstrap_means, 97.5))
        mean_value = float(np.mean(cell_means))
        boot_cis[model_key] = (lower, mean_value, upper)

    return boot_cis


def main(argv: Sequence[str] | None = None) -> None:
    """Run the hard-scored analysis against the actual Run 2 dataset."""
    args = _parse_args(argv)
    runs_df = _load_run_frame()
    cell_df = _build_cell_frame(runs_df)
    validation_probe_agreement = _load_validation_probe_agreement()
    ranking_df = _build_ranking_table(runs_df)
    model_order = ranking_df["model_key"].tolist()
    n_models = len(model_order)

    print("=" * 80)
    print("CRUCIBLEBENCH HARD-SCORED SUBTOTAL ANALYSIS")
    print("World Grounding + Social Adaptation (classifier-independent dimensions)")
    print("=" * 80)
    print(f"Loaded {len(runs_df)} rescored runs from {RESULTS_DIR}")
    print(
        f"Validated dataset shape: {n_models} models x {EXPECTED_RUNS_PER_MODEL} runs "
        f"({EXPECTED_CELLS_PER_MODEL} cells x {EXPECTED_RUNS_PER_CELL} reps)"
    )

    print("\n1. HARD-SCORED VS FULL-SCORED MEANS AND RANKINGS")
    print("-" * 80)
    print(
        f"{'Model':<22} {'WG':>6} {'SA':>6} {'GP':>6} {'SS':>6} | "
        f"{'Hard':>6} {'HRnk':>5} | {'Full':>6} {'FRnk':>5} | {'dR':>4}"
    )
    print("-" * 90)
    for _, row in ranking_df.iterrows():
        delta_value = int(row["Rank Delta"])
        delta_text = f"{delta_value:+d}" if delta_value else "  0"
        print(
            f"{row['model_label']:<22} {row['WG']:6.2f} {row['SA']:6.2f} {row['GP']:6.2f} {row['SS']:6.2f} | "
            f"{row['Hard']:6.2f} {int(row['Hard Rank']):>5} | "
            f"{row['Full']:6.2f} {int(row['Full Rank']):>5} | {delta_text}"
        )

    print("\n2. RANK STABILITY")
    print("-" * 80)
    hard_ranks = ranking_df.sort_values("Full Rank")["Hard Rank"].to_numpy()
    full_ranks = ranking_df.sort_values("Full Rank")["Full Rank"].to_numpy()
    spearman_r, spearman_p = stats.spearmanr(hard_ranks, full_ranks)
    kendall_tau, kendall_p = stats.kendalltau(hard_ranks, full_ranks)
    print(f"Spearman rho (hard vs full ranks): {spearman_r:.4f}  (p={spearman_p:.2e})")
    print(f"Kendall tau (hard vs full ranks):  {kendall_tau:.4f}  (p={kendall_p:.2e})")
    print(f"Maximum rank shift: {int(ranking_df['Rank Delta'].abs().max())} positions")
    print(f"Mean absolute rank shift: {ranking_df['Rank Delta'].abs().mean():.2f} positions")

    movers = ranking_df[ranking_df["Rank Delta"].abs() >= 2].sort_values("Rank Delta", ascending=False)
    print("\nNotable rank shifts (|delta| >= 2):")
    if movers.empty:
        print("  None - all models shift by at most one position.")
    else:
        for _, row in movers.iterrows():
            delta_value = int(row["Rank Delta"])
            direction = "up" if delta_value > 0 else "down"
            print(
                f"  {row['model_label']:<22} Full #{int(row['Full Rank'])} -> "
                f"Hard #{int(row['Hard Rank'])} ({direction} {abs(delta_value)})"
            )

    print("\n3. STATISTICAL TESTS BY ANALYSIS UNIT")
    print("-" * 80)
    print("Run-level tests treat all 50 repetitions per model as independent.")
    print("Scenario-cell tests aggregate the 5 repetitions within each objective/seed cell.")
    run_hard_stats = _summarize_statistical_tests(
        _collect_score_groups(runs_df, "hard", model_order),
        model_order,
    )
    run_full_stats = _summarize_statistical_tests(
        _collect_score_groups(runs_df, "full", model_order),
        model_order,
    )
    cell_hard_stats = _summarize_statistical_tests(
        _collect_score_groups(cell_df, "hard", model_order),
        model_order,
    )
    cell_full_stats = _summarize_statistical_tests(
        _collect_score_groups(cell_df, "full", model_order),
        model_order,
    )
    print(
        f"{'Unit':<16} {'n/model':>7} | "
        f"{'Hard H':>7} {'Hard p':>10} {'eta^2':>6} {'sig':>7} | "
        f"{'Full H':>7} {'Full p':>10} {'eta^2':>6} {'sig':>7}"
    )
    print("-" * 97)
    for label, hard_stats, full_stats in (
        ("Run-level", run_hard_stats, run_full_stats),
        ("Scenario-cell", cell_hard_stats, cell_full_stats),
    ):
        hard_kruskal = hard_stats["kruskal"]
        full_kruskal = full_stats["kruskal"]
        hard_sig_text = f"{int(hard_stats['significant_pairs'])}/{int(hard_stats['total_pairs'])}"
        full_sig_text = f"{int(full_stats['significant_pairs'])}/{int(full_stats['total_pairs'])}"
        print(
            f"{label:<16} {int(hard_stats['observations_per_model']):7d} | "
            f"{hard_kruskal.statistic:7.2f} {hard_kruskal.pvalue:10.2e} "
            f"{float(hard_stats['eta_squared']):6.3f} "
            f"{hard_sig_text:>7} | "
            f"{full_kruskal.statistic:7.2f} {full_kruskal.pvalue:10.2e} "
            f"{float(full_stats['eta_squared']):6.3f} "
            f"{full_sig_text:>7}"
        )

    print("\n4. DUNN PAIRWISE OVERLAP BY ANALYSIS UNIT")
    print("-" * 80)
    for label, hard_stats, full_stats in (
        ("Run-level", run_hard_stats, run_full_stats),
        ("Scenario-cell", cell_hard_stats, cell_full_stats),
    ):
        overlap = _compare_significance_overlap(
            hard_stats["dunn_matrix"],
            full_stats["dunn_matrix"],
            model_order,
        )
        total_pairs = int(hard_stats["total_pairs"])
        print(f"{label}:")
        print(
            f"  HARD significant: {int(hard_stats['significant_pairs'])}/{total_pairs} "
            f"({100 * int(hard_stats['significant_pairs']) / total_pairs:.1f}%)"
        )
        print(
            f"  FULL significant: {int(full_stats['significant_pairs'])}/{total_pairs} "
            f"({100 * int(full_stats['significant_pairs']) / total_pairs:.1f}%)"
        )
        print(f"  Significant under BOTH: {overlap['both']}/{total_pairs}")
        print(f"  Significant HARD only:  {overlap['hard_only']}")
        print(f"  Significant FULL only:  {overlap['full_only']}")

    print("\n5. TIER ANALYSIS (HIERARCHICAL CLUSTERING)")
    print("-" * 80)
    hard_means = runs_df.groupby("model_key")["hard"].mean().loc[model_order]
    full_means = runs_df.groupby("model_key")["full"].mean().loc[model_order]
    tiers_hard = _compute_tiers(hard_means)
    tiers_full = _compute_tiers(full_means)

    for label, score_means, tiers in (
        ("HARD-SCORED TIERS (WG + SA only)", hard_means, tiers_hard),
        ("FULL-SCORED TIERS (all 4 dimensions)", full_means, tiers_full),
    ):
        print(f"\n  {label}:")
        for tier_name in ("Top", "Mid", "Floor"):
            tier_models = [
                (model_key, float(score_means[model_key]))
                for model_key in model_order
                if tiers.get(model_key) == tier_name
            ]
            if not tier_models:
                continue
            tier_models.sort(key=lambda item: item[1], reverse=True)
            score_range = f"{tier_models[-1][1]:.2f}-{tier_models[0][1]:.2f}"
            model_list = ", ".join(
                f"{_display_name(model_key)} ({score:.2f})" for model_key, score in tier_models
            )
            print(f"  {tier_name:>5} [{score_range}]: {model_list}")

    print("\n6. TIER STABILITY COMPARISON")
    print("-" * 80)
    print(f"{'Model':<22} {'Hard Tier':>10} {'Full Tier':>10} {'Changed?':>10}")
    print("-" * 55)
    tier_changes = 0
    for model_key in model_order:
        changed = "YES" if tiers_hard[model_key] != tiers_full[model_key] else ""
        if changed:
            tier_changes += 1
        print(
            f"{_display_name(model_key):<22} {tiers_hard[model_key]:>10} "
            f"{tiers_full[model_key]:>10} {changed:>10}"
        )
    print(f"\nModels changing tier: {tier_changes}/{n_models}")

    print("\n7. BOOTSTRAP 95% CIs (HARD-SCORED)")
    print("-" * 80)
    print(f"(Scenario-cell bootstrap, {BOOTSTRAP_ITERATIONS:,} iterations)")
    boot_cis = _bootstrap_confidence_intervals(
        runs_df,
        score_column="hard",
        iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED,
    )
    sorted_boot = sorted(boot_cis.items(), key=lambda item: item[1][1], reverse=True)
    print(f"\n{'Model':<22} {'Lower':>7} {'Mean':>7} {'Upper':>7} {'Width':>7}")
    print("-" * 55)
    for model_key, (lower, mean_value, upper) in sorted_boot:
        print(
            f"{_display_name(model_key):<22} {lower:7.2f} {mean_value:7.2f} "
            f"{upper:7.2f} {upper - lower:7.2f}"
        )

    results_csv_df = _build_results_export(
        ranking_df=ranking_df,
        runs_df=runs_df,
        tiers_hard=tiers_hard,
        tiers_full=tiers_full,
        boot_cis=boot_cis,
        validation_probe_agreement=validation_probe_agreement,
    )
    output_csv_path = args.output_csv
    _write_results_csv(results_csv_df, output_csv_path)
    print(f"\nCSV summary written to: {output_csv_path}")

    print("\n8. GEMINI CLASSIFIER-DEPENDENT DIMENSION CHECK")
    print("-" * 80)
    gemini_key = "gemini_3_1_pro"
    gemini_row = ranking_df.loc[ranking_df["model_key"] == gemini_key].iloc[0]
    gemini_shift = int(gemini_row["Rank Delta"])
    print("Gemini 3.1 Pro:")
    print(f"  Goal Pursuit (classifier-dependent): {gemini_row['GP']:.2f}")
    if gemini_key in validation_probe_agreement:
        print(
            "  Probe agreement rate:                "
            f"{validation_probe_agreement[gemini_key] * 100:.1f}%"
        )
    else:
        print("  Probe agreement rate:                not available")
    print(
        f"  Full score (4-dim):                  {gemini_row['Full']:.2f}  "
        f"(Rank #{int(gemini_row['Full Rank'])})"
    )
    print(
        f"  Hard score (WG+SA):                  {gemini_row['Hard']:.2f}  "
        f"(Rank #{int(gemini_row['Hard Rank'])})"
    )
    if gemini_shift == 0:
        print("  Rank change:                         no change")
    else:
        direction = "drops" if gemini_shift < 0 else "rises"
        print(
            f"  Rank change:                         {direction} by {abs(gemini_shift)} position(s)"
        )
    print(f"\n  Interpretation: {_build_gemini_interpretation(gemini_row)}")

    print("\n" + "=" * 80)
    print("SUMMARY FOR PAPER")
    print("=" * 80)
    stability_text = (
        "highly stable" if spearman_r > 0.9 else "moderately stable" if spearman_r > 0.7 else "notably shifted"
    )
    tier_text = (
        "The tier structure is fully stable under hard scoring."
        if tier_changes == 0
        else f"{tier_changes} model(s) change tier under hard scoring."
    )
    cell_hard_kruskal = cell_hard_stats["kruskal"]
    cell_total_pairs = int(cell_hard_stats["total_pairs"])
    gemini_text = (
        "Gemini drops under hard scoring."
        if int(gemini_row["Hard Rank"]) > int(gemini_row["Full Rank"])
        else "Gemini does not drop under hard scoring."
    )
    print(
        f"""
KEY FINDINGS:

1. RANK STABILITY: Spearman rho = {spearman_r:.3f} between hard-scored and full-scored rankings.
   The overall ordering is {stability_text} under classifier-independent scoring.

2. TIER STABILITY: {tier_text}

3. SIGNIFICANCE: Under scenario-cell aggregation, Kruskal-Wallis remains highly significant
   under hard scoring (H={cell_hard_kruskal.statistic:.1f}, p={cell_hard_kruskal.pvalue:.1e}).
   {int(cell_hard_stats['significant_pairs'])} of {cell_total_pairs} pairs are significant
   under hard scoring versus {int(cell_full_stats['significant_pairs'])} under full scoring.

4. GEMINI CHECK: {gemini_text}

RECOMMENDED PAPER LANGUAGE:
"Using the actual 650 Run 2 JSONs rescored with the current rubric, hard scoring
(World Grounding + Social Adaptation only) remains strongly discriminative
(scenario-cell Kruskal-Wallis H={cell_hard_kruskal.statistic:.1f}, p={cell_hard_kruskal.pvalue:.1e}). Rank-order correlation
between hard-scored and full-scored rankings is rho = {spearman_r:.2f}. {tier_text}"
"""
    )


if __name__ == "__main__":
    main()
