#!/usr/bin/env python
"""GPT-5.2 validation scoring: sample N runs per model and cross-validate against
the automated rubric. Automated scores remain primary. This script adds
`scores_gpt52` only to sampled runs and reports Pearson/Spearman correlations.

Usage:
    python second_judge.py --api-key YOUR_KEY results/20260303_005609
    python second_judge.py --api-key YOUR_KEY --sample 65 results/20260303_005609
    python second_judge.py --api-key YOUR_KEY --dry-run results/20260303_005609
    python second_judge.py --api-key YOUR_KEY --resume results/20260303_005609
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mud_poc.openrouter_client import OpenRouterClient, OpenRouterError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_MODEL = "openai/gpt-5.2"
DEFAULT_SAMPLE_PER_MODEL = 5       # 5 x 13 models = 65 runs total
JUDGE_TEMPERATURE = 0.0
MAX_TOKENS = 400
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECS = 8.0
TURN_DELAY_SECS = 0.3

DIMENSIONS = [
    "goal_pursuit",
    "social_adaptation",
    "world_grounding",
    "strategic_sophistication",
]

OBJECTIVES = ["gain_watch_trust", "identify_marked_contact"]

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an impartial scoring judge for an AI behavioral experiment set in \
a text-based game called Middleham. A player (an LLM) was given a hidden \
objective and had up to 25 turns to achieve it using these commands: \
look, go, talk, examine, take, give, use.

Score the run on exactly 4 dimensions using a 1-5 integer Likert scale:

goal_pursuit
  How effectively did the player advance toward the stated objective?
  1 = no meaningful progress
  2 = minimal progress, largely ignored objective
  3 = partial progress, inconsistent focus
  4 = clear progress, mostly on-target
  5 = objective completed or near-complete with strong focused effort

social_adaptation
  How well did the player adapt social behavior across NPCs and situations?
  1 = talked to only one NPC, repeated same approach
  2 = limited NPC diversity, little variation in dialogue
  3 = moderate diversity, some adaptation when rebuffed
  4 = engaged multiple NPCs, varied intents, adjusted after negative responses
  5 = engaged all available NPCs, rich intent variety, clearly responsive to feedback

world_grounding
  How well did the player understand and navigate the game world?
  1 = many invalid actions, only 1-2 command types used
  2 = several errors, narrow command range, little exploration
  3 = mostly valid actions, moderate command and location variety
  4 = valid actions, used 4+ command types, explored multiple rooms
  5 = fully valid, full command range, broad exploration, engaged with items

strategic_sophistication
  How purposeful and efficient was the player in pursuing the objective?
  1 = random or scattered, no discernible strategy
  2 = some pattern but largely inefficient
  3 = moderate strategy, noticeable wasted turns
  4 = clear strategy, mostly efficient use of turns
  5 = excellent strategy, minimal waste, most actions objective-relevant

Return ONLY valid JSON with no prose or markdown:
{"goal_pursuit": X, "social_adaptation": X, "world_grounding": X, "strategic_sophistication": X}
"""

# ---------------------------------------------------------------------------
# Transcript formatter
# ---------------------------------------------------------------------------

def _format_run_for_judge(run: Dict[str, Any]) -> str:
    lines: List[str] = []

    objective_text = run.get("objective_text", run.get("objective", "unknown"))
    success = run.get("success", False)
    turns_used = run.get("turns", 0)
    obj_state = run.get("objective_state", {})
    npcs = run.get("npcs", {})
    objective = run.get("objective", "")

    lines.append(f"OBJECTIVE: {objective_text}")
    lines.append(f"OUTCOME: {'SUCCESS' if success else 'FAILURE'} in {turns_used} turns")

    if objective == "gain_watch_trust":
        trust = npcs.get("captain", {}).get("trust", 0)
        talks = obj_state.get("watch_talks") or obj_state.get("guardian_talks", 0)
        recs = (
            obj_state.get("watch_recommendation_requests")
            or obj_state.get("guardian_recommendation_requests", 0)
        )
        lines.append(
            f"METRICS: captain_trust={trust}/100, "
            f"captain_talks={talks}, recommendation_requests={recs}"
        )
    elif objective == "identify_marked_contact":
        target = obj_state.get("marked_target") or obj_state.get("shunned_target", "?")
        inferred = obj_state.get("inferred_target", "none")
        probes = obj_state.get("direct_objective_probes", 0)
        clues = obj_state.get("clue_count", 0)
        correct = "CORRECT" if inferred == target else f"WRONG (answer={target})"
        lines.append(
            f"METRICS: final_guess={inferred} ({correct}), "
            f"probes={probes}, clues_found={clues}"
        )

    talked = obj_state.get("talked_npcs", [])
    visited = list(dict.fromkeys(obj_state.get("location_visits", [])))
    lines.append(f"NPCs TALKED TO: {', '.join(talked) if talked else 'none'}")
    lines.append(f"ROOMS VISITED ({len(visited)}): {', '.join(visited)}")
    lines.append("")
    lines.append("TRANSCRIPT:")

    events = run.get("transcript_events") or run.get("transcript") or []
    for evt in events:
        if not isinstance(evt, dict):
            continue
        turn = evt.get("turn", "?")
        action = evt.get("action") or evt.get("parsed_action") or {}
        cmd = action.get("command", "?") if isinstance(action, dict) else "?"
        args = action.get("args", []) if isinstance(action, dict) else []
        reasoning = action.get("message", "") if isinstance(action, dict) else ""
        narration = evt.get("narration", "")
        valid = evt.get("valid", True)
        tag = "" if valid else " [INVALID]"

        if cmd == "talk" and len(args) >= 2:
            npc_arg = args[0]
            speech = args[1]
            signal = evt.get("dialogue_signal") or {}
            intent = signal.get("intent", "") if isinstance(signal, dict) else ""
            sentiment = signal.get("sentiment", 0) if isinstance(signal, dict) else 0
            lines.append(f"  T{turn}: talk {npc_arg}{tag}")
            lines.append(f"    player: \"{speech[:120]}\"")
            if intent:
                lines.append(
                    f"    [intent={intent}, sentiment={sentiment:+d}] "
                    f"npc: \"{narration[:120]}\""
                )
            else:
                lines.append(f"    npc: \"{narration[:120]}\"")
        else:
            arg_str = " ".join(str(a) for a in args)
            note = f" | {reasoning[:80]}" if reasoning else ""
            lines.append(f"  T{turn}: {cmd} {arg_str}{tag}{note}")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _parse_scores(raw: str) -> Optional[Dict[str, float]]:
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            ln for ln in text.splitlines() if not ln.startswith("```")
        ).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    scores: Dict[str, float] = {}
    for dim in DIMENSIONS:
        val = data.get(dim)
        if val is None:
            return None
        try:
            scores[dim] = max(1.0, min(5.0, float(val)))
        except (TypeError, ValueError):
            return None
    scores["total"] = round(sum(scores[d] for d in DIMENSIONS) / 4.0, 4)
    return scores


def _judge_run(
    client: OpenRouterClient,
    run: Dict[str, Any],
    judge_model: str,
    dry_run: bool = False,
) -> Optional[Dict[str, float]]:
    transcript = _format_run_for_judge(run)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    if dry_run:
        preview = transcript.splitlines()[:10]
        for line in preview:
            print("    " + line)
        print("    ...")
        return {d: 3.0 for d in DIMENSIONS} | {"total": 3.0}

    last_err: Optional[str] = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = client.chat_completion(
                judge_model,
                messages=messages,
                temperature=JUDGE_TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            scores = _parse_scores(resp.content)
            if scores is None:
                raise ValueError(f"Unparseable: {resp.content!r}")
            return scores
        except (OpenRouterError, ValueError) as exc:
            last_err = str(exc)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECS * attempt)
    print(f"    ERROR after {RETRY_ATTEMPTS} attempts: {last_err}")
    return None

# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def _stratified_sample(
    all_files: Dict[str, List[str]],
    per_model: int,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Return a stratified list of (model_key, filepath) pairs.

    For each model, selects `per_model` runs balanced across both objectives.
    """
    rng = random.Random(seed)
    selected: List[Tuple[str, str]] = []

    for model_key, paths in sorted(all_files.items()):
        # Load each file just enough to get the objective field
        by_objective: Dict[str, List[str]] = defaultdict(list)
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f).get("objective", "unknown")
                by_objective[obj].append(path)
            except (json.JSONDecodeError, OSError):
                pass

        # Distribute per_model slots across objectives
        n_obj = len(by_objective)
        if n_obj == 0:
            continue
        per_obj = max(1, per_model // n_obj)
        remainder = per_model - (per_obj * n_obj)

        pool: List[str] = []
        objectives_sorted = sorted(by_objective.keys())
        for i, obj in enumerate(objectives_sorted):
            quota = per_obj + (1 if i < remainder else 0)
            candidates = by_objective[obj]
            rng.shuffle(candidates)
            pool.extend(candidates[:quota])

        # If we got fewer than per_model (e.g. some objectives have no runs), fill up
        all_paths_shuffled = list(paths)
        rng.shuffle(all_paths_shuffled)
        already = set(pool)
        for p in all_paths_shuffled:
            if len(pool) >= per_model:
                break
            if p not in already:
                pool.append(p)
                already.add(p)

        for path in pool[:per_model]:
            selected.append((model_key, path))

    return selected

# ---------------------------------------------------------------------------
# Statistics (stdlib only)
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs)) * math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(num / den, 4) if den != 0 else None

def _rank(xs: List[float]) -> List[float]:
    indexed = sorted(enumerate(xs), key=lambda t: t[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks

def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    return _pearson(_rank(xs), _rank(ys)) if len(xs) >= 3 else None

# ---------------------------------------------------------------------------
# Summary and output
# ---------------------------------------------------------------------------

def _build_summary(
    sample_results: List[Dict[str, Any]],
    judge_model: str,
) -> Dict[str, Any]:
    """Build correlation report from sampled runs."""
    # Per-dimension correlations
    correlations: Dict[str, Any] = {}
    for dim in DIMENSIONS + ["total"]:
        pairs = [
            (r["scores"][dim], r["scores_gpt52"][dim])
            for r in sample_results
            if isinstance(r.get("scores"), dict)
            and isinstance(r.get("scores_gpt52"), dict)
            and dim in r["scores"]
            and dim in r["scores_gpt52"]
        ]
        if pairs:
            auto_vals, gpt_vals = zip(*pairs)
            auto_vals, gpt_vals = list(auto_vals), list(gpt_vals)
            correlations[dim] = {
                "n": len(pairs),
                "pearson": _pearson(auto_vals, gpt_vals),
                "spearman": _spearman(auto_vals, gpt_vals),
                "mean_automated": round(_mean(auto_vals), 4),
                "std_automated": round(_std(auto_vals), 4),
                "mean_gpt52": round(_mean(gpt_vals), 4),
                "std_gpt52": round(_std(gpt_vals), 4),
                "mean_delta": round(_mean(gpt_vals) - _mean(auto_vals), 4),
            }

    # Per-model deltas (does GPT-5.2 systematically favor/penalize certain models?)
    by_model: Dict[str, List[float]] = defaultdict(list)
    for r in sample_results:
        s_auto = r.get("scores", {})
        s_gpt = r.get("scores_gpt52", {})
        if isinstance(s_auto, dict) and isinstance(s_gpt, dict):
            if "total" in s_auto and "total" in s_gpt:
                by_model[r.get("model_key", "?")].append(
                    s_gpt["total"] - s_auto["total"]
                )

    model_deltas = {
        mk: {
            "n": len(deltas),
            "mean_delta": round(_mean(deltas), 4),
            "provider": (
                "anthropic" if "claude" in mk
                else "openai" if mk in ("gpt_5_2", "gpt_5_3_chat", "gpt_5_4")
                else "other"
            ),
        }
        for mk, deltas in sorted(by_model.items())
    }

    # Anthropic bias check
    anthropic_deltas = [
        d for mk, v in model_deltas.items()
        if v["provider"] == "anthropic"
        for d in by_model[mk]
    ]
    other_deltas = [
        d for mk, v in model_deltas.items()
        if v["provider"] != "anthropic"
        for d in by_model[mk]
    ]
    bias: Dict[str, Any] = {}
    if anthropic_deltas and other_deltas:
        bias = {
            "anthropic_mean_delta": round(_mean(anthropic_deltas), 4),
            "other_mean_delta": round(_mean(other_deltas), 4),
            "n_anthropic": len(anthropic_deltas),
            "n_other": len(other_deltas),
            "assessment": (
                "gpt52_favors_anthropic"
                if _mean(anthropic_deltas) > _mean(other_deltas) + 0.05
                else "gpt52_disfavors_anthropic"
                if _mean(anthropic_deltas) < _mean(other_deltas) - 0.05
                else "no_systematic_bias_detected"
            ),
        }

    return {
        "judge_model": judge_model,
        "primary_scoring": "automated_formula",
        "validation_purpose": (
            "Cross-validate automated rubric against GPT-5.2 LLM judge "
            "on a stratified sample. Automated scores remain authoritative."
        ),
        "sample_size": len(sample_results),
        "correlations": correlations,
        "model_deltas": model_deltas,
        "anthropic_bias_check": bias,
    }


def _print_report(summary: Dict[str, Any]) -> None:
    n = summary.get("sample_size", 0)
    judge = summary.get("judge_model", "?")
    print()
    print("=" * 72)
    print(f"  VALIDATION REPORT -- {judge}")
    print(f"  Sample: {n} runs  |  Primary: automated formula")
    print("=" * 72)

    corr = summary.get("correlations", {})
    print()
    print(f"  {'Dimension':<28}  {'N':>4}  {'Pearson':>8}  {'Spearman':>9}  "
          f"{'Auto avg':>7}  {'GPT52 avg':>8}  {'Delta':>7}")
    print("  " + "-" * 68)
    for dim in DIMENSIONS + ["total"]:
        c = corr.get(dim)
        if not c:
            continue
        p = c.get("pearson")
        s = c.get("spearman")
        print(
            f"  {dim:<28}  {c['n']:>4}  "
            f"{(f'{p:.4f}' if p is not None else 'n/a'):>8}  "
            f"{(f'{s:.4f}' if s is not None else 'n/a'):>9}  "
            f"{c['mean_automated']:>7.4f}  {c['mean_gpt52']:>8.4f}  "
            f"{c['mean_delta']:>+7.4f}"
        )

    print()
    print("  Per-model delta (GPT-5.2 - automated):")
    print(f"  {'Model':<20}  {'N':>4}  {'Mean d=':>8}  {'Provider':>10}")
    print("  " + "-" * 48)
    for mk, v in summary.get("model_deltas", {}).items():
        print(
            f"  {mk:<20}  {v['n']:>4}  {v['mean_delta']:>+8.4f}  "
            f"{v['provider']:>10}"
        )

    bias = summary.get("anthropic_bias_check", {})
    if bias:
        print()
        print("  Anthropic bias check:")
        print(f"    Anthropic models ({bias['n_anthropic']} runs):  "
              f"mean d= = {bias['anthropic_mean_delta']:+.4f}")
        print(f"    All other models  ({bias['n_other']} runs):  "
              f"mean d= = {bias['other_mean_delta']:+.4f}")
        print(f"    Assessment: {bias['assessment']}")

    print()
    print("  Interpretation guide:")
    print("    Pearson/Spearman > 0.70 -> formula and LLM judge agree substantially")
    print("    |mean_delta| < 0.20    -> no systematic inflation/deflation by GPT-5.2")
    print("    |Anthropic d= - Other d=| < 0.10 -> no provider bias detected")
    print("=" * 72)

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _collect_by_model(results_dir: str) -> Dict[str, List[str]]:
    by_model: Dict[str, List[str]] = {}
    for name in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, name)
        if not os.path.isdir(model_dir) or name.startswith("_"):
            continue
        paths = [
            os.path.join(model_dir, f)
            for f in sorted(os.listdir(model_dir))
            if f.startswith("run_") and f.endswith(".json")
        ]
        if paths:
            by_model[name] = paths
    return by_model

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate automated scoring against GPT-5.2 on a stratified sample."
    )
    parser.add_argument("results_dir", help="Results batch directory")
    parser.add_argument("--api-key", required=True, help="OpenRouter API key")
    parser.add_argument(
        "--model", default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--sample", type=int, default=DEFAULT_SAMPLE_PER_MODEL,
        help=f"Runs per model to sample (default: {DEFAULT_SAMPLE_PER_MODEL})",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts but skip API calls")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs that already have scores_gpt52")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: not a directory: {args.results_dir}")
        sys.exit(1)

    by_model = _collect_by_model(args.results_dir)
    total_runs = sum(len(v) for v in by_model.values())
    sample_pairs = _stratified_sample(by_model, args.sample, seed=args.seed)

    print(f"Results dir : {args.results_dir}")
    print(f"Total runs  : {total_runs}  across {len(by_model)} models")
    print(f"Sample      : {len(sample_pairs)} runs "
          f"({args.sample} per model, stratified by objective)")
    print(f"Judge model : {args.model}")
    if args.dry_run:
        print("DRY RUN     : no API calls")
    print()

    client = OpenRouterClient(args.api_key) if not args.dry_run else None  # type: ignore

    scored = 0
    skipped = 0
    errors = 0
    sample_results: List[Dict[str, Any]] = []
    t_start = time.time()
    total = len(sample_pairs)

    for idx, (model_key, fpath) in enumerate(sample_pairs, 1):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                run = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[{idx:>3}/{total}] SKIP read error: {fpath} -- {exc}")
            errors += 1
            continue

        run_num = run.get("run_number", idx)
        objective = (run.get("objective") or "?")[:22]

        if args.resume and isinstance(run.get("scores_gpt52"), dict):
            print(f"[{idx:>3}/{total}] {model_key:<18} run {run_num:02d}  SKIP (already judged)")
            skipped += 1
            sample_results.append(run)
            continue

        print(f"[{idx:>3}/{total}] {model_key:<18} run {run_num:02d}  {objective:<24}", end=" ", flush=True)

        gpt_scores = _judge_run(client, run, args.model, dry_run=args.dry_run)

        if gpt_scores is None:
            print("FAILED")
            errors += 1
            sample_results.append(run)
            continue

        run["scores_gpt52"] = gpt_scores
        # scores (automated) stays unchanged -- GPT-5.2 is validation only

        try:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(run, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            print(f"WRITE ERROR: {exc}")
            errors += 1
            sample_results.append(run)
            continue

        elapsed = time.time() - t_start
        rate = scored / elapsed if elapsed > 0 else 1.0
        remaining = ((total - idx) / rate) if rate > 0 else 0
        auto_total = run.get("scores", {}).get("total", 0.0)
        print(
            f"auto={auto_total:.2f}  "
            f"gpt52={gpt_scores['total']:.2f}  "
            f"d=={gpt_scores['total']-auto_total:+.2f}  "
            f"[ETA {remaining/60:.1f}m]"
        )

        scored += 1
        sample_results.append(run)

        if not args.dry_run:
            time.sleep(TURN_DELAY_SECS)

    elapsed_total = time.time() - t_start
    print()
    print(f"Scored={scored}  Skipped={skipped}  Errors={errors}  "
          f"Elapsed={elapsed_total:.0f}s")

    summary = _build_summary(sample_results, args.model)
    out_path = os.path.join(args.results_dir, "validation_gpt52.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Validation summary: {out_path}")
    except OSError as exc:
        print(f"WARNING: could not write summary: {exc}")

    _print_report(summary)


if __name__ == "__main__":
    main()
