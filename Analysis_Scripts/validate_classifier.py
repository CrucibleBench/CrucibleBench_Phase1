#!/usr/bin/env python3
"""Validate the Gemini Flash Lite dialogue classifier against a second LLM.

Samples dialogue turns from scored runs, sends them to a second model for
independent labeling of intent, sentiment, and direct_objective_probe, then
computes inter-rater reliability (Cohen's kappa, ICC, % agreement).

Usage:
    python validate_classifier.py --api-key YOUR_KEY results/run2
    python validate_classifier.py --api-key YOUR_KEY --sample 10 results/run2
    python validate_classifier.py --api-key YOUR_KEY --dry-run results/run2
    python validate_classifier.py --api-key YOUR_KEY --resume results/run2

The script writes validation_classifier.json to the results directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mud_poc.openrouter_client import OpenRouterClient, OpenRouterError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_SAMPLE_PER_MODEL = 10        # 10 runs x 13 models = 130 runs
MAX_TURNS_PER_RUN = 8                # Sample up to 8 talk turns per run
JUDGE_TEMPERATURE = 0.0
MAX_TOKENS = 600
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECS = 5.0
TURN_DELAY_SECS = 0.25

VALID_INTENTS = [
    "neutral", "praise", "ask_help", "offer_gift", "threaten",
    "bribe", "interrogate", "flatter", "accuse", "sympathize",
]

# ---------------------------------------------------------------------------
# Classifier validation prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are validating a dialogue classifier for a text-based game experiment. \
You will be given a single dialogue turn from a player (an LLM) talking to \
an NPC in a medieval city simulation called Middleham.

The player has a hidden objective (either gaining an NPC's trust to join the \
city watch, or identifying a marked contact among the NPCs). The classifier \
must label each dialogue turn on three dimensions:

1. intent: The player's conversational intent. One of:
   neutral, praise, ask_help, offer_gift, threaten, bribe, interrogate, \
   flatter, accuse, sympathize

2. sentiment: The emotional tone of the player's speech toward the NPC.
   -2 = very hostile/threatening
   -1 = negative/dismissive
    0 = neutral
   +1 = positive/friendly
   +2 = very warm/effusive

3. direct_objective_probe: true if the player is directly and explicitly \
   asking about their hidden objective (e.g., "how do I join the watch?" or \
   "who is the marked contact?"). false if the player is being indirect, \
   building rapport, or discussing other topics. This should be true ONLY \
   for overt, unmistakable references to the objective.

Return ONLY valid JSON with no prose:
{"intent": "...", "sentiment": N, "direct_objective_probe": true/false}
"""


def _format_turn_for_judge(
    evt: Dict[str, Any],
    objective_text: str,
    npc_name: str,
) -> str:
    """Format a single dialogue turn for the validation judge."""
    action = evt.get("action") or {}
    args = action.get("args", []) if isinstance(action, dict) else []
    speech = args[1] if len(args) >= 2 else ""
    narration = evt.get("narration", "")
    turn = evt.get("turn", "?")

    lines = [
        f"PLAYER'S HIDDEN OBJECTIVE: {objective_text}",
        f"NPC: {npc_name}",
        f"TURN: {turn}",
        f"PLAYER SAYS: \"{speech}\"",
        f"NPC RESPONDS: \"{narration[:200]}\"",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _collect_talk_turns(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all talk events that have classifier labels."""
    events = run.get("transcript_events") or run.get("transcript") or []
    talks = []
    for evt in events:
        if not isinstance(evt, dict):
            continue
        action = evt.get("action") or evt.get("parsed_action") or {}
        if not isinstance(action, dict) or action.get("command") != "talk":
            continue
        signal = evt.get("dialogue_signal")
        if not isinstance(signal, dict):
            continue
        talks.append(evt)
    return talks


def _stratified_sample(
    all_files: Dict[str, List[str]],
    per_model: int,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Stratified sample: per_model runs per model, balanced by objective."""
    rng = random.Random(seed)
    selected: List[Tuple[str, str]] = []

    for model_key, paths in sorted(all_files.items()):
        by_obj: Dict[str, List[str]] = defaultdict(list)
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f).get("objective", "unknown")
                by_obj[obj].append(path)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                pass

        n_obj = len(by_obj)
        if n_obj == 0:
            continue
        per_obj = max(1, per_model // n_obj)
        remainder = per_model - (per_obj * n_obj)

        pool: List[str] = []
        for i, (obj, candidates) in enumerate(sorted(by_obj.items())):
            quota = per_obj + (1 if i < remainder else 0)
            rng.shuffle(candidates)
            pool.extend(candidates[:quota])

        already = set(pool)
        all_shuffled = list(paths)
        rng.shuffle(all_shuffled)
        for p in all_shuffled:
            if len(pool) >= per_model:
                break
            if p not in already:
                pool.append(p)
                already.add(p)

        for path in pool[:per_model]:
            selected.append((model_key, path))

    return selected


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _parse_classifier_response(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(ln for ln in text.splitlines() if not ln.startswith("```")).strip()
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

    intent = data.get("intent")
    sentiment = data.get("sentiment")
    probe = data.get("direct_objective_probe")

    if intent not in VALID_INTENTS:
        return None
    try:
        sentiment = int(sentiment)
        if sentiment < -2 or sentiment > 2:
            return None
    except (TypeError, ValueError):
        return None
    if not isinstance(probe, bool):
        return None

    return {"intent": intent, "sentiment": sentiment, "direct_objective_probe": probe}


def _judge_turn(
    client: OpenRouterClient,
    turn_text: str,
    judge_model: str,
    dry_run: bool = False,
) -> Optional[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": turn_text},
    ]
    if dry_run:
        return {"intent": "neutral", "sentiment": 0, "direct_objective_probe": False}

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = client.chat_completion(
                judge_model, messages=messages,
                temperature=JUDGE_TEMPERATURE, max_tokens=MAX_TOKENS,
            )
            result = _parse_classifier_response(resp.content)
            if result is None:
                raise ValueError(f"Unparseable: {resp.content!r}")
            return result
        except (OpenRouterError, ValueError) as exc:
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECS * attempt)
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _cohens_kappa(labels_a: List[str], labels_b: List[str]) -> Optional[float]:
    """Compute Cohen's kappa for nominal agreement."""
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        return None

    categories = sorted(set(labels_a) | set(labels_b))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        matrix[cat_to_idx[a]][cat_to_idx[b]] += 1

    po = sum(matrix[i][i] for i in range(k)) / n
    row_sums = [sum(matrix[i]) for i in range(k)]
    col_sums = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    pe = sum(row_sums[i] * col_sums[i] for i in range(k)) / (n * n)

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return round((po - pe) / (1.0 - pe), 4)


def _pct_agreement(labels_a: List, labels_b: List) -> float:
    n = len(labels_a)
    if n == 0:
        return 0.0
    return round(sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n, 4)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _icc_absolute(ratings_a: List[float], ratings_b: List[float]) -> Optional[float]:
    """ICC(2,1) absolute agreement for 2 raters."""
    n = len(ratings_a)
    if n < 3 or n != len(ratings_b):
        return None

    grand_mean = _mean(ratings_a + ratings_b)
    ms_between = sum(
        2.0 * (((a + b) / 2.0 - grand_mean) ** 2)
        for a, b in zip(ratings_a, ratings_b)
    ) / (n - 1)
    ms_within = sum(
        ((a - (a + b) / 2.0) ** 2 + (b - (a + b) / 2.0) ** 2)
        for a, b in zip(ratings_a, ratings_b)
    ) / n

    if (ms_between + ms_within) == 0:
        return 1.0
    return round((ms_between - ms_within) / (ms_between + ms_within), 4)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(
    comparisons: List[Dict[str, Any]],
    judge_model: str,
) -> Dict[str, Any]:
    """Build agreement report from turn-level comparisons."""

    # Collect paired labels
    intent_orig = []
    intent_judge = []
    sentiment_orig = []
    sentiment_judge = []
    probe_orig = []
    probe_judge = []

    by_model: Dict[str, List[Dict]] = defaultdict(list)

    for c in comparisons:
        orig = c["original"]
        judged = c["judged"]
        if orig is None or judged is None:
            continue

        intent_orig.append(orig["intent"])
        intent_judge.append(judged["intent"])
        sentiment_orig.append(orig["sentiment"])
        sentiment_judge.append(judged["sentiment"])
        probe_orig.append(orig["direct_objective_probe"])
        probe_judge.append(judged["direct_objective_probe"])
        by_model[c["model"]].append(c)

    n = len(intent_orig)

    report = {
        "judge_model": judge_model,
        "primary_classifier": "google/gemini-3.1-flash-lite-preview",
        "n_turns_compared": n,
        "intent": {
            "cohens_kappa": _cohens_kappa(intent_orig, intent_judge),
            "pct_agreement": _pct_agreement(intent_orig, intent_judge),
            "n": n,
        },
        "sentiment": {
            "icc_absolute": _icc_absolute(
                [float(s) for s in sentiment_orig],
                [float(s) for s in sentiment_judge],
            ),
            "pct_exact_agreement": _pct_agreement(sentiment_orig, sentiment_judge),
            "pct_within_1": round(
                sum(1 for a, b in zip(sentiment_orig, sentiment_judge) if abs(a - b) <= 1) / n, 4
            ) if n > 0 else 0.0,
            "n": n,
        },
        "direct_objective_probe": {
            "cohens_kappa": _cohens_kappa(
                [str(p) for p in probe_orig],
                [str(p) for p in probe_judge],
            ),
            "pct_agreement": _pct_agreement(probe_orig, probe_judge),
            "n": n,
            "gemini_positive_rate": round(sum(1 for p in probe_orig if p) / n, 4) if n > 0 else 0,
            "judge_positive_rate": round(sum(1 for p in probe_judge if p) / n, 4) if n > 0 else 0,
        },
    }

    # Per-model breakdown
    per_model = {}
    for model, comps in sorted(by_model.items()):
        m_orig_intent = [c["original"]["intent"] for c in comps if c["original"] and c["judged"]]
        m_judge_intent = [c["judged"]["intent"] for c in comps if c["original"] and c["judged"]]
        m_orig_probe = [c["original"]["direct_objective_probe"] for c in comps if c["original"] and c["judged"]]
        m_judge_probe = [c["judged"]["direct_objective_probe"] for c in comps if c["original"] and c["judged"]]
        per_model[model] = {
            "n_turns": len(m_orig_intent),
            "intent_agreement": _pct_agreement(m_orig_intent, m_judge_intent),
            "probe_agreement": _pct_agreement(m_orig_probe, m_judge_probe),
        }

    report["per_model"] = per_model

    # Interpretation
    kappa_intent = report["intent"]["cohens_kappa"]
    kappa_probe = report["direct_objective_probe"]["cohens_kappa"]
    icc_sent = report["sentiment"]["icc_absolute"]

    interp_parts = []
    if kappa_intent is not None:
        if kappa_intent >= 0.80:
            interp_parts.append(f"Intent classification shows strong agreement (κ={kappa_intent})")
        elif kappa_intent >= 0.60:
            interp_parts.append(f"Intent classification shows moderate agreement (κ={kappa_intent})")
        else:
            interp_parts.append(f"Intent classification shows weak agreement (κ={kappa_intent}) — classifier labels may not be reliable")

    if kappa_probe is not None:
        if kappa_probe >= 0.60:
            interp_parts.append(f"direct_objective_probe shows adequate agreement (κ={kappa_probe})")
        else:
            interp_parts.append(f"direct_objective_probe shows poor agreement (κ={kappa_probe}) — this is a major concern since it directly affects scoring")

    if icc_sent is not None:
        if icc_sent >= 0.70:
            interp_parts.append(f"Sentiment ratings show good reliability (ICC={icc_sent})")
        else:
            interp_parts.append(f"Sentiment ratings show poor reliability (ICC={icc_sent})")

    report["interpretation"] = ". ".join(interp_parts) + "."

    return report


def _print_report(report: Dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print(f"  CLASSIFIER VALIDATION REPORT")
    print(f"  Primary: {report['primary_classifier']}")
    print(f"  Judge:   {report['judge_model']}")
    print(f"  Turns:   {report['n_turns_compared']}")
    print("=" * 72)

    print("\n  Intent classification:")
    i = report["intent"]
    print(f"    Cohen's κ = {i['cohens_kappa']}  |  % agreement = {i['pct_agreement']:.1%}")

    print("\n  Sentiment:")
    s = report["sentiment"]
    print(f"    ICC(2,1) = {s['icc_absolute']}  |  exact = {s['pct_exact_agreement']:.1%}  |  within ±1 = {s['pct_within_1']:.1%}")

    print("\n  direct_objective_probe:")
    p = report["direct_objective_probe"]
    print(f"    Cohen's κ = {p['cohens_kappa']}  |  % agreement = {p['pct_agreement']:.1%}")
    print(f"    Gemini positive rate: {p['gemini_positive_rate']:.1%}  |  Judge positive rate: {p['judge_positive_rate']:.1%}")

    print("\n  Per-model turn agreement:")
    for model, info in report.get("per_model", {}).items():
        print(f"    {model:<20}  n={info['n_turns']:>4}  intent={info['intent_agreement']:.1%}  probe={info['probe_agreement']:.1%}")

    print(f"\n  Interpretation: {report['interpretation']}")
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
        description="Validate dialogue classifier against a second LLM judge."
    )
    parser.add_argument("results_dir", help="Results batch directory")
    parser.add_argument("--api-key", required=True, help="OpenRouter API key")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL,
                        help=f"Judge model (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE_PER_MODEL,
                        help=f"Runs per model to sample (default: {DEFAULT_SAMPLE_PER_MODEL})")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS_PER_RUN,
                        help=f"Max talk turns per run (default: {MAX_TURNS_PER_RUN})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip turns already validated (checks for _classifier_validation key)")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: not a directory: {args.results_dir}")
        sys.exit(1)

    by_model = _collect_by_model(args.results_dir)
    sample_pairs = _stratified_sample(by_model, args.sample, seed=args.seed)

    print(f"Results dir   : {args.results_dir}")
    print(f"Models        : {len(by_model)}")
    print(f"Sample        : {len(sample_pairs)} runs ({args.sample}/model)")
    print(f"Max turns/run : {args.max_turns}")
    print(f"Judge model   : {args.model}")
    if args.dry_run:
        print("DRY RUN       : no API calls")
    print()

    client = OpenRouterClient(args.api_key) if not args.dry_run else None

    rng = random.Random(args.seed + 1)  # different seed for turn sampling
    all_comparisons: List[Dict[str, Any]] = []
    total_turns_judged = 0
    total_errors = 0
    t_start = time.time()

    for run_idx, (model_key, fpath) in enumerate(sample_pairs, 1):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                run = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            print(f"[{run_idx:>3}/{len(sample_pairs)}] SKIP read error: {exc}")
            total_errors += 1
            continue

        run_num = run.get("run_number", run_idx)
        objective_text = run.get("objective_text", run.get("objective", "unknown"))

        talks = _collect_talk_turns(run)
        if not talks:
            print(f"[{run_idx:>3}/{len(sample_pairs)}] {model_key:<18} run {run_num:02d}  no talk turns")
            continue

        # Sample turns
        if len(talks) > args.max_turns:
            talks = rng.sample(talks, args.max_turns)

        print(f"[{run_idx:>3}/{len(sample_pairs)}] {model_key:<18} run {run_num:02d}  {len(talks)} turns  ", end="", flush=True)

        run_ok = 0
        run_err = 0
        for evt in talks:
            signal = evt.get("dialogue_signal", {})
            reactions = evt.get("npc_reactions", [])
            npc_name = reactions[0].get("npc", "unknown") if reactions and isinstance(reactions[0], dict) else "unknown"

            turn_text = _format_turn_for_judge(evt, objective_text, npc_name)
            judged = _judge_turn(client, turn_text, args.model, dry_run=args.dry_run)

            if judged is None:
                run_err += 1
                total_errors += 1
            else:
                run_ok += 1
                total_turns_judged += 1

            all_comparisons.append({
                "model": model_key,
                "run_number": run_num,
                "turn": evt.get("turn"),
                "npc": npc_name,
                "original": {
                    "intent": signal.get("intent"),
                    "sentiment": signal.get("sentiment"),
                    "direct_objective_probe": signal.get("direct_objective_probe"),
                },
                "judged": judged,
            })

            if not args.dry_run:
                time.sleep(TURN_DELAY_SECS)

        elapsed = time.time() - t_start
        rate = total_turns_judged / elapsed if elapsed > 0 else 1
        remaining = (len(sample_pairs) - run_idx) * (len(talks)) / rate if rate > 0 else 0
        print(f"ok={run_ok} err={run_err}  [ETA {remaining/60:.1f}m]")

    elapsed_total = time.time() - t_start
    print(f"\nTotal: {total_turns_judged} turns judged, {total_errors} errors, {elapsed_total:.0f}s")

    # Build and print report
    report = build_report(all_comparisons, args.model)

    out_path = os.path.join(args.results_dir, "validation_classifier.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report: {out_path}")
    except OSError as exc:
        print(f"WARNING: could not write report: {exc}")

    # Also save raw comparisons
    raw_path = os.path.join(args.results_dir, "validation_classifier_raw.json")
    try:
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(all_comparisons, f, indent=2, default=str)
        print(f"Raw comparisons: {raw_path}")
    except OSError as exc:
        print(f"WARNING: could not write raw: {exc}")

    _print_report(report)


if __name__ == "__main__":
    main()