#!/usr/bin/env python
"""Export all 325 run JSONs to computed_scores.csv using scoring.py.

Usage:
    python export_scores.py
    python export_scores.py results/20260303_005609 computed_scores.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mud_poc.scoring import score_run

RESULTS_DIR = "results/20260303_005609"
OUTPUT_CSV = "computed_scores.csv"

FIELDNAMES = [
    # Identity
    "model", "model_name", "objective", "run_id", "seed",
    # Outcome
    "win", "turns_used", "max_turns",
    # Scores (scoring.py)
    "goal", "social", "grounding", "strategy", "total",
    # GPT-5.2 validation scores (present on ~65 sampled runs, blank otherwise)
    "gpt52_goal", "gpt52_social", "gpt52_grounding", "gpt52_strategy", "gpt52_total",
    # Action quality
    "valid_actions", "invalid_actions", "total_actions", "valid_action_ratio",
    # Command breakdown
    "n_look", "n_go", "n_talk", "n_take", "n_give", "n_use", "n_examine",
    "n_commands_distinct",
    # World exploration
    "unique_rooms_visited", "total_room_visits",
    # Inventory
    "inventory_touched", "knowledge_count",
    # Social
    "npcs_engaged", "total_npc_talks",
    "npc_captain_trust", "npc_captain_suspicion", "npc_captain_talks",
    "npc_keeper_trust",  "npc_keeper_suspicion",  "npc_keeper_talks",
    "npc_merchant_trust","npc_merchant_suspicion","npc_merchant_talks",
    "npc_chaja_trust",   "npc_chaja_suspicion",   "npc_chaja_talks",
    "dialogue_intents_distinct",
    # Classifier
    "classifier_llm_count", "classifier_fallback_count",
    # Objective-specific: watch trust
    "watch_trust", "watch_talks", "watch_rec_requests",
    # Objective-specific: marked contact
    "marked_target", "inferred_target", "marked_correct",
    "direct_probes", "clue_count",
    # Suspect scores (marked contact objective)
    "suspect_captain", "suspect_keeper", "suspect_merchant", "suspect_chaja",
    # Cost / tokens
    "prompt_tokens", "completion_tokens", "estimated_cost_usd", "api_errors",
]


def _extract_row(model_key: str, run: Dict[str, Any]) -> Dict[str, Any]:
    # ----------------------------------------------------------------
    # Re-run scoring.py on the raw data (deterministic, authoritative)
    # score_run() mutates run["scores"] — we read back from that
    # ----------------------------------------------------------------
    scores = score_run(run)

    obj_state = run.get("objective_state", {}) or {}
    npcs_data = run.get("npcs", {}) or {}
    usage = run.get("usage", {}) or {}

    # ----------------------------------------------------------------
    # Events
    # ----------------------------------------------------------------
    events: List[Dict[str, Any]] = [
        e for e in (run.get("transcript_events") or run.get("transcript") or [])
        if isinstance(e, dict)
    ]

    valid_count = sum(1 for e in events if e.get("valid") is True)
    total_actions = len(events)
    invalid_count = total_actions - valid_count
    valid_ratio = round(valid_count / total_actions, 4) if total_actions else 0.0

    # Command counts
    commands: List[str] = []
    for e in events:
        action = e.get("action") or e.get("parsed_action") or {}
        if isinstance(action, dict):
            commands.append(action.get("command", "look"))
    cmd = Counter(commands)

    # ----------------------------------------------------------------
    # Dialogue / classifier
    # ----------------------------------------------------------------
    classifier_llm = 0
    classifier_fallback = 0
    intents: set = set()
    for e in events:
        sig = e.get("dialogue_signal")
        if not isinstance(sig, dict):
            continue
        conf = sig.get("confidence")
        # confidence == 0.55 is the hardcoded fallback signature in classifier.py
        if conf == 0.55:
            classifier_fallback += 1
        else:
            classifier_llm += 1
        intent = sig.get("intent")
        if intent:
            intents.add(intent)

    # ----------------------------------------------------------------
    # NPC data
    # ----------------------------------------------------------------
    def npc(key: str, field: str) -> Optional[int]:
        return npcs_data.get(key, {}).get(field)

    talked_npcs = obj_state.get("talked_npcs") or []
    total_npc_talks = sum(
        (npcs_data.get(k, {}).get("talk_count", 0) for k in npcs_data),
        0,
    )

    # ----------------------------------------------------------------
    # Locations
    # ----------------------------------------------------------------
    loc_visits: List[str] = obj_state.get("location_visits") or []
    unique_rooms = len(set(loc_visits))
    total_room_visits = len(loc_visits)

    # ----------------------------------------------------------------
    # Objective-specific
    # ----------------------------------------------------------------
    objective = run.get("objective", "")

    watch_trust_val = npc("captain", "trust")
    watch_talks_val = obj_state.get("watch_talks") or obj_state.get("guardian_talks")
    watch_rec_val = (
        obj_state.get("watch_recommendation_requests")
        or obj_state.get("guardian_recommendation_requests")
    )

    marked_target = obj_state.get("marked_target") or obj_state.get("shunned_target")
    inferred_target = obj_state.get("inferred_target")
    marked_correct: Optional[int] = None
    if marked_target is not None and inferred_target is not None:
        marked_correct = 1 if inferred_target == marked_target else 0

    suspect_scores = obj_state.get("suspect_scores") or {}

    # ----------------------------------------------------------------
    # GPT-5.2 validation scores (optional — only on sampled runs)
    # ----------------------------------------------------------------
    g52 = run.get("scores_gpt52") or {}

    # ----------------------------------------------------------------
    # Assemble row
    # ----------------------------------------------------------------
    return {
        "model":          model_key,
        "model_name":     run.get("model_name", ""),
        "objective":      objective,
        "run_id":         run.get("run_number", ""),
        "seed":           run.get("seed", ""),
        "win":            1 if run.get("success") else 0,
        "turns_used":     run.get("turns", 0),
        "max_turns":      run.get("max_turns", 25),
        # Automated scores
        "goal":           round(float(scores.get("goal_pursuit", 0)), 4),
        "social":         round(float(scores.get("social_adaptation", 0)), 4),
        "grounding":      round(float(scores.get("world_grounding", 0)), 4),
        "strategy":       round(float(scores.get("strategic_sophistication", 0)), 4),
        "total":          round(float(scores.get("total", 0)), 4),
        # GPT-5.2 validation (blank if not sampled)
        "gpt52_goal":     round(float(g52["goal_pursuit"]), 4)           if "goal_pursuit"             in g52 else "",
        "gpt52_social":   round(float(g52["social_adaptation"]), 4)      if "social_adaptation"        in g52 else "",
        "gpt52_grounding":round(float(g52["world_grounding"]), 4)        if "world_grounding"          in g52 else "",
        "gpt52_strategy": round(float(g52["strategic_sophistication"]), 4)if "strategic_sophistication" in g52 else "",
        "gpt52_total":    round(float(g52["total"]), 4)                  if "total"                    in g52 else "",
        # Action quality
        "valid_actions":       valid_count,
        "invalid_actions":     invalid_count,
        "total_actions":       total_actions,
        "valid_action_ratio":  valid_ratio,
        # Commands
        "n_look":    cmd.get("look", 0),
        "n_go":      cmd.get("go", 0),
        "n_talk":    cmd.get("talk", 0),
        "n_take":    cmd.get("take", 0),
        "n_give":    cmd.get("give", 0),
        "n_use":     cmd.get("use", 0),
        "n_examine": cmd.get("examine", 0),
        "n_commands_distinct": len(set(commands)),
        # Exploration
        "unique_rooms_visited": unique_rooms,
        "total_room_visits":    total_room_visits,
        # Inventory / knowledge
        "inventory_touched": obj_state.get("inventory_touched", 0),
        "knowledge_count":   run.get("knowledge_count", 0),
        # Social
        "npcs_engaged":    len(talked_npcs),
        "total_npc_talks": total_npc_talks,
        "npc_captain_trust":      npc("captain", "trust"),
        "npc_captain_suspicion":  npc("captain", "suspicion"),
        "npc_captain_talks":      npc("captain", "talk_count"),
        "npc_keeper_trust":       npc("keeper", "trust"),
        "npc_keeper_suspicion":   npc("keeper", "suspicion"),
        "npc_keeper_talks":       npc("keeper", "talk_count"),
        "npc_merchant_trust":     npc("merchant", "trust"),
        "npc_merchant_suspicion": npc("merchant", "suspicion"),
        "npc_merchant_talks":     npc("merchant", "talk_count"),
        "npc_chaja_trust":        npc("chaja", "trust"),
        "npc_chaja_suspicion":    npc("chaja", "suspicion"),
        "npc_chaja_talks":        npc("chaja", "talk_count"),
        "dialogue_intents_distinct": len(intents),
        # Classifier
        "classifier_llm_count":      classifier_llm,
        "classifier_fallback_count": classifier_fallback,
        # Objective-specific
        "watch_trust":          watch_trust_val,
        "watch_talks":          watch_talks_val,
        "watch_rec_requests":   watch_rec_val,
        "marked_target":        marked_target,
        "inferred_target":      inferred_target,
        "marked_correct":       marked_correct,
        "direct_probes":        obj_state.get("direct_objective_probes"),
        "clue_count":           obj_state.get("clue_count"),
        "suspect_captain":      suspect_scores.get("captain"),
        "suspect_keeper":       suspect_scores.get("keeper"),
        "suspect_merchant":     suspect_scores.get("merchant"),
        "suspect_chaja":        suspect_scores.get("chaja"),
        # Cost / tokens
        "prompt_tokens":       usage.get("prompt_tokens", ""),
        "completion_tokens":   usage.get("completion_tokens", ""),
        "estimated_cost_usd":  usage.get("estimated_cost_usd", ""),
        "api_errors":          len(run.get("api_errors") or []),
    }


def main() -> None:
    results_dir = sys.argv[1] if len(sys.argv) > 1 else RESULTS_DIR
    output_csv  = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_CSV

    if not os.path.isdir(results_dir):
        print(f"ERROR: not a directory: {results_dir}")
        sys.exit(1)

    # Collect all run files
    run_files: List[tuple] = []
    for model_dir_name in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_dir_name)
        if not os.path.isdir(model_dir) or model_dir_name.startswith("_"):
            continue
        for fname in sorted(os.listdir(model_dir)):
            if fname.startswith("run_") and fname.endswith(".json"):
                run_files.append((model_dir_name, os.path.join(model_dir, fname)))

    print(f"Found {len(run_files)} run files")

    rows = []
    errors = 0
    for model_key, fpath in run_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                run = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  SKIP {fpath}: {exc}")
            errors += 1
            continue
        try:
            rows.append(_extract_row(model_key, run))
        except Exception as exc:
            print(f"  ERROR extracting {fpath}: {exc}")
            errors += 1

    # Sort by model, then objective, then run_id
    rows.sort(key=lambda r: (r["model"], r["objective"], int(r["run_id"] or 0)))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {output_csv}  ({errors} errors)")

    # Quick sanity check
    models = sorted(set(r["model"] for r in rows))
    objectives = sorted(set(r["objective"] for r in rows))
    print(f"Models ({len(models)}): {', '.join(models)}")
    print(f"Objectives: {', '.join(objectives)}")
    wins = sum(r["win"] for r in rows)
    fallbacks = sum(int(r["classifier_fallback_count"] or 0) for r in rows)
    llm_class = sum(int(r["classifier_llm_count"] or 0) for r in rows)
    print(f"Total wins: {wins}/{len(rows)}  ({100*wins/len(rows):.1f}%)")
    print(f"Classifier: {llm_class} LLM calls, {fallbacks} fallbacks "
          f"({100*fallbacks/max(1,llm_class+fallbacks):.1f}% fallback rate)")
    gpt52_rows = sum(1 for r in rows if r.get("gpt52_total") != "")
    print(f"GPT-5.2 validation scores present: {gpt52_rows} rows")


if __name__ == "__main__":
    main()
