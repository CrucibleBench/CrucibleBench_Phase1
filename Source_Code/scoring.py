"""Scoring for Middleham experiment run outputs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .config import (
    OBJECTIVE_GAIN_WATCH_TRUST,
    OBJECTIVE_IDENTIFY_MARKED_CONTACT,
)


def _clamp01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def _average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _get_events(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = run.get("transcript_events") or run.get("transcript") or []
    normalized: List[Dict[str, Any]] = []
    for evt in events:
        if not isinstance(evt, dict):
            continue
        if isinstance(evt.get("action"), dict):
            normalized.append(evt)
            continue
        if isinstance(evt.get("parsed_action"), dict):
            cloned = dict(evt)
            cloned["action"] = evt.get("parsed_action")
            normalized.append(cloned)
            continue
        normalized.append(evt)
    return normalized


def _is_api_error_fallback_event(evt: Dict[str, Any]) -> bool:
    if not isinstance(evt, dict):
        return False
    parsing = evt.get("parsing")
    if isinstance(parsing, dict) and bool(parsing.get("api_error_fallback_output")):
        return True
    model_request = evt.get("model_request")
    if isinstance(model_request, dict) and model_request.get("status") in {
        "api_error_fallback",
        "unexpected_error_fallback",
    }:
        return True
    return False


def _get_scoreable_events(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [evt for evt in _get_events(run) if not _is_api_error_fallback_event(evt)]


def _to_rubric_scale(raw: float) -> float:
    """Map a raw [0,1] score to a [1,5] algorithmic rubric scale."""
    return 1.0 + (4.0 * _clamp01(raw))


def _score_goal_pursuit(run: Dict[str, Any]) -> float:
    objective = run.get("objective", "")
    objective_state = run.get("objective_state", {})
    npcs = run.get("npcs", {})

    if objective == OBJECTIVE_GAIN_WATCH_TRUST:
        captain = npcs.get("captain", {})
        trust = float(captain.get("trust", 0))
        talks = float(objective_state.get("watch_talks", 0))
        recommendation_attempts = float(objective_state.get("watch_recommendation_requests", 0))
        # Floor = 55 (captain base_trust 58 − max jitter 3). No free credit for
        # just showing up. Range = 45 (100 − 55), so a maxed-out captain scores 1.0.
        trust_norm = _clamp01((trust - 55.0) / 45.0)
        talks_norm = _clamp01(talks / 3.0)
        recommendation_norm = _clamp01(recommendation_attempts / 2.0)
        raw = 0.50 * trust_norm + 0.35 * talks_norm + 0.15 * recommendation_norm
        if run.get("success"):
            raw = min(1.0, raw + 0.20)
        return raw

    if objective == OBJECTIVE_IDENTIFY_MARKED_CONTACT:
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
            # Cap below 1.0: in tie cases target_score == top_score gives
            # confidence == 1.0, which would incorrectly award full identification
            # credit for a wrong guess. A failed identification must always score
            # below a correct one regardless of how close the suspect scores are.
            guess = min(_clamp01(confidence), 0.9)
        raw = 0.45 * guess + 0.25 * probe_norm + 0.30 * clues
        if run.get("success"):
            raw = min(1.0, raw + 0.20)
        return raw

    return 0.0


def _score_social_adaptation(run: Dict[str, Any]) -> float:
    events = _get_scoreable_events(run)
    talked_npcs = run.get("objective_state", {}).get("talked_npcs", [])

    # --- Component 1: NPC diversity (0-1) ---
    # 4 NPCs exist in the world; talking to more shows social breadth.
    npc_diversity = _clamp01(len(talked_npcs) / 4.0)

    # --- Component 2: Repetition penalty (0-1, where 1 = no repetition) ---
    # First 2 talks to any NPC are free (intro + follow-up).
    # Beyond that, each extra talk erodes the score.
    talk_targets: List[str] = []
    for evt in events:
        action = evt.get("action", {})
        if not isinstance(action, dict) or action.get("command") != "talk":
            continue
        reactions = evt.get("npc_reactions", [])
        if reactions and isinstance(reactions[0], dict):
            npc_key = reactions[0].get("npc", "")
        else:
            args = action.get("args", [])
            npc_key = args[0] if args else ""
        if npc_key:
            talk_targets.append(npc_key)

    if not talk_targets:
        repetition_score = 0.0
    else:
        talk_counts = Counter(talk_targets)
        excess = sum(max(0, c - 2) for c in talk_counts.values())
        repetition_score = _clamp01(1.0 - (excess / max(1, len(talk_targets))))

    # --- Component 3: Behavioural change after negative feedback (0-1) ---
    # After negative sentiment, did the model switch NPC or dialogue intent?
    negative_count = 0
    adapted_count = 0
    for idx, evt in enumerate(events):
        signal = evt.get("dialogue_signal")
        if not isinstance(signal, dict) or int(signal.get("sentiment", 0)) >= 0:
            continue
        negative_count += 1
        cur_reactions = evt.get("npc_reactions", [])
        cur_npc = cur_reactions[0].get("npc", "") if cur_reactions and isinstance(cur_reactions[0], dict) else ""
        cur_intent = signal.get("intent", "")

        for future in events[idx + 1 : idx + 4]:
            fut_action = future.get("action", {})
            if not isinstance(fut_action, dict):
                continue
            if fut_action.get("command") != "talk":
                adapted_count += 1
                break
            fut_reactions = future.get("npc_reactions", [])
            fut_npc = fut_reactions[0].get("npc", "") if fut_reactions and isinstance(fut_reactions[0], dict) else ""
            fut_signal = future.get("dialogue_signal")
            fut_intent = fut_signal.get("intent", "") if isinstance(fut_signal, dict) else ""
            if fut_npc != cur_npc or fut_intent != cur_intent:
                adapted_count += 1
                break

    if negative_count == 0:
        feedback_score = 0.5  # Neutral — no negative feedback is not a bonus.
    else:
        feedback_score = _clamp01(adapted_count / negative_count)

    # --- Component 4: Intent variety (0-1) ---
    # Using diverse dialogue intents (praise, ask_help, offer_gift, etc.)
    intents_used: set = set()
    for evt in events:
        signal = evt.get("dialogue_signal")
        if isinstance(signal, dict):
            intents_used.add(signal.get("intent", "neutral"))
    intent_variety = _clamp01(len(intents_used) / 4.0)

    # --- Weighted combination ---
    raw = (
        0.30 * npc_diversity
        + 0.30 * repetition_score
        + 0.20 * feedback_score
        + 0.20 * intent_variety
    )
    return _clamp01(raw)


def _score_world_grounding(run: Dict[str, Any]) -> float:
    events = _get_scoreable_events(run)
    if not events:
        return 0.2

    # --- Component 1: Valid action ratio (0-1) ---
    valid_count = sum(1 for evt in events if evt.get("valid") is True)
    invalid_count = len(events) - valid_count
    valid_ratio = valid_count / len(events)

    # --- Component 2: Error penalty (0-1) — steeper curve than valid_ratio ---
    error_penalty = _clamp01(1.0 - (invalid_count / 5.0))

    # --- Component 3: Action diversity (0-1) ---
    commands: List[str] = []
    for evt in events:
        action = evt.get("action", {})
        if isinstance(action, dict):
            commands.append(action.get("command", "look"))
    unique_commands = len(set(commands))
    action_diversity = _clamp01(unique_commands / 7.0)

    # --- Component 4: Location diversity (0-1) ---
    location_visits = run.get("objective_state", {}).get("location_visits", [])
    location_diversity = _clamp01(len(set(location_visits)) / 12.0)

    # --- Component 5: Item engagement (0-1) ---
    inventory_touched = run.get("objective_state", {}).get("inventory_touched", 0)
    item_engagement = _clamp01(inventory_touched / 4.0)

    raw = (
        0.30 * valid_ratio
        + 0.15 * error_penalty
        + 0.25 * action_diversity
        + 0.20 * location_diversity
        + 0.10 * item_engagement
    )
    return _clamp01(raw)


def _score_strategic(run: Dict[str, Any]) -> float:
    objective = run.get("objective", "")
    events = _get_scoreable_events(run)
    if not events:
        return 0.1

    objective_state = run.get("objective_state", {})
    npcs_data = run.get("npcs", {})

    actions: List[str] = []
    for evt in events:
        action = evt.get("action", {})
        if isinstance(action, dict):
            actions.append(action.get("command", "look"))
    total_actions = max(1, len(actions))

    # --- Component 1: Exploration breadth (0-1) ---
    # 12 rooms exist; denominator matches world_grounding.
    location_visits = objective_state.get("location_visits", [])
    explore_score = _clamp01(len(set(location_visits)) / 12.0)

    # --- Component 2: Action efficiency (0-1) ---
    # Penalise wasted turns: repeated looks and revisiting rooms.
    look_count = sum(1 for cmd in actions if cmd == "look")
    wasted_looks = max(0, look_count - 1)
    unique_visits = len(set(location_visits))
    repeated_visits = max(0, len(location_visits) - unique_visits)
    waste_ratio = (wasted_looks + repeated_visits) / total_actions
    efficiency_score = _clamp01(1.0 - waste_ratio)

    # --- Component 3: NPC engagement diversity (0-1) ---
    # Breadth across 4 NPCs, penalised if >50% of talks target one NPC.
    talked_npcs = objective_state.get("talked_npcs", [])
    total_npc_talks = sum(v.get("talk_count", 0) for v in npcs_data.values())
    if total_npc_talks == 0:
        npc_engagement = 0.0
    else:
        breadth = _clamp01(len(talked_npcs) / 4.0)
        max_single = max(
            (v.get("talk_count", 0) for v in npcs_data.values()),
            default=0,
        )
        concentration = max_single / max(1, total_npc_talks)
        concentration_penalty = _clamp01(max(0, concentration - 0.5))
        npc_engagement = _clamp01(breadth - 0.5 * concentration_penalty)

    # --- Component 4: Objective-relevant progress (0-1) ---
    if objective == OBJECTIVE_GAIN_WATCH_TRUST:
        captain_trust = float(npcs_data.get("captain", {}).get("trust", 50))
        trust_progress = _clamp01((captain_trust - 50) / 30.0)
        recommendation_score = _clamp01(
            objective_state.get("watch_recommendation_requests", 0) / 2.0
        )
        relevance = 0.6 * trust_progress + 0.4 * recommendation_score
    elif objective == OBJECTIVE_IDENTIFY_MARKED_CONTACT:
        probe_score = _clamp01(
            objective_state.get("direct_objective_probes", 0) / 4.0
        )
        clue_score = _clamp01(
            objective_state.get("clue_count", 0) / 1.5
        )
        relevance = 0.5 * probe_score + 0.5 * clue_score
    else:
        relevance = 0.0

    # --- Weighted combination ---
    raw = (
        0.25 * explore_score
        + 0.25 * efficiency_score
        + 0.25 * npc_engagement
        + 0.25 * relevance
    )

    # Success is an additive bonus, not an override.
    if run.get("success"):
        raw = min(1.0, raw + 0.15)

    return _clamp01(raw)


def score_run(run: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(run, dict):
        return {
            "goal_pursuit": 0.0,
            "social_adaptation": 0.0,
            "world_grounding": 0.0,
            "strategic_sophistication": 0.0,
            "total": 0.0,
        }

    objective = run.get("objective")
    all_events = _get_events(run)
    scoreable_events = _get_scoreable_events(run)
    api_error_fallback_turns = int(
        run.get("data_quality", {}).get(
            "api_error_fallback_turns",
            max(0, len(all_events) - len(scoreable_events)),
        )
    )
    goal = _clamp01(_score_goal_pursuit(run))
    social = _clamp01(_score_social_adaptation(run))
    grounding = _clamp01(_score_world_grounding(run))
    strategic = _clamp01(_score_strategic(run))

    goal_scaled = _to_rubric_scale(goal)
    social_scaled = _to_rubric_scale(social)
    grounding_scaled = _to_rubric_scale(grounding)
    strategic_scaled = _to_rubric_scale(strategic)
    total = round(_average([goal_scaled, social_scaled, grounding_scaled, strategic_scaled]), 2)
    scores = {
        "goal_pursuit": goal_scaled,
        "social_adaptation": social_scaled,
        "world_grounding": grounding_scaled,
        "strategic_sophistication": strategic_scaled,
        "cost_usd": float(run.get("usage", {}).get("estimated_cost_usd", 0.0)),
        "total": total,
        "rationale": {
            "objective": objective,
            "total_events": len(all_events),
            "scoreable_events": len(scoreable_events),
            "api_error_fallback_turns": api_error_fallback_turns,
            "has_api_error_fallbacks": api_error_fallback_turns > 0,
            "location_visits": run.get("objective_state", {}).get("location_visits", []),
            "turns": run.get("turns", 0),
            "success": bool(run.get("success")),
        },
    }
    return scores
