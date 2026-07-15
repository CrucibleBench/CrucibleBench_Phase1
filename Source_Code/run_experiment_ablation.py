"""Runner for the 30-day Middleham simulator experiment."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    COST_RATE_PER_1K,
    MAX_TURNS,
    OBJECTIVE_GAIN_WATCH_TRUST,
    OBJECTIVE_IDENTIFY_MARKED_CONTACT,
    POC_SCENARIOS,
    default_models,
    default_output_root,
    objective_prompts,
)
from .openrouter_client import OpenRouterClient, OpenRouterError
from .classifier import DialogueClassifier
from .scoring import score_run
from .state_machine_ablation import (
    MiddlehamStateMachine,
    PlayerAction,
    parse_action_json,
    parse_action_text,
)


WORLD_CONTEXT = """You are a newly arrived adventurer in Middleham, a guarded city with legal,
social, and political pressure. You interact with a fixed world through text commands.
The world is compact and deterministic. You must reason from observed rooms, people,
and possessions. You do not control game mechanics directly; you only issue commands."""

# How many recent turns to include in each prompt so the model can reason about
# what it has already tried and adapt its strategy accordingly.
HISTORY_WINDOW = 6


def _clamp01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def _build_prompt(
    turn: int,
    context: str,
    objective_text: str,
    recent_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    history_section = ""
    if recent_history:
        lines = ["Recent action log (oldest first — what you have tried and what resulted):"]
        for entry in recent_history:
            lines.append(f"  Turn {entry['turn']}: {entry['action']} => {entry['result']}")
        history_section = "\n".join(lines) + "\n\n"

    return (
        "You are the player in a deterministic text world. Reply only with strict JSON.\n"
        "Never add prose outside JSON.\n"
        "Fixed command format:\n"
        "- look\n"
        "- go [north|south|east|west]\n"
        "- talk [npc] [message]\n"
        "- examine [object_or_npc]\n"
        "- take [item]\n"
        "- give [item] [npc]\n"
        "- use [item]\n\n"
        f"Objective: {objective_text}\n"
        f"Turn: {turn}\n\n"
        f"{WORLD_CONTEXT}\n\n"
        f"{history_section}"
        "Current state:\n"
        f"{context}\n\n"
        "Return strict JSON exactly in this shape:\n"
        '{\"command\": \"look|go|talk|examine|take|give|use\", \"args\": [...], \"message\": \"...\"}'
    )


def _normalise_usage_tokens(usage: Any) -> Dict[str, int]:
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0}
    prompt_tokens = 0
    for key in ("prompt_tokens", "prompt_eval_count", "input_tokens", "input"):
        if (v := usage.get(key)) is not None:
            prompt_tokens = int(v)
            break
    completion_tokens = 0
    for key in ("completion_tokens", "completion_eval_count", "output_tokens", "output"):
        if (v := usage.get(key)) is not None:
            completion_tokens = int(v)
            break
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = COST_RATE_PER_1K.get(model, {"in": 0.0, "out": 0.0})
    return (prompt_tokens / 1000.0) * rates.get("in", 0.0) + (completion_tokens / 1000.0) * rates.get("out", 0.0)


def _empty_classifier_usage() -> Dict[str, Any]:
    return {
        "enabled": False,
        "display_name": None,
        "configured_model": None,
        "temperature": None,
        "calls": 0,
        "cache_hits": 0,
        "fallback_count": 0,
        "error_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "estimated_cost_usd": 0.0,
        "provider_models": [],
    }


def _empty_parsing_summary() -> Dict[str, int]:
    return {
        "total_turns": 0,
        "model_output_turns": 0,
        "api_error_fallback_turns": 0,
        "strict_json_command_count": 0,
        "text_fallback_count": 0,
        "default_look_fallback_count": 0,
        "empty_output_count": 0,
        "no_json_object_count": 0,
        "json_decode_error_count": 0,
        "json_non_dict_count": 0,
        "json_invalid_command_count": 0,
        "truncated_output_count": 0,
    }


def _resolve_request_max_tokens(extra_body: Optional[Dict[str, Any]], default: int = 240) -> int:
    if not isinstance(extra_body, dict):
        return default
    value = extra_body.get("max_tokens", default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_action_with_diagnostics(
    raw: str,
    *,
    api_error_fallback_output: bool = False,
) -> Tuple[PlayerAction, Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "model_output_available": not api_error_fallback_output,
        "api_error_fallback_output": api_error_fallback_output,
        "strict_json_command": False,
        "text_fallback_used": False,
        "default_look_used": False,
        "failure_reason": None,
        "parse_route": "json",
    }

    if api_error_fallback_output:
        diagnostics["failure_reason"] = "api_error_fallback_output"
        diagnostics["parse_route"] = "api_error_fallback"
        return PlayerAction("look"), diagnostics

    if not raw:
        diagnostics["failure_reason"] = "empty_output"
        diagnostics["default_look_used"] = True
        diagnostics["parse_route"] = "default_look"
        return parse_action_text("look"), diagnostics

    text_action = parse_action_text(raw)
    text_valid = text_action.command in MiddlehamStateMachine.command_aliases
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        diagnostics["failure_reason"] = "no_json_object"
        if text_valid:
            diagnostics["text_fallback_used"] = True
            diagnostics["parse_route"] = "text_fallback"
            return text_action, diagnostics
        diagnostics["default_look_used"] = True
        diagnostics["parse_route"] = "default_look"
        return parse_action_text("look"), diagnostics

    try:
        payload = json.loads(match.group(0))
    except Exception:
        diagnostics["failure_reason"] = "json_decode_error"
        if text_valid:
            diagnostics["text_fallback_used"] = True
            diagnostics["parse_route"] = "text_fallback"
            return text_action, diagnostics
        diagnostics["default_look_used"] = True
        diagnostics["parse_route"] = "default_look"
        return parse_action_text("look"), diagnostics

    if not isinstance(payload, dict):
        diagnostics["failure_reason"] = "json_non_dict"
        diagnostics["default_look_used"] = True
        diagnostics["parse_route"] = "default_look"
        return parse_action_text("look"), diagnostics

    json_command = " ".join(str(payload.get("command", payload.get("action", ""))).split()).lower()
    for canonical, aliases in MiddlehamStateMachine.command_aliases.items():
        if json_command in aliases:
            json_command = canonical
            break
    if json_command in MiddlehamStateMachine.command_aliases:
        diagnostics["strict_json_command"] = True
        return parse_action_json(raw), diagnostics

    diagnostics["failure_reason"] = "json_invalid_command"
    if text_valid:
        diagnostics["text_fallback_used"] = True
        diagnostics["parse_route"] = "text_fallback"
        return text_action, diagnostics
    diagnostics["default_look_used"] = True
    diagnostics["parse_route"] = "default_look"
    return parse_action_text("look"), diagnostics


def _record_parsing_diagnostics(summary: Dict[str, int], diagnostics: Dict[str, Any]) -> None:
    summary["total_turns"] += 1
    if diagnostics.get("model_output_available"):
        summary["model_output_turns"] += 1
    if diagnostics.get("api_error_fallback_output"):
        summary["api_error_fallback_turns"] += 1
    if diagnostics.get("strict_json_command"):
        summary["strict_json_command_count"] += 1
    if diagnostics.get("text_fallback_used"):
        summary["text_fallback_count"] += 1
    if diagnostics.get("default_look_used"):
        summary["default_look_fallback_count"] += 1

    failure_key_map = {
        "empty_output": "empty_output_count",
        "no_json_object": "no_json_object_count",
        "json_decode_error": "json_decode_error_count",
        "json_non_dict": "json_non_dict_count",
        "json_invalid_command": "json_invalid_command_count",
    }
    failure_key = failure_key_map.get(str(diagnostics.get("failure_reason") or ""))
    if failure_key is not None:
        summary[failure_key] += 1


def _balanced_marked_target_deck(total: int, *, shuffle_seed: int) -> List[str]:
    if total <= 0:
        return []
    candidates = ["keeper", "merchant", "peasant"]
    rng = random.Random(shuffle_seed)
    rng.shuffle(candidates)
    full_repeats, remainder = divmod(total, len(candidates))
    deck = candidates * full_repeats + candidates[:remainder]
    rng.shuffle(deck)
    return deck


def _scenario_list(seed_base: int, total: int) -> List[Dict[str, Any]]:
    if total <= 0:
        return []

    objectives = [OBJECTIVE_GAIN_WATCH_TRUST, OBJECTIVE_IDENTIFY_MARKED_CONTACT]
    base_world_count = max(1, min(POC_SCENARIOS, (total + len(objectives) - 1) // len(objectives)))
    base_scenarios = []
    scenario_index = 1
    for idx in range(base_world_count):
        seed = seed_base + idx * 97
        for objective in objectives:
            base_scenarios.append(
                {
                    "index": scenario_index,
                    "seed": seed,
                    "objective": objective,
                }
            )
            scenario_index += 1

    scenarios = []
    repeat_index = 1
    while len(scenarios) < total:
        for base in base_scenarios:
            if len(scenarios) >= total:
                break
            scenarios.append(
                {
                    **base,
                    "repeat_index": repeat_index,
                }
            )
        repeat_index += 1

    objective_shuffle_seeds = {
        OBJECTIVE_GAIN_WATCH_TRUST: seed_base + 104729,
        OBJECTIVE_IDENTIFY_MARKED_CONTACT: seed_base + 130363,
    }
    for objective_key, shuffle_seed in objective_shuffle_seeds.items():
        objective_indices = [
            idx for idx, scenario in enumerate(scenarios) if scenario["objective"] == objective_key
        ]
        deck = _balanced_marked_target_deck(len(objective_indices), shuffle_seed=shuffle_seed)
        for scenario_idx, marked_target in zip(objective_indices, deck):
            scenarios[scenario_idx]["marked_target"] = marked_target
    return scenarios


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Middleham state-machine experiment.")
    parser.add_argument("--seed-base", type=int, default=20260302)
    parser.add_argument("--output-dir", default=default_output_root())
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument(
        "--reset-interval",
        type=int,
        default=10,
        help="Ablation reset interval in turns. Use 0 to disable.",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument(
        "--classifier-key",
        default=os.environ.get("OPENROUTER_CLASSIFIER_KEY"),
        help="OpenRouter API key for the dialogue classifier. Defaults to --api-key if not set.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run without OpenRouter calls.")
    parser.add_argument("--disable-classifier", action="store_true", help="Use fallback classifier.")
    parser.add_argument("--no-score", action="store_true", help="Skip scoring pass.")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model keys to run (e.g. 'qwen_3_5,grok_4,olmo_3_1'). Default: all.",
    )
    parser.add_argument(
        "--skip-models",
        type=str,
        default=None,
        help="Comma-separated model keys to skip (e.g. 'claude_opus,gpt_5_2').",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="DIR",
        help="Resume into an existing results directory (e.g. 'results/20260303_141500'). "
             "Skips completed run files and reruns only missing or invalid runs.",
    )
    parser.add_argument(
        "--progress-dir",
        default=None,
        help="Directory for per-model status JSON and JSONL event files. Defaults to <output-root>/_progress.",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable per-turn progress status files.",
    )
    parser.add_argument(
        "--progress-stdout",
        action="store_true",
        help="Also print compact per-turn progress updates to stdout.",
    )
    return parser


def _is_complete_run_payload(data: Dict[str, Any]) -> bool:
    """Return whether a decoded run payload is complete enough to resume past."""
    transcript_events = data.get("transcript_events")
    return (
        isinstance(transcript_events, list)
        and len(transcript_events) > 0
        and "turns" in data
        and "success" in data
    )


def _run_result_from_payload(data: Dict[str, Any], model_key: str, fpath: str) -> Dict[str, Any]:
    """Convert a saved run payload into the summary row shape."""
    return {
        "model_key": data.get("model_key", model_key),
        "run_number": data.get("run_number", 0),
        "scenario_index": data.get("scenario_index", 0),
        "scenario_repeat_index": data.get("scenario_repeat_index", 0),
        "seed": data.get("seed", 0),
        "temperature": data.get("temperature", 0.3),
        "success": data.get("success", False),
        "objective": data.get("objective", ""),
        "turns_used": data.get("turns", 0),
        "objective_state": data.get("objective_state", {}),
        "output_file": fpath,
        "usage": data.get("usage", {}),
        "classifier": data.get("classifier", {}),
        "parsing": data.get("parsing", {}),
        "data_quality": data.get("data_quality", {}),
        "transcript_events": data.get("transcript_events", []),
        "score": data.get("scores", {}),
    }


def _matches_expected_run(
    data: Dict[str, Any],
    *,
    model_key: str,
    run_number: int,
    scenario: Dict[str, Any],
    temperature: float,
    max_turns: int,
    reset_interval: int,
) -> bool:
    """Return whether a saved run payload matches the current planned run."""
    try:
        stored_max_turns = data.get("max_turns")
        stored_reset_interval = data.get("reset_interval")
        max_turns_match = stored_max_turns is None or int(stored_max_turns) == max_turns
        reset_interval_match = (
            stored_reset_interval is None or int(stored_reset_interval) == reset_interval
        )
        return (
            data.get("model_key") == model_key
            and int(data.get("run_number", -1)) == run_number
            and int(data.get("scenario_index", -1)) == int(scenario["index"])
            and int(data.get("scenario_repeat_index", -1)) == int(scenario["repeat_index"])
            and int(data.get("seed", -1)) == int(scenario["seed"])
            and data.get("objective") == scenario["objective"]
            and abs(float(data.get("temperature", -999.0)) - temperature) < 1e-9
            and max_turns_match
            and reset_interval_match
        )
    except (TypeError, ValueError):
        return False


def _load_completed_run_result(
    output_file: str,
    *,
    model_key: str,
    run_number: int,
    scenario: Dict[str, Any],
    temperature: float,
    max_turns: int,
    reset_interval: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load an existing completed run if it can be safely skipped."""
    if not os.path.exists(output_file):
        return None, "missing"

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return None, "invalid_json"
    except OSError:
        return None, "unreadable"

    if not isinstance(data, dict):
        return None, "invalid_payload"
    if not _is_complete_run_payload(data):
        return None, "incomplete"
    if not _matches_expected_run(
        data,
        model_key=model_key,
        run_number=run_number,
        scenario=scenario,
        temperature=temperature,
        max_turns=max_turns,
        reset_interval=reset_interval,
    ):
        return None, "metadata_mismatch"

    return _run_result_from_payload(data, model_key, output_file), "complete"


def _count_completed_runs(model_dir: str) -> int:
    """Count valid JSON run files in a model output directory."""
    if not os.path.isdir(model_dir):
        return 0
    count = 0
    for fname in os.listdir(model_dir):
        if fname.startswith("run_") and fname.endswith(".json"):
            fpath = os.path.join(model_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and _is_complete_run_payload(data):
                    count += 1
            except (json.JSONDecodeError, OSError):
                pass
    return count


def _load_existing_results(model_dir: str, model_key: str) -> List[Dict[str, Any]]:
    """Load existing run results from a model directory for summary generation."""
    results = []
    if not os.path.isdir(model_dir):
        return results
    for fname in sorted(os.listdir(model_dir)):
        if not (fname.startswith("run_") and fname.endswith(".json")):
            continue
        fpath = os.path.join(model_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and _is_complete_run_payload(data):
                results.append(_run_result_from_payload(data, model_key, fpath))
        except (json.JSONDecodeError, OSError):
            pass
    return results


def _utc_timestamp() -> str:
    """Return a compact UTC timestamp for machine-readable progress events."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _format_progress_line(event: Dict[str, Any]) -> str:
    model = event.get("model_display") or event.get("model_key") or "model"
    run_number = event.get("run_number")
    expected_runs = event.get("expected_runs_for_model")
    turn = event.get("turn")
    max_turns = event.get("max_turns")
    phase = event.get("phase") or event.get("event")
    request_status = event.get("request_status")
    run_part = f"run {run_number}/{expected_runs}" if run_number and expected_runs else ""
    turn_part = f"turn {turn}/{max_turns}" if turn and max_turns else ""
    status_part = f" status={request_status}" if request_status else ""
    return " ".join(
        part for part in ("[progress]", str(model), run_part, turn_part, str(phase) + status_part) if part
    )


def _emit_progress(
    progress_dir: Optional[str],
    model_key: str,
    event: Dict[str, Any],
    *,
    progress_stdout: bool = False,
) -> None:
    """Write a per-model status snapshot and append a progress event."""
    payload = {
        "timestamp_utc": _utc_timestamp(),
        "pid": os.getpid(),
        **event,
    }

    if progress_stdout:
        print(_format_progress_line(payload), flush=True)

    if not progress_dir:
        return

    try:
        os.makedirs(progress_dir, exist_ok=True)
        status_file = os.path.join(progress_dir, f"{model_key}.status.json")
        tmp_file = f"{status_file}.{os.getpid()}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
        os.replace(tmp_file, status_file)

        for events_file in (
            os.path.join(progress_dir, f"{model_key}.events.jsonl"),
            os.path.join(progress_dir, "events.jsonl"),
        ):
            with open(events_file, "a", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, default=str)
                handle.write("\n")
    except OSError:
        pass


@dataclass
class RunResult:
    model_key: str
    run_number: int
    scenario_index: int
    scenario_repeat_index: int
    seed: int
    temperature: float
    success: bool
    objective: str
    turns_used: int
    objective_state: Dict[str, Any]
    output_file: str
    usage: Dict[str, Any]
    score: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "run_number": self.run_number,
            "scenario_index": self.scenario_index,
            "scenario_repeat_index": self.scenario_repeat_index,
            "seed": self.seed,
            "temperature": self.temperature,
            "success": self.success,
            "objective": self.objective,
            "turns_used": self.turns_used,
            "objective_state": self.objective_state,
            "output_file": self.output_file,
            "usage": self.usage,
            "score": self.score,
        }


def _run_single_run(
    *,
    run_number: int,
    scenario_index: int,
    scenario_repeat_index: int,
    model_key: str,
    model_name: str,
    model_display: str,
    objective_key: str,
    objective_text: str,
    seed: int,
    temperature: float,
    api_client: OpenRouterClient,
    classifier: Optional[DialogueClassifier],
    marked_target: Optional[str],
    max_turns: int,
    reset_interval: int,
    output_file: str,
    include_score: bool,
    extra_body: Optional[Dict[str, Any]] = None,
    progress_dir: Optional[str] = None,
    progress_stdout: bool = False,
    expected_runs_for_model: int = 0,
    experiment_started_at: Optional[float] = None,
) -> RunResult:
    run_started_at = time.time()
    expected_runs = expected_runs_for_model or run_number

    def emit(event: str, **fields: Any) -> None:
        payload: Dict[str, Any] = {
            "event": event,
            "phase": fields.pop("phase", event),
            "model_key": model_key,
            "model_display": model_display,
            "model_name": model_name,
            "run_number": run_number,
            "expected_runs_for_model": expected_runs,
            "temperature": temperature,
            "scenario_index": scenario_index,
            "scenario_repeat_index": scenario_repeat_index,
            "seed": seed,
            "objective": objective_key,
            "output_file": output_file,
            "max_turns": max_turns,
            "reset_interval": reset_interval,
            "run_elapsed_seconds": round(time.time() - run_started_at, 3),
        }
        if experiment_started_at is not None:
            payload["experiment_elapsed_seconds"] = round(time.time() - experiment_started_at, 3)
        payload.update(fields)
        _emit_progress(
            progress_dir,
            model_key,
            payload,
            progress_stdout=progress_stdout,
        )

    game = MiddlehamStateMachine(
        seed=seed,
        objective=objective_key,
        classifier=classifier,
        marked_target=marked_target,
        max_turns=max_turns,
        reset_interval=reset_interval,
    )
    if classifier is not None:
        classifier.reset_run_state()

    transcript_events: List[Dict[str, Any]] = []
    api_errors: List[Dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    estimated_cost = 0.0
    parsing_summary = _empty_parsing_summary()
    recent_history: List[Dict[str, str]] = []

    emit("run_started", phase="initializing_run", turn=0)

    while not game.is_complete() and game.turn < max_turns:
        turn = game.turn + 1
        context = game.render_context()
        prompt = _build_prompt(turn, context, objective_text, recent_history)
        api_error_fallback_output = False
        finish_reason = ""
        request_max_tokens = _resolve_request_max_tokens(extra_body)
        request_attempts = 1
        request_status = "success"
        request_error: Optional[str] = None
        provider_model = model_name

        request_started_at = time.time()
        emit(
            "request_started",
            phase="waiting_for_primary_model",
            turn=turn,
            prompt_chars=len(prompt),
            recent_history_entries=len(recent_history),
            request_max_tokens=request_max_tokens,
        )

        try:
            completion = api_client.chat_completion(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an adventurer in a simulated MUD test world."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=request_max_tokens,
                extra_body=extra_body,
            )
            raw_output = completion.content or ""
            usage = _normalise_usage_tokens(completion.usage)
            request_attempts = int(getattr(completion, "attempts", 1) or 1)
            provider_model = str(getattr(completion, "model", model_name) or model_name)
            finish_reason = str(
                ((completion.raw.get("choices") or [{}])[0].get("finish_reason") or "")
            )
        except OpenRouterError as exc:
            raw_output = ""
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            request_attempts = int(getattr(exc, "attempts", 1) or 1)
            request_status = "api_error_fallback"
            request_error = str(exc)
            provider_model = model_name
            api_errors.append(
                {
                    "turn": turn,
                    "status": request_status,
                    "attempts": request_attempts,
                    "retryable": bool(getattr(exc, "retryable", False)),
                    "status_code": getattr(exc, "status_code", None),
                    "error": request_error,
                }
            )
            api_error_fallback_output = True
        except Exception as exc:
            raw_output = ""
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            request_attempts = 1
            request_status = "unexpected_error_fallback"
            request_error = str(exc)
            provider_model = model_name
            api_errors.append(
                {
                    "turn": turn,
                    "status": request_status,
                    "attempts": request_attempts,
                    "retryable": False,
                    "status_code": None,
                    "error": request_error,
                }
            )
            api_error_fallback_output = True

        request_elapsed = time.time() - request_started_at
        emit(
            "request_finished",
            phase="primary_model_response_received",
            turn=turn,
            request_status=request_status,
            request_attempts=request_attempts,
            request_elapsed_seconds=round(request_elapsed, 3),
            request_max_tokens=request_max_tokens,
            finish_reason=finish_reason,
            provider_model=provider_model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            raw_output_chars=len(raw_output),
            request_error=request_error[:500] if request_error else None,
        )

        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        estimated_cost += _estimate_cost(model_name, usage["prompt_tokens"], usage["completion_tokens"])

        action, parsing_diagnostics = _parse_action_with_diagnostics(
            raw_output,
            api_error_fallback_output=api_error_fallback_output,
        )
        _record_parsing_diagnostics(parsing_summary, parsing_diagnostics)
        if finish_reason == "length":
            parsing_summary["truncated_output_count"] += 1

        emit(
            "world_step_started",
            phase="applying_action",
            turn=turn,
            action_command=action.command,
            action_args=list(action.args),
            parse_route=parsing_diagnostics.get("parse_route"),
        )
        record = game.step(action, raw_output)

        # Append this turn to the rolling history window so the model can see
        # what it has already tried when building the next prompt.
        history_action = record.player_command
        if api_error_fallback_output:
            history_action = f"{record.player_command} [api_error_fallback]"
        recent_history.append({
            "turn": str(record.turn),
            "action": history_action,
            "result": record.narration[:200],
        })
        if len(recent_history) > HISTORY_WINDOW:
            recent_history = recent_history[-HISTORY_WINDOW:]

        if not isinstance(record.state_delta, dict):
            record.state_delta = {}
        record.state_delta.setdefault(
            "model_usage",
            {
                "model": model_name,
                "temperature": temperature,
                "provider_model": provider_model,
                "request_status": request_status,
                "attempts": request_attempts,
                "max_tokens": request_max_tokens,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
            },
        )

        transcript_events.append(
            {
                "turn": record.turn,
                "action": {
                    "command": record.parsed_action.get("command", "look"),
                    "args": record.parsed_action.get("args", []),
                    "message": record.parsed_action.get("message", ""),
                },
                "raw_model_output": raw_output,
                "model_request": {
                    "status": request_status,
                    "attempts": request_attempts,
                    "max_tokens": request_max_tokens,
                    "finish_reason": finish_reason,
                    "provider_model": provider_model,
                    "error": request_error,
                },
                "pre_room": record.pre_room,
                "post_room": record.post_room,
                "valid": record.valid,
                "narration": record.narration,
                "dialogue_signal": record.dialogue_signal,
                "npc_reactions": record.npc_reactions,
                "parsing": dict(parsing_diagnostics),
            }
        )

        emit(
            "turn_completed",
            phase="turn_completed",
            turn=record.turn,
            action_command=record.parsed_action.get("command", "look"),
            action_args=record.parsed_action.get("args", []),
            valid=record.valid,
            pre_room=record.pre_room,
            post_room=record.post_room,
            success_so_far=game.objective_complete(),
            turns_remaining=max(0, max_turns - game.turn),
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            estimated_cost_usd=round(estimated_cost, 6),
            api_error_count=len(api_errors),
        )

    run_payload = game.export_run()
    run_payload["run_number"] = run_number
    run_payload["scenario_index"] = scenario_index
    run_payload["scenario_repeat_index"] = scenario_repeat_index
    run_payload["model_key"] = model_key
    run_payload["model_name"] = model_name
    run_payload["seed"] = seed
    run_payload["temperature"] = temperature
    run_payload["objective_text"] = objective_text
    run_payload["api_errors"] = api_errors
    classifier_usage = classifier.usage_snapshot() if classifier is not None else _empty_classifier_usage()
    classifier_cost = float(classifier_usage.get("estimated_cost_usd", 0.0))
    usage_summary = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "primary_model_prompt_tokens": total_prompt_tokens,
        "primary_model_completion_tokens": total_completion_tokens,
        "classifier_prompt_tokens": int(classifier_usage.get("prompt_tokens", 0)),
        "classifier_completion_tokens": int(classifier_usage.get("completion_tokens", 0)),
        "primary_model_estimated_cost_usd": estimated_cost,
        "classifier_estimated_cost_usd": classifier_cost,
        "estimated_cost_usd": estimated_cost + classifier_cost,
    }
    run_payload["usage"] = usage_summary
    run_payload["classifier"] = classifier_usage
    run_payload["parsing"] = parsing_summary
    run_payload["data_quality"] = {
        "has_api_error_fallbacks": bool(parsing_summary.get("api_error_fallback_turns", 0)),
        "api_error_fallback_turns": int(parsing_summary.get("api_error_fallback_turns", 0)),
        "api_error_count": len(api_errors),
        "truncated_output_count": int(parsing_summary.get("truncated_output_count", 0)),
    }
    run_payload["transcript_events"] = transcript_events
    run_payload["max_turns"] = max_turns
    run_payload["reset_interval"] = reset_interval

    scores = {} if not include_score else score_run(run_payload)
    run_payload["scores"] = scores

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(run_payload, handle, ensure_ascii=False, indent=2)

    emit(
        "run_completed",
        phase="run_completed",
        turn=run_payload["turns"],
        success=bool(run_payload["success"]),
        turns_used=run_payload["turns"],
        total_prompt_tokens=usage_summary["prompt_tokens"],
        total_completion_tokens=usage_summary["completion_tokens"],
        estimated_cost_usd=round(float(usage_summary["estimated_cost_usd"]), 6),
        output_file=output_file,
        api_error_count=len(api_errors),
    )

    return RunResult(
        model_key=model_key,
        run_number=run_number,
        scenario_index=scenario_index,
        scenario_repeat_index=scenario_repeat_index,
        seed=seed,
        temperature=temperature,
        success=bool(run_payload["success"]),
        objective=objective_key,
        turns_used=run_payload["turns"],
        objective_state=run_payload["objective_state"],
        output_file=output_file,
        usage=usage_summary,
        score=scores,
    )


def _aggregate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    scored = [r for r in results if r.get("score")]
    avg = lambda key: sum(float(r.get("score", {}).get(key, 0.0)) for r in scored) / len(scored) if scored else 0.0
    classifier_fallback_rate = sum(
        (
            float(r.get("classifier", {}).get("fallback_count", 0))
            / max(1, int(r.get("classifier", {}).get("calls", 0)))
        )
        if r.get("classifier", {}).get("enabled")
        else 0.0
        for r in results
    ) / len(results)
    classifier_error_rate = sum(
        (
            float(r.get("classifier", {}).get("error_count", 0))
            / max(1, int(r.get("classifier", {}).get("calls", 0)))
        )
        if r.get("classifier", {}).get("enabled")
        else 0.0
        for r in results
    ) / len(results)
    classifier_runs = [r for r in results if isinstance(r.get("classifier"), dict) and r.get("classifier")]
    classifier_calls = sum(int(r.get("classifier", {}).get("calls", 0)) for r in classifier_runs)
    classifier_cache_hits = sum(int(r.get("classifier", {}).get("cache_hits", 0)) for r in classifier_runs)
    classifier_fallback_count = sum(int(r.get("classifier", {}).get("fallback_count", 0)) for r in classifier_runs)
    classifier_error_count = sum(int(r.get("classifier", {}).get("error_count", 0)) for r in classifier_runs)
    classifier_health = {
        "runs_with_diagnostics": len(classifier_runs),
        "enabled_runs": sum(1 for r in classifier_runs if r.get("classifier", {}).get("enabled")),
        "configured_models": sorted(
            {
                str(r.get("classifier", {}).get("configured_model"))
                for r in classifier_runs
                if r.get("classifier", {}).get("configured_model")
            }
        ),
        "provider_models": sorted(
            {
                str(provider_model)
                for r in classifier_runs
                for provider_model in r.get("classifier", {}).get("provider_models", [])
                if provider_model
            }
        ),
        "calls": classifier_calls,
        "cache_hits": classifier_cache_hits,
        "cache_hit_rate": classifier_cache_hits / classifier_calls if classifier_calls else 0.0,
        "fallback_count": classifier_fallback_count,
        "fallback_rate": classifier_fallback_count / classifier_calls if classifier_calls else 0.0,
        "error_count": classifier_error_count,
        "error_rate": classifier_error_count / classifier_calls if classifier_calls else 0.0,
    }
    parsing_runs = [r for r in results if isinstance(r.get("parsing"), dict) and r.get("parsing")]
    total_turns = sum(int(r.get("parsing", {}).get("total_turns", 0)) for r in parsing_runs)
    model_output_turns = sum(int(r.get("parsing", {}).get("model_output_turns", 0)) for r in parsing_runs)
    api_error_fallback_turns = sum(int(r.get("parsing", {}).get("api_error_fallback_turns", 0)) for r in parsing_runs)
    strict_json_command_count = sum(int(r.get("parsing", {}).get("strict_json_command_count", 0)) for r in parsing_runs)
    text_fallback_count = sum(int(r.get("parsing", {}).get("text_fallback_count", 0)) for r in parsing_runs)
    default_look_fallback_count = sum(int(r.get("parsing", {}).get("default_look_fallback_count", 0)) for r in parsing_runs)
    empty_output_count = sum(int(r.get("parsing", {}).get("empty_output_count", 0)) for r in parsing_runs)
    no_json_object_count = sum(int(r.get("parsing", {}).get("no_json_object_count", 0)) for r in parsing_runs)
    json_decode_error_count = sum(int(r.get("parsing", {}).get("json_decode_error_count", 0)) for r in parsing_runs)
    json_non_dict_count = sum(int(r.get("parsing", {}).get("json_non_dict_count", 0)) for r in parsing_runs)
    json_invalid_command_count = sum(int(r.get("parsing", {}).get("json_invalid_command_count", 0)) for r in parsing_runs)
    truncated_output_count = sum(int(r.get("parsing", {}).get("truncated_output_count", 0)) for r in parsing_runs)
    json_parse_failure_count = max(0, model_output_turns - strict_json_command_count)
    api_error_run_count = sum(
        1
        for r in parsing_runs
        if int(r.get("parsing", {}).get("api_error_fallback_turns", 0)) > 0
    )
    json_parse_reliability = {
        "runs_with_diagnostics": len(parsing_runs),
        "total_turns": total_turns,
        "model_output_turns": model_output_turns,
        "api_error_fallback_turns": api_error_fallback_turns,
        "api_error_fallback_turn_rate": api_error_fallback_turns / total_turns if total_turns else 0.0,
        "strict_json_command_count": strict_json_command_count,
        "strict_json_command_rate": strict_json_command_count / model_output_turns if model_output_turns else 0.0,
        "json_parse_failure_count": json_parse_failure_count,
        "json_parse_failure_rate": json_parse_failure_count / model_output_turns if model_output_turns else 0.0,
        "text_fallback_count": text_fallback_count,
        "text_fallback_rate": text_fallback_count / model_output_turns if model_output_turns else 0.0,
        "default_look_fallback_count": default_look_fallback_count,
        "default_look_fallback_rate": default_look_fallback_count / model_output_turns if model_output_turns else 0.0,
        "empty_output_count": empty_output_count,
        "no_json_object_count": no_json_object_count,
        "json_decode_error_count": json_decode_error_count,
        "json_non_dict_count": json_non_dict_count,
        "json_invalid_command_count": json_invalid_command_count,
        "truncated_output_count": truncated_output_count,
        "truncated_output_rate": truncated_output_count / model_output_turns if model_output_turns else 0.0,
    }
    return {
        "runs": len(results),
        "success_rate": sum(1 for r in results if r.get("success")) / len(results),
        "avg_turns_used": sum(int(r.get("turns_used", 0)) for r in results) / len(results),
        "avg_goal_pursuit": avg("goal_pursuit"),
        "avg_social_adaptation": avg("social_adaptation"),
        "avg_world_grounding": avg("world_grounding"),
        "avg_strategic_sophistication": avg("strategic_sophistication"),
        "avg_cost_usd": sum(float(r.get("usage", {}).get("estimated_cost_usd", 0.0)) for r in results) / len(results),
        "scored_runs": len(scored),
        "mean_run_fallback_rate": classifier_fallback_rate,
        "mean_run_error_rate": classifier_error_rate,
        "api_error_run_rate": api_error_run_count / len(parsing_runs) if parsing_runs else 0.0,
        "classifier_health": classifier_health,
        "json_parse_reliability": json_parse_reliability,
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.dry_run and not args.api_key:
        raise SystemExit("OPENROUTER_API_KEY or --api-key is required unless --dry-run is set.")

    # --- Model filtering ---
    all_models = default_models()
    include_keys = set(k.strip() for k in args.models.split(",")) if args.models else None
    exclude_keys = set(k.strip() for k in args.skip_models.split(",")) if args.skip_models else set()

    if include_keys:
        valid_keys = {m.key for m in all_models}
        bad = include_keys - valid_keys
        if bad:
            raise SystemExit(f"Unknown model keys in --models: {bad}. Valid: {sorted(valid_keys)}")
        models = tuple(m for m in all_models if m.key in include_keys)
    else:
        models = tuple(m for m in all_models if m.key not in exclude_keys)

    if not models:
        raise SystemExit("No models selected after filtering. Check --models / --skip-models.")

    # --- Output directory (new or resume) ---
    if args.resume:
        output_root = args.resume
        if not os.path.isdir(output_root):
            raise SystemExit(f"--resume directory does not exist: {output_root}")
        run_timestamp = os.path.basename(output_root)
        print(f"Resuming into: {output_root}", flush=True)
    else:
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_root = os.path.join(args.output_dir, run_timestamp)
        os.makedirs(output_root, exist_ok=True)

    progress_dir = None
    progress_stdout = bool(args.progress_stdout)
    if not args.disable_progress:
        progress_dir = args.progress_dir or os.path.join(output_root, "_progress")
        os.makedirs(progress_dir, exist_ok=True)
        print(f"Progress: {progress_dir}", flush=True)
    else:
        progress_stdout = False

    max_runs_per_model = max(sum(n for _, n in m.run_plan) for m in models)
    scenarios = _scenario_list(args.seed_base, total=max_runs_per_model)

    api_client: OpenRouterClient
    if args.dry_run:
        class _DryClient:
            model: str = "dry_run"

            def chat_completion(self, *_args, **_kwargs):
                return type(
                    "_DryResponse",
                    (),
                    {
                        "content": "{\"command\":\"look\"}",
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                        "model": "dry_run",
                        "attempts": 1,
                        "raw": {"choices": [{"finish_reason": "stop"}]},
                    },
                )()

        api_client = _DryClient()  # type: ignore[assignment]
    else:
        api_client = OpenRouterClient(args.api_key)

    classifier = None
    if not args.disable_classifier and not args.dry_run:
        classifier = DialogueClassifier(api_key=args.classifier_key or args.api_key)

    results: List[Dict[str, Any]] = []
    objective_map = objective_prompts()
    total_runs = sum(sum(n for _, n in m.run_plan) for m in models)
    skipped_runs = 0
    processed_runs = 0
    new_runs = 0
    experiment_start = time.time()

    for model in models:
        model_dir = os.path.join(output_root, model.key)
        expected_runs = sum(n for _, n in model.run_plan)

        # --- Resume: completed files are skipped per run below. ---
        whole_model_resume = False
        if args.resume and whole_model_resume:
            completed = _count_completed_runs(model_dir)
            if completed >= expected_runs:
                print(f"[skip] {model.display} — {completed}/{expected_runs} runs already complete", flush=True)
                skipped_runs += expected_runs
                processed_runs += expected_runs
                # Load existing results for summary
                results.extend(_load_existing_results(model_dir, model.key))
                continue
            elif completed > expected_runs:
                print(f"[partial] {model.display} — {completed}/{expected_runs} runs found, re-running all", flush=True)

        os.makedirs(model_dir, exist_ok=True)
        run_num = 1

        for temperature, run_count in model.run_plan:
            for _idx in range(run_count):
                scenario = scenarios[run_num - 1]
                objective_key = scenario["objective"]
                objective_text = objective_map.get(objective_key, "")
                temp_tag = str(temperature).replace(".", "p")
                output_file = os.path.join(model_dir, f"run_{run_num:02d}_t{temp_tag}.json")

                if args.resume:
                    existing_result, resume_reason = _load_completed_run_result(
                        output_file,
                        model_key=model.key,
                        run_number=run_num,
                        scenario=scenario,
                        temperature=temperature,
                        max_turns=args.max_turns,
                        reset_interval=args.reset_interval,
                    )
                    if existing_result is not None:
                        skipped_runs += 1
                        processed_runs += 1
                        results.append(existing_result)
                        _emit_progress(
                            progress_dir,
                            model.key,
                            {
                                "event": "run_skipped",
                                "phase": "skipped_completed_run",
                                "model_key": model.key,
                                "model_display": model.display,
                                "model_name": model.openrouter_model,
                                "run_number": run_num,
                                "expected_runs_for_model": expected_runs,
                                "temperature": temperature,
                                "scenario_index": int(scenario["index"]),
                                "scenario_repeat_index": int(scenario["repeat_index"]),
                                "seed": int(scenario["seed"]),
                                "objective": objective_key,
                                "output_file": output_file,
                                "max_turns": args.max_turns,
                                "reset_interval": args.reset_interval,
                                "experiment_elapsed_seconds": round(time.time() - experiment_start, 3),
                            },
                            progress_stdout=progress_stdout,
                        )
                        run_num += 1
                        continue
                    if resume_reason != "missing":
                        print(
                            f"[resume] {model.display} run {run_num:02d} is {resume_reason}; rerunning",
                            flush=True,
                        )

                processed_runs += 1
                new_runs += 1
                progress = processed_runs
                print(
                    f"[{progress}/{total_runs}] {model.display} | scenario {scenario['index']} rep {scenario['repeat_index']} | "
                    f"seed {scenario['seed']} | {objective_key}",
                    flush=True,
                )
                result = _run_single_run(
                    run_number=run_num,
                    scenario_index=int(scenario["index"]),
                    scenario_repeat_index=int(scenario["repeat_index"]),
                    model_key=model.key,
                    model_name=model.openrouter_model,
                    model_display=model.display,
                    objective_key=objective_key,
                    objective_text=objective_text,
                    seed=scenario["seed"],
                    temperature=temperature,
                    api_client=api_client,
                    classifier=classifier,
                    marked_target=scenario.get("marked_target"),
                    max_turns=args.max_turns,
                    reset_interval=args.reset_interval,
                    output_file=output_file,
                    include_score=not args.no_score,
                    extra_body=dict(model.extra_body) if model.extra_body else None,
                    progress_dir=progress_dir,
                    progress_stdout=progress_stdout,
                    expected_runs_for_model=expected_runs,
                    experiment_started_at=experiment_start,
                )

                elapsed_total = time.time() - experiment_start
                avg_secs = elapsed_total / max(1, new_runs)
                eta_secs = avg_secs * (total_runs - processed_runs)
                cost = result.usage.get("estimated_cost_usd", 0.0)
                print(f"       done in {result.turns_used}t | cost ${cost:.4f} | ETA {max(0, eta_secs) / 60:.1f}m remaining", flush=True)

                serialized = result.to_dict()
                results.append(serialized)
                run_num += 1

    # --- Summary: include ALL models in output_root (even from other parallel runs) ---
    all_models_for_summary = default_models()
    model_breakdown: Dict[str, Any] = {}
    all_results_for_summary: List[Dict[str, Any]] = []
    for model in all_models_for_summary:
        m_dir = os.path.join(output_root, model.key)
        # Prefer loading from disk so parallel processes see each other's results
        if os.path.isdir(m_dir):
            loaded = _load_existing_results(m_dir, model.key)
            model_breakdown[model.key] = _aggregate_summary(loaded)
            all_results_for_summary.extend(loaded)
        else:
            model_breakdown[model.key] = _aggregate_summary([])

    summary = {
        "generated": run_timestamp,
        "seed_base": args.seed_base,
        "max_turns": args.max_turns,
        "reset_interval": args.reset_interval,
        "total_runs": len(all_results_for_summary),
        "models": model_breakdown,
        "overall": _aggregate_summary(all_results_for_summary),
    }

    summary_file = os.path.join(output_root, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Completed {new_runs} new runs ({skipped_runs} skipped). Output: {output_root}")
    print(f"Summary: {summary_file} ({len(all_results_for_summary)} total runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
