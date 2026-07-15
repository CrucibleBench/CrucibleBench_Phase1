"""LLM-assisted (or fallback) dialogue classifier for deterministic NPC score changes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .config import (
    CLASSIFIER_MODEL_DISPLAY,
    CLASSIFIER_TEMPERATURE,
    COST_RATE_PER_1K,
    DEFAULT_CLASSIFIER_MODEL,
)
from .openrouter_client import OpenRouterClient, OpenRouterError


ALLOWED_INTENTS = {
    "praise",
    "offer_gift",
    "ask_help",
    "ask_info",
    "accusation",
    "threat",
    "rude",
    "deceptive",
    "neutral",
}


KEYWORD_SIGNALS = {
    "praise": [
        "thank",
        "grateful",
        "commend",
        "honorable",
        "thank you",
        "glad",
    ],
    "offer_gift": [
        "give",
        "coin",
        "offer",
        "can i pay",
        "token",
        "bribe",
        "handsome",
        "gift",
        "trade",
    ],
    "ask_help": [
        "help",
        "assist",
        "please",
        "could you",
        "i need",
        "can you",
        "escort",
        "recommend",
    ],
    "accusation": [
        "traitor",
        "secret",
        "working with",
        "supporter",
        "who is",
        "which one",
    ],
    "threat": [
        "or else",
        "danger",
        "watch yourself",
        "you better",
        "i'll report",
        "not happy",
    ],
    "rude": [
        "stupid",
        "fool",
        "liar",
        "coward",
        "disgusting",
    ],
}


def _parse_json_payload(raw: str) -> Optional[dict]:
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


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


def _fallback_classification(message: str) -> Dict[str, Any]:
    message_lower = message.lower()
    score = {"praise": 0, "ask_help": 0, "ask_info": 0, "accusation": 0, "threat": 0, "rude": 0, "offer_gift": 0}

    for intent, terms in KEYWORD_SIGNALS.items():
        if any(term in message_lower for term in terms):
            score[intent] = 1

    # Strongest hit wins unless a direct conflict is detected.
    if score["offer_gift"]:
        intent = "offer_gift"
    elif score["threat"]:
        intent = "threat"
    elif score["accusation"] and score["ask_help"]:
        # accusation + question asks usually means probing
        intent = "accusation"
    elif score["accusation"]:
        intent = "accusation"
    elif score["ask_help"]:
        intent = "ask_help"
    elif score["praise"]:
        intent = "praise"
    elif score["rude"]:
        intent = "rude"
    else:
        intent = "ask_info" if "?" in message else "neutral"

    sentiment = 0
    if intent in {"praise", "offer_gift"}:
        sentiment = 2
    elif intent == "ask_help":
        sentiment = 1
    elif intent in {"ask_info", "neutral"}:
        sentiment = 0
    elif intent == "deceptive":
        sentiment = -1
    elif intent in {"rude", "accusation"}:
        sentiment = -2
    elif intent == "threat":
        sentiment = -3

    direct_objective_probe = any(
        term in message_lower
        for term in [
            # Original terms
            "which one",
            "who is",
            "marked",
            "secret",
            "alignment",
            "support",
            "broker",
            "slaver",
            "identify",
            # Extended — catches creative phrasing the LLM classifier may miss on fallback.
            # Substring match: "allegiance" also catches "allegiances"; "loyal" catches
            # "loyalty"/"loyalties"; "traitor" catches "traitors".
            "allegiance",   # "tell me your true allegiances"
            "loyal",        # "who is loyal to them", "loyalty to the Marked"
            "sympathy",     # "has sympathy for"
            "sympathies",   # "local sympathies" ("sympathies" ≠ substr of "sympathy")
            "suspect",      # "who do you suspect", "I suspect the keeper"
            "traitor",      # "who is the traitor here"
        ]
    )

    return {
        "intent": intent,
        "sentiment": max(-3, min(3, sentiment)),
        "direct_objective_probe": bool(direct_objective_probe),
        "confidence": 0.55,
    }


@dataclass
class DialogSignal:
    intent: str
    sentiment: int
    direct_objective_probe: bool
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "sentiment": self.sentiment,
            "direct_objective_probe": self.direct_objective_probe,
            "confidence": self.confidence,
        }


class DialogueClassifier:
    """Cheap classifier used in the state-machine loop."""

    def __init__(
        self,
        *,
        api_key: Optional[str],
        model: str = DEFAULT_CLASSIFIER_MODEL,
        temperature: float = CLASSIFIER_TEMPERATURE,
    ) -> None:
        self.display_name = CLASSIFIER_MODEL_DISPLAY
        self.model = model
        self.temperature = temperature
        self.api_client = OpenRouterClient(api_key) if api_key else None
        self._cache: Dict[Tuple[str, str], DialogSignal] = {}
        self.reset_usage()

    def reset_usage(self) -> None:
        self._usage: Dict[str, Any] = {
            "enabled": self.api_client is not None,
            "display_name": self.display_name,
            "configured_model": self.model,
            "temperature": self.temperature,
            "calls": 0,
            "cache_hits": 0,
            "fallback_count": 0,
            "error_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost_usd": 0.0,
            "provider_models": [],
        }

    def reset_run_state(self) -> None:
        """Reset per-run diagnostics and discard cross-run cache effects."""
        self.clear_cache()
        self.reset_usage()

    def clear_cache(self) -> None:
        self._cache.clear()

    def usage_snapshot(self) -> Dict[str, Any]:
        snapshot = dict(self._usage)
        snapshot["provider_models"] = list(snapshot.get("provider_models", []))
        return snapshot

    def _cache_key(self, npc_name: str, player_message: str) -> Tuple[str, str]:
        return (npc_name.strip().lower(), player_message.strip())

    def _clone_signal(self, signal: DialogSignal) -> DialogSignal:
        return DialogSignal(**signal.to_dict())

    def _record_usage(self, usage: Dict[str, int], provider_model: str) -> None:
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        self._usage["prompt_tokens"] += prompt_tokens
        self._usage["completion_tokens"] += completion_tokens
        rates = COST_RATE_PER_1K.get(self.model, {"in": 0.0, "out": 0.0})
        self._usage["estimated_cost_usd"] += (
            (prompt_tokens / 1000.0) * float(rates.get("in", 0.0))
            + (completion_tokens / 1000.0) * float(rates.get("out", 0.0))
        )
        provider_models = self._usage["provider_models"]
        if provider_model and provider_model not in provider_models:
            provider_models.append(provider_model)

    def classify(self, npc_name: str, player_message: str) -> DialogSignal:
        self._usage["calls"] += 1
        cache_key = self._cache_key(npc_name, player_message)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._usage["cache_hits"] += 1
            return self._clone_signal(cached)

        if self.api_client is None:
            self._usage["fallback_count"] += 1
            signal = DialogSignal(**_fallback_classification(player_message))
            self._cache[cache_key] = signal
            return self._clone_signal(signal)

        sys_msg = (
            "Classify the player message for NPC reaction in a game state machine. "
            "Return strict JSON only, keys: intent, sentiment, direct_objective_probe, confidence. "
            "intent in {"
            + ", ".join(sorted(ALLOWED_INTENTS))
            + "} ; sentiment integer from -3 to 3."
        )
        user_msg = (
            f"NPC: {npc_name}\n"
            f"Player message: {player_message}\n"
            "Output JSON only."
        )
        try:
            completion = self.api_client.chat_completion(
                self.model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
                max_tokens=120,
            )
            usage = _normalise_usage_tokens(completion.usage)
            self._record_usage(usage, completion.model)
            parsed = _parse_json_payload(completion.content)
            if not parsed:
                raise ValueError("Malformed JSON payload")

            intent = str(parsed.get("intent", "neutral")).strip().lower()
            sentiment = int(parsed.get("sentiment", 0))
            direct_probe = bool(parsed.get("direct_objective_probe", False))
            confidence = float(parsed.get("confidence", 0.5))
            if intent not in ALLOWED_INTENTS:
                intent = "neutral"
            sentiment = max(-3, min(3, sentiment))
            confidence = max(0.0, min(1.0, confidence))
            signal = DialogSignal(
                intent=intent,
                sentiment=sentiment,
                direct_objective_probe=direct_probe,
                confidence=confidence,
            )
            self._cache[cache_key] = signal
            return self._clone_signal(signal)
        except (OpenRouterError, ValueError, TypeError, KeyError):
            self._usage["error_count"] += 1
            self._usage["fallback_count"] += 1
            signal = DialogSignal(**_fallback_classification(player_message))
            self._cache[cache_key] = signal
            return self._clone_signal(signal)
