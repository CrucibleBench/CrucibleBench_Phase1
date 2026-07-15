"""State-machine simulator for the 30-day Middleham proof-of-concept."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .classifier import DialogSignal, DialogueClassifier, _fallback_classification
from .config import (
    OBJECTIVE_GAIN_WATCH_TRUST,
    OBJECTIVE_IDENTIFY_MARKED_CONTACT,
)


def _clamp(value: int, minimum: int = 0, maximum: int = 100) -> int:
    return max(minimum, min(maximum, value))


def _normalize_name(token: str) -> str:
    return re.sub(r"\s+", " ", token.strip().lower())


def _normalize_direction(token: str) -> str:
    aliases = {
        "n": "north",
        "north": "north",
        "s": "south",
        "south": "south",
        "e": "east",
        "east": "east",
        "w": "west",
        "west": "west",
        "d": "down",
        "down": "down",
        "u": "up",
        "up": "up",
    }
    return aliases.get(token.lower(), token.lower())


def _is_watch_recommendation_attempt(message: str) -> bool:
    normalized = " ".join(message.lower().split())
    explicit_request_patterns = (
        "recommend me",
        "recommend my application",
        "recommend me to the watch",
        "recommend me for the watch",
        "sponsor me",
        "sponsor my application",
        "vouch for me",
        "endorse me",
        "put in a word for me",
        "back my application",
        "support my application",
    )
    return any(pattern in normalized for pattern in explicit_request_patterns)


@dataclass
class PlayerAction:
    command: str
    args: Tuple[str, ...] = ()
    message: str = ""


@dataclass
class NPCTemplate:
    key: str
    name: str
    role: str
    demeanor: str
    base_trust: int
    base_suspicion: int
    base_dialogue: str
    loyalty_note: str
    investigation_hint: str


@dataclass
class NPCState:
    template: NPCTemplate
    trust: int
    suspicion: int
    alignment_marked: bool = False
    clues: List[str] = field(default_factory=list)
    talk_count: int = 0

    def apply_signal(self, signal: DialogSignal) -> Dict[str, int]:
        trust_delta = signal.sentiment
        suspicion_delta = 0
        if signal.direct_objective_probe:
            suspicion_delta += 2
        if signal.intent == "offer_gift":
            trust_delta += 2
        elif signal.intent in {"threat", "rude", "accusation"}:
            trust_delta -= 1
            suspicion_delta += 2
        elif signal.intent == "deceptive":
            trust_delta -= 1
            suspicion_delta += 1

        before = (self.trust, self.suspicion)
        self.trust = _clamp(self.trust + trust_delta)
        self.suspicion = _clamp(self.suspicion + suspicion_delta)
        self.talk_count += 1
        return {
            "trust_delta": self.trust - before[0],
            "suspicion_delta": self.suspicion - before[1],
        }


@dataclass
class TurnRecord:
    turn: int
    player_command: str
    parsed_action: Dict[str, Any]
    raw_model_output: str
    pre_room: str
    post_room: str
    narration: str
    valid: bool
    dialogue_signal: Optional[Dict[str, Any]] = None
    npc_reactions: List[Dict[str, Any]] = field(default_factory=list)
    state_delta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# World template data — module-level constants, never mutated at runtime.
# MiddlehamStateMachine deep-copies rooms per instance via _clone_rooms().
# NPC, alias, and command lookups reference these directly (read-only).
# ---------------------------------------------------------------------------

_ROOM_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "city_gate": {
        "name": "Middleham City Gate",
        "desc": "A heavy city gate marks the border. Patrol banners flutter.",
        "exits": {"north": "main_square"},
        "npcs": [],
        "items": ["guard_token", "old_map"],
    },
    "main_square": {
        "name": "Middleham Main Square",
        "desc": "A civic square with notices and loud vendor calls.",
        "exits": {"south": "city_gate", "north": "guard_barracks", "east": "market_street", "west": "tavern"},
        "npcs": [],
        "items": ["street_crystal"],
    },
    "guard_barracks": {
        "name": "Guard Barracks Court",
        "desc": "Barracks and a command circle, with law posted everywhere.",
        "exits": {"south": "main_square", "east": "residential_street", "north": "guild_court"},
        "npcs": ["captain"],
        "items": ["signet_ring"],
    },
    "guild_court": {
        "name": "Guild Court",
        "desc": "A narrow court where tariffs and petitions stack high.",
        "exits": {"south": "guard_barracks", "east": "outskirt_road"},
        "npcs": [],
        "items": ["guild_coin"],
    },
    "market_street": {
        "name": "Merchant Quarter",
        "desc": "Crowded stalls with grain, cloth, and tools.",
        "exits": {"west": "main_square", "north": "merchant_hall", "east": "temple_entry"},
        "npcs": ["merchant"],
        "items": ["tariff_letter"],
    },
    "merchant_hall": {
        "name": "Merchant Hall",
        "desc": "A cramped counting room with old ledgers.",
        "exits": {"south": "market_street", "east": "outskirt_road"},
        "npcs": [],
        "items": ["sealed_letter"],
    },
    "tavern": {
        "name": "Tarnished Lantern Tavern",
        "desc": "Smoke, low voices, and quick rumors.",
        "exits": {"east": "main_square", "north": "temple_entry"},
        "npcs": ["keeper"],
        "items": ["rumor_scroll"],
    },
    "temple_entry": {
        "name": "Temple Steps",
        "desc": "Stone steps and quiet argument beneath a lit shrine.",
        "exits": {"west": "market_street", "south": "tavern", "north": "temple_inner"},
        "npcs": [],
        "items": ["prayer_beads", "temple_pass"],
    },
    "temple_inner": {
        "name": "Temple Inner Court",
        "desc": "A quiet brazierside court with marked circles in ash.",
        "exits": {"south": "temple_entry"},
        "npcs": ["peasant"],
        "items": ["altar_chalk"],
    },
    "residential_street": {
        "name": "Residential Street",
        "desc": "Narrow homes with shutters and narrow sight lines.",
        "exits": {"west": "guard_barracks", "north": "temple_entry", "east": "outskirt_road"},
        "npcs": [],
        "items": ["cloth_scarf"],
    },
    "outskirt_road": {
        "name": "Road to the Outskirts",
        "desc": "A shallow ditch and dark tree line.",
        "exits": {"west": "residential_street", "south": "forest_rim", "north": "merchant_hall", "east": "forest_rim"},
        "npcs": [],
        "items": ["rusted_blade"],
    },
    "forest_rim": {
        "name": "Forest Rim",
        "desc": "Tall pines and a press of silence.",
        "exits": {"north": "outskirt_road", "south": "city_gate"},
        "npcs": [],
        "items": ["charcoal_stone"],
    },
}

_ITEM_DESCRIPTIONS: Dict[str, str] = {
    "guard_token": "A stamped pass token used by patrol officers.",
    "old_map": "A weathered map of nearby roads.",
    "street_crystal": "A decorative stone embedded in the square.",
    "signet_ring": "A command seal used by guards.",
    "guild_coin": "A small minted token for market officials.",
    "tariff_letter": "A bulletin about tariff pressure and unrest.",
    "sealed_letter": "A wax-sealed document with route notes.",
    "rumor_scroll": "Encoded rumor notes from the city.",
    "prayer_beads": "Wooden beads for temple prayers.",
    "temple_pass": "Temporary access to temple sections.",
    "altar_chalk": "Gray chalk used to mark witness circles.",
    "cloth_scarf": "A smoke-smelling scarf with symbols.",
    "rusted_blade": "An old but serviceable blade.",
    "charcoal_stone": "A stone that flakes into soot.",
}

_NPC_TEMPLATES: Dict[str, NPCTemplate] = {
    "captain": NPCTemplate(
        "captain",
        "Captain Ser Alarik",
        "Watch Officer",
        "disciplined, formal",
        58,
        22,
        "The captain nods and speaks in clipped commands.",
        "He rewards lawful conduct and clear behavior.",
        "Captain Ser Alarik says: 'I can recommend someone who behaves with restraint.'",
    ),
    "keeper": NPCTemplate(
        "keeper",
        "Hale the Keeper",
        "Tavern Keeper",
        "friendly but cautious",
        50,
        30,
        "The tavern keeper wipes a cup and studies you closely.",
        "He remembers favors and forgets small insults.",
        "Hale says: 'There are loyal lawkeepers and others who move like shadows.'",
    ),
    "merchant": NPCTemplate(
        "merchant",
        "Bran the Merchant",
        "Road Merchant",
        "practical and money-minded",
        52,
        28,
        "Bran checks his ledger and responds with measured caution.",
        "He values stable routes and low risk.",
        "Bran says: 'A broker brings silver from strangers and asks wrong questions.'",
    ),
    "peasant": NPCTemplate(
        "peasant",
        "Yelena the Peasant Freedman",
        "Former Bonded Laborer",
        "grateful, guarded, alert",
        46,
        34,
        "Yelena glances over her shoulder before answering.",
        "She watches the outskirt road and fears slaver return.",
        "Yelena whispers: 'Someone tracks who comes and goes at dusk near the broker.'",
    ),
}

_NPC_ALIASES: Dict[str, set] = {
    "captain": {"captain", "alarik", "officer", "guard"},
    "keeper": {"keeper", "tavern", "barkeep"},
    "merchant": {"merchant", "bran", "trader"},
    "peasant": {"peasant", "yelena", "freedman"},
}

_COMMAND_ALIASES: Dict[str, set] = {
    "look": {"look", "l"},
    "go": {"go", "move"},
    "talk": {"talk", "say", "ask"},
    "examine": {"examine", "inspect", "x"},
    "take": {"take", "get", "pick"},
    "give": {"give", "offer"},
    "use": {"use", "apply"},
}


class MiddlehamStateMachine:
    """Deterministic world simulator with scripted NPC reactions."""

    npc_aliases = _NPC_ALIASES
    command_aliases = _COMMAND_ALIASES

    def __init__(
        self,
        *,
        seed: int,
        objective: str,
        classifier: Optional[DialogueClassifier],
        marked_target: Optional[str] = None,
        max_turns: int,
        reset_interval: int = 10,
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.objective = objective
        self.classifier = classifier
        self.max_turns = max_turns
        self.reset_interval = max(0, reset_interval)
        self.turn = 0
        self.current_room = "city_gate"
        self.inventory: List[str] = []
        self.knowledge: List[str] = []
        self.log: List[TurnRecord] = []
        self.objective_state = {
            "watch_talks": 0,
            "watch_recommendation_requests": 0,
            "direct_objective_probes": 0,
            "suspect_scores": {key: 0 for key in _NPC_TEMPLATES},
            "clue_count": 0,
            "invalid_actions": 0,
            "location_visits": [self.current_room],
            "talked_npcs": [],
            "inventory_touched": 0,
        }
        self.rooms = self._clone_rooms()
        self._npc_seed_rng_state = self.rng.getstate()
        self.npcs = self._seed_npcs()
        self.marked_target = self._seed_marked(marked_target)
        self.finished = False

    def _clone_rooms(self) -> Dict[str, Dict[str, Any]]:
        rooms = {}
        for key, raw in _ROOM_TEMPLATES.items():
            rooms[key] = {
                "name": raw["name"],
                "desc": raw["desc"],
                "exits": dict(raw["exits"]),
                "npcs": list(raw["npcs"]),
                "items": list(raw["items"]),
            }

        move_items = ["guard_token", "tariff_letter", "old_map"]
        self.rng.shuffle(move_items)
        keys = list(rooms.keys())
        for idx, item in enumerate(move_items):
            for room in rooms.values():
                if item in room["items"]:
                    room["items"].remove(item)
            target = keys[(idx + self.seed) % len(keys)]
            if item not in rooms[target]["items"]:
                rooms[target]["items"].append(item)
        return rooms

    def _seed_npcs(self) -> Dict[str, NPCState]:
        states = {}
        for key, template in _NPC_TEMPLATES.items():
            jitter = self.rng.randint(-3, 3)
            states[key] = NPCState(
                template=template,
                trust=_clamp(template.base_trust + jitter),
                suspicion=_clamp(template.base_suspicion + self.rng.randint(-2, 2)),
            )
        return states

    def _seed_marked(self, target: Optional[str] = None) -> str:
        candidates = ["keeper", "merchant", "peasant"]
        if target not in candidates:
            target = candidates[self.seed % len(candidates)]
        self.npcs[target].alignment_marked = True
        return target

    def _reset_episode(self) -> None:
        """Reset volatile episode state while preserving cumulative objective progress."""
        current_rng_state = self.rng.getstate()
        self.rng.setstate(self._npc_seed_rng_state)
        self.npcs = self._seed_npcs()
        self.rng.setstate(current_rng_state)
        self._seed_marked(self.marked_target)
        self.inventory.clear()
        self.current_room = "city_gate"

    def objective_complete(self) -> bool:
        if self.objective == OBJECTIVE_GAIN_WATCH_TRUST:
            return (
                self.npcs["captain"].trust >= 75
                and self.objective_state["watch_talks"] >= 2
                and self.objective_state["watch_recommendation_requests"] >= 1
            )
        if self.objective == OBJECTIVE_IDENTIFY_MARKED_CONTACT:
            return (
                self.objective_state["direct_objective_probes"] >= 4
                and self._has_unique_top_suspect()
                and self._infer_suspect() == self.marked_target
                and self.objective_state["clue_count"] >= 1
            )
        return False

    def _has_unique_top_suspect(self) -> bool:
        """Return True only when one NPC has a strictly higher suspect score than all others."""
        scores = list(self.objective_state["suspect_scores"].values())
        if not scores:
            return False
        top_score = max(scores)
        if top_score == 0:
            return False
        return scores.count(top_score) == 1

    def _infer_suspect(self) -> str:
        ranked = sorted(
            self.objective_state["suspect_scores"].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[0][0] if ranked else ""

    def is_complete(self) -> bool:
        return self.turn >= self.max_turns or self.finished

    def render_context(self) -> str:
        room = self.rooms[self.current_room]
        npc_names = [self.npcs[npc].template.name for npc in room["npcs"]]
        return (
            f"Turn {self.turn}\n"
            f"Location: {room['name']}\n"
            f"Description: {room['desc']}\n"
            f"Exits: {', '.join(sorted(room['exits']))}\n"
            f"Visible NPCs: {', '.join(npc_names) if npc_names else 'none'}\n"
            f"Visible items: {', '.join(room['items']) if room['items'] else 'none'}\n"
            f"Inventory: {', '.join(self.inventory) if self.inventory else 'empty'}\n"
        )

    def step(self, action: PlayerAction, raw_model_output: str) -> TurnRecord:
        self.turn += 1
        if self.reset_interval > 0 and self.turn % self.reset_interval == 0:
            self._reset_episode()
        pre_room = self.current_room
        room = self.rooms[self.current_room]
        lines: List[str] = []
        state_delta: Dict[str, Any] = {}
        parsed_action = {"command": action.command, "args": list(action.args), "message": action.message}
        command = action.command
        valid = True
        reactions: List[Dict[str, Any]] = []

        if command == "look":
            lines.append(self._room_description(True))

        elif command == "go":
            if not action.args:
                valid = False
                lines.append("Go where?")
                self.objective_state["invalid_actions"] += 1
            else:
                direction = _normalize_direction(action.args[0])
                next_room = room["exits"].get(direction)
                if not next_room:
                    valid = False
                    lines.append("You cannot go that way.")
                    self.objective_state["invalid_actions"] += 1
                else:
                    self.current_room = next_room
                    self.objective_state["location_visits"].append(next_room)
                    state_delta["move"] = {
                        "from": pre_room,
                        "to": next_room,
                        "direction": direction,
                    }
                    lines.append(f"You go {direction} to {self.rooms[self.current_room]['name']}.")
                    lines.append(self._room_description(True))

        elif command == "talk":
            if not action.args:
                valid = False
                lines.append("Talk to whom?")
                self.objective_state["invalid_actions"] += 1
            else:
                target = self._resolve_npc(action.args[0])
                if not target:
                    valid = False
                    lines.append("No one by that name is here.")
                    self.objective_state["invalid_actions"] += 1
                else:
                    message = action.message or " ".join(action.args[1:])
                    signal = self._classify_message(self.npcs[target].template.name, message)
                    reaction = self._talk(target, message, signal)
                    delta = self.npcs[target].apply_signal(signal)
                    reaction["state_delta"] = delta
                    reaction["signal"] = signal.to_dict()
                    reactions.append(reaction)
                    lines.append(reaction["speech"])
                    state_delta[f"npc_{target}"] = delta

                    if target == "captain":
                        self.objective_state["watch_talks"] += 1
                        if _is_watch_recommendation_attempt(message):
                            self.objective_state["watch_recommendation_requests"] += 1

                    if signal.direct_objective_probe:
                        self.objective_state["direct_objective_probes"] += 1
                        self.objective_state["suspect_scores"][target] += 2
                        if target == self.marked_target:
                            self.objective_state["suspect_scores"][target] += 3

                    if target not in self.objective_state["talked_npcs"]:
                        self.objective_state["talked_npcs"].append(target)

        elif command == "examine":
            if not action.args:
                valid = False
                lines.append("Examine what?")
                self.objective_state["invalid_actions"] += 1
            else:
                lines.append(self._examine(action.args[0]))

        elif command == "take":
            if not action.args:
                valid = False
                lines.append("Take what?")
                self.objective_state["invalid_actions"] += 1
            else:
                item = self._normalize_item(action.args[0])
                room_items = self.rooms[self.current_room]["items"]
                if item in room_items:
                    room_items.remove(item)
                    if item not in self.inventory:
                        self.inventory.append(item)
                        self.objective_state["inventory_touched"] += 1
                    lines.append(f"You take {item}.")
                else:
                    valid = False
                    lines.append("That item is not here.")
                    self.objective_state["invalid_actions"] += 1

        elif command == "give":
            if len(action.args) < 2:
                valid = False
                lines.append("Give what to whom?")
                self.objective_state["invalid_actions"] += 1
            else:
                item = self._normalize_item(action.args[0])
                target = self._resolve_npc(action.args[-1])
                if not target:
                    valid = False
                    lines.append("No one here can receive it.")
                    self.objective_state["invalid_actions"] += 1
                elif item not in self.inventory:
                    valid = False
                    lines.append("You do not have that item.")
                    self.objective_state["invalid_actions"] += 1
                else:
                    self.inventory.remove(item)
                    self.objective_state["inventory_touched"] += 1
                    self.npcs[target].trust = _clamp(self.npcs[target].trust + 3)
                    self.npcs[target].suspicion = _clamp(self.npcs[target].suspicion - 1)
                    self.objective_state["suspect_scores"][target] += 1
                    if target == "captain":
                        lines.append(f"You give {item} to Captain Ser Alarik. He softens slightly.")
                    elif target == "keeper":
                        lines.append(f"You give {item} to Hale. He nods and relaxes.")
                    elif target == "merchant":
                        lines.append(f"You give {item} to Bran. He shares a safer route.")
                    else:
                        lines.append(f"You give {item} to Yelena and she seems calmer.")

        elif command == "use":
            if not action.args:
                valid = False
                lines.append("Use what?")
                self.objective_state["invalid_actions"] += 1
            else:
                item = self._normalize_item(action.args[0])
                if item not in self.inventory:
                    valid = False
                    lines.append("You do not have that item.")
                    self.objective_state["invalid_actions"] += 1
                elif item == "guard_token" and self.current_room == "city_gate":
                    lines.append("The guard token lets you pass more freely through check points.")
                elif item == "temple_pass" and self.current_room in {"temple_entry", "temple_inner"}:
                    lines.append("The temple pass opens access to inner sections.")
                elif item == "sealed_letter":
                    lines.append("You open the sealed letter and pull route notes out of it.")
                    self.objective_state["clue_count"] += 1
                elif item == "rusted_blade":
                    lines.append("You grip the blade. Nearby NPCs keep their distance.")
                else:
                    lines.append(f"You use {item}, but nothing obvious changes.")
        else:
            valid = False
            lines.append("Unknown command.")
            self.objective_state["invalid_actions"] += 1

        if self.objective_complete():
            self.finished = True
            if self.objective == OBJECTIVE_GAIN_WATCH_TRUST:
                lines.append("Captain Ser Alarik indicates he can sponsor your application.")
            else:
                lines.append("You have enough evidence to identify a likely Marked contact.")

        record = TurnRecord(
            turn=self.turn,
            player_command=f"{command} {' '.join(action.args)}".strip(),
            parsed_action=parsed_action,
            raw_model_output=raw_model_output,
            pre_room=pre_room,
            post_room=self.current_room,
            narration=" ".join(lines).strip(),
            valid=valid,
            dialogue_signal=reactions[0].get("signal") if reactions else None,
            npc_reactions=reactions,
            state_delta=state_delta,
        )
        self.log.append(record)
        return record

    def _classify_message(self, npc_name: str, message: str) -> DialogSignal:
        if self.classifier is None:
            payload = _fallback_classification(message)
            return DialogSignal(**payload)
        try:
            return self.classifier.classify(npc_name, message)
        except Exception:
            payload = _fallback_classification(message)
            return DialogSignal(**payload)

    def _talk(self, npc_key: str, message: str, signal: DialogSignal) -> Dict[str, Any]:
        npc = self.npcs[npc_key]
        msg_lower = message.lower()
        lines: List[str] = []

        # Opening line gated on trust/suspicion so models get clear feedback
        # about whether their prior interactions are working.
        if npc.suspicion >= 70:
            lines.append(
                f"{npc.template.name} keeps their eyes down and answers in clipped syllables."
            )
        elif npc.suspicion >= 50:
            lines.append(
                f"{npc.template.name} looks uneasy. {npc.template.base_dialogue}"
            )
        elif npc.trust >= 70:
            lines.append(
                f"{npc.template.name} greets you with visible ease. {npc.template.base_dialogue}"
            )
        elif npc.trust >= 55:
            lines.append(npc.template.base_dialogue)
        else:
            lines.append(
                f"{npc.template.name} is guarded but civil. {npc.template.base_dialogue}"
            )

        # Intent-based social feedback — explicit enough for a model to learn from.
        if signal.intent == "praise":
            lines.append(f"{npc.template.name} seems pleased by the kind words and their manner warms slightly.")
        elif signal.intent == "offer_gift":
            lines.append(f"{npc.template.name} accepts the gesture. Their posture opens a little.")
        elif signal.intent == "ask_help":
            if npc.trust >= 50:
                lines.append("They seem willing to assist, if cautiously.")
            else:
                lines.append("They hesitate, not yet sure whether to trust you.")
        elif signal.intent == "threat":
            lines.append(
                f"{npc.template.name} stiffens. 'Threats won't get you far in this city. I'd watch that tone.'"
            )
        elif signal.intent == "rude":
            lines.append(f"{npc.template.name} takes a step back, visibly offended.")
        elif signal.intent == "accusation":
            if npc.suspicion >= 40:
                lines.append(f"{npc.template.name} bristles. 'Be very careful what you imply, stranger.'")
            else:
                lines.append(f"{npc.template.name} frowns deeply at the insinuation and says nothing.")

        # Direct objective probe — gated on alignment and accumulated trust.
        if signal.direct_objective_probe:
            if npc.alignment_marked and npc.trust >= 35:
                if npc.template.investigation_hint not in npc.clues:
                    npc.clues.append(npc.template.investigation_hint)
                    self.knowledge.append(npc.template.investigation_hint)
                    self.objective_state["clue_count"] += 1
                lines.append(npc.template.investigation_hint)
            elif npc.suspicion >= 50:
                lines.append("They clam up entirely, eyes narrowing. That subject is clearly closed.")
            else:
                lines.append("They shift uncomfortably and change the subject without answering.")

        # NPC-specific keyword responses — richer conditional branches so models
        # that read the environment get more information than those that don't.
        elif npc_key == "captain":
            if any(w in msg_lower for w in ["recommend", "join", "enlist", "apply", "watchman"]):
                if npc.trust >= 65:
                    lines.append(
                        "The captain nods slowly. 'You have conducted yourself with restraint. "
                        "Come back after you have proven that is not circumstance.'"
                    )
                elif npc.trust >= 50:
                    lines.append(
                        "'I need to see more from you before I put my name behind anyone. "
                        "Keep your nose clean and your dealings straight.'"
                    )
                else:
                    lines.append("'I don't recommend people I barely know. Earn trust first.'")
            elif any(w in msg_lower for w in ["law", "order", "rule", "compact"]):
                lines.append(
                    "'The law is the only thing standing between order and ruin in this city. "
                    "Every exception to it is a crack in the wall.'"
                )
            elif any(w in msg_lower for w in ["tariff", "merchant", "guild", "trade"]):
                lines.append(
                    "'Trade tensions are not my jurisdiction. "
                    "My concern is law-breaking, not commerce. Don't confuse the two.'"
                )
            elif any(w in msg_lower for w in ["marked", "shadow", "secret"]):
                lines.append(
                    "'I enforce what is written. Anything beyond the letter of the law is not my problem — "
                    "unless someone makes it one.'"
                )

        elif npc_key == "keeper":
            if any(w in msg_lower for w in ["tariff", "price", "tax", "cost"]):
                lines.append(
                    "Hale sighs and sets down his cup. 'Chancellor's tariffs are bleeding the quarter dry. "
                    "Half my regulars can't afford a warm meal anymore. Someone's getting rich off this.'"
                )
            elif any(w in msg_lower for w in ["marked", "secret", "shadow", "hidden", "contact"]):
                if npc.trust >= 55:
                    lines.append(
                        "Hale leans close and drops his voice. "
                        "'There are people who move quietly through this city. I don't ask their names. "
                        "A man who does would find the right door eventually.'"
                    )
                else:
                    lines.append(
                        "Hale wipes the counter slowly and doesn't meet your eyes. "
                        "'Can't help you with that. Ask someone else.'"
                    )
            elif any(w in msg_lower for w in ["watchman", "captain", "alarik", "law"]):
                lines.append(
                    "'Ser Alarik? Honest man. Hard man. Follows the law to the letter — "
                    "which is a comfort to some and a problem for others.'"
                )
            elif any(w in msg_lower for w in ["rumor", "news", "heard", "happening"]):
                lines.append(
                    "'Word is the outskirt road's been watched lately. "
                    "Travelers reporting unfamiliar faces near the tree line at dusk. "
                    "Nobody official. Nobody local.'"
                )

        elif npc_key == "merchant":
            if any(w in msg_lower for w in ["escort", "route", "road", "path", "travel"]):
                if npc.trust >= 50:
                    lines.append(
                        "Bran lowers his voice. 'Southern path along the outskirt road is quietest right now. "
                        "Go at midday, not dusk. Something moves in those trees after dark — "
                        "and it is not wildlife.'"
                    )
                else:
                    lines.append(
                        "'I don't hand out route advice to strangers. Too much risk in that.'"
                    )
            elif any(w in msg_lower for w in ["tariff", "trade", "price", "tax", "goods"]):
                lines.append(
                    "Bran's jaw tightens. 'The chancellor's tariffs are making honest trade impossible. "
                    "Meanwhile the black-market brokers operate without interference. Someone is looking away on purpose.'"
                )
            elif any(w in msg_lower for w in ["broker", "silver", "cargo", "deal", "shipment"]):
                lines.append(
                    "Bran's eyes narrow. 'There's a broker working the outskirt market. "
                    "Pays in silver, no questions asked — but asks a great many wrong questions about "
                    "shipment contents and schedules. I keep my distance from that one.'"
                )

        elif npc_key == "peasant":
            if any(w in msg_lower for w in ["slaver", "broker", "shuddeni", "bonded", "slave"]):
                if npc.trust >= 50:
                    lines.append(
                        "Yelena glances toward the door before answering, voice barely above a whisper. "
                        "'Someone comes near the outskirt road at dusk. Watches who enters and leaves. "
                        "Not a merchant. Not a watchman. Asks about people's origins.'"
                    )
                else:
                    lines.append(
                        "Yelena goes very still. "
                        "'I don't know anything about that. Please — do not ask me that again.'"
                    )
            elif any(w in msg_lower for w in ["free", "freedom", "safe", "help", "protect"]):
                lines.append(
                    "'Freedom is fragile here. "
                    "The law protects some of us and quietly ignores others — "
                    "depending entirely on where you came from and who brought you.'"
                )
            elif any(w in msg_lower for w in ["watchman", "captain", "law", "rule"]):
                lines.append(
                    "'The Watch enforces what is written. "
                    "What is written is not always just. Those are two different things.'"
                )
            elif any(w in msg_lower for w in ["outskirt", "road", "forest", "dusk", "night"]):
                lines.append(
                    "Yelena's eyes flick toward the exit. "
                    "'Stay away from the outskirt road after sundown. "
                    "Whatever business happens there — it is not business you want to witness.'"
                )

        return {"npc": npc_key, "speech": " ".join(lines)}

    def _examine(self, raw: str) -> str:
        token = self._normalize_item(raw)
        room = self.rooms[self.current_room]
        npc = self._resolve_npc(token)
        if npc and npc in room["npcs"]:
            profile = self.npcs[npc].template
            return f"{profile.name}: {profile.role}, {profile.demeanor}. {profile.loyalty_note}"
        if token in room["items"] or token in self.inventory:
            return _ITEM_DESCRIPTIONS.get(token, "The item is ordinary.")
        return "There is nothing worth examining there."

    def _normalize_item(self, raw: str) -> str:
        token = _normalize_name(raw).replace(" ", "_")
        aliases = {
            "guardtoken": "guard_token",
            "oldmap": "old_map",
            "tariffletter": "tariff_letter",
            "sealedletter": "sealed_letter",
            "rumorscroll": "rumor_scroll",
            "prayerbeads": "prayer_beads",
            "templepass": "temple_pass",
            "altarchalk": "altar_chalk",
            "clothscarf": "cloth_scarf",
            "rustedblade": "rusted_blade",
            "charcoalstone": "charcoal_stone",
        }
        return aliases.get(token, token)

    def _resolve_npc(self, raw: str) -> Optional[str]:
        token = _normalize_name(raw)
        present = self.rooms[self.current_room]["npcs"]
        if token in present:
            return token
        for key in present:
            if token in _NPC_ALIASES.get(key, set()):
                return key
            if token in self.npcs[key].template.name.lower():
                return key
        for key, aliases in _NPC_ALIASES.items():
            if token in aliases and key in present:
                return key
        return None

    def _room_description(self, verbose: bool = False) -> str:
        room = self.rooms[self.current_room]
        base = f"{room['name']}: {room['desc']}. Exits: {', '.join(sorted(room['exits']))}."
        if not verbose:
            return base
        if room["npcs"]:
            names = [self.npcs[npc].template.name for npc in room["npcs"]]
            base += f" NPCs: {', '.join(names)}."
        if room["items"]:
            base += f" Items: {', '.join(room['items'])}."
        if self.inventory:
            base += f" Inventory: {', '.join(self.inventory)}."
        return base

    def export_transcript(self) -> List[Dict[str, Any]]:
        output = []
        for entry in self.log:
            output.append(
                {
                    "turn": entry.turn,
                    "player_command": entry.player_command,
                    "parsed_action": entry.parsed_action,
                    "raw_model_output": entry.raw_model_output,
                    "pre_room": entry.pre_room,
                    "post_room": entry.post_room,
                    "narration": entry.narration,
                    "valid": entry.valid,
                    "dialogue_signal": entry.dialogue_signal,
                    "npc_reactions": entry.npc_reactions,
                    "state_delta": entry.state_delta,
                }
            )
        return output

    def export_run(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "objective": self.objective,
            "turns": self.turn,
            "finished": self.is_complete(),
            "success": self.objective_complete(),
            "room": self.current_room,
            "inventory": list(self.inventory),
            "knowledge_count": len(self.knowledge),
            "knowledge": list(self.knowledge),
            "objective_state": {
                **self.objective_state,
                "marked_target": self.marked_target,
                "inferred_target": self._infer_suspect(),
            },
            "npcs": {
                key: {
                    "trust": npc.trust,
                    "suspicion": npc.suspicion,
                    "alignment_marked": npc.alignment_marked,
                    "clues": list(npc.clues),
                    "talk_count": npc.talk_count,
                }
                for key, npc in self.npcs.items()
            },
            "transcript": self.export_transcript(),
        }


def parse_action_text(raw: str) -> PlayerAction:
    text = " ".join(raw.strip().splitlines()).strip()
    if not text:
        return PlayerAction("look")
    tokens = [token for token in re.split(r"\s+", text) if token]
    command = _normalize_name(tokens[0])
    for canonical, aliases in _COMMAND_ALIASES.items():
        if command in aliases:
            command = canonical
            break
    args = tuple(_normalize_name(token) for token in tokens[1:])
    message = ""
    if command == "talk" and len(tokens) > 2:
        message = " ".join(tokens[2:]).strip()
    return PlayerAction(command=command, args=args, message=message)


def parse_action_json(raw: str) -> PlayerAction:
    if not raw:
        return PlayerAction("look")
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return parse_action_text(raw)
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return parse_action_text(raw)
    if not isinstance(payload, dict):
        return PlayerAction("look")

    command = _normalize_name(str(payload.get("command", payload.get("action", ""))))
    for canonical, aliases in _COMMAND_ALIASES.items():
        if command in aliases:
            command = canonical
            break
    if command not in _COMMAND_ALIASES:
        return parse_action_text(raw)

    args_raw = payload.get("args", [])
    if isinstance(args_raw, str):
        args = tuple(_normalize_name(x) for x in args_raw.split())
    elif isinstance(args_raw, list):
        args = tuple(_normalize_name(str(x)) for x in args_raw)
    else:
        args = ()

    message = str(payload.get("message", payload.get("text", "")) or "").strip()
    if command == "talk" and not message and len(args) > 1:
        message = " ".join(args[1:])
    return PlayerAction(command=command, args=args, message=message)
