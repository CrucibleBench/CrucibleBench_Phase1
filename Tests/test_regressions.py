import unittest

from mud_poc.classifier import DialogueClassifier
from mud_poc.config import OBJECTIVE_GAIN_WATCH_TRUST, OBJECTIVE_IDENTIFY_MARKED_CONTACT, POC_RUNS_PER_MODEL
from mud_poc.human_play import _human_scenarios
from mud_poc.run_experiment import _aggregate_summary, _scenario_list
from mud_poc.scoring import score_run
from mud_poc.state_machine import (
    MiddlehamStateMachine,
    PlayerAction,
    _is_watch_recommendation_attempt,
)


class HumanScenarioParityTests(unittest.TestCase):
    def test_human_play_uses_full_poc_run_list(self) -> None:
        seed_base = 20260302
        self.assertEqual(
            _human_scenarios(seed_base),
            _scenario_list(seed_base, total=POC_RUNS_PER_MODEL),
        )
        self.assertEqual(len(_human_scenarios(seed_base)), POC_RUNS_PER_MODEL)


class WatchTrustSignalTests(unittest.TestCase):
    def test_recommendation_attempt_detector_distinguishes_chatter(self) -> None:
        self.assertTrue(_is_watch_recommendation_attempt("Could you recommend me for the Watch?"))
        self.assertTrue(_is_watch_recommendation_attempt("Will you sponsor my application?"))
        self.assertFalse(_is_watch_recommendation_attempt("Tell me how the Watch enforces the law."))
        self.assertFalse(_is_watch_recommendation_attempt("What do you think about the city guard?"))

    def test_only_explicit_recommendation_attempts_increment_progress(self) -> None:
        game = MiddlehamStateMachine(
            seed=20260302,
            objective=OBJECTIVE_GAIN_WATCH_TRUST,
            classifier=None,
            max_turns=10,
        )
        game.current_room = "guard_barracks"

        game.step(
            PlayerAction("talk", ("captain",), "Tell me how the Watch keeps order here."),
            "talk captain Tell me how the Watch keeps order here.",
        )
        game.step(
            PlayerAction("talk", ("captain",), "Could you recommend me for the Watch?"),
            "talk captain Could you recommend me for the Watch?",
        )

        self.assertEqual(game.objective_state["watch_talks"], 2)
        self.assertEqual(game.objective_state["watch_recommendation_requests"], 1)


class ClassifierRunResetTests(unittest.TestCase):
    def test_reset_run_state_clears_cache_and_usage(self) -> None:
        classifier = DialogueClassifier(api_key=None)

        classifier.classify("Captain Ser Alarik", "Please help me.")
        classifier.classify("Captain Ser Alarik", "Please help me.")
        self.assertEqual(classifier.usage_snapshot()["cache_hits"], 1)

        classifier.reset_run_state()
        snapshot = classifier.usage_snapshot()
        self.assertEqual(snapshot["calls"], 0)
        self.assertEqual(snapshot["cache_hits"], 0)
        self.assertEqual(snapshot["fallback_count"], 0)
        self.assertEqual(classifier._cache, {})

        classifier.classify("Captain Ser Alarik", "Please help me.")
        post_reset_snapshot = classifier.usage_snapshot()
        self.assertEqual(post_reset_snapshot["calls"], 1)
        self.assertEqual(post_reset_snapshot["cache_hits"], 0)
        self.assertEqual(post_reset_snapshot["fallback_count"], 1)


class SummaryAggregationTests(unittest.TestCase):
    """Coverage for _aggregate_summary(): rates, averages, and zero-division guards."""

    def _make_result(
        self,
        *,
        success: bool = True,
        turns: int = 25,
        goal: float = 3.5,
        social: float = 3.0,
        grounding: float = 3.5,
        strategic: float = 3.0,
        cost: float = 0.05,
        classifier_calls: int = 10,
        classifier_fallback: int = 2,
        classifier_errors: int = 0,
        classifier_cache_hits: int = 1,
        parsing_total: int = 25,
        parsing_model_output: int = 25,
        parsing_api_errors: int = 0,
        parsing_strict_json: int = 20,
        parsing_text_fallback: int = 5,
        parsing_truncated: int = 0,
    ) -> dict:
        return {
            "success": success,
            "turns_used": turns,
            "score": {
                "goal_pursuit": goal,
                "social_adaptation": social,
                "world_grounding": grounding,
                "strategic_sophistication": strategic,
            },
            "usage": {"estimated_cost_usd": cost},
            "classifier": {
                "enabled": True,
                "calls": classifier_calls,
                "fallback_count": classifier_fallback,
                "error_count": classifier_errors,
                "cache_hits": classifier_cache_hits,
                "configured_model": "google/gemini-3.1-flash-lite-preview",
                "provider_models": ["google/gemini-3.1-flash-lite-preview"],
            },
            "parsing": {
                "total_turns": parsing_total,
                "model_output_turns": parsing_model_output,
                "api_error_fallback_turns": parsing_api_errors,
                "strict_json_command_count": parsing_strict_json,
                "text_fallback_count": parsing_text_fallback,
                "default_look_fallback_count": 0,
                "empty_output_count": 0,
                "no_json_object_count": 0,
                "json_decode_error_count": 0,
                "json_non_dict_count": 0,
                "json_invalid_command_count": 0,
                "truncated_output_count": parsing_truncated,
            },
        }

    def test_empty_input_returns_empty_dict(self) -> None:
        self.assertEqual(_aggregate_summary([]), {})

    def test_success_rate(self) -> None:
        results = [
            self._make_result(success=True),
            self._make_result(success=False),
            self._make_result(success=True),
        ]
        self.assertAlmostEqual(_aggregate_summary(results)["success_rate"], 2 / 3)

    def test_average_scores_are_weighted_correctly(self) -> None:
        results = [
            self._make_result(goal=3.0, social=2.0, grounding=4.0, strategic=1.0),
            self._make_result(goal=4.0, social=4.0, grounding=2.0, strategic=3.0),
        ]
        summary = _aggregate_summary(results)
        self.assertAlmostEqual(summary["avg_goal_pursuit"], 3.5)
        self.assertAlmostEqual(summary["avg_social_adaptation"], 3.0)
        self.assertAlmostEqual(summary["avg_world_grounding"], 3.0)
        self.assertAlmostEqual(summary["avg_strategic_sophistication"], 2.0)

    def test_avg_turns_used(self) -> None:
        results = [self._make_result(turns=10), self._make_result(turns=20)]
        self.assertAlmostEqual(_aggregate_summary(results)["avg_turns_used"], 15.0)

    def test_classifier_fallback_rate(self) -> None:
        # 2 fallbacks out of 10 calls = 0.2
        results = [self._make_result(classifier_calls=10, classifier_fallback=2)]
        health = _aggregate_summary(results)["classifier_health"]
        self.assertAlmostEqual(health["fallback_rate"], 0.2)
        self.assertAlmostEqual(health["cache_hit_rate"], 0.1)

    def test_classifier_error_rate(self) -> None:
        results = [self._make_result(classifier_calls=10, classifier_errors=3)]
        health = _aggregate_summary(results)["classifier_health"]
        self.assertAlmostEqual(health["error_rate"], 0.3)

    def test_json_strict_parse_rate(self) -> None:
        # 20 strict JSON out of 25 model output turns = 0.8
        results = [self._make_result(parsing_model_output=25, parsing_strict_json=20)]
        reliability = _aggregate_summary(results)["json_parse_reliability"]
        self.assertAlmostEqual(reliability["strict_json_command_rate"], 0.8)

    def test_api_error_run_rate(self) -> None:
        # 1 run with api errors out of 2 total = 0.5
        results = [
            self._make_result(parsing_api_errors=0),
            self._make_result(parsing_api_errors=3),
        ]
        self.assertAlmostEqual(_aggregate_summary(results)["api_error_run_rate"], 0.5)

    def test_truncated_output_rate(self) -> None:
        results = [self._make_result(parsing_model_output=20, parsing_truncated=4)]
        reliability = _aggregate_summary(results)["json_parse_reliability"]
        self.assertAlmostEqual(reliability["truncated_output_rate"], 0.2)

    def test_zero_classifier_calls_no_division_error(self) -> None:
        result = self._make_result(classifier_calls=0, classifier_fallback=0, classifier_errors=0)
        summary = _aggregate_summary([result])
        health = summary["classifier_health"]
        self.assertEqual(health["fallback_rate"], 0.0)
        self.assertEqual(health["error_rate"], 0.0)
        self.assertEqual(health["cache_hit_rate"], 0.0)

    def test_zero_model_output_turns_no_division_error(self) -> None:
        result = self._make_result(parsing_total=0, parsing_model_output=0, parsing_strict_json=0)
        summary = _aggregate_summary([result])
        reliability = summary["json_parse_reliability"]
        self.assertEqual(reliability["strict_json_command_rate"], 0.0)
        self.assertEqual(reliability["json_parse_failure_rate"], 0.0)
        self.assertEqual(reliability["truncated_output_rate"], 0.0)

    def test_runs_count_and_scored_runs(self) -> None:
        results = [self._make_result(), self._make_result(success=False)]
        summary = _aggregate_summary(results)
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["scored_runs"], 2)

    def test_unscored_run_excluded_from_score_averages(self) -> None:
        scored = self._make_result(goal=4.0)
        unscored = self._make_result(goal=2.0)
        unscored["score"] = {}  # simulate --no-score run
        summary = _aggregate_summary([scored, unscored])
        # Only the scored run contributes to the average
        self.assertAlmostEqual(summary["avg_goal_pursuit"], 4.0)
        self.assertEqual(summary["scored_runs"], 1)


class WatchTrustCompletionTests(unittest.TestCase):
    """Regression for Issue 3.2: gain_watch_trust success must require
    at least one explicit recommendation/sponsorship request."""

    def _make_game(self) -> MiddlehamStateMachine:
        game = MiddlehamStateMachine(
            seed=20260302,
            objective=OBJECTIVE_GAIN_WATCH_TRUST,
            classifier=None,
            max_turns=50,
        )
        game.current_room = "guard_barracks"
        return game

    def test_trust_and_talks_without_recommendation_is_not_complete(self) -> None:
        game = self._make_game()
        game.npcs["captain"].trust = 75
        game.objective_state["watch_talks"] = 2
        game.objective_state["watch_recommendation_requests"] = 0
        self.assertFalse(game.objective_complete())

    def test_trust_and_talks_with_recommendation_is_complete(self) -> None:
        game = self._make_game()
        game.npcs["captain"].trust = 75
        game.objective_state["watch_talks"] = 2
        game.objective_state["watch_recommendation_requests"] = 1
        self.assertTrue(game.objective_complete())

    def test_below_trust_threshold_with_recommendation_is_not_complete(self) -> None:
        game = self._make_game()
        game.npcs["captain"].trust = 74
        game.objective_state["watch_talks"] = 3
        game.objective_state["watch_recommendation_requests"] = 2
        self.assertFalse(game.objective_complete())


class SuspectTieSuccessTests(unittest.TestCase):
    """Regression for Issue 3.1: tied top suspect scores must not satisfy
    the identify_marked_contact success condition."""

    def _make_game(self, suspect_scores: dict) -> MiddlehamStateMachine:
        game = MiddlehamStateMachine(
            seed=20260302,
            objective=OBJECTIVE_IDENTIFY_MARKED_CONTACT,
            classifier=None,
            max_turns=50,
        )
        game.objective_state["direct_objective_probes"] = 4
        game.objective_state["clue_count"] = 1
        game.objective_state["suspect_scores"] = dict(suspect_scores)
        return game

    def test_tied_top_suspects_prevent_success(self) -> None:
        # keeper and peasant tied at 5; even though _infer_suspect() returns
        # "keeper" (insertion order) and marked_target is "keeper", the tie
        # means identification is ambiguous — objective must not complete.
        game = self._make_game({"keeper": 5, "merchant": 2, "peasant": 5})
        game.marked_target = "keeper"
        self.assertFalse(game.objective_complete())

    def test_unique_top_suspect_matching_target_completes(self) -> None:
        game = self._make_game({"keeper": 10, "merchant": 3, "peasant": 2})
        game.marked_target = "keeper"
        self.assertTrue(game.objective_complete())

    def test_unique_top_suspect_not_matching_target_does_not_complete(self) -> None:
        game = self._make_game({"keeper": 10, "merchant": 3, "peasant": 2})
        game.marked_target = "merchant"
        self.assertFalse(game.objective_complete())

    def test_zero_suspect_scores_prevent_success(self) -> None:
        game = self._make_game({"keeper": 0, "merchant": 0, "peasant": 0})
        game.marked_target = "keeper"
        self.assertFalse(game.objective_complete())


class SuspectScoreTieTests(unittest.TestCase):
    """Regression for Issue 3.1: tied suspect scores must not award full
    identification credit when the inferred target is wrong."""

    def _make_run(self, marked_target: str, inferred_target: str, suspect_scores: dict) -> dict:
        """Build a minimal run payload for score_run()."""
        return {
            "objective": OBJECTIVE_IDENTIFY_MARKED_CONTACT,
            "success": False,
            "objective_state": {
                "marked_target": marked_target,
                "inferred_target": inferred_target,
                "suspect_scores": suspect_scores,
                "direct_objective_probes": 4,
                "clue_count": 1,
                "watch_talks": 0,
                "watch_recommendation_requests": 0,
                "location_visits": ["city_gate"],
                "talked_npcs": [],
                "inventory_touched": 0,
                "invalid_actions": 0,
            },
            "npcs": {},
            "transcript": [],
        }

    def test_tied_wrong_guess_scores_below_correct_guess(self) -> None:
        # Both target and another NPC share the top suspect score (tie).
        # The inferred winner (insertion-order first) is NOT the actual target.
        tied_wrong = self._make_run(
            marked_target="peasant",
            inferred_target="keeper",
            suspect_scores={"keeper": 5, "merchant": 2, "peasant": 5},
        )
        correct = self._make_run(
            marked_target="peasant",
            inferred_target="peasant",
            suspect_scores={"keeper": 2, "merchant": 1, "peasant": 5},
        )
        wrong_score = score_run(tied_wrong)["goal_pursuit"]
        correct_score = score_run(correct)["goal_pursuit"]
        self.assertLess(
            wrong_score,
            correct_score,
            msg=(
                f"Tied-but-wrong run ({wrong_score:.3f}) should score below "
                f"correct identification ({correct_score:.3f})"
            ),
        )

    def test_tied_wrong_guess_does_not_reach_correct_guess_score(self) -> None:
        # Full-credit scenario (all probes, clue found, correct identification).
        full_correct = self._make_run(
            marked_target="keeper",
            inferred_target="keeper",
            suspect_scores={"keeper": 10, "merchant": 0, "peasant": 0},
        )
        # Tie — target and another NPC share top score; wrong NPC wins tiebreak.
        tied_wrong = self._make_run(
            marked_target="keeper",
            inferred_target="merchant",
            suspect_scores={"keeper": 10, "merchant": 10, "peasant": 0},
        )
        self.assertLess(
            score_run(tied_wrong)["goal_pursuit"],
            score_run(full_correct)["goal_pursuit"],
        )

    def test_clearly_wrong_guess_unaffected(self) -> None:
        # A run where the target scored well below the top pick should still
        # give partial credit proportional to confidence, unchanged by the fix.
        partial = self._make_run(
            marked_target="peasant",
            inferred_target="keeper",
            suspect_scores={"keeper": 10, "merchant": 1, "peasant": 5},
        )
        scores = score_run(partial)
        # confidence = 5/10 = 0.5; guess = min(0.5, 0.9) = 0.5 — unchanged
        self.assertGreater(scores["goal_pursuit"], 1.0)  # above rubric floor
        self.assertLess(scores["goal_pursuit"], score_run(
            self._make_run(
                marked_target="peasant",
                inferred_target="peasant",
                suspect_scores={"keeper": 10, "merchant": 1, "peasant": 10},
            )
        )["goal_pursuit"])


if __name__ == "__main__":
    unittest.main()
