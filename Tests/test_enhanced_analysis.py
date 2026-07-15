import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "enhanced_analysis.py"
MODULE_SPEC = importlib.util.spec_from_file_location("enhanced_analysis", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec for {MODULE_PATH}")
enhanced_analysis = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(enhanced_analysis)


class EnhancedAnalysisHelperTests(unittest.TestCase):
    def test_resolve_config_path_defaults_to_project_root(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            resolved = enhanced_analysis._resolve_config_path(
                "CRUCIBLE_TEST_RESULTS_DIR", "results", "run2"
            )

        self.assertEqual(resolved, str(PROJECT_ROOT / "results" / "run2"))

    def test_resolve_config_path_honors_environment_override(self) -> None:
        override = str(PROJECT_ROOT / "custom-output")
        with mock.patch.dict(os.environ, {"CRUCIBLE_TEST_OUTPUT_DIR": override}, clear=False):
            resolved = enhanced_analysis._resolve_config_path("CRUCIBLE_TEST_OUTPUT_DIR")

        self.assertEqual(resolved, override)

    def test_bh_fdr_adjust_matches_expected_values(self) -> None:
        adjusted = enhanced_analysis.bh_fdr_adjust([0.01, 0.03, 0.04])

        self.assertEqual(adjusted, [0.03, 0.04, 0.04])

    def test_run_dunn_test_is_symmetric_and_detects_large_differences(self) -> None:
        result = enhanced_analysis.run_dunn_test(
            {
                "low": [1.0, 1.0, 1.0, 1.0],
                "high": [10.0, 10.0, 10.0, 10.0],
                "same_as_low": [1.0, 1.0, 1.0, 1.0],
            }
        )

        self.assertEqual(result["low"]["low"], 1.0)
        self.assertEqual(result["low"]["high"], result["high"]["low"])
        self.assertEqual(result["low"]["same_as_low"], 1.0)
        self.assertLess(result["low"]["high"], 0.05)

    def test_get_initial_npc_trust_reconstructs_starting_value(self) -> None:
        run = {
            "npcs": {"captain": {"trust": 81}},
            "transcript_events": [
                {
                    "turn": 2,
                    "action": {"command": "talk", "args": ["captain ser alarik", "hello"]},
                    "npc_reactions": [{"npc": "captain", "state_delta": {"trust_delta": 5}}],
                },
                {
                    "turn": 3,
                    "action": {"command": "talk", "args": ["captain ser alarik", "follow-up"]},
                    "npc_reactions": [{"npc": "captain", "state_delta": {"trust_delta": 1}}],
                },
            ],
        }

        initial_trust = enhanced_analysis._get_initial_npc_trust(
            run,
            "captain",
            run["transcript_events"],
        )

        self.assertEqual(initial_trust, 75.0)

    def test_extract_event_times_uses_reconstructed_initial_captain_trust(self) -> None:
        run = {
            "objective": "gain_watch_trust",
            "npcs": {"captain": {"trust": 81}},
            "transcript_events": [
                {
                    "turn": 1,
                    "action": {"command": "look", "args": []},
                    "npc_reactions": [],
                    "dialogue_signal": None,
                },
                {
                    "turn": 2,
                    "action": {"command": "talk", "args": ["captain ser alarik", "hello"]},
                    "npc_reactions": [{"npc": "captain", "state_delta": {"trust_delta": 5}}],
                    "dialogue_signal": None,
                },
                {
                    "turn": 3,
                    "action": {
                        "command": "talk",
                        "args": ["captain ser alarik", "can you recommend me for the watch?"],
                    },
                    "npc_reactions": [{"npc": "captain", "state_delta": {"trust_delta": 1}}],
                    "dialogue_signal": {"direct_objective_probe": True},
                },
            ],
        }

        event_times = enhanced_analysis.extract_event_times(run)

        self.assertEqual(event_times["turn_first_talk"], 2)
        self.assertEqual(event_times["turn_first_captain_talk"], 2)
        self.assertEqual(event_times["turn_captain_trust_70"], 0)
        self.assertEqual(event_times["turn_captain_trust_80"], 2)
        self.assertEqual(event_times["turn_first_rec_request"], 3)
        self.assertEqual(event_times["turn_first_probe"], 3)


if __name__ == "__main__":
    unittest.main()
