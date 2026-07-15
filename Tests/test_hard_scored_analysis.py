from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

import hard_scored_analysis as hsa


class HardScoredAnalysisTests(unittest.TestCase):
    def test_build_cell_frame_aggregates_by_cell(self) -> None:
        runs_df = pd.DataFrame(
            [
                {
                    "model_key": "alpha",
                    "model_label": "Alpha",
                    "objective": "obj_a",
                    "seed": "1",
                    "cell_key": "obj_a::1",
                    "run_number": 1,
                    "success": True,
                    "goal": 4.0,
                    "social": 3.0,
                    "ground": 5.0,
                    "strat": 2.0,
                    "hard": 4.0,
                    "full": 3.5,
                },
                {
                    "model_key": "alpha",
                    "model_label": "Alpha",
                    "objective": "obj_a",
                    "seed": "1",
                    "cell_key": "obj_a::1",
                    "run_number": 2,
                    "success": False,
                    "goal": 2.0,
                    "social": 1.0,
                    "ground": 3.0,
                    "strat": 4.0,
                    "hard": 2.0,
                    "full": 2.5,
                },
            ]
        )

        with mock.patch.object(hsa, "EXPECTED_MODELS", 1), mock.patch.object(
            hsa, "EXPECTED_CELLS_PER_MODEL", 1
        ), mock.patch.object(hsa, "EXPECTED_RUNS_PER_CELL", 2):
            cell_df = hsa._build_cell_frame(runs_df)

        self.assertEqual(len(cell_df), 1)
        self.assertEqual(int(cell_df.loc[0, "run_count"]), 2)
        self.assertAlmostEqual(float(cell_df.loc[0, "success_rate"]), 0.5)
        self.assertAlmostEqual(float(cell_df.loc[0, "hard"]), 3.0)
        self.assertAlmostEqual(float(cell_df.loc[0, "full"]), 3.0)

    def test_build_ranking_table_uses_full_precision_for_ranks(self) -> None:
        runs_df = pd.DataFrame(
            [
                {
                    "model_key": "alpha",
                    "model_label": "Alpha",
                    "ground": 3.0,
                    "social": 3.0,
                    "goal": 3.0,
                    "strat": 3.0,
                    "hard": 3.333400,
                    "full": 3.500000,
                },
                {
                    "model_key": "beta",
                    "model_label": "Beta",
                    "ground": 3.0,
                    "social": 3.0,
                    "goal": 3.0,
                    "strat": 3.0,
                    "hard": 3.333490,
                    "full": 3.400000,
                },
            ]
        )

        ranking_df = hsa._build_ranking_table(runs_df).set_index("model_key")

        self.assertEqual(ranking_df.loc["beta", "Hard Rank"], 1)
        self.assertEqual(ranking_df.loc["alpha", "Hard Rank"], 2)

    def test_build_gemini_interpretation_matches_rank_direction(self) -> None:
        cases = [
            (-6, "lands below its full-score rank"),
            (2, "lands above its full-score rank"),
            (0, "matches its full-score rank"),
        ]

        for rank_delta, expected_fragment in cases:
            with self.subTest(rank_delta=rank_delta):
                gemini_row = pd.Series(
                    {
                        "Hard Rank": 5 - rank_delta,
                        "Full Rank": 5,
                        "Rank Delta": rank_delta,
                    }
                )

                interpretation = hsa._build_gemini_interpretation(gemini_row)

                self.assertIn(expected_fragment, interpretation)

    def test_build_results_export_writes_csv(self) -> None:
        ranking_df = pd.DataFrame(
            [
                {
                    "model_key": "gemini_3_1_pro",
                    "model_label": "Gemini 3.1 Pro",
                    "WG": 4.410000,
                    "SA": 3.000000,
                    "GP": 4.340000,
                    "SS": 3.910000,
                    "Hard": 3.705000,
                    "Full": 3.915000,
                    "Hard Rank": 9,
                    "Full Rank": 3,
                    "Rank Delta": -6,
                }
            ]
        )
        runs_df = pd.DataFrame(
            [
                {"model_key": "gemini_3_1_pro", "cell_key": "a::1", "success": True},
                {"model_key": "gemini_3_1_pro", "cell_key": "a::1", "success": False},
            ]
        )
        results_df = hsa._build_results_export(
            ranking_df=ranking_df,
            runs_df=runs_df,
            tiers_hard={"gemini_3_1_pro": "Mid"},
            tiers_full={"gemini_3_1_pro": "Top"},
            boot_cis={"gemini_3_1_pro": (3.57, 3.71, 3.85)},
            validation_probe_agreement={"gemini_3_1_pro": 0.217},
        )

        expected_columns = [
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

        self.assertEqual(list(results_df.columns), expected_columns)
        self.assertEqual(int(results_df.loc[0, "run_count"]), 2)
        self.assertEqual(int(results_df.loc[0, "cell_count"]), 1)
        self.assertAlmostEqual(results_df.loc[0, "success_rate"], 0.5)
        self.assertAlmostEqual(results_df.loc[0, "hard_ci_width"], 0.28)
        self.assertTrue(bool(results_df.loc[0, "tier_changed"]))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "hard_scored_analysis.csv"
            hsa._write_results_csv(results_df, output_path)

            written_df = pd.read_csv(output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(list(written_df.columns), expected_columns)
            self.assertEqual(int(written_df.loc[0, "hard_rank"]), 9)
            self.assertAlmostEqual(float(written_df.loc[0, "probe_agreement"]), 0.217)


if __name__ == "__main__":
    unittest.main()
