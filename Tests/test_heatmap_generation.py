from pathlib import Path
import tempfile
import unittest

import pandas as pd

import heatmap_generation as hg


def _build_run_frame() -> pd.DataFrame:
    rows = []
    for repeat_index, total in enumerate((4.0, 2.0), start=1):
        rows.append(
            {
                "model": "alpha",
                "seed": "seed_a",
                "objective": "trust",
                "scenario_index": 1,
                "repeat_index": repeat_index,
                "total": total,
                "social_adaptation": 4.0,
                "world_grounding": 2.0,
            }
        )
    for repeat_index, total in enumerate((5.0, 5.0), start=1):
        rows.append(
            {
                "model": "alpha",
                "seed": "seed_b",
                "objective": "identify",
                "scenario_index": 2,
                "repeat_index": repeat_index,
                "total": total,
                "social_adaptation": 3.0,
                "world_grounding": 5.0,
            }
        )
    return pd.DataFrame(rows)


def _build_large_cell_frame() -> pd.DataFrame:
    rows = []
    for scenario_index in range(10):
        rows.extend(
            [
                {
                    "model": "high",
                    "scenario_index": scenario_index,
                    "objective": "trust" if scenario_index < 5 else "identify",
                    "seed": f"seed_{scenario_index}",
                    "run_count": 5,
                    "total": 5.0,
                    "hard_scored": 4.9,
                },
                {
                    "model": "mid",
                    "scenario_index": scenario_index,
                    "objective": "trust" if scenario_index < 5 else "identify",
                    "seed": f"seed_{scenario_index}",
                    "run_count": 5,
                    "total": 3.0,
                    "hard_scored": 3.1,
                },
                {
                    "model": "low",
                    "scenario_index": scenario_index,
                    "objective": "trust" if scenario_index < 5 else "identify",
                    "seed": f"seed_{scenario_index}",
                    "run_count": 5,
                    "total": 1.0,
                    "hard_scored": 1.1,
                },
            ]
        )
    return pd.DataFrame(rows)


class HeatmapGenerationTests(unittest.TestCase):
    def test_build_scenario_cell_frame_averages_repeats(self) -> None:
        cell_frame = hg.build_scenario_cell_frame(_build_run_frame())

        self.assertEqual(len(cell_frame), 2)
        first_cell = cell_frame.loc[cell_frame["scenario_index"] == 1].iloc[0]
        second_cell = cell_frame.loc[cell_frame["scenario_index"] == 2].iloc[0]

        self.assertEqual(int(first_cell["run_count"]), 2)
        self.assertEqual(float(first_cell["total"]), 3.0)
        self.assertEqual(float(first_cell["hard_scored"]), 3.0)

        self.assertEqual(int(second_cell["run_count"]), 2)
        self.assertEqual(float(second_cell["total"]), 5.0)
        self.assertEqual(float(second_cell["hard_scored"]), 4.0)

    def test_build_heatmap_result_orders_models_by_mean_score(self) -> None:
        spec = hg.HeatmapSpec(
            score_column="total",
            display_name="Scenario-cell total",
            title="Test Heatmap",
            output_filename="test.png",
        )

        result = hg.build_heatmap_result(_build_large_cell_frame(), spec)

        self.assertEqual(result.model_order, ["high", "mid", "low"])
        self.assertEqual(result.mean_scores.loc["high"], 5.0)
        self.assertEqual(result.mean_scores.loc["mid"], 3.0)
        self.assertEqual(result.mean_scores.loc["low"], 1.0)
        self.assertEqual(result.significant_pairs, 3)
        self.assertEqual(result.total_pairs, 3)

    def test_count_significant_pairs_counts_upper_triangle_only(self) -> None:
        p_value_frame = pd.DataFrame(
            [
                [1.0, 0.04, 0.20],
                [0.04, 1.0, 0.001],
                [0.20, 0.001, 1.0],
            ],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )

        self.assertEqual(hg.count_significant_pairs(p_value_frame), 2)

    def test_build_discrepancy_message_warns_when_count_is_far_off(self) -> None:
        warning = hg.build_discrepancy_message(significant_pairs=31, expected_pairs=23, tolerance=3)
        no_warning = hg.build_discrepancy_message(
            significant_pairs=24,
            expected_pairs=23,
            tolerance=3,
        )

        self.assertIsNotNone(warning)
        self.assertIn("31 significant pairs", warning)
        self.assertIsNone(no_warning)

    def test_render_heatmap_writes_png(self) -> None:
        spec = hg.HeatmapSpec(
            score_column="total",
            display_name="Scenario-cell total",
            title="Test Heatmap",
            output_filename="test.png",
        )
        result = hg.build_heatmap_result(_build_large_cell_frame(), spec)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "heatmap.png"
            hg.render_heatmap(result, output_path)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
