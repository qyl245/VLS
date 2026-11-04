from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_5_metrics as metrics


class AngleTests(unittest.TestCase):
    def test_three_point_angle_returns_expected_angle(self) -> None:
        points = np.zeros((2, 25, 3), dtype=np.float64)
        points[:, 1] = [1.0, 0.0, 0.0]
        points[:, 2] = [0.0, 0.0, 0.0]
        points[:, 3] = [0.0, 1.0, 0.0]
        values = metrics.three_point_angle(points, (1, 2, 3))
        np.testing.assert_allclose(values, [90.0, 90.0], atol=1e-6)

    def test_fit3d_knee_angle_matches_mediapipe_definition(self) -> None:
        joints = np.zeros((1, 25, 3), dtype=np.float64)
        joints[0, 1] = [0.0, 0.0, 1.0]
        joints[0, 2] = [0.0, 0.0, 0.0]
        joints[0, 3] = [1.0, 0.0, 0.0]
        value = metrics.fit3d_angle_series(joints, "squat", "left_knee_flexion")
        self.assertAlmostEqual(float(value[0]), 90.0, places=6)


class SideSignalSelectionTests(unittest.TestCase):
    def _signal(self, start: float, stop: float) -> np.ndarray:
        return np.linspace(start, stop, 200)

    def test_near_side_is_preferred_when_usable(self) -> None:
        selection = metrics.choose_side_phase_signal(
            "pushup",
            "side_left",
            "left",
            {
                "left_elbow_flexion": self._signal(70.0, 165.0),
                "right_elbow_flexion": self._signal(80.0, 160.0),
            },
            {"left": 0.95, "right": 0.90},
        )
        self.assertEqual(selection["selected_side"], "left")
        self.assertEqual(selection["selection_reason"], "near_side_preferred")
        self.assertFalse(selection["low_confidence"])

    def test_far_side_never_replaces_occluded_near_side(self) -> None:
        selection = metrics.choose_side_phase_signal(
            "pushup",
            "side_left",
            "left",
            {
                "left_elbow_flexion": self._signal(150.0, 153.0),
                "right_elbow_flexion": self._signal(70.0, 165.0),
            },
            {"left": 0.95, "right": 0.90},
        )
        self.assertIsNone(selection["selected_side"])
        self.assertIsNone(selection["selected_angle"])
        self.assertEqual(
            selection["selection_reason"], "insufficient_near_side_signal_range"
        )
        self.assertTrue(selection["low_confidence"])
        self.assertFalse(selection["far_side_substitution"])

    def test_near_side_visibility_failure_is_rejected(self) -> None:
        selection = metrics.choose_side_phase_signal(
            "pushup",
            "side_left",
            "left",
            {
                "left_elbow_flexion": self._signal(70.0, 165.0),
                "right_elbow_flexion": self._signal(80.0, 160.0),
            },
            {"left": 0.20, "right": 0.95},
        )
        self.assertIsNone(selection["selected_side"])
        self.assertEqual(
            selection["selection_reason"], "insufficient_near_side_visibility"
        )

    def test_camera_geometry_overrides_conflicting_view_suffix(self) -> None:
        selection = metrics.choose_side_phase_signal(
            "pushup",
            "side_left",
            "right",
            {
                "left_elbow_flexion": self._signal(70.0, 165.0),
                "right_elbow_flexion": self._signal(80.0, 160.0),
            },
            {"left": 0.95, "right": 0.95},
        )
        self.assertEqual(selection["near_side"], "right")
        self.assertEqual(selection["selected_side"], "right")

    def test_front_view_has_no_side_signal_selection(self) -> None:
        self.assertIsNone(
            metrics.choose_side_phase_signal(
                "squat",
                "front",
                "left",
                {"left_knee_flexion": self._signal(70.0, 170.0)},
                {"left": 0.95, "right": 0.95},
            )
        )


class ErrorStatsTests(unittest.TestCase):
    def test_angle_error_stats_ignore_nan_and_report_p90(self) -> None:
        predicted = np.asarray([10.0, 20.0, np.nan, 40.0])
        reference = np.asarray([12.0, 18.0, 20.0, 30.0])
        result = metrics.angle_error_stats(predicted, reference)
        self.assertEqual(result["valid_frames"], 3)
        self.assertAlmostEqual(result["mae_deg"], 14.0 / 3.0, places=6)
        self.assertAlmostEqual(result["bias_deg"], 10.0 / 3.0, places=6)
        self.assertIsNotNone(result["p90_abs_error_deg"])

    def test_cluster_bootstrap_is_deterministic(self) -> None:
        values = {"s03_squat": 4.0, "s04_squat": 8.0, "s05_squat": 12.0}
        first = metrics.cluster_bootstrap_ci(values, seed=7, samples=200)
        second = metrics.cluster_bootstrap_ci(values, seed=7, samples=200)
        self.assertEqual(first, second)
        self.assertEqual(first["clusters"], 3)


class CameraDepthSideTests(unittest.TestCase):
    def test_near_side_uses_camera_space_depth(self) -> None:
        points = np.zeros((3, 25, 3), dtype=np.float64)
        points[:, [1, 2, 3], 2] = 4.0
        points[:, [4, 5, 6], 2] = 4.4
        result = metrics.near_side_from_camera_points(points, "squat", [0, 1, 2])
        self.assertEqual(result["near_side"], "left")
        self.assertAlmostEqual(result["signed_depth_margin"], 0.4, places=6)

    def test_right_side_can_be_near(self) -> None:
        points = np.zeros((2, 25, 3), dtype=np.float64)
        points[:, [11, 12, 13], 2] = 5.2
        points[:, [14, 15, 16], 2] = 4.8
        result = metrics.near_side_from_camera_points(points, "pushup", [0, 1])
        self.assertEqual(result["near_side"], "right")


if __name__ == "__main__":
    unittest.main()
