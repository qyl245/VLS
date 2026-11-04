from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import render_task1_5_gt_overlay as overlay


class GtOverlayTests(unittest.TestCase):
    def test_three_point_angle_handles_right_angle(self) -> None:
        points = np.zeros((1, 25, 3), dtype=np.float64)
        points[0, 1] = [1.0, 0.0, 0.0]
        points[0, 2] = [0.0, 0.0, 0.0]
        points[0, 3] = [0.0, 1.0, 0.0]
        angle = overlay.three_point_angle(points, (1, 2, 3))
        self.assertAlmostEqual(float(angle[0]), 90.0, places=6)

    def test_rep_frame_selection_keeps_first_middle_last(self) -> None:
        self.assertEqual(overlay.select_rep_frames([10]), [10])
        self.assertEqual(overlay.select_rep_frames([10, 20, 30]), [10, 20, 30])
        self.assertEqual(
            overlay.select_rep_frames([10, 20, 30, 40, 50, 60]),
            [10, 40, 60],
        )

    def test_rep_semantics_flags_extension_labels(self) -> None:
        angles = np.asarray(
            [170.0, 160.0, 70.0, 160.0, 170.0, 160.0, 70.0, 160.0, 170.0],
            dtype=np.float64,
        )
        result = overlay.rep_semantics(angles, [0, 4, 8])
        self.assertEqual(
            result["provisional_semantics"], "near_extension_top_or_boundary"
        )
        self.assertEqual(
            result["bottom_gt_status"], "not_confirmed_do_not_use_as_bottom"
        )
        self.assertLess(result["interval_min_median_deg"], 80.0)


if __name__ == "__main__":
    unittest.main()
