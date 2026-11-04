from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_p2a_robustness as rr


class P2ARobustnessTests(unittest.TestCase):
    def test_tail_interval_iou_accepts_small_smoothing_shift(self) -> None:
        baseline = (1663, 1679)
        self.assertEqual(rr.interval_iou((1663, 1679), baseline), 1.0)
        self.assertAlmostEqual(rr.interval_iou((1661, 1679), baseline), 17 / 19)
        self.assertEqual(rr.interval_iou((1500, 1600), baseline), 0.0)

    def test_noisy_angles_preserve_nan_time_and_non_angle_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "9999"
            source.mkdir(parents=True)
            path = source / "angles_wide.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["frame_id", "timestamp_ms", "left_knee_flexion", "spine_stability"]
                )
                writer.writerow([0, 0, "10.0", "0.25"])
                writer.writerow([1, 33, "NaN", "0.50"])

            destination = root / "noisy"
            rng = np.random.default_rng(7)
            rr.write_noisy_angles(
                root / "source",
                destination,
                sigma=2.0,
                rng=rng,
                columns=("left_knee_flexion",),
            )

            with (destination / "9999" / "angles_wide.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["frame_id"], "0")
        self.assertEqual(rows[0]["timestamp_ms"], "0")
        self.assertEqual(rows[0]["spine_stability"], "0.25")
        self.assertNotEqual(float(rows[0]["left_knee_flexion"]), 10.0)
        self.assertTrue(math.isnan(float(rows[1]["left_knee_flexion"])))

    def test_noisy_landmarks_only_change_target_nose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            record = {
                "frame_id": 0,
                "timestamp_ms": 0,
                "detected": True,
                "landmarks_world": [[0.0, 0.0, 0.0] for _ in range(33)],
            }
            for video_id in ("0153", "0173"):
                (source / f"{video_id}_landmarks.json").write_text(
                    json.dumps([record])
                )

            destination = root / "noisy"
            rr.write_noisy_landmarks(
                source,
                destination,
                sigma=0.01,
                rng=np.random.default_rng(11),
            )
            target = json.loads((destination / "0173_landmarks.json").read_text())
            unchanged = json.loads((destination / "0153_landmarks.json").read_text())

        self.assertNotEqual(target[0]["landmarks_world"][rr.NOSE_INDEX], [0.0, 0.0, 0.0])
        self.assertEqual(target[0]["landmarks_world"][1], [0.0, 0.0, 0.0])
        self.assertEqual(unchanged[0]["landmarks_world"][rr.NOSE_INDEX], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
