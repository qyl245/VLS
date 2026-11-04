from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_task1_5_fit3d as evaluation


class EvaluationTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[dict, Path]:
        joints = np.zeros((3, 25, 3), dtype=np.float64)
        joints[:, 1] = [0.0, 0.0, 1.0]
        joints[:, 2] = [0.0, 0.0, 0.0]
        joints[:, 3] = [1.0, 0.0, 0.0]
        joints[:, 4] = [1.0, 0.0, 1.0]
        joints[:, 5] = [1.0, 0.0, 0.0]
        joints[:, 6] = [2.0, 0.0, 0.0]
        joints_path = root / "joints.json"
        joints_path.write_text(json.dumps({"joints3d_25": joints.tolist()}))

        analysis_dir = root / "analysis"
        clip_dir = analysis_dir / "clip_a"
        angles_dir = clip_dir / "angles"
        angles_dir.mkdir(parents=True)
        (clip_dir / "analysis.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "quality_status": "pass",
                    "rep_count": 6,
                    "quality": {
                        "checks": {
                            "pose_detection_rate": 1.0,
                            "q10_key_visibility": 0.9,
                            "body_in_frame_rate": 1.0,
                        }
                    },
                    "segmentation": {
                        "phase_angle": "left_knee_flexion",
                        "signal_selection": {
                            "selected_angle": "left_knee_flexion",
                            "selected_side": "left",
                            "selection_reason": "near_side_preferred",
                            "low_confidence": False,
                        },
                    },
                }
            )
        )
        columns = [
            "timestamp_ms",
            "left_knee_flexion",
            "right_knee_flexion",
            "left_hip_flexion",
            "right_hip_flexion",
            "torso_flexion",
        ]
        with (angles_dir / "angles_wide.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for index, knee in enumerate((80.0, 90.0, 100.0)):
                writer.writerow(
                    {
                        "timestamp_ms": index * 20,
                        "left_knee_flexion": knee,
                        "right_knee_flexion": 90.0,
                        "left_hip_flexion": 90.0,
                        "right_hip_flexion": 90.0,
                        "torso_flexion": 90.0,
                    }
                )

        clip = {
            "clip_id": "clip_a",
            "subject_id": "s03",
            "camera_id": "cam",
            "exercise": "squat",
            "action_variant": "barbell_back_squat",
            "split": "dev",
            "view": "front",
            "has_gt": True,
            "smoke_only": False,
            "gt": {"joints3d_25": str(joints_path)},
            "view_geometry": {"angle_from_frontal_normal_deg": 20.0, "camera_side": "left"},
        }
        return clip, analysis_dir

    def test_evaluate_clip_flags_selected_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clip, analysis_dir = self._write_fixture(root)
            rows, quality = evaluation.evaluate_clip(clip, analysis_dir)
        self.assertEqual(quality["analysis_status"], "ok")
        left_knee = next(row for row in rows if row["angle_name"] == "left_knee_flexion")
        self.assertTrue(left_knee["selected_signal"])
        self.assertAlmostEqual(left_knee["mae_deg"], 20.0 / 3.0, places=6)
        self.assertEqual(left_knee["valid_frames"], 3)

    def test_gate_thresholds(self) -> None:
        self.assertEqual(evaluation.gate_for_mae(9.0), "pass")
        self.assertEqual(evaluation.gate_for_mae(12.0), "low_confidence")
        self.assertEqual(evaluation.gate_for_mae(16.0), "disabled")
        self.assertEqual(evaluation.gate_for_mae(None), "insufficient_data")

    def test_missing_analysis_is_preserved_as_failure(self) -> None:
        clip = {
            "clip_id": "missing",
            "subject_id": "s03",
            "camera_id": "cam",
            "exercise": "squat",
            "view": "front",
            "split": "dev",
            "has_gt": True,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            rows, quality = evaluation.evaluate_clip(clip, Path(temp_dir))
        self.assertEqual(rows, [])
        self.assertEqual(quality["analysis_status"], "missing")


if __name__ == "__main__":
    unittest.main()
