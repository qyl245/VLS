from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_task1_5_views
import build_task1_5_fit3d_manifest as builder


class ViewClassificationTests(unittest.TestCase):
    def test_thresholds_map_to_front_oblique_and_side(self) -> None:
        self.assertEqual(builder.classify_view(20.0, "front", "left", 4.0), "front")
        self.assertEqual(builder.classify_view(20.0, "rear", "right", 4.0), "rear")
        self.assertEqual(
            builder.classify_view(45.0, "front", "left", 4.0), "oblique_left"
        )
        self.assertEqual(
            builder.classify_view(75.0, "rear", "right", 4.0), "side_right"
        )
        self.assertEqual(builder.classify_view(75.0, "rear", "right", 40.0), "unknown")

    def test_view_geometry_distinguishes_camera_positions(self) -> None:
        joints = np.zeros((2, 25, 3), dtype=np.float64)
        for frame in range(2):
            joints[frame, 1] = [0.0, -0.1, 0.0]
            joints[frame, 4] = [0.0, 0.1, 0.0]
            joints[frame, 11] = [0.0, -0.1, 1.0]
            joints[frame, 14] = [0.0, 0.1, 1.0]

        front = builder.view_geometry(joints, [0, 1], np.array([-3.0, 0.0, 1.0]))
        self.assertEqual(front["view"], "front")
        self.assertLess(front["frontal_axis_dot"], 0.0)

        rear = builder.view_geometry(joints, [0, 1], np.array([3.0, 0.0, 1.0]))
        self.assertEqual(rear["view"], "rear")

        side = builder.view_geometry(joints, [0, 1], np.array([0.0, 3.0, 1.0]))
        self.assertEqual(side["view"], "side_right")
        self.assertAlmostEqual(side["angle_from_frontal_normal_deg"], 90.0, places=6)

        oblique = builder.view_geometry(joints, [0, 1], np.array([3.0, 3.0, 0.0]))
        self.assertEqual(oblique["view"], "oblique_right")
        self.assertAlmostEqual(oblique["angle_from_frontal_normal_deg"], 45.0, places=6)

    def test_subject_split_is_subject_level(self) -> None:
        self.assertEqual(builder.split_for_subject("s03"), "dev")
        self.assertEqual(builder.split_for_subject("s09"), "holdout")
        self.assertEqual(builder.split_for_subject("s12"), "smoke")
        with self.assertRaises(ValueError):
            builder.split_for_subject("unknown")


class MetricPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = json.loads(
            (ROOT / "configs" / "task1_4_exercise_specs.json").read_text()
        )

    def test_side_metrics_remain_candidates_and_disabled(self) -> None:
        policy = builder.metric_policy("pushup", "side_left", self.specs)
        self.assertIn("elbow_flexion_rom", policy["side_metric_candidates"])
        self.assertIn("body_line_angle", policy["side_metric_candidates"])
        self.assertIn("body_line_angle", policy["disabled_metrics"])
        self.assertNotIn("body_line_angle", policy["usable_metrics"])
        self.assertFalse(policy["side_metrics_enabled"])
        self.assertEqual(policy["side_metrics_status"], "pending_t1_5b_validation")

    def test_front_metrics_follow_view_support(self) -> None:
        frontal = builder.metric_policy("pushup", "front", self.specs)
        self.assertIn("elbow_asymmetry_bottom", frontal["usable_metrics"])
        self.assertEqual(frontal["side_metric_candidates"], [])

        side = builder.metric_policy("pushup", "side_left", self.specs)
        self.assertNotIn("elbow_asymmetry_bottom", side["usable_metrics"])
        self.assertIn("elbow_asymmetry_bottom", side["disabled_metrics"])

    def test_metric_groups_match_t1_4_specs(self) -> None:
        base, side, frontal = builder.exercise_metric_groups("squat", self.specs)
        self.assertEqual(
            base, ["rep_duration_cv", "rep_duration_s", "spine_stability_bottom"]
        )
        self.assertEqual(
            side,
            [
                "bottom_knee_angle",
                "hip_depth_ratio",
                "hip_flexion_rom",
                "knee_flexion_rom",
                "torso_lean_bottom",
            ],
        )
        self.assertEqual(
            frontal,
            ["hip_asymmetry_bottom", "knee_asymmetry_bottom", "knee_valgus_bottom"],
        )


class ManifestValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (ROOT / "configs" / "task1_5_fit3d_clips.json").read_text()
        )
        cls.specs = json.loads(
            (ROOT / "configs" / "task1_4_exercise_specs.json").read_text()
        )

    def test_frozen_manifest_passes_schema_validation(self) -> None:
        result = builder.validate_manifest(
            self.manifest, self.specs, check_files=False, require_full_counts=True
        )
        self.assertTrue(result["passed"], result["errors"])
        self.assertIn("train_clips=64", result["checks"])
        self.assertIn("unique_reps=96", result["checks"])

    def test_manifest_contains_reviewed_view_decision(self) -> None:
        overrides = builder.load_view_overrides(
            ROOT / "configs" / "task1_5_view_overrides.json"
        )
        train = [clip for clip in self.manifest["clips"] if clip["has_gt"]]
        self.assertEqual(len(overrides), 64)
        for clip in train:
            self.assertEqual(overrides[clip["clip_id"]], clip["auto_view"])
            self.assertEqual(clip["view"], clip["auto_view"])

    def test_side_candidates_must_not_be_enabled(self) -> None:
        manifest = json.loads(json.dumps(self.manifest))
        clip = next(item for item in manifest["clips"] if item["exercise"] == "pushup")
        candidate = clip["side_metric_candidates"][0]
        clip["usable_metrics"].append(candidate)
        clip["disabled_metrics"].remove(candidate)
        result = builder.validate_manifest(
            manifest, self.specs, check_files=False, require_full_counts=False
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("side candidates" in error for error in result["errors"]))

    def test_duplicate_ids_and_bad_rep_range_are_rejected(self) -> None:
        manifest = json.loads(json.dumps(self.manifest))
        manifest["clips"][1]["clip_id"] = manifest["clips"][0]["clip_id"]
        train = next(clip for clip in manifest["clips"] if clip["has_gt"])
        train["gt"]["rep_frames"][0] = train["annotation_frames"]
        result = builder.validate_manifest(
            manifest, self.specs, check_files=False, require_full_counts=False
        )
        self.assertFalse(result["passed"])
        self.assertIn("clip_id is not unique", result["errors"])
        self.assertTrue(any("rep frame outside" in error for error in result["errors"]))

    def test_override_loading_supports_object_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "overrides.json"
            path.write_text(
                json.dumps({"overrides": {"clip": {"view": "side_left"}}})
            )
            self.assertEqual(builder.load_view_overrides(path), {"clip": "side_left"})


class AuditHelperTests(unittest.TestCase):
    def test_audit_frame_uses_median_rep_plus_offset(self) -> None:
        clip = {
            "trim": {"end_frame": 1000},
            "gt": {"rep_frames": [100, 200, 300, 400, 500, 600]},
        }
        self.assertEqual(audit_task1_5_views.audit_frame(clip, 30), 430)
        clip["trim"]["end_frame"] = 420
        self.assertEqual(audit_task1_5_views.audit_frame(clip, 30), 420)


if __name__ == "__main__":
    unittest.main()
