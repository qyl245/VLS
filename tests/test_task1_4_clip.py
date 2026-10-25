from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_clip


class TimeAndConfigTests(unittest.TestCase):
    def test_source_time_is_absolute_video_time(self) -> None:
        self.assertEqual(analyze_clip.source_time_s(1937, 30.0), 64.566667)
        self.assertEqual(analyze_clip.clip_time_s(1937, 1892, 30.0), 1.5)

    def test_squat_and_pushup_boundary_config_are_explicit(self) -> None:
        specs = json.loads((ROOT / "configs" / "task1_4_exercise_specs.json").read_text())
        for exercise, landmark in (
            ("squat", "standing_knee_extended"),
            ("pushup", "arms_extended_top"),
        ):
            guidance = analyze_clip.phase_guidance_for(exercise, specs)
            self.assertEqual(guidance.phase_landmark, landmark)
            self.assertEqual(guidance.polarity, "peak")
            self.assertEqual(guidance.boundary_snap.mode, "plateau_entry")


class PhaseCycleTests(unittest.TestCase):
    def test_phase_cycles_use_top_boundaries(self) -> None:
        fps = 30.0
        frame_count = 240
        x = np.arange(frame_count, dtype=np.float64)
        signal = 125.0 + 50.0 * np.cos(2.0 * np.pi * x / 60.0)
        boundaries, evidence = analyze_clip.find_phase_cycle_boundaries(
            signal, fps, 0.4, 5.0
        )
        self.assertEqual(evidence["status"], "applied")
        self.assertGreaterEqual(len(boundaries), 4)
        for frame in boundaries:
            self.assertGreaterEqual(signal[frame], evidence["high_threshold"])

    def test_partial_and_misaligned_tail_reps_are_rejected(self) -> None:
        phase_values = 135.0 + 45.0 * np.sin(np.linspace(0, 8 * np.pi, 400))
        valid = {
            "start_frame": 0,
            "end_frame": 59,
            "bottom_frame": 30,
            "start_angle_deg": 180.0,
            "end_angle_deg": 180.0,
            "confidence": 0.9,
            "shape_similarity": 0.9,
        }
        malformed = {
            **valid,
            "start_frame": 120,
            "end_frame": 160,
            "bottom_frame": 121,
            "start_angle_deg": 90.0,
            "end_angle_deg": 90.0,
        }
        reps = []
        for rep_id in range(1, 4):
            reps.append({**valid, "rep_id": rep_id, "start_frame": (rep_id - 1) * 60})
        for rep_id in range(1, 4):
            reps[rep_id - 1]["end_frame"] = rep_id * 60 - 1
            reps[rep_id - 1]["bottom_frame"] = rep_id * 60 - 30
        reps.append({**malformed, "rep_id": 4})
        kept, rejected = analyze_clip.assess_rep_records(reps, phase_values)
        self.assertEqual(len(kept), 3)
        self.assertEqual([rep["rep_id"] for rep in kept], [1, 2, 3])
        self.assertEqual(len(rejected), 1)
        self.assertIn("missing_interior_bottom", rejected[0]["reject_reasons"])


class QualityAndSummaryTests(unittest.TestCase):
    def test_front_view_disables_sagittal_metrics(self) -> None:
        clip = {
            "view": "front",
            "usable_metrics": ["rep_duration_s", "knee_asymmetry_bottom"],
            "disabled_metrics": ["knee_flexion_rom", "torso_lean_bottom"],
        }
        pose_quality = {
            "frame_coverage": 1.0,
            "pose_detection_rate": 1.0,
            "q10_key_visibility": 0.9,
            "body_in_frame_rate": 1.0,
            "mean_key_visibility": 0.9,
            "key_joint_visible_rate": 0.9,
            "all_exercise_joints_visible_rate": 0.4,
            "all_required_visible_rate": 0.4,
        }
        segmentation = {"reps": [{"rep_id": 1}, {"rep_id": 2}, {"rep_id": 3}], "rejected_reps": []}
        quality = analyze_clip.build_quality(clip, pose_quality, segmentation)
        self.assertEqual(quality["status"], "pass")
        self.assertTrue(quality["view_support"]["sagittal_metrics_disabled"])
        self.assertIn("all_exercise_joints_visible_rate", quality["checks"])

    def test_summary_is_rebuilt_from_all_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            for clip_id, reps in (("a", 4), ("b", 5)):
                analyze_clip.write_json(
                    output_dir / clip_id / "analysis.json",
                    {
                        "schema_version": analyze_clip.SCHEMA_VERSION,
                        "clip_id": clip_id,
                        "exercise": "squat",
                        "view": "front",
                        "split": "dev",
                        "status": "ok",
                        "quality_status": "pass",
                        "rep_count": reps,
                        "rejected_rep_count": 1,
                        "quality": {"warnings": [], "failures": []},
                        "segmentation": {"angle": "left_knee_flexion", "method": "aifit"},
                    },
                )
            summary = analyze_clip.rebuild_summary(output_dir)
            self.assertEqual(summary["clip_count"], 2)
            self.assertEqual(summary["complete_rep_count"], 9)
            self.assertEqual(summary["rejected_rep_count"], 2)


if __name__ == "__main__":
    unittest.main()
