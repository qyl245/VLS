from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class PostureGateTests(unittest.TestCase):
    def test_pushup_gate_rejects_upright_reps_and_renumbers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            angles_dir = Path(temp_dir)
            landmarks = []
            for frame in range(100, 181):
                world = [[0.0, 0.0, 0.0] for _ in range(33)]
                shoulder_y = -0.10 if frame <= 140 else -0.50
                world[11][1] = shoulder_y
                world[12][1] = shoulder_y
                world[23][1] = 0.0
                world[24][1] = 0.0
                landmarks.append(
                    {"frame_id": frame - 100, "source_frame": frame, "landmarks_world": world}
                )
            (angles_dir / "landmarks.json").write_text(json.dumps(landmarks))

            specs = json.loads(
                (ROOT / "configs" / "task1_4_exercise_specs.json").read_text()
            )
            clip = {
                "exercise": "pushup",
                "trim": {"start_frame": 100, "end_frame": 180},
            }
            segmentation = {
                "status": "ok",
                "candidate_sources": {"selected": "aifit"},
                "reps": [
                    {
                        "rep_id": 1,
                        "start_frame": 110,
                        "end_frame": 140,
                        "bottom_frame": 125,
                        "start_time_s": 110 / 30.0,
                        "end_time_s": 140 / 30.0,
                    },
                    {
                        "rep_id": 2,
                        "start_frame": 150,
                        "end_frame": 180,
                        "bottom_frame": 165,
                        "start_time_s": 150 / 30.0,
                        "end_time_s": 180 / 30.0,
                    },
                ],
                "phases": [
                    {"rep_id": 1, "phase": "bottom", "frame_index": 125},
                    {"rep_id": 2, "phase": "bottom", "frame_index": 165},
                ],
                "rejected_reps": [],
            }
            gated = analyze_clip.apply_exercise_posture_gate(
                segmentation, clip, angles_dir, specs
            )

        self.assertEqual(len(gated["reps"]), 1)
        self.assertEqual(gated["reps"][0]["rep_id"], 1)
        self.assertEqual([phase["frame_index"] for phase in gated["phases"]], [125])
        self.assertEqual(len(gated["rejected_reps"]), 1)
        self.assertEqual(
            gated["rejected_reps"][0]["reject_reasons"], "exercise_posture_mismatch"
        )
        self.assertEqual(
            gated["posture_gate"]["reps"][1]["bottom_value"], 0.5
        )


class SegmentationCandidateTests(unittest.TestCase):
    def test_phase_signal_candidate_wins_when_it_has_more_valid_reps(self) -> None:
        clip = {"exercise": "pushup"}
        specs = {"exercises": {"pushup": {"phase": {"signal": "left_elbow_flexion"}}}}
        calls = []

        def fake_segment(clip, specs, fps, angle_override=None):
            calls.append(angle_override)
            angle = angle_override or "right_elbow_flexion"
            count = 2 if angle_override is None else 5
            return {
                "angle": angle,
                "reps": [{"rep_id": index + 1} for index in range(count)],
                "rejected_reps": [],
            }

        def fake_gate(segmentation, clip, angles_dir, specs):
            return segmentation

        with mock.patch.object(
            analyze_clip, "segment_clip", side_effect=fake_segment
        ), mock.patch.object(
            analyze_clip, "apply_exercise_posture_gate", side_effect=fake_gate
        ):
            selected = analyze_clip.segment_with_posture_gate(
                clip, specs, 30.0, Path("/tmp")
            )

        self.assertEqual(selected["angle"], "left_elbow_flexion")
        self.assertEqual(selected["segmentation_candidates"]["selected"], "phase_signal")
        self.assertEqual(
            selected["segmentation_candidates"]["primary"]["valid_rep_count"], 2
        )
        self.assertEqual(
            selected["segmentation_candidates"]["phase_signal"]["valid_rep_count"], 5
        )
        self.assertEqual(calls, [None, "left_elbow_flexion"])

    def test_primary_candidate_wins_when_alternative_has_no_more_reps(self) -> None:
        clip = {"exercise": "squat"}
        specs = {"exercises": {"squat": {"phase": {"signal": "left_knee_flexion"}}}}

        def fake_segment(clip, specs, fps, angle_override=None):
            angle = angle_override or "right_knee_flexion"
            count = 10 if angle_override is None else 3
            return {
                "angle": angle,
                "reps": [{"rep_id": index + 1} for index in range(count)],
                "rejected_reps": [],
            }

        def fake_gate(segmentation, clip, angles_dir, specs):
            return segmentation

        with mock.patch.object(
            analyze_clip, "segment_clip", side_effect=fake_segment
        ), mock.patch.object(
            analyze_clip, "apply_exercise_posture_gate", side_effect=fake_gate
        ):
            selected = analyze_clip.segment_with_posture_gate(
                clip, specs, 30.0, Path("/tmp")
            )

        self.assertEqual(selected["angle"], "right_knee_flexion")
        self.assertEqual(selected["segmentation_candidates"]["selected"], "primary")


if __name__ == "__main__":
    unittest.main()
