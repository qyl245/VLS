from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_task1_5d_reps as evaluation


class CountAndPrimaryTests(unittest.TestCase):
    def test_median_rounds_half_up(self) -> None:
        self.assertEqual(evaluation.median_round_half_up([5, 5, 6, 6]), 6)
        self.assertEqual(evaluation.median_round_half_up([4, 5, 5, 5]), 5)

    def test_primary_camera_uses_largest_view_angle(self) -> None:
        clips = [
            {
                "clip_id": "camera_a",
                "camera_id": "a",
                "view": "front",
                "view_geometry": {"angle_from_frontal_normal_deg": 20.0},
            },
            {
                "clip_id": "camera_b",
                "camera_id": "b",
                "view": "side_left",
                "view_geometry": {"angle_from_frontal_normal_deg": 85.0},
            },
            {
                "clip_id": "camera_c",
                "camera_id": "c",
                "view": "oblique_left",
                "view_geometry": {"angle_from_frontal_normal_deg": 60.0},
            },
        ]
        self.assertEqual(evaluation.choose_primary_camera(clips)["clip_id"], "camera_b")


class EventMatchingTests(unittest.TestCase):
    def test_gate_uses_quarter_median_period(self) -> None:
        self.assertEqual(evaluation.event_gate_frames([100, 200, 300]), 25)

    def test_hungarian_matching_is_one_to_one(self) -> None:
        matches = evaluation.match_events([100, 200], [105, 195, 250], 20)
        self.assertEqual(matches, [(0, 0, 5), (1, 1, 5)])

    def test_matching_prioritizes_in_gate_pairs(self) -> None:
        matches = evaluation.match_events(
            [283, 390, 489, 594, 692, 802],
            [396, 512, 592, 719, 811, 935],
            26,
        )
        self.assertEqual(
            matches,
            [(1, 0, 6), (2, 1, 23), (3, 2, 2), (5, 4, 9)],
        )

    def test_rejection_classification_distinguishes_true_and_non_gt(self) -> None:
        rejected = [
            {
                "candidate_rep_id": 1,
                "top_frame": 198,
                "start_frame": 198,
                "end_frame": 240,
                "confidence": 0.9,
                "reject_reasons": "low_local_amplitude",
            },
            {
                "candidate_rep_id": 2,
                "top_frame": 275,
                "start_frame": 275,
                "end_frame": 300,
                "confidence": 0.9,
                "reject_reasons": "low_local_amplitude",
            },
        ]
        rows = evaluation.classify_rejections(
            rejected, [100, 200, 300], [(0, 0, 2)], 20
        )
        self.assertEqual(rows[0]["classification"], "conservative_refusal_true_event")
        self.assertEqual(rows[1]["classification"], "safe_refusal_non_gt")


class ClipEvaluationTests(unittest.TestCase):
    def _write_analysis(self, root: Path) -> Path:
        analysis_dir = root / "analysis"
        clip_dir = analysis_dir / "clip_a"
        clip_dir.mkdir(parents=True)
        reps = [
            {
                "rep_id": 1,
                "top_frame": 105,
                "start_frame": 104,
                "end_frame": 150,
                "confidence": 0.9,
                "status": "complete",
            },
            {
                "rep_id": 2,
                "top_frame": 250,
                "start_frame": 250,
                "end_frame": 290,
                "confidence": 0.9,
                "status": "complete",
            },
            {
                "rep_id": 3,
                "top_frame": 405,
                "start_frame": 405,
                "end_frame": 450,
                "confidence": 0.9,
                "status": "complete",
            },
        ]
        rejected = [
            {
                "candidate_rep_id": 1,
                "top_frame": 196,
                "start_frame": 196,
                "end_frame": 220,
                "confidence": 0.9,
                "reject_reasons": "low_local_amplitude",
                "status": "rejected",
            }
        ]
        payload = {
            "status": "ok",
            "quality_status": "warn",
            "rep_count": 3,
            "segmentation": {
                "method": "aifit",
                "phase_angle": "left_knee_flexion",
                "reps": reps,
                "rejected_reps": rejected,
            },
        }
        (clip_dir / "analysis.json").write_text(json.dumps(payload))
        return analysis_dir

    def test_clip_metrics_flag_high_confidence_false_positive(self) -> None:
        clip = {
            "clip_id": "clip_a",
            "subject_id": "s03",
            "camera_id": "camera_a",
            "exercise": "squat",
            "split": "dev",
            "view": "front",
            "view_geometry": {"angle_from_frontal_normal_deg": 20.0},
            "gt": {"rep_frames": [100, 200, 300, 400]},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis_dir = self._write_analysis(Path(temp_dir))
            summary, events, matches, rejected, intervals = evaluation.evaluate_clip(
                clip, analysis_dir
            )
        self.assertEqual(summary["pred_rep_count"], 3)
        self.assertEqual(summary["event_tp"], 2)
        self.assertEqual(summary["event_fp"], 1)
        self.assertEqual(summary["event_fn"], 2)
        self.assertEqual(summary["high_confidence_fp_count"], 1)
        self.assertEqual(summary["rejected_true_event_count"], 1)
        self.assertEqual(
            rejected[0]["classification"], "conservative_refusal_true_event"
        )
        self.assertEqual(len(events), 3)
        self.assertEqual(len(matches), 4)
        self.assertEqual(intervals, [])


class SubjectActionAggregationTests(unittest.TestCase):
    def test_consensus_is_computed_per_action_not_per_camera(self) -> None:
        clips = []
        clip_rows = []
        for index, (count, angle) in enumerate(((7, 70.0), (6, 80.0), (5, 60.0), (5, 50.0))):
            camera_id = f"camera_{index}"
            clips.append(
                {
                    "clip_id": f"clip_{index}",
                    "subject_id": "s03",
                    "exercise": "pushup",
                    "camera_id": camera_id,
                    "view": "side_left",
                    "view_geometry": {"angle_from_frontal_normal_deg": angle},
                }
            )
            clip_rows.append(
                {
                    "clip_id": f"clip_{index}",
                    "subject_id": "s03",
                    "exercise": "pushup",
                    "split": "dev",
                    "camera_id": camera_id,
                    "view": "side_left",
                    "view_angle_deg": angle,
                    "gt_rep_count": 6,
                    "pred_rep_count": count,
                    "count_usable": True,
                    "event_tp": count,
                    "event_fp": 0,
                    "event_fn": 6 - count,
                    "event_abs_error_median_frames": 3.0,
                    "event_abs_error_p90_frames": 5.0,
                    "rejected_candidate_count": 0,
                    "high_confidence_fp_count": 0,
                }
            )
        rows = evaluation.build_subject_action_rows(clips, clip_rows)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["consensus_pred_count"], 6)
        self.assertTrue(rows[0]["consensus_count_exact"])
        self.assertEqual(rows[0]["primary_camera_id"], "camera_1")
        self.assertEqual(rows[0]["primary_pred_count"], 6)
        self.assertEqual(rows[0]["primary_view_angle_deg"], 80.0)


if __name__ == "__main__":
    unittest.main()
