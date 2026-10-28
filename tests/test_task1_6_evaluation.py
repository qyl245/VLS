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

import audit_task1_6_failures as audit
import evaluate_task1_6 as evaluation


def make_event(frame: int, confidence: float, *, set_ids: list[str] | None = None) -> dict:
    return {
        "frame": frame,
        "confidence": {"combined": confidence},
        "set_ids": set_ids or [],
    }


def make_clip(
    clip_id: str = "clip_a",
    *,
    cluster_id: str = "cluster_a",
    gt_frames: list[int] | None = None,
) -> dict:
    return {
        "clip_id": clip_id,
        "cluster_id": cluster_id,
        "action": "pushup",
        "view": "side_left",
        "camera_id": "camera_a",
        "subject_id": "s01",
        "fps": 30.0,
        "gt": {
            "events": [
                {"event": "top", "clip_frame": frame}
                for frame in (gt_frames or [])
            ]
        },
    }


def make_analysis(frames: list[int], confidence: float = 0.9) -> dict:
    return {
        "clip_id": "clip_a",
        "status": "ok",
        "sets": [],
        "diagnostics": {
            "selected_top_events": [
                make_event(frame, confidence, set_ids=["set_1"])
                for frame in frames
            ]
        },
    }


class EventMetricTests(unittest.TestCase):
    def test_event_metrics_counts_matches_and_high_confidence_fp(self) -> None:
        metrics = evaluation.event_metrics(
            [100, 200, 300],
            [
                make_event(102, 0.9, set_ids=["set_1"]),
                make_event(204, 0.8, set_ids=["set_1"]),
                make_event(305, 0.7, set_ids=["set_1"]),
                make_event(500, 0.6),
                make_event(520, 0.2),
            ],
            high_confidence_threshold=0.5,
        )
        self.assertEqual(metrics["tp"], 3)
        self.assertEqual(metrics["fp"], 2)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["high_confidence_event_fp"], 1)
        self.assertEqual(metrics["outside_set_high_confidence_fp"], 1)
        self.assertTrue(metrics["first_gt_matched"])
        self.assertTrue(metrics["last_gt_matched"])
        self.assertEqual(metrics["event_abs_error_median_frames"], 4.0)

    def test_event_metrics_handles_empty_gt_and_predictions(self) -> None:
        metrics = evaluation.event_metrics(
            [], [], high_confidence_threshold=0.5
        )
        self.assertEqual(metrics["tp"], 0)
        self.assertFalse(metrics["first_gt_matched"])
        self.assertFalse(metrics["last_gt_matched"])
        self.assertIsNone(metrics["event_recall"])
        self.assertIsNone(metrics["event_precision"])

    def test_event_metrics_does_not_collapse_with_head_tail_extras(self) -> None:
        metrics = evaluation.event_metrics(
            [283, 390, 489, 594, 692, 802],
            [
                make_event(396, 0.9, set_ids=["set_1"]),
                make_event(512, 0.8, set_ids=["set_1"]),
                make_event(592, 0.7, set_ids=["set_1"]),
                make_event(719, 0.6, set_ids=["set_1"]),
                make_event(811, 0.5, set_ids=["set_1"]),
                make_event(935, 0.4, set_ids=["set_1"]),
            ],
            high_confidence_threshold=0.5,
        )
        self.assertEqual(metrics["tp"], 4)
        self.assertEqual(metrics["fp"], 2)
        self.assertEqual(metrics["fn"], 2)
        self.assertEqual(metrics["high_confidence_event_fp"], 1)

    def test_aggregate_event_rows_supports_legacy_fp_key(self) -> None:
        rows = [
            {
                "cluster_id": "cluster_a",
                "gt_event_count": 3,
                "pred_event_count": 4,
                "tp": 2,
                "first_gt_matched": True,
                "last_gt_matched": False,
                "high_confidence_fp": 2,
                "outside_set_high_confidence_fp": 1,
                "_abs_errors": [2, 4],
            },
            {
                "cluster_id": "cluster_b",
                "gt_event_count": 2,
                "pred_event_count": 2,
                "tp": 2,
                "first_gt_matched": True,
                "last_gt_matched": True,
                "high_confidence_event_fp": 1,
                "outside_set_high_confidence_fp": 0,
                "_abs_errors": [1],
            },
        ]
        summary = evaluation.aggregate_event_rows(rows)
        self.assertEqual(summary["clip_count"], 2)
        self.assertEqual(summary["cluster_count"], 2)
        self.assertEqual(summary["tp"], 4)
        self.assertEqual(summary["fp"], 2)
        self.assertEqual(summary["fn"], 1)
        self.assertEqual(summary["high_confidence_event_fp"], 3)
        self.assertEqual(summary["outside_set_high_confidence_fp"], 1)
        self.assertEqual(summary["count_mae"], 0.5)
        self.assertEqual(summary["count_exact_correct"], 1)
        self.assertEqual(summary["first_event_recall"], 1.0)
        self.assertEqual(summary["last_event_recall"], 0.5)

    def test_count_consensus_uses_rounded_median(self) -> None:
        rows = [
            {"cluster_id": "cluster_a", "gt_event_count": 6, "pred_event_count": 5},
            {"cluster_id": "cluster_a", "gt_event_count": 6, "pred_event_count": 6},
            {"cluster_id": "cluster_a", "gt_event_count": 6, "pred_event_count": 8},
            {"cluster_id": "cluster_b", "gt_event_count": 4, "pred_event_count": 4},
            {"cluster_id": "cluster_b", "gt_event_count": 4, "pred_event_count": 5},
        ]
        summary = evaluation.count_consensus(rows)
        self.assertEqual(summary["cluster_count"], 2)
        self.assertEqual(summary["rows"][0]["consensus_count"], 6)
        self.assertEqual(summary["rows"][1]["consensus_count"], 5)
        self.assertEqual(summary["count_mae"], 0.5)
        self.assertEqual(summary["exact_count_correct"], 1)

    def test_count_consensus_rejects_inconsistent_gt(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent GT count"):
            evaluation.count_consensus(
                [
                    {"cluster_id": "cluster_a", "gt_event_count": 6, "pred_event_count": 6},
                    {"cluster_id": "cluster_a", "gt_event_count": 5, "pred_event_count": 5},
                ]
            )

    def test_top_to_top_interval_error_uses_matched_consecutive_events(self) -> None:
        clip = make_clip(gt_frames=[100, 200, 300])
        analyses = {"clip_a": make_analysis([103, 204, 305])}
        self.assertEqual(
            evaluation.top_to_top_interval_errors([clip], analyses),
            [1, 1],
        )
        self.assertEqual(
            evaluation.top_to_top_interval_errors(
                [clip], analyses, selected_clip_ids=set()
            ),
            [],
        )


class RobustnessAndControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.signals = {
            "left_elbow_flexion": np.arange(10, dtype=float),
            "right_elbow_flexion": np.arange(10, dtype=float) + 100.0,
        }

    def test_transform_signals_core_invariances(self) -> None:
        scaled, fps = evaluation.transform_signals(
            self.signals, "amplitude_scale_0.8", fps=30.0, seed=1
        )
        self.assertEqual(fps, 30.0)
        self.assertAlmostEqual(float(np.median(scaled["left_elbow_flexion"])), 4.5)
        self.assertAlmostEqual(float(scaled["left_elbow_flexion"][0]), 0.9)

        offset, _ = evaluation.transform_signals(
            self.signals, "offset_plus_15", fps=30.0, seed=1
        )
        self.assertEqual(offset["left_elbow_flexion"][0], 15.0)

        flipped, _ = evaluation.transform_signals(
            self.signals, "flip_lr", fps=30.0, seed=1
        )
        self.assertEqual(
            set(flipped),
            {"right_elbow_flexion", "left_elbow_flexion"},
        )
        np.testing.assert_array_equal(
            flipped["left_elbow_flexion"], self.signals["right_elbow_flexion"]
        )

        resampled, target_fps = evaluation.transform_signals(
            self.signals, "fps_15", fps=30.0, seed=1
        )
        self.assertEqual(target_fps, 15.0)
        self.assertEqual(len(resampled["left_elbow_flexion"]), 5)
        self.assertAlmostEqual(float(resampled["left_elbow_flexion"][1]), 2.0)

    def test_transform_signals_noise_missing_and_idle_padding(self) -> None:
        noisy_a, _ = evaluation.transform_signals(
            self.signals, "noise_sigma_2", fps=30.0, seed=7
        )
        noisy_b, _ = evaluation.transform_signals(
            self.signals, "noise_sigma_2", fps=30.0, seed=7
        )
        np.testing.assert_array_equal(
            noisy_a["left_elbow_flexion"], noisy_b["left_elbow_flexion"]
        )
        self.assertFalse(
            np.array_equal(noisy_a["left_elbow_flexion"], self.signals["left_elbow_flexion"])
        )

        missing, _ = evaluation.transform_signals(
            self.signals, "missing_5pct", fps=30.0, seed=7
        )
        self.assertTrue(np.isnan(missing["left_elbow_flexion"]).any())

        padded, _ = evaluation.transform_signals(
            self.signals, "idle_pad_2s", fps=30.0, seed=7
        )
        self.assertEqual(len(padded["left_elbow_flexion"]), 130)
        self.assertEqual(padded["left_elbow_flexion"][0], 4.5)

    def test_transform_rejects_unknown_transform(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown transformation"):
            evaluation.transform_signals(
                self.signals, "unknown", fps=30.0, seed=1
            )

    def test_negative_control_series_are_deterministic(self) -> None:
        first = evaluation.negative_control_series(
            "idle_constant", fps=30.0, seconds=2.0
        )
        second = evaluation.negative_control_series(
            "idle_constant", fps=30.0, seconds=2.0
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 60)
        self.assertAlmostEqual(float(np.mean(first)), 120.0, delta=0.2)
        with self.assertRaisesRegex(ValueError, "unknown negative control"):
            evaluation.negative_control_series(
                "unknown", fps=30.0, seconds=2.0
            )

    def test_negative_controls_emit_no_complete_reps(self) -> None:
        segmenter_config = json.loads(
            (ROOT / "configs" / "task1_6_segmenter.json").read_text()
        )
        chain_config = json.loads(
            (ROOT / "configs" / "task1_6_chain.json").read_text()
        )
        rows = evaluation.evaluate_negative_controls(
            segmenter_config=segmenter_config,
            chain_config=chain_config,
        )
        by_id = {row["control_id"]: row for row in rows}
        self.assertEqual(by_id["idle_constant"]["status"], "no_motion")
        self.assertTrue(all(row["passed"] for row in rows))


class SealedAndGateTests(unittest.TestCase):
    def test_load_sealed_annotations_only_accepts_reviewed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "template.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "segment_id",
                        "gt_start_frame",
                        "gt_end_frame",
                        "review_status",
                        "boundary_confidence",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "segment_id": "accept",
                        "gt_start_frame": "10",
                        "gt_end_frame": "20",
                        "review_status": "accept",
                        "boundary_confidence": "high",
                    }
                )
                writer.writerow(
                    {
                        "segment_id": "adjust",
                        "gt_start_frame": "30",
                        "gt_end_frame": "40",
                        "review_status": "adjust",
                        "boundary_confidence": "medium",
                    }
                )
                writer.writerow(
                    {
                        "segment_id": "pending",
                        "gt_start_frame": "50",
                        "gt_end_frame": "60",
                        "review_status": "pending",
                        "boundary_confidence": "high",
                    }
                )
                writer.writerow(
                    {
                        "segment_id": "delete",
                        "gt_start_frame": "70",
                        "gt_end_frame": "80",
                        "review_status": "delete",
                        "boundary_confidence": "low",
                    }
                )
                writer.writerow(
                    {
                        "segment_id": "invalid",
                        "gt_start_frame": "90",
                        "gt_end_frame": "85",
                        "review_status": "accept",
                        "boundary_confidence": "high",
                    }
                )
            annotations = evaluation.load_sealed_annotations(path)
        self.assertEqual(set(annotations), {"accept", "adjust"})
        self.assertEqual(annotations["accept"][0]["start_frame"], 10)
        self.assertEqual(annotations["adjust"][0]["review_status"], "adjust")
        self.assertEqual(evaluation.load_sealed_annotations(Path("/missing")), {})

    def test_build_go_no_go_marks_qevd_not_evaluable(self) -> None:
        all_cameras = {
            "first_event_recall": 0.8,
            "last_event_recall": 0.8,
            "high_confidence_event_fp": 0,
        }
        primary = {
            "event_abs_error_median_frames": 5.0,
            "event_abs_error_p90_frames": 12.0,
            "high_confidence_event_fp": 0,
        }
        consensus = {
            "cluster_count": 8,
            "count_mae": 0.0,
            "exact_count_correct": 8,
        }
        confidence = {
            "calibrated_threshold": {"high_confidence_fp": 0}
        }
        sealed = {"annotation_ready_count": 0}
        gates = evaluation.build_go_no_go(
            all_cameras=all_cameras,
            primary=primary,
            consensus=consensus,
            confidence_summary=confidence,
            top_interval_median=3.0,
            sealed_summary=sealed,
        )
        fit3d = [row for row in gates if row["dataset"] == evaluation.FIT3D_DATASET]
        qevd = [row for row in gates if row["dataset"] == evaluation.QEVD_DATASET]
        self.assertEqual(len(fit3d), 10)
        self.assertTrue(all(row["status"] == "pass" for row in fit3d))
        self.assertEqual(len(qevd), 7)
        self.assertTrue(all(row["status"] == "not_evaluable" for row in qevd))
        self.assertEqual(
            next(
                row
                for row in fit3d
                if row["gate_id"] == "fit3d_exact_count"
            )["target"],
            ">= 7/8",
        )

    def test_build_go_no_go_fails_fixed_metric_gates(self) -> None:
        gates = evaluation.build_go_no_go(
            all_cameras={
                "first_event_recall": 0.5,
                "last_event_recall": 0.5,
                "high_confidence_event_fp": 2,
            },
            primary={
                "event_abs_error_median_frames": 6.0,
                "event_abs_error_p90_frames": 13.0,
                "high_confidence_event_fp": 1,
            },
            consensus={
                "cluster_count": 8,
                "count_mae": 0.5,
                "exact_count_correct": 6,
            },
            confidence_summary={
                "calibrated_threshold": {"high_confidence_fp": 1}
            },
            top_interval_median=4.0,
            sealed_summary={"annotation_ready_count": 0},
        )
        failed = {row["gate_id"] for row in gates if row["status"] == "fail"}
        self.assertEqual(
            failed,
            {
                "fit3d_exact_count",
                "fit3d_first_event_recall",
                "fit3d_last_event_recall",
                "fit3d_primary_event_median",
                "fit3d_primary_event_p90",
                "fit3d_primary_high_confidence_fp",
                "fit3d_all_camera_high_confidence_fp",
                "fit3d_complete_rep_high_confidence_fp",
                "fit3d_top_to_top_interval_not_degraded",
            },
        )

    def test_load_t1_5_baseline_preserves_event_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            match_fields = [
                "clip_id",
                "subject_id",
                "exercise",
                "split",
                "camera_role",
                "gt_event_index",
                "match_status",
                "pred_event_index",
                "abs_error_frames",
            ]
            matches = [
                {
                    "clip_id": "clip_a",
                    "subject_id": "s01",
                    "exercise": "pushup",
                    "split": "holdout",
                    "camera_role": "primary_closest_to_side",
                    "gt_event_index": "1",
                    "match_status": "matched",
                    "pred_event_index": "1",
                    "abs_error_frames": "2",
                },
                {
                    "clip_id": "clip_a",
                    "subject_id": "s01",
                    "exercise": "pushup",
                    "split": "holdout",
                    "camera_role": "primary_closest_to_side",
                    "gt_event_index": "2",
                    "match_status": "matched",
                    "pred_event_index": "2",
                    "abs_error_frames": "5",
                },
            ]
            event_fields = [
                "clip_id",
                "split",
                "camera_role",
                "match_status",
                "high_confidence_false_positive",
            ]
            events = [
                {
                    "clip_id": "clip_a",
                    "split": "holdout",
                    "camera_role": "primary_closest_to_side",
                    "match_status": "matched",
                    "high_confidence_false_positive": "False",
                },
                {
                    "clip_id": "clip_a",
                    "split": "holdout",
                    "camera_role": "primary_closest_to_side",
                    "match_status": "matched",
                    "high_confidence_false_positive": "False",
                },
                {
                    "clip_id": "clip_a",
                    "split": "holdout",
                    "camera_role": "primary_closest_to_side",
                    "match_status": "false_positive",
                    "high_confidence_false_positive": "True",
                },
            ]
            subject_fields = [
                "split",
                "consensus_abs_count_error",
                "consensus_count_exact",
            ]
            subjects = [
                {
                    "split": "holdout",
                    "consensus_abs_count_error": "0",
                    "consensus_count_exact": "True",
                }
            ]
            interval_fields = ["split", "interval_error_frames"]
            intervals = [{"split": "holdout", "interval_error_frames": "3"}]

            def write_rows(name: str, fields: list[str], rows: list[dict[str, str]]) -> None:
                with (directory / name).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)

            write_rows("rep_matches.csv", match_fields, matches)
            write_rows("rep_events.csv", event_fields, events)
            write_rows("subject_action_summary.csv", subject_fields, subjects)
            write_rows("rep_intervals.csv", interval_fields, intervals)
            baseline = evaluation.load_t1_5_baseline(directory)

        self.assertEqual(
            baseline["all_cameras"]["event_abs_error_median_frames"], 3.5
        )
        self.assertEqual(baseline["all_cameras"]["event_abs_error_p90_frames"], 4.7)
        self.assertEqual(baseline["all_cameras"]["high_confidence_event_fp"], 1)
        self.assertEqual(baseline["count_consensus"]["count_mae"], 0.0)
        self.assertEqual(
            baseline["count_consensus"]["exact_count_correct"], 1
        )
        self.assertEqual(
            baseline["top_to_top_interval_median_frames"], 3.0
        )


class ReproducibilityTests(unittest.TestCase):
    def test_reproducibility_tracks_scope_and_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "a.csv").write_text("value\n1\n")
            (output_dir / "b.csv").write_text("value\n2\n")
            scope = {"skip_robustness": False, "skip_sealed": False}

            first = evaluation.reproducibility_check(
                output_dir, ["a.csv", "b.csv"], scope=scope
            )
            self.assertFalse(first["checked"])
            self.assertEqual(first["reason"], "no_previous_run")

            second = evaluation.reproducibility_check(
                output_dir, ["a.csv", "b.csv"], scope=scope
            )
            self.assertTrue(second["checked"])
            self.assertTrue(second["identical_csv"])

            (output_dir / "a.csv").write_text("value\n3\n")
            changed = evaluation.reproducibility_check(
                output_dir, ["a.csv", "b.csv"], scope=scope
            )
            self.assertTrue(changed["checked"])
            self.assertFalse(changed["identical_csv"])
            self.assertEqual(changed["reason"], "csv_hash_mismatch")

            other_scope = {"skip_robustness": True, "skip_sealed": False}
            scope_changed = evaluation.reproducibility_check(
                output_dir, ["a.csv", "b.csv"], scope=other_scope
            )
            self.assertFalse(scope_changed["checked"])
            self.assertIsNone(scope_changed["identical_csv"])
            self.assertEqual(scope_changed["reason"], "scope_changed")
            payload = json.loads(
                (output_dir / "artifact_hashes.json").read_text()
            )
            self.assertEqual(payload["scope"], other_scope)
            self.assertIn("artifacts", payload)


def make_clip_row(**overrides: object) -> dict[str, str]:
    row = {
        "clip_id": "clip_a",
        "cluster_id": "cluster_a",
        "subject_id": "s01",
        "camera_id": "camera_a",
        "action": "pushup",
        "view": "side_left",
        "is_primary_camera": "True",
        "count_error": "0",
        "first_gt_matched": "True",
        "last_gt_matched": "True",
        "event_abs_error_median_frames": "4",
        "event_abs_error_p90_frames": "8",
        "high_confidence_event_fp": "0",
        "high_confidence_rep_fp": "0",
        "no_confident_set": "False",
    }
    row.update(overrides)
    return {key: str(value) for key, value in row.items()}


class FailureAuditTests(unittest.TestCase):
    def test_failure_categories_and_severity(self) -> None:
        row = make_clip_row(
            count_error="1",
            first_gt_matched="False",
            last_gt_matched="False",
            high_confidence_event_fp="2",
            high_confidence_rep_fp="1",
            event_abs_error_p90_frames="17",
        )
        categories = audit.clip_failure_categories(row, is_primary=True)
        self.assertEqual(
            categories,
            [
                "first_event_miss",
                "last_event_miss",
                "count_error",
                "event_high_conf_fp",
                "rep_high_conf_fp",
                "event_p90_exceeds_gate",
            ],
        )
        self.assertEqual(audit.severity_score(row, categories), 11.0)
        non_primary = audit.clip_failure_categories(row, is_primary=False)
        self.assertNotIn("event_p90_exceeds_gate", non_primary)

    def test_build_failure_index_prioritizes_and_uses_zero_frame(self) -> None:
        rows = [
            make_clip_row(
                clip_id="bad",
                last_gt_matched="False",
                count_error="1",
            ),
            make_clip_row(clip_id="clean"),
        ]
        matches = [
            {
                "clip_id": "bad",
                "match_status": "false_negative",
                "gt_frame": "0",
                "pred_frame": "",
                "abs_error_frames": "",
            },
            {
                "clip_id": "clean",
                "match_status": "matched",
                "gt_frame": "0",
                "pred_frame": "0",
                "abs_error_frames": "0",
            },
        ]
        failures = audit.build_failure_index(rows, matches)
        self.assertEqual(failures[0]["clip_id"], "bad")
        self.assertEqual(failures[0]["priority"], "high")
        self.assertEqual(failures[0]["evidence_gt_frame"], 0)
        self.assertEqual(failures[1]["clip_id"], "clean")
        self.assertEqual(failures[1]["priority"], "none")
        self.assertEqual(
            audit.evidence_frames("clean", matches, []),
            (0, 0, "largest_matched_error"),
        )

    def test_stratified_failures_and_empty_inputs(self) -> None:
        failures = audit.build_failure_index(
            [
                make_clip_row(clip_id="a", action="pushup", count_error="1"),
                make_clip_row(clip_id="b", action="squat"),
            ],
            [],
        )
        strata = audit.stratified_failures(failures, "action")
        by_action = {row["action"]: row for row in strata}
        self.assertEqual(by_action["pushup"]["failed_clip_count"], 1)
        self.assertEqual(by_action["squat"]["failed_clip_count"], 0)
        self.assertEqual(audit.build_failure_index([], []), [])
        self.assertEqual(audit.stratified_failures([], "action"), [])

    def test_build_qevd_review_routes_each_status(self) -> None:
        rows = [
            {
                "clip_id": "conflict",
                "action": "pushup",
                "status": "ok",
                "quality_gate_conflict": "True",
                "gt_evaluable": "False",
            },
            {
                "clip_id": "pending",
                "action": "pushup",
                "status": "ok",
                "quality_gate_conflict": "False",
                "gt_evaluable": "False",
            },
            {
                "clip_id": "unsupported",
                "action": "good_morning",
                "status": "unsupported_action_config",
                "quality_gate_conflict": "False",
                "gt_evaluable": "False",
            },
            {
                "clip_id": "missing",
                "action": "squat",
                "status": "preprocessed_input_missing",
                "quality_gate_conflict": "False",
                "gt_evaluable": "False",
            },
        ]
        review = {row["clip_id"]: row for row in audit.build_qevd_review(rows)}
        self.assertEqual(
            review["conflict"]["required_action"],
            "review_quality_gate_and_annotate_before_acceptance",
        )
        self.assertEqual(
            review["pending"]["required_action"], "pending_human_annotation"
        )
        self.assertEqual(
            review["unsupported"]["required_action"],
            "add_frozen_action_contract_or_confirm_zero_rep_expectation",
        )
        self.assertEqual(
            review["missing"]["required_action"], "generate_preprocessed_input"
        )
        self.assertEqual(audit.build_qevd_review([]), [])

    def test_build_qevd_review_routes_quality_and_motion_exclusions(self) -> None:
        rows = [
            {
                "clip_id": "quality",
                "action": "pushup",
                "status": "rejected_quality",
                "quality_action": "blocked",
                "excluded_by_quality": "True",
                "excluded_by_motion": "False",
                "gt_evaluable": "False",
            },
            {
                "clip_id": "idle",
                "action": "pushup",
                "status": "no_motion",
                "quality_action": "reported",
                "excluded_by_quality": "False",
                "excluded_by_motion": "True",
                "gt_evaluable": "False",
            },
        ]
        review = {row["clip_id"]: row for row in audit.build_qevd_review(rows)}
        self.assertEqual(
            review["quality"]["required_action"], "excluded_by_quality_gate"
        )
        self.assertEqual(
            review["idle"]["required_action"],
            "motion_refused_no_reportable_reps",
        )

    def test_svg_figures_encode_status_without_fake_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            figures_dir = Path(tmp)
            audit.build_figures(
                figures_dir,
                go_no_go=[
                    {"dataset": "Fit3D", "gate_id": "pass_gate", "status": "pass"},
                    {"dataset": "Fit3D", "gate_id": "fail_gate", "status": "fail"},
                    {
                        "dataset": "QEVD",
                        "gate_id": "pending_gate",
                        "status": "not_evaluable",
                    },
                ],
                failure_rows=[
                    {"failure_categories": "last_event_miss|count_error"}
                ],
                robustness_summary=[
                    {"transform_id": "flip_lr", "structure_match_rate": "1.0"}
                ],
            )
            gate_svg = (figures_dir / "holdout_gates.svg").read_text()
            self.assertIn("status only", gate_svg)
            self.assertIn("PASS", gate_svg)
            self.assertIn("FAIL", gate_svg)
            self.assertIn("NOT EVALUABLE", gate_svg)
            self.assertIn('fill="#d1495b"', gate_svg)
            self.assertTrue((figures_dir / "failure_categories.svg").is_file())
            self.assertTrue((figures_dir / "robustness_structure_match.svg").is_file())

            empty_dir = figures_dir / "empty"
            audit.build_figures(
                empty_dir,
                go_no_go=[],
                failure_rows=[],
                robustness_summary=[],
            )
            for name in (
                "holdout_gates.svg",
                "failure_categories.svg",
                "robustness_structure_match.svg",
            ):
                self.assertIn("<svg", (empty_dir / name).read_text())


if __name__ == "__main__":
    unittest.main()
