from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import segment_reps as sr


class SyntheticSegmentationTests(unittest.TestCase):
    fps = 30.0

    @staticmethod
    def wide_for(values: np.ndarray, action: str = "squat") -> dict[str, np.ndarray]:
        return {
            angle: values.copy()
            for angle in sr.ACTION_CANDIDATE_ANGLES[action]
        }

    @staticmethod
    def segment(action: str, count: int, fps: float) -> sr.ActionSegment:
        return sr.ActionSegment(
            video_id="9999",
            action=action,
            action_display=action,
            start_frame=0,
            end_frame=count - 1,
            start_time_s=0.0,
            end_time_s=count / fps,
        )

    def test_periodic_signal_estimates_expected_period(self) -> None:
        count = 600
        frames = np.arange(count)
        values = 140.0 + 30.0 * np.sin(2.0 * np.pi * frames / 30.0)

        prepared = sr.prepare_signal(values)
        self.assertIsNotNone(prepared)
        period = sr.estimate_period(prepared.normalized, self.fps, 0.4, 5.0)

        self.assertIsNotNone(period)
        self.assertGreaterEqual(period[0], 29)
        self.assertLessEqual(period[0], 31)

    def test_rest_platform_splits_active_bouts(self) -> None:
        count = 600
        frames = np.arange(count)
        values = 140.0 + 30.0 * np.sin(2.0 * np.pi * frames / 30.0)
        values[180:300] = 140.0
        segment = self.segment("squat", count, self.fps)

        prepared = sr.prepare_signal(values)
        self.assertIsNotNone(prepared)
        trace = sr.build_activity_trace(
            self.wide_for(values),
            segment,
            "left_knee_flexion",
            self.fps,
            window_s=0.5,
            threshold=0.10,
            close_gap_s=0.2,
        )
        self.assertIsNotNone(trace)
        low_intervals = sr.significant_low_intervals(
            trace,
            period_frames=30.0,
            fps=self.fps,
            min_nonrep_s=0.5,
        )
        windows = sr.active_bout_windows(prepared, low_intervals, count)

        self.assertTrue(
            any(min(end, 280) >= max(start, 200) for start, end in low_intervals)
        )
        self.assertEqual(len(windows), 2)
        self.assertLess(windows[0]["end"], windows[1]["start"])

    def test_active_bout_reps_overlap_an_active_core(self) -> None:
        count = 600
        frames = np.arange(count)
        values = 140.0 + 30.0 * np.sin(2.0 * np.pi * frames / 30.0)
        values[180:300] = 140.0
        wide = self.wide_for(values)
        segment = self.segment("squat", count, self.fps)

        prepared = sr.prepare_signal(values)
        self.assertIsNotNone(prepared)
        period = sr.estimate_period(prepared.normalized, self.fps, 0.4, 5.0)
        trace = sr.build_activity_trace(
            wide,
            segment,
            "left_knee_flexion",
            self.fps,
            window_s=0.5,
            threshold=0.10,
            close_gap_s=0.2,
        )
        self.assertIsNotNone(trace)
        low_intervals = sr.significant_low_intervals(
            trace,
            period_frames=float(period[0]) if period else 0.0,
            fps=self.fps,
            min_nonrep_s=0.5,
        )
        result, bouts, _ = sr.run_aifit_with_active_bouts(
            values,
            prepared,
            trace,
            low_intervals,
            segment,
            "left_knee_flexion",
            self.fps,
            0.4,
            5.0,
            True,
        )

        self.assertIsNotNone(result)
        active = [bout for bout in bouts if bout.status == "active"]
        self.assertTrue(active)
        for rep in result.reps:
            overlap = max(
                min(rep.end_frame, bout.end_frame) - max(rep.start_frame, bout.start_frame) + 1
                for bout in active
            )
            self.assertGreater(overlap, 0)

    def test_filter_drops_reps_entirely_outside_active_core(self) -> None:
        result = sr.MethodResult(
            method="aifit",
            angle="synthetic",
            reps=[
                sr.Rep(1, 60, 80, 70, 0.9),
                sr.Rep(2, 100, 130, 115, 0.8),
                sr.Rep(3, 145, 170, 158, 0.7),
            ],
            period_frames=30.0,
            similarity=0.8,
            valid_fraction=1.0,
            objective=0.0,
        )

        sr.filter_reps_to_active_core(result, core_start_frame=90, core_end_frame=160)

        self.assertEqual(
            [(rep.start_frame, rep.end_frame) for rep in result.reps],
            [(100, 130), (145, 170)],
        )
        self.assertEqual([rep.rep_id for rep in result.reps], [1, 2])
        self.assertEqual(len(result.rejected_reps), 1)
        self.assertEqual(result.rejected_reps[0].reason, "outside_active_bout")
        self.assertEqual(result.rejected_reps[0].evidence["overlap_frames"], 0)

    def test_robust_template_downweights_shifted_outlier(self) -> None:
        points = np.linspace(0.0, 2.0 * np.pi, 48)
        base = np.sin(points) + 0.3 * np.sin(2.0 * points)
        curves = np.vstack(
            [
                base,
                base + 0.01,
                base - 0.01,
                base + 0.02,
                base - 0.02,
                np.roll(base, 10) * 0.7,
            ]
        )

        mean_template = sr.robust_template(curves, "mean")
        irls_template = sr.robust_template(curves, "irls")

        self.assertGreater(
            sr.correlation(irls_template, base),
            sr.correlation(mean_template, base),
        )

    def test_confidence_breakdown_marks_motion_outlier_without_deleting(self) -> None:
        count = 180
        frames = np.arange(count, dtype=np.float64)
        signal = np.sin(2.0 * np.pi * frames / 60.0)
        reps = [
            sr.Rep(1, 0, 59, None, 0.9),
            sr.Rep(2, 60, 119, None, 0.9),
            sr.Rep(3, 120, 179, None, 0.9),
        ]
        activity = np.ones(count, dtype=np.float64)
        activity[120:] = 3.0
        trace = sr.ActivityTrace(
            activity=activity,
            visibility=np.ones(count, dtype=np.float64),
            main_speed=activity.copy(),
            low_intervals=[],
            evidence={"threshold": 0.10},
        )
        phase_evidence = {
            "extrema_frames": [0, 60, 120],
            "tolerance_frames": 6,
        }

        enriched = sr.enrich_rep_confidences(
            reps,
            signal=signal,
            valid=np.ones(count, dtype=bool),
            offset=0,
            phase_evidence=phase_evidence,
            period_frames=60.0,
            phase_signal_available=True,
            trace=trace,
            segment_start_frame=0,
        )

        self.assertEqual(len(enriched), len(reps))
        self.assertEqual(
            [(rep.start_frame, rep.end_frame) for rep in enriched],
            [(rep.start_frame, rep.end_frame) for rep in reps],
        )
        self.assertTrue(all(0.0 <= rep.visibility <= 1.0 for rep in enriched))
        self.assertTrue(all(0.0 <= rep.shape_similarity <= 1.0 for rep in enriched))
        self.assertTrue(all(0.0 <= rep.phase_consistency <= 1.0 for rep in enriched))
        self.assertLess(enriched[-1].motion_evidence, enriched[0].motion_evidence)
        self.assertLess(enriched[-1].review_score, enriched[0].review_score)
        self.assertIn("low_motion_evidence", enriched[-1].reject_reason)
        self.assertGreater(enriched[-1].confidence, 0.0)

    def test_renumber_reps_preserves_confidence_fields(self) -> None:
        rep = sr.Rep(
            9,
            10,
            20,
            None,
            0.8,
            visibility=0.9,
            shape_similarity=0.7,
            phase_consistency=0.6,
            motion_evidence=0.5,
            review_score=0.65,
            review_priority="medium",
            reject_reason="low_shape_similarity",
        )

        renumbered = sr.renumber_reps([rep])

        self.assertEqual(renumbered[0].rep_id, 1)
        self.assertEqual(renumbered[0].review_score, 0.65)
        self.assertEqual(renumbered[0].reject_reason, "low_shape_similarity")

    def test_variable_period_stays_in_plausible_range(self) -> None:
        count = 720
        frames = np.arange(count, dtype=np.float64)
        periods = 30.0 + 6.0 * np.sin(2.0 * np.pi * frames / 240.0)
        values = 120.0 + 25.0 * np.sin(np.cumsum(2.0 * np.pi / periods))

        prepared = sr.prepare_signal(values)
        self.assertIsNotNone(prepared)
        period = sr.estimate_period(prepared.normalized, self.fps, 0.4, 5.0)

        self.assertIsNotNone(period)
        self.assertGreaterEqual(period[0], 24)
        self.assertLessEqual(period[0], 36)

    def test_low_visibility_handling(self) -> None:
        count = 600
        frames = np.arange(count)
        base = 120.0 + 25.0 * np.sin(2.0 * np.pi * frames / 30.0)

        sparse = base.copy()
        sparse[::4] = np.nan
        prepared = sr.prepare_signal(sparse)
        self.assertIsNotNone(prepared)
        self.assertGreater(prepared.valid.mean(), 0.70)

        mostly_missing = base.copy()
        mostly_missing[::2] = np.nan
        mostly_missing[: count // 2] = np.nan
        self.assertIsNone(sr.prepare_signal(mostly_missing))

    def test_half_cycle_tail_is_rejected(self) -> None:
        result = sr.MethodResult(
            method="aifit",
            angle="synthetic",
            reps=[
                sr.Rep(1, 0, 59, 30, 0.9),
                sr.Rep(2, 60, 119, 90, 0.9),
                sr.Rep(3, 120, 179, 150, 0.9),
                sr.Rep(4, 180, 199, 190, 0.9),
            ],
            period_frames=60.0,
            similarity=0.9,
            valid_fraction=1.0,
            objective=0.0,
        )

        sr.reject_incomplete_tail(result, fit_end_frame=200, fps=self.fps)

        self.assertEqual(len(result.reps), 3)
        self.assertEqual(result.rejected_reps[-1].reason, "incomplete_rep")
        self.assertEqual([rep.rep_id for rep in result.reps], [1, 2, 3])

    def test_phase_guidance_selects_landmark_phase(self) -> None:
        count = 610
        frames = np.arange(count, dtype=np.float64)
        main = np.sin(2.0 * np.pi * frames / 30.0)
        phase = np.cos(2.0 * np.pi * (frames - 15.0) / 30.0)
        guidance = sr.PhaseGuidance(
            phase_landmark="synthetic_peak",
            annotation_polarity="peak",
            signal=sr.PhaseSignalSpec(source="angle", name="synthetic"),
            polarity="peak",
        )

        baseline = sr.stage_a(main, 30)
        guided = sr.stage_a(main, 30, phase_signal=phase, phase_guidance=guidance)

        self.assertIsNotNone(baseline)
        self.assertIsNotNone(guided)
        assert baseline is not None
        assert guided is not None
        self.assertEqual(guided[3]["status"], "applied")
        self.assertLessEqual(abs(guided[1] % 30 - 15), 1)
        self.assertNotEqual(guided[1], baseline[1])

    def test_phase_guidance_falls_back_without_enough_extrema(self) -> None:
        count = 600
        frames = np.arange(count, dtype=np.float64)
        main = np.sin(2.0 * np.pi * frames / 30.0)
        phase = np.full(count, 10.0)
        phase[100] = 20.0
        phase[400] = 20.0
        guidance = sr.PhaseGuidance(
            phase_landmark="synthetic_sparse",
            annotation_polarity="peak",
            signal=sr.PhaseSignalSpec(source="angle", name="synthetic"),
            polarity="peak",
        )

        baseline = sr.stage_a(main, 30)
        guided = sr.stage_a(main, 30, phase_signal=phase, phase_guidance=guidance)

        self.assertIsNotNone(baseline)
        self.assertIsNotNone(guided)
        assert baseline is not None
        assert guided is not None
        self.assertEqual(guided[3]["status"], "fallback")
        self.assertEqual(guided[3]["reason"], "too_few_extrema")
        self.assertEqual(guided[1], baseline[1])

    def test_phase_guidance_rejects_boundary_count_change(self) -> None:
        count = 600
        frames = np.arange(count, dtype=np.float64)
        main = np.sin(2.0 * np.pi * frames / 30.0)
        phase = np.cos(2.0 * np.pi * (frames - 25.0) / 30.0)
        guidance = sr.PhaseGuidance(
            phase_landmark="synthetic_tail_phase",
            annotation_polarity="peak",
            signal=sr.PhaseSignalSpec(source="angle", name="synthetic"),
            polarity="peak",
        )

        baseline = sr.stage_a(main, 30)
        guided = sr.stage_a(main, 30, phase_signal=phase, phase_guidance=guidance)

        self.assertIsNotNone(baseline)
        self.assertIsNotNone(guided)
        assert baseline is not None
        assert guided is not None
        self.assertEqual(guided[3]["status"], "fallback")
        self.assertEqual(guided[3]["reason"], "phase_changes_boundary_count")
        self.assertEqual(len(guided[0]), len(baseline[0]))
        self.assertEqual(guided[1], baseline[1])

    def test_plateau_snap_moves_residual_boundaries(self) -> None:
        period = 40
        count = 240
        phase = np.full(count, -2.0)
        for center in (40, 80, 120, 160, 200):
            phase[center - 5 : center + 6] = 2.0
        guidance = sr.PhaseGuidance(
            phase_landmark="synthetic_plateau",
            annotation_polarity="peak",
            signal=sr.PhaseSignalSpec(source="angle", name="synthetic"),
            polarity="peak",
            boundary_snap=sr.BoundarySnapGuidance(mode="plateau_entry"),
        )
        boundaries = np.asarray([37, 77, 117, 157, 197, 220])

        snapped, evidence = sr.snap_boundaries_to_phase(
            boundaries,
            period,
            phase,
            guidance,
        )

        self.assertEqual(evidence["status"], "applied")
        self.assertGreater(evidence["changed_boundaries"], 0)
        self.assertTrue(np.all(np.diff(snapped) > 0))
        for index in range(4):
            self.assertLessEqual(abs(int(snapped[index]) - (40 + 40 * index)), 4)

    def test_plateau_snap_falls_back_without_phase_signal(self) -> None:
        guidance = sr.PhaseGuidance(
            phase_landmark="synthetic_flat_plateau",
            annotation_polarity="peak",
            signal=sr.PhaseSignalSpec(source="angle", name="synthetic"),
            polarity="peak",
            boundary_snap=sr.BoundarySnapGuidance(mode="plateau_entry"),
        )
        boundaries = np.asarray([10, 50, 90, 130])

        snapped, evidence = sr.snap_boundaries_to_phase(
            boundaries,
            40,
            np.full(180, 5.0),
            guidance,
        )

        np.testing.assert_array_equal(snapped, boundaries)
        self.assertEqual(evidence["status"], "fallback")
        self.assertEqual(evidence["reason"], "phase_signal_unavailable")

    def test_plateau_snap_rejects_invalid_width(self) -> None:
        count = 160
        phase = np.full(count, -2.0)
        for center in (15, 35, 55, 75):
            phase[center - 4 : center + 5] = 2.0
        guidance = sr.PhaseGuidance(
            phase_landmark="synthetic_short_plateau",
            annotation_polarity="peak",
            signal=sr.PhaseSignalSpec(source="angle", name="synthetic"),
            polarity="peak",
            boundary_snap=sr.BoundarySnapGuidance(
                mode="plateau_entry",
                min_width_fraction=0.50,
            ),
        )
        boundaries = np.asarray([10, 30, 50, 70, 90])

        snapped, evidence = sr.snap_boundaries_to_phase(
            boundaries,
            60,
            phase,
            guidance,
        )

        np.testing.assert_array_equal(snapped, boundaries)
        self.assertEqual(evidence["status"], "fallback")
        self.assertEqual(evidence["reason"], "invalid_width")

    def test_plateau_snap_preserves_rep_count(self) -> None:
        count = 600
        frames = np.arange(count, dtype=np.float64)
        signal = np.sin(2.0 * np.pi * frames / 30.0)
        phase = np.full(count, -2.0)
        for center in range(30, count, 30):
            phase[max(0, center - 5) : center + 6] = 2.0
        guidance = sr.PhaseGuidance(
            phase_landmark="synthetic_plateau_count",
            annotation_polarity="peak",
            signal=sr.PhaseSignalSpec(source="angle", name="synthetic"),
            polarity="peak",
            min_extrema=100,
            boundary_snap=sr.BoundarySnapGuidance(mode="plateau_entry"),
        )
        valid = np.ones(count, dtype=bool)

        baseline = sr.run_aifit(
            signal,
            valid,
            "synthetic",
            0,
            self.fps,
            0.4,
            5.0,
            True,
            phase_signal=phase,
            phase_guidance=guidance,
            use_phase_snap=False,
        )
        snapped = sr.run_aifit(
            signal,
            valid,
            "synthetic",
            0,
            self.fps,
            0.4,
            5.0,
            True,
            phase_signal=phase,
            phase_guidance=guidance,
            use_phase_snap=True,
        )

        self.assertIsNotNone(baseline)
        self.assertIsNotNone(snapped)
        assert baseline is not None
        assert snapped is not None
        self.assertEqual(len(snapped.reps), len(baseline.reps))
        self.assertIn(
            snapped.phase_guidance["boundary_snap"]["status"],
            {"applied", "fallback"},
        )

    def test_evidence_angle_scoring_prefers_periodic_candidate(self) -> None:
        count = 900
        frames = np.arange(count, dtype=np.float64)
        rng = np.random.default_rng(20261025)
        wide = {}
        for angle in sr.ACTION_CANDIDATE_ANGLES["squat"]:
            if angle == "left_knee_flexion":
                wide[angle] = 140.0 + 25.0 * np.sin(2.0 * np.pi * frames / 60.0)
            else:
                wide[angle] = 140.0 + rng.normal(0.0, 35.0, count)
        segment = self.segment("squat", count, self.fps)

        legacy = sr.choose_auto_angle(
            wide,
            segment,
            self.fps,
            0.4,
            5.0,
            scoring="legacy",
        )
        evidence = sr.choose_auto_angle(
            wide,
            segment,
            self.fps,
            0.4,
            5.0,
            scoring="evidence",
        )

        self.assertIsNotNone(legacy)
        self.assertIsNotNone(evidence)
        assert legacy is not None
        assert evidence is not None
        self.assertNotEqual(legacy.angle, "left_knee_flexion")
        self.assertEqual(evidence.angle, "left_knee_flexion")
        self.assertEqual(evidence.evidence["selected_by"], "cluster")
        self.assertTrue(evidence.evidence["candidates"])
        selected = next(
            item for item in evidence.evidence["candidates"] if item["selected"]
        )
        self.assertGreater(selected["periodicity"], 0.5)

    def test_evidence_angle_scoring_marks_ambiguous_selection(self) -> None:
        count = 600
        frames = np.arange(count, dtype=np.float64)
        values = 140.0 + 25.0 * np.sin(2.0 * np.pi * frames / 60.0)
        wide = self.wide_for(values, "squat")
        segment = self.segment("squat", count, self.fps)

        evidence = sr.choose_auto_angle(
            wide,
            segment,
            self.fps,
            0.4,
            5.0,
            scoring="evidence",
        )

        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertTrue(evidence.evidence["angle_ambiguous"])
        self.assertEqual(evidence.evidence["selected_by"], "legacy_tie_break")

    def test_air_jump_tail_filter_requires_two_independent_evidence(self) -> None:
        reps = [
            sr.Rep(1, 0, 39, 20, 0.9, shape_similarity=0.9, motion_evidence=0.9),
            sr.Rep(2, 40, 79, 60, 0.9, shape_similarity=0.9, motion_evidence=0.9),
            sr.Rep(3, 80, 119, 100, 0.9, shape_similarity=0.9, motion_evidence=0.9),
            sr.Rep(4, 120, 159, 140, 0.4, shape_similarity=0.2, motion_evidence=0.001),
        ]
        result = sr.MethodResult(
            method="aifit",
            angle="synthetic",
            reps=reps,
            period_frames=40.0,
            similarity=0.8,
            valid_fraction=1.0,
            objective=0.0,
            phase_guidance={"status": "fallback", "extrema_frames": [20, 60, 100]},
        )
        segment = self.segment("air_jump_rope", 200, self.fps)

        removed = sr.reject_air_jump_rope_transition_tail(
            result,
            segment,
            fit_end_frame=160,
        )

        self.assertTrue(removed)
        self.assertEqual(len(result.reps), 3)
        self.assertEqual([rep.rep_id for rep in result.reps], [1, 2, 3])
        self.assertEqual(
            result.rejected_reps[-1].reason,
            "air_jump_rope_transition_tail",
        )
        self.assertEqual(result.rejected_reps[-1].evidence["support_count"], 3)

        result.reps.append(
            sr.Rep(4, 120, 159, 140, 0.9, shape_similarity=0.9, motion_evidence=0.9)
        )
        result.reps = sr.renumber_reps(result.reps)
        rejected = sr.reject_air_jump_rope_transition_tail(
            result,
            segment,
            fit_end_frame=160,
        )
        self.assertFalse(rejected)
        self.assertEqual(len(result.reps), 4)


    def test_rep_metadata_covers_all_action_segments(self) -> None:
        metadata = sr.load_rep_metadata(ROOT / "configs" / "rep_metadata.json")
        with (ROOT / "results" / "segmentation" / "action_segments.csv").open(newline="") as handle:
            segment_keys = {
                (row["video_id"].zfill(4), row["action"])
                for row in csv.DictReader(handle)
            }

        self.assertEqual(len(segment_keys), 22)
        self.assertEqual(set(metadata), segment_keys)
        self.assertTrue(all(entry["rep_unit"] in sr.VALID_REP_UNITS for entry in metadata.values()))
        self.assertTrue(all(entry["phase_landmark"] for entry in metadata.values()))
        self.assertTrue(
            all(entry["phase_polarity"] in sr.VALID_PHASE_POLARITIES for entry in metadata.values())
        )

    def test_walking_lunge_metadata_uses_stride_for_both_segments(self) -> None:
        metadata = sr.load_rep_metadata(ROOT / "configs" / "rep_metadata.json")

        for video_id in ("0162", "0218"):
            entry = metadata[(video_id, "walking_lunges")]
            self.assertEqual(entry["rep_unit"], "stride")
            self.assertEqual(entry["phase_polarity"], "peak")
        self.assertEqual(
            metadata[("0162", "walking_lunges")]["phase_landmark"],
            "left_leg_forward_lowest",
        )
        self.assertEqual(
            metadata[("0218", "walking_lunges")]["phase_landmark"],
            "right_leg_forward_lowest",
        )


if __name__ == "__main__":
    unittest.main()
