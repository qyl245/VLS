from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_6_phase_states as phase_states


def synthetic_wave(
    fps: float,
    duration_s: float,
    period_s: float,
    *,
    amplitude: float = 40.0,
    offset: float = 120.0,
    noise: float = 0.0,
    missing_fraction: float = 0.0,
    seed: int = 20261026,
) -> np.ndarray:
    count = int(round(fps * duration_s))
    times = np.arange(count, dtype=np.float64) / fps
    wave = np.cos(2.0 * np.pi * times / period_s)
    values = np.clip(wave, -0.55, 0.60) * amplitude + offset
    rng = np.random.default_rng(seed)
    if noise:
        values = values + rng.normal(0.0, noise, count)
    if missing_fraction:
        missing = rng.choice(
            count,
            size=int(round(count * missing_fraction)),
            replace=False,
        )
        values[missing] = np.nan
    return values


class PhaseStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.action = {
            "rep_unit": "cycle",
            "top_polarity": "peak",
            "bottom_polarity": "trough",
            "min_period_s": 0.70,
            "max_period_s": 5.00,
            "phase_signal_candidates": [],
            "phase_signal_pairs": [],
        }

    def detect(
        self,
        values: np.ndarray,
        fps: float = 50.0,
        action: dict | None = None,
    ) -> dict:
        return phase_states.detect_phase_states(
            values,
            fps,
            action or self.action,
            "synthetic_flexion",
            "plateau_midpoint",
        )

    def test_anchor_order_and_state_containment(self) -> None:
        result = self.detect(synthetic_wave(50.0, 8.0, 1.6))
        self.assertEqual(result["status"], "ok")
        for state in result["top_states"] + result["bottom_states"]:
            anchors = state["anchors"]
            self.assertLessEqual(
                state["start_frame"], anchors["plateau_entry"]
            )
            self.assertLessEqual(
                anchors["plateau_entry"], anchors["plateau_midpoint"]
            )
            self.assertLessEqual(
                anchors["plateau_midpoint"], anchors["plateau_exit"]
            )
            self.assertLessEqual(anchors["plateau_exit"], state["end_frame"])
            for anchor in anchors.values():
                self.assertGreaterEqual(anchor, state["start_frame"])
                self.assertLessEqual(anchor, state["end_frame"])

    def test_plateau_midpoint_tracks_synthetic_top_centers(self) -> None:
        fps = 50.0
        period_s = 1.6
        result = self.detect(synthetic_wave(fps, 8.0, period_s))
        interior = [
            state
            for state in result["top_states"]
            if state["kind"] == "interior"
        ]
        self.assertEqual(len(interior), 4)
        expected = [
            period_s * fps,
            2.0 * period_s * fps,
            3.0 * period_s * fps,
            4.0 * period_s * fps,
        ]
        observed = [state["anchors"]["plateau_midpoint"] for state in interior]
        for actual, target in zip(observed, expected):
            self.assertLessEqual(abs(actual - target), 2.0)

    def test_endpoint_partial_only_at_leading_and_trailing_cycles(self) -> None:
        result = self.detect(synthetic_wave(50.0, 8.0, 1.6))
        statuses = [rep["status"] for rep in result["reps"]]
        self.assertEqual(statuses[0], "endpoint_partial")
        self.assertEqual(statuses[-1], "endpoint_partial")
        self.assertTrue(all(status == "complete" for status in statuses[1:-1]))
        self.assertEqual(result["reps"][0]["partial_side"], "start")
        self.assertEqual(result["reps"][-1]["partial_side"], "end")

    def test_amplitude_and_offset_do_not_change_state_count(self) -> None:
        baseline = self.detect(synthetic_wave(50.0, 8.0, 1.6))
        scaled = self.detect(
            synthetic_wave(50.0, 8.0, 1.6, amplitude=11.0, offset=-35.0)
        )
        self.assertEqual(
            len(baseline["top_states"]), len(scaled["top_states"])
        )
        self.assertEqual(
            len(baseline["bottom_states"]), len(scaled["bottom_states"])
        )
        self.assertEqual(len(baseline["reps"]), len(scaled["reps"]))

    def test_signal_polarity_flip_preserves_states(self) -> None:
        values = synthetic_wave(50.0, 8.0, 1.6)
        flipped_action = {**self.action, "top_polarity": "trough"}
        baseline = self.detect(values)
        flipped = self.detect(-values, action=flipped_action)
        self.assertEqual(
            [state["anchors"]["plateau_midpoint"] for state in baseline["top_states"]],
            [state["anchors"]["plateau_midpoint"] for state in flipped["top_states"]],
        )

    def test_frame_rate_resampling_preserves_cycle_count(self) -> None:
        counts = []
        for fps in (15.0, 30.0, 50.0, 60.0):
            result = self.detect(synthetic_wave(fps, 8.0, 1.6), fps=fps)
            self.assertEqual(result["status"], "ok")
            counts.append(len(result["reps"]))
        self.assertEqual(counts, [5, 5, 5, 5])

    def test_noise_jitter_and_missing_frames_preserve_cycle_count(self) -> None:
        values = synthetic_wave(
            50.0,
            8.0,
            1.6,
            noise=1.5,
            missing_fraction=0.12,
            seed=7,
        )
        result = self.detect(values)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["top_states"]), 6)
        self.assertEqual(len(result["reps"]), 5)
        self.assertGreaterEqual(result["valid_fraction"], 0.85)

    def test_short_sequence_and_half_rep_do_not_emit_complete_rep(self) -> None:
        short = self.detect(synthetic_wave(50.0, 0.30, 1.6))
        self.assertEqual(short["status"], "no_motion")
        self.assertFalse(
            any(rep["status"] == "complete" for rep in short["reps"])
        )

        one_cycle = self.detect(synthetic_wave(50.0, 1.6, 1.6))
        self.assertFalse(
            any(rep["status"] == "complete" for rep in one_cycle["reps"])
        )

    def test_missing_bottom_cannot_be_filled_as_complete_rep(self) -> None:
        top = {
            "start_frame": 20,
            "end_frame": 40,
            "kind": "interior",
            "status": "valid",
            "stable_fraction": 1.0,
            "mean_angle": 170.0,
            "quality_flags": [],
        }
        second_top = {
            **top,
            "start_frame": 100,
            "end_frame": 120,
        }
        self.assertEqual(
            phase_states.build_rep_candidates(
                [top, second_top],
                [],
                phase_signal="x",
                rep_unit="cycle",
                period_frames=80.0,
                periodicity=1.0,
            ),
            [],
        )
        self.assertEqual(
            phase_states.build_rep_candidates(
                [top],
                [{**top, "mean_angle": 70.0, "kind": "interior"}],
                phase_signal="x",
                rep_unit="cycle",
                period_frames=80.0,
                periodicity=1.0,
            ),
            [],
        )

    def test_bilateral_signal_uses_mean_and_survives_one_missing_side(self) -> None:
        action = phase_states.action_config(
            "pushup",
            json.loads(
                (ROOT / "configs" / "task1_4_exercise_specs.json").read_text()
            ),
        )
        left = synthetic_wave(50.0, 4.0, 1.6)
        right = left + 4.0
        signals = {"left_elbow_flexion": left, "right_elbow_flexion": right}
        name = phase_states.select_bilateral_signal(signals, 50.0, action)
        self.assertEqual(name, "elbow_flexion_bilateral_mean")
        self.assertTrue(np.allclose(signals[name], left + 2.0))

        missing_right = left.copy()
        missing_right[:] = np.nan
        one_side = {
            "left_elbow_flexion": left,
            "right_elbow_flexion": missing_right,
        }
        name = phase_states.select_bilateral_signal(one_side, 50.0, action)
        self.assertEqual(name, "elbow_flexion_bilateral_mean")
        self.assertTrue(np.allclose(one_side[name], left))

    def test_configured_signal_falls_back_to_available_action_candidate(self) -> None:
        specs = json.loads(
            (ROOT / "configs" / "task1_4_exercise_specs.json").read_text()
        )
        action = phase_states.action_config("pushup", specs)
        signals = {
            "left_elbow_flexion": synthetic_wave(50.0, 4.0, 1.6),
        }
        selected = phase_states.resolve_phase_signal(
            signals,
            50.0,
            action,
            "t1_6_bilateral_mean",
        )
        self.assertEqual(selected, "left_elbow_flexion")

    def test_event_matching_is_one_to_one_and_gated(self) -> None:
        self.assertEqual(phase_states.event_gate_frames([100, 200]), 25)
        matches = phase_states.match_events(
            [100, 200],
            [110, 190, 330],
            25,
        )
        self.assertEqual(matches, [(0, 0, 10), (1, 1, 10)])
        self.assertEqual(
            phase_states.match_events([100], [130], 25),
            [],
        )

    def test_event_matching_maximizes_in_gate_matches(self) -> None:
        matches = phase_states.match_events(
            [283, 390, 489, 594, 692, 802],
            [396, 512, 592, 719, 811, 935],
            26,
        )
        self.assertEqual(
            matches,
            [(1, 0, 6), (2, 1, 23), (3, 2, 2), (5, 4, 9)],
        )

    def test_frozen_config_core_has_no_source_specific_tokens(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "task1_6_segmenter.json").read_text()
        )
        core_text = json.dumps(config["core"], ensure_ascii=False)
        for token in (
            "Fit3D",
            "QEVD",
            "s03",
            "s11",
            "50591643",
            "58860488",
            "60457274",
            "65906101",
            "fixed_rep_count",
            "fixed_frame_offset",
            "camera_id",
        ):
            self.assertNotIn(token, core_text)
        self.assertEqual(
            config["core"]["anchor"]["policy"], "plateau_midpoint"
        )
        self.assertEqual(
            config["core"]["phase_signal"]["policy"],
            "t1_6_bilateral_mean",
        )


if __name__ == "__main__":
    unittest.main()
