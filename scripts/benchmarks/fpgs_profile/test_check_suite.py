# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check timing scope and incompatible cached-baseline rejection without a simulator."""

import unittest
from types import SimpleNamespace

from check_suite import PRESETS, GraphTimer, comparison, physics_launches_per_step


class TestSuiteScale(unittest.TestCase):
    def test_every_check_retains_the_required_world_count(self):
        for name, protocol in PRESETS.items():
            with self.subTest(mode=name):
                self.assertEqual(protocol["worlds"], 16384)


class TestQuickGraphTiming(unittest.TestCase):
    def fixture(self):
        physics, auxiliary = object(), object()
        manager = SimpleNamespace(_graph=physics)
        records = []
        default = SimpleNamespace(record_event=lambda event: records.append(("default", event)))
        explicit = SimpleNamespace(record_event=lambda event: records.append(("explicit", event)))

        def launch(graph, stream=None):
            records.append(("launch", graph, stream))
            return "result"

        warp = SimpleNamespace(
            Event=lambda *args, **kwargs: object(),
            capture_launch=launch,
            get_stream=lambda device: default,
            synchronize_device=lambda device: None,
            get_event_elapsed_time=lambda *args, **kwargs: 0.5,
        )
        return warp, manager, physics, auxiliary, explicit, records

    def test_physics_only_and_explicit_stream(self):
        warp, manager, physics, auxiliary, explicit, records = self.fixture()
        original = warp.capture_launch
        with GraphTimer(warp, manager, "cuda:0", 2) as timer:
            self.assertEqual(warp.capture_launch(auxiliary), "result")
            warp.capture_launch(physics, stream=explicit)
            manager._graph = auxiliary
            warp.capture_launch(auxiliary)
        self.assertIs(warp.capture_launch, original)
        self.assertEqual(timer.milliseconds(1, 2), [1.0])
        self.assertEqual(
            [row[0] for row in records], ["launch", "explicit", "launch", "explicit", "default", "launch", "default"]
        )

    def test_missing_launch_fails_and_wrapper_is_restored(self):
        warp, manager, physics, _, _, _ = self.fixture()
        original = warp.capture_launch
        with self.assertRaisesRegex(RuntimeError, "expected"):
            with GraphTimer(warp, manager, "cuda:0", 2) as timer:
                warp.capture_launch(physics)
                raise RuntimeError("expected")
        self.assertIs(warp.capture_launch, original)
        with self.assertRaisesRegex(RuntimeError, "Missing physics"):
            timer.milliseconds(1, 2)

    def test_graph_that_contains_the_full_decimation_loop(self):
        warp, manager, physics, _, _, _ = self.fixture()
        env = SimpleNamespace(
            unwrapped=SimpleNamespace(cfg=SimpleNamespace(decimation=4), _physics_handles_decimation=True)
        )
        with GraphTimer(warp, manager, "cuda:0", 1) as timer:
            warp.capture_launch(physics)
        self.assertEqual(timer.milliseconds(1, physics_launches_per_step(env)), [0.5])
        env.unwrapped._physics_handles_decimation = False
        self.assertEqual(physics_launches_per_step(env), 4)


class TestCachedComparison(unittest.TestCase):
    def test_incompatible_protocol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "different hardware"):
            comparison({"comparison_key": {"worlds": 256}}, {"comparison_key": {"worlds": 16384}})

    def test_changes_and_unstable_blocks_are_flagged(self):
        def summary(mean, blocks, state="same"):
            return {
                "comparison_key": {"worlds": 16384},
                "results": [
                    {
                        "scene": "fixture",
                        "complete": True,
                        "model": {},
                        "checks": [
                            {
                                "state_sha256": state,
                                "sim_dt": 0.01,
                                "decimation": 2,
                                "solver_dt": 0.005,
                                "substeps": 2,
                                "pgs_iterations": 8,
                            }
                        ],
                        "timing": {"mean_ms": mean, "blocks_ms": blocks},
                    }
                ],
            }

        baseline = summary(10, [10, 10, 10])
        self.assertEqual(comparison(baseline, summary(10.5, [10.5] * 3))[0]["status"], "investigate")
        self.assertEqual(comparison(baseline, summary(10, [9, 10, 11]))[0]["status"], "investigate")
        self.assertEqual(
            comparison(baseline, summary(10, [10] * 3, state="different"))[0]["status"],
            "investigate: initial state or configuration changed",
        )
        self.assertEqual(comparison(baseline, summary(10.01, [10.01] * 3))[0]["status"], "within screen tolerance")


if __name__ == "__main__":
    unittest.main()
