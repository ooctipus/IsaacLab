# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for octibenchmark.bench_cfg module."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from octibenchmark.bench_cfg import BenchmarkMatrix, BenchmarkRun, Launcher, Phase


class TestBenchmarkRun(unittest.TestCase):
    """Tests for BenchmarkRun dataclass."""

    def _make_run(self, **kwargs):
        defaults = dict(
            task="Isaac-Repose-Cube-Shadow-Vision-Direct-v0",
            num_envs=2048,
            hydra_overrides={
                "preset": "presets=newton,newton_renderer,rgb",
                "resolution": "env.tiled_camera.width=128 env.tiled_camera.height=128",
            },
            launcher=Launcher.NON_RL,
            phase=Phase.STEP,
            num_frames=100,
            warmup_frames=10,
            max_iterations=10,
        )
        defaults.update(kwargs)
        return BenchmarkRun(**defaults)

    def test_tag_contains_all_dimensions(self):
        run = self._make_run()
        tag = run.tag
        self.assertIn("envs2048", tag)
        self.assertIn("newton_newton_renderer_rgb", tag)
        self.assertIn("128_128", tag)
        self.assertIn("non_rl", tag)

    def test_tag_no_hydra_overrides(self):
        run = self._make_run(hydra_overrides={})
        tag = run.tag
        self.assertIn("envs2048", tag)
        self.assertIn("non_rl", tag)

    def test_all_hydra_args_with_overrides(self):
        run = self._make_run()
        args = run.all_hydra_args
        self.assertIn("presets=newton,newton_renderer,rgb", args)
        self.assertIn("env.tiled_camera.width=128", args)
        self.assertIn("env.tiled_camera.height=128", args)

    def test_all_hydra_args_empty_overrides(self):
        run = self._make_run(hydra_overrides={})
        args = run.all_hydra_args
        self.assertEqual(args, [])

    def test_all_hydra_args_includes_constant(self):
        run = self._make_run(hydra_args=["env.decimation=4"])
        args = run.all_hydra_args
        self.assertIn("env.decimation=4", args)
        # Sweep overrides come before constant args
        self.assertIn("presets=newton,newton_renderer,rgb", args)

    def test_script_path_non_rl(self):
        run = self._make_run(launcher=Launcher.NON_RL)
        self.assertTrue(str(run.script_path).endswith("benchmark.py"))

    def test_script_path_rsl_rl(self):
        run = self._make_run(launcher=Launcher.RSL_RL)
        self.assertTrue(str(run.script_path).endswith("benchmark_train.py"))

    def test_nsys_command_non_rl(self):
        run = self._make_run(launcher=Launcher.NON_RL)
        cmd = run.nsys_command("/tmp/out")
        self.assertEqual(cmd[0], "nsys")
        self.assertIn("--num_frames", cmd)
        self.assertIn("100", cmd)
        self.assertNotIn("--max_iterations", cmd)
        # Hydra overrides appear at the end
        cmd_str = " ".join(cmd)
        self.assertIn("presets=newton,newton_renderer,rgb", cmd_str)

    def test_nsys_command_rsl_rl(self):
        run = self._make_run(launcher=Launcher.RSL_RL, max_iterations=5)
        cmd = run.nsys_command("/tmp/out")
        self.assertIn("--max_iterations", cmd)
        self.assertIn("5", cmd)
        self.assertNotIn("--num_frames", cmd)

    def test_nsys_command_includes_phase(self):
        run = self._make_run(phase=Phase.STARTUP)
        cmd = run.nsys_command("/tmp/out")
        phase_idx = cmd.index("--phase")
        self.assertEqual(cmd[phase_idx + 1], "startup")

    def test_nsys_command_step_phase(self):
        run = self._make_run(phase=Phase.STEP)
        cmd = run.nsys_command("/tmp/out")
        phase_idx = cmd.index("--phase")
        self.assertEqual(cmd[phase_idx + 1], "step")

    def test_tag_step_phase_no_suffix(self):
        """Step phase is the default, so it doesn't appear in the tag."""
        run = self._make_run(phase=Phase.STEP)
        self.assertNotIn("step", run.tag)

    def test_tag_startup_phase_has_suffix(self):
        run = self._make_run(phase=Phase.STARTUP)
        self.assertIn("startup", run.tag)


class TestBenchmarkMatrix(unittest.TestCase):
    """Tests for BenchmarkMatrix cross-product generation."""

    def test_basic_matrix_no_sweeps(self):
        matrix = BenchmarkMatrix(
            tasks=["TaskA"],
            num_envs=[1024, 2048],
        )
        runs = matrix.runs()
        self.assertEqual(len(runs), 2)

    def test_single_sweep_axis(self):
        matrix = BenchmarkMatrix(
            tasks=["TaskA"],
            num_envs=[2048, 4096],
            hydra_sweeps={
                "preset": [
                    "presets=newton,newton_renderer,rgb",
                    "presets=newton,newton_renderer,depth",
                ],
            },
        )
        runs = matrix.runs()
        # 1 task × 2 envs × 2 presets = 4
        self.assertEqual(len(runs), 4)

    def test_multi_sweep_axes(self):
        matrix = BenchmarkMatrix(
            tasks=["TaskA"],
            num_envs=[1024, 2048],
            hydra_sweeps={
                "preset": ["presets=a", "presets=b"],
                "resolution": ["env.w=64 env.h=64", "env.w=128 env.h=128"],
            },
        )
        runs = matrix.runs()
        # 1 task × 2 envs × 2 presets × 2 resolutions = 8
        self.assertEqual(len(runs), 8)

    def test_multi_task(self):
        matrix = BenchmarkMatrix(
            tasks=["TaskA", "TaskB"],
            num_envs=[2048],
            hydra_sweeps={"preset": ["presets=x"]},
        )
        runs = matrix.runs()
        self.assertEqual(len(runs), 2)

    def test_rsl_rl_launcher(self):
        matrix = BenchmarkMatrix(
            tasks=["TaskA"],
            num_envs=[2048],
            hydra_sweeps={"preset": ["presets=x"]},
            launcher=Launcher.RSL_RL,
            max_iterations=5,
        )
        runs = matrix.runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].launcher, Launcher.RSL_RL)
        self.assertEqual(runs[0].max_iterations, 5)

    def test_hydra_args_propagated(self):
        matrix = BenchmarkMatrix(
            tasks=["TaskA"],
            num_envs=[2048],
            hydra_args=["env.foo=bar"],
        )
        runs = matrix.runs()
        self.assertIn("env.foo=bar", runs[0].all_hydra_args)

    def test_hydra_sweeps_in_run_overrides(self):
        matrix = BenchmarkMatrix(
            tasks=["TaskA"],
            num_envs=[2048],
            hydra_sweeps={
                "physics": ["presets=newton", "presets=physx"],
            },
        )
        runs = matrix.runs()
        self.assertEqual(len(runs), 2)
        overrides = {r.hydra_overrides["physics"] for r in runs}
        self.assertEqual(overrides, {"presets=newton", "presets=physx"})

    def test_dry_run_no_error(self):
        matrix = BenchmarkMatrix(
            tasks=["TaskA"],
            num_envs=[2048],
            hydra_sweeps={"preset": ["presets=x"]},
        )
        results = matrix.execute(dry_run=True, output_dir="/tmp/test_dry_cfg")
        self.assertEqual(results, {})

    def test_tag_uniqueness(self):
        matrix = BenchmarkMatrix(
            tasks=["TaskA"],
            num_envs=[2048, 4096],
            hydra_sweeps={
                "preset": ["presets=a", "presets=b"],
                "res": ["w=64 h=64", "w=128 h=128"],
            },
        )
        runs = matrix.runs()
        tags = [r.tag for r in runs]
        # All tags should be unique
        self.assertEqual(len(tags), len(set(tags)))

    def test_override_string_with_spaces(self):
        """Multi-token override strings are split correctly into CLI args."""
        matrix = BenchmarkMatrix(
            tasks=["TaskA"],
            num_envs=[2048],
            hydra_sweeps={
                "camera": ["env.cam.w=128 env.cam.h=128"],
            },
        )
        runs = matrix.runs()
        args = runs[0].all_hydra_args
        self.assertIn("env.cam.w=128", args)
        self.assertIn("env.cam.h=128", args)

    def test_phase_default_is_step(self):
        matrix = BenchmarkMatrix(
            tasks=["TaskA"],
            num_envs=[2048],
        )
        runs = matrix.runs()
        self.assertEqual(runs[0].phase, Phase.STEP)

    def test_phase_startup(self):
        matrix = BenchmarkMatrix(
            tasks=["TaskA"],
            num_envs=[2048],
            phase=Phase.STARTUP,
        )
        runs = matrix.runs()
        self.assertEqual(runs[0].phase, Phase.STARTUP)

    def test_startup_phase_in_nsys_command(self):
        matrix = BenchmarkMatrix(
            tasks=["TaskA"],
            num_envs=[2048],
            phase=Phase.STARTUP,
        )
        runs = matrix.runs()
        cmd = runs[0].nsys_command("/tmp/out")
        phase_idx = cmd.index("--phase")
        self.assertEqual(cmd[phase_idx + 1], "startup")


if __name__ == "__main__":
    unittest.main()
