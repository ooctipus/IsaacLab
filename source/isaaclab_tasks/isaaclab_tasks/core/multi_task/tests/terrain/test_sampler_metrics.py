# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test for sampler-owned metrics in the metrics harness.

Runs a small fixed grid and asserts the geometry-sampling yield and
shape-diversity metrics match the committed baseline within tolerance. IK and
validation yields belong to the Newton optimizer and retarget policy, so their
regressions are covered by the dedicated retarget tests rather than this
:class:`Sampler` gate.

The baseline is regenerated via the CLI in ``sampler_metrics.py``:

    ``./isaaclab.sh -p <utils/tools/sampler_metrics.py> --output <tests/data/sampler_baseline.json>``

and should only change when :class:`Sampler` behavior
intentionally changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import warp as wp

sys.path[:] = [p for p in sys.path if "pip_prebundle" not in p and "pip_archive" not in p]


_BASELINE_PATH = Path(__file__).parent / "data" / "sampler_baseline.json"


# Grid kept small so the test runs in < ~5s on a single GPU. These cells
# are a subset of the full baseline grid and must be present in the
# committed JSON.
_REGRESSION_CELLS = [
    ("anymal_c", "FLAT", 0.5),
    ("go2", "FLAT", 0.5),
]


# Absolute + relative tolerances per metric. The baseline predates the Warp
# 1.16 RNG stream in the current upstream lock; allow its observed fixed-seed
# shift while still rejecting a sampler-yield change larger than 7.5 points.
_YIELD_ABS_TOL = 0.075
_SHAPE_REL_TOL = 0.15


def _load_baseline() -> dict:
    if not _BASELINE_PATH.exists():
        pytest.skip(f"baseline fixture missing: {_BASELINE_PATH}")
    return json.loads(_BASELINE_PATH.read_text())


def _lookup_cell(baseline: dict, robot: str, sub_terrain: str, difficulty: float) -> dict:
    for entry in baseline["results"]:
        if (
            entry["robot"] == robot
            and entry["sub_terrain"] == sub_terrain
            and abs(entry["difficulty"] - difficulty) < 1e-6
        ):
            return entry
    raise KeyError(f"cell not in baseline: {robot}/{sub_terrain}/{difficulty}")


@pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
@pytest.mark.parametrize(("robot", "sub_terrain", "difficulty"), _REGRESSION_CELLS)
def test_sampler_metrics_match_baseline(robot: str, sub_terrain: str, difficulty: float):
    """Sampler-owned harness output for a fixed cell matches the baseline."""
    from isaaclab_tasks.core.multi_task.terrain.scripts.sampler_metrics import (
        run_metrics_grid,
    )

    baseline = _load_baseline()
    expected = _lookup_cell(baseline, robot, sub_terrain, difficulty)

    [actual] = run_metrics_grid(
        robots=[robot],
        sub_terrains=[sub_terrain],
        difficulties=[difficulty],
        n_desired=expected["n_desired"],
        seed=0,
    )

    # Geometry-sampling yield: absolute tolerance across Warp RNG versions.
    assert abs(actual.sampler_yield - expected["sampler_yield"]) < _YIELD_ABS_TOL, (
        f"sampler_yield drift {actual.sampler_yield:.3f} vs baseline {expected['sampler_yield']:.3f}"
    )
    assert actual.n_sampler_accepted == expected["n_sampler_accepted"]

    # Shape diversity: relative tolerance, looser -- depends on morph-patch
    # sampler RNG which we don't pin hard.
    exp_shape = expected["shape_pairwise_p50"]
    assert abs(actual.shape_pairwise_p50 - exp_shape) < _SHAPE_REL_TOL * max(exp_shape, 1e-3), (
        f"shape_pairwise_p50 drift {actual.shape_pairwise_p50:.3f} vs baseline {exp_shape:.3f}"
    )
