# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test for the sampler metrics harness.

Runs a small fixed grid and asserts the yields and shape-diversity metrics
match the committed baseline within tolerance. This pins
:class:`Sampler` behavior so the Phase 1 abstraction refactor
(typed :class:`SamplerOutput`, class rename) can be verified as non-breaking.

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


# Absolute + relative tolerances per metric. Chosen empirically to catch a
# drift of more than a couple of percent while tolerating stochastic noise
# from morphological patch sampling.
_YIELD_ABS_TOL = 0.05
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
def test_matches_baseline(robot: str, sub_terrain: str, difficulty: float):
    """Harness output for a fixed cell matches the committed baseline."""
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

    # Yield rates: abs tolerance -- stochastic but tight.
    assert abs(actual.sampler_yield - expected["sampler_yield"]) < _YIELD_ABS_TOL, (
        f"sampler_yield drift {actual.sampler_yield:.3f} vs baseline {expected['sampler_yield']:.3f}"
    )
    assert abs(actual.ik_yield - expected["ik_yield"]) < _YIELD_ABS_TOL, (
        f"ik_yield drift {actual.ik_yield:.3f} vs baseline {expected['ik_yield']:.3f}"
    )
    assert abs(actual.final_yield - expected["final_yield"]) < _YIELD_ABS_TOL, (
        f"final_yield drift {actual.final_yield:.3f} vs baseline {expected['final_yield']:.3f}"
    )

    # Shape diversity: relative tolerance, looser -- depends on morph-patch
    # sampler RNG which we don't pin hard.
    exp_shape = expected["shape_pairwise_p50"]
    assert abs(actual.shape_pairwise_p50 - exp_shape) < _SHAPE_REL_TOL * max(exp_shape, 1e-3), (
        f"shape_pairwise_p50 drift {actual.shape_pairwise_p50:.3f} vs baseline {exp_shape:.3f}"
    )

    # Must still reach the requested placement count.
    assert actual.n_final == expected["n_final"], f"n_final drift {actual.n_final} vs baseline {expected['n_final']}"
