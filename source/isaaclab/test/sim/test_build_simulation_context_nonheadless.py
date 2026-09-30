# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This test has a lot of duplication with ``test_build_simulation_context_headless.py``.

This is intentional to ensure that the tests are run in both headless and non-headless modes,
and we currently can't re-build the simulation app in a script.

If you need to make a change to this test, please make sure to also make the same change to
``test_build_simulation_context_headless.py``.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
from isaaclab_physx.physics import PhysxCfg

from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.simulation_context import build_simulation_context

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("gravity_enabled", [True, False])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("dt", [0.01, 0.1])
def test_build_simulation_context_explicit_cfg(gravity_enabled, device, dt):
    """Test that the simulation context is built from the provided configuration."""
    with build_simulation_context(
        sim_cfg=SimulationCfg(
            physics=PhysxCfg(), dt=dt, gravity=(0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
        ),
        device=device,
    ) as sim:
        if gravity_enabled:
            assert sim.cfg.gravity == (0.0, 0.0, -9.81)
        else:
            assert sim.cfg.gravity == (0.0, 0.0, 0.0)

        assert sim.cfg.device == device
        assert sim.cfg.dt == dt


def test_build_simulation_context_cfg():
    """Test that the simulation context honors sim_cfg's values, with an explicit
    device override winning when both ``sim_cfg`` and ``device`` are passed.

    Most test callers pass both kwargs together expecting the device kwarg to
    win; the override branch in :func:`build_simulation_context` exists for
    that case. ``gravity`` and ``dt`` are not overridable by the helper's
    kwargs (only sim_cfg's values are used).
    """
    dt = 0.001
    # Non-standard gravity
    gravity = (0.0, 0.0, -1.81)
    device = "cuda:0"

    cfg = SimulationCfg(
        physics=PhysxCfg(),
        gravity=gravity,
        device=device,
        dt=dt,
    )

    # Pass only sim_cfg: gravity, device, and dt all come from it.
    with build_simulation_context(
        sim_cfg=cfg,
    ) as sim:
        assert sim.cfg.gravity == gravity
        assert sim.cfg.device == device
        assert sim.cfg.dt == dt

    # Pass sim_cfg and an explicit device override: device kwarg wins.
    with build_simulation_context(sim_cfg=cfg, device="cpu") as sim:
        assert sim.cfg.gravity == gravity
        assert sim.cfg.device == "cpu"
        assert sim.cfg.dt == dt
