# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the core Newton VBD integration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from isaaclab_newton.cloner import NewtonReplicateContext
from isaaclab_newton.physics import NewtonManager, NewtonVBDManager, VBDSolverCfg

_SIM_CONTEXT = SimpleNamespace(cfg=SimpleNamespace(device="cpu"))


def _manager() -> NewtonVBDManager:
    manager = object.__new__(NewtonVBDManager)
    manager._newton = NewtonReplicateContext(_SIM_CONTEXT)
    return manager


def test_vbd_colors_builder_around_base_lifecycle(monkeypatch):
    """VBD colors the clone-built builder before finalization."""
    manager = _manager()
    events = []

    class Builder:
        def color(self):
            events.append("color")

    manager._newton._builder = Builder()
    monkeypatch.setattr(NewtonManager, "start_simulation", lambda self: events.append("start"))

    manager.start_simulation()

    assert events == ["color", "start"]


@pytest.mark.parametrize("external_rigid_solver", [False, True])
def test_vbd_solver_force_input_capability(external_rigid_solver):
    """VBD accepts rigid forces only when it integrates rigid bodies."""
    manager = _manager()
    solver = object()
    manager._create_solver = lambda model, cfg: solver

    manager._build_solver(object(), VBDSolverCfg(integrate_with_external_rigid_solver=external_rigid_solver))

    assert manager._solver is solver
    assert manager._use_single_state is False
    assert manager._needs_collision_pipeline is True
    assert manager._newton.supports_rigid_body_force_input() is not external_rigid_solver


def test_vbd_rebuilds_particle_bvh_before_physics_step(monkeypatch):
    """VBD rebuilds its particle BVH before the base physics step."""
    manager = _manager()
    events = []
    state = object()

    class Solver:
        def rebuild_bvh(self, solver_state):
            events.append(("rebuild", solver_state))

    monkeypatch.setattr(NewtonManager, "_simulate_physics_only", lambda self: events.append(("step", self)))
    manager._newton._model = SimpleNamespace(particle_count=1)
    manager._solver = Solver()
    manager._newton._state_0 = state

    manager._simulate_physics_only()

    assert events == [("rebuild", state), ("step", manager)]
