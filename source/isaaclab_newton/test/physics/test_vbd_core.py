# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the core Newton VBD integration."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from isaaclab_newton.physics import NewtonCfg, NewtonManager, NewtonSoftContactCfg

from isaaclab.physics import PhysicsManager


@pytest.mark.parametrize(
    ("soft_contact_cfg", "expected"),
    [
        pytest.param(None, (7.0, 8.0, 9.0), id="preserve"),
        pytest.param(
            NewtonSoftContactCfg(soft_contact_ke=11.0, soft_contact_kd=12.0, soft_contact_mu=13.0),
            (11.0, 12.0, 13.0),
            id="override",
        ),
    ],
)
def test_soft_contact_cfg_updates_finalized_model(monkeypatch, soft_contact_cfg, expected):
    """Soft-contact configuration updates the finalized model when provided."""
    state_values = []

    class Model:
        soft_contact_ke = 7.0
        soft_contact_kd = 8.0
        soft_contact_mu = 9.0
        world_count = 0
        articulation_count = 0

        def set_gravity(self, gravity):
            pass

        def state(self):
            state_values.append((self.soft_contact_ke, self.soft_contact_kd, self.soft_contact_mu))
            return object()

        def control(self):
            return object()

    class Builder:
        body_label = ()
        up_axis = None

        def finalize(self, *, device):
            return model

    model = Model()
    monkeypatch.setattr(PhysicsManager, "_cfg", NewtonCfg(soft_contact_cfg=soft_contact_cfg), raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu", raising=False)
    monkeypatch.setattr(NewtonManager, "_builder", Builder(), raising=False)
    monkeypatch.setattr(NewtonManager, "_up_axis", "Z", raising=False)
    monkeypatch.setattr(NewtonManager, "_gravity_vector", (0.0, 0.0, -9.81), raising=False)
    monkeypatch.setattr(NewtonManager, "_num_envs", 0, raising=False)
    monkeypatch.setattr(NewtonManager, "_clone_physics_only", True, raising=False)
    monkeypatch.setattr(NewtonManager, "_pending_extended_state_attributes", set(), raising=False)
    monkeypatch.setattr(NewtonManager, "_pending_extended_contact_attributes", set(), raising=False)
    for attr in (
        "_model",
        "_state_0",
        "_state_1",
        "_control",
        "_adapter",
        "_use_newton_actuators_active",
        "_world_reset_mask",
        "_fk_reset_mask",
    ):
        monkeypatch.setattr(NewtonManager, attr, getattr(NewtonManager, attr, None), raising=False)
    monkeypatch.setattr(NewtonManager, "_cl_pending_sites", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_drain_stale_cuda_error", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "dispatch_event", classmethod(lambda cls, event: None))

    NewtonManager.start_simulation()

    assert (model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu) == expected

    assert state_values == [expected, expected]


def test_start_requires_clone_plan_builder(monkeypatch):
    """Newton startup rejects a lifecycle that did not dispatch its clone context."""
    monkeypatch.setattr(NewtonManager, "_builder", None)
    monkeypatch.setattr(NewtonManager, "_drain_stale_cuda_error", classmethod(lambda cls: None))

    with pytest.raises(RuntimeError, match="clone-plan dispatch did not publish"):
        NewtonManager.start_simulation()


def test_vbd_colors_prebuilt_builder_before_start(monkeypatch):
    """VBD colors a prebuilt builder before starting simulation."""
    physics = importlib.import_module("isaaclab_newton.physics")
    deformable_module = importlib.import_module("isaaclab_contrib.deformable.deformable_object")
    events = []

    class Builder:
        def color(self):
            events.append("color")

    monkeypatch.setattr(physics.NewtonVBDManager, "_builder", Builder())
    monkeypatch.setattr(NewtonManager, "start_simulation", classmethod(lambda cls: events.append("start")))
    monkeypatch.setattr(deformable_module, "setup_registered_deformable_fabric_sync", lambda manager_cls: None)

    physics.NewtonVBDManager.start_simulation()

    assert events == ["color", "start"]


@pytest.mark.parametrize("external_rigid_solver", [False, True])
def test_vbd_solver_force_input_capability(monkeypatch, external_rigid_solver):
    """VBD accepts rigid forces only when it integrates rigid bodies."""
    physics = importlib.import_module("isaaclab_newton.physics")
    solver = object()
    monkeypatch.setattr(physics.NewtonVBDManager, "_create_solver", lambda model, cfg: solver)
    monkeypatch.setattr(NewtonManager, "_solver", None)
    monkeypatch.setattr(NewtonManager, "_use_single_state", True)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", False)
    monkeypatch.setattr(NewtonManager, "_supports_rigid_body_force_input", False)

    solver_cfg = physics.VBDSolverCfg(integrate_with_external_rigid_solver=external_rigid_solver)
    physics.NewtonVBDManager._build_solver(object(), solver_cfg)

    assert NewtonManager._solver is solver
    assert NewtonManager._supports_rigid_body_force_input is not external_rigid_solver


def test_vbd_rebuilds_particle_bvh_before_physics_step(monkeypatch):
    """VBD rebuilds its particle BVH before the base physics step."""
    physics = importlib.import_module("isaaclab_newton.physics")
    events = []
    state = object()

    class Solver:
        def rebuild_bvh(self, solver_state):
            events.append(("rebuild", solver_state))

    def simulate_physics_only(cls):
        events.append(("step", cls))

    monkeypatch.setattr(NewtonManager, "_simulate_physics_only", classmethod(simulate_physics_only))
    monkeypatch.setattr(physics.NewtonVBDManager, "_model", SimpleNamespace(particle_count=1))
    monkeypatch.setattr(physics.NewtonVBDManager, "_solver", Solver())
    monkeypatch.setattr(physics.NewtonVBDManager, "_state_0", state)

    physics.NewtonVBDManager._simulate_physics_only()

    assert events == [("rebuild", state), ("step", physics.NewtonVBDManager)]
