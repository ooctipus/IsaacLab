# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the custom coupling manager."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from isaaclab_newton.cloner import NewtonReplicateContext
from isaaclab_newton.physics import MJWarpSolverCfg, VBDSolverCfg

import isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager as manager_module
from isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager import NewtonCoupledMJWarpVBDManager
from isaaclab_contrib.custom_coupling.newton_manager_cfg import CoupledMJWarpVBDSolverCfg

_SIM_CONTEXT = SimpleNamespace(
    cfg=SimpleNamespace(device="cpu", gravity=(0.0, 0.0, -9.81)),
    get_or_create_backend=lambda backend_type, *args, clone_role=None, **kwargs: backend_type(*args, **kwargs),
)


def _make_manager(
    solver_cfg: CoupledMJWarpVBDSolverCfg | None = None,
) -> NewtonCoupledMJWarpVBDManager:
    cfg = solver_cfg or CoupledMJWarpVBDSolverCfg()
    manager = cfg.class_type(cfg)
    manager._newton = NewtonReplicateContext(_SIM_CONTEXT)
    return manager


def test_register_builder_attributes_includes_nested_solvers() -> None:
    """The custom coupled manager delegates builder setup to both configured children."""
    manager = _make_manager()
    manager._bind_context(_SIM_CONTEXT)
    builder = manager.create_builder()

    assert builder.has_custom_attribute("mujoco:condim")


def test_reset_forwards_to_both_subsolvers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the real sub-solvers instead of the dummy solver slot."""
    rigid_solver = MagicMock()
    rigid_solver.use_mujoco_cpu = False
    soft_solver = MagicMock()
    state = object()
    world_mask = object()

    manager = _make_manager()
    manager._rigid_solver = rigid_solver
    manager._soft_solver = soft_solver
    manager._newton._state_0 = state

    manager._reset_solver_internals(world_mask)

    rigid_solver.reset.assert_called_once_with(state, world_mask=world_mask, flags=0)
    soft_solver.reset.assert_called_once_with(state, world_mask=world_mask, flags=0)


def test_reset_skips_all_false_cpu_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CPU warm-start state when no world needs reset."""
    rigid_solver = MagicMock()
    rigid_solver.use_mujoco_cpu = True
    soft_solver = MagicMock()
    world_mask = MagicMock()
    world_mask.numpy.return_value.any.return_value = False

    manager = _make_manager()
    manager._rigid_solver = rigid_solver
    manager._soft_solver = soft_solver

    manager._reset_solver_internals(world_mask)

    rigid_solver.reset.assert_not_called()
    soft_solver.reset.assert_not_called()


@pytest.mark.parametrize(
    ("solver_cfg", "match"),
    [
        (CoupledMJWarpVBDSolverCfg(coupling_mode="invalid"), "coupling_mode"),
        (
            CoupledMJWarpVBDSolverCfg(rigid_solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False)),
            "MJWarp internal contacts",
        ),
        (
            CoupledMJWarpVBDSolverCfg(soft_solver_cfg=VBDSolverCfg()),
            "VBD external rigid-body integration",
        ),
    ],
)
def test_build_solver_rejects_invalid_configuration(solver_cfg: CoupledMJWarpVBDSolverCfg, match: str) -> None:
    manager = _make_manager(solver_cfg)
    with pytest.raises(ValueError, match=match):
        manager._build_solver(MagicMock(), solver_cfg)


def test_build_solver_rejects_contact_sensors(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _make_manager()
    manager._report_contacts = True

    with pytest.raises(NotImplementedError, match="contact sensors are not supported"):
        manager._build_solver(MagicMock(), manager.cfg)


def test_build_solver_sets_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    solver_cfg = CoupledMJWarpVBDSolverCfg()
    manager = _make_manager(solver_cfg)
    manager._report_contacts = False
    manager._supports_contact_sensors = True
    manager._solver = None
    manager._use_single_state = True
    manager._needs_collision_pipeline = False
    manager._newton._supports_rigid_body_force_input = False
    rigid_manager = MagicMock()
    soft_manager = MagicMock()
    monkeypatch.setattr(solver_cfg.rigid_solver_cfg, "class_type", rigid_manager)
    monkeypatch.setattr(solver_cfg.soft_solver_cfg, "class_type", soft_manager)
    monkeypatch.setattr(manager_module, "SolverBase", MagicMock())

    manager._build_solver(MagicMock(), solver_cfg)

    assert manager._supports_contact_sensors is False
    assert manager._newton.supports_rigid_body_force_input() is True


@pytest.mark.parametrize("mode", ["one_way", "two_way"])
def test_step_preserves_input_forces(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    state_in = MagicMock()
    state_in.body_f = object()
    state_in.particle_f = MagicMock()
    state_out = MagicMock()
    control = object()
    contacts = object()
    collision_pipeline = MagicMock()
    rigid_solver = MagicMock()
    soft_solver = MagicMock()
    reactions = MagicMock()

    manager = _make_manager()
    manager._newton._contacts = contacts
    manager._collision_pipeline = collision_pipeline
    manager._rigid_solver = rigid_solver
    manager._soft_solver = soft_solver
    manager._apply_reactions = reactions

    getattr(manager, f"_step_{mode}")(state_in, state_out, control, 0.01)

    state_in.clear_forces.assert_not_called()
    state_in.particle_f.zero_.assert_not_called()
    state_out.clear_forces.assert_called_once_with()
    collision_pipeline.collide.assert_called_once_with(state_in, contacts)
    rigid_solver.step.assert_called_once_with(state_in, state_out, control, None, 0.01)
    soft_solver.step.assert_called_once_with(state_in, state_out, control, contacts, 0.01)
    if mode == "two_way":
        reactions.assert_called_once_with(state_in, state_out, 0.01)
    else:
        reactions.assert_not_called()


def test_solver_specific_clear_releases_subsolvers(monkeypatch: pytest.MonkeyPatch) -> None:
    base_clear = MagicMock()
    monkeypatch.setattr(manager_module.NewtonVBDManager, "_solver_specific_clear", lambda self: base_clear())
    manager = _make_manager()
    base_clear.reset_mock()
    manager._rigid_solver = object()
    manager._soft_solver = object()
    manager._coupling_mode = "two_way"

    manager._solver_specific_clear()

    base_clear.assert_called_once_with()
    assert manager._rigid_solver is None
    assert manager._soft_solver is None
    assert manager._coupling_mode is None
