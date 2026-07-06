# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for exact IK objective coordinate mappings."""

from types import SimpleNamespace

import newton.ik as ik
import numpy as np
import pytest
import torch

from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.cfg import IKObjectiveJointDefaultCfg
from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.context import IKObjectiveBuildContext
from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.gravity_torque import IKObjectiveGravityTorque
from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.joint_default import IKObjectiveJointDefault
from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.joint_pin import IKObjectiveJointPin
from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.mesh_collision import IKObjectiveMeshCollision
from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.stability_margin import IKObjectiveStabilityMargin


def _context(
    q_starts: tuple[int, ...],
    qd_starts: tuple[int, ...],
    names: tuple[str, ...],
    default_joint_q: tuple[float, ...],
    n_root_coords: int,
) -> IKObjectiveBuildContext:
    topology = SimpleNamespace(
        joint_count=len(names),
        joint_q_start=np.asarray(q_starts, dtype=np.int32),
        joint_qd_start=np.asarray(qd_starts, dtype=np.int32),
    )
    kinematics = SimpleNamespace(
        topology=topology,
        joint_names=list(names),
        default_joint_q=np.asarray(default_joint_q, dtype=np.float32),
        n_root_coords=n_root_coords,
    )
    return IKObjectiveBuildContext(kinematics=kinematics, asset_name="robot", batch_size=1)


def test_joint_default_keeps_every_fixed_base_scalar_joint() -> None:
    context = _context((0, 1, 2), (0, 1, 2), ("shoulder", "elbow"), (0.1, 0.2), 0)

    objective = IKObjectiveJointDefault(IKObjectiveJointDefaultCfg(skip_root=True), context)

    np.testing.assert_array_equal(objective._coordinate_indices_np, np.asarray((0, 1), dtype=np.int32))
    np.testing.assert_array_equal(objective._velocity_indices_np, np.asarray((0, 1), dtype=np.int32))


def test_joint_default_skips_only_the_free_root() -> None:
    context = _context(
        (0, 7, 8),
        (0, 6, 7),
        ("floating_base", "elbow"),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2),
        7,
    )

    objective = IKObjectiveJointDefault(IKObjectiveJointDefaultCfg(skip_root=True), context)

    np.testing.assert_array_equal(objective._coordinate_indices_np, np.asarray((7,), dtype=np.int32))
    np.testing.assert_array_equal(objective._velocity_indices_np, np.asarray((6,), dtype=np.int32))


def test_joint_default_rejects_non_scalar_ball_coordinates() -> None:
    context = _context((0, 4), (0, 3), ("ball",), (0.0, 0.0, 0.0, 1.0), 0)

    with pytest.raises(ValueError, match="do not map one-to-one"):
        IKObjectiveJointDefault(IKObjectiveJointDefaultCfg(skip_root=True), context)


def test_joint_default_memory_estimate_counts_each_persistent_scalar_once() -> None:
    context = _context((0, 1, 2), (0, 1, 2), ("shoulder", "elbow"), (0.1, 0.2), 0)
    objective = IKObjectiveJointDefault(IKObjectiveJointDefaultCfg(skip_root=True), context)

    estimate = objective.estimate_memory(
        SimpleNamespace(),
        ik.IKJacobianType.ANALYTIC,
        n_problems=17,
        n_batch=17,
        total_residuals=2,
    )

    assert estimate == 2 * (4 + 4 + 4)


def test_joint_pin_memory_estimate_scales_only_its_target_workspace() -> None:
    objective = IKObjectiveJointPin(
        coordinate_indices=np.asarray((7, 8), dtype=np.int32),
        dof_indices=np.asarray((7, 8), dtype=np.int32),
        targets=torch.empty(1, 2),
        weight=1.0,
    )

    estimate = objective.estimate_memory(
        SimpleNamespace(),
        ik.IKJacobianType.ANALYTIC,
        n_problems=17,
        n_batch=17,
        total_residuals=2,
    )
    next_estimate = objective.estimate_memory(
        SimpleNamespace(),
        ik.IKJacobianType.ANALYTIC,
        n_problems=18,
        n_batch=18,
        total_residuals=2,
    )

    assert estimate == 2 * (4 + 4) + 17 * 2 * 4
    assert next_estimate - estimate == 2 * 4


def test_gravity_torque_memory_estimate_matches_owned_array_layout() -> None:
    """Gravity estimate must count fixed mechanics and every batch subtree COM row."""
    objective = IKObjectiveGravityTorque.__new__(IKObjectiveGravityTorque)
    objective.n_rev = 3
    names = (
        "_parent_bodies_np",
        "_axes_local_np",
        "_downstream_bodies_np",
        "_downstream_offsets_np",
        "_subtree_mass_np",
        "_subtree_inv_mass_np",
        "_body_com_np",
        "_body_mass_np",
        "_jac_code_np",
        "_jac_ratio_np",
        "_jac_c_idx_np",
    )
    arrays = (
        np.zeros(3, np.int32),
        np.zeros((3, 3), np.float32),
        np.zeros(8, np.int32),
        np.zeros(4, np.int32),
        np.zeros(3, np.float32),
        np.zeros(3, np.float32),
        np.zeros((6, 3), np.float32),
        np.zeros(6, np.float32),
        np.zeros((3, 5), np.uint8),
        np.zeros((3, 5), np.float32),
        np.zeros((3, 5), np.int32),
    )
    for name, values in zip(names, arrays, strict=True):
        setattr(objective, name, values)
    fixed = sum(values.nbytes for values in arrays)

    analytic = objective.estimate_memory(SimpleNamespace(), ik.IKJacobianType.ANALYTIC, 5, 5, 7)
    autodiff = objective.estimate_memory(SimpleNamespace(), ik.IKJacobianType.AUTODIFF, 5, 5, 7)

    assert analytic == fixed + 5 * 3 * 12
    assert autodiff == analytic + 3 * 5 * 7 * 4


def test_stability_memory_estimate_matches_owned_array_layout() -> None:
    """Stability estimate must count support, active-edge, and subtree workspaces exactly."""
    objective = IKObjectiveStabilityMargin.__new__(IKObjectiveStabilityMargin)
    objective.n_supports = 4
    objective.n_joints = 5
    names = (
        "_support_body_indices_np",
        "_body_mass_np",
        "_body_com_np",
        "_dof_to_joint_np",
        "_joint_subtree_bodies_np",
        "_joint_subtree_offsets_np",
        "_joint_subtree_mass_np",
        "_joint_subtree_inv_mass_np",
        "_support_in_subtree_np",
    )
    arrays = (
        np.zeros(4, np.int32),
        np.zeros(6, np.float32),
        np.zeros((6, 3), np.float32),
        np.zeros(5, np.int32),
        np.zeros(12, np.int32),
        np.zeros(6, np.int32),
        np.zeros(5, np.float32),
        np.zeros(5, np.float32),
        np.zeros((4, 5), np.uint8),
    )
    for name, values in zip(names, arrays, strict=True):
        setattr(objective, name, values)
    fixed = sum(values.nbytes for values in arrays)
    per_batch = 4 * (8 + 4) + 5 * 12 + 3 * 4 + 2 * 4 + 2 * 8

    analytic = objective.estimate_memory(SimpleNamespace(), ik.IKJacobianType.ANALYTIC, 5, 5, 7)
    autodiff = objective.estimate_memory(SimpleNamespace(), ik.IKJacobianType.AUTODIFF, 5, 5, 7)

    assert analytic == fixed + 5 * per_batch
    assert autodiff == analytic + 5 * 7 * 4


def test_mesh_collision_memory_estimate_matches_owned_array_layout() -> None:
    """Collision estimate must distinguish analytic ancestry and autodiff basis storage."""
    objective = IKObjectiveMeshCollision.__new__(IKObjectiveMeshCollision)
    objective._probe_offsets_np = np.zeros((4, 3), np.float32)
    objective._probe_bodies_np = np.zeros(4, np.int32)
    objective._probe_contact_slots_np = np.zeros(4, np.int32)
    objective._probe_affects_dof_np = np.zeros((4, 5), np.uint8)
    objective._contact_mask_t = torch.empty(1, 4, dtype=torch.uint8)
    base = (
        objective._probe_offsets_np.nbytes
        + objective._probe_bodies_np.nbytes
        + objective._probe_contact_slots_np.nbytes
    )

    analytic = objective.estimate_memory(SimpleNamespace(), ik.IKJacobianType.ANALYTIC, 5, 5, 7)
    autodiff = objective.estimate_memory(SimpleNamespace(), ik.IKJacobianType.AUTODIFF, 5, 5, 7)
    mixed = objective.estimate_memory(SimpleNamespace(), ik.IKJacobianType.MIXED, 5, 5, 7)

    assert analytic == base + objective._probe_affects_dof_np.nbytes
    assert autodiff == base + 5 * 7 * 4
    assert mixed == analytic + 5 * 7 * 4
