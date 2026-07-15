# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytical tests for shared contact-wrench feasibility diagnostics."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch
import warp as wp

from isaaclab_tasks.core.multi_task.kinematics import (
    ContactFeasibilityWorkspace,
    NewtonKinematics,
    contact_feasibility_evaluate,
)
from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex
from isaaclab_tasks.core.multi_task.motion.data.frames import MotionGeneralizedCoordinates
from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionTrajectorySolveCfg
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import (
    _DYNAMICS_QUALITY_NAMES,
    _DYNAMICS_QUALITY_START,
    _DYNAMICS_QUALITY_STOP,
    _QUALITY_NAMES,
    _TARGET_COORDINATE_QUALITY_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table_builder import (
    _MotionCoordinateCandidate,
    _stored_corpus_quality,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import (
    _MotionAcceptedContactEvidence,
    _populate_motion_contact_quality,
)
from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import _G1FrameTarget
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import _SmplFrameTarget


def _single_clip_index(frame_count: int) -> MotionClipIndex:
    return MotionClipIndex(
        source_content_sha256="0" * 64,
        skeleton_identity_sha256s=("1" * 64,),
        clips=(MotionClipIndex.Clip("clip", frame_count, 30.0, "2" * 64, 0),),
    )


def _evaluate(*args, **kwargs):
    """Evaluate with one explicit test certificate policy."""
    return contact_feasibility_evaluate(
        *args,
        iterations=96,
        effort_weight=1.0,
        force_regularization=1.0e-6,
        **kwargs,
    )


def _free_body_kinematics(device: str, *, mass: float = 2.0):
    builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Z)
    body = builder.add_body(mass=mass)
    model = builder.finalize(device=device)
    return SimpleNamespace(model=model, topology=NewtonKinematics._build_topology(model)), body


def _inputs(model, frame_count: int, acceleration_x_mps2: float = 0.0):
    joint_q = torch.from_numpy(np.repeat(model.joint_q.numpy()[None], frame_count, axis=0))
    joint_qd = torch.zeros((frame_count, model.joint_dof_count), dtype=torch.float32)
    joint_qdd = torch.zeros_like(joint_qd)
    joint_qdd[:, 0] = acceleration_x_mps2
    support_point = torch.zeros((1, 3), dtype=torch.float32)
    support_active = torch.ones((frame_count, 1), dtype=torch.bool)
    support_normal = torch.zeros((frame_count, 1, 3), dtype=torch.float32)
    support_normal[..., 2] = 1.0
    friction = torch.full((frame_count, 1), 0.8, dtype=torch.float32)
    return joint_q, joint_qd, joint_qdd, support_point, support_active, support_normal, friction


def _revolute_kinematics(*, effort_limit: float):
    builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Y)
    root = builder.add_link(mass=1.0)
    body = builder.add_link(mass=2.0)
    builder.body_com[body] = wp.vec3(0.5, 0.0, 0.0)
    root_joint = builder.add_joint_fixed(parent=-1, child=root)
    joint = builder.add_joint_revolute(parent=root, child=body, axis=newton.Axis.Z, effort_limit=effort_limit)
    builder.add_articulation([root_joint, joint])
    model = builder.finalize(device="cpu")
    return SimpleNamespace(model=model, topology=NewtonKinematics._build_topology(model)), body


def test_static_free_body_balances_gravity_with_one_support() -> None:
    """A support below the COM must recover weight with zero base residual."""
    kinematics, body = _free_body_kinematics("cpu")
    args = _inputs(kinematics.model, 4)

    result = _evaluate(
        kinematics,
        *args[:3],
        support_body_indices=(body,),
        support_point_body_m=args[3],
        support_active=args[4],
        support_normal_world=args[5],
        friction_coefficient=args[6],
        segment_offsets=(0, 4),
    )

    torch.testing.assert_close(result.contact_wrench_world[:, 0, 2], torch.full((4,), 19.62), atol=2.0e-3, rtol=0.0)
    torch.testing.assert_close(result.balance_force_residual_n, torch.zeros(4), atol=2.0e-3, rtol=0.0)
    torch.testing.assert_close(result.balance_torque_residual_nm, torch.zeros(4), atol=2.0e-3, rtol=0.0)
    assert torch.all(result.friction_margin_n[:, 0] > 0.0)
    assert torch.all(result.normal_force_n[:, 0] > 0.0)


def test_inactive_support_cannot_cancel_gravity() -> None:
    """Inactive source contacts must contribute exactly zero wrench."""
    kinematics, body = _free_body_kinematics("cpu")
    args = list(_inputs(kinematics.model, 3))
    args[4].zero_()

    result = _evaluate(
        kinematics,
        *args[:3],
        support_body_indices=(body,),
        support_point_body_m=args[3],
        support_active=args[4],
        support_normal_world=args[5],
        friction_coefficient=args[6],
        segment_offsets=(0, 3),
    )

    assert torch.count_nonzero(result.contact_wrench_world) == 0
    torch.testing.assert_close(result.balance_force_residual_n, torch.full((3,), 19.62), atol=2.0e-3, rtol=0.0)
    assert torch.all(torch.isnan(result.friction_margin_n))


def test_coulomb_projection_exposes_infeasible_horizontal_acceleration() -> None:
    """Demand beyond mu*g must stay on the cone and leave a balance residual."""
    kinematics, body = _free_body_kinematics("cpu")
    args = list(_inputs(kinematics.model, 3, acceleration_x_mps2=4.0))
    args[6].fill_(0.1)

    result = _evaluate(
        kinematics,
        *args[:3],
        support_body_indices=(body,),
        support_point_body_m=args[3],
        support_active=args[4],
        support_normal_world=args[5],
        friction_coefficient=args[6],
        segment_offsets=(0, 3),
    )

    force = result.contact_wrench_world[:, 0, :3]
    tangent = torch.linalg.vector_norm(force[:, :2], dim=-1)
    torch.testing.assert_close(tangent, 0.1 * force[:, 2], atol=2.0e-4, rtol=0.0)
    assert torch.all(result.balance_force_residual_n > 5.0)
    torch.testing.assert_close(result.friction_margin_n[:, 0], torch.zeros(3), atol=2.0e-4, rtol=0.0)


def test_body_local_support_offset_rotates_into_com_wrench() -> None:
    """A nonzero local contact point must produce the exact world COM torque."""
    kinematics, body = _free_body_kinematics("cpu")
    args = list(_inputs(kinematics.model, 2))
    half_sqrt = 2.0**-0.5
    args[0][:, 3:7] = torch.tensor((0.0, 0.0, half_sqrt, half_sqrt))
    args[3][0, 0] = 0.5

    result = _evaluate(
        kinematics,
        *args[:3],
        support_body_indices=(body,),
        support_point_body_m=args[3],
        support_active=args[4],
        support_normal_world=args[5],
        friction_coefficient=args[6],
        segment_offsets=(0, 2),
    )

    assert torch.all(result.contact_wrench_world[:, 0, 2] > 0.1)
    torch.testing.assert_close(
        result.contact_wrench_world[:, 0, 3],
        0.5 * result.contact_wrench_world[:, 0, 2],
        atol=2.0e-4,
        rtol=0.0,
    )


def test_repeated_support_body_uses_independent_surface_points() -> None:
    """Heel/toe points on one body must balance through distinct COM torques."""
    kinematics, body = _free_body_kinematics("cpu")
    joint_q, joint_qd, joint_qdd, _point, _active, _normal, _friction = _inputs(kinematics.model, 2)
    points = torch.tensor(((-0.25, 0.0, 0.0), (0.25, 0.0, 0.0)), dtype=torch.float32)
    active = torch.ones((2, 2), dtype=torch.bool)
    normal = torch.zeros((2, 2, 3), dtype=torch.float32)
    normal[..., 2] = 1.0
    friction = torch.full((2, 2), 0.8, dtype=torch.float32)

    result = _evaluate(
        kinematics,
        joint_q,
        joint_qd,
        joint_qdd,
        support_body_indices=(body, body),
        support_point_body_m=points,
        support_active=active,
        support_normal_world=normal,
        friction_coefficient=friction,
        segment_offsets=(0, 2),
    )

    torch.testing.assert_close(result.normal_force_n[:, 0], result.normal_force_n[:, 1], atol=2.0e-4, rtol=0.0)
    torch.testing.assert_close(
        result.contact_wrench_world[:, 0, 3:],
        -result.contact_wrench_world[:, 1, 3:],
        atol=2.0e-4,
        rtol=0.0,
    )
    torch.testing.assert_close(result.balance_torque_residual_nm, torch.zeros(2), atol=2.0e-4, rtol=0.0)


def test_post_selection_quality_expands_one_source_slot_to_heel_and_toe() -> None:
    """Final trajectory rows get diagnostics while exact rows remain unavailable."""
    kinematics, body = _free_body_kinematics("cpu")
    clips = (
        MotionClipIndex.Clip("exact", 2, 30.0, "2" * 64, 0),
        MotionClipIndex.Clip("trajectory_a", 3, 30.0, "3" * 64, 0),
        MotionClipIndex.Clip("trajectory_b", 3, 30.0, "4" * 64, 0),
    )
    clip_index = MotionClipIndex("0" * 64, ("1" * 64,), clips)
    joint_q = torch.from_numpy(np.repeat(kinematics.model.joint_q.numpy()[None], 8, axis=0))
    coordinates = MotionGeneralizedCoordinates(joint_q, None)
    quality = torch.zeros((3, len(_QUALITY_NAMES)), dtype=torch.float32)
    quality[:, _DYNAMICS_QUALITY_START:_DYNAMICS_QUALITY_STOP].fill_(torch.nan)
    evidence = _MotionAcceptedContactEvidence(
        sequence_indices=(1, 2),
        source_stable=torch.ones((6, 1), dtype=torch.bool),
        support_body_indices=(body, body),
        support_point_body_m=torch.tensor(((-0.25, 0.0, 0.0), (0.25, 0.0, 0.0)), dtype=torch.float32),
        support_channel_slots=torch.tensor((0, 0), dtype=torch.int64),
        policy=MotionTrajectorySolveCfg.DynamicsCfg(
            friction_coefficient=0.7,
            iterations=96,
            effort_weight=1.0,
            force_regularization=1.0e-6,
        ),
    )

    class IdentityBuilder:
        @staticmethod
        def write_joint_position_newton(source: MotionGeneralizedCoordinates, output: torch.Tensor) -> None:
            output.copy_(source.joint_q)

    target = IdentityBuilder()
    target.kinematics = kinematics
    _populate_motion_contact_quality(target, clip_index, coordinates, quality, evidence)

    assert tuple(_DYNAMICS_QUALITY_NAMES) == _QUALITY_NAMES[_DYNAMICS_QUALITY_START:_DYNAMICS_QUALITY_STOP]
    assert torch.isnan(quality[0, _DYNAMICS_QUALITY_START:_DYNAMICS_QUALITY_STOP]).all()
    torch.testing.assert_close(quality[1:, _DYNAMICS_QUALITY_START], torch.zeros(2), atol=2.0e-3, rtol=0.0)
    torch.testing.assert_close(quality[1:, _DYNAMICS_QUALITY_START + 1], torch.zeros(2), atol=2.0e-3, rtol=0.0)
    assert torch.isnan(quality[1:, _DYNAMICS_QUALITY_START + 2]).all()
    assert torch.all(quality[1:, _DYNAMICS_QUALITY_START + 3] > 0.0)
    assert torch.all(quality[1:, _DYNAMICS_QUALITY_START + 4] > 0.0)
    assert torch.count_nonzero(quality[1:, _DYNAMICS_QUALITY_START + 5]) == 0


def test_stored_coordinate_quality_marks_dynamics_unavailable() -> None:
    """Exact and analytic coordinate routes must not fabricate dynamics evidence."""
    clip_index = _single_clip_index(2)
    coordinates = MotionGeneralizedCoordinates(torch.zeros((2, 7), dtype=torch.float32), None)
    candidate = _MotionCoordinateCandidate(
        target=SimpleNamespace(),
        clip_index=clip_index,
        coordinates=coordinates,
        target_coordinate_evidence=torch.zeros((1, len(_TARGET_COORDINATE_QUALITY_NAMES))),
        device="cpu",
    )

    quality = _stored_corpus_quality(candidate, torch.ones(1, dtype=torch.bool))

    assert torch.isnan(quality[:, _DYNAMICS_QUALITY_START:_DYNAMICS_QUALITY_STOP]).all()


def test_support_point_shape_is_constant_over_frames() -> None:
    """Per-frame world contact storage must fail at the public boundary."""
    kinematics, body = _free_body_kinematics("cpu")
    args = list(_inputs(kinematics.model, 2))

    with pytest.raises(ValueError, match="support_point_body_m"):
        _evaluate(
            kinematics,
            *args[:3],
            support_body_indices=(body,),
            support_point_body_m=args[3][None].expand(2, -1, -1).contiguous(),
            support_active=args[4],
            support_normal_world=args[5],
            friction_coefficient=args[6],
            segment_offsets=(0, 2),
        )


def test_contact_changes_are_reported_but_not_turned_into_a_gate() -> None:
    """Segment diagnostics identify impact boundaries without returning acceptance."""
    kinematics, body = _free_body_kinematics("cpu")
    args = list(_inputs(kinematics.model, 5))
    args[4][:2].zero_()

    result = _evaluate(
        kinematics,
        *args[:3],
        support_body_indices=(body,),
        support_point_body_m=args[3],
        support_active=args[4],
        support_normal_world=args[5],
        friction_coefficient=args[6],
        segment_offsets=(0, 5),
    )

    assert result.contact_transition.tolist() == [False, False, True, False, False]
    assert result.segment_contact_transition_count.tolist() == [1]
    assert not hasattr(result, "accepted")


def test_workspace_reuses_capacity_for_shorter_complete_segment_batch() -> None:
    """A shorter second batch must reuse large mechanics/contact allocations."""
    kinematics, body = _free_body_kinematics("cpu")
    workspace = ContactFeasibilityWorkspace(kinematics, 5, (body,))
    pointers = (
        workspace.body_q.data_ptr(),
        workspace.generalized_free.data_ptr(),
        workspace.body_f.data_ptr(),
        workspace.contact_map.data_ptr(),
    )
    first = _inputs(kinematics.model, 5)
    workspace.evaluate(
        *first[:3],
        support_point_body_m=first[3],
        support_active=first[4],
        support_normal_world=first[5],
        friction_coefficient=first[6],
        segment_offsets=(0, 5),
        iterations=96,
        effort_weight=1.0,
        force_regularization=1.0e-6,
    )
    second = _inputs(kinematics.model, 2)
    result = workspace.evaluate(
        *second[:3],
        support_point_body_m=second[3],
        support_active=second[4],
        support_normal_world=second[5],
        friction_coefficient=second[6],
        segment_offsets=(0, 2),
        iterations=96,
        effort_weight=1.0,
        force_regularization=1.0e-6,
    )

    assert result.generalized_effort.shape == (2, 6)
    assert pointers == (
        workspace.body_q.data_ptr(),
        workspace.generalized_free.data_ptr(),
        workspace.body_f.data_ptr(),
        workspace.contact_map.data_ptr(),
    )
    estimate_one = ContactFeasibilityWorkspace.estimate_memory(kinematics, 1, 1)
    estimate_two = ContactFeasibilityWorkspace.estimate_memory(kinematics, 2, 1)
    estimate_five = ContactFeasibilityWorkspace.estimate_memory(kinematics, 5, 1)
    estimate_ten = ContactFeasibilityWorkspace.estimate_memory(kinematics, 10, 1)
    assert estimate_ten - estimate_five == 5 * (estimate_two - estimate_one)
    owned = (
        workspace.joint_q,
        workspace.joint_qd,
        workspace.joint_qdd,
        workspace.body_q,
        workspace.body_qd,
        workspace.body_com_world,
        workspace.support_position_world,
        workspace.generalized_free,
        workspace.generalized_trial,
        workspace.body_f,
        workspace.contact_map,
        workspace.support_active,
        workspace.support_normal_world,
        workspace.friction_coefficient,
        workspace.axes,
    )
    persistent = sum(tensor.numel() * tensor.element_size() for tensor in owned)
    persistent += newton.dynamics.DynamicsInverse.estimate_memory(kinematics.model, workspace.frame_capacity)
    assert estimate_five >= persistent


def test_g1_write_joint_position_newton_is_exact_identity() -> None:
    """G1 stored trajectory coordinates must copy to Newton without conversion."""
    joint_q = torch.arange(30, dtype=torch.float32).view(3, 10)
    coordinates = MotionGeneralizedCoordinates(joint_q, None)
    output = torch.empty_like(joint_q)

    _G1FrameTarget.write_joint_position_newton(object.__new__(_G1FrameTarget), coordinates, output)

    torch.testing.assert_close(output, joint_q)


def test_smpl_stored_coordinates_round_trip_to_newton_joint_position() -> None:
    """SMPL storage wxyz conversion must invert exactly in a batch workspace."""
    joint_q = torch.zeros((3, 10), dtype=torch.float32)
    joint_q[:, 2] = 1.0
    joint_q[:, 3:7] = torch.tensor((0.1, 0.2, 0.3, 0.92736185))
    joint_q[:, 7:] = torch.tensor(((0.0, 0.1, 0.2), (0.1, 0.2, 0.3), (0.2, 0.3, 0.4)))
    builder = object.__new__(_SmplFrameTarget)
    object.__setattr__(builder, "reference_coordinate_names", ("joint_x", "joint_y", "joint_z"))
    object.__setattr__(builder, "_coordinate_axes", torch.eye(3).unsqueeze(0))
    stored = builder.coordinates_from_newton(joint_q, _single_clip_index(3))
    output = torch.empty_like(joint_q)

    builder.write_joint_position_newton(stored, output)

    torch.testing.assert_close(output, joint_q)


def test_effort_margin_uses_topology_bounds() -> None:
    """Required generalized effort must be compared with the exact model bound."""
    kinematics, body = _revolute_kinematics(effort_limit=1.0)
    args = list(_inputs(kinematics.model, 2))
    args[4].zero_()

    result = _evaluate(
        kinematics,
        *args[:3],
        support_body_indices=(body,),
        support_point_body_m=args[3],
        support_active=args[4],
        support_normal_world=args[5],
        friction_coefficient=args[6],
        segment_offsets=(0, 2),
    )

    assert torch.all(result.generalized_effort[:, 0] > 9.0)
    assert torch.all(result.generalized_effort_margin[:, 0] < -8.0)
    assert torch.all(result.effort_margin_ratio < -8.0)


def test_passive_non_root_mechanism_is_rejected() -> None:
    """Zero-range non-root mechanisms must not be silently treated as actuated."""
    builder = newton.ModelBuilder(gravity=0.0)
    root = builder.add_link(mass=1.0)
    child = builder.add_link(mass=1.0)
    root_joint = builder.add_joint_fixed(parent=-1, child=root)
    passive_joint = builder.add_joint_revolute(parent=root, child=child, axis=newton.Axis.Z, effort_limit=0.0)
    builder.add_articulation([root_joint, passive_joint])
    model = builder.finalize(device="cpu")
    kinematics = SimpleNamespace(model=model, topology=NewtonKinematics._build_topology(model))
    args = list(_inputs(model, 2))
    args[4].zero_()

    with pytest.raises(NotImplementedError, match="non-root"):
        _evaluate(
            kinematics,
            *args[:3],
            support_body_indices=(child,),
            support_point_body_m=args[3],
            support_active=args[4],
            support_normal_world=args[5],
            friction_coefficient=args[6],
            segment_offsets=(0, 2),
        )


def test_cuda_result_stays_device_resident() -> None:
    """The batched public Newton path must preserve CUDA residency."""
    if not torch.cuda.is_available():
        return
    kinematics, body = _free_body_kinematics("cuda:0")
    args = tuple(value.cuda() for value in _inputs(kinematics.model, 8))

    with pytest.raises(ValueError, match="share one device"):
        _evaluate(
            kinematics,
            *args[:3],
            support_body_indices=(body,),
            support_point_body_m=args[3].cpu(),
            support_active=args[4],
            support_normal_world=args[5],
            friction_coefficient=args[6],
            segment_offsets=(0, 8),
        )

    result = _evaluate(
        kinematics,
        *args[:3],
        support_body_indices=(body,),
        support_point_body_m=args[3],
        support_active=args[4],
        support_normal_world=args[5],
        friction_coefficient=args[6],
        segment_offsets=(0, 8),
    )

    assert all(getattr(result, field.name).device.type == "cuda" for field in fields(result))
