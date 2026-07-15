# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Semantic tests for motion trajectory velocity ownership."""

from __future__ import annotations

import ast
import inspect
import math
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.utils.math import quat_from_rotation_vector

from isaaclab_tasks.core.multi_task.kinematics import (
    ordered_hinge_coordinate_velocity,
    ordered_hinge_rotation,
    time_backward_difference_segmented,
    time_quaternion_angular_velocity_segmented,
)
from isaaclab_tasks.core.multi_task.kinematics.trajectory import IKTrajectorySolver
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import (
    _motion_frame_seed_global_gather,
    _motion_frame_seed_local_gather,
    _motion_frame_seed_local_scatter,
    _motion_frame_seed_project,
    _motion_initializer_validate_or_restore,
    _motion_scalar_velocity_box_witness,
    motion_solve_trajectory,
)
from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import _G1FrameTarget
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import _SmplFrameTarget
from isaaclab_tasks.core.multi_task.motion.robots.target import write_velocity_canonical

wp.init()


def _free_root_trajectory() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return two clips whose boundary jump must never become a velocity edge."""
    root_x = torch.tensor((0.0, 1.0, 3.0, 100.0, 101.0, 104.0), dtype=torch.float32)
    root_yaw = torch.tensor((0.0, 0.1, 0.4, 2.0, 2.2, 2.6), dtype=torch.float32)
    joint = torch.tensor((0.0, 2.0, 5.0, -30.0, -29.0, -27.0), dtype=torch.float32)
    joint_q = torch.zeros((6, 8), dtype=torch.float32)
    joint_q[:, 0].copy_(root_x)
    root_rotation_vector = torch.stack((root_yaw * 0.0, root_yaw * 0.0, root_yaw), dim=-1)
    joint_q[:, 3:7].copy_(quat_from_rotation_vector(root_rotation_vector))
    joint_q[:, 7].copy_(joint)
    offsets = torch.tensor((0, 3, 6), dtype=torch.int32)
    step_seconds = torch.tensor((0.5, 0.25), dtype=torch.float32)
    return joint_q, offsets, step_seconds


class _ScalarTrajectoryTarget:
    """One scalar target used to isolate shared free-root velocity semantics."""

    def __init__(self, body_com: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        model = SimpleNamespace(
            joint_coord_count=8,
            joint_dof_count=7,
            body_count=1,
            body_com=wp.array([body_com], dtype=wp.vec3, device="cpu"),
        )
        self.kinematics = SimpleNamespace(model=model)
        self.kinematic_tree = SimpleNamespace(root_body_index=0)

    def write_nonroot_velocity_canonical(
        self,
        joint_q: torch.Tensor,
        clip_offsets: torch.Tensor,
        step_seconds: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Write the single non-root backward edge into full Newton storage."""
        values = time_backward_difference_segmented(joint_q[:, 7:], clip_offsets.to(torch.int64), step_seconds)
        output[:, 6:].copy_(values)


def test_canonical_velocity_is_a_segmented_backward_edge() -> None:
    """Frame velocity describes the edge reaching it; each clip head repeats its first real edge."""
    joint_q, offsets, step_seconds = _free_root_trajectory()
    joint_qd = torch.empty((joint_q.shape[0], joint_q.shape[1] - 1), dtype=torch.float32)
    write_velocity_canonical(_ScalarTrajectoryTarget(), joint_q, offsets, step_seconds, joint_qd)

    torch.testing.assert_close(joint_qd[:, 0], torch.tensor((2.0, 2.0, 4.0, 4.0, 4.0, 12.0)))
    torch.testing.assert_close(joint_qd[:, 5], torch.tensor((0.2, 0.2, 0.6, 0.8, 0.8, 1.6)), atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(joint_qd[:, 6], torch.tensor((4.0, 4.0, 6.0, 4.0, 4.0, 8.0)))
    assert math.isfinite(float(joint_qd.abs().max()))


def test_scalar_velocity_box_witness_is_exact_and_idempotent() -> None:
    """The physical seed is an exact bounded-Lipschitz witness, not a nonlinear Phase-I guess."""
    device = torch.device("cpu")
    joint_q = torch.tensor(
        ((0.0, -3.0), (4.0, 3.0), (-4.0, -2.0), (2.0, 2.0), (1.5, -2.0), (-2.0, 2.0), (2.0, -2.0)),
        dtype=torch.float32,
        device=device,
    )
    clip_offsets = torch.tensor((0, 4, 7), dtype=torch.int32, device=device)
    step_seconds = torch.tensor((0.5, 0.25), dtype=torch.float32, device=device)
    coordinate_indices = torch.tensor((0, 1), dtype=torch.int32, device=device)
    dof_indices = torch.tensor((0, 1), dtype=torch.int32, device=device)
    coordinate_lower = torch.tensor((-2.0, -1.0), dtype=torch.float32, device=device)
    coordinate_upper = torch.tensor((2.0, 1.0), dtype=torch.float32, device=device)
    velocity_lower = torch.tensor((-1.0, -2.0), dtype=torch.float32, device=device)
    velocity_upper = torch.tensor((1.0, 2.0), dtype=torch.float32, device=device)
    reachable_lower = torch.empty_like(joint_q)
    reachable_upper = torch.empty_like(joint_q)

    def project() -> torch.Tensor:
        active = torch.ones(2, dtype=torch.int32, device=device)
        wp.launch(
            _motion_scalar_velocity_box_witness,
            dim=(2, 2),
            inputs=[
                wp.from_torch(joint_q),
                wp.from_torch(coordinate_indices),
                wp.from_torch(dof_indices),
                wp.from_torch(coordinate_lower),
                wp.from_torch(coordinate_upper),
                wp.from_torch(velocity_lower),
                wp.from_torch(velocity_upper),
                wp.from_torch(step_seconds),
                wp.from_torch(clip_offsets),
                wp.from_torch(active),
                2,
                2,
            ],
            outputs=[wp.from_torch(reachable_lower), wp.from_torch(reachable_upper)],
            device=str(device),
        )
        wp.synchronize_device(str(device))
        return active

    torch.testing.assert_close(project(), torch.ones(2, dtype=torch.int32))
    assert torch.all(joint_q >= coordinate_lower)
    assert torch.all(joint_q <= coordinate_upper)
    for segment, step in enumerate(step_seconds):
        start = int(clip_offsets[segment])
        stop = int(clip_offsets[segment + 1])
        velocity = (joint_q[start + 1 : stop] - joint_q[start : stop - 1]) / step
        assert torch.all(velocity >= velocity_lower - 1.0e-6)
        assert torch.all(velocity <= velocity_upper + 1.0e-6)

    witness = joint_q.clone()
    torch.testing.assert_close(project(), torch.ones(2, dtype=torch.int32))
    torch.testing.assert_close(joint_q, witness, rtol=0.0, atol=0.0)


def test_scalar_velocity_box_witness_reports_incompatible_rate_and_box_bounds() -> None:
    """Empty backward reachable intervals are reported as a real feasibility failure."""
    joint_q = torch.zeros((3, 1), dtype=torch.float32)
    active = torch.ones(1, dtype=torch.int32)
    scratch_lower = torch.empty_like(joint_q)
    scratch_upper = torch.empty_like(joint_q)
    wp.launch(
        _motion_scalar_velocity_box_witness,
        dim=(1, 1),
        inputs=[
            wp.from_torch(joint_q),
            wp.array((0,), dtype=wp.int32, device="cpu"),
            wp.array((0,), dtype=wp.int32, device="cpu"),
            wp.array((0.0,), dtype=wp.float32, device="cpu"),
            wp.array((0.0,), dtype=wp.float32, device="cpu"),
            wp.array((1.0,), dtype=wp.float32, device="cpu"),
            wp.array((2.0,), dtype=wp.float32, device="cpu"),
            wp.array((1.0,), dtype=wp.float32, device="cpu"),
            wp.array((0, 3), dtype=wp.int32, device="cpu"),
            wp.from_torch(active),
            1,
            1,
        ],
        outputs=[wp.from_torch(scratch_lower), wp.from_torch(scratch_upper)],
        device="cpu",
    )
    wp.synchronize_device("cpu")

    torch.testing.assert_close(active, torch.zeros(1, dtype=torch.int32))


def test_canonical_free_root_is_quaternion_sign_invariant_and_uses_com_velocity() -> None:
    """Root qd uses shortest world angular velocity and the rotated COM offset."""
    target = _ScalarTrajectoryTarget(body_com=(1.0, 0.0, 0.0))
    joint_q = torch.zeros((2, 8), dtype=torch.float32)
    joint_q[:, 6] = 1.0
    rotation_vector = torch.tensor(((0.0, 0.0, 0.0), (0.0, 0.0, 0.2)), dtype=torch.float32)
    rotations = quat_from_rotation_vector(rotation_vector)
    rotations[1].neg_()
    joint_q[:, 3:7].copy_(rotations)
    offsets = torch.tensor((0, 2), dtype=torch.int32)
    step_seconds = torch.tensor((0.5,), dtype=torch.float32)
    joint_qd = torch.empty((2, 7), dtype=torch.float32)

    write_velocity_canonical(target, joint_q, offsets, step_seconds, joint_qd)

    expected_angular = torch.tensor(((0.0, 0.0, 0.4), (0.0, 0.0, 0.4)))
    expected_linear = torch.tensor(
        (
            (0.0, 0.4, 0.0),
            (-0.4 * math.sin(0.2), 0.4 * math.cos(0.2), 0.0),
        )
    )
    torch.testing.assert_close(joint_qd[:, 3:6], expected_angular, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(joint_qd[:, :3], expected_linear, atol=1.0e-6, rtol=1.0e-6)


def test_local_frame_seed_projection_keeps_velocity_coupling_in_trajectory_solver() -> None:
    """Frame IK owns coordinate validity while the whole-trajectory solve owns native-time coupling."""
    source = inspect.getsource(motion_solve_trajectory)
    tree = ast.parse(source)
    step_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "frame_seed_local_optimizer"
        and node.func.attr == "step"
    ]
    assert len(step_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in step_calls[0].keywords}
    assert isinstance(keywords["iterations"], ast.Name)
    assert keywords["iterations"].id == "_FRAME_SEED_LOCAL_ITERATIONS"
    assert isinstance(keywords["projection"], ast.Name)
    projector_name = keywords["projection"].id
    assert isinstance(keywords["projection_interval"], ast.Constant)
    assert keywords["projection_interval"].value == 1

    projectors = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == projector_name]
    assert len(projectors) == 1
    launches = [
        node
        for node in ast.walk(projectors[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch"
        and node.args
        and isinstance(node.args[0], ast.Name)
    ]
    assert len(launches) == 1
    assert launches[0].args[0].id == "_motion_frame_seed_project"

    def launch_names(call: ast.Call, argument: str) -> set[str]:
        values = next(keyword.value for keyword in call.keywords if keyword.arg == argument)
        assert isinstance(values, ast.List)
        return {node.id for node in ast.walk(values) if isinstance(node, ast.Name)}

    assert {"values", "frame_seed_local_joint_q_wp", "source_root_fixed"} <= launch_names(launches[0], "inputs")
    global_projectors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "project_global_frame_seeds"
    ]
    assert len(global_projectors) == 1
    global_launches = [
        node
        for node in ast.walk(global_projectors[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "_motion_frame_seed_project"
    ]
    assert len(global_launches) == 1
    assert {"frame_seed_global_joint_q_wp", "source_root_fixed"} <= launch_names(global_launches[0], "inputs")

    local_gather_launches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "_motion_frame_seed_local_gather"
    ]
    assert len(local_gather_launches) == 1
    assert {"initializer_joint_q_wp", "initializer_baseline_wp"} <= launch_names(local_gather_launches[0], "inputs")
    assert {
        "frame_seed_local_joint_q_wp",
        "frame_seed_local_candidate_joint_q_wp",
        "frame_seed_frame_indices_wp",
        "frame_seed_frame_active_wp",
    } <= launch_names(local_gather_launches[0], "outputs")

    optimizer_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "frame_seed_local_optimizer" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "ik"
        and node.value.func.attr == "IKOptimizerLM"
    ]
    assert len(optimizer_assignments) == 1
    optimizer_keywords = {keyword.arg: keyword.value for keyword in optimizer_assignments[0].value.keywords}
    problem_indices = optimizer_keywords["problem_idx"]
    assert isinstance(problem_indices, ast.Call)
    assert isinstance(problem_indices.func, ast.Attribute)
    assert isinstance(problem_indices.func.value, ast.Name)
    assert problem_indices.func.value.id == "wp"
    assert problem_indices.func.attr == "from_torch"
    assert isinstance(problem_indices.args[0], ast.Name)
    assert problem_indices.args[0].id == "frame_seed_local_problem_indices"

    cost_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "frame_seed_local_optimizer"
        and node.func.attr == "compute_costs"
    ]
    assert len(cost_calls) == 1
    assert isinstance(cost_calls[0].args[0], ast.Name)
    assert cost_calls[0].args[0].id == "frame_seed_local_candidate_joint_q_wp"

    local_scatter_launches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "_motion_frame_seed_local_scatter"
    ]
    assert len(local_scatter_launches) == 1
    assert {
        "frame_seed_local_candidate_joint_q_wp",
        "frame_seed_local_cost_wp",
        "initializer_baseline_wp",
        "frame_seed_frame_indices_wp",
        "frame_seed_frame_active_wp",
        "initializer_coordinate_indices_wp",
        "initializer_coordinate_lower_wp",
        "initializer_coordinate_upper_wp",
        "source_root_fixed",
    } <= launch_names(local_scatter_launches[0], "inputs")
    assert {"initializer_joint_q_wp"} <= launch_names(local_scatter_launches[0], "outputs")
    assert step_calls[0].lineno < cost_calls[0].lineno < local_scatter_launches[0].lineno

    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    forbidden_frame_velocity_owners = {
        "_motion_frame_seed_project_local",
        "direct_coordinate_qd_indices",
        "frame_seed_direct_coordinate_dof_indices",
        "frame_seed_direct_coordinate_dof_indices_wp",
        "frame_seed_previous_joint_q",
        "frame_seed_previous_joint_q_wp",
        "frame_seed_velocity_lower_wp",
        "frame_seed_velocity_upper_wp",
        "step_seconds_wp",
    }
    assert names.isdisjoint(forbidden_frame_velocity_owners)

    coupled_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "solver"
        and node.func.attr == "solve"
    ]
    assert len(coupled_calls) == 1
    coupled_keywords = {keyword.arg for keyword in coupled_calls[0].keywords}
    assert {"joint_velocity", "velocity_lower", "velocity_upper"} <= coupled_keywords


def test_frame_seed_projection_freezes_only_target_fixed_roots() -> None:
    """Expanded SMPL candidates use their mapped roots while G1 roots remain optimized."""
    root_reference = torch.zeros((2, 9), dtype=torch.float32)
    root_reference[0, :3] = torch.tensor((1.0, 2.0, 3.0))
    root_reference[1, :3] = torch.tensor((-1.0, -2.0, -3.0))
    root_reference[0, 6] = 1.0
    root_reference[1, 5:7] = torch.tensor((0.6, 0.8))
    candidates = torch.zeros((6, 9), dtype=torch.float32)
    candidates[:, :3] = torch.arange(18, dtype=torch.float32).view(6, 3)
    candidates[:, 3:7] = torch.tensor((1.0, 2.0, 3.0, 4.0))
    candidates[:, 7] = torch.linspace(-3.0, 3.0, 6)
    candidates[:, 8] = torch.linspace(2.0, -2.0, 6)
    coordinate_indices = torch.tensor((7, 8), dtype=torch.int64)
    coordinate_lower = torch.tensor((-1.0, -0.5))
    coordinate_upper = torch.tensor((1.0, 0.5))

    fixed_candidates = candidates.clone()
    wp.launch(
        _motion_frame_seed_project,
        dim=(6, 2),
        inputs=[
            wp.from_torch(fixed_candidates),
            wp.from_torch(root_reference),
            wp.from_torch(coordinate_indices),
            wp.from_torch(coordinate_lower),
            wp.from_torch(coordinate_upper),
            6,
            2,
            3,
            wp.uint8(1),
        ],
        device="cpu",
    )

    torch.testing.assert_close(fixed_candidates[:, :7], root_reference[:, :7].repeat_interleave(3, dim=0))
    torch.testing.assert_close(fixed_candidates[:, 7], candidates[:, 7].clamp(-1.0, 1.0))
    torch.testing.assert_close(fixed_candidates[:, 8], candidates[:, 8].clamp(-0.5, 0.5))

    optimized_candidates = candidates.clone()
    wp.launch(
        _motion_frame_seed_project,
        dim=(6, 2),
        inputs=[
            wp.from_torch(optimized_candidates),
            wp.from_torch(root_reference),
            wp.from_torch(coordinate_indices),
            wp.from_torch(coordinate_lower),
            wp.from_torch(coordinate_upper),
            6,
            2,
            3,
            wp.uint8(0),
        ],
        device="cpu",
    )
    torch.testing.assert_close(optimized_candidates[:, :3], candidates[:, :3])
    torch.testing.assert_close(torch.linalg.vector_norm(optimized_candidates[:, 3:7], dim=-1), torch.ones(6))


@pytest.mark.parametrize("preinvalid", (False, True))
def test_invalid_frame_seed_restores_baseline_but_remains_inactive(preinvalid: bool) -> None:
    """A finite inspection fallback never erases discovered or preexisting invalidity."""
    baseline = torch.zeros((2, 8), dtype=torch.float32)
    baseline[:, 6] = 1.0
    joint_q = baseline.clone()
    joint_q[1, 7] = 0.5 if preinvalid else float("nan")
    joint_qd = torch.zeros((2, 7), dtype=torch.float32)
    coordinate_indices = torch.tensor((7,), dtype=torch.int64)
    coordinate_lower = torch.tensor((-1.0,), dtype=torch.float32)
    coordinate_upper = torch.tensor((1.0,), dtype=torch.float32)
    clip_offsets = torch.tensor((0, 2), dtype=torch.int32)
    segment_active = torch.tensor((int(not preinvalid),), dtype=torch.int32)

    wp.launch(
        _motion_initializer_validate_or_restore,
        dim=1,
        inputs=[
            wp.from_torch(joint_q),
            wp.from_torch(baseline),
            wp.from_torch(joint_qd),
            wp.from_torch(coordinate_indices),
            wp.from_torch(coordinate_lower),
            wp.from_torch(coordinate_upper),
            wp.from_torch(clip_offsets),
            1,
            8,
            7,
            1,
        ],
        outputs=[wp.from_torch(segment_active)],
        device="cpu",
    )

    torch.testing.assert_close(joint_q, baseline)
    torch.testing.assert_close(segment_active, torch.zeros_like(segment_active))


def test_frame_seed_gathers_true_variable_clip_frames() -> None:
    """Global and local seed buffers gather exact frames without clip padding."""
    frame_count = 5
    source_landmarks = torch.arange(frame_count, dtype=torch.float32).view(1, frame_count, 1).expand(-1, -1, 3).clone()
    source_rotations = torch.zeros((2, frame_count, 4), dtype=torch.float32)
    source_rotations[0, :, 0] = torch.arange(frame_count, dtype=torch.float32)
    source_rotations[1, :, 1] = torch.arange(frame_count, dtype=torch.float32) + 10.0
    source_rotations[:, :, 3] = 1.0
    source_directions = (source_landmarks + 10.0).clone()
    source_joint_q = torch.zeros((frame_count, 9), dtype=torch.float32)
    source_joint_q[:, 0] = torch.arange(frame_count, dtype=torch.float32)
    source_joint_q[:, 6] = 1.0
    clip_offsets = torch.tensor((0, 3, 5), dtype=torch.int32)
    global_landmarks = torch.empty((1, 2, 3), dtype=torch.float32)
    global_rotations = torch.empty((2, 2, 4), dtype=torch.float32)
    global_directions = torch.empty((1, 2, 3), dtype=torch.float32)
    global_joint_q = torch.empty((2, 9), dtype=torch.float32)

    wp.launch(
        _motion_frame_seed_global_gather,
        dim=2,
        inputs=[
            wp.from_torch(source_landmarks),
            wp.from_torch(source_rotations),
            wp.from_torch(source_directions),
            wp.from_torch(source_joint_q),
            wp.from_torch(clip_offsets),
            2,
            1,
            2,
            1,
            9,
        ],
        outputs=[
            wp.from_torch(global_landmarks),
            wp.from_torch(global_rotations),
            wp.from_torch(global_directions),
            wp.from_torch(global_joint_q),
        ],
        device="cpu",
    )

    torch.testing.assert_close(global_landmarks[0, :, 0], torch.tensor((0.0, 3.0)))
    torch.testing.assert_close(global_rotations, source_rotations[:, torch.tensor((0, 3))])
    torch.testing.assert_close(global_directions[0, :, 0], torch.tensor((10.0, 13.0)))
    torch.testing.assert_close(global_joint_q[:, 0], torch.tensor((0.0, 3.0)))

    solved_joint_q = source_joint_q.clone()
    solved_joint_q[3, 7] = 0.25
    solved_joint_q[0, 0] = 100.0
    solved_joint_q[3, 0] = 300.0
    local_landmarks = torch.empty((1, 2, 3), dtype=torch.float32)
    local_rotations = torch.empty((2, 2, 4), dtype=torch.float32)
    local_directions = torch.empty((1, 2, 3), dtype=torch.float32)
    local_reference_joint_q = torch.empty((2, 9), dtype=torch.float32)
    local_candidate_joint_q = torch.empty((4, 9), dtype=torch.float32)
    frame_indices = torch.empty(2, dtype=torch.int32)
    frame_active = torch.empty(2, dtype=torch.int32)
    wp.launch(
        _motion_frame_seed_local_gather,
        dim=2,
        inputs=[
            wp.from_torch(source_landmarks),
            wp.from_torch(source_rotations),
            wp.from_torch(source_directions),
            wp.from_torch(solved_joint_q),
            wp.from_torch(clip_offsets),
            wp.from_torch(source_joint_q),
            1,
            2,
            1,
            2,
            1,
            9,
        ],
        outputs=[
            wp.from_torch(local_landmarks),
            wp.from_torch(local_rotations),
            wp.from_torch(local_directions),
            wp.from_torch(local_reference_joint_q),
            wp.from_torch(local_candidate_joint_q),
            wp.from_torch(frame_indices),
            wp.from_torch(frame_active),
        ],
        device="cpu",
    )

    torch.testing.assert_close(frame_indices, torch.tensor((1, 4), dtype=torch.int32))
    torch.testing.assert_close(frame_active, torch.ones_like(frame_active))
    torch.testing.assert_close(local_landmarks[0, :, 0], torch.tensor((1.0, 4.0)))
    torch.testing.assert_close(local_rotations, source_rotations[:, torch.tensor((1, 4))])
    torch.testing.assert_close(local_directions[0, :, 0], torch.tensor((11.0, 14.0)))
    torch.testing.assert_close(local_reference_joint_q, source_joint_q[torch.tensor((1, 4))])
    expected_candidates = torch.stack((solved_joint_q[0], source_joint_q[1], solved_joint_q[3], source_joint_q[4]))
    torch.testing.assert_close(local_candidate_joint_q, expected_candidates)

    wp.launch(
        _motion_frame_seed_local_gather,
        dim=2,
        inputs=[
            wp.from_torch(source_landmarks),
            wp.from_torch(source_rotations),
            wp.from_torch(source_directions),
            wp.from_torch(solved_joint_q),
            wp.from_torch(clip_offsets),
            wp.from_torch(source_joint_q),
            2,
            2,
            1,
            2,
            1,
            9,
        ],
        outputs=[
            wp.from_torch(local_landmarks),
            wp.from_torch(local_rotations),
            wp.from_torch(local_directions),
            wp.from_torch(local_reference_joint_q),
            wp.from_torch(local_candidate_joint_q),
            wp.from_torch(frame_indices),
            wp.from_torch(frame_active),
        ],
        device="cpu",
    )
    torch.testing.assert_close(frame_indices, torch.tensor((2, 4), dtype=torch.int32))
    torch.testing.assert_close(frame_active, torch.tensor((1, 0), dtype=torch.int32))
    torch.testing.assert_close(local_reference_joint_q, source_joint_q[torch.tensor((2, 4))])
    expected_candidates = torch.stack((solved_joint_q[1], source_joint_q[2], solved_joint_q[4], source_joint_q[4]))
    torch.testing.assert_close(local_candidate_joint_q, expected_candidates)


def test_local_frame_seed_selection_prefers_the_lowest_cost_hard_feasible_branch() -> None:
    """Branch selection is exact, deterministic, and falls back only when both candidates are invalid."""
    clip_count = 7
    coordinate_count = 8
    baseline_joint_q = torch.zeros((clip_count, coordinate_count), dtype=torch.float32)
    baseline_joint_q[:, 6] = 1.0
    baseline_joint_q[:, 7] = torch.tensor((-0.10, -0.20, -0.30, -0.40, -0.50, -0.60, -0.70))
    candidates = baseline_joint_q.repeat_interleave(2, dim=0)
    candidates[:, 7] = torch.tensor(
        (0.10, 0.20, 0.11, 0.21, 0.12, 0.22, float("nan"), 0.23, 0.14, 0.24, 2.00, 0.25, 0.16, 0.26)
    )
    candidate_cost = torch.tensor(
        (1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 0.0, 5.0, float("nan"), float("inf"), 0.0, 10.0, 1.0, 0.0)
    )
    frame_indices = torch.arange(clip_count, dtype=torch.int32)
    frame_active = torch.tensor((1, 1, 1, 1, 1, 1, 0), dtype=torch.int32)
    coordinate_indices = torch.tensor((7,), dtype=torch.int64)
    coordinate_lower = torch.tensor((-1.0,), dtype=torch.float32)
    coordinate_upper = torch.tensor((1.0,), dtype=torch.float32)
    joint_q = torch.full_like(baseline_joint_q, -9.0)

    wp.launch(
        _motion_frame_seed_local_scatter,
        dim=clip_count,
        inputs=[
            wp.from_torch(candidates),
            wp.from_torch(candidate_cost),
            wp.from_torch(baseline_joint_q),
            wp.from_torch(frame_indices),
            wp.from_torch(frame_active),
            wp.from_torch(coordinate_indices),
            wp.from_torch(coordinate_lower),
            wp.from_torch(coordinate_upper),
            clip_count,
            coordinate_count,
            1,
            wp.uint8(1),
        ],
        outputs=[wp.from_torch(joint_q)],
        device="cpu",
    )

    torch.testing.assert_close(joint_q[:6, 6], torch.ones(6))
    torch.testing.assert_close(
        joint_q[:6, 7],
        torch.tensor((0.10, 0.21, 0.12, 0.23, -0.50, 0.25)),
    )
    torch.testing.assert_close(joint_q[6], torch.full((coordinate_count,), -9.0))


class _LinearOptimizer:
    """Small exact optimizer contract used to exercise trajectory projection."""

    def __init__(self, target: torch.Tensor) -> None:
        self.device = wp.get_device(str(target.device))
        self.n_batch = target.shape[0]
        self.n_residuals = 1
        self.n_dofs = 1
        self.n_coords = 1
        self.target = target.contiguous()
        self.jacobian = torch.ones((self.n_batch, 1, 1), dtype=torch.float32, device=target.device)
        self.residuals = torch.empty_like(target)
        self._wp_jacobian = wp.from_torch(self.jacobian)
        self._wp_residuals = wp.from_torch(self.residuals)

    def compute_residuals(self, joint_q, residuals=None):
        active = joint_q.shape[0]
        self.residuals[:active].copy_(wp.to_torch(joint_q)).sub_(self.target[:active])
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
            return residuals
        return self._wp_residuals[:active]

    def linearize(self, joint_q, residuals=None, jacobian=None):
        active = joint_q.shape[0]
        self.residuals[:active].copy_(wp.to_torch(joint_q)).sub_(self.target[:active])
        residual_view = self._wp_residuals[:active] if residuals is None else residuals
        jacobian_view = self._wp_jacobian[:active] if jacobian is None else jacobian
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
        if jacobian is not None:
            wp.copy(jacobian, self._wp_jacobian[:active])
        return residual_view, jacobian_view

    def integrate(self, joint_q, delta, joint_q_out, *, step_size=1.0):
        output = wp.to_torch(joint_q_out)
        output.copy_(wp.to_torch(joint_q))
        output.add_(wp.to_torch(delta), alpha=step_size)


def test_velocity_projection_bounds_the_same_segmented_backward_edges() -> None:
    """Projection constrains physical within-clip edges and ignores the flat-corpus boundary jump."""
    offsets = torch.tensor((0, 3, 6), dtype=torch.int32)
    step_seconds = torch.tensor((1.0, 0.25), dtype=torch.float32)
    target = torch.tensor(((0.0,), (8.0,), (-8.0,), (100.0,), (-100.0,), (100.0,)))
    optimizer = _LinearOptimizer(target)
    solver = IKTrajectorySolver(
        optimizer,
        max_segments=2,
        max_equality_residuals_per_frame=3,
        damping=1.0e-5,
        krylov_max_iterations=32,
        krylov_relative_tolerance=1.0e-4,
    )
    output = torch.empty_like(target)
    solver.solve(
        torch.zeros_like(target),
        output,
        offsets,
        step_seconds,
        torch.ones(1),
        torch.zeros((3, 1)),
        coordinate_bounds=IKTrajectorySolver.CoordinateBounds(
            coordinate_indices=torch.empty(0, dtype=torch.int32),
            dof_indices=torch.empty(0, dtype=torch.int32),
            lower=torch.empty(0),
            upper=torch.empty(0),
        ),
        joint_velocity=torch.zeros_like(target),
        velocity_lower=torch.tensor((-0.5,)),
        velocity_upper=torch.tensor((0.5,)),
        segment_active=torch.ones(2, dtype=torch.int32),
        segment_direction_valid=torch.empty(2, dtype=torch.bool),
        segment_globalization_succeeded=torch.empty(2, dtype=torch.bool),
    )

    projected_velocity = time_backward_difference_segmented(output, offsets.to(torch.int64), step_seconds)
    torch.testing.assert_close(projected_velocity[[0, 3]], projected_velocity[[1, 4]])
    assert torch.all(projected_velocity >= -0.5 - 1.0e-6)
    assert torch.all(projected_velocity <= 0.5 + 1.0e-6)
    assert abs(float(output[3] - output[2])) > 0.5 * float(step_seconds[1])


def _target_and_coordinates(
    robot: str, device: str | torch.device = "cpu"
) -> tuple[_G1FrameTarget | _SmplFrameTarget, torch.Tensor]:
    """Construct the minimum target-owned velocity state without simulator mechanics."""
    device = torch.device(device)
    body_com = wp.array([(0.0, 0.0, 0.0)], dtype=wp.vec3, device=str(device))
    if robot == "g1":
        target = object.__new__(_G1FrameTarget)
        reference = SimpleNamespace(
            model=SimpleNamespace(joint_coord_count=36, joint_dof_count=35, body_count=1, body_com=body_com)
        )
        object.__setattr__(target, "kinematics", reference)
        object.__setattr__(target, "kinematic_tree", SimpleNamespace(num_coordinates=29, root_body_index=0))
        joint_q = torch.zeros((6, 36), dtype=torch.float32, device=device)
    else:
        target = object.__new__(_SmplFrameTarget)
        reference = SimpleNamespace(
            model=SimpleNamespace(joint_coord_count=76, joint_dof_count=75, body_count=1, body_com=body_com)
        )
        object.__setattr__(target, "kinematics", reference)
        object.__setattr__(target, "reference_coordinate_names", tuple(f"joint_{index}" for index in range(69)))
        object.__setattr__(target, "_coordinate_axes", torch.eye(3, device=device).repeat(23, 1, 1))
        object.__setattr__(target, "_kinematic_tree", SimpleNamespace(num_coordinates=69, root_body_index=0))
        joint_q = torch.zeros((6, 76), dtype=torch.float32, device=device)
    joint_q[:, 6] = 1.0
    joint_q[:, 7] = joint_q.new_tensor((0.0, 0.2, 0.5, -0.3, -0.2, 0.0))
    return target, joint_q


def test_smpl_canonical_velocity_uses_ordered_d6_edges_not_raw_euler_differences() -> None:
    """Coupled SMPL coordinates use ordered-D6 physical edge rates."""
    target, joint_q = _target_and_coordinates("smpl")
    joint_q[:, 7:10] = torch.tensor(
        (
            (0.0, 0.0, 0.0),
            (0.6, 0.8, 1.0),
            (1.0, 1.2, 1.4),
            (-0.4, 0.2, 0.7),
            (0.2, 0.9, 1.1),
            (0.7, 1.1, 1.5),
        )
    )
    offsets_i32 = torch.tensor((0, 3, 6), dtype=torch.int32)
    offsets_i64 = offsets_i32.to(torch.int64)
    step_seconds = torch.tensor((0.5, 0.25), dtype=torch.float32)
    canonical = torch.zeros((6, 75), dtype=torch.float32)

    target.write_nonroot_velocity_canonical(joint_q, offsets_i32, step_seconds, canonical)

    coordinates = joint_q[:, 7:].view(6, 23, 3)
    rotations = ordered_hinge_rotation(coordinates, target._coordinate_axes)
    angular_velocity = time_quaternion_angular_velocity_segmented(rotations, offsets_i64, step_seconds)
    canonical_edges = ordered_hinge_coordinate_velocity(coordinates, target._coordinate_axes, angular_velocity).flatten(
        1
    )
    canonical_edges[2].copy_(canonical_edges[1])
    canonical_edges[5].copy_(canonical_edges[4])
    expected = torch.empty_like(canonical_edges)
    expected[0].copy_(canonical_edges[0])
    expected[1:3].copy_(canonical_edges[:2])
    expected[3].copy_(canonical_edges[3])
    expected[4:6].copy_(canonical_edges[3:5])
    torch.testing.assert_close(canonical[:, 6:], expected, atol=3.0e-6, rtol=3.0e-6)

    raw = time_backward_difference_segmented(joint_q[:, 7:], offsets_i64, step_seconds)
    assert torch.max(torch.abs(canonical[:, 6:] - raw)) > 0.05


@pytest.mark.parametrize("robot", ("g1", "smpl"))
def test_target_canonical_velocity_has_cpu_cuda_parity_and_captures(robot: str) -> None:
    """Robot-owned canonical derivative kernels preserve results under native Warp capture."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")
    cpu_target, cpu_joint_q = _target_and_coordinates(robot)
    cuda_target, cuda_joint_q = _target_and_coordinates(robot, "cuda:0")
    cpu_offsets = torch.tensor((0, 3, 6), dtype=torch.int32)
    cpu_steps = torch.tensor((0.5, 0.25), dtype=torch.float32)
    cuda_offsets = cpu_offsets.cuda()
    cuda_steps = cpu_steps.cuda()
    cpu_output = torch.zeros((6, cpu_joint_q.shape[1] - 1), dtype=torch.float32)
    cuda_output = torch.zeros_like(cpu_output, device="cuda:0")

    write_velocity_canonical(cpu_target, cpu_joint_q, cpu_offsets, cpu_steps, cpu_output)
    write_velocity_canonical(cuda_target, cuda_joint_q, cuda_offsets, cuda_steps, cuda_output)
    wp.synchronize_device("cuda:0")
    torch.testing.assert_close(cuda_output.cpu(), cpu_output, atol=3.0e-6, rtol=3.0e-6)

    cuda_output.fill_(float("nan"))
    cuda_target.write_nonroot_velocity_canonical(cuda_joint_q, cuda_offsets, cuda_steps, cuda_output)
    wp.synchronize_device("cuda:0")
    wp.capture_begin(device="cuda:0")
    cuda_target.write_nonroot_velocity_canonical(cuda_joint_q, cuda_offsets, cuda_steps, cuda_output)
    graph = wp.capture_end(device="cuda:0")
    wp.capture_launch(graph)
    wp.synchronize_device("cuda:0")
    assert torch.isfinite(cuda_output[:, 6:]).all()
