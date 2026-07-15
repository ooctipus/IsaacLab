# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused mathematical and ownership gates for shared motion retargeting."""

import ast
import inspect
import math
import textwrap
from pathlib import Path
from types import SimpleNamespace

import torch
import warp as wp

from isaaclab_tasks.core.multi_task.motion.mdp.commands import motion_trajectory as trajectory
from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionTrajectorySolveCfg

_SOURCE_EVIDENCE = {"source_landmark_position_m", "source_direction_point_position_m"}


def _function_tree(function) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef))


def _target_root(target: ast.expr) -> str | None:
    while isinstance(target, ast.Subscript):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def test_contact_preparation_cannot_mutate_source_evidence() -> None:
    """Interval targets may write support anchors but keep source evidence read-only."""
    for node in ast.walk(_function_tree(trajectory._motion_contact_interval_targets)):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            targets = (node.target,)
        else:
            continue
        assert _SOURCE_EVIDENCE.isdisjoint(_target_root(target) for target in targets)

    interval_source = inspect.getsource(trajectory._motion_contact_interval_targets)
    assert all(name in interval_source for name in _SOURCE_EVIDENCE)
    assert not hasattr(trajectory, "_motion_contact_two_bone_targets")
    assert not hasattr(trajectory, "_motion_contact_seed_objectives")


def test_target_ground_gauge_owns_the_complete_robot_vertical_contract() -> None:
    """Stable support selects one signed post-source gauge before contact preparation."""
    gauge_source = inspect.getsource(trajectory._motion_target_ground_gauge)
    for name in (
        "joint_q",
        "source_landmark_position_m",
        "source_direction_point_position_m",
    ):
        assert f"{name}[" in gauge_source and "+= shift" in gauge_source
    assert "target_support_position_m" not in gauge_source
    assert "source_contact_probe_position_m" not in gauge_source
    assert "collision_probe" not in gauge_source
    assert "shift = shift_sum / weight_sum" in gauge_source
    assert "source_channel_stable" in gauge_source

    solve_source = inspect.getsource(trajectory.motion_solve_trajectory)
    gauge_call = solve_source.index("            _motion_target_ground_gauge,")
    initial_copy = solve_source.index("targets.initial_joint_q[:frame_count].copy_(joint_q)")
    contact_prepare = solve_source.index("prepare_contact_targets(frame_count, clip_offsets, phase_active)")
    assert initial_copy < gauge_call < contact_prepare
    assert "source_channel_confidence" in gauge_source
    gauge_launch = solve_source[gauge_call:contact_prepare]
    assert "workspace.source_channel_confidence" in gauge_launch
    assert "workspace.source_channel_stable" in gauge_launch
    assert "targets.initial_joint_q[:frame_count, :7].copy_(joint_q[:, :7])" in gauge_launch


def test_target_ground_gauge_is_signed_clip_constant_and_stable_only() -> None:
    """The gauge is a signed clip translation fitted to stable planar patches, never swing collision."""
    wp.init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    frame_count = 3
    body_q = torch.zeros((frame_count, 1, 7), dtype=torch.float32, device=device)
    body_q[..., 4] = math.sqrt(0.5)
    body_q[..., 6] = math.sqrt(0.5)
    body_q[:, 0, 2] = torch.tensor((0.8, 1.0, 100.0), dtype=torch.float32, device=device)
    confidence = torch.ones((frame_count, 1), dtype=torch.float32, device=device)
    stable = torch.tensor(((1,), (1,), (0,)), dtype=torch.uint8, device=device)
    obstacle_pose = torch.zeros((frame_count, 7), dtype=torch.float32, device=device)
    body_indices = torch.zeros(1, dtype=torch.int64, device=device)
    support_points = torch.tensor(((0.2, 0.0, 0.0), (0.1, 0.0, 0.0)), dtype=torch.float32, device=device)
    support_slots = torch.zeros(2, dtype=torch.int64, device=device)
    clip_offsets = torch.tensor((0, frame_count), dtype=torch.int32, device=device)
    segment_active = torch.ones(1, dtype=torch.int32, device=device)
    joint_q = torch.zeros((frame_count, 7), dtype=torch.float32, device=device)
    source_positions = torch.zeros((1, frame_count, 3), dtype=torch.float32, device=device)
    source_points = torch.zeros_like(source_positions)

    wp.launch(
        trajectory._motion_target_ground_gauge,
        dim=1,
        inputs=[
            wp.from_torch(obstacle_pose),
            wp.from_torch(body_q),
            wp.from_torch(body_indices),
            wp.from_torch(support_points),
            wp.from_torch(support_slots),
            wp.from_torch(confidence),
            wp.from_torch(stable),
            wp.from_torch(clip_offsets),
            wp.from_torch(segment_active),
            1,
            1,
            1,
            1,
            2,
            0.0,
        ],
        outputs=[
            wp.from_torch(joint_q),
            wp.from_torch(source_positions),
            wp.from_torch(source_points),
        ],
        device=str(device),
    )
    wp.synchronize_device(str(device))

    expected = torch.full((frame_count,), -0.7)
    torch.testing.assert_close(joint_q[:, 2].cpu(), expected)
    for values in (source_positions, source_points):
        torch.testing.assert_close(values[0, :, 2].cpu(), expected)


def test_clearance_lift_uses_complete_owned_body_chain() -> None:
    """The deepest ankle-or-child probe owns lift while unrelated penetration is ignored."""
    wp.init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    body_q = torch.zeros((1, 3, 7), dtype=torch.float32, device=device)
    body_q[..., 6] = 1.0
    body_q[:, 0, 2] = 0.05
    body_q[:, 1, 2] = 0.02
    body_q[:, 2, 2] = -10.0
    obstacle_pose = torch.zeros((1, 7), dtype=torch.float32, device=device)
    obstacle_pose[:, 6] = 2.0
    probe_bodies = torch.tensor((0, 0, 1, 2), dtype=torch.int64, device=device)
    probe_offsets = torch.tensor(
        ((-0.1, 0.0, -0.01), (0.1, 0.0, -0.02), (0.0, 0.0, -0.04), (0.0, 0.0, -2.0)),
        dtype=torch.float32,
        device=device,
    )
    probe_normal_slots = torch.tensor((0, 0, 0, -1), dtype=torch.int64, device=device)
    lift = torch.empty((1, 1), dtype=torch.float32, device=device)

    wp.launch(
        trajectory._motion_clearance_lift,
        dim=(1, 1),
        inputs=[
            wp.from_torch(body_q),
            wp.from_torch(obstacle_pose),
            wp.from_torch(probe_bodies),
            wp.from_torch(probe_offsets),
            wp.from_torch(probe_normal_slots),
            1,
            1,
            4,
            -0.00198,
            0.0,
        ],
        outputs=[wp.from_torch(lift)],
        device=str(device),
    )
    wp.synchronize_device(str(device))

    torch.testing.assert_close(lift[:, 0].cpu(), torch.tensor((0.02,)), atol=1.0e-6, rtol=0.0)


def test_normal_ownership_forces_transition_and_any_clearance_refinement() -> None:
    """Transition owners and stable clearance lifts refine while rejected clips remain inactive."""
    wp.init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    normal_owned = torch.tensor(((0.0,), (1.0,), (1.0,), (1.0,), (1.0,)), dtype=torch.float32, device=device)
    stable = torch.tensor(((0,), (0,), (1,), (0,), (0,)), dtype=torch.uint8, device=device)
    clearance_lift = torch.tensor(((0.0,), (0.0,), (0.02,), (0.0,), (0.03,)), device=device)
    clip_offsets = torch.arange(6, dtype=torch.int32, device=device)
    active = torch.tensor((1, 1, 1, 1, 0), dtype=torch.uint8, device=device)
    required = torch.empty_like(active)

    wp.launch(
        trajectory._motion_normal_ownership_segments,
        dim=5,
        inputs=[
            wp.from_torch(normal_owned),
            wp.from_torch(stable),
            wp.from_torch(clearance_lift),
            wp.from_torch(clip_offsets),
            wp.from_torch(active),
            5,
            1,
        ],
        outputs=[wp.from_torch(required)],
        device=str(device),
    )
    wp.synchronize_device(str(device))

    torch.testing.assert_close(required.cpu(), torch.tensor((0, 1, 1, 1, 0), dtype=torch.uint8))


def test_source_plane_selection_is_wired_to_contact_confidence() -> None:
    """Absolute normal ownership stays separate from contact tangent and direction semantics."""
    solve_source = inspect.getsource(trajectory.motion_solve_trajectory)
    build_start = solve_source.index("    def build_system(")
    build_stop = solve_source.index("    residual_layout =", build_start)
    build_source = solve_source[build_start:build_stop]
    assert "factory(term_cfg, targets, contact_normal_owned, contact_confidence)" in build_source
    assert "_IKObjectiveSourceFidelityGuard(" in build_source
    assert "                contact_normal_owned," in build_source
    assert "                contact_confidence," in build_source
    global_source = inspect.getsource(trajectory.motion_objective_source_global_position)
    assert "position_normal_channel_slots" in global_source
    assert "_IKObjectiveSourcePhasePosition(" in global_source

    assert "contact_stable" not in build_source
    guard_residual = inspect.getsource(trajectory._source_fidelity_guard_residuals)
    guard_jacobian = inspect.getsource(trajectory._source_fidelity_guard_jacobian)
    assert "position_normal_channel_slots" in guard_residual
    assert "error = wp.vec3(error[0], error[1], 0.0)" in guard_residual
    assert "source_channel_normal_owned[target_frame, contact_channel]" in guard_residual
    assert "source_channel_confidence[target_frame, contact_channel]" in guard_residual
    assert "direction_contact_channel_slots" in guard_residual
    assert "source_channel_confidence[target_frame, required]" not in guard_residual
    assert "position_normal_channel_slots" in guard_jacobian
    assert "velocity = wp.vec3(velocity[0], velocity[1], 0.0)" in guard_jacobian
    assert not hasattr(trajectory, "_source_position_contact_channel")

    assert "source_channel_normal_owned[frame, contact_channel]" in guard_jacobian
    assert "source_channel_confidence[frame, contact_channel]" in guard_jacobian
    assert "direction_contact_channel_slots" in guard_jacobian
    assert "source_channel_confidence[frame, required]" not in guard_jacobian
    quality_source = inspect.getsource(trajectory._trajectory_clip_quality)
    frame_start = quality_source.index("        _trajectory_quality_frames,")
    frame_stop = quality_source.index("        _trajectory_quality_clips,", frame_start)
    frame_launch = quality_source[frame_start:frame_stop]
    assert "source_channel_confidence" in frame_launch
    assert "source_channel_normal_owned" in frame_launch
    assert "position_normal_channel_slots" in frame_launch
    assert "contact_direction_row_tensor" in frame_launch
    assert "direction_contact_channel_slots" in frame_launch
    assert "required_direction_row_tensor" in frame_launch
    assert "contact_distal_point_body_m" in frame_launch
    assert "source_channel_stable" not in frame_launch


def test_contact_activity_separates_vertex_confidence_from_end_indexed_edges() -> None:
    """Per-frame geometry uses confidence squared while no-slip uses binary stable edges."""
    wp.init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stable = torch.tensor(((0,), (0,), (1,), (0,)), dtype=torch.uint8, device=device)
    edge_stable = torch.tensor(((0,), (0,), (1,), (1,)), dtype=torch.uint8, device=device)
    confidence = torch.tensor(((0.0,), (0.25,), (1.0,), (0.0,)), dtype=torch.float32, device=device)
    activity = torch.empty((4, 2), dtype=torch.float32, device=device)
    clearance_lift = torch.tensor(((0.0,), (0.0,), (0.0,), (0.05,)), dtype=torch.float32, device=device)
    normal_owned = torch.empty_like(confidence)

    wp.launch(
        trajectory._motion_contact_activity,
        dim=(4, 1),
        inputs=[
            wp.from_torch(stable),
            wp.from_torch(edge_stable),
            wp.from_torch(confidence),
            wp.from_torch(clearance_lift),
            4,
            1,
        ],
        outputs=[wp.from_torch(normal_owned), wp.from_torch(activity)],
        device=str(device),
    )
    wp.synchronize_device(str(device))

    torch.testing.assert_close(normal_owned[:, 0].cpu(), torch.tensor((0.0, 1.0, 1.0, 1.0)))
    expected_activity = torch.tensor(((0.0, 0.0), (0.0625, 0.0), (1.0, 1.0), (1.0, 1.0)))
    torch.testing.assert_close(activity.cpu(), expected_activity)

    source = inspect.getsource(trajectory._motion_contact_activity)
    assert "confidence * confidence" in source
    assert "channel_clearance_lift_m" in source and "vertex_activity = 1.0" in source
    assert "channel_edge_stable" in source


def test_interval_targets_become_immutable_before_relative_contact_solve() -> None:
    """The prepared interval target is fixed while the solver minimizes residual first differences."""
    source = inspect.getsource(trajectory.motion_solve_trajectory)
    prepare_definition = source.index("    def prepare_contact_targets(")
    solve_batches = source.index("    for clip_start, clip_stop in zip(", prepare_definition)
    preparation = source[prepare_definition:solve_batches]
    assert preparation.count("outputs=[wp.from_torch(targets.target_support_position_m)]") == 2
    assert "_trajectory_support_points" in preparation
    assert "_motion_contact_interval_targets" in preparation
    assert "two_bone" not in preparation
    assert "contact_seed" not in preparation

    prepare_call = source.index("prepare_contact_targets(frame_count, clip_offsets, phase_active)", solve_batches)
    contact_solve = source.index("solve_phase(", prepare_call)
    assert "target_support_position_m" not in source[prepare_call:contact_solve]
    assert "target_support_position_m" not in source[contact_solve : source.index("batch_coordinates =", contact_solve)]
    assert not hasattr(trajectory, "_motion_contact_two_bone_targets")


def test_monolithic_weights_keep_declared_terms_without_acceptance_guards_or_phase_trust() -> None:
    """One objective retains declared source/physical/contact terms but no transactional rows."""
    layout = trajectory._MotionTrajectoryResidualLayout(
        source_global_position=slice(0, 2),
        source_rotation=slice(2, 4),
        source_direction_point=slice(4, 6),
        source_fidelity_guard=slice(6, 10),
        contact=slice(10, 20),
        activity_group_by_residual=torch.full((28,), -1, dtype=torch.int32),
        first_difference_group_by_residual=torch.full((28,), -1, dtype=torch.int32),
        joint_default=slice(20, 22),
        joint_reference=slice(22, 24),
        collision_objective=slice(24, 26),
        nonpenetration_objective=slice(26, 28),
    )
    cfg = MotionTrajectorySolveCfg()
    targets = SimpleNamespace(required_position_rows=(0,), required_direction_rows=(0,), support_patch_offsets=(0, 2))
    base = torch.empty(layout.residual_count)
    temporal = torch.empty((3, layout.residual_count))

    trajectory._motion_monolithic_weights(layout, cfg, targets, base, temporal)

    for rows in (layout.source_global_position, layout.source_rotation, layout.source_direction_point):
        assert torch.all(base[rows] == 1.0)
    assert torch.all(base[layout.joint_default] == cfg.joint_default_position_weight)
    assert torch.all(base[layout.collision_objective] == 1.0)
    torch.testing.assert_close(base[layout.contact], torch.tensor((1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)))
    assert torch.count_nonzero(base[layout.source_fidelity_guard]) == 0
    assert torch.count_nonzero(base[layout.joint_reference]) == 0
    assert torch.count_nonzero(temporal[:, layout.joint_reference]) == 0
    assert torch.all(base[layout.nonpenetration_objective] == 1.0)
    expected_relative_precision = (cfg.contact.point_tolerance_m / cfg.acceptance.contact.slip_speed_upper_mps) ** 2
    assert torch.all(temporal[0, 14:20] == expected_relative_precision)
    assert torch.count_nonzero(temporal[1:, layout.contact]) == 0


def test_source_fidelity_inequalities_reserve_two_kkt_tolerances() -> None:
    """Hard source bounds stay strictly inside publication limits after one relative KKT residual."""
    policy = MotionTrajectorySolveCfg.AcceptanceCfg.SourceCfg()
    tolerance = 1.0e-4
    layout = SimpleNamespace(source_fidelity_guard=slice(7, 11))
    targets = SimpleNamespace(
        required_position_rows=(0,),
        required_direction_rows=(0,),
        source_landmark_position_m=torch.empty((1, 1, 3)),
    )
    inequalities = trajectory._motion_source_fidelity_inequalities(layout, targets, policy, tolerance)
    publication_upper = torch.tensor(
        (
            policy.required_position_upper_m,
            policy.required_distal_position_upper_m,
            2.0 * math.sin(0.5 * policy.required_distal_direction_upper_rad),
            2.0 * math.sin(0.5 * policy.root_rotation_upper_rad),
        )
    )
    interior_scale = 1.0 / (1.0 + 2.0 * tolerance)
    torch.testing.assert_close(inequalities.residual_indices, torch.arange(7, 11, dtype=torch.int32))
    torch.testing.assert_close(inequalities.upper, publication_upper * interior_scale)
    assert torch.all(inequalities.upper * (1.0 + tolerance) < publication_upper)


def test_contact_layout_separates_vertex_and_end_indexed_edge_ownership() -> None:
    """Contact rows map vertex confidence and planted-point first differences independently."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    targets = SimpleNamespace(
        position_body_indices=(0, 1),
        rotation_body_indices=(0,),
        direction_body_indices=(1,),
        support_patch_offsets=(0, 3),
        coordinate_indices=torch.arange(2, dtype=torch.int64, device=device),
        required_position_rows=(0,),
        required_direction_rows=(0,),
        source_landmark_position_m=torch.zeros((2, 2, 3), dtype=torch.float32, device=device),
    )
    layout = trajectory._motion_trajectory_residual_layout(targets, 2, MotionTrajectorySolveCfg().objectives)
    groups = layout.activity_group_by_residual.cpu()
    edge_groups = layout.first_difference_group_by_residual.cpu()
    torch.testing.assert_close(groups[layout.contact], torch.zeros_like(groups[layout.contact]))
    torch.testing.assert_close(
        edge_groups[layout.contact.start : layout.contact.start + 4], -torch.ones(4, dtype=torch.int32)
    )
    torch.testing.assert_close(
        edge_groups[layout.contact.start + 4 : layout.contact.stop], torch.ones(9, dtype=torch.int32)
    )
    outside_contact = torch.cat((edge_groups[: layout.contact.start], edge_groups[layout.contact.stop :]))
    torch.testing.assert_close(outside_contact, -torch.ones_like(outside_contact))

    for forbidden in (
        "contact_equality_residual_starts",
        "contact_seed_precision",
        "contact_seed_position",
        "contact_seed_rotation",
    ):
        assert not hasattr(layout, forbidden)
    assert not hasattr(trajectory, "_motion_contact_seed_objectives")


def test_monolithic_solve_has_no_acceptance_transaction_or_constrained_velocity_path() -> None:
    """The production call is one projected solve whose quality is reported only afterward."""
    source = inspect.getsource(trajectory.motion_solve_trajectory)
    weights = source.index("_motion_monolithic_weights(")
    solve = source.index("solve_phase(", weights)
    finish = source.index("_motion_phase_finish(", solve)
    quality = source.index("_trajectory_recompute_clip_quality(", finish)
    call = source[solve:finish]

    assert source.count("solve_phase(") == 2
    assert source.count("_motion_monolithic_weights(") == 1
    assert "source_inequalities" not in source
    assert "inequalities=None" in call
    assert "velocity_bounds=(workspace.source_velocity_lower, workspace.source_velocity_upper)" in call
    assert "residual_activity=residual_activity" in call
    assert "terminal_accepted=" not in call
    assert "terminal_acceptance=" not in call
    assert "adaptive_recovery=" not in call
    assert "_motion_phase_copy_selected(" not in source
    assert "_motion_source_fidelity_accepted(" not in source[weights:quality]
    assert "_motion_contact_rows_accepted(" not in source[weights:quality]
    assert {"source_velocity_lower", "source_velocity_upper"} <= (
        trajectory._MotionTrajectoryWorkspace.__annotations__.keys()
    )


def test_terminal_acceptance_scope_separates_source_from_contact() -> None:
    """Source fitting never waits for contact quality that only the contact phase owns."""
    wp.init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_count = 3
    quality = torch.zeros((clip_count, trajectory._TRAJECTORY_METRIC_COUNT), dtype=torch.float32, device=device)
    quality[:, trajectory._METRIC_CONTACT_APPLICABLE] = 1.0
    quality[:, trajectory._METRIC_CONTACT_STABLE_COUNT] = 1.0
    quality[:, trajectory._METRIC_SOURCE_CONTACT_CONFIDENCE] = 0.5
    quality[0, trajectory._METRIC_CONTACT_GAP] = 1.0
    quality[1, trajectory._METRIC_SOURCE_REQUIRED_POSITION] = 1.0
    flags = torch.ones(clip_count, dtype=torch.bool, device=device)

    def accepted_for(mode: int) -> torch.Tensor:
        accepted = torch.zeros(clip_count, dtype=torch.bool, device=device)
        active = torch.ones(clip_count, dtype=torch.int32, device=device)
        wp.launch(
            trajectory._motion_terminal_acceptance_update,
            dim=clip_count,
            inputs=[
                wp.from_torch(quality),
                wp.from_torch(flags),
                wp.from_torch(flags),
                wp.from_torch(flags),
                wp.uint8(mode),
                clip_count,
                trajectory._METRIC_SOURCE_REQUIRED_POSITION,
                trajectory._METRIC_SOURCE_REQUIRED_DISTAL_POSITION,
                trajectory._METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION,
                trajectory._METRIC_SOURCE_ROOT_ROTATION,
                trajectory._METRIC_CONTACT_GAP,
                trajectory._METRIC_CONTACT_TILT,
                trajectory._METRIC_CONTACT_SLIP_SPEED,
                trajectory._METRIC_CONTACT_CUMULATIVE_DRIFT,
                trajectory._METRIC_CONTACT_APPLICABLE,
                trajectory._METRIC_CONTACT_STABLE_COUNT,
                trajectory._METRIC_SOURCE_CONTACT_CONFIDENCE,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
            ],
            outputs=[wp.from_torch(accepted), wp.from_torch(active)],
            device=str(device),
        )
        wp.synchronize_device(str(device))
        torch.testing.assert_close(active.cpu(), (~accepted).to(torch.int32).cpu())
        return accepted.cpu()

    torch.testing.assert_close(accepted_for(trajectory._TERMINAL_ACCEPT_SOURCE), torch.tensor((True, False, True)))
    torch.testing.assert_close(
        accepted_for(trajectory._TERMINAL_ACCEPT_SOURCE_CONTACT), torch.tensor((False, False, True))
    )
    torch.testing.assert_close(accepted_for(trajectory._TERMINAL_ACCEPT_CONSTRAINTS), torch.tensor((True, True, True)))


def test_monolithic_solve_keeps_final_iterate_without_rollback() -> None:
    """The sole joint solve is followed by evidence, not a certified-state transaction."""
    source = inspect.getsource(trajectory.motion_solve_trajectory)
    weights = source.index("_motion_monolithic_weights(")
    solve = source.index("solve_phase(", weights)
    finish = source.index("_motion_phase_finish(", solve)
    quality = source.index("_trajectory_recompute_clip_quality(", finish)

    assert weights < solve < finish < quality
    assert "_motion_phase_copy_selected(" not in source
    assert "outputs=[wp.from_torch(workspace.certified_joint_q)]" not in source
    assert "outputs=[wp.from_torch(workspace.joint_q)]" not in source[finish:quality]


def test_shared_retarget_math_has_no_dataset_robot_route_branch() -> None:
    """Shared projection and trajectory math cannot branch on a LAFAN/G1 product route."""
    motion_root = Path(trajectory.__file__).parents[2]
    shared = (motion_root / "retarget.py").read_text(encoding="utf-8")
    shared += Path(trajectory.__file__).read_text(encoding="utf-8")

    for token in ("lafan", "cmu", "g1", "smpl"):
        assert token not in shared.lower()


def _distal_quality(
    actual_point: tuple[float, float, float],
    source_point: tuple[float, float, float],
    confidence: float,
    normal_owned: float | None = None,
    contact_point: tuple[float, float, float] = (1.0, 0.0, 0.0),
    actual_base: tuple[float, float, float] = (0.0, 0.0, 0.0),
    source_base: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    wp.init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    body_q = torch.zeros((1, 2, 7), dtype=torch.float32, device=device)
    body_q[..., 6] = 1.0
    body_q[0, 0, :3] = torch.tensor(actual_base, dtype=torch.float32, device=device)
    body_q[0, 1, :3] = torch.tensor(actual_point, dtype=torch.float32, device=device)
    source_positions = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)
    source_positions[0, 0] = torch.tensor(source_base, dtype=torch.float32, device=device)
    source_rotations = torch.zeros((1, 1, 4), dtype=torch.float32, device=device)
    source_rotations[..., 3] = 1.0
    source_points = torch.tensor(source_point, dtype=torch.float32, device=device).reshape(1, 1, 3)
    source_confidence = torch.tensor(((confidence,),), dtype=torch.float32, device=device)
    ownership = float(confidence > 0.0) if normal_owned is None else normal_owned
    source_normal_owned = torch.tensor(((ownership,),), dtype=torch.float32, device=device)
    contact_point_body = torch.tensor(contact_point, dtype=torch.float32, device=device).reshape(1, 3)
    quality = torch.empty((1, trajectory._TRAJECTORY_METRIC_COUNT), dtype=torch.float32, device=device)
    wp.launch(
        trajectory._trajectory_quality_frames,
        dim=1,
        inputs=[
            wp.from_torch(body_q),
            wp.from_torch(source_positions),
            wp.from_torch(torch.tensor((0,), dtype=torch.int64, device=device)),
            wp.from_torch(torch.tensor((0,), dtype=torch.int64, device=device)),
            wp.from_torch(torch.tensor((0,), dtype=torch.int64, device=device)),
            wp.from_torch(torch.tensor((0,), dtype=torch.int64, device=device)),
            wp.from_torch(source_rotations),
            wp.from_torch(torch.tensor((0,), dtype=torch.int64, device=device)),
            wp.from_torch(source_points),
            wp.from_torch(torch.tensor((1,), dtype=torch.int64, device=device)),
            wp.from_torch(torch.tensor((0,), dtype=torch.int64, device=device)),
            wp.from_torch(torch.tensor((0,), dtype=torch.int64, device=device)),
            wp.from_torch(torch.tensor((0,), dtype=torch.int64, device=device)),
            wp.from_torch(torch.tensor((0,), dtype=torch.int64, device=device)),
            wp.from_torch(source_confidence, dtype=wp.float32),
            wp.from_torch(source_normal_owned, dtype=wp.float32),
            wp.from_torch(torch.zeros((1, 3), dtype=torch.float32, device=device)),
            wp.from_torch(contact_point_body),
            1,
            1,
            1,
            1,
            1,
            1,
            trajectory._TRAJECTORY_METRIC_COUNT,
            trajectory._METRIC_SOURCE_REQUIRED_POSITION,
            trajectory._METRIC_SOURCE_REQUIRED_DISTAL_POSITION,
            trajectory._METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION,
            trajectory._METRIC_SOURCE_ROOT_ROTATION,
            trajectory._METRIC_SOURCE_ALL_POSITION,
            trajectory._METRIC_SOURCE_ALL_DISTAL_POSITION,
            trajectory._METRIC_SOURCE_ALL_LANDMARK_DIRECTION,
            trajectory._METRIC_SOURCE_ALL_DISTAL_DIRECTION,
            trajectory._METRIC_SOURCE_NONROOT_ROTATION,
        ],
        outputs=[wp.from_torch(quality)],
        device=str(device),
    )
    wp.synchronize_device(str(device))
    return quality[0].cpu()


def test_contact_active_base_quality_releases_height_without_hiding_full_3d_diagnostics() -> None:
    """Contact owns base height while required planar and raw full-3D source metrics remain distinct."""
    arguments = dict(
        actual_point=(1.0, 0.0, 1.0),
        source_point=(1.0, 0.0, 0.0),
        actual_base=(0.0, 0.0, 1.0),
        source_base=(0.0, 0.0, 0.0),
    )
    contact_active = _distal_quality(**arguments, confidence=0.25)
    swing = _distal_quality(**arguments, confidence=0.0)

    assert contact_active[trajectory._METRIC_SOURCE_REQUIRED_POSITION] == 0.0
    torch.testing.assert_close(contact_active[trajectory._METRIC_SOURCE_ALL_POSITION], torch.tensor(1.0))
    torch.testing.assert_close(swing[trajectory._METRIC_SOURCE_REQUIRED_POSITION], torch.tensor(1.0))


def test_contact_active_distal_quality_uses_support_plane_without_hiding_full_3d_diagnostics() -> None:
    """Confidence transitions use XY while all-row diagnostics and true swing retain full 3-D geometry."""
    contact_active = _distal_quality((1.0, 0.0, 1.0), (1.0, 0.0, 0.0), 0.25)
    swing = _distal_quality((1.0, 0.0, 1.0), (1.0, 0.0, 0.0), 0.0)

    assert contact_active[trajectory._METRIC_SOURCE_REQUIRED_DISTAL_POSITION] == 0.0
    assert contact_active[trajectory._METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION] == 0.0
    torch.testing.assert_close(contact_active[trajectory._METRIC_SOURCE_ALL_DISTAL_POSITION], torch.tensor(1.0))
    torch.testing.assert_close(
        contact_active[trajectory._METRIC_SOURCE_ALL_DISTAL_DIRECTION],
        torch.tensor(math.pi / 4),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(swing[trajectory._METRIC_SOURCE_REQUIRED_DISTAL_POSITION], torch.tensor(1.0))
    torch.testing.assert_close(
        swing[trajectory._METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION], torch.tensor(math.pi / 4), atol=1.0e-6, rtol=0.0
    )

    degenerate_active = _distal_quality((0.0, 0.0, 1.0), (0.0, 0.0, 2.0), 0.25)
    vertical_swing = _distal_quality((0.0, 0.0, 1.0), (0.0, 0.0, 2.0), 0.0)
    assert torch.isinf(degenerate_active[trajectory._METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION])
    assert vertical_swing[trajectory._METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION] == 0.0


def test_contact_active_quality_reconstructs_calibrated_tangent_without_mutating_raw_diagnostics() -> None:
    """Contact metrics use calibrated radius while raw source diagnostics retain source geometry."""
    contact_active = _distal_quality((1.0, 0.0, 0.0), (0.5, 0.0, 0.0), 0.25)
    swing = _distal_quality((1.0, 0.0, 0.0), (0.5, 0.0, 0.0), 0.0)

    assert contact_active[trajectory._METRIC_SOURCE_REQUIRED_DISTAL_POSITION] == 0.0
    torch.testing.assert_close(swing[trajectory._METRIC_SOURCE_REQUIRED_DISTAL_POSITION], torch.tensor(0.5))
    for quality in (contact_active, swing):
        torch.testing.assert_close(quality[trajectory._METRIC_SOURCE_ALL_DISTAL_POSITION], torch.tensor(0.5))


def test_contact_distal_point_exclusively_owns_a_coincident_position_row() -> None:
    """Contact cannot constrain one physical toe point to incompatible source and planar targets."""
    wp.init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    position_limit_m = 0.020
    distal_limit_m = 0.030
    target_separation_m = 0.07258451730012894
    source_base_x = -0.1
    base_error_m = 0.04
    contact_length_m = target_separation_m - source_base_x
    assert target_separation_m > position_limit_m + distal_limit_m
    assert math.isclose(
        target_separation_m / (position_limit_m + distal_limit_m), 1.4516903460025787, rel_tol=0.0, abs_tol=1.0e-12
    )

    # Body 1 is both the required toe-position carrier and the required distal point at zero offset.
    # Contact retargets that point from raw target x=0 to planar target x=target_separation_m.
    body_q = torch.zeros((1, 2, 7), dtype=torch.float32, device=device)
    body_q[..., 6] = 1.0
    body_q[0, 0, 0] = source_base_x + base_error_m
    body_q[0, 1, 0] = target_separation_m
    source_positions = torch.zeros((2, 1, 3), dtype=torch.float32, device=device)
    source_positions[0, 0, 0] = source_base_x
    source_points = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)
    source_rotations = torch.zeros((1, 1, 4), dtype=torch.float32, device=device)
    source_rotations[..., 3] = 1.0
    position_bodies = torch.tensor((0, 1), dtype=torch.int64, device=device)
    required_position_rows = torch.tensor((0, 1), dtype=torch.int64, device=device)
    position_channels = torch.tensor((0, 0), dtype=torch.int64, device=device)
    parent_rows = torch.tensor((0, 0), dtype=torch.int64, device=device)
    direction_bodies = torch.tensor((1,), dtype=torch.int64, device=device)
    direction_position_rows = torch.tensor((0,), dtype=torch.int64, device=device)
    required_direction_rows = torch.tensor((0,), dtype=torch.int64, device=device)
    direction_point_body = torch.zeros((1, 3), dtype=torch.float32, device=device)
    contact_point_body = torch.tensor(((contact_length_m, 0.0, 0.0),), dtype=torch.float32, device=device)
    confidence = torch.tensor(((0.1,),), dtype=torch.float32, device=device)
    normal_owned = torch.ones_like(confidence)

    guard = torch.empty((1, 5), dtype=torch.float32, device=device)
    problem_indices = torch.zeros(1, dtype=torch.int32, device=device)
    wp.launch(
        trajectory._source_fidelity_guard_residuals,
        dim=(1, 5),
        inputs=[
            wp.from_torch(body_q, dtype=wp.transform),
            wp.from_torch(source_positions),
            wp.from_torch(source_points),
            wp.from_torch(source_rotations),
            wp.from_torch(position_bodies),
            wp.from_torch(direction_bodies),
            wp.from_torch(direction_position_rows),
            wp.from_torch(required_position_rows),
            wp.from_torch(position_channels),
            wp.from_torch(required_direction_rows),
            wp.from_torch(torch.tensor((0,), dtype=torch.int64, device=device)),
            wp.from_torch(required_direction_rows),
            wp.from_torch(confidence, dtype=wp.float32),
            wp.from_torch(normal_owned, dtype=wp.float32),
            wp.from_torch(direction_point_body),
            wp.from_torch(contact_point_body),
            2,
            1,
            0,
            0,
            wp.from_torch(problem_indices),
        ],
        outputs=[wp.from_torch(guard)],
        device=str(device),
    )

    screw = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32, device=device)
    ancestry = torch.ones((5, 1), dtype=torch.uint8, device=device)
    guard_jacobian = torch.zeros((1, 5, 1), dtype=torch.float32, device=device)
    wp.launch(
        trajectory._source_fidelity_guard_jacobian,
        dim=(1, 5, 1),
        inputs=[
            wp.from_torch(body_q, dtype=wp.transform),
            wp.from_torch(screw, dtype=wp.spatial_vector),
            wp.from_torch(source_positions),
            wp.from_torch(source_points),
            wp.from_torch(source_rotations),
            wp.from_torch(position_bodies),
            wp.from_torch(direction_bodies),
            wp.from_torch(direction_position_rows),
            wp.from_torch(required_position_rows),
            wp.from_torch(position_channels),
            wp.from_torch(required_direction_rows),
            wp.from_torch(torch.tensor((0,), dtype=torch.int64, device=device)),
            wp.from_torch(required_direction_rows),
            wp.from_torch(confidence, dtype=wp.float32),
            wp.from_torch(normal_owned, dtype=wp.float32),
            wp.from_torch(direction_point_body),
            wp.from_torch(contact_point_body),
            2,
            1,
            0,
            wp.from_torch(ancestry, dtype=wp.uint8),
            0,
        ],
        outputs=[wp.from_torch(guard_jacobian)],
        device=str(device),
    )

    quality = torch.empty((1, trajectory._TRAJECTORY_METRIC_COUNT), dtype=torch.float32, device=device)
    wp.launch(
        trajectory._trajectory_quality_frames,
        dim=1,
        inputs=[
            wp.from_torch(body_q),
            wp.from_torch(source_positions),
            wp.from_torch(position_bodies),
            wp.from_torch(required_position_rows),
            wp.from_torch(position_channels),
            wp.from_torch(parent_rows),
            wp.from_torch(source_rotations),
            wp.from_torch(torch.tensor((0,), dtype=torch.int64, device=device)),
            wp.from_torch(source_points),
            wp.from_torch(direction_bodies),
            wp.from_torch(direction_position_rows),
            wp.from_torch(required_direction_rows),
            wp.from_torch(torch.tensor((0,), dtype=torch.int64, device=device)),
            wp.from_torch(required_direction_rows),
            wp.from_torch(confidence, dtype=wp.float32),
            wp.from_torch(normal_owned, dtype=wp.float32),
            wp.from_torch(direction_point_body),
            wp.from_torch(contact_point_body),
            1,
            2,
            1,
            1,
            2,
            1,
            trajectory._TRAJECTORY_METRIC_COUNT,
            trajectory._METRIC_SOURCE_REQUIRED_POSITION,
            trajectory._METRIC_SOURCE_REQUIRED_DISTAL_POSITION,
            trajectory._METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION,
            trajectory._METRIC_SOURCE_ROOT_ROTATION,
            trajectory._METRIC_SOURCE_ALL_POSITION,
            trajectory._METRIC_SOURCE_ALL_DISTAL_POSITION,
            trajectory._METRIC_SOURCE_ALL_LANDMARK_DIRECTION,
            trajectory._METRIC_SOURCE_ALL_DISTAL_DIRECTION,
            trajectory._METRIC_SOURCE_NONROOT_ROTATION,
        ],
        outputs=[wp.from_torch(quality)],
        device=str(device),
    )
    wp.synchronize_device(str(device))

    # Contact owns the coincident toe row; the raw all-position diagnostic remains visible.
    actual = torch.stack(
        (
            guard[0, 0],
            guard[0, 1],
            guard_jacobian[0, 0, 0],
            guard_jacobian[0, 1, 0],
            quality[0, trajectory._METRIC_SOURCE_REQUIRED_POSITION],
            quality[0, trajectory._METRIC_SOURCE_REQUIRED_DISTAL_POSITION],
            quality[0, trajectory._METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION],
            quality[0, trajectory._METRIC_SOURCE_ALL_POSITION],
            quality[0, trajectory._METRIC_SOURCE_ALL_DISTAL_POSITION],
        )
    ).cpu()
    expected = torch.tensor(
        (base_error_m, 0.0, 1.0, 0.0, base_error_m, 0.0, 0.0, target_separation_m, target_separation_m)
    )
    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=0.0)


def test_collision_only_quality_releases_absolute_height_but_preserves_true_swing_direction() -> None:
    """A rigid collision lift cannot acquire contact tangent or direction semantics."""
    collision = _distal_quality(
        (1.0, 0.0, 2.0),
        (1.0, 0.0, 0.0),
        confidence=0.0,
        normal_owned=1.0,
        actual_base=(0.0, 0.0, 1.0),
        source_base=(0.0, 0.0, 0.0),
    )

    assert collision[trajectory._METRIC_SOURCE_REQUIRED_POSITION] == 0.0
    assert collision[trajectory._METRIC_SOURCE_REQUIRED_DISTAL_POSITION] == 0.0
    torch.testing.assert_close(
        collision[trajectory._METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION],
        torch.tensor(math.pi / 4),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(collision[trajectory._METRIC_SOURCE_ALL_POSITION], torch.tensor(1.0))
    torch.testing.assert_close(collision[trajectory._METRIC_SOURCE_ALL_DISTAL_POSITION], torch.tensor(2.0))


def test_source_position_objective_separates_contact_point_and_normal_ownership() -> None:
    """A coincident distal point is released in contact while base and clearance XY remain source-owned."""
    wp.init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    body = torch.zeros((1, 1, 7), dtype=torch.float32, device=device)
    body[..., 6] = 1.0
    body[0, 0, :3] = torch.tensor((1.0, 0.0, 1.0), device=device)
    target = torch.tensor(((0.5, 0.0, 0.0),), dtype=torch.float32, device=device)
    source_base = torch.zeros_like(target)
    problem_indices = torch.zeros(1, dtype=torch.int32, device=device)
    screw = torch.tensor([[[1.0, 0.0, 1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32, device=device)
    affects_dof = torch.ones(1, dtype=torch.uint8, device=device)

    results = []
    jacobians = []
    cases = (
        (0.25, 1.0, True),
        (0.25, 1.0, False),
        (0.0, 0.0, True),
        (0.0, 1.0, True),
    )
    for confidence_value, ownership, contact_point_owned in cases:
        confidence = torch.tensor(((confidence_value,),), dtype=torch.float32, device=device)
        normal_owned = torch.tensor(((ownership,),), dtype=torch.float32, device=device)
        residual = torch.empty((1, 3), dtype=torch.float32, device=device)
        jacobian = torch.zeros((1, 3, 1), dtype=torch.float32, device=device)
        wp.launch(
            trajectory._source_phase_position_residuals,
            dim=1,
            inputs=[
                wp.from_torch(body, dtype=wp.transform),
                wp.from_torch(target, dtype=wp.vec3),
                wp.from_torch(source_base, dtype=wp.vec3),
                wp.from_torch(confidence, dtype=wp.float32),
                wp.from_torch(normal_owned, dtype=wp.float32),
                0,
                wp.vec3(0.0, 0.0, 0.0),
                0,
                contact_point_owned,
                0.0,
                0,
                1.0,
                wp.from_torch(problem_indices),
            ],
            outputs=[wp.from_torch(residual)],
            device=str(device),
        )
        wp.launch(
            trajectory._source_phase_position_jacobian,
            dim=(1, 1),
            inputs=[
                wp.from_torch(body, dtype=wp.transform),
                wp.from_torch(screw, dtype=wp.spatial_vector),
                wp.from_torch(confidence, dtype=wp.float32),
                wp.from_torch(normal_owned, dtype=wp.float32),
                wp.from_torch(affects_dof, dtype=wp.uint8),
                0,
                wp.vec3(0.0, 0.0, 0.0),
                0,
                contact_point_owned,
                0,
                1.0,
            ],
            outputs=[wp.from_torch(jacobian)],
            device=str(device),
        )
        wp.synchronize_device(str(device))
        results.append(residual.cpu())
        jacobians.append(jacobian.cpu())

    torch.testing.assert_close(results[0], torch.zeros_like(results[0]))
    torch.testing.assert_close(results[1], torch.tensor(((-0.5, 0.0, 0.0),)))
    torch.testing.assert_close(results[2], torch.tensor(((-0.5, 0.0, -1.0),)))
    torch.testing.assert_close(results[3], torch.tensor(((-0.5, 0.0, 0.0),)))
    torch.testing.assert_close(jacobians[0], torch.zeros_like(jacobians[0]))
    torch.testing.assert_close(jacobians[1], torch.tensor([[[-1.0], [0.0], [0.0]]]))
    torch.testing.assert_close(jacobians[2], torch.tensor([[[-1.0], [0.0], [-1.0]]]))
    torch.testing.assert_close(jacobians[3], torch.tensor([[[-1.0], [0.0], [0.0]]]))


def test_contact_active_source_guards_match_planar_and_true_swing_geometry() -> None:
    """Distal guard residuals and Jacobians share the confidence-owned projection."""
    wp.init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source_positions = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)
    source_points = torch.tensor([[[1.0, 0.0, 1.0]]], dtype=torch.float32, device=device)
    source_rotations = torch.zeros((1, 1, 4), dtype=torch.float32, device=device)
    source_rotations[..., 3] = 1.0
    indices = torch.zeros(1, dtype=torch.int64, device=device)
    direction_bodies = torch.ones(1, dtype=torch.int64, device=device)
    point_body = torch.zeros((1, 3), dtype=torch.float32, device=device)
    contact_point_body = torch.tensor(((1.0, 0.0, 0.0),), dtype=torch.float32, device=device)
    problem_indices = torch.zeros(1, dtype=torch.int32, device=device)

    def residual(yaw: float, confidence_value: float, normal_owned_value: float | None = None) -> torch.Tensor:
        body = torch.zeros((1, 2, 7), dtype=torch.float32, device=device)
        body[..., 6] = 1.0
        body[0, 1, :3] = torch.tensor((math.cos(yaw), math.sin(yaw), 0.5), device=device)
        confidence = torch.tensor(((confidence_value,),), dtype=torch.float32, device=device)
        ownership = float(confidence_value > 0.0) if normal_owned_value is None else normal_owned_value
        normal_owned = torch.tensor(((ownership,),), dtype=torch.float32, device=device)
        values = torch.empty((1, 4), dtype=torch.float32, device=device)
        wp.launch(
            trajectory._source_fidelity_guard_residuals,
            dim=(1, 4),
            inputs=[
                wp.from_torch(body, dtype=wp.transform),
                wp.from_torch(source_positions),
                wp.from_torch(source_points),
                wp.from_torch(source_rotations),
                wp.from_torch(indices),
                wp.from_torch(direction_bodies),
                wp.from_torch(indices),
                wp.from_torch(indices),
                wp.from_torch(indices),
                wp.from_torch(indices),
                wp.from_torch(indices),
                wp.from_torch(indices),
                wp.from_torch(confidence, dtype=wp.float32),
                wp.from_torch(normal_owned, dtype=wp.float32),
                wp.from_torch(point_body),
                wp.from_torch(contact_point_body),
                1,
                1,
                0,
                0,
                wp.from_torch(problem_indices),
            ],
            outputs=[wp.from_torch(values)],
            device=str(device),
        )
        wp.synchronize_device(str(device))
        return values[0, 1:3].cpu()

    yaw = 0.2
    body = torch.zeros((1, 2, 7), dtype=torch.float32, device=device)
    body[..., 6] = 1.0
    body[0, 1, :3] = torch.tensor((math.cos(yaw), math.sin(yaw), 0.5), device=device)
    screw = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)
    ancestry = torch.zeros((4, 1), dtype=torch.uint8, device=device)
    ancestry[1:3, 0] = 1
    epsilon = 1.0e-3
    planar_chord = 2.0 * math.sin(0.5 * yaw)
    swing_direction_chord = math.sqrt(2.0 - 2.0 * (math.cos(yaw) + 0.5) / math.sqrt(2.5))
    expected = (
        (0.25, 1.0, torch.tensor((planar_chord, planar_chord))),
        (0.0, 0.0, torch.tensor((math.hypot(planar_chord, 0.5), swing_direction_chord))),
        (0.0, 1.0, torch.tensor((planar_chord, swing_direction_chord))),
    )

    for confidence_value, ownership, expected_values in expected:
        confidence = torch.tensor(((confidence_value,),), dtype=torch.float32, device=device)
        normal_owned = torch.tensor(((ownership,),), dtype=torch.float32, device=device)
        jacobian = torch.zeros((1, 4, 1), dtype=torch.float32, device=device)
        wp.launch(
            trajectory._source_fidelity_guard_jacobian,
            dim=(1, 4, 1),
            inputs=[
                wp.from_torch(body, dtype=wp.transform),
                wp.from_torch(screw, dtype=wp.spatial_vector),
                wp.from_torch(source_positions),
                wp.from_torch(source_points),
                wp.from_torch(source_rotations),
                wp.from_torch(indices),
                wp.from_torch(direction_bodies),
                wp.from_torch(indices),
                wp.from_torch(indices),
                wp.from_torch(indices),
                wp.from_torch(indices),
                wp.from_torch(indices),
                wp.from_torch(indices),
                wp.from_torch(confidence, dtype=wp.float32),
                wp.from_torch(normal_owned, dtype=wp.float32),
                wp.from_torch(point_body),
                wp.from_torch(contact_point_body),
                1,
                1,
                0,
                wp.from_torch(ancestry, dtype=wp.uint8),
                0,
            ],
            outputs=[wp.from_torch(jacobian)],
            device=str(device),
        )
        wp.synchronize_device(str(device))

        actual = residual(yaw, confidence_value, ownership)
        torch.testing.assert_close(actual, expected_values, atol=1.0e-6, rtol=0.0)
        finite_difference = (
            residual(yaw + epsilon, confidence_value, ownership) - residual(yaw - epsilon, confidence_value, ownership)
        ) / (2.0 * epsilon)
        torch.testing.assert_close(jacobian[0, 1:3, 0].cpu(), finite_difference, atol=2.0e-3, rtol=2.0e-3)


def test_noncontact_required_direction_does_not_index_contact_evidence() -> None:
    """A required hand-like row stays strict 3-D while contact tensors remain foot-only."""
    wp.init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    body_q = torch.zeros((1, 3, 7), dtype=torch.float32, device=device)
    body_q[..., 6] = 1.0
    body_q[0, 1, :3] = torch.tensor((1.0, 0.0, 0.0), device=device)
    body_q[0, 2, :3] = torch.tensor((0.0, 1.0, 0.2), device=device)
    source_positions = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)
    source_points = torch.tensor((((1.0, 0.0, 0.0),), ((0.0, 1.0, 0.0),)), device=device)
    source_rotations = torch.tensor((((0.0, 0.0, 0.0, 1.0),),), device=device).transpose(0, 1)
    position_bodies = torch.tensor((0,), dtype=torch.int64, device=device)
    required_position_rows = torch.tensor((0,), dtype=torch.int64, device=device)
    position_channels = torch.tensor((0,), dtype=torch.int64, device=device)
    parent_rows = torch.tensor((0,), dtype=torch.int64, device=device)
    direction_bodies = torch.tensor((1, 2), dtype=torch.int64, device=device)
    direction_position_rows = torch.tensor((0, 0), dtype=torch.int64, device=device)
    contact_direction_rows = torch.tensor((0,), dtype=torch.int64, device=device)
    direction_contact_channels = torch.tensor((0, -1), dtype=torch.int64, device=device)
    required_direction_rows = torch.tensor((0, 1), dtype=torch.int64, device=device)
    confidence = torch.ones((1, 1), dtype=torch.float32, device=device)
    normal_owned = torch.ones_like(confidence)
    direction_point_body = torch.zeros((2, 3), dtype=torch.float32, device=device)
    contact_point_body = torch.tensor(((1.0, 0.0, 0.0),), device=device)
    problem_indices = torch.zeros(1, dtype=torch.int32, device=device)

    guard = torch.empty((1, 6), dtype=torch.float32, device=device)
    wp.launch(
        trajectory._source_fidelity_guard_residuals,
        dim=(1, 6),
        inputs=[
            wp.from_torch(body_q, dtype=wp.transform),
            wp.from_torch(source_positions),
            wp.from_torch(source_points),
            wp.from_torch(source_rotations),
            wp.from_torch(position_bodies),
            wp.from_torch(direction_bodies),
            wp.from_torch(direction_position_rows),
            wp.from_torch(required_position_rows),
            wp.from_torch(position_channels),
            wp.from_torch(contact_direction_rows),
            wp.from_torch(direction_contact_channels),
            wp.from_torch(required_direction_rows),
            wp.from_torch(confidence, dtype=wp.float32),
            wp.from_torch(normal_owned, dtype=wp.float32),
            wp.from_torch(direction_point_body),
            wp.from_torch(contact_point_body),
            1,
            2,
            0,
            0,
            wp.from_torch(problem_indices),
        ],
        outputs=[wp.from_torch(guard)],
        device=str(device),
    )

    screw = torch.tensor([[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]], device=device)
    ancestry = torch.ones((6, 1), dtype=torch.uint8, device=device)
    jacobian = torch.zeros((1, 6, 1), dtype=torch.float32, device=device)
    wp.launch(
        trajectory._source_fidelity_guard_jacobian,
        dim=(1, 6, 1),
        inputs=[
            wp.from_torch(body_q, dtype=wp.transform),
            wp.from_torch(screw, dtype=wp.spatial_vector),
            wp.from_torch(source_positions),
            wp.from_torch(source_points),
            wp.from_torch(source_rotations),
            wp.from_torch(position_bodies),
            wp.from_torch(direction_bodies),
            wp.from_torch(direction_position_rows),
            wp.from_torch(required_position_rows),
            wp.from_torch(position_channels),
            wp.from_torch(contact_direction_rows),
            wp.from_torch(direction_contact_channels),
            wp.from_torch(required_direction_rows),
            wp.from_torch(confidence, dtype=wp.float32),
            wp.from_torch(normal_owned, dtype=wp.float32),
            wp.from_torch(direction_point_body),
            wp.from_torch(contact_point_body),
            1,
            2,
            0,
            wp.from_torch(ancestry, dtype=wp.uint8),
            0,
        ],
        outputs=[wp.from_torch(jacobian)],
        device=str(device),
    )

    quality = torch.empty((1, trajectory._TRAJECTORY_METRIC_COUNT), dtype=torch.float32, device=device)
    wp.launch(
        trajectory._trajectory_quality_frames,
        dim=1,
        inputs=[
            wp.from_torch(body_q),
            wp.from_torch(source_positions),
            wp.from_torch(position_bodies),
            wp.from_torch(required_position_rows),
            wp.from_torch(position_channels),
            wp.from_torch(parent_rows),
            wp.from_torch(source_rotations),
            wp.from_torch(position_bodies),
            wp.from_torch(source_points),
            wp.from_torch(direction_bodies),
            wp.from_torch(direction_position_rows),
            wp.from_torch(contact_direction_rows),
            wp.from_torch(direction_contact_channels),
            wp.from_torch(required_direction_rows),
            wp.from_torch(confidence, dtype=wp.float32),
            wp.from_torch(normal_owned, dtype=wp.float32),
            wp.from_torch(direction_point_body),
            wp.from_torch(contact_point_body),
            1,
            1,
            1,
            2,
            1,
            2,
            trajectory._TRAJECTORY_METRIC_COUNT,
            trajectory._METRIC_SOURCE_REQUIRED_POSITION,
            trajectory._METRIC_SOURCE_REQUIRED_DISTAL_POSITION,
            trajectory._METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION,
            trajectory._METRIC_SOURCE_ROOT_ROTATION,
            trajectory._METRIC_SOURCE_ALL_POSITION,
            trajectory._METRIC_SOURCE_ALL_DISTAL_POSITION,
            trajectory._METRIC_SOURCE_ALL_LANDMARK_DIRECTION,
            trajectory._METRIC_SOURCE_ALL_DISTAL_DIRECTION,
            trajectory._METRIC_SOURCE_NONROOT_ROTATION,
        ],
        outputs=[wp.from_torch(quality)],
        device=str(device),
    )
    wp.synchronize_device(str(device))

    torch.testing.assert_close(guard[0, 1:3].cpu(), torch.tensor((0.0, 0.2)), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(jacobian[0, 1:3, 0].cpu(), torch.tensor((0.0, 1.0)), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(
        quality[0, trajectory._METRIC_SOURCE_REQUIRED_DISTAL_POSITION].cpu(),
        torch.tensor(0.2),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert torch.isfinite(guard[0, 4])
    assert guard[0, 4] > 0.0
    assert torch.isfinite(quality[0, trajectory._METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION])
