# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture gates for source-independent motion-frame construction."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.core.multi_task.motion.data import MotionSkeleton
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import _TARGET_COORDINATE_QUALITY_NAMES

_MULTI_TASK_ROOT = Path(__file__).parents[1]
_MOTION_ROOT = _MULTI_TASK_ROOT / "motion"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _skeleton() -> MotionSkeleton:
    return MotionSkeleton(
        identifier="source-a",
        content_sha256=_sha256("source-artifact-a"),
        body_names=("root", "hip", "knee"),
        parent_indices=(-1, 0, 1),
        rest_translation_m=((0.0, 0.0, 0.8), (0.0, 0.1, -0.2), (0.0, 0.0, -0.4)),
        rest_rotation_wxyz=((1.0, 0.0, 0.0, 0.0),) * 3,
        joint_names=("hip_y", "knee_y"),
        joint_child_body_indices=(1, 2),
        joint_axes=((0.0, 1.0, 0.0),) * 2,
        root_translation_frame="world",
        root_rotation_convention="wxyz",
        landmarks=(MotionSkeleton.Landmark("pelvis", "root", "root"),),
        landmark_rotation_policy="anatomical_root",
    )


def _top_level_symbols(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return classes, functions


def _attribute_calls(nodes: list[ast.stmt], names: set[str]) -> set[str]:
    return {
        node.func.attr
        for statement in nodes
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in names
    }


def test_motion_source_clip_is_the_only_decoded_clip_contract() -> None:
    """Every source must expose the same two lazy mathematical views."""
    source = importlib.import_module("isaaclab_tasks.core.multi_task.motion.data.source")
    clip_type = getattr(source, "MotionSourceClip")

    class Clip:
        source_fps = 30.0
        frame_count = 3

        def free_root_coordinates(self, source_skeleton, *, device):
            del source_skeleton
            return torch.empty(3, 9, device=device), None

        def local_pose(self, source_skeleton, *, device):
            del source_skeleton
            return torch.empty(3, 3, device=device), torch.empty(3, 3, 4, device=device)

    assert isinstance(Clip(), clip_type)
    for retired in ("MotionGeneralizedCoordinateClip", "MotionPoseAxisAngleClip", "MotionLocalBodyPoseClip"):
        assert not hasattr(source, retired)


@pytest.mark.parametrize(
    ("robot", "target_factory", "projection_factory"),
    (
        ("g1", "g1_frame_target", "g1_source_projection"),
        ("smpl", "smpl_frame_target", "smpl_source_projection"),
    ),
)
def test_motion_robot_owns_internal_target_and_source_factories(
    robot: str,
    target_factory: str,
    projection_factory: str,
) -> None:
    """A robot exposes only deprecated builder shims while active target factories remain private."""
    reference = _MOTION_ROOT / "robots" / robot / "reference.py"
    classes, functions = _top_level_symbols(reference)
    deprecated_class = "G1FrameBuilder" if robot == "g1" else "SmplFrameBuilder"
    deprecated_factory = "g1_frame_builder" if robot == "g1" else "smpl_frame_builder"
    assert {name for name in classes if name.endswith("FrameBuilder")} == {deprecated_class}
    assert {target_factory, projection_factory} <= functions
    assert {name for name in functions if name.endswith("_frame_builder")} == {deprecated_factory}

    stub = (_MOTION_ROOT / "robots" / robot / "__init__.pyi").read_text(encoding="utf-8")
    assert target_factory not in stub
    assert projection_factory not in stub
    assert deprecated_class in stub
    assert deprecated_factory in stub
    assert "deprecated_frame_builder" not in stub
    assert not (reference.parent / "deprecated_frame_builder.py").exists()
    source = reference.read_text(encoding="utf-8")
    assert "MotionSemanticProjection" not in source
    assert "MotionSemanticTargets" not in source
    for retired in (
        "G1PoseFrameBuilder",
        "G1LocalBodyPoseFrameBuilder",
        "g1_pose_frame_builder",
        "g1_local_body_pose_frame_builder",
        "SmplGeneralizedCoordinateFrameBuilder",
        "SmplLocalBodyPoseFrameBuilder",
        "smpl_generalized_coordinate_frame_builder",
        "smpl_local_body_pose_frame_builder",
    ):
        assert retired not in stub


def test_smpl_source_projection_uses_exact_compatible_analytic_or_common_trajectory_route() -> None:
    """SMPL selects one visible route without dataset-specific dispatch or coordinate clamping."""
    path = _MOTION_ROOT / "robots" / "smpl" / "reference.py"
    source = path.read_text(encoding="utf-8")
    route = source[source.index("def smpl_source_projection(") : source.index("_FRAME_BUILDER_MIGRATION")]

    assert "amass" not in source.casefold()
    assert "class _SmplCompatiblePoseSource(Protocol):" in source
    assert "return MotionSourceProjectionAnalytic(" in route
    assert "source.compatible_pose_profile_sha256 == SMPL_COMPATIBLE_POSE_PROFILE_SHA256" in route
    assert "source.smpl_subject_model(" in route
    assert "target.neutral_calibration_model()" in route
    assert '"subject_model_sha256"' in route and '"neutral_model_sha256"' in route
    assert '"subject_artifact_sha256"' in route and '"neutral_artifact_sha256"' in route
    exact = route.index("return MotionSourceProjectionExact(")
    analytic = route.index("return MotionSourceProjectionAnalytic(")
    trajectory = route.index("return MotionSourceProjectionTrajectory(")
    assert exact < analytic < trajectory
    assert "projection = MotionTrajectoryProjection(" in route
    assert "target_projection=projection" in route
    assert "clamp" not in route.casefold()
    assert "smpl_lbs_models" not in source
    builder = (_MOTION_ROOT / "mdp" / "commands" / "motion_task_table_builder.py").read_text(encoding="utf-8")
    assert "Motion production requires MotionSourceProjectionTrajectory" not in builder
    assert "invalid_routes" not in builder
    assert "_validate_motion_manifest(source_index, declared_output_index)" not in builder


def test_smpl_cross_source_initializer_is_explicitly_constrained() -> None:
    """A mapped SMPL pose remains a seed and never bypasses constrained frame IK."""
    reference = (_MOTION_ROOT / "robots" / "smpl" / "reference.py").read_text(encoding="utf-8")
    retarget = (_MOTION_ROOT / "retarget.py").read_text(encoding="utf-8")

    assert 'initializer_policy="batched_frame_ik"' in reference
    assert "rotation_map_or_frame_ik" not in reference
    assert "rotation_map_or_frame_ik" not in retarget
    assert "coordinate.clamp_" in reference


def test_g1_foot_correspondence_uses_reference_ankle_roll_geometry() -> None:
    """G1 feet use ProtoMotions' ankle-roll endpoint and raw ankle-to-toe direction."""
    path = _MOTION_ROOT / "robots" / "g1" / "reference.py"
    source = path.read_text(encoding="utf-8")
    start = source.index("_G1_RETARGET_TARGETS =")
    stop = source.index("_G1_REQUIRED_POSITION_ROLES =", start)
    layout = source[start:stop]

    assert 'Landmark("left_ankle", "left_ankle_roll_link", 2,' in layout
    assert 'Landmark("right_ankle", "right_ankle_roll_link", 5,' in layout
    left_foot = (
        '"left_foot", "left_ankle", "left_ankle", "left_toe", "between_positions",\n'
        '        "left_ankle_roll_link",\n'
        "        (0.15, 0.0, 0.0),"
    )
    right_foot = (
        '"right_foot", "right_ankle", "right_ankle", "right_toe", "between_positions",\n'
        '        "right_ankle_roll_link",\n'
        "        (0.15, 0.0, 0.0),"
    )
    assert left_foot in layout
    assert right_foot in layout
    assert "ankle_pitch_link" not in layout


def test_motion_root_selects_target_and_projection_only_from_the_robot_axis() -> None:
    """A dataset token must never select or wrap either robot factory."""
    path = _MULTI_TASK_ROOT / "motion_env_cfg.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    target = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MotionTargetKinematicsCfg"
    )
    assignments = {
        name.id: node.value
        for node in target.body
        if isinstance(node, ast.Assign)
        for name in node.targets
        if isinstance(name, ast.Name)
    }

    expected = {
        "default": ("smpl_frame_target", "smpl_source_projection"),
        "g1": ("g1_frame_target", "g1_source_projection"),
    }
    for alternative, factory_names in expected.items():
        value = assignments[alternative]
        assert isinstance(value, ast.Call)
        keywords = {keyword.arg: keyword.value for keyword in value.keywords}
        for field, factory_name in zip(("target_factory", "source_projection_factory"), factory_names, strict=True):
            assert isinstance(keywords[field], ast.Name)
            assert keywords[field].id == factory_name
            assert all(not isinstance(node, ast.Name) or node.id != "preset" for node in ast.walk(keywords[field]))
    assert isinstance(assignments["smpl"], ast.Name) and assignments["smpl"].id == "default"

    root_source = path.read_text(encoding="utf-8")
    assert root_source.count("MotionExactFamilyCfg(") == 1
    assert root_source.count("MotionAnalyticFamilyCfg(") == 1
    assert root_source.count("MotionTrajectoryFamilyCfg(") == 1
    assert "MotionExactMaterializationCriterionCfg" not in root_source
    assert "MotionConstraintGeometryFeasibleCriterionCfg()" in root_source
    assert "MotionInnerSolveConvergedCriterionCfg()" not in root_source
    assert "MotionRequiredRefinementConvergedCriterionCfg()" in root_source
    assert "MotionNonlinearPhasesConvergedCriterionCfg" not in root_source
    assert "MotionConstraintFeasibleCriterionCfg" not in root_source
    assert "MotionSourceFidelityCriterionCfg()" in root_source
    assert "MotionContactCriterionCfg()" in root_source
    assert root_source.count("MotionGroundPenetrationCriterionCfg()") == 3
    assert "MotionSourceGlobalPositionCriterionCfg" not in root_source
    assert "MotionSourceRootRotationCriterionCfg" not in root_source
    assert "MotionContactGeometryCriterionCfg" not in root_source
    assert "upper_m=0.15" not in root_source and "upper_rad=0.75" not in root_source


def test_motion_coordinate_identity_ignores_source_provenance() -> None:
    """Dataset names and artifact bytes do not change mathematical coordinate compatibility."""
    skeleton = _skeleton()
    renamed = dataclasses.replace(
        skeleton,
        identifier="renamed-source",
        content_sha256=_sha256("different-but-coordinate-equivalent-artifact"),
    )

    assert skeleton.identity_sha256 != renamed.identity_sha256
    assert skeleton.coordinate_identity_sha256 == renamed.coordinate_identity_sha256


@pytest.mark.parametrize(
    "changes",
    (
        {"parent_indices": (-1, 0, 0)},
        {"rest_translation_m": ((0.0, 0.0, 0.8), (0.0, 0.2, -0.2), (0.0, 0.0, -0.4))},
        {"rest_rotation_wxyz": ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))},
        {"joint_names": ("knee_y", "hip_y")},
        {"joint_child_body_indices": (2, 1)},
        {"joint_axes": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))},
        {"root_translation_frame": "root"},
        {"root_rotation_convention": "xyzw"},
        {"position_unit": "cm"},
        {"angle_unit": "deg"},
    ),
)
def test_motion_coordinate_identity_detects_mathematical_changes(changes: dict[str, object]) -> None:
    """Topology, rest geometry, coordinate layout, conventions, and units define compatibility."""
    skeleton = _skeleton()
    changed = dataclasses.replace(skeleton, **changes)

    assert skeleton.coordinate_identity_sha256 != changed.coordinate_identity_sha256


@pytest.mark.parametrize(("robot", "target_class"), (("g1", "_G1FrameTarget"), ("smpl", "_SmplFrameTarget")))
def test_motion_robot_separates_target_materialization_from_source_views(robot: str, target_class: str) -> None:
    """Robot targets own mechanics while each source route exposes only its valid operation."""
    from isaaclab_tasks.core.multi_task.motion.data.frames import (
        MotionSourceProjectionAnalytic,
        MotionSourceProjectionExact,
        MotionSourceProjectionTrajectory,
    )

    path = _MOTION_ROOT / "robots" / robot / "reference.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    target_methods = {
        node.name for node in classes[target_class].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    target_fields = {
        node.target.id
        for node in classes[target_class].body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert {
        "kinematics",
        "kinematic_tree",
        "materialization_minimum_frames",
        "allocate_coordinates",
        "coordinates_from_newton",
        "write_joint_position_newton",
        "write_nonroot_velocity_canonical",
        "materialize_coordinates",
    } <= target_methods | target_fields
    assert not any(name.endswith("SourceProjection") for name in classes)

    fields_by_route = {
        MotionSourceProjectionExact: {"convert_coordinates"},
        MotionSourceProjectionAnalytic: {"output_clip_index", "convert_clip"},
        MotionSourceProjectionTrajectory: {"target_projection"},
    }
    route_operations = {"convert_coordinates", "output_clip_index", "convert_clip", "target_projection"}
    common = {"source_skeleton", "target", "version", "construction_identity_sha256"}
    for route, operations in fields_by_route.items():
        field_names = {field.name for field in dataclasses.fields(route)}
        assert field_names == common | operations
        assert not field_names & (route_operations - operations)

    builder_path = _MOTION_ROOT / "mdp" / "commands" / "motion_task_table_builder.py"
    trajectory_path = _MOTION_ROOT / "mdp" / "commands" / "motion_trajectory.py"
    builder = ast.parse(builder_path.read_text(encoding="utf-8"), filename=str(builder_path))
    trajectory = ast.parse(trajectory_path.read_text(encoding="utf-8"), filename=str(trajectory_path))
    builder_functions = {node.name: node for node in builder.body if isinstance(node, ast.FunctionDef)}
    trajectory_functions = {node.name: node for node in trajectory.body if isinstance(node, ast.FunctionDef)}
    exact = builder_functions["motion_generate_exact_coordinates"]
    landmarks = trajectory_functions["_motion_source_evidence"]
    views = {"free_root_coordinates", "local_pose"}
    assert _attribute_calls(exact.body, views) == {"free_root_coordinates"}
    assert _attribute_calls(landmarks.body, views) == {"local_pose"}


def test_trajectory_solver_uses_one_memory_planned_whole_clip_workspace() -> None:
    """One memory-planned solver owns restoration and objective whole-clip phases."""
    path = _MOTION_ROOT / "mdp" / "commands" / "motion_trajectory.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    trajectory_functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    solve = trajectory_functions["motion_solve_trajectory"]
    solve_source = ast.get_source_segment(source, solve)
    assert solve_source is not None
    phase_solve = next(
        node for node in ast.walk(solve) if isinstance(node, ast.FunctionDef) and node.name == "solve_phase"
    )
    phase_solve_source = ast.get_source_segment(source, phase_solve)
    assert phase_solve_source is not None
    calls = [
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(solve)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    ]
    solver_calls = [
        node
        for node in ast.walk(solve)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "solve"
        and isinstance(node.func.value, ast.Name)
    ]
    step_calls = [
        node
        for node in ast.walk(solve)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        and isinstance(node.func.value, ast.Name)
    ]
    nested_functions = {node.name for node in ast.walk(solve) if isinstance(node, ast.FunctionDef)}
    assert calls.count("plan_trajectory_memory") == 1
    assert calls.count("IKOptimizerLM") == 2
    assert calls.count("IKTrajectorySolver") == 1
    assert calls.count("create_ik_solver") == 1
    assert [node.func.value.id for node in solver_calls].count("solver") == 1
    assert [node.func.value.id for node in solver_calls].count("frame_seed_global_solver") == 1
    assert [node.func.value.id for node in step_calls].count("frame_seed_local_optimizer") == 1
    assert calls.count("allocate_coordinates") == 1
    assert not {"solve_pose", "project_pose"} & nested_functions
    assert "pose_solver" not in solve_source
    assert "torch.maximum" not in solve_source
    assert "torch.minimum" not in solve_source
    assert "motion_initialize_trajectory_coordinates" not in calls
    assert calls.count("step") == 1
    assert "fill" in calls
    assert "sampler=ik.IKSampler.GAUSS" in solve_source
    assert "n_seeds=_FRAME_SEED_GLOBAL_SEEDS" in solve_source
    assert "iterations=_FRAME_SEED_LOCAL_ITERATIONS" in solve_source
    assert "direct_coordinate_qd_indices" not in solve_source

    local_frame_loops = [
        node
        for node in ast.walk(solve)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "relative_frame"
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
    ]
    assert len(local_frame_loops) == 1
    forbidden_allocations = {
        node.func.attr
        for node in ast.walk(local_frame_loops[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    }
    assert not forbidden_allocations & {"arange", "as_tensor", "empty", "full", "ones", "tensor", "zeros"}
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert constants["_FRAME_SEED_GLOBAL_SEEDS"] == 64
    assert constants["_FRAME_SEED_GLOBAL_ITERATIONS"] == 200
    assert constants["_FRAME_SEED_LOCAL_ITERATIONS"] == 24
    assert constants["_FRAME_SEED_LOCAL_CANDIDATES"] == 2
    assert "batch_constraint_geometry_feasible.fill_(True)" not in solve_source
    assert "phase_active.fill_(1)" not in solve_source
    assert "out=batch_constraint_geometry_feasible" in solve_source

    global_project_source = ast.get_source_segment(source, trajectory_functions["_motion_frame_seed_project"])
    local_gather_source = ast.get_source_segment(source, trajectory_functions["_motion_frame_seed_local_gather"])
    objective_source = ast.get_source_segment(source, trajectory_functions["_motion_frame_seed_objectives"])
    local_project = next(
        node
        for node in ast.walk(solve)
        if isinstance(node, ast.FunctionDef) and node.name == "project_local_frame_seeds"
    )
    local_project_source = ast.get_source_segment(source, local_project)
    assert all(
        part is not None
        for part in (global_project_source, local_gather_source, local_project_source, objective_source)
    )
    assert "_motion_frame_seed_project_local" not in trajectory_functions
    assert "baseline_joint_q" not in global_project_source
    assert "selected_joint_q[" in local_gather_source
    assert "previous_frame, coordinate" in local_gather_source
    assert "baseline_joint_q[frame, coordinate]" in local_gather_source
    assert "target_candidate_joint_q" in local_gather_source
    assert "previous_joint_q" not in local_gather_source
    assert "_motion_frame_seed_project" in local_project_source
    assert "velocity_lower" not in local_project_source
    assert "velocity_upper" not in local_project_source
    assert "step_seconds" not in local_project_source
    assert "motion_objective_source_global_position" in objective_source
    assert "motion_objective_source_rotation" in objective_source
    assert "motion_objective_source_root_rotation" not in objective_source

    assert "(_FRAME_SEED_LOCAL_CANDIDATES * max_batch_clips, coordinate_count)" in solve_source
    assert ".repeat_interleave(_FRAME_SEED_LOCAL_CANDIDATES)" in solve_source
    assert "problem_idx=wp.from_torch(frame_seed_local_problem_indices)" in solve_source
    local_frame_loop_source = ast.get_source_segment(source, local_frame_loops[0])
    assert local_frame_loop_source is not None
    assert not any(route in local_frame_loop_source.lower() for route in ("lafan", "cmu", "g1", "smpl"))

    local_gather = solve_source.index("_motion_frame_seed_local_gather")
    local_project_seed = solve_source.index(
        "project_local_frame_seeds(frame_seed_local_candidate_joint_q_wp)", local_gather
    )
    local_step = solve_source.index("frame_seed_local_optimizer.step(", local_project_seed)
    local_final_cost = solve_source.index("frame_seed_local_optimizer.compute_costs(", local_step)
    local_scatter = solve_source.index("_motion_frame_seed_local_scatter", local_final_cost)
    assert local_gather < local_project_seed < local_step < local_final_cost < local_scatter

    phase_calls = sorted(
        (
            node
            for node in ast.walk(solve)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "solve_phase"
        ),
        key=lambda node: node.lineno,
    )
    assert len(phase_calls) == 1
    monolithic_call = ast.get_source_segment(source, phase_calls[0])
    assert monolithic_call is not None

    seed_validation = solve_source.index("_motion_initializer_validate_or_restore")
    initial_snapshot = solve_source.index("targets.initial_joint_q[:frame_count].copy_(joint_q)", seed_validation)
    gauge = solve_source.index("_motion_target_ground_gauge", initial_snapshot)
    witness = solve_source.index("_motion_scalar_velocity_box_witness", gauge)
    contact_targets = solve_source.index("prepare_contact_targets(frame_count, clip_offsets, phase_active)", witness)
    weights = solve_source.index("_motion_monolithic_weights(", contact_targets)
    solve_call = solve_source.index(monolithic_call, weights)
    finish = solve_source.index("_motion_phase_finish(", solve_call)
    final_quality = solve_source.rindex("_trajectory_recompute_clip_quality(")
    early_inactive_return = phase_solve_source.index("if not bool(torch.any(phase_active)):")

    assert early_inactive_return < phase_solve_source.index("solver.solve(")
    assert seed_validation < initial_snapshot < gauge < witness < contact_targets
    assert contact_targets < weights < solve_call < finish < final_quality
    assert "frozen_dof_indices=root_dof_indices if source_root_fixed else None" in monolithic_call
    assert "inequalities=None" in monolithic_call
    assert "velocity_bounds=(workspace.source_velocity_lower, workspace.source_velocity_upper)" in monolithic_call
    assert "restore_initial_root=source_root_fixed" in monolithic_call
    assert "residual_activity=residual_activity" in monolithic_call
    assert "terminal_acceptance=" not in monolithic_call
    assert "adaptive_recovery=" not in monolithic_call
    assert "source_inequalities" not in solve_source
    assert "_motion_phase_copy_selected(" not in solve_source
    assert "feasibility_only=True" not in solve_source
    assert "ResidualEqualities" not in solve_source
    assert "contact_equality" not in solve_source
    assert "contact_seed" not in solve_source
    assert "max_equality_residuals_per_frame=0" in solve_source
    assert (
        "first_difference_group_by_residual=workspace.residual_layout.first_difference_group_by_residual"
    ) in solve_source
    assert "joint_q[:, :7].copy_(targets.initial_joint_q[:frame_count, :7])" in phase_solve_source
    assert "maximum_local_frames = max(" in solve_source
    assert solve_source.count("targets.initial_joint_q[:frame_count].copy_(joint_q)") == 1


def test_trajectory_phase_outcomes_keep_failure_axes_independent() -> None:
    """Geometry, inner-solve, globalization, and nonlinear outcomes retain distinct truth tables."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import (
        _motion_phase_finish,
        _motion_phase_update,
    )

    active = torch.tensor([1, 1, 1, 0], dtype=torch.int32)
    geometry_feasible = torch.ones(4, dtype=torch.bool)
    inner_converged = torch.ones(4, dtype=torch.bool)
    globalization_succeeded = torch.ones(4, dtype=torch.bool)
    nonlinear_converged = torch.ones(4, dtype=torch.bool)
    iteration_geometry_feasible = torch.tensor([True, False, True, True])
    iteration_inner_converged = torch.tensor([False, True, True, True])
    iteration_globalization_succeeded = torch.tensor([True, True, False, True])
    phase_converged = torch.empty(4, dtype=torch.bool)

    _motion_phase_update(
        active,
        geometry_feasible,
        inner_converged,
        globalization_succeeded,
        iteration_geometry_feasible,
        iteration_inner_converged,
        iteration_globalization_succeeded,
    )
    _motion_phase_finish(
        active,
        geometry_feasible,
        inner_converged,
        globalization_succeeded,
        nonlinear_converged,
        phase_converged,
    )

    assert active.tolist() == [0, 0, 0, 0]
    assert geometry_feasible.tolist() == [True, False, True, True]
    assert inner_converged.tolist() == [False, True, True, True]
    assert globalization_succeeded.tolist() == [True, True, False, True]
    assert nonlinear_converged.tolist() == [False, False, False, True]


def test_invalid_seed_masks_remain_false_after_phase_finish() -> None:
    """Inactive invalid seeds cannot become accepted through phase finalization."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import _motion_phase_finish

    active = torch.zeros(1, dtype=torch.int32)
    geometry_feasible = torch.zeros(1, dtype=torch.bool)
    inner_converged = torch.zeros(1, dtype=torch.bool)
    globalization_succeeded = torch.zeros(1, dtype=torch.bool)
    nonlinear_converged = torch.zeros(1, dtype=torch.bool)
    phase_converged = torch.empty(1, dtype=torch.bool)

    _motion_phase_finish(
        active,
        geometry_feasible,
        inner_converged,
        globalization_succeeded,
        nonlinear_converged,
        phase_converged,
    )

    assert not geometry_feasible.item()
    assert not inner_converged.item()
    assert not globalization_succeeded.item()
    assert not nonlinear_converged.item()
    assert not phase_converged.item()


def test_trajectory_requires_each_phase_to_converge_without_conflating_failure_axes() -> None:
    """Source convergence gates contact and contact convergence remains independently mandatory."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import _motion_phase_finish

    # Rows: both converge, contact reaches cap, source reaches cap, geometry failure,
    # inner-solve failure, globalization failure, and invalid frame seed.
    active = torch.tensor([0, 0, 1, 0, 0, 0, 0], dtype=torch.int32)
    geometry_feasible = torch.tensor([True, True, True, False, True, True, False])
    inner_converged = torch.tensor([True, True, True, True, False, True, False])
    globalization_succeeded = torch.tensor([True, True, True, True, True, False, False])
    nonlinear_converged = torch.tensor([True, True, True, True, True, True, False])
    phase_converged = torch.empty(7, dtype=torch.bool)

    _motion_phase_finish(
        active,
        geometry_feasible,
        inner_converged,
        globalization_succeeded,
        nonlinear_converged,
        phase_converged,
    )
    assert phase_converged.tolist() == [True, True, False, False, False, False, False]
    assert nonlinear_converged.tolist() == [True, True, False, False, False, False, False]
    assert geometry_feasible.tolist() == [True, True, True, False, True, True, False]
    assert inner_converged.tolist() == [True, True, True, True, False, True, False]
    assert globalization_succeeded.tolist() == [True, True, True, True, True, False, False]

    active.copy_(phase_converged)
    active[0] = 0
    _motion_phase_finish(
        active,
        geometry_feasible,
        inner_converged,
        globalization_succeeded,
        nonlinear_converged,
        phase_converged,
    )
    assert nonlinear_converged.tolist() == [True, False, False, False, False, False, False]


def test_contact_height_alignment_is_one_rigid_gauge_per_eligible_clip() -> None:
    """The signed proposal is clip-rigid and leaves source evidence immutable."""
    import warp as wp

    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import (
        _motion_contact_align_clip_height,
    )

    wp.init()

    frame_count = 6
    obstacle_pose = torch.zeros((frame_count, 7), dtype=torch.float32)
    obstacle_pose[:, 6] = 1.0
    body_q = torch.zeros((frame_count, 1, 7), dtype=torch.float32)
    body_q[:, 0, 2] = 1.0
    body_q[:, 0, 6] = 1.0
    probe_bodies = torch.zeros(1, dtype=torch.int64)
    probe_offsets = torch.zeros((1, 3), dtype=torch.float32)
    support_channel_slots = torch.tensor([0], dtype=torch.int64)
    stable = torch.zeros((frame_count, 1), dtype=torch.uint8)
    stable[:3, 0] = torch.tensor((1, 0, 1), dtype=torch.uint8)
    clip_offsets = torch.tensor([0, 3, 6], dtype=torch.int32)
    segment_active = torch.tensor([1, 0], dtype=torch.int32)
    joint_q = torch.zeros((frame_count, 7), dtype=torch.float32)
    joint_q[:, 2] = torch.tensor((0.0, 0.1, 0.3, 1.0, 1.2, 1.5))
    original_root_height = joint_q[:, 2].clone()
    target_support = torch.zeros((1, frame_count, 3), dtype=torch.float32)
    target_support[0, :, 2] = torch.tensor((0.2, 1.0, 0.4, 0.5, 0.6, 0.7))
    original_support_height = target_support[0, :, 2].clone()

    wp.launch(
        _motion_contact_align_clip_height,
        dim=segment_active.shape[0],
        inputs=[
            wp.from_torch(obstacle_pose),
            wp.from_torch(body_q),
            wp.from_torch(probe_bodies),
            wp.from_torch(probe_offsets),
            wp.from_torch(support_channel_slots),
            wp.from_torch(stable),
            wp.from_torch(clip_offsets),
            wp.from_torch(segment_active),
            segment_active.shape[0],
            1,
            1,
            0.0,
            1,
        ],
        outputs=[
            wp.from_torch(joint_q),
            wp.from_torch(target_support),
        ],
        device="cpu",
    )

    expected_shift = torch.tensor((-0.3, -0.3, -0.3, 0.0, 0.0, 0.0))
    torch.testing.assert_close(joint_q[:, 2] - original_root_height, expected_shift)
    torch.testing.assert_close(target_support[0, :, 2] - original_support_height, expected_shift)
    torch.testing.assert_close(joint_q[1:3, 2] - joint_q[:2, 2], original_root_height[1:3] - original_root_height[:2])


def test_contact_height_alignment_starts_contact_refinement_clear_of_ground() -> None:
    """The clip gauge must not leave a tilted target support patch below its obstacle plane."""
    import warp as wp

    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import (
        _motion_contact_align_clip_height,
    )

    wp.init()
    obstacle_pose = torch.zeros((3, 7), dtype=torch.float32)
    obstacle_pose[:, 6] = 1.0
    body_q = torch.zeros((3, 1, 7), dtype=torch.float32)
    body_q[:, 0, 2] = -0.4
    body_q[:, 0, 6] = 1.0
    probe_bodies = torch.zeros(1, dtype=torch.int64)
    probe_offsets = torch.zeros((1, 3), dtype=torch.float32)
    support_channel_slots = torch.zeros(3, dtype=torch.int64)
    stable = torch.ones((3, 1), dtype=torch.uint8)
    clip_offsets = torch.tensor((0, 3), dtype=torch.int32)
    segment_active = torch.ones(1, dtype=torch.int32)
    joint_q = torch.zeros((3, 7), dtype=torch.float32)
    target_support = torch.zeros((3, 3, 3), dtype=torch.float32)
    target_support[:, :, 2] = torch.tensor((-0.4, 0.2, 0.2))[:, None]

    wp.launch(
        _motion_contact_align_clip_height,
        dim=1,
        inputs=[
            wp.from_torch(obstacle_pose),
            wp.from_torch(body_q),
            wp.from_torch(probe_bodies),
            wp.from_torch(probe_offsets),
            wp.from_torch(support_channel_slots),
            wp.from_torch(stable),
            wp.from_torch(clip_offsets),
            wp.from_torch(segment_active),
            1,
            1,
            3,
            0.0,
            1,
        ],
        outputs=[
            wp.from_torch(joint_q),
            wp.from_torch(target_support),
        ],
        device="cpu",
    )

    assert torch.min(target_support[:, :, 2]) >= 0.0


def test_trajectory_solver_cfg_exposes_one_whole_clip_policy() -> None:
    """The trajectory policy owns contact, temporal, damping, and bounded solve choices."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionTrajectorySolveCfg

    cfg = MotionTrajectorySolveCfg()
    fields = {field.name for field in dataclasses.fields(cfg)}

    assert cfg.max_iterations == 200
    assert cfg.convergence_tolerance == 1.0e-6
    assert cfg.convergence_check_interval == 1
    assert cfg.krylov_max_iterations == 128
    assert cfg.krylov_relative_tolerance == 1.0e-4
    assert cfg.contact.enter_height_m <= cfg.contact.exit_height_m
    assert cfg.contact.enter_speed_mps <= cfg.contact.exit_speed_mps
    assert "phases" not in fields
    assert {
        "acceptance",
        "contact",
        "source_position_velocity_weight",
        "source_position_acceleration_weight",
        "source_rotation_velocity_weight",
        "source_rotation_acceleration_weight",
        "joint_default_position_weight",
        "joint_temporal_velocity_weight",
        "joint_temporal_acceleration_weight",
        "joint_temporal_jerk_weight",
    } <= fields
    assert (
        not {
            "joint_reference_velocity_weight",
            "joint_reference_acceleration_weight",
            "joint_reference_jerk_weight",
        }
        & fields
    )
    assert cfg.joint_default_position_weight == 0.0025
    from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.cfg import (
        IKObjectiveJointDefaultCfg,
        IKObjectiveJointPinCfg,
    )

    for term_type in (IKObjectiveJointDefaultCfg, IKObjectiveJointPinCfg):
        scaled_objectives = tuple(
            dataclasses.replace(objective, weight=0.5) if type(objective) is term_type else objective
            for objective in cfg.objectives
        )
        with pytest.raises(ValueError, match="unit weight"):
            MotionTrajectorySolveCfg(objectives=scaled_objectives)
    assert MotionTrajectorySolveCfg(convergence_check_interval=2).convergence_check_interval == 2
    with pytest.raises(ValueError, match="convergence tolerance"):
        MotionTrajectorySolveCfg(convergence_tolerance=None)
    with pytest.raises(ValueError, match="convergence tolerance"):
        MotionTrajectorySolveCfg(convergence_tolerance=-1.0)
    with pytest.raises(ValueError, match="max_iterations"):
        MotionTrajectorySolveCfg(max_iterations=0)
    with pytest.raises(ValueError, match="check interval"):
        MotionTrajectorySolveCfg(convergence_check_interval=0)


def test_trajectory_solver_reads_canonical_topology_velocity_bounds() -> None:
    """Motion trajectory solving must consume the shared scene-mechanics limit owner."""
    trajectory_path = _MOTION_ROOT / "mdp" / "commands" / "motion_trajectory.py"
    source = trajectory_path.read_text(encoding="utf-8")

    assert "reference.topology.joint_velocity_lower" in source
    assert "reference.topology.joint_velocity_upper" in source
    assert "model.joint_velocity_limit" not in source


def test_trajectory_objectives_are_fully_declared_at_the_config_boundary() -> None:
    """The runtime must neither inject hidden objectives nor depend on tuple positions."""
    from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.cfg import (
        IKObjectiveJointPinCfg,
        IKObjectiveMeshCollisionCfg,
    )
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionTrajectorySolveCfg

    cfg = MotionTrajectorySolveCfg()
    assert sum(isinstance(objective, IKObjectiveJointPinCfg) for objective in cfg.objectives) == 1
    assert sum(isinstance(objective, IKObjectiveMeshCollisionCfg) for objective in cfg.objectives) == 1

    trajectory_path = _MOTION_ROOT / "mdp" / "commands" / "motion_trajectory.py"
    tree = ast.parse(trajectory_path.read_text(encoding="utf-8"), filename=str(trajectory_path))
    imported_names = {
        alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names
    }
    assert "IKObjectiveJointPin" not in imported_names
    solve = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "motion_solve_trajectory"
    )
    assert not any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "cfg"
        and node.value.attr == "objectives"
        and isinstance(node.slice, ast.Constant)
        for node in ast.walk(solve)
    )


def test_trajectory_collision_reuses_declared_contact_probe_ownership() -> None:
    """Motion binds target-owned probe geometry and continuous confidence to generic soft collision."""
    trajectory_path = _MOTION_ROOT / "mdp" / "commands" / "motion_trajectory.py"
    source = trajectory_path.read_text(encoding="utf-8")

    assert (
        "probe_bodies, probe_offsets, probe_contact_slots, probe_normal_slots = validate_collision_probe_geometry("
        in source
    )
    assert "probe_contact_slots_np = probe_contact_slots.detach().cpu().numpy()" in source
    assert "collision_probes_sample(" not in source
    assert "probe_contact_slots=probe_contact_slots_np" in source
    assert "contact_confidence=contact_confidence" in source
    assert "source_channel_confidence = torch.empty(" in source
    assert "source_probe_active=torch.empty(" in source
    assert "source_probe_stable=torch.empty(" in source
    assert "source_channel_stable=torch.empty(" in source
    assert "source_channel_edge_stable=torch.empty(" in source
    assert "probe_contact_slots=None" not in source
    assert "contact_confidence=None" not in source
    assert "collision_contact_mask" not in source


def test_motion_collision_declares_soft_and_nonpenetration_objectives() -> None:
    """Anticipatory shaping and all-probe nonpenetration have separate declared owners."""
    from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.cfg import (
        IKObjectiveMeshCollisionCfg,
        IKObjectiveMeshNonpenetrationCfg,
    )
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionTrajectorySolveCfg

    cfg = MotionTrajectorySolveCfg()
    assert sum(isinstance(item, IKObjectiveMeshCollisionCfg) for item in cfg.objectives) == 1
    assert sum(isinstance(item, IKObjectiveMeshNonpenetrationCfg) for item in cfg.objectives) == 1
    assert not hasattr(cfg, "constraints")
    trajectory_path = _MOTION_ROOT / "mdp" / "commands" / "motion_trajectory.py"
    source = trajectory_path.read_text(encoding="utf-8")
    assert "collision_objective" in source
    assert "nonpenetration_objective" in source
    assert "IKConstraintMeshClearanceCfg" not in source
    assert "clearance_cfg" not in source
    assert "physical_inequalities" not in source


def test_trajectory_contact_geometry_activates_nonpenetration() -> None:
    """The physical/contact objective activates ungated nonpenetration rows."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionTrajectorySolveCfg
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import (
        _motion_contact_weights,
        _motion_source_weights,
        _MotionTrajectoryResidualLayout,
    )

    cfg = MotionTrajectorySolveCfg()
    layout = _MotionTrajectoryResidualLayout(
        source_global_position=slice(0, 6),
        source_rotation=slice(6, 9),
        source_direction_point=slice(9, 12),
        source_fidelity_guard=slice(27, 31),
        contact=slice(12, 19),
        activity_group_by_residual=torch.full((31,), -1, dtype=torch.int32),
        first_difference_group_by_residual=torch.full((31,), -1, dtype=torch.int32),
        joint_default=slice(19, 21),
        joint_reference=slice(21, 23),
        collision_objective=slice(23, 25),
        nonpenetration_objective=slice(25, 27),
    )
    targets = SimpleNamespace(required_position_rows=(0,), required_direction_rows=(0,), support_patch_offsets=(0, 1))
    base = torch.empty(layout.residual_count)
    temporal = torch.empty((3, layout.residual_count))

    _motion_source_weights(layout, cfg, targets, base, temporal)
    assert torch.all(base[layout.joint_default] == cfg.joint_default_position_weight)
    assert torch.count_nonzero(base[layout.contact]) == 0
    assert torch.count_nonzero(temporal[:, layout.contact]) == 0
    assert torch.count_nonzero(base[layout.nonpenetration_objective]) == 0

    _motion_contact_weights(layout, cfg, targets, base, temporal)
    torch.testing.assert_close(base[layout.contact], torch.tensor((1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0)))
    expected = (cfg.contact.point_tolerance_m / cfg.acceptance.contact.slip_speed_upper_mps) ** 2
    assert torch.all(temporal[0, layout.contact.start + 4 : layout.contact.stop] == expected)
    assert torch.count_nonzero(temporal[1:, layout.contact]) == 0
    assert torch.all(base[layout.joint_reference] == 1.0)
    assert torch.all(base[layout.collision_objective] == 1.0)
    assert torch.all(base[layout.nonpenetration_objective] == 1.0)


def test_trajectory_residual_layout_orders_soft_contact_rows_by_channel() -> None:
    """Each contact channel maps vertex rows and point edge rows to independent activity groups."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionTrajectorySolveCfg
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import _motion_trajectory_residual_layout

    targets = SimpleNamespace(
        position_body_indices=(10, 11, 12),
        rotation_body_indices=(20, 21),
        direction_body_indices=(25, 26),
        source_landmark_position_m=torch.zeros(3, 1, 3),
        support_patch_offsets=(0, 2, 3),
        coordinate_indices=torch.arange(4),
        required_position_rows=(0,),
        required_direction_rows=(0,),
    )
    layout = _motion_trajectory_residual_layout(targets, 7, MotionTrajectorySolveCfg().objectives)

    assert layout.source_global_position == slice(0, 9)
    assert layout.source_rotation == slice(9, 15)
    assert layout.source_direction_point == slice(15, 21)
    assert layout.contact == slice(21, 38)
    torch.testing.assert_close(layout.activity_group_by_residual[21:31], torch.zeros(10, dtype=torch.int32))
    torch.testing.assert_close(layout.activity_group_by_residual[31:38], torch.ones(7, dtype=torch.int32))
    torch.testing.assert_close(layout.first_difference_group_by_residual[21:25], -torch.ones(4, dtype=torch.int32))
    torch.testing.assert_close(layout.first_difference_group_by_residual[25:31], torch.full((6,), 2, dtype=torch.int32))
    torch.testing.assert_close(layout.first_difference_group_by_residual[31:35], -torch.ones(4, dtype=torch.int32))
    torch.testing.assert_close(layout.first_difference_group_by_residual[35:38], torch.full((3,), 3, dtype=torch.int32))
    assert layout.joint_default == slice(38, 42)
    assert layout.joint_reference == slice(42, 46)
    assert layout.collision_objective == slice(46, 53)
    assert layout.nonpenetration_objective == slice(53, 60)
    assert layout.source_fidelity_guard == slice(60, 64)
    assert layout.residual_count == 64


def test_trajectory_residual_layout_supports_interleaved_nonpenetration() -> None:
    """Nonpenetration rows follow declaration order without changing appended source guards."""
    from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.cfg import IKObjectiveMeshNonpenetrationCfg
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionTrajectorySolveCfg
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import _motion_trajectory_residual_layout

    targets = SimpleNamespace(
        position_body_indices=(10, 11, 12),
        rotation_body_indices=(20,),
        direction_body_indices=(25, 26),
        source_landmark_position_m=torch.zeros(3, 1, 3),
        support_patch_offsets=(0, 2, 3),
        coordinate_indices=torch.arange(4),
        required_position_rows=(0,),
        required_direction_rows=(0,),
    )
    terms = MotionTrajectorySolveCfg().objectives
    nonpenetration = next(term for term in terms if isinstance(term, IKObjectiveMeshNonpenetrationCfg))
    other_terms = tuple(term for term in terms if term is not nonpenetration)
    interleaved = (*other_terms[:4], nonpenetration, *other_terms[4:])
    layout = _motion_trajectory_residual_layout(targets, 7, interleaved)

    assert layout.contact == slice(18, 35)
    assert layout.nonpenetration_objective == slice(35, 42)
    assert layout.joint_default == slice(42, 46)
    assert layout.joint_reference == slice(46, 50)
    assert layout.collision_objective == slice(50, 57)
    assert layout.source_fidelity_guard == slice(57, 61)
    assert layout.residual_count == 61


def _solved_candidate(
    trajectory_quality: torch.Tensor,
    constraint_geometry_feasible: torch.Tensor,
    inner_solve_converged: torch.Tensor,
    nonlinear_refinement_required: torch.Tensor,
    nonlinear_phases_converged: torch.Tensor,
):
    """Build one complete solved-stage record for criterion tests."""
    from isaaclab_tasks.core.multi_task.motion.data.frames import MotionGeneralizedCoordinates
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionTrajectorySolveCfg
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table_builder import (
        _MotionTrajectorySolvedCandidate,
    )

    clip_count = trajectory_quality.shape[0]
    acceptance = MotionTrajectorySolveCfg().acceptance
    return _MotionTrajectorySolvedCandidate(
        target=object(),
        clip_index=SimpleNamespace(clips=(object(),) * clip_count),
        coordinates=MotionGeneralizedCoordinates(torch.zeros((clip_count, 1)), None),
        trajectory_quality=trajectory_quality,
        target_coordinate_evidence=torch.zeros(
            (clip_count, len(_TARGET_COORDINATE_QUALITY_NAMES)), dtype=torch.float32
        ),
        constraint_geometry_feasible=constraint_geometry_feasible,
        inner_solve_converged=inner_solve_converged,
        nonlinear_refinement_required=nonlinear_refinement_required,
        nonlinear_phases_converged=nonlinear_phases_converged,
        acceptance=acceptance,
        device="cpu",
        contact_evidence=None,
        view_evidence=None,
    )


def test_trajectory_acceptance_uses_hard_state_while_retaining_health_diagnostics() -> None:
    """Geometry and refinement are gates while inner-solve health remains diagnostic evidence."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import (
        MotionConstraintGeometryFeasibleCriterionCfg,
        MotionInnerSolveConvergedCriterionCfg,
        MotionRequiredRefinementConvergedCriterionCfg,
        MotionTrajectoryFamilyCfg,
    )
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import (
        motion_criterion_constraint_geometry_feasible,
        motion_criterion_inner_solve_converged,
        motion_criterion_required_refinement_converged,
    )

    candidate = _solved_candidate(
        torch.zeros((3, 5)),
        torch.tensor((True, False, True)),
        torch.tensor((False, True, True)),
        torch.tensor((False, True, True)),
        torch.tensor((False, False, True)),
    )
    assert not hasattr(candidate, "constraint_feasible")
    criteria = MotionTrajectoryFamilyCfg().criteria
    geometry_criterion = next(
        item for item in criteria if isinstance(item, MotionConstraintGeometryFeasibleCriterionCfg)
    )
    assert all(not isinstance(item, MotionInnerSolveConvergedCriterionCfg) for item in criteria)
    inner_criterion = MotionInnerSolveConvergedCriterionCfg()
    nonlinear_criterion = next(
        item for item in criteria if isinstance(item, MotionRequiredRefinementConvergedCriterionCfg)
    )
    rows = torch.tensor((2, 1, 0))

    geometry_accepted = motion_criterion_constraint_geometry_feasible(geometry_criterion, candidate, rows)
    inner_accepted = motion_criterion_inner_solve_converged(inner_criterion, candidate, rows)
    nonlinear_accepted = motion_criterion_required_refinement_converged(nonlinear_criterion, candidate, rows)

    torch.testing.assert_close(geometry_accepted, torch.tensor((True, False, True)))
    torch.testing.assert_close(inner_accepted, torch.tensor((True, True, False)))
    torch.testing.assert_close(nonlinear_accepted, torch.tensor((True, False, True)))


def test_publishability_source_fidelity_gates_only_target_required_metrics() -> None:
    """Required target observables are hard while all-row morphology residuals remain diagnostic."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import (
        MotionSourceFidelityCriterionCfg,
        MotionTrajectorySolveCfg,
    )
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import _TRAJECTORY_METRIC_NAMES
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import motion_criterion_source_fidelity

    criterion = MotionSourceFidelityCriterionCfg()
    acceptance = MotionTrajectorySolveCfg().acceptance
    policy = acceptance.source
    assert (
        policy.required_position_upper_m,
        policy.required_distal_position_upper_m,
        policy.required_distal_direction_upper_rad,
        policy.root_rotation_upper_rad,
    ) == (0.020, 0.030, 0.100, 0.100)
    assert vars(criterion) == {"class_type": criterion.class_type}
    hard_names = (
        "source_required_position_max_m",
        "source_required_distal_position_max_m",
        "source_required_distal_direction_max_rad",
        "source_root_rotation_max_rad",
    )
    diagnostic_names = (
        "source_all_position_max_m",
        "source_all_distal_position_max_m",
        "source_all_landmark_direction_max_rad",
        "source_all_distal_direction_max_rad",
        "source_nonroot_rotation_max_rad",
    )
    assert _TRAJECTORY_METRIC_NAMES[:9] == (*hard_names, *diagnostic_names)
    hard_columns = tuple(_TRAJECTORY_METRIC_NAMES.index(name) for name in hard_names)
    diagnostic_columns = tuple(_TRAJECTORY_METRIC_NAMES.index(name) for name in diagnostic_names)
    quality = torch.zeros((9, len(_TRAJECTORY_METRIC_NAMES)), dtype=torch.float32)
    quality[:, hard_columns] = torch.tensor((0.020, 0.030, 0.100, 0.100))
    quality[0, diagnostic_columns] = float("nan")
    quality[1, diagnostic_columns] = float("inf")
    for row, column in enumerate(hard_columns, start=2):
        quality[row, column] += 1.0e-3
    quality[6, hard_columns[2]] = float("nan")
    quality[7, hard_columns[2]] = float("inf")
    quality[8, hard_columns[0]] = -1.0

    candidate = SimpleNamespace(trajectory_quality=quality, acceptance=acceptance)
    accepted = motion_criterion_source_fidelity(criterion, candidate, torch.arange(9))

    torch.testing.assert_close(
        accepted,
        torch.tensor((True, True, False, False, False, False, False, False, False)),
    )


def test_publishability_contact_gate_has_explicit_honest_na_semantics() -> None:
    """Contact N/A is explicit while every applicable physical maximum is bounded."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import (
        MotionContactCriterionCfg,
        MotionTrajectorySolveCfg,
    )
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import _TRAJECTORY_METRIC_NAMES
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import motion_criterion_contact

    criterion = MotionContactCriterionCfg()
    acceptance = MotionTrajectorySolveCfg().acceptance
    policy = acceptance.contact
    assert (
        policy.gap_upper_m,
        policy.tilt_upper_rad,
        policy.slip_speed_upper_mps,
        policy.cumulative_drift_upper_m,
    ) == (0.010, 0.100, 0.050, 0.020)
    assert vars(criterion) == {"class_type": criterion.class_type}
    metrics = tuple(
        _TRAJECTORY_METRIC_NAMES.index(name)
        for name in (
            "contact_gap_max_m",
            "contact_tilt_max_rad",
            "contact_slip_speed_max_mps",
            "contact_cumulative_drift_max_m",
        )
    )
    applicable = _TRAJECTORY_METRIC_NAMES.index("contact_applicable")
    count = _TRAJECTORY_METRIC_NAMES.index("contact_stable_frame_channel_count")
    confidence = _TRAJECTORY_METRIC_NAMES.index("source_contact_confidence_mean")
    quality = torch.zeros((15, len(_TRAJECTORY_METRIC_NAMES)), dtype=torch.float32)
    quality[:, metrics] = torch.tensor((0.010, 0.100, 0.050, 0.020))
    quality[:, applicable] = 1.0
    quality[:, count] = 1.0
    quality[:, confidence] = 0.5
    for row, column in enumerate(metrics, start=1):
        quality[row, column] += 1.0e-3
    quality[5, metrics] = float("nan")
    quality[5, applicable] = 0.0
    quality[5, count] = 0.0
    quality[6, count] = 0.0
    quality[7, metrics[0]] = float("nan")
    quality[8, confidence] = 0.0
    quality[9, confidence] = 1.0
    quality[10, applicable] = 0.0
    quality[10, count] = 0.0
    quality[11, applicable] = 2.0
    quality[12, count] = 1.5
    quality[13, confidence] = float("nan")
    quality[14, metrics] = float("nan")
    quality[14, applicable] = 0.0
    quality[14, count] = -1.0

    candidate = SimpleNamespace(trajectory_quality=quality, acceptance=acceptance)
    accepted = motion_criterion_contact(criterion, candidate, torch.arange(15))

    torch.testing.assert_close(
        accepted,
        torch.tensor(
            (True, False, False, False, False, True, False, False, True, True, False, False, False, False, False)
        ),
    )
    no_contact = SimpleNamespace(trajectory_quality=quality[5:6], acceptance=acceptance)
    assert not motion_criterion_contact(criterion, no_contact, torch.tensor((0,)))[0]
    optional_contact = MotionTrajectorySolveCfg.AcceptanceCfg(
        contact=MotionTrajectorySolveCfg.AcceptanceCfg.ContactCfg(require_any_stable_contact=False)
    )
    no_contact = SimpleNamespace(trajectory_quality=quality[5:6], acceptance=optional_contact)
    assert motion_criterion_contact(criterion, no_contact, torch.tensor((0,)))[0]


def test_motion_target_factory_is_outside_route_execution() -> None:
    """The exact public builder delegates to a core with one target outside route loops."""
    public_path = _MOTION_ROOT / "mdp" / "commands" / "motion_task_table.py"
    builder_path = _MOTION_ROOT / "mdp" / "commands" / "motion_task_table_builder.py"
    public_tree = ast.parse(public_path.read_text(encoding="utf-8"), filename=str(public_path))
    builder_tree = ast.parse(builder_path.read_text(encoding="utf-8"), filename=str(builder_path))
    public = next(
        node
        for node in public_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_motion_task_table"
    )
    core = next(
        node
        for node in builder_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_motion_task_table"
    )
    calls = [
        node
        for node in ast.walk(core)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "target_factory"
    ]
    loop_nodes = {node for loop in ast.walk(core) if isinstance(loop, ast.For) for node in ast.walk(loop)}

    assert [argument.arg for argument in public.args.args] == ["command_cfg", "scene_cfg", "device"]
    assert not public.args.kwonlyargs
    assert len(calls) == 1
    assert calls[0] not in loop_nodes


def test_motion_route_production_fails_closed_while_inspection_retains_rejected_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production rejects any failed clip while inspection retains every route candidate and its evidence."""
    from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex
    from isaaclab_tasks.core.multi_task.motion.data.frames import (
        MotionGeneralizedCoordinates,
        MotionSourceProjectionExact,
    )
    from isaaclab_tasks.core.multi_task.motion.mdp.commands import motion_task_table_builder as module

    source_sha = _sha256("route-source")
    family_sha = _sha256("route-family")
    family = object()

    def group(skeleton_id: int, source_indices: tuple[int, ...]):
        skeleton_sha = _sha256(f"skeleton-{skeleton_id}")
        projection = MotionSourceProjectionExact(
            source_skeleton=SimpleNamespace(identity_sha256=skeleton_sha),
            target=object(),
            version="test_exact_v1",
            construction_identity_sha256=_sha256(f"projection-{skeleton_id}"),
            convert_coordinates=lambda joint_q, joint_qd, source_fps: MotionGeneralizedCoordinates(joint_q, joint_qd),
        )
        index = MotionClipIndex(
            source_content_sha256=source_sha,
            skeleton_identity_sha256s=(skeleton_sha,),
            clips=tuple(
                MotionClipIndex.Clip(f"clip-{index}", 1, 30.0, _sha256(f"clip-{index}"), 0) for index in source_indices
            ),
        )
        return module._MotionGroupPlan(
            skeleton_id=skeleton_id,
            source_clip_indices=source_indices,
            source_index=index,
            output_index=index,
            projection=projection,
            family=family,
            family_name="exact",
            family_identity_sha256=family_sha,
        )

    route = module._compose_motion_routes((group(0, (0, 2)), group(1, (1, 3))))[0]
    calls = []

    def execute(_family, candidate, _criteria, _rng):
        calls.append(type(candidate))
        assert not hasattr(candidate, "retain_source_position")
        assert candidate.source_clip_indices == (0, 2, 1, 3)
        assert candidate.projection_indices == (0, 0, 1, 1)
        candidate.coordinates.joint_q.zero_()
        built = module._MotionCoordinateCandidate(
            target=candidate.target,
            clip_index=candidate.output_index,
            coordinates=candidate.coordinates,
            target_coordinate_evidence=torch.zeros((4, len(_TARGET_COORDINATE_QUALITY_NAMES)), dtype=torch.float32),
            device=candidate.device,
        )
        return SimpleNamespace(
            candidates=built,
            accepted_mask=torch.tensor((True, False, True, False)),
            selected_indices=None,
        )

    class Target:
        def allocate_coordinates(self, frame_count: int, *, device: str) -> MotionGeneralizedCoordinates:
            return MotionGeneralizedCoordinates(torch.empty((frame_count, 2), device=device), None)

    monkeypatch.setattr(module, "execute_task_family", execute)
    with pytest.raises(ValueError, match="rejected clips before publication") as error:
        module._build_motion_route(route, Target(), object(), object(), "cpu")
    inspected = module._build_motion_route(route, Target(), object(), object(), "cpu", inspection=True)

    message = str(error.value)
    assert "route_indices=(1, 3)" in message
    assert "source_indices=(2, 3)" in message
    assert "clip_ids=('clip-2', 'clip-3')" in message
    assert calls == [module._MotionExactSourceCandidate, module._MotionExactSourceCandidate]
    assert inspected.output_source_indices == (0, 2, 1, 3)
    assert inspected.output_skeleton_ids == (0, 0, 1, 1)
    torch.testing.assert_close(inspected.quality[:, 1], torch.tensor((1.0, 0.0, 1.0, 0.0)))
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import _QUALITY_NAMES

    for status_name in (
        "constraint_geometry_feasible",
        "inner_solve_converged",
        "nonlinear_refinement_required",
        "nonlinear_phases_converged",
    ):
        assert torch.isnan(inspected.quality[:, _QUALITY_NAMES.index(status_name)]).all()


def test_motion_corpus_finalizer_preserves_every_source_clip_and_acceptance() -> None:
    """Finalization restores source order without dropping or relabeling rejected inspection rows."""
    from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex
    from isaaclab_tasks.core.multi_task.motion.data.frames import MotionGeneralizedCoordinates
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table_builder import (
        _finish_motion_groups,
        _MotionStoredSequence,
    )

    identities = tuple(_sha256(f"skeleton-{index}") for index in range(3))
    index = MotionClipIndex(
        source_content_sha256=_sha256("complete-source"),
        skeleton_identity_sha256s=identities,
        clips=tuple(MotionClipIndex.Clip(f"clip-{item}", 1, 30.0, _sha256(f"clip-{item}"), item) for item in range(3)),
    )
    joint_q = torch.arange(6, dtype=torch.float32).view(3, 2)
    coordinate_bank = MotionGeneralizedCoordinates(joint_q.clone(), None)
    coordinate_scratch = MotionGeneralizedCoordinates(torch.empty((1, 2)), None)
    quality = torch.tensor(((10.0, 1.0, 11.0), (20.0, 0.0, 21.0), (30.0, 1.0, 31.0)))
    records = [
        _MotionStoredSequence(
            source_clip_index=source_index,
            order_in_source=0,
            skeleton_id=source_index,
            clip=index.clips[source_index],
            coordinate_start=source_index,
            coordinate_stop=source_index + 1,
            contact_evidence=None,
            view_evidence=None,
            quality=quality[source_index],
        )
        for source_index in (2, 0, 1)
    ]

    output_index, output_coordinates, output_quality, contact_evidence, view_evidence = _finish_motion_groups(
        index, coordinate_bank, coordinate_scratch, records
    )

    assert output_index == index
    torch.testing.assert_close(output_coordinates.joint_q, joint_q)
    torch.testing.assert_close(output_quality, quality)
    assert contact_evidence is None
    assert view_evidence is None
    assert records == []


def test_motion_manifest_rejects_identity_order_boundary_clock_and_provenance_changes() -> None:
    """The centralized manifest gate rejects every publication-scope mutation."""
    from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table_builder import (
        _validate_motion_manifest,
    )

    clips = (
        MotionClipIndex.Clip("first", 2, 30.0, _sha256("first"), 0, "source", 4),
        MotionClipIndex.Clip("second", 3, 60.0, _sha256("second"), 0, "source", 20),
    )
    expected = MotionClipIndex(_sha256("manifest"), (_sha256("skeleton"),), clips)
    mutations = (
        dataclasses.replace(expected, source_content_sha256=_sha256("other-manifest")),
        dataclasses.replace(expected, clips=(clips[1], clips[0])),
        dataclasses.replace(expected, clips=(dataclasses.replace(clips[0], clip_id="renamed"), clips[1])),
        dataclasses.replace(expected, clips=(dataclasses.replace(clips[0], frame_count=3), clips[1])),
        dataclasses.replace(expected, clips=(dataclasses.replace(clips[0], source_fps=24.0), clips[1])),
        dataclasses.replace(expected, clips=(dataclasses.replace(clips[0], source_frame_start=5), clips[1])),
    )

    for actual in mutations:
        with pytest.raises(ValueError, match="declared clip manifest"):
            _validate_motion_manifest(expected, actual)


def test_motion_trajectory_batch_remaps_only_its_used_skeletons() -> None:
    """A complete-clip batch keeps dense identities when it spans a subset of source mechanics."""
    from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import _motion_clip_batch_index

    identities = tuple(_sha256(f"batch-skeleton-{index}") for index in range(3))
    index = MotionClipIndex(
        source_content_sha256=_sha256("multi-skeleton-batch-source"),
        skeleton_identity_sha256s=identities,
        clips=tuple(
            MotionClipIndex.Clip(
                clip_id=f"clip-{clip_index}",
                frame_count=clip_index + 1,
                source_fps=30.0,
                content_sha256=_sha256(f"batch-clip-{clip_index}"),
                skeleton_id=skeleton_id,
            )
            for clip_index, skeleton_id in enumerate((0, 1, 2, 1))
        ),
    )

    batch = _motion_clip_batch_index(index, 1, 4)

    assert batch.skeleton_identity_sha256s == identities[1:]
    assert tuple(clip.skeleton_id for clip in batch.clips) == (0, 1, 0)
    assert batch.clip_ids == ("clip-1", "clip-2", "clip-3")
    assert batch.offsets == (0, 2, 5, 9)


def test_trajectory_stream_requires_target_static_tensor_reuse() -> None:
    """Every source projection reuses target-owned static trajectory tensor pointers."""
    from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import _MotionSourceEvidenceStream
    from isaaclab_tasks.core.multi_task.motion.retarget import MotionTrajectoryTargets

    index = MotionClipIndex(
        source_content_sha256=_sha256("pointer-source"),
        skeleton_identity_sha256s=(_sha256("pointer-skeleton"),),
        clips=(
            MotionClipIndex.Clip("first", 1, 30.0, _sha256("first-pointer"), 0),
            MotionClipIndex.Clip("second", 1, 30.0, _sha256("second-pointer"), 0),
        ),
    )
    body_indices = torch.tensor((0,), dtype=torch.int64)
    coordinate_indices = torch.tensor((7,), dtype=torch.int64)
    support_slots = torch.tensor((0,), dtype=torch.int64)
    targets = MotionTrajectoryTargets(
        position_body_indices=(0,),
        root_body_index=0,
        source_root_policy="optimized",
        initializer_policy="direct",
        parent_rows=(-1,),
        parent_row_tensor=torch.tensor((-1,), dtype=torch.int64),
        position_weights=(1.0,),
        required_position_rows=(0,),
        required_position_row_tensor=torch.tensor((0,), dtype=torch.int64),
        position_normal_channel_slots=torch.full((len(body_indices),), -1, dtype=torch.int64),
        position_body_index_tensor=body_indices,
        rotation_body_indices=(0,),
        rotation_weights=(10.0,),
        source_landmark_rotation_xyzw=torch.tensor((((0.0, 0.0, 0.0, 1.0),),)),
        direction_body_indices=(),
        direction_position_rows=(),
        direction_weights=(),
        contact_direction_rows=(),
        contact_direction_row_tensor=torch.empty(0, dtype=torch.int64),
        direction_contact_channel_slots=torch.empty(0, dtype=torch.int64),
        required_direction_rows=(),
        required_direction_row_tensor=torch.empty(0, dtype=torch.int64),
        direction_body_index_tensor=torch.empty(0, dtype=torch.int64),
        direction_position_row_tensor=torch.empty(0, dtype=torch.int64),
        direction_point_body_m=torch.empty((0, 3)),
        source_direction_point_position_m=torch.empty((0, 1, 3)),
        direction_length_values_m=(),
        source_landmark_position_m=torch.zeros((1, 1, 3)),
        initial_joint_q=torch.zeros((1, 8)),
        segment_lengths_m=torch.ones(1),
        segment_length_values_m=(1.0,),
        coordinate_indices=coordinate_indices,
        coordinate_lower_limits_rad=torch.full((1,), -1.0),
        coordinate_upper_limits_rad=torch.full((1,), 1.0),
        source_contact_probe_position_m=torch.zeros((1, 1, 3)),
        contact_channel_probe_offsets=torch.tensor((0, 1), dtype=torch.int32),
        target_support_position_m=torch.zeros((1, 1, 3)),
        contact_body_indices=body_indices,
        contact_normal_body=torch.tensor(((0.0, 0.0, 1.0),)),
        contact_forward_body=torch.tensor(((1.0, 0.0, 0.0),)),
        contact_distal_point_body_m=torch.zeros((1, 3)),
        leg_chain_body_indices=torch.zeros((1, 3), dtype=torch.int64),
        leg_chain_parent_body_indices=torch.zeros(1, dtype=torch.int64),
        leg_knee_hint_anatomy=torch.tensor(((1.0, 0.0, 0.0),)),
        leg_knee_hint_root=torch.tensor(((1.0, 0.0, 0.0),)),
        leg_segment_lengths_m=torch.ones((1, 2)),
        support_patch_offsets=(0, 1),
        support_body_indices=body_indices,
        support_point_body_m=torch.zeros((1, 3)),
        support_channel_slots=support_slots,
    )
    second = dataclasses.replace(
        targets,
        source_landmark_position_m=targets.source_landmark_position_m.clone(),
        source_landmark_rotation_xyzw=targets.source_landmark_rotation_xyzw.clone(),
        initial_joint_q=targets.initial_joint_q.clone(),
        source_contact_probe_position_m=targets.source_contact_probe_position_m.clone(),
        contact_channel_probe_offsets=targets.contact_channel_probe_offsets,
        target_support_position_m=targets.target_support_position_m.clone(),
    )
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import _motion_workspace_targets

    workspace = _motion_workspace_targets(targets, 7)
    assert not hasattr(workspace, "source_position_m")
    assert workspace.source_landmark_position_m.shape == (1, 7, 3)

    assert workspace.source_landmark_rotation_xyzw.shape == (1, 7, 4)
    stream = _MotionSourceEvidenceStream(iter(()), index, targets, None)
    stream._validate(targets)
    stream.clip = 1
    stream._validate(second)

    with pytest.raises(ValueError, match="landmark identity"):
        stream._validate(dataclasses.replace(second, position_weights=(2.0,)))
    with pytest.raises(ValueError, match="landmark identity"):
        stream._validate(dataclasses.replace(second, rotation_weights=(9.0,)))
    with pytest.raises(ValueError, match="landmark identity"):
        stream._validate(dataclasses.replace(second, direction_weights=(1.0,)))
    with pytest.raises(ValueError, match="landmark identity"):
        stream._validate(dataclasses.replace(second, source_root_policy="fixed"))
    with pytest.raises(ValueError, match="static tensor"):
        stream._validate(
            dataclasses.replace(second, contact_direction_row_tensor=torch.tensor((0,), dtype=torch.int64))
        )
    with pytest.raises(ValueError, match="static tensor"):
        stream._validate(
            dataclasses.replace(
                second,
                direction_contact_channel_slots=torch.tensor((0,), dtype=torch.int64),
            )
        )
    with pytest.raises(ValueError, match="static tensor"):
        stream._validate(dataclasses.replace(second, coordinate_indices=coordinate_indices.clone()))


def test_motion_task_table_has_no_obsolete_dense_or_statistics_paths() -> None:
    """Removed diagnostics and dense anchor construction must stay absent."""
    command_root = _MOTION_ROOT / "mdp" / "commands"
    source = "\n".join(
        (command_root / name).read_text(encoding="utf-8")
        for name in ("motion_task_table.py", "motion_task_table_builder.py", "motion_trajectory.py")
    )
    assert "solve_statistics" not in source
    assert "_trajectory_contact_anchor_grid" not in source


def test_motion_trajectory_peak_helpers_cover_later_largest_projection() -> None:
    """Peak arithmetic follows the largest later clip and the exact greedy batch layout."""
    from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import (
        _trajectory_coexisting_peak_bytes,
        _trajectory_max_batch_clips,
        _trajectory_projection_peak_bytes,
    )

    index = MotionClipIndex(
        source_content_sha256=_sha256("peak-source"),
        skeleton_identity_sha256s=(_sha256("peak-a"), _sha256("peak-b")),
        clips=(
            MotionClipIndex.Clip("short", 2, 30.0, _sha256("short"), 0),
            MotionClipIndex.Clip("largest-later", 11, 30.0, _sha256("largest-later"), 1),
        ),
    )
    candidate = SimpleNamespace(clip_index=index, source_body_counts=(3, 9))
    targets = SimpleNamespace(
        position_body_indices=(0, 1),
        root_body_index=0,
        direction_body_indices=(),
        target_support_position_m=torch.empty((3, 1, 3)),
        source_contact_probe_position_m=torch.empty((2, 1, 3)),
        coordinate_indices=torch.empty(5),
    )
    reference = SimpleNamespace(model=SimpleNamespace(joint_coord_count=12, joint_dof_count=11, body_count=7))
    peak = _trajectory_projection_peak_bytes(candidate, targets, reference)

    first_only = dataclasses.replace(
        index, clips=index.clips[:1], skeleton_identity_sha256s=index.skeleton_identity_sha256s[:1]
    )
    first_candidate = SimpleNamespace(clip_index=first_only, source_body_counts=(3,))
    assert peak == 16_016
    assert _trajectory_projection_peak_bytes(first_candidate, targets, reference) == 2_384
    assert _trajectory_max_batch_clips((6, 4, 3, 2), 9) == 3
    assert _trajectory_max_batch_clips((6, 4, 3, 2), 6) == 2
    assert _trajectory_coexisting_peak_bytes(1_000, (15_312, 3_968)) == 16_312
