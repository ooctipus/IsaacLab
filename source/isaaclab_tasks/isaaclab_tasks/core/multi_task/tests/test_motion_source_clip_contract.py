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

import pytest
import torch

from isaaclab_tasks.core.multi_task.motion.data import MotionSkeleton

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

        def semantic_local_pose(self, source_skeleton, *, device):
            del source_skeleton
            return torch.empty(3, 3, device=device), torch.empty(3, 3, 4, device=device)

    assert isinstance(Clip(), clip_type)
    for retired in ("MotionGeneralizedCoordinateClip", "MotionPoseAxisAngleClip", "MotionLocalBodyPoseClip"):
        assert not hasattr(source, retired)


@pytest.mark.parametrize(
    ("robot", "builder_class", "builder_factory"),
    (("g1", "G1FrameBuilder", "g1_frame_builder"), ("smpl", "SmplFrameBuilder", "smpl_frame_builder")),
)
def test_motion_robot_exports_one_frame_builder(
    robot: str,
    builder_class: str,
    builder_factory: str,
) -> None:
    """A robot owns one source-independent frame builder and one factory."""
    reference = _MOTION_ROOT / "robots" / robot / "reference.py"
    classes, functions = _top_level_symbols(reference)
    public_builders = {name for name in classes if not name.startswith("_") and name.endswith("FrameBuilder")}
    public_factories = {name for name in functions if name.endswith("_frame_builder")}

    assert public_builders == {builder_class}
    assert public_factories == {builder_factory}

    stub = (_MOTION_ROOT / "robots" / robot / "__init__.pyi").read_text(encoding="utf-8")
    assert builder_class in stub
    assert builder_factory in stub
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


def test_motion_root_selects_frame_builder_only_from_the_robot_axis() -> None:
    """A dataset token must never select or wrap the robot frame-builder factory."""
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

    expected = {"default": "smpl_frame_builder", "g1": "g1_frame_builder"}
    for alternative, factory_name in expected.items():
        value = assignments[alternative]
        assert isinstance(value, ast.Call)
        keyword = next(keyword for keyword in value.keywords if keyword.arg == "frame_builder_factory")
        assert isinstance(keyword.value, ast.Name)
        assert keyword.value.id == factory_name
        assert all(not isinstance(node, ast.Name) or node.id != "preset" for node in ast.walk(keyword.value))
    assert isinstance(assignments["smpl"], ast.Name) and assignments["smpl"].id == "default"

    root_source = path.read_text(encoding="utf-8")
    assert root_source.count("MotionExactFamilyCfg(") == 1
    assert root_source.count("MotionSemanticFamilyCfg(") == 1
    assert "MotionExactMaterializationCriterionCfg" not in root_source
    assert 'MotionObjectiveMeasureCriterionCfg(objective="landmark_position", upper=0.15)' in root_source
    assert 'MotionObjectiveMeasureCriterionCfg(objective="landmark_rotation"' not in root_source


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


@pytest.mark.parametrize(("robot", "builder_class"), (("g1", "G1FrameBuilder"), ("smpl", "SmplFrameBuilder")))
def test_motion_robot_builder_materializes_exactly_one_source_view(robot: str, builder_class: str) -> None:
    """Robot builders expose only the direct family-stage construction contract."""
    path = _MOTION_ROOT / "robots" / robot / "reference.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    builder = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == builder_class)
    methods = {node.name for node in builder.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "build_frames" not in methods
    assert {
        "build_exact_coordinates",
        "generate_semantic_targets",
        "semantic_reference_kinematics",
        "semantic_target_tree",
        "build_semantic_corpus",
    } <= methods
    assert "solve_semantic_targets" not in methods

    table_path = _MOTION_ROOT / "mdp" / "commands" / "motion_task_table.py"
    table = ast.parse(table_path.read_text(encoding="utf-8"), filename=str(table_path))
    functions = {node.name: node for node in table.body if isinstance(node, ast.FunctionDef)}
    exact = functions["motion_generate_exact_coordinates"]
    semantic = functions["_motion_semantic_targets"]
    views = {"free_root_coordinates", "semantic_local_pose"}
    assert _attribute_calls(exact.body, views) == {"free_root_coordinates"}
    assert _attribute_calls(semantic.body, views) == {"semantic_local_pose"}


def test_semantic_solver_uses_one_corpus_workspace_and_continuous_solve() -> None:
    """Semantic clips stream through one executor without solver restarts or per-clip workspaces."""
    path = _MOTION_ROOT / "mdp" / "commands" / "motion_task_table.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    solve = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "motion_solve_semantic_sequence"
    )
    calls = [
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(solve)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    ]
    assert calls.count("execute_ik_batches") == 1
    assert calls.count("IKSolver") == 1
    assert "solve" in calls
    assert "step" not in calls
    assert "kinematic_seed_target_rotations" in calls
    assert "fill" in calls


def test_semantic_convergence_checks_follow_coordinate_projection() -> None:
    """Early convergence can occur only after root and hard-limit projection."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionSemanticSolveCfg

    cfg = MotionSemanticSolveCfg()
    assert cfg.max_iterations == 30
    assert cfg.projection_interval == cfg.convergence_check_interval == 5
    assert "projected_iteration_chunks" not in {field.name for field in dataclasses.fields(cfg)}


def test_semantic_bad_node_splits_two_maximal_source_ranges() -> None:
    """One bad node is removed while both builder-safe neighboring runs survive."""
    from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionSemanticSegmentSelectionCfg
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import (
        _motion_semantic_runs,
        _MotionCorpusCandidate,
        motion_select_semantic_segments,
    )

    index = MotionClipIndex(
        source_content_sha256=_sha256("node-split-source"),
        clips=(MotionClipIndex.Clip("clip", 9, 30.0, _sha256("node-split-clip")),),
    )
    quality = torch.zeros(9, 3)
    candidate = _MotionCorpusCandidate(object(), None, index, "cpu", None, semantic_quality=quality)
    accepted = torch.ones(9, dtype=torch.bool)
    accepted[4] = False
    cfg = MotionSemanticSegmentSelectionCfg()

    runs = _motion_semantic_runs(index, accepted, quality[:, 2], cfg.max_branch_jump_rad)
    selected = motion_select_semantic_segments(cfg, candidate, accepted, None, object())

    torch.testing.assert_close(runs.starts, torch.tensor((0, 5)))
    torch.testing.assert_close(runs.stops, torch.tensor((4, 9)))
    torch.testing.assert_close(selected, torch.tensor((0, 1, 2, 3, 5, 6, 7, 8)))


def test_semantic_bad_edge_keeps_both_endpoints_in_separate_ranges() -> None:
    """A discontinuous incoming edge cuts the run without rejecting either endpoint."""
    from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionSemanticSegmentSelectionCfg
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import (
        _motion_semantic_runs,
        _MotionCorpusCandidate,
        motion_select_semantic_segments,
    )

    index = MotionClipIndex(
        source_content_sha256=_sha256("edge-split-source"),
        clips=(MotionClipIndex.Clip("clip", 8, 30.0, _sha256("edge-split-clip")),),
    )
    quality = torch.zeros(8, 3)
    quality[4, 2] = 4.0
    candidate = _MotionCorpusCandidate(object(), None, index, "cpu", None, semantic_quality=quality)
    accepted = torch.ones(8, dtype=torch.bool)
    cfg = MotionSemanticSegmentSelectionCfg(max_branch_jump_rad=3.0)

    runs = _motion_semantic_runs(index, accepted, quality[:, 2], cfg.max_branch_jump_rad)
    selected = motion_select_semantic_segments(cfg, candidate, accepted, None, object())

    torch.testing.assert_close(runs.starts, torch.tensor((0, 4)))
    torch.testing.assert_close(runs.stops, torch.tensor((4, 8)))
    torch.testing.assert_close(selected, torch.arange(8))


def test_semantic_residual_criterion_rejects_a_bad_finite_solution() -> None:
    """Finite semantic residuals above the declared quality gate are rejected."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import (
        MotionObjectiveMeasureCriterionCfg,
        MotionSemanticFamilyCfg,
    )
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import (
        _MotionCorpusCandidate,
        motion_criterion_objective_measure,
    )

    candidate = _MotionCorpusCandidate(
        builder=object(),
        source=None,
        clip_index=object(),
        device="cpu",
        frames=object(),
        semantic_quality=torch.tensor(((0.151, 0.0, 0.0),)),
    )
    criterion = next(
        item
        for item in MotionSemanticFamilyCfg().criteria
        if isinstance(item, MotionObjectiveMeasureCriterionCfg) and item.objective == "landmark_position"
    )

    accepted = motion_criterion_objective_measure(criterion, candidate, torch.tensor((0,)))

    torch.testing.assert_close(accepted, torch.tensor((False,)))


def test_semantic_orientation_is_quality_evidence_not_an_acceptance_gate() -> None:
    """An arbitrarily large orientation residual must not reject position-valid semantic output."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import (
        MotionObjectiveMeasureCriterionCfg,
        MotionSemanticFamilyCfg,
    )
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import (
        _MotionCorpusCandidate,
        motion_criterion_objective_measure,
    )

    objective_criteria = tuple(
        item for item in MotionSemanticFamilyCfg().criteria if isinstance(item, MotionObjectiveMeasureCriterionCfg)
    )
    candidate = _MotionCorpusCandidate(
        builder=object(),
        source=None,
        clip_index=object(),
        device="cpu",
        frames=object(),
        semantic_quality=torch.tensor(((0.1, 100.0, 0.0),)),
    )

    assert tuple(item.objective for item in objective_criteria) == ("landmark_position",)
    accepted = motion_criterion_objective_measure(objective_criteria[0], candidate, torch.tensor((0,)))
    torch.testing.assert_close(accepted, torch.tensor((True,)))


def test_derived_segment_clip_preserves_full_identity_and_stable_source_ranges() -> None:
    """Full spans reuse source identity while partial spans retain deterministic provenance."""
    from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import _derived_segment_clip

    source = MotionClipIndex.Clip("origin", 8, 30.0, _sha256("origin"))

    assert _derived_segment_clip(source, 0, 8) is source
    first = _derived_segment_clip(source, 0, 4)
    duplicate = _derived_segment_clip(source, 0, 4)
    second = _derived_segment_clip(source, 4, 8)

    assert first == duplicate
    assert first.source_clip_id == second.source_clip_id == "origin"
    assert (first.source_frame_start, first.source_frame_stop) == (0, 4)
    assert (second.source_frame_start, second.source_frame_stop) == (4, 8)
    assert first.clip_id != second.clip_id
    assert first.content_sha256 != second.content_sha256


def test_semantic_finalizer_materializes_one_corpus_with_source_balanced_segments() -> None:
    """A cut corpus is built once with segment-local quality and immutable source mass."""
    from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex, MotionFrames
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionSemanticSegmentSelectionCfg
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import (
        _finalize_semantic_corpus,
        _MotionCorpusCandidate,
    )

    class Builder:
        def __init__(self) -> None:
            self.inputs: list[tuple[torch.Tensor, MotionClipIndex]] = []

        def build_semantic_corpus(self, joint_q: torch.Tensor, clip_index: MotionClipIndex) -> MotionFrames:
            self.inputs.append((joint_q.clone(), clip_index))
            frame_count = joint_q.shape[0]
            root_rotation = torch.zeros(frame_count, 4)
            root_rotation[:, 3] = 1.0
            return MotionFrames(
                root_position=torch.zeros(frame_count, 3),
                root_rotation=root_rotation,
                root_linear_velocity=torch.zeros(frame_count, 3),
                root_angular_velocity=torch.zeros(frame_count, 3),
                joint_position=joint_q.clone(),
                joint_velocity=torch.zeros_like(joint_q),
            )

    index = MotionClipIndex(
        source_content_sha256=_sha256("segmented-source"),
        clips=(
            MotionClipIndex.Clip("origin", 8, 30.0, _sha256("origin")),
            MotionClipIndex.Clip("native", 3, 60.0, _sha256("native")),
        ),
    )
    joint_q = torch.arange(index.total_frames, dtype=torch.float32).unsqueeze(1)
    quality = torch.zeros(index.total_frames, 3)
    quality[4, 2] = 4.0
    builder = Builder()
    candidate = _MotionCorpusCandidate(
        builder,
        None,
        index,
        "cpu",
        None,
        semantic_joint_q=joint_q,
        semantic_quality=quality,
    )
    accepted = torch.ones(index.total_frames, dtype=torch.bool)
    selected = torch.arange(index.total_frames)
    selection = MotionSemanticSegmentSelectionCfg(max_branch_jump_rad=3.0)

    result_index, frames, result_quality = _finalize_semantic_corpus(candidate, accepted, selected, selection)

    assert len(builder.inputs) == 1
    torch.testing.assert_close(builder.inputs[0][0], joint_q)
    assert builder.inputs[0][1] is result_index
    assert result_index.clips[2] is index.clips[1]
    assert tuple(clip.source_clip_id for clip in result_index.clips[:2]) == ("origin", "origin")
    assert tuple((clip.source_frame_start, clip.source_frame_stop) for clip in result_index.clips[:2]) == (
        (0, 4),
        (4, 8),
    )
    torch.testing.assert_close(result_quality[:, 4], torch.zeros(3))
    torch.testing.assert_close(result_quality[:, 5], torch.tensor((0.5, 0.5, 1.0)))

    duplicate_index, duplicate_frames, duplicate_quality = _finalize_semantic_corpus(
        candidate, accepted, selected, selection
    )
    assert duplicate_index == result_index
    for name in frames.stored_fields:
        assert torch.equal(duplicate_frames.field(name), frames.field(name))
    assert torch.equal(duplicate_quality, result_quality)
