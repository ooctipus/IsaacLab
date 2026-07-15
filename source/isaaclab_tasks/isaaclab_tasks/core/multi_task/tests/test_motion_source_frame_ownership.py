# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused parity checks for typed motion sources and robot-owned frames."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab.utils.math import convert_quat, quat_from_rotation_vector, quat_mul

from isaaclab_tasks.core.multi_task.kinematics import KinematicTree, ordered_hinge_rotation
from isaaclab_tasks.core.multi_task.motion.data import (
    MotionSkeleton,
    MotionSourceClip,
)
from isaaclab_tasks.core.multi_task.motion.data.sources import (
    CmuHumEnvSmplClip,
    LafanG1Clip,
)


def _g1_skeleton() -> MotionSkeleton:
    body_names = ("pelvis", *("torso_link" if index == 15 else f"body_{index}" for index in range(1, 30)))
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return MotionSkeleton(
        identifier="test_g1",
        content_sha256="0" * 64,
        body_names=body_names,
        parent_indices=(-1, *range(29)),
        rest_translation_m=((0.0, 0.0, 0.0),) * 30,
        rest_rotation_wxyz=((1.0, 0.0, 0.0, 0.0),) * 30,
        joint_names=tuple(f"joint_{index}" for index in range(29)),
        joint_child_body_indices=tuple(range(1, 30)),
        joint_axes=tuple(axes[index % 3] for index in range(29)),
        root_translation_frame="world",
        root_rotation_convention="axis_angle",
        landmark_rotation_policy="calibrated_body",
    )


def _smpl_skeleton() -> MotionSkeleton:
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return MotionSkeleton(
        identifier="test_smpl",
        content_sha256="1" * 64,
        body_names=tuple(f"body_{index}" for index in range(24)),
        parent_indices=(-1, *range(23)),
        rest_translation_m=((0.0, 0.0, 0.0),) * 24,
        rest_rotation_wxyz=((1.0, 0.0, 0.0, 0.0),) * 24,
        joint_names=tuple(f"joint_{index}" for index in range(69)),
        joint_child_body_indices=tuple(body for body in range(1, 24) for _ in range(3)),
        joint_axes=axes * 23,
        root_translation_frame="world",
        root_rotation_convention="wxyz",
        landmark_rotation_policy="calibrated_body",
    )


def test_lafan_typed_clip_exposes_exact_and_semantic_views_once() -> None:
    """The LAFAN decoder exposes identical coordinate and semantic root facts."""
    skeleton = _g1_skeleton()
    pose = np.zeros((5, 30, 3), dtype=np.float32)
    root = np.arange(15, dtype=np.float32).reshape(5, 3) * 0.01
    clip = LafanG1Clip(root_translation=root, pose_axis_angle=pose, source_fps=30.0)

    assert isinstance(clip, MotionSourceClip)
    joint_q, joint_qd = clip.free_root_coordinates(skeleton, device="cpu")
    semantic_root, semantic_rotation = clip.local_pose(skeleton, device="cpu")
    assert joint_qd is None
    torch.testing.assert_close(joint_q[:, :3], semantic_root)
    torch.testing.assert_close(joint_q[:, 3:7], semantic_rotation[:, 0])
    torch.testing.assert_close(joint_q[:, 7:], torch.zeros(5, 29))


def test_lafan_exact_view_does_not_materialize_semantic_rotations(monkeypatch: pytest.MonkeyPatch) -> None:
    """The native-coordinate route must not enter the semantic-pose route."""
    skeleton = _g1_skeleton()
    pose = np.zeros((5, 30, 3), dtype=np.float32)
    root = np.arange(15, dtype=np.float32).reshape(5, 3) * 0.01
    clip = LafanG1Clip(root_translation=root, pose_axis_angle=pose, source_fps=30.0)

    def reject_semantic_materialization(*_args, **_kwargs):
        raise AssertionError("exact decoding entered semantic materialization")

    monkeypatch.setattr(LafanG1Clip, "local_pose", reject_semantic_materialization)

    joint_q, joint_qd = clip.free_root_coordinates(skeleton, device="cpu")

    assert joint_qd is None
    torch.testing.assert_close(joint_q[:, :3], torch.from_numpy(root))
    torch.testing.assert_close(joint_q[:, 7:], torch.zeros(5, 29))


@pytest.mark.parametrize("field_name", ("root_translation", "pose_axis_angle"))
def test_lafan_clip_rejects_nonfinite_source_rows(field_name: str) -> None:
    """The typed source boundary rejects non-finite positions and rotations once."""
    values = {
        "root_translation": np.zeros((5, 3), dtype=np.float32),
        "pose_axis_angle": np.zeros((5, 30, 3), dtype=np.float32),
    }
    values[field_name].reshape(-1)[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        LafanG1Clip(**values, source_fps=30.0)


def test_cmu_typed_clip_preserves_humenv_local_rotation_decode() -> None:
    """The typed CMU seam reconstructs the same declared XYZ local rotation."""
    frame_count = 3
    qpos = np.zeros((frame_count, 76), dtype=np.float32)
    qpos[:, 3] = 1.0
    qpos[:, 7] = np.asarray((0.0, 0.2, -0.4), dtype=np.float32)
    clip = CmuHumEnvSmplClip(
        generalized_position=qpos,
        generalized_velocity=np.zeros((frame_count, 75), dtype=np.float32),
        source_fps=30.0,
    )

    assert isinstance(clip, MotionSourceClip)
    root, rotation = clip.local_pose(_smpl_skeleton(), device="cpu")
    half_angle = 0.5 * torch.from_numpy(qpos[:, 7])
    expected = torch.stack(
        (torch.cos(half_angle), torch.sin(half_angle), torch.zeros_like(half_angle), torch.zeros_like(half_angle)),
        dim=-1,
    )

    torch.testing.assert_close(root, torch.zeros(frame_count, 3))
    torch.testing.assert_close(rotation[:, 0], torch.tensor((0.0, 0.0, 0.0, 1.0)).expand(frame_count, 4))
    torch.testing.assert_close(rotation[:, 1], convert_quat(expected, to="xyzw"))


def test_ordered_hinge_rotation_preserves_source_multiply_order_bitwise() -> None:
    """Shared hinge composition must not add an identity multiply to source materialization."""
    generator = torch.Generator().manual_seed(17)
    coordinates = torch.randn(257, 5, 3, dtype=torch.float32, generator=generator)
    axes = torch.eye(3, dtype=torch.float32).expand(5, -1, -1)
    coordinate_rotation = quat_from_rotation_vector(coordinates.unsqueeze(-1) * axes.unsqueeze(0))
    expected = quat_mul(
        quat_mul(coordinate_rotation[..., 0, :], coordinate_rotation[..., 1, :]),
        coordinate_rotation[..., 2, :],
    )

    actual = ordered_hinge_rotation(coordinates, axes)

    assert torch.equal(actual, expected)


def test_kinematic_tree_is_derived_from_reference_model(tmp_path: Path) -> None:
    """Generic target topology comes from the verified model, not motion source state."""
    path = tmp_path / "g1_target.xml"
    path.write_text("<mujoco/>", encoding="utf-8")
    axes = torch.zeros(35, 3, dtype=torch.float32)
    axes[6:, 0] = 1.0
    reference = SimpleNamespace(
        mjcf_path=str(path),
        body_names=["pelvis", *(f"body_{index}" for index in range(1, 30))],
        joint_names=["root", *(f"joint_{index}" for index in range(29))],
        joint_q_names=[*(f"root_{index}" for index in range(7)), *(f"joint_{index}" for index in range(29))],
        topology=SimpleNamespace(
            body_count=30,
            joint_count=30,
            coordinate_count=36,
            joint_parent=np.asarray((-1, *range(29)), dtype=np.int32),
            joint_child=np.arange(30, dtype=np.int32),
            joint_q_start=np.asarray((0, 7, *range(8, 37)), dtype=np.int32),
            joint_qd_start=np.asarray((0, 6, *range(7, 36)), dtype=np.int32),
            joint_dof_dim=np.asarray(((0, 6), *((0, 1),) * 29), dtype=np.int32),
            joint_axis=axes.numpy(),
            joint_limit_lower=np.full(35, -1.0, dtype=np.float32),
            joint_limit_upper=np.full(35, 1.0, dtype=np.float32),
            body_parent=np.asarray((-1, *range(29)), dtype=np.int32),
        ),
    )

    target = KinematicTree.from_newton(reference)

    assert target.parent_indices == (-1, *range(29))
    assert target.joint_child_body_indices == tuple(range(1, 30))
    assert target.coordinate_axes == ((1.0, 0.0, 0.0),) * 29
    assert target.root_body_index == 0


def test_robot_mdp_implementations_have_no_generic_motion_fallbacks() -> None:
    """Robot and solver implementations must not leak back into generic Motion MDP modules."""
    import isaaclab_tasks.core.multi_task.motion as motion_package

    motion_root = Path(motion_package.__file__).parent
    for name in (
        "actions.py",
        "actions_cfg.py",
        "curriculums.py",
        "events.py",
        "history.py",
        "observations.py",
        "reset_sources.py",
        "runtime.py",
    ):
        assert not (motion_root / "mdp" / name).exists()
    assert not (motion_root / "robots" / "smpl" / "transition.py").exists()
    assert not (motion_root / "robots" / "smpl" / "events.py").exists()
    assert not (motion_root / "robots" / "g1" / "history.py").exists()
    assert not (motion_root / "robots" / "g1" / "transition.py").exists()
    assert not (motion_root / "robots" / "g1" / "rewards.py").exists()
    assert not (motion_root / "mdp" / "commands" / "observations.py").exists()

    for relative in (
        "robots/g1/actions.py",
        "robots/smpl/observations.py",
        "robots/smpl/reset.py",
        "../mdp/native_mujoco_action.py",
    ):
        assert (motion_root / relative).resolve().exists()


def test_robot_reference_builders_do_not_import_concrete_motion_sources() -> None:
    """Robot materializers consume representation protocols, never dataset implementations."""
    motion_root = Path(__file__).parents[1] / "motion"
    violations = []
    for path in sorted((motion_root / "robots").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports_relative_source = node.level > 0 and (
                    module == "data.sources" or module.startswith("data.sources.")
                )
                if imports_relative_source or ".motion.data.sources" in module:
                    violations.append((path.relative_to(motion_root), module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if ".motion.data.sources" in alias.name:
                        violations.append((path.relative_to(motion_root), alias.name))

    assert violations == []


def test_source_declarations_are_visible_at_the_root_and_old_facades_are_absent() -> None:
    """Concrete source declarations live in the root and import coordinates from data owners."""
    motion_root = Path(__file__).parents[1] / "motion"
    root_source = (motion_root.parent / "motion_env_cfg.py").read_text(encoding="utf-8")

    assert "class MotionSourcesCfg(PresetCfg):" in root_source
    assert 'identifier="cmu_humenv_smpl"' in root_source
    assert 'identifier="lafan_g1_29dof"' in root_source
    assert "from .motion.data.sources import open_cmu_humenv_smpl_source, open_lafan_g1_source" in root_source
    assert "reference_kinematics_factory" not in root_source
    assert not (motion_root / "config" / "sources.py").exists()
    assert not (motion_root / "config" / "source_skeletons.py").exists()


def test_source_skeleton_has_no_source_owned_distal_endpoint_schema() -> None:
    """Sources own landmark frames while target robots own endpoint geometry."""
    motion_root = Path(__file__).parents[1] / "motion"
    skeleton_source = (motion_root / "data" / "skeleton.py").read_text(encoding="utf-8")
    concrete_sources = (
        motion_root / "data" / "sources" / "lafan_bvh.py",
        motion_root / "data" / "sources" / "amass_smplh.py",
        motion_root / "data" / "sources" / "cmu_humenv_smpl_coordinates.py",
        motion_root / "data" / "sources" / "lafan_g1_29dof_coordinates.py",
    )
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in concrete_sources)

    assert "rotation_body_name" in skeleton_source
    for forbidden in ("class DistalPoint", "distal_points"):
        assert forbidden not in skeleton_source
    assert "MotionSkeleton.DistalPoint" not in source_text
    assert "distal_points" not in source_text


def test_robot_and_data_runtime_dependency_direction_is_one_way() -> None:
    """Robot and data implementations must not depend on downstream command or learner owners."""

    def runtime_imports(tree: ast.Module) -> tuple[ast.Import | ast.ImportFrom, ...]:
        """Return imports that execute outside TYPE_CHECKING guards."""
        guarded = {
            child
            for node in tree.body
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
            for child in ast.walk(node)
        }
        return tuple(
            node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in guarded
        )

    motion_root = Path(__file__).parents[1] / "motion"
    forbidden_robot_modules = ("rsl_rl", ".config.agents", ".evaluation")
    for path in sorted((motion_root / "robots").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in runtime_imports(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not any(token in module for token in forbidden_robot_modules), (path, module)
                imports_motion_commands = (node.level > 0 and module.startswith("mdp.commands")) or (
                    ".motion.mdp.commands" in module
                )
                assert not imports_motion_commands, (path, module)
            elif isinstance(node, ast.Import):
                assert not any(token in alias.name for alias in node.names for token in forbidden_robot_modules), path

    for path in sorted((motion_root / "data").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in runtime_imports(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports_robot = (node.level > 0 and module.startswith("robots")) or ".motion.robots" in module
                imports_agents = (node.level > 0 and module.startswith("config.agents")) or (
                    ".motion.config.agents" in module
                )
                assert not imports_robot and not imports_agents, (path, module)
            elif isinstance(node, ast.Import):
                assert all(
                    ".motion.robots" not in alias.name and ".motion.config.agents" not in alias.name
                    for alias in node.names
                ), path


def test_g1_robot_schema_does_not_alias_lafan_source_schema() -> None:
    """G1 owns its behavior axes while LAFAN remains an independent source schema."""
    motion_root = Path(__file__).parents[1] / "motion"
    g1_root = motion_root / "robots" / "g1"
    robot_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(g1_root.glob("*.py")))

    assert "LAFAN_G1_" not in robot_text
    assert "lafan_g1_29dof_skeleton" not in robot_text
    assert "g1_motion_skeleton" not in robot_text
    assert "G1LafanFrameBuilder" not in robot_text
    assert "g1_lafan_frame_builder" not in robot_text
    assert "g1_humenv_frame_builder" not in robot_text

    materializer_text = (g1_root / "reference.py").read_text(encoding="utf-8")
    assert "source.joint_names == target.joint_names" in materializer_text
    assert "G1HumEnvFrameBuilder" not in robot_text
    assert "cmu_to_g1_frame_builder" not in robot_text
    assert "retargeted_lafan_to_g1_frame_builder" not in robot_text
    assert "source.body_names == target.body_names" in materializer_text

    articulation = ast.parse((g1_root / "articulation.py").read_text(encoding="utf-8"))
    assignments = {
        node.targets[0].id: node.value
        for node in articulation.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
    }
    assert isinstance(assignments["G1_BEHAVIOR_BODY_NAMES"], ast.Tuple)
    assert isinstance(assignments["G1_BEHAVIOR_JOINT_NAMES"], ast.Tuple)


def test_smpl_articulation_does_not_alias_dataset_coordinates() -> None:
    """The robot asset must not re-export a source-owned skeleton or retain dead name mirrors."""
    articulation_path = Path(__file__).parents[1] / "motion" / "robots" / "smpl" / "articulation.py"
    source = articulation_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(articulation_path))
    definitions = {
        node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assignments = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    assert "smpl_motion_skeleton" not in definitions
    assert {"_SMPL_SIMULATOR_BODY_NAMES", "_SMPL_SIMULATOR_JOINT_NAMES"}.isdisjoint(assignments)
    assert "data.sources" not in source


def test_smpl_target_reference_model_is_robot_owned() -> None:
    """The reusable target Newton model belongs to SMPL, not one concrete motion source."""
    motion_root = Path(__file__).parents[1] / "motion"
    robot_source = (motion_root / "robots" / "smpl" / "reference.py").read_text(encoding="utf-8")
    coordinate_source = (motion_root / "data" / "sources" / "cmu_humenv_smpl_coordinates.py").read_text(
        encoding="utf-8"
    )

    assert "SMPL_HUMENV_MJCF_PATH" in robot_source
    assert "def smpl_reference_kinematics" in robot_source
    assert "SMPL_HUMENV_MJCF_PATH" not in coordinate_source
    assert "def smpl_reference_kinematics" not in coordinate_source
    assert "reference.mechanics_identity_sha256" in robot_source
    assert "scene-derived NewtonKinematics" in robot_source


def test_g1_robot_does_not_own_generic_kinematics_or_temporal_math() -> None:
    """Generic trees, projections, hinge fitting, and time operators belong to kinematics."""
    source = Path(__file__).parents[1] / "motion" / "robots" / "g1" / "reference.py"
    text = source.read_text(encoding="utf-8")

    for forbidden in (
        "class G1TargetSkeleton",
        "class G1LocalRotationProjection",
        "def fit_ordered_hinge_coordinates",
        "def _compose_ordered_hinges",
        "def _unwrap_angle_time",
        "def _gradient_time",
        "def _gaussian_filter_time",
        "def _angular_velocity_raw",
        "def _angular_velocity",
        "MotionSkeleton |",
    ):
        assert forbidden not in text


def test_cross_robot_builders_resolve_source_owned_landmarks() -> None:
    """Target robot modules must not encode the other robot's source body names."""
    motion_root = Path(__file__).parents[1] / "motion"
    smpl_reference = (motion_root / "robots" / "smpl" / "reference.py").read_text(encoding="utf-8")
    g1_reference = (motion_root / "robots" / "g1" / "reference.py").read_text(encoding="utf-8")
    retarget = (motion_root / "retarget.py").read_text(encoding="utf-8")

    for source_body_name in (
        "left_hip_pitch_link",
        "left_hip_yaw_link",
        "right_hip_pitch_link",
        "left_wrist_roll_link",
        "right_wrist_yaw_link",
    ):
        assert source_body_name not in smpl_reference
    assert "source_skeleton.landmarks" not in smpl_reference
    assert "source_skeleton.landmarks" not in g1_reference
    assert "self.source_skeleton.landmarks" in retarget
    assert "KinematicTreeRotationProjection" not in g1_reference


def test_cmu_source_reuses_shared_ordered_hinge_rotation() -> None:
    """A source decoder must declare coordinates, not own generic hinge composition."""
    source = Path(__file__).parents[1] / "motion" / "data" / "sources" / "cmu_humenv_smpl.py"
    text = source.read_text(encoding="utf-8")

    assert "ordered_hinge_rotation" in text
    assert "quat_from_rotation_vector" not in text
    assert "quat_mul" not in text


def test_motion_robot_modules_reuse_shared_identity_and_pose_math() -> None:
    """Robot code must consume shared identity and transform equations without wrappers."""
    motion_root = Path(__file__).parents[1] / "motion"
    articulation_owners = (
        motion_root / "robots" / "g1" / "articulation.py",
        motion_root / "robots" / "smpl" / "articulation.py",
    )
    reference_builders = (
        motion_root / "robots" / "g1" / "reference.py",
        motion_root / "robots" / "smpl" / "reference.py",
    )
    for path in (*articulation_owners, *reference_builders):
        text = path.read_text(encoding="utf-8")
        assert "hashlib" not in text
        assert "file_digest" not in text
        assert "def _sha256" not in text

    for path in reference_builders:
        text = path.read_text(encoding="utf-8")
        assert "file_sha256(" in text

    for path in reference_builders:
        text = path.read_text(encoding="utf-8")
        assert "validate_sha256(" in text
    for path in articulation_owners:
        assert "file_sha256(" not in path.read_text(encoding="utf-8")

    observations = (motion_root / "robots" / "g1" / "observations.py").read_text(encoding="utf-8")
    assert "quat_apply_inverse(root_rotation_xyzw, gravity_world)" in observations
    assert "quat_apply(quat_conjugate(root_rotation_xyzw), gravity_world)" not in observations

    frames = (motion_root / "robots" / "g1" / "frames.py").read_text(encoding="utf-8")
    assert "combine_frame_transforms(" in frames
    assert "head_position = body_position_world" not in frames

    smpl_reference = (motion_root / "robots" / "smpl" / "reference.py").read_text(encoding="utf-8")
    assert "body_rotation.reshape(-1, 4)" not in smpl_reference
