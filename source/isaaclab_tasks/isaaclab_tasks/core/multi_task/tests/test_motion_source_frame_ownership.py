# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused parity checks for typed motion sources and robot-owned frames."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from isaaclab.utils.math import quat_from_rotation_vector, quat_mul

from isaaclab_tasks.core.multi_task.kinematics import KinematicTree, ordered_hinge_rotation
from isaaclab_tasks.core.multi_task.motion.data import (
    MotionGeneralizedCoordinateClip,
    MotionLocalBodyPoseClip,
    MotionPoseAxisAngleClip,
    MotionSkeleton,
)
from isaaclab_tasks.core.multi_task.motion.data.sources import (
    CmuHumEnvSmplClip,
    LafanG1Clip,
)
from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import G1PoseFrameBuilder


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
    )


class _ReferenceKinematics:
    def __init__(self, skeleton: MotionSkeleton, path: Path) -> None:
        self.body_names = list(skeleton.body_names)
        self.joint_names = ["root", *skeleton.joint_names]
        self.mjcf_path = str(path)
        self.device = "cpu"
        self.model = SimpleNamespace(
            body_count=skeleton.num_bodies,
            joint_coord_count=7 + skeleton.num_joints,
            joint_dof_count=6 + skeleton.num_joints,
        )

    def eval_fk_batched_torch(
        self,
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor,
        body_q: torch.Tensor,
        body_qd: torch.Tensor,
    ) -> None:
        del joint_qd
        body_q.zero_()
        body_q[..., :3].copy_(joint_q[:, None, :3])
        body_q[..., 3:].copy_(joint_q[:, None, 3:7])
        body_qd.zero_()


def test_lafan_typed_clip_preserves_direct_g1_frame_construction(tmp_path: Path) -> None:
    """The typed source seam must not change any target-G1 frame tensor."""
    path = tmp_path / "g1.xml"
    path.write_text("<mujoco/>", encoding="utf-8")
    skeleton = _g1_skeleton()
    with path.open("rb") as stream:
        reference_mjcf_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    builder = G1PoseFrameBuilder(
        target_tree=KinematicTree(
            body_names=skeleton.body_names,
            joint_names=skeleton.joint_names,
            parent_indices=skeleton.parent_indices,
            joint_child_body_indices=skeleton.joint_child_body_indices,
            joint_axes=tuple(axis for axis in skeleton.joint_axes if axis is not None),
        ),
        pose_coordinate_identity_sha256=skeleton.identity_sha256,
        reference_kinematics=_ReferenceKinematics(skeleton, path),
        reference_mjcf_sha256=reference_mjcf_sha256,
        live_joint_names=skeleton.joint_names,
        live_body_names=skeleton.body_names,
    )
    pose = np.zeros((5, 30, 3), dtype=np.float32)
    root = np.arange(15, dtype=np.float32).reshape(5, 3) * 0.01
    clip = LafanG1Clip(root_translation=root, pose_axis_angle=pose, source_fps=30.0)

    typed = builder.build_frames(clip, device="cpu")
    assert isinstance(clip, MotionPoseAxisAngleClip)
    assert isinstance(clip, MotionLocalBodyPoseClip)
    direct = builder.build_pose_frames(torch.from_numpy(pose), torch.from_numpy(root), 30.0)

    assert typed.stored_fields == direct.stored_fields
    for name in typed.stored_fields:
        torch.testing.assert_close(typed.field(name), direct.field(name))


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

    assert isinstance(clip, MotionGeneralizedCoordinateClip)
    assert isinstance(clip, MotionLocalBodyPoseClip)
    rotation = clip.local_body_rotation_wxyz(_smpl_skeleton(), device="cpu")
    half_angle = 0.5 * torch.from_numpy(qpos[:, 7])
    expected = torch.stack(
        (torch.cos(half_angle), torch.sin(half_angle), torch.zeros_like(half_angle), torch.zeros_like(half_angle)),
        dim=-1,
    )

    torch.testing.assert_close(rotation[:, 0], torch.tensor((1.0, 0.0, 0.0, 0.0)).expand(frame_count, 4))
    torch.testing.assert_close(rotation[:, 1], expected)


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
        model=SimpleNamespace(
            body_count=30,
            joint_count=30,
            joint_parent=torch.tensor((-1, *range(29)), dtype=torch.int32),
            joint_child=torch.arange(30, dtype=torch.int32),
            joint_qd_start=torch.tensor((0, 6, *range(7, 36)), dtype=torch.int32),
            joint_axis=axes,
        ),
    )

    target = KinematicTree.from_newton(reference)

    assert target.parent_indices == (-1, *range(29))
    assert target.joint_child_body_indices == tuple(range(1, 30))
    assert target.joint_axes == ((1.0, 0.0, 0.0),) * 29
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
    assert "from .motion.data.sources import (" in root_source
    assert "reference_kinematics_factory=smpl_humenv_reference_kinematics" in root_source
    assert not (motion_root / "config" / "sources.py").exists()
    assert not (motion_root / "config" / "source_skeletons.py").exists()


def test_robot_and_data_runtime_dependency_direction_is_one_way() -> None:
    """Robot and data implementations must not depend on downstream command or learner owners."""
    motion_root = Path(__file__).parents[1] / "motion"
    forbidden_robot_modules = ("rsl_rl", ".config.agents", ".evaluation")
    for path in sorted((motion_root / "robots").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
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
        for node in ast.walk(tree):
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
    assert "if source.joint_names != target.joint_names:" in materializer_text
    assert "G1HumEnvFrameBuilder" not in robot_text
    assert "cmu_to_g1_frame_builder" not in robot_text
    assert "retargeted_lafan_to_g1_frame_builder" not in robot_text
    assert "if source.body_names != target.body_names:" in materializer_text

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


def test_smpl_robot_does_not_own_humenv_reference_interpreter() -> None:
    """HumEnv interpreter assets and construction belong to the source-coordinate owner."""
    motion_root = Path(__file__).parents[1] / "motion"
    robot_source = (motion_root / "robots" / "smpl" / "reference.py").read_text(encoding="utf-8")
    coordinate_source = (motion_root / "data" / "sources" / "cmu_humenv_smpl_coordinates.py").read_text(
        encoding="utf-8"
    )

    for forbidden in ("SMPL_HUMENV_MJCF_PATH", "def smpl_reference_kinematics"):
        assert forbidden not in robot_source
    assert "def smpl_humenv_reference_kinematics" in coordinate_source

    assert "file_sha256(reference.mjcf_path)" in robot_source
    assert "reference_mjcf_sha256 != SMPL_HUMENV_MJCF_SHA256" in robot_source


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
