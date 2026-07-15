# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the motion task table and StateCommand payload."""

from __future__ import annotations

import hashlib
import inspect
import math
import weakref
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.utils.math import quat_from_rotation_vector, quat_slerp

import isaaclab_tasks.core.multi_task.motion.data as motion_data_module
from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics
from isaaclab_tasks.core.multi_task.mdp.commands.state_command import TaskTableView
from isaaclab_tasks.core.multi_task.motion.data import (
    MotionClipIndex,
    MotionFrames,
    MotionResetState,
    MotionSkeleton,
)
from isaaclab_tasks.core.multi_task.motion.data.frames import (
    MotionGeneralizedCoordinates,
    MotionSourceProjectionAnalytic,
    MotionSourceProjectionExact,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands import (
    MotionSampler,
    MotionStatePayload,
    MotionTaskTable,
    MotionTaskTableCfg,
    build_motion_task_table,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands import commands_cfg as commands_cfg_module
from isaaclab_tasks.core.multi_task.motion.mdp.commands import motion_task_table as table_module
from isaaclab_tasks.core.multi_task.motion.mdp.commands import motion_task_table_builder as table_builder_module
from isaaclab_tasks.core.multi_task.motion.mdp.commands import motion_trajectory as trajectory_module
from isaaclab_tasks.core.multi_task.tests.motion_table_test_utils import motion_task_table


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


_LIVE_JOINT_NAMES = ("joint_b", "joint_a")


def _reference() -> SimpleNamespace:
    """Return the exact mechanics fields retained by a synthetic table."""
    return SimpleNamespace(
        model=SimpleNamespace(joint_coord_count=7 + len(_LIVE_JOINT_NAMES)),
        n_root_coords=7,
        builder=object(),
        default_joint_q=[0.0] * (7 + len(_LIVE_JOINT_NAMES)),
    )


def _build_synthetic_task_table(cfg: MotionTaskTableCfg, device: str) -> MotionTaskTable:
    """Build through the production scene-owned mechanics path with a test reference."""
    scene_cfg = SimpleNamespace(robot=object())
    with patch.object(NewtonKinematics, "from_articulation", return_value=_reference()):
        return build_motion_task_table(SimpleNamespace(task_table=cfg), scene_cfg, device)


def _inspect_synthetic_task_table(cfg: MotionTaskTableCfg, device: str, *, sequence_limit: int = 16) -> TaskTableView:
    """Inspect through the production scene-owned mechanics path with a test reference."""
    scene_cfg = SimpleNamespace(robot=object())
    with patch.object(NewtonKinematics, "from_articulation", return_value=_reference()):
        return table_builder_module.build_motion_task_table_inspection(
            SimpleNamespace(task_table=cfg), scene_cfg, device, sequence_limit=sequence_limit
        )


def _skeleton() -> MotionSkeleton:
    return MotionSkeleton(
        identifier="motion_command_test",
        content_sha256=_hash("skeleton"),
        body_names=("root", "body"),
        parent_indices=(-1, 0),
        rest_translation_m=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        rest_rotation_wxyz=((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        joint_names=("joint_a", "joint_b"),
        joint_child_body_indices=(1, 1),
        joint_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        root_translation_frame="world",
        root_rotation_convention="wxyz",
        landmark_rotation_policy="calibrated_body",
    )


def _index() -> MotionClipIndex:
    return MotionClipIndex(
        source_content_sha256=_hash("source"),
        skeleton_identity_sha256s=(_skeleton().identity_sha256,),
        clips=(
            MotionClipIndex.Clip(
                clip_id="clip_a",
                frame_count=3,
                source_fps=2.0,
                content_sha256=_hash("clip-a"),
                skeleton_id=0,
            ),
            MotionClipIndex.Clip(
                clip_id="clip_b",
                frame_count=4,
                source_fps=4.0,
                content_sha256=_hash("clip-b"),
                skeleton_id=0,
            ),
        ),
    )


def _empty_frames(frame_count: int) -> MotionFrames:
    return MotionFrames(
        root_position=torch.empty(frame_count, 3),
        root_rotation=torch.empty(frame_count, 4),
        root_linear_velocity=torch.empty(frame_count, 3),
        root_angular_velocity=torch.empty(frame_count, 3),
        joint_position=torch.empty(frame_count, 2),
        joint_velocity=torch.empty(frame_count, 2),
    )


def test_motion_table_identity_changes_with_joint_order() -> None:
    """Robot joint-column order participates exactly once in table identity."""
    index = _index()
    frames = _empty_frames(index.total_frames)
    arguments = (
        index,
        "test_decoder_v1",
        frames,
    )
    suffix = (
        (),
        "test_builder_v1",
        _hash("builder"),
        "source_frames",
        "exact",
        _hash("exact-family"),
    )

    ordered = table_module._table_identity(*arguments, ("joint_a", "joint_b"), *suffix)
    repeated = table_module._table_identity(*arguments, ("joint_a", "joint_b"), *suffix)
    reversed_order = table_module._table_identity(*arguments, ("joint_b", "joint_a"), *suffix)

    assert ordered == repeated
    assert ordered != reversed_order


def test_motion_table_identity_changes_with_selected_family_policy() -> None:
    """Generation, solve, acceptance, and selection policy participates in table identity."""
    index = _index()
    frames = _empty_frames(index.total_frames)
    arguments = (
        index,
        "test_decoder_v1",
        frames,
        ("joint_a", "joint_b"),
        (),
        "test_builder_v1",
        _hash("builder"),
        "source_frames",
        "trajectory",
    )

    original = table_module._table_identity(*arguments, _hash("semantic-family-v1"))
    changed = table_module._table_identity(*arguments, _hash("semantic-family-v2"))

    assert original != changed


def test_motion_table_identity_hashes_each_source_clip_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The source provenance list preserves exact source cardinality and order."""
    index = _index()
    captured: dict[str, object] = {}

    def capture(value: dict[str, object]) -> str:
        captured.update(value)
        return _hash("captured-table")

    monkeypatch.setattr(table_module, "canonical_sha256", capture)
    table_module._table_identity(
        index,
        "test_decoder_v1",
        _empty_frames(index.total_frames),
        ("joint_a", "joint_b"),
        (),
        "test_builder_v1",
        _hash("builder"),
        "source_frames",
        "exact",
        _hash("exact-family"),
    )

    source_clips = captured["source_clips"]
    assert isinstance(source_clips, list)
    assert [clip["clip_id"] for clip in source_clips] == list(index.clip_ids)


def test_motion_table_identity_changes_with_decoder_and_clip_skeleton_assignment() -> None:
    """Decoder math and clip-to-skeleton ownership are table provenance."""
    index = _index()
    frames = _empty_frames(index.total_frames)
    suffix = (
        frames,
        ("joint_a", "joint_b"),
        (),
        "test_builder_v1",
        _hash("builder"),
        "source_frames",
        "exact",
        _hash("exact-family"),
    )
    original = table_module._table_identity(index, "decoder_v1", *suffix)
    changed_decoder = table_module._table_identity(index, "decoder_v2", *suffix)
    assigned = MotionClipIndex(
        source_content_sha256=index.source_content_sha256,
        skeleton_identity_sha256s=(index.skeleton_identity_sha256s[0], _hash("second-skeleton")),
        clips=(index.clips[0], replace(index.clips[1], skeleton_id=1)),
    )
    changed_assignment = table_module._table_identity(assigned, "decoder_v1", *suffix)

    assert original != changed_decoder
    assert original != changed_assignment


def _clip_frames(clip_number: int, frame_count: int) -> MotionFrames:
    frame = torch.arange(frame_count, dtype=torch.float32)
    base = 100.0 * clip_number + frame
    angle = frame * (math.pi / 2.0)
    zeros = torch.zeros_like(angle)
    return MotionFrames(
        root_position=torch.stack((base, base + 1.0, base + 2.0), dim=-1),
        root_rotation=torch.stack(
            (zeros, zeros, torch.sin(0.5 * angle), torch.cos(0.5 * angle)),
            dim=-1,
        ),
        root_linear_velocity=torch.stack((base + 30.0, base + 31.0, base + 32.0), dim=-1),
        root_angular_velocity=torch.stack((base + 40.0, base + 41.0, base + 42.0), dim=-1),
        # Construction has already reordered source (joint_a, joint_b) to the
        # live simulator order (joint_b, joint_a).
        joint_position=torch.stack((base + 20.0, base + 10.0), dim=-1),
        joint_velocity=torch.stack((base + 60.0, base + 50.0), dim=-1),
    )


def _synthetic_joint_q(clip_number: int, frame_count: int, device: str | torch.device) -> torch.Tensor:
    """Return one valid free-root corpus in the synthetic target's Newton order."""
    joint_q = torch.zeros((frame_count, 7 + len(_LIVE_JOINT_NAMES)), dtype=torch.float32, device=device)
    joint_q[:, 0] = float(clip_number)
    joint_q[:, 6] = 1.0
    return joint_q


class _SyntheticMotionClip:
    """Small exact-coordinate source clip used by table-construction tests."""

    def __init__(self, clip_number: int, frame_count: int, source_fps: float) -> None:
        self.clip_number = clip_number
        self.frame_count = frame_count
        self.source_fps = source_fps

    def free_root_coordinates(
        self, source_skeleton: MotionSkeleton, *, device: str | torch.device
    ) -> tuple[torch.Tensor, None]:
        del source_skeleton
        return _synthetic_joint_q(self.clip_number, self.frame_count, device), None

    def local_pose(
        self, source_skeleton: MotionSkeleton, *, device: str | torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del source_skeleton, device
        raise AssertionError("The exact-coordinate fixture must not decode semantic poses.")


class _SyntheticFrameTarget:
    """One synthetic robot target owning coordinate storage and materialization."""

    version = "test_target_v1"
    construction_identity_sha256 = _hash("test-target-construction")
    coordinate_profile_sha256 = _hash("test-target-coordinate-profile")
    joint_names = _LIVE_JOINT_NAMES
    collision_geometry_identity_sha256 = _hash("test-target-collision-geometry")
    joint_q_indices = (7, 8)
    reference_frame_names: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        coordinate_refs: list[weakref.ReferenceType[torch.Tensor]] | None = None,
        coordinate_live_scalars: list[int] | None = None,
    ) -> None:
        self.coordinate_refs = [] if coordinate_refs is None else coordinate_refs
        self.coordinate_live_scalars = [] if coordinate_live_scalars is None else coordinate_live_scalars
        self.allocation_sizes: list[int] = []
        self.materialize_calls = 0
        self.live_coordinate_scalars_at_materialization = 0
        self._device = torch.device("cpu")
        self._kinematic_tree = SimpleNamespace(
            num_coordinates=len(_LIVE_JOINT_NAMES),
            root_body_index=0,
            coordinate_q_indices=(7, 8),
            coordinate_qd_indices=(6, 7),
            coordinate_lower_limits_rad=(-1.0e6, -1.0e6),
            coordinate_upper_limits_rad=(1.0e6, 1.0e6),
        )
        self._kinematics = SimpleNamespace(
            n_root_coords=7,
            device=self._device,
            model=SimpleNamespace(
                joint_coord_count=9,
                joint_dof_count=8,
                body_count=1,
                body_com=wp.array([(0.0, 0.0, 0.0)], dtype=wp.vec3, device="cpu"),
            ),
            builder=object(),
            default_joint_q=[0.0] * 9,
            topology=SimpleNamespace(
                joint_velocity_lower=np.full(8, -1.0e6, dtype=np.float32),
                joint_velocity_upper=np.full(8, 1.0e6, dtype=np.float32),
            ),
            eval_fk_batched_torch=self._eval_fk_batched_torch,
        )

    @property
    def materialization_minimum_frames(self) -> int:
        """Minimum complete synthetic clip length."""
        return 1

    def trajectory_seed_joint_q(
        self,
        *,
        root_position_m: torch.Tensor,
        rotation_body_indices: tuple[int, ...],
        landmark_rotation_xyzw: torch.Tensor,
    ) -> torch.Tensor:
        """Seed the synthetic free root while leaving its two nonroot coordinates at zero."""
        frame_count = root_position_m.shape[0] if root_position_m.ndim == 2 else -1
        if (
            frame_count < 1
            or root_position_m.shape != (frame_count, 3)
            or rotation_body_indices != (0,)
            or landmark_rotation_xyzw.shape != (1, frame_count, 4)
            or root_position_m.dtype is not torch.float32
            or landmark_rotation_xyzw.dtype is not torch.float32
            or root_position_m.device != self._device
            or landmark_rotation_xyzw.device != self._device
            or not bool(torch.all(torch.isfinite(root_position_m)))
            or not bool(torch.all(torch.isfinite(landmark_rotation_xyzw)))
        ):
            raise ValueError("Synthetic trajectory seed requires one finite target-root pose per frame.")
        joint_q = torch.zeros((frame_count, 9), dtype=torch.float32, device=self._device)
        joint_q[:, :3].copy_(root_position_m)
        joint_q[:, 3:7].copy_(landmark_rotation_xyzw[0])
        return joint_q

    def allocate_coordinates(self, frame_count: int, *, device: str | torch.device) -> MotionGeneralizedCoordinates:
        self._device = torch.device(device)
        assert self._device.type in ("cpu", "cuda")
        self._kinematics.device = self._device
        coordinates = MotionGeneralizedCoordinates(
            torch.empty((frame_count, 9), dtype=torch.float32, device=device), None
        )
        self.coordinate_refs.append(weakref.ref(coordinates.joint_q))
        self.allocation_sizes.append(frame_count)
        live_tensors = (reference() for reference in self.coordinate_refs)
        live_storages = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() // tensor.element_size()
            for tensor in live_tensors
            if tensor is not None
        }
        self.coordinate_live_scalars.append(sum(live_storages.values()))
        return coordinates

    @property
    def kinematics(self):
        return self._kinematics

    @property
    def kinematic_tree(self):
        return self._kinematic_tree

    @property
    def collision_probe_body_indices(self) -> torch.Tensor:
        return torch.tensor((0,), dtype=torch.int64, device=self._device)

    @property
    def collision_probe_offsets_m(self) -> torch.Tensor:
        return torch.zeros((1, 3), dtype=torch.float32, device=self._device)

    @property
    def collision_probe_contact_slots(self) -> torch.Tensor:
        return torch.full((1,), -1, dtype=torch.int64, device=self._device)

    @property
    def collision_probe_normal_channel_slots(self) -> torch.Tensor:
        return torch.full((1,), -1, dtype=torch.int64, device=self._device)

    @staticmethod
    def _eval_fk_batched_torch(
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor,
        body_q: torch.Tensor,
        body_qd: torch.Tensor,
    ) -> None:
        """Evaluate the synthetic one-body free root without allocating."""
        del joint_qd
        body_q[:, 0].copy_(joint_q[:, :7])
        body_qd.zero_()

    def coordinates_from_newton(
        self, joint_q: torch.Tensor, clip_index: MotionClipIndex
    ) -> MotionGeneralizedCoordinates:
        if joint_q.shape != (clip_index.total_frames, 9):
            raise ValueError("Synthetic Newton coordinates differ from the declared corpus.")
        return MotionGeneralizedCoordinates(joint_q.contiguous(), None)

    def write_joint_position_newton(self, coordinates: MotionGeneralizedCoordinates, output: torch.Tensor) -> None:
        """Write synthetic coordinates without changing their representation."""
        output.copy_(coordinates.joint_q)

    def write_nonroot_velocity_canonical(
        self,
        joint_q: torch.Tensor,
        clip_offsets: torch.Tensor,
        step_seconds: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Write zero synthetic target velocities into Newton storage."""
        del joint_q, clip_offsets, step_seconds
        output[:, 6:].zero_()

    def materialize_coordinates(
        self, coordinates: MotionGeneralizedCoordinates, clip_index: MotionClipIndex
    ) -> MotionFrames:
        self.materialize_calls += 1
        tensors = [reference() for reference in self.coordinate_refs]
        tensors.append(coordinates.joint_q)
        storages = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() // tensor.element_size()
            for tensor in tensors
            if tensor is not None
        }
        self.live_coordinate_scalars_at_materialization = sum(storages.values())
        frames = _empty_frames(clip_index.total_frames)
        for clip_index_value, clip in enumerate(clip_index.clips):
            start, stop = clip_index.offsets[clip_index_value : clip_index_value + 2]
            clip_number = int(coordinates.joint_q[start, 0])
            frames._copy_clip_(start, stop, _clip_frames(clip_number, clip.frame_count))
        return frames


def _SyntheticExactProjection(
    source_skeleton: MotionSkeleton,
    target: _SyntheticFrameTarget,
    *,
    failure: str | None = None,
    nonfinite_clip: int | None = None,
) -> MotionSourceProjectionExact:
    """Return one concrete exact source route with optional failure injection."""

    def convert(
        joint_q: torch.Tensor, joint_qd: torch.Tensor | None, source_fps: float
    ) -> MotionGeneralizedCoordinates:
        del joint_qd, source_fps
        if failure is not None:
            raise RuntimeError(failure)
        coordinates = joint_q.clone()
        if int(joint_q[0, 0]) == nonfinite_clip:
            coordinates[0, 0] = torch.nan
        return MotionGeneralizedCoordinates(coordinates, None)

    return MotionSourceProjectionExact(
        source_skeleton=source_skeleton,
        target=target,
        version="test_builder_v1",
        construction_identity_sha256=_hash("test-builder-construction"),
        convert_coordinates=convert,
    )


_CONTACT_CHANNELS = (
    MotionTaskTableCfg.ContactChannelCfg(name="left_foot", source_probe_roles=("left_ankle", "left_toe")),
    MotionTaskTableCfg.ContactChannelCfg(name="right_foot", source_probe_roles=("right_ankle", "right_toe")),
)


def _fixed_projection_factory(projection: object) -> Callable:
    """Return one source-projection factory that always yields the same fixture."""

    def factory(*_args):
        return projection

    return factory


def _synthetic_target_kinematics(
    target: _SyntheticFrameTarget,
    source_projection_factory: Callable,
) -> MotionTaskTableCfg.TargetKinematicsCfg:
    """Bind one synthetic target and a caller-owned source projection factory."""

    def target_factory(
        _reference,
        _contact_patches,
        *,
        calibration_artifact_root: str,
        calibration: MotionTaskTableCfg.TargetKinematicsCfg.CalibrationCfg | None,
    ) -> _SyntheticFrameTarget:
        del calibration_artifact_root
        assert calibration is None
        return target

    return MotionTaskTableCfg.TargetKinematicsCfg(
        target_factory=target_factory,
        source_projection_factory=source_projection_factory,
        physics_types=(object,),
        contact_patches=(
            MotionTaskTableCfg.TargetKinematicsCfg.ContactPatchCfg(channel="left_foot", body_name="left_support"),
            MotionTaskTableCfg.TargetKinematicsCfg.ContactPatchCfg(channel="right_foot", body_name="right_support"),
        ),
    )


def _table(*, task_row_mode: str = "source_frames") -> MotionTaskTable:
    index = _index()
    frames = _empty_frames(index.total_frames)
    for clip_number, clip in enumerate(index.clips):
        start, end = index.offsets[clip_number : clip_number + 2]
        frames._copy_clip_(start, end, _clip_frames(clip_number, clip.frame_count))
    return motion_task_table(
        index,
        frames,
        _LIVE_JOINT_NAMES,
        (),
        "test_builder_v1",
        _hash("test-builder-construction"),
        task_row_mode,
        "test_decoder_v1",
    )


def _sampler(
    table: MotionTaskTable,
    *,
    reset_sources: tuple[tuple[str, float], ...] = (("reference", 1.0),),
) -> MotionSampler:
    return MotionSampler(table, reset_sources, capacity=1024, seed=7301)


class _Robot:
    joint_names = _LIVE_JOINT_NAMES

    def __init__(self, num_envs: int) -> None:
        self.root_state = torch.zeros(num_envs, 13)
        self.joint_position = torch.zeros(num_envs, 2)
        self.joint_velocity = torch.zeros(num_envs, 2)
        self.root_velocity_frame: str | None = None

    def write_root_link_pose_to_sim_index(self, *, root_pose: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.root_state[env_ids, :7] = root_pose

    def write_root_link_velocity_to_sim_index(self, *, root_velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.root_state[env_ids, 7:] = root_velocity
        self.root_velocity_frame = "link"

    def write_root_com_velocity_to_sim_index(self, *, root_velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.root_state[env_ids, 7:] = root_velocity
        self.root_velocity_frame = "center_of_mass"

    def write_joint_position_to_sim_index(self, *, position: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.joint_position[env_ids] = position

    def write_joint_velocity_to_sim_index(self, *, velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.joint_velocity[env_ids] = velocity


class _Scene:
    def __init__(self, robot: _Robot, origins: torch.Tensor) -> None:
        self.robot = robot
        self.env_origins = origins

    def __getitem__(self, name: str) -> _Robot:
        assert name == "robot"
        return self.robot


class _IdentityReset:
    """Return resolved reference rows unchanged for payload-only tests."""

    def __init__(self, reset_source_names: tuple[str, ...]) -> None:
        self.reset_source_names = reset_source_names

    def __call__(
        self,
        reference: MotionResetState,
        reset_source_indices: torch.Tensor,
        generator: torch.Generator,
    ) -> MotionResetState:
        del reset_source_indices, generator
        return reference


def _payload(
    num_envs: int,
    *,
    task_row_mode: str = "source_frames",
    states_relative: bool = True,
    reset_sources: tuple[tuple[str, float], ...] = (("reference", 1.0),),
    reset_transform_factory: Callable[..., object] | None = None,
    root_velocity_frame: str = "link",
) -> tuple[MotionStatePayload, MotionTaskTable, _Robot]:
    table = _table(task_row_mode=task_row_mode)
    robot = _Robot(num_envs)
    origins = torch.zeros(num_envs, 3)
    origins[:, 0] = torch.arange(num_envs, dtype=torch.float32) * 10.0
    env = SimpleNamespace(
        cfg=SimpleNamespace(seed=7301),
        device=torch.device("cpu"),
        num_envs=num_envs,
        step_dt=0.25,
        scene=_Scene(robot, origins),
        obs_buf={},
    )
    if reset_transform_factory is None:
        reset_source_names = tuple(name for name, _probability in reset_sources)

        def identity_reset_factory(**_kwargs: object) -> _IdentityReset:
            return _IdentityReset(reset_source_names)

        reset_transform_factory = identity_reset_factory

    payload_cfg = SimpleNamespace(
        robot_asset_name="robot",
        reset_transform_factory=reset_transform_factory,
        reset_transform_params={},
        reset_transform_binds={},
        root_velocity_frame=root_velocity_frame,
        reset_sources=reset_sources,
    )
    table_cfg = SimpleNamespace(task_row_mode=task_row_mode)
    cfg = SimpleNamespace(payload=payload_cfg, task_table=table_cfg, states_relative=states_relative)
    return MotionStatePayload(cfg, env, table), table, robot


def _bind(payload: MotionStatePayload, task_rows: torch.Tensor) -> None:
    env_ids = torch.arange(task_rows.shape[0], dtype=torch.int64)
    payload.bind(env_ids, task_rows)


def test_motion_payload_contract_contains_only_reset_state() -> None:
    """Motion owns reset materialization and sampling, not success or per-step facade state."""
    assert not hasattr(commands_cfg_module.MotionStatePayloadCfg, "step_fields")
    assert not hasattr(commands_cfg_module.MotionStatePayloadCfg, "command_fields")
    assert hasattr(MotionSampler, "reset_sampling_scope")
    assert not hasattr(MotionStatePayload, "evaluation_scope")
    assert not hasattr(MotionStatePayload, "command_std")
    assert not hasattr(MotionStatePayload, "get_task_done")
    assert not hasattr(MotionStatePayload, "get_task_reward")


def test_uniform_sample_clock_rejects_a_clip_without_a_pre_endpoint_sample() -> None:
    """A one-frame clip must fail before it can create duplicate expert offsets."""
    original = _index()
    index = replace(original, clips=(replace(original.clips[0], frame_count=1),))
    frames = _clip_frames(0, 1)
    table = motion_task_table(
        index,
        frames,
        _LIVE_JOINT_NAMES,
        (),
        "test_builder_v1",
        _hash("test-builder-construction"),
        "source_frames",
        _skeleton().identity_sha256,
    )

    with pytest.raises(ValueError, match="at least one sample before its source endpoint"):
        table.sample("uniform_before_source_end", 0.02)


def test_motion_task_rows_select_exact_reset_state_deterministically() -> None:
    """Explicit descriptor rows must materialize the same robot reset state every time."""
    payload, table, robot = _payload(3)
    task_rows = torch.tensor((4, 0, 2))
    _bind(payload, task_rows)

    assert table.num_tasks == 7
    expected_root = torch.tensor(((101.0, 102.0, 103.0), (10.0, 1.0, 2.0), (22.0, 3.0, 4.0)))
    expected_joint = torch.tensor(((121.0, 111.0), (20.0, 10.0), (22.0, 12.0)))
    torch.testing.assert_close(robot.root_state[:, :3], expected_root)
    torch.testing.assert_close(robot.joint_position, expected_joint)

    _bind(payload, task_rows)
    torch.testing.assert_close(robot.root_state[:, :3], expected_root)
    torch.testing.assert_close(robot.joint_position, expected_joint)


def test_motion_clip_start_rows_use_normal_binding_and_reset_semantics() -> None:
    """Clip-start descriptor rows select exact source time through ordinary binding."""
    payload, table, robot = _payload(2)
    env_ids = torch.tensor((0, 1), dtype=torch.int64)
    clip_indices = torch.tensor((1, 0), dtype=torch.int64)

    payload.bind(env_ids, table.clip_start_rows.index_select(0, clip_indices))

    torch.testing.assert_close(
        robot.root_state[:, :3],
        torch.tensor(((100.0, 101.0, 102.0), (10.0, 1.0, 2.0))),
    )
    torch.testing.assert_close(
        robot.joint_position,
        torch.tensor(((120.0, 110.0), (20.0, 10.0))),
    )


def test_reset_sampling_scope_selects_exact_starts_and_restores_sampler_state() -> None:
    """Evaluation reset policy belongs to the sampler and leaves training draws unchanged."""
    table = _table(task_row_mode="clip_time_ranges")
    sampler = _sampler(table, reset_sources=(("reference", 0.25), ("alternative", 0.75)))
    ranges = table.reset_time_ranges_seconds
    generator_state = sampler.generator.get_state().clone()
    probabilities = sampler.reset_source_probabilities.clone()

    with sampler.reset_sampling_scope(19, "reference"):
        assert sampler.reset_time_mode == "range_start"
        torch.testing.assert_close(sampler.reset_source_probabilities, torch.tensor((1.0, 0.0)))
        reset_times = torch.empty(ranges.shape[0])
        sampler.sample_reset_times(ranges, reset_times)
        torch.testing.assert_close(reset_times, ranges[:, 0])

    assert sampler.reset_time_mode == "uniform"
    torch.testing.assert_close(sampler.reset_source_probabilities, probabilities)
    torch.testing.assert_close(sampler.generator.get_state(), generator_state)


def test_motion_payload_writes_reset_velocity_in_declared_root_frame() -> None:
    """G1-style reset velocities must target the COM field used by legacy root state."""
    payload, _table, robot = _payload(1, root_velocity_frame="center_of_mass")

    _bind(payload, torch.tensor((0,)))

    assert robot.root_velocity_frame == "center_of_mass"
    torch.testing.assert_close(robot.root_state[:, 7:], torch.tensor(((30.0, 31.0, 32.0, 40.0, 41.0, 42.0),)))


@pytest.mark.parametrize(
    ("route_name", "family_name"),
    (("exact", "native_coordinates"), ("analytic", "direct_coordinates")),
)
def test_motion_production_accepts_declared_stored_coordinate_routes(route_name: str, family_name: str) -> None:
    """Production executes exact and analytic routes through their declared families."""
    index = _index()

    class Source:
        closed = False

        def inspect(self) -> MotionClipIndex:
            return index

        def skeleton(self, skeleton_id: int) -> MotionSkeleton:
            assert skeleton_id == 0
            return _skeleton()

        def clips(self, clip_indices: tuple[int, ...]):
            for clip_number in clip_indices:
                clip = index.clips[clip_number]
                yield clip_number, _SyntheticMotionClip(clip_number, clip.frame_count, clip.source_fps)

        def close(self) -> None:
            self.closed = True

    source = Source()
    target = _SyntheticFrameTarget()
    if route_name == "exact":
        projection = _SyntheticExactProjection(_skeleton(), target)
    else:
        projection = MotionSourceProjectionAnalytic(
            source_skeleton=_skeleton(),
            target=target,
            version="test_analytic_v1",
            construction_identity_sha256=_hash("test-analytic-construction"),
            output_clip_index=lambda source_index: source_index,
            convert_clip=lambda clip: MotionGeneralizedCoordinates(
                _synthetic_joint_q(clip.clip_number, clip.frame_count, "cpu"), None
            ),
        )
    split = SimpleNamespace(
        source_content_sha256=index.source_content_sha256,
        clip_count=len(index.clips),
        frame_count=index.total_frames,
    )
    source_cfg = SimpleNamespace(
        purpose="production",
        train=split,
        evaluation=split,
        open_split=lambda _root, _split: source,
        decoder_version="test_decoder_v1",
    )
    cfg = MotionTaskTableCfg(
        source=source_cfg,
        contact_channels=_CONTACT_CHANNELS,
        source_artifact_root="unused",
        motion_split="train",
        target_kinematics=_synthetic_target_kinematics(target, _fixed_projection_factory(projection)),
        families=(
            commands_cfg_module.MotionExactFamilyCfg(name="native_coordinates"),
            commands_cfg_module.MotionAnalyticFamilyCfg(name="direct_coordinates"),
            commands_cfg_module.MotionTrajectoryFamilyCfg(name="trajectory_solve"),
        ),
        task_row_mode="clip_time_ranges",
    )
    table = _build_synthetic_task_table(cfg, "cpu")

    assert table.family_name == family_name
    assert table.construction_version == projection.version
    assert table.clip_index.total_frames == index.total_frames
    assert target.allocation_sizes == [index.total_frames, max(clip.frame_count for clip in index.clips)]
    assert source.closed


def test_motion_analytic_capacity_uses_declared_output_clock_when_upsampling() -> None:
    """Analytic capacity follows its declared output index, not raw input frame count."""
    source_index = MotionClipIndex(
        source_content_sha256=_hash("analytic-upsample-source"),
        skeleton_identity_sha256s=(_skeleton().identity_sha256,),
        clips=(MotionClipIndex.Clip("clip", 2, 10.0, _hash("analytic-upsample-clip"), 0),),
    )

    class Source:
        def inspect(self) -> MotionClipIndex:
            return source_index

        def skeleton(self, skeleton_id: int) -> MotionSkeleton:
            assert skeleton_id == 0
            return _skeleton()

        def clips(self, clip_indices: tuple[int, ...]):
            assert clip_indices == (0,)
            yield 0, _SyntheticMotionClip(0, 2, 10.0)

        def close(self) -> None:
            pass

    def output_index(index: MotionClipIndex) -> MotionClipIndex:
        clip = index.clips[0]
        return MotionClipIndex(
            source_content_sha256=index.source_content_sha256,
            skeleton_identity_sha256s=index.skeleton_identity_sha256s,
            clips=(replace(clip, frame_count=4, source_fps=20.0),),
        )

    def convert_clip(clip: _SyntheticMotionClip) -> MotionGeneralizedCoordinates:
        return MotionGeneralizedCoordinates(_synthetic_joint_q(clip.clip_number, 4, "cpu"), None)

    target = _SyntheticFrameTarget()
    projection = MotionSourceProjectionAnalytic(
        source_skeleton=_skeleton(),
        target=target,
        version="test_analytic_v1",
        construction_identity_sha256=_hash("test-analytic-construction"),
        output_clip_index=output_index,
        convert_clip=convert_clip,
    )
    split = SimpleNamespace(source_content_sha256=source_index.source_content_sha256, clip_count=1, frame_count=2)
    cfg = MotionTaskTableCfg(
        source=SimpleNamespace(
            train=split,
            purpose="oracle",
            evaluation=split,
            open_split=lambda _root, _split: Source(),
            decoder_version="analytic_upsample_v1",
        ),
        contact_channels=_CONTACT_CHANNELS,
        source_artifact_root="unused",
        motion_split="train",
        target_kinematics=_synthetic_target_kinematics(target, _fixed_projection_factory(projection)),
        task_row_mode="clip_time_ranges",
    )

    inspection = _inspect_synthetic_task_table(cfg, "cpu", sequence_limit=1)

    assert target.allocation_sizes == [4, 4]
    assert target.coordinate_live_scalars == [36, 72]
    assert target.live_coordinate_scalars_at_materialization == 36
    torch.testing.assert_close(inspection.sequences.offsets, torch.tensor((0, 4)))
    torch.testing.assert_close(inspection.sequences.frame_dt, torch.tensor((0.05,)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TorchScript fusion.")
def test_motion_table_native_smpl_is_invariant_to_torchscript_cache_phase() -> None:
    """Native SMPL corpus values must not depend on cold versus fused scripted math."""
    from isaaclab.utils.math import quat_apply

    from isaaclab_tasks.core.multi_task.motion.data.sources import CmuHumEnvSmplClip, cmu_humenv_smpl_skeleton
    from isaaclab_tasks.core.multi_task.motion.robots.smpl.articulation import SMPL_MOTION_ARTICULATION_CFG
    from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import (
        smpl_frame_target,
        smpl_source_projection,
    )

    frame_count = 257
    time = np.arange(frame_count, dtype=np.float32) / np.float32(30.0)
    coordinate = np.arange(69, dtype=np.float32)[None]
    angle = np.float32(0.2) * np.sin(np.float32(0.3) * time)
    generalized_position = np.zeros((frame_count, 76), dtype=np.float32)
    generalized_position[:, 2] = 1.0
    generalized_position[:, 3] = np.cos(angle)
    generalized_position[:, 6] = np.sin(angle)
    generalized_velocity = np.zeros((frame_count, 75), dtype=np.float32)
    generalized_velocity[:, :3] = np.stack(
        (
            np.float32(0.2) * np.cos(time),
            -np.float32(0.07) * np.sin(np.float32(0.7) * time),
            np.float32(0.039) * np.cos(np.float32(1.3) * time),
        ),
        axis=-1,
    )
    generalized_velocity[:, 3:6] = np.stack(
        (
            np.float32(0.1) * np.sin(np.float32(0.2) * time),
            np.float32(0.2) * np.cos(np.float32(0.4) * time),
            np.float32(0.3) * np.sin(np.float32(0.6) * time),
        ),
        axis=-1,
    )
    generalized_velocity[:, 6:] = np.float32(0.01) * np.cos(
        time[:, None] * (np.float32(0.3) + coordinate * np.float32(0.01)) + coordinate * np.float32(0.02)
    )
    clip = CmuHumEnvSmplClip(generalized_position, generalized_velocity, 30.0)
    index = MotionClipIndex(
        source_content_sha256=_hash("native-smpl-cache-phase-source"),
        skeleton_identity_sha256s=(cmu_humenv_smpl_skeleton().identity_sha256,),
        clips=(
            MotionClipIndex.Clip(
                clip_id="native_smpl",
                frame_count=frame_count,
                source_fps=30.0,
                content_sha256=_hash("native-smpl-cache-phase-clip"),
                skeleton_id=0,
            ),
        ),
    )

    class Source:
        def inspect(self) -> MotionClipIndex:
            return index

        def skeleton(self, skeleton_id: int) -> MotionSkeleton:
            assert skeleton_id == 0
            return cmu_humenv_smpl_skeleton()

        def clips(self, clip_indices: tuple[int, ...]):
            assert clip_indices == (0,)
            yield 0, clip

        def close(self) -> None:
            pass

    split = SimpleNamespace(
        source_content_sha256=index.source_content_sha256,
        clip_count=1,
        frame_count=frame_count,
    )
    source_cfg = SimpleNamespace(
        purpose="production",
        train=split,
        evaluation=split,
        open_split=lambda _root, _split: Source(),
        decoder_version="test_decoder_v1",
    )
    cfg = MotionTaskTableCfg(
        source=source_cfg,
        contact_channels=_CONTACT_CHANNELS,
        source_artifact_root="unused",
        motion_split="train",
        target_kinematics=MotionTaskTableCfg.TargetKinematicsCfg(
            target_factory=smpl_frame_target,
            source_projection_factory=smpl_source_projection,
            physics_types=(object,),
            contact_patches=(
                MotionTaskTableCfg.TargetKinematicsCfg.ContactPatchCfg(
                    channel="left_foot",
                    body_name="L_Ankle",
                ),
                MotionTaskTableCfg.TargetKinematicsCfg.ContactPatchCfg(
                    channel="right_foot",
                    body_name="R_Ankle",
                ),
            ),
        ),
        task_row_mode="clip_time_ranges",
    )

    quat_apply._debug_flush_compilation_cache()
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    scene_cfg = SimpleNamespace(robot=SMPL_MOTION_ARTICULATION_CFG)
    first = build_motion_task_table(SimpleNamespace(task_table=cfg), scene_cfg, "cuda:0")
    second = build_motion_task_table(SimpleNamespace(task_table=cfg), scene_cfg, "cuda:0")

    assert first.frames.stored_fields == second.frames.stored_fields
    for name in first.frames.stored_fields:
        assert torch.equal(first.field(name), second.field(name)), name
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TorchScript fusion.")
def test_motion_table_g1_angular_velocity_is_invariant_to_torchscript_cache_phase() -> None:
    """G1-shaped quaternion differentiation must be stable across graph phases."""
    from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

    from isaaclab_tasks.core.multi_task.kinematics import time_gaussian_filter, time_quaternion_angular_velocity

    index = MotionClipIndex(
        source_content_sha256=_hash("g1-angular-cache-phase-source"),
        skeleton_identity_sha256s=(_skeleton().identity_sha256,),
        clips=(
            MotionClipIndex.Clip(
                clip_id="g1_angular",
                frame_count=257,
                source_fps=30.0,
                content_sha256=_hash("g1-angular-cache-phase-clip"),
                skeleton_id=0,
            ),
        ),
    )

    class Source:
        def inspect(self) -> MotionClipIndex:
            return index

        def skeleton(self, skeleton_id: int) -> MotionSkeleton:
            assert skeleton_id == 0
            return _skeleton()

        def clips(self, clip_indices: tuple[int, ...]):
            assert clip_indices == (0,)
            yield 0, _SyntheticMotionClip(0, 257, 30.0)

        def close(self) -> None:
            pass

    class G1AngularTarget(_SyntheticFrameTarget):
        def materialize_coordinates(
            self,
            coordinates: MotionGeneralizedCoordinates,
            clip_index: MotionClipIndex,
        ) -> MotionFrames:
            del coordinates
            frame_count = clip_index.total_frames
            source_fps = clip_index.clips[0].source_fps
            time = torch.arange(frame_count, dtype=torch.float32, device="cuda:0") / source_fps
            angle = 0.7 * torch.sin(0.4 * time)
            root_rotation = torch.zeros(frame_count, 4, dtype=torch.float32, device="cuda:0")
            root_rotation[:, 2] = torch.sin(0.5 * angle)
            root_rotation[:, 3] = torch.cos(0.5 * angle)
            body_rotation = root_rotation[:, None].expand(frame_count, 31, 4).contiguous()
            body_angular_velocity = time_gaussian_filter(
                time_quaternion_angular_velocity(body_rotation.unsqueeze(0), 1.0 / source_fps)
            ).squeeze(0)
            return MotionFrames(
                root_position=torch.zeros(frame_count, 3, device="cuda:0"),
                root_rotation=root_rotation,
                root_linear_velocity=torch.zeros(frame_count, 3, device="cuda:0"),
                root_angular_velocity=body_angular_velocity[:, 0].contiguous(),
                joint_position=torch.zeros(frame_count, 2, device="cuda:0"),
                joint_velocity=torch.zeros(frame_count, 2, device="cuda:0"),
            )

    target = G1AngularTarget()
    projection = _SyntheticExactProjection(_skeleton(), target)
    split = SimpleNamespace(source_content_sha256=index.source_content_sha256, clip_count=1, frame_count=257)
    source_cfg = SimpleNamespace(
        purpose="production",
        train=split,
        evaluation=split,
        open_split=lambda _root, _split: Source(),
        decoder_version="test_decoder_v1",
    )
    cfg = MotionTaskTableCfg(
        source=source_cfg,
        contact_channels=_CONTACT_CHANNELS,
        source_artifact_root="unused",
        motion_split="train",
        target_kinematics=_synthetic_target_kinematics(target, _fixed_projection_factory(projection)),
        task_row_mode="clip_time_ranges",
    )

    for function in (quat_mul, quat_conjugate, axis_angle_from_quat):
        function._debug_flush_compilation_cache()
    first = _build_synthetic_task_table(cfg, "cuda:0")
    second = _build_synthetic_task_table(cfg, "cuda:0")

    for name in first.frames.stored_fields:
        assert torch.equal(first.field(name), second.field(name)), name


def test_motion_source_iteration_rejects_a_clock_changed_after_inspection() -> None:
    """Decoded clips must use the exact sample rate retained by the source index."""
    index = _index()

    class Source:
        def clips(self, clip_indices: tuple[int, ...]):
            for clip_number in clip_indices:
                clip = index.clips[clip_number]
                source_fps = 3.0 if clip_number == 0 else clip.source_fps
                yield clip_number, _SyntheticMotionClip(clip_number, clip.frame_count, source_fps)

    candidate = table_builder_module._MotionExactSourceCandidate(
        target=object(),
        projections=(object(),),
        projection_indices=(0, 0),
        source=Source(),
        source_index=index,
        output_index=index,
        source_clip_indices=tuple(range(len(index.clips))),
        device="cpu",
        coordinates=MotionGeneralizedCoordinates(torch.empty((index.total_frames, 1)), None),
    )

    with pytest.raises(ValueError, match="clock or identity"):
        tuple(trajectory_module._source_clips(candidate))


def test_motion_inspection_retains_pass_fail_pass_without_compaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspection retains every source clip and its original acceptance result."""
    index = MotionClipIndex(
        source_content_sha256=_hash("pass-fail-pass-source"),
        skeleton_identity_sha256s=(_skeleton().identity_sha256,),
        clips=tuple(
            MotionClipIndex.Clip(
                clip_id=f"clip_{index}",
                frame_count=frame_count,
                source_fps=2.0,
                content_sha256=_hash(f"pass-fail-pass-{index}"),
                skeleton_id=0,
            )
            for index, frame_count in enumerate((2, 3, 4))
        ),
    )

    class Source:
        def inspect(self) -> MotionClipIndex:
            return index

        def skeleton(self, skeleton_id: int) -> MotionSkeleton:
            assert skeleton_id == 0
            return _skeleton()

        def clips(self, clip_indices: tuple[int, ...]):
            for clip_number in clip_indices:
                clip = index.clips[clip_number]
                yield clip_number, _SyntheticMotionClip(clip_number, clip.frame_count, clip.source_fps)

        def close(self) -> None:
            pass

    target = _SyntheticFrameTarget()
    projection = _SyntheticExactProjection(_skeleton(), target)
    coordinate_finite = table_builder_module.motion_criterion_target_coordinates

    def reject_middle(cfg, candidate, rows):
        return coordinate_finite(cfg, candidate, rows) & (rows != 1)

    monkeypatch.setattr(table_builder_module, "motion_criterion_target_coordinates", reject_middle)
    split = SimpleNamespace(
        source_content_sha256=index.source_content_sha256,
        clip_count=len(index.clips),
        frame_count=index.total_frames,
    )
    cfg = MotionTaskTableCfg(
        source=SimpleNamespace(
            purpose="oracle",
            train=split,
            evaluation=split,
            open_split=lambda _root, _split: Source(),
            decoder_version="test_decoder_v1",
        ),
        contact_channels=_CONTACT_CHANNELS,
        source_artifact_root="unused",
        motion_split="train",
        target_kinematics=_synthetic_target_kinematics(target, _fixed_projection_factory(projection)),
        task_row_mode="clip_time_ranges",
    )

    inspection = _inspect_synthetic_task_table(cfg, "cpu", sequence_limit=3)

    torch.testing.assert_close(inspection.sequences.offsets, torch.tensor((0, 2, 5, 9)))
    torch.testing.assert_close(
        inspection.state_bank.root_pose[:, 0, 0],
        torch.tensor((0.0, 1.0, 100.0, 101.0, 102.0, 200.0, 201.0, 202.0, 203.0)),
    )
    accepted = inspection.quality.values[:, inspection.quality.names.index("accepted")].bool()
    torch.testing.assert_close(accepted, torch.tensor((True, False, True)))
    rejected = next(point for point in inspection.points if point.name == "rejected_frames")
    torch.testing.assert_close(
        rejected.valid.squeeze(1),
        torch.tensor((False, False, True, True, True, False, False, False, False)),
    )


def test_motion_oracle_inspection_retains_rejected_multiskeleton_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oracle inspection retains every source candidate and its original rejection marker."""
    skeletons = (
        _skeleton(),
        replace(_skeleton(), identifier="motion_command_test_survivor", content_sha256=_hash("survivor-skeleton")),
        replace(_skeleton(), identifier="motion_command_test_late", content_sha256=_hash("late-skeleton")),
    )
    index = MotionClipIndex(
        source_content_sha256=_hash("rejected-group-source"),
        skeleton_identity_sha256s=tuple(skeleton.identity_sha256 for skeleton in skeletons),
        clips=tuple(
            MotionClipIndex.Clip(
                clip_id=f"clip_{clip_index}",
                frame_count=2,
                source_fps=10.0,
                content_sha256=_hash(f"rejected-group-{clip_index}"),
                skeleton_id=clip_index,
            )
            for clip_index in range(3)
        ),
    )

    class Source:
        def inspect(self) -> MotionClipIndex:
            return index

        def skeleton(self, skeleton_id: int) -> MotionSkeleton:
            return skeletons[skeleton_id]

        def clips(self, clip_indices: tuple[int, ...]):
            for clip_index in clip_indices:
                clip = index.clips[clip_index]
                yield clip_index, _SyntheticMotionClip(clip_index, clip.frame_count, clip.source_fps)

        def close(self) -> None:
            pass

    target = _SyntheticFrameTarget()

    def source_projection(
        source_skeleton: MotionSkeleton,
        _target: object,
        _source: object,
        _contact_channels: object,
        _contact_offsets: torch.Tensor,
    ):
        return _SyntheticExactProjection(source_skeleton, target)

    coordinate_finite = table_builder_module.motion_criterion_target_coordinates

    def reject_first_skeleton(cfg, candidate, rows):
        accepted = coordinate_finite(cfg, candidate, rows)
        clip_starts = torch.tensor(candidate.clip_index.offsets[:-1], dtype=torch.int64, device=rows.device)
        clip_numbers = candidate.coordinates.joint_q.index_select(0, clip_starts)[:, 0].to(torch.int64)
        return accepted & (clip_numbers[rows] != 0)

    monkeypatch.setattr(table_builder_module, "motion_criterion_target_coordinates", reject_first_skeleton)

    split = SimpleNamespace(
        source_content_sha256=index.source_content_sha256,
        clip_count=len(index.clips),
        frame_count=index.total_frames,
    )
    source_cfg = SimpleNamespace(
        purpose="oracle",
        train=split,
        evaluation=split,
        open_split=lambda _root, _split: Source(),
        decoder_version="rejected_group_decoder_v1",
    )
    cfg = MotionTaskTableCfg(
        source=source_cfg,
        contact_channels=_CONTACT_CHANNELS,
        source_artifact_root="unused",
        motion_split="train",
        target_kinematics=_synthetic_target_kinematics(target, source_projection),
        families=(
            commands_cfg_module.MotionExactFamilyCfg(),
            commands_cfg_module.MotionAnalyticFamilyCfg(),
            commands_cfg_module.MotionTrajectoryFamilyCfg(),
        ),
        task_row_mode="clip_time_ranges",
    )

    diagnostic_calls = []
    populate_contact_quality = trajectory_module._populate_motion_contact_quality
    record_calls = []
    store_motion_route = table_builder_module._store_motion_route

    def track_stored_records(*args):
        store_motion_route(*args)
        records = args[5]
        record_calls.append(tuple(record.source_clip_index for record in records))

    monkeypatch.setattr(table_builder_module, "_store_motion_route", track_stored_records)

    def track_contact_quality(*args):
        diagnostic_calls.append("inspection")
        return populate_contact_quality(*args)

    monkeypatch.setattr(table_builder_module, "_populate_motion_contact_quality", track_contact_quality)
    inspection = _inspect_synthetic_task_table(cfg, "cpu", sequence_limit=len(index.clips))
    assert diagnostic_calls == ["inspection"]
    assert record_calls == [(0, 1, 2)]

    assert inspection.sequences.sequence_count == 3
    assert inspection.sequences.frame_count == 6
    accepted = inspection.quality.values[:, inspection.quality.names.index("accepted")].bool()
    torch.testing.assert_close(accepted, torch.tensor((False, True, True)))
    rejected = next(point for point in inspection.points if point.name == "rejected_frames")
    torch.testing.assert_close(rejected.valid.squeeze(1), torch.tensor((True, True, False, False, False, False)))

    def reject_every_skeleton(_cfg, _candidate, rows):
        return torch.zeros_like(rows, dtype=torch.bool)

    cfg.families[0].criteria[0].class_type = reject_every_skeleton
    all_rejected = _inspect_synthetic_task_table(cfg, "cpu", sequence_limit=len(index.clips))
    all_rejected_accepted = all_rejected.quality.values[:, all_rejected.quality.names.index("accepted")]
    assert all_rejected.sequences.sequence_count == 3
    assert not bool(torch.any(all_rejected_accepted))
    all_rejected_markers = next(point for point in all_rejected.points if point.name == "rejected_frames")
    assert bool(torch.all(all_rejected_markers.valid))
    assert diagnostic_calls == ["inspection", "inspection"]
    assert record_calls == [(0, 1, 2), (0, 1, 2)]


def test_motion_inspection_limits_the_source_prefix_before_projection_decode_and_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspection solves exactly source rows ``[0, limit)`` without accepted-row backfill."""
    skeletons = (
        _skeleton(),
        replace(_skeleton(), identifier="motion_command_test_late", content_sha256=_hash("late-skeleton")),
    )
    index = MotionClipIndex(
        source_content_sha256=_hash("inspection-prefix-source"),
        skeleton_identity_sha256s=tuple(skeleton.identity_sha256 for skeleton in skeletons),
        clips=(
            MotionClipIndex.Clip("clip_0", 2, 10.0, _hash("inspection-prefix-clip-0"), 0),
            MotionClipIndex.Clip("clip_1", 3, 10.0, _hash("inspection-prefix-clip-1"), 1),
        ),
    )
    skeleton_calls = []
    clip_calls = []

    class Source:
        def inspect(self) -> MotionClipIndex:
            return index

        def skeleton(self, skeleton_id: int) -> MotionSkeleton:
            skeleton_calls.append(skeleton_id)
            return skeletons[skeleton_id]

        def clips(self, clip_indices: tuple[int, ...]):
            clip_calls.append(clip_indices)
            for clip_index in clip_indices:
                clip = index.clips[clip_index]
                yield clip_index, _SyntheticMotionClip(clip_index, clip.frame_count, clip.source_fps)

        def close(self) -> None:
            pass

    target = _SyntheticFrameTarget()
    projection_calls = []

    def source_projection(source_skeleton: MotionSkeleton, target: object, *_args):
        projection_calls.append(source_skeleton.identifier)
        return _SyntheticExactProjection(source_skeleton, target)

    coordinate_finite = table_builder_module.motion_criterion_target_coordinates

    def reject_first_source(cfg, candidate, rows):
        return coordinate_finite(cfg, candidate, rows) & (rows != 0)

    monkeypatch.setattr(table_builder_module, "motion_criterion_target_coordinates", reject_first_source)
    split = SimpleNamespace(
        source_content_sha256=index.source_content_sha256,
        clip_count=len(index.clips),
        frame_count=index.total_frames,
    )
    cfg = MotionTaskTableCfg(
        source=SimpleNamespace(
            purpose="production",
            train=split,
            evaluation=split,
            open_split=lambda _root, _split: Source(),
            decoder_version="inspection_prefix_decoder_v1",
        ),
        contact_channels=_CONTACT_CHANNELS,
        source_artifact_root="unused",
        motion_split="train",
        target_kinematics=_synthetic_target_kinematics(target, source_projection),
        task_row_mode="clip_time_ranges",
    )

    inspection = _inspect_synthetic_task_table(cfg, "cpu", sequence_limit=1)

    assert skeleton_calls == [0, 1]
    assert projection_calls == [skeletons[0].identifier]
    assert clip_calls == [(0,)]
    assert target.allocation_sizes == [2, 2]
    assert inspection.sequences.sequence_count == 1
    accepted = inspection.quality.values[:, inspection.quality.names.index("accepted")]
    torch.testing.assert_close(accepted, torch.zeros(1))


def test_motion_task_table_rejects_incomplete_source_projection_before_allocation() -> None:
    """The source table validates each projection before allocating coordinate storage."""
    index = _index()

    class Source:
        closed = False

        def inspect(self) -> MotionClipIndex:
            return index

        def skeleton(self, skeleton_id: int) -> MotionSkeleton:
            assert skeleton_id == 0
            return _skeleton()

        def clips(self, clip_indices: tuple[int, ...]):
            raise AssertionError("An invalid source projection must fail before consuming source clips.")

        def close(self) -> None:
            self.closed = True

    class IncompleteProjection:
        version = "test_builder_v1"
        construction_identity_sha256 = _hash("test-builder-construction")
        joint_names = _LIVE_JOINT_NAMES
        reference_frame_names: tuple[str, ...] = ()

        def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionFrames:
            raise AssertionError("An invalid frame builder must fail before allocating frames.")

    source = Source()
    split = SimpleNamespace(
        source_content_sha256=index.source_content_sha256,
        clip_count=len(index.clips),
        frame_count=index.total_frames,
    )
    source_cfg = SimpleNamespace(
        purpose="production",
        train=split,
        evaluation=split,
        open_split=lambda _root, _split: source,
        decoder_version="test_decoder_v1",
    )

    def incomplete_projection_factory(*_args):
        return IncompleteProjection()

    cfg = MotionTaskTableCfg(
        source=source_cfg,
        contact_channels=_CONTACT_CHANNELS,
        source_artifact_root="unused",
        motion_split="train",
        target_kinematics=_synthetic_target_kinematics(
            _SyntheticFrameTarget(),
            incomplete_projection_factory,
        ),
        task_row_mode="clip_time_ranges",
    )
    with pytest.raises(TypeError, match="source_projection_factory"):
        _build_synthetic_task_table(cfg, "cpu")
    assert source.closed


def test_motion_task_table_cfg_closes_source_when_frame_construction_fails() -> None:
    """The config-owned source lifetime must end even when one clip cannot build."""
    index = _index()

    class Source:
        closed = False

        def inspect(self) -> MotionClipIndex:
            return index

        def skeleton(self, skeleton_id: int) -> MotionSkeleton:
            assert skeleton_id == 0
            return _skeleton()

        def clips(self, clip_indices: tuple[int, ...]):
            clip_number = clip_indices[0]
            clip = index.clips[clip_number]
            yield clip_number, _SyntheticMotionClip(0, clip.frame_count, clip.source_fps)

        def close(self) -> None:
            self.closed = True

    source = Source()
    split = SimpleNamespace(
        source_content_sha256=index.source_content_sha256,
        clip_count=len(index.clips),
        frame_count=index.total_frames,
    )
    source_cfg = SimpleNamespace(
        purpose="oracle",
        train=split,
        evaluation=split,
        open_split=lambda _root, _split: source,
        decoder_version="test_decoder_v1",
    )

    def failing_projection_factory(source_skeleton, target, *_args):
        return _SyntheticExactProjection(source_skeleton, target, failure="synthetic frame construction failure")

    cfg = MotionTaskTableCfg(
        source=source_cfg,
        contact_channels=_CONTACT_CHANNELS,
        source_artifact_root="unused",
        motion_split="train",
        target_kinematics=_synthetic_target_kinematics(
            _SyntheticFrameTarget(),
            failing_projection_factory,
        ),
        task_row_mode="clip_time_ranges",
    )
    with pytest.raises(RuntimeError, match="synthetic frame construction failure"):
        _inspect_synthetic_task_table(cfg, "cpu", sequence_limit=len(index.clips))
    assert source.closed


def test_motion_bind_maps_reset_fields_to_simulator_joint_order_and_env_origins() -> None:
    """The robot codec and payload must produce exact simulator-ready reset rows."""
    payload, table, robot = _payload(2)
    assert not hasattr(payload, "reset_state")
    _bind(payload, torch.tensor((4, 1)))
    expected_position = torch.tensor(((101.0, 102.0, 103.0), (11.0, 2.0, 3.0)))
    torch.testing.assert_close(robot.root_state[:, :3], expected_position)
    torch.testing.assert_close(robot.root_state[:, 3:7], payload._resolver.reference["root_rotation"][:2])
    torch.testing.assert_close(robot.joint_position, torch.tensor(((121.0, 111.0), (21.0, 11.0))))
    torch.testing.assert_close(robot.joint_velocity, torch.tensor(((161.0, 151.0), (61.0, 51.0))))


def test_reset_transform_receives_sampled_source_and_table_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """The profile transform must receive sampled modes and the table-owned generator."""
    sampled_sources = torch.tensor((1, 0))
    sample_generators: list[torch.Generator] = []

    def sample(
        probabilities: torch.Tensor,
        count: int,
        replacement: bool,
        *,
        generator: torch.Generator,
        out: torch.Tensor,
    ) -> torch.Tensor:
        torch.testing.assert_close(probabilities, torch.tensor((0.8, 0.2)))
        assert count == 2
        assert replacement
        sample_generators.append(generator)
        return out.copy_(sampled_sources)

    monkeypatch.setattr(torch, "multinomial", sample)
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Generator]] = []

    def transform(
        decoded: MotionResetState,
        reset_source_indices: torch.Tensor,
        generator: torch.Generator,
    ) -> MotionResetState:
        calls.append(
            (
                decoded.root_position.clone(),
                reset_source_indices.clone(),
                generator,
            )
        )
        return MotionResetState(
            root_position=decoded.root_position + 10.0 * reset_source_indices[:, None],
            root_rotation_xyzw=decoded.root_rotation_xyzw,
            root_linear_velocity_world=decoded.root_linear_velocity_world,
            root_angular_velocity_world=decoded.root_angular_velocity_world,
            joint_position=decoded.joint_position,
            joint_velocity=decoded.joint_velocity,
        )

    setattr(transform, "reset_source_names", ("reference", "fall"))

    payload, _table, robot = _payload(
        2,
        states_relative=False,
        reset_sources=(("reference", 0.8), ("fall", 0.2)),
        reset_transform_factory=lambda **_kwargs: transform,
    )
    _bind(payload, torch.tensor((0, 4)))

    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], torch.tensor(((0.0, 1.0, 2.0), (101.0, 102.0, 103.0))))
    torch.testing.assert_close(calls[0][1], sampled_sources)
    assert calls[0][2] is payload.sampler.generator
    assert sample_generators == [payload.sampler.generator]
    expected_position = torch.tensor(((10.0, 11.0, 12.0), (101.0, 102.0, 103.0)))
    torch.testing.assert_close(robot.root_state[:, :3], expected_position)


def test_payload_rejects_reset_transform_source_name_mismatch() -> None:
    """Reset-source indices are meaningful only under one exact ordered name contract."""
    with pytest.raises(ValueError, match="source names differ"):
        _payload(
            1,
            reset_sources=(("reference", 0.8), ("fall", 0.2)),
            reset_transform_factory=lambda **_kwargs: SimpleNamespace(reset_source_names=("reference", "lie_down")),
        )


def test_task_row_modes_keep_reset_source_distributions_compact() -> None:
    """80/20 and 70/30 reset modes must not duplicate descriptor rows."""
    source_table = _table(task_row_mode="source_frames")
    source = _sampler(
        source_table,
        reset_sources=(("motion", 0.8), ("fall", 0.2)),
    )
    ranges_table = _table(task_row_mode="clip_time_ranges")
    ranges = _sampler(
        ranges_table,
        reset_sources=(("reference", 0.7), ("lie_down", 0.3)),
    )

    assert source_table.clip_indices.shape == (7,)
    assert ranges_table.clip_indices.shape == (2,)
    assert source.reset_source_names == ("motion", "fall")
    assert ranges.reset_source_names == ("reference", "lie_down")
    torch.testing.assert_close(source.reset_source_probabilities, torch.tensor((0.8, 0.2)))
    torch.testing.assert_close(ranges.reset_source_probabilities, torch.tensor((0.7, 0.3)))


def test_unequal_clip_lengths_keep_clip_sampling_uniform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clip selection and conditional frame selection must be separate exact draws."""
    table = _table(task_row_mode="source_frames")
    sampler = _sampler(table)

    assert table.num_tasks == 7
    assert len(table.clip_ids) == 2
    torch.testing.assert_close(table.clip_indices, torch.tensor((0, 0, 0, 1, 1, 1, 1)))
    torch.testing.assert_close(sampler._sampling_row_starts, torch.tensor((0, 3)))
    torch.testing.assert_close(sampler._sampling_row_counts, torch.tensor((3, 4)))
    clip_draws = torch.tensor((0, 1, 0, 1), dtype=torch.int64)
    frame_draws = torch.tensor((0.0, 0.0, 0.999, 0.74))

    def sample_clips(weights, count, replacement, *, generator, out):
        torch.testing.assert_close(weights, torch.ones(2))
        assert weights.data_ptr() == sampler.clip_priorities.data_ptr()
        assert count == clip_draws.shape[0]
        assert replacement
        assert generator is sampler.generator
        return out.copy_(clip_draws)

    def sample_frames(shape, *, device, generator, out):
        assert shape == frame_draws.shape
        assert device == table.device
        assert generator is sampler.generator
        return out.copy_(frame_draws)

    monkeypatch.setattr(torch, "multinomial", sample_clips)
    monkeypatch.setattr(torch, "rand", sample_frames)

    first = sampler.sample_rows(clip_draws.shape[0])
    pointer = first.data_ptr()
    torch.testing.assert_close(first, torch.tensor((0, 3, 2, 5)))
    second = sampler.sample_rows(clip_draws.shape[0])
    torch.testing.assert_close(second, torch.tensor((0, 3, 2, 5)))
    assert second.data_ptr() == pointer


def test_one_row_per_clip_sampling_is_exactly_the_clip_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip-range table must not consume a redundant within-clip random draw."""
    table = _table(task_row_mode="clip_time_ranges")
    sampler = _sampler(table)
    expected = torch.tensor((1, 0, 1, 1), dtype=torch.int64)
    torch.testing.assert_close(sampler._sampling_row_starts, torch.tensor((0, 1)))
    torch.testing.assert_close(sampler._sampling_row_counts, torch.ones(2, dtype=torch.int64))

    def sample(
        weights: torch.Tensor,
        count: int,
        replacement: bool,
        *,
        generator: torch.Generator,
        out: torch.Tensor,
    ) -> torch.Tensor:
        torch.testing.assert_close(weights, torch.ones(2))
        assert weights.data_ptr() == sampler.clip_priorities.data_ptr()
        assert count == expected.shape[0]
        assert replacement
        assert generator is sampler.generator
        return out.copy_(expected)

    def reject_frame_draw(*_args: object, **_kwargs: object) -> torch.Tensor:
        pytest.fail("one-row clip sampling must not draw a local row")

    monkeypatch.setattr(torch, "multinomial", sample)
    monkeypatch.setattr(torch, "rand", reject_frame_draw)

    torch.testing.assert_close(sampler.sample_rows(expected.shape[0]), expected)


def test_motion_sampler_priorities_are_total_clip_mass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Priority updates must change clip mass without multiplying it by row count."""
    table = _table(task_row_mode="source_frames")
    sampler = _sampler(table)
    priorities = torch.tensor((2.0, 8.0))
    sampler.clip_priorities.copy_(priorities)
    clip_draws = torch.tensor((1, 0, 1), dtype=torch.int64)
    frame_draws = torch.tensor((0.99, 0.5, 0.0))

    row_probabilities = torch.cat((torch.full((3,), 0.2 / 3.0), torch.full((4,), 0.8 / 4.0)))
    torch.testing.assert_close(row_probabilities[:3].sum(), torch.tensor(0.2))
    torch.testing.assert_close(row_probabilities[3:].sum(), torch.tensor(0.8))

    def sample(
        weights: torch.Tensor,
        count: int,
        replacement: bool,
        *,
        generator: torch.Generator,
        out: torch.Tensor,
    ) -> torch.Tensor:
        torch.testing.assert_close(weights, priorities)
        assert weights.data_ptr() == sampler.clip_priorities.data_ptr()
        assert count == clip_draws.shape[0]
        assert replacement
        assert generator is sampler.generator
        return out.copy_(clip_draws)

    def sample_frames(
        shape: torch.Size,
        *,
        device: torch.device,
        generator: torch.Generator,
        out: torch.Tensor,
    ) -> torch.Tensor:
        assert shape == frame_draws.shape
        assert device == table.device
        assert generator is sampler.generator
        return out.copy_(frame_draws)

    monkeypatch.setattr(torch, "multinomial", sample)
    monkeypatch.setattr(torch, "rand", sample_frames)

    sampled_rows = sampler.sample_rows(clip_draws.shape[0])

    torch.testing.assert_close(sampled_rows, torch.tensor((6, 1, 3)))
    torch.testing.assert_close(table.clip_indices[sampled_rows], clip_draws)


def test_motion_sampler_priority_reset_restores_equal_clip_mass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset must restore one unit of mass per clip regardless of row count."""
    table = _table(task_row_mode="source_frames")
    sampler = _sampler(table)
    sampler.clip_priorities.copy_(torch.tensor((2.0, 8.0)))
    sampler.clip_priorities.fill_(1.0)
    expected = torch.tensor((0, 1), dtype=torch.int64)

    def sample(
        weights: torch.Tensor,
        count: int,
        replacement: bool,
        *,
        generator: torch.Generator,
        out: torch.Tensor,
    ) -> torch.Tensor:
        torch.testing.assert_close(weights, torch.ones(2))
        assert count == expected.shape[0]
        assert replacement
        assert generator is sampler.generator
        return out.copy_(expected)

    def sample_frames(
        shape: torch.Size,
        *,
        device: torch.device,
        generator: torch.Generator,
        out: torch.Tensor,
    ) -> torch.Tensor:
        assert shape == expected.shape
        assert device == table.device
        assert generator is sampler.generator
        return out.zero_()

    monkeypatch.setattr(torch, "multinomial", sample)
    monkeypatch.setattr(torch, "rand", sample_frames)

    torch.testing.assert_close(sampler.clip_priorities, torch.ones(2))
    torch.testing.assert_close(sampler.sample_rows(expected.shape[0]), torch.tensor((0, 3)))


@pytest.mark.parametrize("probabilities", ((0.8, 0.2), (0.7, 0.3)))
def test_reset_source_sampling_uses_declared_distribution_exactly(
    probabilities: tuple[float, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling must forward each compact distribution without descriptor coupling."""
    declared = torch.tensor(probabilities)
    table = _table(task_row_mode="clip_time_ranges")
    sampler = _sampler(
        table,
        reset_sources=(("reference", probabilities[0]), ("alternative", probabilities[1])),
    )
    expected = torch.tensor((1, 0, 1, 1), dtype=torch.int64)

    def sample(
        weights: torch.Tensor,
        count: int,
        replacement: bool,
        *,
        generator: torch.Generator,
        out: torch.Tensor,
    ) -> torch.Tensor:
        torch.testing.assert_close(weights, declared)
        assert weights.data_ptr() == sampler.reset_source_probabilities.data_ptr()
        assert count == expected.shape[0]
        assert replacement
        assert generator is sampler.generator
        return out.copy_(expected)

    monkeypatch.setattr(torch, "multinomial", sample)
    output = torch.empty_like(expected)
    sampler.sample_reset_sources(output)
    torch.testing.assert_close(output, expected)


def test_reset_time_sampling_switches_between_uniform_draws_and_exact_range_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sampler owns both ordinary uniform reset times and exact evaluation starts."""
    table = _table(task_row_mode="clip_time_ranges")
    sampler = _sampler(table)
    ranges = torch.tensor(((1.0, 3.0), (5.0, 9.0)))
    fractions = torch.tensor((0.25, 0.75))

    def sample_uniform(
        shape: torch.Size,
        *,
        device: torch.device,
        generator: torch.Generator,
        out: torch.Tensor,
    ) -> torch.Tensor:
        assert shape == torch.Size((2,))
        assert device == table.device
        assert generator is sampler.generator
        return out.copy_(fractions)

    monkeypatch.setattr(torch, "rand", sample_uniform)
    output = torch.empty(2)
    sampler.sample_reset_times(ranges, output)
    torch.testing.assert_close(output, torch.tensor((1.5, 8.0)))

    sampler.set_reset_time_mode("range_start")

    def reject_uniform_draw(*_args: object, **_kwargs: object) -> torch.Tensor:
        pytest.fail("exact range-start sampling must not consume a uniform draw")

    monkeypatch.setattr(torch, "rand", reject_uniform_draw)
    sampler.sample_reset_times(ranges, output)
    torch.testing.assert_close(output, torch.tensor((1.0, 5.0)))


def test_payload_exposes_only_reset_materialization_state() -> None:
    """Motion payload state is limited to reset pointers, interpolation scratch, and robot binding."""
    payload, _table, _robot = _payload(2)

    assert payload.command_dim == 0
    for stale_name in (
        "motion_facts",
        "reference",
        "evaluation_scope",
        "get_state",
        "command_std",
        "get_task_done",
        "get_task_reward",
        "_transition",
        "environment_reward",
        "auxiliary_evidence",
        "raw_evidence",
        "history_value",
    ):
        assert not hasattr(payload, stale_name)
    assert not hasattr(commands_cfg_module.MotionStatePayloadCfg, "transition_factory")


def test_reset_slerp_matches_core_shortest_arc_without_replacing_storage() -> None:
    """The fixed-output reset slerp exactly follows the shared quaternion law."""
    first = quat_from_rotation_vector(torch.zeros(2, 3))
    second = quat_from_rotation_vector(torch.tensor(((0.0, 0.0, 1.0), (1.0e-5, 0.0, 0.0))))
    second[0].neg_()
    source = torch.stack((first[0], second[0], first[1], second[1]))
    table = SimpleNamespace(
        device=torch.device("cpu"),
        interpolation=lambda _name: "slerp",
        field=lambda _name: source,
    )
    field = MotionStatePayload._ReferenceResolver._Field(table, "root_rotation", capacity=2)
    frame0 = torch.tensor((0, 2), dtype=torch.int64)
    frame1 = torch.tensor((1, 3), dtype=torch.int64)
    alpha = torch.tensor((0.35, 0.65))
    pointer = field.output.data_ptr()

    field.resolve(frame0, frame1, alpha)

    torch.testing.assert_close(field.output, quat_slerp(source[frame0], source[frame1], alpha))
    field.resolve(frame0, frame1, 1.0 - alpha)
    torch.testing.assert_close(field.output, quat_slerp(source[frame0], source[frame1], 1.0 - alpha))
    assert field.output.data_ptr() == pointer


def test_reset_interpolation_matches_allocating_table_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixed-output reset interpolation follows the immutable table's reference law."""
    payload, table, robot = _payload(1, states_relative=False)

    def sample_midpoint(_sampler: MotionSampler, ranges: torch.Tensor, output: torch.Tensor) -> None:
        assert ranges.shape == (1, 2)
        output.fill_(0.25)

    monkeypatch.setattr(MotionSampler, "sample_reset_times", sample_midpoint)
    _bind(payload, torch.tensor((0,)))
    oracle = table.reference_view(torch.zeros(1, dtype=torch.int64), torch.full((1,), 0.25))

    for name, value in payload._resolver.reference.items():
        torch.testing.assert_close(value[:1], oracle.field(name))
    torch.testing.assert_close(robot.joint_position, oracle.field("joint_position"))
    torch.testing.assert_close(robot.root_state[:, :3], oracle.field("root_position"))


def test_update_does_not_advance_or_rematerialize_reset_state() -> None:
    """Completed policy steps do not create a second motion clock inside the reset payload."""
    payload, _table, robot = _payload(2)
    _bind(payload, torch.tensor((0, 1)))
    reset = {name: value.clone() for name, value in payload._resolver.reference.items()}
    root = robot.root_state.clone()
    joints = robot.joint_position.clone()

    payload.update(0.25, torch.empty(2, 0), torch.empty(2, 0))

    for name, value in reset.items():
        torch.testing.assert_close(payload._resolver.reference[name], value)
    torch.testing.assert_close(robot.root_state, root)
    torch.testing.assert_close(robot.joint_position, joints)


@pytest.mark.parametrize("num_envs", (1, 16, 1024))
def test_motion_reset_state_is_clone_invariant(num_envs: int) -> None:
    """One descriptor follows the same fixed reset law at all required vector scales."""
    payload, _table, robot = _payload(num_envs, states_relative=False)
    task_rows = torch.zeros(num_envs, dtype=torch.int64)
    _bind(payload, task_rows)
    pointers = {name: value.data_ptr() for name, value in payload._resolver.reference.items()}

    _bind(payload, task_rows)

    for value in payload._resolver.reference.values():
        torch.testing.assert_close(value, value[:1].expand_as(value))
    torch.testing.assert_close(robot.root_state, robot.root_state[:1].expand_as(robot.root_state))
    torch.testing.assert_close(robot.joint_position, robot.joint_position[:1].expand_as(robot.joint_position))
    assert {name: value.data_ptr() for name, value in payload._resolver.reference.items()} == pointers


def test_partial_reset_only_resolves_and_writes_selected_rows() -> None:
    """A partial reset reuses fixed storage and leaves every unselected environment unchanged."""
    payload, _table, robot = _payload(4)
    _bind(payload, torch.zeros(4, dtype=torch.int64))
    root_before = robot.root_state.clone()
    joint_before = robot.joint_position.clone()
    reference_tail = {name: value[2:].clone() for name, value in payload._resolver.reference.items()}
    pointers = {
        "root_state": payload._root_state.data_ptr(),
        **{name: value.data_ptr() for name, value in payload._resolver.reference.items()},
    }

    payload.bind(torch.tensor((1, 3)), torch.tensor((4, 4)))

    torch.testing.assert_close(robot.root_state[(0, 2), :], root_before[(0, 2), :])
    torch.testing.assert_close(robot.joint_position[(0, 2), :], joint_before[(0, 2), :])
    torch.testing.assert_close(
        robot.root_state[(1, 3), :3], torch.tensor(((111.0, 102.0, 103.0), (131.0, 102.0, 103.0)))
    )
    for name, value in reference_tail.items():
        torch.testing.assert_close(payload._resolver.reference[name][2:], value)
    current_pointers = {
        "root_state": payload._root_state.data_ptr(),
        **{name: value.data_ptr() for name, value in payload._resolver.reference.items()},
    }
    assert current_pointers == pointers


def test_motion_reset_path_has_no_table_view_or_host_sync_calls() -> None:
    """Reset lookup retains caller-owned storage and avoids allocating table conveniences."""
    source = "\n".join(
        (
            inspect.getsource(MotionStatePayload.bind),
            inspect.getsource(MotionStatePayload.update),
            inspect.getsource(MotionStatePayload._ReferenceResolver.resolve),
            inspect.getsource(MotionStatePayload._ReferenceResolver._Field.resolve),
        )
    )
    for forbidden in (
        ".item(",
        ".tolist(",
        ".contiguous(",
        ".reference_view(",
        ".select(",
        "torch.unique(",
        "torch.empty(",
        "torch.zeros(",
        "torch.cat(",
    ):
        assert forbidden not in source
    assert tuple(inspect.signature(MotionStatePayload._ReferenceResolver.resolve).parameters) == ("self", "count")
    assert "raw_evidence" not in inspect.getsource(MotionStatePayload.bind)
    assert "auxiliary_evidence" not in inspect.getsource(MotionStatePayload)

    full_source = inspect.getsource(MotionStatePayload).lower()
    for forbidden in ("metamotivo", "bfm", "discriminator", "learner_value"):
        assert forbidden not in full_source


def test_motion_task_table_owns_frames_without_bank_indirection() -> None:
    """Frame storage and interpretation must belong to the selected task table."""
    assert not hasattr(motion_data_module, "RobotMotionBank")
    assert not hasattr(motion_data_module, "RobotMotionLayout")
    assert not hasattr(MotionTaskTable, "FrameLayout")
    assert "bank_provider" not in inspect.getsource(commands_cfg_module.MotionTaskTableCfg)
    assert "RobotMotionBank" not in inspect.getsource(table_module)
    assert not hasattr(MotionClipIndex.Clip, "valid")
    for name in ("Writer", "writer", "build", "from_storage", "clip_valid", "source_clip_ids", "splits"):
        assert not hasattr(MotionTaskTable, name)
    assert all(not hasattr(owner, "memory_bytes") for owner in (MotionFrames, MotionTaskTable, MotionSampler))


def test_motion_task_table_excludes_mutable_sampling_ownership() -> None:
    table_source = inspect.getsource(MotionTaskTable)
    table_module_source = inspect.getsource(table_module)
    for forbidden in (
        "class MotionSampler:",
        "_reset_source_data",
        "clip_priorities",
        "generator",
        "reset_source_probabilities",
        "reset_source_names",
        "def sample_rows(",
        "def sample_reset_sources(",
        "def sample_reset_times(",
        "def set_clip_priorities(",
        "def reset_clip_priorities(",
        "def validate_clip_priorities(",
        "def task_sampling_law(",
    ):
        assert forbidden not in table_source
        assert forbidden not in table_module_source

    sampler_path = Path(table_module.__file__).with_name("motion_sampler.py")
    assert sampler_path.is_file()
    sampler_source = sampler_path.read_text(encoding="utf-8")
    assert "class MotionSampler:" in sampler_source
    assert "clip_priorities" in sampler_source
    assert "motion_task_sampling_law(" not in inspect.getsource(MotionSampler)

    table_cfg_source = inspect.getsource(commands_cfg_module.MotionTaskTableCfg)
    payload_cfg_source = inspect.getsource(commands_cfg_module.MotionStatePayloadCfg)
    assert "reset_sources" not in table_cfg_source
    assert "reset_sources" in payload_cfg_source

    payload_source = inspect.getsource(MotionStatePayload.__init__)
    assert "self.sampler = MotionSampler(" in payload_source
    assert "payload_cfg.reset_sources" in payload_source
    assert "table_cfg.reset_sources" not in payload_source


def test_motion_table_builds_interleaved_source_skeleton_groups_before_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every skeleton decodes separately inside one route execution and returns to source order."""
    skeletons = (
        _skeleton(),
        replace(_skeleton(), identifier="motion_command_test_tall", content_sha256=_hash("skeleton-tall")),
    )
    index = MotionClipIndex(
        source_content_sha256=_hash("interleaved-source"),
        skeleton_identity_sha256s=tuple(skeleton.identity_sha256 for skeleton in skeletons),
        clips=tuple(
            MotionClipIndex.Clip(
                clip_id=f"clip_{clip_index}",
                frame_count=frame_count,
                source_fps=10.0,
                content_sha256=_hash(f"interleaved-clip-{clip_index}"),
                skeleton_id=skeleton_id,
            )
            for clip_index, (frame_count, skeleton_id) in enumerate(((2, 0), (3, 1), (2, 0), (3, 1)))
        ),
    )

    class Source:
        def __init__(self) -> None:
            self.skeleton_calls: list[int] = []
            self.clip_calls: list[tuple[int, ...]] = []

        def inspect(self) -> MotionClipIndex:
            return index

        def skeleton(self, skeleton_id: int) -> MotionSkeleton:
            self.skeleton_calls.append(skeleton_id)
            return skeletons[skeleton_id]

        def clips(self, clip_indices: tuple[int, ...]):
            assert self.skeleton_calls == [0, 1]
            self.clip_calls.append(clip_indices)
            for clip_index in clip_indices:
                clip = index.clips[clip_index]
                yield clip_index, _SyntheticMotionClip(clip_index, clip.frame_count, clip.source_fps)

        def close(self) -> None:
            pass

    source = Source()
    coordinate_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    coordinate_live_scalars: list[int] = []
    target = _SyntheticFrameTarget(coordinate_refs=coordinate_refs, coordinate_live_scalars=coordinate_live_scalars)
    projections = []
    contact_offsets_seen = []

    def source_projection(
        source_skeleton: MotionSkeleton,
        _target: object,
        _source: object,
        _contact_channels: object,
        contact_offsets: torch.Tensor,
    ):
        contact_offsets_seen.append(contact_offsets)
        projection = _SyntheticExactProjection(source_skeleton, target)
        projections.append(projection)
        return projection

    split = SimpleNamespace(
        source_content_sha256=index.source_content_sha256,
        clip_count=len(index.clips),
        frame_count=index.total_frames,
    )
    source_cfg = SimpleNamespace(
        purpose="oracle",
        train=split,
        evaluation=split,
        open_split=lambda _root, _split: source,
        decoder_version="interleaved_decoder_v1",
    )
    cfg = MotionTaskTableCfg(
        source=source_cfg,
        contact_channels=_CONTACT_CHANNELS,
        source_artifact_root="unused",
        motion_split="train",
        target_kinematics=_synthetic_target_kinematics(target, source_projection),
        task_row_mode="clip_time_ranges",
    )

    family_executions = 0
    execute_task_family = table_builder_module.execute_task_family

    def count_family_execution(*args, **kwargs):
        nonlocal family_executions
        family_executions += 1
        return execute_task_family(*args, **kwargs)

    monkeypatch.setattr(table_builder_module, "execute_task_family", count_family_execution)
    inspection = _inspect_synthetic_task_table(cfg, "cpu", sequence_limit=len(index.clips))

    assert family_executions == 1
    assert not hasattr(table_module, "_combine_motion_groups")
    assert "coordinates" in table_builder_module._MotionBuiltRoute.__dataclass_fields__
    assert "frames" not in table_builder_module._MotionBuiltRoute.__dataclass_fields__
    assert len(projections) == 2
    assert contact_offsets_seen[0] is contact_offsets_seen[1]
    assert target.materialize_calls == 1
    assert target.live_coordinate_scalars_at_materialization >= index.total_frames
    assert source.clip_calls == [(0, 2), (1, 3)]
    torch.testing.assert_close(inspection.sequences.offsets, torch.tensor(index.offsets))
    torch.testing.assert_close(
        inspection.state_bank.root_pose[:, 0, 0],
        torch.tensor((0.0, 1.0, 100.0, 101.0, 102.0, 200.0, 201.0, 300.0, 301.0, 302.0)),
    )
    accepted = inspection.quality.values[:, inspection.quality.names.index("accepted")]
    torch.testing.assert_close(accepted, torch.ones(4))


def test_motion_inspection_validates_full_skeleton_identity_before_source_prefix() -> None:
    """A skeleton beyond the inspection prefix is verified before any source clip is decoded."""
    skeletons = (
        _skeleton(),
        replace(_skeleton(), identifier="motion_command_test_tall", content_sha256=_hash("skeleton-tall")),
    )
    index = MotionClipIndex(
        source_content_sha256=_hash("changed-skeleton-source"),
        skeleton_identity_sha256s=(skeletons[0].identity_sha256, _hash("wrong-skeleton-identity")),
        clips=(
            MotionClipIndex.Clip("clip_0", 2, 10.0, _hash("changed-skeleton-clip-0"), 0),
            MotionClipIndex.Clip("clip_1", 2, 10.0, _hash("changed-skeleton-clip-1"), 1),
        ),
    )

    class Source:
        decoded = False

        def inspect(self) -> MotionClipIndex:
            return index

        def skeleton(self, skeleton_id: int) -> MotionSkeleton:
            return skeletons[skeleton_id]

        def clips(self, clip_indices: tuple[int, ...]):
            self.decoded = True
            raise AssertionError(f"Unexpected decode: {clip_indices}")

        def close(self) -> None:
            pass

    source = Source()
    split = SimpleNamespace(
        source_content_sha256=index.source_content_sha256,
        clip_count=len(index.clips),
        frame_count=index.total_frames,
    )

    def exact_projection_factory(source_skeleton, target, *_args):
        return _SyntheticExactProjection(source_skeleton, target)

    cfg = MotionTaskTableCfg(
        source=SimpleNamespace(
            purpose="production",
            train=split,
            evaluation=split,
            open_split=lambda _root, _split: source,
            decoder_version="changed_skeleton_decoder_v1",
        ),
        contact_channels=_CONTACT_CHANNELS,
        source_artifact_root="unused",
        motion_split="train",
        target_kinematics=_synthetic_target_kinematics(
            _SyntheticFrameTarget(),
            exact_projection_factory,
        ),
        task_row_mode="clip_time_ranges",
    )

    with pytest.raises(ValueError, match="identity changed after inspection"):
        _inspect_synthetic_task_table(cfg, "cpu", sequence_limit=1)
    assert not source.decoded
