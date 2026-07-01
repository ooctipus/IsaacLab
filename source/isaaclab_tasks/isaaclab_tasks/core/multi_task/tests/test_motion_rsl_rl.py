# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the motion-table RSL-RL expert boundary."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.models.forward_backward_model import ForwardBackwardObservationSchema

from isaaclab_tasks.core.multi_task.motion.config.robots.g1 import (
    _SIMULATOR_JOINT_NAMES as _G1_LIVE_JOINT_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.config.robots.g1 import (
    G1_BEHAVIOR_BODY_NAMES as _G1_BEHAVIOR_BODY_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.config.robots.g1 import (
    G1_BEHAVIOR_JOINT_NAMES as _G1_BEHAVIOR_JOINT_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex, MotionSampleGrid
from isaaclab_tasks.core.multi_task.motion.frames import G1_HEAD_FRAME_NAME
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import MotionTaskTable
from isaaclab_tasks.core.multi_task.motion.mdp.observations import (
    g1_privileged_observation,
    g1_released_expert_state,
)
from isaaclab_tasks.core.multi_task.motion.rsl_rl import (
    motion_expert_buffer_g1,
    motion_expert_buffer_smpl_cmu,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


_G1_LIVE_BODY_NAMES = (
    _G1_BEHAVIOR_BODY_NAMES[0],
    *_G1_BEHAVIOR_BODY_NAMES[1::2],
    *_G1_BEHAVIOR_BODY_NAMES[2::2],
)
_G1_REFERENCE_FRAME_NAMES = (*_G1_LIVE_BODY_NAMES, G1_HEAD_FRAME_NAME)
_SMPL_JOINT_NAMES = tuple(f"joint_{index}" for index in range(69))


def _index(frame_counts: tuple[int, ...], semantic_level: str) -> MotionClipIndex:
    clips = tuple(
        MotionClipIndex.Clip(
            clip_id=f"clip_{index}",
            source_path=f"fixture/{index}",
            frame_count=count,
            source_fps=30.0,
            split="train",
            tags=(),
            content_sha256=_hash(f"clip-{index}"),
        )
        for index, count in enumerate(frame_counts)
    )
    return MotionClipIndex(
        source_content_sha256=_hash("source"),
        skeleton_sha256=_hash("source-skeleton"),
        semantic_level=semantic_level,
        license="fixture",
        clips=clips,
    )


def _table(
    index: MotionClipIndex,
    frames: MotionTaskTable.Frames,
    joint_names: tuple[str, ...],
    reference_frame_names: tuple[str, ...],
    *,
    frame_builder_version: str,
    expert_sample_grid: MotionSampleGrid,
) -> MotionTaskTable:
    return MotionTaskTable.from_storage(
        index,
        frames,
        joint_names,
        reference_frame_names,
        frame_builder_version,
        _hash(f"{frame_builder_version}-construction"),
        "clip_time_ranges",
        (("reference", 1.0),),
        expert_sample_grid,
        seed=7,
    )


def _smpl_table(frame_counts: tuple[int, ...]) -> MotionTaskTable:
    index = _index(frame_counts, "smpl_robot_state_and_observation")
    count = index.total_frames
    observation = torch.arange(count * 358, dtype=torch.float32).view(count, 358)
    frames = MotionTaskTable.Frames(
        root_position=torch.zeros(count, 3),
        root_rotation=torch.nn.functional.pad(torch.ones(count, 1), (3, 0)),
        root_linear_velocity=torch.zeros(count, 3),
        root_angular_velocity=torch.zeros(count, 3),
        joint_position=torch.zeros(count, 69),
        joint_velocity=torch.zeros(count, 69),
        observation=observation,
    )
    return _table(
        index,
        frames,
        _SMPL_JOINT_NAMES,
        (),
        frame_builder_version="smpl_humenv_direct_v1",
        expert_sample_grid=MotionSampleGrid.source_rows(),
    )


def _g1_table(
    frame_counts: tuple[int, ...],
    *,
    semantic_level: str = "robot_pose_g1_not_canonical_lafan",
    frame_builder_version: str = "g1_motionlib_parity_v1",
) -> MotionTaskTable:
    index = _index(frame_counts, semantic_level)
    count = index.total_frames
    time = torch.arange(count, dtype=torch.float32).view(count, 1)
    body_axis = torch.arange(31, dtype=torch.float32).view(1, 31, 1)
    coordinate_axis = torch.arange(3, dtype=torch.float32).view(1, 1, 3)
    body_position = time[:, None] * 0.01 + body_axis * 0.1 + coordinate_axis * 0.001
    body_rotation = torch.zeros(count, 31, 4)
    body_rotation[..., 3] = 1.0
    body_linear_velocity = time[:, None] * 0.02 + body_axis * 0.2 + coordinate_axis * 0.002
    body_angular_velocity = time[:, None] * 0.03 + body_axis * 0.3 + coordinate_axis * 0.003
    joint_axis = torch.arange(29, dtype=torch.float32).view(1, 29)
    joint_position = time * 0.04 + joint_axis * 0.004
    joint_velocity = time * 0.05 + joint_axis * 0.005
    return _table(
        index,
        MotionTaskTable.Frames(
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            body_position=body_position,
            body_rotation=body_rotation,
            body_linear_velocity=body_linear_velocity,
            body_angular_velocity=body_angular_velocity,
        ),
        _G1_LIVE_JOINT_NAMES,
        _G1_REFERENCE_FRAME_NAMES,
        frame_builder_version=frame_builder_version,
        expert_sample_grid=MotionSampleGrid.uniform_before_source_end(step_seconds=0.02),
    )


def _env(
    table: MotionTaskTable,
    default_joint_position: torch.Tensor | None = None,
    default_joint_offset: torch.Tensor | None = None,
):
    payload = SimpleNamespace(table=table)
    command = SimpleNamespace(payload=payload)
    command_manager = SimpleNamespace(get_term=lambda name: command if name == "motion" else None)
    physical_body_names = table.reference_frame_names[:-1] if table.reference_frame_names else ()
    robot = SimpleNamespace(joint_names=table.joint_names, body_names=physical_body_names)
    scene = {"robot": robot}

    action = SimpleNamespace(
        joint_names=_G1_BEHAVIOR_JOINT_NAMES,
        joint_default_position=(torch.zeros(29) if default_joint_position is None else default_joint_position),
        default_joint_offset=(torch.zeros(2, 29) if default_joint_offset is None else default_joint_offset),
    )
    action_manager = SimpleNamespace(get_term=lambda name: action if name == "joint_position" else None)
    return SimpleNamespace(
        unwrapped=SimpleNamespace(command_manager=command_manager, action_manager=action_manager, scene=scene),
    )


def test_smpl_provider_reuses_exact_native_observation_rows() -> None:
    table = _smpl_table((5, 4))
    schema = ForwardBackwardObservationSchema.from_config(
        {"policy": 358},
        {"backward": ("policy",)},
    )

    expert = motion_expert_buffer_smpl_cmu(
        _env(table),
        schema,
        "cpu",
        command_name="motion",
        window_lengths=(2,),
        seed=7,
    )

    assert expert.frames.shape == (9, 358)
    assert expert.frames.data_ptr() == table.frames.observation.data_ptr()
    assert expert.clip_offsets.tolist() == [0, 5, 9]
    assert expert.clip_ids == table.clip_ids == ("clip_0", "clip_1")
    assert expert.schema.dataset_id == "smpl_robot_state_and_observation:smpl_humenv_direct_v1"


def test_g1_provider_projects_exact_state_and_privileged_facts_at_50_hz() -> None:
    table = _g1_table((10, 7))
    schema = ForwardBackwardObservationSchema.from_config(
        {"state": 64, "privileged_state": 463},
        {"backward": ("state", "privileged_state")},
    )
    default_joint_position = torch.linspace(-0.2, 0.2, 29)

    expert = motion_expert_buffer_g1(
        _env(table, default_joint_position),
        schema,
        "cpu",
        command_name="motion",
        window_lengths=(2,),
        seed=11,
    )

    assert expert.frames.shape == (25, 527)
    assert expert.clip_offsets.tolist() == [0, 15, 25]
    assert expert.clip_ids == table.clip_ids == ("clip_0", "clip_1")
    reference = table.reference_view(torch.tensor([0]), torch.tensor([0.02]))
    joint_indices = torch.tensor([_G1_LIVE_JOINT_NAMES.index(name) for name in _G1_BEHAVIOR_JOINT_NAMES])
    behavior_frame_names = (*_G1_BEHAVIOR_BODY_NAMES, G1_HEAD_FRAME_NAME)
    body_indices = torch.tensor([_G1_REFERENCE_FRAME_NAMES.index(name) for name in behavior_frame_names])
    joint_position = reference.field("joint_position").index_select(1, joint_indices)
    joint_velocity = reference.field("joint_velocity").index_select(1, joint_indices)
    body_position = reference.field("body_position").index_select(1, body_indices)
    body_rotation = reference.field("body_rotation").index_select(1, body_indices)
    body_linear_velocity = reference.field("body_linear_velocity").index_select(1, body_indices)
    body_angular_velocity = reference.field("body_angular_velocity").index_select(1, body_indices)
    state = g1_released_expert_state(
        body_rotation[:, 0],
        body_angular_velocity[:, 0],
        joint_position,
        joint_velocity,
        default_joint_position,
    )
    privileged = g1_privileged_observation(
        body_position,
        body_rotation,
        body_linear_velocity,
        body_angular_velocity,
    )
    torch.testing.assert_close(expert.frames[1:2], torch.cat((state, privileged), dim=-1))


def test_g1_expert_identity_tracks_canonical_defaults_but_not_randomized_offsets() -> None:
    table = _g1_table((10,))
    schema = ForwardBackwardObservationSchema.from_config(
        {"state": 64, "privileged_state": 463},
        {"backward": ("state", "privileged_state")},
    )
    canonical = torch.linspace(-0.2, 0.2, 29)

    def build(defaults: torch.Tensor, offsets: torch.Tensor):
        return motion_expert_buffer_g1(
            _env(table, defaults, offsets),
            schema,
            "cpu",
            command_name="motion",
            window_lengths=(2,),
            seed=17,
        )

    first = build(canonical, torch.zeros(2, 29))
    different_episode_offsets = build(canonical, torch.full((2, 29), 0.02))
    changed_canonical = canonical.clone()
    changed_canonical[0] += 0.1
    different_defaults = build(changed_canonical, torch.zeros(2, 29))

    torch.testing.assert_close(first.frames, different_episode_offsets.frames, rtol=0.0, atol=0.0)
    assert first.schema.data_hash == different_episode_offsets.schema.data_hash
    assert not torch.equal(first.frames, different_defaults.frames)
    assert first.schema.data_hash != different_defaults.schema.data_hash


def test_g1_expert_rejects_per_environment_default_rows() -> None:
    table = _g1_table((10,))
    schema = ForwardBackwardObservationSchema.from_config(
        {"state": 64, "privileged_state": 463},
        {"backward": ("state", "privileged_state")},
    )

    with pytest.raises(ValueError, match="29 behavior-ordered joint defaults"):
        motion_expert_buffer_g1(
            _env(table, torch.zeros(2, 29)),
            schema,
            "cpu",
            command_name="motion",
            window_lengths=(2,),
        )


def test_g1_provider_accepts_smpl_source_after_target_frame_building() -> None:
    table = _g1_table(
        (8, 6),
        semantic_level="smpl_robot_state_and_observation",
        frame_builder_version="g1_smpl_humenv_ordered_hinge_fit_v1",
    )
    schema = ForwardBackwardObservationSchema.from_config(
        {"state": 64, "privileged_state": 463},
        {"backward": ("state", "privileged_state")},
    )

    expert = motion_expert_buffer_g1(
        _env(table),
        schema,
        "cpu",
        command_name="motion",
        window_lengths=(2,),
        seed=13,
    )

    assert expert.frames.shape == (21, 527)
    assert expert.clip_offsets.tolist() == [0, 12, 21]
    assert expert.schema.dataset_id == ("smpl_robot_state_and_observation:g1_smpl_humenv_ordered_hinge_fit_v1")


def test_g1_provider_rejects_semantically_reordered_backward_routes() -> None:
    table = _g1_table((10,))
    schema = ForwardBackwardObservationSchema.from_config(
        {"state": 64, "privileged_state": 463},
        {"backward": ("privileged_state", "state")},
    )

    with pytest.raises(ValueError, match="64\\+463 state/privileged"):
        motion_expert_buffer_g1(
            _env(table),
            schema,
            "cpu",
            command_name="motion",
            window_lengths=(2,),
        )
