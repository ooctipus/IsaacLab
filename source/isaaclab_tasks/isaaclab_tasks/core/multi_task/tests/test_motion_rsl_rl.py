# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the motion-table RSL-RL expert boundary."""

from __future__ import annotations

import ast
import hashlib
import inspect
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.models.forward_backward_model import ForwardBackwardObservationSchema
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer

import isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_expert as expert_bridge
from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex, MotionFrames
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import MotionTaskTable
from isaaclab_tasks.core.multi_task.motion.robots.g1.articulation import (
    G1_BEHAVIOR_BODY_NAMES as _G1_BEHAVIOR_BODY_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.robots.g1.articulation import (
    G1_BEHAVIOR_JOINT_NAMES as _G1_BEHAVIOR_JOINT_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.robots.g1.articulation import (
    G1_SIMULATOR_BODY_NAMES as _G1_LIVE_BODY_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.robots.g1.articulation import (
    G1_SIMULATOR_JOINT_NAMES as _G1_LIVE_JOINT_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.robots.g1.frames import G1_HEAD_FRAME_NAME
from isaaclab_tasks.core.multi_task.motion.robots.g1.observations import (
    g1_bfm_expert_observation_fields,
    g1_bfm_expert_target,
    g1_bfm_privileged_observation,
)
from isaaclab_tasks.core.multi_task.motion.robots.smpl.observations import smpl_expert_target, smpl_humenv_observation
from isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_expert import forward_backward_expert_buffer
from isaaclab_tasks.core.multi_task.tests.motion_table_test_utils import motion_task_table

_G1_FIELD_WIDTHS = {
    "joint_position": 29,
    "joint_velocity": 29,
    "projected_gravity": 3,
    "base_angular_velocity": 3,
    "privileged_state": 463,
}
_G1_BACKWARD_ROUTE = tuple(_G1_FIELD_WIDTHS)
_G1_TARGET = "isaaclab_tasks.core.multi_task.motion.robots.g1.observations:g1_bfm_expert_target"
_SMPL_TARGET = "isaaclab_tasks.core.multi_task.motion.robots.smpl.observations:smpl_expert_target"
_SMPL_BINDS = ("env.unwrapped.scene['robot']",)
_G1_BINDS = _SMPL_BINDS + ("env.unwrapped.action_manager.get_term('joint_position')",)
_SOURCE_BIND = "env.unwrapped.command_manager.get_term('motion').table"
_PRIORITIES_BIND = "env.unwrapped.command_manager.get_term('motion').payload.sampler.clip_priorities"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


_G1_REFERENCE_FRAME_NAMES = (*_G1_LIVE_BODY_NAMES, G1_HEAD_FRAME_NAME)
_SMPL_JOINT_NAMES = tuple(f"joint_{index}" for index in range(69))
_SMPL_BODY_NAMES = tuple(f"body_{index}" for index in range(24))


def test_expert_bridge_is_robot_neutral_and_defines_no_adapters() -> None:
    tree = ast.parse(inspect.getsource(expert_bridge))
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

    assert classes.isdisjoint({"_MotionFieldSource", "_MotionPayload", "_IndexedMotionFields"})
    assert "_motion_payload" not in functions
    source = inspect.getsource(expert_bridge).lower()
    assert all(token not in source for token in ("g1", "lafan", "smpl", "cmu"))
    assert g1_bfm_expert_target.__module__.endswith("motion.robots.g1.observations")
    assert smpl_expert_target.__module__.endswith("motion.robots.smpl.observations")
    assert tuple(inspect.signature(g1_bfm_expert_target).parameters) == ("robot", "action", "table", "field")
    assert tuple(inspect.signature(smpl_expert_target).parameters) == ("robot", "table", "field")


def _index(frame_counts: tuple[int, ...]) -> MotionClipIndex:
    clips = tuple(
        MotionClipIndex.Clip(
            clip_id=f"clip_{index}",
            frame_count=count,
            source_fps=30.0,
            content_sha256=_hash(f"clip-{index}"),
        )
        for index, count in enumerate(frame_counts)
    )
    return MotionClipIndex(source_content_sha256=_hash("source"), clips=clips)


def _table(
    index: MotionClipIndex,
    frames: MotionFrames,
    joint_names: tuple[str, ...],
    reference_frame_names: tuple[str, ...],
    *,
    frame_builder_version: str,
) -> MotionTaskTable:
    return motion_task_table(
        index,
        frames,
        joint_names,
        reference_frame_names,
        frame_builder_version,
        _hash(f"{frame_builder_version}-construction"),
        "clip_time_ranges",
        _hash("source-skeleton"),
    )


def _smpl_table(frame_counts: tuple[int, ...]) -> MotionTaskTable:
    index = _index(frame_counts)
    count = index.total_frames
    time = torch.arange(count, dtype=torch.float32).view(count, 1, 1)
    body_axis = torch.arange(24, dtype=torch.float32).view(1, 24, 1)
    coordinate_axis = torch.arange(3, dtype=torch.float32).view(1, 1, 3)
    body_position = time * 0.01 + body_axis * 0.1 + coordinate_axis * 0.001
    body_rotation = torch.zeros(count, 24, 4)
    body_rotation[..., 3] = 1.0
    body_linear_velocity = time * 0.02 + body_axis * 0.2 + coordinate_axis * 0.002
    body_angular_velocity = time * 0.03 + body_axis * 0.3 + coordinate_axis * 0.003
    frames = MotionFrames(
        joint_position=torch.zeros(count, 69),
        joint_velocity=torch.zeros(count, 69),
        body_position=body_position,
        body_rotation=body_rotation,
        body_linear_velocity=body_linear_velocity,
        body_angular_velocity=body_angular_velocity,
    )
    return _table(
        index,
        frames,
        _SMPL_JOINT_NAMES,
        _SMPL_BODY_NAMES,
        frame_builder_version="smpl_generalized_coordinate_exact_mjcf_v1",
    )


def _smpl_observation(table: MotionTaskTable) -> torch.Tensor:
    return smpl_humenv_observation(
        table.field("body_position"),
        table.field("body_rotation"),
        table.field("body_linear_velocity"),
        table.field("body_angular_velocity"),
    )


def _g1_table(
    frame_counts: tuple[int, ...],
    *,
    frame_builder_version: str = "g1_motionlib_parity_v1",
) -> MotionTaskTable:
    index = _index(frame_counts)
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
        MotionFrames(
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
    )


def _env(
    table: MotionTaskTable,
    default_joint_position: torch.Tensor | None = None,
    default_joint_offset: torch.Tensor | None = None,
):
    payload = SimpleNamespace(table=table, sampler=SimpleNamespace(clip_priorities=torch.ones(len(table.clip_ids))))
    command = SimpleNamespace(payload=payload, table=table)
    command_manager = SimpleNamespace(get_term=lambda name: command if name == "motion" else None)
    physical_body_names = table.reference_frame_names
    if physical_body_names and physical_body_names[-1] == G1_HEAD_FRAME_NAME:
        physical_body_names = physical_body_names[:-1]
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


def test_expert_provider_consumes_one_strict_clock_record() -> None:
    """The runtime expert boundary must consume the grouped clock without flattening."""
    table = _smpl_table((5, 4))
    schema = ForwardBackwardObservationSchema.from_config({"policy": 358}, {"backward": ("policy",)})
    kwargs = {
        "source_bind": _SOURCE_BIND,
        "priorities_bind": _PRIORITIES_BIND,
        "target_projection": _SMPL_TARGET,
        "target_projection_binds": _SMPL_BINDS,
        "window_lengths": (2,),
    }

    expert = forward_backward_expert_buffer(
        _env(table),
        schema,
        "cpu",
        clock={"sampling_mode": "source_rows", "sampling_step_seconds": None},
        **kwargs,
    )

    assert expert.frames.shape == (9, 358)
    parameters = inspect.signature(forward_backward_expert_buffer).parameters
    assert "clock" in parameters
    assert {"sampling_mode", "sampling_step_seconds"}.isdisjoint(parameters)

    with pytest.raises(ValueError, match="missing required field"):
        forward_backward_expert_buffer(
            _env(table),
            schema,
            "cpu",
            clock={"sampling_mode": "source_rows"},
            **kwargs,
        )
    with pytest.raises(ValueError, match="Unknown expert clock fields"):
        forward_backward_expert_buffer(
            _env(table),
            schema,
            "cpu",
            clock={"sampling_mode": "source_rows", "sampling_step_seconds": None, "extra": True},
            **kwargs,
        )


def test_projection_bind_expressions_are_resolved_once() -> None:
    """Each explicitly owned projection input must be resolved once at expert construction."""
    table = _g1_table((10,))
    schema = ForwardBackwardObservationSchema.from_config(
        _G1_FIELD_WIDTHS,
        {"backward": _G1_BACKWARD_ROUTE},
    )
    env = _env(table)
    action = env.unwrapped.action_manager.get_term("joint_position")

    class CountingScene(dict[str, object]):
        def __init__(self, values: dict[str, object]) -> None:
            super().__init__(values)
            self.reads = 0

        def __getitem__(self, key: str) -> object:
            self.reads += 1
            return super().__getitem__(key)

    scene = CountingScene(env.unwrapped.scene)
    env.unwrapped.scene = scene
    action_reads = 0

    def get_action(name: str) -> object:
        nonlocal action_reads
        action_reads += 1
        assert name == "joint_position"
        return action

    env.unwrapped.action_manager = SimpleNamespace(get_term=get_action)
    forward_backward_expert_buffer(
        env,
        schema,
        "cpu",
        source_bind=_SOURCE_BIND,
        priorities_bind=_PRIORITIES_BIND,
        clock={"sampling_mode": "uniform_before_source_end", "sampling_step_seconds": 0.02},
        target_projection=_G1_TARGET,
        target_projection_binds=_G1_BINDS,
        window_lengths=(2,),
    )

    assert scene.reads == 1
    assert action_reads == 1


def test_smpl_provider_projects_target_physical_body_fields() -> None:
    table = _smpl_table((5, 4))
    schema = ForwardBackwardObservationSchema.from_config(
        {"policy": 358},
        {"backward": ("policy",)},
    )

    expert = forward_backward_expert_buffer(
        _env(table),
        schema,
        "cpu",
        source_bind=_SOURCE_BIND,
        priorities_bind=_PRIORITIES_BIND,
        clock={"sampling_mode": "source_rows", "sampling_step_seconds": None},
        target_projection=_SMPL_TARGET,
        target_projection_binds=_SMPL_BINDS,
        window_lengths=(2,),
        seed=7,
    )

    assert expert.frames.shape == (9, 358)
    torch.testing.assert_close(expert.frames, _smpl_observation(table))
    assert not hasattr(table.frames, "observation")
    assert expert.clip_offsets.tolist() == [0, 5, 9]
    assert expert.clip_ids == table.clip_ids == ("clip_0", "clip_1")
    assert expert.schema.dataset_id == f"{_hash('source')}:smpl_generalized_coordinate_exact_mjcf_v1"


def test_g1_provider_projects_exact_state_and_privileged_facts_at_50_hz() -> None:
    table = _g1_table((10, 7))
    schema = ForwardBackwardObservationSchema.from_config(
        _G1_FIELD_WIDTHS,
        {"backward": _G1_BACKWARD_ROUTE},
    )
    default_joint_position = torch.linspace(-0.2, 0.2, 29)

    expert = forward_backward_expert_buffer(
        _env(table, default_joint_position),
        schema,
        "cpu",
        source_bind=_SOURCE_BIND,
        priorities_bind=_PRIORITIES_BIND,
        clock={"sampling_mode": "uniform_before_source_end", "sampling_step_seconds": 0.02},
        target_projection=_G1_TARGET,
        target_projection_binds=_G1_BINDS,
        window_lengths=(2,),
        seed=11,
    )

    assert expert.frames.shape == (25, 527)
    assert isinstance(expert, ForwardBackwardExpertBuffer)
    assert expert.clip_length_values == (15, 10)
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
    fields = g1_bfm_expert_observation_fields(
        body_rotation[:, 0],
        body_angular_velocity[:, 0],
        joint_position,
        joint_velocity,
        default_joint_position,
    )
    privileged = g1_bfm_privileged_observation(
        body_position,
        body_rotation,
        body_linear_velocity,
        body_angular_velocity,
    )
    expected = torch.cat((*fields.values(), privileged), dim=-1)
    torch.testing.assert_close(expert.frames[1:2], expected)


def test_g1_expert_identity_tracks_canonical_defaults_but_not_randomized_offsets() -> None:
    table = _g1_table((10,))
    schema = ForwardBackwardObservationSchema.from_config(
        _G1_FIELD_WIDTHS,
        {"backward": _G1_BACKWARD_ROUTE},
    )
    canonical = torch.linspace(-0.2, 0.2, 29)

    def build(defaults: torch.Tensor, offsets: torch.Tensor):
        return forward_backward_expert_buffer(
            _env(table, defaults, offsets),
            schema,
            "cpu",
            source_bind=_SOURCE_BIND,
            priorities_bind=_PRIORITIES_BIND,
            clock={"sampling_mode": "uniform_before_source_end", "sampling_step_seconds": 0.02},
            target_projection=_G1_TARGET,
            target_projection_binds=_G1_BINDS,
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
        _G1_FIELD_WIDTHS,
        {"backward": _G1_BACKWARD_ROUTE},
    )

    with pytest.raises(ValueError, match="29 behavior-ordered joint defaults"):
        forward_backward_expert_buffer(
            _env(table, torch.zeros(2, 29)),
            schema,
            "cpu",
            source_bind=_SOURCE_BIND,
            priorities_bind=_PRIORITIES_BIND,
            clock={"sampling_mode": "uniform_before_source_end", "sampling_step_seconds": 0.02},
            target_projection=_G1_TARGET,
            target_projection_binds=_G1_BINDS,
            window_lengths=(2,),
        )


def test_g1_provider_accepts_smpl_source_after_target_frame_building() -> None:
    table = _g1_table(
        (8, 6),
        frame_builder_version="g1_local_body_pose_ordered_hinge_fit_v1",
    )
    schema = ForwardBackwardObservationSchema.from_config(
        _G1_FIELD_WIDTHS,
        {"backward": _G1_BACKWARD_ROUTE},
    )

    expert = forward_backward_expert_buffer(
        _env(table),
        schema,
        "cpu",
        source_bind=_SOURCE_BIND,
        priorities_bind=_PRIORITIES_BIND,
        clock={"sampling_mode": "uniform_before_source_end", "sampling_step_seconds": 0.02},
        target_projection=_G1_TARGET,
        target_projection_binds=_G1_BINDS,
        window_lengths=(2,),
        seed=13,
    )

    assert expert.frames.shape == (21, 527)
    assert expert.clip_offsets.tolist() == [0, 12, 21]
    assert expert.schema.dataset_id == f"{_hash('source')}:g1_local_body_pose_ordered_hinge_fit_v1"


def test_g1_provider_packs_named_targets_in_declared_backward_order() -> None:
    table = _g1_table((10,))
    reordered_schema = ForwardBackwardObservationSchema.from_config(
        _G1_FIELD_WIDTHS,
        {"backward": ("privileged_state", *_G1_BACKWARD_ROUTE[:-1])},
    )
    canonical_schema = ForwardBackwardObservationSchema.from_config(
        _G1_FIELD_WIDTHS,
        {"backward": _G1_BACKWARD_ROUTE},
    )
    kwargs = {
        "source_bind": _SOURCE_BIND,
        "priorities_bind": _PRIORITIES_BIND,
        "clock": {"sampling_mode": "uniform_before_source_end", "sampling_step_seconds": 0.02},
        "target_projection": _G1_TARGET,
        "target_projection_binds": _G1_BINDS,
        "window_lengths": (2,),
    }

    reordered = forward_backward_expert_buffer(_env(table), reordered_schema, "cpu", **kwargs)
    canonical = forward_backward_expert_buffer(_env(table), canonical_schema, "cpu", **kwargs)

    torch.testing.assert_close(
        reordered.frames, torch.cat((canonical.frames[:, 64:], canonical.frames[:, :64]), dim=-1)
    )


def test_expert_provider_retains_the_command_sampler_priority_tensor() -> None:
    table = _smpl_table((5, 4))
    schema = ForwardBackwardObservationSchema.from_config({"policy": 358}, {"backward": ("policy",)})
    env = _env(table)
    priorities = env.unwrapped.command_manager.get_term("motion").payload.sampler.clip_priorities

    expert = forward_backward_expert_buffer(
        env,
        schema,
        "cpu",
        source_bind=_SOURCE_BIND,
        priorities_bind=_PRIORITIES_BIND,
        clock={"sampling_mode": "source_rows", "sampling_step_seconds": None},
        target_projection=_SMPL_TARGET,
        target_projection_binds=_SMPL_BINDS,
        window_lengths=(2,),
    )
    expert.set_priorities(torch.tensor((2.0, 3.0)))

    assert expert.priorities.data_ptr() == priorities.data_ptr()
    torch.testing.assert_close(priorities, torch.tensor((2.0, 3.0)))


def test_segment_boundaries_control_short_and_long_expert_window_eligibility() -> None:
    """Expert windows of length 8 and 257 never cross retained semantic segment boundaries."""
    table = _smpl_table((9, 258, 8, 257))
    schema = ForwardBackwardObservationSchema.from_config({"policy": 358}, {"backward": ("policy",)})
    expert = forward_backward_expert_buffer(
        _env(table),
        schema,
        "cpu",
        source_bind=_SOURCE_BIND,
        priorities_bind=_PRIORITIES_BIND,
        clock={"sampling_mode": "source_rows", "sampling_step_seconds": None},
        target_projection=_SMPL_TARGET,
        target_projection_binds=_SMPL_BINDS,
        window_lengths=(8, 257),
        seed=23,
    )

    torch.testing.assert_close(expert._eligible_priorities[8], torch.tensor((1.0, 1.0, 0.0, 1.0)))
    torch.testing.assert_close(expert._eligible_priorities[257], torch.tensor((0.0, 1.0, 0.0, 0.0)))
    for window_length in (8, 257):
        batch = expert.sample(64, window_length)
        segment_ends = expert.clip_offsets.index_select(0, batch.clip_ids + 1)
        assert torch.all(batch.frame_indices[:, -1] < segment_ends)
