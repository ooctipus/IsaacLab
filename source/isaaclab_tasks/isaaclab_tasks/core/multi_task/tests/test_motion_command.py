# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the motion task table and StateCommand payload."""

from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.math import quat_from_rotation_vector, quat_slerp

import isaaclab_tasks.core.multi_task.motion.data as motion_data_module
from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex, MotionFrames, MotionResetState, MotionSkeleton
from isaaclab_tasks.core.multi_task.motion.mdp.commands import (
    MotionSampler,
    MotionStatePayload,
    MotionTaskTable,
    MotionTaskTableCfg,
    build_motion_task_table,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands import commands_cfg as commands_cfg_module
from isaaclab_tasks.core.multi_task.motion.mdp.commands import motion_task_table as table_module


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


_LIVE_JOINT_NAMES = ("joint_b", "joint_a")


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
    )


def _index() -> MotionClipIndex:
    return MotionClipIndex(
        source_content_sha256=_hash("source"),
        clips=(
            MotionClipIndex.Clip(
                clip_id="clip_a",
                frame_count=3,
                source_fps=2.0,
                content_sha256=_hash("clip-a"),
            ),
            MotionClipIndex.Clip(
                clip_id="clip_b",
                frame_count=4,
                source_fps=4.0,
                content_sha256=_hash("clip-b"),
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


def _table(*, task_row_mode: str = "source_frames") -> MotionTaskTable:
    index = _index()
    frames = _empty_frames(index.total_frames)
    for clip_number, clip in enumerate(index.clips):
        start, end = index.offsets[clip_number : clip_number + 2]
        frames._copy_clip_(start, end, _clip_frames(clip_number, clip.frame_count))
    return MotionTaskTable(
        index,
        frames,
        _LIVE_JOINT_NAMES,
        (),
        "test_builder_v1",
        _hash("test-builder-construction"),
        task_row_mode,
        _skeleton().identity_sha256,
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
    table = MotionTaskTable(
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


def test_motion_task_table_cfg_streams_direct_source_into_exact_table_storage() -> None:
    """Config construction must allocate once and bind builder output directly."""
    index = _index()

    class Source:
        closed = False

        def inspect(self) -> MotionClipIndex:
            return index

        def clips(self):
            for clip_number, clip in enumerate(index.clips):
                yield clip.clip_id, {"clip_number": clip_number, "frame_count": clip.frame_count}

        def close(self) -> None:
            self.closed = True

    class Builder:
        version = "test_builder_v1"
        construction_identity_sha256 = _hash("test-builder-construction")
        joint_names = _LIVE_JOINT_NAMES
        reference_frame_names: tuple[str, ...] = ()
        allocated: MotionFrames | None = None

        def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionFrames:
            assert torch.device(device) == torch.device("cpu")
            self.allocated = _empty_frames(frame_count)
            return self.allocated

        def build_frames(self, fields: dict[str, int], *, device: str | torch.device) -> MotionFrames:
            assert torch.device(device) == torch.device("cpu")
            return _clip_frames(fields["clip_number"], fields["frame_count"])

    source = Source()
    builder = Builder()
    split = SimpleNamespace(
        source_content_sha256=index.source_content_sha256,
        clip_count=len(index.clips),
        frame_count=index.total_frames,
    )
    source_cfg = SimpleNamespace(
        train=split,
        evaluation=split,
        open_split=lambda _root, _split: source,
        build_skeleton=_skeleton,
    )
    cfg = MotionTaskTableCfg(
        source=source_cfg,
        source_artifact_root="unused",
        motion_split="train",
        frame_builder_factory=lambda _source_skeleton, _reference, _robot: builder,
        reference_kinematics_factory=lambda _root, _device: object(),
        task_row_mode="clip_time_ranges",
    )
    env = SimpleNamespace(
        device=torch.device("cpu"),
        cfg=SimpleNamespace(seed=7301),
        scene={"robot": object()},
    )
    table = build_motion_task_table(
        SimpleNamespace(task_table=cfg, payload=SimpleNamespace(robot_asset_name="robot")), env
    )

    assert table.frames is builder.allocated
    torch.testing.assert_close(table.clip_indices, torch.tensor((0, 1)))
    torch.testing.assert_close(
        table.reset_time_ranges_seconds,
        torch.tensor(((0.0, 1.0), (0.0, 0.75))),
    )
    assert source.closed


def test_motion_task_table_rejects_incomplete_frame_builder_before_allocation() -> None:
    """The source table validates one complete builder contract before allocating frame storage."""
    index = _index()

    class Source:
        closed = False

        def inspect(self) -> MotionClipIndex:
            return index

        def clips(self):
            raise AssertionError("An invalid frame builder must fail before consuming source clips.")

        def close(self) -> None:
            self.closed = True

    class IncompleteBuilder:
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
        train=split,
        evaluation=split,
        open_split=lambda _root, _split: source,
        build_skeleton=_skeleton,
    )
    cfg = MotionTaskTableCfg(
        source=source_cfg,
        source_artifact_root="unused",
        motion_split="train",
        frame_builder_factory=lambda _source_skeleton, _reference, _robot: IncompleteBuilder(),
        reference_kinematics_factory=lambda _root, _device: object(),
        task_row_mode="clip_time_ranges",
    )
    env = SimpleNamespace(device=torch.device("cpu"), cfg=SimpleNamespace(seed=7301), scene={"robot": object()})

    with pytest.raises(TypeError, match="MotionFrameBuilder"):
        build_motion_task_table(SimpleNamespace(task_table=cfg, payload=SimpleNamespace(robot_asset_name="robot")), env)
    assert source.closed


def test_motion_task_table_cfg_closes_source_when_frame_construction_fails() -> None:
    """The config-owned source lifetime must end even when one clip cannot build."""
    index = _index()

    class Source:
        closed = False

        def inspect(self) -> MotionClipIndex:
            return index

        def clips(self):
            yield index.clips[0].clip_id, {}

        def close(self) -> None:
            self.closed = True

    class Builder:
        version = "test_builder_v1"
        construction_identity_sha256 = _hash("test-builder-construction")
        joint_names = _LIVE_JOINT_NAMES
        reference_frame_names: tuple[str, ...] = ()

        def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionFrames:
            return _empty_frames(frame_count)

        def build_frames(self, fields: dict, *, device: str | torch.device) -> MotionFrames:
            raise RuntimeError("synthetic frame construction failure")

    source = Source()
    split = SimpleNamespace(
        source_content_sha256=index.source_content_sha256,
        clip_count=len(index.clips),
        frame_count=index.total_frames,
    )
    source_cfg = SimpleNamespace(
        train=split,
        evaluation=split,
        open_split=lambda _root, _split: source,
        build_skeleton=_skeleton,
    )
    cfg = MotionTaskTableCfg(
        source=source_cfg,
        source_artifact_root="unused",
        motion_split="train",
        frame_builder_factory=lambda _source_skeleton, _reference, _robot: Builder(),
        reference_kinematics_factory=lambda _root, _device: object(),
        task_row_mode="clip_time_ranges",
    )
    env = SimpleNamespace(device=torch.device("cpu"), cfg=SimpleNamespace(seed=7301), scene={"robot": object()})

    with pytest.raises(RuntimeError, match="synthetic frame construction failure"):
        build_motion_task_table(SimpleNamespace(task_table=cfg, payload=SimpleNamespace(robot_asset_name="robot")), env)
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
