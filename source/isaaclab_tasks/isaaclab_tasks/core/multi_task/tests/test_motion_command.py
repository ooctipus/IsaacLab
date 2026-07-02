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
from types import SimpleNamespace

import pytest
import torch

import isaaclab_tasks.core.multi_task.motion.data as motion_data_module
from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex, MotionSampleGrid, MotionSkeleton
from isaaclab_tasks.core.multi_task.motion.mdp.commands import (
    MotionStatePayload,
    MotionTaskTable,
    MotionTaskTableCfg,
    build_motion_task_table,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands import commands_cfg as commands_cfg_module
from isaaclab_tasks.core.multi_task.motion.mdp.commands import motion_state_payload as payload_module
from isaaclab_tasks.core.multi_task.motion.mdp.commands import motion_task_table as table_module


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


_LIVE_JOINT_NAMES = ("joint_b", "joint_a")
_TEST_EXPERT_GRID = MotionSampleGrid.source_rows()


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


def _index(*, valid: tuple[bool, bool] = (True, True)) -> MotionClipIndex:
    return MotionClipIndex(
        source_content_sha256=_hash("source"),
        skeleton_sha256=_skeleton().identity_sha256,
        semantic_level="robot_state",
        license="test-only",
        clips=(
            MotionClipIndex.Clip(
                clip_id="clip_a",
                source_path="clip_a.tensor",
                frame_count=3,
                source_fps=2.0,
                split="train",
                tags=(),
                content_sha256=_hash("clip-a"),
                valid=valid[0],
            ),
            MotionClipIndex.Clip(
                clip_id="clip_b",
                source_path="clip_b.tensor",
                frame_count=4,
                source_fps=4.0,
                split="train",
                tags=(),
                content_sha256=_hash("clip-b"),
                valid=valid[1],
            ),
        ),
    )


def _empty_frames(frame_count: int) -> MotionTaskTable.Frames:
    return MotionTaskTable.Frames(
        root_position=torch.empty(frame_count, 3),
        root_rotation=torch.empty(frame_count, 4),
        root_linear_velocity=torch.empty(frame_count, 3),
        root_angular_velocity=torch.empty(frame_count, 3),
        joint_position=torch.empty(frame_count, 2),
        joint_velocity=torch.empty(frame_count, 2),
        observation=torch.empty(frame_count, 1),
    )


def _clip_frames(clip_number: int, frame_count: int) -> MotionTaskTable.Frames:
    frame = torch.arange(frame_count, dtype=torch.float32)
    base = 100.0 * clip_number + frame
    angle = frame * (math.pi / 2.0)
    zeros = torch.zeros_like(angle)
    return MotionTaskTable.Frames(
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
        observation=torch.full((frame_count, 1), 5.0 + clip_number),
    )


def _table(
    *,
    valid: tuple[bool, bool] = (True, True),
    task_row_mode: str = "source_frames",
    reset_sources: tuple[tuple[str, float], ...] = (("reference", 1.0),),
) -> MotionTaskTable:
    index = _index(valid=valid)
    frames = _empty_frames(index.total_frames)
    writer = MotionTaskTable.writer(
        index,
        frames,
        _LIVE_JOINT_NAMES,
        (),
        "test_builder_v1",
        _hash("test-builder-construction"),
        task_row_mode,
        reset_sources,
        _TEST_EXPERT_GRID,
        seed=7301,
    )
    for clip_number, clip in enumerate(index.clips):
        writer.write_clip(clip.clip_id, _clip_frames(clip_number, clip.frame_count))
    return writer.finish()


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


def _payload(
    num_envs: int,
    *,
    states_relative: bool = True,
    command_fields: tuple[str, ...] = (),
    step_fields: tuple[str, ...] | None = None,
    step_dt: float = 0.25,
    episode_length_steps: int = 8,
    reset_sources: tuple[tuple[str, float], ...] = (("reference", 1.0),),
    auxiliary_evidence: tuple[str, ...] = (),
    reset_transform_factory: Callable[
        [object],
        Callable[
            [MotionStatePayload.ResetState, torch.Tensor, torch.Generator],
            MotionStatePayload.ResetState,
        ],
    ]
    | None = None,
    root_velocity_frame: str = "link",
) -> tuple[MotionStatePayload, MotionTaskTable, _Robot]:
    table = _table(reset_sources=reset_sources)
    robot = _Robot(num_envs)
    origins = torch.zeros(num_envs, 3)
    origins[:, 0] = torch.arange(num_envs, dtype=torch.float32) * 10.0
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=num_envs,
        step_dt=step_dt,
        scene=_Scene(robot, origins),
    )
    evidence = SimpleNamespace(
        name="tracking_cost",
        width=1,
        unit="m",
        anchor="transition_reached_physics",
    )
    payload_cfg = SimpleNamespace(
        robot_asset_name="robot",
        reset_transform_factory=reset_transform_factory,
        step_fields=table.frames.available_fields if step_fields is None else step_fields,
        command_fields=command_fields,
        episode_length_steps=episode_length_steps,
        history_fields=(),
        history_length=0,
        raw_evidence=(evidence,),
        root_velocity_frame=root_velocity_frame,
        auxiliary_evidence=auxiliary_evidence,
    )
    cfg = SimpleNamespace(payload=payload_cfg, states_relative=states_relative)
    return MotionStatePayload(cfg, env, table), table, robot


def _bind(payload: MotionStatePayload, task_rows: torch.Tensor) -> None:
    env_ids = torch.arange(task_rows.shape[0], dtype=torch.int64)
    payload.bind(env_ids, task_rows)


def test_motion_task_rows_select_clip_reset_time_and_source_deterministically() -> None:
    """Explicit descriptor rows must bind the same clip/time metadata every time."""
    payload, table, _robot = _payload(3)
    task_rows = torch.tensor((4, 0, 2))
    _bind(payload, task_rows)

    torch.testing.assert_close(payload.clip_indices, torch.tensor((1, 0, 0)))
    torch.testing.assert_close(
        payload.reset_time_ranges_seconds,
        torch.tensor(((0.25, 0.25), (0.0, 0.0), (1.0, 1.0))),
    )
    torch.testing.assert_close(payload.reset_source_indices, torch.zeros(3, dtype=torch.int64))
    torch.testing.assert_close(payload.reference_time_seconds, torch.tensor((0.25, 0.0, 1.0)))
    assert table.num_tasks == 7

    first = {name: value.clone() for name, value in payload.motion_facts.items()}
    _bind(payload, task_rows)
    for name, value in first.items():
        torch.testing.assert_close(payload.motion_facts[name], value)


def test_motion_exact_clip_binding_uses_reference_start_and_normal_reset_semantics() -> None:
    """Evaluation resets must bypass task/source sampling without bypassing reset semantics."""
    payload, _table, robot = _payload(2)
    env_ids = torch.tensor((0, 1), dtype=torch.int64)
    clip_indices = torch.tensor((1, 0), dtype=torch.int64)

    payload.bind_clip_start(env_ids, clip_indices)

    torch.testing.assert_close(payload.clip_indices, clip_indices)
    torch.testing.assert_close(payload.reset_time_ranges_seconds, torch.zeros(2, 2))
    torch.testing.assert_close(payload.reset_source_indices, torch.zeros(2, dtype=torch.int64))
    torch.testing.assert_close(payload.reference_time_seconds, torch.zeros(2))
    torch.testing.assert_close(
        robot.root_state[:, :3],
        torch.tensor(((100.0, 101.0, 102.0), (10.0, 1.0, 2.0))),
    )
    torch.testing.assert_close(
        robot.joint_position,
        torch.tensor(((120.0, 110.0), (20.0, 10.0))),
    )


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
        source_skeleton = _skeleton()
        version = "test_builder_v1"
        construction_identity_sha256 = _hash("test-builder-construction")
        joint_names = _LIVE_JOINT_NAMES
        reference_frame_names: tuple[str, ...] = ()
        allocated: MotionTaskTable.Frames | None = None

        def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionTaskTable.Frames:
            assert torch.device(device) == torch.device("cpu")
            self.allocated = _empty_frames(frame_count)
            return self.allocated

        def build_frames(self, fields: dict[str, int], *, device: str | torch.device) -> MotionTaskTable.Frames:
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
    )
    cfg = MotionTaskTableCfg(
        source=source_cfg,
        source_artifact_root="unused",
        motion_split="train",
        frame_builder_factory=lambda _env: builder,
        reference_kinematics_factory=lambda _env: object(),
        expert_sample_grid=_TEST_EXPERT_GRID,
        task_row_mode="clip_time_ranges",
        reset_sources=(("reference", 0.8), ("fall", 0.2)),
    )
    env = SimpleNamespace(
        device=torch.device("cpu"),
        cfg=SimpleNamespace(seed=7301),
    )
    table = build_motion_task_table(SimpleNamespace(task_table=cfg), env)

    assert table.frames is builder.allocated
    torch.testing.assert_close(table.clip_indices, torch.tensor((0, 1)))
    torch.testing.assert_close(
        table.reset_time_ranges_seconds,
        torch.tensor(((0.0, 1.0), (0.0, 0.75))),
    )
    assert table.reset_source_names == ("reference", "fall")
    torch.testing.assert_close(table.reset_source_probabilities, torch.tensor((0.8, 0.2)))
    assert table.seed == 7301
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
        source_skeleton = _skeleton()
        version = "test_builder_v1"
        construction_identity_sha256 = _hash("test-builder-construction")
        joint_names = _LIVE_JOINT_NAMES
        reference_frame_names: tuple[str, ...] = ()

        def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionTaskTable.Frames:
            return _empty_frames(frame_count)

        def build_frames(self, fields: dict, *, device: str | torch.device) -> MotionTaskTable.Frames:
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
    )
    cfg = MotionTaskTableCfg(
        source=source_cfg,
        source_artifact_root="unused",
        motion_split="train",
        frame_builder_factory=lambda _env: Builder(),
        reference_kinematics_factory=lambda _env: object(),
        expert_sample_grid=_TEST_EXPERT_GRID,
        task_row_mode="clip_time_ranges",
        reset_sources=(("reference", 1.0),),
    )
    env = SimpleNamespace(device=torch.device("cpu"), cfg=SimpleNamespace(seed=7301))

    with pytest.raises(RuntimeError, match="synthetic frame construction failure"):
        build_motion_task_table(SimpleNamespace(task_table=cfg), env)
    assert source.closed


def test_motion_bind_maps_reset_fields_to_simulator_joint_order_and_env_origins() -> None:
    """The robot codec and payload must produce exact simulator-ready reset rows."""
    payload, _table, robot = _payload(2)
    _bind(payload, torch.tensor((4, 1)))
    expected_position = torch.tensor(((101.0, 102.0, 103.0), (11.0, 2.0, 3.0)))
    torch.testing.assert_close(robot.root_state[:, :3], expected_position)
    torch.testing.assert_close(robot.root_state[:, 3:7], payload.reference["root_rotation"])
    torch.testing.assert_close(robot.joint_position, torch.tensor(((121.0, 111.0), (21.0, 11.0))))
    torch.testing.assert_close(robot.joint_velocity, torch.tensor(((161.0, 151.0), (61.0, 51.0))))
    torch.testing.assert_close(payload.reset_state.root_position, expected_position)


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
    ) -> torch.Tensor:
        torch.testing.assert_close(probabilities, torch.tensor((0.8, 0.2)))
        assert count == 2
        assert replacement
        sample_generators.append(generator)
        return sampled_sources

    monkeypatch.setattr(torch, "multinomial", sample)
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Generator]] = []

    def transform(
        decoded: MotionStatePayload.ResetState,
        reset_source_indices: torch.Tensor,
        generator: torch.Generator,
    ) -> MotionStatePayload.ResetState:
        calls.append(
            (
                decoded.root_position.clone(),
                reset_source_indices.clone(),
                generator,
            )
        )
        return MotionStatePayload.ResetState(
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
        reset_transform_factory=lambda _env: transform,
    )
    _bind(payload, torch.tensor((0, 4)))

    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], torch.tensor(((0.0, 1.0, 2.0), (101.0, 102.0, 103.0))))
    torch.testing.assert_close(calls[0][1], sampled_sources)
    assert calls[0][2] is payload.table.generator
    assert sample_generators == [payload.table.generator]
    expected_position = torch.tensor(((10.0, 11.0, 12.0), (101.0, 102.0, 103.0)))
    torch.testing.assert_close(payload.reset_state.root_position, expected_position)
    torch.testing.assert_close(robot.root_state[:, :3], expected_position)


def test_payload_rejects_reset_transform_source_name_mismatch() -> None:
    """Reset-source indices are meaningful only under one exact ordered name contract."""
    with pytest.raises(ValueError, match="source names differ"):
        _payload(
            1,
            reset_sources=(("reference", 0.8), ("fall", 0.2)),
            reset_transform_factory=lambda _env: SimpleNamespace(reset_source_names=("reference", "lie_down")),
        )


def test_source_frame_rows_use_output_offsets_when_invalid_clips_are_skipped() -> None:
    """Exact-frame row times must stay clip-local when earlier storage clips are invalid."""
    table = _table(valid=(False, True), task_row_mode="source_frames")

    torch.testing.assert_close(table.clip_indices, torch.ones(4, dtype=torch.int64))
    torch.testing.assert_close(table.clip_start_rows, torch.tensor((-1, 0), dtype=torch.int64))
    expected_time = torch.arange(4, dtype=torch.float32) / 4.0
    torch.testing.assert_close(table.reset_time_ranges_seconds[:, 0], expected_time)
    torch.testing.assert_close(table.reset_time_ranges_seconds[:, 1], expected_time)
    assert table.reset_source_names == ("reference",)
    torch.testing.assert_close(table.reset_source_probabilities, torch.ones(1))


def test_task_row_modes_keep_reset_source_distributions_compact() -> None:
    """80/20 and 70/30 reset modes must not duplicate descriptor rows."""
    source = _table(
        task_row_mode="source_frames",
        reset_sources=(("motion", 0.8), ("fall", 0.2)),
    )
    ranges = _table(
        task_row_mode="clip_time_ranges",
        reset_sources=(("reference", 0.7), ("lie_down", 0.3)),
    )

    assert source.clip_indices.shape == (7,)
    assert ranges.clip_indices.shape == (2,)
    assert source.reset_source_names == ("motion", "fall")
    assert ranges.reset_source_names == ("reference", "lie_down")
    torch.testing.assert_close(source.reset_source_probabilities, torch.tensor((0.8, 0.2)))
    torch.testing.assert_close(ranges.reset_source_probabilities, torch.tensor((0.7, 0.3)))
    assert source.task_sampling_law == "clip_categorical_then_discrete_source_frame_v1"
    assert ranges.task_sampling_law == "clip_categorical_then_continuous_time_v1"


def test_unequal_clip_lengths_keep_clip_sampling_uniform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clip selection and conditional frame selection must be separate exact draws."""
    table = _table(task_row_mode="source_frames")

    assert table.num_tasks == 7
    assert len(table.clip_ids) == 2
    torch.testing.assert_close(table.clip_indices, torch.tensor((0, 0, 0, 1, 1, 1, 1)))
    torch.testing.assert_close(table._sampling_row_starts, torch.tensor((0, 3)))
    torch.testing.assert_close(table._sampling_row_counts, torch.tensor((3, 4)))
    clip_draws = torch.tensor((0, 1, 0, 1), dtype=torch.int64)
    frame_draws = torch.tensor((0.0, 0.0, 0.999, 0.74))

    def sample_clips(weights, count, replacement, *, generator):
        torch.testing.assert_close(weights, torch.ones(2))
        assert weights.data_ptr() == table.clip_priorities.data_ptr()
        assert count == clip_draws.shape[0]
        assert replacement
        assert generator is table.generator
        return clip_draws

    def sample_frames(count, *, device, generator):
        assert count == frame_draws.shape[0]
        assert device == table.device
        assert generator is table.generator
        return frame_draws

    monkeypatch.setattr(torch, "multinomial", sample_clips)
    monkeypatch.setattr(torch, "rand", sample_frames)

    torch.testing.assert_close(table.sample_rows(clip_draws.shape[0]), torch.tensor((0, 3, 2, 5)))


def test_one_row_per_clip_sampling_is_exactly_the_clip_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip-range table must not consume a redundant within-clip random draw."""
    table = _table(task_row_mode="clip_time_ranges")
    expected = torch.tensor((1, 0, 1, 1), dtype=torch.int64)
    torch.testing.assert_close(table._sampling_row_starts, torch.tensor((0, 1)))
    torch.testing.assert_close(table._sampling_row_counts, torch.ones(2, dtype=torch.int64))

    def sample(
        weights: torch.Tensor,
        count: int,
        replacement: bool,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        torch.testing.assert_close(weights, torch.ones(2))
        assert weights.data_ptr() == table.clip_priorities.data_ptr()
        assert count == expected.shape[0]
        assert replacement
        assert generator is table.generator
        return expected

    def reject_frame_draw(*_args: object, **_kwargs: object) -> torch.Tensor:
        pytest.fail("one-row clip sampling must not draw a local row")

    monkeypatch.setattr(torch, "multinomial", sample)
    monkeypatch.setattr(torch, "rand", reject_frame_draw)

    torch.testing.assert_close(table.sample_rows(expected.shape[0]), expected)


def test_task_table_priorities_are_total_clip_mass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Priority updates must change clip mass without multiplying it by row count."""
    table = _table(task_row_mode="source_frames")
    priorities = torch.tensor((2.0, 8.0))
    table.set_clip_priorities(table.clip_ids, priorities)
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
    ) -> torch.Tensor:
        torch.testing.assert_close(weights, priorities)
        assert weights.data_ptr() == table.clip_priorities.data_ptr()
        assert count == clip_draws.shape[0]
        assert replacement
        assert generator is table.generator
        return clip_draws

    def sample_frames(
        count: int,
        *,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        assert count == frame_draws.shape[0]
        assert device == table.device
        assert generator is table.generator
        return frame_draws

    monkeypatch.setattr(torch, "multinomial", sample)
    monkeypatch.setattr(torch, "rand", sample_frames)

    sampled_rows = table.sample_rows(clip_draws.shape[0])

    torch.testing.assert_close(sampled_rows, torch.tensor((6, 1, 3)))
    torch.testing.assert_close(table.clip_indices[sampled_rows], clip_draws)
    previous = table.clip_priorities.clone()
    with pytest.raises(ValueError, match="clip ids"):
        table.set_clip_priorities(("clip_b", "clip_a"), torch.ones(2))
    torch.testing.assert_close(table.clip_priorities, previous)


def test_task_table_priority_reset_restores_equal_clip_mass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset must restore one unit of mass per clip regardless of row count."""
    table = _table(task_row_mode="source_frames")
    table.set_clip_priorities(table.clip_ids, torch.tensor((2.0, 8.0)))
    table.reset_clip_priorities()
    expected = torch.tensor((0, 1), dtype=torch.int64)

    def sample(
        weights: torch.Tensor,
        count: int,
        replacement: bool,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        torch.testing.assert_close(weights, torch.ones(2))
        assert count == expected.shape[0]
        assert replacement
        assert generator is table.generator
        return expected

    def sample_frames(
        count: int,
        *,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        assert count == expected.shape[0]
        assert device == table.device
        assert generator is table.generator
        return torch.zeros(count)

    monkeypatch.setattr(torch, "multinomial", sample)
    monkeypatch.setattr(torch, "rand", sample_frames)

    torch.testing.assert_close(table.clip_priorities, torch.ones(2))
    torch.testing.assert_close(table.sample_rows(expected.shape[0]), torch.tensor((0, 3)))


@pytest.mark.parametrize("probabilities", ((0.8, 0.2), (0.7, 0.3)))
def test_reset_source_sampling_uses_declared_distribution_exactly(
    probabilities: tuple[float, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling must forward each compact distribution without descriptor coupling."""
    declared = torch.tensor(probabilities)
    table = _table(
        task_row_mode="clip_time_ranges",
        reset_sources=(("reference", probabilities[0]), ("alternative", probabilities[1])),
    )
    expected = torch.tensor((1, 0, 1, 1), dtype=torch.int64)

    def sample(
        weights: torch.Tensor,
        count: int,
        replacement: bool,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        torch.testing.assert_close(weights, declared)
        assert weights.data_ptr() == table.reset_source_probabilities.data_ptr()
        assert count == expected.shape[0]
        assert replacement
        assert generator is table.generator
        return expected

    monkeypatch.setattr(torch, "multinomial", sample)
    torch.testing.assert_close(table.sample_reset_sources(expected.shape[0]), expected)


def test_payload_owns_ordered_raw_and_auxiliary_evidence_metadata() -> None:
    """One payload contract must own both raw storage order and learner column order."""
    payload, _table, _robot = _payload(2, auxiliary_evidence=("tracking_cost",))

    assert payload.raw_evidence_names == ("tracking_cost",)
    assert tuple(payload.raw_evidence) == payload.raw_evidence_names
    assert tuple(spec.name for spec in payload.raw_evidence_specs) == payload.raw_evidence_names
    assert payload.auxiliary_evidence_names == ("tracking_cost",)
    assert tuple(spec.name for spec in payload.auxiliary_evidence_specs) == payload.auxiliary_evidence_names
    assert payload.raw_evidence_value.shape == (2, 1)
    assert payload.raw_evidence_value.is_contiguous()
    assert (
        payload.raw_evidence["tracking_cost"].untyped_storage().data_ptr()
        == payload.raw_evidence_value.untyped_storage().data_ptr()
    )


def test_payload_rejects_ambiguous_auxiliary_evidence_selection() -> None:
    """Learner columns must be unique scalar channels selected from declared raw evidence."""
    with pytest.raises(ValueError, match="unique and nonempty"):
        _payload(1, auxiliary_evidence=("tracking_cost", "tracking_cost"))
    with pytest.raises(ValueError, match="Unknown auxiliary evidence channels"):
        _payload(1, auxiliary_evidence=("missing",))


@pytest.mark.parametrize("episode_length_steps", (300, 501))
def test_episode_phase_reaches_one_on_the_exact_timeout_edge(episode_length_steps: int) -> None:
    """The payload clock and native timeout edge must count the same applied action."""
    payload, _table, _robot = _payload(1, episode_length_steps=episode_length_steps)
    _bind(payload, torch.tensor((0,)))
    payload.episode_relative_step.fill_(episode_length_steps - 1)

    payload.record_step()

    torch.testing.assert_close(payload.episode_relative_step, torch.tensor((episode_length_steps,)))
    torch.testing.assert_close(payload.episode_phase, torch.ones(1))


def test_record_step_advances_interpolated_reference_and_named_raw_evidence() -> None:
    """Pre-final step recording must advance once and match the table interpolation oracle."""
    payload, _table, _robot = _payload(1, command_fields=("joint_position", "observation"))
    _bind(payload, torch.tensor((0,)))
    payload.raw_evidence["tracking_cost"].fill_(2.5)
    payload.record_step()

    oracle = payload.table.reference_view(payload.clip_indices, payload.reference_time_seconds)
    for name in payload.reference:
        torch.testing.assert_close(payload.reference[name], oracle.field(name))
    torch.testing.assert_close(payload.episode_relative_step, torch.ones(1, dtype=torch.int64))
    torch.testing.assert_close(payload.episode_time_seconds, torch.tensor((0.25,)))
    torch.testing.assert_close(payload.episode_phase, torch.tensor((0.125,)))
    torch.testing.assert_close(payload.motion_facts["reference_phase"], torch.tensor((0.25,)))
    torch.testing.assert_close(payload.raw_evidence["tracking_cost"], torch.tensor(((2.5,),)))

    time_before_compute = payload.reference_time_seconds.clone()
    command = torch.empty(1, payload.command_dim)
    payload.update(123.0, command, torch.empty(1, 0))
    torch.testing.assert_close(payload.reference_time_seconds, time_before_compute)
    torch.testing.assert_close(command[:, :2], payload.reference["joint_position"])
    torch.testing.assert_close(command[:, 2:], payload.reference["observation"])


@pytest.mark.parametrize("same_rotation", (False, True))
def test_payload_slerp_matches_allocating_table_reference_for_signed_quaternions(
    same_rotation: bool,
) -> None:
    """In-place hot-path slerp must follow the table shortest-arc law exactly."""
    payload, table, _robot = _payload(1, step_dt=0.25, step_fields=("root_rotation",))
    if same_rotation:
        table.frames.root_rotation[1].copy_(-table.frames.root_rotation[0])
    else:
        table.frames.root_rotation[1].neg_()
    _bind(payload, torch.tensor((0,)))

    payload.record_step()

    oracle = table.reference_view(payload.clip_indices, payload.reference_time_seconds)
    torch.testing.assert_close(
        payload.reference["root_rotation"],
        oracle.field("root_rotation"),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_reset_only_fields_are_not_gathered_after_each_transition() -> None:
    """FB environments advance clock metadata without refreshing reset-only trajectory tensors."""
    payload, _table, _robot = _payload(1, step_fields=())
    _bind(payload, torch.tensor((0,)))
    before = {name: value.clone() for name, value in payload._resolved_reference.items()}

    payload.record_step()

    torch.testing.assert_close(payload.reference_time_seconds, torch.tensor((0.25,)))
    torch.testing.assert_close(payload.motion_facts["reference_phase"], torch.tensor((0.25,)))
    for name, value in before.items():
        torch.testing.assert_close(payload._resolved_reference[name], value)


def test_reference_tail_is_explicit_and_holds_the_last_frame_without_terminating() -> None:
    """A reference may end before its episode without wrapping or ending the task."""
    payload, _table, _robot = _payload(1, step_dt=0.5)
    payload.bind_clip_start(torch.tensor((0,)), torch.tensor((0,)))
    payload.raw_evidence["tracking_cost"].zero_()

    payload.record_step()
    assert payload.motion_facts["tail_valid"].item()
    payload.record_step()
    assert payload.motion_facts["tail_valid"].item()
    endpoint = {name: value.clone() for name, value in payload.reference.items()}

    payload.record_step()
    assert not payload.motion_facts["tail_valid"].item()
    torch.testing.assert_close(payload.motion_facts["tail_elapsed_seconds"], torch.tensor((0.5,)))
    torch.testing.assert_close(payload.motion_facts["reference_phase"], torch.ones(1))
    for name, value in endpoint.items():
        torch.testing.assert_close(payload.reference[name], value)
    assert not payload.get_task_done().any()
    assert not payload.get_task_reward().any()


class _TransitionState:
    def __init__(self, num_envs: int) -> None:
        self.value = torch.zeros(num_envs, dtype=torch.int64)
        self.reset_calls: list[torch.Tensor] = []

    def reset(self, env_ids: torch.Tensor) -> None:
        self.value.index_fill_(0, env_ids, 0)
        self.reset_calls.append(env_ids.clone())


def test_terminal_evidence_survives_bind_while_transition_state_clears_reset_rows() -> None:
    """Same-step reset clears cross-edge state without destroying terminal evidence."""
    payload, _table, _robot = _payload(2)
    _bind(payload, torch.tensor((0, 0)))
    transition_state = _TransitionState(2)
    payload.attach_transition_state(transition_state)
    evidence = torch.tensor(((3.0,), (4.0,)))
    payload.raw_evidence["tracking_cost"].copy_(evidence)
    payload.record_step()
    transition_state.value.copy_(payload.episode_relative_step)
    torch.testing.assert_close(payload.raw_evidence["tracking_cost"], evidence)

    payload.bind(torch.tensor((0,)), torch.tensor((1,)))
    torch.testing.assert_close(transition_state.value, torch.tensor((0, 1)))
    torch.testing.assert_close(payload.raw_evidence["tracking_cost"], evidence)
    assert transition_state.reset_calls[0].tolist() == [0]


@pytest.mark.parametrize("num_envs", (1, 16, 1024))
def test_motion_reference_and_temporal_state_are_clone_invariant(num_envs: int) -> None:
    """One descriptor follows the same fixed tensor law at all required vector scales."""
    payload, _table, _robot = _payload(num_envs, states_relative=False)
    _bind(payload, torch.zeros(num_envs, dtype=torch.int64))
    pointers = {
        **{f"reference.{name}": value.data_ptr() for name, value in payload.reference.items()},
        **{f"motion.{name}": value.data_ptr() for name, value in payload.motion_facts.items()},
        **{f"evidence.{name}": value.data_ptr() for name, value in payload.raw_evidence.items()},
    }
    payload.raw_evidence["tracking_cost"].fill_(2.0)
    for _ in range(3):
        payload.record_step()

    for value in payload.reference.values():
        torch.testing.assert_close(value, value[:1].expand_as(value))
    for name in ("episode_relative_step", "reference_time_seconds", "reference_phase", "tail_valid"):
        value = payload.motion_facts[name]
        torch.testing.assert_close(value, value[:1].expand_as(value))
    torch.testing.assert_close(
        payload.raw_evidence["tracking_cost"],
        payload.raw_evidence["tracking_cost"][:1].expand(num_envs, -1),
    )
    current_pointers = {
        **{f"reference.{name}": value.data_ptr() for name, value in payload.reference.items()},
        **{f"motion.{name}": value.data_ptr() for name, value in payload.motion_facts.items()},
        **{f"evidence.{name}": value.data_ptr() for name, value in payload.raw_evidence.items()},
    }
    assert current_pointers == pointers


def test_motion_update_path_has_no_convenience_view_or_host_sync_calls() -> None:
    """Runtime lookup must retain caller-owned storage and avoid allocating table conveniences."""
    source = "\n".join(
        (
            inspect.getsource(MotionStatePayload.record_step),
            inspect.getsource(MotionStatePayload.update),
            inspect.getsource(payload_module._MotionReferenceResolver.resolve),
            inspect.getsource(payload_module._MotionReferenceResolver._Field.resolve),
        )
    )
    for forbidden in (
        ".item(",
        ".tolist(",
        ".contiguous(",
        ".reference_view(",
        "torch.unique(",
        "torch.empty(",
        "torch.zeros(",
    ):
        assert forbidden not in source
    assert tuple(inspect.signature(MotionStatePayload.record_step).parameters) == ("self",)
    bind_source = inspect.getsource(MotionStatePayload.bind)
    assert "_raw_evidence" not in bind_source
    assert "raw_evidence_value" not in bind_source

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
