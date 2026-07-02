# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for isolated tracking evaluation and atomic curriculum updates."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import random
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer, ForwardBackwardExpertSchema
from tensordict import TensorDict

from isaaclab.envs.mdp import randomize_rigid_body_material

import isaaclab_tasks.core.multi_task.motion.tracking as tracking_module
from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex, MotionSampleGrid
from isaaclab_tasks.core.multi_task.motion.frames import G1_HEAD_FRAME_NAME
from isaaclab_tasks.core.multi_task.motion.impl import uniform_emd_warp as uniform_emd_module
from isaaclab_tasks.core.multi_task.motion.mdp.actions import MotionJointPositionAction, MotionMujocoControlAction
from isaaclab_tasks.core.multi_task.motion.mdp.commands import MotionStatePayload, MotionTaskTable
from isaaclab_tasks.core.multi_task.motion.mdp.curriculums import MotionPenaltyScaleCurriculum
from isaaclab_tasks.core.multi_task.motion.mdp.events import MotionPushVelocity
from isaaclab_tasks.core.multi_task.motion.tracking import (
    MotionTrackingCurriculum,
    MotionTrackingEvaluation,
    g1_motion_tracking_evaluator,
    motion_tracking_priorities,
)
from isaaclab_tasks.core.multi_task.motion_env import MotionImitationEnv

_CLIP_IDS = ("clip_a", "clip_b")
_G1_JOINT_NAMES = tuple(f"joint_{index}" for index in range(29))
_G1_BODY_NAMES = tuple(f"body_{index}" for index in range(30))
_G1_REFERENCE_FRAME_NAMES = (*_G1_BODY_NAMES, G1_HEAD_FRAME_NAME)
_SMPL_CLIP_IDS = ("clip_a", "clip_b", "clip_c", "clip_d")
_SMPL_LENGTHS = (6, 5, 4, 3)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _expert() -> ForwardBackwardExpertBuffer:
    schema = ForwardBackwardExpertSchema(
        dataset_id="tracking-test",
        data_hash="data",
        feature_schema_hash="features",
        clip_offsets_hash="offsets",
        expert_feature_width=3,
        num_frames=6,
        num_clips=2,
        window_lengths=(1,),
    )
    return ForwardBackwardExpertBuffer(
        torch.zeros(6, 3),
        torch.tensor((0, 3, 6), dtype=torch.int64),
        torch.ones(2),
        schema,
        clip_ids=_CLIP_IDS,
    )


def _g1_tracking_expert(device: str | torch.device = "cpu") -> ForwardBackwardExpertBuffer:
    offsets = (0, 3, 7)
    offset_tensor = torch.tensor(offsets, dtype=torch.int64, device=device)
    frames = torch.zeros(offsets[-1], 527, device=device)
    for clip_index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        phase = torch.arange(end - start, dtype=torch.float32, device=device) + 10.0 * clip_index
        frames[start:end, :29] = phase[:, None] * torch.linspace(0.01, 0.29, 29, device=device)
        frames[start:end, 29:64] = phase[:, None] * 0.001
        frames[start:end, 64:] = phase[:, None] * 0.002
    schema = ForwardBackwardExpertSchema(
        dataset_id="g1-tracking-test",
        data_hash="data",
        feature_schema_hash="features",
        clip_offsets_hash="offsets",
        expert_feature_width=527,
        num_frames=offsets[-1],
        num_clips=len(_CLIP_IDS),
        window_lengths=(1,),
    )
    return ForwardBackwardExpertBuffer(
        frames,
        offset_tensor,
        torch.ones(len(_CLIP_IDS), device=device),
        schema,
        clip_ids=_CLIP_IDS,
    )


class _TrackingObservationSchema:
    field_widths = (("state", 64), ("privileged_state", 463))

    @staticmethod
    def route(name: str) -> tuple[str, ...]:
        assert name == "backward"
        return ("state", "privileged_state")


class _TrackingModel:
    observation_schema = _TrackingObservationSchema()

    def __init__(self) -> None:
        self.deterministic_calls: list[bool] = []

    @staticmethod
    def backward_map(observations: TensorDict) -> torch.Tensor:
        return observations["state"][:, :4]

    @staticmethod
    def context_project(context: torch.Tensor) -> torch.Tensor:
        return context

    def action_sample(
        self,
        observations: TensorDict,
        context: torch.Tensor,
        *,
        deterministic: bool,
    ) -> torch.Tensor:
        self.deterministic_calls.append(deterministic)
        assert observations.batch_size == context.shape[:1]
        return torch.zeros(context.shape[0], 29, device=context.device)


class _TrackingEnvironment:
    def __init__(
        self,
        expert: ForwardBackwardExpertBuffer,
        *,
        num_envs: int = 1,
    ) -> None:
        self.expert = expert
        self.num_envs = num_envs
        self.max_episode_length = 4
        self.cfg = SimpleNamespace(
            commands=SimpleNamespace(motion=SimpleNamespace(payload=SimpleNamespace(episode_length_steps=4)))
        )
        self.device = expert.device
        self.default_joint_position = torch.linspace(-0.2, 0.2, 29, device=self.device)
        robot = SimpleNamespace(
            joint_names=_G1_JOINT_NAMES,
            body_names=_G1_BODY_NAMES,
            data=SimpleNamespace(
                joint_pos=SimpleNamespace(torch=torch.empty(num_envs, 29, device=self.device)),
            ),
        )
        table = _table()
        payload = object.__new__(MotionStatePayload)
        payload.table = table
        payload.robot = robot
        self.payload = payload
        command = SimpleNamespace(table=table, payload=payload)
        action = object.__new__(MotionJointPositionAction)
        action._joint_names = _G1_JOINT_NAMES
        action._joint_ids_tensor = torch.arange(29, dtype=torch.int64, device=self.device)
        action._asset = robot
        action._joint_position = torch.empty(num_envs, 29, device=self.device)
        action.joint_default_position = self.default_joint_position
        self.command_manager = SimpleNamespace(get_term=lambda name: command if name == "motion" else None)
        self.action_manager = SimpleNamespace(get_term=lambda name: action if name == "joint_position" else None)
        self.reset_assignments: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self._clips = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._steps = torch.zeros(num_envs, dtype=torch.int64, device=self.device)

    def _observation(self) -> TensorDict:
        lengths = self.expert.clip_lengths.index_select(0, self._clips)
        local_steps = torch.minimum(self._steps, lengths - 1)
        starts = self.expert.clip_offsets.index_select(0, self._clips)
        frames = self.expert.frames.index_select(0, starts + local_steps)
        return TensorDict(
            {
                "state": frames[:, :64],
                "last_action": torch.zeros(self.num_envs, 29, device=self.device),
                "history_actor": torch.zeros(self.num_envs, 372, device=self.device),
                "privileged_state": frames[:, 64:],
            },
            batch_size=[self.num_envs],
        )

    def _write_joint_position(self) -> None:
        lengths = self.expert.clip_lengths.index_select(0, self._clips)
        local_steps = torch.minimum(self._steps, lengths - 1)
        starts = self.expert.clip_offsets.index_select(0, self._clips)
        target = self.expert.frames.index_select(0, starts + local_steps)[:, :29] + self.default_joint_position
        self.payload.robot.data.joint_pos.torch.copy_(target)

    def reset_motion_clips(
        self,
        clip_indices: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[TensorDict, dict]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._clips.index_copy_(0, env_ids, clip_indices)
        self._steps.index_fill_(0, env_ids, 0)
        self.reset_assignments.append(
            (
                tuple(env_ids.cpu().tolist()),
                tuple(clip_indices.cpu().tolist()),
            )
        )
        self._write_joint_position()
        return self._observation(), {}

    def step(self, _action: torch.Tensor):
        self._steps.add_(1)
        self._write_joint_position()
        return (
            self._observation(),
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            {},
        )


def _table() -> MotionTaskTable:
    clips = tuple(
        MotionClipIndex.Clip(
            clip_id=clip_id,
            source_path=f"{clip_id}.tensor",
            frame_count=count,
            source_fps=50.0,
            split="train",
            tags=(),
            content_sha256=_hash(clip_id),
        )
        for clip_id, count in zip(_CLIP_IDS, (4, 5), strict=True)
    )
    index = MotionClipIndex(
        source_content_sha256=_hash("tracking-source"),
        skeleton_sha256=_hash("tracking-skeleton"),
        semantic_level="g1_tracking",
        license="test-only",
        clips=clips,
    )
    frame_count = index.total_frames
    body_rotation = torch.zeros(frame_count, 31, 4)
    body_rotation[..., 3] = 1.0
    frames = MotionTaskTable.Frames(
        joint_position=torch.zeros(frame_count, 29),
        joint_velocity=torch.zeros(frame_count, 29),
        body_position=torch.zeros(frame_count, 31, 3),
        body_rotation=body_rotation,
        body_linear_velocity=torch.zeros(frame_count, 31, 3),
        body_angular_velocity=torch.zeros(frame_count, 31, 3),
    )
    return MotionTaskTable.from_storage(
        index,
        frames,
        _G1_JOINT_NAMES,
        _G1_REFERENCE_FRAME_NAMES,
        "tracking_builder_v1",
        _hash("tracking-builder"),
        "clip_time_ranges",
        (("reference", 0.7), ("lie_down", 0.3)),
        MotionSampleGrid.uniform_before_source_end(step_seconds=0.02),
        seed=0,
    )


def _smpl_tracking_expert_table(
    device: str | torch.device,
) -> tuple[ForwardBackwardExpertBuffer, MotionTaskTable]:
    clips = tuple(
        MotionClipIndex.Clip(
            clip_id=clip_id,
            source_path=f"{clip_id}.hdf5",
            frame_count=length,
            source_fps=50.0,
            split="evaluation",
            tags=(),
            content_sha256=_hash(f"smpl-{clip_id}"),
        )
        for clip_id, length in zip(_SMPL_CLIP_IDS, _SMPL_LENGTHS, strict=True)
    )
    index = MotionClipIndex(
        source_content_sha256=_hash("smpl-tracking-source"),
        skeleton_sha256=_hash("smpl-tracking-skeleton"),
        semantic_level="smpl_tracking",
        license="test-only",
        clips=clips,
    )
    frame_count = index.total_frames
    observation = torch.zeros(frame_count, 358, dtype=torch.float32, device=device)
    for clip_index, (start, end) in enumerate(zip(index.offsets[:-1], index.offsets[1:], strict=True)):
        phase = torch.arange(end - start, dtype=torch.float32, device=device) + 10.0 * clip_index
        observation[start:end].copy_(phase[:, None] * torch.linspace(0.001, 0.358, 358, device=device))
    root_rotation = torch.zeros(frame_count, 4, dtype=torch.float32, device=device)
    root_rotation[:, 3] = 1.0
    frames = MotionTaskTable.Frames(
        root_position=torch.zeros(frame_count, 3, dtype=torch.float32, device=device),
        root_rotation=root_rotation,
        root_linear_velocity=torch.zeros(frame_count, 3, dtype=torch.float32, device=device),
        root_angular_velocity=torch.zeros(frame_count, 3, dtype=torch.float32, device=device),
        joint_position=torch.zeros(frame_count, 69, dtype=torch.float32, device=device),
        joint_velocity=torch.zeros(frame_count, 69, dtype=torch.float32, device=device),
        observation=observation,
    )
    table = MotionTaskTable.from_storage(
        index,
        frames,
        tuple(f"joint_{index}" for index in range(69)),
        (),
        "smpl_tracking_builder_v1",
        _hash("smpl-tracking-builder"),
        "source_frames",
        (("motion", 1.0),),
        MotionSampleGrid.source_rows(),
        seed=0,
    )
    schema = ForwardBackwardExpertSchema(
        dataset_id="smpl-packed-tracking-test",
        data_hash="data",
        feature_schema_hash="features",
        clip_offsets_hash="offsets",
        expert_feature_width=358,
        num_frames=frame_count,
        num_clips=len(_SMPL_CLIP_IDS),
        window_lengths=(1,),
    )
    expert = ForwardBackwardExpertBuffer(
        table.field("observation"),
        table.clip_offsets,
        torch.ones(len(_SMPL_CLIP_IDS), device=device),
        schema,
        clip_ids=_SMPL_CLIP_IDS,
    )
    return expert, table


class _PackedTrackingObservationSchema:
    field_widths = (("policy", 358),)

    @staticmethod
    def route(name: str) -> tuple[str, ...]:
        assert name == "backward"
        return ("policy",)


class _PackedTrackingModel:
    observation_schema = _PackedTrackingObservationSchema()
    context_dim = 4

    def __init__(self, policy_count: int) -> None:
        self.policy_count = policy_count
        self.action_calls = 0

    @staticmethod
    def backward_map(observations: TensorDict) -> torch.Tensor:
        return observations["policy"][..., :4]

    @staticmethod
    def context_project(context: torch.Tensor) -> torch.Tensor:
        return context

    def action_deterministic(self, observations: TensorDict, context: torch.Tensor) -> torch.Tensor:
        assert observations.batch_size == context.shape[:2]
        self.action_calls += 1
        return torch.zeros(*observations.batch_size, 69, dtype=torch.float32, device=context.device)


class _PackedTrackingEnvironment:
    def __init__(
        self,
        expert: ForwardBackwardExpertBuffer,
        table: MotionTaskTable,
        *,
        policy_count: int,
        lanes_per_policy: int,
    ) -> None:
        self.expert = expert
        self.num_envs = policy_count * lanes_per_policy
        self.max_episode_length = max(_SMPL_LENGTHS)
        self.device = expert.device
        self.policy_count = policy_count
        self.lanes_per_policy = lanes_per_policy
        payload = object.__new__(MotionStatePayload)
        payload.table = table
        command = SimpleNamespace(table=table, payload=payload)
        action = object.__new__(MotionMujocoControlAction)
        action.cfg = SimpleNamespace(action_width=69)
        self.command_manager = SimpleNamespace(get_term=lambda name: command if name == "motion" else None)
        self.action_manager = SimpleNamespace(get_term=lambda name: action if name == "joint_position" else None)
        self._clips = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.reset_events: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def _observation(self) -> TensorDict:
        lengths = self.expert.clip_lengths.index_select(0, self._clips)
        local_steps = torch.minimum(self._steps, lengths - 1)
        starts = self.expert.clip_offsets.index_select(0, self._clips)
        values = self.expert.frames.index_select(0, starts + local_steps).clone()
        policies = torch.arange(self.num_envs, device=self.device) // self.lanes_per_policy
        offsets = policies.to(torch.float32) * 0.25 + self._clips.to(torch.float32) * 0.05
        values[:, :214].add_(offsets.unsqueeze(1))
        return TensorDict({"policy": values}, batch_size=[self.num_envs])

    def reset_motion_clips_selected(
        self,
        clip_indices: torch.Tensor,
        *,
        env_ids: torch.Tensor,
    ) -> tuple[TensorDict, dict]:
        self._clips.index_copy_(0, env_ids, clip_indices)
        self._steps.index_fill_(0, env_ids, 0)
        self.reset_events.append((tuple(env_ids.cpu().tolist()), tuple(clip_indices.cpu().tolist())))
        return self._observation(), {}

    def step(self, action: torch.Tensor):
        assert action.shape == (self.num_envs, 69)
        self._steps.add_(1)
        return (
            self._observation(),
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            {},
        )


class _Algorithm:
    def __init__(self, expert: ForwardBackwardExpertBuffer) -> None:
        self.expert = expert
        self.model = torch.nn.Linear(1, 1)
        self.resets: list[tuple[TensorDict, torch.Tensor]] = []

    def process_env_reset(self, observations: TensorDict, reset: torch.Tensor) -> None:
        self.resets.append((observations, reset.clone()))


class _Environment:
    def __init__(self, table: MotionTaskTable, *, fail_reset: bool = False) -> None:
        command = SimpleNamespace(table=table)
        self.unwrapped = SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: command if name == "motion" else None),
            evaluation_transaction=lambda _seed: nullcontext(),
        )
        self.num_envs = 2
        self.reset_count = 0
        self.fail_reset = fail_reset

    def reset(self) -> tuple[TensorDict, dict]:
        self.reset_count += 1
        if self.fail_reset:
            raise RuntimeError("reset failed")
        return TensorDict({"state": torch.zeros(2, 1)}, batch_size=[2]), {}


def _evaluation() -> MotionTrackingEvaluation:
    return MotionTrackingEvaluation(
        clip_ids=("clip_b", "clip_a"),
        emd=torch.tensor((3.0, 0.0)),
        obs_state_emd=torch.tensor((2.0, 1.0)),
        source_frame_counts=torch.tensor((4, 3), dtype=torch.int64),
        evaluated_frame_counts=torch.tensor((4, 3), dtype=torch.int64),
        coverage_fraction=torch.ones(2),
        duration_seconds=1.25,
    )


def test_tracking_priority_formula_reorders_stable_ids_and_clamps_emd() -> None:
    priorities = motion_tracking_priorities(_evaluation(), _CLIP_IDS, "cpu")

    torch.testing.assert_close(priorities, torch.tensor((2.0, 16.0)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_matches_released_pot_operator() -> None:
    """Exact GPU assignment must match the released CPU oracle without a runtime dependency."""
    optimal_transport = pytest.importorskip("o" + "t")
    generator = torch.Generator().manual_seed(17)
    device = torch.device("cuda:0")
    for count, width in ((1, 29), (2, 23), (7, 29), (32, 23)):
        observed_cpu = torch.randn(count, width, generator=generator)
        target_cpu = torch.randn(count, width, generator=generator)
        observed_norm = observed_cpu.square().sum(dim=1).reshape(-1, 1)
        target_norm = target_cpu.square().sum(dim=1).reshape(1, -1)
        cost = torch.sqrt(
            torch.clamp(
                observed_norm + target_norm - 2.0 * torch.matmul(observed_cpu, target_cpu.mT),
                min=0.0,
            )
        ).numpy()
        weights = np.ones(count) / count
        expected = optimal_transport.emd2(weights, weights, cost, numItermax=100_000)
        output = torch.empty(1, dtype=torch.float64, device=device)
        workspace = tracking_module._UniformEmdWorkspace(lengths=(count,), device=device, feature_width=width)

        workspace.compute(
            observed_cpu.to(device).unsqueeze(0),
            target_cpu.to(device).unsqueeze(0),
            output,
        )

        assert output.item() == pytest.approx(expected, rel=0.0, abs=2.0e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_matches_frozen_cpu_oracle_for_variable_lengths() -> None:
    """GPU assignment must preserve released exact costs without copying traces to the host."""
    workspace_type = getattr(tracking_module, "_UniformEmdWorkspace")
    device = torch.device("cuda:0")
    observed = torch.tensor(
        (
            ((0.0, 0.0), (2.0, 0.0), (9.0, 9.0), (9.0, 9.0)),
            ((0.0, 1.0), (2.0, 1.0), (4.0, 1.0), (6.0, 1.0)),
        ),
        device=device,
    )
    target = torch.tensor(
        (
            ((3.0, 0.0), (0.0, 1.0), (8.0, 8.0), (8.0, 8.0)),
            ((6.0, 2.0), (0.0, 0.0), (5.0, 1.0), (2.0, 3.0)),
        ),
        device=device,
    )
    output = torch.empty(2, dtype=torch.float64, device=device)

    workspace = workspace_type(lengths=(2, 4), device=device, feature_width=2)
    workspace.compute(observed, target, output)

    torch.testing.assert_close(output.cpu(), torch.tensor((1.0, 1.25), dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_flat_rows_match_dense_variable_length_costs() -> None:
    """Compact observed rows and strided expert-owned targets must preserve exact dense EMD."""
    device = torch.device("cuda:0")
    lengths = (2, 4, 3)
    starts = (0, 2, 6)
    observed_flat = torch.arange(27, dtype=torch.float32, device=device).view(9, 3) * 0.1
    target_storage = torch.zeros(9, 5, dtype=torch.float32, device=device)
    target_storage[:, :3].copy_(observed_flat.flip(0))
    target_flat = target_storage[:, :3]
    dense_observed = torch.zeros(3, 4, 3, device=device)
    dense_target = torch.zeros_like(dense_observed)
    for row, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        dense_observed[row, :length].copy_(observed_flat[start : start + length])
        dense_target[row, :length].copy_(target_flat[start : start + length])
    workspace = tracking_module._UniformEmdWorkspace(lengths=lengths, device=device, feature_width=3)
    dense_output = torch.empty(3, dtype=torch.float64, device=device)
    flat_output = torch.empty_like(dense_output)
    start_tensor = torch.tensor(starts, dtype=torch.int64, device=device)

    workspace.compute(dense_observed, dense_target, dense_output)
    workspace.compute_flat(observed_flat, start_tensor, target_flat, start_tensor, flat_output)

    torch.testing.assert_close(flat_output, dense_output, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_uses_stable_lowest_column_ties() -> None:
    """Equal costs must produce the same lowest-column assignment on every launch."""
    device = torch.device("cuda:0")
    values = torch.zeros(1, 4, 1, device=device)
    workspace = tracking_module._UniformEmdWorkspace(lengths=(4,), device=device, feature_width=1)
    output = torch.empty(1, dtype=torch.float64, device=device)

    workspace.compute(values, values, output)
    bucket = workspace._buckets[0]

    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(
        bucket.matching[0, 1:5].cpu(),
        torch.tensor((1, 2, 3, 4), dtype=torch.int32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_block_preserves_variable_length_ties_across_launches() -> None:
    """Every cooperative clip block must retain deterministic lowest-column ties."""
    device = torch.device("cuda:0")
    values = torch.zeros(3, 64, 2, device=device)
    workspace = tracking_module._UniformEmdWorkspace(lengths=(1, 17, 64), device=device, feature_width=2)
    output = torch.empty(3, dtype=torch.float64, device=device)

    workspace.compute(values, values, output)
    first_scratch = tuple(
        (
            bucket.matching.clone(),
            bucket.potential_rows.clone(),
            bucket.potential_columns.clone(),
        )
        for bucket in workspace._buckets
    )
    workspace.compute(values, values, output)

    torch.testing.assert_close(output, torch.zeros_like(output))
    for bucket, (matching, potential_rows, potential_columns) in zip(
        workspace._buckets,
        first_scratch,
        strict=True,
    ):
        torch.testing.assert_close(bucket.matching, matching)
        torch.testing.assert_close(bucket.potential_rows, potential_rows)
        torch.testing.assert_close(bucket.potential_columns, potential_columns)
        for bucket_row, count in enumerate(bucket.lengths.tolist()):
            torch.testing.assert_close(
                bucket.matching[bucket_row, 1 : count + 1].cpu(),
                torch.arange(1, count + 1, dtype=torch.int32),
            )


def test_gpu_uniform_emd_compute_reuses_preallocated_scratch() -> None:
    """The repeated assignment path must not allocate, synchronize, or repair layouts."""
    constructor = inspect.getsource(tracking_module._UniformEmdWorkspace.__init__)
    source = inspect.getsource(tracking_module._UniformEmdWorkspace.compute)
    flat_source = inspect.getsource(tracking_module._UniformEmdWorkspace.compute_flat)

    for forbidden in (".cpu(", ".tolist(", ".item(", ".contiguous("):
        assert forbidden not in constructor
    for forbidden in ("torch.empty", "torch.zeros", "torch.tensor", ".cpu(", ".tolist(", ".item(", ".contiguous("):
        assert forbidden not in source
        assert forbidden not in flat_source


def test_gpu_uniform_emd_dispatches_scalar_and_cooperative_kernels_by_bucket() -> None:
    """Exact assignment must avoid cooperative barrier overhead on short buckets."""
    compute_source = inspect.getsource(tracking_module._UniformEmdWorkspace.compute)
    scalar_source = inspect.getsource(uniform_emd_module.uniform_assignment_cost_scalar)
    cooperative_source = inspect.getsource(uniform_emd_module.uniform_assignment_cost)
    module_source = inspect.getsource(uniform_emd_module)

    assert uniform_emd_module.UNIFORM_ASSIGNMENT_BLOCK_DIM == 256
    assert tracking_module._UniformEmdWorkspace._SCALAR_FRAME_EXTENT_MAX == 512
    assert "bucket.frame_extent <= self._SCALAR_FRAME_EXTENT_MAX" in compute_source
    assert "wp.launch(" in compute_source
    assert "wp.launch_tiled(" in compute_source
    assert "dim=[bucket_size]" in compute_source
    assert "block_dim=UNIFORM_ASSIGNMENT_BLOCK_DIM" in compute_source
    assert "batch = wp.tid()" in scalar_source
    assert "batch, lane = wp.tid()" in cooperative_source
    assert "@wp.func_native" in module_source
    assert "__syncthreads()" in module_source
    assert "other_value == own_value" in module_source
    assert "other_column < own_column" in module_source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_hybrid_dispatch_is_exact_across_boundary() -> None:
    """Scalar 512-frame and cooperative 513-frame buckets must preserve the same exact solution."""
    device = torch.device("cuda:0")
    max_frames = 513
    values = torch.arange(max_frames, dtype=torch.float32, device=device).view(1, max_frames, 1)
    observed = values.expand(2, -1, -1).contiguous()
    target = observed.clone()
    output = torch.empty(2, dtype=torch.float64, device=device)
    workspace = tracking_module._UniformEmdWorkspace(lengths=(512, 513), device=device, feature_width=1)

    workspace.compute(observed, target, output)

    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_workspace_uses_fixed_power_of_two_buckets() -> None:
    """Immutable clip lengths must determine compact dense buffers once at construction."""
    device = torch.device("cuda:0")

    workspace = tracking_module._UniformEmdWorkspace(lengths=(27, 34, 64, 129, 499), device=device, feature_width=3)

    assert workspace.capacity == 5
    assert workspace.max_frames == 499
    assert tuple(bucket.frame_group_bound for bucket in workspace._buckets) == (32, 64, 256, 512)
    assert tuple(bucket.frame_extent for bucket in workspace._buckets) == (27, 64, 129, 499)
    assert tuple(bucket.row_indices.numel() for bucket in workspace._buckets) == (1, 2, 1, 1)
    assert sum(bucket.cost.numel() for bucket in workspace._buckets) == 274_563
    assert not hasattr(workspace, "cost")
    assert not hasattr(workspace, "observed_square")
    assert not hasattr(workspace, "target_square")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_bucket_compute_reuses_every_device_allocation() -> None:
    """Repeated bucket calls must not grow current or peak device allocation after warmup."""
    device = torch.device("cuda:0")
    workspace = tracking_module._UniformEmdWorkspace(lengths=(27, 34, 64, 129), device=device, feature_width=3)
    observed = torch.randn(4, 129, 3, device=device)
    target = torch.randn_like(observed)
    output = torch.empty(4, dtype=torch.float64, device=device)
    workspace.compute(observed, target, output)
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    workspace.compute(observed, target, output)
    workspace.compute(observed, target, output)
    torch.cuda.synchronize(device)

    assert torch.cuda.memory_allocated(device) == baseline
    assert torch.cuda.max_memory_allocated(device) == baseline


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_flat_compute_reuses_every_device_allocation() -> None:
    """Repeated compact-row transport must reuse all bucket and gather storage."""
    device = torch.device("cuda:0")
    lengths = (27, 34, 64, 129)
    starts = (0, 27, 61, 125)
    workspace = tracking_module._UniformEmdWorkspace(lengths=lengths, device=device, feature_width=3)
    observed = torch.randn(sum(lengths), 3, device=device)
    target_storage = torch.randn(sum(lengths), 5, device=device)
    target = target_storage[:, :3]
    start_tensor = torch.tensor(starts, dtype=torch.int64, device=device)
    output = torch.empty(4, dtype=torch.float64, device=device)
    workspace.compute_flat(observed, start_tensor, target, start_tensor, output)
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    workspace.compute_flat(observed, start_tensor, target, start_tensor, output)
    workspace.compute_flat(observed, start_tensor, target, start_tensor, output)
    torch.cuda.synchronize(device)

    assert torch.cuda.memory_allocated(device) == baseline
    assert torch.cuda.max_memory_allocated(device) == baseline


def test_motion_tracking_refill_queues_are_stable_and_cover_every_clip_once() -> None:
    """Longest-work-first queues must use stable ids and assign every clip exactly once."""
    queues = tracking_module._motion_tracking_refill_queues(
        ("z", "a", "b", "c"),
        (5, 5, 4, 2),
        lane_count=2,
    )

    assert queues == ((1, 2), (0, 3))
    assert sorted(clip for queue in queues for clip in queue) == [0, 1, 2, 3]
    assert tuple(sum((5, 5, 4, 2)[clip] for clip in queue) for queue in queues) == (9, 7)

    balanced = tracking_module._motion_tracking_refill_queues(
        ("a", "b", "c", "d"),
        (5, 4, 3, 2),
        lane_count=2,
    )
    balanced_work = tuple(sum((5, 4, 3, 2)[clip] for clip in queue) for queue in balanced)
    assert balanced_work == (7, 7)
    assert max(balanced_work) == max(5, math.ceil(sum((5, 4, 3, 2)) / 2))


def test_motion_tracking_refill_lane_count_derives_smpl_theoretical_lower_bound() -> None:
    """The frozen SMPL aggregate requires 25 lanes to approach its longest-clip bound."""
    action_counts = (3663, 639, *((466,) * 180))

    assert len(action_counts) == 182
    assert sum(action_counts) == 88_182
    assert tracking_module.motion_tracking_refill_lane_count(action_counts) == 25


def test_smpl_packed_evaluator_has_explicit_additive_boundary() -> None:
    """Packed checkpoint evaluation must not overload native curriculum semantics."""
    evaluator = tracking_module.smpl_motion_tracking_evaluator_packed

    assert tuple(inspect.signature(evaluator).parameters) == (
        "model",
        "env",
        "expert",
        "clip_ids",
        "policy_count",
    )
    assert "motion_tracking_refill_lane_count" in tracking_module.__all__
    assert "smpl_motion_tracking_evaluator_packed" in tracking_module.__all__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production evaluator.")
def test_smpl_packed_evaluator_refills_once_and_batches_exact_emd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checkpoint-major policies must share one rollout and one compact exact-EMD call."""
    device = torch.device("cuda:0")
    policy_count = 2
    lanes_per_policy = 2
    expert, table = _smpl_tracking_expert_table(device)
    model = _PackedTrackingModel(policy_count)
    env = _PackedTrackingEnvironment(
        expert,
        table,
        policy_count=policy_count,
        lanes_per_policy=lanes_per_policy,
    )
    compute_receipts: list[tuple[torch.Size, torch.Size, int]] = []
    original_compute_flat = tracking_module._UniformEmdWorkspace.compute_flat

    def compute_flat(workspace, observed, observed_starts, target, target_starts, output):
        compute_receipts.append((observed.shape, target.shape, output.shape[0]))
        return original_compute_flat(workspace, observed, observed_starts, target, target_starts, output)

    synchronize_calls = 0
    original_synchronize = torch.cuda.synchronize

    def synchronize(device=None):
        nonlocal synchronize_calls
        synchronize_calls += 1
        original_synchronize(device)

    monkeypatch.setattr(tracking_module._UniformEmdWorkspace, "compute_flat", compute_flat)
    monkeypatch.setattr(tracking_module.torch.cuda, "synchronize", synchronize)

    evaluations = tracking_module.smpl_motion_tracking_evaluator_packed(
        model,
        env,
        expert,
        _SMPL_CLIP_IDS,
        policy_count=policy_count,
    )

    assert env.reset_events == [
        ((0, 1, 2, 3), (0, 1, 0, 1)),
        ((1, 3), (2, 2)),
        ((0, 2), (3, 3)),
    ]
    assert model.action_calls == 7
    assert synchronize_calls == 1
    assert compute_receipts == [(torch.Size((28, 214)), torch.Size((18, 214)), 8)]
    assert len(evaluations) == policy_count
    assert all(evaluation.clip_ids == _SMPL_CLIP_IDS for evaluation in evaluations)

    metric_lengths = tuple(length - 1 for length in _SMPL_LENGTHS)
    max_length = max(metric_lengths)
    observed = torch.zeros(policy_count * len(_SMPL_CLIP_IDS), max_length, 214, device=device)
    target = torch.zeros_like(observed)
    for policy in range(policy_count):
        for clip, (start, end, length) in enumerate(zip((0, 6, 11, 15), (6, 11, 15, 18), metric_lengths, strict=True)):
            row = policy * len(_SMPL_CLIP_IDS) + clip
            clip_target = expert.frames[start + 1 : end, :214]
            target[row, :length].copy_(clip_target)
            offset = torch.tensor(policy, dtype=torch.float32, device=device) * 0.25
            offset.add_(torch.tensor(clip, dtype=torch.float32, device=device), alpha=0.05)
            observed[row, :length].copy_(clip_target).add_(offset)
    expected = torch.empty(policy_count * len(_SMPL_CLIP_IDS), dtype=torch.float64, device=device)
    workspace = tracking_module._UniformEmdWorkspace(
        lengths=metric_lengths * policy_count,
        device=device,
        feature_width=214,
    )
    workspace.compute(observed, target, expected)

    actual = torch.stack(tuple(evaluation.emd for evaluation in evaluations))
    torch.testing.assert_close(actual, expected.view(policy_count, len(_SMPL_CLIP_IDS)), rtol=0.0, atol=0.0)


def test_smpl_packed_evaluator_keeps_compact_trace_and_single_transport_call() -> None:
    """Packed rollout storage must scale with valid rows rather than clip padding."""
    source = inspect.getsource(tracking_module.smpl_motion_tracking_evaluator_packed)

    assert "trace = torch.empty(policy_count * metric_rows_per_policy, 214" in source
    assert source.count("workspace.compute_flat(") == 1
    assert "target_traces" not in source
    assert ".cpu(" not in source
    assert ".tolist(" not in source
    assert ".item(" not in source


def test_native_tracking_assignment_shuffles_fixed_domain_randomization_rows() -> None:
    """The curriculum evaluator must retain released shuffled replica assignment."""
    python_state = random.getstate()
    random.seed(17)
    try:
        assigned, representatives = tracking_module._native_tracking_assignment(8, 3, torch.device("cpu"))
    finally:
        random.setstate(python_state)

    torch.testing.assert_close(assigned, torch.tensor((0, 1, 0, 1, 2, 2, 1, 0)))
    torch.testing.assert_close(representatives, torch.tensor((0, 1, 4)))


def test_tracking_context_table_matches_released_clip_safe_future_mean() -> None:
    """Rolling eight-style contexts must shorten at each clip tail without crossing clips."""

    class Schema:
        field_widths = (("policy", 3),)

        @staticmethod
        def route(name: str) -> tuple[str, ...]:
            assert name == "backward"
            return ("policy",)

    class Model:
        context_dim = 3
        context_normalization = False
        observation_schema = Schema()

        @staticmethod
        def backward_map(observations: TensorDict) -> torch.Tensor:
            return observations["policy"]

    frames = torch.arange(18, dtype=torch.float32).view(6, 3)
    schema = ForwardBackwardExpertSchema(
        dataset_id="context-test",
        data_hash="data",
        feature_schema_hash="features",
        clip_offsets_hash="offsets",
        expert_feature_width=3,
        num_frames=6,
        num_clips=2,
        window_lengths=(1,),
    )
    expert = ForwardBackwardExpertBuffer(
        frames,
        torch.tensor((0, 3, 6), dtype=torch.int64),
        torch.ones(2),
        schema,
        clip_ids=_CLIP_IDS,
    )

    contexts = tracking_module._tracking_context_table(
        Model(),
        expert,
        (3, 3),
        window_length=2,
        inference_batch_size=2,
    )

    torch.testing.assert_close(contexts[1], frames[1:3].mean(dim=0))
    torch.testing.assert_close(contexts[2], frames[2])
    torch.testing.assert_close(contexts[4], frames[4:6].mean(dim=0))
    torch.testing.assert_close(contexts[5], frames[5])


def test_tracking_evaluation_omits_absent_profile_diagnostics() -> None:
    """A profile with only native EMD must not fabricate a G1 obs-state diagnostic."""
    evaluation = MotionTrackingEvaluation(
        clip_ids=("clip",),
        emd=torch.tensor((1.25,), dtype=torch.float64),
        obs_state_emd=None,
        source_frame_counts=torch.tensor((3,), dtype=torch.int64),
        evaluated_frame_counts=torch.tensor((3,), dtype=torch.int64),
        coverage_fraction=torch.ones(1, dtype=torch.float64),
        duration_seconds=0.5,
    )

    assert evaluation.serializable_metrics() == {
        "clip": {
            "emd": 1.25,
            "num_frames": 3,
            "source_num_frames": 3,
            "evaluated_num_frames": 3,
            "coverage_fraction": 1.0,
        }
    }


def test_smpl_tracking_accepts_exact_t_minus_one_episode_horizon() -> None:
    """Reset frame zero is not a metric row, so T source frames require exactly T-1 actions."""
    metric, evaluated = tracking_module._tracking_frame_counts(
        torch.tensor((4,), dtype=torch.int64),
        episode_length=3,
        include_reset_frame=False,
        allow_horizon_truncation=False,
    )

    torch.testing.assert_close(metric, torch.tensor((3,), dtype=torch.int64))
    torch.testing.assert_close(evaluated, metric)
    with pytest.raises(RuntimeError, match="horizon is shorter"):
        tracking_module._tracking_frame_counts(
            torch.tensor((4,), dtype=torch.int64),
            episode_length=2,
            include_reset_frame=False,
            allow_horizon_truncation=False,
        )


def test_tracking_final_frame_uses_pre_reset_observation() -> None:
    """Same-step autoreset must not replace a valid final metric row with the new spawn."""
    metric = tracking_module._TrackingMetricProjection(
        torch.empty(0, 2),
        lambda observations: observations["policy"],
    )
    reached = torch.empty(2, 2)
    current = TensorDict({"policy": torch.tensor(((1.0, 1.0), (2.0, 2.0)))}, batch_size=[2])
    final = TensorDict({"policy": torch.tensor(((3.0, 3.0), (4.0, 4.0)))}, batch_size=[2])

    tracking_module._write_tracking_reached_values(
        reached,
        metric,
        current,
        {"final_obs": final, "final_obs_valid": torch.tensor((False, True))},
        2,
    )

    torch.testing.assert_close(reached, torch.tensor(((1.0, 1.0), (4.0, 4.0))))


def test_tracking_done_is_allowed_only_at_the_final_reached_frame() -> None:
    """A T=4 SMPL clip reaches its final metric row at zero-based action step two."""
    counts = torch.tensor((3,), dtype=torch.int64)

    assert tracking_module._tracking_done_is_premature(0, counts, include_reset_frame=False).item()
    assert tracking_module._tracking_done_is_premature(1, counts, include_reset_frame=False).item()
    assert not tracking_module._tracking_done_is_premature(2, counts, include_reset_frame=False).item()


def test_tracking_evaluator_does_not_copy_rollout_traces_to_host() -> None:
    """The production evaluator must keep rollout traces and EMD calculation on the GPU."""
    source = inspect.getsource(tracking_module._motion_tracking_evaluator)
    context_source = inspect.getsource(tracking_module._tracking_context_table)

    for forbidden in (".cpu(", ".tolist(", ".item("):
        assert forbidden not in source
        assert forbidden not in context_source


def test_tracking_curriculum_retains_native_replicated_chunks() -> None:
    """The shared evaluator must not substitute packed refill for native replicas."""
    source = inspect.getsource(tracking_module._motion_tracking_evaluator)

    assert "_native_tracking_assignment(env.num_envs, chunk_count" in source
    assert "representative_env_rows" in source
    assert "env.reset_motion_clips(assigned_source)" in source
    assert "env_ids=" not in source
    assert "lengths=evaluated_length_values" in source
    assert "device=expert.device" in source
    assert "workspace.compute(representative, target, output)" in source


def test_tracking_runtime_does_not_import_transport_dependencies() -> None:
    """POT and SciPy may validate the metric in tests but cannot enter the runtime boundary."""
    imports: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(tracking_module))):
        if isinstance(node, ast.Import):
            imports.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module.partition(".")[0])

    assert imports.isdisjoint({"o" + "t", "scipy"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production evaluator.")
def test_concrete_tracking_evaluator_writes_stable_native_metrics() -> None:
    """One shared evaluator must reset exact clips and retain released full/diagnostic metrics."""
    expert = _g1_tracking_expert("cuda:0")
    model = _TrackingModel()
    env = _TrackingEnvironment(expert)

    evaluation = g1_motion_tracking_evaluator(model, env, expert, _CLIP_IDS)

    assert evaluation.clip_ids == _CLIP_IDS
    assert env.reset_assignments == [((0,), (0,)), ((0,), (1,))]
    assert model.deterministic_calls == [True] * 5
    expected_emd = torch.tensor(
        (0.0, 0.0009765625),
        dtype=torch.float64,
        device=expert.device,
    )
    torch.testing.assert_close(evaluation.emd, expected_emd)
    assert evaluation.metrics["clip_a"]["num_frames"] == 3
    assert evaluation.metrics["clip_b"]["num_frames"] == 4
    assert evaluation.metrics["clip_a"]["coverage_fraction"] == 1.0
    assert evaluation.metrics["clip_b"]["coverage_fraction"] == 1.0
    assert evaluation.metrics["clip_a"]["emd"] == 0.0
    torch.testing.assert_close(
        evaluation.metrics["clip_b"]["obs_state_emd"],
        torch.tensor(0.0006905339541845024, dtype=torch.float64, device=expert.device),
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production evaluator.")
def test_native_tracking_evaluator_retains_more_envs_than_clips() -> None:
    """The 1024-env G1 curriculum topology requires shuffled clip replicas, not packed refill."""
    expert = _g1_tracking_expert("cuda:0")
    model = _TrackingModel()
    env = _TrackingEnvironment(expert, num_envs=3)
    python_state = random.getstate()
    random.seed(17)
    try:
        evaluation = g1_motion_tracking_evaluator(model, env, expert, _CLIP_IDS)
    finally:
        random.setstate(python_state)

    assert evaluation.clip_ids == _CLIP_IDS
    assert len(env.reset_assignments) == 1
    assert env.reset_assignments[0][0] == (0, 1, 2)
    assert set(env.reset_assignments[0][1]) == {0, 1}
    assert model.deterministic_calls == [True] * 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production evaluator.")
def test_tracking_caps_long_clips_before_the_native_timeout_edge() -> None:
    """Cross-composed clips must report prefix coverage instead of resetting mid-metric."""
    expert = _g1_tracking_expert("cuda:0")
    model = _TrackingModel()
    env = _TrackingEnvironment(expert)
    env.max_episode_length = 3

    evaluation = g1_motion_tracking_evaluator(model, env, expert, _CLIP_IDS)

    assert evaluation.metrics["clip_a"]["source_num_frames"] == 3
    assert evaluation.metrics["clip_a"]["evaluated_num_frames"] == 3
    assert evaluation.metrics["clip_a"]["coverage_fraction"] == 1.0
    assert evaluation.metrics["clip_b"]["source_num_frames"] == 4
    assert evaluation.metrics["clip_b"]["evaluated_num_frames"] == 3
    assert evaluation.metrics["clip_b"]["coverage_fraction"] == 0.75


def test_curriculum_validates_updates_resets_and_records_atomically(tmp_path) -> None:
    table = _table()
    expert = _expert()
    algorithm = _Algorithm(expert)
    env = _Environment(table)
    coordinator = MotionTrackingCurriculum(
        env,
        algorithm,
        str(tmp_path),
        "cpu",
        evaluator=lambda *_args: _evaluation(),
    )

    observations = coordinator.on_transition(8)

    torch.testing.assert_close(table.clip_priorities, torch.tensor((2.0, 16.0)))
    torch.testing.assert_close(expert.priorities, torch.tensor((2.0, 16.0)))
    assert env.reset_count == 1
    assert observations is algorithm.resets[0][0]
    assert torch.all(algorithm.resets[0][1])
    record = json.loads((tmp_path / "tracking_curriculum" / "8.json").read_text())
    assert record["schema"] == "motion_tracking_curriculum_v1"
    assert record["clip_ids"] == list(_CLIP_IDS)
    assert record["priorities"] == [2.0, 16.0]
    assert record["duration_seconds"] == 1.25
    assert record["metrics"]["clip_a"]["emd"] == 0.0
    assert record["metrics"]["clip_b"]["source_num_frames"] == 4


def test_curriculum_validates_both_targets_before_first_priority_mutation(tmp_path, monkeypatch) -> None:
    table = _table()
    expert = _expert()
    algorithm = _Algorithm(expert)
    coordinator = MotionTrackingCurriculum(
        _Environment(table),
        algorithm,
        str(tmp_path),
        "cpu",
        evaluator=lambda *_args: _evaluation(),
    )
    original = table.clip_priorities.clone()

    def reject(_priorities: torch.Tensor) -> None:
        raise ValueError("expert rejected priorities")

    monkeypatch.setattr(expert, "validate_priorities", reject)
    with pytest.raises(ValueError, match="expert rejected"):
        coordinator.on_transition(8)

    torch.testing.assert_close(table.clip_priorities, original)
    assert not (tmp_path / "tracking_curriculum" / "8.json").exists()


def test_curriculum_reset_failure_rolls_back_both_samplers_and_record(tmp_path) -> None:
    """A failed reset must restore exact pre-event sampling state and publish nothing."""
    table = _table()
    expert = _expert()
    algorithm = _Algorithm(expert)
    env = _Environment(table, fail_reset=True)
    coordinator = MotionTrackingCurriculum(
        env,
        algorithm,
        str(tmp_path),
        "cpu",
        evaluator=lambda *_args: _evaluation(),
    )

    with pytest.raises(RuntimeError, match="reset failed"):
        coordinator.on_transition(8)

    torch.testing.assert_close(table.clip_priorities, torch.ones(2))
    torch.testing.assert_close(expert.priorities, torch.ones(2))
    assert env.reset_count == 1
    assert not (tmp_path / "tracking_curriculum" / "8.json").exists()
    assert not (tmp_path / "tracking_curriculum" / "8.tmp").exists()


def test_curriculum_state_restores_both_priority_targets_and_event_history(tmp_path) -> None:
    table = _table()
    expert = _expert()
    coordinator = MotionTrackingCurriculum(
        _Environment(table),
        _Algorithm(expert),
        str(tmp_path),
        "cpu",
        evaluator=lambda *_args: _evaluation(),
    )
    coordinator.on_transition(8)
    state = coordinator.state_dict()
    assert "table_priorities_active" not in state
    table.clip_priorities.fill_(1.0)
    expert.set_priorities(torch.ones(2))

    coordinator.load_state_dict(state)

    torch.testing.assert_close(table.clip_priorities, torch.tensor((2.0, 16.0)))
    torch.testing.assert_close(expert.priorities, torch.tensor((2.0, 16.0)))
    assert coordinator.applied_transitions == [8]


def test_curriculum_state_before_first_event_restores_uniform_sampler(tmp_path) -> None:
    """A transition-zero-pending checkpoint must retain equal clip mass."""
    table = _table()
    expert = _expert()
    coordinator = MotionTrackingCurriculum(
        _Environment(table),
        _Algorithm(expert),
        str(tmp_path),
        "cpu",
        evaluator=lambda *_args: _evaluation(),
    )
    state = coordinator.state_dict()
    table.set_clip_priorities(_CLIP_IDS, torch.tensor((2.0, 16.0)))
    expert.set_priorities(torch.tensor((2.0, 16.0)))
    coordinator.applied_transitions.append(8)

    coordinator.load_state_dict(state)

    torch.testing.assert_close(table.clip_priorities, torch.ones(2))
    torch.testing.assert_close(expert.priorities, torch.ones(2))
    assert coordinator.applied_transitions == []


def _motion_table_draws(table: MotionTaskTable) -> dict[str, torch.Tensor]:
    """Draw the four stochastic reset decisions owned by one motion task table."""
    count = 8
    return {
        "row": table.sample_rows(count),
        "source": table.sample_reset_sources(count),
        "time": torch.rand(count, generator=table.generator),
        "fall": torch.randint(8192, (count,), generator=table.generator),
    }


def _assert_motion_table_draws_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


def _run_motion_evaluation_transaction(prior_table_draws: int) -> dict[str, torch.Tensor]:
    table = _table()
    table.generator.manual_seed(11)
    torch.rand(prior_table_draws, generator=table.generator)
    training_state = table.generator.get_state().clone()
    expected_generator = torch.Generator()
    expected_generator.set_state(training_state)
    expected_table = _table()
    object.__setattr__(expected_table, "generator", expected_generator)
    expected_training_draws = _motion_table_draws(expected_table)

    term = SimpleNamespace(
        table=table,
        time_left=torch.tensor((1.0, 2.0)),
        command_counter=torch.tensor((3, 4), dtype=torch.int64),
    )
    push = object.__new__(MotionPushVelocity)
    push._elapsed_seconds = torch.tensor((11, 22), dtype=torch.int32)
    push._interval_seconds = torch.tensor((1, 2), dtype=torch.int32)
    startup_material = object.__new__(randomize_rigid_body_material)
    penalty = object.__new__(MotionPenaltyScaleCurriculum)
    penalty._scale = 0.25
    penalty._average_episode_length = 41.0
    event_manager = SimpleNamespace(
        active_terms={"startup": ["robot_material"], "interval": ["push"]},
        get_term_cfg=lambda name: SimpleNamespace(func={"robot_material": startup_material, "push": push}[name]),
        _interval_term_time_left=[torch.tensor((0.25, 0.75))],
        _reset_term_last_triggered_step_id=[torch.tensor((7, 9), dtype=torch.int32)],
        _reset_term_last_triggered_once=[torch.tensor((True, False))],
    )
    curriculum_manager = SimpleNamespace(
        active_terms=("penalty_scale",),
        get_term=lambda name: penalty if name == "penalty_scale" else None,
        _curriculum_state={
            "penalty_scale": {
                "penalty_scale": 0.25,
                "average_episode_length": 41.0,
            }
        },
    )
    env = object.__new__(MotionImitationEnv)
    env._is_closed = True
    env.common_step_counter = 17
    env._sim_step_counter = 255
    env.episode_length_buf = torch.tensor((5, 6), dtype=torch.int64)
    env.command_manager = SimpleNamespace(
        active_terms=("motion",),
        get_term=lambda name: term if name == "motion" else None,
    )
    env.event_manager = event_manager
    env.curriculum_manager = curriculum_manager
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    expected_global = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    with MotionImitationEnv.evaluation_transaction(env, 99):
        evaluation_draws = _motion_table_draws(table)
        random.random()
        np.random.rand()
        torch.rand(1)
        env.common_step_counter = 100
        env._sim_step_counter = 1500
        env.episode_length_buf.zero_()
        term.time_left.zero_()
        term.command_counter.zero_()
        env.event_manager._interval_term_time_left[0].zero_()
        env.event_manager._reset_term_last_triggered_step_id[0].zero_()
        env.event_manager._reset_term_last_triggered_once[0].logical_not_()
        env.curriculum_manager._curriculum_state["penalty_scale"]["penalty_scale"] = 0.75
        push._elapsed_seconds.zero_()
        push._interval_seconds.fill_(3)
        penalty._scale = 0.75
        penalty._average_episode_length = 120.0

    actual_global = (random.random(), np.random.rand(), torch.rand(1))
    assert actual_global[0] == expected_global[0]
    assert actual_global[1] == expected_global[1]
    torch.testing.assert_close(actual_global[2], expected_global[2])
    _assert_motion_table_draws_equal(_motion_table_draws(table), expected_training_draws)
    assert env.common_step_counter == 17
    assert env._sim_step_counter == 255
    torch.testing.assert_close(env.episode_length_buf, torch.tensor((5, 6)))
    torch.testing.assert_close(term.time_left, torch.tensor((1.0, 2.0)))
    torch.testing.assert_close(term.command_counter, torch.tensor((3, 4)))
    torch.testing.assert_close(env.event_manager._interval_term_time_left[0], torch.tensor((0.25, 0.75)))
    torch.testing.assert_close(
        env.event_manager._reset_term_last_triggered_step_id[0],
        torch.tensor((7, 9), dtype=torch.int32),
    )
    torch.testing.assert_close(
        env.event_manager._reset_term_last_triggered_once[0],
        torch.tensor((True, False)),
    )
    assert env.curriculum_manager._curriculum_state == {
        "penalty_scale": {
            "penalty_scale": 0.25,
            "average_episode_length": 41.0,
        }
    }
    torch.testing.assert_close(push._elapsed_seconds, torch.tensor((11, 22), dtype=torch.int32))
    torch.testing.assert_close(push._interval_seconds, torch.tensor((1, 2), dtype=torch.int32))
    assert penalty.scale == 0.25
    assert penalty.average_episode_length == 41.0
    return evaluation_draws


def test_motion_evaluation_transaction_isolates_rng_clocks_and_stateful_terms() -> None:
    """Evaluator draws are seed-only and every persistent training clock resumes exactly."""
    first = _run_motion_evaluation_transaction(prior_table_draws=0)
    second = _run_motion_evaluation_transaction(prior_table_draws=37)
    _assert_motion_table_draws_equal(first, second)
