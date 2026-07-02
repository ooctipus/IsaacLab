# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable-ID tracking evaluation and curriculum transactions."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import warp as wp
from rsl_rl.modules.forward_backward import trajectory_context_sequence
from rsl_rl.runners import RunnerLifecycleExtension
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict, TensorDictBase

from .frames import G1_HEAD_FRAME_NAME
from .impl.uniform_emd_warp import (
    UNIFORM_ASSIGNMENT_BLOCK_DIM,
    uniform_assignment_cost,
    uniform_assignment_cost_scalar,
    uniform_assignment_prepare_bucket,
    uniform_assignment_prepare_flat_bucket,
)
from .mdp.actions import MotionJointPositionAction, MotionMujocoControlAction
from .mdp.commands import MotionStatePayload, MotionTaskTable


@dataclass(frozen=True, slots=True)
class MotionTrackingEvaluation:
    """Per-clip tracking results emitted by an isolated evaluator.

    The evaluator owns simulator rollout and EMD calculation. For native G1
    parity, :attr:`emd` is the released unprefixed curriculum metric: optimal
    transport between observed and reference full joint-position sequences
    shaped ``[time, 29]``. The complete metrics also retain the released
    ``obs_state_emd`` diagnostic over ``state[..., :23]``; that diagnostic does
    not substitute for the curriculum metric.

    """

    clip_ids: tuple[str, ...]
    emd: torch.Tensor
    obs_state_emd: torch.Tensor | None
    source_frame_counts: torch.Tensor
    evaluated_frame_counts: torch.Tensor
    coverage_fraction: torch.Tensor
    duration_seconds: float

    def __post_init__(self) -> None:
        """Reject incomplete, duplicate, or nonfinite evaluation results."""
        if len(self.clip_ids) == 0 or len(set(self.clip_ids)) != len(self.clip_ids):
            raise ValueError("Tracking evaluation clip_ids must be nonempty and unique.")
        count = len(self.clip_ids)
        diagnostics = () if self.obs_state_emd is None else (self.obs_state_emd,)
        floating = (self.emd, *diagnostics, self.coverage_fraction)
        integer = (self.source_frame_counts, self.evaluated_frame_counts)
        if any(value.shape != (count,) or not value.is_floating_point() for value in floating):
            raise ValueError("Tracking floating metrics must contain one value per clip.")
        if any(value.shape != (count,) or value.dtype is not torch.int64 for value in integer):
            raise ValueError("Tracking frame counts must be int64 with one value per clip.")
        if any(value.device != self.emd.device for value in (*floating, *integer)):
            raise ValueError("Tracking metrics must remain together on one device.")
        if any(value.requires_grad for value in floating):
            raise ValueError("Tracking metrics must be detached.")
        torch._assert_async(
            torch.all(torch.stack(tuple(torch.all(torch.isfinite(value)) for value in floating))),
            "Tracking floating metrics must be finite.",
        )
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0.0:
            raise ValueError("Tracking evaluation duration must be finite and non-negative.")

    @property
    def metrics(self) -> Mapping[str, Mapping[str, object]]:
        """Expose per-clip GPU scalar views without materializing host records."""
        metrics: dict[str, dict[str, object]] = {
            clip_id: {
                "emd": self.emd[index],
                "num_frames": self.evaluated_frame_counts[index],
                "source_num_frames": self.source_frame_counts[index],
                "evaluated_num_frames": self.evaluated_frame_counts[index],
                "coverage_fraction": self.coverage_fraction[index],
            }
            for index, clip_id in enumerate(self.clip_ids)
        }
        if self.obs_state_emd is not None:
            for index, clip_id in enumerate(self.clip_ids):
                metrics[clip_id]["obs_state_emd"] = self.obs_state_emd[index]
        return metrics

    def serializable_metrics(self) -> dict[str, dict[str, float | int]]:
        """Copy final scalar rows once at the JSON artifact boundary."""
        columns = [
            self.emd,
            self.source_frame_counts.to(torch.float64),
            self.evaluated_frame_counts.to(torch.float64),
            self.coverage_fraction,
        ]
        if self.obs_state_emd is not None:
            columns.append(self.obs_state_emd)
        rows = torch.stack(columns, dim=-1).cpu()
        metrics: dict[str, dict[str, float | int]] = {
            clip_id: {
                "emd": float(row[0]),
                "num_frames": int(row[2]),
                "source_num_frames": int(row[1]),
                "evaluated_num_frames": int(row[2]),
                "coverage_fraction": float(row[3]),
            }
            for clip_id, row in zip(self.clip_ids, rows, strict=True)
        }
        if self.obs_state_emd is not None:
            for clip_id, row in zip(self.clip_ids, rows, strict=True):
                metrics[clip_id]["obs_state_emd"] = float(row[4])
        return metrics


class _UniformEmdWorkspace:
    """Power-of-two-grouped dense GPU buckets for exact variable-length assignment."""

    _MINIMUM_FRAME_CAPACITY = 32
    _SCALAR_FRAME_EXTENT_MAX = 2 * UNIFORM_ASSIGNMENT_BLOCK_DIM
    _MAXIMUM_BUCKET_COUNT = 8

    @dataclass(slots=True)
    class Bucket:
        """Fixed tensors for one power-of-two length group."""

        frame_group_bound: int
        frame_extent: int
        row_indices: torch.Tensor
        lengths: torch.Tensor
        observed: torch.Tensor
        target: torch.Tensor
        observed_norm: torch.Tensor
        target_norm: torch.Tensor
        cost: torch.Tensor
        potential_rows: torch.Tensor
        potential_columns: torch.Tensor
        matching: torch.Tensor
        previous: torch.Tensor
        minimum: torch.Tensor
        used: torch.Tensor
        output: torch.Tensor

    def __init__(
        self,
        lengths: tuple[int, ...],
        device: torch.device,
        feature_width: int = 29,
    ) -> None:
        if (
            not isinstance(lengths, tuple)
            or not lengths
            or any(type(value) is not int or value < 1 for value in lengths)
            or not isinstance(device, torch.device)
            or device.type != "cuda"
            or feature_width < 1
        ):
            raise ValueError("GPU EMD requires immutable positive frame lengths, a CUDA device, and a feature width.")

        grouped_rows: dict[int, list[int]] = {}
        for row, length in enumerate(lengths):
            frame_group_bound = max(self._MINIMUM_FRAME_CAPACITY, 1 << (length - 1).bit_length())
            grouped_rows.setdefault(frame_group_bound, []).append(row)
        if len(grouped_rows) > self._MAXIMUM_BUCKET_COUNT:
            raise ValueError("GPU EMD supports at most eight nonempty power-of-two frame buckets.")

        wp.init()
        self.capacity = len(lengths)
        self.max_frames = max(lengths)
        self.device = device
        self.feature_width = feature_width
        buckets = []
        for frame_group_bound, rows in sorted(grouped_rows.items()):
            frame_extent = max(lengths[row] for row in rows)
            row_indices = torch.tensor(rows, dtype=torch.int64, device=self.device)
            bucket_lengths = torch.tensor(
                [lengths[row] for row in rows],
                dtype=torch.int64,
                device=self.device,
            )
            bucket_size = len(rows)
            observed = torch.empty(
                bucket_size,
                frame_extent,
                feature_width,
                dtype=torch.float32,
                device=self.device,
            )
            target = torch.empty_like(observed)
            observed_norm = torch.empty(
                bucket_size,
                frame_extent,
                1,
                dtype=torch.float32,
                device=self.device,
            )
            target_norm = torch.empty_like(observed_norm)
            cost = torch.empty(
                bucket_size,
                frame_extent,
                frame_extent,
                dtype=torch.float32,
                device=self.device,
            )
            scratch_shape = (bucket_size, frame_extent + 1)
            potential_rows = torch.empty(scratch_shape, dtype=torch.float64, device=self.device)
            potential_columns = torch.empty_like(potential_rows)
            matching = torch.empty(scratch_shape, dtype=torch.int32, device=self.device)
            previous = torch.empty_like(matching)
            minimum = torch.empty_like(potential_rows)
            used = torch.empty_like(matching)
            buckets.append(
                self.Bucket(
                    frame_group_bound=frame_group_bound,
                    frame_extent=frame_extent,
                    row_indices=row_indices,
                    lengths=bucket_lengths,
                    observed=observed,
                    target=target,
                    observed_norm=observed_norm,
                    target_norm=target_norm,
                    cost=cost,
                    potential_rows=potential_rows,
                    potential_columns=potential_columns,
                    matching=matching,
                    previous=previous,
                    minimum=minimum,
                    used=used,
                    output=torch.empty(bucket_size, dtype=torch.float64, device=self.device),
                )
            )
        self._buckets = tuple(buckets)

    def compute(
        self,
        observed: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Write exact transport costs through the immutable fixed bucket layout."""
        batch_size = observed.shape[0]
        if (
            observed.ndim != 3
            or target.shape != observed.shape
            or observed.shape[:2] != (self.capacity, self.max_frames)
            or observed.dtype is not torch.float32
            or target.dtype is not torch.float32
            or observed.device != self.device
            or target.device != self.device
            or not observed.is_contiguous()
            or not target.is_contiguous()
            or output.shape != (self.capacity,)
            or output.dtype is not torch.float64
            or output.device != self.device
            or observed.shape[2] < 1
            or observed.shape[2] > self.feature_width
            or batch_size != self.capacity
        ):
            raise ValueError("GPU EMD inputs do not match the fixed workspace contract.")
        feature_width = observed.shape[2]
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        for bucket in self._buckets:
            bucket_size = bucket.row_indices.shape[0]
            wp.launch(
                uniform_assignment_prepare_bucket,
                dim=(bucket_size, bucket.frame_extent),
                inputs=[
                    wp.from_torch(observed),
                    wp.from_torch(target),
                    wp.from_torch(bucket.row_indices),
                    self.max_frames,
                    feature_width,
                    self.feature_width,
                    wp.from_torch(bucket.observed),
                    wp.from_torch(bucket.target),
                ],
                stream=stream,
            )
            torch.bmm(bucket.observed, bucket.target.mT, out=bucket.cost)
            torch.square(bucket.observed, out=bucket.observed)
            torch.sum(bucket.observed, dim=-1, keepdim=True, out=bucket.observed_norm)
            torch.square(bucket.target, out=bucket.target)
            torch.sum(bucket.target, dim=-1, keepdim=True, out=bucket.target_norm)
            bucket.cost.mul_(-2.0)
            bucket.cost.add_(bucket.observed_norm)
            bucket.cost.add_(bucket.target_norm.mT)
            bucket.cost.clamp_min_(0.0).sqrt_()
            assignment_inputs = [
                wp.from_torch(bucket.cost),
                wp.from_torch(bucket.lengths),
                wp.from_torch(bucket.potential_rows),
                wp.from_torch(bucket.potential_columns),
                wp.from_torch(bucket.matching),
                wp.from_torch(bucket.previous),
                wp.from_torch(bucket.minimum),
                wp.from_torch(bucket.used),
                wp.from_torch(bucket.output),
            ]
            if bucket.frame_extent <= self._SCALAR_FRAME_EXTENT_MAX:
                wp.launch(
                    uniform_assignment_cost_scalar,
                    dim=bucket_size,
                    inputs=assignment_inputs,
                    stream=stream,
                )
            else:
                wp.launch_tiled(
                    uniform_assignment_cost,
                    dim=[bucket_size],
                    block_dim=UNIFORM_ASSIGNMENT_BLOCK_DIM,
                    inputs=assignment_inputs,
                    stream=stream,
                )
            output.index_copy_(0, bucket.row_indices, bucket.output)

    def compute_flat(
        self,
        observed: torch.Tensor,
        observed_starts: torch.Tensor,
        target: torch.Tensor,
        target_starts: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Write exact costs from compact flat clip rows without dense trace padding."""
        if (
            observed.ndim != 2
            or target.ndim != 2
            or observed.shape[1] != target.shape[1]
            or observed.dtype is not torch.float32
            or target.dtype is not torch.float32
            or observed.device != self.device
            or target.device != self.device
            or not observed.is_contiguous()
            or observed.shape[1] < 1
            or observed.shape[1] > self.feature_width
            or observed_starts.shape != (self.capacity,)
            or target_starts.shape != (self.capacity,)
            or observed_starts.dtype is not torch.int64
            or target_starts.dtype is not torch.int64
            or observed_starts.device != self.device
            or target_starts.device != self.device
            or not observed_starts.is_contiguous()
            or not target_starts.is_contiguous()
            or output.shape != (self.capacity,)
            or output.dtype is not torch.float64
            or output.device != self.device
        ):
            raise ValueError("Flat GPU EMD inputs do not match the fixed workspace contract.")
        feature_width = observed.shape[1]
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        for bucket in self._buckets:
            bucket_size = bucket.row_indices.shape[0]
            wp.launch(
                uniform_assignment_prepare_flat_bucket,
                dim=(bucket_size, bucket.frame_extent),
                inputs=[
                    wp.from_torch(observed),
                    wp.from_torch(target),
                    wp.from_torch(observed_starts),
                    wp.from_torch(target_starts),
                    wp.from_torch(bucket.row_indices),
                    wp.from_torch(bucket.lengths),
                    feature_width,
                    self.feature_width,
                    wp.from_torch(bucket.observed),
                    wp.from_torch(bucket.target),
                ],
                stream=stream,
            )
            torch.bmm(bucket.observed, bucket.target.mT, out=bucket.cost)
            torch.square(bucket.observed, out=bucket.observed)
            torch.sum(bucket.observed, dim=-1, keepdim=True, out=bucket.observed_norm)
            torch.square(bucket.target, out=bucket.target)
            torch.sum(bucket.target, dim=-1, keepdim=True, out=bucket.target_norm)
            bucket.cost.mul_(-2.0)
            bucket.cost.add_(bucket.observed_norm)
            bucket.cost.add_(bucket.target_norm.mT)
            bucket.cost.clamp_min_(0.0).sqrt_()
            assignment_inputs = [
                wp.from_torch(bucket.cost),
                wp.from_torch(bucket.lengths),
                wp.from_torch(bucket.potential_rows),
                wp.from_torch(bucket.potential_columns),
                wp.from_torch(bucket.matching),
                wp.from_torch(bucket.previous),
                wp.from_torch(bucket.minimum),
                wp.from_torch(bucket.used),
                wp.from_torch(bucket.output),
            ]
            if bucket.frame_extent <= self._SCALAR_FRAME_EXTENT_MAX:
                wp.launch(
                    uniform_assignment_cost_scalar,
                    dim=bucket_size,
                    inputs=assignment_inputs,
                    stream=stream,
                )
            else:
                wp.launch_tiled(
                    uniform_assignment_cost,
                    dim=[bucket_size],
                    block_dim=UNIFORM_ASSIGNMENT_BLOCK_DIM,
                    inputs=assignment_inputs,
                    stream=stream,
                )
            output.index_copy_(0, bucket.row_indices, bucket.output)


def _as_tensordict(observations: object, num_envs: int) -> TensorDictBase:
    """Expose environment observation mappings through the model's TensorDict contract."""
    if isinstance(observations, TensorDictBase):
        return observations
    if not isinstance(observations, Mapping):
        raise TypeError("Motion tracking observations must be a TensorDict or tensor mapping.")
    if any(not isinstance(value, torch.Tensor) for value in observations.values()):
        raise TypeError("Motion tracking observation fields must be tensors.")
    return TensorDict(dict(observations), batch_size=[num_envs])


def _expert_frame_tensordict(model: Any, frames: torch.Tensor) -> TensorDictBase:
    """Split prebuilt expert frames into the model's backward route."""
    schema = model.observation_schema
    route = tuple(schema.route("backward"))
    widths = dict(schema.field_widths)
    values: dict[str, torch.Tensor] = {}
    offset = 0
    for name in route:
        width = widths[name]
        end = offset + width
        values[name] = frames[:, offset:end]
        offset = end
    if offset != frames.shape[1]:
        raise ValueError("Expert frame width does not match the model backward route.")
    return TensorDict(values, batch_size=[frames.shape[0]])


def _expert_frame_tensordict_packed(
    model: Any,
    frames: torch.Tensor,
    policy_count: int,
) -> TensorDictBase:
    """Split expert frames into a zero-copy checkpoint-major backward route."""
    schema = model.observation_schema
    route = tuple(schema.route("backward"))
    widths = dict(schema.field_widths)
    values: dict[str, torch.Tensor] = {}
    offset = 0
    for name in route:
        width = widths[name]
        end = offset + width
        values[name] = frames[:, offset:end].unsqueeze(0).expand(policy_count, -1, -1)
        offset = end
    if offset != frames.shape[1]:
        raise ValueError("Expert frame width does not match the packed model backward route.")
    return TensorDict(values, batch_size=[policy_count, frames.shape[0]])


def motion_tracking_refill_lane_count(action_counts: tuple[int, ...]) -> int:
    """Return the fewest lanes whose ideal work bound reaches the longest clip.

    Args:
        action_counts: Positive per-clip policy action counts in stable clip order.

    Returns:
        The clip-count-limited ceiling of total work divided by longest-clip work.
    """
    if (
        not isinstance(action_counts, tuple)
        or not action_counts
        or any(type(count) is not int or count < 1 for count in action_counts)
    ):
        raise ValueError("Tracking refill action counts must be an immutable nonempty tuple of positive integers.")
    longest = max(action_counts)
    return min(len(action_counts), (sum(action_counts) + longest - 1) // longest)


def _motion_tracking_refill_queues(
    clip_ids: tuple[str, ...],
    action_counts: tuple[int, ...],
    *,
    lane_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Distribute stable clip rows through deterministic longest-processing-time queues."""
    if (
        not isinstance(clip_ids, tuple)
        or len(clip_ids) != len(action_counts)
        or len(set(clip_ids)) != len(clip_ids)
        or any(not isinstance(clip_id, str) or not clip_id for clip_id in clip_ids)
    ):
        raise ValueError("Tracking refill clip ids must be a unique nonempty tuple aligned to action counts.")
    motion_tracking_refill_lane_count(action_counts)
    if type(lane_count) is not int or lane_count < 1 or lane_count > len(clip_ids):
        raise ValueError("Tracking refill requires one to one-lane-per-clip execution lanes.")

    queues: list[list[int]] = [[] for _lane in range(lane_count)]
    lane_work = [0] * lane_count
    for clip_index in sorted(range(len(clip_ids)), key=lambda index: (-action_counts[index], clip_ids[index], index)):
        lane = min(range(lane_count), key=lambda index: (lane_work[index], index))
        queues[lane].append(clip_index)
        lane_work[lane] += action_counts[clip_index]
    return tuple(tuple(queue) for queue in queues)


def _native_tracking_assignment(
    num_envs: int,
    num_motions: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce released shuffled motion pairing and first-env representatives."""
    if num_envs < 1 or num_motions < 1 or num_motions > num_envs:
        raise ValueError("Native tracking assignment requires 1 <= num_motions <= num_envs.")
    shuffled_env_indices = list(range(num_envs))
    random.shuffle(shuffled_env_indices)
    assigned_motion_rows = [env_index % num_motions for env_index in shuffled_env_indices]
    representative_env_rows = [assigned_motion_rows.index(motion_row) for motion_row in range(num_motions)]
    return (
        torch.tensor(assigned_motion_rows, dtype=torch.int64, device=device),
        torch.tensor(representative_env_rows, dtype=torch.int64, device=device),
    )


@dataclass(frozen=True, slots=True)
class _TrackingMetricProjection:
    """One profile-owned projection consumed by the shared rollout and EMD math."""

    target_frames: torch.Tensor
    observe: Callable[[TensorDictBase], torch.Tensor]


def _tracking_context_table(
    model: Any,
    expert: ForwardBackwardExpertBuffer,
    clip_lengths: tuple[int, ...],
    *,
    window_length: int,
    inference_batch_size: int = 8192,
) -> torch.Tensor:
    """Materialize clip-safe rolling contexts once from immutable expert frames."""
    if (
        len(clip_lengths) != len(expert.clip_ids)
        or any(type(length) is not int or length < 2 for length in clip_lengths)
        or window_length < 1
        or inference_batch_size < 1
    ):
        raise ValueError("Tracking contexts require clips of at least two frames and positive inference sizes.")
    expected_lengths = torch.tensor(clip_lengths, dtype=torch.int64, device=expert.device)
    torch._assert_async(
        torch.all(expert.clip_lengths == expected_lengths),
        "Tracking context metadata and expert clip lengths differ.",
    )
    frame_count = expert.frames.shape[0]
    backward = torch.empty(
        frame_count,
        model.context_dim,
        dtype=expert.frames.dtype,
        device=expert.device,
    )
    for start in range(0, frame_count, inference_batch_size):
        end = min(start + inference_batch_size, frame_count)
        observations = _expert_frame_tensordict(model, expert.frames[start:end])
        backward[start:end].copy_(model.backward_map(observations))

    contexts = torch.empty_like(backward)
    offsets = [0]
    for length in clip_lengths:
        offsets.append(offsets[-1] + length)
    radius = model.context_dim**0.5 if model.context_normalization else None
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        if end - start < 2:
            raise ValueError("Tracking clips must contain at least two expert frames.")
        contexts[start + 1 : end].copy_(
            trajectory_context_sequence(
                backward[start + 1 : end],
                window_length,
                include_partial=True,
                radius=radius,
            )
        )
    return contexts


def _tracking_context_table_packed(
    model: Any,
    expert: ForwardBackwardExpertBuffer,
    clip_lengths: tuple[int, ...],
    *,
    policy_count: int,
    window_length: int,
    inference_batch_size: int = 8192,
) -> torch.Tensor:
    """Materialize projected clip-safe contexts for one checkpoint stack."""
    if (
        type(policy_count) is not int
        or policy_count < 1
        or getattr(model, "policy_count", None) != policy_count
        or len(clip_lengths) != len(expert.clip_ids)
        or any(type(length) is not int or length < 2 for length in clip_lengths)
        or window_length < 1
        or inference_batch_size < 1
    ):
        raise ValueError("Packed contexts require aligned policies, clips, and positive inference sizes.")
    expected_lengths = torch.tensor(clip_lengths, dtype=torch.int64, device=expert.device)
    torch._assert_async(
        torch.all(expert.clip_lengths == expected_lengths),
        "Packed context metadata and expert clip lengths differ.",
    )
    frame_count = expert.frames.shape[0]
    backward = torch.empty(
        policy_count,
        frame_count,
        model.context_dim,
        dtype=expert.frames.dtype,
        device=expert.device,
    )
    for start in range(0, frame_count, inference_batch_size):
        end = min(start + inference_batch_size, frame_count)
        observations = _expert_frame_tensordict_packed(model, expert.frames[start:end], policy_count)
        backward[:, start:end].copy_(model.backward_map(observations))

    contexts = torch.empty_like(backward)
    offsets = [0]
    for length in clip_lengths:
        offsets.append(offsets[-1] + length)
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        means = trajectory_context_sequence(
            backward[:, start + 1 : end],
            window_length,
            include_partial=True,
            radius=None,
        )
        contexts[:, start + 1 : end].copy_(model.context_project(means))
    return contexts


def _tracking_frame_counts(
    raw_frame_counts: torch.Tensor,
    *,
    episode_length: int,
    include_reset_frame: bool,
    allow_horizon_truncation: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native metric rows and the uninterrupted rows safe to evaluate."""
    trace_offset = int(include_reset_frame)
    metric_frame_counts = raw_frame_counts if include_reset_frame else raw_frame_counts - 1
    torch._assert_async(torch.all(metric_frame_counts > 0), "Tracking clips contain no metric frames.")
    if allow_horizon_truncation:
        evaluated_frame_counts = metric_frame_counts.clamp(max=episode_length)
    else:
        torch._assert_async(
            torch.all(metric_frame_counts - trace_offset <= episode_length),
            "Tracking evaluation horizon is shorter than an expert clip.",
        )
        evaluated_frame_counts = metric_frame_counts
    return metric_frame_counts, evaluated_frame_counts


def _tracking_done_is_premature(
    step: int,
    evaluated_frame_counts: torch.Tensor,
    *,
    include_reset_frame: bool,
) -> torch.Tensor:
    """Return rows reset before the final metric frame reached by this rollout."""
    required_steps = evaluated_frame_counts - int(include_reset_frame)
    return step < required_steps - 1


def _write_tracking_reached_values(
    destination: torch.Tensor,
    metric: _TrackingMetricProjection,
    observations: TensorDictBase,
    extras: Mapping[str, object],
    num_envs: int,
) -> None:
    """Write reached values, selecting exact pre-reset observations on done rows."""
    destination.copy_(metric.observe(observations))
    if "final_obs" not in extras:
        return
    final_observations = _as_tensordict(extras["final_obs"], num_envs)
    final_values = metric.observe(final_observations)
    final_valid = extras.get("final_obs_valid")
    if not isinstance(final_valid, torch.Tensor) or final_valid.shape != (num_envs,):
        raise ValueError("Tracking final observations require one validity flag per environment.")
    torch.where(final_valid.unsqueeze(-1), final_values, destination, out=destination)


@torch.no_grad()
def _motion_tracking_evaluator(
    model: Any,
    env: Any,
    expert: ForwardBackwardExpertBuffer,
    clip_ids: tuple[str, ...],
    *,
    context_for_frames: Callable[[torch.Tensor], torch.Tensor],
    metrics: tuple[_TrackingMetricProjection, ...],
    include_reset_frame: bool,
    allow_horizon_truncation: bool,
) -> MotionTrackingEvaluation:
    """Run native shuffled-replica tracking and profile-neutral exact EMD math."""
    started = time.perf_counter()
    if not isinstance(expert, ForwardBackwardExpertBuffer):
        raise TypeError("Motion tracking requires the already-built ForwardBackwardExpertBuffer.")
    if expert.clip_ids != clip_ids:
        raise ValueError("Tracking clip ids must match the expert buffer in stable order.")
    if not clip_ids or env.num_envs < 1 or not callable(getattr(env, "reset_motion_clips", None)):
        raise TypeError("Motion tracking requires clips and a vector environment with exact clip resets.")
    if not metrics or len(metrics) > 2:
        raise ValueError("Motion tracking requires one primary metric and at most one diagnostic.")
    if any(
        metric.target_frames.ndim != 2
        or metric.target_frames.shape[0] != expert.frames.shape[0]
        or metric.target_frames.dtype is not torch.float32
        or metric.target_frames.device != expert.device
        for metric in metrics
    ):
        raise ValueError("Tracking target projections must align with float32 expert frames on device.")

    command = env.command_manager.get_term("motion")
    table = command.table
    payload = command.payload
    if not isinstance(table, MotionTaskTable) or not isinstance(payload, MotionStatePayload):
        raise TypeError("Motion tracking requires MotionTaskTable and MotionStatePayload.")
    if payload.table is not table:
        raise ValueError("The tracking payload and command must share one trajectory table.")
    source_positions = {clip_id: index for index, clip_id in enumerate(table.source_clip_ids)}
    if any(clip_id not in source_positions for clip_id in clip_ids):
        raise ValueError("Every tracking clip must exist in the environment trajectory table.")
    source_clips = {clip.clip_id: clip for clip in table.clip_index.clips}
    expert_length_values = tuple(
        table.expert_sample_grid.sample_count(
            frame_count=source_clips[clip_id].frame_count,
            source_fps=source_clips[clip_id].source_fps,
        )
        for clip_id in clip_ids
    )
    metric_length_values = tuple(length if include_reset_frame else length - 1 for length in expert_length_values)
    evaluated_length_values = (
        tuple(min(length, env.max_episode_length) for length in metric_length_values)
        if allow_horizon_truncation
        else metric_length_values
    )
    expert_positions = {clip_id: index for index, clip_id in enumerate(expert.clip_ids)}
    clip_expert_indices = torch.tensor(
        [expert_positions[clip_id] for clip_id in clip_ids],
        dtype=torch.int64,
        device=expert.device,
    )
    clip_source_indices = torch.tensor(
        [source_positions[clip_id] for clip_id in clip_ids],
        dtype=torch.int64,
        device=expert.device,
    )
    raw_frame_counts = expert.clip_lengths.index_select(0, clip_expert_indices)
    expected_raw_frame_counts = torch.tensor(expert_length_values, dtype=torch.int64, device=expert.device)
    torch._assert_async(
        torch.all(raw_frame_counts == expected_raw_frame_counts),
        "Tracking source metadata and expert clip lengths differ.",
    )
    trace_offset = int(include_reset_frame)
    metric_frame_counts, evaluated_frame_counts = _tracking_frame_counts(
        raw_frame_counts,
        episode_length=env.max_episode_length,
        include_reset_frame=include_reset_frame,
        allow_horizon_truncation=allow_horizon_truncation,
    )

    max_frames = max(evaluated_length_values)
    chunk_capacity = min(env.num_envs, len(clip_ids))
    evaluation_capacity = len(clip_ids)
    feature_width = max(metric.target_frames.shape[1] for metric in metrics)
    workspace = _UniformEmdWorkspace(
        lengths=evaluated_length_values,
        device=expert.device,
        feature_width=feature_width,
    )
    emd_values = tuple(torch.empty(len(clip_ids), dtype=torch.float64, device=expert.device) for _metric in metrics)
    traces = tuple(
        torch.empty(env.num_envs, max_frames, metric.target_frames.shape[1], device=expert.device) for metric in metrics
    )
    representative_traces = tuple(
        torch.empty(evaluation_capacity, max_frames, metric.target_frames.shape[1], device=expert.device)
        for metric in metrics
    )
    target_indices = torch.empty(chunk_capacity, max_frames, dtype=torch.int64, device=expert.device)
    target_traces = tuple(
        torch.empty(evaluation_capacity, max_frames, metric.target_frames.shape[1], device=expert.device)
        for metric in metrics
    )
    frame_grid = torch.arange(max_frames, dtype=torch.int64, device=expert.device)
    unexpected_done = torch.empty(env.num_envs, dtype=torch.bool, device=expert.device)

    for chunk_start in range(0, len(clip_ids), env.num_envs):
        chunk_count = min(env.num_envs, len(clip_ids) - chunk_start)
        chunk_end = chunk_start + chunk_count
        chunk_expert = clip_expert_indices[chunk_start:chunk_end]
        chunk_lengths = evaluated_frame_counts[chunk_start:chunk_end]
        chunk_source = clip_source_indices[chunk_start:chunk_end]
        assignment, representative_env_rows = _native_tracking_assignment(env.num_envs, chunk_count, expert.device)
        assigned_expert = chunk_expert.index_select(0, assignment)
        assigned_source = chunk_source.index_select(0, assignment)
        assigned_offsets = expert.clip_offsets.index_select(0, assigned_expert)
        assigned_raw_lengths = expert.clip_lengths.index_select(0, assigned_expert)
        assigned_evaluated_lengths = chunk_lengths.index_select(0, assignment)

        reset_result = env.reset_motion_clips(assigned_source)
        observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        observations = _as_tensordict(observations, env.num_envs)
        unexpected_done.zero_()
        if include_reset_frame:
            for trace, metric in zip(traces, metrics, strict=True):
                trace[:, 0].copy_(metric.observe(observations))
        max_steps = max(evaluated_length_values[chunk_start:chunk_end]) - trace_offset

        for step in range(max_steps):
            local_frame = torch.full_like(assigned_raw_lengths, step + 1)
            torch.minimum(local_frame, assigned_raw_lengths - 1, out=local_frame)
            frame_indices = assigned_offsets + local_frame
            context = context_for_frames(frame_indices)
            action = model.action_sample(observations, context, deterministic=True)
            observations, _reward, terminated, truncated, extras = env.step(action)
            observations = _as_tensordict(observations, env.num_envs)
            for trace, metric in zip(traces, metrics, strict=True):
                _write_tracking_reached_values(
                    trace[:, step + trace_offset],
                    metric,
                    observations,
                    extras,
                    env.num_envs,
                )
            premature = _tracking_done_is_premature(
                step,
                assigned_evaluated_lengths,
                include_reset_frame=include_reset_frame,
            )
            unexpected_done.logical_or_((terminated | truncated) & premature)

        torch._assert_async(
            torch.all(~unexpected_done),
            "A tracking rollout reset before its expert reference horizon ended.",
        )
        chunk_offsets = expert.clip_offsets.index_select(0, chunk_expert)
        chunk_target_indices = target_indices[:chunk_count]
        torch.minimum(frame_grid.unsqueeze(0), chunk_lengths.unsqueeze(1) - 1, out=chunk_target_indices)
        chunk_target_indices.add_(chunk_offsets.unsqueeze(1) + (0 if include_reset_frame else 1))

        for metric, trace, representative, target in zip(
            metrics,
            traces,
            representative_traces,
            target_traces,
            strict=True,
        ):
            chunk_representative = representative[chunk_start:chunk_end]
            torch.index_select(trace, 0, representative_env_rows, out=chunk_representative)
            chunk_target = target[chunk_start:chunk_end]
            torch.index_select(
                metric.target_frames,
                0,
                chunk_target_indices.view(-1),
                out=chunk_target.view(-1, metric.target_frames.shape[1]),
            )

    for representative, target, output in zip(representative_traces, target_traces, emd_values, strict=True):
        workspace.compute(representative, target, output)

    if expert.device.type == "cuda":
        torch.cuda.synchronize(expert.device)
    coverage = evaluated_frame_counts.to(torch.float64) / metric_frame_counts
    return MotionTrackingEvaluation(
        clip_ids=clip_ids,
        emd=emd_values[0],
        obs_state_emd=None if len(emd_values) == 1 else emd_values[1],
        source_frame_counts=metric_frame_counts,
        evaluated_frame_counts=evaluated_frame_counts,
        coverage_fraction=coverage,
        duration_seconds=time.perf_counter() - started,
    )


class MotionTrackingEvaluator(Protocol):
    """Callable Isaac-native rollout evaluator consumed by the curriculum."""

    def __call__(
        self,
        model: Any,
        env: Any,
        expert: ForwardBackwardExpertBuffer,
        clip_ids: tuple[str, ...],
    ) -> MotionTrackingEvaluation:
        """Evaluate every stable clip and return EMD aligned to ``clip_ids``."""


@torch.no_grad()
def g1_motion_tracking_evaluator(
    model: Any,
    env: Any,
    expert: ForwardBackwardExpertBuffer,
    clip_ids: tuple[str, ...],
) -> MotionTrackingEvaluation:
    """Evaluate G1 clips with exact resets, mean actions, and released EMDs.

    A clip longer than one uninterrupted native episode is evaluated on its
    maximal pre-timeout prefix. Only G1-owned context and metric projections
    live here; reset, rollout, terminal-observation, and EMD math are shared.
    """
    command = env.command_manager.get_term("motion")
    table = command.table
    payload = command.payload
    if not isinstance(table, MotionTaskTable) or not isinstance(payload, MotionStatePayload):
        raise TypeError("G1 tracking requires MotionTaskTable and MotionStatePayload.")
    if payload.table is not table:
        raise ValueError("The G1 tracking payload and command must share one trajectory table.")
    if len(table.joint_names) != 29 or table.joint_names != tuple(payload.robot.joint_names):
        raise ValueError("The G1 tracking evaluator requires 29 simulator-ordered joints.")
    expected_reference_frames = (*payload.robot.body_names, G1_HEAD_FRAME_NAME)
    if table.reference_frame_names != expected_reference_frames:
        raise ValueError("G1 tracking requires live bodies followed by the head_link reference frame.")
    action_term = env.action_manager.get_term("joint_position")
    if not isinstance(action_term, MotionJointPositionAction):
        raise TypeError("G1 tracking requires MotionJointPositionAction.")
    if len(action_term.joint_names) != 29 or action_term.joint_ids.shape != (29,):
        raise ValueError("G1 tracking requires one declared 29-joint behavior axis.")
    default_joint_position = action_term.joint_default_position
    if default_joint_position.shape != (29,) or default_joint_position.device != expert.device:
        raise ValueError("G1 tracking requires 29 default joint positions on the expert device.")
    if expert.schema.expert_feature_width != 527 or expert.frames.shape[1] != 527:
        raise ValueError("G1 tracking requires the 64+463 expert frame layout.")

    def context_for_frames(frame_indices: torch.Tensor) -> torch.Tensor:
        target_frames = expert.frames.index_select(0, frame_indices)
        target_observations = _expert_frame_tensordict(model, target_frames)
        return model.context_project(model.backward_map(target_observations))

    def observe_joint_position(_observations: TensorDictBase) -> torch.Tensor:
        return action_term.joint_position

    def observe_state(observations: TensorDictBase) -> torch.Tensor:
        state = observations.get("state")
        if state is None or state.shape[0] != env.num_envs or state.shape[1] < 23:
            raise ValueError("G1 tracking requires one live state observation per environment.")
        return state[:, :23]

    target_joint_position = expert.frames[:, :29] + default_joint_position
    return _motion_tracking_evaluator(
        model,
        env,
        expert,
        clip_ids,
        context_for_frames=context_for_frames,
        metrics=(
            _TrackingMetricProjection(target_joint_position, observe_joint_position),
            _TrackingMetricProjection(expert.frames[:, :23], observe_state),
        ),
        include_reset_frame=True,
        allow_horizon_truncation=True,
    )


@torch.no_grad()
def smpl_motion_tracking_evaluator(
    model: Any,
    env: Any,
    expert: ForwardBackwardExpertBuffer,
    clip_ids: tuple[str, ...],
) -> MotionTrackingEvaluation:
    """Evaluate the native held-out HumEnv pose metric without autoreset truncation.

    The released protocol resets to source frame zero, applies one deterministic
    action for every later source frame, and computes exact uniform transport
    over the first 214 values of those reached and target observations. Contexts
    are clip-safe means of the next up-to-eight backward features.
    """
    command = env.command_manager.get_term("motion")
    table = command.table
    payload = command.payload
    if not isinstance(table, MotionTaskTable) or not isinstance(payload, MotionStatePayload):
        raise TypeError("SMPL tracking requires MotionTaskTable and MotionStatePayload.")
    if payload.table is not table or table.reference_frame_names:
        raise ValueError("SMPL tracking requires one projected-observation table without reference frames.")
    if expert.frames.shape[1] != 358 or expert.schema.expert_feature_width != 358:
        raise ValueError("SMPL tracking requires the native 358-wide expert observation route.")
    if table.field("observation").shape != expert.frames.shape:
        raise ValueError("SMPL tracking table and expert observation layouts differ.")
    action = env.action_manager.get_term("joint_position")
    if not isinstance(action, MotionMujocoControlAction) or action.action_dim != 69:
        raise TypeError("SMPL tracking requires the native 69-wide MuJoCo control action.")

    source_clips = {clip.clip_id: clip for clip in table.clip_index.clips}
    expert_length_values = tuple(
        table.expert_sample_grid.sample_count(
            frame_count=source_clips[clip_id].frame_count,
            source_fps=source_clips[clip_id].source_fps,
        )
        for clip_id in expert.clip_ids
    )
    contexts = _tracking_context_table(model, expert, expert_length_values, window_length=8)

    def context_for_frames(frame_indices: torch.Tensor) -> torch.Tensor:
        return contexts.index_select(0, frame_indices)

    def observe_pose(observations: TensorDictBase) -> torch.Tensor:
        policy = observations.get("policy")
        if policy is None or policy.shape != (env.num_envs, 358):
            raise ValueError("SMPL tracking requires one 358-wide live policy observation per environment.")
        return policy[:, :214]

    return _motion_tracking_evaluator(
        model,
        env,
        expert,
        clip_ids,
        context_for_frames=context_for_frames,
        metrics=(_TrackingMetricProjection(expert.frames[:, :214], observe_pose),),
        include_reset_frame=False,
        allow_horizon_truncation=False,
    )


@torch.no_grad()
def smpl_motion_tracking_evaluator_packed(
    model: Any,
    env: Any,
    expert: ForwardBackwardExpertBuffer,
    clip_ids: tuple[str, ...],
    *,
    policy_count: int,
) -> tuple[MotionTrackingEvaluation, ...]:
    """Evaluate checkpoint-major SMPL policies through one packed simulator rollout.

    Args:
        model: Functional checkpoint stack with checkpoint-major inference methods.
        env: Flat vector environment laid out as ``[policy, lane]``.
        expert: Shared immutable SMPL expert buffer.
        clip_ids: Stable expert clip order.
        policy_count: Number of checkpoint policies stacked by :paramref:`model`.

    Returns:
        One stable-order evaluation view per checkpoint policy.
    """
    started = time.perf_counter()
    if (
        type(policy_count) is not int
        or policy_count < 1
        or getattr(model, "policy_count", None) != policy_count
        or not isinstance(expert, ForwardBackwardExpertBuffer)
        or expert.clip_ids != clip_ids
    ):
        raise ValueError("Packed SMPL tracking requires aligned policies, expert storage, and stable clip ids.")
    if env.num_envs < policy_count or env.num_envs % policy_count != 0:
        raise ValueError("Packed SMPL environment lanes must divide exactly across checkpoint policies.")
    lanes_per_policy = env.num_envs // policy_count
    if lanes_per_policy > len(clip_ids):
        raise ValueError("Packed SMPL tracking requires no more lanes per policy than clips.")
    if not callable(getattr(env, "reset_motion_clips_selected", None)):
        raise TypeError("Packed SMPL tracking requires the selected exact-reset lifecycle.")
    for name in ("backward_map", "context_project", "action_deterministic"):
        if not callable(getattr(model, name, None)):
            raise TypeError(f"Packed SMPL model must expose {name}().")

    command = env.command_manager.get_term("motion")
    table = command.table
    payload = command.payload
    if not isinstance(table, MotionTaskTable) or not isinstance(payload, MotionStatePayload):
        raise TypeError("Packed SMPL tracking requires MotionTaskTable and MotionStatePayload.")
    if payload.table is not table or table.reference_frame_names:
        raise ValueError("Packed SMPL tracking requires one projected-observation table without reference frames.")
    if expert.frames.shape[1] != 358 or expert.schema.expert_feature_width != 358:
        raise ValueError("Packed SMPL tracking requires the native 358-wide expert observation route.")
    if table.field("observation").shape != expert.frames.shape:
        raise ValueError("Packed SMPL table and expert observation layouts differ.")
    action = env.action_manager.get_term("joint_position")
    if not isinstance(action, MotionMujocoControlAction) or action.action_dim != 69:
        raise TypeError("Packed SMPL tracking requires the native 69-wide MuJoCo control action.")

    source_positions = {clip_id: index for index, clip_id in enumerate(table.source_clip_ids)}
    if any(clip_id not in source_positions for clip_id in clip_ids):
        raise ValueError("Every packed tracking clip must exist in the environment trajectory table.")
    source_clips = {clip.clip_id: clip for clip in table.clip_index.clips}
    expert_length_values = tuple(
        table.expert_sample_grid.sample_count(
            frame_count=source_clips[clip_id].frame_count,
            source_fps=source_clips[clip_id].source_fps,
        )
        for clip_id in clip_ids
    )
    metric_length_values = tuple(length - 1 for length in expert_length_values)
    lane_queues = _motion_tracking_refill_queues(
        clip_ids,
        metric_length_values,
        lane_count=lanes_per_policy,
    )
    lane_work = tuple(sum(metric_length_values[clip_index] for clip_index in queue) for queue in lane_queues)
    makespan = max(lane_work)

    raw_frame_counts = expert.clip_lengths
    expected_raw_frame_counts = torch.tensor(expert_length_values, dtype=torch.int64, device=expert.device)
    torch._assert_async(
        torch.all(raw_frame_counts == expected_raw_frame_counts),
        "Packed source metadata and expert clip lengths differ.",
    )
    metric_frame_counts, evaluated_frame_counts = _tracking_frame_counts(
        raw_frame_counts,
        episode_length=env.max_episode_length,
        include_reset_frame=False,
        allow_horizon_truncation=False,
    )
    contexts = _tracking_context_table_packed(
        model,
        expert,
        expert_length_values,
        policy_count=policy_count,
        window_length=8,
    )

    expert_offset_values = []
    metric_offset_values = []
    expert_cursor = 0
    metric_cursor = 0
    for expert_length, metric_length in zip(expert_length_values, metric_length_values, strict=True):
        expert_offset_values.append(expert_cursor)
        metric_offset_values.append(metric_cursor)
        expert_cursor += expert_length
        metric_cursor += metric_length
    metric_rows_per_policy = metric_cursor

    policy_frame_values = [[0] * lanes_per_policy for _step in range(makespan)]
    premature_values = [[False] * lanes_per_policy for _step in range(makespan)]
    reset_lane_values: list[list[int]] = [[] for _step in range(makespan)]
    reset_clip_values: list[list[int]] = [[] for _step in range(makespan)]
    active_lane_values: list[list[int]] = [[] for _step in range(makespan)]
    active_destination_values: list[list[int]] = [[] for _step in range(makespan)]
    for lane, queue in enumerate(lane_queues):
        lane_step = 0
        held_frame = 0
        for clip_index in queue:
            reset_lane_values[lane_step].append(lane)
            reset_clip_values[lane_step].append(clip_index)
            work = metric_length_values[clip_index]
            expert_offset = expert_offset_values[clip_index]
            metric_offset = metric_offset_values[clip_index]
            for local_step in range(work):
                step = lane_step + local_step
                policy_frame_values[step][lane] = expert_offset + local_step + 1
                premature_values[step][lane] = local_step < work - 1
                active_lane_values[step].append(lane)
                active_destination_values[step].append(metric_offset + local_step)
            lane_step += work
            held_frame = expert_offset + work
        for step in range(lane_step, makespan):
            policy_frame_values[step][lane] = held_frame

    policy_frame_indices = torch.tensor(policy_frame_values, dtype=torch.int64, device=expert.device)
    premature_masks = torch.tensor(
        [values * policy_count for values in premature_values],
        dtype=torch.bool,
        device=expert.device,
    )
    reset_env_values = []
    reset_source_values = []
    reset_event_offsets = [0]
    active_env_values = []
    trace_destination_values = []
    active_event_offsets = [0]
    for lanes, clips, active_lanes, destinations in zip(
        reset_lane_values,
        reset_clip_values,
        active_lane_values,
        active_destination_values,
        strict=True,
    ):
        if lanes:
            reset_env_values.extend(
                policy * lanes_per_policy + lane for policy in range(policy_count) for lane in lanes
            )
            reset_source_values.extend(
                source_positions[clip_ids[clip_index]] for _policy in range(policy_count) for clip_index in clips
            )
        reset_event_offsets.append(len(reset_env_values))
        active_env_values.extend(
            policy * lanes_per_policy + lane for policy in range(policy_count) for lane in active_lanes
        )
        trace_destination_values.extend(
            policy * metric_rows_per_policy + destination
            for policy in range(policy_count)
            for destination in destinations
        )
        active_event_offsets.append(len(active_env_values))

    reset_env_ids = torch.tensor(reset_env_values, dtype=torch.int64, device=expert.device)
    reset_source_indices = torch.tensor(reset_source_values, dtype=torch.int64, device=expert.device)
    active_env_ids = torch.tensor(active_env_values, dtype=torch.int64, device=expert.device)
    trace_destinations = torch.tensor(trace_destination_values, dtype=torch.int64, device=expert.device)

    trace = torch.empty(policy_count * metric_rows_per_policy, 214, dtype=torch.float32, device=expert.device)
    reached_values = torch.empty(env.num_envs, 214, dtype=torch.float32, device=expert.device)
    selected_values = torch.empty_like(reached_values)
    context_values = torch.empty(
        policy_count,
        lanes_per_policy,
        model.context_dim,
        dtype=expert.frames.dtype,
        device=expert.device,
    )
    unexpected_done = torch.zeros(env.num_envs, dtype=torch.bool, device=expert.device)
    step_done = torch.empty_like(unexpected_done)

    def observe_pose(observations: TensorDictBase) -> torch.Tensor:
        policy = observations.get("policy")
        if policy is None or policy.shape != (env.num_envs, 358):
            raise ValueError("Packed SMPL tracking requires one 358-wide live observation per environment.")
        return policy[:, :214]

    metric = _TrackingMetricProjection(expert.frames[:, :214], observe_pose)
    observations: TensorDictBase | None = None
    for step in range(makespan):
        reset_start = reset_event_offsets[step]
        reset_end = reset_event_offsets[step + 1]
        if reset_start != reset_end:
            reset_result = env.reset_motion_clips_selected(
                reset_source_indices[reset_start:reset_end],
                env_ids=reset_env_ids[reset_start:reset_end],
            )
            reset_observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            observations = _as_tensordict(reset_observations, env.num_envs)
        if observations is None:
            raise RuntimeError("Packed SMPL tracking did not initialize every environment lane.")
        packed_observations = observations.view(policy_count, lanes_per_policy)
        torch.index_select(contexts, 1, policy_frame_indices[step], out=context_values)
        packed_action = model.action_deterministic(packed_observations, context_values)
        observations, _reward, terminated, truncated, extras = env.step(
            packed_action.view(env.num_envs, action.action_dim)
        )
        observations = _as_tensordict(observations, env.num_envs)
        _write_tracking_reached_values(reached_values, metric, observations, extras, env.num_envs)
        active_start = active_event_offsets[step]
        active_end = active_event_offsets[step + 1]
        active_count = active_end - active_start
        torch.index_select(
            reached_values,
            0,
            active_env_ids[active_start:active_end],
            out=selected_values[:active_count],
        )
        trace.index_copy_(
            0,
            trace_destinations[active_start:active_end],
            selected_values[:active_count],
        )
        torch.logical_or(terminated, truncated, out=step_done)
        step_done.logical_and_(premature_masks[step])
        unexpected_done.logical_or_(step_done)

    torch._assert_async(
        torch.all(~unexpected_done),
        "A packed tracking rollout reset before its expert reference horizon ended.",
    )
    packed_lengths = metric_length_values * policy_count
    observed_starts = torch.tensor(
        tuple(
            policy * metric_rows_per_policy + offset
            for policy in range(policy_count)
            for offset in metric_offset_values
        ),
        dtype=torch.int64,
        device=expert.device,
    )
    target_starts = torch.tensor(
        tuple(offset + 1 for _policy in range(policy_count) for offset in expert_offset_values),
        dtype=torch.int64,
        device=expert.device,
    )
    emd = torch.empty(policy_count * len(clip_ids), dtype=torch.float64, device=expert.device)
    workspace = _UniformEmdWorkspace(lengths=packed_lengths, device=expert.device, feature_width=214)
    workspace.compute_flat(trace, observed_starts, expert.frames[:, :214], target_starts, emd)

    if expert.device.type == "cuda":
        torch.cuda.synchronize(expert.device)
    duration_seconds = time.perf_counter() - started
    emd_by_policy = emd.view(policy_count, len(clip_ids))
    coverage = torch.ones_like(metric_frame_counts, dtype=torch.float64)
    return tuple(
        MotionTrackingEvaluation(
            clip_ids=clip_ids,
            emd=emd_by_policy[policy],
            obs_state_emd=None,
            source_frame_counts=metric_frame_counts,
            evaluated_frame_counts=evaluated_frame_counts,
            coverage_fraction=coverage,
            duration_seconds=duration_seconds,
        )
        for policy in range(policy_count)
    )


def motion_tracking_priorities(
    evaluation: MotionTrackingEvaluation,
    clip_ids: tuple[str, ...],
    device: str | torch.device,
) -> torch.Tensor:
    """Return released exponential priorities in stable task-table order."""
    if set(evaluation.clip_ids) != set(clip_ids) or len(evaluation.clip_ids) != len(clip_ids):
        raise ValueError("Tracking evaluation does not cover the configured stable clip ids.")
    positions = {clip_id: index for index, clip_id in enumerate(evaluation.clip_ids)}
    order = torch.tensor([positions[clip_id] for clip_id in clip_ids], dtype=torch.int64, device=evaluation.emd.device)
    emd = evaluation.emd.index_select(0, order).to(device=device, dtype=torch.float32)
    return torch.pow(2.0, torch.clamp(emd, min=0.5, max=2.0) * 2.0)


class MotionTrackingCurriculum(RunnerLifecycleExtension):
    """Coordinate isolated evaluation and one atomic two-sampler update."""

    def __init__(
        self,
        env: Any,
        algorithm: Any,
        log_dir: str | None,
        device: str,
        *,
        evaluator: str | Callable[..., MotionTrackingEvaluation],
        command_name: str = "motion",
        evaluation_seed: int = 0,
        record_directory: str = "tracking_curriculum",
    ) -> None:
        """Bind stable table/expert identities and the injected evaluator."""
        super().__init__(env, algorithm, log_dir, device)
        command = env.unwrapped.command_manager.get_term(command_name)
        table = command.table
        expert = algorithm.expert
        if not isinstance(table, MotionTaskTable):
            raise TypeError("Motion tracking curriculum requires MotionTaskTable.")
        if not isinstance(expert, ForwardBackwardExpertBuffer):
            raise TypeError("Motion tracking curriculum requires ForwardBackwardExpertBuffer.")
        if table.clip_ids != expert.clip_ids:
            raise ValueError("Motion task and expert stable clip ids must match exactly.")
        evaluation_transaction = getattr(env.unwrapped, "evaluation_transaction", None)
        if not callable(evaluation_transaction):
            raise TypeError("Motion environment must expose evaluation_transaction(seed).")
        self.table = table
        self.expert = expert
        self.clip_ids = table.clip_ids
        self.evaluator: MotionTrackingEvaluator = resolve_callable(evaluator)  # type: ignore[assignment]
        self.evaluation_seed = evaluation_seed
        self.record_directory = record_directory
        self.applied_transitions: list[int] = []

    def on_transition(self, transition: int) -> TensorDictBase:
        """Evaluate, validate both samplers, commit, reset, and publish one record."""
        if self.log_dir is None:
            raise RuntimeError("Motion tracking curriculum requires a runner log directory.")
        destination = Path(self.log_dir) / self.record_directory / f"{transition}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Tracking curriculum event already exists: {destination}")

        model = self.algorithm.model
        was_training = model.training
        model.eval()
        try:
            with self.env.unwrapped.evaluation_transaction(self.evaluation_seed):
                evaluation = self.evaluator(model, self.env.unwrapped, self.expert, self.clip_ids)
        finally:
            model.train(was_training)
        if not isinstance(evaluation, MotionTrackingEvaluation):
            raise TypeError("Motion tracking evaluator must return MotionTrackingEvaluation.")
        priorities = motion_tracking_priorities(evaluation, self.clip_ids, self.table.device)
        expert_priorities = priorities.to(self.expert.device)

        # Validate both targets before either mutable sampler changes.
        self.table.validate_clip_priorities(self.clip_ids, priorities)
        self.expert.validate_priorities(expert_priorities)
        previous_table = self.table.clip_priorities.clone()
        previous_expert = self.expert.priorities.clone()

        temporary = destination.with_suffix(".tmp")
        with temporary.open("x") as stream:
            json.dump(
                {
                    "schema": "motion_tracking_curriculum_v1",
                    "transition": transition,
                    "evaluation_seed": self.evaluation_seed,
                    "duration_seconds": evaluation.duration_seconds,
                    "clip_ids": self.clip_ids,
                    "priorities": priorities.cpu().tolist(),
                    "metrics": evaluation.serializable_metrics(),
                },
                stream,
                indent=2,
                sort_keys=True,
            )
        try:
            self.table.set_clip_priorities(self.clip_ids, priorities)
            self.expert.set_priorities(expert_priorities)
            reset_result = self.env.reset()
            reset_observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            if not isinstance(reset_observations, TensorDictBase):
                raise TypeError("Motion environment reset must return TensorDict observations.")
            reset = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
            self.algorithm.process_env_reset(reset_observations, reset)
            temporary.replace(destination)
        except BaseException:
            self.table.set_clip_priorities(self.clip_ids, previous_table)
            self.expert.set_priorities(previous_expert)
            temporary.unlink(missing_ok=True)
            raise
        self.applied_transitions.append(transition)
        return reset_observations

    def state_dict(self) -> dict[str, object]:
        """Return exact curriculum state for checkpoint resume."""
        return {
            "clip_ids": self.clip_ids,
            "table_priorities": self.table.clip_priorities.clone(),
            "expert_priorities": self.expert.priorities.clone(),
            "applied_transitions": tuple(self.applied_transitions),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore exact sampler priorities and completed event history."""
        if state.get("clip_ids") != self.clip_ids:
            raise ValueError("Tracking curriculum checkpoint clip ids do not match the environment.")
        table_priorities = state.get("table_priorities")
        expert_priorities = state.get("expert_priorities")
        transitions = state.get("applied_transitions")
        if not isinstance(table_priorities, torch.Tensor) or not isinstance(expert_priorities, torch.Tensor):
            raise TypeError("Tracking curriculum checkpoint priorities must be tensors.")
        if not isinstance(transitions, tuple) or not all(isinstance(value, int) for value in transitions):
            raise TypeError("Tracking curriculum applied_transitions must be a tuple of integers.")
        self.table.validate_clip_priorities(self.clip_ids, table_priorities)
        self.expert.validate_priorities(expert_priorities)
        if not torch.equal(table_priorities.to(expert_priorities), expert_priorities):
            raise ValueError("Tracking curriculum table and expert priorities must match exactly.")
        self.table.set_clip_priorities(self.clip_ids, table_priorities)
        self.expert.set_priorities(expert_priorities)
        self.applied_transitions = list(transitions)


__all__ = [
    "MotionTrackingCurriculum",
    "MotionTrackingEvaluation",
    "MotionTrackingEvaluator",
    "g1_motion_tracking_evaluator",
    "motion_tracking_refill_lane_count",
    "motion_tracking_priorities",
    "smpl_motion_tracking_evaluator",
    "smpl_motion_tracking_evaluator_packed",
]
