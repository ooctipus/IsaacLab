# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL forward-backward rollout evaluation and sampling curriculum."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Protocol

import torch
from rsl_rl.env import VecEnv
from rsl_rl.modules.forward_backward import trajectory_context_sequence
from rsl_rl.modules.reward_channels import get_forward_backward_schema_hash
from rsl_rl.runners import OffPolicyRunner
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict, TensorDictBase

from ...metrics import UniformAssignmentWorkspace


class _SequenceCommand(Protocol):
    """Command boundary required for exact sequence resets."""

    randomize_command_indices: bool

    def bind_rows(self, env_ids: torch.Tensor, task_rows: torch.Tensor) -> None:
        """Bind exact task rows to environment rows."""


class _EvaluationHistory(Protocol):
    """Learner-owned derived observations for one isolated evaluation rollout."""

    def decorate_current(self, observations: TensorDictBase) -> TensorDictBase:
        """Attach derived history before evaluating the current action."""

    def advance(self, current: TensorDictBase, returned: TensorDictBase, done: torch.Tensor) -> None:
        """Advance one same-step transition and attach returned history."""


class _EvaluationHistoryFactory(Protocol):
    """Factory for one independent learner-owned history session."""

    def __call__(self, observations: TensorDictBase) -> _EvaluationHistory | None:
        """Return a fresh session, or none when the learner has no derived history."""


class _EvaluationScope(Protocol):
    """Domain-owned temporary reset policy used during evaluation."""

    def __call__(self, seed: int, reset_source_name: str | None) -> AbstractContextManager[None]:
        """Return a context that isolates evaluation reset state."""


class _ObservationSchema(Protocol):
    """Observation layout needed to split immutable expert frames."""

    field_widths: tuple[tuple[str, int], ...]

    def route(self, name: str) -> tuple[str, ...]:
        """Return the ordered fields consumed by one model route."""


class _TrackingModel(Protocol):
    """Model operations consumed by deterministic tracking evaluation."""

    observation_schema: _ObservationSchema
    context_dim: int
    context_normalization: bool

    def backward_map(self, observations: TensorDictBase) -> torch.Tensor:
        """Encode expert observations into backward features."""

    def action_sample(
        self, observations: TensorDictBase, context: torch.Tensor, *, deterministic: bool
    ) -> torch.Tensor:
        """Return actions conditioned on observations and contexts."""


class _TrackingAlgorithm(Protocol):
    """Learner lifecycle operations consumed by tracking transitions."""

    expert: ForwardBackwardExpertBuffer
    model: _TrackingModel

    def eval_mode(self) -> None:
        """Enter the learner-owned evaluation mode."""

    def train_mode(self) -> None:
        """Restore learner-owned training and frozen-child modes."""

    def evaluation_history(self, observations: TensorDictBase) -> _EvaluationHistory | None:
        """Create one isolated rollout-history session."""

    def process_env_reset(self, observations: TensorDictBase, reset: torch.Tensor) -> None:
        """Synchronize learner state after the evaluation reset."""


@contextmanager
def forward_backward_evaluation_scope(
    env: VecEnv,
    command: _SequenceCommand,
    domain_scope: _EvaluationScope,
    seed: int,
    *,
    reset_source_name: str | None,
) -> Iterator[None]:
    """Isolate global RNG and command reset state for deterministic evaluation."""
    if type(seed) is not int:
        raise TypeError("Evaluation seed must be an integer.")
    if not isinstance(command.randomize_command_indices, bool):
        raise TypeError("Sequence command randomize_command_indices must be a boolean.")
    if not callable(domain_scope):
        raise TypeError("Evaluation scope binding must be callable.")
    device = torch.device(env.device)
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device() if device.type == "cuda" else None
    )
    cuda_devices = [] if device_index is None else [device_index]
    randomize_command_indices = command.randomize_command_indices
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.default_generators[device_index].manual_seed(seed)
        command.randomize_command_indices = False
        try:
            with domain_scope(seed, reset_source_name):
                yield
        finally:
            command.randomize_command_indices = randomize_command_indices


def reset_tracking_sequences(
    env: VecEnv,
    command: _SequenceCommand,
    sequence_start_rows: torch.Tensor,
    sequence_indices: torch.Tensor,
) -> tuple[TensorDictBase, Mapping[str, object]]:
    """Bind exact sequence-start rows, then perform one ordinary environment reset."""
    if command.randomize_command_indices:
        raise RuntimeError("Tracking sequence resets require forward_backward_evaluation_scope().")
    device = torch.device(env.device)
    if (
        sequence_start_rows.ndim != 1
        or sequence_start_rows.dtype is not torch.int64
        or sequence_start_rows.device != device
    ):
        raise ValueError("Sequence start rows must be a one-dimensional int64 tensor on the environment device.")
    if (
        sequence_indices.shape != (env.num_envs,)
        or sequence_indices.dtype is not torch.int64
        or sequence_indices.device != device
    ):
        raise ValueError("Sequence indices must contain one int64 entry per environment on the environment device.")
    if not callable(getattr(command, "bind_rows", None)):
        raise TypeError("Sequence command must expose bind_rows().")
    env_ids = torch.arange(env.num_envs, dtype=torch.int64, device=device)
    command.bind_rows(env_ids, sequence_start_rows.index_select(0, sequence_indices))
    return env.reset()


def expert_frame_tensordict(model: _TrackingModel, frames: torch.Tensor) -> TensorDictBase:
    """Split prebuilt expert frames into the model's backward route."""
    schema = model.observation_schema
    route = tuple(schema.route("backward"))
    widths = dict(schema.field_widths)
    values: dict[str, torch.Tensor] = {}
    offset = 0
    for name in route:
        end = offset + widths[name]
        values[name] = frames[:, offset:end]
        offset = end
    if offset != frames.shape[1]:
        raise ValueError("Expert frame width does not match the model backward route.")
    return TensorDict(values, batch_size=[frames.shape[0]])


def _tracking_assignment(
    num_envs: int,
    num_targets: int,
    device: torch.device,
    rng: random.Random,
    *,
    shuffle: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign all environment lanes and select one representative per target."""
    if num_envs < 1 or num_targets < 1 or num_targets > num_envs:
        raise ValueError("Tracking assignment requires 1 <= num_targets <= num_envs.")
    assigned_rows = [env_index % num_targets for env_index in range(num_envs)]
    if shuffle:
        rng.shuffle(assigned_rows)
    representative_rows = [-1] * num_targets
    for env_index, target_row in enumerate(assigned_rows):
        if representative_rows[target_row] < 0:
            representative_rows[target_row] = env_index
    return (
        torch.tensor(assigned_rows, dtype=torch.int64, device=device),
        torch.tensor(representative_rows, dtype=torch.int64, device=device),
    )


@dataclass(frozen=True, slots=True)
class _TrackingMetric:
    """One uniform-assignment target paired with an environment observation field."""

    name: str
    target_frames: torch.Tensor
    observation_name: str
    projection: Callable[[torch.Tensor], torch.Tensor] | None = None
    assignment_metric: str = "uniform_assignment"

    def __post_init__(self) -> None:
        """Reject unnamed or empty observation projections."""
        if not self.name or not self.observation_name:
            raise ValueError("Tracking metric and observation names must be nonempty.")
        if self.target_frames.ndim != 2 or self.target_frames.dtype is not torch.float32:
            raise ValueError("Tracking targets must be a frame-by-feature matrix.")
        if self.assignment_metric != "uniform_assignment":
            raise ValueError("Tracking supports only the explicit uniform_assignment metric.")

    def observe(self, observations: TensorDictBase) -> torch.Tensor:
        """Return the configured live observation projection."""
        values = observations.get(self.observation_name)
        if values is None or values.ndim != 2:
            raise ValueError(f"Tracking observation {self.observation_name!r} is absent or not a feature matrix.")
        values = values if self.projection is None else self.projection(values)
        if values.ndim != 2 or values.shape[1] != self.target_frames.shape[1]:
            raise ValueError(f"Tracking observation {self.observation_name!r} differs from its target projection.")
        return values


@dataclass(frozen=True, slots=True)
class ForwardBackwardTrackingEvaluation:
    """Per-sequence tracking results emitted by an isolated evaluator."""

    sequence_ids: tuple[str, ...]
    metric_values: Mapping[str, torch.Tensor]
    source_frame_counts: torch.Tensor
    evaluated_frame_counts: torch.Tensor
    coverage_fraction: torch.Tensor
    duration_seconds: float

    def __post_init__(self) -> None:
        """Reject incomplete, duplicate, or nonfinite evaluation results."""
        if not self.sequence_ids or len(set(self.sequence_ids)) != len(self.sequence_ids):
            raise ValueError("Tracking sequence_ids must be nonempty and unique.")
        count = len(self.sequence_ids)
        if not self.metric_values or any(not isinstance(name, str) or not name for name in self.metric_values):
            raise ValueError("Tracking metric names must be nonempty strings.")
        floating = (*self.metric_values.values(), self.coverage_fraction)
        integer = (self.source_frame_counts, self.evaluated_frame_counts)
        if any(value.shape != (count,) or not value.is_floating_point() for value in floating):
            raise ValueError("Tracking floating metrics must contain one value per sequence.")
        if any(value.shape != (count,) or value.dtype is not torch.int64 for value in integer):
            raise ValueError("Tracking frame counts must be int64 with one value per sequence.")
        device = next(iter(self.metric_values.values())).device
        if any(value.device != device for value in (*floating, *integer)):
            raise ValueError("Tracking metrics must remain together on one device.")
        if any(value.requires_grad for value in floating):
            raise ValueError("Tracking metrics must be detached.")
        torch._assert_async(
            torch.all(torch.stack(tuple(torch.all(torch.isfinite(value)) for value in floating))),
            "Tracking floating metrics must be finite.",
        )
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0.0:
            raise ValueError("Tracking evaluation duration must be finite and non-negative.")


def tracking_frame_counts(
    raw_frame_counts: torch.Tensor,
    *,
    episode_length: int,
    include_reset_frame: bool,
    allow_horizon_truncation: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native metric rows and the uninterrupted rows safe to evaluate."""
    trace_offset = int(include_reset_frame)
    metric_frame_counts = raw_frame_counts if include_reset_frame else raw_frame_counts - 1
    torch._assert_async(torch.all(metric_frame_counts > 0), "Tracking sequences contain no metric frames.")
    if allow_horizon_truncation:
        evaluated_frame_counts = metric_frame_counts.clamp(max=episode_length + trace_offset)
    else:
        torch._assert_async(
            torch.all(metric_frame_counts - trace_offset <= episode_length),
            "Tracking evaluation horizon is shorter than an expert sequence.",
        )
        evaluated_frame_counts = metric_frame_counts
    return metric_frame_counts, evaluated_frame_counts


def _tracking_final_observations(
    extras: Mapping[str, object],
    required_final: torch.Tensor,
    num_envs: int,
) -> TensorDictBase | None:
    """Normalize and validate exact same-step final observations once."""
    if required_final.shape != (num_envs,) or required_final.dtype is not torch.bool:
        raise ValueError("Tracking required-final flags must contain one boolean per environment.")
    final_observations = extras.get("final_obs")
    if final_observations is None:
        torch._assert_async(
            torch.all(~required_final),
            "A consumed done tracking row is missing its exact final observation.",
        )
        return None
    if not isinstance(final_observations, TensorDictBase):
        if not isinstance(final_observations, Mapping) or any(
            not isinstance(value, torch.Tensor) for value in final_observations.values()
        ):
            raise TypeError("RSL VecEnv final_obs must be a TensorDict or tensor mapping.")
        final_observations = TensorDict(dict(final_observations), batch_size=[num_envs])
    if "final_obs_valid" in extras:
        final_valid = extras["final_obs_valid"]
        if (
            not isinstance(final_valid, torch.Tensor)
            or final_valid.shape != (num_envs,)
            or final_valid.dtype is not torch.bool
            or final_valid.device != required_final.device
        ):
            raise ValueError("Tracking final observations require one boolean validity flag per environment.")
    else:
        final_valid = required_final
    torch._assert_async(
        torch.all(~required_final | final_valid),
        "A consumed done tracking row has no valid final observation.",
    )
    return final_observations


def _write_tracking_reached_values(
    destination: torch.Tensor,
    metric: _TrackingMetric,
    observations: TensorDictBase,
    final_observations: TensorDictBase | None,
    required_final: torch.Tensor,
) -> None:
    """Write reached values after shared final-observation validation."""
    destination.copy_(metric.observe(observations))
    if final_observations is None:
        return
    final_values = metric.observe(final_observations)
    torch.where(required_final.unsqueeze(-1), final_values, destination, out=destination)


@torch.no_grad()
def forward_backward_tracking_evaluator(  # noqa: C901
    model: _TrackingModel,
    env: VecEnv,
    expert: ForwardBackwardExpertBuffer,
    sequence_ids: tuple[str, ...],
    *,
    command: _SequenceCommand,
    sequence_start_rows: torch.Tensor,
    history_factory: _EvaluationHistoryFactory,
    projections: tuple[Mapping[str, object], ...],
    context_window_length: int,
    include_reset_frame: bool,
    allow_horizon_truncation: bool,
    assignment_rng: random.Random,
    shuffle_assignments: bool,
) -> ForwardBackwardTrackingEvaluation:
    """Evaluate named expert/live projections with one exact sequence protocol."""
    if not isinstance(expert, ForwardBackwardExpertBuffer):
        raise TypeError("Tracking requires the already-built ForwardBackwardExpertBuffer.")
    if expert.clip_ids != sequence_ids:
        raise ValueError("Tracking sequence ids must match the expert buffer in stable order.")
    if env.num_envs < 1:
        raise ValueError("Tracking requires a nonempty vector environment.")
    if (
        sequence_start_rows.shape != (len(sequence_ids),)
        or sequence_start_rows.dtype is not torch.int64
        or sequence_start_rows.device != expert.device
    ):
        raise ValueError("Sequence start rows must align with expert sequence ids on the expert device.")
    if not isinstance(assignment_rng, random.Random):
        raise TypeError("Tracking assignment_rng must be random.Random.")
    if not callable(history_factory):
        raise TypeError("Tracking history_factory must be callable.")
    if expert.device.type == "cuda":
        torch.cuda.synchronize(expert.device)
    started = time.perf_counter()

    expert_observations = expert_frame_tensordict(model, expert.frames)
    metrics = []
    projection_fields = {"metric_name", "target_name", "observation_name", "projection", "assignment_metric"}
    for projection_cfg in projections:
        if set(projection_cfg) != projection_fields:
            raise ValueError("Tracking projections must contain the exact typed projection fields.")
        metric_name = projection_cfg["metric_name"]
        target_name = projection_cfg["target_name"]
        observation_name = projection_cfg["observation_name"]
        projection_path = projection_cfg["projection"]
        assignment_metric = projection_cfg["assignment_metric"]
        if not all(isinstance(value, str) and value for value in (metric_name, target_name, observation_name)):
            raise TypeError("Tracking metric, target, and observation names must be nonempty strings.")
        if projection_path is not None and (not isinstance(projection_path, str) or not projection_path):
            raise TypeError("Tracking projection must be a qualified callable path or None.")
        if not isinstance(assignment_metric, str):
            raise TypeError("Tracking assignment metric must be a string.")
        projection = None if projection_path is None else resolve_callable(projection_path)
        target = expert_observations.get(target_name)
        if target is None:
            raise ValueError(f"Expert observations do not contain tracking target {target_name!r}.")
        target = (target if projection is None else projection(target)).contiguous()
        metrics.append(_TrackingMetric(metric_name, target, observation_name, projection, assignment_metric))
    if not metrics or len({metric.name for metric in metrics}) != len(metrics):
        raise ValueError("Tracking metric names must be nonempty and unique.")
    if any(
        metric.target_frames.shape[0] != expert.frames.shape[0] or metric.target_frames.device != expert.device
        for metric in metrics
    ):
        raise ValueError("Tracking targets must align with float32 expert frames on device.")

    contexts = _tracking_context_table(model, expert, window_length=context_window_length)

    expert_length_values = expert.clip_length_values
    metric_length_values = tuple(length if include_reset_frame else length - 1 for length in expert_length_values)
    trace_offset = int(include_reset_frame)
    evaluated_length_values = (
        tuple(min(length, env.max_episode_length + trace_offset) for length in metric_length_values)
        if allow_horizon_truncation
        else metric_length_values
    )
    metric_frame_counts, evaluated_frame_counts = tracking_frame_counts(
        expert.clip_lengths,
        episode_length=env.max_episode_length,
        include_reset_frame=include_reset_frame,
        allow_horizon_truncation=allow_horizon_truncation,
    )

    observed_start_values = []
    observed_frame_count = 0
    for length in evaluated_length_values:
        observed_start_values.append(observed_frame_count)
        observed_frame_count += length
    observed_starts = torch.tensor(observed_start_values, dtype=torch.int64, device=expert.device)
    target_starts = expert.clip_offsets[:-1].clone()
    target_starts.add_(0 if include_reset_frame else 1)
    chunk_capacity = min(env.num_envs, len(sequence_ids))
    feature_width = max(metric.target_frames.shape[1] for metric in metrics)
    workspace = UniformAssignmentWorkspace(
        lengths=evaluated_length_values,
        device=expert.device,
        feature_width=feature_width,
    )
    outputs = tuple(torch.empty(len(sequence_ids), dtype=torch.float64, device=expert.device) for _ in metrics)
    observed_traces = tuple(
        torch.empty(
            observed_frame_count,
            metric.target_frames.shape[1],
            dtype=metric.target_frames.dtype,
            device=expert.device,
        )
        for metric in metrics
    )
    reached_values = tuple(
        torch.empty(
            env.num_envs,
            metric.target_frames.shape[1],
            dtype=metric.target_frames.dtype,
            device=expert.device,
        )
        for metric in metrics
    )
    representative_values = tuple(
        torch.empty(
            chunk_capacity,
            metric.target_frames.shape[1],
            dtype=metric.target_frames.dtype,
            device=expert.device,
        )
        for metric in metrics
    )
    expert_indices = torch.arange(len(sequence_ids), dtype=torch.int64, device=expert.device)
    unexpected_done = torch.empty(env.num_envs, dtype=torch.bool, device=expert.device)
    representative_mask = torch.empty(env.num_envs, dtype=torch.bool, device=expert.device)
    local_frame = torch.empty(env.num_envs, dtype=torch.int64, device=expert.device)
    context_indices = torch.empty_like(local_frame)
    context = torch.empty(env.num_envs, model.context_dim, dtype=expert.frames.dtype, device=expert.device)
    done = torch.empty(env.num_envs, dtype=torch.bool, device=expert.device)
    needed = torch.empty_like(done)
    required_final = torch.empty_like(done)
    premature = torch.empty_like(done)

    for chunk_start in range(0, len(sequence_ids), env.num_envs):
        chunk_count = min(env.num_envs, len(sequence_ids) - chunk_start)
        chunk_end = chunk_start + chunk_count
        chunk_expert = expert_indices[chunk_start:chunk_end]
        chunk_lengths = evaluated_frame_counts[chunk_start:chunk_end]
        assignment, representative_env_rows = _tracking_assignment(
            env.num_envs,
            chunk_count,
            expert.device,
            assignment_rng,
            shuffle=shuffle_assignments,
        )
        representative_mask.zero_()
        representative_mask[representative_env_rows] = True
        assigned_expert = chunk_expert.index_select(0, assignment)
        assigned_offsets = expert.clip_offsets.index_select(0, assigned_expert)
        assigned_raw_lengths = expert.clip_lengths.index_select(0, assigned_expert)
        assigned_evaluated_lengths = chunk_lengths.index_select(0, assignment)
        assigned_last_frames = assigned_raw_lengths - 1
        assigned_final_steps = assigned_evaluated_lengths - trace_offset - 1
        chunk_length_values = evaluated_length_values[chunk_start:chunk_end]
        max_chunk_frames = max(chunk_length_values)
        frame_counts = []
        frame_local_rows = []
        frame_destination_values = []
        for frame in range(max_chunk_frames):
            active = tuple(index for index, length in enumerate(chunk_length_values) if frame < length)
            frame_counts.append(len(active))
            padding = (0,) * (chunk_count - len(active))
            frame_local_rows.append((*active, *padding))
            frame_destination_values.append(
                tuple(observed_start_values[chunk_start + index] + frame for index in active) + padding
            )
        frame_local_rows_tensor = torch.tensor(frame_local_rows, dtype=torch.int64, device=expert.device)
        frame_source_rows = representative_env_rows.index_select(0, frame_local_rows_tensor.view(-1)).view(
            max_chunk_frames, chunk_count
        )
        frame_destination_rows = torch.tensor(frame_destination_values, dtype=torch.int64, device=expert.device)

        reset_result = reset_tracking_sequences(env, command, sequence_start_rows, assigned_expert)
        if (
            not isinstance(reset_result, tuple)
            or len(reset_result) != 2
            or not isinstance(reset_result[0], TensorDictBase)
            or not isinstance(reset_result[1], Mapping)
        ):
            raise TypeError("Tracking requires the RSL VecEnv reset TensorDict/info pair.")
        observations = reset_result[0]
        unexpected_done.zero_()
        history = history_factory(observations)
        if history is not None:
            observations = history.decorate_current(observations)
        if include_reset_frame:
            active_count = frame_counts[0]
            for observed, representative, metric in zip(
                observed_traces,
                representative_values,
                metrics,
                strict=True,
            ):
                torch.index_select(
                    metric.observe(observations),
                    0,
                    frame_source_rows[0, :active_count],
                    out=representative[:active_count],
                )
                observed.index_copy_(0, frame_destination_rows[0, :active_count], representative[:active_count])
        max_steps = max(evaluated_length_values[chunk_start:chunk_end]) - trace_offset

        for step in range(max_steps):
            local_frame.fill_(step + 1)
            torch.minimum(local_frame, assigned_last_frames, out=local_frame)
            torch.add(assigned_offsets, local_frame, out=context_indices)
            torch.index_select(contexts, 0, context_indices, out=context)
            action = model.action_sample(observations, context, deterministic=True)
            step_result = env.step(action)
            if not isinstance(step_result, tuple) or len(step_result) != 4:
                raise TypeError("Tracking requires the RSL VecEnv four-value step contract.")
            returned, _reward, dones, extras = step_result
            if not isinstance(returned, TensorDictBase):
                raise TypeError("Tracking requires TensorDict observations from the RSL VecEnv wrapper.")
            if not isinstance(dones, torch.Tensor) or dones.shape != (env.num_envs,) or dones.device != expert.device:
                raise ValueError("Tracking done flags must contain one row per environment on the expert device.")
            if not isinstance(extras, Mapping):
                raise TypeError("Tracking step extras must be a mapping.")
            torch.ne(dones, 0, out=done)
            torch.gt(assigned_evaluated_lengths, step + trace_offset, out=needed)
            needed.logical_and_(representative_mask)
            torch.logical_and(done, needed, out=required_final)
            final_observations = _tracking_final_observations(extras, required_final, env.num_envs)
            frame = step + trace_offset
            active_count = frame_counts[frame]
            for observed, representative, scratch, metric in zip(
                observed_traces,
                representative_values,
                reached_values,
                metrics,
                strict=True,
            ):
                _write_tracking_reached_values(
                    scratch,
                    metric,
                    returned,
                    final_observations,
                    required_final,
                )
                torch.index_select(
                    scratch,
                    0,
                    frame_source_rows[frame, :active_count],
                    out=representative[:active_count],
                )
                observed.index_copy_(0, frame_destination_rows[frame, :active_count], representative[:active_count])
            if history is not None:
                history.advance(observations, returned, done)
            observations = returned
            torch.gt(assigned_final_steps, step, out=premature)
            premature.logical_and_(done).logical_and_(representative_mask)
            unexpected_done.logical_or_(premature)

        torch._assert_async(
            torch.all(~unexpected_done),
            "A tracking rollout reset before its expert reference horizon ended.",
        )
    for observed, metric, output in zip(observed_traces, metrics, outputs, strict=True):
        workspace.compute_flat(observed, observed_starts, metric.target_frames, target_starts, output)
    if expert.device.type == "cuda":
        torch.cuda.synchronize(expert.device)
    return ForwardBackwardTrackingEvaluation(
        sequence_ids=sequence_ids,
        metric_values={metric.name: output for metric, output in zip(metrics, outputs, strict=True)},
        source_frame_counts=metric_frame_counts,
        evaluated_frame_counts=evaluated_frame_counts,
        coverage_fraction=evaluated_frame_counts.to(torch.float64) / metric_frame_counts,
        duration_seconds=time.perf_counter() - started,
    )


def _tracking_context_table(
    model: _TrackingModel,
    expert: ForwardBackwardExpertBuffer,
    *,
    window_length: int,
    inference_batch_size: int = 8192,
) -> torch.Tensor:
    """Materialize sequence-safe rolling contexts once from immutable expert frames."""
    clip_lengths = expert.clip_length_values
    if any(length < 2 for length in clip_lengths) or window_length < 1 or inference_batch_size < 1:
        raise ValueError("Tracking contexts require sequences of at least two frames and positive inference sizes.")
    frame_count = expert.frames.shape[0]
    backward = torch.empty(
        frame_count,
        model.context_dim,
        dtype=expert.frames.dtype,
        device=expert.device,
    )
    for start in range(0, frame_count, inference_batch_size):
        end = min(start + inference_batch_size, frame_count)
        backward[start:end].copy_(model.backward_map(expert_frame_tensordict(model, expert.frames[start:end])))

    contexts = torch.empty_like(backward)
    offset = 0
    radius = model.context_dim**0.5 if model.context_normalization else None
    for length in clip_lengths:
        end = offset + length
        sequence_contexts = trajectory_context_sequence(
            backward[offset:end],
            window_length,
            include_partial=True,
            radius=radius,
        )
        contexts[offset:end].copy_(sequence_contexts)
        offset = end
    return contexts


def forward_backward_tracking_priorities(
    evaluation: ForwardBackwardTrackingEvaluation,
    sequence_ids: tuple[str, ...],
    device: str | torch.device,
    *,
    metric_name: str,
    metric_minimum: float,
    metric_maximum: float,
    exponent_scale: float,
    exponent_base: float,
) -> torch.Tensor:
    """Map one named metric to exponential priorities in stable sequence order."""
    if evaluation.sequence_ids != sequence_ids:
        raise ValueError("Tracking evaluation must retain the configured stable sequence order.")
    if metric_name not in evaluation.metric_values:
        raise ValueError(f"Tracking evaluation does not contain priority metric {metric_name!r}.")
    if not all(math.isfinite(value) for value in (metric_minimum, metric_maximum, exponent_scale, exponent_base)):
        raise ValueError("Tracking priority parameters must be finite.")
    if metric_minimum > metric_maximum or exponent_base <= 0.0:
        raise ValueError("Tracking priority bounds or exponent base are invalid.")
    metric = evaluation.metric_values[metric_name]
    ordered = metric.to(device=device, dtype=torch.float32)
    return torch.pow(exponent_base, torch.clamp(ordered, min=metric_minimum, max=metric_maximum) * exponent_scale)


def _resolve_binding(expression: str, values: Mapping[str, object]) -> object:
    """Evaluate one required binding at the connector boundary."""
    if not isinstance(expression, str) or not expression:
        raise TypeError("Forward-backward curriculum bindings must be nonempty expressions.")
    return eval(expression, {}, dict(values))  # noqa: S307


class ForwardBackwardTrackingCurriculum:
    """Evaluate sequence tracking and update the shared motion-sampling law."""

    def __init__(
        self,
        env: VecEnv,
        algorithm: _TrackingAlgorithm,
        device: str,
        *,
        command_bind: str,
        projections: tuple[Mapping[str, object], ...],
        context_window_length: int,
        include_reset_frame: bool,
        allow_horizon_truncation: bool,
        shuffle_assignments: bool,
        priority_metric_name: str,
        priority_metric_minimum: float,
        priority_metric_maximum: float,
        priority_exponent_scale: float,
        priority_exponent_base: float,
        reset_source_name: str | None,
        evaluation_seed: int = 0,
    ) -> None:
        """Bind one motion command and the learner state that shares its sampling weights."""
        command = _resolve_binding(command_bind, {"env": env, "algorithm": algorithm})
        table = getattr(command, "table", None)
        payload = getattr(command, "payload", None)
        sampler = getattr(payload, "sampler", None)
        expert = algorithm.expert
        sequence_ids = getattr(table, "clip_ids", None)
        sequence_start_rows = getattr(table, "clip_start_rows", None)
        sampling_priorities = getattr(sampler, "clip_priorities", None)
        evaluation_scope = getattr(sampler, "reset_sampling_scope", None)
        if not callable(getattr(command, "bind_rows", None)) or not isinstance(
            getattr(command, "randomize_command_indices", None), bool
        ):
            raise TypeError("Command binding must resolve to the exact sequence-reset protocol.")
        if not isinstance(expert, ForwardBackwardExpertBuffer):
            raise TypeError("Tracking curriculum requires ForwardBackwardExpertBuffer.")
        if not isinstance(sequence_ids, tuple) or any(
            not isinstance(value, str) or not value for value in sequence_ids
        ):
            raise TypeError("The bound command table must expose a tuple of nonempty clip ids.")
        if sequence_ids != expert.clip_ids:
            raise ValueError("Environment and expert stable sequence ids must match exactly.")
        if not isinstance(sequence_start_rows, torch.Tensor):
            raise TypeError("The bound command table must expose tensor clip start rows.")
        if not isinstance(sampling_priorities, torch.Tensor):
            raise TypeError("The bound command sampler must expose tensor clip priorities.")
        if sampling_priorities.data_ptr() != expert.priorities.data_ptr():
            raise ValueError("The environment sampler and expert buffer must share one canonical priority tensor.")
        if not callable(evaluation_scope):
            raise TypeError("The bound command sampler must expose reset_sampling_scope().")
        if not callable(getattr(algorithm, "evaluation_history", None)):
            raise TypeError("Tracking curriculum requires the learner evaluation_history() factory.")
        if not callable(getattr(algorithm, "eval_mode", None)) or not callable(getattr(algorithm, "train_mode", None)):
            raise TypeError("Tracking curriculum requires learner-owned eval_mode() and train_mode().")
        if (
            sequence_start_rows.shape != (len(sequence_ids),)
            or sequence_start_rows.dtype is not torch.int64
            or sequence_start_rows.device != expert.device
        ):
            raise ValueError("Sequence start rows must align with stable ids on the expert device.")
        expert.validate_priorities(expert.priorities)
        if type(evaluation_seed) is not int:
            raise TypeError("Tracking evaluation_seed must be an integer.")

        resolved_projections = tuple(dict(projection) for projection in projections)
        self.env = env
        self.algorithm = algorithm
        self.device = device
        self.command = command
        self.expert = expert
        self.sequence_ids = sequence_ids
        self.sequence_start_rows = sequence_start_rows
        self.evaluation_scope = evaluation_scope
        self.projections = resolved_projections
        self.context_window_length = context_window_length
        self.include_reset_frame = include_reset_frame
        self.allow_horizon_truncation = allow_horizon_truncation
        self.shuffle_assignments = shuffle_assignments
        self.priority_metric_name = priority_metric_name
        self.priority_metric_minimum = priority_metric_minimum
        self.priority_metric_maximum = priority_metric_maximum
        self.priority_exponent_scale = priority_exponent_scale
        self.priority_exponent_base = priority_exponent_base
        self.reset_source_name = reset_source_name
        self.evaluation_seed = evaluation_seed
        self.assignment_rng = random.Random(evaluation_seed)
        self.protocol_hash = get_forward_backward_schema_hash(
            {
                "allow_horizon_truncation": allow_horizon_truncation,
                "command_bind": command_bind,
                "context_window_length": context_window_length,
                "evaluation_seed": evaluation_seed,
                "expert_schema_hash": expert.schema.schema_hash,
                "include_reset_frame": include_reset_frame,
                "priority_exponent_base": priority_exponent_base,
                "priority_exponent_scale": priority_exponent_scale,
                "priority_metric_maximum": priority_metric_maximum,
                "priority_metric_minimum": priority_metric_minimum,
                "priority_metric_name": priority_metric_name,
                "projections": resolved_projections,
                "reset_source_name": reset_source_name,
                "sequence_ids": sequence_ids,
                "sequence_start_rows": sequence_start_rows.detach().cpu().tolist(),
                "shuffle_assignments": shuffle_assignments,
                "version": 2,
            }
        )

    def update(self) -> TensorDictBase:
        """Evaluate tracking, update canonical priorities, and reset collection state."""
        assignment_rng_state = self.assignment_rng.getstate()
        try:
            self.algorithm.eval_mode()
            try:
                with forward_backward_evaluation_scope(
                    self.env,
                    self.command,
                    self.evaluation_scope,
                    self.evaluation_seed,
                    reset_source_name=self.reset_source_name,
                ):
                    evaluation = forward_backward_tracking_evaluator(
                        self.algorithm.model,
                        self.env,
                        self.expert,
                        self.sequence_ids,
                        command=self.command,
                        history_factory=self.algorithm.evaluation_history,
                        sequence_start_rows=self.sequence_start_rows,
                        projections=self.projections,
                        context_window_length=self.context_window_length,
                        include_reset_frame=self.include_reset_frame,
                        allow_horizon_truncation=self.allow_horizon_truncation,
                        shuffle_assignments=self.shuffle_assignments,
                        assignment_rng=self.assignment_rng,
                    )
            finally:
                self.algorithm.train_mode()
            priorities = forward_backward_tracking_priorities(
                evaluation,
                self.sequence_ids,
                self.expert.device,
                metric_name=self.priority_metric_name,
                metric_minimum=self.priority_metric_minimum,
                metric_maximum=self.priority_metric_maximum,
                exponent_scale=self.priority_exponent_scale,
                exponent_base=self.priority_exponent_base,
            )
            self.expert.validate_priorities(priorities)
        except BaseException:
            self.assignment_rng.setstate(assignment_rng_state)
            raise

        self.expert.set_priorities(priorities)
        reset_result = self.env.reset()
        if (
            not isinstance(reset_result, tuple)
            or len(reset_result) != 2
            or not isinstance(reset_result[0], TensorDictBase)
            or not isinstance(reset_result[1], Mapping)
        ):
            raise TypeError("Environment reset must return the RSL VecEnv TensorDict/info pair.")
        reset_observations, _reset_info = reset_result
        reset = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.algorithm.process_env_reset(reset_observations, reset)
        return reset_observations

    def state_dict(self) -> dict[str, object]:
        """Return curriculum protocol and assignment RNG state for exact resume."""
        return {
            "protocol_hash": self.protocol_hash,
            "sequence_ids": self.sequence_ids,
            "assignment_rng_state": self.assignment_rng.getstate(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore assignment RNG state under the same curriculum protocol."""
        expected = {"protocol_hash", "sequence_ids", "assignment_rng_state"}
        if set(state_dict) != expected:
            raise ValueError("Tracking curriculum checkpoint fields do not match the current contract.")
        if state_dict["protocol_hash"] != self.protocol_hash:
            raise ValueError("Tracking curriculum checkpoint protocol does not match the current configuration.")
        if state_dict["sequence_ids"] != self.sequence_ids:
            raise ValueError("Tracking curriculum sequence ids do not match the environment.")
        assignment_rng = random.Random()
        assignment_rng.setstate(state_dict["assignment_rng_state"])
        self.assignment_rng.setstate(assignment_rng.getstate())


class ForwardBackwardTrackingRunner(OffPolicyRunner):
    """Run off-policy FB learning with an optional sequence-tracking curriculum."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the learner and its declared tracking curriculum."""
        super().__init__(env, train_cfg, log_dir, device)
        self.tracking_curriculum: ForwardBackwardTrackingCurriculum | None = None
        self.tracking_interval_transitions: int | None = None
        self._tracking_last_transition: int | None = None
        configured = self.cfg["tracking_curriculum"]
        if configured is None:
            return
        curriculum_cfg = dict(configured)
        interval = curriculum_cfg.pop("interval_transitions")
        if type(interval) is not int:
            raise TypeError("Tracking curriculum interval_transitions must be an integer.")
        collection_block = self.env.num_envs * int(self.cfg["num_steps_per_env"])
        if interval < 1 or interval % collection_block:
            raise ValueError("Tracking curriculum interval must be a positive multiple of one collection block.")
        self.tracking_curriculum = ForwardBackwardTrackingCurriculum(self.env, self.alg, self.device, **curriculum_cfg)
        self.tracking_interval_transitions = interval

    def _run_tracking_curriculum(self, transition: int, observations: TensorDictBase | None) -> TensorDictBase:
        """Run one due curriculum update and return its reset observations."""
        curriculum = self.tracking_curriculum
        interval = self.tracking_interval_transitions
        if curriculum is None or interval is None:
            raise RuntimeError("Tracking curriculum work requires a configured curriculum.")
        if transition and transition % interval:
            if observations is None:
                raise ValueError("Periodic tracking requires current observations when no update is due.")
            return observations
        if self._tracking_last_transition is not None and transition <= self._tracking_last_transition:
            if transition == self._tracking_last_transition and observations is not None:
                return observations
            raise RuntimeError("Tracking curriculum transitions must increase monotonically.")
        reset_observations = curriculum.update()
        if tuple(reset_observations.batch_size) != (self.env.num_envs,):
            raise ValueError("Tracking reset observations must contain one row per environment.")
        self._tracking_last_transition = transition
        return reset_observations.to(self.device)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run transition-zero tracking once, then use the ordinary off-policy loop."""
        if self.tracking_curriculum is not None and self._tracking_last_transition is None:
            self._run_tracking_curriculum(0, None)
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def _update(self, observations: TensorDictBase) -> tuple[TensorDictBase, list[dict[str, torch.Tensor]]]:
        """Update the learner, then run tracking at due transition counts."""
        observations, metrics = super()._update(observations)
        if self.tracking_curriculum is not None:
            observations = self._run_tracking_curriculum(self.collected_transitions, observations)
        return observations, metrics

    def state_dict(self) -> dict[str, object]:
        """Extend the ordinary runner checkpoint with curriculum cadence and RNG state."""
        state_dict = super().state_dict()
        state_dict["tracking_curriculum"] = (
            None
            if self.tracking_curriculum is None
            else {
                "last_transition": self._tracking_last_transition,
                "state_dict": self.tracking_curriculum.state_dict(),
            }
        )
        return state_dict

    def load_state_dict(
        self,
        state_dict: dict[str, object],
        load_cfg: dict | None = None,
        strict: bool = True,
    ) -> None:
        """Restore learner state followed by the matching tracking curriculum state."""
        super().load_state_dict(state_dict, load_cfg, strict)
        tracking_state = state_dict.get("tracking_curriculum")
        curriculum = self.tracking_curriculum
        interval = self.tracking_interval_transitions
        if curriculum is None:
            if tracking_state is not None:
                raise ValueError("Checkpoint tracking curriculum differs from the configured runner.")
            return
        if interval is None or not isinstance(tracking_state, dict):
            raise ValueError("Checkpoint is missing configured tracking curriculum state.")
        last_transition = tracking_state.get("last_transition")
        if last_transition is not None:
            if type(last_transition) is not int or last_transition < 0:
                raise ValueError("Tracking curriculum last_transition must be null or a non-negative integer.")
            if last_transition and last_transition % interval:
                raise ValueError("Checkpoint tracking transition is outside the configured cadence.")
            if last_transition > self.collected_transitions:
                raise ValueError("Checkpoint tracking transition exceeds collected transitions.")
        elif self.collected_transitions:
            raise ValueError("A checkpoint without a tracking event cannot contain transitions.")
        curriculum_state = tracking_state.get("state_dict")
        if not isinstance(curriculum_state, dict):
            raise TypeError("Checkpoint tracking curriculum state_dict must be a dictionary.")
        curriculum.load_state_dict(curriculum_state)
        self._tracking_last_transition = last_transition
