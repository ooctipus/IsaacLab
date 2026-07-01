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
from rsl_rl.runners import RunnerLifecycleExtension
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict, TensorDictBase

from .frames import G1_HEAD_FRAME_NAME
from .impl.uniform_emd_warp import uniform_assignment_cost
from .mdp.actions import MotionJointPositionAction
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
    obs_state_emd: torch.Tensor
    source_frame_counts: torch.Tensor
    evaluated_frame_counts: torch.Tensor
    coverage_fraction: torch.Tensor
    duration_seconds: float

    def __post_init__(self) -> None:
        """Reject incomplete, duplicate, or nonfinite evaluation results."""
        if len(self.clip_ids) == 0 or len(set(self.clip_ids)) != len(self.clip_ids):
            raise ValueError("Tracking evaluation clip_ids must be nonempty and unique.")
        count = len(self.clip_ids)
        floating = (self.emd, self.obs_state_emd, self.coverage_fraction)
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
        return {
            clip_id: {
                "emd": self.emd[index],
                "obs_state_emd": self.obs_state_emd[index],
                "num_frames": self.evaluated_frame_counts[index],
                "source_num_frames": self.source_frame_counts[index],
                "evaluated_num_frames": self.evaluated_frame_counts[index],
                "coverage_fraction": self.coverage_fraction[index],
            }
            for index, clip_id in enumerate(self.clip_ids)
        }

    def serializable_metrics(self) -> dict[str, dict[str, float | int]]:
        """Copy final scalar rows once at the JSON artifact boundary."""
        rows = torch.stack(
            (
                self.emd,
                self.obs_state_emd,
                self.source_frame_counts.to(torch.float64),
                self.evaluated_frame_counts.to(torch.float64),
                self.coverage_fraction,
            ),
            dim=-1,
        ).cpu()
        return {
            clip_id: {
                "emd": float(row[0]),
                "obs_state_emd": float(row[1]),
                "num_frames": int(row[3]),
                "source_num_frames": int(row[2]),
                "evaluated_num_frames": int(row[3]),
                "coverage_fraction": float(row[4]),
            }
            for clip_id, row in zip(self.clip_ids, rows, strict=True)
        }


class _UniformEmdWorkspace:
    """Fixed GPU scratch for exact variable-length uniform assignment."""

    def __init__(self, capacity: int, max_frames: int, device: torch.device) -> None:
        if capacity < 1 or max_frames < 1 or device.type != "cuda":
            raise ValueError("GPU EMD requires positive capacity and frame count on CUDA.")
        wp.init()
        self.capacity = capacity
        self.max_frames = max_frames
        self.device = device
        self.cost = torch.empty(capacity, max_frames, max_frames, dtype=torch.float32, device=device)
        self.observed_norm = torch.empty(capacity, max_frames, 1, dtype=torch.float32, device=device)
        self.target_norm = torch.empty_like(self.observed_norm)
        self.observed_square = torch.empty(capacity, max_frames, 29, dtype=torch.float32, device=device)
        self.target_square = torch.empty_like(self.observed_square)
        scratch_shape = (capacity, max_frames + 1)
        self.potential_rows = torch.empty(scratch_shape, dtype=torch.float64, device=device)
        self.potential_columns = torch.empty_like(self.potential_rows)
        self.matching = torch.empty(scratch_shape, dtype=torch.int32, device=device)
        self.previous = torch.empty_like(self.matching)
        self.minimum = torch.empty_like(self.potential_rows)
        self.used = torch.empty_like(self.matching)

    def compute(
        self,
        observed: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Write exact uniform transport costs without allocating assignment scratch."""
        batch_size = observed.shape[0]
        expected_prefix = (batch_size, self.max_frames)
        if (
            observed.ndim != 3
            or target.shape != observed.shape
            or observed.shape[:2] != expected_prefix
            or observed.dtype is not torch.float32
            or target.dtype is not torch.float32
            or observed.device != self.device
            or target.device != self.device
            or not observed.is_contiguous()
            or not target.is_contiguous()
            or lengths.shape != (batch_size,)
            or lengths.dtype is not torch.int64
            or lengths.device != self.device
            or output.shape != (batch_size,)
            or output.dtype is not torch.float64
            or output.device != self.device
            or batch_size < 1
            or batch_size > self.capacity
        ):
            raise ValueError("GPU EMD inputs do not match the fixed workspace contract.")
        torch._assert_async(
            torch.all((lengths > 0) & (lengths <= self.max_frames)),
            "GPU EMD lengths must be within the fixed frame capacity.",
        )
        cost = self.cost[:batch_size]
        observed_norm = self.observed_norm[:batch_size]
        target_norm = self.target_norm[:batch_size]
        feature_width = observed.shape[2]
        observed_square = self.observed_square[:batch_size, :, :feature_width]
        target_square = self.target_square[:batch_size, :, :feature_width]
        torch.square(observed, out=observed_square)
        torch.square(target, out=target_square)
        torch.sum(observed_square, dim=-1, keepdim=True, out=observed_norm)
        torch.sum(target_square, dim=-1, keepdim=True, out=target_norm)
        torch.bmm(observed, target.mT, out=cost)
        cost.mul_(-2.0)
        cost.add_(observed_norm)
        cost.add_(target_norm.mT)
        cost.clamp_min_(0.0).sqrt_()
        wp.launch(
            uniform_assignment_cost,
            dim=batch_size,
            inputs=[
                wp.from_torch(self.cost),
                wp.from_torch(lengths),
                wp.from_torch(self.potential_rows),
                wp.from_torch(self.potential_columns),
                wp.from_torch(self.matching),
                wp.from_torch(self.previous),
                wp.from_torch(self.minimum),
                wp.from_torch(self.used),
                wp.from_torch(output),
            ],
            stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)),
        )


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
    maximal prefix. This keeps the metric free of hidden reset discontinuities;
    each record exposes source/evaluated frame counts and coverage explicitly.
    """
    started = time.perf_counter()
    if not isinstance(expert, ForwardBackwardExpertBuffer):
        raise TypeError("Motion tracking requires the already-built ForwardBackwardExpertBuffer.")
    if expert.clip_ids != clip_ids:
        raise ValueError("Tracking clip ids must match the expert buffer in stable order.")
    if not clip_ids:
        raise ValueError("Motion tracking requires at least one clip.")
    if env.num_envs < 1 or not callable(getattr(env, "reset_motion_clips", None)):
        raise TypeError("Motion tracking requires a vector environment with exact clip resets.")

    command = env.command_manager.get_term("motion")
    table = command.table
    payload = command.payload
    if not isinstance(table, MotionTaskTable) or not isinstance(payload, MotionStatePayload):
        raise TypeError("Motion tracking requires MotionTaskTable and MotionStatePayload.")
    if payload.table is not table:
        raise ValueError("The tracking payload and command must share one trajectory table.")
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

    source_positions = {clip_id: index for index, clip_id in enumerate(table.source_clip_ids)}
    if any(clip_id not in source_positions for clip_id in clip_ids):
        raise ValueError("Every tracking clip must exist in the environment trajectory table.")
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
    max_frames = env.cfg.commands.motion.payload.episode_length_steps
    source_frame_counts = expert.clip_lengths.index_select(0, clip_expert_indices)
    evaluated_frame_counts = source_frame_counts.clamp(max=max_frames)
    torch._assert_async(
        torch.all(evaluated_frame_counts > 1),
        "Tracking clips must contain at least two expert frames.",
    )
    emd_values = torch.empty(len(clip_ids), dtype=torch.float64, device=expert.device)
    state_emd_values = torch.empty_like(emd_values)
    workspace = _UniformEmdWorkspace(
        capacity=min(env.num_envs, len(clip_ids)),
        max_frames=max_frames,
        device=expert.device,
    )
    frame_grid = torch.arange(max_frames, dtype=torch.int64, device=expert.device)
    chunk_capacity = min(env.num_envs, len(clip_ids))
    joint_trace = torch.empty(env.num_envs, max_frames, 29, device=expert.device)
    state_trace = torch.empty(env.num_envs, max_frames, 23, device=expert.device)
    representative_joint_trace = torch.empty(chunk_capacity, max_frames, 29, device=expert.device)
    representative_state_trace = torch.empty(chunk_capacity, max_frames, 23, device=expert.device)
    target_indices = torch.empty(chunk_capacity, max_frames, dtype=torch.int64, device=expert.device)
    target_relative_joint = torch.empty(chunk_capacity, max_frames, 29, device=expert.device)
    target_joint = torch.empty_like(target_relative_joint)
    target_state = torch.empty(chunk_capacity, max_frames, 23, device=expert.device)
    unexpected_done = torch.empty(env.num_envs, dtype=torch.bool, device=expert.device)

    for chunk_start in range(0, len(clip_ids), env.num_envs):
        chunk_ids = clip_ids[chunk_start : chunk_start + env.num_envs]
        chunk_count = len(chunk_ids)
        chunk_end = chunk_start + chunk_count
        chunk_expert = clip_expert_indices[chunk_start:chunk_end]
        chunk_lengths = evaluated_frame_counts[chunk_start:chunk_end]
        chunk_source = clip_source_indices[chunk_start:chunk_end]
        assignment, representative_env_rows = _native_tracking_assignment(env.num_envs, chunk_count, expert.device)
        assigned_expert = chunk_expert.index_select(0, assignment)
        assigned_source = chunk_source.index_select(0, assignment)

        reset_result = env.reset_motion_clips(assigned_source)
        observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        observations = _as_tensordict(observations, env.num_envs)
        assigned_offsets = expert.clip_offsets.index_select(0, assigned_expert)
        assigned_lengths = expert.clip_lengths.index_select(0, assigned_expert)
        evaluation_lengths = assigned_lengths.clamp(max=max_frames)
        context_lengths = evaluation_lengths - 1
        torch._assert_async(
            torch.all(context_lengths > 0),
            "Tracking clips must contain at least two expert frames.",
        )
        max_steps = int(chunk_lengths.max().item()) - 1

        joint_trace[:, 0].copy_(action_term.joint_position)
        state_trace[:, 0].copy_(observations["state"][:, :23])
        unexpected_done.zero_()

        for step in range(max_steps):
            local_frame = torch.remainder(step, context_lengths).add_(1)
            frame_indices = assigned_offsets + local_frame
            target_frames = expert.frames.index_select(0, frame_indices)
            target_observations = _expert_frame_tensordict(model, target_frames)
            context = model.context_project(model.backward_map(target_observations))
            action = model.action_sample(observations, context, deterministic=True)
            observations, _reward, terminated, truncated, _extras = env.step(action)
            observations = _as_tensordict(observations, env.num_envs)
            joint_trace[:, step + 1].copy_(action_term.joint_position)
            state_trace[:, step + 1].copy_(observations["state"][:, :23])
            active = step < context_lengths
            unexpected_done.logical_or_((terminated | truncated) & active)

        torch._assert_async(
            torch.all(~unexpected_done),
            "A tracking rollout reset before its expert reference horizon ended.",
        )
        chunk_offsets = expert.clip_offsets.index_select(0, chunk_expert)
        chunk_target_indices = target_indices[:chunk_count]
        torch.minimum(
            frame_grid.unsqueeze(0),
            chunk_lengths.unsqueeze(1) - 1,
            out=chunk_target_indices,
        )
        chunk_target_indices.add_(chunk_offsets.unsqueeze(1))
        chunk_target_relative = target_relative_joint[:chunk_count]
        torch.index_select(
            expert.frames[:, :29],
            0,
            chunk_target_indices.view(-1),
            out=chunk_target_relative.view(-1, 29),
        )
        chunk_target_joint = target_joint[:chunk_count]
        chunk_target_joint.copy_(chunk_target_relative).add_(default_joint_position)
        chunk_target_state = target_state[:chunk_count]
        chunk_target_state.copy_(chunk_target_relative[:, :, :23])
        chunk_joint_trace = representative_joint_trace[:chunk_count]
        chunk_state_trace = representative_state_trace[:chunk_count]
        torch.index_select(joint_trace, 0, representative_env_rows, out=chunk_joint_trace)
        torch.index_select(state_trace, 0, representative_env_rows, out=chunk_state_trace)
        workspace.compute(
            chunk_joint_trace,
            chunk_target_joint,
            chunk_lengths,
            emd_values[chunk_start:chunk_end],
        )
        workspace.compute(
            chunk_state_trace,
            chunk_target_state,
            chunk_lengths,
            state_emd_values[chunk_start:chunk_end],
        )

    torch.cuda.synchronize(expert.device)
    coverage = evaluated_frame_counts.to(torch.float64) / source_frame_counts
    return MotionTrackingEvaluation(
        clip_ids=clip_ids,
        emd=emd_values,
        obs_state_emd=state_emd_values,
        source_frame_counts=source_frame_counts,
        evaluated_frame_counts=evaluated_frame_counts,
        coverage_fraction=coverage,
        duration_seconds=time.perf_counter() - started,
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
        previous_table_active = self.table.clip_priorities_active
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
            if previous_table_active:
                self.table.set_clip_priorities(self.clip_ids, previous_table)
            else:
                self.table.reset_clip_priorities()
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
            "table_priorities_active": self.table.clip_priorities_active,
            "expert_priorities": self.expert.priorities.clone(),
            "applied_transitions": tuple(self.applied_transitions),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore exact sampler priorities and completed event history."""
        if state.get("clip_ids") != self.clip_ids:
            raise ValueError("Tracking curriculum checkpoint clip ids do not match the environment.")
        table_priorities = state.get("table_priorities")
        expert_priorities = state.get("expert_priorities")
        table_priorities_active = state.get("table_priorities_active")
        transitions = state.get("applied_transitions")
        if not isinstance(table_priorities, torch.Tensor) or not isinstance(expert_priorities, torch.Tensor):
            raise TypeError("Tracking curriculum checkpoint priorities must be tensors.")
        if not isinstance(table_priorities_active, bool):
            raise TypeError("Tracking curriculum table priority activation must be a boolean.")
        if not isinstance(transitions, tuple) or not all(isinstance(value, int) for value in transitions):
            raise TypeError("Tracking curriculum applied_transitions must be a tuple of integers.")
        self.table.validate_clip_priorities(self.clip_ids, table_priorities)
        self.expert.validate_priorities(expert_priorities)
        if not torch.equal(table_priorities.to(expert_priorities), expert_priorities):
            raise ValueError("Tracking curriculum table and expert priorities must match exactly.")
        if table_priorities_active:
            self.table.set_clip_priorities(self.clip_ids, table_priorities)
        else:
            if not torch.equal(table_priorities, torch.ones_like(table_priorities)):
                raise ValueError("Inactive table priorities must be uniform.")
            self.table.reset_clip_priorities()
        self.expert.set_priorities(expert_priorities)
        self.applied_transitions = list(transitions)


__all__ = [
    "MotionTrackingCurriculum",
    "MotionTrackingEvaluation",
    "MotionTrackingEvaluator",
    "g1_motion_tracking_evaluator",
    "motion_tracking_priorities",
]
