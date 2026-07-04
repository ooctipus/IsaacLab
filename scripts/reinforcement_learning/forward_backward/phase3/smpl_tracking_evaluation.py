# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SMPL tracking orchestration over shared rollout and robot tensor projections."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from typing import Any, Literal

import torch
from rsl_rl.modules.forward_backward import trajectory_context_sequence
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer
from tensordict import TensorDict, TensorDictBase

from isaaclab_tasks.core.multi_task.mdp.native_mujoco_action import NativeMujocoControlAction
from isaaclab_tasks.core.multi_task.metrics import UniformAssignmentWorkspace
from isaaclab_tasks.core.multi_task.motion.mdp.commands import MotionStatePayload, MotionTaskTable
from isaaclab_tasks.core.multi_task.motion.robots.smpl.observations import smpl_humenv_tracking_pose
from isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_tracking import (
    ForwardBackwardTrackingEvaluation,
    forward_backward_tracking_evaluator,
    reset_tracking_sequences,
    tracking_frame_counts,
)


@torch.no_grad()
def smpl_motion_tracking_evaluator(
    model: Any,
    env: Any,
    expert: ForwardBackwardExpertBuffer,
    clip_ids: tuple[str, ...],
    *,
    sampling_mode: Literal["source_rows", "uniform_before_source_end"],
    sampling_step_seconds: float | None,
    evaluation_seed: int = 0,
) -> ForwardBackwardTrackingEvaluation:
    """Evaluate the native held-out HumEnv pose metric without autoreset truncation.

    The released protocol resets to source frame zero, applies one deterministic
    action for every later source frame, and computes exact uniform transport
    over the first 214 values of those reached and target observations. Contexts
    are clip-safe means of the next up-to-eight backward features.
    """
    if sampling_mode != "source_rows" or sampling_step_seconds is not None:
        raise ValueError("Faithful SMPL tracking requires literal source-row sampling with no sample step.")
    command = env.unwrapped.command_manager.get_term("motion")
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
    action = env.unwrapped.action_manager.get_term("control")
    if not isinstance(action, NativeMujocoControlAction) or action.action_dim != 69:
        raise TypeError("SMPL tracking requires the native 69-wide MuJoCo control action.")

    sequence_start_rows = table.clip_start_rows
    return forward_backward_tracking_evaluator(
        model,
        env,
        expert,
        clip_ids,
        command=command,
        history_factory=lambda _observations: None,
        sequence_start_rows=sequence_start_rows,
        projections=(
            {
                "metric_name": "emd",
                "target_name": "policy",
                "observation_name": "policy",
                "projection": (
                    "isaaclab_tasks.core.multi_task.motion.robots.smpl.observations:smpl_humenv_tracking_pose"
                ),
                "assignment_metric": "uniform_assignment",
            },
        ),
        context_window_length=8,
        include_reset_frame=False,
        allow_horizon_truncation=False,
        shuffle_assignments=False,
        assignment_rng=random.Random(evaluation_seed),
    )


def smpl_motion_tracking_evaluator_packed(
    model: Any,
    env: Any,
    expert: ForwardBackwardExpertBuffer,
    clip_ids: tuple[str, ...],
    *,
    policy_count: int,
    sampling_mode: Literal["source_rows", "uniform_before_source_end"],
    sampling_step_seconds: float | None,
    evaluation_seed: int = 0,
) -> tuple[ForwardBackwardTrackingEvaluation, ...]:
    """Evaluate checkpoint-major SMPL policies with one full-reset rollout.

    Every policy receives one lane per clip. This fixed layout avoids a second,
    partial environment-reset lifecycle while preserving checkpoint-major GPU
    inference and exact per-clip uniform assignment.
    """
    if sampling_mode != "source_rows" or sampling_step_seconds is not None:
        raise ValueError("Packed SMPL tracking requires literal source-row sampling with no sample step.")
    started = time.perf_counter()
    if (
        type(policy_count) is not int
        or policy_count < 1
        or type(evaluation_seed) is not int
        or getattr(model, "policy_count", None) != policy_count
        or not isinstance(expert, ForwardBackwardExpertBuffer)
        or expert.clip_ids != clip_ids
    ):
        raise ValueError("Packed SMPL tracking requires aligned policies, expert storage, and stable clip ids.")
    clip_count = len(clip_ids)
    if env.num_envs != policy_count * clip_count:
        raise ValueError("Packed SMPL tracking requires exactly one environment lane per policy and clip.")
    for name in ("backward_map", "context_project", "action_deterministic"):
        if not callable(getattr(model, name, None)):
            raise TypeError(f"Packed SMPL model must expose {name}().")

    command = env.unwrapped.command_manager.get_term("motion")
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
    action = env.unwrapped.action_manager.get_term("control")
    if not isinstance(action, NativeMujocoControlAction) or action.action_dim != 69:
        raise TypeError("Packed SMPL tracking requires the native 69-wide MuJoCo control action.")

    if table.clip_ids != clip_ids:
        raise ValueError("Every packed tracking sequence must align with the environment table.")
    expert_length_values = expert.clip_length_values
    metric_length_values = tuple(length - 1 for length in expert_length_values)
    raw_frame_counts = expert.clip_lengths
    expected_raw_frame_counts = torch.tensor(expert_length_values, dtype=torch.int64, device=expert.device)
    torch._assert_async(
        torch.all(raw_frame_counts == expected_raw_frame_counts),
        "Packed source metadata and expert clip lengths differ.",
    )
    metric_frame_counts, evaluated_frame_counts = tracking_frame_counts(
        raw_frame_counts,
        episode_length=env.max_episode_length,
        include_reset_frame=False,
        allow_horizon_truncation=False,
    )
    contexts = tracking_context_table_packed(
        model,
        expert,
        expert_length_values,
        policy_count=policy_count,
        window_length=8,
    )

    max_frames = max(metric_length_values)
    clip_offsets = expert.clip_offsets[:-1]
    metric_lengths = metric_frame_counts
    frame_grid = torch.arange(max_frames, dtype=torch.int64, device=expert.device)
    local_frames = torch.minimum(frame_grid.unsqueeze(0), metric_lengths.unsqueeze(1) - 1)
    expert_frame_indices = clip_offsets.unsqueeze(1) + local_frames + 1
    target = smpl_humenv_tracking_pose(expert.frames.index_select(0, expert_frame_indices.reshape(-1)))
    target = target.view(clip_count, max_frames, 214)
    target = target.unsqueeze(0).expand(policy_count, -1, -1, -1).contiguous()
    observed = torch.empty_like(target)
    reached = torch.empty(env.num_envs, 214, dtype=torch.float32, device=expert.device)
    context_values = torch.empty(
        policy_count,
        clip_count,
        model.context_dim,
        dtype=expert.frames.dtype,
        device=expert.device,
    )
    unexpected_done = torch.zeros(env.num_envs, dtype=torch.bool, device=expert.device)

    sequence_start_rows = table.clip_start_rows
    sequence_indices = torch.arange(clip_count, dtype=torch.int64, device=expert.device).repeat(policy_count)
    reset_result = reset_tracking_sequences(env, command, sequence_start_rows, sequence_indices)
    if (
        not isinstance(reset_result, tuple)
        or len(reset_result) != 2
        or not isinstance(reset_result[0], TensorDictBase)
        or not isinstance(reset_result[1], Mapping)
    ):
        raise TypeError("Packed SMPL tracking requires the RSL VecEnv reset TensorDict/info pair.")
    observations = reset_result[0]
    for step in range(max_frames):
        context_indices = clip_offsets + torch.minimum(
            metric_lengths, metric_lengths.new_full(metric_lengths.shape, step + 1)
        )
        torch.index_select(contexts, 1, context_indices, out=context_values)
        packed_observations = observations.view(policy_count, clip_count)
        packed_action = model.action_deterministic(packed_observations, context_values)
        observations, _reward, dones, extras = env.step(packed_action.view(env.num_envs, action.action_dim))
        if not isinstance(observations, TensorDictBase):
            raise TypeError("Packed SMPL tracking requires TensorDict observations from the RSL VecEnv wrapper.")
        if not isinstance(dones, torch.Tensor) or dones.shape != (env.num_envs,) or dones.device != expert.device:
            raise ValueError("Packed SMPL done flags must contain one row per environment on the expert device.")
        if not isinstance(extras, Mapping):
            raise TypeError("Packed SMPL tracking extras must be a mapping.")
        done = dones != 0
        policy = observations.get("policy")
        if policy is None or policy.shape != (env.num_envs, 358):
            raise ValueError("Packed SMPL tracking requires one 358-wide live observation per environment.")
        reached.copy_(smpl_humenv_tracking_pose(policy))
        required_final = done & (step < metric_lengths).repeat(policy_count)
        final_observations = extras.get("final_obs")
        if final_observations is None:
            torch._assert_async(
                torch.all(~required_final), "Packed SMPL reached rows require exact final observations."
            )
        else:
            if not isinstance(final_observations, TensorDictBase):
                if not isinstance(final_observations, Mapping) or any(
                    not isinstance(value, torch.Tensor) for value in final_observations.values()
                ):
                    raise TypeError("Packed SMPL final observations must be a TensorDict or tensor mapping.")
                final_observations = TensorDict(dict(final_observations), batch_size=[env.num_envs])
            final_valid = extras.get("final_obs_valid")
            if (
                not isinstance(final_valid, torch.Tensor)
                or final_valid.shape != (env.num_envs,)
                or final_valid.dtype is not torch.bool
            ):
                raise ValueError("Packed SMPL final observations require one validity flag per environment.")
            torch._assert_async(
                torch.all(~required_final | final_valid),
                "A consumed packed SMPL row has no valid final observation.",
            )
            final = final_observations.get("policy")
            if final is None or final.shape != (env.num_envs, 358):
                raise ValueError("Packed SMPL final observations must preserve the 358-wide policy route.")
            torch.where(required_final.unsqueeze(-1), smpl_humenv_tracking_pose(final), reached, out=reached)
        observed[:, :, step].copy_(reached.view(policy_count, clip_count, 214))
        premature = (step < metric_lengths - 1).repeat(policy_count)
        unexpected_done.logical_or_(done & premature)

    torch._assert_async(
        torch.all(~unexpected_done),
        "A packed tracking rollout reset before its expert reference horizon ended.",
    )
    packed_lengths = metric_length_values * policy_count
    emd = torch.empty(policy_count * clip_count, dtype=torch.float64, device=expert.device)
    workspace = UniformAssignmentWorkspace(lengths=packed_lengths, device=expert.device, feature_width=214)
    workspace.compute(
        observed.view(policy_count * clip_count, max_frames, 214),
        target.view(policy_count * clip_count, max_frames, 214),
        emd,
    )

    if expert.device.type == "cuda":
        torch.cuda.synchronize(expert.device)
    duration_seconds = time.perf_counter() - started
    emd_by_policy = emd.view(policy_count, clip_count)
    coverage = torch.ones_like(metric_frame_counts, dtype=torch.float64)
    return tuple(
        ForwardBackwardTrackingEvaluation(
            sequence_ids=clip_ids,
            metric_values={"emd": emd_by_policy[policy_index]},
            source_frame_counts=metric_frame_counts,
            evaluated_frame_counts=evaluated_frame_counts,
            coverage_fraction=coverage,
            duration_seconds=duration_seconds,
        )
        for policy_index in range(policy_count)
    )


def expert_frame_tensordict_packed(
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


def tracking_context_table_packed(
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
        observations = expert_frame_tensordict_packed(model, expert.frames[start:end], policy_count)
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
