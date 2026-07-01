# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL expert-corpus attachment for completed motion task tables."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol

import torch
from rsl_rl.models.forward_backward_model import ForwardBackwardObservationSchema
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer, ForwardBackwardExpertSchema

from .config.robots import G1_BEHAVIOR_BODY_NAMES, G1_BEHAVIOR_JOINT_NAMES
from .data import MotionSampleGrid
from .data._identity import canonical_sha256
from .frames import G1_HEAD_FRAME_NAME
from .mdp.commands.motion_task_table import MotionTaskTable
from .mdp.observations import g1_privileged_observation, g1_released_expert_state

_SMPL_EXPERT_PROJECTION = "native_humenv_observation_float32_v1"
_G1_EXPERT_PROJECTION = "released_behavior_axes_v2"

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class _MotionFieldSource(Protocol):
    """Named robot facts on one already-resolved expert sample grid."""

    def field(self, name: str) -> torch.Tensor:
        """Return every resolved row for one named robot fact."""


class _MotionPayload(Protocol):
    """Environment-owned completed motion data."""

    table: MotionTaskTable


class _IndexedMotionFields:
    """Gather selected source rows while preserving one field-access contract."""

    def __init__(self, table: MotionTaskTable, indices: torch.Tensor) -> None:
        self.table = table
        self.indices = indices

    def field(self, name: str) -> torch.Tensor:
        """Gather one field in the selected row order."""
        return torch.index_select(self.table.field(name), 0, self.indices)


def _motion_payload(
    env: RslRlVecEnvWrapper,
    command_name: str,
    device: str | torch.device,
) -> _MotionPayload:
    """Read the active robot ordering and completed table from one command."""
    command = env.unwrapped.command_manager.get_term(command_name)
    payload = command.payload
    table = payload.table
    if not isinstance(table, MotionTaskTable):
        raise TypeError(f"Command {command_name!r} does not own a MotionTaskTable.")
    if table.device != torch.device(device):
        raise ValueError(f"Motion table is on {table.device}, but the learner is on {torch.device(device)}.")
    return payload


def _resolved_fields(
    table: MotionTaskTable,
    grid: MotionSampleGrid,
) -> tuple[_MotionFieldSource, tuple[int, ...]]:
    """Resolve all valid clips on one declared expert clock without crossing clips."""
    selected = tuple(index for index, clip in enumerate(table.clip_index.clips) if clip.valid)
    if not selected:
        raise ValueError("The motion table contains no valid expert clips.")
    counts = tuple(
        grid.sample_count(
            frame_count=table.clip_index.clips[index].frame_count,
            source_fps=table.clip_index.clips[index].source_fps,
        )
        for index in selected
    )
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    offset_values = tuple(offsets)

    all_source_rows = (
        selected == tuple(range(len(table.clip_index.clips))) and grid.mode is MotionSampleGrid.Mode.SOURCE_ROWS
    )
    if all_source_rows:
        return table, offset_values

    selected_tensor = torch.tensor(selected, dtype=torch.int64, device=table.device)
    counts_tensor = torch.tensor(counts, dtype=torch.int64, device=table.device)
    clip_positions = torch.repeat_interleave(
        torch.arange(len(selected), dtype=torch.int64, device=table.device),
        counts_tensor,
    )
    flat_indices = torch.arange(offset_values[-1], dtype=torch.int64, device=table.device)
    starts = torch.tensor(offset_values[:-1], dtype=torch.int64, device=table.device)
    local_samples = flat_indices - starts[clip_positions]
    clip_indices = selected_tensor[clip_positions]
    if grid.mode is MotionSampleGrid.Mode.SOURCE_ROWS:
        global_indices = table.clip_offsets[clip_indices] + local_samples
        return _IndexedMotionFields(table, global_indices), offset_values

    sample_times = grid.time_seconds(local_samples, table.source_fps[clip_indices])
    return table.reference_view(clip_indices, sample_times), offset_values


def _offsets_hash(offsets: tuple[int, ...]) -> str:
    """Hash clip boundaries without copying a device tensor to the host."""
    digest = hashlib.sha256()
    for value in offsets:
        digest.update(value.to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _expert_buffer(
    table: MotionTaskTable,
    observation_schema: ForwardBackwardObservationSchema,
    grid: MotionSampleGrid,
    frames: torch.Tensor,
    offsets: tuple[int, ...],
    *,
    clip_ids: tuple[str, ...],
    dataset_id: str,
    projection_identity: object,
    window_lengths: tuple[int, ...],
    seed: int,
) -> ForwardBackwardExpertBuffer:
    """Bind projected frames and deterministic corpus identity to RSL-RL storage."""
    width = observation_schema.route_width("backward")
    if frames.ndim != 2 or frames.shape != (offsets[-1], width):
        raise ValueError(f"Expert projection must have shape {(offsets[-1], width)}, got {tuple(frames.shape)}.")
    if frames.device != table.device or frames.dtype is not torch.float32 or frames.requires_grad:
        raise ValueError("Expert projection must be detached float32 on the motion-table device.")
    if len(clip_ids) != len(offsets) - 1:
        raise ValueError("Expert clip ids must align exactly with resolved clip offsets.")
    data_hash = canonical_sha256(
        {
            "table": table.cache_identity,
            "sample_grid": {"mode": grid.mode.value, "step_seconds": grid.step_seconds},
            "projection": projection_identity,
        }
    )
    clip_offsets = torch.tensor(offsets, dtype=torch.int64, device=table.device)
    priorities = torch.ones(len(offsets) - 1, dtype=torch.float32, device=table.device)
    schema = ForwardBackwardExpertSchema(
        dataset_id=dataset_id,
        data_hash=data_hash,
        feature_schema_hash=observation_schema.schema_hash,
        clip_offsets_hash=_offsets_hash(offsets),
        expert_feature_width=width,
        num_frames=frames.shape[0],
        num_clips=len(offsets) - 1,
        window_lengths=window_lengths,
    )
    return ForwardBackwardExpertBuffer(frames, clip_offsets, priorities, schema, seed=seed, clip_ids=clip_ids)


def motion_expert_buffer_smpl_cmu(
    env: RslRlVecEnvWrapper,
    observation_schema: ForwardBackwardObservationSchema,
    device: str,
    *,
    command_name: str,
    window_lengths: tuple[int, ...],
    seed: int = 0,
) -> ForwardBackwardExpertBuffer:
    """Return the exact stored 358-wide HumEnv expert corpus.

    Args:
        env: RSL-RL wrapper around the motion environment.
        observation_schema: Learner observation fields and routes.
        device: Learner tensor device.
        command_name: Name of the state command that owns the motion table.
        window_lengths: Expert edge-window lengths available to the learner.
        seed: Expert sampler seed.

    Returns:
        Immutable clip-safe expert buffer sharing native observation rows.
    """
    payload = _motion_payload(env, command_name, device)
    table = payload.table
    grid = table.expert_sample_grid
    dataset_id = f"{table.clip_index.semantic_level}:{table.frame_builder_version}"
    if table.reference_frame_names:
        raise ValueError("SMPL-CMU expert tables must not contain reference-frame columns.")
    if table.field("observation").shape[1:] != (358,):
        raise ValueError("The motion table does not contain 358-wide SMPL-CMU observations.")
    if observation_schema.route("backward") != ("policy",) or dict(observation_schema.field_widths)["policy"] != 358:
        raise ValueError("SMPL-CMU expert features require the 358-wide ('policy',) backward route.")
    fields, offsets = _resolved_fields(table, grid)
    return _expert_buffer(
        table,
        observation_schema,
        grid,
        fields.field("observation"),
        offsets,
        clip_ids=table.clip_ids,
        dataset_id=dataset_id,
        projection_identity=_SMPL_EXPERT_PROJECTION,
        window_lengths=window_lengths,
        seed=seed,
    )


def motion_expert_buffer_g1(
    env: RslRlVecEnvWrapper,
    observation_schema: ForwardBackwardObservationSchema,
    device: str,
    *,
    command_name: str,
    window_lengths: tuple[int, ...],
    seed: int = 0,
) -> ForwardBackwardExpertBuffer:
    """Return the exact 50 Hz G1 state-plus-privileged expert corpus.

    Args:
        env: RSL-RL wrapper around the motion environment.
        observation_schema: Learner observation fields and routes.
        device: Learner tensor device.
        command_name: Name of the state command that owns the motion table.
        window_lengths: Expert edge-window lengths available to the learner.
        seed: Expert sampler seed.

    Returns:
        Immutable clip-safe expert buffer on the declared 50 Hz sample grid.
    """
    payload = _motion_payload(env, command_name, device)
    table = payload.table
    grid = table.expert_sample_grid
    dataset_id = f"{table.clip_index.semantic_level}:{table.frame_builder_version}"
    robot = env.unwrapped.scene["robot"]
    if table.joint_names != tuple(robot.joint_names):
        raise ValueError("G1 expert trajectory joints differ from the live articulation order.")
    expected_reference_frames = (*robot.body_names, G1_HEAD_FRAME_NAME)
    if table.reference_frame_names != expected_reference_frames:
        raise ValueError("G1 expert reference frames must be the live body order followed by head_link.")

    expected_shapes = {
        "joint_position": (29,),
        "joint_velocity": (29,),
        "body_position": (31, 3),
        "body_rotation": (31, 4),
        "body_linear_velocity": (31, 3),
        "body_angular_velocity": (31, 3),
    }
    actual_shapes = {name: table.field(name).shape[1:] for name in expected_shapes}
    if actual_shapes != expected_shapes:
        raise ValueError(f"G1 expert trajectory shapes differ: expected {expected_shapes}, got {actual_shapes}.")
    widths = dict(observation_schema.field_widths)
    if observation_schema.route("backward") != ("state", "privileged_state") or (
        widths["state"],
        widths["privileged_state"],
    ) != (64, 463):
        raise ValueError("G1 expert features require the 64+463 state/privileged backward route.")

    action = env.unwrapped.action_manager.get_term("joint_position")
    behavior_joint_names = tuple(action.joint_names)
    if behavior_joint_names != G1_BEHAVIOR_JOINT_NAMES:
        raise ValueError("G1 action joints differ from the declared behavior axis.")
    if set(table.joint_names) != set(behavior_joint_names):
        raise ValueError("G1 trajectory and behavior joint names differ.")
    joint_indices = torch.tensor(
        [table.joint_names.index(name) for name in behavior_joint_names],
        dtype=torch.int64,
        device=table.device,
    )

    behavior_frame_names = (*G1_BEHAVIOR_BODY_NAMES, G1_HEAD_FRAME_NAME)
    if set(table.reference_frame_names) != set(behavior_frame_names):
        raise ValueError("G1 trajectory and behavior reference-frame names differ.")
    body_indices = torch.tensor(
        [table.reference_frame_names.index(name) for name in behavior_frame_names],
        dtype=torch.int64,
        device=table.device,
    )
    default_joint_position = action.joint_default_position.to(device=table.device)
    if default_joint_position.shape != (29,) or default_joint_position.device != table.device:
        raise ValueError("G1 expert projection requires 29 behavior-ordered joint defaults on the table device.")
    fields, offsets = _resolved_fields(table, grid)
    frames = torch.empty((offsets[-1], 527), dtype=torch.float32, device=table.device)
    joint_position = fields.field("joint_position").index_select(1, joint_indices)
    joint_velocity = fields.field("joint_velocity").index_select(1, joint_indices)
    body_rotation = fields.field("body_rotation").index_select(1, body_indices)
    body_angular_velocity = fields.field("body_angular_velocity").index_select(1, body_indices)
    state = g1_released_expert_state(
        body_rotation[:, 0],
        body_angular_velocity[:, 0],
        joint_position,
        joint_velocity,
        default_joint_position,
    )
    frames[:, :64].copy_(state)
    del joint_position, joint_velocity, state

    body_position = fields.field("body_position").index_select(1, body_indices)
    body_linear_velocity = fields.field("body_linear_velocity").index_select(1, body_indices)
    privileged_state = g1_privileged_observation(
        body_position,
        body_rotation,
        body_linear_velocity,
        body_angular_velocity,
    )
    frames[:, 64:].copy_(privileged_state)
    del body_angular_velocity, body_linear_velocity, body_position, body_rotation, privileged_state
    return _expert_buffer(
        table,
        observation_schema,
        grid,
        frames,
        offsets,
        clip_ids=table.clip_ids,
        dataset_id=dataset_id,
        window_lengths=window_lengths,
        seed=seed,
        projection_identity={
            "version": _G1_EXPERT_PROJECTION,
            "joint_names": behavior_joint_names,
            "body_names": G1_BEHAVIOR_BODY_NAMES,
            "joint_default_position": default_joint_position.detach().cpu().tolist(),
        },
    )
