# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test-only canonical views for synthetic motion tables."""

from __future__ import annotations

import torch

from isaaclab_tasks.core.multi_task.mdp.commands.state_command import (
    ResetStateBank,
    ResetStateLayout,
    TaskTableKinematicView,
    TaskTableQuality,
    TaskTableSequenceIndex,
    TaskTableView,
)
from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex, MotionFrames
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import MotionTaskTable

_EXACT_FAMILY_IDENTITY = "3b73dbb8162089bc92699e793d1ff759786b833aa956ca0d7272a7c2842668ea"


def motion_task_table(
    clip_index: MotionClipIndex,
    frames: MotionFrames,
    joint_names: tuple[str, ...],
    *args,
    **kwargs,
) -> MotionTaskTable:
    """Construct a synthetic table with one exact canonical robot view."""
    frames.validate_values()
    device = frames.device
    root_pose = torch.cat((frames.field("root_position"), frames.field("root_rotation")), dim=-1).unsqueeze(1)
    root_velocity = torch.cat(
        (frames.field("root_linear_velocity"), frames.field("root_angular_velocity")), dim=-1
    ).unsqueeze(1)
    layout = ResetStateLayout(
        names=("robot",),
        kinds=("articulation",),
        joint_names=(joint_names,),
        joint_offsets=(0, len(joint_names)),
    )
    bank = ResetStateBank(
        layout,
        root_pose,
        root_velocity,
        frames.field("joint_position"),
        frames.field("joint_velocity"),
    )
    kinematics = TaskTableKinematicView(
        model_builder_state=object(),
        joint_q_default=torch.zeros(7 + len(joint_names), device=device),
        root_entity_names=("robot",),
        root_state_indices=torch.tensor((0,), dtype=torch.int64, device=device),
        root_q_indices=torch.arange(7, dtype=torch.int64, device=device).reshape(1, 7),
        joint_coordinate_names=tuple(("robot", name) for name in joint_names),
        joint_state_indices=torch.arange(len(joint_names), dtype=torch.int64, device=device),
        joint_q_indices=torch.arange(7, 7 + len(joint_names), dtype=torch.int64, device=device),
    )
    view = TaskTableView(
        sequences=TaskTableSequenceIndex(
            offsets=torch.tensor(clip_index.offsets, dtype=torch.int64, device=device),
            frame_dt=torch.tensor(
                [1.0 / clip.source_fps for clip in clip_index.clips], dtype=torch.float32, device=device
            ),
        ),
        state_bank=bank,
        kinematic_view=kinematics,
        quality=TaskTableQuality(
            names=("base_priority",),
            values=torch.ones(len(clip_index.clips), 1, dtype=torch.float32, device=device),
            scope="sequence",
        ),
    )
    return MotionTaskTable(
        clip_index,
        frames,
        joint_names,
        *args,
        family_name=kwargs.pop("family_name", "exact_coordinates"),
        family_identity_sha256=kwargs.pop("family_identity_sha256", _EXACT_FAMILY_IDENTITY),
        view=view,
        **kwargs,
    )
