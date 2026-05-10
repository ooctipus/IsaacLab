# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward composition for primitive graph local outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...kernels_wp import compose_reward

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp
    from .bindings import PrimitiveGraphLocalPlan


def compose_primitive_graph_local_warp(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan) -> None:
    """Advance composer state using dense slot activations.

    The graph dispatch still writes ``outputs.buf_activation``. Reading that
    dense env-slot layout is faster than indirect local-row reads until the
    public output contract no longer requires dense debug slot tensors.
    """
    wp.launch(
        compose_reward,
        dim=(command.num_envs,),
        inputs=[
            plan.env_slots,
            plan.spec,
            plan.composer_state,
            plan.outputs,
            plan.episode_length_buf_wp,
            plan.effective_max_episode_length_wp,
            0.5,
            float(command.cfg.quality_easing),
        ],
        device=str(command.device),
    )
