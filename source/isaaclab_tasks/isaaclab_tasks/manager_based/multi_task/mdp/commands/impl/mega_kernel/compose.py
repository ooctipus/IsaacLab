# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward composition phase for the current mega-kernel backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...kernels_wp import compose_reward

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp
    from .bindings import MegaKernelPlan


def compose_warp(command: MultiTaskCommandWarp, plan: MegaKernelPlan) -> None:
    """Advance composer state and write reward/success/progress outputs."""
    wp.launch(
        compose_reward,
        dim=(command.num_envs,),
        inputs=[
            plan.env_slots,
            plan.spec,
            plan.composer_state,
            plan.outputs,
            wp.from_torch(command._env.episode_length_buf),
            wp.from_torch(command._effective_max_episode_length),
            0.5,
            float(command.cfg.quality_easing),
        ],
        device=str(command.device),
    )
