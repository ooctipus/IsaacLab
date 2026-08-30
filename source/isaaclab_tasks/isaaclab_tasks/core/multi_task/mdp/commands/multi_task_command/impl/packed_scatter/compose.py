# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward composition phase for the packed_scatter backend.

Uses the same dense-output composer as :mod:`mega_kernel.compose` — packed_scatter
produces the legacy dense ``command_reach``/``command_track`` output layout, so
the composer reads from the agreement-layer plan-struct fields identically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..compose_select import use_parallel_compose
from ..kernels_wp import compose_reward, compose_reward_parallel

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp
    from .bindings import PackedScatterPlan


def compose_warp(command: MultiTaskCommandWarp, plan: PackedScatterPlan) -> None:
    """Advance composer state and write reward/success/progress outputs."""
    inputs = [
        plan.env_slots,
        plan.spec,
        plan.composer_state,
        plan.outputs,
        plan.episode_length_buf_wp,
        plan.effective_max_episode_length_wp,
        0.5,
        float(command.cfg.quality_easing),
    ]
    if use_parallel_compose(command.k_max):
        wp.launch_tiled(
            compose_reward_parallel,
            dim=[command.num_envs],
            inputs=inputs,
            block_dim=max(command.k_max, 32),
            device=str(command.device),
        )
    else:
        wp.launch(
            compose_reward,
            dim=(command.num_envs,),
            inputs=inputs,
            device=str(command.device),
        )
