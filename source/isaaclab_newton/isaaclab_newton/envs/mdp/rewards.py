# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-specific reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .terminations import solver_reset_required

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def zero_reward_on_solver_reset(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Zero the current reward for worlds whose Newton solver requires reset.

    This term must be configured after all additive reward terms with a
    nonzero weight. It edits the
    reward manager's current total in place so even a non-finite earlier reward
    is replaced with zero.

    Args:
        env: The environment. It must use Newton's MuJoCo Warp solver.

    Returns:
        Zero reward contribution for every environment, shape (num_envs,).
    """
    reset_required = solver_reset_required(env)
    env.reward_manager._reward_buf[reset_required] = 0.0
    return torch.zeros_like(env.reward_manager._reward_buf)
