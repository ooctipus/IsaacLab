# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def skip_reward_term(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term: str,
    curriculum_term: str = "terrain_levels",
):
    term_cfg = env.reward_manager.get_term_cfg(reward_term)
    if term_cfg.weight == 0.0:
        return
    success_rate = env.curriculum_manager.get_term(curriculum_term).success_rates.mean()
    if (success_rate > 0.2 and env.common_step_counter > 100) or env.common_step_counter > 20000:
        # Set weight to zero so manager skips computing it
        term_cfg.weight = 0.0
        # Additionally, replace the callable with a zero-return stub
        if hasattr(term_cfg.func, "reset"):
            # keep simple lambda style, but make signatures flexible to avoid TypeErrors
            term_cfg.func.reset = lambda *args, **kwargs: None
            term_cfg.func.__call__ = lambda *args, **kwargs: torch.zeros(env.num_envs, device=env.device)
        else:
            term_cfg.func = lambda env, **kwargs: torch.zeros(env.num_envs, device=env.device)


def stricten_success_term(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term: str,
    curriculum_term: str = "terrain_levels",
):
    term_cfg = env.termination_manager.get_term_cfg(term)
    success_rate = env.curriculum_manager.get_term(curriculum_term).success_rates.mean()
    if success_rate > 0.1 and env.common_step_counter > 100:
        term_cfg.params["thresh"][2] = 0.5
        term_cfg.params["thresh"][3] = 0.5


def activate_reward_term(env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term: str):
    term_cfg = env.reward_manager.get_term_cfg(reward_term)
    if env.common_step_counter < 5000 and term_cfg.weight != 0.0:
        term_cfg.weight = 0.0
    if env.common_step_counter > 5000 and term_cfg.weight == 0.0:
        term_cfg.weight = 250.0
