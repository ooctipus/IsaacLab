# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import RelativeStateCommand


def command_success(env: ManagerBasedRLEnv):
    command_term: RelativeStateCommand = env.command_manager.get_term("goal_point")
    return command_term.get_task_reward()
