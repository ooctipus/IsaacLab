# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def success_termination(env: ManagerBasedRLEnv, command_name: str = "reset_state") -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    return command_term.get_task_done()
