# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_tasks.core.multi_task.curriculum import get_reset_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_state(env: ManagerBasedRLEnv, reset_assets: list[str]):
    return get_reset_state(env, slice(None), reset_assets, is_relative=True)
