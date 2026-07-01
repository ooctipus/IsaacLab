# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared motion-imitation task registration."""

import gymnasium as gym

from isaaclab.utils.module import lazy_export

from . import agents

gym.register(
    id="Isaac-Motion-Imitation-v0",
    entry_point="isaaclab_tasks.core.multi_task.motion_env:MotionImitationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.core.multi_task.motion_env_cfg:MotionImitationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_fb_cfg:MotionForwardBackwardRunnerPresetsCfg",
    },
)

lazy_export()
