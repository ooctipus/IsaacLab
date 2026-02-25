
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Multi-task rollout environment
##
gym.register(
    id="Isaac-Franka-Multi-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_multitask_env_cfg:FrankaMultiTaskEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-MultiRobot-Multi-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_multitask_env_cfg:MultiRobotMultiTaskManipulationEnvCfg",
    },
    disable_env_checker=True,
)
