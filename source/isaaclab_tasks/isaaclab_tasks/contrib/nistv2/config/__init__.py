# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory v2 task registration."""

import gymnasium as gym

from isaaclab_tasks.contrib.nist.config import franka_factory_cfg as _franka_factory_cfg  # noqa: F401

from . import agents

gym.register(
    id="IsaacContrib-Factory-V2-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.nistv2.factory_env_cfg:FactoryBaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FactoryPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Factory-V2-Video-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.nistv2.factory_video_env_cfg:FactoryVideoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FactoryPPORunnerCfg",
    },
)
