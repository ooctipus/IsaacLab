# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Full-board Factory registration."""

import gymnasium as gym

from isaaclab_tasks.contrib.nist.config import franka_factory_cfg as _franka_factory_cfg  # noqa: F401

from . import agents

gym.register(
    id="IsaacContrib-Factory-Board-Reset-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.nistv2.factory_env_cfg:FactoryBoardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FactoryBoardPPORunnerCfg",
        "rsl_rl_mlp_cfg_entry_point": (
            "isaaclab_tasks.contrib.nist.config.agents.rsl_rl_ppo_cfg:FactoryPPORunnerCfg"
        ),
    },
)
