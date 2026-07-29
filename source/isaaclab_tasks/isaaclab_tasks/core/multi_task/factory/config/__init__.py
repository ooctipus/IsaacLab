# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory task registration.

A single gym environment is registered.  The robot (and every robot-specific
override) is chosen declaratively through the preset system -- e.g.::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Factory-v0 presets=franka,peg_insert_4mm

See :mod:`..mdp_presets.robots` for the robot registry and how to add
new robots.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Factory-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.core.multi_task.factory_env_cfg:FactoryBaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FactoryPPORunnerCfg",
    },
)
