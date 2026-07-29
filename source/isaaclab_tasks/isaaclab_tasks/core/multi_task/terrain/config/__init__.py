# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Position-locomotion task registration.

A single gym environment is registered.  The robot (and every robot-specific
override) is chosen declaratively through the preset system -- e.g.::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Position-v0 presets=anymal_c

See :mod:`.mdp_presets.robot_presets` for the robot registry and how to add
new robots.
"""

import gymnasium as gym

gym.register(
    id="Isaac-Position-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.core.multi_task.position_env_cfg:LocomotionPositionCommandEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.core.multi_task.terrain.config.rsl_rl_cfg:PositionRunnerCfg",
    },
)

gym.register(
    id="Isaac-Position-MultiTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.core.multi_task.multi_task_env_cfg:MultiTaskEnvCfg",
        # Separate runner preset: the multi-task env publishes ``policy`` +
        # ``task`` obs groups (no height scanner), so the default PPO runner
        # needs a matching ``obs_groups`` key. See
        # :class:`MultiTaskLocomotionPPORunnerCfg` for the override.
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.core.multi_task.terrain.config.rsl_rl_cfg:MultiTaskLocomotionPPORunnerCfg"
        ),
    },
)
