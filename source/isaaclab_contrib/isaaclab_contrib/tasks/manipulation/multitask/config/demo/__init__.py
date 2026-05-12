
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="Isaac-Registry-Multi-Robot-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_registry_multi_robot_reach_env_cfg:RegistryMultiRobotReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:RegistryMultiRobotReachPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Registry-Multi-Robot-Reach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_registry_multi_robot_reach_env_cfg:RegistryMultiRobotReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:RegistryMultiRobotReachPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Registry-Multi-Robot-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_registry_multi_robot_lift_env_cfg:RegistryMultiRobotLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:RegistryMultiRobotLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Registry-Multi-Robot-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_registry_multi_robot_lift_env_cfg:RegistryMultiRobotLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:RegistryMultiRobotLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Registry-Franka-Multi-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_registry_franka_multi_task_env_cfg:RegistryFrankaMultiTaskEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:RegistryFrankaMultiTaskPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Registry-Franka-Multi-Task-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_registry_franka_multi_task_env_cfg:RegistryFrankaMultiTaskEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:RegistryFrankaMultiTaskPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Registry-Multi-Robot-Multi-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_registry_multi_robot_multi_task_env_cfg:RegistryMultiRobotMultiTaskEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:RegistryMultiRobotMultiTaskPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Registry-Multi-Robot-Multi-Task-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_registry_multi_robot_multi_task_env_cfg:RegistryMultiRobotMultiTaskEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:RegistryMultiRobotMultiTaskPPORunnerCfg",
    },
    disable_env_checker=True,
)
