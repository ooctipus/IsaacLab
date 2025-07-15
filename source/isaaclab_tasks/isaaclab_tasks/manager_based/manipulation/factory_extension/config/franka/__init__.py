# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Factory-Franka-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaFactoryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FactoryPPORunnerCfg",
        "state_machine_entry_point": f"{__name__}.state_machines.factory_franka_state_machine:FactoryFrankaStateMachineRelJointPositionAdapterCfg",
    },
)

gym.register(
    id="Factory-Franka-Ik-Del-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_del_env_cfg:FrankaFactoryIkDelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FactoryPPORunnerCfg",
        "state_machine_entry_point": f"{__name__}.state_machines.factory_franka_state_machine:FactoryFrankaStateMachineCfg"
    },
)

gym.register(
    id="Factory-Franka-Ik-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_del_env_cfg:FrankaFactoryIkAbsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FactoryPPORunnerCfg",
        "state_machine_entry_point": f"{__name__}.state_machines.factory_franka_state_machine:FactoryFrankaStateMachineCfg"
    },
)
