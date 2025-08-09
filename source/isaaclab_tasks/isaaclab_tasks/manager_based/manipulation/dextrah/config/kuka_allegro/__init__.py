# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dextra Kuka Allegro environments.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# State Observation
gym.register(
    id="Dextrah-Kuka-Allegro-Reorient-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dextrah_kuka_allegro_env_cfg:DextrahKukaAllegroReorientEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DextrahKukaAllegroPPORunnerCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)

gym.register(
    id="Dextrah-Kuka-Allegro-Reorient-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dextrah_kuka_allegro_env_cfg:DextrahKukaAllegroReorientEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DextrahKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)

# Camera Observation
gym.register(
    id="Dextrah-Kuka-Allegro-Reorient-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dextrah_kuka_allegro_camera_env_cfg:DextrahKukaAllegroReorientCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DextrahKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)


gym.register(
    id="Dextrah-Kuka-Allegro-Reorient-Camera-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dextrah_kuka_allegro_camera_env_cfg:DextrahKukaAllegroReorientCameraEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DextrahKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)

# Dextrah Lift Environments
# State Observation
gym.register(
    id="Dextrah-Kuka-Allegro-Lift-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dextrah_kuka_allegro_env_cfg:DextrahKukaAllegroLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DextrahKukaAllegroPPORunnerCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)

gym.register(
    id="Dextrah-Kuka-Allegro-Lift-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dextrah_kuka_allegro_env_cfg:DextrahKukaAllegroLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DextrahKukaAllegroPPORunnerCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)


# Camera Observation
gym.register(
    id="Dextrah-Kuka-Allegro-Lift-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dextrah_kuka_allegro_camera_env_cfg:DextrahKukaAllegroLiftCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DextrahKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)



gym.register(
    id="Dextrah-Kuka-Allegro-Lift-Camera-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dextrah_kuka_allegro_camera_env_cfg:DextrahKukaAllegroLiftCameraEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DextrahKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)