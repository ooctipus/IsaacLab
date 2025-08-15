# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

# Depth Camera Observation
gym.register(
    id="Dextrah-Kuka-Allegro-Reorient-Depth-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dextrah_kuka_allegro_camera_env_cfg:DextrahKukaAllegroReorientDepthCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DextrahKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)


gym.register(
    id="Dextrah-Kuka-Allegro-Reorient-Depth-Camera-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dextrah_kuka_allegro_camera_env_cfg:DextrahKukaAllegroReorientDepthCameraEnvCfg_PLAY"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DextrahKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)


gym.register(
    id="Dextrah-Kuka-Allegro-Reorient-RGB-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dextrah_kuka_allegro_camera_env_cfg:DextrahKukaAllegroReorientRGBCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DextrahKukaAllegroPPORunnermeraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)


gym.register(
    id="Dextrah-Kuka-Allegro-Reorient-RGB-Camera-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dextrah_kuka_allegro_camera_env_cfg:DextrahKukaAllegroReorientRGBCameraEnvCfg_PLAY"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DextrahKukaAllegroPPORunnermeraCfg",
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


# Depth Camera Observation
gym.register(
    id="Dextrah-Kuka-Allegro-Lift-Depth-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dextrah_kuka_allegro_camera_env_cfg:DextrahKukaAllegroLiftDepthCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DextrahKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)


gym.register(
    id="Dextrah-Kuka-Allegro-Lift-Depth-Camera-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dextrah_kuka_allegro_camera_env_cfg:DextrahKukaAllegroLiftDepthCameraEnvCfg_PLAY"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DextrahKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)

gym.register(
    id="Dextrah-Kuka-Allegro-Lift-RGB-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dextrah_kuka_allegro_camera_env_cfg:DextrahKukaAllegroLiftRGBCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DextrahKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)


gym.register(
    id="Dextrah-Kuka-Allegro-Lift-RGB-Camera-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dextrah_kuka_allegro_camera_env_cfg:DextrahKukaAllegroLiftRGBCameraEnvCfg_PLAY"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DextrahKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)

