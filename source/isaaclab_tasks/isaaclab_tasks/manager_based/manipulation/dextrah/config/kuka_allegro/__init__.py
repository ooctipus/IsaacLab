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

# Dexsuite Reorient Environments
# State Observation
gym.register(
    id="Dexsuite-Kuka-Allegro-Reorient-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroReorientEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)

gym.register(
    id="Dexsuite-Kuka-Allegro-Reorient-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroReorientEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)

gym.register(
    id="Dexsuite-Kuka-Allegro-Shelves-Reorient-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroShelvesReorientEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)

gym.register(
    id="Dexsuite-Kuka-Allegro-Shelves-Reorient-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroShelvesReorientEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)

gym.register(
    id="Dexsuite-Kuka-Allegro-Shelves-Place-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroShelvesPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)

gym.register(
    id="Dexsuite-Kuka-Allegro-Shelves-Place-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroShelvesPlaceEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)

# Camera Observation
gym.register(
    id="Dexsuite-Kuka-Allegro-Reorient-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_camera_env_cfg:DexsuiteKukaAllegroReorientCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DexsuiteKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)


gym.register(
    id="Dexsuite-Kuka-Allegro-Reorient-Camera-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_camera_env_cfg:DexsuiteKukaAllegroReorientCameraEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DexsuiteKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)

# Dexsuite Lift Environments
# State Observation
gym.register(
    id="Dexsuite-Kuka-Allegro-Lift-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)

gym.register(
    id="Dexsuite-Kuka-Allegro-Lift-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)


# Camera Observation
gym.register(
    id="Dexsuite-Kuka-Allegro-Lift-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_camera_env_cfg:DexsuiteKukaAllegroLiftCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DexsuiteKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)



gym.register(
    id="Dexsuite-Kuka-Allegro-Lift-Camera-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_camera_env_cfg:DexsuiteKukaAllegroLiftCameraEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pbt_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:DexsuiteKukaAllegroPPORunnerCameraCfg",
        "configurable_entry_point": f"{__name__}.configurables:EnvConfigurables",
    },
)