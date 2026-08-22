# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static and reset-selectable Factory task registrations.

The task ships one robot, so the robot-specific fields bind the ``default`` of
each robot preset and need no selector. The static task selects one assembly
through the preset system -- e.g.::

    uv run isaaclab train --task IsaacContrib-Factory-Franka \
        presets=peg_insert_4mm physics=newton_mjwarp

The variant task samples the assembly pair at reset. See
:mod:`..factory_presets` for the robot registry and how to add new robots.
"""

import gymnasium as gym

from . import agents

# Imported for its side effect: the module sets each robot-specific field on the
# ``..factory_presets`` classes at import time, and the robot cannot resolve until it has
# run. Keeping it here rather than in ``factory_presets`` leaves that module free of any
# reference to a particular robot.
from . import franka_factory_cfg as _franka_factory_cfg  # noqa: F401

gym.register(
    id="IsaacContrib-Factory-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.nist.factory_env_cfg:FactoryBaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FactoryPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Factory-Video-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.nist.factory_video_env_cfg:FactoryVideoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FactoryPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Factory-Variant-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.nist.factory_variant_env_cfg:FactoryVariantEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FactoryVariantPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Factory-Variant-Video-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.nist.factory_video_env_cfg:FactoryVariantVideoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FactoryVariantPPORunnerCfg",
    },
)
