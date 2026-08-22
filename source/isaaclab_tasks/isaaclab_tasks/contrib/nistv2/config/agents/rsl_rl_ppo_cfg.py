# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for complete-board Factory training."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from isaaclab_tasks.contrib.nist.config.agents.models import SimBaModelCfg
from isaaclab_tasks.contrib.nist.config.agents.rsl_rl_ppo_cfg import FactoryPPORunnerCfg


@configclass
class FactoryBoardPPORunnerCfg(FactoryPPORunnerCfg):
    """PPO runner using SimBa directly on the complete-board policy state."""

    actor = SimBaModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        obs_normalization=True,
        hidden_dim=256,
        num_blocks=2,
        expansion_factor=4,
        activation="swish",
    )
    critic = SimBaModelCfg(
        obs_normalization=True,
        hidden_dim=256,
        num_blocks=2,
        expansion_factor=4,
        activation="swish",
    )
