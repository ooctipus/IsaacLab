# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
)

from isaaclab_tasks.utils import PresetCfg

from ...kuka_allegro.agents.rsl_rl_ppo_cfg import ALGO_CFG, CNN_POLICY_CFG, STATE_POLICY_CFG


@configclass
class DexsuiteKukaSharpaPPOBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 250
    experiment_name = (MISSING,)  # type: ignore
    obs_groups = (MISSING,)  # type: ignore
    actor = (MISSING,)  # type: ignore
    critic = (MISSING,)  # type: ignore
    algorithm = MISSING  # type: ignore


@configclass
class DexsuiteKukaSharpaPPORunnerCfg(PresetCfg):
    default = DexsuiteKukaSharpaPPOBaseRunnerCfg().replace(
        experiment_name="dexsuite_kuka_sharpa",
        obs_groups={"actor": ["policy", "proprio", "perception"], "critic": ["policy", "proprio", "perception"]},
        actor=STATE_POLICY_CFG,
        critic=STATE_POLICY_CFG,
        algorithm=ALGO_CFG,
    )

    single_camera = DexsuiteKukaSharpaPPOBaseRunnerCfg().replace(
        experiment_name="dexsuite_kuka_sharpa_single_camera",
        obs_groups={"actor": ["policy", "proprio", "base_image"], "critic": ["policy", "proprio", "perception"]},
        actor=CNN_POLICY_CFG,
        critic=STATE_POLICY_CFG,
        algorithm=ALGO_CFG.replace(num_mini_batches=16),
    )

    duo_camera = DexsuiteKukaSharpaPPOBaseRunnerCfg().replace(
        experiment_name="dexsuite_kuka_sharpa_duo_camera",
        obs_groups={
            "actor": ["policy", "proprio", "base_image", "wrist_image"],
            "critic": ["policy", "proprio", "perception"],
        },
        actor=CNN_POLICY_CFG,
        critic=STATE_POLICY_CFG,
        algorithm=ALGO_CFG.replace(num_mini_batches=16),
    )
