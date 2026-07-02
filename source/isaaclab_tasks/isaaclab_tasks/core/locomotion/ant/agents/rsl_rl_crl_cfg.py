# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CRL agent configuration for the Ant goal-reaching task."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.multi_task.rl.rsl_rl import (
    RslRlCrlAlgorithmCfg,
    RslRlCrlRunnerCfg,
    RslRlHerCfg,
    RslRlResidualMLPCfg,
)


@configclass
class AntCRLRunnerCfg(RslRlCrlRunnerCfg):
    """CRL replay-runner config for the Ant goal-reaching task."""

    num_steps_per_env = 62
    max_iterations = 5000
    save_interval = 500
    experiment_name = "ant_crl"

    obs_groups = {
        "actor": ["current_state", "policy", "target_state"],
        "critic": ["current_state", "policy", "target_state"],
    }

    actor: RslRlResidualMLPCfg = RslRlResidualMLPCfg(
        hidden_dim=256,
        depth=64,
        num_layers_per_block=4,
        expand=1,
        activation="swish",
    )

    critic: RslRlResidualMLPCfg = RslRlResidualMLPCfg(
        hidden_dim=256,
        depth=64,
        num_layers_per_block=4,
        expand=1,
        activation="swish",
        repr_dim=64,
    )

    algorithm: RslRlCrlAlgorithmCfg = RslRlCrlAlgorithmCfg(
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        max_replay_size=10000,
        min_replay_size=1000,
        replay_ratio=0.08,
        num_sgd_steps=100,
        logsumexp_penalty_coeff=0.1,
        entropy_param=0.5,
        her_cfg=RslRlHerCfg(gamma=0.99),
    )
