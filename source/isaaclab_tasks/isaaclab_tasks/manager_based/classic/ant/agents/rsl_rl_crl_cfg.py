# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CRL agent configuration for the Ant goal-reaching task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlCrlAlgorithmCfg, RslRlHerCfg, RslRlOnPolicyRunnerCfg


@configclass
class AntCRLRunnerCfg(RslRlOnPolicyRunnerCfg):
    """OnPolicyRunner config that dispatches to CRL via ``algorithm.class_name``."""

    num_steps_per_env = 62
    max_iterations = 5000
    save_interval = 500
    experiment_name = "ant_crl"

    obs_groups = {
        "actor": ["policy", "task"],
        "critic": ["policy", "task"],
    }

    algorithm: RslRlCrlAlgorithmCfg = RslRlCrlAlgorithmCfg(
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        batch_size=256,
        max_replay_size=10000,
        min_replay_size=1000,
        num_sgd_steps=800,
        logsumexp_penalty_coeff=0.1,
        entropy_param=0.5,
        # fixed_alpha=0.1,
        hidden_dim=256,
        depth=32,
        num_layers_per_block=4,
        expand=1,
        activation="swish",
        repr_dim=64,
        sample_window_length=1000,
        her_cfg=RslRlHerCfg(
            gamma=0.99,
            goal_group="task",
            achieved_goal_group="policy",
            achieved_goal_slice_start=0,
            achieved_goal_slice_end=3,
        ),
    )
