# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO runner cfgs for the Meta-World+ Sawyer envs.

All three MT3 tasks share the same actor/critic architecture and PPO
hyperparameters; only ``experiment_name`` differs.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

_POLICY = RslRlPpoActorCriticCfg(
    # Reduced from 0.3 → 0.1. With 0.3, ``Policy/mean_std`` drifted up to
    # ~0.65 over training, so action noise was huge: even after the agent
    # discovered "close gripper near cube + lift", the next step's noisy
    # gripper action would drop the cube. A deterministic probe
    # (check_pick_lift.py) showed that grasp+lift physics works perfectly
    # with clean actions — the failure is RL commitment, not physics.
    init_noise_std=0.1,
    actor_obs_normalization=True,
    critic_obs_normalization=True,
    actor_hidden_dims=[256, 128, 64],
    critic_hidden_dims=[256, 128, 64],
    activation="elu",
)

_ALGO = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    # Match Meta-World paper's PPO: higher entropy + lower lr keep the policy
    # exploring after it locks onto the caging local minimum, which is what
    # enables push/pick-place to discover the phase-bonus regime
    # ("close to cube AND gripper open"). With entropy_coef=0.001 and
    # lr=1e-3 the agent collapsed onto closed-gripper caging (mean reward
    # ~3.87) and never tried opening the gripper to unlock the +5×in_place
    # bonus.
    entropy_coef=0.01,
    num_learning_epochs=10,
    num_mini_batches=4,
    learning_rate=3.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class MetaworldReachSawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    experiment_name = "metaworld_reach_sawyer"
    run_name = ""
    policy = _POLICY
    algorithm = _ALGO


@configclass
class MetaworldPushSawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    experiment_name = "metaworld_push_sawyer"
    run_name = ""
    policy = _POLICY
    algorithm = _ALGO


_ALGO_PICK_PLACE = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    # Pick-place needs the policy to *commit* to a grasp-and-carry strategy
    # once it discovers it. The shared ``entropy_coef=0.01`` is good for
    # push (where exploration unlocks the contact phase) but actively hurts
    # pick-place: with mean_std drifting up to ~0.65, the policy keeps
    # opening/closing the gripper noisily, dropping the cube. Lower entropy
    # for this task only.
    entropy_coef=0.001,
    num_learning_epochs=10,
    num_mini_batches=4,
    learning_rate=3.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class MetaworldPickPlaceSawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    experiment_name = "metaworld_pick_place_sawyer"
    run_name = ""
    policy = _POLICY
    algorithm = _ALGO_PICK_PLACE


@configclass
class MetaworldDrawerOpenSawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    experiment_name = "metaworld_drawer_open_sawyer"
    run_name = ""
    policy = _POLICY
    algorithm = _ALGO


@configclass
class MetaworldDrawerCloseSawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    experiment_name = "metaworld_drawer_close_sawyer"
    run_name = ""
    policy = _POLICY
    algorithm = _ALGO


@configclass
class MetaworldButtonPressTopdownSawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    experiment_name = "metaworld_button_press_topdown_sawyer"
    run_name = ""
    policy = _POLICY
    algorithm = _ALGO


@configclass
class MetaworldDoorOpenSawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    experiment_name = "metaworld_door_open_sawyer"
    run_name = ""
    policy = _POLICY
    algorithm = _ALGO


@configclass
class MetaworldWindowOpenSawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    experiment_name = "metaworld_window_open_sawyer"
    run_name = ""
    policy = _POLICY
    algorithm = _ALGO


@configclass
class MetaworldWindowCloseSawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    experiment_name = "metaworld_window_close_sawyer"
    run_name = ""
    policy = _POLICY
    algorithm = _ALGO


@configclass
class MetaworldPegInsertSideSawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    experiment_name = "metaworld_peg_insert_side_sawyer"
    run_name = ""
    policy = _POLICY
    algorithm = _ALGO_PICK_PLACE  # peg-insert needs the lower entropy too


# Multi-task algo: between single-task entropy_coef (0.01) and "killed"
# (0.001). v2 of MT3 with entropy_coef=0.001 collapsed action_std to 0.02
# and got 0% success on all three tasks (reach single-task hits ~30% but
# multi-task drops to 0). Push reward 0.9 (effective 2.7/push-env, vs
# single-task 3.9). v1 with 0.01 blew up std to 23.7. Try 0.003 for v3.
_ALGO_MULTITASK = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.003,
    num_learning_epochs=10,
    num_mini_batches=4,
    learning_rate=3.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


# Larger actor/critic network for multi-task. Single-task uses [256,128,64];
# multi-task needs more capacity to specialise per-task within the shared
# weights.
_POLICY_MULTITASK = RslRlPpoActorCriticCfg(
    init_noise_std=0.3,  # higher than single-task 0.1 — wider initial exploration
    actor_obs_normalization=True,
    critic_obs_normalization=True,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
)


@configclass
class MetaworldMT3SawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Heterogeneous MT3 (reach + push + pick-place) shared-policy runner."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "metaworld_mt3_sawyer_multitask"
    run_name = ""
    policy = _POLICY_MULTITASK
    algorithm = _ALGO_MULTITASK


@configclass
class MetaworldMT10SawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Heterogeneous MT10 shared-policy runner."""

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 250
    experiment_name = "metaworld_mt10_sawyer_multitask"
    run_name = ""
    policy = _POLICY_MULTITASK
    algorithm = _ALGO_MULTITASK


@configclass
class MetaworldMT5SawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Heterogeneous MT5 shared-policy runner (5 manipulation tasks)."""

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 250
    experiment_name = "metaworld_mt5_sawyer_multitask"
    run_name = ""
    policy = _POLICY_MULTITASK
    algorithm = _ALGO_MULTITASK


@configclass
class MetaworldMT15SawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Heterogeneous MT15 shared-policy runner (15 articulated tasks)."""

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 250
    experiment_name = "metaworld_mt15_sawyer_multitask"
    run_name = ""
    policy = _POLICY_MULTITASK
    algorithm = _ALGO_MULTITASK


@configclass
class MetaworldMT25SawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Heterogeneous MT25 shared-policy runner (MT15 + 10 articulated tasks)."""

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 250
    experiment_name = "metaworld_mt25_sawyer_multitask"
    run_name = ""
    policy = _POLICY_MULTITASK
    algorithm = _ALGO_MULTITASK


@configclass
class MetaworldMT50SawyerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Heterogeneous MT50 shared-policy runner (MT25 + 25 cube/destination tasks)."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 250
    experiment_name = "metaworld_mt50_sawyer_multitask"
    run_name = ""
    policy = _POLICY_MULTITASK
    algorithm = _ALGO_MULTITASK
