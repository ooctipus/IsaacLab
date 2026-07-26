# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils import PresetCfg, preset

# Shared PPO hyper-parameters reused by both the plain-PPO and value-shift variants.
_FACTORY_PPO_KWARGS = dict(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=6e-3,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-4,
    schedule="adaptive",
    gamma=0.995,
    lam=0.90,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class ValueShiftAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm cfg + bind expressions for the value-shift curriculum.

    Bind expressions are popped off this cfg by
    :meth:`ValueShiftPPO.construct_algorithm` (so they never reach
    ``PPO.__init__``) and ``eval``-ed against ``{env, alg, setattr}``. They wire
    :class:`ValueShiftPPO`'s three buffers to the matching
    :class:`ValueShiftSamplingStrategy` living on the ``reset_strategies``
    accumulator term. Only meaningful when that accumulator's sampler preset
    includes value-shift scoring (``value_shift`` / ``beta_value_shift``).
    """

    class_name: str = "isaaclab_tasks.core.multi_task.rl.rsl_rl.algorithms:ValueShiftPPO"
    # ``env`` here is the ``RslRlVecEnvWrapper`` from the runner -- managers
    # live on the underlying ``ManagerBasedRLEnv`` accessed via ``.unwrapped``.
    bind_observation_exp: str = (
        "setattr(alg, '_obs_cache',"
        " env.unwrapped.event_manager.get_term_cfg('reset_strategies').func"
        "._sampler._impl.value_shift_strategy.observation_cache)"
    )
    bind_current_value_exp: str = (
        "setattr(alg, '_cur_buf',"
        " env.unwrapped.event_manager.get_term_cfg('reset_strategies').func"
        "._sampler._impl.value_shift_strategy.cur_val)"
    )
    bind_value_diff_exp: str = (
        "setattr(alg, '_diff_buf',"
        " env.unwrapped.event_manager.get_term_cfg('reset_strategies').func"
        "._sampler._impl.value_shift_strategy.diff_val)"
    )


@configclass
class PpoAlgorithmCfg(PresetCfg):
    actor_critic = RslRlPpoAlgorithmCfg(class_name="PPO", **_FACTORY_PPO_KWARGS)
    default = actor_critic
    value_shift = ValueShiftAlgorithmCfg(**_FACTORY_PPO_KWARGS)
    beta_value_shift = value_shift


@configclass
class FactoryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 200
    experiment_name = "factory"
    obs_groups = preset(
        default={"actor": ["policy"], "critic": ["policy"]},
        actor_critic={"actor": ["policy"], "critic": ["policy"]},
    )  # type: ignore
    actor = RslRlMLPModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        obs_normalization=True,
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    critic = RslRlMLPModelCfg(
        obs_normalization=True,
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    algorithm = PpoAlgorithmCfg()  # type: ignore
