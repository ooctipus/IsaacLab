# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


@configclass
class SuccessEstimatorPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    success_estimator_learning_rate: float = 1e-4
    """Learning rate for the success estimator optimizer. Only used with :class:`SuccessEstimatorPPO`."""

    success_loss_coef: float = 1.0
    """Loss coefficient for the success estimator. Only used with :class:`SuccessEstimatorPPO`."""

    success_returns_method: Literal["bootstrap", "hindsight_mc"] = "hindsight_mc"
    """Method for computing success estimator return targets.

    - ``"bootstrap"``: TD(0)-style single-step bootstrap at every step using the
      network's own P(s_{t+1}). Original method — targets depend on the network's
      current accuracy, leading to slow early convergence.
    - ``"hindsight_mc"``: Monte Carlo hindsight relabeling. For completed episodes
      the true binary outcome is propagated to all states in the episode. Only
      episodes still in-progress at the rollout boundary are bootstrapped.
    """
