# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils.configclass import configclass


@configclass
class UniformSamplingCfg:
    """Sample slots uniformly at random. No success rates needed."""

    pass


@configclass
class BetaSamplingCfg:
    """Sample slots using Beta-weighted probabilities peaked near a target success rate.

    Attributes:
        success_rate_bind: Eval expression resolved against ``self`` (the calling
            ManagerTermBase instance) to obtain the per-slot success rate tensor.
        target: Desired success rate peak in [0, 1].
        kappa: Concentration around target.
        temperature: Softmax temperature controlling sharpness.
    """

    success_rate_bind: str = "self.success_rate"
    target: float = 0.5
    kappa: float = 1.0
    temperature: float = 2.0
