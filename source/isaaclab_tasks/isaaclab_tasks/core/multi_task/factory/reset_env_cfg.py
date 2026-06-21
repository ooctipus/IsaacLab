# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Factory reset-state sampler presets.

The sim-in-the-loop reset strategies that used to live here (``SCENE_RESET`` and
its three tagged branches) were replaced by the offline Newton-IK pipeline
(:mod:`..factory.retarget`), wired through
:attr:`FactoryResetStateTableCfg.pipeline_cfg`.
"""

from isaaclab_tasks.core.multi_task.curriculum import (
    BetaSamplingStrategyCfg,
    FrontierSamplingStrategyCfg,
    SamplerCfg,
    UniformSamplingStrategyCfg,
    ValueShiftSamplingStrategyCfg,
)
from isaaclab_tasks.utils import preset

FACTORY_RESET_SAMPLER_PRESETS = preset(
    default=SamplerCfg(
        strategies=[BetaSamplingStrategyCfg(target=0.66, kappa=2.0, weight=1.0, success_rate_bind="success_rates")],
        eps=1e-3,
    ),
    uniform=SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0),
    monitor=SamplerCfg(
        strategies=[BetaSamplingStrategyCfg(target=0.66, kappa=2.0, weight=1.0, success_rate_bind="success_rates")],
        eps=1e-3,
    ),
    # ``beta`` is a semantic alias of ``monitor``: same Beta rolling-monitor
    # curriculum. Useful as a no-frontier baseline when sweeping ``frontier`` and
    # ``dil*`` so run names read "what's the curriculum?" rather than "what's the
    # rate source?".
    beta=SamplerCfg(
        strategies=[BetaSamplingStrategyCfg(target=0.66, kappa=2.0, weight=1.0, success_rate_bind="success_rates")],
        eps=1e-3,
    ),
    frontier=SamplerCfg(
        strategies=[
            BetaSamplingStrategyCfg(target=0.66, kappa=2.0, weight=1.0, success_rate_bind="success_rates"),
            FrontierSamplingStrategyCfg(
                k=8,
                dilation_steps=preset(default=2, dil1=1, dil2=2, dil3=3, dil4=4, dil5=5),  # type: ignore
                weight=0.5,
                success_rate_bind="success_rates",
            ),
        ],
        eps=1e-3,
    ),
    # Value-shift prioritizes table slots whose critic value moved most between
    # updates. The reset-state command owns stored states and slot application;
    # the strategy itself lives on the curriculum sampler.
    value_shift=SamplerCfg(
        strategies=[
            ValueShiftSamplingStrategyCfg(
                weight=1.0,
                obs_cache_bind="env.command_manager.get_term('reset_state').get_spawn_obs_cache()",
            )
        ],
        eps=1e-3,
    ),
    beta_value_shift=SamplerCfg(
        strategies=[
            BetaSamplingStrategyCfg(target=0.5, kappa=1.0, weight=1.0, success_rate_bind="success_rates"),
            ValueShiftSamplingStrategyCfg(
                weight=0.05,
                obs_cache_bind="env.command_manager.get_term('reset_state').get_spawn_obs_cache()",
            ),
        ],
        eps=1e-3,
    ),
)
