# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.multi_task.curriculum import (
    BetaSamplingStrategyCfg,
    FrontierSamplingStrategyCfg,
    SamplerCfg,
    SuccessMonitorCfg,
    UniformSamplingStrategyCfg,
)
from isaaclab_tasks.utils import PresetCfg, preset

from .. import mdp


@configclass
class PositionSamplerCfg:
    terrain_levels = CurrTerm(
        func=mdp.terrain_spawn_goal_pair_success_rate_levels,
        params={
            "sampling": preset(
                default=SamplerCfg(
                    strategies=[BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0)],
                    eps=1e-8,
                ),
                uniform=SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0),
                beta66=SamplerCfg(
                    strategies=[BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0)],
                    eps=1e-8,
                ),
                beta50=SamplerCfg(
                    strategies=[BetaSamplingStrategyCfg(target=0.50, kappa=1.0, weight=1.0)],
                    eps=1e-8,
                ),
                frontier=SamplerCfg(
                    strategies=[
                        BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0),
                        FrontierSamplingStrategyCfg(k=8, dilation_steps=2, weight=0.5),
                    ],
                    eps=1e-8,
                ),
                frontier_l1=SamplerCfg(
                    strategies=[
                        BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0),
                        FrontierSamplingStrategyCfg(k=8, dilation_steps=2, weight=1.0),
                    ],
                    eps=1e-8,
                ),
                frontier_l2=SamplerCfg(
                    strategies=[
                        BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0),
                        FrontierSamplingStrategyCfg(k=8, dilation_steps=2, weight=2.0),
                    ],
                    eps=1e-8,
                ),
                frontier_l5=SamplerCfg(
                    strategies=[
                        BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0),
                        FrontierSamplingStrategyCfg(k=8, dilation_steps=2, weight=5.0),
                    ],
                    eps=1e-8,
                ),
                frontier_uniform=SamplerCfg(
                    strategies=[
                        UniformSamplingStrategyCfg(weight=1.0),
                        FrontierSamplingStrategyCfg(k=8, dilation_steps=2, weight=2.0),
                    ],
                    eps=1e-8,
                ),
            ),
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=20),
            "success_term": "success",
        },
    )


@configclass
class CRLSamplerCfg:
    """Terrain curriculum without reward-dependent terms."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_spawn_goal_pair_success_rate_levels,
        params={
            "sampling": SamplerCfg(
                strategies=[BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0)],
                eps=1e-8,
            ),
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=100),
            "success_term": "success",
        },
    )


@configclass
class AdvancedSkillsSamplerCfg:
    pass
    # TODO(Mateo)


@configclass
class CurriculumPresetCfg(PresetCfg):
    position = PositionSamplerCfg()
    crl = CRLSamplerCfg()
    advanced_skills = AdvancedSkillsSamplerCfg()
    default = position
