# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.multi_task.curriculum import (
    BetaSignalCfg,
    CurriculumCfg,
    FrontierSignalCfg,
    SuccessMonitorCfg,
    UniformSignalCfg,
)
from isaaclab_tasks.utils import PresetCfg, preset

from .. import mdp


@configclass
class PositionCurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.terrain_spawn_goal_pair_success_rate_levels,
        params={
            "sampling": preset(
                default=CurriculumCfg(
                    signals=[(BetaSignalCfg(target=0.66, kappa=1.0, eps=1e-8), 1.0)],
                    eps=1e-8,
                ),
                uniform=CurriculumCfg(signals=[(UniformSignalCfg(), 1.0)], eps=0.0),
                beta66=CurriculumCfg(
                    signals=[(BetaSignalCfg(target=0.66, kappa=1.0, eps=1e-8), 1.0)],
                    eps=1e-8,
                ),
                beta50=CurriculumCfg(
                    signals=[(BetaSignalCfg(target=0.50, kappa=1.0, eps=1e-8), 1.0)],
                    eps=1e-8,
                ),
                frontier=CurriculumCfg(
                    signals=[
                        (BetaSignalCfg(target=0.66, kappa=1.0, eps=1e-3), 1.0),
                        (FrontierSignalCfg(k=8, dilation_steps=1), 0.5),
                    ],
                    eps=1e-3,
                ),
                frontier_l1=CurriculumCfg(
                    signals=[
                        (BetaSignalCfg(target=0.66, kappa=1.0, eps=1e-3), 1.0),
                        (FrontierSignalCfg(k=8, dilation_steps=1), 1.0),
                    ],
                    eps=1e-3,
                ),
                frontier_l2=CurriculumCfg(
                    signals=[
                        (BetaSignalCfg(target=0.66, kappa=1.0, eps=1e-3), 1.0),
                        (FrontierSignalCfg(k=8, dilation_steps=1), 2.0),
                    ],
                    eps=1e-3,
                ),
                frontier_l5=CurriculumCfg(
                    signals=[
                        (BetaSignalCfg(target=0.66, kappa=1.0, eps=1e-3), 1.0),
                        (FrontierSignalCfg(k=8, dilation_steps=1), 5.0),
                    ],
                    eps=1e-3,
                ),
                frontier_uniform=CurriculumCfg(
                    signals=[
                        (UniformSignalCfg(), 1.0),
                        (FrontierSignalCfg(k=8, dilation_steps=1), 2.0),
                    ],
                    eps=1e-3,
                ),
            ),
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=20),
            "success_term": "success",
        },
    )


@configclass
class CRLCurriculumCfg:
    """Terrain curriculum without reward-dependent terms."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_spawn_goal_pair_success_rate_levels,
        params={
            "sampling": CurriculumCfg(
                signals=[(BetaSignalCfg(target=0.66, kappa=1.0, eps=1e-8), 1.0)],
                eps=1e-8,
            ),
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=100),
            "success_term": "success",
        },
    )


@configclass
class AdvancedSkillsCurriculumCfg:
    pass
    # TODO(Mateo)


@configclass
class CurriculumPresetCfg(PresetCfg):
    position = PositionCurriculumCfg()
    crl = CRLCurriculumCfg()
    advanced_skills = AdvancedSkillsCurriculumCfg()
    default = position
