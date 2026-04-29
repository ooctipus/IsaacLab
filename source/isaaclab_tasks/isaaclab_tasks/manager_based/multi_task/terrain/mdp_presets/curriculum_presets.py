# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.multi_task.mdp.util import (
    BetaSamplingCfg,
    SuccessMonitorCfg,
    UniformSamplingCfg,
)
from isaaclab_tasks.utils import PresetCfg, preset

from .. import mdp


@configclass
class PositionCurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.terrain_spawn_goal_pair_success_rate_levels,
        params={
            "sampling": preset(
                default=BetaSamplingCfg(target=0.66, kappa=1.0),
                uniform=UniformSamplingCfg(),
                beta66=BetaSamplingCfg(target=0.66, kappa=1.0),
                beta100=BetaSamplingCfg(target=1.0, kappa=1.0),
            ),
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=50),
            "success_term": "success",
        },
    )
    # remove_explore_reward = CurrTerm(func=mdp.skip_reward_term, params={"reward_term": "explore"})


@configclass
class CRLCurriculumCfg:
    """Terrain curriculum without reward-dependent terms."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_spawn_goal_pair_success_rate_levels,
        params={
            "sampling": BetaSamplingCfg(target=0.66, kappa=1.0),
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=100),
            "success_term": "success",
        },
    )


@configclass
class AdvancedSkillsCurriculumCfg:
    pass
    # TODO(Mateo)


@configclass
class CurriculumCfg(PresetCfg):
    position = PositionCurriculumCfg()
    crl = CRLCurriculumCfg()
    advanced_skills = AdvancedSkillsCurriculumCfg()
    default = position
