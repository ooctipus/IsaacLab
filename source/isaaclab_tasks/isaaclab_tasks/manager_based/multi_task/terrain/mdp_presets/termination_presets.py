# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from ...mdp.terminations import BaseTerminationsCfg
from .. import mdp
from .robots.robot_presets import NonFootContactBodyNamesCfg


@configclass
class PositionTerminationsCfg(BaseTerminationsCfg):
    """Locomotion termination cfg.

    Inherits ``time_out`` and ``abnormal`` from
    :class:`~isaaclab_tasks.manager_based.multi_task.mdp.terminations.BaseTerminationsCfg`.
    Adds:

    - ``oob`` — fires when the robot's env-relative root drops below 20 m
      (replaces the older ``mdp.root_height_below_minimum``-based ``drop``,
      which compared against absolute world z and broke for terrains with
      non-zero spawn heights).
    - ``base_contact`` — fires on impact contact (force above 3× bodyweight)
      against any non-foot body. Allows routine soft contact (kneeling,
      climbing, leaning) while terminating on shock impacts.
    - ``success`` — episode-success termination from the goal-tracking command.
    """

    oob = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "in_bound_range": {"x": (-1e6, 1e6), "y": (-1e6, 1e6), "z": (-20.0, 1e6)},
        },
    )

    base_contact = DoneTerm(
        func=mdp.illegal_contact_ratio,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=NonFootContactBodyNamesCfg()),  # type: ignore
            "threshold_ratio": 3.0,
        },
    )

    success = DoneTerm(func=mdp.success_terminate, time_out=True)


@configclass
class AdvancedSkillsTerminationsCfg:
    pass
    # TODO(Mateo)


@configclass
class TerminationsCfg(PresetCfg):
    position = PositionTerminationsCfg()
    advanced_skills = AdvancedSkillsTerminationsCfg()
    default = position
