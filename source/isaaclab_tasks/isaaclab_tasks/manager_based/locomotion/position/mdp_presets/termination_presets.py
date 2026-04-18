# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from .. import mdp
from .robots.robot_presets import BaseContactBodyNamesCfg


@configclass
class PositionTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    drop = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -20})

    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BaseContactBodyNamesCfg()),  # type: ignore
            "threshold": 1.0,
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
