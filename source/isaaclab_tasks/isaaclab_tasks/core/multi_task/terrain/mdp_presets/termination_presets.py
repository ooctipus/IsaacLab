# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from ...mdp.terminations import BaseTerminationsCfg, joint_reaction_overload
from .. import mdp
from .robots.robot_presets import NonFootContactBodyNamesCfg


@configclass
class PositionTerminationsCfg(BaseTerminationsCfg):
    """Locomotion termination cfg.

    Inherits ``time_out`` from
    :class:`~isaaclab_tasks.core.multi_task.mdp.terminations.BaseTerminationsCfg`,
    disables the inherited ``abnormal`` alias, and adds:

    - ``abnormal_robot`` — legacy joint-velocity-limit watchdog. The inherited
      ``abnormal`` alias is disabled to keep logs and term names aligned with
      the verified working position runs.
    - ``drop`` — legacy absolute root-height guard at ``minimum_height=-20``.
    - ``base_contact`` — fires on impact contact (force above 3× bodyweight)
      against the configured non-foot body set.
    - ``joint_reaction`` — legacy bodyweight-scaled joint reaction overload
      guard via the ``joint_wrench`` sensor.
    - ``success`` — episode-success termination from the goal-tracking command.
    """

    abnormal = None
    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)

    drop = DoneTerm(func=base_mdp.root_height_below_minimum, params={"minimum_height": -20.0})

    base_contact = DoneTerm(
        func=mdp.illegal_contact_ratio,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=NonFootContactBodyNamesCfg()),  # type: ignore[arg-type]
            "threshold_ratio": 3.0,
        },
    )

    joint_reaction = DoneTerm(
        func=joint_reaction_overload,
        params={
            "sensor_cfg": SceneEntityCfg("joint_wrench"),
            "force_ratio": 6.0,
        },
    )

    success = DoneTerm(func=mdp.success_terminate)


@configclass
class TerminationsCfg(PresetCfg):
    position = PositionTerminationsCfg()
    default = position
