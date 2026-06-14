# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from ...mdp.terminations import BaseTerminationsCfg
from .. import mdp


@configclass
class TimeoutTerminationsCfg(BaseTerminationsCfg):
    """Termination terms for the timeout-terminate formulation."""

    oob = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "in_bound_range": {"x": (-0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)},
        },
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact_ratio,  # type: ignore
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensors", body_names=".*"),
            "threshold_ratio": 2.0,
        },
    )
    joint_reaction = DoneTerm(
        func=mdp.joint_reaction_overload,  # type: ignore
        params={
            "sensor_cfg": SceneEntityCfg("joint_wrench"),
            "force_ratio": 1.0,
            "force_mode": "off_axis",
        },
    )


@configclass
class SuccessTerminationsCfg(BaseTerminationsCfg):
    """Termination terms for the success-terminate formulation."""

    oob = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "in_bound_range": {"x": (-0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)},
        },
    )
    bad_contact = DoneTerm(
        func=mdp.illegal_contact_ratio,  # type: ignore
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensors", body_names=".*"),
            "threshold_ratio": 3.0,
        },
    )
    joint_reaction = DoneTerm(
        func=mdp.joint_reaction_overload,  # type: ignore
        params={
            "sensor_cfg": SceneEntityCfg("joint_wrench"),
            "force_ratio": 5.0,
            "force_mode": "off_axis",
        },
    )
    success = DoneTerm(func=mdp.success_termination)
