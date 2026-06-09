# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from ...mdp.terminations import BaseTerminationsCfg
from .. import mdp
from ..factory_presets import FactoryAssemblyProfileCfg, HeldAssetAlignOffsetCfg



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
    progress_context = DoneTerm(
        func=mdp.progress_context,  # type: ignore
        params={
            "success_threshold": 0.001,
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            "held_asset_offset": HeldAssetAlignOffsetCfg(),
            "assembly_profile": FactoryAssemblyProfileCfg(),
        },
    )


@configclass
class SuccessTerminationsV0Cfg(BaseTerminationsCfg):
    """Legacy termination terms for the success-terminate formulation."""

    oob = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "in_bound_range": {"x": (-0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)},
        },
    )
    progress_context = DoneTerm(
        func=mdp.progress_context,  # type: ignore
        params={
            "success_threshold": 0.001,
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            "held_asset_offset": HeldAssetAlignOffsetCfg(),
            "assembly_profile": FactoryAssemblyProfileCfg(),
        },
    )
    success = DoneTerm(func=mdp.success_termination)


@configclass
class SuccessTerminationsV1Cfg(BaseTerminationsCfg):
    """Contact-aware termination terms for the success-terminate formulation."""

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
    progress_context = DoneTerm(
        func=mdp.progress_context,  # type: ignore
        params={
            "success_threshold": 0.001,
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            "held_asset_offset": HeldAssetAlignOffsetCfg(),
            "assembly_profile": FactoryAssemblyProfileCfg(),
        },
    )
    success = DoneTerm(func=mdp.success_termination)
