# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .. import mdp
from ..factory_presets import EndEffectorBodyCfg, JointEffortNamesCfg


@configclass
class TimeoutRewardsCfg:
    """Reward terms for the timeout-terminate formulation (success is not terminal)."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-1e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-1e-4)
    joint_effort = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointEffortNamesCfg())},  # type:ignore
        weight=-1e-4,
    )
    early_termination = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "abnormal"}, weight=-0.01)  # type: ignore
    reach_reward = RewTerm(
        func=mdp.reach_reward,
        weight=0.1,
        params={
            "std": 1.0,
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "ee_cfg": SceneEntityCfg("robot", body_names=EndEffectorBodyCfg()),  # type:ignore
        },
    )
    progress_reward_fine = RewTerm(func=mdp.progress_reward, weight=0.1, params={"std": 0.005})
    success_reward = RewTerm(func=mdp.success_reward, weight=1.0)


@configclass
class SuccessRewardsV0Cfg:
    """Legacy reward terms for the success-terminate formulation."""

    mech_work = RewTerm(func=mdp.mechanical_power, weight=-0.000025)
    early_termination = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "abnormal"}, weight=-0.01)  # type: ignore
    success_reward = RewTerm(func=mdp.success_reward, weight=100.0)


@configclass
class SuccessRewardsV1Cfg:
    """Contact-aware reward terms for the success-terminate formulation."""

    success_reward = RewTerm(func=mdp.success_reward, weight=5.0)
    mech_work = RewTerm(func=mdp.mechanical_power, weight=-0.000025)
    undesired_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.01,
        params={"sensor_cfg": SceneEntityCfg("contact_sensors", body_names="^(?!.*fingertip).*$"), "threshold": 1.0},
    )
    # early_termination = RewTerm(func=mdp.is_terminated_term, params={"term_keys": ["abnormal", "bad_contact", "oob", "joint_reaction"]}, weight=-0.1)  # type: ignore
