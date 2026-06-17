# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .. import mdp


@configclass
class PositionRewardsCfg:
    # task rewards
    success = RewTerm(func=mdp.command_success, weight=5.0)

    mech_work = RewTerm(func=mdp.mechanical_power, weight=-0.000025)

    undesired_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.01,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*(?:(FOOT))).*$"), "threshold": 1.0},
    )


@configclass
class PositionComposedRewardsCfg:
    reward_composer = RewTerm(
        func=mdp.reward_compose,
        weight=1.0,
        params={
            "success": RewTerm(func=mdp.command_success, weight=5.0),
            "quality": {
                "mech_work": RewTerm(func=mdp.mechanical_power, weight=-0.000025),
                "undesired_contact": RewTerm(
                    func=mdp.undesired_contacts,
                    weight=-0.01,
                    params={
                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*(?:(FOOT))).*$"),
                        "threshold": 1.0,
                    },
                ),
            },
        },
    )


@configclass
class EmptyRewardsCfg:
    """No rewards. Used by self-supervised algorithms (e.g. CRL) where
    learning is driven by a contrastive loss, not a reward signal."""

    pass


@configclass
class RewardsCfg(PresetCfg):
    rew_v1 = PositionRewardsCfg()
    rew_v2 = PositionComposedRewardsCfg()
    position = rew_v1
    crl = EmptyRewardsCfg()
    default = rew_v1
