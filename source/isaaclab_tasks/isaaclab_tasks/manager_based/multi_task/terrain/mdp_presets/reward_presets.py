# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import PresetCfg

from .. import mdp
from .robots.robot_presets import FootBodyNamesCfg


@configclass
class PositionRewardsCfg:
    # task rewards
    success = RewTerm(func=mdp.command_success, weight=50.0)

    mech_work = RewTerm(func=mdp.mechanical_power, weight=-0.0001)

    undesired_contact = RewTerm(
        func=mdp.contact_penalty,
        weight=-0.02,
        params={
            "exclude_contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FootBodyNamesCfg(),  # type: ignore
            ),
            "threshold": 1.0,
        },
    )

    fail = RewTerm(func=mdp.is_terminated_term, params={"term_keys": ["oob", "base_contact"]}, weight=-25.0)


@configclass
class EmptyRewardsCfg:
    """No rewards. Used by self-supervised algorithms (e.g. CRL) where
    learning is driven by a contrastive loss, not a reward signal."""

    pass


@configclass
class AdvancedSkillsRewardsCfg:
    pass
    # TODO(Mateo)


@configclass
class RewardsCfg(PresetCfg):
    position = PositionRewardsCfg()
    crl = EmptyRewardsCfg()
    advanced_skills = AdvancedSkillsRewardsCfg()
    default = position
