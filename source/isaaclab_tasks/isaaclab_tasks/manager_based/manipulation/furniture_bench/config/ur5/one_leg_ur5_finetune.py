from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

import isaaclab_assets.robots.ur5 as ur5
from .one_leg_ur5 import OneLegUr5RelJointPosition, EvalEventCfg

from ... import mdp as task_mdp
from ... import assembly_data

from ... import assembly_data
from ... import mdp as task_mdp


@configclass
class FinetuneRewardsCfg:
    """Reward terms for finetuning with action penalties."""

    # safety reward

    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)

    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-1e-4)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
    )

    # success reward
    progress_context = RewTerm(
        func=task_mdp.ProgressContext,  # type: ignore
        weight=0.1,
        params={
            "held_asset_cfg": SceneEntityCfg("leg"),
            "fixed_asset_cfg": SceneEntityCfg("table_top"),
            "held_asset_offset": assembly_data.KEYPOINTS_TABLELEG.center_axis_bottom,
            "fixed_asset_offset": assembly_data.KEYPOINTS_TABLETOPHOLE.hole0_leg_assembled_offset,
        },
    )

    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0)


@configclass
class OneLegUr5FinetuneRelJointPosition(OneLegUr5RelJointPosition):
    """OneLegUr5 environment configuration for finetuning with larger action penalties."""

    rewards: FinetuneRewardsCfg = FinetuneRewardsCfg()
    actions: ur5.Ur5RelativeJointPositionActionClipped = ur5.Ur5RelativeJointPositionActionClipped()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 48.0

@configclass
class OneLegUr5FinetuneEvalRelJointPosition(OneLegUr5FinetuneRelJointPosition):
    """OneLegUr5 environment configuration for finetuning with larger action penalties."""

    events: EvalEventCfg = EvalEventCfg()
