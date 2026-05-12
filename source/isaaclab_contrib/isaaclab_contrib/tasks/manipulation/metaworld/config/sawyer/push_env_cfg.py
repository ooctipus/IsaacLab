# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MT3 push env: Sawyer rig + V2 push reward.

Push uses *uniform-rectangle sampling*, so its command + reward cfgs live
here rather than in :mod:`metaworld_specs` (which holds fixed-pair
specs for the 15 real-asset tasks).
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from ...metaworld_env_base import MetaworldEnvCfg
from ...metaworld_scenes_cfg import SawyerCubeSceneCfg

_TCP_FRAME_CFG = SceneEntityCfg("tcp_frame")
_OBJECT_CFG = SceneEntityCfg("keypoint_frame")


@configclass
class _CommandsCfg:
    """Push: cube on table, goal also at table height (cube slides to goal)."""

    ee_pose = mdp.MetaworldPairedCommandCfg(
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=False,
        object_name="cube",
        frame_transformer_name="tcp_frame",
        obj_low=(-0.1, 0.6, 0.02),
        obj_high=(0.1, 0.7, 0.02),
        goal_low=(-0.1, 0.8, 0.01),
        goal_high=(0.1, 0.9, 0.02),
    )


@configclass
class _RewardsCfg:
    """V2 push reward — linear-combo shape ``2 * caging`` + phase bonus + override.

    The phase bonus's ``self_mult=1.0`` reproduces Meta-World's verbatim
    ``reward += 1 + reward + 5 * in_place``.
    """

    push_v2 = RewardTermCfg(
        func=mdp.linear_combo_shape,
        weight=1.0,
        params={
            "cfg": mdp.LinearComboShapeCfg(
                terms=[
                    mdp.WeightedAtomCfg(
                        weight=2.0,
                        atom=mdp.gripper_caging,
                        atom_kwargs={
                            "frame_transformer_cfg": _TCP_FRAME_CFG,
                            "keypoint_frame_cfg": _OBJECT_CFG,
                            "goal_command_name": "ee_pose",
                            "params": mdp.GripperCagingParams(
                                obj_radius=0.015,
                                pad_success_thresh=0.05,
                                object_reach_radius=0.01,
                                xz_thresh=0.005,
                                high_density=True,
                            ),
                        },
                    ),
                ],
                in_place_distance=mdp.obj_to_target_dist,
                in_place_distance_kwargs={
                    "keypoint_frame_cfg": _OBJECT_CFG,
                    "goal_command_name": "ee_pose",
                },
                in_place_margin=mdp.obj_init_to_target_dist,
                in_place_margin_kwargs={"goal_command_name": "ee_pose"},
                in_place_success_radius=mdp.PUSH_TARGET_RADIUS,
                phase=mdp.PhaseBonusCfg(
                    triggers=[
                        mdp.TriggerCfg(
                            atom=mdp.tcp_to_obj_dist,
                            op="<",
                            threshold=0.02,
                            atom_kwargs={
                                "frame_transformer_cfg": _TCP_FRAME_CFG,
                                "keypoint_frame_cfg": _OBJECT_CFG,
                            },
                        ),
                        mdp.TriggerCfg(atom=mdp.gripper_open, op=">", threshold=0.0),
                    ],
                    offset=1.0,
                    in_place_mult=5.0,
                    self_mult=1.0,
                ),
                success_override=mdp.SuccessOverrideCfg(
                    quantity=mdp.obj_to_target_dist,
                    threshold=mdp.PUSH_TARGET_RADIUS,
                    op="<",
                    value=10.0,
                    atom_kwargs={
                        "keypoint_frame_cfg": _OBJECT_CFG,
                        "goal_command_name": "ee_pose",
                    },
                ),
            )
        },
    )

    success = RewardTermCfg(
        func=mdp.keypoint_at_target,
        weight=0.0,
        params={
            "goal_command_name": "ee_pose",
            "keypoint_frame_cfg": _OBJECT_CFG,
        },
    )

    action_rate = RewardTermCfg(func="isaaclab.envs.mdp:action_rate_l2", weight=-1e-4)


@configclass
class MetaworldPushSawyerEnvCfg(MetaworldEnvCfg):
    """Push with Sawyer rig."""

    scene: SawyerCubeSceneCfg = SawyerCubeSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: _CommandsCfg = _CommandsCfg()
    rewards: _RewardsCfg = _RewardsCfg()
