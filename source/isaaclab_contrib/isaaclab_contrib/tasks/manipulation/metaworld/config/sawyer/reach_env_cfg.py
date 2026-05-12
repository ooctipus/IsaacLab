# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MT3 reach env: Sawyer rig + V2 reach reward.

Reach uses *uniform-rectangle sampling* (not fixed pairs like the 15
real-asset tasks), so its command + reward cfgs live here rather than in
:mod:`metaworld_specs`.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from ...metaworld_env_base import MetaworldEnvCfg
from ...metaworld_scenes_cfg import SawyerCubeSceneCfg

_TCP_FRAME_CFG = SceneEntityCfg("tcp_frame")


@configclass
class _CommandsCfg:
    """Reach: cube on table, goal floats above (z up to 0.30 m)."""

    ee_pose = mdp.MetaworldPairedCommandCfg(
        # One sample per episode (paper-faithful — no mid-episode resampling).
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=False,
        object_name="cube",
        frame_transformer_name="tcp_frame",
        obj_low=(-0.1, 0.6, 0.02),
        obj_high=(0.1, 0.7, 0.02),
        goal_low=(-0.1, 0.8, 0.05),
        goal_high=(0.1, 0.9, 0.30),
    )


@configclass
class _RewardsCfg:
    """V2 reach reward — pure tolerance shape, scaled to ``[0, 10]``."""

    reach_v2 = RewardTermCfg(
        func=mdp.tolerance_shape,
        weight=1.0,
        params={
            "cfg": mdp.ToleranceShapeCfg(
                distance=mdp.tcp_to_target_dist,
                distance_kwargs={
                    "frame_transformer_cfg": _TCP_FRAME_CFG,
                    "goal_command_name": "ee_pose",
                },
                # Tolerance margin = ``‖hand_init - target‖``. We pin
                # ``hand_init`` to the realised TCP pose produced by our
                # default joint config. MW achieves this via ``_reset_hand``
                # driving the mocap to ``hand_init_pos = (0, 0.6, 0.2)``;
                # we don't run that loop, so we use the realised pose as
                # the constant.
                margin=mdp.hand_init_to_target_dist,
                margin_kwargs={
                    "goal_command_name": "ee_pose",
                    "hand_init_pos_e": (-0.067, 0.571, 0.132),
                },
                success_radius=mdp.REACH_TARGET_RADIUS,
                scale=10.0,
            )
        },
    )

    success = RewardTermCfg(
        func=mdp.reach_success,
        weight=0.0,
        params={
            "frame_transformer_cfg": _TCP_FRAME_CFG,
            "goal_command_name": "ee_pose",
        },
    )

    action_rate = RewardTermCfg(func="isaaclab.envs.mdp:action_rate_l2", weight=-1e-4)


@configclass
class MetaworldReachSawyerEnvCfg(MetaworldEnvCfg):
    """Reach with Sawyer rig."""

    scene: SawyerCubeSceneCfg = SawyerCubeSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: _CommandsCfg = _CommandsCfg()
    rewards: _RewardsCfg = _RewardsCfg()
