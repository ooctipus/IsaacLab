# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.core.multi_task.curriculum import (
    BetaSamplingStrategyCfg,
    FrontierSamplingStrategyCfg,
    SamplerCfg,
    SuccessMonitorCfg,
    UniformSamplingStrategyCfg,
    ValueShiftSamplingStrategyCfg,
)
from isaaclab_tasks.utils import preset

from . import mdp
from .assembly_keypoints import NIST_BOARD_CFG
from .factory_presets import (
    EndEffectorBodyCfg,
    FactoryAssemblyProfileCfg,
    FixedAssetMapCfg,
    FixedAssetTipCfg,
    GraspedPoseRangeCfg,
    GripperGraspOffsetCfg,
    GripperJointNamesCfg,
    HeldAssetAlignOffsetCfg,
    HeldAssetGraspDiameterCfg,
    HeldAssetGraspMiddleCfg,
    HeldAssetGraspPointCfg,
    IKJointNamesCfg,
)

GRIPPER_GRASP_ASSET_IN_AIR = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms": {
            "reset_asset_in_air": EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.15, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (0.015, 0.2),
                        "roll": (-1.57, 1.57),
                        "pitch": (-1.57, 1.57),
                        "yaw": (-3.14, 3.14),
                    },
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("held_asset"),
                },
            ),
            "reset_end_effector_around_held_asset": EventTerm(
                func=mdp.reset_end_effector_around_asset,
                mode="reset",
                params={
                    "fixed_asset_cfg": SceneEntityCfg("held_asset"),
                    "fixed_asset_offset": HeldAssetGraspMiddleCfg(),
                    "pose_range_b": {
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.015, 0.025),
                        "roll": (3.141 - 0.1, 3.141 + 0.1),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-2.09, 2.09),
                    },
                    "robot_ik_cfg": SceneEntityCfg(
                        "robot", joint_names=IKJointNamesCfg(), body_names=EndEffectorBodyCfg()
                    ),
                    "ik_iterations": (5, 30),
                },
            ),
            "grasp_held_asset": EventTerm(
                func=mdp.grasp_held_asset,
                mode="reset",
                params={
                    "robot_cfg": SceneEntityCfg(
                        "robot", joint_names=GripperJointNamesCfg(), body_names=EndEffectorBodyCfg()
                    ),
                    "held_asset_diameter": HeldAssetGraspDiameterCfg(),
                },
            ),
        }
    },
)


ASSEMBLE_FIRST_THEN_GRIPPER_CLOSE = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms": {
            "reset_held_asset_on_fixed_asset": EventTerm(
                func=mdp.reset_held_asset_on_fixed_asset,
                mode="reset",
                params={
                    "assembly_profile": FactoryAssemblyProfileCfg(),
                    "held_asset_align_offset": HeldAssetAlignOffsetCfg(),
                    "assembly_fraction_range": (0.0, 1.1),
                    "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
                    "held_asset_cfg": SceneEntityCfg("held_asset"),
                },
            ),
            "reset_end_effector_around_held_asset": EventTerm(
                func=mdp.reset_end_effector_around_asset,
                mode="reset",
                params={
                    "fixed_asset_cfg": SceneEntityCfg("held_asset"),
                    "fixed_asset_offset": HeldAssetGraspMiddleCfg(),
                    "pose_range_b": {
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.015, 0.025),
                        "roll": (3.141 - 0.1, 3.141 + 0.1),
                        "pitch": (-1.0, 1.0),
                        "yaw": (-2.09, 2.09),
                    },
                    "robot_ik_cfg": SceneEntityCfg(
                        "robot", joint_names=IKJointNamesCfg(), body_names=EndEffectorBodyCfg()
                    ),
                    "ik_iterations": (15, 25),
                },
            ),
            "grasp_held_asset": EventTerm(
                func=mdp.grasp_held_asset,
                mode="reset",
                params={
                    "robot_cfg": SceneEntityCfg(
                        "robot", joint_names=GripperJointNamesCfg(), body_names=EndEffectorBodyCfg()
                    ),
                    "held_asset_diameter": HeldAssetGraspDiameterCfg(),
                },
            ),
        }
    },
)

GRIPPER_CLOSE_FIRST_THEN_ASSET_IN_GRIPPER = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms": {
            "reset_end_effector_around_fixed_asset": EventTerm(
                func=mdp.reset_end_effector_around_asset,
                mode="reset",
                params={
                    "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
                    "fixed_asset_offset": FixedAssetTipCfg(),
                    "pose_range_b": GraspedPoseRangeCfg(),
                    "robot_ik_cfg": SceneEntityCfg(
                        "robot", joint_names=IKJointNamesCfg(), body_names=EndEffectorBodyCfg()
                    ),
                    "ik_iterations": (10, 20),
                },
            ),
            "reset_held_asset_in_hand": EventTerm(
                func=mdp.reset_held_asset_in_gripper,
                mode="reset",
                params={
                    "holding_body_cfg": SceneEntityCfg("robot", body_names=EndEffectorBodyCfg()),
                    "held_asset_cfg": SceneEntityCfg("held_asset"),
                    "held_asset_graspable_offset": HeldAssetGraspPointCfg(),
                    "held_asset_inhand_range": {},
                    "gripper_grasp_offset": GripperGraspOffsetCfg(),
                },
            ),
            "grasp_held_asset": EventTerm(
                func=mdp.grasp_held_asset,
                mode="reset",
                params={
                    "robot_cfg": SceneEntityCfg(
                        "robot", joint_names=GripperJointNamesCfg(), body_names=EndEffectorBodyCfg()
                    ),
                    "held_asset_diameter": HeldAssetGraspDiameterCfg(),
                    "flexible_angle": False,
                },
            ),
        }
    },
)


SCENE_RESET = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms": {
            "reset_robot": EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.0, 0.0),
                    "velocity_range": (-0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "reset_board": EventTerm(
                func=mdp.reset_root_state_uniform_on_offset,
                mode="reset",
                params={
                    "offset": NIST_BOARD_CFG.nist_board_center,
                    "pose_range": {"x": (-0.00, 0.00), "y": (-0.05, 0.05), "yaw": (-3.14, 3.14)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("nistboard"),
                },
            ),
            "reset_fixed_asset": EventTerm(
                func=mdp.reset_fixed_assets,
                mode="reset",
                params={
                    "asset_map": FixedAssetMapCfg(),
                },
            ),
            "reset_strategies": EventTerm(
                func=mdp.TermChoice,
                mode="reset",
                params={
                    "terms": preset(
                        default={
                            "grasp_asset_in_air": GRIPPER_GRASP_ASSET_IN_AIR,
                            "start_assembled": ASSEMBLE_FIRST_THEN_GRIPPER_CLOSE,
                            "start_grasped_then_assembled": GRIPPER_CLOSE_FIRST_THEN_ASSET_IN_GRIPPER,
                        },
                        eval={"grasp_asset_in_air": ASSEMBLE_FIRST_THEN_GRIPPER_CLOSE},
                    ),
                    "sampling": preset(
                        default=SamplerCfg(
                            strategies=[
                                BetaSamplingStrategyCfg(
                                    target=0.5, kappa=1.0, weight=1.0, success_rate_bind="success_rates"
                                )
                            ],
                            eps=1e-3,
                        ),
                        uniform=SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0),
                        monitor=SamplerCfg(
                            strategies=[
                                BetaSamplingStrategyCfg(
                                    target=0.5, kappa=1.0, weight=1.0, success_rate_bind="success_rates"
                                )
                            ],
                            eps=1e-3,
                        ),
                    ),
                    "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=100),
                    "report": preset(accumulator=False, choice=True, default=False),
                },
            ),
        }
    },
)


FACTORY_RESET_SAMPLER_PRESETS = preset(
    default=SamplerCfg(
        strategies=[BetaSamplingStrategyCfg(target=0.5, kappa=1.0, weight=1.0, success_rate_bind="success_rates")],
        eps=1e-3,
    ),
    uniform=SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0),
    monitor=SamplerCfg(
        strategies=[BetaSamplingStrategyCfg(target=0.5, kappa=1.0, weight=1.0, success_rate_bind="success_rates")],
        eps=1e-3,
    ),
    # ``beta`` is a semantic alias of ``monitor``: same Beta rolling-monitor
    # curriculum. Useful as a no-frontier baseline when sweeping ``frontier`` and
    # ``dil*`` so run names read "what's the curriculum?" rather than "what's the
    # rate source?".
    beta=SamplerCfg(
        strategies=[BetaSamplingStrategyCfg(target=0.5, kappa=1.0, weight=1.0, success_rate_bind="success_rates")],
        eps=1e-3,
    ),
    frontier=SamplerCfg(
        strategies=[
            BetaSamplingStrategyCfg(target=0.5, kappa=1.0, weight=1.0, success_rate_bind="success_rates"),
            FrontierSamplingStrategyCfg(
                k=8,
                dilation_steps=preset(default=2, dil1=1, dil2=2, dil3=3, dil4=4, dil5=5),  # type: ignore
                weight=0.5,
                success_rate_bind="success_rates",
            ),
        ],
        eps=1e-3,
    ),
    # Value-shift prioritizes table slots whose critic value moved most between
    # updates. The binds reach the accumulator event term for stored states and
    # slot application; the strategy itself lives on the curriculum sampler.
    value_shift=SamplerCfg(
        strategies=[
            ValueShiftSamplingStrategyCfg(
                weight=1.0,
                state_buffer_bind="env.event_manager.get_term_cfg('reset_strategies').func.state_data",
                cmd_indices_bind="env.event_manager.get_term_cfg('reset_strategies').func.sampled_slots",
                resample_command_fn_bind="env.event_manager.get_term_cfg('reset_strategies').func.apply_sampled_slots",
                get_critic_obs_fn_bind="lambda: env.observation_manager.compute()",
            )
        ],
        eps=1e-3,
    ),
    beta_value_shift=SamplerCfg(
        strategies=[
            BetaSamplingStrategyCfg(target=0.5, kappa=1.0, weight=1.0, success_rate_bind="success_rates"),
            ValueShiftSamplingStrategyCfg(
                weight=0.05,
                state_buffer_bind="env.event_manager.get_term_cfg('reset_strategies').func.state_data",
                cmd_indices_bind="env.event_manager.get_term_cfg('reset_strategies').func.sampled_slots",
                resample_command_fn_bind="env.event_manager.get_term_cfg('reset_strategies').func.apply_sampled_slots",
                get_critic_obs_fn_bind="lambda: env.observation_manager.compute()",
            ),
        ],
        eps=1e-3,
    ),
)


ACCUMULATOR_RESET = EventTerm(
    func=mdp.reset_accumulator,
    mode="reset",
    params={
        "reset_assets": ["nistboard", "fixed_asset", "held_asset", "robot"],
        "acceptance_conditions": {
            "object_collision_free": mdp.CollisionAnalyzerCfg(
                num_points=32,
                max_dist=0.5,
                min_dist=-0.0005,
                asset_cfg=SceneEntityCfg("held_asset"),
                obstacle_cfgs=[SceneEntityCfg("fixed_asset"), SceneEntityCfg("robot")],
            ),
        },
        "state_table_size": preset(default=32768, eval=64),
        "state_tag_names_bind": "list(reset_term.func.terms['reset_strategies'].func.term_partitions.keys())",
        "state_tag_indices_bind": "reset_term.func.terms['reset_strategies'].func.term_samples",
        "reset_term": SCENE_RESET,
        "report": True,
    },
)

RESET_STRATEGIES = preset(accumulator=ACCUMULATOR_RESET, choice=SCENE_RESET, default=ACCUMULATOR_RESET)
