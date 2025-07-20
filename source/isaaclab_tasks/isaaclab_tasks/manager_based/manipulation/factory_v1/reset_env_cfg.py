from dataclasses import MISSING
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from . import mdp


STAGING_EVENTS = EventTerm(
    func=mdp.TermChoice,
    mode="reset",
    params={
        "terms": {
            "reset_held_asset_on_fixed_asset": EventTerm(
                func=mdp.reset_held_asset_on_fixed_asset,
                mode="reset",
                params={
                    "assembled_offset": MISSING,
                    "entry_offset": MISSING,
                    "assembly_fraction_range": (0., 1.),
                    "assembly_ratio": (0., 0., 0.),
                    "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
                    "held_asset_cfg": SceneEntityCfg("held_asset"),
                }
            ),
            "reset_asset_on_table": EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.015, 0.015)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("held_asset")
                }
            ),
            "reset_asset_in_air": EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.015, 0.015)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("held_asset")
                }
            ),
        },
        "sampling_strategy": "uniform"
    }
)

PLAYER_EVENTS = EventTerm(
    func=mdp.TermChoice,
    mode="reset",
    params={
        "terms": {},
        "sampling_strategy": "uniform"
    }
)

TASK_ASSIGNING_EVENTS = EventTerm(
    func=mdp.TermChoice,
    mode="reset",
    params={
        "terms": {
            "reset_end_effector_around_fixed_asset": EventTerm(
                func=mdp.reset_end_effector_around_asset,
                mode="reset",
                params={
                    "fixed_asset_cfg": MISSING,
                    "fixed_asset_offset": MISSING,
                    "pose_range_b": MISSING,
                    "robot_ik_cfg": SceneEntityCfg("robot"),
                }
            ),
        },
        "sampling_strategy": "uniform"
    }
)


PLAYER_PREPARE_FOR_TASK_EVENTS = EventTerm(
    func=mdp.TermChoice,
    mode="reset",
    params={
        "terms": {
            "move_held_asset_in_hand_then_grasp": EventTerm(
                func=mdp.ChainedResetTerms,
                mode="reset",
                params={
                    "terms": {
                        "reset_held_asset_in_hand": EventTerm(
                            func=mdp.reset_held_asset_in_gripper,
                            mode="reset",
                            params={
                                "holding_body_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                                "held_asset_cfg": SceneEntityCfg("held_asset"),
                                "held_asset_graspable_offset": MISSING,
                                "held_asset_inhand_range": {},
                            }
                        ),
                        "grasp_held_asset": EventTerm(
                            func=mdp.grasp_held_asset,
                            mode="reset",
                            params={
                                "robot_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                                "held_asset_diameter": MISSING
                            }
                        )
                    }
                }
            ),
            "move_and_grasb_asset": EventTerm(
                func=mdp.ChainedResetTerms,
                mode="reset",
                params={
                    "terms":{
                        "reset_end_effector_around_held_asset": EventTerm(
                            func=mdp.reset_end_effector_around_asset,
                            mode="reset",
                            params={
                                "fixed_asset_cfg": MISSING,
                                "fixed_asset_offset": MISSING,
                                "pose_range_b": MISSING,
                                "robot_ik_cfg": SceneEntityCfg("robot"),
                                "ik_iterations": 30,
                            }
                        ),
                        "grasp_held_asset": EventTerm(
                            func=mdp.grasp_held_asset,
                            mode="reset",
                            params={
                                "robot_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                                "held_asset_diameter": MISSING
                            }
                        ),
                    }
                }
            )
        },
        "sampling_strategy": "uniform"
    }
)
