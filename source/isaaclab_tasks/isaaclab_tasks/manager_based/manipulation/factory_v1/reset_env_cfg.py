from dataclasses import MISSING
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from . import mdp


GRIPPER_GRASP_ASSET_IN_AIR = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms":{
            "reset_asset_in_air": EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.015, 0.2)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("held_asset")
                }
            ),
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
                    "robot_cfg": SceneEntityCfg("robot", body_names="end_effector"), "held_asset_diameter": MISSING
                }
            ),
        }
    }
)

ASSEMBLE_FISRT_THEN_GRIPPER_CLOSE = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms":{
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
                    "robot_cfg": SceneEntityCfg("robot", body_names="end_effector"), "held_asset_diameter": MISSING
                }
            ),
        }
    }
)

GRIPPER_CLOSE_FIRST_THEN_ASSET_IN_GRIPPER = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms":{
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
                    "robot_cfg": SceneEntityCfg("robot", body_names="end_effector"), "held_asset_diameter": MISSING
                }
            ),
        }
    }
)