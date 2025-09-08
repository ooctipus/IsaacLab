import isaaclab.envs.mdp as orbit_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import numpy as np

from ... import assembly_data
from ... import mdp as task_mdp
from .one_leg_ur5 import OneLegUr5RelJointPosition, TerminationsCfg, EventCfg

@configclass
class BaseEventCfg(EventCfg):
    """Configuration for randomization."""

    reset_table_top_position = EventTerm(
        func=orbit_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # TODO(pat): table reset position conservative
            "pose_range": {
                "x": (0.4, 0.5),
                "y": (0.1, 0.2),
                "z": (0.001, 0.011),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-np.pi / 12, np.pi / 12),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("table_top"),
        },
    )

@configclass
class PositionRangeTerminationCfg(TerminationsCfg):
    position_range_termination = DoneTerm(
        func=task_mdp.check_position_range,
        params={
            "asset_cfg": SceneEntityCfg("leg"),
            "x_range": None,  # Skip x range check
            "y_range": None,  # Skip y range check
            "z_range": (0.0, 1.0),  # 100cm range in z
        }
    )

@configclass
class InitStatesTerminationCfg(TerminationsCfg):
    """Configuration for pose deviation termination."""

    position_range_termination = DoneTerm(
        func=task_mdp.check_position_range,
        params={
            "asset_cfg": SceneEntityCfg("leg"),
            "x_range": None,  # Skip x range check
            "y_range": None,  # Skip y range check
            "z_range": (0.0, 1.0),  # 100cm range in z
        }
    )

    pose_deviation_termination = DoneTerm(func=task_mdp.check_pose_deviation)


@configclass
class ReachingEventCfg(BaseEventCfg):
    """Configuration for randomization."""

    reset_end_effector = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,  # type: ignore
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset_offset": None,
            "pose_range_b": {
                "x": (0.3, 0.7), "y": (-0.4, 0.4), "z": (0.15, 0.5),
                "roll": (0.0, 0.0), "pitch": (np.pi / 4, 3 * np.pi / 4), "yaw": (np.pi / 2, 3 * np.pi / 2)
            },
            "robot_ik_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"),
        },
    )

    reset_leg_position = EventTerm(
        func=orbit_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.4, 0.5),
                "y": (0.3, 0.4),
                "z": (0.1, 0.1),
                "roll": (0.0, 0.0),
                "pitch": (np.pi / 2, np.pi / 2),
                "yaw": (- np.pi / 4, np.pi / 4),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("leg"),
        },
    )

@configclass
class GraspedEventCfg(BaseEventCfg):
    """Configuration for randomization."""

    reset_leg_position = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "datasets": ["furniture_datasets/reaching_init_states_rgb_narrow_dataset_preprocessed.pt"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "failure_rate_sampling": False
        }
    )

    reset_end_effector = EventTerm(
        func=task_mdp.reset_end_effector_grasp_fixed_asset,  # type: ignore
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("leg"),
            "fixed_asset_offset": assembly_data.KEYPOINTS_TABLELEG.graspable,
            "robot_object_held_offset": assembly_data.KEYPOINTS_ROBOTIQGRIPPER.offset,
            "support_asset_cfg": SceneEntityCfg("table"),
            "pose_range_b": {
                "x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.04)  #(-0.02, 0.01)
            },
            # TODO(pat): 30-60 degree range
            "grasp_angle_range": (0.3, 0.7),
            "yaw_choices": [0.0, np.pi, 2*np.pi],
            "robot_ik_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"),
        },
    )

    grasp_held_asset = EventTerm(
        func=task_mdp.robotiq_gripper_grasp_held_asset,
        mode="reset",
        params={
            "robot_ik_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_inner_finger", "right_inner_finger"],
                joint_names=["right.*", "left.*", "finger_joint"],
            ),
            "held_asset_diameter": assembly_data.KEYPOINTS_TABLELEG.diameter
        }
    )

@configclass
class InsertionEventCfg(BaseEventCfg):
    """Configuration for randomization."""

    reset_end_effector = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,  # type: ignore
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset_offset": None,
            "pose_range_b": {
                "x": (0.3, 0.7), "y": (-0.4, 0.4), "z": (0.15, 0.5),
                "roll": (0.0, 0.0), "pitch": (np.pi / 4, 3 * np.pi / 4), "yaw": (np.pi / 2, 3 * np.pi / 2)
            },
            "robot_ik_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"),
        },
    )

    reset_held_asset_in_hand = EventTerm(
        func=task_mdp.reset_held_asset,
        mode="reset",
        params={
            "holding_body_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "held_asset_cfg": SceneEntityCfg("leg"),
            "robot_object_held_offset": assembly_data.KEYPOINTS_ROBOTIQGRIPPER.offset,
            "held_asset_graspable_offset": assembly_data.KEYPOINTS_TABLELEG.graspable,
            "held_asset_inhand_range": {
                # TODO(pat): 60 degree range
                "pitch": (-1, 1),
            }
        },
    )

    grasp_held_asset = EventTerm(
        func=task_mdp.robotiq_gripper_grasp_held_asset,
        mode="reset",
        params={
            "robot_ik_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_inner_finger", "right_inner_finger"],
                joint_names=["right.*", "left.*", "finger_joint"],
            ),
            "held_asset_diameter": assembly_data.KEYPOINTS_TABLELEG.diameter
        }
    )


@configclass
class AssembledEventCfg(BaseEventCfg):
    """Configuration for randomization."""

    reset_leg_in_hole = EventTerm(
        func=task_mdp.reset_held_asset,
        mode="reset",
        params={
            "holding_body_cfg": SceneEntityCfg("table_top"),
            "held_asset_cfg": SceneEntityCfg("leg"),
            "robot_object_held_offset": assembly_data.KEYPOINTS_TABLETOPHOLE.hole0_leg_assembled_offset,
            "held_asset_graspable_offset": assembly_data.KEYPOINTS_TABLELEG.center_axis_bottom,
            "held_asset_inhand_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.025),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            }
        },
    )

    reset_end_effector = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,  # type: ignore
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset_offset": None,
            "pose_range_b": {
                "x": (0.3, 0.7), "y": (-0.4, 0.4), "z": (0.15, 0.5),
                "roll": (0.0, 0.0), "pitch": (np.pi / 4, 3 * np.pi / 4), "yaw": (np.pi / 2, 3 * np.pi / 2)
            },
            "robot_ik_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"),
        },
    )



@configclass
class AssembledGraspedEventCfg(BaseEventCfg):
    """Configuration for randomization."""

    reset_leg_in_hole = EventTerm(
        func=task_mdp.reset_held_asset,
        mode="reset",
        params={
            "holding_body_cfg": SceneEntityCfg("table_top"),
            "held_asset_cfg": SceneEntityCfg("leg"),
            "robot_object_held_offset": assembly_data.KEYPOINTS_TABLETOPHOLE.hole0_leg_assembled_offset,
            "held_asset_graspable_offset": assembly_data.KEYPOINTS_TABLELEG.center_axis_bottom,
            "held_asset_inhand_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.025),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            }
        },
    )

    reset_end_effector = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,  # type: ignore
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("leg"),
            "fixed_asset_offset": assembly_data.KEYPOINTS_TABLELEG.graspable,
            "robot_object_held_offset": assembly_data.KEYPOINTS_ROBOTIQGRIPPER.offset,
            "pose_range_b": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (-0.015, 0.025),
                # TODO(pat): 60 degree range
                "roll": (0.0, 0.0), "pitch": (-1., 1.), "yaw": (0.0, 2*np.pi)
            },
            "robot_ik_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"),
        },
    )

    grasp_held_asset = EventTerm(
        func=task_mdp.robotiq_gripper_grasp_held_asset,
        mode="reset",
        params={
            "robot_ik_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_inner_finger", "right_inner_finger"],
                joint_names=["right.*", "left.*", "finger_joint"],
            ),
            "held_asset_diameter": assembly_data.KEYPOINTS_TABLELEG.diameter
        }
    )

@configclass
class OneLegUr5RGBRelJointPositionReaching(OneLegUr5RelJointPosition):
    events: ReachingEventCfg = ReachingEventCfg()
    terminations: PositionRangeTerminationCfg = PositionRangeTerminationCfg()

@configclass
class OneLegUr5RGBRelJointPositionGrasped(OneLegUr5RelJointPosition):
    events: GraspedEventCfg = GraspedEventCfg()
    terminations: InitStatesTerminationCfg = InitStatesTerminationCfg()

@configclass
class OneLegUr5RGBRelJointPositionInsertion(OneLegUr5RelJointPosition):
    events: InsertionEventCfg = InsertionEventCfg()
    terminations: InitStatesTerminationCfg = InitStatesTerminationCfg()

@configclass
class OneLegUr5RGBRelJointPositionAssembled(OneLegUr5RelJointPosition):
    events: AssembledEventCfg = AssembledEventCfg()
    terminations: InitStatesTerminationCfg = InitStatesTerminationCfg()

@configclass
class OneLegUr5RGBRelJointPositionAssembledGrasped(OneLegUr5RelJointPosition):
    events: AssembledGraspedEventCfg = AssembledGraspedEventCfg()
    terminations: InitStatesTerminationCfg = InitStatesTerminationCfg()
