import warp as wp
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from ....state_machine import StateCfg as State
from ....state_machine import ConditionCfg as Condition
from ....state_machine import ExecCfg as Exec
from . import conditions as cond
from . import executions as exec
from ....mdp.data_cfg import KeyPointDataCfg as Kp
from ....mdp.data_cfg import AlignmentDataCfg as Align
from ....tasks import Offset


@configclass
class FactoryFrankaState:
    """States for the state machine."""

    REST = wp.constant(0)
    ALIGN_HELD_ASSET = wp.constant(1)
    ALIGN_ASSET = wp.constant(11)
    LIFT_UP = wp.constant(2)
    APPROACH_ABOVE_FIXED_ASSET = wp.constant(3)
    INSERT = wp.constant(4)
    SCREW = wp.constant(5)
    RELEASE = wp.constant(6)
    UNWIND = wp.constant(7)
    PRE_GRASP = wp.constant(8)
    GRASP = wp.constant(9)
    DONE = wp.constant(10)


@configclass
class GripperState:
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


state = FactoryFrankaState()
gripper_state = GripperState()


FRANKA_FACTORY_STATES = {

    state.REST.__str__(): State(
        prev_states=[],
        pre_condition={
            "always": Condition(func=cond.always, args={})},
        ee_exec=Exec(
            func=exec.ee_stay_still,
            args={"ee_src_cfg" : Kp(term="manipulation_key_points", kp_attr="key_points_src", kp_names="robot_object_held")}
        ),
        gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
        limits=(0.02, 0.20),
        noise=0.0005
    ),

    state.ALIGN_HELD_ASSET.__str__(): State(
        prev_states=[state.REST],
        pre_condition={
            "gripper_open": Condition(
                func=cond.gripper_open,
                args={"robot_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint1"), "threshold": 0.035}
            )},
        post_condition={
            "grasp_aligned" : Condition(
                func=cond.gripper_aligned_with_held_asset,
                args={
                    "manipulation_alignment_cfg": Align(term="manipulation_alignment_data"),
                    "pos_threshold": (0.02, 0.02, 0.07),
                    "only_pos": True,
                }
            )},
        ee_exec=Exec(
            func=exec.align_gripper_to_held_asset_grasp_point,
            args={
                "supporting_asset_cfg": SceneEntityCfg("assets", object_collection_names="kit_tray"),
                "grasp_alignment_cfg": Align(term="manipulation_alignment_data"),
                "grasp_kp_cfg": Kp(term="manipulation_key_points", kp_names="asset_grasp"),
                "object_holding_offset_cfg" : Kp(term="manipulation_key_points", kp_attr="key_points_offset", kp_names="robot_object_held"),
                "object_holding_kp_cfg" : Kp(term="manipulation_key_points", kp_names="robot_object_held"),
                "offset": Offset(pos=(0, 0, 0.06)),
            }),
        gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
        limits=(0.02, 0.20),
        noise=0.0005
    ),

    state.PRE_GRASP.__str__(): State(
        prev_states=[state.ALIGN_HELD_ASSET, state.UNWIND],
        pre_condition={
            "gripper_open": Condition(
                func=cond.gripper_open,
                args={"robot_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint1"), "threshold": 0.035}
            )},
        post_condition={
            "grasp_aligned" : Condition(
                func=cond.gripper_aligned_with_held_asset,
                args={
                    "manipulation_alignment_cfg": Align(term="manipulation_alignment_data"),
                    "only_pos": True,
                    "pos_threshold": (0.03, 0.03, 0.03),
                }
            )},
        ee_exec=Exec(
            func=exec.align_gripper_to_held_asset_grasp_point,
            args={
                "supporting_asset_cfg": SceneEntityCfg("assets", object_collection_names="kit_tray"),
                "grasp_alignment_cfg": Align(term="manipulation_alignment_data"),
                "grasp_kp_cfg": Kp(term="manipulation_key_points", kp_names="asset_grasp"),
                "object_holding_offset_cfg" : Kp(term="manipulation_key_points", kp_attr="key_points_offset", kp_names="robot_object_held"),
                "object_holding_kp_cfg" : Kp(term="manipulation_key_points", kp_names="robot_object_held"),
            }),
        gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
        limits=(0.02, 0.20),
        noise=0.0005
    ),

    state.GRASP.__str__(): State(
        prev_states=[state.PRE_GRASP],
        pre_condition={},
        post_condition={
            "grasped" : Condition(
                func=cond.Grasped,
                args={
                    "robot_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint1"),
                    "history_length": 5,
                }
            )},
        ee_exec=Exec(
            func=exec.align_gripper_to_held_asset_grasp_point,
            args={
                "supporting_asset_cfg": SceneEntityCfg("assets", object_collection_names="kit_tray"),
                "grasp_alignment_cfg": Align(term="manipulation_alignment_data"),
                "grasp_kp_cfg": Kp(term="manipulation_key_points", kp_names="asset_grasp"),
                "object_holding_offset_cfg" : Kp(term="manipulation_key_points", kp_attr="key_points_offset", kp_names="robot_object_held"),
                "object_holding_kp_cfg" : Kp(term="manipulation_key_points", kp_names="robot_object_held"),
            }),
        gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
        limits=(0.02, 0.10),
        noise=0.0005
    ),

    state.LIFT_UP.__str__(): State(
        prev_states=[state.GRASP],
        pre_condition={
            "entry_not_aligned" : Condition(
                func=cond.held_asset_insertion_not_position_aligned_with_fixed_asset_entry,
                args={
                    "auxiliary_alignment_cfg": Align(term="auxiliary_task_alignment_data"),
                    "pos_threshold": (0.005, 0.005, 1.0),
                }
            )},
        post_condition={
            "grasped_asset_lifted" : Condition(
                func=cond.held_asset_lifted,
                args={
                    "aligning_key_point_cfg": Kp(term="auxiliary_task_key_points", kp_names="asset_align"),
                    "threshold": 0.20
                }
            )},
        ee_exec=Exec(
            func=exec.lift_up_execution,
            args={
                "supporting_asset_cfg": SceneEntityCfg("assets", object_collection_names="kit_tray"),
                "object_holding_kp_cfg" : Kp(term="manipulation_key_points", kp_names="robot_object_held"),
                "object_holding_offset_cfg" : Kp(term="manipulation_key_points", kp_attr="key_points_offset", kp_names="robot_object_held"),
            }),
        gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
        limits=(0.015, 0.10),
        noise=0.0005
    ),

    state.APPROACH_ABOVE_FIXED_ASSET.__str__(): State(
        prev_states=[state.LIFT_UP, state.REST],
        pre_condition={
            "grasp_aligned" : Condition(
                func=cond.gripper_aligned_with_held_asset,
                args={
                    "manipulation_alignment_cfg": Align(term="manipulation_alignment_data"),
                    "only_pos": True,
                    "pos_threshold": (0.03, 0.03, 0.03),
                }
            ),
            "grasped_asset_lifted" : Condition(
                func=cond.held_asset_lifted,
                args={
                    "aligning_key_point_cfg": Kp(term="auxiliary_task_key_points", kp_names="asset_align"),
                    "threshold": 0.06
                }
            )},
        post_condition={
            "speed_low" : Condition(
                func=cond.SpeedLow,
                args={
                    "robot_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                    "history_length": 10,
                    "speed_limit": 0.0075
                }
            )},
        ee_exec=Exec(
            func=exec.align_holding_asset,
            args={
                "asset_alignment_cfg" : Align(term="auxiliary_task_alignment_data"),
                "grasp_alignment_cfg" : Align(term="manipulation_alignment_data"),
                "asset_align_against_kp_cfg" : Kp(term="auxiliary_task_key_points", kp_names="asset_align_against"),
                "asset_align_src_cfg" : Kp(term="task_key_points", kp_attr="key_points_src", kp_names="asset_align"),
                "asset_align_offset_cfg" : Kp(term="task_key_points", kp_attr="key_points_offset", kp_names="asset_align"),
                "asset_align_kp_cfg": Kp(term="task_key_points", kp_names="asset_align"),
                "object_holding_kp_cfg" : Kp(term="manipulation_key_points", kp_names="robot_object_held"),
                "object_holding_offset_cfg" : Kp(term="manipulation_key_points", kp_attr="key_points_offset", kp_names="robot_object_held"),
                "offset": Offset(pos=(0, 0, 0.005)),
            }),
        gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
        limits=(0.01, 0.20),
        noise=0.00005
    ),

    state.ALIGN_ASSET.__str__(): State(
        prev_states=[state.LIFT_UP, state.REST],
        pre_condition={
            "grasp_aligned" : Condition(
                func=cond.gripper_aligned_with_held_asset,
                args={
                    "manipulation_alignment_cfg": Align(term="manipulation_alignment_data"),
                    "only_pos": True,
                    "pos_threshold": (0.03, 0.03, 0.03),
                }
            ),
            "grasped_asset_lifted" : Condition(
                func=cond.held_asset_lifted,
                args={
                    "aligning_key_point_cfg": Kp(term="auxiliary_task_key_points", kp_names="asset_align"),
                    "threshold": 0.06
                }
            ),
            "task_type_align": Condition(func=cond.task_type, args={"task_type": [0]})},
        post_condition={
            "speed_low" : Condition(
                func=cond.SpeedLow,
                args={
                    "robot_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                    "history_length": 10,
                    "speed_limit": 0.0075
                }
            )},
        ee_exec=Exec(
            func=exec.align_holding_asset,
            args={
                "asset_alignment_cfg" : Align(term="alignment_data"),
                "grasp_alignment_cfg" : Align(term="manipulation_alignment_data"),
                "asset_align_against_kp_cfg" : Kp(term="task_key_points", kp_names="asset_align_against"),
                "asset_align_src_cfg" : Kp(term="task_key_points", kp_attr="key_points_src", kp_names="asset_align"),
                "asset_align_offset_cfg" : Kp(term="task_key_points", kp_attr="key_points_offset", kp_names="asset_align"),
                "asset_align_kp_cfg": Kp(term="task_key_points", kp_names="asset_align"),
                "object_holding_kp_cfg" : Kp(term="manipulation_key_points", kp_names="robot_object_held"),
                "object_holding_offset_cfg" : Kp(term="manipulation_key_points", kp_attr="key_points_offset", kp_names="robot_object_held"),
                "offset": Offset(pos=(0.0, 0.0, 0.01)),
            }),
        gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
        limits=(0.02, 0.20),
        noise=0.00005
    ),

    state.INSERT.__str__(): State(
        prev_states=[state.APPROACH_ABOVE_FIXED_ASSET],
        pre_condition={
            "entry_aligned" : Condition(
                func=cond.held_asset_insertion_aligned_with_fixed_asset_entry,
                args={
                    "auxiliary_alignment_cfg": Align(term="auxiliary_task_alignment_data"),
                    "pos_threshold": (0.005, 0.005, 0.01),
                }
            ),
            "speed_low" : Condition(
                func=cond.SpeedLow,
                args={
                    "robot_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                    "history_length": 3,
                    "speed_limit": 0.0075
                }
            ),
            "insert_or_screw": Condition(func=cond.task_type, args={"task_type": [1, 2]})},
        post_condition={
            "speed_low" : Condition(
                func=cond.GraspAssetSpeedLow,
                args={
                    "grasp_kp_asset_id_cfg": Kp(term="manipulation_key_points", kp_attr="key_points_asset_id", kp_names="asset_grasp"),
                    "asset_cfg": SceneEntityCfg("assets"),
                    "history_length": 2,
                    "entry_alignment_cfg" : Align(term="auxiliary_task_alignment_data"),
                    "speed_limit": 0.0075
                }
            )},
        ee_exec=Exec(
            func=exec.insertion_execution,
            args={
                "entry_alignment_cfg" : Align(term="auxiliary_task_alignment_data"),
                "grasp_alignment_cfg" : Align(term="manipulation_alignment_data"),
                "entry_kp_cfg" : Kp(term="auxiliary_task_key_points", kp_names="asset_align_against"),
                "align_kp_cfg" : Kp(term="task_key_points", kp_names="asset_align"),
                "align_src_cfg" : Kp(term="task_key_points", kp_attr="key_points_src", kp_names="asset_align"),
                "align_offset_cfg" : Kp(term="task_key_points", kp_attr="key_points_offset", kp_names="asset_align"),
                "asset_align_kp_cfg": Kp(term="task_key_points", kp_names="asset_align"),
                "object_holding_kp_cfg" : Kp(term="manipulation_key_points", kp_names="robot_object_held"),
                "object_holding_offset_cfg" : Kp(term="manipulation_key_points", kp_attr="key_points_offset", kp_names="robot_object_held"),
                "object_holding_src_cfg" : Kp(term="manipulation_key_points", kp_attr="key_points_src", kp_names="robot_object_held"),
            }),
        gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
        limits=(0.0010, 0.05),
        noise=0.00002
    ),

    state.SCREW.__str__(): State(
        prev_states=[state.INSERT, state.GRASP],
        pre_condition={
            "entry_aligned" : Condition(
                func=cond.held_asset_insertion_position_aligned_with_fixed_asset_entry,
                args={
                    "auxiliary_alignment_cfg": Align(term="auxiliary_task_alignment_data"),
                    "pos_threshold": (0.005, 0.005, 1.0),   # TODO: lower threshold on z
                }
            ),
            "is_screw_task": Condition(func=cond.task_type, args={"task_type": [2]}),
            },
        post_condition={
            "wrist_limit_reached": Condition(
                func=cond.wrist_counter_clockwise_limit_reached,
                args={"robot_cfg": SceneEntityCfg("robot", joint_names="panda_joint7")}
            )},
        ee_exec=Exec(
            func=exec.screw_execution,
            args={
                "nist_board_cfg": SceneEntityCfg("assets", object_collection_names="nist_board"),
                "task_alignment_cfg" : Align(term="auxiliary_task_alignment_data"),
                "grasp_alignment_cfg" : Align(term="manipulation_alignment_data"),
                "align_src_cfg" : Kp(term="task_key_points", kp_attr="key_points_src", kp_names="asset_align"),
                "align_kp_cfg" : Kp(term="task_key_points", kp_names="asset_align"),
                "align_offset_cfg" : Kp(term="task_key_points", kp_attr="key_points_offset", kp_names="asset_align"),
                "align_against_kp_cfg" : Kp(term="auxiliary_task_key_points", kp_names="asset_align_against"),
                "object_holding_src_cfg" : Kp(term="manipulation_key_points", kp_attr="key_points_src", kp_names="robot_object_held"),
                "object_holding_kp_cfg" : Kp(term="manipulation_key_points", kp_names="robot_object_held"),
                "object_holding_offset_cfg" : Kp(term="manipulation_key_points", kp_attr="key_points_offset", kp_names="robot_object_held"),
            }          
        ),
        gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
        limits=(0.0004, 0.10),
        noise=0.0
    ),

    state.UNWIND.__str__(): State(
        prev_states=[state.RELEASE],
        pre_condition={
            "task_not_success" : Condition(
                func=cond.task_not_success,
                args={"task_alignment_cfg": Align(term="alignment_data")}
            ),
            "gripper_open": Condition(
                func=cond.gripper_open,
                args={"robot_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint1"), "threshold": 0.035}
            ),
            # "grasp_aligned" : Condition(
            #     func=cond.gripper_aligned_with_held_asset,
            #     args={
            #         "manipulation_alignment_cfg": Align(term="manipulation_alignment_data"),
            #         "only_pos": True,
            #         "pos_threshold": (0.03, 0.03, 0.03),
            #     }
            # )
            # Condition(func=cond.gripper_aligned_with_held_asset, args={"interpolate": True}),
        },
        post_condition={
            "gripper_open": Condition(
                func=cond.gripper_open,
                args={"robot_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint1"), "threshold": 0.035}
            ),
            "wrist_limit_reached": Condition(
                func=cond.wrist_clockwise_limit_reached,
                args={"robot_cfg": SceneEntityCfg("robot", joint_names="panda_joint7")}
            )
        },
        ee_exec=Exec(
            func=exec.unwind_execution,
            args={
                "nist_board_cfg": SceneEntityCfg("assets", object_collection_names="nist_board"),
                "grasp_alignment_cfg": Align(term="manipulation_alignment_data"),
                "grasp_kp_cfg": Kp(term="manipulation_key_points", kp_names="asset_grasp"),
                "object_holding_src_cfg" : Kp(term="manipulation_key_points", kp_attr="key_points_src", kp_names="robot_object_held"),
                "object_holding_kp_cfg" : Kp(term="manipulation_key_points", kp_names="robot_object_held"),
            }           
        ),
        gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
        limits=(0.0015, 0.15),
        noise=0.0
    ),

    # state.PRE_GRASP.__str__(): State(
    #     prev_states=[state.UNWIND],
    #     pre_condition=[Condition(func=cond.wrist_clockwise_limit_reached, args={"robot_cfg": SceneEntityCfg("robot", joint_names="panda_joint7")})],
    #     ee_exec=Exec(func=exec.align_gripper_to_held_asset_grasp_point, args={"interpolate": True}),
    #     gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
    #     limits=(0.185, 0.50),
    #     noise=0.0025
    # ),

    # state.GRASP.__str__(): State(
    #     prev_states=[state.PRE_GRASP],
    #     pre_condition=[Condition(func=cond.gripper_aligned_with_held_asset, args={"interpolate": True})],
    #     ee_exec=Exec(func=exec.align_gripper_to_held_asset_grasp_point, args={"interpolate": True}),
    #     gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
    #     limits=(0.185, 0.50),
    #     noise=0.0025
    # ),

    state.RELEASE.__str__(): State(
        prev_states=[state.INSERT, state.SCREW],
        pre_condition={
            "do_not_skip_screw": Condition(
                func=cond.not_during_screw_task, 
                args={"robot_cfg": SceneEntityCfg("robot", joint_names="panda_joint7"),
                    "task_alignment_cfg": Align(term="alignment_data")}
            )},
        post_condition={
            "gripper_open": Condition(
                func=cond.gripper_open,
                args={"robot_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint1"), "threshold": 0.035}
            )},
        ee_exec=Exec(func=exec.ee_stay_still, args={"ee_src_cfg": Kp(term="manipulation_key_points", kp_attr="key_points_src", kp_names="robot_object_held")}),
        gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
        limits=(0.001, 0.05),
        noise=0.00025
    ),

    state.DONE.__str__(): State(
        prev_states=[state.RELEASE, state.ALIGN_ASSET, state.UNWIND],
        pre_condition={
            "task_success" : Condition(
                func=cond.task_success,
                args={"task_alignment_cfg": Align(term="alignment_data")}
            )},
        ee_exec=Exec(func=exec.ee_stay_still, args={"ee_src_cfg": Kp(term="manipulation_key_points", kp_attr="key_points_src", kp_names="robot_object_held")}),
        gripper_exec=Exec(func=exec.gripper_stay_still, args={"action_idx": -1}),
        limits=(0.02, 0.05),
        noise=0.0
    ),

}
