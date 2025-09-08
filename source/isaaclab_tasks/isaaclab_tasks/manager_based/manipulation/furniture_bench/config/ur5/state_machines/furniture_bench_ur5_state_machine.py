import warp as wp
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers import DifferentialIKControllerCfg

from ....state_machine import StateCfg as State
from ....state_machine import ConditionCfg as Condition
from ....state_machine import ExecCfg as Exec
from ....state_machine import StateMachineCfg
from ....state_machine import RelJointPosAdapterCfg

from . import conditions as cond
from . import executions as exec


# @configclass
# class Ur5FurnitureBenchState:
#     """States for the state machine."""

#     # REST = wp.constant(0)
#     # APPROACH_HELD_ASSET = wp.constant(1)
#     # PICK_HELD_ASSET = wp.constant(2)
#     # LIFT_UP = wp.constant(3)
#     # APPROACH_ABOVE_FIXED_ASSET = wp.constant(4)
#     # INSERT = wp.constant(5)
#     # SCREW = wp.constant(6)
#     # RELEASE = wp.constant(7)
#     # UNWIND = wp.constant(8)
#     # PRE_GRASP = wp.constant(9)
#     # GRASP = wp.constant(10)
#     # DONE = wp.constant(11)



# @configclass
# class GripperState:
#     OPEN = wp.constant(1.0)
#     CLOSE = wp.constant(-1.0)


# state = Ur5FurnitureBenchState()
# gripper_state = GripperState()


# @configclass
# class SmScrewTaskCfg(StateMachineCfg):

#     states_cfg = {

#         state.APPROACH_HELD_ASSET: State(
#             prev_states=[state.REST],
#             pre_condition=[Condition(func=cond.always)],
#             ee_exec=Exec(func=exec.align_gripper_to_held_asset_grasp_point, args={"interpolate": True}),
#             gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
#             limits=(0.185, 0.50),
#             noise=0.0025,
#         ),

#         state.PICK_HELD_ASSET: State(
#             prev_states=[state.APPROACH_HELD_ASSET],
#             pre_condition=[Condition(func=cond.gripper_aligned_with_held_asset, args={"interpolate": True})],
#             ee_exec=Exec(func=exec.align_gripper_to_held_asset_grasp_point, args={"interpolate": True}),
#             gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
#             limits=(0.185, 0.50),
#             noise=0.0025,
#         ),

#         state.LIFT_UP: State(
#             prev_states=[state.PICK_HELD_ASSET],
#             pre_condition=[
#                 Condition(func=cond.gripper_aligned_with_held_asset, args={"interpolate": True}),
#                 Condition(func=cond.gripper_grasp_object, args={"robot_cfg": SceneEntityCfg("robot", joint_names="finger_joint")}),
#             ],
#             ee_exec=Exec(func=exec.lift_up_execution),
#             gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
#             limits=(0.185, 0.50),
#             noise=0.0025,
#         ),

#         state.APPROACH_ABOVE_FIXED_ASSET: State(
#             prev_states=[state.LIFT_UP],
#             pre_condition=[
#                 Condition(func=cond.gripper_aligned_with_held_asset, args={"interpolate": True}),
#                 Condition(func=cond.gripper_grasp_object, args={"robot_cfg": SceneEntityCfg("robot", joint_names="finger_joint")}),
#                 Condition(func=cond.held_asset_lifted),
#             ],
#             ee_exec=Exec(func=exec.aligning_held_asset_insertion_to_fixed_asset_entry),
#             gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
#             limits=(0.185, 0.50),
#             noise=0.0025,
#         ),

#         state.INSERT: State(
#             prev_states=[state.APPROACH_ABOVE_FIXED_ASSET],
#             pre_condition=[
#                 Condition(func=cond.held_asset_insertion_aligned_with_fixed_asset_entry)
#             ],
#             ee_exec=Exec(func=exec.insertion_execution),
#             gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
#             limits=(0.185, 0.50),
#             noise=0.0025,
#         ),

#         state.SCREW: State(
#             prev_states=[state.INSERT, state.GRASP],
#             pre_condition=[
#                 Condition(func=cond.held_asset_insertion_aligned_with_fixed_asset_entry, args={"check_fully_inserted": True})
#             ],
#             ee_exec=Exec(func=exec.screw_execution),
#             gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
#             limits=(0.05, 0.50),
#             noise=0.001,
#         ),

#         state.RELEASE: State(
#             prev_states=[state.SCREW],
#             pre_condition=[Condition(func=cond.wrist_counter_clockwise_limit_reached, args={"robot_cfg": SceneEntityCfg("robot", joint_names="wrist_3_joint")})],
#             ee_exec=Exec(func=exec.align_gripper_to_held_asset_grasp_point, args={"interpolate": True}),
#             gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
#             limits=(0.005, 0.25),
#             noise=0.0025,
#         ),

#         state.UNWIND: State(
#             prev_states=[state.RELEASE],
#             pre_condition=[
#                 Condition(func=cond.gripper_open, args={"robot_cfg": SceneEntityCfg("robot", joint_names="finger_joint")}),
#                 Condition(func=cond.gripper_aligned_with_held_asset, args={"interpolate": True}),
#             ],
#             ee_exec=Exec(func=exec.unwind_execution),
#             gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
#             limits=(0.185, 2.50),
#             noise=0.0,
#         ),

#         state.PRE_GRASP: State(
#             prev_states=[state.UNWIND],
#             pre_condition=[Condition(func=cond.wrist_clockwise_limit_reached, args={"robot_cfg": SceneEntityCfg("robot", joint_names="wrist_3_joint")})],
#             ee_exec=Exec(func=exec.align_gripper_to_held_asset_grasp_point, args={"interpolate": True}),
#             gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
#             limits=(0.185, 0.50),
#             noise=0.0025,
#         ),

#         state.GRASP: State(
#             prev_states=[state.PRE_GRASP],
#             pre_condition=[Condition(func=cond.gripper_aligned_with_held_asset, args={"interpolate": True})],
#             ee_exec=Exec(func=exec.align_gripper_to_held_asset_grasp_point, args={"interpolate": True}),
#             gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
#             limits=(0.185, 0.50),
#             noise=0.0025,
#         ),

#         state.DONE: State(
#             prev_states=[state.SCREW],
#             pre_condition=[Condition(func=cond.held_asset_fully_assembled_on_fixed_asset)],
#             ee_exec=Exec(func=exec.align_gripper_to_held_asset_grasp_point, args={"interpolate": True}),
#             gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
#             limits=(0.05, 0.50),
#             noise=0.0,
#         ),

#     }


# @configclass
# class SmScrewTaskRelJointPositionAdapterCfg(SmScrewTaskCfg):
#     action_adapter_cfg = RelJointPosAdapterCfg(
#         ik_solver=DifferentialInverseKinematicsActionCfg(
#             asset_name="robot",
#             joint_names=["shoulder.*", "elbow.*", "wrist.*"],
#             body_name="robotiq_base_link",
#             controller=DifferentialIKControllerCfg(
#                 command_type="pose", use_relative_mode=False, ik_method="dls"
#             ),
#             scale=1,
#         )
#    )



@configclass
class Ur5FurnitureBenchState:
    """States for the state machine."""

    REST = wp.constant(0)
    APPROACH_WAY_ABOVE_FIXED_ASSET = wp.constant(1)
    APPROACH_ABOVE_FIXED_ASSET = wp.constant(2)
    INSERT = wp.constant(3)
    RELEASE_AFTER_INSERT = wp.constant(4)
    PRE_GRASP_AFTER_INSERT = wp.constant(5)
    GRASP_AFTER_INSERT = wp.constant(6)
    INSERT_ALIGNED = wp.constant(7)
    SCREW = wp.constant(8)
    RELEASE = wp.constant(9)
    UNWIND = wp.constant(10)
    PRE_GRASP = wp.constant(11)
    GRASP = wp.constant(12)
    DONE = wp.constant(13)





@configclass
class GripperState:
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


state = Ur5FurnitureBenchState()
gripper_state = GripperState()


@configclass
class SmScrewTaskCfg(StateMachineCfg):

    states_cfg = {

        state.APPROACH_WAY_ABOVE_FIXED_ASSET: State(
            prev_states=[state.REST],
            pre_condition=[Condition(func=cond.always)
            ],
            ee_exec=Exec(func=exec.aligning_held_asset_insertion_way_above_fixed_asset_entry),
            gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
            limits=(0.185, 0.50),
            noise=0.0025,
        ),

        state.APPROACH_ABOVE_FIXED_ASSET: State(
            prev_states=[state.APPROACH_WAY_ABOVE_FIXED_ASSET],
            pre_condition=[
                Condition(func=cond.held_asset_insertion_aligned_with_way_above_fixed_asset_entry)
            ],
            ee_exec=Exec(func=exec.aligning_held_asset_insertion_to_fixed_asset_entry),
            gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
            limits=(0.185, 0.50),
            noise=0.0025,
        ),

        state.INSERT: State(
            prev_states=[state.APPROACH_ABOVE_FIXED_ASSET],
            pre_condition=[
                Condition(func=cond.held_asset_insertion_aligned_with_fixed_asset_entry)
            ],
            ee_exec=Exec(func=exec.insertion_execution),
            gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
            limits=(0.185, 0.50),
            noise=0.0025,
        ),

        state.RELEASE_AFTER_INSERT: State(
            prev_states=[state.INSERT],
            pre_condition=[Condition(func=cond.held_asset_insertion_aligned_with_fixed_asset_entry, args={"check_fully_inserted": True})],
            ee_exec=Exec(func=exec.keep_ee_pose, args={}),
            gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
            limits=(0.005, 0.25),
            noise=0.0025,
        ),

        state.PRE_GRASP_AFTER_INSERT: State(
            prev_states=[state.RELEASE_AFTER_INSERT],
            pre_condition=[
                Condition(func=cond.gripper_open, args={"robot_cfg": SceneEntityCfg("robot", joint_names="finger_joint")}),
            ],
            ee_exec=Exec(func=exec.align_gripper_to_held_asset_grasp_point, args={"interpolate": True}),
            gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
            limits=(0.185, 0.50),
            noise=0.0025,
        ),

        state.GRASP_AFTER_INSERT: State(
            prev_states=[state.PRE_GRASP_AFTER_INSERT],
            pre_condition=[Condition(func=cond.gripper_aligned_with_held_asset, args={"interpolate": True})],
            ee_exec=Exec(func=exec.align_gripper_to_held_asset_grasp_point, args={"interpolate": True}),
            gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
            limits=(0.185, 0.50),
            noise=0.0025,
        ),

        state.INSERT_ALIGNED: State(
            prev_states=[state.GRASP_AFTER_INSERT],
            pre_condition=[
                Condition(func=cond.gripper_grasp_object, args={"robot_cfg": SceneEntityCfg("robot", joint_names="finger_joint")})
            ],
            ee_exec=Exec(func=exec.insertion_execution),
            gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
            limits=(0.185, 0.50),
            noise=0.0025,
        ),

        state.SCREW: State(
            prev_states=[state.INSERT_ALIGNED, state.GRASP],
            pre_condition=[
                Condition(func=cond.held_asset_insertion_aligned_with_fixed_asset_entry, args={"check_fully_inserted": True, "pos_threshold": 0.05})
            ],
            ee_exec=Exec(func=exec.screw_execution),
            gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
            limits=(0.05, 0.50),
            noise=0.001,
        ),

        state.RELEASE: State(
            prev_states=[state.SCREW],
            pre_condition=[Condition(func=cond.wrist_counter_clockwise_limit_reached, args={"robot_cfg": SceneEntityCfg("robot", joint_names="wrist_3_joint")})],
            ee_exec=Exec(func=exec.align_gripper_to_held_asset_grasp_point, args={"interpolate": True}),
            gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
            limits=(0.005, 0.25),
            noise=0.0025,
        ),

        state.UNWIND: State(
            prev_states=[state.RELEASE],
            pre_condition=[
                Condition(func=cond.gripper_open, args={"robot_cfg": SceneEntityCfg("robot", joint_names="finger_joint")}),
                Condition(func=cond.gripper_aligned_with_held_asset, args={"interpolate": True}),
            ],
            ee_exec=Exec(func=exec.unwind_execution),
            gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
            limits=(0.185, 2.50),
            noise=0.0,
        ),

        state.PRE_GRASP: State(
            prev_states=[state.UNWIND],
            pre_condition=[Condition(func=cond.wrist_clockwise_limit_reached, args={"robot_cfg": SceneEntityCfg("robot", joint_names="wrist_3_joint")})],
            ee_exec=Exec(func=exec.align_gripper_to_held_asset_grasp_point, args={"interpolate": True}),
            gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
            limits=(0.185, 0.50),
            noise=0.0025,
        ),

        state.GRASP: State(
            prev_states=[state.PRE_GRASP],
            pre_condition=[Condition(func=cond.gripper_aligned_with_held_asset, args={"interpolate": True})],
            ee_exec=Exec(func=exec.align_gripper_to_held_asset_grasp_point, args={"interpolate": True}),
            gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.CLOSE}),
            limits=(0.185, 0.50),
            noise=0.0025,
        ),

        state.DONE: State(
            prev_states=[state.SCREW],
            pre_condition=[Condition(func=cond.held_asset_fully_assembled_on_fixed_asset)],
            ee_exec=Exec(func=exec.align_gripper_to_held_asset_grasp_point, args={"interpolate": True}),
            gripper_exec=Exec(func=exec.gripper_action, args={"gripper_command": gripper_state.OPEN}),
            limits=(0.05, 0.50),
            noise=0.0,
        ),

    }


@configclass
class SmScrewTaskRelJointPositionAdapterCfg(SmScrewTaskCfg):
    action_adapter_cfg = RelJointPosAdapterCfg(
        ik_solver=DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder.*", "elbow.*", "wrist.*"],
            body_name="robotiq_base_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            scale=1,
        )
    )






