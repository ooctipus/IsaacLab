from isaaclab.utils import configclass

from ... import mdp
from ...factory_assets_cfg import FRANKA_PANDA_CFG
from ...factory_env_cfg import FactoryEnvCfg, FactoryActionsCfg


@configclass
class ActionCfg(FactoryActionsCfg):
    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.02,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.0},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class FrankaFactoryEnvMixIn:
    actions: ActionCfg = ActionCfg()

    def __post_init__(self: FactoryEnvCfg):
        super().__post_init__()  # type:ignore
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type:ignore
        self.scene.robot.actuators["panda_arm1"].stiffness = 80.0
        self.scene.robot.actuators["panda_arm1"].damping = 4.0
        self.scene.robot.actuators["panda_arm2"].stiffness = 80.0
        self.scene.robot.actuators["panda_arm2"].damping = 4.0
        cmd_e = self.commands.task_command.reset_terms_when_resample
        # cmd_e["reset_end_effector_around_fixed_asset"].params["robot_ik_cfg"].joint_names = ["panda_joint.*"]
        # cmd_e["reset_end_effector_around_fixed_asset"].params["robot_ik_cfg"].body_names = "panda_fingertip_centered"
        # cmd_e["grasp_held_asset"].params["robot_cfg"].body_names = "panda_fingertip_centered"
        # cmd_e["grasp_held_asset"].params["robot_cfg"].joint_names = "panda_finger_joint[1-2]"
        
        cmd_e['reset_held_asset_in_gripper'].params['params']['reset_end_effector_around_fixed_asset']['robot_ik_cfg'].joint_names = ["panda_joint.*"]
        cmd_e['reset_held_asset_in_gripper'].params['params']['reset_end_effector_around_fixed_asset']['robot_ik_cfg'].body_names = "panda_fingertip_centered"
        cmd_e['reset_held_asset_in_gripper'].params['params']["grasp_held_asset"]["robot_cfg"].body_names = "panda_fingertip_centered"
        cmd_e['reset_held_asset_in_gripper'].params['params']["grasp_held_asset"]["robot_cfg"].joint_names = "panda_finger_joint[1-2]"


@configclass
class FrankaFactoryEnvCfg(FrankaFactoryEnvMixIn, FactoryEnvCfg):
    pass
