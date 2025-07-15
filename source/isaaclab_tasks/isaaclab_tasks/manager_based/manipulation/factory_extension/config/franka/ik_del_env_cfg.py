from isaaclab.utils import configclass

from ... import mdp
from ...factory_assets_cfg import FRANKA_PANDA_CFG
from ...factory_env_cfg import FactoryEnvCfg, FactoryActionsCfg


@configclass
class IkDelActionCfg(FactoryActionsCfg):
    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.02,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.00},
    )


@configclass
class IkAbsActionCfg(FactoryActionsCfg):
    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1.0,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.00},
    )


@configclass
class FrankaFactoryIkDelEnvMixIn:
    actions: IkDelActionCfg = IkDelActionCfg()

    def __post_init__(self: FactoryEnvCfg):
        super().__post_init__()  # type:ignore
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type:ignore
        self.scene.robot.actuators["panda_arm1"].stiffness = 100
        self.scene.robot.actuators["panda_arm2"].stiffness = 100
        self.scene.robot.actuators["panda_arm1"].damping = 5
        self.scene.robot.actuators["panda_arm2"].damping = 5

        cmd_e = self.commands.task_command.reset_terms_when_resample
        asset_in_gripper_params = cmd_e["reset_held_asset_in_gripper"].params["params"]
        asset_in_gripper_params["grasp_held_asset"]["robot_cfg"].body_names = "panda_fingertip_centered"
        asset_in_gripper_params["grasp_held_asset"]["robot_cfg"].joint_names = "panda_finger_joint[1-2]"
        asset_in_gripper_params["reset_end_effector_around_fixed_asset"]["robot_ik_cfg"].joint_names = ["panda_joint.*"]
        asset_in_gripper_params["reset_end_effector_around_fixed_asset"]["robot_ik_cfg"].body_names = "panda_fingertip_centered"


@configclass
class FrankaFactoryIkAbsEnvMixIn:
    actions: IkAbsActionCfg = IkAbsActionCfg()

    def __post_init__(self: FactoryEnvCfg):
        super().__post_init__()  # type:ignore
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type:ignore
        self.scene.robot.actuators["panda_arm1"].stiffness = 800
        self.scene.robot.actuators["panda_arm2"].stiffness = 700
        self.scene.robot.actuators["panda_arm1"].damping = 30
        self.scene.robot.actuators["panda_arm2"].damping = 30

        cmd_e = self.commands.task_command.reset_terms_when_resample
        asset_in_gripper_params = cmd_e["reset_held_asset_in_gripper"].params["params"]
        asset_in_gripper_params["grasp_held_asset"]["robot_cfg"].body_names = "panda_fingertip_centered"
        asset_in_gripper_params["grasp_held_asset"]["robot_cfg"].joint_names = "panda_finger_joint[1-2]"
        asset_in_gripper_params["reset_end_effector_around_fixed_asset"]["robot_ik_cfg"].joint_names = ["panda_joint.*"]
        asset_in_gripper_params["reset_end_effector_around_fixed_asset"]["robot_ik_cfg"].body_names = "panda_fingertip_centered"

@configclass
class FrankaFactoryIkDelEnvCfg(FrankaFactoryIkDelEnvMixIn, FactoryEnvCfg):
    pass


@configclass
class FrankaFactoryIkAbsEnvCfg(FrankaFactoryIkAbsEnvMixIn, FactoryEnvCfg):
    pass
