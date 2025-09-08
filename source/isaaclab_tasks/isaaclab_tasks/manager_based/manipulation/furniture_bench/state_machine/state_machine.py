import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from .. import assembly_data
from . import utils as sm_math_utils

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject, Articulation
    from .state_machine_cfg import StateMachineCfg

# viz for debug, remove when done debugging
# from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
# frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
# frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
# pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/debug_transform"))


class StateMachine:
    def __init__(self, cfg, env, device: str | None = None):
        self.cfg: StateMachineCfg = cfg
        self.env = env
        self.device = device if device else env.device
        self.state_machine = self.cfg.states_cfg
        self.action_adapter = self.cfg.action_adapter_cfg.class_type(self.cfg.action_adapter_cfg, env)  # type: ignore

        # initialize the state machine
        self.sm_state = torch.full((env.num_envs,), 0, dtype=torch.int32, device=self.device)

        # desire state
        self.des_gripper_state = torch.zeros((env.num_envs,), device=self.device)
        self.des_ee_pose = torch.zeros((env.num_envs, 7), device=self.device)

        # assembled held_asset pitch and roll
        self.assembled_held_asset_pitch_roll = torch.zeros((env.num_envs, 2), device=self.device)

        # approach above hole offset
        self.approach_above_hole_offset_xyz = torch.zeros((env.num_envs, 3), device=self.device)
        self.approach_above_hole_offset_xyz[:, 2] = 0.01
        self.way_above_hole_offset_xyz = torch.zeros((env.num_envs, 3), device=self.device)
        self.way_above_hole_offset_xyz[:, 2] = 0.05

        # screw offset
        self.screw_turn_delta = -math.pi / 8
        self.screw_pos_delta = -0.005

        # unwind offset
        self.unwind_delta = math.pi / 4

        # episodic data
        self.init_hole_pose = torch.zeros((env.num_envs, 7), device=self.device)

        self.desired_held_asset_grasp_quat_w = torch.zeros((env.num_envs, 4), device=self.device)

        self._resolve()

    def _resolve(self):
        for value in self.cfg.states_cfg.values():
            for arg in value.gripper_exec.args.values():
                if isinstance(arg, SceneEntityCfg):
                    arg.resolve(self.env.scene)
            for condition in value.pre_condition:
                for arg in condition.args.values():
                    if isinstance(arg, SceneEntityCfg):
                        arg.resolve(self.env.scene)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        """Reset the state machine for the given environment instances."""
        if env_ids is None:
            env_ids = slice(None)  # type: ignore
        # reset the state machine
        self.sm_state[env_ids] = 0

    def act(self, env_ids, ee_des, gripper_des, max_pos, max_rot, noise):
        self.des_gripper_state[env_ids] = gripper_des
        self.des_ee_pose[env_ids] = sm_math_utils.se3_step(self.ee_pose_w[env_ids], ee_des, max_pos=max_pos, max_rot=max_rot)
        self.des_ee_pose[env_ids, :3] += torch.randn_like(ee_des[:, :3]) * noise

    def compute(self):
        
        #current_state = self.sm_state[0].item()

        self._update_state(
            assembly_data.KEYPOINTS_TABLELEG.graspable,
            assembly_data.KEYPOINTS_TABLELEG.center_axis_bottom,
            assembly_data.KEYPOINTS_TABLETOPHOLE.hole0_tip_offset,
            assembly_data.KEYPOINTS_TABLETOPHOLE.hole0_leg_assembled_offset,
            assembly_data.KEYPOINTS_ROBOTIQGRIPPER.offset,
        )

        for state, automaton in self.state_machine.items():
            mask = torch.zeros_like(self.sm_state, device=self.device, dtype=torch.bool)
            for pre_state in automaton.prev_states:
                mask = mask | (self.sm_state == pre_state)
            if torch.any(mask):
                for pre_condition in automaton.pre_condition:
                    mask = mask & pre_condition.func(self.env, self, **pre_condition.args)
                self.sm_state[mask] = state

        for state, automaton in self.state_machine.items():
            mask = self.sm_state == state
            if torch.any(mask):
                des_ee = automaton.ee_exec.func(self.env, self, mask, **automaton.ee_exec.args)
                des_gripper = automaton.gripper_exec.func(self.env, self, mask, **automaton.gripper_exec.args)
                self.act(mask, des_ee, des_gripper, max_pos=automaton.limits[0], max_rot=automaton.limits[1], noise=automaton.noise)

        des_pose_b = sm_math_utils.poses_subtract(self.robot.data.root_state_w[:, :7], self.des_ee_pose)
        action = torch.cat([des_pose_b, self.des_gripper_state.unsqueeze(-1)], dim=-1)
        adapter_action = self.action_adapter.compute(action[:, :7])

        # if current_state != self.sm_state[0].item():
        #     print("New state: ", self.sm_state[0].item())

        return torch.cat([adapter_action, action[:, 7:]], dim=1)

    def _update_state(
        self,
        held_asset_grasp_point_offset: assembly_data.Offset,
        held_asset_insertion_offset: assembly_data.Offset,
        fixed_asset_entry_offset: assembly_data.Offset,
        fixed_asset_held_assembled_offset: assembly_data.Offset,
        ee_object_held_offset: assembly_data.Offset,    
    ) -> None:
        
        num_envs = self.env.num_envs

        held_asset: RigidObject = self.env.scene["leg"]
        fixed_asset: RigidObject = self.env.scene["table_top"]
        supporting_asset: RigidObject = self.env.scene["table"]
        self.robot: Articulation = self.env.scene["robot"]

        self.held_asset_pose_w = held_asset.data.root_state_w[:, :7]
        self.fixed_asset_pose_w = fixed_asset.data.root_state_w[:, :7]
        self.supporting_asset_pose_w = supporting_asset.data.root_state_w[:, :7]

        # end-effector
        ee_body_idx = self.robot.data.body_names.index("robotiq_base_link")
        self.ee_pose_w = self.robot.data.body_link_state_w[..., ee_body_idx, :7]

        # gripper_width
        gripper_joint_id = self.robot.data.joint_names.index("finger_joint")
        self.gripper_joint_pos = self.robot.data.joint_pos[:, gripper_joint_id]

        # robot grasp point
        self.ee_object_held_offset_pose = torch.tensor(ee_object_held_offset.pose).to(self.device).repeat(num_envs, 1)
        self.ee_object_held_pose_w = sm_math_utils.poses_combine(self.ee_pose_w, self.ee_object_held_offset_pose)

        # held_asset key points
        self.held_asset_grasp_offset_pose = torch.tensor(held_asset_grasp_point_offset.pose).to(self.device).repeat(num_envs, 1)
        self.held_asset_grasp_pose_w = sm_math_utils.poses_combine(self.held_asset_pose_w, self.held_asset_grasp_offset_pose)

        self.held_asset_insertion_offset_pose = torch.tensor(held_asset_insertion_offset.pose).to(self.device).repeat(num_envs, 1)
        self.held_asset_insertion_pose_w = sm_math_utils.poses_combine(self.held_asset_pose_w, self.held_asset_insertion_offset_pose)

        # fixed_asset key points
        self.fixed_asset_entry_offset_pose = torch.tensor(fixed_asset_entry_offset.pose).to(self.device).repeat(num_envs, 1)
        self.fixed_asset_entry_pose_w = sm_math_utils.poses_combine(self.fixed_asset_pose_w, self.fixed_asset_entry_offset_pose)

        self.fixed_asset_held_assembled_offset_pose = torch.tensor(fixed_asset_held_assembled_offset.pose).to(self.device).repeat(num_envs, 1)
        self.fixed_asset_held_assembled_pose_w = sm_math_utils.poses_combine(self.fixed_asset_pose_w, self.fixed_asset_held_assembled_offset_pose)

        self.support_asset_z_axis = math_utils.matrix_from_quat(self.supporting_asset_pose_w[:, 3:])[..., 2]
        episode_begin_envs = self.env.episode_length_buf == 0

        self.desired_held_asset_grasp_quat_w[:] = sm_math_utils.interpolate_grasp_quat(
            held_asset_grasp_point_quat_w=self.held_asset_grasp_pose_w[:, 3:],
            grasped_object_quat_in_ee_frame=self.ee_object_held_pose_w[:, 3:],
            secondary_z_axis=self.support_asset_z_axis[:],
        )

        # Apply a tilt to the desired grasp quaternion to lift from above
        # Example: Rotate around the X-axis by 15 degrees (convert to radians)
        # Apply a tilt to the desired grasp quaternion to lift from above
        # Example: Rotate around the X-axis by 15 degrees (convert to radians)
        # tilt_angle = torch.tensor(15 * math.pi / 180, device=self.device)  # Create the tilt angle on the correct device
        # tilt_quaternion = math_utils.quat_from_euler_xyz(
        #     torch.tensor(0.0, device=self.device), -tilt_angle, torch.tensor(0.0, device=self.device)
        # )

        # # Expand tilt_quaternion to match the batch size
        # tilt_quaternion = tilt_quaternion.unsqueeze(0).repeat(self.desired_held_asset_grasp_quat_w.shape[0], 1)

        # # Apply the tilt to the desired grasp quaternion
        # self.desired_held_asset_grasp_quat_w[:] = math_utils.quat_mul(
        #     self.desired_held_asset_grasp_quat_w, tilt_quaternion
        # )

        self.held_asset_inserted_in_fixed_asset_pos = (0.2 * self.fixed_asset_entry_pose_w[:, :3] + 0.8 * self.fixed_asset_held_assembled_pose_w[:, :3])

        self.init_hole_pose[episode_begin_envs] = self.fixed_asset_held_assembled_pose_w[episode_begin_envs]
