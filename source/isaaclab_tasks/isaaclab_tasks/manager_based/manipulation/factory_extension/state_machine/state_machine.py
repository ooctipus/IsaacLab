import torch
import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import DataManager
from . import utils as sm_math_utils

if TYPE_CHECKING:
    from .state_machine_cfg import StateMachineCfg

# viz for debug, remove when done debugging
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
frame_marker_cfg.markers["frame"].scale = (0.075, 0.075, 0.075)
pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))


class StateMachine:
    def __init__(self, cfg, env: ManagerBasedRLEnv, device: str | None = None):
        self.cfg: StateMachineCfg = cfg
        self.env = env
        self.ENVS_ARANGE = torch.arange(env.num_envs, device=env.device)
        self.device = device if device else env.device
        self.state_machine = {int(key) : val for key, val in self.cfg.states_cfg.items()}
        self.action_adapter = self.cfg.action_adapter_cfg.class_type(self.cfg.action_adapter_cfg, env)  # type: ignore
        self.cfg.robot_cfg.resolve(env.scene)
        self.robot = env.scene[self.cfg.robot_cfg.name]

        # initialize the state machine
        self.sm_state = torch.full((env.num_envs,), 0, dtype=torch.int32, device=self.device)
        # desire state
        self.des_gripper_state = torch.zeros((env.num_envs,), device=self.device)
        self.des_ee_pose = torch.zeros((env.num_envs, 7), device=self.device)

        self.data_manager: DataManager = DataManager(self.cfg.data, env)
        print("[INFO] State Machine Data Manager: ", self.data_manager)
        self.data_manager.reset()

        if not hasattr(env, "extensions"):
            setattr(env, "extensions", {})
        env.extensions['fsm'] = self
        env.data_manager._terms.update(self.data_manager._terms)

        self.class_conditions = []
        self.class_executions = []
        self._resolve()

    def _resolve(self):
        for value in self.cfg.states_cfg.values():
            for arg in value.gripper_exec.args.values():
                if isinstance(arg, SceneEntityCfg):
                    arg.resolve(self.env.scene)
            for arg in value.ee_exec.args.values():
                if isinstance(arg, SceneEntityCfg):
                    arg.resolve(self.env.scene)
            for condition in value.post_condition.values():
                for arg in condition.args.values():
                    if isinstance(arg, SceneEntityCfg):
                        arg.resolve(self.env.scene)
                if inspect.isclass(condition.func):
                    condition.func = condition.func(condition, self.env)
                    self.class_conditions.append(condition.func)
            for condition in value.pre_condition.values():
                for arg in condition.args.values():
                    if isinstance(arg, SceneEntityCfg):
                        arg.resolve(self.env.scene)
                if inspect.isclass(condition.func):
                    condition.func = condition.func(condition, self.env)
                    self.class_conditions.append(condition.func)

    def reset(self):
        self.reset_idx()

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        """Reset the state machine for the given environment instances."""
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs)  # type: ignore
        # reset the state machine
        self.sm_state[env_ids] = 0
        for condition in self.class_conditions:
            condition.reset(env_ids)
        for execution in self.class_executions:
            execution.reset(env_ids)

    def act(self, env_ids, ee_des, gripper_des, max_pos, max_rot, noise):
        self.des_gripper_state[env_ids] = gripper_des
        current_ee_pose = self.robot.data.body_link_state_w[env_ids, self.cfg.robot_cfg.body_ids, :7].view(-1, 7)
        self.des_ee_pose[env_ids] = sm_math_utils.se3_step(current_ee_pose, ee_des, max_pos=max_pos, max_rot=max_rot)
        self.des_ee_pose[env_ids, :3] += torch.randn_like(ee_des[:, :3]) * noise

    def compute(self):
        self.data_manager.compute(dt=self.env.step_dt)
        # implement post condition check:
        current_state_post_conditions_satisfied = torch.ones_like(self.sm_state, device=self.device, dtype=torch.bool)
        for state_idx, automaton in self.state_machine.items():
            envs_in_current_state_mask = (self.sm_state == state_idx)
            if torch.any(envs_in_current_state_mask):
                env_ids = self.ENVS_ARANGE[envs_in_current_state_mask]
                all_posts_met_for_these_envs = torch.ones_like(env_ids, device=self.device, dtype=torch.bool)
                for name, post_condition in automaton.post_condition.items():
                    result = post_condition.func(self.env, env_ids, **post_condition.args)
                    if automaton.debug_log:
                        print(f"{name} : {result}")
                    all_posts_met_for_these_envs &= result
                current_state_post_conditions_satisfied[env_ids] = all_posts_met_for_these_envs

        next_sm_state = self.sm_state.clone()
        for target_state_idx, automaton in self.state_machine.items():
            transition_candidates_mask = torch.zeros_like(self.sm_state, device=self.device, dtype=torch.bool)
            for prev_state_idx in automaton.prev_states:
                eligible_from_prev_state = (self.sm_state == prev_state_idx) & current_state_post_conditions_satisfied
                transition_candidates_mask |= eligible_from_prev_state

            if torch.any(transition_candidates_mask):
                for name, pre_condition in automaton.pre_condition.items():
                    if not torch.any(transition_candidates_mask):
                        break
                    env_ids = self.ENVS_ARANGE[transition_candidates_mask]
                    result = pre_condition.func(self.env, env_ids, **pre_condition.args)
                    if automaton.debug_log:
                        print(f"{name} : {result}")
                    transition_candidates_mask[env_ids] = transition_candidates_mask[env_ids] & result
                next_sm_state[transition_candidates_mask] = target_state_idx
        self.sm_state = next_sm_state

        for state, automaton in self.state_machine.items():
            mask = self.sm_state == state
            if torch.any(mask):
                env_ids = self.ENVS_ARANGE[mask]
                des_ee = automaton.ee_exec.func(self.env, env_ids, **automaton.ee_exec.args)
                pose_marker.visualize(des_ee[:,:3], des_ee[:,3:])
                
                des_gripper = automaton.gripper_exec.func(self.env, env_ids, **automaton.gripper_exec.args)
                self.act(env_ids, des_ee, des_gripper, max_pos=automaton.limits[0], max_rot=automaton.limits[1], noise=automaton.noise)

        des_pose_b = sm_math_utils.poses_subtract(self.robot.data.root_state_w[:, :7], self.des_ee_pose)
        action = torch.cat([des_pose_b, self.des_gripper_state.unsqueeze(-1)], dim=-1)
        adapter_action = self.action_adapter.compute(action[:, :7])
        # tasks = self.env.command_manager.get_term("task_command").tasks
        # tid = self.env.command_manager.get_command("task_command")
        # for i in range(len(tid)):
        #     print(f"task {i}: {tasks[tid[i].item()]} - state: {self.sm_state[i]}")
        return torch.cat([adapter_action, action[:, 7:]], dim=1)
