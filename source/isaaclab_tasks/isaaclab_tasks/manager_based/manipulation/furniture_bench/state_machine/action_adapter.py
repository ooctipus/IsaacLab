import torch
from typing import TYPE_CHECKING
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

if TYPE_CHECKING:
    from .action_adapter_cfg import RelJointPosAdapterCfg, ActionAdapterCfg


class ActionAdapter:
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        self.cfg: ActionAdapterCfg = cfg

    @property
    def joint_ids(self):
        return slice(None)

    def compute(self, action: torch.Tensor) -> torch.Tensor:
        return action


class RelJointPosAdapter(ActionAdapter):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        self.cfg: RelJointPosAdapterCfg = cfg
        self.ik_solver_cfg = self.cfg.ik_solver
        self.ik_solver: DifferentialInverseKinematicsAction = \
            self.cfg.ik_solver.class_type(self.cfg.ik_solver, env)  # type: ignore

    @property
    def joint_ids(self):
        return self.ik_solver._joint_ids

    def compute(self, action: torch.Tensor) -> torch.Tensor:
        robot = self.ik_solver._asset
        original_joint_pos_des = robot.data.joint_pos_target[:, self.joint_ids]
        self.ik_solver.process_actions(action)
        self.ik_solver.apply_actions()
        joint_pos_des = robot.data.joint_pos_target[:, self.joint_ids]
        robot.set_joint_position_target(original_joint_pos_des, self.joint_ids)
        cur_joint_pos = robot.data.joint_pos[:, self.joint_ids]

        return joint_pos_des - cur_joint_pos
