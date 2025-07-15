from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from .action_adapter import RelJointPosAdapter, ActionAdapter


@configclass
class ActionAdapterCfg:

    class_type: type[ActionAdapter] = ActionAdapter


@configclass
class RelJointPosAdapterCfg(ActionAdapterCfg):

    class_type: type[RelJointPosAdapter] = RelJointPosAdapter

    ik_solver: DifferentialInverseKinematicsActionCfg = MISSING  # type: ignore
