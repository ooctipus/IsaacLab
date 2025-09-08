from dataclasses import MISSING
from isaaclab.utils import configclass

from .state_machine import StateMachine
from .action_adapter_cfg import ActionAdapterCfg


@configclass
class ConditionCfg:
    func: callable = MISSING
    args: dict[str, any] = {}


@configclass
class ExecCfg:
    func: callable = MISSING
    args: dict[str, any] = {}


@configclass
class StateCfg:
    prev_states: list[int] = MISSING  # type: ignore
    pre_condition: list[ConditionCfg] = MISSING  # type: ignore
    ee_exec: ExecCfg = MISSING  # type: ignore
    gripper_exec: ExecCfg = MISSING  # type: ignore
    limits: tuple[float, float] = MISSING  # type: ignore
    noise: float = 0


@configclass
class StateMachineCfg:
    class_type: type[StateMachine] = StateMachine

    states_cfg: dict[int, StateCfg] = {}

    action_adapter_cfg: ActionAdapterCfg = ActionAdapterCfg()
