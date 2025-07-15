from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg

from ....factory_env_cfg import ASSEMBLY_TASKS
from ....tasks import TaskKeyPointCfg
from ....tasks import SuccessCondition
from ....tasks import nist_board_tasks_auxiliary_key_points_cfg as aux_tasks
from .... import mdp
from ....mdp.data_cfg import KeyPointDataCfg as Kp

from ....state_machine import StateMachineCfg
from ....state_machine import RelJointPosAdapterCfg

from .states import FRANKA_FACTORY_STATES


@configclass
class FactoryAuxiliaryTasksCfg:

    # assembly tasks
    auxiliary_tasks: list[TaskKeyPointCfg] = []

    def __post_init__(self):
        self.auxiliary_tasks = [getattr(aux_tasks, task.__class__.__name__)() for task in ASSEMBLY_TASKS.tasks]

    @property
    def task_names(self) -> list[str]:
        return [type(task).__name__ for task in self.auxiliary_tasks]

    @property
    def auxiliary_tasks_dict(self) -> dict[str, TaskKeyPointCfg]:
        return {type(task).__name__: task for task in self.auxiliary_tasks}

    @property
    def auxiliary_task_alignment_metric(self) -> list[SuccessCondition]:
        return [SuccessCondition(pos_threshold=(0.002, 0.002, 0.002), rot_threshold=(0.025, 0.025, 0.025)) for _ in ASSEMBLY_TASKS.tasks]


AUXILIARY_TASKS = FactoryAuxiliaryTasksCfg()


@configclass
class FactoryAuxiliaryDataCfg:
    """Auxiliary Data for Factory State Machine"""

    auxiliary_task_key_points = mdp.KeyPointTrackerCfg(
        spec=AUXILIARY_TASKS.auxiliary_tasks_dict,
        context_id_callback=mdp.task_id_callback,
    )

    auxiliary_task_alignment_data = mdp.AlignmentMetricCfg(
        spec=AUXILIARY_TASKS.auxiliary_task_alignment_metric,
        context_id_callback=mdp.task_id_callback,
        data_manager_hook='env.extensions["fsm"].data_manager',
        align_kp_cfg=Kp(term="auxiliary_task_key_points", kp_names="asset_align"),
        align_against_kp_cfg=Kp(term="auxiliary_task_key_points", kp_names="asset_align_against")
    )


@configclass
class FactoryFrankaStateMachineCfg(StateMachineCfg):

    states_cfg = FRANKA_FACTORY_STATES

    data = FactoryAuxiliaryDataCfg()

    robot_cfg = SceneEntityCfg("robot", body_names="panda_hand")


@configclass
class FactoryFrankaStateMachineRelJointPositionAdapterCfg(FactoryFrankaStateMachineCfg):

    action_adapter_cfg = RelJointPosAdapterCfg(
        ik_solver=DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            scale=1,
        )
    )
