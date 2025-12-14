from dataclasses import MISSING
from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from .multi_task_command import MultiTaskCommand


@configclass
class MultiTaskCfg(CommandTermCfg):

    @configclass
    class BaseTaskCfg:
        # where to read x_cur
        asset_cfg: SceneEntityCfg = MISSING      # name + ids (joint_ids/body_ids)

        metric_kernel: int = MISSING             # ERROR_KERNEL_ID
        state_kernel: int = MISSING              # STATE_KERNEL_ID
        activation_kernel: int = MISSING         # ACTIVATION_KERNEL_ID
        activation_kernel_param: float = MISSING

    @configclass
    class TrackingTaskCfg(BaseTaskCfg):
        pass

    @configclass
    class InstantaneousTaskCfg(BaseTaskCfg):
        pass

    class_type: type = MultiTaskCommand
    tasks: dict[str, list[BaseTaskCfg]] = MISSING
