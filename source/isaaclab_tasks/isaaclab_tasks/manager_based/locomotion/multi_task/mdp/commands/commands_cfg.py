import torch
from dataclasses import MISSING
from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .multi_task_command import MultiTaskCommand

@configclass
class MinMaxSampler:
    kernel: int = MISSING
    minimum: list[float] = MISSING
    maximum: list[float] = MISSING

    def get_kernel_input(self, device="cpu") -> torch.Tensor:
        """Return sampler params as a flat 1D tensor.

        Encoding (interleaved pairs):
            [min0, range0, min1, range1, ...]
        Padding is done later by MultiTaskCommand._build_spec (zeros appended).
        """
        mn = torch.tensor(self.minimum, device=device, dtype=torch.float32)
        mx = torch.tensor(self.maximum, device=device, dtype=torch.float32)
        rg = mx - mn

        out = torch.empty(mn.numel() * 2, device=device, dtype=torch.float32)
        out[0::2] = mn
        out[1::2] = rg
        return out


@configclass
class MultiTaskCfg(CommandTermCfg):
    @configclass
    class BaseTaskCfg:
        asset_cfg: SceneEntityCfg = MISSING
        metric_kernel: int = MISSING
        state_kernel: int = MISSING
        sampler: MinMaxSampler = MISSING
        activation_kernel: int = MISSING
        activation_kernel_param: float = MISSING

    @configclass
    class TrackingTaskCfg(BaseTaskCfg):
        pass

    @configclass
    class InstantaneousTaskCfg(BaseTaskCfg):
        pass

    class_type: type = MultiTaskCommand
    tasks: dict[str, list[BaseTaskCfg]] = MISSING
