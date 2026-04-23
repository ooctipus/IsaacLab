# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import torch

from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .multi_task_command import MultiTaskCommand


@configclass
class MinMaxSampler:
    kernel: int = MISSING
    minimum: list[float] = MISSING
    maximum: list[float] = MISSING
    out_dim: int | None = None
    """Override for the kernel's output dimension.

    Some kernels emit a tensor whose last dim is different from the ``len(minimum)``
    param count — e.g. :data:`SAMPLER_KERNEL_ID.EULER_UNIFORM_TO_QUAT` takes 3 Euler
    pairs and emits a 4-vector quaternion. Set ``out_dim`` to the real output size so
    the command term's ``target_dim_max`` (derived from ``len(get_kernel_input()) // 2``)
    is ≥ the state kernel's output dim.

    When ``out_dim > len(minimum)``, the encoded param tensor is zero-padded so its
    length is ``2 * out_dim``. Padded pairs carry no information — the kernel simply
    ignores them. Leave this ``None`` for kernels whose output dim equals the param
    count (e.g. :data:`SAMPLER_KERNEL_ID.UNIFORM`).
    """

    def get_kernel_input(self, device="cpu") -> torch.Tensor:
        """Return sampler params as a flat 1D tensor.

        Encoding (interleaved pairs): ``[min0, range0, min1, range1, ...]``. If
        :attr:`out_dim` exceeds ``len(minimum)``, trailing zero pairs are appended.
        :class:`MultiTaskCommand._build_spec` additionally zero-pads rows to the
        maximum ``P`` across subtasks.
        """
        mn = torch.tensor(self.minimum, device=device, dtype=torch.float32)
        mx = torch.tensor(self.maximum, device=device, dtype=torch.float32)
        rg = mx - mn

        n = mn.numel()
        target_n = max(n, self.out_dim or 0)
        if target_n > n:
            pad = torch.zeros(target_n - n, device=device, dtype=torch.float32)
            mn = torch.cat([mn, pad])
            rg = torch.cat([rg, pad])

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
