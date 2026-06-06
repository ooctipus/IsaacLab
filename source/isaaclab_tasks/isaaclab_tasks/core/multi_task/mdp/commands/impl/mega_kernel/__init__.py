# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Current production backend: one branchy ``dispatch_mega`` kernel."""

from .backend import MegaKernelBackend
from .bindings import MegaKernelPlan, RotationBinding, build_mega_kernel_plan, build_rotation_bindings
from .compose import compose_warp
from .execute import dispatch_mega_warp
from .read import fill_unified_buffer_warp
from .rotation import rotate_canonical_slots_to_body_frame_warp

__all__ = [
    "MegaKernelBackend",
    "MegaKernelPlan",
    "RotationBinding",
    "build_mega_kernel_plan",
    "build_rotation_bindings",
    "compose_warp",
    "dispatch_mega_warp",
    "fill_unified_buffer_warp",
    "rotate_canonical_slots_to_body_frame_warp",
]
