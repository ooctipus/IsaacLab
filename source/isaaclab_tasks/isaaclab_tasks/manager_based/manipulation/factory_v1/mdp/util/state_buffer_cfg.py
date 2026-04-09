# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class StateBufferCfg:
    """Configuration for :class:`StateBuffer`.

    Attributes:
        size: Maximum number of states the buffer can hold.
        tag_names_bind: Eval expression to obtain tag name list at runtime.
        tag_indices_bind: Eval expression to obtain per-env tag index tensor at runtime.
    """

    size: int = 32768
    tag_names_bind: str | None = None
    tag_indices_bind: str | None = None
