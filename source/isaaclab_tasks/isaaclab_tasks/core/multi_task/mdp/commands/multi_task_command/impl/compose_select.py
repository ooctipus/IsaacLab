# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared compose-kernel selection.

The serial 1-thread-per-env composer is faster when ``k_max`` is small enough
that block-per-env launches would underutilize the warp (each block uses only
``k_max`` of 32 lanes). The block-per-env parallel composer wins once
``k_max`` reaches the warp size — at ``k_max ≈ 288`` the parallel path is
~2.3× faster on the compose phase alone.

Default: pick based on ``k_max``. Override with ``MULTI_TASK_COMPOSE_PARALLEL``
for benchmarking / debugging.
"""

from __future__ import annotations

import os

# Crossover where the parallel composer starts beating the serial one,
# determined empirically (RTX 5090, Warp 1.12). Roughly equals warp size:
# below this, the per-block thread count is too small to fill a warp.
_PARALLEL_K_MAX_THRESHOLD = 32

# Optional override:
#   "1" forces parallel, "0" forces serial, unset → adaptive on k_max.
_COMPOSE_OVERRIDE = os.environ.get("MULTI_TASK_COMPOSE_PARALLEL")


def use_parallel_compose(k_max: int) -> bool:
    """Return True if the parallel compose kernel should be launched."""
    if _COMPOSE_OVERRIDE == "1":
        return True
    if _COMPOSE_OVERRIDE == "0":
        return False
    return k_max >= _PARALLEL_K_MAX_THRESHOLD
