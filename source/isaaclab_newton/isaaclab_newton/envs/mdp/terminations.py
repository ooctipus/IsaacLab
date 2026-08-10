# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-specific termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab_newton.physics import NewtonManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def solver_reset_required(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate worlds that the active Newton solver has marked for reset.

    The MuJoCo Warp manager updates the cached mask once after all solver
    substeps in a controller step. This term only returns its zero-copy Torch
    view; the environment's normal reset events restore physical state and
    activate Newton's masked solver reset.

    Args:
        env: The environment. It must use Newton's MuJoCo Warp solver.

    Returns:
        Per-environment termination flags, shape ``(num_envs,)``.

    Raises:
        RuntimeError: If the active Newton solver does not expose reset information.
    """
    reset_required = NewtonManager.get_solver_reset_required()
    if reset_required.shape != (env.num_envs,):
        raise RuntimeError(
            f"Newton solver reset mask has shape {tuple(reset_required.shape)}; expected ({env.num_envs},)."
        )
    return reset_required
