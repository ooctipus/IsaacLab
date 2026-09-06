# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FeatherPGS solver implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from newton import State
from newton.solvers import SolverFeatherPGS

from .newton_solver import NewtonSolver

if TYPE_CHECKING:
    from .feather_pgs_solver_cfg import FeatherPGSSolverCfg


class FeatherPGSSolver(NewtonSolver):
    """Newton's FeatherPGS solver, which double-buffers state and takes contacts from Newton's collision pipeline."""

    cfg: FeatherPGSSolverCfg

    solver_class = SolverFeatherPGS

    def reset(self, state: State, world_mask: wp.array) -> None:
        """Clear solver history for the masked environment worlds.

        Isaac Lab's mask has a final entry for Newton's global world, while FeatherPGS takes one entry per model world.
        """
        solver = self.native
        solver.reset(state, world_mask=world_mask[: solver.model.world_count], flags=0)
