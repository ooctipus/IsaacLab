# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Featherstone Newton manager."""

from __future__ import annotations

from newton import Model
from newton.solvers import SolverFeatherstone

from .featherstone_manager_cfg import FeatherstoneSolverCfg
from .newton_manager import NewtonManager


class NewtonFeatherstoneManager(NewtonManager):
    """:class:`NewtonManager` specialization for the Featherstone solver.

    Always uses Newton's :class:`CollisionPipeline` for contact handling.
    """

    def _create_solver(self, model: Model, solver_cfg: FeatherstoneSolverCfg) -> SolverFeatherstone:
        """Construct the configured Featherstone solver."""
        return SolverFeatherstone(model, **self._filter_solver_kwargs(SolverFeatherstone, solver_cfg))

    def _build_solver(self, model: Model, solver_cfg: FeatherstoneSolverCfg) -> None:
        """Construct :class:`SolverFeatherstone` and populate the base-class slots.

        Featherstone always uses Newton's :class:`CollisionPipeline` and steps
        with separate input/output states, so the flags are fixed.
        """
        self._solver = self._create_solver(model, solver_cfg)
        self._use_single_state = False
        self._needs_collision_pipeline = True
        self._newton._supports_rigid_body_force_input = True
