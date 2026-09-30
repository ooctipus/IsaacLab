# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD Newton manager."""

from __future__ import annotations

from newton import Model
from newton.solvers import SolverVBD

from .newton_manager import NewtonManager
from .vbd_manager_cfg import VBDSolverCfg


class NewtonVBDManager(NewtonManager):
    """Newton manager specialization for the VBD solver."""

    def start_simulation(self) -> None:
        """Color the VBD builder before simulation starts."""
        if self._newton._builder is not None:
            self._newton._builder.color()
        super().start_simulation()

    def _create_solver(self, model: Model, solver_cfg: VBDSolverCfg) -> SolverVBD:
        """Construct the configured VBD solver."""
        return SolverVBD(model, **self._filter_solver_kwargs(SolverVBD, solver_cfg))

    def _build_solver(self, model: Model, solver_cfg: VBDSolverCfg) -> None:
        """Construct VBD and configure its base-manager state."""
        self._solver = self._create_solver(model, solver_cfg)
        self._use_single_state = False
        self._needs_collision_pipeline = True
        self._newton._supports_rigid_body_force_input = not solver_cfg.integrate_with_external_rigid_solver

    def _simulate_physics_only(self) -> None:
        """Rebuild the VBD particle BVH before stepping physics."""
        if self._newton._model.particle_count > 0 and hasattr(self._solver, "rebuild_bvh"):
            self._solver.rebuild_bvh(self._newton._state_0)
        super()._simulate_physics_only()
