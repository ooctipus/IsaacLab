# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FeatherPGS solver implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from newton import CollisionPipeline, Model, State
from newton.solvers import SolverFeatherPGS

from .newton_solver import NewtonSolver, solver_kwargs

if TYPE_CHECKING:
    from .feather_pgs_solver_cfg import FeatherPGSSolverCfg

logger = logging.getLogger(__name__)


class FeatherPGSSolver(NewtonSolver):
    """Newton's FeatherPGS solver, which double-buffers state and takes contacts from Newton's collision pipeline."""

    cfg: FeatherPGSSolverCfg

    solver_class = SolverFeatherPGS

    collision_pipeline: CollisionPipeline | None = None
    """The collision pipeline built over the solver's model, once contacts are allocated."""

    contact_drop_worlds: np.ndarray | None = None
    """Worlds whose rigid contacts the last step's status check found dropped, or ``None``.

    A boolean mask of one entry per world; a task may end those episodes.
    """

    _contact_drop_passes: int = 0
    """Collision passes that dropped contacts, as of the last status check."""

    _contact_drop_seen: int = 0
    """The last collision pass whose dropped contacts the status check reported."""

    def build(self, model: Model, mode: wp.DeterministicMode, env_template: str | None) -> SolverFeatherPGS:
        """Construct FeatherPGS, passing its diagnostics as Newton's one record."""
        kwargs = solver_kwargs(self.solver_class, self.cfg, mode)
        diagnostics = self.cfg.diagnostics.to_dict()
        if diagnostics["sync_timings"] is not None:
            diagnostics["sync_timings"] = tuple(diagnostics["sync_timings"])
        kwargs["diagnostics"] = SolverFeatherPGS.Diagnostics(**diagnostics)
        return self.solver_class(model, **kwargs)

    def create_collision_pipeline(self, **kwargs) -> CollisionPipeline:
        """Construct the collision pipeline with the configured FeatherPGS collision choices."""
        kwargs["convex_collision"] = self.cfg.convex_collision
        kwargs["contact_export"] = self.cfg.contact_export
        self.collision_pipeline = super().create_collision_pipeline(**kwargs)
        return self.collision_pipeline

    def path_selection(self) -> dict:
        """Read the solver's and collision pipeline's path choices after initialization, outside graph capture."""
        if self.collision_pipeline is None:
            raise RuntimeError("Initialize FeatherPGS before inspecting selected paths")
        return {
            "solver": self.native.get_path_selection(),
            "collision": self.collision_pipeline.get_collision_selection(),
        }

    def reset(self, state: State, world_mask: wp.array) -> None:
        """Clear solver history for the masked environment worlds.

        Isaac Lab's mask has a final entry for Newton's global world, while FeatherPGS takes one entry per model world.
        """
        solver = self.native
        solver.reset(state, world_mask=world_mask[: solver.model.world_count], flags=0)

    def check_step(self, captured: bool, debug: bool) -> None:
        """Report the rigid contacts the collision passes dropped for want of rows, without a synchronization.

        The pipeline's contact pool grows past a pass that dropped contacts; each step whose completed passes dropped
        some logs them, with their worlds, which :attr:`contact_drop_worlds` holds until the next step.
        """
        contact_drops = getattr(self.collision_pipeline, "contact_drops", None)
        self.contact_drop_worlds = None
        if contact_drops is None:
            return
        drops = contact_drops()
        if drops.passes == self._contact_drop_passes:
            return
        latched = drops.worlds > self._contact_drop_seen
        self.contact_drop_worlds = latched[:-1].copy()
        logger.warning(
            "FeatherPGS: %d collision pass(es) up to pass %d dropped rigid contacts for want of rows (%d dropped so"
            " far) in %d world(s)%s; the contact buffers grow for the next step",
            drops.passes - self._contact_drop_passes,
            drops.last_pass,
            drops.contacts,
            int(self.contact_drop_worlds.sum()),
            " and between global shapes" if latched[-1] else "",
        )
        self._contact_drop_passes = drops.passes
        self._contact_drop_seen = drops.last_pass
