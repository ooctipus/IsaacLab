# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp Newton manager."""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
import warp as wp
from newton import Contacts, Model
from newton.solvers import SolverBase, SolverMuJoCo

from .mjwarp_manager_cfg import MJWarpSolverCfg
from .mjwarp_tendon_control import MjWarpTendonControl
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)


class NewtonMJWarpManager(NewtonManager):
    """:class:`NewtonManager` specialization for the MuJoCo Warp solver.

    Owns construction of :class:`SolverMuJoCo`, contact-buffer allocation in
    both internal-MuJoCo and Newton-pipeline contact modes, and the debug
    convergence logging emitted from :meth:`_log_solver_debug` when
    :attr:`MJWarpSolverCfg.debug_mode` is enabled.
    """

    _builder_attribute_solvers: ClassVar[tuple[type[SolverBase], ...]] = (SolverMuJoCo,)

    def _create_solver(self, model: Model, solver_cfg: MJWarpSolverCfg) -> SolverMuJoCo:
        """Construct the configured MuJoCo Warp solver."""
        return SolverMuJoCo(model, **self._filter_solver_kwargs(SolverMuJoCo, solver_cfg))

    def _build_solver(self, model: Model, solver_cfg: MJWarpSolverCfg) -> None:
        """Construct :class:`SolverMuJoCo` and populate the base-class slots.

        Filters cfg fields against the solver's ``__init__`` signature so
        manager metadata such as ``class_type`` is not forwarded. Sets
        :attr:`self._needs_collision_pipeline` to
        ``True`` only when ``use_mujoco_contacts=False``.
        """
        self._solver = self._create_solver(model, solver_cfg)
        self._use_single_state = True
        self._needs_collision_pipeline = not solver_cfg.use_mujoco_contacts
        self._newton._supports_rigid_body_force_input = True

        if solver_cfg.use_mujoco_contacts and solver_cfg.collision_cfg is not None:
            raise ValueError(
                "MJWarpSolverCfg.collision_cfg cannot be set when use_mujoco_contacts=True. Either set "
                "use_mujoco_contacts=False or remove collision_cfg."
            )

    def create_fixed_tendon_control(self, articulation):
        """Build the MuJoCo tendon adapter for ``articulation``.

        Args:
            articulation: Newton articulation to drive.

        Returns:
            The adapter, or None when no MuJoCo actuator transmits to any of its tendons.
        """
        return MjWarpTendonControl.create(articulation, self.get_model())

    def _initialize_contacts(self) -> None:
        """Allocate contact buffers.

        Delegates to the base implementation when Newton's
        :class:`CollisionPipeline` is active.  When ``use_mujoco_contacts=True``
        the solver runs MuJoCo's internal collision detection, so this method
        instead pre-allocates a :class:`Contacts` buffer sized to the solver's
        maximum contact count; ``solver.update_contacts`` later populates it
        from MuJoCo data for contact-sensor reporting.
        """
        if self._needs_collision_pipeline:
            super()._initialize_contacts()
            return
        if self._solver is not None:
            self._newton._contacts = Contacts(
                rigid_contact_max=self._solver.get_max_contact_count(),
                soft_contact_max=0,
                device=self._device,
                requested_attributes=self._newton._model.get_requested_contact_attributes(),
            )

    def _reset_solver_internals(self, world_mask: wp.array | None) -> None:
        """Clear MuJoCo Warp solver-internal state for flagged worlds.

        Specializes the base hook, whose :meth:`SolverBase.reset` call resolves
        to :meth:`SolverMuJoCo.reset` here: with ``flags=0`` it zeroes only the
        solver-owned buffers persisting across steps (``qacc_warmstart``,
        ``qfrc_applied``, ``xfrc_applied``, ``ctrl``, ``act``) for the flagged
        worlds, while the joint state IsaacLab authored during the env reset is
        left untouched.  Without this, a NaN produced in one solve persists
        across :meth:`isaaclab.envs.ManagerBasedEnv.reset` because the next
        solver substep warm-starts from the NaN — the world is then permanently
        dead.  See https://github.com/newton-physics/newton/issues/1266.

        With ``use_mujoco_cpu=True`` the solver owns a single global ``MjData``
        and its reset path is not mask-aware — it clears the buffers for every
        world.  Since this hook fires on every step/forward boundary (usually
        with an all-``False`` mask), the CPU path is gated on at least one
        world actually being flagged so warm-starting is not defeated on every
        step.

        Args:
            world_mask: Per-world bool mask of shape ``(world_count + 1,)``.
                Entries before the last select local worlds; the final entry
                selects global entities in world -1. ``None`` is a no-op.
        """
        if world_mask is None:
            return
        if self._solver.use_mujoco_cpu and not world_mask.numpy().any():
            return
        # flags=0 skips the joint-state reset to model defaults: IsaacLab owns
        # joint_q/joint_qd and has already written the authored reset pose.
        self._solver.reset(self._newton._state_0, world_mask=world_mask, flags=0)

    def _log_solver_debug(self) -> None:
        """Optionally log MuJoCo solver convergence at the end of step."""
        cfg = self._cfg
        if cfg is not None and cfg.debug_mode:  # type: ignore[union-attr]
            data = self._get_solver_convergence_steps()
            logger.info(f"Solver convergence data: {data}")
            if data["max"] == self._solver.mjw_model.opt.iterations:
                logger.warning(f"Solver didn't converge! max_iter={data['max']}")

    def _get_solver_convergence_steps(self) -> dict[str, float | int]:
        """Return MuJoCo Warp solver convergence statistics.

        Reads ``mjw_data.solver_niter`` (only available on
        :class:`SolverMuJoCo`) and summarizes per-environment iteration counts.
        """
        niter = self._solver.mjw_data.solver_niter.numpy()
        return {
            "max": np.max(niter),
            "mean": np.mean(niter),
            "min": np.min(niter),
            "std": np.std(niter),
        }
