# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp Newton manager."""

from __future__ import annotations

import inspect
import logging
import warnings

import numpy as np
from newton import Contacts, Model, ModelBuilder
from newton.solvers import SolverMuJoCo
from newton.usd import SchemaResolverMjc

from isaaclab.physics import PhysicsManager

from .mjwarp_manager_cfg import MJWarpSolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)


class NewtonMJWarpManager(NewtonManager):
    """:class:`NewtonManager` specialization for the MuJoCo Warp solver.

    Owns construction of :class:`SolverMuJoCo`, contact-buffer allocation in
    both internal-MuJoCo and Newton-pipeline contact modes, and the debug
    convergence logging emitted from :meth:`_log_solver_debug` when
    :attr:`NewtonCfg.debug_mode` is enabled.
    """

    @classmethod
    def _usd_schema_resolvers(cls) -> tuple[object, ...]:
        """Include authored MuJoCo fields in every MJWarp USD import."""
        return (SchemaResolverMjc(), *super()._usd_schema_resolvers())

    @classmethod
    def _convert_mjc_equality_constraints(cls) -> bool:
        """Preserve native MuJoCo equality rows for :class:`SolverMuJoCo`."""
        return False

    @classmethod
    def _register_builder_attributes(cls, builder: ModelBuilder) -> None:
        """Register MuJoCo fields before USD import and builder replication."""
        if not builder.has_custom_attribute("mujoco:solref"):
            SolverMuJoCo.register_custom_attributes(builder)

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: MJWarpSolverCfg) -> None:
        """Construct :class:`SolverMuJoCo` and populate the base-class slots.

        Filters cfg fields against the solver's ``__init__`` signature so
        non-constructor metadata (``solver_type``, ``class_type``) and the
        ignored deprecated ``ls_parallel`` field are not forwarded. Sets
        :attr:`NewtonManager._needs_collision_pipeline` to
        ``True`` only when ``use_mujoco_contacts=False``.
        """
        ignored = {"class_type", "solver_type", "ls_parallel"}
        valid = set(inspect.signature(SolverMuJoCo.__init__).parameters) - {"self", "model"} - ignored
        kwargs = {k: v for k, v in solver_cfg.to_dict().items() if k in valid}
        if solver_cfg.enable_native_ccd:
            solver = SolverMuJoCo(model, **kwargs)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"(Geom .*|Pair .*): authored margin=.*zeroed for NATIVECCD/MULTICCD compatibility.*",
                    category=UserWarning,
                    module=r"newton\._src\.solvers\.mujoco\.solver_mujoco",
                )
                solver = SolverMuJoCo(model, **kwargs)
            cls._disable_native_ccd(solver)
        NewtonManager._solver = solver
        NewtonManager._use_single_state = True
        NewtonManager._needs_collision_pipeline = not solver_cfg.use_mujoco_contacts

        cfg = PhysicsManager._cfg
        # Cross-config validation that needs both halves.
        if solver_cfg.use_mujoco_contacts and cfg.collision_cfg is not None:
            raise ValueError(
                "NewtonCfg: collision_cfg cannot be set when "
                "solver_cfg.use_mujoco_contacts=True. Either set "
                "use_mujoco_contacts=False or remove collision_cfg."
            )

    @classmethod
    def step(cls) -> None:
        """Step MuJoCo-Warp and publish body kinematics from the final generalized state.

        MuJoCo-Warp owns generalized-coordinate integration. Re-evaluating Newton
        forward kinematics once at the simulation-step boundary makes
        :attr:`newton.State.body_q` and :attr:`newton.State.body_qd` consistent
        with the final ``joint_q`` and ``joint_qd`` exposed to asset data and
        terminal-observation consumers.
        """
        sim = PhysicsManager._sim
        is_playing = sim is not None and sim.is_playing()
        super().step()
        if is_playing:
            cls.forward()

    @staticmethod
    def _disable_native_ccd(solver: SolverMuJoCo) -> None:
        """Select MJWarp's margin-compatible primitive collision path.

        MuJoCo Warp rejects nonzero margins for native box-pair CCD and for
        eligible multi-contact CCD pairs. :class:`MJWarpSolverCfg` rejects the
        multi-contact combination before this method runs. Newton currently
        zeroes all authored margins while constructing such a model, so restore
        the registered shape and explicit-pair values before CUDA graph capture.
        """
        mujoco = solver._mujoco
        disableflags = int(solver.mj_model.opt.disableflags)
        disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        solver.mj_model.opt.disableflags = disableflags
        solver.mjw_model.opt.disableflags = disableflags

        solver._zero_margins_for_native_ccd = False
        use_mujoco_contacts = solver._use_mujoco_contacts
        solver._use_mujoco_contacts = False
        try:
            solver._update_geom_properties()
            solver._update_pair_properties()
        finally:
            solver._use_mujoco_contacts = use_mujoco_contacts

        solver.mj_model.geom_margin[:] = solver.mjw_model.geom_margin.numpy()[0]
        solver.mj_model.geom_gap[:] = solver.mjw_model.geom_gap.numpy()[0]
        if solver.mj_model.npair:
            solver.mj_model.pair_margin[:] = solver.mjw_model.pair_margin.numpy()[0]
            solver.mj_model.pair_gap[:] = solver.mjw_model.pair_gap.numpy()[0]

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Allocate contact buffers.

        Delegates to the base implementation when Newton's
        :class:`CollisionPipeline` is active.  When ``use_mujoco_contacts=True``
        the solver runs MuJoCo's internal collision detection, so this method
        instead pre-allocates a :class:`Contacts` buffer sized to the solver's
        maximum contact count; ``solver.update_contacts`` later populates it
        from MuJoCo data for contact-sensor reporting.
        """
        if cls._needs_collision_pipeline:
            super()._initialize_contacts()
            return
        if cls._solver is not None:
            NewtonManager._contacts = Contacts(
                rigid_contact_max=cls._solver.get_max_contact_count(),
                soft_contact_max=0,
                device=PhysicsManager._device,
                requested_attributes=cls._model.get_requested_contact_attributes(),
            )

    @classmethod
    def _log_solver_debug(cls) -> None:
        """Optionally log MuJoCo solver convergence at the end of step."""
        cfg = PhysicsManager._cfg
        if cfg is not None and cfg.debug_mode:  # type: ignore[union-attr]
            data = cls._get_solver_convergence_steps()
            logger.info(f"Solver convergence data: {data}")
            if data["max"] == cls._solver.mjw_model.opt.iterations:
                logger.warning(f"Solver didn't converge! max_iter={data['max']}")

    @classmethod
    def _get_solver_convergence_steps(cls) -> dict[str, float | int]:
        """Return MuJoCo Warp solver convergence statistics.

        Reads ``mjw_data.solver_niter`` (only available on
        :class:`SolverMuJoCo`) and summarizes per-environment iteration counts.
        """
        niter = cls._solver.mjw_data.solver_niter.numpy()
        return {
            "max": np.max(niter),
            "mean": np.mean(niter),
            "min": np.min(niter),
            "std": np.std(niter),
        }
