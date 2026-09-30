# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino Newton manager."""

from __future__ import annotations

import logging
from typing import ClassVar

import warp as wp
from newton import Model, eval_fk
from newton._src.solvers.kamino.config import (
    CollisionDetectorConfig,
    ConstrainedDynamicsConfig,
    ConstraintStabilizationConfig,
    DVISolverConfig,
    ForwardKinematicsSolverConfig,
    MaterialManagerConfig,
    PADMMSolverConfig,
)
from newton.solvers import SolverBase, SolverKamino

from .kamino_manager_cfg import KaminoDVISolverCfg, KaminoPADMMSolverCfg, _KaminoSolverCfgBase
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)


def _model_has_loop_closing_joints(model: Model) -> bool:
    """Return whether ``model`` contains converted loop-closing articulation joints.

    Newton stores regular tree joints in ``[articulation_start[i], articulation_end[i])`` and
    loop-closing joints in ``[articulation_end[i], articulation_start[i + 1])``. Loop closures
    are present when the next articulation sentinel exceeds the tree joint end for any
    articulation.

    Args:
        model: Finalized Newton model to inspect.

    Returns:
        ``True`` if at least one articulation has loop-closing joints.
    """
    articulation_start = model.articulation_start
    articulation_end = model.articulation_end
    if articulation_start is None or articulation_end is None:
        return False
    articulation_start_np = articulation_start.numpy()
    articulation_end_np = articulation_end.numpy()
    if articulation_end_np.shape[0] == 0:
        return False
    return bool((articulation_start_np[1:] > articulation_end_np).any())


class NewtonKaminoManager(NewtonManager):
    """:class:`NewtonManager` specialization for the Kamino solver.

    Uses Newton's :class:`CollisionPipeline` unless
    its ``use_collision_detector`` field is ``True``, in which case Kamino's
    internal collision detector handles contact generation.
    """

    # Annotate the concrete solver type.
    _solver: SolverKamino
    _builder_attribute_solvers: ClassVar[tuple[type[SolverBase], ...]] = (SolverKamino,)

    def _get_kamino_solver_cfg(self) -> _KaminoSolverCfgBase:
        cfg = self._cfg
        if cfg is None:
            raise RuntimeError("Physics manager is not initialized.")
        if not isinstance(cfg, _KaminoSolverCfgBase):
            raise TypeError(f"Expected a Kamino solver configuration, got {type(cfg).__name__}.")
        return cfg

    def _eval_fk_impl(self, world_reset_mask: wp.array | None, fk_mask: wp.array | None) -> None:
        """Update body states from joint coordinates.

        For the Kamino (maximal-coordinate) solver, body poses/velocities are the authoritative
        simulation state. When ``use_fk_solver`` is enabled, this calls
        :meth:`SolverKamino.reset`, which runs Kamino's loop-closure forward kinematics: it reads
        body poses/velocities from the joint coordinates (including the base body's pose/twist)
        and writes back a consistent full joint and body state.

        When ``use_fk_solver`` is disabled, falls back to Newton's articulated ``eval_fk`` over
        ``fk_mask``; the caller is then responsible for writing constraint-consistent joint values.

        Args:
            world_reset_mask: Per-world mask passed to :meth:`SolverKamino.reset` (``None`` means all).
            fk_mask: Per-articulation mask of articulations to update (``None`` means all).
        """
        if self._get_kamino_solver_cfg().use_fk_solver:
            self._solver.reset(
                self._newton._state_0,
                world_mask=world_reset_mask,
                config=SolverKamino.ResetConfig.from_joints(),
            )
        else:
            eval_fk(
                self._newton._model,
                self._newton._state_0.joint_q,
                self._newton._state_0.joint_qd,
                self._newton._state_0,
                fk_mask,
            )

            # Reset solver internals without performing Kamino's FK.
            self._solver.reset(
                self._newton._state_0,
                world_mask=world_reset_mask,
                config=SolverKamino.ResetConfig.preserve(),
            )

    def _reset_solver_internals(self, world_mask: wp.array | None) -> None:
        """Skip the generic solver reset.

        :meth:`_eval_fk_impl` already performs the masked
        :meth:`SolverKamino.reset` with an explicit reset configuration.

        Args:
            world_mask: Unused; accepted to match the base hook signature.
        """

    def _create_solver(self, model: Model, solver_cfg: _KaminoSolverCfgBase) -> SolverKamino:
        """Construct the configured Kamino solver."""
        collision_detector = None
        if solver_cfg.use_collision_detector:
            collision_detector = CollisionDetectorConfig(
                **{key: value for key, value in solver_cfg.collision_detector.to_dict().items() if value is not None}
            )

        padmm = PADMMSolverConfig()
        dvi = DVISolverConfig()
        if isinstance(solver_cfg, KaminoPADMMSolverCfg):
            dynamics_solver = "padmm"
            padmm = PADMMSolverConfig(**solver_cfg.dynamics_solver_cfg.to_dict())
        elif isinstance(solver_cfg, KaminoDVISolverCfg):
            dynamics_solver = "dvi"
            dvi = DVISolverConfig(**solver_cfg.dynamics_solver_cfg.to_dict())
        else:
            raise TypeError(f"Expected a concrete Kamino solver configuration, got {type(solver_cfg).__name__}.")

        config = SolverKamino.Config(
            dynamics_solver=dynamics_solver,
            integrator=solver_cfg.integrator,
            use_collision_detector=solver_cfg.use_collision_detector,
            use_fk_solver=True if solver_cfg.use_fk_solver is None else solver_cfg.use_fk_solver,
            sparse_jacobian=solver_cfg.sparse_jacobian,
            sparse_dynamics=solver_cfg.sparse_dynamics,
            rotation_correction=solver_cfg.rotation_correction,
            angular_velocity_damping=solver_cfg.angular_velocity_damping,
            collect_solver_info=solver_cfg.collect_solver_info,
            compute_solution_metrics=solver_cfg.compute_solution_metrics,
            collision_detector=collision_detector,
            fk=ForwardKinematicsSolverConfig(**solver_cfg.fk.to_dict()),
            constraints=ConstraintStabilizationConfig(**solver_cfg.constraints.to_dict()),
            dynamics=(
                None if solver_cfg.dynamics is None else ConstrainedDynamicsConfig(**solver_cfg.dynamics.to_dict())
            ),
            materials=MaterialManagerConfig(**solver_cfg.materials.to_dict()),
            padmm=padmm,
            dvi=dvi,
        )
        config.validate()
        return SolverKamino(model, config)

    def _build_solver(self, model: Model, solver_cfg: _KaminoSolverCfgBase) -> None:
        """Construct :class:`SolverKamino` and populate the base-class slots.

        Sets :attr:`self._needs_collision_pipeline` to ``True`` only
        when ``use_collision_detector=False`` (Kamino's internal detector
        handles contacts otherwise).

        Kamino treats body state as authoritative. The shared pre-step
        :meth:`self.forward` boundary reconciles authored joint state
        only for worlds selected by :attr:`self._world_reset_mask`.

        Raises:
            RuntimeError: If the model has more than one articulation per environment. The Kamino
                interface in IsaacLab currently only supports one articulation per environment.
        """
        # Set the max contacts per world if specified.
        if solver_cfg.max_contacts_per_world is not None:
            model.rigid_contact_max = int(solver_cfg.max_contacts_per_world) * model.world_count
            logger.info(
                "[KAMINO] Capping rigid_contact_max to %d (%d/world * %d worlds)",
                model.rigid_contact_max,
                solver_cfg.max_contacts_per_world,
                model.world_count,
            )

        # Set the use_fk_solver flag based on the model's articulation structure if not specified by user.
        if solver_cfg.use_fk_solver is None:
            solver_cfg.use_fk_solver = _model_has_loop_closing_joints(model)

        if solver_cfg.use_fk_solver and model.articulation_count != model.world_count:
            raise RuntimeError(
                "The Kamino FK solver requires exactly one articulation per environment, but the model"
                f" has {model.articulation_count} articulations across {model.world_count} environments."
                " Multiple articulations per environment are not yet supported in Kamino's FK solver."
            )

        self._solver = self._create_solver(model, solver_cfg)
        self._use_single_state = False
        self._needs_collision_pipeline = not solver_cfg.use_collision_detector
        self._newton._supports_rigid_body_force_input = True
