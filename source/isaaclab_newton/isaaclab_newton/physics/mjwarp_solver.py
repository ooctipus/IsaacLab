# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp solver implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from newton import Contacts, Model, State
from newton.solvers import SolverMuJoCo

from .newton_solver import NewtonSolver, solver_kwargs

if TYPE_CHECKING:
    from .mjwarp_solver_cfg import MJWarpSolverCfg
    from .mjwarp_tendon_control import MjWarpTendonControl

logger = logging.getLogger(__name__)

_SENSORS_BY_STATE_ATTRIBUTE = {"body_qdd": "the IMU or PVA sensor", "body_parent_f": "the joint-wrench sensor"}


@wp.kernel(enable_backward=False)
def _detect_solver_reset_required(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    qacc: wp.array2d(dtype=wp.float32),
    qfrc_actuator: wp.array2d(dtype=wp.float32),
    qacc_warmstart: wp.array2d(dtype=wp.float32),
    qfrc_applied: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    act: wp.array2d(dtype=wp.float32),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
    reset_required: wp.array(dtype=wp.bool),
):
    """Reduce policy-facing and persistent MuJoCo state validity per world."""
    world_id = wp.tid()
    invalid = wp.bool(False)

    for i in range(qpos.shape[1]):
        if not wp.isfinite(qpos[world_id, i]):
            invalid = True
    for i in range(qvel.shape[1]):
        if not wp.isfinite(qvel[world_id, i]):
            invalid = True
    for i in range(qacc.shape[1]):
        if not wp.isfinite(qacc[world_id, i]):
            invalid = True
    for i in range(qfrc_actuator.shape[1]):
        if not wp.isfinite(qfrc_actuator[world_id, i]):
            invalid = True
    for i in range(qacc_warmstart.shape[1]):
        if not wp.isfinite(qacc_warmstart[world_id, i]):
            invalid = True
    for i in range(qfrc_applied.shape[1]):
        if not wp.isfinite(qfrc_applied[world_id, i]):
            invalid = True
    for i in range(ctrl.shape[1]):
        if not wp.isfinite(ctrl[world_id, i]):
            invalid = True
    for i in range(act.shape[1]):
        if not wp.isfinite(act[world_id, i]):
            invalid = True
    for body_id in range(xfrc_applied.shape[1]):
        force = xfrc_applied[world_id, body_id]
        for component_id in range(6):
            if not wp.isfinite(force[component_id]):
                invalid = True

    reset_required[world_id] = invalid


class MJWarpSolver(NewtonSolver):
    """Newton's MuJoCo Warp solver, which steps in place on one state."""

    cfg: MJWarpSolverCfg

    solver_class = SolverMuJoCo
    single_state = True
    max_deterministic_mode = wp.DeterministicMode.RUN_TO_RUN

    @classmethod
    def applied_deterministic_mode(cls, cfg: MJWarpSolverCfg, mode: wp.DeterministicMode) -> wp.DeterministicMode:
        """MuJoCo on the CPU is reproducible on its own and Warp's mode does not reach it, so none applies."""
        if not cfg.use_mujoco_cpu:
            return mode
        if mode != wp.DeterministicMode.NOT_GUARANTEED:
            logger.info("MuJoCo CPU backend is already reproducible; Newton's deterministic mode is not applied.")
        return wp.DeterministicMode.NOT_GUARANTEED

    @classmethod
    def coupling_error(cls, cfg: MJWarpSolverCfg, in_place: bool) -> str | None:
        """MuJoCo on the CPU resets one global state, which a coupled entry cannot mask per world."""
        if cfg.use_mujoco_cpu:
            return (
                "enables MJWarpSolverCfg.use_mujoco_cpu, whose global reset state cannot yet preserve the per-world"
                " reset-mask lifecycle inside a coupled entry"
            )
        return None

    def build(self, model: Model, mode: wp.DeterministicMode, env_template: str | None) -> SolverMuJoCo:
        """Construct MuJoCo Warp; a determinism guarantee disables its sensor stage, which cannot honor it."""
        deterministic = mode != wp.DeterministicMode.NOT_GUARANTEED
        blocked = set(model.get_requested_state_attributes()) & _SENSORS_BY_STATE_ATTRIBUTE.keys()
        if deterministic and blocked:
            sensors = sorted({_SENSORS_BY_STATE_ATTRIBUTE[attr] for attr in blocked})
            raise ValueError(
                f"This task does not support deterministic physics: it uses {' and '.join(sensors)},"
                f" reading {sorted(blocked)}. Those attributes come from MuJoCo's post-constraint pass,"
                " which runs inside the sensor stage that a determinism guarantee must disable, so the"
                " values would never be refreshed. Remove the sensors, or drop the determinism request"
                f" (deterministic_mode={mode.name})."
            )
        kwargs = solver_kwargs(self.solver_class, self.cfg, mode)
        # ls_parallel is deprecated in newton; forwarding it (even as False) emits a warning.
        kwargs.pop("ls_parallel", None)
        if deterministic:
            kwargs["disable_sensors"] = True
        if model._has_body_variants:
            kwargs["separate_worlds"] = True
        return self.solver_class(model, **kwargs)

    def __init__(self, cfg, model, mode, env_template=None):
        super().__init__(cfg, model, mode, env_template)
        self.reset_required = wp.zeros(model.world_count, dtype=wp.bool, device=model.device)

    def supports_capture(self) -> bool:
        return not self.cfg.use_mujoco_cpu

    def uses_collision_pipeline(self) -> bool:
        return not self.cfg.use_mujoco_contacts

    def create_contacts(self) -> Contacts:
        """Allocate contacts sized to MuJoCo's contact buffer for sensor contact indexing."""
        solver = self.native
        model = solver.model
        return Contacts(
            rigid_contact_max=solver.get_max_contact_count(),
            soft_contact_max=0,
            device=model.device,
            requested_attributes=model.get_requested_contact_attributes(),
        )

    def reset(self, state: State, world_mask: wp.array) -> None:
        """Clear solver history and synchronize authored joint state for masked worlds.

        With ``flags=0`` MuJoCo zeroes only solver-owned buffers that persist across steps, leaving the authored joint
        state untouched; otherwise a NaN from one solve persists across resets through the warm start
        (https://github.com/newton-physics/newton/issues/1266). Sparse or disabled imports require an explicit
        synchronization before the next substep can overwrite the authored state. The native CPU backend owns
        one ``MjData`` for template world 0, so other worlds' resets must leave its state and history untouched.
        """
        solver = self.native
        if solver.use_mujoco_cpu and not world_mask.numpy()[0]:
            return
        solver.reset(state, world_mask=world_mask, flags=0)
        if solver.update_data_interval != 1:
            # Newton has no public masked import. With sleeping disabled,
            # reset(flags=0) skips it, so reuse the pinned solver's conversion.
            data = solver.mj_data if solver.use_mujoco_cpu else solver.mjw_data
            solver._update_mjc_data(data, solver.model, state, world_mask=world_mask)

    def post_step(self, state: State) -> None:
        """Reduce persistent solver state validity once after the physics substeps."""
        solver = self.native
        data = solver.mj_data if solver.use_mujoco_cpu else solver.mjw_data
        arrays = (
            data.qpos,
            data.qvel,
            data.qacc,
            data.qfrc_actuator,
            data.qacc_warmstart,
            data.qfrc_applied,
            data.ctrl,
            data.act,
            data.xfrc_applied,
        )
        if solver.use_mujoco_cpu:
            self.reset_required.fill_(any(not np.isfinite(array).all() for array in arrays))
        else:
            wp.launch(
                _detect_solver_reset_required,
                dim=self.reset_required.shape[0],
                inputs=[*arrays],
                outputs=[self.reset_required],
                device=state.body_q.device,
            )

    def check_step(self, captured: bool, debug: bool) -> None:
        """Log MuJoCo solver convergence statistics in debug mode."""
        if not debug:
            return
        solver = self.native
        niter = solver.mjw_data.solver_niter.numpy()
        data = {"max": np.max(niter), "mean": np.mean(niter), "min": np.min(niter), "std": np.std(niter)}
        logger.info(f"Solver convergence data: {data}")
        if np.max(niter) == solver.mjw_model.opt.iterations:
            logger.warning(f"Solver didn't converge! max_iter={np.max(niter)}")

    @classmethod
    def fixed_tendon_control(cls, cfg: MJWarpSolverCfg, articulation, model: Model) -> MjWarpTendonControl | None:
        """Build the MuJoCo tendon adapter, or ``None`` when no MuJoCo actuator drives the articulation's tendons."""
        from .mjwarp_tendon_control import create_mjwarp_tendon_control

        return create_mjwarp_tendon_control(articulation, model)
