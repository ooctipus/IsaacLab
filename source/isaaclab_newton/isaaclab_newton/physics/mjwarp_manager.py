# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp Newton manager."""

from __future__ import annotations

import logging

import numpy as np
import torch
import warp as wp
from mujoco_warp._src.types import vec5
from newton import Contacts, Model
from newton.solvers import SolverMuJoCo

from isaaclab.physics import PhysicsManager

from .mesh_variants import build_mesh_variant_sets
from .mjwarp_manager_cfg import MJWarpSolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)


def _contact_force_fn():
    from mujoco_warp._src.support import contact_force_fn

    return contact_force_fn


@wp.kernel(enable_backward=False)
def _copy_external_contact_forces(
    contact_count: wp.array(dtype=wp.int32),
    contact_to_mujoco: wp.array(dtype=wp.int32),
    cone: int,
    frame: wp.array(dtype=wp.mat33f),
    friction: wp.array(dtype=vec5),
    dimension: wp.array(dtype=wp.int32),
    constraint_address: wp.array2d(dtype=wp.int32),
    world: wp.array(dtype=wp.int32),
    adhesion: wp.array(dtype=wp.float32),
    constraint_force: wp.array2d(dtype=wp.float32),
    constraints_per_world: int,
    mujoco_contact_count: wp.array(dtype=wp.int32),
    force: wp.array(dtype=wp.spatial_vector),
):
    contact = wp.tid()
    force[contact] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if contact >= contact_count[0]:
        return

    mujoco_contact = contact_to_mujoco[contact]
    if mujoco_contact < 0 or mujoco_contact >= mujoco_contact_count[0]:
        return

    force[contact] = -wp.static(_contact_force_fn())(
        cone,
        frame,
        friction,
        dimension,
        constraint_address,
        adhesion,
        constraint_force,
        constraints_per_world,
        mujoco_contact_count,
        world[mujoco_contact],
        mujoco_contact,
        True,
    )


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


class NewtonMJWarpManager(NewtonManager):
    """:class:`NewtonManager` specialization for the MuJoCo Warp solver.

    Owns construction of :class:`SolverMuJoCo`, contact-buffer allocation in
    both internal-MuJoCo and Newton-pipeline contact modes, and the debug
    convergence logging emitted from :meth:`_log_solver_debug` when
    :attr:`NewtonCfg.debug_mode` is enabled.
    """

    _supports_mesh_variants = True

    @classmethod
    def _create_solver(
        cls,
        model: Model,
        solver_cfg: MJWarpSolverCfg,
        *,
        mesh_variant_sets: tuple[SolverMuJoCo.MeshVariantSet, ...] = (),
    ) -> SolverMuJoCo:
        """Construct the configured MuJoCo Warp solver."""
        kwargs = cls._filter_solver_kwargs(SolverMuJoCo, solver_cfg)
        # ls_parallel is deprecated in newton; forwarding it (even as False) emits a warning.
        kwargs.pop("ls_parallel", None)
        if mesh_variant_sets:
            kwargs["mesh_variant_sets"] = mesh_variant_sets
        return SolverMuJoCo(model, **kwargs)

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: MJWarpSolverCfg) -> None:
        """Construct :class:`SolverMuJoCo` and populate the base-class slots.

        Filters cfg fields against the solver's ``__init__`` signature so
        non-constructor metadata (``solver_type``, ``class_type``) and the
        ignored deprecated ``ls_parallel`` field are not forwarded. Sets
        :attr:`NewtonManager._needs_collision_pipeline` to
        ``True`` only when ``use_mujoco_contacts=False``.
        """
        sim = PhysicsManager._sim
        clone_plan = sim.get_clone_plan() if sim is not None else None
        mesh_variant_sets, render_sets = build_mesh_variant_sets(
            NewtonManager._mesh_variant_cfgs,
            model,
            NewtonManager._cl_protos,
            clone_plan,
            NewtonManager._cl_visual_protos,
            NewtonManager._mesh_variant_resource_shapes,
        )
        NewtonManager._cl_visual_protos = {}
        NewtonManager._mesh_variant_render_sets = render_sets
        NewtonManager._solver = cls._create_solver(model, solver_cfg, mesh_variant_sets=mesh_variant_sets)
        NewtonManager._use_single_state = True
        NewtonManager._needs_collision_pipeline = not solver_cfg.use_mujoco_contacts
        NewtonManager._supports_rigid_body_force_input = True
        NewtonManager._solver_reset_required = wp.zeros(model.world_count, dtype=wp.bool, device=model.device)
        NewtonManager._solver_reset_required_torch = wp.to_torch(NewtonManager._solver_reset_required)

        cfg = PhysicsManager._cfg
        # Cross-config validation that needs both halves.
        if solver_cfg.use_mujoco_contacts and cfg.collision_cfg is not None:
            raise ValueError(
                "NewtonCfg: collision_cfg cannot be set when "
                "solver_cfg.use_mujoco_contacts=True. Either set "
                "use_mujoco_contacts=False or remove collision_cfg."
            )

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
            solver = NewtonManager._solver
            if solver is not None and NewtonManager._collision_pipeline is not None:
                sleep_filter = solver.collision_sleep_filter
                if sleep_filter is not None:
                    NewtonManager._collision_pipeline.configure_sleep_filter(*sleep_filter)
            return
        if cls._solver is not None:
            NewtonManager._contacts = Contacts(
                rigid_contact_max=cls._solver.get_max_contact_count(),
                soft_contact_max=0,
                device=PhysicsManager._device,
                requested_attributes=cls._model.get_requested_contact_attributes(),
            )

    @classmethod
    def _update_contacts_for_sensors(cls, contacts: Contacts) -> None:
        if not cls._needs_collision_pipeline:
            super()._update_contacts_for_sensors(contacts)
            return

        solver = cls._solver
        contact_to_mujoco = solver._contact_tid_to_cid
        if contacts.force is None or contact_to_mujoco is None:
            raise RuntimeError("MJWarp contact reporting was not initialized.")
        data = solver.mjw_data
        wp.launch(
            _copy_external_contact_forces,
            dim=contact_to_mujoco.shape[0],
            inputs=[
                contacts.rigid_contact_count,
                contact_to_mujoco,
                solver.mjw_model.opt.cone,
                data.contact.frame,
                data.contact.friction,
                data.contact.dim,
                data.contact.efc_address,
                data.contact.worldid,
                data.contact.adhesion,
                data.efc.force,
                data.njmax,
                data.nacon,
            ],
            outputs=[contacts.force],
            device=cls._model.device,
        )

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array | None) -> None:
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
        if cls._solver.use_mujoco_cpu and not world_mask.numpy().any():
            return
        # flags=0 skips the joint-state reset to model defaults: IsaacLab owns
        # joint_q/joint_qd and has already written the authored reset pose.
        cls._solver.reset(cls._state_0, world_mask=world_mask, flags=0)

    @classmethod
    def _update_solver_reset_required(cls) -> None:
        """Reduce MuJoCo state validity to one cached flag per world.

        The GPU path is a single graph-safe Warp launch. The manager invokes it
        once after all solver substeps and decimation iterations, so MDP terms
        only read the resulting mask.
        """
        reset_required = NewtonManager._solver_reset_required
        if reset_required is None:
            return

        solver = cls._solver
        if solver.use_mujoco_cpu:
            data = solver.mj_data
            if data is None:
                reset_required.zero_()
                return
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
            reset_required.fill_(any(not np.isfinite(array).all() for array in arrays))
            return

        data = solver.mjw_data
        if data is None:
            reset_required.zero_()
            return
        wp.launch(
            _detect_solver_reset_required,
            dim=reset_required.shape[0],
            inputs=[
                data.qpos,
                data.qvel,
                data.qacc,
                data.qfrc_actuator,
                data.qacc_warmstart,
                data.qfrc_applied,
                data.ctrl,
                data.act,
                data.xfrc_applied,
            ],
            outputs=[reset_required],
            device=reset_required.device,
        )

    @classmethod
    def _get_nonfinite_diagnostic_tensors(cls) -> dict[str, torch.Tensor]:
        """Return zero-copy views of MJWarp state inspected after a failed transition."""
        solver = cls._solver
        data = solver.mj_data if solver.use_mujoco_cpu else solver.mjw_data
        if data is None:
            return {}

        tensors = {}
        for name in (
            "qpos",
            "qvel",
            "qacc",
            "qfrc_actuator",
            "qacc_warmstart",
            "qfrc_applied",
            "ctrl",
            "act",
            "xfrc_applied",
            "xpos",
            "xquat",
            "cvel",
        ):
            value = getattr(data, name, None)
            if value is not None:
                tensors[name] = torch.from_numpy(value) if isinstance(value, np.ndarray) else wp.to_torch(value)
        return tensors

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
