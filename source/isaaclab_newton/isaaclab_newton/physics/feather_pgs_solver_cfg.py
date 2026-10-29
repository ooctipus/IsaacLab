# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton's FeatherPGS solver."""

from __future__ import annotations

import math
from typing import Literal

from isaaclab.utils.configclass import configclass

from .newton_manager_cfg import NewtonSolverCfg


@configclass
class FeatherPGSSolverCfg(NewtonSolverCfg):
    """Configuration for Newton's reduced-coordinate FeatherPGS solver.

    FeatherPGS evaluates articulated-body dynamics in generalized coordinates
    and resolves contacts and enabled joint constraints with projected
    Gauss-Seidel iterations. It always uses Newton's collision pipeline.
    """

    class_type: type | str = "{DIR}.feather_pgs_solver:FeatherPGSSolver"

    solver_type: str = "feather_pgs"
    """Solver type metadata."""

    convex_collision: Literal["auto", "ordinary", "rejection", "convex_cells"] = "auto"
    """Experimental convex collision implementation.

    ``auto`` selects ordinary queries, rejection screening or convex cells from predicted cost;
    unsupported pipelines use ordinary queries. Other values force the named method
    or raise an error. Overrides legacy coherent-convex/cells environment variables.
    """

    contact_export: Literal["auto", "tiled", "compact"] = "auto"
    """Experimental reduced-contact export mapping.

    ``auto`` uses compact workers for supported CUDA mesh/heightfield reducers;
    other pipelines retain tiled export. ``compact`` requires admission and
    ``tiled`` forces the original mapping. Overrides ``NEWTON_REDUCER_COMPACT_EXPORT``.
    Contact selection and reduction policy are unchanged.
    """

    sparse_mass_matrix: Literal["auto", "on", "off"] = "auto"
    """Experimental native sparse factor selection.

    ``auto`` uses supported topology and device resources. ``on`` requires that path;
    ``off`` keeps the ordinary representation. Research packed-factor paths are separate.
    Supported split sweeps, velocity limits and grouped dynamics compose automatically.
    Legacy sparse-composition opt-outs apply to ``auto``; ``on`` overrides those optional gates.
    Explicit dense kernel choices disable automatic native sparse factors and conflict with ``on``.
    """

    angular_damping: float = 0.05
    """Angular velocity damping rate [s^-1]."""

    cached_dynamics: Literal["auto", "on", "off"] = "auto"
    """Experimental paired cached prediction and next-state publication.

    ``auto`` selects an admitted cache for the topology and existing factor/row
    policy. ``on`` requires admission; ``off`` uses ordinary prediction/publication.
    Explicit choices override legacy cache environment settings. ``auto`` honors
    those settings when present. Mixed held factors require an explicit opt-in.
    Explicit dense kernels disable automatic caches and conflict with ``on``.
    This choice preserves factor refresh intervals and numerical budgets.
    """

    update_mass_matrix_interval: int = 1
    """How often to update the mass matrix, in simulation steps."""

    contact_friction_gap_threshold: float = math.inf
    """Gap threshold [m] for enabling tangential contact friction rows."""

    contact_friction_position_iterations: int = -1
    """Number of position iterations used for contact-friction anchoring."""

    contact_friction_shared_anchor: bool = False
    """Whether contact-friction rows use shared anchors."""

    contact_friction_anchor_limit: int = 0
    """Maximum stored contact-friction anchors per world."""

    contact_friction_scale: float = 1.0
    """Dimensionless scale factor applied to contact friction limits."""

    contact_shared_anchor: bool = False
    """Whether normal contact rows use shared anchors."""

    enable_contact_friction: bool = True
    """Whether to enable Coulomb contact friction in PGS."""

    enable_joint_limits: bool = False
    """Whether to enforce joint position limits as unilateral PGS constraints."""

    joint_limit_activation_gap: float = 0.20
    """Distance [m or rad, depending on joint type] from a joint limit at which rows activate.

    A finite gap allocates a row only when the joint is within that distance of the corresponding limit.
    Setting this to ``math.inf`` allocates lower and upper rows for every finite limit on every step.
    """

    enable_joint_velocity_limits: bool = False
    """Whether to enforce per-DOF joint velocity limits as PGS constraints.

    This option requires :attr:`pgs_mode` to be ``"matrix_free"``.
    """

    velocity_limit_activation_fraction: float = 0.7
    """Dimensionless fraction of each joint velocity limit at which velocity-limit rows activate.

    The default allocates rows once a finite limit reaches 70%, leaving headroom for the
    one-step activation delay. Set this to ``0.0`` to allocate rows for every limited degree
    of freedom on every step.
    """

    fuse_joint_velocity_limits: bool = True
    """Whether to fuse joint velocity-limit rows into the matrix-free constraint solve."""

    mf_gs_incremental_rows: int = 0
    """Row budget of the incremental-residual matrix-free Gauss-Seidel kernel.

    Worlds with only dense articulated rows and at most this many rows solve through a
    shared-memory response block instead of re-evaluating the row velocity per visit.
    Set to ``0`` to disable the kernel and use the legacy per-row path everywhere.
    """

    mf_gs_parallel_rows: int = 0
    """Row budget of the scaled parallel projection sweep (0 disables it).

    Worlds with only dense articulated rows and at most this many rows are solved with all rows updated in
    parallel (per-row step scaling, Nesterov momentum) instead of the row-serial kernels. Takes precedence over
    :attr:`mf_gs_incremental_rows`.

    This experimental projection does not preserve the legacy solver's converged velocities. Keep this
    setting at ``0`` when legacy solver semantics are required; matching contact statistics is insufficient.
    """

    mf_gs_parallel_sweeps: int = 24
    """Maximum parallel sweeps per launch; the kernel exits early once no impulse changes."""

    mf_gs_parallel_nesterov: bool = True
    """Whether the parallel sweep uses Nesterov momentum."""

    pgs_iterations: int = 8
    """Number of Gauss-Seidel iterations per simulation step."""

    pgs_velocity_iterations: int = 0
    """Number of velocity-level Gauss-Seidel iterations per simulation step."""

    pgs_beta: float = 0.05
    """Dimensionless ERP-style position correction factor."""

    pgs_cfm: float = 1.0e-6
    """Regularization added to Newton's impulse-space Delassus diagonal.

    Units follow the constraint row: [kg^-1] for contact and prismatic rows, and [(kg·m²)^-1] for revolute and angular
    rows.
    """

    pgs_omega: float = 1.0
    """Dimensionless successive over-relaxation factor for PGS."""

    pgs_velocity_drive_mode: Literal["active", "freeze"] = "freeze"
    """Drive-row handling during velocity-only PGS: ``"active"`` updates rows and ``"freeze"`` holds impulses."""

    dense_max_constraints: int = 32
    """Maximum articulation-contact constraints per world."""

    pgs_warmstart: bool = False
    """Whether to warm-start impulses from the previous frame."""

    mf_warmstart: bool = False
    """Whether to reuse matched matrix-free contact impulses from the previous frame."""

    mf_warmstart_decay: float = 1.0
    """Scale applied to matched matrix-free contact impulses before reuse."""

    pgs_mode: Literal["dense", "split", "matrix_free"] = "split"
    """Constraint solve layout.

    ``"dense"`` builds the full Delassus matrix, ``"split"`` uses dense articulated rows and matrix-free free-body
    rows, and ``"matrix_free"`` avoids constructing the dense matrix.
    """

    articulated_contact_response: Literal["immediate", "propagation", "propagation-fused", "propagation-colored"] = (
        "immediate"
    )
    """Response algorithm for matrix-free contacts on articulated bodies."""

    propagation_same_articulation_rows: bool = False
    """Whether propagation batches rows that belong to the same articulation."""

    propagation_cached_response: bool = True
    """Whether propagation reuses cached articulated-body responses."""

    propagation_cached_response_max_bodies: int = 8
    """Maximum body count for cached articulated-body responses."""

    pgs_schedule: Literal["interleaved", "contact_then_internal", "physx_grasp"] = "interleaved"
    """Ordering schedule for matrix-free contact and internal rows.

    Values other than ``"interleaved"`` require :attr:`pgs_mode` to be ``"matrix_free"``.
    """

    mf_max_constraints: int = 512
    """Maximum matrix-free constraints per world."""

    cholesky_kernel: Literal["tiled", "loop", "auto"] = "auto"
    """Cholesky kernel: ``"tiled"``, ``"loop"``, or ``"auto"``."""

    trisolve_kernel: Literal["tiled", "loop", "auto"] = "auto"
    """Tri-solve kernel: ``"tiled"``, ``"loop"``, or ``"auto"``."""

    hinv_jt_kernel: Literal["tiled", "par_row", "auto"] = "auto"
    """H^-1 J^T kernel: ``"tiled"``, ``"par_row"``, or ``"auto"``."""

    delassus_kernel: Literal["tiled", "par_row_col", "auto"] = "auto"
    """Delassus accumulation kernel: ``"tiled"``, ``"par_row_col"``, or ``"auto"``."""

    pgs_kernel: Literal["auto", "loop", "tiled_row", "tiled_contact", "streaming"] = "auto"
    """Dense PGS kernel. ``auto`` uses ``loop`` on CPU and ``tiled_row`` on CUDA.

    Explicit choices require a supported device and split mode. Matrix-free mode
    owns its sweep kernel and requires ``auto`` here.
    """

    pgs_chunk_size: int | None = None
    """Chunk size for streaming PGS kernels. ``None`` selects default behavior."""

    use_parallel_streams: bool = True
    """Whether to dispatch size groups on separate CUDA streams."""

    double_buffer: bool = True
    """Whether to use double buffering for solver internals."""

    nvtx: bool = False
    """Whether to enable NVTX profiling ranges."""

    pgs_debug: bool = False
    """Whether to enable PGS debug diagnostics."""

    drive_mode: Literal["augmented", "physx_pgs"] = "augmented"
    """Joint drive model.

    ``"augmented"`` uses implicit PD, while ``"physx_pgs"`` uses PhysX-style rows and requires
    :attr:`pgs_mode` to be ``"matrix_free"``.
    """

    serial_kernel_block_dim: int = 256
    """Block dimension for serial matrix-free kernels."""

    tile_threads: int = 64
    """Thread count for tiled kernels."""

    row_watermark: bool = False
    """Whether to enable row high-water watermark diagnostics/allocation path."""

    restitution_velocity_threshold: float = 0.5
    """Relative normal velocity threshold below which restitution is disabled [m/s]."""
