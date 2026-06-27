# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from isaaclab.physics import PhysicsCfg
from isaaclab.utils.configclass import configclass

from .newton_collision_cfg import NewtonCollisionPipelineCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager

logger = logging.getLogger(__name__)


@configclass
class NewtonSolverCfg:
    """Configuration for Newton solver-related parameters.

    These parameters are used to configure the Newton solver. For more information, see the `Newton documentation`_.

    Subclasses set :attr:`class_type` to their matching :class:`NewtonManager`
    subclass; :class:`NewtonCfg` propagates that to its own
    :attr:`NewtonCfg.class_type` in :meth:`NewtonCfg.__post_init__` so that
    ``SimulationContext`` resolves the correct manager via the existing
    dispatch path.

    .. _Newton documentation: https://newton.readthedocs.io/en/latest/
    """

    class_type: type[NewtonManager] | str = "{DIR}.newton_manager:NewtonManager"
    """Manager class for this solver.

    Default points at the abstract :class:`NewtonManager`; concrete subclasses
    override it.
    """

    solver_type: str = "None"
    """Solver type metadata (deprecated).

    .. deprecated::
        Manager dispatch is now driven by :attr:`class_type`; this field is
        retained as metadata for logging and debugging only.  Do not branch on
        ``solver_type`` in new code.
    """


@configclass
class NewtonShapeCfg:
    """Default per-shape collision properties applied to all shapes in a Newton scene.

    Mirrors Newton's :attr:`ModelBuilder.default_shape_cfg`. Only fields Isaac
    Lab actually overrides are declared here; unspecified fields keep Newton's
    upstream default. The struct is forwarded onto Newton's upstream
    ``ShapeConfig`` via :func:`~isaaclab.utils.checked_apply` at builder
    construction.
    """

    margin: float = 0.0
    """Default per-shape collision margin [m].

    A nonzero margin (e.g. ``0.01``) is required for stable contact on
    triangle-mesh terrain — without it, lightweight robots fail to learn
    rough-terrain locomotion on Newton. Newton's upstream default is ``0.0``.
    """

    gap: float = 0.01
    """Default per-shape contact gap [m]. Newton's upstream default is ``None``."""

    ke: float = 2.5e3
    """Default per-shape contact elastic stiffness [N/m].

    Governs how hard a contact resists penetration; consumed by the SemiImplicit,
    Featherstone, and MuJoCo (Newton) solver paths. Newton's upstream default is
    ``2.5e3``. Raise it (e.g. ``1e7``) for stiff, low-penetration contacts in
    contact-rich assembly tasks.
    """

    kd: float = 100.0
    """Default per-shape contact damping coefficient.

    Damps the normal contact response. Newton's upstream default is ``100.0``.
    """


@configclass
class ReplayBufferCfg:
    """Configuration for the opt-in one-step replay buffer."""

    enabled: bool = False
    """Whether to record replay transitions."""

    export_envs_only: bool = True
    """When True, export replay state only for the NaN envs."""

    record_state: bool = True
    """Record pre/post Newton state arrays."""

    record_control: bool = True
    """Record control inputs such as ``control.joint_f``."""

    record_solver: bool = True
    """Record selected MJWarp solver vectors."""

    record_contacts: bool = False
    """Record contact data in the hot path. Expensive; normally use cold recompute/export."""

    record_mjwarp_context: bool = False
    """Record full MJWarp solver context. Very expensive; use only for focused diagnosis."""


@configclass
class NanReplayCfg:
    """Configuration for the NaN replay debug buffer.

    When attached to :class:`NewtonCfg`, a rolling buffer of GPU state snapshots
    is kept.  If a NaN is detected after a physics step the buffer is exported to
    disk (with optional USD scene export) and simulation is halted.
    """

    buffer_size: int = 200
    """Number of state snapshots to keep in the rolling buffer.

    Capped at 2000 to bound GPU memory use.
    """

    export_path: str = "./nan_debug/"
    """Directory for exported ``.npz`` and ``.usd`` files."""

    export_envs_only: bool = True
    """When True and simulation has multiple envs, export only the env(s)
    that contain NaN.

    This keeps the replay file small and replayable with a single-env scene.
    When False, the full state (all envs) is exported.
    """

    max_exports: int = 5
    """Maximum number of NaN export events before halting simulation.

    Each export event covers a distinct set of newly-NaN env_ids.  After this
    many exports the debug buffer stops recording and the physics step raises
    ``RuntimeError``.
    """

    per_substep: bool = False
    """When True, NaN detection runs after every solver substep (not just once per
    env-step), capturing the exact substep where the NaN is *born* together with the
    last finite pre-substep state and the solver internals at that substep.

    This requires the CUDA graph to be disabled (host-side checks cannot run inside a
    captured graph), so set :attr:`NewtonCfg.use_cuda_graph` to ``False`` alongside it.
    Slower, but it isolates the failing substep before the NaN propagates across the
    state — use for deep diagnosis, not production.
    """

    replay: ReplayBufferCfg = ReplayBufferCfg()
    """Optional one-step replay recorder configuration."""


@configclass
class NewtonCfg(PhysicsCfg):
    """Configuration for Newton physics manager.

    This configuration includes Newton-specific simulation settings and solver configuration.

    The active :class:`NewtonManager` subclass is determined by
    :attr:`solver_cfg.class_type`, which :meth:`__post_init__` propagates to
    :attr:`class_type` so that ``SimulationContext`` resolves the right
    manager subclass automatically.  User code keeps the existing two-level
    shape ``NewtonCfg(solver_cfg=...)`` and does not need to set
    :attr:`class_type` explicitly.
    """

    class_type: type[NewtonManager] | str | None = None
    """The class type of the :class:`NewtonManager`.

    Auto-set in :meth:`__post_init__` from :attr:`solver_cfg.class_type`.
    Users normally do not set this directly.
    """

    num_substeps: int = 1
    """Number of substeps to use for the solver."""

    collision_decimation: int = 0
    """Re-collide every N solver substeps within a physics tick (``0`` = once per tick)."""

    debug_mode: bool = False
    """Whether to enable debug mode for the solver."""

    use_cuda_graph: bool = True
    """Whether to use CUDA graphing when simulating.

    If set to False, the simulation performance will be severely degraded.
    """

    solver_cfg: NewtonSolverCfg | None = None
    """Solver configuration. If None (default), MJWarpSolverCfg is used by default."""

    collision_cfg: NewtonCollisionPipelineCfg | None = None
    """Newton collision pipeline configuration.

    Controls how Newton's :class:`CollisionPipeline` is configured when it is active.
    The pipeline is active when the solver delegates collision detection to Newton:

    - :class:`MJWarpSolverCfg` with ``use_mujoco_contacts=False``,
    - :class:`KaminoSolverCfg` with ``use_collision_detector=False``,
    - :class:`XPBDSolverCfg` (always),
    - :class:`FeatherstoneSolverCfg` (always).

    :class:`~isaaclab_newton.physics.MPMSolverCfg` does not use this pipeline;
    implicit MPM treats rigid geometry as colliders internally.

    If ``None`` (default), a pipeline with ``broad_phase="explicit"`` is created
    automatically.  Set this to a :class:`NewtonCollisionPipelineCfg` to customize
    parameters such as broad phase algorithm, contact limits, or hydroelastic mode.

    .. note::
        Setting this while ``MJWarpSolverCfg.use_mujoco_contacts=True`` raises
        :class:`ValueError`.  When ``KaminoSolverCfg.use_collision_detector=True``,
        the field is ignored because Kamino's internal detector handles contacts.
    """

    default_shape_cfg: NewtonShapeCfg = NewtonShapeCfg()
    """Default per-shape collision properties applied to every shape in the scene.

    Forwarded to Newton's :attr:`ModelBuilder.default_shape_cfg` at builder
    construction via :func:`~isaaclab.utils.checked_apply`. See
    :class:`NewtonShapeCfg` for the declared fields.
    """

    simplify_meshes: bool = True
    """Whether Newton replication simplifies mesh colliders to convex hulls.

    Keep this enabled for most rigid-body scenes. Disable it when exact triangle
    meshes are intentional, for example thin or hollow MPM colliders.
    """

    nan_replay: NanReplayCfg | None = None
    """NaN replay debug buffer configuration.

    Set to a :class:`NanReplayCfg` instance to enable the rolling state buffer
    and automatic NaN detection / export.  ``None`` (default) disables the
    debug buffer entirely (zero overhead).
    """

    def __post_init__(self):
        # NewtonCfg.class_type is auto-derived from solver_cfg.class_type.
        # Refuse a user-set value: setting both is ambiguous and was
        # previously silently overwritten.
        if self.class_type is not None:
            raise TypeError("Cannot manually set NewtonCfg.class_type; it is auto-derived from solver_cfg.class_type.")
        if self.solver_cfg is None:
            from .mjwarp_manager_cfg import MJWarpSolverCfg

            self.solver_cfg = MJWarpSolverCfg()
        self.class_type = self.solver_cfg.class_type

        # Mid-tick re-collide is silently disabled when collision_decimation >= num_substeps.
        if self.collision_decimation > 0 and self.collision_decimation >= self.num_substeps:
            logger.warning(
                "NewtonCfg.collision_decimation=%d is >= num_substeps=%d; mid-tick re-collide is disabled. "
                "Set 0 < collision_decimation < num_substeps to enable.",
                self.collision_decimation,
                self.num_substeps,
            )
