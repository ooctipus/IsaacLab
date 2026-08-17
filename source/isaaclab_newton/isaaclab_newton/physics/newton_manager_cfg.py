# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from isaaclab.physics import PhysicsCfg
from isaaclab.utils.configclass import configclass

from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg

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
class NewtonSoftContactCfg:
    """Global soft-contact parameters applied to the finalized Newton model."""

    soft_contact_ke: float = 1.0e3
    """Body-particle and particle self-contact stiffness [N/m].

    Effective body-particle stiffness is ``0.5 * (soft_contact_ke + shape_ke)``,
    where ``shape_ke`` is the rigid shape's material stiffness.
    """

    soft_contact_kd: float = 10.0
    """Body-particle contact damping [N*s/m]."""

    soft_contact_mu: float = 0.5
    """Body-particle contact friction coefficient [dimensionless].

    Effective body-particle friction is ``sqrt(soft_contact_mu * shape_mu)``,
    where ``shape_mu`` is the rigid shape's material friction coefficient.
    """


@configclass
class NewtonShapeCfg:
    """Default per-shape collision properties applied to all shapes in a Newton scene.

    Mirrors Newton's :attr:`ModelBuilder.default_shape_cfg`. Fields that Isaac
    Lab overrides or exposes for user overrides are declared here; fields not
    represented keep Newton's upstream defaults. The struct is forwarded onto
    Newton's upstream ``ShapeConfig`` via
    :func:`~isaaclab.utils.checked_apply` at builder construction.
    """

    margin: float = 0.0
    """Default per-shape collision margin [m].

    A nonzero margin (e.g. ``0.01``) is required for stable contact on
    triangle-mesh terrain — without it, lightweight robots fail to learn
    rough-terrain locomotion on Newton. Newton's upstream default is ``0.0``.
    """

    gap: float = 0.01
    """Default per-shape contact gap [m]. Newton's upstream default is ``None``."""

    # Defaults mirror Newton's ShapeConfig defaults so an unspecified field is a no-op.
    ke: float = 2.5e3
    """Default per-shape normal contact stiffness [N/m].

    Applied to shapes that lack an explicit material; per-asset materials
    override it. Mirrors Newton's ``ShapeConfig.ke`` default.
    """

    kd: float = 100.0
    """Default per-shape normal contact damping [N*s/m].

    Applied to shapes that lack an explicit material; per-asset materials
    override it. Mirrors Newton's ``ShapeConfig.kd`` default.
    """

    mu: float = 1.0
    """Default per-shape friction coefficient [dimensionless].

    Applied to shapes that lack an explicit material; per-asset materials
    override it. Mirrors Newton's ``ShapeConfig.mu`` default.
    """

@configclass
class NewtonDebugReplayCfg:
    """Configuration for opt-in transition replay capture."""

    enabled: bool = False
    """Whether to record replay transitions."""

    record_state: bool = True
    """Whether to record pre/post Newton state arrays."""

    record_control: bool = True
    """Whether to record control inputs such as ``control.joint_f``."""

    record_solver: bool = True
    """Whether to record the active solver provider."""

    record_contacts: bool = False
    """Whether to record live contact data for each transition."""

    record_collision_pipeline: bool = False
    """Whether to record the external collision pipeline for each transition.

    This retains rolling collision internals in both replay pre/post snapshots
    and can materially increase memory use. The finalized external Newton
    collision pipeline is a required load-time provider when enabled.
    """

    record_operations: bool = False
    """Whether to record solver-specific transient operations for exact replay.

    This capability requires a compatible operation provider. Physics
    initialization fails when the active solver cannot provide the requested
    transient operation snapshots.
    """

    include_fields: tuple[str, ...] = ("*",)
    """Glob patterns selecting fields from the discovered replay schema."""

    exclude_fields: tuple[str, ...] = ()
    """Glob patterns excluding fields from the discovered replay schema."""


@configclass
class NewtonDebugCaptureCfg:
    """Configuration for strict Newton physics incident capture.

    When attached to :class:`NewtonCfg`, a rolling history of physics state is
    retained. Non-finite state produces versioned incident artifacts containing
    the complete discovered provider schema and optional transition replay.
    """

    history_length: int = 200
    """Number of state snapshots retained in chronological history."""

    output_dir: str = "./physics_debug/"
    """Directory for incident ``.npz`` and optional ``.usd`` artifacts."""

    record_scene: bool = False
    """Whether to export optional static full-stage USD geometry and context.

    Scene export requires an available USD stage and clone plan at incident time.
    The USD is not world-focused or stamped with post-step Newton transforms;
    captured arrays remain authoritative for failing dynamic state.
    """

    failed_worlds_only: bool = True
    """Whether world-scoped artifacts contain only the failed world.

    Global entities required by the failed world remain included. Set this to
    ``False`` to retain the full multi-world state in every artifact.
    """

    max_incidents: int = 5
    """Maximum number of distinct incident events to export.

    One event can produce multiple artifacts when several worlds fail in the
    same step. This limit controls retention only; stopping is controlled
    independently by :attr:`halt_on_incident`.
    """

    halt_on_incident: bool = True
    """Whether to halt simulation immediately after the first incident.

    Continuing to step a non-finite physics state is unsafe, so this defaults
    to ``True`` independently of :attr:`max_incidents`.
    """

    fail_on_capture_error: bool = True
    """Whether a runtime capture failure raises after saving a partial artifact.

    Required provider and schema failures always fail during physics
    initialization. This option controls cold-path failures such as scene export
    after the state history has already been preserved as a partial artifact.
    """

    max_gpu_bytes: int = 2 * 1024**3
    """Maximum total GPU memory allocated by state and replay histories.

    Physics initialization fails with a field-by-field report before an
    allocation would exceed this budget. Fields are never silently dropped.
    """

    record_model: bool = True
    """Whether to record discovered Newton model arrays and metadata."""

    record_control: bool = True
    """Whether to record the live applied Newton control in each cold incident."""

    record_contacts: bool = False
    """Whether to record active Newton contact data on an incident."""

    record_collision_pipeline: bool = False
    """Whether to record the finalized external Newton collision pipeline.

    When enabled, the external collision pipeline is a required load-time
    provider. Solvers that perform collision detection internally should record
    their collision state through :attr:`record_solver` instead.
    """

    record_solver: bool = True
    """Whether to record the active solver's discovered arrays and metadata."""

    record_operations: bool = False
    """Whether to retain the latest solver-specific transient operations.

    This is independent of transition replay. Enabling it requires a compatible
    operation provider and active solver, and makes operation values available
    to incident archives and automatic non-finite detection.
    """

    include_private_fields: bool = True
    """Whether to include private fields from provider instance storage.

    Comprehensive discovery reads private ``vars`` and declared slots when this
    option is enabled. It never invokes properties or descends into opaque
    unregistered runtime resources.
    """

    readback_preflight: bool = False
    """Whether to synchronously read each selected field during initialization.

    The recorder logs the field path before each read. If a native backend
    terminates the process while accessing an invalid array, the last emitted
    path identifies the failing field. Disable this expensive diagnostic after
    narrowing the capture selection.
    """

    detect_nonfinite_in: tuple[str, ...] = ("state",)
    """Incident providers scanned for NaN and infinity values.

    An empty tuple disables automatic scans for trigger-only capture.
    Allowed names are ``"state"``, ``"model"``, ``"control"``, ``"contacts"``,
    ``"solver"``, ``"collision_pipeline"``, ``"context"``, and
    ``"operations"``. Recorded retained providers must also have their
    corresponding ``record_*`` option enabled. Context detection scans
    registered workflow values and requires a capturable context provider when
    the recorder binds. Solver detection includes retained Hessian,
    factorization, and constraint arrays. Transient operation detection requires
    :attr:`record_operations` and is independent of transition replay.
    """

    detect_nonfinite_include_fields: tuple[str, ...] = ("*",)
    """Glob patterns selecting recorded fields for automatic non-finite scans.

    These patterns affect detection only; matching values remain governed by
    :attr:`include_fields` and are always archived when recorded. Every pattern
    must match a recorded floating-point or complex field under
    :attr:`detect_nonfinite_in`.
    """

    detect_nonfinite_exclude_fields: tuple[str, ...] = ()
    """Glob patterns excluded from automatic non-finite scans.

    Use this for allocated but mode-inactive buffers whose contents are not
    initialized by the active solver path. Excluded values remain archived.
    """

    include_fields: tuple[str, ...] = ("*",)
    """Glob patterns selecting fields from discovered incident providers."""

    exclude_fields: tuple[str, ...] = ()
    """Glob patterns excluding fields from discovered incident providers."""

    capture_per_substep: bool = False
    """Whether incident detection runs after every solver substep.

    This requires CUDA graph capture to be disabled because host-side checks
    cannot run inside a captured graph. It records the exact failing substep and
    its last finite pre-state for deep diagnosis.
    """

    replay: NewtonDebugReplayCfg = NewtonDebugReplayCfg()
    """Optional transition replay configuration."""


def _validate_bool_fields(owner: object, field_names: tuple[str, ...], prefix: str) -> None:
    """Validate boolean configuration fields."""
    for field_name in field_names:
        if not isinstance(getattr(owner, field_name), bool):
            raise TypeError(f"{prefix}.{field_name} must be a bool.")


def _validate_pattern_fields(
    owner: object,
    prefix: str,
    include_field: str,
    exclude_field: str,
) -> None:
    """Validate one strict include/exclude glob tuple pair."""
    for field_name, allow_empty in ((include_field, False), (exclude_field, True)):
        patterns = getattr(owner, field_name)
        if not isinstance(patterns, tuple):
            raise TypeError(f"{prefix}.{field_name} must be a tuple of strings.")
        if not patterns and not allow_empty:
            raise ValueError(f"{prefix}.{field_name} must not be empty.")
        if any(not isinstance(pattern, str) or not pattern for pattern in patterns):
            raise ValueError(f"{prefix}.{field_name} must contain only non-empty strings.")
        if len(set(patterns)) != len(patterns):
            raise ValueError(f"{prefix}.{field_name} must not contain duplicates.")


def _validate_debug_patterns(owner: object, prefix: str) -> None:
    """Validate strict archive include and exclude glob tuples."""
    _validate_pattern_fields(owner, prefix, "include_fields", "exclude_fields")


def _validate_debug_replay_cfg(replay: object) -> NewtonDebugReplayCfg:
    """Validate transition replay configuration and return its narrowed type."""
    if not isinstance(replay, NewtonDebugReplayCfg):
        raise TypeError("NewtonCfg.debug_capture.replay must be a NewtonDebugReplayCfg instance.")
    _validate_bool_fields(
        replay,
        (
            "enabled",
            "record_state",
            "record_control",
            "record_solver",
            "record_contacts",
            "record_collision_pipeline",
            "record_operations",
        ),
        "NewtonCfg.debug_capture.replay",
    )
    if replay.record_operations and not replay.enabled:
        raise ValueError(
            "NewtonCfg.debug_capture.replay.record_operations=True requires "
            "NewtonCfg.debug_capture.replay.enabled=True."
        )
    if replay.record_collision_pipeline and not replay.enabled:
        raise ValueError(
            "NewtonCfg.debug_capture.replay.record_collision_pipeline=True requires "
            "NewtonCfg.debug_capture.replay.enabled=True."
        )
    _validate_debug_patterns(replay, "NewtonCfg.debug_capture.replay")
    if replay.enabled and not any(
        (
            replay.record_state,
            replay.record_control,
            replay.record_solver,
            replay.record_contacts,
            replay.record_collision_pipeline,
            replay.record_operations,
        )
    ):
        raise ValueError("NewtonCfg.debug_capture.replay must record at least one provider when enabled.")
    return replay


def _validate_detect_nonfinite_in(capture: NewtonDebugCaptureCfg) -> None:
    """Validate provider selection for automatic non-finite incidents."""
    providers = capture.detect_nonfinite_in
    if not isinstance(providers, tuple):
        raise TypeError("NewtonCfg.debug_capture.detect_nonfinite_in must be a tuple of strings.")
    if any(not isinstance(provider, str) or not provider for provider in providers):
        raise ValueError("NewtonCfg.debug_capture.detect_nonfinite_in must contain only non-empty strings.")
    if len(set(providers)) != len(providers):
        raise ValueError("NewtonCfg.debug_capture.detect_nonfinite_in must not contain duplicates.")

    allowed = (
        "state",
        "model",
        "control",
        "contacts",
        "solver",
        "collision_pipeline",
        "context",
        "operations",
    )
    invalid = [provider for provider in providers if provider not in allowed]
    if invalid:
        raise ValueError(
            "NewtonCfg.debug_capture.detect_nonfinite_in contains unsupported providers "
            f"{invalid}; expected only {list(allowed)}."
        )

    required_record_flags = {
        "model": "record_model",
        "control": "record_control",
        "contacts": "record_contacts",
        "solver": "record_solver",
        "collision_pipeline": "record_collision_pipeline",
        "operations": "record_operations",
    }
    for provider in providers:
        record_flag = required_record_flags.get(provider)
        if record_flag is not None and not getattr(capture, record_flag):
            raise ValueError(
                f"NewtonCfg.debug_capture.detect_nonfinite_in includes {provider!r}, which requires {record_flag}=True."
            )


def _validate_debug_capture_cfg(
    capture: NewtonDebugCaptureCfg | None,
    use_cuda_graph: bool,
) -> None:
    """Validate strict incident capture configuration."""
    if capture is None:
        return
    if not isinstance(capture, NewtonDebugCaptureCfg):
        raise TypeError("NewtonCfg.debug_capture must be a NewtonDebugCaptureCfg instance or None.")

    for field_name in ("history_length", "max_incidents", "max_gpu_bytes"):
        value = getattr(capture, field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"NewtonCfg.debug_capture.{field_name} must be an int.")
        if value < 1:
            raise ValueError(f"NewtonCfg.debug_capture.{field_name} must be at least 1.")
    if not isinstance(capture.output_dir, str):
        raise TypeError("NewtonCfg.debug_capture.output_dir must be a str.")
    if not capture.output_dir.strip():
        raise ValueError("NewtonCfg.debug_capture.output_dir must not be empty.")

    _validate_bool_fields(
        capture,
        (
            "failed_worlds_only",
            "halt_on_incident",
            "fail_on_capture_error",
            "record_scene",
            "record_model",
            "record_control",
            "record_contacts",
            "record_collision_pipeline",
            "record_solver",
            "record_operations",
            "include_private_fields",
            "readback_preflight",
            "capture_per_substep",
        ),
        "NewtonCfg.debug_capture",
    )
    replay = _validate_debug_replay_cfg(capture.replay)
    _validate_debug_patterns(capture, "NewtonCfg.debug_capture")
    _validate_pattern_fields(
        capture,
        "NewtonCfg.debug_capture",
        "detect_nonfinite_include_fields",
        "detect_nonfinite_exclude_fields",
    )
    _validate_detect_nonfinite_in(capture)
    if use_cuda_graph and (capture.capture_per_substep or replay.enabled or capture.record_operations):
        raise ValueError(
            "Newton debug_capture.capture_per_substep, debug_capture.replay.enabled, and "
            "debug_capture.record_operations are not CUDA graph safe. Set NewtonCfg.use_cuda_graph=False "
            "or disable those debug_capture options."
        )

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

    deterministic_mode: Literal["not_guaranteed", "run_to_run", "gpu_to_gpu"] = "not_guaranteed"
    """Determinism guarantee applied to the Newton solver and collision pipeline.

    The values ``"not_guaranteed"``, ``"run_to_run"``, and ``"gpu_to_gpu"``
    map to the corresponding ``warp.DeterministicMode`` values. Deterministic
    execution increases memory use and can reduce simulation performance.

    .. warning::

       Deterministic contact ordering adds sorting work and allocates buffers
       sized for the configured maximum contact count. Runtime and memory
       overhead therefore grow with contact capacity. Enable this mode only
       when its reproducibility guarantee is required.

    MJWarp on the GPU with
    :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.disable_sensors` set to
    ``True``, XPBD, and Featherstone support this setting. Newton raises an
    error during solver initialization for unsupported solvers rather than
    silently running them without the requested guarantee.
    """

    solver_cfg: NewtonSolverCfg | None = None
    """Solver configuration. If None (default), MJWarpSolverCfg is used by default."""

    soft_contact_cfg: NewtonSoftContactCfg | None = None
    """Global soft-contact parameters applied after model finalization.

    If ``None``, Newton model defaults are preserved.
    """

    collision_cfg: NewtonCollisionPipelineCfg | None = None
    """Newton collision pipeline configuration.

    Controls how Newton's :class:`CollisionPipeline` is configured when it is active.
    The pipeline is active when the solver delegates collision detection to Newton:

    - :class:`MJWarpSolverCfg` with ``use_mujoco_contacts=False``,
    - :class:`KaminoPADMMSolverCfg` or :class:`KaminoDVISolverCfg` with
      ``use_collision_detector=False``,
    - :class:`XPBDSolverCfg` (always),
    - :class:`VBDSolverCfg` (always),
    - :class:`FeatherstoneSolverCfg` (always).

    :class:`~isaaclab_newton.physics.MPMSolverCfg` does not use this pipeline;
    implicit MPM treats rigid geometry as colliders internally.

    If ``None`` (default), a pipeline with ``broad_phase="explicit"`` is created
    automatically.  Set this to a :class:`NewtonCollisionPipelineCfg` to customize
    parameters such as broad phase algorithm, contact limits, or hydroelastic mode.

    .. note::
        Setting this while ``MJWarpSolverCfg.use_mujoco_contacts=True`` raises
        :class:`ValueError`.  When a Kamino solver config has ``use_collision_detector=True``,
        the field is ignored because Kamino's internal detector handles contacts.
    """

    default_shape_cfg: NewtonShapeCfg = NewtonShapeCfg()
    """Default per-shape collision properties applied to every shape in the scene.

    Forwarded to Newton's :attr:`ModelBuilder.default_shape_cfg` at builder
    construction via :func:`~isaaclab.utils.checked_apply`. See
    :class:`NewtonShapeCfg` for the declared fields.
    """

    load_visual_shapes: bool | None = None
    """Whether Newton replication imports visual-only geometry from USD.

    ``None`` imports it only when a viewer, an offscreen ``rgb_array`` capture, or a
    camera sensor is active, so headless training does not pay the USD parse time and
    memory for shapes nothing draws. Set to ``True`` to always import it, which is
    needed when a ray-cast sensor must hit geometry that carries no collider.
    """

    bvh_constructor_geometry: Literal["lbvh", "sah", "cubql"] = "cubql"
    """BVH construction algorithm for mesh geometry colliders.

    Selects the bounding-volume-hierarchy builder Newton uses for the triangle
    meshes of collision geometry, forwarded to :attr:`ModelBuilder.BvhConfig`.
    Trades build time against query (traversal) quality:

    - ``"lbvh"``: linear BVH; fastest to build, lowest-quality tree.
    - ``"sah"``: surface-area-heuristic BVH; slower build, tighter tree with
      faster ray/overlap queries.
    - ``"cubql"``: cuBQL GPU builder; balances fast construction with good tree
      quality on the GPU (default).
    """

    bvh_constructor_scene: Literal["lbvh", "sah"] = "sah"
    """BVH construction algorithm for the top-level scene (broad-phase) hierarchy.

    Selects the builder for the BVH over all colliders used during broad-phase
    culling, forwarded to :attr:`ModelBuilder.BvhConfig`. See
    :attr:`bvh_constructor_geometry` for the ``"lbvh"`` / ``"sah"`` trade-off;
    ``"cubql"`` is not available for the scene hierarchy.
    """

    bvh_constructor_gaussian: Literal["lbvh", "sah", "cubql"] = "cubql"
    """BVH construction algorithm for Gaussian-splat primitives.

    Selects the builder for the BVH over 3D Gaussian primitives (used by the
    Gaussian renderer/collision path), forwarded to
    :attr:`ModelBuilder.BvhConfig`. See :attr:`bvh_constructor_geometry` for the
    ``"lbvh"`` / ``"sah"`` / ``"cubql"`` trade-off.
    """

    debug_capture: NewtonDebugCaptureCfg | None = None
    """Strict physics debug capture configuration.

    Set to :class:`NewtonDebugCaptureCfg` to enable rolling state history,
    non-finite incident artifacts, and optional transition replay. ``None``
    disables capture and its runtime overhead.
    """

    def __post_init__(self):
        # NewtonCfg.class_type is auto-derived from solver_cfg.class_type.
        # Refuse a user-set value: setting both is ambiguous and was
        # previously silently overwritten.
        if self.class_type is not None:
            raise TypeError("Cannot manually set NewtonCfg.class_type; it is auto-derived from solver_cfg.class_type.")
        if self.deterministic_mode not in ("not_guaranteed", "run_to_run", "gpu_to_gpu"):
            raise ValueError(
                "NewtonCfg.deterministic_mode must be 'not_guaranteed', 'run_to_run', or 'gpu_to_gpu', "
                f"got {self.deterministic_mode!r}."
            )
        if self.solver_cfg is None:
            from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg

            self.solver_cfg = MJWarpSolverCfg()

        self.class_type = self.solver_cfg.class_type

        _validate_debug_capture_cfg(self.debug_capture, self.use_cuda_graph)

        # Mid-tick re-collide is silently disabled when collision_decimation >= num_substeps.
        if self.collision_decimation > 0 and self.collision_decimation >= self.num_substeps:
            logger.warning(
                "NewtonCfg.collision_decimation=%d is >= num_substeps=%d; mid-tick re-collide is disabled. "
                "Set 0 < collision_decimation < num_substeps to enable.",
                self.collision_decimation,
                self.num_substeps,
            )
