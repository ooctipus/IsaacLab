# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for per-solver Newton manager instances."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from inspect import signature
from types import SimpleNamespace

import isaaclab_newton.physics.kamino_manager as kamino_manager_module
import isaaclab_newton.physics.newton_manager as newton_manager_module
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.cloner import NewtonReplicateContext
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    HydroelasticSDFCfg,
    KaminoDVICfg,
    KaminoDVISolverCfg,
    KaminoDynamicsCfg,
    KaminoPADMMCfg,
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCollisionPipelineCfg,
    NewtonFeatherstoneManager,
    NewtonKaminoManager,
    NewtonManager,
    NewtonMJWarpManager,
    NewtonMPMManager,
    NewtonShapeCfg,
    NewtonVBDManager,
    NewtonXPBDManager,
    VBDSolverCfg,
    XPBDSolverCfg,
)
from isaaclab_newton.physics.mpm_manager import _make_solver_config
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg, MPMPointsCfg
from isaaclab_newton.sim.spawners.mpm.mpm import emit_mpm_particles
from newton import ShapeFlags
from newton.solvers import SolverFeatherstone, SolverImplicitMPM, SolverKamino, SolverMuJoCo, SolverVBD, SolverXPBD

from isaaclab.cloner import ReplicateSession
from isaaclab.physics import PhysicsEvent
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context

# ---------------------------------------------------------------------------
# Lightweight (no sim) parametrisation
# ---------------------------------------------------------------------------

# (solver_cfg_factory, expected_manager, expected_solver_cls,
#  expected_use_single_state, expected_needs_collision_pipeline)
SOLVER_MATRIX = [
    pytest.param(
        lambda: MJWarpSolverCfg(use_mujoco_contacts=True),
        NewtonMJWarpManager,
        SolverMuJoCo,
        True,
        False,
        id="mjwarp_internal_contacts",
    ),
    pytest.param(
        lambda: MJWarpSolverCfg(use_mujoco_contacts=False),
        NewtonMJWarpManager,
        SolverMuJoCo,
        True,
        True,
        id="mjwarp_newton_pipeline",
    ),
    pytest.param(
        lambda: XPBDSolverCfg(),
        NewtonXPBDManager,
        SolverXPBD,
        False,
        True,
        id="xpbd",
    ),
    pytest.param(
        lambda: VBDSolverCfg(),
        NewtonVBDManager,
        SolverVBD,
        False,
        True,
        id="vbd",
    ),
    pytest.param(
        lambda: FeatherstoneSolverCfg(),
        NewtonFeatherstoneManager,
        SolverFeatherstone,
        False,
        True,
        id="featherstone",
    ),
    pytest.param(
        lambda: KaminoPADMMSolverCfg(use_collision_detector=True),
        NewtonKaminoManager,
        SolverKamino,
        False,
        False,
        id="kamino_internal_contacts",
    ),
    pytest.param(
        lambda: KaminoPADMMSolverCfg(use_collision_detector=False),
        NewtonKaminoManager,
        SolverKamino,
        False,
        True,
        id="kamino_newton_pipeline",
    ),
    pytest.param(
        lambda: MPMSolverCfg(max_iterations=2, voxel_size=0.05),
        NewtonMPMManager,
        SolverImplicitMPM,
        True,
        False,
        id="implicit_mpm",
    ),
]

RIGID_BODY_FORCE_INPUT_SUPPORT = {
    NewtonMJWarpManager: True,
    NewtonVBDManager: True,
    NewtonXPBDManager: True,
    NewtonFeatherstoneManager: True,
    NewtonKaminoManager: True,
    NewtonMPMManager: False,
}

_SIM_CONTEXT = SimpleNamespace(cfg=SimpleNamespace(device="cpu"))


def _manager_for(solver_cfg=None, **kwargs):
    cfg = (solver_cfg or XPBDSolverCfg()).replace(**kwargs)
    manager = cfg.class_type(cfg)
    manager._newton = NewtonReplicateContext(_SIM_CONTEXT)
    manager._newton.bind_physics(cfg, manager._builder_attribute_solvers)
    manager._scene_data_backend = newton_manager_module.NewtonSceneDataBackend(manager)
    return manager


@contextmanager
def _direct_builder_simulation(sim_cfg):
    """Build a simulation whose empty scene has completed the normal clone lifecycle."""
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        with ReplicateSession((), num_clones=1, env_spacing=1.0):
            pass
        yield sim


def _kamino_config(solver_cfg, monkeypatch):
    class _RecordingKamino:
        Config = SolverKamino.Config

        def __init__(self, _model, config):
            self.config = config

    monkeypatch.setattr(kamino_manager_module, "SolverKamino", _RecordingKamino)
    return _manager_for(solver_cfg)._create_solver(object(), solver_cfg).config


def test_newton_resource_requires_its_simulation_owner_at_construction():
    with pytest.raises(TypeError, match="sim_context"):
        NewtonReplicateContext()
    assert "_bind_simulation" not in NewtonReplicateContext.__dict__


@pytest.mark.parametrize("physics_first", [True, False], ids=["physics_first", "scene_first"])
def test_newton_registry_resource_is_independent_of_consumer_order(physics_first):
    """Physics and scene consumers converge on one native resource in either order."""
    stage = object()
    context = object.__new__(SimulationContext)
    context.stage = stage
    context.cfg = SimpleNamespace(device="cpu", gravity=(0.0, 0.0, -9.81))
    context._backend_registry = {}
    context._backend_clone_roles = {}
    context._clone_plan = None
    manager = NewtonXPBDManager(XPBDSolverCfg(default_shape_cfg=NewtonShapeCfg(margin=0.123)))

    def scene_consumer():
        return context.get_or_create_backend(NewtonReplicateContext, context, clone_role="scene")

    if physics_first:
        manager._bind_context(context)
        resource = scene_consumer()
    else:
        resource = scene_consumer()
        manager._bind_context(context)

    assert type(resource) is NewtonReplicateContext
    assert not isinstance(manager, NewtonReplicateContext)
    assert manager._newton is resource
    assert resource._physics_cfg is manager.cfg
    assert resource._sim is context
    assert resource._sim.stage is stage
    assert resource._sim.cfg.device == "cpu"
    assert "stage" not in resource.__dict__
    assert resource.load_visual_shapes is False
    assert "_device" not in resource.__dict__
    assert context._backend_registry == {NewtonReplicateContext: resource}
    assert context._backend_clone_roles == {NewtonReplicateContext: {"physics", "scene"}}
    assert resource.create_builder().default_shape_cfg.margin == pytest.approx(0.123)

    model, state, control = object(), object(), object()
    resource._model, resource._state_0, resource._control = model, state, control
    assert (manager.get_model(), manager.get_state_0(), resource.get_control()) == (model, state, control)


# ---------------------------------------------------------------------------
# class_type wiring (no SimulationContext required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline",
    SOLVER_MATRIX,
)
def test_solver_cfg_class_type_resolves_to_subclass(
    solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline
):
    """Each ``*SolverCfg.class_type`` resolves to its matching manager subclass."""
    solver_cfg = solver_cfg_factory()
    # ``class_type`` is a lazy ``"module:Class"`` reference; calling its
    # ``_resolve()`` returns the actual class. ``__name__`` works without
    # forcing import (LazyType caches metadata) and is sufficient identity.
    assert solver_cfg.class_type.__name__ == expected_manager.__name__


def test_solver_kwargs_include_newton_deterministic_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Solver construction receives the mode on its concrete physics config."""
    manager = _manager_for()
    manager._deterministic_mode = wp.DeterministicMode.GPU_TO_GPU

    kwargs = manager._filter_solver_kwargs(SolverXPBD, XPBDSolverCfg())

    assert kwargs["deterministic"] == wp.DeterministicMode.GPU_TO_GPU


@pytest.mark.parametrize(
    "solver_cfg",
    [
        pytest.param(KaminoPADMMSolverCfg(), id="kamino_padmm"),
        pytest.param(MPMSolverCfg(), id="implicit_mpm"),
        pytest.param(MJWarpSolverCfg(use_mujoco_cpu=True), id="mujoco_cpu"),
        pytest.param(MJWarpSolverCfg(), id="mujoco_warp_sensors"),
    ],
)
def test_deterministic_mode_rejects_unsupported_solver_cfg(solver_cfg) -> None:
    """Unsupported solvers should not silently ignore a determinism guarantee."""
    with pytest.raises(ValueError, match="not supported"):
        _manager_for()._validate_deterministic_solver_cfg(solver_cfg, wp.DeterministicMode.GPU_TO_GPU)


@pytest.mark.parametrize(
    "solver_cfg_cls, solver_cfg_kwargs",
    [
        pytest.param(FeatherstoneSolverCfg, {}, id="featherstone"),
        pytest.param(MJWarpSolverCfg, {"disable_sensors": True}, id="mujoco_warp"),
        pytest.param(XPBDSolverCfg, {}, id="xpbd"),
    ],
)
def test_deterministic_mode_accepts_supported_solver_cfg_subclasses(solver_cfg_cls, solver_cfg_kwargs) -> None:
    """Custom subclasses of supported solver configs should retain deterministic support."""

    class CustomSolverCfg(solver_cfg_cls):
        pass

    _manager_for()._validate_deterministic_solver_cfg(
        CustomSolverCfg(**solver_cfg_kwargs), wp.DeterministicMode.GPU_TO_GPU
    )


def test_deterministic_collision_pipeline_matches_expanded_contact_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deterministic sorting buffers should grow with the solver contact buffer."""
    pipeline_calls: list[dict] = []

    class FakeCollisionPipeline:
        def __init__(self, _model, **kwargs):
            pipeline_calls.append(kwargs)
            self._rigid_contact_max = kwargs.get("rigid_contact_max", 1)

        def contacts(self):
            return SimpleNamespace(rigid_contact_max=self._rigid_contact_max)

    manager = _manager_for()
    solver = SimpleNamespace(get_max_contact_count=lambda: 2)
    monkeypatch.setattr(newton_manager_module, "CollisionPipeline", FakeCollisionPipeline)
    manager._needs_collision_pipeline = True
    manager._solver = solver
    manager._newton._model = SimpleNamespace()
    manager._deterministic_mode = wp.DeterministicMode.GPU_TO_GPU

    manager._initialize_contacts()

    assert pipeline_calls == [
        {"broad_phase": "explicit", "deterministic": True},
        {"broad_phase": "explicit", "deterministic": True, "rigid_contact_max": 2},
    ]
    assert manager._newton._contacts.rigid_contact_max == 2


def test_collision_pipeline_cfg_translation_is_manager_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runtime manager translates nested collision data into Newton's native config."""
    pipeline_calls = []

    class FakeCollisionPipeline:
        def __init__(self, _model, **kwargs):
            pipeline_calls.append(kwargs)

        def contacts(self):
            return SimpleNamespace(rigid_contact_max=1)

    manager = _manager_for(
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="sap",
            sdf_hydroelastic_config=HydroelasticSDFCfg(buffer_fraction=0.5),
        )
    )
    monkeypatch.setattr(newton_manager_module, "CollisionPipeline", FakeCollisionPipeline)
    manager._needs_collision_pipeline = True
    manager._collision_cfg = manager.cfg.collision_cfg
    manager._newton._model = object()
    manager._deterministic_mode = wp.DeterministicMode.NOT_GUARANTEED

    manager._initialize_contacts()

    assert pipeline_calls[0]["broad_phase"] == "sap"
    assert pipeline_calls[0]["sdf_hydroelastic_config"].buffer_fraction == pytest.approx(0.5)


def test_resource_sensor_task_builds_and_refits_bvhs_before_rendering():
    """The registry resource owns the sole scene-query task path."""

    state = SimpleNamespace(body_q=None, particle_q=None)
    status = {"shape_refit": False, "particle_refit": False, "rendered": False}

    class Provider:
        point_count = 0

        def transform_generation(self, _name=None):
            return 0

        def request_transforms(self, _output_format):
            return None

        def point_generation(self):
            return 0

    class FakeModel:
        body_count = 0
        shape_count = 1
        particle_count = 1
        bvh_shapes = None
        bvh_particles = None

        def bvh_build_shapes(self, current_state):
            assert current_state is state
            self.bvh_shapes = object()

        def bvh_build_particles(self, current_state):
            assert current_state is state
            self.bvh_particles = object()

        def bvh_refit_shapes(self, current_state):
            assert current_state is state
            status["shape_refit"] = True

        def bvh_refit_particles(self, current_state):
            assert current_state is state
            status["particle_refit"] = True

    manager = _manager_for()
    model = FakeModel()
    provider = Provider()
    manager._newton._sim = SimpleNamespace(cfg=SimpleNamespace(device="cpu"), get_scene_data_provider=lambda: provider)

    def render():
        assert model.bvh_shapes is not None
        assert model.bvh_particles is not None
        assert status["shape_refit"]
        assert status["particle_refit"]
        status["rendered"] = True

    manager._newton._model = model
    manager._newton._state_0 = state
    manager._newton._register_sensor_task("render", render)
    manager._newton._update_sensor_tasks("render")

    assert status["rendered"]
    assert tuple(manager._newton._sensor_tasks) == ("render",)
    manager._newton._unregister_sensor_task("render")
    assert not manager._newton._sensor_tasks
    assert not hasattr(manager, "_update_sensor_tasks")


def test_resource_sensor_task_graph_is_reused_until_native_state_changes(monkeypatch):
    """CUDA scene queries capture on the resource and rebuild after pointer changes."""

    captures = []
    launches = []
    task_calls = []

    class Flags:
        def __getitem__(self, index):
            return index

        def assign(self, values):
            self.values = values.copy()

    @contextmanager
    def capture(**_kwargs):
        graph = object()
        captures.append(graph)
        yield SimpleNamespace(graph=graph)

    monkeypatch.setattr(wp, "ScopedDevice", lambda _device: nullcontext())
    monkeypatch.setattr(wp, "ScopedCapture", capture)
    monkeypatch.setattr(wp, "zeros", lambda *_args, **_kwargs: Flags())
    monkeypatch.setattr(wp, "capture_if", lambda _condition, fn: fn())
    monkeypatch.setattr(wp, "capture_launch", launches.append)

    provider = SimpleNamespace(
        transform_generation=lambda _name=None: 0,
        request_transforms=lambda _format: None,
        point_generation=lambda: 0,
    )
    manager = _manager_for(use_cuda_graph=True)
    manager._newton._sim = SimpleNamespace(
        cfg=SimpleNamespace(device="cuda:0"), get_scene_data_provider=lambda: provider
    )
    manager._newton._model = SimpleNamespace(body_count=0, shape_count=0, particle_count=0)
    manager._newton._state_0 = SimpleNamespace(body_q=None, particle_q=None)
    manager._newton._register_sensor_task("render", lambda: task_calls.append(None))

    manager._newton._update_sensor_tasks("render")
    manager._newton._update_sensor_tasks("render")

    assert len(captures) == 1
    assert launches == [captures[0], captures[0]]
    assert len(task_calls) == 2  # eager warm-up and graph recording, never eager replay

    manager._newton._state_0 = SimpleNamespace(body_q=object(), particle_q=None)
    manager._newton._update_sensor_tasks("render")

    assert len(captures) == 2
    assert launches[-1] is captures[1]
    manager._newton._unregister_sensor_task("render")
    assert manager._newton._sensor_capture is None
    assert not hasattr(manager, "_sensor_graph")


def test_sensor_bvh_shape_flags_are_fixed_before_builder_creation():
    """Builder finalization includes collision-only shapes without a later BVH rebuild."""
    import newton

    resource = NewtonReplicateContext(_SIM_CONTEXT)
    flags = ShapeFlags.VISIBLE | ShapeFlags.COLLIDE_SHAPES
    resource._sensor_bvh_shape_flags = flags
    builder = resource.create_builder()
    body = builder.add_body()
    builder.add_shape_sphere(body, cfg=newton.ModelBuilder.ShapeConfig(is_visible=False))

    model = builder.finalize(device="cpu")

    assert builder.default_bvh_cfg.shape_flags == flags
    assert model.bvh_shape_count_enabled == 1
    assert model.bvh_shapes is not None


def test_sensor_task_registration_has_no_raycast_bvh_fallback():
    """Raycast BVH requirements belong to builder creation, not task registration."""
    assert "include_collision_shapes" not in signature(NewtonReplicateContext._register_sensor_task).parameters
    assert not hasattr(NewtonReplicateContext(_SIM_CONTEXT), "_sensor_bvh_has_collision_shapes")


def test_newton_shape_cfg_defaults_match_newton_shape_config():
    """``NewtonShapeCfg`` contact defaults mirror Newton's ``ShapeConfig``.

    Guards the invariant that keeps ``checked_apply`` a no-op for envs that do
    not override ``ke``/``kd``/``mu``: if Newton's upstream defaults drift, this
    fails instead of silently clobbering every Newton scene's shape materials.
    """
    import newton

    upstream = newton.ModelBuilder().default_shape_cfg
    shape_cfg = NewtonShapeCfg()
    assert shape_cfg.ke == upstream.ke
    assert shape_cfg.kd == upstream.kd
    assert shape_cfg.mu == upstream.mu


def test_mpm_solver_cfg_maps_only_newton_solver_fields():
    """MPM config forwarding ignores Isaac Lab manager metadata."""

    solver_cfg = MPMSolverCfg(max_iterations=7, voxel_size=0.04)

    newton_cfg = _make_solver_config(solver_cfg)

    assert newton_cfg.max_iterations == 7
    assert newton_cfg.voxel_size == 0.04
    assert not hasattr(newton_cfg, "class_type")
    # Manager-level stepping option must not leak into the Newton solver config.
    assert not hasattr(newton_cfg, "project_outside_colliders")


@pytest.mark.parametrize(
    "solver_cfg",
    [
        MJWarpSolverCfg(),
        MPMSolverCfg(),
        XPBDSolverCfg(),
        VBDSolverCfg(),
        FeatherstoneSolverCfg(),
        KaminoPADMMSolverCfg(),
    ],
)
def test_solver_cfgs_use_class_type_as_the_only_manager_selector(solver_cfg):
    """Solver cfgs do not carry a duplicate string selector."""
    assert not hasattr(solver_cfg, "solver_type")


def test_mjwarp_cfg_has_no_ignored_parallel_line_search_flag():
    """MJWarp cfg exposes only line-search controls consumed by Newton."""
    assert not hasattr(MJWarpSolverCfg(), "ls_parallel")


@pytest.mark.parametrize("mode", ["forward", "backward"])
def test_mpm_solver_cfg_preserves_canonical_collider_velocity_modes(mode, recwarn):
    """Canonical collider velocity modes pass through without deprecation warnings."""
    newton_cfg = _make_solver_config(MPMSolverCfg(collider_velocity_mode=mode))

    assert newton_cfg.collider_velocity_mode == mode
    assert not [warning for warning in recwarn if issubclass(warning.category, DeprecationWarning)]


# Tuples of ``(field_name, non_default_value)`` covering every solver-tunable
# field on :class:`MPMSolverCfg`. Each entry exercises the implementation-side
# SolverImplicitMPM.Config construction so a Newton field rename or accidental
# drop is caught here instead of silently producing wrong-physics runs.
_MPM_FIELD_VALUES = [
    ("max_iterations", 13),
    ("tolerance", 5.0e-5),
    ("solver", "gauss-seidel"),
    ("warmstart_mode", "particles"),
    ("collider_velocity_mode", "backward"),
    ("voxel_size", 0.0375),
    ("grid_type", "dense"),
    ("grid_padding", 4),
    ("max_active_cell_count", 1024),
    ("max_leaf_node_count", 512),
    ("max_lower_node_count", 128),
    ("max_upper_node_count", 32),
    ("separate_worlds", True),
    ("transfer_scheme", "pic"),
    ("integration_scheme", "gimp"),
    ("critical_fraction", 0.25),
    ("air_drag", 0.5),
    ("collider_normal_from_sdf_gradient", True),
    ("collider_basis", "Q1"),
    ("strain_basis", "P1d"),
    ("velocity_basis", "B2"),
]


@pytest.mark.parametrize("field_name, value", _MPM_FIELD_VALUES)
def test_mpm_solver_cfg_forwards_every_solver_field(field_name, value):
    """Every tunable MPM cfg field round-trips into ``SolverImplicitMPM.Config``.

    Guards against MPM manager construction dropping or mis-naming a field if
    Newton's config surface changes.
    """
    solver_cfg = MPMSolverCfg(**{field_name: value})
    newton_cfg = _make_solver_config(solver_cfg)
    assert hasattr(newton_cfg, field_name), (
        f"{field_name!r} disappeared from SolverImplicitMPM.Config — MPMSolverCfg needs to drop or rename it."
    )
    assert getattr(newton_cfg, field_name) == value


_KAMINO_PADMM_FIELD_VALUES = [
    ("max_iterations", 13),
    ("primal_tolerance", 1.0e-5),
    ("dual_tolerance", 1.0e-5),
    ("compl_tolerance", 1.0e-5),
    ("restart_tolerance", 0.5),
    ("rho_0", 0.5),
    ("rho_min", 1.0e-4),
    ("a_0", 0.5),
    ("alpha", 11.0),
    ("tau", 1.6),
    ("eta", 1.0e-4),
    ("penalty_update_freq", 2),
    ("penalty_update_method", "balanced"),
    ("linear_solver_tolerance", 1.0e-3),
    ("linear_solver_tolerance_ratio", 0.1),
    ("use_acceleration", False),
    ("use_graph_conditionals", False),
    ("warmstart_mode", "none"),
    ("contact_warmstart_method", "geom_pair_net_force"),
]

_KAMINO_DVI_FIELD_VALUES = [
    ("tolerance", 1.0e-4),
    ("regularization", 1.0e-5),
    ("omega", 1.5),
    ("max_alternating_iterations", 15),
    ("inequality_sweeps_per_iteration", 2),
    ("bilateral_solve_interval", 2),
    ("bilateral_solver_type", "LLTBRCM"),
    ("bilateral_solver_kwargs", {"block_size": 32}),
    ("warmstart_mode", "internal"),
    ("contact_warmstart_method", "geom_pair_net_force"),
]

_KAMINO_DYNAMICS_FIELD_VALUES = [
    ("preconditioning", False),
    ("linear_solver_type", "LLTBRCM"),
    ("linear_solver_kwargs", {"maxiter": 9}),
]


@pytest.mark.parametrize("field_name, value", _KAMINO_PADMM_FIELD_VALUES)
def test_kamino_manager_forwards_padmm_fields(field_name, value, monkeypatch):
    """Every tunable P-ADMM cfg field round-trips into ``PADMMSolverConfig``."""
    solver_cfg = KaminoPADMMSolverCfg(
        dynamics_solver_cfg=KaminoPADMMCfg(**{field_name: value}),
        sparse_jacobian=True if field_name == "penalty_update_method" else None,
        sparse_dynamics=field_name == "penalty_update_method",
    )
    newton_cfg = _kamino_config(solver_cfg, monkeypatch)
    assert hasattr(newton_cfg.padmm, field_name), (
        f"{field_name!r} disappeared from PADMMSolverConfig — KaminoPADMMCfg needs to drop or rename it."
    )
    assert getattr(newton_cfg.padmm, field_name) == value


@pytest.mark.parametrize("field_name, value", _KAMINO_DVI_FIELD_VALUES)
def test_kamino_manager_forwards_dvi_fields(field_name, value, monkeypatch):
    """Every tunable DVI cfg field round-trips into ``DVISolverConfig``."""
    solver_cfg = KaminoDVISolverCfg(
        dynamics=KaminoDynamicsCfg(preconditioning=False),
        dynamics_solver_cfg=KaminoDVICfg(**{field_name: value}),
    )
    newton_cfg = _kamino_config(solver_cfg, monkeypatch)
    assert hasattr(newton_cfg.dvi, field_name), (
        f"{field_name!r} disappeared from DVISolverConfig — KaminoDVICfg needs to drop or rename it."
    )
    assert getattr(newton_cfg.dvi, field_name) == value


@pytest.mark.parametrize("field_name, value", _KAMINO_DYNAMICS_FIELD_VALUES)
def test_kamino_manager_forwards_dynamics_fields(field_name, value, monkeypatch):
    """Every tunable dynamics cfg field round-trips into ``ConstrainedDynamicsConfig``."""
    solver_type = KaminoDVISolverCfg if field_name == "preconditioning" and value is False else KaminoPADMMSolverCfg
    solver_cfg = solver_type(dynamics=KaminoDynamicsCfg(**{field_name: value}))
    newton_cfg = _kamino_config(solver_cfg, monkeypatch)
    assert hasattr(newton_cfg.dynamics, field_name), (
        f"{field_name!r} disappeared from ConstrainedDynamicsConfig — KaminoDynamicsCfg needs updating."
    )
    assert getattr(newton_cfg.dynamics, field_name) == value


def test_kamino_manager_excludes_cfg_metadata(monkeypatch):
    """Isaac Lab metadata and manager-only fields do not leak into Newton config."""
    solver_cfg = KaminoPADMMSolverCfg(max_contacts_per_world=32)
    newton_cfg = _kamino_config(solver_cfg, monkeypatch)
    assert not hasattr(newton_cfg, "class_type")
    assert not hasattr(newton_cfg, "max_contacts_per_world")


def test_kamino_concrete_solver_configs_select_their_backends(monkeypatch):
    """Concrete solver config types select their corresponding Newton backends."""
    padmm_cfg = KaminoPADMMSolverCfg(sparse_jacobian=True)
    dvi_cfg = KaminoDVISolverCfg()
    assert _kamino_config(padmm_cfg, monkeypatch).dynamics_solver == "padmm"
    assert _kamino_config(dvi_cfg, monkeypatch).dynamics_solver == "dvi"


def test_kamino_dvi_rejects_preconditioning(monkeypatch):
    """DVI preserves Newton's preconditioning compatibility check."""
    solver_cfg = KaminoDVISolverCfg(dynamics=KaminoDynamicsCfg(preconditioning=True))
    with pytest.raises(ValueError, match="preconditioning"):
        _kamino_config(solver_cfg, monkeypatch)


@pytest.mark.parametrize(
    ("solver_cfg", "active", "inactive"),
    [
        (MJWarpSolverCfg(), "mujoco:condim", ("kamino:max_solver_iterations", "mpm:young_modulus")),
        (KaminoPADMMSolverCfg(), "kamino:max_solver_iterations", ("mujoco:condim", "mpm:young_modulus")),
    ],
)
def test_rigid_solver_registers_only_its_builder_attributes(solver_cfg, active, inactive):
    """A rigid solver declares its own builder schema and no inactive solver schema."""
    builder = _manager_for(solver_cfg).create_builder()

    assert builder.has_custom_attribute(active)
    assert all(not builder.has_custom_attribute(name) for name in inactive)


def test_clone_source_builder_has_no_solver_dependency():
    """The active manager's builder factory, not the cloner, owns solver attributes."""
    import isaaclab_newton.cloner.newton_clone_utils as clone_utils

    assert not hasattr(clone_utils, "solvers")


def test_mpm_prepare_builder_makes_kinematic_bodies_massless():
    """Kinematic bodies must be massless so MPM treats them as kinematic colliders."""
    import newton

    manager = _manager_for(MPMSolverCfg())
    builder = newton.ModelBuilder()
    kinematic_body = builder.add_body(
        mass=0.35,
        inertia=wp.mat33(1.0),
        is_kinematic=True,
        label="kinematic_collider",
    )
    dynamic_body = builder.add_body(
        mass=1.2,
        inertia=wp.mat33(2.0),
        is_kinematic=False,
        label="dynamic_body",
    )

    manager._prepare_builder_for_finalize(builder)

    assert builder.body_flags[kinematic_body] & int(newton.BodyFlags.KINEMATIC)
    assert builder.body_mass[kinematic_body] == 0.0
    assert builder.body_inv_mass[kinematic_body] == 0.0
    assert np.allclose(np.array(builder.body_inertia[kinematic_body]), 0.0)
    assert np.allclose(np.array(builder.body_inv_inertia[kinematic_body]), 0.0)

    assert builder.body_mass[dynamic_body] == pytest.approx(1.2)
    assert builder.body_inv_mass[dynamic_body] == pytest.approx(1.0 / 1.2)
    assert np.allclose(np.array(builder.body_inertia[dynamic_body]), 2.0)


@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_mpm_prepare_builder_converts_convex_mesh_before_solver_construction():
    """Convex meshes must become triangle meshes before implicit MPM consumes the model."""
    import newton

    manager = _manager_for(MPMSolverCfg(max_iterations=2, voxel_size=0.05))
    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    body = builder.add_body(label="convex_mesh_collider")
    mesh = newton.Mesh(
        vertices=[(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)],
        indices=[0, 1, 2],
    )
    shape = builder.add_shape_mesh(body, mesh=mesh)
    builder.shape_type[shape] = newton.GeoType.CONVEX_MESH
    builder.add_particles(
        pos=[(0.0, 0.0, 0.1)],
        vel=[(0.0, 0.0, 0.0)],
        mass=[0.01],
        radius=[0.02],
        custom_attributes={
            "mpm:viscosity": 50.0,
            "mpm:friction": 0.0,
            "mpm:tensile_yield_ratio": 1.0,
            "mpm:yield_pressure": 1.0e15,
            "mpm:yield_stress": 0.0,
            "mpm:young_modulus": 1.0e15,
            "mpm:damping": 0.0,
        },
    )

    manager._prepare_builder_for_finalize(builder)
    model = builder.finalize(device="cuda:0")
    solver = manager._create_solver(model, manager.cfg)

    assert builder.shape_type[shape] == newton.GeoType.MESH
    assert isinstance(solver, SolverImplicitMPM)


def test_mpm_end_to_end_with_particle_custom_attributes():
    """End-to-end MPM step through the declarative particle spawner."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=MPMSolverCfg(
            max_iterations=2,
            voxel_size=0.05,
            use_cuda_graph=False,
        ),
    )

    with _direct_builder_simulation(sim_cfg) as sim:
        manager = sim._physics_manager
        builder = manager.create_builder()
        positions = [(0.0, 0.0, 0.10), (0.05, 0.0, 0.10), (0.0, 0.05, 0.10)]
        emit_mpm_particles(
            builder,
            MPMPointsCfg(
                positions=positions,
                mass=0.01,
                radius=0.02,
                material=MPMParticleMaterialCfg(viscosity=50.0, friction=0.0),
            ),
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
        assert builder.has_custom_attribute("mpm:young_modulus")
        manager.set_builder(builder)

        sim.reset()
        assert isinstance(manager._solver, SolverImplicitMPM)
        sim.step(render=False)


@pytest.mark.parametrize("project_outside", [True, False])
def test_mpm_project_outside_colliders_gates_projection(project_outside):
    """``project_outside_colliders`` controls whether ``project_outside`` runs per substep.

    Wraps the solver's ``project_outside`` with a counter after ``sim.reset()``
    (``use_cuda_graph=False`` keeps the Python callable on the step path) and
    runs one tick. The call count is positive only when the flag is set.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=MPMSolverCfg(
            max_iterations=2,
            voxel_size=0.05,
            project_outside_colliders=project_outside,
            use_cuda_graph=False,
        ),
    )

    with _direct_builder_simulation(sim_cfg) as sim:
        manager = sim._physics_manager
        builder = manager.create_builder()
        emit_mpm_particles(
            builder,
            MPMPointsCfg(
                positions=[(0.0, 0.0, 0.10), (0.05, 0.0, 0.10), (0.0, 0.05, 0.10)],
                mass=0.01,
                radius=0.02,
                material=MPMParticleMaterialCfg(viscosity=50.0, friction=0.0),
            ),
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
        manager.set_builder(builder)
        sim.reset()

        calls = {"n": 0}
        original_project = manager._solver.project_outside

        def counting_project(*args, **kwargs):
            calls["n"] += 1
            return original_project(*args, **kwargs)

        manager._solver.project_outside = counting_project
        try:
            sim.step(render=False)
        finally:
            manager._solver.project_outside = original_project

        if project_outside:
            assert calls["n"] >= 1
        else:
            assert calls["n"] == 0


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({"grid_type": "fixed"}, True, id="fixed"),
        pytest.param({}, True, id="bounded_sparse"),
        pytest.param({"max_active_cell_count": -1}, False, id="unbounded_sparse"),
        pytest.param({"grid_type": "dense"}, False, id="dense"),
        pytest.param({"grid_padding": 1}, False, id="padded_sparse"),
        pytest.param({"velocity_basis": "P0"}, False, id="velocity_basis"),
        pytest.param({"strain_basis": "GIMP"}, False, id="strain_basis"),
        pytest.param({"collider_basis": "GIMP"}, False, id="collider_basis"),
    ],
)
def test_mpm_cuda_graph_capture_supports_static_topology(monkeypatch, overrides, expected):
    """Only fixed and capacity-bounded rebuildable sparse grids support outer capture."""
    values = {
        "grid_type": "sparse",
        "max_active_cell_count": 1024,
        "grid_padding": 0,
        "velocity_basis": "Q1",
        "strain_basis": "P0",
        "collider_basis": "S2",
    }
    manager = _manager_for(MPMSolverCfg())
    manager._solver = SimpleNamespace(**(values | overrides))

    assert manager._supports_cuda_graph_capture() is expected


def test_mpm_status_check_runs_only_after_graph_capture(monkeypatch):
    """Sparse-grid asynchronous failures are queried only after graph replay."""
    manager = _manager_for(MPMSolverCfg())
    calls = []
    solver = SimpleNamespace(check_sparse_grid_rebuild_status=lambda: calls.append("check"))
    monkeypatch.setattr(manager, "_implicit_mpm_solvers", lambda: (solver,))

    manager._check_solver_status()
    manager._graph = object()
    manager._check_solver_status()

    assert calls == ["check"]


def test_nested_mpm_solver_discovery_is_cached(monkeypatch):
    """A coupled solver's immutable entry table is traversed only once per solver instance."""
    mpm_solver = object.__new__(SolverImplicitMPM)

    class CoupledSolver:
        calls = 0

        def entry_names(self):
            self.calls += 1
            return ("media",)

        def solver(self, _name):
            return mpm_solver

    manager = _manager_for(MPMSolverCfg())
    root = CoupledSolver()
    manager._solver = root

    assert manager._implicit_mpm_solvers() == (mpm_solver,)
    assert manager._implicit_mpm_solvers() == (mpm_solver,)
    assert root.calls == 1


def test_mpm_supported_cuda_graph_capture_defers_until_step():
    """A bounded sparse grid must not capture before reset-authored topology exists."""
    manager = _manager_for(MPMSolverCfg(), use_cuda_graph=True)
    manager._device = "cuda:0"
    manager._solver = SimpleNamespace(
        grid_type="sparse",
        max_active_cell_count=1024,
        grid_padding=0,
        velocity_basis="Q1",
        strain_basis="P0",
        collider_basis="S2",
    )
    manager._newton._mpm_object_registry.append(object())

    manager._schedule_graph_capture()

    assert manager._graph is None
    assert manager._graph_capture_pending is True


def test_mpm_unsupported_cuda_graph_capture_fails_explicitly(monkeypatch):
    """Selecting graph mode for an unbounded sparse grid must fail during initialization."""
    manager = _manager_for(MPMSolverCfg(grid_type="sparse"), use_cuda_graph=True)
    manager._device = "cuda:0"
    manager._solver = SimpleNamespace(
        grid_type="sparse",
        max_active_cell_count=-1,
        grid_padding=0,
        velocity_basis="Q1",
        strain_basis="P0",
        collider_basis="S2",
    )
    manager._graph = object()
    manager._graph_capture_pending = True

    with pytest.raises(RuntimeError, match="does not support CUDA graph capture"):
        manager._schedule_graph_capture()

    assert manager._graph is None
    assert manager._graph_capture_pending is False


def test_deferred_cuda_graph_capture_requires_a_graph(monkeypatch):
    """A deferred capture returning no graph must not execute the selected mode eagerly."""

    class EmptyCapture:
        graph = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    manager = _manager_for(use_cuda_graph=True)
    manager._device = "cuda:0"
    manager._sim = SimpleNamespace(is_playing=lambda: True)
    manager._world_reset_mask = None
    manager._reset_solver_internals_delegate = lambda _mask: None
    manager._graph_capture_pending = True
    monkeypatch.setattr(manager, "_is_all_graphable", lambda: False)
    monkeypatch.setattr(manager, "_reconcile_state", lambda: None)
    monkeypatch.setattr(manager, "_simulate_physics_only", lambda: None)
    monkeypatch.setattr(wp, "ScopedCapture", lambda **_kwargs: EmptyCapture())

    with pytest.raises(RuntimeError, match="capture produced no graph"):
        manager.step()


def test_cuda_graph_capture_defers_gc_without_forcing_a_collection(monkeypatch):
    """Capture should restore automatic collection without a synchronous full sweep."""
    events = []
    with monkeypatch.context() as scoped:
        scoped.setattr(newton_manager_module.gc, "isenabled", lambda: True)
        scoped.setattr(newton_manager_module.gc, "disable", lambda: events.append("disable"))
        scoped.setattr(newton_manager_module.gc, "enable", lambda: events.append("enable"))
        scoped.setattr(newton_manager_module.gc, "collect", lambda: events.append("collect"))
        with newton_manager_module._paused_gc():
            events.append("capture")
    assert events == ["disable", "capture", "enable"]


def test_cuda_graph_capture_uses_simulation_device(monkeypatch):
    """CUDA graph capture should use the simulation device instead of Warp's default device."""

    captures = []
    captured_graph = object()

    class FakeScopedCapture:
        def __init__(self, *, device, force_module_load):
            captures.append((device, force_module_load))
            self.graph = captured_graph

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    manager = _manager_for(use_cuda_graph=True)
    manager._device = "cuda:1"
    manager._sim = SimpleNamespace(is_playing=lambda: True)
    manager._world_reset_mask = None
    manager._reset_solver_internals_delegate = lambda _mask: None
    monkeypatch.setattr(manager, "_is_all_graphable", lambda: False)
    monkeypatch.setattr(manager, "_reconcile_state", lambda: None)
    monkeypatch.setattr(manager, "_simulate_physics_only", lambda: None)
    monkeypatch.setattr(wp, "ScopedCapture", FakeScopedCapture)
    monkeypatch.setattr(wp, "capture_launch", lambda graph: None)
    monkeypatch.setattr(wp, "get_device", lambda _device: SimpleNamespace(is_cuda=False))

    manager._schedule_graph_capture()
    assert manager._graph is captured_graph
    manager.step()

    assert captures == [("cuda:1", False)]


# ---------------------------------------------------------------------------
# Manager state-refresh boundaries (no SimulationContext required)
# ---------------------------------------------------------------------------


def test_empty_cable_plan_skips_model_discovery():
    """An empty plan must not scan replicated model labels for cable geometry."""
    manager = _manager_for(use_cuda_graph=False)
    manager._newton._model = SimpleNamespace()
    manager._initialize_cable_publication(SimpleNamespace(point_bindings=lambda _kind: ()))
    assert manager._scene_data_backend._cable_publication.data.segment_counts is None


def test_transform_publication_returns_pointer_without_deferred_physics(monkeypatch):
    """SDP reads never reconcile generalized-coordinate writes on a consumer's behalf."""
    manager = _manager_for(use_cuda_graph=False)
    forwards = []
    monkeypatch.setattr(manager, "forward", lambda: forwards.append(None))
    manager._world_reset_mask = wp.ones(1, dtype=wp.bool, device="cpu")
    manager._fk_reset_mask = wp.zeros(1, dtype=wp.bool, device="cpu")

    publication = manager._scene_data_backend.transform_publication
    assert publication is manager._scene_data_backend._transform_publication
    assert manager._scene_data_backend.transform_publication is publication
    assert forwards == []


def test_forward_consumes_existing_reset_masks(monkeypatch):
    """The existing device masks are the complete input to masked FK and the solver reset hook."""
    manager = _manager_for()
    world_mask = wp.array([False, True], dtype=wp.bool, device="cpu")
    fk_mask = wp.array([True, False], dtype=wp.bool, device="cpu")
    observed: list[tuple[list[bool], list[bool]]] = []
    solver_resets: list[list[bool]] = []

    def record_fk(worlds, articulations):
        observed.append((worlds.numpy().tolist(), articulations.numpy().tolist()))

    class _RecordingSolver:
        def reset(self, state, world_mask=None, flags=0):
            solver_resets.append(world_mask.numpy().tolist())

    manager._world_reset_mask = world_mask
    manager._fk_reset_mask = fk_mask
    manager._reconciliation_pending = True
    manager._eval_fk = record_fk
    manager._solver = _RecordingSolver()
    manager._reset_solver_internals_delegate = manager._reset_solver_internals
    callbacks = []
    manager._post_step_callbacks = [lambda: callbacks.append(None)]
    manager._scene_data_backend._transform_publication.dirty = False
    manager._scene_data_backend._cable_publication.dirty = False

    manager.forward()

    assert observed == [([False, True], [True, False])]
    assert solver_resets == [[False, True]]
    assert callbacks == [None]
    assert manager._scene_data_backend._transform_publication.dirty
    assert manager._scene_data_backend._cable_publication.dirty
    assert world_mask.numpy().tolist() == [False, False]
    assert fk_mask.numpy().tolist() == [False, False]
    assert not manager._reconciliation_pending

    manager.forward()
    assert observed == [([False, True], [True, False])]
    assert solver_resets == [[False, True]]
    assert callbacks == [None, None]


def test_step_reconciles_authored_state_once(monkeypatch):
    """Physics consumes authored reset masks once, not again on a clean step."""
    manager = _manager_for(use_cuda_graph=False)
    manager._device = "cpu"
    manager._sim = SimpleNamespace(is_playing=lambda: True)
    manager._world_reset_mask = wp.array([True], dtype=wp.bool, device="cpu")
    manager._fk_reset_mask = wp.array([True], dtype=wp.bool, device="cpu")
    manager._reconciliation_pending = True
    resets = []
    fk_evaluations = []
    manager._reset_solver_internals_delegate = lambda mask: resets.append(mask.numpy().tolist())
    manager._eval_fk = lambda worlds, articulations: fk_evaluations.append(
        (worlds.numpy().tolist(), articulations.numpy().tolist())
    )
    monkeypatch.setattr(manager, "_is_all_graphable", lambda: False)
    monkeypatch.setattr(manager, "_simulate_physics_only", lambda: None)
    monkeypatch.setattr(manager, "_mark_state_dirty", lambda: None)

    manager.step()
    manager.step()

    assert resets == [[True]]
    assert fk_evaluations == [([True], [True])]


def test_captured_invalidation_keeps_reconciliation_at_explicit_boundaries(monkeypatch):
    """A captured mask writer can replay without another Python invalidation call."""
    manager = _manager_for()
    manager._device = "cpu"
    manager._newton._model = SimpleNamespace(world_count=1)
    manager._world_reset_mask = wp.zeros(1, dtype=wp.bool, device="cpu")
    manager._fk_reset_mask = wp.zeros(1, dtype=wp.bool, device="cpu")
    monkeypatch.setattr(wp, "get_device", lambda _device: SimpleNamespace(is_capturing=True))

    manager.invalidate_fk()
    manager._reset_solver_internals_delegate = lambda _mask: None
    reconciliations = []
    manager._eval_fk = lambda _worlds, _articulations: reconciliations.append(None)

    manager.forward()
    manager.forward()

    assert reconciliations == [None, None]
    assert not manager._reconciliation_pending
    assert manager._reconciliation_replayable


def test_forward_dispatches_active_mpm_reset_hook_through_base_manager(monkeypatch):
    """The MPM manager instance owns its solver-specific reset behavior."""
    manager = _manager_for(MPMSolverCfg())
    world_mask = wp.array([True, False], dtype=wp.bool, device="cpu")
    fk_mask = wp.array([], dtype=wp.bool, device="cpu")

    class _RejectingSolver:
        def reset(self, state, world_mask=None, flags=0):
            raise AssertionError("the base reset hook must not run for implicit MPM")

    manager._world_reset_mask = world_mask
    manager._fk_reset_mask = fk_mask
    manager._reconciliation_pending = True
    manager._eval_fk = lambda worlds, articulations: None
    manager._solver = _RejectingSolver()
    manager._reset_solver_internals_delegate = manager._reset_solver_internals

    manager.forward()

    assert world_mask.numpy().tolist() == [False, False]


# ---------------------------------------------------------------------------
# Manager class hierarchy and factory contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "manager",
    [
        NewtonMJWarpManager,
        NewtonXPBDManager,
        NewtonVBDManager,
        NewtonFeatherstoneManager,
        NewtonKaminoManager,
        NewtonMPMManager,
    ],
)
def test_subclass_of_newton_manager(manager):
    """All concrete managers inherit from :class:`NewtonManager`."""
    assert issubclass(manager, NewtonManager)
    # Subclasses must override the abstract factory.
    assert manager._build_solver is not NewtonManager._build_solver
    assert manager._create_solver is not NewtonManager._create_solver


def test_manager_clear_preserves_shared_resource():
    """Manager teardown leaves registry-owned Newton state for remaining consumers."""
    manager = _manager_for()
    manager._newton._supports_rigid_body_force_input = True
    model = manager._newton._model = object()
    manager._solver = object()

    manager.clear()

    assert manager._newton._model is model
    assert manager._newton._supports_rigid_body_force_input is True
    assert manager._solver is None


def test_registry_resource_exposes_physics_force_capability():
    resource = NewtonReplicateContext(_SIM_CONTEXT)
    assert not resource.supports_rigid_body_force_input()

    manager = _manager_for()
    manager._newton._supports_rigid_body_force_input = True
    assert manager._newton.supports_rigid_body_force_input()

    manager._newton._supports_rigid_body_force_input = False
    assert not manager._newton.supports_rigid_body_force_input()


def test_initialize_solver_dispatches_physics_ready_before_graph_capture(monkeypatch):
    """The generic ready event sees a usable solver before graph capture."""
    events: list[tuple[str, bool, bool] | str] = []
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        physics=MJWarpSolverCfg(use_cuda_graph=False),
    )

    with _direct_builder_simulation(sim_cfg) as sim:
        manager = sim._physics_manager
        builder = manager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        manager.set_builder(builder)
        manager.register_callback(
            lambda _payload: events.append(
                ("ready", manager._newton.supports_rigid_body_force_input(), manager._eval_fk == manager._eval_fk_impl)
            ),
            PhysicsEvent.PHYSICS_READY,
        )
        monkeypatch.setattr(manager, "_schedule_graph_capture", lambda: events.append("capture"))

        sim.reset()

    assert events == [("ready", True, True), "capture"]


# ---------------------------------------------------------------------------
# End-to-end: build each solver via SimulationContext
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, expected_solver_cls,"
    " expected_use_single_state, expected_needs_collision_pipeline",
    SOLVER_MATRIX,
)
def test_initialize_solver_populates_canonical_state(
    solver_cfg_factory,
    expected_manager,
    expected_solver_cls,
    expected_use_single_state,
    expected_needs_collision_pipeline,
    monkeypatch,
):
    """Build each solver on the manager instance selected by ``cfg.class_type``.

    The builder is pre-populated directly (instead of relying on a USD stage)
    with either a minimal particle grid for MPM or a one-body / one-joint scene
    for rigid/articulation solvers:

    1. :class:`SolverImplicitMPM` receives particles and their custom attributes
       from the declarative MPM spawner.
    2. :class:`SolverMuJoCo` requires at least one joint to convert the model
       to MJCF; a ground-plane-only scene fails MJCF conversion.
    3. Kamino's internal collision detector requires collidable geometry to
       construct its collision pipeline.
    4. The test supplies the registry-owned builder directly, so it does not depend on USD assets.
    """
    solver_cfg = solver_cfg_factory()
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=solver_cfg.replace(use_cuda_graph=False),
    )

    with _direct_builder_simulation(sim_cfg) as sim:
        manager = sim._physics_manager
        assert type(manager) is expected_manager

        state_calls = []
        model_state = newton_manager_module.Model.state

        def record_state(model, *args, **kwargs):
            state_calls.append(model)
            return model_state(model, *args, **kwargs)

        monkeypatch.setattr(newton_manager_module.Model, "state", record_state)

        builder = manager.create_builder()
        if expected_solver_cls is SolverImplicitMPM:
            emit_mpm_particles(
                builder,
                MPMGridCfg(
                    lower=(-0.05, -0.05, 0.10),
                    upper=(0.05, 0.05, 0.20),
                    voxel_size=0.05,
                    particle_placement="cell_center",
                    mass=0.01,
                    radius=0.02,
                ),
                position=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )
        elif expected_solver_cls is SolverVBD:
            builder.add_cloth_mesh(
                pos=wp.vec3(0.0, 0.0, 0.1),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0),
                vertices=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.1, 0.0, 0.0), wp.vec3(0.0, 0.1, 0.0)],
                indices=[0, 1, 2],
                density=1.0,
                particle_radius=0.01,
            )
        else:
            # Pre-populate the builder with a minimal scene so MJCF conversion has
            # something to work with.
            body = builder.add_body(mass=1.0)
            builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
            if isinstance(solver_cfg, (KaminoPADMMSolverCfg, KaminoDVISolverCfg)) and solver_cfg.use_collision_detector:
                builder.add_shape_sphere(body=body, radius=0.05)
                builder.add_ground_plane()
        manager.set_builder(builder)

        # Force resolution and bring up the solver.
        expected_supports_force_input = RIGID_BODY_FORCE_INPUT_SUPPORT[expected_manager]
        manager._newton._supports_rigid_body_force_input = not expected_supports_force_input
        sim.reset()

        assert isinstance(manager._solver, expected_solver_cls)
        assert manager._use_single_state is expected_use_single_state
        assert len(state_calls) == (1 if expected_use_single_state else 2)
        assert (manager._newton._state_1 is manager._newton._state_0) is expected_use_single_state
        assert manager._needs_collision_pipeline is expected_needs_collision_pipeline
        assert manager._newton.supports_rigid_body_force_input() is expected_supports_force_input
        assert manager._reset_solver_internals_delegate.__self__ is manager
        assert manager._reset_solver_internals_delegate.__func__ is expected_manager._reset_solver_internals

        # ``_contacts`` is allocated whichever way contacts are handled
        # (MuJoCo internal buffer or Newton pipeline output).
        # Kamino with internal contacts and MPM do not currently set contacts.
        if expected_solver_cls not in (SolverKamino, SolverImplicitMPM):
            assert manager._newton._contacts is not None

        published_body_q = manager._scene_data_backend.transform_publication.data.transforms
        published_particle_q = manager._scene_data_backend.point_publications["points"].data.points
        assert published_body_q is manager._newton._state_0.body_q
        assert published_particle_q is manager._newton._state_0.particle_q

        # One step proves both dispatch and the setup-time SDP pointer binding end-to-end.
        sim.step(render=False)
        assert published_body_q is manager._newton._state_0.body_q
        assert published_particle_q is manager._newton._state_0.particle_q


def test_mjwarp_internal_contacts_with_collision_cfg_raises():
    """Combining ``use_mujoco_contacts=True`` with a ``collision_cfg`` is rejected.

    The check lives in :meth:`NewtonMJWarpManager._build_solver`, so it fires
    during :meth:`NewtonManager.initialize_solver` rather than cfg construction.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=MJWarpSolverCfg(
            use_mujoco_contacts=True,
            collision_cfg=NewtonCollisionPipelineCfg(),
            use_cuda_graph=False,
        ),
    )

    with _direct_builder_simulation(sim_cfg) as sim:
        manager = sim._physics_manager
        builder = manager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        manager.set_builder(builder)

        with pytest.raises(ValueError, match="collision_cfg cannot be set"):
            sim.reset()


@pytest.mark.parametrize(
    "num_substeps, collision_decimation, expected_mid_loop_collides",
    [
        (8, 0, 0),  # Feature disabled.
        (8, 2, 3),  # Re-collide after substeps 2, 4, 6 (skip last).
        (8, 4, 1),  # Re-collide after substep 4 only.
        (8, 7, 1),  # Re-collide after substep 7 only.
        (8, 8, 0),  # Gated off (>= num_substeps).
    ],
)
def test_collision_decimation_invokes_mid_loop_collide(num_substeps, collision_decimation, expected_mid_loop_collides):
    """``_run_solver_substeps`` re-invokes ``collide`` at the expected substeps.

    Wraps the selected manager's collision pipeline with a counter and
    runs one physics tick. The collide-call count is ``1`` (top-of-tick) plus
    one per matching mid-loop substep, excluding the last substep.

    The scene has a free-joint sphere falling onto a ground plane so the
    broadphase actually generates pairs — guards against a future change
    that skips ``collide()`` when there are no collidable shapes.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=MJWarpSolverCfg(
            use_mujoco_contacts=False,
            num_substeps=num_substeps,
            collision_decimation=collision_decimation,
            use_cuda_graph=False,
        ),
    )

    with _direct_builder_simulation(sim_cfg) as sim:
        manager = sim._physics_manager
        builder = manager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_free(child=body)
        builder.add_shape_sphere(body=body, radius=0.05)
        builder.add_ground_plane()
        # Lift the sphere to 0.5 m above the plane so the scene is non-degenerate.
        # joint_q for a free joint is [tx, ty, tz, qx, qy, qz, qw].
        builder.joint_q[-7:] = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
        manager.set_builder(builder)
        sim.reset()

        # Wrap collide() with a counter — must run after sim.reset() so the
        # pipeline is allocated, and use_cuda_graph=False so the wrapped
        # Python callable isn't bypassed by a captured graph.
        calls = {"n": 0}
        original_collide = manager._collision_pipeline.collide

        def counting_collide(state, contacts):
            calls["n"] += 1
            return original_collide(state, contacts)

        manager._collision_pipeline.collide = counting_collide
        try:
            sim.step(render=False)
        finally:
            manager._collision_pipeline.collide = original_collide

        # Expect: 1 (top-of-tick) + expected_mid_loop_collides.
        assert calls["n"] == 1 + expected_mid_loop_collides


@pytest.mark.parametrize("use_single_state", [True, False], ids=["single_state", "double_state"])
def test_state_force_callback_runs_before_every_solver_substep(monkeypatch, use_single_state):
    """Viewer forces are applied to each current input state before solver stepping."""
    events = []

    class _State:
        def __init__(self, name):
            self.name = name

        def clear_forces(self):
            pass

    manager = _manager_for()
    state_0 = _State("state_0")
    state_1 = _State("state_1")

    manager._newton._state_0 = state_0
    manager._newton._state_1 = state_1
    manager._newton._control = object()
    manager._solver_dt = 0.001
    manager._num_substeps = 2
    manager._collision_decimation = 0
    manager._needs_collision_pipeline = False
    manager._use_single_state = use_single_state
    manager._newton._state_force_callbacks = [lambda state: events.append(("force", state.name))]
    manager._step_solver = lambda state_in, state_out, *_args: events.append(("step", state_in.name, state_out.name))

    manager._run_solver_substeps(contacts=None)
    assert manager._newton._state_0 is state_0

    if use_single_state:
        assert events == [
            ("force", "state_0"),
            ("step", "state_0", "state_0"),
            ("force", "state_0"),
            ("step", "state_0", "state_0"),
        ]
    else:
        assert events == [
            ("force", "state_0"),
            ("step", "state_0", "state_1"),
            ("force", "state_1"),
            ("step", "state_1", "state_0"),
        ]


# ---------------------------------------------------------------------------
# Regression: an env reset written through the data layer must land in the
# manager's canonical _state_0 after an odd number of steps when CUDA graphs
# are disabled (the use_cuda_graph state-swap gating bug).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_steps", [1, 3])
def test_reset_lands_in_state_0_after_odd_kamino_steps_without_cuda_graph(num_steps):
    """An env reset written through the data-layer binding lands in ``_state_0``.

    Kamino is double-buffered (``_use_single_state=False``), so each substep
    ping-pongs ``_state_0`` / ``_state_1``. With a single substep the loop must
    copy the result back into ``_state_0`` instead of swapping, otherwise after
    an *odd* number of steps the canonical ``_state_0`` ends up on the other
    buffer. This copy-on-last was previously gated on ``use_cuda_graph``, so with
    CUDA graphs disabled ``_state_0`` flipped buffers and env-reset writes landed
    in the stale buffer.

    :class:`~isaaclab_newton.assets.ArticulationData` binds its joint-state write
    target to ``_state_0.joint_q`` once at setup (``_sim_bind_joint_pos``) and
    never re-binds on env resets, so a flipped ``_state_0`` makes reset writes
    miss the live state. This test reproduces that contract without a full USD
    articulation: it caches the same ``_state_0.joint_q`` binding, steps Kamino an
    odd number of times, writes a sentinel through the cached binding (mimicking
    the reset write), and asserts the manager's ``_state_0`` observes it.

    Without the fix the swap-on-last flips ``_state_0`` for odd ``num_steps`` and
    the sentinel lands in ``_state_1`` instead, so the final assertion fails.
    """
    sentinel = 1.2345
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=KaminoPADMMSolverCfg(
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )

    with _direct_builder_simulation(sim_cfg) as sim:
        manager = sim._physics_manager
        builder = manager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        manager.set_builder(builder)
        sim.reset()

        # Kamino keeps separate input/output states; the bug only exists there.
        assert manager._use_single_state is False
        # The data layer binds its joint-state write target to _state_0 at setup.
        reset_target = manager._newton._state_0.joint_q
        published_body_q = manager._scene_data_backend.transform_publication.data.transforms
        assert published_body_q is manager._newton._state_0.body_q
        assert reset_target.shape[0] > 0  # guard against a vacuous assertion

        for _ in range(num_steps):
            sim.step(render=False)

        # An env reset writes joint state through the (still bound) target.
        reset_target.fill_(sentinel)

        # The reset must be visible in the manager's canonical _state_0; if the
        # buffer flipped it landed in _state_1 instead.
        canonical_joint_q = manager._newton._state_0.joint_q.numpy()
        assert published_body_q is manager._newton._state_0.body_q
        assert np.allclose(canonical_joint_q, sentinel), (
            f"reset write did not land in _state_0 after {num_steps} steps: {canonical_joint_q}"
        )


def _build_collision_scene(sim, num_boxes=8):
    """Add ``num_boxes`` free-falling boxes over a ground plane.

    Uses ``MJWarpSolverCfg(use_mujoco_contacts=False)`` so the Newton collision
    pipeline / contacts are allocated on ``sim.reset()``.
    """
    builder = sim._physics_manager.create_builder()
    for _ in range(num_boxes):
        body = builder.add_body(mass=1.0)
        builder.add_joint_free(child=body)
        builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    sim._physics_manager.set_builder(builder)


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_hard_reset_preserves_native_identity_then_steps(use_cuda_graph):
    """A second hard reset preserves the cloned resources and remains runnable."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=MJWarpSolverCfg(
            use_mujoco_contacts=False,
            num_substeps=2,
            use_cuda_graph=use_cuda_graph,
        ),
    )

    with _direct_builder_simulation(sim_cfg) as sim:
        _build_collision_scene(sim)
        manager = sim._physics_manager

        sim.reset()
        assert manager._needs_collision_pipeline is True
        sim.step(render=False)
        native = (
            manager._newton._model,
            manager._newton._state_0,
            manager._newton._state_1,
            manager._newton._control,
            manager._solver,
            manager._collision_pipeline,
            manager._graph,
            manager._scene_data_backend.transform_publication.data.transforms,
            manager._scene_data_backend.point_publications["points"].data.points,
        )

        sim.reset()
        after_reset = (
            manager._newton._model,
            manager._newton._state_0,
            manager._newton._state_1,
            manager._newton._control,
            manager._solver,
            manager._collision_pipeline,
            manager._graph,
            manager._scene_data_backend.transform_publication.data.transforms,
            manager._scene_data_backend.point_publications["points"].data.points,
        )
        assert all(before is after for before, after in zip(native, after_reset, strict=True))

        sim.step(render=False)
        wp.synchronize_device("cuda:0")
        backend = manager._scene_data_backend
        assert backend.transform_publication.data.transforms is manager._newton._state_0.body_q
        assert backend.point_publications["points"].data.points is manager._newton._state_0.particle_q
