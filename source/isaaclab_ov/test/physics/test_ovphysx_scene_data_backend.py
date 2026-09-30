# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the OVPhysX scene-data backend and manager instance."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# The OVPhysX runtime wheel is optional. Skip gracefully when it is not installed;
# CI jobs that need OVPhysX coverage install it explicitly.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

import isaaclab_ov.physics.ovphysx_manager as ovphysx_manager
from isaaclab_ov.physics import OvPhysxCfg

from isaaclab.cloner import ClonePlan
from isaaclab.cloner.clone_plan import DeformableLayout, RigidBodyLayout
from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

_manager_cfg = OvPhysxCfg()
_manager = _manager_cfg.class_type(_manager_cfg)
assert _manager.cfg is _manager_cfg


def _completed_plan(num_envs: int = 0, **topology) -> ClonePlan:
    env_ids = tuple(range(num_envs))
    return ClonePlan(
        sources=(),
        destinations=(),
        clone_mask=torch.zeros((0, num_envs), dtype=torch.bool),
        env_ids=torch.tensor(env_ids, dtype=torch.long),
        is_complete=True,
        _env_ids_cpu=env_ids,
        **topology,
    )


@pytest.fixture(autouse=True)
def _close_test_views():
    from isaaclab_ov.sim.views import OvPhysxView

    existing_views = set(OvPhysxView._live_views)
    yield
    for view in OvPhysxView._live_views - existing_views:
        view.close()


@pytest.fixture(scope="module", autouse=True)
def _register_ovphysx_schemas_before_test_stages():
    """Register OvPhysX schemas before this module creates any USD stage."""
    _manager._prepare_stage_creation()


def test_manager_forced_rewarm_invalidates_bindings_before_loading(monkeypatch):
    """A forced re-warm invalidates views before replacing their attached stage."""
    from isaaclab.physics import PhysicsEvent

    calls = []
    monkeypatch.setattr(_manager, "_warmup_done", False)
    monkeypatch.setattr(_manager, "_ovstage", object())
    monkeypatch.setattr(_manager, "_scene_data_backend", SimpleNamespace(_invalidate=lambda: None))
    monkeypatch.setattr(_manager, "_warmup_and_load", lambda: calls.append("warmup"))
    monkeypatch.setattr(
        _manager,
        "dispatch_event",
        lambda event, payload=None: calls.append(event),
    )

    _manager.reset()

    assert calls == [PhysicsEvent.STOP, "warmup", PhysicsEvent.PHYSICS_READY]


def test_manager_registers_clone_context_by_type(monkeypatch):
    """OVPhysX registers its simulation-owned clone context by backend type."""
    from isaaclab_ov.cloner import OvReplicateContext

    cfg = OvPhysxCfg()
    manager = cfg.class_type(cfg)
    clone_context = object()
    simulation = SimpleNamespace(
        cfg=SimpleNamespace(device="cpu", gravity=(0.0, 0.0, -9.81)),
        _physics_scene_prim=object(),
        get_or_create_backend=MagicMock(return_value=clone_context),
    )
    monkeypatch.setattr(manager, "_ensure_physx_schemas_registered", lambda: None)
    monkeypatch.setattr(manager, "_configure_physx_scene_prim", lambda *_args: None)

    manager._bind_context(simulation)

    assert manager._clone_ctx is clone_context
    simulation.get_or_create_backend.assert_called_once_with(
        OvReplicateContext,
        simulation,
        clone_role="physics",
    )


@pytest.mark.parametrize(
    ("device", "gpu_index", "expected_cpu_mode", "expected_active_cuda_gpus"),
    [("cpu", 0, True, None), ("gpu", 2, False, "2")],
)
def test_manager_supports_pinned_runtime_api(device, gpu_index, expected_cpu_mode, expected_active_cuda_gpus):
    """The pinned OVPhysX wheel keeps its constructor, step, and reset API."""

    class PinnedPhysX:
        cpu_mode = None

        @classmethod
        def set_cpu_mode(cls, enabled):
            cls.cpu_mode = enabled

        def __init__(self, *, active_cuda_gpus=None, config=None):
            self.constructor = {"active_cuda_gpus": active_cuda_gpus, "config": config}
            self.calls = []

        def step_sync(self, *, dt):
            self.calls.append(("step_sync", dt))

        def reset_stage(self):
            self.calls.append(("reset_stage",))
            return 23

        def wait_op(self, operation):
            self.calls.append(("wait_op", operation))

    runtime = SimpleNamespace(
        PhysX=PinnedPhysX,
        PhysXConfig=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    physx = _manager._create_physx_instance(runtime, device, gpu_index, None)
    _manager._step_physx(physx, dt=0.02)
    _manager._reset_physx_stage(physx)

    assert PinnedPhysX.cpu_mode is expected_cpu_mode
    assert physx.constructor["active_cuda_gpus"] == expected_active_cuda_gpus
    assert physx.constructor["config"].num_threads == 8
    assert physx.calls == [("step_sync", 0.02), ("reset_stage",), ("wait_op", 23)]


def test_manager_updates_fk_only_when_forwarding_without_a_step(monkeypatch):
    """A completed physics step already updates link poses; only write-only forwarding needs FK."""
    manager = OvPhysxCfg().class_type(OvPhysxCfg())
    calls = []
    manager._physx = SimpleNamespace(
        step_sync=lambda *, dt: calls.append(("step", dt)),
        update_articulations_kinematic=lambda: calls.append(("fk",)),
    )
    manager._scene_data_backend = SimpleNamespace(
        _invalidate=lambda **kwargs: calls.append(("invalidate", kwargs)),
    )
    monkeypatch.setattr(manager, "get_physics_dt", lambda: 0.02)

    manager.step()
    manager.forward()

    assert calls == [
        ("step", 0.02),
        ("invalidate", {}),
        ("fk",),
        ("invalidate", {"points": False}),
    ]
    assert manager.get_simulation_time() == pytest.approx(0.02)


def test_manager_attaches_and_releases_owned_ovstage(monkeypatch):
    """The manager owns OVStage from population through PhysX release."""
    events = []

    class FakeWriteFloorOp:
        def __init__(self, ordinal):
            self._ordinal = ordinal

        def wait(self):
            events.append(("seal", self._ordinal))

    class FakeStage:
        def __init__(self, name):
            events.append(("stage", name))

        def advance_write_floor(self, ordinal):
            return FakeWriteFloorOp(ordinal)

        def destroy(self):
            events.append(("destroy",))

    class FakePhysX:
        def attach_ovstage(self, stage, read_ordinal):
            events.append(("attach", stage, read_ordinal))

        def reset_stage(self):
            events.append(("reset",))
            return 17

        def wait_op(self, op):
            events.append(("wait", op))

        def release(self):
            events.append(("release",))

    fake_ovstage = ModuleType("ovstage")
    fake_ovstage.PopulationDomain = SimpleNamespace(ALL="all")
    fake_ovstage.population = SimpleNamespace(
        open_usd_from_string=lambda stage, usda, ordinal, domains: events.append(
            ("populate", stage, usda, ordinal, domains)
        )
    )
    monkeypatch.setitem(sys.modules, "ovstage", fake_ovstage)
    # The manager builds its stage through the shared helper so every stage in the process gets
    # the same ovstage configuration; that is the seam to fake, not ``ovstage.Stage``.
    monkeypatch.setattr(ovphysx_manager, "create_ovstage", FakeStage)

    previous_physx = _manager._physx
    previous_ovstage = _manager._ovstage
    previous_clone_ctx = _manager._clone_ctx
    physx = FakePhysX()
    _manager._physx = physx
    _manager._ovstage = None
    _manager._clone_ctx = SimpleNamespace(stage_usda="#usda 1.0", physics_clone_rows=())
    monkeypatch.setattr(
        _manager,
        "_close_physx_views",
        lambda value: events.append(("close_views", value)),
    )
    try:
        _manager._attach_ovstage()
        stage = _manager._ovstage
        _manager._release_physx()
    finally:
        _manager._physx = previous_physx
        _manager._ovstage = previous_ovstage
        _manager._clone_ctx = previous_clone_ctx

    # The seal must land between population and attach: ovphysx reads sealed data
    # only, so attaching at an unsealed ordinal silently yields an empty scene.
    assert events == [
        ("stage", "isaaclab"),
        ("populate", stage, "#usda 1.0", 1, "all"),
        ("seal", 1),
        ("attach", stage, 1),
        ("close_views", physx),
        ("reset",),
        ("wait", 17),
        ("release",),
        ("destroy",),
    ]


def test_manager_applies_gravity_at_consecutive_sealed_ordinals(monkeypatch):
    """Gravity control writes share one ordinal and advance it only after sealing."""
    events = []

    class Operation:
        def __init__(self, event):
            self.event = event

        def wait(self):
            events.append(("wait", self.event))

    class PathDictionary:
        def __init__(self, stage):
            events.append(("paths", stage))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            events.append(("close_paths",))

        def create_path_list_from_strings(self, paths):
            events.append(("path_list", paths))
            return 17

        def destroy_path_list(self, path_list):
            events.append(("destroy_path_list", path_list))

    class Query:
        def __enter__(self):
            return 23

        def __exit__(self, *_args):
            events.append(("close_query",))

    class Stage:
        def query_from_path_list(self, path_list):
            events.append(("query", path_list))
            return Query()

        def write_attribute(self, query, name, ordinal, values, *, is_array):
            events.append(("write", query, name, ordinal, np.asarray(values).tolist(), is_array))
            return Operation(name)

        def advance_write_floor(self, *, ordinal):
            events.append(("seal", ordinal))
            return Operation("seal")

    class PhysX:
        def update_from_ovstage(self, start_ordinal, end_ordinal):
            events.append(("update", start_ordinal, end_ordinal))

    fake_ovstage = ModuleType("ovstage")
    fake_ovstage.PathDictionary = PathDictionary
    monkeypatch.setitem(sys.modules, "ovstage", fake_ovstage)
    cfg = OvPhysxCfg()
    manager = cfg.class_type(cfg)
    manager._sim = SimpleNamespace(cfg=SimpleNamespace(physics_prim_path="/physicsScene"))
    manager._physx = PhysX()
    manager._ovstage = Stage()

    manager.set_gravity((0.0, -6.0, -8.0))
    manager.set_gravity((0.0, 0.0, 0.0))

    writes = [event for event in events if event[0] == "write"]
    assert [(query, name, ordinal, is_array) for _, query, name, ordinal, _values, is_array in writes] == [
        (23, "physics:gravityDirection", 2, False),
        (23, "physics:gravityMagnitude", 2, False),
        (23, "physics:gravityDirection", 3, False),
        (23, "physics:gravityMagnitude", 3, False),
    ]
    np.testing.assert_allclose(writes[0][4], [[0.0, -0.6, -0.8]])
    assert [write[4] for write in writes[1:]] == [[10.0], [[0.0, 0.0, -1.0]], [0.0]]
    assert [event for event in events if event[0] in ("seal", "update")] == [
        ("seal", 2),
        ("update", 2, 2),
        ("seal", 3),
        ("update", 3, 3),
    ]
    assert events.count(("destroy_path_list", 17)) == events.count(("close_query",)) == 2


def test_manager_destroys_ovstage_when_population_fails(monkeypatch):
    """A failed in-memory population does not leak its OVStage allocation."""
    destroyed = []

    class FakeStage:
        def __init__(self, name):
            self.name = name

        def destroy(self):
            destroyed.append(self.name)

    def fail_population(*args, **kwargs):
        raise RuntimeError("population failed")

    fake_ovstage = ModuleType("ovstage")
    fake_ovstage.PopulationDomain = SimpleNamespace(ALL="all")
    fake_ovstage.population = SimpleNamespace(open_usd_from_string=fail_population)
    monkeypatch.setitem(sys.modules, "ovstage", fake_ovstage)
    monkeypatch.setattr(ovphysx_manager, "create_ovstage", FakeStage)

    previous_ovstage = _manager._ovstage
    previous_clone_ctx = _manager._clone_ctx
    _manager._ovstage = None
    _manager._clone_ctx = SimpleNamespace(stage_usda="#usda 1.0", physics_clone_rows=())
    try:
        with pytest.raises(RuntimeError, match="population failed"):
            _manager._attach_ovstage()
        assert _manager._ovstage is None
    finally:
        _manager._ovstage = previous_ovstage
        _manager._clone_ctx = previous_clone_ctx

    assert destroyed == ["isaaclab"]


def test_manager_keeps_kit_physx_provider_and_registers_deformable_schema(monkeypatch, tmp_path):
    """Keep Kit's PhysX provider while registering the wheel's deformable schema."""

    class FakeRegistry:
        def __init__(self):
            self.get_all_calls = 0
            self.registered_paths = []

        def GetAllPlugins(self):
            self.get_all_calls += 1
            return [SimpleNamespace(name="physxSchema")]

        def RegisterPlugins(self, path):
            self.registered_paths.append(path)

    registry = FakeRegistry()
    fake_pxr = ModuleType("pxr")
    fake_pxr.Plug = SimpleNamespace(Registry=lambda: registry)
    fake_ovphysx = ModuleType("ovphysx")
    deformable_schema_path = tmp_path / "ovphysx" / "plugins" / "usd" / "OmniUsdPhysicsDeformableSchema" / "resources"
    deformable_schema_path.mkdir(parents=True)
    fake_ovphysx.codeless_schema_paths = lambda: [deformable_schema_path]
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    monkeypatch.setitem(sys.modules, "ovphysx", fake_ovphysx)

    previous = _manager._physx_schemas_registered
    _manager._physx_schemas_registered = False
    try:
        _manager._ensure_physx_schemas_registered()
    finally:
        _manager._physx_schemas_registered = previous

    assert registry.get_all_calls == 1
    assert registry.registered_paths == [[str(deformable_schema_path)]]


def test_ovphysx_cfg_is_declarative():
    """Configuration names its implementation without owning construction or lifecycle."""
    cfg = OvPhysxCfg()

    assert cfg.class_type == "isaaclab_ov.physics.ovphysx_manager:OvPhysxManager"
    for method in ("build", "initialize", "_prepare_stage_creation"):
        assert method not in type(cfg).__dict__


def test_ovphysx_manager_registers_schemas_during_pre_stage_setup(monkeypatch):
    """The selected OvPhysX manager registers schemas in its pre-stage hook."""
    calls = []
    monkeypatch.setattr(_manager, "_ensure_physx_schemas_registered", lambda: calls.append(_manager))

    _manager._prepare_stage_creation()

    assert calls == [_manager]


def test_selected_physics_instance_prepares_itself_before_stage_creation(monkeypatch):
    """Simulation construction invokes the selected manager instance before creating the stage."""
    import isaaclab.sim.simulation_context as simulation_context_module
    from isaaclab.sim import SimulationCfg, SimulationContext

    class StageCreationReached(Exception):
        """Signal that initialization reached stage creation."""

    class StubPhysxManager:
        """Stand in for the Kit-only manager while recording its pre-stage hook."""

        def __init__(self, cfg):
            self.cfg = cfg

        def _prepare_stage_creation(self):
            events.append("physx")

    events = []
    monkeypatch.setattr(simulation_context_module, "has_kit", lambda: False)

    def _stop_at_stage_creation():
        events.append("stage")
        raise StageCreationReached

    monkeypatch.setattr(simulation_context_module, "create_new_stage", _stop_at_stage_creation)
    physics_cfg = OvPhysxCfg()
    physics_cfg.class_type = StubPhysxManager
    cfg = SimulationCfg(physics=physics_cfg, create_stage_in_memory=True)

    with pytest.raises(StageCreationReached):
        SimulationContext(cfg)

    assert events == ["physx", "stage"]
    assert SimulationContext.instance() is None


def test_first_transform_request_creates_one_binding_for_exact_declared_rigid_paths():
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxSceneDataBackend

    b = OvPhysxSceneDataBackend()
    paths = [
        "/World/envs/env_0/Robot/cart",
        "/World/envs/env_0/Robot/pole",
        "/World/envs/env_1/Robot/cart",
        "/World/envs/env_1/Robot/pole",
    ]
    clone_mask = np.ones(2, dtype=np.bool_)
    plan = _completed_plan(
        2,
        rigid_body_prototypes=tuple(
            RigidBodyLayout(
                f"/World/envs/env_{{}}/Robot/{name}",
                f"/World/envs/env_*/Robot/{name}",
                0,
                None,
                name,
                clone_mask=clone_mask,
            )
            for name in ("cart", "pole")
        ),
    )
    created: list[SimpleNamespace] = []

    class FakePhysX:
        def create_tensor_binding(self, *, prim_paths, tensor_type):
            binding = SimpleNamespace(
                prim_paths=list(prim_paths),
                tensor_type=tensor_type,
                shape=(4, 7),
                dtype=SimpleNamespace(code=2, bits=32, lanes=1),
                count=4,
                read=lambda dst: None,
                destroy=lambda: None,
            )
            created.append(binding)
            return binding

    b.setup(FakePhysX(), plan, "cpu")

    assert created == []
    assert b.transform_publication.data.transforms.shape == (4,)
    provider = SceneDataProvider(b)
    output = provider.request_transforms(SceneDataFormat.Transform)
    assert len(created) == 1
    assert created[0].prim_paths == paths
    assert provider.request_transforms(SceneDataFormat.Transform) is output
    assert len(created) == 1


def test_transform_request_materializes_once_after_multiple_invalidations():
    import warp as _wp

    _wp.init()

    from isaaclab_ov.physics.ovphysx_manager import OvPhysxSceneDataBackend

    b = OvPhysxSceneDataBackend()
    b.transform_publication.data.transforms = _wp.zeros((3,), dtype=_wp.transformf, device="cpu")
    published_transforms = b.transform_publication.data.transforms
    reads = 0

    def fake_read(name, dst):
        nonlocal reads
        import numpy as np

        reads += 1
        host = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        host = np.vstack([host, [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).astype(np.float32)
        flat_dst = _wp.array(ptr=dst.ptr, shape=(3, 7), dtype=_wp.float32, device="cpu", copy=False)
        _wp.copy(flat_dst, _wp.from_numpy(host, dtype=_wp.float32, device="cpu"))

    b._rigid_view = SimpleNamespace(read_into=fake_read)
    b._point_publication.dirty = False

    b._invalidate(points=False)
    b._invalidate(points=False)
    assert reads == 0
    assert b.transform_publication.dirty
    assert not b.point_publications["points"].dirty

    provider = SceneDataProvider(b)
    out = provider.request_transforms(SceneDataFormat.Transform)
    assert provider.request_transforms(SceneDataFormat.Transform) is out
    assert reads == 1
    assert out.transforms is published_transforms

    merged_host = out.transforms.numpy()  # (3,) of transformf -> view as float32 (3, 7) for assertion
    # Each transformf is 7 floats (pos.xyz + quat.xyzw). Verify row 0 / 1 / 2 contents.
    flat = merged_host.view("<f4").reshape((3, 7))
    assert flat[0, 0] == 1.0
    assert flat[1, 0] == 2.0
    assert flat[2, 0] == 3.0


def test_manager_returns_scene_data_backend_instance():
    """The manager returns its own cached scene-data backend."""
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxSceneDataBackend

    # Reset class state and inject a fresh backend instance.
    _manager._scene_data_backend = OvPhysxSceneDataBackend()
    try:
        out = _manager.get_scene_data_backend()
        assert isinstance(out, OvPhysxSceneDataBackend)
        assert out is _manager._scene_data_backend
    finally:
        _manager._scene_data_backend = None


def test_manager_returns_none_when_backend_uninitialized():
    """Before warmup, ``get_scene_data_backend`` returns the uninitialized ``None``."""
    saved = _manager._scene_data_backend
    _manager._scene_data_backend = None
    try:
        assert _manager.get_scene_data_backend() is None
    finally:
        _manager._scene_data_backend = saved


def test_transform_request_propagates_binding_creation_failure():
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxSceneDataBackend

    b = OvPhysxSceneDataBackend()
    path = "/World/envs/env_0/Robot/cart"
    plan = _completed_plan(rigid_body_prototypes=(RigidBodyLayout(path, path, 0, None, "body"),))

    class FlakyPhysX:
        def create_tensor_binding(self, **kwargs):
            raise RuntimeError("simulated wheel-side failure")

    b.setup(FlakyPhysX(), plan, "cpu")
    with pytest.raises(RuntimeError, match="simulated wheel-side failure"):
        SceneDataProvider(b).request_transforms(SceneDataFormat.Transform)


def test_transform_request_propagates_binding_read_failure():
    import warp as _wp

    _wp.init()

    from isaaclab_ov.physics.ovphysx_manager import OvPhysxSceneDataBackend

    b = OvPhysxSceneDataBackend()
    b._transform_publication.data.transforms = _wp.zeros((1,), dtype=_wp.transformf, device="cpu")

    def bad_read(name, dst):
        raise RuntimeError("simulated read failure")

    b._rigid_view = SimpleNamespace(read_into=bad_read)
    with pytest.raises(RuntimeError, match="simulated read failure"):
        SceneDataProvider(b).request_transforms(SceneDataFormat.Transform)


def test_setup_deformable_bindings_passes_surface_tensor_types(monkeypatch):
    """Surface SceneData views must pass OVPhysX deformable tensor-type kwargs.

    Regression: constructing ``OvPhysxDeformableBodyView`` without
    ``simulation_nodal_position_type`` / ``simulation_element_indices_type``
    left cloth ``point_count`` at 0, so OVRTX wrote identity-xformed rest
    buffers and the Franka cloth disappeared.
    """
    from isaaclab_ov import tensor_types as TT
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxSceneDataBackend

    b = OvPhysxSceneDataBackend()
    captured: dict = {}
    entries = (
        DeformableLayout(
            "/World/envs/env_0/Deformable",
            "/World/envs/env_0/Deformable/sim_mesh",
            "/World/envs/env_0/Deformable/geometry/mesh",
            "/World/envs/env_*/Deformable",
            "surface",
            4,
            4,
            None,
            None,
            0,
            0,
        ),
        DeformableLayout(
            "/World/envs/env_1/Deformable",
            "/World/envs/env_1/Deformable/sim_mesh",
            "/World/envs/env_1/Deformable/geometry/mesh",
            "/World/envs/env_*/Deformable",
            "surface",
            4,
            4,
            None,
            None,
            0,
            1,
        ),
    )

    class _FakeView:
        count = 2
        max_simulation_nodes_per_body = 4
        # Views may report a child mesh; SceneData must publish discovered roots.
        prim_paths = [entries[0].sim_mesh_path, entries[1].sim_mesh_path]

        def __init__(self, physx, **kwargs):
            captured.update(kwargs)

        def read_into(self, tensor_type, dst):
            captured["read_tensor_type"] = tensor_type

    monkeypatch.setattr(
        "isaaclab_ov.assets.deformable_object.views.OvPhysxDeformableBodyView",
        _FakeView,
    )
    b._setup_deformable_bindings(physx=object(), plan=_completed_plan(2, deformables=entries), device="cpu")

    assert captured["simulation_nodal_position_type"] == TT.SURFACE_DEFORMABLE_SIM_POSITION
    assert captured["simulation_element_indices_type"] == TT.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES
    assert TT.SURFACE_DEFORMABLE_SIM_POSITION in captured["tensor_types"]
    assert TT.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES in captured["tensor_types"]
    publication = b.point_publications["points"]
    assert isinstance(publication.data, SceneDataFormat.BodyPoints)
    assert publication.data.points[0].shape == (2, 4)
    assert "read_tensor_type" not in captured
    b._materialize(publication)
    assert captured["read_tensor_type"] == TT.SURFACE_DEFORMABLE_SIM_POSITION


def test_deformable_publication_preserves_plan_order_across_native_views(monkeypatch):
    """Static absolute offsets preserve plan order across type and native-view ordering."""
    import warp as wp
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxSceneDataBackend

    reads = 0
    entries = (
        DeformableLayout(
            "/env_0/Cloth", "/env_0/Cloth/sim", "/env_0/Cloth/vis", "/env_*/Cloth", "surface", 2, 2, None, None, 0, 0
        ),
        DeformableLayout(
            "/env_1/Soft", "/env_1/Soft/sim", "/env_1/Soft/vis", "/env_*/Soft", "volume", 3, 3, None, None, 1, 1
        ),
        DeformableLayout(
            "/env_2/Cloth", "/env_2/Cloth/sim", "/env_2/Cloth/vis", "/env_*/Cloth", "surface", 2, 2, None, None, 0, 2
        ),
    )
    starts = {entries[0].root_path: 10.0, entries[1].root_path: 30.0, entries[2].root_path: 20.0}

    class _FakeView:
        def __init__(self, physx, **kwargs):
            self.prim_paths = list(reversed(kwargs["prim_paths"]))
            self.count = len(self.prim_paths)
            self.max_simulation_nodes_per_body = next(
                entry.vertex_count for entry in entries if entry.root_path == self.prim_paths[0]
            )

        def read_into(self, tensor_type, dst):
            nonlocal reads
            reads += 1
            values = [
                [[starts[path] + index, 0.0, 0.0] for index in range(self.max_simulation_nodes_per_body)]
                for path in self.prim_paths
            ]
            wp.copy(dst, wp.array(values, dtype=wp.vec3f, device="cpu"))

    monkeypatch.setattr("isaaclab_ov.assets.deformable_object.views.OvPhysxDeformableBodyView", _FakeView)
    backend = OvPhysxSceneDataBackend()
    plan = _completed_plan(3, deformables=entries)
    backend._setup_deformable_bindings(object(), plan, "cpu")
    publication = backend.point_publications["points"]
    provider = SceneDataProvider(backend)
    provider._bind_point_plan(plan)

    assert reads == 0
    assert provider.request_points(SceneDataFormat.BodyPoints) is publication.data
    assert reads == len(backend._deformable_reads)
    points = provider.request_points(SceneDataFormat.Points).points.numpy()
    assert reads == len(backend._deformable_reads)
    assert backend.transform_publication.dirty
    assert points[:, 0].tolist() == [10.0, 11.0, 30.0, 31.0, 32.0, 20.0, 21.0]


def test_setup_runs_deformable_bindings_without_rigid_bodies(monkeypatch):
    """Deformable-only scenes must still create SceneData geometry bindings."""
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxSceneDataBackend

    b = OvPhysxSceneDataBackend()
    called: dict[str, object] = {}

    def _fake_setup_deformable_bindings(self, physx, plan, device):
        called["physx"] = physx
        called["plan"] = plan
        called["device"] = device

    monkeypatch.setattr(
        OvPhysxSceneDataBackend,
        "_setup_deformable_bindings",
        _fake_setup_deformable_bindings,
    )

    physx = object()
    plan = _completed_plan()
    b.setup(physx, plan, "cpu")

    assert called == {"physx": physx, "plan": plan, "device": "cpu"}
    assert b.transform_publication.data.transforms is None
