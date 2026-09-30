# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import isaaclab_visualizers.newton.newton_visualization_markers as newton_markers
import isaaclab_visualizers.newton.newton_visualizer as newton_visualizer
import isaaclab_visualizers.rerun.rerun_visualizer as rerun_visualizer
import isaaclab_visualizers.viser.viser_visualizer as viser_visualizer
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_physx.physics import PhysxCfg
from isaaclab_visualizers.kit.kit_visualization_markers import KitVisualizationMarkers
from isaaclab_visualizers.kit.kit_visualizer import KitVisualizer
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonGLVisualizerCfg
from isaaclab_visualizers.rerun.rerun_visualizer_cfg import RerunVisualizerCfg
from isaaclab_visualizers.viser.viser_visualizer_cfg import ViserVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.cloner import ClonePlan
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.math import random_orientation

pytestmark = pytest.mark.integration


def _bare_visualizer(visualizer_type, cfg):
    visualizer = object.__new__(visualizer_type)
    visualizer.cfg = cfg
    if visualizer_type is KitVisualizer:
        visualizer._runtime_headless = cfg.headless
    return visualizer


class _FakeNewtonBackend:
    model = "dummy-model"
    state = {"state": "ok"}
    last_provider = None

    def get_model(self):
        return self.model

    def request_visualization_state(self, provider):
        self.last_provider = provider
        return self.state


def _activate_visualizer_owner(monkeypatch: pytest.MonkeyPatch) -> _FakeNewtonBackend:
    backend = _FakeNewtonBackend()
    context = object.__new__(SimulationContext)
    context.stage = object()
    context.cfg = type("Cfg", (), {"device": "cpu"})()
    context.get_or_create_backend = lambda *args, **kwargs: backend
    monkeypatch.setattr(SimulationContext, "instance", staticmethod(lambda: context))
    return backend


@pytest.fixture
def sim():
    """Create a blank new stage for each test."""
    # Simulation time-step
    dt = 0.01
    # Open a new stage
    sim_utils.create_new_stage()
    # Load kit helper
    sim_context = SimulationContext(SimulationCfg(physics=PhysxCfg(), dt=dt))
    yield sim_context
    # Cleanup
    sim_context._disable_app_control_on_stop_handle = True  # prevent timeout
    sim_context.stop()
    sim_context.clear_instance()
    sim_utils.close_stage()


class _FakeMarkerVisualizer:
    marker_type = newton_markers.NewtonVisualizationMarkers

    def __init__(self, *, enable_markers: bool = True, pumps_app_update: bool = False):
        self.cfg = type("Cfg", (), {"enable_markers": enable_markers})()
        self._pumps_app_update = pumps_app_update

    def pumps_app_update(self):
        return self._pumps_app_update

    def stop(self):
        pass

    def close(self):
        pass


def _construct_marker(sim: SimulationContext, cfg: VisualizationMarkersCfg, visualizer=None):
    """Construct a marker from one explicit plan and visualizer cfg choice."""
    if visualizer is None:
        visualizer = _bare_visualizer(KitVisualizer, KitVisualizerCfg())
    sim._visualizers.append(visualizer)
    try:
        with cloner.ReplicateSession((cfg,), 1, 0.0):
            return cfg.class_type(cfg)
    finally:
        sim._visualizers.remove(visualizer)


def test_kit_marker_env_filter_uses_the_registered_instancer_directly():
    """Kit partial visibility updates its exact registered marker and restores prior authorship."""

    class _Attribute:
        def __init__(self):
            self.value = None
            self.cleared = False

        def HasAuthoredValue(self):
            return False

        def Get(self):
            return self.value

        def Set(self, value):
            self.value = value

        def Clear(self):
            self.cleared = True
            self.value = None

    attr = _Attribute()
    marker = object.__new__(KitVisualizationMarkers)
    marker._count = 4
    marker._invisible_ids_backup = None
    marker._instancer_manager = type("Instancer", (), {"GetInvisibleIdsAttr": lambda _self: attr})()

    marker.set_visible_envs({1, 3}, 4)
    assert list(attr.value) == [0, 2]

    marker.clear_visible_envs()
    assert attr.cleared


def test_instantiation(sim):
    """Marker construction supports a first update without environment ownership."""
    config = VisualizationMarkersCfg(
        prim_path="/World/Visuals/test",
        markers={
            "test": sim_utils.SphereCfg(radius=1.0),
        },
    )
    test_marker = _construct_marker(sim, config)
    test_marker.visualize(translations=np.zeros((1, 3)))
    assert test_marker.num_prototypes == 1


def test_unplanned_marker_cfg_is_inert_without_marker_visualizer(sim):
    """A marker cfg needs no plan entry when no active visualizer can draw it."""
    config = VisualizationMarkersCfg(
        prim_path="/World/Visuals/unplanned",
        markers={"test": sim_utils.SphereCfg(radius=1.0)},
    )

    sim.vis_marker_registry.prepare((), ())
    marker = config.class_type(config)
    marker.visualize(translations=np.zeros((2, 3)))

    assert marker.count == 2
    assert not sim.stage.GetPrimAtPath(config.prim_path).IsValid()


def test_unplanned_marker_cfg_is_rejected_with_marker_visualizer(sim):
    """Marker-capable visualizers accept only plan-owned marker cfgs."""
    config = VisualizationMarkersCfg(
        prim_path="/World/Visuals/unplanned",
        markers={"test": sim_utils.SphereCfg(radius=1.0)},
    )
    sim.vis_marker_registry.prepare((), (newton_markers.NewtonVisualizationMarkers,))
    with pytest.raises(ValueError, match="not covered by the clone plan"):
        config.class_type(config)


def test_rendering_context_authors_visible_usd_point_instancer(sim):
    """Rendering-active contexts should create visible USD marker prims."""
    from pxr import UsdGeom

    sim._has_offscreen_render = True
    config = VisualizationMarkersCfg(
        prim_path="/World/Visuals/rendered_marker",
        markers={
            "failure": sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.15, 0.15)),
                visible=True,
            ),
            "success": sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.25, 0.15)),
                visible=True,
            ),
        },
    )
    test_marker = _construct_marker(sim, config)
    test_marker.visualize(
        translations=torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], device=sim.device),
        marker_indices=torch.tensor([0, 1], device=sim.device),
    )

    stage = sim_utils.get_current_stage()
    instancer_prim = stage.GetPrimAtPath(test_marker.prim_path)
    instancer = UsdGeom.PointInstancer(instancer_prim)

    assert instancer_prim.IsValid()
    assert instancer
    assert UsdGeom.Imageable(instancer_prim).GetVisibilityAttr().Get() != UsdGeom.Tokens.invisible
    assert len(instancer.GetPositionsAttr().Get()) == 2
    assert list(instancer.GetProtoIndicesAttr().Get()) == [0, 1]


def test_environment_ids_author_point_instance_scene_partitions(sim):
    """Per-instance environment IDs should author vertex-interpolated scene-partition tokens."""
    from pxr import Sdf, UsdGeom

    sim._has_offscreen_render = True
    sim.set_setting("/isaaclab/render/rtx_sensors", True)
    stage = sim_utils.get_current_stage()
    for env_id in range(2):
        env_prim = stage.DefinePrim(f"/World/envs/env_{env_id}", "Xform")
        env_prim.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set(f"env_{env_id}")

    config = VisualizationMarkersCfg(
        prim_path="/World/Visuals/partitioned_marker",
        markers={"test": sim_utils.SphereCfg(radius=0.1)},
    )
    test_marker = _construct_marker(sim, config)
    test_marker.visualize(
        translations=torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], device=sim.device),
        environment_ids=torch.tensor([1, 0], device=sim.device),
    )

    instancer_prim = stage.GetPrimAtPath(test_marker.prim_path)
    primvar = UsdGeom.PrimvarsAPI(instancer_prim).GetPrimvar("omni:scenePartition")
    assert primvar
    assert primvar.GetTypeName() == Sdf.ValueTypeNames.TokenArray
    assert primvar.GetInterpolation() == UsdGeom.Tokens.vertex
    assert list(primvar.Get()) == ["env_1", "env_0"]


def test_environment_ids_require_active_scene_partitions(sim):
    """Environment IDs should not partition markers when renderer stage preparation is inactive."""
    from pxr import UsdGeom

    sim._has_offscreen_render = True
    config = VisualizationMarkersCfg(
        prim_path="/World/Visuals/unpartitioned_marker",
        markers={"test": sim_utils.SphereCfg(radius=0.1)},
    )
    test_marker = _construct_marker(sim, config)
    test_marker.visualize(
        translations=torch.tensor([[0.0, 0.0, 0.0]], device=sim.device),
        environment_ids=torch.tensor([0], device=sim.device),
    )

    instancer_prim = sim_utils.get_current_stage().GetPrimAtPath(test_marker.prim_path)
    primvar = UsdGeom.PrimvarsAPI(instancer_prim).GetPrimvar("omni:scenePartition")
    assert not primvar or not primvar.GetAttr().HasAuthoredValueOpinion()


def test_environment_ids_must_match_marker_count(sim):
    """Each marker instance should require one environment ID."""
    sim._has_offscreen_render = True
    config = VisualizationMarkersCfg(
        prim_path="/World/Visuals/mismatched_partition_marker",
        markers={"test": sim_utils.SphereCfg(radius=0.1)},
    )
    test_marker = _construct_marker(sim, config)

    with pytest.raises(ValueError, match="one index per marker"):
        test_marker.visualize(
            translations=torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], device=sim.device),
            environment_ids=torch.tensor([0], device=sim.device),
        )


def test_first_visualize_defaults_to_first_prototype_when_count_matches_prototypes(sim):
    """Omitted marker indices should not preserve initialization prototype placeholders."""
    from pxr import UsdGeom

    sim._has_offscreen_render = True
    config = VisualizationMarkersCfg(
        prim_path="/World/Visuals/default_marker_indices",
        markers={
            "frame": sim_utils.SphereCfg(radius=0.1),
            "line": sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        },
    )
    test_marker = _construct_marker(sim, config)

    test_marker.visualize(translations=torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], device=sim.device))

    instancer = UsdGeom.PointInstancer(sim_utils.get_current_stage().GetPrimAtPath(test_marker.prim_path))
    assert list(instancer.GetProtoIndicesAttr().Get()) == [0, 0]


def test_usd_marker(sim):
    """Test with marker from a USD."""
    # create a marker
    config = FRAME_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/test_frames"
    test_marker = _construct_marker(sim, config)

    # play the simulation
    sim.reset()
    # create a buffer
    num_frames = 0
    # run with randomization of poses
    for count in range(1000):
        # sample random poses
        if count % 50 == 0:
            num_frames = torch.randint(10, 1000, (1,)).item()
            frame_translations = torch.randn(num_frames, 3, device=sim.device)
            frame_rotations = random_orientation(num_frames, device=sim.device)
            # set the marker
            test_marker.visualize(translations=frame_translations, orientations=frame_rotations)
        # update the kit
        sim.step()
        # asset that count is correct
        assert test_marker.count == num_frames


def test_multiple_prototypes_marker(sim):
    """Test with multiple prototypes of spheres."""
    # create a marker
    config = POSITION_GOAL_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/test_protos"
    test_marker = _construct_marker(sim, config)

    # play the simulation
    sim.reset()
    # run with randomization of poses
    for count in range(1000):
        # sample random poses
        if count % 50 == 0:
            num_frames = torch.randint(100, 1000, (1,)).item()
            frame_translations = torch.randn(num_frames, 3, device=sim.device)
            # randomly choose a prototype
            marker_indices = torch.randint(0, test_marker.num_prototypes, (num_frames,), device=sim.device)
            # set the marker
            test_marker.visualize(translations=frame_translations, marker_indices=marker_indices)
        # update the kit
        sim.step()


def test_visualization_skips_updates_when_invisible(sim):
    """When invisible, visualize should not update marker state."""
    # create a marker
    config = POSITION_GOAL_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/test_protos"
    test_marker = _construct_marker(sim, config)

    # play the simulation
    sim.reset()

    # check that visibility is true
    assert test_marker.is_visible()
    frame_translations = torch.randn(4, 3, device=sim.device)
    marker_indices = torch.zeros(4, dtype=torch.int32, device=sim.device)
    test_marker.visualize(translations=frame_translations, marker_indices=marker_indices)
    assert test_marker.count == 4

    # update the kit
    sim.step()
    # make invisible
    test_marker.set_visibility(False)

    # check that visibility is false
    assert not test_marker.is_visible()
    test_marker.visualize(
        translations=torch.randn(8, 3, device=sim.device),
        marker_indices=torch.zeros(8, dtype=torch.int32, device=sim.device),
    )

    assert test_marker.count == 4


def test_newton_marker_backend_registers_and_updates_state_without_frame_capture(sim):
    """Newton marker backend state should be registered and ready for Newton-family viewers."""
    config = POSITION_GOAL_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/newton_marker_state"
    test_marker = _construct_marker(sim, config, _FakeMarkerVisualizer(pumps_app_update=False))
    translations = torch.arange(6, dtype=torch.float32, device=sim.device).reshape(2, 3)
    marker_indices = torch.tensor([0, 0], device=sim.device)

    test_marker.visualize(translations=translations, marker_indices=marker_indices)

    newton_backend = test_marker._backends[0]
    assert isinstance(newton_backend, newton_markers.NewtonVisualizationMarkers)
    assert newton_backend in sim.vis_marker_registry.get_groups()
    assert torch.equal(newton_backend.translations, translations)
    assert torch.equal(newton_backend.marker_indices, marker_indices.to(dtype=torch.int32))
    assert newton_backend.count == 2


def test_newton_visualizer_step_renders_markers(monkeypatch: pytest.MonkeyPatch):
    """NewtonVisualizer.step should ask active Newton marker groups to render."""
    backend = _activate_visualizer_owner(monkeypatch)
    marker_calls = []

    class _FakeViewer:
        _update_frequency = 1

        def __init__(self):
            self.calls = []
            self.show_contacts = False

        def is_paused(self):
            return False

        def is_running(self):
            return True

        def begin_frame(self, sim_time):
            self.calls.append(("begin_frame", sim_time))

        def log_state(self, state):
            self.calls.append(("log_state", state))

        def end_frame(self):
            self.calls.append(("end_frame",))

    class _FakeProvider:
        num_envs = 4

    def _fake_render_markers(viewer, visible_env_ids, num_envs):
        marker_calls.append((viewer, visible_env_ids, num_envs))

    monkeypatch.setattr(newton_visualizer, "render_newton_visualization_markers", _fake_render_markers)

    viewer = _FakeViewer()
    provider = _FakeProvider()
    visualizer = newton_visualizer.NewtonVisualizer(NewtonGLVisualizerCfg(enable_markers=True))
    visualizer._is_initialized = True
    visualizer._is_closed = False
    visualizer._viewer = viewer
    visualizer._scene_data_provider = provider
    visualizer._clone_plan = ClonePlan(
        (), (), np.empty((0, 4), dtype=np.bool_), np.arange(4), is_complete=True, _env_ids_cpu=(0, 1, 2, 3)
    )
    visualizer._resolved_visible_env_ids = [1, 3]

    visualizer.step(0.25)

    assert viewer.calls == [("begin_frame", pytest.approx(0.25)), ("log_state", {"state": "ok"}), ("end_frame",)]
    assert marker_calls == [(viewer, [1, 3], 4)]
    assert backend.last_provider is provider

    def _raise_marker_render(*args, **kwargs):
        raise RuntimeError("marker render failed")

    monkeypatch.setattr(newton_visualizer, "render_newton_visualization_markers", _raise_marker_render)
    with pytest.raises(RuntimeError, match="marker render failed"):
        visualizer.step(0.25)
    assert viewer.calls[-1] == ("end_frame",)


def test_viser_visualizer_marker_failure_propagates_after_ending_frame(monkeypatch: pytest.MonkeyPatch):
    """Viser should close its frame and expose marker rendering failures."""
    backend = _activate_visualizer_owner(monkeypatch)
    marker_calls = []

    class _FakeViewer:
        def __init__(self):
            self.calls = []

        def begin_frame(self, sim_time: float) -> None:
            self.calls.append(("begin_frame", sim_time))

        def log_state(self, state) -> None:
            self.calls.append(("log_state", state))

        def end_frame(self) -> None:
            self.calls.append(("end_frame",))

    class _FakeProvider:
        pass

    def _fake_create_viewer(self, record_to_viser: str | None, metadata: dict | None = None):
        self._viewer = viewer

    def _raise_marker_render(*args, **kwargs):
        marker_calls.append((args, kwargs))
        raise RuntimeError("marker overlay failed")

    provider = _FakeProvider()
    viewer = _FakeViewer()
    monkeypatch.setattr(viser_visualizer.ViserVisualizer, "_create_viewer", _fake_create_viewer)
    monkeypatch.setattr(viser_visualizer, "render_newton_visualization_markers", _raise_marker_render)

    visualizer = viser_visualizer.ViserVisualizer(ViserVisualizerCfg())
    plan = ClonePlan(
        (),
        (),
        np.empty((0, 4), dtype=np.bool_),
        env_ids=np.arange(4),
        is_complete=True,
        _env_ids_cpu=(0, 1, 2, 3),
    )
    visualizer.initialize(provider, plan)

    with pytest.raises(RuntimeError, match="marker overlay failed"):
        visualizer.step(0.25)

    assert marker_calls
    assert viewer.calls == [("begin_frame", pytest.approx(0.25)), ("log_state", {"state": "ok"}), ("end_frame",)]
    assert backend.last_provider is provider


def test_rerun_visualizer_marker_failure_still_ends_frame(monkeypatch: pytest.MonkeyPatch):
    """Rerun should close the frame even if marker rendering raises."""
    backend = _activate_visualizer_owner(monkeypatch)

    class _FakeViewer:
        def __init__(self):
            self.calls = []

        def set_model(self, model):
            pass

        def set_visible_worlds(self, worlds):
            pass

        def set_world_offsets(self, offsets):
            pass

        def is_paused(self):
            return False

        def begin_frame(self, sim_time):
            self.calls.append(("begin_frame", sim_time))

        def log_state(self, state):
            self.calls.append(("log_state", state))

        def end_frame(self):
            self.calls.append(("end_frame",))

        def close(self):
            pass

    class _FakeProvider:
        num_envs = 4

    def _raise_marker_render(*args, **kwargs):
        raise RuntimeError("marker render failed")

    viewer = _FakeViewer()
    monkeypatch.setattr(rerun_visualizer, "NewtonViewerRerun", lambda **kwargs: viewer)
    monkeypatch.setattr(rerun_visualizer, "_ensure_rerun_server", lambda **kwargs: ("rerun+http://test", False))
    monkeypatch.setattr(rerun_visualizer.RerunVisualizer, "_apply_camera_pose", lambda self, pose: None)
    monkeypatch.setattr(rerun_visualizer, "render_newton_visualization_markers", _raise_marker_render)

    visualizer = rerun_visualizer.RerunVisualizer(RerunVisualizerCfg())
    provider = _FakeProvider()
    plan = ClonePlan(
        (),
        (),
        np.empty((0, 4), dtype=np.bool_),
        env_ids=np.arange(4),
        is_complete=True,
        _env_ids_cpu=(0, 1, 2, 3),
    )
    visualizer.initialize(provider, plan)

    with pytest.raises(RuntimeError, match="marker render failed"):
        visualizer.step(0.25)

    assert backend.last_provider is provider
    assert [call[0] for call in viewer.calls] == ["begin_frame", "log_state", "end_frame"]


def test_newton_marker_mesh_registration_is_per_viewer(monkeypatch: pytest.MonkeyPatch):
    marker = object.__new__(newton_markers.NewtonVisualizationMarkers)
    marker._registered_meshes = set()

    class _FakeMesh:
        vertices = np.zeros((1, 3), dtype=np.float32)
        indices = np.zeros((3,), dtype=np.int32)
        normals = np.zeros((0, 3), dtype=np.float32)
        uvs = np.zeros((0, 2), dtype=np.float32)

    class _FakeViewer:
        def __init__(self):
            self.meshes = []

        def log_mesh(self, name, vertices, indices, **kwargs):
            self.meshes.append((name, vertices, indices, kwargs))

    monkeypatch.setattr(newton_markers, "_create_mesh", lambda cfg: _FakeMesh())
    monkeypatch.setattr(newton_markers.wp, "array", lambda value, dtype=None: value)

    spec = newton_markers._NewtonMarkerSpec(renderer="mesh", mesh_type="box", mesh_params={"size": (1.0, 1.0, 1.0)})
    viewer_a = _FakeViewer()
    viewer_b = _FakeViewer()

    marker._ensure_mesh_registered(viewer_a, "/Visuals/marker/meshes/arrow", spec)
    marker._ensure_mesh_registered(viewer_a, "/Visuals/marker/meshes/arrow", spec)
    marker._ensure_mesh_registered(viewer_b, "/Visuals/marker/meshes/arrow", spec)

    assert len(viewer_a.meshes) == 1
    assert len(viewer_b.meshes) == 1


class _FakeNewtonMarkerMesh:
    vertices = np.zeros((1, 3), dtype=np.float32)
    indices = np.zeros((3,), dtype=np.int32)
    normals = np.zeros((0, 3), dtype=np.float32)
    uvs = np.zeros((0, 2), dtype=np.float32)


_NEWTON_MARKER_SPECS = {
    "arrow": newton_markers._NewtonMarkerSpec(
        renderer="mesh",
        mesh_type="box",
        mesh_params={"size": (1.0, 1.0, 1.0)},
        color=(1.0, 1.0, 1.0),
        texture=np.zeros((2, 2, 3), dtype=np.uint8),
    ),
    "sphere": newton_markers._NewtonMarkerSpec(renderer="mesh", mesh_type="sphere", mesh_params={"radius": 1.0}),
    "frame": newton_markers._NewtonMarkerSpec(renderer="frame"),
}


class _FakeNewtonMarkerViewer:
    def __init__(self, world_offsets):
        self.world_offsets = world_offsets
        self.meshes = []
        self.instances = []
        self.lines = []

    def log_mesh(self, name, vertices, indices, **kwargs):
        self.meshes.append((name, vertices, indices, kwargs))

    def log_instances(self, batch_name, mesh_name, xforms, scales, colors, materials, hidden=False):
        self.instances.append(
            {
                "batch_name": batch_name,
                "mesh_name": mesh_name,
                "xforms": xforms,
                "scales": scales,
                "colors": colors,
                "materials": materials,
                "hidden": hidden,
            }
        )

    def log_lines(self, batch_name, starts, ends, colors, width=None, hidden=False):
        self.lines.append(
            {
                "batch_name": batch_name,
                "starts": starts,
                "ends": ends,
                "colors": colors,
                "width": width,
                "hidden": hidden,
            }
        )


def _make_newton_marker_for_render(
    *,
    marker_names: list[str],
    translations: torch.Tensor,
    marker_indices: torch.Tensor | None = None,
    visible: bool = True,
):
    marker = object.__new__(newton_markers.NewtonVisualizationMarkers)
    marker_cfg_type = type("MarkerCfg", (), {"visual_material": None})
    marker.cfg = type("Cfg", (), {"markers": {name: marker_cfg_type() for name in marker_names}})()
    marker.group_id = "/Visuals/marker::test"
    marker.visible = visible
    marker.translations = translations
    marker.orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32).repeat(translations.shape[0], 1)
    marker.scales = torch.ones((translations.shape[0], 3), dtype=torch.float32)
    marker.marker_indices = marker_indices
    marker.count = translations.shape[0]
    marker._registered_meshes = set()
    marker._warned_unsupported = set()
    marker._marker_specs = {name: _NEWTON_MARKER_SPECS[name] for name in marker_names}
    return marker


def _patch_newton_marker_render_deps(
    monkeypatch: pytest.MonkeyPatch, world_offsets: np.ndarray | None = None, world_offsets_device: str = "cpu"
):
    if world_offsets is None:
        world_offsets = np.zeros((4, 3), dtype=np.float32)
    warp_world_offsets = wp.array(world_offsets, dtype=wp.vec3, device=world_offsets_device)

    monkeypatch.setattr(newton_markers, "_create_mesh", lambda cfg: _FakeNewtonMarkerMesh())
    monkeypatch.setattr(newton_markers.wp, "array", lambda value, dtype=None: value)
    monkeypatch.setattr(newton_markers.wp, "from_torch", lambda value, dtype=None: value.detach().cpu().numpy())
    return warp_world_offsets


def test_newton_marker_partial_update_preserves_prototype_indices():
    marker = _make_newton_marker_for_render(
        marker_names=["arrow", "sphere"],
        translations=torch.zeros((4, 3), dtype=torch.float32),
        marker_indices=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
    )
    expected_indices = marker.marker_indices

    marker.visualize(
        translations=torch.ones((4, 3), dtype=torch.float32),
        orientations=None,
        scales=None,
        marker_indices=None,
    )

    assert marker.marker_indices is expected_indices


def test_newton_marker_render_filters_visible_envs(monkeypatch: pytest.MonkeyPatch):
    world_offsets = _patch_newton_marker_render_deps(monkeypatch)
    translations = torch.arange(8, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow"],
        translations=translations,
        marker_indices=torch.zeros(8, dtype=torch.int32),
    )
    viewer = _FakeNewtonMarkerViewer(world_offsets)

    marker.render(viewer, visible_env_ids=[1, 3], num_envs=4)

    assert len(viewer.instances) == 1
    assert viewer.instances[0]["hidden"] is False
    assert viewer.instances[0]["xforms"][:, 0].tolist() == [2.0, 3.0, 6.0, 7.0]


@pytest.mark.parametrize(
    ("visible_env_ids", "expected"),
    [
        ([1, 3], [12.0, 13.0, 36.0, 37.0]),
        (None, [0.0, 1.0, 12.0, 13.0, 24.0, 25.0, 36.0, 37.0]),
    ],
)
def test_newton_marker_render_applies_world_offsets(
    monkeypatch: pytest.MonkeyPatch, visible_env_ids: list[int] | None, expected: list[float]
):
    world_offsets_device = "cuda:0" if wp.is_cuda_available() else "cpu"
    world_offsets = _patch_newton_marker_render_deps(
        monkeypatch,
        np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        world_offsets_device=world_offsets_device,
    )
    translations = torch.arange(8, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow"],
        translations=translations,
        marker_indices=torch.zeros(8, dtype=torch.int32),
    )
    viewer = _FakeNewtonMarkerViewer(world_offsets)

    marker.render(viewer, visible_env_ids=visible_env_ids, num_envs=4)

    assert viewer.instances[0]["xforms"][:, 0].tolist() == expected


def test_newton_marker_render_preserves_reordered_env_state(monkeypatch: pytest.MonkeyPatch):
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 200.0, 300.0],
            [200.0, 400.0, 600.0],
            [300.0, 600.0, 900.0],
        ],
        dtype=np.float32,
    )
    world_offsets = _patch_newton_marker_render_deps(monkeypatch, offsets)
    translations = torch.arange(24, dtype=torch.float32).reshape(3, 8).T
    orientations = torch.arange(32, dtype=torch.float32).reshape(4, 8).T
    scales = torch.arange(1, 25, dtype=torch.float32).reshape(3, 8).T
    marker = _make_newton_marker_for_render(
        marker_names=["arrow"],
        translations=translations,
        marker_indices=torch.zeros(8, dtype=torch.int32),
    )
    marker.orientations = orientations
    marker.scales = scales
    source_state = (
        marker.translations,
        marker.orientations,
        marker.scales,
        marker.marker_indices,
    )
    source_values = tuple(value.clone() for value in source_state)
    viewer = _FakeNewtonMarkerViewer(world_offsets)

    selections = (
        ([3, 1], torch.tensor([6, 7, 2, 3])),
        ([1, 3], torch.tensor([2, 3, 6, 7])),
    )

    monkeypatch.setattr(
        newton_markers.torch,
        "arange",
        lambda *args, **kwargs: pytest.fail("render should not construct environment index tensors"),
    )

    for visible_env_ids, expected_indices in selections:
        selected_offsets = torch.from_numpy(offsets[np.repeat(visible_env_ids, 2)])
        expected_positions = translations[expected_indices] + selected_offsets
        expected_xforms = torch.cat((expected_positions, orientations[expected_indices]), dim=1)
        expected_scales = scales[expected_indices]
        marker.render(viewer, visible_env_ids=visible_env_ids, num_envs=4)
        call = viewer.instances[-1]
        assert call["hidden"] is False
        np.testing.assert_allclose(call["xforms"], expected_xforms.numpy(), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(call["scales"], expected_scales.numpy(), rtol=0.0, atol=0.0)

    current_state = (
        marker.translations,
        marker.orientations,
        marker.scales,
        marker.marker_indices,
    )
    for current, source, expected in zip(current_state, source_state, source_values):
        assert current is source
        assert torch.equal(current, expected)
    assert len(viewer.instances) == 2


def test_newton_marker_render_hides_empty_env_selection(monkeypatch: pytest.MonkeyPatch):
    world_offsets = _patch_newton_marker_render_deps(monkeypatch)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow"],
        translations=torch.zeros((8, 3), dtype=torch.float32),
    )
    viewer = _FakeNewtonMarkerViewer(world_offsets)

    marker.render(viewer, visible_env_ids=[], num_envs=4)

    assert len(viewer.instances) == 1
    assert viewer.instances[0]["hidden"] is True


def test_newton_marker_render_defaults_to_first_prototype(monkeypatch: pytest.MonkeyPatch):
    world_offsets = _patch_newton_marker_render_deps(monkeypatch)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow", "sphere"],
        translations=torch.zeros((4, 3), dtype=torch.float32),
    )
    marker.orientations = None
    marker.scales = None
    viewer = _FakeNewtonMarkerViewer(world_offsets)

    marker.render(viewer, visible_env_ids=None, num_envs=4)

    visible_instances = [call for call in viewer.instances if not call["hidden"]]
    assert len(visible_instances) == 1
    assert visible_instances[0]["batch_name"] == "/Visuals/marker::test/arrow"
    assert visible_instances[0]["xforms"][:, 3:].tolist() == [[0.0, 0.0, 0.0, 1.0]] * 4
    assert visible_instances[0]["scales"].tolist() == [[1.0, 1.0, 1.0]] * 4
    hidden_batches = [call["batch_name"] for call in viewer.instances if call["hidden"]]
    assert hidden_batches == ["/Visuals/marker::test/sphere"]


def test_newton_marker_render_keeps_global_batch_unmodified(monkeypatch: pytest.MonkeyPatch):
    world_offsets = _patch_newton_marker_render_deps(
        monkeypatch,
        np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    translations = torch.arange(3, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow"],
        translations=translations,
        marker_indices=torch.zeros(3, dtype=torch.int32),
    )
    viewer = _FakeNewtonMarkerViewer(world_offsets)

    marker.render(viewer, visible_env_ids=[1, 3], num_envs=4)

    assert viewer.instances[0]["xforms"][:, 0].tolist() == [0.0, 1.0, 2.0]


def test_newton_marker_render_routes_instances_by_prototype(monkeypatch: pytest.MonkeyPatch):
    world_offsets = _patch_newton_marker_render_deps(monkeypatch)
    translations = torch.arange(4, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow", "sphere"],
        translations=translations,
        marker_indices=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
    )
    viewer = _FakeNewtonMarkerViewer(world_offsets)

    marker.render(viewer, visible_env_ids=None, num_envs=4)

    visible_instances = [call for call in viewer.instances if not call["hidden"]]
    assert [call["batch_name"] for call in visible_instances] == [
        "/Visuals/marker::test/arrow",
        "/Visuals/marker::test/sphere",
    ]
    assert [call["xforms"].shape[0] for call in visible_instances] == [2, 2]
    assert visible_instances[0]["materials"][:, 3].tolist() == [1.0, 1.0]
    assert visible_instances[1]["materials"][:, 3].tolist() == [0.0, 0.0]


def test_newton_marker_render_hides_unselected_prototypes(monkeypatch: pytest.MonkeyPatch):
    world_offsets = _patch_newton_marker_render_deps(monkeypatch)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow", "sphere", "frame"],
        translations=torch.zeros((3, 3), dtype=torch.float32),
        marker_indices=torch.zeros(3, dtype=torch.int32),
    )
    viewer = _FakeNewtonMarkerViewer(world_offsets)

    marker.render(viewer, visible_env_ids=None, num_envs=3)

    hidden_instances = [call for call in viewer.instances if call["hidden"]]
    assert [call["batch_name"] for call in hidden_instances] == ["/Visuals/marker::test/sphere"]
    assert viewer.lines == [
        {
            "batch_name": "/Visuals/marker::test/frame",
            "starts": None,
            "ends": None,
            "colors": None,
            "width": None,
            "hidden": True,
        }
    ]
