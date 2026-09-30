# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from isaaclab.test.utils import resolve_test_sim_device, test_devices

# launch omniverse app
simulation_app = AppLauncher(headless=True, device=resolve_test_sim_device()).app

"""Rest everything follows."""

import weakref

import numpy as np
import pytest
from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_physx.physics import PhysxCfg

import omni.timeline

import isaaclab.sim as sim_utils
import isaaclab.sim.simulation_context as simulation_context_module
import isaaclab.sim.utils.stage as stage_utils
from isaaclab import cloner
from isaaclab.physics import PhysicsEvent
from isaaclab.sim import SimulationCfg, SimulationContext

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Setup and teardown for each test."""
    # Setup: Clear any existing simulation context and create a fresh stage
    SimulationContext.clear_instance()
    sim_utils.create_new_stage()

    # Yield for the test
    yield

    SimulationContext.clear_instance()


"""
Basic Configuration Tests
"""


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("device", test_devices())
def test_init(device):
    """Test the simulation context initialization."""
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

    cfg = SimulationCfg(
        physics=PhysxCfg(),
        device=device,
        physics_prim_path="/Physics/PhysX",
        gravity=(0.0, -0.5, -0.5),
        physics_material=RigidBodyMaterialCfg(),
        render_interval=5,
    )
    # sim = SimulationContext(cfg)
    # TODO: Figure out why keyword argument doesn't work.
    # note: added a fix in Isaac Sim 2023.1 for this.
    sim = SimulationContext(cfg=cfg)

    # verify stage is valid
    assert sim.stage is not None
    # verify device property
    assert sim.device == device
    # verify no RTX sensors are available
    assert not sim.get_setting("/isaaclab/render/rtx_sensors")

    # obtain physics scene from USD (string-based schema: physxScene:*)
    from pxr import UsdPhysics

    physics_scene_prim = sim.stage.GetPrimAtPath("/Physics/PhysX")
    assert physics_scene_prim.IsValid()
    assert sim._physics_scene_prim == physics_scene_prim
    physics_scene = UsdPhysics.Scene(physics_scene_prim)
    physics_hz = physics_scene_prim.GetAttribute("physxScene:timeStepsPerSecond").Get()
    assert physics_scene_prim.GetAttribute("physxScene:envIdInBoundsBitCount").Get() == 4
    assert not physics_scene_prim.HasAttribute("physxScene:backend")
    physics_dt = 1.0 / physics_hz
    assert physics_dt == cfg.dt

    # check valid paths
    assert sim.stage.GetPrimAtPath("/Physics/PhysX").IsValid()
    assert sim.stage.GetPrimAtPath("/Physics/PhysX/defaultMaterial").IsValid()
    # check valid gravity
    gravity_dir, gravity_mag = (
        physics_scene.GetGravityDirectionAttr().Get(),
        physics_scene.GetGravityMagnitudeAttr().Get(),
    )
    gravity = np.array(gravity_dir) * gravity_mag
    np.testing.assert_almost_equal(gravity, cfg.gravity)


@pytest.mark.isaacsim_ci
def test_context_creates_kit_owned_stage(monkeypatch) -> None:
    """The context creates the one real stage through Kit when none exists."""
    import omni.usd

    def _reject_pure_stage_creation() -> None:
        pytest.fail("SimulationContext must let Kit create the process-owned stage.")

    stale_stage = sim_utils.get_current_stage()
    stage_utils._context.stage = None
    assert omni.usd.get_context().get_stage() is None
    monkeypatch.setattr(sim_utils, "create_new_stage", _reject_pure_stage_creation)
    monkeypatch.setattr(simulation_context_module, "create_new_stage", _reject_pure_stage_creation)

    with sim_utils.build_simulation_context(sim_cfg=SimulationCfg(physics=PhysxCfg()), create_new_stage=False) as sim:
        assert sim.stage is not stale_stage
        assert sim.stage is sim_utils.get_current_stage()
        assert sim.stage is omni.usd.get_context().get_stage()


@pytest.mark.isaacsim_ci
def test_context_removes_kit_implicit_cameras() -> None:
    """Kit infrastructure cameras never enter the scene's clone plan."""
    from pxr import Usd, UsdGeom

    stage = sim_utils.get_current_stage()
    camera_paths = tuple(f"/OmniverseKit_{name}" for name in ("Persp", "Front", "Top", "Right"))
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path in camera_paths:
            UsdGeom.Camera.Define(stage, path)

    sim = SimulationContext(SimulationCfg(physics=PhysxCfg()))

    assert all(not sim.stage.GetPrimAtPath(path).IsValid() for path in camera_paths)


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    "physics_cfg",
    [PhysxCfg(), MJWarpSolverCfg()],
    ids=["physx", "newton"],
)
def test_stop_is_dispatched_for_lazy_class_type(physics_cfg):
    """``PhysicsEvent.STOP`` must be dispatched even when ``class_type`` is declared lazily.

    Configs declare ``class_type`` as a ``"module:Class"`` string, which proxies attribute access
    but is a ``str``. The active-manager identity check in ``PhysicsManager.close`` therefore
    never matched the class, and ``STOP`` never reached any sensor or asset.
    """
    sim = SimulationContext(SimulationCfg(physics=physics_cfg))

    stopped = []

    def on_stop(_):
        stopped.append(True)

    sim._physics_manager.register_callback(on_stop, PhysicsEvent.STOP, name="test_stop")
    SimulationContext.clear_instance()

    assert stopped, "PhysicsEvent.STOP was not dispatched at teardown"


@pytest.mark.isaacsim_ci
def test_clear_instance_closes_renderers():
    """``clear_instance`` must close registered renderers rather than leave them to garbage collection.

    Renderer state outlives a camera's transient render data, so stage-bound resources cannot be
    released from ``cleanup``. The OVRTX ovstage path holds its stage in a ``contextlib.ExitStack``,
    which has no finalizer, so collection never runs the context managers that own it.
    """
    sim = SimulationContext(SimulationCfg(physics=PhysxCfg()))

    closed = []

    class _Renderer:
        def close(self):
            closed.append(True)

    sim._renderer_entries.append(_Renderer())  # noqa: SLF001
    SimulationContext.clear_instance()

    assert closed, "registered renderers were not closed at teardown"


@pytest.mark.isaacsim_ci
def test_instance_before_creation():
    """Test accessing instance before creating returns None."""
    # clear any existing instance
    SimulationContext.clear_instance()

    # accessing instance before creation should return None
    assert SimulationContext.instance() is None


@pytest.mark.isaacsim_ci
def test_second_construction_is_rejected():
    """A live context must be obtained explicitly instead of reconstructed."""
    sim1 = SimulationContext(sim_utils.SimulationCfg(physics=PhysxCfg()))
    with pytest.raises(RuntimeError, match="SimulationContext already exists"):
        SimulationContext(sim_utils.SimulationCfg(physics=PhysxCfg()))
    assert SimulationContext.instance() is sim1

    # try to delete the singleton
    sim1.clear_instance()
    assert sim1.instance() is None
    # create new instance
    sim3 = SimulationContext(sim_utils.SimulationCfg(physics=PhysxCfg()))
    assert sim1 is not sim3
    assert sim1.instance() is sim3.instance()
    # clear instance
    sim3.clear_instance()


"""
Property Tests.
"""


@pytest.mark.isaacsim_ci
def test_carb_setting():
    """Test setting carb settings."""
    sim = SimulationContext(sim_utils.SimulationCfg(physics=PhysxCfg()))
    # known carb setting
    sim.set_setting("/physics/physxDispatcher", False)
    assert sim.get_setting("/physics/physxDispatcher") is False
    # unknown carb setting
    sim.set_setting("/myExt/test_value", 42)
    assert sim.get_setting("/myExt/test_value") == 42


@pytest.mark.isaacsim_ci
def test_headless_mode():
    """Test that render mode is headless since we are running in headless mode."""
    sim = SimulationContext(sim_utils.SimulationCfg(physics=PhysxCfg()))
    # check default render mode (no GUI and no offscreen rendering)
    assert not sim.has_gui and not sim.has_offscreen_render


"""
Timeline Operations Tests.
"""


@pytest.mark.isaacsim_ci
def test_timeline_play_stop():
    """Test timeline play and stop operations."""
    sim = SimulationContext(sim_utils.SimulationCfg(physics=PhysxCfg()))

    # initially simulation should be stopped
    assert sim.is_stopped()
    assert not sim.is_playing()

    # start the simulation
    sim.play()
    assert sim.is_playing()
    assert not sim.is_stopped()

    # disable callback to prevent app from continuing
    sim._disable_app_control_on_stop_handle = True  # type: ignore
    # stop the simulation
    sim.stop()
    assert sim.is_stopped()
    assert not sim.is_playing()


@pytest.mark.isaacsim_ci
def test_timeline_pause():
    """Test timeline pause operation."""
    sim = SimulationContext(sim_utils.SimulationCfg(physics=PhysxCfg()))

    # start the simulation
    sim.play()
    assert sim.is_playing()

    # pause the simulation
    sim.pause()
    assert not sim.is_playing()
    assert not sim.is_stopped()  # paused is different from stopped


"""
Reset and Step Tests
"""


@pytest.mark.isaacsim_ci
def test_reset():
    """Test simulation reset."""
    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)
    with cloner.ReplicateSession((), num_clones=1, env_spacing=0.0):
        pass

    # reset the simulation
    sim.reset()

    # check that simulation is playing after reset
    assert sim.is_playing()

    # check that physics sim view is created
    assert sim.physics_sim_view is not None


@pytest.mark.isaacsim_ci
def test_reset_soft():
    """Test soft reset (without stopping simulation)."""
    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)
    with cloner.ReplicateSession((), num_clones=1, env_spacing=0.0):
        pass

    # perform initial reset
    sim.reset()
    assert sim.is_playing()

    # perform soft reset
    sim.reset(soft=True)

    # simulation should still be playing
    assert sim.is_playing()


@pytest.mark.isaacsim_ci
def test_forward():
    """Test forward propagation for fabric updates."""
    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)
    with cloner.ReplicateSession((), num_clones=1, env_spacing=0.0):
        pass

    sim.reset()

    # call forward
    sim.forward()

    # should not raise any errors
    assert sim.is_playing()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("render", [True, False])
def test_step(render):
    """Test stepping simulation with and without rendering."""
    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)
    with cloner.ReplicateSession((), num_clones=1, env_spacing=0.0):
        pass

    sim.reset()

    # step with rendering
    for _ in range(10):
        sim.step(render=render)

    # simulation should still be playing
    assert sim.is_playing()


@pytest.mark.isaacsim_ci
def test_render():
    """Test rendering simulation."""
    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)
    with cloner.ReplicateSession((), num_clones=1, env_spacing=0.0):
        pass

    sim.reset()

    # render
    for _ in range(10):
        sim.render()

    # simulation should still be playing
    assert sim.is_playing()


"""
Stage Operations Tests
"""


@pytest.mark.isaacsim_ci
def test_get_initial_stage():
    """Test getting the initial stage."""
    sim = SimulationContext(sim_utils.SimulationCfg(physics=PhysxCfg()))

    # get initial stage
    stage = sim.stage

    # verify stage is valid
    assert stage is not None
    assert stage == sim.stage


@pytest.mark.isaacsim_ci
def test_clear_stage():
    """Test clearing the stage."""
    sim = SimulationContext(sim_utils.SimulationCfg(physics=PhysxCfg()))

    # create some objects
    cube_cfg1 = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg1.func("/World/Cube1", cube_cfg1)
    cube_cfg2 = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg2.func("/World/Cube2", cube_cfg2)

    # verify objects exist
    assert sim.stage.GetPrimAtPath("/World/Cube1").IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cube2").IsValid()

    # clear the stage
    sim.clear_stage()

    # verify objects are removed but World and Physics remain
    assert not sim.stage.GetPrimAtPath("/World/Cube1").IsValid()
    assert not sim.stage.GetPrimAtPath("/World/Cube2").IsValid()
    assert sim.stage.GetPrimAtPath("/World").IsValid()
    assert sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path).IsValid()  # type: ignore[union-attr]


"""
Physics Configuration Tests
"""


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("solver_type", [0, 1])  # 0=PGS, 1=TGS
def test_solver_type(solver_type):
    """Test different solver types."""
    cfg = SimulationCfg(physics=PhysxCfg(solver_type=solver_type))
    sim = SimulationContext(cfg)

    # obtain physics scene from USD (string-based: physxScene:solverType)
    physics_scene_prim = sim.stage.GetPrimAtPath(cfg.physics_prim_path)
    solver_type_str = "PGS" if solver_type == 0 else "TGS"
    assert physics_scene_prim.GetAttribute("physxScene:solverType").Get() == solver_type_str


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("dt", [0.01, 0.02, 0.005])
def test_physics_dt(dt):
    """Test that physics time step is properly configured."""
    cfg = SimulationCfg(physics=PhysxCfg(), dt=dt)
    sim = SimulationContext(cfg)

    # obtain physics scene from USD (string-based: physxScene:timeStepsPerSecond)
    physics_scene_prim = sim.stage.GetPrimAtPath(cfg.physics_prim_path)
    physics_hz = physics_scene_prim.GetAttribute("physxScene:timeStepsPerSecond").Get()
    physics_dt = 1.0 / physics_hz
    assert abs(physics_dt - dt) < 1e-6


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("gravity", [(0.0, 0.0, 0.0), (0.0, 0.0, -9.81), (0.5, 0.5, 0.5)])
def test_custom_gravity(gravity):
    """Test that gravity can be properly set."""
    from pxr import UsdPhysics

    cfg = SimulationCfg(physics=PhysxCfg(), gravity=gravity)
    sim = SimulationContext(cfg)

    # obtain physics scene from USD
    physics_scene_prim = sim.stage.GetPrimAtPath(cfg.physics_prim_path)
    physics_scene = UsdPhysics.Scene(physics_scene_prim)

    gravity_dir, gravity_mag = (
        physics_scene.GetGravityDirectionAttr().Get(),
        physics_scene.GetGravityMagnitudeAttr().Get(),
    )
    actual_gravity = np.array(gravity_dir) * gravity_mag
    np.testing.assert_almost_equal(actual_gravity, cfg.gravity, decimal=6)


"""
Callback Tests.
"""


@pytest.mark.isaacsim_ci
def test_timeline_callbacks_on_play():
    """Test that timeline callbacks are triggered on play event."""
    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)

    # create a simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    # create a flag to track callback execution
    callback_state = {"play_called": False, "stop_called": False}

    # define callback functions
    def on_play_callback(event):
        callback_state["play_called"] = True

    def on_stop_callback(event):
        callback_state["stop_called"] = True

    # register callbacks
    timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
    play_handle = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY),
        lambda event: on_play_callback(event),
        order=20,
    )
    stop_handle = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.STOP),
        lambda event: on_stop_callback(event),
        order=20,
    )

    try:
        # ensure callbacks haven't been called yet
        assert not callback_state["play_called"]
        assert not callback_state["stop_called"]

        # play the simulation - this should trigger play callback
        sim.play()
        assert callback_state["play_called"]
        assert not callback_state["stop_called"]

        # reset flags
        callback_state["play_called"] = False

        # disable app control to prevent hanging
        sim._disable_app_control_on_stop_handle = True  # type: ignore

        # stop the simulation - this should trigger stop callback
        sim.stop()
        assert callback_state["stop_called"]

    finally:
        # cleanup callbacks
        if play_handle is not None:
            play_handle.unsubscribe()
        if stop_handle is not None:
            stop_handle.unsubscribe()


@pytest.mark.isaacsim_ci
def test_timeline_callbacks_with_weakref():
    """Test that timeline callbacks work correctly with weak references (similar to asset_base.py)."""

    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)

    # create a simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    # create a test object that will be weakly referenced
    class CallbackTracker:
        def __init__(self):
            self.play_count = 0
            self.stop_count = 0

        def on_play(self, event):
            self.play_count += 1

        def on_stop(self, event):
            self.stop_count += 1

    # create an instance of the callback tracker
    tracker = CallbackTracker()

    # define safe callback wrapper (similar to asset_base.py pattern)
    def safe_callback(callback_name, event, obj_ref):
        """Safely invoke a callback on a weakly-referenced object."""
        try:
            obj = obj_ref()  # Dereference the weakref
            if obj is not None:
                getattr(obj, callback_name)(event)
        except ReferenceError:
            # Object has been deleted; ignore
            pass

    # register callbacks with weakref
    obj_ref = weakref.ref(tracker)
    timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()

    play_handle = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY),
        lambda event, obj_ref=obj_ref: safe_callback("on_play", event, obj_ref),
        order=20,
    )
    stop_handle = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.STOP),
        lambda event, obj_ref=obj_ref: safe_callback("on_stop", event, obj_ref),
        order=20,
    )

    try:
        # verify callbacks haven't been called
        assert tracker.play_count == 0
        assert tracker.stop_count == 0

        # trigger play event
        sim.play()
        assert tracker.play_count == 1
        assert tracker.stop_count == 0

        # disable app control to prevent hanging
        sim._disable_app_control_on_stop_handle = True  # type: ignore

        # trigger stop event
        sim.stop()
        assert tracker.play_count == 1
        assert tracker.stop_count == 1

        # delete the tracker object
        del tracker

        # trigger events again - callbacks should handle the deleted object gracefully
        sim.play()
        # disable app control again
        sim._disable_app_control_on_stop_handle = True  # type: ignore
        sim.stop()
        # should not raise any errors

    finally:
        # cleanup callbacks
        if play_handle is not None:
            play_handle.unsubscribe()
        if stop_handle is not None:
            stop_handle.unsubscribe()


@pytest.mark.isaacsim_ci
def test_multiple_callbacks_on_same_event():
    """Test that multiple callbacks can be registered for the same event."""
    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)

    # create tracking for multiple callbacks
    callback_counts = {"callback1": 0, "callback2": 0, "callback3": 0}

    def callback1(event):
        callback_counts["callback1"] += 1

    def callback2(event):
        callback_counts["callback2"] += 1

    def callback3(event):
        callback_counts["callback3"] += 1

    # register multiple callbacks for play event
    timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
    handle1 = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY), lambda event: callback1(event), order=20
    )
    handle2 = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY), lambda event: callback2(event), order=21
    )
    handle3 = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY), lambda event: callback3(event), order=22
    )

    try:
        # verify none have been called
        assert all(count == 0 for count in callback_counts.values())

        # trigger play event
        sim.play()

        # all callbacks should have been called
        assert callback_counts["callback1"] == 1
        assert callback_counts["callback2"] == 1
        assert callback_counts["callback3"] == 1

    finally:
        # cleanup all callbacks
        if handle1 is not None:
            handle1.unsubscribe()
        if handle2 is not None:
            handle2.unsubscribe()
        if handle3 is not None:
            handle3.unsubscribe()


@pytest.mark.isaacsim_ci
def test_callback_execution_order():
    """Test that callbacks are executed in the correct order based on priority."""
    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)

    # track execution order
    execution_order = []

    def callback_low_priority(event):
        execution_order.append("low")

    def callback_medium_priority(event):
        execution_order.append("medium")

    def callback_high_priority(event):
        execution_order.append("high")

    # register callbacks with different priorities (lower order = higher priority)
    timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
    handle_high = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY), lambda event: callback_high_priority(event), order=5
    )
    handle_medium = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY), lambda event: callback_medium_priority(event), order=10
    )
    handle_low = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY), lambda event: callback_low_priority(event), order=15
    )

    try:
        # trigger play event
        sim.play()

        # verify callbacks were executed in correct order
        assert len(execution_order) == 3
        assert execution_order[0] == "high"
        assert execution_order[1] == "medium"
        assert execution_order[2] == "low"

    finally:
        # cleanup callbacks
        if handle_high is not None:
            handle_high.unsubscribe()
        if handle_medium is not None:
            handle_medium.unsubscribe()
        if handle_low is not None:
            handle_low.unsubscribe()


@pytest.mark.isaacsim_ci
def test_callback_unsubscribe():
    """Test that unsubscribing callbacks works correctly."""
    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)

    # create callback counter
    callback_count = {"count": 0}

    def on_play_callback(event):
        callback_count["count"] += 1

    # register callback
    timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
    play_handle = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY), lambda event: on_play_callback(event), order=20
    )

    try:
        # trigger play event
        sim.play()
        assert callback_count["count"] == 1

        # stop simulation
        sim._disable_app_control_on_stop_handle = True  # type: ignore
        sim.stop()

        # unsubscribe the callback
        play_handle.unsubscribe()
        play_handle = None

        # trigger play event again
        sim.play()

        # callback should not have been called again (still 1)
        assert callback_count["count"] == 1

    finally:
        # cleanup if needed
        if play_handle is not None:
            play_handle.unsubscribe()


@pytest.mark.isaacsim_ci
def test_pause_event_callback():
    """Test that pause event callbacks are triggered correctly."""
    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)

    # create callback tracker
    callback_state = {"pause_called": False}

    def on_pause_callback(event):
        callback_state["pause_called"] = True

    # register pause callback
    timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
    pause_handle = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PAUSE), lambda event: on_pause_callback(event), order=20
    )

    try:
        # play the simulation first
        sim.play()
        assert not callback_state["pause_called"]

        # pause the simulation
        sim.pause()

        # callback should have been triggered
        assert callback_state["pause_called"]

    finally:
        # cleanup
        if pause_handle is not None:
            pause_handle.unsubscribe()


@pytest.mark.isaacsim_ci
def test_physx_lifecycle_events_are_dispatched_once():
    """PhysX publishes one READY per view and one root deletion during teardown."""
    sim = SimulationContext(SimulationCfg(dt=0.01, physics=PhysxCfg()))
    with cloner.ReplicateSession((), num_clones=1, env_spacing=1.0):
        pass

    ready = []
    deleted = []
    sim._physics_manager.register_callback(lambda payload: ready.append(payload), PhysicsEvent.PHYSICS_READY)
    sim._physics_manager.register_callback(lambda payload: deleted.append(payload), PhysicsEvent.PRIM_DELETION)

    sim.reset()
    sim.reset()
    assert ready == [{}]

    SimulationContext.clear_instance()
    assert deleted == [{"prim_path": "/"}]


# ------------------------------------------------------------------
# render callbacks
# ------------------------------------------------------------------


def test_render_callback_is_invoked_on_render():
    """Callback registered via add_render_callback fires on every render() call."""
    from unittest.mock import MagicMock

    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)

    cb = MagicMock()
    sim.add_render_callback("test_cb", cb)

    sim.render()
    sim.render()

    assert cb.call_count == 2
    cb.assert_called_with(None)

    SimulationContext.clear_instance()


def test_render_callback_ordering():
    """Callbacks fire in ascending order value; lower order fires first."""
    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)

    call_log: list[str] = []
    sim.add_render_callback("second", lambda _: call_log.append("second"), order=10)
    sim.add_render_callback("first", lambda _: call_log.append("first"), order=0)
    sim.add_render_callback("third", lambda _: call_log.append("third"), order=20)

    sim.render()

    assert call_log == ["first", "second", "third"]

    SimulationContext.clear_instance()


def test_render_callback_replace_on_same_name():
    """Re-registering with the same name silently replaces the old callback."""
    from unittest.mock import MagicMock

    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)

    old_cb = MagicMock()
    new_cb = MagicMock()
    sim.add_render_callback("cb", old_cb)
    sim.add_render_callback("cb", new_cb)

    sim.render()

    old_cb.assert_not_called()
    new_cb.assert_called_once_with(None)

    SimulationContext.clear_instance()


def test_remove_render_callback_stops_invocation():
    """remove_render_callback prevents a registered callback from firing."""
    from unittest.mock import MagicMock

    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)

    cb = MagicMock()
    sim.add_render_callback("cb", cb)
    sim.remove_render_callback("cb")

    sim.render()

    cb.assert_not_called()

    SimulationContext.clear_instance()


def test_remove_render_callback_noop_for_unknown_name():
    """remove_render_callback is a no-op when the name was never registered."""
    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)

    sim.remove_render_callback("nonexistent")  # must not raise

    SimulationContext.clear_instance()
