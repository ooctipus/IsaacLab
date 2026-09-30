# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

pytestmark = pytest.mark.integration


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    # articulation
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
        ),
        actuators={
            "joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=1.0),
        },
    )
    # rigid object
    rigid_obj = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RigidObj",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
    )


@configclass
class DebugMarkerSceneCfg(InteractiveSceneCfg):
    """Scene with one sensor-owned debug-marker family."""

    contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Body", debug_vis=True)


@pytest.fixture
def setup_scene(request):
    """Create simulation context with the specified device."""
    device = request.getfixturevalue("device")
    with build_simulation_context(
        sim_cfg=sim_utils.SimulationCfg(physics=PhysxCfg()),
        device=device,
    ) as sim:
        sim._app_control_on_stop_handle = None

        def make_scene(num_envs: int, env_spacing: float = 1.0):
            scene_cfg = MySceneCfg(num_envs=num_envs, env_spacing=env_spacing)
            return scene_cfg

        yield make_scene, sim
    # Note: cleanup is handled by build_simulation_context's finally block


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_relative_flag(device, setup_scene):
    make_scene, sim = setup_scene
    scene_cfg = make_scene(num_envs=4)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # test relative == False produces different result than relative == True
    assert_state_different(scene.get_state(is_relative=False), scene.get_state(is_relative=True))

    # test is relative == False
    prev_state = scene.get_state(is_relative=False)
    joint_pos = torch.rand_like(scene["robot"].data.joint_pos.torch)
    joint_vel = torch.rand_like(scene["robot"].data.joint_pos.torch)
    scene["robot"].write_joint_position_to_sim_index(position=joint_pos)
    scene["robot"].write_joint_velocity_to_sim_index(velocity=joint_vel)
    next_state = scene.get_state(is_relative=False)
    assert_state_different(prev_state, next_state)
    scene.reset_to(prev_state, is_relative=False)
    assert_state_equal(prev_state, scene.get_state(is_relative=False))

    # test is relative == True
    prev_state = scene.get_state(is_relative=True)
    joint_pos = torch.rand_like(scene["robot"].data.joint_pos.torch)
    joint_vel = torch.rand_like(scene["robot"].data.joint_pos.torch)
    scene["robot"].write_joint_position_to_sim_index(position=joint_pos)
    scene["robot"].write_joint_velocity_to_sim_index(velocity=joint_vel)
    next_state = scene.get_state(is_relative=True)
    assert_state_different(prev_state, next_state)
    scene.reset_to(prev_state, is_relative=True)
    assert_state_equal(prev_state, scene.get_state(is_relative=True))


def test_relative_deformable_state():
    env_origins = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    nodal_position = torch.arange(60, dtype=torch.float32).reshape(4, 5, 3)
    nodal_velocity = torch.zeros_like(nodal_position)
    written_state = {}
    deformable = SimpleNamespace(
        data=SimpleNamespace(
            nodal_pos_w=SimpleNamespace(torch=nodal_position),
            nodal_vel_w=SimpleNamespace(torch=nodal_velocity),
        ),
        write_nodal_pos_to_sim=lambda value, env_ids: written_state.update(position=value, env_ids=env_ids),
        write_nodal_velocity_to_sim=lambda value, env_ids: written_state.update(velocity=value),
    )
    scene = SimpleNamespace(
        device="cpu",
        env_origins=env_origins,
        _articulations={},
        _cable_objects={},
        _deformable_objects={"object": deformable},
        _rigid_objects={},
        _surface_grippers={},
        _rigid_object_collections={},
        write_data_to_sim=lambda: None,
    )

    state = InteractiveScene.get_state(scene, is_relative=True)

    torch.testing.assert_close(
        state["deformable_object"]["object"]["nodal_position"], nodal_position - env_origins[:, None, :]
    )

    env_ids = torch.tensor([3, 1])
    reset_nodal_position = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
    reset_nodal_velocity = torch.ones_like(reset_nodal_position)
    reset_state = {
        "deformable_object": {
            "object": {
                "nodal_position": reset_nodal_position,
                "nodal_velocity": reset_nodal_velocity,
            },
        }
    }

    InteractiveScene.reset_to(scene, reset_state, env_ids=env_ids, is_relative=True)

    torch.testing.assert_close(written_state["position"], reset_nodal_position + env_origins[env_ids, None, :])
    torch.testing.assert_close(written_state["velocity"], reset_nodal_velocity)
    torch.testing.assert_close(written_state["env_ids"], env_ids)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_reset_to_env_ids_input_types(device, setup_scene):
    make_scene, sim = setup_scene
    scene_cfg = make_scene(num_envs=4)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # test env_ids = None
    prev_state = scene.get_state()
    joint_pos = torch.rand_like(scene["robot"].data.joint_pos.torch)
    joint_vel = torch.rand_like(scene["robot"].data.joint_pos.torch)
    scene["robot"].write_joint_position_to_sim_index(position=joint_pos)
    scene["robot"].write_joint_velocity_to_sim_index(velocity=joint_vel)
    scene.reset_to(prev_state, env_ids=None)
    assert_state_equal(prev_state, scene.get_state())

    # test env_ids = torch tensor
    joint_pos = torch.rand_like(scene["robot"].data.joint_pos.torch)
    joint_vel = torch.rand_like(scene["robot"].data.joint_pos.torch)
    scene["robot"].write_joint_position_to_sim_index(position=joint_pos)
    scene["robot"].write_joint_velocity_to_sim_index(velocity=joint_vel)
    scene.reset_to(prev_state, env_ids=torch.arange(scene.num_envs, device=scene.device, dtype=torch.int32))
    assert_state_equal(prev_state, scene.get_state())


def test_scene_publishes_plan_before_replicate(monkeypatch: pytest.MonkeyPatch):
    """A cfg-driven scene publishes the exact plan it forwards to replication."""
    import isaaclab.cloner.replicate_session as replicate_session_module

    captured: list = []

    def fake_replicate(plan, *, replicate_physics=True):
        captured.append((plan, replicate_physics, sim_utils.SimulationContext.instance().get_clone_plan()))
        return plan

    monkeypatch.setattr(replicate_session_module, "replicate", fake_replicate)

    with build_simulation_context(
        sim_cfg=sim_utils.SimulationCfg(physics=PhysxCfg()),
        device="cpu",
    ) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(MySceneCfg(num_envs=4, env_spacing=1.0))
        root_layer_identifier = scene.stage.GetRootLayer().identifier

    assert len(captured) == 1
    plan, replicate_physics, published = captured[0]
    assert published is plan
    assert plan.sources == (
        "/World/envs/env_0/Robot",
        "/World/envs/env_0/RigidObj",
        "/World/defaultGroundPlane",
    )
    assert plan.destinations == (
        "/World/envs/env_{}/Robot",
        "/World/envs/env_{}/RigidObj",
        "/World/defaultGroundPlane",
    )
    assert plan.clone_mask.shape == (3, 4)
    assert plan.root_layer_identifier == root_layer_identifier
    assert replicate_physics is True


def test_scene_collects_enabled_nested_debug_markers():
    """A scene participant's enabled markers join its flat clone manifest."""
    scene = object.__new__(InteractiveScene)
    scene.cfg = DebugMarkerSceneCfg(num_envs=2, env_spacing=1.0)
    scene._env_fmt = scene.cfg.clone_cfg.clone_template

    cfgs, _ = scene._collect_asset_cfgs()
    plan = cloner.make_clone_plan(cfgs, num_clones=scene.cfg.num_envs, env_spacing=scene.cfg.env_spacing)

    marker_cfgs = (
        scene.cfg.contact.visualizer_cfg,
        scene.cfg.contact.normal_force_visualizer_cfg,
        scene.cfg.contact.friction_force_visualizer_cfg,
    )
    assert all(id(cfg) in plan.cfg_rows for cfg in marker_cfgs)

    scene.cfg.contact.debug_vis = False
    cfgs, _ = scene._collect_asset_cfgs()
    planned_ids = {id(cfg) for cfg in cfgs}
    assert all(id(cfg) not in planned_ids for cfg in marker_cfgs)


def test_empty_scene_owns_clone_lifecycle():
    """Even an empty InteractiveScene publishes its one plan and authors every environment root."""
    with build_simulation_context(
        sim_cfg=sim_utils.SimulationCfg(physics=PhysxCfg()),
        device="cpu",
    ) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(InteractiveSceneCfg(num_envs=4, env_spacing=1.0))

        plan = sim.get_clone_plan()
        assert plan is not None
        assert plan.sources == ()
        env_template = scene.cfg.clone_cfg.clone_template
        grid_positions = cloner.grid_transforms(4, 1.0)[0]
        torch.testing.assert_close(scene.env_origins, torch.from_numpy(grid_positions))
        assert all(scene.stage.GetPrimAtPath(env_template.format(i)).IsValid() for i in range(4))


def test_clone_plan_from_env_0_supports_standalone_composition():
    """A tool without InteractiveScene may explicitly publish, construct, and replicate one plan."""
    with build_simulation_context(
        sim_cfg=sim_utils.SimulationCfg(physics=PhysxCfg()),
        device="cpu",
    ) as sim:
        sim._app_control_on_stop_handle = None
        clone_cfg = cloner.CloneCfg()
        env_template = clone_cfg.clone_template
        cube_cfg = RigidObjectCfg(
            prim_path=f"{env_template.format('[^/]+')}/Cube",
            spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        )
        positions = cloner.grid_transforms(4, 1.0)[0] + np.asarray((0.25, 0.5, 0.75), dtype=np.float32)
        plan = cloner.clone_plan_from_env_0(clone_cfg, (cube_cfg,), 4, 1.0, positions=positions)
        cube_cfg.class_type(cube_cfg)
        plan = cloner.replicate(plan)

        assert sim.get_clone_plan() is plan
        np.testing.assert_allclose(plan.positions, positions)


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("replicate_physics", [True, False])
def test_replicate_physics_flag_controls_physx_replicator(device, replicate_physics, setup_scene, monkeypatch):
    """replicate_physics=False must not register the PhysX replicator while envs still simulate.

    The True case asserts the spy actually intercepts registration, so the False case
    cannot pass vacuously.
    """
    physx_replicate_module = pytest.importorskip("isaaclab_physx.cloner.replicate")

    register_calls: list = []
    real_get_iface = physx_replicate_module.get_physx_replicator_interface

    class SpyInterface:
        def __init__(self, real):
            self._real = real

        def register_replicator(self, *args, **kwargs):
            register_calls.append(args)
            return self._real.register_replicator(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    monkeypatch.setattr(
        physx_replicate_module, "get_physx_replicator_interface", lambda: SpyInterface(real_get_iface())
    )

    make_scene, sim = setup_scene
    scene_cfg = make_scene(num_envs=3)
    scene_cfg.replicate_physics = replicate_physics
    scene = InteractiveScene(scene_cfg)
    if not scene.physics_backend.startswith("physx"):
        pytest.skip("PhysX replicator flag is only meaningful on a PhysX backend.")
    sim.reset()

    if replicate_physics:
        assert len(register_calls) > 0
    else:
        assert register_calls == []
    # all environments exist and simulate on both paths
    assert scene["rigid_obj"].data.root_pos_w.torch.shape[0] == 3
    assert scene["robot"].data.joint_pos.torch.shape[0] == 3
    for _ in range(2):
        sim.step()
        scene.update(sim.get_physics_dt())
    assert torch.isfinite(scene["rigid_obj"].data.root_pos_w.torch).all()
    assert torch.isfinite(scene["robot"].data.joint_pos.torch).all()


def test_collect_asset_cfgs_flattens_collections_at_the_composition_root():
    """InteractiveScene gives the cloner a flat, explicit cfg manifest."""
    scene = object.__new__(InteractiveScene)
    cube_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
    )
    shape_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shape",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[sim_utils.ConeCfg(radius=0.1, height=0.2), sim_utils.SphereCfg(radius=0.1)]
        ),
    )
    scene.cfg = SimpleNamespace(
        num_envs=2,
        clone_cfg=cloner.CloneCfg(),
        objects=RigidObjectCollectionCfg(rigid_objects={"cube": cube_cfg, "shape": shape_cfg}),
        ground=AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg()),
    )
    scene._env_fmt = scene.cfg.clone_cfg.clone_template

    cfgs, valid_set = scene._collect_asset_cfgs()

    prim_paths = sorted(c.prim_path for c in cfgs)
    assert prim_paths == ["/World/Ground", "{ENV_REGEX_NS}/Cube", "{ENV_REGEX_NS}/Shape"]
    assert valid_set is None


def test_asset_cfgs_do_not_override_backend_clone_dispatch():
    """Clone backends belong to the simulation registry, not individual assets."""
    scene_cfg = MySceneCfg(num_envs=2, env_spacing=1.0)
    assert "cloning_contexts" not in scene_cfg.rigid_obj.__dataclass_fields__
    assert "cloning_contexts" not in scene_cfg.robot.__dataclass_fields__


def assert_state_equal(s1: dict, s2: dict, path=""):
    """
    Recursively assert that s1 and s2 have the same nested keys
    and that every tensor leaf is exactly equal.
    """
    assert set(s1.keys()) == set(s2.keys()), f"Key mismatch at {path}: {s1.keys()} vs {s2.keys()}"
    for k in s1:
        v1, v2 = s1[k], s2[k]
        subpath = f"{path}.{k}" if path else k
        if isinstance(v1, dict):
            assert isinstance(v2, dict), f"Type mismatch at {subpath}"
            assert_state_equal(v1, v2, path=subpath)
        else:
            # leaf: should be a torch.Tensor
            assert isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor), f"Expected tensors at {subpath}"
            if not torch.equal(v1, v2):
                diff = (v1 - v2).abs().max()
                pytest.fail(f"Tensor mismatch at {subpath}, max abs diff = {diff}")


def assert_state_different(s1: dict, s2: dict, path=""):
    """
    Recursively scan s1 and s2 (which must have identical keys) and
    succeed as soon as you find one tensor leaf that differs.
    If you reach the end with everything equal, fail the test.
    """
    assert set(s1.keys()) == set(s2.keys()), f"Key mismatch at {path}: {s1.keys()} vs {s2.keys()}"
    for k in s1:
        v1, v2 = s1[k], s2[k]
        subpath = f"{path}.{k}" if path else k
        if isinstance(v1, dict):
            # recurse; if any nested call returns (i.e. finds a diff), we propagate success
            try:
                assert_state_different(v1, v2, path=subpath)
                return
            except AssertionError:
                continue
        else:
            assert isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor), f"Expected tensors at {subpath}"
            if not torch.equal(v1, v2):
                return  # found a difference → success
    pytest.fail(f"No differing tensor found in nested state at {path}")
