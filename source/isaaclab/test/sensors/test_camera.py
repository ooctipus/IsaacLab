# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import copy
import random

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import omni.replicator.core as rep
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors.camera import Camera, CameraCfg

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.isaacsim_ci]

# sample camera poses
POSITION = (2.5, 2.5, 2.5)
# Quaternions in xyzw format
QUAT_ROS = (0.33985114, 0.82047325, -0.42470819, -0.17591989)
QUAT_OPENGL = (0.17591988, 0.42470818, 0.82047324, 0.33985113)
QUAT_WORLD = (-0.27984815, -0.1159169, 0.88047623, -0.3647052)


def _assert_quat_close(actual, expected, **kwargs):
    """Assert quaternions match while allowing the equivalent negated representation."""
    if hasattr(actual, "torch"):
        actual = actual.torch
    if hasattr(expected, "torch"):
        expected = expected.torch
    actual = torch.as_tensor(actual)
    expected = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
    expected = torch.where((actual * expected).sum(dim=-1, keepdim=True) < 0.0, -expected, expected)
    torch.testing.assert_close(actual, expected, **kwargs)


# NOTE: setup and teardown are own function to allow calling them in the tests

# resolutions
HEIGHT = 240
WIDTH = 320


def _cameras_in_plan(*cfgs: CameraCfg) -> tuple[Camera, ...]:
    """Build every camera in a scene inside its one clone plan.

    Every asset the scene draws is declared to a plan and copied by the cloning pass, cameras
    included: a camera reads where its copies land from the plan, so the plan has to exist before
    it is built. These cameras sit outside the per-environment namespace, so the plan gives each
    one row that every environment shares.
    """
    assets = _scene_cfgs()
    with cloner.ReplicateSession((*assets, *cfgs), num_clones=1, env_spacing=0.0):
        sim = sim_utils.SimulationContext.instance()
        plan = sim.get_clone_plan()
        for cfg in assets:
            source_path = next(path for path in cloner.query.cfg_source_paths(plan, cfg) if path is not None)
            cfg.spawn.func(
                source_path,
                cfg.spawn,
                translation=cfg.init_state.pos,
                orientation=cfg.init_state.rot,
            )
        return tuple(Camera(cfg) for cfg in cfgs)


def _camera_in_plan(cfg: CameraCfg) -> Camera:
    """Build one camera inside the scene's clone plan."""
    return _cameras_in_plan(cfg)[0]


def _scene_cfgs() -> tuple[AssetBaseCfg, ...]:
    """Return the complete plan-owned scene drawn by the camera tests."""
    assets = [
        AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()),
        AssetBaseCfg(
            prim_path="/World/Light/GreySphere",
            spawn=sim_utils.SphereLightCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(4.5, 3.5, 10.0)),
        ),
        AssetBaseCfg(
            prim_path="/World/Light/WhiteSphere",
            spawn=sim_utils.SphereLightCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-4.5, 3.5, 10.0)),
        ),
    ]
    random.seed(0)
    rng = np.random.default_rng(0)
    for index in range(10):
        prim_type = random.choice(("Cube", "Sphere", "Cylinder"))
        common = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            semantic_tags=[("class", prim_type)],
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(random.random(), random.random(), random.random())
            ),
        )
        if prim_type == "Cube":
            spawn = sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5), **common)
        elif prim_type == "Sphere":
            spawn = sim_utils.SphereCfg(radius=0.25, **common)
        else:
            spawn = sim_utils.CylinderCfg(radius=0.25, height=0.5, **common)
        position = (rng.random(3) - np.asarray([0.05, 0.05, -1.0])) * np.asarray([1.5, 1.5, 0.5])
        assets.append(
            AssetBaseCfg(
                prim_path=f"/World/Objects/Obj_{index:02d}",
                spawn=spawn,
                init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(position)),
            )
        )
    return tuple(assets)


def setup() -> tuple[sim_utils.SimulationContext, CameraCfg, float]:
    camera_cfg = CameraCfg(
        height=HEIGHT,
        width=WIDTH,
        prim_path="/World/Camera",
        update_period=0,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        renderer_cfg=IsaacRtxRendererCfg(),
    )
    # Create a new stage
    sim_utils.create_new_stage()
    # Simulation time-step
    dt = 0.01
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(physics=PhysxCfg(), dt=dt)
    sim = sim_utils.SimulationContext(sim_cfg)
    # load stage
    sim_utils.update_stage()
    return sim, camera_cfg, dt


def teardown(sim: sim_utils.SimulationContext):
    # Cleanup
    # close all the opened viewport from before.
    rep.vp_manager.destroy_hydra_textures("Replicator")
    # stop simulation
    sim.stop()
    # clear the stage
    sim.clear_instance()


@pytest.fixture
def setup_sim_camera():
    """Create a simulation context."""
    sim, camera_cfg, dt = setup()
    yield sim, camera_cfg, dt
    teardown(sim)


def test_camera_init(setup_sim_camera):
    """Test camera initialization."""
    # Create camera configuration
    sim, camera_cfg, _dt = setup_sim_camera
    # Create camera
    camera = _camera_in_plan(camera_cfg)
    assert camera._render_spec.camera_source_prim_paths == (camera_cfg.prim_path,)
    assert camera.prim_paths == (camera_cfg.prim_path,)
    assert tuple(str(camera.GetPath()) for camera in camera._prototype_sensor_prims) == (camera_cfg.prim_path,)
    # Check simulation parameter is set correctly
    assert sim.get_setting("/isaaclab/render/rtx_sensors")
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[0].GetPath().pathString == camera_cfg.prim_path
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)

    # Check buffers that exist and have correct shapes
    assert camera.data.pos_w.torch.shape == (1, 3)
    assert camera.data.quat_w_ros.torch.shape == (1, 4)
    assert camera.data.quat_w_world.torch.shape == (1, 4)
    assert camera.data.quat_w_opengl.torch.shape == (1, 4)
    assert camera.data.intrinsic_matrices.torch.shape == (1, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)
    assert camera.data.info == {camera_cfg.data_types[0]: None}

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(sim.cfg.dt)
        # check image data
        for im_data in camera.data.output.values():
            assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 1)


def test_camera_survives_a_stop_and_replay(setup_sim_camera):
    """Regression: a camera must still render after physics is stopped and played again.

    ``PhysicsEvent.STOP`` used to drop the camera's backend reference while leaving its render spec
        in place, and recovery was keyed on the spec -- so it never fired and the next
        ``PHYSICS_READY`` dereferenced ``None`` in ``create_render_data``. The backend is shared and
        outlives the stop, so the reference is simply never dropped now.
    """
    sim, camera_cfg, dt = setup_sim_camera
    camera = _camera_in_plan(camera_cfg)

    sim.reset()
    assert camera.is_initialized
    renderer_before = camera._renderer
    assert renderer_before is not None

    sim.stop()
    # STOP releases the per-play render data, which the camera owns. The backend is shared, owned by
    # SimulationContext, and outlives the cycle -- so the reference stays good and the description stands.
    assert camera._render_data is None
    assert camera._renderer is renderer_before
    assert camera._render_spec is not None
    # Reading poses between the stop and the replay must say so, not trip over a released view.
    with pytest.raises(RuntimeError, match="sim.play"):
        camera._update_poses()

    sim.reset()

    assert camera.is_initialized
    assert camera._renderer is renderer_before, "replay must reuse the backend, not lose or rebuild it"
    assert camera._render_data is not None
    # The camera still produces frames on the far side of the cycle.
    for _ in range(2):
        sim.step()
        camera.update(dt)
    assert camera.data.output["distance_to_image_plane"].torch.shape[0] == 1


def test_camera_init_offset(setup_sim_camera):
    """Test camera initialization with offset using different conventions."""
    sim, camera_cfg, dt = setup_sim_camera
    # define the same offset in all conventions
    # -- ROS convention
    cam_cfg_offset_ros = copy.deepcopy(camera_cfg)
    cam_cfg_offset_ros.update_latest_camera_pose = True
    cam_cfg_offset_ros.offset = CameraCfg.OffsetCfg(
        pos=POSITION,
        rot=QUAT_ROS,
        convention="ros",
    )
    cam_cfg_offset_ros.prim_path = "/World/CameraOffsetRos"
    # -- OpenGL convention
    cam_cfg_offset_opengl = copy.deepcopy(camera_cfg)
    cam_cfg_offset_opengl.update_latest_camera_pose = True
    cam_cfg_offset_opengl.offset = CameraCfg.OffsetCfg(
        pos=POSITION,
        rot=QUAT_OPENGL,
        convention="opengl",
    )
    cam_cfg_offset_opengl.prim_path = "/World/CameraOffsetOpengl"
    # -- World convention
    cam_cfg_offset_world = copy.deepcopy(camera_cfg)
    cam_cfg_offset_world.update_latest_camera_pose = True
    cam_cfg_offset_world.offset = CameraCfg.OffsetCfg(
        pos=POSITION,
        rot=QUAT_WORLD,
        convention="world",
    )
    cam_cfg_offset_world.prim_path = "/World/CameraOffsetWorld"
    camera_ros, camera_opengl, camera_world = _cameras_in_plan(
        cam_cfg_offset_ros, cam_cfg_offset_opengl, cam_cfg_offset_world
    )
    plan = sim.get_clone_plan()
    for cfg in (cam_cfg_offset_ros, cam_cfg_offset_opengl, cam_cfg_offset_world):
        np.testing.assert_allclose(plan.match_frames(cfg.prim_path)[0].pose[:3], POSITION)

    # play sim
    sim.reset()

    # Every convention describes the same camera pose; validate the SDP-backed camera state.
    for camera in (camera_ros, camera_opengl, camera_world):
        np.testing.assert_allclose(camera.data.pos_w.torch[0].cpu().numpy(), POSITION, rtol=1e-5)
        _assert_quat_close(camera.data.quat_w_ros.torch[0], QUAT_ROS, rtol=1e-5, atol=1e-5)
        _assert_quat_close(camera.data.quat_w_opengl[0], QUAT_OPENGL, rtol=1e-5, atol=1e-5)
        _assert_quat_close(camera.data.quat_w_world[0], QUAT_WORLD, rtol=1e-5, atol=1e-5)


def test_multi_camera_init(setup_sim_camera):
    """Test multi-camera initialization."""
    sim, camera_cfg, dt = setup_sim_camera
    # create two cameras with different prim paths
    # -- camera 1
    cam_cfg_1 = copy.deepcopy(camera_cfg)
    cam_cfg_1.prim_path = "/World/Camera_1"
    # -- camera 2
    cam_cfg_2 = copy.deepcopy(camera_cfg)
    cam_cfg_2.prim_path = "/World/Camera_2"
    cam_1, cam_2 = _cameras_in_plan(cam_cfg_1, cam_cfg_2)

    # play sim
    sim.reset()

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        cam_1.update(dt)
        cam_2.update(dt)
        # check image data
        for cam in [cam_1, cam_2]:
            for im_data in cam.data.output.values():
                assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 1)


def test_multi_camera_with_different_resolution(setup_sim_camera):
    """Test multi-camera initialization with cameras having different image resolutions."""
    sim, camera_cfg, dt = setup_sim_camera
    # create two cameras with different prim paths
    # -- camera 1
    cam_cfg_1 = copy.deepcopy(camera_cfg)
    cam_cfg_1.prim_path = "/World/Camera_1"
    # -- camera 2
    cam_cfg_2 = copy.deepcopy(camera_cfg)
    cam_cfg_2.prim_path = "/World/Camera_2"
    cam_cfg_2.height = 240
    cam_cfg_2.width = 320
    cam_1, cam_2 = _cameras_in_plan(cam_cfg_1, cam_cfg_2)

    # play sim
    sim.reset()

    # perform rendering
    sim.step()
    # update camera
    cam_1.update(dt)
    cam_2.update(dt)
    # check image sizes
    assert cam_1.data.output["distance_to_image_plane"].shape == (1, camera_cfg.height, camera_cfg.width, 1)
    assert cam_2.data.output["distance_to_image_plane"].shape == (1, cam_cfg_2.height, cam_cfg_2.width, 1)


def test_camera_init_intrinsic_matrix(setup_sim_camera):
    """Test camera initialization from intrinsic matrix."""
    sim, camera_cfg, dt = setup_sim_camera
    # get the first camera
    camera_1 = _camera_in_plan(camera_cfg)
    # get intrinsic matrix
    sim.reset()
    intrinsic_matrix = camera_1.data.intrinsic_matrices[0].cpu().flatten().tolist()
    teardown(sim)
    # reinit the first camera
    sim, camera_cfg, dt = setup()
    # initialize from intrinsic matrix
    intrinsic_camera_cfg = CameraCfg(
        height=HEIGHT,
        width=WIDTH,
        prim_path="/World/Camera_2",
        update_period=0,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic_matrix,
            width=WIDTH,
            height=HEIGHT,
            focal_length=24.0,
            focus_distance=400.0,
            clipping_range=(0.1, 1.0e5),
        ),
        renderer_cfg=IsaacRtxRendererCfg(),
    )
    camera_1, camera_2 = _cameras_in_plan(camera_cfg, intrinsic_camera_cfg)

    # play sim
    sim.reset()

    # update cameras
    camera_1.update(dt)
    camera_2.update(dt)

    # check image data
    torch.testing.assert_close(
        camera_1.data.output["distance_to_image_plane"],
        camera_2.data.output["distance_to_image_plane"],
        rtol=5e-3,
        atol=1e-4,
    )
    # check that both intrinsic matrices are the same
    torch.testing.assert_close(
        camera_1.data.intrinsic_matrices[0],
        camera_2.data.intrinsic_matrices[0],
        rtol=5e-3,
        atol=1e-4,
    )


@pytest.mark.parametrize("update_latest_camera_pose", [False, True])
def test_camera_set_world_poses(setup_sim_camera, update_latest_camera_pose):
    """Test that an explicitly set world pose is reflected in the data buffers."""
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.update_latest_camera_pose = update_latest_camera_pose
    # init camera
    camera = _camera_in_plan(camera_cfg)
    # play sim
    sim.reset()

    position = np.asarray([POSITION], dtype=np.float32)
    orientation = np.asarray([QUAT_WORLD], dtype=np.float32)
    # set new pose
    camera.set_world_poses(position, orientation, convention="world")

    # check if transform correctly set in output
    np.testing.assert_allclose(camera.data.pos_w.warp.numpy(), position)
    _assert_quat_close(camera.data.quat_w_world.warp.numpy(), orientation, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("update_latest_camera_pose", [False, True])
def test_camera_set_world_poses_from_view(setup_sim_camera, update_latest_camera_pose):
    """Test that a pose set from eye/target is reflected in the data buffers."""
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.update_latest_camera_pose = update_latest_camera_pose
    # init camera
    camera = _camera_in_plan(camera_cfg)
    # play sim
    sim.reset()

    eyes_np = np.asarray([POSITION], dtype=np.float32)
    targets_np = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    eyes = torch.tensor(eyes_np, dtype=torch.float32, device=camera.device)
    quat_ros_gt = torch.tensor([QUAT_ROS], dtype=torch.float32, device=camera.device)
    # set new pose
    camera.set_world_poses_from_view(eyes_np, targets_np)

    # check if transform correctly set in output
    torch.testing.assert_close(camera.data.pos_w.torch, eyes)
    _assert_quat_close(camera.data.quat_w_ros.torch, quat_ros_gt)


def test_intrinsic_matrix(setup_sim_camera):
    """Checks that the camera's set and retrieve methods work for intrinsic matrix."""
    sim, camera_cfg, dt = setup_sim_camera
    # enable update latest camera pose
    camera_cfg.update_latest_camera_pose = True
    # init camera
    camera = _camera_in_plan(camera_cfg)
    # play sim
    sim.reset()
    # Desired properties (obtained from realsense camera at 320x240 resolution)
    rs_intrinsic_matrix = [229.8, 0.0, 160.0, 0.0, 229.8, 120.0, 0.0, 0.0, 1.0]
    rs_intrinsic_matrix = np.asarray(rs_intrinsic_matrix, dtype=float).reshape(1, 3, 3)
    rs_intrinsic_matrix_tensor = torch.tensor(rs_intrinsic_matrix, dtype=torch.float32, device=camera.device)
    # Set matrix into simulator
    camera.set_intrinsic_matrices(rs_intrinsic_matrix_tensor)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # Check that matrix is correct
        assert np.isclose(rs_intrinsic_matrix[0, 0, 0], camera.data.intrinsic_matrices.torch[0, 0, 0].item())
        assert np.isclose(rs_intrinsic_matrix[0, 1, 1], camera.data.intrinsic_matrices.torch[0, 1, 1].item())
        assert np.isclose(rs_intrinsic_matrix[0, 0, 2], camera.data.intrinsic_matrices.torch[0, 0, 2].item())
        assert np.isclose(rs_intrinsic_matrix[0, 1, 2], camera.data.intrinsic_matrices.torch[0, 1, 2].item())


def test_depth_clipping(setup_sim_camera):
    """Test depth clipping.

    .. note::

        This test is the same for all camera models to enforce the same clipping behavior.
    """
    # get camera cfgs
    sim, _, dt = setup_sim_camera
    camera_cfg_zero = CameraCfg(
        prim_path="/World/CameraZero",
        offset=CameraCfg.OffsetCfg(pos=(2.5, 2.5, 6.0), rot=(0.362, 0.873, -0.302, -0.125), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg().from_intrinsic_matrix(
            focal_length=38.0,
            intrinsic_matrix=[380.08, 0.0, 467.79, 0.0, 380.08, 262.05, 0.0, 0.0, 1.0],
            height=540,
            width=960,
            clipping_range=(0.1, 10),
        ),
        height=540,
        width=960,
        data_types=["distance_to_image_plane", "distance_to_camera"],
        renderer_cfg=IsaacRtxRendererCfg(depth_clipping_behavior="zero"),
    )
    camera_cfg_none = copy.deepcopy(camera_cfg_zero)
    camera_cfg_none.prim_path = "/World/CameraNone"
    camera_cfg_none.renderer_cfg.depth_clipping_behavior = "none"
    camera_cfg_max = copy.deepcopy(camera_cfg_zero)
    camera_cfg_max.prim_path = "/World/CameraMax"
    camera_cfg_max.renderer_cfg.depth_clipping_behavior = "max"
    camera_zero, camera_none, camera_max = _cameras_in_plan(camera_cfg_zero, camera_cfg_none, camera_cfg_max)

    # Play sim
    sim.reset()

    camera_zero.update(dt)
    camera_none.update(dt)
    camera_max.update(dt)

    # none clipping should contain inf values
    assert torch.isinf(camera_none.data.output["distance_to_camera"]).any()
    assert torch.isinf(camera_none.data.output["distance_to_image_plane"]).any()
    assert (
        camera_none.data.output["distance_to_camera"][~torch.isinf(camera_none.data.output["distance_to_camera"])].min()
        >= camera_cfg_zero.spawn.clipping_range[0]
    )
    assert (
        camera_none.data.output["distance_to_camera"][~torch.isinf(camera_none.data.output["distance_to_camera"])].max()
        <= camera_cfg_zero.spawn.clipping_range[1]
    )
    assert (
        camera_none.data.output["distance_to_image_plane"][
            ~torch.isinf(camera_none.data.output["distance_to_image_plane"])
        ].min()
        >= camera_cfg_zero.spawn.clipping_range[0]
    )
    assert (
        camera_none.data.output["distance_to_image_plane"][
            ~torch.isinf(camera_none.data.output["distance_to_camera"])
        ].max()
        <= camera_cfg_zero.spawn.clipping_range[1]
    )

    # zero clipping should result in zero values
    assert torch.all(
        camera_zero.data.output["distance_to_camera"][torch.isinf(camera_none.data.output["distance_to_camera"])] == 0.0
    )
    assert torch.all(
        camera_zero.data.output["distance_to_image_plane"][
            torch.isinf(camera_none.data.output["distance_to_image_plane"])
        ]
        == 0.0
    )
    assert (
        camera_zero.data.output["distance_to_camera"][camera_zero.data.output["distance_to_camera"] != 0.0].min()
        >= camera_cfg_zero.spawn.clipping_range[0]
    )
    assert camera_zero.data.output["distance_to_camera"].max() <= camera_cfg_zero.spawn.clipping_range[1]
    assert (
        camera_zero.data.output["distance_to_image_plane"][
            camera_zero.data.output["distance_to_image_plane"] != 0.0
        ].min()
        >= camera_cfg_zero.spawn.clipping_range[0]
    )
    assert camera_zero.data.output["distance_to_image_plane"].max() <= camera_cfg_zero.spawn.clipping_range[1]

    # max clipping should result in max values
    assert torch.all(
        camera_max.data.output["distance_to_camera"][torch.isinf(camera_none.data.output["distance_to_camera"])]
        == camera_cfg_zero.spawn.clipping_range[1]
    )
    assert torch.all(
        camera_max.data.output["distance_to_image_plane"][
            torch.isinf(camera_none.data.output["distance_to_image_plane"])
        ]
        == camera_cfg_zero.spawn.clipping_range[1]
    )
    assert camera_max.data.output["distance_to_camera"].min() >= camera_cfg_zero.spawn.clipping_range[0]
    assert camera_max.data.output["distance_to_camera"].max() <= camera_cfg_zero.spawn.clipping_range[1]
    assert camera_max.data.output["distance_to_image_plane"].min() >= camera_cfg_zero.spawn.clipping_range[0]
    assert camera_max.data.output["distance_to_image_plane"].max() <= camera_cfg_zero.spawn.clipping_range[1]


def test_camera_resolution_all_colorize(setup_sim_camera):
    """Test camera resolution is correctly set for all types with colorization enabled."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = [
        "rgb",
        "rgba",
        "albedo",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation",
        "instance_id_segmentation_fast",
    ]
    camera_cfg.renderer_cfg.colorize_instance_id_segmentation = True
    camera_cfg.renderer_cfg.colorize_instance_segmentation = True
    camera_cfg.renderer_cfg.colorize_semantic_segmentation = True
    # Create camera
    camera = _camera_in_plan(camera_cfg)

    # Play sim
    sim.reset()

    camera.update(dt)

    # expected sizes
    hw_1c_shape = (1, camera_cfg.height, camera_cfg.width, 1)
    hw_2c_shape = (1, camera_cfg.height, camera_cfg.width, 2)
    hw_3c_shape = (1, camera_cfg.height, camera_cfg.width, 3)
    hw_4c_shape = (1, camera_cfg.height, camera_cfg.width, 4)
    # access image data and compare shapes
    output = camera.data.output
    assert output["rgb"].shape == hw_3c_shape
    assert output["rgba"].shape == hw_4c_shape
    assert output["albedo"].shape == hw_4c_shape
    assert output["depth"].shape == hw_1c_shape
    assert output["distance_to_camera"].shape == hw_1c_shape
    assert output["distance_to_image_plane"].shape == hw_1c_shape
    assert output["normals"].shape == hw_3c_shape
    assert output["motion_vectors"].shape == hw_2c_shape
    assert output["semantic_segmentation"].shape == hw_4c_shape
    assert output["instance_segmentation"].shape == hw_4c_shape
    assert output["instance_id_segmentation_fast"].shape == hw_4c_shape

    # access image data and compare dtype
    output = camera.data.output
    assert output["rgb"].dtype == wp.uint8
    assert output["rgba"].dtype == wp.uint8
    assert output["albedo"].dtype == wp.uint8
    assert output["depth"].dtype == wp.float32
    assert output["distance_to_camera"].dtype == wp.float32
    assert output["distance_to_image_plane"].dtype == wp.float32
    assert output["normals"].dtype == wp.float32
    assert output["motion_vectors"].dtype == wp.float32
    assert output["semantic_segmentation"].dtype == wp.uint8
    assert output["instance_segmentation"].dtype == wp.uint8
    assert output["instance_id_segmentation_fast"].dtype == wp.uint8


def test_camera_resolution_no_colorize(setup_sim_camera):
    """Test camera resolution is correctly set for all types with no colorization enabled."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = [
        "rgb",
        "rgba",
        "albedo",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation",
        "instance_id_segmentation_fast",
    ]
    camera_cfg.renderer_cfg.colorize_instance_id_segmentation = False
    camera_cfg.renderer_cfg.colorize_instance_segmentation = False
    camera_cfg.renderer_cfg.colorize_semantic_segmentation = False
    # Create camera
    camera = _camera_in_plan(camera_cfg)

    # Play sim
    sim.reset()
    camera.update(dt)

    # expected sizes
    hw_1c_shape = (1, camera_cfg.height, camera_cfg.width, 1)
    hw_2c_shape = (1, camera_cfg.height, camera_cfg.width, 2)
    hw_3c_shape = (1, camera_cfg.height, camera_cfg.width, 3)
    hw_4c_shape = (1, camera_cfg.height, camera_cfg.width, 4)
    # access image data and compare shapes
    output = camera.data.output
    assert output["rgb"].shape == hw_3c_shape
    assert output["rgba"].shape == hw_4c_shape
    assert output["albedo"].shape == hw_4c_shape
    assert output["depth"].shape == hw_1c_shape
    assert output["distance_to_camera"].shape == hw_1c_shape
    assert output["distance_to_image_plane"].shape == hw_1c_shape
    assert output["normals"].shape == hw_3c_shape
    assert output["motion_vectors"].shape == hw_2c_shape
    assert output["semantic_segmentation"].shape == hw_1c_shape
    assert output["instance_segmentation"].shape == hw_1c_shape
    assert output["instance_id_segmentation_fast"].shape == hw_1c_shape

    # access image data and compare dtype
    output = camera.data.output
    assert output["rgb"].dtype == wp.uint8
    assert output["rgba"].dtype == wp.uint8
    assert output["albedo"].dtype == wp.uint8
    assert output["depth"].dtype == wp.float32
    assert output["distance_to_camera"].dtype == wp.float32
    assert output["distance_to_image_plane"].dtype == wp.float32
    assert output["normals"].dtype == wp.float32
    assert output["motion_vectors"].dtype == wp.float32
    assert output["semantic_segmentation"].dtype == wp.int32
    assert output["instance_segmentation"].dtype == wp.int32
    assert output["instance_id_segmentation_fast"].dtype == wp.int32


def test_camera_large_resolution_all_colorize(setup_sim_camera):
    """Test camera resolution is correctly set for all types with colorization enabled."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = [
        "rgb",
        "rgba",
        "albedo",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation",
        "instance_id_segmentation_fast",
    ]
    camera_cfg.renderer_cfg.colorize_instance_id_segmentation = True
    camera_cfg.renderer_cfg.colorize_instance_segmentation = True
    camera_cfg.renderer_cfg.colorize_semantic_segmentation = True
    camera_cfg.width = 512
    camera_cfg.height = 512
    # Create camera
    camera = _camera_in_plan(camera_cfg)

    # Play sim
    sim.reset()

    camera.update(dt)

    # expected sizes
    hw_1c_shape = (1, camera_cfg.height, camera_cfg.width, 1)
    hw_2c_shape = (1, camera_cfg.height, camera_cfg.width, 2)
    hw_3c_shape = (1, camera_cfg.height, camera_cfg.width, 3)
    hw_4c_shape = (1, camera_cfg.height, camera_cfg.width, 4)
    # access image data and compare shapes
    output = camera.data.output
    assert output["rgb"].shape == hw_3c_shape
    assert output["rgba"].shape == hw_4c_shape
    assert output["albedo"].shape == hw_4c_shape
    assert output["depth"].shape == hw_1c_shape
    assert output["distance_to_camera"].shape == hw_1c_shape
    assert output["distance_to_image_plane"].shape == hw_1c_shape
    assert output["normals"].shape == hw_3c_shape
    assert output["motion_vectors"].shape == hw_2c_shape
    assert output["semantic_segmentation"].shape == hw_4c_shape
    assert output["instance_segmentation"].shape == hw_4c_shape
    assert output["instance_id_segmentation_fast"].shape == hw_4c_shape

    # access image data and compare dtype
    output = camera.data.output
    assert output["rgb"].dtype == wp.uint8
    assert output["rgba"].dtype == wp.uint8
    assert output["albedo"].dtype == wp.uint8
    assert output["depth"].dtype == wp.float32
    assert output["distance_to_camera"].dtype == wp.float32
    assert output["distance_to_image_plane"].dtype == wp.float32
    assert output["normals"].dtype == wp.float32
    assert output["motion_vectors"].dtype == wp.float32
    assert output["semantic_segmentation"].dtype == wp.uint8
    assert output["instance_segmentation"].dtype == wp.uint8
    assert output["instance_id_segmentation_fast"].dtype == wp.uint8


def test_camera_resolution_rgb_only(setup_sim_camera):
    """Test camera resolution is correctly set for RGB only."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = ["rgb"]
    # Create camera
    camera = _camera_in_plan(camera_cfg)

    # Play sim
    sim.reset()

    camera.update(dt)

    # expected sizes
    hw_3c_shape = (1, camera_cfg.height, camera_cfg.width, 3)
    # access image data and compare shapes
    output = camera.data.output
    assert output["rgb"].shape == hw_3c_shape
    # access image data and compare dtype
    assert output["rgb"].dtype == wp.uint8


def test_camera_resolution_rgba_only(setup_sim_camera):
    """Test camera resolution is correctly set for RGBA only."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = ["rgba"]
    # Create camera
    camera = _camera_in_plan(camera_cfg)

    # Play sim
    sim.reset()

    camera.update(dt)

    # expected sizes
    hw_4c_shape = (1, camera_cfg.height, camera_cfg.width, 4)
    # access image data and compare shapes
    output = camera.data.output
    assert output["rgba"].shape == hw_4c_shape
    # access image data and compare dtype
    assert output["rgba"].dtype == wp.uint8


def test_camera_resolution_albedo_only(setup_sim_camera):
    """Test camera resolution is correctly set for albedo only."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = ["albedo"]
    # Create camera
    camera = _camera_in_plan(camera_cfg)

    # Play sim
    sim.reset()

    camera.update(dt)

    # expected sizes
    hw_4c_shape = (1, camera_cfg.height, camera_cfg.width, 4)
    # access image data and compare shapes
    output = camera.data.output
    assert output["albedo"].shape == hw_4c_shape
    # access image data and compare dtype
    assert output["albedo"].dtype == wp.uint8


@pytest.mark.parametrize(
    "data_type",
    ["simple_shading_constant_diffuse", "simple_shading_diffuse_mdl", "simple_shading_full_mdl"],
)
def test_camera_resolution_simple_shading_only(setup_sim_camera, data_type):
    """Test camera resolution is correctly set for simple shading only."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = [data_type]
    # Create camera
    camera = _camera_in_plan(camera_cfg)

    # Play sim
    sim.reset()

    camera.update(dt)

    # expected sizes
    hw_3c_shape = (1, camera_cfg.height, camera_cfg.width, 3)
    # access image data and compare shapes
    output = camera.data.output
    assert output[data_type].shape == hw_3c_shape
    # access image data and compare dtype
    assert output[data_type].dtype == wp.uint8


def test_camera_resolution_depth_only(setup_sim_camera):
    """Test camera resolution is correctly set for depth only."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = ["depth"]
    # Create camera
    camera = _camera_in_plan(camera_cfg)

    # Play sim
    sim.reset()

    camera.update(dt)

    # expected sizes
    hw_1c_shape = (1, camera_cfg.height, camera_cfg.width, 1)
    # access image data and compare shapes
    output = camera.data.output
    assert output["depth"].shape == hw_1c_shape
    # access image data and compare dtype
    assert output["depth"].dtype == wp.float32


def test_sensor_print(setup_sim_camera):
    """Test sensor print is working correctly."""
    # Create sensor
    sim, camera_cfg, dt = setup_sim_camera
    sensor = _camera_in_plan(camera_cfg)
    # Play sim
    sim.reset()
    # print info
    print(sensor)


def setup_with_device(device) -> tuple[sim_utils.SimulationContext, CameraCfg, float]:
    camera_cfg = CameraCfg(
        height=128,
        width=256,
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 4.0), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
        prim_path="/World/Camera",
        update_period=0,
        data_types=["rgb", "distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        renderer_cfg=IsaacRtxRendererCfg(),
    )
    sim_utils.create_new_stage()
    dt = 0.01
    sim_cfg = sim_utils.SimulationCfg(physics=PhysxCfg(), dt=dt, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_utils.update_stage()
    return sim, camera_cfg, dt


@pytest.fixture(scope="function")
def setup_camera_device(device):
    """Fixture with explicit device parametrization for GPU/CPU testing."""
    sim, camera_cfg, dt = setup_with_device(device)
    yield sim, camera_cfg, dt
    teardown(sim)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_camera_multi_regex_init(setup_camera_device, device):
    """Test multi-camera initialization with regex prim paths and content validation."""
    sim, camera_cfg, dt = setup_camera_device

    num_cameras = 9
    for i in range(num_cameras):
        sim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.prim_path = "/World/Origin_[^/]*/CameraSensor"
    camera = _camera_in_plan(camera_cfg)

    sim.reset()

    assert camera.is_initialized
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)

    assert camera.data.pos_w.torch.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.torch.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.torch.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.torch.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.torch.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    for _ in range(10):
        sim.step()
        camera.update(dt)
        for im_type, im_data in camera.data.output.items():
            if im_type == "rgb":
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
                for i in range(4):
                    assert (im_data[i] / 255.0).mean() > 0.0
            elif im_type == "distance_to_camera":
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
                for i in range(4):
                    assert im_data[i].mean() > 0.0
    del camera


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_camera_all_annotators(setup_camera_device, device):
    """Test all supported annotators produce correct shapes, dtypes, content, and info."""
    sim, camera_cfg, dt = setup_camera_device
    all_annotator_types = [
        "rgb",
        "rgba",
        "albedo",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation",
        "instance_id_segmentation_fast",
    ]

    num_cameras = 9
    for i in range(num_cameras):
        sim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = all_annotator_types
    camera_cfg.prim_path = "/World/Origin_[^/]*/CameraSensor"
    camera = _camera_in_plan(camera_cfg)

    sim.reset()

    assert camera.is_initialized
    assert sorted(camera.data.output.keys()) == sorted(all_annotator_types)

    for _ in range(10):
        sim.step()
        camera.update(dt)
        for data_type, im_data in camera.data.output.items():
            if data_type in ["rgb", "normals"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
            elif data_type in [
                "rgba",
                "albedo",
                "semantic_segmentation",
                "instance_segmentation",
                "instance_id_segmentation_fast",
            ]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
                for i in range(num_cameras):
                    assert (im_data[i] / 255.0).mean() > 0.0
            elif data_type in ["motion_vectors"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 2)
                for i in range(num_cameras):
                    assert im_data[i].mean() != 0.0
            elif data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
                for i in range(num_cameras):
                    assert im_data[i].mean() > 0.0

    output = camera.data.output
    info = camera.data.info
    assert output["rgb"].dtype == wp.uint8
    assert output["rgba"].dtype == wp.uint8
    assert output["albedo"].dtype == wp.uint8
    assert output["depth"].dtype == wp.float32
    assert output["distance_to_camera"].dtype == wp.float32
    assert output["distance_to_image_plane"].dtype == wp.float32
    assert output["normals"].dtype == wp.float32
    assert output["motion_vectors"].dtype == wp.float32
    assert output["semantic_segmentation"].dtype == wp.uint8
    assert output["instance_segmentation"].dtype == wp.uint8
    assert output["instance_id_segmentation_fast"].dtype == wp.uint8
    assert isinstance(info["semantic_segmentation"], dict)
    assert isinstance(info["instance_segmentation"], dict)
    assert isinstance(info["instance_id_segmentation_fast"], dict)

    del camera


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_camera_segmentation_non_colorize(setup_camera_device, device):
    """Test segmentation outputs with colorization disabled produce correct dtypes and info."""
    sim, camera_cfg, dt = setup_camera_device
    num_cameras = 9
    for i in range(num_cameras):
        sim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["semantic_segmentation", "instance_segmentation", "instance_id_segmentation_fast"]
    camera_cfg.prim_path = "/World/Origin_[^/]*/CameraSensor"
    camera_cfg.renderer_cfg.colorize_semantic_segmentation = False
    camera_cfg.renderer_cfg.colorize_instance_segmentation = False
    camera_cfg.renderer_cfg.colorize_instance_id_segmentation = False
    camera = _camera_in_plan(camera_cfg)

    sim.reset()

    for _ in range(5):
        sim.step()
        camera.update(dt)

    for seg_type in camera_cfg.data_types:
        assert camera.data.output[seg_type].shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
        assert camera.data.output[seg_type].dtype == wp.int32
        assert isinstance(camera.data.info[seg_type], dict)

    del camera


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_camera_normals_unit_length(setup_camera_device, device):
    """Test that normals output vectors have approximately unit length."""
    sim, camera_cfg, dt = setup_camera_device
    num_cameras = 9
    for i in range(num_cameras):
        sim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["normals"]
    camera_cfg.prim_path = "/World/Origin_[^/]*/CameraSensor"
    camera = _camera_in_plan(camera_cfg)

    sim.reset()

    for _ in range(10):
        sim.step()
        camera.update(dt)
        im_data = camera.data.output["normals"]
        assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
        for i in range(4):
            assert im_data[i].mean() > 0.0
        norms = torch.linalg.norm(im_data, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-9)

    assert camera.data.output["normals"].dtype == wp.float32
    del camera


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_camera_data_types_ordering(setup_camera_device, device):
    """Test that requesting specific data types produces the expected output keys."""
    sim, camera_cfg, dt = setup_camera_device
    camera_cfg_distance = copy.deepcopy(camera_cfg)
    camera_cfg_distance.data_types = ["distance_to_camera"]
    camera_cfg_distance.prim_path = "/World/CameraDistance"

    camera_cfg_depth = copy.deepcopy(camera_cfg)
    camera_cfg_depth.data_types = ["depth"]
    camera_cfg_depth.prim_path = "/World/CameraDepth"

    camera_cfg_both = copy.deepcopy(camera_cfg)
    camera_cfg_both.data_types = ["distance_to_camera", "depth"]
    camera_cfg_both.prim_path = "/World/CameraBoth"
    camera_distance, camera_depth, camera_both = _cameras_in_plan(
        camera_cfg_distance, camera_cfg_depth, camera_cfg_both
    )

    sim.reset()

    assert camera_distance.is_initialized
    assert camera_depth.is_initialized
    assert camera_both.is_initialized
    assert list(camera_distance.data.output.keys()) == ["distance_to_camera"]
    assert list(camera_depth.data.output.keys()) == ["depth"]
    assert list(camera_both.data.output.keys()) == ["depth", "distance_to_camera"]

    del camera_distance
    del camera_depth
    del camera_both


@pytest.mark.parametrize("device", ["cuda:0"])
def test_camera_frame_offset(setup_camera_device, device):
    """Test that camera reflects scene color changes without frame-offset lag."""
    sim, camera_cfg, dt = setup_camera_device
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.height = 480
    camera_cfg.width = 480
    camera = _camera_in_plan(camera_cfg)

    stage = sim_utils.get_current_stage()
    for i in range(10):
        prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
        color = Gf.Vec3f(1, 1, 1)
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])

    sim.reset()

    for _ in range(100):
        sim.step()
        camera.update(dt)

    image_before = camera.data.output["rgb"].clone() / 255.0

    for i in range(10):
        prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
        color = Gf.Vec3f(0, 0, 0)
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])

    sim.step()
    camera.update(dt)

    image_after = camera.data.output["rgb"].clone() / 255.0

    assert torch.abs(image_after - image_before).mean() > 0.01

    del camera


@pytest.mark.parametrize(
    ("data_types", "expected_names", "expected_message"),
    [
        (["rgba", "depth", "normals"], ["depth", "normals"], "does not support requested data types"),
        (["rgba", "not_a_render_buffer_kind"], ["not_a_render_buffer_kind"], "Unknown camera data types"),
    ],
)
def test_camera_rejects_unsupported_data_types(setup_sim_camera, data_types, expected_names, expected_message):
    """A renderer must implement every output the camera requests."""
    from isaaclab.renderers.base_renderer import BaseRenderer

    sim, camera_cfg, _dt = setup_sim_camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = data_types

    from isaaclab.sensors.camera.camera_data import RenderBufferKind, RenderBufferSpec

    class _PartialRenderer(BaseRenderer):
        """Publishes only ``rgba`` in its supported-output contract."""

        def __init__(self, cfg=None):
            self.cfg = cfg

        def supported_output_types(self):
            return {RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8)}

        def prepare_stage(self, stage, plan):
            pass

        def create_render_data(self, sensor):
            return object()

        def set_outputs(self, render_data, output_data):
            pass

        def update(self, render_data, intrinsics):
            pass

        def render(self, render_data):
            pass

        def read_output(self, render_data, camera_data):
            pass

        def cleanup(self, render_data):
            pass

    camera_cfg.renderer_cfg.class_type = _PartialRenderer
    camera = _camera_in_plan(camera_cfg)
    with pytest.raises(ValueError) as exc_info:
        sim.reset()
    assert expected_message in str(exc_info.value)
    assert all(name in str(exc_info.value) for name in expected_names)

    del camera


def test_camera_raises_on_instance_segmentation_fast(setup_sim_camera):
    """Camera raises ValueError when the renamed data type 'instance_segmentation_fast' is used."""
    _sim, camera_cfg, _dt = setup_sim_camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["instance_segmentation_fast"]
    with pytest.raises(ValueError, match="instance_segmentation"):
        _camera_in_plan(camera_cfg)


def test_camera_rejects_multiple_simple_shading_modes(setup_sim_camera):
    """A camera cannot silently select one of two explicitly requested shading modes."""
    _sim, camera_cfg, _dt = setup_sim_camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["simple_shading_constant_diffuse", "simple_shading_full_mdl"]

    with pytest.raises(ValueError, match="multiple simple shading modes"):
        _camera_in_plan(camera_cfg)


@pytest.mark.parametrize("data_types", [[], ["rgb", "rgb"]])
def test_camera_rejects_empty_or_duplicate_outputs(setup_sim_camera, data_types):
    """Each camera declares a non-empty set of outputs exactly once."""
    _sim, camera_cfg, _dt = setup_sim_camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = data_types

    with pytest.raises(ValueError, match="at least one|duplicate"):
        _camera_in_plan(camera_cfg)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_camera_pose_update_reflected_in_render(setup_camera_device, device):
    """Camera pose changes via FrameView should be visible in rendered depth.

    Moves the camera close then far, renders depth, and verifies that the mean
    valid depth from the far position is significantly larger (>1.5×) than the
    close position. This validates that the named SDP camera publication reaches
    the exact plan-owned RTX camera sink.
    """
    sim, _unused_cam_cfg, dt = setup_camera_device

    cam_cfg = CameraCfg(
        prim_path="/World/PoseTestCam",
        height=128,
        width=256,
        update_period=0,
        update_latest_camera_pose=True,
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        renderer_cfg=IsaacRtxRendererCfg(),
    )
    camera = _camera_in_plan(cam_cfg)
    try:
        sim.reset()

        target = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        max_range = cam_cfg.spawn.clipping_range[1]

        # -- close position --
        eyes_close = np.asarray([[2.0, 2.0, 2.0]], dtype=np.float32)
        camera.set_world_poses_from_view(eyes_close, target)
        sim.step()
        camera.update(dt)
        depth_close = camera.data.output["distance_to_camera"].clone()

        # -- far position --
        eyes_far = np.asarray([[8.0, 8.0, 8.0]], dtype=np.float32)
        camera.set_world_poses_from_view(eyes_far, target)
        sim.step()
        camera.update(dt)
        depth_far = camera.data.output["distance_to_camera"].clone()

        # -- validate --
        valid_close = depth_close[depth_close < max_range]
        valid_far = depth_far[depth_far < max_range]

        assert valid_close.numel() > 0, "No valid close-range depth pixels"
        assert valid_far.numel() > 0, "No valid far-range depth pixels"

        mean_close = valid_close.mean().item()
        mean_far = valid_far.mean().item()

        assert mean_far > mean_close * 1.5, (
            f"Far depth ({mean_far:.2f}) should be > 1.5× close depth ({mean_close:.2f}). "
            "Camera pose change may not be reaching the renderer."
        )
    finally:
        del camera


def test_camera_invalidate_before_initialize(setup_sim_camera):
    """Invalidation on a camera that never initialized does not raise."""
    _, camera_cfg, _ = setup_sim_camera
    camera = _camera_in_plan(camera_cfg)
    try:
        view = camera._view
        assert not camera.is_initialized
        camera._invalidate_initialize_callback(None)
        assert camera._view is view
    finally:
        del camera
