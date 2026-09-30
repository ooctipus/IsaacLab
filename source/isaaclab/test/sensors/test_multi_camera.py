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
from flaky import flaky
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import omni.replicator.core as rep
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.sensors.camera import Camera, CameraCfg

pytestmark = [pytest.mark.integration, pytest.mark.rendering]


def _cameras_in_plan(cfgs: list[CameraCfg], num_clones: int) -> list[Camera]:
    """Build cameras inside one clone plan, the way a scene does.

    Every asset the scene draws is declared to a plan and copied by the cloning pass, cameras
    included: a camera reads where its copies land from the plan, so the plan has to exist before
    it is built.
    """
    with cloner.ReplicateSession(cfgs, num_clones, env_spacing=0.0):
        return [cfg.class_type(cfg) for cfg in cfgs]


@pytest.fixture()
def setup_camera():
    """Create a blank new stage for each test."""
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
    # Create a new stage
    sim_utils.create_new_stage()
    # Simulation time-step
    dt = 0.01
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(physics=PhysxCfg(), dt=dt)
    sim = sim_utils.SimulationContext(sim_cfg)
    # populate scene
    _populate_scene()
    # load stage
    sim_utils.update_stage()
    yield camera_cfg, sim, dt
    # Teardown
    rep.vp_manager.destroy_hydra_textures("Replicator")
    # stop simulation
    sim.stop()
    # clear the stage
    sim.clear_instance()


@pytest.mark.isaacsim_ci
def test_multi_camera_init(setup_camera):
    """Test initialization of multiple batched cameras."""
    camera_cfg, sim, dt = setup_camera
    num_camera_groups = 3
    num_clones = 7

    camera_cfgs = []
    for i in range(num_camera_groups):
        cfg = copy.deepcopy(camera_cfg)
        cfg.prim_path = f"{{ENV_REGEX_NS}}/CameraSensor_{i}"
        camera_cfgs.append(cfg)
    cameras = _cameras_in_plan(camera_cfgs, num_clones)

    # Check simulation parameter is set correctly
    assert sim.get_setting("/isaaclab/render/rtx_sensors")

    # Play sim
    sim.reset()

    for i, camera in enumerate(cameras):
        # Check if camera is initialized
        assert camera.is_initialized
        # Check the plan assigned the expected camera path and spawned a camera prototype.
        assert camera._render_spec.camera_prim_paths[1] == f"/World/envs/env_1/CameraSensor_{i}"
        assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)

    for camera in cameras:
        # Check buffers that exists and have correct shapes
        assert camera.data.pos_w.torch.shape == (num_clones, 3)
        assert camera.data.quat_w_ros.torch.shape == (num_clones, 4)
        assert camera.data.quat_w_world.torch.shape == (num_clones, 4)
        assert camera.data.quat_w_opengl.torch.shape == (num_clones, 4)
        assert camera.data.intrinsic_matrices.torch.shape == (num_clones, 3, 3)
        assert camera.data.image_shape == (camera.cfg.height, camera.cfg.width)

    # Simulate physics
    for _ in range(10):
        # Initialize data arrays
        rgbs = []
        distances = []

        # perform rendering
        sim.step()
        for camera in cameras:
            # update camera
            camera.update(dt)
            # check image data
            for data_type, im_data in camera.data.output.items():
                if data_type == "rgb":
                    im_data = im_data.clone() / 255.0
                    assert im_data.shape == (num_clones, camera.cfg.height, camera.cfg.width, 3)
                    for j in range(num_clones):
                        assert (im_data[j]).mean().item() > 0.0
                    rgbs.append(im_data)
                elif data_type == "distance_to_camera":
                    im_data = im_data.clone()
                    im_data[torch.isinf(im_data)] = 0
                    assert im_data.shape == (num_clones, camera.cfg.height, camera.cfg.width, 1)
                    for j in range(num_clones):
                        assert im_data[j].mean().item() > 0.0
                    distances.append(im_data)

        # Check data from camera groups are consistent, assumes >1 group
        for i in range(1, num_camera_groups):
            assert torch.abs(rgbs[0] - rgbs[i]).mean() < 0.05  # images of same color should be below 0.001
            assert torch.abs(distances[0] - distances[i]).mean() < 0.01  # distances of same scene should be 0

    for camera in cameras:
        del camera


@pytest.mark.isaacsim_ci
def test_all_annotators_multi_camera(setup_camera):
    """Test multiple batched cameras with all supported annotators."""
    camera_cfg, sim, dt = setup_camera
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

    num_camera_groups = 2
    num_clones = 9

    camera_cfgs = []
    for i in range(num_camera_groups):
        cfg = copy.deepcopy(camera_cfg)
        cfg.data_types = all_annotator_types
        cfg.prim_path = f"{{ENV_REGEX_NS}}/CameraSensor_{i}"
        camera_cfgs.append(cfg)
    cameras = _cameras_in_plan(camera_cfgs, num_clones)

    # Check simulation parameter is set correctly
    assert sim.get_setting("/isaaclab/render/rtx_sensors")

    # Play sim
    sim.reset()

    for i, camera in enumerate(cameras):
        # Check if camera is initialized
        assert camera.is_initialized
        # Check the plan assigned the expected camera path and spawned a camera prototype.
        assert camera._render_spec.camera_prim_paths[1] == f"/World/envs/env_1/CameraSensor_{i}"
        assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
        assert sorted(camera.data.output.keys()) == sorted(all_annotator_types)

    for camera in cameras:
        # Check buffers that exists and have correct shapes
        assert camera.data.pos_w.torch.shape == (num_clones, 3)
        assert camera.data.quat_w_ros.torch.shape == (num_clones, 4)
        assert camera.data.quat_w_world.torch.shape == (num_clones, 4)
        assert camera.data.quat_w_opengl.torch.shape == (num_clones, 4)
        assert camera.data.intrinsic_matrices.torch.shape == (num_clones, 3, 3)
        assert camera.data.image_shape == (camera.cfg.height, camera.cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        for camera in cameras:
            # update camera
            camera.update(dt)
            # check image data
            for data_type, im_data in camera.data.output.items():
                if data_type in ["rgb", "normals"]:
                    assert im_data.shape == (num_clones, camera.cfg.height, camera.cfg.width, 3)
                elif data_type in [
                    "rgba",
                    "albedo",
                    "semantic_segmentation",
                    "instance_segmentation",
                    "instance_id_segmentation_fast",
                ]:
                    assert im_data.shape == (num_clones, camera.cfg.height, camera.cfg.width, 4)
                    for i in range(num_clones):
                        assert (im_data[i] / 255.0).mean().item() > 0.0
                elif data_type in ["motion_vectors"]:
                    assert im_data.shape == (num_clones, camera.cfg.height, camera.cfg.width, 2)
                    for i in range(num_clones):
                        assert im_data[i].mean().item() != 0.0
                elif data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
                    assert im_data.shape == (num_clones, camera.cfg.height, camera.cfg.width, 1)
                    for i in range(num_clones):
                        assert im_data[i].mean().item() > 0.0

    for camera in cameras:
        # access image data and compare dtype
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

    for camera in cameras:
        del camera


@flaky(max_runs=3, min_passes=1)
@pytest.mark.isaacsim_ci
def test_different_resolution_multi_camera(setup_camera):
    """Test multiple batched cameras with different resolutions."""
    camera_cfg, sim, dt = setup_camera
    num_camera_groups = 2
    num_clones = 6

    camera_cfgs = []
    resolutions = [(16, 16), (23, 765)]
    for i in range(num_camera_groups):
        cfg = copy.deepcopy(camera_cfg)
        cfg.prim_path = f"{{ENV_REGEX_NS}}/CameraSensor_{i}"
        cfg.height, cfg.width = resolutions[i]
        camera_cfgs.append(cfg)
    cameras = _cameras_in_plan(camera_cfgs, num_clones)

    # Check simulation parameter is set correctly
    assert sim.get_setting("/isaaclab/render/rtx_sensors")

    # Play sim
    sim.reset()

    for i, camera in enumerate(cameras):
        # Check if camera is initialized
        assert camera.is_initialized
        # Check the plan assigned the expected camera path and spawned a camera prototype.
        assert camera._render_spec.camera_prim_paths[1] == f"/World/envs/env_1/CameraSensor_{i}"
        assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)

    for camera in cameras:
        # Check buffers that exists and have correct shapes
        assert camera.data.pos_w.torch.shape == (num_clones, 3)
        assert camera.data.quat_w_ros.torch.shape == (num_clones, 4)
        assert camera.data.quat_w_world.torch.shape == (num_clones, 4)
        assert camera.data.quat_w_opengl.torch.shape == (num_clones, 4)
        assert camera.data.intrinsic_matrices.torch.shape == (num_clones, 3, 3)
        assert camera.data.image_shape == (camera.cfg.height, camera.cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        for camera in cameras:
            # update camera
            camera.update(dt)
            # check image data
            for data_type, im_data in camera.data.output.items():
                if data_type == "rgb":
                    im_data = im_data.clone() / 255.0
                    assert im_data.shape == (num_clones, camera.cfg.height, camera.cfg.width, 3)
                    for j in range(num_clones):
                        assert (im_data[j]).mean().item() > 0.0
                elif data_type == "distance_to_camera":
                    im_data = im_data.clone()
                    assert im_data.shape == (num_clones, camera.cfg.height, camera.cfg.width, 1)
                    for j in range(num_clones):
                        assert im_data[j].mean().item() > 0.0

    for camera in cameras:
        del camera


@pytest.mark.isaacsim_ci
@flaky(max_runs=3, min_passes=1)
def test_frame_offset_multi_camera(setup_camera):
    """Test frame offset updates with multiple batched cameras."""
    camera_cfg, sim, dt = setup_camera
    num_camera_groups = 4
    num_clones = 4

    camera_cfgs = []
    for i in range(num_camera_groups):
        cfg = copy.deepcopy(camera_cfg)
        cfg.prim_path = f"{{ENV_REGEX_NS}}/CameraSensor_{i}"
        camera_cfgs.append(cfg)
    cameras = _cameras_in_plan(camera_cfgs, num_clones)

    # modify scene to be less stochastic
    stage = sim_utils.get_current_stage()
    for i in range(10):
        prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
        color = Gf.Vec3f(1, 1, 1)
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])

    # play sim
    sim.reset()

    # simulate some steps first to make sure objects are settled
    for i in range(100):
        # step simulation
        sim.step()
        # update cameras
        for camera in cameras:
            camera.update(dt)

    # collect image data
    image_befores = [camera.data.output["rgb"].clone() / 255.0 for camera in cameras]

    # update scene
    for i in range(10):
        prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
        color = Gf.Vec3f(0, 0, 0)
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])

    # update rendering
    sim.step()

    # update cameras
    for camera in cameras:
        camera.update(dt)

    # make sure the image is different
    image_afters = [camera.data.output["rgb"].clone() / 255.0 for camera in cameras]

    # check difference is above threshold
    for i in range(num_camera_groups):
        image_before = image_befores[i]
        image_after = image_afters[i]
        assert torch.abs(image_after - image_before).mean() > 0.02  # images of same color should be below 0.001

    for camera in cameras:
        del camera


@flaky(max_runs=3, min_passes=1)
@pytest.mark.isaacsim_ci
def test_frame_different_poses_multi_camera(setup_camera):
    """Test multiple batched cameras at different poses render different images."""
    camera_cfg, sim, dt = setup_camera
    num_camera_groups = 3
    num_clones = 4
    positions = [(0.0, 0.0, 4.0), (0.0, 0.0, 2.0), (0.0, 0.0, 3.0)]
    rotations = [(0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)]

    camera_cfgs = []
    for i in range(num_camera_groups):
        cfg = copy.deepcopy(camera_cfg)
        cfg.prim_path = f"{{ENV_REGEX_NS}}/CameraSensor_{i}"
        cfg.offset = CameraCfg.OffsetCfg(pos=positions[i], rot=rotations[i], convention="ros")
        camera_cfgs.append(cfg)
    cameras = _cameras_in_plan(camera_cfgs, num_clones)

    # Play sim
    sim.reset()

    # Simulate physics
    for _ in range(10):
        # Initialize data arrays
        rgbs = []
        distances = []

        # perform rendering
        sim.step()
        for camera in cameras:
            # update camera
            camera.update(dt)
            # check image data
            for data_type, im_data in camera.data.output.items():
                if data_type == "rgb":
                    im_data = im_data.clone() / 255.0
                    assert im_data.shape == (num_clones, camera.cfg.height, camera.cfg.width, 3)
                    for j in range(num_clones):
                        assert (im_data[j]).mean().item() > 0.0
                    rgbs.append(im_data)
                elif data_type == "distance_to_camera":
                    im_data = im_data.clone()
                    # replace inf with 0
                    im_data[torch.isinf(im_data)] = 0
                    assert im_data.shape == (num_clones, camera.cfg.height, camera.cfg.width, 1)
                    for j in range(num_clones):
                        assert im_data[j].mean().item() > 0.0
                    distances.append(im_data)

        # Check data from camera groups are different, assumes >1 group
        for i in range(1, num_camera_groups):
            assert torch.abs(rgbs[0] - rgbs[i]).mean() > 0.04  # images of same color should be below 0.001
            assert torch.abs(distances[0] - distances[i]).mean() > 0.01  # distances of same scene should be 0

    for camera in cameras:
        del camera


"""
Helper functions.
"""


def _populate_scene():
    """Add prims to the scene."""
    # TODO: this causes hang with Kit 107.3???
    # # Ground-plane
    # cfg = sim_utils.GroundPlaneCfg()
    # cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.SphereLightCfg()
    cfg.func("/World/Light/GreySphere", cfg, translation=(4.5, 3.5, 10.0))
    cfg.func("/World/Light/WhiteSphere", cfg, translation=(-4.5, 3.5, 10.0))
    # Random objects
    random.seed(0)
    for i in range(10):
        # sample random position
        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])
        # create prim
        prim_type = random.choice(["Cube", "Sphere", "Cylinder"])
        prim = sim_utils.create_prim(
            f"/World/Objects/Obj_{i:02d}",
            prim_type,
            translation=position,
            scale=(0.25, 0.25, 0.25),
            semantic_label=prim_type,
        )
        # cast to geom prim
        geom_prim = getattr(UsdGeom, prim_type)(prim)
        # set random color
        color = Gf.Vec3f(random.random(), random.random(), random.random())
        geom_prim.CreateDisplayColorAttr()
        geom_prim.GetDisplayColorAttr().Set([color])
        # add rigid body and collision properties using Isaac Lab schemas
        prim_path = f"/World/Objects/Obj_{i:02d}"
        sim_utils.define_rigid_body_properties(prim_path, sim_utils.RigidBodyPropertiesCfg())
        sim_utils.define_mass_properties(prim_path, sim_utils.MassPropertiesCfg(mass=5.0))
        sim_utils.define_collision_properties(prim_path, sim_utils.CollisionPropertiesCfg())
