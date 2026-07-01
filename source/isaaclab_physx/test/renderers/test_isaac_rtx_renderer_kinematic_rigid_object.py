# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematic rigid-object rendering regression tests.

These tests exercise the PhysX/Fabric-to-RTX synchronization path for kinematic
rigid bodies. Isaac Sim 5.0 could update a kinematic body's physics transform
without updating its rendered transform, and could lose non-uniform scale while
synchronizing the body into Fabric.

Launch Isaac Sim Simulator first.
"""

from isaaclab.app import AppLauncher

# launch omniverse app -- cameras are required to read back RTX depth.
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import numpy as np
import pytest
import torch
from isaaclab_physx.assets import Articulation, RigidObject

import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import build_simulation_context
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

pytestmark = pytest.mark.isaacsim_ci

_NUM_ENVS = 2
_ENV_SPACING = 4.0
_CAMERA_DISTANCE = 2.0
_CAMERA_HEIGHT = 120
_CAMERA_WIDTH = 160
_OBJECT_SCALE = (1.0, 1.0, 8.0)
_OBJECT_SHIFT = 0.45
_MAX_OBJECT_DEPTH = 10.0
_MIN_OBJECT_PIXELS = 50
_MIN_CENTROID_SHIFT = 10.0


def _spawn_scene(with_articulation: bool) -> tuple[RigidObject, Articulation | None, Camera]:
    """Spawn two root-scaled kinematic cubes and one camera per environment."""
    for env_id in range(_NUM_ENVS):
        sim_utils.create_prim(
            f"/World/envs/env_{env_id}",
            "Xform",
            translation=(env_id * _ENV_SPACING, 0.0, 0.0),
        )

    rigid_object = RigidObject(
        RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                rigid_props=[UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True)],
                scale=_OBJECT_SCALE,
            ),
        )
    )

    articulation = None
    if with_articulation:
        articulation = Articulation(
            ArticulationCfg(
                prim_path="/World/envs/env_.*/Articulation",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=(f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd")
                ),
                init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, -20.0, 0.0)),
                actuators={
                    "joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=1.0),
                },
            )
        )

    camera = Camera(
        CameraCfg(
            prim_path="/World/envs/env_.*/Camera",
            height=_CAMERA_HEIGHT,
            width=_CAMERA_WIDTH,
            update_period=0.0,
            update_latest_camera_pose=True,
            data_types=["distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 100.0),
            ),
        )
    )
    return rigid_object, articulation, camera


def _measure_depth_mask(depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the silhouette height, width, and horizontal centroid for each camera."""
    valid = torch.isfinite(depth[..., 0]) & (depth[..., 0] < _MAX_OBJECT_DEPTH)
    pixel_counts = valid.sum(dim=(1, 2))
    assert torch.all(pixel_counts >= _MIN_OBJECT_PIXELS), (
        f"Expected at least {_MIN_OBJECT_PIXELS} object pixels per camera, got {pixel_counts.tolist()}."
    )

    silhouette_heights = valid.any(dim=2).sum(dim=1)
    silhouette_widths = valid.any(dim=1).sum(dim=1)
    image_x = torch.arange(depth.shape[2], device=depth.device, dtype=torch.float32)
    centroids_x = (valid * image_x.view(1, 1, -1)).sum(dim=(1, 2)) / pixel_counts
    return silhouette_heights, silhouette_widths, centroids_x


def _write_pose_and_render(sim, rigid_object: RigidObject, camera: Camera, root_poses: torch.Tensor) -> torch.Tensor:
    """Write a rigid-object pose, advance one frame, and return rendered depth."""
    rigid_object.write_root_pose_to_sim_index(root_pose=root_poses)
    sim.step()
    rigid_object.update(sim.cfg.dt)
    camera.update(sim.cfg.dt)
    torch.testing.assert_close(
        rigid_object.data.root_link_pose_w.torch,
        root_poses,
        rtol=0.0,
        atol=1.0e-4,
    )
    return camera.data.output["distance_to_image_plane"].torch.clone()


@pytest.mark.parametrize("with_articulation", [False, True])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_kinematic_rigid_object_scale_and_pose_are_rendered(device, with_articulation):
    """Root scale and public pose writes must reach RTX for a kinematic rigid object."""
    with build_simulation_context(device=device) as sim:
        sim._app_control_on_stop_handle = None
        rigid_object, articulation, camera = _spawn_scene(with_articulation)

        try:
            sim.reset()
            assert rigid_object.is_initialized
            if articulation is not None:
                assert articulation.is_initialized

            env_x = np.arange(_NUM_ENVS, dtype=np.float32) * _ENV_SPACING
            camera_eyes = np.column_stack(
                (env_x, np.full(_NUM_ENVS, -_CAMERA_DISTANCE, dtype=np.float32), np.zeros(_NUM_ENVS, dtype=np.float32))
            )
            camera_targets = np.column_stack(
                (env_x, np.zeros(_NUM_ENVS, dtype=np.float32), np.zeros(_NUM_ENVS, dtype=np.float32))
            )
            camera.set_world_poses_from_view(camera_eyes, camera_targets)

            center_poses = torch.zeros((_NUM_ENVS, 7), device=rigid_object.device)
            center_poses[:, 0] = torch.arange(_NUM_ENVS, device=rigid_object.device) * _ENV_SPACING
            center_poses[:, 6] = 1.0

            # A lost root scale would render the DexCube nearly square instead of tall.
            center_depth = _write_pose_and_render(sim, rigid_object, camera, center_poses)
            center_heights, center_widths, center_centroids = _measure_depth_mask(center_depth)
            assert torch.all(center_heights > 3 * center_widths), (
                "Expected the root-scaled kinematic cubes to render as tall silhouettes, got "
                f"heights={center_heights.tolist()} and widths={center_widths.tolist()}."
            )
            assert torch.all(center_heights > _CAMERA_HEIGHT // 4), (
                "Expected root-scaled silhouettes to span more than one quarter of the image, got "
                f"heights={center_heights.tolist()}."
            )
            assert torch.all(torch.abs(center_centroids - (_CAMERA_WIDTH - 1) / 2) < 8.0), (
                f"Expected centered silhouettes, got horizontal centroids {center_centroids.tolist()}."
            )

            # Moving the kinematic body to opposite sides must move the rendered silhouette too.
            negative_poses = center_poses.clone()
            negative_poses[:, 0] -= _OBJECT_SHIFT
            negative_depth = _write_pose_and_render(sim, rigid_object, camera, negative_poses)
            _, _, negative_centroids = _measure_depth_mask(negative_depth)

            positive_poses = center_poses.clone()
            positive_poses[:, 0] += _OBJECT_SHIFT
            positive_depth = _write_pose_and_render(sim, rigid_object, camera, positive_poses)
            _, _, positive_centroids = _measure_depth_mask(positive_depth)

            negative_delta = negative_centroids - center_centroids
            positive_delta = positive_centroids - center_centroids
            assert torch.all(negative_delta.abs() > _MIN_CENTROID_SHIFT), (
                f"Rendered negative-shift centroids moved too little: {negative_delta.tolist()}."
            )
            assert torch.all(positive_delta.abs() > _MIN_CENTROID_SHIFT), (
                f"Rendered positive-shift centroids moved too little: {positive_delta.tolist()}."
            )
            assert torch.all(negative_delta * positive_delta < 0.0), (
                "Expected opposite object translations to move rendered silhouettes in opposite directions, got "
                f"deltas {negative_delta.tolist()} and {positive_delta.tolist()}."
            )
        finally:
            del camera, rigid_object, articulation
            rep.vp_manager.destroy_hydra_textures("Replicator")
