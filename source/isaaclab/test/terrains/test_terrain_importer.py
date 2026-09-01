# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from typing import Literal

import numpy as np
import pytest
import trimesh

from pxr import UsdGeom

import isaaclab.terrains as terrain_gen
from isaaclab import cloner as lab_cloner
from isaaclab.sim import PreviewSurfaceCfg, build_simulation_context, get_first_matching_child_prim
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_terrain_generation(device):
    """Generates assorted terrains and tests that the resulting mesh has the correct size."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Handler for terrains importing
        terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            max_init_terrain_level=None,
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            num_envs=1,
        )
        with lab_cloner.ReplicateSession([lab_cloner.CloneCfg(), terrain_importer_cfg], 1, 1.0):
            terrain_importer = terrain_importer_cfg.class_type(terrain_importer_cfg)

        # check if mesh prim path exists
        mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
        assert mesh_prim_path in terrain_importer.terrain_prim_paths

        # obtain underling mesh
        mesh = _obtain_collision_mesh(mesh_prim_path, mesh_type="Mesh")
        assert mesh is not None

        # calculate expected size from config
        cfg = terrain_importer.cfg.terrain_generator
        assert cfg is not None
        expectedSizeX = cfg.size[0] * cfg.num_rows + 2 * cfg.border_width
        expectedSizeY = cfg.size[1] * cfg.num_cols + 2 * cfg.border_width

        # get size from mesh bounds
        bounds = mesh.bounds
        actualSize = abs(bounds[1] - bounds[0])

        assert actualSize[0] == pytest.approx(expectedSizeX)
        assert actualSize[1] == pytest.approx(expectedSizeY)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_custom_material", [True, False])
def test_plane(device, use_custom_material):
    """Generates a plane and tests that the resulting mesh has the correct size."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        # create custom material
        visual_material = PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)) if use_custom_material else None
        # Handler for terrains importing
        terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            num_envs=1,
            env_spacing=1.0,
            visual_material=visual_material,
        )
        with lab_cloner.ReplicateSession([lab_cloner.CloneCfg(), terrain_importer_cfg], 1, 1.0):
            terrain_importer = terrain_importer_cfg.class_type(terrain_importer_cfg)

        # check if mesh prim path exists
        mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
        assert mesh_prim_path in terrain_importer.terrain_prim_paths

        # obtain underling mesh
        mesh = _obtain_collision_mesh(mesh_prim_path, mesh_type="Plane")
        assert mesh is None


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_usd(device):
    """Imports terrain from a usd and tests that the resulting mesh has the correct size."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Handler for terrains importing
        terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
            num_envs=1,
            env_spacing=1.0,
        )
        with lab_cloner.ReplicateSession([lab_cloner.CloneCfg(), terrain_importer_cfg], 1, 1.0):
            terrain_importer = terrain_importer_cfg.class_type(terrain_importer_cfg)

        # check if mesh prim path exists
        mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
        assert mesh_prim_path in terrain_importer.terrain_prim_paths

        # obtain underling mesh
        mesh = _obtain_collision_mesh(mesh_prim_path, mesh_type="Mesh")
        assert mesh is not None

        # expect values from USD file
        expectedSizeX = 96
        expectedSizeY = 96

        # get size from mesh bounds
        bounds = mesh.bounds
        actualSize = abs(bounds[1] - bounds[0])

        assert actualSize[0] == pytest.approx(expectedSizeX)
        assert actualSize[1] == pytest.approx(expectedSizeY)


def _obtain_collision_mesh(mesh_prim_path: str, mesh_type: Literal["Mesh", "Plane"]) -> trimesh.Trimesh | None:
    """Get the collision mesh from the terrain."""
    # traverse the prim and get the collision mesh
    mesh_prim = get_first_matching_child_prim(mesh_prim_path, lambda prim: prim.GetTypeName() == mesh_type)
    # check it is valid
    assert mesh_prim.IsValid()

    if mesh_prim.GetTypeName() == "Mesh":
        # cast into UsdGeomMesh
        mesh_prim = UsdGeom.Mesh(mesh_prim)
        # store the mesh
        vertices = np.asarray(mesh_prim.GetPointsAttr().Get())
        faces = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        return None
