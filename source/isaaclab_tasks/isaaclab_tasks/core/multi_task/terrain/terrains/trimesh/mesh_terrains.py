# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import io
import os
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import requests
import scipy.spatial.transform as tf
import torch
import trimesh
import yaml
from scipy.spatial.transform import Rotation as R

from isaaclab.terrains.trimesh.mesh_terrains import inverted_pyramid_stairs_terrain, pyramid_stairs_terrain
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshInvertedPyramidStairsTerrainCfg, MeshPyramidStairsTerrainCfg
from isaaclab.terrains.trimesh.utils import make_border, make_plane

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def _box_transformed(extents, transform: np.ndarray) -> trimesh.Trimesh:
    """Create a box and apply a transform without stochastic mesh processing."""
    mesh = trimesh.creation.box(extents)
    mesh.vertices = trimesh.transform_points(mesh.vertices, transform)
    return mesh


def _cylinder_transformed(radius: float, height: float, *, sections: int, transform: np.ndarray) -> trimesh.Trimesh:
    """Create a cylinder and apply a transform without stochastic mesh processing."""
    mesh = trimesh.creation.cylinder(radius, height, sections=sections)
    mesh.vertices = trimesh.transform_points(mesh.vertices, transform)
    return mesh


def obj_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshObjTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray] | tuple[list[trimesh.Trimesh], np.ndarray]:
    mesh: trimesh.Trimesh = trimesh.load(cfg.obj_path)  # type: ignore
    xy_scale = cfg.size / (mesh.bounds[1] - mesh.bounds[0])[:2]
    # set the height scale to the average between length and width scale to preserve as much original shap as possible
    height_scale = (xy_scale[0] + xy_scale[1]) / 2
    xyz_scale = np.array([*xy_scale, height_scale])
    mesh.apply_scale(xyz_scale)
    translation = -mesh.bounds[0]
    mesh.apply_translation(translation)

    extend = mesh.bounds[1] - mesh.bounds[0]
    origin = (*((extend[:2]) / 2), mesh.bounds[1][2] / 2)

    if isinstance(cfg.spawn_origin_path, str):
        spawning_option = np.load(cfg.spawn_origin_path, allow_pickle=True)
        spawning_option *= xyz_scale
        spawning_option += translation
        # insert the center of the terrain as the first indices
        # the rest of the indices represents the spawning locations
        return [mesh], np.insert(spawning_option, 0, origin, axis=0)
    else:
        return [mesh], np.array(origin)


def terrain_gen(
    difficulty: float, cfg: mesh_terrains_cfg.TerrainGenCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray] | tuple[list[trimesh.Trimesh], np.ndarray]:
    terrain_yaml = {
        "terrain": {
            "shape": [int(cfg.size[0] / 2), int(cfg.size[1] / 2)],
            "height": cfg.height,
            "levels": cfg.levels,
            "include_overhang": cfg.include_overhang,
            "all_terrain_styles": ["stair", "ramp", "box", "platform", "random_box", "perlin", "wall"],
            "terrain_styles": cfg.terrain_styles,
        }
    }
    terrain_style = "_".join(cfg.terrain_styles)
    os.makedirs(os.path.dirname(cfg.yaml_path), exist_ok=True)
    yaml_file_path = cfg.yaml_path.replace(".yaml", f"_{terrain_style}.yaml")
    with open(yaml_file_path, "w") as file:
        yaml.dump(terrain_yaml, file, default_flow_style=False)

    mesh_origin_dir = os.path.dirname(cfg.obj_path)
    mesh_dir = os.path.dirname(mesh_origin_dir)
    # Prepare the command and arguments for the subprocess
    command = [
        "python",
        cfg.python_script,
        "--input_path",
        yaml_file_path,
        "--enable_sdf",
        "--mesh_dir",
        mesh_dir,
        "--mesh_name",
        f"{terrain_style}",
    ]

    # Invoke the subprocess and run the other script
    try:
        result = subprocess.run(command, check=True, capture_output=True)
        print("Subprocess completed successfully!")
        print("Output:", result.stdout.decode())
        print("Errors:", result.stderr.decode())
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with error: {e}")
        print(f"Subprocess output: {e.output.decode()}")
        print(f"Subprocess stderr: {e.stderr.decode()}")

    return obj_terrain(difficulty, cfg)


def load_mesh(terrain_mesh_path: str) -> trimesh.Trimesh:
    """Load a mesh from a URL or a local file."""
    if terrain_mesh_path.startswith("http"):
        # Load from URL
        response = requests.get(terrain_mesh_path)
        if response.status_code == 200:
            mesh = trimesh.load(io.BytesIO(response.content), file_type="obj")
            return mesh  # type: ignore
        else:
            raise Exception(f"Failed to load mesh from {terrain_mesh_path}")
    else:
        # Load from local path
        return trimesh.load(terrain_mesh_path)  # type: ignore


def load_numpy(spawnfile_path: str) -> np.ndarray:
    """Load a NumPy array from a URL or a local file."""
    if spawnfile_path.startswith("http"):
        # Load from URL
        response = requests.get(spawnfile_path)
        if response.status_code == 200:
            data = np.load(io.BytesIO(response.content), allow_pickle=True)
            return data
        else:
            raise Exception(f"Failed to load NumPy file from {spawnfile_path}")
    else:
        # Load from local path
        return np.load(spawnfile_path, allow_pickle=True)


def stones_everywhere_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshStonesEverywhereTerrainCfg, *, rng: np.random.Generator
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    # check to ensure square terrain
    assert cfg.size[0] == cfg.size[1], "The terrain should be square"

    # resolve the terrain configuration based on the difficulty
    gap_width = cfg.w_gap[0] + difficulty * (cfg.w_gap[1] - cfg.w_gap[0])
    stone_width = cfg.w_stone[0] - difficulty * (cfg.w_stone[0] - cfg.w_stone[1])
    s_max = cfg.s_max[0] + difficulty * (cfg.s_max[1] - cfg.s_max[0])
    h_max = cfg.h_max[0] + difficulty * (cfg.h_max[1] - cfg.h_max[0])

    # initialize list of meshes
    meshes_list = list()

    # compute the number of stones in x and y directions
    num_stones_axis = int(cfg.size[0] / (gap_width + stone_width))

    # constants
    terrain_height = -cfg.holes_depth
    device = torch.device("cpu")

    # generate the border
    border_width = cfg.size[0] - num_stones_axis * (gap_width + stone_width)
    if border_width > 0:
        border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
        border_inner_size = (cfg.size[0] - border_width, cfg.size[1] - border_width)
        # create border meshes
        make_borders = make_border(cfg.size, border_inner_size, terrain_height, border_center)
        meshes_list += make_borders
    # create a template grid of the terrain height
    grid_dim = [stone_width, stone_width, terrain_height]
    grid_position = [0.5 * (stone_width + gap_width), 0.5 * (stone_width + gap_width), -terrain_height / 2]
    template_box = _box_transformed(grid_dim, trimesh.transformations.translation_matrix(grid_position))
    # extract vertices and faces
    template_vertices = template_box.vertices  # (8, 3)
    template_faces = template_box.faces

    # repeat the template box vertices to space the terrain(num_boxes_axis**2, 8, 3)
    vertices = torch.tensor(template_vertices, device=device).repeat(num_stones_axis**2, 1, 1)
    # create a meshgrid to offset the vertices
    x = torch.arange(0, num_stones_axis, device=device)
    y = torch.arange(0, num_stones_axis, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xx = xx.flatten().view(-1, 1)
    yy = yy.flatten().view(-1, 1)
    xx_yy = torch.cat((xx, yy), dim=1)
    # offset the vertices
    offsets = (
        (stone_width + gap_width) * xx_yy
        + border_width / 2
        + torch.as_tensor(rng.uniform(-s_max, s_max, size=tuple(xx_yy.shape)), device=device)
    )
    vertices[:, :, :2] += offsets.unsqueeze(1)

    # add noise on height
    num_boxes = len(vertices)
    h_noise = torch.zeros((num_boxes, 3), device=device)
    h_noise[:, 2].copy_(torch.as_tensor(rng.uniform(-h_max, h_max, size=num_boxes), device=device))
    # reshape noise to match the vertices (num_boxes, 4, 3)
    # only top vertices are affected
    vertices_noise = torch.zeros((num_boxes, 4, 3), device=device)
    vertices_noise += h_noise.unsqueeze(1)
    # add height only to the top vertices of the box
    vertices[vertices[:, :, 2] == 0] += vertices_noise.view(-1, 3)
    # move to numpy
    vertices = vertices.reshape(-1, 3).cpu().numpy()

    # create faces for boxes(num_boxes, 12, 3), each box has 6 faces, each face has 2 triangles
    faces = torch.tensor(template_faces, device=device).repeat(num_boxes, 1, 1)
    face_offsets = torch.arange(0, num_boxes, device=device).unsqueeze(1).repeat(1, 12) * 8
    faces += face_offsets.unsqueeze(2)
    faces = faces.view(-1, 3).cpu().numpy()

    # convert to trimesh
    grid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    meshes_list.append(grid_mesh)

    # add a platform in the center of the terrain that is accessible from all sides
    dim = (cfg.platform_width, cfg.platform_width, terrain_height + h_max)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2 + h_max / 2)
    box_platform = _box_transformed(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_platform)

    # specify the origin of the terrain
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], h_max])

    return meshes_list, origin


def balance_beams_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshBalanceBeamsTerrainCfg, *, rng: np.random.Generator
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    # check to ensure square terrain
    assert cfg.size[0] == cfg.size[1], "The terrain should be square"

    stone_width = cfg.w_stone[0] - difficulty * (cfg.w_stone[0] - cfg.w_stone[1])
    h_offset = cfg.h_offset[0] + difficulty * (cfg.h_offset[1] - cfg.h_offset[0])
    mid_gap = cfg.mid_gap[0] + difficulty * (cfg.mid_gap[1] - cfg.mid_gap[0])

    meshes_list = list()
    num_stones = int(((cfg.size[0] - 0.25 - cfg.platform_width) / 2 - 1) / stone_width)

    terrain_height = 1
    device = torch.device("cpu")

    border_width = (cfg.size[1] - cfg.platform_width) / 2 - 1 - num_stones * stone_width
    if border_width > 0:
        border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
        border_inner_size = (cfg.size[0] - border_width - 2, cfg.size[1] - border_width - 2)
        # create border meshes
        make_borders = make_border(cfg.size, border_inner_size, terrain_height, border_center)
        meshes_list += make_borders

    grid_dim = [stone_width, stone_width, terrain_height]
    grid_position = [0.5 * stone_width, 0.5 * stone_width, -0.5 * terrain_height]
    template_box = _box_transformed(grid_dim, trimesh.transformations.translation_matrix(grid_position))
    # extract vertices and faces
    template_vertices = template_box.vertices  # (8, 3)
    template_faces = template_box.faces
    # repeat the template box vertices to space the terrain(num_stones, 8, 3)
    vertices = torch.tensor(template_vertices, device=device).repeat(num_stones, 1, 1)
    index = torch.arange(0, num_stones, device=device)
    vertices[:, :, 0] += cfg.size[0] / 2 + cfg.platform_width / 2
    vertices[:, :, 0] += (index * stone_width).unsqueeze(-1)
    vertices[(index % 2) == 0, :, 1] += cfg.size[1] / 2 - mid_gap / 2 - stone_width / 2
    vertices[(index % 2) == 1, :, 1] += cfg.size[1] / 2 + mid_gap / 2 - stone_width / 2

    num_boxes = len(vertices)
    h_noise = torch.zeros((num_boxes, 3), device=device)
    h_noise[:, 2].copy_(torch.as_tensor(rng.uniform(-h_offset, h_offset, size=num_boxes), device=device))
    # reshape noise to match the vertices (num_boxes, 4, 3)
    # only top vertices are affected
    vertices_noise = torch.zeros((num_boxes, 4, 3), device=device)
    vertices_noise += h_noise.unsqueeze(1)
    # add height only to the top vertices of the box
    vertices[vertices[:, :, 2] == 0] += vertices_noise.view(-1, 3)
    # move to numpy
    vertices = vertices.reshape(-1, 3).cpu().numpy()

    # create faces for boxes(num_boxes, 12, 3), each box has 6 faces, each face has 2 triangles
    faces = torch.tensor(template_faces, device=device).repeat(num_boxes, 1, 1)
    face_offsets = torch.arange(0, num_boxes, device=device).unsqueeze(1).repeat(1, 12) * 8
    faces += face_offsets.unsqueeze(2)
    faces = faces.view(-1, 3).cpu().numpy()

    # convert to trimesh
    grid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # randomly rotate beams around the terrain center to vary the approach direction
    rotation_angle = rng.choice([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
    if rotation_angle:
        rotation_transform = trimesh.transformations.rotation_matrix(
            rotation_angle,
            [0.0, 0.0, 1.0],
            [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0],
        )
        grid_mesh.vertices = trimesh.transform_points(grid_mesh.vertices, rotation_transform)
    meshes_list.append(grid_mesh)

    # add a platform in the center of the terrain that is accessible from all sides
    dim = (cfg.platform_width, cfg.platform_width, terrain_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    box_platform = _box_transformed(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_platform)

    # specify the origin of the terrain
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0])

    return meshes_list, origin


def stepping_beams_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshSteppingBeamsTerrainCfg, *, rng: np.random.Generator
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    stone_width = cfg.w_stone[0] - difficulty * (cfg.w_stone[0] - cfg.w_stone[1])
    h_offset = cfg.h_offset[0] + difficulty * (cfg.h_offset[1] - cfg.h_offset[0])
    gap_width = cfg.gap[0] + difficulty * (cfg.gap[1] - cfg.gap[0])
    yaw = cfg.yaw[0] + difficulty * (cfg.yaw[1] - cfg.yaw[0])
    assert cfg.yaw[0] < cfg.yaw[1], "The yaw range should be in ascending order(0 means no yaw)"
    low_stone_l = cfg.l_stone[0]
    high_stone_l = cfg.l_stone[1]

    meshes_list = list()
    num_stones = int(((cfg.size[0] - cfg.platform_width) / 2) / (gap_width + stone_width))

    terrain_height = 1

    border_width = (cfg.size[1] - cfg.platform_width) / 2 - num_stones * (stone_width + gap_width)

    if border_width > 0:
        border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
        border_inner_size = (cfg.size[0] - border_width - 2, cfg.size[1] - border_width - 2)
        # create border meshes
        make_borders = make_border(cfg.size, border_inner_size, terrain_height, border_center)
        meshes_list += make_borders
    # calculate the center of all the stones
    # add random noise to the center
    # add noise to the height
    # create the stones
    for i in range(num_stones):
        transform = np.eye(4)
        grid_dim = [
            stone_width,
            low_stone_l + rng.uniform(0, high_stone_l - low_stone_l),
            terrain_height + rng.uniform(-h_offset, h_offset),
        ]
        center = [
            cfg.size[0] / 2
            + cfg.platform_width / 2
            + (i + 1) * gap_width
            + (i + 0.5) * stone_width
            + rng.uniform(-0.25, 0.25) * gap_width,
            cfg.size[1] / 2 + rng.uniform(-0.1, 0.1) * grid_dim[1],
            -terrain_height / 2,
        ]
        transform[0:3, -1] = np.asarray(center)
        # create rotation matrix
        transform[0:3, 0:3] = R.from_euler("z", rng.uniform(-yaw, yaw), degrees=True).as_matrix()
        meshes_list.append(_box_transformed(grid_dim, transform))
    # add a platform in the center of the terrain that is accessible from all sides
    dim = (cfg.platform_width, cfg.platform_width, terrain_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    box_platform = _box_transformed(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_platform)

    # specify the origin of the terrain
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0])

    return meshes_list, origin


def box_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshDiversityBoxTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    #
    box_width = cfg.box_width_range[1] - difficulty * (cfg.box_width_range[1] - cfg.box_width_range[0])
    box_length = cfg.box_length_range[1] - difficulty * (cfg.box_length_range[1] - cfg.box_length_range[0])
    box_height = cfg.box_height_range[0] + difficulty * (cfg.box_height_range[1] - cfg.box_height_range[0])
    meshes_list = []
    terrain_height = 1.0
    middle_height = 0.0
    # check if box_gap_range is a tuple
    if isinstance(cfg.box_gap_range, tuple):
        # Task of jumping over neighboring boxes
        gap_width = cfg.box_gap_range[0] + difficulty * (cfg.box_gap_range[1] - cfg.box_gap_range[0])
        # generate the box at the origin
        box_dim = (box_width, box_length, box_height + terrain_height)
        pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2 + box_height / 2)
        box = _box_transformed(box_dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(box)
        # generate the neighboring boxes
        box_dim = (box_width, box_length, box_height + terrain_height)
        offset_x = box_width / 2 + box_width / 2 + gap_width
        pos = (cfg.size[0] / 2 + offset_x, cfg.size[1] / 2, -terrain_height / 2 + box_height / 2)
        box = _box_transformed(box_dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(box)
        middle_height = box_height
    elif cfg.box_gap_range is None:
        # Task for climbing up or down boxes
        if cfg.up_or_down == "up":
            # for climbing up
            box_dim = (box_width, box_length, box_height + terrain_height)
            offset_x = box_width
            pos = (cfg.size[0] / 2 + offset_x, cfg.size[1] / 2, -terrain_height / 2 + box_height / 2)
            box = _box_transformed(box_dim, trimesh.transformations.translation_matrix(pos))
            meshes_list.append(box)
            middle_height = 0.0
        elif cfg.up_or_down == "down":
            # for climbing down
            box_dim = (box_width, box_length, box_height + terrain_height)
            pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2 + box_height / 2)
            box = _box_transformed(box_dim, trimesh.transformations.translation_matrix(pos))
            meshes_list.append(box)
            middle_height = box_height
        else:
            raise ValueError("up_or_down should be either 'up' or 'down'")
    else:
        raise ValueError("box_gap_range should be a tuple or None")

    # generate the ground
    pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground = _box_transformed(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground)

    # specify the origin of the terrain
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, middle_height])

    return meshes_list, origin


def passage_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPassageTerrainCfg, *, rng: np.random.Generator
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    if isinstance(cfg.passage_width, tuple):
        width = cfg.passage_width[1] - difficulty * (cfg.passage_width[1] - cfg.passage_width[0])
    elif isinstance(cfg.passage_width, float):
        width = cfg.passage_width
    else:
        raise ValueError("passage_width should be a tuple or a float")
    if isinstance(cfg.passage_length, tuple):
        length = cfg.passage_length[0] + difficulty * (cfg.passage_length[1] - cfg.passage_length[0])
    elif isinstance(cfg.passage_length, float):
        length = cfg.passage_length
    else:
        raise ValueError("passage_length should be a tuple or a float")
    if isinstance(cfg.passage_height, tuple):
        height = cfg.passage_height[1] - difficulty * (cfg.passage_height[1] - cfg.passage_height[0])
    elif isinstance(cfg.passage_height, float):
        height = cfg.passage_height
    else:
        raise ValueError("passage_height should be a tuple or a float")
    # generate the passage
    meshes_list = []
    terrain_height = 1.0
    offset_x = 1.0
    # four legs of the passage
    dim = (0.05 + rng.uniform(0.0, 0.1), 0.05 + rng.uniform(0.0, 0.1), terrain_height + height)
    pos1 = (offset_x + cfg.size[0] / 2 - length / 2, cfg.size[1] / 2 - width / 2, -terrain_height / 2 + height / 2)
    box1 = _box_transformed(dim, trimesh.transformations.translation_matrix(pos1))
    meshes_list.append(box1)
    pos2 = (offset_x + cfg.size[0] / 2 - length / 2, cfg.size[1] / 2 + width / 2, -terrain_height / 2 + height / 2)
    box2 = _box_transformed(dim, trimesh.transformations.translation_matrix(pos2))
    meshes_list.append(box2)
    pos3 = (offset_x + cfg.size[0] / 2 + length / 2, cfg.size[1] / 2 - width / 2, -terrain_height / 2 + height / 2)
    box3 = _box_transformed(dim, trimesh.transformations.translation_matrix(pos3))
    meshes_list.append(box3)
    pos4 = (offset_x + cfg.size[0] / 2 + length / 2, cfg.size[1] / 2 + width / 2, -terrain_height / 2 + height / 2)
    box4 = _box_transformed(dim, trimesh.transformations.translation_matrix(pos4))
    meshes_list.append(box4)
    # top of the passage
    dim = (length + dim[0], width + dim[1], 0.05 + rng.uniform(0, 0.1))
    pos = (offset_x + cfg.size[0] / 2, cfg.size[1] / 2, dim[2] / 2 + height)
    top = _box_transformed(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(top)
    # ground
    pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground = _box_transformed(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground)

    # specify the origin of the terrain
    origin = np.array([cfg.size[0] / 2 - 1.0, cfg.size[1] / 2, 0.0])

    return meshes_list, origin


def structured_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshStructuredTerrainCfg, *, rng: np.random.Generator
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    mesh_list = []
    terrain = cfg.terrain_type
    # generate the terrain
    if terrain == "obstacles":
        origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
        for i in range(12):
            if i < 8:
                length = rng.uniform(0.2, 2.0)
                width = rng.uniform(0.2, 2.0)
                height = rng.uniform(0.08, 0.25)
            else:
                length = rng.uniform(0.2, 1.0)
                width = rng.uniform(0.2, 1.0)
                height = 3.0
            center = (
                cfg.size[0] / 2 + rng.uniform(1, cfg.size[0] / 2) * (-1) ** (rng.integers(1, 3)),
                cfg.size[1] / 2 + rng.uniform(1, cfg.size[0] / 2) * (-1) ** (rng.integers(1, 3)),
                height / 2,
            )
            transform = np.eye(4)
            transform[0:3, -1] = np.asarray(center)
            # create the box
            dims = (length, width, height)
            mesh = _box_transformed(dims, transform=transform)
            mesh_list.append(mesh)
        # add walls
        if rng.uniform(0, 1) > 0.1:
            center_pts = [(0, 0, 0), (cfg.size[0], 0, 0), (0, cfg.size[1], 0), (cfg.size[0], cfg.size[1], 0)]
            for i, center in enumerate(center_pts):
                if rng.uniform(0, 1) > 0.5:
                    continue
                length = cfg.size[0] * rng.uniform(0.2, 0.4)
                width = cfg.size[1] * rng.uniform(0.2, 0.4)
                height = 6.0
                transform = np.eye(4)
                c = (center[0] + (-1) ** i * length / 2, center[1] + (-1) ** (i // 2) * width / 2, center[2])
                transform[0:3, -1] = np.asarray(c)
                # create the box
                dims = (length, width, height)
                mesh = _box_transformed(dims, transform=transform)
                mesh_list.append(mesh)
        # add plane
        ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
        mesh_list.append(ground_plane)

    elif terrain == "stairs":
        step_width = rng.uniform(0.2, 0.5)
        _mesh_list, origin = pyramid_stairs_terrain(
            difficulty,
            MeshPyramidStairsTerrainCfg(
                size=cfg.size,
                border_width=1.0,
                step_height_range=(0.08, 0.20),
                step_width=step_width,
                platform_width=2.0,
            ),
        )
        mesh_list += _mesh_list
        # add walls
        if rng.uniform(0, 1) > 0.05:
            center_pts = [(0, 0, 0), (cfg.size[0], 0, 0), (0, cfg.size[1], 0), (cfg.size[0], cfg.size[1], 0)]
            for i, center in enumerate(center_pts):
                if rng.uniform(0, 1) > 0.75:
                    continue
                length = cfg.size[0] * rng.uniform(0.3, 0.4)
                width = cfg.size[1] * rng.uniform(0.3, 0.4)
                height = 6.0
                transform = np.eye(4)
                c = (center[0] + (-1) ** i * length / 2, center[1] + (-1) ** (i // 2) * width / 2, center[2])
                transform[0:3, -1] = np.asarray(c)
                # create the box
                dims = (length, width, height)
                mesh = _box_transformed(dims, transform=transform)
                mesh_list.append(mesh)
    elif terrain == "inverted_stairs":
        step_width = rng.uniform(0.2, 0.5)
        # inverted prymaid
        _mesh_list, origin = inverted_pyramid_stairs_terrain(
            difficulty,
            MeshInvertedPyramidStairsTerrainCfg(
                size=cfg.size,
                border_width=1.0,
                step_height_range=(0.08, 0.20),
                step_width=step_width,
                platform_width=2.0,
            ),
        )
        mesh_list += _mesh_list
        # add walls
        if rng.uniform(0, 1) > 0.05:
            center_pts = [(0, 0, 0), (cfg.size[0], 0, 0), (0, cfg.size[1], 0), (cfg.size[0], cfg.size[1], 0)]
            for i, center in enumerate(center_pts):
                if rng.uniform(0, 1) > 0.75:
                    continue
                length = cfg.size[0] * rng.uniform(0.3, 0.4)
                width = cfg.size[1] * rng.uniform(0.3, 0.4)
                height = 6.0
                transform = np.eye(4)
                c = (center[0] + (-1) ** i * length / 2, center[1] + (-1) ** (i // 2) * width / 2, center[2])
                transform[0:3, -1] = np.asarray(c)
                # create the box
                dims = (length, width, height)
                mesh = _box_transformed(dims, transform=transform)
                mesh_list.append(mesh)
    elif terrain == "walls":
        origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
        # add walls
        center_pts = [(0, 0, 0), (cfg.size[0], 0, 0), (0, cfg.size[1], 0), (cfg.size[0], cfg.size[1], 0)]
        for i, center in enumerate(center_pts):
            if rng.uniform(0, 1) > 0.75:
                continue
            length = cfg.size[0] * rng.uniform(0.3, 0.4)
            width = cfg.size[1] * rng.uniform(0.3, 0.4)
            height = 6.0
            transform = np.eye(4)
            c = (center[0] + (-1) ** i * length / 2, center[1] + (-1) ** (i // 2) * width / 2, center[2])
            transform[0:3, -1] = np.asarray(c)
            # create the box
            dims = (length, width, height)
            mesh = _box_transformed(dims, transform=transform)
            mesh_list.append(mesh)
        # add plane
        ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
        mesh_list.append(ground_plane)
    else:
        raise ValueError(f"terrain_type {terrain} is not supported")
    # update the origin in a free space
    return mesh_list, origin


def maze_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshMazeTerrainCfg, *, rng: np.random.Generator
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a maze terrain on a 2D grid.

    Uses iterative DFS (recursive backtracker) to carve passages, then
    extrudes remaining wall segments into axis-aligned boxes.

    Args:
        difficulty: Difficulty parameter in [0, 1].
        cfg: Terrain configuration.

    Returns:
        A list of trimesh meshes and the terrain origin [m].
    """
    from isaaclab.terrains.trimesh.utils import make_plane

    cols = int(_resolve_range(cfg.grid_cols, difficulty))
    rows = int(_resolve_range(cfg.grid_rows, difficulty))
    cols = max(cols, 2)
    rows = max(rows, 2)
    wall_h = _resolve_range(cfg.wall_height, difficulty)
    thick = cfg.wall_thickness

    cell_w = cfg.size[0] / cols
    cell_h = cfg.size[1] / rows

    # walls stored as sets of (col, row, direction) where direction is 'h' or 'v'
    # 'h' = horizontal wall on the SOUTH edge of cell (col, row): from (col, row+1) to (col+1, row+1)
    # 'v' = vertical wall on the EAST edge of cell (col, row): from (col+1, row) to (col+1, row+1)
    # plus boundary walls
    h_walls: set[tuple[int, int]] = set()  # (col, row) means wall between row and row+1 at col
    v_walls: set[tuple[int, int]] = set()  # (col, row) means wall between col and col+1 at row

    # initialize all interior walls
    for c in range(cols):
        for r in range(rows - 1):
            h_walls.add((c, r))  # horizontal wall below cell (c, r)
    for c in range(cols - 1):
        for r in range(rows):
            v_walls.add((c, r))  # vertical wall right of cell (c, r)

    # iterative DFS maze carver
    visited = [[False] * rows for _ in range(cols)]
    stack: list[tuple[int, int]] = []
    start_c, start_r = cols // 2, rows // 2
    visited[start_c][start_r] = True
    stack.append((start_c, start_r))

    while stack:
        c, r = stack[-1]
        neighbors = []
        if c > 0 and not visited[c - 1][r]:
            neighbors.append((c - 1, r, "v", c - 1, r))  # remove v_wall at (c-1, r)
        if c < cols - 1 and not visited[c + 1][r]:
            neighbors.append((c + 1, r, "v", c, r))  # remove v_wall at (c, r)
        if r > 0 and not visited[c][r - 1]:
            neighbors.append((c, r - 1, "h", c, r - 1))  # remove h_wall at (c, r-1)
        if r < rows - 1 and not visited[c][r + 1]:
            neighbors.append((c, r + 1, "h", c, r))  # remove h_wall at (c, r)

        if neighbors:
            nc, nr, wtype, wc, wr = neighbors[int(rng.integers(len(neighbors)))]
            if wtype == "h":
                h_walls.discard((wc, wr))
            else:
                v_walls.discard((wc, wr))
            visited[nc][nr] = True
            stack.append((nc, nr))
        else:
            stack.pop()

    # optionally remove extra walls to open up the maze
    if cfg.open_ratio > 0:
        all_remaining = [(wc, wr, "h") for wc, wr in h_walls] + [(wc, wr, "v") for wc, wr in v_walls]
        rng.shuffle(all_remaining)
        to_remove = int(len(all_remaining) * cfg.open_ratio)
        for wc, wr, wtype in all_remaining[:to_remove]:
            if wtype == "h":
                h_walls.discard((wc, wr))
            else:
                v_walls.discard((wc, wr))

    # build node grid with optional jitter
    # nodes[c][r] = (x, y) position of grid node at column c, row r
    # grid has (cols+1) x (rows+1) nodes
    grid_noise = getattr(cfg, "grid_noise", 0.0)
    nodes = [[None] * (rows + 1) for _ in range(cols + 1)]
    for c in range(cols + 1):
        for r in range(rows + 1):
            x = c * cell_w
            y = r * cell_h
            if grid_noise > 0 and 0 < c < cols and 0 < r < rows:
                x += rng.uniform(-grid_noise, grid_noise) * cell_w
                y += rng.uniform(-grid_noise, grid_noise) * cell_h
            nodes[c][r] = np.array([x, y])

    # build wall segments as (p0, p1) pairs
    segments: list[tuple[np.ndarray, np.ndarray]] = []

    for c, r in h_walls:
        segments.append((nodes[c][r + 1], nodes[c + 1][r + 1]))
    for c, r in v_walls:
        segments.append((nodes[c + 1][r], nodes[c + 1][r + 1]))

    if getattr(cfg, "boundary_walls", False):
        for c in range(cols):
            for r_edge in [0, rows]:
                segments.append((nodes[c][r_edge], nodes[c + 1][r_edge]))
        for r in range(rows):
            for c_edge in [0, cols]:
                segments.append((nodes[c_edge][r], nodes[c_edge][r + 1]))

    # create a single watertight mesh by unioning 2D wall rectangles then extruding
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    half_t = thick / 2
    polys = []
    for p0, p1 in segments:
        d = p1 - p0
        length = np.linalg.norm(d)
        if length < 1e-6:
            continue
        perp = np.array([-d[1], d[0]]) / length * half_t
        corners = [
            p0 - perp,
            p0 + perp,
            p1 + perp,
            p1 - perp,
        ]
        polys.append(Polygon(corners))

    meshes: list[trimesh.Trimesh] = []
    if polys:
        merged = unary_union(polys)
        if merged.geom_type == "MultiPolygon":
            parts = list(merged.geoms)
        else:
            parts = [merged]
        for poly in parts:
            wall_mesh = trimesh.creation.extrude_polygon(poly, wall_h)
            meshes.append(wall_mesh)

    # ground plane
    meshes.append(make_plane(cfg.size, height=0.0, center_zero=False))

    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])
    return meshes, origin


# ======================================================================
# Stone generator
# ======================================================================


def _generate_stone(
    radius: float, height_scale: float, roughness: float, *, rng: np.random.Generator
) -> trimesh.Trimesh:
    """Generate a natural-looking stone mesh with varied shapes.

    Randomly produces round boulders, elongated slabs, angular blocks,
    and occasionally tilted/standing stones.
    """
    # choose base shape: round (icosphere) or angular (box-ish)
    angular = rng.random() < 0.3
    if angular:
        stone = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        # chamfer edges by subdividing and projecting partially toward sphere
        stone = stone.subdivide()
        verts = stone.vertices.copy()
        norms_v = np.linalg.norm(verts, axis=1, keepdims=True)
        sphere_verts = verts / np.maximum(norms_v, 1e-12)
        blend = rng.uniform(0.15, 0.5)
        verts = verts * (1 - blend) + sphere_verts * blend * norms_v.max()
    else:
        stone = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        verts = stone.vertices.copy()

    # random axis scaling — wider range for more variety
    sx = rng.uniform(0.4, 1.6)
    sy = rng.uniform(0.4, 1.6)
    # occasionally tall/standing stones vs flat/squat
    if rng.random() < 0.2:
        sz = height_scale * rng.uniform(1.2, 2.5)
    else:
        sz = height_scale * rng.uniform(0.3, 1.0)
    verts[:, 0] *= sx
    verts[:, 1] *= sy
    verts[:, 2] *= sz

    # random 3D rotation (yaw + optional pitch/roll tilt)
    yaw = rng.uniform(0, 2 * np.pi)
    pitch = rng.uniform(-0.4, 0.4) if rng.random() < 0.4 else 0.0
    roll = rng.uniform(-0.4, 0.4) if rng.random() < 0.4 else 0.0
    rot = tf.Rotation.from_euler("zyx", [yaw, pitch, roll]).as_matrix()
    verts = verts @ rot.T

    # low-frequency smooth deformation (3 random lobes)
    norms_v = np.linalg.norm(verts, axis=1, keepdims=True)
    unit = verts / np.maximum(norms_v, 1e-12)
    angles = np.arctan2(verts[:, 1], verts[:, 0])
    for _ in range(3):
        freq = rng.uniform(1.5, 3.5)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.05, roughness * 0.7)
        verts += unit * (amp * np.cos(freq * angles + phase))[:, None]

    # fine surface noise
    fine_noise = rng.uniform(-roughness * 0.3, roughness * 0.3, size=len(verts))
    unit = verts / np.maximum(np.linalg.norm(verts, axis=1, keepdims=True), 1e-12)
    verts += unit * fine_noise[:, None]

    # flatten bottom so stone sits on ground
    verts[:, 2] = np.maximum(verts[:, 2], verts[:, 2].min() * 0.3)

    verts *= radius
    stone.vertices = verts
    return stone


# ======================================================================
# Contour terrain
# ======================================================================


def _perlin_2d(shape: tuple[int, int], scale: float, octaves: int, seed: int) -> np.ndarray:
    """Generate a 2D fractal noise field using value noise with smoothstep interpolation."""
    rng = np.random.default_rng(seed)
    rows, cols = shape
    result = np.zeros(shape, dtype=np.float64)

    for o in range(octaves):
        freq = int(max(2, scale * (2**o)))
        amp = 0.5**o

        # random values at grid nodes
        grid = rng.uniform(0, 1, (freq + 1, freq + 1))

        # sample positions in grid space
        gy = np.linspace(0, freq - 1e-9, rows)
        gx = np.linspace(0, freq - 1e-9, cols)

        # integer and fractional parts
        iy = gy.astype(int)
        ix = gx.astype(int)
        fy = gy - iy
        fx = gx - ix

        # smoothstep
        sy = fy * fy * (3 - 2 * fy)
        sx = fx * fx * (3 - 2 * fx)

        # vectorized bilinear interpolation with smoothstep
        iy1 = np.minimum(iy + 1, freq)
        ix1 = np.minimum(ix + 1, freq)
        v00 = grid[np.ix_(iy, ix)]
        v10 = grid[np.ix_(iy, ix1)]
        v01 = grid[np.ix_(iy1, ix)]
        v11 = grid[np.ix_(iy1, ix1)]
        wx = sx[np.newaxis, :]
        wy = sy[:, np.newaxis]
        layer = (v00 * (1 - wx) + v10 * wx) * (1 - wy) + (v01 * (1 - wx) + v11 * wx) * wy

        result += amp * layer

    result -= result.min()
    rng_val = result.max() - result.min()
    if rng_val > 1e-9:
        result /= rng_val
    return result


def contour_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshContourTerrainCfg, *, rng: np.random.Generator
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate stepped contour terrain from a noise heightfield.

    Args:
        difficulty: Difficulty parameter in [0, 1].
        cfg: Terrain configuration.

    Returns:
        A list of trimesh meshes and the terrain origin [m].
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from isaaclab.terrains.trimesh.utils import make_plane

    num_levels = int(_resolve_range(cfg.num_levels, difficulty))
    level_h = _resolve_range(cfg.level_height, difficulty)
    seed = cfg.noise_seed if cfg.noise_seed is not None else int(rng.integers(0, 2**31 + 1))
    smoothing = cfg.smoothing

    res = 100
    noise_field = _perlin_2d((res, res), cfg.noise_scale, cfg.noise_octaves, seed)

    x_coords = np.linspace(0, cfg.size[0], res)
    y_coords = np.linspace(0, cfg.size[1], res)

    thresholds = np.linspace(0.1, 0.9, num_levels)

    meshes: list[trimesh.Trimesh] = []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for level_idx, thresh in enumerate(thresholds):
        fig, ax = plt.subplots()
        cs = ax.contourf(x_coords, y_coords, noise_field, levels=[thresh, 1.1])
        plt.close(fig)

        polys = []
        for level_segs in cs.allsegs:
            for seg in level_segs:
                if len(seg) < 3:
                    continue
                try:
                    p = Polygon(seg)
                    if smoothing > 0:
                        p = p.simplify(smoothing, preserve_topology=True)
                    if p.is_valid and p.area > 0.1:
                        polys.append(p)
                except Exception:
                    continue

        if not polys:
            continue

        merged = unary_union(polys)
        if merged.geom_type == "MultiPolygon":
            parts = list(merged.geoms)
        else:
            parts = [merged]

        z_base = level_idx * level_h
        for poly in parts:
            if poly.is_empty or poly.area < 0.1:
                continue
            step_mesh = trimesh.creation.extrude_polygon(poly, level_h)
            step_mesh.apply_translation([0, 0, z_base])
            meshes.append(step_mesh)

    # scatter stones on terraces
    stones_cfg = getattr(cfg, "stones", None)
    if stones_cfg is not None:
        n_stones = int(_resolve_range(stones_cfg.num_stones, difficulty))
        s_min, s_max = stones_cfg.size_range

        for _ in range(n_stones):
            sx = rng.uniform(0, cfg.size[0])
            sy = rng.uniform(0, cfg.size[1])
            # find which terrace level this point sits on
            gi = min(int(sx / cfg.size[0] * res), res - 1)
            gj = min(int(sy / cfg.size[1] * res), res - 1)
            nval = noise_field[gj, gi]
            stone_level = 0
            for thresh in thresholds:
                if nval >= thresh:
                    stone_level += 1
            sz = stone_level * level_h

            radius = rng.uniform(s_min, s_max)
            stone = _generate_stone(radius, stones_cfg.height_scale, stones_cfg.roughness, rng=rng)
            stone_bottom = stone.vertices[:, 2].min()
            stone.apply_translation([sx, sy, sz - stone_bottom])
            meshes.append(stone)

    meshes.append(make_plane(cfg.size, height=0.0, center_zero=False))

    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])
    return meshes, origin


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from *a* to *b* by factor *t*."""
    return a + t * (b - a)


def _resolve_range(value: float | tuple[float, float], difficulty: float) -> float:
    """If *value* is a ``(start, end)`` tuple, interpolate by *difficulty*."""
    if isinstance(value, tuple):
        return _lerp(value[0], value[1], difficulty)
    return float(value)


def _sample_yaw_angles(
    num_bars: int,
    distribution: str,
    bar_width: float,
    platform_radius: float,
    *,
    rng: np.random.Generator,
) -> list[float]:
    """Return a sorted list of yaw angles for the beams."""
    if distribution == "uniform":
        return [i * (2 * np.pi) / num_bars for i in range(num_bars)]

    if distribution == "random":
        min_sep = 1.2 * bar_width / platform_radius
        angles: list[float] = []

        for i in range(num_bars):
            for _ in range(1000):
                candidate = rng.uniform(0, 2 * np.pi)
                if all(min(abs(candidate - a), 2 * np.pi - abs(candidate - a)) >= min_sep for a in angles):
                    angles.append(candidate)
                    break
            else:
                angles.append(i * (2 * np.pi) / num_bars)
        angles.sort()
        return angles

    raise ValueError(f"Invalid beam_distribution '{distribution}'. Expected 'uniform' or 'random'.")


def _beam_reach(yaw: float, half_x: float, half_y: float) -> float:
    """Distance from center to a rectangle edge along direction *yaw*."""
    cos_yaw = abs(np.cos(yaw))
    sin_yaw = abs(np.sin(yaw))
    if cos_yaw < 1e-9:
        return half_y
    if sin_yaw < 1e-9:
        return half_x
    return min(half_x / cos_yaw, half_y / sin_yaw)


def _bezier_point(p0: np.ndarray, p1: np.ndarray, ctrl: np.ndarray, t: float) -> np.ndarray:
    """Evaluate a quadratic bezier curve at parameter *t* in [0, 1]."""
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t**2 * p1


def _bezier_tangent(p0: np.ndarray, p1: np.ndarray, ctrl: np.ndarray, t: float) -> np.ndarray:
    """Tangent direction of a quadratic bezier at parameter *t*."""
    tang = 2 * (1 - t) * (ctrl - p0) + 2 * t * (p1 - ctrl)
    norm = np.linalg.norm(tang)
    return tang / norm if norm > 1e-9 else np.array([1.0, 0.0])


def _generate_passway(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    z_start: float,
    z_end: float,
    bar_width: float,
    bar_height: float,
    beam_style_cfg,
    difficulty: float,
    *,
    rng: np.random.Generator,
    start_inset: float = 0.0,
    curvature: float = 0.0,
) -> list[trimesh.Trimesh]:
    """Generate a passway (flat or box-segmented) between two 2D points.

    Args:
        start_xy: Start point ``(x, y)`` in world frame.
        end_xy: End point ``(x, y)`` in world frame.
        z_start: Box-center z at the start end.
        z_end: Box-center z at the far end.
        bar_width: Width of the passway [m].
        bar_height: Thickness (z extent) of each box [m].
        beam_style_cfg: A :class:`FlatBeamCfg` or :class:`BoxBeamCfg` instance.
        difficulty: Difficulty in [0, 1] for resolving range fields.
        start_inset: Distance to inset the first segment from the start point [m].
        curvature: Maximum lateral offset of the bezier control point as a
            fraction of the straight-line distance. 0 means straight.

    Returns:
        A list of trimesh meshes composing the passway.
    """
    delta = end_xy - start_xy
    straight_length = float(np.linalg.norm(delta))
    if straight_length < 1e-6:
        return []

    # build bezier control point
    mid_xy = (start_xy + end_xy) / 2
    if curvature > 0:
        perp = np.array([-delta[1], delta[0]]) / straight_length
        lateral = rng.uniform(-curvature, curvature) * straight_length
        ctrl_xy = mid_xy + perp * lateral
    else:
        ctrl_xy = mid_xy

    is_box = type(beam_style_cfg).__name__ == "BoxBeamCfg"
    meshes: list[trimesh.Trimesh] = []
    transform = np.eye(4)

    if not is_box and curvature == 0:
        direction = delta / straight_length
        yaw = float(np.arctan2(direction[1], direction[0]))
        rot = tf.Rotation.from_euler("z", yaw).as_matrix()
        mid_z = (z_start + z_end) / 2
        z_drop = z_start - z_end
        pitch = np.arctan2(z_drop, straight_length) if abs(z_drop) > 1e-6 else 0.0
        beam_rot = rot @ tf.Rotation.from_euler("y", pitch).as_matrix()
        transform[0:3, 0:3] = beam_rot
        transform[:3, -1] = [mid_xy[0], mid_xy[1], mid_z]
        meshes.append(_box_transformed([straight_length, bar_width, bar_height], transform.copy()))
    else:
        if not is_box:
            box_length = bar_width
            box_gap_val = 0.0
            pos_var = (0.0, 0.0, 0.0)
            yaw_var = 0.0
        else:
            box_length = _resolve_range(beam_style_cfg.box_length, difficulty)
            box_gap_val = _resolve_range(beam_style_cfg.box_gap, difficulty)
            pos_var = beam_style_cfg.box_position_variation
            yaw_var = beam_style_cfg.box_yaw_variation

        stride = box_length + box_gap_val
        n_across = max(1, int(bar_width / stride)) if bar_width >= box_length * 2 else 1
        across_span = n_across * box_length + (n_across - 1) * box_gap_val
        y_start_off = -across_span / 2 + box_length / 2

        # walk along the bezier by arc-length approximation
        arc_len = 0.0
        prev_pt = start_xy.copy()
        num_samples = max(int(straight_length / (box_length * 0.5)), 20)
        arc_table: list[tuple[float, float]] = [(0.0, 0.0)]
        for si in range(1, num_samples + 1):
            t_s = si / num_samples
            pt = _bezier_point(start_xy, end_xy, ctrl_xy, t_s)
            arc_len += float(np.linalg.norm(pt - prev_pt))
            arc_table.append((arc_len, t_s))
            prev_pt = pt
        total_arc = arc_len

        def arc_to_t(s: float) -> float:
            for k in range(1, len(arc_table)):
                s0, t0 = arc_table[k - 1]
                s1, t1 = arc_table[k]
                if s <= s1:
                    frac = (s - s0) / (s1 - s0) if s1 > s0 else 0.0
                    return t0 + frac * (t1 - t0)
            return 1.0

        cursor = start_inset
        while cursor + box_length <= total_arc:
            s_center = cursor + box_length / 2
            t_param = arc_to_t(s_center)
            center_xy = _bezier_point(start_xy, end_xy, ctrl_xy, t_param)
            tangent = _bezier_tangent(start_xy, end_xy, ctrl_xy, t_param)
            seg_yaw = float(np.arctan2(tangent[1], tangent[0]))
            seg_z = _lerp(z_start, z_end, t_param)

            seg_rot_base = tf.Rotation.from_euler("z", seg_yaw).as_matrix()

            for j in range(n_across):
                dx = rng.uniform(-pos_var[0], pos_var[0]) if pos_var[0] > 0 else 0.0
                dy = rng.uniform(-pos_var[1], pos_var[1]) if pos_var[1] > 0 else 0.0
                dz = rng.uniform(-pos_var[2], pos_var[2]) if pos_var[2] > 0 else 0.0
                x_stagger = (stride * 0.5) * (j % 2) if n_across > 1 else 0.0

                local_offset = seg_rot_base @ np.array([x_stagger + dx, y_start_off + j * stride + dy, 0.0])
                world = np.array([center_xy[0] + local_offset[0], center_xy[1] + local_offset[1], seg_z + dz])

                seg_rot = (
                    seg_rot_base @ tf.Rotation.from_euler("z", rng.uniform(-yaw_var, yaw_var)).as_matrix()
                    if yaw_var > 0
                    else seg_rot_base
                )
                transform[0:3, 0:3] = seg_rot
                transform[:3, -1] = world
                box_w = box_length if is_box else bar_width
                meshes.append(_box_transformed([box_length, box_w, bar_height], transform.copy()))
            cursor += stride

    return meshes


def beam_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshRadiatingBeamTerrainCfg, *, rng: np.random.Generator
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with beams radiating from a central platform to an outer border.

    Args:
        difficulty: Difficulty parameter in [0, 1].
        cfg: Terrain configuration.

    Returns:
        A list of trimesh meshes and the terrain origin [m].
    """
    bar_height = _resolve_range(cfg.bar_height_range, difficulty)
    bar_width = cfg.bar_width_range[0] - difficulty * (cfg.bar_width_range[0] - cfg.bar_width_range[1])
    num_bars = int(_resolve_range(cfg.num_bars, difficulty))

    platform_elevation = _resolve_range(cfg.platform_height, difficulty) if cfg.platform_height is not None else 0.0
    noise = getattr(cfg, "platform_height_noise", 0.0)
    if noise > 0:
        platform_elevation += rng.uniform(-noise, noise)

    platform_radius = cfg.platform_width * 0.5
    terrain_center = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1]])

    plat_top = platform_elevation
    border_top = 0.0
    z_inner = plat_top - bar_height / 2
    z_outer = border_top - bar_height / 2

    # -- central platform --
    meshes: list[trimesh.Trimesh] = []
    plat_bottom = min(plat_top, border_top) - bar_height
    plat_total_h = plat_top - plat_bottom
    plat_center_z = (plat_top + plat_bottom) / 2
    plat_tf = trimesh.transformations.translation_matrix((*terrain_center, plat_center_z))
    meshes.append(_cylinder_transformed(platform_radius, plat_total_h, sections=6, transform=plat_tf))

    yaw_angles = _sample_yaw_angles(num_bars, cfg.beam_distribution, bar_width, max(platform_radius, 0.1), rng=rng)

    border_cfg = getattr(cfg, "border", None)
    if border_cfg is not None and hasattr(border_cfg, "inner_size"):
        border_half = np.array(border_cfg.inner_size) * 0.5
    else:
        border_half = np.array(cfg.size) * 0.5

    beam_style_cfg = getattr(cfg, "beam_style", None) or mesh_terrains_cfg.FlatBeamCfg()

    for yaw in yaw_angles:
        beam_length = _beam_reach(yaw, border_half[0], border_half[1])
        direction = np.array([np.cos(yaw), np.sin(yaw)])
        start_xy = terrain_center + direction * platform_radius * 0.85
        end_xy = terrain_center + direction * beam_length
        meshes += _generate_passway(
            start_xy, end_xy, z_inner, z_outer, bar_width, bar_height, beam_style_cfg, difficulty, rng=rng
        )

    # -- border geometry --
    border_center_z = border_top - bar_height / 2
    if border_cfg is not None:
        border_type = type(border_cfg).__name__
        if border_type == "SquareBorderCfg":
            meshes += make_border(cfg.size, border_cfg.inner_size, bar_height, (*terrain_center, border_center_z))
        elif border_type == "PlatformBorderCfg":
            for yaw in yaw_angles:
                end_dist = _beam_reach(yaw, border_half[0], border_half[1]) + border_cfg.radius
                end_xy = terrain_center + np.array([np.cos(yaw), np.sin(yaw)]) * end_dist
                p_tf = trimesh.transformations.translation_matrix((*end_xy, border_center_z))
                meshes.append(_cylinder_transformed(border_cfg.radius, bar_height, sections=6, transform=p_tf))

    if getattr(cfg, "ground_plane", True):
        ground_z = min(plat_bottom, border_top - bar_height)
        meshes.append(make_plane(cfg.size, ground_z, center_zero=False))

    origin = np.array([terrain_center[0], terrain_center[1], max(plat_top, border_top)])
    return meshes, origin


# ======================================================================
# Floating island terrain
# ======================================================================


def _sample_island_positions(
    num: int,
    size: tuple[float, float],
    bounding_radius: float,
    margin: float,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rejection-sample *num* non-overlapping 2D positions within *size*."""
    pad = bounding_radius + margin
    positions: list[np.ndarray] = []
    min_dist = 2 * bounding_radius + margin

    for _ in range(num * 2000):
        pt = np.array(
            [
                rng.uniform(pad, size[0] - pad),
                rng.uniform(pad, size[1] - pad),
            ]
        )
        if all(np.linalg.norm(pt - p) >= min_dist for p in positions):
            positions.append(pt)
            if len(positions) >= num:
                break

    return np.array(positions)


def _build_graph(
    positions: np.ndarray,
    graph_cfg,
) -> set[tuple[int, int]]:
    """Build a set of undirected edges ``{(i, j), ...}`` over island positions."""
    from scipy.spatial import Delaunay

    n = len(positions)
    if n < 2:
        return set()

    graph_type = type(graph_cfg).__name__

    if graph_type == "DelaunayGraphCfg":
        tri = Delaunay(positions)
        edges: set[tuple[int, int]] = set()
        for simplex in tri.simplices:
            for a, b in [(0, 1), (1, 2), (0, 2)]:
                edge = (min(simplex[a], simplex[b]), max(simplex[a], simplex[b]))
                edges.add(edge)
        return edges

    dists = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)

    if graph_type == "MSTGraphCfg":
        visited = {0}
        edges = set()
        while len(visited) < n:
            best_dist = float("inf")
            best_edge = (0, 0)
            for v in visited:
                for u in range(n):
                    if u not in visited and dists[v, u] < best_dist:
                        best_dist = dists[v, u]
                        best_edge = (min(v, u), max(v, u))
            edges.add(best_edge)
            visited.add(best_edge[0])
            visited.add(best_edge[1])
        return edges

    if graph_type == "KNNGraphCfg":
        k = min(getattr(graph_cfg, "k", 3), n - 1)
        edges = set()
        for i in range(n):
            neighbors = np.argsort(dists[i])[1 : k + 1]
            for j in neighbors:
                edges.add((min(i, int(j)), max(i, int(j))))
        return edges

    raise ValueError(f"Unknown graph config type: {graph_type}")


def _prune_edges_through_islands(
    edges: set[tuple[int, int]],
    positions: np.ndarray,
    radii: np.ndarray,
) -> set[tuple[int, int]]:
    """Remove edges whose line segment passes through a third island."""
    pruned: set[tuple[int, int]] = set()

    for i, j in edges:
        p0, p1 = positions[i], positions[j]
        seg = p1 - p0
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-6:
            continue
        seg_dir = seg / seg_len

        blocked = False
        for k in range(len(positions)):
            if k in (i, j):
                continue
            # distance from island k center to the line segment
            v = positions[k] - p0
            proj = float(np.dot(v, seg_dir))
            proj = max(radii[i], min(seg_len - radii[j], proj))
            closest = p0 + seg_dir * proj
            dist = float(np.linalg.norm(positions[k] - closest))
            if dist < radii[k] * 1.1:
                blocked = True
                break

        if not blocked:
            pruned.add((i, j))

    return pruned


def floating_island_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshFloatingIslandTerrainCfg, *, rng: np.random.Generator
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain of floating islands connected by passways.

    Args:
        difficulty: Difficulty parameter in [0, 1].
        cfg: Terrain configuration.

    Returns:
        A list of trimesh meshes and the terrain origin [m].
    """
    num_islands = int(_resolve_range(cfg.num_islands, difficulty))
    island_height = cfg.island_height
    passway_height = cfg.passway_height if cfg.passway_height is not None else island_height
    passway_width = _resolve_range(cfg.passway_width, difficulty)
    height_var = cfg.island_height_variation

    island_cfg = cfg.island_style
    is_cylinder = type(island_cfg).__name__ == "CylinderIslandCfg"

    if is_cylinder:
        island_radius = _resolve_range(island_cfg.radius, difficulty)
        bounding_r = island_radius
    else:
        island_length = _resolve_range(island_cfg.length, difficulty)
        island_width_val = _resolve_range(island_cfg.width, difficulty)
        bounding_r = np.sqrt(island_length**2 + island_width_val**2) / 2

    positions = _sample_island_positions(num_islands, cfg.size, bounding_r, cfg.island_margin, rng=rng)
    actual_n = len(positions)

    z_offsets = np.array([rng.uniform(-height_var, height_var) if height_var > 0 else 0.0 for _ in range(actual_n)])

    # -- generate island meshes --
    meshes: list[trimesh.Trimesh] = []
    radii = np.full(actual_n, bounding_r)

    for idx in range(actual_n):
        center_z = z_offsets[idx] - island_height / 2
        if is_cylinder:
            island_tf = trimesh.transformations.translation_matrix((*positions[idx], center_z))
            meshes.append(_cylinder_transformed(island_radius, island_height, sections=8, transform=island_tf))
        else:
            yaw = rng.uniform(0, 2 * np.pi)
            rot = tf.Rotation.from_euler("z", yaw).as_matrix()
            xf = np.eye(4)
            xf[0:3, 0:3] = rot
            xf[:3, -1] = [positions[idx][0], positions[idx][1], center_z]
            meshes.append(_box_transformed([island_length, island_width_val, island_height], xf.copy()))

    # -- build and prune graph --
    edges = _build_graph(positions, cfg.graph)
    edges = _prune_edges_through_islands(edges, positions, radii)

    # -- generate passways --
    passway_style = cfg.passway_style
    passway_curvature = getattr(cfg, "passway_curvature", 0.0)

    for i, j in edges:
        delta = positions[j] - positions[i]
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            continue
        direction = delta / dist

        start_xy = positions[i] + direction * bounding_r
        end_xy = positions[j] - direction * bounding_r

        z_start = z_offsets[i] - passway_height / 2
        z_end = z_offsets[j] - passway_height / 2

        meshes += _generate_passway(
            start_xy,
            end_xy,
            z_start,
            z_end,
            passway_width,
            passway_height,
            passway_style,
            difficulty,
            rng=rng,
            curvature=passway_curvature,
        )

    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])
    return meshes, origin
