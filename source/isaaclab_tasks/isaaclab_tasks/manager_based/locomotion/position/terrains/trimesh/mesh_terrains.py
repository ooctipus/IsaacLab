# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import io
import numpy as np
import os
import random
import scipy.spatial.transform as tf
import subprocess
import torch
import trimesh
import yaml
from scipy.spatial.transform import Rotation as R
from typing import TYPE_CHECKING

import requests
from isaaclab.terrains.trimesh.mesh_terrains import inverted_pyramid_stairs_terrain, pyramid_stairs_terrain
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshInvertedPyramidStairsTerrainCfg, MeshPyramidStairsTerrainCfg
from isaaclab.terrains.trimesh.utils import make_border, make_plane


if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def obj_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshObjTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray] | tuple[list[trimesh.Trimesh], np.ndarray]:
    mesh: trimesh.Trimesh = trimesh.load(cfg.obj_path)  # type: ignore
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
            return mesh  # type: ignore
        else:
            raise Exception(f"Failed to load mesh from {terrain_mesh_path}")
    else:
        # Load from local path
        return trimesh.load(terrain_mesh_path)  # type: ignore

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
    difficulty: float, cfg: mesh_terrains_cfg.MeshStonesEverywhereTerrainCfg
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    template_box = trimesh.creation.box(grid_dim, trimesh.transformations.translation_matrix(grid_position))
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
        + (2 * torch.rand(*xx_yy.shape, device=xx_yy.device) - 1) * s_max
    )
    vertices[:, :, :2] += offsets.unsqueeze(1)

    # add noise on height
    num_boxes = len(vertices)
    h_noise = torch.zeros((num_boxes, 3), device=device)
    h_noise[:, 2].uniform_(-h_max, h_max)
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
    box_platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_platform)

    # specify the origin of the terrain
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], h_max])

    return meshes_list, origin


def balance_beams_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshBalanceBeamsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    # check to ensure square terrain
    assert cfg.size[0] == cfg.size[1], "The terrain should be square"

    stone_width = cfg.w_stone[0] - difficulty * (cfg.w_stone[0] - cfg.w_stone[1])
    h_offset = cfg.h_offset[0] + difficulty * (cfg.h_offset[1] - cfg.h_offset[0])
    mid_gap = cfg.mid_gap[0] + difficulty * (cfg.mid_gap[1] - cfg.mid_gap[0])

    meshes_list = list()
    num_stones = int(((cfg.size[0] - 0.25 - cfg.platform_width) / 2 - 1) / stone_width)

    terrain_height = 1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    border_width = (cfg.size[1] - cfg.platform_width) / 2 - 1 - num_stones * stone_width
    if border_width > 0:
        border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
        border_inner_size = (cfg.size[0] - border_width - 2, cfg.size[1] - border_width - 2)
        # create border meshes
        make_borders = make_border(cfg.size, border_inner_size, terrain_height, border_center)
        meshes_list += make_borders

    grid_dim = [stone_width, stone_width, terrain_height]
    grid_position = [0.5 * stone_width, 0.5 * stone_width, -0.5 * terrain_height]
    template_box = trimesh.creation.box(grid_dim, trimesh.transformations.translation_matrix(grid_position))
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
    h_noise[:, 2].uniform_(-h_offset, h_offset)
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
    rotation_angle = random.choice([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
    if rotation_angle:
        rotation_transform = trimesh.transformations.rotation_matrix(
            rotation_angle,
            [0.0, 0.0, 1.0],
            [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0],
        )
        grid_mesh.apply_transform(rotation_transform)
    meshes_list.append(grid_mesh)

    # add a platform in the center of the terrain that is accessible from all sides
    dim = (cfg.platform_width, cfg.platform_width, terrain_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    box_platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_platform)

    # specify the origin of the terrain
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0])

    return meshes_list, origin


def stepping_beams_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshSteppingBeamsTerrainCfg
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
            low_stone_l + random.uniform(0, high_stone_l - low_stone_l),
            terrain_height + random.uniform(-h_offset, h_offset),
        ]
        center = [
            cfg.size[0] / 2
            + cfg.platform_width / 2
            + (i + 1) * gap_width
            + (i + 0.5) * stone_width
            + random.uniform(-0.25, 0.25) * gap_width,
            cfg.size[1] / 2 + random.uniform(-0.1, 0.1) * grid_dim[1],
            -terrain_height / 2,
        ]
        transform[0:3, -1] = np.asarray(center)
        # create rotation matrix
        transform[0:3, 0:3] = R.from_euler("z", random.uniform(-yaw, yaw), degrees=True).as_matrix()
        meshes_list.append(trimesh.creation.box(grid_dim, transform))
    # add a platform in the center of the terrain that is accessible from all sides
    dim = (cfg.platform_width, cfg.platform_width, terrain_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    box_platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
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
        box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(box)
        # generate the neighboring boxes
        box_dim = (box_width, box_length, box_height + terrain_height)
        offset_x = box_width / 2 + box_width / 2 + gap_width
        pos = (cfg.size[0] / 2 + offset_x, cfg.size[1] / 2, -terrain_height / 2 + box_height / 2)
        box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(box)
        middle_height = box_height
    elif cfg.box_gap_range is None:
        # Task for climbing up or down boxes
        if cfg.up_or_down == "up":
            # for climbing up
            box_dim = (box_width, box_length, box_height + terrain_height)
            offset_x = box_width
            pos = (cfg.size[0] / 2 + offset_x, cfg.size[1] / 2, -terrain_height / 2 + box_height / 2)
            box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(pos))
            meshes_list.append(box)
            middle_height = 0.0
        elif cfg.up_or_down == "down":
            # for climbing down
            box_dim = (box_width, box_length, box_height + terrain_height)
            pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2 + box_height / 2)
            box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(pos))
            meshes_list.append(box)
            middle_height = box_height
        else:
            raise ValueError("up_or_down should be either 'up' or 'down'")
    else:
        raise ValueError("box_gap_range should be a tuple or None")

    # generate the ground
    pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground)

    # specify the origin of the terrain
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, middle_height])

    return meshes_list, origin


def passage_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPassageTerrainCfg
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
    dim = (0.05 + np.random.uniform(0.0, 0.1), 0.05 + np.random.uniform(0.0, 0.1), terrain_height + height)
    pos1 = (offset_x + cfg.size[0] / 2 - length / 2, cfg.size[1] / 2 - width / 2, -terrain_height / 2 + height / 2)
    box1 = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos1))
    meshes_list.append(box1)
    pos2 = (offset_x + cfg.size[0] / 2 - length / 2, cfg.size[1] / 2 + width / 2, -terrain_height / 2 + height / 2)
    box2 = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos2))
    meshes_list.append(box2)
    pos3 = (offset_x + cfg.size[0] / 2 + length / 2, cfg.size[1] / 2 - width / 2, -terrain_height / 2 + height / 2)
    box3 = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos3))
    meshes_list.append(box3)
    pos4 = (offset_x + cfg.size[0] / 2 + length / 2, cfg.size[1] / 2 + width / 2, -terrain_height / 2 + height / 2)
    box4 = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos4))
    meshes_list.append(box4)
    # top of the passage
    dim = (length + dim[0], width + dim[1], 0.05 + np.random.uniform(0, 0.1))
    pos = (offset_x + cfg.size[0] / 2, cfg.size[1] / 2, dim[2] / 2 + height)
    top = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(top)
    # ground
    pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground)

    # specify the origin of the terrain
    origin = np.array([cfg.size[0] / 2 - 1.0, cfg.size[1] / 2, 0.0])

    return meshes_list, origin


def structured_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshStructuredTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    mesh_list = []
    terrain = cfg.terrain_type
    # generate the terrain
    if terrain == "obstacles":
        origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
        for i in range(12):
            if i < 8:
                length = random.uniform(0.2, 2.0)
                width = random.uniform(0.2, 2.0)
                height = random.uniform(0.08, 0.25)
            else:
                length = random.uniform(0.2, 1.0)
                width = random.uniform(0.2, 1.0)
                height = 3.0
            center = (
                cfg.size[0] / 2 + random.uniform(1, cfg.size[0] / 2) * (-1) ** (random.randint(1, 2)),
                cfg.size[1] / 2 + random.uniform(1, cfg.size[0] / 2) * (-1) ** (random.randint(1, 2)),
                height / 2,
            )
            transform = np.eye(4)
            transform[0:3, -1] = np.asarray(center)
            # create the box
            dims = (length, width, height)
            mesh = trimesh.creation.box(dims, transform=transform)
            mesh_list.append(mesh)
        # add walls
        if random.uniform(0, 1) > 0.1:
            center_pts = [(0, 0, 0), (cfg.size[0], 0, 0), (0, cfg.size[1], 0), (cfg.size[0], cfg.size[1], 0)]
            for i, center in enumerate(center_pts):
                if random.uniform(0, 1) > 0.5:
                    continue
                length = cfg.size[0] * random.uniform(0.2, 0.4)
                width = cfg.size[1] * random.uniform(0.2, 0.4)
                height = 6.0
                transform = np.eye(4)
                c = (center[0] + (-1) ** i * length / 2, center[1] + (-1) ** (i // 2) * width / 2, center[2])
                transform[0:3, -1] = np.asarray(c)
                # create the box
                dims = (length, width, height)
                mesh = trimesh.creation.box(dims, transform=transform)
                mesh_list.append(mesh)
        # add plane
        ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
        mesh_list.append(ground_plane)

    elif terrain == "stairs":
        step_width = random.uniform(0.2, 0.5)
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
        if random.uniform(0, 1) > 0.05:
            center_pts = [(0, 0, 0), (cfg.size[0], 0, 0), (0, cfg.size[1], 0), (cfg.size[0], cfg.size[1], 0)]
            for i, center in enumerate(center_pts):
                if random.uniform(0, 1) > 0.75:
                    continue
                length = cfg.size[0] * random.uniform(0.3, 0.4)
                width = cfg.size[1] * random.uniform(0.3, 0.4)
                height = 6.0
                transform = np.eye(4)
                c = (center[0] + (-1) ** i * length / 2, center[1] + (-1) ** (i // 2) * width / 2, center[2])
                transform[0:3, -1] = np.asarray(c)
                # create the box
                dims = (length, width, height)
                mesh = trimesh.creation.box(dims, transform=transform)
                mesh_list.append(mesh)
    elif terrain == "inverted_stairs":
        step_width = random.uniform(0.2, 0.5)
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
        if random.uniform(0, 1) > 0.05:
            center_pts = [(0, 0, 0), (cfg.size[0], 0, 0), (0, cfg.size[1], 0), (cfg.size[0], cfg.size[1], 0)]
            for i, center in enumerate(center_pts):
                if random.uniform(0, 1) > 0.75:
                    continue
                length = cfg.size[0] * random.uniform(0.3, 0.4)
                width = cfg.size[1] * random.uniform(0.3, 0.4)
                height = 6.0
                transform = np.eye(4)
                c = (center[0] + (-1) ** i * length / 2, center[1] + (-1) ** (i // 2) * width / 2, center[2])
                transform[0:3, -1] = np.asarray(c)
                # create the box
                dims = (length, width, height)
                mesh = trimesh.creation.box(dims, transform=transform)
                mesh_list.append(mesh)
    elif terrain == "walls":
        origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
        # add walls
        center_pts = [(0, 0, 0), (cfg.size[0], 0, 0), (0, cfg.size[1], 0), (cfg.size[0], cfg.size[1], 0)]
        for i, center in enumerate(center_pts):
            if random.uniform(0, 1) > 0.75:
                continue
            length = cfg.size[0] * random.uniform(0.3, 0.4)
            width = cfg.size[1] * random.uniform(0.3, 0.4)
            height = 6.0
            transform = np.eye(4)
            c = (center[0] + (-1) ** i * length / 2, center[1] + (-1) ** (i // 2) * width / 2, center[2])
            transform[0:3, -1] = np.asarray(c)
            # create the box
            dims = (length, width, height)
            mesh = trimesh.creation.box(dims, transform=transform)
            mesh_list.append(mesh)
        # add plane
        ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
        mesh_list.append(ground_plane)
    else:
        raise ValueError(f"terrain_type {terrain} is not supported")
    # update the origin in a free space
    return mesh_list, origin


def beam_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshBeamTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with beams connecting a central platform to the outer border.

    The terrain consists of a cylindrical platform at the center connected to the border region
    by multiple straight beams acting as bridges. The number, width, and height of the beams
    are defined by the configuration, and the border is generated based on `inner_size`.

    Args:
        difficulty: The difficulty of the terrain, a value between 0 and 1.
        cfg: The configuration for the beam terrain.

    Returns:
        A tuple containing the list of triangle meshes composing the terrain and the origin position (in meters).
    """
    # resolve the terrain configuration
    bar_height = cfg.bar_height_range[0] + difficulty * (cfg.bar_height_range[1] - cfg.bar_height_range[0])
    bar_width = cfg.bar_width_range[1] - difficulty * (cfg.bar_width_range[1] - cfg.bar_width_range[0])

    # initialize list of meshes
    meshes_list = list()
    # Generate a platform in the middle
    platform_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -bar_height / 2)
    platform_transform = trimesh.transformations.translation_matrix(platform_center)

    platform = trimesh.creation.cylinder(cfg.platform_width * 0.5, bar_height, sections=6, transform=platform_transform)

    meshes_list.append(platform)
    # Generate bars to connect the platform to the terrain
    transform = np.eye(4)
    platform_center_np = np.asarray(platform_center)
    yaw = 0.0
    for _ in range(cfg.num_bars):
        # compute the length of the bar based on the yaw
        bar_length = cfg.size[0] // 2

        quad_yaw = yaw % (np.pi / 2)  # Angle within current quadrant
        if quad_yaw <= np.pi / 4:
            bar_length /= np.math.cos(quad_yaw)
        else:
            bar_length /= np.math.sin(quad_yaw)
        transform[0:3, 0:3] = tf.Rotation.from_euler("z", yaw).as_matrix()
        bar_center_offset_w = transform[0:3, 0:3] @ np.array([bar_length / 2, 0, 0])
        transform[:3, -1] = platform_center_np + bar_center_offset_w

        # add the bar to the mesh
        dim = [bar_length - bar_width, bar_width, bar_height]
        bar = trimesh.creation.box(dim, transform.copy())
        meshes_list.append(bar)
        # increment the yaw
        yaw += (2 * np.pi) / cfg.num_bars

    # Generate the exterior border
    meshes_list += make_border(cfg.size, cfg.border_size, bar_height, platform_center)

    # Generate the ground
    ground = make_plane(cfg.size, -bar_height, center_zero=False)
    meshes_list.append(ground)

    # specify the origin of the terrain
    origin = np.asarray([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    return meshes_list, origin
