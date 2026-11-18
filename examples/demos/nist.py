# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a ten-part NIST assembly policy.

    uv run --extra ovrtx isaaclab demo nist

The pretrained TorchScript policy is downloaded automatically.
The exported policy includes normalization and takes robot history [N, 195] and
slot history [N, 5, 10, 40]. At startup, the seed selects ten distinct asset
identities from the original 19-asset catalog. The selection stays fixed across
all environments and episodes. Each reset randomizes the board
and drops ten separated parts across a square, with all ten parts unfinished
and the robot at home. There is no reset bank or grasp-state initialization.
The default RTX view uses a physical ground plane without a backdrop.
Use --num_envs to run independent boards in parallel, each resetting on completion or timeout.
Physics remains at 100 Hz and control at 25 Hz.
The policy was trained on MJWarp, the default; --physics feather_pgs runs the same scene on FeatherPGS.
Use --visualizer none --max_steps 1 for a headless smoke test.
"""

from __future__ import annotations

import argparse
import random
from typing import NamedTuple

import numpy as np
import torch
from isaaclab_newton.physics import FeatherPGSSolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.sim.schemas import (
    MujocoRigidBodyCfg,
    NewtonArticulationCfg,
    NewtonCollisionCfg,
    NewtonSDFCollisionCfg,
)
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg
from newton import GeoType, ShapeFlags

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import (
    EventTermCfg,
    ManagerTermBase,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg, resolve_presets


class Assembly(NamedTuple):
    """Static asset metadata: positions [m], masses [kg], xyzw poses. Local frame rotations are identity."""

    identity: int
    name: str
    plug: str
    socket: str
    board_pose: tuple[float, ...]
    held_align: tuple[float, float, float]
    fixed_tip: tuple[float, float, float]
    seated: tuple[float, float, float]
    plug_mass: float
    socket_mass: float = 0.05


# The exported actor uses this 19-entry order for each selected part's one-hot identity.
# fmt: off
ASSEMBLIES = (
    Assembly(0, "nut_thread_m8", "nut_m8", "bolt_m8",
             (0.0473, -0.0407, 0.0172, 1.0, 0.0, 0.0, 0.0),
             (0.0, 0.0, 0.0093), (0.0, 0.0, 0.026), (0.0, 0.0, 0.018), 0.03),
    Assembly(1, "nut_thread_m12", "nut_m12", "bolt_m12",
             (0.3473, -0.2665, 0.0212, 1.0, 0.0, 0.0, 0.0),
             (0.0, 0.0, 0.013), (0.0, 0.0, 0.035), (0.0, 0.0, 0.0218), 0.03),
    Assembly(2, "nut_thread_m16", "nut_m16", "bolt_m16",
             (0.04715, -0.3416, 0.0194, 1.0, 0.0, 0.0, 0.0),
             (0.0, 0.0, 0.01), (0.0, 0.0, 0.035), (0.0, 0.0, 0.022), 0.03),
    Assembly(3, "gear_mesh_small", "gear_small", "gear_base",
             (0.0474, -0.1713, -0.0002, 0.7071, -0.7071, 0.0, 0.0),
             (0.05075, 0.0, 0.005), (0.0508, 0.0, 0.025), (0.05075, 0.0, 0.005), 0.019),
    Assembly(4, "gear_mesh_medium", "gear_medium", "gear_base",
             (0.0474, -0.1713, -0.0002, 0.7071, -0.7071, 0.0, 0.0),
             (0.02025, 0.0, 0.005), (0.02025, 0.0, 0.025), (0.02025, 0.0, 0.005), 0.012),
    Assembly(5, "gear_mesh_large", "gear_large", "gear_base",
             (0.0474, -0.1713, -0.0002, 0.7071, -0.7071, 0.0, 0.0),
             (-0.0303, 0.0, 0.005), (-0.0303, 0.0, 0.025), (-0.0303, 0.0, 0.005), 0.019),
    Assembly(6, "rod_insert_4mm", "round_peg_4mm", "round_hole_4mm",
             (0.3473, -0.1918, -0.0001, 0.7071, -0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(7, "rod_insert_8mm", "round_peg_8mm", "round_hole_8mm",
             (0.3473, -0.1164, -0.0001, 0.7071, -0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(8, "rod_insert_12mm", "round_peg_12mm", "round_hole_12mm",
             (0.1226, -0.0422, -0.0001, 0.7071, -0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(9, "rod_insert_16mm", "round_peg_16mm", "round_hole_16mm",
             (0.1221, -0.2665, -0.0001, 0.7071, -0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(10, "peg_insert_4mm", "rectangular_peg_4mm", "rectangular_hole_4mm",
             (0.1971, -0.1915, -0.0003, 0.7071, -0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(11, "peg_insert_8mm", "rectangular_peg_8mm", "rectangular_hole_8mm",
             (0.2717, -0.2659, -0.0003, 0.7071, -0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(12, "peg_insert_12mm", "rectangular_peg_12mm", "rectangular_hole_12mm",
             (0.1971, -0.0413, -0.0003, 0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(13, "peg_insert_16mm", "rectangular_peg_16mm", "rectangular_hole_16mm",
             (0.3472, -0.0413, -0.0003, 0.7071, 0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(14, "usba", "usb_a_plug", "usb_a_socket",
             (0.2721, -0.0415, -0.0001, 0.7071, 0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0335), (0.0, 0.0, 0.0416), (0.0, 0.0, 0.0335), 0.05, 0.012),
    Assembly(15, "waterproof", "waterproof_plug", "waterproof_socket",
             (0.1981, -0.1166, -0.0002, 0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 0.021), (0.0, 0.0, 0.034), (0.0, 0.0, 0.021), 0.05),
    Assembly(16, "bnc", "bnc_plug", "bnc_socket",
             (0.2797, -0.1915, 0.0, 0.7071, 0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0107), (0.0, 0.0, 0.0107), (0.0, 0.0, 0.0107), 0.05),
    Assembly(17, "dsub", "dsub_plug", "dsub_socket",
             (0.2129, -0.2659, -0.019, 0.7071, -0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.0061), (0.0, 0.0, 0.0), 0.005),
    Assembly(18, "rj45", "rj45_plug", "rj45_socket",
             (0.3473, -0.3415, 0.0, 0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 0.015), (0.0, 0.0, 0.028), (0.0, 0.0, 0.015), 0.05),
)
# fmt: on
ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory/NIST"
GRASP = (0.0, 0.0, 0.107, 0.0, -1.0, 0.0, 0.0)
WORKSPACE_LOWER = (0.0, -0.675, -0.05)
WORKSPACE_UPPER = (1.0, 0.675, 1.0)
BOARD_CENTER_LOCAL = (0.19232847446867662, -0.191768517928605, 0.0)
# Simple spatial ranges; board orientation and asset identity do not change their bounds.
BOARD_CENTER_RANGE = ((0.40, 0.50), (-0.03, 0.03), (0.0206, 0.0206))
DROP_POSITION_RANGE = ((0.15, 0.85), (-0.35, 0.35), (0.12, 0.14))
CONTACT_MATERIALS = [
    UsdPhysicsRigidBodyMaterialCfg(static_friction=0.75, dynamic_friction=0.75),
    NewtonMaterialCfg(contact_stiffness=1.6e5, contact_damping=800.0),
]
# The SDF band covers the reach a pass predicts contacts over (the speculative ceiling below): past the band an SDF has
# no distance or normal to give a predicted contact.
SDF_BAND = 0.015
SOCKET_MESH_COLLISION_PROPS = NewtonSDFCollisionCfg(
    sdf_max_resolution=256,
    sdf_narrow_band_inner=-SDF_BAND,
    sdf_narrow_band_outer=SDF_BAND,
)


def grasp_pose(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Grasp-point position [m] and xyzw orientation in the robot root frame."""
    robot = env.scene[robot_cfg.name]
    hand = robot.data.body_link_pose_w.torch[:, robot_cfg.body_ids[0]]
    offset = hand.new_tensor(GRASP).expand_as(hand)
    position, rotation = math_utils.combine_frame_transforms(hand[:, :3], hand[:, 3:], offset[:, :3], offset[:, 3:])
    root = robot.data.root_link_pose_w.torch
    return torch.cat(math_utils.subtract_frame_transforms(root[:, :3], root[:, 3:], position, rotation), dim=-1)


def grasp_velocity(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Grasp-point linear [m/s] and angular [rad/s] velocity in the robot root frame."""
    robot = env.scene[robot_cfg.name]
    hand_id = robot_cfg.body_ids[0]
    hand_rotation = robot.data.body_link_quat_w.torch[:, hand_id]
    velocity = robot.data.body_link_vel_w.torch[:, hand_id]
    lever = math_utils.quat_apply(hand_rotation, velocity.new_tensor(GRASP[:3]).expand(env.num_envs, -1))
    linear = velocity[:, :3] + torch.cross(velocity[:, 3:], lever, dim=-1)
    root_rotation = robot.data.root_link_quat_w.torch
    return torch.cat(
        (
            math_utils.quat_apply_inverse(root_rotation, linear),
            math_utils.quat_apply_inverse(root_rotation, velocity[:, 3:]),
        ),
        dim=-1,
    )


def reset_parts(env: ManagerBasedEnv, env_ids: torch.Tensor | slice, assemblies: tuple[Assembly, ...]) -> None:
    """Sample the board centre and loose parts across a square workspace [m]."""
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    count = len(env_ids)
    if count == 0:
        return
    robot = env.scene["robot"]
    joints = robot.data.default_joint_pos.torch[env_ids]
    robot.write_joint_state_to_sim_index(position=joints, velocity=torch.zeros_like(joints), env_ids=env_ids)

    board = env.scene["nistboard"]
    board_pose = board.data.default_root_pose.torch[env_ids].clone()
    yaw = torch.rand(count, device=env.device) * 6.28 - 3.14
    board_pose[:, 3:] = math_utils.quat_mul(
        board_pose[:, 3:], math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    )
    bounds = board_pose.new_tensor(BOARD_CENTER_RANGE)
    center = bounds[:, 0] + torch.rand((count, 3), device=env.device) * (bounds[:, 1] - bounds[:, 0])
    board_pose[:, :3] = center - math_utils.quat_apply(
        board_pose[:, 3:], board_pose.new_tensor(BOARD_CENTER_LOCAL).expand(count, -1)
    )
    board_pose[:, :3] += env.scene.env_origins[env_ids]
    board.write_root_link_pose_to_sim_index(root_pose=board_pose, env_ids=env_ids)
    origins = env.scene.env_origins[env_ids]
    fixture_poses = [torch.cat((board_pose[:, :3] - origins, board_pose[:, 3:]), dim=1)]
    # Compose from raw xyzw poses, preserving the quaternion convention seen during training.
    for socket, part in {part.socket: part for part in assemblies}.items():
        offset = board_pose.new_tensor(part.board_pose)
        position, rotation = math_utils.combine_frame_transforms(
            board_pose[:, :3], board_pose[:, 3:], offset[:3].expand(count, -1), offset[3:].expand(count, -1)
        )
        env.scene[f"fixed_{socket}"].write_root_link_pose_to_sim_index(
            root_pose=torch.cat((position, rotation), dim=1), env_ids=env_ids
        )
        fixture_poses.append(torch.cat((position - origins, rotation), dim=1))

    bounds = board_pose.new_tensor(DROP_POSITION_RANGE)
    positions, rotations = draw_part_poses(count, bounds)
    fixtures = Fixtures(torch.stack(fixture_poses, dim=1), fixture_meshes(env, assemblies))
    separate_overlapping_parts(positions, rotations, bounds, part_meshes(env), spawn_generator(env), fixtures)
    for slot in range(10):
        asset = env.scene[f"held_{slot:02d}"]
        pose = asset.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] = positions[:, slot] + env.scene.env_origins[env_ids]
        pose[:, 3:7] = rotations[:, slot]
        asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        asset.write_root_com_velocity_to_sim_index(root_velocity=pose.new_zeros((len(env_ids), 6)), env_ids=env_ids)


# Loose parts' origins are drawn at least this far apart [m]; their collision geometry reaches up to 93 mm from them.
PART_SPACING = 0.1
# Each part's roll, pitch and yaw are drawn within these magnitudes [rad].
PART_TILT = (1.57, 1.57, 3.14)


class PartMesh(NamedTuple):
    """A loose part's collision geometry in its root frame [m]: its bounding box's centre and half extents, its
    vertices, and its triangles and edges as vertex indices."""

    centre: torch.Tensor
    half: torch.Tensor
    vertices: torch.Tensor
    triangles: torch.Tensor
    edges: torch.Tensor


def part_mesh(vertices: torch.Tensor, triangles: torch.Tensor) -> PartMesh:
    """A closed triangle mesh as a :class:`PartMesh`."""
    edges = (
        torch.cat((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])).sort(dim=1).values.unique(dim=0)
    )
    lower, upper = vertices.min(dim=0).values, vertices.max(dim=0).values
    return PartMesh((lower + upper) * 0.5, (upper - lower) * 0.5, vertices, triangles, edges)


def body_mesh(model, body: int, device: str) -> PartMesh:
    """A body's colliding shapes as one closed triangle mesh in its frame: mesh shapes as they are, boxes by their
    corners."""
    shape_body = model.shape_body.numpy()
    flags = model.shape_flags.numpy()
    types = model.shape_type.numpy()
    scale = model.shape_scale.numpy()
    transform = model.shape_transform.numpy()
    vertices, triangles, start = [], [], 0
    for shape in np.flatnonzero(shape_body == body):
        if not flags[shape] & ShapeFlags.COLLIDE_SHAPES:
            continue
        source = model.shape_source[int(shape)]
        if types[shape] == GeoType.BOX:
            local, faces = _box(scale[shape][:3])
        elif source is not None:
            local, faces = np.asarray(source.vertices) * scale[shape][:3], np.asarray(source.indices).reshape(-1, 3)
        else:
            continue
        rotation = math_utils.matrix_from_quat(torch.tensor(transform[shape, 3:], device=device))
        local = torch.tensor(local, dtype=torch.float32, device=device)
        vertices.append(local @ rotation.T + torch.tensor(transform[shape, :3], device=device))
        triangles.append(torch.tensor(faces, device=device) + start)
        start += len(local)
    return part_mesh(torch.cat(vertices), torch.cat(triangles))


def _box(half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A box's corners and outward triangles, half extents ``half``."""
    corners = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]) * half
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return corners, np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])


def _env0_body(model, name: str) -> int:
    """The index of the first world's body under the prim ``name`` of the environment (the first path component)."""
    return next(
        i for i, label in enumerate(model.body_label) if str(label).partition("/env_0/")[2].split("/")[0] == name
    )


def part_meshes(env: ManagerBasedEnv) -> tuple[PartMesh, ...]:
    """The ten loose parts' collision meshes, read once from the Newton model's first world."""
    meshes = getattr(env, "_nist_part_meshes", None)
    if meshes is None:
        model = env.sim.backend_manager.backend.model
        meshes = tuple(body_mesh(model, _env0_body(model, f"held_{slot:02d}"), env.device) for slot in range(10))
        env._nist_part_meshes = meshes
    return meshes


def fixture_meshes(env: ManagerBasedEnv, assemblies: tuple[Assembly, ...]) -> tuple[PartMesh, ...]:
    """The board's and the sockets' collision meshes in :func:`reset_parts`' order, read once from the first world."""
    meshes = getattr(env, "_nist_fixture_meshes", None)
    if meshes is None:
        model = env.sim.backend_manager.backend.model
        names = ["NistBoard"] + [f"fixed_{socket}" for socket in {part.socket: part for part in assemblies}]
        meshes = tuple(body_mesh(model, _env0_body(model, name), env.device) for name in names)
        env._nist_fixture_meshes = meshes
    return meshes


class Fixtures(NamedTuple):
    """The fixed geometry a loose part must spawn clear of: each world's poses [m, xyzw] in its frame, shaped
    (worlds, fixtures, 7), and each fixture's collision mesh."""

    poses: torch.Tensor
    meshes: tuple[PartMesh, ...]


def spawn_generator(env: ManagerBasedEnv) -> torch.Generator:
    """The generator that redraws overlapping parts, seeded once from the environment's seed: its draws leave the
    shared stream, and with it every other world's spawn, as they would be without them."""
    generator = getattr(env, "_nist_spawn_generator", None)
    if generator is None:
        generator = torch.Generator(device=env.device)
        generator.manual_seed(env.cfg.seed)
        env._nist_spawn_generator = generator
    return generator


def draw_part_poses(count: int, bounds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``count`` worlds' ten loose parts' positions [m] and xyzw orientations from the shared stream: origins
    ``PART_SPACING`` apart in the plane inside ``bounds``, orientations within ``PART_TILT``."""
    positions = bounds.new_empty((count, 10, 3))
    rotations = bounds.new_empty((count, 10, 4))
    for slot in range(10):
        pending = torch.arange(count, device=bounds.device)
        for _ in range(512):
            if len(pending) == 0:
                break
            candidate = bounds[:, 0] + torch.rand((len(pending), 3), device=bounds.device) * (
                bounds[:, 1] - bounds[:, 0]
            )
            clear = (
                (candidate[:, None, :2] - positions[pending, :slot, :2]).square().sum(dim=-1) >= PART_SPACING**2
            ).all(dim=1)
            positions[pending[clear], slot] = candidate[clear]
            pending = pending[~clear]
        if len(pending):
            raise RuntimeError("Could not sample separated loose parts inside the drop region.")
        angles = (torch.rand((count, 3), device=bounds.device) * 2.0 - 1.0) * bounds.new_tensor(PART_TILT)
        rotations[:, slot] = math_utils.quat_from_euler_xyz(*angles.unbind(-1))
    return positions, rotations


def separate_overlapping_parts(
    positions: torch.Tensor,
    rotations: torch.Tensor,
    bounds: torch.Tensor,
    meshes: tuple[PartMesh, ...],
    generator: torch.Generator,
    fixtures: Fixtures,
) -> None:
    """Redraw, in place, the later part of each pair whose collision meshes overlap, and a part whose mesh overlaps a
    fixture's, from ``generator``, until none does; worlds without an overlap keep their poses."""
    later = overlapping_part(positions, rotations, meshes, fixtures)
    for _ in range(256):
        worlds = torch.nonzero(later >= 0).flatten()
        if len(worlds) == 0:
            return
        slot = later[worlds]
        draws = torch.rand((len(worlds), 6), device=positions.device, generator=generator)
        candidate = bounds[:, 0] + draws[:, :3] * (bounds[:, 1] - bounds[:, 0])
        rotation = math_utils.quat_from_euler_xyz(
            *((draws[:, 3:] * 2.0 - 1.0) * bounds.new_tensor(PART_TILT)).unbind(-1)
        )
        gaps = (positions[worlds, :, :2] - candidate[:, None, :2]).square().sum(dim=-1)
        gaps[torch.arange(len(worlds), device=positions.device), slot] = PART_SPACING**2
        clear = (gaps >= PART_SPACING**2).all(dim=1)
        positions[worlds[clear], slot[clear]] = candidate[clear]
        rotations[worlds[clear], slot[clear]] = rotation[clear]
        later[worlds] = overlapping_part(
            positions[worlds], rotations[worlds], meshes, Fixtures(fixtures.poses[worlds], fixtures.meshes)
        )
    raise RuntimeError("Could not separate the loose parts' collision geometry inside the drop region.")


def overlapping_part(
    positions: torch.Tensor, rotations: torch.Tensor, meshes: tuple[PartMesh, ...], fixtures: Fixtures
) -> torch.Tensor:
    """Each world's part to redraw, -1 where none overlaps: the later part of its first pair of parts whose collision
    meshes overlap, else its first part whose mesh overlaps a fixture's.

    Every pair's bounding boxes are tested at once (separating axes); those that overlap are decided exactly by their
    meshes (:func:`meshes_overlap`), in pair order: by the later part, then the earlier; then the parts against the
    fixtures, by part, then fixture.
    """
    parts = len(meshes)
    pairs = [(a, b) for b in range(1, parts) for a in range(b)]
    pairs += [(parts + f, a) for a in range(parts) for f in range(len(fixtures.meshes))]
    every = meshes + fixtures.meshes
    first = torch.tensor([a for a, _ in pairs], device=positions.device)
    second = torch.tensor([b for _, b in pairs], device=positions.device)
    origins = torch.cat((positions, fixtures.poses[..., :3]), dim=1)
    frames = math_utils.matrix_from_quat(torch.cat((rotations, fixtures.poses[..., 3:]), dim=1))
    centres = torch.stack([mesh.centre for mesh in every])
    halves = torch.stack([mesh.half for mesh in every])
    world_centres = origins + (frames @ centres[None, :, :, None])[..., 0]
    boxes = boxes_overlap(
        world_centres[:, first],
        frames[:, first],
        halves[first],
        world_centres[:, second],
        frames[:, second],
        halves[second],
    )
    later = [-1] * positions.shape[0]
    for world, pair in torch.nonzero(boxes).tolist():
        a, b = pairs[pair]
        if later[world] >= 0:
            continue
        rotation = frames[world, a].T @ frames[world, b]
        translation = frames[world, a].T @ (origins[world, b] - origins[world, a])
        if meshes_overlap(every[a], every[b], rotation, translation):
            later[world] = b
    return torch.tensor(later, dtype=torch.long, device=positions.device)


def boxes_overlap(
    centre_a: torch.Tensor,
    axes_a: torch.Tensor,
    half_a: torch.Tensor,
    centre_b: torch.Tensor,
    axes_b: torch.Tensor,
    half_b: torch.Tensor,
) -> torch.Tensor:
    """Whether oriented boxes overlap, by the fifteen separating axes: centres [..., 3], axes [..., 3, 3] (columns),
    half extents [..., 3]."""
    r = axes_a.transpose(-1, -2) @ axes_b
    reach = r.abs() + 1.0e-6
    t = ((centre_b - centre_a)[..., None, :] @ axes_a)[..., 0, :]
    separated = torch.zeros(t.shape[:-1], dtype=torch.bool, device=t.device)
    for i in range(3):
        separated |= t[..., i].abs() > half_a[..., i] + (half_b * reach[..., i, :]).sum(-1)
        separated |= (t * r[..., :, i]).sum(-1).abs() > (half_a * reach[..., :, i]).sum(-1) + half_b[..., i]
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        for j in range(3):
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            ra = half_a[..., i1] * reach[..., i2, j] + half_a[..., i2] * reach[..., i1, j]
            rb = half_b[..., j1] * reach[..., i, j2] + half_b[..., j2] * reach[..., i, j1]
            separated |= (t[..., i2] * r[..., i1, j] - t[..., i1] * r[..., i2, j]).abs() > ra + rb
    return ~separated


def meshes_overlap(a: PartMesh, b: PartMesh, rotation: torch.Tensor, translation: torch.Tensor) -> bool:
    """Whether two closed meshes' solids intersect, ``b`` posed in ``a``'s frame: an edge of either crosses a triangle
    of the other, or, failing that, one holds the other whole (a vertex inside, by winding number).

    Every crossing lies in both meshes' bounding boxes, so only the edges and triangles that reach the boxes' common
    part are tested.
    """
    vertices_b = b.vertices @ rotation.T + translation
    lower = torch.maximum(a.centre - a.half, vertices_b.min(dim=0).values)
    upper = torch.minimum(a.centre + a.half, vertices_b.max(dim=0).values)
    if bool((lower > upper).any()):
        return False

    def near(vertices, items):
        corners = vertices[items]
        return items[((corners.min(dim=1).values <= upper) & (corners.max(dim=1).values >= lower)).all(dim=1)]

    edges_a, triangles_a = near(a.vertices, a.edges), near(a.vertices, a.triangles)
    edges_b, triangles_b = near(vertices_b, b.edges), near(vertices_b, b.triangles)
    return (
        _edges_cross(a.vertices, edges_a, vertices_b, triangles_b)
        or _edges_cross(vertices_b, edges_b, a.vertices, triangles_a)
        or _holds(a.vertices, a.triangles, vertices_b[:1])
        or _holds(vertices_b, b.triangles, a.vertices[:1])
    )


def _edges_cross(vertices: torch.Tensor, edges: torch.Tensor, others: torch.Tensor, triangles: torch.Tensor) -> bool:
    """Whether a segment ``edges`` of ``vertices`` crosses a triangle of ``others`` (Moller-Trumbore)."""
    if len(edges) == 0 or len(triangles) == 0:
        return False
    corner = others[triangles[:, 0]]
    e1 = others[triangles[:, 1]] - corner
    e2 = others[triangles[:, 2]] - corner
    chunk = max(1, (1 << 22) // len(triangles))
    for k in range(0, len(edges), chunk):
        start = vertices[edges[k : k + chunk, 0]][:, None]
        direction = vertices[edges[k : k + chunk, 1]][:, None] - start
        h = torch.cross(direction.expand(-1, len(triangles), -1), e2[None].expand(direction.shape[0], -1, -1), dim=-1)
        det = (e1[None] * h).sum(-1)
        inverse = 1.0 / torch.where(det != 0.0, det, torch.ones_like(det))
        s = start - corner[None]
        u = (s * h).sum(-1) * inverse
        q = torch.cross(s, e1[None].expand_as(s), dim=-1)
        v = (direction * q).sum(-1) * inverse
        t = (e2[None] * q).sum(-1) * inverse
        if bool(((det != 0.0) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)).any()):
            return True
    return False


def _holds(vertices: torch.Tensor, triangles: torch.Tensor, points: torch.Tensor) -> bool:
    """Whether the closed mesh holds any of ``points`` (generalized winding number above one half)."""
    a, b, c = (vertices[triangles[:, k]][None] - points[:, None] for k in range(3))
    la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
    solid = 2.0 * torch.atan2(
        (a * torch.cross(b, c, dim=-1)).sum(-1),
        la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb,
    )
    return bool((solid.sum(-1).abs() > 2.0 * torch.pi).any())


class SlotGeometry(ManagerTermBase):
    """The exported actor's three relative poses and full-catalog one-hot per object."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        assemblies = cfg.params["assemblies"]
        self.held = tuple(env.scene[f"held_{slot:02d}"] for slot in range(10))
        self.fixed = tuple(env.scene[f"fixed_{part.socket}"] for part in assemblies)
        self.robot = env.scene["robot"]
        self.hand_id = self.robot.find_bodies("panda_hand")[0][0]
        self.held_align = torch.tensor([(*part.held_align, 0, 0, 0, 1) for part in assemblies], device=self.device)
        self.fixed_tip = torch.tensor([(*part.fixed_tip, 0, 0, 0, 1) for part in assemblies], device=self.device)
        self.seated = torch.tensor([(*part.seated, 0, 0, 0, 1) for part in assemblies], device=self.device)
        self.identity = torch.eye(len(ASSEMBLIES), device=self.device)[[part.identity for part in assemblies]]

    def __call__(self, env: ManagerBasedEnv, assemblies: tuple[Assembly, ...]) -> torch.Tensor:
        held = torch.stack([asset.data.root_link_pose_w.torch for asset in self.held], dim=1)
        fixed = torch.stack([asset.data.root_link_pose_w.torch for asset in self.fixed], dim=1)
        hand = self.robot.data.body_link_pose_w.torch[:, self.hand_id]
        grasp = hand.new_tensor(GRASP).expand_as(hand)
        ee_pos, ee_quat = math_utils.combine_frame_transforms(hand[:, :3], hand[:, 3:], grasp[:, :3], grasp[:, 3:])
        ee_pos, ee_quat = ee_pos[:, None].expand(-1, 10, -1), ee_quat[:, None].expand(-1, 10, -1)
        align, tip = self.held_align.expand_as(held), self.fixed_tip.expand_as(fixed)
        held_pos, held_quat = math_utils.combine_frame_transforms(
            held[..., :3], held[..., 3:], align[..., :3], align[..., 3:]
        )
        fixed_pos, fixed_quat = math_utils.combine_frame_transforms(
            fixed[..., :3], fixed[..., 3:], tip[..., :3], tip[..., 3:]
        )
        poses = (
            *math_utils.subtract_frame_transforms(fixed_pos, fixed_quat, held_pos, held_quat),
            *math_utils.subtract_frame_transforms(ee_pos, ee_quat, held_pos, held_quat),
            *math_utils.subtract_frame_transforms(ee_pos, ee_quat, fixed_pos, fixed_quat),
        )
        seated = self.seated.expand_as(fixed)
        goal_pos, goal_quat = math_utils.combine_frame_transforms(
            fixed[..., :3], fixed[..., 3:], seated[..., :3], seated[..., 3:]
        )
        error_pos, error_quat = math_utils.subtract_frame_transforms(goal_pos, goal_quat, held_pos, held_quat)
        x, y, z, w = error_quat.unbind(-1)
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
        pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
        aligned = roll.abs() + pitch.abs() < 0.025
        self.assembled = aligned & (error_pos[..., :2].norm(dim=-1) < 0.0025) & (error_pos[..., 2] < 0.001)
        position = held[..., :3] - env.scene.env_origins[:, None]
        lower, upper = position.new_tensor(WORKSPACE_LOWER), position.new_tensor(WORKSPACE_UPPER)
        self.invalid = (
            (~torch.isfinite(held)).any(dim=(1, 2))
            | ~torch.isfinite(hand).all(dim=1)
            | ((position < lower) | (position > upper)).any(dim=(1, 2))
        )
        return torch.cat((*poses, self.identity.expand(env.num_envs, -1, -1)), dim=-1)


def assembly_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Report assembled parts and terminate complete or invalid boards."""
    term = env.observation_manager.cfg["slots"].geometry
    geometry = term.func
    geometry(env, **term.params)
    assembled = geometry.assembled.sum(dim=1)
    robot = env.scene["robot"]
    failed = (
        geometry.invalid
        | ~torch.isfinite(robot.data.joint_pos.torch).all(dim=1)
        | ~torch.isfinite(robot.data.joint_vel.torch).all(dim=1)
    )
    success = (assembled == 10) & ~failed
    env.extras["assembly_count"] = assembled
    env.extras["assembly_success"] = success
    env.extras["assembly_steps"] = env.episode_length_buf.clone()
    return success | failed


@configclass
class RobotObservations(ObservationGroupCfg):
    """Term-major robot history expected by the exported actor."""

    enable_corruption = False
    history_length = 5
    end_effector_pose_b = ObservationTermCfg(
        func=grasp_pose, params={"robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    end_effector_vel_lin_ang_b = ObservationTermCfg(
        func=grasp_velocity, params={"robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    joint_pos = ObservationTermCfg(func=mdp.joint_pos)
    joint_vel = ObservationTermCfg(func=mdp.joint_vel)
    prev_action = ObservationTermCfg(func=mdp.last_action)


@configclass
class SlotObservations(ObservationGroupCfg):
    """Unflattened [history, object, feature] input to the slot encoder."""

    enable_corruption = False
    history_length = 5
    flatten_history_dim = False
    geometry = ObservationTermCfg(func=SlotGeometry)


@configclass
class Actions:
    """Seven relative arm targets and one binary gripper command."""

    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale={"(?!panda_joint7).*": 0.02, "panda_joint7": 0.2},
        use_zero_offset=True,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class NistSceneCfg(InteractiveSceneCfg):
    """Ground, lighting, robot, and board for the NIST assembly demo."""

    num_envs: int = 1
    env_spacing: float = 2.0
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.65, 0.65, 0.65),
            physics_material=[
                UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                NewtonMaterialCfg(contact_stiffness=2500.0, contact_damping=100.0),
            ],
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0),
    )
    nistboard = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/NistBoard",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/Taskboard/nistboard.usd",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            physics_material=CONTACT_MATERIALS,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.452824, 0.19145, 0.0206), rot=(0.0, 1.0, 0.0, 0.0)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        joint_ordering=[f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"],
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/franka_panda.usda",
            rigid_props=MujocoRigidBodyCfg(gravcomp=1.0),
            articulation_props=NewtonArticulationCfg(self_collision_enabled=False),
            collision_props=PhysxCollisionCfg(contact_offset=0.005, rest_offset=0.0),
            mesh_collision_props=sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull"),
            physics_material=[
                UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                NewtonMaterialCfg(contact_stiffness=1.0e6, contact_damping=2000.0),
            ],
            joint_drive_props=sim_utils.UsdPhysicsDriveCfg(),
            ensure_drives_exist=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            }
        ),
        actuators={
            "panda_arm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-7]"],
                joint_effort_limit={"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0},
                actuator_velocity_limit={"panda_joint[1-4]": 2.175, "panda_joint[5-7]": 2.61},
                joint_velocity_limit={"panda_joint[1-4]": 20.0, "panda_joint[5-7]": 25.0},
                stiffness={
                    "panda_joint[1-4]": 600.0,
                    "panda_joint5": 250.0,
                    "panda_joint6": 150.0,
                    "panda_joint7": 50.0,
                },
                damping={"panda_joint[1-4]": 50.0, "panda_joint5": 30.0, "panda_joint6": 25.0, "panda_joint7": 15.0},
                armature={"panda_joint[1-2]": 0.6057, "panda_joint[3-4]": 0.4625, "panda_joint[5-7]": 0.2055},
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint1"],
                joint_effort_limit=70.0,
                actuator_velocity_limit=0.2,
                joint_velocity_limit=2.0,
                stiffness=350.0,
                damping=175.0,
                armature=0.1,
            ),
            "panda_finger2_passive": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint2"],
                joint_effort_limit=1.0,
                actuator_velocity_limit=0.2,
                joint_velocity_limit=2.0,
                stiffness=0.0,
                damping=0.0,
                armature=0.1,
            ),
        },
    )


@configclass
class NistPhysicsCfg(PresetCfg):
    """Newton solvers of the demo: MJWarp, which the policy was trained on, and FeatherPGS."""

    default = NewtonCfg(
        num_substeps=8,
        use_cuda_graph=True,
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=3200,
            nconmax=2400,
            impratio=1.0,
            cone="pyramidal",
            update_data_interval=2,
            ls_parallel=False,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="sap", max_triangle_pairs=1_000_000, rigid_contact_max=2400
        ),
    )
    # FeatherPGS grows its contact rows from demand; its triangle pairs are sized with the scene below. Its contacts
    # are rigid: the spin damping stands in for the rolling loss of MJWarp's soft ground, without which round parts
    # roll out of the workspace.
    feather_pgs = NewtonCfg(
        num_substeps=8,
        use_cuda_graph=True,
        solver_cfg=FeatherPGSSolverCfg(
            angular_damping=1.0,
            enable_joint_limits=True,
            joint_limit_activation_gap=0.1,
            dense_max_constraints=384,
            mf_max_constraints=4096,
        ),
        # Every finite collider through Newton's SDF narrow phase, at the parts' and sockets' own SDF settings: a
        # part against the board is then one SDF pair, not one GJK/MPR query a triangle.
        # A pass predicts its contacts over the window FeatherPGS holds them (the backend's collision horizon), each
        # pair as far as its closing speed carries it, up to the SDF band: a part dropped from 12-14 cm lands at about
        # 1.6 m/s, 16 mm a 10 ms window. Under a lower ceiling a surface past it makes no contact at the pass before
        # landing, and the part lands millimetres deep in its fixture (5 mm: 46,556 contacts born deeper than 1 mm in
        # the drops at 8,192 worlds; 15 mm: 6,350; 25 mm: 6,300); a ceiling past the band predicts contacts the SDF
        # cannot place (a 15 mm ceiling on a 5 mm band: fixed-geometry contacts to 13.6 mm deep in a 128-world run).
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="sap",
            sdf_all_shapes=NewtonCollisionPipelineCfg.SDFAllShapesCfg(
                sdf_max_resolution=256, sdf_narrow_band_inner=-SDF_BAND, sdf_narrow_band_outer=SDF_BAND
            ),
            speculative_contact_gap_max=SDF_BAND,
        ),
    )


@configclass
class NistEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based configuration for the ten-part NIST demo."""

    scene: NistSceneCfg = NistSceneCfg()
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.01, render_interval=2, physics=NistPhysicsCfg())
    actions: Actions = Actions()
    observations = {"policy": RobotObservations(), "slots": SlotObservations()}
    events = {
        "reset_parts": EventTermCfg(func=reset_parts, mode="reset", params={}),
        "robot_material": EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "static_friction_range": (0.75, 0.75),
                "dynamic_friction_range": (0.75, 0.75),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        ),
    }

    rewards = {}
    terminations = {
        "assembly": TerminationTermCfg(func=assembly_terminated),
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }
    episode_length_s = 140.0
    decimation = 4
    seed: int = 1701
    assemblies: tuple[Assembly, ...] = ()

    def __post_init__(self) -> None:
        """Add the seed-selected parts and their manager terms."""
        scene = self.scene
        assemblies = tuple(random.Random(self.seed).sample(ASSEMBLIES, 10))
        self.assemblies = assemblies
        self.observations["slots"].geometry.params["assemblies"] = assemblies
        self.events["reset_parts"].params["assemblies"] = assemblies
        mjwarp_collision = self.sim.physics.default.collision_cfg
        mjwarp_collision.max_triangle_pairs = max(1_000_000, scene.num_envs * 50_000)
        mjwarp_collision.rigid_contact_max = scene.num_envs * 2400
        # Measured under FeatherPGS: up to 9,900 triangle pairs and 10,900 reduction candidates a world, which the
        # pipeline's own estimate (9,600 a world) does not hold.
        self.sim.physics.feather_pgs.collision_cfg.max_triangle_pairs = max(1_000_000, scene.num_envs * 16_384)
        board = scene.nistboard.init_state
        board_pos, board_quat = torch.tensor(board.pos), torch.tensor(board.rot)
        for socket, part in {part.socket: part for part in assemblies}.items():
            offset = torch.tensor(part.board_pose)
            pos, quat = math_utils.combine_frame_transforms(board_pos, board_quat, offset[:3], offset[3:])
            setattr(
                scene,
                f"fixed_{socket}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/fixed_{socket}",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=f"{ASSET_DIR}/{socket}.usd",
                        collision_props=[PhysxCollisionCfg(rest_offset=0.0), NewtonCollisionCfg(contact_gap=0.005)],
                        mesh_collision_props=SOCKET_MESH_COLLISION_PROPS,
                        physics_material=CONTACT_MATERIALS,
                        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                        mass_props=sim_utils.MassCfg(mass=part.socket_mass),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos.tolist()), rot=tuple(quat.tolist())),
                ),
            )
        for slot, part in enumerate(assemblies):
            name = f"held_{slot:02d}"
            setattr(
                scene,
                name,
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{name}",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=f"{ASSET_DIR}/{part.plug}.usd",
                        collision_props=[
                            PhysxCollisionCfg(rest_offset=0.0, contact_offset=0.0025),
                            NewtonCollisionCfg(contact_gap=0.005),
                        ],
                        mesh_collision_props=SOCKET_MESH_COLLISION_PROPS,
                        physics_material=CONTACT_MATERIALS,
                        mass_props=sim_utils.MassCfg(mass=part.plug_mass),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0 + slot * 0.1)),
                ),
            )
            self.events[f"inertia_{slot}"] = EventTermCfg(
                func=mdp.randomize_rigid_body_inertia,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(name),
                    "inertia_distribution_params": (1.0e-5, 1.0e-5),
                    "operation": "add",
                    "diagonal_only": True,
                },
            )


def main() -> None:
    """Load the pretrained TorchScript actor and run complete ten-part assembly episodes."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_envs", type=int, default=1, help="Independent boards simulated in parallel.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
    parser.add_argument(
        "--physics", default="newton_mjwarp", choices=["newton_mjwarp", "feather_pgs"], help="Newton solver."
    )
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton_rtx"])
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("num_envs must be positive.")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("max_steps must be positive or -1.")
    policy = torch.jit.load(
        retrieve_file_path(f"{ISAACLAB_NUCLEUS_DIR}/PretrainedCheckpoints/rsl_rl/demo-alphaNIST.pt"),
        map_location=args.device,
    ).eval()
    cfg = NistEnvCfg(scene=NistSceneCfg(num_envs=args.num_envs))
    cfg.sim.physics = resolve_presets(cfg.sim.physics, (args.physics,))
    cfg.sim.device = args.device
    print("Assets:", ", ".join(part.name for part in cfg.assemblies), flush=True)
    # The solver is selected above; forwarding --physics would replace it with a backend's defaults.
    with launch_simulation(cfg=cfg, launcher_args=vars(args) | {"physics": None}) as physics:
        cfg.sim.physics = physics
        env = ManagerBasedRLEnv(cfg)
        try:
            center = env.scene.env_origins.mean(dim=0).cpu()
            target = center + torch.tensor((0.35, 0.0, 0.4))
            span = env.scene.env_origins.amax(dim=0) - env.scene.env_origins.amin(dim=0)
            camera_scale = 1.0 + 1.25 * float(span[:2].max()) / cfg.scene.env_spacing
            eye = target + camera_scale * torch.tensor((1.2, -0.65, 0.8))
            env.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
            obs, _ = env.reset()
            completed = successes = 0
            if obs["policy"].shape != (args.num_envs, 195) or obs["slots"].shape != (args.num_envs, 5, 10, 40):
                raise ValueError(f"Policy input mismatch: {obs['policy'].shape}, {obs['slots'].shape}")
            with torch.inference_mode():
                step = 0
                while env.sim.is_running() and (args.max_steps < 0 or step < args.max_steps):
                    actions = policy(obs["policy"], obs["slots"])
                    if not torch.isfinite(actions).all():
                        raise RuntimeError("Exported actor returned nonfinite actions.")
                    obs, _, terminated, timed_out, extras = env.step(actions)
                    step += 1
                    for env_id in torch.nonzero(terminated | timed_out).flatten().tolist():
                        completed += 1
                        successes += int(extras["assembly_success"][env_id])
                        count = int(extras["assembly_count"][env_id])
                        duration = int(extras["assembly_steps"][env_id]) * env.step_dt
                        print(
                            f"Episode {completed} (env {env_id}): {count}/10 assembled, {duration:.1f}s; "
                            f"complete-board successes {successes}/{completed}",
                            flush=True,
                        )
        finally:
            env.close()


if __name__ == "__main__":
    main()
