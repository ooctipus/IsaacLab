# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot kinematics and collider geometry for Factory task-table construction.

The kinematic model is the ROBOT ONLY (Franka, fixed-base, at the origin).
Neither the held nor fixed asset lives in the chain:

* The held asset is a canonical-frame collider mesh plus a per-problem pose --
  grasp contact pairs are sampled on it (:class:`~.samplers.GraspPairSampler`)
  and it acts as a per-problem-posed obstacle for the gripper.
* The fixed asset and the table are static world-frame meshes.

The sampled held-to-end-effector transform varies per candidate, so the held
asset is posed evidence rather than a model body.
Finger-pad contact offsets are derived from FK (the parallel jaw closes along
EE-y), not from annotations.

This fork's ``isaaclab.utils.math`` and warp/newton all use the ``(x, y, z, w)``
quaternion layout, so no quaternion permutation is applied here (viser is
``(w, x, y, z)``; that conversion happens only at the viser boundary).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

import isaaclab.utils.math as math_utils

from ...kinematics import NewtonKinematics
from ...kinematics.collider_geometry import (
    collider_mesh_load,
    mesh_edges,
    mesh_points_farthest_sample,
    model_collision_edges,
    model_collision_mesh,
    model_collision_shape_indices,
    model_shape_surface_probes,
    points_transform_xyzw,
)

if TYPE_CHECKING:
    from .cfg import FactoryGeometryCfg


def factory_default_joint_q(geometry: FactoryGeometry) -> torch.Tensor:
    """Return the shared default coordinates with a mid-open Factory gripper."""
    joint_q = torch.tensor(geometry.kinematics.default_joint_q, device=geometry.device)
    joint_q[geometry.finger_coords] = 0.02
    return joint_q


def factory_eval_fk(kinematics: NewtonKinematics, joint_q: torch.Tensor) -> torch.Tensor:
    """Evaluate a Factory batch through the shared Newton kinematics owner."""
    body_q, _ = kinematics.eval_fk_batched(wp.from_torch(joint_q, dtype=wp.float32))
    return wp.to_torch(body_q).view(joint_q.shape[0], kinematics.model.body_count, 7)


class FactoryGeometry:
    """Factory contact, obstacle, and task-frame geometry around shared kinematics.

    Attributes:
        model: The finalized Newton model (Franka only, fixed-base at the origin).
        ee_body: End-effector body index.
        gripper_bodies: Gripper body indices probed for obstacle-collision checks.
        pad_bodies: Finger body indices ordered (+jaw-y, -jaw-y), the IK target links.
        pad_offsets: Pad contact-point offsets in the finger body frames [m], shape ``[2, 3]``.
        held_verts: Held-asset collider vertices in the held body frame [m], shape ``[V, 3]``.
        held_faces: Held-asset collider faces (outward wound), shape ``[F, 3]``.
        held_mesh: Held-asset :class:`warp.Mesh` in the HELD frame (per-problem posed obstacle).
        held_probes: Held-asset surface probe offsets in the held frame [m], shape ``[P, 3]``.
        obstacle_geom: ``name -> (world vertices [V, 3], faces [F, 3])`` per static
            obstacle (the scene assets in ``obstacle_asset_names``).
        static_obstacles: ``name -> warp.Mesh`` static world-frame obstacle meshes.
        board_verts: Nistboard collider vertices in its spawned-root frame [m], ``[V, 3]``.
        board_faces: Nistboard collider faces (outward wound), ``[F, 3]``.
        board_mesh: Nistboard :class:`warp.Mesh` in its own frame (per-sub-world posed).
        board_probes: Nistboard surface probe offsets in its frame [m], ``[P, 3]``.
        board_init_pos: Nistboard canonical scene position [m], ``[3]``.
        board_init_quat: Nistboard canonical scene orientation (xyzw), ``[4]``.
        board_asset_geom: Collider vertices [m] and faces for every mapped board asset.
        board_asset_meshes: Per-asset :class:`warp.Mesh` values in their own frames.
    """

    def __init__(
        self,
        kinematics: NewtonKinematics,
        cfg: FactoryGeometryCfg,
        scene_cfg,
        probe_count: int,
        rng: np.random.Generator,
    ) -> None:
        self.cfg = cfg
        self.kinematics = kinematics
        self.model = kinematics.model
        self.builder = kinematics.builder
        self.device = kinematics.device
        if probe_count < 1:
            raise ValueError("Factory geometry probe count must be positive.")
        self._self_adjacency: dict[int, np.ndarray] = {}

        held_usd = getattr(scene_cfg, cfg.held_asset_cfg.name).spawn.usd_path

        names = kinematics.body_names
        self.ee_body = names.index(cfg.robot.ee_body_name)
        self._finger_bodies = [names.index(n) for n in cfg.robot.finger_body_names]
        self.gripper_bodies = [names.index(n) for n in cfg.robot.gripper_body_names if n in names]
        if not self.gripper_bodies:
            raise ValueError(
                f"none of gripper_body_names {cfg.robot.gripper_body_names} found; available: {[n for n in names if n]}"
            )
        self.nq = self.model.joint_coord_count
        self.ndof = self.model.joint_dof_count
        self.body_count = self.model.body_count
        self.arm_coords, self.arm_dofs, self.arm_joint_names = kinematics.find_body_chain_joint_coordinates(
            cfg.robot.ee_body_name
        )
        self.finger_coords, self.finger_dofs, self.finger_joint_names = kinematics.find_body_child_joint_coordinates(
            cfg.robot.finger_body_names
        )
        default_joint_q = torch.tensor(kinematics.default_joint_q, device=self.device)
        self.arm_stance = default_joint_q[self.arm_coords]
        self.pad_bodies, self.pad_offsets = self._derive_pad_offsets()

        # --- held asset: canonical mesh + probes (per-problem posed) ---
        self.held_verts, self.held_faces = collider_mesh_load(held_usd, self.device)
        self.held_mesh = wp.Mesh(
            points=wp.array(self.held_verts, dtype=wp.vec3, device=self.device),
            indices=wp.array(self.held_faces.reshape(-1), dtype=wp.int32, device=self.device),
        )
        tri = self.held_verts[self.held_faces]
        self.held_probes = mesh_points_farthest_sample(
            np.concatenate((self.held_verts, tri.mean(axis=1))),
            probe_count,
        ).astype(np.float32)

        # --- static obstacles: the scene assets named in cfg.obstacle_asset_names,
        # each flattened into a world-frame mesh at its scene init pose. The scene
        # cfg is the single source of truth for asset geometry/poses.
        self.obstacle_geom: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in cfg.obstacle_asset_names:
            asset_cfg = getattr(scene_cfg, name)
            spawn_scale = getattr(asset_cfg.spawn, "scale", None)
            v, f = collider_mesh_load(asset_cfg.spawn.usd_path, self.device, scale=spawn_scale)
            pos = np.asarray(asset_cfg.init_state.pos, dtype=np.float32)
            quat = np.asarray(asset_cfg.init_state.rot, dtype=np.float32)  # xyzw in this fork
            self.obstacle_geom[name] = (points_transform_xyzw(v, pos, quat), f)
        self.static_obstacles = {
            name: wp.Mesh(
                points=wp.array(v, dtype=wp.vec3, device=self.device),
                indices=wp.array(f.reshape(-1), dtype=wp.int32, device=self.device),
            )
            for name, (v, f) in self.obstacle_geom.items()
        }

        # Posed board and fixed-asset meshes share one declared relative transform.
        board_cfg = getattr(scene_cfg, cfg.board.board_asset_cfg.name)
        board_scale = getattr(board_cfg.spawn, "scale", None)
        self.board_verts, self.board_faces = collider_mesh_load(
            board_cfg.spawn.usd_path,
            self.device,
            scale=board_scale,
        )
        self.board_mesh = wp.Mesh(
            points=wp.array(self.board_verts, dtype=wp.vec3, device=self.device),
            indices=wp.array(self.board_faces.reshape(-1), dtype=wp.int32, device=self.device),
        )
        tri_b = self.board_verts[self.board_faces]
        self.board_probes = mesh_points_farthest_sample(
            np.concatenate((self.board_verts, tri_b.mean(axis=1))),
            128,
        ).astype(np.float32)
        self.board_init_pos = np.asarray(board_cfg.init_state.pos, dtype=np.float32)
        self.board_init_quat = np.asarray(board_cfg.init_state.rot, dtype=np.float32)  # xyzw
        self.board_asset_geom = {}
        for name in cfg.board.fixed_asset_map:
            asset_cfg = getattr(scene_cfg, name)
            scale = getattr(asset_cfg.spawn, "scale", None)
            vertices, faces = collider_mesh_load(asset_cfg.spawn.usd_path, self.device, scale=scale)
            self.board_asset_geom[name] = (vertices, faces)
        self.board_asset_meshes = {
            name: wp.Mesh(
                points=wp.array(vertices, dtype=wp.vec3, device=self.device),
                indices=wp.array(faces.reshape(-1), dtype=wp.int32, device=self.device),
            )
            for name, (vertices, faces) in self.board_asset_geom.items()
        }
        self.board_edge_p0, self.board_edge_p1 = mesh_edges(self.board_verts, self.board_faces)
        self.board_group_names = (cfg.board.board_asset_cfg.name, *tuple(self.board_asset_geom))
        group_probes = [self.board_probes]
        group_edge_p0 = [self.board_edge_p0]
        group_edge_p1 = [self.board_edge_p1]
        group_edge_bodies = [np.zeros(self.board_edge_p0.shape[0], dtype=np.int32)]
        for body_index, (vertices, faces) in enumerate(self.board_asset_geom.values(), start=1):
            triangles = vertices[faces]
            group_probes.append(
                mesh_points_farthest_sample(
                    np.concatenate((vertices, triangles.mean(axis=1))),
                    128,
                ).astype(np.float32)
            )
            edge_p0, edge_p1 = mesh_edges(vertices, faces)
            group_edge_p0.append(edge_p0)
            group_edge_p1.append(edge_p1)
            group_edge_bodies.append(np.full(edge_p0.shape[0], body_index, dtype=np.int32))
        self.board_group_probes = tuple(group_probes)
        self.board_group_edge_p0 = np.concatenate(group_edge_p0).astype(np.float32)
        self.board_group_edge_p1 = np.concatenate(group_edge_p1).astype(np.float32)
        self.board_group_edge_bodies = np.concatenate(group_edge_bodies)

        self._setup_gripper_probes(probe_count, rng)
        self._setup_robot_probes(probe_count, rng)
        self._setup_robot_edges()
        self._setup_robot_full_collision_geometry(probe_count, rng)

    def _derive_pad_offsets(self) -> tuple[list[int], torch.Tensor]:
        """Derive pad-contact offsets from the facing finger collider planes.

        FK supplies each nominal pad center. Each point is projected onto the
        actual inward-facing support plane of its finger collision mesh, then
        verified against the same collider plane at a second opening.

        Returns:
            ``(pad_bodies[2], pad_offsets[2, 3])`` ordered (+jaw-y finger, -jaw-y finger).
        """

        def pads_world(opening: float) -> tuple[torch.Tensor, torch.Tensor]:
            joint_q = factory_default_joint_q(self).unsqueeze(0)
            joint_q[0, self.finger_coords] = opening
            body_q = factory_eval_fk(self.kinematics, joint_q)[0]
            ee_position, ee_rotation = body_q[self.ee_body, :3], body_q[self.ee_body, 3:7]
            ee_y = math_utils.quat_apply(
                ee_rotation.unsqueeze(0), torch.tensor(((0.0, 1.0, 0.0),), device=self.device)
            )[0]
            return body_q, torch.stack((ee_position + opening * ee_y, ee_position - opening * ee_y))

        def facing_plane(body_q: torch.Tensor, body: int, opposing_body: int) -> tuple[torch.Tensor, torch.Tensor]:
            inward_world = torch.nn.functional.normalize(body_q[opposing_body, :3] - body_q[body, :3], dim=0)
            inward_body = math_utils.quat_apply_inverse(body_q[body, 3:7].unsqueeze(0), inward_world.unsqueeze(0))[0]
            vertices, _ = model_collision_mesh(self.model, model_collision_shape_indices(self.model, body))
            vertices = torch.tensor(vertices, device=self.device)
            return inward_body, (vertices @ inward_body).max()

        body_q, nominal_pads = pads_world(0.02)
        ee_position = body_q[self.ee_body, :3]
        first_projection = float(
            ((body_q[self._finger_bodies[0], :3] - ee_position) * (nominal_pads[0] - ee_position)).sum()
        )
        ordered = list(self._finger_bodies) if first_projection > 0 else list(reversed(self._finger_bodies))
        offsets = []
        for index, body in enumerate(ordered):
            offset, _ = math_utils.subtract_frame_transforms(
                body_q[body, :3].unsqueeze(0),
                body_q[body, 3:7].unsqueeze(0),
                nominal_pads[index].unsqueeze(0),
            )
            inward, support = facing_plane(body_q, body, ordered[1 - index])
            offset = offset[0] + (support - torch.dot(offset[0], inward)) * inward
            offsets.append(offset)
        offsets = torch.stack(offsets)

        body_q, _ = pads_world(0.035)
        for index, body in enumerate(ordered):
            inward, support = facing_plane(body_q, body, ordered[1 - index])
            error = float(torch.abs(torch.dot(offsets[index], inward) - support))
            if error > 1.0e-6:
                raise RuntimeError(f"pad offset differs from the finger collider plane by {error:.2e} m")
        return ordered, offsets

    def _setup_gripper_probes(self, probe_count: int, rng: np.random.Generator) -> None:
        """Surface probes + collider-mesh targets for the gripper bodies.

        The probes drive gripper-to-held queries; per-body collider meshes drive
        the symmetric held-to-gripper query needed between sparse probes.
        """
        sptr = self.model.shape_source_ptr.numpy()
        shape_tf = self.model.shape_transform.numpy()
        per = max(1, probe_count // len(self.gripper_bodies))
        offsets, bodies, t_body, t_mesh, t_tf = [], [], [], [], []
        for gb in self.gripper_bodies:
            shapes = model_collision_shape_indices(self.model, gb)
            if shapes.size == 0:
                raise RuntimeError(f"no collision mesh on gripper body {gb}")
            off = model_shape_surface_probes(self.model, int(shapes[0]), per, rng)
            offsets.append(off)
            bodies.append(np.full(len(off), gb, dtype=np.int32))
            t_body.append(gb)
            t_mesh.append(int(sptr[shapes[0]]))
            t_tf.append(shape_tf[shapes[0], :7])
        surface_offsets = np.concatenate(offsets)
        surface_bodies = np.concatenate(bodies)
        self.gripper_probes = np.concatenate((surface_offsets, self.pad_offsets.cpu().numpy())).astype(np.float32)
        self.gripper_probe_bodies = np.concatenate((surface_bodies, np.asarray(self.pad_bodies, dtype=np.int32)))
        self.gripper_target_bodies = np.array(t_body, dtype=np.int32)
        self.gripper_target_meshes = np.array(t_mesh, dtype=np.uint64)
        self.gripper_target_tf = np.array(t_tf, dtype=np.float32)

    def _setup_robot_probes(self, probe_count: int, rng: np.random.Generator) -> None:
        """Surface probes across ALL robot links except the base, for the
        robot-vs-static-obstacle criteria -- the arm/elbow must clear the table
        too, not just the gripper. The base link is excluded because it
        legitimately sits on its mount."""
        sbody = self.model.shape_body.numpy()
        shapes = [shape for shape in model_collision_shape_indices(self.model) if int(sbody[shape]) > 0]
        per = max(1, probe_count // len(shapes))
        offsets, bodies = [], []
        for si in shapes:
            off = model_shape_surface_probes(self.model, int(si), per, rng)
            offsets.append(off)
            bodies.append(np.full(len(off), int(sbody[si]), dtype=np.int32))
        self.robot_probes = np.concatenate(offsets)
        self.robot_probe_bodies = np.concatenate(bodies)

    def _setup_robot_edges(self) -> None:
        """Collider edges across robot links (base excluded) for the edge-crossing
        criterion: point probes miss thin obstacles (the ~4 mm board) slicing
        between them, while an edge-vs-mesh raycast detects any surface crossing
        regardless of probe density."""
        shape_bodies = self.model.shape_body.numpy()
        shapes = np.asarray(
            [shape for shape in model_collision_shape_indices(self.model) if int(shape_bodies[shape]) > 0],
            dtype=np.int32,
        )
        self.robot_edge_p0, self.robot_edge_p1, self.robot_edge_bodies = model_collision_edges(
            self.model,
            shapes,
        )

    def _setup_robot_full_collision_geometry(self, probe_count: int, rng: np.random.Generator) -> None:
        """Build all-link probes, edges, and shape targets once.

        Board qualification uses these complete colliders at the default robot
        pose. Self-collision reuses the probes and targets with a kinematic
        adjacency filter.
        """
        sbody = self.model.shape_body.numpy()
        sptr = self.model.shape_source_ptr.numpy()
        shape_tf = self.model.shape_transform.numpy()
        robot_shapes = model_collision_shape_indices(self.model)
        per = max(1, probe_count // len(robot_shapes))
        offsets, pbody, tbody, tmesh, ttf = [], [], [], [], []
        for si in robot_shapes:
            b = int(sbody[si])
            off = model_shape_surface_probes(self.model, int(si), per, rng)
            offsets.append(off)
            pbody.append(np.full(len(off), b, dtype=np.int32))
            tbody.append(b)
            tmesh.append(int(sptr[si]))
            ttf.append(shape_tf[si, :7])  # transformf layout: pos (3) + quat xyzw (4)
        self.robot_full_probes = np.concatenate(offsets)
        self.robot_full_probe_bodies = np.concatenate(pbody)
        self.robot_full_edge_p0, self.robot_full_edge_p1, self.robot_full_edge_bodies = model_collision_edges(
            self.model, robot_shapes
        )
        self.robot_target_bodies = np.array(tbody, dtype=np.int32)
        self.robot_target_meshes = np.array(tmesh, dtype=np.uint64)
        self.robot_target_tf = np.array(ttf, dtype=np.float32)

    def self_adjacency(self, hops: int) -> np.ndarray:
        """Return the cached body-adjacency mask for one criterion radius."""
        if hops < 0:
            raise ValueError("Factory self-collision adjacency hops cannot be negative.")
        adjacency = self._self_adjacency.get(hops)
        if adjacency is None:
            adjacency = self.kinematics.body_adjacency(hops)
            self._self_adjacency[hops] = adjacency
        return adjacency
