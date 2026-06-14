# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton model + geometry for the offline factory IK pipeline.

The kinematic model is the ROBOT ONLY (Franka, fixed-base, at the origin).
Neither the held asset (nut) nor the fixed asset (bolt) lives in the chain:

* The held asset is a canonical-frame collider mesh plus a per-problem pose --
  grasp contact pairs are sampled on it (:class:`~.samplers.GraspPairSampler`)
  and it acts as a per-problem-posed obstacle for the gripper.
* The fixed asset and the table are static world-frame meshes.

This replaces the earlier formulation that welded the nut to the end-effector
at a hand-annotated grasp keypoint; with sampled grasps the held-to-EE
transform varies per candidate, so the held asset cannot be a model body.
Finger-pad contact offsets are derived from FK (the parallel jaw closes along
EE-y), not from annotations.

This fork's ``isaaclab.utils.math`` and warp/newton all use the ``(x, y, z, w)``
quaternion layout, so no quaternion permutation is applied here (viser is
``(w, x, y, z)``; that conversion happens only at the viser boundary).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import newton
import numpy as np
import torch
import warp as wp
from newton import GeoType
from newton._src.sim.ik.ik_common import eval_fk_batched

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from .cfg import FactoryIKPipelineCfg


def _quat_xyzw_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) ``v`` by quaternion ``q`` (xyzw)."""
    x, y, z, w = q
    t = 2.0 * np.cross(np.broadcast_to([x, y, z], v.shape), v)
    return v + w * t + np.cross(np.broadcast_to([x, y, z], v.shape), t)


def _fps(verts: np.ndarray, n: int) -> np.ndarray:
    """Farthest-point subset of ``verts`` (``n`` points)."""
    n = min(n, len(verts))
    sel = [0]
    d = np.full(len(verts), np.inf)
    for _ in range(n - 1):
        d = np.minimum(d, np.linalg.norm(verts - verts[sel[-1]], axis=1))
        sel.append(int(np.argmax(d)))
    return verts[sel]


# unit box (half-extent 1) triangulation, outward wound
_BOX_VERTS = np.array(
    [[sx, sy, sz] for sz in (-1.0, 1.0) for sy in (-1.0, 1.0) for sx in (-1.0, 1.0)], dtype=np.float32
)
_BOX_FACES = np.array(
    [
        [0, 2, 3],
        [0, 3, 1],  # bottom (-z)
        [4, 5, 7],
        [4, 7, 6],  # top (+z)
        [0, 1, 5],
        [0, 5, 4],  # front (-y)
        [2, 6, 7],
        [2, 7, 3],  # back (+y)
        [0, 4, 6],
        [0, 6, 2],  # left (-x)
        [1, 3, 7],
        [1, 7, 5],  # right (+x)
    ],
    dtype=np.int32,
)


def _mesh_edges(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique edge endpoint pairs of a triangle mesh, ``([E, 3], [E, 3])``."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return verts[e[:, 0]].astype(np.float32), verts[e[:, 1]].astype(np.float32)


def load_collider_mesh(
    usd_path: str, device: str, scale: tuple | None = None, visual: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Load a USD's collision (or visual) shapes as one mesh in the asset root frame.

    By default gathers COLLISION shapes: mesh sources as-is, BOX primitives (the
    collision approximation on the table/board assets) triangulated from their
    half-extents. With ``visual=True``, gathers the non-colliding mesh shapes
    instead (the detailed render geometry), falling back to the collision set when
    an asset has no separate visual meshes.

    The asset must be a SINGLE rigid body, and the body's authored root pose is
    deliberately NOT composed in: the IsaacLab spawner REPLACES the root prim's
    transform with the cfg ``init_state`` (e.g. ``pat_vention.usd`` authors a -90deg
    z rotation the live env never sees), so the returned mesh is in the
    spawned-root frame -- compose it with ``init_state`` only. ``scale`` mirrors the
    spawn cfg's root-prim scale. Returns ``(vertices[V, 3], faces[F, 3])`` with
    faces re-wound (if needed) so face normals point out of the material --
    antipodal sampling and signed-distance queries both depend on the normal sign.
    """
    builder = newton.ModelBuilder()
    builder.add_usd(usd_path, floating=False, skip_mesh_approximation=True)
    if builder.body_count != 1:
        raise ValueError(f"{usd_path} added {builder.body_count} bodies; expected a single rigid body")
    model = builder.finalize(device=device)

    flags = model.shape_flags.numpy()
    stype = model.shape_type.numpy()
    sscale = model.shape_scale.numpy()
    shape_tf = wp.to_torch(model.shape_transform).cpu().numpy()
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    is_collide = [bool(int(flags[si]) & collide) for si in range(model.shape_count)]
    if visual:
        sel = [si for si in range(model.shape_count) if not is_collide[si] and model.shape_source[si] is not None]
        if not sel:  # no separate visual geometry; fall back to the collision set
            sel = [si for si in range(model.shape_count) if is_collide[si]]
    else:
        sel = [si for si in range(model.shape_count) if is_collide[si]]
    verts, faces, base = [], [], 0
    for si in sel:
        if model.shape_source[si] is not None:
            v = np.asarray(model.shape_source[si].vertices, dtype=np.float32).reshape(-1, 3)
            f = np.asarray(model.shape_source[si].indices, dtype=np.int32).reshape(-1, 3)
        elif int(stype[si]) == int(GeoType.BOX):
            v = _BOX_VERTS * sscale[si]
            f = _BOX_FACES
        else:
            raise ValueError(f"unsupported collision shape type {GeoType(int(stype[si])).name} in {usd_path}")
        verts.append(_quat_xyzw_rot(shape_tf[si, 3:7], v) + shape_tf[si, :3])
        faces.append(f + base)
        base += len(v)
    if not verts:
        raise RuntimeError(f"no collision meshes found in {usd_path}")
    v_np = np.concatenate(verts)
    f_np = np.concatenate(faces)
    if scale is not None:
        v_np = v_np * np.asarray(scale, dtype=np.float32)

    # Orient outward: the signed volume of a closed, consistently-wound mesh is
    # positive when face normals point out of the material.
    v0, v1, v2 = v_np[f_np[:, 0]], v_np[f_np[:, 1]], v_np[f_np[:, 2]]
    if float((np.cross(v0, v1) * v2).sum()) < 0.0:
        f_np = f_np[:, ::-1].copy()
    return v_np, f_np


class FactoryIKModel:
    """Owns the robot-only Newton model and the asset/obstacle geometry around it.

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
        obstacle_spec: ``name -> (usd_path, spawn_scale, world pos, world quat xyzw)``
            per static obstacle, for tooling that re-loads e.g. the visual geometry.
        static_obstacles: ``name -> warp.Mesh`` static world-frame obstacle meshes.
        board_verts: Nistboard collider vertices in its spawned-root frame [m], ``[V, 3]``.
        board_faces: Nistboard collider faces (outward wound), ``[F, 3]``.
        board_mesh: Nistboard :class:`warp.Mesh` in its own frame (per-sub-world posed).
        board_probes: Nistboard surface probe offsets in its frame [m], ``[P, 3]``.
        board_init_pos: Nistboard canonical scene position [m], ``[3]``.
        board_init_quat: Nistboard canonical scene orientation (xyzw), ``[4]``.
        fixed_verts: Fixed-asset collider vertices in its spawned-root frame [m], ``[V, 3]``.
        fixed_faces: Fixed-asset collider faces (outward wound), ``[F, 3]``.
        fixed_mesh: Fixed-asset :class:`warp.Mesh` in its own frame (per-sub-world posed).
        base_mesh: Robot base collider as a static world :class:`warp.Mesh` (the base
            is excluded from the robot probes; the sampled board is checked against it).
    """

    def __init__(self, cfg: FactoryIKPipelineCfg):
        self.cfg = cfg
        self.device = cfg.device
        self._rng = np.random.default_rng(cfg.seed)
        # probe budgets come from the criterion cfgs (membership in cfg.criteria is
        # what enables a gate); sets are built regardless so the avoidance
        # objectives and the relief pass can share them, with safe defaults
        from .cfg import CollisionCheckCfg, find_criterion

        crit = find_criterion(cfg.robot.criteria, CollisionCheckCfg)
        self._n_gripper_probes = crit.n_samples if crit else 240
        self._n_robot_probes = crit.n_samples if crit else 240
        self._n_held_probes = crit.n_samples if crit else 240
        self._n_self_probes = crit.n_samples if crit else 240
        self._self_hops = crit.adjacency_hops if crit else 2

        if cfg.scene is None:
            raise ValueError(
                "pipeline cfg.scene is unset: the env wiring assigns env.cfg.scene; standalone tools use"
                " resolve_from_task()"
            )
        scene = cfg.scene
        held_usd = getattr(scene, cfg.placement.held_asset_cfg.name).spawn.usd_path
        fixed_usd = getattr(scene, cfg.board.fixed_asset_cfg.name).spawn.usd_path

        # --- robot-only kinematic model ---
        builder = newton.ModelBuilder()
        robot_usd = cfg.robot.usd_path or getattr(getattr(scene, cfg.robot.asset_cfg.name).spawn, "usd_path", "")
        if not robot_usd:
            raise ValueError(
                "no robot USD: set cfg.robot.usd_path or provide a scene whose robot entry has a USD spawn"
            )
        res = builder.add_usd(robot_usd, collapse_fixed_joints=False)
        names = [""] * builder.body_count
        for path, idx in res.get("path_body_map", {}).items():
            names[idx] = path.rsplit("/", 1)[-1]
        self.ee_body = names.index(cfg.robot.ee_body_name)
        self._finger_bodies = [names.index(n) for n in cfg.robot.finger_body_names]
        self.gripper_bodies = [names.index(n) for n in cfg.robot.gripper_body_names if n in names]
        if not self.gripper_bodies:
            raise ValueError(
                f"none of gripper_body_names {cfg.robot.gripper_body_names} found; available: {[n for n in names if n]}"
            )
        self.model = builder.finalize(device=cfg.device)
        self.nq = self.model.joint_coord_count
        self.ndof = self.model.joint_dof_count
        self.body_count = self.model.body_count
        self._resolve_coords()
        self.pad_bodies, self.pad_offsets = self._derive_pad_offsets()

        # --- held asset: canonical mesh + probes (per-problem posed) ---
        self.held_verts, self.held_faces = load_collider_mesh(held_usd, cfg.device)
        self.held_mesh = wp.Mesh(
            points=wp.array(self.held_verts, dtype=wp.vec3, device=cfg.device),
            indices=wp.array(self.held_faces.reshape(-1), dtype=wp.int32, device=cfg.device),
        )
        tri = self.held_verts[self.held_faces]
        self.held_probes = _fps(np.concatenate([self.held_verts, tri.mean(axis=1)]), self._n_held_probes).astype(
            np.float32
        )

        # --- static obstacles: the scene assets named in cfg.obstacle_asset_names,
        # each flattened into a world-frame mesh at its scene init pose. The scene
        # cfg is the single source of truth for asset geometry/poses.
        self.obstacle_geom: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.obstacle_spec: dict[str, tuple[str, tuple | None, np.ndarray, np.ndarray]] = {}
        for name in cfg.obstacle_asset_names:
            asset_cfg = getattr(scene, name)
            spawn_scale = getattr(asset_cfg.spawn, "scale", None)
            v, f = load_collider_mesh(asset_cfg.spawn.usd_path, cfg.device, scale=spawn_scale)
            pos = np.asarray(asset_cfg.init_state.pos, dtype=np.float32)
            quat = np.asarray(asset_cfg.init_state.rot, dtype=np.float32)  # xyzw in this fork
            self.obstacle_geom[name] = (_quat_xyzw_rot(quat, v) + pos, f)
            self.obstacle_spec[name] = (asset_cfg.spawn.usd_path, spawn_scale, pos, quat)
        self.static_obstacles = {
            name: wp.Mesh(
                points=wp.array(v, dtype=wp.vec3, device=cfg.device),
                indices=wp.array(f.reshape(-1), dtype=wp.int32, device=cfg.device),
            )
            for name, (v, f) in self.obstacle_geom.items()
        }

        # --- the posed assembly group: nistboard + fixed asset (bolt), canonical
        # meshes with per-sub-world poses. The bolt rides the board at its keypoint
        # offset, so they always move together at the same relative pose.
        board_cfg = getattr(scene, cfg.board.board_asset_cfg.name)
        board_scale = getattr(board_cfg.spawn, "scale", None)
        self.board_verts, self.board_faces = load_collider_mesh(board_cfg.spawn.usd_path, cfg.device, scale=board_scale)
        self.board_mesh = wp.Mesh(
            points=wp.array(self.board_verts, dtype=wp.vec3, device=cfg.device),
            indices=wp.array(self.board_faces.reshape(-1), dtype=wp.int32, device=cfg.device),
        )
        tri_b = self.board_verts[self.board_faces]
        self.board_probes = _fps(np.concatenate([self.board_verts, tri_b.mean(axis=1)]), 128).astype(np.float32)
        self.board_init_pos = np.asarray(board_cfg.init_state.pos, dtype=np.float32)
        self.board_init_quat = np.asarray(board_cfg.init_state.rot, dtype=np.float32)  # xyzw
        self.board_spec = (board_cfg.spawn.usd_path, board_scale)
        self.fixed_verts, self.fixed_faces = load_collider_mesh(fixed_usd, cfg.device)
        self.fixed_mesh = wp.Mesh(
            points=wp.array(self.fixed_verts, dtype=wp.vec3, device=cfg.device),
            indices=wp.array(self.fixed_faces.reshape(-1), dtype=wp.int32, device=cfg.device),
        )
        self.fixed_spec = (fixed_usd, None)

        # robot base collider as a static world mesh: the base is excluded from the
        # robot probe set (it sits on its mount), but the SAMPLED board must not
        # intersect it -- checked at board-sampling time.
        base_shapes = self._collision_shapes(0)
        shape_tf0 = wp.to_torch(self.model.shape_transform).cpu().numpy()
        b0 = self.eval_fk(self.default_joint_q().unsqueeze(0))[0, 0].cpu().numpy()
        bv, bf, base_off = [], [], 0
        for si in base_shapes:
            v = np.asarray(self.model.shape_source[si].vertices, dtype=np.float32).reshape(-1, 3)
            f = np.asarray(self.model.shape_source[si].indices, dtype=np.int32).reshape(-1, 3)
            v = _quat_xyzw_rot(shape_tf0[si, 3:7], v) + shape_tf0[si, :3]
            bv.append(_quat_xyzw_rot(b0[3:7], v) + b0[:3])
            bf.append(f + base_off)
            base_off += len(v)
        self.base_mesh = wp.Mesh(
            points=wp.array(np.concatenate(bv), dtype=wp.vec3, device=cfg.device),
            indices=wp.array(np.concatenate(bf).reshape(-1), dtype=wp.int32, device=cfg.device),
        )

        self._setup_gripper_probes()
        self._setup_robot_probes()
        self._setup_robot_edges()
        self._setup_self_collision()

        # board collider edges (board frame) for the reverse crossing direction
        self.board_edge_p0, self.board_edge_p1 = _mesh_edges(self.board_verts, self.board_faces)

    def _resolve_coords(self) -> None:
        """Resolve arm/finger joint coordinate + DOF indices from the model topology.

        No hardcoded layout: the arm is the root-to-EE chain, the fingers are the
        joints whose child bodies are the finger bodies. A multi-finger end
        effector only changes ``finger_body_names`` and the pad indicators.
        """
        jp = self.model.joint_parent.numpy()
        jc = self.model.joint_child.numpy()
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        labels = [str(lbl).rsplit("/", 1)[-1] for lbl in self.model.joint_label]
        child_joint = {int(jc[j]): j for j in range(self.model.joint_count)}
        arm: list[int] = []
        arm_dofs: list[int] = []
        arm_names: list[str] = []
        b = self.ee_body
        while b in child_joint:
            j = child_joint[b]
            coords = list(range(int(q_start[j]), int(q_start[j + 1])))
            arm = coords + arm
            arm_dofs = list(range(int(qd_start[j]), int(qd_start[j + 1]))) + arm_dofs
            arm_names = [labels[j]] * len(coords) + arm_names
            b = int(jp[j])
        self.arm_coords = arm
        self.arm_dofs = arm_dofs
        self.arm_joint_names = arm_names
        self.finger_coords, self.finger_dofs, self.finger_joint_names = [], [], []
        for fb in self._finger_bodies:
            j = child_joint[fb]
            coords = list(range(int(q_start[j]), int(q_start[j + 1])))
            self.finger_coords += coords
            self.finger_dofs += list(range(int(qd_start[j]), int(qd_start[j + 1])))
            self.finger_joint_names += [labels[j]] * len(coords)
        self.arm_stance = self._resolve_stance()

    def _resolve_stance(self) -> torch.Tensor:
        """Arm stance per arm coord [rad or m], from ``cfg.robot.default_joint_q`` or the
        scene robot's ``init_state.joint_pos`` (joint-name patterns supported)."""
        src = self.cfg.robot.default_joint_q
        if src is None:
            robot_entry = getattr(self.cfg.scene, self.cfg.robot.asset_cfg.name)
            src = getattr(getattr(robot_entry, "init_state", None), "joint_pos", None)
        if not isinstance(src, dict):
            raise ValueError(
                "no joint stance: set cfg.robot.default_joint_q (name -> value) or provide a scene whose robot entry"
                " carries init_state.joint_pos"
            )
        values = []
        for name in self.arm_joint_names:
            for pattern, value in src.items():
                if pattern == name or re.fullmatch(pattern, name):
                    values.append(float(value))
                    break
            else:
                raise ValueError(f"joint stance missing for arm joint {name!r} (patterns: {list(src)})")
        return torch.tensor(values, device=self.device)

    def eval_fk(self, joint_q: torch.Tensor) -> torch.Tensor:
        """Batched forward kinematics. Returns body transforms ``[N, body_count, 7]`` (pos + xyzw)."""
        n = joint_q.shape[0]
        bq = wp.zeros((n, self.body_count), dtype=wp.transformf, device=self.device)
        eval_fk_batched(
            self.model,
            wp.from_torch(joint_q.contiguous()),
            wp.zeros((n, self.ndof), dtype=wp.float32, device=self.device),
            bq,
            wp.zeros((n, self.body_count), dtype=wp.spatial_vectorf, device=self.device),
        )
        return wp.to_torch(bq).view(n, self.body_count, 7)

    def default_joint_q(self) -> torch.Tensor:
        """Default joint coordinates: the franka arm stance, fingers mid-open, shape ``[nq]``."""
        jq = torch.tensor(self.model.joint_q.numpy(), device=self.device)
        jq[self.arm_coords] = self.arm_stance
        jq[self.finger_coords] = 0.02
        return jq

    def _derive_pad_offsets(self) -> tuple[list[int], torch.Tensor]:
        """Pad-contact offsets in the finger body frames, derived from FK (no annotation).

        The parallel jaw closes along EE-y and the finger joint translates each finger
        body along it, so at finger coordinate ``q`` the pad inner surface passes through
        ``ee_pos +/- q * ee_y``. Expressing that point in each finger's body frame gives a
        constant offset; verified at a second opening before returning.

        Returns:
            ``(pad_bodies[2], pad_offsets[2, 3])`` ordered (+jaw-y finger, -jaw-y finger).
        """

        def pads_world(q_f: float) -> tuple[torch.Tensor, torch.Tensor]:
            jq = torch.tensor(self.model.joint_q.numpy(), device=self.device).unsqueeze(0).clone()
            jq[0, self.arm_coords] = self.arm_stance
            jq[0, self.finger_coords] = q_f
            b = self.eval_fk(jq)[0]
            ee_pos, ee_quat = b[self.ee_body, :3], b[self.ee_body, 3:7]
            ee_y = math_utils.quat_apply(ee_quat.unsqueeze(0), torch.tensor([[0.0, 1.0, 0.0]], device=self.device))[0]
            return b, torch.stack([ee_pos + q_f * ee_y, ee_pos - q_f * ee_y])

        b0, pads0 = pads_world(0.02)
        ee_pos = b0[self.ee_body, :3]
        proj0 = float(((b0[self._finger_bodies[0], :3] - ee_pos) * (pads0[0] - ee_pos)).sum())
        ordered = list(self._finger_bodies) if proj0 > 0 else list(reversed(self._finger_bodies))
        offsets = []
        for k, fb in enumerate(ordered):
            p, _ = math_utils.subtract_frame_transforms(
                b0[fb, :3].unsqueeze(0), b0[fb, 3:7].unsqueeze(0), pads0[k].unsqueeze(0)
            )
            offsets.append(p[0])
        offsets = torch.stack(offsets)
        b1, pads1 = pads_world(0.035)
        for k, fb in enumerate(ordered):
            world = math_utils.quat_apply(b1[fb, 3:7].unsqueeze(0), offsets[k].unsqueeze(0))[0] + b1[fb, :3]
            err = float((world - pads1[k]).norm())
            if err > 1e-5:
                raise RuntimeError(f"pad offset not rigid in the finger frame (err={err:.2e} m)")
        return ordered, offsets

    def _collision_shapes(self, body: int | None = None) -> list[int]:
        """Collision shape indices carrying a source mesh (optionally on one body)."""
        flags = self.model.shape_flags.numpy()
        sbody = self.model.shape_body.numpy()
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        return [
            si
            for si in range(self.model.shape_count)
            if (int(flags[si]) & collide)
            and self.model.shape_source[si] is not None
            and (body is None or int(sbody[si]) == body)
        ]

    def _shape_surface_probes(self, si: int, n: int, shape_tf: np.ndarray) -> np.ndarray:
        """FPS ``n`` points on collision shape ``si``, lifted into its body frame [m], ``[n, 3]``.

        Candidates are area-weighted random surface samples (not just vertices and
        centroids), so low-poly box colliders (e.g. the Franka fingers, 8 vertices)
        get probes across their face interiors at any requested density.
        """
        v = np.asarray(self.model.shape_source[si].vertices, dtype=np.float32).reshape(-1, 3)
        f = np.asarray(self.model.shape_source[si].indices, dtype=np.int32).reshape(-1, 3)
        tri = v[f]
        areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        m_cand = max(4 * n, 256)
        fi = self._rng.choice(len(f), size=m_cand, p=areas / areas.sum())
        u = self._rng.random((m_cand, 1), dtype=np.float32)
        w = self._rng.random((m_cand, 1), dtype=np.float32)
        flip = (u + w) > 1.0
        u, w = np.where(flip, 1.0 - u, u), np.where(flip, 1.0 - w, w)
        pts = tri[fi, 0] + u * (tri[fi, 1] - tri[fi, 0]) + w * (tri[fi, 2] - tri[fi, 0])
        return (_quat_xyzw_rot(shape_tf[si, 3:7], _fps(pts, n)) + shape_tf[si, :3]).astype(np.float32)

    def _setup_gripper_probes(self) -> None:
        """Surface probes + collider-mesh targets for the gripper bodies.

        The probes drive the gripper-points-vs-nut queries; the per-body collider
        meshes (``gripper_target_*``) drive the SYMMETRIC nut-points-vs-gripper
        queries -- point-vs-mesh checks are one-directional, so a nut corner poking
        into a finger face between probes is only visible from the nut's side.
        """
        sptr = self.model.shape_source_ptr.numpy()
        shape_tf = wp.to_torch(self.model.shape_transform).cpu().numpy()
        per = max(1, self._n_gripper_probes // len(self.gripper_bodies))
        offsets, bodies, t_body, t_mesh, t_tf = [], [], [], [], []
        for gb in self.gripper_bodies:
            shapes = self._collision_shapes(gb)
            if not shapes:
                raise RuntimeError(f"no collision mesh on gripper body {gb}")
            off = self._shape_surface_probes(shapes[0], per, shape_tf)
            offsets.append(off)
            bodies.append(np.full(len(off), gb, dtype=np.int32))
            t_body.append(gb)
            t_mesh.append(int(sptr[shapes[0]]))
            t_tf.append(shape_tf[shapes[0], :7])
        self.gripper_probes = np.concatenate(offsets)
        self.gripper_probe_bodies = np.concatenate(bodies)
        self.gripper_target_bodies = np.array(t_body, dtype=np.int32)
        self.gripper_target_meshes = np.array(t_mesh, dtype=np.uint64)
        self.gripper_target_tf = np.array(t_tf, dtype=np.float32)

    def _setup_robot_probes(self) -> None:
        """Surface probes across ALL robot links except the base, for the
        robot-vs-static-obstacle criteria -- the arm/elbow must clear the table
        too, not just the gripper. The base link is excluded because it
        legitimately sits on its mount."""
        sbody = self.model.shape_body.numpy()
        shape_tf = wp.to_torch(self.model.shape_transform).cpu().numpy()
        shapes = [si for si in self._collision_shapes() if int(sbody[si]) > 0]
        per = max(1, self._n_robot_probes // len(shapes))
        offsets, bodies = [], []
        for si in shapes:
            off = self._shape_surface_probes(si, per, shape_tf)
            offsets.append(off)
            bodies.append(np.full(len(off), int(sbody[si]), dtype=np.int32))
        self.robot_probes = np.concatenate(offsets)
        self.robot_probe_bodies = np.concatenate(bodies)

    def _setup_robot_edges(self) -> None:
        """Collider edges across robot links (base excluded) for the edge-crossing
        criterion: point probes miss thin obstacles (the ~4 mm board) slicing
        between them, while an edge-vs-mesh raycast detects any surface crossing
        regardless of probe density."""
        sbody = self.model.shape_body.numpy()
        shape_tf = wp.to_torch(self.model.shape_transform).cpu().numpy()
        p0s, p1s, bodies = [], [], []
        for si in self._collision_shapes():
            if int(sbody[si]) == 0:
                continue
            v = np.asarray(self.model.shape_source[si].vertices, dtype=np.float32).reshape(-1, 3)
            f = np.asarray(self.model.shape_source[si].indices, dtype=np.int32).reshape(-1, 3)
            p0, p1 = _mesh_edges(v, f)
            p0s.append(_quat_xyzw_rot(shape_tf[si, 3:7], p0) + shape_tf[si, :3])
            p1s.append(_quat_xyzw_rot(shape_tf[si, 3:7], p1) + shape_tf[si, :3])
            bodies.append(np.full(len(p0), int(sbody[si]), dtype=np.int32))
        self.robot_edge_p0 = np.concatenate(p0s).astype(np.float32)
        self.robot_edge_p1 = np.concatenate(p1s).astype(np.float32)
        self.robot_edge_bodies = np.concatenate(bodies)

    def _setup_self_collision(self) -> None:
        """Robot link-vs-link probes/targets + a kinematic-adjacency filter.

        Probes on every robot collision shape are signed-distance-tested against every
        other robot link's mesh, skipping pairs within
        ``self_collision_adjacency_hops`` joints -- their colliders are designed to touch.
        """
        sbody = self.model.shape_body.numpy()
        sptr = self.model.shape_source_ptr.numpy()
        shape_tf = wp.to_torch(self.model.shape_transform).cpu().numpy()
        robot_shapes = self._collision_shapes()
        per = max(1, self._n_self_probes // len(robot_shapes))
        offsets, pbody, tbody, tmesh, ttf = [], [], [], [], []
        for si in robot_shapes:
            b = int(sbody[si])
            off = self._shape_surface_probes(si, per, shape_tf)
            offsets.append(off)
            pbody.append(np.full(len(off), b, dtype=np.int32))
            tbody.append(b)
            tmesh.append(int(sptr[si]))
            ttf.append(shape_tf[si, :7])  # transformf layout: pos (3) + quat xyzw (4)
        self.self_probes = np.concatenate(offsets)
        self.self_probe_bodies = np.concatenate(pbody)
        self.self_target_bodies = np.array(tbody, dtype=np.int32)
        self.self_target_meshes = np.array(tmesh, dtype=np.uint64)
        self.self_target_tf = np.array(ttf, dtype=np.float32)
        self.self_adjacency = self._adjacency_matrix(self._self_hops)

    def _adjacency_matrix(self, hops: int) -> np.ndarray:
        """``uint8[body_count, body_count]``: 1 where bodies are within ``hops`` joints (or equal)."""
        nb = self.body_count
        jp, jc = self.model.joint_parent.numpy(), self.model.joint_child.numpy()
        step = np.eye(nb, dtype=np.int32)  # self + direct edges
        for j in range(self.model.joint_count):
            p, c = int(jp[j]), int(jc[j])
            if p >= 0 and c >= 0:
                step[p, c] = step[c, p] = 1
        reach = np.eye(nb, dtype=np.int32)
        for _ in range(max(1, hops)):
            reach = (reach @ step > 0).astype(np.int32)
        return (reach > 0).astype(np.uint8)
