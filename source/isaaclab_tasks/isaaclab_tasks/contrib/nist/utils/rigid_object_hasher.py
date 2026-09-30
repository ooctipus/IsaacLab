# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.cloner.clone_plan import GeometryLayout
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import matrix_from_quat


class RigidObjectHasher:
    """Group immutable plan-declared colliders by source geometry."""

    def __init__(self, num_envs: int, prim_path_pattern: str):
        sim = SimulationContext.instance()
        plan = None if sim is None else sim.get_clone_plan()
        if plan is None or not plan.is_complete:
            raise RuntimeError("RigidObjectHasher requires a completed clone plan.")
        bodies = {body.path: body for body in plan.match_rigid_body_subtrees(prim_path_pattern)}

        collider_geometries: list[GeometryLayout] = []
        collider_keys: list[str] = []
        collider_body_names: list[str] = []
        collider_env_ids: list[int] = []
        collider_rel_pos: list[torch.Tensor] = []
        collider_rel_mat: list[torch.Tensor] = []
        root_ids = [-1] * num_envs
        root_signatures: dict[tuple, int] = {}
        for target, geometries in plan.match_geometry_targets(prim_path_pattern):
            if target.env_id is None or target.env_id >= num_envs:
                raise RuntimeError(f"Collider target {target.path!r} is not in the requested environment range.")
            colliders = tuple(geometry for geometry in geometries if geometry.collision)
            if not colliders:
                raise RuntimeError(f"Planned object {target.path!r} has no collision geometry.")
            signature = []
            for geometry in colliders:
                frame = geometry.frame
                if frame.body_path is None:
                    raise RuntimeError(f"Planned collider {geometry.path!r} has no rigid-body owner.")
                pose = frame.pose
                rotation = matrix_from_quat(torch.tensor(pose[3:], dtype=torch.float32))
                collider_geometries.append(geometry)
                collider_keys.append(geometry.source_path)
                collider_body_names.append(bodies[frame.body_path].name)
                collider_env_ids.append(target.env_id)
                collider_rel_pos.append(torch.tensor(pose[:3], dtype=torch.float32))
                collider_rel_mat.append(rotation)
                signature.append((geometry.source_path, pose))
            key = tuple(signature)
            root_ids[target.env_id] = root_signatures.setdefault(key, len(root_signatures))
        if any(root_id < 0 for root_id in root_ids):
            raise RuntimeError(f"Collider expression {prim_path_pattern!r} does not cover every environment.")

        self.num_root = num_envs
        self.collider_geometries = collider_geometries
        self.collider_keys = collider_keys
        self.collider_body_names = collider_body_names
        self.collider_prim_env_ids = torch.tensor(collider_env_ids, dtype=torch.int64)
        self.collider_rel_pos = torch.stack(collider_rel_pos)
        self.collider_rel_mat = torch.stack(collider_rel_mat)
        self.collider_rel_mat_inv = torch.linalg.inv(self.collider_rel_mat.to(torch.float64)).to(torch.float32)
        self.root_prim_hashes = torch.tensor(root_ids, dtype=torch.int64)
        self.root_prim_scales = torch.ones(num_envs, 3)
