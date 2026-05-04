# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.utils.warp import convert_to_warp_mesh

from ..curriculum.reset_state import temporary_seed
from .rigid_object_hasher import RigidObjectHasher

if TYPE_CHECKING:
    import trimesh


@cache
def _load_mesh_tensors(prim):
    tm = prim_to_trimesh(prim)
    verts = torch.from_numpy(tm.vertices.astype("float32"))
    faces = torch.from_numpy(tm.faces.astype("int64"))
    return verts, faces


def sample_object_point_cloud(
    num_envs: int,
    num_points: int,
    prim_path_pattern: str,
    device: str = "cuda",
    rigid_object_hasher: RigidObjectHasher | None = None,
    seed: int = 42,
) -> torch.Tensor | None:
    """Sample a point cloud on the collision geometry of an asset.

    Robust to heterogeneous collider counts across envs. Uses
    ``RigidObjectHasher`` to deduplicate identical colliders.
    """
    try:
        from pytorch3d.ops import sample_farthest_points, sample_points_from_meshes
        from pytorch3d.structures import Meshes
    except ImportError as err:
        raise ImportError("sample_object_point_cloud requires the optional dependency 'pytorch3d'.") from err

    hasher = (
        rigid_object_hasher
        if rigid_object_hasher is not None
        else RigidObjectHasher(num_envs, prim_path_pattern, device=device)
    )

    if hasher.num_root == 0:
        return None

    replicated_env = torch.all(hasher.root_prim_hashes == hasher.root_prim_hashes[0])
    if replicated_env:
        mask_env0 = hasher.collider_prim_env_ids == 0
        verts_list, faces_list = zip(*[_load_mesh_tensors(p) for p, m in zip(hasher.collider_prims, mask_env0) if m])
        meshes = Meshes(verts=[v.to(device) for v in verts_list], faces=[f.to(device) for f in faces_list])
        rel_pos = hasher.collider_rel_pos[mask_env0]
        rel_mat = hasher.collider_rel_mat[mask_env0]
    else:
        verts_list, faces_list = zip(*[_load_mesh_tensors(p) for p in hasher.collider_prims])
        meshes = Meshes(verts=[v.to(device) for v in verts_list], faces=[f.to(device) for f in faces_list])
        rel_pos = hasher.collider_rel_pos
        rel_mat = hasher.collider_rel_mat
    with temporary_seed(seed):
        samp = sample_points_from_meshes(meshes, num_points * 2)
        local, _ = sample_farthest_points(samp, K=num_points)
        root = torch.einsum("nij,npj->npi", rel_mat.to(device), local) + rel_pos.to(device).unsqueeze(1)

        if replicated_env:
            buf = root.reshape(1, -1, 3)
            merged, _ = sample_farthest_points(buf, K=num_points)
            result = merged.view(1, num_points, 3).expand(num_envs, -1, -1) * hasher.root_prim_scales.unsqueeze(1)
        else:
            env_ids = hasher.collider_prim_env_ids.to(device)
            counts = torch.bincount(env_ids, minlength=hasher.num_root)
            max_c = int(counts.max().item())
            buf = torch.zeros((hasher.num_root, max_c * num_points, 3), device=device, dtype=root.dtype)
            placed = torch.zeros_like(counts)
            for i in range(len(hasher.collider_prims)):
                r = int(env_ids[i].item())
                start = placed[r].item() * num_points
                buf[r, start : start + num_points] = root[i]
                placed[r] += 1
            merged, _ = sample_farthest_points(buf, K=num_points)
            result = merged * hasher.root_prim_scales.unsqueeze(1)

    return result


def _triangulate_faces(prim) -> np.ndarray:
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    faces = []
    it = iter(indices)
    for cnt in counts:
        poly = [next(it) for _ in range(cnt)]
        for k in range(1, cnt - 1):
            faces.append([poly[0], poly[k], poly[k + 1]])
    return np.asarray(faces, dtype=np.int64)


def create_primitive_mesh(prim) -> trimesh.Trimesh:
    import trimesh
    from trimesh.transformations import rotation_matrix

    from pxr import UsdGeom

    prim_type = prim.GetTypeName()
    if prim_type == "Cube":
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        return trimesh.creation.box(extents=(size, size, size))
    elif prim_type == "Sphere":
        r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        return trimesh.creation.icosphere(subdivisions=3, radius=r)
    elif prim_type == "Cylinder":
        c = UsdGeom.Cylinder(prim)
        return trimesh.creation.cylinder(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    elif prim_type == "Capsule":
        c = UsdGeom.Capsule(prim)
        tri_mesh = trimesh.creation.capsule(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
        if c.GetAxisAttr().Get() == "X":
            R = rotation_matrix(np.radians(-90), [0, 1, 0])
            tri_mesh.apply_transform(R)
        elif c.GetAxisAttr().Get() == "Y":
            R = rotation_matrix(np.radians(90), [1, 0, 0])
            tri_mesh.apply_transform(R)
        return tri_mesh
    elif prim_type == "Cone":
        c = UsdGeom.Cone(prim)
        radius = c.GetRadiusAttr().Get()
        height = c.GetHeightAttr().Get()
        mesh = trimesh.creation.cone(radius=radius, height=height)
        mesh.apply_translation((0.0, 0.0, -height / 2.0))
        return mesh
    else:
        raise KeyError(f"{prim_type} is not a valid primitive mesh type")


def prim_to_trimesh(prim, relative_to_world=False) -> trimesh.Trimesh:
    import trimesh

    import omni
    from pxr import UsdGeom

    if prim.GetTypeName() == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        faces = _triangulate_faces(prim)
        mesh_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    else:
        mesh_tm = create_primitive_mesh(prim)

    if relative_to_world:
        tf = np.array(omni.usd.get_world_transform_matrix(prim)).T
        mesh_tm.apply_transform(tf)

    return mesh_tm


def prim_to_warp_mesh(prim, device, relative_to_world=False) -> wp.Mesh:
    from pxr import UsdGeom

    if prim.GetTypeName() == "Mesh":
        mesh_prim = UsdGeom.Mesh(prim)
        points = np.asarray(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    else:
        mesh = create_primitive_mesh(prim)
        points = mesh.vertices.astype(np.float32)
        indices = mesh.faces.astype(np.int32)

    if relative_to_world:
        import omni

        tf = np.array(omni.usd.get_world_transform_matrix(prim)).T
        points = (points @ tf[:3, :3].T) + tf[:3, 3]

    wp_mesh = convert_to_warp_mesh(points, indices, device=device)
    return wp_mesh
