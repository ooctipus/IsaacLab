import omni
import numpy as np
import torch
from functools import lru_cache
import trimesh
import warp as wp
from trimesh.transformations import rotation_matrix
from pxr import UsdGeom
import isaaclab.utils.math as math_utils
from isaaclab.utils.warp import convert_to_warp_mesh
from .rigid_object_hasher import RigidObjectHasher

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points

@lru_cache(maxsize=None)
def _load_mesh_tensors(prim):
    tm  = prim_to_trimesh(prim)
    verts = torch.from_numpy(tm.vertices.astype("float32"))
    faces = torch.from_numpy(tm.faces.astype("int64"))
    return verts, faces

def sample_object_point_cloud(
    num_envs: int,
    num_points: int,
    prim_path_pattern: str,
    device: str = "cuda",  # assume GPU
    rigid_object_hasher: RigidObjectHasher | None = None,
) -> torch.Tensor | None:

    hasher = (
        rigid_object_hasher
        if rigid_object_hasher is not None
        else RigidObjectHasher(num_envs, prim_path_pattern, device=device)
    )

    if hasher.num_root == 0:
        return None
    
    replicated_env = torch.all(hasher.root_prim_hashes == hasher.root_prim_hashes[0])
    if replicated_env:
        # Pick env 0’s colliders
        mask_env0 = (hasher.collider_prim_env_ids == 0)
        verts_list, faces_list = zip(*[_load_mesh_tensors(p) for p, m in zip(hasher.collider_prims, mask_env0) if m])
        meshes = Meshes(verts=[v.to(device) for v in verts_list], faces=[f.to(device) for f in faces_list])
        rel_tf = hasher.collider_prim_relative_transforms[mask_env0]
    else:
        # Build all envs's colliders
        verts_list, faces_list = zip(*[_load_mesh_tensors(p) for p in hasher.collider_prims])
        meshes = Meshes(verts=[v.to(device) for v in verts_list], faces=[f.to(device) for f in faces_list])
        rel_tf = hasher.collider_prim_relative_transforms

    # Uniform‐surface sample then scale to root
    samp  = sample_points_from_meshes(meshes, num_points * 2)
    local, _ = sample_farthest_points(samp, K=num_points)
    t_rel, q_rel, s_rel = rel_tf[:, :3].unsqueeze(1), rel_tf[:, 3:7].unsqueeze(1), rel_tf[:, 7:].unsqueeze(1)
    world = math_utils.quat_apply(q_rel.expand(-1, num_points, -1), local * s_rel) + t_rel
    
    # Merge Colliders
    if replicated_env:
        buf = world.reshape(1, -1, 3)
        merged, _ = sample_farthest_points(buf, K=num_points)
        result = merged.view(1, num_points, 3).expand(num_envs, -1, -1) * hasher.root_prim_scales.unsqueeze(1)
    else:
        # 4) Scatter each collider into a padded per‐root buffer
        env_ids = hasher.collider_prim_env_ids.to(device)  # (M,)
        counts = torch.bincount(env_ids, minlength=hasher.num_root)   # (num_root,)
        max_c = int(counts.max().item())
        buf = torch.zeros((hasher.num_root, max_c * num_points, 3), device=device, dtype=world.dtype)
        # track how many placed in each root
        placed = torch.zeros_like(counts)
        for i in range(len(hasher.collider_prims)):
            r = int(env_ids[i].item())
            start = placed[r].item() * num_points
            buf[r, start:start+num_points] = world[i]
            placed[r] += 1
        # 5) One batch‐FPS to merge per‐root
        merged, _ = sample_farthest_points(buf, K=num_points)
        result = merged * hasher.root_prim_scales.unsqueeze(1)

    return result

def _triangulate_faces(prim) -> np.ndarray:
    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    faces = []
    it = iter(indices)
    for cnt in counts:
        poly = [next(it) for _ in range(cnt)]
        for k in range(1, cnt-1):
            faces.append([poly[0], poly[k], poly[k+1]])
    return np.asarray(faces, dtype=np.int64)


def create_primitive_mesh(prim) -> trimesh.Trimesh:
    prim_type = prim.GetTypeName()
    if prim_type == "Cube":
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        return trimesh.creation.box(extents=(size, size, size))
    elif prim_type == "Sphere":
        r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        return trimesh.creation.icosphere(subdivisions=3, radius=r)
    elif prim_type == "Cylinder":
        c = UsdGeom.Cylinder(prim)
        return trimesh.creation.cylinder(
            radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get()
        )
    elif prim_type == "Capsule":
        c = UsdGeom.Capsule(prim)
        tri_mesh = trimesh.creation.capsule(
            radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get()
        )
        if c.GetAxisAttr().Get() == "X":
            # rotate −90° about Y to point the length along +X
            R = rotation_matrix(np.radians(-90), [0, 1, 0])
            tri_mesh.apply_transform(R)
        elif c.GetAxisAttr().Get() == "Y":
            # rotate +90° about X to point the length along +Y
            R = rotation_matrix(np.radians(90), [1, 0, 0])
            tri_mesh.apply_transform(R)
        return tri_mesh
        
    elif prim_type == "Cone":
        c = UsdGeom.Cone(prim)
        radius = c.GetRadiusAttr().Get()
        height = c.GetHeightAttr().Get()
        mesh = trimesh.creation.cone(radius=radius, height=height)
        # shift all vertices down by height/2 for usd / trimesh cone primitive definiton discrepancy
        mesh.apply_translation((0.0, 0.0, -height/2.0))
        return mesh
    else:
        raise KeyError(f"{prim_type} is not a valid primitive mesh type")


def prim_to_trimesh(prim, relative_to_world=False) -> trimesh.Trimesh:
    if prim.GetTypeName() == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        faces = _triangulate_faces(prim)
        mesh_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    else:
        mesh_tm = create_primitive_mesh(prim)

    if relative_to_world:
        tf = np.array(omni.usd.get_world_transform_matrix(prim)).T  # shape (4,4)
        mesh_tm.apply_transform(tf)

    return mesh_tm


def prim_to_warp_mesh(prim, device, relative_to_world=False) -> wp.Mesh:
    if prim.GetTypeName() == "Mesh":
        mesh_prim = UsdGeom.Mesh(prim)
        points = np.asarray(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    else:
        mesh = create_primitive_mesh(prim)
        points = mesh.vertices.astype(np.float32)
        indices = mesh.faces.astype(np.int32)

    if relative_to_world:
        tf = np.array(omni.usd.get_world_transform_matrix(prim)).T  # (4,4)
        points = (points @ tf[:3, :3].T) + tf[:3, 3]

    wp_mesh = convert_to_warp_mesh(points, indices, device=device)
    return wp_mesh


def set_reset_state(env, states: torch.Tensor, env_ids: torch.Tensor, keys: list[str], is_relative: bool = False):
    idx = 0
    for name, articulation in env.scene._articulations.items():
        if name in keys:
            root_state = states[:, idx : idx + 13].clone()
            if is_relative:
                root_state[:, :3] += env.scene.env_origins[env_ids]
            articulation.write_root_state_to_sim(root_state, env_ids=env_ids)
            # joint state
            n_j = articulation.num_joints
            joint_position = states[:, idx + 13 : idx + 13 + n_j].clone()
            joint_velocity = states[:, idx + 13 + n_j : idx + 13 + 2 * n_j].clone()
            articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
            idx += (13 + 2 * n_j)
    # rigid objects
    for name, rigid_object in env.scene._rigid_objects.items():
        if name in keys:
            root_state = states[:, idx : idx + 13].clone()
            if is_relative:
                root_state[:, :3] += env.scene.env_origins[env_ids]
            rigid_object.write_root_state_to_sim(root_state, env_ids)
            idx += 13


def get_reset_state(env, env_id: torch.Tensor, keys: list[str], is_relative=False):
    states = []
    # articulations
    for name, articulation in env.scene._articulations.items():
        if name in keys:
            state = articulation.data.root_state_w[env_id].clone()
            if is_relative:
                state[:, :3] -= env.scene.env_origins[env_id]
            states.append(state)
            states.append(articulation.data.joint_pos[env_id].clone())
            states.append(articulation.data.joint_vel[env_id].clone())
    # rigid objects
    for name, rigid_object in env.scene._rigid_objects.items():
        if name in keys:
            state = rigid_object.data.root_state_w[env_id].clone()
            if is_relative:
                state[:, :3] -= env.scene.env_origins[env_id]
            states.append(state)
    return torch.cat(states, dim=-1)


@wp.kernel
def get_sign_distance(
    queries:        wp.array(dtype=wp.vec3),   # [E_bad * N]
    mesh_handles:   wp.array(dtype=wp.uint64), # [E_bad * max_prims]
    prim_counts:    wp.array(dtype=wp.int32),  # [E_bad]
    max_dist:       float,
    num_points:     int,
    max_prims:      int,
    signs:          wp.array(dtype=float),     # [E_bad * N]
):
    tid = wp.tid()
    env_id = tid // num_points
    q = queries[tid]
    # accumulator for the lowest‐sign (start large)
    best_sign = float(1)

    base = env_id * max_prims
    for p in range(prim_counts[env_id]):
        mid = mesh_handles[base + p]
        if mid != 0:
            mp = wp.mesh_query_point(mid, q, max_dist)
            if mp.result and mp.sign < best_sign:
                best_sign = mp.sign
    # write final values exactly once
    signs[tid] = best_sign


@wp.kernel
def get_sign_distance_no_mem(
    queries: wp.array(dtype=wp.vec3),    # [E_bad * N]
    mesh_handles: wp.array(dtype=wp.uint64),  # [E_bad * max_prims]
    prim_counts: wp.array(dtype=wp.int32),   # [E_bad]
    handle_root_pos: wp.array(dtype=wp.vec3),    # [E_bad * max_prims]
    handle_root_quat: wp.array(dtype=wp.quat),    # [E_bad * max_prims]
    handle_root_scale: wp.array(dtype=wp.vec3),    # [E_bad * max_prims]
    rel_pos: wp.array(dtype=wp.vec3),    # [E_bad * max_prims]
    rel_quat: wp.array(dtype=wp.quat),    # [E_bad * max_prims]
    rel_scale: wp.array(dtype=wp.vec3),    # [E_bad * max_prims]
    max_dist: float,
    num_points: int,
    max_prims: int,
    signs: wp.array(dtype=float)       # [E_bad * N]
):
    tid    = wp.tid()                        # global thread index
    env_id = tid // num_points               # which environment
    q_w    = queries[tid]                    # world‐space query point
    base   = env_id * max_prims              # start index for this env’s handles
    best = float(1)                              # accumulator for min signed distance

    # transform world→root and root→local per‐handle
    for p in range(prim_counts[env_id]):
        idx = base + p
        mid = mesh_handles[idx]
        if mid != 0:
            # // 1) world → root‐local
            v = q_w - handle_root_pos[idx]
            inv_rq = wp.quat_inverse(handle_root_quat[idx])
            v2 = wp.quat_rotate(inv_rq, v)
            
            s     = handle_root_scale[idx]
            inv_s = wp.vec3(1.0/s.x, 1.0/s.y, 1.0/s.z)
            v3    = comp_mul(v2, inv_s)

            # // 2 root‐local → mesh‐local
            v4 = v3 - rel_pos[idx]
            inv_lq = wp.quat_inverse(rel_quat[idx])
            v5 = wp.quat_rotate(inv_lq, v4)
            
            rs    = rel_scale[idx]
            inv_rs= wp.vec3(1.0/rs.x, 1.0/rs.y, 1.0/rs.z)
            q_local = comp_mul(v5, inv_rs)

            mp = wp.mesh_query_point(mid, q_local, max_dist)
            if mp.result and mp.sign < best:
                best = mp.sign

    signs[tid] = best

@wp.func
def comp_mul(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    return wp.vec3(a.x * b.x, a.y * b.y, a.z * b.z)