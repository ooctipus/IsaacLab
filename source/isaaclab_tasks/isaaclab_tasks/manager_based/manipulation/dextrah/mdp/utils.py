import os
import omni
import hashlib
import numpy as np
import torch
import trimesh
import logging
import warp as wp
from trimesh.sample import sample_surface
from pxr import UsdGeom, Gf
import isaacsim.core.utils.prims as prim_utils
from isaaclab.sim.utils import get_all_matching_child_prims
from isaaclab.utils.warp import convert_to_warp_mesh


def sample_object_point_cloud(num_envs: int, num_points: int, prim_path: str,
                              cache_dir: str = "/tmp/isaaclab/sample_point_cloud",
                              device: str = "cpu") -> torch.Tensor:
    """
    Samples point clouds for each environment instance by collecting points
    from all matching USD prims under `prim_path`, then downsamples to
    exactly `num_points` per env using farthest-point sampling.

    Caches:
      - per-prim raw samples in `cache_dir/prim_samples/<hash>.npy`
      - final downsampled clouds in `cache_dir/final_samples/<hash>.npy`

    Args:
        num_envs (int): Number of environment instances.
        num_points (int): Points per instance.
        prim_path (str): USD prim path template with '{i}'.
        cache_dir (str): Base directory for on-disk caching.

    Returns:
        torch.Tensor: Shape (num_envs, num_points, 3).
    """
    # Prepare cache directories
    prim_cache_dir = os.path.join(cache_dir, "prim_samples")
    final_cache_dir = os.path.join(cache_dir, "final_samples")
    os.makedirs(prim_cache_dir, exist_ok=True)
    os.makedirs(final_cache_dir, exist_ok=True)

    points = torch.zeros((num_envs, num_points, 3), dtype=torch.float32, device=device)
    xform_cache = UsdGeom.XformCache()

    for i in range(num_envs):
        # Resolve prim path
        obj_path = prim_path.replace(".*", str(i))
        # Gather prims
        prims = get_all_matching_child_prims(
            obj_path,
            predicate=lambda p: p.GetTypeName() in (
                "Mesh","Cube","Sphere","Cylinder","Capsule","Cone"
            )
        )
        if not prims:
            raise KeyError(f"No valid prims under {obj_path}")

        # Compute per-prim geometry hashes for env-level cache key
        prim_hashes = []
        for prim in prims:
            prim_type = prim.GetTypeName()
            hasher = hashlib.sha256()
            hasher.update(prim_type.encode())
            if prim_type == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
                hasher.update(verts.tobytes())
            else:
                if prim_type == "Cube":
                    size = UsdGeom.Cube(prim).GetSizeAttr().Get()
                    hasher.update(np.float32(size).tobytes())
                elif prim_type == "Sphere":
                    r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
                    hasher.update(np.float32(r).tobytes())
                elif prim_type == "Cylinder":
                    c = UsdGeom.Cylinder(prim)
                    hasher.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                    hasher.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                elif prim_type == "Capsule":
                    c = UsdGeom.Capsule(prim)
                    hasher.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                    hasher.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                elif prim_type == "Cone":
                    c = UsdGeom.Cone(prim)
                    hasher.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                    hasher.update(np.float32(c.GetHeightAttr().Get()).tobytes())
            prim_hashes.append(hasher.hexdigest())
        
        # Compute USD scale between root and first prim
        object_prim = prim_utils.get_prim_at_path(obj_path)
        base_scale = torch.tensor(object_prim.GetAttribute("xformOp:scale").Get(), dtype=torch.float32)
        world_root = xform_cache.GetLocalToWorldTransform(object_prim)
        world_child = xform_cache.GetLocalToWorldTransform(prims[0])
        rel = world_root.GetInverse() * world_child
        sx = Gf.Vec3d(rel[0][0], rel[1][0], rel[2][0]).GetLength()
        sy = Gf.Vec3d(rel[0][1], rel[1][1], rel[2][1]).GetLength()
        sz = Gf.Vec3d(rel[0][2], rel[1][2], rel[2][2]).GetLength()
        usd_scale = (base_scale * torch.tensor([sx, sy, sz], dtype=torch.float32)).to(device)

        # Final cache key combines all prim hashes and num_points
        env_key = "_".join(sorted(prim_hashes)) + f"_{num_points}"
        env_hash = hashlib.sha256(env_key.encode()).hexdigest()
        final_file = os.path.join(final_cache_dir, f"{env_hash}.npy")

        # Load from final cache if present
        if os.path.exists(final_file):
            arr = np.load(final_file)
            if arr.shape == (num_points, 3):
                points[i] = torch.from_numpy(arr).to(device)  * usd_scale.unsqueeze(0)
                continue

        # Collect samples from each prim with per-prim caching
        all_samples = []
        for prim, prim_hash in zip(prims, prim_hashes):
            prim_type = prim.GetTypeName()
            prim_file = os.path.join(prim_cache_dir, f"{prim_hash}.npy")
            # Load or sample
            if os.path.exists(prim_file):
                arr = np.load(prim_file)
                if arr.shape[0] >= num_points:
                    samples = arr[:num_points]
                else:
                    samples = None
            else:
                samples = None
            if samples is None:
                if prim_type == "Mesh":
                    mesh = UsdGeom.Mesh(prim)
                    verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
                    faces = _triangulate_faces(prim)
                    mesh_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                else:
                    mesh_tm = create_primitive_mesh(prim)

                face_weights = mesh_tm.area_faces
                samples_np, _ = sample_surface(mesh_tm, num_points * 2, face_weight=face_weights)
                # prim-level FPS down to num_points
                tensor_pts = torch.from_numpy(samples_np.astype(np.float32)).to(device)
                prim_idxs = fps(tensor_pts, num_points)
                samples = tensor_pts[prim_idxs].cpu().numpy()
                if prim_type == "Cone":
                    samples[:, 2] -= UsdGeom.Cone(prim).GetHeightAttr().Get() / 2
                # save prim-level cache
                np.save(prim_file, samples)
            all_samples.append(samples)

        # Downsample combined samples if multiple prims
        if len(all_samples) == 1: # if this prim is not composite mesh, then we don't need to fps
            samples_final = torch.from_numpy(all_samples[0]).to(device)
        else:
            combined = torch.cat([torch.from_numpy(s) for s in all_samples], dim=0).to(device)
            idxs = fps(combined, num_points)
            samples_final = combined[idxs]

        # Save final downsampled cloud
        np.save(final_file, samples_final.cpu().numpy())
        points[i] = samples_final * usd_scale.unsqueeze(0)

    return points


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
        return trimesh.creation.capsule(
            radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get()
        )
    elif prim_type == "Cone":  # Cone
        c = UsdGeom.Cone(prim)
        return trimesh.creation.cone(
            radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get()
        )
    else:
        raise KeyError(f"{prim_type} is not a valid primitive mesh type")


def fps(points: torch.Tensor, n_samples: int, memory_threashold= 2 * 1024 ** 3) -> torch.Tensor:  # 2 GiB
    device = points.device
    N = points.shape[0]
    elem_size = points.element_size()
    bytes_needed = N * N * elem_size
    if bytes_needed <= memory_threashold:
        dist_mat = torch.cdist(points, points)
        sampled_idx = torch.zeros(n_samples, dtype=torch.long, device=device)
        min_dists = torch.full((N,), float('inf'), device=device)
        farthest = torch.randint(0, N, (1,), device=device)
        for j in range(n_samples):
            sampled_idx[j] = farthest
            min_dists = torch.minimum(min_dists, dist_mat[farthest].view(-1))
            farthest = torch.argmax(min_dists)
        return sampled_idx
    logging.warning(f"FPS fallback to iterative (needed {bytes_needed} > {memory_threashold})")
    sampled_idx = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)
    farthest = torch.randint(0, N, (1,), device=device)
    for j in range(n_samples):
        sampled_idx[j] = farthest
        dist = torch.norm(points - points[farthest], dim=1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances)
    return sampled_idx


def prim_to_warp_mesh(prim, device) -> wp.Mesh:
    transform_matrix = np.array(omni.usd.get_world_transform_matrix(prim)).T
    if prim.GetTypeName() == "Mesh":
        # cast into UsdGeomMesh
        mesh_prim = UsdGeom.Mesh(prim)
        points = np.asarray(mesh_prim.GetPointsAttr().Get())
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
    else:
        mesh = create_primitive_mesh(prim)
        points = mesh.vertices
        indices = mesh.faces
        
    points = np.matmul(points, transform_matrix[:3, :3].T) + transform_matrix[:3, 3]
    wp_mesh = convert_to_warp_mesh(points, indices, device=device)
    return wp_mesh


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