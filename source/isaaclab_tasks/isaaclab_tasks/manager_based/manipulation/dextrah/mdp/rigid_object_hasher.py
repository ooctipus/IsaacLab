import hashlib
import numpy as np
import torch
import re
from pxr import UsdPhysics
from pxr import UsdGeom, Gf, Usd
import isaacsim.core.utils.prims as prim_utils
import isaaclab.utils.math as math_utils

HASH_STORE = {
    "warp_mesh_store":{}
}

class RigidObjectHasher:
    """Compute per-root and per-collider 64-bit hashes of transform+geometry."""

    def __init__(self, num_envs, prim_path_pattern, device="cpu"):
        self.prim_path_pattern = prim_path_pattern
        self.device = device
        if prim_path_pattern in HASH_STORE:
            return

        HASH_STORE[prim_path_pattern] = {
            "num_roots": 0,
            "collider_prims": [],
            "collider_prim_hashes": [],
            "collider_prim_env_ids": [],
            "collider_prim_relative_transforms": [],
            "root_prim_hashes": []
        }
        stor = HASH_STORE[prim_path_pattern]
        xform_cache = UsdGeom.XformCache()
        # prim_paths = prim_utils.get_all_matching_child_prims(
        #     "/World/envs", predicate=lambda p: bool(pattern.match(p)) and prim_utils.get_prim_at_path(p).HasAPI(UsdPhysics.RigidBodyAPI)
        # )
        prim_paths = [prim_path_pattern.replace(".*", f"{i}", 1) for i in range(num_envs)]

        num_roots = len(prim_paths)
        collider_prim_env_ids = []
        collider_prims: list[Usd.Prim] = []
        collider_prim_relative_transforms = []
        collider_prim_hashes = []
        root_prim_hashes = []
        for i in range(num_roots):
            # 1: Get all child prims that are colliders, count them, and store their belonging env id
            coll_prims = prim_utils.get_all_matching_child_prims(
                prim_paths[i], predicate=lambda p: prim_utils.get_prim_at_path(p).GetTypeName() in (
                    "Mesh","Cube","Sphere","Cylinder","Capsule","Cone"
                ) and prim_utils.get_prim_at_path(p).HasAPI(UsdPhysics.CollisionAPI)
            )
            if len(coll_prims) == 0:
                return
            collider_prims.extend(coll_prims)
            collider_prim_env_ids.extend([i] * len(coll_prims))
            
            # 2: Get relative transforms of all collider prims
            root_xf = Gf.Transform(xform_cache.GetLocalToWorldTransform(prim_utils.get_prim_at_path(prim_paths[i])))
            ts, qs, ss = [], [], []
            q_root = root_xf.GetRotation().GetQuat()
            q_root = torch.tensor([q_root.GetReal(), *q_root.GetImaginary()], dtype=torch.float32, device="cpu")
            t_root = torch.tensor([*root_xf.GetTranslation()], dtype=torch.float32, device="cpu")
            s_root = torch.tensor([*root_xf.GetScale()], dtype=torch.float32, device="cpu")
            for prim in coll_prims:
                child_xf = Gf.Transform(xform_cache.GetLocalToWorldTransform(prim))
                q_child = child_xf.GetRotation().GetQuat()      # Gf.Quatd
                qs.append(torch.tensor([q_child.GetReal(), *q_child.GetImaginary()], dtype=torch.float32, device="cpu"))
                ts.append(torch.tensor([*child_xf.GetTranslation()], dtype=torch.float32, device="cpu"))
                ss.append(torch.tensor([*child_xf.GetScale()], dtype=torch.float32, device="cpu"))

            t, q, s = torch.stack(ts), torch.stack(qs), torch.stack(ss)
            tq_rel = math_utils.subtract_frame_transforms(t_root.repeat(len(t), 1), q_root.repeat(len(t), 1), t, q)
            s_rel = s / s_root
            coll_relative_transform = torch.cat([tq_rel[0], tq_rel[1], s_rel], dim=1).flatten()
            collider_prim_relative_transforms.append(coll_relative_transform)
           
            # 3: Store the collider prims hash
            root_hash = hashlib.sha256()
            for prim, prim_rel_tf in zip(coll_prims, coll_relative_transform.numpy()):
                h = hashlib.sha256()
                h.update(prim_rel_tf.tobytes())
                prim_type = prim.GetTypeName()
                h.update(prim_type.encode("utf-8"))
                if prim_type == "Mesh":
                    verts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float32)
                    h.update(verts.tobytes())
                else:
                    if prim_type == "Cube":
                        s = UsdGeom.Cube(prim).GetSizeAttr().Get()
                        h.update(np.float32(s).tobytes())
                    elif prim_type == "Sphere":
                        r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
                        h.update(np.float32(r).tobytes())
                    elif prim_type == "Cylinder":
                        c = UsdGeom.Cylinder(prim)
                        h.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                        h.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                    elif prim_type == "Capsule":
                        c = UsdGeom.Capsule(prim)
                        h.update(c.GetAxisAttr().Get().encode("utf-8"))
                        h.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                        h.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                    elif prim_type == "Cone":
                        c = UsdGeom.Cone(prim)
                        h.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                        h.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                collider_hash = h.digest()
                root_hash.update(collider_hash)
                collider_prim_hashes.append(int.from_bytes(collider_hash[:8], "little", signed=True))
            small = int.from_bytes(root_hash.digest()[:8], "little", signed=True)
            root_prim_hashes.append(small)

        stor["num_roots"] = num_roots
        stor["collider_prims"] = collider_prims
        stor["collider_prim_hashes"] = torch.tensor(collider_prim_hashes, dtype=torch.int64, device=device)
        stor["collider_prim_env_ids"] = torch.tensor(collider_prim_env_ids, dtype=torch.int64, device=device)
        stor["collider_prim_relative_transforms"] = torch.cat(collider_prim_relative_transforms).view(-1, 10).to(device)
        stor["root_prim_hashes"] = torch.tensor(root_prim_hashes, dtype=torch.int64, device=device)

    @property
    def num_root(self) -> int:
        return self.get_key("num_roots")

    @property
    def root_prim_hashes(self) -> torch.Tensor:
        return self.get_key("root_prim_hashes")
    
    @property
    def collider_prim_relative_transforms(self) -> torch.Tensor:
        return self.get_key("collider_prim_relative_transforms")
    
    @property
    def collider_prim_hashes(self) -> torch.Tensor:
        return self.get_key("collider_prim_hashes")
    
    @property
    def collider_prims(self) -> list[Usd.Prim]:
        return self.get_key("collider_prims")
    
    @property
    def collider_prim_env_ids(self) -> torch.Tensor:
        return self.get_key("collider_prim_env_ids")
    
    def get_key(self, key: str):
        """Get the hash store for the hasher."""
        return HASH_STORE.get(self.prim_path_pattern, {}).get(key, None)
    
    def get_warp_mesh_store(self):
        """Get the warp mesh store for the hasher."""
        return HASH_STORE["warp_mesh_store"]
    
    def get_hash_store(self):
        """Get the entire hash store"""
        return HASH_STORE
