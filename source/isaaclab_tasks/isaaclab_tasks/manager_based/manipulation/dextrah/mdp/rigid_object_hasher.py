import hashlib
import numpy as np
import torch
import re
from pxr import UsdPhysics
from pxr import UsdGeom, Gf, Usd
import isaacsim.core.utils.prims as prim_utils
import isaaclab.utils.math as math_utils

class RigidObjectHasher:
    """Compute per-root and per-collider 64-bit hashes of transform+geometry."""

    def __init__(self, prim_path_pattern, device="cpu"):
        pattern = re.compile(f"^{prim_path_pattern}$")
        self.prim_path_pattern = prim_path_pattern
        self.device = device
        self.xform_cache = UsdGeom.XformCache()
        prim_paths = prim_utils.get_all_matching_child_prims(
            "/World", predicate=lambda p: bool(pattern.match(p)) and prim_utils.get_prim_at_path(p).HasAPI(UsdPhysics.RigidBodyAPI)
        )
        self.num_roots = len(prim_paths)
        self._collider_prim_env_ids = []
        self._collider_prims: list[Usd.Prim] = []
        self._collider_prim_relative_transforms = []
        self._collider_prim_hashes = []
        self._root_prim_hashes = []
        for i in range(self.num_roots):
            # 1: Get all child prims that are colliders, count them, and store their belonging env id
            collider_prims = prim_utils.get_all_matching_child_prims(
                prim_paths[i].GetPath(), predicate=lambda p: prim_utils.get_prim_at_path(p).GetTypeName() in (
                    "Mesh","Cube","Sphere","Cylinder","Capsule","Cone"
                ) and prim_utils.get_prim_at_path(p).HasAPI(UsdPhysics.CollisionAPI)
            )
            if len(collider_prims) == 0:
                return
            self._collider_prims.extend(collider_prims)
            self._collider_prim_env_ids.extend([i] * len(collider_prims))
            
            # 2: Get relative transforms of all collider prims
            root_xf = Gf.Transform(self.xform_cache.GetLocalToWorldTransform(prim_paths[i]))
            ts, qs, ss = [], [], []
            q_root = root_xf.GetRotation().GetQuat()
            q_root = torch.tensor([q_root.GetReal(), *q_root.GetImaginary()], dtype=torch.float32, device="cpu")
            t_root = torch.tensor([*root_xf.GetTranslation()], dtype=torch.float32, device="cpu")
            s_root = torch.tensor([*root_xf.GetScale()], dtype=torch.float32, device="cpu")
            for prim in collider_prims:
                child_xf = Gf.Transform(self.xform_cache.GetLocalToWorldTransform(prim))
                q_child = child_xf.GetRotation().GetQuat()      # Gf.Quatd
                qs.append(torch.tensor([q_child.GetReal(), *q_child.GetImaginary()], dtype=torch.float32, device="cpu"))
                ts.append(torch.tensor([*child_xf.GetTranslation()], dtype=torch.float32, device="cpu"))
                ss.append(torch.tensor([*child_xf.GetScale()], dtype=torch.float32, device="cpu"))

            t, q, s = torch.stack(ts), torch.stack(qs), torch.stack(ss)
            tq_rel = math_utils.subtract_frame_transforms(t_root.repeat(len(t), 1), q_root.repeat(len(t), 1), t, q)
            s_rel = s / s_root
            coll_relative_transform = torch.cat([tq_rel[0], tq_rel[1], s_rel], dim=1).flatten()
            self._collider_prim_relative_transforms.append(coll_relative_transform)
           
            # 3: Store the collider prims hash
            root_hash = hashlib.sha256()
            for prim, prim_rel_tf in zip(collider_prims, coll_relative_transform.numpy()):
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
                self._collider_prim_hashes.append(int.from_bytes(collider_hash[:8], "little", signed=True))
            small = int.from_bytes(root_hash.digest()[:8], "little", signed=True)
            self._root_prim_hashes.append(small)
        
        self._root_prim_hashes = torch.tensor(self._root_prim_hashes, dtype=torch.int64, device=device)
        self._collider_prim_hashes = torch.tensor(self._collider_prim_hashes, dtype=torch.int64, device=device)
        self._collider_prim_env_ids = torch.tensor(self._collider_prim_env_ids, dtype=torch.int64, device=device)
        self._collider_prim_relative_transforms = torch.cat(self._collider_prim_relative_transforms).view(-1, 10).to(device)

    @property
    def root_prim_hashes(self):
        return self._root_prim_hashes
    
    @property
    def collider_prim_relative_transforms(self):
        return self._collider_prim_relative_transforms
    
    @property
    def collider_prim_hashes(self):
        return self._collider_prim_hashes
    
    @property
    def collider_prims(self) -> list[Usd.Prim]:
        return self._collider_prims
    
    @property
    def collider_prim_env_ids(self):
        return self._collider_prim_env_ids
