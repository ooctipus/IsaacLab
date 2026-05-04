# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import hashlib

import numpy as np
import torch
import warp as wp

HASH_STORE = {"warp_mesh_store": {}}


class RigidObjectHasher:
    """Compute per-root and per-collider hashes and full 3x3 transforms."""

    def __init__(self, num_envs, prim_path_pattern, device="cpu"):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        from isaaclab.sim import get_all_matching_child_prims
        from isaaclab.sim.utils import get_current_stage

        self.prim_path_pattern = prim_path_pattern
        self.device = device
        if prim_path_pattern in HASH_STORE:
            return

        HASH_STORE[prim_path_pattern] = {
            "num_roots": 0,
            "collider_prims": [],
            "collider_prim_hashes": [],
            "collider_prim_env_ids": [],
            "collider_rel_pos": [],
            "collider_rel_mat": [],
            "collider_rel_mat_inv": [],
            "root_prim_hashes": [],
            "root_prim_scales": [],
        }
        stor = HASH_STORE[prim_path_pattern]
        xform_cache = UsdGeom.XformCache()
        prim_paths = [prim_path_pattern.replace(".*", f"{i}", 1) for i in range(num_envs)]

        num_roots = len(prim_paths)
        collider_prim_env_ids = []
        collider_prims: list[Usd.Prim] = []
        collider_rel_pos_list = []
        collider_rel_mat_list = []
        collider_rel_mat_inv_list = []
        collider_prim_hashes = []
        root_prim_hashes = []
        root_prim_scales = []

        stage = get_current_stage()

        def is_collider(p):
            return p.GetTypeName() in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone") and p.HasAPI(
                UsdPhysics.CollisionAPI
            )

        for i in range(num_roots):
            coll_prims = get_all_matching_child_prims(
                prim_paths[i],
                predicate=is_collider,
                traverse_instance_prims=True,
            )
            if len(coll_prims) == 0:
                return
            collider_prims.extend(coll_prims)
            collider_prim_env_ids.extend([i] * len(coll_prims))

            root_xf = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prim_paths[i]))
            root_tf = Gf.Transform(root_xf)
            root_prim_scales.append(torch.tensor(root_tf.GetScale()))

            root_hash = hashlib.sha256()
            env_collider_hashes = []
            env_rel_pos = []
            env_rel_mat = []
            env_rel_mat_inv = []
            for prim in coll_prims:
                child_xf = xform_cache.GetLocalToWorldTransform(prim)
                rel_mat4 = np.array(child_xf * root_xf.GetInverse(), dtype=np.float64)
                rel_mat3 = torch.tensor(rel_mat4[:3, :3].T, dtype=torch.float32)
                rel_pos = torch.tensor(rel_mat4[3, :3], dtype=torch.float32)
                rel_mat3_inv = torch.linalg.inv(rel_mat3.to(torch.float64)).to(torch.float32)

                env_rel_pos.append(rel_pos)
                env_rel_mat.append(rel_mat3)
                env_rel_mat_inv.append(rel_mat3_inv)

                h = hashlib.sha256()
                h.update(np.round(rel_mat4[:3, :3] * 50).astype(np.int64))
                h.update(np.round(rel_mat4[3, :3] * 50).astype(np.int64))
                prim_type = prim.GetTypeName()
                h.update(prim_type.encode("utf-8"))
                if prim_type == "Mesh":
                    verts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float32)
                    h.update(verts.tobytes())
                elif prim_type == "Cube":
                    h.update(np.float32(UsdGeom.Cube(prim).GetSizeAttr().Get()).tobytes())
                elif prim_type == "Sphere":
                    h.update(np.float32(UsdGeom.Sphere(prim).GetRadiusAttr().Get()).tobytes())
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
                env_collider_hashes.append(int.from_bytes(collider_hash[:8], "little", signed=True))

            collider_prim_hashes.extend(env_collider_hashes)
            collider_rel_pos_list.extend(env_rel_pos)
            collider_rel_mat_list.extend(env_rel_mat)
            collider_rel_mat_inv_list.extend(env_rel_mat_inv)

            root_hash_int = int.from_bytes(root_hash.digest()[:8], "little", signed=True)
            root_prim_hashes.append(root_hash_int)

        stor["num_roots"] = num_roots
        stor["collider_prims"] = collider_prims
        stor["collider_prim_hashes"] = torch.tensor(collider_prim_hashes, dtype=torch.int64, device="cpu")
        stor["collider_prim_env_ids"] = torch.tensor(collider_prim_env_ids, dtype=torch.int64, device="cpu")
        stor["collider_rel_pos"] = (
            torch.stack(collider_rel_pos_list).to("cpu") if collider_rel_pos_list else torch.empty(0, 3)
        )
        stor["collider_rel_mat"] = (
            torch.stack(collider_rel_mat_list).to("cpu") if collider_rel_mat_list else torch.empty(0, 3, 3)
        )
        stor["collider_rel_mat_inv"] = (
            torch.stack(collider_rel_mat_inv_list).to("cpu") if collider_rel_mat_inv_list else torch.empty(0, 3, 3)
        )
        stor["root_prim_hashes"] = torch.tensor(root_prim_hashes, dtype=torch.int64, device="cpu")
        stor["root_prim_scales"] = torch.stack(root_prim_scales).to("cpu")

    @property
    def num_root(self) -> int:
        return self.get_val("num_roots")

    @property
    def root_prim_hashes(self) -> torch.Tensor:
        return self.get_val("root_prim_hashes").to(self.device)

    @property
    def root_prim_scales(self) -> torch.Tensor:
        return self.get_val("root_prim_scales").to(self.device)

    @property
    def collider_prim_hashes(self) -> torch.Tensor:
        return self.get_val("collider_prim_hashes").to(self.device)

    @property
    def collider_prims(self) -> list[any]:
        return self.get_val("collider_prims")

    @property
    def collider_prim_env_ids(self) -> torch.Tensor:
        return self.get_val("collider_prim_env_ids").to(self.device)

    @property
    def collider_rel_pos(self) -> torch.Tensor:
        return self.get_val("collider_rel_pos").to(self.device)

    @property
    def collider_rel_mat(self) -> torch.Tensor:
        return self.get_val("collider_rel_mat").to(self.device)

    @property
    def collider_rel_mat_inv(self) -> torch.Tensor:
        return self.get_val("collider_rel_mat_inv").to(self.device)

    def get_val(self, key: str):
        return HASH_STORE.get(self.prim_path_pattern, {}).get(key, None)

    def set_val(self, key: str, val: any):
        if isinstance(val, torch.Tensor):
            val = val.to("cpu")
        HASH_STORE[self.prim_path_pattern][key] = val

    def get_warp_mesh_store(self) -> dict[int, wp.Mesh]:
        return HASH_STORE["warp_mesh_store"]

    def get_hash_store(self) -> dict[int, any]:
        return HASH_STORE
