# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import numpy as np
import torch

from isaaclab.cloner import ClonePlan
from isaaclab.cloner.clone_plan import FrameLayout, GeometryLayout, RigidBodyLayout
from isaaclab.sim import SimulationContext

from isaaclab_tasks.contrib.nist.factory_scenes_cfg import FactorySceneBase
from isaaclab_tasks.contrib.nist.utils.rigid_object_hasher import RigidObjectHasher


def test_factory_scene_declares_collision_geometry():
    """NIST collision consumers declare their geometry on the scene composition root."""
    assert FactorySceneBase(env_spacing=2.0).geometry_prim_paths == ("{ENV_REGEX_NS}/[^/]+",)


def test_rigid_object_hasher_consumes_plan_geometry(monkeypatch):
    """Collider identity, ownership, and transforms come entirely from the clone plan."""
    vertices = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32)
    faces = np.asarray(((0, 1, 2),), dtype=np.int32)
    clone_mask = np.ones(2, dtype=np.bool_)
    root = "/World/envs/env_{}/Object"
    body = f"{root}/Body"
    mesh = f"{body}/Mesh"
    root_frame = FrameLayout(root, root.format(0), "/World/envs/env_{}", 0, None, clone_mask=clone_mask)
    mesh_frame = FrameLayout(
        mesh,
        mesh.format(0),
        body,
        0,
        None,
        body_path=body,
        body_view_path="/World/envs/env_*/Object/Body",
        pose=(0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        clone_mask=clone_mask,
    )
    body_prototype = RigidBodyLayout(
        body,
        "/World/envs/env_*/Object/Body",
        0,
        None,
        "Body",
        "/World/envs/env_0/Object/Body",
        clone_mask=clone_mask,
    )
    geometry = GeometryLayout(
        mesh, mesh.format(0), 0, vertices, faces, mesh_frame, collision=True, clone_mask=clone_mask
    )
    plan = ClonePlan(
        sources=(root.format(0),),
        destinations=(root,),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
        env_ids=torch.arange(2),
        is_complete=True,
        frame_prototypes=(root_frame, mesh_frame),
        rigid_body_prototypes=(body_prototype,),
        geometry_prototypes=(geometry,),
        _env_ids_cpu=(0, 1),
    )
    monkeypatch.setattr(
        SimulationContext,
        "_instance",
        SimpleNamespace(get_clone_plan=lambda: plan),
    )

    hasher = RigidObjectHasher(2, r"/World/envs/env_[^/]+/Object")

    assert [geometry.source_path for geometry in hasher.collider_geometries] == [mesh.format(0)] * 2
    assert hasher.collider_keys == ["/World/envs/env_0/Object/Body/Mesh"] * 2
    assert hasher.collider_body_names == ["Body", "Body"]
    torch.testing.assert_close(hasher.collider_rel_pos, torch.tensor(((0.25, 0.0, 0.0),) * 2))
    torch.testing.assert_close(hasher.root_prim_hashes, torch.zeros(2, dtype=torch.int64))
