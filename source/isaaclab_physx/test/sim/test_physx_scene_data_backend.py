# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

pytest.importorskip("pxr")
pytest.importorskip("omni.physics.tensors")

import isaaclab_physx
import isaaclab_physx.physics.physx_manager as physx_manager
from isaaclab_physx.cloner import PhysxReplicateContext
from isaaclab_physx.physics import PhysxManager

import isaaclab.sim.utils.stage as stage_utils
from isaaclab.cloner import UsdReplicateContext
from isaaclab.physics import PhysicsManager


def test_manager_registers_clone_resources_by_type(monkeypatch):
    """PhysX declares its USD and native clone resources during manager initialization."""
    stage = object()
    simulation = SimpleNamespace(
        cfg=SimpleNamespace(physics=object(), device="cpu"),
        stage=stage,
        get_or_create_backend=MagicMock(),
        set_setting=MagicMock(),
        add_render_callback=MagicMock(),
    )
    monkeypatch.setattr(PhysicsManager, "_sim", PhysicsManager._sim)
    monkeypatch.setattr(PhysicsManager, "_cfg", PhysicsManager._cfg)
    monkeypatch.setattr(PhysicsManager, "_device", PhysicsManager._device)
    monkeypatch.setattr(PhysicsManager, "_sim_time", PhysicsManager._sim_time)
    monkeypatch.setattr(PhysxManager, "_stage_id", PhysxManager._stage_id)
    monkeypatch.setattr(PhysxManager, "_anim_recorder", PhysxManager._anim_recorder)
    monkeypatch.setattr(PhysxManager, "_scene_data_backend", PhysxManager._scene_data_backend)
    monkeypatch.setattr(isaaclab_physx, "_subscribe_to_simulation_manager_enable", lambda: None)
    monkeypatch.setattr(isaaclab_physx, "_patch_isaacsim_simulation_manager", lambda: None)
    monkeypatch.setattr(stage_utils, "get_current_stage_id", lambda: 1)
    monkeypatch.setattr(PhysxManager, "_setup_subscriptions", classmethod(lambda cls: None))
    monkeypatch.setattr(PhysxManager, "_configure_physics", classmethod(lambda cls: None))
    monkeypatch.setattr(PhysxManager, "_load_fabric", classmethod(lambda cls: None))
    monkeypatch.setattr(physx_manager, "AnimationRecorder", lambda _simulation: object())
    monkeypatch.setattr(physx_manager.omni.kit.app, "get_app", lambda: SimpleNamespace(update=lambda: None))

    PhysxManager.initialize(simulation)

    assert simulation.get_or_create_backend.call_args_list == [
        call(UsdReplicateContext, stage, clone_role="physics"),
        call(PhysxReplicateContext, stage, clone_role="physics"),
    ]
    assert PhysxReplicateContext.clones_whole_env is True


@pytest.mark.parametrize("joint_has_rigid_body_api", [False, True])
def test_rigid_body_view_uses_exact_path_for_joint_name_collision(monkeypatch, joint_has_rigid_body_api):
    """Joint names must keep same-named rigid bodies out of wildcard views."""
    from isaaclab_physx.physics import physx_manager
    from isaaclab_physx.physics.physx_manager import PhysxSceneDataBackend

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    body_prim = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/robot0_forearm").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    unique_body_prim = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/torso").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(unique_body_prim)
    joint_prim = UsdPhysics.FixedJoint.Define(stage, "/World/envs/env_0/Robot/joints/robot0_forearm").GetPrim()
    if joint_has_rigid_body_api:
        UsdPhysics.RigidBodyAPI.Apply(joint_prim)

    captured_paths = []

    class _SimulationView:
        def create_rigid_body_view(self, body_paths):
            captured_paths.extend(body_paths)
            return SimpleNamespace(prim_paths=body_paths)

    monkeypatch.setattr(
        physx_manager.omni.usd,
        "get_context",
        lambda: SimpleNamespace(get_stage=lambda: stage),
    )

    backend = PhysxSceneDataBackend()
    backend.simulation_view = _SimulationView()
    backend.get_rigid_body_view()

    assert captured_paths == [
        "/World/envs/env_*/Robot/torso",
        "/World/envs/env_0/Robot/robot0_forearm",
    ]


def test_discover_deformable_geometry_publishes_discovered_roots(monkeypatch):
    """PhysX deformable views may report child meshes; geometry_paths must be roots."""
    from isaaclab_physx.physics import physx_manager
    from isaaclab_physx.physics.physx_manager import PhysxSceneDataBackend

    from isaaclab.scene_data.deformable_discovery import DeformableStageEntry

    class _FakeDeformableView:
        _backend = object()
        max_simulation_nodes_per_body = 8
        prim_paths = [
            "/World/envs/env_0/Deformable/sim_mesh",
            "/World/envs/env_1/Deformable/sim_mesh",
        ]

    class _SimulationView:
        def create_volume_deformable_body_view(self, patterns):
            return None

        def create_surface_deformable_body_view(self, patterns):
            return _FakeDeformableView()

    monkeypatch.setattr(
        physx_manager.omni.usd,
        "get_context",
        lambda: SimpleNamespace(get_stage=lambda: object()),
    )
    monkeypatch.setattr(
        physx_manager,
        "discover_deformables_on_stage",
        lambda stage: [
            DeformableStageEntry(
                root_path="/World/envs/env_0/Deformable",
                sim_mesh_path="/World/envs/env_0/Deformable/sim_mesh",
                vis_mesh_path="/World/envs/env_0/Deformable/vis_mesh",
                deformable_type="surface",
                vertex_count=4,
                vis_vertex_count=4,
            ),
            DeformableStageEntry(
                root_path="/World/envs/env_1/Deformable",
                sim_mesh_path="/World/envs/env_1/Deformable/sim_mesh",
                vis_mesh_path="/World/envs/env_1/Deformable/vis_mesh",
                deformable_type="surface",
                vertex_count=4,
                vis_vertex_count=4,
            ),
        ],
    )

    backend = PhysxSceneDataBackend()
    backend.simulation_view = _SimulationView()
    backend._discover_deformable_geometry()

    assert backend.geometry_paths == [
        "/World/envs/env_0/Deformable",
        "/World/envs/env_1/Deformable",
    ]
    assert backend.geometry_counts == [4, 4]
