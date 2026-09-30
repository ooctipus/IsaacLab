# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import warp as wp

pytest.importorskip("pxr")
pytest.importorskip("omni.physics.tensors")

from isaaclab.cloner import ClonePlan
from isaaclab.cloner.clone_plan import DeformableLayout, RigidBodyLayout
from isaaclab.scene_data import SceneDataFormat, SceneDataProvider


def _completed_plan(env_ids: int | tuple[int, ...] = 0, **topology) -> ClonePlan:
    env_ids = tuple(range(env_ids)) if isinstance(env_ids, int) else env_ids
    return ClonePlan(
        sources=(),
        destinations=(),
        clone_mask=torch.zeros((0, len(env_ids)), dtype=torch.bool),
        env_ids=torch.tensor(env_ids, dtype=torch.long),
        is_complete=True,
        _env_ids_cpu=env_ids,
        **topology,
    )


def test_manager_registers_clone_resources_by_type(monkeypatch):
    """PhysX registers each simulation-owned clone resource by backend type."""
    import isaaclab_physx
    import isaaclab_physx.physics.physx_manager as physx_manager
    from isaaclab_physx.cloner import PhysxReplicateContext
    from isaaclab_physx.physics import PhysxCfg, PhysxManager

    from isaaclab.cloner import UsdReplicateContext
    from isaaclab.physics import PhysicsManager

    cfg = PhysxCfg()
    manager = object.__new__(PhysxManager)
    PhysicsManager.__init__(manager, cfg)
    stage = object()
    backend_calls = []
    simulation = SimpleNamespace(
        cfg=SimpleNamespace(device="cpu"),
        stage=stage,
        get_or_create_backend=lambda backend_type, *args, **kwargs: backend_calls.append((backend_type, args, kwargs)),
        set_setting=MagicMock(),
    )
    monkeypatch.setattr(isaaclab_physx, "_subscribe_to_simulation_manager_enable", lambda: None)
    monkeypatch.setattr(isaaclab_physx, "_patch_isaacsim_simulation_manager", lambda: None)
    monkeypatch.setattr(manager, "_setup_subscriptions", lambda: None)
    monkeypatch.setattr(manager, "_configure_physics", lambda: None)
    monkeypatch.setattr(physx_manager, "AnimationRecorder", lambda _simulation: object())
    monkeypatch.setattr(physx_manager.omni.kit.app, "get_app", lambda: SimpleNamespace(update=lambda: None))

    manager._bind_context(simulation)

    assert backend_calls == [
        (
            UsdReplicateContext,
            (stage,),
            {"clone_role": "physics"},
        ),
        (
            PhysxReplicateContext,
            (stage,),
            {"clone_role": "physics"},
        ),
    ]


def test_publish_is_passive():
    from isaaclab_physx.physics.physx_manager import PhysxSceneDataBackend

    class _SimulationView:
        def update_articulations_kinematic(self):
            raise AssertionError("SDP publication must not advance physics")

    backend = PhysxSceneDataBackend("cpu")
    view = _SimulationView()
    backend.setup(view, _completed_plan())

    backend.publish(points=False)
    assert backend.transform_publication.data.transforms is None
    assert backend.transform_publication.dirty


def test_setup_uses_exact_declared_rigid_order():
    """Exact plan paths preserve SDP order and cannot overmatch same-named joint prims."""
    from isaaclab_physx.physics.physx_manager import PhysxSceneDataBackend

    plan = _completed_plan(
        (0, 2),
        rigid_body_prototypes=(
            RigidBodyLayout(
                "/World/envs/env_{}/Robot/base",
                "/World/envs/env_*/Robot/base",
                0,
                None,
                "base",
                clone_mask=np.ones(2, dtype=np.bool_),
            ),
            RigidBodyLayout("/World/Global", "/World/Global", 1, None, "Global"),
        ),
    )
    declared = tuple(plan.iter_rigid_body_paths())
    captured: list[str] = []

    class _SimulationView:
        def __init__(self, paths):
            self.paths = paths
            self.transforms = wp.zeros(len(paths), dtype=wp.transformf, device="cpu")
            self.get_transforms = MagicMock(return_value=self.transforms)

        def create_rigid_body_view(self, paths):
            captured.extend(paths)
            return SimpleNamespace(
                prim_paths=self.paths,
                count=len(self.paths),
                get_transforms=self.get_transforms,
            )

    backend = PhysxSceneDataBackend("cpu")
    simulation_view = _SimulationView(list(declared))
    backend.setup(simulation_view, plan)
    published_transforms = backend.transform_publication.data.transforms

    assert captured == list(declared)
    assert simulation_view.get_transforms.call_count == 1
    backend.publish(points=False)
    assert backend.transform_publication.data.transforms is published_transforms
    assert simulation_view.get_transforms.call_count == 2

    with pytest.raises(RuntimeError, match="did not preserve"):
        backend.setup(_SimulationView(list(reversed(declared))), plan)


def test_setup_declares_native_deformable_bindings_and_sdp_gathers_on_request(monkeypatch):
    from isaaclab_physx.physics.physx_manager import PhysxSceneDataBackend

    entries = (
        DeformableLayout(
            root_path="/World/envs/env_0/Cloth",
            sim_mesh_path="/World/envs/env_0/Cloth/sim",
            vis_mesh_path="/World/envs/env_0/Cloth/visual",
            view_path="/World/envs/env_*/Cloth",
            deformable_type="surface",
            vertex_count=4,
            vis_vertex_count=4,
            point_indices=None,
            point_weights=None,
            row=0,
            env_id=0,
        ),
        DeformableLayout(
            root_path="/World/envs/env_1/Soft",
            sim_mesh_path="/World/envs/env_1/Soft/sim",
            vis_mesh_path="/World/envs/env_1/Soft/visual",
            view_path="/World/envs/env_*/Soft",
            deformable_type="volume",
            vertex_count=3,
            vis_vertex_count=3,
            point_indices=None,
            point_weights=None,
            row=1,
            env_id=1,
        ),
        DeformableLayout(
            root_path="/World/envs/env_2/Cloth",
            sim_mesh_path="/World/envs/env_2/Cloth/sim",
            vis_mesh_path="/World/envs/env_2/Cloth/visual",
            view_path="/World/envs/env_*/Cloth",
            deformable_type="surface",
            vertex_count=5,
            vis_vertex_count=5,
            point_indices=None,
            point_weights=None,
            row=0,
            env_id=2,
        ),
    )

    class _FakeDeformableView:
        _backend = object()
        count = 2
        max_simulation_nodes_per_body = 8
        prim_paths = [entries[2].sim_mesh_path, entries[0].root_path]

        def __init__(self):
            self.reads = 0
            self.positions = wp.array(
                [
                    [[20.0 + index, 0.0, 0.0] for index in range(8)],
                    [[10.0 + index, 0.0, 0.0] for index in range(8)],
                ],
                dtype=wp.vec3f,
                device="cpu",
            )

        def get_simulation_nodal_positions(self):
            self.reads += 1
            return self.positions

    deformable_view = _FakeDeformableView()

    class _FakeVolumeView:
        _backend = object()
        count = 1
        max_simulation_nodes_per_body = 4
        prim_paths = [entries[1].sim_mesh_path]

        def __init__(self):
            self.positions = wp.array([[[30.0 + index, 0.0, 0.0] for index in range(4)]], dtype=wp.vec3f, device="cpu")

        def get_simulation_nodal_positions(self):
            return self.positions

    class _SimulationView:
        def create_volume_deformable_body_view(self, patterns):
            assert patterns == ["/World/envs/env_*/Soft"]
            return _FakeVolumeView()

        def create_surface_deformable_body_view(self, patterns):
            assert patterns == ["/World/envs/env_*/Cloth"]
            return deformable_view

    plan = _completed_plan(3, deformables=entries)
    backend = PhysxSceneDataBackend("cpu")
    backend.setup(_SimulationView(), plan)
    publication = backend.point_publications["points"]
    provider = SceneDataProvider(backend)
    provider._bind_point_plan(plan)
    launches = []
    launch = wp.launch

    def traced_launch(kernel, *args, **kwargs):
        launches.append(kernel.key)
        return launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", traced_launch)

    published_points = publication.data.points
    assert len(published_points) == 2
    assert deformable_view.reads == 1

    backend.publish(transforms=False)
    assert publication.data.points is published_points
    assert deformable_view.reads == 2
    assert launches == []
    assert provider.request_points(SceneDataFormat.BodyPoints) is publication.data
    assert launches == []
    points = provider.request_points(SceneDataFormat.Points).points.numpy()
    assert len(launches) == 2
    assert points[:, 0].tolist() == [
        10.0,
        11.0,
        12.0,
        13.0,
        30.0,
        31.0,
        32.0,
        20.0,
        21.0,
        22.0,
        23.0,
        24.0,
    ]
    assert deformable_view.reads == 1


def test_setup_rejects_deformable_count_larger_than_native_padding():
    from isaaclab_physx.physics.physx_manager import PhysxSceneDataBackend

    entry = DeformableLayout("/Cloth", "/Cloth/sim", "/Cloth/vis", "/Cloth", "surface", 9, 9, None, None, 0, None)
    view = SimpleNamespace(_backend=object(), count=1, max_simulation_nodes_per_body=8, prim_paths=[entry.root_path])
    simulation_view = SimpleNamespace(create_surface_deformable_body_view=lambda patterns: view)

    with pytest.raises(RuntimeError, match="declares 9 nodes"):
        PhysxSceneDataBackend("cpu").setup(simulation_view, _completed_plan(deformables=(entry,)))
