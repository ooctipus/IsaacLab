# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

newton = pytest.importorskip("newton")

from isaaclab_newton.assets.mpm_object import MPMObject, MPMObjectCfg
from isaaclab_newton.assets.mpm_object.mpm_object import MPMObjectRegistryEntry, _planned_worlds
from isaaclab_newton.physics import MPMSolverCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg, MPMPointsCfg

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import DeformableObjectCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.configclass import configclass


def test_mpm_particle_material_emits_custom_attributes():
    """MPM materials are value cfgs forwarded as Newton custom attributes, not USD material spawners."""
    from isaaclab_newton.sim.spawners.mpm.mpm import _material_custom_attributes

    attrs = _material_custom_attributes(MPMParticleMaterialCfg(viscosity=0.1))

    assert attrs["mpm:friction"] == pytest.approx(0.68)
    assert attrs["mpm:viscosity"] == pytest.approx(0.1)
    assert "density" not in attrs


def test_mpm_object_cfg_resolves_asset_class():
    cfg = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sand",
        spawn=MPMGridCfg(lower=(0.0, 0.0, 0.0), upper=(0.1, 0.1, 0.1), voxel_size=0.1),
    )

    assert cfg.class_type.__name__ == MPMObject.__name__


def test_mpm_grid_emission_records_constant_offsets_per_env():
    builder = newton.ModelBuilder()

    cfg = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sand",
        spawn=MPMGridCfg(
            lower=(0.0, 0.0, 0.0),
            upper=(0.1, 0.1, 0.1),
            voxel_size=0.1,
            particles_per_cell=1.0,
            jitter=0.0,
            particle_placement="cell_center",
        ),
    )
    entry = MPMObjectRegistryEntry(cfg)

    entry.add_to_builder(builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    entry.add_to_builder(builder, 1, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

    assert entry.particles_per_object == 1
    assert entry.particle_offsets == [0, 1]
    assert builder.particle_count == 2


def test_mpm_points_emission_records_constant_offsets_per_env():
    builder = newton.ModelBuilder()

    cfg = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fluid",
        spawn=MPMPointsCfg(
            positions=((0.0, 0.0, 0.0), (0.0, 0.0, 0.1), (0.0, 0.1, 0.0)),
            velocities=((0.0, 0.0, 0.0), (0.0, 0.0, 0.1), (0.0, 0.1, 0.0)),
            mass=0.01,
            radius=0.02,
            material=MPMParticleMaterialCfg(viscosity=0.1, friction=0.0),
        ),
    )
    entry = MPMObjectRegistryEntry(cfg)

    entry.add_to_builder(builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    entry.add_to_builder(builder, 1, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0])

    assert entry.particles_per_object == 3
    assert entry.particle_offsets == [0, 3]
    assert builder.particle_count == 6


def test_mpm_worlds_follow_partial_clone_plan_rows(monkeypatch):
    """Builder worlds use the plan's env ids and omit worlds the asset does not occupy."""
    from isaaclab import cloner
    from isaaclab.sim import SimulationContext

    plan = cloner.ClonePlan(
        sources=("/World/envs/env_4/Sand",),
        destinations=("/World/envs/env_{}/Sand",),
        clone_mask=torch.tensor(((False, True, False, True),)),
        env_ids=torch.tensor((2, 4, 8, 9)),
    )
    monkeypatch.setattr(SimulationContext, "_instance", SimpleNamespace(get_clone_plan=lambda: plan))

    assert _planned_worlds("/World/envs/env_[^/]+/Sand") == (1, 3)


def test_mpm_builder_emits_only_worlds_covered_by_its_plan_rows():
    builder = newton.ModelBuilder()
    cfg = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fluid",
        spawn=MPMPointsCfg(positions=((0.0, 0.0, 0.0),), mass=0.01, radius=0.02),
    )
    entry = MPMObjectRegistryEntry(
        cfg,
        planned_worlds=(1, 3),
    )

    for world in range(4):
        entry.add_to_builder(builder, world, [float(world), 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

    assert entry.particle_offsets == [0, 1]
    assert entry.particles_per_object == 1
    assert builder.particle_count == 2


def test_mixed_deformable_mpm_builder_follows_direct_cfg_plan_order():
    """Reversed asset construction still produces the direct cfg's zero-copy point order."""

    @configclass
    class DirectCfg:
        num_envs: int = 2
        env_spacing: float = 0.5
        cloth: DeformableObjectCfg = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cloth",
            spawn=sim_utils.MeshRectangleCfg(
                size=(0.1, 0.1),
                resolution=(2, 2),
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                physics_material=NewtonSurfaceDeformableBodyMaterialCfg(density=0.02, particle_radius=0.005),
            ),
        )
        media: MPMObjectCfg = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMPointsCfg(positions=((0.0, 0.0, 0.0),), mass=0.01, radius=0.02),
        )

    cfg = DirectCfg()
    sim_cfg = SimulationCfg(physics=MPMSolverCfg(max_iterations=2), device="cuda:0")
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        with cloner.ReplicateSession((cfg.cloth, cfg.media), cfg.num_envs, cfg.env_spacing):
            media = cfg.media.class_type(cfg.media)
            cloth = cfg.cloth.class_type(cfg.cloth)

        layout = sim.get_clone_plan()
        bindings = {binding.path: (binding.source_offset, binding.source_count) for binding in layout.point_bindings()}
        cloth_layouts = tuple(entry for entry in layout.deformables if "/Cloth" in entry.root_path)
        media_layouts = tuple(entry for entry in layout.point_clouds if "/Sand" in entry.path)

        assert [bindings[entry.vis_mesh_path][0] for entry in cloth_layouts] == cloth._registry_entry.particle_offsets
        assert [bindings[entry.path][0] for entry in media_layouts] == media._registry_entry.particle_offsets
        assert sim._physics_manager._newton._builder.particle_count == sum(
            binding.source_count for binding in layout.point_bindings()
        )


def test_mpm_object_initializes_from_interactive_scene():
    @configclass
    class MPMSceneCfg(InteractiveSceneCfg):
        media = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(
                lower=(0.0, 0.0, 0.0),
                upper=(0.1, 0.1, 0.1),
                voxel_size=0.1,
                particle_placement="cell_center",
                visible=False,
            ),
        )

    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=MPMSolverCfg(max_iterations=2, voxel_size=0.05, use_cuda_graph=False),
    )

    scene_cfg = MPMSceneCfg(num_envs=2, env_spacing=1.0)
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = scene_cfg.class_type(scene_cfg)
        sim.reset()

        media = scene["media"]
        assert media.num_instances == 2
        assert media.particles_per_object == 1
        assert media.data.particle_pos_w.torch.shape == (2, 1, 3)
        assert {binding.path for binding in sim.get_clone_plan().point_bindings()} == {
            "/World/envs/env_0/Sand",
            "/World/envs/env_1/Sand",
        }

        default_state = media.data.default_particle_state_w.torch.clone()
        shifted_state = default_state[0:1].clone()
        shifted_state[..., 2] += 0.05

        media.write_particle_state_to_sim_index(
            shifted_state,
            env_ids=torch.tensor([0], device=sim.device, dtype=torch.int32),
        )
        torch.testing.assert_close(media.data.particle_state_w.torch[0:1], shifted_state)

        media.reset(env_ids=[0])
        torch.testing.assert_close(media.data.particle_state_w.torch[0], default_state[0])


def test_mpm_solver_refreshes_kinematic_rigid_body_transforms():
    import isaaclab.sim as sim_utils  # noqa: PLC0415

    @configclass
    class MPMSceneCfg(InteractiveSceneCfg):
        collider = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/KinematicBox",
            spawn=sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1),
                rigid_props=sim_utils.NewtonRigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(collision_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
        )
        media = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(lower=(-0.05, -0.05, 0.3), upper=(0.05, 0.05, 0.4), voxel_size=0.05),
        )

    sim_cfg = SimulationCfg(
        dt=1.0 / 60.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=MPMSolverCfg(max_iterations=2, voxel_size=0.05, use_cuda_graph=False),
    )

    scene_cfg = MPMSceneCfg(num_envs=1, env_spacing=0.0)
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = scene_cfg.class_type(scene_cfg)
        sim.reset()

        collider = scene["collider"]
        angle = 0.5
        root_pose = torch.tensor(
            [[0.1, 0.0, 0.25, 0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle)]],
            dtype=torch.float32,
            device=collider.device,
        )
        collider.write_root_link_pose_to_sim_index(root_pose=root_pose)
        sim.step(render=False)

        manager = sim._physics_manager
        body_labels = list(manager.get_model().body_label)
        body_idx = body_labels.index("/World/envs/env_0/KinematicBox")
        body_q = manager.get_state_0().body_q.numpy()[body_idx]

        np.testing.assert_allclose(body_q, root_pose.detach().cpu().numpy()[0], rtol=1.0e-5, atol=1.0e-6)


def test_mpm_object_binds_plan_owned_points_without_a_visualizer():
    @configclass
    class MPMSceneCfg(InteractiveSceneCfg):
        media = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(
                lower=(0.0, 0.0, 0.0),
                upper=(0.1, 0.1, 0.1),
                voxel_size=0.1,
                visual_color=(0.1, 0.2, 0.3),
            ),
        )

    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=MPMSolverCfg(max_iterations=2, voxel_size=0.05, use_cuda_graph=False),
    )

    scene_cfg = MPMSceneCfg(num_envs=2, env_spacing=1.0)
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = scene_cfg.class_type(scene_cfg)

        from pxr import UsdGeom  # noqa: PLC0415

        assert {binding.path for binding in sim.get_clone_plan().point_bindings()} == {
            "/World/envs/env_0/Sand",
            "/World/envs/env_1/Sand",
        }
        source_points = UsdGeom.Points(scene.stage.GetPrimAtPath("/World/envs/env_0/Sand"))
        assert source_points
        assert len(source_points.GetPointsAttr().Get()) == 8
        sim.reset()

        media = scene["media"]
        manager = sim._physics_manager
        provider = sim.get_scene_data_provider()
        assert provider.request_points(SceneDataFormat.Points).points is manager.get_state_0().particle_q
        bindings = {
            binding.path: (binding.source_offset, binding.source_count)
            for binding in sim.get_clone_plan().point_bindings()
        }

        for env_idx, offset in enumerate(media._recorded_particle_offsets):
            prim_path = f"/World/envs/env_{env_idx}/Sand"
            assert bindings[prim_path] == (offset, media.particles_per_object)


def test_mpm_point_publication_follows_particle_state():
    @configclass
    class MPMSceneCfg(InteractiveSceneCfg):
        media = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(
                lower=(0.0, 0.0, 0.1),
                upper=(0.1, 0.1, 0.2),
                voxel_size=0.05,
                visual_color=(0.1, 0.2, 0.3),
            ),
        )

    sim_cfg = SimulationCfg(
        dt=1.0 / 60.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=MPMSolverCfg(max_iterations=2, voxel_size=0.05, use_cuda_graph=False),
    )

    scene_cfg = MPMSceneCfg(num_envs=1, env_spacing=0.0)
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = scene_cfg.class_type(scene_cfg)
        sim.reset()

        media = scene["media"]
        manager = sim._physics_manager
        provider = sim.get_scene_data_provider()
        binding = sim.get_clone_plan().point_bindings()[0]
        offset, count = binding.source_offset, binding.source_count
        published = provider.request_points(SceneDataFormat.Points)
        assert published.points is manager.get_state_0().particle_q
        generation_before = provider.point_generation()
        points_before = wp.to_torch(published.points)[offset : offset + count].clone()

        for _ in range(3):
            sim.step(render=False)
            scene.update(sim.get_physics_dt())

        published_after = provider.request_points(SceneDataFormat.Points)
        assert published_after is published
        assert provider.point_generation() > generation_before
        points_after = wp.to_torch(published_after.points)[offset : offset + count]
        particle_pos = media.data.particle_pos_w.torch[0]

        assert torch.max(torch.abs(points_after - points_before)) > 0.0
        torch.testing.assert_close(points_after, particle_pos, rtol=1.0e-5, atol=1.0e-6)
