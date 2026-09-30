# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton replication builder ownership."""

import importlib
from types import SimpleNamespace

import newton
import numpy as np
import pytest
from isaaclab_newton.cloner import copy_newton_clone_source, newton_builder_world_hook, newton_physics_replicate

from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

from isaaclab.cloner import ClonePlan
from isaaclab.sim import SimulationContext

replicate_module = importlib.import_module("isaaclab_newton.cloner.replicate")
clone_utils_module = importlib.import_module("isaaclab_newton.cloner.newton_clone_utils")


def test_source_builders_import_each_source_once(monkeypatch):
    """Repeated plan rows must share one imported Newton source builder."""
    imported = []

    def build(_stage, source, *_args):
        imported.append(source)
        return object()

    monkeypatch.setattr(clone_utils_module, "_build_source_builder", build)
    builders = clone_utils_module.build_source_builders(
        object(), ("/World/Ground", "/World/Robot", "/World/Ground"), object, ()
    )

    assert imported == ["/World/Ground", "/World/Robot"]
    assert tuple(builders) == ("/World/Ground", "/World/Robot")


def test_newton_builder_world_hook_owns_one_registration(monkeypatch):
    """The scope rejects duplicates and preserves unrelated hooks during cleanup."""

    def existing(*_args):
        pass

    def temporary(*_args):
        pass

    def added_later(*_args):
        pass

    resource = SimpleNamespace(_per_world_builder_hooks=[existing])
    hooks = resource._per_world_builder_hooks

    with pytest.raises(ValueError, match="stop"):
        with newton_builder_world_hook(resource, temporary):
            assert hooks == [existing, temporary]
            with pytest.raises(RuntimeError, match="already registered"):
                with newton_builder_world_hook(resource, temporary):
                    pass
            assert hooks == [existing, temporary]
            hooks.append(added_later)
            raise ValueError("stop")

    assert hooks == [existing, added_later]

    with pytest.raises(RuntimeError, match="already registered"):
        with newton_builder_world_hook(resource, existing):
            pass
    assert hooks == [existing, added_later]


def test_copy_newton_clone_source_owns_mutable_geometry(monkeypatch):
    """Finalizing a copied prototype must not mutate cloner-retained shape sources."""
    source = newton.ModelBuilder()
    body = source.add_body()
    mesh = newton.Mesh(vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], indices=[0, 1, 2])
    source.add_shape_mesh(body, mesh=mesh)
    resource = SimpleNamespace(_cl_protos={"/World/Source": source})

    copied = copy_newton_clone_source(resource, "/World/Source")

    assert copied.shape_source[0] is not source.shape_source[0]


def test_explicit_global_import_uses_global_world():
    """Declared global colliders remain in Newton world -1 after model finalization."""
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    UsdGeom.Xform.Define(stage, "/World")
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    UsdLux.DistantLight.Define(stage, "/World/Light")
    global_paths = ("/World/Ground", "/World/Light")

    plan = ClonePlan(
        sources=global_paths,
        destinations=global_paths,
        clone_mask=np.zeros((len(global_paths), 2), dtype=np.bool_),
        env_ids=np.arange(2, dtype=np.int64),
        positions=np.zeros((2, 3), dtype=np.float32),
        is_complete=True,
        _env_ids_cpu=(0, 1),
    )
    sim = SimpleNamespace(stage=stage, cfg=SimpleNamespace(device="cpu"), get_clone_plan=lambda: plan)
    resource = replicate_module.NewtonReplicateContext(sim)
    builder, *_ = replicate_module._build_newton_builder_from_mapping(
        resource=resource,
        plan=plan,
        stage=stage,
        sources=(),
        destinations=(),
        env_ids=plan.env_ids,
        mapping=np.empty((0, 2), dtype=np.bool_),
        positions=plan.positions,
        load_visual_shapes=False,
    )
    model = builder.finalize("cpu")
    ground_index = model.shape_label.index("/World/Ground")
    assert model.shape_world.numpy()[ground_index] == -1
    assert model.world_count == 2
    assert "/World/Light" not in model.shape_label  # USD lights are not Newton physics entities.


def test_homogeneous_asset_rows_replicate_as_one_environment(monkeypatch):
    """A flat homogeneous plan keeps asset ownership without copying each builder per world."""
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    for name in ("BoxA", "BoxB"):
        body = UsdGeom.Xform.Define(stage, f"/World/envs/env_0/{name}")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        collision = UsdGeom.Cube.Define(stage, f"/World/envs/env_0/{name}/Collision")
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim())

    sources = ("/World/envs/env_0/BoxA", "/World/envs/env_0/BoxB")
    plan = ClonePlan(
        sources=sources,
        destinations=("/World/envs/env_{}/BoxA", "/World/envs/env_{}/BoxB"),
        clone_mask=np.ones((2, 2), dtype=np.bool_),
        env_ids=np.arange(2, dtype=np.int64),
        positions=np.zeros((2, 3), dtype=np.float32),
        is_complete=True,
        _env_ids_cpu=(0, 1),
    )
    sim = SimpleNamespace(stage=stage, cfg=SimpleNamespace(device="cpu"), get_clone_plan=lambda: plan)
    resource = replicate_module.NewtonReplicateContext(sim)
    original = replicate_module.replicate_builder_mapping
    replicated_sources = None
    routing_keys = None

    def capture_sources(**kwargs):
        nonlocal replicated_sources, routing_keys
        replicated_sources = kwargs["sources"]
        routing_keys = tuple(kwargs["source_builders"])
        return original(**kwargs)

    monkeypatch.setattr(replicate_module, "replicate_builder_mapping", capture_sources)
    builder, _, retained = replicate_module._build_newton_builder_from_mapping(
        resource=resource,
        plan=plan,
        stage=stage,
        sources=sources,
        destinations=plan.destinations,
        env_ids=plan.env_ids,
        mapping=plan.clone_mask,
        positions=plan.positions,
        load_visual_shapes=False,
    )

    assert replicated_sources == ("/World/envs/env_0",)
    assert routing_keys == replicated_sources
    assert tuple(retained) == sources
    assert builder.world_count == 2


def test_renderer_only_row_stays_out_of_newton_replication(monkeypatch):
    """A plan-owned camera must not become empty native work in every Newton world."""
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    body_path = "/World/envs/env_0/Box"
    body = UsdGeom.Xform.Define(stage, body_path)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    collision = UsdGeom.Cube.Define(stage, f"{body_path}/Collision")
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    camera_path = "/World/envs/env_0/Camera"
    UsdGeom.Camera.Define(stage, camera_path)

    sources = (body_path, camera_path)
    plan = ClonePlan(
        sources=sources,
        destinations=("/World/envs/env_{}/Box", "/World/envs/env_{}/Camera"),
        clone_mask=np.ones((2, 2), dtype=np.bool_),
        env_ids=np.arange(2, dtype=np.int64),
        positions=np.zeros((2, 3), dtype=np.float32),
        is_complete=True,
        _env_ids_cpu=(0, 1),
    )
    sim = SimpleNamespace(stage=stage, cfg=SimpleNamespace(device="cpu"), get_clone_plan=lambda: plan)
    resource = replicate_module.NewtonReplicateContext(sim)
    original = replicate_module.replicate_builder_mapping
    replicated_sources = None

    def capture_sources(**kwargs):
        nonlocal replicated_sources
        replicated_sources = kwargs["sources"]
        return original(**kwargs)

    monkeypatch.setattr(replicate_module, "replicate_builder_mapping", capture_sources)
    builder, _, retained = replicate_module._build_newton_builder_from_mapping(
        resource, plan, stage, sources, plan.destinations, plan.env_ids, plan.clone_mask, plan.positions
    )

    assert replicated_sources == (body_path,)
    assert tuple(retained) == sources
    assert builder.world_count == 2


def test_cached_nested_rows_preserve_environment_root():
    """Cached nested rows must coexist with an independently replicated environment root."""
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    for env_id in range(2):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}")
    paths = (
        "/World/envs/env_0/RootBody",
        "/World/envs/env_0/VariantA",
        "/World/envs/env_1/VariantB",
        "/World/envs/env_0/SharedA",
        "/World/envs/env_0/SharedB",
    )
    for path in paths:
        body = UsdGeom.Xform.Define(stage, path)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

    sources = ("/World/envs/env_0", *paths[1:])
    plan = ClonePlan(
        sources=sources,
        destinations=(
            "/World/envs/env_{}",
            "/World/envs/env_{}/Variant",
            "/World/envs/env_{}/Variant",
            "/World/envs/env_{}/SharedA",
            "/World/envs/env_{}/SharedB",
        ),
        clone_mask=np.asarray(((1, 1), (1, 0), (0, 1), (1, 1), (1, 1)), dtype=np.bool_),
        env_ids=np.arange(2, dtype=np.int64),
        positions=np.zeros((2, 3), dtype=np.float32),
        is_complete=True,
        _env_ids_cpu=(0, 1),
    )
    sim = SimpleNamespace(stage=stage, cfg=SimpleNamespace(device="cpu"), get_clone_plan=lambda: plan)
    resource = replicate_module.NewtonReplicateContext(sim)
    builder, *_ = replicate_module._build_newton_builder_from_mapping(
        resource, plan, stage, sources, plan.destinations, plan.env_ids, plan.clone_mask, plan.positions
    )

    assert builder.body_label == [
        "/World/envs/env_0/RootBody",
        "/World/envs/env_0/Variant",
        "/World/envs/env_0/SharedA",
        "/World/envs/env_0/SharedB",
        "/World/envs/env_1/RootBody",
        "/World/envs/env_1/Variant",
        "/World/envs/env_1/SharedA",
        "/World/envs/env_1/SharedB",
    ]


def test_raw_newton_replicate_uses_numpy_mapping(monkeypatch):
    """The public direct entry point builds through the simulation-owned Newton resource."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    body = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Box")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    collision = UsdGeom.Cube.Define(stage, "/World/envs/env_0/Box/Collision")
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())

    class Simulation:
        cfg = SimpleNamespace(device="cpu")

        def __init__(self):
            self.resource = None

        def get_or_create_backend(self, backend_type, *args):
            if self.resource is None:
                self.resource = backend_type(*args)
            return self.resource

    sim = Simulation()
    monkeypatch.setattr(SimulationContext, "instance", staticmethod(lambda: sim))
    builder, metadata = newton_physics_replicate(
        stage,
        ("/World/envs/env_0/Box",),
        ("/World/envs/env_{}/Box",),
        np.arange(2, dtype=np.int64),
        mapping=np.ones((1, 2), dtype=np.bool_),
        positions=np.zeros((2, 3), dtype=np.float32),
    )

    assert builder.world_count == 2
    assert metadata == {}
