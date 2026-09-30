# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clone-plan scene-layout declaration tests."""

from __future__ import annotations

import numpy as np
import pytest

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.cloner import ClonePlan
from isaaclab.cloner.scene_layout import declare_scene_layout


def _add_api_schemas(prim: Usd.Prim, schemas: list[str]) -> None:
    api_schemas = Sdf.TokenListOp()
    api_schemas.explicitItems = schemas
    prim.SetMetadata("apiSchemas", api_schemas)


def _surface(stage: Usd.Stage, root_path: str, count: int) -> None:
    mesh = UsdGeom.Mesh.Define(stage, root_path)
    _add_api_schemas(mesh.GetPrim(), ["OmniPhysicsDeformableBodyAPI", "OmniPhysicsSurfaceDeformableSimAPI"])
    mesh.CreatePointsAttr([Gf.Vec3f(float(index), 0.0, 0.0) for index in range(count)])


def _volume(stage: Usd.Stage, root_path: str, body_schema: str = "OmniPhysicsDeformableBodyAPI") -> None:
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    _add_api_schemas(root, [body_schema])
    tet = UsdGeom.TetMesh.Define(stage, f"{root_path}/simulation")
    _add_api_schemas(tet.GetPrim(), ["OmniPhysicsVolumeDeformableSimAPI"])
    points = [
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(1.0, 0.0, 0.0),
        Gf.Vec3f(0.0, 1.0, 0.0),
        Gf.Vec3f(0.0, 0.0, 1.0),
    ]
    tet.CreatePointsAttr(points)
    tet.CreateTetVertexIndicesAttr([Gf.Vec4i(0, 1, 2, 3)])
    visual = UsdGeom.Mesh.Define(stage, f"{root_path}/visual")
    visual.CreatePointsAttr(points)


def _cable(stage: Usd.Stage, path: str, count: int = 4) -> UsdGeom.BasisCurves:
    curve = UsdGeom.BasisCurves.Define(stage, path)
    _add_api_schemas(curve.GetPrim(), ["PhysicsCurvesDeformableSimAPI"])
    curve.CreatePointsAttr([Gf.Vec3f(0.0, float(index), 0.0) for index in range(count)])
    curve.CreateCurveVertexCountsAttr([count])
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    return curve


def _heterogeneous_plan() -> ClonePlan:
    return ClonePlan(
        sources=("/World/envs/env_0/Variant", "/World/envs/env_1/Variant", "/World/Global"),
        destinations=("/World/envs/env_{}/Variant", "/World/envs/env_{}/Variant", "/World/Global"),
        clone_mask=np.asarray([[True, False, True, False], [False, True, False, True], [False, False, False, False]]),
        env_ids=np.arange(4),
    )


def test_declares_exact_heterogeneous_partial_and_global_layout_from_prototypes() -> None:
    stage = Usd.Stage.CreateInMemory()
    for path in ("/World/envs/env_0/Variant/A", "/World/envs/env_1/Variant/B", "/World/Global/body"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, path).GetPrim())
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/envs/env_0/Variant/joint").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(joint)
    _surface(stage, "/World/envs/env_0/Variant/Cloth", 3)
    _volume(stage, "/World/envs/env_1/Variant/Soft")
    _cable(stage, "/World/envs/env_0/Variant/Cable")

    declared = declare_scene_layout(_heterogeneous_plan(), stage)
    layout = declared
    assert layout.is_complete
    frames = layout.match_frames(r"/World/envs/env_[^/]+/Variant/[AB]")
    assert [(entry.path, entry.source_path, entry.parent_path, entry.row, entry.env_id) for entry in frames] == [
        ("/World/envs/env_0/Variant/A", "/World/envs/env_0/Variant/A", "/World/envs/env_0/Variant", 0, 0),
        ("/World/envs/env_1/Variant/B", "/World/envs/env_1/Variant/B", "/World/envs/env_1/Variant", 1, 1),
        ("/World/envs/env_2/Variant/A", "/World/envs/env_0/Variant/A", "/World/envs/env_2/Variant", 0, 2),
        ("/World/envs/env_3/Variant/B", "/World/envs/env_1/Variant/B", "/World/envs/env_3/Variant", 1, 3),
    ]
    assert layout.match_frames("/World/Global/body")[0].env_id is None
    assert tuple(layout.iter_rigid_body_paths()) == (
        "/World/Global/body",
        "/World/envs/env_0/Variant/A",
        "/World/envs/env_1/Variant/B",
        "/World/envs/env_2/Variant/A",
        "/World/envs/env_3/Variant/B",
    )
    assert [(entry.path, entry.row, entry.env_id, entry.view_path) for entry in layout.rigid_body_prototypes] == [
        ("/World/envs/env_{}/Variant/A", 0, None, "/World/envs/env_*/Variant/A"),
        ("/World/envs/env_{}/Variant/B", 1, None, "/World/envs/env_*/Variant/B"),
        ("/World/Global/body", 2, None, "/World/Global/body"),
    ]
    with pytest.raises(ValueError, match="incompatible clone-plan body paths"):
        layout.match_rigid_body(r"/World/envs/env_[^/]+/Variant/[AB]")
    deformables = [
        (entry.root_path, entry.deformable_type, entry.vertex_count, entry.row, entry.env_id)
        for entry in layout.deformables
    ]
    assert deformables == [
        ("/World/envs/env_0/Variant/Cloth", "surface", 3, 0, 0),
        ("/World/envs/env_1/Variant/Soft", "volume", 4, 1, 1),
        ("/World/envs/env_2/Variant/Cloth", "surface", 3, 0, 2),
        ("/World/envs/env_3/Variant/Soft", "volume", 4, 1, 3),
    ]
    assert layout.deformables[-1].sim_mesh_path == "/World/envs/env_3/Variant/Soft/simulation"
    assert layout.deformables[-1].vis_mesh_path == "/World/envs/env_3/Variant/Soft/visual"
    assert layout.deformables[-1].indices.shape == (1, 4)
    assert [(entry.path, entry.segment_count, entry.row, entry.env_id) for entry in layout.cables] == [
        ("/World/envs/env_0/Variant/Cable", 3, 0, 0),
        ("/World/envs/env_2/Variant/Cable", 3, 0, 2),
    ]
    assert {entry.view_path for entry in layout.cables} == {"/World/envs/env_*/Variant/Cable"}
    assert not stage.GetPrimAtPath("/World/envs/env_2/Variant").IsValid()


def test_native_deformable_view_matching_is_exact_and_preserves_native_order() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Global")
    _surface(stage, "/World/envs/env_0/Variant/Cloth", 3)
    _volume(stage, "/World/envs/env_1/Variant/Soft")
    layout = declare_scene_layout(_heterogeneous_plan(), stage)
    assert layout.is_complete

    surfaces = tuple(entry for entry in layout.deformables if entry.deformable_type == "surface")
    surface_paths = [surfaces[1].sim_mesh_path, surfaces[0].root_path]
    assert [entry.root_path for entry in layout.match_deformables("surface", surface_paths)] == [
        "/World/envs/env_2/Variant/Cloth",
        "/World/envs/env_0/Variant/Cloth",
    ]

    with pytest.raises(ValueError, match="undeclared path"):
        layout.match_deformables("surface", ["/World/envs/env_9/Variant/Cloth"])


def test_layout_declares_newton_deformable_body_schema() -> None:
    stage = Usd.Stage.CreateInMemory()
    _volume(stage, "/World/envs/env_0/Soft", "PhysicsDeformableBodyAPI")
    plan = ClonePlan(
        sources=("/World/envs/env_0/Soft",),
        destinations=("/World/envs/env_{}/Soft",),
        clone_mask=np.ones((1, 1), dtype=np.bool_),
        env_ids=np.arange(1),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete
    assert [entry.root_path for entry in layout.deformables] == ["/World/envs/env_0/Soft"]


def test_global_heightfield_geometry_is_declared_without_a_sensor_request() -> None:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/Ground")
    root.AddTranslateOp().Set((2.0, 3.0, 4.0))
    root.GetPrim().CreateAttribute("newton:heightfield:resolution", Sdf.ValueTypeNames.Float).Set(0.25)
    mesh = UsdGeom.Mesh.Define(stage, "/World/Ground/mesh")
    mesh.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 1, 3])
    plan = ClonePlan(
        sources=("/World/Ground",),
        destinations=("/World/Ground",),
        clone_mask=np.zeros((1, 2), dtype=np.bool_),
        env_ids=np.arange(2),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete and len(layout.geometry_prototypes) == 1
    geometry = layout.geometry_prototypes[0]
    assert geometry.heightfield == ("/World/Ground", 0.25)
    np.testing.assert_allclose(geometry.vertices, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
    assert geometry.faces.tolist() == [[0, 1, 2], [0, 1, 3]]
    assert geometry.frame.pose[:3] == pytest.approx((2.0, 3.0, 4.0))


def test_global_heightfield_geometry_can_be_owned_by_a_nested_plan_row() -> None:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/Ground")
    root.GetPrim().CreateAttribute("newton:heightfield:resolution", Sdf.ValueTypeNames.Float).Set(0.25)
    mesh = UsdGeom.Mesh.Define(stage, "/World/Ground/mesh")
    mesh.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    plan = ClonePlan(
        sources=("/World/Ground", "/World/Ground/mesh"),
        destinations=("/World/Ground", "/World/Ground/mesh"),
        clone_mask=np.zeros((2, 1), dtype=np.bool_),
        env_ids=np.arange(1),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete and len(layout.geometry_prototypes) == 1
    geometry = layout.geometry_prototypes[0]
    assert geometry.row == 1
    assert geometry.source_path == "/World/Ground/mesh"
    assert geometry.heightfield == ("/World/Ground", 0.25)


def test_layout_declaration_fails_for_missing_or_duplicate_plan_ownership() -> None:
    stage = Usd.Stage.CreateInMemory()
    missing = ClonePlan(
        sources=("/World/envs/env_0/Missing",),
        destinations=("/World/envs/env_{}/Missing",),
        clone_mask=np.ones((1, 1), dtype=np.bool_),
        env_ids=np.arange(1),
    )
    with pytest.raises(ValueError, match="source prim is absent"):
        declare_scene_layout(missing, stage)

    body = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    duplicate = ClonePlan(
        sources=("/World/envs/env_0/Body", "/World/envs/env_0/Body"),
        destinations=("/World/envs/env_{}/Body", "/World/envs/env_{}/Body"),
        clone_mask=np.ones((2, 1), dtype=np.bool_),
        env_ids=np.arange(1),
    )
    with pytest.raises(ValueError, match="multiple rows"):
        declare_scene_layout(duplicate, stage)


def test_layout_rejects_unsupported_cable_topology() -> None:
    stage = Usd.Stage.CreateInMemory()
    cable = _cable(stage, "/World/Cable")
    cable.GetWrapAttr().Set(UsdGeom.Tokens.periodic)
    plan = ClonePlan(
        sources=("/World/Cable",),
        destinations=("/World/Cable",),
        clone_mask=np.zeros((1, 1), dtype=np.bool_),
        env_ids=np.arange(1),
    )

    with pytest.raises(ValueError, match="open linear curve"):
        declare_scene_layout(plan, stage)


def test_cable_matching_requires_one_planned_topology() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Cable")
    _cable(stage, "/World/envs/env_0/Cable/geometry/mesh")
    plan = ClonePlan(
        sources=("/World/envs/env_0/Cable",),
        destinations=("/World/envs/env_{}/Cable",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete
    cables = layout.match_cables(r"/World/envs/env_[^/]+/Cable")
    assert [entry.env_id for entry in cables] == [0, 1]
    assert {entry.view_path for entry in cables} == {"/World/envs/env_*/Cable/geometry/mesh"}


def test_point_streams_merge_deformables_point_clouds_and_cables_in_plan_order() -> None:
    """Authored Points need no cfg hook and every stream follows clone-plan environment columns."""
    stage = Usd.Stage.CreateInMemory()
    _surface(stage, "/World/envs/env_4/A/Cloth", 3)
    _cable(stage, "/World/envs/env_4/A/Cable")
    sand = UsdGeom.Points.Define(stage, "/World/envs/env_4/A/Sand")
    sand.CreatePointsAttr([Gf.Vec3f(), Gf.Vec3f(0.0, 0.0, 1.0)])
    _volume(stage, "/World/envs/env_2/B/Soft")
    spray = UsdGeom.Points.Define(stage, "/World/envs/env_2/B/Spray")
    spray.CreatePointsAttr([Gf.Vec3f()])
    plan = ClonePlan(
        sources=("/World/envs/env_4/A", "/World/envs/env_2/B"),
        destinations=("/World/envs/env_{}/A", "/World/envs/env_{}/B"),
        clone_mask=np.asarray(((True, False, True, False), (False, True, False, True))),
        env_ids=np.asarray((4, 2, 9, 7)),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete
    assert [(entry.path, entry.count, entry.row, entry.env_id) for entry in layout.point_clouds] == [
        ("/World/envs/env_4/A/Sand", 2, 0, 4),
        ("/World/envs/env_2/B/Spray", 1, 1, 2),
        ("/World/envs/env_9/A/Sand", 2, 0, 9),
        ("/World/envs/env_7/B/Spray", 1, 1, 7),
    ]
    assert layout.point_stream_names == ("points", "cables")
    assert [
        (binding.path, binding.source_offset, binding.source_count, binding.output_offset, binding.output_count)
        for binding in layout.point_bindings()
    ] == [
        ("/World/envs/env_4/A/Cloth", 0, 3, 0, 3),
        ("/World/envs/env_4/A/Sand", 3, 2, 3, 2),
        ("/World/envs/env_2/B/Soft/visual", 5, 4, 5, 4),
        ("/World/envs/env_2/B/Spray", 9, 1, 9, 1),
        ("/World/envs/env_9/A/Cloth", 10, 3, 10, 3),
        ("/World/envs/env_9/A/Sand", 13, 2, 13, 2),
        ("/World/envs/env_7/B/Soft/visual", 15, 4, 15, 4),
        ("/World/envs/env_7/B/Spray", 19, 1, 19, 1),
    ]
    assert [
        (binding.path, binding.source_offset, binding.source_count, binding.output_offset, binding.output_count)
        for binding in layout.point_bindings("cables")
    ] == [
        ("/World/envs/env_4/A/Cable", 0, 4, 0, 4),
        ("/World/envs/env_9/A/Cable", 4, 4, 4, 4),
    ]


def test_volume_visual_vertices_map_from_plan_owned_simulation_tetrahedra() -> None:
    stage = Usd.Stage.CreateInMemory()
    _volume(stage, "/World/envs/env_0/Soft")
    visual = UsdGeom.Mesh(stage.GetPrimAtPath("/World/envs/env_0/Soft/visual"))
    visual.GetPointsAttr().Set([Gf.Vec3f(0.25, 0.25, 0.25)])
    plan = ClonePlan(
        sources=("/World/envs/env_0/Soft",),
        destinations=("/World/envs/env_{}/Soft",),
        clone_mask=np.ones((1, 1), dtype=np.bool_),
        env_ids=np.arange(1),
    )

    binding = declare_scene_layout(plan, stage).point_bindings()[0]

    assert (binding.source_count, binding.output_count) == (4, 1)
    assert binding.source_indices.tolist() == [[0, 1, 2, 3]]
    np.testing.assert_allclose(binding.weights, [[0.25, 0.25, 0.25, 0.25]])


def test_surface_visual_vertices_map_from_plan_owned_simulation_triangles() -> None:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Cloth").GetPrim()
    _add_api_schemas(root, ["OmniPhysicsDeformableBodyAPI"])
    simulation = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Cloth/simulation")
    _add_api_schemas(simulation.GetPrim(), ["OmniPhysicsSurfaceDeformableSimAPI"])
    simulation.CreatePointsAttr([Gf.Vec3f(), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    simulation.CreateFaceVertexIndicesAttr([0, 1, 2])
    visual = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Cloth/visual")
    visual.CreatePointsAttr([Gf.Vec3f(0.25, 0.25, 0.0)])
    plan = ClonePlan(
        sources=("/World/envs/env_0/Cloth",),
        destinations=("/World/envs/env_{}/Cloth",),
        clone_mask=np.ones((1, 1), dtype=np.bool_),
        env_ids=np.arange(1),
    )

    binding = declare_scene_layout(plan, stage).point_bindings()[0]

    assert (binding.source_count, binding.output_count) == (3, 1)
    assert binding.source_indices.tolist() == [[0, 1, 2, 0]]
    np.testing.assert_allclose(binding.weights, [[0.5, 0.25, 0.25, 0.0]])


def test_nested_rows_assign_bodies_to_the_most_specific_owner() -> None:
    stage = Usd.Stage.CreateInMemory()
    for path in ("/World/envs/env_0/Robot/base", "/World/envs/env_0/Robot/tool"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, path).GetPrim())
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_0/Robot/tool"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot/tool"),
        clone_mask=np.ones((2, 2), dtype=np.bool_),
        env_ids=np.arange(2),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete
    assert [(entry.path, entry.row) for entry in layout.rigid_body_prototypes] == [
        ("/World/envs/env_{}/Robot/base", 0),
        ("/World/envs/env_{}/Robot/tool", 1),
    ]
    assert tuple(layout.iter_rigid_body_paths()) == (
        "/World/envs/env_0/Robot/base",
        "/World/envs/env_0/Robot/tool",
        "/World/envs/env_1/Robot/base",
        "/World/envs/env_1/Robot/tool",
    )


def test_layout_includes_rigid_bodies_below_instanceable_prototypes() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Prototype")
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Prototype/body").GetPrim())
    source = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Asset").GetPrim()
    source.GetReferences().AddInternalReference("/Prototype")
    source.SetInstanceable(True)
    plan = ClonePlan(
        sources=("/World/envs/env_0/Asset",),
        destinations=("/World/envs/env_{}/Asset",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.asarray([5, 1]),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete
    assert list(layout.iter_rigid_body_paths()) == [
        "/World/envs/env_5/Asset/body",
        "/World/envs/env_1/Asset/body",
    ]


def test_frame_bindings_are_fully_declared_from_the_prototype() -> None:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    root.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))
    body = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Body")
    body.AddTranslateOp().Set(Gf.Vec3d(2.0, 0.0, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mount = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Body/Mount")
    mount.AddTranslateOp().Set(Gf.Vec3d(0.25, 0.0, 0.0))
    mount.AddScaleOp().Set(Gf.Vec3f(2.0, 3.0, 4.0))
    marker = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Marker")
    marker.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.0))
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2),
        positions=np.asarray([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete
    mount_1 = layout.match_frames("/World/envs/env_1/Body/Mount")[0]
    assert mount_1.body_path == "/World/envs/env_1/Body"
    assert mount_1.body_view_path == "/World/envs/env_*/Body"
    assert mount_1.pose == pytest.approx((0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    assert mount_1.parent_body_path == "/World/envs/env_1/Body"
    assert mount_1.parent_pose == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    assert mount_1.scale == (2.0, 3.0, 4.0)
    marker_1 = layout.match_frames("/World/envs/env_1/Marker")[0]
    assert marker_1.body_path is None
    assert marker_1.pose == pytest.approx((5.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))


def test_requested_ray_geometry_is_declared_once_and_rebased_exactly() -> None:
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/envs/env_0/Asset/Target")
    mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Asset/Target/mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.AddScaleOp().Set(Gf.Vec3f(2.0, 3.0, 1.0))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("sdf")
    target_expr = r"/World/envs/env_[^/]+/Asset/Target"
    plan = ClonePlan(
        sources=("/World/envs/env_0/Asset",),
        destinations=("/World/envs/env_{}/Asset",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2),
        geometry_requests=(target_expr,),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete
    groups = layout.match_geometry_targets(target_expr)
    assert [target.path for target, _ in groups] == [
        "/World/envs/env_0/Asset/Target",
        "/World/envs/env_1/Asset/Target",
    ]
    assert [geometry.path for _, geometries in groups for geometry in geometries] == [
        "/World/envs/env_0/Asset/Target/mesh",
        "/World/envs/env_1/Asset/Target/mesh",
    ]
    first, second = groups[0][1][0], groups[1][1][0]
    assert first.vertices is second.vertices
    assert first.faces is second.faces
    assert first.collision and second.collision
    assert first.view_path == second.view_path == "/World/envs/env_*/Asset/Target/mesh"
    assert first.collision_approximation == second.collision_approximation == "sdf"
    assert not first.vertices.flags.writeable
    assert not first.faces.flags.writeable
    assert first.vertices[:, 0].max() == pytest.approx(2.0)
    assert first.vertices[:, 1].max() == pytest.approx(3.0)


def test_requested_geometry_preserves_rotated_non_uniform_scale() -> None:
    stage = Usd.Stage.CreateInMemory()
    target = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Asset/Target")
    target.AddScaleOp().Set(Gf.Vec3f(2.0, 1.0, 0.5))
    mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Asset/Target/mesh")
    points = [Gf.Vec3f(), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.AddRotateZOp().Set(45.0)
    target_expr = r"/World/envs/env_[^/]+/Asset/Target"
    plan = ClonePlan(
        sources=("/World/envs/env_0/Asset",),
        destinations=("/World/envs/env_{}/Asset",),
        clone_mask=np.ones((1, 1), dtype=np.bool_),
        env_ids=np.arange(1),
        geometry_requests=(target_expr,),
    )

    geometry = declare_scene_layout(plan, stage).match_geometry_targets(target_expr)[0][1][0]
    pose = geometry.frame.pose
    planned_transform = Gf.Matrix4d().SetRotate(Gf.Quatd(pose[6], Gf.Vec3d(*pose[3:6])))
    planned_transform.SetTranslateOnly(Gf.Vec3d(*pose[:3]))
    expected_transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    expected = [expected_transform.Transform(Gf.Vec3d(point)) for point in points]
    actual = [planned_transform.Transform(Gf.Vec3d(*map(float, point))) for point in geometry.vertices]

    np.testing.assert_allclose(actual, expected, atol=1.0e-6)


def test_unused_geometry_request_is_strict_only_when_consumed() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Ground")
    plan = ClonePlan(
        sources=("/World/Ground",),
        destinations=("/World/Ground",),
        clone_mask=np.zeros((1, 1), dtype=np.bool_),
        env_ids=np.arange(1),
        geometry_requests=("/World/Missing",),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete
    assert layout.geometry_prototypes == ()
    with pytest.raises(ValueError, match="not covered by the clone plan"):
        layout.match_geometry_targets("/World/Missing")


def test_articulation_roots_joints_and_child_anchors_are_plan_facts() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    link = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/link")
    base = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/base")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    link.GetPrim().CreateAttribute("isaac:nameOverride", Sdf.ValueTypeNames.String).Set("tool")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/envs/env_0/Robot/joints/hinge")
    joint.GetPrim().CreateAttribute("isaac:nameOverride", Sdf.ValueTypeNames.String).Set("joint")
    _add_api_schemas(joint.GetPrim(), ["PhysxTendonAxisRootAPI"])
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateBody1Rel().SetTargets([link.GetPath()])
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.1, 0.2, 0.3))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete
    articulations = layout.match_articulations(r"/World/envs/env_[^/]+/Robot")
    assert [entry.root_path for entry in articulations] == [
        "/World/envs/env_0/Robot/base",
        "/World/envs/env_1/Robot/base",
    ]
    assert {entry.view_path for entry in articulations} == {"/World/envs/env_*/Robot/base"}
    assert [(body.path, body.name) for body in articulations[1].bodies] == [
        ("/World/envs/env_1/Robot/base", "base"),
        ("/World/envs/env_1/Robot/link", "tool"),
    ]
    assert list(layout.iter_rigid_body_paths()) == [
        "/World/envs/env_0/Robot/base",
        "/World/envs/env_0/Robot/link",
        "/World/envs/env_1/Robot/base",
        "/World/envs/env_1/Robot/link",
    ]
    planned_joint = articulations[1].joints[0]
    assert planned_joint.path == "/World/envs/env_1/Robot/joints/hinge"
    assert planned_joint.name == "joint"
    assert planned_joint.parent_path == "/World/envs/env_1/Robot/base"
    assert planned_joint.child_path == "/World/envs/env_1/Robot/link"
    assert planned_joint.tendon_type == "fixed"
    assert planned_joint.pose == pytest.approx((0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0))


def test_articulation_bodies_cover_joint_roots_and_jointless_roots() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/JointRoot")
    body = UsdGeom.Xform.Define(stage, "/World/JointRoot/body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    root_joint = UsdPhysics.FixedJoint.Define(stage, "/World/JointRoot/root")
    UsdPhysics.ArticulationRootAPI.Apply(root_joint.GetPrim())
    root_joint.CreateBody1Rel().SetTargets([body.GetPath()])

    singleton = UsdGeom.Xform.Define(stage, "/World/Singleton")
    UsdPhysics.ArticulationRootAPI.Apply(singleton.GetPrim())
    only_body = UsdGeom.Xform.Define(stage, "/World/Singleton/only_body")
    UsdPhysics.RigidBodyAPI.Apply(only_body.GetPrim())
    plan = ClonePlan(
        sources=("/World/JointRoot", "/World/Singleton"),
        destinations=("/World/JointRoot", "/World/Singleton"),
        clone_mask=np.zeros((2, 1), dtype=np.bool_),
        env_ids=np.arange(1),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete
    assert [body.path for body in layout.match_articulations("/World/JointRoot")[0].bodies] == ["/World/JointRoot/body"]
    assert [body.path for body in layout.match_articulations("/World/Singleton")[0].bodies] == [
        "/World/Singleton/only_body"
    ]


def test_articulation_branch_order_is_canonical() -> None:
    stage = Usd.Stage.CreateInMemory()
    base = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/base")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    branches = {}
    for name in ("branch_z", "branch_a"):
        branch = UsdGeom.Xform.Define(stage, f"/World/envs/env_0/Robot/{name}")
        UsdPhysics.RigidBodyAPI.Apply(branch.GetPrim())
        branches[name] = branch
    root_joint = UsdPhysics.FixedJoint.Define(stage, "/World/envs/env_0/Robot/root_joint")
    root_joint.CreateBody1Rel().SetTargets([base.GetPath()])
    for joint_name, branch_name in (("z_joint", "branch_z"), ("a_joint", "branch_a")):
        joint = UsdPhysics.FixedJoint.Define(stage, f"/World/envs/env_0/Robot/{joint_name}")
        endpoints = (branches[branch_name], base) if joint_name == "z_joint" else (base, branches[branch_name])
        joint.CreateBody0Rel().SetTargets([endpoints[0].GetPath()])
        joint.CreateBody1Rel().SetTargets([endpoints[1].GetPath()])
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=np.ones((1, 1), dtype=np.bool_),
        env_ids=np.arange(1),
    )

    articulation = declare_scene_layout(plan, stage).match_articulations("/World/envs/env_0/Robot/base")[0]

    assert [body.name for body in articulation.bodies] == ["base", "branch_a", "branch_z"]
    assert [joint.name for joint in articulation.joints] == ["a_joint", "root_joint", "z_joint"]


def test_backend_metadata_is_declared_from_exact_plan_sources() -> None:
    stage = Usd.Stage.CreateInMemory()
    source = "/World/envs/env_0/Asset"
    body = UsdGeom.Xform.Define(stage, f"{source}/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    _add_api_schemas(body, [*body.GetAppliedSchemas(), "PhysxContactReportAPI"])
    _surface(stage, f"{source}/cloth", 3)
    cloth = stage.GetPrimAtPath(f"{source}/cloth")
    material = UsdShade.Material.Define(stage, f"{source}/materials/soft")
    material.GetPrim().CreateAttribute("newton:density", Sdf.ValueTypeNames.Float).Set(123.0)
    UsdShade.MaterialBindingAPI.Apply(cloth)
    UsdShade.MaterialBindingAPI(cloth).Bind(
        material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )
    gripper = stage.DefinePrim(f"{source}/gripper", "IsaacSurfaceGripper")
    for name, value in {
        "isaac:maxGripDistance": 0.01,
        "isaac:coaxialForceLimit": 11.0,
        "isaac:shearForceLimit": 12.0,
        "isaac:retryInterval": 0.25,
    }.items():
        gripper.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
    plan = ClonePlan(
        sources=(source,),
        destinations=("/World/envs/env_{}/Asset",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.asarray((4, 2)),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete
    assert all(body.contact_report for body in layout.match_rigid_body_subtrees(r"/World/envs/env_[^/]+/Asset/body"))
    deformables = layout.match_deformable_subtrees(r"/World/envs/env_[^/]+/Asset/cloth")
    assert [entry.material_path for entry in deformables] == [
        "/World/envs/env_4/Asset/materials/soft",
        "/World/envs/env_2/Asset/materials/soft",
    ]
    assert {entry.material_view_path for entry in deformables} == {"/World/envs/env_*/Asset/materials/soft"}
    assert {entry.source_path for entry in deformables} == {"/World/envs/env_0/Asset/cloth"}
    assert all(entry.vertices.shape == (3, 3) for entry in deformables)
    assert all(dict(entry.material_attributes)["newton:density"] == pytest.approx(123.0) for entry in deformables)
    grippers = layout.match_surface_grippers(r"/World/envs/env_[^/]+/Asset/gripper")
    assert [entry.env_id for entry in grippers] == [4, 2]
    assert {entry.view_path for entry in grippers} == {"/World/envs/env_*/Asset/gripper"}
    for entry in grippers:
        assert (
            entry.max_grip_distance,
            entry.coaxial_force_limit,
            entry.shear_force_limit,
            entry.retry_interval,
        ) == pytest.approx((0.01, 11.0, 12.0, 0.25))


def test_contact_matching_uses_planned_descendants_and_segment_safe_leaf_expressions() -> None:
    stage = Usd.Stage.CreateInMemory()
    source = "/World/envs/env_0/Robot"
    for path in (
        f"{source}/left_shoulder",
        f"{source}/pelvis",
        f"{source}/pelvis/left_hip",
        f"{source}/pelvis/left_knee",
    ):
        body = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        _add_api_schemas(body, [*body.GetAppliedSchemas(), "PhysxContactReportAPI"])
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"{source}/left_unreported").GetPrim())
    UsdGeom.Mesh.Define(stage, f"{source}/left_shoulder/geometry/left_visual")
    collision = UsdGeom.Mesh.Define(stage, f"{source}/left_shoulder/collisions/left_collision").GetPrim()
    UsdPhysics.CollisionAPI.Apply(collision)
    plan = ClonePlan(
        sources=(source,),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete
    bodies = layout.match_contact_bodies(r"/World/envs/env_[^/]+/Robot/left_.*")
    assert [body.path for body in bodies] == [
        "/World/envs/env_0/Robot/left_shoulder",
        "/World/envs/env_0/Robot/pelvis/left_hip",
        "/World/envs/env_0/Robot/pelvis/left_knee",
        "/World/envs/env_1/Robot/left_shoulder",
        "/World/envs/env_1/Robot/pelvis/left_hip",
        "/World/envs/env_1/Robot/pelvis/left_knee",
    ]


def test_predeclared_empty_layout_never_fetches_a_stage() -> None:
    plan = ClonePlan(
        sources=(),
        destinations=(),
        clone_mask=np.zeros((0, 0), dtype=np.bool_),
        is_complete=True,
    )
    assert declare_scene_layout(plan, object()) is plan


def test_frame_and_geometry_declaration_stays_prototype_sized_for_many_clones(monkeypatch) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Asset/Target")
    mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Asset/Target/mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    plan = ClonePlan(
        sources=("/World/envs/env_0/Asset",),
        destinations=("/World/envs/env_{}/Asset",),
        clone_mask=np.ones((1, 4096), dtype=np.bool_),
        env_ids=np.arange(4096),
        geometry_requests=(r"/World/envs/env_[^/]+/Asset/Target",),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete
    assert len(layout.frame_prototypes) == 2
    assert len(layout.geometry_prototypes) == 1
    assert layout.frame_prototypes[0].clone_mask is layout.frame_prototypes[1].clone_mask
    assert layout.geometry_prototypes[0].clone_mask is layout.geometry_prototypes[0].frame.clone_mask
    assert not hasattr(layout, "frames")
    assert not hasattr(layout, "geometries")
    groups = layout.match_geometry_targets("/World/envs/env_4095/Asset/Target")
    assert [(target.path, geometries[0].path) for target, geometries in groups] == [
        ("/World/envs/env_4095/Asset/Target", "/World/envs/env_4095/Asset/Target/mesh")
    ]

    def reject_materialization(*_args):
        pytest.fail("Compact prototype matching must not materialize per-environment layout records.")

    monkeypatch.setattr(ClonePlan, "_materialize_frame", reject_materialization)
    monkeypatch.setattr(ClonePlan, "_materialize_geometry", reject_materialization)
    [(target, geometries, clone_mask)] = layout.match_geometry_prototypes(r"/World/envs/env_[^/]+/Asset/Target")
    assert target.path == "/World/envs/env_{}/Asset/Target"
    assert geometries == layout.geometry_prototypes
    assert np.count_nonzero(clone_mask) == 4096


def test_rigid_and_articulation_declaration_stays_prototype_sized_for_many_clones(monkeypatch) -> None:
    stage = Usd.Stage.CreateInMemory()
    source = "/World/envs/env_0/Robot"
    UsdGeom.Xform.Define(stage, source)
    base = UsdGeom.Xform.Define(stage, f"{source}/base")
    link = UsdGeom.Xform.Define(stage, f"{source}/link")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{source}/joint")
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateBody1Rel().SetTargets([link.GetPath()])
    plan = ClonePlan(
        sources=(source,),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=np.ones((1, 4096), dtype=np.bool_),
        env_ids=np.arange(4096),
    )

    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete
    assert len(layout.rigid_body_prototypes) == 2
    assert len(layout.articulation_prototypes) == 1
    articulation = layout.articulation_prototypes[0]
    assert len(articulation.joints) == 1
    assert articulation.clone_mask is layout.rigid_body_prototypes[0].clone_mask
    assert articulation.clone_mask is layout.rigid_body_prototypes[1].clone_mask
    assert not hasattr(layout, "rigid_bodies")
    assert not hasattr(layout, "iter_rigid_bodies")
    assert not hasattr(layout, "articulations")
    materialized = []
    materialize = ClonePlan._materialize_articulation

    def count_materialized(self, prototype, column):
        materialized.append(column)
        return materialize(self, prototype, column)

    monkeypatch.setattr(ClonePlan, "_materialize_articulation", count_materialized)
    representative = layout.match_articulation(r"/World/envs/env_[^/]+/Robot/base")
    assert representative.root_path == "/World/envs/env_0/Robot/base"
    assert materialized == [0]
    exact = layout.match_articulations("/World/envs/env_4095/Robot/base")[0]
    assert exact.root_path == "/World/envs/env_4095/Robot/base"
    assert [body.path for body in exact.bodies] == [
        "/World/envs/env_4095/Robot/base",
        "/World/envs/env_4095/Robot/link",
    ]
    assert exact.joints[0].path == "/World/envs/env_4095/Robot/joint"

    monkeypatch.setattr(
        ClonePlan,
        "_materialize_rigid_body",
        lambda *_args: pytest.fail("Compact rigid-body matching must not expand clone records."),
    )
    assert len(tuple(layout.iter_rigid_body_paths())) == 8192
    body = layout.match_rigid_body(r"/World/envs/env_[^/]+/Robot/base")
    assert body is layout.rigid_body_prototypes[0]
    with pytest.raises(ValueError, match="does not resolve exactly one body"):
        layout.match_rigid_body(r"/World/envs/env_[^/]+/Robot")


def test_match_articulation_rejects_heterogeneous_newton_actuator_declarations() -> None:
    stage = Usd.Stage.CreateInMemory()
    for env_id, kp in enumerate((10.0, 20.0)):
        root_path = f"/World/envs/env_{env_id}/Robot"
        root = UsdGeom.Xform.Define(stage, root_path)
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        body = UsdGeom.Xform.Define(stage, f"{root_path}/body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root_path}/joint")
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        actuator = stage.DefinePrim(f"{root_path}/actuator", "NewtonActuator")
        _add_api_schemas(actuator, ["NewtonPDControlAPI"])
        actuator.CreateRelationship("newton:targets").SetTargets([joint.GetPath()])
        actuator.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(kp)
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_1/Robot"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot"),
        clone_mask=np.asarray([[True, False], [False, True]]),
        env_ids=np.arange(2),
    )
    layout = declare_scene_layout(plan, stage)

    assert layout.is_complete
    exact = layout.match_articulation("/World/envs/env_0/Robot")
    assert exact.joints[0].newton_actuator.controller_arguments[0] == ("kp", 10.0)
    with pytest.raises(ValueError, match="incompatible Newton actuator declarations"):
        layout.match_articulation(r"/World/envs/env_[^/]+/Robot")


@pytest.mark.parametrize("schema", [UsdGeom.Cube, UsdGeom.PointInstancer, UsdGeom.Camera])
def test_layout_ignores_stage_content_outside_plan_sources(schema) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage, "/World/Planned/geometry")
    schema.Define(stage, "/World/Unplanned")
    plan = ClonePlan(
        sources=("/World/Planned",),
        destinations=("/World/Planned",),
        clone_mask=np.zeros((1, 1), dtype=np.bool_),
        env_ids=np.arange(1),
    )

    layout = declare_scene_layout(plan, stage)
    assert layout.is_complete
    assert all("Unplanned" not in frame.path for frame in layout.frame_prototypes)


def test_layout_allows_unplanned_non_drawable_stage_support() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage, "/World/Planned/geometry")
    UsdGeom.Scope.Define(stage, "/World/Support")
    UsdGeom.Xform.Define(stage, "/World/Support/frame")
    UsdPhysics.Scene.Define(stage, "/World/Support/physics")
    UsdShade.Material.Define(stage, "/World/Support/material")
    plan = ClonePlan(
        sources=("/World/Planned",),
        destinations=("/World/Planned",),
        clone_mask=np.zeros((1, 1), dtype=np.bool_),
        env_ids=np.arange(1),
    )

    assert declare_scene_layout(plan, stage).is_complete
