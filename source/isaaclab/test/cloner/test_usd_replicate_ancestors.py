# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for USD replication resource ownership."""

from types import SimpleNamespace

import numpy as np
import pytest

from pxr import Gf, Sdf, Usd, UsdGeom

import isaaclab.cloner.usd as usd_cloner
from isaaclab.cloner import ClonePlan, UsdReplicateContext
from isaaclab.renderers import fabric_visual_material
from isaaclab.scene_data.scene_data_backend import SceneDataFormat


def _completed_plan() -> ClonePlan:
    return ClonePlan((), (), np.zeros((0, 0), dtype=np.bool_), np.empty(0, dtype=np.int64), is_complete=True)


def _make_stage_with_source(source_path: str) -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    for prefix in Sdf.Path(source_path).GetPrefixes():
        stage.DefinePrim(prefix, "Xform")
    return stage


def _replicate(stage: Usd.Stage, sources, destinations, env_ids, positions=None) -> None:
    usd_cloner.usd_replicate(stage, sources, destinations, env_ids, positions=positions)


def test_usd_replicate_defines_nested_destination_ancestors():
    """Copied prims under a nested scope compose as defined prims in target envs."""
    stage = _make_stage_with_source("/World/envs/env_0/Groceries/Object")

    _replicate(
        stage,
        sources=["/World/envs/env_0/Groceries/Object"],
        destinations=["/World/envs/env_{}/Groceries/Object"],
        env_ids=np.asarray([0, 1], dtype=np.int64),
    )

    copied_scope = stage.GetPrimAtPath("/World/envs/env_1/Groceries")
    copied_prim = stage.GetPrimAtPath("/World/envs/env_1/Groceries/Object")
    assert copied_scope.IsDefined(), "intermediate ancestor must compose as a defined prim"
    assert copied_prim.IsDefined(), "copied prim must compose as a defined prim"


def test_usd_replicate_keeps_existing_ancestor_specs():
    """Ancestors already defined in the target env are left untouched."""
    stage = _make_stage_with_source("/World/envs/env_0/Groceries/Object")
    for prefix in Sdf.Path("/World/envs/env_1/Groceries").GetPrefixes():
        stage.DefinePrim(prefix, "Xform")

    _replicate(
        stage,
        sources=["/World/envs/env_0/Groceries/Object"],
        destinations=["/World/envs/env_{}/Groceries/Object"],
        env_ids=np.asarray([0, 1], dtype=np.int64),
    )

    scope = stage.GetPrimAtPath("/World/envs/env_1/Groceries")
    assert scope.IsDefined()
    assert scope.GetTypeName() == "Xform"
    assert stage.GetPrimAtPath("/World/envs/env_1/Groceries/Object").IsDefined()


def test_usd_replicate_does_not_reauthor_plan_owned_env_root():
    """An asset-row copy preserves the environment transform already authored by the plan."""
    stage = _make_stage_with_source("/World/envs/env_0/Robot")
    env_1 = UsdGeom.Xform.Define(stage, "/World/envs/env_1")
    env_1.AddTranslateOp().Set(Gf.Vec3d(5.0, 0.0, 0.0))

    _replicate(
        stage,
        sources=["/World/envs/env_0/Robot"],
        destinations=["/World/envs/env_{}/Robot"],
        env_ids=np.asarray([0, 1], dtype=np.int64),
        positions=np.asarray([[0.0, 0.0, 0.0], [99.0, 0.0, 0.0]], dtype=np.float32),
    )

    assert env_1.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation() == Gf.Vec3d(5.0, 0.0, 0.0)


def test_fabric_visual_material_writer_requires_initialized_fabric() -> None:
    context = UsdReplicateContext(Usd.Stage.CreateInMemory())

    with pytest.raises(RuntimeError, match="initialized USD Fabric destinations"):
        context.create_fabric_visual_material_writer(())


def test_fabric_selection_maps_exact_plan_order_and_rejects_missing_paths() -> None:
    selection = SimpleNamespace(GetPaths=lambda: ("/extra", "/B", "/A"), GetCount=lambda: 3)

    path_slots = UsdReplicateContext._selection_path_slots(selection)

    assert UsdReplicateContext._exact_path_slots(path_slots, ("/A", "/B"), "transform") == [2, 1]
    with pytest.raises(RuntimeError, match="missing plan paths"):
        UsdReplicateContext._exact_path_slots(path_slots, ("/A", "/C"), "transform")


@pytest.mark.parametrize("paths, count", [(("/A", "/A"), 2), (("/A",), 2)])
def test_fabric_selection_rejects_invalid_path_metadata(paths: tuple[str, ...], count: int) -> None:
    selection = SimpleNamespace(GetPaths=lambda: paths, GetCount=lambda: count)

    with pytest.raises(RuntimeError, match="invalid path metadata"):
        UsdReplicateContext._selection_path_slots(selection)


def test_fabric_preparation_requires_fsd_before_usdrt_import(monkeypatch) -> None:
    monkeypatch.setattr(usd_cloner, "get_settings_manager", lambda: SimpleNamespace(get=lambda *_: False))
    context = UsdReplicateContext(Usd.Stage.CreateInMemory())

    with pytest.raises(RuntimeError, match="/app/useFabricSceneDelegate=true"):
        context._prepare_fabric(object(), "cpu", _completed_plan())


def test_fabric_visual_material_writer_uses_the_resource_binder(monkeypatch) -> None:
    context = UsdReplicateContext(Usd.Stage.CreateInMemory())
    fabric_stage = object()
    context._fabric_stage = fabric_stage
    calls = []
    monkeypatch.setattr(
        fabric_visual_material,
        "FabricVisualMaterialWriter",
        lambda bind, batches: calls.append((bind, batches)) or "writer",
    )

    assert context.create_fabric_visual_material_writer(()) == "writer"
    assert calls[0][0].__self__ is context
    assert calls[0][0].__name__ == "_bind_fabric_visual_material"
    assert calls[0][1] == ()


def test_fabric_selection_rebind_disables_the_structural_hint_once(monkeypatch) -> None:
    """Every changed plan selection invalidates one hierarchy propagation hint."""

    class Selection:
        def __init__(self, paths: tuple[str, ...], topology_changed: bool = True):
            self.paths = paths
            self.topology_changed = topology_changed
            self.prepare_count = 0

        def PrepareForReuse(self) -> bool:
            self.prepare_count += 1
            return self.topology_changed

        def GetPaths(self) -> tuple[str, ...]:
            return self.paths

        def GetCount(self) -> int:
            return len(self.paths)

    hints = []
    provider = SimpleNamespace(_fabric_generation=0)
    hierarchy = SimpleNamespace(update_world_xforms_gpu=lambda hint: hints.append(hint) or True)
    context = UsdReplicateContext(Usd.Stage.CreateInMemory())
    context._fabric_provider = provider
    context._fabric_hierarchy = hierarchy
    context._fabric_hierarchy_generation = 0
    context._fabric_device = "cpu"
    context._fabric_plan = SimpleNamespace(iter_rigid_body_paths=lambda: ("/Body",))
    monkeypatch.setattr(usd_cloner.wp, "array", lambda values, **_: tuple(values))
    monkeypatch.setattr(usd_cloner.wp, "fabricarray", lambda *_, **__: object())
    monkeypatch.setattr(usd_cloner.wp, "fabricarrayarray", lambda *_, **__: object())
    monkeypatch.setattr(usd_cloner.wp, "indexedfabricarray", lambda *_, **__: object())
    monkeypatch.setattr(usd_cloner.wp, "synchronize_device", lambda *_: None)

    point_selection = Selection(("/Points",))
    context._fabric_point_selections["points"] = point_selection
    context._fabric_point_outputs["points"] = SimpleNamespace()
    context._fabric_point_paths["points"] = ("/Points",)
    context._prepare_fabric_output(SceneDataFormat.FabricMeshPoints, "points")
    assert context._fabric_topology_changed
    context._update_fabric_hierarchy()
    assert not context._fabric_topology_changed

    provider._fabric_generation += 1
    transform_selection = Selection(("/Body",))
    context._fabric_transform_selection = transform_selection
    context._fabric_transform_output = SimpleNamespace()
    context._prepare_fabric_output(SceneDataFormat.FabricMatrix44, None)
    assert transform_selection.prepare_count == 1
    assert context._fabric_topology_changed
    context._update_fabric_hierarchy()

    provider._fabric_generation += 1
    transform_selection.topology_changed = False
    camera_selection = Selection(("/Camera",))
    context._fabric_camera_selection = camera_selection
    context._fabric_named_transform_outputs["camera"] = (SimpleNamespace(), ("/Camera",))
    context._prepare_fabric_output(SceneDataFormat.FabricMatrix44, "camera")
    assert transform_selection.prepare_count == 1
    assert camera_selection.prepare_count == 1
    assert context._fabric_topology_changed
    context._update_fabric_hierarchy()

    provider._fabric_generation += 1
    camera_selection.topology_changed = False
    context._prepare_fabric_output(SceneDataFormat.FabricMatrix44, "camera")
    assert transform_selection.prepare_count == 1
    assert camera_selection.prepare_count == 2
    context._update_fabric_hierarchy()

    assert hints == [False, False, False, True]
