# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less tests for :class:`VisualMaterial` entities and the bucket color randomization event."""

from types import SimpleNamespace

import pytest
import torch

from pxr import Sdf, Usd, UsdShade

import isaaclab.assets.visual_material.base_visual_material as visual_material_module
import isaaclab.envs.mdp.visual_events as visual_events
import isaaclab.sim.spawners.materials.visual_materials as visual_materials_module
from isaaclab.assets import VisualMaterialCfg
from isaaclab.assets.visual_material import BaseVisualMaterial as VisualMaterial
from isaaclab.cloner import ClonePlan
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners.materials import PbrMdlCfg, PreviewSurfaceCfg


class _RenderContextRecorder:
    def __init__(self):
        self.calls = []

    def notify_visual_material_written(self, writes):
        for w in writes:
            values = w.values.clone() if hasattr(w.values, "clone") else list(w.values)
            self.calls.append((list(w.material_paths), list(w.shader_paths), w.attr_name, w.semantic, values))

    def register_visual_material_textures(self, texture_paths):
        pass


@pytest.fixture
def stage(monkeypatch):
    """In-memory stage wired into the material entity, the material spawners, and kitless authoring."""
    import isaaclab.sim.utils.stage as stage_utils_module

    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(stage_utils_module, "get_current_stage", lambda: stage)
    monkeypatch.setattr(visual_material_module, "get_current_stage", lambda: stage)
    monkeypatch.setattr(visual_materials_module, "get_current_stage", lambda: stage)
    monkeypatch.setattr(visual_materials_module, "has_kit", lambda: False)
    return stage


@pytest.fixture
def render_context(monkeypatch):
    recorder = _RenderContextRecorder()
    fake_sim = SimpleNamespace(render_context=recorder)
    monkeypatch.setattr(SimulationContext, "instance", classmethod(lambda cls: fake_sim))
    return recorder


def _author_mdl_material(stage, prim_path: str, sub_identifier: str) -> None:
    material = UsdShade.Material.Define(stage, prim_path)
    shader = UsdShade.Shader.Define(stage, f"{prim_path}/Shader")
    shader.SetSourceAsset(Sdf.AssetPath(f"{sub_identifier}.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier(sub_identifier, "mdl")
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.CreateOutput("out", Sdf.ValueTypeNames.Token))


def test_pbr_mdl_material_resolves_omnipbr_color_input(stage):
    material = VisualMaterial(
        VisualMaterialCfg(prim_path="/World/Materials/bucket", spawn=PbrMdlCfg(diffuse_color_constant=(0.1, 0.2, 0.3)))
    )

    assert material.material_prim_path == "/World/Materials/bucket"
    assert material._shader_prim_path == "/World/Materials/bucket/Shader"
    assert material._channels["color"].input_name == "diffuse_color_constant"
    assert tuple(material.read_channel("color")) == pytest.approx((0.1, 0.2, 0.3))


def test_preview_surface_material_resolves_diffuse_color_input(stage):
    material = VisualMaterial(
        VisualMaterialCfg(prim_path="/World/Materials/bucket", spawn=PreviewSurfaceCfg(diffuse_color=(0.4, 0.5, 0.6)))
    )

    assert material._channels["color"].input_name == "diffuseColor"
    assert tuple(material.read_channel("color")) == pytest.approx((0.4, 0.5, 0.6))


def test_wraps_existing_material_without_spawn(stage):
    _author_mdl_material(stage, "/World/Materials/baked", "OmniPBR")

    material = VisualMaterial(VisualMaterialCfg(prim_path="/World/Materials/baked", spawn=None))

    assert material._channels["color"].input_name == "diffuse_color_constant"
    # channels are default-authored at init: detached renderers cannot write attributes that
    # were absent from their exported stage
    assert tuple(material.read_channel("color")) == pytest.approx((0.18, 0.18, 0.18))


@pytest.mark.parametrize(
    ("prim_path", "match"),
    [
        ("/World/Materials/env_.*/mat", "concrete absolute prim path"),
        ("relative/path", "concrete absolute prim path"),
    ],
)
def test_invalid_prim_paths_rejected(stage, prim_path, match):
    with pytest.raises(ValueError, match=match):
        VisualMaterial(VisualMaterialCfg(prim_path=prim_path, spawn=None))


def test_missing_material_without_spawn_rejected(stage):
    with pytest.raises(ValueError, match="does not exist"):
        VisualMaterial(VisualMaterialCfg(prim_path="/World/Materials/missing", spawn=None))


def test_unsupported_mdl_family_rejected(stage):
    _author_mdl_material(stage, "/World/Materials/velvet", "OmniSurfaceVelvet")

    with pytest.raises(ValueError, match="unsupported MDL sub-identifier"):
        VisualMaterial(VisualMaterialCfg(prim_path="/World/Materials/velvet", spawn=None))


def test_omni_glass_family_resolves_glass_channels(stage):
    _author_mdl_material(stage, "/World/Materials/glass", "OmniGlass")

    material = VisualMaterial(
        VisualMaterialCfg(prim_path="/World/Materials/glass", spawn=None, channels=("color", "roughness", "ior"))
    )

    assert material._channels["color"].input_name == "glass_color"
    assert material._channels["roughness"].input_name == "frosting_roughness"
    assert material._channels["ior"].input_name == "glass_ior"


def test_unconnected_material_rejected(stage):
    UsdShade.Material.Define(stage, "/World/Materials/empty")

    with pytest.raises(ValueError, match="no connected surface shader"):
        VisualMaterial(VisualMaterialCfg(prim_path="/World/Materials/empty", spawn=None))


def test_write_color_authors_usd_and_notifies_renderers(stage, render_context):
    material = VisualMaterial(VisualMaterialCfg(prim_path="/World/Materials/bucket", spawn=PbrMdlCfg()))

    material.write_channels([material], {"color": torch.tensor([[0.2, 0.4, 0.6]])})

    assert tuple(material.read_channel("color")) == pytest.approx((0.2, 0.4, 0.6))
    (material_paths, shader_paths, attr_name, semantic, colors) = render_context.calls[0]
    assert material_paths == ["/World/Materials/bucket"]
    assert shader_paths == ["/World/Materials/bucket/Shader"]
    assert attr_name == "inputs:diffuse_color_constant"
    assert semantic == "color"
    assert torch.allclose(colors, torch.tensor([[0.2, 0.4, 0.6]]))


def test_write_helper_batches_one_notify(stage, render_context):
    materials = [
        VisualMaterial(VisualMaterialCfg(prim_path=f"/World/Materials/bucket_{i}", spawn=PbrMdlCfg())) for i in range(3)
    ]
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    VisualMaterial.write_channels(materials, {"color": colors})

    assert len(render_context.calls) == 1
    assert [tuple(m.read_channel("color")) for m in materials] == [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]


def test_write_helper_validates_color_shape(stage, render_context):
    material = VisualMaterial(VisualMaterialCfg(prim_path="/World/Materials/bucket", spawn=PbrMdlCfg()))

    with pytest.raises(ValueError, match="expects 1 rows"):
        VisualMaterial.write_channels([material], {"color": torch.zeros(2, 3)})
    assert render_context.calls == []


def test_per_part_material_bindings_resolve_on_spawned_asset(stage, tmp_path, monkeypatch):
    """visual_material_bindings binds shared bucket materials to specific sub-prims of a USD asset."""
    import isaaclab.sim.spawners.from_files.from_files as from_files_module
    import isaaclab.sim.utils.prims as prims_module
    import isaaclab.sim.utils.stage as stage_utils_module

    monkeypatch.setattr(from_files_module, "get_current_stage", lambda: stage)
    monkeypatch.setattr(prims_module, "get_current_stage", lambda: stage)
    monkeypatch.setattr(stage_utils_module, "get_current_stage", lambda: stage)

    asset_file = tmp_path / "two_part_asset.usda"
    asset_file.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
)
def Xform "Asset"
{
    def Xform "body"
    {
        def Cube "mesh"
        {
        }
    }
    def Xform "arm"
    {
        def Cube "mesh"
        {
        }
    }
}
"""
    )
    for name in ("body_color", "arm_color"):
        VisualMaterial(VisualMaterialCfg(prim_path=f"/World/Materials/{name}", spawn=PbrMdlCfg()))

    import isaaclab.sim as sim_utils

    cfg = sim_utils.UsdFileCfg(
        usd_path=str(asset_file),
        visual_material_bindings={
            "body": "/World/Materials/body_color",
            "arm": "/World/Materials/arm_color",
        },
    )
    cfg.func("/World/Robot", cfg)

    for part, material in (("body", "body_color"), ("arm", "arm_color")):
        bound = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(f"/World/Robot/{part}/mesh")).ComputeBoundMaterial()[0]
        assert str(bound.GetPrim().GetPath()) == f"/World/Materials/{material}", part


def test_per_part_material_bindings_apply_to_instanceable_parts(stage, tmp_path, monkeypatch):
    """Bindings author on instanceable part prims (physics-style nested walking would skip them)."""
    import isaaclab.sim as sim_utils
    import isaaclab.sim.spawners.from_files.from_files as from_files_module
    import isaaclab.sim.utils.prims as prims_module
    import isaaclab.sim.utils.stage as stage_utils_module

    monkeypatch.setattr(from_files_module, "get_current_stage", lambda: stage)
    monkeypatch.setattr(prims_module, "get_current_stage", lambda: stage)
    monkeypatch.setattr(stage_utils_module, "get_current_stage", lambda: stage)

    asset_file = tmp_path / "instanceable_asset.usda"
    asset_file.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
)
def Xform "proto"
{
    def Cube "mesh"
    {
    }
}
def Xform "Asset"
{
    def Xform "body" (
        instanceable = true
        prepend references = </proto>
    )
    {
    }
}
"""
    )
    VisualMaterial(VisualMaterialCfg(prim_path="/World/Materials/body_color", spawn=PbrMdlCfg()))

    cfg = sim_utils.UsdFileCfg(
        usd_path=str(asset_file),
        visual_material_bindings={"body": "/World/Materials/body_color"},
    )
    cfg.func("/World/Robot", cfg)

    body = stage.GetPrimAtPath("/World/Robot/body")
    assert body.IsInstance()
    mesh = stage.GetPrimAtPath("/World/Robot/body/mesh")
    bound = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
    assert str(bound.GetPrim().GetPath()) == "/World/Materials/body_color"


def test_per_part_material_bindings_reject_absolute_parts(stage, tmp_path, monkeypatch):
    import isaaclab.sim as sim_utils
    import isaaclab.sim.spawners.from_files.from_files as from_files_module

    monkeypatch.setattr(from_files_module, "get_current_stage", lambda: stage)
    asset_file = tmp_path / "asset.usda"
    asset_file.write_text('#usda 1.0\n(\n    defaultPrim = "Asset"\n)\ndef Xform "Asset" {}\n')

    cfg = sim_utils.UsdFileCfg(
        usd_path=str(asset_file),
        visual_material_bindings={"/World/elsewhere": "/World/Materials/x"},
    )
    with pytest.raises(ValueError, match="asset-relative"):
        cfg.func("/World/Robot", cfg)


class _Scene:
    def __init__(self, entries):
        self._entries = entries

    def __getitem__(self, name):
        return self._entries[name]


def _term_env(stage, names=("bucket_a", "bucket_b")):
    entries = {
        name: VisualMaterial(VisualMaterialCfg(prim_path=f"/World/Materials/{name}", spawn=PbrMdlCfg()))
        for name in names
    }
    return SimpleNamespace(scene=_Scene(entries)), entries


def _term_cfg(materials, colors=None):
    return SimpleNamespace(
        params={
            "materials": materials,
            "colors": colors if colors is not None else {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        }
    )


def test_event_samples_one_color_per_bucket(stage, render_context):
    env, entries = _term_env(stage)
    cfg = _term_cfg([SceneEntityCfg("bucket_a"), SceneEntityCfg("bucket_b")], colors=((0.2, 0.3, 0.4), (0.5, 0.6, 0.7)))

    term = visual_events.randomize_visual_color(cfg, env)
    term(env, torch.tensor([1]), **cfg.params)

    (material_paths, _shaders, _attr, _semantic, colors) = render_context.calls[0]
    assert material_paths == ["/World/Materials/bucket_a", "/World/Materials/bucket_b"]
    assert colors.shape == (2, 3)
    assert torch.all(colors >= torch.tensor([0.2, 0.3, 0.4])) and torch.all(colors <= torch.tensor([0.5, 0.6, 0.7]))
    for name in entries:
        assert entries[name].read_channel("color") is not None


def test_event_ignores_env_ids_by_design(stage, render_context):
    env, _entries = _term_env(stage, names=("bucket_a",))
    cfg = _term_cfg(SceneEntityCfg("bucket_a"))

    term = visual_events.randomize_visual_color(cfg, env)
    term(env, None, **cfg.params)
    term(env, torch.tensor([0]), **cfg.params)
    term(env, slice(None), **cfg.params)

    # every fire writes the full bucket set regardless of the env_ids selection
    assert [len(call[0]) for call in render_context.calls] == [1, 1, 1]


def test_event_rejects_non_material_entities(stage):
    env = SimpleNamespace(scene=_Scene({"robot": SimpleNamespace()}))
    cfg = _term_cfg([SceneEntityCfg("robot")])

    with pytest.raises(TypeError, match="VisualMaterial"):
        visual_events.randomize_visual_color(cfg, env)


def test_event_requires_at_least_one_material(stage):
    env, _entries = _term_env(stage)
    cfg = _term_cfg([])

    with pytest.raises(ValueError, match="at least one material"):
        visual_events.randomize_visual_color(cfg, env)


"""
Per-environment materials.
"""

_ENV_NS = "/World/envs/env_0"
_PER_ENV_SOURCE = "/World/envs/env_0/Materials/style"
_PER_ENV_CLONE = "/World/envs/env_1/Materials/style"


def _homogeneous_plan(num_envs: int = 2) -> ClonePlan:
    return ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones(1, num_envs, dtype=torch.bool),
    )


@pytest.fixture
def per_env_sim(monkeypatch):
    """Fake SimulationContext exposing both the notify recorder and a settable clone plan."""
    fake_sim = SimpleNamespace(render_context=_RenderContextRecorder(), plan=_homogeneous_plan())
    fake_sim.get_clone_plan = lambda: fake_sim.plan
    monkeypatch.setattr(SimulationContext, "instance", classmethod(lambda cls: fake_sim))
    return fake_sim


def _per_env_material(stage, channels=("color",)) -> VisualMaterial:
    return VisualMaterial(
        VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Materials/style", spawn=PbrMdlCfg(), channels=channels)
    )


def _replicate_material(stage) -> None:
    """Copy the source material to env_1, standing in for the scene's replication pass."""
    stage.DefinePrim("/World/envs/env_1/Materials", "Scope")
    layer = stage.GetRootLayer()
    Sdf.CopySpec(layer, Sdf.Path(_PER_ENV_SOURCE), layer, Sdf.Path(_PER_ENV_CLONE))


def test_per_env_material_spawns_in_source_env_and_derives_clone_paths(stage, per_env_sim):
    material = _per_env_material(stage)

    assert material.is_per_env
    assert material.material_prim_path == _PER_ENV_SOURCE
    assert stage.GetPrimAtPath(_PER_ENV_SOURCE).IsValid()
    # clone paths come from the published clone plan, not from the stage (the clones may not exist yet)
    assert material.env_material_paths == [_PER_ENV_SOURCE, _PER_ENV_CLONE]
    assert material._env_shader_paths == [f"{_PER_ENV_SOURCE}/Shader", f"{_PER_ENV_CLONE}/Shader"]
    assert len(material.env_material_paths) == 2


def test_per_env_material_requires_covering_plan_row(stage, per_env_sim):
    material = _per_env_material(stage)

    per_env_sim.plan = ClonePlan(
        sources=("/World/envs/env_0/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="not replicated to every"):
        material.env_material_paths

    per_env_sim.plan = None
    with pytest.raises(RuntimeError, match="not available yet"):
        material.env_material_paths


def test_per_env_write_addresses_selected_envs(stage, per_env_sim):
    material = _per_env_material(stage)
    _replicate_material(stage)

    # full write: one row per environment
    VisualMaterial.write_channels([material], {"color": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])})
    # partial write: only env 1's clone is re-authored and notified
    VisualMaterial.write_channels([material], {"color": torch.tensor([[[0.9, 0.1, 0.1]]])}, env_ids=torch.tensor([1]))

    assert tuple(material.read_channel("color", env_id=0)) == pytest.approx((1.0, 0.0, 0.0))
    assert tuple(material.read_channel("color", env_id=1)) == pytest.approx((0.9, 0.1, 0.1))
    (full_paths, _shaders, attr_name, semantic, full_values) = per_env_sim.render_context.calls[0]
    assert full_paths == [_PER_ENV_SOURCE, _PER_ENV_CLONE]
    assert attr_name == "inputs:diffuse_color_constant"
    assert semantic == "color"
    assert torch.allclose(full_values, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    (partial_paths, partial_shaders, _attr, _semantic, partial_values) = per_env_sim.render_context.calls[1]
    assert partial_paths == [_PER_ENV_CLONE]
    assert partial_shaders == [f"{_PER_ENV_CLONE}/Shader"]
    assert torch.allclose(partial_values, torch.tensor([[0.9, 0.1, 0.1]]))


def test_per_env_write_swaps_textures_per_env(stage, per_env_sim):
    material = VisualMaterial(
        VisualMaterialCfg(
            prim_path="{ENV_REGEX_NS}/Materials/style",
            spawn=PbrMdlCfg(),
            channels=("texture",),
            texture_pool=("/textures/a.png", "/textures/b.png"),
        )
    )
    _replicate_material(stage)

    VisualMaterial.write_channels([material], {"texture": [["/textures/a.png", "/textures/b.png"]]})

    assert material.read_channel("texture", env_id=0).path == "/textures/a.png"
    assert material.read_channel("texture", env_id=1).path == "/textures/b.png"
    (material_paths, _shaders, attr_name, semantic, values) = per_env_sim.render_context.calls[-1]
    assert material_paths == [_PER_ENV_SOURCE, _PER_ENV_CLONE]
    assert attr_name == "inputs:diffuse_texture"
    assert semantic == "texture"
    assert values == ["/textures/a.png", "/textures/b.png"]


def _per_env_term_env(stage, num_envs: int = 2):
    entries = {"style": _per_env_material(stage)}
    scene = _Scene(entries)
    scene.num_envs = num_envs
    return SimpleNamespace(scene=scene), entries


def test_color_event_honors_env_ids_for_per_env_materials(stage, per_env_sim):
    env, _entries = _per_env_term_env(stage)
    _replicate_material(stage)
    cfg = _term_cfg([SceneEntityCfg("style")])

    term = visual_events.randomize_visual_color(cfg, env)
    term(env, torch.tensor([1]), **cfg.params)
    term(env, None, **cfg.params)
    term(env, slice(None), **cfg.params)

    # one notify row per selected environment: [1], all, all
    assert [len(call[0]) for call in per_env_sim.render_context.calls] == [1, 2, 2]


def test_material_event_honors_env_ids_for_per_env_materials(stage, per_env_sim):
    entries = {"style": _per_env_material(stage, channels=("color", "roughness"))}
    scene = _Scene(entries)
    scene.num_envs = 2
    env = SimpleNamespace(scene=scene)
    _replicate_material(stage)
    cfg = SimpleNamespace(
        params={
            "materials": [SceneEntityCfg("style")],
            "channels": {"color": ((0.1, 0.1, 0.1), (0.9, 0.9, 0.9)), "roughness": (0.1, 0.9)},
        }
    )

    term = visual_events.randomize_visual_material(cfg, env)
    term(env, torch.tensor([1]), **cfg.params)

    # two channels -> two writes, each carrying only env 1's clone
    assert len(per_env_sim.render_context.calls) == 2
    assert all(call[0] == [_PER_ENV_CLONE] for call in per_env_sim.render_context.calls)
    assert tuple(entries["style"].read_channel("color", env_id=0)) == pytest.approx((0.18, 0.18, 0.18))


def test_color_event_rejects_mixed_granularities(stage, per_env_sim):
    entries = {
        "style": _per_env_material(stage),
        "bucket_a": VisualMaterial(VisualMaterialCfg(prim_path="/World/Materials/bucket_a", spawn=PbrMdlCfg())),
    }
    scene = _Scene(entries)
    scene.num_envs = 2
    env = SimpleNamespace(scene=scene)
    cfg = _term_cfg([SceneEntityCfg("style"), SceneEntityCfg("bucket_a")])

    with pytest.raises(ValueError, match="Cannot mix"):
        visual_events.randomize_visual_color(cfg, env)
