# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Isaac RTX renderer output contract."""

from __future__ import annotations

import inspect
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch
import warp as wp
from packaging import version

from isaaclab.renderers import RenderBufferKind, RenderBufferSpec
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg

pytestmark = pytest.mark.isaacsim_ci

ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING = "/rtx/scenePartitioning/showAllPartitionsByDefault"


@pytest.fixture(autouse=True)
def active_simulation_context(monkeypatch):
    """Give directly constructed renderers their simulation-scoped clone registry."""
    from isaaclab.sim import SimulationContext

    context = object.__new__(SimulationContext)
    context.stage = object()
    context._backend_registry = {}
    context._backend_clone_roles = {}
    context._clone_plan = None
    monkeypatch.setattr(SimulationContext, "_instance", context)


def _install_omni_stubs(monkeypatch):
    omni_module = sys.modules.get("omni", types.ModuleType("omni"))
    replicator_module = types.ModuleType("omni.replicator")
    replicator_core_module = types.ModuleType("omni.replicator.core")
    syntheticdata_module = types.ModuleType("omni.syntheticdata")
    usd_module = MagicMock()

    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator_module)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", replicator_core_module)
    monkeypatch.setitem(sys.modules, "omni.syntheticdata", syntheticdata_module)
    monkeypatch.setitem(sys.modules, "omni.usd", usd_module)
    monkeypatch.setattr(omni_module, "replicator", replicator_module, raising=False)
    monkeypatch.setattr(omni_module, "syntheticdata", syntheticdata_module, raising=False)
    monkeypatch.setattr(omni_module, "usd", usd_module, raising=False)
    monkeypatch.setattr(replicator_module, "core", replicator_core_module, raising=False)

    return replicator_core_module, syntheticdata_module


def test_isaac_rtx_supported_output_types_include_rgb_hdr(monkeypatch):
    """Isaac RTX advertises RGB_HDR as a 3-channel float renderer output."""
    _install_omni_stubs(monkeypatch)
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

    renderer = IsaacRtxRenderer.__new__(IsaacRtxRenderer)
    renderer.cfg = IsaacRtxRendererCfg()
    with patch("isaaclab_physx.renderers.isaac_rtx_renderer.get_isaac_sim_version", return_value=version.parse("6.0")):
        specs = renderer.supported_output_types()

    assert specs[RenderBufferKind.RGB_HDR] == RenderBufferSpec(3, wp.float32)


def test_create_render_data_uses_unique_sdf_safe_render_product_name(monkeypatch):
    """Each tiled render product gets a fresh ``rp_<uuid4.hex>`` name.

    Unique names avoid collisions across concurrent tiled cameras and sequential
    create/destroy cycles in one Kit process (e.g. ``simple_shading_*`` pytest).
    uuid4 provides 122 random bits, so birthday-paradox collision chance among n
    names is ~n^2 / 2^123 — negligible for Isaac Lab workloads.
    """
    replicator_core_module, syntheticdata_module = _install_omni_stubs(monkeypatch)
    monkeypatch.setattr(syntheticdata_module, "SyntheticData", MagicMock(), raising=False)

    import isaaclab_physx.renderers.isaac_rtx_renderer as rtx_renderer
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

    from pxr import Sdf

    # Stub Kit settings / stage so create_render_data can run without Isaac Sim.
    # has_gui=False keeps the depth-only color-render branch inactive for rgb cameras.
    settings = MagicMock()
    settings.get.return_value = False
    stage = MagicMock()

    # Capture the ``name=`` kwarg passed to Replicator; the returned HydraTexture
    # and annotator registry only need to exist so create_render_data can finish.
    rp = MagicMock()
    rp.path = "/Render/rp_test"
    create_tiled = MagicMock(return_value=rp)
    annotator = MagicMock()
    registry = MagicMock()
    registry.get_annotator.return_value = annotator
    replicator_core_module.create = SimpleNamespace(render_product_tiled=create_tiled)
    replicator_core_module.AnnotatorRegistry = registry

    # Minimal CameraRenderSpec: one rgb tiled camera is enough to exercise naming.
    spec = SimpleNamespace(
        camera_prim_paths=["/World/envs/env_0/Camera"],
        device="cpu",
        cfg=CameraCfg(
            prim_path="/World/Camera",
            data_types=["rgb"],
            width=64,
            height=64,
            spawn=PinholeCameraCfg(),
            renderer_cfg=IsaacRtxRendererCfg(),
            isp_cfg=None,
        ),
    )
    renderer = rtx_renderer.IsaacRtxRenderer.__new__(rtx_renderer.IsaacRtxRenderer)
    renderer.cfg = IsaacRtxRendererCfg()
    renderer._stage = stage

    # Create many products with the same spec: names must still all differ (the
    # sequential simple_shading_* / multi-camera collision case this fix targets).
    num_names = 256
    names: list[str] = []
    with (
        patch.object(rtx_renderer, "get_settings_manager", return_value=settings),
        patch.object(rtx_renderer, "get_isaac_sim_version", return_value=version.parse("6.0")),
    ):
        for _ in range(num_names):
            renderer.create_render_data(spec)
            names.append(create_tiled.call_args.kwargs["name"])

    # Camera paths are clone-plan outputs, not prims for the renderer to rediscover or validate.
    stage.GetPrimAtPath.assert_not_called()

    # Every call must mint a distinct name — a reused default was the original bug.
    assert len(set(names)) == num_names
    for name in names:
        # Contract: ``rp_`` + uuid4().hex so the token is a valid USD identifier
        # (no hyphens) and cannot collide with path-derived names.
        assert name.startswith("rp_")
        hex_part = name.removeprefix("rp_")
        # uuid4().hex is 32 lowercase hex digits (128 bits; 122 of them random).
        assert len(hex_part) == 32
        assert all(c in "0123456789abcdef" for c in hex_part)
        # Replicator builds a USD prim from this name; reject illegal identifiers.
        assert Sdf.Path.IsValidIdentifier(name)
        assert Sdf.Path.IsValidPathString(f"/Render/{name}")


@pytest.mark.parametrize(
    ("data_types", "expected_shading_mode", "expected_minimal_render_mode"),
    [
        pytest.param(["simple_shading_constant_diffuse"], 1, True, id="constant_diffuse"),
        pytest.param(["simple_shading_diffuse_mdl"], 2, True, id="diffuse_mdl"),
        pytest.param(["simple_shading_full_mdl"], 3, True, id="full_mdl"),
        pytest.param(["rgb", "simple_shading_full_mdl"], 3, False, id="rgb_keeps_path_tracing"),
        pytest.param(["rgba", "simple_shading_full_mdl"], 3, False, id="rgba_keeps_path_tracing"),
        pytest.param(["rgb_hdr", "simple_shading_full_mdl"], 3, False, id="rgb_hdr_keeps_path_tracing"),
    ],
)
def test_simple_shading_configures_its_render_product(
    monkeypatch, data_types, expected_shading_mode, expected_minimal_render_mode
):
    """Simple shading must configure its product without altering requested color output.

    Selecting a shading level alone leaves the product in ``RealTimePathTracing`` and keeps the
    full path-tracing cost. Products without regular color output are switched to RTX Minimal;
    mixed color products retain path tracing. The shading level is authored on the render product
    rather than through process-wide carb settings.
    """
    replicator_core_module, syntheticdata_module = _install_omni_stubs(monkeypatch)
    monkeypatch.setattr(syntheticdata_module, "SyntheticData", MagicMock(), raising=False)

    import isaaclab_physx.renderers.isaac_rtx_renderer as rtx_renderer
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

    from pxr import Sdf, Usd, UsdGeom

    rp = MagicMock()
    rp.path = "/Render/OmniverseKit/HydraTextures/rp_test"
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Scope.Define(stage, rp.path)

    annotator = MagicMock()
    registry = MagicMock()
    registry.get_annotator.return_value = annotator
    replicator_core_module.create = SimpleNamespace(render_product_tiled=MagicMock(return_value=rp))
    replicator_core_module.AnnotatorRegistry = registry

    spec = SimpleNamespace(
        camera_prim_paths=["/World/envs/env_0/Camera"],
        device="cpu",
        cfg=SimpleNamespace(
            data_types=data_types,
            width=64,
            height=64,
            isp_cfg=None,
            background_color=None,
            colorize_semantic_segmentation=False,
            colorize_instance_segmentation=False,
            colorize_instance_id_segmentation=False,
        ),
    )
    renderer = rtx_renderer.IsaacRtxRenderer.__new__(rtx_renderer.IsaacRtxRenderer)
    renderer.cfg = IsaacRtxRendererCfg()
    renderer._stage = stage

    with patch.object(rtx_renderer, "get_isaac_sim_version", return_value=version.parse("6.0")):
        renderer.create_render_data(spec)

    layer = stage.GetSessionLayer()
    mode = layer.GetAttributeAtPath(Sdf.Path(rp.path).AppendProperty("omni:rtx:minimal:mode"))
    assert mode.default == expected_shading_mode
    assert mode.typeName == Sdf.ValueTypeNames.Int
    render_mode = layer.GetAttributeAtPath(Sdf.Path(rp.path).AppendProperty("omni:rtx:rendermode"))
    if expected_minimal_render_mode:
        assert render_mode.default == "Minimal"
        assert render_mode.typeName == Sdf.ValueTypeNames.Token
    else:
        assert render_mode is None


def test_simple_shading_rejects_multiple_modes(monkeypatch):
    """One render product cannot request conflicting simple-shading modes."""
    _install_omni_stubs(monkeypatch)
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer

    renderer = IsaacRtxRenderer.__new__(IsaacRtxRenderer)
    spec = SimpleNamespace(
        cfg=SimpleNamespace(data_types=["simple_shading_constant_diffuse", "simple_shading_full_mdl"])
    )
    with pytest.raises(ValueError, match="Multiple simple shading modes"):
        renderer._resolve_simple_shading_mode(spec)


def test_render_product_uuid_name_format_is_sdf_safe():
    """``rp_{uuid4().hex}`` matches the create_render_data naming contract and is SDF-safe."""
    import uuid

    from pxr import Sdf

    names = [f"rp_{uuid.uuid4().hex}" for _ in range(64)]
    assert len(set(names)) == len(names)
    for name in names:
        assert name.startswith("rp_")
        hex_part = name.removeprefix("rp_")
        assert len(hex_part) == 32
        int(hex_part, 16)  # raises if not hex
        assert "-" not in name
        assert Sdf.Path.IsValidIdentifier(name)
        assert Sdf.Path.IsValidPathString(f"/Render/{name}")


def test_prepare_cameras_normalizes_explicit_cfg_and_targets_plan_prototypes(monkeypatch):
    """PPISP is explicit while RTX overrides target clone-plan prototypes."""
    _install_omni_stubs(monkeypatch)
    import isaaclab_physx.renderers.isaac_rtx_renderer as rtx_renderer

    resolved_isp = object()
    normalize = MagicMock(return_value=resolved_isp)
    apply = MagicMock()
    ppisp = types.ModuleType("isaaclab_ppisp")
    ppisp.normalize_ppisp_cfg = normalize
    ppisp.apply_rtx_exposure_overrides = apply
    monkeypatch.setitem(sys.modules, "isaaclab_ppisp", ppisp)

    renderer = rtx_renderer.IsaacRtxRenderer.__new__(rtx_renderer.IsaacRtxRenderer)
    renderer._camera_prim_paths = []
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

    stage = MagicMock()
    isp_cfg = object()
    spec = SimpleNamespace(
        camera_source_prim_paths=("/World/prototypes/red/Camera", "/World/prototypes/blue/Camera"),
        camera_prim_paths=("/Scene/worlds/world_2/Camera", "/Scene/worlds/world_7/Camera"),
        cfg=CameraCfg(
            prim_path="/World/Camera",
            data_types=[],
            width=1,
            height=1,
            spawn=PinholeCameraCfg(),
            renderer_cfg=IsaacRtxRendererCfg(),
            isp_cfg=isp_cfg,
        ),
    )

    settings = MagicMock()
    configured_isp = spec.cfg.isp_cfg
    with patch.object(rtx_renderer, "get_settings_manager", return_value=settings):
        renderer.prepare_cameras(stage, spec)

    assert renderer._camera_prim_paths == list(spec.camera_prim_paths)
    settings.set_bool.assert_called_once_with("/rtx/rtpt/gaussian/skipTonemapping/enabled", False)
    normalize.assert_called_once_with(configured_isp)
    apply.assert_called_once_with(stage, list(spec.camera_source_prim_paths))
    stage.GetPrimAtPath.assert_not_called()


def test_plan_owned_scene_data_is_not_rediscovered_per_frame(monkeypatch):
    """Architecture gate: camera, scene data, and env ownership never regress to global discovery."""
    _install_omni_stubs(monkeypatch)
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer

    prepare_stage = inspect.getsource(IsaacRtxRenderer.prepare_stage)
    create_render_data = inspect.getsource(IsaacRtxRenderer.create_render_data)
    update = inspect.getsource(IsaacRtxRenderer.update)
    assert all(lookup not in update for lookup in ("SimulationContext.instance()", "get_clone_plan()"))
    assert all(stage_write not in update for stage_write in ("Sdf.", "Gf.", "GetRootLayer", ".default", ".Set("))
    main = update.index("request_transforms(SceneDataFormat.FabricMatrix44)")
    first_hierarchy = update.index("_update_fabric_hierarchy()", main)
    camera = update.index("SceneDataFormat.FabricMatrix44, name=render_data.spec.cfg.prim_path")
    second_hierarchy = update.index("_update_fabric_hierarchy()", first_hierarchy + 1)
    assert main < first_hierarchy < camera < second_hierarchy
    assert all(discovery not in prepare_stage for discovery in ("GetPrimAtPath", "PrimRange", "Traverse"))
    assert "under(" not in prepare_stage
    assert "GetPrefixes()" in prepare_stage
    assert "ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION" not in prepare_stage
    assert "isaac_rtx_per_env_scene_partition_enabled" not in prepare_stage
    assert all(discovery not in create_render_data for discovery in ("GetPrimAtPath", "PrimRange", "Traverse"))
    assert "UsdGeom.Camera" not in create_render_data


def test_prepare_stage_partitions_only_plan_owned_camera_paths(monkeypatch):
    """Scene partitions use clone-plan env roots and registered cameras, ignoring USD siblings."""
    _install_omni_stubs(monkeypatch)
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer

    from pxr import Usd, UsdGeom

    from isaaclab.cloner import ClonePlan

    stage = Usd.Stage.CreateInMemory()
    camera_orders = {}
    for env_id in (2, 7):
        root = f"/Scene/worlds/world_{env_id}"
        UsdGeom.Xform.Define(stage, root)
        camera = UsdGeom.Camera.Define(stage, f"{root}/Camera")
        xformable = UsdGeom.Xformable(camera)
        xformable.AddTranslateOp()
        xformable.AddOrientOp()
        xformable.AddScaleOp()
        camera_orders[camera.GetPath().pathString] = tuple(op.GetOpName() for op in xformable.GetOrderedXformOps())
        UsdGeom.Camera.Define(stage, f"{root}/UnplannedCamera")
    plan = ClonePlan(
        sources=("/Scene/worlds/world_2/Robot",),
        destinations=("/Scene/worlds/world_{}/Robot",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
        env_ids=torch.tensor([2, 7]),
    )
    renderer = IsaacRtxRenderer.__new__(IsaacRtxRenderer)
    renderer._camera_prim_paths = [f"/Scene/worlds/world_{env_id}/Camera" for env_id in (2, 7)]

    renderer.prepare_stage(stage, plan)

    for env_id in (2, 7):
        root = f"/Scene/worlds/world_{env_id}"
        token = f"world_{env_id}"
        assert stage.GetPrimAtPath(root).GetAttribute("primvars:omni:scenePartition").Get() == token
        camera = stage.GetPrimAtPath(f"{root}/Camera")
        assert camera.GetAttribute("omni:scenePartition").Get() == token
        assert (
            tuple(op.GetOpName() for op in UsdGeom.Xformable(camera).GetOrderedXformOps())
            == camera_orders[f"{root}/Camera"]
        )
        assert not camera.HasAttribute("xformOp:transform")
        assert not stage.GetPrimAtPath(f"{root}/UnplannedCamera").HasAttribute("omni:scenePartition")


def test_render_rejects_unset_output_sinks(monkeypatch):
    """A requested frame cannot disappear because camera output binding was skipped."""
    _install_omni_stubs(monkeypatch)
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer

    renderer = IsaacRtxRenderer.__new__(IsaacRtxRenderer)
    render_data = SimpleNamespace(spec=object(), output_data=None)

    with pytest.raises(RuntimeError, match="outputs must be set"):
        renderer.render(render_data)


def test_render_applies_each_renderers_semantic_filter(monkeypatch):
    """Shared SyntheticData state follows the camera making the segmentation request."""
    _, syntheticdata_module = _install_omni_stubs(monkeypatch)
    synthetic_data = MagicMock()
    syntheticdata_module.SyntheticData = synthetic_data
    import isaaclab_physx.renderers.isaac_rtx_renderer as rtx_renderer

    renderer = rtx_renderer.IsaacRtxRenderer.__new__(rtx_renderer.IsaacRtxRenderer)
    renderer.cfg = SimpleNamespace(semantic_filter=["class", "material"])
    spec = SimpleNamespace(cfg=SimpleNamespace(data_types=["instance_segmentation"]))
    render_data = SimpleNamespace(spec=spec, output_data={})

    with (
        patch.object(rtx_renderer, "ensure_isaac_rtx_render_update", side_effect=RuntimeError("stop")),
        pytest.raises(RuntimeError, match="stop"),
    ):
        renderer.render(render_data)

    synthetic_data.Get.return_value.set_instance_mapping_semantic_filter.assert_called_once_with("class:*; material:*")


_MISSING = object()


def test_init_shares_one_usd_context_and_rtx_runtime(monkeypatch):
    """Camera clients share one scene clone context and one process-global RTX runtime."""
    _install_omni_stubs(monkeypatch)
    import isaaclab_physx.renderers.isaac_rtx_renderer as rtx_renderer
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

    from isaaclab.cloner import UsdReplicateContext
    from isaaclab.sim import SimulationContext

    call_order = []
    settings = MagicMock()
    settings.get.return_value = False

    with (
        patch.object(rtx_renderer, "get_settings_manager", return_value=settings),
        patch.object(rtx_renderer, "enable_extension", side_effect=lambda _name: call_order.append("enable")),
        patch.object(
            rtx_renderer, "apply_isaac_rtx_global_settings", side_effect=lambda *_args: call_order.append("settings")
        ),
        patch.object(rtx_renderer, "ensure_rtx_hydra_engine_attached"),
    ):
        first = rtx_renderer.IsaacRtxRenderer(IsaacRtxRendererCfg())
        second = rtx_renderer.IsaacRtxRenderer(IsaacRtxRendererCfg())

    registry = SimulationContext.instance()._backend_registry
    assert first._clone_ctx is second._clone_ctx is registry[UsdReplicateContext]
    assert type(registry[rtx_renderer._IsaacRtxRuntime]) is rtx_renderer._IsaacRtxRuntime
    assert set(registry) == {UsdReplicateContext, rtx_renderer._IsaacRtxRuntime}
    assert call_order == ["enable", "settings"]


def test_init_rejects_conflicting_process_global_settings(monkeypatch):
    """All camera clients must agree on the settings of their shared RTX runtime."""
    _install_omni_stubs(monkeypatch)
    import isaaclab_physx.renderers.isaac_rtx_renderer as rtx_renderer
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

    settings = MagicMock()
    settings.get.return_value = False
    with (
        patch.object(rtx_renderer, "get_settings_manager", return_value=settings),
        patch.object(rtx_renderer, "enable_extension"),
        patch.object(rtx_renderer, "apply_isaac_rtx_global_settings"),
        patch.object(rtx_renderer, "ensure_rtx_hydra_engine_attached"),
    ):
        rtx_renderer.IsaacRtxRenderer(IsaacRtxRendererCfg())
        conflicting = IsaacRtxRendererCfg()
        conflicting.global_settings.enable_reflections = True
        with pytest.raises(ValueError, match="process-global"):
            rtx_renderer.IsaacRtxRenderer(conflicting)


@pytest.mark.parametrize("configured_value", [None, False, True])
def test_init_applies_only_explicit_global_spectator_view_setting(monkeypatch, configured_value):
    """Isaac RTX should preserve launch intent unless the renderer config overrides it."""
    _install_omni_stubs(monkeypatch)
    import isaaclab_physx.renderers.isaac_rtx_renderer as rtx_renderer
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import (
        IsaacRtxRendererCfg,
        IsaacRtxRendererGlobalSettingsCfg,
    )

    settings = MagicMock()
    settings.get.return_value = False
    with (
        patch.object(rtx_renderer, "get_settings_manager", return_value=settings),
        patch.object(rtx_renderer, "enable_extension"),
        patch.object(rtx_renderer, "ensure_rtx_hydra_engine_attached"),
    ):
        rtx_renderer.IsaacRtxRenderer(
            IsaacRtxRendererCfg(
                global_settings=IsaacRtxRendererGlobalSettingsCfg(
                    show_all_partitions_by_default=configured_value,
                )
            )
        )

    spectator_calls = [
        setting_call
        for setting_call in settings.set.call_args_list
        if setting_call.args[0] == ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING
    ]
    expected_calls = (
        [] if configured_value is None else [call(ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING, configured_value)]
    )
    assert spectator_calls == expected_calls


@pytest.mark.parametrize(
    ("stored", "expected_called"),
    [
        pytest.param(True, True, id="deterministic-true-applies-settings"),
        pytest.param(False, False, id="deterministic-false-skips-settings"),
        pytest.param(_MISSING, False, id="deterministic-missing-skips-settings"),
    ],
)
def test_deterministic_flag_gates_rtx_determinism_settings(monkeypatch, stored, expected_called):
    """IsaacRtxRenderer applies RTX determinism settings only when ``/isaaclab/render/deterministic`` is true."""
    _install_omni_stubs(monkeypatch)
    import isaaclab_physx.renderers.isaac_rtx_renderer as rtx_renderer
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

    # RTX rendering requires cameras to be enabled.
    settings_values = {"/isaaclab/cameras_enabled": True}
    if stored is not _MISSING:
        settings_values["/isaaclab/render/deterministic"] = stored

    settings = MagicMock()
    settings.get.side_effect = settings_values.get
    determinism_mock = MagicMock()

    with (
        patch.object(rtx_renderer, "get_settings_manager", return_value=settings),
        patch.object(rtx_renderer, "enable_extension"),
        patch.object(rtx_renderer, "apply_isaac_rtx_global_settings"),
        patch.object(rtx_renderer, "apply_isaac_rtx_determinism_settings", determinism_mock),
        patch.object(rtx_renderer, "ensure_rtx_hydra_engine_attached"),
    ):
        rtx_renderer.IsaacRtxRenderer(IsaacRtxRendererCfg())

    assert determinism_mock.called is expected_called
    if expected_called:
        determinism_mock.assert_called_once_with(settings)


def test_isaac_rtx_read_output_clears_stale_metadata_and_keeps_seeded_keys(monkeypatch):
    """read_output replaces (not merges): a dropped annotator info resets its info entry, seeded keys persist."""
    _install_omni_stubs(monkeypatch)
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer

    from isaaclab.sensors.camera.camera_data import CameraData

    renderer = IsaacRtxRenderer.__new__(IsaacRtxRenderer)

    # ``camera_data.info`` is seeded with one key per output (mirrors ``camera_data.output``); both start None.
    camera_data = CameraData()
    camera_data.info = {"rgb": None, "semantic_segmentation": None}

    # Frame 1: the segmentation annotator emits metadata, so its info lands in camera_data.info.
    id_to_labels = {"2": {"class": "cartpole"}}
    render_data = SimpleNamespace(renderer_info={"semantic_segmentation": {"idToLabels": id_to_labels}})
    renderer.read_output(render_data, camera_data)
    assert camera_data.info["semantic_segmentation"] == {"idToLabels": id_to_labels}

    # Frame 2: the annotator emits no info (``renderer_info`` value is None or the key is gone).
    render_data = SimpleNamespace(renderer_info={"semantic_segmentation": None})
    renderer.read_output(render_data, camera_data)

    # The stale idToLabels must be cleared, and the seeded keys (rgb, semantic_segmentation) must remain.
    assert camera_data.info == {"rgb": None, "semantic_segmentation": None}


def test_isaac_rtx_publishes_fabric_visual_material_writer(monkeypatch):
    """Isaac RTX exposes the shared USD resource's Fabric writer factory."""
    _install_omni_stubs(monkeypatch)
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer

    class Resource:
        def create_fabric_visual_material_writer(self, batches):
            return batches

    resource = Resource()
    renderer = IsaacRtxRenderer.__new__(IsaacRtxRenderer)
    renderer._clone_ctx = resource

    assert renderer.visual_material_writer == resource.create_fabric_visual_material_writer
