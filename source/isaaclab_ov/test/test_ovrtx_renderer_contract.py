# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX renderer output contract."""

import importlib.util
import inspect
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import PinholeCameraCfg

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.cloner import OvReplicateContext  # noqa: E402
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers import ovrtx_renderer as ovrtx_renderer_module  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderData, OVRTXRenderer  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_scene import OvrtxScene  # noqa: E402
    from ovrtx import BindingFlag, DataAccess, PrimMode, Semantic  # noqa: E402
else:
    OvReplicateContext = None
    OVRTXRenderData = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    ovrtx_renderer_module = None
    OvrtxScene = object

_SPAWN = PinholeCameraCfg(
    focal_length=24.0,
    focus_distance=400.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 1.0e5),
)


def _make_camera_cfg(data_types: list[str]) -> CameraCfg:
    return CameraCfg(
        height=8,
        width=16,
        prim_path="/World/Camera",
        spawn=_SPAWN,
        data_types=data_types,
        renderer_cfg=OVRTXRendererCfg(),
    )


def _make_ovrtx_render_data() -> OVRTXRenderData:
    rd = OVRTXRenderData.__new__(OVRTXRenderData)
    rd.width = 16
    rd.height = 8
    rd.num_envs = 2
    rd.warp_buffers = {}
    rd.renderer_info = {}
    rd.ppisp_pipeline = None
    return rd


def _make_ovrtx_renderer_without_backend() -> OVRTXRenderer:
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    renderer._clone_ctx = SimpleNamespace(
        scene=None,
        _update_ovrtx=MagicMock(side_effect=RuntimeError("OVRTX updates require an initialized scene-data provider.")),
        _remove_renderer=MagicMock(),
    )
    renderer._client_id = 0
    renderer._spec = None
    renderer._device = "cpu"
    renderer._camera_xforms = None
    renderer._render_product_paths = []
    renderer._render_product_usd = ""
    renderer._output_id_color_buffers = {}
    renderer._initialized_scene = False
    renderer._render_var_keys = dict(ovrtx_renderer_module.RENDER_VAR_FRAME_KEYS)
    return renderer


def test_ovrtx_native_renderer_uses_the_camera_cuda_device(monkeypatch):
    """Compatible cameras share the first camera's native resource; incompatible cfgs fail."""
    configs = []
    scenes = []
    simulation = object.__new__(ovrtx_renderer_module.SimulationContext)
    simulation.stage = object()
    simulation._backend_registry = {}
    simulation._backend_clone_roles = {}
    simulation._clone_plan = None
    monkeypatch.setattr(ovrtx_renderer_module.SimulationContext, "instance", lambda: simulation)

    class Backend:
        def __init__(self, config):
            configs.append(config)

    monkeypatch.setattr(ovrtx_renderer_module, "Renderer", Backend)
    device = SimpleNamespace(is_cuda=True, ordinal=1)
    monkeypatch.setattr(ovrtx_renderer_module.wp, "get_device", lambda _device: device)
    monkeypatch.setattr(
        ovrtx_renderer_module,
        "OvrtxScene",
        lambda backend: scenes.append(backend) or object(),
    )

    renderers = [OVRTXRenderer(OVRTXRendererCfg()), OVRTXRenderer(OVRTXRendererCfg())]
    assert configs == []

    for index, renderer in enumerate(renderers):
        renderer.prepare_cameras(
            object(),
            SimpleNamespace(
                device="cuda:1",
                num_instances=1,
                camera_source_prim_paths=(f"/World/source/Camera_{index}",),
                camera_prim_paths=(f"/World/envs/env_0/Camera_{index}",),
                cfg=_make_camera_cfg(["rgb"]),
            ),
        )

    assert len(configs) == 1
    assert configs[0].active_cuda_gpus == "1"
    assert len(scenes) == 1
    assert renderers[0]._clone_ctx.scene is renderers[1]._clone_ctx.scene

    incompatible = OVRTXRenderer(OVRTXRendererCfg(log_level="info"))
    with pytest.raises(ValueError, match="same CUDA device, native renderer configuration"):
        incompatible.prepare_cameras(
            object(),
            SimpleNamespace(
                device="cuda:1",
                num_instances=1,
                camera_source_prim_paths=("/World/source/Camera_2",),
                camera_prim_paths=("/World/envs/env_0/Camera_2",),
                cfg=_make_camera_cfg(["rgb"]),
            ),
        )
    assert len(configs) == 1


def test_prepare_cameras_normalizes_explicit_cfg_and_targets_plan_prototypes(monkeypatch):
    """OVRTX normalizes explicit PPISP and overrides clone-plan prototypes."""
    resolved_isp = object()
    normalize = MagicMock(return_value=resolved_isp)
    apply = MagicMock()
    ppisp = types.ModuleType("isaaclab_ppisp")
    ppisp.normalize_ppisp_cfg = normalize
    ppisp.apply_rtx_exposure_overrides = apply
    monkeypatch.setitem(sys.modules, "isaaclab_ppisp", ppisp)

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._spec = None
    renderer._clone_ctx._configure_ovrtx = MagicMock()
    stage = MagicMock()
    isp_cfg = object()
    spec = SimpleNamespace(
        device="cuda:0",
        num_instances=2,
        camera_source_prim_paths=("/World/prototypes/red/Camera", "/World/prototypes/blue/Camera"),
        cfg=SimpleNamespace(
            isp_cfg=isp_cfg,
            data_types=["rgb"],
            width=16,
            height=8,
            background_color=None,
        ),
    )
    monkeypatch.setattr(
        ovrtx_renderer_module.wp, "get_device", lambda _device: SimpleNamespace(is_cuda=True, ordinal=0)
    )
    renderer.prepare_cameras(stage, spec)

    normalize.assert_called_once_with(isp_cfg)
    apply.assert_called_once_with(stage, list(spec.camera_source_prim_paths))
    stage.GetPrimAtPath.assert_not_called()
    renderer._clone_ctx._configure_ovrtx.assert_called_once()


def test_prepare_cameras_rejects_a_second_spec():
    """An OVRTX renderer never silently keeps the first of two camera specifications."""
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer._spec = object()

    with pytest.raises(RuntimeError, match="exactly one CameraRenderSpec"):
        renderer.prepare_cameras(object(), object())


def test_planned_ovrtx_camera_paths_are_not_rediscovered_from_usd():
    """Architecture gate: camera ownership never regresses to renderer-side USD discovery."""
    source = inspect.getsource(OVRTXRenderer.prepare_cameras)
    assert all(discovery not in source for discovery in ("GetPrimAtPath", "PrimRange", "Traverse"))


def test_ovrtx_camera_delegates_its_named_transform_write_to_the_shared_context():
    """The context owns publication-generation gating for every OVRTX write."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._clone_ctx._update_ovrtx = MagicMock()
    renderer._camera_xforms = object()

    renderer.update(SimpleNamespace(transform_stream="camera"), object())

    renderer._clone_ctx._update_ovrtx.assert_called_once_with(renderer._camera_xforms, "camera")


def test_ovrtx_two_camera_clients_write_each_clean_generation_exactly_once():
    """Two clients share object writes while independently gating their named camera streams."""

    class Provider:
        def __init__(self):
            self.generations = {None: 1, "camera_a": 1, "camera_b": 1}

        def request_transforms(self, _output_format, name=None):
            return SimpleNamespace(matrices=f"matrices:{name}")

        def transform_generation(self, name=None):
            return self.generations[name]

    scene = SimpleNamespace(transform_format=OvrtxScene.transform_format, write_xforms=MagicMock())
    context = OvReplicateContext.__new__(OvReplicateContext)
    context._ovrtx_scene = scene
    context._scene_data_provider = Provider()
    context._object_xforms = "objects"
    context._point_bindings = {}
    context._sdp_transform_generation = -1
    context._sdp_camera_transform_generations = {}
    context._sdp_point_generations = {}
    renderers = [_make_ovrtx_renderer_without_backend(), _make_ovrtx_renderer_without_backend()]
    for renderer, stream in zip(renderers, ("camera_a", "camera_b"), strict=True):
        renderer._clone_ctx = context
        renderer._camera_xforms = stream
        renderer.update(SimpleNamespace(transform_stream=stream), object())
        renderer.update(SimpleNamespace(transform_stream=stream), object())

    assert scene.write_xforms.call_args_list == [
        call("objects", "matrices:None"),
        call("camera_a", "matrices:camera_a"),
        call("camera_b", "matrices:camera_b"),
    ]


def test_ovrtx_update_requires_sdp_and_clone_time_camera_bindings() -> None:
    renderer = _make_ovrtx_renderer_without_backend()
    render_data = SimpleNamespace(transform_stream="camera")
    renderer._camera_xforms = object()
    with pytest.raises(RuntimeError, match="initialized scene-data provider"):
        renderer.update(render_data, object())
    renderer._camera_xforms = None
    renderer._clone_ctx._update_ovrtx = MagicMock()
    with pytest.raises(RuntimeError, match="clone-time camera bindings"):
        renderer.update(render_data, object())
    renderer._clone_ctx._update_ovrtx.assert_not_called()


def test_ovrtx_initialize_requires_a_camera_specification() -> None:
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._spec = None

    with pytest.raises(RuntimeError, match="before a camera supplies"):
        renderer.initialize()


def test_ovrtx_render_rejects_missing_native_prerequisites_and_frames() -> None:
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._initialized_scene = True
    renderer._render_product_paths = []
    render_data = _make_ovrtx_render_data()

    with pytest.raises(RuntimeError, match="without its render product"):
        renderer.render(render_data)

    renderer._render_product_paths = ["/Render/Product"]
    renderer._clone_ctx._render_ovrtx = lambda _paths: {}
    with pytest.raises(RuntimeError, match="produced no frame"):
        renderer.render(render_data)


def test_ovrtx_supported_output_types_key_set():
    """OVRTX publishes the documented key set and per-output spec."""
    renderer = _make_ovrtx_renderer_without_backend()
    specs = renderer.supported_output_types()

    assert set(specs.keys()) == {
        RenderBufferKind.RGB,
        RenderBufferKind.RGBA,
        RenderBufferKind.RGB_HDR,
        RenderBufferKind.ALBEDO,
        RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE,
        RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL,
        RenderBufferKind.SIMPLE_SHADING_FULL_MDL,
        RenderBufferKind.SEMANTIC_SEGMENTATION,
        RenderBufferKind.INSTANCE_SEGMENTATION,
        RenderBufferKind.DEPTH,
        RenderBufferKind.DISTANCE_TO_IMAGE_PLANE,
        RenderBufferKind.DISTANCE_TO_CAMERA,
        RenderBufferKind.NORMALS,
        RenderBufferKind.MOTION_VECTORS,
    }
    assert specs[RenderBufferKind.RGBA] == RenderBufferSpec(4, wp.uint8)
    assert specs[RenderBufferKind.RGB_HDR] == RenderBufferSpec(3, wp.float32)
    assert specs[RenderBufferKind.DEPTH] == RenderBufferSpec(1, wp.float32)
    assert specs[RenderBufferKind.MOTION_VECTORS] == RenderBufferSpec(2, wp.float32)


def test_ovrtx_set_outputs_wraps_caller_torch_zero_copy():
    """OVRTXRenderer.set_outputs publishes warp views over the caller's warp storage."""
    renderer = _make_ovrtx_renderer_without_backend()

    if not torch.cuda.is_available():
        pytest.skip("OVRTX zero-copy wrapping requires a CUDA device")
    device = "cuda"

    cfg = _make_camera_cfg(["rgb", "rgba", "depth"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device=device,
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_render_data()
    renderer.set_outputs(render_data, data.output)

    assert set(render_data.warp_buffers.keys()) >= {"rgba", "depth"}
    assert render_data.warp_buffers["rgba"].ptr == data.output["rgba"].warp.ptr
    assert render_data.warp_buffers["depth"].ptr == data.output["depth"].warp.ptr
    assert "rgb" not in render_data.warp_buffers


def test_ovrtx_set_outputs_wraps_requested_rgb_hdr_output():
    """OVRTXRenderer.set_outputs publishes a zero-copy view for requested RGB_HDR."""
    renderer = _make_ovrtx_renderer_without_backend()

    if not torch.cuda.is_available():
        pytest.skip("OVRTX zero-copy wrapping requires a CUDA device")
    device = "cuda"

    cfg = _make_camera_cfg(["rgb_hdr"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device=device,
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_render_data()
    renderer.set_outputs(render_data, data.output)

    assert render_data.warp_buffers["rgb_hdr"].ptr == data.output["rgb_hdr"].warp.ptr


def test_ovrtx_set_outputs_routes_ppisp_buffers_through_warp_buffers():
    """OVRTXRenderer.set_outputs stores PPISP source/destination in warp_buffers."""
    renderer = _make_ovrtx_renderer_without_backend()

    cfg = _make_camera_cfg(["rgb"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device="cpu",
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_render_data()
    render_data.ppisp_pipeline = object()
    renderer.set_outputs(render_data, data.output)

    assert render_data.warp_buffers["rgba"].ptr == data.output["rgba"].warp.ptr
    assert "rgb_hdr" in render_data.warp_buffers
    assert render_data.warp_buffers["rgb_hdr"].shape == (2, 8, 16, 3)
    assert render_data.warp_buffers["rgb_hdr"].dtype is wp.float32


def test_ovrtx_process_frame_skips_ldr_rgba_when_ppisp_is_active():
    """PPISP owns RGBA output, so OVRTX LdrColor should not pre-fill it."""

    class FailingRenderVar:
        def map(self, *args, **kwargs):
            raise AssertionError("PPISP RGBA output must not read OVRTX LdrColor")

    class Frame:
        render_vars = {"LdrColor": FailingRenderVar()}

    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    render_data.ppisp_pipeline = object()

    renderer._process_render_frame(render_data, Frame(), {"rgba": object()})


def test_ovrtx_ppisp_hdr_source_is_cloned_to_output_device(monkeypatch):
    """PPISP HdrColor source is moved to the HDR output buffer device."""

    class FakeArray:
        device = "cuda:1"

    class OutputArray:
        device = "cuda:0"

    cloned = object()
    clone_calls = []

    def fake_clone(src, *, device):
        clone_calls.append((src, device))
        return cloned

    monkeypatch.setattr(wp, "clone", fake_clone)

    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    render_data.ppisp_pipeline = object()
    source = FakeArray()

    assert renderer._prepare_ppisp_hdr_source(render_data, source, {"rgb_hdr": OutputArray()}) is cloned
    assert clone_calls == [(source, "cuda:0")]


class _FakeArray:
    def __init__(self, shape, dtype=wp.float32):
        self.shape = shape
        self.dtype = dtype


def test_process_frame_extracts_each_requested_pixel_aov_once(monkeypatch):
    """Every authored pixel AOV reaches each matching output buffer in one extraction."""

    class RenderVar:
        def __init__(self, value):
            self.value = value

        def map(self, *, device, sync_stream):
            assert device == ovrtx_renderer_module.Device.CUDA
            assert sync_stream == 17
            return nullcontext(self.value)

    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    sources = {
        "LdrColor": _FakeArray((8, 16, 4), wp.uint8),
        "DistanceToImagePlaneSD": _FakeArray((8, 16, 1)),
        "DistanceToCameraSD": _FakeArray((8, 16, 1)),
        "DiffuseAlbedoSD": _FakeArray((8, 16, 4), wp.uint8),
        "HdrColor": _FakeArray((8, 16, 4)),
        "NormalSD": _FakeArray((8, 16, 4)),
        "TargetMotionSD": _FakeArray((8, 16, 4)),
    }
    outputs = {
        "rgba": _FakeArray((2, 8, 16, 4), wp.uint8),
        "simple_shading_diffuse_mdl": _FakeArray((2, 8, 16, 3), wp.uint8),
        "depth": _FakeArray((2, 8, 16, 1)),
        "distance_to_image_plane": _FakeArray((2, 8, 16, 1)),
        "distance_to_camera": _FakeArray((2, 8, 16, 1)),
        "albedo": _FakeArray((2, 8, 16, 4), wp.uint8),
        "rgb_hdr": _FakeArray((2, 8, 16, 3)),
        "normals": _FakeArray((2, 8, 16, 3)),
        "motion_vectors": _FakeArray((2, 8, 16, 2)),
    }
    calls = []
    monkeypatch.setattr(ovrtx_renderer_module, "_gpu_side_render_var_sync_enabled", lambda: True)
    monkeypatch.setattr(wp, "get_stream", lambda _device: SimpleNamespace(cuda_stream=17))
    monkeypatch.setattr(wp, "from_dlpack", lambda value: value)
    monkeypatch.setattr(
        renderer, "_launch_extract_all_tiles", lambda _rd, source, output: calls.append((source, output))
    )

    renderer._process_render_frame(
        render_data,
        SimpleNamespace(render_vars={name: RenderVar(source) for name, source in sources.items()}),
        outputs,
    )

    assert calls == [
        (sources["LdrColor"], outputs["rgba"]),
        (sources["LdrColor"], outputs["simple_shading_diffuse_mdl"]),
        (sources["DistanceToImagePlaneSD"], outputs["depth"]),
        (sources["DistanceToImagePlaneSD"], outputs["distance_to_image_plane"]),
        (sources["DistanceToCameraSD"], outputs["distance_to_camera"]),
        (sources["DiffuseAlbedoSD"], outputs["albedo"]),
        (sources["HdrColor"], outputs["rgb_hdr"]),
        (sources["NormalSD"], outputs["normals"]),
        (sources["TargetMotionSD"], outputs["motion_vectors"]),
    ]


def test_launch_extract_all_tiles_rejects_wider_output_channels():
    """An output wider than the tiled input would read out of bounds, so it must raise before launching."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    render_data = _make_ovrtx_render_data()

    with pytest.raises(ValueError, match="out of bounds"):
        renderer._launch_extract_all_tiles(render_data, _FakeArray((8, 16, 3)), _FakeArray((2, 8, 16, 4)))


def test_launch_extract_all_tiles_launches_kernel_when_channels_are_compatible(monkeypatch):
    """Equal or narrower output channel counts pass validation and reach the kernel launch."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    render_data = _make_ovrtx_render_data()
    render_data.num_cols = 2

    launch_calls = []
    monkeypatch.setattr(wp, "launch", lambda **kwargs: launch_calls.append(kwargs))

    tiled_buffer = _FakeArray((8, 16, 4))
    output_buffer = _FakeArray((2, 8, 16, 3))
    renderer._launch_extract_all_tiles(render_data, tiled_buffer, output_buffer)

    assert len(launch_calls) == 1
    assert launch_calls[0]["inputs"][:2] == [tiled_buffer, output_buffer]


def test_ovrtx_read_output_copies_no_pixel_data():
    """OVRTXRenderer.read_output copies no pixel data; with empty renderer_info it leaves info untouched."""
    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    camera_data = CameraData()
    camera_data.info = {}
    camera_data._output = {}

    result = renderer.read_output(render_data, camera_data)
    assert result is None
    assert render_data.warp_buffers == {}
    assert camera_data.info == {}
    assert camera_data.output == {}


def test_ovrtx_read_output_forwards_renderer_info():
    """OVRTXRenderer.read_output forwards render_data.renderer_info (e.g. semantic idToLabels) into info."""
    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    id_to_labels = {"2": {"class": "cartpole"}}
    render_data.renderer_info = {"semantic_segmentation": {"idToLabels": id_to_labels}}

    camera_data = CameraData()
    camera_data.info = {"semantic_segmentation": None}
    camera_data._output = {}

    renderer.read_output(render_data, camera_data)
    assert camera_data.info["semantic_segmentation"] == {"idToLabels": id_to_labels}


def test_ovrtx_read_output_clears_stale_metadata_and_keeps_seeded_keys():
    """read_output replaces (not merges): a dropped render var resets its info entry, seeded keys persist."""
    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()

    # ``camera_data.info`` is seeded with one key per output (mirrors ``camera_data.output``); both start None.
    camera_data = CameraData()
    camera_data.info = {"rgb": None, "semantic_segmentation": None}
    camera_data._output = {}

    # Frame 1: the SemanticIdMap render var is present, so its metadata lands in info.
    id_to_labels = {"2": {"class": "cartpole"}}
    render_data.renderer_info = {"semantic_segmentation": {"idToLabels": id_to_labels}}
    renderer.read_output(render_data, camera_data)
    assert camera_data.info["semantic_segmentation"] == {"idToLabels": id_to_labels}

    # Frame 2: render() rebuilds renderer_info from scratch and the SemanticIdMap is gone this frame.
    render_data.renderer_info = {}
    renderer.read_output(render_data, camera_data)

    # The stale idToLabels must be cleared, and the seeded keys (rgb, semantic_segmentation) must remain.
    assert camera_data.info == {"rgb": None, "semantic_segmentation": None}


def test_ovrtx_semantic_spec_follows_colorize_flag():
    """Semantic segmentation output spec is colorized RGBA (uint8) or raw int32 IDs per the cfg flag."""
    colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    colorized.cfg = OVRTXRendererCfg(colorize_semantic_segmentation=True)
    assert colorized.supported_output_types()[RenderBufferKind.SEMANTIC_SEGMENTATION] == RenderBufferSpec(4, wp.uint8)

    non_colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    non_colorized.cfg = OVRTXRendererCfg(colorize_semantic_segmentation=False)
    assert non_colorized.supported_output_types()[RenderBufferKind.SEMANTIC_SEGMENTATION] == RenderBufferSpec(
        1, wp.int32
    )


def test_ovrtx_instance_segmentation_spec_follows_colorize_flag():
    """Instance segmentation output spec is colorized RGBA (uint8) or raw int32 IDs per the cfg flag."""
    colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    colorized.cfg = OVRTXRendererCfg(colorize_instance_segmentation=True)
    assert colorized.supported_output_types()[RenderBufferKind.INSTANCE_SEGMENTATION] == RenderBufferSpec(4, wp.uint8)

    non_colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    non_colorized.cfg = OVRTXRendererCfg(colorize_instance_segmentation=False)
    assert non_colorized.supported_output_types()[RenderBufferKind.INSTANCE_SEGMENTATION] == RenderBufferSpec(
        1, wp.int32
    )


def test_ovrtx_cleanup_releases_only_the_given_render_data():
    """``cleanup`` releases the render data's own buffers and leaves the renderer usable.

    The stage queries, tensor bindings and render products the renderer holds are shared with
    every other camera that resolved to it, so a single camera's cleanup must not take them.
    """
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._initialized_scene = True

    render_data = _make_ovrtx_render_data()
    render_data.warp_buffers = {"rgba": wp.zeros((8, 16, 4), dtype=wp.uint8, device="cpu")}
    render_data.renderer_info = {"semantic_segmentation": {"idToLabels": {}}}
    render_data.ppisp_pipeline = object()

    renderer.cleanup(render_data)

    assert render_data.warp_buffers == {}
    assert render_data.renderer_info == {}
    assert render_data.ppisp_pipeline is None

    assert renderer._render_product_paths == ["/Render/RenderProduct_camera"]
    assert renderer._initialized_scene is True


def test_ovrtx_cleanup_without_render_data_keeps_renderer_state():
    """``cleanup(None)`` has nothing to release and must not disturb the renderer."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._initialized_scene = True

    renderer.cleanup(None)

    assert renderer._render_product_paths == ["/Render/RenderProduct_camera"]
    assert renderer._initialized_scene is True


class _NativeBinding:
    def __init__(self, events: list[str], label: str):
        self.events = events
        self.label = label
        self.writes = []

    def write(self, data, **kwargs) -> None:
        self.writes.append((data, kwargs))

    def unbind(self) -> None:
        self.events.append(f"unbind:{self.label}")


class _NativeSceneBackend:
    def __init__(self, events: list[str], existing_paths: set[str] | None = None):
        self.events = events
        self.calls = []
        self.bindings = []
        self.existing_paths = existing_paths

    def open_usd_from_string(self, usd_text: str) -> None:
        self.calls.append(("open_usd_from_string", usd_text))

    def clone_usd(self, source: str, targets: list[str]) -> None:
        self.calls.append(("clone_usd", source, targets))

    def write_attribute(self, *args, **kwargs) -> None:
        self.calls.append(("write_attribute", args, kwargs))

    def write_array_attribute(self, *args, **kwargs) -> None:
        self.calls.append(("write_array_attribute", args, kwargs))

    def bind_attribute(self, **kwargs):
        return self._bind("bind_attribute", kwargs)

    def bind_array_attribute(self, **kwargs):
        return self._bind("bind_array_attribute", kwargs)

    def step(self, *, render_products: set[str], delta_time: float):
        self.calls.append(("step", render_products, delta_time))
        return {"products": render_products}

    def destroy(self) -> None:
        self.events.append("destroy")

    def _bind(self, operation: str, kwargs: dict):
        missing = set(kwargs["prim_paths"]) - self.existing_paths if self.existing_paths is not None else set()
        if missing and kwargs["prim_mode"] == PrimMode.MUST_EXIST:
            raise RuntimeError(f"Missing prims: {sorted(missing)}")
        label = kwargs["prim_paths"][0].rsplit("/", maxsplit=1)[-1]
        binding = _NativeBinding(self.events, label)
        self.bindings.append(binding)
        self.calls.append((operation, kwargs, binding))
        return binding


def _make_scene(events: list[str]) -> OvrtxScene:
    """Build a native scene with four persistent bindings for teardown tests."""
    backend = _NativeSceneBackend(events)
    scene = OvrtxScene(backend)
    for name in ("camera", "object", "deformable", "particle"):
        scene.bind([f"/{name}"])
    events.clear()
    backend.calls.clear()
    return scene


def test_ovrtx_close_releases_the_shared_scene_after_the_last_camera_client():
    """One camera cannot tear down the context-owned scene while another still uses it."""
    events: list[str] = []
    renderers = [_make_ovrtx_renderer_without_backend(), _make_ovrtx_renderer_without_backend()]
    context = OvReplicateContext.__new__(OvReplicateContext)
    context._renderers = renderers.copy()
    context._ovrtx_scene = _make_scene(events)
    context._ovrtx_key = ("cuda:0",)
    context._scene_data_provider = object()
    context._object_xforms = object()
    context._point_bindings = {"points": (object(), [0], [1])}
    material_writer = context._visual_material_writer = MagicMock()
    context._sdp_transform_generation = 1
    context._sdp_camera_transform_generations = {"camera": 1}
    context._sdp_point_generations = {"points": 1}
    for renderer in renderers:
        renderer._clone_ctx = context
        renderer._camera_xforms = object()
        renderer._render_product_paths = ["/Render/RenderProduct_camera"]
        renderer._render_product_usd = "product"
        renderer._output_id_color_buffers = {"semantic_segmentation": object()}
        renderer._initialized_scene = True

    renderers[0].close()

    assert events == []
    material_writer.close.assert_not_called()
    assert context._ovrtx_scene is not None
    assert context._renderers == [renderers[1]]

    renderers[1].close()

    material_writer.close.assert_called_once_with()
    assert context._visual_material_writer is None
    assert events == ["unbind:camera", "unbind:object", "unbind:deformable", "unbind:particle", "destroy"]
    assert context._ovrtx_scene is None
    assert context._scene_data_provider is None
    assert context._point_bindings == {}
    assert context._sdp_camera_transform_generations == {}
    for renderer in renderers:
        assert renderer._camera_xforms is None
        assert renderer._render_product_paths == []
        assert renderer._output_id_color_buffers == {}
        assert renderer._initialized_scene is False


def test_scene_close_unbinds_native_bindings_before_destroying_the_renderer():
    events: list[str] = []
    scene = _make_scene(events)

    scene.close()

    assert events == ["unbind:camera", "unbind:object", "unbind:deformable", "unbind:particle", "destroy"]
    assert scene._bindings == []
    assert scene._renderer is None


def test_scene_uses_native_renderer_clone_bind_write_and_step(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    backend = _NativeSceneBackend(events)
    scene = OvrtxScene(backend)
    camera_paths = ["/World/envs/env_0/Camera", "/World/envs/env_1/Camera"]
    point_paths = ["/World/envs/env_0/Cloth", "/World/envs/env_1/Cloth"]

    scene.open("#usda 1.0")
    scene.clone("/World/envs/env_0/Robot", ["/World/envs/env_1/Robot"])
    scene.write_tokens(["/World/envs/env_0"], "primvars:omni:scenePartition", ["env_0"])
    scene.point_render_products_at(["/Render/Product"], camera_paths)
    camera = scene.bind(camera_paths)
    points = scene.bind(point_paths, "points", dtype=np.float32, shape=(3,), is_array=True)
    scene.write_reset_xform_stack(camera_paths)

    transforms = SimpleNamespace(device="cuda:0")
    monkeypatch.setattr(wp, "get_stream", lambda _device: SimpleNamespace(cuda_stream=17))
    scene.write_xforms(camera, transforms)
    source = np.arange(18, dtype=np.float32).reshape(6, 3)
    scene.write_points((points, [1, 4], [2, 1]), source)

    products = scene.step(["/Render/Product"])

    assert backend.calls[0] == ("open_usd_from_string", "#usda 1.0")
    assert backend.calls[1] == (
        "clone_usd",
        "/World/envs/env_0/Robot",
        ["/World/envs/env_1/Robot"],
    )
    token_write = backend.calls[2]
    assert token_write[1] == (["/World/envs/env_0"], "primvars:omni:scenePartition", ["env_0"])
    assert token_write[2] == {"semantic": Semantic.TOKEN_STRING, "prim_mode": PrimMode.CREATE_NEW}
    relationship_write = backend.calls[3]
    assert relationship_write[1] == (["/Render/Product"], "camera", [camera_paths])
    assert relationship_write[2] == {"prim_mode": PrimMode.MUST_EXIST}

    xform_bind = backend.calls[4]
    assert xform_bind[0] == "bind_attribute"
    assert xform_bind[1] == {
        "prim_paths": camera_paths,
        "attribute_name": "omni:xform",
        "dtype": None,
        "shape": None,
        "prim_mode": PrimMode.MUST_EXIST,
        "flags": BindingFlag.OPTIMIZE,
        "semantic": Semantic.XFORM_MAT4x4,
    }
    point_bind = backend.calls[5]
    assert point_bind[0] == "bind_array_attribute"
    assert point_bind[1] == {
        "prim_paths": point_paths,
        "attribute_name": "points",
        "dtype": np.float32,
        "shape": (3,),
        "prim_mode": PrimMode.MUST_EXIST,
        "flags": BindingFlag.OPTIMIZE,
    }

    assert camera.binding.writes == [(transforms, {"data_access": DataAccess.ASYNC, "cuda_stream": 17})]
    point_slices, point_kwargs = points.binding.writes[0]
    assert point_kwargs == {"data_access": DataAccess.ASYNC}
    assert all(np.shares_memory(point_slice, source) for point_slice in point_slices)
    assert backend.calls[-1] == ("step", {"/Render/Product"}, 1.0 / 60.0)
    assert products == {"products": {"/Render/Product"}}


def test_scene_rejects_a_missing_exact_plan_binding():
    scene = OvrtxScene(_NativeSceneBackend([], {"/World/Present"}))

    scene.bind(["/World/Present"])
    with pytest.raises(RuntimeError, match="Missing prims.*World/Missing"):
        scene.bind(["/World/Missing"])


def test_scene_close_is_idempotent():
    """A second ``close`` releases nothing again, so a repeated teardown cannot double-free."""
    events: list[str] = []
    scene = _make_scene(events)

    scene.close()
    events.clear()
    scene.close()

    assert events == []
