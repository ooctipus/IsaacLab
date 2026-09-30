# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for typed visual-material writes into OVRTX-owned scenes."""

import importlib.util
import inspect
from types import SimpleNamespace

import pytest
import torch

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = pytest.mark.skipif(
    bool(_MISSING_MODULES), reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}"
)

if not _MISSING_MODULES:
    from isaaclab_ov.cloner import OvReplicateContext
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer
    from isaaclab_ov.renderers.ovrtx_scene import OvrtxScene
    from isaaclab_ov.renderers.visual_materials import OVRTXVisualMaterialWriter
    from ovrtx import DataAccess

    from isaaclab.renderers.base_renderer import VisualMaterialBatch


class _Completion:
    def __init__(self, callback=None):
        self.wait_count = 0
        self._callback = callback

    def wait(self):
        self.wait_count += 1
        if self._callback is not None:
            self._callback()


class _Binding:
    def __init__(self, events: list[str], attribute_name: str):
        self.events = events
        self.attribute_name = attribute_name
        self.writes = []
        self.unbind_count = 0

    def write_async(self, values, **kwargs):
        self.events.append(f"write:{self.attribute_name}")
        completion = _Completion(lambda: self.events.append(f"wait:{self.attribute_name}"))
        self.writes.append((values, kwargs, completion))
        return completion

    def unbind(self):
        self.unbind_count += 1


def _renderer():
    events: list[str] = []

    class Backend:
        def __init__(self):
            self.bindings = []
            self.fail_step = False

        def bind_attribute(self, **kwargs):
            binding = _Binding(events, kwargs["attribute_name"])
            self.bindings.append(binding)
            return binding

        def step(self, *, render_products, **_kwargs):
            events.append("step")
            if self.fail_step:
                raise ValueError("step failed")
            return {path: SimpleNamespace(frames=[SimpleNamespace(render_vars={})]) for path in render_products}

    scene = OvrtxScene(Backend())
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = SimpleNamespace(colorize_semantic_segmentation=False, colorize_instance_segmentation=False)
    renderer._initialized_scene = True
    renderer._clone_ctx = OvReplicateContext.__new__(OvReplicateContext)
    renderer._clone_ctx._ovrtx_scene = scene
    renderer._clone_ctx._scene_data_provider = object()
    renderer._clone_ctx._visual_material_writer = None
    return renderer, events


def _batch(channel: str, input_names: tuple[str, ...], values: torch.Tensor) -> VisualMaterialBatch:
    material_paths = tuple(f"/Looks/{channel}_{index}" for index in range(len(values)))
    return VisualMaterialBatch(
        channel,
        material_paths,
        tuple(f"{path}/Shader" for path in material_paths),
        input_names,
        values,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiles_queries_and_publishes_selected_channels_zero_copy():
    renderer, _events = _renderer()
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device="cuda")
    roughness = torch.tensor([0.1, 0.9], device="cuda")
    texture_scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    writer = renderer.visual_material_writer(
        (
            _batch("color", ("diffuse_color_constant", "diffuse_color_constant", "diffuseColor"), colors),
            _batch("roughness", ("roughness", "roughness"), roughness),
            _batch("texture_scale", ("texture_scale", "texture_scale"), texture_scale),
        )
    )

    writer(
        {"texture_scale": torch.tensor([0], dtype=torch.int32, device="cuda")},
        torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    scene = renderer._clone_ctx.scene
    assert all(binding.writes == [] for binding in scene._renderer.bindings)
    assert len(scene._bindings) == 4
    writer.publish()

    writes = {binding.attribute_name: binding.writes for binding in scene._renderer.bindings if binding.writes}
    assert set(writes) == {"inputs:texture_scale"}
    write = writes["inputs:texture_scale"][0]
    assert write[0].untyped_storage().data_ptr() == texture_scale.untyped_storage().data_ptr()
    assert write[1]["data_access"] == DataAccess.ASYNC
    assert write[1]["cuda_event"] == writer._event.cuda_event
    writer.drain()
    assert write[2].wait_count == 1


def test_render_publishes_and_drains_material_writes_at_backend_boundary():
    renderer, events = _renderer()
    renderer._render_product_paths = ["/Render/Product"]

    class Writer:
        def publish(self):
            events.append("publish")

        def drain(self):
            events.append("drain")

    writer = Writer()
    renderer._clone_ctx._visual_material_writer = writer
    renderer.render(SimpleNamespace(ppisp_pipeline=None, renderer_info={}, warp_buffers={}))

    assert events == ["publish", "step", "drain"]


@pytest.mark.parametrize(
    ("failure", "expected_events"),
    (("publish", ["publish", "drain"]), ("step", ["publish", "step", "drain"])),
)
def test_drain_does_not_mask_publish_or_step_failure(failure, expected_events):
    renderer, events = _renderer()
    renderer._render_product_paths = ["/Render/Product"]

    class Writer:
        def publish(self):
            events.append("publish")
            if failure == "publish":
                raise ValueError("publish failed")

        def drain(self):
            events.append("drain")
            raise RuntimeError("drain failed")

    writer = Writer()
    renderer._clone_ctx._visual_material_writer = writer

    renderer._clone_ctx.scene._renderer.fail_step = failure == "step"
    with pytest.raises(ValueError, match=failure):
        renderer.render(SimpleNamespace(ppisp_pipeline=None, renderer_info={}, warp_buffers={}))

    assert events == expected_events


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_writer_close_drains_and_releases_compiled_addresses():
    renderer, _events = _renderer()
    writer = renderer.visual_material_writer((_batch("roughness", ("roughness",), torch.zeros(1, device="cuda")),))
    writer(None)
    writer.publish()
    writer.close()

    scene = renderer._clone_ctx.scene
    assert scene._bindings == []
    assert all(binding.unbind_count == 1 for binding in scene._renderer.bindings)
    assert all(write[2].wait_count == 1 for binding in scene._renderer.bindings for write in binding.writes)
    assert writer._addresses == []
    assert writer._buffers == {}
    assert renderer._clone_ctx._visual_material_writer is None
    renderer._clone_ctx._ovrtx_scene = None
    writer.close()


@pytest.mark.parametrize("values", [torch.zeros(1, dtype=torch.float64), torch.zeros(1, 4)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compilation_rejects_unsupported_device_buffers_before_backend_mutation(values):
    renderer, _events = _renderer()
    valid = _batch("color", ("diffuseColor",), torch.zeros(1, 3, device="cuda"))
    invalid = _batch("unsupported", ("value",), values.to(device="cuda"))

    with pytest.raises(TypeError, match="float, float2, or float3"):
        renderer.visual_material_writer((valid, invalid))

    assert renderer._clone_ctx._visual_material_writer is None
    assert renderer._clone_ctx.scene._bindings == []


def test_compilation_rejects_host_buffers_without_fallback():
    renderer, _events = _renderer()
    with pytest.raises(RuntimeError, match="one CUDA device"):
        renderer.visual_material_writer((_batch("color", ("diffuseColor",), torch.zeros(1, 3)),))

    assert renderer._clone_ctx._visual_material_writer is None
    assert renderer._clone_ctx.scene._bindings == []


def test_writer_factory_requires_ingested_detached_scene():
    renderer, _events = _renderer()
    renderer._clone_ctx._scene_data_provider = None
    with pytest.raises(RuntimeError, match="ingest its detached scene"):
        renderer.visual_material_writer((_batch("color", ("diffuseColor",), torch.zeros(1, 3)),))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_multiple_cameras_share_one_writer_and_submit_each_dirty_channel_once():
    first, events = _renderer()
    second = OVRTXRenderer.__new__(OVRTXRenderer)
    second.cfg = first.cfg
    second._initialized_scene = True
    second._clone_ctx = first._clone_ctx
    for renderer, product in ((first, "/Render/First"), (second, "/Render/Second")):
        renderer._render_product_paths = [product]

    assert first.visual_material_writer == second.visual_material_writer
    writer = first.visual_material_writer((_batch("roughness", ("roughness",), torch.zeros(1, device="cuda")),))
    with pytest.raises(RuntimeError, match="only one visual-material writer"):
        second.visual_material_writer((_batch("roughness", ("roughness",), torch.zeros(1, device="cuda")),))
    writer()
    render_data = SimpleNamespace(ppisp_pipeline=None, renderer_info={}, warp_buffers={})
    first.render(render_data)
    second.render(render_data)
    writer()
    first.render(render_data)
    second.render(render_data)

    assert first._clone_ctx._visual_material_writer is writer
    assert events.count("write:inputs:roughness") == 2
    assert events.count("step") == 4
    writer.close()


def test_material_runtime_has_no_host_or_usd_path():
    source = inspect.getsource(OVRTXVisualMaterialWriter)
    for forbidden in (".cpu(", ".numpy(", ".tolist(", "pxr", "Usd", "Sdf"):
        assert forbidden not in source
    renderer_source = inspect.getsource(OVRTXRenderer)
    assert "_visual_material_writer_ref" not in renderer_source
    assert "_create_visual_material_writer" not in renderer_source
