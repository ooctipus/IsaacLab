# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the renderer→camera output contract."""

import gc
import inspect
import sys
from types import SimpleNamespace

import pytest
import warp as wp

pytest.importorskip("isaaclab_physx")

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab.scene_data import SceneDataFormat, SceneDataPublication
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import PinholeCameraCfg, SimulationContext
from isaaclab.utils.warp import ProxyArray

pytestmark = [pytest.mark.integration, pytest.mark.rendering]

_SPAWN = PinholeCameraCfg(
    focal_length=24.0,
    focus_distance=400.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 1.0e5),
)


def test_newton_warp_supported_output_types_key_set():
    """NewtonWarpRenderer publishes the documented key set."""
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

    renderer = NewtonWarpRenderer.__new__(NewtonWarpRenderer)
    renderer.cfg = NewtonWarpRendererCfg()
    specs = renderer.supported_output_types()

    assert set(specs.keys()) == {
        RenderBufferKind.RGB,
        RenderBufferKind.RGBA,
        RenderBufferKind.RGB_HDR,
        RenderBufferKind.ALBEDO,
        RenderBufferKind.DEPTH,
        RenderBufferKind.DISTANCE_TO_CAMERA,
        RenderBufferKind.DISTANCE_TO_IMAGE_PLANE,
        RenderBufferKind.NORMALS,
        RenderBufferKind.SEMANTIC_SEGMENTATION,
        RenderBufferKind.INSTANCE_SEGMENTATION,
    }
    assert specs[RenderBufferKind.RGB_HDR] == RenderBufferSpec(3, wp.float32)


@pytest.mark.parametrize("colorize", [True, False])
def test_newton_warp_segmentation_spec_follows_colorize_flags(colorize):
    """Segmentation specs are RGBA uint8 when colorized, else single-channel int32 (matching RTX)."""
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

    renderer = NewtonWarpRenderer.__new__(NewtonWarpRenderer)
    renderer.cfg = NewtonWarpRendererCfg(
        colorize_semantic_segmentation=colorize,
        colorize_instance_segmentation=colorize,
    )
    specs = renderer.supported_output_types()

    expected = RenderBufferSpec(4, wp.uint8) if colorize else RenderBufferSpec(1, wp.int32)
    for kind in (
        RenderBufferKind.SEMANTIC_SEGMENTATION,
        RenderBufferKind.INSTANCE_SEGMENTATION,
    ):
        assert specs[kind] == expected


def test_newton_warp_wraps_requested_rgb_hdr_output():
    """NewtonWarpRenderer wires requested RGB_HDR proxies to the Newton HDR output slot."""
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    wp.init()
    from isaaclab_newton.renderers.newton_warp_renderer import RenderData

    from isaaclab.utils.warp.proxy_array import ProxyArray

    fake_sensor = SimpleNamespace(model=SimpleNamespace(world_count=2, device="cpu"))
    render_data = RenderData(
        fake_sensor,
        SimpleNamespace(
            cfg=CameraCfg(
                width=4,
                height=3,
                prim_path="/World/Camera",
                spawn=_SPAWN,
                renderer_cfg=NewtonWarpRendererCfg(),
            )
        ),
    )
    hdr_proxy = ProxyArray(wp.zeros((2, 3, 4, 3), dtype=wp.float32, device="cpu"))

    render_data.set_outputs({str(RenderBufferKind.RGB_HDR): hdr_proxy})

    assert render_data.outputs.hdr_color_image is not None
    assert render_data.get_output(RenderBufferKind.RGB_HDR) is render_data.outputs.hdr_color_image


def test_newton_warp_rejects_an_unknown_output_at_binding() -> None:
    """Direct renderer callers cannot regain Camera's rejected-output fallback."""
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    from isaaclab_newton.renderers.newton_warp_renderer import RenderData

    render_data = RenderData(
        SimpleNamespace(model=SimpleNamespace(world_count=1, device="cpu")),
        SimpleNamespace(
            cfg=CameraCfg(
                width=4,
                height=3,
                prim_path="/World/Camera",
                spawn=_SPAWN,
                renderer_cfg=NewtonWarpRendererCfg(),
            )
        ),
    )

    with pytest.raises(ValueError, match="does not support output 'unknown'"):
        render_data.set_outputs({"unknown": object()})


def _make_camera_cfg(data_types: list[str]) -> CameraCfg:
    return CameraCfg(
        height=8,
        width=16,
        prim_path="/World/Camera",
        spawn=_SPAWN,
        data_types=data_types,
        renderer_cfg=IsaacRtxRendererCfg(),
    )


def test_camera_data_allocates_supported_subset_and_aliases_rgb():
    """CameraData allocates the intersection of requested + supported and aliases rgb into rgba."""
    cfg = _make_camera_cfg(["rgb", "rgba", "depth"])
    specs = {
        RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
        RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
        RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
        RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
    }
    data = CameraData.allocate(
        data_types=cfg.data_types, height=8, width=16, num_views=2, device="cpu", supported_specs=specs
    )

    assert set(data.output.keys()) == {"rgba", "rgb", "depth"}
    assert data.output["rgba"].shape == (2, 8, 16, 4)
    assert data.output["rgba"].dtype == wp.uint8
    assert data.output["depth"].shape == (2, 8, 16, 1)
    assert data.output["depth"].dtype == wp.float32
    assert data.output["rgb"].warp.ptr == data.output["rgba"].warp.ptr
    assert data.image_shape == (8, 16)
    assert data.info == {"rgba": None, "rgb": None, "depth": None}


def test_camera_data_drops_requested_types_not_in_supported_specs():
    """Requested types absent from supported_specs are absent from data.output."""
    cfg = _make_camera_cfg(["rgb", "normals"])
    specs = {
        RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
        RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
    }
    data = CameraData.allocate(
        data_types=cfg.data_types, height=4, width=4, num_views=1, device="cpu", supported_specs=specs
    )

    assert "normals" not in data.output
    assert {"rgb", "rgba"} <= set(data.output.keys())


def test_camera_data_no_arg_construction_yields_empty_container():
    """Bare CameraData() produces an all-None container."""
    data = CameraData()
    assert data.pos_w is None
    assert data.quat_w_world is None
    assert data.intrinsic_matrices is None
    assert data.output is None
    assert data.info is None
    assert data.image_shape is None


def test_camera_data_segmentation_dtype_follows_supported_spec():
    """CameraData consumes the layout dtype declared by the renderer spec."""
    cfg = _make_camera_cfg(["instance_segmentation"])
    raw_specs = {RenderBufferKind.INSTANCE_SEGMENTATION: RenderBufferSpec(1, wp.int32)}
    colorized_specs = {RenderBufferKind.INSTANCE_SEGMENTATION: RenderBufferSpec(4, wp.uint8)}

    raw = CameraData.allocate(
        data_types=cfg.data_types, height=4, width=4, num_views=1, device="cpu", supported_specs=raw_specs
    )
    colorized = CameraData.allocate(
        data_types=cfg.data_types, height=4, width=4, num_views=1, device="cpu", supported_specs=colorized_specs
    )

    assert raw.output["instance_segmentation"].dtype == wp.int32
    assert raw.output["instance_segmentation"].shape == (1, 4, 4, 1)
    assert colorized.output["instance_segmentation"].dtype == wp.uint8
    assert colorized.output["instance_segmentation"].shape == (1, 4, 4, 4)


def test_camera_data_allocate_raises_on_unknown_name():
    """An unknown data_types name raises ValueError naming the offender."""
    supported_specs = {RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8)}
    with pytest.raises(ValueError) as exc_info:
        CameraData.allocate(
            data_types=["not_a_real_type"],
            height=4,
            width=4,
            num_views=1,
            device="cpu",
            supported_specs=supported_specs,
        )
    assert "not_a_real_type" in str(exc_info.value)
    assert "RenderBufferKind" in str(exc_info.value)


##
# Camera renderer lifecycle.
##


def test_renderers_initialize_once_after_the_clone_plan_completes():
    """The complete pre-clone renderer set initializes once before camera sensors use it."""
    from isaaclab.renderers.base_renderer import BaseRenderer

    events: list[str] = []

    class _Backend(BaseRenderer):
        def initialize(self) -> None:
            events.append("initialize")

        def supported_output_types(self):
            return {}

        def create_render_data(self, _spec) -> object:
            events.append("create_render_data")
            return object()

        def set_outputs(self, *_args) -> None:
            pass

        def update(self, *_args) -> None:
            pass

        def render(self, _render_data) -> None:
            pass

        def read_output(self, *_args) -> None:
            pass

        def cleanup(self, _render_data) -> None:
            pass

    ctx = object.__new__(SimulationContext)
    ctx._renderer_entries = []
    ctx._renderers_initialized = False
    ctx._clone_plan = SimpleNamespace(is_complete=False)
    cfg = IsaacRtxRendererCfg()
    cfg.class_type = lambda _cfg: _Backend()
    first = ctx.get_renderer(cfg)
    second = ctx.get_renderer(cfg)

    ctx._clone_plan = SimpleNamespace(is_complete=True)
    ctx._initialize_renderers()
    ctx._initialize_renderers()

    assert first is not second
    assert events == ["initialize", "initialize"]
    late_cfg = NewtonWarpRendererCfg()
    late_cfg.class_type = lambda _cfg: _Backend()
    with pytest.raises(RuntimeError, match="before the clone plan completes"):
        ctx.get_renderer(late_cfg)


def test_renderer_initialization_rejects_an_incomplete_clone_plan():
    """Renderer initialization cannot race cloning or recover through a camera-local drain."""
    ctx = object.__new__(SimulationContext)
    ctx._renderer_entries = [SimpleNamespace(initialize=lambda: None)]
    ctx._renderers_initialized = False
    ctx._clone_plan = SimpleNamespace(is_complete=False)

    with pytest.raises(RuntimeError, match="completed clone plan"):
        ctx._initialize_renderers()


@pytest.mark.parametrize(("pose_follows_physics", "pose_event"), [(True, "poses"), (False, "frame")])
def test_camera_updates_scene_camera_and_output_in_order(pose_follows_physics, pose_event):
    """Only body-attached cameras refresh their SDP pose during a renderer exchange."""
    from isaaclab.sensors.camera import Camera

    events = []

    class _Renderer:
        def update(self, render_data, intrinsics):
            events.append(("update", render_data, intrinsics))

        def render(self, render_data):
            events.append(("render", render_data))

        def read_output(self, render_data, camera_data):
            events.append(("read", render_data, camera_data))

    render_data = object()
    camera_data = SimpleNamespace(intrinsic_matrices=object())
    camera = SimpleNamespace(
        _renderer=_Renderer(),
        _render_data=render_data,
        _data=camera_data,
        _pose_follows_physics=pose_follows_physics,
        cfg=SimpleNamespace(update_latest_camera_pose=True),
        _env_mask_has_any=lambda _mask: True,
        _update_poses=lambda **_kwargs: events.append("poses"),
        _update_camera_state=lambda **_kwargs: events.append("frame"),
    )

    Camera._update_buffers_impl(camera, object())

    assert events == [
        pose_event,
        ("update", render_data, camera_data.intrinsic_matrices),
        ("render", render_data),
        ("read", render_data, camera_data),
    ]


def test_camera_full_pose_refresh_publishes_named_sdp_data():
    """A full camera pose refresh publishes native OpenGL transforms through named SDP data."""
    positions = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3f, device="cpu")
    orientations = wp.array([[0.0, 0.0, 0.0, 1.0]], dtype=wp.quatf, device="cpu")
    publication = SceneDataPublication(SceneDataFormat.Vec3_Quat(), False)
    camera = SimpleNamespace(
        _view=SimpleNamespace(
            get_world_poses=lambda _indices: (ProxyArray(positions), ProxyArray(orientations)),
        ),
        _render_data=object(),
        _render_pose_publication=publication,
        _resolve_env_ids_wp=lambda _env_ids: None,
        _update_camera_state=lambda **_kwargs: None,
    )

    Camera._update_poses(camera, env_mask=object())

    assert publication.data.positions.ptr == positions.ptr
    assert publication.data.orientations.ptr == orientations.ptr
    assert publication.dirty


@pytest.mark.parametrize("from_view", [False, True])
def test_camera_pose_setters_refresh_named_sdp_publication(from_view):
    """Both explicit camera pose setters refresh the named SDP publication immediately."""
    events = []

    class _Writer:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def set_poses(self, *_args):
            events.append("write")

    camera = SimpleNamespace(
        _device="cpu",
        _view=SimpleNamespace(count=1, xform_world_space_writer=_Writer),
        _resolve_env_ids_wp=lambda _env_ids: None,
        _update_poses=lambda **_kwargs: events.append("publish"),
        cfg=SimpleNamespace(update_latest_camera_pose=False),
    )
    camera.set_world_poses = Camera.set_world_poses.__get__(camera)

    if from_view:
        Camera.set_world_poses_from_view(camera, [[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    else:
        camera.set_world_poses([[1.0, 2.0, 3.0]], [[0.0, 0.0, 0.0, 1.0]], convention="opengl")

    assert events == ["write", "publish"]


def test_camera_cannot_be_built_without_a_simulation(monkeypatch: pytest.MonkeyPatch):
    """A camera registers physics callbacks, so there is no 'built before the simulation' path.

    This is why ``Camera.__init__`` describes itself to its backend unconditionally: the branch
    that used to guard against a missing simulation could never be taken.
    """
    from isaaclab.sensors.camera import Camera

    monkeypatch.setattr("isaaclab.sim.SimulationContext.instance", staticmethod(lambda: None))
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)

    with pytest.raises(RuntimeError, match="requires an active SimulationContext"):
        Camera(
            CameraCfg(
                prim_path="/World/Cam",
                height=8,
                width=8,
                data_types=["rgb"],
                spawn=_SPAWN,
                renderer_cfg=IsaacRtxRendererCfg(),
            )
        )
    gc.collect()

    assert unraisable == []


def test_camera_never_fetches_planned_prims_from_stage():
    """Architecture gate: planned camera handles are retained before initialization."""
    from isaaclab.sensors.camera import Camera

    source = inspect.getsource(Camera)
    assert all(discovery not in source for discovery in ("GetPrimAtPath", "PrimRange", "Traverse"))
