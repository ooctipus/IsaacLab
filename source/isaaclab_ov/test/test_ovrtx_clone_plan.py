# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX clone-plan consumption and OVRTX-side cloning."""

from __future__ import annotations

import importlib.util
import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from isaaclab.cloner import ClonePlan
from isaaclab.cloner.clone_plan import RigidBodyLayout
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat, SceneDataProvider, SceneDataPublication
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import PinholeCameraCfg, SimulationContext

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
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_scene import OvrtxScene  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_usd import build_render_product_as_string  # noqa: E402

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402
else:
    OvReplicateContext = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    OvrtxScene = object
    build_render_product_as_string = None
    Sdf = None
    Gf = None
    Usd = None
    UsdGeom = None
    UsdShade = None


_OVRTX_STAGE_FILE = "ovrtx_renderer_stage.usda"


def _make_multi_env_stage(num_envs: int) -> Usd.Stage:
    """Build an in-memory stage with distinguishable content per environment."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")

    for env_idx in range(num_envs):
        env_path = f"/World/envs/env_{env_idx}"
        UsdGeom.Xform.Define(stage, env_path)
        UsdGeom.Xform.Define(stage, f"{env_path}/Robot")
        UsdGeom.Xform.Define(stage, f"{env_path}/Object")
        UsdGeom.Xform.Define(stage, f"{env_path}/Light")
        UsdGeom.Xform.Define(stage, f"{env_path}/Object_env{env_idx}_only")
        UsdGeom.Camera.Define(stage, f"{env_path}/Camera")

    return stage


def _make_camera_render_spec(num_envs: int = 1) -> CameraRenderSpec:
    spawn = PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 1.0e5),
    )
    cfg = CameraCfg(
        height=8,
        width=16,
        prim_path="/World/envs/env_0/Camera",
        spawn=spawn,
        data_types=["rgb"],
        renderer_cfg=OVRTXRendererCfg(),
    )
    camera_paths = tuple(f"/World/envs/env_{env_idx}/Camera" for env_idx in range(num_envs))
    return CameraRenderSpec(
        cfg=cfg,
        device="cpu",
        camera_source_prim_paths=("/World/envs/env_0/Camera",),
        camera_prim_paths=camera_paths,
    )


class _RecordingScene(OvrtxScene):
    """An OVRTX scene test double that records calls instead of drawing."""

    def __init__(self):
        super().__init__(renderer=None)
        self.calls: list[tuple] = []
        self.clone_error: Exception | None = None
        self.transforms: np.ndarray | None = None
        self.transform_pointers: list[int] = []
        self.transform_paths: list[tuple[str, ...]] = []

    def open(self, usd_text: str) -> None:
        self.calls.append(("open", usd_text))

    def clone(self, source, targets) -> None:
        if self.clone_error is not None:
            raise self.clone_error
        self.calls.append(("clone", source, list(targets)))

    def write_tokens(self, paths, attribute, tokens) -> None:
        self.calls.append(("write_tokens", attribute, list(paths), list(tokens)))

    def write_reset_xform_stack(self, paths) -> None:
        self.calls.append(("reset_xform_stack", list(paths)))

    def point_render_products_at(self, product_paths, camera_paths) -> None:
        self.calls.append(("point_render_products_at", list(product_paths), list(camera_paths)))

    def bind(self, paths, *_args, **_kwargs):
        self.calls.append(("bind", list(paths)))
        return SimpleNamespace(paths=list(paths))

    def write_xforms(self, handle, transforms) -> None:
        self.transform_pointers.append(transforms.__array_interface__["data"][0])
        self.transform_paths.append(tuple(handle.paths))
        self.transforms = transforms

    def write_points(self, binding, particle_q) -> None:
        self.calls.append(("write_points", binding[0].paths))

    def step(self, product_paths) -> dict:
        return {}

    def close(self) -> None:
        self.calls.append(("close",))


class _CountingStage:
    """USD stage proxy that counts the one flatten-and-serialize snapshot."""

    def __init__(self, stage: Usd.Stage):
        self._stage = stage
        self.export_count = 0

    def __getattr__(self, name: str):
        return getattr(self._stage, name)

    def Flatten(self):  # noqa: N802  (USD API spelling)
        self.export_count += 1
        return self._stage.Flatten()


class _Simulation:
    def __init__(self, stage):
        self.stage = stage
        self.cfg = SimpleNamespace(physics_prim_path="/physicsScene")
        self.plan: ClonePlan | None = None

    def get_clone_plan(self) -> ClonePlan | None:
        return self.plan


def _make_ovrtx_renderer_without_backend(num_envs: int = 1) -> OVRTXRenderer:
    """Build a renderer whose OVRTX scene records calls, so the cloning path can run headless."""
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    renderer._device = "cpu"
    renderer._clone_ctx = OvReplicateContext(_Simulation(_make_multi_env_stage(max(2, num_envs))))
    renderer._client_id = renderer._clone_ctx._add_renderer(renderer)
    renderer._clone_ctx._ovrtx_key = (0, "", "", None)
    renderer._clone_ctx._ovrtx_scene = _RecordingScene()
    _set_spec(renderer, _make_camera_render_spec(num_envs))
    renderer._initialized_scene = False
    renderer._camera_xforms = None
    renderer._output_id_color_buffers = {}
    return renderer


def _set_spec(renderer: OVRTXRenderer, spec: CameraRenderSpec) -> None:
    """Give a headless camera client its declarative render product."""
    renderer._spec = spec
    renderer._render_product_usd, path = build_render_product_as_string(
        width=spec.cfg.width,
        height=spec.cfg.height,
        num_envs=spec.num_instances,
        data_types=spec.cfg.data_types,
        camera_prim_path=spec.camera_source_prim_paths[0],
        render_scope_name=f"Render_{renderer._client_id}",
    )
    renderer._render_product_paths = [path]


def _calls_of(context: OvReplicateContext, name: str) -> list[tuple]:
    """Every recorded scene call of one kind."""
    return [call for call in context.scene.calls if call[0] == name]


def _queue_plan(
    renderer: OVRTXRenderer, plan: ClonePlan, stage: Usd.Stage | _CountingStage | None = None
) -> OvReplicateContext:
    """Fill the renderer's context the way :func:`~isaaclab.cloner.replicate` does."""
    ctx = renderer._clone_ctx
    if stage is not None and not ctx._replicated:
        ctx._sim = _Simulation(stage)
        ctx.stage = stage
    if not plan.is_complete:
        env_ids = plan.env_ids if plan.env_ids is not None else np.arange(plan.clone_mask.shape[1], dtype=np.int64)
        plan = replace(
            plan,
            env_ids=env_ids,
            is_complete=True,
            rigid_body_prototypes=tuple(
                RigidBodyLayout(
                    destination,
                    destination.replace("{}", "*"),
                    row,
                    None,
                    source.rsplit("/", 1)[-1],
                    source,
                    clone_mask=plan.clone_mask[row],
                )
                for row, (source, destination) in enumerate(zip(plan.sources, plan.destinations, strict=True))
            ),
            _env_ids_cpu=tuple(env_ids.tolist()),
        )
    ctx._sim.plan = plan
    return ctx


def _replicate_and_snapshot(context: OvReplicateContext) -> str:
    context.replicate(context._sim.plan)
    return context.stage_usda


def _env_root_plan(num_envs: int, positions: np.ndarray | None = None) -> ClonePlan:
    """A representative asset-level plan beneath pre-authored environment roots."""
    return ClonePlan(
        sources=("/World/envs/env_0/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=np.ones((1, num_envs), dtype=np.bool_),
        env_ids=np.arange(num_envs, dtype=np.int64),
        positions=np.zeros((num_envs, 3), dtype=np.float32) if positions is None else positions,
    )


def test_heterogeneous_supported_rows_clone_natively():
    """A rigid heterogeneous plan keeps prototypes in the snapshot and clones each row natively."""
    renderer = _make_ovrtx_renderer_without_backend(num_envs=4)
    _queue_plan(
        renderer,
        ClonePlan(
            sources=("/World/envs/env_0/Robot", "/World/envs/env_1/Object", "/World/envs/env_0/Light"),
            destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object", "/World/envs/env_{}/Light"),
            clone_mask=np.array(
                [
                    [True, True, True, True],
                    [False, False, True, True],
                    [False, False, False, False],
                ],
                dtype=np.bool_,
            ),
            positions=np.zeros((4, 3), dtype=np.float32),
        ),
        stage=_make_multi_env_stage(4),
    )

    renderer._clone_ctx.replicate(renderer._clone_ctx._sim.plan)

    assert _calls_of(renderer._clone_ctx, "clone") == [
        (
            "clone",
            "/World/envs/env_0/Robot",
            ["/World/envs/env_1/Robot", "/World/envs/env_2/Robot", "/World/envs/env_3/Robot"],
        ),
        ("clone", "/World/envs/env_1/Object", ["/World/envs/env_2/Object", "/World/envs/env_3/Object"]),
    ]
    identity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    assert renderer._clone_ctx.physics_clone_rows == (
        (
            "/World/envs/env_0/Robot",
            ("/World/envs/env_1/Robot", "/World/envs/env_2/Robot", "/World/envs/env_3/Robot"),
            (identity, identity, identity),
        ),
        (
            "/World/envs/env_1/Object",
            ("/World/envs/env_2/Object", "/World/envs/env_3/Object"),
            (identity, identity),
        ),
    )
    layer = Sdf.Layer.CreateAnonymous("snapshot.usda")
    assert layer.ImportFromString(renderer._clone_ctx.stage_usda)
    assert layer.GetPrimAtPath("/World/envs/env_0/Robot") is not None
    assert layer.GetPrimAtPath("/World/envs/env_1/Object") is not None
    assert layer.GetPrimAtPath("/World/envs/env_1/Robot") is None
    assert layer.GetPrimAtPath("/World/envs/env_2/Object") is None


def test_snapshot_retains_only_planned_flattened_dependencies():
    """Internal prototypes and bound materials reachable from a planned source survive trimming."""
    stage = _make_multi_env_stage(2)
    planned = UsdGeom.Xform.Define(stage, "/Library/Planned")
    collision = UsdGeom.Cube.Define(stage, "/Library/Planned/Collision")
    UsdGeom.Sphere.Define(stage, "/Library/Planned/Visual")
    material = UsdShade.Material.Define(stage, "/Looks/Planned")
    UsdShade.Shader.Define(stage, "/Looks/Planned/Shader").CreateIdAttr("UsdPreviewSurface")
    UsdShade.MaterialBindingAPI.Apply(collision.GetPrim()).Bind(material)
    robot = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    robot.GetReferences().AddInternalReference(planned.GetPath())
    robot.SetInstanceable(True)

    unplanned = UsdGeom.Xform.Define(stage, "/Library/Unplanned")
    UsdGeom.Capsule.Define(stage, "/Library/Unplanned/Geometry")
    obj = stage.GetPrimAtPath("/World/envs/env_0/Object")
    obj.GetReferences().AddInternalReference(unplanned.GetPath())
    obj.SetInstanceable(True)

    renderer = _make_ovrtx_renderer_without_backend(num_envs=2)
    _queue_plan(renderer, _env_root_plan(2), stage=stage)
    renderer._clone_ctx.replicate(renderer._clone_ctx._sim.plan)

    layer = Sdf.Layer.CreateAnonymous("snapshot.usda")
    assert layer.ImportFromString(renderer._clone_ctx.stage_usda)
    snapshot = Usd.Stage.Open(layer)
    assert snapshot.GetPrimAtPath("/World/envs/env_0/Robot/Collision").IsValid()
    assert snapshot.GetPrimAtPath("/World/envs/env_0/Robot/Visual").IsValid()
    assert snapshot.GetPrimAtPath("/Looks/Planned/Shader").IsValid()
    assert not snapshot.GetPrimAtPath("/World/envs/env_0/Object").IsValid()
    assert not snapshot.GetPrimAtPath("/Library/Unplanned").IsValid()
    assert len([spec for spec in layer.pseudoRoot.nameChildren if spec.name.startswith("Flattened_")]) == 1


def test_replicate_raises_on_clone_failure():
    """A failing clone surfaces as RuntimeError naming the row."""
    renderer = _make_ovrtx_renderer_without_backend(num_envs=2)
    _queue_plan(renderer, _env_root_plan(2), stage=_make_multi_env_stage(2))
    renderer._clone_ctx.scene.clone_error = OSError("clone failed")

    with pytest.raises(RuntimeError, match="Failed to copy /World/envs/env_0/Robot"):
        renderer._clone_ctx.replicate(renderer._clone_ctx._sim.plan)


def test_replicate_does_not_reauthor_plan_positions():
    """ReplicateSession owns env placement; OVRTX only consumes the planned asset rows."""
    renderer = _make_ovrtx_renderer_without_backend(num_envs=3)
    positions = np.array([[0.0, 0.0, 0.0], [2.0, -1.0, 0.5], [-3.0, 4.0, 1.5]])
    stage = _make_multi_env_stage(3)
    for env_id, position in enumerate(positions.tolist()):
        UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/envs/env_{env_id}")).AddTranslateOp().Set(Gf.Vec3d(*position))
    _queue_plan(renderer, _env_root_plan(3, positions), stage=stage)

    renderer._clone_ctx.replicate(renderer._clone_ctx._sim.plan)

    assert [call[0] for call in renderer._clone_ctx.scene.calls] == [
        "open",
        "reset_xform_stack",
        "clone",
        "write_tokens",
        "write_tokens",
        "point_render_products_at",
        "bind",
    ]
    _name, source, targets = _calls_of(renderer._clone_ctx, "clone")[0]
    assert (source, targets) == (
        "/World/envs/env_0/Robot",
        ["/World/envs/env_1/Robot", "/World/envs/env_2/Robot"],
    )
    layer = Sdf.Layer.CreateAnonymous("snapshot.usda")
    assert layer.ImportFromString(renderer._clone_ctx.stage_usda)
    snapshot = Usd.Stage.Open(layer)
    assert _calls_of(renderer._clone_ctx, "reset_xform_stack") == [
        ("reset_xform_stack", ["/World/envs/env_0/Robot", "/World/envs/env_0/Camera"])
    ]
    for env_id, position in enumerate(positions.tolist()):
        matrix = UsdGeom.Xformable(snapshot.GetPrimAtPath(f"/World/envs/env_{env_id}")).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        assert tuple(matrix.ExtractTranslation()) == pytest.approx(position)


def test_physics_uses_one_plan_proven_whole_environment_clone():
    """OVPhysX avoids native leaf cloning when a rigid plan is exactly collapsible."""
    renderer = _make_ovrtx_renderer_without_backend(num_envs=3)
    positions = np.array([[1.0, 2.0, 3.0], [-2.0, 4.0, 0.5], [7.0, -1.0, 2.5]])
    context = _queue_plan(renderer, _env_root_plan(3, positions), stage=_make_multi_env_stage(3))

    context.replicate(context._sim.plan)

    assert context.physics_clone_rows == (
        (
            "/World/envs/env_0",
            ("/World/envs/env_1", "/World/envs/env_2"),
            ((-2.0, 4.0, 0.5, 0.0, 0.0, 0.0, 1.0), (7.0, -1.0, 2.5, 0.0, 0.0, 0.0, 1.0)),
        ),
    )


def test_inactive_unsupported_row_does_not_materialize_supported_rows():
    """An unsupported prototype that reaches no environment cannot force active rows into USDA."""
    renderer = _make_ovrtx_renderer_without_backend(num_envs=3)
    positions = np.array([[0.0, 0.0, 0.0], [1.5, -2.0, 0.25], [3.0, 4.0, 0.5]])
    _queue_plan(
        renderer,
        ClonePlan(
            sources=("/World/envs/env_0/Robot", "/World/envs/env_1/Object"),
            destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"),
            clone_mask=np.array([[False, False, False], [False, True, True]]),
            env_ids=np.arange(3),
            positions=positions,
            is_complete=True,
            deformables=(SimpleNamespace(row=0),),
            _env_ids_cpu=(0, 1, 2),
        ),
        stage=_make_multi_env_stage(3),
    )

    renderer._clone_ctx.replicate(renderer._clone_ctx._sim.plan)

    assert _calls_of(renderer._clone_ctx, "clone") == [
        ("clone", "/World/envs/env_1/Object", ["/World/envs/env_2/Object"])
    ]
    layer = Sdf.Layer.CreateAnonymous("snapshot.usda")
    assert layer.ImportFromString(renderer._clone_ctx.stage_usda)
    assert layer.GetPrimAtPath("/World/envs/env_1/Object") is not None
    assert layer.GetPrimAtPath("/World/envs/env_2/Object") is None
    assert layer.GetPrimAtPath("/World/envs/env_0/Robot") is None


def test_active_unsupported_row_resets_materialized_targets_without_native_clone():
    """Materialized rows pin exact targets because no native clone can inherit source metadata."""
    num_envs = 3
    renderer = _make_ovrtx_renderer_without_backend(num_envs=num_envs)
    clone_mask = np.ones((2, num_envs), dtype=np.bool_)
    rigid_body_prototypes = (
        RigidBodyLayout(
            "/World/envs/env_{}/Robot",
            "/World/envs/env_*/Robot",
            0,
            None,
            "Robot",
            "/World/envs/env_0/Robot",
            clone_mask=clone_mask[0],
        ),
    )
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_0/Camera"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Camera"),
        clone_mask=clone_mask,
        env_ids=np.arange(num_envs),
        positions=np.zeros((num_envs, 3)),
        is_complete=True,
        rigid_body_prototypes=rigid_body_prototypes,
        deformables=(SimpleNamespace(row=0),),
        _env_ids_cpu=tuple(range(num_envs)),
    )
    _queue_plan(renderer, plan, stage=_make_multi_env_stage(num_envs))

    renderer._clone_ctx.replicate(renderer._clone_ctx._sim.plan)

    assert _calls_of(renderer._clone_ctx, "clone") == []
    assert _calls_of(renderer._clone_ctx, "reset_xform_stack") == [
        (
            "reset_xform_stack",
            [
                *(f"/World/envs/env_{env_id}/Robot" for env_id in range(num_envs)),
                *(f"/World/envs/env_{env_id}/Camera" for env_id in range(num_envs)),
            ],
        )
    ]


def test_snapshot_is_unavailable_before_replication():
    """Consumers cannot bypass the one clone lifecycle to serialize the stage."""
    renderer = _make_ovrtx_renderer_without_backend()

    with pytest.raises(RuntimeError, match="before clone replication completes"):
        _ = renderer._clone_ctx.stage_usda


def test_create_render_data_requires_a_bound_scene():
    """A camera built outside a replication session leaves the renderer without a scene."""
    renderer = _make_ovrtx_renderer_without_backend()

    with pytest.raises(RuntimeError, match="OVRTX has no scene to render"):
        renderer.create_render_data(_make_camera_render_spec(num_envs=1))


def test_replicate_serializes_stage_once_when_writing_debug_dump(tmp_path: Path):
    """Writing the final debug stage does not serialize the source stage a second time."""
    stage = _CountingStage(_make_multi_env_stage(2))
    renderer = _make_ovrtx_renderer_without_backend(num_envs=2)
    renderer._clone_ctx._ovrtx_key = (0, "", "", str(tmp_path))
    _queue_plan(renderer, _env_root_plan(2), stage=stage)

    renderer._clone_ctx.replicate(renderer._clone_ctx._sim.plan)

    assert stage.export_count == 1
    assert {path.name for path in tmp_path.iterdir()} == {_OVRTX_STAGE_FILE}


def test_camera_renderers_share_one_clone_context_and_stage_export(monkeypatch):
    """Multiple OVRTX camera clients consume one registry resource and one stage snapshot."""
    stage = _CountingStage(_make_multi_env_stage(2))
    simulation = object.__new__(SimulationContext)
    simulation.stage = stage
    simulation._backend_registry = {}
    simulation._backend_clone_roles = {}
    simulation._clone_plan = None
    simulation._renderer_entries = []
    simulation._renderers_initialized = False
    monkeypatch.setattr(SimulationContext, "_instance", simulation)

    first = simulation.get_renderer(OVRTXRendererCfg())
    second = simulation.get_renderer(OVRTXRendererCfg())
    context = first._clone_ctx
    context._ovrtx_key = (0, "", "", None)
    context._ovrtx_scene = _RecordingScene()
    _set_spec(first, _make_camera_render_spec(2))
    _set_spec(second, _make_camera_render_spec(2))
    _queue_plan(first, _env_root_plan(2), stage=stage)

    context.replicate(context._sim.plan)

    assert second._clone_ctx is context
    assert simulation._backend_registry == {OvReplicateContext: context}
    assert simulation._backend_clone_roles == {OvReplicateContext: {"scene"}}
    assert stage.export_count == 1


def test_shared_scene_writes_combined_stage_dump(tmp_path: Path):
    """The scene the renderer opens is the export plus its render product."""
    renderer = _make_ovrtx_renderer_without_backend(num_envs=1)
    renderer._clone_ctx._ovrtx_key = (0, "", "", str(tmp_path))
    _queue_plan(renderer, _env_root_plan(1), stage=_make_multi_env_stage(1))

    renderer._clone_ctx.replicate(renderer._clone_ctx._sim.plan)

    combined_text = (tmp_path / _OVRTX_STAGE_FILE).read_text(encoding="utf-8")
    assert combined_text.startswith("#usda 1.0")
    assert 'def Scope "Render_0"' in combined_text
    assert 'def RenderProduct "RenderProduct"' in combined_text
    assert [usd_text for _name, usd_text in _calls_of(renderer._clone_ctx, "open")] == [combined_text]


def test_replicate_refreshes_camera_relationship_after_cloning():
    """Replicating a multi-environment scene tags the copies and rewrites the RenderProduct cameras.

    The copies only exist once the rows have been copied, so the render product is pointed at them
    at the end of replication rather than when the scene is opened. None of it reads physics, so it
    does not wait for the simulation to be played.
    """
    num_envs = 4
    renderer = _make_ovrtx_renderer_without_backend(num_envs=num_envs)
    _queue_plan(renderer, _env_root_plan(num_envs), stage=_make_multi_env_stage(num_envs))

    renderer._clone_ctx.replicate(renderer._clone_ctx._sim.plan)

    camera_paths = [f"/World/envs/env_{env_id}/Camera" for env_id in range(num_envs)]
    env_paths = [f"/World/envs/env_{env_id}" for env_id in range(num_envs)]
    env_names = [f"env_{env_id}" for env_id in range(num_envs)]
    assert _calls_of(renderer._clone_ctx, "write_tokens") == [
        ("write_tokens", "primvars:omni:scenePartition", env_paths, env_names),
        ("write_tokens", "omni:scenePartition", camera_paths, env_names),
    ]
    assert _calls_of(renderer._clone_ctx, "point_render_products_at") == [
        ("point_render_products_at", ["/Render_0/RenderProduct"], camera_paths)
    ]
    assert renderer._render_product_paths == ["/Render_0/RenderProduct"]
    # The copies only exist once the rows are copied, so the scene is only told about them after.
    call_names = [call[0] for call in renderer._clone_ctx.scene.calls]
    assert call_names.index("clone") < call_names.index("point_render_products_at")
    # The camera transforms are bound before anything can render, without waiting for physics.
    assert renderer._camera_xforms is not None
    assert renderer._camera_xforms.paths == camera_paths


def test_replicate_targets_the_plan_camera_paths_without_reconstructing_them(monkeypatch: pytest.MonkeyPatch):
    """OVRTX binds exact camera destinations even outside its former fixed relative path."""
    num_envs = 4
    camera_env_ids = (1, 3)
    stage = _make_multi_env_stage(1)
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/SensorRig")
    UsdGeom.Camera.Define(stage, "/World/envs/env_0/SensorRig/Optical")
    renderer = _make_ovrtx_renderer_without_backend(num_envs=num_envs)
    _set_spec(
        renderer,
        CameraRenderSpec(
            cfg=renderer._spec.cfg,
            device="cpu",
            camera_source_prim_paths=("/World/envs/env_0/SensorRig/Optical",),
            camera_prim_paths=tuple(f"/World/envs/env_{env_id}/SensorRig/Optical" for env_id in camera_env_ids),
        ),
    )
    _queue_plan(renderer, _env_root_plan(num_envs), stage=stage)
    monkeypatch.setattr(
        "isaaclab_ov.cloner.replicate.cloner.path.under",
        lambda *_args: pytest.fail("Camera partitions must not search every environment root."),
    )

    renderer._clone_ctx.replicate(renderer._clone_ctx._sim.plan)

    camera_paths = list(renderer._spec.camera_prim_paths)
    assert _calls_of(renderer._clone_ctx, "write_tokens")[1] == (
        "write_tokens",
        "omni:scenePartition",
        camera_paths,
        [f"env_{env_id}" for env_id in camera_env_ids],
    )
    assert _calls_of(renderer._clone_ctx, "point_render_products_at") == [
        ("point_render_products_at", ["/Render_0/RenderProduct"], camera_paths)
    ]
    assert renderer._camera_xforms.paths == camera_paths


def test_replicate_binds_the_camera_of_a_single_environment_scene():
    """Regression: a scene of one env has nothing to copy, but its camera must still be bound.

    Replication returns early from the copy loop when no row has a target. The camera transform
    binding is not part of that loop, so a single-env scene would otherwise render from a camera
    nothing ever moves -- a still image with no error to explain it.
    """
    renderer = _make_ovrtx_renderer_without_backend(num_envs=1)
    _queue_plan(renderer, _env_root_plan(1), stage=_make_multi_env_stage(1))

    renderer._clone_ctx.replicate(renderer._clone_ctx._sim.plan)

    assert _calls_of(renderer._clone_ctx, "clone") == []
    assert renderer._camera_xforms is not None
    assert renderer._camera_xforms.paths == ["/World/envs/env_0/Camera"]
    assert _calls_of(renderer._clone_ctx, "write_tokens") == [
        ("write_tokens", "primvars:omni:scenePartition", ["/World/envs/env_0"], ["env_0"]),
        ("write_tokens", "omni:scenePartition", ["/World/envs/env_0/Camera"], ["env_0"]),
    ]
    assert _calls_of(renderer._clone_ctx, "point_render_products_at") == [
        ("point_render_products_at", ["/Render_0/RenderProduct"], ["/World/envs/env_0/Camera"])
    ]


def test_ovrtx_writes_each_clean_and_dirty_transform_generation_once(monkeypatch: pytest.MonkeyPatch):
    """OVRTX writes each object and named camera generation exactly once."""

    class Backend(SceneDataBackend):
        def __init__(self):
            poses = np.array([[1, 2, 3, 0, 0, 0, 1], [-4, 5, 6, 0, 0, 0, 1]], dtype=np.float32)
            self.data = SceneDataFormat.Transform()
            self.data.transforms = wp.array(poses, dtype=wp.transformf, device="cpu")
            self.publication = SceneDataPublication(self.data, True)

        @property
        def transform_publication(self) -> SceneDataPublication:
            return self.publication

        @property
        def point_publications(self) -> dict:
            return {}

    backend = Backend()
    provider = SceneDataProvider(backend)
    camera_publication = SceneDataPublication(backend.data, True)
    provider.register_transforms("camera", camera_publication)
    conversions = []
    convert = provider._convert_transforms

    def record_conversion(*args):
        conversions.append(args)
        convert(*args)

    monkeypatch.setattr(provider, "_convert_transforms", record_conversion)
    paths = ["/World/envs/env_0/Object", "/World/envs/env_1/Object"]
    rigid_body_prototypes = (
        RigidBodyLayout(
            "/World/envs/env_{}/Object",
            "/World/envs/env_*/Object",
            0,
            None,
            "Object",
            "/World/envs/env_0/Object",
            clone_mask=np.ones(2, dtype=np.bool_),
        ),
    )
    plan = replace(
        _env_root_plan(2),
        sources=("/World/envs/env_0/Object",),
        destinations=("/World/envs/env_{}/Object",),
        is_complete=True,
        rigid_body_prototypes=rigid_body_prototypes,
        _env_ids_cpu=(0, 1),
    )
    sim = SimpleNamespace(
        physics_backend="ovphysx",
        get_scene_data_provider=lambda: provider,
        get_clone_plan=lambda: plan,
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: sim)
    renderer = _make_ovrtx_renderer_without_backend(num_envs=2)
    context = _queue_plan(renderer, plan, stage=_make_multi_env_stage(2))
    context.replicate(context._sim.plan)

    renderer.initialize()

    assert renderer._clone_ctx._scene_data_provider is provider
    assert _calls_of(renderer._clone_ctx, "bind")[-1] == ("bind", paths)
    monkeypatch.setattr(SimulationContext, "instance", lambda: pytest.fail("per-frame global context lookup"))
    render_data = SimpleNamespace(transform_stream="camera")
    renderer.update(render_data, object())
    renderer.update(render_data, object())

    expected = np.tile(np.eye(4), (2, 1, 1))
    expected[:, 3, :3] = [[1, 2, 3], [-4, 5, 6]]
    np.testing.assert_array_equal(renderer._clone_ctx.scene.transforms, expected)
    assert provider.transform_generation() == 1
    assert len(conversions) == 2
    assert all(conversion[1]._cls is SceneDataFormat.TransposedMatrix44d for conversion in conversions)
    assert renderer._clone_ctx.scene.transform_pointers[0] == conversions[0][1].matrices.__array_interface__["data"][0]
    assert len(set(renderer._clone_ctx.scene.transform_pointers)) == 2
    assert renderer._clone_ctx.scene.transform_paths.count(tuple(paths)) == 1
    assert renderer._clone_ctx.scene.transform_paths.count(tuple(renderer._spec.camera_prim_paths)) == 1

    backend.publication.dirty = True
    camera_publication.dirty = True
    renderer.update(render_data, object())

    assert provider.transform_generation() == 2
    assert provider.transform_generation("camera") == 2
    assert renderer._clone_ctx.scene.transform_paths.count(tuple(paths)) == 2
    assert renderer._clone_ctx.scene.transform_paths.count(tuple(renderer._spec.camera_prim_paths)) == 2


def test_replicating_twice_into_one_context_fails_loudly():
    """One context replicates exactly once and rejects late camera clients."""
    renderer = _make_ovrtx_renderer_without_backend(num_envs=2)
    plan = _env_root_plan(2)
    _queue_plan(renderer, plan, stage=_make_multi_env_stage(2))
    renderer._clone_ctx.replicate(renderer._clone_ctx._sim.plan)

    with pytest.raises(RuntimeError, match="exactly once"):
        renderer._clone_ctx.replicate(renderer._clone_ctx._sim.plan)

    with pytest.raises(RuntimeError, match="cannot join.*after replication"):
        renderer._clone_ctx._add_renderer(object())


def test_snapshot_carries_the_planned_prototype():
    """The exact prototype named by the plan is present in OVRTX's transport payload."""
    num_envs = 4
    renderer = _make_ovrtx_renderer_without_backend(num_envs=num_envs)
    _queue_plan(renderer, _env_root_plan(num_envs), stage=_make_multi_env_stage(num_envs))

    exported = _replicate_and_snapshot(renderer._clone_ctx)

    assert 'def Xform "env_0"' in exported
    assert 'def Xform "Robot"' in exported
    assert 'def Xform "Object_env0_only"' not in exported


def test_snapshot_does_not_mutate_live_stage_with_renderer_metadata():
    """Renderer-specific tokens are authored only in each consumer's OVStage."""
    stage = _make_multi_env_stage(1)
    UsdGeom.Camera.Define(stage, "/World/envs/env_0/UnplannedCamera")
    renderer = _make_ovrtx_renderer_without_backend(num_envs=2)
    _queue_plan(renderer, _env_root_plan(2), stage=stage)

    renderer._clone_ctx.replicate(renderer._clone_ctx._sim.plan)

    root_layer = stage.GetRootLayer()
    env_attr = root_layer.GetAttributeAtPath(
        Sdf.Path("/World/envs/env_0").AppendProperty("primvars:omni:scenePartition")
    )
    camera_attr = root_layer.GetAttributeAtPath(
        Sdf.Path("/World/envs/env_0/Camera").AppendProperty("omni:scenePartition")
    )
    unplanned_attr = root_layer.GetAttributeAtPath(
        Sdf.Path("/World/envs/env_0/UnplannedCamera").AppendProperty("omni:scenePartition")
    )
    assert env_attr is None
    assert camera_attr is None
    assert unplanned_attr is None


def test_shared_snapshot_excludes_renderer_world_space_overrides():
    """OVPhysX's shared snapshot must not receive OVRTX-only reset-stack state."""
    renderer = _make_ovrtx_renderer_without_backend(num_envs=2)
    _queue_plan(renderer, _env_root_plan(2), stage=_make_multi_env_stage(2))

    renderer._clone_ctx.replicate(renderer._clone_ctx._sim.plan)

    layer = Sdf.Layer.CreateAnonymous("shared-ov-snapshot.usda")
    assert layer.ImportFromString(renderer._clone_ctx.stage_usda)
    snapshot = Usd.Stage.Open(layer)
    robot = UsdGeom.Xformable(snapshot.GetPrimAtPath("/World/envs/env_0/Robot"))
    assert not robot.GetResetXformStack()


def test_snapshot_never_walks_the_finished_stage_or_exposes_an_export_fallback():
    """Architecture gate: the snapshot gates exact plan paths and has one lifecycle entry point."""
    source = inspect.getsource(OvReplicateContext)
    assert all(discovery not in source for discovery in ("PrimRange", "Traverse"))
    assert not hasattr(OvReplicateContext, "export_stage")


def test_snapshot_is_plan_gated_for_a_single_env():
    """A single environment does not exempt unplanned stage content from the plan gate."""
    renderer = _make_ovrtx_renderer_without_backend(num_envs=1)
    _queue_plan(renderer, _env_root_plan(1), stage=_make_multi_env_stage(1))

    exported = _replicate_and_snapshot(renderer._clone_ctx)
    assert 'def Xform "Robot"' in exported
    assert 'def Xform "Object"' not in exported
    assert 'def Camera "Camera"' not in exported


def test_ov_context_never_claims_whole_environment_cloning():
    """ReplicateSession authors environment roots and passes asset-level rows to OV."""
    assert OvReplicateContext.clones_whole_env is False


def test_snapshot_retains_structural_env_roots_without_exporting_their_content():
    """The snapshot retains ReplicateSession's roots while gating their children by the plan."""
    num_envs, num_prototypes = 6, 2
    stage = _make_multi_env_stage(num_envs)
    renderer = _make_ovrtx_renderer_without_backend(num_envs=num_envs)
    _queue_plan(
        renderer,
        ClonePlan(
            sources=tuple(f"/World/envs/env_{i}/Object_env{i}_only" for i in range(num_prototypes)),
            destinations=("/World/envs/env_{}/Object_env0_only", "/World/envs/env_{}/Object_env1_only"),
            clone_mask=np.array(
                [[True, True, True, False, False, False], [False, False, False, True, True, True]],
                dtype=np.bool_,
            ),
            positions=np.zeros((num_envs, 3)),
        ),
        stage=stage,
    )

    exported = _replicate_and_snapshot(renderer._clone_ctx)

    for env_idx in range(num_envs):
        assert f'"env_{env_idx}"' in exported, f"env_{env_idx} missing from the exported scene"


def test_snapshot_carries_every_source_of_a_partially_shared_scene():
    """Every prototype named by a partially shared plan is present in the transport payload."""
    num_envs = 4
    renderer = _make_ovrtx_renderer_without_backend(num_envs=num_envs)
    _queue_plan(
        renderer,
        ClonePlan(
            sources=("/World/envs/env_0/Object_env0_only", "/World/envs/env_3/Object_env3_only"),
            destinations=("/World/envs/env_{}/Object_env0_only", "/World/envs/env_{}/Object_env3_only"),
            clone_mask=np.array(
                [[True, True, False, False], [False, False, True, True]],
                dtype=np.bool_,
            ),
            positions=np.zeros((num_envs, 3)),
        ),
        stage=_make_multi_env_stage(num_envs),
    )

    exported = _replicate_and_snapshot(renderer._clone_ctx)

    assert 'def Xform "Object_env0_only"' in exported
    assert 'def Xform "Object_env3_only"' in exported
