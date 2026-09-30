# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Non-GPU tests for the backend lifecycle matrix harness."""

from __future__ import annotations

import ast
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "benchmarks" / "benchmark_backend_lifecycle.py"
SPEC = importlib.util.spec_from_file_location("benchmark_backend_lifecycle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
LIFECYCLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LIFECYCLE
SPEC.loader.exec_module(LIFECYCLE)


def test_matrix_has_only_the_ov_kit_exclusion() -> None:
    cases = LIFECYCLE.enumerate_cases()
    excluded = [case for case in cases if LIFECYCLE.exclusion_reason(case) is not None]

    assert len(cases) == 90
    assert len(excluded) == 17
    assert LIFECYCLE.matrix_manifest()["runnable_cases"] == 73
    for case in cases:
        has_ov = case.physics == "ovphysx" or case.renderer == "ovrtx"
        has_kit = case.physics == "isaacsim_physx" or case.renderer == "isaacsim_rtx" or case.visualizer == "kit"
        assert (LIFECYCLE.exclusion_reason(case) is not None) == (has_ov and has_kit)


def test_default_run_selects_every_non_excluded_case(monkeypatch) -> None:
    selected = []
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    monkeypatch.setattr(LIFECYCLE, "_run_parent", lambda cases, _args: selected.extend(cases) or 0)

    assert LIFECYCLE.main() == 0
    assert len(selected) == 73
    assert all(LIFECYCLE.exclusion_reason(case) is None for case in selected)


def test_point_run_covers_cross_backend_native_family_and_mpm_publishers(monkeypatch) -> None:
    selected = []
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--points"])
    monkeypatch.setattr(LIFECYCLE, "_run_parent", lambda cases, _args: selected.extend(cases) or 0)

    assert LIFECYCLE.main() == 0
    assert tuple(selected) == LIFECYCLE.POINT_CASES
    point_physics = tuple(LIFECYCLE.POINT_ASSET_MODULES)
    expected_pairs = {
        (physics, renderer)
        for physics, renderer in itertools.product(point_physics, LIFECYCLE.RENDERERS)
        if LIFECYCLE.exclusion_reason(LIFECYCLE.Case(physics, renderer, "none")) is None
    }
    assert LIFECYCLE.point_manifest()["total_cases"] == len(expected_pairs) == 10
    assert {(case.physics, case.renderer) for case in selected} == expected_pairs
    assert all(LIFECYCLE.exclusion_reason(case) is None for case in selected)
    assert {case.visualizer for case in selected} == {"newton_gl", "newton_rtx", "rerun", "viser", "kit"}
    assert LIFECYCLE.Case("newton_mpm", "newton_renderer", "newton_gl") in selected
    assert LIFECYCLE.Case("ovphysx", "ovrtx", "viser") in selected
    assert LIFECYCLE.Case("isaacsim_physx", "isaacsim_rtx", "kit") in selected


def test_cartpole_presets_resolve_into_flat_direct_cfg() -> None:
    cfg = LIFECYCLE._resolve_env_cfg(LIFECYCLE.Case("newton_mjwarp", "newton_renderer", "none"), "cpu")

    assert isinstance(cfg, LIFECYCLE.LifecycleDirectCfg)
    assert cfg.scene is None
    assert cfg.ground.prim_path == "/World/ground"
    assert cfg.robot.prim_path.endswith("/Robot")
    assert cfg.camera.prim_path.endswith("/Camera")
    assert cfg.light.prim_path == "/World/Light"
    assert cfg.marker.prim_path == "/Visuals/LifecycleMarker"
    assert cfg.sim.device == "cpu"


def test_environment_count_is_resolved_into_the_direct_cfg() -> None:
    case = LIFECYCLE.Case("newton_mjwarp", "newton_renderer", "none")

    assert LIFECYCLE._resolve_env_cfg(case, "cpu", num_envs=3).num_envs == 3
    assert LIFECYCLE._resolve_point_cfg(LIFECYCLE.POINT_CASES[0], "cpu", num_envs=3).num_envs == 3


def test_newton_rtx_resolves_the_direct_cfg_planned_camera() -> None:
    case = LIFECYCLE.Case("newton_mjwarp", "newton_renderer", "newton_rtx")

    cfg = LIFECYCLE._resolve_env_cfg(case, "cpu")

    assert cfg.sim.visualizer_cfgs.streaming_camera == cfg.camera.prim_path
    assert cfg.sim.visualizer_cfgs.streaming_gt_types == ("rgb",)
    assert "rgb" in cfg.camera.data_types


@pytest.mark.parametrize("case", LIFECYCLE.POINT_CASES, ids=lambda case: case.name)
def test_point_presets_resolve_into_flat_direct_cfg(case) -> None:
    cfg = LIFECYCLE._resolve_point_cfg(case, "cpu")

    assert isinstance(cfg, LIFECYCLE.PointLifecycleDirectCfg)
    assert cfg.scene is None
    assert cfg.deformable.prim_path.endswith("/Deformable")
    assert cfg.deformable.visualizer_cfg is None
    assert cfg.camera.prim_path.endswith("/Camera")
    assert cfg.sim.visualizer_cfgs.visualizer_type == case.visualizer
    assert cfg.sim.device == "cpu"


def test_point_direct_cfg_contributes_only_declared_scene_rows_to_the_clone_plan() -> None:
    from isaaclab.cloner import make_clone_plan

    cfg = LIFECYCLE._resolve_point_cfg(LIFECYCLE.POINT_CASES[0], "cpu")
    plan = make_clone_plan(
        (cfg.deformable, cfg.camera),
        num_clones=cfg.num_envs,
        env_spacing=cfg.env_spacing,
        env_template=cfg.clone_cfg.clone_template,
    )

    assert plan.sources == ("/World/envs/env_0/Deformable", "/World/envs/env_0/Camera")
    assert plan.cfg_rows[id(cfg.deformable)] == (0,)
    assert plan.cfg_rows[id(cfg.camera)] == (1,)


@pytest.mark.parametrize(
    ("renderer", "expected_type"),
    [
        ("newton_renderer", "NewtonWarpRendererCfg"),
        ("ovrtx", "OVRTXRendererCfg"),
        ("isaacsim_rtx", "IsaacRtxRendererCfg"),
    ],
)
def test_camera_owns_renderer_selection_through_the_typed_selector(renderer: str, expected_type: str) -> None:
    cfg = LIFECYCLE._resolve_env_cfg(LIFECYCLE.Case("newton_mjwarp", renderer, "none"), "cpu")

    assert type(cfg.camera.renderer_cfg).__name__ == expected_type


@pytest.mark.parametrize("visualizer", LIFECYCLE.VISUALIZERS)
def test_visualizers_resolve_from_the_direct_cfg_through_the_typed_selector(visualizer: str) -> None:
    cfg = LIFECYCLE._resolve_env_cfg(LIFECYCLE.Case("newton_mjwarp", "newton_renderer", visualizer), "cpu")

    if visualizer == "none":
        assert cfg.sim.visualizer_cfgs == []
    else:
        assert cfg.sim.visualizer_cfgs.visualizer_type == visualizer


def test_kit_visualizer_references_the_declared_clone_plan_camera() -> None:
    from isaaclab.cloner import make_clone_plan

    cfg = LIFECYCLE._resolve_env_cfg(LIFECYCLE.Case("newton_mjwarp", "newton_renderer", "kit"), "cpu")
    plan = make_clone_plan(
        (cfg.ground, cfg.robot, cfg.camera, cfg.light, cfg.marker),
        num_clones=cfg.num_envs,
        env_spacing=cfg.env_spacing,
        env_template=cfg.clone_cfg.clone_template,
    )

    assert cfg.sim.visualizer_cfgs.streaming_camera == cfg.camera.prim_path
    assert plan.cfg_rows[id(cfg.camera)]
    assert id(cfg.sim.visualizer_cfgs) not in plan.cfg_rows


def test_harness_has_no_scene_manager_and_declares_every_planned_cfg() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cfg_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LifecycleDirectCfg")
    declared = {
        target.id
        for node in cfg_class.body
        if isinstance(node, ast.AnnAssign) and isinstance((target := node.target), ast.Name)
    }

    assert "InteractiveScene" not in source
    assert "setup_scene" not in source
    assert "GetPrimAtPath" not in source
    assert "PrimRange" not in source
    assert ".clone_context(" not in source
    assert ".build(" not in source
    assert ".build_visualizer(" not in source
    assert "_make_visualizer_cfg" not in source
    assert "resolve_task_config" not in source
    assert "env_cfg.sim.visualizer_cfgs =" not in source
    assert {"scene", "num_envs", "env_spacing", "clone_cfg", "ground", "robot", "camera", "light", "marker"} <= declared
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in cfg_class.body)

    assert "MultiBackendVisualizerCfg(" in source
    assert not [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name.endswith("VisualizerCfg")]

    for class_name in ("PointPhysicsCfg", "PointDeformableCfg", "PointLifecycleDirectCfg"):
        point_cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in point_cfg.body)


def test_harness_runtime_has_one_plan_and_only_exact_cfg_construction() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    sessions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ReplicateSession"
    ]
    worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_worker")
    runtime_calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call)]
    constructions = {(ast.unparse(call.func), tuple(ast.unparse(arg) for arg in call.args)) for call in runtime_calls}

    assert len(sessions) == 1
    assert [ast.unparse(arg) for arg in sessions[0].args] == ["(env_cfg,)"]
    assert ("env_cfg.robot.class_type", ("env_cfg.robot",)) in constructions
    assert ("env_cfg.deformable.class_type", ("env_cfg.deformable",)) in constructions
    assert ("env_cfg.camera.class_type", ("env_cfg.camera",)) in constructions
    assert ("env_cfg.marker.class_type", ("env_cfg.marker",)) in constructions
    assert not [
        ast.unparse(call) for call in runtime_calls if ast.unparse(call.func).rsplit(".", 1)[-1].endswith("Cfg")
    ]
    launch_context = next(
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.With)
        and any(ast.unparse(item.context_expr.func) == "launch_simulation" for item in node.items)
    )
    assert any(
        isinstance(node, ast.Call) and ast.unparse(node.func) == "write_report" for node in ast.walk(launch_context)
    )


def test_manifest_trace_targets_are_selected_without_runtime_imports() -> None:
    newton = LIFECYCLE._manifest_trace_targets(LIFECYCLE.Case("newton_vbd", "newton_renderer", "none"))
    ovrtx = LIFECYCLE._manifest_trace_targets(LIFECYCLE.Case("ovphysx", "ovrtx", "none"))

    assert ("isaaclab.scene_data.scene_data_provider", "SceneDataProvider.request_transforms") in newton
    assert ("isaaclab.scene_data.scene_data_provider", "SceneDataProvider._convert_transforms") in newton
    assert ("isaaclab_newton.physics.newton_manager", "NewtonManager.start_simulation") in newton
    assert (
        "isaaclab_newton.cloner.replicate",
        "NewtonReplicateContext.finalize_visualization_model",
    ) in newton
    assert ("isaaclab_newton.renderers.newton_warp_renderer", "NewtonWarpRenderer.update") in newton
    assert ("isaaclab_newton.renderers.newton_warp_renderer", "NewtonWarpRenderer.render") in newton
    assert not any(function.startswith("SceneDataProvider.sync_") for _, function in newton)
    assert ("isaaclab_ov.renderers.ovrtx_scene", "OvrtxScene.write_xforms") not in newton
    assert ("isaaclab_newton.assets.articulation.articulation", "Articulation.__init__") not in ovrtx
    assert ("isaaclab_ov.physics.ovphysx_manager", "OvPhysxManager._warmup_and_load") in ovrtx
    assert ("isaaclab_ov.physics.ovphysx_manager", "OvPhysxManager._attach_ovstage") in ovrtx
    assert ("isaaclab_ov.renderers.ovrtx_scene", "OvrtxScene.write_xforms") in ovrtx
    assert ("isaaclab_ov.renderers.ovrtx_renderer", "OVRTXRenderer.update") in ovrtx
    assert ("isaaclab_ov.cloner.replicate", "OvReplicateContext._snapshot_stage") in ovrtx


def test_newton_rtx_trace_has_no_native_viewer_or_newton_scene_resource() -> None:
    case = LIFECYCLE.Case("ovphysx", "ovrtx", "newton_rtx")

    modules = LIFECYCLE._lifecycle_modules(case)
    assert "newton._src.viewer.viewer_rtx" not in modules
    assert "isaaclab_newton.cloner.replicate" not in modules


def test_construction_gate_records_only_the_exact_cfg() -> None:
    class Visualizer:
        def __init__(self, cfg):
            self.cfg = cfg

    cfg = object()
    tracer = LIFECYCLE._CallTracer(LIFECYCLE.Case("newton_mjwarp", "newton_renderer", "none"), 10)
    tracer.patch_method(Visualizer, "__init__", "visualizer.Visualizer.__init__")
    visualizer = Visualizer(cfg)

    assert visualizer.cfg is cfg
    assert tracer.construction_calls[0]["argument_ids"] == [hex(id(cfg))]
    snapshot = LIFECYCLE._construction_snapshot(visualizer, cfg, tracer)
    assert snapshot["exact_cfg_construction"]
    assert snapshot["construction_ms"] is not None
    assert not LIFECYCLE._construction_snapshot(Visualizer, cfg, tracer)["exact_cfg_construction"]


def test_conversion_trace_keys_each_sdp_move_by_generation_and_format() -> None:
    class InputFormat:
        __name__ = "InputFormat"

    class OutputFormat:
        __name__ = "OutputFormat"

    class Provider:
        _generations = {("transforms", "camera"): 3}

        def _convert_transforms(self, input, output, count, name):
            pass

    input = type("Input", (), {"_cls": InputFormat})()
    output = type("Output", (), {"_cls": OutputFormat})()
    tracer = LIFECYCLE._CallTracer(LIFECYCLE.Case("newton_mjwarp", "newton_renderer", "none"), 10)
    tracer.patch_method(Provider, "_convert_transforms", "SceneDataProvider._convert_transforms")

    Provider()._convert_transforms(input, output, 1, "camera")

    assert tracer.scene_data_conversions == [
        {
            "kind": "transforms",
            "stream": "camera",
            "generation": 3,
            "input_format": "InputFormat",
            "output_format": "OutputFormat",
        }
    ]


def test_conversion_trace_records_point_stream_and_generation() -> None:
    InputFormat = type("Points", (), {})
    OutputFormat = type("FabricMeshPoints", (), {})
    input = type("Input", (), {"_cls": InputFormat})()
    output = type("Output", (), {"_cls": OutputFormat})()

    class Provider:
        def __init__(self):
            self._backend = SimpleNamespace(point_publications={"particles": SimpleNamespace(data=input)})
            self._generations = {("points", "particles"): 4}

        def _convert_points(self, input, output, count):
            pass

    tracer = LIFECYCLE._CallTracer(LIFECYCLE.Case("newton_mjwarp", "newton_renderer", "none"), 10)
    tracer.patch_method(Provider, "_convert_points", "SceneDataProvider._convert_points")

    Provider()._convert_points(input, output, 12)

    assert tracer.scene_data_conversions == [
        {
            "kind": "points",
            "stream": "particles",
            "generation": 4,
            "input_format": "Points",
            "output_format": "FabricMeshPoints",
        }
    ]


def test_geometry_kernel_trace_distinguishes_requested_sdp_work() -> None:
    InputFormat = type("BodyPoints", (), {})
    OutputFormat = type("Points", (), {})
    input = type("Input", (), {"_cls": InputFormat})()
    output = type("Output", (), {"_cls": OutputFormat})()

    def geometry_kernel():
        pass

    geometry_kernel.__module__ = "isaaclab.scene_data.geometry_points"
    kernel = SimpleNamespace(func=geometry_kernel, key="body_points_to_points_kernel")
    tracer = LIFECYCLE._CallTracer(LIFECYCLE.POINT_CASES[0], 10, points=True)

    class Provider:
        def __init__(self):
            self._backend = SimpleNamespace(point_publications={"points": SimpleNamespace(data=input)})
            self._generations = {("points", "points"): 1}

        def _convert_points(self, input, output, count):
            tracer.record_geometry_launch(kernel)

    tracer.patch_method(Provider, "_convert_points", "SceneDataProvider._convert_points")
    Provider()._convert_points(input, output, 1)
    tracer.record_geometry_launch(kernel)

    assert tracer.geometry_launches == [
        {"kernel": "body_points_to_points_kernel", "phase": "trace_install", "inside_sdp": True},
        {"kernel": "body_points_to_points_kernel", "phase": "trace_install", "inside_sdp": False},
    ]


def test_point_snapshot_separates_logical_conversions_from_payload_copies() -> None:
    class Points:
        vars = {"points": object()}

    class HostPoints:
        vars = {"points": object()}

    source_buffer = SimpleNamespace(ptr=11, shape=(4,), device="cuda:0")
    converted_buffer = SimpleNamespace(ptr=12, shape=(4,), device="cpu")
    source = SimpleNamespace(_cls=Points, points=source_buffer)
    converted = SimpleNamespace(_cls=HostPoints, points=converted_buffer)
    publication = SimpleNamespace(
        data=source,
        dirty=True,
    )
    provider = SimpleNamespace(
        _backend=SimpleNamespace(point_publications={"points": publication}),
        _cache={
            ("points", "points", Points): (3, source),
            ("points", "points", HostPoints): (3, converted),
        },
        _fabric_point_outputs={"points": object()},
    )
    tracer = SimpleNamespace(
        scene_data_conversions=[
            {
                "kind": "points",
                "stream": "points",
                "generation": 3,
                "input_format": "Points",
                "output_format": "HostPoints",
            }
        ]
    )
    binding = SimpleNamespace(
        path="/World/envs/env_0/Deformable",
        source_offset=0,
        source_count=4,
        output_offset=0,
        output_count=4,
    )
    layout = SimpleNamespace(point_bindings=lambda name: (binding,))

    snapshot = LIFECYCLE._point_data_snapshot(provider, tracer, layout)

    assert snapshot["streams"][0]["dirty"]
    assert snapshot["streams"][0]["source"]["fields"]["points"]["pointers"] == ["0xb"]
    assert [row["logical_conversion_count"] for row in snapshot["requests"]] == [0, 1]
    assert [row["payload_copy_count"] for row in snapshot["requests"]] == [0, 1]
    assert [row["exact_aliases_publisher"] for row in snapshot["requests"]] == [True, False]
    assert [row["payload_aliases_publisher"] for row in snapshot["requests"]] == [True, False]
    assert snapshot["fabric_requests"] == []
    assert snapshot["zero_or_one_payload_copy"]
    assert snapshot["exact_conversions"]


def test_sdp_contract_is_derived_from_formats_and_payload_pointers() -> None:
    class Input:
        vars = {"payload": object()}

    class Output:
        vars = {"payload": object()}

    source_buffer = SimpleNamespace(ptr=11, shape=(4,), device="cuda:0")
    output_buffer = SimpleNamespace(ptr=12, shape=(4,), device="cuda:0")
    source = SimpleNamespace(_cls=Input, payload=source_buffer)
    output = SimpleNamespace(_cls=Output, payload=output_buffer)

    native = LIFECYCLE._sdp_request_contract(source, source, Input, 0)
    converted = LIFECYCLE._sdp_request_contract(source, output, Output, 1)
    invalid_alias = LIFECYCLE._sdp_request_contract(source, source, Output, 0)

    assert native["pointer_contract"] and native["exact_payload_copy_count"]
    assert converted["pointer_contract"] and converted["exact_payload_copy_count"]
    assert not invalid_alias["pointer_contract"]
    assert not invalid_alias["exact_payload_copy_count"]


def test_sdp_contract_reports_index_enrichment_without_a_payload_copy() -> None:
    transform = SimpleNamespace(ptr=11, shape=(4,), device="cuda:0")
    indices = SimpleNamespace(ptr=12, shape=(4,), device="cuda:0")
    source = SimpleNamespace(_cls=LIFECYCLE.SceneDataFormat.Transform, transforms=transform)
    output = SimpleNamespace(
        _cls=LIFECYCLE.SceneDataFormat.IndexedTransform,
        transforms=transform,
        source_indices=indices,
    )

    contract = LIFECYCLE._sdp_request_contract(source, output, LIFECYCLE.SceneDataFormat.IndexedTransform, 0)

    assert contract["logical_conversion_count"] == 1
    assert contract["payload_copy_count"] == contract["expected_payload_copy_count"] == 0
    assert contract["payload_aliases_publisher"] and not contract["exact_aliases_publisher"]
    assert contract["identity_index_attachment"]
    assert contract["exact_payload_copy_count"] and contract["pointer_contract"]


def test_scene_data_request_records_the_exact_consumer_and_provider() -> None:
    Output = type("Output", (), {})

    class Provider:
        def request_transforms(self, output_format, name=None):
            return output_format()

    class Renderer:
        def __init__(self, provider):
            self.provider = provider

        def update(self, _render_data, _intrinsics):
            return self.provider.request_transforms(Output, name="camera")

    class Sensor:
        def __init__(self, renderer):
            self.renderer = renderer

        def update(self):
            return self.renderer.update(None, None)

    provider = Provider()
    renderer = Renderer(provider)
    sensor = Sensor(renderer)
    tracer = LIFECYCLE._CallTracer(LIFECYCLE.Case("newton_mjwarp", "newton_renderer", "none"), 10)
    tracer.phase = "movement_render_probe"
    tracer.patch_method(Provider, "request_transforms", "IsaacLab-SDP.SceneDataProvider.request_transforms")
    tracer.patch_method(Renderer, "update", "lifecycle.Renderer.update")
    tracer.patch_method(Sensor, "update", "IsaacLab-Sensors.SensorBase.update")

    output = sensor.update()

    assert tracer.scene_data_requests == [
        {
            "kind": "transforms",
            "phase": "movement_render_probe",
            "provider_id": hex(id(provider)),
            "consumer_id": hex(id(renderer)),
            "stream": "camera",
            "format": "Output",
            "output": LIFECYCLE._describe(output),
        }
    ]


def test_request_count_gate_rejects_missing_duplicate_and_unknown_stream_requests() -> None:
    expected = {None: 2, "camera": 2}
    exact = [{"stream": stream} for stream in (None, "camera", None, "camera")]

    assert LIFECYCLE._has_exact_request_counts(exact, expected)
    assert not LIFECYCLE._has_exact_request_counts(exact[:-1], expected)
    assert not LIFECYCLE._has_exact_request_counts(exact + [{"stream": "camera"}], expected)
    assert not LIFECYCLE._has_exact_request_counts(exact[:-1] + [{"stream": "other"}], expected)


def test_point_request_is_attributed_to_the_visualizer() -> None:
    Points = type("Points", (), {})

    class Provider:
        def request_points(self, output_format, name="points"):
            return output_format()

    class Visualizer:
        def __init__(self, provider):
            self.provider = provider

        def step(self, _dt):
            return self.provider.request_points(Points)

    provider = Provider()
    visualizer = Visualizer(provider)
    tracer = LIFECYCLE._CallTracer(LIFECYCLE.POINT_CASES[0], 10, points=True)
    tracer.phase = "point_render_probe"
    tracer.patch_method(Provider, "request_points", "IsaacLab-SDP.SceneDataProvider.request_points")
    tracer.patch_method(Visualizer, "step", "lifecycle.Visualizer.step")

    visualizer.step(1.0 / 60.0)

    assert tracer.scene_data_requests[0]["consumer_id"] == hex(id(visualizer))
    assert tracer.scene_data_requests[0]["stream"] == "points"


def test_newton_rtx_declares_only_the_exact_planned_camera_input() -> None:
    provider = object()
    camera = object()
    presenter = SimpleNamespace(
        _scene_data_provider=None,
        _streaming=SimpleNamespace(camera=camera),
    )
    requests = {"visualizer[0]": []}
    case = LIFECYCLE.Case("newton_mpm", "ovrtx", "newton_rtx")

    snapshot = LIFECYCLE._visualizer_input_snapshot(case, [presenter], provider, camera, requests)

    assert snapshot["declared"] == ["camera"]
    assert snapshot["visualizers"][0]["camera_bound"]
    assert not snapshot["visualizers"][0]["sdp_bound"]
    assert snapshot["visualizers"][0]["sdp_requests"] == 0
    assert snapshot["valid"]

    presenter._scene_data_provider = provider
    assert not LIFECYCLE._visualizer_input_snapshot(case, [presenter], provider, camera, requests)["valid"]
    presenter._scene_data_provider = None
    requests["visualizer[0]"] = [{}]
    assert not LIFECYCLE._visualizer_input_snapshot(case, [presenter], provider, camera, requests)["valid"]


def test_dormant_sdp_visualizer_keeps_its_declared_provider_link() -> None:
    provider = object()
    visualizer = SimpleNamespace(_scene_data_provider=provider, _streaming=None)
    requests = {"visualizer[0]": []}
    case = LIFECYCLE.Case("newton_mjwarp", "newton_renderer", "newton_gl")

    snapshot = LIFECYCLE._visualizer_input_snapshot(case, [visualizer], provider, object(), requests)

    assert snapshot["visualizers"][0]["sdp_bound"]
    assert snapshot["visualizers"][0]["sdp_requests"] == 0
    assert snapshot["valid"]

    visualizer._scene_data_provider = None
    assert not LIFECYCLE._visualizer_input_snapshot(case, [visualizer], provider, object(), requests)["valid"]


def test_clone_lifecycle_gate_requires_one_plan_and_dispatch() -> None:
    class Tracer:
        counts = {label: {"calls": 1} for label in LIFECYCLE.CLONE_LIFECYCLE_LABELS.values()}

    assert LIFECYCLE._clone_lifecycle_snapshot(Tracer())["exactly_one"]

    Tracer.counts[LIFECYCLE.CLONE_LIFECYCLE_LABELS["make_clone_plan"]]["calls"] = 2
    assert not LIFECYCLE._clone_lifecycle_snapshot(Tracer())["exactly_one"]


def test_registry_gate_requires_each_cfg_consumer_to_resolve_the_same_typed_resource() -> None:
    class NewtonReplicateContext:
        pass

    resource = NewtonReplicateContext()
    key = LIFECYCLE._describe(NewtonReplicateContext)["type"]
    backend = LIFECYCLE._describe(resource)
    tracer = type(
        "Tracer",
        (),
        {
            "backend_requests": [
                {"phase": "context_construction_and_backend_bind", "key": key, "backend": backend},
                {"phase": "clone_and_component_construction", "key": key, "backend": backend},
            ]
        },
    )()
    sim = type("Simulation", (), {"_backend_registry": {NewtonReplicateContext: resource}})()
    case = LIFECYCLE.Case("newton_mjwarp", "newton_renderer", "none")

    assert LIFECYCLE._registry_snapshot(case, sim, tracer)["one_resource_per_type"]

    tracer.backend_requests.pop()
    assert not LIFECYCLE._registry_snapshot(case, sim, tracer)["one_resource_per_type"]


def test_newton_finalization_gate_requires_only_the_resource_owning_path() -> None:
    physics_case = LIFECYCLE.Case("newton_mjwarp", "newton_renderer", "newton_gl")
    visualization_case = LIFECYCLE.Case("ovphysx", "newton_renderer", "viser")
    tracer = SimpleNamespace(
        lifecycle_events=[
            {"call": "trace.NewtonVBDManager.start_simulation", "logical_call": True},
            {"call": "trace.NewtonManager.start_simulation", "logical_call": False},
        ]
    )

    assert LIFECYCLE._newton_finalization_snapshot(physics_case, tracer)["exactly_once_when_required"]
    tracer.lifecycle_events.append({"call": "trace.NewtonManager.start_simulation", "logical_call": True})
    assert not LIFECYCLE._newton_finalization_snapshot(physics_case, tracer)["exactly_once_when_required"]

    tracer.lifecycle_events = [
        {"call": "trace.NewtonReplicateContext.finalize_visualization_model", "logical_call": True}
    ]
    assert LIFECYCLE._newton_finalization_snapshot(visualization_case, tracer)["exactly_once_when_required"]


def test_second_newton_hard_reset_requires_stable_native_identity_and_consumers() -> None:
    class NewtonReplicateContext:
        load_visual_shapes = True

        def __init__(self):
            self.model = object()
            self.state_0 = object()
            self._sensor_state = self.state_0
            self.state_1 = object()
            self.control = object()

        def get_model(self):
            return self.model

        def get_state_0(self):
            return self.state_0

        def get_state_1(self):
            return self.state_1

        def get_control(self):
            return self.control

    resource = NewtonReplicateContext()
    physics = SimpleNamespace(_newton=resource)
    renderer = SimpleNamespace(_newton_backend=resource, _newton_model=resource.model, _state=resource.state_0)
    visualizer = SimpleNamespace(_newton_backend=resource, _model=resource.model, _state=resource.state_0)

    class Simulation:
        _backend_registry = {NewtonReplicateContext: resource}
        replace_on_reset = False
        resets = 0

        def reset(self):
            self.resets += 1
            if self.replace_on_reset:
                resource.model = object()
                resource.state_0 = object()
                resource.state_1 = object()
                resource.control = object()

    sim = Simulation()
    case = LIFECYCLE.Case("newton_mjwarp", "newton_renderer", "newton_gl")

    stable = LIFECYCLE._newton_hard_reset_snapshot(case, sim, physics, renderer, [visualizer])
    assert sim.resets == 1
    assert stable["passed"]
    assert all(stable["stable"].values())

    visualizer._state = None
    assert LIFECYCLE._newton_resource_snapshot(case, sim, physics, renderer, [visualizer])["shared"]
    visualizer._state = object()
    assert not LIFECYCLE._newton_resource_snapshot(case, sim, physics, renderer, [visualizer])["shared"]
    visualizer._state = resource.state_0

    sim.replace_on_reset = True
    changed = LIFECYCLE._newton_hard_reset_snapshot(case, sim, physics, renderer, [visualizer])
    assert sim.resets == 2
    assert not changed["passed"]
    assert not changed["stable"]["model"]


def test_initialization_gate_requires_each_exact_component_after_replication() -> None:
    physics = object()
    renderer = object()
    clone = {
        "sequence": 3,
        "phase": "clone_and_component_construction",
        "call": LIFECYCLE.CLONE_LIFECYCLE_LABELS["replication_dispatch"],
        "receiver_id": None,
    }
    reset = {
        "sequence": 4,
        "phase": "post_clone_reset",
        "call": "lifecycle.example.Physics.reset",
        "receiver_id": hex(id(physics)),
        "duration_ms": 1.0,
    }
    initialize = {
        "sequence": 5,
        "phase": "post_clone_reset",
        "call": "lifecycle.example.Renderer.initialize",
        "receiver_id": hex(id(renderer)),
        "duration_ms": 1.0,
    }
    tracer = type("Tracer", (), {"lifecycle_events": [clone, reset, initialize]})()

    assert LIFECYCLE._initialization_snapshot(tracer, physics, renderer)["after_clone"]

    tracer.lifecycle_events.remove(reset)
    assert not LIFECYCLE._initialization_snapshot(tracer, physics, renderer)["after_clone"]


def test_visualizer_lifecycle_requires_init_step_running_and_close() -> None:
    class Visualizer:
        is_closed = False

        def is_running(self):
            return not self.is_closed

    visualizer = Visualizer()
    provider = object()
    plan = object()
    receiver_id = hex(id(visualizer))
    events = [
        {
            "sequence": 1,
            "phase": "clone_and_component_construction",
            "call": LIFECYCLE.CLONE_LIFECYCLE_LABELS["replication_dispatch"],
            "receiver_id": None,
        },
        {
            "sequence": 2,
            "phase": "post_clone_reset",
            "call": "lifecycle.Visualizer.initialize",
            "receiver_id": receiver_id,
            "duration_ms": 1.0,
            "argument_ids": [hex(id(provider)), hex(id(plan))],
        },
        {
            "sequence": 3,
            "phase": "movement_render_probe",
            "call": "lifecycle.Visualizer.step",
            "receiver_id": receiver_id,
            "duration_ms": 1.0,
        },
    ]
    tracer = type("Tracer", (), {"lifecycle_events": events})()

    assert LIFECYCLE._visualizer_lifecycle_snapshot([visualizer], tracer, provider, plan)["ready"]
    assert not LIFECYCLE._visualizer_lifecycle_snapshot([visualizer], tracer, provider, plan)["closed"]

    events[1]["argument_ids"].reverse()
    assert not LIFECYCLE._visualizer_lifecycle_snapshot([visualizer], tracer, provider, plan)["ready"]
    events[1]["argument_ids"].reverse()

    visualizer.is_closed = True
    events.append(
        {
            "sequence": 4,
            "phase": "simulation_close",
            "call": "lifecycle.Visualizer.close",
            "receiver_id": receiver_id,
            "duration_ms": 1.0,
        }
    )
    assert LIFECYCLE._visualizer_lifecycle_snapshot([visualizer], tracer, provider, plan)["closed"]


def test_visualizer_lifecycle_counts_nested_base_wrapper_as_one_logical_initialize() -> None:
    class BaseVisualizer:
        def initialize(self, provider, plan):
            self.inputs = (provider, plan)

    class Visualizer(BaseVisualizer):
        is_closed = False

        def initialize(self, provider, plan):
            super().initialize(provider, plan)

        def is_running(self):
            return True

    visualizer = Visualizer()
    provider = object()
    plan = object()
    tracer = LIFECYCLE._CallTracer(LIFECYCLE.Case("newton_mjwarp", "newton_renderer", "newton_gl"), 10)
    tracer.lifecycle_events.append(
        {
            "sequence": -1,
            "phase": "clone_and_component_construction",
            "call": LIFECYCLE.CLONE_LIFECYCLE_LABELS["replication_dispatch"],
            "receiver_id": None,
        }
    )
    tracer.patch_method(BaseVisualizer, "initialize", "lifecycle.BaseVisualizer.initialize")
    tracer.patch_method(Visualizer, "initialize", "lifecycle.Visualizer.initialize")
    tracer.phase = "post_clone_reset"

    visualizer.initialize(provider, plan)

    initialize_events = [event for event in tracer.lifecycle_events if event["call"].endswith(".initialize")]
    row = LIFECYCLE._visualizer_lifecycle_snapshot([visualizer], tracer, provider, plan)["visualizers"][0]
    assert [event["logical_call"] for event in initialize_events] == [True, False]
    assert visualizer.inputs == (provider, plan)
    assert row["initialize_calls"] == 1
    assert row["initialized_after_clone"] and row["exact_shared_inputs"]


@pytest.mark.parametrize(
    ("physics", "renderer", "visualizer", "ov_snapshot", "ovrtx_construct"),
    [
        ("newton_mjwarp", "newton_renderer", "none", 0, 0),
        ("newton_mjwarp", "ovrtx", "none", 1, 1),
        ("ovphysx", "newton_renderer", "none", 1, 0),
        ("ovphysx", "ovrtx", "none", 1, 1),
        ("newton_mjwarp", "newton_renderer", "newton_rtx", 0, 0),
        ("newton_mpm", "ovrtx", "newton_rtx", 1, 1),
    ],
)
def test_stage_lifecycle_gate_counts_each_snapshot_owner_once(
    physics: str, renderer: str, visualizer: str, ov_snapshot: int, ovrtx_construct: int
) -> None:
    class Tracer:
        counts = {
            "lifecycle.isaaclab.renderers.base_renderer.BaseRenderer.prepare_stage": {"calls": 1},
            LIFECYCLE.USD_SERIALIZATION_LABELS["layer_string"]: {"calls": ov_snapshot},
            LIFECYCLE.OVRTX_NATIVE_LABELS["construct"]: {"calls": ovrtx_construct},
        }

    case = LIFECYCLE.Case(physics, renderer, visualizer)
    assert LIFECYCLE._stage_lifecycle_snapshot(case, Tracer())["exactly_once_when_required"]

    Tracer.counts["lifecycle.isaaclab.renderers.base_renderer.BaseRenderer.prepare_stage"]["calls"] = 2
    assert not LIFECYCLE._stage_lifecycle_snapshot(case, Tracer())["exactly_once_when_required"]


@pytest.mark.parametrize(
    ("label", "calls"),
    [
        (LIFECYCLE.USD_SERIALIZATION_LABELS["layer_file"], 1),
        (LIFECYCLE.OVRTX_NATIVE_LABELS["construct"], 2),
        (LIFECYCLE.OVRTX_NATIVE_LABELS["open"], 1),
    ],
)
def test_stage_lifecycle_rejects_a_private_newton_rtx_stage_or_runtime(label: str, calls: int) -> None:
    class Tracer:
        counts = {
            "lifecycle.isaaclab.renderers.base_renderer.BaseRenderer.prepare_stage": {"calls": 1},
            LIFECYCLE.USD_SERIALIZATION_LABELS["layer_string"]: {"calls": 1},
            LIFECYCLE.OVRTX_NATIVE_LABELS["construct"]: {"calls": 1},
            label: {"calls": calls},
        }

    case = LIFECYCLE.Case("newton_mpm", "ovrtx", "newton_rtx")

    assert not LIFECYCLE._stage_lifecycle_snapshot(case, Tracer())["exactly_once_when_required"]


def test_newton_rtx_probe_renders_twice_only_for_that_visualizer() -> None:
    class NewtonRTXVisualizer:
        def __init__(self):
            self.calls = 0

        def render_rgb_array(self):
            self.calls += 1
            return np.zeros((2, 3, 3), dtype=np.uint8)

    visualizer = NewtonRTXVisualizer()
    snapshot = LIFECYCLE._newton_rtx_probe(LIFECYCLE.Case("newton_vbd", "newton_renderer", "newton_rtx"), [visualizer])

    assert visualizer.calls == 2
    assert snapshot["calls"] == snapshot["expected_calls"] == 2
    assert snapshot["frames"] == [
        {"shape": [2, 3, 3], "dtype": "uint8"},
        {"shape": [2, 3, 3], "dtype": "uint8"},
    ]
    assert snapshot["passed"]


def test_image_delta_detects_segmentation_motion() -> None:
    before = np.array([[[[1], [1]], [[2], [0]]]], dtype=np.int32)
    stationary = before.copy()
    moved = np.array([[[[1], [2]], [[0], [0]]]], dtype=np.int32)

    baseline_delta = LIFECYCLE._image_delta(before, stationary)
    moved_delta = LIFECYCLE._image_delta(stationary, moved)

    assert baseline_delta["changed_pixels"] == 0
    assert moved_delta["changed_pixels"] == 2
    assert moved_delta["changed_pixels"] > baseline_delta["changed_pixels"]


def test_worker_report_parser_checks_case_identity(tmp_path: Path) -> None:
    case = LIFECYCLE.Case("ovphysx", "ovrtx", "rerun")
    path = tmp_path / "worker.json"
    expected = {
        "schema_version": LIFECYCLE.SCHEMA_VERSION,
        "case": case.as_dict(),
        "status": "failed",
        "error": {"type": "RuntimeError", "message": "frozen"},
    }
    path.write_text(json.dumps(expected), encoding="utf-8")

    assert LIFECYCLE.load_worker_report(path, case) == expected
    with pytest.raises(ValueError, match="belongs to"):
        LIFECYCLE.load_worker_report(path, LIFECYCLE.Case("newton_mjwarp", "ovrtx", "none"))


def test_timeout_report_preserves_the_latest_atomic_checkpoint(tmp_path: Path) -> None:
    case = LIFECYCLE.Case("newton_mpm", "ovrtx", "newton_rtx")
    path = tmp_path / "worker.json"
    partial = {
        "schema_version": LIFECYCLE.SCHEMA_VERSION,
        "case": case.as_dict(),
        "status": "failed",
        "checkpoints": [{"name": "before_newton_rtx_frame_1", "trace_counts": {"clone": {"calls": 1}}}],
    }
    path.write_text(json.dumps(partial), encoding="utf-8")

    report = LIFECYCLE._timeout_process_report(
        case,
        ["python", str(SCRIPT)],
        path,
        timeout=12.5,
        stdout=b"started",
        stderr=b"still rendering",
        points=True,
    )

    assert report["checkpoints"] == partial["checkpoints"]
    assert report["error"] == {"type": "WorkerProcessError", "message": "Worker exceeded 12.5 seconds"}
    assert report["process"]["stdout_tail"] == "started"
