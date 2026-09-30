# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit backend lifecycle composition and communication in one minimal direct workflow.

The parent process enumerates the backend matrix and launches every runnable case in a fresh
subprocess. Each worker resolves one flat data-only config, installs process-local call wrappers,
runs one clone plan, and writes a compact JSON report. No scene manager or production tracing hooks
are required.

Examples::

    uv run --extra ovrtx python scripts/benchmarks/benchmark_backend_lifecycle.py --list
    uv run --extra ovrtx python scripts/benchmarks/benchmark_backend_lifecycle.py \
        --case newton_mjwarp__ovrtx__none --output backend_lifecycle.json
    uv run python scripts/benchmarks/benchmark_backend_lifecycle.py \
        --points --output backend_point_lifecycle.json
"""

from __future__ import annotations

import argparse
import functools
import importlib
import itertools
import json
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import MPMSolverCfg, NewtonSolverCfg, VBDSolverCfg
from isaaclab_newton.sim import NewtonDeformableBodyMaterialCfg, NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.mpm import MPMPointsCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim import PhysxDeformableBodyMaterialCfg, PhysxDeformableBodyPropertiesCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg
from isaaclab_visualizers.rerun import RerunVisualizerCfg
from isaaclab_visualizers.viser import ViserVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.cloner import CloneCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene_data import SceneDataFormat
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpolePhysicsCfg
from isaaclab_tasks.utils import PresetCfg, resolve_config
from isaaclab_tasks.utils.presets import (
    MultiBackendRendererCfg,
    MultiBackendSimulationCfg,
    MultiBackendVisualizerCfg,
)

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

TASK = "Isaac-Cartpole-Camera-Direct"
POINT_TASK = "Deformable-Point-Lifecycle-Direct"
PHYSICS = ("newton_mjwarp", "newton_kamino", "newton_vbd", "ovphysx", "isaacsim_physx")
RENDERERS = ("newton_renderer", "ovrtx", "isaacsim_rtx")
VISUALIZERS = ("none", "newton_gl", "newton_rtx", "rerun", "viser", "kit")
SCHEMA_VERSION = 5
NUM_ENVS = 4
READINESS_FRAMES = 3
MOVEMENT_OFFSET_M = 0.75
TIMELINE_LIMIT = 2048
ROOT = Path(__file__).resolve().parents[2]
TRACE_MANIFEST = Path(__file__).with_name("nsys_trace.json")
CORE_TRACE_DOMAINS = set("IsaacLab-Sim IsaacLab-SDP IsaacLab-Sensors IsaacLab-Assets".split())  # noqa: SIM905
RENDERER_TRACE_DOMAINS = {
    "newton_renderer": {"NewtonWarpRenderer"},
    "ovrtx": {"OVRTXRenderer", "OVRTXScene"},
    "isaacsim_rtx": set(),
}
LIFECYCLE_METHODS = set(
    "__enter__ __exit__ __init__ _bind_context _convert_points _convert_transforms cleanup close "  # noqa: SIM905
    "_initialize_impl _snapshot_stage finalize_visualization_model forward "
    "get_or_create_backend initialize initialize_solver prepare_stage publish render replicate reset step "
    "render_rgb_array start_simulation update update_visualization_state write_data_to_sim".split()
)
PHYSICS_MODULES = {
    "newton_mjwarp": {
        "isaaclab_newton.assets.articulation.articulation",
        "isaaclab_newton.physics.newton_manager",
        "isaaclab_newton.physics.mjwarp_manager",
    },
    "newton_kamino": {
        "isaaclab_newton.assets.articulation.articulation",
        "isaaclab_newton.physics.newton_manager",
        "isaaclab_newton.physics.kamino_manager",
    },
    "newton_vbd": {
        "isaaclab_newton.assets.articulation.articulation",
        "isaaclab_newton.physics.newton_manager",
        "isaaclab_newton.physics.vbd_manager",
    },
    "newton_mpm": {
        "isaaclab_newton.assets.mpm_object.mpm_object",
        "isaaclab_newton.physics.mpm_manager",
        "isaaclab_newton.physics.newton_manager",
    },
    "ovphysx": {"isaaclab_ov.assets.articulation.articulation", "isaaclab_ov.physics.ovphysx_manager"},
    "isaacsim_physx": {
        "isaaclab_physx.assets.articulation.articulation",
        "isaaclab_physx.physics.physx_manager",
    },
}
POINT_ASSET_MODULES = {
    "newton_vbd": "isaaclab_newton.assets.deformable_object.deformable_object",
    "newton_mpm": "isaaclab_newton.assets.mpm_object.mpm_object",
    "ovphysx": "isaaclab_ov.assets.deformable_object.deformable_object",
    "isaacsim_physx": "isaaclab_physx.assets.deformable_object.deformable_object",
}
RENDERER_MODULES = {
    "newton_renderer": "isaaclab_newton.renderers.newton_warp_renderer",
    "ovrtx": "isaaclab_ov.renderers.ovrtx_renderer",
    "isaacsim_rtx": "isaaclab_physx.renderers.isaac_rtx_renderer",
}
VISUALIZER_MODULES = {
    "newton_gl": "isaaclab_visualizers.newton.newton_visualizer",
    "newton_rtx": "isaaclab_visualizers.newton.newton_visualizer",
    "rerun": "isaaclab_visualizers.rerun.rerun_visualizer",
    "viser": "isaaclab_visualizers.viser.viser_visualizer",
    "kit": "isaaclab_visualizers.kit.kit_visualizer",
}
VISUALIZER_INPUTS = {
    "none": (),
    "newton_gl": ("sdp",),
    "newton_rtx": ("camera",),
    "rerun": ("sdp",),
    "viser": ("sdp",),
    "kit": ("sdp",),
}
CLONE_LIFECYCLE_LABELS = {
    "session_init": "lifecycle.isaaclab.cloner.replicate_session.ReplicateSession.__init__",
    "session_enter": "lifecycle.isaaclab.cloner.replicate_session.ReplicateSession.__enter__",
    "session_exit": "lifecycle.isaaclab.cloner.replicate_session.ReplicateSession.__exit__",
    "make_clone_plan": "lifecycle.isaaclab.cloner.replicate_session.make_clone_plan",
    "replication_dispatch": "lifecycle.isaaclab.cloner.replicate_session.replicate",
}
USD_SERIALIZATION_LABELS = {
    "layer_file": "lifecycle.pxr.Sdf.Layer.Export",
    "layer_string": "lifecycle.pxr.Sdf.Layer.ExportToString",
    "stage_string": "lifecycle.pxr.Usd.Stage.ExportToString",
}
OVRTX_NATIVE_LABELS = {
    "construct": "lifecycle.ovrtx.Renderer.__init__",
    "open": "lifecycle.ovrtx.Renderer.open_usd",
}
OVRTX_XFORM_WRITE_LABEL = "OVRTXScene.OvrtxScene.write_xforms"
CHECKPOINT_BOUNDARIES = {
    USD_SERIALIZATION_LABELS["layer_file"]: "usd_layer_export",
    **{label: f"ovrtx_renderer_{name}" for name, label in OVRTX_NATIVE_LABELS.items()},
}
LIFECYCLE_VISUALIZER_CFGS = MultiBackendVisualizerCfg(
    newton_gl=NewtonGLVisualizerCfg(
        headless=True,
        enable_picking=False,
        window_width=320,
        window_height=240,
        streaming_view=False,
    ),
    newton_rtx=NewtonRTXVisualizerCfg(
        headless=True,
        window_width=320,
        window_height=240,
        streaming_view=True,
        streaming_camera="{ENV_REGEX_NS}/Camera",
        streaming_gt_types=("rgb",),
    ),
    rerun=RerunVisualizerCfg(open_browser=False),
    viser=ViserVisualizerCfg(open_browser=False, verbose=False),
    kit=KitVisualizerCfg(headless=True, create_viewport=False),
)
LIFECYCLE_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Camera",
    offset=CameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
    data_types=["rgb", "instance_segmentation"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 20.0),
    ),
    width=96,
    height=96,
    renderer_cfg=MultiBackendRendererCfg(),
)


@configclass
class LifecycleDirectCfg(DirectRLEnvCfg):
    """Flat, data-only configuration for the lifecycle audit."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physics=CartpolePhysicsCfg(),
        visualizer_cfgs=LIFECYCLE_VISUALIZER_CFGS,
    )
    decimation: int = 2
    episode_length_s: float = 5.0
    action_space: int = 1
    observation_space: list[int] = [3, 96, 96]
    state_space: int = 4
    scene: object | None = None
    num_envs: int = NUM_ENVS
    env_spacing: float = 20.0
    clone_cfg: CloneCfg = CloneCfg()
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = CARTPOLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=CARTPOLE_CFG.spawn.replace(semantic_tags=[("class", "cartpole")]),
    )
    camera: CameraCfg = LIFECYCLE_CAMERA_CFG
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(-0.14644663035869598, -0.3535534143447876, -0.3535534143447876, 0.8535533547401428)
        ),
    )
    marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/LifecycleMarker",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        },
    )


@configclass
class PointPhysicsCfg(PresetCfg):
    """Physics implementations represented by the point-publication audit."""

    newton_vbd: NewtonSolverCfg = VBDSolverCfg(iterations=3, num_substeps=2)
    newton_mpm: NewtonSolverCfg = MPMSolverCfg(max_iterations=2, voxel_size=0.05, use_cuda_graph=False)
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    isaacsim_physx: PhysxCfg = PhysxCfg()
    default = newton_vbd


@configclass
class PointDeformableCfg(PresetCfg):
    """Equivalent backend-native deformable declarations."""

    newton_vbd: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        visualizer_cfg=None,
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            edge_refinement=2.0,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            physics_material=NewtonDeformableBodyMaterialCfg(density=500.0, k_mu=1.0e4, k_lambda=1.0e4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
        ),
    )
    newton_mpm: MPMObjectCfg = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=MPMObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        visualizer_cfg=None,
        spawn=MPMPointsCfg(
            positions=((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0), (0.0, 0.0, 0.05)),
            mass=0.01,
            radius=0.02,
        ),
    )
    physx: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        visualizer_cfg=None,
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            edge_refinement=2.0,
            deformable_props=PhysxDeformableBodyPropertiesCfg(),
            physics_material=PhysxDeformableBodyMaterialCfg(
                density=500.0,
                youngs_modulus=2.6e4,
                poissons_ratio=0.3,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
        ),
    )
    ovphysx = physx
    isaacsim_physx = physx


@configclass
class PointLifecycleDirectCfg(DirectRLEnvCfg):
    """Minimal data-only deformable and camera composition for live point auditing."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        dt=1.0 / 120.0,
        render_interval=1,
        physics=PointPhysicsCfg(),
        visualizer_cfgs=LIFECYCLE_VISUALIZER_CFGS,
    )
    decimation: int = 1
    episode_length_s: float = 1.0
    action_space: int = 0
    observation_space: int = 0
    state_space: int = 0
    scene: object | None = None
    num_envs: int = NUM_ENVS
    env_spacing: float = 2.0
    clone_cfg: CloneCfg = CloneCfg()
    deformable: PointDeformableCfg = PointDeformableCfg()
    camera: CameraCfg = LIFECYCLE_CAMERA_CFG


@dataclass(frozen=True, order=True)
class Case:
    """One physics, renderer, and visualizer selection."""

    physics: str
    renderer: str
    visualizer: str

    @property
    def name(self) -> str:
        """Stable command-line and report identifier."""
        return f"{self.physics}__{self.renderer}__{self.visualizer}"

    def as_dict(self) -> dict[str, str]:
        """Return the JSON representation."""
        return {"physics": self.physics, "renderer": self.renderer, "visualizer": self.visualizer}


POINT_CASES = (
    Case("newton_vbd", "newton_renderer", "newton_gl"),
    Case("newton_mpm", "newton_renderer", "newton_gl"),
    Case("ovphysx", "newton_renderer", "rerun"),
    Case("isaacsim_physx", "newton_renderer", "viser"),
    Case("newton_vbd", "ovrtx", "newton_rtx"),
    Case("newton_mpm", "ovrtx", "newton_rtx"),
    Case("newton_vbd", "isaacsim_rtx", "kit"),
    Case("newton_mpm", "isaacsim_rtx", "kit"),
    Case("ovphysx", "ovrtx", "viser"),
    Case("isaacsim_physx", "isaacsim_rtx", "kit"),
)


def enumerate_cases() -> tuple[Case, ...]:
    """Return the complete Cartesian product in stable order."""
    return tuple(Case(*selection) for selection in itertools.product(PHYSICS, RENDERERS, VISUALIZERS))


def exclusion_reason(case: Case) -> str | None:
    """Return the sole matrix exclusion: mixing OV and Kit/IsaacSim components."""
    has_ov = case.physics == "ovphysx" or case.renderer == "ovrtx"
    has_kit = case.physics == "isaacsim_physx" or case.renderer == "isaacsim_rtx" or case.visualizer == "kit"
    if has_ov and has_kit:
        return "OV components cannot share a process with Kit/IsaacSim components"
    return None


def matrix_manifest() -> dict[str, Any]:
    """Describe the complete, runnable, and deliberately excluded matrices."""
    cases = enumerate_cases()
    excluded = [(case, reason) for case in cases if (reason := exclusion_reason(case)) is not None]
    return {
        "physics": list(PHYSICS),
        "renderers": list(RENDERERS),
        "visualizers": list(VISUALIZERS),
        "total_cases": len(cases),
        "runnable_cases": len(cases) - len(excluded),
        "excluded_cases": [dict(case=case.as_dict(), name=case.name, reason=reason) for case, reason in excluded],
    }


def point_manifest() -> dict[str, Any]:
    """Describe the representative point publishers and renderer sinks."""
    return {
        "scenario": "deformable_points",
        "total_cases": len(POINT_CASES),
        "runnable_cases": len(POINT_CASES),
        "excluded_cases": [],
        "cases": [dict(case=case.as_dict(), name=case.name) for case in POINT_CASES],
    }


def load_worker_report(path: Path, case: Case) -> dict[str, Any]:
    """Load and minimally validate one worker report."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read worker report {path}: {exc}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Worker report {path} has an unsupported schema")
    if report.get("case") != case.as_dict():
        raise ValueError(f"Worker report {path} belongs to {report.get('case')}, expected {case.as_dict()}")
    if report.get("status") not in {"passed", "failed"}:
        raise ValueError(f"Worker report {path} has invalid status {report.get('status')!r}")
    return report


def _case_from_name(name: str) -> Case:
    cases = {case.name: case for case in (*enumerate_cases(), *POINT_CASES)}
    try:
        return cases[name]
    except KeyError as exc:
        raise ValueError(f"Unknown case {name!r}; run with --list to inspect valid names") from exc


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tail(output: str | bytes | None, limit: int = 4000) -> str:
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    return (output or "")[-limit:]


def _manifest_trace_targets(
    case: Case, manifest: list[dict[str, Any]] | None = None, *, points: bool = False
) -> dict[tuple[str, str], str]:
    """Return selected ``(module, qualname)`` targets without importing optional runtimes."""
    if manifest is None:
        manifest = json.loads(TRACE_MANIFEST.read_text(encoding="utf-8"))
    domains = CORE_TRACE_DOMAINS | RENDERER_TRACE_DOMAINS[case.renderer]
    modules = _lifecycle_modules(case, points=points)
    targets = {}
    for group in manifest:
        if group.get("domain") not in domains:
            continue
        for entry in group.get("functions", ()):
            entry = {"function": entry} if isinstance(entry, str) else entry
            module = entry.get("module", group.get("module"))
            function = entry.get("function")
            if module in modules and function:
                targets[(module, function)] = f"{group['domain']}.{function}"
    return targets


def _lifecycle_modules(case: Case, *, points: bool = False) -> set[str]:
    modules = {
        "isaaclab.cloner.clone_plan",
        "isaaclab.cloner.replicate_session",
        "isaaclab.cloner.usd",
        "isaaclab.assets.articulation.articulation",
        "isaaclab.markers.visualization_markers",
        "isaaclab.renderers.base_renderer",
        "isaaclab.scene_data.scene_data_provider",
        "isaaclab.sensors.camera.camera",
        "isaaclab.sensors.sensor_base",
        "isaaclab.sim.simulation_context",
        RENDERER_MODULES[case.renderer],
        *PHYSICS_MODULES[case.physics],
    }
    if case.visualizer != "none":
        modules.add(VISUALIZER_MODULES[case.visualizer])
    if points:
        modules.add(POINT_ASSET_MODULES[case.physics])
    if (
        case.physics.startswith("newton_")
        or case.renderer == "newton_renderer"
        or case.visualizer
        in {
            "newton_gl",
            "rerun",
            "viser",
        }
    ):
        modules.add("isaaclab_newton.cloner.replicate")
    modules.update(
        module
        for enabled, module in (
            (case.physics == "ovphysx" or case.renderer == "ovrtx", "isaaclab_ov.cloner.replicate"),
            (case.physics == "isaacsim_physx", "isaaclab_physx.cloner.replicate"),
            (case.renderer == "ovrtx", "isaaclab_ov.renderers.ovrtx_scene"),
        )
        if enabled
    )
    return modules


class _CallTracer:
    """Process-local wrappers for selected Python lifecycle calls."""

    def __init__(self, case: Case, limit: int, *, points: bool = False):
        self.limit = limit
        self.phase = "trace_install"
        self.started_ns = time.perf_counter_ns()
        self.targets = _manifest_trace_targets(case, points=points)
        self.lifecycle_modules = _lifecycle_modules(case, points=points)
        self.counts: dict[str, dict[str, float | int]] = {}
        self.timeline: list[dict[str, Any]] = []
        self.construction_calls: list[dict[str, Any]] = []
        self.backend_requests: list[dict[str, Any]] = []
        self.lifecycle_events: list[dict[str, Any]] = []
        self.scene_data_requests: list[dict[str, Any]] = []
        self.scene_data_conversions: list[dict[str, Any]] = []
        self.geometry_launches: list[dict[str, Any]] = []
        self.total_events = 0
        self._patched: set[tuple[int, str]] = set()
        self._consumer_stack: list[str] = []
        self._lifecycle_stack: list[tuple[str | None, str]] = []
        self._point_conversion_depth = 0
        self.checkpoint_callback: Callable[[str], None] | None = None

    def _invoke(self, label: str, receiver: Any, function, *args, **kwargs):
        started_ns = time.perf_counter_ns()
        event = {
            "sequence": self.total_events,
            "at_ms": round((started_ns - self.started_ns) / 1_000_000, 3),
            "phase": self.phase,
            "call": label,
            "receiver_id": None if receiver is None else hex(id(receiver)),
        }
        if label.endswith(".__init__"):
            arguments = args[1:] if receiver is not None else args
            event["argument_ids"] = [hex(id(argument)) for argument in arguments]
            self.construction_calls.append(event)
        method = label.rsplit(".", 1)[-1]
        if method == "initialize":
            arguments = args[1:] if receiver is not None else args
            event["argument_ids"] = [hex(id(argument)) for argument in arguments]
        lifecycle_call = (
            label in CLONE_LIFECYCLE_LABELS.values()
            or "initialize" in method
            or method
            in {
                "close",
                "finalize_visualization_model",
                "reset",
                "start_simulation",
                "step",
            }
        )
        lifecycle_key = (event["receiver_id"], method)
        if lifecycle_call:
            event["logical_call"] = lifecycle_key not in self._lifecycle_stack
            self.lifecycle_events.append(event)
        if label.endswith("SceneDataProvider._convert_transforms"):
            arguments = args[1:] if receiver is not None else args
            input, output, _count, *rest = arguments
            stream = rest[0] if rest else kwargs.get("name")
            conversion = {
                "kind": "transforms",
                "stream": stream,
                "generation": receiver._generations[("transforms", stream)],
                "input_format": getattr(input, "_cls", type(input)).__name__,
                "output_format": getattr(output, "_cls", type(output)).__name__,
            }
            event["conversion"] = conversion
            self.scene_data_conversions.append(conversion)
        elif label.endswith("SceneDataProvider._convert_points"):
            arguments = args[1:] if receiver is not None else args
            input, output, _count = arguments[:3]
            stream = next(
                (
                    name
                    for name, publication in receiver._backend.point_publications.items()
                    if publication.data is input
                ),
                None,
            )
            conversion = {
                "kind": "points",
                "stream": stream,
                "generation": receiver._generations.get(("points", stream)),
                "input_format": getattr(input, "_cls", type(input)).__name__,
                "output_format": getattr(output, "_cls", type(output)).__name__,
            }
            event["conversion"] = conversion
            self.scene_data_conversions.append(conversion)
        self.total_events += 1
        if len(self.timeline) < self.limit:
            self.timeline.append(event)
        stats = self.counts.setdefault(label, {"calls": 0, "errors": 0, "total_ms": 0.0})
        stats["calls"] += 1
        checkpoint_name = CHECKPOINT_BOUNDARIES.get(label)
        checkpoint_call = int(stats["calls"])
        if checkpoint_name is not None and self.checkpoint_callback is not None:
            self.checkpoint_callback(f"before_{checkpoint_name}_{checkpoint_call}")
        is_consumer_call = receiver is not None and (
            method in {"initialize", "render", "render_rgb_array", "step"}
            or method == "update"
            and "Renderer." in label
        )
        if is_consumer_call:
            self._consumer_stack.append(hex(id(receiver)))
        point_conversion = label.endswith("SceneDataProvider._convert_points")
        if point_conversion:
            self._point_conversion_depth += 1
        if lifecycle_call:
            self._lifecycle_stack.append(lifecycle_key)
        try:
            result = function(*args, **kwargs)
            if label.endswith(".SimulationContext.get_or_create_backend"):
                arguments = args[1:] if receiver is not None else args
                backend_type = arguments[0]
                request = {
                    "phase": self.phase,
                    "key": _describe(backend_type)["type"],
                    "backend": _describe(result),
                }
                event["backend_request"] = request
                self.backend_requests.append(request)
            if label.endswith("SceneDataProvider.request_transforms"):
                arguments = args[1:] if receiver is not None else args
                output_format = arguments[0] if arguments else kwargs["output_format"]
                stream = arguments[1] if len(arguments) > 1 else kwargs.get("name")
                request = {
                    "kind": "transforms",
                    "phase": self.phase,
                    "provider_id": hex(id(receiver)),
                    "consumer_id": self._consumer_stack[-1] if self._consumer_stack else None,
                    "stream": stream,
                    "format": output_format.__name__,
                    "output": _describe(result),
                }
                event["scene_data_request"] = request
                self.scene_data_requests.append(request)
            elif label.endswith("SceneDataProvider.request_points"):
                arguments = args[1:] if receiver is not None else args
                output_format = arguments[0] if arguments else kwargs["output_format"]
                stream = arguments[1] if len(arguments) > 1 else kwargs.get("name", "points")
                request = {
                    "kind": "points",
                    "phase": self.phase,
                    "provider_id": hex(id(receiver)),
                    "consumer_id": self._consumer_stack[-1] if self._consumer_stack else None,
                    "stream": stream,
                    "format": output_format.__name__,
                    "output": _describe(result),
                }
                event["scene_data_request"] = request
                self.scene_data_requests.append(request)
            return result
        except BaseException:
            stats["errors"] += 1
            raise
        finally:
            if point_conversion:
                self._point_conversion_depth -= 1
            if is_consumer_call:
                self._consumer_stack.pop()
            if lifecycle_call:
                self._lifecycle_stack.pop()
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            stats["total_ms"] = round(float(stats["total_ms"]) + elapsed_ms, 3)
            event["duration_ms"] = round(elapsed_ms, 3)
            if checkpoint_name is not None and self.checkpoint_callback is not None:
                self.checkpoint_callback(f"after_{checkpoint_name}_{checkpoint_call}")

    def patch_method(self, cls: type, name: str, label: str) -> None:
        descriptor = cls.__dict__.get(name)
        key = (id(cls), name)
        if descriptor is None or key in self._patched:
            return
        self._patched.add(key)
        if isinstance(descriptor, (classmethod, staticmethod)):
            original = descriptor.__func__
            receiver = isinstance(descriptor, classmethod)

            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                owner = args[0] if receiver else None
                return self._invoke(label, owner, original, *args, **kwargs)

            setattr(cls, name, type(descriptor)(wrapped))
        elif callable(descriptor):

            @functools.wraps(descriptor)
            def wrapped(receiver, *args, **kwargs):
                return self._invoke(label, receiver, descriptor, receiver, *args, **kwargs)

            setattr(cls, name, wrapped)

    def patch_target(self, module_name: str, qualname: str, label: str) -> None:
        owner = importlib.import_module(module_name)
        *parents, name = qualname.split(".")
        for part in parents:
            owner = getattr(owner, part)
        if isinstance(owner, type):
            self.patch_method(owner, name, label)
        elif callable(function := getattr(owner, name, None)):
            key = (id(owner), name)
            if key in self._patched:
                return
            self._patched.add(key)

            @functools.wraps(function)
            def wrapped(*args, **kwargs):
                return self._invoke(label, None, function, *args, **kwargs)

            setattr(owner, name, wrapped)

    def record_geometry_launch(self, kernel: Any) -> None:
        """Record core point kernels and whether the SDP conversion owns the launch."""
        function = getattr(kernel, "func", None)
        if getattr(function, "__module__", None) == "isaaclab.scene_data.geometry_points":
            self.geometry_launches.append(
                {
                    "kernel": getattr(kernel, "key", type(kernel).__name__),
                    "phase": self.phase,
                    "inside_sdp": self._point_conversion_depth > 0,
                }
            )

    def patch_warp_launch(self) -> None:
        """Trace point-geometry kernels without wrapping Warp kernel objects."""
        import warp as wp  # noqa: PLC0415

        launch = wp.launch

        @functools.wraps(launch)
        def wrapped(kernel, *args, **kwargs):
            self.record_geometry_launch(kernel)
            return launch(kernel, *args, **kwargs)

        wp.launch = wrapped

    def report(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "timeline": self.timeline,
            "backend_requests": self.backend_requests,
            "lifecycle_events": self.lifecycle_events,
            "events": self.total_events,
            "manifest_targets": len(self.targets),
            "scene_data_requests": self.scene_data_requests,
            "scene_data_conversions": self.scene_data_conversions,
            "geometry_launches": self.geometry_launches,
            "timeline_truncated": self.total_events > len(self.timeline),
        }


@contextmanager
def _phase(report: dict[str, Any], name: str, tracer: _CallTracer | None = None):
    previous_phase = tracer.phase if tracer is not None else None
    if tracer is not None:
        tracer.phase = name
    started_ns = time.perf_counter_ns()
    try:
        yield
    except BaseException:
        report["_failed_phase"] = name
        raise
    finally:
        elapsed_ms = round((time.perf_counter_ns() - started_ns) / 1_000_000, 3)
        report["phase_timings_ms"][name] = elapsed_ms
        print(f"phase {name}: {elapsed_ms:g} ms", flush=True)
        if tracer is not None:
            tracer.phase = previous_phase


def _install_trace(tracer: _CallTracer) -> None:
    """Wrap manifest targets plus common lifecycle methods in selected backend modules."""
    from pxr import Sdf, Usd  # noqa: PLC0415

    physics_manager = _import_type("isaaclab.physics.physics_manager", "PhysicsManager")
    tracer.patch_method(
        physics_manager, "__init__", "lifecycle.isaaclab.physics.physics_manager.PhysicsManager.__init__"
    )
    for cls, method, label in (
        (Sdf.Layer, "Export", USD_SERIALIZATION_LABELS["layer_file"]),
        (Sdf.Layer, "ExportToString", USD_SERIALIZATION_LABELS["layer_string"]),
        (Usd.Stage, "ExportToString", USD_SERIALIZATION_LABELS["stage_string"]),
    ):
        tracer.patch_method(cls, method, label)
    if "isaaclab_ov.renderers.ovrtx_scene" in tracer.lifecycle_modules:
        ovrtx_renderer = _import_type("ovrtx", "Renderer")
        methods = (("__init__", OVRTX_NATIVE_LABELS["construct"]), ("open_usd", OVRTX_NATIVE_LABELS["open"]))
        for method, label in methods:
            tracer.patch_method(ovrtx_renderer, method, label)
    for (module, qualname), label in tracer.targets.items():
        tracer.patch_target(module, qualname, label)
    tracer.patch_warp_launch()
    suffixes = (
        "Articulation",
        "Backend",
        "Context",
        "Manager",
        "Markers",
        "Object",
        "Provider",
        "Renderer",
        "Session",
        "Visualizer",
    )
    for module_name in tracer.lifecycle_modules:
        module = importlib.import_module(module_name)
        for cls in vars(module).values():
            if isinstance(cls, type) and cls.__module__ == module_name and cls.__name__.endswith(suffixes):
                for method in LIFECYCLE_METHODS:
                    tracer.patch_method(cls, method, f"lifecycle.{module_name}.{cls.__qualname__}.{method}")
        for function in LIFECYCLE_METHODS & vars(module).keys():
            tracer.patch_target(module_name, function, f"lifecycle.{module_name}.{function}")


def _import_type(module_name: str, class_name: str) -> type:
    return getattr(importlib.import_module(module_name), class_name)


def _resolve_env_cfg(case: Case, device: str, num_envs: int = NUM_ENVS):
    overrides = [
        f"physics={case.physics}",
        f"renderer={case.renderer}",
        f"sim.device={device}",
        f"num_envs={num_envs}",
    ]
    if case.visualizer != "none":
        overrides.append(f"visualizer={case.visualizer}")
    env_cfg = resolve_config(LifecycleDirectCfg(seed=0), overrides)
    env_cfg.validate()
    return env_cfg


def _resolve_point_cfg(case: Case, device: str, num_envs: int = NUM_ENVS):
    if case not in POINT_CASES:
        raise ValueError(f"Unsupported point lifecycle case {case.name!r}")
    overrides = [
        f"physics={case.physics}",
        f"renderer={case.renderer}",
        f"visualizer={case.visualizer}",
        f"sim.device={device}",
        f"num_envs={num_envs}",
    ]
    env_cfg = resolve_config(
        PointLifecycleDirectCfg(seed=0),
        overrides,
    )
    env_cfg.validate()
    return env_cfg


def _launcher_scan_snapshot(case: Case, env_cfg: Any, launcher_args: dict[str, Any]) -> dict[str, Any]:
    from isaaclab.app.sim_launcher import _get_kit_runtime_sources, scan  # noqa: PLC0415

    result = scan(env_cfg, launcher_args)
    kit_sources = _get_kit_runtime_sources(result, launcher_args)
    expected_kit = case.physics == "isaacsim_physx" or case.renderer == "isaacsim_rtx" or case.visualizer == "kit"
    return {"kit_sources": list(kit_sources), "matches_case": bool(kit_sources) == expected_kit}


def _describe(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    value_type = value if isinstance(value, type) else type(value)
    return {"type": f"{value_type.__module__}.{value_type.__qualname__}", "id": hex(id(value))}


def _buffer_snapshot(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    value_cls = getattr(value, "_cls", None)
    field_names = getattr(value_cls, "vars", {}) or getattr(type(value), "__dataclass_fields__", {})
    fields = {}
    for name in field_names:
        buffer = getattr(value, name, None)
        buffers = buffer if isinstance(buffer, tuple) else (buffer,)
        pointers = [hex(int(item.ptr)) for item in buffers if getattr(item, "ptr", None)]
        if not pointers:
            pointers = [hex(int(bucket.ptr)) for item in buffers for bucket in getattr(item, "buckets", ())]
        shapes = [list(item.shape) for item in buffers if hasattr(item, "shape")]
        devices = {str(item.device) for item in buffers if hasattr(item, "device")}
        fields[name] = {
            "pointers": pointers,
            "shape": shapes[0] if len(shapes) == 1 else shapes or None,
            "device": devices.pop() if len(devices) == 1 else sorted(devices) or None,
        }
    return {"format": type(value).__name__ if value_cls is None else value_cls.__name__, "fields": fields}


def _publication_count(value: Any) -> int:
    """Derive one publication count from its pointer fields."""
    value_cls = getattr(value, "_cls", None)
    field_names = getattr(value_cls, "vars", {}) or getattr(type(value), "__dataclass_fields__", {})
    counts = {
        int(buffer.shape[0])
        for name in field_names
        if (buffer := getattr(value, name, None)) is not None and hasattr(buffer, "shape")
    }
    if len(counts) > 1:
        raise RuntimeError("SDP publication fields expose different element counts.")
    return counts.pop() if counts else 0


def _publisher_aliases(output: Any, publisher: Any) -> tuple[bool, bool, bool]:
    output_snapshot = _buffer_snapshot(output)
    publisher_snapshot = _buffer_snapshot(publisher)
    if output_snapshot is None or publisher_snapshot is None:
        return False, False, False
    output_pointers = {name: field["pointers"] for name, field in output_snapshot["fields"].items()}
    publisher_pointers = {name: field["pointers"] for name, field in publisher_snapshot["fields"].items()}
    has_payload = any(publisher_pointers.values())
    payload_alias = has_payload and all(
        output_pointers.get(name) == pointers for name, pointers in publisher_pointers.items()
    )
    index_attachment = set(output_pointers) - set(publisher_pointers) == {"source_indices"} and bool(
        output_pointers["source_indices"]
    )
    return payload_alias and output_pointers == publisher_pointers, payload_alias, index_attachment


def _sdp_request_contract(source: Any, output: Any, output_format: type, payload_copy_count: int) -> dict[str, Any]:
    """Classify logical format conversion independently from payload movement."""
    source_format = getattr(source, "_cls", type(source))
    native = source_format is output_format
    exact_alias, payload_alias, index_attachment = _publisher_aliases(output, source)
    identity_index_attachment = (
        source_format is SceneDataFormat.Transform
        and output_format is SceneDataFormat.IndexedTransform
        and payload_alias
        and index_attachment
    )
    expected_payload_copy_count = int(not native and not identity_index_attachment)
    pointer_contract = exact_alias if native else payload_alias if identity_index_attachment else not payload_alias
    return {
        "source_format": source_format.__name__,
        "format": output_format.__name__,
        "output": _buffer_snapshot(output),
        "native_format": native,
        "logical_conversion_count": int(not native),
        "payload_copy_count": payload_copy_count,
        "expected_payload_copy_count": expected_payload_copy_count,
        "exact_payload_copy_count": payload_copy_count == expected_payload_copy_count,
        "exact_aliases_publisher": exact_alias,
        "payload_aliases_publisher": payload_alias,
        "identity_index_attachment": identity_index_attachment,
        "pointer_contract": pointer_contract,
    }


def _clone_lifecycle_snapshot(tracer: _CallTracer) -> dict[str, Any]:
    calls = {name: int(tracer.counts.get(label, {}).get("calls", 0)) for name, label in CLONE_LIFECYCLE_LABELS.items()}
    return {"calls": calls, "exactly_one": all(count == 1 for count in calls.values())}


def _stage_lifecycle_snapshot(case: Case, tracer: _CallTracer) -> dict[str, Any]:
    """Summarize the single clone handoff and reject private visualizer stages."""
    calls = {
        "renderer_prepare": sum(
            int(stats["calls"])
            for label, stats in tracer.counts.items()
            if ".renderers." in label and label.endswith(".prepare_stage")
        ),
        "ov_snapshot": int(tracer.counts.get(USD_SERIALIZATION_LABELS["layer_string"], {}).get("calls", 0)),
        "layer_file_export": int(tracer.counts.get(USD_SERIALIZATION_LABELS["layer_file"], {}).get("calls", 0)),
        "ovrtx_native_construct": int(tracer.counts.get(OVRTX_NATIVE_LABELS["construct"], {}).get("calls", 0)),
        "ovrtx_legacy_open": int(tracer.counts.get(OVRTX_NATIVE_LABELS["open"], {}).get("calls", 0)),
        "unowned_stage_string_export": int(
            tracer.counts.get(USD_SERIALIZATION_LABELS["stage_string"], {}).get("calls", 0)
        ),
    }
    expected = {
        "renderer_prepare": 1,
        "ov_snapshot": int(case.physics == "ovphysx" or case.renderer == "ovrtx"),
        "layer_file_export": 0,
        "ovrtx_native_construct": int(case.renderer == "ovrtx"),
        "ovrtx_legacy_open": 0,
        "unowned_stage_string_export": 0,
    }
    no_private_runtime = (
        calls["layer_file_export"] == 0
        and calls["ovrtx_legacy_open"] == 0
        and calls["ovrtx_native_construct"] == int(case.renderer == "ovrtx")
    )
    return {
        "calls": calls,
        "expected": expected,
        "newton_rtx_owns_no_stage_or_ovrtx": case.visualizer != "newton_rtx" or no_private_runtime,
        "exactly_once_when_required": calls == expected,
    }


def _registry_snapshot(case: Case, sim: Any, tracer: _CallTracer) -> dict[str, Any]:
    """Prove that every consumer independently resolved the same typed resource."""
    expected = {
        "NewtonReplicateContext": int(case.physics.startswith("newton_"))
        + int(case.renderer == "newton_renderer")
        + int(case.visualizer in {"newton_gl", "rerun", "viser"}),
        "PhysxReplicateContext": int(case.physics == "isaacsim_physx"),
        "UsdReplicateContext": int(case.physics == "isaacsim_physx")
        + int(case.renderer == "isaacsim_rtx")
        + int(case.visualizer == "kit"),
        "_IsaacRtxRuntime": int(case.renderer == "isaacsim_rtx"),
        "OvReplicateContext": int(case.physics == "ovphysx") + int(case.renderer == "ovrtx"),
    }
    expected = {name: count for name, count in expected.items() if count}
    actual: dict[str, list[dict[str, Any]]] = {}
    for request in tracer.backend_requests:
        label = request["key"].rsplit(".", 1)[-1]
        actual.setdefault(label, []).append(request)
    counts = {name: len(requests) for name, requests in actual.items()}
    stable_results = all(len({request["backend"]["id"] for request in requests}) == 1 for requests in actual.values())
    registered = {
        _describe(backend_type)["type"]: hex(id(backend)) for backend_type, backend in sim._backend_registry.items()
    }
    requests_match_registry = all(
        registered.get(request["key"]) == request["backend"]["id"]
        for requests in actual.values()
        for request in requests
    )
    before_initialization = all(
        request["phase"] in {"context_construction_and_backend_bind", "clone_and_component_construction"}
        for requests in actual.values()
        for request in requests
    )
    return {
        "requests": actual,
        "counts": counts,
        "expected_counts": expected,
        "one_resource_per_type": counts == expected
        and stable_results
        and requests_match_registry
        and before_initialization,
    }


def _initialization_snapshot(tracer: _CallTracer, physics_manager: Any, renderer: Any) -> dict[str, Any]:
    """Require physics and renderer initialization on their exact instances after cloning."""
    clone_events = [event for event in tracer.lifecycle_events if event["call"] in CLONE_LIFECYCLE_LABELS.values()]
    last_clone_sequence = max((event["sequence"] for event in clone_events), default=-1)
    components = {}
    for name, value, method in (("physics", physics_manager, "reset"), ("renderer", renderer, "initialize")):
        events = [
            event
            for event in tracer.lifecycle_events
            if event["receiver_id"] == hex(id(value))
            and event["call"].endswith(f".{method}")
            and event.get("logical_call", True)
        ]
        components[name] = {
            "object": _describe(value),
            "method": method,
            "calls": len(events),
            "duration_ms": max((event["duration_ms"] for event in events), default=None),
            "after_clone": bool(events)
            and all(
                event["sequence"] > last_clone_sequence and event["phase"] == "post_clone_reset" for event in events
            ),
        }
    return {"components": components, "after_clone": all(row["after_clone"] for row in components.values())}


def _visualizer_lifecycle_snapshot(
    visualizers: list[Any],
    tracer: _CallTracer,
    provider: Any,
    plan: Any,
    *,
    probe_phase: str = "movement_render_probe",
) -> dict[str, Any]:
    """Report exact shared-plan initialization, stepping, running, and teardown."""
    clone_sequences = [
        event["sequence"] for event in tracer.lifecycle_events if event["call"] in CLONE_LIFECYCLE_LABELS.values()
    ]
    last_clone_sequence = max(clone_sequences, default=-1)
    rows = []
    for visualizer in visualizers:
        receiver_id = hex(id(visualizer))
        events = {
            method: [
                event
                for event in tracer.lifecycle_events
                if event["receiver_id"] == receiver_id
                and event["call"].endswith(f".{method}")
                and event.get("logical_call", True)
            ]
            for method in ("initialize", "step", "close")
        }
        initialized_after_clone = bool(events["initialize"]) and all(
            event["sequence"] > last_clone_sequence and event["phase"] == "post_clone_reset"
            for event in events["initialize"]
        )
        exact_shared_inputs = bool(events["initialize"]) and all(
            event.get("argument_ids") == [hex(id(provider)), hex(id(plan))] for event in events["initialize"]
        )
        stepped = any(event["phase"] == probe_phase for event in events["step"])
        closed_in_teardown = bool(events["close"]) and all(
            event["phase"] == "simulation_close" for event in events["close"]
        )
        rows.append(
            {
                "object": _describe(visualizer),
                "initialize_calls": len(events["initialize"]),
                "initialize_ms": max((event["duration_ms"] for event in events["initialize"]), default=None),
                "step_calls": len(events["step"]),
                "step_ms": round(sum(event["duration_ms"] for event in events["step"]), 3),
                "close_calls": len(events["close"]),
                "close_ms": max((event["duration_ms"] for event in events["close"]), default=None),
                "initialized_after_clone": initialized_after_clone,
                "exact_shared_inputs": exact_shared_inputs,
                "stepped": stepped,
                "running": bool(visualizer.is_running()) and not visualizer.is_closed,
                "is_closed": visualizer.is_closed,
                "closed_in_teardown": closed_in_teardown,
            }
        )
    return {
        "visualizers": rows,
        "ready": all(
            row["initialized_after_clone"]
            and row["exact_shared_inputs"]
            and row["stepped"]
            and row["running"]
            and not row["is_closed"]
            for row in rows
        ),
        "closed": all(row["is_closed"] and row["closed_in_teardown"] for row in rows),
    }


def _newton_resource_snapshot(case: Case, sim: Any, physics_manager: Any, renderer: Any, visualizers: list[Any]):
    """Verify all Newton consumers hold the registry's one Model/State/Control owner."""
    resources = [
        backend
        for backend_type, backend in sim._backend_registry.items()
        if backend_type.__name__ == "NewtonReplicateContext"
    ]
    expected_consumers = (
        int(case.physics.startswith("newton_"))
        + int(case.renderer == "newton_renderer")
        + int(case.visualizer in {"newton_gl", "rerun", "viser"})
    )
    expected_visual_shapes = case.renderer == "newton_renderer" or case.visualizer in {
        "newton_gl",
        "rerun",
        "viser",
    }
    if not expected_consumers:
        return {
            "resource": None,
            "owners": {},
            "native_objects": {},
            "shared": True,
            "load_visual_shapes": None,
            "expected_visual_shapes": expected_visual_shapes,
            "visual_shapes_match": not expected_visual_shapes,
        }
    if len(resources) != 1:
        return {
            "resource": None,
            "owners": {},
            "native_objects": {},
            "shared": False,
            "load_visual_shapes": None,
            "expected_visual_shapes": expected_visual_shapes,
            "visual_shapes_match": False,
        }

    resource = resources[0]
    consumers = {}
    if case.physics.startswith("newton_"):
        consumers["physics"] = getattr(physics_manager, "_newton", None)
    if case.renderer == "newton_renderer":
        consumers["renderer"] = getattr(renderer, "_newton_backend", None)
    for index, visualizer in enumerate(visualizers):
        if hasattr(visualizer, "_newton_backend"):
            consumers[f"visualizer[{index}]"] = visualizer._newton_backend

    model = resource.get_model()
    state = resource.get_state_0()
    native_objects = {
        "model": _describe(model),
        "state": _describe(state),
        "state_1": _describe(resource.get_state_1()),
        "control": _describe(resource.get_control()),
    }
    model_refs = ([] if case.renderer != "newton_renderer" else [getattr(renderer, "_newton_model", None)]) + [
        getattr(visualizer, "_model", None) for visualizer in visualizers if hasattr(visualizer, "_newton_backend")
    ]
    state_refs = [] if case.renderer != "newton_renderer" else [getattr(resource, "_sensor_state", None)]
    visualizer_state_refs = [
        getattr(visualizer, "_state", None) for visualizer in visualizers if hasattr(visualizer, "_newton_backend")
    ]
    return {
        "resource": _describe(resource),
        "owners": {name: _describe(value) for name, value in consumers.items()},
        "native_objects": native_objects,
        "load_visual_shapes": resource.load_visual_shapes,
        "expected_visual_shapes": expected_visual_shapes,
        "visual_shapes_match": resource.load_visual_shapes == expected_visual_shapes,
        "shared": len(consumers) == expected_consumers
        and all(value is resource for value in consumers.values())
        and all(value is model for value in model_refs)
        and all(value is state for value in state_refs)
        and all(value is None or value is state for value in visualizer_state_refs)
        and (not case.physics.startswith("newton_") or resource.get_control() is not None),
    }


def _newton_finalization_snapshot(case: Case, tracer: _CallTracer) -> dict[str, Any]:
    """Require exactly one model finalization by the registry resource's owning path."""
    uses_newton = (
        case.physics.startswith("newton_")
        or case.renderer == "newton_renderer"
        or case.visualizer in {"newton_gl", "rerun", "viser"}
    )
    expected = {
        "physics": int(case.physics.startswith("newton_")),
        "visualization": int(uses_newton and not case.physics.startswith("newton_")),
    }
    events = [event for event in tracer.lifecycle_events if event.get("logical_call", True)]
    calls = {
        "physics": sum(event["call"].endswith(".start_simulation") for event in events),
        "visualization": sum(event["call"].endswith(".finalize_visualization_model") for event in events),
    }
    return {"calls": calls, "expected": expected, "exactly_once_when_required": calls == expected}


def _newton_hard_reset_snapshot(
    case: Case, sim: Any, physics_manager: Any, renderer: Any, visualizers: list[Any]
) -> dict[str, Any]:
    """Require a second Newton hard reset to preserve the plan-owned native objects."""
    if not case.physics.startswith("newton_"):
        return {"required": False, "before": {}, "after": {}, "stable": {}, "passed": True}

    before_snapshot = _newton_resource_snapshot(case, sim, physics_manager, renderer, visualizers)
    before = {"resource": before_snapshot["resource"], **before_snapshot["native_objects"]}
    sim.reset()
    after_snapshot = _newton_resource_snapshot(case, sim, physics_manager, renderer, visualizers)
    after = {"resource": after_snapshot["resource"], **after_snapshot["native_objects"]}
    stable = {name: before[name] == after[name] for name in before}
    return {
        "required": True,
        "before": before,
        "after": after,
        "stable": stable,
        "passed": before_snapshot["shared"]
        and after_snapshot["shared"]
        and all(before.values())
        and all(stable.values()),
    }


def _construction_snapshot(value: Any, cfg: Any, tracer: _CallTracer) -> dict[str, Any]:
    calls = [call for call in tracer.construction_calls if call["receiver_id"] == hex(id(value))]
    exact_calls = [call for call in calls if call["argument_ids"] == [hex(id(cfg))]]
    return {
        "object": _describe(value),
        "cfg": _describe(cfg),
        "construction_ms": max((call["duration_ms"] for call in exact_calls), default=None),
        "exact_cfg_construction": not isinstance(value, type) and bool(exact_calls),
    }


def _point_data_snapshot(provider: Any, tracer: _CallTracer, layout: Any) -> dict[str, Any]:
    """Report each live point publication and its current renderer request contract."""
    conversions = [row for row in tracer.scene_data_conversions if row["kind"] == "points"]
    requests = []
    for (kind, name, output_format), (generation, output) in provider._cache.items():
        if kind != "points":
            continue
        publication = provider._backend.point_publications[name]
        source = publication.data
        payload_copy_count = sum(
            row["stream"] == name and row["generation"] == generation and row["output_format"] == output_format.__name__
            for row in conversions
        )
        requests.append(
            {
                "stream": name,
                "generation": generation,
                **_sdp_request_contract(source, output, output_format, payload_copy_count),
            }
        )
    fabric_requests = [row for row in requests if row["format"] == "FabricMeshPoints"]
    publications = provider._backend.point_publications
    geometry_launches = list(getattr(tracer, "geometry_launches", ()))
    streams = []
    for name, publication in publications.items():
        bindings = layout.point_bindings(name)
        count = sum(binding.source_count for binding in bindings)
        if not count:
            continue
        streams.append(
            {
                "name": name,
                "count": count,
                "dirty": publication.dirty,
                "bindings": [
                    {
                        "path": binding.path,
                        "source_offset": binding.source_offset,
                        "source_count": binding.source_count,
                        "output_offset": binding.output_offset,
                        "output_count": binding.output_count,
                    }
                    for binding in bindings
                ],
                "source": _buffer_snapshot(publication.data),
            }
        )
    return {
        "streams": streams,
        "requests": requests,
        "fabric_requests": fabric_requests,
        "conversion_events": conversions,
        "zero_or_one_payload_copy": all(row["payload_copy_count"] in {0, 1} for row in requests),
        "geometry_launches": geometry_launches,
        "request_driven_kernels": all(row["inside_sdp"] for row in geometry_launches),
        "exact_conversions": all(row["exact_payload_copy_count"] and row["pointer_contract"] for row in requests)
        and all(row["inside_sdp"] for row in geometry_launches),
    }


def _visualizer_input_snapshot(
    case: Case,
    visualizers: list[Any],
    provider: Any,
    camera: Any,
    consumer_requests: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Audit each visualizer against its declared data input."""
    declared = VISUALIZER_INPUTS[case.visualizer]
    rows = []
    for index, visualizer in enumerate(visualizers):
        requests = consumer_requests[f"visualizer[{index}]"]
        sdp_bound = visualizer._scene_data_provider is provider
        camera_bound = getattr(getattr(visualizer, "_streaming", None), "camera", None) is camera
        sdp_valid = sdp_bound if "sdp" in declared else not sdp_bound and not requests
        rows.append(
            {
                "visualizer": _describe(visualizer),
                "declared": list(declared),
                "sdp_bound": sdp_bound,
                "sdp_requests": len(requests),
                "camera_bound": camera_bound,
                "valid": sdp_valid and ("camera" not in declared or camera_bound),
            }
        )
    return {"declared": list(declared), "visualizers": rows, "valid": all(row["valid"] for row in rows)}


def _has_exact_request_counts(requests: list[dict[str, Any]], expected: dict[str | None, int]) -> bool:
    """Return whether requests contain exactly the expected number for every stream."""
    return len(requests) == sum(expected.values()) and all(
        sum(request["stream"] == stream for request in requests) == count for stream, count in expected.items()
    )


def _reference_snapshot(
    case: Case, cfg: LifecycleDirectCfg, sim: Any, robot: Any, camera: Any, marker: Any, tracer: _CallTracer
) -> dict[str, Any]:
    provider = sim.get_scene_data_provider()
    renderers = sim._renderer_entries
    visualizers = list(sim.visualizers)
    registry = list(sim._backend_registry.items())
    plan = sim.get_clone_plan()
    layout = plan
    physics_manager = sim._physics_manager
    planned_cfgs = (cfg.ground, cfg.robot, cfg.camera, cfg.light, cfg.marker)
    plan_rows = {
        f"{index}:{type(value).__name__}:{value.prim_path}": []
        if plan is None
        else list(plan.cfg_rows.get(id(value), ()))
        for index, value in enumerate(planned_cfgs)
    }
    visualizer_cfgs = sim._visualizer_cfgs
    construction = {
        "physics": _construction_snapshot(physics_manager, cfg.sim.physics, tracer),
        "robot": _construction_snapshot(robot, cfg.robot, tracer),
        "camera": _construction_snapshot(camera, cfg.camera, tracer),
        "marker": _construction_snapshot(marker, cfg.marker, tracer),
        "renderer": _construction_snapshot(camera._renderer, camera.cfg.renderer_cfg, tracer),
        "visualizers": [
            _construction_snapshot(visualizer, visualizer_cfg, tracer)
            for visualizer, visualizer_cfg in zip(visualizers, visualizer_cfgs, strict=True)
        ],
    }

    registry_snapshot = _registry_snapshot(case, sim, tracer)
    initialization_snapshot = _initialization_snapshot(tracer, physics_manager, camera._renderer)
    visualizer_lifecycle = _visualizer_lifecycle_snapshot(visualizers, tracer, provider, plan)
    newton_resource_snapshot = _newton_resource_snapshot(case, sim, physics_manager, camera._renderer, visualizers)

    movement_requests = [
        request
        for request in tracer.scene_data_requests
        if request["kind"] == "transforms" and request["phase"] == "movement_render_probe"
    ]
    consumers = {"renderer": camera._renderer}
    consumers.update({f"visualizer[{index}]": value for index, value in enumerate(visualizers)})
    consumer_requests = {
        name: [request for request in movement_requests if request["consumer_id"] == hex(id(value))]
        for name, value in consumers.items()
    }
    renderer_requests = consumer_requests["renderer"]
    movement_frames = 2 * READINESS_FRAMES + 1
    visualizer_inputs = _visualizer_input_snapshot(case, visualizers, provider, camera, consumer_requests)
    communication = {
        "request_counts": {name: len(requests) for name, requests in consumer_requests.items()},
        "renderer_main": any(request["stream"] is None for request in renderer_requests),
        "renderer_camera": any(request["stream"] == cfg.camera.prim_path for request in renderer_requests),
        "renderer_request_count_exact": _has_exact_request_counts(
            renderer_requests, {None: movement_frames, cfg.camera.prim_path: movement_frames}
        ),
        "one_provider": all(request["provider_id"] == hex(id(provider)) for request in movement_requests),
    }
    communication["required_requests"] = all(
        communication[name]
        for name in ("renderer_main", "renderer_camera", "renderer_request_count_exact", "one_provider")
    )

    physics_transform_source = provider._backend.transform_publication.data
    transform_conversions = [
        conversion for conversion in tracer.scene_data_conversions if conversion["kind"] == "transforms"
    ]
    transform_requests = []
    for (kind, name, output_format), (generation, output) in provider._cache.items():
        if kind != "transforms":
            continue
        transform_source = physics_transform_source if name is None else provider._transform_publications[name].data
        payload_copy_count = sum(
            conversion["stream"] == name
            and conversion["generation"] == generation
            and conversion["output_format"] == output_format.__name__
            for conversion in transform_conversions
        )
        transform_requests.append(
            {
                "stream": name,
                "generation": generation,
                **_sdp_request_contract(transform_source, output, output_format, payload_copy_count),
            }
        )
    fabric_row = next((request for request in transform_requests if request["format"] == "FabricMatrix44"), None)
    fabric_request = {
        "requested": fabric_row is not None,
        "generation": None if fabric_row is None else fabric_row["generation"],
        "output": None if fabric_row is None else fabric_row["output"],
        "logical_conversion_count": 0 if fabric_row is None else fabric_row["logical_conversion_count"],
        "payload_copy_count": 0 if fabric_row is None else fabric_row["payload_copy_count"],
        "exact_payload_copy_count": fabric_row is None or fabric_row["exact_payload_copy_count"],
    }
    conversion_keys = [
        (
            conversion["kind"],
            conversion.get("stream"),
            conversion["generation"],
            conversion["input_format"],
            conversion["output_format"],
        )
        for conversion in tracer.scene_data_conversions
    ]
    no_redundant_conversions = len(conversion_keys) == len(set(conversion_keys))
    point_data = _point_data_snapshot(provider, tracer, layout)
    return {
        "components": {
            "physics": _describe(physics_manager),
            "sdp": _describe(provider),
            "publisher": _describe(provider._backend),
            "renderer": _describe(camera._renderer),
            "visualizers": [_describe(value) for value in visualizers],
            "marker": _describe(marker),
            "marker_backends": [_describe(value) for value in marker._backends],
        },
        "clone_plan": {
            "object": _describe(plan),
            "sources": [] if plan is None else list(plan.sources),
            "destinations": [] if plan is None else list(plan.destinations),
            "cfg_rows": plan_rows,
            "all_cfgs_covered": all(plan_rows.values()),
        },
        "clone_lifecycle": _clone_lifecycle_snapshot(tracer),
        "stage_lifecycle": _stage_lifecycle_snapshot(case, tracer),
        "initialization_lifecycle": initialization_snapshot,
        "visualizer_lifecycle": visualizer_lifecycle,
        "construction": construction,
        "backend_registry": [
            {
                "backend_type": _describe(backend_type)["type"],
                "clone_roles": sorted(sim._backend_clone_roles.get(backend_type, ())),
                "backend": _describe(backend),
            }
            for backend_type, backend in registry
        ],
        "registry_resolution": registry_snapshot,
        "newton_resource": newton_resource_snapshot,
        "communication": communication,
        "visualizer_inputs": visualizer_inputs,
        "links": {
            "robot_to_physics": robot._physics_manager is physics_manager,
            "physics_to_sdp": physics_manager.get_scene_data_backend() is provider._backend,
            "camera_to_simulation": camera._renderer in renderers,
            "renderer_to_sdp": bool(renderer_requests)
            and all(request["provider_id"] == hex(id(provider)) for request in renderer_requests),
            "visualizer_inputs": visualizer_inputs["valid"],
            "visualizers_to_plan": all(value._clone_plan is plan for value in visualizers),
        },
        "data_movement": {
            "transform_count": _publication_count(physics_transform_source),
            "transform_paths": list(layout.iter_rigid_body_paths()),
            "transform_generation": provider.transform_generation(),
            "transform_publication": _buffer_snapshot(physics_transform_source),
            "fabric_output": fabric_request["output"],
            "fabric_generation": fabric_request["generation"],
            "transform_requests": transform_requests,
            "fabric_request": fabric_request,
            "conversion_events": tracer.scene_data_conversions,
            "exact_conversions": no_redundant_conversions
            and all(request["exact_payload_copy_count"] for request in transform_requests)
            and all(request["pointer_contract"] for request in transform_requests)
            and point_data["exact_conversions"],
            "point_streams": point_data["streams"],
            "point_requests": point_data["requests"],
            "fabric_point_requests": point_data["fabric_requests"],
            "point_zero_or_one_payload_copy": point_data["zero_or_one_payload_copy"],
            "calls": {label: stats for label, stats in tracer.counts.items() if ".SceneDataProvider." in label},
            "publication_calls": {
                label: stats for label, stats in tracer.counts.items() if label.endswith("SceneDataBackend.publish")
            },
        },
    }


def _point_reference_snapshot(
    case: Case,
    cfg: PointLifecycleDirectCfg,
    sim: Any,
    deformable: Any,
    camera: Any,
    tracer: _CallTracer,
) -> dict[str, Any]:
    """Audit the exact clone, construction, registry, and point-consumer graph."""
    provider = sim.get_scene_data_provider()
    renderer = camera._renderer
    visualizers = list(sim.visualizers)
    plan = sim.get_clone_plan()
    layout = plan
    physics_manager = sim._physics_manager
    planned_cfgs = (cfg.deformable, cfg.camera)
    plan_rows = {
        f"{index}:{type(value).__name__}:{value.prim_path}": []
        if plan is None
        else list(plan.cfg_rows.get(id(value), ()))
        for index, value in enumerate(planned_cfgs)
    }
    construction = {
        "physics": _construction_snapshot(physics_manager, cfg.sim.physics, tracer),
        "deformable": _construction_snapshot(deformable, cfg.deformable, tracer),
        "camera": _construction_snapshot(camera, cfg.camera, tracer),
        "renderer": _construction_snapshot(renderer, camera.cfg.renderer_cfg, tracer),
        "visualizers": [
            _construction_snapshot(visualizer, visualizer_cfg, tracer)
            for visualizer, visualizer_cfg in zip(visualizers, sim._visualizer_cfgs, strict=True)
        ],
    }
    consumers = {"renderer": renderer} | {
        f"visualizer[{index}]": visualizer for index, visualizer in enumerate(visualizers)
    }
    requests = [
        request
        for request in tracer.scene_data_requests
        if request["kind"] == "points" and request["phase"] == "point_render_probe"
    ]
    consumer_requests = {
        name: [request for request in requests if request["consumer_id"] == hex(id(consumer))]
        for name, consumer in consumers.items()
    }
    visualizer_inputs = _visualizer_input_snapshot(case, visualizers, provider, camera, consumer_requests)
    renderer_requests = consumer_requests["renderer"]
    communication = {
        "request_counts": {name: len(rows) for name, rows in consumer_requests.items()},
        "renderer_points": bool(renderer_requests),
        "renderer_request_count_exact": _has_exact_request_counts(
            renderer_requests, {name: 2 for name in layout.point_stream_names}
        ),
        "one_provider": bool(requests) and all(request["provider_id"] == hex(id(provider)) for request in requests),
    }
    communication["required_requests"] = all(
        communication[name] for name in ("renderer_points", "renderer_request_count_exact", "one_provider")
    )
    point_data = _point_data_snapshot(provider, tracer, layout)
    point_data["calls"] = {label: stats for label, stats in tracer.counts.items() if ".SceneDataProvider." in label}
    point_data["publication_calls"] = {
        label: stats for label, stats in tracer.counts.items() if label.endswith("SceneDataBackend.publish")
    }
    registry = list(sim._backend_registry.items())
    visualizer_lifecycle = _visualizer_lifecycle_snapshot(
        visualizers, tracer, provider, plan, probe_phase="point_render_probe"
    )
    return {
        "components": {
            "physics": _describe(physics_manager),
            "sdp": _describe(provider),
            "publisher": _describe(provider._backend),
            "deformable": _describe(deformable),
            "renderer": _describe(renderer),
            "visualizers": [_describe(value) for value in visualizers],
        },
        "clone_plan": {
            "object": _describe(plan),
            "sources": [] if plan is None else list(plan.sources),
            "destinations": [] if plan is None else list(plan.destinations),
            "cfg_rows": plan_rows,
            "all_cfgs_covered": all(plan_rows.values()),
        },
        "clone_lifecycle": _clone_lifecycle_snapshot(tracer),
        "stage_lifecycle": _stage_lifecycle_snapshot(case, tracer),
        "initialization_lifecycle": _initialization_snapshot(tracer, physics_manager, renderer),
        "visualizer_lifecycle": visualizer_lifecycle,
        "construction": construction,
        "backend_registry": [
            {
                "backend_type": _describe(backend_type)["type"],
                "clone_roles": sorted(sim._backend_clone_roles.get(backend_type, ())),
                "backend": _describe(backend),
            }
            for backend_type, backend in registry
        ],
        "registry_resolution": _registry_snapshot(case, sim, tracer),
        "newton_resource": _newton_resource_snapshot(case, sim, physics_manager, renderer, visualizers),
        "communication": communication,
        "visualizer_inputs": visualizer_inputs,
        "links": {
            "deformable_to_physics": deformable._physics_manager is physics_manager,
            "physics_to_sdp": physics_manager.get_scene_data_backend() is provider._backend,
            "camera_to_simulation": renderer in sim._renderer_entries,
            "renderer_to_sdp": bool(consumer_requests["renderer"]),
            "visualizer_inputs": visualizer_inputs["valid"],
            "visualizers_to_plan": all(value._clone_plan is plan for value in visualizers),
        },
        "data_movement": point_data,
    }


def _point_probe(
    cfg: PointLifecycleDirectCfg,
    sim: Any,
    deformable: Any,
    camera: Any,
    tracer: _CallTracer,
    checkpoint: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Publish one dirty point generation and request it twice from every configured sink."""
    provider = sim.get_scene_data_provider()
    layout = sim.get_clone_plan()
    streams = layout.point_stream_names
    generations_before = {name: provider.point_generation(name) for name in streams}
    if checkpoint is not None:
        checkpoint("before_first_sim_step")
    sim.step(render=False)
    if checkpoint is not None:
        checkpoint("after_first_sim_step")
    deformable.update(cfg.sim.dt)
    publications = {name: provider._backend.point_publications[name] for name in streams}
    dirty_before_request = {name: publication.dirty for name, publication in publications.items()}
    publisher_before = {name: _buffer_snapshot(publication.data) for name, publication in publications.items()}

    if checkpoint is not None:
        checkpoint("before_probe_render")
    sim.render()
    if checkpoint is not None:
        checkpoint("after_probe_render")
    if checkpoint is not None:
        checkpoint("before_first_camera_update")
    camera.update(cfg.sim.dt, force_recompute=True)
    if checkpoint is not None:
        checkpoint("after_first_camera_update")
    first = _point_data_snapshot(provider, tracer, layout)
    if checkpoint is not None:
        checkpoint("before_second_camera_update")
    camera.update(cfg.sim.dt, force_recompute=True)
    if checkpoint is not None:
        checkpoint("after_second_camera_update")
    second = _point_data_snapshot(provider, tracer, layout)

    generations_after = {name: provider.point_generation(name) for name in streams}
    dirty_after_request = {name: publication.dirty for name, publication in publications.items()}
    publisher_after = {name: _buffer_snapshot(publication.data) for name, publication in publications.items()}
    first_outputs = [(row["stream"], row["format"], row["output"]) for row in first["requests"]]
    second_outputs = [(row["stream"], row["format"], row["output"]) for row in second["requests"]]
    generation_advanced = bool(streams) and all(
        generations_after[name] == generations_before[name] + 1 for name in streams
    )
    dirty_latched = bool(streams) and all(dirty_before_request.values())
    dirty_cleared = bool(streams) and not any(dirty_after_request.values())
    publisher_pointer_stable = publisher_before == publisher_after
    consumer_pointer_stable = bool(first_outputs) and first_outputs == second_outputs
    passed = (
        generation_advanced
        and dirty_latched
        and dirty_cleared
        and publisher_pointer_stable
        and consumer_pointer_stable
        and second["zero_or_one_payload_copy"]
        and second["exact_conversions"]
    )
    return {
        "streams": list(streams),
        "generations_before": generations_before,
        "generations_after": generations_after,
        "dirty_before_request": dirty_before_request,
        "dirty_after_request": dirty_after_request,
        "publisher_before": publisher_before,
        "publisher_after": publisher_after,
        "first_outputs": first_outputs,
        "second_outputs": second_outputs,
        "generation_advanced_once": generation_advanced,
        "dirty_latched": dirty_latched,
        "dirty_cleared": dirty_cleared,
        "publisher_pointer_stable": publisher_pointer_stable,
        "consumer_pointer_stable": consumer_pointer_stable,
        "zero_or_one_payload_copy": second["zero_or_one_payload_copy"],
        "exact_conversions": second["exact_conversions"],
        "passed": passed,
    }


def _newton_rtx_probe(
    case: Case, visualizers: list[Any], checkpoint: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Capture the selected planned-camera composite twice."""
    viewers = [visualizer for visualizer in visualizers if type(visualizer).__name__ == "NewtonRTXVisualizer"]
    frames = []
    for visualizer in viewers:
        for frame_index in range(2):
            if checkpoint is not None:
                checkpoint(f"before_newton_rtx_frame_{frame_index + 1}")
            frames.append(visualizer.render_rgb_array())
            if checkpoint is not None:
                checkpoint(f"after_newton_rtx_frame_{frame_index + 1}")
    expected_calls = 2 * int(case.visualizer == "newton_rtx")
    return {
        "calls": len(frames),
        "expected_calls": expected_calls,
        "frames": [
            None if frame is None else {"shape": list(frame.shape), "dtype": str(frame.dtype)} for frame in frames
        ],
        "passed": len(frames) == expected_calls and all(frame is not None for frame in frames),
    }


def _camera_output(camera: Any):
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    output = camera.data.output["instance_segmentation"]
    tensor = output if isinstance(output, torch.Tensor) else output.torch
    return np.asarray(tensor.detach().cpu()).copy()


def _render_output(cfg: LifecycleDirectCfg, sim: Any, robot: Any, camera: Any):
    started = time.perf_counter()
    sim.render()
    robot.update(cfg.sim.dt)
    camera.update(cfg.sim.dt, force_recompute=True)
    return _camera_output(camera), 1000.0 * (time.perf_counter() - started)


def _ovrtx_clean_xform_write_probe(case: Case, camera: Any, tracer: _CallTracer) -> dict[str, Any]:
    """Require one extra clean OVRTX update to produce no native transform writes."""
    before = int(tracer.counts.get(OVRTX_XFORM_WRITE_LABEL, {}).get("calls", 0))
    if case.renderer == "ovrtx":
        previous_phase, tracer.phase = tracer.phase, "ovrtx_clean_xform_write_probe"
        try:
            camera._renderer.update(camera._render_data, camera.data.intrinsic_matrices)
        finally:
            tracer.phase = previous_phase
    after = int(tracer.counts.get(OVRTX_XFORM_WRITE_LABEL, {}).get("calls", 0))
    calls = after - before
    required = case.renderer == "ovrtx"
    passed = not required or before > 0 and calls == 0
    return {"required": required, "calls": calls, "expected_calls": 0, "passed": passed}


def _image_delta(before: Any, after: Any) -> dict[str, float | int]:
    import numpy as np  # noqa: PLC0415

    delta = before != after
    changed = np.any(delta != 0, axis=-1) if delta.ndim >= 3 else delta != 0
    return {
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_fraction": float(np.count_nonzero(changed) / changed.size),
    }


def _movement_probe(
    cfg: LifecycleDirectCfg, sim: Any, robot: Any, camera: Any, cart_dof_idx: list[int]
) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    render_timings = []
    for _ in range(READINESS_FRAMES):
        baseline_a, render_ms = _render_output(cfg, sim, robot, camera)
        render_timings.append(render_ms)
    baseline_b, render_ms = _render_output(cfg, sim, robot, camera)
    render_timings.append(render_ms)
    baseline_delta = _image_delta(baseline_a, baseline_b)
    provider = sim.get_scene_data_provider()
    generation_before = provider.transform_generation()

    joint_before = robot.data.joint_pos.torch.clone()
    root_before = robot.data.root_pose_w.torch.clone()
    target = joint_before.clone()
    cart_index = int(cart_dof_idx[0])
    target[:, cart_index] += MOVEMENT_OFFSET_M
    root_target = root_before.clone()
    root_target[:, 1] += MOVEMENT_OFFSET_M
    robot.write_root_pose_to_sim_index(root_pose=root_target)
    robot.write_joint_position_to_sim_index(position=target)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(target))
    sim.forward()
    sim.step(render=False)
    robot.update(cfg.sim.dt)
    joint_after = robot.data.joint_pos.torch.clone()
    root_after = robot.data.root_pose_w.torch.clone()

    movement_frame_deltas = []
    for _ in range(READINESS_FRAMES):
        moved_output, render_ms = _render_output(cfg, sim, robot, camera)
        render_timings.append(render_ms)
        movement_frame_deltas.append(_image_delta(baseline_b, moved_output))
    generation_after = provider.transform_generation()
    moved_delta = _image_delta(baseline_b, moved_output)
    cart_delta = float(torch.max(torch.abs(joint_after[:, cart_index] - joint_before[:, cart_index])).item())
    root_delta = float(torch.max(torch.linalg.vector_norm(root_after[:, :3] - root_before[:, :3], dim=-1)).item())
    cart_changed = cart_delta > 1.0e-4
    root_changed = root_delta > 1.0e-4
    physics_changed = cart_changed or root_changed
    segmentation_threshold = int(baseline_delta["changed_pixels"])
    segmentation_changed = int(moved_delta["changed_pixels"]) > segmentation_threshold
    return {
        "frame_render_timings_ms": render_timings,
        "requested_cart_offset_m": MOVEMENT_OFFSET_M,
        "requested_root_offset_m": MOVEMENT_OFFSET_M,
        "observed_cart_delta_m": cart_delta,
        "observed_root_delta_m": root_delta,
        "cart_changed": cart_changed,
        "root_changed": root_changed,
        "physics_changed": physics_changed,
        "capability_discrepancies": [] if cart_changed else ["direct cart joint write was not observed"],
        "baseline_segmentation_delta": baseline_delta,
        "movement_segmentation_delta": moved_delta,
        "movement_frame_deltas": movement_frame_deltas,
        "baseline_segmentation_values": np.unique(baseline_b).tolist(),
        "movement_segmentation_values": np.unique(moved_output).tolist(),
        "segmentation_changed_pixels_threshold": segmentation_threshold,
        "segmentation_changed": segmentation_changed,
        "transform_generation_before": generation_before,
        "transform_generation_after": generation_after,
        "dirty_generation_advanced": generation_after > generation_before,
        "passed": physics_changed and segmentation_changed and generation_after > generation_before,
    }


def _worker(case: Case, args: argparse.Namespace) -> int:
    point_scenario = args._worker_points
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": POINT_TASK if point_scenario else TASK,
        "scenario": "deformable_points" if point_scenario else "rigid",
        "case": case.as_dict(),
        "name": case.name,
        "status": "failed",
        "phase_timings_ms": {},
    }
    tracer: _CallTracer | None = None
    sim = None
    visualizers = []
    runtime_started_ns = time.perf_counter_ns()
    report_written = False

    def write_report() -> None:
        nonlocal report_written
        report["phase_timings_ms"]["runtime_total"] = round(
            (time.perf_counter_ns() - runtime_started_ns) / 1_000_000, 3
        )
        if tracer is not None:
            report["trace"] = tracer.report()
        _write_json(args._worker_output, report)
        report_written = True

    def checkpoint(name: str) -> None:
        """Atomically preserve the latest phase and trace counts for native stalls or crashes."""
        report.setdefault("checkpoints", []).append(
            {
                "name": name,
                "at_ms": round((time.perf_counter_ns() - runtime_started_ns) / 1_000_000, 3),
                "phase": None if tracer is None else tracer.phase,
                "trace_counts": {}
                if tracer is None
                else {label: dict(stats) for label, stats in tracer.counts.items()},
            }
        )
        write_report()

    try:
        with _phase(report, "config_resolution"):
            env_cfg = (
                _resolve_point_cfg(case, args.device, args.num_envs)
                if point_scenario
                else _resolve_env_cfg(case, args.device, args.num_envs)
            )

        from isaaclab.app.sim_launcher import launch_simulation  # noqa: PLC0415

        launcher_args = {
            "device": args.device,
            "headless": True,
            "enable_cameras": True,
            "deterministic": True,
            "livestream": 0,
        }
        with _phase(report, "launcher_scan"):
            report["launcher_scan"] = _launcher_scan_snapshot(case, env_cfg, launcher_args)
        if not report["launcher_scan"]["matches_case"]:
            raise RuntimeError(
                f"Launcher scan disagrees with {case.name}: kit_sources={report['launcher_scan']['kit_sources']}"
            )
        launch_started_ns = time.perf_counter_ns()
        with launch_simulation(env_cfg, launcher_args):
            report["phase_timings_ms"]["runtime_launch"] = round(
                (time.perf_counter_ns() - launch_started_ns) / 1_000_000, 3
            )
            tracer = _CallTracer(case, TIMELINE_LIMIT, points=point_scenario)
            tracer.checkpoint_callback = checkpoint
            with _phase(report, "trace_install", tracer):
                _install_trace(tracer)

            import torch  # noqa: PLC0415

            from isaaclab.cloner import ReplicateSession  # noqa: PLC0415
            from isaaclab.sim import SimulationContext  # noqa: PLC0415

            try:
                with _phase(report, "context_construction_and_backend_bind", tracer):
                    sim = SimulationContext(env_cfg.sim)
                    visualizers = list(sim.visualizers)
                with _phase(report, "clone_and_component_construction", tracer):
                    checkpoint("before_clone")
                    asset_cfgs = (
                        (env_cfg.deformable, env_cfg.deformable.visualizer_cfg, env_cfg.camera)
                        if point_scenario
                        else (env_cfg.ground, env_cfg.robot, env_cfg.camera, env_cfg.light, env_cfg.marker)
                    )
                    with ReplicateSession(
                        asset_cfgs,
                        num_clones=env_cfg.num_envs,
                        env_spacing=env_cfg.env_spacing,
                        clone_strategy=env_cfg.clone_cfg.clone_strategy,
                        env_template=env_cfg.clone_cfg.clone_template,
                        replicate_physics=env_cfg.clone_cfg.replicate_physics,
                    ):
                        if point_scenario:
                            deformable = env_cfg.deformable.class_type(env_cfg.deformable)
                        else:
                            for asset_cfg in (env_cfg.ground, env_cfg.light):
                                asset_cfg.class_type(asset_cfg)
                            robot = env_cfg.robot.class_type(env_cfg.robot)
                            marker = env_cfg.marker.class_type(env_cfg.marker)
                        camera = env_cfg.camera.class_type(env_cfg.camera)
                    checkpoint("after_clone")
                with _phase(report, "post_clone_reset", tracer):
                    checkpoint("before_reset")
                    sim.reset()
                    checkpoint("after_reset")
                    if point_scenario:
                        deformable.update(env_cfg.sim.dt)
                        checkpoint("before_reset_render")
                        sim.render()
                        checkpoint("after_reset_render")
                    else:
                        robot.update(env_cfg.sim.dt)
                        marker.visualize(torch.tensor([[0.0, 0.0, 1.0]], device=env_cfg.sim.device))
                        cart_dof_idx, _ = robot.find_joints("slider_to_cart")
                    checkpoint("before_reset_camera_update")
                    camera.update(env_cfg.sim.dt, force_recompute=True)
                    checkpoint("after_reset_camera_update")
                probe_phase = "point_render_probe" if point_scenario else "movement_render_probe"
                with _phase(report, probe_phase, tracer):
                    report["probe"] = (
                        _point_probe(env_cfg, sim, deformable, camera, tracer, checkpoint)
                        if point_scenario
                        else _movement_probe(env_cfg, sim, robot, camera, cart_dof_idx)
                    )
                    report["probe"]["ovrtx_clean_xform_writes"] = _ovrtx_clean_xform_write_probe(case, camera, tracer)
                    report["probe"]["passed"] &= report["probe"]["ovrtx_clean_xform_writes"]["passed"]
                    checkpoint("before_newton_rtx_probe")
                    report["probe"]["newton_rtx"] = _newton_rtx_probe(case, visualizers, checkpoint)
                    checkpoint("after_newton_rtx_probe")
                    report["probe"]["passed"] &= report["probe"]["newton_rtx"]["passed"]
                with _phase(report, "composition_audit", tracer):
                    report["composition"] = (
                        _point_reference_snapshot(case, env_cfg, sim, deformable, camera, tracer)
                        if point_scenario
                        else _reference_snapshot(case, env_cfg, sim, robot, camera, marker, tracer)
                    )
                with _phase(report, "second_hard_reset_audit", tracer):
                    if case.physics.startswith("newton_"):
                        checkpoint("before_second_hard_reset")
                    report["composition"]["second_hard_reset"] = _newton_hard_reset_snapshot(
                        case, sim, sim._physics_manager, camera._renderer, visualizers
                    )
                    if case.physics.startswith("newton_"):
                        checkpoint("after_second_hard_reset")
                    report["composition"]["newton_finalization"] = _newton_finalization_snapshot(case, tracer)
                construction = report["composition"]["construction"]
                names = (
                    ("physics", "deformable", "camera", "renderer")
                    if point_scenario
                    else (
                        "physics",
                        "robot",
                        "camera",
                        "renderer",
                        "marker",
                    )
                )
                construction_rows = [construction[name] for name in names]
                construction_rows.extend(construction["visualizers"])
                movement = report["composition"]["data_movement"]
                visualizer_lifecycle = report["composition"]["visualizer_lifecycle"]
                report["gates"] = {
                    "all_cfgs_planned": report["composition"]["clone_plan"]["all_cfgs_covered"],
                    "one_clone_lifecycle": report["composition"]["clone_lifecycle"]["exactly_one"],
                    "one_stage_handoff": report["composition"]["stage_lifecycle"]["exactly_once_when_required"],
                    "initialization_after_clone": report["composition"]["initialization_lifecycle"]["after_clone"]
                    and all(row["initialized_after_clone"] for row in visualizer_lifecycle["visualizers"]),
                    "visualizers_share_clone_plan": all(
                        row["exact_shared_inputs"] for row in visualizer_lifecycle["visualizers"]
                    ),
                    "exact_cfg_construction": all(row["exact_cfg_construction"] for row in construction_rows),
                    "one_resource_per_type": report["composition"]["registry_resolution"]["one_resource_per_type"]
                    and report["composition"]["newton_resource"]["shared"],
                    "visual_geometry_scope": report["composition"]["newton_resource"]["visual_shapes_match"],
                    "stable_newton_hard_reset": report["composition"]["second_hard_reset"]["passed"],
                    "one_newton_model_finalize": report["composition"]["newton_finalization"][
                        "exactly_once_when_required"
                    ],
                    "sdp_data_boundary": report["composition"]["communication"]["required_requests"]
                    and movement["exact_conversions"],
                    "visualizer_runtime": visualizer_lifecycle["ready"],
                    "direct_links": all(report["composition"]["links"].values()),
                    "point_publication" if point_scenario else "motion_and_segmentation": report["probe"]["passed"],
                }
            except BaseException as exc:
                report["status"] = "failed"
                report["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "phase": report.pop("_failed_phase", None),
                    "traceback": traceback.format_exc()[-12000:],
                }
            finally:
                if sim is not None:
                    provider = sim.get_scene_data_provider()
                    plan = sim.get_clone_plan()
                    try:
                        checkpoint("before_teardown")
                        with _phase(report, "simulation_close", tracer):
                            sim.stop()
                            sim.clear_instance()
                        checkpoint("after_teardown")
                    except BaseException as exc:
                        cleanup_error = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "phase": report.pop("_failed_phase", None),
                            "traceback": traceback.format_exc()[-12000:],
                        }
                        report["cleanup_error" if "error" in report else "error"] = cleanup_error
                    if "gates" in report:
                        teardown = _visualizer_lifecycle_snapshot(visualizers, tracer, provider, plan)
                        report["composition"]["visualizer_lifecycle"]["teardown"] = teardown
                        report["gates"]["visualizer_close"] = teardown["closed"]
                    sim = None
                if "gates" in report:
                    report["failed_gates"] = [name for name, passed in report["gates"].items() if not passed]
                    if report["failed_gates"] and "error" not in report:
                        report["error"] = {
                            "type": "LifecycleAuditError",
                            "message": f"Lifecycle audit failed: {', '.join(report['failed_gates'])}",
                        }
                    elif not report["failed_gates"] and "error" not in report:
                        report["status"] = "passed"
                write_report()
    except BaseException as exc:
        report_written = False
        report["status"] = "failed"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "phase": report.pop("_failed_phase", None),
            "traceback": traceback.format_exc()[-12000:],
        }
    finally:
        if not report_written:
            write_report()
    return 0 if report["status"] == "passed" else 1


def _failed_process_report(
    case: Case,
    command: list[str],
    *,
    returncode: int | None,
    stdout: str,
    stderr: str,
    error: str,
    points: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": POINT_TASK if points else TASK,
        "scenario": "deformable_points" if points else "rigid",
        "case": case.as_dict(),
        "name": case.name,
        "status": "failed",
        "error": {"type": "WorkerProcessError", "message": error},
        "process": {
            "command": command,
            "returncode": returncode,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
        },
    }


def _timeout_process_report(
    case: Case,
    command: list[str],
    worker_output: Path,
    *,
    timeout: float,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    points: bool = False,
) -> dict[str, Any]:
    """Attach a timeout to the latest atomic worker checkpoint when one exists."""
    failed = _failed_process_report(
        case,
        command,
        returncode=None,
        stdout=_tail(stdout),
        stderr=_tail(stderr),
        error=f"Worker exceeded {timeout:g} seconds",
        points=points,
    )
    try:
        report = load_worker_report(worker_output, case)
    except ValueError:
        return failed
    report.update(status="failed", error=failed["error"], process=failed["process"])
    return report


def _run_parent(cases: tuple[Case, ...], args: argparse.Namespace) -> int:
    started = datetime.now(timezone.utc)
    point_scenario = args.points
    aggregate: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": POINT_TASK if point_scenario else TASK,
        "scenario": "deformable_points" if point_scenario else "rigid",
        "started_at": started.isoformat(),
        "matrix": point_manifest() if point_scenario else matrix_manifest(),
        "requested_cases": [case.name for case in cases],
        "cases": [],
        "summary": {},
    }
    _write_json(args.output, aggregate)
    with tempfile.TemporaryDirectory(prefix="isaaclab-backend-lifecycle-") as temporary_dir:
        for index, case in enumerate(cases, start=1):
            worker_output = Path(temporary_dir) / f"{case.name}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--_worker-case",
                case.name,
                "--_worker-output",
                str(worker_output),
                "--device",
                args.device,
                "--num-envs",
                str(args.num_envs),
            ]
            if point_scenario:
                command.append("--_worker-points")
            print(f"[{index:02d}/{len(cases):02d}] {case.name}", flush=True)
            process_started_ns = time.perf_counter_ns()
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    check=False,
                )
                try:
                    case_report = load_worker_report(worker_output, case)
                except ValueError as exc:
                    case_report = _failed_process_report(
                        case,
                        command,
                        returncode=result.returncode,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        error=str(exc),
                        points=point_scenario,
                    )
                else:
                    case_report["process"] = {
                        "command": command,
                        "returncode": result.returncode,
                        "duration_ms": round((time.perf_counter_ns() - process_started_ns) / 1_000_000, 3),
                        "stdout_tail": _tail(result.stdout),
                        "stderr_tail": _tail(result.stderr),
                    }
                    if result.returncode != 0 and case_report["status"] == "passed":
                        case_report["status"] = "failed"
                        case_report["error"] = {
                            "type": "WorkerProcessError",
                            "message": f"Worker exited with code {result.returncode} after reporting success",
                        }
            except subprocess.TimeoutExpired as exc:
                case_report = _timeout_process_report(
                    case,
                    command,
                    worker_output,
                    timeout=args.timeout,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                    points=point_scenario,
                )
            aggregate["cases"].append(case_report)
            passed = sum(report["status"] == "passed" for report in aggregate["cases"])
            failed = len(aggregate["cases"]) - passed
            aggregate["summary"] = {
                "passed": passed,
                "failed": failed,
                "excluded": len(aggregate["matrix"]["excluded_cases"]),
                "complete": len(aggregate["cases"]) == len(cases),
            }
            _write_json(args.output, aggregate)
            print(f"    {case_report['status'].upper()}", flush=True)

    aggregate["finished_at"] = datetime.now(timezone.utc).isoformat()
    aggregate["duration_s"] = (datetime.now(timezone.utc) - started).total_seconds()
    aggregate["summary"]["success"] = aggregate["summary"]["failed"] == 0
    _write_json(args.output, aggregate)
    print(f"Wrote {args.output}")
    return 0 if aggregate["summary"]["success"] else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", dest="list_cases", help="Print the matrix without running it.")
    parser.add_argument("--case", action="append", default=[], help="Run one case name; may be repeated.")
    parser.add_argument(
        "--points",
        action="store_true",
        help="Run the point-publication manifest instead of the rigid matrix.",
    )
    parser.add_argument("--output", type=Path, default=Path("backend_lifecycle.json"), help="Aggregate JSON path.")
    parser.add_argument("--device", default="cuda:0", help="Simulation device passed to every worker.")
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS, help="Number of cloned environments.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Timeout for each worker [s].")
    parser.add_argument("--_worker-case", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-points", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.num_envs <= 0:
        parser.error("--num-envs must be greater than zero")
    if args._worker_case and args._worker_output is None:
        parser.error("worker mode requires --_worker-output")
    return args


def main() -> int:
    args = _parse_args()
    if args._worker_case:
        return _worker(_case_from_name(args._worker_case), args)
    manifest = point_manifest() if args.points else matrix_manifest()
    if args.list_cases:
        if not args.points:
            manifest["cases"] = [
                dict(case=case.as_dict(), name=case.name, exclusion=exclusion_reason(case))
                for case in enumerate_cases()
            ]
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    default_cases = POINT_CASES if args.points else enumerate_cases()
    requested = tuple(_case_from_name(name) for name in args.case) if args.case else default_cases
    if args.points and any(case not in POINT_CASES for case in requested):
        invalid = ", ".join(case.name for case in requested if case not in POINT_CASES)
        raise SystemExit(f"Requested case(s) are absent from the point manifest: {invalid}")
    if (
        not args.points
        and args.case
        and (excluded := [(case, exclusion_reason(case)) for case in requested if exclusion_reason(case)])
    ):
        details = ", ".join(f"{case.name} ({reason})" for case, reason in excluded)
        raise SystemExit(f"Requested case(s) are deliberately excluded: {details}")
    runnable = requested if args.points else tuple(case for case in requested if exclusion_reason(case) is None)
    return _run_parent(runnable, args)


if __name__ == "__main__":
    raise SystemExit(main())
