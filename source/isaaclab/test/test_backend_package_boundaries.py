# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structural dependency gates for independently removable backend packages."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_ROOT = _REPO_ROOT / "source"
_BACKEND_PACKAGES = ("isaaclab_newton", "isaaclab_physx", "isaaclab_ov")
_RENDERER_ROOTS = (
    _SOURCE_ROOT / "isaaclab/isaaclab/renderers",
    _SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/renderers",
    _SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/renderers",
    _SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/renderers",
)
_RENDERER_VISUALIZER_ROOTS = (
    *_RENDERER_ROOTS,
    _SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers",
)
_CFG_LIFECYCLE_METHODS = {
    "build",
    "build_sink",
    "build_visualizer",
    "clone_context",
    "create_visualizer",
    "get_visualizer_type",
    "make_signal_fn",
    "to_pipeline_args",
    "to_solver_config",
    "_get_dynamics_solver_config",
    "_post_spawn",
}
_PHYSICS_MANAGER_FILES = (
    "isaaclab/isaaclab/physics/physics_manager.py",
    "isaaclab_physx/isaaclab_physx/physics/physx_manager.py",
    "isaaclab_newton/isaaclab_newton/physics/newton_manager.py",
    "isaaclab_newton/isaaclab_newton/physics/mjwarp_manager.py",
    "isaaclab_newton/isaaclab_newton/physics/xpbd_manager.py",
    "isaaclab_newton/isaaclab_newton/physics/vbd_manager.py",
    "isaaclab_newton/isaaclab_newton/physics/featherstone_manager.py",
    "isaaclab_newton/isaaclab_newton/physics/kamino_manager.py",
    "isaaclab_newton/isaaclab_newton/physics/mpm_manager.py",
    "isaaclab_ov/isaaclab_ov/physics/ovphysx_manager.py",
    "isaaclab_contrib/isaaclab_contrib/coupling/coupler.py",
    "isaaclab_contrib/isaaclab_contrib/custom_coupling/coupled_mjwarp_vbd_manager.py",
)
_PHYSICS_MANAGER_NAMES = {
    "PhysicsManager",
    "PhysxManager",
    "OvPhysxManager",
    "NewtonManager",
    "NewtonMJWarpManager",
    "NewtonXPBDManager",
    "NewtonVBDManager",
    "NewtonFeatherstoneManager",
    "NewtonKaminoManager",
    "NewtonMPMManager",
    "NewtonCouplerManager",
    "NewtonCoupledMJWarpVBDManager",
}
_EXACT_CFG_CONSTRUCTION_SITES = {
    "isaaclab/isaaclab/sim/simulation_context.py": ("self.cfg.physics", "cfg"),
    "isaaclab/isaaclab/envs/direct_rl_env.py": ("self.cfg.scene",),
    "isaaclab/isaaclab/envs/direct_marl_env.py": ("self.cfg.scene",),
    "isaaclab/isaaclab/envs/manager_based_env.py": ("self.cfg.scene",),
    "isaaclab/isaaclab/envs/leapp_deployment_env.py": ("cfg.scene",),
}


def _imported_roots(node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name.partition(".")[0] for alias in node.names}
    if node.level == 0 and node.module:
        return {node.module.partition(".")[0]}
    return set()


def _is_cfg_expression(node: ast.expr) -> bool:
    """Return whether an expression names a cfg object or one of its members."""
    while isinstance(node, ast.Attribute):
        if node.attr == "cfg" or node.attr.endswith("_cfg"):
            return True
        node = node.value
    return isinstance(node, ast.Name) and (node.id == "cfg" or node.id.endswith("_cfg"))


def test_backend_packages_do_not_import_sibling_backends_or_optional_contrib() -> None:
    """Each backend package must remain removable without another backend or optional contrib."""
    offenders = []
    for package in _BACKEND_PACKAGES:
        package_root = _SOURCE_ROOT / package / package
        forbidden_roots = (set(_BACKEND_PACKAGES) - {package}) | {"isaaclab_contrib"}
        for path in sorted((*package_root.rglob("*.py"), *package_root.rglob("*.pyi"))):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            lines = source.splitlines()
            imports = (node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
            for node in sorted(imports, key=lambda item: item.lineno):
                if _imported_roots(node) & forbidden_roots:
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert not offenders, "Backend packages import sibling or optional packages:\n" + "\n".join(offenders)


def test_renderers_and_visualizers_do_not_reach_into_physics() -> None:
    """Keep publishers and physics ownership behind scene-data and resource contracts."""
    offenders = []
    for production_root in _RENDERER_VISUALIZER_ROOTS:
        for path in sorted((*production_root.rglob("*.py"), *production_root.rglob("*.pyi"))):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            lines = source.splitlines()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    physics_import = any("physics" in alias.name.split(".") for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    physics_import = "physics" in (node.module or "").split(".")
                else:
                    physics_import = False
                reaches_physics = isinstance(node, ast.Attribute) and node.attr in {
                    "_get_backend",
                    "_physics_step_count",
                    "backend",
                    "get_contacts",
                    "get_state_0",
                    "get_state_1",
                    "physics_backend",
                    "_physics_manager",
                    "physics_manager",
                    "update_from_sdp",
                }
                reaches_physics |= isinstance(node, ast.Name) and node.id == "_last_render_update_key"
                if physics_import or reaches_physics:
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert not offenders, "Renderer/visualizer production code crosses a provider boundary:\n" + "\n".join(
        sorted(offenders)
    )


def test_ovrtx_does_not_convert_published_transforms() -> None:
    """OVRTX requests its sink layout from SDP and never gathers or converts physics state itself."""
    renderer_root = _SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/renderers"
    sources = "\n".join(path.read_text(encoding="utf-8") for path in renderer_root.glob("*.py"))
    forbidden = {
        "_rigid_source_indices",
        "request_transforms(SceneDataFormat.Transform)",
        "sync_rigid_transforms_kernel",
        "wp.transform_to_matrix",
    }
    offenders = sorted(symbol for symbol in forbidden if symbol in sources)

    assert not offenders, "OVRTX retains renderer-side transform conversion: " + ", ".join(offenders)


def test_ovrtx_scene_requests_native_publications_instead_of_moving_physics_data() -> None:
    """Every native ingest requests its exact SDP format."""
    path = _SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/renderers/ovrtx_scene.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    scene = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OvrtxScene")
    dynamic_writes = "\n".join(
        ast.get_source_segment(source, node) or ""
        for node in scene.body
        if isinstance(node, ast.FunctionDef) and node.name in {"write_xforms", "write_points"}
    )

    assert "transform_format = SceneDataFormat.TransposedMatrix44d" in source
    assert "point_format = SceneDataFormat.HostMeshPoints" in source
    assert ".numpy()" not in dynamic_writes
    assert "ascontiguousarray" not in source
    assert "synchronize_device" not in dynamic_writes
    assert "DataAccess.ASYNC" in dynamic_writes


def test_ovrtx_has_one_concrete_scene_path() -> None:
    """OVRTX cannot regain a selector, fallback, or camera-client-owned native scene."""
    scene = (_SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/renderers/ovrtx_scene.py").read_text(encoding="utf-8")
    renderer = (_SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/renderers/ovrtx_renderer.py").read_text(encoding="utf-8")
    context = (_SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/cloner/replicate.py").read_text(encoding="utf-8")
    forbidden = {
        "ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS",
        "ISAAC_LAB_OVRTX_USE_OVSTAGE",
        "OVSTAGE_AVAILABLE",
        "OvrtxLegacyScene",
        "OvstageScene",
        "make_ovrtx_scene",
        "ovrtx_use_ovstage_enabled",
        "suppress_deprecation_warnings",
        "import ovstage",
        "advance_write_floor",
    }

    assert not sorted(symbol for symbol in forbidden if symbol in scene + renderer)
    assert "ovstage" not in scene.lower()
    assert "ordinal" not in scene
    assert "open_usd_from_string" in scene
    assert "clone_usd" in scene
    assert "self._ovrtx_scene = scene_type(renderer_type(config))" in context
    assert "scene.attach()" not in context
    assert "_ovrtx_renderer" not in context
    assert "_ovrtx_temp_usd_dir" not in context
    assert "self._scene =" not in renderer
    assert "self._renderer =" not in renderer
    assert "def scene(" not in renderer
    assert "def write_matrices(" not in scene


def test_ovrtx_segmentation_never_invents_missing_native_metadata() -> None:
    """Incomplete native instance maps fail instead of being relabelled UNLABELLED."""
    source = (_SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/renderers/ovrtx_annotator_utils.py").read_text(encoding="utf-8")

    assert "stable_id_to_path[stable_id]" in source
    assert "semantic_id_to_labels[semantic_id]" in source
    assert ".get(stable_id" not in source
    assert ".get(semantic_id" not in source


def test_ovrtx_selected_lifecycle_never_silently_skips_required_work() -> None:
    """A selected OVRTX update requires one provider and planned camera bindings."""
    source = (_SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/renderers/ovrtx_renderer.py").read_text(encoding="utf-8")
    context = (_SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/cloner/replicate.py").read_text(encoding="utf-8")

    assert "self._clone_ctx._update_ovrtx(self._camera_xforms, render_data.transform_stream)" in source
    assert "OVRTX updates require an initialized scene-data provider" in context
    assert "OVRTX updates require clone-time camera bindings" in source
    assert "OVRTX cannot initialize before a camera supplies" in source
    assert "OVRTX initialized without its render product" in source
    assert "OVRTX produced no frame for render product" in source


def test_ov_articulations_do_not_reparse_the_exported_physics_stage() -> None:
    """OV articulation metadata is retained from its planned prototype before cloning."""
    manager_source = (_SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/physics/ovphysx_manager.py").read_text(encoding="utf-8")
    articulation_source = (_SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/assets/articulation/articulation.py").read_text(
        encoding="utf-8"
    )

    assert "_stage_usda" not in manager_source + articulation_source
    assert "ImportFromString" not in articulation_source
    assert "stage.Traverse()" not in articulation_source


def test_renderers_do_not_fetch_a_global_stage() -> None:
    """Renderers receive clone-time stages explicitly instead of consulting process globals."""
    sources = "\n".join(path.read_text(encoding="utf-8") for root in _RENDERER_ROOTS for path in root.rglob("*.py"))
    assert "get_current_stage" not in sources
    assert "omni.usd.get_context().get_stage" not in sources


def test_xr_pose_consumers_use_planned_scene_data() -> None:
    """XR camera anchoring and target rebasing cannot rediscover physics poses through USDRT."""
    utils = (_SOURCE_ROOT / "isaaclab_teleop/isaaclab_teleop/xr_anchor_utils.py").read_text(encoding="utf-8")
    device = (_SOURCE_ROOT / "isaaclab_teleop/isaaclab_teleop/isaac_teleop_device.py").read_text(encoding="utf-8")
    cfg = (_SOURCE_ROOT / "isaaclab_teleop/isaaclab_teleop/xr_cfg.py").read_text(encoding="utf-8")
    widget = (_SOURCE_ROOT / "isaaclab/isaaclab/ui/xr_widgets/instruction_widget.py").read_text(encoding="utf-8")
    forbidden = {"get_current_stage", "get_current_stage_id", "omni.usd", "usdrt", "GetPrimAtPath"}

    assert "plan.match_frames(prim_path)" in utils
    assert "request_transforms(SceneDataFormat.HostTransposedMatrix44d" in utils
    assert "SpatialSource.new_prim_path_source(prim_path_source)" in widget
    assert "CopyFabricPrim" not in widget
    assert "CopyPrim" not in widget
    assert not sorted(symbol for symbol in forbidden if symbol in utils + device + widget)
    assert "CUSTOM" not in cfg + utils
    assert "anchor_rotation_custom_func" not in cfg + utils
    assert "Callable" not in cfg


def test_task_runtime_geometry_comes_from_the_clone_plan() -> None:
    """Task rewards and reset checks cannot walk the completed USD scene for geometry."""
    trocar = (_SOURCE_ROOT / "isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/mdp/rewards.py").read_text(
        encoding="utf-8"
    )
    trocar_cfg = (
        _SOURCE_ROOT / "isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/g129_dex3_env_cfg.py"
    ).read_text(encoding="utf-8")
    stack = (_SOURCE_ROOT / "isaaclab_tasks/isaaclab_tasks/contrib/stack/mdp/stack_events.py").read_text(
        encoding="utf-8"
    )
    lift = "\n".join(
        (_SOURCE_ROOT / path).read_text(encoding="utf-8")
        for path in (
            "isaaclab_tasks/isaaclab_tasks/core/lift/mdp/utils.py",
            "isaaclab_tasks/isaaclab_tasks/core/lift/mdp/events.py",
        )
    )
    forbidden = {
        "get_current_stage",
        "find_matching_prims",
        "get_all_matching_child_prims",
        "GetPrimAtPath",
        '"/World/envs/env_0',
    }

    assert "FrameTransformerCfg(" in trocar_cfg
    assert "target_pos_w" in trocar
    assert "plan.match_geometry_targets" in lift
    assert "plan.frames" not in lift
    assert "plan.geometries" not in lift
    assert "geometry.frame" in lift
    assert "light_prim = env.scene[asset_cfg.name].prim" in stack
    assert not sorted(symbol for symbol in forbidden if symbol in trocar + stack + lift)


def test_newton_physics_has_no_renderer_conditioned_runtime() -> None:
    """Newton physics publishes scene data without selecting a renderer or Kit transport."""
    root = _SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/physics"
    sources = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    forbidden = {"has_kit", "_usdrt_stage"}

    assert not sorted(symbol for symbol in forbidden if symbol in sources)


def test_renderer_stage_preparation_requires_a_clone_plan() -> None:
    """A renderer cannot draw a scene assembled outside the one cloning lifecycle."""
    paths = (
        "isaaclab/isaaclab/renderers/base_renderer.py",
        "isaaclab_physx/isaaclab_physx/renderers/isaac_rtx_renderer.py",
        "isaaclab_newton/isaaclab_newton/renderers/newton_warp_renderer.py",
        "isaaclab_newton/isaaclab_newton/renderers/segmentation.py",
    )
    sources = "\n".join((_SOURCE_ROOT / path).read_text(encoding="utf-8") for path in paths)

    assert "ClonePlan | None" not in sources
    assert sources.count("requires an active clone plan") == len(paths)


def test_renderers_cannot_synthesize_unplanned_lights() -> None:
    """Lights are scene declarations covered by ClonePlan, never renderer-created fallbacks."""
    sources = "\n".join(path.read_text(encoding="utf-8") for root in _RENDERER_ROOTS for path in root.rglob("*.py"))
    forbidden = {"create_default_light(", "UsdLux.", "DistantLight.Define(", "DomeLight.Define("}

    assert not sorted(token for token in forbidden if token in sources)


def test_newton_segmentation_rejects_shapes_outside_the_clone_plan() -> None:
    """An unplanned Newton shape cannot silently appear as UNLABELLED geometry."""
    source = (_SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/renderers/segmentation.py").read_text(encoding="utf-8")

    assert "cloner.query.path_to_source(plan, prim_path)" in source
    assert "is not covered by the clone plan" in source


def test_clone_composition_never_discovers_content_outside_plan_sources() -> None:
    """Clone composition traverses declared sources instead of auditing a finished stage."""
    source = (_SOURCE_ROOT / "isaaclab/isaaclab/cloner/scene_layout.py").read_text(encoding="utf-8")
    forbidden = {
        "Flatten",
        "GetChildren",
        "GetDefaultPrim",
        "GetObjectAtPath",
        "GetPrimAtPath",
        "GetPseudoRoot",
        "PrimRange",
        "Traverse",
        "TraverseAll",
    }
    allowed = {
        (
            "isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualization_markers.py",
            "_process_prototype_prim",
            "GetChildren",
        )
    }
    offenders = []
    for root in _RENDERER_VISUALIZER_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in forbidden:
                    owner = node
                    while owner in parents and not isinstance(owner, ast.FunctionDef):
                        owner = parents[owner]
                    key = (str(path.relative_to(_SOURCE_ROOT)), getattr(owner, "name", "<module>"), node.attr)
                    if key in allowed:
                        continue
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {node.attr}")

    assert "GetPseudoRoot" not in source
    assert "Unplanned drawable prims" not in source
    assert not offenders, "Renderers or visualizers rediscover the completed stage:\n" + "\n".join(offenders)


def test_clone_plan_has_no_nested_scene_layout_owner() -> None:
    """Compiled topology lives directly on ``ClonePlan``, never behind a second aggregate."""
    offenders = []
    package_roots = tuple(
        root / root.name for root in _SOURCE_ROOT.iterdir() if root.is_dir() and (root / root.name).is_dir()
    )
    for path in (path for root in package_roots for path in root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative_path = path.relative_to(_SOURCE_ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SceneLayout":
                offenders.append(f"{relative_path}:{node.lineno}: nested SceneLayout class")
            elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                alias.name == "SceneLayout" for alias in node.names
            ):
                offenders.append(f"{relative_path}:{node.lineno}: SceneLayout import")
            elif isinstance(node, ast.Name) and node.id == "SceneLayout":
                offenders.append(f"{relative_path}:{node.lineno}: SceneLayout type")
            elif isinstance(node, ast.Constant) and node.value == "SceneLayout":
                offenders.append(f"{relative_path}:{node.lineno}: SceneLayout forward type")
            elif isinstance(node, ast.Attribute) and node.attr == "scene_layout":
                offenders.append(f"{relative_path}:{node.lineno}: nested scene_layout access")
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "scene_layout"
            ):
                offenders.append(f"{relative_path}:{node.lineno}: scene_layout field")

    assert not offenders, "ClonePlan topology regained a nested owner:\n" + "\n".join(offenders)


def test_newton_composes_geometry_only_from_plan_sources() -> None:
    """Newton cannot recover global geometry by importing the surrounding USD stage."""
    source = (_SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/cloner/replicate.py").read_text(encoding="utf-8")

    assert "builder.add_usd(" not in source
    assert "(*sources, *global_sources)" in source
    assert 'if "{}" not in destination' in source


def test_isaac_rtx_never_returns_a_frame_after_a_failed_prerequisite() -> None:
    """A selected RTX path fails instead of returning stale, black, or partially streamed output."""
    renderer = (_SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/renderers/isaac_rtx_renderer.py").read_text(
        encoding="utf-8"
    )
    utils_path = _SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/renderers/isaac_rtx_renderer_utils.py"
    utils = utils_path.read_text(encoding="utf-8")
    functions = {
        node.name: node
        for node in ast.walk(ast.parse(utils, filename=str(utils_path)))
        if isinstance(node, ast.FunctionDef)
    }

    assert "will be ignored" not in renderer
    assert "disableColorRender" not in renderer
    assert "proceeding anyway" not in utils
    assert not any(isinstance(node, ast.Try) for node in ast.walk(functions["ensure_rtx_hydra_engine_attached"]))
    assert "raise TimeoutError" in utils

    renderer_functions = {
        node.name: ast.get_source_segment(renderer, node) or ""
        for node in ast.walk(ast.parse(renderer))
        if isinstance(node, ast.FunctionDef)
    }
    lifecycle_paths = [
        _SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_context.py",
        *(_SOURCE_ROOT / path for path in _PHYSICS_MANAGER_FILES),
        *(_SOURCE_ROOT / "isaaclab/isaaclab/envs").rglob("*.py"),
        *(_SOURCE_ROOT / "isaaclab_experimental/isaaclab_experimental/envs").rglob("*.py"),
    ]
    lifecycle = "\n".join(path.read_text(encoding="utf-8") for path in lifecycle_paths)

    assert "assets_loading" not in lifecycle
    assert "wait_for_textures" not in lifecycle
    assert "ensure_isaac_rtx_render_update()" in renderer_functions["render"]
    assert "get_stage_streaming_status()" in ast.get_source_segment(utils, functions["_get_stage_streaming_busy"])
    assert "_wait_for_streaming_complete()" in ast.get_source_segment(
        utils, functions["ensure_isaac_rtx_render_update"]
    )
    assert "set_instance_mapping_semantic_filter" not in renderer_functions["create_render_data"]
    assert "set_instance_mapping_semantic_filter" in renderer_functions["render"]


def test_camera_pose_crosses_the_renderer_boundary_only_through_sdp() -> None:
    """Camera frames publish once; each renderer requests its sink layout from SDP."""
    camera = (_SOURCE_ROOT / "isaaclab/isaaclab/sensors/camera/camera.py").read_text(encoding="utf-8")
    base = (_SOURCE_ROOT / "isaaclab/isaaclab/renderers/base_renderer.py").read_text(encoding="utf-8")
    newton = (_SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/renderers/newton_warp_renderer.py").read_text(
        encoding="utf-8"
    )
    isaac_rtx = (_SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/renderers/isaac_rtx_renderer.py").read_text(
        encoding="utf-8"
    )
    ovrtx = (_SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/renderers/ovrtx_renderer.py").read_text(encoding="utf-8")
    ovrtx_context = (_SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/cloner/replicate.py").read_text(encoding="utf-8")
    ovrtx_kernels = (_SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/renderers/ovrtx_renderer_kernels.py").read_text(
        encoding="utf-8"
    )

    forbidden_updates = {"update_transforms", "update_geometries", "update_camera"}
    for class_name, source in (
        ("BaseRenderer", base),
        ("NewtonWarpRenderer", newton),
        ("IsaacRtxRenderer", isaac_rtx),
        ("OVRTXRenderer", ovrtx),
    ):
        renderer = next(
            node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        methods = {node.name: node for node in renderer.body if isinstance(node, ast.FunctionDef)}
        assert forbidden_updates.isdisjoint(methods)
        assert [argument.arg for argument in methods["update"].args.args] == ["self", "render_data", "intrinsics"]
    assert "SceneDataPublication(SceneDataFormat.Vec3_Quat()" in camera
    assert "GetStageUpAxis" not in camera
    assert camera.count("self._renderer.update(") == 1
    assert not any(f"self._renderer.{name}(" in camera for name in forbidden_updates)
    assert not any(name in camera for name in ("renderer_type", "get_settings_manager", "skipTonemapping"))
    assert "request_transforms(\n            SceneDataFormat.Transform, name=render_data.transform_stream" in newton
    assert "provider.request_transforms(scene.transform_format, name=transform_stream)" in ovrtx_context
    assert "SceneDataFormat.FabricMatrix44, name=render_data.spec.cfg.prim_path" in isaac_rtx
    assert all(
        rejected not in isaac_rtx
        for rejected in ("_camera_xform_specs", "Gf.Matrix4d", "!resetXformStack!", "HostTransposedMatrix44d")
    )
    assert "convert_camera_frame_orientation_convention" not in newton + ovrtx
    assert "create_camera_transforms_kernel" not in ovrtx + ovrtx_kernels


def test_newton_renderer_has_one_visualization_state_resolver() -> None:
    """The native resource resolves state once before launching its registered renderer task."""
    path = _SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/renderers/newton_warp_renderer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    renderer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NewtonWarpRenderer")
    methods = {node.name: node for node in renderer.body if isinstance(node, ast.FunctionDef)}

    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "request_visualization_state"
        for node in ast.walk(methods["update"])
    )
    assert not any(isinstance(node, ast.Attribute) and node.attr == "_state" for node in ast.walk(renderer))
    render_calls = [node for node in ast.walk(methods["render"]) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "_update_sensor_tasks" for call in render_calls
    )
    launch_calls = [node for node in ast.walk(methods["_launch_render"]) if isinstance(node, ast.Call)]
    sensor_update = next(
        call
        for call in launch_calls
        if isinstance(call.func, ast.Attribute) and ast.unparse(call.func.value) == "self.newton_sensor"
    )
    assert ast.unparse(sensor_update.args[0]) == "self._newton_backend._sensor_state"


def test_camera_requests_have_no_silent_output_or_calibration_fallback() -> None:
    """An explicit camera contract either works exactly or fails at its source."""
    camera = (_SOURCE_ROOT / "isaaclab/isaaclab/sensors/camera/camera.py").read_text(encoding="utf-8")
    renderers = "\n".join(
        (_SOURCE_ROOT / path).read_text(encoding="utf-8")
        for path in (
            "isaaclab_physx/isaaclab_physx/renderers/isaac_rtx_renderer.py",
            "isaaclab_ov/isaaclab_ov/renderers/ovrtx_renderer.py",
            "isaaclab_newton/isaaclab_newton/renderers/newton_warp_renderer.py",
        )
    )
    ovrtx_usd = (_SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/renderers/ovrtx_usd.py").read_text(encoding="utf-8")
    tasks = "\n".join(path.read_text(encoding="utf-8") for path in (_SOURCE_ROOT / "isaaclab_tasks").rglob("*.py"))

    assert "does not support requested data types" in camera
    assert "must request at least one renderer output" in camera
    assert "contains duplicate outputs" in camera
    assert "selects multiple simple shading modes" in camera
    assert "will not produce them" not in camera
    assert "falling back to the focal-length/aperture projection" not in camera
    assert "skipped one or more cameras" not in camera
    assert "Multiple simple shading" in renderers
    assert "does not support output" in renderers
    assert "not yet supported" not in renderers
    assert "does not implement the requested OpenCV lens-distortion model" in renderers
    assert "Using the first" not in renderers
    assert 'else ["rgb"]' not in renderers
    assert "data_types if data_types else" not in ovrtx_usd
    assert "Warp renderer only supports data types" not in tasks


def test_visualization_markers_do_not_walk_usd_stages() -> None:
    """Marker backends consume cfg assets without walking a completed scene stage."""
    facade = (_SOURCE_ROOT / "isaaclab/isaaclab/markers/visualization_markers.py").read_text(encoding="utf-8")
    kit = (_SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualization_markers.py").read_text(
        encoding="utf-8"
    )
    newton = (
        _SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers/newton/newton_visualization_markers.py"
    ).read_text(encoding="utf-8")

    assert "get_current_stage" not in facade + kit + newton
    assert "get_next_free_prim_path" not in facade + kit + newton
    assert "GetPrimAtPath" not in kit
    assert "stage.Traverse()" not in newton
    assert "Usd.Stage.Open" not in newton
    assert ".cpu().numpy()" not in newton
    assert "wp.from_torch" in newton
    assert 'renderer="none"' not in newton
    assert "marker will not be rendered" not in newton
    assert "_warned_marker_render_failure" not in (
        _SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers/viser/viser_visualizer.py"
    ).read_text(encoding="utf-8")


def test_visualizers_do_not_hide_requested_rendering_failures() -> None:
    """Requested visualization modes fail explicitly instead of selecting another path."""
    kit = (_SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualizer.py").read_text(encoding="utf-8")
    kit_cfg = (_SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualizer_cfg.py").read_text(
        encoding="utf-8"
    )
    viser = (_SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers/viser/viser_visualizer.py").read_text(
        encoding="utf-8"
    )
    simulation_context = (_SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_context.py").read_text(encoding="utf-8")
    kit_functions = {
        node.name: node
        for node in ast.walk(ast.parse(kit))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    viser_functions = {
        node.name: node
        for node in ast.walk(ast.parse(viser))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "Streaming view skipped" not in kit
    assert "falls back to plain colour" not in (
        _SOURCE_ROOT / "isaaclab/isaaclab/visualizers/streaming_view.py"
    ).read_text(encoding="utf-8")
    assert "streaming_view=True requires an explicit streaming_camera" in (
        _SOURCE_ROOT / "isaaclab/isaaclab/visualizers/streaming_view.py"
    ).read_text(encoding="utf-8")
    assert "next(iter(self.scene_cameras.values()))" not in (
        _SOURCE_ROOT / "isaaclab/isaaclab/visualizers/streaming_view.py"
    ).read_text(encoding="utf-8")
    assert "falling back to CPU upload" not in kit
    assert "_warned_gpu_upload_failure" not in kit
    assert "defaulting to world" not in kit
    assert "App update skipped" not in kit
    assert ".get(dock_position_name" not in kit
    assert "silently\n        skipped" not in kit_cfg
    for function_name in ("step", "is_training_paused"):
        assert not any(isinstance(node, ast.ExceptHandler) for node in ast.walk(kit_functions[function_name]))
    for function_name in (
        "_dock_image_window_async",
        "_dock_viewport_async",
        "_update_asset_tracking_camera",
        "_apply_viewer_origin_to_camera",
    ):
        assert not any(isinstance(node, ast.Return) for node in ast.walk(kit_functions[function_name]))
    assert "suppress(" not in (ast.get_source_segment(kit, kit_functions["render_rgb_array"]) or "")
    assert not any(isinstance(node, ast.Try) for node in ast.walk(kit_functions["_upload_camera_image_to_panel"]))
    for function_name in ("_push_streaming_frame", "_try_apply_viser_camera_view"):
        assert not any(isinstance(node, ast.Try) for node in ast.walk(viser_functions[function_name]))
    render = next(
        node
        for node in ast.walk(ast.parse(simulation_context))
        if isinstance(node, ast.FunctionDef) and node.name == "render"
    )
    assert not any(isinstance(node, ast.Try) for node in ast.walk(render))
    assert not (_SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/renderers/kit_viewport_utils.py").exists()


def test_visualizer_runtime_has_no_broad_exception_catches() -> None:
    """Visualizer runtime failures reach the lifecycle owner instead of being swallowed."""
    root = _SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for handler in (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)):
            names = (
                {"bare"}
                if handler.type is None
                else {
                    node.id if isinstance(node, ast.Name) else node.attr
                    for node in ast.walk(handler.type)
                    if isinstance(node, (ast.Name, ast.Attribute))
                }
            )
            if names & {"bare", "Exception", "BaseException"}:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{handler.lineno}")

    assert not offenders, "Visualizer runtime swallows broad exceptions:\n" + "\n".join(offenders)


def test_cfgs_have_no_lifecycle_factories() -> None:
    """Configs select a class; neither definitions nor callers put lifecycle on cfgs."""
    offenders = []
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        if "test" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        lines = source.splitlines()
        for cfg_node in (
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name.endswith("Cfg")
        ):
            for member in cfg_node.body:
                if (
                    isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and member.name in _CFG_LIFECYCLE_METHODS
                ):
                    offenders.append(
                        f"{path.relative_to(_REPO_ROOT)}:{member.lineno}: {lines[member.lineno - 1].strip()}"
                    )
        for node in ast.walk(tree):
            calls_factory = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _CFG_LIFECYCLE_METHODS
                and _is_cfg_expression(node.func.value)
            )
            if calls_factory:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert not offenders, "Configs own lifecycle factories:\n" + "\n".join(offenders)


def test_teleop_pipeline_has_one_data_cfg_construction_path() -> None:
    """Teleop pipelines are selected by data cfg and constructed exactly once from it."""
    cfg_path = _SOURCE_ROOT / "isaaclab_teleop/isaaclab_teleop/isaac_teleop_cfg.py"
    lifecycle_path = _SOURCE_ROOT / "isaaclab_teleop/isaaclab_teleop/session_lifecycle.py"
    cfg_source = cfg_path.read_text(encoding="utf-8")
    lifecycle_source = lifecycle_path.read_text(encoding="utf-8")
    teleop_source = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (
            _SOURCE_ROOT / "isaaclab_teleop/isaaclab_teleop",
            _SOURCE_ROOT / "isaaclab_tasks/isaaclab_tasks",
        )
        for path in root.rglob("*.py")
    )
    pipeline_cfg = next(
        node
        for node in ast.parse(cfg_source, filename=str(cfg_path)).body
        if isinstance(node, ast.ClassDef) and node.name == "TeleopPipelineCfg"
    )

    assert [node.target.id for node in pipeline_cfg.body if isinstance(node, ast.AnnAssign)] == ["class_type"]
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in pipeline_cfg.body)
    assert lifecycle_source.count("pipeline_cfg.class_type(pipeline_cfg)") == 1
    assert "pipeline_builder" not in teleop_source
    assert "retargeters_to_tune" not in teleop_source

    offenders = []
    task_root = _SOURCE_ROOT / "isaaclab_tasks/isaaclab_tasks"
    for path in task_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if isinstance(node, ast.FunctionDef) and "pipeline" in node.name:
                if any(isinstance(member, ast.Return) and isinstance(member.value, ast.Tuple) for member in node.body):
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: tuple-return pipeline")
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "IsaacTeleopCfg":
                if not any(keyword.arg == "pipeline_cfg" for keyword in node.keywords):
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: missing pipeline_cfg")

    assert not offenders, "Teleop tasks bypass the data-cfg pipeline contract:\n" + "\n".join(offenders)


def test_camera_has_one_explicit_renderer_config_channel() -> None:
    """Cameras require a renderer cfg; no default factory or tiled compatibility API remains."""
    camera_root = _SOURCE_ROOT / "isaaclab/isaaclab/sensors/camera"
    camera_cfg = (camera_root / "camera_cfg.py").read_text(encoding="utf-8")
    backend_utils = (_SOURCE_ROOT / "isaaclab/isaaclab/utils/backend_utils.py").read_text(encoding="utf-8")
    task_presets = (_SOURCE_ROOT / "isaaclab_tasks/isaaclab_tasks/utils/presets.py").read_text(encoding="utf-8")
    forbidden_fields = {
        "_DEPRECATED_RENDERER_FIELD_DEFAULTS",
        "colorize_instance_id_segmentation",
        "colorize_instance_segmentation",
        "colorize_semantic_segmentation",
        "depth_clipping_behavior",
        "semantic_filter",
        "semantic_segmentation_mapping",
    }

    assert "renderer_cfg: RendererCfg = MISSING" in camera_cfg
    assert "def __post_init__" not in camera_cfg
    assert "get_default_renderer_cfg" not in camera_cfg + backend_utils
    assert "set_isaac_rtx_global_settings" not in task_presets
    assert not (camera_root / "tiled_camera.py").exists()
    assert not (camera_root / "tiled_camera_cfg.py").exists()
    assert not (camera_root / "camera_isp.py").exists()
    assert not {field for field in forbidden_fields if field in camera_cfg}

    offenders = []
    missing_renderer = []
    scan_roots = (_SOURCE_ROOT, _REPO_ROOT / "scripts")
    paths = (path for root in scan_roots for suffix in ("*.py", "*.pyi") for path in root.rglob(suffix))
    for path in sorted(paths):
        if "test" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        names.update(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))
        if names.intersection({"CameraISPMode", "TiledCamera", "TiledCameraCfg"}):
            offenders.append(str(path.relative_to(_REPO_ROOT)))
        for node in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if name == "CameraCfg" and not any(keyword.arg == "renderer_cfg" for keyword in node.keywords):
                missing_renderer.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
    assert not offenders, "Production retains tiled camera compatibility symbols:\n" + "\n".join(offenders)
    assert not missing_renderer, "Camera declarations omit renderer_cfg:\n" + "\n".join(missing_renderer)


def test_nested_camera_uses_the_shared_sensor_lifecycle() -> None:
    """Composite sensors plan and construct cameras before cloning, then use normal callbacks."""
    tactile_root = _SOURCE_ROOT / "isaaclab_contrib/isaaclab_contrib/sensors/tacsl_sensor"
    implementation = (tactile_root / "visuotactile_sensor.py").read_text(encoding="utf-8")
    configuration = (tactile_root / "visuotactile_sensor_cfg.py").read_text(encoding="utf-8")

    assert "enable_camera_tactile" not in implementation + configuration
    assert "camera_cfg.class_type(camera_cfg)" in implementation
    assert "Camera(self.cfg.camera_cfg)" not in implementation
    assert "_camera_sensor._initialize_impl" not in implementation
    assert "_camera_sensor._is_initialized" not in implementation
    assert 'self.cfg.prim_path.rsplit("/", 1)' not in implementation


def test_simulation_context_does_not_synthesize_visualizer_cfgs() -> None:
    """Visualizer instances come only from concrete configs declared on ``SimulationCfg``."""
    source = (_SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_context.py").read_text(encoding="utf-8")
    forbidden = {
        "/isaaclab/visualizer/types",
        "/isaaclab/visualizer/explicit",
        "/isaaclab/visualizer/disable_all",
        "/isaaclab/visualizer/max_visible_envs",
        "/isaaclab/xr/auto_start",
        "_create_default_visualizer_configs",
        "_apply_default_visualizer_cfg",
        "_get_cli_visualizer_types",
        "_apply_visualizer_cli_overrides",
        "_is_cli_visualizer_explicit",
        "_is_cli_visualizer_disable_all",
        "resolve_visualizer_cfgs",
        "resolve_visualizer_types",
        "update_visualizers",
    }
    offenders = sorted(value for value in forbidden if value in source)

    assert not offenders, "SimulationContext still owns visualizer selection: " + ", ".join(offenders)


def test_visualizers_cannot_declare_scene_cameras() -> None:
    """Camera sensors belong to scene cfgs and the clone plan, never visualizer cfgs."""
    sources = {
        "InteractiveScene": (_SOURCE_ROOT / "isaaclab/isaaclab/scene/interactive_scene.py").read_text(encoding="utf-8"),
        "VisualizerCfg": (_SOURCE_ROOT / "isaaclab/isaaclab/visualizers/visualizer_cfg.py").read_text(encoding="utf-8"),
        "StreamingView": (_SOURCE_ROOT / "isaaclab/isaaclab/visualizers/streaming_view.py").read_text(encoding="utf-8"),
    }
    forbidden = {
        "InteractiveScene": {"_declare_visualizer_cameras", "visualizer_camera("},
        "VisualizerCfg": {"CameraCfg", "streaming_follow"},
        "StreamingView": {"StreamingCameraCfg", "owned_camera", "streaming_follow"},
    }
    offenders = [
        f"{owner}: {symbol}" for owner, symbols in forbidden.items() for symbol in symbols if symbol in sources[owner]
    ]

    assert not offenders, "Visualizers still own scene cameras: " + ", ".join(sorted(offenders))


def test_camera_registry_is_owned_by_simulation_context_not_sdp_or_scene() -> None:
    """Direct cfg and InteractiveScene cameras use one composition-root registry."""
    provider_source = (_SOURCE_ROOT / "isaaclab/isaaclab/scene_data/scene_data_provider.py").read_text(encoding="utf-8")
    context_source = (_SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_context.py").read_text(encoding="utf-8")
    scene_source = (_SOURCE_ROOT / "isaaclab/isaaclab/scene/interactive_scene.py").read_text(encoding="utf-8")
    sensor_source = (_SOURCE_ROOT / "isaaclab/isaaclab/sensors/sensor_base.py").read_text(encoding="utf-8")
    camera_source = (_SOURCE_ROOT / "isaaclab/isaaclab/sensors/camera/camera.py").read_text(encoding="utf-8")
    forbidden = {"_interactive_scene", "register_interactive_scene", "set_interactive_scene", "get_interactive_scene"}
    offenders = sorted(symbol for symbol in forbidden if symbol in provider_source + context_source + scene_source)

    assert all(
        symbol not in provider_source
        for symbol in ("register_sensor", "register_camera_sensor", "get_camera_sensors", "_sensors", "def num_envs(")
    )
    assert "def register_camera_sensor(" in context_source
    assert "def get_camera_sensors(" in context_source
    assert "register_camera_sensor(self)" in camera_source
    assert "_physics_manager._sim" not in camera_source
    assert "SimulationContext.instance().get_clone_plan()" not in camera_source
    assert "self._view.initialize(self._clone_plan, self._scene_data_provider)" in camera_source
    assert "get_scene_data_provider().register_sensor(self)" not in sensor_source
    assert not offenders, "Sensor discovery still depends on InteractiveScene: " + ", ".join(offenders)


def test_visualizer_initialization_has_no_backend_specific_lifecycle() -> None:
    """Visualizers initialize only from the backend-neutral physics-ready event."""
    context_source = (_SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_context.py").read_text(encoding="utf-8")
    newton_source = (_SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/physics/newton_manager.py").read_text(
        encoding="utf-8"
    )
    forbidden = {
        "_initialize_visualizers": context_source,
        "config_filter": context_source,
        "_prepare_newton_visualizer_for_capture": context_source + newton_source,
        "_requires_pre_capture_newton_init": context_source + newton_source,
    }
    offenders = sorted(name for name, source in forbidden.items() if name in source)

    assert not offenders, "Backend-specific visualizer lifecycle remains: " + ", ".join(offenders)


def test_visualizers_initialize_from_the_exact_shared_provider_and_clone_plan() -> None:
    """Every visualizer receives data and topology once from the composition root."""
    paths = (
        _SOURCE_ROOT / "isaaclab/isaaclab/visualizers/base_visualizer.py",
        *(_SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers").rglob("*.py"),
    )
    initialize_methods = []
    num_env_accesses = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        initialize_methods.extend(
            (path, node) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "initialize"
        )
        num_env_accesses.extend(
            f"{path.relative_to(_REPO_ROOT)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "num_envs"
        )

    assert initialize_methods
    assert not num_env_accesses, "Visualizers derive topology outside ClonePlan:\n" + "\n".join(num_env_accesses)
    assert all(
        [argument.arg for argument in method.args.args] == ["self", "scene_data_provider", "clone_plan"]
        for _path, method in initialize_methods
    )

    context_path = _SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_context.py"
    context = ast.parse(context_path.read_text(encoding="utf-8"), filename=str(context_path))
    calls = [
        node
        for node in ast.walk(context)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "visualizer"
        and node.func.attr == "initialize"
    ]
    assert [ast.unparse(argument) for call in calls for argument in call.args] == [
        "self._scene_data_provider",
        "self._clone_plan",
    ]

    provider = (_SOURCE_ROOT / "isaaclab/isaaclab/scene_data/scene_data_provider.py").read_text(encoding="utf-8")
    assert "def num_envs(" not in provider


def test_kit_visualizer_uses_the_plan_and_registries_without_stage_discovery() -> None:
    """Kit resolves cameras, tracked assets, and markers from their declared owners."""
    visualizer_source = (_SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualizer.py").read_text(
        encoding="utf-8"
    )
    marker_source = (
        _SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualization_markers.py"
    ).read_text(encoding="utf-8")
    forbidden = {
        "scene_data_provider.usd_stage",
        "GetPrimAtPath",
        "Usd.PrimRange",
        '"/Visuals"',
        '"/World/Visuals"',
        "_set_usd_camera_pose",
        "_apply_visual_point_instancer_visibility",
        "asset.data",
        "isaaclab_physx",
    }
    offenders = sorted(symbol for symbol in forbidden if symbol in visualizer_source)

    assert "UsdGeom.Camera.Define(sim.stage, cfg.prim_path)" in visualizer_source
    assert "id(self.cfg) not in clone_plan.cfg_rows" in visualizer_source
    assert "request_transforms(SceneDataFormat.Vec3_Quat" in visualizer_source
    assert "marker_type = KitVisualizationMarkers" in visualizer_source
    assert "env_prims" not in visualizer_source
    assert "GetVisibilityAttr" not in visualizer_source
    assert "Tokens.invisible" not in visualizer_source
    assert "vis_marker_registry" not in marker_source
    assert not offenders, "Kit retains scene discovery or a direct physics channel: " + ", ".join(offenders)


def test_visualization_markers_are_plan_owned_and_have_no_lazy_backend_selector() -> None:
    """Resolved visualizer cfgs select marker state once, before plan replication."""
    facade_path = _SOURCE_ROOT / "isaaclab/isaaclab/markers/visualization_markers.py"
    registry_source = (_SOURCE_ROOT / "isaaclab/isaaclab/markers/vis_marker_registry.py").read_text(encoding="utf-8")
    session_source = (_SOURCE_ROOT / "isaaclab/isaaclab/cloner/replicate_session.py").read_text(encoding="utf-8")
    facade_source = facade_path.read_text(encoding="utf-8")
    forbidden = {
        "_ensure_backends_initialized",
        "_ensure_kit_backend",
        "_ensure_newton_backend",
        "isaaclab_visualizers",
        "is_rendering",
        "has_offscreen_render",
        ".visualizers",
    }

    assert "sim.vis_marker_registry.get(cfg)" in facade_source
    assert "visualizer.cfg.enable_markers and visualizer.marker_type is not None" in session_source
    assert "sim.vis_marker_registry.prepare(marker_cfgs, marker_types)" in session_source
    assert "rows = self._plan.cfg_rows.get(id(cfg))" in session_source
    assert "if rows is None:" in session_source
    assert "marker_type(cfg)" in registry_source
    assert "not covered by the clone plan" in registry_source
    assert not sorted(symbol for symbol in forbidden if symbol in facade_source)


def test_production_marker_owners_construct_their_exact_cfg() -> None:
    """No production owner may bypass ``VisualizationMarkersCfg.class_type``."""
    offenders = []
    for root in (_SOURCE_ROOT, _REPO_ROOT / "scripts"):
        for path in sorted(root.rglob("*.py")):
            if "test" in path.parts or path == _SOURCE_ROOT / "isaaclab/isaaclab/markers/visualization_markers.py":
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            lines = source.splitlines()
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "VisualizationMarkers"
                ):
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert not offenders, "Marker owners bypass their cfg class_type:\n" + "\n".join(offenders)

    registry = (_SOURCE_ROOT / "isaaclab/isaaclab/markers/vis_marker_registry.py").read_text(encoding="utf-8")
    assert "id(cfg)" not in registry


def test_environment_composition_roots_own_the_only_clone_session() -> None:
    """InteractiveScene owns cloning; environment wrappers only construct their configured scene."""
    scene_source = (_SOURCE_ROOT / "isaaclab/isaaclab/scene/interactive_scene.py").read_text(encoding="utf-8")
    roots = (
        "isaaclab/isaaclab/envs/direct_rl_env.py",
        "isaaclab/isaaclab/envs/direct_marl_env.py",
        "isaaclab/isaaclab/envs/manager_based_env.py",
        "isaaclab/isaaclab/envs/leapp_deployment_env.py",
        "isaaclab_experimental/isaaclab_experimental/envs/direct_rl_env_warp.py",
        "isaaclab_experimental/isaaclab_experimental/envs/manager_based_env_warp.py",
    )

    assert "with cloner.ReplicateSession(" in scene_source
    for relative in roots:
        source = (_SOURCE_ROOT / relative).read_text(encoding="utf-8")
        assert "ReplicateSession(" not in source
        assert ".class_type(" in source
        assert "clone_cfg.resource_key" not in source


def test_production_scripts_do_not_use_raw_grid_cloner() -> None:
    """Production scripts compose replication through the core clone lifecycle."""
    offenders = []
    for path in sorted((_REPO_ROOT / "scripts").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "GridCloner" in source or "isaacsim.core.cloner" in source:
            offenders.append(str(path.relative_to(_REPO_ROOT)))

    assert not offenders, "Production scripts bypass ReplicateSession: " + ", ".join(offenders)


def test_terrain_importer_cfg_does_not_own_clone_layout() -> None:
    """Terrain origins consume the root plan instead of declaring another grid."""
    offenders = []
    for root in (_REPO_ROOT / "scripts", _SOURCE_ROOT):
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                terrain_cfg = (
                    isinstance(call.func, ast.Name)
                    and call.func.id == "TerrainImporterCfg"
                    or isinstance(call.func, ast.Attribute)
                    and call.func.attr == "TerrainImporterCfg"
                )
                if terrain_cfg:
                    for keyword in call.keywords:
                        if keyword.arg in {"num_envs", "env_spacing"}:
                            offenders.append(f"{path.relative_to(_REPO_ROOT)}:{call.lineno}: {keyword.arg}")

    assert not offenders, "TerrainImporterCfg still duplicates clone layout:\n" + "\n".join(offenders)


def test_simulation_context_does_not_synthesize_clone_resources() -> None:
    """Clone resources are registered by the consumers selected from concrete cfgs."""
    source = (_SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_context.py").read_text(encoding="utf-8")
    assert "UsdReplicateContext" not in source


def test_fabric_is_owned_by_the_usd_clone_context_and_prepared_by_consumers() -> None:
    """FSD is bound vectorially from the plan and is absent from SimulationContext and SDP."""
    simulation = (_SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_context.py").read_text(encoding="utf-8")
    provider = (_SOURCE_ROOT / "isaaclab/isaaclab/scene_data/scene_data_provider.py").read_text(encoding="utf-8")
    usd = (_SOURCE_ROOT / "isaaclab/isaaclab/cloner/usd.py").read_text(encoding="utf-8")
    camera = (_SOURCE_ROOT / "isaaclab/isaaclab/sensors/camera/camera.py").read_text(encoding="utf-8")
    base_view = (_SOURCE_ROOT / "isaaclab/isaaclab/sim/views/base_frame_view.py").read_text(encoding="utf-8")
    view_factory = (_SOURCE_ROOT / "isaaclab/isaaclab/sim/views/frame_view.py").read_text(encoding="utf-8")
    frame_view = (_SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/sim/views/physx_frame_view.py").read_text(
        encoding="utf-8"
    )
    site_views = (frame_view,) + tuple(
        (_SOURCE_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in (
            "isaaclab_newton/isaaclab_newton/sim/views/newton_site_frame_view.py",
            "isaaclab_ov/isaaclab_ov/sim/views/ovphysx_frame_view.py",
        )
    )
    mimic = (_SOURCE_ROOT / "isaaclab_mimic/isaaclab_mimic/locomanipulation_sdg/scene_utils.py").read_text(
        encoding="utf-8"
    )
    benchmark = (_REPO_ROOT / "scripts/benchmarks/benchmark_xform_prim_view.py").read_text(encoding="utf-8")
    consumers = (
        "isaaclab_physx/isaaclab_physx/renderers/isaac_rtx_renderer.py",
        "isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualizer.py",
    )

    assert "_prepare_scene_data_fabric" not in simulation
    assert all(symbol not in provider for symbol in ("SelectPrims", "GetPrimAtPath", "usdrt", "hierarchy"))
    assert "def _prepare_fabric(" in usd
    assert "def _prepare_fabric_output(" in usd
    assert "self._prepare_fabric_output" in provider
    assert "plan.iter_rigid_body_paths()" in usd
    assert "plan.point_stream_names" in usd
    assert "plan.point_bindings(name)" in usd
    assert "stage.SynchronizeToFabric()" in usd
    assert 'require_prim_type="Camera"' in usd
    assert "wp.indexedfabricarray" in usd
    assert "update_world_xforms_gpu(True)" not in usd
    assert "update_world_xforms_gpu(not self._fabric_topology_changed)" in usd
    assert 'get("/app/useFabricSceneDelegate", False)' in usd
    tree = ast.parse(usd)
    prepare = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_prepare_fabric"
    )
    prepare_source = ast.get_source_segment(usd, prepare)
    assert "want_paths=True" in prepare_source
    assert 'require_applied_schemas=["PhysicsRigidBodyAPI"]' in prepare_source
    assert all(
        forbidden not in prepare_source
        for forbidden in (
            "GetPrimAtPath",
            "PrimRange",
            "Traverse",
            "CreateAttribute",
            "CreateFabricHierarchyWorldMatrixAttr",
            "CreateFabricHierarchyLocalMatrixAttr",
            "SetLocalXformFromUsd",
            "SetWorldXformFromUsd",
            "AddAppliedSchema",
            "TRANSFORM_INDEX_ATTR",
            "_FRAME_INDEX_ATTR",
            "plan.iter_frames()",
        )
    )
    assert "def create_fabric_visual_material_writer(" in usd
    assert all(
        token not in base_view + view_factory + camera + frame_view + "".join(site_views) + mimic + benchmark
        for token in (
            "_clone_context_type",
            "clone_context",
        )
    )
    assert 'physics_backend == "physx"' not in camera
    assert "SimulationContext.instance" not in view_factory
    assert "return simulation_context.physics_backend" in view_factory
    assert all("get_or_create_backend(" not in source for source in site_views)
    assert camera.index("self._view = FrameView(") < camera.index("self._view.initialize(")
    assert "view.initialize(self.scene.clone_plan" in mimic
    assert benchmark.index("xform_view = view_type(") < benchmark.index("with ReplicateSession(")
    assert benchmark.index("with ReplicateSession(") < benchmark.index("xform_view.initialize(")
    assert "physics_manager._usd_clone_ctx" not in frame_view
    fabric_material = (_SOURCE_ROOT / "isaaclab/isaaclab/renderers/fabric_visual_material.py").read_text(
        encoding="utf-8"
    )
    assert "self._bind_fabric_visual_material" in usd
    assert "want_paths=True" in usd
    assert all(symbol not in fabric_material for symbol in ("GetPrimAtPath", "SelectPrims", "import usdrt"))
    assert "provider._transform_paths" not in usd
    assert all(symbol not in usd for symbol in ("provider._point_streams", "provider._point_bindings"))
    for relative_path in consumers:
        source = (_SOURCE_ROOT / relative_path).read_text(encoding="utf-8")
        assert "._prepare_fabric(" in source
        assert all(symbol not in source for symbol in ("PrepareForReuse", "_clone_ctx._refresh_fabric"))
    for relative_path in consumers:
        source = (_SOURCE_ROOT / relative_path).read_text(encoding="utf-8")
        assert "return self._clone_ctx.create_fabric_visual_material_writer" in source
    assert not (_SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/renderers/visual_material.py").exists()


def test_physx_frame_view_does_not_drive_renderer_storage() -> None:
    """PhysX FrameView consumes native SDP state without owning Fabric or a clone backend."""
    forbidden = {
        "fabric_enabled",
        "omni.physx.fabric",
        "omni.physxfabric",
        "get_physx_fabric_interface",
        "_sync_fabric_after_resume",
        "_re_sync_fabric",
        "fabricUpdateTransformations",
    }
    offenders = []
    paths = [
        *(_SOURCE_ROOT / "isaaclab/isaaclab").rglob("*.py"),
        *(_SOURCE_ROOT / "isaaclab_physx/isaaclab_physx").rglob("*.py"),
        *(_SOURCE_ROOT / "isaaclab_rl/isaaclab_rl").rglob("*.py"),
        *(_SOURCE_ROOT / "isaaclab_tasks/isaaclab_tasks").rglob("*.py"),
        *(_REPO_ROOT / "scripts").rglob("*.py"),
        *(_REPO_ROOT / "apps").glob("*.kit"),
    ]
    for path in sorted(paths):
        source = path.read_text(encoding="utf-8")
        for symbol in forbidden:
            if symbol in source:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}: {symbol}")

    assert not offenders, "Production retains a competing PhysX sync path:\n" + "\n".join(offenders)

    frame_view = (_SOURCE_ROOT / "isaaclab/isaaclab/sim/views/frame_view.py").read_text(encoding="utf-8")
    physx = (_SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/physics/physx_manager.py").read_text(encoding="utf-8")
    physx_view = (_SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/sim/views/physx_frame_view.py").read_text(
        encoding="utf-8"
    )
    physx_exports = (_SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/sim/views/__init__.pyi").read_text(encoding="utf-8")
    assert '"physx": "PhysxFrameView"' in frame_view
    assert "PhysxFrameView" in physx_exports and "FabricFrameView" not in physx_exports
    assert "UsdFrameView" not in frame_view
    assert 'sim.set_setting(f"/physics/{key}", False)' in physx
    assert "request_transforms(SceneDataFormat.Transform)" in physx_view
    assert all(symbol not in physx_view for symbol in ("FabricMatrix44", "UsdReplicateContext", "_prepare_fabric"))
    assert not (_SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/sim/views/fabric_frame_view.py").exists()


def test_fabric_transform_apps_do_not_enable_geometry_streaming() -> None:
    """RTX's mesh-streaming path cannot consume dynamic transforms directly from Fabric reliably."""
    app_root = _REPO_ROOT / "apps"
    offenders = []
    for path in sorted(app_root.glob("*.kit")):
        source = path.read_text(encoding="utf-8")
        if "readTransformsFromFabricInRenderDelegate = true" in source and "UJITSO.geometry = false" not in source:
            offenders.append(str(path.relative_to(_REPO_ROOT)))

    assert not offenders, "Fabric transform apps enable incompatible RTX geometry streaming: " + ", ".join(offenders)


def test_production_has_no_removed_viewer_channel() -> None:
    """The deprecated ViewerCfg path stays removed without rejecting public visualizer defaults."""
    legacy_controller = _SOURCE_ROOT / "isaaclab/isaaclab/envs/ui/viewport_camera_controller.py"
    offenders = [str(legacy_controller.relative_to(_REPO_ROOT))] if legacy_controller.exists() else []
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        if "test" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        lines = source.splitlines()
        for node in ast.walk(tree):
            legacy_attribute = isinstance(node, ast.Attribute) and node.attr == "_apply_deprecated_viewer_cfg"
            legacy_import = (
                isinstance(node, ast.ImportFrom)
                and node.module == "isaaclab.envs"
                and any(alias.name == "ViewerCfg" for alias in node.names)
            )
            if legacy_attribute or legacy_import:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert not offenders, "Production retains a second visualizer configuration channel:\n" + "\n".join(offenders)


def test_clone_session_registers_usd_cloning_by_type() -> None:
    """USD-only cloning requests the one simulation-owned USD backend type."""
    cfg_source = (_SOURCE_ROOT / "isaaclab/isaaclab/cloner/cloner_cfg.py").read_text(encoding="utf-8")
    session_path = _SOURCE_ROOT / "isaaclab/isaaclab/cloner/replicate_session.py"
    session_source = session_path.read_text(encoding="utf-8")
    session_tree = ast.parse(session_source, filename=str(session_path))
    calls = [
        node
        for node in ast.walk(session_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_or_create_backend"
    ]

    assert "resource_key" not in cfg_source
    assert len(calls) == 1
    assert isinstance(calls[0].args[0], ast.Name) and calls[0].args[0].id == "UsdReplicateContext"
    assert "usd_keys" not in session_source


def test_finished_stage_source_resolver_is_deleted() -> None:
    """A public fallback cannot reconstruct clone ownership from the finished USD stage."""
    paths = (
        _SOURCE_ROOT / "isaaclab/isaaclab/sim/utils/queries.py",
        _SOURCE_ROOT / "isaaclab/isaaclab/sim/utils/__init__.pyi",
        _SOURCE_ROOT / "isaaclab/isaaclab/sim/__init__.pyi",
    )

    assert all("resolve_matching_prims_from_source" not in path.read_text(encoding="utf-8") for path in paths)


def test_clone_lifecycle_has_no_fabric_cloning_switch() -> None:
    """The deprecated cfg field is inert and no consumer retains a Fabric cloning policy."""
    offenders = []
    compatibility_cfg = _SOURCE_ROOT / "isaaclab/isaaclab/scene/interactive_scene_cfg.py"
    for root in (_SOURCE_ROOT, _REPO_ROOT / "scripts"):
        for path in sorted(root.rglob("*.py")):
            if "test" in path.parts:
                continue
            if path != compatibility_cfg and "clone_in_fabric" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(_REPO_ROOT)))

    compatibility = compatibility_cfg.read_text(encoding="utf-8")
    assert "clone_in_fabric: bool = False" in compatibility and "Deprecated legacy Fabric cloning flag" in compatibility
    assert not offenders, "Production consumes the deprecated clone_in_fabric field:\n" + "\n".join(offenders)


def test_composition_roots_construct_exactly_from_cfg_class_type() -> None:
    """Physics, renderers, visualizers, and scenes use only ``cfg.class_type(cfg)``."""
    missing = []
    for relative_path, expressions in _EXACT_CFG_CONSTRUCTION_SITES.items():
        path = _SOURCE_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        for expression in expressions:
            expected = ast.parse(expression, mode="eval").body
            found = any(
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "class_type"
                and ast.dump(call.func.value) == ast.dump(expected)
                and len(call.args) == 1
                and ast.dump(call.args[0]) == ast.dump(expected)
                and not call.keywords
                for call in calls
            )
            if not found:
                missing.append(f"{relative_path}: {expression}.class_type({expression})")

    assert not missing, "Composition roots bypass exact cfg construction:\n" + "\n".join(missing)


def test_physics_composition_has_no_implicit_selection() -> None:
    """Physics selection is complete before ``SimulationContext`` construction."""
    context_path = _SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_context.py"
    context_source = context_path.read_text(encoding="utf-8")
    context_tree = ast.parse(context_source, filename=str(context_path))
    context_class = next(
        node for node in context_tree.body if isinstance(node, ast.ClassDef) and node.name == "SimulationContext"
    )
    methods = {
        node.name: node
        for node in context_class.body
        if isinstance(node, ast.FunctionDef) and node.name in {"__new__", "__init__"}
    }
    build_context = next(
        node
        for node in context_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_simulation_context"
    )
    forbidden = {
        "_resolve_physics_cfg",
        "_resolve_physx_auto_cfg",
        "PhysxAutoCfg",
        "isaaclab_newton",
        "isaaclab_ov",
        "isaaclab_physx",
        "cfg: SimulationCfg | None",
        "if sim_cfg is None",
        ".physics.default",
    }

    assert set(methods) == {"__new__", "__init__"}
    assert all(len(method.args.args) == 2 and method.args.args[1].arg == "cfg" for method in methods.values())
    assert all(not method.args.defaults for method in methods.values())
    assert any(isinstance(node, ast.Raise) for node in ast.walk(methods["__new__"]))
    assert not any(
        isinstance(node, ast.Return) and node.value is not None and ast.unparse(node.value) == "cls._instance"
        for node in ast.walk(methods["__new__"])
    )
    assert [argument.arg for argument in build_context.args.args] == ["sim_cfg"]
    shortcuts = {"dt", "gravity_enabled", "add_ground_plane", "add_lighting", "auto_add_lighting"}
    assert not shortcuts.intersection(
        argument.arg for argument in (*build_context.args.args, *build_context.args.kwonlyargs)
    )
    assert not sorted(token for token in forbidden if token in context_source)

    simulation_cfg = (_SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_cfg.py").read_text(encoding="utf-8")
    manager_cfg = (_SOURCE_ROOT / "isaaclab/isaaclab/physics/physics_manager_cfg.py").read_text(encoding="utf-8")
    task_presets = (_SOURCE_ROOT / "isaaclab_tasks/isaaclab_tasks/utils/presets.py").read_text(encoding="utf-8")
    env_cfgs = "\n".join(
        (_SOURCE_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in (
            "isaaclab/isaaclab/envs/direct_rl_env_cfg.py",
            "isaaclab/isaaclab/envs/direct_marl_env_cfg.py",
            "isaaclab/isaaclab/envs/manager_based_env_cfg.py",
        )
    )
    assert "physics: PhysicsCfg = MISSING" in simulation_cfg
    manager_tree = ast.parse(manager_cfg)
    auto_cfg = next(
        node for node in manager_tree.body if isinstance(node, ast.ClassDef) and node.name == "PhysxAutoCfg"
    )
    assert not any(
        isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "class_type"
        for node in auto_cfg.body
    )
    assert "physics: PhysxCfg = PhysxCfg()" in task_presets
    assert env_cfgs.count("sim: SimulationCfg = MISSING") == 3

    newton_cfg_path = _SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/physics/newton_manager_cfg.py"
    newton_cfg_source = newton_cfg_path.read_text(encoding="utf-8")
    newton_cfg_tree = ast.parse(newton_cfg_source, filename=str(newton_cfg_path))
    newton_solver_cfg = next(
        node for node in newton_cfg_tree.body if isinstance(node, ast.ClassDef) and node.name == "NewtonSolverCfg"
    )
    assert not any(isinstance(node, ast.ClassDef) and node.name == "NewtonCfg" for node in newton_cfg_tree.body)
    assert not any(
        isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "solver_cfg"
        for node in newton_solver_cfg.body
    )
    assert not any(isinstance(node, ast.FunctionDef) for node in newton_solver_cfg.body)
    newton_manager = (_SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/physics/newton_manager.py").read_text(
        encoding="utf-8"
    )
    assert "cfg.solver_cfg" not in newton_manager

    physics_backend = next(
        node for node in context_class.body if isinstance(node, ast.FunctionDef) and node.name == "physics_backend"
    )
    assert [ast.unparse(node.value) for node in ast.walk(physics_backend) if isinstance(node, ast.Return)] == [
        "self.cfg.physics.backend"
    ]
    launcher_source = (_SOURCE_ROOT / "isaaclab/isaaclab/app/sim_launcher.py").read_text(encoding="utf-8")
    assert "type(pcfg).__name__" not in launcher_source

    offenders = []
    abstract_newton_cfg_calls = []
    legacy_newton_cfg = []
    for root in (_SOURCE_ROOT, _REPO_ROOT / "scripts", _REPO_ROOT / "tools"):
        for path in sorted(root.rglob("*.py")):
            if "test" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if any(
                (
                    isinstance(node, (ast.Name, ast.ClassDef))
                    and getattr(node, "id", getattr(node, "name", None)) == "NewtonCfg"
                )
                or (isinstance(node, ast.Attribute) and node.attr == "NewtonCfg")
                or (
                    isinstance(node, (ast.Import, ast.ImportFrom))
                    and any(alias.name == "NewtonCfg" for alias in node.names)
                )
                for node in ast.walk(tree)
            ):
                legacy_newton_cfg.append(str(path.relative_to(_REPO_ROOT)))
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                name = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
                if name == "NewtonSolverCfg" and "test" not in path.parts:
                    abstract_newton_cfg_calls.append(f"{path.relative_to(_REPO_ROOT)}:{call.lineno}")
                keywords = {keyword.arg for keyword in call.keywords}
                incomplete = name == "SimulationCfg" and "physics" not in keywords
                incomplete |= name == "SimulationContext" and not call.args and "cfg" not in keywords
                incomplete |= name == "build_simulation_context" and "sim_cfg" not in keywords
                incomplete |= name == "build_simulation_context" and bool(shortcuts.intersection(keywords))
                if incomplete:
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{call.lineno}: {name}")

    assert not offenders, "Physics composition retains implicit cfg selection:\n" + "\n".join(offenders)
    assert not legacy_newton_cfg, "Physics composition retains the NewtonCfg wrapper:\n" + "\n".join(legacy_newton_cfg)
    assert not abstract_newton_cfg_calls, "Production instantiates the abstract NewtonSolverCfg:\n" + "\n".join(
        abstract_newton_cfg_calls
    )

    legacy_docs = []
    for root in (_REPO_ROOT / "docs/source", _REPO_ROOT / "skills"):
        for suffix in ("*.rst", "*.md"):
            for path in root.rglob(suffix):
                if "NewtonCfg" in path.read_text(encoding="utf-8"):
                    legacy_docs.append(str(path.relative_to(_REPO_ROOT)))
    assert not legacy_docs, "Maintained guidance documents the removed NewtonCfg wrapper:\n" + "\n".join(
        sorted(legacy_docs)
    )


def test_simulation_context_owns_renderers_without_a_render_context_layer() -> None:
    """Camera renderer ownership has one composition root and no compatibility wrapper."""
    render_context = _SOURCE_ROOT / "isaaclab/isaaclab/renderers/render_context.py"
    simulation_context = _SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_context.py"
    source = simulation_context.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(simulation_context))
    simulation_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SimulationContext"
    )
    get_renderer = next(
        node for node in simulation_class.body if isinstance(node, ast.FunctionDef) and node.name == "get_renderer"
    )

    assert not render_context.exists()
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "render_context" for node in simulation_class.body
    )
    assert any(
        ast.unparse(node) == "cfg.class_type(cfg)" for node in ast.walk(get_renderer) if isinstance(node, ast.Call)
    )
    assert [ast.unparse(node.value) for node in ast.walk(get_renderer) if isinstance(node, ast.Return)] == ["renderer"]
    source = ast.unparse(get_renderer)
    assert "stored_cfg == cfg" not in source
    assert "type(stored_cfg) is type(cfg)" not in source
    assert "renderer_type" not in source
    assert "global_settings" not in source
    camera_source = (_SOURCE_ROOT / "isaaclab/isaaclab/sensors/camera/camera.py").read_text(encoding="utf-8")
    context_source = simulation_context.read_text(encoding="utf-8")
    assert "_uninitialized_renderers" not in context_source
    assert "._initialize_renderers()" not in camera_source
    assert 'name="initialize_renderers"' in context_source
    renderer_init = next(
        node
        for node in ast.walk(simulation_class)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "register_callback"
        and node.args
        and ast.unparse(node.args[0]) == "self._initialize_renderers"
    )
    assert ast.unparse(renderer_init.args[1]) == "PhysicsEvent.PHYSICS_READY"
    assert next(keyword.value.value for keyword in renderer_init.keywords if keyword.arg == "order") == 5


def test_camera_benchmark_uses_direct_cfg_and_one_clone_lifecycle() -> None:
    """The standalone benchmark exercises the architecture instead of bypassing it."""
    source = (_REPO_ROOT / "scripts/benchmarks/benchmark_cameras.py").read_text(encoding="utf-8")
    assert "class DirectBenchmarkCfg" in source
    assert "with ReplicateSession(" in source
    assert "cfg.camera.class_type(cfg.camera)" in source
    forbidden = {"InteractiveScene", "design_scene", "create_prim(", "RigidObject(", "Camera(", "RayCasterCamera("}
    offenders = sorted(symbol for symbol in forbidden if symbol in source)

    assert not offenders, "Camera benchmark bypasses direct cfg ownership: " + ", ".join(offenders)


def test_frame_view_benchmark_has_one_plan_owned_scene() -> None:
    """The retained FrameView benchmark declares its attachment frame before cloning."""
    benchmark_root = _REPO_ROOT / "scripts/benchmarks"
    source = (benchmark_root / "benchmark_xform_prim_view.py").read_text(encoding="utf-8")
    forbidden = {"InteractiveScene", "get_current_stage", "DefinePrim", "create_prim("}

    assert not (benchmark_root / "benchmark_view_comparison.py").exists()
    assert "ReplicateSession(" in source
    assert "SensorFrameCfg(" in source
    assert not sorted(token for token in forbidden if token in source)


def test_h1_demo_camera_is_declared_in_its_scene_cfg() -> None:
    """The viewport selects a plan-owned robot child and never writes a physics-derived pose."""
    source = (_REPO_ROOT / "scripts/demos/h1_locomotion.py").read_text(encoding="utf-8")

    assert "third_person_camera = AssetBaseCfg(" in source
    assert 'prim_path="{ENV_REGEX_NS}/Robot/torso_link/third_person_camera"' in source
    assert "env_cfg.scene.third_person_camera.prim_path" in source
    assert "cloner.query.destination_paths(" in source
    assert not any(
        token in source
        for token in (
            "ViewportCameraState",
            "_update_camera",
            ".data.root_pos_w",
            ".data.root_quat_w",
            "get_active_camera",
            "get_current_stage",
            "set_position_world",
            "set_target_world",
        )
    )


def test_scene_data_has_no_output_mutating_escape_hatches() -> None:
    """All renderer-facing data movement must be initiated by a format request."""
    forbidden = {"scene_data_formats", "sync_extras", "write_points_to_fabric"}
    offenders = []
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        if "test" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        lines = source.splitlines()
        for node in ast.walk(tree):
            name = node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
            if isinstance(node, ast.Attribute):
                name = node.attr
            if name in forbidden:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert not offenders, "Scene-data output mutation bypasses request APIs:\n" + "\n".join(offenders)


def test_sdp_requests_only_materialize_dirty_publications() -> None:
    """SDP demand may fill a published pointer but never advances or updates physics."""
    backend = (_SOURCE_ROOT / "isaaclab/isaaclab/scene_data/scene_data_backend.py").read_text(encoding="utf-8")
    provider = (_SOURCE_ROOT / "isaaclab/isaaclab/scene_data/scene_data_provider.py").read_text(encoding="utf-8")
    physics = "\n".join((_SOURCE_ROOT / path).read_text(encoding="utf-8") for path in _PHYSICS_MANAGER_FILES)
    assert "def refresh(" not in backend
    assert "self._backend.refresh(" not in provider
    assert "self._backend.publish(" not in provider
    assert provider.count("self._backend._materialize(publication)") == 1
    assert "get_scene_data_provider(" not in physics
    asset_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for package in ("isaaclab_newton", "isaaclab_physx", "isaaclab_ov")
        for path in (_SOURCE_ROOT / package / package / "assets").rglob("*.py")
    )
    assert "_mark_scene_data_dirty" not in asset_sources
    assert "_physics_manager._mark_" not in asset_sources
    assert "_scene_data_backend.publish(" not in asset_sources
    assert "_physics_manager.forward(" not in asset_sources
    assert "_ensure_fk_fresh" not in asset_sources
    assert "_fk_timestamp" not in asset_sources
    assert "update_articulations_kinematic" not in asset_sources

    newton_source = (_SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/physics/newton_manager.py").read_text(
        encoding="utf-8"
    )
    assert "_cable_sync_cpu_buffers" not in newton_source
    newton_tree = ast.parse(newton_source)
    newton_manager = next(
        node for node in ast.walk(newton_tree) if isinstance(node, ast.ClassDef) and node.name == "NewtonManager"
    )
    invalidation_methods = {
        node.name: node
        for node in newton_manager.body
        if isinstance(node, ast.FunctionDef) and node.name in {"invalidate_fk", "invalidate_body_state"}
    }
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("_mark_")
        for method in invalidation_methods.values()
        for node in ast.walk(method)
    )
    newton_backend = next(
        node
        for node in ast.walk(newton_tree)
        if isinstance(node, ast.ClassDef) and node.name == "NewtonSceneDataBackend"
    )
    backend_methods = {node.name: node for node in newton_backend.body if isinstance(node, ast.FunctionDef)}
    assert "publish" not in backend_methods
    assert "_scene_data_backend.publish(" not in ast.unparse(newton_manager)
    setup = backend_methods["setup"]
    assert not any(
        (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "copy")
        or (isinstance(node, ast.Constant) and node.value == "cpu")
        for node in ast.walk(setup)
    )

    impure_getters = []
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        if "test" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            if not any(
                (isinstance(base, ast.Name) and base.id == "SceneDataBackend")
                or (isinstance(base, ast.Attribute) and base.attr == "SceneDataBackend")
                for base in class_node.bases
            ):
                continue
            for getter in (
                node
                for node in class_node.body
                if isinstance(node, ast.FunctionDef)
                and node.name in {"transform_publication", "point_publications"}
                and any(
                    isinstance(decorator, ast.Name) and decorator.id == "property" for decorator in node.decorator_list
                )
            ):
                body = getter.body[1:] if ast.get_docstring(getter, clean=False) is not None else getter.body
                if (
                    len(body) != 1
                    or not isinstance(body[0], ast.Return)
                    or any(
                        isinstance(node, (ast.Call, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom))
                        for node in ast.walk(body[0])
                    )
                ):
                    impure_getters.append(f"{path.relative_to(_REPO_ROOT)}:{getter.lineno}: {getter.name}")

    assert not impure_getters, "Scene-data getters perform deferred work:\n" + "\n".join(impure_getters)
    simulation_context = ast.parse(
        (_SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_context.py").read_text(encoding="utf-8")
    )
    context = next(node for node in simulation_context.body if isinstance(node, ast.ClassDef))
    methods = {node.name: node for node in context.body if isinstance(node, ast.FunctionDef)}
    step_calls = [
        node
        for node in ast.walk(methods["step"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    physics_step = next(
        node.lineno
        for node in step_calls
        if node.func.attr == "step"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "_physics_manager"
    )
    render_call = next(node.lineno for node in step_calls if node.func.attr == "render")
    render_calls = [
        node
        for node in ast.walk(methods["render"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert physics_step < render_call
    assert not any(
        isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "_physics_manager"
        for node in render_calls
    )


def test_direct_task_state_refresh_is_owned_by_environment_lifecycle() -> None:
    """Direct tasks declare derived-state refreshes but never schedule them themselves."""
    offenders = []
    task_root = _SOURCE_ROOT / "isaaclab_tasks/isaaclab_tasks/core"
    for path in sorted(task_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_compute_intermediate_values":
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: legacy refresh method")
            if isinstance(node, ast.FunctionDef) and node.name in {
                "_get_dones",
                "_get_observations",
                "_get_rewards",
            }:
                for assignment in (child for child in ast.walk(node) if isinstance(child, ast.Assign)):
                    writes_task_state = any(
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        for target in assignment.targets
                    )
                    reads_scene_state = any(
                        isinstance(child, ast.Attribute) and child.attr == "data"
                        for child in ast.walk(assignment.value)
                    )
                    if writes_task_state and reads_scene_state:
                        offenders.append(
                            f"{path.relative_to(_REPO_ROOT)}:{assignment.lineno}: getter-owned state refresh"
                        )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_refresh_task_state"
            ):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: task-owned refresh call")

    assert not offenders, "Derived task state bypasses the direct-environment lifecycle:\n" + "\n".join(offenders)


def test_newton_scene_queries_have_one_resource_owned_sdp_path() -> None:
    """Newton physics ownership must not select a second native renderer path."""
    manager = (_SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/physics/newton_manager.py").read_text(encoding="utf-8")
    resource = (_SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/cloner/replicate.py").read_text(encoding="utf-8")
    forbidden_manager = {
        "def _register_sensor_task(",
        "def _update_sensor_tasks(",
        "def _capture_sensor_graph(",
        "_sensor_graph",
    }
    forbidden_delegation = {
        "self._physics_manager._register_sensor_task(",
        "self._physics_manager._update_sensor_tasks(",
        "self._physics_manager._unregister_sensor_task(",
    }

    assert not {symbol for symbol in forbidden_manager if symbol in manager}
    assert not {symbol for symbol in forbidden_delegation if symbol in resource}


def test_newton_resource_never_reaches_back_into_physics_manager() -> None:
    """The shared Newton resource owns cfg-derived state without manager delegation."""
    resource = (_SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/cloner/replicate.py").read_text(encoding="utf-8")
    managers = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((_SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/physics").glob("*manager.py"))
    )
    contrib = "\n".join(
        (_SOURCE_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in (
            "isaaclab_contrib/isaaclab_contrib/coupling/coupler.py",
            "isaaclab_contrib/isaaclab_contrib/custom_coupling/coupled_mjwarp_vbd_manager.py",
        )
    )

    assert "_physics_manager" not in resource
    assert "_register_builder_attributes" not in resource + managers + contrib
    assert "def register_state_force_callback(" not in managers


def test_physics_exposes_current_scene_data_before_ready_consumers_initialize() -> None:
    """Ready callbacks see either bind-once native pointers or a current publication."""
    newton_path = _SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/physics/newton_manager.py"
    newton_tree = ast.parse(newton_path.read_text(encoding="utf-8"), filename=str(newton_path))
    newton_backend = next(
        node for node in newton_tree.body if isinstance(node, ast.ClassDef) and node.name == "NewtonSceneDataBackend"
    )
    newton_manager = next(
        node for node in newton_tree.body if isinstance(node, ast.ClassDef) and node.name == "NewtonManager"
    )
    backend_methods = {node.name: node for node in newton_backend.body if isinstance(node, ast.FunctionDef)}
    manager_methods = {node.name: node for node in newton_manager.body if isinstance(node, ast.FunctionDef)}
    setup_source = ast.unparse(backend_methods["setup"])
    start_source = ast.unparse(manager_methods["start_simulation"])
    reset_source = ast.unparse(manager_methods["reset"])
    initialize_source = ast.unparse(manager_methods["initialize_solver"])
    assert "data.transforms = state.body_q" in setup_source
    assert start_source.index("_state_0 =") < start_source.index("_scene_data_backend.setup")
    assert reset_source.index("start_simulation") < reset_source.index("initialize_solver")
    assert initialize_source.index("PhysicsEvent.PHYSICS_READY") < initialize_source.index("_initialize_contacts")

    ov_path = _SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/physics/ovphysx_manager.py"
    ov_tree = ast.parse(ov_path.read_text(encoding="utf-8"), filename=str(ov_path))
    ov_manager = next(node for node in ov_tree.body if isinstance(node, ast.ClassDef) and node.name == "OvPhysxManager")
    reset_source = ast.unparse(
        next(node for node in ov_manager.body if isinstance(node, ast.FunctionDef) and node.name == "reset")
    )
    assert reset_source.index("_scene_data_backend._invalidate") < reset_source.index("PhysicsEvent.PHYSICS_READY")


def test_visualizers_never_fetch_native_physics_views() -> None:
    """Visualizer dynamics enter only through SDP; raw sensor objects are never handed through."""
    root = _SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers"
    forbidden = {
        "body_physx_view",
        "contact_pos_w",
        "force_matrix_w",
        "get_contact_sensors",
        "get_transforms",
        "net_forces_w",
    }
    offenders = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert not offenders, "Visualizers fetch native physics state:\n" + "\n".join(offenders)
    provider = (_SOURCE_ROOT / "isaaclab/isaaclab/scene_data/scene_data_provider.py").read_text(encoding="utf-8")
    assert "get_contact_sensors" not in provider
    adapter = (_SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers/newton_adapter.py").read_text(encoding="utf-8")
    assert "def apply_viewer_visible_worlds(" not in adapter


def test_renderer_refresh_is_not_keyed_to_the_physics_clock() -> None:
    """SDP dirty generations are authoritative, including multiple writes within one step."""
    paths = (
        _SOURCE_ROOT / "isaaclab/isaaclab/sensors/camera/camera.py",
        _SOURCE_ROOT / "isaaclab/isaaclab/envs/direct_rl_env.py",
        _SOURCE_ROOT / "isaaclab/isaaclab/envs/direct_marl_env.py",
        _SOURCE_ROOT / "isaaclab/isaaclab/envs/manager_based_env.py",
        _SOURCE_ROOT / "isaaclab/isaaclab/envs/manager_based_rl_env.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = {"_last_scene_state_step", "reset_scene_state_cadence", "physics_step_count: int"}
    offenders = sorted(symbol for symbol in forbidden if symbol in source)

    assert not offenders, "Renderer refresh retains physics-clock coupling: " + ", ".join(offenders)


def test_scene_data_does_not_patch_warp_codegen() -> None:
    """Fabric destinations stay plain holders instead of changing Warp's private type system."""
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in (_SOURCE_ROOT / "isaaclab/isaaclab/scene_data").glob("*.py")
    )
    assert "warp._src.codegen" not in sources
    assert "_enable_fabric_arrays_in_structs" not in sources


def test_physx_scene_data_backends_consume_only_the_completed_plan() -> None:
    """Both PhysX packages bind exact plan paths without rediscovering the replicated stage."""
    backends = {
        "isaaclab_physx/isaaclab_physx/physics/physx_manager.py": "PhysxSceneDataBackend",
        "isaaclab_ov/isaaclab_ov/physics/ovphysx_manager.py": "OvPhysxSceneDataBackend",
    }
    forbidden_attributes = {"GetPrimAtPath", "GetPseudoRoot", "PrimRange", "Traverse"}
    forbidden_names = {
        "DeformableStageEntry",
        "discover_deformables_on_stage",
        "get_current_stage",
        "group_deformable_root_paths_for_views",
    }
    offenders = []
    for relative_path, class_name in backends.items():
        path = _SOURCE_ROOT / relative_path
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        backend = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        setup = next(node for node in backend.body if isinstance(node, ast.FunctionDef) and node.name == "setup")
        plan_arg = next((arg for arg in setup.args.args if arg.arg == "plan"), None)
        if plan_arg is None or plan_arg.annotation is None or ast.unparse(plan_arg.annotation) != "ClonePlan":
            offenders.append(f"{relative_path}: setup does not require ClonePlan")
        for node in ast.walk(backend):
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes:
                offenders.append(f"{relative_path}:{node.lineno}: {node.attr}")
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                offenders.append(f"{relative_path}:{node.lineno}: {node.id}")

    assert not offenders, "PhysX scene data rediscovers clone output:\n" + "\n".join(offenders)


def test_physics_publishers_do_not_pack_or_convert_dynamic_geometry() -> None:
    """Every dynamic-geometry kernel belongs to the request-driven SDP conversion."""
    paths = tuple(
        _SOURCE_ROOT / relative
        for relative in (
            "isaaclab_physx/isaaclab_physx/physics/physx_manager.py",
            "isaaclab_ov/isaaclab_ov/physics/ovphysx_manager.py",
            "isaaclab_newton/isaaclab_newton/physics/newton_manager.py",
        )
    )
    forbidden = {
        "_merged_points",
        "_point_packings",
        "pack_body_slices_kernel",
        "_compute_cable_points",
        "_cable_point_buffers",
        "isaaclab.scene_data.geometry_points",
    }
    offenders = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        offenders.extend(f"{path.relative_to(_SOURCE_ROOT)}: {name}" for name in forbidden if name in source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name not in {
                "publish",
                "_mark_transforms_dirty",
                "_mark_particles_dirty",
                "_mark_state_dirty",
            }:
                continue
            dynamic_source = ast.get_source_segment(source, node) or ""
            if "wp.launch" in dynamic_source:
                offenders.append(f"{path.relative_to(_SOURCE_ROOT)}:{node.lineno}: {node.name} launches a kernel")

    assert not offenders, "Physics publishers still convert dynamic geometry:\n" + "\n".join(offenders)


def test_native_point_publications_do_not_duplicate_plan_counts() -> None:
    """Native pointer bundles cannot regain topology already owned by ClonePlan."""
    path = _SOURCE_ROOT / "isaaclab/isaaclab/scene_data/scene_data_backend.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    formats = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SceneDataFormat")
    for class_name in ("BodyPoints", "CablePoints"):
        point_format = next(node for node in formats.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        fields = {
            node.target.id
            for node in point_format.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        assert "count" not in fields


def test_core_assets_and_sensors_do_not_discover_clone_output() -> None:
    """Asset and sensor bases resolve plan-owned paths without a finished-stage search."""
    paths = (
        _SOURCE_ROOT / "isaaclab/isaaclab/assets/asset_base.py",
        _SOURCE_ROOT / "isaaclab/isaaclab/sensors/sensor_base.py",
    )
    forbidden = {
        "find_first_matching_prim",
        "find_matching_prim_paths",
        "find_matching_prims",
        "GetPseudoRoot",
        "PrimRange",
        "resolve_matching_prims_from_source",
        "Traverse",
    }
    offenders = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        lines = source.splitlines()
        path_forbidden = forbidden | ({"GetPrimAtPath"} if path.name == "sensor_base.py" else set())
        for node in ast.walk(tree):
            name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
            if name in path_forbidden:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert not offenders, "Core assets or sensors rediscover clone output:\n" + "\n".join(offenders)
    sensor_tree = ast.parse(paths[1].read_text(encoding="utf-8"))
    sensor_base = next(
        node for node in sensor_tree.body if isinstance(node, ast.ClassDef) and node.name == "SensorBase"
    )
    initialize = next(
        node for node in sensor_base.body if isinstance(node, ast.FunctionDef) and node.name == "_initialize_impl"
    )
    assert not any(isinstance(node, ast.Attribute) and node.attr == "match_frames" for node in ast.walk(initialize))


def test_runtime_scene_owners_take_the_stage_from_simulation_context() -> None:
    """Assets, sensors, scenes, and terrain share their composition root's stage."""
    paths = (
        _SOURCE_ROOT / "isaaclab/isaaclab/assets/asset_base.py",
        _SOURCE_ROOT / "isaaclab/isaaclab/sensors/sensor_base.py",
        _SOURCE_ROOT / "isaaclab/isaaclab/scene/interactive_scene.py",
        _SOURCE_ROOT / "isaaclab/isaaclab/terrains/terrain_importer.py",
    )
    offenders = [str(path.relative_to(_REPO_ROOT)) for path in paths if "get_current_stage" in path.read_text()]

    assert not offenders, "Runtime scene owners fetch a process-global stage:\n" + "\n".join(offenders)


def test_cloner_exposes_one_composed_replication_lifecycle() -> None:
    """The public API exposes plan composition and direct backend operations, but no copy shortcut."""
    cloner_root = _SOURCE_ROOT / "isaaclab/isaaclab/cloner"
    exports = (cloner_root / "__init__.pyi").read_text(encoding="utf-8")
    for public in ("ReplicateSession", "make_clone_plan", "replicate", "usd_replicate"):
        assert f'"{public}"' in exports
    assert "whole_env_copy" not in exports


def test_terrain_layout_is_owned_only_by_the_clone_plan() -> None:
    """Terrain cannot carry or reconstruct a second environment layout."""
    cfg_source = (_SOURCE_ROOT / "isaaclab/isaaclab/terrains/terrain_importer_cfg.py").read_text(encoding="utf-8")
    importer_source = (_SOURCE_ROOT / "isaaclab/isaaclab/terrains/terrain_importer.py").read_text(encoding="utf-8")
    scene_source = (_SOURCE_ROOT / "isaaclab/isaaclab/scene/interactive_scene.py").read_text(encoding="utf-8")

    assert "num_envs:" not in cfg_source
    assert "env_spacing:" not in cfg_source
    assert "plan.positions" in importer_source
    assert "grid_transforms" not in importer_source
    assert "asset_cfg.num_envs" not in scene_source
    assert "asset_cfg.env_spacing" not in scene_source


def test_articulation_ordering_keeps_public_conventions_without_an_orphan_cache() -> None:
    """Public symbolic conventions remain available without duplicate state on articulations."""
    articulation_root = _SOURCE_ROOT / "isaaclab/isaaclab/assets/articulation"
    cfg_source = (articulation_root / "articulation_cfg.py").read_text(encoding="utf-8")
    exports = (articulation_root / "__init__.pyi").read_text(encoding="utf-8")
    base_source = (articulation_root / "base_articulation.py").read_text(encoding="utf-8")

    assert "ArticulationOrderingConvention" in cfg_source
    assert "list[str] | tuple[str, ...] | str | ArticulationOrderingConvention | None" in cfg_source
    for public in (
        "ArticulationOrderingConvention",
        "apply_articulation_ordering_preset",
        "parse_articulation_ordering_convention",
        "get_articulation_name_ordering",
    ):
        assert public in exports
    assert "_ordering_convention_name_cache" not in base_source


def test_action_terms_only_reference_plan_owned_scene_sensors() -> None:
    """Action terms consume named scene sensors instead of constructing and initializing hidden ones."""
    source = (_SOURCE_ROOT / "isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py").read_text(encoding="utf-8")
    forbidden = {
        "ContactSensor(",
        "ContactSensorCfg(",
        "FrameTransformer(",
        "FrameTransformerCfg(",
        "_initialize_impl",
        "resolve_matching_prims_from_source",
    }
    offenders = sorted(symbol for symbol in forbidden if symbol in source)

    assert "env.scene[self.cfg.contact_sensor_name]" in source
    assert "env.scene[self.cfg.task_frame_sensor_name]" in source
    assert not offenders, "Action terms own sensors outside the clone plan: " + ", ".join(offenders)


def test_event_terms_use_plan_metadata_and_explicit_visual_targets() -> None:
    """Runtime events cannot discover physics or visual targets from the finished stage."""
    path = _SOURCE_ROOT / "isaaclab/isaaclab/envs/mdp/events.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name
        in {
            "_RandomizeRigidBodyMaterialOvPhysx",
            "randomize_visual_color",
            "randomize_visual_texture_material",
        }
    }
    forbidden = {
        "GetPrimAtPath",
        "HasAPI",
        "PrimRange",
        "Traverse",
        "compare_versions",
        "event_name",
        "resolve_matching_prims_from_source",
        "send_og_event",
        "trigger",
    }
    offenders = []
    for class_name, class_node in classes.items():
        for node in ast.walk(class_node):
            name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
            if name in forbidden:
                offenders.append(f"{class_name}:{node.lineno}: {name}")

    assert set(classes) == {
        "_RandomizeRigidBodyMaterialOvPhysx",
        "randomize_visual_color",
        "randomize_visual_texture_material",
    }
    ovphysx_source = ast.get_source_segment(source, classes["_RandomizeRigidBodyMaterialOvPhysx"])
    assert "plan.match_articulations" in ovphysx_source
    assert ".bodies" in ovphysx_source
    assert source.count('cfg.params["visual_prim_path"]') == 2
    assert "pattern_with_visuals" not in source
    assert not offenders, "Event terms rediscover runtime targets:\n" + "\n".join(offenders)


def test_backend_assets_and_sensors_do_not_discover_clone_output() -> None:
    """Backend runtime initialization consumes only plan and native-view metadata."""
    relative_paths = (
        "isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py",
        "isaaclab_physx/isaaclab_physx/assets/rigid_object/rigid_object.py",
        "isaaclab_physx/isaaclab_physx/assets/rigid_object_collection/rigid_object_collection.py",
        "isaaclab_physx/isaaclab_physx/assets/deformable_object/deformable_object.py",
        "isaaclab_physx/isaaclab_physx/assets/surface_gripper/surface_gripper.py",
        "isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py",
        "isaaclab_physx/isaaclab_physx/sensors/frame_transformer/frame_transformer.py",
        "isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py",
        "isaaclab_newton/isaaclab_newton/assets/cable_object/cable_object.py",
        "isaaclab_newton/isaaclab_newton/assets/rigid_object/rigid_object.py",
        "isaaclab_newton/isaaclab_newton/assets/rigid_object_collection/rigid_object_collection.py",
        "isaaclab_ov/isaaclab_ov/assets/articulation/articulation.py",
        "isaaclab_ov/isaaclab_ov/assets/rigid_object/rigid_object.py",
        "isaaclab_ov/isaaclab_ov/assets/rigid_object_collection/rigid_object_collection.py",
        "isaaclab_ov/isaaclab_ov/assets/deformable_object/deformable_object.py",
        "isaaclab_ov/isaaclab_ov/sensors/contact_sensor/contact_sensor.py",
        "isaaclab_ov/isaaclab_ov/sensors/frame_transformer/frame_transformer.py",
    )
    forbidden = {
        "GetPrimAtPath",
        "GetPseudoRoot",
        "PrimRange",
        "Traverse",
        "find_first_matching_prim",
        "find_matching_prims",
        "get_all_matching_child_prims",
        "get_current_stage",
        "resolve_matching_prims_from_source",
    }
    offenders = []
    sources = {}
    for relative_path in relative_paths:
        path = _SOURCE_ROOT / relative_path
        source = sources[relative_path] = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        lines = source.splitlines()
        for node in ast.walk(tree):
            name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
            if name in forbidden:
                offenders.append(f"{relative_path}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert not offenders, "Backend assets or sensors rediscover clone output:\n" + "\n".join(offenders)
    rigid_sources = [source for path, source in sources.items() if "/assets/rigid_object" in path]
    assert all(".match_rigid_body(" in source and "match_rigid_body_subtrees" not in source for source in rigid_sources)


def test_newton_actuator_runtime_consumes_only_clone_plan_declarations() -> None:
    """PhysX-family actuator initialization cannot rediscover the finished stage."""
    relative_paths = (
        "isaaclab/isaaclab/actuators/newton/adapter.py",
        "isaaclab/isaaclab/actuators/newton/physx_runtime.py",
        "isaaclab_physx/isaaclab_physx/assets/articulation/actuator_control.py",
        "isaaclab_ov/isaaclab_ov/assets/articulation/actuator_control.py",
    )
    forbidden = {
        "GetPrimAtPath",
        "GetPseudoRoot",
        "PrimRange",
        "find_first_matching_prim",
        "get_current_stage",
    }
    sources = {}
    offenders = []
    for relative_path in relative_paths:
        path = _SOURCE_ROOT / relative_path
        source = sources[relative_path] = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
            if name in forbidden:
                offenders.append(f"{relative_path}:{node.lineno}: {name}")

    adapter = sources[relative_paths[0]]
    controls = sources[relative_paths[2]] + sources[relative_paths[3]]
    layout_compiler = (_SOURCE_ROOT / "isaaclab/isaaclab/cloner/scene_layout.py").read_text(encoding="utf-8")
    newton_root = _SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/actuators"

    assert "def from_layout(" in adapter and "joint.newton_actuator" in adapter
    assert controls.count("layout=articulation._articulation_layout") == 2
    assert "NewtonActuatorLayout(" in layout_compiler and "parse_actuator_prim(prim)" in layout_compiler
    assert not offenders, "Newton actuator runtime rediscovers clone output:\n" + "\n".join(offenders)
    assert not (newton_root / "physx_wrapper.py").exists()


def test_frame_views_consume_only_plan_declared_bindings() -> None:
    """Runtime frame views never rediscover frames, ancestors, or Fabric selections."""
    plan_only = (
        "isaaclab_physx/isaaclab_physx/sim/views/physx_frame_view.py",
        "isaaclab_newton/isaaclab_newton/sim/views/newton_site_frame_view.py",
        "isaaclab_ov/isaaclab_ov/sim/views/ovphysx_frame_view.py",
    )
    forbidden = {
        "GetParent",
        "GetPrimAtPath",
        "PrimRange",
        "SelectPrims",
        "SimulationContext.instance",
        "FabricMatrix44",
        "UsdReplicateContext",
        "_prepare_fabric",
        "cl_register_site",
        "find_matching_prims",
        "get_current_stage",
    }
    offenders = []
    for relative_path in plan_only:
        path = _SOURCE_ROOT / relative_path
        source = path.read_text(encoding="utf-8")
        for symbol in forbidden:
            if symbol in source:
                offenders.append(f"{relative_path}: {symbol}")

    if (_SOURCE_ROOT / "isaaclab/isaaclab/sim/views/xform_prim_view.py").exists():
        offenders.append("isaaclab/isaaclab/sim/views/xform_prim_view.py: rejected compatibility alias")
    if (_SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/sim/views/fabric_frame_view.py").exists():
        offenders.append("isaaclab_physx/isaaclab_physx/sim/views/fabric_frame_view.py: rejected renderer-owned view")
    exports = "\n".join(
        (_SOURCE_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in ("isaaclab/isaaclab/sim/__init__.pyi", "isaaclab/isaaclab/sim/views/__init__.pyi")
    )
    if "XformPrimView" in exports:
        offenders.append("isaaclab.sim exports rejected XformPrimView compatibility alias")
    base_tree = ast.parse((_SOURCE_ROOT / "isaaclab/isaaclab/sim/views/base_frame_view.py").read_text(encoding="utf-8"))
    base = next(node for node in base_tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseFrameView")
    ambiguous = {node.name for node in base.body if isinstance(node, ast.FunctionDef)} & {"get_scales", "set_scales"}
    if ambiguous:
        offenders.append(f"BaseFrameView retains ambiguous scale API: {sorted(ambiguous)}")
    ov_view = (_SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/sim/views/ovphysx_view.py").read_text(encoding="utf-8")
    if "\nOvPhysxViewError =" in ov_view:
        offenders.append("OvPhysxView retains module-level compatibility error alias")

    assert not offenders, "FrameView runtime retains fallback discovery:\n" + "\n".join(sorted(offenders))


def test_newton_actuator_authoring_uses_only_plan_sources() -> None:
    """Articulation construction authors every exact plan source without a cfg hook or stage search."""
    schema = (_SOURCE_ROOT / "isaaclab/isaaclab/sim/schemas/schemas_actuators.py").read_text(encoding="utf-8")
    base = (_SOURCE_ROOT / "isaaclab/isaaclab/assets/articulation/base_articulation.py").read_text(encoding="utf-8")
    forbidden = {"find_first_matching_prim", "find_matching_prims", "get_current_stage", "_post_spawn"}
    offenders = sorted(symbol for symbol in forbidden if symbol in schema + base)

    assert "cloner.query.iter_sources(plan, prim_path)" in schema
    assert "define_actuator_properties(self.cfg.prim_path, self.cfg.actuators, self.stage)" in base
    assert not offenders, "Newton actuator authoring retains a second ownership path: " + ", ".join(offenders)


def test_fabric_destinations_have_one_owner() -> None:
    """The USD clone context owns exact sinks; physics and SDP own no native Fabric handles."""
    provider = (_SOURCE_ROOT / "isaaclab/isaaclab/scene_data/scene_data_provider.py").read_text(encoding="utf-8")
    usd = (_SOURCE_ROOT / "isaaclab/isaaclab/cloner/usd.py").read_text(encoding="utf-8")
    newton = "\n".join(
        (_SOURCE_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in (
            "isaaclab_newton/isaaclab_newton/cloner/replicate.py",
            "isaaclab_newton/isaaclab_newton/cloner/newton_clone_utils.py",
            "isaaclab_newton/isaaclab_newton/physics/newton_manager.py",
        )
    )
    assert "def usd_stage" not in provider
    assert "def usdrt_stage" not in provider
    assert "SelectPrims" not in provider
    assert "GetPrimAtPath" not in provider
    assert "SelectPrims" in usd
    assert "GetPrimAtPath" in usd
    assert "_cl_fabric_body_bindings" not in newton
    assert "_initialize_fabric_body_prims" not in newton
    assert "_initialize_fabric_particle_prims" not in newton


def test_newton_model_initialization_has_no_finished_stage_fallback() -> None:
    """Newton consumes one clone-built model and plan-declared terrain without stage discovery."""
    manager_path = _SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/physics/newton_manager.py"
    manager = manager_path.read_text(encoding="utf-8")
    vbd = (_SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/physics/vbd_manager.py").read_text(encoding="utf-8")
    for forbidden in (
        "instantiate_builder_from_stage",
        "_cl_inject_sites_fallback",
        "get_current_stage",
        "GetPrimAtPath",
        "stage.Traverse",
    ):
        assert forbidden not in manager + vbd

    replicate_path = _SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/cloner/replicate.py"
    tree = ast.parse(replicate_path.read_text(encoding="utf-8"), filename=str(replicate_path))
    terrain_builder = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_build_newton_builder_from_mapping"
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"GetPrimAtPath", "PrimRange", "Traverse"}
        for node in ast.walk(terrain_builder)
    )


def test_mpm_visual_geometry_is_plan_owned() -> None:
    """MPM assets declare one renderer-neutral point prim and never inspect visualizer selection."""
    mpm_root = _SOURCE_ROOT / "isaaclab_newton/isaaclab_newton"
    asset_source = (mpm_root / "assets/mpm_object/mpm_object.py").read_text(encoding="utf-8")
    forbidden = {"resolve_visualizer_cfgs", "visualizer_type", "/World/Visuals/MPMParticles", "_create_kit_points"}
    offenders = sorted(value for value in forbidden if value in asset_source)

    assert not (mpm_root / "sim/spawners/mpm/visualization.py").exists()
    assert not offenders, "MPM runtime still owns visualizer-specific geometry: " + ", ".join(offenders)


def test_deformable_visual_geometry_is_plan_owned() -> None:
    """Newton deformables retain source offsets but publish no destination metadata."""
    source = (_SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/assets/deformable_object/deformable_object.py").read_text(
        encoding="utf-8"
    )
    forbidden = {"setup_registered_deformable_fabric_sync", "_clone_physics_only"}
    offenders = sorted(value for value in forbidden if value in source)

    assert "plan.match_deformable_subtrees" in source
    assert "particle_visual_prims" not in source
    assert not offenders, "Deformable visualization still reconstructs clone output: " + ", ".join(offenders)


def test_scene_data_publications_are_pointer_dirty_only_and_topology_is_plan_owned() -> None:
    """SDP publishes pointer + dirty while the clone plan alone owns topology."""
    backend = (_SOURCE_ROOT / "isaaclab/isaaclab/scene_data/scene_data_backend.py").read_text(encoding="utf-8")
    provider = (_SOURCE_ROOT / "isaaclab/isaaclab/scene_data/scene_data_provider.py").read_text(encoding="utf-8")
    newton = "\n".join(
        (_SOURCE_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in (
            "isaaclab_newton/isaaclab_newton/assets/deformable_object/deformable_object.py",
            "isaaclab_newton/isaaclab_newton/assets/mpm_object/mpm_object.py",
            "isaaclab_newton/isaaclab_newton/cloner/replicate.py",
        )
    )
    clone_plan = (_SOURCE_ROOT / "isaaclab/isaaclab/cloner/clone_plan.py").read_text(encoding="utf-8")

    backend_tree = ast.parse(backend)
    publication = next(
        node for node in backend_tree.body if isinstance(node, ast.ClassDef) and node.name == "SceneDataPublication"
    )
    publication_fields = [
        node.target.id
        for node in publication.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    assert publication_fields == ["data", "dirty"]
    assert all(symbol not in backend for symbol in ("TransformPublication", "PointPublication"))
    assert all(symbol not in backend for symbol in ("transform_count", "transform_paths", "transforms_dirty"))

    provider_functions = {
        node.name: node for node in ast.walk(ast.parse(provider)) if isinstance(node, ast.FunctionDef)
    }
    refresh_source = ast.unparse(provider_functions["_refresh_generation"])
    assert refresh_source.index("_materialize(publication)") < refresh_source.index("generation += 1")
    assert [argument.arg for argument in provider_functions["request_transforms"].args.args] == [
        "self",
        "output_format",
        "name",
    ]
    assert [argument.arg for argument in provider_functions["request_points"].args.args] == [
        "self",
        "output_format",
        "name",
    ]
    assert all(
        symbol not in provider
        for symbol in (
            "create_mapping",
            "create_point_mapping",
            "def _point_bindings",
            "def _point_streams",
            "_transform_cache",
            "_point_cache",
            "convert_Transform_to_Transform",
            "convert_Points_to_Points",
        )
    )
    assert "self._cache" in provider
    assert all(symbol not in newton for symbol in ("particle_visual_prims", "planned_vis_prim_paths"))
    assert "class PointCloudLayout:" in clone_plan
    assert "def point_bindings(" in clone_plan


def test_contrib_runtime_topology_comes_from_the_clone_plan() -> None:
    """Contrib runtime consumes topology declared once from clone-plan prototypes."""
    paths = (
        "isaaclab_newton/isaaclab_newton/assets/deformable_object/deformable_object.py",
        "isaaclab_contrib/isaaclab_contrib/sensors/tacsl_sensor/visuotactile_sensor.py",
    )
    discovery = {
        "GetPrimAtPath",
        "PrimRange",
        "find_first_matching_prim",
        "get_all_matching_child_prims",
        "get_first_matching_child_prim",
        "resolve_matching_prims_from_source",
    }
    offenders = []
    sources = {}
    for relative_path in paths:
        path = _SOURCE_ROOT / relative_path
        source = sources[relative_path] = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for function in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
            uses_discovery = any(
                (isinstance(node, ast.Attribute) and node.attr in discovery)
                or (isinstance(node, ast.Name) and node.id in discovery)
                for node in ast.walk(function)
            )
            if uses_discovery:
                offenders.append(f"{relative_path}:{function.lineno}: {function.name}")

    assert "plan.match_deformable_subtrees" in sources[paths[0]]
    assert "plan.match_rigid_body_subtrees" in sources[paths[1]]
    assert "plan.match_geometry_targets" in sources[paths[1]]
    assert not offenders, "Contrib runtime retains stage discovery:\n" + "\n".join(offenders)


def test_task_geometry_hashing_has_no_unplanned_stage_fallback() -> None:
    """NIST geometry variants come only from the completed clone plan."""
    source = (_SOURCE_ROOT / "isaaclab_tasks/isaaclab_tasks/contrib/nist/utils/rigid_object_hasher.py").read_text(
        encoding="utf-8"
    )

    assert "plan.match_geometry_targets(prim_path_pattern)" in source
    assert "collider_geometries" in source
    assert "GetPrimAtPath" not in source
    assert "Usd.PrimRange" not in source
    assert "get_all_matching_child_prims" not in source
    assert "resolve_matching_prims_from_source" not in source
    assert "get_current_stage" not in source


def test_occupancy_map_does_not_author_post_clone_visualization() -> None:
    """Runtime path planning cannot add an unplanned mesh and material to the stage."""
    utility = (_SOURCE_ROOT / "isaaclab_mimic/isaaclab_mimic/locomanipulation_sdg/occupancy_map_utils.py").read_text(
        encoding="utf-8"
    )
    script = (_REPO_ROOT / "scripts/imitation_learning/locomanipulation_sdg/generate_data.py").read_text(
        encoding="utf-8"
    )

    assert all(symbol not in utility for symbol in ("UsdGeom", "UsdShade", "occupancy_map_add_to_stage"))
    assert all(symbol not in script for symbol in ("draw_visualization", "omni.usd"))


def test_nist_collision_body_selection_comes_from_the_plan() -> None:
    """NIST collision sampling selects named body views from the declared layout."""
    source = (_SOURCE_ROOT / "isaaclab_tasks/isaaclab_tasks/contrib/nist/utils/collision_analyzer.py").read_text(
        encoding="utf-8"
    )

    assert "layout.match_rigid_body_subtrees" in source
    assert "layout.match_articulation" in source
    assert "resolve_matching_prims_from_source" not in source


def test_newton_viewers_draw_the_sdp_ordered_state() -> None:
    """Newton-backed viewers do not bypass the state assembled through scene data."""
    visualizer_root = _SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers"
    paths = (
        visualizer_root / "newton/newton_visualizer.py",
        visualizer_root / "rerun/rerun_visualizer.py",
        visualizer_root / "viser/viser_visualizer.py",
    )
    offenders = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        if "log_state_particles(self, state)" not in source:
            offenders.append(f"{path.relative_to(_REPO_ROOT)}: missing SDP-ordered state")
        for forbidden in ("_point_provider", "request_points(", "super()._log_particles"):
            if forbidden in source:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}: {forbidden}")

    assert not offenders, "Newton-backed viewers retain a second particle channel:\n" + "\n".join(offenders)


def test_newton_render_state_is_requested_through_sdp() -> None:
    """The clone-built state is only a wrapper around SDP transform and point pointers."""
    resource = (_SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/cloner/replicate.py").read_text(encoding="utf-8")
    consumers = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            _SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/renderers/newton_warp_renderer.py",
            _SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers/newton/newton_visualizer.py",
            _SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers/rerun/rerun_visualizer.py",
            _SOURCE_ROOT / "isaaclab_visualizers/isaaclab_visualizers/viser/viser_visualizer.py",
        )
    )

    assert "request_visualization_state" in consumers
    assert "provider.request_transforms(SceneDataFormat.IndexedTransform" in resource
    assert "provider.request_points(SceneDataFormat.Points" in resource
    assert "published_count != self._model.body_count" in resource
    assert "if self._state_0.body_q is not None" not in resource
    assert "_renderer_wants_visual_shapes" not in resource
    assert "visual_shapes_required" not in resource
    assert not any(name in consumers for name in ("get_state_0()", "get_state_1()", "get_contacts()"))


def test_physics_does_not_inspect_visualizer_selection() -> None:
    """Physics publishes state without branching on consumer cfgs or process settings."""
    forbidden = {"resolve_visualizer_cfgs", "resolve_visualizer_types", "visualizer_type", "cameras_enabled"}
    offenders = []
    for relative_path in _PHYSICS_MANAGER_FILES:
        path = _SOURCE_ROOT / relative_path
        source = path.read_text(encoding="utf-8")
        offenders.extend(
            f"{path.relative_to(_REPO_ROOT)}: {symbol}" for symbol in sorted(forbidden) if symbol in source
        )

    assert not offenders, "Physics managers inspect renderer or visualizer selection:\n" + "\n".join(offenders)


def test_isaacsim_physx_publishes_at_physics_boundaries() -> None:
    """PhysX publishes after stepping and explicit forwarding, never from render consumers."""
    path = _SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/physics/physx_manager.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    backend = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PhysxSceneDataBackend"
    )
    manager = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PhysxManager")
    backend_methods = {node.name: node for node in backend.body if isinstance(node, ast.FunctionDef)}
    methods = {node.name: node for node in manager.body if isinstance(node, ast.FunctionDef)}
    publish_source = ast.unparse(backend_methods["publish"])
    forward_source = ast.unparse(methods["forward"])
    step_source = ast.unparse(methods["step"])

    assert "update_articulations_kinematic" not in publish_source
    assert forward_source.index("update_articulations_kinematic") < forward_source.index("_scene_data_backend.publish")
    assert "update_articulations_kinematic" not in step_source
    assert "_scene_data_backend.publish(" in step_source
    assert "_scene_data_backend.publish(" in forward_source
    assert ast.unparse(methods["_warmup_and_create_views"]).count("create_simulation_view") == 1
    assert "_view_warp" not in ast.unparse(manager)


def test_physics_has_no_render_cadence_hooks() -> None:
    """Dynamic render consumers pull publications through scene data only."""
    forbidden = {"pre_render", "after_visualizers_render", "video_capture_backend"}
    offenders = []
    for relative_path in _PHYSICS_MANAGER_FILES:
        path = _SOURCE_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in forbidden:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {node.name}")

    production = "\n".join(
        path.read_text(encoding="utf-8") for path in _SOURCE_ROOT.rglob("*.py") if "test" not in path.parts
    )
    if "requires_forward_before_step" in production:
        offenders.append("production: requires_forward_before_step")

    context_path = _SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_context.py"
    context_tree = ast.parse(context_path.read_text(encoding="utf-8"), filename=str(context_path))
    for function in (
        node
        for node in ast.walk(context_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"render", "update_visualizers"}
    ):
        for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
            if (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Attribute)
                and call.func.value.attr == "physics_manager"
            ):
                location = f"{context_path.relative_to(_REPO_ROOT)}:{call.lineno}"
                offenders.append(f"{location}: physics_manager.{call.func.attr}")

    assert not offenders, "Physics retains render-cadence hooks:\n" + "\n".join(offenders)


def test_physics_configuration_does_not_depend_on_render_or_gui_state() -> None:
    """Physics consumes resolved cfg values without renderer- or GUI-dependent overrides."""
    forbidden = {"render_interval", "minFrameRate", "/isaaclab/has_gui"}
    offenders = []
    for relative_path in _PHYSICS_MANAGER_FILES:
        path = _SOURCE_ROOT / relative_path
        source = path.read_text(encoding="utf-8")
        offenders.extend(
            f"{path.relative_to(_REPO_ROOT)}: {symbol}" for symbol in sorted(forbidden) if symbol in source
        )

    assert not offenders, "Physics configuration depends on rendering state:\n" + "\n".join(offenders)


def test_scene_query_support_has_one_cfg_owner_and_no_fallback() -> None:
    """Shared scene-query intent belongs to SimulationCfg and reaches each PhysX backend directly."""
    simulation_cfg_path = _SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_cfg.py"
    physx_cfg_path = _SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/physics/physx_manager_cfg.py"

    def class_fields(path: Path, class_name: str) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        return {
            node.target.id for node in cfg.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }

    assert "enable_scene_query_support" in class_fields(simulation_cfg_path, "SimulationCfg")
    assert "enable_scene_query_support" not in class_fields(physx_cfg_path, "PhysxCfg")

    for relative_path in (
        "isaaclab_physx/isaaclab_physx/physics/physx_manager.py",
        "isaaclab_ov/isaaclab_ov/physics/ovphysx_manager.py",
    ):
        path = _SOURCE_ROOT / relative_path
        source = path.read_text(encoding="utf-8")
        assert ".enable_scene_query_support" in source
        assert 'getattr(sim_cfg, "enable_scene_query_support"' not in source
        assert 'hasattr(sim_cfg, "enable_scene_query_support"' not in source


def test_selected_gpu_ccd_never_silently_changes_physics() -> None:
    """Unsupported selected physics features fail instead of running a different configuration."""
    path = _SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/physics/physx_manager.py"
    source = path.read_text(encoding="utf-8")

    assert "if cfg.enable_ccd and is_gpu:" in source
    assert 'raise ValueError("PhysxCfg.enable_ccd is unsupported with GPU dynamics' in source
    assert '"physxScene:enableCCD", cfg.enable_ccd' in source
    assert "CCD disabled" not in source
    assert "cfg.enable_ccd and not is_gpu" not in source


def test_physics_managers_have_no_class_lifecycle_or_state() -> None:
    """Physics resources and lifecycle state belong to constructed manager instances."""
    offenders = []
    for relative_path in _PHYSICS_MANAGER_FILES:
        path = _SOURCE_ROOT / relative_path
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        lines = source.splitlines()
        for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            if class_node.name not in _PHYSICS_MANAGER_NAMES:
                continue
            for node in class_node.body:
                classmethod = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                    isinstance(decorator, ast.Name) and decorator.id == "classmethod"
                    for decorator in node.decorator_list
                )
                immutable_capability = (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and isinstance(node.annotation, ast.Subscript)
                    and isinstance(node.annotation.value, ast.Name)
                    and node.annotation.value.id == "ClassVar"
                    and isinstance(node.value, (ast.Constant, ast.Tuple))
                ) or (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "supports_anim_recording"
                    and isinstance(node.value, ast.Constant)
                )
                state = (isinstance(node, ast.Assign) and not immutable_capability) or (
                    isinstance(node, ast.AnnAssign) and node.value is not None and not immutable_capability
                )
                if classmethod or state:
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert not offenders, "Physics managers own class lifecycle or state:\n" + "\n".join(offenders)


def test_physics_managers_have_no_redundant_dt_alias() -> None:
    """Consumers use the one backend-neutral physics time-step API."""
    sources = "\n".join((_SOURCE_ROOT / path).read_text(encoding="utf-8") for path in _PHYSICS_MANAGER_FILES)
    assert "def get_dt(" not in sources


def test_physics_lifecycle_has_no_physx_compatibility_bus() -> None:
    """Core listeners use one neutral lifecycle; PhysX dispatches each boundary once."""
    core_sources = "\n".join(
        (_SOURCE_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in (
            "isaaclab/isaaclab/assets/asset_base.py",
            "isaaclab/isaaclab/sensors/sensor_base.py",
        )
    )
    manager_path = _SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/physics/physx_manager.py"
    physx_sources = manager_path.read_text(encoding="utf-8") + (
        _SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/physics/__init__.pyi"
    ).read_text(encoding="utf-8")
    lifecycle_sources = (
        (_SOURCE_ROOT / "isaaclab/isaaclab/physics/physics_manager.py").read_text(encoding="utf-8")
        + core_sources
        + physx_sources
    )
    newton_source = (_SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/physics/newton_manager.py").read_text(
        encoding="utf-8"
    )
    forbidden = {
        "IsaacEvents",
        "_PHYSICS_EVENT_TO_ISAAC_EVENT",
        "_subscribe_to_event",
        "_subscribe_isaac",
        "get_physics_sim_device",
        "store_callback_exception",
        "PRE_PHYSICS_STEP",
        "POST_PHYSICS_STEP",
    }

    assert "isaaclab_physx" not in core_sources
    assert "type(self._physics_manager).__name__" not in core_sources
    assert "def register_callback(" not in newton_source
    assert "wrap_weak_ref: bool" not in lifecycle_sources
    assert not {symbol for symbol in forbidden if symbol in lifecycle_sources}
    assert physx_sources.count("dispatch_event(PhysicsEvent.PHYSICS_READY") == 1
    assert physx_sources.count("dispatch_event(PhysicsEvent.PRIM_DELETION") == 1


def test_newton_manager_composes_the_registry_resource() -> None:
    """The registry object exclusively owns Newton native and clone state."""
    native_fields = {
        "_builder",
        "_cl_fabric_body_bindings",
        "_cl_pending_sites",
        "_cl_protos",
        "_cl_site_index_map",
        "_contacts",
        "_control",
        "_deformable_registry",
        "_model",
        "_mpm_object_registry",
        "_num_envs",
        "_pending_extended_state_attributes",
        "_per_world_builder_hooks",
        "_queue",
        "_sdp_generation",
        "_sensor_bvh_has_collision_shapes",
        "_sensor_state",
        "_sensor_state_dirty",
        "_sensor_tasks",
        "_state_0",
        "_state_1",
        "_transform_data_mapping",
        "_up_axis",
    }
    offenders = []
    for relative_path in _PHYSICS_MANAGER_FILES:
        path = _SOURCE_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        lines = path.read_text(encoding="utf-8").splitlines()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr in native_fields
            ):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    manager_path = _SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/physics/newton_manager.py"
    manager_tree = ast.parse(manager_path.read_text(encoding="utf-8"), filename=str(manager_path))
    manager_class = next(
        node for node in manager_tree.body if isinstance(node, ast.ClassDef) and node.name == "NewtonManager"
    )
    bases = {ast.unparse(base) for base in manager_class.bases}
    if "NewtonReplicateContext" in bases:
        offenders.append(
            f"{manager_path.relative_to(_REPO_ROOT)}:{manager_class.lineno}: inherits NewtonReplicateContext"
        )
    source = manager_path.read_text(encoding="utf-8")
    for forbidden in ("resource is not self", "lambda: self"):
        if forbidden in source:
            offenders.append(f"{manager_path.relative_to(_REPO_ROOT)}: {forbidden}")

    assert not offenders, "Newton managers duplicate or conditionally replace registry state:\n" + "\n".join(offenders)


def test_backend_registry_is_keyed_only_by_backend_type() -> None:
    """Every consumer resolves one simulation-owned resource directly from its backend class."""
    context_path = _SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_context.py"
    context_source = context_path.read_text(encoding="utf-8")
    context_tree = ast.parse(context_source, filename=str(context_path))
    context_class = next(
        node for node in context_tree.body if isinstance(node, ast.ClassDef) and node.name == "SimulationContext"
    )
    registry_method = next(
        node
        for node in context_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_or_create_backend"
    )
    assert not any(
        isinstance(node, ast.Call) and ast.unparse(node.func) == "isinstance" for node in ast.walk(registry_method)
    )
    assert "factory" not in {argument.arg for argument in registry_method.args.args}
    arguments = (
        *registry_method.args.posonlyargs,
        *registry_method.args.args,
        *registry_method.args.kwonlyargs,
    )
    assert "resource_key" not in {argument.arg for argument in arguments}
    annotations = {
        node.target.attr: ast.unparse(node.annotation)
        for node in ast.walk(context_class)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Attribute)
        and isinstance(node.target.value, ast.Name)
        and node.target.value.id == "self"
    }
    assert annotations["_backend_registry"] == "dict[type[object], object]"
    assert annotations["_backend_clone_roles"] == "dict[type[object], set[str]]"
    assert not any(
        isinstance(node, ast.Tuple) and any(ast.unparse(item) == "backend_type" for item in node.elts)
        for node in ast.walk(registry_method)
    )
    constructors = [
        node
        for node in ast.walk(registry_method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "backend_type"
    ]
    assert len(constructors) == 1
    assert not any(
        root in _imported_roots(node)
        for node in ast.walk(context_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for root in _BACKEND_PACKAGES
    )
    assert "ReplicateContext" not in context_source

    for relative_path, class_name in {
        "isaaclab/isaaclab/physics/physics_manager_cfg.py": "PhysicsCfg",
        "isaaclab/isaaclab/renderers/renderer_cfg.py": "RendererCfg",
        "isaaclab/isaaclab/visualizers/visualizer_cfg.py": "VisualizerCfg",
        "isaaclab/isaaclab/cloner/cloner_cfg.py": "CloneCfg",
    }.items():
        path = _SOURCE_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        cfg_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        assert not any(
            isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "resource_key"
            for node in cfg_class.body
        )

    expected_keys = {
        "isaaclab_newton/isaaclab_newton/physics/newton_manager.py": {"NewtonReplicateContext"},
        "isaaclab_newton/isaaclab_newton/renderers/newton_warp_renderer.py": {"NewtonReplicateContext"},
        "isaaclab_visualizers/isaaclab_visualizers/newton/newton_visualizer.py": {"NewtonReplicateContext"},
        "isaaclab_visualizers/isaaclab_visualizers/rerun/rerun_visualizer.py": {"NewtonReplicateContext"},
        "isaaclab_visualizers/isaaclab_visualizers/viser/viser_visualizer.py": {"NewtonReplicateContext"},
        "isaaclab_physx/isaaclab_physx/physics/physx_manager.py": {
            "UsdReplicateContext",
            "PhysxReplicateContext",
        },
        "isaaclab_physx/isaaclab_physx/assets/deformable_object/deformable_object.py": {"UsdReplicateContext"},
        "isaaclab_physx/isaaclab_physx/renderers/isaac_rtx_renderer.py": {
            "UsdReplicateContext",
            "_IsaacRtxRuntime",
        },
        "isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualizer.py": {"UsdReplicateContext"},
        "isaaclab_ov/isaaclab_ov/physics/ovphysx_manager.py": {"OvReplicateContext"},
        "isaaclab_ov/isaaclab_ov/renderers/ovrtx_renderer.py": {"OvReplicateContext"},
    }
    for relative_path, expected in expected_keys.items():
        path = _SOURCE_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_or_create_backend"
        ]
        keys = {ast.unparse(node.args[0]) for node in calls}
        assert keys == expected, f"{relative_path} registers {keys}, expected {expected}."
        for call in calls:
            assert isinstance(call.args[0], ast.Name), f"{relative_path} uses a non-class registry key."
            assert not any(isinstance(node, ast.Lambda) for node in ast.walk(call))
            assert not any(keyword.arg == "resource_key" for keyword in call.keywords)

    offenders = []
    production_paths = (*_SOURCE_ROOT.rglob("*.py"), *(_REPO_ROOT / "scripts").rglob("*.py"))
    for path in sorted(path for path in production_paths if "test" not in path.parts):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for call in ast.walk(tree):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "get_or_create_backend"
            ):
                continue
            if not call.args or not isinstance(call.args[0], ast.Name):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{call.lineno}")
            if any(isinstance(node, ast.Lambda) for node in ast.walk(call)):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{call.lineno}")
            if any(keyword.arg == "resource_key" for keyword in call.keywords):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{call.lineno}")
    assert not offenders, "Backend registry calls use non-class keys or factory lambdas:\n" + "\n".join(offenders)

    resource_key_offenders = []
    for path in sorted(path for path in production_paths if "test" not in path.parts):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "resource_key" in line:
                resource_key_offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not resource_key_offenders, "Production retains resource_key:\n" + "\n".join(resource_key_offenders)

    isaac_rtx = (_SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/renderers/isaac_rtx_renderer.py").read_text(
        encoding="utf-8"
    )
    assert "type(cfg.global_settings)" not in isaac_rtx
    assert "lambda: cfg.global_settings" not in isaac_rtx

    ov_cloner = _SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/cloner"
    assert not (ov_cloner / "ovrtx_replicate.py").exists()
    assert not (ov_cloner.parent / "_clone.py").exists()
    ov_sources = "\n".join(path.read_text(encoding="utf-8") for path in ov_cloner.glob("*.py"))
    legacy = {
        "OvPhysxReplicateContext",
        "OvrtxReplicateContext",
        "PHYSICS_CONTEXT",
        "require_full_stage",
        "register_clone_recipe",
    }
    assert not {symbol for symbol in legacy if symbol in ov_sources}

    replicate_source = (ov_cloner / "replicate.py").read_text(encoding="utf-8")
    exports = (ov_cloner / "__init__.py").read_text(encoding="utf-8")
    assert '"ovphysx_replicate"' in exports
    assert "def ovphysx_replicate(" in replicate_source
    forbidden_probes = {"Traverse", "PrimRange", "GetPrimTypeInfo", "GetAppliedSchemas", "HasAPI"}
    assert "_rows_requiring_authored_copies" not in replicate_source
    assert not {probe for probe in forbidden_probes if probe in replicate_source}
    assert replicate_source.count("ExportToString") == 1


def test_backend_registry_consumers_do_not_clear_shared_resources() -> None:
    """Only SimulationContext teardown may clear values returned by the backend registry."""
    offenders = []
    for path in sorted(path for path in _SOURCE_ROOT.rglob("*.py") if "test" not in path.parts):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        targets = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "get_or_create_backend"
            ):
                continue
            assigned = node.targets if isinstance(node, ast.Assign) else (node.target,)
            targets.update(ast.unparse(target) for target in assigned)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "clear"
                and ast.unparse(node.func.value) in targets
            ):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert not offenders, "Registry consumers clear shared backend resources:\n" + "\n".join(offenders)


def test_production_does_not_use_newton_manager_as_a_singleton() -> None:
    """Consumers hold the SimulationContext's manager instance, never the manager class."""
    offenders = []
    for path in sorted(path for path in _SOURCE_ROOT.rglob("*.py") if "test" not in path.parts):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        lines = source.splitlines()
        for node in ast.walk(tree):
            class_access = (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "NewtonManager"
            )
            stores_class = isinstance(node, (ast.Assign, ast.AnnAssign)) and (
                isinstance(node.value, ast.Name)
                and node.value.id == "NewtonManager"
                or isinstance(node.value, ast.Attribute)
                and node.value.attr == "NewtonManager"
            )
            if class_access or stores_class:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert not offenders, "Production code uses NewtonManager as process-global state:\n" + "\n".join(offenders)


def test_production_does_not_read_or_write_manager_class_state() -> None:
    """Consumers may call stateless/unbound helpers, but never read or write manager class state."""
    offenders = []
    for root in (_SOURCE_ROOT, _REPO_ROOT / "scripts"):
        for path in sorted(root.rglob("*.py")):
            if "test" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            lines = source.splitlines()
            parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in _PHYSICS_MANAGER_NAMES
                ):
                    continue
                parent = parents.get(node)
                if isinstance(parent, ast.Call) and parent.func is node:
                    continue
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert not offenders, "Production reads or writes manager class state:\n" + "\n".join(offenders)


def test_physics_scene_is_authored_once_and_passed_to_managers() -> None:
    """The composition root hands both physics backends its exact authored scene prim."""
    context = (_SOURCE_ROOT / "isaaclab/isaaclab/sim/simulation_context.py").read_text(encoding="utf-8")
    managers = tuple(
        (_SOURCE_ROOT / path).read_text(encoding="utf-8")
        for path in (
            "isaaclab_physx/isaaclab_physx/physics/physx_manager.py",
            "isaaclab_ov/isaaclab_ov/physics/ovphysx_manager.py",
        )
    )

    assert ".stage.Traverse()" not in context
    assert "GetPrimAtPath(cfg.physics_prim_path)" not in context
    assert "self._physics_scene_prim = self._init_usd_physics_scene()" in context
    assert all("GetPrimAtPath" not in manager and "_physics_scene_prim" in manager for manager in managers)


def test_physx_replicator_does_not_fetch_a_physics_scene() -> None:
    """The manager configures its exact scene; replication owns no scene lookup or clone extension."""
    source = (_SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/cloner/replicate.py").read_text(encoding="utf-8")
    assert "GetPrimAtPath" not in source
    assert "/physicsScene" not in source
    assert "isaacsim.core.cloner" not in source


def test_runtime_sensors_do_not_discover_usd() -> None:
    """Runtime sensors consume exact clone-plan facts and never recover them from a stage."""
    roots = (
        _SOURCE_ROOT / "isaaclab/isaaclab/sensors/ray_caster",
        _SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/sensors/ray_caster",
        _SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/sensors/joint_wrench",
        _SOURCE_ROOT / "isaaclab_physx/isaaclab_physx/sensors/pva",
        _SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/sensors/ray_caster",
        _SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/sensors/joint_wrench",
        _SOURCE_ROOT / "isaaclab_ov/isaaclab_ov/sensors/pva",
        _SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/sensors/ray_caster",
        _SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/sensors/joint_wrench",
        _SOURCE_ROOT / "isaaclab_newton/isaaclab_newton/sensors/pva",
    )
    forbidden = (
        "find_matching_prims",
        "find_first_matching_prim",
        "get_all_matching_child_prims",
        "resolve_matching_prims_from_source",
        "Usd.PrimRange",
        "GetPrimAtPath",
        "GetStageUpAxis",
        "get_current_stage",
        "omni.usd",
        ".stage",
        "from pxr",
        "import pxr",
    )
    offenders = [
        f"{path.relative_to(_REPO_ROOT)}: {token}"
        for root in roots
        for path in sorted(root.rglob("*.py"))
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    ]

    assert not offenders, "Runtime sensors retain USD discovery:\n" + "\n".join(offenders)


def test_joint_frame_fixtures_are_authored_before_the_clone_plan() -> None:
    """Joint-frame coverage enters through asset cfgs instead of editing the finished stage."""
    paths = (
        _SOURCE_ROOT / "isaaclab_physx/test/sensors/test_joint_wrench_sensor.py",
        _SOURCE_ROOT / "isaaclab_ov/test/sensors/test_joint_wrench_sensor.py",
    )
    sources = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert sources.count("_non_identity_joint_asset(tmp_path") == len(paths)
    assert "_set_child_joint_frame" not in sources
    assert "scene.stage.Traverse()" not in sources


def test_runtime_does_not_construct_interactive_scenes_directly() -> None:
    """All runtime code exercises the same explicit clone lifecycle."""
    offenders = []
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        if "test" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            if name in {"InteractiveScene", "InteractiveSceneWarp"}:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert not offenders, "Call cfg.class_type(cfg) inside ReplicateSession instead:\n" + "\n".join(offenders)


def test_production_does_not_escape_the_simulation_physics_manager() -> None:
    """Runtime consumers use SimulationContext operations, registry resources, or injected bindings."""
    offenders = []
    for root in (_SOURCE_ROOT, _REPO_ROOT / "scripts"):
        for path in sorted(root.rglob("*.py")):
            if "test" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines()
            for node in ast.walk(ast.parse(source, filename=str(path))):
                if isinstance(node, ast.Attribute) and node.attr == "physics_manager":
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert not offenders, "SimulationContext exposes its physics manager:\n" + "\n".join(offenders)
