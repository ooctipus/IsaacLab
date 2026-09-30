# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One clone-plan snapshot shared by every OV consumer in a simulation."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab import cloner
from isaaclab.sim import SimulationContext

if TYPE_CHECKING:
    import ovstage

    from isaaclab.cloner.clone_plan import ClonePlan
    from isaaclab.renderers.base_renderer import VisualMaterialBatch

    from isaaclab_ov.renderers.visual_materials import OVRTXVisualMaterialWriter

logger = logging.getLogger(__name__)

CloneTransform = tuple[float, float, float, float, float, float, float]
_CloneRow = tuple[str, str, list[str], list[CloneTransform]]


def _clone_transform(matrix: Gf.Matrix4d) -> CloneTransform:
    """Convert a USD pose matrix to an OVPhysX xyzw clone transform."""
    matrix = matrix.RemoveScaleShear()
    position = matrix.ExtractTranslation()
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    return (*map(float, position), *map(float, imaginary), float(quaternion.GetReal()))


def _clone_rows(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: np.ndarray,
    mapping: np.ndarray,
    positions: np.ndarray | None,
    quaternions: np.ndarray | None = None,
) -> list[_CloneRow]:
    """Build OV clone rows from one NumPy source-to-environment mapping."""
    expected_shape = (len(sources), len(env_ids))
    if mapping.shape != expected_shape:
        raise ValueError(f"mapping must have shape {expected_shape}, got {mapping.shape}.")
    if positions is not None and positions.shape != (len(env_ids), 3):
        raise ValueError(f"positions must have shape [num_envs, 3], got {list(positions.shape)}.")
    if quaternions is not None and quaternions.shape != (len(env_ids), 4):
        raise ValueError(f"quaternions must have shape [num_envs, 4], got {list(quaternions.shape)}.")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    rows = []
    for row, (source, destination) in enumerate(zip(sources, destinations, strict=True)):
        if "{}" not in destination:
            continue
        columns = np.flatnonzero(mapping[row])
        if not len(columns):
            continue
        source_prim = stage.GetPrimAtPath(source)
        if not source_prim.IsValid():
            raise ValueError(f"OV clone source prim is not valid on the stage: {source}")
        matched = cloner.path.match(source, destination)
        source_env_id = int(matched.instance) if matched is not None and matched.instance.isdigit() else None
        source_world = xform_cache.GetLocalToWorldTransform(source_prim).RemoveScaleShear()
        if source_env_id is None:
            source_anchor_world = Gf.Matrix4d(1.0)
        else:
            prefix, _ = cloner.path.split(destination)
            source_anchor_path = f"{prefix}{source_env_id}"
            source_anchor = stage.GetPrimAtPath(source_anchor_path)
            if not source_anchor.IsValid():
                raise ValueError(f"OV clone source anchor prim is not valid on the stage: {source_anchor_path}")
            source_anchor_world = xform_cache.GetLocalToWorldTransform(source_anchor).RemoveScaleShear()
        source_relative = source_world * source_anchor_world.GetInverse()

        targets = []
        transforms = []
        for column in columns:
            env_id = int(env_ids[column])
            target = destination.format(env_id)
            if target == source:
                continue
            target_env_world = Gf.Matrix4d(1.0)
            if positions is not None:
                target_env_world.SetTranslateOnly(Gf.Vec3d(*map(float, positions[column])))
            if quaternions is not None:
                quaternion = quaternions[column]
                target_env_world.SetRotateOnly(Gf.Quatd(float(quaternion[3]), Gf.Vec3d(*map(float, quaternion[:3]))))
            targets.append(target)
            transforms.append(_clone_transform(source_relative * target_env_world))
        rows.append((source, destination, targets, transforms))
    return rows


def _expand_for_ovstage(usda: str, rows: Sequence[tuple[str, Sequence[str]]]) -> str:
    """Return ``usda`` with instancing cleared and every clone row copied onto its targets.

    OVStage ignores a material binding that a clone overrides on an instanceable prim, so instancing is
    expanded. Its native clone also leaves a copied prim's internal connections pointing at the source
    environment, which leaves cloned materials without their shaders, so the rows are copied here and
    their intra-subtree paths rebased onto each target.
    """
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(usda)
    pending = [layer.pseudoRoot]
    while pending:
        spec = pending.pop()
        pending.extend(spec.nameChildren)
        if spec.HasInfo("instanceable"):
            spec.ClearInfo("instanceable")
    for source, targets in rows:
        source_path = Sdf.Path(source)
        for target in targets:
            target_path = Sdf.Path(target)
            if not Sdf.CopySpec(layer, source_path, layer, target_path):
                raise RuntimeError(f"Failed to copy {source!r} to {target!r} in the OVStage snapshot.")
            pending = [layer.GetPrimAtPath(target_path)]
            while pending:
                spec = pending.pop()
                pending.extend(spec.nameChildren)
                for prop in (*spec.attributes, *spec.relationships):
                    items = prop.connectionPathList if hasattr(prop, "connectionPathList") else prop.targetPathList
                    rebased = [
                        path.ReplacePrefix(source_path, target_path) if path.HasPrefix(source_path) else path
                        for path in items.explicitItems
                    ]
                    if rebased != list(items.explicitItems):
                        items.explicitItems = rebased
    return layer.ExportToString()


def _whole_env_copy(plan: ClonePlan) -> tuple[str, str] | None:
    """Return a whole-environment native copy only when every replicated row proves it safe."""
    if plan.env_ids is None or not plan.env_ids.size:
        return None
    rows = tuple(
        row
        for row, destination in enumerate(plan.destinations)
        if "{}" in destination and bool(plan.clone_mask[row].any())
    )
    prefixes = {cloner.path.split(plan.destinations[row])[0] for row in rows}
    if not rows or len(prefixes) != 1 or not bool(plan.clone_mask[list(rows)].all()):
        return None
    template = f"{prefixes.pop()}{{}}"
    source_env = template.format(int(plan.env_ids[0]))
    if any(not cloner.path.under(plan.sources[row], source_env) for row in rows):
        return None
    return source_env, template


class OvReplicateContext:
    """Snapshot and replicate one planned USD scene for OVPhysX and OVRTX."""

    replicate_priority = -200
    clones_whole_env = False

    def __init__(self, simulation_context: SimulationContext):
        self._sim = simulation_context
        self.stage = simulation_context.stage
        self._renderers: list[Any] = []
        self._next_renderer_id = 0
        self._ovrtx_key: tuple[int, str, str, str | None] | None = None
        self._ovrtx_scene: Any = None
        self._scene_data_provider: Any = None
        self._object_xforms: Any = None
        self._point_bindings: dict[str, tuple[Any, list[int], list[int]]] = {}
        self._visual_material_writer: OVRTXVisualMaterialWriter | None = None
        self._sdp_transform_generation = -1
        self._sdp_camera_transform_generations: dict[str | None, int] = {}
        self._sdp_point_generations: dict[str, int] = {}
        self._rows: list[_CloneRow] = []
        self._direct_physics_rows: list[tuple[str, tuple[str, ...], tuple[CloneTransform, ...]]] = []
        self._physics_initialized = False
        self._materialized_rows: frozenset[int] = frozenset()
        self._env_prim_paths: list[str] = []
        self._stage_usda: str | None = None
        self._ovstage_requested = False
        self._replicated = False

    def _request_ovstage(self) -> None:
        """Declare a consumer that builds its own rendering ovstage from this context's snapshot.

        The snapshot keeps the environment roots only for consumers that render it, so this must be
        called before the clone plan is replicated.
        """
        if self._replicated:
            raise RuntimeError("An OVStage consumer cannot join a clone context after replication.")
        self._ovstage_requested = True

    def _add_renderer(self, renderer: Any) -> int:
        """Register an OVRTX renderer that consumes this context's snapshot and clone rows."""
        if self._replicated:
            raise RuntimeError("An OVRTX renderer cannot join a clone context after replication.")
        self._renderers.append(renderer)
        renderer_id = self._next_renderer_id
        self._next_renderer_id += 1
        return renderer_id

    def _configure_ovrtx(
        self,
        key: tuple[int, str, str],
        config: Any,
        renderer_type: type,
        scene_type: type,
        temp_usd_dir: str | None,
    ) -> None:
        """Create the one native OVRTX resource, or validate a later camera against it."""
        if self._ovrtx_key is None:
            from isaaclab_ov.renderers.ovrtx_shader_cache import redirect_shader_cache  # noqa: PLC0415

            redirect_shader_cache(config)
            self._ovrtx_key = (*key, temp_usd_dir)
            self._ovrtx_scene = scene_type(renderer_type(config))
        elif self._ovrtx_key != (*key, temp_usd_dir):
            raise ValueError(
                "Every OVRTX camera in one simulation must resolve the same CUDA device, native renderer"
                " configuration, and debug-stage directory."
            )

    @property
    def scene(self) -> Any:
        """The single OVRTX scene shared by every registered camera client."""
        if self._ovrtx_scene is None:
            raise RuntimeError("OVRTX cannot build its scene before a camera supplies the render device.")
        return self._ovrtx_scene

    def replicate(self, plan: ClonePlan) -> None:
        """Create the planned snapshot and populate the shared OVRTX scene once."""
        if self._replicated:
            raise RuntimeError("An OV clone context can replicate exactly once.")
        if self._sim.get_clone_plan() is not plan or not plan.is_complete:
            raise RuntimeError("OV replication requires a completed clone plan.")
        if plan.env_ids is None or plan.positions is None:
            raise ValueError("OV cloning requires environment ids and positions.")
        prefixes = {cloner.path.split(destination)[0] for destination in plan.destinations if "{}" in destination}
        if len(prefixes) > 1:
            raise ValueError(f"OV clone destinations must share one env namespace, got {plan.destinations}.")
        self._rows = _clone_rows(
            self.stage, plan.sources, plan.destinations, plan.env_ids, plan.clone_mask, plan.positions
        )
        if prefixes:
            prefix = prefixes.pop()
            self._env_prim_paths = [f"{prefix}{int(env_id)}" for env_id in plan.env_ids]
        self._replicated = True
        materialize = any(
            "{}" in plan.destinations[entry.row] and bool(plan.clone_mask[entry.row].any())
            for entries in (plan.deformables, plan.cables, plan.point_clouds, plan.surface_grippers)
            for entry in entries
        )
        self._materialized_rows = frozenset(range(len(self._rows))) if materialize else frozenset()
        self._stage_usda = self._snapshot_stage(plan)
        if not self._renderers:
            return
        if (
            self._ovrtx_scene is None
            or self._ovrtx_key is None
            or any(not renderer._render_product_usd for renderer in self._renderers)
        ):
            raise RuntimeError("Every OVRTX renderer must receive its camera specification before replication.")

        combined = self._stage_usda + "\n\n" + "\n".join(renderer._render_product_usd for renderer in self._renderers)
        if self._ovrtx_key[3] is not None:
            output = Path(self._ovrtx_key[3]) / "ovrtx_renderer_stage.usda"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(combined, encoding="utf-8")
            logger.info("Wrote USD file: %s", output)
        scene = self.scene
        scene.open(combined)
        if self._materialized_rows:
            reset_paths = [
                *plan.iter_rigid_body_paths(),
                *(path for renderer in self._renderers for path in renderer._spec.camera_prim_paths),
            ]
        else:
            reset_paths = [
                *(body.source_path or body.path for body in plan.rigid_body_prototypes),
                *(path for renderer in self._renderers for path in renderer._spec.camera_source_prim_paths),
            ]
        if reset_paths:
            scene.write_reset_xform_stack(list(dict.fromkeys(reset_paths)))
        for source, targets in self.clone_rows:
            try:
                scene.clone(source, targets)
            except Exception as exc:
                raise RuntimeError(f"Failed to copy {source} onto {len(targets)} path(s): {exc}") from exc

        partition_tokens = dict(zip(map(Sdf.Path, self._env_prim_paths), self.env_names, strict=True))
        scene.write_tokens(self._env_prim_paths, "primvars:omni:scenePartition", list(partition_tokens.values()))
        camera_partitions = {
            camera_path: partition_tokens[ancestor]
            for renderer in self._renderers
            for camera_path in renderer._spec.camera_prim_paths
            for ancestor in Sdf.Path(camera_path).GetPrefixes()
            if ancestor in partition_tokens
        }
        if camera_partitions:
            scene.write_tokens(list(camera_partitions), "omni:scenePartition", list(camera_partitions.values()))
        for renderer in self._renderers:
            camera_paths = list(renderer._spec.camera_prim_paths)
            scene.point_render_products_at(renderer._render_product_paths, camera_paths)
            renderer._camera_xforms = scene.bind(camera_paths)

    def _initialize_ovrtx(self, provider: Any, plan: Any) -> None:
        """Bind and attach the shared OVRTX scene to the simulation's SDP once."""
        if self._scene_data_provider is not None:
            if provider is not self._scene_data_provider:
                raise RuntimeError("One OVRTX scene cannot bind multiple scene-data providers.")
            return
        scene = self.scene
        rigid_paths = list(plan.iter_rigid_body_paths())
        if rigid_paths:
            self._object_xforms = scene.bind(rigid_paths)
        for stream in plan.point_stream_names:
            bindings = plan.point_bindings(stream)
            handle = scene.bind(
                [binding.path for binding in bindings], "points", dtype="float32", shape=(3,), is_array=True
            )
            scene.pin_world_space(handle)
            self._point_bindings[stream] = (
                handle,
                [binding.output_offset for binding in bindings],
                [binding.output_count for binding in bindings],
            )
        self._scene_data_provider = provider

    def _update_ovrtx(self, camera_xforms: Any = None, transform_stream: str | None = None) -> None:
        """Write each dirty shared or camera SDP publication into the OVRTX scene once."""
        provider = self._scene_data_provider
        if provider is None:
            raise RuntimeError("OVRTX updates require an initialized scene-data provider.")
        scene = self.scene
        if self._object_xforms is not None:
            transforms = provider.request_transforms(scene.transform_format)
            generation = provider.transform_generation()
            if generation != self._sdp_transform_generation:
                scene.write_xforms(self._object_xforms, transforms.matrices)
                self._sdp_transform_generation = generation
        for stream, binding in self._point_bindings.items():
            points = provider.request_points(scene.point_format, stream)
            if points is None:
                raise RuntimeError(f"OVRTX bound point stream {stream!r}, but SDP did not publish it.")
            generation = provider.point_generation(stream)
            if self._sdp_point_generations.get(stream) != generation:
                scene.write_points(binding, points.points)
                self._sdp_point_generations[stream] = generation
        if camera_xforms is not None:
            transforms = provider.request_transforms(scene.transform_format, name=transform_stream)
            generation = provider.transform_generation(transform_stream)
            if self._sdp_camera_transform_generations.get(transform_stream) != generation:
                scene.write_xforms(camera_xforms, transforms.matrices)
                self._sdp_camera_transform_generations[transform_stream] = generation

    def create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> OVRTXVisualMaterialWriter:
        """Compile the one material writer owned by this shared OVRTX scene."""
        from isaaclab_ov.renderers.visual_materials import OVRTXVisualMaterialWriter

        if self._scene_data_provider is None:
            raise RuntimeError("OVRTX must ingest its detached scene before material writes are compiled.")
        if self._visual_material_writer is not None:
            raise RuntimeError("One OVRTX scene can own only one visual-material writer.")
        self._visual_material_writer = OVRTXVisualMaterialWriter(self, batches)
        return self._visual_material_writer

    def _render_ovrtx(self, render_product_paths: Sequence[str]) -> Any:
        """Publish shared material writes and render one camera client's products."""
        writer = self._visual_material_writer
        try:
            if writer is not None:
                writer.publish()
            products = self.scene.step(render_product_paths)
        except BaseException:
            if writer is not None:
                with contextlib.suppress(Exception):
                    writer.drain()
            raise
        if writer is not None:
            writer.drain()
        return products

    def _remove_renderer(self, renderer: Any) -> None:
        """Release the shared native resource after its final camera client closes."""
        if renderer not in self._renderers:
            return
        self._renderers.remove(renderer)
        if self._renderers:
            return
        if self._visual_material_writer is not None:
            self._visual_material_writer.close()
        if self._ovrtx_scene is not None:
            self._ovrtx_scene.close()
        self._ovrtx_scene = self._ovrtx_key = None
        self._scene_data_provider = self._object_xforms = None
        self._point_bindings.clear()
        self._visual_material_writer = None
        self._sdp_transform_generation = -1
        self._sdp_camera_transform_generations.clear()
        self._sdp_point_generations.clear()

    @property
    def stage_usda(self) -> str:
        """The one cached, plan-gated USDA snapshot for this simulation."""
        if self._stage_usda is None:
            raise RuntimeError("The OV stage snapshot is unavailable before clone replication completes.")
        return self._stage_usda

    @property
    def clone_rows(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Native-safe source-to-target rows for consumer-owned cloning."""
        rows = tuple(
            (source, tuple(targets))
            for index, (source, _destination, targets, _transforms) in enumerate(self._rows)
            if index not in self._materialized_rows and targets
        )
        top_level = []
        for source, targets in rows:
            ancestors = tuple(
                (other_source, other_targets)
                for other_source, other_targets in rows
                if source != other_source and cloner.path.under(source, other_source)
            )
            for ancestor, ancestor_targets in ancestors:
                carried = tuple(cloner.path.rebase(source, ancestor, target) for target in ancestor_targets)
                if carried != targets:
                    raise RuntimeError(
                        f"OV cannot collapse nested clone row {source!r}: ancestor {ancestor!r} carries it to"
                        f" {carried}, but the plan targets {targets}."
                    )
            if not ancestors:
                top_level.append((source, targets))
        return tuple(top_level)

    def create_ovstage(self) -> ovstage.Stage:
        """Build a populated rendering ovstage for a consumer that draws it, e.g. Newton's ``ViewerRTX``.

        The stage holds the plan-gated snapshot with every native clone row already copied onto its
        targets, so each environment has its own prims and visual materials, and USD instancing is
        expanded so per-environment material bindings apply. It uses GPU hierarchy computation, which
        ``ViewerRTX`` requires of a stage it borrows. The caller owns the stage.

        Returns:
            The populated :class:`ovstage.Stage`.
        """
        import ovrtx  # noqa: PLC0415
        import ovstage  # noqa: PLC0415

        from isaaclab_ov.stage import create_ovstage  # noqa: PLC0415

        ovrtx.register_schema_paths()
        stage = create_ovstage("isaaclab_render", gpu_hierarchy=True)
        ovstage.population.open_usd_from_string(stage, _expand_for_ovstage(self.stage_usda, self.clone_rows), ordinal=1)
        stage.advance_write_floor(1).wait()
        return stage

    @property
    def physics_clone_rows(self) -> tuple[tuple[str, tuple[str, ...], tuple[CloneTransform, ...]], ...]:
        """Return whole-environment or exact heterogeneous rows OVPhysX can clone natively."""
        if self._materialized_rows:
            rows = ()
        else:
            plan = self._sim.get_clone_plan()
            assert plan is not None and plan.positions is not None
            whole_env = _whole_env_copy(plan)
            if whole_env is None:
                rows = tuple(
                    (source, tuple(targets), tuple(transforms))
                    for source, _destination, targets, transforms in self._rows
                    if targets
                )
            elif len(self._env_prim_paths) < 2:
                rows = ()
            else:
                rows = (
                    (
                        self._env_prim_paths[0],
                        tuple(self._env_prim_paths[1:]),
                        tuple((*map(float, position), 0.0, 0.0, 0.0, 1.0) for position in plan.positions[1:]),
                    ),
                )
        return (*rows, *self._direct_physics_rows)

    @property
    def env_prim_paths(self) -> tuple[str, ...]:
        """Environment root prim paths in clone-plan environment order."""
        return tuple(self._env_prim_paths)

    @property
    def env_names(self) -> tuple[str, ...]:
        """Environment root prim names in clone-plan environment order."""
        return tuple(Sdf.Path(path).name for path in self._env_prim_paths)

    def _snapshot_stage(self, plan: ClonePlan) -> str:
        """Flatten once, retain exact prototypes, and author rows without a safe native clone."""
        layer = self.stage.Flatten()
        sources = tuple(Sdf.Path(source) for source, _destination, _targets, _poses in self._rows)
        global_sources = tuple(
            Sdf.Path(source)
            for source, destination in zip(plan.sources, plan.destinations, strict=True)
            if "{}" not in destination
        )
        with Sdf.ChangeBlock():
            for env_path_string in self._env_prim_paths:
                env_path = Sdf.Path(env_path_string)
                env_spec = layer.GetPrimAtPath(env_path)
                if env_spec is not None:
                    self._retain_sources(
                        env_spec, env_path, frozenset(source for source in sources if source.HasPrefix(env_path))
                    )
            copies = sorted(
                (
                    (Sdf.Path(source), Sdf.Path(target))
                    for index in self._materialized_rows
                    for source, _destination, targets, _poses in (self._rows[index],)
                    for target in targets
                ),
                key=lambda item: item[0].pathElementCount,
            )
            copied = []
            for source, target in copies:
                if any(
                    source.HasPrefix(root) and source.ReplacePrefix(root, destination) == target
                    for root, destination in copied
                ):
                    continue
                self._copy_source(layer, source.pathString, target.pathString)
                copied.append((source, target))
            retained = {*sources, *global_sources, Sdf.Path(self._sim.cfg.physics_prim_path)}
            pending = list(retained)
            while pending:
                root = pending.pop()
                root_spec = layer.GetPrimAtPath(root)
                if root_spec is None:
                    continue
                prims = [root_spec]
                while prims:
                    prim = prims.pop()
                    prims.extend(prim.nameChildren.values())
                    paths = [
                        *(item.primPath for item in prim.referenceList.GetAppliedItems() if not item.assetPath),
                        *(item.primPath for item in prim.payloadList.GetAppliedItems() if not item.assetPath),
                        *prim.inheritPathList.GetAppliedItems(),
                        *prim.specializesList.GetAppliedItems(),
                        *(path for rel in prim.relationships.values() for path in rel.targetPathList.GetAppliedItems()),
                        *(p for attr in prim.attributes.values() for p in attr.connectionPathList.GetAppliedItems()),
                    ]
                    for path in paths:
                        dependency = path.GetPrimPath()
                        if (
                            dependency.isEmpty
                            or not dependency.IsAbsolutePath()
                            or layer.GetPrimAtPath(dependency) is None
                            or any(
                                dependency.HasPrefix(existing) or existing.HasPrefix(dependency)
                                for existing in retained
                            )
                        ):
                            continue
                        retained.add(dependency)
                        pending.append(dependency)
            if self._materialized_rows or self._renderers or self._ovstage_requested:
                retained.update(Sdf.Path(path) for path in self._env_prim_paths)
            self._retain_sources(layer.pseudoRoot, Sdf.Path.absoluteRootPath, frozenset(retained))
        logger.info(
            "Serialized the simulation's one plan-gated OV snapshot (%d authored, %d native clone rows)",
            len(self._materialized_rows),
            len(self._rows) - len(self._materialized_rows),
        )
        return layer.ExportToString()

    @staticmethod
    def _copy_source(layer: Any, source: str, target: str) -> None:
        """Copy one capability-selected prototype beneath its planned structural ancestors."""
        target_path = Sdf.Path(target)
        if layer.GetPrimAtPath(target_path) is not None:
            raise RuntimeError(f"OV clone target already exists in the plan-gated snapshot: {target!r}.")
        parent_path = target_path.GetParentPath()
        if layer.GetPrimAtPath(parent_path) is None:
            missing = []
            ancestor = parent_path
            while layer.GetPrimAtPath(ancestor) is None:
                missing.append(ancestor)
                ancestor = ancestor.GetParentPath()
            Sdf.CreatePrimInLayer(layer, parent_path)
            for path in missing:
                spec = layer.GetPrimAtPath(path)
                spec.specifier = Sdf.SpecifierDef
                spec.typeName = "Xform"
        if not Sdf.CopySpec(layer, Sdf.Path(source), layer, target_path):
            raise RuntimeError(f"Failed to copy planned OV source {source!r} to {target!r}.")

    @classmethod
    def _retain_sources(cls, prim_spec: Any, prim_path: Sdf.Path, sources: frozenset[Sdf.Path]) -> None:
        """Trim one environment subtree to the ancestor chains and subtrees named as sources."""
        if prim_path in sources:
            return
        for child_name in list(prim_spec.nameChildren.keys()):
            child_path = prim_path.AppendChild(child_name)
            if child_path in sources:
                continue
            child_sources = frozenset(source for source in sources if source.HasPrefix(child_path))
            if child_sources:
                cls._retain_sources(prim_spec.nameChildren[child_name], child_path, child_sources)
            else:
                del prim_spec.nameChildren[child_name]


def ovphysx_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: np.ndarray,
    mapping: np.ndarray,
    positions: np.ndarray | None = None,
    quaternions: np.ndarray | None = None,
) -> None:
    """Publish one raw NumPy source-to-environment mapping to the active OvPhysX resource."""
    sim = SimulationContext.instance()
    if sim is None:
        raise RuntimeError("OvPhysX replication requires an active SimulationContext.")
    context = sim.get_or_create_backend(OvReplicateContext, sim)
    if context._physics_initialized:
        raise RuntimeError("OvPhysX clone rows must be declared before physics initialization.")
    context._direct_physics_rows.extend(
        (source, tuple(targets), tuple(transforms))
        for source, _destination, targets, transforms in _clone_rows(
            stage, sources, destinations, env_ids, mapping, positions, quaternions
        )
        if targets
    )
