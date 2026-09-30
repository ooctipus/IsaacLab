# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.scene_data.scene_data_backend import SceneDataFormat

from ._fabric_notices import disabled_fabric_change_notifies
from .path import split, under

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.renderers.base_renderer import VisualMaterialBatch
    from isaaclab.renderers.fabric_visual_material import FabricVisualMaterialWriter
    from isaaclab.scene_data import SceneDataProvider

    from .clone_plan import ClonePlan


def _select_columns(env_ids: np.ndarray, mask: np.ndarray | None, row: int) -> np.ndarray:
    """Return the mask columns selected by a replication row."""
    if mask is None:
        return np.arange(len(env_ids))
    row_mask = mask if mask.ndim == 1 else mask[row]
    return np.flatnonzero(row_mask)


class UsdReplicateContext:
    """Apply routed clone-plan rows to one USD stage."""

    # USD destinations must exist before native physics contexts consume them.
    replicate_priority = -100

    def __init__(self, stage: Usd.Stage):
        """Initialize the context.

        Args:
            stage: USD stage to author replicated prim specs into.
        """
        self.stage = stage
        self._fabric_provider: SceneDataProvider | None = None
        self._fabric_stage: Any | None = None
        self._fabric_hierarchy: Any | None = None
        self._fabric_hierarchy_generation = -1
        self._fabric_topology_changed = False
        self._fabric_device: str | None = None
        self._fabric_plan: ClonePlan | None = None
        self._fabric_transform_selection: Any | None = None
        self._fabric_transform_output: SceneDataFormat.FabricMatrix44 | None = None
        self._fabric_named_transform_outputs: dict[str, tuple[Any, tuple[str, ...]]] = {}
        self._fabric_point_selections: dict[str, Any] = {}
        self._fabric_point_outputs: dict[str, SceneDataFormat.FabricMeshPoints] = {}
        self._fabric_point_paths: dict[str, tuple[str, ...]] = {}
        self._fabric_camera_selection: Any | None = None
        self._fabric_camera_world: Any | None = None
        self._fabric_camera_local: Any | None = None
        self._fabric_camera_slots: dict[str, int] = {}

    def replicate(self, plan: ClonePlan) -> None:
        """Apply this context's routed rows from a clone plan.

        Args:
            plan: Replication layout shared by every clone backend.
        """
        if plan.env_ids is None:
            raise ValueError("ClonePlan.env_ids is required for replication.")
        rows = [
            row
            for row, destination in enumerate(plan.destinations)
            if "{}" in destination and bool(plan.clone_mask[row].any())
        ]
        if not rows:
            return

        # A homogeneous per-asset plan is one environment subtree copy in USD. Keeping the
        # semantic rows in the plan while collapsing this backend operation avoids redundant
        # CopySpec work without weakening the plan contract.
        prefixes = {split(plan.destinations[row])[0] for row in rows}
        source_env = None
        if len(prefixes) == 1 and bool(plan.clone_mask[rows].all()):
            prefix = prefixes.pop()
            candidate = f"{prefix}{int(plan.env_ids[0])}"
            if all(under(plan.sources[row], candidate) for row in rows):
                source_env = candidate
        if source_env is not None:
            replication_rows = [(source_env, f"{prefix}{{}}", plan.env_ids, plan.positions, None)]
        else:
            replication_rows = []
            for row in rows:
                columns = _select_columns(plan.env_ids, plan.clone_mask, row)
                target_envs = plan.env_ids[columns]
                positions = None if plan.positions is None else plan.positions[columns]
                replication_rows.append((plan.sources[row], plan.destinations[row], target_envs, positions, None))

        # Suspend Fabric's per-Sdf.CopySpec notice listener for the duration of the copy work;
        # no-op outside a live Kit application.
        with disabled_fabric_change_notifies(self.stage):
            self._apply(replication_rows)

    def _apply(
        self,
        replication_rows: list[tuple[str, str, np.ndarray, np.ndarray | None, np.ndarray | None]],
    ) -> None:
        """Author the supplied copy specs into the stage's root layer."""
        # pxr must be imported after Kit starts; importing it with this module can bind
        # a different USD runtime before Kit initializes its plugins.
        from pxr import Gf, Sdf, UsdGeom, Vt  # noqa: PLC0415

        rl = self.stage.GetRootLayer()

        def dp_depth(template: str) -> int:
            """Return destination prim path depth for stable parent-first replication."""
            dp = template.format(0)
            return Sdf.Path(dp).pathElementCount

        rows_by_depth: dict[int, list[tuple[str, str, np.ndarray, np.ndarray | None, np.ndarray | None]]] = {}
        for row in replication_rows:
            rows_by_depth.setdefault(dp_depth(row[1]), []).append(row)

        for depth in sorted(rows_by_depth):
            with Sdf.ChangeBlock():
                for src, tmpl, target_envs, positions, quaternions in rows_by_depth[depth]:
                    _, clone_suffix = split(tmpl)
                    is_instance_root = clone_suffix == ""

                    for column, wid in enumerate(target_envs):
                        wid = int(wid)
                        dp = tmpl.format(wid)
                        Sdf.CreatePrimInLayer(rl, dp)
                        # ``CreatePrimInLayer`` authors missing intermediate ancestors (e.g. the
                        # ``Groceries`` scope in ``env_{}/Groceries/Object``) as ``over`` specs. A
                        # ``def`` copied below an ``over`` ancestor never composes as defined, so
                        # Hydra skips it and its references stay unexpanded. Promote such ancestors
                        # to ``def``; for ancestors already defined elsewhere this is a no-op.
                        ancestor = Sdf.Path(dp).GetParentPath()
                        while ancestor != Sdf.Path.absoluteRootPath:
                            ancestor_spec = rl.GetPrimAtPath(ancestor)
                            if ancestor_spec is None or ancestor_spec.specifier != Sdf.SpecifierOver:
                                break
                            ancestor_spec.specifier = Sdf.SpecifierDef
                            ancestor = ancestor.GetParentPath()
                        if src != dp:
                            Sdf.CopySpec(rl, Sdf.Path(src), rl, Sdf.Path(dp))

                        # Author positions/quaternions for instance roots only.
                        if is_instance_root and (positions is not None or quaternions is not None):
                            ps = rl.GetPrimAtPath(dp)
                            op_names = []
                            if positions is not None:
                                p = positions[column]
                                t_attr = ps.GetAttributeAtPath(dp + ".xformOp:translate")
                                if t_attr is None:
                                    t_attr = Sdf.AttributeSpec(ps, "xformOp:translate", Sdf.ValueTypeNames.Double3)
                                t_attr.default = Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
                                op_names.append("xformOp:translate")
                            if quaternions is not None:
                                q = quaternions[column]
                                o_attr = ps.GetAttributeAtPath(dp + ".xformOp:orient")
                                if o_attr is None:
                                    o_attr = Sdf.AttributeSpec(ps, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
                                o_attr.default = Gf.Quatd(float(q[3]), Gf.Vec3d(float(q[0]), float(q[1]), float(q[2])))
                                op_names.append("xformOp:orient")
                            if op_names:
                                op_order = ps.GetAttributeAtPath(dp + ".xformOpOrder") or Sdf.AttributeSpec(
                                    ps, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                                )
                                op_order.default = Vt.TokenArray(op_names)

    def create_fabric_visual_material_writer(
        self, batches: tuple[VisualMaterialBatch, ...]
    ) -> FabricVisualMaterialWriter:
        """Compile material writes against this clone resource's Fabric stage."""
        from isaaclab.renderers.fabric_visual_material import FabricVisualMaterialWriter

        if self._fabric_stage is None:
            raise RuntimeError("Fabric visual materials require initialized USD Fabric destinations.")
        return FabricVisualMaterialWriter(self._bind_fabric_visual_material, batches)

    def _bind_fabric_visual_material(
        self, batch: VisualMaterialBatch, input_name: str, rows: tuple[int, ...]
    ) -> tuple[Any, wp.array]:
        """Bind one exact plan-derived shader group into the clone-owned Fabric stage."""
        import usdrt

        value_types = {
            (): usdrt.Sdf.ValueTypeNames.Float,
            (2,): usdrt.Sdf.ValueTypeNames.Float2,
            (3,): usdrt.Sdf.ValueTypeNames.Color3f,
        }
        trailing_shape = tuple(batch.values.shape[1:])
        if trailing_shape not in value_types:
            raise TypeError(f"Unsupported Fabric visual-material tensor shape: {trailing_shape}.")
        shader_paths = tuple(batch.shader_paths[row] for row in rows)
        marker = f"isaaclab:visualMaterial:{batch.channel}:{input_name}"
        attribute_name = f"inputs:{input_name}"
        selection = self._fabric_stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.UInt, marker, usdrt.Usd.Access.Read),
                (value_types[trailing_shape], attribute_name, usdrt.Usd.Access.ReadWrite),
            ],
            device=str(batch.values.device),
            want_paths=True,
        )
        selected_paths = tuple(map(str, selection.GetPaths()))
        if len(set(shader_paths)) != len(shader_paths) or set(selected_paths) != set(shader_paths):
            raise RuntimeError(
                f"Fabric visual-material selection for {attribute_name!r} resolved {selected_paths!r}; "
                f"the clone plan declared {shader_paths!r}."
            )
        slots = {path: slot for slot, path in enumerate(selected_paths)}
        inverse = [-1] * len(batch.values)
        for row, path in zip(rows, shader_paths, strict=True):
            inverse[row] = slots[path]
        return selection, wp.array(inverse, dtype=wp.int32, device=str(batch.values.device))

    def _prepare_fabric(self, provider: SceneDataProvider, device: str, plan: ClonePlan) -> None:
        """Bind exact plan-owned destinations from the FSD-populated Fabric stage."""
        if self._fabric_provider is not None:
            if self._fabric_provider is not provider:
                raise RuntimeError("One USD clone context cannot bind multiple scene-data providers.")
            return
        if not get_settings_manager().get("/app/useFabricSceneDelegate", False):
            raise RuntimeError("Fabric scene output requires an experience with /app/useFabricSceneDelegate=true.")
        transform_paths = tuple(plan.iter_rigid_body_paths())
        point_paths = {
            name: tuple(binding.path for binding in plan.point_bindings(name)) for name in plan.point_stream_names
        }

        import usdrt  # noqa: PLC0415
        from pxr import UsdUtils  # noqa: PLC0415

        usdrt_hierarchy = importlib.import_module("usdrt.hierarchy")
        stage_id = UsdUtils.StageCache.Get().GetId(self.stage).ToLongInt()
        stage = usdrt.Usd.Stage.Attach(stage_id)
        stage.SynchronizeToFabric()
        hierarchy = usdrt_hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            stage.GetFabricId(), stage.GetStageIdAsStageId()
        )
        if hierarchy is None:
            raise RuntimeError("USD Fabric has no transform hierarchy.")

        hierarchy.update_world_xforms()
        transform_attrs = [
            (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read),
            (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:localMatrix", usdrt.Usd.Access.ReadWrite),
        ]
        if transform_paths:
            selection = stage.SelectPrims(
                require_applied_schemas=["PhysicsRigidBodyAPI"],
                require_attrs=transform_attrs,
                device=device,
                want_paths=True,
            )
            path_slots = self._selection_path_slots(selection)
            slots = wp.array(
                self._exact_path_slots(path_slots, transform_paths, "transform"), dtype=wp.int32, device=device
            )
            self._fabric_transform_selection = selection
            self._fabric_transform_output = SceneDataFormat.FabricMatrix44(
                matrices=wp.indexedfabricarray(fa=wp.fabricarray(selection, "omni:fabric:worldMatrix"), indices=slots),
                local_matrices=wp.indexedfabricarray(
                    fa=wp.fabricarray(selection, "omni:fabric:localMatrix"), indices=slots
                ),
                source_indices=wp.array(range(len(transform_paths)), dtype=wp.int32, device=device),
            )

        selection = stage.SelectPrims(
            require_prim_type="Camera", require_attrs=transform_attrs, device=device, want_paths=True
        )
        self._fabric_camera_selection = selection
        self._fabric_camera_world = wp.fabricarray(selection, "omni:fabric:worldMatrix")
        self._fabric_camera_local = wp.fabricarray(selection, "omni:fabric:localMatrix")
        self._fabric_camera_slots = self._selection_path_slots(selection)

        for name, paths in point_paths.items():
            selection = stage.SelectPrims(
                require_attrs=[
                    (usdrt.Sdf.ValueTypeNames.Point3fArray, "points", usdrt.Usd.Access.ReadWrite),
                    (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read),
                ],
                device=device,
                want_paths=True,
            )
            self._fabric_point_selections[name] = selection
            self._fabric_point_paths[name] = paths
            binding_slots = wp.array(
                self._exact_path_slots(self._selection_path_slots(selection), paths, f"{name!r} point"),
                dtype=wp.int32,
                device=device,
            )
            self._fabric_point_outputs[name] = SceneDataFormat.FabricMeshPoints(
                points=wp.fabricarrayarray(data=selection, attrib="points", dtype=wp.vec3f),
                world_matrices=wp.fabricarray(data=selection, attrib="omni:fabric:worldMatrix"),
                binding_slots=binding_slots,
            )

        self._fabric_stage = stage
        self._fabric_hierarchy = hierarchy
        self._fabric_device = device
        self._fabric_plan = plan
        self._fabric_provider = provider
        provider._bind_fabric_outputs(self._prepare_fabric_output)

    @staticmethod
    def _selection_path_slots(selection: Any) -> dict[str, int]:
        """Return one vectorized Fabric selection's path-to-slot mapping."""
        paths = tuple(map(str, selection.GetPaths()))
        if len(paths) != selection.GetCount() or len(set(paths)) != len(paths):
            raise RuntimeError("Fabric selection returned invalid path metadata.")
        return {path: slot for slot, path in enumerate(paths)}

    @staticmethod
    def _exact_path_slots(path_slots: dict[str, int], paths: tuple[str, ...], label: str) -> list[int]:
        """Map exact plan paths to selection slots, rejecting incomplete Fabric population."""
        missing = tuple(path for path in paths if path not in path_slots)
        if missing:
            raise RuntimeError(f"Fabric {label} destinations are missing plan paths: {missing!r}.")
        return [path_slots[path] for path in paths]

    def _prepare_fabric_output(self, output_format: Any, name: str | None) -> Any | None:
        """Rebind one SDP destination immediately before its conversion."""
        if output_format is SceneDataFormat.FabricMeshPoints:
            selection = self._fabric_point_selections.get(name)
            if selection is None:
                return None
            if not selection.PrepareForReuse():
                return self._fabric_point_outputs[name]
            self._fabric_topology_changed = True
            self._fabric_hierarchy_generation = -1
            output = self._fabric_point_outputs[name]
            output.points = wp.fabricarrayarray(data=selection, attrib="points", dtype=wp.vec3f)
            output.world_matrices = wp.fabricarray(data=selection, attrib="omni:fabric:worldMatrix")
            output.binding_slots = wp.array(
                self._exact_path_slots(
                    self._selection_path_slots(selection), self._fabric_point_paths[name], f"{name!r} point"
                ),
                dtype=wp.int32,
                device=self._fabric_device,
            )
            return output
        if output_format is not SceneDataFormat.FabricMatrix44:
            raise TypeError(f"Unsupported clone-owned Fabric format: {output_format!r}.")

        if name is None:
            selection = self._fabric_transform_selection
            if selection is not None and selection.PrepareForReuse():
                output = self._fabric_transform_output
                paths = tuple(self._fabric_plan.iter_rigid_body_paths())
                slots = wp.array(
                    self._exact_path_slots(self._selection_path_slots(selection), paths, "transform"),
                    dtype=wp.int32,
                    device=self._fabric_device,
                )
                output.matrices = wp.indexedfabricarray(
                    fa=wp.fabricarray(selection, "omni:fabric:worldMatrix"), indices=slots
                )
                output.local_matrices = wp.indexedfabricarray(
                    fa=wp.fabricarray(selection, "omni:fabric:localMatrix"), indices=slots
                )
                self._fabric_topology_changed = True
                self._fabric_hierarchy_generation = -1
            return self._fabric_transform_output

        selection = self._fabric_camera_selection
        if selection is None:
            raise RuntimeError("Fabric camera destinations were not prepared before use.")
        if selection.PrepareForReuse():
            self._fabric_camera_world = wp.fabricarray(selection, "omni:fabric:worldMatrix")
            self._fabric_camera_local = wp.fabricarray(selection, "omni:fabric:localMatrix")
            self._fabric_camera_slots = self._selection_path_slots(selection)
            for output, paths in self._fabric_named_transform_outputs.values():
                self._bind_fabric_camera_output(output, paths)
            self._fabric_topology_changed = True
            self._fabric_hierarchy_generation = -1

        binding = self._fabric_named_transform_outputs.get(name)
        if binding is None:
            frames = self._fabric_plan.match_frames(name)
            paths = tuple(frame.path for frame in frames)
            if any(frame.parent_path is None for frame in frames):
                raise RuntimeError(f"Fabric transform destinations for {name!r} must have planned parents.")
            output = SceneDataFormat.FabricMatrix44(
                source_indices=wp.array(range(len(frames)), dtype=wp.int32, device=self._fabric_device)
            )
            self._fabric_named_transform_outputs[name] = (output, paths)
            self._bind_fabric_camera_output(output, paths)
        else:
            output, _ = binding
        return output

    def _bind_fabric_camera_output(self, output: Any, paths: tuple[str, ...]) -> None:
        """Bind one plan-exact named camera output to the vectorized Camera selection."""
        slots = wp.array(
            self._exact_path_slots(self._fabric_camera_slots, paths, "camera"),
            dtype=wp.int32,
            device=self._fabric_device,
        )
        output.matrices = wp.indexedfabricarray(fa=self._fabric_camera_world, indices=slots)
        output.local_matrices = wp.indexedfabricarray(fa=self._fabric_camera_local, indices=slots)

    def _update_fabric_hierarchy(self) -> None:
        """Propagate converted local matrices through the renderer-owned hierarchy."""
        if self._fabric_hierarchy is None:
            raise RuntimeError("Fabric destinations were not prepared before use.")
        generation = self._fabric_provider._fabric_generation
        if generation == self._fabric_hierarchy_generation:
            return
        if not self._fabric_hierarchy.update_world_xforms_gpu(not self._fabric_topology_changed):
            raise RuntimeError("Fabric GPU hierarchy propagation failed.")
        wp.synchronize_device(self._fabric_device)
        self._fabric_hierarchy_generation = generation
        self._fabric_topology_changed = False


def usd_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: np.ndarray,
    mask: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    quaternions: np.ndarray | None = None,
) -> None:
    """Replicate USD prims directly for standalone tooling and tests.

    Production clone lifecycles route a :class:`~isaaclab.cloner.ClonePlan` through
    :meth:`UsdReplicateContext.replicate`; this wrapper retains direct control over raw
    mappings for tools that do not own a clone plan.

    Args:
        stage: USD stage.
        sources: Source prim paths.
        destinations: Destination formattable templates with ``"{}"`` for env index.
        env_ids: Environment indices.
        mask: Optional per-source or shared mask. ``None`` selects all.
        positions: Optional positions [m], shape ``[E, 3]``. Authored as ``xformOp:translate`` only
            for env-instance root destinations (``.../env_{}``).
        quaternions: Optional orientations in xyzw order, shape ``[E, 4]``. Authored as
            ``xformOp:orient`` only for env-instance root destinations (``.../env_{}``).
    """
    replication_rows = []
    for row, source in enumerate(sources):
        columns = _select_columns(env_ids, mask, row)
        target_envs = env_ids[columns]
        row_positions = None if positions is None else positions[columns]
        row_quaternions = None if quaternions is None else quaternions[columns]
        replication_rows.append((source, destinations[row], target_envs, row_positions, row_quaternions))
    context = UsdReplicateContext(stage)
    if replication_rows:
        with disabled_fabric_change_notifies(stage):
            context._apply(replication_rows)
