# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import copy
import logging
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from newton import Control, Heightfield, Model, ModelBuilder, ModelFlags, ShapeFlags, State
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from isaaclab.cloner import ClonePlan
from isaaclab.cloner import path as clone_path
from isaaclab.cloner import query as clone_query
from isaaclab.scene_data import SceneDataFormat, SceneDataProvider
from isaaclab.sim.utils.queries import path_expr_to_glob
from isaaclab.utils import checked_apply
from isaaclab.utils.string import resolve_matching_names

from isaaclab_newton.cloner.newton_clone_utils import (
    _quat_rotate,
    build_source_builders,
    replicate_builder_mapping,
)

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.assets import BaseArticulation
    from isaaclab.renderers.base_renderer import VisualMaterialBatch
    from isaaclab.sim import SimulationContext

    from isaaclab_newton.physics.newton_manager_cfg import NewtonSolverCfg
    from isaaclab_newton.renderers.visual_material import VisualMaterialWriter, VisualShapeColorWriter

logger = logging.getLogger(__name__)


def copy_newton_clone_source(resource, source_path: str, xform: wp.transform | None = None) -> ModelBuilder:
    """Copy a retained clone-source builder without sharing mutable shape geometry.

    Args:
        source_path: Clone-plan source prim path retained during Newton replication.
        xform: Optional transform applied while copying the source.

    Returns:
        An independent builder that is safe to finalize or extend.

    Raises:
        RuntimeError: If Newton replication did not retain the requested source.
    """
    source = resource._cl_protos.get(source_path)
    if source is None:
        raise RuntimeError(f"No retained Newton clone source for {source_path!r}.")
    builder = ModelBuilder(up_axis=source.up_axis)
    if xform is None:
        builder.add_builder(source)
    else:
        builder.add_builder(source, xform=xform)
    builder.shape_source = [
        value.copy() if callable(getattr(value, "copy", None)) else copy.copy(value) for value in builder.shape_source
    ]
    return builder


@contextlib.contextmanager
def newton_builder_world_hook(
    resource,
    hook: Callable[[ModelBuilder, int, np.ndarray, np.ndarray], None],
) -> Iterator[None]:
    """Temporarily extend every world built by Newton replication.

    The callback must not already be registered. On exit, the context removes
    only its callback and preserves hooks owned by other callers.

    Args:
        hook: Callback receiving the builder, world index, world position [m],
            and world orientation quaternion in xyzw order during replication.

    Yields:
        Control while the callback is registered.

    Raises:
        RuntimeError: If the callback is already registered.
    """
    hooks = resource._per_world_builder_hooks
    if hook in hooks:
        raise RuntimeError("Newton world-builder hook is already registered.")
    hooks.append(hook)
    try:
        yield
    finally:
        if hook in hooks:
            hooks.remove(hook)


def _build_newton_builder_from_mapping(
    resource,
    plan: ClonePlan,
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: np.ndarray,
    mapping: np.ndarray,
    positions: np.ndarray | None = None,
    quaternions: np.ndarray | None = None,
    up_axis: str = "Z",
    load_visual_shapes: bool = False,
) -> tuple[ModelBuilder, dict, dict[str, ModelBuilder]]:
    """Build a Newton model builder from clone mapping inputs.

    Returns the per-source builders so the committing path can retain them for
    single-model consumers such as the batched Newton IK action.
    """
    if positions is None:
        positions = np.zeros((mapping.shape[1], 3), dtype=np.float32)
    if quaternions is None:
        quaternions = np.zeros((mapping.shape[1], 4), dtype=np.float32)
        quaternions[:, 3] = 1.0

    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    builder = resource.create_builder(up_axis=up_axis)
    if not plan.is_complete:
        raise RuntimeError("Newton replication requires a completed clone plan.")
    global_sources = tuple(
        source for source, destination in zip(plan.sources, plan.destinations) if "{}" not in destination
    )
    # Swap plan-declared global terrain colliders before importing their USD sources.
    hf_ignore_paths = []
    for geometry in plan.geometry_prototypes:
        if geometry.heightfield is None:
            continue
        if geometry.frame.body_path is not None:
            raise RuntimeError(f"Heightfield {geometry.path!r} must be world-attached.")
        pose = np.asarray(geometry.frame.pose, dtype=np.float32)
        world = _quat_rotate(pose[3:], geometry.vertices) + pose[:3]
        mesh = wp.Mesh(
            points=wp.array(world, dtype=wp.vec3, device=resource._sim.cfg.device),
            indices=wp.array(geometry.faces.reshape(-1), dtype=wp.int32, device=resource._sim.cfg.device),
        )
        heightfield, xform = Heightfield.create_from_mesh(mesh, geometry.heightfield[1])
        builder.add_shape_heightfield(heightfield=heightfield, xform=xform)
        hf_ignore_paths.append(geometry.heightfield[0])

    registered_deformables = tuple(
        (entry, plan.match_deformable_subtrees(entry.prim_path)) for entry in resource._deformable_registry
    )
    deformable_ignore_paths = tuple(
        dict.fromkeys(item.source_path for _entry, entries in registered_deformables for item in entries)
    )
    if None in deformable_ignore_paths:
        raise RuntimeError("A registered Newton deformable has no clone-plan source path.")

    source_builders = build_source_builders(
        stage,
        (*sources, *global_sources),
        lambda: resource.create_builder(up_axis=up_axis),
        schema_resolvers,
        ignore_paths=(*hf_ignore_paths, *deformable_ignore_paths) or None,
        load_visual_shapes=load_visual_shapes,
    )
    for source in global_sources:
        builder.add_builder(source_builders.pop(source))

    # A shared plan also contains renderer-only rows such as cameras. Their USD imports
    # produce no Newton entities, so they are not native replication work.
    replication_rows = tuple(
        row
        for row, source in enumerate(sources)
        if any(ModelBuilder._builder_merge_counts(source_builders[source]).values())
        or any(source_builders[source]._custom_frequency_counts.values())
    )
    replication_sources = tuple(sources[row] for row in replication_rows)
    replication_destinations = tuple(destinations[row] for row in replication_rows)
    replication_mapping = mapping[list(replication_rows)]
    replication_builders = {source: source_builders[source] for source in replication_sources}

    # Collapse a fully homogeneous scene to its environment root so Newton can use its
    # native batched replication. Heterogeneous row combinations are cached downstream.
    if len(replication_sources) > 1 and mapping.shape[1] and bool(replication_mapping.all()):
        prefixes = {clone_path.split(destination)[0] for destination in replication_destinations}
        if len(prefixes) == 1:
            prefix = prefixes.pop()
            source_env = f"{prefix}{int(env_ids[0])}"
            if all(clone_path.under(source, source_env) for source in replication_sources):
                prototype = resource.create_builder(up_axis=up_axis)
                for source in replication_sources:
                    prototype.add_builder(source_builders[source])
                replication_sources = (source_env,)
                replication_destinations = (f"{prefix}{{}}",)
                replication_mapping = np.ones((1, mapping.shape[1]), dtype=np.bool_)
                replication_builders = {source_env: prototype}

    # Inject registered sites into source builders (and global sites into main builder).
    global_sites, source_sites, root_sites = resource._cl_inject_sites(builder, replication_builders)

    point_order = {binding.path: binding.source_offset for binding in plan.point_bindings()}
    hook_order = {
        id(entry): min(point_order[item.vis_mesh_path] for item in entries) for entry, entries in registered_deformables
    }
    hook_order.update(
        (
            id(entry),
            min(point_order[path] for path in clone_query.destination_paths(plan, entry.cfg.prim_path).values()),
        )
        for entry in resource._mpm_object_registry
    )
    hooks = sorted(
        resource._per_world_builder_hooks,
        key=lambda hook: hook_order.get(id(getattr(hook, "__self__", None)), -1),
    )

    local_site_map, _world_xforms, _fabric_body_bindings = replicate_builder_mapping(
        builder=builder,
        sources=replication_sources,
        mapping=replication_mapping,
        positions=positions,
        quaternions=quaternions,
        source_builders=replication_builders,
        destinations=replication_destinations,
        env_ids=env_ids,
        source_site_indices=source_sites,
        env_root_sites=root_sites,
        per_world_builder_hooks=hooks,
    )

    site_index_map = {label: (idx, None) for label, idx in global_sites.items()}
    site_index_map.update((label, (None, per_world)) for label, per_world in local_site_map.items())
    return builder, site_index_map, source_builders


def _verify_point_layout(builder, plan, env_ids, hook_bindings) -> None:
    """Verify that Newton's native particle pointer follows canonical clone-plan order."""
    hook_paths = {path for path, *_ in hook_bindings}
    env_columns = {int(env_id): column for column, env_id in enumerate(env_ids)}
    expected = {}
    if plan.is_complete:
        for entry in plan.deformables:
            if entry.vis_mesh_path in hook_paths:
                continue
            if entry.env_id is None:
                world = -1
                destination = plan.destinations[entry.row]
            else:
                try:
                    world = env_columns[entry.env_id]
                except KeyError as exc:
                    raise RuntimeError(f"Clone-plan deformable {entry.root_path!r} has no Newton world.") from exc
                destination = plan.destinations[entry.row].format(entry.env_id)
            source_path = clone_path.rebase(entry.sim_mesh_path, destination, plan.sources[entry.row])
            expected.setdefault((world, source_path), []).append(entry)

    imported_bindings = []
    for family, deformable_type in (("cloth", "surface"), ("soft", "volume")):
        fields = tuple(
            getattr(builder, f"_{family}_{suffix}") for suffix in ("label", "world", "particle_start", "particle_end")
        )
        if len({len(field) for field in fields}) != 1:
            raise RuntimeError(f"Newton {family} particle registry has inconsistent columns.")
        for source_path, world, start, end in zip(*fields, strict=True):
            candidates = expected.get((int(world), source_path), [])
            if not candidates:
                raise RuntimeError(
                    f"Newton imported {deformable_type} {source_path!r} in world {int(world)} is undeclared or"
                    " duplicated."
                )
            entry = candidates.pop(0)
            start, end = int(start), int(end)
            if entry.deformable_type != deformable_type or end - start != entry.vertex_count:
                raise RuntimeError(f"Newton particle count does not match clone-plan deformable {entry.root_path!r}.")
            if start < 0 or end > builder.particle_count or start >= end:
                raise RuntimeError(
                    f"Newton deformable {entry.root_path!r} has invalid particle range [{start}, {end})."
                )
            if any(int(particle_world) != int(world) for particle_world in builder.particle_world[start:end]):
                raise RuntimeError(f"Newton deformable {entry.root_path!r} crosses particle worlds.")
            imported_bindings.append((entry.vis_mesh_path, start, end - start))

    missing = [entry.root_path for entries in expected.values() for entry in entries]
    if missing:
        raise RuntimeError(f"Clone-plan deformables were not imported by Newton: {missing!r}.")

    bindings = sorted((*hook_bindings, *imported_bindings), key=lambda item: item[1])
    if len({path for path, *_ in bindings}) != len(bindings):
        raise RuntimeError("Newton particle bindings contain duplicate visual paths.")
    cursor = 0
    for path, offset, count in bindings:
        if count <= 0 or offset != cursor:
            raise RuntimeError(f"Newton particle binding {path!r} leaves a gap or overlaps at offset {offset}.")
        cursor += count
    if cursor != builder.particle_count:
        raise RuntimeError(f"Newton particle bindings cover {cursor} of {builder.particle_count} particles.")
    planned = (
        ()
        if not plan.is_complete
        else tuple((binding.path, binding.source_offset, binding.source_count) for binding in plan.point_bindings())
    )
    if tuple(bindings) != planned:
        raise RuntimeError(f"Newton particle order {tuple(bindings)!r} does not match clone plan {planned!r}.")


def _replicate_plan(
    resource: NewtonReplicateContext,
    plan: ClonePlan,
    stage: Usd.Stage,
    rows: Sequence[int],
    *,
    quaternions: np.ndarray | None = None,
) -> ModelBuilder:
    """Build and publish one Newton builder from selected rows of a completed plan."""
    if not plan.is_complete or plan.env_ids is None or plan.positions is None:
        raise RuntimeError("Newton replication requires a completed clone plan.")
    sources = tuple(plan.sources[row] for row in rows)
    destinations = tuple(plan.destinations[row] for row in rows)
    mapping = plan.clone_mask[list(rows)]
    builder, site_index_map, source_builders = _build_newton_builder_from_mapping(
        resource=resource,
        plan=plan,
        stage=stage,
        sources=sources,
        destinations=destinations,
        env_ids=plan.env_ids,
        mapping=mapping,
        positions=plan.positions,
        quaternions=quaternions,
        up_axis=resource._up_axis,
        load_visual_shapes=resource.load_visual_shapes,
    )
    env_columns = {int(env_id): column for column, env_id in enumerate(plan.env_ids)}
    hook_bindings = []
    for entry in resource._deformable_registry:
        deformables = plan.match_deformable_subtrees(entry.prim_path)
        paths = (
            {column: deformables[0].vis_mesh_path for column in env_columns.values()}
            if deformables[0].env_id is None
            else {env_columns[item.env_id]: item.vis_mesh_path for item in deformables}
        )
        if entry.planned_worlds is None:
            raise RuntimeError(f"Newton deformable {entry.prim_path!r} has no clone-plan worlds.")
        hook_bindings.extend(
            (paths[world], int(offset), int(entry.particles_per_body))
            for world, offset in zip(entry.planned_worlds, entry.particle_offsets, strict=True)
        )
    for entry in resource._mpm_object_registry:
        paths = {
            env_columns[env_id]: path
            for env_id, path in clone_query.destination_paths(plan, entry.cfg.prim_path).items()
        }
        if entry.planned_worlds is None:
            raise RuntimeError(f"Newton MPM object {entry.cfg.prim_path!r} has no clone-plan worlds.")
        hook_bindings.extend(
            (paths[world], int(offset), int(entry.particles_per_object))
            for world, offset in zip(entry.planned_worlds, entry.particle_offsets, strict=True)
        )
    _verify_point_layout(builder, plan, plan.env_ids, hook_bindings)
    resource._cl_site_index_map = site_index_map
    resource._cl_protos = source_builders
    resource.set_builder(builder)
    resource._num_envs = mapping.shape[1]
    return builder


class NewtonReplicateContext:
    """Simulation-scoped Newton native resource built from one clone plan.

    The resource is independent of :class:`~isaaclab.physics.PhysicsManager`. Physics, renderers,
    and visualizers get or create this same object from the simulation registry. It exclusively
    owns the builder, model, state, control, contacts, and clone registries.
    """

    replicate_priority = 0
    """A physics backend parses the stage to build its model, so it runs after the backends
    that author the copies (USD at ``-100``) and export the prototypes (OVRTX at ``-200``)."""

    clones_whole_env = False
    """Newton builds each asset into its model builder, so it keeps the plan's per-asset rows
    rather than collapsing them into one copy of the environment."""

    def __init__(self, sim_context: SimulationContext, *, up_axis: str = "Z"):
        """Initialize the context.

        Args:
            sim_context: Simulation context that owns this native resource.
            up_axis: Up axis for the Newton model builder.
        """
        self._up_axis = up_axis
        self.load_visual_shapes = False
        self._sim = sim_context
        self._physics_cfg: NewtonSolverCfg | None = None
        self._builder_attribute_solvers: tuple[type, ...] = ()
        self.clear()

    def clear(self) -> None:
        """Release native and clone state while retaining the simulation binding."""
        self._sdp_generation: tuple[int, int] | None = None
        self._builder: ModelBuilder | None = None
        self._model: Model | None = None
        self._state_0: State | None = None
        self._state_1: State | None = None
        self._control: Control | None = None
        self._contacts = None
        self._num_envs: int | None = None
        self._cl_pending_sites: dict[tuple[str | None, bool, tuple[float, ...]], tuple[str, wp.transform]] = {}
        self._cl_site_index_map: dict[str, tuple[int | None, list[list[int]] | None]] = {}
        self._cl_protos: dict[str, ModelBuilder] = {}
        self._deformable_registry: list = []
        self._mpm_object_registry: list = []
        self._per_world_builder_hooks: list[Callable[[ModelBuilder, int, np.ndarray, np.ndarray], None]] = []
        self._pending_extended_state_attributes: set[str] = set()
        self._active_extended_state_attributes: set[str] = set()
        self._sensor_tasks: dict[str, Callable[[], None]] = {}
        self._sensor_state: State | None = None
        self._sensor_state_dirty = True
        self._sensor_capture: tuple[wp.Graph, wp.array, np.ndarray] | None = None
        self._sensor_bvh_shape_flags = ShapeFlags.VISIBLE
        self._state_force_callbacks: list[Callable[[State], None]] = []
        self._supports_rigid_body_force_input = False
        self._model_changes: set[ModelFlags] = set()
        self._articulation_views: dict[str, object] = {}

    def bind_physics(self, cfg: NewtonSolverCfg, builder_attribute_solvers: tuple[type, ...] = ()) -> None:
        """Bind the resolved Newton physics configuration before clone-plan dispatch."""
        if self._physics_cfg is not None and self._physics_cfg is not cfg:
            raise RuntimeError("A Newton resource cannot have multiple physics configurations.")
        self._physics_cfg = cfg
        self._builder_attribute_solvers = builder_attribute_solvers

    def set_builder(self, builder: ModelBuilder) -> None:
        """Set the Newton model builder owned by this resource."""
        self._builder = builder

    def create_builder(self, up_axis: str | None = None, **kwargs) -> ModelBuilder:
        """Create a Newton builder with the active backend's declarative defaults."""
        cfg = self._physics_cfg
        builder = ModelBuilder(up_axis=up_axis or self._up_axis, **kwargs)
        for solver_type in self._builder_attribute_solvers:
            solver_type.register_custom_attributes(builder)
        builder.default_bvh_cfg = ModelBuilder.BvhConfig(
            mesh_constructor=getattr(cfg, "bvh_constructor_geometry", None),
            gaussian_constructor=getattr(cfg, "bvh_constructor_gaussian", None),
            shape_constructor=getattr(cfg, "bvh_constructor_scene", None),
            shape_flags=self._sensor_bvh_shape_flags,
        )
        shape_cfg = getattr(cfg, "default_shape_cfg", None)
        if shape_cfg is None:
            from isaaclab_newton.physics.newton_manager_cfg import NewtonShapeCfg

            shape_cfg = NewtonShapeCfg()
        checked_apply(shape_cfg, builder.default_shape_cfg)
        return builder

    def cl_register_site(self, body_pattern: str | None, xform: wp.transform, *, per_world: bool = False) -> str:
        """Register a site request for injection before replication."""
        if per_world and body_pattern is not None:
            raise ValueError("per_world site registration requires body_pattern=None.")
        key = (body_pattern, per_world, tuple(xform))
        if key not in self._cl_pending_sites:
            label = f"ft_{len(self._cl_pending_sites)}"
            self._cl_pending_sites[key] = (label, xform)
        return self._cl_pending_sites[key][0]

    def request_extended_state_attribute(self, attr: str) -> None:
        """Request an extended state attribute before model finalization."""
        self._pending_extended_state_attributes.add(attr)

    def _cl_inject_sites(
        self, main_builder: ModelBuilder, source_builders: dict[str, ModelBuilder]
    ) -> tuple[dict[str, int], dict[int, dict[str, list[int]]], dict[str, wp.transform]]:
        """Inject registered global, per-world, and body-relative sites into builders."""
        global_sites: dict[str, int] = {}
        source_sites: dict[int, dict[str, list[int]]] = {}
        root_sites: dict[str, wp.transform] = {}
        for (body_pattern, per_world, _), (label, xform) in self._cl_pending_sites.items():
            if per_world:
                root_sites[label] = xform
            elif body_pattern is None:
                global_sites[label] = main_builder.add_site(body=-1, xform=xform, label=label)
            else:
                matched = False
                for source_builder in (*source_builders.values(), main_builder):
                    if source_builder is main_builder and matched:
                        break
                    indices, names = resolve_matching_names(
                        body_pattern, list(source_builder.body_label), raise_when_no_match=False
                    )
                    if not indices:
                        continue
                    matched = True
                    source_sites.setdefault(id(source_builder), {})[label] = [
                        source_builder.add_site(body=index, xform=xform, label=f"{name}/{label}")
                        for index, name in zip(indices, names)
                    ]
                if not matched:
                    raise ValueError(f"Site {label!r} body pattern {body_pattern!r} matched no Newton bodies.")
        self._cl_pending_sites.clear()
        return global_sites, source_sites, root_sites

    def get_model(self) -> Model | None:
        """Return the native Newton model owned by this resource."""
        return self._model

    def create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> VisualMaterialWriter:
        """Compile material-to-shape addresses for the shared native model."""
        from isaaclab_newton.renderers.visual_material import VisualMaterialWriter

        if self._model is None:
            raise RuntimeError("Newton visual materials require an initialized model.")
        return VisualMaterialWriter(self._model, batches)

    def create_visual_shape_color_writer(
        self, asset: BaseArticulation, body_names: tuple[str, ...]
    ) -> VisualShapeColorWriter:
        """Compile selected plan-owned articulation-body shape addresses."""
        from newton.selection import ArticulationView

        from isaaclab_newton.renderers.visual_material import VisualShapeColorWriter

        if self._model is None:
            raise RuntimeError("Newton visual-shape randomization requires an initialized model.")
        plan = self._sim.get_clone_plan()
        if plan is None or not plan.is_complete:
            raise RuntimeError("Newton visual-shape randomization requires a completed clone plan.")
        path_expr = asset.cfg.prim_path + (asset.cfg.articulation_root_prim_path or "")
        view_path = plan.match_articulation(path_expr).view_path
        view = self._articulation_views.get(view_path)
        if view is None:
            view = self._articulation_views[view_path] = ArticulationView(
                self._model, path_expr_to_glob(view_path), verbose=False
            )
        return VisualShapeColorWriter(self._model, view, body_names)

    def get_state_0(self) -> State | None:
        """Return the current native physics state."""
        return self._state_0

    def get_state_1(self) -> State | None:
        """Return the next solver state when this resource owns physics."""
        return self._state_1

    def get_control(self) -> Control | None:
        """Return the native Newton control buffer."""
        return self._control

    def register_state_force_callback(self, callback: Callable[[State], None]) -> None:
        """Register a graph-safe callback that applies forces before every solver substep."""
        if callback not in self._state_force_callbacks:
            self._state_force_callbacks.append(callback)

    def supports_rigid_body_force_input(self) -> bool:
        """Return whether the configured Newton solver consumes rigid-body forces."""
        return self._supports_rigid_body_force_input

    def add_model_change(self, change: ModelFlags) -> None:
        """Record a native model mutation for the owning solver."""
        self._model_changes.add(change)

    def finalize_visualization_model(self) -> None:
        """Finalize the clone-built model used when another backend owns physics."""
        if self._builder is None:
            raise RuntimeError("Newton replication did not produce a model builder.")
        self._model = self._builder.finalize(device=self._sim.cfg.device)
        self._state_0 = self._model.state()
        self._model.num_envs = self._num_envs
        self._sensor_state_dirty = True

    def request_visualization_state(self, provider: SceneDataProvider) -> State | None:
        """Return the clone-built state with dynamic pointers requested through SDP."""
        if self._state_0 is None or self._model is None:
            return self._state_0
        transforms = provider.request_transforms(SceneDataFormat.IndexedTransform)
        published_count = 0 if transforms is None else transforms.transforms.shape[0]
        if published_count != self._model.body_count:
            raise RuntimeError(
                f"SDP published {published_count} transforms for a Newton model with {self._model.body_count} bodies."
            )
        points = None
        if self._state_0.particle_q is not None:
            points = provider.request_points(SceneDataFormat.Points)
            published_count = 0 if points is None else points.points.shape[0]
            if published_count != self._model.particle_count:
                raise RuntimeError(
                    f"SDP published {published_count} points for a Newton model with {self._model.particle_count}."
                )
        generation = (provider.transform_generation(), provider.point_generation())
        if generation == self._sdp_generation:
            return self._state_0
        if transforms is not None:
            if self._state_0.body_q is not transforms.transforms:
                self._sensor_capture = None
            self._state_0.body_q = transforms.transforms
        if points is not None:
            if self._state_0.particle_q is not points.points:
                self._sensor_capture = None
            self._state_0.particle_q = points.points
        self._sdp_generation = generation
        self._sensor_state_dirty = True
        return self._state_0

    def _register_sensor_task(self, name: str, update_fn: Callable[[], None]) -> None:
        """Register a scene-query task on the native model."""
        if name in self._sensor_tasks:
            raise ValueError(f"Newton sensor task {name!r} is already registered.")
        model = self.get_model()
        state = self.request_visualization_state(self._sim.get_scene_data_provider())
        if model is None or state is None:
            raise RuntimeError("Registering a Newton sensor task requires an initialized model and state.")
        if model.shape_count > 0:
            if model.bvh_shapes is None:
                model.bvh_build_shapes(state)
        if model.particle_count > 0 and model.bvh_particles is None:
            model.bvh_build_particles(state)
        self._sensor_tasks[name] = update_fn
        self._sensor_state = state
        self._sensor_state_dirty = True
        self._sensor_capture = None

    def _unregister_sensor_task(self, name: str) -> None:
        """Remove a scene-query task."""
        if self._sensor_tasks.pop(name, None) is not None:
            self._sensor_capture = None

    def _update_sensor_tasks(self, *names: str) -> None:
        """Refresh native acceleration structures and run requested tasks."""
        state = self.request_visualization_state(self._sim.get_scene_data_provider())
        for name in names:
            if name not in self._sensor_tasks:
                raise KeyError(f"Newton sensor task {name!r} is not registered.")
        if state is not self._sensor_state:
            self._sensor_state = state
            self._sensor_state_dirty = True
            self._sensor_capture = None
        use_cuda_graph = bool(getattr(self._physics_cfg, "use_cuda_graph", False)) and "cuda" in str(
            self._sim.cfg.device
        )
        if use_cuda_graph and self._sensor_capture is None:
            self._capture_sensor_graph()
        if self._sensor_capture is None:
            if self._sensor_state_dirty:
                self._refit_sensor_bvhs()
                self._sensor_state_dirty = False
            for name in names:
                self._sensor_tasks[name]()
            return

        graph, flags, flags_host = self._sensor_capture
        flags_host.fill(0)
        flags_host[0] = int(self._sensor_state_dirty)
        task_names = tuple(self._sensor_tasks)
        for name in names:
            flags_host[1 + task_names.index(name)] = 1
        flags.assign(flags_host)
        wp.capture_launch(graph)
        self._sensor_state_dirty = False

    def _refit_sensor_bvhs(self) -> None:
        """Refit native acceleration structures against the current sensor state."""
        if self._model is None or self._sensor_state is None:
            return
        if self._model.shape_count > 0 and self._model.bvh_shapes is not None:
            self._model.bvh_refit_shapes(self._sensor_state)
        if self._model.particle_count > 0 and self._model.bvh_particles is not None:
            self._model.bvh_refit_particles(self._sensor_state)

    def _capture_sensor_graph(self) -> None:
        """Capture BVH refits and registered scene queries once on CUDA."""
        device = self._sim.cfg.device
        with wp.ScopedDevice(device):
            self._refit_sensor_bvhs()
            for update_fn in self._sensor_tasks.values():
                update_fn()

        flags = wp.zeros(1 + len(self._sensor_tasks), dtype=wp.int32, device=device)
        flags_host = np.zeros(1 + len(self._sensor_tasks), dtype=np.int32)
        update_fns = tuple(self._sensor_tasks.values())

        def pipeline() -> None:
            wp.capture_if(flags[0:1], self._refit_sensor_bvhs)
            for index, update_fn in enumerate(update_fns):
                wp.capture_if(flags[index + 1 : index + 2], update_fn)

        with wp.ScopedCapture(device=device) as capture:
            pipeline()
        if capture.graph is None:
            raise RuntimeError("Newton sensor CUDA capture produced no graph.")
        self._sensor_capture = (capture.graph, flags, flags_host)
        logger.info("Captured Newton sensor graph with %d task(s).", len(self._sensor_tasks))

    def replicate(self, plan: ClonePlan) -> None:
        """Build the shared Newton resource directly from a completed clone plan."""
        rows = tuple(
            row
            for row, destination in enumerate(plan.destinations)
            if "{}" in destination and bool(plan.clone_mask[row].any())
        )
        _replicate_plan(self, plan, self._sim.stage, rows)
        if self._physics_cfg is None:
            self.finalize_visualization_model()


def newton_physics_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: np.ndarray,
    mapping: np.ndarray,
    positions: np.ndarray | None = None,
    quaternions: np.ndarray | None = None,
    up_axis: str = "Z",
    global_paths: tuple[str, ...] = (),
) -> tuple[ModelBuilder, dict[str, Any]]:
    """Build a Newton model directly from one raw NumPy clone mapping.

    This low-level entry point shares the active simulation's Newton resource but does not
    enqueue or dispatch a second clone lifecycle.

    Args:
        stage: USD stage containing the source assets.
        sources: Source prim paths used for cloning.
        destinations: Destination path templates containing ``{}`` for environment ids.
        env_ids: Environment ids in clone-column order.
        mapping: Boolean source-to-environment mapping, shape ``[len(sources), len(env_ids)]``.
        positions: Per-environment world positions [m], shape ``[len(env_ids), 3]``.
        quaternions: Per-environment xyzw orientations, shape ``[len(env_ids), 4]``.
        up_axis: Newton model up axis.
        global_paths: Shared scene roots imported once into Newton world ``-1``.

    Returns:
        The populated builder and empty legacy stage metadata.
    """
    from isaaclab.sim import SimulationContext  # noqa: PLC0415

    env_ids = np.asarray(env_ids)
    mapping = np.asarray(mapping, dtype=np.bool_)
    if mapping.shape != (len(sources), len(env_ids)):
        raise ValueError(f"mapping must have shape {(len(sources), len(env_ids))}, got {mapping.shape}.")
    if positions is None:
        positions = np.zeros((len(env_ids), 3), dtype=np.float32)
    else:
        positions = np.asarray(positions, dtype=np.float32)
    if quaternions is not None:
        quaternions = np.asarray(quaternions, dtype=np.float32)
    global_mask = np.zeros((len(global_paths), len(env_ids)), dtype=np.bool_)
    plan = ClonePlan(
        sources=(*sources, *global_paths),
        destinations=(*destinations, *global_paths),
        clone_mask=np.concatenate((mapping, global_mask)),
        env_ids=env_ids,
        positions=positions,
        global_paths=global_paths,
        is_complete=True,
        _env_ids_cpu=tuple(map(int, env_ids)),
        _positions_cpu=positions,
    )
    sim = SimulationContext.instance()
    if sim is None:
        raise RuntimeError("Newton replication requires an active SimulationContext.")
    resource = sim.get_or_create_backend(NewtonReplicateContext, sim)
    resource._up_axis = up_axis
    builder = _replicate_plan(resource, plan, stage, range(len(sources)), quaternions=quaternions)
    return builder, {}
