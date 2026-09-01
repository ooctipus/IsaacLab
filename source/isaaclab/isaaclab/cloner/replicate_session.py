# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clone-plan dispatch and :class:`ReplicateSession` lifecycle."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from pxr import Gf, Sdf, UsdGeom, Vt

from isaaclab.sim import SimulationContext

from .clone_plan import _make_clone_plan, _plan_cfgs
from .collision_filter import filter_collisions
from .path import under
from .usd import UsdReplicateContext

if TYPE_CHECKING:
    from .clone_plan import ClonePlan


def _build_plan(cfgs: Iterable[Any], num_clones: int, env_spacing: float, *, homogeneous: bool = False) -> ClonePlan:
    """Build one cfg-derived plan and register every context it declares."""
    sim = SimulationContext.instance()
    if sim is None:
        raise RuntimeError("Clone planning requires an active SimulationContext.")
    if sim.get_clone_plan() is not None:
        raise RuntimeError("A SimulationContext owns exactly one clone lifecycle.")

    plan_cfgs = _plan_cfgs((*cfgs, sim.cfg, *sim._get_visualizer_cfgs()))
    clone_cfg, cfg_contexts, scene_contexts = plan_cfgs[2:]
    explicit_contexts = {
        context_type for contexts in cfg_contexts.values() if contexts is not None for context_type in contexts
    } | set(scene_contexts)
    context_roles = {
        context_type: set(sim._backend_clone_roles.get(context_type, ())) for context_type in sim._backend_registry
    }
    needs_usd_scene = UsdReplicateContext in scene_contexts or not clone_cfg.replicate_physics
    if UsdReplicateContext in explicit_contexts or needs_usd_scene:
        context_roles.setdefault(UsdReplicateContext, set())
    if needs_usd_scene:
        context_roles[UsdReplicateContext].add("scene")

    plan = _make_clone_plan(
        *plan_cfgs,
        num_clones=num_clones,
        env_spacing=env_spacing,
        device=sim.device,
        context_roles=context_roles,
    )
    if homogeneous:
        source_env = plan.env_template.format(int(plan.env_ids[0]))
        env_rows = [row for row, destination in enumerate(plan.destinations) if "{}" in destination]
        if any(not bool(plan.clone_mask[row].all()) or not under(plan.sources[row], source_env) for row in env_rows):
            raise ValueError("clone_plan_from_env_0 requires one homogeneous cfg-derived environment prototype.")
    if UsdReplicateContext in explicit_contexts or needs_usd_scene:
        sim.get_or_create_backend(UsdReplicateContext, sim.stage, clone_role="scene" if needs_usd_scene else None)
    return plan


def _publish_plan(plan: ClonePlan) -> None:
    """Publish a plan and author its environment frames before prototype construction."""
    sim = SimulationContext.instance()
    if sim is None:
        raise RuntimeError("Clone planning requires an active SimulationContext.")
    sim.set_clone_plan(plan)
    sim._clone_plan_dispatched = None

    root_layer = sim.stage.GetRootLayer()
    UsdGeom.Xform.Define(sim.stage, plan.env_template.rsplit("/", 1)[0])
    with Sdf.ChangeBlock():
        for env_id, position in zip(plan.env_ids.tolist(), plan.positions.cpu().tolist(), strict=True):
            root = Sdf.CreatePrimInLayer(root_layer, plan.env_template.format(env_id))
            root.specifier = Sdf.SpecifierDef
            root.typeName = "Xform"
            translate = Sdf.AttributeSpec(root, "xformOp:translate", Sdf.ValueTypeNames.Double3)
            translate.default = Gf.Vec3d(*position)
            order = Sdf.AttributeSpec(root, "xformOpOrder", Sdf.ValueTypeNames.TokenArray)
            order.default = Vt.TokenArray(["xformOp:translate"])
    sim._clone_plan_dispatched = False


def replicate(plan: ClonePlan) -> None:
    """Replicate one published plan after its prototypes have been constructed.

    Args:
        plan: Plan returned by :func:`clone_plan_from_env_0`.

    Raises:
        RuntimeError: If no simulation is active, the plan cannot be dispatched, or a routed context is unregistered.
        ValueError: If ``plan`` is not the active simulation's published plan.
    """
    sim = SimulationContext.instance()
    if sim is None:
        raise RuntimeError("Clone-plan replication requires an active SimulationContext.")
    if sim.get_clone_plan() is not plan:
        raise ValueError("replicate() requires the plan published on the active SimulationContext.")
    if sim._clone_plan_dispatched is not False:
        raise RuntimeError("A clone plan can be replicated exactly once.")

    # Dispatch is a one-shot transition. A failure leaves the simulation unusable rather than
    # retrying a partially replicated plan.
    sim._clone_plan_dispatched = None
    context_rows = plan.context_rows
    missing = [context_type for context_type in context_rows if context_type not in sim._backend_registry]
    if missing:
        names = ", ".join(f"{context_type.__module__}.{context_type.__qualname__}" for context_type in missing)
        raise RuntimeError(f"Clone contexts must be registered before plan dispatch: {names}.")

    contexts = []
    model_contexts = []
    for context_type in context_rows:
        roles = sim._backend_clone_roles.get(context_type, set())
        context = sim._backend_registry[context_type]
        if "model" in roles:
            model_contexts.append(context)
        elif not roles or "scene" in roles or (plan.replicate_physics and "physics" in roles):
            contexts.append(context)

    for context in sorted(contexts, key=lambda item: item.replicate_priority):
        context.replicate(plan)
    sim._pending_clone_model_contexts = tuple(sorted(model_contexts, key=lambda item: item.replicate_priority))

    physics_contexts = (
        sim._backend_registry[context_type]
        for context_type, roles in sim._backend_clone_roles.items()
        if "physics" in roles
    )
    if plan.filter_collisions and any(context.uses_physx_collision_groups for context in physics_contexts):
        filter_collisions(
            sim.stage,
            sim.cfg.physics_prim_path,
            "/World/collisions",
            [plan.env_template.format(int(env_id)) for env_id in plan.env_ids.tolist()],
            list(plan.collision_paths),
        )
    sim._clone_plan_dispatched = True


def clone_plan_from_env_0(cfg: Any, num_envs: int, env_spacing: float) -> ClonePlan:
    """Build and publish one homogeneous cfg-derived plan before prototype construction.

    Args:
        cfg: Complete configuration root containing the clone policy and every prim author.
        num_envs: Number of target environments.
        env_spacing: Grid spacing between environment origins [m].

    Returns:
        The plan to pass to :func:`replicate` after constructing the environment-zero prototypes.

    Raises:
        RuntimeError: If no simulation is active or it already owns a clone plan.
        ValueError: If ``cfg`` does not describe one homogeneous environment prototype.
    """
    plan = _build_plan((cfg,), num_envs, env_spacing, homogeneous=True)
    _publish_plan(plan)
    return plan


class ReplicateSession:
    """Own one clone plan around explicit scene construction.

    The session publishes the plan on entry, before cfg-owned constructors author their exact
    prototype paths. On exit it materializes the stage and queues model construction for the first
    hard reset, after any intervening stage edits.
    """

    def __init__(self, cfgs: Iterable[Any], num_clones: int, env_spacing: float):
        """Capture the declarative scene inputs for one cloning lifecycle.

        Args:
            cfgs: Resolved configuration roots. Every nested prim-authoring cfg is planned.
            num_clones: Number of target environments.
            env_spacing: Grid spacing between environment origins [m].
        """
        self._roots = tuple(cfgs)
        self._num_clones = num_clones
        self._env_spacing = env_spacing
        self._plan: ClonePlan | None = None

    def __enter__(self) -> ReplicateSession:
        if self._plan is not None:
            raise RuntimeError("A SimulationContext owns exactly one clone lifecycle.")
        self._plan = _build_plan(self._roots, self._num_clones, self._env_spacing)
        _publish_plan(self._plan)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            assert self._plan is not None
            replicate(self._plan)
        else:
            sim = SimulationContext.instance()
            if sim is not None and sim.get_clone_plan() is self._plan:
                sim._clone_plan_dispatched = None
