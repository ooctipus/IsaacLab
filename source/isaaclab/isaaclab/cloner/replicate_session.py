# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clone-plan dispatch and :class:`ReplicateSession` lifecycle."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

from pxr import Gf, Sdf, UsdGeom, Vt

from isaaclab.sim import SimulationContext

from .clone_plan import _make_clone_plan, _plan_cfgs
from .collision_filter import filter_collisions
from .path import under
from .usd import UsdReplicateContext

if TYPE_CHECKING:
    from .clone_plan import ClonePlan
    from .cloner_cfg import CloneCfg


_ACTIVE_PLAN: ContextVar[ClonePlan | None] = ContextVar("isaaclab_clone_plan", default=None)


def _active_plan() -> ClonePlan | None:
    """Return the plan in the current lexical clone lifecycle."""
    return _ACTIVE_PLAN.get()


def _replicate(plan: ClonePlan) -> None:
    """Dispatch stage cloning and queue model construction for first reset."""
    sim = SimulationContext.instance()
    if sim is None:
        raise RuntimeError("Clone-plan replication requires an active SimulationContext.")

    context_rows = plan.context_rows
    missing = [context_type for context_type in context_rows if context_type not in sim._backend_registry]
    if missing:
        names = ", ".join(f"{context_type.__module__}.{context_type.__qualname__}" for context_type in missing)
        raise RuntimeError(f"Clone contexts must be registered before session dispatch: {names}.")

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
        self._clone_cfg: CloneCfg | None = None
        self._active_plan_token: Token[ClonePlan | None] | None = None

    def __enter__(self) -> ReplicateSession:
        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("ReplicateSession requires an active SimulationContext.")
        if self._plan is not None or sim.get_clone_plan() is not None or _active_plan() is not None:
            raise RuntimeError("A SimulationContext owns exactly one clone lifecycle.")

        plan_cfgs = _plan_cfgs((*self._roots, sim.cfg, *sim._get_visualizer_cfgs()))
        clone_cfg, cfg_contexts, scene_contexts = plan_cfgs[2:]
        self._clone_cfg = clone_cfg
        explicit_contexts = {
            context_type for contexts in cfg_contexts.values() if contexts is not None for context_type in contexts
        } | set(scene_contexts)
        if UsdReplicateContext in explicit_contexts and UsdReplicateContext not in sim._backend_registry:
            sim.get_or_create_backend(UsdReplicateContext, sim.stage)
        if UsdReplicateContext in scene_contexts or not clone_cfg.replicate_physics:
            sim.get_or_create_backend(UsdReplicateContext, sim.stage, clone_role="scene")

        context_roles = {
            context_type: set(sim._backend_clone_roles.get(context_type, ())) for context_type in sim._backend_registry
        }
        self._plan = _make_clone_plan(
            *plan_cfgs,
            num_clones=self._num_clones,
            env_spacing=self._env_spacing,
            device=sim.device,
            context_roles=context_roles,
        )
        sim.set_clone_plan(self._plan)

        env_template = clone_cfg.clone_template
        root_layer = sim.stage.GetRootLayer()
        UsdGeom.Xform.Define(sim.stage, env_template.rsplit("/", 1)[0])
        with Sdf.ChangeBlock():
            for env_id, position in zip(self._plan.env_ids.tolist(), self._plan.positions.cpu().tolist(), strict=True):
                root = Sdf.CreatePrimInLayer(root_layer, env_template.format(env_id))
                root.specifier = Sdf.SpecifierDef
                root.typeName = "Xform"
                translate = Sdf.AttributeSpec(root, "xformOp:translate", Sdf.ValueTypeNames.Double3)
                translate.default = Gf.Vec3d(*position)
                order = Sdf.AttributeSpec(root, "xformOpOrder", Sdf.ValueTypeNames.TokenArray)
                order.default = Vt.TokenArray(["xformOp:translate"])
        self._active_plan_token = _ACTIVE_PLAN.set(self._plan)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if exc_type is None:
                assert self._plan is not None and self._clone_cfg is not None
                _replicate(self._plan)
                sim = SimulationContext.instance()
                assert sim is not None
                physics_contexts = (
                    sim._backend_registry[context_type]
                    for context_type, roles in sim._backend_clone_roles.items()
                    if "physics" in roles
                )
                if self._clone_cfg.filter_collisions and any(
                    context.uses_physx_collision_groups for context in physics_contexts
                ):
                    filter_collisions(
                        sim.stage,
                        sim.cfg.physics_prim_path,
                        "/World/collisions",
                        [self._plan.env_template.format(int(env_id)) for env_id in self._plan.env_ids.tolist()],
                        list(self._plan.collision_paths),
                    )
        finally:
            if self._active_plan_token is not None:
                _ACTIVE_PLAN.reset(self._active_plan_token)
                self._active_plan_token = None


@contextmanager
def from_env_0(cfg: Any, num_envs: int, env_spacing: float) -> Iterator[None]:
    """Construct one homogeneous environment prototype and replicate it to every environment.

    The complete configuration root is planned before entering the scope. Every environment-scoped
    cfg must therefore describe the same prototype; heterogeneous or multi-variant layouts belong in
    an :class:`~isaaclab.scene.InteractiveSceneCfg` instead.

    Args:
        cfg: Complete configuration root containing the clone policy and every prim author.
        num_envs: Number of target environments.
        env_spacing: Grid spacing between environment origins [m].
    """
    with ReplicateSession((cfg,), num_envs, env_spacing) as session:
        assert session._plan is not None
        plan = session._plan
        source_env = plan.env_template.format(int(plan.env_ids[0]))
        env_rows = [row for row, destination in enumerate(plan.destinations) if "{}" in destination]
        if any(not bool(plan.clone_mask[row].all()) or not under(plan.sources[row], source_env) for row in env_rows):
            raise ValueError("from_env_0 requires one homogeneous cfg-derived environment prototype.")
        yield
