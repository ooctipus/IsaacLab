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
from isaaclab.utils.version import has_kit

from .clone_plan import _make_clone_plan
from .cloner_cfg import DEFAULT_ENV_TEMPLATE
from .query import clone_rows, whole_env_copy
from .usd import UsdReplicateContext

if TYPE_CHECKING:
    from .clone_plan import ClonePlan


def _replicate(plan: ClonePlan, *, replicate_physics: bool = True) -> None:
    """Dispatch ``plan`` once to every registered clone backend."""
    sim = SimulationContext.instance()
    if sim is None:
        raise RuntimeError("Clone-plan replication requires an active SimulationContext.")
    contexts = [
        sim._backend_registry[backend_type]
        for backend_type, roles in sim._backend_clone_roles.items()
        if replicate_physics or roles != {"physics"}
    ]
    rows = clone_rows(plan)
    sources = [plan.sources[row] for row in rows]
    destinations = [plan.destinations[row] for row in rows]
    mask = plan.clone_mask[rows]
    collapsed = whole_env_copy(plan)
    for context in sorted(contexts, key=lambda item: item.replicate_priority):
        if context.clones_whole_env and collapsed is not None:
            context.replicate([collapsed[0]], [collapsed[1]], plan.env_ids, mask[:1], positions=plan.positions)
        else:
            context.replicate(sources, destinations, plan.env_ids, mask, positions=plan.positions)
    sim._clone_plan_dispatched = True


class ReplicateSession:
    """Own and dispatch one clone plan around explicit scene construction.

    The session publishes the plan on entry, before cfg-owned constructors author their exact
    prototype paths, and dispatches that same plan to registered backends on exit.
    """

    def __init__(
        self,
        cfgs: Iterable[Any],
        num_clones: int,
        env_spacing: float,
        device: str,
        *,
        replicate_physics: bool = True,
        env_template: str = DEFAULT_ENV_TEMPLATE,
    ):
        """Capture the declarative scene inputs for one cloning lifecycle.

        Args:
            cfgs: Resolved configuration roots. Every nested prim-authoring cfg is planned.
            num_clones: Number of target environments.
            env_spacing: Grid spacing between environment origins [m].
            device: Torch device for plan tensors.
            replicate_physics: Whether contexts used only by physics receive the plan.
            env_template: Replicated environment path template; ``{}`` marks the environment index.
        """
        self._roots = tuple(cfgs)
        self._replicate_physics = replicate_physics
        self._kwargs = {
            "num_clones": num_clones,
            "env_spacing": env_spacing,
            "device": device,
            "env_template": env_template,
        }
        self._plan: ClonePlan | None = None

    def __enter__(self) -> ReplicateSession:
        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("ReplicateSession requires an active SimulationContext.")
        if self._plan is not None or sim.get_clone_plan() is not None:
            raise RuntimeError("A SimulationContext owns exactly one ReplicateSession lifecycle.")
        if has_kit() or not self._replicate_physics:
            sim.get_or_create_backend(UsdReplicateContext, sim.stage, clone_role="scene")

        self._plan = _make_clone_plan(self._roots, **self._kwargs)
        sim.set_clone_plan(self._plan)

        env_template = self._kwargs["env_template"]
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
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            assert self._plan is not None
            _replicate(self._plan, replicate_physics=self._replicate_physics)
