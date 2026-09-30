# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clone-plan publication and dispatch."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any

import numpy as np

from isaaclab.sim import SimulationContext

from ._fabric_notices import disabled_fabric_change_notifies
from .clone_plan import ClonePlan, make_clone_plan
from .cloner_cfg import DEFAULT_ENV_TEMPLATE
from .cloner_strategies import sequential
from .scene_layout import declare_scene_layout
from .usd import UsdReplicateContext


def replicate(plan: ClonePlan, *, replicate_physics: bool = True) -> ClonePlan:
    """Complete and dispatch the active plan once to every registered clone backend."""
    sim = SimulationContext.instance()
    if sim is None:
        raise RuntimeError("Clone-plan replication requires an active SimulationContext.")
    if sim.get_clone_plan() is not plan:
        raise ValueError("replicate() requires the active SimulationContext's ClonePlan.")

    if plan.root_layer_identifier is None:
        raise ValueError("replicate() requires a ClonePlan published for the active stage.")
    plan = declare_scene_layout(plan, sim.stage)
    sim.set_clone_plan(plan)

    contexts = [
        sim._backend_registry[backend_type]
        for backend_type, roles in sim._backend_clone_roles.items()
        if replicate_physics or roles != {"physics"}
    ]
    with disabled_fabric_change_notifies(sim.stage):
        for context in sorted(contexts, key=lambda item: item.replicate_priority):
            context.replicate(plan)
        for renderer in sim._renderer_entries:
            renderer.prepare_stage(sim.stage, plan)
    return plan


class ReplicateSession:
    """Own one clone plan and dispatch it from a context block.

    Interactive scenes use this internally. Standalone tools may use it for heterogeneous
    planning when an interactive scene is not a suitable dependency.
    """

    def __init__(
        self,
        cfgs: Iterable[Any],
        num_clones: int,
        env_spacing: float,
        *,
        global_paths: Iterable[str] = (),
        geometry_prim_paths: Iterable[str] = (),
        clone_strategy: Callable[[np.ndarray, int], np.ndarray] = sequential,
        valid_set: np.ndarray | None = None,
        replicate_physics: bool = True,
        env_template: str = DEFAULT_ENV_TEMPLATE,
    ):
        """Capture the explicit configuration manifest for one cloning lifecycle."""
        self._cfgs = tuple(cfgs)
        self._replicate_physics = replicate_physics
        self._env_template = env_template
        self._kwargs = dict(
            num_clones=num_clones,
            env_spacing=env_spacing,
            global_paths=global_paths,
            geometry_prim_paths=geometry_prim_paths,
            clone_strategy=clone_strategy,
            valid_set=valid_set,
            env_template=env_template,
        )
        self._plan: ClonePlan | None = None

    def __enter__(self) -> ReplicateSession:
        from pxr import Gf, Sdf, UsdGeom, Vt  # noqa: PLC0415

        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("Clone planning requires an active SimulationContext.")
        if self._plan is not None or sim.get_clone_plan() is not None:
            raise RuntimeError("A SimulationContext owns exactly one clone lifecycle.")
        if not self._replicate_physics:
            sim.get_or_create_backend(UsdReplicateContext, sim.stage, clone_role="scene")

        self._plan = replace(
            make_clone_plan(self._cfgs, **self._kwargs),
            root_layer_identifier=sim.stage.GetRootLayer().identifier,
        )
        sim.set_clone_plan(self._plan)

        root_layer = sim.stage.GetRootLayer()
        UsdGeom.Xform.Define(sim.stage, self._env_template.rsplit("/", 1)[0])
        with Sdf.ChangeBlock():
            for env_id, position in zip(self._plan.env_ids, self._plan.positions, strict=True):
                path = self._env_template.format(int(env_id))
                root = Sdf.CreatePrimInLayer(root_layer, path)
                root.specifier = Sdf.SpecifierDef
                root.typeName = "Xform"
                translate = Sdf.AttributeSpec(root, "xformOp:translate", Sdf.ValueTypeNames.Double3)
                translate.default = Gf.Vec3d(*map(float, position))
                order = Sdf.AttributeSpec(root, "xformOpOrder", Sdf.ValueTypeNames.TokenArray)
                order.default = Vt.TokenArray(["xformOp:translate"])

        from isaaclab.markers import VisualizationMarkersCfg  # noqa: PLC0415

        marker_types = tuple(
            dict.fromkeys(
                visualizer.marker_type
                for visualizer in sim.visualizers
                if visualizer.cfg.enable_markers and visualizer.marker_type is not None
            )
        )
        marker_cfgs = tuple(cfg for cfg in self._cfgs if isinstance(cfg, VisualizationMarkersCfg))
        sim.vis_marker_registry.prepare(marker_cfgs, marker_types)
        for cfg in marker_cfgs:
            rows = self._plan.cfg_rows.get(id(cfg))
            if rows is None:
                raise ValueError(f"VisualizationMarkersCfg at {cfg.prim_path!r} has no clone-plan row.")
            for row in rows:
                source_path = self._plan.sources[row]
                if not sim.stage.GetPrimAtPath(source_path).IsValid():
                    UsdGeom.Xform.Define(sim.stage, source_path)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self._plan = replicate(self.plan, replicate_physics=self._replicate_physics)

    @property
    def plan(self) -> ClonePlan:
        """Return the plan produced on entry."""
        if self._plan is None:
            raise RuntimeError("ReplicateSession.plan is only available after entering the context.")
        return self._plan
