# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture and dispatch tests for clone-plan replication."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pxr import Usd

import isaaclab.cloner as cloner
import isaaclab.cloner.replicate_session as replicate_session
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import ClonePlan
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners import MultiAssetSpawnerCfg, MultiUsdFileCfg, SpawnerCfg


class _Context:
    replicate_priority = 0

    def __init__(self, name: str, calls: list[tuple[str, ClonePlan]]):
        self.name = name
        self.calls = calls

    def replicate(self, plan: ClonePlan):
        self.calls.append((self.name, plan))


def _plan(rows: int = 2, *, complete: bool = False) -> ClonePlan:
    return ClonePlan(
        sources=tuple(f"/World/envs/env_0/Asset_{row}" for row in range(rows)),
        destinations=tuple(f"/World/envs/env_{{}}/Asset_{row}" for row in range(rows)),
        clone_mask=np.ones((rows, 3), dtype=np.bool_),
        env_ids=np.arange(3, dtype=np.int64),
        positions=np.zeros((3, 3), dtype=np.float32),
        root_layer_identifier="stage",
        is_complete=complete,
        _env_ids_cpu=(0, 1, 2) if complete else (),
    )


def _session_simulation(stage: Usd.Stage, *, visualizers=(), marker_registry=None):
    simulation = SimpleNamespace(
        stage=stage,
        visualizers=visualizers,
        vis_marker_registry=marker_registry or SimpleNamespace(prepare=lambda _cfgs, _types: None),
        clone_plan=None,
    )
    simulation.get_clone_plan = lambda: simulation.clone_plan
    simulation.set_clone_plan = lambda plan: setattr(simulation, "clone_plan", plan)
    return simulation


def _dispatch_simulation(plan: ClonePlan, registry: dict, roles: dict):
    prepared = []
    simulation = SimpleNamespace(
        stage=object(),
        _backend_registry=registry,
        _backend_clone_roles=roles,
        _renderer_entries=(SimpleNamespace(prepare_stage=lambda stage, value: prepared.append((stage, value))),),
        clone_plan=plan,
    )
    simulation.get_clone_plan = lambda: simulation.clone_plan
    simulation.set_clone_plan = lambda value: setattr(simulation, "clone_plan", value)
    return simulation, prepared


def test_cfgs_do_not_own_backend_routing_or_spawn_paths():
    """Backend routing belongs to the simulation registry, not cfg-side escape hatches."""
    assert "cloning_contexts" not in AssetBaseCfg.__dataclass_fields__
    assert "cloning_contexts" not in SensorBaseCfg.__dataclass_fields__
    assert "spawn_path" not in SpawnerCfg.__dataclass_fields__
    assert "spawn_paths" not in MultiAssetSpawnerCfg.__dataclass_fields__
    assert "spawn_paths" not in MultiUsdFileCfg.__dataclass_fields__
    assert not hasattr(cloner, "REPLICATION_QUEUE")
    assert not hasattr(cloner, "queue_replication")


@pytest.mark.parametrize("valid_set", [np.asarray([["0"]]), np.asarray([[0 + 1j]])])
def test_make_clone_plan_rejects_non_integer_combinations(valid_set):
    cfg = SimpleNamespace(prim_path="/World/envs/env_[^/]+/Robot", spawn=SimpleNamespace())
    with pytest.raises(ValueError, match="integer prototype indices"):
        cloner.make_clone_plan((cfg,), 2, 1.0, valid_set=valid_set)


def test_grid_transforms_always_returns_float32():
    positions, orientations = cloner.grid_transforms(2, np.float64(1.0))
    assert positions.dtype == orientations.dtype == np.float32


def test_replicate_dispatches_one_declared_plan_in_priority_order(monkeypatch: pytest.MonkeyPatch):
    calls = []

    class Late(_Context):
        replicate_priority = 1

    class Early(_Context):
        replicate_priority = -1

    plan = _plan()
    simulation, prepared = _dispatch_simulation(
        plan,
        {Late: Late("late", calls), Early: Early("early", calls)},
        {Late: {"physics"}, Early: {"scene"}},
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    monkeypatch.setattr(
        replicate_session, "declare_scene_layout", lambda value, _stage: replace(value, is_complete=True)
    )
    monkeypatch.setattr(replicate_session, "disabled_fabric_change_notifies", lambda _stage: nullcontext())

    declared = replicate_session.replicate(plan)

    assert calls == [("early", declared), ("late", declared)]
    assert simulation.clone_plan is declared
    assert prepared == [(simulation.stage, declared)]


def test_replicate_physics_false_skips_only_physics_only_contexts(monkeypatch: pytest.MonkeyPatch):
    calls = []

    class Physics(_Context):
        pass

    class Shared(_Context):
        pass

    class Scene(_Context):
        pass

    plan = _plan()
    simulation, _ = _dispatch_simulation(
        plan,
        {
            Physics: Physics("physics", calls),
            Shared: Shared("shared", calls),
            Scene: Scene("scene", calls),
        },
        {Physics: {"physics"}, Shared: {"physics", "scene"}, Scene: {"scene"}},
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    monkeypatch.setattr(
        replicate_session, "declare_scene_layout", lambda value, _stage: replace(value, is_complete=True)
    )
    monkeypatch.setattr(replicate_session, "disabled_fabric_change_notifies", lambda _stage: nullcontext())

    declared = replicate_session.replicate(plan, replicate_physics=False)

    assert calls == [("shared", declared), ("scene", declared)]


def test_replicate_requires_the_active_plan(monkeypatch: pytest.MonkeyPatch):
    active = _plan()
    simulation, _ = _dispatch_simulation(active, {}, {})
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    with pytest.raises(ValueError, match="active SimulationContext"):
        replicate_session.replicate(_plan())


def test_replicate_session_authors_environment_roots_from_its_plan(monkeypatch: pytest.MonkeyPatch):
    stage = Usd.Stage.CreateInMemory()
    simulation = _session_simulation(stage)
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    session = cloner.ReplicateSession((), num_clones=4, env_spacing=0.5)
    session.__enter__()

    assert simulation.clone_plan is session.plan
    for env_id, position in zip(session.plan.env_ids, session.plan.positions, strict=True):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}")
        assert prim.IsValid()
        assert tuple(prim.GetAttribute("xformOp:translate").Get()) == tuple(map(float, position))


def test_replicate_session_prepares_top_level_marker_cfg(monkeypatch: pytest.MonkeyPatch):
    """A scene-declared marker participates in the same plan before backend initialization."""

    class MarkerState:
        pass

    marker_cfg = VisualizationMarkersCfg(prim_path="/Visuals/Test", markers={"sphere": SimpleNamespace()})
    prepared = []
    stage = Usd.Stage.CreateInMemory()
    visualizer = SimpleNamespace(cfg=SimpleNamespace(enable_markers=True), marker_type=MarkerState)
    simulation = _session_simulation(
        stage,
        visualizers=(visualizer,),
        marker_registry=SimpleNamespace(prepare=lambda cfgs, types: prepared.append((cfgs, types))),
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    session = cloner.ReplicateSession((marker_cfg,), 1, 0.0)
    session.__enter__()

    assert session.plan.cfg_rows[id(marker_cfg)]
    assert prepared == [((marker_cfg,), (MarkerState,))]
    assert stage.GetPrimAtPath(marker_cfg.prim_path).IsValid()


def test_replicate_physics_false_registers_one_usd_resource(monkeypatch: pytest.MonkeyPatch):
    stage = Usd.Stage.CreateInMemory()
    simulation = _session_simulation(stage)
    simulation._backend_registry = {}
    simulation._backend_clone_roles = {}
    simulation._clone_plan = None
    simulation.get_or_create_backend = lambda backend_type, *args, **kwargs: SimulationContext.get_or_create_backend(
        simulation, backend_type, *args, **kwargs
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    cloner.ReplicateSession((), 1, 0.0, replicate_physics=False).__enter__()

    assert isinstance(simulation._backend_registry[cloner.UsdReplicateContext], cloner.UsdReplicateContext)
    assert simulation._backend_clone_roles == {cloner.UsdReplicateContext: {"scene"}}


def test_simulation_context_publishes_one_preliminary_then_declared_plan():
    simulation = object.__new__(SimulationContext)
    simulation._clone_plan = None
    simulation._renderers_initialized = False
    bound = []
    simulation._scene_data_provider = SimpleNamespace(_bind_point_plan=bound.append)
    preliminary = _plan()
    declared = replace(preliminary, is_complete=True, _env_ids_cpu=(0, 1, 2))

    simulation.set_clone_plan(preliminary)
    with pytest.raises(RuntimeError, match="exactly one clone-plan lifecycle"):
        simulation.set_clone_plan(replace(declared, positions=declared.positions.copy()))
    simulation.set_clone_plan(declared)

    assert simulation.get_clone_plan() is declared
    assert bound == [declared]
    with pytest.raises(RuntimeError, match="exactly one clone-plan lifecycle"):
        simulation.set_clone_plan(_plan(1))
