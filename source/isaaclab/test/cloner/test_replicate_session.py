# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture and dispatch tests for the single clone-plan lifecycle."""

from types import SimpleNamespace

import pytest
import torch

from pxr import Usd

import isaaclab.cloner as cloner
import isaaclab.cloner.replicate_session as replicate_session
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.cloner import ClonePlan
from isaaclab.cloner.clone_plan import _grid_positions
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners import MultiAssetSpawnerCfg, MultiUsdFileCfg, SpawnerCfg


class _Context:
    clones_whole_env = False
    replicate_priority = 0

    def __init__(self, name: str, calls: list[str] | None = None):
        self.name = name
        self.calls = calls
        self.mappings = []
        self.replicated = 0

    def replicate(self, sources, destinations, env_ids, mask, *, positions=None):
        self.mappings.append((tuple(sources), tuple(destinations), mask.clone()))
        self.replicated += 1
        if self.calls is not None:
            self.calls.append(self.name)


def _plan(rows: int = 2) -> ClonePlan:
    return ClonePlan(
        sources=tuple(f"/World/envs/env_0/Asset_{row}" for row in range(rows)),
        destinations=tuple(f"/World/envs/env_{{}}/Asset_{row}" for row in range(rows)),
        clone_mask=torch.ones((rows, 3), dtype=torch.bool),
        env_ids=torch.arange(3, dtype=torch.long),
        positions=torch.zeros((3, 3)),
    )


def _simulation(registry=None, roles=None, stage=None):
    simulation = SimpleNamespace(
        _backend_registry={} if registry is None else registry,
        _backend_clone_roles={} if roles is None else roles,
        _clone_plan=None,
        _clone_plan_dispatched=False,
        stage=stage,
    )
    simulation.get_clone_plan = lambda: simulation._clone_plan
    simulation.set_clone_plan = lambda plan: setattr(simulation, "_clone_plan", plan)
    simulation.get_or_create_backend = lambda backend_type, *args, **kwargs: SimulationContext.get_or_create_backend(
        simulation, backend_type, *args, **kwargs
    )
    return simulation


def test_cfgs_and_consumers_do_not_own_clone_lifecycle():
    """Cfgs carry data; the plan and simulation registry own clone dispatch."""
    assert "cloning_contexts" not in AssetBaseCfg.__dataclass_fields__
    assert "cloning_contexts" not in SensorBaseCfg.__dataclass_fields__
    assert "spawn_path" not in SpawnerCfg.__dataclass_fields__
    assert "spawn_paths" not in MultiAssetSpawnerCfg.__dataclass_fields__
    assert "spawn_paths" not in MultiUsdFileCfg.__dataclass_fields__
    assert "random_choice" not in MultiAssetSpawnerCfg.__dataclass_fields__
    assert "random_choice" not in MultiUsdFileCfg.__dataclass_fields__
    assert not hasattr(AssetBaseCfg, "_post_spawn")
    assert not hasattr(ArticulationCfg, "_post_spawn")
    assert not hasattr(cloner, "REPLICATION_QUEUE")
    assert not hasattr(cloner, "queue_replication")


def test_replicate_session_authors_every_environment_root(monkeypatch):
    """The composition root authors environment frames before entity construction."""
    stage = Usd.Stage.CreateInMemory()
    simulation = _simulation(stage=stage)
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    session = cloner.ReplicateSession([], 4, 0.5, "cpu")
    session.__enter__()

    for env_id, position in enumerate(_grid_positions(4, 0.5)):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}")
        assert prim.IsValid()
        assert tuple(prim.GetAttribute("xformOp:translate").Get()) == tuple(float(value) for value in position)


@pytest.mark.parametrize(("replicate_physics", "kit_available"), [(False, False), (True, True)])
def test_session_registers_one_required_usd_scene_context(monkeypatch, replicate_physics, kit_available):
    """USD-only and Kit cloning register one scene context."""
    stage = Usd.Stage.CreateInMemory()
    simulation = _simulation(stage=stage)
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    monkeypatch.setattr(replicate_session, "_replicate", lambda *args, **kwargs: None)
    monkeypatch.setattr(replicate_session, "has_kit", lambda: kit_available)

    with cloner.ReplicateSession([], 2, 1.0, "cpu", replicate_physics=replicate_physics):
        pass

    assert tuple(simulation._backend_registry) == (cloner.UsdReplicateContext,)
    assert simulation._backend_clone_roles == {cloner.UsdReplicateContext: {"scene"}}


def test_replicate_session_rejects_a_second_lifecycle(monkeypatch):
    """One simulation cannot publish a second plan or re-enter its session."""
    stage = Usd.Stage.CreateInMemory()
    simulation = _simulation(stage=stage)
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    first = cloner.ReplicateSession([], 1, 0.0, "cpu")

    with first:
        pass

    with pytest.raises(RuntimeError, match="exactly one ReplicateSession lifecycle"):
        first.__enter__()
    with pytest.raises(RuntimeError, match="exactly one ReplicateSession lifecycle"):
        cloner.ReplicateSession([], 1, 0.0, "cpu").__enter__()


def test_replicate_dispatches_each_registered_context_once(monkeypatch):
    """Per-asset and whole-environment backends consume the same plan once."""
    per_asset = _Context("per_asset")
    whole_env = _Context("whole_env")
    whole_env.clones_whole_env = True
    per_asset_type = type("PerAsset", (), {})
    whole_env_type = type("WholeEnv", (), {})
    simulation = _simulation(
        {per_asset_type: per_asset, whole_env_type: whole_env},
        {per_asset_type: {"physics", "scene"}, whole_env_type: {"scene"}},
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    plan = _plan()

    replicate_session._replicate(plan)

    assert per_asset.mappings[0][:2] == (plan.sources, plan.destinations)
    assert whole_env.mappings[0][0] == ("/World/envs/env_0",)
    assert per_asset.replicated == whole_env.replicated == 1
    assert simulation._clone_plan_dispatched


def test_replicate_does_not_dispatch_inactive_variant_rows(monkeypatch):
    """A bookkeeping slot with no environments must not ask a backend to parse a missing prim."""
    context = _Context("context")
    context_type = type("Context", (), {})
    simulation = _simulation({context_type: context}, {context_type: {"scene"}})
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    plan = _plan()
    plan.clone_mask[1] = False

    replicate_session._replicate(plan)

    sources, destinations, mask = context.mappings[0]
    assert sources == plan.sources[:1]
    assert destinations == plan.destinations[:1]
    assert torch.equal(mask, plan.clone_mask[:1])


def test_replicate_physics_false_skips_only_physics_only_contexts(monkeypatch):
    """A backend shared with a scene consumer still receives USD-only cloning."""
    physics = _Context("physics")
    shared = _Context("shared")
    scene = _Context("scene")
    physics_type = type("Physics", (), {})
    shared_type = type("Shared", (), {})
    scene_type = type("Scene", (), {})
    simulation = _simulation(
        {physics_type: physics, shared_type: shared, scene_type: scene},
        {physics_type: {"physics"}, shared_type: {"physics", "scene"}, scene_type: {"scene"}},
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    replicate_session._replicate(_plan(1), replicate_physics=False)

    assert physics.replicated == 0
    assert shared.replicated == scene.replicated == 1


def test_replicate_orders_contexts_by_priority(monkeypatch):
    calls = []
    late = _Context("late", calls)
    late.replicate_priority = 10
    early = _Context("early", calls)
    early.replicate_priority = -10
    late_type = type("Late", (), {})
    early_type = type("Early", (), {})
    simulation = _simulation(
        {late_type: late, early_type: early},
        {late_type: {"scene"}, early_type: {"scene"}},
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    replicate_session._replicate(_plan(1))

    assert calls == ["early", "late"]
