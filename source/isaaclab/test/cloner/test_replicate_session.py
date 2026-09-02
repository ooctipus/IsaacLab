# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture and dispatch tests for the single clone-plan lifecycle."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from pxr import Usd

import isaaclab.cloner as cloner
import isaaclab.cloner.replicate_session as replicate_session
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.cloner import ClonePlan
from isaaclab.cloner.clone_plan import _grid_positions, _make_clone_plan, _plan_cfgs
from isaaclab.renderers import RendererCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners import MultiAssetSpawnerCfg, MultiUsdFileCfg, SpawnerCfg


class _Context:
    replicate_priority = 0
    uses_physx_collision_groups = False

    def __init__(self, name: str, calls: list[str] | None = None, whole_env: bool = False):
        self.name = name
        self.calls = calls
        self.mappings = []
        self.replicated = 0
        self.whole_env = whole_env

    def replicate(self, plan):
        sources, destinations, mask = cloner.query._clone_mapping(
            plan, plan.context_rows[type(self)], whole_env=self.whole_env
        )
        self.mappings.append((tuple(sources), tuple(destinations), mask.clone()))
        self.replicated += 1
        if self.calls is not None:
            self.calls.append(self.name)


def _plan(rows: int = 2, *, replicate_physics: bool = True) -> ClonePlan:
    return ClonePlan(
        sources=tuple(f"/World/envs/env_0/Asset_{row}" for row in range(rows)),
        destinations=tuple(f"/World/envs/env_{{}}/Asset_{row}" for row in range(rows)),
        clone_mask=torch.ones((rows, 3), dtype=torch.bool),
        env_ids=torch.arange(3, dtype=torch.long),
        positions=torch.zeros((3, 3)),
        replicate_physics=replicate_physics,
    )


def _simulation(registry=None, roles=None, stage=None):
    simulation = SimpleNamespace(
        _backend_registry={} if registry is None else registry,
        _backend_clone_roles={} if roles is None else roles,
        _clone_plan=None,
        _clone_plan_dispatched=False,
        _pending_clone_model_contexts=(),
        device="cpu",
        stage=stage,
        cfg=SimpleNamespace(physics_prim_path="/physicsScene"),
    )
    simulation.get_clone_plan = lambda: simulation._clone_plan
    simulation.set_clone_plan = lambda plan: SimulationContext.set_clone_plan(simulation, plan)
    simulation.get_or_create_backend = lambda backend_type, *args, **kwargs: SimulationContext.get_or_create_backend(
        simulation, backend_type, *args, **kwargs
    )
    simulation._get_visualizer_cfgs = lambda: []
    return simulation


def test_cfgs_and_consumers_do_not_own_clone_lifecycle():
    """Cfgs carry data; the plan and simulation registry own clone dispatch."""
    assert AssetBaseCfg(prim_path="/World/Asset").cloning_contexts is None
    assert SensorBaseCfg(class_type=object, prim_path="/World/Sensor").cloning_contexts == ()
    assert "spawn_path" not in SpawnerCfg.__dataclass_fields__
    assert "spawn_paths" not in MultiAssetSpawnerCfg.__dataclass_fields__
    assert "spawn_paths" not in MultiUsdFileCfg.__dataclass_fields__
    assert "random_choice" not in MultiAssetSpawnerCfg.__dataclass_fields__
    assert "random_choice" not in MultiUsdFileCfg.__dataclass_fields__
    assert not hasattr(AssetBaseCfg, "_post_spawn")
    assert not hasattr(ArticulationCfg, "_post_spawn")
    assert not hasattr(cloner, "REPLICATION_QUEUE")
    assert not hasattr(cloner, "queue_replication")


def test_clone_planning_does_not_depend_on_registered_scene():
    """A generic clone plan does not validate itself through an InteractiveScene."""
    assert not hasattr(SimulationContext, "_validate_clone_plan")


def test_clone_plan_from_env_0_requires_explicit_cfgs(monkeypatch):
    """A Direct environment cfg is not an implicit clone-participant inventory."""

    @dataclass
    class DirectCfg:
        class_type: object
        robot_cfg: object

    @dataclass
    class NonAuthorCfg:
        class_type: object

    simulation = _simulation(stage=Usd.Stage.CreateInMemory())
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    direct_cfg = DirectCfg(object, AssetBaseCfg(prim_path="/World/Asset"))
    with pytest.raises(TypeError, match="prim-authoring cfgs"):
        cloner.clone_plan_from_env_0(cloner.CloneCfg(), [direct_cfg], 2, 1.0)
    with pytest.raises(TypeError, match="prim-authoring cfgs"):
        cloner.clone_plan_from_env_0(cloner.CloneCfg(), [NonAuthorCfg(object)], 2, 1.0)
    with pytest.raises(TypeError, match="clone_cfg must be a CloneCfg"):
        cloner.clone_plan_from_env_0(AssetBaseCfg(prim_path="/World/Asset"), (), 2, 1.0)
    with pytest.raises(TypeError, match="not CloneCfg"):
        cloner.clone_plan_from_env_0(cloner.CloneCfg(), [cloner.CloneCfg()], 2, 1.0)

    assert simulation.get_clone_plan() is None


def test_replicate_session_authors_every_environment_root(monkeypatch):
    """The composition root authors environment frames before entity construction."""
    stage = Usd.Stage.CreateInMemory()
    simulation = _simulation(stage=stage)
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    with cloner.ReplicateSession([cloner.CloneCfg()], 4, 0.5):
        for env_id, position in enumerate(_grid_positions(4, 0.5)):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}")
            assert prim.IsValid()
            assert tuple(prim.GetAttribute("xformOp:translate").Get()) == tuple(float(value) for value in position)


def test_replicate_session_exception_makes_partial_plan_non_dispatchable(monkeypatch):
    """Failed construction cannot later replicate a partial scene."""
    simulation = _simulation(stage=Usd.Stage.CreateInMemory())
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    with pytest.raises(RuntimeError, match="construction failed"):
        with cloner.ReplicateSession([cloner.CloneCfg()], 2, 1.0):
            raise RuntimeError("construction failed")

    assert simulation._clone_plan_dispatched is None
    with pytest.raises(RuntimeError, match="exactly once"):
        cloner.replicate(simulation.get_clone_plan())


def test_plan_publication_failure_is_non_dispatchable(monkeypatch):
    """Partially authored environment frames cannot be replicated."""
    simulation = _simulation(stage=Usd.Stage.CreateInMemory())
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    def fail_publication(*_args):
        raise RuntimeError("publication failed")

    monkeypatch.setattr(replicate_session.Sdf, "CreatePrimInLayer", fail_publication)
    with pytest.raises(RuntimeError, match="publication failed"):
        cloner.clone_plan_from_env_0(cloner.CloneCfg(), (), 2, 1.0)

    assert simulation._clone_plan_dispatched is None
    with pytest.raises(RuntimeError, match="exactly once"):
        cloner.replicate(simulation.get_clone_plan())


def test_clone_plan_from_env_0_publishes_before_explicit_replication(monkeypatch):
    """Direct construction receives the plan before explicit dispatch."""
    context_type = type("SceneContext", (_Context,), {})
    context = context_type("scene")
    simulation = _simulation({context_type: context}, {context_type: {"scene"}}, Usd.Stage.CreateInMemory())
    asset_cfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        cloning_contexts=(context_type,),
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    plan = cloner.clone_plan_from_env_0(cloner.CloneCfg(), (asset_cfg,), 2, 1.0)

    assert simulation.get_clone_plan() is plan
    assert simulation._clone_plan_dispatched is False
    assert context.replicated == 0
    cloner.replicate(plan)
    assert simulation._clone_plan_dispatched is True
    assert context.replicated == 1


def test_clone_plan_from_env_0_accepts_an_empty_direct_scene(monkeypatch):
    """An empty direct environment is a valid homogeneous prototype."""
    simulation = _simulation(stage=Usd.Stage.CreateInMemory())
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    plan = cloner.clone_plan_from_env_0(cloner.CloneCfg(), (), 2, 1.0)
    cloner.replicate(plan)

    assert simulation._clone_plan_dispatched is True


def test_clone_plan_from_env_0_rejects_multi_variant_layout_before_publication(monkeypatch):
    """A heterogeneous cfg must use the InteractiveScene-owned lifecycle."""
    simulation = _simulation(stage=Usd.Stage.CreateInMemory())
    asset_cfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=MultiAssetSpawnerCfg(assets_cfg=[sim_utils.ConeCfg(), sim_utils.SphereCfg()]),
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    with pytest.raises(ValueError, match="one homogeneous cfg-derived environment prototype"):
        cloner.clone_plan_from_env_0(cloner.CloneCfg(replicate_physics=False), (asset_cfg,), 2, 1.0)
    assert simulation.get_clone_plan() is None
    assert simulation._backend_registry == {}
    assert simulation._backend_clone_roles == {}


def test_replicate_session_requires_one_clone_cfg(monkeypatch):
    """A direct workflow must declare clone policy instead of receiving hidden defaults."""
    simulation = _simulation(stage=Usd.Stage.CreateInMemory())
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    with pytest.raises(ValueError, match="requires one CloneCfg"):
        cloner.ReplicateSession([], 1, 0.0).__enter__()


@pytest.mark.parametrize(
    ("replicate_physics", "consumer"),
    [
        (False, None),
        (True, RendererCfg(cloning_contexts=(cloner.UsdReplicateContext,))),
    ],
)
def test_session_registers_one_required_usd_scene_context(monkeypatch, replicate_physics, consumer):
    """USD-only cloning and declarative whole-scene consumers register one scene context."""
    stage = Usd.Stage.CreateInMemory()
    simulation = _simulation(stage=stage)
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    monkeypatch.setattr(replicate_session, "replicate", lambda *args, **kwargs: None)

    roots = [cloner.CloneCfg(replicate_physics=replicate_physics)]
    if consumer is not None:
        roots.append(consumer)
    with cloner.ReplicateSession(roots, 2, 1.0):
        pass

    assert cloner.UsdReplicateContext in simulation._backend_registry
    assert simulation._backend_clone_roles[cloner.UsdReplicateContext] == {"scene"}


def test_session_does_not_infer_usd_context_from_runtime_capability(monkeypatch):
    """A clone backend is requested by cfg data, never by the installed runtime."""
    simulation = _simulation(stage=Usd.Stage.CreateInMemory())
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    monkeypatch.setattr(replicate_session, "replicate", lambda *args, **kwargs: None)

    with cloner.ReplicateSession([cloner.CloneCfg()], 2, 1.0):
        pass

    assert simulation._backend_registry == {}


@pytest.mark.parametrize(("filter_collisions", "expected_calls"), [(True, 1), (False, 0)])
def test_session_owns_physx_collision_filtering(monkeypatch, filter_collisions, expected_calls):
    """Session exit filters once only when the clone policy and backend contract require it."""
    stage = Usd.Stage.CreateInMemory()
    context_type = type("PhysxContext", (_Context,), {})
    context = context_type("physx")
    context.uses_physx_collision_groups = True
    simulation = _simulation({context_type: context}, {context_type: {"physics"}}, stage)
    simulation.cfg = SimpleNamespace(physics_prim_path="/physicsScene")
    calls = []
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    monkeypatch.setattr(replicate_session, "filter_collisions", lambda *args: calls.append(args))

    clone_cfg = cloner.CloneCfg(clone_template="/Scene/worlds/world_{}", filter_collisions=filter_collisions)
    with cloner.ReplicateSession([clone_cfg], 2, 1.0):
        pass

    assert len(calls) == expected_calls
    if calls:
        assert calls[0][3] == ["/Scene/worlds/world_0", "/Scene/worlds/world_1"]


def test_replicate_session_rejects_a_second_lifecycle(monkeypatch):
    """One simulation cannot publish a second plan or re-enter its session."""
    stage = Usd.Stage.CreateInMemory()
    simulation = _simulation(stage=stage)
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    first = cloner.ReplicateSession([cloner.CloneCfg()], 1, 0.0)

    with first:
        pass

    with pytest.raises(RuntimeError, match="exactly one clone lifecycle"):
        first.__enter__()
    with pytest.raises(RuntimeError, match="exactly one clone lifecycle"):
        cloner.ReplicateSession([cloner.CloneCfg()], 1, 0.0).__enter__()


def test_plan_routes_cfg_rows_declaratively():
    """None, exact tuples, empty tuples, and scene roles retain distinct meanings."""
    physics_type = type("Physics", (), {})
    exact_type = type("Exact", (), {})
    scene_type = type("Scene", (), {})
    whole_scene_type = type("WholeScene", (), {})
    default_cfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Default", spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)))
    exact_cfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Exact",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        cloning_contexts=(exact_type,),
    )
    empty_cfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Empty",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        cloning_contexts=(),
    )
    renderer_cfg = RendererCfg(cloning_contexts=(whole_scene_type,))

    plan = _make_clone_plan(
        *_plan_cfgs([cloner.CloneCfg(), default_cfg, exact_cfg, empty_cfg, renderer_cfg]),
        num_clones=2,
        env_spacing=1.0,
        device="cpu",
        context_roles={physics_type: {"physics"}, scene_type: {"scene"}},
    )

    assert plan.context_rows == {
        physics_type: plan.cfg_rows[id(default_cfg)],
        exact_type: plan.cfg_rows[id(exact_cfg)],
        scene_type: tuple(range(3)),
        whole_scene_type: tuple(range(3)),
    }


def test_plan_unions_repeated_context_requests_for_one_covering_row():
    """A child and its parent can request one context without duplicating their shared row."""
    context_type = type("Context", (), {})
    parent = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        cloning_contexts=(context_type,),
    )
    child = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Camera",
        spawn=sim_utils.PinholeCameraCfg(),
        cloning_contexts=(context_type,),
    )

    plan = _make_clone_plan(
        *_plan_cfgs([cloner.CloneCfg(), child, parent]),
        num_clones=2,
        env_spacing=1.0,
        device="cpu",
        context_roles={},
    )

    assert plan.context_rows[context_type] == plan.cfg_rows[id(parent)] == (0,)


def test_plan_retains_empty_physics_context_for_model_publication():
    """A physics context still publishes an empty model when no cfg requests rows."""
    physics_type = type("Physics", (), {})

    plan = _make_clone_plan(
        *_plan_cfgs([cloner.CloneCfg()]),
        num_clones=2,
        env_spacing=1.0,
        device="cpu",
        context_roles={physics_type: {"physics"}},
    )

    assert plan.context_rows == {physics_type: ()}


def test_plan_rejects_non_class_context_references():
    cfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        cloning_contexts=(lambda: None,),
    )

    with pytest.raises(TypeError, match="must contain only context classes"):
        _plan_cfgs([cloner.CloneCfg(), cfg])


def test_session_registers_explicit_usd_context_without_kit(monkeypatch):
    """Explicit core USD routing is available kitless without importing a backend package."""
    stage = Usd.Stage.CreateInMemory()
    simulation = _simulation(stage=stage)
    cfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        cloning_contexts=(cloner.UsdReplicateContext,),
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    monkeypatch.setattr(replicate_session, "replicate", lambda *args, **kwargs: None)

    with cloner.ReplicateSession([cloner.CloneCfg(), cfg], 2, 1.0):
        pass

    assert tuple(simulation._backend_registry) == (cloner.UsdReplicateContext,)
    assert simulation._backend_clone_roles == {}


def test_replicate_dispatches_each_registered_context_once(monkeypatch):
    """Per-asset and whole-environment backends consume one routed mapping."""
    per_asset_type = type("PerAsset", (_Context,), {})
    whole_env_type = type("WholeEnv", (_Context,), {})
    per_asset = per_asset_type("per_asset")
    whole_env = whole_env_type("whole_env", whole_env=True)
    simulation = _simulation(
        {per_asset_type: per_asset, whole_env_type: whole_env},
        {per_asset_type: {"physics", "scene"}, whole_env_type: {"scene"}},
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    plan = _plan()
    plan.context_rows.update({per_asset_type: (0, 1), whole_env_type: (0, 1)})
    simulation._clone_plan = plan

    cloner.replicate(plan)

    assert per_asset.mappings[0][:2] == (plan.sources, plan.destinations)
    assert whole_env.mappings[0][0] == ("/World/envs/env_0",)
    assert per_asset.replicated == whole_env.replicated == 1


def test_replicate_rejects_a_plan_not_published_by_the_simulation(monkeypatch):
    """Dispatch accepts only the plan owned by the active simulation."""
    simulation = _simulation()
    simulation._clone_plan = _plan(0)
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    with pytest.raises(ValueError, match="plan published on the active SimulationContext"):
        cloner.replicate(_plan(0))

    assert simulation._clone_plan_dispatched is False


def test_replicate_rejects_duplicate_dispatch(monkeypatch):
    """A published plan can materialize its scene exactly once."""
    simulation = _simulation()
    plan = _plan(0)
    simulation._clone_plan = plan
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    cloner.replicate(plan)
    with pytest.raises(RuntimeError, match="exactly once"):
        cloner.replicate(plan)

    assert simulation._clone_plan_dispatched is True


def test_replicate_dispatches_only_context_routed_rows(monkeypatch):
    """A context receives only the plan rows assigned to it."""
    context_type = type("Context", (_Context,), {})
    context = context_type("context")
    simulation = _simulation({context_type: context}, {context_type: {"scene"}})
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    plan = _plan()
    plan.clone_mask[1] = False
    plan.context_rows[context_type] = (0,)
    simulation._clone_plan = plan

    cloner.replicate(plan)

    sources, destinations, mask = context.mappings[0]
    assert sources == plan.sources[:1]
    assert destinations == plan.destinations[:1]
    assert torch.equal(mask, plan.clone_mask[:1])


def test_replicate_dispatches_empty_physics_plan(monkeypatch):
    """A native physics context runs once even when no asset row is routed."""
    context_type = type("Context", (_Context,), {})
    context = context_type("context")
    simulation = _simulation({context_type: context}, {context_type: {"physics"}})
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    plan = _plan(0)
    plan.context_rows[context_type] = ()
    simulation._clone_plan = plan

    cloner.replicate(plan)

    assert context.replicated == 1


def test_replicate_physics_false_skips_only_physics_only_contexts(monkeypatch):
    """Scene contexts run while mandatory models wait for the initialization phase."""
    physics_type = type("Physics", (_Context,), {})
    model_type = type("Model", (_Context,), {})
    shared_type = type("Shared", (_Context,), {})
    scene_type = type("Scene", (_Context,), {})
    physics = physics_type("physics")
    model = model_type("model")
    shared = shared_type("shared")
    scene = scene_type("scene")
    simulation = _simulation(
        {physics_type: physics, model_type: model, shared_type: shared, scene_type: scene},
        {
            physics_type: {"physics"},
            model_type: {"physics", "model"},
            shared_type: {"physics", "scene"},
            scene_type: {"scene"},
        },
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    plan = _plan(1, replicate_physics=False)
    plan.context_rows.update({physics_type: (0,), model_type: (0,), shared_type: (0,), scene_type: (0,)})
    simulation._clone_plan = plan

    cloner.replicate(plan)

    assert physics.replicated == 0
    assert model.replicated == 0
    assert shared.replicated == scene.replicated == 1
    assert simulation._pending_clone_model_contexts == (model,)


def test_first_hard_reset_dispatches_models_after_stage_edits_once(monkeypatch):
    """Model construction follows stage edits and precedes physics finalization exactly once."""
    calls = []
    model_type = type("Model", (_Context,), {})
    model = model_type("model", calls)
    simulation = _simulation({model_type: model}, {model_type: {"physics", "model"}})
    plan = _plan(0)
    plan.context_rows[model_type] = ()
    simulation._clone_plan = plan
    simulation.physics_manager = SimpleNamespace(
        reset=lambda soft: calls.append(f"physics:{soft}"), play=lambda: calls.append("play")
    )
    simulation._visualizers = []
    simulation.initialize_visualizers = lambda: None
    simulation._render_context = SimpleNamespace(finalize_consumers=lambda *_args, **_kwargs: None)
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    cloner.replicate(plan)
    assert model.replicated == 0
    assert simulation._pending_clone_model_contexts == (model,)
    calls.append("stage_edit")
    SimulationContext.reset(simulation)
    SimulationContext.reset(simulation)

    assert calls == ["stage_edit", "model", "physics:False", "play", "physics:False", "play"]
    assert simulation._pending_clone_model_contexts == ()


@pytest.mark.parametrize("dispatch_state", [False, None])
def test_reset_rejects_a_plan_without_successful_dispatch(dispatch_state):
    """Physics cannot initialize while clone dispatch is pending or failed."""
    simulation = _simulation()
    simulation._clone_plan = _plan(0)
    simulation._clone_plan_dispatched = dispatch_state

    with pytest.raises(RuntimeError, match="dispatch must complete"):
        SimulationContext.reset(simulation)


def test_soft_reset_cannot_skip_pending_model_initialization(monkeypatch):
    """A first soft reset fails before either model or physics initialization runs."""
    model_type = type("Model", (_Context,), {})
    model = model_type("model")
    simulation = _simulation({model_type: model}, {model_type: {"physics", "model"}})
    plan = _plan(0)
    plan.context_rows[model_type] = ()
    simulation._clone_plan = plan
    simulation.physics_manager = SimpleNamespace(reset=lambda _soft: pytest.fail("physics reset ran"))
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    cloner.replicate(plan)

    with pytest.raises(RuntimeError, match="first reset must initialize clone-plan models"):
        SimulationContext.reset(simulation, soft=True)

    assert model.replicated == 0
    assert simulation._pending_clone_model_contexts == (model,)


def test_model_phase_does_not_repeat_a_successful_context_after_a_later_failure():
    """A failed model leaves itself pending without rebuilding earlier published models."""
    first = _Context("first")

    class _FailingContext(_Context):
        def replicate(self, _plan):
            raise RuntimeError("model failed")

    failing = _FailingContext("failing")
    simulation = _simulation()
    simulation._clone_plan = _plan(0)
    simulation._clone_plan_dispatched = True
    simulation._clone_plan.context_rows[type(first)] = ()
    simulation._pending_clone_model_contexts = (first, failing)

    with pytest.raises(RuntimeError, match="model failed"):
        SimulationContext.reset(simulation)

    assert first.replicated == 1
    assert simulation._pending_clone_model_contexts == (failing,)


def test_replicate_rejects_an_unregistered_explicit_context(monkeypatch):
    context_type = type("MissingContext", (), {})
    plan = _plan(1)
    plan.context_rows[context_type] = (0,)
    simulation = _simulation()
    simulation._clone_plan = plan
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)

    with pytest.raises(RuntimeError, match="MissingContext"):
        cloner.replicate(plan)


def test_whole_environment_context_requires_every_routed_row(monkeypatch):
    """A whole-env backend receives flat rows when cfg routing selects only part of the plan."""
    context_type = type("WholeEnv", (_Context,), {})
    context = context_type("whole_env", whole_env=True)
    simulation = _simulation({context_type: context})
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    plan = _plan(2)
    plan.context_rows[context_type] = (0,)
    simulation._clone_plan = plan

    cloner.replicate(plan)

    assert context.mappings[0][0] == (plan.sources[0],)


def test_replicate_orders_contexts_by_priority(monkeypatch):
    calls = []
    late_type = type("Late", (_Context,), {})
    early_type = type("Early", (_Context,), {})
    late = late_type("late", calls)
    late.replicate_priority = 10
    early = early_type("early", calls)
    early.replicate_priority = -10
    simulation = _simulation(
        {late_type: late, early_type: early},
        {late_type: {"scene"}, early_type: {"scene"}},
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: simulation)
    plan = _plan(1)
    plan.context_rows.update({late_type: (0,), early_type: (0,)})
    simulation._clone_plan = plan

    cloner.replicate(plan)

    assert calls == ["early", "late"]
