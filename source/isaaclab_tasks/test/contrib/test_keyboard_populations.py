# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor assignment validation and exact population publication, without physics assets."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import newton
import numpy as np
import pytest
import torch

from isaaclab_tasks.contrib.keyboard.keyboard_populations import KeyboardPopulations, prepare_keyboard_prototype
from isaaclab_tasks.contrib.keyboard.selection_paths import NewtonSelectorCfg, resolve_selection
from isaaclab_tasks.contrib.keyboard.so101_population_env import SO101KeyboardPopulationEnv


@pytest.mark.parametrize("setting", ["manager", "prototype", "prototype_mode"])
def test_deterministic_population_modes_are_rejected_before_authoring(setting):
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Populations", "", overrides=["physics=newton_mjwarp"])
    if setting == "manager":
        cfg.sim.physics.deterministic = True
    elif setting == "prototype":
        cfg.sim.physics.prototype_physics.deterministic = True
    else:
        cfg.sim.physics.prototype_physics.deterministic_mode = "run_to_run"
    with pytest.raises(ValueError, match="Deterministic"):
        KeyboardPopulations(SimpleNamespace(cfg=cfg))


@pytest.mark.parametrize("missing", ["enabled", "ik", "prototype_coverage", "single_coverage"])
def test_native_replay_rejects_incomplete_snapshot_curriculum_before_authoring(missing):
    from isaaclab_tasks.contrib.keyboard.keyboard_worlds import KeyboardWorlds
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Worlds", "", overrides=["physics=newton_mjwarp"])
    cfg.commands.typing.reset.bank_path = "unused.pt"
    cfg.commands.typing.reset.replay_only = True
    if missing in ("enabled", "ik"):
        setattr(cfg.commands.typing.reset, missing, False if missing == "enabled" else None)
        message = "enabled typing curriculum"
    else:
        cfg.keyboard_variants = None if missing == "single_coverage" else (cfg.scene.keyboard.spawn,) * 3
        cfg.commands.typing.reset.bank_path = None
        cfg.commands.typing.reset.buffer_size = 0 if missing == "single_coverage" else 2
        message = "at least one snapshot per keyboard prototype"
    with (
        patch("isaaclab_tasks.contrib.keyboard.keyboard_worlds.generate_keyboard") as generate,
        pytest.raises(ValueError, match=message),
    ):
        KeyboardWorlds(SimpleNamespace(cfg=cfg))
    generate.assert_not_called()


@pytest.mark.parametrize(
    "prototype_count, buffer_size, replay_only, bank_path",
    [(3, 3, True, None), (3, 2, False, None), (3, 0, True, "unused.pt"), (None, 1, True, None)],
)
def test_native_buffer_coverage_preserves_admitted_reset_modes(prototype_count, buffer_size, replay_only, bank_path):
    from isaaclab_tasks.contrib.keyboard.keyboard_worlds import KeyboardWorlds
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Worlds", "", overrides=["physics=newton_mjwarp"])
    cfg.keyboard_variants = None if prototype_count is None else (cfg.scene.keyboard.spawn,) * prototype_count
    reset = cfg.commands.typing.reset
    reset.buffer_size, reset.replay_only, reset.bank_path = buffer_size, replay_only, bank_path
    with (
        patch(
            "isaaclab_tasks.contrib.keyboard.keyboard_worlds.generate_keyboard",
            side_effect=RuntimeError("authoring reached"),
        ),
        pytest.raises(RuntimeError, match="authoring reached"),
    ):
        KeyboardWorlds(SimpleNamespace(cfg=cfg))
    assert reset.buffer_size == buffer_size  # Admission never silently enlarges the configured bank.


@pytest.mark.parametrize("phase", ["second_prototype", "backend", "native_binding"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_bank_construction_retires_acquired_bindings_without_gc(phase, cleanup_fails):
    from isaaclab_tasks.contrib.keyboard.newton_selection import BODY, NewtonSelections
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Populations", "", overrides=["physics=newton_mjwarp"])
    cfg.keyboard_variants = (cfg.scene.keyboard.spawn,) * 2
    cfg.terminations.excessive_contact = None
    selector = NewtonSelectorCfg(BODY, ".*", count_per_world=1)
    startup_error, cleanup_error = MemoryError("startup allocation"), RuntimeError("cleanup failed")
    references, calls = [], []

    def prepare(_env, *_):
        if phase == "second_prototype" and references:
            raise startup_error
        builder = newton.ModelBuilder()
        builder.begin_world()
        body = builder.add_body(label="/body")
        builder.add_shape_box(body, hx=0.03, hy=0.03, hz=0.03)
        builder.end_world()
        source = NewtonSelections(builder.finalize("cpu"))
        resolve_selection(source, selector)
        references.extend((weakref.ref(source), weakref.ref(source.model)))
        return SimpleNamespace(model=source.model), source

    def backend(_):
        if phase == "backend":
            raise startup_error
        return SimpleNamespace()

    def bind(bank):
        model = bank._sources[0].model.replicate(1)
        owner = NewtonSelections(model, state=model.state(), source=bank._sources[0])
        bank._owners[0] = owner
        bank._bindings["body"] = resolve_selection(owner, selector)
        references.extend((weakref.ref(owner), weakref.ref(model), weakref.ref(owner.state)))
        raise startup_error

    original_close = KeyboardPopulations.close

    def close(bank):
        calls.append("close")
        original_close(bank)
        if cleanup_fails:
            raise cleanup_error

    key = SimpleNamespace(label="backspace", slot=0)
    layout = SimpleNamespace(active_key_count=6, active_keys=(key,), keys=(key,), slot_count=6)
    env = SimpleNamespace(
        cfg=cfg,
        device="cpu",
        num_envs=2,
        all_env_ids=torch.arange(2),
        physics_dt=0.01,
        sim=SimpleNamespace(get_or_create_backend=backend, physics_manager=SimpleNamespace(install=lambda _: None)),
    )
    enabled = gc.isenabled()
    gc.disable()
    try:
        with (
            patch("isaaclab_tasks.contrib.keyboard.keyboard_populations.generate_keyboard", return_value=layout),
            patch.object(KeyboardPopulations, "_manager_configs", return_value=()),
            patch("isaaclab_tasks.contrib.keyboard.keyboard_populations.prepare_keyboard_prototype", new=prepare),
            patch.object(KeyboardPopulations, "_bind_native", new=bind),
            patch.object(KeyboardPopulations, "close", new=close),
            pytest.raises(MemoryError) as error,
        ):
            KeyboardPopulations(env)
        assert error.value is startup_error
        assert calls == ["close"]
        assert startup_error.__cause__ is (cleanup_error if cleanup_fails else None)
        assert not hasattr(env, "keyboard_variants")  # No partially constructed bank is published.
        startup_error.__traceback__ = cleanup_error.__traceback__ = None
        del error  # pytest's ExceptionInfo independently retains the failing frame.
        assert references and all(reference() is None for reference in references)
    finally:
        if enabled:
            gc.enable()


def test_failed_local_prototype_preparation_retires_unreturned_source_without_gc():
    from isaaclab_tasks.contrib.keyboard.newton_selection import BODY, NewtonSelections
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Populations", "", overrides=["physics=newton_mjwarp"])
    selectors = (NewtonSelectorCfg(BODY, ".*", count_per_world=1),)
    references = []

    def add_usd(builder, *args, **kwargs):
        body = builder.add_body(label="/body")
        builder.add_shape_box(body, hx=0.03, hy=0.03, hz=0.03)

    def create_source(model):
        source = NewtonSelections(model)
        references.extend((weakref.ref(source), weakref.ref(model)))
        return source

    # The source cache is populated before the deliberately unmatched robot selector fails.
    env = SimpleNamespace(
        cfg=cfg, device="cpu", sim=SimpleNamespace(physics_manager=SimpleNamespace(create_builder=newton.ModelBuilder))
    )
    spawn = SimpleNamespace(func=lambda *args, **kwargs: None)
    for asset in (cfg.scene.robot, cfg.scene.plane):
        asset.spawn = spawn
    enabled = gc.isenabled()
    gc.disable()
    try:
        with (
            patch.object(newton.ModelBuilder, "add_usd", new=add_usd),
            patch("isaaclab_tasks.contrib.keyboard.keyboard_populations.NewtonSelections", new=create_source),
            pytest.raises(ValueError, match="matched no") as error,
        ):
            prepare_keyboard_prototype(env, spawn, cfg.sim.physics.prototype_physics, selectors)
        error.value.__traceback__ = None
        del error
        assert references and all(reference() is None for reference in references)
    finally:
        if enabled:
            gc.enable()


def _populations():
    populations = object.__new__(KeyboardPopulations)
    populations.env = object.__new__(SO101KeyboardPopulationEnv)
    populations.env.num_envs, populations.env.device = 6, "cpu"
    populations.env._is_closed = False
    populations.env._population_bindings_valid = True
    populations.layouts = (object(), object(), object())
    populations.variant_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    populations.desired_variant_ids = populations.variant_ids.clone()
    populations._actors = (np.array([0, 1]), np.array([2, 3]), np.array([4, 5]))
    populations.worlds = tuple(torch.from_numpy(actors) for actors in populations._actors)
    populations._bindings = {}
    populations._owners = []
    populations._sources = []
    populations._bind_native = lambda: None
    populations.redistribution_count = populations.last_changed_worlds = 0
    populations.last_redistribution_ms = 0.0
    backend = SimpleNamespace(counts=(2, 2, 2), populations=(object(), object(), object()), calls=[])

    def replace(counts, *, survivors):
        backend.calls.append((counts, survivors))
        backend.populations = tuple(
            old if old_count == count else object()
            for old, old_count, count in zip(backend.populations, backend.counts, counts, strict=True)
        )
        backend.counts = counts

    backend.replace = replace
    populations.backend = backend
    return populations


@pytest.mark.parametrize("operation", ["request_variants", "apply", "apply_pending_variants"])
@pytest.mark.parametrize(
    "ids",
    [
        torch.tensor([-1]),
        torch.tensor([6]),
        torch.tensor([1, 1]),
        torch.tensor([1.0]),
        torch.tensor([[1]]),
        torch.empty(1, dtype=torch.long, device="meta"),
    ],
)
def test_invalid_actor_requests_fail_before_mutation(operation, ids):
    populations = _populations()
    original = populations.variant_ids.clone()
    with pytest.raises((ValueError, RuntimeError, TypeError, IndexError)):
        if operation == "apply_pending_variants":
            populations.apply_pending_variants(ids)
        else:
            getattr(populations, operation)(ids, torch.zeros_like(ids, dtype=torch.long))
    torch.testing.assert_close(populations.variant_ids, original)
    torch.testing.assert_close(populations.desired_variant_ids, original)
    assert not populations.backend.calls


@pytest.mark.parametrize("operation", ["request_variants", "apply"])
@pytest.mark.parametrize(
    "variants",
    [
        torch.tensor([-1]),
        torch.tensor([3]),
        torch.tensor([1.0]),
        torch.tensor([[1]]),
        torch.tensor([1, 2]),
        torch.empty(1, dtype=torch.long, device="meta"),
    ],
)
def test_invalid_variants_fail_before_mutation(operation, variants):
    populations = _populations()
    original = populations.variant_ids.clone()
    with pytest.raises((ValueError, RuntimeError, TypeError, IndexError)):
        getattr(populations, operation)(torch.tensor([0]), variants)
    torch.testing.assert_close(populations.variant_ids, original)
    torch.testing.assert_close(populations.desired_variant_ids, original)
    assert not populations.backend.calls


def test_request_stays_on_the_tensor_device_without_publishing_a_model():
    populations = _populations()
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("request readback")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("request readback")),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("request scalar readback")),
    ):
        populations.request_variants(torch.tensor([1, 4]), torch.tensor([2, 0]))
    assert populations.desired_variant_ids.tolist() == [0, 2, 1, 1, 0, 2]
    assert populations.variant_ids.tolist() == [0, 0, 1, 1, 2, 2]
    assert not populations.backend.calls


@pytest.mark.parametrize(
    "assignments",
    [torch.tensor([0, 0, 1, 1, 2, 3]), torch.tensor([0, 0, -1, 1, 2, 2]), torch.zeros((2, 3), dtype=torch.long)],
)
def test_invalid_complete_plan_fails_before_backend_publication(assignments):
    populations = _populations()
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        populations._replace(assignments)
    assert not populations.backend.calls
    assert populations.variant_ids.tolist() == [0, 0, 1, 1, 2, 2]
    assert [actors.tolist() for actors in populations._actors] == [[0, 1], [2, 3], [4, 5]]


def test_simultaneous_same_count_swaps_keep_survivor_rows_and_runtime_identity():
    populations = _populations()
    original_runtimes = populations.backend.populations
    changed = populations.apply(torch.tensor([0, 2]), torch.tensor([1, 0]))
    assert changed.tolist() == [0, 2]
    assert populations.backend.populations == original_runtimes
    assert [actors.tolist() for actors in populations._actors] == [[2, 1], [0, 3], [4, 5]]
    assert populations.backend.calls == [((2, 2, 2), (([1], [1]), ([1], [1]), ([0, 1], [0, 1])))]
    assert populations.variant_ids.tolist() == [1, 0, 0, 1, 2, 2]
    assert populations.desired_variant_ids.tolist() == [1, 0, 0, 1, 2, 2]
    assert populations.apply_pending_variants(torch.arange(6)).numel() == 0
    assert len(populations.backend.calls) == 1


def test_resize_moves_surviving_tail_rows_to_the_new_exact_population():
    populations = _populations()
    populations.apply(torch.tensor([0]), torch.tensor([1]))
    assert populations.backend.counts == (1, 3, 2)
    assert [actors.tolist() for actors in populations._actors] == [[1], [2, 3, 0], [4, 5]]
    assert populations.backend.calls[0][1] == (([1], [0]), ([0, 1], [0, 1]), ([0, 1], [0, 1]))


def test_backend_failure_leaves_published_actor_maps_unchanged():
    populations = _populations()
    populations.backend.replace = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("transfer failed"))
    with pytest.raises(RuntimeError, match="transfer failed"):
        populations.apply(torch.tensor([0]), torch.tensor([1]))
    assert populations.variant_ids.tolist() == [0, 0, 1, 1, 2, 2]
    assert populations.desired_variant_ids.tolist() == [0, 0, 1, 1, 2, 2]
    assert [actors.tolist() for actors in populations._actors] == [[0, 1], [2, 3], [4, 5]]
    assert populations.env._population_bindings_valid


@pytest.mark.parametrize("replacement", ["replaced", "removed", "allocation_failure"])
def test_native_rebinding_retires_replaced_owners_without_cyclic_gc(replacement):
    from isaaclab_tasks.contrib.keyboard.newton_selection import BODY, NewtonSelectionGroup, NewtonSelections

    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_body(label="/body", mass=1.0)
    builder.end_world()
    source = NewtonSelections(builder.finalize("cpu"))
    selector = NewtonSelectorCfg(BODY, ".*", count_per_world=1)
    resolve_selection(source, selector)
    model = source.model.replicate(2)
    owner = NewtonSelections(model, state=model.state(), control=model.control(), source=source)
    survivor_model = source.model.replicate(2)
    survivor = NewtonSelections(survivor_model, state=survivor_model.state(), source=source)
    binding = NewtonSelectionGroup(
        selector.index_domain,
        ((resolve_selection(owner, selector), torch.tensor([0, 1])),),
        2,
        policy_width=selector.policy_width,
    )
    survivor_binding = resolve_selection(survivor, selector)
    references = [weakref.ref(value) for value in (owner, model, owner.state, owner.control)]
    bank = KeyboardPopulations.__new__(KeyboardPopulations)
    bank.env = SimpleNamespace(native_contacts={})
    bank._owners, bank._sources = [owner, survivor], [source, source]
    bank._reset_masks, bank._contact_sources = [None, None], [None, None]
    bank._bindings = {"bodies": binding}
    new_model = source.model.replicate(2)
    new_population = SimpleNamespace(
        model=new_model, state_0=new_model.state(), control=new_model.control(), solver=None
    )
    unchanged_population = SimpleNamespace(model=survivor_model)
    bank.backend = SimpleNamespace(
        populations=(None if replacement == "removed" else new_population, unchanged_population)
    )
    del owner, model
    enabled = gc.isenabled()
    gc.disable()
    try:
        if replacement == "allocation_failure":
            allocation_error = MemoryError("replacement allocation")
            with (
                patch(
                    "isaaclab_tasks.contrib.keyboard.keyboard_populations.NewtonSelections",
                    side_effect=allocation_error,
                ),
                pytest.raises(MemoryError, match="replacement allocation"),
            ):
                bank._bind_native()
            # The old owner remains reachable for close, but cannot repopulate its cache.
            with pytest.raises(RuntimeError, match="retired"):
                resolve_selection(bank._owners[0], selector)
            # A retained test exception intentionally holds its failing frame;
            # release that external reference before checking owner lifetime.
            allocation_error.__traceback__ = None
        else:
            bank._bind_native()
            assert bank._owners[1] is survivor
            assert resolve_selection(survivor, selector) is survivor_binding
            active_owner = bank._owners[0] if replacement == "replaced" else survivor
            binding.rebind(((resolve_selection(active_owner, selector), torch.tensor([0, 1])),))
            assert all(reference() is None for reference in references)
        bank.close()
        del binding
        assert all(reference() is None for reference in references)
        with pytest.raises(RuntimeError, match="retired"):
            resolve_selection(source, selector)
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("failure_phase", ["native_bind", "selection_rebind"])
def test_post_publication_failure_rejects_continuation_and_allows_explicit_close(failure_phase):
    from isaaclab_tasks.contrib.keyboard.newton_selection import BODY, NewtonSelections

    populations = _populations()
    env = populations.env
    builder = newton.ModelBuilder()
    builder.begin_world()
    body = builder.add_body(label="/body")
    builder.add_shape_box(body, hx=0.03, hy=0.03, hz=0.03)
    builder.end_world()
    model = builder.finalize("cpu")
    state = model.state()
    model_ref, state_ref = weakref.ref(model), weakref.ref(state)
    binding = resolve_selection(NewtonSelections(model, state=state), NewtonSelectorCfg(BODY, path=".*"))
    env.cfg = SimpleNamespace(body=binding)
    env.keyboard_variants = populations
    env.obs_buf, env.native_contacts = {}, {}
    env.sim = SimpleNamespace(stop=Mock(), clear_instance=Mock())
    del model, state, binding

    original_error = MemoryError("injected binding allocation failure")
    if failure_phase == "native_bind":
        populations._bind_native = Mock(side_effect=original_error)
    else:
        populations._selector_cfgs = {"first": object(), "second": object()}
        populations._parts = lambda cfg: ()
        populations._bindings = {"first": Mock(), "second": Mock()}
        populations._bindings["second"].rebind.side_effect = original_error
    with pytest.raises(MemoryError) as error:
        populations.apply(torch.tensor([0]), torch.tensor([1]))
    assert error.value is original_error
    assert populations.backend.counts == (1, 3, 2)
    if failure_phase == "selection_rebind":
        populations._bindings["first"].rebind.assert_called_once()
    env.sim.stop.assert_not_called()
    env.sim.clear_instance.assert_not_called()

    assignments = populations.variant_ids.clone()
    desired = populations.desired_variant_ids.clone()
    for operation, args in (
        (env.step, (torch.zeros(6, 1),)),
        (env.reset, ()),
        (env.forward, ()),
        (populations.request_variants, (torch.tensor([1]), torch.tensor([2]))),
        (populations.apply, (torch.tensor([1]), torch.tensor([2]))),
        (populations.apply_pending_variants, (torch.arange(6),)),
        (populations.reconcile_state, (torch.ones(6, dtype=torch.bool), 0)),
    ):
        with pytest.raises(RuntimeError, match="bindings.*failed"):
            operation(*args)
    torch.testing.assert_close(populations.variant_ids, assignments)
    torch.testing.assert_close(populations.desired_variant_ids, desired)
    assert len(populations.backend.calls) == 1

    # Explicit close remains usable after failure and releases real native objects.
    env.close()
    env.close()
    env.sim.stop.assert_called_once()
    env.sim.clear_instance.assert_called_once()
    gc.collect()
    assert model_ref() is None and state_ref() is None


@pytest.mark.parametrize("setting", ["manager", "prototype", "prototype_mode"])
def test_native_world_determinism_rejected_before_authoring(setting):
    from isaaclab_tasks.contrib.keyboard.keyboard_worlds import KeyboardWorlds
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Worlds", "", overrides=["physics=newton_mjwarp"])
    cfg.commands.typing.reset.bank_path = "unused.pt"
    if setting == "manager":
        cfg.sim.physics.deterministic = True
    elif setting == "prototype":
        cfg.sim.physics.prototype_physics.deterministic = True
    else:
        cfg.sim.physics.prototype_physics.deterministic_mode = "run_to_run"
    with pytest.raises(ValueError, match="Deterministic"):
        KeyboardWorlds(SimpleNamespace(cfg=cfg))


@pytest.mark.parametrize("phase", ["second_prototype", "backend", "native_binding"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_native_world_construction_retires_sources_without_gc(phase, cleanup_fails):
    import warp as wp

    from isaaclab_tasks.contrib.keyboard import keyboard_worlds
    from isaaclab_tasks.contrib.keyboard.newton_selection import BODY, NewtonSelections
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Worlds", "", overrides=["physics=newton_mjwarp"])
    cfg.keyboard_variants = (cfg.scene.keyboard.spawn,) * 2
    cfg.commands.typing.reset.bank_path = "unused.pt"
    cfg.commands.typing.reset.replay_only = True
    cfg.terminations.excessive_contact = None
    startup_error, cleanup_error = MemoryError("startup allocation"), RuntimeError("cleanup failed")
    references, calls = [], []

    def prepare(*_, **__):
        if phase == "second_prototype" and references:
            raise startup_error
        builder = newton.ModelBuilder()
        builder.begin_world()
        body = builder.add_body(label="/body")
        builder.add_shape_box(body, hx=0.03, hy=0.03, hz=0.03)
        builder.end_world()
        source = NewtonSelections(builder.finalize("cpu"))
        resolve_selection(source, NewtonSelectorCfg(BODY, ".*", count_per_world=1))
        references.extend((weakref.ref(source), weakref.ref(source.model)))
        native_model = SimpleNamespace(opt=SimpleNamespace(timestep=wp.zeros(1, dtype=float, device="cpu")))
        return SimpleNamespace(model=source.model, mjw_model=native_model, mjw_data=object()), source

    backend_owner = SimpleNamespace(runtime=SimpleNamespace(populations=()), close=Mock())

    def backend(_):
        if phase == "backend":
            raise startup_error
        return backend_owner

    original_close = keyboard_worlds.KeyboardWorlds.close

    def bind(*_, **__):
        raise startup_error

    def close(bank):
        calls.append("close")
        original_close(bank)
        if cleanup_fails:
            raise cleanup_error

    key = SimpleNamespace(label="backspace", slot=0)
    layout = SimpleNamespace(active_key_count=6, active_keys=(key,), keys=(key,), slot_count=6)
    env = SimpleNamespace(
        cfg=cfg,
        device="cpu",
        num_envs=2,
        all_env_ids=torch.arange(2),
        physics_dt=0.01,
        sim=SimpleNamespace(get_or_create_backend=backend),
    )
    enabled = gc.isenabled()
    gc.disable()
    try:
        with (
            patch.object(keyboard_worlds, "generate_keyboard", return_value=layout),
            patch.object(keyboard_worlds, "prepare_keyboard_prototype", new=prepare),
            patch.object(keyboard_worlds.KeyboardWorlds, "_manager_configs", return_value=()),
            patch.object(keyboard_worlds.KeyboardWorlds, "_snapshot_columns", new=lambda *_: ()),
            patch.object(keyboard_worlds.KeyboardWorlds, "close", new=close),
            patch.object(keyboard_worlds.mjw, "replicate_data", return_value=object()),
            patch.object(keyboard_worlds.mjw, "forward"),
            patch.object(keyboard_worlds.mjw, "make_step_workspace"),
            patch.object(keyboard_worlds.mjw, "step"),
            patch.object(keyboard_worlds, "NewtonMuJoCoMapping", new=lambda *_: object()),
            patch.object(keyboard_worlds, "MuJoCoSelections", new=bind),
            pytest.raises(MemoryError) as error,
        ):
            keyboard_worlds.KeyboardWorlds(env)
        assert error.value is startup_error
        assert calls == ["close"]
        assert startup_error.__cause__ is (cleanup_error if cleanup_fails else None)
        backend_owner.close.assert_not_called()  # SimulationContext owns registered backends.
        startup_error.__traceback__ = cleanup_error.__traceback__ = None
        del error
        assert references and all(reference() is None for reference in references)
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("spare_bytes", [-1, True, 1.5])
def test_native_spare_budget_rejects_invalid_values_before_prototype_allocation(spare_bytes):
    from isaaclab_tasks.contrib.keyboard import keyboard_worlds
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Worlds", "", overrides=["physics=newton_mjwarp"])
    cfg.worlds_spare_memory_budget_bytes = spare_bytes
    with (
        patch.object(keyboard_worlds, "prepare_keyboard_prototype", side_effect=AssertionError("prototype allocation")),
        pytest.raises(ValueError, match="spare backing budget"),
    ):
        keyboard_worlds.KeyboardWorlds(SimpleNamespace(cfg=cfg))


def test_backing_demand_only_releases_valid_source_lifetimes():
    import warp as wp
    from newton.worlds import WorldDirectoryData, WorldOperation, create_world_commands

    from isaaclab_tasks.contrib.keyboard.keyboard_worlds import _backing_demand

    data = WorldDirectoryData()
    for name, values in {
        "prototype": [0],
        "slot": [0],
        "slot_starts": [0, 1, 2],
        "slot_id": [0, -1],
        "live_count": [1, 0],
    }.items():
        setattr(data, name, wp.array(values, dtype=int, device="cpu"))
    data.generation = wp.array([3], dtype=wp.uint64, device="cpu")
    commands = create_world_commands(4, device="cpu")
    commands.count.fill_(4)
    commands.operation.fill_(int(WorldOperation.RESET))
    commands.prototype.fill_(1)
    commands.world_id.assign(np.array([0, 0, -1, 4], dtype=np.int32))
    commands.generation.assign(np.array([3, 2, 3, 3], dtype=np.uint64))
    demand = wp.zeros((2, 2), dtype=int, device="cpu")
    wp.launch(_backing_demand, 2, [commands, data, demand], device="cpu")
    # Destination reservation may be conservative; only the one valid source is released.
    np.testing.assert_array_equal(demand.numpy(), [[1, 0], [4, 4]])


@pytest.mark.parametrize("fail_publication", [False, True])
def test_snapshot_publication_updates_staged_variants_only_after_success(fail_publication):
    import warp as wp

    from isaaclab_tasks.contrib.keyboard.keyboard_worlds import KeyboardWorlds

    bank = KeyboardWorlds.__new__(KeyboardWorlds)
    bank.env = SimpleNamespace(num_envs=2, _check_active=lambda: None)
    bank.layouts = (object(), object())
    bank.variant_ids = torch.tensor([0, 0])
    bank.desired_variant_ids = torch.tensor([0, 1])
    bank._staged_variant_ids = torch.tensor([0, 0])
    bank._payload = torch.zeros((2, 3))
    bank._payload_enabled = wp.zeros(1, dtype=int, device="cpu")
    bank.reset_publication_count = 0
    bank.last_changed_worlds = bank.last_reset_publication_ms = 0

    def publish(env_ids, variants):
        assert bank.staged_variant_ids(env_ids).tolist() == [0]
        if fail_publication:
            raise RuntimeError("publication failed")
        bank.variant_ids[env_ids] = variants

    bank._submit = publish
    ids, variants = torch.tensor([0]), torch.tensor([1])
    if fail_publication:
        with pytest.raises(RuntimeError, match="publication failed"):
            bank.reset_from_snapshot(ids, variants, torch.ones((1, 3)))
        assert bank.staged_variant_ids(ids).tolist() == [0]
        assert bank.variant_ids.tolist() == [0, 0]
        assert bank.reset_publication_count == 0
    else:
        bank.reset_from_snapshot(ids, variants, torch.ones((1, 3)))
        assert bank.staged_variant_ids(ids).tolist() == [1]
        assert bank.variant_ids.tolist() == [1, 0]
        assert bank.reset_publication_count == 1
    # A future requested distribution remains independent of this snapshot restore.
    assert bank.desired_variant_ids.tolist() == [0, 1]
