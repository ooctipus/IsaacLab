# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared typing curriculum uses task-owned logical cohorts, never model-local prefixes."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab_tasks.contrib.keyboard.mdp.commands import typing_commands
from isaaclab_tasks.contrib.keyboard.mdp.commands.typing_commands import LetterTypingCommand
from isaaclab_tasks.utils.success_monitor import SuccessMonitor, SuccessMonitorCfg


def _command(heterogeneous=True):
    wp.init()
    command = object.__new__(LetterTypingCommand)
    variants = torch.tensor([0, 1, 0, 2, 1])
    active = torch.zeros((5, 12), dtype=torch.bool)
    if heterogeneous:
        for ids, slots in (([0, 2], [0, 1, 2, 3]), ([1, 4], [4, 5, 6]), ([3], [7, 8, 9, 10, 11])):
            active[torch.tensor(ids)[:, None], slots] = True
    else:
        active[:, :4] = True
    bank = SimpleNamespace(variant_ids=variants, layouts=(0, 1, 2), backspaces=torch.tensor([3, 6, 11]))
    command._env = SimpleNamespace(
        num_envs=5, device="cpu", all_env_ids=torch.arange(5), keyboard_variants=bank if heterogeneous else None
    )
    command.cfg = SimpleNamespace(letter_length=(1, 3), reset=SimpleNamespace(match_prob=0.5))
    command.key_joints = SimpleNamespace(dense_active=lambda: active)
    command.num_keys = 12
    command.max_len = 3
    command._default_backspace = torch.full((5,), 3)
    command._typeable = torch.arange(12)
    command._resample_seed = 0
    command._cur_oversample = 4
    command._cur_feat_w = (1.0, 1.0, 1.0, 1.0, 1.0)
    return command


def test_candidate_sampling_uses_only_the_requested_noncontiguous_cohort():
    command = _command()
    worlds = torch.tensor([4, 1])
    keys, counts, backspace = command._sampling_keys(worlds)
    assert wp.to_torch(keys)[:, :2].tolist() == [[4, 5], [4, 5]]
    assert wp.to_torch(counts).tolist() == [2, 2]
    assert wp.to_torch(backspace).tolist() == [True, True]
    for sample in (command._oversample(64, worlds), command._sample_diverse_states(16, worlds)):
        target, typed, lengths, typed_lengths = sample[:4]
        assert torch.isin(target[target >= 0], torch.tensor([4, 5])).all()
        assert torch.isin(typed[typed >= 0], torch.tensor([4, 5])).all()
        assert ((lengths != typed_lengths) | (target != typed).any(dim=1)).all()
    assert sample[0].shape == (16, 3)
    with pytest.raises(ValueError, match="nonempty"):
        command._sampling_keys(torch.empty(0, dtype=torch.long))


def test_explicit_all_world_cohort_preserves_baseline_samples():
    implicit, explicit = _command(False), _command(False)
    for expected, actual in zip(
        implicit._oversample(64), explicit._oversample(64, explicit._env.all_env_ids), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("heterogeneous", [False, True])
@pytest.mark.parametrize("fail_during_build", [False, True])
def test_buffer_build_uses_logical_ids_and_restores_original_state(monkeypatch, heterogeneous, fail_during_build):
    command = _command(heterogeneous)
    env = command._env
    original = torch.arange(5, dtype=torch.float32)[:, None]
    physical = original.clone()
    requested, solved, finished = [], [], []
    active_cohort = None

    def curriculum_worlds(variant):
        nonlocal active_cohort
        active_cohort = (
            env.all_env_ids if not heterogeneous else env.all_env_ids[env.keyboard_variants.variant_ids == variant]
        )
        requested.append(variant)
        return active_cohort.flip(0)  # Ordering and gaps must not be mistaken for a world prefix.

    def solve(ids):
        assert torch.isin(ids, active_cohort).all()
        solved.extend(ids.tolist())
        physical[ids] = command.target[ids, :1].float() + 100
        if fail_during_build:
            raise RuntimeError("IK failed")

    env.curriculum_worlds = curriculum_worlds
    env.finish_curriculum = lambda variants: finished.append(None if variants is None else variants.clone())
    command._solve_reset_pose = solve
    command._log_buffer_stats = lambda: None
    command._cur_buffer_size = 13
    command._buffer_built = False
    command._buf_state = None
    command.target = torch.full((5, 3), -1, dtype=torch.long)
    command.typed = torch.full_like(command.target, -1)
    command.target_len = torch.zeros(5, dtype=torch.long)
    command.typed_len = torch.zeros(5, dtype=torch.long)
    command.prefix_len = torch.zeros(5, dtype=torch.long)
    command._buf_target = torch.full((13, 3), -1, dtype=torch.long)
    command._buf_typed = torch.full_like(command._buf_target, -1)
    command._buf_target_len = torch.zeros(13, dtype=torch.long)
    command._buf_typed_len = torch.zeros(13, dtype=torch.long)
    command._buf_variant = torch.zeros(13, dtype=torch.long)
    command._buf_reach = torch.zeros(13)
    body_poses = torch.zeros((5, 1, 7))
    body_poses[:, 0, 0] = torch.arange(5) + 10
    body_poses[:, 0, -1] = 1
    command.cfg.reset.ik = SimpleNamespace(body=SimpleNamespace(read_state=lambda _: body_poses))
    command.cfg.reset_roots = command.cfg.reset_coords = command.cfg.reset_dofs = None
    command._ik_offset = torch.zeros((5, 3))
    command._ik_hover = torch.zeros(3)
    command.target_key_pos_w = lambda: torch.zeros((5, 3))
    monkeypatch.setattr(typing_commands, "capture_reset_state", lambda _, ids, *args: physical[ids].clone())

    def restore(_, snapshot, ids, *args):
        physical[ids] = snapshot

    monkeypatch.setattr(typing_commands, "restore_reset_state", restore)
    if fail_during_build:
        with pytest.raises(RuntimeError, match="IK failed"):
            command._build_buffer()
        assert not command._buffer_built
    else:
        command._build_buffer()
        assert command._buffer_built
        assert requested == ([0, 1, 2] if heterogeneous else [0])
        assert command._buf_state.shape == (13, 1)
        torch.testing.assert_close(command._buf_state[:, 0], command._buf_target[:, 0].float() + 100)
        torch.testing.assert_close(command._buf_reach, torch.tensor(solved, dtype=torch.float32) + 10)
        if heterogeneous:
            assert torch.bincount(command._buf_variant).tolist() == [5, 4, 4]
            for variant, allowed in enumerate(([0, 1, 2], [4, 5], [7, 8, 9, 10])):
                target = command._buf_target[command._buf_variant == variant]
                assert torch.isin(target[target >= 0], torch.tensor(allowed)).all()
    torch.testing.assert_close(physical, original)
    assert len(finished) == 1
    if heterogeneous:
        torch.testing.assert_close(finished[0], env.keyboard_variants.variant_ids)
    else:
        assert finished[0] is None


def test_curriculum_has_no_scene_or_physics_singleton_dependency():
    import ast

    source = Path(typing_commands.__file__).read_text()
    assert "_env.scene" not in source
    assert "sim.forward()" not in source
    assert "NewtonManager" not in source
    tree = ast.parse(source)
    solve = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_solve_reset_pose"
    )
    for loop in (node for node in ast.walk(solve) if isinstance(node, ast.For)):
        assert not any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "forward"
            for node in ast.walk(loop)
        ), "IK iterations must not reset native solver state."


@pytest.mark.parametrize("interrupted", [[False] * 5, [False, True, True, False, False], [True] * 5])
def test_administrative_cuts_reset_every_actor_without_recording_outcomes(monkeypatch, interrupted):
    from isaaclab.managers import CommandTerm

    command = _command()
    command._env.episode_interrupted = torch.tensor(interrupted)
    command._cur_enabled = command._buffer_built = True
    command._env_source = torch.tensor([0, 1, 2, 3, -1])
    command.distance = torch.tensor([0.0, 3.0, 0.0, 2.0, 0.0])
    command._start_distance = torch.tensor([1, 1, 2, 3, 1])
    command._distance_bands = (2, 4)
    command._split_last = {"buffer/success_rate": 0.7, "uniform/success_rate": 0.8}
    command._buf_avg_distance = command._reset_ik = None
    command.success_monitor = SuccessMonitor(
        SuccessMonitorCfg(monitored_history_len=8, target_success_rate=0.5),
        num_partitions=1,
        partition_size=4,
        device="cpu",
    )
    command.success_monitor.success_rate.fill_(0.5)
    reset_calls = []

    def reset_all_selected(term, env_ids):
        assert term._episode_reset
        reset_calls.append(env_ids.clone())
        term.distance[env_ids] = 5.0
        term._env_source[env_ids] = -1
        return {}

    monkeypatch.setattr(CommandTerm, "reset", reset_all_selected)
    metrics = command.reset(command._env.all_env_ids)
    expected_sizes = (~command._env.episode_interrupted[:4]).to(command.success_monitor.success_size.dtype)
    torch.testing.assert_close(command.success_monitor.success_size.flatten(), expected_sizes)
    expected_rates = torch.where(expected_sizes.bool(), torch.tensor([1.0, 0.0, 1.0, 0.0]), torch.full((4,), 0.5))
    torch.testing.assert_close(command.success_monitor.success_rate.flatten(), expected_rates)
    if all(interrupted):
        assert metrics["buffer/success_rate"] == 0.7 and metrics["uniform/success_rate"] == 0.8
    else:
        assert metrics["buffer/success_rate"] == 0.5 and metrics["uniform/success_rate"] == 1.0
        assert metrics["buffer/distance<2"] == (1.0 if interrupted[1] else 0.5)
    assert len(reset_calls) == 1
    torch.testing.assert_close(reset_calls[0], torch.arange(5))
    torch.testing.assert_close(command._start_distance, torch.full((5,), 5))
    assert not command._episode_reset


@pytest.mark.parametrize("curriculum", [False, True])
@pytest.mark.parametrize("env_ids", [[], [4, 1, 0], [0, 1, 2, 3, 4]])
def test_reset_statistics_use_one_bulk_readback_and_preserve_censored_bands(monkeypatch, curriculum, env_ids):
    from isaaclab.managers import CommandTerm

    command = _command()
    command._env.episode_interrupted = torch.tensor([False, False, True, False, True])
    command._cur_enabled = command._buffer_built = curriculum
    command._env_source = torch.tensor([0, -1, 1, -1, 2])
    command.distance = torch.tensor([0.0, 2.0, 0.0, 3.0, 0.0])
    command._start_distance = torch.tensor([1, 2, 3, 4, 5])
    command._distance_bands = (0, 2, 4, 8)
    command._buf_avg_distance = command._reset_ik = None
    command.success_monitor = SuccessMonitor(SuccessMonitorCfg(), 1, 3, "cpu")
    command._split_last = {"uniform/distance<0": 0.75, "buffer/success_rate": 0.25, "buffer/distance<2": 0.125}
    expected = {}
    for tag in ("uniform", "buffer"):
        for band in (None, *command._distance_bands):
            key = f"{tag}/success_rate" if band is None else f"{tag}/distance<{band}"
            samples = [
                index
                for index in env_ids
                if not command._env.episode_interrupted[index]
                and (curriculum and command._env_source[index] >= 0) == (tag == "buffer")
                and (band is None or command._start_distance[index] < band)
            ]
            expected[key] = (
                sum(command.distance[index] == 0 for index in samples).item() / len(samples)
                if samples
                else command._split_last.get(key, 0.0)
            )
    monkeypatch.setattr(CommandTerm, "reset", lambda term, env_ids: {})
    readbacks = []
    cpu = torch.Tensor.cpu

    def read_bulk(tensor, *args, **kwargs):
        readbacks.append((tensor.shape, tensor.dtype))
        return cpu(tensor, *args, **kwargs)

    def reject_scalar(tensor):
        raise AssertionError("Reset statistics must not read scalar tensors one at a time.")

    monkeypatch.setattr(torch.Tensor, "cpu", read_bulk)
    monkeypatch.setattr(torch.Tensor, "__float__", reject_scalar)
    monkeypatch.setattr(torch.Tensor, "__int__", reject_scalar)
    actual = command.reset(torch.tensor(env_ids, dtype=torch.long))
    assert actual == expected
    assert readbacks == [(torch.Size((10, 2)), torch.int64)]


def _preview_command():
    command = _command(False)
    command.cfg.resampling_time_range = (10.0, 10.0)
    command.cfg.actuation_fraction = 0.5
    command.cfg.key_dofs = SimpleNamespace(
        read_model=lambda name: -torch.ones(5, 12) if name == "joint_limit_lower" else torch.zeros(5, 12)
    )
    positions = torch.zeros(5, 12)
    command.key_joints.read_state = lambda name: positions
    command.target = torch.tensor([[0, 1, -1]]).expand(5, -1).clone()
    command.typed = torch.full((5, 3), -1, dtype=torch.long)
    command.target_len = torch.full((5,), 2, dtype=torch.long)
    for name in ("typed_len", "prefix_len", "max_prefix", "min_prefix", "command_counter"):
        setattr(command, name, torch.zeros(5, dtype=torch.long))
    command._prev_pressed = torch.zeros(5, 12, dtype=torch.bool)
    for name in ("_just_reset", "new_high", "new_low"):
        setattr(command, name, torch.zeros(5, dtype=torch.bool))
    command.time_left = torch.full((5,), 10.0)
    command.distance = torch.full((5,), 2.0)
    command.metrics = {"distance": command.distance}
    command._press_level = None
    command._cur_enabled = command._episode_reset = False
    command._env.command_manager = SimpleNamespace(get_term=lambda name: command)
    return command, positions


@pytest.mark.parametrize("case", ["press", "multiple", "backspace", "held", "just_reset", "timer"])
def test_successor_preview_reuses_typing_update_preserving_live_state_and_rng(case):
    from isaaclab_tasks.contrib.keyboard.mdp.observations import typed_keys_onehot

    command, positions = _preview_command()
    positions[0, 0] = -0.75
    if case == "multiple":
        positions[0, 1] = -0.75
    elif case == "backspace":
        positions.zero_()
        positions[0, 3] = -0.75
        command.typed[0, 0] = 0
        command.typed_len[0] = 1
    elif case == "held":
        command._prev_pressed[0, 0] = True
    elif case == "just_reset":
        command._just_reset[0] = True
    elif case == "timer":
        command.time_left[0] = 0.01
    references = {name: value for name, value in vars(command).items() if isinstance(value, torch.Tensor)}
    values = {name: value.clone() for name, value in references.items()}
    metrics, seed, press = command.metrics, command._resample_seed, command._press_level
    rng = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]), command.preview_step(0.04):
        preview = {name: getattr(command, name).clone() for name in values}
        observation = typed_keys_onehot(command._env, "typing")
        successor_seed = command._resample_seed
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert command.metrics is metrics and command._press_level is press and command._resample_seed == seed
    for name, original in references.items():
        assert getattr(command, name) is original
        torch.testing.assert_close(original, values[name], rtol=0, atol=0)
    command.compute(0.04)
    for name, value in preview.items():
        torch.testing.assert_close(getattr(command, name), value, rtol=0, atol=0)
    torch.testing.assert_close(typed_keys_onehot(command._env, "typing"), observation, rtol=0, atol=0)
    assert command._resample_seed == successor_seed


@pytest.mark.parametrize("inside_compute", [False, True])
def test_successor_preview_restores_original_state_on_failure(monkeypatch, inside_compute):
    command, positions = _preview_command()
    positions[0, 0] = -0.75
    original = {name: value for name, value in vars(command).items() if isinstance(value, torch.Tensor)}
    values = {name: value.clone() for name, value in original.items()}
    metrics = command.metrics
    if inside_compute:
        update = command._update_command

        def fail():
            update()
            raise RuntimeError("preview failed")

        monkeypatch.setattr(command, "_update_command", fail)
    with pytest.raises(RuntimeError, match="preview failed"), command.preview_step(0.04):
        raise RuntimeError("preview failed")
    assert command.metrics is metrics and command._press_level is None
    for name, value in original.items():
        assert getattr(command, name) is value
        torch.testing.assert_close(value, values[name], rtol=0, atol=0)


def test_final_observation_contains_current_press_and_successor_history_without_committing():
    from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg

    from isaaclab_tasks.contrib.keyboard.mdp.observations import typed_keys_onehot
    from isaaclab_tasks.contrib.keyboard.mdp.rewards import letter_typing_progress, typing_success
    from isaaclab_tasks.contrib.keyboard.mdp.terminations import typing_complete

    command, positions = _preview_command()
    env = command._env
    env.sim = SimpleNamespace(is_playing=lambda: True)
    plain = ObservationGroupCfg(concatenate_terms=True)
    plain.typed = ObservationTermCfg(func=typed_keys_onehot, params={"command_name": "typing"})
    history = ObservationGroupCfg(concatenate_terms=True)
    history.typed = ObservationTermCfg(func=typed_keys_onehot, params={"command_name": "typing"}, history_length=2)
    manager = ObservationManager({"plain": plain, "history": history}, env)
    manager.compute(update_history=True)
    before = (letter_typing_progress(env, "typing"), typing_success(env, "typing"), typing_complete(env, "typing"))
    positions[0, 0] = -0.75
    with torch.random.fork_rng(devices=[]), command.preview_step(0.04):
        successor = manager.preview()
    assert command.typed[0, 0] == -1
    assert successor["plain"][0, 0] == 1
    assert successor["history"][0, 36] == 1
    assert manager._group_obs_term_history_buffer["history"]["typed"]._num_pushes.tolist() == [1] * 5
    after = (letter_typing_progress(env, "typing"), typing_success(env, "typing"), typing_complete(env, "typing"))
    for old, current in zip(before, after, strict=True):
        torch.testing.assert_close(current, old, rtol=0, atol=0)
    command.compute(0.04)
    actual = manager.compute(update_history=True)
    for name in successor:
        torch.testing.assert_close(successor[name], actual[name], rtol=0, atol=0)
