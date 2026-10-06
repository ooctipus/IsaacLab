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

from isaaclab.utils import clone

from isaaclab_tasks.contrib.keyboard.mdp.commands import typing_commands
from isaaclab_tasks.contrib.keyboard.mdp.commands.typing_commands import LetterTypingCommand
from isaaclab_tasks.utils.success_monitor import SuccessMonitor, SuccessMonitorCfg


def _dense_scalar_field(values, active=None):
    from isaaclab_tasks.contrib.keyboard.newton_selection import NewtonScalarField, _ScalarSource

    field = NewtonScalarField()
    source = _ScalarSource()
    source.values = wp.from_torch(values.flatten())
    field.sources = wp.array([source], dtype=_ScalarSource, device=values.device.type)
    field.sources._source_values = source.values
    field.source_ids = wp.zeros(values.shape[0], dtype=wp.int32, device=values.device.type)
    field.ids = wp.from_torch(torch.arange(values.numel(), dtype=torch.int32).reshape(values.shape))
    field.active = wp.from_torch(torch.ones_like(values, dtype=torch.bool) if active is None else active)
    return field


@pytest.mark.parametrize("history", [1, 3, 10, 50])
@pytest.mark.parametrize("masked", [False, True])
def test_success_history_preserves_order_and_excludes_invalid_slots(history, masked):
    monitor = SuccessMonitor(SuccessMonitorCfg(monitored_history_len=history), 1, 7, "cpu")
    monitor.success_rate.fill_(0.5)
    outcomes = [[0.0] * history for _ in range(7)]
    pointers, sizes, rates = [0] * 7, [0] * 7, [0.5] * 7
    generator = torch.Generator().manual_seed(123)
    for count in (0, 1, 81, 13, 64, 200, 31):
        slots = torch.randint(0, 6, (count,), generator=generator)
        success = torch.rand(count, generator=generator) > 0.4
        valid = torch.rand(count, generator=generator) > 0.3 if masked else torch.ones(count, dtype=torch.bool)
        if masked:
            slots[~valid] = -999
        for slot in range(7):
            updates = [float(success[i]) for i in range(count) if valid[i] and slots[i] == slot][-history:]
            for result in updates:
                outcomes[slot][pointers[slot]] = result
                pointers[slot] = (pointers[slot] + 1) % history
            if updates:
                sizes[slot] = min(history, sizes[slot] + len(updates))
                rates[slot] = sum(outcomes[slot]) / sizes[slot]
        monitor.success_update(slots, success, valid=valid if masked else None)
        torch.testing.assert_close(monitor.success_buf, torch.tensor(outcomes))
        torch.testing.assert_close(monitor.success_pointer, torch.tensor(pointers))
        torch.testing.assert_close(monitor.success_size, torch.tensor(sizes))
        torch.testing.assert_close(monitor.success_rate, torch.tensor(rates))


def test_success_history_keeps_masked_int64_identity_and_rejects_invalid_descriptors():
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        monitor = SuccessMonitor(SuccessMonitorCfg(monitored_history_len=3), 1, 2, "cpu")
    finally:
        torch.set_default_dtype(original_dtype)
    assert monitor.success_buf.dtype == monitor.success_rate.dtype == torch.float32
    monitor.success_update(torch.tensor([0, 2**32]), torch.tensor([True, False]), valid=torch.tensor([True, False]))
    assert monitor.success_size.tolist() == [1, 0]
    assert monitor.success_pointer.tolist() == [1, 0]
    assert monitor.success_buf.tolist() == [[1, 0, 0], [0, 0, 0]]
    original = monitor.success_buf.clone()
    for slots, outcomes, valid in (
        (torch.tensor([0.5]), torch.tensor([True]), None),
        (torch.tensor([[0]]), torch.tensor([[True]]), None),
        (torch.tensor([0]), torch.tensor([True, False]), None),
        (torch.tensor([0]), torch.tensor([True]), torch.tensor([1])),
    ):
        with pytest.raises(ValueError):
            monitor.success_update(slots, outcomes, valid=valid)
        torch.testing.assert_close(monitor.success_buf, original)
    for invalid in (-1, 2, 2**32):
        # The masked tail must not conceal an earlier invalid included identity.
        with pytest.raises(RuntimeError, match="must index the monitor bank"):
            monitor.success_update(
                torch.tensor([invalid, invalid]), torch.tensor([True, False]), valid=torch.tensor([True, False])
            )
        torch.testing.assert_close(monitor.success_buf, original)
    with pytest.raises(ValueError, match="at least one episode"):
        SuccessMonitor(SuccessMonitorCfg(monitored_history_len=0), 1, 2, "cpu")


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
    bank = SimpleNamespace(variant_ids=variants, layouts=(0, 1, 2), backspace_slots=torch.tensor([3, 6, 11]))
    command._env = SimpleNamespace(
        num_envs=5, device="cpu", all_env_ids=torch.arange(5), keyboard_variants=bank if heterogeneous else None
    )
    command._env.reset_variant_ids = lambda ids: variants[ids]
    command.cfg = SimpleNamespace(
        letter_length=(1, 3), reset=SimpleNamespace(match_prob=0.5, bank_path=None, replay_only=False)
    )
    command.key_joints = SimpleNamespace(dense_active=lambda: active)
    command.cfg.actuation_fraction = 0.5
    command.cfg.key_dofs = SimpleNamespace(
        scalar_field=lambda source, name: _dense_scalar_field(
            -torch.ones(5, 12) if name == "joint_limit_lower" else torch.zeros(5, 12)
        )
    )
    command.key_joints.scalar_field = lambda source, name: _dense_scalar_field(torch.zeros(5, 12), active)
    command.typed = torch.full((5, 3), -1, dtype=torch.long)
    command.typed_len = torch.zeros(5, dtype=torch.long)
    command._prev_pressed = torch.zeros(5, 12, dtype=torch.bool)
    command._just_reset = torch.zeros(5, dtype=torch.bool)
    command.num_keys = 12
    command.max_len = 3
    command._default_backspace = torch.full((5,), 3)
    command._typeable = torch.arange(12)
    command._resample_seed = 0
    command._episode_reset = False
    command._prototype_membership = active[torch.tensor([0, 1, 3])].clone() if heterogeneous else None
    command._cur_oversample = 4
    command._cur_feat_w = (1.0, 1.0, 1.0, 1.0, 1.0)
    return command


@pytest.mark.parametrize("replay_only", [False, True])
def test_native_world_config_copy_preserves_explicit_reset_mode(replay_only):
    from isaaclab_tasks.contrib.keyboard.so101_env_cfg import SO101KeyboardWorldsEnvCfg

    cfg = SO101KeyboardWorldsEnvCfg()
    cfg.commands.typing.reset.replay_only = replay_only
    assert clone(cfg).commands.typing.reset.replay_only is replay_only


def test_replay_only_samples_requested_prototypes_without_normal_weight_floor():
    command = _command()
    command.cfg.reset.replay_only = True
    command._cur_sample_eps = 1e-4
    command._cur_normal_weight = 0.0
    command._buf_variant = torch.tensor([0, 1, 2, 2])
    command.success_monitor = SimpleNamespace(target_weights=lambda: torch.zeros(4))
    desired = torch.tensor([2, 2, 1, 0, 0])
    command._env.reset_variant_ids = lambda ids: desired[ids]
    for _ in range(16):
        selected = command._sample_sources(torch.arange(5))
        assert (selected >= 0).all()
        torch.testing.assert_close(command._buf_variant[selected], desired)
    assert not torch.equal(desired, command._env.keyboard_variants.variant_ids)


def test_candidate_sampling_uses_only_the_requested_noncontiguous_cohort(monkeypatch):
    command = _command()
    monkeypatch.setattr(wp, "synchronize", lambda: pytest.fail("Sampling must preserve the task's stream ordering"))
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


def test_prospective_sampling_uses_requested_layout_without_changing_committed_membership():
    command = _command()
    original = command.key_joints.dense_active().clone()
    requested = torch.tensor([2, 0])
    keys, counts, backspace = command._sampling_keys(torch.tensor([4, 1]), requested)
    assert wp.to_torch(keys)[0, :4].tolist() == [7, 8, 9, 10]
    assert wp.to_torch(keys)[1, :3].tolist() == [0, 1, 2]
    assert wp.to_torch(counts).tolist() == [4, 3]
    assert wp.to_torch(backspace).tolist() == [True, True]
    worlds = torch.tensor([4, 1])
    for sample in (command._oversample(64, worlds, requested), command._sample_diverse_states(16, worlds, requested)):
        for tokens in sample[:2]:
            assert torch.isin(tokens[tokens >= 0], torch.tensor([0, 1, 2, 7, 8, 9, 10])).all()
    torch.testing.assert_close(command.key_joints.dense_active(), original, rtol=0, atol=0)
    with pytest.raises(ValueError, match="cohort"):
        command._sampling_keys(torch.tensor([4]), requested)


def test_homogeneous_sources_exclude_incompatible_superset_bank_rows():
    command = _command(False)
    command._cur_sample_eps, command._cur_normal_weight, command._cur_buffer_size = 1e-4, 0.1, 5
    command.success_monitor = SimpleNamespace(target_weights=lambda: torch.ones(5))
    command._buffer_compatible = torch.tensor([False, True, False, True, False])
    for replay_only in (False, True):
        command.cfg.reset.replay_only = replay_only
        source = command._sample_sources(torch.arange(200))
        assert torch.isin(source, torch.tensor([1, 3] if replay_only else [-1, 1, 3])).all()


def test_homogeneous_bank_requires_explicit_ambiguous_variant_and_valid_typing_schema(tmp_path, monkeypatch):
    import hashlib

    from isaaclab_tasks.contrib.keyboard.keyboard_worlds import KeyboardWorlds
    from isaaclab_tasks.contrib.keyboard.keyboards import keyboard_geometry

    command = _command(False)
    command._cur_buffer_size = 2
    command.cfg.reset.bank_variant = None
    command.cfg.reset_roots = command.cfg.reset_coords = command.cfg.reset_dofs = SimpleNamespace(width=1)
    command._env.cfg = SimpleNamespace(scene=SimpleNamespace(keyboard=SimpleNamespace(spawn=object())))
    labels = ["a", "b", "backspace"]
    layout = SimpleNamespace(active_keys=[SimpleNamespace(label=label) for label in labels])
    monkeypatch.setattr(keyboard_geometry, "generate_keyboard", lambda _: layout)
    bank = {
        "_buf_state": torch.zeros((2, 9)),
        "_buf_target": torch.tensor([[0, -1, -1], [1, -1, -1]]),
        "_buf_typed": torch.full((2, 3), -1),
        "_buf_target_len": torch.ones(2, dtype=torch.long),
        "_buf_typed_len": torch.zeros(2, dtype=torch.long),
        "_buf_reach": torch.zeros(2),
        "_buf_variant": torch.tensor([0, 1]),
    }
    contract = {
        "format": 1,
        "buffer_size": 2,
        "max_len": 3,
        "root_width": 1,
        "coord_width": 1,
        "dof_width": 1,
        "active_labels": [labels, labels],
    }
    path = tmp_path / "bank.pt"

    def save():
        manifest = {
            name: {"sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()} for name, value in bank.items()
        }
        torch.save({"contract": contract, "bank": bank, "tensor_manifest": manifest, "avg_distance": 1.0}, path)

    save()
    with pytest.raises(ValueError, match="ambiguous"):
        command._load_buffer(path)
    command.cfg.reset.bank_variant = 0
    command._load_buffer(path)
    assert command._buffer_compatible.tolist() == [True, False]
    command.cfg.reset.bank_variant = 1
    command._load_buffer(path)
    assert command._buffer_compatible.tolist() == [False, True]
    contract.update(format=2, robot_counts=[1, 2], arm_order=["Robot", "Robot_1"])
    command.cfg.reset.bank_variant = None
    save()
    command._load_buffer(path)
    assert command._buffer_compatible.tolist() == [True, False]
    native = object.__new__(KeyboardWorlds)
    native.layouts, native.robot_counts = (layout, layout), (2, 1)
    command._env.keyboard_variants = native
    with pytest.raises(ValueError, match="arm topology differs"):
        command._load_buffer(path)
    command._env.keyboard_variants = None
    contract["arm_order"] = ["Robot_1", "Robot"]
    save()
    with pytest.raises(ValueError, match="arm topology or ordering"):
        command._load_buffer(path)
    contract.update(format=1, robot_counts=[1, 1])
    command.cfg.reset.bank_variant = 1
    contract["active_labels"][1] = ["other"]
    save()
    with pytest.raises(ValueError, match="matching keyboard labels"):
        command._load_buffer(path)
    command.cfg.reset.bank_variant = 0
    bank["_buf_typed_len"][0] = 4
    save()
    with pytest.raises(ValueError, match="lengths"):
        command._load_buffer(path)
    bank["_buf_typed_len"][0] = 0
    bank["_buf_target"][0, 0] = -1
    save()
    with pytest.raises(ValueError, match="declared lengths"):
        command._load_buffer(path)


@pytest.mark.parametrize("publish", [False, True])
@pytest.mark.parametrize("inverse_fails", [False, True])
def test_native_normal_ik_completes_prospective_payload_before_publication_and_preserves_rng(
    monkeypatch, publish, inverse_fails
):
    import newton

    from isaaclab_tasks.contrib.keyboard.keyboard_worlds import KeyboardWorlds
    from isaaclab_tasks.contrib.keyboard.mdp.reset import ResetKinematics, sample_root_poses

    builder = newton.ModelBuilder()
    builder.begin_world()
    root, tip = (builder.add_link(mass=1.0) for _ in range(2))
    fixed = builder.add_joint_fixed(parent=-1, child=root)
    slider = builder.add_joint_prismatic(parent=root, child=tip, axis=(1, 0, 0))
    builder.add_articulation([fixed, slider])
    builder.end_world()
    model = builder.finalize("cpu")
    bank = object.__new__(KeyboardWorlds)
    bank.variant_ids = torch.zeros(3, dtype=torch.long)
    bank.backspace_slots = torch.tensor([1, 1])
    bank.reset_defaults = torch.zeros((2, 16))
    bank.reset_defaults[:, 6] = bank.reset_defaults[:, 13] = 1
    bank.reset_defaults[0, 7], bank.reset_defaults[1, 7] = 0.05, 0.3
    bank.reset_keyboard_roots = torch.tensor([[False, True], [False, True]])
    bank.reset_key_root = torch.ones((2, 2), dtype=torch.long)
    bank.reset_key_local = torch.zeros((2, 2, 3))
    bank.reset_robot_columns = torch.tensor([[[0]], [[0]]])
    bank.reset_ik_columns = torch.tensor([0])
    bank.reset_robot_roots = torch.tensor([[0], [0]])
    bank.reset_robot_limits = torch.tensor([-1.0, 1.0]).expand(2, 1, 1, 2)
    bank.reset_kinematics = ResetKinematics(model, [0], [0], [tip], 3, ((0, 0, 0),))
    desired = torch.tensor([1, 0, 1])
    command = object.__new__(LetterTypingCommand)
    published = []

    def forbidden(*args, **kwargs):
        raise AssertionError("Native staging must not read or write live selections")

    selected = SimpleNamespace(width=1, write_state=forbidden, read_state=forbidden, dense_active=forbidden)
    command._env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        keyboard_variants=bank,
        reset_variant_ids=lambda ids: desired[ids],
        forward=forbidden,
        invalidate_fk=forbidden,
        restore_reset_snapshot=lambda ids, variants, payload: published.append(
            (ids.clone(), variants.clone(), payload.clone())
        ),
    )
    ik = SimpleNamespace(joints=selected, bodies=selected)
    command.cfg = SimpleNamespace(
        robot_joints=selected,
        robot_dofs=selected,
        reset_roots=SimpleNamespace(width=2),
        reset_coords=SimpleNamespace(width=1),
        reset=SimpleNamespace(
            ik=ik,
            ik_seed_joint_noise=0.01,
            pre_solve_reset=SimpleNamespace(params={"pose_range": {}, "velocity_range": {}}),
        ),
    )
    command._reset_ik, command._ik_iters = ik, (1, 1)
    command._default_robot_q = torch.zeros((3, 1))
    command._robot_limits = command._ik_limits = torch.tensor([-1.0, 1.0]).expand(3, 1, 2)
    command._ik_offsets, command._ik_hover = torch.zeros((1, 3)), torch.zeros(3)
    command.target = torch.zeros((3, 1), dtype=torch.long)
    command.target_len = torch.ones(3, dtype=torch.long)
    command.typed_len = command.prefix_len = torch.zeros(3, dtype=torch.long)
    monkeypatch.setattr(wp, "synchronize", lambda: pytest.fail("Reset IK must preserve the task's stream ordering"))
    ids = torch.tensor([2, 0])
    before = bank.reset_defaults.clone()
    torch.manual_seed(721)
    if inverse_fails:
        monkeypatch.setattr(
            torch.linalg,
            "inv_ex",
            lambda matrix: (torch.zeros_like(matrix), torch.ones(matrix.shape[:-2], dtype=torch.int32)),
        )
        with pytest.raises(RuntimeError, match="position inversion failed"):
            command._solve_reset_pose(ids, publish=publish)
        assert not published
        torch.testing.assert_close(bank.reset_defaults, before, atol=0, rtol=0)
        return
    completed = command._solve_reset_pose(ids, publish=publish)
    final_rng = torch.random.get_rng_state()
    assert len(published) == int(publish)
    if publish:
        actual_ids, actual_variants, payload = published[0]
    else:
        actual_ids, payload = completed
        actual_variants = desired[actual_ids]
    torch.testing.assert_close(actual_ids, ids)
    torch.testing.assert_close(actual_variants, desired[ids])
    torch.testing.assert_close(payload[:, 7], torch.full((2,), 0.3))
    torch.testing.assert_close(payload[:, 15], torch.zeros(2))
    torch.testing.assert_close(bank.reset_defaults, before, atol=0, rtol=0)
    assert bank.variant_ids.tolist() == [0, 0, 0]
    torch.manual_seed(721)
    sample_root_poses(before[desired[ids], :14].reshape(2, 2, 7), {}, {})
    seed = (torch.rand((2, 1)) * 2 - 1) * 0.01
    torch.randint(1, 2, (3,))
    assert torch.equal(final_rng, torch.random.get_rng_state())
    expected = seed[:, 0] + torch.clamp((0.3 - seed[:, 0]) / (1 + 0.05**2), -0.2, 0.2)
    torch.testing.assert_close(payload[:, 14], expected, atol=1e-7, rtol=0)


def test_native_reset_uses_each_arms_next_owned_key_root_and_limits_without_overwriting_keys():
    import newton

    from isaaclab_tasks.contrib.keyboard.keyboard_worlds import KeyboardWorlds
    from isaaclab_tasks.contrib.keyboard.newton_selection import NewtonSelections
    from isaaclab_tasks.contrib.keyboard.selection_paths import NewtonSelectorCfg

    bank = object.__new__(KeyboardWorlds)
    bank.env = SimpleNamespace(device="cpu", num_envs=3)
    bank.robot_counts = (1, 2)
    bank.key_arm_ids = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 1, 1, 1]])
    bank.variant_ids = torch.zeros(3, dtype=torch.long)
    bank.backspace_slots = torch.full((2,), 4, dtype=torch.long)
    bank._q_offset, bank._qd_offset = 21, 28
    bank._sources, bank._solvers = [], []
    for count in bank.robot_counts:
        builder = newton.ModelBuilder()
        builder.begin_world()
        for arm in range(count):
            name = "Robot" if arm == 0 else "Robot_1"
            placement = wp.transform(((arm * 2 - 1) * 0.2 if count == 2 else 0.0, 0.0, 0.0), wp.quat_identity())
            root = builder.add_link(label=f"/{name}/base", xform=placement, mass=1.0)
            tip = builder.add_link(label=f"/{name}/tip", xform=placement, mass=1.0)
            fixed = builder.add_joint_fixed(parent=-1, child=root, parent_xform=placement)
            limit = (0.08 if arm == 0 else 0.07) if count == 2 else 1.0
            joint = builder.add_joint_prismatic(
                parent=root,
                child=tip,
                axis=(1, 0, 0),
                label=f"/{name}/joint",
                limit_lower=-limit,
                limit_upper=limit,
            )
            finger = builder.add_link(label=f"/{name}/tip_right", mass=1.0)
            finger_joint = builder.add_joint_fixed(
                parent=tip, child=finger, parent_xform=wp.transform((-0.02, 0, 0), wp.quat_identity())
            )
            builder.add_articulation([fixed, joint, finger_joint])
        keyboard_x = 0.1 if count == 2 else 0.05
        placement = wp.transform((keyboard_x, 0, 0), wp.quat_identity())
        root = builder.add_link(label="/Keyboard/base", xform=placement, mass=1.0)
        joints = [builder.add_joint_fixed(parent=-1, child=root, parent_xform=placement)]
        for key, x in enumerate((-0.2, -0.18, 0.09, 0.14, 0.2)):
            body = builder.add_link(
                label=f"/Keyboard/key{key}", xform=wp.transform((keyboard_x + x, 0, 0), wp.quat_identity()), mass=1.0
            )
            joints.append(
                builder.add_joint_prismatic(
                    parent=root,
                    child=body,
                    axis=(0, 0, 1),
                    label=f"/Keyboard/joint{key}",
                    parent_xform=wp.transform((x, 0, 0), wp.quat_identity()),
                )
            )
            builder.joint_q[-1] = 0.011 + 0.001 * key
        builder.add_articulation(joints)
        builder.end_world()
        model = builder.finalize("cpu")
        bank._sources.append(NewtonSelections(model))
        bank._solvers.append(SimpleNamespace(model=model))
    body, q, qd = (getattr(newton.Model.AttributeFrequency, name) for name in ("BODY", "JOINT_COORD", "JOINT_DOF"))
    robot_path, key_path = "/Robot(?:_1)?/joint", "/Keyboard/joint.*"
    cfg = SimpleNamespace(
        robot_joints=NewtonSelectorCfg(q, robot_path, policy_width=2),
        robot_dofs=NewtonSelectorCfg(qd, robot_path, policy_width=2),
        reset_roots=NewtonSelectorCfg(body, ("/Robot(?:_1)?/base", "/Keyboard/base"), policy_width=3),
        reset_coords=NewtonSelectorCfg(q, (robot_path, key_path), policy_width=7),
        reset_dofs=NewtonSelectorCfg(qd, (robot_path, key_path), policy_width=7),
        key_bodies=NewtonSelectorCfg(body, "/Keyboard/key.*", policy_width=5),
        soft_joint_pos_limit_factor=1.0,
        reset=SimpleNamespace(
            ik=SimpleNamespace(
                joints=NewtonSelectorCfg(q, robot_path, policy_width=2),
                dofs=NewtonSelectorCfg(qd, robot_path, policy_width=2),
                bodies=NewtonSelectorCfg(body, ("/Robot(?:_1)?/tip", "/Robot(?:_1)?/tip_right"), policy_width=4),
                tip_offsets=((0, 0, 0), (0, 0, 0)),
            ),
            ik_seed_joint_noise=0.0,
            pre_solve_reset=SimpleNamespace(
                params={
                    "roots": NewtonSelectorCfg(body, "/Keyboard/base"),
                    "pose_range": {},
                    "velocity_range": {},
                }
            ),
        ),
    )
    bank._prepare_reset_staging(cfg)
    before = bank.reset_defaults.clone()
    command = object.__new__(LetterTypingCommand)
    command._env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        keyboard_variants=bank,
        reset_variant_ids=lambda ids: torch.tensor([1, 0, 0])[ids],
    )
    command.cfg = cfg
    cfg.reset_roots = SimpleNamespace(width=3)
    cfg.reset_coords = SimpleNamespace(width=7)
    command._reset_ik, command._ik_iters = cfg.reset.ik, (1, 1)
    # Live rows describe a different prototype; native reset must use the prospective limits.
    command._robot_limits = command._ik_limits = torch.zeros((3, 2, 2))
    command._ik_offsets, command._ik_hover = torch.zeros((2, 3)), torch.zeros(3)
    cases = (
        ([0, 1, 2, 3], 0, 0, 0.01 / 1.0025),  # Skip other-half keys, choose the first owned future key.
        ([2, 0, 0, 3, 2], 1, 1, 0.04 / 1.0025),  # Ignore consumed keys; repeats retain sequence order.
        ([0, 2, 3], 0, 1, 0.07),  # Backspace takes priority over future letters for its owner.
        ([0], 0, 0, None),  # No future work for the right arm: stay within its half.
        ([0, 3], 2, 3, 0.07),  # Completed prefix with an extra character still needs Backspace.
    )
    for sequence, prefix, typed, right_q in cases:
        command.target = torch.full((3, 6), -1, dtype=torch.long)
        command.target[:, 0] = 0
        command.target[0, : len(sequence)] = torch.tensor(sequence)
        command.target_len = torch.ones(3, dtype=torch.long)
        command.target_len[0] = len(sequence)
        command.prefix_len = torch.tensor([prefix, 0, 0])
        command.typed_len = torch.tensor([typed, 0, 0])
        torch.manual_seed(721)
        ids, snapshot = command._solve_reset_pose(torch.tensor([2, 0]), publish=False)
        assert ids.tolist() == [2, 0]
        torch.testing.assert_close(
            snapshot[0, 21:28], torch.tensor([-0.13 / 1.0025, 0.011, 0.012, 0.013, 0.014, 0.015, 0.0])
        )
        torch.testing.assert_close(snapshot[1, 21], torch.tensor(0.08))
        if right_q is None:
            assert torch.isclose(snapshot[1, 22], torch.tensor([0.01 / 1.0025, 0.04 / 1.0025, 0.07]), atol=1e-7).any()
        else:
            torch.testing.assert_close(snapshot[1, 22], torch.tensor(right_q), atol=1e-7, rtol=0)
        torch.testing.assert_close(snapshot[1, 23:28], torch.tensor([0.011, 0.012, 0.013, 0.014, 0.015]))
        torch.testing.assert_close(snapshot[:, 28:], torch.zeros((2, 7)))
        torch.testing.assert_close(bank.reset_defaults, before, rtol=0, atol=0)
        assert bank.variant_ids.tolist() == [0, 0, 0]


def test_reset_approach_keeps_camera_up_independently_of_seed_roll():
    import math

    from isaaclab.utils.math import quat_apply, quat_from_angle_axis

    command = object.__new__(LetterTypingCommand)
    command._env = SimpleNamespace(num_envs=3, device="cpu")
    command._ik_finger_axis = torch.tensor([[0.0, 0.0, -1.0]]).expand(3, -1)
    command._ik_roll, command._ik_pitch, command._ik_yaw = 0.0, math.pi / 4, 0.0
    seeds = quat_from_angle_axis(torch.tensor([0.0, math.pi / 2, -math.pi / 2]), command._ik_finger_axis)
    orientation = command._approach_target_quat(seeds)
    finger = quat_apply(orientation, command._ik_finger_axis)
    camera = quat_apply(orientation, torch.tensor([[0.0, 1.0, 0.0]]).expand(3, -1))
    torch.testing.assert_close(finger, torch.tensor([[2**-0.5, 0.0, -(2**-0.5)]]).expand(3, -1), atol=1e-6, rtol=0)
    torch.testing.assert_close(camera, torch.tensor([[2**-0.5, 0.0, 2**-0.5]]).expand(3, -1), atol=1e-6, rtol=0)


def test_explicit_all_world_cohort_preserves_baseline_samples():
    implicit, explicit = _command(False), _command(False)
    for expected, actual in zip(
        implicit._oversample(64), explicit._oversample(64, explicit._env.all_env_ids), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("heterogeneous", "pending_first_reset"), [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("fail_during_build", [False, True])
def test_buffer_restores_state_and_pending_reset(monkeypatch, heterogeneous, pending_first_reset, fail_during_build):
    command = _command(heterogeneous)
    env = command._env
    original = torch.arange(5, dtype=torch.float32)[:, None]
    physical = original.clone()
    requested, solved, finished = [], [], []
    active_cohort = None
    pending = None
    if heterogeneous:
        pending = (env.keyboard_variants.variant_ids + int(pending_first_reset)) % 3
        original_pending = pending.clone()
        env.reset_variant_ids = lambda ids: pending[ids]

    def curriculum_worlds(variant):
        nonlocal active_cohort
        active_cohort = (
            env.all_env_ids if not heterogeneous else env.all_env_ids[env.keyboard_variants.variant_ids == variant]
        )
        requested.append(variant)
        if pending is not None:
            pending[active_cohort] = variant
        return active_cohort.flip(0)  # Ordering and gaps must not be mistaken for a world prefix.

    def solve(ids):
        assert torch.isin(ids, active_cohort).all()
        solved.extend(ids.tolist())
        physical[ids] = command.target[ids, :1].float() + 100
        if fail_during_build:
            raise RuntimeError("IK failed")

    env.curriculum_worlds = curriculum_worlds

    def finish(variants):
        finished.append(None if variants is None else variants.clone())
        if pending is not None:
            pending.copy_(variants)

    env.finish_curriculum = finish
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
    command.cfg.reset.ik = SimpleNamespace(bodies=SimpleNamespace(read_state=lambda _: body_poses))
    command.cfg.reset_roots = command.cfg.reset_coords = command.cfg.reset_dofs = None
    command._ik_offsets = torch.zeros((1, 3))
    command._ik_hover = torch.zeros(3)
    command.target_key_pos_w = lambda: torch.zeros((5, 3))
    monkeypatch.setattr(typing_commands, "capture_reset_state", lambda _, ids, *args: physical[ids].clone())

    def restore(ids, variants, snapshot):
        if heterogeneous:
            torch.testing.assert_close(variants, env.keyboard_variants.variant_ids)
        physical[ids] = snapshot

    env.restore_reset_snapshot = restore
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
        torch.testing.assert_close(finished[0], original_pending)
        torch.testing.assert_close(pending, original_pending)
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


@pytest.mark.parametrize("keys,width", [(6, 1), (108, 4)])
def test_typing_kernels_preserve_edges_order_overflow_and_prefix_extremes(keys, width):
    import numpy as np
    import warp as wp

    from isaaclab_tasks.contrib.keyboard.mdp.commands.typing_commands import _advance_typing, _typing_metrics

    rng = np.random.default_rng(891)
    worlds = 80
    pressed = rng.random((worlds, keys)) < 0.4
    previous = rng.random((worlds, keys)) < 0.4
    just_reset = np.arange(worlds) % 4 == 0
    backspace = rng.integers(0, keys, worlds, dtype=np.int64)
    lengths = rng.integers(0, width + 1, worlds, dtype=np.int64)
    typed = rng.integers(0, keys, (worlds, width), dtype=np.int64)
    typed[np.arange(width) >= lengths[:, None]] = -1
    expected = typed.copy()
    expected_lengths = lengths.copy()
    for world in range(worlds):
        if just_reset[world]:
            continue
        edges = set(np.flatnonzero(pressed[world] & ~previous[world]).tolist())
        letters = typed[world, : lengths[world]].tolist()
        if int(backspace[world]) in edges:
            letters = letters[:-1]
        letters = (letters + sorted(edges - {int(backspace[world])}))[:width]
        expected[world] = letters + [-1] * (width - len(letters))
        expected_lengths[world] = len(letters)
    inputs = [wp.array(value, device="cpu") for value in (backspace, previous, just_reset, typed, lengths)]
    q = _dense_scalar_field(-torch.from_numpy(pressed).float())
    lower = _dense_scalar_field(-torch.ones(worlds, keys))
    upper = _dense_scalar_field(torch.zeros(worlds, keys))
    wp.launch(_advance_typing, worlds, inputs=[q, lower, upper, 0.5, *inputs, True], device="cpu")
    np.testing.assert_array_equal(inputs[1].numpy(), pressed)
    assert not inputs[2].numpy().any()
    np.testing.assert_array_equal(inputs[3].numpy(), expected)
    np.testing.assert_array_equal(inputs[4].numpy(), expected_lengths)

    target = rng.integers(0, keys, (worlds, width), dtype=np.int64)
    target[::2] = expected[::2]
    target_lengths = rng.integers(0, width + 1, worlds, dtype=np.int64)
    old_max = rng.integers(0, width + 1, worlds, dtype=np.int64)
    old_min = rng.integers(0, width + 1, worlds, dtype=np.int64)
    expected_prefix = np.zeros(worlds, dtype=np.int64)
    for world in range(worlds):
        equal = (
            target[world, : min(target_lengths[world], expected_lengths[world])]
            == expected[world, : min(target_lengths[world], expected_lengths[world])]
        )
        expected_prefix[world] = np.argmin(equal) if not equal.all() else len(equal)
    outputs = [wp.zeros(worlds, dtype=dtype, device="cpu") for dtype in (wp.int64, float)]
    maxima, minima = wp.array(old_max, device="cpu"), wp.array(old_min, device="cpu")
    high, low = wp.zeros(worlds, dtype=wp.bool, device="cpu"), wp.zeros(worlds, dtype=wp.bool, device="cpu")
    wp.launch(
        _typing_metrics,
        worlds,
        inputs=[
            wp.array(target, device="cpu"),
            inputs[3],
            wp.array(target_lengths, device="cpu"),
            inputs[4],
            *outputs,
            maxima,
            minima,
            high,
            low,
        ],
        device="cpu",
    )
    for actual, wanted in (
        (outputs[0], expected_prefix),
        (outputs[1], target_lengths + expected_lengths - 2 * expected_prefix),
        (maxima, np.maximum(old_max, expected_prefix)),
        (minima, np.minimum(old_min, expected_prefix)),
        (high, expected_prefix > old_max),
        (low, expected_prefix < old_min),
    ):
        np.testing.assert_array_equal(actual.numpy(), wanted)


def _preview_command():
    command = _command(False)
    command.cfg.resampling_time_range = (10.0, 10.0)
    command.cfg.actuation_fraction = 0.5
    command.cfg.key_dofs = SimpleNamespace(
        scalar_field=lambda source, name: _dense_scalar_field(
            -torch.ones(5, 12) if name == "joint_limit_lower" else torch.zeros(5, 12)
        )
    )
    positions = torch.zeros(5, 12)
    command.key_joints.scalar_field = lambda source, name: _dense_scalar_field(
        positions, command.key_joints.dense_active()
    )
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
    command._cur_enabled = command._episode_reset = False
    command._env.command_manager = SimpleNamespace(get_term=lambda name: command)
    return command, positions


def test_typing_transition_handles_simultaneous_presses_backspace_and_reset_edges_once():
    from isaaclab_tasks.contrib.keyboard.mdp.rewards import letter_typing_progress, typing_success
    from isaaclab_tasks.contrib.keyboard.mdp.terminations import typing_complete

    command, positions = _preview_command()
    command.cfg.resampling_time_range = None
    # Both collaborating arms press during one control; extra keys prevent success.
    positions[0, :2] = -0.75
    positions[1, :3] = -0.75
    command.typed[2, :2] = torch.tensor([0, 2])
    command.typed_len[2] = 2
    positions[2, 1] = positions[2, 3] = -0.75  # Backspace then the correct replacement.
    command._just_reset[3] = True
    positions[3, 0] = -0.75  # A reset-held key is adopted, not typed.
    command.advance()
    assert command.typed.tolist() == [[0, 1, -1], [0, 1, 2], [0, 1, -1], [-1, -1, -1], [-1, -1, -1]]
    assert typing_complete(command._env, "typing").tolist() == [True, False, True, False, False]
    assert typing_success(command._env, "typing").tolist() == [1, 0, 1, 0, 0]
    assert letter_typing_progress(command._env, "typing").tolist() == [1, 1, 1, 0, 0]
    before = command.typed.clone()
    command.compute(0.04)  # Manager housekeeping cannot consume another edge or erase this step's reward.
    torch.testing.assert_close(command.typed, before)
    assert command.new_high.tolist() == [True, True, True, False, False]
    command.advance()  # Held keys do not repeat on the following control.
    torch.testing.assert_close(command.typed, before)
    assert not command.new_high.any()


def test_episode_reset_adopts_snapshot_keys_before_the_first_action():
    command, positions = _preview_command()
    command.cfg.resampling_time_range = None
    command._env.episode_interrupted = torch.zeros(5, dtype=torch.bool)
    command._buffer_built = False
    command._reset_ik = command._buf_avg_distance = None
    command._start_distance = torch.ones(5, dtype=torch.long)
    command._distance_bands, command._split_last = (), {}
    # With one typeable key and no Backspace, resets always start with an empty buffer.
    active = torch.zeros(5, 12, dtype=torch.bool)
    active[:, 0] = True
    command.key_joints.dense_active = lambda: active
    command.cfg.letter_length = (1, 1)
    positions[0, 0] = -0.75
    command.reset(torch.arange(5))
    assert not command._just_reset.any()
    assert command._prev_pressed[:, 0].tolist() == [True, False, False, False, False]
    positions[1, 0] = -0.75
    command.advance()
    assert command.typed_len.tolist() == [0, 1, 0, 0, 0]
    assert command.distance.tolist() == [1, 0, 1, 1, 1]


@pytest.mark.parametrize("native", [False, True])
def test_completed_control_drives_typing_rewards_termination_and_terminal_observations(native):
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import (
        EventManager,
        ObservationGroupCfg,
        ObservationManager,
        ObservationTermCfg,
        RewardManager,
        RewardTermCfg,
        TerminationManager,
        TerminationTermCfg,
    )

    from isaaclab_tasks.contrib.keyboard.mdp.observations import typed_keys_onehot
    from isaaclab_tasks.contrib.keyboard.mdp.rewards import letter_typing_progress, typing_success
    from isaaclab_tasks.contrib.keyboard.mdp.terminations import typing_complete
    from isaaclab_tasks.contrib.keyboard.so101_env import SO101KeyboardEnv
    from isaaclab_tasks.contrib.keyboard.so101_env_cfg import EventCfg
    from isaaclab_tasks.contrib.keyboard.so101_population_env import SO101KeyboardPopulationEnv

    command, positions = _preview_command()
    command.cfg.resampling_time_range = None
    env = command._env
    env.physics_dt = env.step_dt = 0.04
    env.cfg = SimpleNamespace(
        decimation=1,
        sim=SimpleNamespace(render_interval=1, physics=None),
        compute_final_obs=True,
        redistribution_interval=1,
        redistribution_mode="episode_boundary",
    )
    env.episode_length_buf = torch.zeros(5, dtype=torch.long)
    env.common_step_counter = env._sim_step_counter = 0
    env.episode_interrupted = torch.zeros(5, dtype=torch.bool)
    env.extras = {}
    env.render_enabled = env.has_rtx_sensors = env._physics_handles_decimation = False
    env.video_recorders = []
    env._check_active = env.forward = lambda: None
    env.scene = SimpleNamespace(write_data_to_sim=lambda: None, update=lambda **kwargs: None)
    env.action_manager = SimpleNamespace(process_action=lambda action: None, apply_action=lambda: None)
    env.recorder_manager = SimpleNamespace(
        active_terms=[],
        record_pre_step=lambda: None,
        record_post_physics_decimation_step=lambda: None,
        record_pre_reset=lambda ids: None,
        record_post_reset=lambda ids: None,
    )

    def physics(**kwargs):
        positions[0, :2] = -0.75  # Both arms complete the shared sequence in this control.
        positions[1, :3] = -0.75  # An extra simultaneous key prevents success.
        positions[2, 0] = -0.75  # Progress on the exact timeout control must be credited.

    env.sim = SimpleNamespace(
        is_playing=lambda: True, is_rendering=False, step=physics, consume_reset_request=lambda: False
    )
    env.event_manager = EventManager({"typing": EventCfg().advance_typing}, env)
    env.command_manager.compute = command.compute
    env.termination_manager = TerminationManager(
        {
            "success": TerminationTermCfg(func=typing_complete, params={"command_name": "typing"}),
            "timeout": TerminationTermCfg(
                func=lambda env: torch.tensor([False, False, True, False, False]), time_out=True
            ),
        },
        env,
    )
    env.reward_manager = RewardManager(
        {
            "progress": RewardTermCfg(func=letter_typing_progress, weight=2.0, params={"command_name": "typing"}),
            "success": RewardTermCfg(func=typing_success, weight=50.0, params={"command_name": "typing"}),
        },
        env,
    )
    group = ObservationGroupCfg(concatenate_terms=True)
    group.typed = ObservationTermCfg(func=typed_keys_onehot, params={"command_name": "typing"})
    env.observation_manager = ObservationManager({"policy": group}, env)
    env._compute_final_observations = lambda: SO101KeyboardEnv._compute_final_observations(env)
    env.keyboard_variants = SimpleNamespace(
        desired_variant_ids=torch.zeros(5),
        variant_ids=torch.zeros(5),
        request_variants=lambda ids: None,
        redistribution_count=0,
        last_changed_worlds=0,
        last_redistribution_ms=0,
        populated_prototype_count=1,
        live_dof_count=6,
    )
    command._env.keyboard_variants.backspace_slots = torch.tensor([3])
    env.keyboard_variants.variant_ids = env.keyboard_variants.variant_ids.long()
    env._apply_variant_requests = lambda ids: torch.empty(0, dtype=torch.long)
    resets = []

    def reset(ids):
        resets.append(ids.tolist())
        command.typed[ids] = -1
        command.typed_len[ids] = command.prefix_len[ids] = command.max_prefix[ids] = command.min_prefix[ids] = 0
        command.distance[ids] = 2
        command.new_high[ids] = command.new_low[ids] = False
        command._just_reset[ids] = True
        command._update_command()

    env._reset_idx = reset
    step = SO101KeyboardPopulationEnv.step if native else ManagerBasedRLEnv.step
    obs, reward, terminated, truncated, extras = step(env, torch.zeros(5, 6))
    assert terminated.tolist() == [True, False, False, False, False]
    assert truncated.tolist() == [False, False, True, False, False]
    torch.testing.assert_close(reward, 0.04 * torch.tensor([52.0, 2.0, 2.0, 0.0, 0.0]))
    assert resets == [[0, 2]]
    assert extras["final_obs"]["policy"][0, [0, 13]].tolist() == [1, 1]
    assert extras["final_obs"]["policy"][2, 0] == 1
    assert obs["policy"][0].count_nonzero() == obs["policy"][2].count_nonzero() == 0
    assert command.typed[1].tolist() == [0, 1, 2]
    assert command.distance[1] == 1  # No delayed stale success may survive the extra press.


@pytest.mark.parametrize("timer", [None, "running", "expired"])
def test_final_observation_preserves_completed_typing_and_pre_reset_history(timer):
    from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg

    from isaaclab_tasks.contrib.keyboard.mdp.observations import typed_keys_onehot
    from isaaclab_tasks.contrib.keyboard.so101_env import SO101KeyboardEnv

    command, positions = _preview_command()
    command.cfg.resampling_time_range = None
    env = command._env
    env.sim = SimpleNamespace(is_playing=lambda: True)
    plain = ObservationGroupCfg(concatenate_terms=True)
    plain.typed = ObservationTermCfg(func=typed_keys_onehot, params={"command_name": "typing"})
    history = ObservationGroupCfg(concatenate_terms=True)
    history.typed = ObservationTermCfg(func=typed_keys_onehot, params={"command_name": "typing"}, history_length=2)
    manager = ObservationManager({"plain": plain, "history": history}, env)
    env.observation_manager, env.step_dt = manager, 0.04
    manager.compute(update_history=True)
    positions[0, :2] = -0.75
    command.advance()
    if timer is not None:
        command.cfg.resampling_time_range = (10.0, 10.0)
        command.time_left[1] = 0.01 if timer == "expired" else 10.0
    references = {name: value for name, value in vars(command).items() if isinstance(value, torch.Tensor)}
    before = {name: value.clone() for name, value in references.items()}
    seed = command._resample_seed
    rng = torch.random.get_rng_state().clone()
    terminal = SO101KeyboardEnv._compute_final_observations(env)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert command._resample_seed == seed
    for name, original in references.items():
        assert getattr(command, name) is original
        torch.testing.assert_close(original, before[name], rtol=0, atol=0)
    assert command.typed[0, :2].tolist() == [0, 1]
    assert command.distance[0] == 0
    assert terminal["plain"][0, 0] == terminal["plain"][0, 13] == 1
    assert terminal["history"][0, 36] == terminal["history"][0, 49] == 1
    assert manager._group_obs_term_history_buffer["history"]["typed"]._num_pushes.tolist() == [1] * 5
    command.compute(0.04)
    actual = manager.compute(update_history=True)
    for name in terminal:
        torch.testing.assert_close(terminal[name], actual[name], rtol=0, atol=0)


@pytest.mark.parametrize("source", [(2, -1, 1), (2, 0, 1), (-1, -1, -1)])
@pytest.mark.parametrize("invalid_replay", [False, True])
@pytest.mark.parametrize("replay_only", [False, True])
def test_native_reset_publishes_one_complete_mixed_payload(monkeypatch, source, invalid_replay, replay_only):
    from isaaclab.managers import CommandTerm

    from isaaclab_tasks.contrib.keyboard.keyboard_worlds import KeyboardWorlds

    command = _command()
    if replay_only and -1 in source:
        pytest.skip("Replay-only sampling cannot return a normal-reset source.")
    command.cfg.reset.replay_only = replay_only
    bank = object.__new__(KeyboardWorlds)
    bank.__dict__.update(vars(command._env.keyboard_variants))
    bank.reset_defaults = torch.zeros((3, 2))
    command._env.keyboard_variants = bank
    desired = torch.tensor([1, 0, 2, 1, 2])
    command._env.reset_variant_ids = lambda ids: desired[ids]
    command._env.episode_interrupted = torch.zeros(5, dtype=torch.bool)
    command._env_source = torch.full((5,), -1)
    if source == (2, 0, 1) and not invalid_replay:
        del bank.reset_defaults  # A replay-only bank has no normal IK staging metadata.
    command._cur_enabled = command._buffer_built = True
    command._sample_sources = lambda ids: torch.tensor(source)
    command.success_monitor = SuccessMonitor(SuccessMonitorCfg(), 1, 3, "cpu")
    command._buf_variant = torch.arange(3)
    command._buf_state = torch.tensor([[100.0, 200.0], [101.0, 201.0], [102.0, 202.0]])
    command._buf_target = torch.tensor([[0, -1, -1], [4, -1, -1], [7, -1, -1]])
    command._buf_typed = torch.full((3, 3), -1)
    command._buf_target_len, command._buf_typed_len = torch.ones(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)
    if invalid_replay:
        command._buf_target[2, 0] = 0  # Active in the old world, invalid in requested prototype2.
    command.target, command.typed = torch.full((5, 3), -1), torch.full((5, 3), -1)
    for name in ("target_len", "typed_len", "prefix_len", "max_prefix", "min_prefix"):
        setattr(command, name, torch.zeros(5, dtype=torch.long))
    for name in ("new_high", "new_low", "_just_reset"):
        setattr(command, name, torch.zeros(5, dtype=torch.bool))
    command._prev_pressed = torch.zeros((5, 12), dtype=torch.bool)
    command.distance = torch.ones(5)
    command._start_distance = torch.ones(5, dtype=torch.long)
    command._distance_bands, command._split_last, command._buf_avg_distance = (), {}, None
    command._reset_ik = object()
    publications, solved = [], []
    original_variants = bank.variant_ids.clone()

    def normal(ids):
        command.target[ids] = -1
        command.target[ids, 0] = torch.tensor([0, 4, 7])[desired[ids]]
        command.typed[ids] = -1
        command.target_len[ids], command.typed_len[ids] = 1, 0
        command.distance[ids] = 1

    def solve(ids, *, publish):
        assert not publish and not publications
        torch.testing.assert_close(bank.variant_ids, original_variants)
        solved.extend(ids.tolist())
        # Deliberately return a different order to exercise actor-to-payload row mapping.
        return ids.flip(0), torch.stack((ids.flip(0) + 900, ids.flip(0) + 800), dim=1).float()

    def publish(ids, variants, payload):
        publications.append((ids.clone(), variants.clone(), payload.clone()))
        bank.variant_ids[ids] = variants

    def base_reset(term, ids):
        term._resample_command(ids)
        return {}

    command._resample_normal, command._solve_reset_pose = normal, solve
    command._env.restore_reset_snapshot = publish
    command.key_joints.dense_active = lambda: pytest.fail("Deferred replay must not read committed membership")
    monkeypatch.setattr(CommandTerm, "reset", base_reset)
    ids = torch.tensor([4, 1, 3])
    command.reset(ids)
    assert len(publications) == 1
    actual_ids, variants, payload = publications[0]
    torch.testing.assert_close(actual_ids, ids)
    torch.testing.assert_close(variants, desired[ids])
    expected = []
    expected_normal = []
    for actor, snapshot in zip(ids.tolist(), source, strict=True):
        if snapshot < 0 or (invalid_replay and snapshot == 2):
            expected.append([actor + 900.0, actor + 800.0])
            expected_normal.append(actor)
        else:
            expected.append([snapshot + 100.0, snapshot + 200.0])
    torch.testing.assert_close(payload, torch.tensor(expected))
    assert solved == expected_normal
    assert not command._episode_reset


def test_failed_command_reset_always_clears_episode_window(monkeypatch):
    from isaaclab.managers import CommandTerm

    command = _command(False)
    command._env.episode_interrupted = torch.zeros(5, dtype=torch.bool)
    command._cur_enabled = command._buffer_built = False
    command.distance = torch.ones(5)
    command._start_distance = torch.ones(5, dtype=torch.long)
    command._distance_bands, command._split_last = (), {}

    def fail(term, ids):
        assert term._episode_reset
        raise RuntimeError("sample failed")

    monkeypatch.setattr(CommandTerm, "reset", fail)
    with pytest.raises(RuntimeError, match="sample failed"):
        command.reset(torch.tensor([4, 1]))
    assert not command._episode_reset


def test_snapshot_restore_updates_only_requested_typing_rows_and_resets_watermarks():
    command = _command()
    command._buf_state = torch.zeros((3, 2))
    command._buf_variant = torch.tensor([2, 1, 1])
    command._buf_target = torch.tensor([[7, 8, -1], [4, -1, -1], [5, 6, -1]])
    command._buf_typed = torch.tensor([[7, 9, 8], [4, -1, -1], [-1, -1, -1]])
    command._buf_target_len = torch.tensor([2, 1, 2])
    command._buf_typed_len = torch.tensor([3, 1, 0])
    command.target, command.typed = torch.full((5, 3), 123), torch.full((5, 3), 456)
    for name in ("target_len", "typed_len", "prefix_len", "max_prefix", "min_prefix"):
        setattr(command, name, torch.full((5,), 7, dtype=torch.long))
    for name in ("new_high", "new_low", "_just_reset"):
        setattr(command, name, torch.ones(5, dtype=torch.bool))
    command._prev_pressed = torch.ones((5, 12), dtype=torch.bool)
    command.distance = torch.full((5,), 77.0)
    names = (
        "target",
        "typed",
        "target_len",
        "typed_len",
        "prefix_len",
        "max_prefix",
        "min_prefix",
        "new_high",
        "new_low",
        "_just_reset",
        "_prev_pressed",
        "distance",
    )
    before = {name: getattr(command, name).clone() for name in names}
    ids = torch.tensor([3, 1, 4])
    command._restore_snapshot(ids, torch.arange(3), publish=False)
    torch.testing.assert_close(command.target[ids], command._buf_target)
    torch.testing.assert_close(command.typed[ids], command._buf_typed)
    torch.testing.assert_close(command.target_len[ids], command._buf_target_len)
    torch.testing.assert_close(command.typed_len[ids], command._buf_typed_len)
    for name in ("prefix_len", "min_prefix", "max_prefix"):
        assert getattr(command, name)[ids].tolist() == [1, 1, 0]
    assert command.distance[ids].tolist() == [3, 0, 2]
    assert command._just_reset[ids].all()
    assert not command._prev_pressed[ids].any()
    assert not command.new_high[ids].any() and not command.new_low[ids].any()
    for name in names:
        torch.testing.assert_close(getattr(command, name)[[0, 2]], before[name][[0, 2]], rtol=0, atol=0)
