# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for exact motion-imitation event schedules."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.managers import EventTermCfg, SceneEntityCfg

from isaaclab_tasks.core.multi_task.motion.config.environment import MotionEventsPresetsCfg
from isaaclab_tasks.core.multi_task.motion.mdp.commands import MotionStatePayload
from isaaclab_tasks.core.multi_task.motion.mdp.events import MotionPushVelocity
from isaaclab_tasks.core.multi_task.motion.mdp.reset_sources import SmplMocapAndFallReset


class _Asset:
    def __init__(self, num_envs: int) -> None:
        root_velocity = torch.arange(num_envs * 6, dtype=torch.float32).view(num_envs, 6)
        self.data = SimpleNamespace(root_vel_w=SimpleNamespace(torch=root_velocity))
        self.writes: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.mask_writes: list[tuple[torch.Tensor, torch.Tensor]] = []

    def write_root_velocity_to_sim_mask(self, root_velocity: torch.Tensor, env_mask: wp.array) -> None:
        mask = wp.to_torch(env_mask).clone()
        self.mask_writes.append((root_velocity.clone(), mask))
        env_ids = torch.arange(mask.shape[0], device=mask.device)[mask]
        if env_ids.numel() > 0:
            selected_velocity = root_velocity[mask].clone()
            self.data.root_vel_w.torch[mask] = selected_velocity
            self.writes.append((selected_velocity, env_ids))


def _make_term(
    *,
    num_envs: int = 4,
    event_interval_seconds: float = 1.0,
    is_global_time: bool = True,
    interval_seconds: tuple[int, int] = (1, 3),
) -> tuple[MotionPushVelocity, _Asset]:
    asset = _Asset(num_envs)
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={"robot": asset},
    )
    cfg = EventTermCfg(
        func=MotionPushVelocity,
        mode="interval",
        interval_range_s=(event_interval_seconds, event_interval_seconds),
        is_global_time=is_global_time,
        resample_interval_on_reset=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "interval_seconds_integer_high_exclusive": interval_seconds,
            "linear_velocity_range_m_s": (-0.5, 0.5),
            "angular_velocity_range_rad_s": (-0.5, 0.5),
        },
    )
    return MotionPushVelocity(cfg, env), asset


def _step(term: MotionPushVelocity) -> None:
    params = term.cfg.params
    term(
        term._env,
        None,
        params["asset_cfg"],
        params["interval_seconds_integer_high_exclusive"],
        params["linear_velocity_range_m_s"],
        params["angular_velocity_range_rad_s"],
    )


def test_integer_second_intervals_match_released_schedule() -> None:
    term, _ = _make_term(num_envs=4096)

    assert term._interval_second_choices.tolist() == [1, 2]
    assert torch.all((term._interval_seconds == 1) | (term._interval_seconds == 2))


def test_g1_push_uses_global_one_second_event_cadence() -> None:
    push = MotionEventsPresetsCfg().g1_lafan.push

    assert push.interval_range_s == (1.0, 1.0)
    assert push.is_global_time
    assert not push.resample_interval_on_reset


def test_only_exact_elapsed_seconds_match_triggers_and_resample() -> None:
    torch.manual_seed(7)
    term, asset = _make_term(num_envs=3, interval_seconds=(2, 4))
    term._interval_seconds.copy_(torch.tensor([2, 3, 2], dtype=torch.int32))
    initial_velocity = asset.data.root_vel_w.torch.clone()

    _step(term)
    assert asset.writes == []
    assert term._elapsed_seconds.tolist() == [1, 1, 1]

    _step(term)

    pushed_velocity, pushed_ids = asset.writes.pop()
    assert pushed_ids.tolist() == [0, 2]
    assert term._elapsed_seconds.tolist() == [0, 2, 0]
    assert set(term._interval_seconds[pushed_ids].tolist()).issubset({2, 3})
    increments = pushed_velocity - initial_velocity[pushed_ids]
    assert torch.all(increments[:, :2].abs() <= 0.5)
    assert torch.equal(increments[:, 2], torch.zeros(2))
    assert torch.all(increments[:, 3:].abs() <= 0.5)


@pytest.mark.parametrize(
    ("elapsed_seconds", "expected_mask"),
    [
        ([0, 0, 0, 0], [False, False, False, False]),
        ([0, 1, 0, 1], [False, True, False, True]),
        ([1, 1, 1, 1], [True, True, True, True]),
    ],
)
def test_push_selection_uses_one_preallocated_mask(
    monkeypatch: pytest.MonkeyPatch,
    elapsed_seconds: list[int],
    expected_mask: list[bool],
) -> None:
    term, asset = _make_term(num_envs=4, interval_seconds=(2, 3))
    term._elapsed_seconds.copy_(torch.tensor(elapsed_seconds, dtype=torch.int32))
    term._interval_seconds.fill_(2)
    trigger_mask_ptr = term._trigger_mask.data_ptr()

    def _reject_dynamic_selection(*args: object, **kwargs: object) -> None:
        raise AssertionError("push selection must not allocate through torch.nonzero")

    monkeypatch.setattr(torch, "nonzero", _reject_dynamic_selection)
    _step(term)

    assert term._trigger_mask.data_ptr() == trigger_mask_ptr
    assert len(asset.mask_writes) == 1
    torch.testing.assert_close(asset.mask_writes[0][1], torch.tensor(expected_mask))


def test_episode_reset_preserves_push_schedule() -> None:
    term, _ = _make_term(num_envs=3)
    term._elapsed_seconds.copy_(torch.tensor([11, 22, 33], dtype=torch.int32))
    term._interval_seconds.copy_(torch.tensor([1, 2, 1], dtype=torch.int32))

    term.reset(env_ids=torch.tensor([0, 2]))

    assert term._elapsed_seconds.tolist() == [11, 22, 33]
    assert term._interval_seconds.tolist() == [1, 2, 1]


@pytest.mark.parametrize("interval_seconds", ((0, 2), (2, 2)))
def test_invalid_schedule_fails_at_construction(interval_seconds: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="positive and increasing"):
        _make_term(interval_seconds=interval_seconds)


@pytest.mark.parametrize(
    ("event_interval_seconds", "is_global_time"),
    [
        (0.02, True),
        (2.0, True),
        (1.0, False),
    ],
)
def test_push_requires_one_global_event_per_second(event_interval_seconds: float, is_global_time: bool) -> None:
    with pytest.raises(ValueError, match="one global event exactly once per second"):
        _make_term(event_interval_seconds=event_interval_seconds, is_global_time=is_global_time)


@pytest.mark.parametrize("physics_dt_seconds", (0.0, -0.01, float("nan"), float("inf")))
def test_smpl_fall_reset_requires_positive_finite_physics_timestep(physics_dt_seconds: float) -> None:
    """Fall synthesis must consume one valid timestep from the selected profile."""
    with pytest.raises(ValueError, match="finite positive physics timestep"):
        SmplMocapAndFallReset(
            object(),
            random_actions_high_exclusive=5,
            physics_dt_seconds=physics_dt_seconds,
            physics_steps_per_action=15,
        )


def test_smpl_native_reset_preserves_source_state_without_physx_ground_snap() -> None:
    """Native MuJoCo resets must not rewrite a valid source state for PhysX."""
    joint_position = torch.stack((torch.zeros(69), torch.linspace(-1.2, 1.2, 69)))
    root_position = torch.tensor(((0.0, 0.0, -0.1), (0.0, 0.0, 0.2)))
    root_rotation = torch.zeros(2, 4)
    root_rotation[:, 3] = 1.0
    reference = MotionStatePayload.ResetState(
        root_position=root_position,
        root_rotation_xyzw=root_rotation,
        root_linear_velocity_world=torch.zeros(2, 3),
        root_angular_velocity_world=torch.zeros(2, 3),
        joint_position=joint_position,
        joint_velocity=torch.zeros_like(joint_position),
    )
    reset = object.__new__(SmplMocapAndFallReset)
    reset._POOL_SIZE = 1
    reset._pool = MotionStatePayload.ResetState(
        root_position=reference.root_position[:1],
        root_rotation_xyzw=reference.root_rotation_xyzw[:1],
        root_linear_velocity_world=reference.root_linear_velocity_world[:1],
        root_angular_velocity_world=reference.root_angular_velocity_world[:1],
        joint_position=reference.joint_position[:1],
        joint_velocity=reference.joint_velocity[:1],
    )

    selected = reset(
        reference,
        torch.zeros(2, dtype=torch.int64),
        torch.Generator(device="cpu").manual_seed(0),
    )

    for field in reference.__dataclass_fields__:
        torch.testing.assert_close(getattr(selected, field), getattr(reference, field))
