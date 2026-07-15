# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for exact motion-imitation event schedules."""

import ast
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from isaaclab_tasks.core.multi_task.mdp import RootVelocityPushDiscrete
from isaaclab_tasks.core.multi_task.motion.data import MotionResetState
from isaaclab_tasks.core.multi_task.motion.robots.g1.reset import G1ReferenceAndLieDownReset
from isaaclab_tasks.core.multi_task.motion.robots.smpl.articulation import SMPL_MOTION_ARTICULATION_CFG
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reset import SmplHumEnvMocapAndFallReset
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils import resolve_presets

from isaaclab_assets.robots.smpl import smpl_constants

_SMPL_LIVE_JOINT_NAMES = tuple(
    f"{body}_x_{body}_y_{body}_z:{component}" for body in smpl_constants.MUJOCO_BODY_NAMES[1:] for component in range(3)
)


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
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> tuple[RootVelocityPushDiscrete, _Asset]:
    asset = _Asset(num_envs)
    if velocity_range is None:
        velocity_range = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.0, 0.0),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        }
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={"robot": asset},
    )
    cfg = EventTermCfg(
        func=RootVelocityPushDiscrete,
        mode="interval",
        interval_range_s=(event_interval_seconds, event_interval_seconds),
        is_global_time=is_global_time,
        resample_interval_on_reset=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "interval_seconds_range": interval_seconds,
            "velocity_range": velocity_range,
        },
    )
    return RootVelocityPushDiscrete(cfg, env), asset


def _step(term: RootVelocityPushDiscrete) -> None:
    params = term.cfg.params
    term(
        term._env,
        None,
        params["asset_cfg"],
        params["interval_seconds_range"],
        params["velocity_range"],
    )


def test_integer_second_intervals_match_bfm_schedule() -> None:
    term, _ = _make_term(num_envs=4096)

    assert term._interval_second_choices.tolist() == [1, 2]
    assert torch.all((term._interval_seconds == 1) | (term._interval_seconds == 2))


def test_g1_push_uses_global_one_second_event_cadence() -> None:
    push = resolve_presets(
        MotionImitationEnvCfg(),
        selected={
            "g1",
            "bfm_lafan",
            "physx",
            "timing_sim200_control50_horizon501",
            "sampling_clip_time",
            "randomization_physics_observation_pose_push",
        },
    ).events.push

    assert push.interval_range_s == (1.0, 1.0)
    assert push.is_global_time
    assert push.func is RootVelocityPushDiscrete
    assert push.params["interval_seconds_range"] == (1, 3)
    assert push.params["velocity_range"] == {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.0, 0.0),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (-0.5, 0.5),
    }


def test_motion_startup_randomization_reuses_isaaclab_events() -> None:
    events = resolve_presets(
        MotionImitationEnvCfg(),
        selected={
            "g1",
            "bfm_lafan",
            "physx",
            "timing_sim200_control50_horizon501",
            "sampling_clip_time",
            "randomization_physics_observation_pose_push",
        },
    ).events

    assert events.robot_material.func is isaaclab_mdp.randomize_rigid_body_material
    assert events.robot_material.params["num_buckets"] == 1024
    assert events.body_mass.func is isaaclab_mdp.randomize_rigid_body_mass
    assert events.torso_com.func is isaaclab_mdp.randomize_rigid_body_com


def test_smpl_native_asset_owns_exact_humenv_inertial_properties() -> None:
    """The selected MJCF must carry exact masses and inertias without a startup rewrite."""
    mujoco = pytest.importorskip("mujoco")
    robot = mujoco.MjModel.from_xml_path(smpl_constants.SMPL_ROBOT_MJCF_PATH)
    reference = mujoco.MjModel.from_xml_path(smpl_constants.SMPL_HUMENV_MJCF_PATH)
    robot_ids = np.asarray(
        [mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, name) for name in smpl_constants.MUJOCO_BODY_NAMES]
    )
    reference_ids = np.asarray(
        [mujoco.mj_name2id(reference, mujoco.mjtObj.mjOBJ_BODY, name) for name in smpl_constants.MUJOCO_BODY_NAMES]
    )

    assert SMPL_MOTION_ARTICULATION_CFG.spawn.asset_path == smpl_constants.SMPL_ROBOT_MJCF_PATH
    assert np.all(robot_ids >= 0) and np.all(reference_ids >= 0)
    np.testing.assert_array_equal(robot.body_mass[robot_ids], reference.body_mass[reference_ids])
    np.testing.assert_array_equal(robot.body_inertia[robot_ids], reference.body_inertia[reference_ids])
    np.testing.assert_array_equal(robot.body_iquat[robot_ids], reference.body_iquat[reference_ids])


def test_smpl_inertial_startup_rewrite_stays_absent() -> None:
    """SMPL physics is asset data, not duplicated constants or an environment event."""
    motion_root = Path(__file__).parents[1] / "motion"
    root_source = (motion_root.parent / "motion_env_cfg.py").read_text(encoding="utf-8")
    smpl = resolve_presets(
        MotionImitationEnvCfg(),
        selected={"smpl", "humenv_cmu", "newton_mjwarp", "timing_sim450_control30_horizon300", "sampling_source_rows"},
    )

    assert vars(smpl.events) == {}
    assert "set_smpl_body_mass_inertia" not in root_source
    assert not (motion_root / "mdp" / "events.py").exists()
    assert not hasattr(smpl_constants, "HUMENV_BODY_MASS")
    assert not hasattr(smpl_constants, "HUMENV_BODY_INERTIA")


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


@pytest.mark.parametrize(
    "velocity_range",
    ({"forward": (-0.5, 0.5)}, {"x": (0.5, -0.5)}, {"yaw": (0.0, float("inf"))}),
)
def test_invalid_root_velocity_range_fails_at_construction(
    velocity_range: dict[str, tuple[float, float]],
) -> None:
    with pytest.raises(ValueError, match="Root velocity range"):
        _make_term(velocity_range=velocity_range)


@pytest.mark.parametrize("physics_dt_seconds", (0.0, -0.01, float("nan"), float("inf")))
def test_smpl_fall_reset_requires_positive_finite_physics_timestep(physics_dt_seconds: float) -> None:
    """Fall synthesis must require one finite positive physics timestep."""
    with pytest.raises(ValueError, match="finite positive physics timestep"):
        SmplHumEnvMocapAndFallReset(
            seed=0,
            device="cpu",
            capacity=1,
            live_joint_names=tuple(smpl_constants.MUJOCO_JOINT_NAMES),
            physics_dt_seconds=physics_dt_seconds,
            physics_steps_per_action=15,
            random_actions_high_exclusive=5,
            fall_pool_size=1,
            initial_root_height_m=1.0,
            initial_root_quaternion_component_range=(0.0, 1.0),
            control_range=(-0.5, 0.5),
        )


@pytest.mark.parametrize("physics_steps_per_action", (0, -1))
def test_smpl_fall_reset_requires_positive_physics_steps_per_action(physics_steps_per_action: int) -> None:
    """Fall synthesis must require a positive number of physics steps per action."""
    with pytest.raises(ValueError, match="positive action and physics-step counts"):
        SmplHumEnvMocapAndFallReset(
            seed=0,
            device="cpu",
            capacity=1,
            live_joint_names=tuple(smpl_constants.MUJOCO_JOINT_NAMES),
            physics_dt_seconds=1.0 / 450.0,
            physics_steps_per_action=physics_steps_per_action,
            random_actions_high_exclusive=5,
            fall_pool_size=1,
            initial_root_height_m=1.0,
            initial_root_quaternion_component_range=(0.0, 1.0),
            control_range=(-0.5, 0.5),
        )


def test_smpl_fall_reset_requires_positive_pool_size() -> None:
    """Fall synthesis must require a positive reservoir capacity."""
    with pytest.raises(ValueError, match="positive pool size"):
        SmplHumEnvMocapAndFallReset(
            seed=0,
            device="cpu",
            capacity=1,
            live_joint_names=tuple(smpl_constants.MUJOCO_JOINT_NAMES),
            physics_dt_seconds=1.0 / 450.0,
            physics_steps_per_action=15,
            random_actions_high_exclusive=5,
            fall_pool_size=0,
            initial_root_height_m=1.0,
            initial_root_quaternion_component_range=(0.0, 1.0),
            control_range=(-0.5, 0.5),
        )


def test_smpl_native_reset_preserves_source_state_without_physx_ground_snap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native MuJoCo resets preserve references in fixed-capacity output storage."""
    joint_position = torch.stack((torch.zeros(69), torch.linspace(-1.2, 1.2, 69)))
    root_position = torch.tensor(((0.0, 0.0, -0.1), (0.0, 0.0, 0.2)))
    root_rotation = torch.zeros(2, 4)
    root_rotation[:, 3] = 1.0
    reference = MotionResetState(
        root_position=root_position,
        root_rotation_xyzw=root_rotation,
        root_linear_velocity_world=torch.zeros(2, 3),
        root_angular_velocity_world=torch.zeros(2, 3),
        joint_position=joint_position,
        joint_velocity=torch.zeros_like(joint_position),
    )
    pool = MotionResetState(
        root_position=reference.root_position[:1],
        root_rotation_xyzw=reference.root_rotation_xyzw[:1],
        root_linear_velocity_world=reference.root_linear_velocity_world[:1],
        root_angular_velocity_world=reference.root_angular_velocity_world[:1],
        joint_position=reference.joint_position[:1],
        joint_velocity=reference.joint_velocity[:1],
    )
    monkeypatch.setattr(SmplHumEnvMocapAndFallReset, "_build_pool", lambda _self, _device: pool)
    reset = SmplHumEnvMocapAndFallReset(
        seed=0,
        device="cpu",
        capacity=2,
        live_joint_names=_SMPL_LIVE_JOINT_NAMES,
        physics_dt_seconds=1.0 / 450.0,
        physics_steps_per_action=15,
        random_actions_high_exclusive=5,
        fall_pool_size=1,
        initial_root_height_m=1.0,
        initial_root_quaternion_component_range=(0.0, 1.0),
        control_range=(-0.5, 0.5),
    )
    generator = torch.Generator(device="cpu").manual_seed(0)
    assert reset._pool is pool
    selected = reset(reference, torch.zeros(2, dtype=torch.int64), generator)

    for field in reference.__dataclass_fields__:
        torch.testing.assert_close(getattr(selected, field), getattr(reference, field))
    pointers = {field: getattr(selected, field).data_ptr() for field in reference.__dataclass_fields__}

    repeated = reset(reference, torch.zeros(2, dtype=torch.int64), generator)
    partial_reference = MotionResetState(
        **{field: getattr(reference, field)[:1] for field in reference.__dataclass_fields__}
    )
    partial = reset(partial_reference, torch.zeros(1, dtype=torch.int64), generator)
    assert {field: getattr(repeated, field).data_ptr() for field in reference.__dataclass_fields__} == pointers
    assert {field: getattr(partial, field).data_ptr() for field in reference.__dataclass_fields__} == pointers


def test_g1_lie_down_reset_matches_shared_quaternion_math_and_reuses_storage() -> None:
    """G1 lie-down resets preserve BFM-Zero's random law in fixed storage."""
    root_rotation = torch.zeros(2, 4)
    root_rotation[:, 3] = 1.0
    reference = MotionResetState(
        root_position=torch.tensor(((0.0, 0.0, 1.0), (1.0, 2.0, 1.2))),
        root_rotation_xyzw=root_rotation,
        root_linear_velocity_world=torch.zeros(2, 3),
        root_angular_velocity_world=torch.zeros(2, 3),
        joint_position=torch.zeros(2, 29),
        joint_velocity=torch.zeros(2, 29),
    )
    reset = G1ReferenceAndLieDownReset(
        capacity=2,
        device="cpu",
        lie_down_root_height_m=0.5,
        lie_down_roll_magnitude_rad=0.5 * torch.pi,
        lie_down_negative_roll_probability=0.5,
    )
    generator = torch.Generator(device="cpu").manual_seed(7)
    selected = reset(reference, torch.tensor((0, 1), dtype=torch.int64), generator)

    oracle_generator = torch.Generator(device="cpu").manual_seed(7)
    sign = 1.0 if torch.rand((), generator=oracle_generator) < 0.5 else -1.0
    angle = torch.tensor((sign * (-0.5 * torch.pi),))
    axis = torch.tensor(((1.0, 0.0, 0.0),))
    expected_rotation = quat_mul(quat_from_angle_axis(angle, axis), reference.root_rotation_xyzw[1:2])
    torch.testing.assert_close(selected.root_position[0], reference.root_position[0])
    torch.testing.assert_close(selected.root_position[1, 2], torch.tensor(0.5))
    torch.testing.assert_close(selected.root_rotation_xyzw[0], reference.root_rotation_xyzw[0])
    torch.testing.assert_close(selected.root_rotation_xyzw[1:2], expected_rotation)
    pointers = {field: getattr(selected, field).data_ptr() for field in reference.__dataclass_fields__}

    repeated = reset(reference, torch.tensor((1, 0), dtype=torch.int64), generator)
    partial_reference = MotionResetState(
        **{field: getattr(reference, field)[:1] for field in reference.__dataclass_fields__}
    )
    partial = reset(partial_reference, torch.zeros(1, dtype=torch.int64), generator)
    assert {field: getattr(repeated, field).data_ptr() for field in reference.__dataclass_fields__} == pointers
    assert {field: getattr(partial, field).data_ptr() for field in reference.__dataclass_fields__} == pointers


def test_motion_reset_hot_paths_write_torch_ops_through_out_buffers() -> None:
    """Every tensor-producing reset operation writes caller-stable storage."""
    out_operations = {"eq", "index_select", "lt", "mul", "rand", "randint", "where"}
    for function in (
        G1ReferenceAndLieDownReset.__call__,
        SmplHumEnvMocapAndFallReset.__call__,
        SmplHumEnvMocapAndFallReset._select,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
                if node.func.attr in out_operations:
                    assert any(keyword.arg == "out" for keyword in node.keywords), (function, node.func.attr)
                assert node.func.attr not in {"cat", "empty", "stack", "tensor", "zeros"}
            assert node.func.attr not in {"clone", "contiguous", "new_tensor"}
