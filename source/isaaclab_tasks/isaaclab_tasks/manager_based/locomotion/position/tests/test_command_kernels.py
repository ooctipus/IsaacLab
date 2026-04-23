# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the per-subtask kernels used by :class:`MultiTaskCommand`.

Kernels are small pure functions; each test feeds synthetic tensors and checks the
output against a closed-form expectation. Keeps the dispatch surface honest without
needing a live env.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_tasks.manager_based.locomotion.position.mdp.commands.kernels import (
    ACTIVATION_KERNEL_ID,
    ACTIVATION_KERNELS,
    DELTA_KERNELS,
    METRIC_KERNEL_ID,
    METRIC_KERNELS,
    SAMPLER_KERNEL_ID,
    SAMPLER_KERNELS,
    STATE_KERNEL_ID,
    STATE_KERNELS,
    body_lin_speed,
    body_position_z,
    geometric_error,
    geometric_subtract,
    quaternion_error,
    quaternion_subtract,
)

# -----------------------------------------------------------------------------
# Activation kernels (error → score in [0, 1])
# -----------------------------------------------------------------------------


def test_tanh_activation_monotonic_in_error():
    """``1 - tanh(error/std)`` decreases monotonically with error."""
    kernel = ACTIVATION_KERNELS[int(ACTIVATION_KERNEL_ID.TANH)]
    std = torch.tensor(0.2)
    errors = torch.tensor([0.0, 0.1, 0.2, 0.5, 1.0])
    scores = kernel(errors, std)
    assert scores[0].item() == 1.0  # error=0 → perfect score
    for i in range(1, len(scores)):
        assert scores[i] < scores[i - 1]
    assert scores.min() > 0.0  # tanh never saturates to 1


def test_tanh_activation_exact_values():
    """Pin ``1 - tanh(error/std)`` at specific inputs — catches scale/formula regressions.

    Monotonicity alone doesn't catch mutations like ``1 - tanh(err/std)/2`` or
    ``1 - tanh(err*std)``. These exact-value checks do.
    """
    kernel = ACTIVATION_KERNELS[int(ACTIVATION_KERNEL_ID.TANH)]
    std = torch.tensor(0.5)
    errors = torch.tensor([0.0, 0.5, 1.0, 2.0])
    scores = kernel(errors, std)
    # For std=0.5: err/std = [0, 1, 2, 4]; 1-tanh = [1, 1-tanh(1), 1-tanh(2), 1-tanh(4)]
    expected = torch.tensor([1.0, 1.0 - math.tanh(1.0), 1.0 - math.tanh(2.0), 1.0 - math.tanh(4.0)])
    assert torch.allclose(scores, expected, atol=1e-6), f"tanh kernel drifted: {scores} vs {expected}"


def test_less_activation_returns_predicate():
    """``less_kernel`` returns ``error < threshold`` as a bool/float."""
    kernel = ACTIVATION_KERNELS[int(ACTIVATION_KERNEL_ID.LESS)]
    threshold = torch.tensor(0.5)
    errors = torch.tensor([0.0, 0.3, 0.49, 0.5, 0.51, 1.0])
    scores = kernel(errors, threshold).float()
    assert torch.allclose(scores, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))


def test_greater_activation_returns_predicate():
    """``greater_kernel`` returns ``error > threshold``."""
    kernel = ACTIVATION_KERNELS[int(ACTIVATION_KERNEL_ID.GREATER)]
    threshold = torch.tensor(0.5)
    errors = torch.tensor([0.0, 0.3, 0.5, 0.51, 1.0])
    scores = kernel(errors, threshold).float()
    assert torch.allclose(scores, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))


# -----------------------------------------------------------------------------
# Metric kernels (delta tensor → scalar error)
# -----------------------------------------------------------------------------


def test_geometric_error_is_l2_norm():
    """``geometric_error`` is the Euclidean norm along the last dim."""
    delta = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 2.0]])
    err = geometric_error(delta)
    assert torch.allclose(err, torch.tensor([5.0, 0.0, math.sqrt(5.0)]))


def test_geometric_error_singleton_last_dim():
    """Single-component vectors → scalar absolute value."""
    delta = torch.tensor([[2.5], [-1.0], [0.0]])
    err = geometric_error(delta)
    assert torch.allclose(err, torch.tensor([2.5, 1.0, 0.0]))


def test_quaternion_error_zero_for_identity():
    """Identity quaternion ``[0, 0, 0, 1]`` → zero rotation magnitude."""
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    err = quaternion_error(identity)
    assert torch.allclose(err, torch.zeros(1), atol=1e-6)


def test_quaternion_error_pi_rotation():
    """180° rotation about z → error magnitude ≈ π."""
    # Quaternion for 180° about z: xyzw = [0, 0, 1, 0].
    quat = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    err = quaternion_error(quat)
    assert abs(err.item() - math.pi) < 1e-5


# -----------------------------------------------------------------------------
# Delta kernels (current, target) → raw delta tensor
# -----------------------------------------------------------------------------


def test_geometric_subtract_is_target_minus_current():
    """Vector delta is ``target - current`` (NOT the other way round)."""
    x_cur = torch.tensor([[1.0, 2.0, 3.0]])
    x_tgt = torch.tensor([[4.0, 6.0, 8.0]])
    delta = geometric_subtract(x_cur, x_tgt)
    assert torch.allclose(delta, torch.tensor([[3.0, 4.0, 5.0]]))


def test_quaternion_subtract_identity():
    """Same quaternion in both slots → delta quaternion norm ≈ identity (zero axis-angle)."""
    q = torch.tensor([[0.1, 0.2, 0.3, 0.9273618]])  # not normalized but close
    q = q / q.norm(dim=-1, keepdim=True)
    delta_q = quaternion_subtract(q, q)
    err = quaternion_error(delta_q)
    assert err.item() < 1e-5


def test_quaternion_subtract_90_deg_roundtrip():
    """Target = 90° rotation from current → delta encodes 90°."""
    # Current is identity; target is 90° about x.
    q_cur = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    q_tgt = quat_from_euler_xyz(torch.tensor([math.pi / 2]), torch.tensor([0.0]), torch.tensor([0.0]))
    delta_q = quaternion_subtract(q_cur, q_tgt)
    err = quaternion_error(delta_q)
    assert abs(err.item() - math.pi / 2) < 1e-5


# -----------------------------------------------------------------------------
# Sampler kernels (interleaved [min, range] params → uniform sample)
# -----------------------------------------------------------------------------


def test_uniform_sampler_respects_range():
    """Samples fall within ``[min, min + range]`` for each dim."""
    kernel = SAMPLER_KERNELS[int(SAMPLER_KERNEL_ID.UNIFORM)]
    # Interleaved: [min0, range0, min1, range1] → dim 0 in [-1, 2], dim 1 in [5, 8]
    params = torch.tensor([-1.0, 3.0, 5.0, 3.0]).unsqueeze(0).expand(200, -1)
    torch.manual_seed(0)
    samples = kernel(params)
    # Shape: params [*, 2D] -> samples [*, D]
    assert samples.shape == (200, 2)
    assert samples[:, 0].min() >= -1.0 and samples[:, 0].max() <= 2.0
    assert samples[:, 1].min() >= 5.0 and samples[:, 1].max() <= 8.0


def test_uniform_sampler_zero_range_is_constant():
    """Zero range → every sample equals the min value."""
    kernel = SAMPLER_KERNELS[int(SAMPLER_KERNEL_ID.UNIFORM)]
    params = torch.tensor([2.5, 0.0, -3.0, 0.0]).unsqueeze(0).expand(50, -1)
    samples = kernel(params)
    assert torch.allclose(samples[:, 0], torch.full((50,), 2.5))
    assert torch.allclose(samples[:, 1], torch.full((50,), -3.0))


def test_euler_to_quat_sampler_outputs_unit_quats():
    """``EULER_UNIFORM_TO_QUAT`` emits 4-vec quaternions with norm ≈ 1."""
    kernel = SAMPLER_KERNELS[int(SAMPLER_KERNEL_ID.EULER_UNIFORM_TO_QUAT)]
    # Interleaved (min, range) for 3 Euler dims + 1 padding pair (out_dim=4).
    # Roll in [-1, 1], Pitch in [-0.5, 0.5], Yaw in [-3.14, 3.14].
    params = (
        torch.tensor(
            [-1.0, 2.0, -0.5, 1.0, -3.14, 6.28, 0.0, 0.0],
            dtype=torch.float32,
        )
        .unsqueeze(0)
        .expand(100, -1)
    )
    torch.manual_seed(0)
    samples = kernel(params)
    assert samples.shape == (100, 4)
    norms = samples.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(100), atol=1e-5), (
        f"quaternion norms not ≈1 (max deviation {(norms - 1.0).abs().max().item():.3g})"
    )


class _StubScene:
    """Minimal subscriptable scene whose ``scene[name]`` returns a fixed articulation.

    Also carries ``env_origins`` so kernels that need env-local translations work.
    """

    def __init__(self, articulation, env_origins: torch.Tensor):
        self._articulation = articulation
        self.env_origins = env_origins

    def __getitem__(self, _key):
        return self._articulation


def _make_stub_env(articulation, env_origins: torch.Tensor):
    env = type("_E", (), {})()
    env.scene = _StubScene(articulation, env_origins)
    return env


def _make_articulation_with_body_pos(body_pos: torch.Tensor):
    """Wrap a ``[N, num_bodies, 3]`` torch tensor as a warp-backed articulation stub."""
    import warp as wp

    data = type("_D", (), {})()
    data.body_pos_w = wp.from_torch(body_pos, dtype=wp.vec3)
    return type("_A", (), {"data": data})()


def _make_articulation_with_body_lin_vel(lin_vel: torch.Tensor):
    import warp as wp

    data = type("_D", (), {})()
    data.body_lin_vel_w = wp.from_torch(lin_vel, dtype=wp.vec3)
    return type("_A", (), {"data": data})()


def _make_asset_cfg_for_single_body(body_index: int):
    from isaaclab.managers import SceneEntityCfg

    asset_cfg = SceneEntityCfg("robot")
    asset_cfg.body_ids = [body_index]
    return asset_cfg


def test_body_position_z_returns_only_z_component():
    """``BODY_POS_Z`` emits a stride-1 tensor carrying the env-local z coordinate.

    Verifies env-origin subtraction and that x/y components don't leak into the output.
    """
    body_pos = torch.tensor([[[1.0, 2.0, 0.35]]])  # [N=1, bodies=1, xyz]
    env_origins = torch.tensor([[0.0, 0.0, 0.1]])  # non-zero z — must be subtracted
    env = _make_stub_env(_make_articulation_with_body_pos(body_pos), env_origins)
    asset_cfg = _make_asset_cfg_for_single_body(0)

    out = body_position_z(env, slice(None), asset_cfg)
    # Last dim must be stride 1.
    assert out.shape[-1] == 1
    # Value must be z - env_origin_z, with xy ignored entirely.
    assert abs(out.flatten()[0].item() - (0.35 - 0.1)) < 1e-6


def test_body_lin_speed_returns_velocity_magnitude_direction_invariant():
    """``BODY_LIN_SPEED`` emits stride-1 ``||v||`` and is direction-invariant."""
    asset_cfg = _make_asset_cfg_for_single_body(0)
    env_origins = torch.zeros(1, 3)  # unused by body_lin_speed but the scene carries it

    # (3, 4, 0) → speed 5.
    lin_vel = torch.tensor([[[3.0, 4.0, 0.0]]])
    env = _make_stub_env(_make_articulation_with_body_lin_vel(lin_vel), env_origins)
    out = body_lin_speed(env, slice(None), asset_cfg)
    assert out.shape[-1] == 1
    assert abs(out.flatten()[0].item() - 5.0) < 1e-6

    # Direction-invariance: (-3, -4, 0) also yields 5.
    lin_vel = torch.tensor([[[-3.0, -4.0, 0.0]]])
    env = _make_stub_env(_make_articulation_with_body_lin_vel(lin_vel), env_origins)
    out = body_lin_speed(env, slice(None), asset_cfg)
    assert abs(out.flatten()[0].item() - 5.0) < 1e-6

    # Zero velocity → zero speed.
    lin_vel = torch.zeros(1, 1, 3)
    env = _make_stub_env(_make_articulation_with_body_lin_vel(lin_vel), env_origins)
    out = body_lin_speed(env, slice(None), asset_cfg)
    assert out.flatten()[0].item() == 0.0


def test_state_kernels_tuple_aligns_with_enum():
    """New state-kernel enum entries index the right functions in ``STATE_KERNELS``."""
    assert STATE_KERNELS[int(STATE_KERNEL_ID.BODY_POS_Z)] is body_position_z
    assert STATE_KERNELS[int(STATE_KERNEL_ID.BODY_LIN_SPEED)] is body_lin_speed


def test_euler_to_quat_zero_range_yields_identity_when_center_zero():
    """All-zero Euler input → identity quaternion ``[0, 0, 0, 1]``."""
    kernel = SAMPLER_KERNELS[int(SAMPLER_KERNEL_ID.EULER_UNIFORM_TO_QUAT)]
    params = torch.zeros(1, 8)
    samples = kernel(params)
    assert samples.shape == (1, 4)
    # xyzw identity has w = 1, xyz = 0.
    assert abs(samples[0, 3].item() - 1.0) < 1e-6
    assert samples[0, :3].abs().max().item() < 1e-6


# -----------------------------------------------------------------------------
# Kernel-id registry sanity (makes sure enums align with tuple indices)
# -----------------------------------------------------------------------------


def test_activation_kernel_ids_align():
    """Each ACTIVATION_KERNEL_ID indexes the matching kernel in ACTIVATION_KERNELS."""
    assert len(ACTIVATION_KERNELS) == len(ACTIVATION_KERNEL_ID)
    for member in ACTIVATION_KERNEL_ID:
        assert ACTIVATION_KERNELS[int(member)] is not None


def test_metric_kernel_ids_align():
    """Each METRIC_KERNEL_ID indexes the matching kernel in METRIC_KERNELS."""
    assert len(METRIC_KERNELS) == len(METRIC_KERNEL_ID)
    for member in METRIC_KERNEL_ID:
        assert METRIC_KERNELS[int(member)] is not None


def test_metric_delta_kernels_align():
    """DELTA_KERNELS and METRIC_KERNELS must be same length (caller assumes ID parity)."""
    assert len(DELTA_KERNELS) == len(METRIC_KERNELS)
