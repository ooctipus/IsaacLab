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
import warp as wp

from isaaclab.utils.math import quat_from_euler_xyz

# ``wp.from_torch`` requires Warp's runtime to be initialized. Previously it
# was initialized as a side effect of the (now TYPE_CHECKING-only) import
# chain ``kernels → isaaclab.envs → isaaclab.sim → warp``. The import-
# violation fix removed that chain to keep cfg construction Kit-free; this
# test file — which calls ``wp.from_torch`` directly in
# ``_make_articulation_with_body_pos`` — now needs an explicit init.
wp.init()

from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.impl.kernels_torch import (
    ACTIVATION_KERNELS,
    BUFFER_KIND_READERS,
    SAMPLER_KERNELS,
    STATE_KERNEL_BUFFER_KIND,
    STATE_KERNEL_COMPUTES,
    _state_identity,
    delta_geometric,
    delta_quaternion,
    metric_geometric,
    metric_quaternion,
    state_kernel_intra_body_offset,
    state_kernel_intra_body_stride,
)
from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.kernel_ids import (
    ACTIVATION_KERNEL_ID,
    BUFFER_KIND,
    SAMPLER_KERNEL_ID,
    STATE_KERNEL_ID,
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
    """``activation_less`` returns ``error < threshold`` as a bool/float."""
    kernel = ACTIVATION_KERNELS[int(ACTIVATION_KERNEL_ID.LESS)]
    threshold = torch.tensor(0.5)
    errors = torch.tensor([0.0, 0.3, 0.49, 0.5, 0.51, 1.0])
    scores = kernel(errors, threshold).float()
    assert torch.allclose(scores, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))


def test_greater_activation_returns_predicate():
    """``activation_greater`` returns ``error > threshold``."""
    kernel = ACTIVATION_KERNELS[int(ACTIVATION_KERNEL_ID.GREATER)]
    threshold = torch.tensor(0.5)
    errors = torch.tensor([0.0, 0.3, 0.5, 0.51, 1.0])
    scores = kernel(errors, threshold).float()
    assert torch.allclose(scores, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))


def test_gaussian_activation_shape_and_exact_values():
    """``exp(-(error/σ)²)`` peaks at 1.0 (zero slope) at error=0, decays to 1/e at error=σ.

    Locks in the soft-safety budget shape — distinguishes Gaussian from TANH
    (which is steepest at error=0). The plateau near 0 is the load-bearing
    behavior: it lets nominal violation amounts contribute essentially full
    activation without imposing gradient pressure to drive them to exactly 0.
    """
    kernel = ACTIVATION_KERNELS[int(ACTIVATION_KERNEL_ID.GAUSSIAN)]
    sigma = torch.tensor(2.0)
    errors = torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0, 6.0])
    scores = kernel(errors, sigma)
    # err/σ = [0, 0.25, 0.5, 1, 2, 3]; exp(-(err/σ)²) = [1, exp(-1/16), exp(-1/4), 1/e, exp(-4), exp(-9)]
    expected = torch.tensor(
        [
            1.0,
            math.exp(-1.0 / 16.0),
            math.exp(-0.25),
            math.exp(-1.0),
            math.exp(-4.0),
            math.exp(-9.0),
        ]
    )
    assert torch.allclose(scores, expected, atol=1e-6), f"gaussian kernel drifted: {scores} vs {expected}"
    # Plateau check — slope at error=0 is 0; small ε gives change ~ε² ≪ ε.
    eps = 1e-3
    near_zero = kernel(torch.tensor([0.0, eps]), sigma)
    assert (1.0 - near_zero[1].item()) < eps, "Gaussian should be flat at error=0 (zero slope)"


# -----------------------------------------------------------------------------
# Metric kernels (delta tensor → scalar error)
# -----------------------------------------------------------------------------


def test_metric_geometric_is_l2_norm():
    """``metric_geometric`` is the Euclidean norm along the last dim."""
    delta = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 2.0]])
    err = metric_geometric(delta)
    assert torch.allclose(err, torch.tensor([5.0, 0.0, math.sqrt(5.0)]))


def test_metric_geometric_singleton_last_dim():
    """Single-component vectors → scalar absolute value."""
    delta = torch.tensor([[2.5], [-1.0], [0.0]])
    err = metric_geometric(delta)
    assert torch.allclose(err, torch.tensor([2.5, 1.0, 0.0]))


def test_metric_quaternion_zero_for_identity():
    """Identity quaternion ``[0, 0, 0, 1]`` → zero rotation magnitude."""
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    err = metric_quaternion(identity)
    assert torch.allclose(err, torch.zeros(1), atol=1e-6)


def test_metric_quaternion_pi_rotation():
    """180° rotation about z → error magnitude ≈ π."""
    # Quaternion for 180° about z: xyzw = [0, 0, 1, 0].
    quat = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    err = metric_quaternion(quat)
    assert abs(err.item() - math.pi) < 1e-5


# -----------------------------------------------------------------------------
# Delta kernels (current, target) → raw delta tensor
# -----------------------------------------------------------------------------


def test_delta_geometric_is_target_minus_current():
    """Vector delta is ``target - current`` (NOT the other way round)."""
    x_cur = torch.tensor([[1.0, 2.0, 3.0]])
    x_tgt = torch.tensor([[4.0, 6.0, 8.0]])
    delta = delta_geometric(x_cur, x_tgt)
    assert torch.allclose(delta, torch.tensor([[3.0, 4.0, 5.0]]))


def test_delta_quaternion_identity():
    """Same quaternion in both slots → delta quaternion norm ≈ identity (zero axis-angle)."""
    q = torch.tensor([[0.1, 0.2, 0.3, 0.9273618]])  # not normalized but close
    q = q / q.norm(dim=-1, keepdim=True)
    delta_q = delta_quaternion(q, q)
    err = metric_quaternion(delta_q)
    assert err.item() < 1e-5


def test_delta_quaternion_90_deg_roundtrip():
    """Target = 90° rotation from current → delta encodes 90°."""
    # Current is identity; target is 90° about x.
    q_cur = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    q_tgt = quat_from_euler_xyz(torch.tensor([math.pi / 2]), torch.tensor([0.0]), torch.tensor([0.0]))
    delta_q = delta_quaternion(q_cur, q_tgt)
    err = metric_quaternion(delta_q)
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


def _proxy_vec3(t: torch.Tensor):
    """Wrap a ``[..., 3]`` contiguous torch tensor as a ProxyArray of ``wp.vec3``."""
    import warp as wp

    from isaaclab.utils.warp import ProxyArray

    return ProxyArray(
        wp.array(
            ptr=t.data_ptr(),
            dtype=wp.vec3,
            shape=tuple(t.shape[:-1]),
            device=str(t.device),
        )
    )


def _make_articulation_with_body_pos(body_pos: torch.Tensor):
    """Wrap a ``[N, num_bodies, 3]`` torch tensor as a warp-backed articulation stub."""
    data = type("_D", (), {})()
    data.body_pos_w = _proxy_vec3(body_pos)
    return type("_A", (), {"data": data})()


def _make_articulation_with_body_lin_vel(lin_vel: torch.Tensor):
    data = type("_D", (), {})()
    data.body_lin_vel_w = _proxy_vec3(lin_vel)
    return type("_A", (), {"data": data})()


def _make_asset_cfg_for_single_body(body_index: int):
    from isaaclab.managers import SceneEntityCfg

    asset_cfg = SceneEntityCfg("robot")
    asset_cfg.body_ids = [body_index]
    return asset_cfg


def test_body_position_w_reader_returns_world_frame_raw():
    """The ``BODY_POS_W`` reader returns the world-frame tensor as a zero-copy view.

    The env-origin subtraction that converts world-frame → env-local is applied
    downstream by the dispatch (Warp kernel ``fill_slab_vec3_env_local`` for
    the Warp path, inline ``raw - env_origins`` for the Torch path). This
    keeps every reader a pure zero-copy view over scene storage — essential
    for pointer stability under any future CUDA Graph capture and for the
    "readers don't transform" contract.
    """
    body_pos = torch.tensor([[[1.0, 2.0, 0.35]]])  # [N=1, bodies=1, xyz]
    env_origins = torch.tensor([[0.0, 0.0, 0.1]])  # non-zero z — NOT subtracted by reader
    env = _make_stub_env(_make_articulation_with_body_pos(body_pos), env_origins)

    reader = BUFFER_KIND_READERS[int(BUFFER_KIND.BODY_POS_W)]
    raw = reader(env, "robot")
    assert raw.shape == (1, 1, 3)
    # World-frame, unmodified.
    assert abs(raw[0, 0, 0].item() - 1.0) < 1e-6
    assert abs(raw[0, 0, 1].item() - 2.0) < 1e-6
    assert abs(raw[0, 0, 2].item() - 0.35) < 1e-6
    # Sanity: reader does not allocate — same storage as the source.
    assert raw.data_ptr() == body_pos.data_ptr()


def test_state_kernels_tuple_aligns_with_enum():
    """``BODY_POS_Z`` routes to the right buffer + intra-body slice + compute triple."""
    kid = int(STATE_KERNEL_ID.BODY_POS_Z)
    assert STATE_KERNEL_BUFFER_KIND[kid] == BUFFER_KIND.BODY_POS_W
    # POS_Z reads one float per body (z only) at intra-body offset 2.
    assert state_kernel_intra_body_offset(kid) == 2
    assert state_kernel_intra_body_stride(kid) == 1
    # The gather indices already project to z-only at spec build, so the compute
    # function is just identity — no per-step reshape needed.
    assert STATE_KERNEL_COMPUTES[kid] is _state_identity


# Canonical-layout tests live in ``test_multi_task_command_mock.py`` — the layout
# is cfg-driven (computed in ``spec._compute_canonical_layout``) and must be
# exercised through a real spec build, not a kernel-local table.


def test_euler_to_quat_zero_range_yields_identity_when_center_zero():
    """All-zero Euler input → identity quaternion ``[0, 0, 0, 1]``."""
    kernel = SAMPLER_KERNELS[int(SAMPLER_KERNEL_ID.EULER_UNIFORM_TO_QUAT)]
    params = torch.zeros(1, 8)
    samples = kernel(params)
    assert samples.shape == (1, 4)
    # xyzw identity has w = 1, xyz = 0.
    assert abs(samples[0, 3].item() - 1.0) < 1e-6
    assert samples[0, :3].abs().max().item() < 1e-6
