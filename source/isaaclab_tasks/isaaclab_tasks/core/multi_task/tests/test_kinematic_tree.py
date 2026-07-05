# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared kinematic-tree rotation operators."""

import pytest
import torch

from isaaclab_tasks.core.multi_task.kinematics import fit_ordered_hinge_coordinates, ordered_hinge_rotation


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise training-time TF32 matmul.")
def test_three_hinge_fit_remains_finite_with_training_tf32() -> None:
    """A valid source rotation must remain finite with the training entry point's TF32 setting."""
    rotation = torch.tensor(
        [[-0.22445420920848846, -0.004686277359724045, 0.02124209515750408, 0.9742419719696045]],
        dtype=torch.float32,
        device="cuda",
    )
    axes = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
    )
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        coordinates, residual = fit_ordered_hinge_coordinates(rotation, axes)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous

    assert torch.all(torch.isfinite(coordinates))
    assert torch.all(torch.isfinite(residual))
    reconstructed = ordered_hinge_rotation(coordinates, axes)
    expected = torch.nn.functional.normalize(rotation, dim=-1)
    torch.testing.assert_close(reconstructed, expected, atol=2.0e-6, rtol=2.0e-6)
