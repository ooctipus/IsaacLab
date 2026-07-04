# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for configuration-owned success-estimator input bindings."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.core.multi_task.rl.rsl_rl.algorithms import SuccessEstimatorPPO
from isaaclab_tasks.core.multi_task.rl.rsl_rl.rl_cfg import RslRlSuccessEstimatorAlgorithmCfg


def _algorithm(outcome_bind: str, mask_bind: str | None = None) -> SuccessEstimatorPPO:
    algorithm = object.__new__(SuccessEstimatorPPO)
    algorithm._success_outcome_bind = outcome_bind
    algorithm._success_train_mask_bind = mask_bind
    algorithm._success_outcome = None
    algorithm._success_train_mask = None
    algorithm._success_mask_enabled = mask_bind is not None
    algorithm.success_predictions = torch.zeros(3)
    return algorithm


def test_success_estimator_binds_completed_outcome_without_environment_extras() -> None:
    """The composition expression should bind the manager-owned transition result by reference."""
    success = torch.tensor((True, False, True))
    manager = SimpleNamespace(get_term=lambda name: success if name == "success" else None)
    env = SimpleNamespace(unwrapped=SimpleNamespace(termination_manager=manager))
    algorithm = _algorithm("env.unwrapped.termination_manager.get_term('success')")

    algorithm.bind_success_inputs(env)

    assert algorithm._success_outcome is success
    assert algorithm._success_outcome.data_ptr() == success.data_ptr()
    assert algorithm._success_train_mask is None


@pytest.mark.parametrize(
    ("value", "error", "message"),
    (
        ([True, False, True], TypeError, "torch.Tensor"),
        (torch.zeros(3, dtype=torch.int64), TypeError, "bool or floating-point"),
        (torch.zeros(2), ValueError, "shape"),
    ),
)
def test_success_estimator_rejects_invalid_bound_outcome(value, error: type[Exception], message: str) -> None:
    """Invalid composition bindings should fail once at construction, not in the rollout hot path."""
    env = SimpleNamespace(value=value)
    algorithm = _algorithm("env.value")

    with pytest.raises(error, match=message):
        algorithm.bind_success_inputs(env)


def test_success_estimator_requires_explicit_outcome_binding() -> None:
    """The learner must not retain an environment-extras compatibility path."""
    algorithm = _algorithm("")

    with pytest.raises(ValueError, match="requires a nonempty success_outcome_bind"):
        algorithm.bind_success_inputs(SimpleNamespace())

    source = inspect.getsource(SuccessEstimatorPPO)
    assert "_legacy_success_inputs" not in source
    assert 'extras["successes"]' not in source


def test_success_estimator_typed_config_serializes_explicit_bindings() -> None:
    """Typed runner configuration must carry composition expressions into construction."""
    cfg = RslRlSuccessEstimatorAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.0,
        desired_kl=0.01,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        success_outcome_bind="env.unwrapped.termination_manager.get_term('success')",
        success_train_mask_bind="env.unwrapped.curriculum_manager.get_term('gate').train_mask",
    )

    values = cfg.to_dict()
    assert values["success_outcome_bind"] == "env.unwrapped.termination_manager.get_term('success')"
    assert values["success_train_mask_bind"] == "env.unwrapped.curriculum_manager.get_term('gate').train_mask"


def test_success_estimator_mask_path_has_no_device_dependent_branch() -> None:
    """The minibatch hot path must not synchronize CUDA to inspect mask contents."""
    assert "mask.all()" not in inspect.getsource(SuccessEstimatorPPO.update)
