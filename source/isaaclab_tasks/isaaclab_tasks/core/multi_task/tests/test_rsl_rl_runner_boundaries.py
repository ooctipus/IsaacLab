# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for distinct generic off-policy, CRL, and deprecated runner ownership."""

import pytest

from isaaclab_rl.rsl_rl import RslRlOffPolicyRunnerCfg

import isaaclab_tasks.core.multi_task.rl.rsl_rl as task_rsl_rl
import isaaclab_tasks.core.multi_task.rl.rsl_rl.runners as task_runners
from isaaclab_tasks.core.locomotion.ant.agents.rsl_rl_crl_cfg import AntCRLRunnerCfg
from isaaclab_tasks.core.multi_task.rl.rsl_rl import RslRlCrlRunnerCfg
from isaaclab_tasks.core.multi_task.rl.rsl_rl.runners.crl_runner import CrlRunner
from isaaclab_tasks.core.multi_task.rl.rsl_rl.runners.off_policy_runner import OffPolicyRunner
from isaaclab_tasks.core.multi_task.terrain.config.rsl_rl_cfg import PositionLocomotionCRLRunnerCfg


def test_generic_off_policy_and_crl_runners_have_distinct_lifecycles() -> None:
    """Known CRL callers use the CRL runner while the public config resolves official RSL-RL."""
    assert (
        RslRlOffPolicyRunnerCfg(
            num_steps_per_env=1,
            num_updates_per_iteration=1,
            max_iterations=1,
            obs_groups={"actor": ["policy"]},
            save_interval=1,
            experiment_name="fixture",
            algorithm={"class_name": "fixture:Algorithm"},
        ).class_type
        == "rsl_rl.runners:OffPolicyRunner"
    )
    assert AntCRLRunnerCfg().class_name == "CrlRunner"
    assert PositionLocomotionCRLRunnerCfg().class_name == "CrlRunner"
    assert issubclass(AntCRLRunnerCfg, RslRlCrlRunnerCfg)
    assert issubclass(PositionLocomotionCRLRunnerCfg, RslRlCrlRunnerCfg)


def test_deprecated_task_runner_config_warns_and_delegates_to_crl() -> None:
    """The old task config path must warn while retaining its CRL behavior for one release."""
    with pytest.warns(DeprecationWarning, match="RslRlOffPolicyRunnerCfg is deprecated"):
        cfg = task_rsl_rl.RslRlOffPolicyRunnerCfg()

    assert isinstance(cfg, RslRlCrlRunnerCfg)
    assert cfg.class_name == "OffPolicyRunner"
    assert cfg.class_type == "isaaclab_tasks.core.multi_task.rl.rsl_rl.runners:OffPolicyRunner"


def test_deprecated_task_runner_warns_and_delegates_to_crl(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old runner class must be a warning boundary around the clean CRL implementation."""
    calls: list[tuple[object, dict, str | None, str]] = []

    def initialize(
        _runner: CrlRunner,
        env: object,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        calls.append((env, train_cfg, log_dir, device))

    monkeypatch.setattr(CrlRunner, "__init__", initialize)
    env = object()
    train_cfg = {"algorithm": "fixture"}
    with pytest.warns(DeprecationWarning, match="OffPolicyRunner is deprecated"):
        runner = OffPolicyRunner(env, train_cfg, log_dir="logs", device="cuda:0")

    assert isinstance(runner, CrlRunner)
    assert task_runners.OffPolicyRunner is OffPolicyRunner
    assert calls == [(env, train_cfg, "logs", "cuda:0")]
