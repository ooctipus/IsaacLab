# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the generic RSL-RL off-policy runner configuration."""

from isaaclab_rl.rsl_rl import RslRlOffPolicyRunnerCfg
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg


def _off_policy_cfg() -> RslRlOffPolicyRunnerCfg:
    """Return one complete minimal mapping-based runner configuration."""
    return RslRlOffPolicyRunnerCfg(
        num_steps_per_env=1,
        num_updates_per_iteration=2,
        max_iterations=3,
        obs_groups={"actor": ["policy"]},
        save_interval=10,
        experiment_name="off_policy_test",
        algorithm={"class_name": "package:Algorithm"},
    )


def test_off_policy_cfg_serializes_the_official_runner_contract() -> None:
    """The public config retains every generic fixed-step runner section."""
    cfg = _off_policy_cfg()

    assert cfg.class_name == "OffPolicyRunner"
    assert str(cfg.class_type) == "rsl_rl.runners:OffPolicyRunner"
    assert cfg.get_algorithm_class_name() == "package:Algorithm"
    assert cfg.to_dict()["num_updates_per_iteration"] == 2
    assert "random_action_steps" not in cfg.to_dict()
    assert "lifecycle_extension" not in cfg.to_dict()
    assert "empirical_normalization" not in cfg.to_dict()
    assert "torch_compile_mode" not in cfg.to_dict()
    assert cfg.to_dict()["num_envs"] is None
    assert cfg.init_at_random_ep_len


def test_runner_environment_count_uses_cli_then_profile_then_environment() -> None:
    """CLI overrides profiles, while generic runners retain the environment count."""
    generic = _off_policy_cfg()
    profiled = generic.replace(num_envs=50)

    assert generic.resolve_num_envs(None, 4096) == 4096
    assert profiled.resolve_num_envs(None, 4096) == 50
    assert profiled.resolve_num_envs(7, 4096) == 7


def test_off_policy_cfg_skips_on_policy_deprecation_migrations() -> None:
    """RSL-RL 5 migration leaves mapping-owned off-policy sections unchanged."""
    cfg = _off_policy_cfg()
    before = cfg.to_dict()

    assert handle_deprecated_rsl_rl_cfg(cfg, "5.3.0") is cfg
    assert cfg.to_dict() == before
