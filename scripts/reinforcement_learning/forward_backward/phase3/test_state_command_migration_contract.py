# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the frozen Phase 3 StateCommand migration boundary."""

from __future__ import annotations

import json
import re
from pathlib import Path

_FIXTURE = Path(__file__).parent / "fixtures" / "state_command_migration_v1.json"


def _load_contract() -> dict:
    return json.loads(_FIXTURE.read_text())


def test_state_command_migration_has_one_owner_per_mutable_concern():
    contract = _load_contract()

    assert contract["schema"] == "forward_backward.phase3.state_command_migration.v1"
    assert set(contract["decision"]) == {
        "task_table",
        "state_command",
        "payload",
        "curriculum",
        "motion_provider",
        "algorithm_bridge",
    }
    assert "selected row ids" in contract["decision"]["state_command"]
    assert "emit simulator reset writes" in contract["decision"]["payload"]
    assert "success rates" in contract["decision"]["curriculum"]
    assert "read-only reset, reference, expert-edge, and evaluation views" in contract["decision"]["motion_provider"]


def test_state_command_migration_preserves_only_the_public_shell_symbols():
    contract = _load_contract()

    assert contract["public_symbols_retained"] == [
        "isaaclab_tasks.core.multi_task.mdp.commands.StateCommand",
        "isaaclab_tasks.core.multi_task.mdp.commands.StateCommandCfg",
    ]
    removals = set(contract["state_command_removals"])
    assert {
        "TensorDict dependency",
        "get_target_obs_cache",
        "get_spawn_obs_cache",
        "success_rates",
        "set_reset_state call in _resample_command",
    } <= removals


def test_state_command_migration_covers_each_controlled_domain_and_caller_test():
    contract = _load_contract()
    units = {unit["name"]: unit for unit in contract["migration_units"]}

    assert set(units) == {
        "shared_command",
        "position_domain",
        "factory_domain",
        "curriculum_ownership",
        "learner_cache_decoupling",
        "superseded_motion_attempt",
    }
    all_paths = [path for unit in units.values() for path in unit["paths"]]
    assert len(all_paths) == len(set(all_paths))
    assert all(not Path(path).is_absolute() for path in all_paths)
    assert units["superseded_motion_attempt"]["action"] == "delete_after_replacement"

    assert contract["caller_tests_to_migrate"] == [
        "source/isaaclab_tasks/isaaclab_tasks/core/multi_task/tests/test_reset_accumulator_state_storage.py",
        "source/isaaclab_tasks/isaaclab_tasks/core/multi_task/tests/terrain/test_command_curriculum.py",
        "source/isaaclab_tasks/isaaclab_tasks/core/multi_task/tests/terrain/test_locomotion_position_future_command_cfg.py",
    ]


def test_state_command_migration_records_valid_pre_migration_content_hashes():
    contract = _load_contract()
    hashes = contract["pre_migration_hashes"]

    assert len(hashes) == 9
    assert all(not Path(path).is_absolute() for path in hashes)
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values())


def test_state_command_migration_gates_require_semantic_and_systems_proof():
    contract = _load_contract()
    gates = "\n".join(contract["gates"] + contract["required_new_tests"])

    for requirement in (
        "fails when the payload bind call is removed",
        "Position pre/post migration tensor oracle",
        "Factory pre/post migration tensor oracle",
        "stable storage",
        "hot resample time",
        "repository pre-commit",
    ):
        assert requirement in gates
