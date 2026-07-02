# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for identity-bound native SMPL-CMU EMD evidence."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).with_name("smpl_cmu_emd_evidence.py")


@pytest.fixture(scope="module")
def module():
    """Load the standalone EMD evidence producer without launching Isaac Sim."""
    spec = importlib.util.spec_from_file_location("smpl_cmu_emd_evidence", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _checkpoint(tmp_path: Path, module) -> Path:
    checkpoint = tmp_path / "model_1000.pt"
    torch.save({"model_state_dict": {"weight": torch.arange(3)}}, checkpoint)
    manifest = {
        "schema": module._COMPACT_SCHEMA,
        "iteration": 1000,
        "collected_transitions": 500_000,
        "output": {
            "filename": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": module._sha256(checkpoint),
        },
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(manifest))
    return checkpoint


def test_checkpoint_identity_closes_compact_bytes_and_transition(tmp_path: Path, module) -> None:
    checkpoint = _checkpoint(tmp_path, module)

    identity = module._checkpoint_identity(checkpoint)

    assert identity["iteration"] == 1000
    assert identity["transition"] == 500_000
    assert identity["sha256"] == module._sha256(checkpoint)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="differ from their manifest"):
        module._checkpoint_identity(checkpoint)


def test_publish_is_atomic_and_exclusive(tmp_path: Path, module) -> None:
    output = tmp_path / "500000.json"

    module._publish(output, {"schema": module._SCHEMA, "status": "measured"})

    assert json.loads(output.read_text()) == {"schema": module._SCHEMA, "status": "measured"}
    assert not tuple(tmp_path.glob(".*.tmp"))
    with pytest.raises(FileExistsError, match="already exists"):
        module._publish(output, {"schema": module._SCHEMA})


def test_runtime_declares_evaluation_split_and_all_three_horizon_authorities(module) -> None:
    """The long held-out clips must not inherit the 300-step training timeout."""
    source = inspect.getsource(module._run)

    assert 'table_cfg.motion_split = "evaluation"' in source
    assert "cfg.commands.motion.payload.episode_length_steps = evaluation_horizon" in source
    assert 'cfg.terminations.time_out.params["applied_actions_before_timeout"] = evaluation_horizon' in source
    assert "cfg.episode_length_s = evaluation_horizon * cfg.sim.dt * cfg.decimation" in source


def test_record_protocol_names_exact_native_projection_without_fake_diagnostic(module) -> None:
    source = inspect.getsource(module._run)

    assert module._SCHEMA == "forward_backward_phase3_native_motion_emd_v2"
    assert "native_humenv_observation_frames_1_to_T_minus_1_columns_0_to_213" in source
    assert "mean_next_up_to_8_backward_features_then_project" in source
    assert "exact_uniform_optimal_assignment_float32_euclidean_cost" in source
    assert "uninterrupted_global_horizon_with_same_step_final_obs_fallback" in source
    assert "must not fabricate the G1-only obs_state_emd diagnostic" in source
    assert '"uniform_emd_warp_sha256": _source_sha256(uniform_emd_warp)' in source


def test_source_identity_unwraps_decorated_runtime_boundaries(tmp_path: Path, module) -> None:
    """A decorator implementation must not replace the measured source owner."""
    decorator = tmp_path / "decorator.py"
    decorator.write_text(
        "import functools\n"
        "def decorate(value):\n"
        "    @functools.wraps(value)\n"
        "    def wrapper():\n"
        "        return value()\n"
        "    return wrapper\n"
    )
    owner = tmp_path / "owner.py"
    owner.write_text("def measured():\n    return 1\n")
    decorator_spec = importlib.util.spec_from_file_location("smpl_emd_source_decorator", decorator)
    owner_spec = importlib.util.spec_from_file_location("smpl_emd_source_owner", owner)
    assert decorator_spec is not None and decorator_spec.loader is not None
    assert owner_spec is not None and owner_spec.loader is not None
    decorator_module = importlib.util.module_from_spec(decorator_spec)
    owner_module = importlib.util.module_from_spec(owner_spec)
    decorator_spec.loader.exec_module(decorator_module)
    owner_spec.loader.exec_module(owner_module)
    owner_module.measured = decorator_module.decorate(owner_module.measured)

    assert module._source_sha256(owner_module.measured) == module._sha256(owner)


def test_native_environment_identity_retains_mjcf_spawner_cloner_and_simulator_hashes(module) -> None:
    digest = "a" * 64
    identity = {
        "python_sources": {name: digest for name in module._NATIVE_ENVIRONMENT_SOURCE_OWNERS},
        "robot_assets": {
            "simulation/smpl_robot.xml": "b" * 64,
            "reference/humanoid.xml": "c" * 64,
        },
    }

    owners = module._native_environment_owner_hashes(identity)

    assert set(owners["python_sources"]) == set(module._NATIVE_ENVIRONMENT_SOURCE_OWNERS)
    assert set(owners["robot_assets"]) == {"simulation/smpl_robot.xml", "reference/humanoid.xml"}
    identity["python_sources"].pop("isaaclab_newton.cloner.replicate")
    with pytest.raises(ValueError, match="missing native source owners"):
        module._native_environment_owner_hashes(identity)


def test_source_inspection_closes_its_controlled_lifetime(module) -> None:
    source = inspect.getsource(module._run)

    assert "source_index = source.inspect()" in source
    assert "finally:\n        source.close()" in source
