# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate import-free byte identity for explicit motion-environment modules."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROBE = Path(__file__).parent / "motion_environment_probe.py"

IDENTITY = Path(__file__).parent / "motion_environment_identity.py"


def _module():
    spec = importlib.util.spec_from_file_location("phase3_motion_environment_identity", IDENTITY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_profiles_map_once_to_independent_axes() -> None:
    module = _module()

    assert module.motion_environment_axes("smpl_cmu") == {
        "smpl",
        "cmu",
        "newton_mjwarp",
        "timing_sim450_control30_horizon300",
        "sampling_source_rows",
    }
    assert module.motion_environment_axes("g1_lafan") == {
        "g1",
        "lafan",
        "physx",
        "timing_sim200_control50_horizon501",
        "sampling_clip_time",
        "evidence_physical_auxiliary",
        "randomization_physics_observation_pose_push",
    }
    assert module.motion_environment_axes("g1_cmu") == {
        "g1",
        "cmu",
        "physx",
        "timing_sim200_control50_horizon501",
        "sampling_clip_time",
        "evidence_physical_auxiliary",
        "randomization_physics_observation_pose_push",
    }
    assert module.motion_runner_axes("smpl_cmu") == {
        "helpers_discriminator",
        "tracking_off",
        "model_plain_2x1024",
        "replay_transition_uniform_2m",
        "schedule_50x10_5m",
        "optimization_lr1e4_implied0p1_actor0p01",
        "context_online_10k",
        "exploration_std0p2_range1",
        "seed_0",
        "expert_clock_source_rows",
    }
    assert module.motion_runner_axes("g1_lafan") == {
        "helpers_discriminator_auxiliary",
        "tracking_reset_frame",
        "tracking_interval_9p6m",
        "model_residual_6x1024",
        "replay_episode_uniform_5120k",
        "schedule_1024x1_211p2m",
        "optimization_lr3e4_implied0_actor0p05",
        "context_expert_half_8192",
        "exploration_std0p05_range5",
        "seed_4728",
        "expert_clock_50hz",
    }
    assert module.motion_runner_axes("g1_cmu") == module.motion_runner_axes("g1_lafan")
    with pytest.raises(ValueError, match="Unsupported motion experiment profile"):
        module.motion_environment_axes("unknown")
    with pytest.raises(ValueError, match="Unsupported motion experiment profile"):
        module.motion_runner_axes("unknown")


def _dependency_identity(module, *, runtime_version: str = "one", source_sha256: str = "a" * 64) -> dict:
    resolved_axes = {"preset": "smpl_cmu"}

    resolved_configuration = {"compute_final_obs": True}
    runtime_dependencies = {"newton": {"module_version": runtime_version}}
    identity = {
        "schema": module._SCHEMA,
        "preset": "smpl_cmu",
        "resolved_axes": resolved_axes,
        "resolved_axes_sha256": module._json_hash(resolved_axes),
        "resolved_configuration": resolved_configuration,
        "resolved_configuration_sha256": module._json_hash(resolved_configuration),
        "runtime_dependencies": runtime_dependencies,
        "runtime_dependencies_sha256": module._json_hash(runtime_dependencies),
        "python_sources": {"motion": source_sha256},
        "robot_assets": {"simulation/robot.xml": "b" * 64},
    }
    return {**identity, "bundle_sha256": module._json_hash(identity)}


def test_probe_selects_importer_from_concrete_source_identity_without_fallback() -> None:
    source = PROBE.read_text()

    assert 'table_cfg.source.identifier == "cmu_humenv_smpl"' in source
    assert 'table_cfg.source.identifier == "lafan_g1_29dof"' in source
    assert 'table_cfg.source.identifier == "g1_lafan"' not in source
    assert "else CmuHumEnvSmplClips" not in source


def test_environment_identity_excludes_derived_sampler_law() -> None:
    """Environment identity stores primitive row mode and reset policy, not a derived runtime label."""
    source = IDENTITY.read_text()

    assert "motion_task_sampling_law" not in source
    assert '"task_sampling_law"' not in source


def test_environment_identity_closes_procedural_ground_contact_and_native_mjcf_runtime() -> None:
    """Ground, contact, and native MJCF owners must participate in byte identity."""
    module = _module()

    assert {
        "isaaclab.sim.schemas.schemas",
        "isaaclab.sim.schemas.schemas_cfg",
        "isaaclab.sim.spawners.shapes.shapes",
        "isaaclab.sim.spawners.materials.physics_materials",
        "isaaclab.sim.spawners.materials.physics_materials_cfg",
        "isaaclab.sim.spawners.shapes.shapes_cfg",
        "isaaclab.sim.utils.prims",
    } <= set(module._COMMON_MODULES)
    assert "isaaclab_tasks.core.multi_task.motion.tracking" not in module._COMMON_MODULES
    assert "isaaclab_newton.sim.schemas.schemas_cfg" in module._SMPL_MODULES

    assert "isaaclab_tasks.core.multi_task.motion.mdp.terminations" not in module._COMMON_MODULES
    assert "isaaclab.envs.mdp.terminations" in module._COMMON_MODULES
    assert "isaaclab_tasks.core.multi_task.motion.robots.g1.history" not in module._G1_MODULES
    assert "isaaclab_tasks.core.multi_task.motion.mdp.commands.observations" not in module._COMMON_MODULES
    assert "isaaclab_tasks.core.multi_task.motion.robots.g1.transition" not in module._G1_MODULES
    assert "isaaclab_tasks.core.multi_task.motion.robots.g1.actions" in module._G1_MODULES
    assert "isaaclab_tasks.core.multi_task.motion.robots.g1.rewards" not in module._G1_MODULES
    assert module._SCHEMA.endswith("dependency_identity_v9")

    assert {
        "isaaclab_newton.sim.spawners.mjcf.mjcf",
        "isaaclab_newton.sim.spawners.mjcf.mjcf_cfg",
    } <= set(module._SMPL_MODULES)
    assert {
        "isaaclab.sim.converters.asset_converter_base",
        "isaaclab.sim.converters.mjcf_converter",
        "isaaclab.sim.spawners.from_files.from_files",
    }.isdisjoint(module._SMPL_MODULES)


def test_environment_identity_closes_only_active_environment_sources() -> None:
    """Provenance must follow the active generic environment, not learner-side evaluation."""
    module = _module()
    active_modules = {
        "isaaclab.envs.manager_based_env",
        "isaaclab.envs.manager_based_env_cfg",
        "isaaclab.envs.manager_based_rl_env",
        "isaaclab.envs.manager_based_rl_env_cfg",
    }

    assert active_modules <= set(module._COMMON_MODULES)
    assert "isaaclab_tasks.core.multi_task.motion_env" not in module._COMMON_MODULES
    assert all(module._module_source_path(name).is_file() for name in active_modules)


def test_simulation_robot_assets_hashes_source_without_generated_cache(tmp_path: Path) -> None:
    """Direct-source spawners bind evidence to source bytes, not generated USD caches."""
    module = _module()
    source = tmp_path / "robot.xml"
    source.write_text("<mujoco model='robot'/>")
    generated = tmp_path / "robot" / "robot.usda"
    generated.parent.mkdir()
    generated.write_text("host-specific generated cache")
    robot = SimpleNamespace(spawn=SimpleNamespace(asset_path=str(source), usd_dir=str(tmp_path)))

    assets = module._simulation_robot_assets(robot)

    assert assets == {"simulation/robot.xml": hashlib.sha256(source.read_bytes()).hexdigest()}
    assert all("usda" not in name for name in assets)


def test_environment_semantic_identity_excludes_host_runtime_but_closes_source_bytes() -> None:
    """Host package versions may differ, while source-byte changes must stale portable evidence."""
    module = _module()
    baseline = _dependency_identity(module)
    other_runtime = _dependency_identity(module, runtime_version="two")
    other_source = _dependency_identity(module, source_sha256="c" * 64)

    semantic_sha256 = module.motion_environment_semantic_sha256
    assert baseline["bundle_sha256"] != other_runtime["bundle_sha256"]
    assert semantic_sha256(baseline) == semantic_sha256(other_runtime)
    assert semantic_sha256(baseline) != semantic_sha256(other_source)

    stale = dict(baseline)
    stale["bundle_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not internally closed"):
        semantic_sha256(stale)


def test_environment_compatibility_never_relabels_historical_provenance() -> None:
    """Source and contract drift are explicit results, not receipt mutations."""
    module = _module()
    historical = _dependency_identity(module)

    assert module.motion_environment_compatibility(historical, historical)["status"] == "exact_producer_match"

    changed_source = _dependency_identity(module, source_sha256="c" * 64)
    source_result = module.motion_environment_compatibility(historical, changed_source)
    assert source_result["status"] == "declared_contract_match_requires_runtime_validation"
    assert source_result["producer_sources_match"] is False
    assert source_result["declared_contract_matches"] is True
    assert historical["python_sources"] == {"motion": "a" * 64}

    changed_contract = dict(historical)
    changed_contract["resolved_configuration"] = {"compute_final_obs": False}
    changed_contract["resolved_configuration_sha256"] = module._json_hash(changed_contract["resolved_configuration"])
    payload = dict(changed_contract)
    payload.pop("bundle_sha256")
    changed_contract["bundle_sha256"] = module._json_hash(payload)
    contract_result = module.motion_environment_compatibility(historical, changed_contract)
    assert contract_result["status"] == "declared_contract_mismatch"
    assert contract_result["declared_contract_matches"] is False


def test_module_source_path_hashes_an_explicit_module_without_importing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module with import-time Kit failure must remain hashable as inert source bytes."""
    source = tmp_path / "explosive" / "runtime.py"
    source.parent.mkdir()
    source.write_text('raise RuntimeError("must not import")\nVALUE = 3\n')
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    module = _module()

    resolved = module._module_source_path("explosive.runtime")

    assert resolved == source.resolve()
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == module._sha256(resolved)
    assert "explosive" not in sys.modules
    assert "explosive.runtime" not in sys.modules


def test_module_source_path_resolves_an_editable_package_without_importing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PEP 660 package root outside sys.path must expose inert target bytes."""
    package_root = tmp_path / "editable_package"
    package_root.mkdir()
    source = package_root / "runtime.py"
    source.write_text('raise RuntimeError("must not import")\nVALUE = 4\n')
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if Path(entry or ".") != tmp_path])
    module = _module()

    def find_spec(name: str):
        assert name == "editable_package"
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(package_root)]
        return spec

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    assert module._module_source_path("editable_package.runtime") == source.resolve()
    assert "editable_package" not in sys.modules
    assert "editable_package.runtime" not in sys.modules


def test_module_source_path_resolves_an_editable_namespace_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PEP 660 project-root location must resolve its nested package bytes."""
    project_root = tmp_path / "project"
    source = project_root / "editable_package" / "runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text('raise RuntimeError("must not import")\nVALUE = 5\n')
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if Path(entry or ".") != project_root])
    module = _module()

    def find_spec(name: str):
        assert name == "editable_package"
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(project_root)]
        return spec

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    assert module._module_source_path("editable_package.runtime") == source.resolve()
    assert "editable_package" not in sys.modules
    assert "editable_package.runtime" not in sys.modules


def test_module_source_path_rejects_ambiguous_explicit_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two distinct sys.path owners must fail instead of selecting by host order."""
    roots = (tmp_path / "one", tmp_path / "two")
    for index, root in enumerate(roots):
        source = root / "package" / "runtime.py"
        source.parent.mkdir(parents=True)
        source.write_text(f"VALUE = {index}\n")
    monkeypatch.setattr(sys, "path", [*(str(root) for root in roots), *sys.path])
    module = _module()

    with pytest.raises(RuntimeError, match="ambiguous"):
        module._module_source_path("package.runtime")


def test_module_source_path_rejects_missing_and_symbolic_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing and symbolic module files must not weaken content closure."""
    real = tmp_path / "real.py"
    real.write_text("VALUE = 1\n")
    symbolic = tmp_path / "package" / "runtime.py"
    symbolic.parent.mkdir()
    symbolic.symlink_to(real)
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    module = _module()

    with pytest.raises(ValueError, match="symbolic"):
        module._module_source_path("package.runtime")
    with pytest.raises(ModuleNotFoundError, match="missing.runtime"):
        module._module_source_path("missing.runtime")
