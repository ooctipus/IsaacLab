# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate profile-specific repeatability and paired Phase 3E systems aggregation."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
PROBE = ROOT / "motion_environment_probe.py"
GPU_OWNERSHIP = ROOT / "gpu_ownership.py"
AGGREGATOR = ROOT / "aggregate_motion_environment_systems.py"
ENVIRONMENT_IDENTITY = ROOT / "motion_environment_identity.py"
CONTRACT_V4 = ROOT / "fixtures/motion_environment_systems_contract_v4.json"


def _module():
    spec = importlib.util.spec_from_file_location("aggregate_motion_environment_systems", AGGREGATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gpu_ownership_module():
    spec = importlib.util.spec_from_file_location("gpu_ownership", GPU_OWNERSHIP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _environment_identity_module():
    spec = importlib.util.spec_from_file_location("motion_environment_identity", ENVIRONMENT_IDENTITY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aggregator_dependency_schema_matches_its_identity_producer() -> None:
    """Systems evidence must advance whenever the environment identity schema advances."""
    assert _module()._DEPENDENCY_SCHEMA == _environment_identity_module()._SCHEMA


def test_aggregator_critical_dependencies_exclude_learner_evaluation() -> None:
    """Critical environment identity must exclude learner-side tracking and metrics."""
    module = _module()
    dependencies = set(module._CRITICAL_COMMON_DEPENDENCIES)
    for profile_dependencies in module._CRITICAL_PROFILE_DEPENDENCIES.values():
        dependencies.update(profile_dependencies)

    assert "isaaclab_tasks.core.multi_task.motion_env" not in dependencies
    assert all(".motion.config.agents" not in name for name in dependencies)
    assert all(".motion.metrics" not in name for name in dependencies)
    assert all(".motion.evaluation" not in name for name in dependencies)


def _signature(scale: float) -> dict[str, object]:
    return {
        "observations": {"state": {"sum": scale, "squared_sum": scale * scale}},
        "reward": {"sum": -scale, "squared_sum": scale * scale},
        "final_observations": {"state": {"sum": 0.5 * scale, "squared_sum": 0.25 * scale * scale}},
    }


def _repeatability_record(scale: float = 1.0) -> dict[str, object]:
    """Return the three-layer repeatability evidence emitted by one probe."""
    return {
        "reset_state_identity": {
            "observations.state": {
                "dtype": "torch.float32",
                "shape": [2, 3],
                "sha256": "a" * 64,
            }
        },
        "one_edge_signature": _signature(scale),
        "trajectory_signature": _signature(scale),
    }


def test_repeatability_contract_keeps_g1_exact_over_the_full_horizon() -> None:
    """The deterministic PhysX profile must reject any full-trajectory difference."""
    module = _module()
    contract = json.loads(CONTRACT_V4.read_text())["profiles"]["g1_lafan"]["repeatability"]
    expected = _repeatability_record()
    actual = _repeatability_record()
    actual["trajectory_signature"]["reward"]["sum"] += 1.0e-7

    with pytest.raises(ValueError, match="full-horizon signature differs"):
        module._compare_repeatability(actual, expected, contract)


def test_repeatability_contract_models_smpl_solver_numerics_at_their_source() -> None:
    """SMPL must bind exact reset state, bound one edge, then require semantic horizon health."""
    module = _module()
    contract = json.loads(CONTRACT_V4.read_text())["profiles"]["smpl_cmu"]["repeatability"]
    expected = _repeatability_record()
    actual = _repeatability_record()
    actual["one_edge_signature"]["observations"]["state"]["sum"] += 5.0e-4
    actual["trajectory_signature"]["reward"]["sum"] += 1.0e6

    result = module._compare_repeatability(actual, expected, contract)

    assert result["reset_state_identity_passed"] is True
    assert result["one_edge"]["comparison"] == "bounded_signature"
    assert result["one_edge"]["maximum_signature_absolute_error"] == pytest.approx(5.0e-4)
    assert result["full_horizon"] == {
        "comparison": "semantic_finiteness",
        "maximum_signature_absolute_error": None,
        "passed": True,
    }


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("reset", "b" * 64, "reset-state identity differs"),
        ("edge", 2.0e-3, "one-edge signature differs"),
    ),
)
def test_repeatability_contract_rejects_smpl_reset_or_one_edge_drift(
    field: str,
    value: str | float,
    match: str,
) -> None:
    """The semantic-horizon rule must not hide reset or immediate solver regressions."""
    module = _module()
    contract = json.loads(CONTRACT_V4.read_text())["profiles"]["smpl_cmu"]["repeatability"]
    expected = _repeatability_record()
    actual = _repeatability_record()
    if field == "reset":
        actual["reset_state_identity"]["observations.state"]["sha256"] = value
    else:
        actual["one_edge_signature"]["reward"]["sum"] += value

    with pytest.raises(ValueError, match=match):
        module._compare_repeatability(actual, expected, contract)


def _dependency_identity() -> dict[str, object]:
    """Return one closed synthetic G1 dependency manifest."""
    module = _module()
    names = module._CRITICAL_COMMON_DEPENDENCIES | module._CRITICAL_PROFILE_DEPENDENCIES["g1_lafan"]
    resolved_axes = {"preset": "g1_lafan"}
    assert "isaaclab.envs.mdp.terminations" in module._CRITICAL_COMMON_DEPENDENCIES
    assert "isaaclab_tasks.core.multi_task.motion.mdp.terminations" not in names
    assert "isaaclab_tasks.core.multi_task.motion.mdp.commands.observations" not in names
    assert "isaaclab_tasks.core.multi_task.motion.robots.g1.history" not in names
    assert "isaaclab_tasks.core.multi_task.motion.robots.g1.transition" not in names
    assert "isaaclab_tasks.core.multi_task.motion.robots.g1.actions" in names
    assert "isaaclab_tasks.core.multi_task.motion.robots.g1.rewards" not in names

    resolved_configuration = {name: {} for name in sorted(module._CRITICAL_CONFIGURATION_AXES)}
    package = {
        "module_version": "1.0",
        "distribution_version": "1.0",
        "module_source_sha256": "3" * 64,
    }
    runtime_dependencies = {
        "isaac_sim": {"version": "6.0", "version_sha256": "4" * 64},
        "joblib": dict(package),
        "newton": dict(package),
        "newton_owners": {
            "model_builder_add_mjcf": {
                "owner": "newton._src.sim.builder:ModelBuilder.add_mjcf",
                "source_sha256": "5" * 64,
            }
        },
        "numpy": dict(package),
        "torch": {
            **package,
            "cuda_version": "12.8",
            "git_version": "6" * 40,
        },
        "warp": dict(package),
    }

    identity = {
        "schema": module._DEPENDENCY_SCHEMA,
        "preset": "g1_lafan",
        "resolved_axes": resolved_axes,
        "resolved_axes_sha256": hashlib.sha256(
            json.dumps(resolved_axes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "resolved_configuration": resolved_configuration,
        "resolved_configuration_sha256": hashlib.sha256(
            json.dumps(resolved_configuration, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "runtime_dependencies": runtime_dependencies,
        "runtime_dependencies_sha256": hashlib.sha256(
            json.dumps(runtime_dependencies, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "python_sources": {name: "1" * 64 for name in sorted(names)},
        "robot_assets": {"g1.usd": "2" * 64},
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return {**identity, "bundle_sha256": hashlib.sha256(canonical).hexdigest()}


def _record(
    *,
    role: str,
    num_envs: int,
    replicate: int,
    pair_index: int | None,
    capture: bool,
    throughput: float,
    peak_allocated: int,
    peak_reserved: int,
) -> dict[str, object]:
    semantic_steps = 501
    terminal_rows = 2 * num_envs if role == "capture_cost" else num_envs
    environment_seed = 7400 + pair_index if pair_index is not None else 7301
    action_seed = 10000 + pair_index if pair_index is not None else 9917
    warmup_steps = 32 if role == "capture_cost" else 8
    pair_position = None
    if pair_index is not None:
        pair_position = int(capture) if pair_index % 2 == 0 else int(not capture)
    return {
        "schema": "forward_backward_phase3e_motion_environment_probe_v9",
        "code_identity": {
            "probe_sha256": "a" * 64,
            "gpu_ownership_sha256": "b" * 64,
            "dependency_identity": _dependency_identity(),
        },
        "evidence": {
            "role": role,
            "replicate": replicate,
            "pair_index": pair_index,
            "pair_position": pair_position,
        },
        "benchmark_gpu_ownership": {
            "required_scope": "physical_gpu" if role == "capture_cost" else None,
            "before_benchmark": (
                {
                    "physical_gpu_uuid": "GPU-fixture-0",
                    "owner_pid": 1000 + 2 * (pair_index or 0) + int(capture),
                    "compute_pids": [1000 + 2 * (pair_index or 0) + int(capture)],
                    "competing_compute_pids": [],
                    "exclusive": True,
                }
                if role == "capture_cost"
                else None
            ),
            "after_benchmark": (
                {
                    "physical_gpu_uuid": "GPU-fixture-0",
                    "owner_pid": 1000 + 2 * (pair_index or 0) + int(capture),
                    "compute_pids": [1000 + 2 * (pair_index or 0) + int(capture)],
                    "competing_compute_pids": [],
                    "exclusive": True,
                }
                if role == "capture_cost"
                else None
            ),
        },
        "execution_started_unix_ns": (
            1 + pair_index * 10 + pair_position
            if pair_index is not None and pair_position is not None
            else 1 + replicate
        ),
        "preset": "g1_lafan",
        "physics_manager": "PhysxManager",
        "physics_details": {"manager": "PhysxManager"},
        "motion_split": "evaluation",
        "source_artifact_root": "/artifacts/phase3/source_deployments/g1_lafan",
        "reference_artifact_root": "/artifacts/phase3/source_deployments/g1_lafan",
        "seed": environment_seed,
        "num_envs": num_envs,
        "action_width": 29,
        "observation_shapes": {"state": [num_envs, 64]},
        "physics_dt": 0.005,
        "control_decimation": 4,
        "control_dt": 0.02,
        "configured_horizon_steps": 500,
        "applied_actions_before_timeout": semantic_steps,
        "semantic_steps": semantic_steps,
        "task_table_identity": "c" * 64,
        "task_table_builder_identity": "d" * 64,
        "task_table_builder_version": "g1_lafan_exact_mjcf_v1",
        "task_table_clip_count": 40,
        "task_table_frame_count": 264_705,
        "action": {
            "seed": action_seed,
            "bound": 0.25,
            "period_steps": 17,
            "minimum": -0.25,
            "maximum": 0.25,
            "sha256": f"{num_envs:064x}",
        },
        "terminated_rows": 0,
        "truncated_rows": num_envs,
        "final_rows": num_envs,
        "captured_final_rows": num_envs,
        "missing_final_rows": 0,
        "terminal_episode_steps": [semantic_steps],
        "applied_rows": semantic_steps * num_envs,
        "finite_observations": True,
        "finite_rewards": True,
        "finite_final_observations": True,
        "nonfinite_observation_rows": 0,
        "reset_source_rows": {"reference": num_envs, "lie_down": 0},
        "reset_state_identity": _repeatability_record()["reset_state_identity"],
        "one_edge_signature": _signature(float(num_envs + (pair_index or 0))),
        "trajectory_signature": _signature(float(num_envs + (pair_index or 0))),
        "benchmark": {
            "capture_final_obs": capture,
            "warmup_steps": warmup_steps,
            "measured_steps": 2 * semantic_steps if role == "capture_cost" else 64,
            "transitions_per_second": throughput,
            "measured_seconds": num_envs * (1.0 / throughput),
            "terminal_rows": terminal_rows,
            "expected_terminal_rows": terminal_rows,
            "captured_final_rows": terminal_rows if capture else 0,
            "missing_final_rows": 0 if capture else terminal_rows,
            "peak_allocated_increment_bytes": peak_allocated,
            "peak_reserved_increment_bytes": peak_reserved,
        },
    }


def _records(tmp_path: Path) -> list[tuple[Path, dict]]:
    records = []
    for num_envs in (1, 16, 1024):
        for replicate in range(2):
            record = _record(
                role="repeatability",
                num_envs=num_envs,
                replicate=replicate,
                pair_index=None,
                capture=True,
                throughput=1000.0,
                peak_allocated=0,
                peak_reserved=0,
            )
            path = tmp_path / f"scale_{num_envs}_{replicate}.json"
            path.write_text(json.dumps(record))
            records.append((path, record))
    for pair_index in range(5):
        for capture in (False, True):
            record = _record(
                role="capture_cost",
                num_envs=1024,
                replicate=pair_index,
                pair_index=pair_index,
                capture=capture,
                throughput=980.0 if capture else 1000.0,
                peak_allocated=4 * 1024 * 1024 if capture else 0,
                peak_reserved=8 * 1024 * 1024 if capture else 0,
            )
            path = tmp_path / f"capture_{pair_index}_{int(capture)}.json"
            path.write_text(json.dumps(record))
            records.append((path, record))
    return records


def test_probe_accumulates_signatures_only_in_the_untimed_semantic_pass() -> None:
    """Trajectory reductions must not contaminate the capture-cost timed loop."""
    tree = ast.parse(PROBE.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    semantic = ast.get_source_segment(PROBE.read_text(), functions["_semantic_pass"])
    benchmark = ast.get_source_segment(PROBE.read_text(), functions["_benchmark_steps"])
    assert semantic is not None and "trajectory_signature" in semantic and "_signature_add_" in semantic
    assert benchmark is not None and "trajectory_signature" not in benchmark and "_signature_add_" not in benchmark
    signature_add = ast.get_source_segment(PROBE.read_text(), functions["_signature_add_"])
    assert signature_add is not None and "to(dtype=torch.float64)" in signature_add


def test_probe_hashes_a_flat_canonical_view_of_singleton_stride_tensors() -> None:
    """Exact reset hashing must support logically contiguous views whose last stride is not one."""
    source = PROBE.read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_tensor_identity")
    function_source = ast.get_source_segment(source, function)
    assert function_source is not None
    assert 'numpy().tobytes(order="C")' in function_source
    assert "view(torch.uint8)" not in function_source
    assert "root_link_state_w" not in source
    assert "body_link_state_w" not in source


def test_probe_snapshots_declared_config_before_environment_construction() -> None:
    """Constructor path normalization must not rewrite the recorded semantic identity."""
    source = PROBE.read_text()
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    main_source = ast.get_source_segment(source, main)
    assert main_source is not None
    snapshot = "configured_cfg = copy.deepcopy(cfg)"
    construction = "env = ManagerBasedRLEnv(cfg=cfg)"
    assert snapshot in main_source
    assert main_source.index(snapshot) < main_source.index(construction)
    assert "cfg=configured_cfg" in main_source


def test_probe_capture_ownership_is_scoped_to_measured_physical_gpu() -> None:
    """Capture cost must isolate its allocator device without requiring other GPUs to be idle."""
    source = PROBE.read_text()
    assert "exclusive_physical_gpu_snapshot" in source
    assert "exclusive_host_gpu_snapshot" not in source
    assert '"required_scope": "physical_gpu"' in source


def test_physical_gpu_ownership_rejects_a_competing_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The measured physical GPU must have exactly one compute owner."""
    module = _gpu_ownership_module()
    monkeypatch.setattr(module.os, "getpid", lambda: 1000)
    completed = module.subprocess.CompletedProcess(
        args=("nvidia-smi",),
        returncode=0,
        stdout="GPU-owner, 1000\nGPU-owner, 2000\nGPU-other, 3000\n",
        stderr="",
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: completed)
    with pytest.raises(RuntimeError, match="GPU-owner.*2000"):
        module.exclusive_physical_gpu_snapshot("cuda:0")


def test_physical_gpu_ownership_allows_compute_on_another_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process on another physical GPU cannot affect the measured CUDA allocator."""
    module = _gpu_ownership_module()
    monkeypatch.setattr(module.os, "getpid", lambda: 1000)
    completed = module.subprocess.CompletedProcess(
        args=("nvidia-smi",),
        returncode=0,
        stdout="GPU-owner, 1000\nGPU-other, 2000\n",
        stderr="",
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: completed)
    snapshot = module.exclusive_physical_gpu_snapshot("cuda:0")
    assert snapshot["physical_gpu_uuid"] == "GPU-owner"
    assert module.validate_same_exclusive_gpu(snapshot, snapshot) == "GPU-owner"


def test_aggregator_rejects_capture_rows_that_violate_counterbalanced_order(tmp_path: Path) -> None:
    """Every pair must declare and execute its frozen off/on order."""
    module = _module()
    records = _records(tmp_path)
    for _path, record in records:
        if record["evidence"]["role"] == "capture_cost":
            pair_index = record["evidence"]["pair_index"]
            record["evidence"]["pair_position"] = int(record["benchmark"]["capture_final_obs"])
            record["execution_started_unix_ns"] = 1 + pair_index * 10 + record["evidence"]["pair_position"]
    with pytest.raises(ValueError, match="counterbalanced capture order"):
        module.aggregate(records, "g1_lafan", CONTRACT_V4)


def test_aggregator_accepts_complete_repeatability_and_paired_matrix(tmp_path: Path) -> None:
    """Two scale repetitions and five balanced cost pairs must produce one passing summary."""
    report = _module().aggregate(_records(tmp_path), "g1_lafan")
    assert report["status"] == "passed"
    assert report["raw_record_count"] == 16
    assert [row["num_envs"] for row in report["repeatability_scale"]["results"]] == [1, 16, 1024]
    scale_results = report["repeatability_scale"]["results"]
    assert all(row["reset_state_identity_passed"] for row in scale_results)
    assert all(row["one_edge"]["maximum_signature_absolute_error"] == 0.0 for row in scale_results)
    assert all(row["full_horizon"]["maximum_signature_absolute_error"] == 0.0 for row in scale_results)
    assert report["capture_cost"]["throughput_loss_fraction"]["mean"] == pytest.approx(0.02)
    assert report["capture_cost"]["peak_allocated_increment_delta_bytes"]["mean"] == 4 * 1024 * 1024
    assert report["gates"]["passed"] is True


def test_aggregator_reset_source_mapping_order_is_not_semantic(tmp_path: Path) -> None:
    """JSON key sorting must not change the reset-source count meaning."""
    records = _records(tmp_path)
    for _path, record in records:
        rows = record["reset_source_rows"]
        record["reset_source_rows"] = {"lie_down": rows["lie_down"], "reference": rows["reference"]}

    assert _module().aggregate(records, "g1_lafan")["status"] == "passed"


def test_aggregator_identity_ignores_checkout_placement(tmp_path: Path) -> None:
    """Artifact deployment paths must not split one content-closed environment identity."""
    records = _records(tmp_path)
    for index, (_path, record) in enumerate(records):
        record["source_artifact_root"] = f"/deployment/source-{index}"
        record["reference_artifact_root"] = f"/deployment/reference-{index}"
    report = _module().aggregate(records, "g1_lafan")

    assert "source_artifact_root" not in report["environment_identity"]
    assert "reference_artifact_root" not in report["environment_identity"]


def test_aggregator_normalizes_native_actuator_rows_per_environment(tmp_path: Path) -> None:
    """Native actuator ownership must be invariant per environment across vector scales."""
    module = _module()
    identities = []
    for _path, record in _records(tmp_path):
        record["physics_manager"] = "NewtonMJWarpManager"
        record["physics_details"] = {
            "manager": "NewtonMJWarpManager",
            "native_action_writer": "NativeMujocoControlAction",
            "native_actuator_rows": 69 * record["num_envs"],
        }
        identities.append(module._environment_identity(record))

    identity = module._assert_equal(identities, "production environment identity")

    assert identity["physics_details"] == {
        "manager": "NewtonMJWarpManager",
        "native_action_writer": "NativeMujocoControlAction",
        "native_actuator_rows_per_env": 69,
    }


def test_aggregator_rejects_nonintegral_native_actuator_ownership(tmp_path: Path) -> None:
    """A total row count that cannot belong equally to every environment is invalid."""
    module = _module()
    record = next(record for _path, record in _records(tmp_path) if record["num_envs"] == 16)
    record["physics_details"] = {
        "manager": "NewtonMJWarpManager",
        "native_action_writer": "NativeMujocoControlAction",
        "native_actuator_rows": 69 * record["num_envs"] + 1,
    }

    with pytest.raises(ValueError, match="native actuator rows"):
        module._environment_identity(record)


def test_aggregator_rejects_an_exact_profile_trajectory_mismatch(tmp_path: Path) -> None:
    """Matching counts with different trajectory reductions violate exact G1 repeatability."""
    records = _records(tmp_path)
    scale = next(
        record
        for _path, record in records
        if record["evidence"]["role"] == "repeatability" and record["evidence"]["replicate"] == 1
    )
    scale["trajectory_signature"]["reward"]["sum"] += 1.0
    with pytest.raises(ValueError, match="full-horizon signature differs"):
        _module().aggregate(records, "g1_lafan")


def test_aggregator_rejects_shared_or_different_capture_gpus(tmp_path: Path) -> None:
    """Capture timing is invalid when a GPU is shared or a pair changes hardware."""
    records = _records(tmp_path)
    first_capture = next(record for _path, record in records if record["evidence"]["role"] == "capture_cost")
    before = first_capture["benchmark_gpu_ownership"]["before_benchmark"]
    before["compute_pids"].append(9001)
    before["competing_compute_pids"].append(9001)
    before["exclusive"] = False
    with pytest.raises(ValueError, match="sole physical-GPU compute ownership"):
        _module().aggregate(records, "g1_lafan")

    records = _records(tmp_path)
    pair_zero = [
        record
        for _path, record in records
        if record["evidence"]["role"] == "capture_cost"
        and record["evidence"]["replicate"] == 0
        and record["evidence"]["pair_index"] == 0
    ]
    pair_zero[1]["benchmark_gpu_ownership"]["before_benchmark"]["physical_gpu_uuid"] = "GPU-fixture-1"
    pair_zero[1]["benchmark_gpu_ownership"]["after_benchmark"]["physical_gpu_uuid"] = "GPU-fixture-1"
    with pytest.raises(ValueError, match="different physical GPUs"):
        _module().aggregate(records, "g1_lafan")


def test_aggregator_rejects_dependency_manifest_that_omits_the_task_table(tmp_path: Path) -> None:
    """A self-consistent partial manifest must not hide a changed output owner."""
    records = _records(tmp_path)
    for _path, record in records:
        dependency = record["code_identity"]["dependency_identity"]
        del dependency["python_sources"]["isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table"]
        identity = {name: value for name, value in dependency.items() if name != "bundle_sha256"}
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        dependency["bundle_sha256"] = hashlib.sha256(canonical).hexdigest()
    with pytest.raises(ValueError, match="omits critical source owners"):
        _module().aggregate(records, "g1_lafan")


@pytest.mark.parametrize(
    "projection,mutate,match",
    (
        (
            "resolved_configuration",
            lambda value: value.pop("actions"),
            "omits critical semantic axes",
        ),
        (
            "runtime_dependencies",
            lambda value: value["newton_owners"].pop("model_builder_add_mjcf"),
            "Newton runtime owner fields differ",
        ),
    ),
)
def test_aggregator_rejects_self_consistent_partial_v5_projections(
    tmp_path: Path,
    projection: str,
    mutate,
    match: str,
) -> None:
    """Canonical hashes must not make an incomplete semantic projection acceptable."""
    records = _records(tmp_path)
    for _path, record in records:
        dependency = record["code_identity"]["dependency_identity"]
        mutate(dependency[projection])
        dependency[f"{projection}_sha256"] = hashlib.sha256(
            json.dumps(dependency[projection], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        identity = {name: value for name, value in dependency.items() if name != "bundle_sha256"}
        dependency["bundle_sha256"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    with pytest.raises(ValueError, match=match):
        _module().aggregate(records, "g1_lafan")


@pytest.mark.parametrize(
    ("preset", "profile_packages"),
    (
        ("smpl_cmu", {"h5py", "mujoco", "mujoco_warp"}),
        ("g1_lafan", {"joblib"}),
        ("g1_cmu", {"h5py"}),
    ),
)
def test_runtime_dependency_validator_requires_exact_preset_packages(
    preset: str,
    profile_packages: set[str],
) -> None:
    """Systems evidence must reject missing and undeclared decoder/runtime packages."""
    package = {
        "module_version": "1.0",
        "distribution_version": "1.0",
        "module_source_sha256": "3" * 64,
    }
    owners = {
        "model_builder_add_mjcf": {
            "owner": "newton._src.sim.builder:ModelBuilder.add_mjcf",
            "source_sha256": "5" * 64,
        }
    }
    if preset == "smpl_cmu":
        owners.update(
            {
                "model_builder_add_usd": {"owner": "newton:ModelBuilder.add_usd", "source_sha256": "5" * 64},
                "schema_resolver_mjc": {"owner": "newton:SchemaResolverMjc", "source_sha256": "5" * 64},
                "solver_mujoco": {"owner": "newton:SolverMuJoCo", "source_sha256": "5" * 64},
            }
        )
    runtime = {
        "isaac_sim": {"version": "6.0", "version_sha256": "4" * 64},
        "newton": dict(package),
        "newton_owners": owners,
        "numpy": dict(package),
        "torch": {
            **package,
            "cuda_version": "12.8",
            "git_version": "6" * 40,
        },
        "warp": dict(package),
        **{name: dict(package) for name in profile_packages},
    }
    module = _module()

    module._validate_runtime_dependencies(runtime, preset)
    for name in {"numpy"} | profile_packages:
        missing = dict(runtime)
        missing.pop(name)
        with pytest.raises(ValueError, match="runtime dependency fields differ"):
            module._validate_runtime_dependencies(missing, preset)
    with pytest.raises(ValueError, match="runtime dependency fields differ"):
        module._validate_runtime_dependencies({**runtime, "undeclared": dict(package)}, preset)
