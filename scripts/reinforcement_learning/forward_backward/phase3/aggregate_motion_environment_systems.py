# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate the repeatability-scale and paired final-capture Phase 3E matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

_PROBE_SCHEMA = "forward_backward_phase3e_motion_environment_probe_v9"
_SUMMARY_SCHEMA = "forward_backward_phase3e_motion_environment_systems_v4"
_CONTRACT_SCHEMA = "forward_backward_phase3e_motion_environment_systems_contract_v4"
_DEFAULT_CONTRACT = Path(__file__).parent / "fixtures/motion_environment_systems_contract_v4.json"
_CODE_IDENTITY_FIELDS = {"probe_sha256", "gpu_ownership_sha256", "dependency_identity"}
_DEPENDENCY_SCHEMA = "forward_backward_phase3_motion_environment_dependency_identity_v7"
_CRITICAL_COMMON_DEPENDENCIES = {
    "motion_environment_identity",
    "isaaclab.sim.schemas.schemas",
    "isaaclab.sim.schemas.schemas_cfg",
    "isaaclab.sim.spawners.shapes.shapes",
    "isaaclab.sim.spawners.shapes.shapes_cfg",
    "isaaclab.sim.utils.prims",
    "isaaclab_tasks.core.multi_task.motion_env",
    "isaaclab_tasks.core.multi_task.motion_env_cfg",
    "isaaclab_tasks.core.multi_task.motion.config.environment",
    "isaaclab_tasks.core.multi_task.motion.config.profiles",
    "isaaclab_tasks.core.multi_task.motion.config.simulations",
    "isaaclab_tasks.core.multi_task.motion.config.sources",
    "isaaclab_tasks.core.multi_task.motion.mdp.actions",
    "isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_state_payload",
    "isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table",
    "isaaclab_tasks.core.multi_task.motion.mdp.runtime",
}
_CRITICAL_PROFILE_DEPENDENCIES = {
    "g1_lafan": {
        "isaaclab_physx.physics.physx_manager",
        "isaaclab_tasks.core.multi_task.motion.config.robots.g1",
        "isaaclab_tasks.core.multi_task.motion.data.importers.bfm_g1_joblib",
        "isaaclab_tasks.core.multi_task.motion.trajectory.g1",
    },
    "g1_cmu": {
        "isaaclab_assets.robots.smpl.smpl_constants",
        "isaaclab_physx.physics.physx_manager",
        "isaaclab_tasks.core.multi_task.motion.config.robots.g1",
        "isaaclab_tasks.core.multi_task.motion.config.robots.smpl",
        "isaaclab_tasks.core.multi_task.motion.data.importers.humenv_hdf5",
        "isaaclab_tasks.core.multi_task.motion.trajectory.g1_smpl",
    },
    "smpl_cmu": {
        "isaaclab_assets.robots.smpl.smpl_cfg",
        "isaaclab_assets.robots.smpl.smpl_constants",
        "isaaclab_newton.cloner.newton_clone_utils",
        "isaaclab_newton.cloner.replicate",
        "isaaclab_newton.physics.mjwarp_manager",
        "isaaclab_newton.physics.newton_manager",
        "isaaclab_newton.sim.schemas.schemas_cfg",
        "isaaclab_newton.sim.spawners.mjcf.mjcf",
        "isaaclab_newton.sim.spawners.mjcf.mjcf_cfg",
        "isaaclab_tasks.core.multi_task.motion.config.robots.smpl",
        "isaaclab_tasks.core.multi_task.motion.data.importers.humenv_hdf5",
        "isaaclab_tasks.core.multi_task.motion.trajectory.smpl",
    },
}

_CRITICAL_CONFIGURATION_AXES = {
    "actions",
    "commands",
    "curriculum",
    "events",
    "observations",
    "rewards",
    "scene",
    "sim",
    "terminations",
}


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one raw record."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _stable(value: object) -> str:
    """Serialize one identity projection deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _assert_equal(values: list[object], label: str) -> object:
    """Require exact equality across one controlled record group."""
    if not values:
        raise ValueError(f"No values were supplied for {label}.")
    expected = _stable(values[0])
    if any(_stable(value) != expected for value in values[1:]):
        raise ValueError(f"Records disagree on {label}.")
    return values[0]


def _is_sha256(value: object) -> bool:
    """Return whether a value is one canonical lowercase SHA-256 digest."""
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_runtime_dependencies(value: object, preset: str) -> None:
    """Require concrete package/build and Newton execution-owner identity."""
    profile_packages = {
        "smpl_cmu": {"h5py", "mujoco", "mujoco_warp"},
        "g1_lafan": {"joblib"},
        "g1_cmu": {"h5py"},
    }.get(preset)
    if profile_packages is None:
        raise ValueError(f"Unsupported motion environment preset: {preset!r}.")
    common = {"isaac_sim", "newton", "newton_owners", "numpy", "torch", "warp"}
    expected = common | profile_packages
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Motion environment runtime dependency fields differ.")

    isaac_sim = value["isaac_sim"]
    if (
        not isinstance(isaac_sim, dict)
        or set(isaac_sim) != {"version", "version_sha256"}
        or not isinstance(isaac_sim["version"], str)
        or not isaac_sim["version"]
        or not _is_sha256(isaac_sim["version_sha256"])
    ):
        raise ValueError("Isaac Sim build identity is incomplete.")

    package_fields = {"module_version", "distribution_version", "module_source_sha256"}
    for name in ("newton", "numpy", "warp", *sorted(profile_packages)):
        package = value[name]
        if (
            not isinstance(package, dict)
            or set(package) != package_fields
            or not isinstance(package["module_version"], str)
            or not package["module_version"]
            or package["distribution_version"] is not None
            and (not isinstance(package["distribution_version"], str) or not package["distribution_version"])
            or not _is_sha256(package["module_source_sha256"])
        ):
            raise ValueError(f"Runtime package identity is incomplete: {name}.")

    torch = value["torch"]
    if (
        not isinstance(torch, dict)
        or set(torch) != package_fields | {"cuda_version", "git_version"}
        or not all(
            isinstance(torch[name], str) and torch[name] for name in ("module_version", "cuda_version", "git_version")
        )
        or torch["distribution_version"] is not None
        and (not isinstance(torch["distribution_version"], str) or not torch["distribution_version"])
        or not _is_sha256(torch["module_source_sha256"])
    ):
        raise ValueError("Torch build identity is incomplete.")

    expected_owners = {"model_builder_add_mjcf"}
    if preset == "smpl_cmu":
        expected_owners |= {"model_builder_add_usd", "schema_resolver_mjc", "solver_mujoco"}
    owners = value["newton_owners"]
    if not isinstance(owners, dict) or set(owners) != expected_owners:
        raise ValueError("Newton runtime owner fields differ.")
    for name, owner in owners.items():
        if (
            not isinstance(owner, dict)
            or set(owner) != {"owner", "source_sha256"}
            or not isinstance(owner["owner"], str)
            or ":" not in owner["owner"]
            or not _is_sha256(owner["source_sha256"])
        ):
            raise ValueError(f"Newton runtime owner identity is incomplete: {name}.")


def _validate_dependency_identity(value: object, preset: str) -> None:
    """Require one unified resolved-axis, source, and robot-asset closure."""
    fields = {
        "schema",
        "preset",
        "resolved_axes",
        "resolved_axes_sha256",
        "resolved_configuration",
        "resolved_configuration_sha256",
        "runtime_dependencies",
        "runtime_dependencies_sha256",
        "python_sources",
        "robot_assets",
        "bundle_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Motion environment dependency identity fields differ.")
    if value["schema"] != _DEPENDENCY_SCHEMA or value["preset"] != preset:
        raise ValueError("Motion environment dependency identity schema or preset differs.")
    resolved_axes = value["resolved_axes"]
    if not isinstance(resolved_axes, dict) or resolved_axes.get("preset") != preset:
        raise ValueError("Motion environment dependency resolved axes differ from the selected preset.")
    if value["resolved_axes_sha256"] != hashlib.sha256(_stable(resolved_axes).encode()).hexdigest():
        raise ValueError("Motion environment dependency resolved-axis hash differs.")
    configuration = value["resolved_configuration"]
    if not isinstance(configuration, dict) or not _CRITICAL_CONFIGURATION_AXES.issubset(configuration):
        raise ValueError("Motion environment dependency resolved configuration omits critical semantic axes.")
    if value["resolved_configuration_sha256"] != hashlib.sha256(_stable(configuration).encode()).hexdigest():
        raise ValueError("Motion environment dependency resolved-configuration hash differs.")

    runtime_dependencies = value["runtime_dependencies"]
    if value["runtime_dependencies_sha256"] != hashlib.sha256(_stable(runtime_dependencies).encode()).hexdigest():
        raise ValueError("Motion environment runtime-dependency hash differs.")
    _validate_runtime_dependencies(runtime_dependencies, preset)

    if preset == "smpl_cmu":
        try:
            solver = configuration["sim"]["physics"]["solver_cfg"]
            ground_spawn = configuration["scene"]["ground"]["spawn"]
            collision = ground_spawn["collision_props"]
            material = ground_spawn["physics_material"]
        except (KeyError, TypeError):
            raise ValueError("SMPL configuration omits solver or global-ground ownership.") from None
        if (
            not isinstance(solver, dict)
            or not {"enable_multiccd", "enable_native_ccd", "integrator", "tolerance"}.issubset(solver)
            or not isinstance(collision, dict)
            or not {"margin", "solimp", "solref"}.issubset(collision)
            or not isinstance(material, dict)
            or not {"dynamic_friction", "static_friction"}.issubset(material)
        ):
            raise ValueError("SMPL configuration omits solver or global-ground contact fields.")

    sources = value["python_sources"]
    assets = value["robot_assets"]
    if not isinstance(sources, dict) or not isinstance(assets, dict) or not assets:
        raise ValueError("Motion environment dependency source or robot-asset manifest is empty.")
    required_sources = _CRITICAL_COMMON_DEPENDENCIES | _CRITICAL_PROFILE_DEPENDENCIES[preset]
    if not required_sources.issubset(sources):
        missing = tuple(sorted(required_sources.difference(sources)))
        raise ValueError(f"Motion environment dependency identity omits critical source owners: {missing}.")
    hashes = [
        value["resolved_axes_sha256"],
        value["resolved_configuration_sha256"],
        value["runtime_dependencies_sha256"],
        *sources.values(),
        *assets.values(),
        value["bundle_sha256"],
    ]
    if any(not _is_sha256(digest) for digest in hashes):
        raise ValueError("Motion environment dependency identity contains an invalid SHA-256.")
    if preset == "smpl_cmu":
        expected_assets = {"reference/humenv.xml", "simulation/robot.xml"}
        if set(assets) != expected_assets:
            raise ValueError("SMPL dependency identity must contain only its native simulation and reference MJCF.")
        converted_owners = {
            "isaaclab.sim.converters.asset_converter_base",
            "isaaclab.sim.converters.mjcf_converter",
            "isaaclab.sim.spawners.from_files.from_files",
        }
        if converted_owners.intersection(sources):
            raise ValueError("SMPL dependency identity must not retain the superseded MJCF-to-USD conversion path.")
    identity = {name: value[name] for name in fields - {"bundle_sha256"}}
    expected = hashlib.sha256(_stable(identity).encode()).hexdigest()
    if value["bundle_sha256"] != expected:
        raise ValueError("Motion environment dependency bundle hash differs from its manifest.")


def _signature_max_error(
    actual: object,
    expected: object,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    path: str = "trajectory_signature",
) -> float:
    """Require identical signature structure and tightly equal float64 reductions."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Signature structure differs at {path}.")
        return max(
            (
                _signature_max_error(
                    actual[name],
                    expected[name],
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                    path=f"{path}.{name}",
                )
                for name in expected
            ),
            default=0.0,
        )
    if (
        isinstance(actual, bool)
        or isinstance(expected, bool)
        or not isinstance(actual, (int, float))
        or not isinstance(expected, (int, float))
    ):
        raise ValueError(f"Signature leaf at {path} is not numeric.")
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
        raise ValueError(f"Signature leaf at {path} is non-finite.")
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise ValueError(f"Deterministic trajectory signature differs at {path}: {actual} != {expected}.")
    return abs(float(actual) - float(expected))


def _compared_signature(
    actual: object,
    expected: object,
    contract: object,
    *,
    label: str,
) -> dict[str, float | str | bool]:
    """Compare one numerical signature under an explicit contract."""
    if not isinstance(contract, dict):
        raise ValueError(f"{label} contract must be a mapping.")
    comparison = contract.get("comparison")
    expected_fields = {"comparison", "relative_tolerance", "absolute_tolerance"}
    if comparison not in {"exact_signature", "bounded_signature"} or set(contract) != expected_fields:
        raise ValueError(f"{label} contract is incomplete or unsupported.")
    relative_tolerance = contract["relative_tolerance"]
    absolute_tolerance = contract["absolute_tolerance"]
    if (
        isinstance(relative_tolerance, bool)
        or not isinstance(relative_tolerance, (int, float))
        or not math.isfinite(relative_tolerance)
        or relative_tolerance < 0.0
        or isinstance(absolute_tolerance, bool)
        or not isinstance(absolute_tolerance, (int, float))
        or not math.isfinite(absolute_tolerance)
        or absolute_tolerance < 0.0
        or comparison == "exact_signature"
        and (relative_tolerance != 0.0 or absolute_tolerance != 0.0)
    ):
        raise ValueError(f"{label} contract has invalid tolerances.")
    try:
        maximum_error = _signature_max_error(
            actual,
            expected,
            relative_tolerance=float(relative_tolerance),
            absolute_tolerance=float(absolute_tolerance),
            path=label,
        )
    except ValueError as error:
        raise ValueError(f"{label} differs: {error}") from None
    return {
        "comparison": comparison,
        "maximum_signature_absolute_error": maximum_error,
        "passed": True,
    }


def _compare_repeatability(
    actual: dict[str, object],
    expected: dict[str, object],
    contract: object,
) -> dict[str, object]:
    """Compare reset, immediate solver edge, and horizon at their proper layers."""
    if not isinstance(contract, dict) or set(contract) != {"reset_state", "one_edge", "full_horizon"}:
        raise ValueError("Profile repeatability contract fields differ.")
    if contract["reset_state"] != "exact_identity":
        raise ValueError("Profile repeatability must require exact reset-state identity.")
    if _stable(actual.get("reset_state_identity")) != _stable(expected.get("reset_state_identity")):
        raise ValueError("reset-state identity differs between repeated runs.")

    one_edge = _compared_signature(
        actual.get("one_edge_signature"),
        expected.get("one_edge_signature"),
        contract["one_edge"],
        label="one-edge signature",
    )
    full_horizon_contract = contract["full_horizon"]
    if not isinstance(full_horizon_contract, dict):
        raise ValueError("full-horizon contract must be a mapping.")
    if full_horizon_contract.get("comparison") == "semantic_finiteness":
        if set(full_horizon_contract) != {"comparison"}:
            raise ValueError("Semantic full-horizon contract cannot declare numerical tolerances.")
        full_horizon: dict[str, object] = {
            "comparison": "semantic_finiteness",
            "maximum_signature_absolute_error": None,
            "passed": True,
        }
    else:
        full_horizon = _compared_signature(
            actual.get("trajectory_signature"),
            expected.get("trajectory_signature"),
            full_horizon_contract,
            label="full-horizon signature",
        )
    return {
        "reset_state_identity_passed": True,
        "one_edge": one_edge,
        "full_horizon": full_horizon,
    }


def _environment_identity(record: dict) -> dict[str, object]:
    """Return the process-independent production environment identity."""
    physics_details = dict(record["physics_details"])
    if "native_actuator_rows" in physics_details:
        native_actuator_rows = physics_details.pop("native_actuator_rows")
        num_envs = record["num_envs"]
        if (
            not isinstance(native_actuator_rows, int)
            or isinstance(native_actuator_rows, bool)
            or native_actuator_rows < 1
            or not isinstance(num_envs, int)
            or isinstance(num_envs, bool)
            or num_envs < 1
            or native_actuator_rows % num_envs
        ):
            raise ValueError("native actuator rows must be a positive integer multiple of num_envs.")
        physics_details["native_actuator_rows_per_env"] = native_actuator_rows // num_envs
    return {
        "schema": record["schema"],
        "code_identity": record["code_identity"],
        "preset": record["preset"],
        "physics_manager": record["physics_manager"],
        "physics_details": physics_details,
        "motion_split": record["motion_split"],
        "action_width": record["action_width"],
        "observation_shapes_without_batch": {name: shape[1:] for name, shape in record["observation_shapes"].items()},
        "physics_dt": record["physics_dt"],
        "control_decimation": record["control_decimation"],
        "control_dt": record["control_dt"],
        "configured_horizon_steps": record["configured_horizon_steps"],
        "applied_actions_before_timeout": record["applied_actions_before_timeout"],
        "semantic_steps": record["semantic_steps"],
        "task_table_identity": record["task_table_identity"],
        "task_table_builder_identity": record["task_table_builder_identity"],
        "task_table_builder_version": record["task_table_builder_version"],
        "task_table_clip_count": record["task_table_clip_count"],
        "task_table_frame_count": record["task_table_frame_count"],
    }


def _run_identity(record: dict) -> dict[str, object]:
    """Return facts that must match between repeatability runs or one capture pair."""
    return {
        "environment": _environment_identity(record),
        "seed": record["seed"],
        "num_envs": record["num_envs"],
        "action": record["action"],
        "observation_shapes": record["observation_shapes"],
    }


def _validate_semantic(record: dict, profile: dict[str, object]) -> None:
    """Require one exact production-capture timeout vector."""
    num_envs = record["num_envs"]
    semantic_steps = record["semantic_steps"]
    if semantic_steps != profile["applied_actions_before_timeout"]:
        raise ValueError("Systems evidence must execute one complete native timeout horizon.")
    expected_profile = {
        "motion_split": profile["motion_split"],
        "physics_dt": profile["physics_dt_seconds"],
        "control_decimation": profile["control_decimation"],
        "control_dt": profile["control_dt_seconds"],
        "configured_horizon_steps": profile["configured_horizon_steps"],
        "applied_actions_before_timeout": profile["applied_actions_before_timeout"],
    }
    for name, value in expected_profile.items():
        if record[name] != value:
            raise ValueError(f"Record {name} differs from the declared profile contract.")
    expected = {
        "terminated_rows": 0,
        "truncated_rows": num_envs,
        "final_rows": num_envs,
        "captured_final_rows": num_envs,
        "missing_final_rows": 0,
        "applied_rows": semantic_steps * num_envs,
    }
    for name, value in expected.items():
        if record[name] != value:
            raise ValueError(f"Semantic count {name} is {record[name]}, expected {value}.")
    if record["terminal_episode_steps"] != [record["applied_actions_before_timeout"]]:
        raise ValueError("Terminal episode-step provenance differs from the native profile.")
    for name in ("finite_observations", "finite_rewards", "finite_final_observations"):
        if record[name] is not True:
            raise ValueError(f"Semantic probe did not prove {name}.")
    if record["nonfinite_observation_rows"] != 0:
        raise ValueError("Semantic probe contains non-finite observation rows.")
    reset_source_rows = record.get("reset_source_rows")
    if (
        not isinstance(reset_source_rows, dict)
        or set(reset_source_rows) != set(profile["reset_source_names"])
        or any(type(count) is not int or count < 0 for count in reset_source_rows.values())
        or sum(reset_source_rows.values()) != num_envs
    ):
        raise ValueError("Semantic reset-source rows differ from the declared task table contract.")
    signature = record["trajectory_signature"]
    if set(signature) != {"observations", "reward", "final_observations"}:
        raise ValueError("Trajectory signature has the wrong channel structure.")
    if signature["observations"].keys() != signature["final_observations"].keys():
        raise ValueError("Current and final trajectory signatures use different observation fields.")
    reset_identity = record.get("reset_state_identity")
    if not isinstance(reset_identity, dict) or not reset_identity:
        raise ValueError("Semantic probe lacks exact reset-state identity.")
    for name, identity in reset_identity.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(identity, dict)
            or set(identity) != {"shape", "dtype", "sha256"}
            or not isinstance(identity["shape"], list)
            or any(type(size) is not int or size < 0 for size in identity["shape"])
            or not isinstance(identity["dtype"], str)
            or not identity["dtype"].startswith("torch.")
            or not _is_sha256(identity["sha256"])
        ):
            raise ValueError("Semantic probe reset-state identity is malformed.")
    one_edge = record.get("one_edge_signature")
    if not isinstance(one_edge, dict) or not one_edge:
        raise ValueError("Semantic probe lacks a one-edge numerical signature.")
    _signature_max_error(
        one_edge,
        one_edge,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        path="one-edge signature",
    )


def _validate_capture_gpu_ownership(record: dict) -> str:
    """Require exclusive ownership of the measured physical GPU around timing."""
    ownership = record.get("benchmark_gpu_ownership")
    if not isinstance(ownership, dict) or ownership.get("required_scope") != "physical_gpu":
        raise ValueError("Capture evidence lacks required physical-GPU ownership proof.")
    snapshots = []
    for boundary in ("before_benchmark", "after_benchmark"):
        snapshot = ownership.get(boundary)
        if not isinstance(snapshot, dict):
            raise ValueError(f"Capture evidence lacks {boundary} GPU ownership proof.")
        required = {
            "physical_gpu_uuid",
            "owner_pid",
            "compute_pids",
            "competing_compute_pids",
            "exclusive",
        }
        if set(snapshot) != required:
            raise ValueError(f"Capture {boundary} GPU ownership fields differ.")
        owner_pid = snapshot["owner_pid"]
        if (
            not isinstance(snapshot["physical_gpu_uuid"], str)
            or not snapshot["physical_gpu_uuid"].startswith("GPU-")
            or type(owner_pid) is not int
            or snapshot["compute_pids"] != [owner_pid]
            or snapshot["competing_compute_pids"] != []
            or snapshot["exclusive"] is not True
        ):
            raise ValueError(f"Capture {boundary} did not prove sole physical-GPU compute ownership.")
        snapshots.append(snapshot)
    uuid = snapshots[0]["physical_gpu_uuid"]
    if snapshots[1]["physical_gpu_uuid"] != uuid or snapshots[1]["owner_pid"] != snapshots[0]["owner_pid"]:
        raise ValueError("Capture process or physical GPU changed across the timed boundary.")
    return uuid


def _statistics(
    values: list[float],
    *,
    expected_count: int,
    student_t_critical: float,
) -> dict[str, float | int | list[float]]:
    """Return sample statistics and a two-sided-normal-label 95% upper estimate."""
    if len(values) != expected_count or any(not math.isfinite(value) for value in values):
        raise ValueError(f"Capture evidence requires exactly {expected_count} finite pairs.")
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values)
    sem = sample_std / math.sqrt(len(values))
    return {
        "count": len(values),
        "values": values,
        "mean": mean,
        "sample_std": sample_std,
        "sem": sem,
        "minimum": min(values),
        "maximum": max(values),
        "upper_95_student_t": mean + student_t_critical * sem,
    }


def aggregate(
    records: list[tuple[Path, dict]],
    preset: str,
    contract_path: Path = _DEFAULT_CONTRACT,
) -> dict[str, object]:
    """Validate and aggregate one complete systems matrix."""
    if not records:
        raise ValueError("No motion-environment probe records were supplied.")
    contract = json.loads(contract_path.read_text())
    if contract.get("schema") != _CONTRACT_SCHEMA or preset not in contract.get("profiles", {}):
        raise ValueError("Systems evidence requires a valid declared profile contract.")
    shared = contract["shared"]
    profile = contract["profiles"][preset]
    scale_contract = shared["repeatability_scale"]
    capture_contract = shared["capture_cost"]
    gate_contract = shared["gates"]
    scale_counts = tuple(scale_contract["num_envs"])
    scale_repetitions = scale_contract["repetitions"]
    capture_pairs = capture_contract["pairs"]
    pair_capture_order = capture_contract.get("pair_capture_order")
    if (
        capture_contract.get("benchmark_gpu_ownership") != "only_compute_process_on_physical_gpu_at_boundaries"
        or not isinstance(pair_capture_order, list)
        or len(pair_capture_order) != capture_pairs
        or any(order not in (["off", "on"], ["on", "off"]) for order in pair_capture_order)
        or sum(order == ["off", "on"] for order in pair_capture_order)
        - sum(order == ["on", "off"] for order in pair_capture_order)
        not in (-1, 0, 1)
    ):
        raise ValueError("Capture contract does not declare a balanced physical-GPU-isolated pair order.")
    for _path, record in records:
        if record.get("schema") != _PROBE_SCHEMA:
            raise ValueError("Every raw record must use the Phase 3E probe-v9 schema.")
        if record.get("preset") != preset:
            raise ValueError(f"Every raw record must use preset {preset!r}.")
        if set(record.get("code_identity", {})) != _CODE_IDENTITY_FIELDS:
            raise ValueError("Raw record does not close over the complete environment data path.")
        _validate_dependency_identity(record["code_identity"]["dependency_identity"], preset)
        _validate_semantic(record, profile)
    environment_identity = _assert_equal(
        [_environment_identity(record) for _path, record in records],
        "production environment identity",
    )

    scale = [(path, record) for path, record in records if record["evidence"]["role"] == "repeatability"]
    capture = [(path, record) for path, record in records if record["evidence"]["role"] == "capture_cost"]
    standalone = [path for path, record in records if record["evidence"]["role"] == "standalone"]
    if standalone:
        raise ValueError(f"Systems matrix contains standalone records: {[path.name for path in standalone]}.")
    if len(scale) != len(scale_counts) * scale_repetitions:
        raise ValueError("Systems matrix requires two repeatability records at each declared scale.")
    if len(capture) != 2 * capture_pairs:
        raise ValueError("Systems matrix requires five paired capture-on/off records.")

    scale_results = []
    for num_envs in scale_counts:
        group = [(path, record) for path, record in scale if record["num_envs"] == num_envs]
        group.sort(key=lambda item: item[1]["evidence"]["replicate"])
        expected_replicates = list(range(scale_repetitions))
        if (
            len(group) != scale_repetitions
            or [item[1]["evidence"]["replicate"] for item in group] != expected_replicates
        ):
            raise ValueError(f"Scale {num_envs} does not contain every declared repeatability replicate.")
        first = group[0][1]
        second = group[1][1]
        _assert_equal([_run_identity(first), _run_identity(second)], f"scale-{num_envs} run identity")
        expected_scale = {
            "seed": scale_contract["environment_seed"],
            "action_seed": scale_contract["action_seed"],
            "action_bound": scale_contract["action_bound"],
            "action_period_steps": scale_contract["action_period_steps"],
            "warmup_steps": scale_contract["warmup_steps"],
            "benchmark_steps": scale_contract["benchmark_steps"],
        }
        for record in (first, second):
            evidence = record["evidence"]
            if (
                evidence["role"] != "repeatability"
                or evidence["pair_index"] is not None
                or evidence["pair_position"] is not None
            ):
                raise ValueError("Scale evidence has capture-pair metadata.")
            benchmark = record["benchmark"]
            actual_scale = {
                "seed": record["seed"],
                "action_seed": record["action"]["seed"],
                "action_bound": record["action"]["bound"],
                "action_period_steps": record["action"]["period_steps"],
                "warmup_steps": benchmark["warmup_steps"],
                "benchmark_steps": benchmark["measured_steps"],
            }
            if actual_scale != expected_scale or benchmark["capture_final_obs"] is not True:
                raise ValueError(f"Scale {num_envs} differs from the declared execution contract.")
        repeatability = _compare_repeatability(second, first, profile["repeatability"])
        scale_results.append(
            {
                "num_envs": num_envs,
                "replicates": [item[1]["evidence"]["replicate"] for item in group],
                "record_sha256": [_sha256(item[0]) for item in group],
                **repeatability,
            }
        )

    pair_results = []
    throughput_losses = []
    seconds_deltas = []
    peak_allocated_deltas = []
    peak_reserved_deltas = []
    capture_steps = capture_contract["measured_native_horizons"] * profile["applied_actions_before_timeout"]
    for pair_index in range(capture_pairs):
        group = [(path, record) for path, record in capture if record["evidence"]["pair_index"] == pair_index]
        if len(group) != 2:
            raise ValueError(f"Capture pair {pair_index} does not contain exactly two records.")
        by_position = {record["evidence"].get("pair_position"): (path, record) for path, record in group}
        if set(by_position) != {0, 1}:
            raise ValueError(f"Capture pair {pair_index} lacks two ordered positions.")
        ordered = [by_position[position] for position in (0, 1)]
        actual_order = ["on" if record["benchmark"]["capture_final_obs"] else "off" for _path, record in ordered]
        if actual_order != pair_capture_order[pair_index]:
            raise ValueError(f"Capture pair {pair_index} violates the counterbalanced capture order.")
        execution_starts = [record.get("execution_started_unix_ns") for _path, record in ordered]
        if (
            any(type(value) is not int or value < 1 for value in execution_starts)
            or execution_starts[0] >= execution_starts[1]
        ):
            raise ValueError(f"Capture pair {pair_index} execution order is absent or reversed.")
        by_capture = {record["benchmark"]["capture_final_obs"]: (path, record) for path, record in group}
        if set(by_capture) != {False, True}:
            raise ValueError(f"Capture pair {pair_index} must contain one capture-on and one capture-off record.")
        off_path, off = by_capture[False]
        on_path, on = by_capture[True]
        off_gpu_uuid = _validate_capture_gpu_ownership(off)
        on_gpu_uuid = _validate_capture_gpu_ownership(on)
        if off_gpu_uuid != on_gpu_uuid:
            raise ValueError(f"Capture pair {pair_index} used different physical GPUs.")
        _assert_equal([_run_identity(off), _run_identity(on)], f"capture-pair-{pair_index} run identity")
        repeatability = _compare_repeatability(on, off, profile["repeatability"])
        for record in (off, on):
            benchmark = record["benchmark"]
            expected_capture = {
                "num_envs": capture_contract["num_envs"],
                "seed": capture_contract["environment_seed_base"] + pair_index,
                "action_seed": capture_contract["action_seed_base"] + pair_index,
                "action_bound": capture_contract["action_bound"],
                "action_period_steps": capture_contract["action_period_steps"],
                "warmup_steps": capture_contract["warmup_steps"],
                "benchmark_steps": capture_steps,
            }
            actual_capture = {
                "num_envs": record["num_envs"],
                "seed": record["seed"],
                "action_seed": record["action"]["seed"],
                "action_bound": record["action"]["bound"],
                "action_period_steps": record["action"]["period_steps"],
                "warmup_steps": benchmark["warmup_steps"],
                "benchmark_steps": benchmark["measured_steps"],
            }
            if actual_capture != expected_capture:
                raise ValueError(f"Capture pair {pair_index} differs from the declared execution contract.")
            if benchmark["terminal_rows"] != benchmark["expected_terminal_rows"]:
                raise ValueError("Capture benchmark terminal accounting differs from its native horizon.")
        if off["benchmark"]["captured_final_rows"] != 0:
            raise ValueError("Capture-off benchmark materialized terminal observations.")
        if on["benchmark"]["missing_final_rows"] != 0:
            raise ValueError("Capture-on benchmark lost terminal observations.")

        off_throughput = off["benchmark"]["transitions_per_second"]
        on_throughput = on["benchmark"]["transitions_per_second"]
        throughput_loss = (off_throughput - on_throughput) / off_throughput
        seconds_delta = on["benchmark"]["measured_seconds"] - off["benchmark"]["measured_seconds"]
        peak_allocated_delta = (
            on["benchmark"]["peak_allocated_increment_bytes"] - off["benchmark"]["peak_allocated_increment_bytes"]
        )
        peak_reserved_delta = (
            on["benchmark"]["peak_reserved_increment_bytes"] - off["benchmark"]["peak_reserved_increment_bytes"]
        )
        throughput_losses.append(throughput_loss)
        seconds_deltas.append(seconds_delta)
        peak_allocated_deltas.append(float(peak_allocated_delta))
        peak_reserved_deltas.append(float(peak_reserved_delta))
        pair_results.append(
            {
                "pair_index": pair_index,
                "capture_order": pair_capture_order[pair_index],
                "execution_started_unix_ns": execution_starts,
                "capture_off_sha256": _sha256(off_path),
                "capture_on_sha256": _sha256(on_path),
                "physical_gpu_uuid": off_gpu_uuid,
                "throughput_loss_fraction": throughput_loss,
                "measured_seconds_delta": seconds_delta,
                "peak_allocated_increment_delta_bytes": peak_allocated_delta,
                "peak_reserved_increment_delta_bytes": peak_reserved_delta,
                "repeatability": repeatability,
            }
        )

    statistics_kwargs = {
        "expected_count": capture_pairs,
        "student_t_critical": gate_contract["student_t_critical_95_two_sided"],
    }
    throughput_statistics = _statistics(throughput_losses, **statistics_kwargs)
    seconds_statistics = _statistics(seconds_deltas, **statistics_kwargs)
    allocated_statistics = _statistics(peak_allocated_deltas, **statistics_kwargs)
    reserved_statistics = _statistics(peak_reserved_deltas, **statistics_kwargs)
    gates = {
        "throughput_loss_upper_95_at_most_0_05": (
            throughput_statistics["upper_95_student_t"] <= gate_contract["throughput_loss_upper_95_fraction"]
        ),
        "mean_peak_allocated_delta_at_most_16_mib": (
            allocated_statistics["mean"] <= gate_contract["mean_peak_allocated_delta_bytes"]
        ),
    }
    return {
        "schema": _SUMMARY_SCHEMA,
        "status": "passed" if all(gates.values()) else "failed",
        "aggregation_identity": {"aggregator_sha256": _sha256(Path(__file__).resolve())},
        "contract_identity": {
            "schema": contract["schema"],
            "file_sha256": _sha256(contract_path),
            "canonical_sha256": hashlib.sha256(_stable(contract).encode()).hexdigest(),
        },
        "preset": preset,
        "environment_identity": environment_identity,
        "raw_record_count": len(records),
        "raw_records": [
            {"name": path.name, "sha256": _sha256(path)}
            for path, _record in sorted(records, key=lambda item: item[0].name)
        ],
        "repeatability_scale": {
            "contract": profile["repeatability"],
            "results": scale_results,
        },
        "capture_cost": {
            "measured_steps": capture_steps,
            "pairs": pair_results,
            "throughput_loss_fraction": throughput_statistics,
            "measured_seconds_delta": seconds_statistics,
            "peak_allocated_increment_delta_bytes": allocated_statistics,
            "peak_reserved_increment_delta_bytes": reserved_statistics,
        },
        "gates": {
            "limits": {
                "throughput_loss_upper_95_fraction": gate_contract["throughput_loss_upper_95_fraction"],
                "mean_peak_allocated_delta_bytes": gate_contract["mean_peak_allocated_delta_bytes"],
            },
            "results": gates,
            "passed": all(gates.values()),
        },
    }


def main() -> None:
    """Load one raw-record directory and atomically persist its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--contract", type=Path, default=_DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*.json"))
    records = [(path, json.loads(path.read_text())) for path in paths]
    report = aggregate(records, args.preset, args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise RuntimeError("Motion-environment systems evidence did not pass its declared gates.")


if __name__ == "__main__":
    main()
