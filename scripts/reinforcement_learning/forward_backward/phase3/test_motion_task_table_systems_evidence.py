# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate persisted Phase 3C MotionTaskTable storage and lookup evidence."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path

from motion_environment_identity import motion_environment_axes

from isaaclab_tasks.core.multi_task.motion.mdp.commands import MotionTaskTableCfg
from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import G1_REFERENCE_MJCF_SHA256
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets

from isaaclab_assets.robots.smpl.smpl_constants import SMPL_HUMENV_MJCF_PATH, SMPL_HUMENV_MJCF_SHA256

ROOT = Path(__file__).parent
BENCHMARK = ROOT / "benchmark_motion_task_table.py"
FIXTURES = ROOT / "fixtures"
RECORDS = (
    FIXTURES / "motion_task_table_lookup_smpl_cmu_cpu_v5.json",
    FIXTURES / "motion_task_table_lookup_g1_lafan_cpu_v5.json",
)


_COMMON_CONSTRUCTION_MODULES = {
    "isaaclab.sim.schemas.schemas_cfg",
    "isaaclab_tasks.core.multi_task.kinematics.newton_kinematics",
    "isaaclab_tasks.core.multi_task.kinematics.newton_kinematics_cfg",
    "isaaclab_tasks.core.multi_task.motion.identity",
    "isaaclab_tasks.core.multi_task.motion.data.clip_index",
    "isaaclab_tasks.core.multi_task.motion.data.source",
    "isaaclab_tasks.core.multi_task.motion.data.skeleton",
    "isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_state_payload",
    "isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_sampler",
    "isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table",
}
_PROFILE_CONSTRUCTION_MODULES = {
    "g1_lafan": {
        "isaaclab_tasks.core.multi_task.motion.robots.g1.articulation",
        "isaaclab_tasks.core.multi_task.motion.data.sources.lafan_g1_29dof",
        "isaaclab_tasks.core.multi_task.motion.robots.g1.frames",
        "isaaclab_tasks.core.multi_task.motion.robots.g1.reference",
    },
    "smpl_cmu": {
        "isaaclab_assets.robots.smpl.smpl_constants",
        "isaaclab_newton.sim.schemas.schemas_cfg",
        "isaaclab_tasks.core.multi_task.motion.robots.smpl.articulation",
        "isaaclab_tasks.core.multi_task.motion.data.sources.cmu_humenv_smpl",
        "isaaclab_tasks.core.multi_task.motion.robots.smpl.frames",
        "isaaclab_tasks.core.multi_task.motion.robots.smpl.reference",
    },
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_hash(value: object) -> str:
    """Hash one canonical JSON-compatible evidence value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_stored_code_identity(value: object) -> dict[str, object]:
    """Validate one historical construction identity without consulting current sources."""
    identity = dict(value)
    payload = {name: member for name, member in identity.items() if name != "bundle_sha256"}
    assert set(identity) == {
        "python_sources",
        "python_symbols",
        "reference_assets",
        "resolved_construction",
        "bundle_sha256",
    }
    assert identity["bundle_sha256"] == _json_hash(payload)
    for group in ("python_sources", "python_symbols", "reference_assets"):
        assert identity[group]
        assert all(isinstance(name, str) and name and _is_sha256(digest) for name, digest in identity[group].items())
    return identity


def _module_sha256(module_name: str) -> str:
    """Hash one independently declared construction source."""
    module = importlib.import_module(module_name)
    path = Path(module.__file__ or "")
    if not path.is_file():
        raise RuntimeError(f"Cannot locate construction source module {module_name!r}.")
    return _sha256(path)


def _symbol_sha256(value: object) -> str:
    """Hash one independently declared construction symbol."""
    return hashlib.sha256(inspect.getsource(value).encode()).hexdigest()


def _callable_name(value: object) -> str:
    """Return one stable module-qualified callable name."""
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(name, str):
        raise TypeError(f"Table construction callable lacks a stable Python identity: {value!r}.")
    return f"{module}:{name}"


def _resolved_construction_contract(preset: str) -> dict[str, object]:
    """Independently project the resolved inputs that determine one table."""
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes(preset))
    table_cfg = cfg.commands.motion.task_table
    payload_cfg = cfg.commands.motion.payload
    source = table_cfg.source
    split = source.train
    skeleton = source.build_skeleton()
    return {
        "preset": preset,
        "control_dt_seconds": float(cfg.sim.dt * cfg.decimation),
        "source": {
            "identifier": source.identifier,
            "format": source.format,
            "semantic_level": source.semantic_level,
            "source_fps": source.source_fps,
            "license": source.license,
            "open_source": _callable_name(source.open_source),
            "skeleton_factory": _callable_name(source.skeleton_factory),
            "skeleton_identity_sha256": skeleton.identity_sha256,
            "split": {
                "name": split.name,
                "artifact": split.artifact,
                "artifact_sha256": split.artifact_sha256,
                "source_content_sha256": split.source_content_sha256,
                "clip_count": split.clip_count,
                "frame_count": split.frame_count,
            },
        },
        "table": {
            "frame_builder_factory": _callable_name(table_cfg.frame_builder_factory),
            "reference_kinematics_factory": _callable_name(table_cfg.reference_kinematics_factory),
            "task_row_mode": table_cfg.task_row_mode,
        },
        "sampler": {"reset_sources": [list(value) for value in payload_cfg.reset_sources]},
    }


def _expected_code_identity(preset: str) -> dict[str, object]:
    """Recompute the exact profile-specific construction bundle."""
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes(preset))
    table_cfg = cfg.commands.motion.task_table
    modules = (
        _COMMON_CONSTRUCTION_MODULES
        | _PROFILE_CONSTRUCTION_MODULES[preset]
        | {
            value.__module__
            for value in (
                table_cfg.source.open_source,
                table_cfg.source.skeleton_factory,
                table_cfg.frame_builder_factory,
                table_cfg.reference_kinematics_factory,
            )
        }
    )
    python_sources = {name: _module_sha256(name) for name in sorted(modules)}
    python_sources["benchmark_motion_task_table"] = _sha256(BENCHMARK)
    python_symbols = {
        "isaaclab_tasks.core.multi_task.motion.mdp.commands.MotionTaskTableCfg": _symbol_sha256(MotionTaskTableCfg)
    }
    if preset == "g1_lafan":
        reference_assets = {"reference/g1_29dof.xml": G1_REFERENCE_MJCF_SHA256}
    else:
        reference_assets = {f"reference/{Path(SMPL_HUMENV_MJCF_PATH).name}": SMPL_HUMENV_MJCF_SHA256}
    identity = {
        "python_sources": python_sources,
        "python_symbols": python_symbols,
        "reference_assets": reference_assets,
        "resolved_construction": _resolved_construction_contract(preset),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return {**identity, "bundle_sha256": hashlib.sha256(canonical).hexdigest()}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_task_table_identity_owns_resolved_inputs_not_neighbor_mdp_modules() -> None:
    """Unrelated environment/runtime edits must not invalidate unchanged table evidence."""
    excluded = {
        "isaaclab_tasks.utils.hydra",
        "isaaclab_tasks.core.multi_task.motion_env_cfg",
        "isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg",
    }
    assert _COMMON_CONSTRUCTION_MODULES.isdisjoint(excluded)
    for preset in ("smpl_cmu", "g1_lafan"):
        identity = _expected_code_identity(preset)
        contract = identity["resolved_construction"]
        assert contract["preset"] == preset
        assert set(contract["table"]) == {
            "frame_builder_factory",
            "reference_kinematics_factory",
            "task_row_mode",
        }
        assert set(contract["sampler"]) == {"reset_sources"}
        assert set(identity["python_symbols"]) == {
            "isaaclab_tasks.core.multi_task.motion.mdp.commands.MotionTaskTableCfg"
        }


def test_benchmark_derives_sampling_label_without_runtime_api() -> None:
    """Receipt labeling must not expand the production sampler API."""
    source = BENCHMARK.read_text()

    assert "sampler.task_sampling_law" not in source
    assert 'table_cfg.task_row_mode == "source_frames"' in source


def test_motion_task_table_records_authenticate_measured_code_and_report_current_compatibility() -> None:
    """Every result authenticates measured inputs while current construction drift stays explicit."""
    expected = {
        "smpl_cmu": {
            "split": "train",
            "clips": 1_638,
            "frames": 730_307,
            "control_dt_seconds": 1.0 / 30.0,
            "physical_row_width": 450,
            "logical_row_width": 463,
            "root_reference_aliases": True,
            "task_row_mode": "source_frames",
            "task_sampling_law": "clip_categorical_then_discrete_source_frame_v1",
            "reset_source_names": ["motion", "fall"],
            "stored_field_names": [
                "joint_position",
                "joint_velocity",
                "body_position",
                "body_rotation",
                "body_linear_velocity",
                "body_angular_velocity",
            ],
        },
        "g1_lafan": {
            "split": "training",
            "clips": 862,
            "frames": 258_600,
            "control_dt_seconds": 0.02,
            "physical_row_width": 461,
            "logical_row_width": 474,
            "root_reference_aliases": True,
            "task_row_mode": "clip_time_ranges",
            "task_sampling_law": "clip_categorical_then_continuous_time_v1",
            "reset_source_names": ["reference", "lie_down"],
            "stored_field_names": [
                "joint_position",
                "joint_velocity",
                "body_position",
                "body_rotation",
                "body_linear_velocity",
                "body_angular_velocity",
            ],
        },
    }
    for path in RECORDS:
        record = _load(path)
        source = record["source"]
        declaration = expected[source["preset"]]
        assert record["schema"] == "forward_backward_phase3c_motion_task_table_lookup_v5"
        stored_identity = _validate_stored_code_identity(record["code_identity"])
        current_identity = _expected_code_identity(source["preset"])
        compatibility = (
            "exact_producer_match"
            if stored_identity == current_identity
            else "producer_changed_requires_fresh_benchmark"
        )
        assert compatibility in {"exact_producer_match", "producer_changed_requires_fresh_benchmark"}
        construction = stored_identity["resolved_construction"]
        assert construction["preset"] == source["preset"]
        assert construction["control_dt_seconds"] == record["parameters"]["control_dt_seconds"]
        split = construction["source"]["split"]
        assert {
            "split": split["name"],
            "artifact": split["artifact"],
            "artifact_sha256": split["artifact_sha256"],
            "source_content_sha256": split["source_content_sha256"],
            "clips": split["clip_count"],
            "frames": split["frame_count"],
        } == {
            name: source[name]
            for name in ("split", "artifact", "artifact_sha256", "source_content_sha256", "clips", "frames")
        }
        assert construction["table"]["task_row_mode"] == record["task_table"]["task_row_mode"]
        assert [name for name, _probability in construction["sampler"]["reset_sources"]] == record["motion_sampler"][
            "reset_source_names"
        ]
        runtime = record["runtime"]
        dependencies = runtime["dependencies"]
        assert runtime["dependencies_sha256"] == _json_hash(dependencies)
        expected_packages = {"newton", "newton_owners", "numpy", "torch", "warp"}
        expected_packages.add("joblib" if source["preset"] == "g1_lafan" else "h5py")
        assert set(dependencies) == expected_packages
        package_fields = {"module_version", "distribution_version", "module_source_sha256"}
        for name in expected_packages - {"newton_owners", "torch"}:
            assert set(dependencies[name]) == package_fields
        assert set(dependencies["torch"]) == package_fields | {"cuda_version", "git_version"}
        assert set(dependencies["newton_owners"]) == {"model_builder_add_mjcf"}
        assert set(dependencies["newton_owners"]["model_builder_add_mjcf"]) == {"owner", "source_sha256"}
        assert {name: source[name] for name in ("split", "clips", "frames")} == {
            name: declaration[name] for name in ("split", "clips", "frames")
        }
        table = record["task_table"]
        for name in (
            "physical_row_width",
            "logical_row_width",
            "root_reference_aliases",
            "task_row_mode",
            "stored_field_names",
        ):
            assert table[name] == declaration[name]
        if source["preset"] == "smpl_cmu":
            assert set(table["stored_field_names"]).isdisjoint(
                {
                    "root_position",
                    "root_rotation",
                    "root_linear_velocity",
                    "root_angular_velocity",
                    "observation",
                }
            )
        sampler = record["motion_sampler"]
        assert sampler["task_sampling_law"] == declaration["task_sampling_law"]
        assert sampler["reset_source_names"] == declaration["reset_source_names"]
        assert sampler["resident_bytes"] > 0
        assert source["remaining_clips_after_build"] == 0
        assert source["remaining_frames_after_build"] == 0
        assert record["parameters"]["control_dt_seconds"] == declaration["control_dt_seconds"]


def test_task_table_benchmark_derives_live_bodies_from_declared_g1_axes() -> None:
    """The benchmark must reuse the environment identity's concrete action-axis projection."""
    source = BENCHMARK.read_text()
    assert "_BODY_NAMES as G1_BODY_NAMES" not in source
    assert "_motion_live_axes(cfg)" in source
    assert "G1_BEHAVIOR_BODY_NAMES" not in source
    assert "G1_BEHAVIOR_JOINT_NAMES" not in source


def test_motion_task_table_storage_pareto_accounts_for_unique_physical_storage() -> None:
    """Resident bytes equal unique trajectory storage plus compact table metadata."""
    expected_tiers = {
        "smpl_cmu": {
            "reset_state_only": (151, "derived_if_stored_alone"),
            "expert_observation_if_stored_alone": (358, "derived_if_stored_alone"),
            "production_shared_root_reference": (450, "materialized"),
            "production_plus_duplicate_expert_projection": (808, "rejected_duplicate_corpus"),
        },
        "g1_lafan": {
            "reset_state_only": (71, "derived_if_stored_alone"),
            "production_shared_root_reference": (461, "materialized"),
            "expert_projection_if_stored_alone": (527, "derived_if_stored_alone"),
            "production_plus_duplicate_expert_projection": (988, "rejected_duplicate_corpus"),
        },
    }
    for path in RECORDS:
        record = _load(path)
        table = record["task_table"]
        source = record["source"]
        assert table["storage_contract_passed"] is True
        assert table["trajectory_bytes"] == source["frames"] * table["physical_row_width"] * 4
        assert table["resident_bytes"] == table["trajectory_bytes"] + table["compact_metadata_bytes"]
        assert table["compact_metadata_bytes"] > 0
        tiers = {tier["name"]: tier for tier in table["storage_pareto"]}
        assert set(tiers) == set(expected_tiers[source["preset"]])
        for name, (width, status) in expected_tiers[source["preset"]].items():
            tier = tiers[name]
            assert tier["float32_scalars_per_frame"] == width
            assert tier["status"] == status
            assert tier["dense_frame_bytes"] == source["frames"] * width * 4
            if status == "materialized":
                assert tier["dense_frame_bytes"] == table["trajectory_bytes"]


def test_motion_task_table_lookup_evidence_has_parity_pointer_stability_and_uncertainty() -> None:
    """Throughput is evidence only when fixed lookup matches its allocating oracle."""
    required = {
        "named_field_lookup",
        "reset_state_gather",
        "allocating_reference_oracle",
        "fixed_runtime_reference",
        "current_reached_reference",
        "bind_exact_capacity_table",
    }
    for path in RECORDS:
        record = _load(path)
        correctness = record["correctness"]
        assert correctness["fixed_reference_output_pointers_stable"] is True
        parity = correctness["fixed_reference_matches_allocating_oracle"]
        assert parity["passed"] is True
        assert set(parity["maximum_absolute_error_by_field"]) == set(record["task_table"]["field_names"])
        assert max(parity["maximum_absolute_error_by_field"].values()) <= 2.0e-6

        throughput = record["throughput"]
        assert set(throughput) == required
        for measurement in throughput.values():
            sample_count = record["parameters"]["samples"]
            assert measurement["sample_count"] == sample_count >= 5
            assert len(measurement["sample_seconds"]) == len(measurement["sample_rates"]) == sample_count
            assert all(value > 0.0 for value in measurement["sample_seconds"])
            assert measurement["minimum_rate"] > 0.0
            assert measurement["minimum_rate"] <= measurement["median_rate"] <= measurement["maximum_rate"]
