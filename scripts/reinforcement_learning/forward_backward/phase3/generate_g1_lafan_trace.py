# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare and capture the frozen native BFM-Zero G1 transition trace."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import pickle
import platform
import subprocess
import sys
import types
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
CONTRACT_MANIFEST = FIXTURES / "native_contract_manifest_v1.json"
PHASE2_ENVIRONMENT = FIXTURES / "g1_lafan_50hz_phase2_environment_v1.json"
EVIDENCE_CHANNELS = FIXTURES / "native_evidence_channels_v1.json"

SEED = 20_260_629
NUM_ENVS = 2
SUBSTEP_FIELDS = (
    "qpos",
    "qvel",
    "body_position",
    "body_rotation_xyzw",
    "body_linear_velocity",
    "body_angular_velocity",
    "contact_force",
    "applied_pd_torque",
)
ACTION_DIM = 29
TRACE_STEPS = 5
TRACE_HORIZON_SECONDS = 0.04
CONTROL_DT_SECONDS = 0.02
OBSERVATION_FIELDS = ("state", "last_action", "history_actor", "privileged_state")
TRACE_CLOSURE_MEMBERS = (
    "meta_humenv_next_step_trace_v1.json",
    "meta_humenv_next_step_trace_v1.npz",
    "g1_lafan_same_step_trace_v1.json",
    "g1_lafan_same_step_trace_v1.npz",
)

RELEASE_REVISION = "b87916f52d3d9e6eeba484f5e80851a235191837"
COMPLETED_CAPACITY_REVISION = "0a4132620e5b588752cf7d77cacda1720d6e4b08"
CADENCE_REVISION = "f0495e864ffcf332f346bd7b55e9aa108cb8b38f"
REFERENCE_CONFIG_SHA256 = "96889c351a919a907f1f6e3001c5213fe5469ffc4fc2d87c0ef52e422eba7d0f"
FINAL_CAPTURE_ADAPTER_SHA256 = "52a35f031b2808479cf0dbf75bdb1ead5c7c4292f33c6f3cb33ca8e367460cc7"
SYNTHETIC_SOURCE_IDENTIFIER = "phase3_synthetic_g1_periodic_v1"
SYNTHETIC_SOURCE_CLIP_IDS = ("synthetic_g1_periodic_00", "synthetic_g1_periodic_01")
SYNTHETIC_SOURCE_FRAMES = 360
SYNTHETIC_SOURCE_FPS = 30
G1_JOINT_AXES = np.eye(3, dtype=np.float32)[
    np.asarray((1, 0, 2, 1, 1, 0, 1, 0, 2, 1, 1, 0, 2, 0, 1, 1, 0, 2, 1, 0, 1, 2, 1, 0, 2, 1, 0, 1, 2))
]
G1_DEFAULT_JOINT_POSITION = np.asarray(
    (-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    dtype=np.float32,
)

NATIVE_SOURCE_HASHES = {
    "humanoidverse/agents/envs/humanoidverse_isaac.py": (
        "6b95ca28e6e45cef75f3a0ee76507d157e871604b95d25baaaf5c653107327eb"
    ),
    "humanoidverse/agents/utils.py": "ee8d66a79b4379b544b121d8d611b42013d15e3691b85fb4b1170119b31bf018",
    "humanoidverse/config/domain_rand/domain_rand.yaml": (
        "faa55c426be6202f7582ba318bac797e46dbedfa70e150e17fd69f2d5f742555"
    ),
    "humanoidverse/config/env/legged_motions.yaml": (
        "06b99581f90f1c89f6dc27aad8c918b66f167759c55e394bbab6b12c308d0726"
    ),
    "humanoidverse/config/exp/bfm_zero/bfm_zero.yaml": (
        "d63b2c4d3707c0127577c816730aa283a01e2b45649006cd6fb7140ae6a8d761"
    ),
    "humanoidverse/config/obs/bfm_zero_obs.yaml": ("3037b6ec30cd1084ae846e03e8f5479e817f9c94c10cce099ab9ace45dfd4f1d"),
    "humanoidverse/config/rewards/reward_bfm_zero.yaml": (
        "4dd2638380d1088e2e6bb964fbfb387dbbd03e98c41c57064d1b11c910d7752e"
    ),
    "humanoidverse/config/robot/g1/g1_29dof_hard_waist.yaml": (
        "b3c8b7d25faca3908d874bef4471f1362385428612cce845d2df5467cd71f53e"
    ),
    "humanoidverse/config/simulator/isaacsim.yaml": (
        "eff29df3a8bf96487cf25371009a17282fe85670b3c5815f1662f1cf29761bc2"
    ),
    "humanoidverse/envs/legged_base_task/legged_robot_base.py": (
        "bc00a197aac294bb7b8200194bc5daf6ed0801d1422151aa811d7ca34fe08776"
    ),
    "humanoidverse/envs/legged_robot_motions/legged_robot_motions.py": (
        "c2adfe2fa3d6db1b7d6ceb0c809aae2faf2f589f8321256ff05d6cba3747a359"
    ),
    "humanoidverse/simulator/isaacsim/isaacsim.py": (
        "c546e0390b6ef2c5cd37655f64f5d677a0c2a13df05c8a11e3508f902f12b4aa"
    ),
    "humanoidverse/utils/motion_lib/motion_lib_base.py": (
        "570cbfcd433820ec225807f16ebdad5fd4da6de9bb55a742dfb2968c15c4eb31"
    ),
    "humanoidverse/utils/motion_lib/torch_humanoid_batch.py": (
        "a720d7c17d1f0987bf11f48dbbb1a457e95a0338cf574c10b12b79cf92f73c77"
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _tensor_summary(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def synthetic_motion_source() -> dict[str, dict[str, np.ndarray | int]]:
    """Return two deterministic smooth G1 clips derived only from analytic functions."""
    frame = np.arange(SYNTHETIC_SOURCE_FRAMES, dtype=np.float64)
    time_seconds = frame / SYNTHETIC_SOURCE_FPS
    joint_index = np.arange(ACTION_DIM, dtype=np.float64)
    default_position = G1_DEFAULT_JOINT_POSITION.astype(np.float64)
    amplitude = 0.018 + 0.003 * np.remainder(joint_index, 5.0)
    source: dict[str, dict[str, np.ndarray | int]] = {}
    for clip_index, clip_id in enumerate(SYNTHETIC_SOURCE_CLIP_IDS):
        clip_phase = 0.43 * clip_index
        fundamental = (
            2.0 * np.pi * (0.31 + 0.04 * clip_index) * time_seconds[:, None] + 0.37 * joint_index[None, :] + clip_phase
        )
        joint_position = (
            default_position[None, :]
            + amplitude[None, :] * np.sin(fundamental)
            + 0.25 * amplitude[None, :] * np.sin(2.0 * fundamental + 0.17)
        )

        pose = np.zeros((SYNTHETIC_SOURCE_FRAMES, ACTION_DIM + 1, 3), dtype=np.float32)
        pose[:, 0, 0] = np.asarray(0.012 * np.sin(0.53 * time_seconds + clip_phase), dtype=np.float32)
        pose[:, 0, 1] = np.asarray(0.009 * np.sin(0.41 * time_seconds + 0.2 + clip_phase), dtype=np.float32)
        pose[:, 0, 2] = np.asarray(0.025 * np.sin(0.29 * time_seconds + 0.4 + clip_phase), dtype=np.float32)
        pose[:, 1:] = np.asarray(joint_position[:, :, None] * G1_JOINT_AXES[None, :, :], dtype=np.float32)

        root = np.empty((SYNTHETIC_SOURCE_FRAMES, 3), dtype=np.float32)
        root[:, 0] = np.asarray(
            0.04 * time_seconds + 0.006 * np.sin(0.47 * time_seconds + clip_phase), dtype=np.float32
        )
        root[:, 1] = np.asarray(0.012 * np.sin(0.37 * time_seconds + clip_phase), dtype=np.float32)
        root[:, 2] = np.asarray(0.80 + 0.008 * np.sin(0.61 * time_seconds + clip_phase), dtype=np.float32)
        source[clip_id] = {
            "root_trans_offset": np.ascontiguousarray(root),
            "pose_aa": np.ascontiguousarray(pose),
            "fps": SYNTHETIC_SOURCE_FPS,
        }
    return source


def _motion_source_content_sha256(source: Mapping[str, Mapping[str, np.ndarray | int]]) -> str:
    """Hash source clip identities, ordered fields, tensor metadata, and tensor values."""
    digest = hashlib.sha256()
    for clip_id, clip in source.items():
        digest.update(clip_id.encode())
        for field_name, value in clip.items():
            digest.update(field_name.encode())
            if isinstance(value, np.ndarray):
                array = np.ascontiguousarray(value)
                digest.update(str(array.dtype).encode())
                digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
                digest.update(array.tobytes())
            else:
                digest.update(json.dumps(value, separators=(",", ":")).encode())
    return digest.hexdigest()


def synthetic_motion_source_declaration() -> dict[str, Any]:
    """Describe the dataset-independent procedural input to the native simulator."""
    source = synthetic_motion_source()
    return {
        "kind": "procedural_synthetic",
        "identifier": SYNTHETIC_SOURCE_IDENTIFIER,
        "generator": "generate_g1_lafan_trace.py::synthetic_motion_source",
        "contains_dataset_values": False,
        "clip_ids": list(SYNTHETIC_SOURCE_CLIP_IDS),
        "clip_count": len(SYNTHETIC_SOURCE_CLIP_IDS),
        "frames_per_clip": SYNTHETIC_SOURCE_FRAMES,
        "fps": SYNTHETIC_SOURCE_FPS,
        "ordered_fields": ["root_trans_offset", "pose_aa", "fps"],
        "root_trans_offset": {"shape": [SYNTHETIC_SOURCE_FRAMES, 3], "dtype": "float32"},
        "pose_aa": {"shape": [SYNTHETIC_SOURCE_FRAMES, ACTION_DIM + 1, 3], "dtype": "float32"},
        "content_sha256": _motion_source_content_sha256(source),
    }


def _synthetic_motion_source_bytes() -> bytes:
    """Serialize the procedural source without reading caller-controlled pickle data."""
    return pickle.dumps(synthetic_motion_source(), protocol=4)


def write_synthetic_motion_source(path: Path) -> dict[str, int | str]:
    """Write the deterministic source in the standard pickle format accepted by joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_synthetic_motion_source_bytes())
    temporary.replace(path)
    return _file_identity(path)


def inspect_synthetic_motion_source(path: Path) -> dict[str, Any]:
    """Prove raw-byte equality with the procedural source without deserializing the file."""
    path = Path(path)
    declaration = synthetic_motion_source_declaration()
    expected_bytes = _synthetic_motion_source_bytes()
    expected_file = {
        "bytes": len(expected_bytes),
        "sha256": hashlib.sha256(expected_bytes).hexdigest(),
    }
    if not path.is_file():
        return {
            "schema": "forward_backward_phase3_synthetic_g1_source_inspection_v1",
            **declaration,
            "expected_file": expected_file,
            "file": None,
            "exact_recipe_match": False,
            "errors": ["source file does not exist"],
        }

    observed_bytes = path.read_bytes()
    exact_match = observed_bytes == expected_bytes
    return {
        "schema": "forward_backward_phase3_synthetic_g1_source_inspection_v1",
        **declaration,
        "expected_file": expected_file,
        "file": _file_identity(path),
        "exact_recipe_match": exact_match,
        "errors": [] if exact_match else ["serialized bytes differ from the procedural recipe"],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_revision(repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def fixed_actions() -> np.ndarray:
    """Return deterministic unsaturated behavior actions for five vector steps."""
    action_index = np.arange(ACTION_DIM, dtype=np.float32)[None, None, :]
    step = np.arange(TRACE_STEPS, dtype=np.float32)[:, None, None]
    env_phase = np.arange(NUM_ENVS, dtype=np.float32)[None, :, None] * np.float32(0.19)
    phase = action_index * np.float32(0.17) + step * np.float32(0.31) + env_phase
    return np.asarray(np.float32(0.12) * np.sin(phase), dtype=np.float32)


def process_actions(actions: np.ndarray) -> np.ndarray:
    """Apply the released normalized-action transform before native control."""
    value = np.asarray(actions, dtype=np.float32) * np.float32(5.0)
    return np.clip(value, np.float32(-5.0), np.float32(5.0))


def compose_environment_reward(values: Mapping[str, float], penalty_scale: float) -> float:
    """Compose raw native evidence with source dt and penalty curriculum exactly once."""
    composition = _load_json(EVIDENCE_CHANNELS)["profiles"]["g1_lafan_50hz"]["compositions"]["environment_scalar"]
    curriculum = set(composition["penalty_curriculum"]["channels"])
    return sum(
        coefficient * composition["control_dt_seconds"] * values[name] * (penalty_scale if name in curriculum else 1.0)
        for name, coefficient in composition["coefficients"].items()
    )


def compose_learner_auxiliary_reward(values: Mapping[str, float]) -> float:
    """Compose the released learner-side auxiliary evidence before normalization."""
    composition = _load_json(EVIDENCE_CHANNELS)["profiles"]["g1_lafan_50hz"]["compositions"]["learner_auxiliary_scalar"]
    return sum(coefficient * values[name] for name, coefficient in composition["coefficients"].items())


def capture_declaration() -> dict[str, Any]:
    """Return the host-independent scientific declaration for the G1 trace."""
    return {
        "schema": "forward_backward_phase3a_g1_lafan_trace_declaration_v1",
        "profile": "g1_lafan_50hz",
        "scientific_contract_manifest": _file_identity(CONTRACT_MANIFEST),
        "native_environment_byte_identity": {
            "released_baseline_revision": RELEASE_REVISION,
            "completed_capacity_run_revision": COMPLETED_CAPACITY_REVISION,
            "later_cadence_revision": CADENCE_REVISION,
            "environment_files_unchanged_across_revisions": True,
            "source_files": NATIVE_SOURCE_HASHES,
        },
        "exact_final_capture_adapter": {
            "revision": CADENCE_REVISION,
            "file": "phase2_adapter/environment.py",
            "sha256": FINAL_CAPTURE_ADAPTER_SHA256,
        },
        "reference_config": {
            "file": "new_model_for_training_code_inference/config.json",
            "sha256": REFERENCE_CONFIG_SHA256,
            "frozen_environment": _file_identity(PHASE2_ENVIRONMENT),
        },
        "motion_data": synthetic_motion_source_declaration(),
        "trace": {
            "seed": SEED,
            "num_envs": NUM_ENVS,
            "steps": TRACE_STEPS,
            "configured_horizon_seconds": TRACE_HORIZON_SECONDS,
            "configured_horizon_steps": 2,
            "timeout_action_index_zero_based": 2,
            "autoreset_mode": "same_step",
            "returned_done_observation": "post_reset",
            "final_done_observation": "exact_pre_reset_capture",
            "action_equation": "processed = clip(5 * behavior, -5, 5)",
        },
        "trace_closure_members": list(TRACE_CLOSURE_MEMBERS),
    }


def inspect_readiness(*, bfm_repo: Path, reference_config: Path, motion_source_path: Path) -> dict[str, Any]:
    """Inspect transient local prerequisites without changing or launching the simulator."""
    repository = Path(bfm_repo).resolve()
    reference = Path(reference_config).resolve()
    motion_source = Path(motion_source_path).resolve()
    blockers: list[str] = []
    observed_sources: dict[str, str] = {}

    if not repository.is_dir():
        blockers.append(f"BFM repository does not exist: {repository}")
    else:
        for relative, expected in NATIVE_SOURCE_HASHES.items():
            path = repository / relative
            if not path.is_file():
                blockers.append(f"Native source is missing: {relative}")
                continue
            observed = _sha256(path)
            observed_sources[relative] = observed
            if observed != expected:
                blockers.append(f"Native source hash differs: {relative}")
        adapter = repository / "phase2_adapter/environment.py"
        if not adapter.is_file():
            blockers.append("Exact-final adapter is missing: phase2_adapter/environment.py")
        elif _sha256(adapter) != FINAL_CAPTURE_ADAPTER_SHA256:
            blockers.append("Exact-final adapter bytes are not the frozen f049 Phase 2 adapter")

    observed_reference_hash = _sha256(reference) if reference.is_file() else None
    if observed_reference_hash is None:
        blockers.append(f"Reference config does not exist: {reference}")
    elif observed_reference_hash != REFERENCE_CONFIG_SHA256:
        blockers.append("Reference config hash differs from the frozen Phase 2 config")
    else:
        frozen_environment = _load_json(PHASE2_ENVIRONMENT)["environment"]
        if _load_json(reference).get("env") != frozen_environment:
            blockers.append("Reference config environment differs from the frozen environment object")

    synthetic_source = inspect_synthetic_motion_source(motion_source)
    if not synthetic_source["exact_recipe_match"]:
        blockers.append("Motion source is not the exact Phase 3 procedural G1 recipe")

    return {
        "schema": "forward_backward_phase3a_g1_lafan_trace_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "bfm_repo": str(repository),
            "reference_config": str(reference),
            "motion_source_path": str(motion_source),
        },
        "observed": {
            "repository_revision": _git_revision(repository) if repository.is_dir() else None,
            "source_hashes": observed_sources,
            "reference_config_sha256": observed_reference_hash,
            "motion_source": synthetic_source,
        },
        "ready": not blockers,
        "blockers": blockers,
    }


class _FinalBodyCapture:
    """Capture native reached state before the Phase 2 adapter resets done rows."""

    def __init__(self, base_env: Any) -> None:
        import torch

        self.base_env = base_env
        self.valid = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
        self.position = torch.zeros_like(base_env._rigid_body_pos_extend)
        self.rotation = torch.zeros_like(base_env._rigid_body_rot_extend)
        self.linear_velocity = torch.zeros_like(base_env._rigid_body_vel_extend)
        self.angular_velocity = torch.zeros_like(base_env._rigid_body_ang_vel_extend)
        self.contact_force = torch.zeros_like(base_env.simulator.contact_forces)
        self._original_reset = base_env.reset_envs_idx

        def reset_with_capture(_instance: Any, env_ids: Any, *args: Any, **kwargs: Any) -> Any:
            if len(env_ids) > 0:
                self._capture(env_ids)
            return self._original_reset(env_ids, *args, **kwargs)

        base_env.reset_envs_idx = types.MethodType(reset_with_capture, base_env)

    def begin_step(self) -> None:
        self.valid.zero_()

    def close(self) -> None:
        self.base_env.reset_envs_idx = self._original_reset

    def _capture(self, env_ids: Any) -> None:
        self.position.index_copy_(0, env_ids, self.base_env._rigid_body_pos_extend[env_ids])
        self.rotation.index_copy_(0, env_ids, self.base_env._rigid_body_rot_extend[env_ids])
        self.linear_velocity.index_copy_(0, env_ids, self.base_env._rigid_body_vel_extend[env_ids])
        self.angular_velocity.index_copy_(0, env_ids, self.base_env._rigid_body_ang_vel_extend[env_ids])
        self.contact_force.index_copy_(0, env_ids, self.base_env.simulator.contact_forces[env_ids])
        self.valid[env_ids] = True


def _numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _append_observation(rows: dict[str, list[np.ndarray]], prefix: str, observation: Any) -> None:
    for name in OBSERVATION_FIELDS:
        rows[f"{prefix}_{name}"].append(_numpy(observation[name]))


def _append_body(rows: dict[str, list[np.ndarray]], prefix: str, base_env: Any) -> None:
    rows[f"{prefix}_body_position"].append(_numpy(base_env._rigid_body_pos_extend))
    rows[f"{prefix}_body_rotation_xyzw"].append(_numpy(base_env._rigid_body_rot_extend))
    rows[f"{prefix}_body_linear_velocity"].append(_numpy(base_env._rigid_body_vel_extend))
    rows[f"{prefix}_body_angular_velocity"].append(_numpy(base_env._rigid_body_ang_vel_extend))


def _append_physics_facts(rows: dict[str, list[np.ndarray]], base_env: Any) -> None:
    """Append source-ordered randomized rigid-body facts for the current edge."""
    num_envs = base_env.num_envs
    num_bodies = base_env.simulator.num_bodies
    rows["current_body_mass"].append(_numpy(base_env.simulator._rigid_body_dr_masses))
    rows["current_body_inertia"].append(
        _numpy(
            base_env.simulator._robot.root_physx_view.get_inertias().to(base_env.device)[:, base_env.simulator.body_ids]
        )
    )
    rows["current_shape_material"].append(
        _numpy(base_env.simulator._robot.root_physx_view.get_material_properties().to(base_env.device))
    )
    rows["current_body_com_pose_xyzw"].append(
        _numpy(base_env.simulator._rigid_body_dr_coms.reshape(num_envs, num_bodies, 7))
    )
    rows["current_contact_force"].append(_numpy(base_env.simulator.contact_forces))
    view = base_env.simulator._robot.root_physx_view
    joint_ids = base_env.simulator.dof_ids
    for name, value in (
        ("stiffness", view.get_dof_stiffnesses()),
        ("damping", view.get_dof_dampings()),
        ("armature", view.get_dof_armatures()),
        ("friction", view.get_dof_friction_coefficients()),
        ("effort_limit", view.get_dof_max_forces()),
        ("position_limit", view.get_dof_limits()),
        ("velocity_limit", view.get_dof_max_velocities()),
    ):
        rows[f"current_joint_{name}"].append(_numpy(value.to(base_env.device)[:, joint_ids]))


def capture(
    *, bfm_repo: Path, reference_config: Path, motion_source_path: Path, device: str
) -> tuple[dict[str, np.ndarray], tuple[str, ...], int]:
    """Run the exact native G1 trace, including pre-reset body and observation capture."""
    import torch

    repository = Path(bfm_repo).resolve()
    source_inspection = inspect_synthetic_motion_source(motion_source_path)
    if not source_inspection["exact_recipe_match"]:
        raise ValueError("Native G1 trace input must exactly match the procedural synthetic source.")
    sys.path.insert(0, str(repository))
    importlib.invalidate_caches()
    try:
        from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
        from humanoidverse.agents.utils import set_seed_everywhere
        from phase2_adapter.environment import BFMZeroVecEnv
    finally:
        sys.path.remove(str(repository))

    set_seed_everywhere(SEED)
    torch.cuda.set_device(torch.device(device))
    config = _load_json(Path(reference_config))
    env_options = copy.deepcopy(config["env"])
    env_options["device"] = device
    env_options["lafan_tail_path"] = str(Path(motion_source_path).resolve())
    env_options["max_episode_length_s"] = TRACE_HORIZON_SECONDS
    native = HumanoidVerseIsaacConfig(**env_options).build(NUM_ENVS)[0]
    environment = BFMZeroVecEnv(native, terminal_profile="correct_terminal", device=device)
    base_env = native._env
    if base_env.config.domain_rand.randomize_ctrl_delay:
        raise ValueError("The frozen G1 trace requires disabled control-delay randomization.")
    physical_body_names = tuple(base_env.simulator.body_names)
    physical_shape_count = int(base_env.simulator._robot.root_physx_view.max_shapes)
    final_body = _FinalBodyCapture(base_env)
    substep_values: dict[str, list[np.ndarray]] = {name: [] for name in SUBSTEP_FIELDS}
    original_simulate = base_env.simulator.simulate_at_each_physics_step

    def simulate_with_capture(_instance: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        result = original_simulate(*call_args, **call_kwargs)
        qpos, qvel = native._get_qpos_qvel(to_numpy=False)
        values = {
            "qpos": qpos,
            "qvel": qvel,
            "body_position": base_env._rigid_body_pos_extend,
            "body_rotation_xyzw": base_env._rigid_body_rot_extend,
            "body_linear_velocity": base_env._rigid_body_vel_extend,
            "body_angular_velocity": base_env._rigid_body_ang_vel_extend,
            "contact_force": base_env.simulator.contact_forces,
            "applied_pd_torque": base_env.torques,
        }
        for name, value in values.items():
            substep_values[name].append(_numpy(value))
        return result

    base_env.simulator.simulate_at_each_physics_step = types.MethodType(simulate_with_capture, base_env.simulator)

    evidence = _load_json(EVIDENCE_CHANNELS)["profiles"]["g1_lafan_50hz"]
    environment_names = tuple(evidence["environment_raw_evidence"])
    auxiliary_names = tuple(evidence["learner_raw_evidence"])
    actions = fixed_actions()
    rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            *(f"{prefix}_{field}" for prefix in ("current", "returned", "final") for field in OBSERVATION_FIELDS),
            *(
                f"{prefix}_body_{field}"
                for prefix in ("current", "returned", "final")
                for field in ("position", "rotation_xyzw", "linear_velocity", "angular_velocity")
            ),
            *(f"substep_{name}" for name in SUBSTEP_FIELDS),
            "current_qpos",
            "current_qvel",
            "returned_qpos",
            "returned_qvel",
            "final_qpos",
            "final_qvel",
            "processed_action",
            "controller_target_joint_position",
            "estimated_pd_torque",
            "current_default_joint_offset",
            "environment_reward",
            "current_body_mass",
            "current_body_inertia",
            "current_shape_material",
            "current_body_com_pose_xyzw",
            "current_contact_force",
            "current_joint_stiffness",
            "current_joint_damping",
            "current_joint_armature",
            "current_joint_friction",
            "current_joint_effort_limit",
            "current_joint_velocity_limit",
            "current_joint_position_limit",
            "returned_contact_force",
            "final_contact_force",
            "environment_reward_recomposed",
            "environment_raw_evidence",
            "learner_auxiliary_reward",
            "learner_auxiliary_raw_evidence",
            "penalty_scale",
            "terminated",
            "truncated",
            "final_observation_valid",
            "action_applied",
            "current_motion_id",
            "current_episode_step",
            "current_reference_time_seconds",
            "target_reference_time_seconds",
            "returned_motion_id",
            "returned_episode_step",
            "returned_reference_time_seconds",
        )
    }

    try:
        for step in range(TRACE_STEPS):
            final_body.begin_step()
            for values in substep_values.values():
                values.clear()
            current = environment.get_observations()
            current_qpos, current_qvel = native._get_qpos_qvel(to_numpy=False)
            current_episode_step = base_env.episode_length_buf.clone()
            current_motion_id = base_env.motion_ids.clone()
            current_motion_start = base_env.motion_start_times.clone()
            default_offset = base_env.default_dof_pos_offset.clone()
            penalty_scale = float(base_env.reward_penalty_scale)

            behavior = torch.from_numpy(actions[step]).to(device=device)
            processed = torch.clamp(behavior * 5.0, -5.0, 5.0)
            effort_limit = torch.as_tensor(base_env.config.robot.dof_effort_limit_list, device=device)
            default_position = base_env.default_dof_pos
            target_position = (
                default_position
                + default_offset
                + processed * base_env.config.robot.control.action_scale * effort_limit / base_env.p_gains
            )

            _append_observation(rows, "current", current)
            _append_body(rows, "current", base_env)
            rows["current_qpos"].append(_numpy(current_qpos))
            rows["current_qvel"].append(_numpy(current_qvel))
            _append_physics_facts(rows, base_env)
            rows["current_default_joint_offset"].append(_numpy(default_offset))
            rows["processed_action"].append(_numpy(processed))
            rows["controller_target_joint_position"].append(_numpy(target_position))
            rows["current_motion_id"].append(_numpy(current_motion_id))
            rows["current_episode_step"].append(_numpy(current_episode_step))
            rows["current_reference_time_seconds"].append(
                _numpy(current_motion_start + current_episode_step * CONTROL_DT_SECONDS)
            )
            rows["target_reference_time_seconds"].append(
                _numpy(current_motion_start + (current_episode_step + 1) * CONTROL_DT_SECONDS)
            )

            returned, reward, done, extras = environment.step(behavior)
            if any(len(values) != 4 for values in substep_values.values()):
                raise RuntimeError("Native G1 trace must capture exactly four physics substeps per action.")
            for name, values in substep_values.items():
                rows[f"substep_{name}"].append(np.stack(values, axis=1))
            returned_qpos, returned_qvel = native._get_qpos_qvel(to_numpy=False)
            truncated = extras["time_outs"].bool()
            terminated = done.bool() & ~truncated
            final_valid = extras["final_obs_valid"].bool()
            mask = final_valid[:, None]

            _append_observation(rows, "returned", returned)
            _append_body(rows, "returned", base_env)
            rows["returned_qpos"].append(_numpy(returned_qpos))
            rows["returned_qvel"].append(_numpy(returned_qvel))
            rows["returned_contact_force"].append(_numpy(base_env.simulator.contact_forces))
            for name in OBSERVATION_FIELDS:
                value = torch.where(mask, extras["final_obs"][name], torch.zeros_like(extras["final_obs"][name]))
                rows[f"final_{name}"].append(_numpy(value))
            rows["final_qpos"].append(
                _numpy(torch.where(mask, extras["final_qpos"], torch.zeros_like(extras["final_qpos"])))
            )
            rows["final_qvel"].append(
                _numpy(torch.where(mask, extras["final_qvel"], torch.zeros_like(extras["final_qvel"])))
            )
            body_mask = final_body.valid[:, None, None]
            rows["final_body_position"].append(
                _numpy(torch.where(body_mask, final_body.position, torch.zeros_like(final_body.position)))
            )
            rows["final_body_rotation_xyzw"].append(
                _numpy(torch.where(body_mask, final_body.rotation, torch.zeros_like(final_body.rotation)))
            )
            rows["final_body_linear_velocity"].append(
                _numpy(torch.where(body_mask, final_body.linear_velocity, torch.zeros_like(final_body.linear_velocity)))
            )
            rows["final_body_angular_velocity"].append(
                _numpy(
                    torch.where(body_mask, final_body.angular_velocity, torch.zeros_like(final_body.angular_velocity))
                )
            )

            raw = base_env.extras["aux_rewards"]
            contact_mask = final_body.valid[:, None, None]
            rows["final_contact_force"].append(
                _numpy(torch.where(contact_mask, final_body.contact_force, torch.zeros_like(final_body.contact_force)))
            )
            raw_environment = np.stack([_numpy(raw[name]) for name in environment_names], axis=-1)
            raw_auxiliary = np.stack([_numpy(raw[name]) for name in auxiliary_names], axis=-1)
            recomposed = np.asarray(
                [
                    compose_environment_reward(dict(zip(environment_names, values, strict=True)), penalty_scale)
                    for values in raw_environment
                ],
                dtype=np.float32,
            )
            auxiliary_reward = np.asarray(
                [
                    compose_learner_auxiliary_reward(dict(zip(auxiliary_names, values, strict=True)))
                    for values in raw_auxiliary
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(_numpy(reward), recomposed, rtol=1.0e-5, atol=1.0e-5)

            rows["estimated_pd_torque"].append(_numpy(base_env.torques))
            rows["environment_reward"].append(_numpy(reward))
            rows["environment_reward_recomposed"].append(recomposed)
            rows["environment_raw_evidence"].append(raw_environment)
            rows["learner_auxiliary_reward"].append(auxiliary_reward)
            rows["learner_auxiliary_raw_evidence"].append(raw_auxiliary)
            rows["penalty_scale"].append(np.full(NUM_ENVS, penalty_scale, dtype=np.float32))
            rows["terminated"].append(_numpy(terminated))
            rows["truncated"].append(_numpy(truncated))
            rows["final_observation_valid"].append(_numpy(final_valid))
            rows["action_applied"].append(np.ones(NUM_ENVS, dtype=np.bool_))
            returned_episode_step = base_env.episode_length_buf.clone()
            returned_motion_start = base_env.motion_start_times.clone()
            rows["returned_motion_id"].append(_numpy(base_env.motion_ids))
            rows["returned_episode_step"].append(_numpy(returned_episode_step))
            rows["returned_reference_time_seconds"].append(
                _numpy(returned_motion_start + (returned_episode_step + 1) * CONTROL_DT_SECONDS)
            )
    finally:
        base_env.simulator.simulate_at_each_physics_step = original_simulate
        final_body.close()
        environment.close()

    tensors = {name: np.stack(values) for name, values in rows.items()}
    tensors["behavior_action"] = actions
    expected_timeout = np.zeros((TRACE_STEPS, NUM_ENVS), dtype=np.bool_)
    expected_timeout[2] = True
    np.testing.assert_array_equal(tensors["truncated"], expected_timeout)
    np.testing.assert_array_equal(tensors["final_observation_valid"], expected_timeout)
    return tensors, physical_body_names, physical_shape_count


def _write_trace_closure(output_root: Path) -> Path:
    members: dict[str, dict[str, int | str]] = {}
    for name in TRACE_CLOSURE_MEMBERS:
        path = output_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Trace closure member is missing: {path}")
        members[name] = _file_identity(path)
    path = output_root / "native_trace_manifest_v1.json"
    _write_json(path, {"schema": "forward_backward_phase3_native_trace_manifest_v1", "members": members})
    return path


def capture_to_disk(
    *, bfm_repo: Path, reference_config: Path, motion_source_path: Path, output_root: Path, device: str
) -> None:
    """Capture the trace and close it together with the existing Meta native trace."""
    output_root.mkdir(parents=True, exist_ok=True)
    tensors, physical_body_names, physical_shape_count = capture(
        bfm_repo=bfm_repo, reference_config=reference_config, motion_source_path=motion_source_path, device=device
    )
    tensor_path = output_root / "g1_lafan_same_step_trace_v1.npz"
    temporary = tensor_path.with_name(tensor_path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **tensors)
    temporary.replace(tensor_path)

    metadata = {
        "schema": "forward_backward_phase3_g1_lafan_trace_v1",
        "profile": "g1_lafan_50hz",
        "declaration": capture_declaration(),
        "source": {
            "repository_revision": _git_revision(Path(bfm_repo)),
            "motion_source": inspect_synthetic_motion_source(motion_source_path),
            "source_files": {name: _sha256(Path(bfm_repo) / name) for name in NATIVE_SOURCE_HASHES},
            "exact_final_capture_adapter_sha256": _sha256(Path(bfm_repo) / "phase2_adapter/environment.py"),
            "trace_generator": _file_identity(Path(__file__)),
            "physical_body_names": list(physical_body_names),
            "physical_shape_count": physical_shape_count,
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "device": device},
        "evidence_order": {
            "environment": list(_load_json(EVIDENCE_CHANNELS)["profiles"]["g1_lafan_50hz"]["environment_raw_evidence"]),
            "learner_auxiliary": list(
                _load_json(EVIDENCE_CHANNELS)["profiles"]["g1_lafan_50hz"]["learner_raw_evidence"]
            ),
        },
        "trace": {
            "file": tensor_path.name,
            "file_sha256": _sha256(tensor_path),
            "tensors": {name: _tensor_summary(value) for name, value in sorted(tensors.items())},
        },
    }
    _write_json(output_root / "g1_lafan_same_step_trace_v1.json", metadata)
    _write_trace_closure(output_root)


def main() -> None:
    """Generate the synthetic source, inspect readiness, or launch native capture."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    source_parser = subparsers.add_parser("generate_source")
    source_parser.add_argument("--output_path", type=Path, required=True)
    for command in ("prepare", "capture"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--bfm_repo", type=Path, required=True)
        subparser.add_argument("--reference_config", type=Path, required=True)
        subparser.add_argument("--motion_source_path", type=Path, required=True)
        subparser.add_argument("--output_root", type=Path, required=True)
        if command == "capture":
            subparser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "generate_source":
        identity = write_synthetic_motion_source(args.output_path)
        print(json.dumps({"path": str(args.output_path.resolve()), **identity}, sort_keys=True))
        return

    readiness = inspect_readiness(
        bfm_repo=args.bfm_repo,
        reference_config=args.reference_config,
        motion_source_path=args.motion_source_path,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "g1_lafan_trace_declaration_v1.json", capture_declaration())
    _write_json(args.output_root / "g1_lafan_trace_readiness.json", readiness)
    if args.command == "capture":
        if not readiness["ready"]:
            raise RuntimeError("G1 trace capture is not ready: " + "; ".join(readiness["blockers"]))
        capture_to_disk(
            bfm_repo=args.bfm_repo,
            reference_config=args.reference_config,
            motion_source_path=args.motion_source_path,
            output_root=args.output_root,
            device=args.device,
        )


if __name__ == "__main__":
    main()
