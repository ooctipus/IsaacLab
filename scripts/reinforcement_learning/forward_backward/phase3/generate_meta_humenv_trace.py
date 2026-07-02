# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture the frozen HumEnv control and NEXT_STEP boundary trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import gymnasium
import mujoco
import numpy as np
from gymnasium.wrappers import TimeAwareObservation
from humenv import __version__ as humenv_version
from humenv import make_humenv

SEED = 20_260_629
NUM_ENVS = 2
ACTION_DIM = 69
OBSERVATION_DIM = 358
QPOS_DIM = 76
QVEL_DIM = 75
MAX_EPISODE_STEPS = 3
TRACE_STEPS = 5
SOURCE_FILES = (
    "humenv/__init__.py",
    "humenv/env.py",
    "humenv/reset.py",
    "humenv/assets/robot.xml",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_summary(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _git_revision(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _verify_executed_source(repository: Path) -> dict[str, str]:
    import humenv

    installed_root = Path(humenv.__file__).resolve().parent.parent
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        repository_file = repository / relative
        installed_file = installed_root / relative
        if not repository_file.is_file() or not installed_file.is_file():
            raise FileNotFoundError(f"HumEnv source file is missing: {relative}.")
        repository_hash = _sha256(repository_file)
        if _sha256(installed_file) != repository_hash:
            raise ValueError(f"Executed HumEnv bytes differ from repository source: {relative}.")
        hashes[relative] = repository_hash
    return hashes


def _fixed_actions() -> np.ndarray:
    action_index = np.arange(ACTION_DIM, dtype=np.float64)[None, None, :]
    step = np.arange(TRACE_STEPS, dtype=np.float64)[:, None, None]
    env_phase = np.arange(NUM_ENVS, dtype=np.float64)[None, :, None] * 0.13
    return 0.05 * np.sin(action_index * 0.17 + step * 0.31 + env_phase)


def _body_state(env: gymnasium.vector.SyncVectorEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = []
    rotations = []
    linear_velocities = []
    angular_velocities = []
    for wrapped in env.envs:
        native = wrapped.unwrapped
        mujoco.mj_kinematics(native.model, native.data)
        positions.append(native.data.xpos[1:25].copy())
        rotations.append(native.data.xquat[1:25].copy())
        linear_velocities.append(native.data.sensordata[:72].reshape(24, 3).copy())
        angular_velocities.append(native.data.sensordata[72:144].reshape(24, 3).copy())
    values = (positions, rotations, linear_velocities, angular_velocities)
    return tuple(np.stack(value) for value in values)


def _edge_state(env: gymnasium.vector.SyncVectorEnv) -> dict[str, np.ndarray]:
    """Capture every mutable MuJoCo input that can affect the next source edge."""
    names = ("control", "qacc_warmstart", "qfrc_applied", "xfrc_applied")
    values: dict[str, list[np.ndarray]] = {name: [] for name in names}
    simulation_time = []
    for wrapped in env.envs:
        data = wrapped.unwrapped.data
        values["control"].append(data.ctrl.copy())
        values["qacc_warmstart"].append(data.qacc_warmstart.copy())
        values["qfrc_applied"].append(data.qfrc_applied.copy())
        values["xfrc_applied"].append(data.xfrc_applied.copy())
        simulation_time.append(np.asarray(data.time, dtype=np.float64))
    output = {name: np.stack(rows) for name, rows in values.items()}
    output["simulation_time_seconds"] = np.stack(simulation_time)
    return output


def _model_state(env: gymnasium.vector.SyncVectorEnv) -> dict[str, np.ndarray]:
    """Capture fixed MuJoCo physics facts required to reproduce one edge."""
    model = env.envs[0].unwrapped.model
    names = (
        "body_parentid",
        "body_pos",
        "body_quat",
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "jnt_type",
        "jnt_bodyid",
        "jnt_pos",
        "jnt_axis",
        "dof_armature",
        "dof_damping",
        "jnt_stiffness",
        "qpos_spring",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_gear",
        "actuator_trnid",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "geom_bodyid",
        "geom_type",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_contype",
        "geom_conaffinity",
        "geom_condim",
        "geom_friction",
        "geom_margin",
        "geom_gap",
        "geom_priority",
        "geom_solmix",
        "geom_solimp",
        "geom_solref",
    )
    values = {f"model_{name}": np.asarray(getattr(model, name)).copy() for name in names}
    option_names = (
        "solver",
        "integrator",
        "cone",
        "jacobian",
        "iterations",
        "ls_iterations",
        "noslip_iterations",
        "sdf_iterations",
        "tolerance",
        "ls_tolerance",
        "noslip_tolerance",
        "sdf_initpoints",
        "impratio",
        "disableflags",
        "enableflags",
        "gravity",
        "timestep",
    )
    values.update({f"model_option_{name}": np.asarray(getattr(model.opt, name)).copy() for name in option_names})
    values["model_option_is_sparse"] = np.asarray(mujoco.mj_isSparse(model))
    return values


def capture(*, include_edge_state: bool = False) -> dict[str, np.ndarray]:
    """Run the exact deterministic native trace and return its tensors."""
    np.random.seed(SEED)
    env, _ = make_humenv(
        num_envs=NUM_ENVS,
        vectorization_mode="sync",
        wrappers=[
            gymnasium.wrappers.FlattenObservation,
            lambda item: TimeAwareObservation(item, flatten=False),
        ],
        max_episode_steps=MAX_EPISODE_STEPS,
        state_init="Default",
        fall_prob=0.2,
        render_mode=None,
        seed=SEED,
    )
    if str(env.autoreset_mode) != "AutoresetMode.NEXT_STEP":
        raise ValueError(f"HumEnv autoreset mode differs: {env.autoreset_mode}.")

    actions = _fixed_actions()
    observation, info = env.reset(seed=SEED)
    current_observation = []
    current_qpos = []
    current_qvel = []
    current_time = []
    current_body_pos = []
    current_body_quat = []
    current_body_lin_vel = []
    current_body_ang_vel = []
    returned_observation = []
    returned_qpos = []
    returned_qvel = []
    returned_time = []
    returned_body_pos = []
    returned_body_quat = []
    returned_body_lin_vel = []
    returned_body_ang_vel = []
    model_state = _model_state(env) if include_edge_state else {}
    edge_state = {
        f"{point}_{name}": []
        for point in ("current", "returned")
        for name in ("control", "qacc_warmstart", "qfrc_applied", "xfrc_applied", "simulation_time_seconds")
    }
    rewards = []
    terminated_rows = []
    truncated_rows = []
    action_applied = []
    previous_done = np.zeros(NUM_ENVS, dtype=np.bool_)
    try:
        for trace_step in range(TRACE_STEPS):
            current_observation.append(np.asarray(observation["obs"]).copy())
            current_qpos.append(np.asarray(info["qpos"]).copy())
            current_qvel.append(np.asarray(info["qvel"]).copy())
            current_time.append(np.asarray(observation["time"]).copy())
            body_state = _body_state(env)
            for output, value in zip(
                (current_body_pos, current_body_quat, current_body_lin_vel, current_body_ang_vel),
                body_state,
                strict=True,
            ):
                output.append(value)
            if include_edge_state:
                for name, value in _edge_state(env).items():
                    edge_state[f"current_{name}"].append(value)

            next_observation, reward, terminated, truncated, next_info = env.step(actions[trace_step])
            terminated = np.asarray(terminated, dtype=np.bool_)
            truncated = np.asarray(truncated, dtype=np.bool_)
            returned_observation.append(np.asarray(next_observation["obs"]).copy())
            returned_qpos.append(np.asarray(next_info["qpos"]).copy())
            returned_qvel.append(np.asarray(next_info["qvel"]).copy())
            returned_time.append(np.asarray(next_observation["time"]).copy())
            body_state = _body_state(env)
            for output, value in zip(
                (returned_body_pos, returned_body_quat, returned_body_lin_vel, returned_body_ang_vel),
                body_state,
                strict=True,
            ):
                output.append(value)
            if include_edge_state:
                for name, value in _edge_state(env).items():
                    edge_state[f"returned_{name}"].append(value)
            rewards.append(np.asarray(reward, dtype=np.float64).copy())
            terminated_rows.append(terminated)
            truncated_rows.append(truncated)
            action_applied.append(~previous_done)
            previous_done = terminated | truncated
            observation = next_observation
            info = next_info
    finally:
        env.close()

    tensors = {
        "actions": actions,
        "current_observation": np.stack(current_observation),
        "current_qpos": np.stack(current_qpos),
        "current_qvel": np.stack(current_qvel),
        "current_time": np.stack(current_time),
        "current_body_pos": np.stack(current_body_pos),
        "current_body_quat": np.stack(current_body_quat),
        "current_body_lin_vel": np.stack(current_body_lin_vel),
        "current_body_ang_vel": np.stack(current_body_ang_vel),
        "returned_observation": np.stack(returned_observation),
        "returned_qpos": np.stack(returned_qpos),
        "returned_qvel": np.stack(returned_qvel),
        "returned_time": np.stack(returned_time),
        "returned_body_pos": np.stack(returned_body_pos),
        "returned_body_quat": np.stack(returned_body_quat),
        "returned_body_lin_vel": np.stack(returned_body_lin_vel),
        "returned_body_ang_vel": np.stack(returned_body_ang_vel),
        "reward": np.stack(rewards),
        "terminated": np.stack(terminated_rows),
        "truncated": np.stack(truncated_rows),
        "action_applied": np.stack(action_applied),
    }
    if include_edge_state:
        tensors.update({name: np.stack(rows) for name, rows in edge_state.items()})
        tensors.update(model_state)
    return tensors


def main() -> None:
    """Write the trace tensors and their exact source/runtime provenance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--humenv_repo", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--include_edge_state", action="store_true")
    args = parser.parse_args()
    repository = args.humenv_repo.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_hashes = _verify_executed_source(repository)
    tensors = capture(include_edge_state=args.include_edge_state)
    version = "v2" if args.include_edge_state else "v1"
    tensor_path = output_root / f"meta_humenv_next_step_trace_{version}.npz"
    temporary = tensor_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **tensors)
    temporary.replace(tensor_path)

    metadata = {
        "schema": f"forward_backward_phase3_meta_humenv_trace_{version}",
        "profile": "smpl_humenv_30hz",
        "source": {
            "repository": "humenv",
            "revision": _git_revision(repository),
            "license": "CC-BY-NC-4.0",
            "files": source_hashes,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "humenv": humenv_version,
        },
        "mdp": {
            "physics_dt_seconds": 1.0 / 450.0,
            "control_decimation": 15,
            "control_dt_seconds": 1.0 / 30.0,
            "action_dim": ACTION_DIM,
            "observation_dim": OBSERVATION_DIM,
            "qpos_dim": QPOS_DIM,
            "qvel_dim": QVEL_DIM,
            "autoreset_mode": "next_step",
            "termination": "never",
            "trace_horizon_steps": MAX_EPISODE_STEPS,
            "production_horizon_steps": 300,
            "reset_profile": "default_t_pose_trace_only",
            "reward": "zero",
        },
        "trace": {
            "seed": SEED,
            "num_envs": NUM_ENVS,
            "steps": TRACE_STEPS,
            "timeout_step": 2,
            "reset_only_step": 3,
            "tensors": {name: _tensor_summary(value) for name, value in sorted(tensors.items())},
            "file": tensor_path.name,
            "file_sha256": _sha256(tensor_path),
        },
    }
    if args.include_edge_state:
        metadata["generator_sha256"] = _sha256(Path(__file__))
        metadata["edge_state"] = {
            "mutable_next_step_inputs": [
                "qpos",
                "qvel",
                "control",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
                "simulation_time_seconds",
            ],
            "fixed_model_facts": sorted(name for name in tensors if name.startswith("model_")),
        }
    metadata_path = output_root / f"meta_humenv_next_step_trace_{version}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
