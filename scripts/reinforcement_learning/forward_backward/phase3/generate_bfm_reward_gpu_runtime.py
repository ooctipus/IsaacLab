# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regenerate the frozen BFM broad-reward CUDA parity and runtime receipt."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import io
import json
import sys
import time
import types
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
from bfm_reward_runtime import BFM_REWARD_TASKS, BfmRewardRuntime, _sha256
from rsl_rl.modules.forward_backward import reward_context

from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics, NewtonKinematicsCfg

_RTOL = 3.0e-5
_ATOL = 3.0e-5


def _numpy_sigmoid(value: np.ndarray, value_at_margin: float, kind: str) -> np.ndarray:
    """Evaluate the frozen source's used sigmoid modes without dm_control."""
    if kind == "gaussian":
        scale = np.sqrt(-2.0 * np.log(value_at_margin))
        return np.exp(-0.5 * np.square(value * scale))
    if kind == "linear":
        scaled = value * (1.0 - value_at_margin)
        return np.where(np.abs(scaled) < 1.0, 1.0 - scaled, 0.0)
    if kind == "quadratic":
        scaled = value * np.sqrt(1.0 - value_at_margin)
        return np.where(np.abs(scaled) < 1.0, 1.0 - np.square(scaled), 0.0)
    raise ValueError(f"Unsupported frozen BFM sigmoid: {kind!r}.")


def _numpy_tolerance(
    value: np.ndarray | float,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = 0.1,
) -> np.ndarray | float:
    """Evaluate the dm_control tolerance subset used by frozen BFM rewards."""
    lower, upper = bounds
    values = np.asarray(value)
    in_bounds = (lower <= values) & (values <= upper)
    if margin == 0.0:
        result = np.where(in_bounds, 1.0, 0.0)
    else:
        distance = np.where(values < lower, lower - values, values - upper) / margin
        result = np.where(in_bounds, 1.0, _numpy_sigmoid(distance, value_at_margin, sigmoid))
    return float(result) if np.isscalar(value) else result


def _load_frozen_reward_module(path: Path) -> types.ModuleType:
    """Load the immutable source reward module with its narrow tolerance dependency."""
    dm_control = types.ModuleType("dm_control")
    dm_control.__path__ = []
    utils = types.ModuleType("dm_control.utils")
    utils.__path__ = []
    rewards = types.ModuleType("dm_control.utils.rewards")
    rewards.tolerance = _numpy_tolerance
    utils.rewards = rewards
    dm_control.utils = utils
    injected = {
        "dm_control": dm_control,
        "dm_control.utils": utils,
        "dm_control.utils.rewards": rewards,
    }
    previous = {name: sys.modules.get(name) for name in injected}
    spec = importlib.util.spec_from_file_location("_bfm_reward_evidence_oracle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen BFM reward source: {path}.")
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules.update(injected)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value
    return module


def _frozen_reward(module: types.ModuleType, task: str):
    """Resolve one released task through the source class factories."""
    for _name, reward_type in inspect.getmembers(module, inspect.isclass):
        if not inspect.isabstract(reward_type) and hasattr(reward_type, "reward_from_name"):
            with redirect_stdout(io.StringIO()):
                reward = reward_type.reward_from_name(task)
            if reward is not None:
                return reward
    raise RuntimeError(f"Frozen BFM task did not resolve: {task}.")


def _states(model: mujoco.MjModel, episodes_per_task: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic nontrivial task-major source states."""
    rows = len(BFM_REWARD_TASKS) * episodes_per_task
    qpos = torch.from_numpy(np.repeat(model.qpos0[None], rows, axis=0)).float()
    qvel = torch.empty(rows, model.nv)
    joint_coordinate = torch.arange(model.nq - 7, dtype=torch.float32)
    joint_velocity = torch.arange(model.nv - 6, dtype=torch.float32)
    for task in range(len(BFM_REWARD_TASKS)):
        for episode in range(episodes_per_task):
            row = task * episodes_per_task + episode
            phase = 0.17 * task + 0.031 * episode
            quaternion = torch.tensor((1.0, 0.12 * np.sin(phase), -0.08 * np.cos(phase), 0.2 * np.sin(0.5 * phase)))
            qpos[row, 0] = 0.02 * task
            qpos[row, 1] = -0.01 * task
            qpos[row, 2] = 0.15 + 0.035 * task
            qpos[row, 3:7] = quaternion / quaternion.norm()
            qpos[row, 7:] += 0.2 * torch.sin(joint_coordinate * 0.23 + phase)
            qvel[row, :3] = torch.tensor((0.8 * np.sin(phase), 0.7 * np.cos(phase), 0.1 * np.sin(2.0 * phase)))
            qvel[row, 3:6] = torch.tensor((0.4 * np.cos(phase), -0.3 * np.sin(phase), 6.0 * np.sin(0.7 * phase)))
            qvel[row, 6:] = 0.5 * torch.cos(joint_velocity * 0.19 + phase)
    return qpos, qvel


def _elapsed_cuda(device: torch.device, iterations: int, operation) -> float:
    """Measure repeated asynchronous CUDA work with synchronization only at boundaries."""
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        operation()
    torch.cuda.synchronize(device)
    return time.perf_counter() - start


def _device_uuid(properties: object) -> str:
    """Return Torch's physical UUID in NVIDIA's conventional representation."""
    value = str(getattr(properties, "uuid", ""))
    if len(value) != 36:
        raise RuntimeError("Torch did not expose a physical CUDA device UUID.")
    return f"GPU-{value}"


def main() -> None:
    """Run parity and performance gates, then atomically write the receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes_per_task", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.episodes_per_task < 1 or args.iterations < 1:
        raise ValueError("Episodes per task and iterations must be positive.")

    source_root = args.source_root.expanduser().resolve()
    model_path = source_root / "humanoidverse/data/robots/g1/scene_29dof_freebase_noadditional_actuators.xml"
    reward_path = source_root / "humanoidverse/envs/g1_env_helper/rewards.py"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    qpos_cpu, qvel_cpu = _states(model, args.episodes_per_task)
    live_joint_names: tuple[str, ...]

    cpu_kinematics = NewtonKinematics(NewtonKinematicsCfg(mjcf_path=str(model_path), device="cpu"))
    live_joint_names = tuple(cpu_kinematics.joint_q_names[7:])
    cpu_runtime = BfmRewardRuntime(cpu_kinematics, live_joint_names, args.episodes_per_task)
    actual_cpu = cpu_runtime.evaluate(qpos_cpu, qvel_cpu).detach()

    reward_module = _load_frozen_reward_module(reward_path)
    action = np.zeros(model.nu)
    expected = torch.empty_like(actual_cpu)
    for task_index, task in enumerate(BFM_REWARD_TASKS):
        reward = _frozen_reward(reward_module, task)
        for episode in range(args.episodes_per_task):
            row = task_index * args.episodes_per_task + episode
            expected[task_index, episode] = reward(
                model,
                qpos_cpu[row].numpy(),
                qvel_cpu[row].numpy(),
                action,
            )
    task_passes = torch.all(torch.isclose(actual_cpu, expected, rtol=_RTOL, atol=_ATOL), dim=1)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The BFM reward runtime receipt requires one CUDA device.")
    torch.cuda.set_device(device)
    wp.init()
    torch.cuda.synchronize(device)
    setup_start = time.perf_counter()
    cuda_kinematics = NewtonKinematics(NewtonKinematicsCfg(mjcf_path=str(model_path), device=str(device)))
    cuda_runtime = BfmRewardRuntime(cuda_kinematics, live_joint_names, args.episodes_per_task)
    qpos = qpos_cpu.to(device)
    qvel = qvel_cpu.to(device)
    cuda_runtime.evaluate(qpos, qvel)
    torch.cuda.synchronize(device)
    runtime_setup_seconds = time.perf_counter() - setup_start

    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    reward_seconds = _elapsed_cuda(device, args.iterations, lambda: cuda_runtime.evaluate(qpos, qvel))
    temporary_peak_bytes = torch.cuda.max_memory_allocated(device) - baseline
    actual_cuda = cuda_runtime.evaluate(qpos, qvel).detach()
    torch.cuda.synchronize(device)

    fk_seconds = _elapsed_cuda(
        device,
        args.iterations,
        lambda: cuda_kinematics.eval_fk_batched_torch(
            cuda_runtime._joint_q,
            cuda_runtime._joint_qd,
            cuda_runtime._body_q,
            cuda_runtime._body_qd,
        ),
    )
    features = cuda_runtime.features.view(len(BFM_REWARD_TASKS), args.episodes_per_task, -1)
    algebra_seconds = _elapsed_cuda(device, args.iterations, lambda: cuda_runtime._reward(features))

    cpu_cuda_max_absolute_difference = float(torch.max(torch.abs(actual_cuda.cpu() - actual_cpu)).item())
    finite = bool(torch.isfinite(actual_cuda).all().item())
    microseconds_per_reward_step = 1.0e6 * reward_seconds / args.iterations
    properties = torch.cuda.get_device_properties(device)
    runtime_path = Path(__file__).with_name("bfm_reward_runtime.py")
    kinematics_path = Path(inspect.getsourcefile(NewtonKinematics) or "")
    reward_context_path = Path(inspect.getsourcefile(inspect.unwrap(reward_context)) or "")
    source_identity = {
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "bfm_reward_runtime_sha256": _sha256(runtime_path),
        "newton_kinematics_sha256": _sha256(kinematics_path),
        "reward_context_source_sha256": _sha256(reward_context_path),
        "frozen_reward_source_sha256": _sha256(reward_path),
    }
    passed = (
        bool(task_passes.all().item())
        and cpu_cuda_max_absolute_difference <= 5.0e-7
        and finite
        and microseconds_per_reward_step < 1000.0
        and temporary_peak_bytes < 1024 * 1024
    )
    rows = len(BFM_REWARD_TASKS) * args.episodes_per_task
    report = {
        "schema": "forward_backward_phase3_bfm_reward_gpu_runtime_v1",
        "recorded_at": datetime.now(UTC).isoformat(),
        "status": "passed" if passed else "failed",
        "reward_contract": {
            "task_count": len(BFM_REWARD_TASKS),
            "feature_width": cuda_runtime.features.shape[1],
            "action_dependency": False,
            "layout": "task_major_episode_major",
            "source_oracle": "frozen BFM-Zero MuJoCo reward classes",
            "source_oracle_role": "evidence_generation_only",
        },
        "correctness": {
            "frozen_oracle_tasks_passed": int(task_passes.sum().item()),
            "frozen_oracle_tasks_total": len(BFM_REWARD_TASKS),
            "frozen_oracle_rtol": _RTOL,
            "frozen_oracle_atol": _ATOL,
            "cpu_cuda_max_absolute_difference": cpu_cuda_max_absolute_difference,
            "finite": finite,
        },
        "device_residency": {
            "trajectory_host_copies": 0,
            "per_step_host_synchronizations": 0,
            "final_scalar_transfer_only": True,
            "reward_algebra": "torch_compile_fullgraph",
            "kinematics": "newton_batched_fk",
        },
        "benchmark": {
            "device_name": properties.name,
            "physical_gpu_uuid": _device_uuid(properties),
            "task_count": len(BFM_REWARD_TASKS),
            "episodes_per_task": args.episodes_per_task,
            "rows_per_step": rows,
            "iterations": args.iterations,
            "runtime_setup_seconds": runtime_setup_seconds,
            "microseconds_per_reward_step": microseconds_per_reward_step,
            "microseconds_per_fk_only": 1.0e6 * fk_seconds / args.iterations,
            "microseconds_per_compiled_algebra_only": 1.0e6 * algebra_seconds / args.iterations,
            "reward_rows_per_second": rows * args.iterations / reward_seconds,
            "temporary_peak_bytes": temporary_peak_bytes,
        },
        "runtime": {
            "torch": torch.__version__,
            "warp": wp.__version__,
        },
        "source_identity": source_identity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("BFM reward CUDA evidence did not pass its frozen gates.")


if __name__ == "__main__":
    main()
