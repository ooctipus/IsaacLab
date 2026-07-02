# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the GPU-native frozen BFM broad-reward boundary."""

from __future__ import annotations

import builtins
import importlib.util
import inspect
import io
import json
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent
RUNTIME = ROOT / "bfm_reward_runtime.py"
PRODUCER = ROOT / "generate_bfm_reward_gpu_runtime.py"
CUDA_EVIDENCE = ROOT / "fixtures/runtime/bfm_reward_gpu_runtime_v1.json"
BFM_SOURCE_IDENTITY = ROOT / "fixtures/bfm_reward_source_identity_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("bfm_reward_runtime", RUNTIME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_root() -> Path:
    root = Path.cwd().parents[1] / "BFM-Zero"
    if not root.is_dir():
        pytest.skip("The frozen BFM-Zero reference checkout is unavailable.")
    return root


def _numpy_sigmoid(value: np.ndarray, value_at_margin: float, kind: str) -> np.ndarray:
    if kind == "gaussian":
        scale = np.sqrt(-2.0 * np.log(value_at_margin))
        return np.exp(-0.5 * np.square(value * scale))
    if kind == "linear":
        scaled = value * (1.0 - value_at_margin)
        return np.where(np.abs(scaled) < 1.0, 1.0 - scaled, 0.0)
    if kind == "quadratic":
        scaled = value * np.sqrt(1.0 - value_at_margin)
        return np.where(np.abs(scaled) < 1.0, 1.0 - np.square(scaled), 0.0)
    raise ValueError(kind)


def _numpy_tolerance(
    value,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = 0.1,
):
    lower, upper = bounds
    values = np.asarray(value)
    in_bounds = (lower <= values) & (values <= upper)
    if margin == 0.0:
        result = np.where(in_bounds, 1.0, 0.0)
    else:
        distance = np.where(values < lower, lower - values, values - upper) / margin
        result = np.where(in_bounds, 1.0, _numpy_sigmoid(distance, value_at_margin, sigmoid))
    return float(result) if np.isscalar(value) else result


def _load_frozen_reward_module(path: Path):
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
    spec = importlib.util.spec_from_file_location("_bfm_reward_parity_oracle", path)
    assert spec is not None and spec.loader is not None
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


def _make_frozen_reward(module, task: str):
    for _name, reward_type in inspect.getmembers(module, inspect.isclass):
        if not inspect.isabstract(reward_type) and hasattr(reward_type, "reward_from_name"):
            with redirect_stdout(io.StringIO()):
                reward = reward_type.reward_from_name(task)
            if reward is not None:
                return reward
    raise AssertionError(f"Frozen task did not resolve: {task}")


@pytest.mark.parametrize("kind", ("gaussian", "linear", "quadratic"))
def test_tolerance_matches_every_sigmoid_used_by_frozen_bfm(kind: str) -> None:
    """The tensor soft interval must retain dm_control's margin parameterization."""
    module = _module()
    values = torch.tensor((-1.0, -0.25, 0.0, 0.25, 1.0))
    result = module.tolerance(values, bounds=(-0.25, 0.25), margin=0.75, sigmoid=kind, value_at_margin=0.1)
    torch.testing.assert_close(result[1:4], torch.ones(3))
    assert result[0].item() == pytest.approx(0.1)
    assert result[4].item() == pytest.approx(0.1)
    assert module.tolerance(torch.tensor(2.0), bounds=(0.0, 1.0), margin=0.0).item() == 0.0


def test_gpu_equations_and_newton_features_match_frozen_cpu_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every assigned task must match the released MuJoCo reward on the same rows."""
    import mujoco

    monkeypatch.setattr(builtins, "_isaaclab_tasks_registered", True, raising=False)
    from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics, NewtonKinematicsCfg

    module = _module()
    root = _reference_root()
    model_path = root / "humanoidverse/data/robots/g1/scene_29dof_freebase_noadditional_actuators.xml"
    reward_module = _load_frozen_reward_module(root / "humanoidverse/envs/g1_env_helper/rewards.py")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    task_count = len(module.BFM_REWARD_TASKS)
    qpos = torch.from_numpy(np.repeat(model.qpos0[None], task_count, axis=0)).float()
    qvel = torch.empty(task_count, model.nv)
    joint_coordinate = torch.arange(model.nq - 7, dtype=torch.float32)
    joint_velocity = torch.arange(model.nv - 6, dtype=torch.float32)
    for row in range(task_count):
        phase = 0.17 * row
        quaternion = torch.tensor((1.0, 0.12 * np.sin(phase), -0.08 * np.cos(phase), 0.2 * np.sin(0.5 * phase)))
        qpos[row, 0] = 0.02 * row
        qpos[row, 1] = -0.01 * row
        qpos[row, 2] = 0.15 + 0.035 * row
        qpos[row, 3:7] = quaternion / quaternion.norm()
        qpos[row, 7:] += 0.2 * torch.sin(joint_coordinate * 0.23 + phase)
        qvel[row, :3] = torch.tensor((0.8 * np.sin(phase), 0.7 * np.cos(phase), 0.1 * np.sin(2.0 * phase)))
        qvel[row, 3:6] = torch.tensor((0.4 * np.cos(phase), -0.3 * np.sin(phase), 6.0 * np.sin(0.7 * phase)))
        qvel[row, 6:] = 0.5 * torch.cos(joint_velocity * 0.19 + phase)

    kinematics = NewtonKinematics(NewtonKinematicsCfg(mjcf_path=str(model_path), device="cpu"))
    runtime = module.BfmRewardRuntime(kinematics, tuple(kinematics.joint_q_names[7:]), episodes_per_task=1)
    actual = runtime.evaluate(qpos, qvel)[:, 0]
    action = np.zeros(model.nu)
    expected = torch.tensor(
        [
            _make_frozen_reward(reward_module, task)(model, qpos[index].numpy(), qvel[index].numpy(), action)
            for index, task in enumerate(module.BFM_REWARD_TASKS)
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected, rtol=3.0e-5, atol=3.0e-5)


def test_reward_runtime_rejects_nonsemantic_joint_order() -> None:
    """Storage-order joints must fail at the FK boundary instead of being patched."""
    module = _module()

    class Model:
        joint_coord_count = module.BFM_QPOS_DIM
        joint_dof_count = module.BFM_QVEL_DIM

    kinematics = types.SimpleNamespace(
        joint_q_names=[*(f"root:{index}" for index in range(7)), "joint_a", "joint_b"],
        model=Model(),
    )
    with pytest.raises(ValueError, match="semantic G1 joint order"):
        module.BfmRewardRuntime(kinematics, ("joint_b", "joint_a"), episodes_per_task=1)


def test_reward_metric_rows_transfer_only_final_reductions() -> None:
    """Serialization must consume reduced task/episode tensors, not trajectories."""
    module = _module()
    evidence_count = len(module.BFM_AUXILIARY_EVIDENCE_NAMES)
    evidence_sum = torch.arange(1, evidence_count + 1, dtype=torch.float32).view(1, 1, -1)
    rows = module.reward_metric_rows(
        ("task",),
        torch.tensor(((7.0,),)),
        evidence_sum,
        2.0 * evidence_sum,
        torch.tensor(((6.0,),)),
        torch.tensor(((2.0,),)),
        torch.tensor(((1.0,),)),
        torch.tensor(((8.0,),)),
        step_count=4,
    )
    metrics = {str(row["metric_name"]): row["metric_value"] for row in rows}
    assert metrics["return"] == 7.0
    assert metrics["auxiliary_cost"] == 1.5
    assert metrics["safety_violation_rate"] == 0.5
    assert metrics["termination_rate"] == 0.25
    assert metrics["action_l2"] == 2.0
    assert metrics["penalty_torques_mean"] == 0.25
    assert metrics["penalty_torques_active_fraction"] == 0.5


def test_reward_source_identity_requires_every_frozen_input(tmp_path: Path) -> None:
    """The tensor derivation remains hash-linked to all released source boundaries."""
    module = _module()
    for relative in module._REFERENCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    identity = module.bfm_reward_source_identity(tmp_path)
    assert tuple(identity) == module._REFERENCE_FILES
    (tmp_path / module._REFERENCE_FILES[-1]).unlink()
    with pytest.raises(ValueError, match="regular non-symbolic"):
        module.bfm_reward_source_identity(tmp_path)


def test_cuda_evidence_is_bound_to_runtime_and_meets_the_frozen_gate() -> None:
    """Measured CUDA parity and latency must describe the current runtime bytes."""
    from rsl_rl.modules.forward_backward import reward_context

    from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics

    module = _module()
    evidence = json.loads(CUDA_EVIDENCE.read_text())
    bfm_source_identity = json.loads(BFM_SOURCE_IDENTITY.read_text())
    kinematics_path = Path(inspect.getsourcefile(NewtonKinematics) or "")
    reward_context_path = Path(inspect.getsourcefile(inspect.unwrap(reward_context)) or "")
    assert evidence["status"] == "passed"
    assert evidence["source_identity"] == {
        "producer_sha256": module._sha256(PRODUCER),
        "bfm_reward_runtime_sha256": module._sha256(RUNTIME),
        "newton_kinematics_sha256": module._sha256(kinematics_path),
        "reward_context_source_sha256": module._sha256(reward_context_path),
        "frozen_reward_source_sha256": bfm_source_identity["files"]["humanoidverse/envs/g1_env_helper/rewards.py"],
    }
    assert evidence["correctness"]["frozen_oracle_tasks_passed"] == len(module.BFM_REWARD_TASKS)
    assert evidence["correctness"]["cpu_cuda_max_absolute_difference"] <= 5.0e-7
    assert evidence["benchmark"]["microseconds_per_reward_step"] < 1000.0
    assert evidence["benchmark"]["temporary_peak_bytes"] < 1024 * 1024


def test_runtime_source_has_no_research_or_host_replay_imports() -> None:
    """The executable boundary must be GPU tensor algebra, not research runtime replay."""
    source = RUNTIME.read_text()
    for forbidden in (
        "from humanoidverse",
        "import humanoidverse",
        "from humenv",
        "import humenv",
        "from dm_control",
        "import dm_control",
        "import mujoco",
        "import numpy",
        "ThreadPoolExecutor",
        ".cpu().numpy()",
        "FrozenBfmRewardRuntime",
    ):
        assert forbidden not in source
