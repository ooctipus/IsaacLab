# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""External-data regression for the LAFAN-to-G1 arm IK branch."""

from __future__ import annotations

import builtins
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

_BENCHMARK_PATH = Path(__file__).with_name("benchmark_lafan_retargeting.py")


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_lafan_retargeting", _BENCHMARK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _artifact_root(variable: str) -> Path:
    value = os.environ.get(variable)
    if value is None:
        pytest.skip(f"{variable} is required for the external-data regression")
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"{variable} does not name an artifact directory: {path}")
    return path


def test_dance1_subject1_clip5_keeps_bilateral_upper_arms_on_source_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify both upper arms stay on the source branch in the known dynamic regression clip."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the LAFAN-to-G1 retargeting regression")

    monkeypatch.setattr(builtins, "_isaaclab_tasks_registered", True, raising=False)
    benchmark = _load_benchmark_module()
    from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics
    from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
    from isaaclab_tasks.utils import resolve_presets

    device = "cuda:0"
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=frozenset(("g1", "lafan")))
    cfg.commands.motion.task_table.source_artifact_root = _artifact_root("ISAACLAB_LAFAN_RAW_ROOT")
    cfg.commands.motion.task_table.motion_split = "train"
    index = benchmark._inspect_source_index(cfg)
    clip_id = "dance1_subject1_clip5"
    clip_row = index.clip_ids.index(clip_id)
    rows = (clip_row,)
    target_cfg = cfg.commands.motion.task_table.target_kinematics
    reference = NewtonKinematics.from_articulation(
        target_cfg.kinematics,
        getattr(cfg.scene, target_cfg.asset_cfg.name),
        device,
    )
    target = target_cfg.target_factory(reference, target_cfg.contact_patches)
    trajectory = target.trajectory_target
    source = benchmark._selected_source_targets(cfg, target, index, rows, device)
    view = cfg.commands.motion.task_table.build_inspection_view(
        cfg.commands.motion,
        cfg.scene,
        device,
        sequence_limit=clip_row + 1,
    )
    benchmark._validate_inspection_index(view, index)
    joint_q = benchmark._inspection_joint_q(view, index, rows, reference).contiguous()
    joint_qd = torch.zeros((joint_q.shape[0], reference.model.joint_dof_count), dtype=torch.float32, device=device)
    body_q = torch.empty((joint_q.shape[0], reference.model.body_count, 7), dtype=torch.float32, device=device)
    body_qd = torch.empty((joint_q.shape[0], reference.model.body_count, 6), dtype=torch.float32, device=device)
    reference.eval_fk_batched_torch(joint_q, joint_qd, body_q, body_qd)

    output_position = body_q.index_select(1, trajectory.position_body_index_tensor)[..., :3]
    source_position = source.source_landmark_position_m.transpose(0, 1)
    role_row = {landmark.role: row for row, landmark in enumerate(trajectory.landmarks)}
    for side in ("left", "right"):
        shoulder = role_row[f"{side}_shoulder"]
        elbow = role_row[f"{side}_elbow"]
        output_direction = torch.nn.functional.normalize(
            output_position[:, elbow] - output_position[:, shoulder], dim=-1
        )
        source_direction = torch.nn.functional.normalize(
            source_position[:, elbow] - source_position[:, shoulder], dim=-1
        )
        error_deg = torch.rad2deg(torch.acos(torch.sum(output_direction * source_direction, dim=-1).clamp(-1.0, 1.0)))
        p95_deg = float(torch.quantile(error_deg, 0.95))
        max_deg = float(error_deg.max())
        assert p95_deg <= 35.0, f"{side} upper-arm p95 branch error: {p95_deg:.2f} deg"
        assert max_deg <= 50.0, f"{side} upper-arm max branch error: {max_deg:.2f} deg"
