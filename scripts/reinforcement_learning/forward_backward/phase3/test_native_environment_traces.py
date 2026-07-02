# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the frozen native environment traces without importing their repositories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from isaaclab_tasks.core.multi_task.motion.mdp.observations import smpl_humenv_observation

FIXTURES = Path(__file__).parent / "fixtures"
META_METADATA = FIXTURES / "meta_humenv_next_step_trace_v1.json"
META_TENSORS = FIXTURES / "meta_humenv_next_step_trace_v1.npz"
META_KEYS = {
    "actions",
    "current_observation",
    "current_qpos",
    "current_qvel",
    "current_time",
    "current_body_pos",
    "current_body_quat",
    "current_body_lin_vel",
    "current_body_ang_vel",
    "returned_observation",
    "returned_qpos",
    "returned_qvel",
    "returned_time",
    "returned_body_pos",
    "returned_body_quat",
    "returned_body_lin_vel",
    "returned_body_ang_vel",
    "reward",
    "terminated",
    "truncated",
    "action_applied",
}
META_EDGE_METADATA = FIXTURES / "meta_humenv_next_step_trace_v2.json"
META_EDGE_TENSORS = FIXTURES / "meta_humenv_next_step_trace_v2.npz"
META_EDGE_MUTABLE_NAMES = (
    "control",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "simulation_time_seconds",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def test_meta_trace_closes_over_exact_tensor_bytes() -> None:
    """Every tensor must match the independently hashed immutable trace manifest."""
    metadata = json.loads(META_METADATA.read_text())
    assert metadata["schema"] == "forward_backward_phase3_meta_humenv_trace_v1"
    assert metadata["trace"]["file_sha256"] == _sha256(META_TENSORS)
    with np.load(META_TENSORS, allow_pickle=False) as tensors:
        assert set(tensors.files) == META_KEYS
        for name in META_KEYS:
            summary = metadata["trace"]["tensors"][name]
            value = tensors[name]
            assert list(value.shape) == summary["shape"]
            assert str(value.dtype) == summary["dtype"]
            assert _tensor_hash(value) == summary["sha256"]


def test_meta_edge_trace_closes_over_complete_model_bytes() -> None:
    """The v2 edge oracle must hash every mutable input and fixed model fact."""
    metadata = json.loads(META_EDGE_METADATA.read_text())
    assert metadata["schema"] == "forward_backward_phase3_meta_humenv_trace_v2"
    assert metadata["trace"]["file_sha256"] == _sha256(META_EDGE_TENSORS)
    fixed_model_facts = set(metadata["edge_state"]["fixed_model_facts"])
    assert {
        "model_geom_margin",
        "model_geom_gap",
        "model_geom_solimp",
        "model_geom_solref",
        "model_option_solver",
        "model_option_integrator",
        "model_option_is_sparse",
        "model_option_timestep",
    } <= fixed_model_facts
    mutable_facts = {f"{point}_{name}" for point in ("current", "returned") for name in META_EDGE_MUTABLE_NAMES}
    expected = META_KEYS | mutable_facts | fixed_model_facts
    with np.load(META_EDGE_TENSORS, allow_pickle=False) as tensors:
        assert set(tensors.files) == expected
        for name in expected:
            summary = metadata["trace"]["tensors"][name]
            value = tensors[name]
            assert list(value.shape) == summary["shape"]
            assert str(value.dtype) == summary["dtype"]
            assert _tensor_hash(value) == summary["sha256"]


def test_meta_edge_trace_preserves_control_contact_and_solver_contract() -> None:
    """The native source edge inputs and contact/solver options must be explicit."""
    with np.load(META_EDGE_TENSORS, allow_pickle=False) as tensors:
        applied = tensors["action_applied"]
        np.testing.assert_array_equal(tensors["returned_control"][applied], tensors["actions"][applied])
        assert not tensors["current_qfrc_applied"].any()
        assert not tensors["current_xfrc_applied"].any()
        np.testing.assert_array_equal(tensors["model_geom_margin"], 0.001)
        np.testing.assert_array_equal(tensors["model_geom_gap"], 0.0)
        assert int(tensors["model_option_solver"]) == 2
        assert int(tensors["model_option_integrator"]) == 3
        assert int(tensors["model_option_cone"]) == 0
        assert int(tensors["model_option_jacobian"]) == 2
        assert int(tensors["model_option_iterations"]) == 100
        assert int(tensors["model_option_ls_iterations"]) == 50
        assert int(tensors["model_option_noslip_iterations"]) == 0
        assert bool(tensors["model_option_is_sparse"]) is True
        np.testing.assert_allclose(tensors["model_option_gravity"], (0.0, 0.0, -9.81), rtol=0.0, atol=0.0)
        assert float(tensors["model_option_timestep"]) == 1.0 / 450.0


def test_meta_edge_trace_separates_active_geometry_from_inactive_provenance() -> None:
    """The zero-mass nose is provenance, not part of the physical collision model."""
    with np.load(META_EDGE_TENSORS, allow_pickle=False) as tensors:
        active = (tensors["model_geom_contype"] != 0) | (tensors["model_geom_conaffinity"] != 0)
        np.testing.assert_array_equal(np.flatnonzero(~active), np.array((15,)))
        assert int(active.sum()) == 25
        assert int(tensors["model_geom_bodyid"][15]) == 14
        assert int(tensors["model_geom_type"][15]) == 3
        np.testing.assert_array_equal(tensors["model_geom_pos"][15], (0.0, 0.07, 0.05))
        np.testing.assert_array_equal(tensors["model_geom_size"][15], (0.02, 0.01, 0.0))


def test_meta_trace_exposes_next_step_reset_only_row() -> None:
    """The timeout successor and following reset-only row must remain distinct."""
    with np.load(META_TENSORS, allow_pickle=False) as tensors:
        assert tensors["actions"].shape == (5, 2, 69)
        assert tensors["current_observation"].shape == (5, 2, 358)
        assert tensors["current_qpos"].shape == (5, 2, 76)
        assert tensors["current_qvel"].shape == (5, 2, 75)
        assert not tensors["terminated"].any()
        np.testing.assert_array_equal(
            tensors["truncated"],
            np.array([[False, False], [False, False], [True, True], [False, False], [False, False]]),
        )
        np.testing.assert_array_equal(
            tensors["action_applied"],
            np.array([[True, True], [True, True], [True, True], [False, False], [True, True]]),
        )
        np.testing.assert_array_equal(tensors["returned_time"].reshape(5, 2), [[1, 1], [2, 2], [3, 3], [0, 0], [1, 1]])
        np.testing.assert_array_equal(tensors["current_time"].reshape(5, 2), [[0, 0], [1, 1], [2, 2], [3, 3], [0, 0]])
        np.testing.assert_array_equal(tensors["current_observation"][1:], tensors["returned_observation"][:-1])
        np.testing.assert_array_equal(tensors["current_qpos"][1:], tensors["returned_qpos"][:-1])
        np.testing.assert_array_equal(tensors["current_qvel"][1:], tensors["returned_qvel"][:-1])
        np.testing.assert_array_equal(tensors["returned_observation"][3], tensors["current_observation"][0])
        np.testing.assert_array_equal(tensors["returned_qpos"][3], tensors["current_qpos"][0])
        np.testing.assert_array_equal(tensors["returned_qvel"][3], tensors["current_qvel"][0])
        assert not tensors["reward"].any()


def test_meta_trace_validates_smpl_heading_body_observation_math() -> None:
    """The reusable Torch observation equation must match native HumEnv elementwise."""
    with np.load(META_TENSORS, allow_pickle=False) as tensors:
        expected = torch.from_numpy(tensors["current_observation"])
        actual = smpl_humenv_observation(
            torch.from_numpy(tensors["current_body_pos"]).flatten(0, 1),
            torch.from_numpy(tensors["current_body_quat"])[..., (1, 2, 3, 0)].flatten(0, 1),
            torch.from_numpy(tensors["current_body_lin_vel"]).flatten(0, 1),
            torch.from_numpy(tensors["current_body_ang_vel"]).flatten(0, 1),
        ).unflatten(0, (5, 2))
        torch.testing.assert_close(actual, expected, rtol=5.0e-7, atol=5.0e-7)

        expected = torch.from_numpy(tensors["returned_observation"])
        actual = smpl_humenv_observation(
            torch.from_numpy(tensors["returned_body_pos"]).flatten(0, 1),
            torch.from_numpy(tensors["returned_body_quat"])[..., (1, 2, 3, 0)].flatten(0, 1),
            torch.from_numpy(tensors["returned_body_lin_vel"]).flatten(0, 1),
            torch.from_numpy(tensors["returned_body_ang_vel"]).flatten(0, 1),
        ).unflatten(0, (5, 2))
        torch.testing.assert_close(actual, expected, rtol=5.0e-7, atol=5.0e-7)
