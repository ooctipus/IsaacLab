# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for isolated tracking evaluation and atomic curriculum updates."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import random
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer, ForwardBackwardExpertSchema
from tensordict import TensorDict

import isaaclab_tasks.core.multi_task.motion.tracking as tracking_module
from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex, MotionSampleGrid
from isaaclab_tasks.core.multi_task.motion.frames import G1_HEAD_FRAME_NAME
from isaaclab_tasks.core.multi_task.motion.mdp.actions import MotionJointPositionAction
from isaaclab_tasks.core.multi_task.motion.mdp.commands import MotionStatePayload, MotionTaskTable
from isaaclab_tasks.core.multi_task.motion.mdp.curriculums import MotionPenaltyScaleCurriculum
from isaaclab_tasks.core.multi_task.motion.mdp.events import MotionPushVelocity
from isaaclab_tasks.core.multi_task.motion.tracking import (
    MotionTrackingCurriculum,
    MotionTrackingEvaluation,
    g1_motion_tracking_evaluator,
    motion_tracking_priorities,
)
from isaaclab_tasks.core.multi_task.motion_env import MotionImitationEnv

_CLIP_IDS = ("clip_a", "clip_b")
_G1_JOINT_NAMES = tuple(f"joint_{index}" for index in range(29))
_G1_BODY_NAMES = tuple(f"body_{index}" for index in range(30))
_G1_REFERENCE_FRAME_NAMES = (*_G1_BODY_NAMES, G1_HEAD_FRAME_NAME)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _expert() -> ForwardBackwardExpertBuffer:
    schema = ForwardBackwardExpertSchema(
        dataset_id="tracking-test",
        data_hash="data",
        feature_schema_hash="features",
        clip_offsets_hash="offsets",
        expert_feature_width=3,
        num_frames=6,
        num_clips=2,
        window_lengths=(1,),
    )
    return ForwardBackwardExpertBuffer(
        torch.zeros(6, 3),
        torch.tensor((0, 3, 6), dtype=torch.int64),
        torch.ones(2),
        schema,
        clip_ids=_CLIP_IDS,
    )


def _g1_tracking_expert(device: str | torch.device = "cpu") -> ForwardBackwardExpertBuffer:
    offsets = torch.tensor((0, 3, 7), dtype=torch.int64, device=device)
    frames = torch.zeros(7, 527, device=device)
    for clip_index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        phase = torch.arange(end - start, dtype=torch.float32, device=device) + 10.0 * clip_index
        frames[start:end, :29] = phase[:, None] * torch.linspace(0.01, 0.29, 29, device=device)
        frames[start:end, 29:64] = phase[:, None] * 0.001
        frames[start:end, 64:] = phase[:, None] * 0.002
    schema = ForwardBackwardExpertSchema(
        dataset_id="g1-tracking-test",
        data_hash="data",
        feature_schema_hash="features",
        clip_offsets_hash="offsets",
        expert_feature_width=527,
        num_frames=7,
        num_clips=2,
        window_lengths=(1,),
    )
    return ForwardBackwardExpertBuffer(
        frames,
        offsets,
        torch.ones(2, device=device),
        schema,
        clip_ids=_CLIP_IDS,
    )


class _TrackingObservationSchema:
    field_widths = (("state", 64), ("privileged_state", 463))

    @staticmethod
    def route(name: str) -> tuple[str, ...]:
        assert name == "backward"
        return ("state", "privileged_state")


class _TrackingModel:
    observation_schema = _TrackingObservationSchema()

    def __init__(self) -> None:
        self.deterministic_calls: list[bool] = []

    @staticmethod
    def backward_map(observations: TensorDict) -> torch.Tensor:
        return observations["state"][:, :4]

    @staticmethod
    def context_project(context: torch.Tensor) -> torch.Tensor:
        return context

    def action_sample(
        self,
        observations: TensorDict,
        context: torch.Tensor,
        *,
        deterministic: bool,
    ) -> torch.Tensor:
        self.deterministic_calls.append(deterministic)
        assert observations.batch_size == context.shape[:1]
        return torch.zeros(context.shape[0], 29, device=context.device)


class _TrackingEnvironment:
    def __init__(self, expert: ForwardBackwardExpertBuffer) -> None:
        self.expert = expert
        self.num_envs = 1
        self.cfg = SimpleNamespace(
            commands=SimpleNamespace(motion=SimpleNamespace(payload=SimpleNamespace(episode_length_steps=4)))
        )
        self.device = expert.device
        self.default_joint_position = torch.linspace(-0.2, 0.2, 29, device=self.device)
        robot = SimpleNamespace(
            joint_names=_G1_JOINT_NAMES,
            body_names=_G1_BODY_NAMES,
            data=SimpleNamespace(
                joint_pos=SimpleNamespace(torch=torch.empty(1, 29, device=self.device)),
            ),
        )
        table = _table()
        payload = object.__new__(MotionStatePayload)
        payload.table = table
        payload.robot = robot
        self.payload = payload
        command = SimpleNamespace(table=table, payload=payload)
        action = object.__new__(MotionJointPositionAction)
        action._joint_names = _G1_JOINT_NAMES
        action._joint_ids_tensor = torch.arange(29, dtype=torch.int64, device=self.device)
        action._asset = robot
        action._joint_position = torch.empty(1, 29, device=self.device)
        action.joint_default_position = self.default_joint_position
        self.command_manager = SimpleNamespace(get_term=lambda name: command if name == "motion" else None)
        self.action_manager = SimpleNamespace(get_term=lambda name: action if name == "joint_position" else None)
        self.reset_assignments: list[int] = []
        self._clip = 0
        self._step = 0

    def _observation(self) -> TensorDict:
        start = int(self.expert.clip_offsets[self._clip])
        frame = self.expert.frames[start + self._step]
        return TensorDict(
            {
                "state": frame[:64].unsqueeze(0),
                "last_action": torch.zeros(1, 29, device=self.device),
                "history_actor": torch.zeros(1, 372, device=self.device),
                "privileged_state": frame[64:].unsqueeze(0),
            },
            batch_size=[1],
        )

    def _write_joint_position(self) -> None:
        start = int(self.expert.clip_offsets[self._clip])
        target = self.expert.frames[start + self._step, :29] + self.default_joint_position
        self.payload.robot.data.joint_pos.torch[0].copy_(target)

    def reset_motion_clips(self, clip_indices: torch.Tensor) -> tuple[TensorDict, dict]:
        self._clip = int(clip_indices[0])
        self._step = 0
        self.reset_assignments.append(self._clip)
        self._write_joint_position()
        return self._observation(), {}

    def step(self, _action: torch.Tensor):
        self._step += 1
        self._write_joint_position()
        return (
            self._observation(),
            torch.zeros(1, device=self.device),
            torch.zeros(1, dtype=torch.bool, device=self.device),
            torch.zeros(1, dtype=torch.bool, device=self.device),
            {},
        )


def _table() -> MotionTaskTable:
    clips = tuple(
        MotionClipIndex.Clip(
            clip_id=clip_id,
            source_path=f"{clip_id}.tensor",
            frame_count=count,
            source_fps=50.0,
            split="train",
            tags=(),
            content_sha256=_hash(clip_id),
        )
        for clip_id, count in zip(_CLIP_IDS, (3, 4), strict=True)
    )
    index = MotionClipIndex(
        source_content_sha256=_hash("tracking-source"),
        skeleton_sha256=_hash("tracking-skeleton"),
        semantic_level="g1_tracking",
        license="test-only",
        clips=clips,
    )
    frame_count = index.total_frames
    body_rotation = torch.zeros(frame_count, 31, 4)
    body_rotation[..., 3] = 1.0
    frames = MotionTaskTable.Frames(
        joint_position=torch.zeros(frame_count, 29),
        joint_velocity=torch.zeros(frame_count, 29),
        body_position=torch.zeros(frame_count, 31, 3),
        body_rotation=body_rotation,
        body_linear_velocity=torch.zeros(frame_count, 31, 3),
        body_angular_velocity=torch.zeros(frame_count, 31, 3),
    )
    return MotionTaskTable.from_storage(
        index,
        frames,
        _G1_JOINT_NAMES,
        _G1_REFERENCE_FRAME_NAMES,
        "tracking_builder_v1",
        _hash("tracking-builder"),
        "clip_time_ranges",
        (("reference", 0.7), ("lie_down", 0.3)),
        MotionSampleGrid.uniform_before_source_end(step_seconds=0.02),
        seed=0,
    )


class _Algorithm:
    def __init__(self, expert: ForwardBackwardExpertBuffer) -> None:
        self.expert = expert
        self.model = torch.nn.Linear(1, 1)
        self.resets: list[tuple[TensorDict, torch.Tensor]] = []

    def process_env_reset(self, observations: TensorDict, reset: torch.Tensor) -> None:
        self.resets.append((observations, reset.clone()))


class _Environment:
    def __init__(self, table: MotionTaskTable, *, fail_reset: bool = False) -> None:
        command = SimpleNamespace(table=table)
        self.unwrapped = SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: command if name == "motion" else None),
            evaluation_transaction=lambda _seed: nullcontext(),
        )
        self.num_envs = 2
        self.reset_count = 0
        self.fail_reset = fail_reset

    def reset(self) -> tuple[TensorDict, dict]:
        self.reset_count += 1
        if self.fail_reset:
            raise RuntimeError("reset failed")
        return TensorDict({"state": torch.zeros(2, 1)}, batch_size=[2]), {}


def _evaluation() -> MotionTrackingEvaluation:
    return MotionTrackingEvaluation(
        clip_ids=("clip_b", "clip_a"),
        emd=torch.tensor((3.0, 0.0)),
        obs_state_emd=torch.tensor((2.0, 1.0)),
        source_frame_counts=torch.tensor((4, 3), dtype=torch.int64),
        evaluated_frame_counts=torch.tensor((4, 3), dtype=torch.int64),
        coverage_fraction=torch.ones(2),
        duration_seconds=1.25,
    )


def test_tracking_priority_formula_reorders_stable_ids_and_clamps_emd() -> None:
    priorities = motion_tracking_priorities(_evaluation(), _CLIP_IDS, "cpu")

    torch.testing.assert_close(priorities, torch.tensor((2.0, 16.0)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_matches_released_pot_operator() -> None:
    """Exact GPU assignment must match the released CPU oracle without a runtime dependency."""
    optimal_transport = pytest.importorskip("o" + "t")
    generator = torch.Generator().manual_seed(17)
    device = torch.device("cuda:0")
    for count, width in ((1, 29), (2, 23), (7, 29), (32, 23)):
        observed_cpu = torch.randn(count, width, generator=generator)
        target_cpu = torch.randn(count, width, generator=generator)
        observed_norm = observed_cpu.square().sum(dim=1).reshape(-1, 1)
        target_norm = target_cpu.square().sum(dim=1).reshape(1, -1)
        cost = torch.sqrt(
            torch.clamp(
                observed_norm + target_norm - 2.0 * torch.matmul(observed_cpu, target_cpu.mT),
                min=0.0,
            )
        ).numpy()
        weights = np.ones(count) / count
        expected = optimal_transport.emd2(weights, weights, cost, numItermax=100_000)
        output = torch.empty(1, dtype=torch.float64, device=device)
        workspace = tracking_module._UniformEmdWorkspace(capacity=1, max_frames=count, device=device)

        workspace.compute(
            observed_cpu.to(device).unsqueeze(0),
            target_cpu.to(device).unsqueeze(0),
            torch.tensor((count,), dtype=torch.int64, device=device),
            output,
        )

        assert output.item() == pytest.approx(expected, rel=0.0, abs=2.0e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_matches_frozen_cpu_oracle_for_variable_lengths() -> None:
    """GPU assignment must preserve released exact costs without copying traces to the host."""
    workspace_type = getattr(tracking_module, "_UniformEmdWorkspace")
    device = torch.device("cuda:0")
    observed = torch.tensor(
        (
            ((0.0, 0.0), (2.0, 0.0), (9.0, 9.0), (9.0, 9.0)),
            ((0.0, 1.0), (2.0, 1.0), (4.0, 1.0), (6.0, 1.0)),
        ),
        device=device,
    )
    target = torch.tensor(
        (
            ((3.0, 0.0), (0.0, 1.0), (8.0, 8.0), (8.0, 8.0)),
            ((6.0, 2.0), (0.0, 0.0), (5.0, 1.0), (2.0, 3.0)),
        ),
        device=device,
    )
    lengths = torch.tensor((2, 4), dtype=torch.int64, device=device)
    output = torch.empty(2, dtype=torch.float64, device=device)

    workspace = workspace_type(capacity=2, max_frames=4, device=device)
    workspace.compute(observed, target, lengths, output)

    torch.testing.assert_close(output.cpu(), torch.tensor((1.0, 1.25), dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_uses_stable_lowest_column_ties() -> None:
    """Equal costs must produce the same lowest-column assignment on every launch."""
    device = torch.device("cuda:0")
    workspace = tracking_module._UniformEmdWorkspace(capacity=1, max_frames=4, device=device)
    values = torch.zeros(1, 4, 1, device=device)
    lengths = torch.tensor((4,), dtype=torch.int64, device=device)
    output = torch.empty(1, dtype=torch.float64, device=device)

    workspace.compute(values, values, lengths, output)

    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(
        workspace.matching[0, 1:5].cpu(),
        torch.tensor((1, 2, 3, 4), dtype=torch.int32),
    )


def test_gpu_uniform_emd_compute_reuses_preallocated_scratch() -> None:
    """The repeated assignment path must not allocate, synchronize, or repair layouts."""
    source = inspect.getsource(tracking_module._UniformEmdWorkspace.compute)

    for forbidden in ("torch.empty", "torch.zeros", ".cpu(", ".item(", ".contiguous("):
        assert forbidden not in source


def test_native_tracking_assignment_shuffles_fixed_domain_randomization_rows() -> None:
    """Tracking must reproduce released shuffled env pairing and first-row selection."""
    assignment = getattr(tracking_module, "_native_tracking_assignment")
    python_state = random.getstate()
    random.seed(17)
    try:
        assigned, representatives = assignment(8, 3, torch.device("cpu"))
    finally:
        random.setstate(python_state)

    torch.testing.assert_close(assigned, torch.tensor((0, 1, 0, 1, 2, 2, 1, 0)))
    torch.testing.assert_close(representatives, torch.tensor((0, 1, 4)))


def test_tracking_evaluator_does_not_copy_rollout_traces_to_host() -> None:
    """The production evaluator must keep rollout traces and EMD calculation on the GPU."""
    source = inspect.getsource(g1_motion_tracking_evaluator)

    assert ".cpu(" not in source
    assert ".tolist(" not in source


def test_tracking_rollout_horizon_comes_from_unique_chunk_clips() -> None:
    """Shuffled duplicate env rows cannot shorten a variable-length partial chunk."""
    source = inspect.getsource(g1_motion_tracking_evaluator)

    assert "chunk_lengths.max().item()" in source
    assert "context_lengths[:chunk_count]" not in source


def test_tracking_runtime_does_not_import_transport_dependencies() -> None:
    """POT and SciPy may validate the metric in tests but cannot enter the runtime boundary."""
    imports: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(tracking_module))):
        if isinstance(node, ast.Import):
            imports.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module.partition(".")[0])

    assert imports.isdisjoint({"o" + "t", "scipy"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production evaluator.")
def test_concrete_tracking_evaluator_reuses_expert_and_mean_actions_across_chunks() -> None:
    """One shared evaluator must reset exact clips and retain released full/diagnostic metrics."""
    expert = _g1_tracking_expert("cuda:0")
    model = _TrackingModel()
    env = _TrackingEnvironment(expert)

    evaluation = g1_motion_tracking_evaluator(model, env, expert, _CLIP_IDS)

    assert evaluation.clip_ids == _CLIP_IDS
    assert env.reset_assignments == [0, 1]
    assert model.deterministic_calls == [True] * 5
    expected_emd = torch.tensor(
        (0.0, 0.0009765625),
        dtype=torch.float64,
        device=expert.device,
    )
    torch.testing.assert_close(evaluation.emd, expected_emd)
    assert evaluation.metrics["clip_a"]["num_frames"] == 3
    assert evaluation.metrics["clip_b"]["num_frames"] == 4
    assert evaluation.metrics["clip_a"]["coverage_fraction"] == 1.0
    assert evaluation.metrics["clip_b"]["coverage_fraction"] == 1.0
    assert evaluation.metrics["clip_a"]["emd"] == 0.0
    torch.testing.assert_close(
        evaluation.metrics["clip_b"]["obs_state_emd"],
        torch.tensor(0.0006905339541845024, dtype=torch.float64, device=expert.device),
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production evaluator.")
def test_tracking_caps_long_clips_before_the_native_timeout_edge() -> None:
    """Cross-composed clips must report prefix coverage instead of resetting mid-metric."""
    expert = _g1_tracking_expert("cuda:0")
    model = _TrackingModel()
    env = _TrackingEnvironment(expert)
    env.cfg.commands.motion.payload.episode_length_steps = 3

    evaluation = g1_motion_tracking_evaluator(model, env, expert, _CLIP_IDS)

    assert evaluation.metrics["clip_a"]["source_num_frames"] == 3
    assert evaluation.metrics["clip_a"]["evaluated_num_frames"] == 3
    assert evaluation.metrics["clip_a"]["coverage_fraction"] == 1.0
    assert evaluation.metrics["clip_b"]["source_num_frames"] == 4
    assert evaluation.metrics["clip_b"]["evaluated_num_frames"] == 3
    assert evaluation.metrics["clip_b"]["coverage_fraction"] == 0.75


def test_curriculum_validates_updates_resets_and_records_atomically(tmp_path) -> None:
    table = _table()
    expert = _expert()
    algorithm = _Algorithm(expert)
    env = _Environment(table)
    coordinator = MotionTrackingCurriculum(
        env,
        algorithm,
        str(tmp_path),
        "cpu",
        evaluator=lambda *_args: _evaluation(),
    )

    observations = coordinator.on_transition(8)

    torch.testing.assert_close(table.clip_priorities, torch.tensor((2.0, 16.0)))
    torch.testing.assert_close(expert.priorities, torch.tensor((2.0, 16.0)))
    assert env.reset_count == 1
    assert observations is algorithm.resets[0][0]
    assert torch.all(algorithm.resets[0][1])
    record = json.loads((tmp_path / "tracking_curriculum" / "8.json").read_text())
    assert record["schema"] == "motion_tracking_curriculum_v1"
    assert record["clip_ids"] == list(_CLIP_IDS)
    assert record["priorities"] == [2.0, 16.0]
    assert record["duration_seconds"] == 1.25
    assert record["metrics"]["clip_a"]["emd"] == 0.0
    assert record["metrics"]["clip_b"]["source_num_frames"] == 4


def test_curriculum_validates_both_targets_before_first_priority_mutation(tmp_path, monkeypatch) -> None:
    table = _table()
    expert = _expert()
    algorithm = _Algorithm(expert)
    coordinator = MotionTrackingCurriculum(
        _Environment(table),
        algorithm,
        str(tmp_path),
        "cpu",
        evaluator=lambda *_args: _evaluation(),
    )
    original = table.clip_priorities.clone()

    def reject(_priorities: torch.Tensor) -> None:
        raise ValueError("expert rejected priorities")

    monkeypatch.setattr(expert, "validate_priorities", reject)
    with pytest.raises(ValueError, match="expert rejected"):
        coordinator.on_transition(8)

    torch.testing.assert_close(table.clip_priorities, original)
    assert not (tmp_path / "tracking_curriculum" / "8.json").exists()


def test_curriculum_reset_failure_rolls_back_both_samplers_and_record(tmp_path) -> None:
    """A failed reset must restore exact pre-event sampling state and publish nothing."""
    table = _table()
    expert = _expert()
    algorithm = _Algorithm(expert)
    env = _Environment(table, fail_reset=True)
    coordinator = MotionTrackingCurriculum(
        env,
        algorithm,
        str(tmp_path),
        "cpu",
        evaluator=lambda *_args: _evaluation(),
    )

    with pytest.raises(RuntimeError, match="reset failed"):
        coordinator.on_transition(8)

    assert not table.clip_priorities_active
    torch.testing.assert_close(table.clip_priorities, torch.ones(2))
    torch.testing.assert_close(expert.priorities, torch.ones(2))
    assert env.reset_count == 1
    assert not (tmp_path / "tracking_curriculum" / "8.json").exists()
    assert not (tmp_path / "tracking_curriculum" / "8.tmp").exists()


def test_curriculum_state_restores_both_priority_targets_and_event_history(tmp_path) -> None:
    table = _table()
    expert = _expert()
    coordinator = MotionTrackingCurriculum(
        _Environment(table),
        _Algorithm(expert),
        str(tmp_path),
        "cpu",
        evaluator=lambda *_args: _evaluation(),
    )
    coordinator.on_transition(8)
    state = coordinator.state_dict()
    table.clip_priorities.fill_(1.0)
    expert.set_priorities(torch.ones(2))

    coordinator.load_state_dict(state)

    torch.testing.assert_close(table.clip_priorities, torch.tensor((2.0, 16.0)))
    torch.testing.assert_close(expert.priorities, torch.tensor((2.0, 16.0)))
    assert coordinator.applied_transitions == [8]


def test_curriculum_state_before_first_event_restores_uniform_sampler(tmp_path) -> None:
    """A transition-zero-pending checkpoint must retain the zero-storage uniform path."""
    table = _table()
    expert = _expert()
    coordinator = MotionTrackingCurriculum(
        _Environment(table),
        _Algorithm(expert),
        str(tmp_path),
        "cpu",
        evaluator=lambda *_args: _evaluation(),
    )
    state = coordinator.state_dict()
    table.set_clip_priorities(_CLIP_IDS, torch.tensor((2.0, 16.0)))
    expert.set_priorities(torch.tensor((2.0, 16.0)))
    coordinator.applied_transitions.append(8)

    coordinator.load_state_dict(state)

    assert not table.clip_priorities_active
    assert table._row_priorities.numel() == 0
    torch.testing.assert_close(table.clip_priorities, torch.ones(2))
    torch.testing.assert_close(expert.priorities, torch.ones(2))
    assert coordinator.applied_transitions == []


def _motion_table_draws(table: MotionTaskTable) -> dict[str, torch.Tensor]:
    """Draw the four stochastic reset decisions owned by one motion task table."""
    count = 8
    return {
        "row": table.sample_rows(count),
        "source": table.sample_reset_sources(count),
        "time": torch.rand(count, generator=table.generator),
        "fall": torch.randint(8192, (count,), generator=table.generator),
    }


def _assert_motion_table_draws_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


def _run_motion_evaluation_transaction(prior_table_draws: int) -> dict[str, torch.Tensor]:
    table = _table()
    table.generator.manual_seed(11)
    torch.rand(prior_table_draws, generator=table.generator)
    training_state = table.generator.get_state().clone()
    expected_generator = torch.Generator()
    expected_generator.set_state(training_state)
    expected_table = _table()
    object.__setattr__(expected_table, "generator", expected_generator)
    expected_training_draws = _motion_table_draws(expected_table)

    term = SimpleNamespace(
        table=table,
        time_left=torch.tensor((1.0, 2.0)),
        command_counter=torch.tensor((3, 4), dtype=torch.int64),
    )
    push = object.__new__(MotionPushVelocity)
    push._elapsed_seconds = torch.tensor((11, 22), dtype=torch.int32)
    push._interval_seconds = torch.tensor((1, 2), dtype=torch.int32)
    penalty = object.__new__(MotionPenaltyScaleCurriculum)
    penalty._scale = 0.25
    penalty._average_episode_length = 41.0
    event_manager = SimpleNamespace(
        active_terms={"interval": ["push"]},
        get_term_cfg=lambda name: SimpleNamespace(func=push) if name == "push" else None,
        _interval_term_time_left=[torch.tensor((0.25, 0.75))],
        _reset_term_last_triggered_step_id=[torch.tensor((7, 9), dtype=torch.int32)],
        _reset_term_last_triggered_once=[torch.tensor((True, False))],
    )
    curriculum_manager = SimpleNamespace(
        active_terms=("penalty_scale",),
        get_term=lambda name: penalty if name == "penalty_scale" else None,
        _curriculum_state={
            "penalty_scale": {
                "penalty_scale": 0.25,
                "average_episode_length": 41.0,
            }
        },
    )
    env = object.__new__(MotionImitationEnv)
    env._is_closed = True
    env.common_step_counter = 17
    env._sim_step_counter = 255
    env.episode_length_buf = torch.tensor((5, 6), dtype=torch.int64)
    env.command_manager = SimpleNamespace(
        active_terms=("motion",),
        get_term=lambda name: term if name == "motion" else None,
    )
    env.event_manager = event_manager
    env.curriculum_manager = curriculum_manager
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    expected_global = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    with MotionImitationEnv.evaluation_transaction(env, 99):
        evaluation_draws = _motion_table_draws(table)
        random.random()
        np.random.rand()
        torch.rand(1)
        env.common_step_counter = 100
        env._sim_step_counter = 1500
        env.episode_length_buf.zero_()
        term.time_left.zero_()
        term.command_counter.zero_()
        env.event_manager._interval_term_time_left[0].zero_()
        env.event_manager._reset_term_last_triggered_step_id[0].zero_()
        env.event_manager._reset_term_last_triggered_once[0].logical_not_()
        env.curriculum_manager._curriculum_state["penalty_scale"]["penalty_scale"] = 0.75
        push._elapsed_seconds.zero_()
        push._interval_seconds.fill_(3)
        penalty._scale = 0.75
        penalty._average_episode_length = 120.0

    actual_global = (random.random(), np.random.rand(), torch.rand(1))
    assert actual_global[0] == expected_global[0]
    assert actual_global[1] == expected_global[1]
    torch.testing.assert_close(actual_global[2], expected_global[2])
    _assert_motion_table_draws_equal(_motion_table_draws(table), expected_training_draws)
    assert env.common_step_counter == 17
    assert env._sim_step_counter == 255
    torch.testing.assert_close(env.episode_length_buf, torch.tensor((5, 6)))
    torch.testing.assert_close(term.time_left, torch.tensor((1.0, 2.0)))
    torch.testing.assert_close(term.command_counter, torch.tensor((3, 4)))
    torch.testing.assert_close(env.event_manager._interval_term_time_left[0], torch.tensor((0.25, 0.75)))
    torch.testing.assert_close(
        env.event_manager._reset_term_last_triggered_step_id[0],
        torch.tensor((7, 9), dtype=torch.int32),
    )
    torch.testing.assert_close(
        env.event_manager._reset_term_last_triggered_once[0],
        torch.tensor((True, False)),
    )
    assert env.curriculum_manager._curriculum_state == {
        "penalty_scale": {
            "penalty_scale": 0.25,
            "average_episode_length": 41.0,
        }
    }
    torch.testing.assert_close(push._elapsed_seconds, torch.tensor((11, 22), dtype=torch.int32))
    torch.testing.assert_close(push._interval_seconds, torch.tensor((1, 2), dtype=torch.int32))
    assert penalty.scale == 0.25
    assert penalty.average_episode_length == 41.0
    return evaluation_draws


def test_motion_evaluation_transaction_isolates_rng_clocks_and_stateful_terms() -> None:
    """Evaluator draws are seed-only and every persistent training clock resumes exactly."""
    first = _run_motion_evaluation_transaction(prior_table_draws=0)
    second = _run_motion_evaluation_transaction(prior_table_draws=37)
    _assert_motion_table_draws_equal(first, second)
