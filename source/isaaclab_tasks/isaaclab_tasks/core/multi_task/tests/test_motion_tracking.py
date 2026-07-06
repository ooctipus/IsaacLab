# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for isolated tracking evaluation and atomic curriculum updates."""

from __future__ import annotations

import ast
import hashlib
import inspect
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer, ForwardBackwardExpertSchema
from tensordict import TensorDict

import isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_tracking as tracking_module
from isaaclab_tasks.core.multi_task.mdp.commands.state_command.state_command import StateCommand
from isaaclab_tasks.core.multi_task.metrics import UniformAssignmentWorkspace
from isaaclab_tasks.core.multi_task.metrics.impl import uniform_assignment_warp as uniform_emd_module
from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex, MotionFrames
from isaaclab_tasks.core.multi_task.motion.mdp.commands import MotionSampler, MotionStatePayload, MotionTaskTable
from isaaclab_tasks.core.multi_task.motion.robots.g1.frames import G1_HEAD_FRAME_NAME
from isaaclab_tasks.core.multi_task.motion.robots.g1.observations import g1_bfm_observation_state_pose
from isaaclab_tasks.core.multi_task.motion.robots.smpl.observations import (
    smpl_humenv_observation,
    smpl_humenv_tracking_pose,
)
from isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_tracking import (
    ForwardBackwardTrackingCurriculum,
    ForwardBackwardTrackingEvaluation,
    forward_backward_tracking_evaluator,
    forward_backward_tracking_priority_scores,
)
from isaaclab_tasks.core.multi_task.tests.motion_table_test_utils import motion_task_table

_CLIP_IDS = ("clip_a", "clip_b")
_G1_JOINT_NAMES = tuple(f"joint_{index}" for index in range(29))
_G1_BODY_NAMES = tuple(f"body_{index}" for index in range(30))
_MOTION_RESET_SOURCES = (("reference", 0.7), ("lie_down", 0.3))
_G1_REFERENCE_FRAME_NAMES = (*_G1_BODY_NAMES, G1_HEAD_FRAME_NAME)
_G1_TRACKING_PROJECTIONS = (
    {
        "metric_name": "emd",
        "target_name": "joint_position",
        "observation_name": "joint_position_unnoised",
        "projection": None,
    },
    {
        "metric_name": "obs_state_emd",
        "target_name": "joint_position",
        "observation_name": "joint_position",
        "projection": ("isaaclab_tasks.core.multi_task.motion.robots.g1.observations:g1_bfm_observation_state_pose"),
    },
)
_TRACKING_PROTOCOL = {
    "context_window_length": 1,
    "include_reset_frame": True,
    "allow_horizon_truncation": True,
    "shuffle_assignments": True,
}
_PRIORITY_PROTOCOL = {
    "metric_name": "emd",
    "metric_minimum": 0.5,
    "metric_maximum": 2.0,
    "exponent_scale": 2.0,
    "exponent_base": 2.0,
}
_CURRICULUM_PROTOCOL = {
    **_TRACKING_PROTOCOL,
    "command_bind": "env.unwrapped.command_manager.get_term('motion')",
    "sequence_ids_bind": "env.unwrapped.command_manager.get_term('motion').table.clip_ids",
    "sequence_start_rows_bind": "env.unwrapped.command_manager.get_term('motion').table.clip_start_rows",
    "evaluation_scope_bind": "env.unwrapped.command_manager.get_term('motion').payload.sampler.reset_sampling_scope",
    "priority_metric_name": "emd",
    "priority_metric_minimum": 0.5,
    "priority_metric_maximum": 2.0,
    "priority_exponent_scale": 2.0,
    "priority_exponent_base": 2.0,
}
_SMPL_CLIP_IDS = ("clip_a", "clip_b", "clip_c", "clip_d")
_SMPL_LENGTHS = (6, 5, 4, 3)


@pytest.fixture(autouse=True)
def _stub_curriculum_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace only curriculum-model evaluations with deterministic generic results."""
    real_evaluator = tracking_module.forward_backward_tracking_evaluator

    def evaluate(model, *args, **kwargs):
        if isinstance(model, torch.nn.Linear):
            model.evaluation_envs.append(args[0])
            shuffled = list(range(8))
            kwargs["assignment_rng"].shuffle(shuffled)
            model.assignment_orders.append(tuple(shuffled))
            return _evaluation()
        return real_evaluator(model, *args, **kwargs)

    monkeypatch.setattr(tracking_module, "forward_backward_tracking_evaluator", evaluate)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _expert(priorities: torch.Tensor | None = None) -> ForwardBackwardExpertBuffer:
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
        torch.ones(2) if priorities is None else priorities,
        schema,
        seed=0,
        clip_ids=_CLIP_IDS,
        clip_length_values=(3, 3),
    )


def _g1_tracking_expert(device: str | torch.device = "cpu") -> ForwardBackwardExpertBuffer:
    offsets = (0, 3, 7)
    offset_tensor = torch.tensor(offsets, dtype=torch.int64, device=device)
    frames = torch.zeros(offsets[-1], 527, device=device)
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
        num_frames=offsets[-1],
        num_clips=len(_CLIP_IDS),
        window_lengths=(1,),
    )
    return ForwardBackwardExpertBuffer(
        frames,
        offset_tensor,
        torch.ones(len(_CLIP_IDS), device=device),
        schema,
        seed=0,
        clip_ids=_CLIP_IDS,
        clip_length_values=(3, 4),
    )


class _TrackingObservationSchema:
    field_widths = (
        ("joint_position", 29),
        ("joint_velocity", 29),
        ("projected_gravity", 3),
        ("base_angular_velocity", 3),
        ("privileged_state", 463),
    )

    @staticmethod
    def route(name: str) -> tuple[str, ...]:
        assert name == "backward"
        return ("joint_position", "joint_velocity", "projected_gravity", "base_angular_velocity", "privileged_state")


class _TrackingModel:
    observation_schema = _TrackingObservationSchema()
    context_dim = 4
    context_normalization = False

    def __init__(self, action_value: float = 0.0) -> None:
        self.deterministic_calls: list[bool] = []
        self.action_value = action_value

    @staticmethod
    def backward_map(observations: TensorDict) -> torch.Tensor:
        return observations["joint_position"][:, :4]

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
        return torch.full((context.shape[0], 29), self.action_value, device=context.device)


class _TrackingEnvironment:
    def __init__(
        self,
        expert: ForwardBackwardExpertBuffer,
        *,
        num_envs: int = 1,
        clip_actions: float | None = None,
    ) -> None:
        self.expert = expert
        self.num_envs = num_envs
        self.max_episode_length = 4
        self.device = expert.device
        self.clip_actions = clip_actions
        self.applied_actions: list[torch.Tensor] = []
        table = _table()
        payload = object.__new__(MotionStatePayload)
        payload.table = table
        payload.sampler = _sampler(table)
        self.payload = payload
        self.command = SimpleNamespace(table=table, payload=payload, randomize_command_indices=False)
        self.command.bind_rows = self._bind_rows
        self.command_manager = SimpleNamespace(get_term=lambda name: self.command if name == "motion" else None)
        self.reset_assignments: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self._clips = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._pending_clips = torch.zeros_like(self._clips)
        self._steps = torch.zeros(num_envs, dtype=torch.int64, device=self.device)

    def _bind_rows(self, env_ids: torch.Tensor, task_rows: torch.Tensor) -> None:
        clip_start_rows = self.command.table.clip_start_rows.to(self.device)
        self._pending_clips.index_copy_(0, env_ids, torch.searchsorted(clip_start_rows, task_rows))

    def _observation(self) -> TensorDict:
        lengths = self.expert.clip_lengths.index_select(0, self._clips)
        local_steps = torch.minimum(self._steps, lengths - 1)
        starts = self.expert.clip_offsets.index_select(0, self._clips)
        frames = self.expert.frames.index_select(0, starts + local_steps)
        joint_position = frames[:, :29]
        return TensorDict(
            {
                "joint_position": joint_position,
                "joint_position_unnoised": joint_position,
                "joint_velocity": frames[:, 29:58],
                "projected_gravity": frames[:, 58:61],
                "base_angular_velocity": frames[:, 61:64],
                "last_action": torch.zeros(self.num_envs, 29, device=self.device),
                "privileged_state": frames[:, 64:],
            },
            batch_size=[self.num_envs],
        )

    def reset_clips(
        self,
        clip_indices: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[TensorDict, dict]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._clips.index_copy_(0, env_ids, clip_indices)
        self._steps.index_fill_(0, env_ids, 0)
        self.reset_assignments.append(
            (
                tuple(env_ids.cpu().tolist()),
                tuple(clip_indices.cpu().tolist()),
            )
        )
        return self._observation(), {}

    def reset(self) -> tuple[TensorDict, dict]:
        return self.reset_clips(self._pending_clips)

    def step(self, action: torch.Tensor):
        applied_action = action if self.clip_actions is None else action.clamp(-self.clip_actions, self.clip_actions)
        self.applied_actions.append(applied_action.clone())
        self._steps.add_(1)
        return (
            self._observation(),
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            {},
        )


def _table() -> MotionTaskTable:
    clips = tuple(
        MotionClipIndex.Clip(
            clip_id=clip_id,
            frame_count=count,
            source_fps=50.0,
            content_sha256=_hash(clip_id),
        )
        for clip_id, count in zip(_CLIP_IDS, (4, 5), strict=True)
    )
    index = MotionClipIndex(
        source_content_sha256=_hash("tracking-source"),
        clips=clips,
    )
    frame_count = index.total_frames
    body_rotation = torch.zeros(frame_count, 31, 4)
    body_rotation[..., 3] = 1.0
    frames = MotionFrames(
        joint_position=torch.zeros(frame_count, 29),
        joint_velocity=torch.zeros(frame_count, 29),
        body_position=torch.zeros(frame_count, 31, 3),
        body_rotation=body_rotation,
        body_linear_velocity=torch.zeros(frame_count, 31, 3),
        body_angular_velocity=torch.zeros(frame_count, 31, 3),
    )
    return motion_task_table(
        index,
        frames,
        _G1_JOINT_NAMES,
        _G1_REFERENCE_FRAME_NAMES,
        "tracking_builder_v1",
        _hash("tracking-builder"),
        "clip_time_ranges",
        _hash("tracking-skeleton"),
    )


def _sampler(table: MotionTaskTable) -> MotionSampler:
    return MotionSampler(table, _MOTION_RESET_SOURCES, capacity=table.num_tasks, seed=0)


def test_reset_tracking_sequences_requires_scope_and_performs_one_ordinary_reset() -> None:
    """Exact evaluation reset must use command selection plus the normal reset lifecycle."""
    table = _table()
    payload = object.__new__(MotionStatePayload)
    payload.sampler = _sampler(table)
    payload.sampler.set_reset_time_mode("range_start")
    command = SimpleNamespace(
        table=table,
        payload=payload,
        randomize_command_indices=True,
        cmd_indices=torch.empty(2, dtype=torch.int64),
    )
    command.bind_rows = lambda env_ids, task_rows: command.cmd_indices.__setitem__(env_ids, task_rows)
    reset_calls = []

    def reset() -> tuple[str, dict]:
        reset_calls.append(command.cmd_indices.clone())
        return "observations", {"reset": True}

    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        command_manager=SimpleNamespace(get_term=lambda name: command if name == "motion" else None),
        reset=reset,
    )
    clip_indices = torch.tensor((1, 0), dtype=torch.int64)

    with pytest.raises(RuntimeError, match="evaluation_scope"):
        tracking_module.reset_tracking_sequences(env, command, table.clip_start_rows, clip_indices)
    command.randomize_command_indices = False
    result = tracking_module.reset_tracking_sequences(
        env,
        command,
        table.clip_start_rows,
        clip_indices,
    )

    expected_rows = table.clip_start_rows.index_select(0, clip_indices)
    assert result == ("observations", {"reset": True})
    assert command.randomize_command_indices is False
    assert len(reset_calls) == 1
    torch.testing.assert_close(reset_calls[0], expected_rows)


def test_forward_backward_evaluation_scope_restores_randomization_after_exception() -> None:
    """Evaluation must not leave ordinary training resets pinned to exact command rows."""
    table = _table()
    payload = object.__new__(MotionStatePayload)
    payload.sampler = _sampler(table)
    command = SimpleNamespace(table=table, payload=payload, randomize_command_indices=True)
    env = SimpleNamespace(
        device="cpu",
        command_manager=SimpleNamespace(get_term=lambda name: command if name == "motion" else None),
    )
    generator_state = payload.sampler.generator.get_state().clone()
    reset_probabilities = payload.sampler.reset_source_probabilities.clone()

    with pytest.raises(RuntimeError, match="evaluation failed"):
        with tracking_module.forward_backward_evaluation_scope(
            env,
            command,
            payload.sampler.reset_sampling_scope,
            17,
            reset_source_name="reference",
        ):
            assert payload.sampler.reset_time_mode == "range_start"
            assert command.randomize_command_indices is False
            raise RuntimeError("evaluation failed")

    assert command.randomize_command_indices is True
    assert payload.sampler.reset_time_mode == "uniform"
    torch.testing.assert_close(payload.sampler.generator.get_state(), generator_state)
    torch.testing.assert_close(payload.sampler.reset_source_probabilities, reset_probabilities)


def _smpl_tracking_expert_table(
    device: str | torch.device,
) -> tuple[ForwardBackwardExpertBuffer, MotionTaskTable]:
    clips = tuple(
        MotionClipIndex.Clip(
            clip_id=clip_id,
            frame_count=length,
            source_fps=50.0,
            content_sha256=_hash(f"smpl-{clip_id}"),
        )
        for clip_id, length in zip(_SMPL_CLIP_IDS, _SMPL_LENGTHS, strict=True)
    )
    index = MotionClipIndex(
        source_content_sha256=_hash("smpl-tracking-source"),
        clips=clips,
    )
    frame_count = index.total_frames
    phase = torch.empty(frame_count, dtype=torch.float32, device=device)
    for clip_index, (start, end) in enumerate(zip(index.offsets[:-1], index.offsets[1:], strict=True)):
        phase[start:end].copy_(torch.arange(end - start, dtype=torch.float32, device=device) + 10.0 * clip_index)
    body_index = torch.arange(24, dtype=torch.float32, device=device)
    linear_rate = 0.01 + body_index * 0.001
    angular_rate = 0.005 + body_index * 0.0002
    body_position = torch.zeros(frame_count, 24, 3, dtype=torch.float32, device=device)
    body_position[..., 0] = phase[:, None] * linear_rate
    body_position[..., 1] = body_index * 0.02
    body_position[..., 2] = 1.0 + phase[:, None] * 0.005 + body_index * 0.001
    angle = phase[:, None] * angular_rate
    body_rotation = torch.zeros(frame_count, 24, 4, dtype=torch.float32, device=device)
    body_rotation[..., 2] = torch.sin(0.5 * angle)
    body_rotation[..., 3] = torch.cos(0.5 * angle)
    body_linear_velocity = torch.zeros(frame_count, 24, 3, dtype=torch.float32, device=device)
    body_linear_velocity[..., 0] = linear_rate * 50.0
    body_linear_velocity[..., 2] = 0.25
    body_angular_velocity = torch.zeros(frame_count, 24, 3, dtype=torch.float32, device=device)
    body_angular_velocity[..., 2] = angular_rate * 50.0
    frames = MotionFrames(
        joint_position=torch.zeros(frame_count, 69, dtype=torch.float32, device=device),
        joint_velocity=torch.zeros(frame_count, 69, dtype=torch.float32, device=device),
        body_position=body_position,
        body_rotation=body_rotation,
        body_linear_velocity=body_linear_velocity,
        body_angular_velocity=body_angular_velocity,
    )
    table = motion_task_table(
        index,
        frames,
        tuple(f"joint_{index}" for index in range(69)),
        tuple(f"body_{index}" for index in range(24)),
        "smpl_tracking_builder_v1",
        _hash("smpl-tracking-builder"),
        "source_frames",
        _hash("smpl-tracking-skeleton"),
    )
    expert_frames = smpl_humenv_observation(
        table.field("body_position"),
        table.field("body_rotation"),
        table.field("body_linear_velocity"),
        table.field("body_angular_velocity"),
    )
    schema = ForwardBackwardExpertSchema(
        dataset_id="smpl-packed-tracking-test",
        data_hash="data",
        feature_schema_hash="features",
        clip_offsets_hash="offsets",
        expert_feature_width=358,
        num_frames=frame_count,
        num_clips=len(_SMPL_CLIP_IDS),
        window_lengths=(1,),
    )
    expert = ForwardBackwardExpertBuffer(
        expert_frames,
        table.clip_offsets,
        torch.ones(len(_SMPL_CLIP_IDS), device=device),
        schema,
        clip_ids=_SMPL_CLIP_IDS,
        clip_length_values=_SMPL_LENGTHS,
    )
    return expert, table


def test_smpl_tracking_expert_projects_physical_table_frames() -> None:
    expert, table = _smpl_tracking_expert_table("cpu")
    expected = smpl_humenv_observation(
        table.field("body_position"),
        table.field("body_rotation"),
        table.field("body_linear_velocity"),
        table.field("body_angular_velocity"),
    )

    assert table.frames.root_storage == "body_row_zero"
    assert "observation" not in table.frames.available_fields
    assert table.reference_frame_names == tuple(f"body_{index}" for index in range(24))
    assert expert.frames.shape == (sum(_SMPL_LENGTHS), 358)
    torch.testing.assert_close(expert.frames, expected)


def test_robot_tracking_projectors_own_named_protocol_geometry() -> None:
    smpl = torch.arange(2 * 358, dtype=torch.float32).view(2, 358)
    g1 = torch.arange(2 * 29, dtype=torch.float32).view(2, 29)

    smpl_pose = smpl_humenv_tracking_pose(smpl)
    g1_pose = g1_bfm_observation_state_pose(g1)

    assert smpl_pose.shape == (2, 214) and smpl_pose.untyped_storage().data_ptr() == smpl.untyped_storage().data_ptr()
    assert g1_pose.shape == (2, 23) and g1_pose.untyped_storage().data_ptr() == g1.untyped_storage().data_ptr()
    torch.testing.assert_close(smpl_pose, smpl[:, :214])
    torch.testing.assert_close(g1_pose, g1[:, :23])
    with pytest.raises(ValueError, match="358-wide"):
        smpl_humenv_tracking_pose(smpl[:, :-1])
    with pytest.raises(ValueError, match="29-joint"):
        g1_bfm_observation_state_pose(g1[:, :-1])


class _Algorithm:
    def __init__(self, expert: ForwardBackwardExpertBuffer) -> None:
        self.expert = expert
        self.model = torch.nn.Linear(1, 1)
        self.model.target_network = torch.nn.Linear(1, 1)
        self.model.observation_normalizers = torch.nn.ModuleDict({"state": torch.nn.Linear(1, 1)})
        self.model.assignment_orders = []
        self.model.evaluation_envs = []
        self.resets: list[tuple[TensorDict, torch.Tensor]] = []
        self.eval_mode_calls = 0
        self.train_mode_calls = 0
        self.train_mode()

    def eval_mode(self) -> None:
        self.eval_mode_calls += 1
        self.model.eval()

    def train_mode(self) -> None:
        self.train_mode_calls += 1
        self.model.train()
        self.model.target_network.eval()
        self.model.observation_normalizers.eval()

    def evaluation_history(self, _observations: TensorDict) -> None:
        return None

    def process_env_reset(self, observations: TensorDict, reset: torch.Tensor) -> None:
        self.resets.append((observations, reset.clone()))


class _Environment:
    def __init__(self, table: MotionTaskTable, *, fail_reset: bool = False) -> None:
        payload = object.__new__(MotionStatePayload)
        payload.table = table
        payload.sampler = _sampler(table)
        command = object.__new__(StateCommand)
        command.table = table
        command.randomize_command_indices = True
        command._payload = payload
        command.time_left = torch.ones(2)
        command.command_counter = torch.zeros(2, dtype=torch.int64)
        command._update_step = 0
        command_manager = SimpleNamespace(
            active_terms=("motion",),
            get_term=lambda name: command if name == "motion" else None,
        )
        self.unwrapped = SimpleNamespace(
            num_envs=2,
            device=torch.device("cpu"),
            command_manager=command_manager,
            observation_manager=SimpleNamespace(
                _group_obs_class_term_cfgs={"state": []},
                _group_obs_class_instances=[],
                _group_obs_term_cfgs={
                    "state": [SimpleNamespace(history_length=0, noise=None, modifiers=None)],
                },
            ),
            event_manager=SimpleNamespace(
                active_terms={},
                _interval_term_time_left=[],
                _reset_term_last_triggered_step_id=[],
                _reset_term_last_triggered_once=[],
            ),
            curriculum_manager=SimpleNamespace(active_terms=(), _curriculum_state={}),
            common_step_counter=0,
            _sim_step_counter=0,
            episode_length_buf=torch.zeros(2, dtype=torch.int64),
        )
        self.num_envs = 2
        self.device = torch.device("cpu")
        self.reset_count = 0
        self.fail_reset = fail_reset

    def reset(self) -> tuple[TensorDict, dict]:
        self.reset_count += 1
        if self.fail_reset:
            raise RuntimeError("reset failed")
        return TensorDict({"state": torch.zeros(2, 1)}, batch_size=[2]), {}


def _evaluation() -> ForwardBackwardTrackingEvaluation:
    return ForwardBackwardTrackingEvaluation(
        sequence_ids=_CLIP_IDS,
        metric_values={"emd": torch.tensor((0.0, 3.0)), "obs_state_emd": torch.tensor((1.0, 2.0))},
        source_frame_counts=torch.tensor((3, 4), dtype=torch.int64),
        evaluated_frame_counts=torch.tensor((3, 4), dtype=torch.int64),
        coverage_fraction=torch.ones(2),
        duration_seconds=1.25,
    )


def test_tracking_priority_formula_requires_stable_order_and_clamps_emd() -> None:
    evaluation = _evaluation()
    reversed_evaluation = ForwardBackwardTrackingEvaluation(
        sequence_ids=tuple(reversed(_CLIP_IDS)),
        metric_values={name: values.flip(0) for name, values in evaluation.metric_values.items()},
        source_frame_counts=evaluation.source_frame_counts.flip(0),
        evaluated_frame_counts=evaluation.evaluated_frame_counts.flip(0),
        coverage_fraction=evaluation.coverage_fraction.flip(0),
        duration_seconds=evaluation.duration_seconds,
    )
    with pytest.raises(ValueError, match="stable sequence order"):
        forward_backward_tracking_priority_scores(reversed_evaluation, _CLIP_IDS, "cpu", **_PRIORITY_PROTOCOL)

    scores = forward_backward_tracking_priority_scores(evaluation, _CLIP_IDS, "cpu", **_PRIORITY_PROTOCOL)
    torch.testing.assert_close(scores, torch.tensor((2.0, 16.0)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_matches_bfm_pot_operator() -> None:
    """Exact GPU assignment must match BFM-Zero's CPU oracle without a runtime dependency."""
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
        workspace = UniformAssignmentWorkspace(lengths=(count,), device=device, feature_width=width)

        workspace.compute(
            observed_cpu.to(device).unsqueeze(0),
            target_cpu.to(device).unsqueeze(0),
            output,
        )

        assert output.item() == pytest.approx(expected, rel=0.0, abs=2.0e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_matches_frozen_cpu_oracle_for_variable_lengths() -> None:
    """GPU assignment must preserve the frozen exact costs without copying traces to the host."""
    workspace_type = UniformAssignmentWorkspace
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
    output = torch.empty(2, dtype=torch.float64, device=device)

    workspace = workspace_type(lengths=(2, 4), device=device, feature_width=2)
    workspace.compute(observed, target, output)

    torch.testing.assert_close(output.cpu(), torch.tensor((1.0, 1.25), dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_flat_rows_match_dense_variable_length_costs() -> None:
    """Compact observed rows and strided expert-owned targets must preserve exact dense EMD."""
    device = torch.device("cuda:0")
    lengths = (2, 4, 3)
    starts = (0, 2, 6)
    observed_flat = torch.arange(27, dtype=torch.float32, device=device).view(9, 3) * 0.1
    target_storage = torch.zeros(9, 5, dtype=torch.float32, device=device)
    target_storage[:, :3].copy_(observed_flat.flip(0))
    target_flat = target_storage[:, :3]
    dense_observed = torch.zeros(3, 4, 3, device=device)
    dense_target = torch.zeros_like(dense_observed)
    for row, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        dense_observed[row, :length].copy_(observed_flat[start : start + length])
        dense_target[row, :length].copy_(target_flat[start : start + length])
    workspace = UniformAssignmentWorkspace(lengths=lengths, device=device, feature_width=3)
    dense_output = torch.empty(3, dtype=torch.float64, device=device)
    flat_output = torch.empty_like(dense_output)
    start_tensor = torch.tensor(starts, dtype=torch.int64, device=device)

    workspace.compute(dense_observed, dense_target, dense_output)
    workspace.compute_flat(observed_flat, start_tensor, target_flat, start_tensor, flat_output)

    torch.testing.assert_close(flat_output, dense_output, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_uses_stable_lowest_column_ties() -> None:
    """Equal costs must produce the same lowest-column assignment on every launch."""
    device = torch.device("cuda:0")
    values = torch.zeros(1, 4, 1, device=device)
    workspace = UniformAssignmentWorkspace(lengths=(4,), device=device, feature_width=1)
    output = torch.empty(1, dtype=torch.float64, device=device)

    workspace.compute(values, values, output)
    bucket = workspace._buckets[0]

    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(
        bucket.matching[0, 1:5].cpu(),
        torch.tensor((1, 2, 3, 4), dtype=torch.int32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_block_preserves_variable_length_ties_across_launches() -> None:
    """Every cooperative clip block must retain deterministic lowest-column ties."""
    device = torch.device("cuda:0")
    values = torch.zeros(3, 64, 2, device=device)
    workspace = UniformAssignmentWorkspace(lengths=(1, 17, 64), device=device, feature_width=2)
    output = torch.empty(3, dtype=torch.float64, device=device)

    workspace.compute(values, values, output)
    first_scratch = tuple(
        (
            bucket.matching.clone(),
            bucket.potential_rows.clone(),
            bucket.potential_columns.clone(),
        )
        for bucket in workspace._buckets
    )
    workspace.compute(values, values, output)

    torch.testing.assert_close(output, torch.zeros_like(output))
    for bucket, (matching, potential_rows, potential_columns) in zip(
        workspace._buckets,
        first_scratch,
        strict=True,
    ):
        torch.testing.assert_close(bucket.matching, matching)
        torch.testing.assert_close(bucket.potential_rows, potential_rows)
        torch.testing.assert_close(bucket.potential_columns, potential_columns)
        for bucket_row, count in enumerate(bucket.lengths.tolist()):
            torch.testing.assert_close(
                bucket.matching[bucket_row, 1 : count + 1].cpu(),
                torch.arange(1, count + 1, dtype=torch.int32),
            )


def test_gpu_uniform_emd_compute_reuses_preallocated_scratch() -> None:
    """The repeated assignment path must not allocate, synchronize, or repair layouts."""
    constructor = inspect.getsource(UniformAssignmentWorkspace.__init__)
    source = inspect.getsource(UniformAssignmentWorkspace.compute)
    flat_source = inspect.getsource(UniformAssignmentWorkspace.compute_flat)

    for forbidden in (".cpu(", ".tolist(", ".item(", ".contiguous("):
        assert forbidden not in constructor
    for forbidden in ("torch.empty", "torch.zeros", "torch.tensor", ".cpu(", ".tolist(", ".item(", ".contiguous("):
        assert forbidden not in source
        assert forbidden not in flat_source


def test_gpu_uniform_emd_dispatches_scalar_and_cooperative_kernels_by_bucket() -> None:
    """Exact assignment must avoid cooperative barrier overhead on short buckets."""
    compute_source = inspect.getsource(UniformAssignmentWorkspace.compute)
    scalar_source = inspect.getsource(uniform_emd_module.uniform_assignment_cost_scalar)
    cooperative_source = inspect.getsource(uniform_emd_module.uniform_assignment_cost)
    module_source = inspect.getsource(uniform_emd_module)

    assert uniform_emd_module.UNIFORM_ASSIGNMENT_BLOCK_DIM == 256
    assert UniformAssignmentWorkspace._SCALAR_FRAME_EXTENT_MAX == 512
    assert "bucket.frame_extent <= self._SCALAR_FRAME_EXTENT_MAX" in compute_source
    assert "wp.launch(" in compute_source
    assert "wp.launch_tiled(" in compute_source
    assert "dim=[bucket_size]" in compute_source
    assert "block_dim=UNIFORM_ASSIGNMENT_BLOCK_DIM" in compute_source
    assert "batch = wp.tid()" in scalar_source
    assert "batch, lane = wp.tid()" in cooperative_source
    assert "@wp.func_native" in module_source
    assert "__syncthreads()" in module_source
    assert "other_value == own_value" in module_source
    assert "other_column < own_column" in module_source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_hybrid_dispatch_is_exact_across_boundary() -> None:
    """Scalar 512-frame and cooperative 513-frame buckets must preserve the same exact solution."""
    device = torch.device("cuda:0")
    max_frames = 513
    values = torch.arange(max_frames, dtype=torch.float32, device=device).view(1, max_frames, 1)
    observed = values.expand(2, -1, -1).contiguous()
    target = observed.clone()
    output = torch.empty(2, dtype=torch.float64, device=device)
    workspace = UniformAssignmentWorkspace(lengths=(512, 513), device=device, feature_width=1)

    workspace.compute(observed, target, output)

    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_workspace_uses_fixed_power_of_two_buckets() -> None:
    """Immutable clip lengths must determine compact dense buffers once at construction."""
    device = torch.device("cuda:0")

    workspace = UniformAssignmentWorkspace(lengths=(27, 34, 64, 129, 499), device=device, feature_width=3)

    assert workspace.capacity == 5
    assert workspace.max_frames == 499
    assert tuple(bucket.frame_group_bound for bucket in workspace._buckets) == (32, 64, 256, 512)
    assert tuple(bucket.frame_extent for bucket in workspace._buckets) == (27, 64, 129, 499)
    assert tuple(bucket.row_indices.numel() for bucket in workspace._buckets) == (1, 2, 1, 1)
    assert sum(bucket.cost.numel() for bucket in workspace._buckets) == 274_563
    assert not hasattr(workspace, "cost")
    assert not hasattr(workspace, "observed_square")
    assert not hasattr(workspace, "target_square")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_bucket_compute_reuses_every_device_allocation() -> None:
    """Repeated bucket calls must not grow current or peak device allocation after warmup."""
    device = torch.device("cuda:0")
    workspace = UniformAssignmentWorkspace(lengths=(27, 34, 64, 129), device=device, feature_width=3)
    observed = torch.randn(4, 129, 3, device=device)
    target = torch.randn_like(observed)
    output = torch.empty(4, dtype=torch.float64, device=device)
    workspace.compute(observed, target, output)
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    workspace.compute(observed, target, output)
    workspace.compute(observed, target, output)
    torch.cuda.synchronize(device)

    assert torch.cuda.memory_allocated(device) == baseline
    assert torch.cuda.max_memory_allocated(device) == baseline


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_gpu_uniform_emd_flat_compute_reuses_every_device_allocation() -> None:
    """Repeated compact-row transport must reuse all bucket and gather storage."""
    device = torch.device("cuda:0")
    lengths = (27, 34, 64, 129)
    starts = (0, 27, 61, 125)
    workspace = UniformAssignmentWorkspace(lengths=lengths, device=device, feature_width=3)
    observed = torch.randn(sum(lengths), 3, device=device)
    target_storage = torch.randn(sum(lengths), 5, device=device)
    target = target_storage[:, :3]
    start_tensor = torch.tensor(starts, dtype=torch.int64, device=device)
    output = torch.empty(4, dtype=torch.float64, device=device)
    workspace.compute_flat(observed, start_tensor, target, start_tensor, output)
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    workspace.compute_flat(observed, start_tensor, target, start_tensor, output)
    workspace.compute_flat(observed, start_tensor, target, start_tensor, output)
    torch.cuda.synchronize(device)

    assert torch.cuda.memory_allocated(device) == baseline
    assert torch.cuda.max_memory_allocated(device) == baseline


def test_forward_backward_tracking_ownership_is_generic() -> None:
    """Production tracking, curriculum, metric, and sampling code have one clear owner each."""
    assert forward_backward_tracking_evaluator.__module__.endswith("rl.rsl_rl.forward_backward_tracking")
    assert ForwardBackwardTrackingEvaluation.__module__.endswith("rl.rsl_rl.forward_backward_tracking")
    assert ForwardBackwardTrackingCurriculum.__module__.endswith("rl.rsl_rl.forward_backward_tracking")
    assert UniformAssignmentWorkspace.__module__.endswith("multi_task.metrics.uniform_assignment")
    tracking_source = inspect.getsource(tracking_module)
    tree = ast.parse(tracking_source)
    symbols = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }
    literals = tuple(
        node.value.lower() for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    comparisons = {ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Compare)}
    assert "_native_tracking_assignment" not in symbols
    assert {
        "ForwardBackwardSequenceCommand",
        "ForwardBackwardEvaluationHistory",
        "ForwardBackwardEvaluationHistoryFactory",
        "ForwardBackwardEvaluationScope",
        "ForwardBackwardTrackingMetric",
    }.isdisjoint(symbols)
    assert {"_SequenceCommand", "_EvaluationHistory", "_EvaluationScope", "_TrackingMetric"} <= symbols
    assert "len(metrics) > 2" not in comparisons
    assert all(all(token not in value for value in literals) for token in ("g1", "lafan", "smpl", "cmu"))
    assert all(
        all(token not in value for value in literals)
        for token in ("obs_state_emd", "uniform_before_source_end", "sample_step_seconds")
    )
    assert "json" not in tracking_source.lower()
    assert "pathlib" not in tracking_source.lower()


def test_tracking_assignment_is_driven_by_shuffle_choice() -> None:
    """Assignment policy must be explicit rather than hidden in the evaluator."""
    python_state = random.getstate()
    random.seed(17)
    try:
        assigned, representatives = tracking_module._tracking_assignment(
            8, 3, torch.device("cpu"), random.Random(17), shuffle=True
        )
    finally:
        random.setstate(python_state)

    torch.testing.assert_close(assigned, torch.tensor((0, 1, 0, 1, 2, 2, 1, 0)))
    torch.testing.assert_close(representatives, torch.tensor((0, 1, 4)))


def test_tracking_evaluation_omits_absent_profile_diagnostics() -> None:
    """A single-metric profile must not fabricate an observation-state diagnostic."""
    evaluation = ForwardBackwardTrackingEvaluation(
        sequence_ids=("clip",),
        metric_values={"emd": torch.tensor((1.25,), dtype=torch.float64)},
        source_frame_counts=torch.tensor((3,), dtype=torch.int64),
        evaluated_frame_counts=torch.tensor((3,), dtype=torch.int64),
        coverage_fraction=torch.ones(1, dtype=torch.float64),
        duration_seconds=0.5,
    )

    assert set(evaluation.metric_values) == {"emd"}


def test_smpl_tracking_accepts_exact_t_minus_one_episode_horizon() -> None:
    """Reset frame zero is not a metric row, so T source frames require exactly T-1 actions."""
    metric, evaluated = tracking_module.tracking_frame_counts(
        torch.tensor((4,), dtype=torch.int64),
        episode_length=3,
        include_reset_frame=False,
        allow_horizon_truncation=False,
    )

    torch.testing.assert_close(metric, torch.tensor((3,), dtype=torch.int64))
    torch.testing.assert_close(evaluated, metric)
    with pytest.raises(RuntimeError, match="horizon is shorter"):
        tracking_module.tracking_frame_counts(
            torch.tensor((4,), dtype=torch.int64),
            episode_length=2,
            include_reset_frame=False,
            allow_horizon_truncation=False,
        )


def test_tracking_final_frame_uses_pre_reset_observation() -> None:
    """Same-step autoreset must not replace a valid final metric row with the new spawn."""
    metric = tracking_module._TrackingMetric("distance", torch.empty(1, 2), "policy")
    reached = torch.empty(2, 2)
    current = TensorDict({"policy": torch.tensor(((1.0, 1.0), (2.0, 2.0)))}, batch_size=[2])
    final = TensorDict({"policy": torch.tensor(((3.0, 3.0), (4.0, 4.0)))}, batch_size=[2])
    required_final = torch.tensor((False, True))
    final = tracking_module._tracking_final_observations(
        {"final_obs": final, "final_obs_valid": torch.tensor((False, True))}, required_final, 2
    )

    tracking_module._write_tracking_reached_values(reached, metric, current, final, required_final)

    torch.testing.assert_close(reached, torch.tensor(((1.0, 1.0), (4.0, 4.0))))


def test_tracking_real_wrapper_final_observation_uses_required_done_rows() -> None:
    """The RSL wrapper forwards ``final_obs`` without inventing a validity side channel."""
    metric = tracking_module._TrackingMetric("distance", torch.empty(1, 2), "policy")
    reached = torch.empty(2, 2)
    current = TensorDict({"policy": torch.tensor(((1.0, 1.0), (2.0, 2.0)))}, batch_size=[2])
    required_final = torch.tensor((False, True))
    final = tracking_module._tracking_final_observations(
        {"final_obs": {"policy": torch.tensor(((3.0, 3.0), (4.0, 4.0)))}}, required_final, 2
    )

    tracking_module._write_tracking_reached_values(reached, metric, current, final, required_final)

    torch.testing.assert_close(reached, torch.tensor(((1.0, 1.0), (4.0, 4.0))))


def test_tracking_consumed_done_row_requires_final_observation() -> None:
    with pytest.raises(RuntimeError, match="missing its exact final observation"):
        tracking_module._tracking_final_observations({}, torch.tensor((False, True)), 2)


def test_tracking_consumed_done_row_requires_valid_final_observation() -> None:
    final = TensorDict({"policy": torch.zeros(2, 2)}, batch_size=[2])

    with pytest.raises(RuntimeError, match="no valid final observation"):
        tracking_module._tracking_final_observations(
            {"final_obs": final, "final_obs_valid": torch.tensor((False, False))},
            torch.tensor((False, True)),
            2,
        )


def test_tracking_ignored_duplicate_done_row_does_not_require_final_observation() -> None:
    metric = tracking_module._TrackingMetric("distance", torch.empty(1, 2), "policy")
    observations = TensorDict(
        {"policy": torch.tensor(((1.0, 2.0), (3.0, 4.0)))},
        batch_size=[2],
    )
    reached = torch.empty(2, 2)

    required_final = torch.zeros(2, dtype=torch.bool)
    final = tracking_module._tracking_final_observations({}, required_final, 2)
    tracking_module._write_tracking_reached_values(reached, metric, observations, final, required_final)

    torch.testing.assert_close(reached, observations["policy"])


def test_tracking_projection_uses_one_named_field_for_live_and_final_rows() -> None:
    """Metric ownership stays in observations instead of robot, command, or action adapters."""
    metric = tracking_module._TrackingMetric(
        "distance", torch.empty(1, 2), "joint_position", lambda values: values[:, :2]
    )
    observations = TensorDict({"joint_position": torch.tensor(((1.0, 2.0, 3.0),))}, batch_size=[1])

    torch.testing.assert_close(metric.observe(observations), torch.tensor(((1.0, 2.0),)))


def test_tracking_evaluator_does_not_copy_rollout_traces_to_host() -> None:
    """The production evaluator must keep rollout traces and EMD calculation on the GPU."""
    source = inspect.getsource(forward_backward_tracking_evaluator)
    context_source = inspect.getsource(tracking_module._tracking_context_table)

    for forbidden in (".cpu(", ".tolist(", ".item("):
        assert forbidden not in source
        assert forbidden not in context_source


def test_tracking_context_table_materializes_the_first_sequence_row() -> None:
    """A one-frame context window must define every row, including each clip's reset row."""
    expert = _g1_tracking_expert()
    model = _TrackingModel()

    contexts = tracking_module._tracking_context_table(model, expert, window_length=1)
    expected = model.backward_map(tracking_module.expert_frame_tensordict(model, expert.frames))

    torch.testing.assert_close(contexts, expected)
    for offset in expert.clip_offsets[:-1]:
        torch.testing.assert_close(contexts[offset], expected[offset])


def test_tracking_evaluator_uses_configured_assignment_for_complete_chunks() -> None:
    """The shared evaluator must route assignment policy through its explicit input."""
    source = inspect.getsource(forward_backward_tracking_evaluator)

    assert "_tracking_assignment(" in source
    assert "shuffle=shuffle_assignments" in source
    assert "representative_env_rows" in source
    assert "reset_tracking_sequences(env, command, sequence_start_rows, assigned_expert)" in source
    assert "env_ids=" not in source
    assert "lengths=evaluated_length_values" in source
    assert "device=expert.device" in source
    assert "workspace.compute_flat(observed, observed_starts, metric.target_frames, target_starts, output)" in source
    assert "torch.empty(env.num_envs, max_frames" not in source
    assert "target_traces" not in source


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
def test_concrete_tracking_evaluator_writes_stable_native_metrics() -> None:
    """One shared evaluator must reset exact clips and retain BFM-Zero's full and diagnostic metrics."""
    expert = _g1_tracking_expert("cuda:0")
    model = _TrackingModel()
    env = _TrackingEnvironment(expert)

    evaluation = forward_backward_tracking_evaluator(
        model,
        env,
        expert,
        _CLIP_IDS,
        command=env.command,
        history_factory=lambda _observations: None,
        sequence_start_rows=env.command.table.clip_start_rows.to(expert.device),
        projections=_G1_TRACKING_PROJECTIONS,
        assignment_rng=random.Random(17),
        **_TRACKING_PROTOCOL,
    )

    assert evaluation.sequence_ids == _CLIP_IDS
    assert env.reset_assignments == [((0,), (0,)), ((0,), (1,))]
    assert model.deterministic_calls == [True] * 5
    expected_emd = torch.zeros(2, dtype=torch.float64, device=expert.device)
    torch.testing.assert_close(evaluation.metric_values["emd"], expected_emd)
    torch.testing.assert_close(evaluation.evaluated_frame_counts, torch.tensor((3, 4), device=expert.device))
    torch.testing.assert_close(evaluation.coverage_fraction, torch.ones(2, dtype=torch.float64, device=expert.device))
    torch.testing.assert_close(
        evaluation.metric_values["obs_state_emd"],
        expected_emd,
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production EMD backend.")
def test_uniform_emd_is_invariant_to_one_shared_feature_translation() -> None:
    """Relative and absolute joint coordinates must produce the same assignment cost."""
    device = torch.device("cuda:0")
    workspace = UniformAssignmentWorkspace(lengths=(4,), device=device, feature_width=3)
    observed = torch.tensor((((0.0, 1.0, 2.0), (2.0, 3.0, 4.0), (1.0, 5.0, 2.0), (4.0, 0.0, 3.0)),), device=device)
    target = torch.tensor((((1.0, 1.0, 1.0), (2.0, 4.0, 3.0), (0.0, 5.0, 2.0), (5.0, 0.0, 4.0)),), device=device)
    translation = torch.tensor((1000.0, -2000.0, 3000.0), device=device)
    relative = torch.empty(1, dtype=torch.float64, device=device)
    absolute = torch.empty_like(relative)

    workspace.compute(observed, target, relative)
    workspace.compute(observed + translation, target + translation, absolute)

    torch.testing.assert_close(absolute, relative, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production evaluator.")
def test_native_tracking_evaluator_retains_more_envs_than_clips() -> None:
    """The 1024-env G1 curriculum topology requires shuffled clip replicas, not packed refill."""
    expert = _g1_tracking_expert("cuda:0")
    model = _TrackingModel()
    env = _TrackingEnvironment(expert, num_envs=3)
    evaluation = forward_backward_tracking_evaluator(
        model,
        env,
        expert,
        _CLIP_IDS,
        command=env.command,
        history_factory=lambda _observations: None,
        sequence_start_rows=env.command.table.clip_start_rows.to(expert.device),
        projections=_G1_TRACKING_PROJECTIONS,
        assignment_rng=random.Random(17),
        **_TRACKING_PROTOCOL,
    )

    assert evaluation.sequence_ids == _CLIP_IDS
    assert len(env.reset_assignments) == 1
    assert env.reset_assignments[0][0] == (0, 1, 2)
    assert set(env.reset_assignments[0][1]) == {0, 1}
    assert model.deterministic_calls == [True] * 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production evaluator.")
def test_native_tracking_evaluator_keeps_vecenv_action_clipping() -> None:
    """Evaluation actions must pass through the same VecEnv clipping boundary as training."""
    expert = _g1_tracking_expert("cuda:0")
    model = _TrackingModel(action_value=2.0)
    env = _TrackingEnvironment(expert, clip_actions=0.25)

    forward_backward_tracking_evaluator(
        model,
        env,
        expert,
        _CLIP_IDS,
        command=env.command,
        history_factory=lambda _observations: None,
        sequence_start_rows=env.command.table.clip_start_rows.to(expert.device),
        projections=_G1_TRACKING_PROJECTIONS,
        assignment_rng=random.Random(17),
        **_TRACKING_PROTOCOL,
    )

    assert env.applied_actions
    assert all(torch.all(action == 0.25) for action in env.applied_actions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production evaluator.")
def test_tracking_caps_long_clips_before_the_native_timeout_edge() -> None:
    """Cross-composed clips must report prefix coverage instead of resetting mid-metric."""
    expert = _g1_tracking_expert("cuda:0")
    model = _TrackingModel()
    env = _TrackingEnvironment(expert)
    env.max_episode_length = 3

    evaluation = forward_backward_tracking_evaluator(
        model,
        env,
        expert,
        _CLIP_IDS,
        command=env.command,
        history_factory=lambda _observations: None,
        sequence_start_rows=env.command.table.clip_start_rows.to(expert.device),
        projections=_G1_TRACKING_PROJECTIONS,
        assignment_rng=random.Random(17),
        **_TRACKING_PROTOCOL,
    )

    torch.testing.assert_close(evaluation.source_frame_counts, torch.tensor((3, 4), device=expert.device))
    torch.testing.assert_close(evaluation.evaluated_frame_counts, torch.tensor((3, 4), device=expert.device))
    torch.testing.assert_close(evaluation.coverage_fraction, torch.ones(2, dtype=torch.float64, device=expert.device))


def _curriculum_environment(
    base_priorities: torch.Tensor | None = None,
) -> tuple[_Environment, ForwardBackwardExpertBuffer, _Algorithm]:
    table = _table()
    env = _Environment(table)
    priorities = env.unwrapped.command_manager.get_term("motion").payload.sampler.clip_priorities
    if base_priorities is not None:
        priorities.copy_(base_priorities)
    expert = _expert(priorities)
    return env, expert, _Algorithm(expert)


def _curriculum(env: _Environment, algorithm: _Algorithm) -> ForwardBackwardTrackingCurriculum:
    return ForwardBackwardTrackingCurriculum(
        env,
        algorithm,
        "cpu",
        projections=_G1_TRACKING_PROJECTIONS,
        **_CURRICULUM_PROTOCOL,
        reset_source_name=_MOTION_RESET_SOURCES[0][0],
    )


def test_curriculum_updates_one_shared_priority_owner_then_resets() -> None:
    env, expert, algorithm = _curriculum_environment()
    coordinator = _curriculum(env, algorithm)
    sampler_priorities = env.unwrapped.command_manager.get_term("motion").payload.sampler.clip_priorities

    observations = coordinator.update()

    assert expert.priorities.data_ptr() == sampler_priorities.data_ptr()
    torch.testing.assert_close(expert.priorities, torch.tensor((2.0, 16.0)))
    torch.testing.assert_close(expert._eligible_priorities[1], torch.tensor((2.0, 16.0)))
    assert env.reset_count == 1
    assert observations is algorithm.resets[0][0]
    assert torch.all(algorithm.resets[0][1])


def test_curriculum_multiplies_immutable_base_priorities_without_compounding() -> None:
    env, expert, algorithm = _curriculum_environment(torch.tensor((0.25, 1.0)))
    coordinator = _curriculum(env, algorithm)

    coordinator.update()
    torch.testing.assert_close(expert.priorities, torch.tensor((0.5, 16.0)))

    coordinator.update()
    torch.testing.assert_close(expert.priorities, torch.tensor((0.5, 16.0)))


def test_curriculum_checkpoint_restores_dynamic_priorities_and_retains_base_mass() -> None:
    env, expert, algorithm = _curriculum_environment(torch.tensor((0.25, 1.0)))
    coordinator = _curriculum(env, algorithm)
    coordinator.update()
    expert_state = expert.state_dict()
    curriculum_state = coordinator.state_dict()

    restored_env, restored_expert, restored_algorithm = _curriculum_environment(torch.tensor((0.25, 1.0)))
    restored_coordinator = _curriculum(restored_env, restored_algorithm)
    restored_expert.load_state_dict(expert_state)
    restored_coordinator.load_state_dict(curriculum_state)

    torch.testing.assert_close(restored_expert.priorities, torch.tensor((0.5, 16.0)))
    restored_coordinator.update()
    torch.testing.assert_close(restored_expert.priorities, torch.tensor((0.5, 16.0)))


def test_curriculum_uses_wrapper_and_restores_algorithm_owned_modes() -> None:
    env, _expert_buffer, algorithm = _curriculum_environment()
    coordinator = _curriculum(env, algorithm)

    coordinator.update()

    assert algorithm.model.evaluation_envs == [env]
    assert algorithm.eval_mode_calls == 1
    assert algorithm.train_mode_calls == 2
    assert algorithm.model.training
    assert not algorithm.model.target_network.training
    assert not algorithm.model.observation_normalizers.training


def test_curriculum_validates_priorities_before_mutating_shared_state(monkeypatch: pytest.MonkeyPatch) -> None:
    env, expert, algorithm = _curriculum_environment()
    coordinator = _curriculum(env, algorithm)
    original = expert.priorities.clone()

    def reject(_priorities: torch.Tensor) -> None:
        raise ValueError("expert rejected priorities")

    monkeypatch.setattr(expert, "validate_priorities", reject)
    with pytest.raises(ValueError, match="expert rejected"):
        coordinator.update()

    torch.testing.assert_close(expert.priorities, original)


def test_curriculum_reset_failure_keeps_committed_sampler_state() -> None:
    table = _table()
    env = _Environment(table, fail_reset=True)
    priorities = env.unwrapped.command_manager.get_term("motion").payload.sampler.clip_priorities
    expert = _expert(priorities)
    coordinator = _curriculum(env, _Algorithm(expert))

    with pytest.raises(RuntimeError, match="reset failed"):
        coordinator.update()

    torch.testing.assert_close(expert.priorities, torch.tensor((2.0, 16.0)))
    assert env.reset_count == 1


def test_curriculum_requires_the_vecenv_reset_pair() -> None:
    env, expert, algorithm = _curriculum_environment()
    env.reset = lambda: TensorDict({"state": torch.zeros(2, 1)}, batch_size=[2])
    coordinator = _curriculum(env, algorithm)

    with pytest.raises(TypeError, match="VecEnv TensorDict/info pair"):
        coordinator.update()

    torch.testing.assert_close(expert.priorities, torch.tensor((2.0, 16.0)))


def test_curriculum_checkpoint_owns_rng_but_not_shared_priorities() -> None:
    env, expert, algorithm = _curriculum_environment()
    coordinator = _curriculum(env, algorithm)
    coordinator.update()
    state = coordinator.state_dict()
    expert.set_priorities(torch.ones(2))

    coordinator.load_state_dict(state)

    assert set(state) == {"protocol_hash", "sequence_ids", "assignment_rng_state"}
    torch.testing.assert_close(expert.priorities, torch.ones(2))


def test_curriculum_assignment_rng_advances_and_checkpoint_restores_next_mapping() -> None:
    env, _expert_buffer, algorithm = _curriculum_environment()
    coordinator = _curriculum(env, algorithm)

    coordinator.update()
    first_order = algorithm.model.assignment_orders[-1]
    state = coordinator.state_dict()
    coordinator.update()
    expected_next_order = algorithm.model.assignment_orders[-1]

    assert expected_next_order != first_order

    restored_env, _restored_expert, restored_algorithm = _curriculum_environment()
    restored = _curriculum(restored_env, restored_algorithm)
    restored.load_state_dict(state)
    restored.update()

    assert restored_algorithm.model.assignment_orders[-1] == expected_next_order


def test_curriculum_checkpoint_rejects_changed_protocol() -> None:
    env, _expert_buffer, algorithm = _curriculum_environment()
    coordinator = _curriculum(env, algorithm)
    state = coordinator.state_dict()
    state["protocol_hash"] = "0" * 64

    with pytest.raises(ValueError, match="protocol"):
        coordinator.load_state_dict(state)
