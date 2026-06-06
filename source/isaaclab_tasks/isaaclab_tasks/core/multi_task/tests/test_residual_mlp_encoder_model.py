# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the SimBa model (:class:`ResidualMLPEncoderModel`), feedforward and recurrent.

A single model covers both variants via the ``memory`` config: ``memory=None`` is feedforward,
``memory={...}`` inserts an RNN. The tests cover the two observation layouts the recurrent path must
handle:

* inference: a 2D ``(B, *feat)`` observation per step, with the hidden state advancing across steps,
* recurrent PPO update: padded 3D ``(T, N, *feat)`` trajectories fed with masks and an initial hidden
  state, which is the layout where the per-group encoders must reshape correctly.

A bug in the encoder reshape (e.g. ``flatten(start_dim=1)``) collapses the trajectory dim into the
feature dim and silently corrupts training, so the batch-mode shape parity is the key regression.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
pytest.importorskip("tensordict")

from rsl_rl.utils import split_and_pad_trajectories  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab_tasks.core.multi_task.rl.rsl_rl.models.residual_mlp_encoder_model import (  # noqa: E402
    ResidualMLPEncoderModel,
)

_OBS_GROUPS = {"actor": ["policy", "task", "height_scan"], "critic": ["policy", "task", "height_scan"]}
_POLICY_DIM = 12
_TASK_DIM = 6
_SCAN_SHAPE = (1, 8, 10)
_ENCODER_CFG = {"height_scan": {"output_dim": 16, "hidden_dims": [32, 32], "activation": "elu"}}
_NUM_ACTIONS = 5


def _make_obs(batch_size: int) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, _POLICY_DIM),
            "task": torch.randn(batch_size, _TASK_DIM),
            "height_scan": torch.randn(batch_size, *_SCAN_SHAPE),
        },
        batch_size=[batch_size],
    )


def _make_actor(rnn_type: str = "lstm", rnn_hidden_dim: int = 24, memory: bool = True) -> ResidualMLPEncoderModel:
    memory_cfg = (
        {"rnn_type": rnn_type, "hidden_dim": rnn_hidden_dim, "num_layers": 1, "forget_bias": 1.0} if memory else None
    )
    return ResidualMLPEncoderModel(
        _make_obs(4),
        _OBS_GROUPS,
        "actor",
        _NUM_ACTIONS,
        hidden_dim=32,
        num_blocks=2,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        head_layer_norm=True,
        memory=memory_cfg,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
        encoder_cfg=_ENCODER_CFG,
    )


class TestResidualMLPEncoderModel:
    def test_feedforward_when_memory_is_none(self):
        actor = _make_actor(memory=False)
        assert actor.is_recurrent is False
        assert actor.rnn is None
        assert actor.get_hidden_state() is None
        out = actor(_make_obs(7))
        assert out.shape == (7, _NUM_ACTIONS)
        assert torch.isfinite(out).all()

    def test_model_is_recurrent_when_memory_set(self):
        actor = _make_actor()
        assert actor.is_recurrent is True
        assert actor.get_hidden_state() is None

    @pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
    def test_inference_forward_shape_and_hidden_state_advances(self, rnn_type):
        actor = _make_actor(rnn_type=rnn_type)
        actor.reset()
        batch = 7
        out = actor(_make_obs(batch))
        assert out.shape == (batch, _NUM_ACTIONS)
        # The hidden state must materialize after a step and have the right env dimension.
        hidden = actor.get_hidden_state()
        h = hidden[0] if isinstance(hidden, tuple) else hidden
        assert h.shape[-2] == batch

    def test_recurrent_batch_mode_matches_unpadded_step_count(self):
        # Build a (T, N) rollout, then split/pad into trajectories the way the PPO storage does and
        # verify the model digests the 3D padded layout and returns the original rollout layout.
        steps, num_envs = 6, 4
        flat = _make_obs(steps * num_envs)
        obs_tn = TensorDict(
            {k: v.reshape(steps, num_envs, *v.shape[1:]) for k, v in flat.items()},
            batch_size=[steps, num_envs],
        )
        dones = torch.zeros(steps, num_envs)
        dones[3, 1] = 1.0  # one mid-rollout episode boundary

        padded_obs, masks = split_and_pad_trajectories(obs_tn, dones)

        actor = _make_actor()
        actor.reset()
        # Zero initial hidden state matching the padded trajectory count, as the storage would supply.
        num_traj = padded_obs.shape[1]
        h0 = torch.zeros(1, num_traj, actor.rnn.rnn.hidden_size)
        c0 = torch.zeros(1, num_traj, actor.rnn.rnn.hidden_size)

        # ``unpad_trajectories`` (inside the RNN) reverses the split/pad back to the original
        # ``(T, num_envs, *feat)`` rollout layout so outputs line up with the stored targets.
        out = actor(padded_obs, masks=masks, hidden_state=(h0, c0))
        assert out.shape == (steps, num_envs, _NUM_ACTIONS)
        assert torch.isfinite(out).all()

    def test_reset_zeroes_done_environments(self):
        actor = _make_actor()
        actor.reset()
        actor(_make_obs(4))
        dones = torch.tensor([1.0, 0.0, 1.0, 0.0])
        actor.reset(dones)
        h, c = actor.get_hidden_state()
        assert torch.count_nonzero(h[:, dones == 1]) == 0
        assert torch.count_nonzero(c[:, dones == 1]) == 0

    def test_lstm_forget_bias_initialized(self):
        actor = _make_actor(rnn_type="lstm")
        hidden = actor.rnn.rnn.hidden_size
        forget_slice = actor.rnn.rnn.bias_ih_l0[hidden : 2 * hidden]
        torch.testing.assert_close(forget_slice, torch.ones_like(forget_slice))


_CNN_CFG = {"height_scan": {"output_channels": [8, 16], "kernel_size": [3, 3], "stride": [2, 2], "activation": "elu"}}
_SCAN_HW = (1, 12, 16)


def _cnn_obs(batch_size: int) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, _POLICY_DIM),
            "task": torch.randn(batch_size, _TASK_DIM),
            "height_scan": torch.randn(batch_size, *_SCAN_HW),
        },
        batch_size=[batch_size],
    )


def _make_cnn_actor(memory: bool = False) -> ResidualMLPEncoderModel:
    memory_cfg = {"rnn_type": "lstm", "hidden_dim": 24, "num_layers": 1} if memory else None
    return ResidualMLPEncoderModel(
        _cnn_obs(4),
        _OBS_GROUPS,
        "actor",
        _NUM_ACTIONS,
        hidden_dim=32,
        num_blocks=2,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        head_layer_norm=True,
        memory=memory_cfg,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
        encoder_cfg=_CNN_CFG,
    )


class TestSimbaCnnEncoder:
    """CNN-encoder SimBa: the height_scan group is a 2D ``(C, H, W)`` image routed through a CNN."""

    def test_cnn_only_builds_without_mlp_encoder(self):
        actor = _make_cnn_actor()
        assert actor.is_recurrent is False
        assert set(actor.obs_groups_encoded) == {"height_scan"}
        assert "height_scan" in actor._cnn_groups
        # passthrough groups are the 1D ones
        assert set(actor.obs_groups) == {"policy", "task"}

    def test_cnn_inference_forward_shape(self):
        actor = _make_cnn_actor()
        out = actor(_cnn_obs(7))
        assert out.shape == (7, _NUM_ACTIONS)
        assert torch.isfinite(out).all()

    def test_cnn_recurrent_batch_mode_shape(self):
        steps, num_envs = 6, 4
        flat = _cnn_obs(steps * num_envs)
        obs_tn = TensorDict(
            {k: v.reshape(steps, num_envs, *v.shape[1:]) for k, v in flat.items()},
            batch_size=[steps, num_envs],
        )
        dones = torch.zeros(steps, num_envs)
        dones[3, 1] = 1.0
        padded_obs, masks = split_and_pad_trajectories(obs_tn, dones)

        actor = _make_cnn_actor(memory=True)
        actor.reset()
        num_traj = padded_obs.shape[1]
        h0 = torch.zeros(1, num_traj, actor.rnn.rnn.hidden_size)
        c0 = torch.zeros(1, num_traj, actor.rnn.rnn.hidden_size)
        out = actor(padded_obs, masks=masks, hidden_state=(h0, c0))
        assert out.shape == (steps, num_envs, _NUM_ACTIONS)
        assert torch.isfinite(out).all()

    def test_position_presets_wire_encoder_types(self):
        try:
            from isaaclab_tasks.core.position.config.rsl_rl_cfg import (
                PositionActorPresetCfg,
                PositionCriticPresetCfg,
            )
        except (ImportError, TypeError) as exc:
            # The position config depends on a newer isaaclab_rl (e.g. PPO ``weight_decay``); skip when
            # the importable isaaclab_rl predates it rather than reporting a spurious failure.
            pytest.skip(f"position config not importable in this environment: {exc}")

        def is_cnn(model_cfg):
            return any(hasattr(c, "output_channels") for c in model_cfg.encoder_cfg.values())

        for preset_cls in (PositionActorPresetCfg, PositionCriticPresetCfg):
            presets = preset_cls()
            assert not is_cnn(presets.simba_mlp)
            assert is_cnn(presets.simba_cnn)
            assert not is_cnn(presets.simba_mlp_big)
            assert is_cnn(presets.simba_cnn_big)
            # back-compat aliases point at the MLP variants
            assert not is_cnn(presets.simba)


class TestSimbaPresetPipeline:
    """End-to-end: the SimBa presets must survive the real ``handle_deprecated -> to_dict ->
    construct`` runner pipeline. The feedforward ``SIMBA`` preset is the known-good baseline; running
    the recurrent ``SIMBA_RNN`` preset through the *identical* path validates the ``memory`` wiring,
    including that the nested ``MemoryCfg`` serializes to a plain dict the model accepts.

    Scoped to the actor: the critic presets parameterize ``activation`` with a hydra ``preset(...)``
    that only resolves under a hydra launch, which is orthogonal to the ``memory`` wiring under test.
    """

    @pytest.mark.parametrize("recurrent", [False, True])
    def test_presets_construct_through_runner_pipeline(self, recurrent):
        from importlib import metadata

        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        from rsl_rl.utils import resolve_callable

        from isaaclab_tasks.core.multi_task.terrain.config.rsl_rl_cfg import PositionLocomotionPPORunnerCfg
        from isaaclab_tasks.core.multi_task.terrain.config.rsl_rl_model_cfg import SIMBA_ACTOR, SIMBA_RNN_ACTOR

        actor_cfg = SIMBA_RNN_ACTOR if recurrent else SIMBA_ACTOR
        assert (actor_cfg.memory is not None) == recurrent

        runner = PositionLocomotionPPORunnerCfg()
        runner.actor = actor_cfg.copy()
        # The deprecation pass iterates actor and critic; give the critic a concrete model cfg (the
        # actor's, which uses a plain-string activation) so it doesn't trip on the unresolved
        # PresetCfg dispatcher. We only construct/assert the actor here.
        runner.critic = actor_cfg.copy()
        runner = handle_deprecated_rsl_rl_cfg(runner, metadata.version("rsl-rl-lib"))
        cfg = runner.to_dict()

        # The nested MemoryCfg must serialize to a plain dict (or None) for the model kwargs.
        assert (cfg["actor"].get("memory") is not None) == recurrent
        if recurrent:
            assert isinstance(cfg["actor"]["memory"], dict)

        obs = _make_obs(4)
        groups = {"actor": _OBS_GROUPS["actor"], "critic": _OBS_GROUPS["critic"]}
        model_kwargs = dict(cfg["actor"])
        model_cls = resolve_callable(model_kwargs.pop("class_name"))
        model = model_cls(obs, groups, "actor", _NUM_ACTIONS, **model_kwargs)
        assert model.is_recurrent == recurrent
        assert (model.rnn is not None) == recurrent
