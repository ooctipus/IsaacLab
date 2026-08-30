# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
pytest.importorskip("tensordict")

from tensordict import TensorDict  # noqa: E402

from isaaclab_tasks.core.multi_task.rl.rsl_rl.models.categorical_value import (  # noqa: E402
    CategoricalResidualMLPEncoderModel,
)
from isaaclab_tasks.core.multi_task.rl.rsl_rl.models.simba_v2 import (  # noqa: E402
    FeatureScaler,
    HypersphericalLERPBlock,
    HypersphericalLinear,
    SimbaV2Head,
)
from isaaclab_tasks.core.multi_task.rl.rsl_rl.models.simba_v2_encoder_model import (  # noqa: E402
    CategoricalSimbaV2EncoderModel,
    SimbaV2EncoderModel,
)

_OBS_GROUPS = {"actor": ["policy", "height_scan"], "critic": ["policy", "height_scan"]}
_ENCODER_CFG = {"height_scan": {"output_dim": 8, "hidden_dims": [16], "activation": "elu"}}


def _make_obs(batch_size: int) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, 6),
            "height_scan": torch.randn(batch_size, 1, 4, 5),
        },
        batch_size=[batch_size],
    )


class TestSimbaV2Blocks:
    def test_backbone_features_are_unit_norm_and_gradients_are_finite(self):
        head = SimbaV2Head(
            input_dim=7,
            output_dim=3,
            hidden_dim=16,
            num_blocks=2,
            expansion=4,
            simplicial_group_size=4,
        )
        inputs = torch.randn(9, 7, requires_grad=True)

        features = head.encode(inputs)
        torch.testing.assert_close(torch.linalg.vector_norm(features, dim=-1), torch.ones(9))

        output = head(inputs)
        output.square().mean().backward()
        assert inputs.grad is not None
        assert torch.isfinite(inputs.grad).all()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in head.parameters())

    def test_source_matched_scaler_and_lerp_initialization(self):
        scaler = FeatureScaler(8, init=0.5, scale=0.25)
        torch.testing.assert_close(scaler(torch.ones(8)), torch.full((8,), 0.5))

        block = HypersphericalLERPBlock(hidden_dim=16, num_blocks=2)
        effective_alpha = block.alpha.gain * block.alpha.forward_scale
        torch.testing.assert_close(effective_alpha, torch.full((16,), 1.0 / 3.0))

    def test_weight_projection_restores_unit_rows_after_optimizer_step(self):
        head = SimbaV2Head(input_dim=7, output_dim=3, hidden_dim=16, num_blocks=1)
        optimizer = torch.optim.Adam(head.parameters(), lr=0.1)
        loss = head(torch.randn(32, 7)).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        linear_layers = [module for module in head.modules() if isinstance(module, HypersphericalLinear)]
        assert any(
            not torch.allclose(torch.linalg.vector_norm(layer.weight, dim=1), torch.ones(layer.output_dim))
            for layer in linear_layers
        )

        head.project_hyperspherical_weights_()
        for layer in linear_layers:
            torch.testing.assert_close(
                torch.linalg.vector_norm(layer.weight, dim=1),
                torch.ones(layer.output_dim),
            )


class TestSimbaV2EncoderModel:
    def test_actor_with_sem_preserves_rsl_rl_contract(self):
        actor = SimbaV2EncoderModel(
            _make_obs(4),
            _OBS_GROUPS,
            "actor",
            output_dim=5,
            hidden_dim=16,
            num_blocks=1,
            obs_normalization=True,
            encoder_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
            encoder_cfg=_ENCODER_CFG,
            simplicial_group_size=4,
        )

        output = actor(_make_obs(7))
        assert output.shape == (7, 5)
        assert torch.isfinite(output).all()
        output.square().mean().backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in actor.parameters())

    def test_rejects_non_gaussian_ppo_distribution(self):
        with pytest.raises(ValueError, match="state-independent GaussianDistribution"):
            SimbaV2EncoderModel(
                _make_obs(4),
                _OBS_GROUPS,
                "actor",
                output_dim=5,
                hidden_dim=16,
                encoder_cfg=_ENCODER_CFG,
                distribution_cfg={
                    "class_name": "HeteroscedasticGaussianDistribution",
                    "init_std": 1.0,
                    "std_type": "log",
                },
            )

    @pytest.mark.parametrize(
        "model_cls",
        [CategoricalResidualMLPEncoderModel, CategoricalSimbaV2EncoderModel],
    )
    def test_categorical_critics_return_expectations_and_expose_logits(self, model_cls):
        kwargs = {
            "hidden_dim": 16,
            "num_blocks": 1,
            "obs_normalization": True,
            "encoder_normalization": True,
            "encoder_cfg": _ENCODER_CFG,
            "num_bins": 11,
            "value_min": -2.0,
            "value_max": 2.0,
        }
        if model_cls is CategoricalResidualMLPEncoderModel:
            kwargs.update(expand=2, activation="relu", norm=True)
        else:
            kwargs.update(expansion=2)
        critic = model_cls(_make_obs(4), _OBS_GROUPS, "critic", output_dim=1, **kwargs)

        obs = _make_obs(7)
        logits = critic.get_value_logits(obs)
        values = critic(obs)

        assert logits.shape == (7, 11)
        assert values.shape == (7, 1)
        torch.testing.assert_close(values, critic.value_from_logits(logits))
        assert torch.all(values >= -2.0)
        assert torch.all(values <= 2.0)

    @pytest.mark.parametrize(
        "model_cls",
        [CategoricalResidualMLPEncoderModel, CategoricalSimbaV2EncoderModel],
    )
    def test_rejects_sem_on_critic(self, model_cls):
        kwargs = {
            "hidden_dim": 16,
            "num_blocks": 1,
            "encoder_cfg": _ENCODER_CFG,
            "simplicial_group_size": 4,
        }
        if model_cls is CategoricalResidualMLPEncoderModel:
            kwargs.update(expand=2, activation="relu", norm=True)
        else:
            kwargs.update(expansion=2)
        with pytest.raises(ValueError, match="SEM is actor-only"):
            model_cls(_make_obs(4), _OBS_GROUPS, "critic", output_dim=1, **kwargs)

    def test_reward_scaled_categorical_critic_requires_symmetric_support(self):
        with pytest.raises(ValueError, match="symmetric around zero"):
            CategoricalSimbaV2EncoderModel(
                _make_obs(4),
                _OBS_GROUPS,
                "critic",
                output_dim=1,
                hidden_dim=16,
                num_blocks=1,
                encoder_cfg=_ENCODER_CFG,
                num_bins=11,
                value_min=-1.0,
                value_max=2.0,
                reward_scaling=True,
            )

    @pytest.mark.parametrize(
        ("actor_name", "critic_name", "categorical"),
        [
            ("SIMBA_V2_ACTOR", "SIMBA_V2_CRITIC", False),
            ("SIMBA_V2_SEM_ACTOR", "SIMBA_V2_CRITIC", False),
            ("SIMBA_V2_SEM_ACTOR", "SIMBA_V2_CATEGORICAL_CRITIC", True),
        ],
    )
    def test_presets_construct_through_runner_pipeline(self, actor_name, critic_name, categorical):
        from importlib import metadata

        from rsl_rl.utils import resolve_callable

        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        from isaaclab_tasks.core.multi_task.terrain.config import rsl_rl_model_cfg
        from isaaclab_tasks.core.multi_task.terrain.config.rsl_rl_cfg import PositionLocomotionPPORunnerCfg

        runner = PositionLocomotionPPORunnerCfg()
        runner.actor = getattr(rsl_rl_model_cfg, actor_name).copy()
        runner.critic = getattr(rsl_rl_model_cfg, critic_name).copy()
        runner = handle_deprecated_rsl_rl_cfg(runner, metadata.version("rsl-rl-lib"))
        cfg = runner.to_dict()

        actor_kwargs = dict(cfg["actor"])
        actor_cls = resolve_callable(actor_kwargs.pop("class_name"))
        actor = actor_cls(_make_obs(4), _OBS_GROUPS, "actor", 5, **actor_kwargs)

        critic_kwargs = dict(cfg["critic"])
        critic_cls = resolve_callable(critic_kwargs.pop("class_name"))
        critic = critic_cls(_make_obs(4), _OBS_GROUPS, "critic", 1, **critic_kwargs)

        assert actor(_make_obs(3)).shape == (3, 5)
        assert critic(_make_obs(3)).shape == (3, 1)
        assert bool(getattr(critic, "is_categorical_value", False)) is categorical
