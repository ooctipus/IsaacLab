# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SimBa model: per-group encoders -> (optional RNN) -> residual MLP head.

A single model covers the feedforward and recurrent SimBa variants and both encoder types. The data
flow per environment step is::

    obs groups --(per-group MLP and/or CNN encoders + obs norm)--> latent
              --(LayerNorm)--> [RNN(lstm|gru)] --> ResidualMLP head --> distribution

Every encoded observation group is routed through a per-group encoder declared in ``encoder_cfg``.
Both encoder kinds are just "encoders": an MLP encoder flattens its input feature dimension, while a
CNN encoder consumes a ``(C, H, W)`` image directly; an entry is treated as a CNN when it carries conv
parameters (``output_channels``) and as an MLP otherwise. The per-group latents are concatenated with
any passthrough groups. When ``memory`` is ``None`` the RNN is omitted and the residual head consumes
the encoder latent directly (the feedforward SimBa). When ``memory`` is provided, a recurrent cell is
inserted between the LayerNorm-normalized encoder latent and the residual head. Because the residual
head is pre-norm and the RNN input is LayerNorm-normalized, the recurrent path is well-conditioned --
unlike a bare ``nn.LSTM`` head, which is the usual reason recurrent policies underperform a strong
feedforward baseline. Keeping the variants in one model makes "memory vs no memory" and "MLP vs CNN
encoder" single config choices rather than separate architectures.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import CNN, MLP, RNN, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from .mlp_encoder_model import MLPEncoderModel
from .mlp_encoder_model import mlp_output_dim as get_mlp_output_dim
from .residual_mlp import ResidualMLP


def _init_lstm_forget_bias(rnn_module: nn.Module, value: float) -> None:
    """Bias the LSTM forget gate toward retention.

    PyTorch packs the four gates as ``(input, forget, cell, output)``; the forget slice is the second
    quarter of each bias vector. Initializing it positive encourages the cell to keep information across
    long horizons (Jozefowicz et al. 2015), which matters for tasks whose memory span exceeds a few
    steps.
    """
    for name, param in rnn_module.named_parameters():
        if name.startswith("bias_ih"):
            hidden = param.shape[0] // 4
            with torch.no_grad():
                param[hidden : 2 * hidden].fill_(value)


class ResidualMLPEncoderModel(MLPEncoderModel):
    """MLPEncoderModel with a residual MLP head and an optional recurrent memory cell."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        expand: int = 4,
        activation: str = "relu",
        norm: bool = True,
        last_activation: str | None = None,
        obs_normalization: bool = False,
        encoder_normalization: bool = False,
        head_layer_norm: bool = False,
        memory: dict[str, Any] | None = None,
        distribution_cfg: dict | None = None,
        encoder_cfg: dict[str, dict[str, Any]] | None = None,
        # ``cnns`` is rsl-rl's encoder-SHARING hook, not a CNN-specific input. When PPO runs with
        # ``share_cnn_encoders=True`` it injects the actor's already-built per-group encoders here
        # (``cfg["critic"]["cnns"] = actor.cnns``) so the critic reuses them instead of building its own.
        # The shared modules are general per-group encoders (MLP or CNN) -- the "cnn" name is rsl-rl's
        # and we keep it for interop. Set by the runner, never by model config.
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
        hidden_dims: list[int] | tuple[int, ...] | None = None,  # noqa: ARG002
    ) -> None:
        if encoder_cfg is None and cnns is None:
            encoder_cfg = {}
        nn.Module.__init__(self)
        if cnns is not None:
            self._encoded_obs_group_keys = list(cnns.keys())
        else:
            self._encoded_obs_group_keys = list(encoder_cfg.keys())
        # An encoder entry is a CNN when it declares conv channels; otherwise it is a flattening MLP.
        self._cnn_groups = (
            {group for group, cfg in encoder_cfg.items() if "output_channels" in cfg}
            if encoder_cfg is not None
            else set()
        )
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        # Full feature shapes (excluding the leading batch dim) per encoded group, captured at
        # construction. They let ``get_latent`` reshape encoders correctly under both the 2D inference
        # layout (``(B, *feat)``) and the padded training layout (``(T, N, *feat)``), and give CNN
        # encoders their ``(C, H, W)`` input shape.
        self._encoded_feature_shapes = [tuple(obs[group].shape[1:]) for group in self.obs_groups_encoded]
        if cnns is not None:
            if set(cnns.keys()) != set(self.obs_groups_encoded):
                raise ValueError("Shared encoders must match encoded observation groups.")
            encoders: dict[str, nn.Module] = dict(cnns)
        else:
            if set(self._encoded_obs_group_keys) != set(self.obs_groups_encoded):
                raise ValueError("encoder_cfg keys must match encoded observation groups.")
            encoders = {}
            for i, obs_group in enumerate(self.obs_groups_encoded):
                if obs_group in self._cnn_groups:
                    channels, height, width = self._encoded_feature_shapes[i]
                    encoders[obs_group] = CNN(
                        input_dim=(height, width), input_channels=channels, **encoder_cfg[obs_group]
                    )
                else:
                    encoders[obs_group] = MLP(input_dim=self.obs_dims_encoded[i], **encoder_cfg[obs_group])
        self.obs_normalization = obs_normalization
        self.obs_normalizer = EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
        self.encoders = encoders if isinstance(encoders, nn.ModuleDict) else nn.ModuleDict(encoders)
        # Alias the encoders under ``.cnns`` so rsl-rl's ``share_cnn_encoders`` path can read them via
        # ``actor.cnns``. These are general per-group encoders (MLP or CNN), not necessarily CNNs.
        self.cnns = self.encoders
        self.encoder_normalization = encoder_normalization
        # CNN-encoded groups carry image observations (already normalized upstream), so they keep an
        # Identity normalizer; only the flattened MLP-encoded groups get an EmpiricalNormalization.
        self.encoder_normalizers = nn.ModuleDict(
            {
                obs_group: (
                    EmpiricalNormalization(self.obs_dims_encoded[i])
                    if encoder_normalization and obs_group not in self._cnn_groups
                    else nn.Identity()
                )
                for i, obs_group in enumerate(self.obs_groups_encoded)
            }
        )
        self.encoder_latent_dim = sum(self._encoder_output_dim(group) for group in self.obs_groups_encoded)
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim
        latent_dim = self._get_latent_dim()
        self.head_norm = nn.LayerNorm(latent_dim) if head_layer_norm else nn.Identity()

        # Optional recurrent memory inserted between the encoder latent and the residual head. When set,
        # the head consumes the RNN hidden state rather than the encoder latent.
        self.rnn: RNN | None = None
        head_input_dim = latent_dim
        if memory is not None:
            rnn_type = memory.get("rnn_type", "lstm")
            rnn_hidden_dim = memory.get("hidden_dim", 256)
            self.rnn = RNN(latent_dim, rnn_hidden_dim, memory.get("num_layers", 1), rnn_type)
            self.is_recurrent = True
            head_input_dim = rnn_hidden_dim
            forget_bias = memory.get("forget_bias", 1.0)
            if forget_bias is not None and rnn_type.lower() == "lstm":
                _init_lstm_forget_bias(self.rnn.rnn, forget_bias)

        self.mlp = ResidualMLP(
            head_input_dim,
            mlp_output_dim,
            hidden_dim,
            num_blocks,
            expand,
            2,
            activation,
            last_activation,
            norm,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def _encoder_output_dim(self, group: str) -> int:
        """Return the flattened latent width an encoded group contributes to the head input."""
        encoder = self.encoders[group]
        if group in self._cnn_groups:
            return int(encoder.output_dim)  # type: ignore[attr-defined]
        return get_mlp_output_dim(encoder)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select passthrough and encoded groups, allowing identity-only SimBa heads."""
        passthrough_groups: list[str] = []
        passthrough_dim = 0
        encoded_groups: list[str] = []
        encoded_dims: list[int] = []
        for obs_group in obs_groups[obs_set]:
            shape = obs[obs_group].shape
            feature_dim = math.prod(shape[1:])
            if obs_group in self._encoded_obs_group_keys:
                encoded_groups.append(obs_group)
                encoded_dims.append(int(feature_dim))
            else:
                if len(shape) != 2:
                    raise ValueError(
                        f"Observation '{obs_group}' must be encoded or flattened before ResidualMLPEncoderModel."
                    )
                passthrough_groups.append(obs_group)
                passthrough_dim += int(shape[-1])
        missing = set(self._encoded_obs_group_keys) - set(encoded_groups)
        if missing:
            raise ValueError(f"encoder_cfg contains observation groups not present in {obs_set}: {sorted(missing)}")
        self.obs_groups_encoded = encoded_groups
        self.obs_dims_encoded = encoded_dims
        return passthrough_groups, passthrough_dim

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Encode observation groups, normalize, then optionally run the recurrent cell.

        Works for both the 2D inference layout (``(B, *feat)``) and the padded training layout
        (``(T, N, *feat)``). Encoders are applied with the leading (time/batch) dimensions folded into a
        single batch dimension and then restored, so a recurrent head receives a proper sequence. CNN
        groups keep their ``(C, H, W)`` structure; MLP groups are flattened.
        """
        encoded = []
        for i, group in enumerate(self.obs_groups_encoded):
            x = obs[group]
            feature_shape = self._encoded_feature_shapes[i]
            lead_shape = x.shape[: x.ndim - len(feature_shape)]
            if group in self._cnn_groups:
                features = self.encoders[group](x.reshape(-1, *feature_shape))
            else:
                flat = x.reshape(-1, self.obs_dims_encoded[i])
                features = self.encoders[group](self.encoder_normalizers[group](flat))
            encoded.append(features.reshape(*lead_shape, features.shape[-1]))
        latent_encoded = torch.cat(encoded, dim=-1) if encoded else None

        if self.obs_groups:
            latent_passthrough = self.obs_normalizer(torch.cat([obs[group] for group in self.obs_groups], dim=-1))
            latent = (
                torch.cat([latent_passthrough, latent_encoded], dim=-1)
                if latent_encoded is not None
                else latent_passthrough
            )
        else:
            if latent_encoded is None:
                raise RuntimeError("ResidualMLPEncoderModel requires at least one passthrough or encoded group.")
            latent = latent_encoded

        latent = self.head_norm(latent)
        if self.rnn is not None:
            latent = self.rnn(latent, masks, hidden_state).squeeze(0)
        return latent

    def update_normalization(self, obs: TensorDict) -> None:
        """Update running normalization stats for the passthrough obs and MLP-encoded groups.

        CNN-encoded groups keep an :class:`~torch.nn.Identity` normalizer (their images are already
        normalized upstream by the observation term), so they are skipped here -- unlike the inherited
        :class:`MLPEncoderModel.update_normalization`, which would call ``.update()`` on every encoder
        normalizer and fail on the Identity.
        """
        if self.obs_normalization and self.obs_groups:
            self.obs_normalizer.update(torch.cat([obs[group] for group in self.obs_groups], dim=-1))
        if self.encoder_normalization:
            for i, group in enumerate(self.obs_groups_encoded):
                if group in self._cnn_groups:
                    continue
                self.encoder_normalizers[group].update(obs[group].reshape(-1, self.obs_dims_encoded[i]))

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if self.rnn is not None:
            self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        return self.rnn.hidden_state if self.rnn is not None else None  # type: ignore[return-value]

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self.rnn is not None:
            self.rnn.detach_hidden_state(dones)
