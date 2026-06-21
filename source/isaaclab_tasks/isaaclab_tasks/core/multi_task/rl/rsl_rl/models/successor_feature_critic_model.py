# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""z-conditioned forward-backward critic (Meta-Motivo / FB-CPR): a shared SimBa encoder + a z-conditioned
forward map ``F(s, z)`` and a backward map ``B(s)``.

The goal/command is REMOVED from the observation; it enters only through ``z = B(g)`` (the backward embedding
of the goal state). The encoder builds the goal-free latent ``h = [o_prop ; enc(o_perc)]`` from
``[proprio, perception]``; ``B(s) = sqrt(d) * normalize(mlp(h))`` (hard L2-norm to the ``sqrt(d)`` sphere,
the FB-CPR backward recipe -- removes the gauge freedom and, with the orthonormality penalty
``E[B B^T]=I``, makes ``B`` span ``R^d`` so the measure ``<F, B>`` sees every ``F`` direction);
``F(s, z) = psi_head([h, z])`` is the z-conditioned forward map, left free (its scale is the occupancy scale).
Their inner product ``M(s, z, s') = <F(s,z), B(s')>`` approximates the discounted state-occupancy measure of
the z-optimal policy. Conditioning ``F`` on diverse ``z`` is what bounds it (single-task, un-conditioned
bilinear FB diverges -- the prior gate confirmed this). The model owns the two maps; the reward read-out
``w`` and the value ``V = <F(s,z), w>`` live in :class:`~rsl_rl.extensions.SuccessorFeatures`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from .residual_mlp import ResidualMLP
from .residual_mlp_encoder_model import ResidualMLPEncoderModel


class SuccessorFeatureCriticModel(ResidualMLPEncoderModel):
    """Shared SimBa encoder + ``psi`` (free) and ``phi`` (hard-normed to sqrt(d)) heads for the SR value.

    Args:
        obs: A sample observation ``TensorDict`` (for sizing).
        obs_groups: The ``{set: [groups]}`` mapping; the encoder reads ``obs_groups[obs_set]`` (command folded in).
        obs_set: The observation set this model reads (``"critic"``).
        output_dim: PPO's requested output width; unused -- the value is the extension's ``<psi, w>`` -- and
            accepted only to match the model factory's positional signature.
        feature_dim: Width ``d`` of ``psi`` / ``phi``.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,  # noqa: ARG002 -- value is the extension's <psi, w>; kept for factory signature parity
        feature_dim: int = 128,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        expand: int = 4,
        activation: str = "swish",
        norm: bool = True,
        **kwargs,
    ) -> None:
        if kwargs.get("memory") is not None:
            raise ValueError("SuccessorFeatureCriticModel is feedforward only (no recurrent memory).")
        # The critic is deterministic (its value is the extension's <psi, w>, not a sampled scalar), so drop any
        # distribution cfg the preset carried and force the parent head to a plain feature head.
        kwargs.pop("distribution_cfg", None)
        # The parent builds the shared encoder + latent and a residual head; that head is reused as phi.
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            feature_dim,  # parent head (self.mlp) -> phi in R^d
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            expand=expand,
            activation=activation,
            norm=norm,
            distribution_cfg=None,  # deterministic; the value is the extension's <psi, w>
            **kwargs,
        )
        self.feature_dim = feature_dim
        latent_dim = self._get_latent_dim()
        # F(s, z): the forward map is conditioned on the goal embedding z (R^d) appended to the latent.
        self.psi_head = ResidualMLP(
            latent_dim + feature_dim, feature_dim, hidden_dim, num_blocks, expand, 2, activation
        )

    def backward(self, obs: TensorDict) -> torch.Tensor:
        """Backward map ``B(s)``, ``[B, feature_dim]`` -- hard L2-normed to the ``sqrt(d)`` sphere.

        The FB-CPR backward recipe: per-sample projection onto ``||B|| = sqrt(d)`` removes the gauge freedom and,
        with the orthonormality penalty ``E[B B^T]=I``, makes the batch's ``B``'s span ``R^d`` -- so the bilinear
        measure ``<F, B>`` exposes every direction of ``F`` (no blind subspace). ``z = B(g)`` is the goal channel.
        """
        return math.sqrt(self.feature_dim) * F.normalize(self.mlp(self.get_latent(obs)), dim=-1)

    def forward_map(self, obs: TensorDict, z: torch.Tensor) -> torch.Tensor:
        """z-conditioned forward map ``F(s, z) = psi_head([h(s), z])``, ``[B, feature_dim]``, left FREE.

        ``z`` (``[B, feature_dim]``, the goal embedding ``B(g)``) is appended to the goal-free latent. ``F``'s
        magnitude carries the occupancy scale ``~1/(1-gamma)``; conditioning on diverse ``z`` is what bounds it.
        """
        return self.psi_head(torch.cat([self.get_latent(obs), z], dim=-1))

    def sf_heads(self, obs: TensorDict, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(F(s, z), B(s))``, ``[B, feature_dim]``, sharing ONE encoder pass.

        ``z`` is the goal embedding. Used where both maps are needed on the same states (the value read-out and
        the FB current-state term); the FB next-state term uses :meth:`backward` / :meth:`forward_map` directly.
        """
        h = self.get_latent(obs)
        phi = math.sqrt(self.feature_dim) * F.normalize(self.mlp(h), dim=-1)  # B(s), sqrt(d) sphere
        psi = self.psi_head(torch.cat([h, z], dim=-1))  # F(s, z), free
        return psi, phi

    def forward(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        """The successor value comes from :meth:`SuccessorFeatures.value` (it owns ``w``); use that instead."""
        raise NotImplementedError(
            "SuccessorFeatureCriticModel has no standalone value head; PPO reads the value via"
            " SuccessorFeatures.value(critic, obs) = <F(s,z), w>. Use backward(obs)/forward_map(obs, z)."
        )
