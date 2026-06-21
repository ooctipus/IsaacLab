# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""z-conditioned actor for the forward-backward successor critic (Meta-Motivo): ``pi(a | s, z)``.

The goal is removed from the actor's observation (the goal-relative delta is gone; the obs carries the robot's
absolute env-local pose instead). The goal enters ONLY through ``z = B(goal)`` -- the same backward embedding the
critic's value/FB use -- so the policy MUST consult ``z`` to know where to head. This mirrors the critic's
z-conditioned forward map: a shared SimBa encoder builds the goal-free latent ``h``, and the policy head reads
``[h, z]``. The caller passes ``z`` DETACHED (the actor trains its policy params via the PPO surrogate; the
backward map is trained reward-free by the FB measure, not through the actor).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from .residual_mlp import ResidualMLP
from .residual_mlp_encoder_model import ResidualMLPEncoderModel


class SuccessorActorModel(ResidualMLPEncoderModel):
    """SimBa encoder + a z-conditioned Gaussian policy head ``pi(a | [h(s), z])``.

    Args:
        obs: A sample observation ``TensorDict`` (for sizing).
        obs_groups: The ``{set: [groups]}`` mapping; the encoder reads ``obs_groups[obs_set]`` (goal-free, no delta).
        obs_set: The observation set this model reads (``"actor"``).
        output_dim: Action dimension.
        feature_dim: Width ``d`` of the goal embedding ``z`` (must equal the critic's ``feature_dim``).
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        expand: int = 4,
        activation: str = "swish",
        norm: bool = True,
        last_activation: str | None = None,
        **kwargs,
    ) -> None:
        if kwargs.get("memory") is not None:
            raise ValueError("SuccessorActorModel is feedforward only (no recurrent memory).")
        # The parent builds the shared encoder + latent, the Gaussian distribution, and a policy head on the
        # latent. We then rebuild that head to additionally take the goal embedding z (mirrors the critic's
        # psi_head taking [h, z]); the distribution is unchanged.
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            expand=expand,
            activation=activation,
            norm=norm,
            last_activation=last_activation,
            **kwargs,
        )
        self.feature_dim = feature_dim
        latent_dim = self._get_latent_dim()
        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        # Rebuild the policy head to read [latent, z] (same shape/init as the parent head, wider input).
        self.mlp = ResidualMLP(
            latent_dim + feature_dim,
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

    def forward(
        self,
        obs: TensorDict,
        z: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Action from ``pi(a | s, z)``: encode ``s``, append the goal embedding ``z``, run the policy head.

        ``z`` (``[B, feature_dim]``, the goal embedding ``B(goal)``, detached by the caller) conditions the
        policy. Returns a sampled action (``stochastic_output``) or the distribution mean. Updates
        ``self.distribution`` so the PPO log-prob / entropy / KL accessors read the z-conditioned distribution.
        """
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(torch.cat([latent, z], dim=-1))
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output
