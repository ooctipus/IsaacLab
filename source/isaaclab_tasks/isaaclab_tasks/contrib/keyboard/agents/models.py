# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL models for the SO-101 keyboard-typing task."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, HiddenState
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict


class SharedEncoderMLPModel(MLPModel):
    """Encode selected 1-D observation groups before the MLP head."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_cfg: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the encoded-observation MLP model."""
        if not encoder_cfg:
            raise ValueError("At least one encoder configuration must be provided.")

        active_obs_groups = obs_groups[obs_set]
        encoder_keys = set(encoder_cfg)
        if not encoder_keys.issubset(active_obs_groups):
            invalid_groups = sorted(encoder_keys - set(active_obs_groups))
            raise ValueError(
                f"The encoder observation groups {invalid_groups} are not part of the '{obs_set}' observation groups"
                f" {active_obs_groups}."
            )

        self.encoder_obs_groups = [group for group in active_obs_groups if group in encoder_keys]
        self.encoder_input_dims = []
        for obs_group in self.encoder_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The MLP encoders only support 1D observations, got shape {obs[obs_group].shape} for"
                    f" '{obs_group}'."
                )
            self.encoder_input_dims.append(obs[obs_group].shape[-1])

        encoders = {}
        self.encoder_latent_dim = 0
        for obs_group, input_dim in zip(self.encoder_obs_groups, self.encoder_input_dims):
            group_cfg = dict(encoder_cfg[obs_group])
            latent_dim = group_cfg["latent_dim"]
            encoders[obs_group] = MLP(
                input_dim=input_dim,
                output_dim=latent_dim,
                hidden_dims=group_cfg["hidden_dims"],
                activation=group_cfg.get("activation", "elu"),
                last_activation=group_cfg.get("last_activation"),
            )
            self.encoder_latent_dim += latent_dim

        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        self.encoders = nn.ModuleDict(encoders)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent from raw observations and encoded groups."""
        latents = [self.encoders[group](obs[group]) for group in self.encoder_obs_groups]
        if self.obs_groups:
            latents.insert(0, super().get_latent(obs))
        return torch.cat(latents, dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update normalization statistics of non-encoded observation groups."""
        if self.obs_groups:
            super().update_normalization(obs)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchSharedEncoderModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxSharedEncoderModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select non-encoded observation groups and compute their total dimension."""
        active_obs_groups = obs_groups[obs_set]
        raw_obs_groups = []
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The MLP model only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            if obs_group not in self.encoder_obs_groups:
                raw_obs_groups.append(obs_group)
                obs_dim += obs[obs_group].shape[-1]
        return raw_obs_groups, obs_dim

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + self.encoder_latent_dim


def _arm_observations(
    robot_state: torch.Tensor, robot_active: torch.Tensor, key_positions: torch.Tensor, key_active: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world observations into active-arm rows, with scalar-first root quaternions."""
    active = robot_active > 0
    indices = active.nonzero()
    worlds, arms = indices[:, 0], indices[:, 1]
    # Clear padding before geometry: absent bodies may contain invalid poses or NaNs.
    states = torch.where(active.unsqueeze(-1), robot_state, 0.0)
    own, other = states[worlds, arms], states[worlds, 1 - arms]
    other_active = active[worlds, 1 - arms].unsqueeze(-1)
    keys_active = key_active > 0
    keys = torch.where(keys_active.unsqueeze(-1), key_positions, 0.0)[worlds]
    keys_active = keys_active[worlds]

    inverse = own[:, 3:7] / own[:, 3:7].square().sum(-1, keepdim=True).clamp_min(1.0e-9)
    scalar, imaginary = inverse[:, :1], -inverse[:, 1:]
    positions = torch.cat((other[:, None, :3], keys), dim=1) - own[:, None, :3]
    imaginary_expanded = imaginary.unsqueeze(1).expand_as(positions)
    cross = 2.0 * torch.cross(imaginary_expanded, positions, dim=-1)
    positions = positions + scalar.unsqueeze(1) * cross + torch.cross(imaginary_expanded, cross, dim=-1)
    other_rotation = torch.cat(
        (
            scalar * other[:, 3:4] - (imaginary * other[:, 4:7]).sum(-1, keepdim=True),
            scalar * other[:, 4:7] + other[:, 3:4] * imaginary + torch.cross(imaginary, other[:, 4:7], dim=-1),
        ),
        dim=-1,
    )
    other_features = torch.cat((positions[:, 0], other_rotation, other[:, 7:]), dim=-1)
    raw = torch.cat(
        (
            own[:, 7:],
            torch.where(other_active, other_features, 0.0),
            other_active.to(own.dtype),
            torch.where(keys_active.unsqueeze(-1), positions[:, 1:], 0.0).flatten(1),
            keys_active.to(own.dtype),
        ),
        dim=-1,
    )
    return raw, worlds


class SharedArmMLPModel(SharedEncoderMLPModel):
    """One SO-101 policy evaluated once per active arm, with world-major PPO samples.

    Observations contain ``policy[N, P]``, ``robot_state[N, 2, 25]``,
    ``robot_active[N, 2]``, ``key_positions[N, K, 3]`` and ``key_active[N, K]``.
    Each robot state holds root position [m], quaternion (wxyz), six joint
    positions [rad], velocities [rad/s], and previous actions. Each perspective
    sees its own joint state, the other arm's root pose in its own root frame
    and joint state, and root-relative key positions. Absolute root poses stay
    in the world observations and do not enter the MLP. Only active perspectives
    enter the shared MLP.

    The actor returns twelve padded world actions, with six Gaussian standard
    deviations shared across arms. The critic averages active perspective values
    into one world value. Neither rewards nor rollout rows are replicated.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_cfg: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Construct an active-arm actor or a pooled world critic."""
        if output_dim != (12 if distribution_cfg is not None else 1):
            raise ValueError("Shared-arm models require twelve actor outputs or one critic output.")
        if encoder_cfg is None or set(encoder_cfg) != {"policy"}:
            raise ValueError("Shared-arm models encode exactly the world-level 'policy' observation group.")
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            6 if distribution_cfg is not None else 1,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
            encoder_cfg,
        )
        if self.distribution is not None and type(self.distribution) is not GaussianDistribution:
            raise ValueError("Shared-arm actors require a state-independent GaussianDistribution.")

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        groups = ["policy", "robot_state", "robot_active", "key_positions", "key_active"]
        if set(obs_groups[obs_set]) != set(groups) or len(obs_groups[obs_set]) != len(groups):
            raise ValueError(f"Shared-arm models require observation groups {groups}.")
        if obs["key_positions"].ndim != 3 or obs["key_positions"].shape[-1] != 3:
            raise ValueError("Key positions must have shape [worlds, keys, 3].")
        self.num_keys = obs["key_positions"].shape[1]
        expected_shapes = {"robot_state": (2, 25), "robot_active": (2,), "key_active": (self.num_keys,)}
        for group, shape in expected_shapes.items():
            if tuple(obs[group].shape[1:]) != shape:
                raise ValueError(f"Observation '{group}' must have trailing shape {shape}.")
        return groups[1:], 44 + 4 * self.num_keys

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Encode each typing sequence once and gather only active arm perspectives."""
        raw, worlds = _arm_observations(
            obs["robot_state"], obs["robot_active"], obs["key_positions"], obs["key_active"]
        )
        return torch.cat((self.obs_normalizer(raw), self.encoders["policy"](obs["policy"])[worlds]), dim=-1)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Evaluate active arms and return world actions or one world value."""
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        active = obs["robot_active"] > 0
        outputs = self.mlp(self.get_latent(obs))
        outputs = outputs.new_zeros((active.shape[0], 2, outputs.shape[-1])).masked_scatter(
            active.unsqueeze(-1), outputs
        )
        if self.distribution is None:
            return outputs.sum(1) / active.sum(-1, keepdim=True).clamp_min(1)
        if stochastic_output:
            self._arm_active = active
            self.distribution.update(outputs)
            outputs = self.distribution.sample()
        return torch.where(active.unsqueeze(-1), outputs, 0.0).flatten(1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Learn normalization statistics from active perspectives only."""
        if self.obs_normalization:
            raw, _ = _arm_observations(obs["robot_state"], obs["robot_active"], obs["key_positions"], obs["key_active"])
            self.obs_normalizer.update(raw)

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return the joint likelihood of the world's active arm actions."""
        actions = torch.where(self._arm_active.unsqueeze(-1), outputs.reshape(-1, 2, 6), 0.0)
        return torch.where(self._arm_active, self.distribution.log_prob(actions), 0.0).sum(-1)

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return entropy summed over active arm action distributions."""
        return torch.where(self._arm_active, self.distribution.entropy, 0.0).sum(-1)

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Keep world rows, including their active mask, in PPO rollout storage."""
        return (*self.distribution.params, self._arm_active)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Sum KL only over arms present in the stored world transition."""
        active = old_params[2] > 0
        mask = active.unsqueeze(-1)
        old = (torch.where(mask, old_params[0], 0.0), torch.where(mask, old_params[1], 1.0))
        new = (torch.where(mask, new_params[0], 0.0), torch.where(mask, new_params[1], 1.0))
        return torch.where(active, self.distribution.kl_divergence(old, new), 0.0).sum(-1)

    def as_jit(self) -> nn.Module:
        """Export dynamic active-arm projection and deterministic world output."""
        return _SharedArmExport(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Export the same five world observation inputs to ONNX."""
        return _SharedArmExport(self, verbose)


class SharedEncoderPPO(PPO):
    """Share the actor's observation encoders with the critic."""

    def __init__(self, actor: MLPModel, critic: MLPModel, storage: RolloutStorage, **kwargs: Any) -> None:
        """Replace the critic's encoders before PPO registers optimizer parameters."""
        if not isinstance(actor, SharedEncoderMLPModel) or not isinstance(critic, SharedEncoderMLPModel):
            raise TypeError("SharedEncoderPPO requires SharedEncoderMLPModel actor and critic models.")
        if (
            actor.encoder_obs_groups != critic.encoder_obs_groups
            or actor.encoder_latent_dim != critic.encoder_latent_dim
        ):
            raise ValueError("The actor and critic encoder configurations must match.")

        # The actor owns the shared modules so optimizer, checkpoint, and gradient traversal see them once.
        del critic.encoders
        object.__setattr__(critic, "encoders", actor.encoders)
        super().__init__(actor, critic, storage, **kwargs)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, Any]
    ) -> None:
        """Bootstrap explicit autoreset truncations from their pre-reset observations."""
        if "final_obs" in extras and "time_outs" in extras:
            # Evaluate the nonrecurrent critic in the rollout's normalization frame.
            with torch.no_grad():
                final = TensorDict(extras["final_obs"], batch_size=obs.batch_size).to(self.device)
                values = self.critic(final).squeeze(-1)
                timeouts = extras["time_outs"].to(device=self.device, dtype=torch.bool)
                rewards = rewards + self.gamma * torch.where(timeouts, values, 0.0)
            extras = {key: value for key, value in extras.items() if key != "time_outs"}
        super().process_env_step(obs, rewards, dones, extras)


class _TorchSharedEncoderModel(nn.Module):
    """Exportable shared-encoder model for TorchScript."""

    def __init__(self, model: SharedEncoderMLPModel) -> None:
        """Create a TorchScript-compatible model copy."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoders = nn.ModuleList([copy.deepcopy(model.encoders[g]) for g in model.encoder_obs_groups])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_raw: torch.Tensor, obs_encoded: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from raw and encoded-group inputs."""
        latents = [self.obs_normalizer(obs_raw)]
        for i, encoder in enumerate(self.encoders):
            latents.append(encoder(obs_encoded[i]))
        return self.deterministic_output(self.mlp(torch.cat(latents, dim=-1)))

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state."""
        pass


class _OnnxSharedEncoderModel(_TorchSharedEncoderModel):
    """Exportable shared-encoder model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: SharedEncoderMLPModel, verbose: bool) -> None:
        """Create an ONNX-compatible model copy."""
        super().__init__(model)
        self.verbose = verbose
        self.encoder_obs_groups = list(model.encoder_obs_groups)
        self.encoder_input_dims = list(model.encoder_input_dims)
        self.obs_dim_raw = model.obs_dim

    def forward(self, obs: torch.Tensor, *obs_encoded: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        return super().forward(obs, list(obs_encoded))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        dummy_raw = torch.zeros(1, self.obs_dim_raw)
        dummy_encoded = [torch.zeros(1, dim) for dim in self.encoder_input_dims]
        return (dummy_raw, *dummy_encoded)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", *self.encoder_obs_groups]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]


class _SharedArmExport(nn.Module):
    """Deterministic shared-arm inference with the same world observation interface."""

    is_recurrent: bool = False

    def __init__(self, model: SharedArmMLPModel, verbose: bool = False) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.encoders["policy"])
        self.mlp = copy.deepcopy(model.mlp)
        self.is_actor = model.distribution is not None
        self.policy_dim = model.encoder_input_dims[0]
        self.num_keys = model.num_keys
        self.verbose = verbose

    def forward(
        self,
        policy: torch.Tensor,
        robot_state: torch.Tensor,
        robot_active: torch.Tensor,
        key_positions: torch.Tensor,
        key_active: torch.Tensor,
    ) -> torch.Tensor:
        """Gather active perspectives, evaluate the shared head, and restore world rows."""
        raw, worlds = _arm_observations(robot_state, robot_active, key_positions, key_active)
        latent = torch.cat((self.obs_normalizer(raw), self.encoder(policy)[worlds]), dim=-1)
        outputs = self.mlp(latent)
        active = robot_active > 0
        outputs = outputs.new_zeros((active.shape[0], 2, outputs.shape[-1])).masked_scatter(
            active.unsqueeze(-1), outputs
        )
        if self.is_actor:
            return outputs.flatten(1)
        return outputs.sum(1) / active.sum(-1, keepdim=True).clamp_min(1)

    @torch.jit.export
    def reset(self) -> None:
        """Keep the stateless export compatible with RSL-RL's inference interface."""
        pass

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Provide a valid two-arm world for ONNX tracing."""
        robots = torch.zeros(1, 2, 25)
        robots[..., 3] = 1.0
        return (
            torch.zeros(1, self.policy_dim),
            robots,
            torch.ones(1, 2),
            torch.zeros(1, self.num_keys, 3),
            torch.ones(1, self.num_keys),
        )

    @property
    def input_names(self) -> list[str]:
        """Return the world observation group names in argument order."""
        return ["policy", "robot_state", "robot_active", "key_positions", "key_active"]

    @property
    def output_names(self) -> list[str]:
        """Return the deterministic output name."""
        return ["actions" if self.is_actor else "values"]
