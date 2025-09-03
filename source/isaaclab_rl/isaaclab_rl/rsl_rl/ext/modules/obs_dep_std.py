# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal, Distribution, constraints
from typing import Optional
from torch.types import _size

from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from ...rl_cfg import RslRlPpoActorCriticCfg

class StateDependentNoiseDistribution(Distribution):
    """
    Distribution class for using generalized State Dependent Exploration (gSDE).
    Paper: https://arxiv.org/abs/2005.05719
    """
    has_rsample = True
    arg_constraints = {
        'mean_actions': constraints.real,
        'log_std': constraints.real,
        'latent_sde': constraints.real
    }

    def __init__(
        self,
        action_dim: int,
        epsilon: float = 1e-6,
        batch_shape: torch.Size = torch.Size(),
        event_shape: torch.Size = torch.Size(),
        validate_args: Optional[bool] = None,
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.distribution = None
        self._latent_sde = None
        self.exploration_mat = None
        self.exploration_matrices = None
        self.weights_dist = None
        super().__init__(batch_shape, event_shape, validate_args)

    def get_std(self, log_std: torch.Tensor) -> torch.Tensor:
        return torch.exp(log_std)

    def sample_weights(self, log_std: torch.Tensor, batch_size: int = 1) -> None:
        std = self.get_std(log_std)
        self.weights_dist = Normal(torch.zeros_like(std), std)
        self.exploration_mat = self.weights_dist.rsample()
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def proba_distribution(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        latent_sde: torch.Tensor
    ) -> 'StateDependentNoiseDistribution':
        # cache for sampling path
        self._latent_sde = latent_sde
        self._mean_actions = mean_actions
        # Move exploration matrices to same device/dtype as latent_sde
        if self.exploration_mat is not None:
            self.exploration_mat = self.exploration_mat.to(latent_sde.device, latent_sde.dtype)
        if self.exploration_matrices is not None:
            self.exploration_matrices = self.exploration_matrices.to(latent_sde.device, latent_sde.dtype)
        variance = torch.mm(self._latent_sde**2, self.get_std(log_std) ** 2)
        self.distribution = Normal(mean_actions, torch.sqrt(variance + self.epsilon))
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(actions)
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        if self.distribution is None or self._latent_sde is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        noise = self.get_noise(self._latent_sde)  # phi(s) @ W (or batched)
        if sample_shape != torch.Size():
            noise = noise.unsqueeze(0).expand(sample_shape + noise.shape)
            mean = self.distribution.mean.unsqueeze(0).expand_as(noise)
            return mean + noise
        return self.distribution.mean + noise

    def sample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)

    @property
    def mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        return self.distribution.mean

    @property
    def mode(self) -> torch.Tensor:
        if self.distribution is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        return self.distribution.mean

    @property
    def variance(self) -> torch.Tensor:
        if self.distribution is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        return self.distribution.variance

    @property
    def stddev(self) -> torch.Tensor:
        if self.distribution is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        return self.distribution.stddev

    @property
    def support(self) -> constraints.Constraint:
        return constraints.real

    def expand(self, batch_shape: _size, _instance=None) -> 'StateDependentNoiseDistribution':
        new = self._get_checked_instance(StateDependentNoiseDistribution, _instance)
        new.action_dim = self.action_dim
        new.epsilon = self.epsilon
        new.distribution = self.distribution
        new._latent_sde = self._latent_sde
        new.exploration_mat = self.exploration_mat
        new.exploration_matrices = self.exploration_matrices
        new.weights_dist = self.weights_dist
        super(StateDependentNoiseDistribution, new).__init__(batch_shape, self._event_shape, validate_args=False)
        return new

    def get_noise(self, latent_sde: torch.Tensor) -> torch.Tensor:
        if self.exploration_matrices is None or len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return torch.mm(latent_sde, self.exploration_mat)
        latent_sde = latent_sde.unsqueeze(dim=1)
        noise = torch.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(dim=1)


class StateDependendNoiseDistributionPatcher:
    
    def __init__(self, policy_cfg: RslRlPpoActorCriticCfg):
        if policy_cfg.noise_std_type == "gsde":
            self.apply_patch()

    def apply_patch(self):
        
        self._orignal_actor_critic_init = ActorCritic.__init__
        self._original_update_distribution = ActorCritic.update_distribution
        self._original_ppo_update = PPO.update
        
        def state_dependent_std_init(actor_critic_self, *args, **kwargs):
            # init base with a supported type
            kwargs["noise_std_type"] = "scalar"
            self._orignal_actor_critic_init(actor_critic_self, *args, **kwargs)
            actor_critic_self.noise_std_type = "gsde"

            num_actions = args[2]
            init_noise_std = kwargs.get("init_noise_std", 1.0)
            actor_hidden_dims = kwargs.get("actor_hidden_dims")
            
            # split actor into body/head without using seq slicing
            layers = list(actor_critic_self.actor.children())
            actor_critic_self._gsde_body = nn.Sequential(*layers[:-1])
            actor_critic_self._gsde_head = layers[-1]  # last Linear
            actor_critic_self.distribution = StateDependentNoiseDistribution(action_dim=num_actions)
            actor_critic_self.log_std = nn.Parameter(torch.ones(actor_hidden_dims[-1], num_actions) * torch.log(torch.tensor(init_noise_std)))
            actor_critic_self.distribution.sample_weights(actor_critic_self.log_std)

        def state_dependent_dist_resampled_update(ppo_self: PPO):
            loss_dict = self._original_ppo_update(ppo_self)
            ppo_self.policy.distribution.sample_weights(ppo_self.policy.log_std, batch_size=ppo_self.storage.num_envs)
            return loss_dict

        def state_dependent_update_distribution(actor_critic_self, observation):
            features = actor_critic_self._gsde_body(observation)
            mean = actor_critic_self._gsde_head(features)
            actor_critic_self.distribution.proba_distribution(mean, actor_critic_self.log_std, features)

        ActorCritic.__init__ = state_dependent_std_init
        ActorCritic.update_distribution = state_dependent_update_distribution
        PPO.update = state_dependent_dist_resampled_update
    
    def remove_patch(self):
        ActorCritic.__init__ = self._orignal_actor_critic_init
        ActorCritic.update_distribution = self._original_update_distribution