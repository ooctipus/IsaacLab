# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage


def project_categorical_targets(returns: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Project scalar returns onto adjacent atoms of an evenly spaced support."""
    if support.ndim != 1 or support.numel() < 2:
        raise ValueError("Categorical value support must be one-dimensional with at least two atoms.")
    delta = support[1] - support[0]
    if delta <= 0.0:
        raise ValueError("Categorical value support must be strictly increasing.")

    targets = torch.minimum(torch.maximum(returns.squeeze(-1), support[0]), support[-1])
    positions = ((targets - support[0]) / delta).clamp(0.0, support.numel() - 1)
    lower = positions.floor().long().clamp_max(support.numel() - 1)
    upper = positions.ceil().long().clamp_max(support.numel() - 1)
    upper_weight = positions - lower
    lower_weight = 1.0 - upper_weight

    probabilities = returns.new_zeros(*targets.shape, support.numel())
    probabilities.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    probabilities.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return probabilities


class DiscountedReturnScaler:
    """Scale rewards by running discounted-return statistics without centering."""

    def __init__(
        self,
        num_envs: int,
        gamma: float,
        max_normalized_return: float,
        epsilon: float,
        device: torch.device | str,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive; got {num_envs}.")
        if max_normalized_return <= 0.0:
            raise ValueError(f"max_normalized_return must be positive; got {max_normalized_return}.")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive; got {epsilon}.")

        self.gamma = gamma
        self.max_normalized_return = max_normalized_return
        self.epsilon = epsilon
        self.discounted_returns = torch.zeros(num_envs, device=device)
        self.mean = torch.zeros((), device=device)
        self.variance = torch.ones((), device=device)
        self.count = torch.full((), 1.0e-4, device=device)
        self.max_abs_return = torch.zeros((), device=device)

    @property
    def denominator(self) -> torch.Tensor:
        """Current reward divisor."""
        standard_deviation = torch.sqrt(self.variance + self.epsilon)
        range_floor = self.max_abs_return / self.max_normalized_return
        return torch.maximum(standard_deviation, range_floor).clamp_min(self.epsilon)

    @torch.no_grad()
    def update_and_scale(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Update discounted-return statistics and scale the current rewards."""
        reward_vector = rewards.reshape(-1).to(self.discounted_returns)
        done_vector = dones.reshape(-1).bool()
        if reward_vector.shape != self.discounted_returns.shape:
            raise ValueError(f"Expected {self.discounted_returns.numel()} rewards; got shape {tuple(rewards.shape)}.")

        self.discounted_returns.mul_(self.gamma * (~done_vector)).add_(reward_vector)
        self._update_statistics(self.discounted_returns)
        self.max_abs_return.copy_(torch.maximum(self.max_abs_return, self.discounted_returns.abs().max()))
        return rewards / self.denominator

    @torch.no_grad()
    def _update_statistics(self, values: torch.Tensor) -> None:
        batch_count = values.numel()
        batch_mean = values.mean()
        batch_variance = values.var(unbiased=False)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.variance * self.count
        m_b = batch_variance * batch_count
        correction = delta.square() * self.count * batch_count / total_count

        self.mean.copy_(new_mean)
        self.variance.copy_((m_a + m_b + correction) / total_count)
        self.count.copy_(total_count)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return checkpointable running state."""
        return {
            "discounted_returns": self.discounted_returns.clone(),
            "mean": self.mean.clone(),
            "variance": self.variance.clone(),
            "count": self.count.clone(),
            "max_abs_return": self.max_abs_return.clone(),
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Restore running state, tolerating a changed environment count."""
        for name in ("mean", "variance", "count", "max_abs_return"):
            getattr(self, name).copy_(state_dict[name].to(getattr(self, name)))
        saved_returns = state_dict["discounted_returns"]
        if saved_returns.shape == self.discounted_returns.shape:
            self.discounted_returns.copy_(saved_returns.to(self.discounted_returns))
        else:
            self.discounted_returns.zero_()


class ComposablePPO(PPO):
    """PPO with opt-in SimBaV2 projection and categorical value learning.

    Scalar critics follow upstream RSL-RL PPO unchanged. A categorical critic
    enables an experimental PPO value objective using two-hot lambda-return
    targets and SimBaV2 reward scaling.
    """

    actor: MLPModel
    critic: MLPModel

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self._categorical_value = bool(getattr(self._raw_critic, "is_categorical_value", False))
        self._validate_hyperspherical_optimizer()
        self._project_hyperspherical_weights()
        self._projection_hook = self.optimizer.register_step_post_hook(self._after_optimizer_step)

        self.reward_scaler: DiscountedReturnScaler | None = None
        if self._categorical_value and bool(getattr(self._raw_critic, "reward_scaling", False)):
            self.reward_scaler = DiscountedReturnScaler(
                storage.num_envs,
                self.gamma,
                float(getattr(self._raw_critic, "reward_scale_max")),
                float(getattr(self._raw_critic, "reward_scale_epsilon")),
                self.device,
            )

    def _validate_hyperspherical_optimizer(self) -> None:
        has_hyperspherical_model = any(
            bool(getattr(model, "hyperspherical_backbone", False)) for model in (self._raw_actor, self._raw_critic)
        )
        if not has_hyperspherical_model:
            return
        if not isinstance(self.optimizer, torch.optim.Adam) or isinstance(self.optimizer, torch.optim.AdamW):
            raise ValueError("SimBaV2 requires the Adam optimizer without weight decay.")
        if any(float(group.get("weight_decay", 0.0)) != 0.0 for group in self.optimizer.param_groups):
            raise ValueError("SimBaV2 requires zero optimizer weight decay.")

    def _after_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,  # noqa: ARG002
        args: tuple[Any, ...],  # noqa: ARG002
        kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        self._project_hyperspherical_weights()

    @torch.no_grad()
    def _project_hyperspherical_weights(self) -> None:
        for model in (self._raw_actor, self._raw_critic):
            project = getattr(model, "project_hyperspherical_weights_", None)
            if project is not None:
                project()

    def process_env_step(
        self,
        obs,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Scale categorical-value rewards before storing PPO transitions."""
        if self.reward_scaler is not None:
            rewards = self.reward_scaler.update_and_scale(rewards, dones)
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        """Run scalar PPO unchanged or the categorical value-loss variant."""
        if not self._categorical_value:
            return super().update()
        return self._update_categorical()

    def _update_categorical(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            value_logits = self._get_value_logits(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            critic = self._raw_critic
            support = getattr(critic, "value_support")
            target_probabilities = project_categorical_targets(batch.returns, support)
            value_loss = -(target_probabilities * F.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }
        if mean_rnd_loss is not None:
            loss_dict["rnd"] = mean_rnd_loss / num_updates
        if mean_symmetry_loss is not None:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates
        if self.reward_scaler is not None:
            loss_dict["reward_scale"] = float(self.reward_scaler.denominator)
        self.storage.clear()
        return loss_dict

    def _get_value_logits(self, observations, masks=None, hidden_state=None) -> torch.Tensor:
        get_logits = getattr(self._raw_critic, "get_value_logits", None)
        if get_logits is None:
            raise TypeError("Categorical critic must implement get_value_logits().")
        return get_logits(observations, masks=masks, hidden_state=hidden_state)

    def save(self) -> dict:
        """Save model state together with reward-scaling statistics."""
        saved_dict = super().save()
        if self.reward_scaler is not None:
            saved_dict["reward_scaler_state_dict"] = self.reward_scaler.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore model and optional reward-scaling state."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        self._validate_hyperspherical_optimizer()
        self._project_hyperspherical_weights()
        if self.reward_scaler is not None and "reward_scaler_state_dict" in loaded_dict:
            self.reward_scaler.load_state_dict(loaded_dict["reward_scaler_state_dict"])
        return load_iteration
