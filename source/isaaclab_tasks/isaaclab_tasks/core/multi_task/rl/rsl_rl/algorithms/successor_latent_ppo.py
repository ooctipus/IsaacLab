# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with a critic-side successor-latent auxiliary value objective."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage import RolloutStorage

if TYPE_CHECKING:
    from rsl_rl.env import VecEnv


class SuccessorLatentPPO(PPO):
    """PPO + vector-valued value learning over critic latents.

    The auxiliary head mirrors the scalar critic target construction, replacing
    scalar rewards with detached critic latents as vector-valued cumulants:

    .. code-block:: text

        V(s_t) ~= r_t      + gamma V(s_{t+1})
        U(s_t) ~= phi(s_t) + gamma U(s_{t+1})

    The first implementation is feedforward-only because recurrent critics need
    hidden-state-aware target construction that differs from plain flat storage.
    """

    def __init__(self, *args, successor_loss_coef: float = 0.01, **kwargs) -> None:
        """Initialize PPO and attach the successor-latent head.

        Args:
            *args: Positional arguments forwarded to :class:`rsl_rl.algorithms.PPO`.
            successor_loss_coef: Coefficient for the successor-latent loss.
            **kwargs: Keyword arguments forwarded to :class:`rsl_rl.algorithms.PPO`.
        """
        super().__init__(*args, **kwargs)
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("SuccessorLatentPPO currently supports feedforward actor/critic models only.")
        if successor_loss_coef < 0.0:
            raise ValueError(f"successor_loss_coef must be non-negative; got {successor_loss_coef}.")

        with torch.inference_mode():
            latent_dim = self.critic.get_latent(self.storage.observations[0]).shape[-1]
        self.successor_loss_coef = successor_loss_coef
        self.successor_head = nn.Linear(latent_dim, latent_dim).to(self.device)
        self.optimizer.add_param_group({"params": self.successor_head.parameters()})
        self._successor_returns: torch.Tensor | None = None

    def compute_returns(self, obs) -> None:
        """Compute scalar PPO returns and vector-valued successor-latent returns."""
        super().compute_returns(obs)
        self._compute_successor_returns(obs)

    def update(self) -> dict[str, float]:
        """Run PPO updates with an additional successor-latent value loss."""
        if self._successor_returns is None:
            raise RuntimeError("SuccessorLatentPPO.update() called before compute_returns().")

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_successor_loss = 0
        mean_successor_target_norm = 0
        mean_successor_prediction_norm = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        for batch, successor_returns in self._mini_batch_generator_with_successor_returns():
            assert batch.observations is not None
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                assert batch.advantages is not None
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            if self.symmetry:
                symmetry = cast(Any, self.symmetry)
                symmetry.augment_batch(batch, original_batch_size)
                if symmetry.use_data_augmentation:
                    assert batch.observations is not None
                    num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                    successor_returns = successor_returns.repeat(num_aug, 1)

            observations = batch.observations
            actions = batch.actions
            values_batch = batch.values
            advantages = batch.advantages
            returns = batch.returns
            old_actions_log_prob = batch.old_actions_log_prob
            old_distribution_params = batch.old_distribution_params
            assert observations is not None
            assert actions is not None
            assert values_batch is not None
            assert advantages is not None
            assert returns is not None
            assert old_actions_log_prob is not None
            assert old_distribution_params is not None

            self.actor(
                observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(actions)
            critic_latent = self.critic.get_latent(observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            values = self.critic.mlp(critic_latent)
            successor_predictions = self.successor_head(critic_latent)

            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(old_actions_log_prob))
            surrogate = -torch.squeeze(advantages) * ratio
            surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = values_batch + (values - values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - returns).pow(2)
                value_losses_clipped = (value_clipped - returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns - values).pow(2).mean()

            successor_loss = (successor_predictions - successor_returns).pow(2).mean()
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + self.successor_loss_coef * successor_loss
                - self.entropy_coef * entropy.mean()
            )

            if self.symmetry:
                symmetry = cast(Any, self.symmetry)
                symmetry_loss = symmetry.compute_loss(self.actor, batch, original_batch_size)
                if symmetry.use_mirror_loss:
                    loss += symmetry.mirror_loss_coeff * symmetry_loss

            if self.rnd:
                rnd = cast(Any, self.rnd)
                rnd_loss = rnd.compute_loss(observations[:original_batch_size])

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                rnd = cast(Any, self.rnd)
                rnd.optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.successor_head.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                rnd = cast(Any, self.rnd)
                rnd.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_successor_loss += successor_loss.item()
            mean_successor_target_norm += successor_returns.norm(dim=-1).mean().item()
            mean_successor_prediction_norm += successor_predictions.norm(dim=-1).mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_successor_loss /= num_updates
        mean_successor_target_norm /= num_updates
        mean_successor_prediction_norm /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()
        self._successor_returns = None

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "successor_latent": mean_successor_loss,
            "successor_latent_target_norm": mean_successor_target_norm,
            "successor_latent_prediction_norm": mean_successor_prediction_norm,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        super().train_mode()
        self.successor_head.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        super().eval_mode()
        self.successor_head.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = super().save()
        saved_dict["successor_head_state_dict"] = self.successor_head.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        should_load_successor = load_cfg is None or load_cfg.get("successor_latent", True)
        if should_load_successor and "successor_head_state_dict" in loaded_dict:
            self.successor_head.load_state_dict(loaded_dict["successor_head_state_dict"], strict=strict)
        return load_iteration

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        raw_actor = getattr(self, "_raw_actor", self.actor)
        raw_critic = getattr(self, "_raw_critic", self.critic)
        model_params = [raw_actor.state_dict(), raw_critic.state_dict(), self.successor_head.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        raw_actor.load_state_dict(model_params[0])
        raw_critic.load_state_dict(model_params[1])
        self.successor_head.load_state_dict(model_params[2])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[3])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them."""
        raw_actor = getattr(self, "_raw_actor", self.actor)
        raw_critic = getattr(self, "_raw_critic", self.critic)
        all_params = chain(raw_actor.parameters(), raw_critic.parameters(), self.successor_head.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    @staticmethod
    def construct_algorithm(obs, env: VecEnv, cfg: dict, device: str) -> SuccessorLatentPPO:
        """Build the SuccessorLatentPPO algorithm through the standard PPO factory."""
        alg: SuccessorLatentPPO = PPO.construct_algorithm(obs, env, cfg, device)  # type: ignore[assignment]
        assert isinstance(alg, SuccessorLatentPPO), (
            f"SuccessorLatentPPO.construct_algorithm expected a SuccessorLatentPPO instance; got {type(alg).__name__}."
            " Check that ``algorithm.class_name`` resolves to SuccessorLatentPPO."
        )
        return alg

    def _compute_successor_returns(self, obs) -> None:
        """Compute lambda-returns for the vector-valued successor-latent head."""
        st = self.storage
        with torch.no_grad():
            successor_latents = torch.stack(
                [self.critic.get_latent(st.observations[step]) for step in range(st.num_transitions_per_env)]
            )
            successor_values = self.successor_head(successor_latents)
            last_successor_value = self.successor_head(self.critic.get_latent(obs))

            successor_advantage = torch.zeros_like(last_successor_value)
            successor_returns = torch.zeros_like(successor_values)
            for step in reversed(range(st.num_transitions_per_env)):
                next_successor_value = (
                    last_successor_value if step == st.num_transitions_per_env - 1 else successor_values[step + 1]
                )
                next_is_not_terminal = 1.0 - st.dones[step].float()
                delta = (
                    successor_latents[step]
                    + next_is_not_terminal * self.gamma * next_successor_value
                    - successor_values[step]
                )
                successor_advantage = delta + next_is_not_terminal * self.gamma * self.lam * successor_advantage
                successor_returns[step] = successor_advantage + successor_values[step]

        self._successor_returns = successor_returns.detach()

    def _mini_batch_generator_with_successor_returns(self):
        """Yield feedforward PPO mini-batches with matching successor-latent targets."""
        st = self.storage
        if st.training_type != "rl":
            raise ValueError("SuccessorLatentPPO only supports reinforcement-learning rollout storage.")
        if self._successor_returns is None:
            raise RuntimeError("Successor-latent returns must be computed before building mini-batches.")

        batch_size = st.num_envs * st.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        indices = torch.randperm(self.num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = st.observations.flatten(0, 1)
        actions = st.actions.flatten(0, 1)
        values = st.values.flatten(0, 1)
        returns = st.returns.flatten(0, 1)
        old_actions_log_prob = st.actions_log_prob.flatten(0, 1)
        advantages = st.advantages.flatten(0, 1)
        successor_returns = self._successor_returns.flatten(0, 1)
        assert observations is not None
        assert st.distribution_params is not None
        old_distribution_params = tuple(p.flatten(0, 1) for p in st.distribution_params)

        for _ in range(self.num_learning_epochs):
            for i in range(self.num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                yield (
                    RolloutStorage.Batch(
                        observations=observations[batch_idx],
                        actions=actions[batch_idx],
                        values=values[batch_idx],
                        advantages=advantages[batch_idx],
                        returns=returns[batch_idx],
                        old_actions_log_prob=old_actions_log_prob[batch_idx],
                        old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                    ),
                    successor_returns[batch_idx],
                )
