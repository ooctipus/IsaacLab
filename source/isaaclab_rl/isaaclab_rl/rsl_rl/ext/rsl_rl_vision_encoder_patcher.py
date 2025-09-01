# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import gymnasium
import numpy as np
import logging
from collections.abc import Mapping
from tensordict import TensorDict
from types import MethodType

from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
from isaaclab.managers import ObservationManager
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.algorithms.ppo import PPO
from ...ext.actor_critic_vision_ext import ActorCriticVision, vision_forward

def single_observation_space_from_obs(obs_dict: TensorDict | dict[str, torch.Tensor]):
    new_gym_space_dict = gymnasium.spaces.Dict()
    for key, obs in obs_dict.items():
        new_gym_space_dict[key] = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(*obs.shape[1:],))
        new_gym_space_dict[key] = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(*obs.shape[1:],))
    return new_gym_space_dict

class ActorCriticVisionExtensionPatcher:
    def __init__(self, a2c_cfg):
        enc = a2c_cfg.get("encoders", None) if isinstance(a2c_cfg, Mapping) else getattr(a2c_cfg, "encoders", None)
        self.vision = ActorCriticVision(enc)

    def apply_patch(self) -> None:
        """Attach encoders to policy instance and patch its observation methods."""
        encoders = self.vision.adapter_cfg.encoder_cfgs
        first_encoder = list(encoders.values())[0]
        if not first_encoder.freeze:
            self._orig_get_actor_obs = ActorCritic.get_actor_obs
            self._orig_get_critic_obs = ActorCritic.get_critic_obs
            self._orig_actor_critic_init = ActorCritic.__init__
            self._orig_ppo_update = PPO.update
            
            def vision_encoder_creation_init(actor_critic_self, *args, **kwargs) -> None:
                obs_arg, *rest = args
                self.vision.encoder_init(obs_arg)
                # when adapter is not freeze, it is a part of the nn.module, and its device is the same as 
                # the ActorCritic Module
                actor_obs = vision_forward(self.vision, obs_arg.to('cpu'), self.vision.group2encoder, obs_arg.batch_size, "cpu")
                new_args = (actor_obs, *rest)
                self._orig_actor_critic_init(actor_critic_self, *new_args, **kwargs)
                actor_critic_self.add_module("encoders", self.vision.encoders)
                actor_critic_self.add_module("projectors", self.vision.projectors)
                actor_critic_self.vision_forward = MethodType(vision_forward, actor_critic_self)
                self.vision.print_vision_encoders()

            def vision_encoded_get_actor_obs(actor_critic_self, obs: TensorDict) -> torch.Tensor:
                encoded_obs = actor_critic_self.vision_forward(obs, self.vision.group2encoder, obs.batch_size, obs.device)
                return self._orig_get_actor_obs(actor_critic_self, encoded_obs)

            def vision_encoded_get_critic_obs_patched(actor_critic_self, obs: TensorDict) -> torch.Tensor:
                encoded_obs = actor_critic_self.vision_forward(obs, self.vision.group2encoder, obs.batch_size, obs.device)
                return self._orig_get_critic_obs(actor_critic_self, encoded_obs)
            
            def sol_update(ppo_self: PPO):
                mean_value_loss = 0
                mean_surrogate_loss = 0
                mean_entropy = 0
                # -- RND loss
                if ppo_self.rnd:
                    mean_rnd_loss = 0
                else:
                    mean_rnd_loss = None
                # -- Symmetry loss
                if ppo_self.symmetry:
                    mean_symmetry_loss = 0
                else:
                    mean_symmetry_loss = None

                # generator for mini batches
                if ppo_self.policy.is_recurrent:
                    generator = ppo_self.storage.recurrent_mini_batch_generator(ppo_self.num_mini_batches, ppo_self.num_learning_epochs)
                else:
                    generator = ppo_self.storage.mini_batch_generator(ppo_self.num_mini_batches, ppo_self.num_learning_epochs)

                # iterate over batches
                for (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    hid_states_batch,
                    masks_batch,
                ) in generator:

                    # number of augmentations per sample
                    # we start with 1 and increase it if we use symmetry augmentation
                    num_aug = 1
                    # original batch size
                    # we assume policy group is always there and needs augmentation
                    original_batch_size = ppo_self.policy.get_actor_obs(obs_batch).shape[0]

                    # check if we should normalize advantages per mini batch
                    if ppo_self.normalize_advantage_per_mini_batch:
                        with torch.no_grad():
                            advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

                    # Perform symmetric augmentation
                    if ppo_self.symmetry and ppo_self.symmetry["use_data_augmentation"]:
                        # augmentation using symmetry
                        data_augmentation_func = ppo_self.symmetry["data_augmentation_func"]
                        # returned shape: [batch_size * num_aug, ...]
                        obs_batch, actions_batch = data_augmentation_func(  # TODO: needs changes on the isaac lab side
                            obs=obs_batch,
                            actions=actions_batch,
                            env=ppo_self.symmetry["_env"],
                        )
                        # compute number of augmentations per sample
                        # we assume policy group is always there and needs augmentation
                        num_aug = int(obs_batch["policy"].shape[0] / original_batch_size)
                        # repeat the rest of the batch
                        # -- actor
                        old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                        # -- critic
                        target_values_batch = target_values_batch.repeat(num_aug, 1)
                        advantages_batch = advantages_batch.repeat(num_aug, 1)
                        returns_batch = returns_batch.repeat(num_aug, 1)

                    # Recompute actions log prob and entropy for current batch of transitions
                    # Note: we need to do this because we updated the policy with the new parameters
                    # -- actor
                    ppo_self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                    actions_log_prob_batch = ppo_self.policy.get_actions_log_prob(actions_batch)
                    # -- critic
                    value_batch = ppo_self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                    # -- entropy
                    # we only keep the entropy of the first augmentation (the original one)
                    mu_batch = ppo_self.policy.action_mean[:original_batch_size]
                    sigma_batch = ppo_self.policy.action_std[:original_batch_size]
                    entropy_batch = ppo_self.policy.entropy[:original_batch_size]

                    # KL
                    if ppo_self.desired_kl is not None and ppo_self.schedule == "adaptive":
                        with torch.inference_mode():
                            kl = torch.sum(
                                torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                                / (2.0 * torch.square(sigma_batch))
                                - 0.5,
                                axis=-1,
                            )
                            kl_mean = torch.mean(kl)

                            # Reduce the KL divergence across all GPUs
                            if ppo_self.is_multi_gpu:
                                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                                kl_mean /= ppo_self.gpu_world_size

                            # Update the learning rate
                            # Perform this adaptation only on the main process
                            # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                            #       then the learning rate should be the same across all GPUs.
                            if ppo_self.gpu_global_rank == 0:
                                if kl_mean > ppo_self.desired_kl * 2.0:
                                    ppo_self.learning_rate = max(1e-5, ppo_self.learning_rate / 1.5)
                                elif kl_mean < ppo_self.desired_kl / 2.0 and kl_mean > 0.0:
                                    ppo_self.learning_rate = min(1e-2, ppo_self.learning_rate * 1.5)

                            # Update the learning rate for all GPUs
                            if ppo_self.is_multi_gpu:
                                lr_tensor = torch.tensor(ppo_self.learning_rate, device=ppo_self.device)
                                torch.distributed.broadcast(lr_tensor, src=0)
                                ppo_self.learning_rate = lr_tensor.item()

                            # Update the learning rate for all parameter groups
                            for param_group in ppo_self.optimizer.param_groups:
                                param_group["lr"] = ppo_self.learning_rate

                    # Surrogate loss
                    ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                    surrogate = -torch.squeeze(advantages_batch) * ratio
                    surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                        ratio, 1.0 - ppo_self.clip_param, 1.0 + ppo_self.clip_param
                    )
                    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                    # Value function loss
                    if ppo_self.use_clipped_value_loss:
                        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                            -ppo_self.clip_param, ppo_self.clip_param
                        )
                        value_losses = (value_batch - returns_batch).pow(2)
                        value_losses_clipped = (value_clipped - returns_batch).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (returns_batch - value_batch).pow(2).mean()

                    loss = surrogate_loss + ppo_self.value_loss_coef * value_loss - ppo_self.entropy_coef * entropy_batch.mean()
                    # Begin Octi Edit
                    loss += getattr(self, "sol", torch.tensor(0.0, device=obs_batch.device))
                    # End Octi Edit
                    # Symmetry loss
                    if ppo_self.symmetry:
                        # obtain the symmetric actions
                        # if we did augmentation before then we don't need to augment again
                        if not ppo_self.symmetry["use_data_augmentation"]:
                            data_augmentation_func = ppo_self.symmetry["data_augmentation_func"]
                            obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=ppo_self.symmetry["_env"])
                            # compute number of augmentations per sample
                            num_aug = int(obs_batch.shape[0] / original_batch_size)

                        # actions predicted by the actor for symmetrically-augmented observations
                        mean_actions_batch = ppo_self.policy.act_inference(obs_batch.detach().clone())

                        # compute the symmetrically augmented actions
                        # note: we are assuming the first augmentation is the original one.
                        #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                        #   However, the symmetry loss is computed using the mean of the distribution.
                        action_mean_orig = mean_actions_batch[:original_batch_size]
                        _, actions_mean_symm_batch = data_augmentation_func(
                            obs=None, actions=action_mean_orig, env=ppo_self.symmetry["_env"]
                        )

                        # compute the loss (we skip the first augmentation as it is the original one)
                        mse_loss = torch.nn.MSELoss()
                        symmetry_loss = mse_loss(
                            mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                        )
                        # add the loss to the total loss
                        if ppo_self.symmetry["use_mirror_loss"]:
                            loss += ppo_self.symmetry["mirror_loss_coeff"] * symmetry_loss
                        else:
                            symmetry_loss = symmetry_loss.detach()

                    # Random Network Distillation loss
                    # TODO: Move this processing to inside RND module.
                    if ppo_self.rnd:
                        # extract the rnd_state
                        # TODO: Check if we still need torch no grad. It is just an affine transformation.
                        with torch.no_grad():
                            rnd_state_batch = ppo_self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                            rnd_state_batch = ppo_self.rnd.state_normalizer(rnd_state_batch)
                        # predict the embedding and the target
                        predicted_embedding = ppo_self.rnd.predictor(rnd_state_batch)
                        target_embedding = ppo_self.rnd.target(rnd_state_batch).detach()
                        # compute the loss as the mean squared error
                        mseloss = torch.nn.MSELoss()
                        rnd_loss = mseloss(predicted_embedding, target_embedding)

                    # Compute the gradients
                    # -- For PPO
                    ppo_self.optimizer.zero_grad()
                    loss.backward()
                    # -- For RND
                    if ppo_self.rnd:
                        ppo_self.rnd_optimizer.zero_grad()  # type: ignore
                        rnd_loss.backward()

                    # Collect gradients from all GPUs
                    if ppo_self.is_multi_gpu:
                        ppo_self.reduce_parameters()

                    # Apply the gradients
                    # -- For PPO
                    nn.utils.clip_grad_norm_(ppo_self.policy.parameters(), ppo_self.max_grad_norm)
                    ppo_self.optimizer.step()
                    # -- For RND
                    if ppo_self.rnd_optimizer:
                        ppo_self.rnd_optimizer.step()

                    # Store the losses
                    mean_value_loss += value_loss.item()
                    mean_surrogate_loss += surrogate_loss.item()
                    mean_entropy += entropy_batch.mean().item()
                    # -- RND loss
                    if mean_rnd_loss is not None:
                        mean_rnd_loss += rnd_loss.item()
                    # -- Symmetry loss
                    if mean_symmetry_loss is not None:
                        mean_symmetry_loss += symmetry_loss.item()

                # -- For PPO
                num_updates = ppo_self.num_learning_epochs * ppo_self.num_mini_batches
                mean_value_loss /= num_updates
                mean_surrogate_loss /= num_updates
                mean_entropy /= num_updates
                # -- For RND
                if mean_rnd_loss is not None:
                    mean_rnd_loss /= num_updates
                # -- For Symmetry
                if mean_symmetry_loss is not None:
                    mean_symmetry_loss /= num_updates
                # -- Clear the storage
                ppo_self.storage.clear()

                # construct the loss dictionary
                loss_dict = {
                    "value_function": mean_value_loss,
                    "surrogate": mean_surrogate_loss,
                    "entropy": mean_entropy,
                }
                if ppo_self.rnd:
                    loss_dict["rnd"] = mean_rnd_loss
                if ppo_self.symmetry:
                    loss_dict["symmetry"] = mean_symmetry_loss

                return loss_dict

            PPO.update = sol_update
            ActorCritic.__init__ = vision_encoder_creation_init
            ActorCritic.get_actor_obs = vision_encoded_get_actor_obs
            ActorCritic.get_critic_obs = vision_encoded_get_critic_obs_patched

            logging.warning("Applied vision patch to ActorCritic; encoders now part of state_dict.")
        else:
            self.original_manager_based_configure_gym_env_spaces = ManagerBasedRLEnv._configure_gym_env_spaces
            self.original_direct_configure_gym_env_spaces = DirectRLEnv._configure_gym_env_spaces
            self.original_observaiton_manager_compute = ObservationManager.compute
            self.original_direct_get_observation = DirectRLEnv._get_observations
            
            def adapted_observation_manager_compute(observation_manager_self, update_history=False):
                obs_buf = self.original_observaiton_manager_compute(observation_manager_self, update_history)
                return self._vision_forward(obs_buf, observation_manager_self.num_envs, observation_manager_self.device)
            
            def adapted_direct_get_obseravtion(lab_env_self:DirectRLEnv):
                obs_buf = self.original_direct_get_observation(lab_env_self)
                return self._vision_forward(obs_buf, lab_env_self.num_envs, lab_env_self.device)
        
            
            def configure_group_obs_to_actor_ctiric_obs_gym_spaces(lab_env_self):
                # When the freeze is True, encode is NOT apart of module, don't have to store raw observation but store
                # the feature in rollout buffer instead
                if isinstance(lab_env_self, ManagerBasedRLEnv):
                    self.original_manager_based_configure_gym_env_spaces(lab_env_self)
                    self.vision.encoder_init(lab_env_self.observation_space, self.vision.adapter_cfg.activation)
                    self.vision.perception_encoder.to(lab_env_self.device)
                    self.vision.feature_projectors.to(lab_env_self.device)
                    encoded_obs = lab_env_self.observation_manager.compute()
                    lab_env_self.single_observation_space = single_observation_space_from_obs(encoded_obs)
                    lab_env_self.observation_space = gymnasium.vector.utils.batch_space(lab_env_self.single_observation_space, lab_env_self.num_envs)
                else:
                    self.vision.encoder_init(lab_env_self.observation_space, self.vision.adapter_cfg.activation)
                    encoded_obs = vision_forward(self.vision, self.original_observaiton_manager_compute(), lab_env_self.num_envs, lab_env_self.device)
                    lab_env_self.single_observation_space = adapt_gym_single_space(self.obs_groups, lab_env_self.single_observation_space)
                    lab_env_self.observation_space = gymnasium.vector.utils.batch_space(lab_env_self.single_observation_space, lab_env_self.num_envs)
                    if lab_env_self.state_space is not None:
                        lab_env_self.state_space = gymnasium.vector.utils.batch_space(lab_env_self.single_observation_space["critic"], lab_env_self.num_envs)
                self.vision.print_vision_encoders()
            
            ObservationManager.compute = adapted_observation_manager_compute
            ManagerBasedRLEnv._configure_gym_env_spaces = configure_group_obs_to_actor_ctiric_obs_gym_spaces

            DirectRLEnv._configure_gym_env_spaces = configure_group_obs_to_actor_ctiric_obs_gym_spaces
            DirectRLEnv._get_observations = adapted_direct_get_obseravtion
    
    def remove_patch(self) -> None:
        if not self.adapter_cfg.freeze:
            """Restore original methods on policy."""
            ActorCritic.__init__ = self._orig_actor_critic_init
            ActorCritic.get_actor_obs = self._orig_get_actor_obs
            ActorCritic.get_critic_obs = self._orig_get_critic_obs
            PPO.update = self._orig_ppo_update
            logging.warning("Removed vision patch from ActorCritic.")
        else:
            ManagerBasedRLEnv._configure_gym_env_spaces = self.original_manager_based_configure_gym_env_spaces
            ObservationManager.compute = self.original_observaiton_manager_compute
            DirectRLEnv._configure_gym_env_spaces = self.original_direct_configure_gym_env_spaces
            DirectRLEnv._get_observations = self.original_direct_get_observation
    