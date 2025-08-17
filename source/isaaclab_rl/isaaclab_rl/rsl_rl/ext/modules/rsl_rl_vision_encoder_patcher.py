# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import gymnasium
import numpy as np
import logging
from collections.abc import Mapping
from tensordict import TensorDict
from types import MethodType

from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
from isaaclab.managers import ObservationManager
from rsl_rl.modules.actor_critic import ActorCritic
from ....ext.actor_critic_vision_ext import ActorCriticVision, vision_forward

def single_observation_space_from_obs(obs_dict: TensorDict | dict[str, torch.Tensor]):
    new_gym_space_dict = gymnasium.spaces.Dict()
    for key, obs in obs_dict.items():
        new_gym_space_dict[key] = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(*obs.shape[1:],))
        new_gym_space_dict[key] = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(*obs.shape[1:],))
    return new_gym_space_dict

class ActorCriticVisionExtensionPatcher:
    def __init__(self, a2c_cfg):
        enc = a2c_cfg.get("encoder", None) if isinstance(a2c_cfg, Mapping) else getattr(a2c_cfg, "encoder", None)
        self.vision = ActorCriticVision(enc)

    def apply_patch(self) -> None:
        """Attach encoders to policy instance and patch its observation methods."""
        if not self.vision.adapter_cfg.freeze:
            self._orig_get_actor_obs = ActorCritic.get_actor_obs
            self._orig_get_critic_obs = ActorCritic.get_critic_obs
            self._orig_init = ActorCritic.__init__
            
            def vision_encoder_creation_init(actor_critic_self, *args, **kwargs) -> None:
                obs_arg, *rest = args
                self.vision.encoder_init(obs_arg, activation=kwargs.get('activation'))
                # when adapter is not freeze, it is a part of the nn.module, and its device is the same as 
                # the ActorCritic Module
                actor_obs = vision_forward(self.vision, obs_arg.to('cpu'), obs_arg.batch_size, "cpu")
                new_args = (actor_obs, *rest)
                self._orig_init(actor_critic_self, *new_args, **kwargs)
                actor_critic_self.add_module("perception_encoder", self.vision.perception_encoder)
                actor_critic_self.add_module("feature_projectors", self.vision.feature_projectors)
                actor_critic_self.vision_forward = MethodType(vision_forward, actor_critic_self)

            def vision_encoded_get_actor_obs(actor_critic_self, obs: TensorDict) -> torch.Tensor:
                encoded_obs = actor_critic_self.vision_forward(obs, obs.batch_size, obs.device)
                return self._orig_get_actor_obs(actor_critic_self, encoded_obs)

            def vision_encoded_get_critic_obs_patched(actor_critic_self, obs: TensorDict) -> torch.Tensor:
                encoded_obs = actor_critic_self.vision_forward(obs, obs.batch_size, obs.device)
                return self._orig_get_critic_obs(actor_critic_self, encoded_obs)

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
            
            ObservationManager.compute = adapted_observation_manager_compute
            ManagerBasedRLEnv._configure_gym_env_spaces = configure_group_obs_to_actor_ctiric_obs_gym_spaces

            DirectRLEnv._configure_gym_env_spaces = configure_group_obs_to_actor_ctiric_obs_gym_spaces
            DirectRLEnv._get_observations = adapted_direct_get_obseravtion
    
    def remove_patch(self) -> None:
        if not self.adapter_cfg.freeze:
            """Restore original methods on policy."""
            ActorCritic.get_actor_obs = self._orig_get_actor_obs
            ActorCritic.get_critic_obs = self._orig_get_critic_obs

            logging.warning("Removed vision patch from ActorCritic.")
        else:
            ManagerBasedRLEnv._configure_gym_env_spaces = self.original_manager_based_configure_gym_env_spaces
            ObservationManager.compute = self.original_observaiton_manager_compute
            DirectRLEnv._configure_gym_env_spaces = self.original_direct_configure_gym_env_spaces
            DirectRLEnv._get_observations = self.original_direct_get_observation
    