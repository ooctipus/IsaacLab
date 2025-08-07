# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
import textwrap
import torch.nn as nn
import gymnasium as gym
import numpy as np
import logging
from typing import List, TYPE_CHECKING
from tensordict import TensorDict
from types import MethodType

from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
from isaaclab.managers import ObservationManager
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation

if TYPE_CHECKING:
    from ...actor_critic_vision_cfg import ActorCriticVisionAdapterCfg

def single_observation_space_from_obs(obs_dict: TensorDict | dict[str, torch.Tensor]):
    new_gym_space_dict = gym.spaces.Dict()
    for key, obs in obs_dict.items():
        new_gym_space_dict[key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(*obs.shape[1:],))
        new_gym_space_dict[key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(*obs.shape[1:],))
    return new_gym_space_dict

class ActorCriticVisionExtensionPatcher:
    """
    Patcher that injects vision encoding into an existing ActorCritic
    by assigning encoders and patching methods at instance-level.

    Example:
        patcher = ActorCriticVisionExtensionPatcher(obs, obs_groups, ...)
        patcher.apply_patch(policy)
        # policy now encodes high-dim inputs and includes encoder state in state_dict
    """
    def __init__(self, actor_critic_cfg, obs_space = None):
        self.encoder_applicable = False
        if hasattr(actor_critic_cfg, "encoder") and getattr(actor_critic_cfg, "encoder") is not None:
            self.adapter_cfg: ActorCriticVisionAdapterCfg = actor_critic_cfg.encoder
            self.encoder_applicable = True
            self.observation_space = None

    def apply_patch(self) -> None:
        """Attach encoders to policy instance and patch its observation methods."""
        if not self.encoder_applicable:
            return
        if not self.adapter_cfg.freeze:
            # if not free, the patch is done by attaching to the torch.nn.Module, so the network is fused together
            # with the network class
            self._orig_get_actor_obs = ActorCritic.get_actor_obs
            self._orig_get_critic_obs = ActorCritic.get_critic_obs
            self._orig_init = ActorCritic.__init__
            
            def vision_encoder_creation_init(actor_critic_self, *args, **kwargs) -> None:
                obs_arg, *rest = args
                self._encoder_init(obs_arg, kwargs.get('activation'))
                # when adapter is not freeze, it is a part of the nn.module, and its device is the same as 
                # the ActorCritic Module
                actor_obs = self._vision_forward(obs_arg.to('cpu'), obs_arg.batch_size, "cpu")
                new_args = (actor_obs, *rest)
                self._orig_init(actor_critic_self, *new_args, **kwargs)
                actor_critic_self.add_module("perception_encoder", self.perception_encoder)
                actor_critic_self.add_module("feature_projectors", self.feature_projectors)
                actor_critic_self._vision_forward = MethodType(self._vision_forward, actor_critic_self)

            def vision_encoded_get_actor_obs(actor_critic_self, obs: TensorDict) -> torch.Tensor:
                return self._orig_get_actor_obs(actor_critic_self, self._vision_forward(obs, obs.batch_size, obs.device))

            def vision_encoded_get_critic_obs_patched(actor_critic_self, obs: TensorDict) -> torch.Tensor:
                return self._orig_get_critic_obs(actor_critic_self, self._vision_forward(obs, obs.batch_size, obs.device))

            # monkey-patch
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
                    self._encoder_init(lab_env_self.observation_space, self.adapter_cfg.activation)
                    self.perception_encoder.to(lab_env_self.device)
                    self.feature_projectors.to(lab_env_self.device)
                    encoded_obs = lab_env_self.observation_manager.compute()
                    lab_env_self.single_observation_space = single_observation_space_from_obs(encoded_obs)
                    lab_env_self.observation_space = gym.vector.utils.batch_space(lab_env_self.single_observation_space, lab_env_self.num_envs)
                else:
                    self._encoder_init(lab_env_self.observation_space, self.adapter_cfg.activation)
                    encoded_obs = self._vision_forward(self.original_observaiton_manager_compute(), lab_env_self.num_envs, lab_env_self.device)
                    lab_env_self.single_observation_space = adapt_gym_single_space(self.obs_groups, lab_env_self.single_observation_space)
                    lab_env_self.observation_space = gym.vector.utils.batch_space(lab_env_self.single_observation_space, lab_env_self.num_envs)
                    if lab_env_self.state_space is not None:
                        lab_env_self.state_space = gym.vector.utils.batch_space(lab_env_self.single_observation_space["critic"], lab_env_self.num_envs)
            
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
            # ManagerBasedRLEnv.step = self.original_manager_based_step
            # ManagerBasedRLEnv.reset = self.original_manager_based_reset
            DirectRLEnv._configure_gym_env_spaces = self.original_direct_configure_gym_env_spaces
            DirectRLEnv.step = self.original_direct_based_step
            DirectRLEnv.reset = self.original_direct_based_reset


    def _vision_forward(self, obs: TensorDict, num_envs, device) -> TensorDict:
        """
        Convert high-dim inputs to feature vectors in a new TensorDict.
        """
        proc = TensorDict({}, batch_size=num_envs, device=device)
        for key, val in obs.items():
            if key in self.perception_encoder:
                proc[key] = self.perception_encoder[key](val)
            elif key in self.feature_projectors:
                proc[key] = self.feature_projectors[key](val)
            else:
                proc[key] = val
        return proc
    
    def _encoder_init(self, obs, activation):
        self.image_keys: List[str] = []
        self.feature_keys: List[str] = []
        for key, val in obs.items():
            if isinstance(val, gym.spaces.Box):
                val = torch.tensor(val.sample())
            if isinstance(val, torch.Tensor) and val.ndim in [3, 4]:
                self.image_keys.append(key)
            elif isinstance(val, torch.Tensor) and val.ndim == 2 and any(
                term in key.lower() for term in ["rgb","image","feature","encoding"]
            ):
                self.feature_keys.append(key)

        # build encoders
        self.adapter_cfg.activation = self.adapter_cfg.activation if self.adapter_cfg.activation else activation
        self.perception_encoder = nn.ModuleDict()
        for key in self.image_keys:
            adapter_cfg_class = self.adapter_cfg.encoder_cfg.class_type
            self.perception_encoder[key] = adapter_cfg_class(obs[key], self.adapter_cfg)

        self.feature_projectors = nn.ModuleDict()
        for key in self.feature_keys:
            dim = obs[key].shape[-1]
            self.feature_projectors[key] = nn.Sequential(
                nn.Linear(dim, max(dim // 2, self.adapter_cfg["image_feature_dim"])),
                resolve_nn_activation(activation),
                nn.Linear(max(dim // 2, self.adapter_cfg["image_feature_dim"]), self.adapter_cfg["image_feature_dim"]),
                resolve_nn_activation(activation),
            )

        print("Vision Encoder initialized with the following modules:")
        if self.perception_encoder:
            print("  Image encoders:")
            for name, encoder in self.perception_encoder.items():
                print(f"\n  • Encoder '{name}':")
                # Print the preprocessing pipeline
                print("      Preprocessing steps:")
                if hasattr(encoder, "_processor_descriptions") and encoder._processor_descriptions:
                    for i, desc in enumerate(encoder._processor_descriptions, start=1):
                        print(f"        {i:02d}. {desc}")
                else:
                    print("        (no preprocessing steps)")

                print("      Architecture:")
                print(textwrap.indent(repr(encoder), "        "))

        if self.feature_projectors:
            print("\n  Feature projectors:")
            for name, projector in self.feature_projectors.items():
                print(f"  • Projector '{name}':")
                print(textwrap.indent(repr(projector), "    "))