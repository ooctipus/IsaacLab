# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
import textwrap
import torch.nn as nn

import logging
from typing import List
from tensordict import TensorDict
from types import MethodType

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation

if TYPE_CHECKING:
    from ...actor_critic_vision_cfg import ActorCriticVisionAdapterCfg


class ActorCriticVisionExtensionPatcher:
    """
    Patcher that injects vision encoding into an existing ActorCritic
    by assigning encoders and patching methods at instance-level.

    Example:
        patcher = ActorCriticVisionExtensionPatcher(obs, obs_groups, ...)
        patcher.apply_patch(policy)
        # policy now encodes high-dim inputs and includes encoder state in state_dict
    """
    def __init__(self, actor_critic_cfg, obs_space):
        self.encoder_applicable = False
        if hasattr(actor_critic_cfg, "encoder") and getattr(actor_critic_cfg, "encoder") is not None:
            self.image_feature_dim = actor_critic_cfg.encoder.image_feature_dim
            self.encoder_type = actor_critic_cfg.encoder.encoder_type
            self.encoder_config = actor_critic_cfg.encoder.encoder_config
            self.freeze_encoder = actor_critic_cfg.encoder.freeze_encoder
            self.normalize = actor_critic_cfg.encoder.normalize
            self.observation_space = obs_space
            self.encoder_applicable = True

    def apply_patch(self) -> None:
        """Attach encoders to policy instance and patch its observation methods."""
        # save originals
        self._orig_get_actor_obs = ActorCritic.get_actor_obs
        self._orig_get_critic_obs = ActorCritic.get_critic_obs
        self._orig_init = ActorCritic.__init__
        if not self.encoder_applicable:
            return
        def vision_encoder_creation_init(actor_critic_self, *args, **kwargs) -> None:
            obs_arg, *rest = args
            self._encoder_init(obs_arg, kwargs.get('activation'))
            actor_obs = self._vision_forward(obs_arg.to('cpu'))
            new_args = (actor_obs, *rest)
            self._orig_init(actor_critic_self, *new_args, **kwargs)
            actor_critic_self.image_encoders = self.image_encoders
            actor_critic_self.add_module("image_encoders", self.image_encoders)
            actor_critic_self.feature_projectors = self.feature_projectors
            actor_critic_self.add_module("feature_projectors", self.feature_projectors)
            actor_critic_self._vision_forward = MethodType(self._vision_forward, actor_critic_self)

        def vision_encoded_get_actor_obs(actor_critic_self, obs: TensorDict) -> torch.Tensor:
            return self._orig_get_actor_obs(actor_critic_self, self._vision_forward(obs))

        def vision_encoded_get_critic_obs_patched(actor_critic_self, obs: TensorDict) -> torch.Tensor:
            return self._orig_get_critic_obs(actor_critic_self, self._vision_forward(obs))

        # monkey-patch
        ActorCritic.__init__ = vision_encoder_creation_init
        ActorCritic.get_actor_obs = vision_encoded_get_actor_obs
        ActorCritic.get_critic_obs = vision_encoded_get_critic_obs_patched

        logging.warning("Applied vision patch to ActorCritic; encoders now part of state_dict.")

    def remove_patch(self) -> None:
        """Restore original methods on policy."""
        ActorCritic.get_actor_obs = policy._orig_get_actor_obs
        ActorCritic.get_critic_obs = policy._orig_get_critic_obs

        logging.warning("Removed vision patch from ActorCritic.")

    def _vision_forward(self, obs: TensorDict) -> TensorDict:
        """
        Convert high-dim inputs to feature vectors in a new TensorDict.
        """
        device = obs.device
        proc = TensorDict({}, batch_size=obs.batch_size, device=obs.device)
        for key, val in obs.items():
            if key in self.image_encoders:
                proc[key] = self.image_encoders[key](val)
            elif key in self.feature_projectors:
                proc[key] = self.feature_projectors[key](val)
            else:
                proc[key] = val
        return proc
    
    def _encoder_init(self, obs, activation):
        self.image_keys: List[str] = []
        self.feature_keys: List[str] = []
        for key, val in obs.items():
            if isinstance(val, torch.Tensor) and val.ndim in [3, 4] :
                self.image_keys.append(key)
            elif isinstance(val, torch.Tensor) and val.ndim == 2 and any(
                term in key.lower() for term in ["rgb","image","feature","encoding"]
            ):
                self.feature_keys.append(key)

        # build encoders
        self.image_encoders = nn.ModuleDict()
        for key in self.image_keys:
            adapter_cfg_class = self.adapter_cfg.encoder_cfg.class_type
            self.perception_encoder[key] = adapter_cfg_class(self.observation_space[key], self.adapter_cfg)

        self.feature_projectors = nn.ModuleDict()
        for key in self.feature_keys:
            dim = obs[key].shape[-1]
            self.feature_projectors[key] = nn.Sequential(
                nn.Linear(dim, max(dim//2, self.image_feature_dim)),
                resolve_nn_activation(activation),
                nn.Linear(max(dim//2, self.image_feature_dim), self.image_feature_dim),
                resolve_nn_activation(activation),
            )

        print("Vision Encoder initialized with the following modules:")
        if self.image_encoders:
            print("  Image encoders:")
            for name, encoder in self.image_encoders.items():
                print(f"\n  • Encoder '{name}':")
                # Print the preprocessing pipeline
                print("      Preprocessing steps:")
                if hasattr(encoder, "_processor_descriptions") and encoder._processor_descriptions:
                    for i, desc in enumerate(encoder._processor_descriptions, start=1):
                        print(f"        {i:02d}. {desc}")
                else:
                    print("        (no preprocessing steps)")

                # Print a little header before the architecture
                print("      Architecture:")
                # indent the repr by 8 spaces so it nests nicely under “Architecture:”
                print(textwrap.indent(repr(encoder), "        "))

        if self.feature_projectors:
            print("\n  Feature projectors:")
            for name, projector in self.feature_projectors.items():
                print(f"  • Projector '{name}':")
                print(textwrap.indent(repr(projector), "    "))