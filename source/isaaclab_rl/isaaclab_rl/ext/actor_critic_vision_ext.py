# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gym.spaces
import torch
import textwrap
import torch.nn as nn
import gymnasium
import gym
from typing import List, Union, Dict
from .actor_critic_vision_cfg import ActorCriticVisionAdapterCfg

def vision_forward(
    self,
    obs: dict[str, torch.Tensor],
    group2encoder: dict[str, str],
    batch_size: int,
    device: str = "cpu"
) -> dict:
    enc = self.encoders
    proc = {}
    for key, val in obs.items():
        if key in group2encoder:
            proc[key] = enc[group2encoder[key]](val)
        else:
            proc[key] = val
    return proc

def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    if act_name == "elu": return torch.nn.ELU()
    elif act_name == "selu": return torch.nn.SELU()
    elif act_name == "relu": return torch.nn.ReLU()
    elif act_name == "crelu": return torch.nn.CELU()
    elif act_name == "lrelu": return torch.nn.LeakyReLU()
    elif act_name == "tanh": return torch.nn.Tanh()
    elif act_name == "sigmoid": return torch.nn.Sigmoid()
    elif act_name == "identity": return torch.nn.Identity()
    else: raise ValueError(f"Invalid activation function '{act_name}'.")

class ActorCriticVision:
    
    ObsSpaceLike = Union[gymnasium.spaces.Dict, gym.spaces.Dict, dict, Dict[str, torch.Tensor]]
    
    def __init__(self, adapter_cfg: ActorCriticVisionAdapterCfg):
        if not isinstance(adapter_cfg, ActorCriticVisionAdapterCfg):
            raise ValueError("Encoder configuration not readable")
        self.adapter_cfg = adapter_cfg

        self.group2encoder: dict[str, str] = {}
        self.encoders = nn.ModuleDict()

        self.group2projector: dict[str, str] = {}
        self.projectors = nn.ModuleDict()

    def encoder_init(self, obs: ObsSpaceLike):
        # process gym.space input so encoder only worries about dict input
        obs_: dict[str, torch.Tensor] = {}
        
        for key, val in obs.items():
            if isinstance(val, (gymnasium.spaces.Box, gym.spaces.Box)):
                obs_[key] = torch.tensor(val.sample())
            elif isinstance(val, torch.Tensor):
                obs_[key] = val

        for encoder_key, encoder_cfg in self.adapter_cfg.encoder_cfgs.items():
            if not (set(obs_.keys()) & set(encoder_cfg.encoding_groups)):
                continue
            first_obs = obs_[encoder_cfg.encoding_groups[0]]
            self.encoders[encoder_key] = encoder_cfg.class_type(first_obs, encoder_cfg)
            for group_key in encoder_cfg.encoding_groups:
                self.group2encoder[group_key] = encoder_key

        if len(self.encoders) > 0 and self.adapter_cfg.projectors_cfg is not None:
            for project_key, projector_cfg in self.adapter_cfg.projectors_cfg.items():
                flat_in = 1  # dummy value need to figure out how to get the output dim
                layers = [nn.Flatten()]
                in_dim = flat_in
                for out_dim in projector_cfg.layers:
                    layers.append(nn.Linear(in_dim, out_dim))
                    layers.append(resolve_nn_activation(projector_cfg.activation))
                    in_dim = out_dim
                self.projectors[project_key] = nn.Sequential(*layers)
                for group_key in projector_cfg.prediction_group:
                    self.group2projector[group_key] = project_key

    def print_vision_encoders(self):
        print("Vision Encoder initialized with the following modules:")
        if self.encoders:
            print("  Image encoders:")
            for name, encoder in self.encoders.items():
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

        if self.projectors:
            print("\n  Feature projectors:")
            for name, projector in self.projectors.items():
                print(f"  • Projector '{name}':")
                print(textwrap.indent(repr(projector), "    "))