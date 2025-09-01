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
from typing import Union, Dict
from tensordict import TensorDict
from rsl_rl.networks import EmpiricalNormalization
from .actor_critic_vision_cfg import ActorCriticVisionAdapterCfg


def vision_forward(
    self,
    obs: dict[str, torch.Tensor],
    group2encoder: dict[str, str],
    batch_size: int,
    device: str = "cpu"
) -> TensorDict:
    enc = self.encoders
    proc = TensorDict({}, batch_size=batch_size, device=device)
    for key, val in obs.items():
        if key in group2encoder:
            proc[key] = enc[group2encoder[key]](val)
        else:
            proc[key] = val
    return proc


def projector_forward(
    self,
    obs_batch: dict[str, torch.Tensor],
    projector_features: dict[str, list[str]],
    projector_prediction_targets: dict[str, list[str]],
) -> dict[str, torch.Tensor]:
    loss = {}
    projectors = self.projectors
    projector_normalizers = self.projector_feature_normalizers
    projector_ground_truth_normalizers = self.projector_ground_truth_normalizers
    for projector_key, feature_keys in projector_features.items():
        in_feature_batch = torch.cat([obs_batch[feat_key] for feat_key in feature_keys], dim=1)
        loss[projector_key] = projectors[projector_key](projector_normalizers[projector_key](in_feature_batch))
        prediction_targets = torch.cat([obs_batch[tgt_key] for tgt_key in projector_prediction_targets[projector_key]], dim=1)
        normalized_prediction_targets = projector_ground_truth_normalizers[projector_key](prediction_targets)
        mse_loss = nn.functional.mse_loss(loss[projector_key], normalized_prediction_targets)
        loss[projector_key] = mse_loss
    return loss


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    if act_name == "elu": return torch.nn.ELU()
    elif act_name == "selu": return torch.nn.SELU()
    elif act_name == "relu": return torch.nn.ReLU()
    elif act_name == "crelu": return torch.nn.CELU()
    elif act_name == "lrelu": return torch.nn.LeakyReLU()
    elif act_name == "tanh": return torch.nn.Tanh()
    elif act_name == "sigmoid": return torch.nn.Sigmoid()
    elif act_name == "identity": return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")


class ActorCriticVision:

    ObsSpaceLike = Union[gymnasium.spaces.Dict, gym.spaces.Dict, TensorDict, Dict[str, torch.Tensor]]

    def __init__(self, adapter_cfg: ActorCriticVisionAdapterCfg):
        if not isinstance(adapter_cfg, ActorCriticVisionAdapterCfg):
            raise ValueError("Encoder configuration not readable")
        self.adapter_cfg = adapter_cfg

        self.group2encoder: dict[str, str] = {}
        self.encoders = nn.ModuleDict()

        self.projector_prediction_targets: dict[str, list[str]] = {}
        self.projector_feature_sources: dict[str, list[str]] = {}
        self.projector_feature_normalizers = nn.ModuleDict()
        self.projector_ground_truth_normalizers = nn.ModuleDict()
        self.projectors = nn.ModuleDict()

    def encoder_init(self, obs: ObsSpaceLike):
        # process gym.space input so encoder only worries about dict input
        obs_ = TensorDict({}, batch_size=1, device="cpu")
        for key, val in obs.items():
            if isinstance(val, (gymnasium.spaces.Box, gym.spaces.Box)):
                sample = torch.tensor(val.sample())[0]
                obs_[key] = sample.view(1, *sample.shape).cpu()
            elif isinstance(val, torch.Tensor):
                obs_[key] = val[0].view(1, *val.shape[1:]).cpu()

        for encoder_key, encoder_cfg in self.adapter_cfg.encoder_cfgs.items():
            if not (set(obs_.keys()) & set(encoder_cfg.encoding_groups)):
                continue
            first_obs = obs_[encoder_cfg.encoding_groups[0]]
            self.encoders[encoder_key] = encoder_cfg.class_type(first_obs, encoder_cfg)
            for group_key in encoder_cfg.encoding_groups:
                self.group2encoder[group_key] = encoder_key

        encoded_obs = vision_forward(self, obs_, self.group2encoder, batch_size=1, device="cpu")
        if len(self.encoders) > 0 and self.adapter_cfg.projectors_cfg is not None:
            for project_key, projector_cfg in self.adapter_cfg.projectors_cfg.items():
                self.projector_feature_sources[project_key] = projector_cfg.features
                self.projector_prediction_targets[project_key] = projector_cfg.predictions
                feature_in_dim = sum([encoded_obs[feat_key].shape[-1] for feat_key in projector_cfg.features])
                prediction_dim = sum([obs_[pred_key].shape[-1] for pred_key in projector_cfg.predictions])

                layers = []
                if projector_cfg.feature_empirical_normalize:
                    self.projector_feature_normalizers[project_key] = EmpiricalNormalization(feature_in_dim)
                    self.projector_ground_truth_normalizers[project_key] = EmpiricalNormalization(prediction_dim)
                else:
                    self.projector_feature_normalizers[project_key] = nn.Identity()
                    self.projector_ground_truth_normalizers[project_key] = nn.Identity()
                prev = feature_in_dim  # <- real encoded feature dim
                for h in projector_cfg.layers:
                    layers += [nn.Linear(prev, h), resolve_nn_activation(projector_cfg.activation)]
                    prev = h
                layers += [nn.Linear(prev, prediction_dim)]  # final linear, no activation
                self.projectors[project_key] = nn.Sequential(*layers)

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