# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import torch
from collections.abc import Mapping
from types import MethodType
from typing import Any

from ..ext.actor_critic_vision_ext import ActorCriticVision, vision_forward

try:
    import rl_games.algos_torch.models as rl_games_models  # type: ignore
    from rl_games.algos_torch.a2c_continuous import A2CAgent  # type: ignore
    from rl_games.algos_torch.central_value import CentralValueTrain  # type: ignore
    from rl_games.algos_torch.models import BaseModelNetwork  # type: ignore
    from rl_games.common.a2c_common import A2CBase, ContinuousA2CBase  # type: ignore
except Exception as e:
    raise ImportError("rl_games is not available, but 'rl_games' patch_lib was requested.") from e


class RLGamesVisionPatch:
    def __init__(self, a2c_cfg: Any):
        enc = a2c_cfg.get("encoders", None) if isinstance(a2c_cfg, Mapping) else getattr(a2c_cfg, "encoders", None)
        self.vision = ActorCriticVision(enc)
        self.encoder_initialized = False

    def apply_patch(self) -> None:
        self._orig_a2c_agent_init = A2CAgent.__init__
        self._orig_central_value_init = CentralValueTrain.__init__
        self._orig_a2c_get_action_values = A2CBase.get_action_values
        self._orig_a2c_get_central_values = A2CBase.get_central_value
        self._orig_norm_obs = BaseModelNetwork.norm_obs

        def vision_encoder_creation_init(model_self, base_name, params) -> None:
            ContinuousA2CBase.__init__(model_self, base_name, params)
            input_sample = {
                group: torch.from_numpy(box.sample()).unsqueeze(0)
                for group, box in model_self.observation_space.items()
            }
            self.vision.encoder_init(input_sample)
            obs = vision_forward(self.vision, input_sample, self.vision.group2encoder, 1, "cpu")
            encoded_actor_shape = sum([shape.shape[1] for shape in obs.values()])
            model_self.obs_shape = (encoded_actor_shape,)
            # make original vision_encoder_creation_init no operation
            _orig_base_init = ContinuousA2CBase.__init__

            def _noop(*args, **kwargs):
                pass

            ContinuousA2CBase.__init__ = _noop
            self._orig_a2c_agent_init(model_self, base_name, params)
            ContinuousA2CBase.__init__ = _orig_base_init  # restor original vision_encoder_creation_init
            model_self.model.add_module("encoders", self.vision.encoders)
            model_self.model.add_module("projectors", self.vision.projectors)
            setattr(model_self.model, "key_order", [k for k in model_self.observation_space.keys()])
            model_self.model.vision_forward = MethodType(vision_forward, model_self.model)
            model_self.optimizer = torch.optim.Adam(
                model_self.model.parameters(),
                float(model_self.last_lr),
                eps=1e-08,
                weight_decay=model_self.weight_decay,
            )
            self.vision.print_vision_encoders()

        def central_value_init(central_value_self, *args, **kwargs):
            state_shape = kwargs["state_shape"]
            input_sample = {group: torch.rand(1, *shape) for group, shape in state_shape.items()}
            self.vision.encoder_init(input_sample)
            obs = vision_forward(self.vision, input_sample, self.vision.group2encoder, 1, "cpu")
            state_shape_flatten = sum([state_shape.shape[1] for state_shape in obs.values()])
            kwargs["state_shape"] = (state_shape_flatten,)
            self._orig_central_value_init(central_value_self, *args, **kwargs)
            central_value_self.model.add_module("encoders", self.vision.encoders)
            central_value_self.model.add_module("projectors", self.vision.projectors)
            setattr(central_value_self.model, "key_order", [k for k in state_shape.keys()])
            central_value_self.model.vision_forward = MethodType(vision_forward, central_value_self.model)
            central_value_self.optimizer = torch.optim.Adam(
                central_value_self.model.parameters(),
                float(central_value_self.lr),
                eps=1e-08,
                weight_decay=central_value_self.weight_decay,
            )
            self.vision.print_vision_encoders()

        def _grad_norm(self, observation):
            # Use rl_games' normalizer in no_grad, then reattach grads to the input.
            if not self.normalize_input:
                return observation
            with torch.no_grad():
                normed = self.running_mean_std(observation)  # updates stats, returns normalized
            # same values as `normed`, but gradients flow as identity from `observation`
            return observation + (normed - observation).detach()

        A2CAgent.__init__ = vision_encoder_creation_init
        CentralValueTrain.__init__ = central_value_init
        BaseModelNetwork.norm_obs = _grad_norm

        _targets = ["ModelA2CContinuous", "ModelA2CContinuousLogStd", "ModelCentralValue"]
        for target in _targets:
            model = getattr(rl_games_models, target).Network
            setattr(self, f"_orig_{target}_forward", model.forward)

            def vision_encoded_forward(actor_critic_self, obs, _target=target):
                first_obs_tensor = list(obs["obs"].values())[0]
                encoded_obs = actor_critic_self.vision_forward(
                    obs["obs"], self.vision.group2encoder, first_obs_tensor.shape[0], first_obs_tensor.device
                )
                obs["obs"] = torch.cat([encoded_obs[key] for key in actor_critic_self.key_order], dim=1)
                result = getattr(self, f"_orig_{_target}_forward")(actor_critic_self, obs)
                return result

            model.forward = vision_encoded_forward
