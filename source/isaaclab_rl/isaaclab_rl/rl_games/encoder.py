# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# --- encoded_a2c_builder.py ---
import torch
import torch.nn as nn
from typing import Any, Dict, Tuple

from rl_games.algos_torch.network_builder import A2CBuilder


class EncodedA2CBuilder(A2CBuilder):

    class Network(A2CBuilder.Network):
        def __init__(self, params: dict[str, Any], **kwargs):
            # -------------------------
            # Step 1: prebuild encoders
            # -------------------------
            pre = self._prepare_encoders(params, kwargs)

            # rewrite kwargs['input_shape'] if encoders are present (no mutation of original)
            kwargs_mod = dict(kwargs)
            if pre is not None:
                kwargs_mod["input_shape"] = pre["encoded_input_shape"]

            # -------------------------
            # Step 2: parent init
            # -------------------------
            super().__init__(params, **kwargs_mod)

            # -------------------------
            # Step 3: attach modules
            # -------------------------
            if pre is not None:
                # register as proper submodules / buffers
                self.encoders = pre["encoders"]  # nn.ModuleDict
                self.encoder_group = pre["encoder_group"]  # Dict[str, str]
                self.obs_keys = pre["obs_keys"]  # List[str]
                # (Optional) keep for clarity/debug
                self._encoded_input_shape = pre["encoded_input_shape"]
                self._raw_group_input_shapes = pre["raw_group_input_shapes"]

        # ---- helpers ---------------------------------------------------------

        @staticmethod
        def _prepare_encoders(params: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any] | None:
            """
            Build encoder modules (NOT registered yet), derive the concatenated encoded input shape,
            and return a small pre-init bundle. If no encoders, return None.

            Returns:
                {
                  "encoders": nn.ModuleDict,
                  "encoder_group": Dict[name -> group_key],
                  "obs_keys": List[str],
                  "encoded_input_shape": Tuple[int],       # e.g., (sum_feature_dims,)
                  "raw_group_input_shapes": Dict[str, Tuple[int,...]],
                }
            """
            enc_cfg = params.get("encoders", None)
            input_shape = kwargs.get("input_shape")
            # DO NOT mutate incoming structures.
            input_shape_cp: dict[str, tuple[int, ...]] = dict(input_shape)

            # Read without popping to avoid side-effects across tasks.
            obs_keys = enc_cfg.get("out_obs", None)
            if not obs_keys:
                # If the user forgot to provide out_obs, nothing to encode safely.
                return None

            encoders = nn.ModuleDict()
            encoder_group: dict[str, str] = {}
            feature_size_dict: dict[str, tuple[int]] = {}

            # Build each encoder and infer its output dim by a dummy forward.
            for name, cfg in enc_cfg.items():
                if name == "out_obs":
                    continue
                if not isinstance(cfg, dict):
                    continue
                if cfg.get("type_class") != "encoder":
                    continue

                # Resolve cfg class & module constructor
                nn_cfg_cls = getattr(NetworkCfgs, f"{cfg['network_class']}Cfg")
                nn_cfg = nn_cfg_cls(**cfg["network_args"])
                nn_args = cfg["type_args"]

                group_key = nn_args["encoding_group"]
                if group_key not in input_shape_cp:
                    raise KeyError(f"[EncodedA2CBuilder] encoding_group '{group_key}' not in input_shape.")

                # Build encoder module (not attached yet).
                encoder_input = input_shape_cp[group_key]
                module = nn_cfg.class_type(encoder_input, nn_cfg)

                # Infer feature dim via dummy forward (CPU is fine; no params registered yet).
                with torch.no_grad():
                    dummy = torch.randn(1, *encoder_input)
                    out = module(dummy)
                    # Expect (N, F) after encoder
                    if out.dim() == 2:
                        feat_dim = out.size(1)
                    else:
                        # if the encoder returns e.g. (N, C, H, W), flatten except batch
                        feat_dim = int(out.shape[1:].numel())

                feature_size_dict[group_key] = (feat_dim,)
                encoders[name] = module
                encoder_group[name] = group_key

            # Update shapes: replace raw group dims with encoded dims
            shaped = dict(input_shape_cp)
            shaped.update(feature_size_dict)

            # Compute the *concatenated* flat shape in the obs_keys order.
            # This is exactly your original reduction to a single tuple (sum of dims).
            try:
                encoded_w = sum(shaped[k][0] for k in obs_keys)
            except Exception as e:
                raise KeyError(
                    "[EncodedA2CBuilder] out_obs contained a key not in shapes. "
                    f"out_obs={obs_keys} available={list(shaped.keys())}"
                ) from e

            encoded_input_shape: tuple[int, ...] = (encoded_w,)

            return {
                "encoders": encoders,
                "encoder_group": encoder_group,
                "obs_keys": obs_keys,
                "encoded_input_shape": encoded_input_shape,
                "raw_group_input_shapes": input_shape_cp,
            }
