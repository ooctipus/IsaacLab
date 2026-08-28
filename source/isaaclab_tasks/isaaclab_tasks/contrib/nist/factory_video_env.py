# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-sequenced Factory variant video playback."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch

from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn

from .utils.variant_event_combinators import variant_reset_accumulator


class SuccessSequencedVariantReset(variant_reset_accumulator):
    """Sample the assembly variant selected by the video environment."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.variant_index = 0

    def _sample_marginally_balanced(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._sample_variant(num_samples, self.variant_index)


class FactoryVariantVideoEnv(ManagerBasedRLEnv):
    """Record one successful visible-environment episode per assembly variant."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        sequence = self.event_manager.get_term_cfg("reset_strategies").func
        if not isinstance(sequence, SuccessSequencedVariantReset):
            raise TypeError("FactoryVariantVideoEnv requires SuccessSequencedVariantReset.")
        self._video_sequence = sequence
        self._video_attempt_start_step = 0
        self._video_segments: list[tuple[int, int, int]] = []
        self._video_complete = False

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        result = super().step(action)
        if self._video_complete:
            for recorder in self.video_recorders:
                recorder.cfg.video_length = self.common_step_counter
                recorder.close()
        return result

    def close(self) -> None:
        try:
            if not self._is_closed:
                self._write_segment_manifest()
        finally:
            super().close()

    def _reset_idx(self, env_ids: Sequence[int]):
        if isinstance(env_ids, torch.Tensor):
            visible_reset = bool((env_ids == 0).any().item())
        else:
            visible_reset = 0 in env_ids
        if visible_reset and not self._video_sequence.precollecting_phase and not self._video_complete:
            end_step = self.common_step_counter
            if bool(self.termination_manager.get_term("success")[0].item()):
                variant = self._video_sequence.variant_index
                self._video_segments.append((variant, self._video_attempt_start_step, end_step))
                if variant + 1 == self._video_sequence._num_variants:
                    self._video_complete = True
                else:
                    self._video_sequence.variant_index += 1
            self._video_attempt_start_step = end_step
        super()._reset_idx(env_ids)

    def _write_segment_manifest(self) -> None:
        if not self.video_recorders:
            return
        recorder_cfg = self.video_recorders[0].cfg
        if recorder_cfg.output_dir is None:
            return
        output_dir = Path(recorder_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        segments = [
            {
                "variant": self._video_sequence.variant_names[variant],
                "start_step": start,
                "end_step": end,
            }
            for variant, start, end in self._video_segments
        ]
        manifest = {
            "complete": self._video_complete,
            "step_dt": self.step_dt,
            "segments": segments,
        }
        path = output_dir / f"{recorder_cfg.output_filename_prefix}_segments.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
