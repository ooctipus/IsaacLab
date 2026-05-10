# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PyTorch reference implementation of :class:`MultiTaskCommand`.

Byte-identical in output to :class:`~.multi_task_command_warp.MultiTaskCommandWarp`
(gated by :mod:`tests.test_multi_task_warp_equivalence`). Kept deliberately slow
and obvious — its role is to be the implementation that the Warp path is
validated against. Do not optimize this file.

Selected when ``MultiTaskCfg.dispatch_backend="torch"``. The factory in
:class:`~..multi_task_command.MultiTaskCommand.__new__` routes construction to
this class automatically; users never reference it directly.
"""

from __future__ import annotations

import torch

from .. import multi_task_command as _base_module
from ..multi_task_command import MultiTaskCommand
from ..reward_composer import multiplicative_terminal_reward
from .kernels_torch import ACTIVATION_KERNELS, BUFFER_KIND, DELTA_KERNELS, METRIC_KERNELS, STATE_KERNEL_COMPUTES

__all__ = ["MultiTaskCommandTorch"]


class MultiTaskCommandTorch(MultiTaskCommand):
    """PyTorch reference — kept for correctness verification.

    The per-step dispatch runs as two PyTorch phases:

    * :meth:`_compute_state_delta_error_reference` — for each slab, fill the
      unified state buffer; for each read group, gather + compute +
      delta/error + scatter into ``_buf_error`` and canonical reach/track.
    * :meth:`_compute_activation_reference` — per ``activation_kid``, mask
      the error buffer and apply the kernel.

    Dispatches ~440 small kernels per step. See
    :class:`~.multi_task_command_warp.MultiTaskCommandWarp` for the optimized
    path (≈8 launches).
    """

    def _dispatch(self, valid_slots: torch.Tensor) -> None:
        """State → delta → metric → activation (two-phase PyTorch dispatch)."""
        self._compute_state_delta_error_reference(valid_slots)
        self._compute_activation_reference(valid_slots)

    def _compose(self, valid_slots: torch.Tensor) -> None:
        """Advance composer state + write terminal reward / success / progress.

        Byte-identical to :meth:`MultiTaskCommandWarp._compose` — the
        equivalence test gates drift.
        """
        # Accumulate and increment (pre-composer — composer reads updated state).
        self._sum_activation.add_(self._buf_activation)
        self._transit_steps.add_(1)

        # Per-slot type masks (composer input only).
        safe_subtask_ids = self._env_subtask_ids.clamp_min(0)
        is_instant_slot = self.spec.is_instant[safe_subtask_ids] & valid_slots
        is_tracking_slot = self.spec.is_tracking[safe_subtask_ids] & valid_slots

        # Quality factor — every tracking subtask contributes one transit-mean
        # activation to a single multiplicative product, raised to a
        # configurable easing exponent in (0, 1]:
        #
        #     quality_factor = ( ∏_{k ∈ tracking} mean_t A_k(t) ) ^ easing
        #
        # "Tracking" here is the unified quality kind that includes both
        # ordinary tracking goals (``expose_in_obs=True``) and soft-safety
        # constraints (``expose_in_obs=False``). The composer doesn't care
        # which is which — they all just contribute to the product. Their
        # only difference is whether their delta channel reaches the policy
        # obs (handled upstream in the canonical layout).
        #
        # Easing softens multiplicative compounding: with K=3 quality dims at
        # f_k = 0.5, raw product is 0.125, but ``^0.5`` gives 0.354 — strong
        # gradient signal preserved even when several dimensions are degraded.
        # Easing = 1.0 recovers the pure product; easing = 0 disables
        # discounting entirely (degenerate). Default ``cfg.quality_easing = 0.5``
        # is a reasonable starting point (matches the geometric mean at K=2).
        steps_denom = self._transit_steps.to(self._sum_activation.dtype).unsqueeze(-1)  # [N, 1]
        per_subtask_mean = self._sum_activation / steps_denom  # [N, k_max]
        # Replace non-tracking slots with 1 so they don't enter the product
        # (multiplicative identity); empty tracking set ⇒ product = 1 ⇒
        # ``1^easing = 1`` so a pure-instant task gets quality_factor = 1.
        quality_per_slot = torch.where(is_tracking_slot, per_subtask_mean, torch.ones_like(per_subtask_mean))
        quality_product = quality_per_slot.prod(dim=-1)  # [N]
        quality_factor = quality_product.pow(self.cfg.quality_easing)  # [N]

        # ``_update_command`` runs at the END of env.step (after reward read and
        # after ``_reset_idx`` zeroes ``episode_length_buf`` for timed-out envs).
        # The outer timeout DoneTerm fires at step ``k`` where
        # ``episode_length_buf[k] == effective_max_episode_length``. Reward at
        # that step is read from ``_task_reward`` set by step ``k-1``'s
        # ``_update_command`` — so we latch the terminal value when
        # ``buf >= max - 1`` (at step ``k-1``), ensuring the read at step ``k``
        # sees it. Uses per-env effective max so pure-tracking envs under the
        # adaptive curriculum latch at their current dynamic length, not the
        # global cap.
        is_timeout = self._env.episode_length_buf >= self._effective_max_episode_length - 1

        out = multiplicative_terminal_reward(
            activation_scores=self._buf_activation,
            is_instant_subtask=is_instant_slot,
            instant_achieved=self._instant_achieved,
            quality_factor=quality_factor,
            is_timeout=is_timeout,
        )
        # Write composer outputs in-place so external views keep tensor identity.
        self._instant_achieved.copy_(out.instant_achieved_next)
        self._task_reward.copy_(out.reward)
        self._task_done_success.copy_(out.done_success)

        # Progress — mean of active-slot activations. Empty slot_count → 0.
        active_count = self._env_slot_count.clamp(min=1).to(self._buf_activation.dtype)
        torch.div(
            (self._buf_activation * valid_slots).sum(dim=1),
            active_count,
            out=self._progress,
        )

    # ------------------------------------------------------------------------
    # Phase 1 — read slabs, compute per-slot error, scatter canonical deltas.
    # ------------------------------------------------------------------------

    def _compute_state_delta_error_reference(self, valid_slots: torch.Tensor) -> None:
        """Two-dispatch pipeline per step:

        1. **READ DISPATCH.** For each unique ``(buffer_kind, asset_name)`` slab
           in the spec's layout, call the reader once and write its output into
           the unified per-step buffer ``_unified_buffer``. After this phase the
           data every kernel will need lives in one contiguous tensor.

        2. **EXECUTE DISPATCH.** For each read group, advanced-index-gather its
           precomputed absolute indices from the unified buffer
           (``[M, N, slice_size]``) and hand the stack to one batched compute
           call. No per-asset lookup here — the execute side sees the unified
           buffer as an opaque state tensor and just slices offsets.

        The outer loops iterate spec-time Python lists (``slab_*_py``,
        ``read_group_*_py``) — never ``unique().tolist()`` over device tensors —
        so no per-step CPU sync. Empty masks fall through naturally as
        zero-length advanced-index gathers / scatters; no per-iteration
        ``.any()`` guards.
        """
        if self.num_subtasks == 0:
            return

        safe_subtask_ids = self._env_subtask_ids.clamp_min(0)  # [N, k_max]
        live = valid_slots
        read_group_per_slot = self.spec.read_group_id[safe_subtask_ids]
        metric_kids_per_slot = self.spec.metric_kernel_id[safe_subtask_ids]

        # READ DISPATCH — fill the unified buffer, one reader call per slab.
        # Slab sizes are fixed at spec build (scene inspection); the size guard
        # below is a metadata-only check (``raw.numel()`` doesn't sync) that
        # turns a misconfigured reader into a clean error rather than a cryptic
        # reshape failure.
        #
        # BODY_POS_W readers return world-frame positions; the env-origin
        # subtraction happens inline here (allocation is OK — reference is
        # explicitly the slow path). The Warp subclass bakes the same
        # subtraction into its own kernel (``fill_slab_vec3_env_local``) to
        # keep the path pointer-stable for CUDA Graph capture.
        slab_kinds = self.spec.slab_buffer_kinds
        slab_assets = self.spec.slab_asset_names
        slab_offsets = self.spec.slab_offsets_py
        slab_sizes = self.spec.slab_sizes_py
        body_pos_w_kind = int(BUFFER_KIND.BODY_POS_W)
        for s in range(len(slab_kinds)):
            kind = slab_kinds[s]
            asset_name = slab_assets[s]
            offset = slab_offsets[s]
            size = slab_sizes[s]
            # Read via the base module so tests can patch BUFFER_KIND_READERS there.
            raw = _base_module.BUFFER_KIND_READERS[kind](self._env, asset_name)  # [N, ...]
            raw_per_env = raw.numel() // self.num_envs
            if raw_per_env != size:
                raise RuntimeError(
                    f"State kernel output dim mismatch for slab (kind={kind}, asset={asset_name}): "
                    f"reader returned {raw_per_env} floats per env, but slab was sized for {size}. "
                    "Reader output shape must be consistent with the spec-time scene inspection."
                )
            if kind == body_pos_w_kind:
                env_origins = self._env.scene.env_origins
                if env_origins.ndim == raw.ndim - 1:
                    env_origins = env_origins.unsqueeze(-2)
                self._unified_buffer[:, offset : offset + size] = (raw - env_origins).reshape(self.num_envs, size)
            else:
                self._unified_buffer[:, offset : offset + size] = raw.reshape(self.num_envs, size)

        # EXECUTE DISPATCH — iterate every read group; empty masks no-op.
        # The build-time stride consistency gate (per ``(state_kid, entity)`` and
        # per ``(state_kid, stride)`` group key) makes any runtime stride sanity
        # check redundant.
        group_state_kids = self.spec.read_group_state_kernel_id_py
        group_metric_kids = self.spec.read_group_metric_kids_py
        for gid in range(len(group_state_kids)):
            mask_group = live & (read_group_per_slot == gid)
            state_kid = group_state_kids[gid]
            compute_fn = STATE_KERNEL_COMPUTES[state_kid]

            gather_idx = self.spec.read_group_gather_indices[gid]  # [M, slice_size]
            gathered = self._unified_buffer[:, gather_idx]  # [N, M, slice_size]
            stacked = gathered.transpose(0, 1).contiguous()  # [M, N, slice_size]
            x_stacked = compute_fn(stacked)
            per_subtask_stride = int(x_stacked.shape[-1])
            arange_s = torch.arange(per_subtask_stride, device=self.device).unsqueeze(0)

            # Dispatch delta / error per unique metric_kid within the group —
            # both the loop universe and its order are spec-time constants.
            for mkid in group_metric_kids[gid]:
                mask_mk = mask_group & (metric_kids_per_slot == mkid)
                flat_env, flat_slot = mask_mk.nonzero(as_tuple=True)
                flat_sids = safe_subtask_ids[flat_env, flat_slot]

                # Pull each subtask's row from the batched output via its position
                # in the group's member list.
                member_idx = self.spec.subtask_member_index[flat_sids]
                x_cur_m = x_stacked[member_idx, flat_env]  # [n_match, stride]

                offsets_m = self._env_slot_offsets[flat_env, flat_slot]
                col_idx = offsets_m.unsqueeze(1) + arange_s
                row_idx = flat_env.unsqueeze(1).expand_as(col_idx)
                tgt_m = self._targets_flat[row_idx, col_idx]

                delta_m = DELTA_KERNELS[mkid](x_cur_m, tgt_m)
                err_m = METRIC_KERNELS[mkid](delta_m)
                self._buf_error[flat_env, flat_slot] = err_m.to(self._buf_error.dtype)

                # Canonical obs scatter — route each subtask into its own tensor
                # (reach for instant, track for tracking). A read group can hold
                # mixed types (same kernel + stride, different type), so split
                # per-subtask. ``canonical_offset = -1`` means no projection
                # (joint kernels) → mask falls through.
                canon_off = self.spec.canonical_offset[flat_sids]
                projected = canon_off >= 0
                is_reach = self.spec.is_instant[flat_sids] & projected
                is_track = (~self.spec.is_instant[flat_sids]) & projected
                canon_col_idx = canon_off.unsqueeze(1) + arange_s
                self._command_reach[row_idx[is_reach], canon_col_idx[is_reach]] = delta_m[is_reach].to(
                    self._command_reach.dtype
                )
                self._command_track[row_idx[is_track], canon_col_idx[is_track]] = delta_m[is_track].to(
                    self._command_track.dtype
                )

        # Rotate spatial 3-vec slots into each asset's root frame — the
        # command_reach / command_track tensors are now body-aligned so
        # downstream obs terms read them as-is (no frame logic required).
        # ``_buf_error`` stays frame-agnostic (L2 norm is rotation-invariant).
        self._rotate_canonical_slots_to_body_frame()

    # ------------------------------------------------------------------------
    # Phase 2 — per-activation_kid kernel over the error buffer.
    # ------------------------------------------------------------------------

    def _compute_activation_reference(self, valid_slots: torch.Tensor) -> None:
        """Activation kernel dispatch over every activation_kid that appears in any subtask.

        Iterates the spec-time list :attr:`spec.unique_activation_kids_py` so the
        outer loop is a CPU-side iteration over Python ints — no
        ``torch.unique(...).tolist()`` sync per step. Empty masks no-op naturally.
        """
        if self.num_subtasks == 0:
            return

        safe_subtask_ids = self._env_subtask_ids.clamp_min(0)
        activation_kids_per_slot = self.spec.activation_kernel_id[safe_subtask_ids]  # [N, k_max]
        activation_params_per_slot = self.spec.activation_kernel_param[safe_subtask_ids]  # [N, k_max]

        for akid in self.spec.unique_activation_kids_py:
            mask = valid_slots & (activation_kids_per_slot == akid)
            err_m = self._buf_error[mask]
            params_m = activation_params_per_slot[mask]
            out = ACTIVATION_KERNELS[akid](err_m, params_m).to(self._buf_activation.dtype)
            self._buf_activation[mask] = out
