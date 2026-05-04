# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native implementation of :class:`MultiTaskCommand`.

Two-phase per-step dispatch — both phases are ``wp.launch`` calls. Collapses
the ~440 per-step kernel launches of the PyTorch reference
(:class:`~.multi_task_command_reference.MultiTaskCommandReference`) into ≈8.
Output is byte-identical to the reference path (gated by
:mod:`tests.test_multi_task_warp_equivalence`).

Selected when ``MultiTaskCfg.use_warp_dispatch=True``. The factory in
:class:`~.multi_task_command.MultiTaskCommand.__new__` routes construction to
this class automatically; users never reference it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from . import multi_task_command as _base_module
from .multi_task_command import MultiTaskCommand

if TYPE_CHECKING:
    from .kernels_wp import ComposerState, EnvSlots, Outputs, StateAccess, SubtaskSpec

__all__ = ["MultiTaskCommandWarp"]


# ---------------------------------------------------------------------------
# Bindings dataclass — owns the int32 tensors that back the Warp structs.
# ---------------------------------------------------------------------------


@dataclass
class _WarpBindings:
    """Long-lived Warp dispatch bindings for :class:`MultiTaskCommandWarp`.

    Holds:

    - The int32 "spec" tensors that mirror the immutable spec tables
      (``spec.py`` stores these as int64 for PyTorch advanced indexing).
      Allocated once at construction; never refreshed.
    - The four Warp struct instances (:class:`EnvSlots`, :class:`SubtaskSpec`,
      :class:`StateAccess`, :class:`Outputs`) that wrap the int32 tensors
      and per-step state tensors as Warp array handles. These are the
      kernel's four actual arguments.

    Note on ownership: Warp arrays are zero-copy views over torch storage,
    so every field in this dataclass must outlive the struct instances.
    Keeping tensors + structs together here makes the lifetime invariant
    obvious.

    Note on env tables: the base class stores ``_env_subtask_ids`` / ``_env_
    slot_count`` / ``_env_slot_offsets`` as int32 natively, so the Warp
    struct's ``env_slots`` fields wrap those torch tensors directly. No
    per-step refresh is needed — the resample path updates the tensors
    in place and the wp.array views observe the changes immediately.
    """

    # Spec (immutable).
    state_kernel_id_i32: torch.Tensor
    metric_kernel_id_i32: torch.Tensor
    activation_kernel_id_i32: torch.Tensor
    state_stride_i32: torch.Tensor
    canonical_offset_i32: torch.Tensor
    is_instant_i32: torch.Tensor
    is_tracking_i32: torch.Tensor
    subtask_gather_offset_i32: torch.Tensor
    subtask_gather_count_i32: torch.Tensor
    gather_indices_flat_i32: torch.Tensor

    # Warp struct bundles — the kernel's actual arguments.
    env_slots: EnvSlots
    spec: SubtaskSpec
    state: StateAccess
    composer_state: ComposerState
    outputs: Outputs


# ---------------------------------------------------------------------------
# Subclass.
# ---------------------------------------------------------------------------


class MultiTaskCommandWarp(MultiTaskCommand):
    """Warp-native dispatch — one mega-kernel launch per step plus slab copies.

    The per-step dispatch runs as two ``wp.launch`` phases:

    * :meth:`_fill_unified_buffer_warp` — one ``fill_slab_copy`` launch per
      slab; reads from the scene assets (via :data:`BUFFER_KIND_READERS`)
      and fills ``_unified_buffer``.
    * :meth:`_dispatch_mega_warp` — one :func:`~.kernels_wp.dispatch_mega`
      launch over ``(num_envs, k_max)`` that performs project → delta →
      metric → activation → scatter per active ``(env, slot)``.

    See :mod:`.kernels_wp` for the kernel implementations. Adding a new state
    kernel = adding one ``@wp.func`` projection plus one ``elif`` branch in
    ``dispatch_mega``; this class doesn't need to change.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Pre-build int32 mirrors + Warp struct handles. Must happen AFTER
        # the base ``__init__`` has allocated the ``_env_*`` / ``_unified_*``
        # / output tensors and built ``self.spec``.
        self._bindings = self._build_warp_bindings()

    def _dispatch(self, valid_slots: torch.Tensor) -> None:
        """State → delta → metric → activation (two ``wp.launch`` phases).

        ``valid_slots`` is accepted for interface parity with the reference
        path but ignored — the mega-kernel guards invalid slots internally
        via ``slot >= env_slot_count[env]``.
        """
        del valid_slots
        self._fill_unified_buffer_warp()
        self._dispatch_mega_warp()
        # Match the reference path: world-frame delta → body-frame for each
        # POS / LIN_VEL / ANG_VEL canonical slot, so ``command_reach`` /
        # ``command_track`` emerge body-aligned and equal the reference-path
        # output byte-for-byte (gated by ``test_multi_task_warp_equivalence``).
        self._rotate_canonical_slots_to_body_frame()

    def _compose(self, valid_slots: torch.Tensor) -> None:
        """Advance composer state + write terminal reward via one ``wp.launch``.

        Folds what the PyTorch reference does as four separate ops (``add_``,
        ``add_``, ``multiplicative_terminal_reward``, three ``copy_``, a
        ``sum/div``) into a single Warp launch. The kernel updates
        ``sum_activation`` / ``transit_steps`` / ``instant_achieved`` in
        place and writes ``task_reward`` / ``task_done_success`` /
        ``progress`` directly to the output tensors — no ``.copy_()``
        roundtrip.
        """
        del valid_slots
        self._compose_warp()

    # ------------------------------------------------------------------------
    # One-time binding setup.
    # ------------------------------------------------------------------------

    def _build_warp_bindings(self) -> _WarpBindings:
        """Construct the :class:`_WarpBindings` for this instance's tensors.

        Allocates int32 mirrors, populates the four Warp structs with
        :func:`wp.from_torch` views, and returns a single dataclass holding
        everything. The returned object owns all tensor lifetimes.
        """
        import warp as wp

        from .kernels_wp import ComposerState, EnvSlots, Outputs, StateAccess, SubtaskSpec

        s = self.spec

        # Int32 mirrors for the immutable spec tables (spec.py stores int64
        # for PyTorch advanced indexing compatibility; Warp needs int32).
        state_kernel_id_i32 = s.state_kernel_id.to(torch.int32)
        metric_kernel_id_i32 = s.metric_kernel_id.to(torch.int32)
        activation_kernel_id_i32 = s.activation_kernel_id.to(torch.int32)
        state_stride_i32 = s.state_stride.to(torch.int32)
        canonical_offset_i32 = s.canonical_offset.to(torch.int32)
        is_instant_i32 = s.is_instant.to(torch.int32)
        is_tracking_i32 = s.is_tracking.to(torch.int32)
        subtask_gather_offset_i32 = s.subtask_gather_offset.to(torch.int32)
        subtask_gather_count_i32 = s.subtask_gather_count.to(torch.int32)
        gather_indices_flat_i32 = s.gather_indices_flat.to(torch.int32)

        # Warp struct bundles. All ``wp.from_torch`` calls happen here,
        # not per-step — the wp.array handles are stable as long as the
        # underlying torch tensors aren't reallocated.
        #
        # Env tables are int32 natively (base class stores them as int32),
        # so we wrap them directly without an intermediate mirror. The
        # resample path writes in place; the wp.array views observe the
        # new values without any refresh step.
        env_slots = EnvSlots()
        env_slots.subtask_ids = wp.from_torch(self._env_subtask_ids)
        env_slots.slot_count = wp.from_torch(self._env_slot_count)
        env_slots.slot_offsets = wp.from_torch(self._env_slot_offsets)

        spec_struct = SubtaskSpec()
        spec_struct.state_kernel_id = wp.from_torch(state_kernel_id_i32)
        spec_struct.metric_kernel_id = wp.from_torch(metric_kernel_id_i32)
        spec_struct.activation_kernel_id = wp.from_torch(activation_kernel_id_i32)
        spec_struct.activation_kernel_param = wp.from_torch(s.activation_kernel_param)
        spec_struct.state_stride = wp.from_torch(state_stride_i32)
        spec_struct.canonical_offset = wp.from_torch(canonical_offset_i32)
        spec_struct.is_instant_flag = wp.from_torch(is_instant_i32)
        spec_struct.is_tracking_flag = wp.from_torch(is_tracking_i32)
        spec_struct.gather_offset = wp.from_torch(subtask_gather_offset_i32)
        spec_struct.gather_count = wp.from_torch(subtask_gather_count_i32)
        spec_struct.gather_indices_flat = wp.from_torch(gather_indices_flat_i32)

        state = StateAccess()
        state.unified = wp.from_torch(self._unified_buffer)
        state.targets_flat = wp.from_torch(self._targets_flat)

        composer_state = ComposerState()
        composer_state.sum_activation = wp.from_torch(self._sum_activation)
        composer_state.transit_steps = wp.from_torch(self._transit_steps)
        composer_state.instant_achieved = wp.from_torch(self._instant_achieved)

        outputs = Outputs()
        outputs.buf_error = wp.from_torch(self._buf_error)
        outputs.buf_activation = wp.from_torch(self._buf_activation)
        outputs.command_reach = wp.from_torch(self._command_reach)
        outputs.command_track = wp.from_torch(self._command_track)
        outputs.task_reward = wp.from_torch(self._task_reward)
        outputs.task_done_success = wp.from_torch(self._task_done_success)
        outputs.progress = wp.from_torch(self._progress)

        return _WarpBindings(
            state_kernel_id_i32=state_kernel_id_i32,
            metric_kernel_id_i32=metric_kernel_id_i32,
            activation_kernel_id_i32=activation_kernel_id_i32,
            state_stride_i32=state_stride_i32,
            canonical_offset_i32=canonical_offset_i32,
            is_instant_i32=is_instant_i32,
            is_tracking_i32=is_tracking_i32,
            subtask_gather_offset_i32=subtask_gather_offset_i32,
            subtask_gather_count_i32=subtask_gather_count_i32,
            gather_indices_flat_i32=gather_indices_flat_i32,
            env_slots=env_slots,
            spec=spec_struct,
            state=state,
            composer_state=composer_state,
            outputs=outputs,
        )

    # ------------------------------------------------------------------------
    # Phase 1 — read slabs into the unified buffer.
    # ------------------------------------------------------------------------

    def _fill_unified_buffer_warp(self) -> None:
        """Read dispatch — one Warp kernel launch per slab.

        Most slabs use :func:`~.kernels_wp.fill_slab_copy` (straight copy
        from the reader's zero-copy view into the unified buffer). The
        BODY_POS_W slab uses :func:`~.kernels_wp.fill_slab_body_pos_env_local`
        to apply the env-origin subtraction during copy — that transformation
        used to live in the reader as a PyTorch ``body_pos - env_origins``
        allocation, which broke the "reader is a zero-copy view" contract
        and blocked CUDA Graph capture. Now it's a pointer-stable Warp
        launch like every other slab.
        """
        import warp as wp

        from .kernels_torch import BUFFER_KIND
        from .kernels_wp import fill_slab_body_pos_env_local, fill_slab_copy

        slab_kinds = self.spec.slab_buffer_kinds
        slab_assets = self.spec.slab_asset_names
        slab_offsets = self.spec.slab_offsets_py
        slab_sizes = self.spec.slab_sizes_py
        device_str = str(self.device)
        unified_wp = self._bindings.state.unified
        body_pos_w_kind = int(BUFFER_KIND.BODY_POS_W)
        for s in range(len(slab_kinds)):
            kind = slab_kinds[s]
            asset_name = slab_assets[s]
            offset = slab_offsets[s]
            size = slab_sizes[s]
            # Read via the base module so tests can patch BUFFER_KIND_READERS there.
            raw = _base_module.BUFFER_KIND_READERS[kind](self._env, asset_name)
            raw_per_env = raw.numel() // self.num_envs
            if raw_per_env != size:
                raise RuntimeError(
                    f"State kernel output dim mismatch for slab (kind={kind}, asset={asset_name}): "
                    f"reader returned {raw_per_env} floats per env, but slab was sized for {size}."
                )
            source = raw.reshape(self.num_envs, size)
            if kind == body_pos_w_kind:
                # Apply env-origin subtraction during the copy.
                wp.launch(
                    fill_slab_body_pos_env_local,
                    dim=(self.num_envs, size),
                    inputs=[
                        wp.from_torch(source),
                        wp.from_torch(self._env.scene.env_origins),
                        unified_wp,
                        offset,
                    ],
                    device=device_str,
                )
            else:
                wp.launch(
                    fill_slab_copy,
                    dim=(self.num_envs, size),
                    inputs=[wp.from_torch(source), unified_wp, offset],
                    device=device_str,
                )

    # ------------------------------------------------------------------------
    # Phase 2 — mega-kernel launch.
    # ------------------------------------------------------------------------

    def _dispatch_mega_warp(self) -> None:
        """Single ``wp.launch`` over ``(num_envs, k_max)`` — fills the four outputs.

        The env tables are int32 natively and the Warp struct's ``env_slots``
        fields wrap them directly, so no per-step refresh is needed.
        """
        import warp as wp

        from .kernels_wp import dispatch_mega

        b = self._bindings
        wp.launch(
            dispatch_mega,
            dim=(self.num_envs, self.k_max),
            inputs=[b.env_slots, b.spec, b.state, b.outputs],
            device=str(self.device),
        )

    # ------------------------------------------------------------------------
    # Phase 3 — composer kernel launch.
    # ------------------------------------------------------------------------

    def _compose_warp(self) -> None:
        """Single ``wp.launch`` over ``(num_envs,)`` — advances composer state
        and writes terminal reward / success / progress outputs."""
        import warp as wp

        from .kernels_wp import compose_reward

        b = self._bindings
        wp.launch(
            compose_reward,
            dim=(self.num_envs,),
            inputs=[
                b.env_slots,
                b.spec,
                b.composer_state,
                b.outputs,
                wp.from_torch(self._env.episode_length_buf),
                # Per-env effective episode length so the timeout latch honors
                # the adaptive ``tracking_episode_length_min_seconds`` curriculum.
                # Without this the kernel would compare against the global cap
                # and never fire for pure-tracking envs on a randomized window —
                # silently zeroing the terminal reward and breaking learning.
                wp.from_torch(self._effective_max_episode_length),
                0.5,  # instant_threshold (matches reward_composer default)
                float(self.cfg.quality_easing),  # eases multiplicative quality compounding
            ],
            device=str(self.device),
        )
