# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-task command base class + torch/warp factory.

:class:`MultiTaskCommand` is the public name users bind via
``MultiTaskCfg.class_type``. It owns all state that is independent of the
dispatch implementation:

- :class:`~.spec.TaskSpec` build + kernel-id validation.
- Per-env routing buffers (subtask ids, slot counts, offsets, targets) and
  composer latches.
- Public output buffers (errors, activations, canonical commands, terminal
  rewards, success flags, progress) allocated directly on the base term.
- ``_resample_command`` / ``_dispatch_samplers`` — target sampling.
- ``_update_command`` — per-step orchestration (buffer zero →
  ``_dispatch`` → composer → progress).

The dispatch — "given the current state, compute output-layout data" — is
delegated to a subclass via the :meth:`_dispatch` method. Subclasses write
into the base term's public output buffers; backend-private working state
(e.g. primitive-local rows) lives on the backend's plan. Two subclasses
ship with the module:

- :class:`~.impl.multi_task_command_torch.MultiTaskCommandTorch` — PyTorch
  reference, selected by ``cfg.dispatch_backend="torch"``.
- :class:`~.multi_task_command_warp.MultiTaskCommandWarp` — Warp composition root
  switchboard, selected by any non-``"torch"`` backend string.

The :meth:`__new__` factory inspects ``cfg.dispatch_backend`` and returns
an instance of the right subclass. Users never instantiate the subclasses
directly — ``MultiTaskCfg.class_type = MultiTaskCommand`` handles routing.

Design invariants (unchanged by the subclass split):

- Per-step cost is ``O(N × k)`` where ``k = max_slot_count`` across tasks.
- Target buffer is flat ``[N, max_task_total_stride]`` — no ``D_max`` padding.
- Type masks are looked up at read time via ``spec.is_instant[env_subtask_ids]``.

Reward composition: terminal-only multiplicative ``G = reach · mean(A_transit)``.
See :mod:`.reward_composer` and ``~/.claude/plans/id-prefer-you-study-cryptic-quail.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm

# ``ManagerBasedRLEnv`` is used only as a type annotation on ``__new__`` and
# ``__init__``. Importing it eagerly would chain to ``isaaclab.sim`` → ``pxr``
# and trip ``test_env_cfg_no_forbidden_imports`` (cfg construction must be
# possible before SimulationApp launches). ``from __future__ import
# annotations`` above makes string-quoted type hints work at runtime.
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .impl.kernels_torch import (
    ACTIVATION_KERNELS,
    BUFFER_KIND_READERS,
    DELTA_KERNELS,
    METRIC_KERNELS,
    SAMPLER_KERNELS,
    STATE_KERNEL_COMPUTES,
)
from .kernel_ids import STATE_KERNEL_ID

# Note: ``BUFFER_KIND_READERS`` is imported into this module (even though the
# base class doesn't dereference it) so tests can ``patch.object(mtc_mod,
# "BUFFER_KIND_READERS", mock_readers)`` to inject synthetic readers. Both
# subclasses read through this module's attribute at call time so the patch
# takes effect. Don't remove.
_ = BUFFER_KIND_READERS
from .spec import build_spec

if TYPE_CHECKING:
    from .multi_task_cfg import MultiTaskCfg

__all__ = ["MultiTaskCommand"]


class MultiTaskCommand(CommandTerm):
    """Multi-task command term — dispatch-agnostic base.

    Construction routes to a subclass via :meth:`__new__` based on
    ``cfg.dispatch_backend``. Subclasses only need to implement
    :meth:`_dispatch`.
    """

    cfg: MultiTaskCfg

    # ------------------------------------------------------------------------
    # Factory — routes ``MultiTaskCommand(cfg, env)`` to the right subclass.
    # ------------------------------------------------------------------------

    def __new__(cls, cfg: MultiTaskCfg, env: ManagerBasedRLEnv):
        """Return a backend subclass based on :attr:`MultiTaskCfg.dispatch_backend`.

        Subclass instances bypass the factory (``cls is MultiTaskCommand``
        check) so they can be instantiated directly in tests.
        """
        if cls is MultiTaskCommand:
            if cfg.dispatch_backend == "torch":
                from .impl.multi_task_command_torch import MultiTaskCommandTorch

                return object.__new__(MultiTaskCommandTorch)
            # Deferred import avoids a circular dependency at module-load
            # time (the subclass imports from this module).
            from .multi_task_command_warp import MultiTaskCommandWarp

            return object.__new__(MultiTaskCommandWarp)
        return object.__new__(cls)

    # ------------------------------------------------------------------------
    # Shared init — buffer allocation, spec build, initial resample.
    # ------------------------------------------------------------------------

    def __init__(self, cfg: MultiTaskCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._max_episode_length = int(env.max_episode_length)

        # Episode-length randomization (uniform per-reset, pure-tracking only).
        # When ``cfg.tracking_episode_length_min_seconds`` is None the feature
        # is off — ``_effective_max_episode_length`` stays at
        # ``max_episode_length`` for every env. When set, every resample of a
        # pure-tracking env draws a fresh episode length uniformly in
        # ``[_random_episode_min_steps, max_episode_length]``.
        self._random_episode_enabled = cfg.tracking_episode_length_min_seconds is not None
        # ``env.step_dt`` is set by :class:`ManagerBasedRLEnv` from
        # ``cfg.sim.dt * cfg.decimation``; mock envs that don't expose it
        # default to 0.02 s (50 Hz — standard for IsaacLab locomotion).
        step_dt = float(getattr(env, "step_dt", 0.02))
        min_s = cfg.tracking_episode_length_min_seconds or 0.0
        self._random_episode_min_steps = max(1, int(round(min_s / step_dt)))

        # Resolve ``cfg.tasks`` if it's still a :class:`PresetCfg` — this
        # covers direct instantiation paths (unit tests, interactive use)
        # where hydra's ``register_task`` preset walk hasn't run. Hydra's
        # path already replaced the preset with a concrete dict before we
        # got here, so this branch is a no-op in that case.
        from isaaclab_tasks.utils import PresetCfg, resolve_presets  # noqa: PLC0415

        if isinstance(cfg.tasks, PresetCfg):
            resolve_presets(cfg)

        self.spec = build_spec(cfg, env.scene, self.device)
        self._validate()
        self.num_tasks = len(self.spec.task_names)
        self.num_subtasks = int(self.spec.state_kernel_id.numel())
        self.k_max = int(self.spec.task_subtask_ids.shape[1])
        self.max_task_total_stride = int(self.spec.task_total_stride.max().item()) if self.num_tasks > 0 else 0

        device = self.device
        num_envs = self.num_envs
        k_max = self.k_max

        # Per-env task assignment.
        self.task_samples = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Per-env active subtask metadata (derived from task_samples at resample).
        # int32 matches Warp's index-array type so the Warp dispatch can wrap
        # these tensors directly — no per-step int64→int32 ``copy_`` refresh.
        # PyTorch's advanced indexing accepts int32 indices unchanged, so the
        # Torch path works on the same storage.
        self._env_subtask_ids = torch.zeros((num_envs, k_max), dtype=torch.int32, device=device)
        self._env_slot_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self._env_slot_offsets = torch.zeros((num_envs, k_max), dtype=torch.int32, device=device)
        self._env_slot_strides = torch.zeros((num_envs, k_max), dtype=torch.int32, device=device)

        # Flat targets buffer: one float per (env, position-within-task) pair.
        self._targets_flat = torch.zeros((num_envs, max(1, self.max_task_total_stride)), device=device)

        # Per-slot composer state (NO dim padding).
        self._sum_activation = torch.zeros((num_envs, k_max), device=device)
        # ``_transit_steps`` is int32 so the Warp composer can wrap it directly;
        # PyTorch operations on it (``.add_(1)``, ``.to(float)``) work unchanged.
        self._transit_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self._instant_achieved = torch.zeros((num_envs, k_max), dtype=torch.bool, device=device)

        # Per-env effective episode length (in env steps). For reach/mixed
        # envs always ``max_episode_length``; for pure-tracking envs under
        # :attr:`MultiTaskCfg.tracking_episode_length_min_seconds`, a fresh
        # uniform sample is drawn on every resample. Init to full length; the
        # per-reset resample path overrides per-env on its first pass.
        self._effective_max_episode_length = torch.full(
            (num_envs,), self._max_episode_length, dtype=torch.int32, device=device
        )

        # Unified per-step state buffer. The read dispatch writes every slab
        # into this single tensor; the execute dispatch only does
        # ``unified[:, indices]`` gathers — no per-asset lookup downstream.
        self._unified_buffer = torch.zeros((num_envs, max(1, self.spec.unified_width)), device=device)

        # Per-channel active mask — ``1.0`` where the channel is populated
        # by a live subtask of the env's current task, ``0.0`` otherwise.
        # Layout matches :attr:`command` (concatenation of reach + track):
        # ``[0, reach_w)`` is reach, ``[reach_w, reach_w + track_w)`` is track.
        # Refreshed on resample; static per episode (task doesn't change
        # mid-episode under current cfg).
        active_mask_width = max(1, self.spec.reach_canonical_width + self.spec.track_canonical_width)
        self._command_active = torch.zeros((num_envs, active_mask_width), device=device)

        # Public-surface output buffers — read by reward / observation / done
        # terms that consume this command. Allocated directly on the base term;
        # backends write into them through the Warp views built once on the
        # public wrapper (``MultiTaskCommandWarp._build_shared_wp_views``).
        # Anything backend-specific beyond these (e.g. primitive-local rows) is
        # owned by the backend's plan, not here.
        self._buf_error = torch.zeros((num_envs, k_max), device=device)
        self._buf_activation = torch.zeros((num_envs, k_max), device=device)
        self._command_reach = torch.zeros((num_envs, max(1, self.spec.reach_canonical_width)), device=device)
        self._command_track = torch.zeros((num_envs, max(1, self.spec.track_canonical_width)), device=device)
        self._task_reward = torch.zeros(num_envs, device=device)
        self._task_done_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._progress = torch.zeros(num_envs, device=device)

        # Pre-allocated per-step scratch buffers. ``_slot_arange`` is a constant
        # ``[0, 1, ..., k_max - 1]`` row used to derive ``_slot_valid`` via an
        # in-place ``torch.lt`` (refreshed every step). Keeping these on the
        # instance (rather than allocating fresh ``torch.arange`` + comparison
        # tensors each ``_update_command``) removes two hot-path allocations
        # and gives stable pointers — needed for any future CUDA Graph capture.
        self._slot_arange = torch.arange(k_max, device=device, dtype=torch.int32).unsqueeze(0)
        self._slot_valid = torch.zeros((num_envs, k_max), dtype=torch.bool, device=device)

        self._resample_command(torch.arange(num_envs, device=device, dtype=torch.long))

    def _validate(self):
        """Assert every kernel id in the spec is within its registered kernel tuple."""
        assert len(DELTA_KERNELS) == len(METRIC_KERNELS), "DELTA_KERNELS and METRIC_KERNELS must align by id"
        assert int(self.spec.metric_kernel_id.max().item()) < len(METRIC_KERNELS)
        assert int(self.spec.state_kernel_id.max().item()) < len(STATE_KERNEL_COMPUTES)
        assert int(self.spec.activation_kernel_id.max().item()) < len(ACTIVATION_KERNELS)
        assert int(self.spec.sampler_kernel_id.max().item()) < len(SAMPLER_KERNELS)

    # ------------------------------------------------------------------------
    # CommandTerm interface — policy-facing properties.
    # ------------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        """Concatenation of reach + track deltas for the legacy ``generated_commands`` path.

        Prefer :attr:`command_reach` and :attr:`command_track` directly — the
        split carries the semantic (instant goal vs tracking condition) that a
        single flat ``command`` hides.
        """
        return torch.cat([self._command_reach, self._command_track], dim=-1)

    @property
    def command_reach(self) -> torch.Tensor:
        """Delta for instant / "reach" subtasks, shape ``[num_envs, reach_canonical_width]``.

        Per-entity sub-blocks hold only the kernels referenced by instant
        subtasks. Each active instant subtask writes ``target - current`` into
        its ``canonical_offset[:+canonical_stride)`` slice of this tensor.
        """
        return self._command_reach

    @property
    def command_track(self) -> torch.Tensor:
        """Delta for tracking subtasks, shape ``[num_envs, track_canonical_width]``.

        Same positional encoding as :attr:`command_reach`, but populated only by
        tracking subtasks. A kernel used by both instant and tracking subtasks
        gets disjoint channels across the two tensors — no aliasing.
        """
        return self._command_track

    @property
    def command_active(self) -> torch.Tensor:
        """Per-channel active mask, shape ``[num_envs, reach_w + track_w]``.

        ``1.0`` iff the channel at that flat offset is populated by a live
        subtask of the env's current task; ``0.0`` otherwise (including joint
        kernels, which have no canonical projection).

        Layout **mirrors :attr:`command`** (``cat([command_reach,
        command_track], dim=-1)``) column-for-column — the policy can
        concatenate the two to get pairwise (delta, active) pairs aligned
        by index. Concretely: ``column i < reach_w + track_w`` of
        ``command_active`` gates ``column i`` of ``command``.

        This is the disambiguator between "channel is inactive for this
        env's task" and "channel is active but delta happens to be zero":
        both are ``0`` in :attr:`command`, but the former is also ``0`` in
        this mask while the latter is ``1``.

        Refreshed only on resample (static per episode). If you add mid-
        episode task rotation, ``_resample_command`` already handles it —
        the existing call site refreshes this tensor via
        :attr:`TaskSpec.task_active_mask`.
        """
        return self._command_active

    @property
    def progress(self) -> torch.Tensor:
        """Per-env progress ∈ [0, 1], shape ``[num_envs]``.

        Mean of the env's active-subtask activations. ``0`` = far from success on
        every active subtask; ``1`` = all active subtasks at their success band.
        Encodes the "sense of activation std" without exposing the std itself to
        the policy — the value is task-normalized across kernels/params.
        """
        return self._progress

    @property
    def task_reward(self) -> torch.Tensor:
        return self._task_reward

    @property
    def task_done(self) -> torch.Tensor:
        return self._task_done_success

    @property
    def buf_error(self) -> torch.Tensor:
        return self._buf_error

    @property
    def effective_max_episode_length(self) -> torch.Tensor:
        """Per-env episode-end step count [num_envs, int32].

        Reach / mixed envs: always ``env.max_episode_length``.
        Pure-tracking envs (when
        :attr:`MultiTaskCfg.tracking_episode_length_min_seconds` is set): a
        fresh uniform sample in ``[_random_episode_min_steps,
        max_episode_length]`` drawn on every resample. Consumers compare
        against ``env.episode_length_buf`` to detect terminal step.
        """
        return self._effective_max_episode_length

    def _sample_tracking_episode_lengths(self, env_ids: torch.Tensor) -> None:
        """Resample :attr:`_effective_max_episode_length` for given envs.

        Applies to the subset of ``env_ids`` whose current task is pure-
        tracking (no instant subtask). Reach / mixed envs keep the full
        ``max_episode_length``. Called from :meth:`_resample_command` — one
        draw per env per episode. No carried state.
        """
        if not self._random_episode_enabled or env_ids.numel() == 0:
            return
        task_idx = self.task_samples[env_ids]
        is_pure_tracking = ~self.spec.task_has_instant[task_idx]  # [n_envs_reset]
        pt_ids = env_ids[is_pure_tracking]
        if pt_ids.numel() == 0:
            # Reach / mixed envs retain the global cap.
            self._effective_max_episode_length[env_ids] = self._max_episode_length
            return
        # Uniform sample in [min_steps, max_steps] inclusive.
        lengths = torch.randint(
            self._random_episode_min_steps,
            self._max_episode_length + 1,
            (pt_ids.numel(),),
            dtype=torch.int32,
            device=self.device,
        )
        # Reset every reset-env to the global cap first, then overwrite the
        # pure-tracking subset — so a task switch from tracking → reach/mixed
        # clears any leftover short length from the prior episode.
        self._effective_max_episode_length[env_ids] = self._max_episode_length
        self._effective_max_episode_length[pt_ids] = lengths

    # ------------------------------------------------------------------------
    # Debug visualization — dispatched via :data:`kernels_viz.VIZ_REGISTRY`.
    #
    # The registry currently covers the four base-tracking state kernels
    # (``BODY_POS`` / ``BODY_QUAT`` / ``BODY_LIN_VEL`` / ``BODY_ANG_VEL``).
    # Each entry declares its marker cfgs + a per-step ``update_fn``. Joint
    # and contact-count kernels are absent — they don't have an honest
    # spatial primitive. See :mod:`.impl.kernels_viz` for the rationale.
    # ------------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Create per-kernel markers from :data:`kernels_viz.VIZ_REGISTRY`.

        Each registered state kernel declares its own marker list — typically
        a green "goal" + a blue "current" pair so the viewer can compare
        target vs live state at a glance. The registry is pure data; no
        scene access.
        """
        self._debug_vis_enabled = bool(debug_vis)
        if debug_vis:
            if not hasattr(self, "_visualizers"):
                from isaaclab.markers import VisualizationMarkers  # noqa: PLC0415

                from .impl.kernels_viz import VIZ_REGISTRY  # noqa: PLC0415

                self._visualizers: dict[str, object] = {}
                for entry in VIZ_REGISTRY.values():
                    for path, cfg in entry.markers:
                        if path not in self._visualizers:
                            self._visualizers[path] = VisualizationMarkers(cfg)
            for vis in self._visualizers.values():
                vis.set_visibility(True)
        else:
            if hasattr(self, "_visualizers"):
                for vis in self._visualizers.values():
                    vis.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:  # noqa: ARG002
        """Update each registered marker for this step.

        Each entry in :data:`kernels_viz.VIZ_REGISTRY` provides an
        ``update_fn`` returning ``dict[marker_path, kwargs]``; unknown paths
        in the dict are ignored.

        Bails cleanly if the scene isn't ready (early-callback race or
        post-crash exception unwind) so we don't compound upstream errors.
        """
        if not getattr(self, "_debug_vis_enabled", False):
            return
        if not hasattr(self, "spec") or not hasattr(self, "task_samples"):
            return
        if self.num_subtasks == 0:
            return
        try:
            if not self._env.scene["robot"].is_initialized:
                return
        except (KeyError, AttributeError):
            return

        from .impl.kernels_torch import STATE_KERNELS as _SK  # noqa: PLC0415
        from .impl.kernels_viz import VIZ_REGISTRY  # noqa: PLC0415

        per_env_offsets = self.spec.task_kernel_target_offset[self.task_samples]  # [N, K]
        env_arange = torch.arange(self.num_envs, device=self.device)
        for kid, entry in VIZ_REGISTRY.items():
            offsets = per_env_offsets[:, kid]
            active = offsets >= 0
            # Skip kernels that no env's current task uses. Without this we
            # gather junk from ``_targets_flat`` at the (-1 → clamp(0)) slot
            # — for kernels like ``BODY_QUAT`` (stride 4) the gathered slice
            # is whatever happens to live at the first 4 cols of the flat
            # buffer (e.g. ``[vx, vy, vz, count]``), which is NOT a unit
            # quaternion. Passing it to ``visualize(orientations=...)`` then
            # crashes inside USD's ``Vt.QuathArray.FromNumpy``. Skipping at
            # the top is correct (and faster) — fully-inactive kernels have
            # nothing useful to draw.
            if not bool(active.any()):
                continue
            stride = _SK[kid].intra_body_stride
            col_idx = offsets.clamp(min=0).unsqueeze(1) + torch.arange(stride, device=self.device).unsqueeze(0)
            row_idx = env_arange.unsqueeze(1).expand_as(col_idx)
            target_per_env = self._targets_flat[row_idx, col_idx]
            try:
                update = entry.update_fn(self, target_per_env, active)
                for path, kwargs in update.items():
                    if path in self._visualizers:
                        self._visualizers[path].visualize(**kwargs)
            except Exception as exc:  # noqa: BLE001
                # One viz_fn failure shouldn't kill the others. Print so
                # Kit doesn't silently swallow it.
                print(f"[multitask-viz] viz update for kid={kid} raised: {exc!r}")

    def _rotate_canonical_slots_to_body_frame(self) -> None:
        """Rotate POS / LIN_VEL / ANG_VEL slots of :attr:`_command_reach` /
        :attr:`_command_track` from world frame into the originating asset's
        root frame.

        The composer's contract is: state kernels and deltas are world-frame;
        the reward (via :func:`METRIC_KERNELS`) uses L2 norms that are
        rotation-invariant, so reward math is unaffected. This helper is
        called at the end of each dispatch (Torch or Warp) so that the
        policy-facing obs tensors come out body-aligned — downstream
        :func:`~..mdp.observations.command_reach` /
        :func:`~..mdp.observations.command_track` read them directly without
        any frame logic.

        For envs whose current task omits a rotatable slot, the slot's value
        was never written this step (dispatch only scatters active slots;
        inactive slots hold the resample-time zero). Rotating zero is a no-op
        so no state leaks.
        """
        from isaaclab.utils.math import quat_apply_inverse  # noqa: PLC0415

        def _as_torch(q):
            # Real articulations expose ``root_quat_w`` as a ``ProxyArray``;
            # the pure-torch mock test (``test_multi_task_command_mock``)
            # stores a torch tensor directly. Handle both.
            if isinstance(q, torch.Tensor):
                return q
            return q.torch

        for asset_name, offsets in self.spec.reach_rotatable_vec3_by_asset.items():
            if not offsets:
                continue
            root_quat_w = _as_torch(self._env.scene[asset_name].data.root_quat_w)  # [N, 4]
            for off in offsets:
                self._command_reach[:, off : off + 3] = quat_apply_inverse(
                    root_quat_w, self._command_reach[:, off : off + 3]
                )
        for asset_name, offsets in self.spec.track_rotatable_vec3_by_asset.items():
            if not offsets:
                continue
            root_quat_w = _as_torch(self._env.scene[asset_name].data.root_quat_w)  # [N, 4]
            for off in offsets:
                self._command_track[:, off : off + 3] = quat_apply_inverse(
                    root_quat_w, self._command_track[:, off : off + 3]
                )

    def _update_metrics(self):
        """Expose pre-activation error terms (bucketed by state kernel) to
        :class:`CommandManager`'s logger.

        ``CommandManager.reset()`` runs AFTER this step's ``_update_command``
        has written :attr:`_buf_error` for envs terminating this step, so the
        per-env rows here reflect the last-step error. Logged under
        ``Metrics/<command_name>/error/<state_kernel>`` (e.g.
        ``error/body_pos`` [m], ``error/body_quat`` [rad],
        ``error/body_lin_vel`` [m/s]). Envs whose current task has no slot of
        a kernel contribute ``0`` to that kernel's mean — the metric's
        magnitude is the task-weighted mean across tasks that *do* use it.

        ``task_reward`` / ``progress`` are deliberately NOT logged here:
        the terminal reward shows up as ``Episode_Reward/task`` from the
        reward manager, and ``progress`` is redundant against the per-kernel
        error bucket below.
        """
        # Bucket per-slot errors by state kernel so each physical quantity
        # (pos, quat, lin_vel, ...) gets its own scalar metric. The spec's
        # ``state_kernel_id`` is ``[num_subtasks]``; gathering with
        # ``_env_subtask_ids`` lifts that to ``[num_envs, k_max]``.
        if self.num_subtasks == 0:
            return
        sk_per_slot = self.spec.state_kernel_id[self._env_subtask_ids.long()]  # [E, k_max]
        slot_range = torch.arange(self.k_max, device=self.device).unsqueeze(0)
        active_mask = slot_range < self._env_slot_count.long().unsqueeze(-1)  # [E, k_max]
        for sk in STATE_KERNEL_ID:
            sk_slots = (sk_per_slot == int(sk)) & active_mask  # [E, k_max]
            # Only emit kernels that actually appear in the current task set
            # — otherwise the metric is a constant zero and clutters the log.
            if not sk_slots.any():
                continue
            count = sk_slots.sum(dim=-1)  # [E]
            summed = (self._buf_error * sk_slots).sum(dim=-1)  # [E]
            # Per-env mean over slots-of-this-kernel; envs without this
            # kernel in their current task → 0. ``reset()``'s mean over
            # done envs is thus a task-weighted mean of the kernel's error.
            per_env = torch.where(count > 0, summed / count.clamp(min=1).float(), summed)
            self.metrics[f"error/{sk.name.lower()}"] = per_env

    # ------------------------------------------------------------------------
    # Resample — target sampling (shared across dispatch paths).
    # ------------------------------------------------------------------------

    def resample_indices(self, env_ids: torch.Tensor) -> None:
        """Assign fresh random task indices to the specified envs."""
        self.task_samples[env_ids] = torch.randint(0, self.num_tasks, (env_ids.numel(),), device=self.device)

    def _resample_command(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self.resample_indices(env_ids)

        task_idx = self.task_samples[env_ids]  # [N']

        # Gather per-env active-subtask metadata directly from the task tables.
        # Padded slots have subtask_id=-1 and stride/offset 0 in the spec tables; the
        # hot path filters them out via ``_env_slot_count`` so we never dereference them.
        #
        # ``.to(int32)`` matches the env-table dtype (base class stores int32 so
        # the Warp dispatch can wrap the tensors directly). This runs once per
        # resample, not per step — not a hot path.
        self._env_subtask_ids[env_ids] = self.spec.task_subtask_ids[task_idx].to(torch.int32)
        self._env_slot_count[env_ids] = self.spec.task_slot_count[task_idx].to(torch.int32)
        self._env_slot_offsets[env_ids] = self.spec.task_slot_offsets[task_idx].to(torch.int32)
        self._env_slot_strides[env_ids] = self.spec.task_slot_strides[task_idx].to(torch.int32)

        # Clear per-env composer state for the resampled envs. Must happen BEFORE the
        # sampler dispatch so the dispatch's writes to ``_targets_flat`` aren't clobbered.
        self._sum_activation[env_ids] = 0.0
        self._transit_steps[env_ids] = 0
        self._instant_achieved[env_ids] = False
        # Clear per-env public outputs so a prior task's values don't leak.
        self._task_reward[env_ids] = 0.0
        self._task_done_success[env_ids] = False
        self._progress[env_ids] = 0.0
        self._command_reach[env_ids] = 0.0
        self._command_track[env_ids] = 0.0

        # Refresh the per-channel active mask from the spec's per-task table.
        # ``task_active_mask`` is entirely a function of the spec, so this is
        # a single indexed copy — no runtime reconstruction. Must happen on
        # EVERY resample (including failure resets) so a prior task's channels
        # don't leak ``1.0`` values into an env that now has fewer active
        # channels.
        self._command_active[env_ids] = self.spec.task_active_mask[task_idx]

        # Draw a fresh random episode length for each pure-tracking env (if
        # the feature is enabled via ``cfg.tracking_episode_length_min_seconds``).
        # No-op for reach / mixed envs, which always use ``max_episode_length``.
        self._sample_tracking_episode_lengths(env_ids)

        # Inactive positions of ``_targets_flat`` may carry stale values from prior
        # tasks; harmless since those positions are never read (slot_count masking
        # confines reads to ``[offset : offset+stride]`` of each active slot).

        # Sample fresh targets for these envs' active subtasks, writing into the flat
        # target buffer at each slot's (offset, stride).
        self._dispatch_samplers(env_ids)
        self._on_resample_command(env_ids)

    def _on_resample_command(self, env_ids: torch.Tensor) -> None:
        """Hook for backend-owned execution plans after task assignment changes."""
        del env_ids

    def _dispatch_samplers(self, env_ids: torch.Tensor) -> None:
        """Run sampler kernels per active (env, slot), writing to the flat target buffer."""
        if env_ids.numel() == 0:
            return

        # Build the instance list: every (env, slot) that is active for the resampled envs.
        n = env_ids.numel()
        slot_idx = torch.arange(self.k_max, device=self.device).unsqueeze(0).expand(n, -1)  # [n, k_max]
        valid = slot_idx < self._env_slot_count[env_ids].unsqueeze(1)  # [n, k_max]
        if not valid.any():
            return
        env_local_idx, slot_flat = valid.nonzero(as_tuple=True)  # each [n_active]
        env_global = env_ids[env_local_idx]  # [n_active]
        subtask_ids = self._env_subtask_ids[env_global, slot_flat]  # [n_active]
        offsets = self._env_slot_offsets[env_global, slot_flat]  # [n_active]
        strides = self._env_slot_strides[env_global, slot_flat]  # [n_active]

        sampler_kids = self.spec.sampler_kernel_id[subtask_ids]  # [n_active]
        params_all = self.spec.sampler_kernel_param[subtask_ids]  # [n_active, Pmax]

        # Dispatch by unique ``(sampler_kid, stride)`` pair rather than just ``sampler_kid``.
        # Within one kid, different subtasks can have different strides (e.g.
        # ``UNIFORM`` with ``minimum=[x, y, z]`` vs ``UNIFORM`` with
        # ``minimum=[a, b, c, d, e]``). If we pooled them, the globally-padded params
        # would produce samples at the pool's max width — writing stride-5 values
        # into a stride-3 slot's slice corrupts the neighbour. Grouping by
        # ``(kid, stride)`` keeps each batch rectangular and correct.
        kid_stride = torch.stack([sampler_kids, strides], dim=1)  # [n_active, 2]
        unique_pairs = torch.unique(kid_stride, dim=0)
        for pair in unique_pairs:
            sampler_kid = int(pair[0].item())
            stride_val = int(pair[1].item())
            group_mask = (sampler_kids == sampler_kid) & (strides == stride_val)
            if not group_mask.any():
                continue

            # Slice params to exactly ``stride_val``'s worth of ``(min, range)`` pairs so
            # the sampler kernel doesn't read past the subtask's real param length into
            # zero padding. ``UNIFORM`` with stride 3 gets ``params[:, :6]``; ``UNIFORM``
            # with stride 5 gets ``params[:, :10]``. ``EULER_UNIFORM_TO_QUAT`` with
            # stride 4 gets ``params[:, :8]`` — it internally only reads the first 6
            # Euler pairs, so the extra 2 (which are zero padding anyway) are ignored.
            params_group = params_all[group_mask, : 2 * stride_val]
            samples = SAMPLER_KERNELS[sampler_kid](params_group)  # [n_group, stride_val]
            assert int(samples.shape[1]) == stride_val, (
                f"sampler kid={sampler_kid} emitted dim {samples.shape[1]} for sliced "
                f"params of length {2 * stride_val} — expected stride {stride_val}"
            )

            sub_env = env_global[group_mask]
            sub_off = offsets[group_mask]
            arange_s = torch.arange(stride_val, device=self.device).unsqueeze(0)
            col_idx = sub_off.unsqueeze(1) + arange_s
            row_idx = sub_env.unsqueeze(1).expand_as(col_idx)
            self._targets_flat[row_idx, col_idx] = samples.to(self._targets_flat.dtype)

    # ------------------------------------------------------------------------
    # Per-step update — orchestrates dispatch + composer + progress.
    # ------------------------------------------------------------------------

    def _update_command(self):
        """Per-step update: dispatch → compose. Both override points are
        implemented by the Torch and Warp subclasses; this method just
        orchestrates buffer clearing and delegation.

        Zero per-step allocations: the slot-valid mask is written in place
        into the pre-allocated :attr:`_slot_valid` buffer. The selected output
        store prepares its own per-step buffers; dense stores clear slot
        tensors, while local stores can avoid dense materialization.
        """
        # Active-slot mask — refresh in place. ``_slot_arange`` is constant.
        torch.lt(self._slot_arange, self._env_slot_count.unsqueeze(1), out=self._slot_valid)

        # Clear per-step dense slot tensors before dispatch. The Warp subclass
        # overrides this method entirely and skips these zeros (dispatch over-
        # writes every active slot; inactive slots are masked in compose).
        self._buf_error.zero_()
        self._buf_activation.zero_()

        # Delegate to the subclass — both phases are polymorphic.
        self._dispatch(self._slot_valid)
        self._compose(self._slot_valid)

    # ------------------------------------------------------------------------
    # Dispatch — overridden by each subclass.
    # ------------------------------------------------------------------------

    def _dispatch(self, valid_slots: torch.Tensor) -> None:
        """Compute backend-owned per-step output-layout data.

        Must be implemented by a subclass. See
        :class:`~.impl.multi_task_command_torch.MultiTaskCommandTorch` and
        :class:`~.multi_task_command_warp.MultiTaskCommandWarp`. Public
        tensors exposed through :attr:`command_reach`, :attr:`command_track`,
        :attr:`task_reward`, and :attr:`progress` must remain semantically
        equivalent across backends.
        """
        raise NotImplementedError(
            f"{type(self).__name__}._dispatch must be overridden. Did you instantiate "
            "MultiTaskCommand directly without going through the __new__ factory?"
        )

    def _compose(self, valid_slots: torch.Tensor) -> None:
        """Advance composer state + write ``task_reward`` / ``task_done_success`` / ``progress``.

        Specifically: accumulate ``_sum_activation``, increment
        ``_transit_steps``, latch ``_instant_achieved``, and write the three
        per-env outputs (terminal multiplicative reward, success flag, mean
        activation progress). Must be implemented by a subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__}._compose must be overridden. Did you instantiate "
            "MultiTaskCommand directly without going through the __new__ factory?"
        )
