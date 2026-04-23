# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-task command term with multiplicative terminal reward — ragged runtime layout.

Design invariants (Stage 2.0, "arbitrary M"):

- Per-step cost is ``O(N × k)`` where ``k = max_slot_count`` across tasks. ``M`` (total
  unique subtasks) appears only in spec tables, never in per-env state. A cfg with
  ``M = 10_000`` subtasks costs the same per step as ``M = 10``.
- Target buffer is **flat**, not padded by ``D_max``. Layout is ``[N, max_task_total_stride]``;
  each slot reads/writes a ``stride[m]``-wide slice at an offset determined by the env's
  assigned task. Mixed-stride tasks (e.g. joint_pos=20 alongside body_pos=3) no longer
  allocate ``D_max`` floats per slot.
- Type masks are looked up at read time via ``spec.is_instant[env_subtask_ids]`` instead
  of being scatter-built per-env at resample. Simpler, always correct.

Reward composition: terminal-only multiplicative ``G = reach · mean(A_transit)``. See
``reward_composer.py`` and ``~/.claude/plans/id-prefer-you-study-cryptic-quail.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm

from .kernels import ACTIVATION_KERNELS, DELTA_KERNELS, METRIC_KERNELS, SAMPLER_KERNELS, STATE_KERNELS
from .reward_composer import multiplicative_terminal_reward
from .spec import build_spec

if TYPE_CHECKING:
    from .multi_task_cfg import MultiTaskCfg

__all__ = ["MultiTaskCommand"]


class MultiTaskCommand(CommandTerm):
    """Multi-task command term emitting a terminal multiplicative reward.

    Runtime state is sized by ``k_max`` (max slots per task), not ``M`` (total unique
    subtasks). Arbitrary ``M`` is supported without per-step cost growth.
    """

    cfg: MultiTaskCfg

    def __init__(self, cfg: MultiTaskCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._max_episode_length = int(env.max_episode_length)

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
        self._env_subtask_ids = torch.zeros((num_envs, k_max), dtype=torch.long, device=device)
        self._env_slot_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._env_slot_offsets = torch.zeros((num_envs, k_max), dtype=torch.long, device=device)
        self._env_slot_strides = torch.zeros((num_envs, k_max), dtype=torch.long, device=device)

        # Flat targets buffer: one float per (env, position-within-task) pair.
        self._targets_flat = torch.zeros((num_envs, max(1, self.max_task_total_stride)), device=device)

        # Per-slot scalar runtime state (NO dim padding).
        self._sum_activation = torch.zeros((num_envs, k_max), device=device)
        self._buf_error = torch.zeros((num_envs, k_max), device=device)
        self._buf_activation = torch.zeros((num_envs, k_max), device=device)
        self._transit_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._instant_achieved = torch.zeros((num_envs, k_max), dtype=torch.bool, device=device)

        # Composer outputs (refreshed every step).
        self._task_reward = torch.zeros(num_envs, device=device)
        self._task_done_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Observation-facing command tensor (flattened target minus state per slot).
        # Shape ``[num_envs, k_max * max_state_stride_per_slot]`` — we reuse the flat
        # target buffer's per-slot stride directly; obs readers see ``max_task_total_stride``
        # floats per env (target - state stacked per slot).
        self._command = torch.zeros((num_envs, max(1, self.max_task_total_stride)), device=device)

        # Per-step caches computed inside ``_update_command``, exposed as properties so
        # observation terms can read them without recomputing. All shape ``[N, k_max]``.
        # See the "activation-as-obs" design note — the policy gets a task-normalized
        # progress signal (``_slot_activation``) plus structural bits (``_slot_is_instant``,
        # ``_slot_is_tracking``, ``_slot_valid``) instead of reward-shaping parameters.
        self._slot_valid = torch.zeros((num_envs, k_max), dtype=torch.bool, device=device)
        self._slot_is_instant = torch.zeros((num_envs, k_max), dtype=torch.bool, device=device)
        self._slot_is_tracking = torch.zeros((num_envs, k_max), dtype=torch.bool, device=device)

        self._resample_command(torch.arange(num_envs, device=device, dtype=torch.long))

    def _validate(self):
        """Assert every kernel id in the spec is within its registered kernel tuple."""
        assert len(DELTA_KERNELS) == len(METRIC_KERNELS), "DELTA_KERNELS and METRIC_KERNELS must align by id"
        assert int(self.spec.metric_kernel_id.max().item()) < len(METRIC_KERNELS)
        assert int(self.spec.state_kernel_id.max().item()) < len(STATE_KERNELS)
        assert int(self.spec.activation_kernel_id.max().item()) < len(ACTIVATION_KERNELS)
        assert int(self.spec.sampler_kernel_id.max().item()) < len(SAMPLER_KERNELS)

    # ------------------------------------------------------------------------
    # CommandTerm interface
    # ------------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        """Flattened delta (target - current) laid out per the env's task's flat layout.

        Shape ``[num_envs, max_task_total_stride]``. Every step this is cleared and
        re-populated only at active-slot positions, so envs with shorter tasks have
        zeros beyond ``task_total_stride[task_samples[env]]``. Policy observation code
        can read the full tensor; trailing zeros are semantically "no data."
        """
        return self._command

    @property
    def task_reward(self) -> torch.Tensor:
        return self._task_reward

    @property
    def task_done(self) -> torch.Tensor:
        return self._task_done_success

    @property
    def buf_error(self) -> torch.Tensor:
        return self._buf_error

    # ---- Per-slot observation feeds ---------------------------------------
    # These expose the engine's own "how well is each subtask doing" signal in a
    # task-normalized, reward-shape-agnostic form. Policy observations should prefer
    # these over raw activation-kernel parameters (``std`` / ``threshold``), which
    # are implementation details of the reward shaping, not task semantics. See the
    # "activation-as-obs" design note for the rationale.

    @property
    def slot_activation(self) -> torch.Tensor:
        """Per-slot activation ``∈ [0, 1]``, shape ``[N, k_max]``.

        For tracking slots: rises as the error shrinks (``1 - tanh(err/std)``).
        For instant slots: ``0`` or ``1`` by the activation kernel's threshold.
        Padded slots are zero. The policy's "how close am I?" signal.
        """
        return self._buf_activation

    @property
    def slot_is_instant(self) -> torch.Tensor:
        """Per-slot "must-achieve" flag, shape ``[N, k_max]`` bool.

        Tells the policy whether this slot is a one-time reach (once achieved, it
        stays latched; policy can divert attention) or requires sustained effort.
        Padded slots are ``False``.
        """
        return self._slot_is_instant

    @property
    def slot_is_tracking(self) -> torch.Tensor:
        """Per-slot "must-maintain" flag, shape ``[N, k_max]`` bool.

        Tells the policy this slot's activation is time-averaged into the terminal
        reward — it must be held high every step, not just achieved once. Padded
        slots are ``False``.
        """
        return self._slot_is_tracking

    @property
    def slot_valid(self) -> torch.Tensor:
        """Per-slot active-in-this-env's-task flag, shape ``[N, k_max]`` bool.

        ``True`` iff the slot index is less than the env's assigned task's slot
        count. Lets policy ignore padded slot features.
        """
        return self._slot_valid

    def _update_metrics(self):
        pass

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
        self._env_subtask_ids[env_ids] = self.spec.task_subtask_ids[task_idx]
        self._env_slot_count[env_ids] = self.spec.task_slot_count[task_idx]
        self._env_slot_offsets[env_ids] = self.spec.task_slot_offsets[task_idx]
        self._env_slot_strides[env_ids] = self.spec.task_slot_strides[task_idx]

        # Clear per-env composer state for the resampled envs. Must happen BEFORE the
        # sampler dispatch so the dispatch's writes to ``_targets_flat`` aren't clobbered.
        self._sum_activation[env_ids] = 0.0
        self._transit_steps[env_ids] = 0
        self._instant_achieved[env_ids] = False
        self._task_reward[env_ids] = 0.0
        self._task_done_success[env_ids] = False
        self._command[env_ids] = 0.0
        # Inactive positions of ``_targets_flat`` may carry stale values from prior
        # tasks; harmless since those positions are never read (slot_count masking
        # confines reads to ``[offset : offset+stride]`` of each active slot).

        # Sample fresh targets for these envs' active subtasks, writing into the flat
        # target buffer at each slot's (offset, stride).
        self._dispatch_samplers(env_ids)

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
    # Per-step update
    # ------------------------------------------------------------------------

    def _update_command(self):
        """Per-step update: state → delta → metric → activation → composer.

        Also refreshes the observation-facing per-slot caches (``_slot_valid``,
        ``_slot_is_instant``, ``_slot_is_tracking``) so obs terms read them directly
        without recomputing.
        """
        # Build the "valid slot" mask once, used by multiple stages + cached for obs.
        slot_arange = torch.arange(self.k_max, device=self.device).unsqueeze(0)  # [1, k_max]
        self._slot_valid = slot_arange < self._env_slot_count.unsqueeze(1)  # [N, k_max]

        # Reset per-step buffers that we'll be scattering into.
        self._buf_error.zero_()
        self._buf_activation.zero_()
        self._command.zero_()

        # ---- State + delta + metric dispatch -------------------------------
        # Iterate over unique (state_kid, entity) groups, read state once per group,
        # then compute delta/error for all matching active slots.
        self._compute_state_delta_error(self._slot_valid)

        # ---- Activation dispatch -------------------------------------------
        self._compute_activation(self._slot_valid)

        # ---- Accumulate ----------------------------------------------------
        self._sum_activation = self._sum_activation + self._buf_activation
        self._transit_steps = self._transit_steps + 1

        # ---- Per-slot type masks (obs-facing + composer input) -------------
        # Look up is_instant / is_tracking per slot via gather, AND'd with valid_slots
        # so padded entries are ``False`` regardless of the clamp_min(0) placeholder.
        safe_subtask_ids = self._env_subtask_ids.clamp_min(0)  # [N, k_max]
        self._slot_is_instant = self.spec.is_instant[safe_subtask_ids] & self._slot_valid
        self._slot_is_tracking = self.spec.is_tracking[safe_subtask_ids] & self._slot_valid

        # ---- Composer ------------------------------------------------------
        is_timeout = self._env.episode_length_buf >= self._max_episode_length

        out = multiplicative_terminal_reward(
            activation_scores=self._buf_activation,
            is_instant_subtask=self._slot_is_instant,
            is_tracking_subtask=self._slot_is_tracking,
            sum_activation=self._sum_activation,
            transit_steps=self._transit_steps,
            instant_achieved=self._instant_achieved,
            is_timeout=is_timeout,
        )
        self._instant_achieved = out.instant_achieved_next
        self._task_reward = out.reward
        self._task_done_success = out.done_success

    # ------------------------------------------------------------------------
    # Internal dispatch helpers
    # ------------------------------------------------------------------------

    def _compute_state_delta_error(self, valid_slots: torch.Tensor) -> None:
        """Read state per unique (state_kid, entity) group; compute delta + error for each active slot.

        Stages:
        1. Identify unique (state_kid, entity) pairs with ≥ 1 active slot.
        2. For each pair, call the state kernel once → ``x_cur[N, stride]``.
        3. For each metric_kid among its subtasks, batch-compute delta + error and
           scatter into ``_buf_error[N, k_max]`` and ``_command`` (flat delta).
        """
        if self.num_subtasks == 0:
            return

        # Subtask ids per (env, slot), with -1 placeholders for padding.
        safe_subtask_ids = self._env_subtask_ids.clamp_min(0)  # [N, k_max]

        # Which (env, slot) pairs are live THIS STEP.
        live = valid_slots  # [N, k_max] bool

        # Spec lookups per slot.
        state_kids_per_slot = self.spec.state_kernel_id[safe_subtask_ids]  # [N, k_max]
        metric_kids_per_slot = self.spec.metric_kernel_id[safe_subtask_ids]  # [N, k_max]
        entity_per_slot = self.spec.subtask_entity_id[safe_subtask_ids]  # [N, k_max]

        # Iterate unique (state_kid, entity) pairs. To avoid materializing a 2D unique,
        # do a nested loop over unique state_kids then unique entities per kid.
        live_state_kids = state_kids_per_slot[live]
        if live_state_kids.numel() == 0:
            return
        for state_kid in torch.unique(live_state_kids).tolist():
            state_kid = int(state_kid)
            mask_state = live & (state_kids_per_slot == state_kid)
            if not mask_state.any():
                continue
            ent_ids = entity_per_slot[mask_state]
            for ent_id in torch.unique(ent_ids).tolist():
                ent_id = int(ent_id)
                mask_group = mask_state & (entity_per_slot == ent_id)
                if not mask_group.any():
                    continue

                # Pick an example subtask from the group to get its asset_cfg.
                example_env, example_slot = mask_group.nonzero(as_tuple=True)
                example_subtask_id = int(safe_subtask_ids[example_env[0], example_slot[0]].item())
                asset_cfg = self.spec.subtask_asset_cfgs[example_subtask_id]
                stride = int(self.spec.state_stride[example_subtask_id].item())

                # Read state for this (state_kid, entity).
                x_cur = STATE_KERNELS[state_kid](self._env, slice(None), asset_cfg)  # [N, ...]
                # Sanity: the state kernel's output dim must match the spec's state_stride
                # for subtasks in this group. If a user cfg's sampler declares e.g. out_dim=4
                # while the state kernel emits 3 floats, the reshape below will fail with an
                # opaque torch error — catch it here with a descriptive message.
                trailing_numel = 1
                for d in x_cur.shape[1:]:
                    trailing_numel *= int(d)
                if trailing_numel != stride:
                    raise RuntimeError(
                        f"State kernel output dim mismatch for subtask {example_subtask_id} "
                        f"(state_kid={state_kid}, entity={ent_id}, asset={asset_cfg.name}): "
                        f"kernel emitted shape {tuple(x_cur.shape[1:])} (numel {trailing_numel}), "
                        f"but state_stride is {stride}. Fix the cfg so the sampler's output "
                        f"dim matches the state kernel's output."
                    )
                x_cur = x_cur.reshape(self.num_envs, stride)  # [N, stride]

                # For each metric_kid within this group, batch-compute delta + error.
                metric_kids_group = metric_kids_per_slot[mask_group]
                for mkid in torch.unique(metric_kids_group).tolist():
                    mkid = int(mkid)
                    mask_mk = mask_group & (metric_kids_per_slot == mkid)
                    if not mask_mk.any():
                        continue
                    flat_env, flat_slot = mask_mk.nonzero(as_tuple=True)  # each [n_match]

                    # Gather current state for matched envs.
                    x_cur_m = x_cur[flat_env]  # [n_match, stride]

                    # Gather target slice from the flat buffer.
                    offsets_m = self._env_slot_offsets[flat_env, flat_slot]  # [n_match]
                    arange_s = torch.arange(stride, device=self.device).unsqueeze(0)
                    col_idx = offsets_m.unsqueeze(1) + arange_s  # [n_match, stride]
                    row_idx = flat_env.unsqueeze(1).expand_as(col_idx)
                    tgt_m = self._targets_flat[row_idx, col_idx]  # [n_match, stride]

                    # Delta (paired with metric by kernel id) and scalar metric.
                    delta_m = DELTA_KERNELS[mkid](x_cur_m, tgt_m)  # [n_match, stride]
                    err_m = METRIC_KERNELS[mkid](delta_m)  # [n_match]

                    # Scatter error back to [N, k_max].
                    self._buf_error[flat_env, flat_slot] = err_m.to(self._buf_error.dtype)

                    # Write delta slice to the observation-facing command tensor,
                    # at the same offsets the target occupied (same flat layout).
                    self._command[row_idx, col_idx] = delta_m.to(self._command.dtype)

    def _compute_activation(self, valid_slots: torch.Tensor) -> None:
        """Activation kernel dispatch over unique activation_kids in live slots."""
        if self.num_subtasks == 0:
            return

        safe_subtask_ids = self._env_subtask_ids.clamp_min(0)
        activation_kids_per_slot = self.spec.activation_kernel_id[safe_subtask_ids]  # [N, k_max]
        activation_params_per_slot = self.spec.activation_kernel_param[safe_subtask_ids]  # [N, k_max]

        live_activation_kids = activation_kids_per_slot[valid_slots]
        if live_activation_kids.numel() == 0:
            return
        for akid in torch.unique(live_activation_kids).tolist():
            akid = int(akid)
            mask = valid_slots & (activation_kids_per_slot == akid)
            if not mask.any():
                continue
            err_m = self._buf_error[mask]
            params_m = activation_params_per_slot[mask]
            out = ACTIVATION_KERNELS[akid](err_m, params_m).to(self._buf_activation.dtype)
            self._buf_activation[mask] = out
