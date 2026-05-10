# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-agnostic reset-strategy combinators.

Three :class:`~isaaclab.managers.ManagerTermBase` subclasses that compose
reset strategies without baking in robot or task semantics:

- :class:`reset_accumulator` — pre-collect validated reset states into a
  shared ring buffer, then sample from it. Optional ``keep_accumulating``
  appends new validated states at runtime.
- :class:`TermChoice` — partition envs over a set of sub-event-terms;
  uniform or success-rate-weighted sampling.
- :class:`ChainedResetTerms` — sequentially apply a list of reset terms,
  optionally gated by a per-env Bernoulli probability.

The combinators couple to a ``progress_context`` termination term that
exposes ``.is_success``: a per-env bool tensor read at runtime to update
the success monitor. Any domain (locomotion, manipulation, …) can satisfy
this contract by registering a termination term named ``progress_context``
whose function exposes an ``.is_success`` attribute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from isaaclab.managers import EventTermCfg, ManagerTermBase

from . import reset_state
from .diagnostics import log_sampler_bins
from .sampling import FrontierSamplingStrategyCfg, Sampler, SamplerCfg
from .state_buffer import StateBuffer
from .state_buffer_cfg import StateBufferCfg
from .state_layout import StateLayout
from .success_monitor_cfg import SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _tagged_report(
    values: torch.Tensor,
    tags: torch.Tensor,
    tag_names: list[str],
    reduction: str = "sum",
) -> dict[str, float]:
    """Aggregate per-slot ``values`` by tag (private to this module)."""
    out: dict[str, float] = {}
    for i, name in enumerate(tag_names):
        mask = tags == i
        if not mask.any():
            out[name] = 0.0
        elif reduction == "mean":
            out[name] = values[mask].mean().item()
        else:
            out[name] = values[mask].sum().item()
    return out


class reset_accumulator(ManagerTermBase):
    """Accumulate validated reset states into a shared buffer and sample from it.

    During the pre-collection phase, reset states are generated and validated
    against acceptance conditions until the buffer is full. After that, every
    call samples from the buffer. Optionally keeps accumulating new valid states
    at runtime (``keep_accumulating=True``).

    All envs share a single ring buffer of validated states stored in
    env-origin-relative coordinates, so any env can be reset to any stored state.
    """

    _shared_buffer: StateBuffer | None = None

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.acceptance_conditions = cfg.params["acceptance_conditions"]
        for key, val in self.acceptance_conditions.items():
            if hasattr(val, "class_type"):
                self.acceptance_conditions[key] = val.class_type(val, env)

        # Filter to scene-resident assets: callers may include "optional" assets
        # that only exist for a subset of presets.
        self._requested_reset_assets = list(cfg.params["reset_assets"])
        present = set(env.scene._articulations) | set(env.scene._rigid_objects)
        self.reset_assets = [a for a in self._requested_reset_assets if a in present]
        state_dim = reset_state.get_reset_state(
            self._env,
            torch.tensor([0], device=env.device),
            self.reset_assets,
        ).shape[-1]
        self._state_buffer_cfg: StateBufferCfg = cfg.params["state_buffer_cfg"]
        buf_cfg = self._state_buffer_cfg
        target_size = int(buf_cfg.size)
        # Oversample-then-thin: buffer fills to ``oversample_capacity`` then
        # compacts back to ``target_size``. Parallel arrays (monitor history,
        # success rate) are sized to the oversample upper bound so they can
        # cover every reachable slot before compaction.
        oversample_capacity = max(target_size, int(target_size * float(buf_cfg.oversample_ratio)))
        max_size = oversample_capacity

        self.state_buffer = StateBuffer(
            max_size=oversample_capacity,
            state_dim=state_dim,
            device=env.device,
            target_size=target_size,
            fps_features=buf_cfg.fps_features,
        )
        self.state_buffer.register_compact_callback(self._on_state_buffer_compact)
        reset_accumulator._shared_buffer = self.state_buffer
        self.sampled_slots = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.precollecting_phase = True
        self._tag_indices_bind: str | None = buf_cfg.tag_indices_bind
        self._tag_names_resolved = False
        self._sampling_cfg: SamplerCfg = cfg.params["sampling"]
        if self._sampling_cfg.max_samples is None:
            self._sampling_cfg.max_samples = env.num_envs

        self.monitor_success_rate = torch.zeros(max_size, device=env.device)
        monitor_cfg: SuccessMonitorCfg = cfg.params["success_monitor_cfg"]
        monitor_cfg.num_monitored_data = max_size
        monitor_cfg.device = env.device
        monitor_cfg.max_updates = env.num_envs if monitor_cfg.max_updates is None else monitor_cfg.max_updates
        self.success_monitor = monitor_cfg.class_type(monitor_cfg, self.monitor_success_rate)

        self.success_rate = torch.zeros(max_size, device=env.device)
        # Sampler: weighted-sum over sampling strategies. Built
        # lazily after precollect since the buffer is empty at __init__.
        # Slots are items here, so the layout uses
        # ``spawn_index=arange(N), target_index=None``. ``cfg.rate_source``
        # selects between the rolling monitor and the predictor's
        # success_estimator output.
        self._layout: StateLayout | None = None
        self._sampler: Sampler | None = None
        self._sampler_log_counter: int = 0

    def _on_state_buffer_compact(self, keep_indices: torch.Tensor) -> None:
        """Permute parallel arrays in lockstep with a buffer compaction.

        The :class:`StateBuffer` thins itself down to ``target_size``
        when oversample is enabled; everything outside the buffer that
        is indexed by slot position must be permuted by the same
        surviving-index map. The sampler's :class:`StateLayout` is
        derived from the buffer's slot xyz, which moves under
        compaction, so the sampler is invalidated and rebuilt lazily
        on the next sampling step.
        """
        n_keep = int(keep_indices.shape[0])
        max_size = self.monitor_success_rate.shape[0]
        # Permute the running rate scratches; tail goes back to zero so
        # newly-appended slots start with no observed history.
        self.monitor_success_rate[:n_keep] = self.monitor_success_rate[keep_indices]
        self.monitor_success_rate[n_keep:max_size] = 0.0
        self.success_rate[:n_keep] = self.success_rate[keep_indices]
        self.success_rate[n_keep:max_size] = 0.0
        # SuccessMonitor's per-slot rolling history aligns with slot
        # position too; carry the kept slots' history along.
        sm = self.success_monitor
        sm.success_buf[:n_keep] = sm.success_buf[keep_indices]
        sm.success_buf[n_keep:max_size] = False
        sm.success_pointer[:n_keep] = sm.success_pointer[keep_indices]
        sm.success_pointer[n_keep:max_size] = 0
        sm.success_size[:n_keep] = sm.success_size[keep_indices]
        sm.success_size[n_keep:max_size] = 0
        sm.success_count[:n_keep] = sm.success_count[keep_indices]
        sm.success_count[n_keep:max_size] = 0
        # Drop the sampler so the next sampling pass rebuilds the
        # StateLayout against the post-compact slot xyz; the frontier strategy
        # reuses its kNN graph internally and needs the fresh coords.
        self._layout = None
        self._sampler = None

    # ------------------------------------------------------------------
    # Buffer accumulation
    # ------------------------------------------------------------------

    def _accumulate(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, reset_term: EventTermCfg):
        """Run a single reset attempt and store valid states in the buffer."""
        if not self._tag_names_resolved:
            buf_cfg = self._state_buffer_cfg
            if buf_cfg.tag_names_bind is not None:
                self.state_buffer.set_tag_names(eval(buf_cfg.tag_names_bind))  # noqa: S307
            self._tag_names_resolved = True

        reset_term.func(env, env_ids, **reset_term.params)
        valid_mask = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
        for _, val in self.acceptance_conditions.items():
            valid_mask &= val(env, env_ids)

        valid_env_ids = env_ids[valid_mask]
        if valid_env_ids.numel() > 0:
            states = reset_state.get_reset_state(self._env, valid_env_ids, self.reset_assets, is_relative=True)
            if self._tag_indices_bind is not None:
                all_tags = eval(self._tag_indices_bind)  # noqa: S307
                self.state_buffer.add_with_tags(states, all_tags[env_ids][valid_mask])
            else:
                self.state_buffer.add(states)
            self._layout = None
            self._sampler = None

        return env_ids[~valid_mask]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        reset_term: EventTermCfg,
        reset_assets: list[str],
        acceptance_conditions: dict,
        state_buffer_cfg: StateBufferCfg,
        success_monitor_cfg: SuccessMonitorCfg,
        sampling: SamplerCfg,
        keep_accumulating: bool = False,
        report: bool = False,
        monitor_exclude_terms: list[str] | tuple[str, ...] = (),
        wandb_3d_asset: str | None = None,
        wandb_3d_relative_to: str | None = None,
        wandb_3d_log_period: int = 100,
    ):
        """Args (additional 3D-vis params):

        wandb_3d_asset: Asset name whose xyz columns drive a periodic
            ``wandb.Object3D`` upload of the per-slot success rate, sampling
            probability, and Δ-success. ``None`` (default) skips upload.
            The asset must be one of :paramref:`reset_assets`.
        wandb_3d_relative_to: If set, plot the position offset of
            :paramref:`wandb_3d_asset` from this reference asset, expressed
            **in the reference asset's body frame** (i.e., the position
            difference is rotated by the reference's inverse orientation).
            For factory: pass ``"fixed_asset"`` here so origin = perfectly
            assembled and axes are aligned with the goal pose, regardless
            of how the fixed asset is oriented per buffer slot. ``None``
            plots env-relative xyz directly.
        wandb_3d_log_period: Number of ``__call__`` invocations between
            wandb pushes. Ignored when :paramref:`wandb_3d_asset` is
            ``None``.
        """
        # 1. Pre-collect until buffer is full
        if reset_assets and list(reset_assets) != self._requested_reset_assets:
            raise ValueError(
                "reset_accumulator reset_assets changed after initialization. "
                f"Expected {self._requested_reset_assets}, got {list(reset_assets)}."
            )
        if self.precollecting_phase:
            all_env_ids = torch.arange(env.num_envs, device=env.device)
            pbar = tqdm(total=self.state_buffer.max_size, desc="reset_accumulator")
            while not self.state_buffer.is_full:
                prev = len(self.state_buffer)
                self._accumulate(env, all_env_ids, reset_term)
                pbar.update(len(self.state_buffer) - prev)
            pbar.close()
            self.precollecting_phase = False

        # 2. Update success monitor with episode outcomes, excluding specified terms
        progress = env.termination_manager.get_term_cfg("progress_context").func
        monitor_ids = env_ids
        if env_ids.numel() > 0 and monitor_exclude_terms:
            exclude_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for term_name in monitor_exclude_terms:
                if term_name in env.termination_manager._term_names:
                    exclude_mask |= env.termination_manager._last_episode_dones[
                        :, env.termination_manager._term_name_to_term_idx[term_name]
                    ]
            monitor_ids = env_ids[~exclude_mask[env_ids]]
        if monitor_ids.numel() > 0:
            self.success_monitor.success_update(self.sampled_slots[monitor_ids], progress.is_success[monitor_ids])

        # Sync the unified success_rate from the active source
        if self._sampling_cfg.rate_source == "estimator":
            if self.state_buffer.success_rates is None:
                raise RuntimeError(
                    "reset_accumulator sampling rate_source='estimator' requires "
                    "state_buffer.success_rates to be populated."
                )
            self.success_rate[:] = self.state_buffer.success_rates
        else:
            self.success_rate[:] = self.monitor_success_rate

        log: dict[str, float] = {}
        if report:
            n_slots = len(self.state_buffer)
            if self.state_buffer.tag_names:
                tags = self.state_buffer.tags[:n_slots]
                names = self.state_buffer.tag_names
                monitor_rates = self.monitor_success_rate[:n_slots]
                monitor_means = _tagged_report(monitor_rates, tags, names, reduction="mean")
                log["Metrics/MonitorSuccessRate"] = monitor_rates.mean().item()
                for name in names:
                    log[f"Metrics/MonitorSuccessRate/{name}"] = monitor_means[name]
                if self.state_buffer.success_rates is not None:
                    estimator_rates = self.state_buffer.success_rates[:n_slots]
                    estimator_means = _tagged_report(estimator_rates, tags, names, reduction="mean")
                    log["Metrics/EstimatorSuccessRate"] = estimator_rates.mean().item()
                    for name in names:
                        log[f"Metrics/EstimatorSuccessRate/{name}"] = estimator_means[name]

        # 3. Optionally accumulate more states
        if keep_accumulating:
            env_ids = self._accumulate(env, env_ids, reset_term)

        # 4. Sample a slot and apply the state. The sampler subsumes
        # Beta / Frontier / Uniform via its strategy composition; one
        # ``probabilities()`` call replaces the legacy 3-branch isinstance.
        probs: torch.Tensor | None = None
        sample_rates: torch.Tensor | None = None
        if env_ids.numel() > 0:
            self._ensure_sampler(wandb_3d_asset, wandb_3d_relative_to)
            assert self._sampler is not None
            sample_rates = self.success_rate[: len(self.state_buffer)]
            num_samples = self._sampling_cfg.max_samples if self._sampling_cfg.warp else len(env_ids)
            assert num_samples is not None
            probs, slot_idx = self._sampler.probabilities_and_sample(sample_rates, int(num_samples))
            slot_idx = slot_idx[: len(env_ids)]
            self.sampled_slots[env_ids] = slot_idx.to(self.sampled_slots.dtype)
            reset_state.set_reset_state(
                self._env,
                self.state_buffer.sample(slot_idx),
                env_ids,
                self.reset_assets,
                is_relative=True,
            )
            if report and self.state_buffer.tag_names:
                tags = self.state_buffer.tags[: len(self.state_buffer)]
                sample_probs = _tagged_report(probs, tags, self.state_buffer.tag_names, reduction="sum")
                for name in self.state_buffer.tag_names:
                    log[f"Metrics/SampleProb/{name}"] = sample_probs[name]

        # 5. Log metrics
        if report:
            if "log" not in env.extras:
                env.extras["log"] = {}
            env.extras["log"].update(log)  # type: ignore

        # 5b. Sampler diagnostic: per-strategy aggregate stats every 50
        # sample steps, plus the bucketed-by-frontier breakdown when a
        # frontier strategy is active. Helper handles non-frontier
        # samplers gracefully so this is unconditional.
        if probs is not None and self._sampler is not None:
            self._sampler_log_counter += 1
            if self._sampler_log_counter % 50 == 0:
                assert sample_rates is not None
                log_sampler_bins(
                    self._sampler,
                    success_rates=sample_rates,
                    probs=probs,
                    log_dict=self._env.extras.setdefault("log", {}),
                )

        # 6. Periodic 3D wandb scatter (opt-in via ``wandb_3d_asset``).
        if wandb_3d_asset is not None and not self.precollecting_phase:
            self._wandb_3d_counter = getattr(self, "_wandb_3d_counter", 0) + 1
            if self._wandb_3d_counter % wandb_3d_log_period == 0:
                self._log_wandb_3d_scatter(wandb_3d_asset, wandb_3d_relative_to)

    # ------------------------------------------------------------------
    # 3D wandb visualization
    # ------------------------------------------------------------------

    def _xyz_offset_for_asset(self, asset_name: str) -> int | None:
        """Return the column offset of an asset's root xyz, or ``None`` if absent.

        State buffer rows are concatenated per-asset slices; the first three
        columns of every asset slice are the asset's world (or env-relative)
        ``(x, y, z)``.
        """
        offset = 0
        reset_asset_set = set(self.reset_assets)
        for name, articulation in self._env.scene._articulations.items():
            if name not in reset_asset_set:
                continue
            if name == asset_name:
                return offset
            offset += 13 + 2 * articulation.num_joints
        for name in self._env.scene._rigid_objects:
            if name not in reset_asset_set:
                continue
            if name == asset_name:
                return offset
            offset += 13
        return None

    def _slot_xyz_tensor(self, asset_name: str | None, relative_to: str | None) -> torch.Tensor | None:
        """Per-slot ``[n_slots, 3]`` xyz of ``asset_name``, optionally in ``relative_to``'s body frame.

        Single source of truth for the slot spatial domain shared by the
        wandb 3D scatter and the frontier kNN graph. Returns ``None`` when
        the asset (or reference) is not in the buffer or the buffer is
        empty -- callers should treat this as "not yet ready".

        - ``relative_to=None``: env-relative ``asset.xyz`` straight from
          the buffer.
        - ``relative_to`` set: ``quat_apply_inverse(ref.quat, asset.xyz −
          ref.xyz)`` — the offset rotated into the reference's body frame
          so axes align with the goal pose regardless of reference
          orientation.
        """
        if asset_name is None:
            return None
        offset = self._xyz_offset_for_asset(asset_name)
        if offset is None:
            return None
        ref_offset: int | None = None
        if relative_to is not None:
            ref_offset = self._xyz_offset_for_asset(relative_to)
            if ref_offset is None:
                return None
        n = len(self.state_buffer)
        if n == 0:
            return None
        xyz = self.state_buffer.data[:n, offset : offset + 3]
        if ref_offset is not None:
            from isaaclab.utils.math import quat_apply_inverse

            ref_xyz = self.state_buffer.data[:n, ref_offset : ref_offset + 3]
            # Asset slice layout: [pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3), ...].
            ref_quat = self.state_buffer.data[:n, ref_offset + 3 : ref_offset + 7]
            xyz = quat_apply_inverse(ref_quat, xyz - ref_xyz)
        return xyz

    def _ensure_sampler(self, wandb_3d_asset: str | None, wandb_3d_relative_to: str | None) -> None:
        """Build the :class:`StateLayout` and sampler once after precollect.

        Slots are items here (1-to-1), so the layout uses
        ``spawn_index = arange(N), target_index = None`` over the
        post-precollect slot count. Coords come from
        :meth:`_slot_xyz_tensor` -- the same ``(asset, relative_to)``
        the wandb 3D scatter uses, so the spatial domain a frontier
        strategy reads is the one the user can see in wandb.

        When ``wandb_3d_asset`` is ``None`` (no spatial domain
        configured), Frontier sampling is rejected with a clear error;
        Beta / Uniform sampling proceeds with a placeholder coords
        tensor (unused by those strategies).
        """
        if self._sampler is not None:
            return
        coords = self._slot_xyz_tensor(wandb_3d_asset, wandb_3d_relative_to)
        if coords is None:
            # Frontier strategies need real coords; non-spatial strategies
            # (Beta / Uniform) don't, so a placeholder works for them.
            needs_spatial = any(
                isinstance(strategy_cfg, FrontierSamplingStrategyCfg) for strategy_cfg in self._sampling_cfg.strategies
            )
            if needs_spatial:
                raise ValueError(
                    "FrontierSamplingStrategyCfg requires a per-slot spatial domain; set "
                    "``wandb_3d_asset`` (and optionally ``wandb_3d_relative_to``) on the "
                    "reset_accumulator term so the buffer's slot xyz can be extracted."
                )
            n = len(self.state_buffer)
            coords = torch.zeros(n, 1, device=self._env.device)
        n = int(coords.shape[0])
        self._layout = StateLayout(
            coords=coords,
            spawn_index=torch.arange(n, device=coords.device, dtype=torch.long),
            target_index=None,
        )
        self._sampler = self._sampling_cfg.class_type(self._sampling_cfg, self._layout)

    def _log_wandb_3d_scatter(self, asset_name: str, relative_to: str | None) -> None:
        """Push per-slot success / sampling / Δ-success as ``wandb.Object3D``.

        Builds a :class:`ScatterDashboard3D` once on first call using
        :meth:`_slot_xyz_tensor` for slot positions:

        - ``asset.xyz`` (env-relative) when ``relative_to`` is ``None``.
        - ``quat_apply_inverse(ref.quat, asset.xyz − ref.xyz)`` when
          ``relative_to`` is set — i.e., the offset rotated into the
          reference asset's body frame so the cloud's axes are aligned
          with the goal pose regardless of the reference's orientation.

        Each call pushes the per-slot panels (success rate, sampling
        prob, Δ-success, plus state frontier when frontier sampling is
        active) as separate Object3D logs so they appear as independent
        wandb panels with their own history slider. No-ops cleanly if
        wandb isn't initialized, either asset isn't in the buffer, or
        the buffer is empty.
        """
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        # Lazy build: dashboard geometry, prev-rate cache, sampling-prob buffer.
        if not hasattr(self, "_wandb_3d_dashboard"):
            xyz = self._slot_xyz_tensor(asset_name, relative_to)
            if xyz is None:
                # Asset (or ref) absent; remember and never retry.
                if self._xyz_offset_for_asset(asset_name) is None or (
                    relative_to is not None and self._xyz_offset_for_asset(relative_to) is None
                ):
                    self._wandb_3d_dashboard = None
                return  # buffer empty: retry next gate
            from isaaclab_tasks.manager_based.multi_task.viz import ScatterDashboard3D

            n = xyz.shape[0]
            self._wandb_3d_dashboard = ScatterDashboard3D(positions=xyz.detach().cpu().numpy())
            self._wandb_3d_n = n
            self._wandb_3d_prev_rates = torch.zeros(n, device=self.success_rate.device)
        if self._wandb_3d_dashboard is None:
            return

        from isaaclab_tasks.manager_based.multi_task.viz import PanelSpec

        n = self._wandb_3d_n
        rates_t = self.success_rate[:n]
        delta_t = rates_t - self._wandb_3d_prev_rates
        self._wandb_3d_prev_rates.copy_(rates_t)

        # Sampling probability mirrors the runtime sampler for sample-mass
        # parity with what the policy actually sees: one ``probabilities()``
        # call drives both the runtime sampler and this panel.
        assert self._sampler is not None
        probs_t = self._sampler.probabilities(rates_t)

        rates = rates_t.detach().cpu().numpy()
        delta = delta_t.detach().cpu().numpy()
        probs = probs_t.detach().cpu().numpy()

        prob_max = max(float(probs.max()), 1e-9)
        delta_range = max(float(abs(delta).max()), 0.05)

        panels = {
            "success_rate_3d": PanelSpec(values=rates, cmap="RdYlGn", vmin=0.0, vmax=1.0, title="success_rate"),
            "sampling_prob_3d": PanelSpec(values=probs, cmap="viridis", vmin=0.0, vmax=prob_max, title="sampling_prob"),
            "delta_success_3d": PanelSpec(
                values=delta, cmap="RdYlGn", vmin=-delta_range, vmax=delta_range, title="delta_success"
            ),
        }
        # Per-strategy score panels: each active non-constant strategy in the
        # sampler gets a heatmap of its raw per-slot score, so the
        # spatial contribution of each strategy is visible separately.
        # Constant strategies (e.g. uniform) carry no info and are
        # skipped.
        score_rows_t = self._sampler.scores(rates_t)
        for i, name in enumerate(self._sampler.names):
            score_t = score_rows_t[i]
            if float(score_t.std()) < 1e-9:
                continue
            score_np = score_t.detach().cpu().numpy()
            panels[f"{name}_score_3d"] = PanelSpec(
                values=score_np,
                cmap="viridis",
                vmin=0.0,
                vmax=max(float(score_np.max()), 1e-9),
                title=f"{name}_score",
            )
        # Note: under the per-task frontier redesign, the strategy's
        # ``score(rates)`` *is* the per-task frontier value -- there is
        # no separate raw spatial signal to surface as a bonus panel,
        # so ``frontier_score_3d`` already shows what was previously
        # split between ``frontier_score`` (above-mean) and
        # ``state_frontier`` (raw).
        log_payload = {
            f"Sampler/{tag}": wandb.Object3D(self._wandb_3d_dashboard.to_object3d(panel))
            for tag, panel in panels.items()
        }
        wandb.log(log_payload)


class TermChoice(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.term_partitions: dict[str, EventTermCfg] = cfg.params["terms"]  # type: ignore
        self.num_partitions = len(self.term_partitions)
        self.term_samples = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
        self.term_success_rate = torch.zeros(self.num_partitions, device=env.device)
        self._sampling_cfg: SamplerCfg = cfg.params["sampling"]
        if self._sampling_cfg.max_samples is None:
            self._sampling_cfg.max_samples = env.num_envs

        # TermChoice items are partition keys, not spatial states; build
        # a placeholder layout (Beta + Uniform strategies ignore coords).
        self._layout = StateLayout(
            coords=torch.zeros(self.num_partitions, 1, device=env.device),
            spawn_index=torch.arange(self.num_partitions, device=env.device, dtype=torch.long),
            target_index=None,
        )
        self._sampler = self._sampling_cfg.class_type(self._sampling_cfg, self._layout)

        # Need a monitor when the sampler has any non-uniform strategy,
        # or when the user explicitly requested reporting.
        needs_rates = any(name != "uniform" for name in self._sampler.names)
        if cfg.params.get("report", False) or needs_rates:
            monitor_cfg: SuccessMonitorCfg = cfg.params["success_monitor_cfg"]
            monitor_cfg.num_monitored_data = self.num_partitions
            monitor_cfg.device = env.device
            monitor_cfg.max_updates = env.num_envs if monitor_cfg.max_updates is None else monitor_cfg.max_updates
            self.success_monitor = monitor_cfg.class_type(monitor_cfg, self.term_success_rate)
        else:
            self.success_monitor = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        terms: dict[str, ManagerTermBase],
        sampling: SamplerCfg,
        success_monitor_cfg: SuccessMonitorCfg,
        report: bool = False,
    ) -> None:
        if self.num_partitions == 0:
            return
        if report:
            log = {
                f"Metrics/SuccessRate/{name}": self.term_success_rate[i].item()
                for i, name in enumerate(self.term_partitions.keys())
            }
            log["Metrics/SuccessRate"] = self.term_success_rate.mean().item()
        if self.success_monitor:
            success = env.termination_manager.get_term_cfg("progress_context").func.is_success
            self.success_monitor.success_update(self.term_samples[env_ids], success[env_ids])

        num_samples = self._sampling_cfg.max_samples if self._sampling_cfg.warp else len(env_ids)
        assert num_samples is not None
        probs, choices = self._sampler.probabilities_and_sample(self.term_success_rate, int(num_samples))
        self.term_samples[env_ids] = choices[: len(env_ids)]
        if report:
            log.update(
                {f"Metrics/SampleProb/{name}": probs[i].item() for i, name in enumerate(self.term_partitions.keys())}
            )

        for i, (_, term_cfg) in enumerate(self.term_partitions.items()):
            term_ids = env_ids[self.term_samples[env_ids] == i]
            if term_ids.numel() > 0:
                term_cfg.func(env, term_ids, **term_cfg.params)

        if report:
            if "log" not in env.extras:
                env.extras["log"] = {}
            env.extras["log"].update(log)  # type: ignore


class ChainedResetTerms(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.terms: dict[str, EventTermCfg] = cfg.params["terms"]  # type: ignore

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        terms: dict[str, callable],
        probability: float = 1.0,
    ) -> None:
        keep = torch.rand(env_ids.size(0), device=env_ids.device) < probability
        if not keep.any():
            return
        env_ids_to_reset = env_ids[keep]
        for _, term in terms.items():
            term.func(env, env_ids_to_reset, **term.params)  # type: ignore
