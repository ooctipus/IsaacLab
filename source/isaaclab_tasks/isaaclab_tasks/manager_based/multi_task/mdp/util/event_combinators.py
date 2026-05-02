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
from .sampling import beta_sampling_probs, tagged_report
from .sampling_cfg import BetaSamplingCfg, UniformSamplingCfg
from .state_buffer import StateBuffer
from .state_buffer_cfg import StateBufferCfg
from .success_monitor_cfg import SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
        self.acceptance_conditions = cfg.params.get("acceptance_conditions")
        for key, val in self.acceptance_conditions.items():
            if hasattr(val, "class_type"):
                self.acceptance_conditions[key] = val.class_type(val, env)

        # Filter to scene-resident assets: callers may include "optional" assets
        # (e.g. ``small_gear`` / ``medium_gear`` for the gear_mesh_* presets) that
        # only exist for a subset of presets. This lets a single reset_assets list
        # cover every preset without per-preset branching.
        self._requested_reset_assets = list(cfg.params.get("reset_assets", []))
        present = set(env.scene._articulations) | set(env.scene._rigid_objects)
        self.reset_assets = [a for a in self._requested_reset_assets if a in present]
        self.reset_state_adapters = reset_state.make_reset_state_adapters(env, self.reset_assets)
        state_dim = reset_state.get_reset_state(
            self._env,
            torch.tensor([0], device=env.device),
            self.reset_state_adapters,
        ).shape[-1]
        buf_cfg: StateBufferCfg = cfg.params.get("state_buffer_cfg", StateBufferCfg())
        max_size = buf_cfg.size

        self.state_buffer = StateBuffer(max_size, state_dim, env.device)
        reset_accumulator._shared_buffer = self.state_buffer
        self.sampled_slots = torch.zeros(env.num_envs, device=env.device, dtype=torch.int)
        self.precollecting_phase = True
        self._tag_indices_bind: str | None = buf_cfg.tag_indices_bind
        self._tag_names_resolved = False
        self._sampling_cfg = cfg.params.get("sampling", UniformSamplingCfg())

        self.monitor_success_rate = torch.zeros(max_size, device=env.device)
        monitor_cfg: SuccessMonitorCfg = cfg.params.get(
            "success_monitor_cfg", SuccessMonitorCfg(num_monitored_data=max_size, device=env.device)
        )
        monitor_cfg.num_monitored_data = max_size
        monitor_cfg.device = env.device
        self.success_monitor = monitor_cfg.class_type(monitor_cfg, self.monitor_success_rate)

        self.success_rate = torch.zeros(max_size, device=env.device)
        self._success_rate_source = "monitor"
        if isinstance(self._sampling_cfg, BetaSamplingCfg) and "state_buffer" in self._sampling_cfg.success_rate_bind:
            self._success_rate_source = "success_estimator"

    # ------------------------------------------------------------------
    # Buffer accumulation
    # ------------------------------------------------------------------

    def _accumulate(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, reset_term: EventTermCfg):
        """Run a single reset attempt and store valid states in the buffer."""
        if not self._tag_names_resolved:
            buf_cfg: StateBufferCfg = self.cfg.params.get("state_buffer_cfg", StateBufferCfg())
            if buf_cfg.tag_names_bind is not None:
                self.state_buffer.set_tag_names(eval(buf_cfg.tag_names_bind))  # noqa: S307
            self._tag_names_resolved = True

        reset_term.func(env, env_ids, **reset_term.params)
        valid_mask = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
        for _, val in self.acceptance_conditions.items():
            valid_mask &= val(env, env_ids)

        valid_env_ids = env_ids[valid_mask]
        if valid_env_ids.numel() > 0:
            states = reset_state.get_reset_state(self._env, valid_env_ids, self.reset_state_adapters, is_relative=True)
            if self._tag_indices_bind is not None:
                all_tags = eval(self._tag_indices_bind)  # noqa: S307
                self.state_buffer.add_with_tags(states, all_tags[env_ids][valid_mask])
            else:
                self.state_buffer.add(states)

        return env_ids[~valid_mask]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        reset_term: EventTermCfg,
        reset_assets: list[str] = [],
        acceptance_conditions: dict = {},
        state_buffer_cfg: StateBufferCfg = StateBufferCfg(),
        success_monitor_cfg: SuccessMonitorCfg | None = None,
        sampling: UniformSamplingCfg | BetaSamplingCfg = UniformSamplingCfg(),
        keep_accumulating: bool = False,
        report: bool = False,
        monitor_exclude_terms: list[str] = [],
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
            self.success_monitor.success_update(
                self.sampled_slots[monitor_ids], progress.is_success[monitor_ids].float()
            )

        # Sync the unified success_rate from the active source
        if self._success_rate_source == "success_estimator" and self.state_buffer.success_rates is not None:
            self.success_rate[:] = self.state_buffer.success_rates
        else:
            self.success_rate[:] = self.monitor_success_rate

        if report:
            log: dict[str, float] = {}
            if self.state_buffer.tag_names:
                tags = self.state_buffer.tags[: len(self.state_buffer)]
                names = self.state_buffer.tag_names
                monitor_means = tagged_report(self.monitor_success_rate, tags, names, reduction="mean")
                monitor_probs = beta_sampling_probs(
                    torch.tensor(list(monitor_means.values()), device=env.device), target=0.5, kappa=1
                )
                log["Metrics/MonitorSuccessRate"] = self.monitor_success_rate.mean().item()
                for i, name in enumerate(names):
                    log[f"Metrics/MonitorSuccessRate/{name}"] = monitor_means[name]
                    log[f"Metrics/MonitorSampleProb/{name}"] = monitor_probs[i].item()
                if self.state_buffer.success_rates is not None:
                    estimator_means = tagged_report(self.state_buffer.success_rates, tags, names, reduction="mean")
                    estimator_probs = beta_sampling_probs(
                        torch.tensor(list(estimator_means.values()), device=env.device), target=0.5, kappa=1
                    )
                    log["Metrics/EstimatorSuccessRate"] = self.state_buffer.success_rates.mean().item()
                    for i, name in enumerate(names):
                        log[f"Metrics/EstimatorSuccessRate/{name}"] = estimator_means[name]
                        log[f"Metrics/EstimatorSampleProb/{name}"] = estimator_probs[i].item()

        # 3. Optionally accumulate more states
        if keep_accumulating:
            env_ids = self._accumulate(env, env_ids, reset_term)

        # 4. Sample a slot and apply the state
        if env_ids.numel() > 0:
            if isinstance(self._sampling_cfg, BetaSamplingCfg):
                probs = beta_sampling_probs(
                    self.success_rate,
                    self._sampling_cfg.target,
                    self._sampling_cfg.kappa,
                    self._sampling_cfg.temperature,
                )
                slot_idx = torch.multinomial(probs, len(env_ids), replacement=True).to(torch.int32)
            else:
                slot_idx = torch.randint(0, self.state_buffer.max_size, (len(env_ids),), device=env.device)
            self.sampled_slots[env_ids] = slot_idx.to(self.sampled_slots.dtype)
            reset_state.set_reset_state(
                self._env,
                self.state_buffer.sample(slot_idx),
                env_ids,
                self.reset_state_adapters,
                is_relative=True,
            )

        # 5. Log metrics
        if report:
            if "log" not in env.extras:
                env.extras["log"] = {}
            env.extras["log"].update(log)  # type: ignore

        # 6. Periodic 3D wandb scatter (opt-in via ``wandb_3d_asset``).
        if wandb_3d_asset is not None and not self.precollecting_phase:
            self._wandb_3d_counter = getattr(self, "_wandb_3d_counter", 0) + 1
            if self._wandb_3d_counter % wandb_3d_log_period == 0:
                self._log_wandb_3d_scatter(wandb_3d_asset, wandb_3d_relative_to, sampling)

    # ------------------------------------------------------------------
    # 3D wandb visualization
    # ------------------------------------------------------------------

    def _xyz_offset_for_asset(self, asset_name: str) -> int | None:
        """Return the column offset of an asset's root xyz, or ``None`` if absent.

        State buffer rows are concatenated per-adapter slices; the first three
        columns of every adapter's slice are the asset's world (or env-relative)
        ``(x, y, z)``.
        """
        offset = 0
        for adapter in self.reset_state_adapters:
            if getattr(adapter, "asset_name", None) == asset_name:
                return offset
            offset += adapter.state_dim(self._env)
        return None

    def _log_wandb_3d_scatter(self, asset_name: str, relative_to: str | None, sampling_cfg) -> None:
        """Push per-slot success / sampling / Δ-success as ``wandb.Object3D``.

        Builds a :class:`ScatterDashboard3D` once on first call. Position
        per slot is:

        - ``asset.xyz`` (env-relative) when ``relative_to`` is ``None``.
        - ``quat_apply_inverse(ref.quat, asset.xyz − ref.xyz)`` when
          ``relative_to`` is set — i.e., the offset rotated into the
          reference asset's body frame so the cloud's axes are aligned
          with the goal pose regardless of the reference's orientation.

        Three panels are pushed each call as separate Object3D logs so they
        appear as independent wandb panels with their own history slider.
        No-ops cleanly if wandb isn't initialized, either asset isn't in
        the buffer, or the buffer is empty.
        """
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        # Lazy build: dashboard geometry, prev-rate cache, sampling-prob buffer.
        if not hasattr(self, "_wandb_3d_dashboard"):
            offset = self._xyz_offset_for_asset(asset_name)
            if offset is None:
                self._wandb_3d_dashboard = None
                return
            ref_offset = None
            if relative_to is not None:
                ref_offset = self._xyz_offset_for_asset(relative_to)
                if ref_offset is None:
                    self._wandb_3d_dashboard = None
                    return
            n = len(self.state_buffer)
            if n == 0:
                return  # buffer not populated yet; retry next gate
            from isaaclab.utils.math import quat_apply_inverse

            from isaaclab_tasks.manager_based.multi_task.utils.visualization import ScatterDashboard3D

            xyz = self.state_buffer.data[:n, offset : offset + 3]
            if ref_offset is not None:
                ref_xyz = self.state_buffer.data[:n, ref_offset : ref_offset + 3]
                # Adapter slice layout: [pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3), ...].
                ref_quat = self.state_buffer.data[:n, ref_offset + 3 : ref_offset + 7]
                xyz = quat_apply_inverse(ref_quat, xyz - ref_xyz)
            positions_xyz = xyz.detach().cpu().numpy()
            self._wandb_3d_dashboard = ScatterDashboard3D(positions=positions_xyz)
            self._wandb_3d_n = n
            self._wandb_3d_prev_rates = torch.zeros(n, device=self.success_rate.device)
        if self._wandb_3d_dashboard is None:
            return

        from isaaclab_tasks.manager_based.multi_task.utils.visualization import PanelSpec

        n = self._wandb_3d_n
        rates_t = self.success_rate[:n]
        delta_t = rates_t - self._wandb_3d_prev_rates
        self._wandb_3d_prev_rates.copy_(rates_t)

        # Sampling probability mirrors the runtime sampler for sample-mass
        # parity with what the policy actually sees.
        if isinstance(sampling_cfg, BetaSamplingCfg):
            probs_t = beta_sampling_probs(rates_t, sampling_cfg.target, sampling_cfg.kappa, sampling_cfg.temperature)
        else:
            probs_t = torch.full_like(rates_t, 1.0 / max(n, 1))

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
        log_payload = {
            f"Curriculum/{tag}": wandb.Object3D(self._wandb_3d_dashboard.to_object3d(panel))
            for tag, panel in panels.items()
        }
        wandb.log(log_payload)


class TermChoice(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.term_partitions: dict[str, EventTermCfg] = cfg.params["terms"]  # type: ignore
        self.num_partitions = len(self.term_partitions)
        self.term_samples = torch.zeros((env.num_envs,), dtype=torch.int, device=env.device)
        self.term_success_rate = torch.zeros(self.num_partitions, device=env.device)
        self._sampling_cfg = cfg.params.get("sampling", UniformSamplingCfg())
        needs_monitor = cfg.params.get("report", False) or isinstance(self._sampling_cfg, BetaSamplingCfg)
        if needs_monitor:
            monitor_cfg: SuccessMonitorCfg = cfg.params.get(
                "success_monitor_cfg",
                SuccessMonitorCfg(num_monitored_data=self.num_partitions, device=env.device),
            )  # type: ignore
            monitor_cfg.num_monitored_data = self.num_partitions
            monitor_cfg.device = env.device
            self.success_monitor = monitor_cfg.class_type(monitor_cfg, self.term_success_rate)
        else:
            self.success_monitor = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        terms: dict[str, ManagerTermBase],
        sampling: UniformSamplingCfg | BetaSamplingCfg = UniformSamplingCfg(),
        success_monitor_cfg: SuccessMonitorCfg | None = None,
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
            self.success_monitor.success_update(self.term_samples[env_ids], success[env_ids].float())

        if isinstance(self._sampling_cfg, BetaSamplingCfg):
            rates = eval(self._sampling_cfg.success_rate_bind)  # noqa: S307
            probs = beta_sampling_probs(
                rates, self._sampling_cfg.target, self._sampling_cfg.kappa, self._sampling_cfg.temperature
            )
            self.term_samples[env_ids] = torch.multinomial(probs, len(env_ids), replacement=True).to(torch.int32)
            if report:
                log.update(
                    {
                        f"Metrics/SampleProb/{name}": probs[i].item()
                        for i, name in enumerate(self.term_partitions.keys())
                    }
                )
        else:
            self.term_samples[env_ids] = torch.randint(
                0, self.num_partitions, (env_ids.size(0),), device=env_ids.device, dtype=self.term_samples.dtype
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
