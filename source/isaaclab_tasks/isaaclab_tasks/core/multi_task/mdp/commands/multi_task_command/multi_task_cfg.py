# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

# Keep the command implementation import TYPE_CHECKING-only so this cfg can be
# constructed (e.g. during hydra's pre-SimulationApp cfg load) without pulling
# in ``isaaclab.envs.ManagerBasedRLEnv`` → ``isaaclab.sim.SimulationContext``
# → ``pxr``. ``class_type`` is stored as a fully-qualified string and resolved
# lazily by :class:`~isaaclab.utils.string.ResolvableString` when the command
# manager calls ``term_cfg.class_type(...)`` after Kit has launched. The test
# ``source/isaaclab_tasks/test/test_env_cfg_no_forbidden_imports.py`` gates
# this contract.
if TYPE_CHECKING:
    from .multi_task_command import MultiTaskCommand


@configclass
class MinMaxSampler:
    kernel: int = MISSING
    minimum: list[float] = MISSING
    maximum: list[float] = MISSING
    out_dim: int | None = None
    """Override for the kernel's output dimension.

    Some kernels emit a tensor whose last dim is different from the ``len(minimum)``
    param count — e.g. :data:`SAMPLER_KERNEL_ID.EULER_UNIFORM_TO_QUAT` takes 3 Euler
    pairs and emits a 4-vector quaternion. Set ``out_dim`` to the real output size so
    the command term's ``target_dim_max`` (derived from ``len(get_kernel_input()) // 2``)
    is ≥ the state kernel's output dim.

    When ``out_dim > len(minimum)``, the encoded param tensor is zero-padded so its
    length is ``2 * out_dim``. Padded pairs carry no information — the kernel simply
    ignores them. Leave this ``None`` for kernels whose output dim equals the param
    count (e.g. :data:`SAMPLER_KERNEL_ID.UNIFORM`).
    """

    def get_kernel_input(self, device="cpu") -> torch.Tensor:
        """Return sampler params as a flat 1D tensor.

        Encoding (interleaved pairs): ``[min0, range0, min1, range1, ...]``. If
        :attr:`out_dim` exceeds ``len(minimum)``, trailing zero pairs are appended.
        :class:`MultiTaskCommand._build_spec` additionally zero-pads rows to the
        maximum ``P`` across subtasks.
        """
        mn = torch.tensor(self.minimum, device=device, dtype=torch.float32)
        mx = torch.tensor(self.maximum, device=device, dtype=torch.float32)
        rg = mx - mn

        n = mn.numel()
        target_n = max(n, self.out_dim or 0)
        if target_n > n:
            pad = torch.zeros(target_n - n, device=device, dtype=torch.float32)
            mn = torch.cat([mn, pad])
            rg = torch.cat([rg, pad])

        out = torch.empty(mn.numel() * 2, device=device, dtype=torch.float32)
        out[0::2] = mn
        out[1::2] = rg
        return out


@configclass
class MultiTaskCfg(CommandTermCfg):
    @configclass
    class BaseTaskCfg:
        asset_cfg: SceneEntityCfg = MISSING
        metric_kernel: int = MISSING
        state_kernel: int = MISSING
        sampler: MinMaxSampler = MISSING
        activation_kernel: int = MISSING
        activation_kernel_param: float = MISSING

    @configclass
    class TrackingTaskCfg(BaseTaskCfg):
        """An ongoing-condition subtask — contributes one factor to ``G``'s
        multiplicative quality term, never gates success.

        After the unified-quality refactor, every "quality dimension" — whether
        a tracking goal the policy is asked to maintain (e.g. body velocity)
        or a soft-safety constraint the policy is asked to respect (e.g. no
        chassis contact, low mechanical power) — flows through the same
        composer path. The per-step activation accumulates, the transit-mean
        is taken, and all per-subtask means multiply together (raised to
        ``cfg.quality_easing``) into the single ``quality_factor`` that the
        composer emits at terminal step:

            G = instant_gate · ( ∏_k mean_t A_k(t) ) ^ quality_easing

        The only difference between a "tracking goal" and a "safety constraint"
        is whether the subtask's delta channel appears in the policy obs:

        - **Tracking goal** (``expose_in_obs=True``, default): the policy sees
          the live ``target − current`` delta in ``goal_track_delta`` and the
          per-channel active bit in ``goal_active``, so it can act on the
          condition.
        - **Safety constraint** (``expose_in_obs=False``): the subtask still
          contributes to ``G``, but its delta never reaches the policy obs.
          The policy learns to satisfy the constraint *implicitly* through
          the reward gradient — the way a human learns "don't touch the hot
          stove" without staring at a thermometer.

        Authoring contract for safety-style subtasks: set ``expose_in_obs=False``,
        target the sampler at the "no violation" point (typically
        ``minimum = maximum = [0.0]``), pair with TANH activation where
        ``activation_kernel_param`` = the violation scale ``σ``.

        Composer properties (see :func:`~.reward_composer.multiplicative_terminal_reward`):

        - ``G ≥ 0`` always — quality factors are bounded in ``[0, 1]``,
          ``instant_gate ∈ {0, 1}``, ``quality_easing ∈ (0, 1]``.
        - Success is always strictly preferred over no-success: ``quality > 0``
          ⇒ a successful env with terrible quality still scores better than a
          failing env with perfect quality.
        - Bootstrap immunity at reach-truncate: ``gate = 0`` zeroes the terminal
          regardless of quality; rsl_rl's ``γ·V(s_T)`` propagates clean
          future-task-value, never future-quality contamination.
        """

        expose_in_obs: bool = True
        """Whether the subtask's delta + active-mask channels appear in the
        policy obs (``goal_track_delta`` / ``goal_active``).

        Defaults to ``True`` for ordinary tracking goals where the policy
        needs the live delta to act. Set ``False`` for soft-safety subtasks
        whose internal violation amount the policy should *not* observe
        directly — the agent should learn to satisfy them implicitly via
        the reward gradient on ``G``."""

    @configclass
    class InstantaneousTaskCfg(BaseTaskCfg):
        """A "must-hit" milestone — gates ``G`` via latched achievement.

        Each instant subtask carries a one-way latch (``instant_achieved``)
        that flips True the first step ``activation > 0.5`` and stays True
        until episode reset. The composer's ``instant_gate`` is the AND of
        every instant subtask's latch (vacuously True when there are none).

        ``G = instant_gate · quality_factor``, so an instant subtask is a
        *prerequisite* for any positive reward: gate=0 zeroes ``G`` at the
        terminal step, regardless of how good the quality factor was. The
        policy must hit every instant milestone at least once during the
        episode, but the moment of achievement need not coincide with the
        terminal step (the latch carries the credit forward).

        Tasks containing at least one instant subtask use the
        ``time_out_reach_truncate`` DoneTerm path (``time_out=True``,
        rsl_rl bootstraps); pure-tracking tasks use ``time_out_track_terminate``
        (``time_out=False``, terminal value is the complete return).

        Authoring contract: pair with ``activation_kernel=LESS`` on a tight
        threshold so the latch fires only on genuine achievement (e.g.
        ``LESS(error, 0.2)`` for a position reach within 20 cm).
        """

        pass

    # Lazy string reference; resolved to ``MultiTaskCommand`` after Kit launches.
    class_type: type[MultiTaskCommand] | str = (
        "isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.multi_task_command:MultiTaskCommand"
    )
    tasks: dict[str, list[BaseTaskCfg]] = MISSING

    quality_easing: float = 1.0
    """Easing exponent on the multiplicative quality term in the composer's ``G``.

    The composer's terminal reward is::

        G = reached · ( ∏_{k ∈ tracking ∪ safety} mean_t A_k(t) ) ^ quality_easing

    where each ``A_k(t)`` is a per-step activation in ``[0, 1]`` (tracking
    error → ``1 − tanh(err/σ)`` for tracking, violation → ``1 − tanh(viol/σ)``
    for safety). Without easing, raw product compounding makes ``G`` collapse
    when several quality dimensions are imperfect — e.g. K=3 dims each at 0.5
    yields 0.125, weak gradient. Easing = 0.5 turns that into 0.354.

    Range:
      - ``1.0`` = no easing (pure product compounding).
      - ``0.5`` = sqrt softening (default; matches the geometric mean at K=2).
      - smaller values = even softer; ``0.0`` is degenerate (always 1, no
        quality signal at all).

    Applied uniformly to every tracking subtask's transit-mean activation —
    "tracking" being the unified quality kind that includes both ordinary
    tracking goals (``expose_in_obs=True``, default) and soft-safety
    constraints (``expose_in_obs=False``). The composer reduces them
    identically; ``expose_in_obs`` only controls whether the subtask's delta
    appears in the policy obs.
    """

    dispatch_backend: str = "torch"
    """Command dispatch backend.

    ``"torch"`` selects the PyTorch reference path. ``"mega_kernel"``
    selects the current Warp backend whose private execution plan is shaped as
    ``(env, slot)``. ``"schedule_ordered_mega"`` keeps the same dense output
    layout but executes each env's slots in fused-schedule order. ``"packed_scatter"``
    selects the Warp backend with a fused-schedule-sorted flat queue and legacy
    output scatter. ``"primitive_queue_local"`` groups by primitive schedule,
    writes local output rows, and composes reward from those local rows.
    ``"primitive_graph_local"`` adds explicit primitive-graph producer nodes
    for reusable current-state, reduction, and contact-predicate work before
    target-specific consumers. New dispatch experiments must become selectable
    through this field before their numbers count as command-level benchmark
    results.
    """

    tracking_episode_length_min_seconds: float | None = None
    """Randomize pure-tracking episode lengths uniformly at each reset.

    ``None`` (default): disabled; pure-tracking envs run the full
    ``env.max_episode_length_s``.

    When set to a float ``L_min``: on every resample of a pure-tracking env,
    draw a fresh episode length uniformly from ``[L_min, max_episode_length_s]``
    (in seconds, rounded to whole env steps). Reach / mixed envs always use
    ``env.max_episode_length``.

    Intuition: short samples let the agent earn near-full ``G = transit_mean``
    on a few steps of decent tracking — strong gradient signal even for a
    random-init policy. Long samples probe sustained tracking. Across the
    batch at every step there's a distribution of remaining-time-to-terminate,
    so the agent always has *some* envs where terminal reward is close by.
    As the policy improves, the long-episode envs contribute more until the
    distribution effectively collapses onto ``max_episode_length_s``.

    Simpler than a per-step error-gated curriculum and has no carried state —
    per-env episode length is reset to a fresh random draw every episode.
    """

    # Note on preset resolution: ``tasks`` may be assigned either a literal
    # ``dict[str, list[BaseTaskCfg]]`` or a :class:`~isaaclab_tasks.utils.PresetCfg`
    # instance (e.g. :class:`MultiTaskTasksPresetCfg`). We deliberately do NOT
    # resolve the preset in ``__post_init__`` — doing so would collapse the
    # preset to its ``default`` value before hydra's ``register_task``
    # ``collect_presets`` walk happens, making CLI overrides like
    # ``presets=velocity`` invisible to the resolver. Hydra resolves the
    # preset into a dict before the env is built; for direct (non-hydra)
    # instantiation, :meth:`MultiTaskCommand.__init__` calls
    # :func:`resolve_presets` itself with an empty selection set, which
    # picks the ``default`` alternative.
