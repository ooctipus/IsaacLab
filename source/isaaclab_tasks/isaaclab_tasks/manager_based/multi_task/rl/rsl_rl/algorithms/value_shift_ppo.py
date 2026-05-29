# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-state value-shift PPO variant for prioritized curriculum sampling.

This module hosts :class:`ValueShiftPPO`, a thin PPO subclass that evaluates
its critic on a fixed observation cache after every ``update()`` and writes
``|V_new - V_prev|`` per cached state. A curriculum sampling strategy (e.g.
:class:`ValueShiftSamplingStrategy`) reads the per-state diff buffer and uses
it as its sampling signal.

Living at the task level keeps ``dep/rsl_rl`` untouched; the runner cfg sets
``algorithm.class_name`` to the fully-qualified module path of this class so
that :func:`rsl_rl.utils.resolve_callable` resolves it correctly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from rsl_rl.algorithms.ppo import PPO

if TYPE_CHECKING:
    from rsl_rl.env import VecEnv
    from tensordict import TensorDict


class ValueShiftPPO(PPO):
    """PPO + per-state critic value-shift signal.

    Three buffers are bound by :meth:`construct_algorithm` via ``eval`` over the
    runner's ``bind_*_exp`` fields:

    * ``_obs_cache``: :class:`tensordict.TensorDict` of cached critic-group
      observations, one row per discretized command state.
    * ``_cur_buf``: ``[num_states]`` float tensor; current iteration's critic
      value per state.
    * ``_diff_buf``: ``[num_states]`` float tensor; ``|V_new - V_prev|`` per
      state, consumed by the value-shift sampling strategy.

    The first :meth:`update` after construction sets ``diff = |V_new|`` because
    ``cur_buf`` is initialised to zero; subsequent calls compute the true
    ``|V_new - V_prev|`` magnitude.
    """

    _obs_cache: TensorDict | None = None
    _cur_buf: torch.Tensor | None = None
    _diff_buf: torch.Tensor | None = None

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        if self._obs_cache is not None:
            assert self._cur_buf is not None and self._diff_buf is not None, (
                "ValueShiftPPO bind state inconsistent: _obs_cache is set but _cur_buf / _diff_buf are not."
            )
            with torch.inference_mode():
                v_new = self.critic(self._obs_cache).squeeze(-1)
            self._diff_buf.copy_((v_new - self._cur_buf).abs())
            self._cur_buf.copy_(v_new)
        return loss_dict

    @staticmethod
    def construct_algorithm(obs, env: VecEnv, cfg: dict, device: str) -> ValueShiftPPO:
        """Build the ValueShiftPPO algorithm and wire its three bind buffers.

        Bind expressions live on the ``algorithm`` sub-cfg (kept inside the
        ``value_shift`` preset variant of :class:`PositionAlgorithmPresetCfg`)
        and are popped here before delegating to
        :meth:`PPO.construct_algorithm` -- otherwise PPO would unpack them as
        kwargs to ``ValueShiftPPO.__init__`` and fail. Each expression is
        ``eval``-ed against a namespace exposing ``env``, ``alg``, and
        ``setattr``, e.g.
        ``"setattr(alg, '_obs_cache', env.curriculum_manager...)"``.
        """
        algorithm_cfg = cfg["algorithm"]
        bind_observation_exp = algorithm_cfg.pop("bind_observation_exp", None)
        bind_current_value_exp = algorithm_cfg.pop("bind_current_value_exp", None)
        bind_value_diff_exp = algorithm_cfg.pop("bind_value_diff_exp", None)

        # Delegate construction to PPO. ``algorithm_cfg["class_name"]`` still
        # points at ValueShiftPPO's FQN, so PPO.construct_algorithm pops it,
        # resolves it via :func:`resolve_callable`, and instantiates this class.
        alg: ValueShiftPPO = PPO.construct_algorithm(obs, env, cfg, device)  # type: ignore[assignment]
        assert isinstance(alg, ValueShiftPPO), (
            f"ValueShiftPPO.construct_algorithm expected a ValueShiftPPO instance; got {type(alg).__name__}."
            " Check that ``algorithm.class_name`` resolves to ValueShiftPPO."
        )

        bind_ns = {"env": env, "alg": alg, "setattr": setattr}
        for expr in (bind_observation_exp, bind_current_value_exp, bind_value_diff_exp):
            if expr is not None:
                eval(expr, bind_ns)  # noqa: S307

        if alg._obs_cache is not None:
            assert alg._cur_buf is not None, "ValueShiftPPO bind_current_value_exp must set ``alg._cur_buf``."
            assert alg._diff_buf is not None, "ValueShiftPPO bind_value_diff_exp must set ``alg._diff_buf``."
            n = alg._obs_cache.batch_size[0]
            assert tuple(alg._cur_buf.shape) == (n,), (
                f"ValueShiftPPO _cur_buf must have shape ({n},); got {tuple(alg._cur_buf.shape)}."
            )
            assert tuple(alg._diff_buf.shape) == (n,), (
                f"ValueShiftPPO _diff_buf must have shape ({n},); got {tuple(alg._diff_buf.shape)}."
            )

        return alg
