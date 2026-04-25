# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrap a native Brax env so it exposes the same interface as :class:`IsaacLabBraxEnv`.

Used by the rollout-parity test (see :mod:`tests.crl.test_rollout_parity`) to
drive our training loop with scaling-crl's built-in Ant/Humanoid envs. Because
we consume the exact same ``crl_core`` update functions and eager-rollout loop
as the IsaacLab path, matching scaling-crl's native curves with this adapter in
place proves that **our training loop** — not anything IsaacLab-specific — is
faithful to the reference implementation.

Key differences from :class:`IsaacLabBraxEnv`:

- No DLPack bridge: Brax envs are already JAX arrays; no conversion needed.
- No dict observation: Brax envs emit a flat obs vector. We embed the "goal
  slice" by asserting that the env's own obs layout already matches the
  scaling-crl convention ``[state (obs_dim), goal (goal_dim)]``.
- Episode length / auto-reset: handled by ``brax.envs.training.wrap`` just like
  scaling-crl's native ``train.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from isaaclab_brax_adapter import BraxLikeState  # type: ignore[import-not-found]

# Registry of (env_id → (obs_dim, goal_start_idx, goal_end_idx)) mirroring the
# constants in scaling-crl/train.py's ``make_env``. Only envs used by the
# rollout-parity test need to be here; extend as needed.
_BRAX_ENV_REGISTRY: dict[str, tuple[int, int, int]] = {
    "ant": (29, 0, 2),
    "reacher": (10, 4, 7),
    "pusher": (20, 10, 13),
    "humanoid": (268, 0, 3),
}


@dataclass
class NativeBraxTaskSpec:
    """Resolved-size spec for a native Brax env."""

    env_id: str
    obs_dim: int
    goal_dim: int
    goal_start_idx: int
    goal_end_idx: int


def make_native_brax_env(
    env_id: str,
    *,
    episode_length: int,
    num_envs: int,
    seed: int = 0,
):
    """Build a vectorized Brax env wrapped with :class:`NativeBraxEnv`.

    Args:
        env_id: Brax env identifier understood by scaling-crl's ``make_env``.
        episode_length: Max episode length (passed to ``envs.training.wrap``).
        num_envs: Number of parallel envs (split the rng key over this many).
        seed: Top-level seed for env reset.

    Returns:
        ``(adapter, spec)`` — the adapter is the wrapped env with BraxLike
        interface, the spec carries the obs/goal dims the caller needs.
    """
    from brax import envs as brax_envs

    if env_id not in _BRAX_ENV_REGISTRY:
        raise KeyError(
            f"Unknown Brax env_id={env_id!r}. Known: {list(_BRAX_ENV_REGISTRY)}. "
            "Add an entry to _BRAX_ENV_REGISTRY with (obs_dim, goal_start, goal_end) "
            "from scaling-crl/train.py's make_env."
        )

    # Build the raw env. Use scaling-crl's exact constructor args for
    # bit-identical behavior.
    if env_id == "ant":
        from envs.ant import Ant  # type: ignore[import-not-found]

        raw_env = Ant(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
    elif env_id == "reacher":
        from envs.reacher import Reacher  # type: ignore[import-not-found]

        raw_env = Reacher(backend="spring")
    elif env_id == "pusher":
        from envs.pusher import Pusher  # type: ignore[import-not-found]

        raw_env = Pusher(backend="spring")
    elif env_id == "humanoid":
        from envs.humanoid import Humanoid  # type: ignore[import-not-found]

        raw_env = Humanoid(
            backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True
        )
    else:
        raise NotImplementedError(env_id)

    wrapped = brax_envs.training.wrap(raw_env, episode_length=episode_length)

    obs_dim, goal_start, goal_end = _BRAX_ENV_REGISTRY[env_id]
    goal_dim = goal_end - goal_start
    spec = NativeBraxTaskSpec(
        env_id=env_id,
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        goal_start_idx=goal_start,
        goal_end_idx=goal_end,
    )
    adapter = NativeBraxEnv(wrapped, spec, num_envs=num_envs, seed=seed)
    return adapter, spec


class NativeBraxEnv:
    """Thin :class:`IsaacLabBraxEnv`-compatible facade over a wrapped Brax env.

    Unlike the IsaacLab adapter we do NOT cross any framework boundary — the
    env is already pure JAX. The class exists purely so the same training loop
    code (``_eager_rollout`` in ``train.py``) can drive it without branching on
    env type.
    """

    def __init__(self, brax_env, spec: NativeBraxTaskSpec, *, num_envs: int, seed: int) -> None:
        import jax

        self._env = brax_env
        self._spec = spec
        self.num_envs = int(num_envs)
        self._seed = int(seed)

        # Action dim is brax_env.action_size (already scalar int on wrapped env).
        self.action_size = int(brax_env.action_size)
        self.observation_size = int(brax_env.observation_size)
        assert self.observation_size == spec.obs_dim + spec.goal_dim, (
            f"Brax env {spec.env_id!r} exposes observation_size={self.observation_size} but the "
            f"registry expects state+goal = {spec.obs_dim}+{spec.goal_dim} = {spec.obs_dim + spec.goal_dim}."
            " Update _BRAX_ENV_REGISTRY if upstream obs layout changed."
        )

        # Pre-jit reset/step for perf and for jit-vs-jit parity against scaling-crl.
        self._reset_jit = jax.jit(self._env.reset)
        self._step_jit = jax.jit(self._env.step)

    @property
    def obs_dim(self) -> int:
        return self._spec.obs_dim

    @property
    def goal_dim(self) -> int:
        return self._spec.goal_dim

    @property
    def goal_start_idx(self) -> int:
        return self._spec.goal_start_idx

    @property
    def goal_end_idx(self) -> int:
        return self._spec.goal_end_idx

    @property
    def backend(self) -> str:
        return "brax-native"

    @property
    def unwrapped(self):
        return self._env

    def reset(self, rng=None) -> BraxLikeState:
        """Reset the vectorized env.

        Args:
            rng: Optional PRNGKey; if None, derived from :attr:`_seed`.
        """
        import jax

        if rng is None:
            rng = jax.random.PRNGKey(self._seed)
        keys = jax.random.split(rng, self.num_envs)
        state = self._reset_jit(keys)
        return _to_brax_like(state)

    def step(self, state: BraxLikeState, action) -> BraxLikeState:
        """Step the env with a per-env action. Expects ``state._brax`` to exist."""
        brax_state = state.info["__brax_state__"]
        nstate = self._step_jit(brax_state, action)
        return _to_brax_like(nstate)


def _to_brax_like(brax_state) -> BraxLikeState:
    """Convert a Brax ``State`` to our :class:`BraxLikeState`.

    We stash the original Brax state inside ``info["__brax_state__"]`` because
    :meth:`NativeBraxEnv.step` needs a Brax-native ``State`` as input (the
    envs's internal pipeline state is not reconstructable from just ``obs``).
    """
    info = dict(brax_state.info)
    info["__brax_state__"] = brax_state
    # ``truncation`` / ``seed`` are already populated by EpisodeWrapper and by
    # scaling-crl's Brax env implementations respectively. If missing on a
    # less-standard env, fall back to zeros.
    import jax.numpy as jnp

    num_envs = brax_state.obs.shape[0]
    info.setdefault("truncation", jnp.zeros(num_envs, dtype=jnp.float32))
    info.setdefault("seed", jnp.zeros(num_envs, dtype=jnp.float32))
    return BraxLikeState(
        obs=brax_state.obs,
        reward=brax_state.reward,
        done=brax_state.done,
        metrics=dict(getattr(brax_state, "metrics", {})),
        info=info,
    )
