# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter wrapping an IsaacLab :class:`ManagerBasedRLEnv` so it looks like a Brax env.

The ``scaling-crl`` training loop (``dep/scaling-crl/train.py``) consumes a Brax-style
env with:

- ``env.reset(rng) -> State`` returning a batched ``State(obs, reward, done, ..., info)``.
- ``env.step(state, action) -> State`` stepping a batched env.
- ``env.observation_size`` / ``env.action_size`` scalars.
- ``obs`` laid out as ``[state_slice, goal_slice]`` where ``state_slice`` has size
  ``obs_dim`` and ``goal_slice`` is the commanded goal of size ``goal_dim``.
- A stable slice inside ``state_slice`` at ``[goal_start_idx:goal_end_idx]`` that is
  the *achieved goal* in the same semantic space as the commanded goal (HER uses it).

Our IsaacLab env produces a dict of torch tensors. The adapter:

1. Concatenates obs groups into a single flat vector.
2. Records where each obs term lives so HER's goal slice is well-defined.
3. Moves the commanded-goal term to the END of the flat vector to match scaling-crl's
   layout expectation.
4. Uses :mod:`dlpack_bridge` for zero-copy torch <-> jax transfers at the step boundary.

This adapter does **not** need to be wrapped with ``envs.training.wrap()`` because
IsaacLab already handles (a) env vectorization, (b) auto-reset on termination, and
(c) episode-length truncation.

Goal representation (Phase 4a status):
    We currently route ``goal_point_commands`` (relative pose in base frame) as both
    the commanded goal AND the reachable-state slice. This is a placeholder. For
    HER to produce semantically valid relabeled goals, the ``[goal_start_idx:
    goal_end_idx]`` slice must be an *absolute reachable pose*. See
    ``IsaacLabBraxEnv._resolve_goal_slice`` for the extension point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import jax

    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class BraxLikeState:
    """Duck-typed :class:`brax.envs.base.State` replacement.

    We avoid importing ``brax.envs.base.State`` because it carries ``pipeline_state``
    fields that are meaningless for IsaacLab and that Flax would try to serialize. A
    plain dataclass with the fields scaling-crl actually reads is sufficient.

    Attributes:
        obs: Flat observation vector, shape ``[num_envs, obs_dim + goal_dim]``.
        reward: Per-env reward, shape ``[num_envs]``. Always zero in the sparse-CRL env.
        done: Per-env done flag (1 if terminated), shape ``[num_envs]``.
        metrics: Optional scalar metrics per env (unused in the sparse setting).
        info: Must contain ``truncation`` (time-out flag) and ``seed`` (placeholder)
              so scaling-crl's ``actor_step`` can find them in ``extra_fields``.
    """

    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: dict[str, jax.Array] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def replace(self, **updates) -> BraxLikeState:
        """Flax-style functional update helper."""
        from dataclasses import replace as _replace

        return _replace(self, **updates)


@dataclass(frozen=True)
class ObsLayout:
    """Flat-obs layout metadata.

    Records where each obs group lives in the flat vector after concatenation.
    The commanded-goal term is moved to the end so ``obs[:, :obs_dim]`` is the state.
    """

    group_order: tuple[str, ...]
    group_slices: dict[str, tuple[int, int]]
    obs_dim: int
    goal_dim: int
    goal_start_idx: int
    goal_end_idx: int
    goal_group_key: str
    goal_term_name: str

    @property
    def total_dim(self) -> int:
        """Full obs vector dim (state + goal)."""
        return self.obs_dim + self.goal_dim


class IsaacLabBraxEnv:
    """Brax-style adapter around :class:`isaaclab.envs.ManagerBasedRLEnv`.

    Args:
        env: Constructed IsaacLab env (not a cfg — the adapter does not create it).
        goal_group_key: Name of the obs group whose concatenated vector is the
            commanded goal. Defaults to ``"task"`` which matches the position-task
            env's ``goal_point_commands`` layout.
        goal_term_name: The specific obs term inside ``goal_group_key`` that holds
            the commanded goal. The adapter extracts exactly this term's slice so
            the goal-suffix layout is unambiguous.
        achieved_goal_group: Name of the obs group whose slice ``[achieved_goal_
            start, achieved_goal_end]`` gives the current achieved goal. If None,
            falls back to the commanded-goal term (a placeholder that should be
            replaced by an absolute-pose observation before real training).
        achieved_goal_slice: ``(start, end)`` indices into the concatenated
            ``achieved_goal_group`` vector identifying the achieved-goal sub-slice.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        *,
        goal_group_key: str = "task",
        goal_term_name: str = "target_pos_env",
        achieved_goal_group: str | None = "achieved_goal",
        achieved_goal_slice: tuple[int, int] | None = None,
    ) -> None:
        self._env = env
        self._goal_group_key = goal_group_key
        self._goal_term_name = goal_term_name
        self._achieved_goal_group = achieved_goal_group
        self._achieved_goal_slice = achieved_goal_slice

        self._layout = self._build_layout()

        self.observation_size: int = self._layout.total_dim
        self.action_size: int = (
            int(env.action_space.shape[-1])
            if hasattr(env, "action_space")
            else int(
                env.num_actions  # type: ignore[attr-defined]
            )
        )
        self.num_envs: int = int(env.num_envs)  # type: ignore[attr-defined]

        # Per-env episode-id counter. HER relies on transitions within a replay-buffer
        # slice belonging to the same episode; the buffer uses ``info["seed"]`` as
        # the episode-identity key (see ``scaling-crl/buffer.py::flatten_crl_fn``).
        # We increment this on every termination so each episode gets a unique id
        # within its env slot, matching what Brax's pusher/reacher envs do.
        import torch

        self._episode_id = torch.zeros(self.num_envs, dtype=torch.float32, device=env.device)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Brax-style property surface
    # ------------------------------------------------------------------

    @property
    def goal_start_idx(self) -> int:
        """Start index of the HER-relabeled-goal slice inside the state vector."""
        return self._layout.goal_start_idx

    @property
    def goal_end_idx(self) -> int:
        """End index (exclusive) of the HER-relabeled-goal slice inside the state vector."""
        return self._layout.goal_end_idx

    @property
    def obs_dim(self) -> int:
        """State-part dim (everything before the commanded-goal suffix)."""
        return self._layout.obs_dim

    @property
    def goal_dim(self) -> int:
        """Commanded-goal suffix dim."""
        return self._layout.goal_dim

    @property
    def layout(self) -> ObsLayout:
        """Return the full flat-obs layout for debugging / HER wiring."""
        return self._layout

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        return self._env

    @property
    def backend(self) -> str:
        """Required by the Brax ``Env`` interface; arbitrary string for IsaacLab."""
        return "isaaclab-physx"

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------

    def reset(self, rng=None) -> BraxLikeState:
        """Reset the env.

        Args:
            rng: Ignored. IsaacLab uses its own RNG; the argument exists only for
                Brax-shape compatibility.

        Returns:
            A :class:`BraxLikeState` with ``obs`` as a jax array of shape
            ``[num_envs, observation_size]`` and zero-initialized reward/done/info.
        """
        import jax.numpy as jnp
        from dlpack_bridge import torch_to_jax  # type: ignore[import-not-found]

        obs_dict, _ = self._env.reset()
        flat = self._flatten_obs(obs_dict)
        obs_j = torch_to_jax(flat)
        zeros = jnp.zeros((self.num_envs,), dtype=jnp.float32)
        # On fresh reset every env starts at episode 0.
        self._episode_id.zero_()
        seed_j = torch_to_jax(self._episode_id)
        return BraxLikeState(
            obs=obs_j,
            reward=zeros,
            done=zeros,
            metrics={},
            info={"truncation": zeros, "seed": seed_j, "first_obs": obs_j},
        )

    def step(self, state: BraxLikeState, action) -> BraxLikeState:
        """Step one env transition.

        Args:
            state: Current adapter state (obs not consumed; IsaacLab holds its
                own state internally). Passed through so the returned state can
                reuse episode-metadata fields (``first_obs`` etc).
            action: Jax array of shape ``[num_envs, action_size]``. Converted to
                a torch tensor via DLPack.

        Returns:
            Next :class:`BraxLikeState`.
        """

        from dlpack_bridge import jax_to_torch, torch_to_jax  # type: ignore[import-not-found]

        action_t = jax_to_torch(action)
        obs_dict, rew_t, terminated_t, truncated_t, info = self._env.step(action_t)

        flat = self._flatten_obs(obs_dict)
        obs_j = torch_to_jax(flat.float())
        rew_j = torch_to_jax(rew_t.float().view(-1))
        done_bool = terminated_t | truncated_t
        done_j = torch_to_jax(done_bool.to(rew_t.dtype).view(-1))
        truncation_j = torch_to_jax(truncated_t.to(rew_t.dtype).view(-1))

        # Increment episode id for envs that just terminated. IsaacLab auto-resets
        # terminated envs in place, so the next ``obs_dict`` already belongs to a
        # fresh episode in those slots — the new seed must differ from the previous.
        self._episode_id += done_bool.to(self._episode_id.dtype).view(-1)
        seed_j = torch_to_jax(self._episode_id)

        new_info = dict(state.info)
        new_info["truncation"] = truncation_j
        new_info["seed"] = seed_j
        return BraxLikeState(
            obs=obs_j,
            reward=rew_j,
            done=done_j,
            metrics={},
            info=new_info,
        )

    # ------------------------------------------------------------------
    # layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> ObsLayout:
        """Resolve the flat-obs layout from a single probe call to ``env.reset()``.

        We do one reset + inspect the dict-shaped observation to learn per-group
        sizes. The commanded-goal group is moved to the tail of the flat vector.
        """
        obs_mgr = getattr(self._env, "observation_manager", None)
        if obs_mgr is None:
            # Probe the obs via env.reset() — required by managers that build
            # shapes lazily. This is harmless before the first real reset.
            obs_dict, _ = self._env.reset()
        else:
            obs_dict = obs_mgr.get_observations()  # type: ignore[attr-defined]

        assert isinstance(obs_dict, dict), (
            f"IsaacLabBraxEnv currently supports dict observations only; got {type(obs_dict)}"
        )
        assert self._goal_group_key in obs_dict, (
            f"goal_group_key={self._goal_group_key!r} not found in obs_dict keys={list(obs_dict.keys())}"
        )

        # Determine the goal-slice size inside the goal group.
        # We only support the case where the goal term occupies the whole goal group.
        # For the position env this is the ``task`` group with a single term
        # ``goal_point_commands``; for a future multi-term task group an explicit
        # per-term slice API would be required.
        goal_group_tensor = obs_dict[self._goal_group_key]
        goal_dim = int(goal_group_tensor.shape[-1])

        # Build the state-side group order (everything except the goal group).
        state_groups = [k for k in obs_dict if k != self._goal_group_key]
        state_groups.sort()  # deterministic

        group_slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for k in state_groups:
            dim = int(obs_dict[k].shape[-1])
            group_slices[k] = (offset, offset + dim)
            offset += dim
        obs_dim = offset
        # Goal goes last
        group_slices[self._goal_group_key] = (obs_dim, obs_dim + goal_dim)

        # Resolve the HER-relabeled-goal slice (into the state region).
        achieved_group = self._achieved_goal_group
        achieved_slice = self._achieved_goal_slice
        if achieved_group is None:
            # Placeholder: fall back to the commanded-goal suffix. This is NOT
            # HER-correct in general — callers should supply ``achieved_goal_group``.
            goal_start_idx = obs_dim
            goal_end_idx = obs_dim + goal_dim
        else:
            if achieved_group not in group_slices:
                raise ValueError(
                    f"achieved_goal_group={achieved_group!r} not in obs groups {list(group_slices.keys())}"
                )
            group_start, group_end = group_slices[achieved_group]
            if achieved_slice is None:
                # Default: use the whole achieved-goal group as the HER slice.
                # This is the intended path when the env exposes a dedicated
                # ``achieved_goal`` obs group matching the goal dim.
                achieved_slice = (0, group_end - group_start)
            s, e = achieved_slice
            goal_start_idx = group_start + s
            goal_end_idx = group_start + e
            if goal_end_idx > group_end:
                raise ValueError(
                    f"achieved_goal_slice={achieved_slice} out of range for group {achieved_group} "
                    f"(size={group_end - group_start})"
                )
            # Sanity: the achieved-goal dim must match the commanded-goal dim so
            # HER's ``goal ← future_state[start:end]`` produces a valid goal vector.
            if (goal_end_idx - goal_start_idx) != goal_dim:
                raise ValueError(
                    f"achieved-goal slice size ({goal_end_idx - goal_start_idx}) "
                    f"does not match commanded-goal dim ({goal_dim}); HER relabel "
                    "would feed mismatched shapes to the critic goal encoder."
                )

        return ObsLayout(
            group_order=tuple(state_groups) + (self._goal_group_key,),
            group_slices=group_slices,
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            goal_start_idx=goal_start_idx,
            goal_end_idx=goal_end_idx,
            goal_group_key=self._goal_group_key,
            goal_term_name=self._goal_term_name,
        )

    def _flatten_obs(self, obs_dict):
        """Concatenate obs groups into a single ``[num_envs, total_dim]`` torch tensor.

        Order: all non-goal groups first (alphabetical), then the goal group.
        Matches ``self._layout.group_order``.
        """
        import torch

        parts = []
        for k in self._layout.group_order:
            parts.append(obs_dict[k])
        return torch.cat(parts, dim=-1)
