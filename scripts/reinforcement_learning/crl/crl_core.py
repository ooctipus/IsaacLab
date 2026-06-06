# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-JAX core of the CRL training loop.

Lives apart from :mod:`train` so it can be imported without launching Isaac Sim —
this is what enables the update-parity regression test to exercise our re-
implementations of scaling-crl's update functions in isolation.

All functions in this module are deliberately ignorant of IsaacLab / adapters /
torch; they consume Flax modules, :class:`flax.training.train_state.TrainState`
objects, and JAX arrays, and return the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CRLCoreConfig:
    """Minimal config shared between the training loop and the update functions.

    Mirrors the subset of ``scaling-crl.train.Args`` that the pure-JAX update
    step actually reads. Kept frozen so it can be hashed into ``jax.jit`` cache
    keys safely.
    """

    obs_dim: int
    goal_dim: int
    goal_start_idx: int
    goal_end_idx: int
    batch_size: int
    gamma: float
    logsumexp_penalty_coeff: float
    target_entropy: float
    disable_entropy: int = 0


def make_update_fns(actor, sa_encoder, g_encoder, cfg: CRLCoreConfig):
    """Build jit-compiled actor-alpha and critic update closures.

    The bodies below are exactly what scaling-crl's ``update_actor_and_alpha`` /
    ``update_critic`` do; see :mod:`tests.crl.scaling_crl_reference` for the
    verbatim upstream copy that this module is tested against.
    """

    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        transitions = jax.tree_util.tree_map(lambda x: x[: cfg.batch_size], transitions)

        def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
            obs = transitions.observation
            state = obs[:, : cfg.obs_dim]
            future_state = transitions.extras["future_state"]
            goal = future_state[:, cfg.goal_start_idx : cfg.goal_end_idx]
            observation = jnp.concatenate([state, goal], axis=1)

            means, log_stds = actor.apply(actor_params, observation)
            stds = jnp.exp(log_stds)
            x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            action = nn.tanh(x_ts)
            log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)

            sa_p, g_p = critic_params["sa_encoder"], critic_params["g_encoder"]
            sa_repr = sa_encoder.apply(sa_p, state, action)
            g_repr = g_encoder.apply(g_p, goal)
            qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

            if cfg.disable_entropy:
                return -jnp.mean(qf_pi), log_prob
            return jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi), log_prob

        def alpha_loss(alpha_params, log_prob):
            alpha = jnp.exp(alpha_params["log_alpha"])
            return jnp.mean(alpha * jax.lax.stop_gradient(-log_prob - cfg.target_entropy))

        (actor_l, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
            training_state.actor_state.params,
            training_state.critic_state.params,
            training_state.alpha_state.params["log_alpha"],
            transitions,
            key,
        )
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        alpha_l, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
        new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

        training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)
        return training_state, {
            "actor_loss": actor_l,
            "alpha_loss": alpha_l,
            "log_alpha": training_state.alpha_state.params["log_alpha"],
            "sample_entropy": -jnp.mean(log_prob),
        }

    @jax.jit
    def update_critic(transitions, training_state, key):
        transitions = jax.tree_util.tree_map(lambda x: x[: cfg.batch_size], transitions)

        def critic_loss(critic_params, transitions, key):
            sa_p, g_p = critic_params["sa_encoder"], critic_params["g_encoder"]
            obs = transitions.observation[:, : cfg.obs_dim]
            action = transitions.action
            sa_repr = sa_encoder.apply(sa_p, obs, action)
            g_repr = g_encoder.apply(g_p, transitions.observation[:, cfg.obs_dim :])
            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
            loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            loss += cfg.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)
            return loss, logsumexp

        (loss, logsumexp), grad = jax.value_and_grad(critic_loss, has_aux=True)(
            training_state.critic_state.params, transitions, key
        )
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state=new_critic_state)
        return training_state, {"critic_loss": loss, "logsumexp": logsumexp.mean()}

    @jax.jit
    def relabel_and_batch(transitions, key):
        """HER relabel a sampled trajectory slice then reshape into minibatches."""
        from buffer import TrajectoryUniformSamplingQueue  # type: ignore[import-not-found]

        batch_keys = jax.random.split(key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (cfg.gamma, cfg.obs_dim, cfg.goal_start_idx, cfg.goal_end_idx), transitions, batch_keys
        )
        transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions)
        perm = jax.random.permutation(key, transitions.observation.shape[0])
        transitions = jax.tree_util.tree_map(lambda x: x[perm], transitions)
        num_full = transitions.observation.shape[0] // cfg.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[: num_full * cfg.batch_size], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, cfg.batch_size) + x.shape[1:]),
            transitions,
        )
        return transitions

    return update_actor_and_alpha, update_critic, relabel_and_batch


def eager_actor_step(actor, actor_params, obs, key):
    """Stochastic actor step: returns ``(action, means, log_stds)``.

    Extracted from our rollout loop so tests can exercise identical action-
    sampling semantics against the reference actor step.
    """
    means, log_stds = actor.apply(actor_params, obs)
    stds = jnp.exp(log_stds)
    action = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))
    return action, means, log_stds


def make_sgd_scan_fn(update_actor_and_alpha, update_critic):
    """Fold the inner SGD-minibatch loop into a single ``jax.lax.scan``.

    Mathematically identical to running::

        for b in range(num_batches):
            key, ak, ck = jax.random.split(key, 3)
            batch = jax.tree_util.tree_map(lambda x: x[b], batches)
            ts, _ = update_actor_and_alpha(batch, ts, ak)
            ts, _ = update_critic(batch, ts, ck)

    but executes in one XLA launch instead of ``2 * num_batches`` Python-side
    dispatches. This is the primary throughput knob for GPU-saturation —
    at depth=4/width=256 the update step is tiny; the bottleneck is
    Python↔XLA round-tripping, not compute.

    Args:
        update_actor_and_alpha: jit'd function ``(batch, ts, key) -> (ts, metrics)``
            as returned by :func:`make_update_fns`.
        update_critic: same signature, critic version.

    Returns:
        A function ``sgd_scan(training_state, batches, key) -> (training_state, metrics)``
        where ``batches`` has leading axis = num SGD batches, and ``metrics`` is a
        per-batch-stacked pytree of scalars.
    """

    @jax.jit
    def sgd_scan(training_state, batches, key):
        def step(carry, batch):
            ts, rng = carry
            rng, ak, ck = jax.random.split(rng, 3)
            ts, am = update_actor_and_alpha(batch, ts, ak)
            ts, cm = update_critic(batch, ts, ck)
            return (ts, rng), {**am, **cm}

        (training_state, _), metrics = jax.lax.scan(step, (training_state, key), batches)
        return training_state, metrics

    return sgd_scan
