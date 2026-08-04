# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a Contrastive RL agent on an IsaacLab position task.

Pipeline:

1. Parse CLI args (tyro-style, compatible with scaling-crl's naming) and configure
   JAX GPU memory *before* importing JAX.
2. Launch Isaac Sim, build the sparse-reward CRL env, wrap with
   :class:`IsaacLabBraxEnv` so it looks like a Brax env.
3. Build the SimBa-style actor + bilinear critic (:class:`SA_encoder` /
   :class:`G_encoder`) by importing them from ``dep/scaling-crl/train.py``.
4. Allocate the HER-capable trajectory replay buffer
   (:class:`TrajectoryUniformSamplingQueue`) from ``dep/scaling-crl/buffer.py``.
5. Run an eager rollout / jit'd update loop. See ``ROLLOUT_ANALYSIS.md`` for the
   reasoning behind not using scaling-crl's jit'd ``get_experience``.

Example invocation (full GPU, smoke test):

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/crl/train.py \\
        --task Isaac-Position-CRL-Anymal-C-v0 \\
        --num_envs 1024 --episode_length 300 --unroll_length 32 \\
        --total_env_steps 1000000 --num_epochs 10 \\
        --actor_depth 4 --critic_depth 4 \\
        --no-track

The ``--no-track`` flag disables wandb; drop it to enable experiment logging.

Environment setup:

- The JAX stack (jax/jaxlib/flax/optax/brax==0.10.1) and IsaacLab must coexist in
  the Python environment used to launch this script. Typical ordering:

  .. code-block:: bash

      ./isaaclab.sh -p -m pip install 'jax[cuda12]==0.4.23' flax==0.7.4 \\
          optax tyro 'brax==0.10.1' mujoco==3.2.6

  Apply the two brax patches: ``bash dep/scaling-crl/apply_brax_patches.sh <venv>``.
- ``dep/scaling-crl`` must be a checkout of the scaling-crl repo; we add it to
  ``sys.path`` at runtime to import its model classes.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. CLI parsing and JAX-memory env-var setup. Must happen before ``import jax``.
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train a Contrastive-RL agent on an IsaacLab task.")
parser.add_argument("--task", type=str, required=True, help="Gym task id, e.g. Isaac-Position-CRL-Anymal-C-v0.")
parser.add_argument("--num_envs", type=int, default=1024, help="Parallel envs in IsaacLab.")
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--total_env_steps", type=int, default=1_000_000)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--episode_length", type=int, default=300, help="Must match env max_episode_length.")
parser.add_argument("--unroll_length", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_replay_size", type=int, default=10_000)
parser.add_argument("--min_replay_size", type=int, default=1_000)
parser.add_argument("--actor_lr", type=float, default=3e-4)
parser.add_argument("--critic_lr", type=float, default=3e-4)
parser.add_argument("--alpha_lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--entropy_param", type=float, default=0.5)
parser.add_argument("--disable_entropy", type=int, default=0)
parser.add_argument("--logsumexp_penalty_coeff", type=float, default=0.1)
parser.add_argument("--actor_depth", type=int, default=4)
parser.add_argument("--critic_depth", type=int, default=4)
parser.add_argument("--actor_network_width", type=int, default=256)
parser.add_argument("--critic_network_width", type=int, default=256)
parser.add_argument("--actor_skip_connections", type=int, default=0)
parser.add_argument("--critic_skip_connections", type=int, default=0)
parser.add_argument("--use_relu", type=int, default=0)
parser.add_argument("--num_sgd_batches_per_training_step", type=int, default=50)
parser.add_argument(
    "--scan_sgd",
    action="store_true",
    default=False,
    help=(
        "Fold the SGD-minibatch loop into a single jax.lax.scan. Mathematically "
        "identical to the default Python for-loop, but executes in one XLA launch; "
        "typically 5-10x throughput at small network sizes. Recommended on GPU."
    ),
)
parser.add_argument("--jax_mem_fraction", type=float, default=0.3, help="JAX GPU memory fraction.")
parser.add_argument("--log_dir", type=str, default="logs/crl")
parser.add_argument(
    "--track", dest="track", action="store_true", default=False, help="Enable wandb logging. Requires --wandb_project."
)
parser.add_argument("--no-track", dest="track", action="store_false")
parser.add_argument(
    "--wandb_project", type=str, default=None, help='E.g. "crl-repro" — used for the reproduction overlay chart.'
)
parser.add_argument(
    "--wandb_group",
    type=str,
    default=None,
    help='E.g. "stage_a_ant_depth4" — groups ours + scaling-crl native under one chart.',
)
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_tags", type=str, nargs="*", default=None)
parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
parser.add_argument(
    "--eval_every", type=int, default=5, help="Run deterministic eval rollouts every N epochs; 0 disables."
)
parser.add_argument("--num_eval_envs", type=int, default=128, help="Parallel eval envs per eval pass.")
parser.add_argument(
    "--goal_success_threshold",
    type=float,
    default=0.5,
    help="Distance threshold for success on native Brax envs (m for Ant).",
)
args_cli, remaining_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_args


# Configure JAX memory fraction *before* the first ``import jax`` anywhere in the
# process. The adapter module does ``import jax`` on first call, not on import,
# so the order here is safe if we configure now.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from dlpack_bridge import configure_jax_memory  # noqa: E402

configure_jax_memory(mem_fraction=args_cli.jax_mem_fraction, preallocate=False)

# ``--task native:<brax_env>`` is the Stage-A reproduction mode: drive our training
# loop with scaling-crl's native Brax env instead of IsaacLab. In this mode we skip
# ``AppLauncher`` entirely because no ``omni.*`` modules are involved.
_NATIVE_BRAX_MODE = args_cli.task.startswith("native:")

if not _NATIVE_BRAX_MODE:
    # ---------------------------------------------------------------------------
    # 1. Launch Isaac Sim and create the env. IsaacLab requires AppLauncher to run
    #    before importing anything that touches ``omni.*``.
    # ---------------------------------------------------------------------------
    from isaaclab.app import AppLauncher  # noqa: E402

    app_launcher = AppLauncher(headless=args_cli.headless, device=args_cli.device)
    simulation_app = app_launcher.app

    # Local imports that require AppLauncher to be live.
    import gymnasium as gym  # noqa: E402, F401

    import isaaclab_tasks  # noqa: F401, E402 — registers CRL tasks
else:
    simulation_app = None  # no Isaac Sim in native-Brax mode

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# 2. Import JAX and scaling-crl pieces. Keep these after AppLauncher so Isaac
#    Sim CUDA init runs before JAX grabs memory.
# ---------------------------------------------------------------------------

SCALING_CRL_DIR = Path(__file__).resolve().parents[3] / "dep" / "scaling-crl"
if not SCALING_CRL_DIR.is_dir():
    raise FileNotFoundError(
        f"Expected scaling-crl checkout at {SCALING_CRL_DIR}. Run: "
        "git clone https://github.com/wang-kevin3290/scaling-crl.git dep/scaling-crl"
    )
sys.path.insert(0, str(SCALING_CRL_DIR))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

# Import just the model classes from scaling-crl's train.py. The ``if __name__ ==
# "__main__":`` block there means bare ``import train`` does not kick off training.
# We rename to avoid collision with our own train.py.
import train as _scaling_crl_train  # noqa: E402
from buffer import TrajectoryUniformSamplingQueue  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

SA_encoder = _scaling_crl_train.SA_encoder
G_encoder = _scaling_crl_train.G_encoder
Actor = _scaling_crl_train.Actor
TrainingState = _scaling_crl_train.TrainingState
Transition = _scaling_crl_train.Transition

from crl_core import CRLCoreConfig, eager_actor_step, make_sgd_scan_fn, make_update_fns  # noqa: E402
from isaaclab_brax_adapter import BraxLikeState  # noqa: E402, F401
from metric_logger import MetricLogger  # noqa: E402

if not _NATIVE_BRAX_MODE:
    from isaaclab_brax_adapter import IsaacLabBraxEnv  # noqa: E402
else:
    from native_brax_env import make_native_brax_env  # noqa: E402


# ---------------------------------------------------------------------------
# 3. Build env + adapter.
# ---------------------------------------------------------------------------


@dataclass
class ResolvedConfig:
    """Runtime config that ties CLI args to env-derived quantities."""

    num_envs: int
    action_size: int
    obs_dim: int
    goal_dim: int
    goal_start_idx: int
    goal_end_idx: int
    unroll_length: int
    episode_length: int
    env_steps_per_actor_step: int
    num_prefill_actor_steps: int
    num_training_steps_per_epoch: int
    batch_size: int
    num_sgd_batches: int
    gamma: float
    logsumexp_penalty_coeff: float
    entropy_param: float
    disable_entropy: int
    target_entropy: float = 0.0
    goal_success_threshold: float = 0.5


def _build_env():
    """Construct the env adapter. Dispatches on ``--task`` prefix.

    - ``native:<brax_env>`` → :class:`NativeBraxEnv` on a scaling-crl Brax env
      (Stage-A reproduction mode).
    - otherwise → :class:`IsaacLabBraxEnv` on the IsaacLab gym-registered task.
    """
    if _NATIVE_BRAX_MODE:
        env_id = args_cli.task[len("native:") :]
        adapter, _spec = make_native_brax_env(
            env_id,
            episode_length=args_cli.episode_length,
            num_envs=args_cli.num_envs,
            seed=args_cli.seed,
        )
        return adapter

    env_cfg = None  # use task defaults
    base_env = gym.make(args_cli.task, cfg=env_cfg)
    base_env.unwrapped.scene.num_envs = args_cli.num_envs  # type: ignore[attr-defined]
    base_env.reset()
    adapter = IsaacLabBraxEnv(base_env.unwrapped)
    return adapter


def _resolve_config(adapter: IsaacLabBraxEnv) -> ResolvedConfig:
    num_envs = adapter.num_envs
    action_size = adapter.action_size
    env_steps_per_actor_step = num_envs * args_cli.unroll_length
    num_prefill_actor_steps = max(1, args_cli.min_replay_size // max(1, args_cli.unroll_length))
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    num_training_steps_per_epoch = max(
        1, (args_cli.total_env_steps - num_prefill_env_steps) // (args_cli.num_epochs * env_steps_per_actor_step)
    )
    return ResolvedConfig(
        num_envs=num_envs,
        action_size=action_size,
        obs_dim=adapter.obs_dim,
        goal_dim=adapter.goal_dim,
        goal_start_idx=adapter.goal_start_idx,
        goal_end_idx=adapter.goal_end_idx,
        unroll_length=args_cli.unroll_length,
        episode_length=args_cli.episode_length,
        env_steps_per_actor_step=env_steps_per_actor_step,
        num_prefill_actor_steps=num_prefill_actor_steps,
        num_training_steps_per_epoch=num_training_steps_per_epoch,
        batch_size=args_cli.batch_size,
        num_sgd_batches=args_cli.num_sgd_batches_per_training_step,
        gamma=args_cli.gamma,
        logsumexp_penalty_coeff=args_cli.logsumexp_penalty_coeff,
        entropy_param=args_cli.entropy_param,
        disable_entropy=args_cli.disable_entropy,
        target_entropy=-args_cli.entropy_param * action_size,
        goal_success_threshold=args_cli.goal_success_threshold,
    )


# ---------------------------------------------------------------------------
# 4. Build actor + critic + optimizers + replay buffer.
# ---------------------------------------------------------------------------


def _build_agents(cfg: ResolvedConfig, rng: jax.Array):
    actor_key, sa_key, g_key, buffer_key = jax.random.split(rng, 4)

    actor = Actor(
        action_size=cfg.action_size,
        network_width=args_cli.actor_network_width,
        network_depth=args_cli.actor_depth,
        skip_connections=args_cli.actor_skip_connections,
        use_relu=args_cli.use_relu,
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, cfg.obs_dim + cfg.goal_dim])),
        tx=optax.adam(learning_rate=args_cli.actor_lr),
    )

    sa_encoder = SA_encoder(
        network_width=args_cli.critic_network_width,
        network_depth=args_cli.critic_depth,
        skip_connections=args_cli.critic_skip_connections,
        use_relu=args_cli.use_relu,
    )
    sa_params = sa_encoder.init(sa_key, np.ones([1, cfg.obs_dim]), np.ones([1, cfg.action_size]))
    g_encoder = G_encoder(
        network_width=args_cli.critic_network_width,
        network_depth=args_cli.critic_depth,
        skip_connections=args_cli.critic_skip_connections,
        use_relu=args_cli.use_relu,
    )
    g_params = g_encoder.init(g_key, np.ones([1, cfg.goal_dim]))
    critic_state = TrainState.create(
        apply_fn=None,
        params={"sa_encoder": sa_params, "g_encoder": g_params},
        tx=optax.adam(learning_rate=args_cli.critic_lr),
    )

    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": log_alpha},
        tx=optax.adam(learning_rate=args_cli.alpha_lr),
    )

    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
    )

    dummy_transition = Transition(
        observation=jnp.zeros((cfg.obs_dim + cfg.goal_dim,)),
        action=jnp.zeros((cfg.action_size,)),
        reward=0.0,
        discount=0.0,
        extras={"state_extras": {"truncation": 0.0, "seed": 0.0}},
    )

    replay_buffer = TrajectoryUniformSamplingQueue(
        max_replay_size=args_cli.max_replay_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=cfg.batch_size,
        num_envs=cfg.num_envs,
        episode_length=cfg.episode_length,
    )
    replay_buffer.insert_internal = jax.jit(replay_buffer.insert_internal)
    replay_buffer.sample_internal = jax.jit(replay_buffer.sample_internal)
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    return actor, sa_encoder, g_encoder, training_state, replay_buffer, buffer_state


# ---------------------------------------------------------------------------
# 5. Eager rollout.
#    The jit'd update functions and HER relabel live in :mod:`crl_core` so the
#    same code path is exercised by ``tests/crl/test_update_parity.py``.
# ---------------------------------------------------------------------------


def _build_deterministic_eval(adapter, actor, episode_length: int, num_eval_envs: int, cfg: ResolvedConfig):
    """Construct a jit'd deterministic-actor eval function.

    Mirrors scaling-crl's :class:`CrlEvaluator`: runs ``num_eval_envs`` parallel
    envs for ``episode_length`` steps with the *deterministic* actor (mean action,
    no exploration noise) and aggregates:

    - ``eval/episode_reward`` — sum of Brax env rewards
    - ``eval/episode_success`` — fraction of envs that reach the commanded goal
      within the threshold by episode end (same semantics as scaling-crl)
    - ``eval/episode_dist_to_goal`` — final goal-distance averaged over envs.

    Works on :class:`NativeBraxEnv` (where the env state is a real Brax state)
    and on :class:`IsaacLabBraxEnv` (where the adapter handles the torch bridge).
    Uses ``adapter.step`` / ``adapter.reset`` in both cases so this function is
    env-type-agnostic.
    """
    from native_brax_env import NativeBraxEnv  # type: ignore[import-not-found]

    obs_dim, goal_start, goal_end = cfg.obs_dim, cfg.goal_start_idx, cfg.goal_end_idx

    assert isinstance(adapter, NativeBraxEnv), (
        "Deterministic eval currently only supported on NativeBraxEnv. Pass --eval_every 0 for IsaacLab env runs."
    )

    success_threshold = getattr(cfg, "goal_success_threshold", 0.5)

    @jax.jit
    def run_episode(actor_params, rng):
        """Run ``episode_length`` deterministic steps, return summary stats."""
        reset_keys = jax.random.split(rng, num_eval_envs)
        env_state = adapter._reset_jit(reset_keys)

        def step_fn(carry, _):
            es, rng = carry
            means, _ = actor.apply(actor_params, es.obs)
            action = jnp.tanh(means)
            nes = adapter._step_jit(es, action)
            state_goal = nes.obs[:, goal_start:goal_end]
            commanded_goal = nes.obs[:, obs_dim + goal_start : obs_dim + goal_end]
            dist = jnp.linalg.norm(state_goal - commanded_goal, axis=-1)
            reached = (dist < success_threshold).astype(jnp.float32)
            return (nes, rng), (es.reward, dist, reached)

        (final_state, _), (per_step_rewards, per_step_dist, per_step_reached) = jax.lax.scan(
            step_fn, (env_state, rng), xs=None, length=episode_length
        )

        total_reward_per_env = per_step_rewards.sum(axis=0)
        final_dist = per_step_dist[-1]  # distance at final step
        reached_any = (per_step_reached.sum(axis=0) > 0).astype(jnp.float32)  # reached goal *ever* during ep
        reached_final = per_step_reached[-1]  # reached goal at episode end
        return {
            "eval/episode_reward": total_reward_per_env.mean(),
            "eval/episode_dist_to_goal": final_dist.mean(),
            "eval/episode_dist_to_goal_min": final_dist.min(),
            "eval/episode_success": reached_final.mean(),
            "eval/episode_success_any": reached_any.mean(),
        }

    return run_episode


def _eager_rollout(adapter, env_state, actor_step_fn, training_state, rng, cfg: ResolvedConfig):
    """Run ``unroll_length`` env steps eagerly, returning stacked transitions.

    Eager (non-jit) wrapper around ``actor_apply`` + ``adapter.step``; env.step
    goes through PhysX (not JAX) so we cannot ``jax.lax.scan`` this.

    Returns:
        - next env_state
        - :class:`Transition` with a leading ``[unroll_length, num_envs]`` axes
    """
    obs_seq, act_seq, rew_seq, disc_seq, trunc_seq, seed_seq = [], [], [], [], [], []
    current_obs = env_state.obs
    for _ in range(cfg.unroll_length):
        rng, key = jax.random.split(rng)
        action, _, _ = actor_step_fn(training_state.actor_state.params, current_obs, key)
        env_state = adapter.step(env_state, action)
        obs_seq.append(current_obs)
        act_seq.append(action)
        rew_seq.append(env_state.reward)
        disc_seq.append(1.0 - env_state.done)
        trunc_seq.append(env_state.info["truncation"])
        seed_seq.append(env_state.info["seed"])
        current_obs = env_state.obs

    def stack(seq):
        return jnp.stack(seq, axis=0)  # [unroll, num_envs, ...]

    transitions = Transition(
        observation=stack(obs_seq),
        action=stack(act_seq),
        reward=stack(rew_seq),
        discount=stack(disc_seq),
        extras={
            "state_extras": {
                "truncation": stack(trunc_seq),
                "seed": stack(seed_seq),
            }
        },
    )
    return env_state, transitions, rng


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Resolve a timestamped run subdirectory so every invocation creates a fresh
    # log without clobbering prior runs. Structure:
    #   logs/crl/<task>_<timestamp>_<seed>/
    #       config.json
    #       metrics.jsonl
    #       tb/  (TensorBoard events)
    import datetime

    run_name = (
        f"{args_cli.task.replace(':', '_').replace('/', '_')}"
        f"_depth{args_cli.critic_depth}"
        f"_seed{args_cli.seed}"
        f"_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = os.path.join(args_cli.log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[CRL] log_dir: {run_dir}", flush=True)

    adapter = _build_env()
    cfg = _resolve_config(adapter)
    print(
        f"[CRL] obs_dim={cfg.obs_dim} goal_dim={cfg.goal_dim} "
        f"goal_slice=[{cfg.goal_start_idx}:{cfg.goal_end_idx}] "
        f"action_size={cfg.action_size} num_envs={cfg.num_envs}",
        flush=True,
    )

    # Snapshot the full CLI config for reproducibility.
    wandb_name = f"ours_{args_cli.task.replace(':', '_')}_depth{args_cli.critic_depth}_seed{args_cli.seed}"
    logger = MetricLogger(
        run_dir,
        config={**vars(args_cli), "resolved": vars(cfg)},
        enable_tensorboard=True,
        wandb_project=args_cli.wandb_project if args_cli.track else None,
        wandb_group=args_cli.wandb_group,
        wandb_entity=args_cli.wandb_entity,
        wandb_name=wandb_name,
        wandb_tags=(args_cli.wandb_tags or []) + ["ours"],
        wandb_mode=args_cli.wandb_mode,
    )

    rng = jax.random.PRNGKey(args_cli.seed)
    rng, agent_rng = jax.random.split(rng)
    actor, sa_encoder, g_encoder, training_state, replay_buffer, buffer_state = _build_agents(cfg, agent_rng)

    core_cfg = CRLCoreConfig(
        obs_dim=cfg.obs_dim,
        goal_dim=cfg.goal_dim,
        goal_start_idx=cfg.goal_start_idx,
        goal_end_idx=cfg.goal_end_idx,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        logsumexp_penalty_coeff=cfg.logsumexp_penalty_coeff,
        target_entropy=cfg.target_entropy,
        disable_entropy=cfg.disable_entropy,
    )
    update_actor_and_alpha, update_critic, relabel_and_batch = make_update_fns(actor, sa_encoder, g_encoder, core_cfg)
    sgd_scan_fn = make_sgd_scan_fn(update_actor_and_alpha, update_critic) if args_cli.scan_sgd else None
    actor_step_fn = jax.jit(lambda params, obs, key: eager_actor_step(actor, params, obs, key))

    # Deterministic eval: only wired up for native Brax envs (where the env lives in JAX).
    # On IsaacLab (torch/physics), eval would need an eager loop — add later if needed.
    eval_fn = None
    if args_cli.eval_every > 0 and _NATIVE_BRAX_MODE:
        eval_fn = _build_deterministic_eval(
            adapter,
            actor,
            episode_length=cfg.episode_length,
            num_eval_envs=args_cli.num_eval_envs,
            cfg=cfg,
        )
        print(
            f"[CRL] deterministic eval: every {args_cli.eval_every} epochs, "
            f"num_eval_envs={args_cli.num_eval_envs}, thresh={cfg.goal_success_threshold}",
            flush=True,
        )
    rng, eval_key = jax.random.split(rng)

    env_state = adapter.reset()

    # Prefill ------------------------------------------------------------
    print(f"[CRL] prefill: {cfg.num_prefill_actor_steps} actor steps", flush=True)
    for _ in range(cfg.num_prefill_actor_steps):
        env_state, tr, rng = _eager_rollout(adapter, env_state, actor_step_fn, training_state, rng, cfg)
        buffer_state = replay_buffer.insert(buffer_state, tr)
        training_state = training_state.replace(env_steps=training_state.env_steps + cfg.env_steps_per_actor_step)

    # Training loop ------------------------------------------------------
    try:
        for epoch in range(args_cli.num_epochs):
            epoch_actor_losses, epoch_critic_losses = [], []
            epoch_log_alpha, epoch_logsumexp, epoch_entropy = [], [], []
            rollout_rewards = []
            rollout_dones = []

            for _ in range(cfg.num_training_steps_per_epoch):
                # collect
                env_state, tr, rng = _eager_rollout(adapter, env_state, actor_step_fn, training_state, rng, cfg)
                rollout_rewards.append(float(tr.reward.mean()))
                rollout_dones.append(float((1.0 - tr.discount).mean()))
                buffer_state = replay_buffer.insert(buffer_state, tr)
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + cfg.env_steps_per_actor_step
                )
                # sample + relabel
                buffer_state, sampled = replay_buffer.sample(buffer_state)
                rng, relabel_key = jax.random.split(rng)
                batches = relabel_and_batch(sampled, relabel_key)

                # sgd across minibatches — scanned fast path or Python loop.
                num_batches = min(batches.observation.shape[0], cfg.num_sgd_batches)
                sliced = jax.tree_util.tree_map(lambda x: x[:num_batches], batches)
                if sgd_scan_fn is not None:
                    rng, sgd_key = jax.random.split(rng)
                    training_state, metrics = sgd_scan_fn(training_state, sliced, sgd_key)
                    # metrics dict has leading [num_batches] axis — average across that.
                    epoch_actor_losses.append(float(metrics["actor_loss"].mean()))
                    epoch_critic_losses.append(float(metrics["critic_loss"].mean()))
                    epoch_log_alpha.append(float(metrics["log_alpha"].mean()))
                    epoch_entropy.append(float(metrics["sample_entropy"].mean()))
                    epoch_logsumexp.append(float(metrics["logsumexp"].mean()))
                else:
                    for b in range(num_batches):
                        rng, ak, ck = jax.random.split(rng, 3)
                        batch = jax.tree_util.tree_map(lambda x: x[b], sliced)
                        training_state, actor_metrics = update_actor_and_alpha(batch, training_state, ak)
                        training_state, critic_metrics = update_critic(batch, training_state, ck)
                        epoch_actor_losses.append(float(actor_metrics["actor_loss"]))
                        epoch_critic_losses.append(float(critic_metrics["critic_loss"]))
                        epoch_log_alpha.append(float(actor_metrics["log_alpha"]))
                        epoch_entropy.append(float(actor_metrics["sample_entropy"]))
                        epoch_logsumexp.append(float(critic_metrics["logsumexp"]))

            # Same namespace prefix as scaling-crl (``training/``) so when both
            # pipelines write to the same wandb project, the charts overlay on
            # common keys. Eval metrics use the ``eval/`` prefix identically.
            env_steps_now = int(training_state.env_steps)
            # Dual-log env_steps: once as the wandb step axis, and once as a
            # value under the same key scaling-crl uses (``training/envsteps``)
            # so cross-pipeline plots can align on a shared x-axis key.
            metrics = {
                "epoch": epoch,
                "env_steps": env_steps_now,
                "training/envsteps": env_steps_now,
                "training/actor_loss": float(np.mean(epoch_actor_losses)),
                "training/critic_loss": float(np.mean(epoch_critic_losses)),
                "training/log_alpha": float(np.mean(epoch_log_alpha)),
                "training/sample_entropy": float(np.mean(epoch_entropy)),
                "training/logsumexp": float(np.mean(epoch_logsumexp)),
                "training/rollout_reward_mean": float(np.mean(rollout_rewards)),
                "training/rollout_done_rate": float(np.mean(rollout_dones)),
            }

            # Deterministic eval pass (every N epochs, including epoch 0 and last).
            if eval_fn is not None and (epoch % args_cli.eval_every == 0 or epoch == args_cli.num_epochs - 1):
                eval_key, eval_subkey = jax.random.split(eval_key)
                eval_metrics = eval_fn(training_state.actor_state.params, eval_subkey)
                for k, v in eval_metrics.items():
                    metrics[k] = float(v)

            logger.log(step=metrics["env_steps"], **metrics)

            print(
                f"[CRL] epoch={epoch} env_steps={metrics['env_steps']} "
                f"actor_loss={metrics['training/actor_loss']:.4f} "
                f"critic_loss={metrics['training/critic_loss']:.4f} "
                f"success={metrics.get('eval/episode_success', float('nan')):.3f}",
                flush=True,
            )
    finally:
        logger.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app is not None:
            simulation_app.close()
