# Phase 3c: scaling-crl rollout analysis

This note records the outcome of inspecting `dep/scaling-crl/train.py`'s rollout
and update loops and the resulting integration decision.

## scaling-crl's rollout structure

```
training_epoch  (@jax.jit + jax.lax.scan)
 └─ training_step  (@jax.jit)
     ├─ get_experience  (@jax.jit, jax.lax.scan over unroll_length)
     │   └─ actor_step: env.step(state, action)
     ├─ replay_buffer.sample
     ├─ TrajectoryUniformSamplingQueue.flatten_crl_fn  (HER relabel, @jax.jit)
     └─ jax.lax.scan(sgd_step, ...)
         └─ update_actor_and_alpha + update_critic  (@jax.jit)
```

Key observations:

1. **`env.step` is inside `jax.jit` + `jax.lax.scan`.** The whole `training_epoch`
   is traced by XLA; env.step is expected to be a pure JAX function.
2. **Replay buffer inserts/samples inside the jit boundary.** Buffer uses
   `flax.struct.dataclass` with a fixed-shape `data` jnp array. Insert/sample are
   jit-safe. No host-side round trips in the hot path.
3. **HER relabel inside the jit boundary.** `flatten_crl_fn` is `@jax.jit`.

## The incompatibility

IsaacLab's `env.step` is a torch / PhysX GPU call — emphatically *not* a pure
JAX function. It can't be inlined into XLA tracing. Options:

**Option A — `jax.pure_callback` wrap env.step.**
Feasible but awkward:
- `pure_callback` returns a Python object; JAX tracing still synthesizes shape
  info from `result_shape_dtypes`. Have to declare obs/reward/done shapes up front.
- Inside `jax.lax.scan`, every iteration crosses the XLA↔Python boundary; overhead
  per step is nontrivial.
- Debugging broken callbacks is hard: error tracebacks surface inside XLA.

**Option B — eager rollout, jit the update.**
Replace `get_experience` with a plain Python loop. Keep `update_actor_and_alpha`,
`update_critic`, `flatten_crl_fn`, and replay-buffer I/O under `@jax.jit`. Only the
actor forward-pass (cheap) happens per env step; everything else batches.

**Option B chosen.** Rationale:
- The bottleneck in IsaacLab-based training is the physics sim, not the update
  step. An eager rollout loop won't significantly slow the outer loop.
- Actor forward passes are small MLPs; jit'd `actor.apply` is already fast.
- Keeps debugging tractable: we can print per-env-step stats, pause on NaN, etc.
- Minimal delta from scaling-crl's code — `update_*` functions are reusable as-is.

## What our train.py does differently

We do **not** call `envs.training.wrap()` on the adapter. That wrapper adds three
things, all of which IsaacLab already handles:

| Wrapper | What it does | IsaacLab equivalent |
|---|---|---|
| `VmapWrapper` | `jax.vmap` over envs | IsaacLab is natively batched |
| `EpisodeWrapper` | Tracks `steps`, injects `truncation` in info | Adapter sets `info["truncation"]` from `truncated_t` |
| `AutoResetWrapper` | Resets done envs in place | IsaacLab auto-resets on termination |

We instead call the adapter methods directly from a rewritten `get_experience`
that iterates `unroll_length` times, building a per-step `Transition` and stacking
them at the end.

## HER compatibility checklist

scaling-crl's `flatten_crl_fn` requires:

1. `transition.observation` laid out as `[state(obs_dim), goal(goal_dim)]` per
   env. **Adapter: yes** (goal group placed last by `_build_layout`).
2. `transition.extras["state_extras"]["seed"]` identifying which episode each
   timestep belongs to, so HER doesn't sample goals across episode boundaries.
   **Adapter: yes** (per-env episode counter incremented on `done`).
3. `transition.extras["state_extras"]["truncation"]` distinguishing time-outs
   from real terminations (used for value-target computation).
   **Adapter: yes** (populated from `truncated_t` by the step call).

## Open question: goal semantic validity

`flatten_crl_fn` relabels: `goal = future_state[:, goal_start_idx:goal_end_idx]`.
For this to be HER-correct, the state slice `[goal_start_idx:goal_end_idx]` must
be in the same semantic space as the commanded goal. Our current default
(`achieved_goal_group=None`) points at the commanded-goal suffix — a placeholder.
Phase 4a decides the real representation; either:

- (A) Modify `position_crl_env_cfg.py` to expose an absolute-pose obs term inside
  the `policy` group and pass its slice indices through to the adapter.
- (B) Add a synthetic state projection in the adapter that computes the absolute
  pose from `policy` state at every step.

Option A is cleaner (no adapter-side math, keeps the adapter framework-agnostic).
