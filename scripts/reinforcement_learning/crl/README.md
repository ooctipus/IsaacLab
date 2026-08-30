# Contrastive Reinforcement Learning (CRL) for IsaacLab

This directory hosts the training entry point for a Contrastive RL agent on the
sparse-reward ANYmal position task, backed by the ``scaling-crl`` (JAX/Brax) code
base.

Companion files:

- [`dlpack_bridge.py`](./dlpack_bridge.py) — zero-copy torch ↔ jax bridge.
- [`isaaclab_brax_adapter.py`](./isaaclab_brax_adapter.py) — wraps
  :class:`isaaclab.envs.ManagerBasedRLEnv` so scaling-crl sees a Brax-style env.
- [`native_brax_env.py`](./native_brax_env.py) — same interface over a native
  Brax env, used for Stage-A reproduction.
- [`crl_core.py`](./crl_core.py) — pure-JAX update and HER-relabel closures,
  importable without IsaacLab (what the update-parity regression test runs).
- [`metric_logger.py`](./metric_logger.py) — per-run JSONL + TensorBoard writer.
- [`train.py`](./train.py) — training entry script. ``--task native:ant``
  selects the Brax-native reproduction path; ``--task Isaac-Position-CRL-
  Anymal-C-v0`` is the main IsaacLab run.
- [`ROLLOUT_ANALYSIS.md`](./ROLLOUT_ANALYSIS.md) — rollout / jit analysis.
- [`STAGE_B_PLAN.md`](./STAGE_B_PLAN.md) — design for the IsaacLab-native
  Ant/Humanoid ports that bridge Stages A and C.

## 1. Environment setup

### 1a. Clone scaling-crl and set up its venv

The JAX training stack lives in a separate uv venv to avoid CUDA-library clashes
with the IsaacLab Python env:

```bash
# (one-time) from the IsaacLab repo root
git clone https://github.com/wang-kevin3290/scaling-crl.git dep/scaling-crl
cd dep/scaling-crl && uv sync && cd -
bash dep/scaling-crl/apply_brax_patches.sh dep/scaling-crl/.venv
```

### 1b. Install JAX + Flax + Optax + Brax inside the IsaacLab env

The IsaacLab env needs the same JAX stack so our adapter can co-locate the torch
tensors (PhysX) with the JAX arrays (training update) via DLPack:

```bash
./isaaclab.sh -p -m pip install \
    'jax[cuda12]==0.4.23' \
    'jaxlib==0.4.23+cuda12.cudnn89' \
    'flax==0.7.4' \
    'optax' \
    'brax==0.10.1' \
    'mujoco==3.2.6' \
    'tyro'
bash dep/scaling-crl/apply_brax_patches.sh $(./isaaclab.sh -p -c "import sys; print(sys.prefix)")
```

The brax patch script is idempotent; rerun it whenever the venv is rebuilt.

## 2. Validation pipeline

Every phase of the plan ships with a self-contained test so a failure points at
one phase, not the accumulated state of multiple phases. Tests are tagged **A**
(framework-agnostic — reusable under a future torch-native CRL rewrite) or
**S** (stack-specific — JAX / DLPack / scaling-crl internals).

### 2a. Interface-level tests

| Phase | Test file | Tag | Needs |
|---|---|---|---|
| 1 | [`tests/crl/test_sparse_env.py`](../../../tests/crl/test_sparse_env.py) | A | IsaacLab (config only); subset with Sim |
| 1 | [`tests/crl/test_goal_semantics.py`](../../../tests/crl/test_goal_semantics.py) | A | IsaacLab (config only) |
| 2 | Manual: see §3 below | S | GPU + scaling-crl venv |
| 3a | [`tests/crl/test_dlpack.py`](../../../tests/crl/test_dlpack.py) | S | CUDA + jax + torch |
| 3b | [`tests/crl/test_adapter_mock.py`](../../../tests/crl/test_adapter_mock.py) | A+S | CUDA |
| 4 | [`tests/crl/test_her_relabel.py`](../../../tests/crl/test_her_relabel.py) | A | scaling-crl venv (JAX) |
| 4 | [`tests/crl/test_info_nce.py`](../../../tests/crl/test_info_nce.py) | A | JAX |
| 4 | [`tests/crl/test_train_init.py`](../../../tests/crl/test_train_init.py) | S | scaling-crl venv |
| 5 | [`tests/crl/test_crl_on_toy_env.py`](../../../tests/crl/test_crl_on_toy_env.py) | A | JAX (slow) |

### 2b. Reproduction-level tests (prove our pipeline == scaling-crl)

These are the tests that answer "are we actually running the paper's algorithm?"
If any of them fail, our re-implementation of scaling-crl's training loop has
drifted from upstream and depth-scaling experiments on AnymalC would be
measuring the drift, not the paper's claim.

| Level | Test file | What it pins | Needs |
|---|---|---|---|
| **Update** | [`tests/crl/test_update_parity.py`](../../../tests/crl/test_update_parity.py) | `crl_core.make_update_fns` is bit-identical to scaling-crl's `update_actor_and_alpha` / `update_critic` when both are jit'd on identical inputs. | JAX |
| **Rollout** | [`tests/crl/test_rollout_parity.py`](../../../tests/crl/test_rollout_parity.py) | Eager rollout + `NativeBraxEnv` produces the same transitions as scaling-crl's `jax.lax.scan`-ed rollout on native Brax Ant. | scaling-crl venv |
| **End-to-end** | [`tests/crl/test_e2e_parity.py`](../../../tests/crl/test_e2e_parity.py) | After *N* full rollout+update iterations our pipeline produces **byte-identical** training state to a verbatim scaling-crl reference loop — catches integration-level drift (buffer management, key routing, minibatch slicing). | scaling-crl venv |
| **End-to-end Stage B** | Deferred — see [`STAGE_B_PLAN.md`](./STAGE_B_PLAN.md) | `train.py --task Isaac-Ant-CRL-v0` matches the Stage-A curves. Proves the IsaacLab adapter itself does not change behavior. | GPU + IsaacLab port of Brax Ant |
| **Stage C** | N/A — the real experiment | `train.py --task Isaac-Position-CRL-Anymal-C-v0` is the target run. | GPU + IsaacLab |

All three reproduction-parity tests currently **pass bit-identically** (13/13 in
the JAX-only validation matrix). This is the strongest possible evidence that
our training loop is scaling-crl's training loop: every parameter, every
transition, every SGD step matches byte-for-byte on a shared environment.

### Running Phase 1 (fast, no Isaac Sim)

```bash
./isaaclab.sh -p -m pytest tests/crl/test_sparse_env.py tests/crl/test_goal_semantics.py \
    -v -k "not needs_sim"
```

### Running Phase 4 framework-agnostic tests (JAX only)

These run inside the scaling-crl venv, which already has JAX:

```bash
cd dep/scaling-crl
uv add --dev pytest   # one-time
JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
    ../../tests/crl/test_her_relabel.py \
    ../../tests/crl/test_info_nce.py \
    ../../tests/crl/test_train_init.py \
    -v
```

### Running Phase 3 DLPack + adapter tests (requires GPU)

```bash
./isaaclab.sh -p -m pytest tests/crl/test_dlpack.py tests/crl/test_adapter_mock.py -v
```

### Running reproduction-level parity tests (the load-bearing ones)

Runs in the scaling-crl venv; JAX-only, CPU sufficient:

```bash
cd dep/scaling-crl
JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
    ../../tests/crl/test_update_parity.py \
    ../../tests/crl/test_rollout_parity.py \
    ../../tests/crl/test_e2e_parity.py \
    -v
```

Expected: 4 update-parity + 2 rollout-parity + 1 e2e-parity tests pass. Any
failure indicates our training loop has drifted from scaling-crl — investigate
before running depth-scaling experiments.

**E2E test details**: drives 5 full ``rollout → buffer insert → HER relabel →
actor update → critic update`` iterations on both our pipeline and a verbatim
scaling-crl reference pipeline, asserting **byte-identical** training states at
every iteration. Takes ~40 s on CPU.

## 3. Phase 2 sanity: verify scaling-crl runs natively

Before integrating, confirm the scaling-crl pipeline works on its built-in Ant
env. This rules out install / CUDA / wandb issues separately from our adapter
code:

```bash
cd dep/scaling-crl
uv run train.py --env_id ant --num_epochs 1 --total_env_steps 10000 \
    --num_envs 64 --batch_size 64 --wandb_mode offline --no-checkpoint
```

Success criteria: script runs to completion, ``critic_loss`` decreases over the
first 200 iterations (InfoNCE learns even in 200 steps on Ant).

## 4. Phase 5 hyperparameters and Phase 6 smoke test

### Hyperparameter table (Phase 5)

scaling-crl defaults are tuned for Brax envs (512 parallel envs, 1000-step
episodes). IsaacLab has different scales:

| Setting | scaling-crl default | our starting value | rationale |
|---|---|---|---|
| ``--num_envs`` | 512 | 1024 | IsaacLab scales well; below 4096 to leave JAX memory headroom |
| ``--episode_length`` | 1000 | 300 | 6 s × 50 Hz (position env default) |
| ``--unroll_length`` | 62 | 32 | must be ≤ ``episode_length``; shorter fills buffer faster |
| ``--critic_depth`` | 4 | 4 | start shallow (scaling experiments are future work) |
| ``--actor_depth`` | 4 | 4 | start shallow |
| ``--*_skip_connections`` | 0 | 0 | residuals off by default — re-enable only after baseline works |
| ``--batch_size`` | 256 | 256 | unchanged |
| ``--max_replay_size`` | 10000 | 10000 | unchanged |
| ``--num_sgd_batches_per_training_step`` | 800 | 50 | eager rollout changes the cost balance; reduce SGD steps per rollout to keep wall-clock manageable |
| ``--jax_mem_fraction`` | — | 0.3 | IsaacLab + PhysX need ~50% of a 24 GB card; 0.3 for JAX leaves headroom |

### Phase 6 smoke test (1M env steps)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/crl/train.py \
    --task Isaac-Position-CRL-Anymal-C-v0 \
    --num_envs 1024 \
    --episode_length 300 \
    --unroll_length 32 \
    --total_env_steps 1000000 \
    --num_epochs 10 \
    --actor_depth 4 --critic_depth 4 \
    --no-track
```

What to watch:

- ``actor_loss`` and ``critic_loss`` printed per epoch. Both should be finite
  from iteration 0.
- ``critic_loss`` (InfoNCE): should decrease over the first few epochs.
- The per-env success rate (available via IsaacLab's extras log) should become
  non-zero and gradually climb. If it stays at zero after 1M steps, likely
  causes: goal-representation mismatch (sanity-check via
  ``tests/crl/test_goal_semantics.py``), HER episode-boundary bug (rerun
  ``tests/crl/test_her_relabel.py``), or insufficient replay capacity (bump
  ``--max_replay_size``).

### Baseline comparison

Phase 6 is most informative against a PPO baseline on the *same* sparse env.
Register a baseline `rsl_rl_cfg_entry_point` on the CRL env (use a trivial one
that picks the ``flat`` obs preset) and run

```bash
uv run isaaclab train --rl_library rsl_rl \
    --task Isaac-Position-CRL-Anymal-C-v0
```

If both curves are flat, the env itself is too hard without dense rewards and
the experimental question is not about CRL vs PPO. If PPO picks up but CRL does
not, the problem is specifically in the CRL pipeline — debug via the tests in
§2.

## 5. Logging and visualization

Each training run produces a timestamped subdirectory under ``--log_dir``:

```
logs/crl/
└── native_ant_depth4_seed1000_20260416-141745/
    ├── config.json          # full CLI args + resolved config (for reproducibility)
    ├── metrics.jsonl        # one JSON record per epoch, streamable
    └── tb/                  # TensorBoard event files
```

Per-epoch metrics logged: ``actor_loss``, ``critic_loss``, ``log_alpha``,
``sample_entropy``, ``critic_logsumexp``, ``rollout_reward_mean``,
``rollout_done_rate``, plus ``env_steps`` as the x-axis.

### TensorBoard (the primary viewer)

```bash
# View one run or all runs (tensorboard auto-discovers subdirs)
tensorboard --logdir logs/crl

# Then open http://localhost:6006
```

For cross-run comparison (e.g. a depth sweep), point ``--logdir`` at the parent
directory — each subdirectory becomes its own legend entry automatically. Live
updates during training work out of the box.

``tensorboardX`` is already installed in the scaling-crl venv. If you're running
with a different Python env: ``pip install tensorboardX tensorboard``.

### JSONL (for scripted post-processing)

```python
import pandas as pd
df = pd.read_json("logs/crl/<run>/metrics.jsonl", lines=True)
df.plot(x="env_steps", y="critic_loss")
```

Useful for batch-generating publication figures or custom multi-seed aggregates
that TensorBoard's smoothing can't do.

## 7. Stage A end-to-end reproduction run (Brax Ant through our pipeline)

Use this to verify that our training loop on a native Brax env reproduces the
same loss curves as scaling-crl's native `train.py`. Do this at least once
before claiming any AnymalC result is faithful:

```bash
# (1) Scaling-crl baseline (runs in dep/scaling-crl venv)
cd dep/scaling-crl
.venv/bin/python train.py --env_id ant --num_envs 512 --batch_size 256 \
    --total_env_steps 2000000 --num_epochs 20 --seed 0 \
    --wandb_mode offline --no-checkpoint

# (2) Our pipeline on the same env (from IsaacLab root)
cd ../..
./dep/scaling-crl/.venv/bin/python scripts/reinforcement_learning/crl/train.py \
    --task native:ant \
    --num_envs 512 --batch_size 256 \
    --total_env_steps 2000000 --num_epochs 20 --seed 0 \
    --num_sgd_batches_per_training_step 800 \
    --no-track
```

Notes on matching:
- Set ``--num_sgd_batches_per_training_step 800`` (scaling-crl's default) not
  our default of 50. This is a *hyperparameter* divergence our default picked
  for eager-rollout tractability; for parity we use upstream's value.
- Both runs should be on the same GPU. JAX device non-determinism is not a
  concern (our update bodies jit to the same HLO at any seed).

Acceptance: overlay the two ``critic_loss`` and ``actor_loss`` curves. They
should be indistinguishable within the tiny numerical-noise band produced by
JAX's scan vs our eager reduction order.

Note that the unit-level update- and rollout-parity tests are the strict version
of this — they check *bit-identical* output. The end-to-end run is a looser
sanity check on top: it catches integration drift (replay-buffer API, prefill
timing, epoch bookkeeping) that the unit tests don't exercise.

## 8. Future: torch-native CRL (Phase 7, out of scope)

The test matrix is designed so that a future torch-native CRL implementation can
reuse every **A**-tagged test verbatim. To port:

1. Rewrite actor, bilinear critic, InfoNCE loss, HER replay buffer, and the SAC-
   style update in PyTorch, ideally co-located under ``dep/rsl_rl`` to match the
   existing SimBa / ResidualMLP infrastructure.
2. Drop ``dlpack_bridge.py`` and remove the JAX dependency.
3. Replace ``test_train_init.py`` with a torch-side equivalent. The other **A**
   tests pass unchanged; they pin the algorithmic contract, not the framework.
