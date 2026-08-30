# Stage B — IsaacLab-native Ant / Humanoid reproduction (design notes)

Stage A proves that **our training loop on a native Brax env** is bit-identical to
scaling-crl. Stage B's job is to prove that **the adapter layer** (and anything
PhysX-specific about IsaacLab's env stepping) does not change the story. If CRL
learns the same curve on IsaacLab-Ant as it does on Brax-Ant, then any
depth-scaling result on AnyMal is attributable to AnyMal, not the adapter.

This document is the scaffold for that work. Implementation is deliberately
deferred — porting a Brax env to IsaacLab is several days of engineering that
should only happen once Stage A depth-4 training is observed to work.

## Scope

- **IsaacLab-Ant**: 8-DoF quadruped, identical joint DOF list to Brax Ant,
  forward-locomotion goal = ``root_xy`` relative to episode-start position.
  29-d obs, 2-d goal slice, matches Brax Ant's registry entry.
- **IsaacLab-Humanoid** (optional, expensive): 17-DoF humanoid, 268-d obs,
  3-d goal slice. Worth having iff IsaacLab-Ant matches but we suspect humanoid-
  specific issues in the adapter.

## Reproduction acceptance bar

Given matched hyperparameters (same ``unroll_length``, ``batch_size``, seeds,
network shapes), the following should hold at 1M env steps:

| Metric | Brax-Ant | IsaacLab-Ant | Tolerance |
|---|---|---|---|
| `critic_loss` at 1M | X ± σ | X' ± σ' | \|mean diff\| ≤ 0.5σ |
| `success_rate` at 1M | Y ± σ | Y' ± σ' | \|mean diff\| ≤ 0.1 |
| Wall-clock for 1M | tB | tIL | tIL within 3× tB |

If the means are indistinguishable within seed variance, the adapter is faithful.

## Implementation plan

### B1. Env port

Create `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/ant_crl/`:

```
ant_crl/
├── __init__.py            # gym.register(Isaac-Ant-CRL-v0, ...)
├── ant_env_cfg.py         # ManagerBasedRLEnvCfg matching Brax Ant specs
├── mdp/
│   ├── observations.py    # root_xy, joint_pos, joint_vel, ... matching Brax's 29-d obs
│   ├── rewards.py         # empty (CRL is self-supervised)
│   ├── terminations.py    # base-upright check → "terminate_when_unhealthy"
│   └── commands.py        # random XY target goal
```

Match Brax Ant's obs layout exactly:

```
[0:2]   root_xy - root_xy_initial           (goal slice = achieved pose)
[2:5]   root_z, base-frame rotation
[5:13]  joint_pos (8 DoFs)
[13:15] root_lin_vel_xy
[15:18] root_lin_vel_z, root_ang_vel
[18:26] joint_vel
[26:27] padded or time_step
[27:29] commanded_goal_xy
```

Exact match is necessary because the goal-slice indices
``(obs_dim=29, goal_start=0, goal_end=2)`` are what the CRL training loop uses.

USD/URDF sourcing: Isaac Sim ships an Ant USD that's geometry-equivalent to
Brax's ant.xml. PhysX joint config needs to mirror Brax (PD gains may differ,
which is fine — the test is "does CRL learn?" not "are trajectories identical
at step 0").

### B2. Commands / goals

Brax Ant's goal is a random XY target within a bounded box (the "reach"
variant). Implement as:

```python
class AntGoalCommandCfg:
    # Sample target on reset from a uniform square [-5, 5] x [-5, 5]
    # relative to the spawn location (so env-local frame).
    ...
```

Mirror `RelativeStateCommand` but simplified to XY-only.

### B3. Termination

Brax's "terminate_when_unhealthy" fires when root_z exits [0.2, 1.0]. Port
directly — IsaacLab already has `root_height_below_minimum` and we can add a
`root_height_above_maximum`.

### B4. Registration

```python
gym.register(
    id="Isaac-Ant-CRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.ant_crl.ant_env_cfg:AntCRLEnvCfg",
    },
)
```

Our `train.py` already picks up non-native tasks via the `IsaacLabBraxEnv`
adapter; just set `--task Isaac-Ant-CRL-v0`.

### B5. Cross-validation run

Once B1–B4 land, run:

```bash
# Stage A ground truth
uv run python train.py --env_id ant --num_epochs 50 --total_env_steps 5000000 \
    --num_envs 512 --batch_size 256 --seed 0,1,2   # (loop over 3 seeds)

# Our Stage A reproduction
./isaaclab.sh -p scripts/reinforcement_learning/crl/train.py --task native:ant \
    --num_epochs 50 --total_env_steps 5000000 --num_envs 512 --batch_size 256 \
    --seed 0,1,2

# Stage B
./isaaclab.sh -p scripts/reinforcement_learning/crl/train.py --task Isaac-Ant-CRL-v0 \
    --num_epochs 50 --total_env_steps 5000000 --num_envs 512 --batch_size 256 \
    --seed 0,1,2
```

Overlay the three sets of 3-seed curves. If Stages A-native, A-ours, and B match
within seed variance → infrastructure is faithful, proceed to Stage C (AnymalC).
If A-native and A-ours match but B diverges → adapter bug. If A-native and A-ours
diverge → we regressed our training loop since the last parity test was run
(re-run `tests/crl/test_update_parity.py`).

## When to actually do this work

Not until Stage 0 (depth=4 AnymalC runs for 1M steps) either clearly succeeds
or clearly fails. Reasons:

- If AnymalC CRL learns at depth=4 already, the Stage B validation is moot —
  we've shown the pipeline works.
- If AnymalC CRL doesn't learn at depth=4, Stage B is the next step: does
  Ant-in-IsaacLab *still* learn the same as Ant-in-Brax? If yes, AnymalC has a
  real task-level issue. If no, the adapter has a subtle bug.

Estimated effort for B: 3–5 days (env port is the bulk; obs-layout matching to
Brax spec is the finicky part).

## Pointers for the implementer

- `dep/scaling-crl/envs/ant.py` — reference obs/reward/termination impls.
- `source/isaaclab_tasks/isaaclab_tasks/core/multi_task/position_env_cfg.py` —
  closest IsaacLab analog to copy structurally.
- ``isaaclab.sensors.ContactSensorCfg`` for the "unhealthy termination" check.
- Keep ``observations.enable_corruption=False`` during reproduction runs —
  Brax envs don't add observation noise, and noise would invalidate the
  cross-stack comparison.
