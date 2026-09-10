# FeatherPGS / Newton physics profiling harness

Measures where an Isaac Lab environment step spends time, separating the Isaac Lab host pipeline from the
Newton physics CUDA graph, and drills into solver kernels with Nsight Systems and Nsight Compute.

All scripts run from the repository root with the uv environment. Outputs default to `outputs/fpgs_profile/`
(override with `OUT_DIR`); select the GPU with `GPU=<index>`.

```bash
# 1. FPS plus host-phase breakdown and model/solver metadata (no profiler)
uv run python scripts/benchmarks/fpgs_profile/run_profiled.py --task Isaac-Lift-Franka --physics feather_pgs \
    --device cuda:1 --num-envs 16384 --output outputs/fpgs_profile/lift.json

# 2. Nsight Systems: per-kernel table, per-phase GPU attribution, graph span/idle, ordered kernel chain
GPU=1 bash scripts/benchmarks/fpgs_profile/nsys_run.sh lift_fpgs feather_pgs Isaac-Lift-Franka
GPU=1 bash scripts/benchmarks/fpgs_profile/nsys_run.sh lift_mjwarp newton_mjwarp Isaac-Lift-Franka
# A/B a solver field without editing presets:
GPU=1 bash scripts/benchmarks/fpgs_profile/nsys_run.sh ant_mf feather_pgs Isaac-Ant --solver-attr pgs_mode=matrix_free

# 3. Nsight Compute: registers, occupancy, waves, L1/L2 hit rates, DRAM/L2 bytes, stall reasons per kernel
GPU=0 bash scripts/benchmarks/fpgs_profile/ncu_run.sh lift_fpgs feather_pgs Isaac-Lift-Franka cuda_kernel_forward 560

# 4. Essential-work floors versus measured solver time per substep
uv run python scripts/benchmarks/fpgs_profile/roofline.py outputs/fpgs_profile/lift_fpgs.json \
    outputs/fpgs_profile/lift_fpgs_analysis.json

# 5. Scorecard across all runs in a directory
uv run python scripts/benchmarks/fpgs_profile/summarize.py outputs/fpgs_profile
```

Notes:

- `run_profiled.py` wraps the env.step phases and the Newton graph launch in NVTX ranges and brackets the traced
  steps with `cudaProfilerStart/Stop`; nsys and ncu attach with `--capture-range=cudaProfilerApi` /
  `--profile-from-start off`. Timed repeats run before the traced window so FPS is unaffected by tracing.
- `analyze_nsys.py` attributes device work to phases through the launch API correlation id, so graph kernels are
  attributed to `physics_graph` and eager torch kernels to the manager phase that launched them. `--sequence`
  prints the ordered kernel chain of one env step with the gap before each kernel.
- Stage classification in `analyze_nsys.py` is a regex table (`STAGES`); extend it when kernels are renamed.
- `ncu_run.sh` uses `--clock-control base`, so ncu durations are longer than nsys durations; use ratios, not
  absolute times, from the ncu table.
- The host floor of a task is measured by running `run_profiled.py` with a small `--num-envs` (for example 256).

## Compare revisions on two GPUs

Install the pinned environment once with `uv sync --locked`, then provide local Newton checkout roots:

```bash
uv run --no-sync python scripts/benchmarks/fpgs_profile/compare_gpus.py \
    --newton baseline=/absolute/path/to/newton-baseline \
    --newton optimized=/absolute/path/to/newton-optimized \
    --gpus 0 1 --repeats 3
```

Each task/revision runs concurrently on the selected GPUs, with both processes finishing before the next batch.
Rounds alternate revision order. Defaults match the handoff: AnymalD and Allegro, 16,384 environments, seed 0,
200 warmup steps, one 40-step timing batch, and three profiled steps. Use repeated `--task anymald` / `--task allegro`
to select tasks, or override the counts with `--num-envs`, `--warmup-steps`, `--steps`, and `--profile-steps`.

The runner requires idle compute devices (desktop graphics is allowed), clears inherited experimental flags,
and selects each Newton checkout through `PYTHONPATH`. Child processes use `UV_NO_SYNC=1`. Keep the checkouts
unchanged while runs are active. A fresh output directory contains the exact source pins and dirty-source hashes,
GPU UUIDs/names/driver, commands and recipes in `manifest.json`, plus per-GPU medians, ranges, finite-state checks,
and contact counts in `summary.json`. Hardware results remain separate. `--output-dir` must name a new directory;
the default creates one under `outputs/fpgs_profile/`. Failed runs report their log path and exit nonzero.
Custom output directories inside a source checkout must be git-ignored to keep source hashes stable.

Analyze the saved captures without using a GPU:

```bash
uv run --no-sync python scripts/benchmarks/fpgs_profile/analyze_structure.py \
    outputs/fpgs_profile/compare_gpus_EXAMPLE --output outputs/fpgs_profile/structure.json
```

This independently verifies graph spans and busy unions against SQLite, separates fused preparation/response from
external stages, and reports kernel-duration distributions and optimistic scope-level speedup ceilings. Summed
stage durations can overlap, so these ceilings are estimates, not guaranteed critical-path savings. The output path
must be new. See the [structural study](../../../reports/fpgs/STRUCTURAL_STUDY_20260910.md) for interpretation.

## Check sweep parity on live task inputs

`check_sweep_parity.py` runs the environment with a baseline Newton checkout and
replays selected parallel-sweep launches using the candidate's solver factory.
After warmup, it copies every launch buffer before the baseline runs, preserves
overlapping array aliases, and compares candidate velocities, impulses, row
metadata, and response buffers bitwise. Candidate outputs remain in scratch
storage. This isolates each kernel comparison from divergence between independent
trajectories; it does not compare complete solver implementations.

Use the installed Isaac Lab environment and two compatible Newton checkouts with
the same sweep-kernel signature. The following small runs use the handoff settings:

```bash
export NEWTON_BASELINE=/absolute/path/to/newton-baseline
export NEWTON_CANDIDATE=/absolute/path/to/newton-optimized
export CUDA_VISIBLE_DEVICES=0
mkdir -p outputs/fpgs_profile
PARITY_DIR=$(mktemp -d outputs/fpgs_profile/sweep_parity_XXXXXX)
for option in ${!FEATHER_PGS_@} ${!NEWTON_NARROW_PHASE_@}; do unset "$option"; done
export FEATHER_PGS_GROUP_LANES=16 FEATHER_PGS_ROWS_MASKED=1 NEWTON_NARROW_PHASE_THREADS_X=4

FEATHER_PGS_INK=1 FEATHER_PGS_MF_EXACT_ROWSUM=1 FEATHER_PGS_WORLD_ROWS=1 \
    uv run --no-sync python scripts/benchmarks/fpgs_profile/check_sweep_parity.py \
    --baseline "$NEWTON_BASELINE" --candidate "$NEWTON_CANDIDATE" \
    --parity-output "$PARITY_DIR/anymald_parity.json" -- \
    --task Isaac-Velocity-Flat-AnymalD --physics feather_pgs --device cuda:0 \
    --num-envs 512 --seed 0 --warmup-steps 200 --steps 8 --repeats 1 --profile-steps 0 \
    --no-graph --no-nvtx --solver-attr grouped_dynamics=True \
    --solver-attr mf_gs_parallel_rows=48 --solver-attr mf_gs_parallel_matrix_free=True \
    --solver-attr lazy_kinematics=True --output "$PARITY_DIR/anymald_run.json"

FEATHER_PGS_INK=1 FEATHER_PGS_TIER_BLOCKS=16384 \
    uv run --no-sync python scripts/benchmarks/fpgs_profile/check_sweep_parity.py \
    --baseline "$NEWTON_BASELINE" --candidate "$NEWTON_CANDIDATE" \
    --parity-output "$PARITY_DIR/allegro_parity.json" -- \
    --task Isaac-Reorient-Cube-Allegro --physics feather_pgs --device cuda:0 \
    --num-envs 512 --seed 0 --warmup-steps 200 --steps 8 --repeats 1 --profile-steps 0 \
    --no-graph --no-nvtx --solver-attr grouped_dynamics=True \
    --solver-attr mf_gs_parallel_rows=128 --solver-attr mf_gs_parallel_matrix_free=True \
    --solver-attr lazy_kinematics=True --output "$PARITY_DIR/allegro_run.json"
```

The default checks 16 launches per tier after warmup; `--checks-per-tier` and
`--check-every` adjust coverage. The report records solver source hashes, flags,
active row tiers, actual buffer updates, bitwise comparisons, and finite state
checks before and after the sampled window. It refuses an existing report path
and exits nonzero for a mismatch, nonfinite state, or absent active multiwarp
coverage. The shared Warp cache is used by default; `--kernel-cache-dir` selects
an isolated cache when needed. Keep both checkouts unchanged during a run.

These checks synchronize and copy device buffers, so their timing output is not
a performance measurement. Measure performance with `compare_gpus.py` separately.
