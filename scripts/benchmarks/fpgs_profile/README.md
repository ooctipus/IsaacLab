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
