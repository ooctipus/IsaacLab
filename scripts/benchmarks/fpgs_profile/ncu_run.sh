#!/bin/bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Collect per-kernel Nsight Compute metrics (occupancy, registers, L1/L2/DRAM, stalls) for one task.
# usage: GPU=0 OUT_DIR=/path ncu_run.sh <name> <physics> <task> [kernel-regex] [launch-count] [run_profiled.py args...]
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
S="${OUT_DIR:-$repo/outputs/fpgs_profile}"
NCU="${NCU:-$(command -v ncu || echo /usr/local/cuda/bin/ncu)}"
name=$1; physics=$2; task=$3; regex=${4:-cuda_kernel_forward}; count=${5:-600}; shift 5 2>/dev/null || shift $#
mkdir -p "$S"
cd "$repo"
METRICS=${METRICS:-gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_bytes.sum,l1tex__t_bytes.sum,\
lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,\
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__maximum_warps_per_active_cycle_pct,\
sm__cycles_active.avg.pct_of_peak_sustained_elapsed,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
smsp__inst_executed.sum,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_misc_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_selected_per_issue_active.ratio,\
smsp__average_warp_latency_per_inst_issued.ratio,\
launch__registers_per_thread,launch__grid_size,launch__block_size,launch__waves_per_multiprocessor,\
launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_blocks,launch__occupancy_limit_warps,\
launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__thread_count,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active}
CUDA_VISIBLE_DEVICES=${GPU:-0} "$NCU" --target-processes all --profile-from-start off --graph-profiling node \
  --kernel-name-base function --kernel-name "regex:$regex" --launch-count $count \
  --metrics $METRICS --cache-control all --clock-control base -f -o "$S/$name" \
  uv run python "$here/run_profiled.py" --physics $physics --task $task --device cuda:0 --repeats 1 --steps 5 --warmup-steps "${WARMUP:-50}" --profile-steps 1 --no-nvtx --output "$S/$name.json" "$@" > "$S/$name.log" 2>&1
echo "ncu exit=$?"
"$NCU" --import "$S/$name.ncu-rep" --csv --page raw > "$S/$name.csv" 2>/dev/null
uv run python "$here/analyze_ncu.py" "$S/$name.csv" --json "$S/${name}_ncu.json" > "$S/${name}_ncu.txt"
head -40 "$S/${name}_ncu.txt"
