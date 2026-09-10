#!/bin/bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Capture one task under Nsight Systems and analyze it.
# usage: GPU=1 OUT_DIR=/path nsys_run.sh <name> <physics> <task> [run_profiled.py args...]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
out="${OUT_DIR:-$repo/outputs/fpgs_profile}"
name=$1; physics=$2; task=$3; shift 3
mkdir -p "$out"
cd "$repo"
CUDA_VISIBLE_DEVICES=${GPU:-0} nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,nvtx \
  --cuda-graph-trace=node --cuda-event-trace=false -f true -o "$out/$name" \
  uv run python "$here/run_profiled.py" --physics "$physics" --task "$task" --device cuda:0 \
  --repeats "${REPEATS:-3}" --steps "${STEPS:-100}" --profile-steps "${PROFILE_STEPS:-10}" --output "$out/$name.json" "$@" \
  > "$out/$name.log" 2>&1
nsys export --type sqlite -f true -o "$out/$name.sqlite" "$out/$name.nsys-rep" > /dev/null 2>&1
uv run python "$here/analyze_nsys.py" "$out/$name.sqlite" --json "$out/${name}_analysis.json" --top 60 --sequence > "$out/${name}_analysis.txt" 2>&1
grep -h "RESULT\|FINAL" "$out/$name.log" | cut -c1-400
