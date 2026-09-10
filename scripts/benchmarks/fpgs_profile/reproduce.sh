#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Reproduce the 2026-09-10 physics-graph comparison from an installed checkout.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
cd "$repo"
export GPU=${GPU:-1}
if [[ -n "${OUT_DIR:-}" ]]; then
    mkdir -p "$OUT_DIR"
else
    mkdir -p "$repo/outputs/fpgs_profile"
    OUT_DIR=$(mktemp -d "$repo/outputs/fpgs_profile/reproduce_20260910_XXXXXX")
fi
export OUT_DIR REPEATS=1 STEPS=40 PROFILE_STEPS=3
for option in ${!FEATHER_PGS_@} ${!NEWTON_NARROW_PHASE_@}; do unset "$option"; done
export FEATHER_PGS_GROUP_LANES=16 FEATHER_PGS_ROWS_MASKED=1 NEWTON_NARROW_PHASE_THREADS_X=4
command -v nsys >/dev/null
command -v uv >/dev/null
printf 'Results: %s\n' "$OUT_DIR"
run() {
    local name=$1 physics=$2 task=$3
    shift 3
    if [[ -n "$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
        echo "GPU $GPU has an active compute process; stop it or choose another GPU." >&2
        exit 1
    fi
    if [[ -e "$OUT_DIR/$name.json" || -e "$OUT_DIR/$name.nsys-rep" || -e "$OUT_DIR/$name.sqlite" ]]; then
        echo "Refusing to overwrite $OUT_DIR/$name; choose a fresh OUT_DIR." >&2
        exit 1
    fi
    printf 'START %s\n' "$name"
    bash "$here/nsys_run.sh" "$name" "$physics" "$task" --num-envs 16384 --seed 0 --warmup-steps 200 "$@"
    rg 'graph_span_us_per_step' "$OUT_DIR/${name}_analysis.txt"
}
FEATHER_PGS_INK=1 FEATHER_PGS_MF_EXACT_ROWSUM=1 FEATHER_PGS_WORLD_ROWS=1 \
    run fpgs_anymald feather_pgs Isaac-Velocity-Flat-AnymalD \
    --solver-attr grouped_dynamics=True --solver-attr mf_gs_parallel_rows=48 \
    --solver-attr mf_gs_parallel_matrix_free=True --solver-attr lazy_kinematics=True
run mjwarp_anymald newton_mjwarp Isaac-Velocity-Flat-AnymalD
FEATHER_PGS_INK=1 FEATHER_PGS_TIER_BLOCKS=16384 \
    run fpgs_allegro feather_pgs Isaac-Reorient-Cube-Allegro \
    --solver-attr grouped_dynamics=True --solver-attr mf_gs_parallel_rows=128 \
    --solver-attr mf_gs_parallel_matrix_free=True --solver-attr lazy_kinematics=True
run mjwarp_allegro newton_mjwarp Isaac-Reorient-Cube-Allegro
run fpgs_g1 feather_pgs Isaac-Velocity-Rough-G1 --solver-attr grouped_dynamics=True
run mjwarp_g1 newton_mjwarp Isaac-Velocity-Rough-G1
run fpgs_kuka feather_pgs Isaac-Lift-KukaAllegro
run mjwarp_kuka newton_mjwarp Isaac-Lift-KukaAllegro
run fpgs_franka feather_pgs Isaac-Lift-Franka
run mjwarp_franka newton_mjwarp Isaac-Lift-Franka
run fpgs_cartpole feather_pgs Isaac-Cartpole
run mjwarp_cartpole newton_mjwarp Isaac-Cartpole
run fpgs_ant feather_pgs Isaac-Ant
run mjwarp_ant newton_mjwarp Isaac-Ant
run fpgs_humanoid feather_pgs Isaac-Humanoid
run mjwarp_humanoid newton_mjwarp Isaac-Humanoid
