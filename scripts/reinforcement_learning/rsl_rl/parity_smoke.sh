#!/usr/bin/env bash
# Single-shot parity validation for the Meta-World port.
#
# Usage::
#
#     ./scripts/reinforcement_learning/rsl_rl/parity_smoke.sh
#
# Steps (in order):
# 1. ``check_weights`` — confirms every Isaac-Metaworld-* env has the dt-fix
#    applied (raw cfg weights × 100). Doesn't boot the simulator.
# 2. ``parity_reward_audit`` — boots one MT3 task at a time and feeds the
#    env's runtime state into MW's pure-Python ``*_v2_reward`` functions.
#    Reports |Δr| per step. Should be ≤ 0.001 for reach / push / pick-place.
# 3. ``probe_push_dynamics`` — boots Isaac-Metaworld-Push-Sawyer-v0 and runs
#    a 200-step deterministic action sequence. Reports cube travel.
#    Should be ≥ 100 mm with ``k_val = 100`` (≪ 5 mm with the old k_val=25).
#
# All three steps are quick (≤ 60 s wall) — no RL training involved.

set -e

cd "$(dirname "$0")/../../.."

echo "============================================================"
echo "Step 1 — check_weights (no sim, pure Python)"
echo "============================================================"
./isaaclab.sh -p source/isaaclab_contrib/isaaclab_contrib/tasks/manipulation/metaworld/utils/parity/check_weights.py 2>&1 | tail -8

echo
echo "============================================================"
echo "Step 2 — reward audit (MT3 tasks, identical-state comparison)"
echo "============================================================"
for task in Reach Push Pick-Place; do
  echo "--- $task ---"
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/parity_reward_audit.py \
      --task "Isaac-Metaworld-${task}-Sawyer-v0" --num_envs 4 --rollout_steps 10 2>&1 \
      | grep -E "^\\s*[0-9]+\\s+[+-]|mean |max " | tail -3
done

echo
echo "============================================================"
echo "Step 3 — push dynamics probe"
echo "============================================================"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/probe_push_dynamics.py \
    --task Isaac-Metaworld-Push-Sawyer-v0 --num_envs 4 --steps 100 2>&1 \
    | grep -E "DIAGNOSIS|SUCCESS|moved" | tail -3
