#!/usr/bin/env bash
# Submit all rsl_rl-trainable environments that log Metrics/success_rate to Osmo
# for cross-task benchmarking. All jobs share a single wandb project so the
# Metrics/success_rate cards from every task land on one dashboard.
#
# Usage:
#   ./benchmark.sh                    # submit every group
#   ./benchmark.sh direct             # submit one group only
#   ./benchmark.sh loco-flat
#   ./benchmark.sh loco-rough
#   ./benchmark.sh manager
#   ./benchmark.sh -d [group]         # dry run (forwards to submit.sh -d)
#
# Edit the variables at the top to change pool / resources / wandb project.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Submission settings ─────────────────────────────────────────────────────
POOL="isaac-dex-l40s-02"
IMAGE="lab"
NUM_CPU=8
NUM_GPU=1
NUM_NODE=1
MEMORY=64
STORAGE=64
WANDB_PROJECT="lab_environment_benchmark"
TRAIN_SCRIPT="scripts/reinforcement_learning/rsl_rl/train.py"

# ─── Task groups (one task ID per line for easy editing; joined with commas) ──

# Direct envs (11)
DIRECT_TASKS=(
  Isaac-Cartpole-Direct-v0
  Isaac-Velocity-Flat-Anymal-C-Direct-v0
  Isaac-Velocity-Rough-Anymal-C-Direct-v0
  Isaac-Quadcopter-Direct-v0
  Isaac-Franka-Cabinet-Direct-v0
  Isaac-Humanoid-Direct-v0
  Isaac-Ant-Direct-v0
  Isaac-Repose-Cube-Allegro-Direct-v0
  Isaac-Repose-Cube-Shadow-Direct-v0
  Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0
  Isaac-Repose-Cube-Shadow-Vision-Direct-v0
)

# Manager-based locomotion velocity – Flat (11)
LOCO_FLAT_TASKS=(
  Isaac-Velocity-Flat-Unitree-A1-v0
  Isaac-Velocity-Flat-Anymal-B-v0
  Isaac-Velocity-Flat-Anymal-C-v0
  Isaac-Velocity-Flat-Anymal-D-v0
  Isaac-Velocity-Flat-Cassie-v0
  Isaac-Velocity-Flat-Digit-v0
  Isaac-Velocity-Flat-G1-v0
  Isaac-Velocity-Flat-Unitree-Go1-v0
  Isaac-Velocity-Flat-Unitree-Go2-v0
  Isaac-Velocity-Flat-H1-v0
  Isaac-Velocity-Flat-Spot-v0
)

# Manager-based locomotion velocity – Rough (10; Spot has no rough)
LOCO_ROUGH_TASKS=(
  Isaac-Velocity-Rough-Unitree-A1-v0
  Isaac-Velocity-Rough-Anymal-B-v0
  Isaac-Velocity-Rough-Anymal-C-v0
  Isaac-Velocity-Rough-Anymal-D-v0
  Isaac-Velocity-Rough-Cassie-v0
  Isaac-Velocity-Rough-Digit-v0
  Isaac-Velocity-Rough-G1-v0
  Isaac-Velocity-Rough-Unitree-Go1-v0
  Isaac-Velocity-Rough-Unitree-Go2-v0
  Isaac-Velocity-Rough-H1-v0
)

# Manager-based classic + manipulation + navigation (16)
MANAGER_TASKS=(
  Isaac-Cartpole-v0
  Isaac-Humanoid-v0
  Isaac-Ant-v0
#   Isaac-Open-Drawer-Franka-v0
#   Isaac-Open-Drawer-OpenArm-v0
#   Isaac-Lift-Cube-Franka-v0
#   Isaac-Lift-Cube-OpenArm-v0
#   Isaac-Reach-Franka-v0
#   Isaac-Reach-Franka-OSC-v0
#   Isaac-Reach-UR10-v0
#   Isaac-Reach-OpenArm-v0
#   Isaac-Repose-Cube-Allegro-v0
#   Isaac-Repose-Cube-Allegro-NoVelObs-v0
#   Isaac-Dexsuite-Kuka-Allegro-Reorient-v0
#   Isaac-Dexsuite-Kuka-Allegro-Lift-v0
#   Isaac-Navigation-Flat-Anymal-C-v0
)

# ─── Helpers ─────────────────────────────────────────────────────────────────
DRY=""
if [[ "${1:-}" == "-d" || "${1:-}" == "--dry" ]]; then
  DRY="-d"
  shift
fi
GROUP="${1:-all}"

join_tasks() {
  local IFS=,
  echo "$*"
}

submit_group() {
  local name="$1"
  shift
  local tasks
  tasks="$(join_tasks "$@")"
  echo
  echo "=========================================="
  echo "  Submitting group: $name  ($# tasks)"
  echo "=========================================="
  ./submit.sh ${DRY} -s "$TRAIN_SCRIPT" \
    --task="$tasks" \
    pool="$POOL" \
    image="$IMAGE" \
    num_cpu="$NUM_CPU" \
    num_gpu="$NUM_GPU" \
    num_node="$NUM_NODE" \
    memory="$MEMORY" \
    storage="$STORAGE" \
    --logger=wandb \
    --log_project_name="$WANDB_PROJECT"
}

# ─── Dispatch ────────────────────────────────────────────────────────────────
case "$GROUP" in
  direct)      submit_group direct          "${DIRECT_TASKS[@]}" ;;
  loco-flat)   submit_group loco-flat       "${LOCO_FLAT_TASKS[@]}" ;;
  loco-rough)  submit_group loco-rough      "${LOCO_ROUGH_TASKS[@]}" ;;
  manager)     submit_group manager         "${MANAGER_TASKS[@]}" ;;
  all)
    submit_group direct          "${DIRECT_TASKS[@]}"
    submit_group loco-flat       "${LOCO_FLAT_TASKS[@]}"
    submit_group loco-rough      "${LOCO_ROUGH_TASKS[@]}"
    submit_group manager         "${MANAGER_TASKS[@]}"
    ;;
  *)
    echo "Unknown group: '$GROUP'"
    echo "Valid groups: all | direct | loco-flat | loco-rough | manager"
    exit 1
    ;;
esac
