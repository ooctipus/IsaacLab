#!/usr/bin/env bash
# queue_videos.sh
# Usage:
#   ./queue_videos.sh [A-B] <command and args...>
# Example (upper bound exclusive; [0-25] runs seeds 0..24):
#   ./queue_videos.sh [0-25] python scripts/reinforcement_learning/rsl_rl/play.py \
#     --task=Dexsuite-Kuka-Allegro-Shelves-Place-State-v0 \
#     --num_envs=1 env.scene.object=geometry env.commands.object_pose.debug_vis=True \
#     env.episode_length_s=2.5 --headless --video --video_length=600 --enable_cameras \
#     --checkpoint=logs/rsl_rl/dexsuite_kuka_allegro/shelves_collapsed/model_2250.pt

set -u -o pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 [A-B] <command and args...>"
  echo "Example: $0 [0-25] python scripts/.../play.py --task=... --video ..."
  exit 1
fi

range="$1"; shift

# Parse [A-B] or A-B ; runs seeds A..B-1 (upper bound exclusive)
if [[ "$range" =~ ^\[*([0-9]+)-([0-9]+)\]*$ ]]; then
  start="${BASH_REMATCH[1]}"
  end="${BASH_REMATCH[2]}"
else
  echo "Invalid range: '$range' (use like [0-25])" >&2
  exit 1
fi

if (( end <= start )); then
  echo "Range end must be greater than start (got $start-$end)" >&2
  exit 1
fi

# Copy original args and remove any existing seed flags (we'll inject our own)
orig=( "$@" )
clean=()
i=0
while (( i < ${#orig[@]} )); do
  arg="${orig[$i]}"
  case "$arg" in
    --seed=*)          : ;;                    # drop
    seed=*)            : ;;                    # drop (Hydra-style seed=123)
    --seed)            (( i++ )) ;;            # drop this and its value
    *)                 clean+=( "$arg" ) ;;
  esac
  (( i++ ))
done

# Run sequentially for each seed
for (( s=start; s<end; s++ )); do
  echo "==== Running seed $s ===="
  # Append --seed=<s> at the end so it overrides earlier positions
  if ! "${clean[@]}" --seed="$s"; then
    echo "Command failed for seed $s (continuing)" >&2
  fi
done

echo "All queued runs completed."
