#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <python_script> [args ... --seed=[START,END] ...]"
  exit 1
fi

PYTHON_BIN="${PYTHON:-python}"   # override with: PYTHON=isaac310 ./record_play.sh ...

script="$1"; shift
args=("$@")

# Find the --seed arg and capture the spec
seed_idx=-1
seed_spec=""
for i in "${!args[@]}"; do
  case "${args[i]}" in
    --seed=*)
      seed_spec="${args[i]#--seed=}"
      seed_idx=$i
      break
      ;;
    --seed)
      if (( i+1 < ${#args[@]} )); then
        seed_spec="${args[i+1]}"
        seed_idx=$i
      fi
      break
      ;;
  esac
done

if [[ -z "$seed_spec" ]]; then
  echo "Error: --seed=[START,END] (or a single integer) is required." >&2
  exit 1
fi

# Parse seed range
start=""
end=""
if [[ "$seed_spec" =~ ^\[(\-?[0-9]+),(\-?[0-9]+)\]$ ]]; then
  start="${BASH_REMATCH[1]}"
  end="${BASH_REMATCH[2]}"
elif [[ "$seed_spec" =~ ^\-?[0-9]+$ ]]; then
  start="$seed_spec"
  end="$seed_spec"
else
  echo "Error: seed must be [START,END] or a single integer." >&2
  exit 1
fi

# Remove the original --seed spec from args
if [[ ${args[$seed_idx]:-} == "--seed" ]]; then
  # remove "--seed" and its value
  args=("${args[@]:0:$seed_idx}" "${args[@]:$((seed_idx+2))}")
else
  # remove "--seed=..."
  args=("${args[@]:0:$seed_idx}" "${args[@]:$((seed_idx+1))}")
fi

trap 'echo -e "\nAborted."; exit 130' INT

# Iterate seeds
if (( start <= end )); then
  seq_list=$(seq "$start" "$end")
else
  seq_list=$(seq "$start" -1 "$end")
fi

for s in $seq_list; do
  echo "===== Running seed $s ====="
  "$PYTHON_BIN" "$script" "${args[@]}" --seed="$s"
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "Run failed for seed $s (exit $status). Stopping." >&2
    exit $status
  fi
done

echo "All runs complete."
