#!/usr/bin/env bash
set -euo pipefail

function usage() {
  cat <<EOF
Usage:
  $0 [--pbt|-p] <spec.yaml> <script> [key=val ...]
  $0 [--cancel|-c] <prefix-start> <num_to_cancel>
  $0 [--submit|-s]  <spec.yaml> <script> [cluster_key=val ...] [fixed_key=val ...] sweep_key=val1,val2,...
EOF
  exit 1
}

truthy() { case "${1,,}" in 1|true|yes|on) return 0;; *) return 1;; esac; }
falsy() { case "${1,,}" in 0|false|no|off) return 0;; *) return 1;; esac; }

handle_cli_flag() {
  local arg="$1"
  case "$arg" in
    --video|--enable_cameras)
      cli_flags+=( "$arg" )               # already correct (store_true style)
      ;;
    --video=*|--enable_cameras=*)
      local name="${arg%%=*}"             # e.g., --video
      local val="${arg#*=}"               # e.g., True / False
      if truthy "$val"; then
        cli_flags+=( "$name" )            # normalize to bare flag
      elif falsy "$val"; then
        :                                  # drop flag entirely
      else
        echo "Invalid boolean for $name: $val" >&2; exit 2
      fi
      ;;
    *)
      cli_flags+=( "$arg" )               # pass everything else through
      ;;
  esac
}

# Validate that spec and script exist
function require_spec_and_script() {
  [[ -f "$1" ]] || { echo "Spec file not found: $1" >&2; exit 1; }
  [[ -f "$2" ]] || { echo "Script not found: $2" >&2; exit 1; }
}

# Build the --set cluster string from the associative array
function build_cluster_str() {
  local keys=(image num_gpu num_cpu memory platform dataset num_node storage)
  local str=""
  for k in "${keys[@]}"; do
    str+=" $k=${cluster[$k]}"
  done
  echo "${str# }"
}

# Parse and remove any cluster overrides *in this shell*
parsed_args=()
function parse_cluster_args() {
  parsed_args=()
  for arg in "$@"; do
    if [[ $arg == *=* ]]; then
      key=${arg%%=*}; val=${arg#*=}
      if [[ -v "cluster[$key]" ]]; then
        cluster[$key]="$val"
        continue
      fi
    fi
    parsed_args+=( "$arg" )
  done
}

# entrypoint
(( $# > 0 )) || usage

mode="submit"
case "$1" in
  -p|--pbt)     mode="pbt";    shift ;;
  -c|--cancel)  mode="cancel"; shift ;;
  -s|--submit)  mode="submit"; shift ;;
  *)             mode="submit";       ;;
esac

# CANCEL mode
if [[ "$mode" == "cancel" ]]; then
  (( $# == 2 )) || usage
  pattern="$1"; count="$2"
  prefix="${pattern%-*}"; start="${pattern##*-}"
  for ((i = start; i < start + count; i++)); do
    osmo workflow cancel "${prefix}-${i}"
  done
  exit
fi

# For both PBT and SUBMIT we need at least spec+script
if [[ "$mode" == "pbt" || "$mode" == "submit" ]]; then
  (( $# >= 2 )) || usage
  spec="$1"; script="$2"; shift 2
  require_spec_and_script "$spec" "$script"
fi

# default pool to platform settings
declare -A pool_to_platform=(
  [isaac-dev-h100-01]="dgx-h100"
  [isaac-dev-l40-03]="ovx-l40"
  [isaac-dev-l40s-03]="ovx-l40s"
  [isaac-dex-l40s-02]="ovx-l40s"
  [isaac-dex-l40s-03]="ovx-l40s"
  [isaac-dev-l40s-04]="ovx-l40s"
)

# default cluster settings
declare -A cluster=(
  [image]="factory"
  [num_gpu]=1
  [num_cpu]=2
  [num_node]=1
  [memory]=64
  [storage]=64
  [platform]="dgx-h100"
  [dataset]="isaac-lab-ppo-model"
)

# -------------------------------------------------------------------
# PBT Submission
# -------------------------------------------------------------------
if [[ "$mode" == "pbt" ]]; then
  parse_cluster_args "$@"
  args=( "${parsed_args[@]}" )

  declare -A kv
  declare -a cli_flags=()
  pool_explicit=""
  platform_explicit=""

  for arg in "${args[@]}"; do
    if [[ $arg == --* ]]; then
      handle_cli_flag "$arg"
    elif [[ $arg == *=* ]]; then
      key=${arg%%=*}; val=${arg#*=}
      if [[ "$key" == "pool" ]]; then
        pool_explicit="$val"
        continue
      elif [[ "$key" == "platform" ]]; then
        platform_explicit="$val"
        kv["$key"]="$val"
        cluster[platform]="$val"  # honor platform override
        continue
      fi
      kv["$key"]="$val"
    else
      echo "Invalid argument: $arg" >&2
      usage
    fi
  done


  # Resolve pool → platform if pool was given
  resolved_pool="${pool_explicit:-isaac-dev-h100-01}"
  if [[ -n "$pool_explicit" ]]; then
    if [[ -v "pool_to_platform[$pool_explicit]" ]]; then
      cluster[platform]="${pool_to_platform[$pool_explicit]}"
      resolved_pool="$pool_explicit"
    else
      echo "Error: unknown pool '$pool_explicit'." >&2
      exit 1
    fi
  fi

  : "${kv[num_populations]:?Missing required: num_populations}"

  date_time=$(date +'%Y-%m-%dT%H:%M:%S')
  cluster_str=$(build_cluster_str)

  echo "⏳ Submitting PBT: populations=${kv[num_populations]}"

  for (( idx=0; idx<kv[num_populations]; idx++ )); do
    set_pairs=(
      "agent.pbt.enabled=True"
      "agent.pbt.num_policies=${kv[num_populations]}"
      "agent.pbt.policy_idx=${idx}"
      "agent.pbt.workspace=${kv[num_populations]}agents_${date_time}"
      "agent.hydra.run.dir=/mnt/amlfs/shared/workspace/octi"
      # "agent.pbt.interval_steps=10000000"
      "--wandb-name=${idx}_${kv[num_populations]}"
    )
    # forward any other user overrides (e.g. agent.params.config.*)
    for k in "${!kv[@]}"; do
      if [[ "$k" == "num_populations" || "$k" == "args" ]]; then
        continue
      fi
      set_pairs+=( "$k=${kv[$k]}" )
    done

    all_set="args=${set_pairs[*]} ${cli_flags[*]}"
    cmd=( osmo workflow submit "$spec" --pool "$resolved_pool" --set script="$script" $cluster_str "$all_set" )
    echo "+ ${cmd[@]}"
    "${cmd[@]}"
  done
  exit
fi

# -------------------------------------------------------------------
# Submit
# -------------------------------------------------------------------
if [[ "$mode" == "submit" ]]; then
  parse_cluster_args "$@"
  args=( "${parsed_args[@]}" )

  declare -A fixed sweep_map
  declare -a cli_flags=()
  pool_explicit=""
  platform_explicit=""

  for arg in "${args[@]}"; do
    if [[ $arg == --* ]]; then
      handle_cli_flag "$arg"
    elif [[ $arg == *=* ]]; then
      key=${arg%%=*}; val=${arg#*=}
      if [[ "$key" == "pool" ]]; then
        pool_explicit="$val"
        continue
      elif [[ "$key" == "platform" ]]; then
        platform_explicit="$val"
        cluster[platform]="$val"  # honor platform override
        continue
      fi

      first="${val:0:1}"; last="${val: -1}"
      if [[ "$val" == *,* ]] && ! { [[ "$first" == "[" && "$last" == "]" ]] \
                                || [[ "$first" == "(" && "$last" == ")" ]] \
                                || [[ "$first" == "{" && "$last" == "}" ]]; }; then
        sweep_map["$key"]="$val"
      else
        fixed["$key"]="$val"
      fi
    else
      echo "Invalid argument: $arg" >&2
      usage
    fi
  done

  # Resolve pool → platform if pool was given
  resolved_pool="${pool_explicit:-isaac-dev-h100-01}"
  if [[ -n "$pool_explicit" ]]; then
    if [[ -v "pool_to_platform[$pool_explicit]" ]]; then
      cluster[platform]="${pool_to_platform[$pool_explicit]}"
      resolved_pool="$pool_explicit"
    else
      echo "Error: unknown pool '$pool_explicit'." >&2
      exit 1
    fi
  fi

  cluster_str=$(build_cluster_str)

  # fixed args
  fixed_str=""
  for k in "${!fixed[@]}"; do
    fixed_str+=" $k=${fixed[$k]}"
  done
  fixed_str=${fixed_str# }

  # build all combos
  combos=( "" )
  for key in "${!sweep_map[@]}"; do
    IFS=',' read -r -a vals <<< "${sweep_map[$key]}"
    new=()
    for base in "${combos[@]}"; do
      for v in "${vals[@]}"; do
        [[ -z "$base" ]] \
          && new+=( "$key=$v" ) \
          || new+=( "$base $key=$v" )
      done
    done
    combos=( "${new[@]}" )
  done

  # submit each combo
  for combo in "${combos[@]}"; do
    all_set="args=$fixed_str $combo ${cli_flags[*]}"
    cmd=( osmo workflow submit "$spec" --pool "$resolved_pool" --set script="$script" $cluster_str "$all_set" )
    echo "+ ${cmd[@]}"
    "${cmd[@]}"
  done
  exit
fi

usage
