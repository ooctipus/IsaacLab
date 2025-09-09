#!/usr/bin/env bash
set -euo pipefail

function usage() {
  cat <<EOF
Usage:
  $0 [--pbt|-p] <script> [key=val ...]
  $0 [--cancel|-c] <prefix-start> <num_to_cancel>
  $0 [--submit|-s]  <script> [cluster_key=val ...] [fixed_key=val ...] sweep_key=val1,val2,...
EOF
  exit 1
}

# Prefer these spec paths; we’ll select one based on num_node
SPEC_SINGLE="/home/zhengyuz/workflow_specs/single_node.yaml"
SPEC_MULTI="/home/zhengyuz/workflow_specs/multi_node.yaml"

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

# (validation happens after we know the call form)

# Build the --set cluster string from the associative array
function build_cluster_str() {
  local keys=(image num_gpu num_cpu memory platform dataset num_node storage master_port)
  local str=""
  for k in "${keys[@]}"; do
    str+=" $k=${cluster[$k]}"
  done
  echo "${str# }"
}

# Decide which spec to use based on num_node (1 => single, >1 => multi)
function choose_spec_by_nodes() {
  local nodes="${cluster[num_node]:-1}"
  if (( nodes > 1 )); then spec="$SPEC_MULTI"; else spec="$SPEC_SINGLE"; fi
  [[ -f "$spec" ]] || { echo "Spec file not found: $spec" >&2; exit 1; }
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

# For PBT/SUBMIT: first positional is always <script>
if [[ "$mode" == "pbt" || "$mode" == "submit" ]]; then
  (( $# >= 1 )) || usage
  spec=""                     # will be chosen by num_node later
  script="$1"; shift 1
  [[ -f "$script" ]] || { echo "Script not found: $script" >&2; exit 1; }
fi

# default pool to platform settings
declare -A pool_to_platform=(
  [isaac-dev-h100-01]="dgx-h100"
  [isaac-dev-l40-03]="ovx-l40"
  [isaac-srl-l40-04]="ovx-l40"
  [isaac-dex-l40s-04]="ovx-l40s"
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
  [master_port]=29400
)

# -------------------------------------------------------------------
# PBT Submission
# -------------------------------------------------------------------
if [[ "$mode" == "pbt" ]]; then
  parse_cluster_args "$@"
  args=( "${parsed_args[@]}" )

  declare -A kv
  declare -a cli_flags=()
  declare -a hydra_dels=()
  pool_explicit=""
  platform_explicit=""

  for arg in "${args[@]}"; do
    if [[ $arg == --* ]]; then
      handle_cli_flag "$arg"
    elif [[ $arg == "~"* ]]; then
      # bare Hydra delete operator (must be quoted by caller to avoid shell tilde expansion)
      hydra_dels+=( "$arg" )
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
  # Pick the right spec now that cluster overrides are parsed
  choose_spec_by_nodes
  cluster_str=$(build_cluster_str)

  echo "⏳ Submitting PBT: populations=${kv[num_populations]}"

  for (( idx=0; idx<kv[num_populations]; idx++ )); do
    set_pairs=(
      "agent.pbt.enabled=True"
      "agent.pbt.num_policies=${kv[num_populations]}"
      "agent.pbt.policy_idx=${idx}"
      "agent.pbt.workspace=${kv[num_populations]}agents_${date_time}"
      "agent.pbt.directory=/mnt/amlfs-05/shared/workspace/octi"
      "--wandb-name=${idx}_${kv[num_populations]}"
      "--seed=-1"
    )
    # forward any other user overrides (e.g. agent.params.config.*)
    for k in "${!kv[@]}"; do
      if [[ "$k" == "num_populations" || "$k" == "args" ]]; then
        continue
      fi
      set_pairs+=( "$k=${kv[$k]}" )
    done

    all_set="args=${set_pairs[*]} ${hydra_dels[*]} ${cli_flags[*]}"
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
  declare -a hydra_dels=()
  pool_explicit=""
  platform_explicit=""

  for arg in "${args[@]}"; do
    # keep special bool flags normalized
    if [[ $arg == --video || $arg == --enable_cameras || $arg == --video=* || $arg == --enable_cameras=* ]]; then
      handle_cli_flag "$arg"
      continue
    fi

    # Hydra delete operator (bare "~path.to.key")
    if [[ $arg == "~"* ]]; then
      hydra_dels+=( "$arg" )
      continue
    fi

    # parse key=val (this now also catches --key=val like --task=...)
    if [[ $arg == *=* ]]; then
      key=${arg%%=*}; val=${arg#*=}
      if [[ "$key" == "pool" ]]; then
        pool_explicit="$val"; continue
      elif [[ "$key" == "platform" ]]; then
        platform_explicit="$val"
        cluster[platform]="$val"
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
    elif [[ $arg == --* ]]; then
      # bare flags (no '=') pass through
      cli_flags+=( "$arg" )
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

  # Pick the right spec now that cluster overrides are parsed
  choose_spec_by_nodes
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
    all_set="args=$fixed_str $combo ${hydra_dels[*]} ${cli_flags[*]}"
    cmd=( osmo workflow submit "$spec" --pool "$resolved_pool" --set script="$script" $cluster_str "$all_set" )
    echo "+ ${cmd[@]}"
    "${cmd[@]}"
  done
  exit
fi

usage
