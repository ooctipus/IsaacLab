#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper that delegates to cluster/lib.py.
# All logic lives in Python for readability and testability.
#
# Usage:
#   ./submit.sh [-s|--submit] <script> [key=val ...]
#   ./submit.sh [-p|--pbt]    <script> [key=val ...]
#   ./submit.sh [-c|--cancel] <prefix-N> <count>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mode="submit"
dry=""
while [[ "${1:-}" == -* ]]; do
  case "$1" in
    -p|--pbt)     mode="pbt";    shift ;;
    -c|--cancel)  mode="cancel"; shift ;;
    -s|--submit)  mode="submit"; shift ;;
    -d|--dry)     dry="1";       shift ;;
    *) break ;;
  esac
done

DRY_RUN="$dry" exec python3 "$SCRIPT_DIR/cluster/lib.py" "$mode" "$@"
