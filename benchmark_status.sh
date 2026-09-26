#!/usr/bin/env bash
# Print status of today's isaac-lab-* Osmo workflows, grouped by state.
#
# Usage:
#   ./benchmark_status.sh              # today
#   ./benchmark_status.sh 2026-04-27   # specific date (YYYY-MM-DD)
#   ./benchmark_status.sh -v           # also show error tail for FAILED jobs

set -uo pipefail

VERBOSE=""
if [[ "${1:-}" == "-v" || "${1:-}" == "--verbose" ]]; then
  VERBOSE="1"
  shift
fi

DATE="${1:-$(date +%Y-%m-%d)}"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Querying Osmo for isaac-lab-* workflows submitted on or after $DATE..."
osmo workflow list --name isaac-lab --submitted-after "$DATE" \
  --count 200 --order desc --format-type json 2>/dev/null \
  | jq -r '.workflows[] | "\(.name)\t\(.status)"' > "$TMP/status.tsv"

n=$(wc -l <"$TMP/status.tsv")
echo "Found $n workflows. Resolving task names in parallel..."

cut -f1 "$TMP/status.tsv" | xargs -P 12 -I {} bash -c '
  task=$(osmo workflow spec {} 2>/dev/null | grep -oE -- "--task=[^ ]+" | head -1 | sed "s/--task=//")
  echo -e "{}\t${task:-UNKNOWN}"
' > "$TMP/tasks.tsv"

join -t $'\t' <(sort "$TMP/status.tsv") <(sort "$TMP/tasks.tsv") > "$TMP/combined.tsv"

echo
echo "=== Status counts ==="
awk -F'\t' '{print $2}' "$TMP/combined.tsv" | sort | uniq -c | sort -rn

echo
echo "=== Tasks by status ==="
for status in $(awk -F'\t' '{print $2}' "$TMP/combined.tsv" | sort -u); do
  count=$(awk -F'\t' -v s="$status" '$2==s' "$TMP/combined.tsv" | wc -l)
  echo "── $status ($count) ─────────────────────────────────────"
  awk -F'\t' -v s="$status" '$2==s {printf "  %-50s %s\n", $3, $1}' "$TMP/combined.tsv"
  if [[ -n "$VERBOSE" && "$status" == FAILED* ]]; then
    while IFS=$'\t' read -r wf st task; do
      [[ "$st" == "$status" ]] || continue
      echo "    --- $wf ($task) error tail ---"
      osmo workflow logs "$wf" 2>&1 \
        | grep -E "Error|Exception|Traceback|ModuleNotFound" \
        | head -3 \
        | sed 's/^/      /'
    done <"$TMP/combined.tsv"
  fi
done
