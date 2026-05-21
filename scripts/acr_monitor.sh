#!/usr/bin/env bash
# Poll an ACR job every 5 minutes until it leaves pending/allocated/running.
# Usage: ./scripts/acr_monitor.sh <job_id_prefix> [interval_sec]
set -euo pipefail

JOB_PREFIX="${1:?Pass job id or prefix (e.g. e1b706ce6c24)}"
INTERVAL="${2:-300}"
USER="${MARIGOLD_DGX_USER:-elad.e}"
LOG="${MARIGOLD_ISILON_ROOT:-/isilon/Automotive/RnD/elad.e/Dev/research/marigold_depth_mechanics}/results/acr_monitor_${JOB_PREFIX}.log"

mkdir -p "$(dirname "$LOG")"
ACTIVE="pending allocated running"

_log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$LOG"
}

_match_line() {
  acr jobs --user "$USER" 2>/dev/null | grep -E "${JOB_PREFIX}" | head -1 || true
}

_log "monitor start job=${JOB_PREFIX} interval=${INTERVAL}s log=${LOG}"
sleep "$INTERVAL"

while true; do
  line="$(_match_line)"
  if [[ -z "$line" ]]; then
    _log "job not found in acr jobs (may have aged out)"
    break
  fi
  status="$(echo "$line" | awk '{print $3}')"
  _log "status=${status} | ${line}"
  if [[ ! " ${ACTIVE} " =~ " ${status} " ]]; then
    _log "terminal status=${status} — stopping monitor"
    tail -15 "${MARIGOLD_ISILON_ROOT:-/isilon/Automotive/RnD/elad.e/Dev/research/marigold_depth_mechanics}/results/finish_acr.log" 2>/dev/null | tee -a "$LOG" || true
    break
  fi
  sleep "$INTERVAL"
done
