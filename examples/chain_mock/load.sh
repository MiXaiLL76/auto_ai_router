#!/usr/bin/env bash
# Drives the gateway across the demo models so you can watch, live on /vhealth,
# the per-tier priority cascade move traffic: within one primary group by SWRR
# weight, then group -> group as each tier's local cumulative cap fills, then
# regional AIR credentials getting banned by fail2ban once the sub-routers
# start answering 429, then recovery back up the tiers after the windows clear.
# See README.md for the full picture.
set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8080}"
MASTER_KEY="${MASTER_KEY:-sk-gateway}"
DURATION="${DURATION:-120}"          # seconds
SLEEP="${SLEEP:-0.15}"               # delay between requests (≈ 1/SLEEP req/s)
# Override to focus load, e.g. MODELS="chat-smart" ./load.sh
MODELS_STR="${MODELS:-chat-smart chat-fast chat-reason embed-v1}"
read -r -a MODELS <<< "$MODELS_STR"

echo "hammering $GATEWAY_URL for ${DURATION}s @ ~$(awk "BEGIN{printf \"%.0f\", 1/$SLEEP}")/s across: ${MODELS[*]}"
echo "watch it live:  $GATEWAY_URL/vhealth   (regions: :8081 :8082 :8083 /vhealth)"
echo

declare -A ok fail
end=$((SECONDS + DURATION))
i=0
while [ "$SECONDS" -lt "$end" ]; do
  model="${MODELS[$((i % ${#MODELS[@]}))]}"
  i=$((i + 1))
  status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    -X POST "$GATEWAY_URL/v1/chat/completions" \
    -H "Authorization: Bearer $MASTER_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}")
  if [ "$status" = "200" ]; then ok[$model]=$(( ${ok[$model]:-0} + 1 )); else fail[$model]=$(( ${fail[$model]:-0} + 1 )); fi
  printf '%(%H:%M:%S)T  %-12s -> %s\n' -1 "$model" "$status"
  sleep "$SLEEP"
done

echo
echo "── summary ──"
for m in "${MODELS[@]}"; do
  printf '  %-12s  ok=%-4s  non-200=%-4s\n' "$m" "${ok[$m]:-0}" "${fail[$m]:-0}"
done
