#!/usr/bin/env bash
# Hammers the main router across the 3 demo models so you can watch, live on
# /vhealth, credentials get eaten through their token budget, get banned by
# fail2ban once the mock starts answering 429, and the priority cascade move
# traffic to the next router. See README.md for the full picture.
set -euo pipefail

MAIN_URL="${MAIN_URL:-http://localhost:8080}"
MASTER_KEY="${MASTER_KEY:-sk-main}"
DURATION="${DURATION:-90}"
MODELS=(model-a model-b model-c)

echo "hammering $MAIN_URL for ${DURATION}s across: ${MODELS[*]}"
echo "watch it live at $MAIN_URL/vhealth"
echo

end=$((SECONDS + DURATION))
i=0
while [ "$SECONDS" -lt "$end" ]; do
  model="${MODELS[$((i % ${#MODELS[@]}))]}"
  i=$((i + 1))
  status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    -X POST "$MAIN_URL/v1/chat/completions" \
    -H "Authorization: Bearer $MASTER_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}")
  printf '%(%H:%M:%S)T  %-10s -> %s\n' -1 "$model" "$status"
  sleep 0.2
done
echo "done"
