#!/usr/bin/env bash
# Start the dark Streamlit demo + a public Cloudflare quick tunnel.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pkill -f 'cloudflared tunnel --url http://127.0.0.1:7860' 2>/dev/null || true
if ! curl -sf -o /dev/null --max-time 3 http://127.0.0.1:7860/; then
  screen -S sevcap-demo -X quit 2>/dev/null || true
  screen -dmS sevcap-demo bash -lc "cd '$ROOT' && exec .venv/bin/streamlit run demo/app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true"
  for _ in $(seq 1 20); do
    curl -sf -o /dev/null --max-time 2 http://127.0.0.1:7860/ && break
    sleep 1
  done
fi

screen -S sevcap-cf -X quit 2>/dev/null || true
: > /tmp/sevcap-cf.log
screen -dmS sevcap-cf bash -lc 'exec cloudflared tunnel --url http://127.0.0.1:7860 2>&1 | tee /tmp/sevcap-cf.log'

URL=""
for _ in $(seq 1 30); do
  URL=$(rg -o 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' /tmp/sevcap-cf.log 2>/dev/null | head -1 || true)
  if [[ -n "$URL" ]]; then
    if curl -sf -o /dev/null --max-time 15 "$URL/"; then
      echo "$URL" | tee /tmp/sevcap-public-url.txt
      exit 0
    fi
  fi
  sleep 1
done
echo "Tunnel created but not reachable yet. Latest log:" >&2
tail -20 /tmp/sevcap-cf.log >&2
[[ -n "$URL" ]] && echo "$URL"
exit 1
