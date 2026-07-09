#!/bin/bash
cd /Users/saksham56/Developer/sev-cap
set -a; source .env; set +a
export PATH="/Users/saksham56/Developer/sev-cap/.venv/bin:$PATH"
while true; do
  echo "$(date) starting run_server" >> demo/out/watchdog.log
  .venv/bin/python demo/run_server.py >> demo/out/app.log 2>&1
  rc=$?
  echo "$(date) run_server exited rc=$rc" >> demo/out/watchdog.log
  sleep 2
done
