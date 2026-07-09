#!/bin/bash
cd /Users/saksham56/Developer/sev-cap
set -a
source .env
set +a
exec .venv/bin/python demo/app.py
