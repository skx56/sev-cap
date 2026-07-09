"""Launch Gradio without share=True; pair with cloudflared for a public URL."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import importlib.util

spec = importlib.util.spec_from_file_location("sevcap_demo_app", ROOT / "demo" / "app.py")
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

ui = mod.build_ui()
ui.queue(default_concurrency_limit=1)
port = int(os.environ.get("SEVCAP_DEMO_PORT", "7860"))
print(f"Starting Gradio on 0.0.0.0:{port}", flush=True)
# prevent_thread_lock=True so we control the keep-alive ourselves — Gradio 6
# has been observed to return from launch() and let the process exit under nohup.
ui.launch(
    server_name="0.0.0.0",
    server_port=port,
    share=False,
    show_error=True,
    prevent_thread_lock=True,
)
print("Gradio server thread started; keeping process alive", flush=True)
(Path(__file__).parent / "out" / "server.pid").write_text(str(os.getpid()))
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    pass
