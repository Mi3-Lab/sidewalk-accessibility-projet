#!/usr/bin/env python3
"""
Minimal study server: serves static files + accepts POST /submit.
Submissions saved to results/user_study/submissions/<session_id>.json
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).parent          # apps/accessibility-study/
PROJECT_ROOT = ROOT.parent.parent     # project root
SUBMISSIONS_DIR = PROJECT_ROOT / "results" / "user_study" / "submissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


class StudyHandler(SimpleHTTPRequestHandler):
    # Serve static files from ROOT
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def do_POST(self):
        if urlparse(self.path).path != "/submit":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body)
        except Exception as exc:
            self.send_error(400, str(exc))
            return

        session_id = payload.get("participant", {}).get("session_id", f"unknown-{datetime.now(timezone.utc).isoformat()}")
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        outfile = SUBMISSIONS_DIR / f"{safe_id}.json"
        payload["server_received_at"] = datetime.now(timezone.utc).isoformat()
        with open(outfile, "w") as f:
            json.dump(payload, f, indent=2)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "file": outfile.name}).encode())
        print(f"[submit] saved → {outfile.name}", flush=True)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        # Suppress noisy static-file logs; keep submit logs
        if "/submit" in (args[0] if args else ""):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    server = HTTPServer((host, port), StudyHandler)
    print(f"Study server on http://{host}:{port}/  submissions → {SUBMISSIONS_DIR}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
