# Accessibility Study App

Static MVP for collecting low-burden accessibility judgments from mobility-aid users.

## Run On The University Server

From the repository root:

```bash
./run_user_study_app.sh start
```

First check from inside the server:

```text
http://127.0.0.1:8080/study/
```

If you are opening the app from your laptop, use SSH port forwarding. In a terminal on your laptop, run:

```bash
ssh -N -L 8080:127.0.0.1:8080 <your-user>@<your-university-login-host>
```

Then open this in your laptop browser:

```text
http://127.0.0.1:8080/study/
```

The direct server URL, for example `http://<server-hostname>:8080/study/`, may hang even on VPN if the university firewall blocks inbound ports. The SSH tunnel is the reliable path for testing.

If you use VS Code Remote SSH, open the `Ports` panel, forward port `8080`, and open the forwarded local URL. The short `/study/` path redirects to the app.

The launcher uses `tmux` when available so the app keeps running after your SSH command finishes. Check it with:

```bash
./run_user_study_app.sh status
```

Stop the server:

```bash
./run_user_study_app.sh stop
```

## What It Does

- Collects a short mobility profile.
- Shows image passability trials.
- Shows route preference trials.
- Autosaves progress in the browser.
- Exports JSON and CSV.
- Runs without a backend for pilot testing.

## Pilot Use

Use the static app for Phase 0 and Phase 1. For the main study, add a backend endpoint that receives the same exported JSON schema.

The current image paths reference `paper/images/` so the app should be served from the repository root.
