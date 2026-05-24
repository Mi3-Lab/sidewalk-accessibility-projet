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
- Submits the final response through Netlify Forms when deployed on Netlify.
- Exports JSON and CSV.
- Runs without a custom backend for pilot testing.

## Public Deployment

Build the deployable Netlify folder:

```bash
python scripts/build_public_study.py
```

Then deploy the generated `public-study/` folder using Netlify Drop or the Netlify CLI. See:

```text
docs/DEPLOY_PUBLIC_STUDY.md
```

The current image paths reference `data/generalization/images/pittsburgh/`, so the app should be served from the repository root during local testing. The build script rewrites paths for public deployment.
