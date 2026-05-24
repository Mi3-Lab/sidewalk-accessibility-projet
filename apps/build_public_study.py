#!/usr/bin/env python3
"""Build a small static bundle for public study deployment."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "accessibility-study"
OUT = ROOT / "public-study"
STUDY = OUT / "study"
ASSETS = STUDY / "assets" / "images"


def rewrite_image_path(src: str) -> str:
    name = Path(src).name
    return f"assets/images/{name}"


def iter_trial_images(trials: dict):
    for trial in trials.get("imageTrials", []):
        yield trial["src"]
    for trial in trials.get("routeTrials", []):
        for route_key in ("routeA", "routeB"):
            route = trial.get(route_key, {})
            if "images" in route:
                for item in route["images"]:
                    yield item["src"]
            elif "src" in route:
                yield route["src"]


def iter_html_images(html: str):
    pattern = r'\.\./\.\./data/generalization/images/pittsburgh/[^"]+'
    yield from re.findall(pattern, html)


def rewrite_trials(trials: dict) -> dict:
    rewritten = json.loads(json.dumps(trials))
    for trial in rewritten.get("imageTrials", []):
        trial["src"] = rewrite_image_path(trial["src"])
    for trial in rewritten.get("routeTrials", []):
        for route_key in ("routeA", "routeB"):
            route = trial.get(route_key, {})
            if "images" in route:
                for item in route["images"]:
                    item["src"] = rewrite_image_path(item["src"])
            elif "src" in route:
                route["src"] = rewrite_image_path(route["src"])
    return rewritten


def rewrite_html(html: str) -> str:
    return re.sub(
        r'(\.\./\.\./data/generalization/images/pittsburgh/)([^"]+)',
        r"assets/images/\2",
        html,
    )


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (STUDY / "data").mkdir(parents=True)
    ASSETS.mkdir(parents=True)

    trials = json.loads((APP / "data" / "trials.json").read_text())

    for filename in ("styles.css", "app.js"):
        shutil.copy2(APP / filename, STUDY / filename)

    source_html = (APP / "index.html").read_text()
    html = rewrite_html(source_html)
    (STUDY / "index.html").write_text(html)

    (STUDY / "data" / "trials.json").write_text(
        json.dumps(rewrite_trials(trials), indent=2) + "\n"
    )

    seen = set()
    for src in [*iter_trial_images(trials), *iter_html_images(source_html)]:
        source = (APP / src).resolve()
        if source in seen:
            continue
        seen.add(source)
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, ASSETS / source.name)

    (OUT / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0; url=study/">'
        '<link rel="canonical" href="study/">'
        '<title>Sidewalk Accessibility Study</title>'
    )
    (OUT / "netlify.toml").write_text(
        "[build]\n"
        '  publish = "."\n\n'
        "[[redirects]]\n"
        '  from = "/*"\n'
        '  to = "/study/index.html"\n'
        "  status = 200\n"
    )

    print(f"Built {OUT}")
    print(f"Copied {len(seen)} unique image assets")


if __name__ == "__main__":
    main()
