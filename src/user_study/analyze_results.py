#!/usr/bin/env python3
"""
Analyze user study submissions from results/user_study/submissions/*.json

Outputs:
  results/user_study/analysis/summary.json  — headline numbers
  results/user_study/analysis/calibration.csv — per-image model vs. user agreement
  results/user_study/analysis/routing.csv     — per-trial route preference
  results/user_study/analysis/report.txt      — human-readable summary
"""
from __future__ import annotations
import json, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, binom

SUBS_DIR   = Path("results/user_study/submissions")
TRIALS_FILE = Path("apps/accessibility-study/data/trials.json")
OUT_DIR    = Path("results/user_study/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AIDS_MAP = {
    "manual_wheelchair":   "manual_wheelchair",
    "motorized_wheelchair":"motorized_wheelchair",
    "mobility_scooter":    "mobility_scooter",
    "walker":              "walker",
    "walking_cane":        "walking_cane",
}

# ── Load trials metadata ──────────────────────────────────────────────────
with open(TRIALS_FILE) as f:
    trials_meta = json.load(f)

image_meta = {tr["id"]: tr for tr in trials_meta["imageTrials"]}
route_meta = {rt["id"]: rt for rt in trials_meta["routeTrials"]}

# ── Load all submissions ──────────────────────────────────────────────────
SKIP_SESSIONS = {"test-session-001.json", "preflight.json", "test-validation-001.json"}

subs = []
for path in sorted(SUBS_DIR.glob("*.json")):
    if path.name in SKIP_SESSIONS:
        continue
    with open(path) as f:
        subs.append(json.load(f))

print(f"Loaded {len(subs)} submissions")
if not subs:
    print("No submissions yet — run the study first.")
    raise SystemExit(0)

# ── Build flat dataframes ─────────────────────────────────────────────────
img_rows, route_rows = [], []

for sub in subs:
    participant = sub.get("participant", {})
    aid = AIDS_MAP.get(participant.get("primary_aid", ""), None)
    name = participant.get("participant_name") or participant.get("session_id", "?")

    for rec in sub.get("image_trials", []):
        if rec.get("skipped"):
            continue
        iid = rec.get("image_id")
        meta = image_meta.get(iid, {})
        per_aid_p = meta.get("per_aid_p_yes", {})
        model_p = per_aid_p.get(aid, meta.get("model_p_yes")) if aid else meta.get("model_p_yes")
        img_rows.append({
            "session_id":    participant.get("session_id"),
            "participant":   name,
            "aid":           aid,
            "image_id":      iid,
            "response":      rec["response"],
            "yes":           1 if rec["response"] == "yes" else 0,
            "no":            1 if rec["response"] == "no" else 0,
            "unsure":        1 if rec["response"] == "unsure" else 0,
            "model_p_yes":   model_p,
            "ps_class":      meta.get("ps_class"),
            "response_ms":   rec.get("response_time_ms"),
        })

    for rec in sub.get("route_trials", []):
        if rec.get("skipped"):
            continue
        pair_id = rec.get("route_pair_id")
        meta = route_meta.get(pair_id, {})
        a_type = rec.get("route_a_type") or (meta.get("routeA", {}).get("type"))
        b_type = rec.get("route_b_type") or (meta.get("routeB", {}).get("type"))
        sel = rec.get("selected_route")
        preferred_accessible = None
        if sel == "route_a" and a_type == "accessible":
            preferred_accessible = True
        elif sel == "route_b" and b_type == "accessible":
            preferred_accessible = True
        elif sel in ("route_a", "route_b"):
            preferred_accessible = False
        route_rows.append({
            "session_id":           participant.get("session_id"),
            "participant":          name,
            "aid":                  aid,
            "pair_id":              pair_id,
            "ors_pair_idx":         meta.get("ors_pair_idx"),
            "accessible_delta":     meta.get("accessible_route_delta"),
            "selected":             sel,
            "preferred_accessible": preferred_accessible,
            "response_ms":          rec.get("response_time_ms"),
        })

img_df   = pd.DataFrame(img_rows)
route_df = pd.DataFrame(route_rows)
print(f"Image responses: {len(img_df)}  |  Route responses: {len(route_df)}")

# ── 1. Participant summary ─────────────────────────────────────────────────
participants = []
for sub in subs:
    p = sub.get("participant", {})
    u = sub.get("usability", {}) or {}
    n_img = sum(1 for r in sub.get("image_trials",[]) if not r.get("skipped"))
    n_rt  = sum(1 for r in sub.get("route_trials",[]) if not r.get("skipped"))
    participants.append({
        "name":       p.get("participant_name") or p.get("session_id","?"),
        "aid":        p.get("primary_aid"),
        "n_image":    n_img,
        "n_route":    n_rt,
        "ease":       u.get("ease_score"),
        "info":       u.get("information_score"),
        "reuse":      u.get("reuse_score"),
        "free_text":  u.get("free_text"),
    })
part_df = pd.DataFrame(participants)

# ── 2. Model calibration: p_yes vs observed yes-rate per image ─────────────
calib_rows = []
if not img_df.empty:
    for iid, grp in img_df.groupby("image_id"):
        for aid_val, agrp in grp.groupby("aid"):
            n = len(agrp)
            yes_rate = agrp["yes"].mean()
            model_p = agrp["model_p_yes"].iloc[0]
            calib_rows.append({
                "image_id":  iid,
                "aid":       aid_val,
                "n":         n,
                "yes_rate":  yes_rate,
                "model_p_yes": model_p,
                "ps_class":  agrp["ps_class"].iloc[0],
            })
calib_df = pd.DataFrame(calib_rows)

# Spearman correlation (only images with ≥3 responses)
corr_rows = []
if not calib_df.empty:
    for aid_val, adf in calib_df[calib_df["n"] >= 3].groupby("aid"):
        if len(adf) < 5:
            continue
        r, p = spearmanr(adf["model_p_yes"], adf["yes_rate"])
        corr_rows.append({"aid": aid_val, "n_images": len(adf), "spearman_r": round(r,3), "p_value": round(p,4)})
corr_df = pd.DataFrame(corr_rows)

# ── 3. Route preference ────────────────────────────────────────────────────
route_summary = {}
if not route_df.empty:
    valid = route_df[route_df["preferred_accessible"].notna()].copy()
    n_total = len(valid)
    n_prefer_acc = int(valid["preferred_accessible"].sum())
    pct = n_prefer_acc / n_total if n_total else 0
    # Wilson 95% CI
    from math import sqrt
    z = 1.96
    lo = (pct + z**2/(2*n_total) - z*sqrt(pct*(1-pct)/n_total + z**2/(4*n_total**2))) / (1 + z**2/n_total) if n_total else 0
    hi = (pct + z**2/(2*n_total) + z*sqrt(pct*(1-pct)/n_total + z**2/(4*n_total**2))) / (1 + z**2/n_total) if n_total else 1
    route_summary = {
        "n_responses": n_total,
        "n_prefer_accessible": n_prefer_acc,
        "pct_prefer_accessible": round(pct, 3),
        "wilson_95ci_lo": round(lo, 3),
        "wilson_95ci_hi": round(hi, 3),
    }

# ── 4. Usability ───────────────────────────────────────────────────────────
usability = {}
if not part_df.empty:
    for col in ["ease","info","reuse"]:
        vals = part_df[col].dropna()
        if len(vals):
            usability[col] = {"mean": round(vals.mean(),2), "n": len(vals)}

# ── Save ───────────────────────────────────────────────────────────────────
summary = {
    "n_participants": len(subs),
    "n_image_responses": len(img_df),
    "n_route_responses": len(route_df),
    "aids_represented": sorted(img_df["aid"].dropna().unique().tolist()) if not img_df.empty else [],
    "calibration_correlations": corr_df.to_dict("records") if not corr_df.empty else [],
    "route_preference": route_summary,
    "usability": usability,
    "per_participant": part_df.to_dict("records"),
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

if not calib_df.empty:
    calib_df.to_csv(OUT_DIR / "calibration.csv", index=False)
if not route_df.empty:
    route_df.to_csv(OUT_DIR / "routing.csv", index=False)

# ── Human-readable report ──────────────────────────────────────────────────
lines = [
    "USER STUDY ANALYSIS REPORT",
    "=" * 50,
    f"Participants: {len(subs)}",
    f"Image responses: {len(img_df)}",
    f"Route responses: {len(route_df)}",
    "",
]
if not part_df.empty:
    lines.append("PARTICIPANTS:")
    for _, row in part_df.iterrows():
        lines.append(f"  {row['name']:20s} {row['aid']:25s} img={row['n_image']} rt={row['n_route']}")
    lines.append("")

if not corr_df.empty:
    lines.append("MODEL CALIBRATION (Spearman r, model p_yes vs user yes-rate):")
    for _, row in corr_df.iterrows():
        stars = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        lines.append(f"  {row['aid']:25s} r={row['spearman_r']:+.3f}  p={row['p_value']:.4f} {stars}  (n={row['n_images']} images)")
    lines.append("")

if route_summary:
    rs = route_summary
    lines.append("ROUTE PREFERENCE:")
    lines.append(f"  {rs['n_prefer_accessible']}/{rs['n_responses']} ({100*rs['pct_prefer_accessible']:.1f}%) prefer accessible route")
    lines.append(f"  Wilson 95% CI: [{100*rs['wilson_95ci_lo']:.1f}%, {100*rs['wilson_95ci_hi']:.1f}%]")
    lines.append("")

if usability:
    lines.append("USABILITY (1–5 scale):")
    for k, v in usability.items():
        lines.append(f"  {k}: {v['mean']:.2f}  (n={v['n']})")

report = "\n".join(lines)
print(report)
with open(OUT_DIR / "report.txt", "w") as f:
    f.write(report)
print(f"\nSaved to {OUT_DIR}/")
