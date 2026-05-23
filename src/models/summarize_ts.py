#!/usr/bin/env python3
"""
Summarize temperature-scaling Hard-CE results and compare to Soft-KL and Hard-CE.

Generates:
  - Console table for quick inspection
  - LaTeX table to paste into manuscript/appendix

Usage:
    python src/models/summarize_ts.py
"""
import json
from pathlib import Path

ENCODERS_ORDER = [
    ("clip-vit-b32",   "CLIP B/32"),
    ("clip-vit-b16",   "CLIP B/16"),
    ("clip-vit-l14",   "CLIP L/14"),
    ("dinov2-base",    "DINOv2-base"),
    ("dinov2-large",   "DINOv2-large"),
    ("siglip2-base",   "SigLIP2-base"),
    ("siglip2-so400m", "SigLIP2-SO400M"),
    ("vit-b16-sup",    "ViT-B/16-sup"),
]

def mean_across_aids(d: dict, key: str) -> float:
    vals = [v[key] for v in d["aids"].values() if key in v]
    return sum(vals) / len(vals) if vals else float("nan")

def load_encoder(base_dir: str, enc: str) -> dict | None:
    p = Path(base_dir) / enc / "cv_results.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    rows = []
    for enc_key, enc_name in ENCODERS_ORDER:
        ts_data = load_encoder("results/cv/hard_ce_ts", enc_key)
        sk_data = load_encoder("results/cv/soft_kl",    enc_key)
        hce_data = load_encoder("results/cv/hard_ce",   enc_key)

        if ts_data is None:
            print(f"[MISSING] {enc_name}")
            continue

        raw  = mean_across_aids(ts_data,  "brier_soft_raw_mean")
        ts   = mean_across_aids(ts_data,  "brier_soft_ts_mean")
        T    = mean_across_aids(ts_data,  "temperature_mean")
        sk   = mean_across_aids(sk_data,  "brier_soft_mean") if sk_data else float("nan")
        hce  = mean_across_aids(hce_data, "brier_soft_mean") if hce_data else float("nan")

        rows.append({
            "name":  enc_name,
            "sk":    sk,
            "hce":   hce,
            "ts":    ts,
            "raw":   raw,
            "T":     T,
            "ratio_sk_ts": ts / sk if sk > 0 else float("nan"),
        })
        print(
            f"{enc_name:<16} | Soft-KL={sk:.3f} | Hard-CE={hce:.3f} | "
            f"Hard-CE-TS={ts:.3f} (T={T:.2f}) | ratio={ts/sk:.1f}x"
        )

    if not rows:
        return

    mean_sk  = sum(r["sk"]  for r in rows) / len(rows)
    mean_hce = sum(r["hce"] for r in rows) / len(rows)
    mean_ts  = sum(r["ts"]  for r in rows) / len(rows)
    mean_T   = sum(r["T"]   for r in rows) / len(rows)
    print(f"\n{'Mean':<16} | Soft-KL={mean_sk:.3f} | Hard-CE={mean_hce:.3f} | Hard-CE-TS={mean_ts:.3f} (T={mean_T:.2f})")
    print(f"Reduction Hard-CE → TS: {100*(1-mean_ts/mean_hce):.0f}%")
    print(f"Remaining gap TS vs Soft-KL: {mean_ts/mean_sk:.1f}x")

    # LaTeX table
    print("\n\n% ─── LaTeX Table ───────────────────────────────────────────────")
    print(r"""
\begin{table}[!ht]
\caption{\textbf{Temperature-scaled Hard-CE vs.\ Hard-CE and Soft-KL (soft Brier $\downarrow$).}
  For each fold we hold out $20\%$ of training panoramas as a calibration split, minimise NLL over a scalar temperature $T$, and apply the scaled probe to the test fold. Temperature scaling reduces Hard-CE Brier by ${\approx}30\%$ on average, but Soft-KL remains $>2\times$ better across all encoders, confirming that the gain from soft-label training is structural rather than a consequence of overconfidence alone.}
\label{tab:supp_temp_scaling}
\centering\scriptsize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Encoder} & \textbf{Soft-KL} & \textbf{Hard-CE} & \textbf{Hard-CE+TS} & \textbf{Mean $T$} \\
\midrule""")
    for r in rows:
        best = min(r["sk"], r["ts"])
        sk_str  = f"\\textbf{{{r['sk']:.3f}}}" if r["sk"] == best else f"{r['sk']:.3f}"
        ts_str  = f"\\textbf{{{r['ts']:.3f}}}" if r["ts"] == best else f"{r['ts']:.3f}"
        print(f"{r['name']:<16} & {sk_str} & {r['hce']:.3f} & {ts_str} & {r['T']:.2f} \\\\")
    print(r"""\midrule""")
    print(f"\\textit{{Mean}}        & \\textbf{{{mean_sk:.3f}}} & {mean_hce:.3f} & {mean_ts:.3f} & {mean_T:.2f} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


if __name__ == "__main__":
    main()
