#!/usr/bin/env python3
"""
Does the soft-label advantage survive fair tuning on a public benchmark?

On our own data the headline "3.2x better calibration from Soft-KL" turned out to
be a property of an untuned hyperparameter: weight decay was left at 1e-4, which
is near-zero regularisation for a linear map fit to few points, and it happens to
be close to the worst possible setting for the Hard-CE baseline (whose error is
dominated by overconfidence that L2 shrinkage fixes). Tuning each loss at its own
optimum takes the gap from 3.2x to about 1.4x.

That is a finding about our paper. Whether it is a finding about the *field*
depends on whether it reproduces on the benchmark the soft-label literature is
built on. CIFAR-10H (Peterson et al., ICCV 2019) provides human label
distributions for the CIFAR-10 test set, ~51 annotations per image.

The two datasets sit in very different regimes, which is itself worth reporting:

    CIFAR-10H  mean normalised entropy 0.067, IQR 0.000-0.084,
               4393/10000 items at total consensus — mostly easy, some ambiguous
    sidewalk   mean normalised entropy 0.847, IQR 0.805-0.923,
               no easy items at all — uniformly contested

So CIFAR-10H has an easy/hard axis to learn and ours does not. If the soft-label
advantage also shrinks under fair tuning there, the effect is not specific to a
degenerate regime.

Protocol is deliberately identical to the sidewalk experiment: frozen encoder,
linear probe, Soft-KL vs Hard-CE, weight-decay sweep, repeated seeds, soft Brier
plus entropy correlation.

Usage:
    python src/models/cifar10h_check.py \
        --counts data/cifar10h/cifar10h-counts.npy \
        --encoder clip-vit-b32 --n_images 4000 --n_seeds 5 \
        --output_dir results/cifar10h
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from crossval import brier_score_soft, entropy_correlation, expected_calibration_error
from train import ENCODERS, PROBE_EPOCHS, PROBE_LR, extract_encoder_features, load_encoder, set_seed

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

WD_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]


def train_probe(X, y_soft, n_classes, device, loss_name, seed, weight_decay):
    set_seed(seed)
    probe = nn.Linear(X.shape[1], n_classes).to(device)
    opt = optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=weight_decay)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_soft, dtype=torch.float32, device=device)
    hard = y_t.argmax(dim=-1)

    probe.train()
    for _ in range(PROBE_EPOCHS):
        opt.zero_grad()
        logits = probe(X_t)
        if loss_name == "soft_kl":
            loss = F.kl_div(F.log_softmax(logits, dim=-1), y_t, reduction="none").sum(-1).mean()
        else:
            loss = F.cross_entropy(logits, hard)
        loss.backward()
        opt.step()
    probe.eval()
    return probe


@torch.no_grad()
def predict(probe, X, device):
    return torch.softmax(
        probe(torch.tensor(X, dtype=torch.float32, device=device)), dim=-1
    ).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--counts", default="data/cifar10h/cifar10h-counts.npy")
    p.add_argument("--cifar_root", default="/tmp/cifar10",
                   help="Where torchvision keeps the CIFAR-10 download. Defaults "
                        "outside the repo: it is ~180 MB of redownloadable data "
                        "and has no business in version control.")
    p.add_argument("--output_dir", default="results/cifar10h")
    p.add_argument("--encoder", default="clip-vit-b32", choices=list(ENCODERS))
    p.add_argument("--n_images", type=int, default=4000,
                   help="Subsample of the 10000 test images, for speed.")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    counts = np.load(args.counts).astype(np.float64)
    soft = counts / counts.sum(axis=1, keepdims=True)

    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root=args.cifar_root, train=False, download=True)
    assert len(ds) == counts.shape[0], f"{len(ds)} images vs {counts.shape[0]} label rows"

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.n_images, len(ds)), replace=False)
    images = [ds[i][0].convert("RGB") for i in idx]
    y_soft = soft[idx].astype(np.float32)

    Hn = -(np.clip(y_soft, 1e-12, 1) * np.log(np.clip(y_soft, 1e-12, 1))).sum(1) / np.log(10)
    print(f"Encoder: {args.encoder}   images: {len(images)}")
    print(f"Human entropy (normalised): mean {Hn.mean():.3f}, "
          f"IQR {np.percentile(Hn,25):.3f}-{np.percentile(Hn,75):.3f}, "
          f"{int((Hn < 0.01).sum())} at consensus\n")

    model, processor, enc_device, enc_type = load_encoder(args.encoder)
    # extract_encoder_features takes file paths, so the images have to hit disk.
    # They go to a real temporary directory that is cleaned up afterwards — an
    # earlier version wrote them under results/ and 4000 PNGs ended up in a commit.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="cifar10h_") as tmpdir:
        paths = []
        for j, im in enumerate(images):
            fp = Path(tmpdir) / f"{j:05d}.png"
            im.save(fp)
            paths.append(str(fp))
        X_all = extract_encoder_features(model, processor, enc_device, paths, enc_type)
    print(f"features: {X_all.shape}\n")

    records = []
    for seed in range(args.seed, args.seed + args.n_seeds):
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=seed)
        for tr, te in kf.split(X_all):
            sc = StandardScaler(with_mean=False).fit(X_all[tr])
            Xtr, Xte = sc.transform(X_all[tr]), sc.transform(X_all[te])
            yte = y_soft[te]
            lte = yte.argmax(1)
            for wd in WD_GRID:
                for loss in ("soft_kl", "hard_ce"):
                    probe = train_probe(Xtr, y_soft[tr], y_soft.shape[1], device, loss, seed, wd)
                    pr = predict(probe, Xte, device)
                    records.append({
                        "seed": seed, "wd": wd, "loss": loss,
                        "brier_soft": brier_score_soft(pr, yte),
                        "macro_f1": float(f1_score(lte, pr.argmax(1), average="macro", zero_division=0)),
                        "ece": expected_calibration_error(pr, lte),
                        "entropy_corr": entropy_correlation(pr, yte),
                    })
        print(f"  seed {seed} done")

    summary = {}
    for wd in WD_GRID:
        for loss in ("soft_kl", "hard_ce"):
            rs = [r for r in records if r["wd"] == wd and r["loss"] == loss]
            summary[f"{loss}@{wd}"] = {
                m: float(np.nanmean([r[m] for r in rs]))
                for m in ("brier_soft", "macro_f1", "ece", "entropy_corr")
            }

    with open(out / f"cifar10h_{args.encoder}.json", "w") as f:
        json.dump({"config": vars(args), "summary": summary, "records": records}, f, indent=2)

    print(f"\n{'weight decay':>13}{'Soft-KL':>11}{'Hard-CE':>11}{'ratio':>9}"
          f"{'H-corr soft':>13}{'H-corr hard':>13}")
    print("-" * 70)
    for wd in WD_GRID:
        s = summary[f"soft_kl@{wd}"]; h = summary[f"hard_ce@{wd}"]
        print(f"{wd:>13.0e}{s['brier_soft']:>11.4f}{h['brier_soft']:>11.4f}"
              f"{h['brier_soft']/s['brier_soft']:>8.2f}x"
              f"{s['entropy_corr']:>13.3f}{h['entropy_corr']:>13.3f}")

    bs = min(WD_GRID, key=lambda w: summary[f"soft_kl@{w}"]["brier_soft"])
    bh = min(WD_GRID, key=lambda w: summary[f"hard_ce@{w}"]["brier_soft"])
    rs = summary[f"soft_kl@{bs}"]["brier_soft"]; rh = summary[f"hard_ce@{bh}"]["brier_soft"]
    print("-" * 70)
    print(f"  best Soft-KL {rs:.4f} at wd={bs:g}   best Hard-CE {rh:.4f} at wd={bh:g}")
    print(f"  ratio at the paper's wd=1e-4 : "
          f"{summary['hard_ce@0.0001']['brier_soft']/summary['soft_kl@0.0001']['brier_soft']:.2f}x")
    print(f"  FAIR ratio, each at its own optimum : {rh/rs:.2f}x")
    print(f"\nSaved → {out / f'cifar10h_{args.encoder}.json'}")


if __name__ == "__main__":
    main()
