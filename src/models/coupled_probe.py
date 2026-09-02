#!/usr/bin/env python3
"""
Two changes to how the probe is trained, both motivated by properties of this
dataset that the current Soft-KL setup ignores.

**1. The target is an estimate, not a fact.** Each (image, aid) distribution is
estimated from 141-240 votes, giving a binomial standard error of ~0.035 on
p_yes (95% CI ±0.068). KL against the empirical distribution treats it as exact
and then re-weights samples by an ad-hoc entropy term, 1 - H/Hmax. The
statistically motivated alternative is to model the *counts*:

    multinomial      -sum_c n_c log p_c
                     Dropping constants this is N x CE(empirical, p): identical
                     in shape to the current loss but weighted by the actual
                     number of votes, i.e. by the precision of each target,
                     rather than by a hand-chosen entropy function.

    dirichlet_mult   The compound Dirichlet-multinomial marginal likelihood. The
                     model emits a concentration alpha rather than a point on the
                     simplex, so it can express uncertainty *about the
                     distribution itself* (epistemic) separately from the spread
                     of the distribution (aleatoric). Overdispersion relative to
                     a multinomial is exactly what finite-vote noise looks like.

**2. The five aids are not five separate problems.** Their p_yes values correlate
at 0.886 on average (0.824-0.946). Training five independent probes on 52 points
each throws that away. The `lowrank` head factorises W_a = U V_a with U shared
across aids, so all 260 rows inform the shared subspace while each aid keeps its
own read-out.

Every variant is evaluated identically: soft Brier against held-out human vote
distributions, panorama-level folds, repeated over seeds. Crucially all variants
share the *same* fold partition within a seed, so the comparison is paired.

Usage:
    python src/models/coupled_probe.py \
        --tallies_json data/processed/tallies_firebase.json \
        --images_dir   data/images/sidewalk-images \
        --encoder      dinov2-large \
        --n_folds 5 --n_seeds 10 \
        --output_dir   results/coupled/dinov2-large
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from crossval import (
    _features_for,
    brier_score_soft,
    build_feature_cache,
    expected_calibration_error,
)
from train import (
    AIDS,
    CLASS3,
    ENCODERS,
    LABEL_MAP,
    PROBE_EPOCHS,
    PROBE_LR,
    PROBE_WD,
    RANDOM_STATE,
    SOFT_COLS,
    find_image_path,
    load_encoder,
    set_seed,
)

COUNT_COLS = ["no", "unsure", "yes"]        # raw vote counts in tallies_firebase.json
LOSSES = ["soft_kl", "hard_ce", "multinomial", "dirichlet_mult"]
HEADS  = ["independent", "lowrank", "lowrank_peraid"]


# ── Heads ────────────────────────────────────────────────────────────────────

class IndependentHead(nn.Module):
    """One linear probe per aid — the current setup, restated as a joint module
    so that every variant trains through the same loop."""

    def __init__(self, feature_dim: int, n_aids: int, n_classes: int = 3):
        super().__init__()
        self.probes = nn.ModuleList(
            [nn.Linear(feature_dim, n_classes) for _ in range(n_aids)]
        )

    def forward(self, x: torch.Tensor, aid_idx: torch.Tensor) -> torch.Tensor:
        out = torch.stack([p(x) for p in self.probes], dim=1)   # (N, n_aids, C)
        return out.gather(
            1, aid_idx.view(-1, 1, 1).expand(-1, 1, out.size(-1))
        ).squeeze(1)


class LowRankHead(nn.Module):
    """W_a = U V_a: a shared subspace U plus a per-aid read-out V_a.

    All 260 rows train U, so the 0.886 correlation between aids is used instead
    of discarded, while each aid keeps enough freedom to differ.
    """

    def __init__(self, feature_dim: int, n_aids: int, rank: int, n_classes: int = 3):
        super().__init__()
        self.U = nn.Linear(feature_dim, rank, bias=False)
        self.V = nn.ModuleList([nn.Linear(rank, n_classes) for _ in range(n_aids)])

    def forward(self, x: torch.Tensor, aid_idx: torch.Tensor) -> torch.Tensor:
        z = self.U(x)
        out = torch.stack([v(z) for v in self.V], dim=1)
        return out.gather(
            1, aid_idx.view(-1, 1, 1).expand(-1, 1, out.size(-1))
        ).squeeze(1)


class LowRankPerAidHead(nn.Module):
    """The control for LowRankHead: same low-rank bottleneck, but each aid gets
    its own U_a, so nothing is shared between aids.

    Without this, a win for `lowrank` is ambiguous — the shared head both pools
    information across aids *and* has far fewer parameters per aid, so the gain
    could be plain capacity control on 52 points rather than cross-aid transfer.
    This head has the same capacity per aid and no sharing, which separates the
    two explanations.
    """

    def __init__(self, feature_dim: int, n_aids: int, rank: int, n_classes: int = 3):
        super().__init__()
        self.U = nn.ModuleList(
            [nn.Linear(feature_dim, rank, bias=False) for _ in range(n_aids)]
        )
        self.V = nn.ModuleList([nn.Linear(rank, n_classes) for _ in range(n_aids)])

    def forward(self, x: torch.Tensor, aid_idx: torch.Tensor) -> torch.Tensor:
        out = torch.stack([v(u(x)) for u, v in zip(self.U, self.V)], dim=1)
        return out.gather(
            1, aid_idx.view(-1, 1, 1).expand(-1, 1, out.size(-1))
        ).squeeze(1)


# ── Losses ───────────────────────────────────────────────────────────────────

def loss_soft_kl(logits, target_soft, counts, weights):
    """Current method: KL to the empirical distribution, entropy-weighted."""
    log_p = F.log_softmax(logits, dim=-1)
    kl = F.kl_div(log_p, target_soft, reduction="none").sum(dim=-1)
    return (kl * weights).mean()


def loss_hard_ce(logits, target_soft, counts, weights):
    """The paper's Hard-CE baseline: cross-entropy on the argmax label, weighted
    by the same entropy term. Included here so the headline Soft-KL vs Hard-CE
    ratio can be re-measured under the low-rank head — if the bottleneck helps
    Hard-CE just as much, the ratio the paper leads with is unchanged; if it
    helps Hard-CE more, the central claim shrinks."""
    hard = target_soft.argmax(dim=-1)
    ce = F.cross_entropy(logits, hard, reduction="none")
    return (ce * weights).mean()


def loss_multinomial(logits, target_soft, counts, weights):
    """-sum_c n_c log p_c, normalised by the mean count so the learning rate
    stays comparable. Each target is weighted by how many votes back it."""
    log_p = F.log_softmax(logits, dim=-1)
    nll = -(counts * log_p).sum(dim=-1)
    return (nll / counts.sum(dim=-1).mean()).mean()


def loss_dirichlet_mult(logits, target_soft, counts, weights, log_conc):
    """Dirichlet-multinomial marginal likelihood of the observed counts.

    alpha = concentration * softmax(logits). Large concentration collapses to the
    multinomial; small concentration admits overdispersion, which is what a
    finite number of votes actually produces.

        log P(n | alpha) = logGamma(A) - logGamma(N + A)
                           + sum_c [ logGamma(n_c + a_c) - logGamma(a_c) ]
    """
    p = F.softmax(logits, dim=-1)
    alpha = p * log_conc.exp()
    A = alpha.sum(dim=-1)
    N = counts.sum(dim=-1)
    ll = (
        torch.lgamma(A) - torch.lgamma(N + A)
        + (torch.lgamma(counts + alpha) - torch.lgamma(alpha)).sum(dim=-1)
    )
    return (-ll / N).mean()


# ── Training ─────────────────────────────────────────────────────────────────

def train_variant(
    X: np.ndarray, aid_idx: np.ndarray, target_soft: np.ndarray,
    counts: np.ndarray, weights: np.ndarray,
    feature_dim: int, n_aids: int, loss_name: str, head_name: str,
    device: torch.device, seed: int, rank: int, epochs: int = PROBE_EPOCHS,
    weight_decay: float = PROBE_WD,
):
    set_seed(seed)
    if head_name == "independent":
        head = IndependentHead(feature_dim, n_aids)
    elif head_name == "lowrank":
        head = LowRankHead(feature_dim, n_aids, rank)
    else:
        head = LowRankPerAidHead(feature_dim, n_aids, rank)
    head = head.to(device)

    params = list(head.parameters())
    # One global concentration, learned. Initialised near the typical vote count
    # so the model starts close to the plain multinomial and can loosen from there.
    log_conc = torch.tensor(float(np.log(200.0)), device=device, requires_grad=True)
    if loss_name == "dirichlet_mult":
        params.append(log_conc)

    opt = optim.Adam(params, lr=PROBE_LR, weight_decay=weight_decay)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    a_t = torch.tensor(aid_idx, dtype=torch.long, device=device)
    y_t = torch.tensor(target_soft, dtype=torch.float32, device=device)
    n_t = torch.tensor(counts, dtype=torch.float32, device=device)
    w_t = torch.tensor(weights, dtype=torch.float32, device=device)

    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = head(X_t, a_t)
        if loss_name == "soft_kl":
            loss = loss_soft_kl(logits, y_t, n_t, w_t)
        elif loss_name == "hard_ce":
            loss = loss_hard_ce(logits, y_t, n_t, w_t)
        elif loss_name == "multinomial":
            loss = loss_multinomial(logits, y_t, n_t, w_t)
        else:
            loss = loss_dirichlet_mult(logits, y_t, n_t, w_t, log_conc)
        loss.backward()
        opt.step()

    head.eval()
    return head, float(log_conc.detach().exp())


@torch.no_grad()
def predict(head, X, aid_idx, device):
    logits = head(
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(aid_idx, dtype=torch.long, device=device),
    )
    return torch.softmax(logits, dim=-1).cpu().numpy()


# ── Experiment ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tallies_json", required=True)
    p.add_argument("--images_dir",   required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--encoder", default="dinov2-large", choices=list(ENCODERS))
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed",    type=int, default=RANDOM_STATE)
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--rank",    type=int, default=16,
                   help="Shared subspace size for the lowrank head.")
    p.add_argument("--weight_decay", type=float, default=PROBE_WD,
                   help="L2 on the probe. The pipeline default is 1e-4, which is "
                        "very weak for a d=1024 map fit to 52 points; sweeping it "
                        "tests whether the low-rank gain is just regularisation.")
    args = p.parse_args()

    seeds = [args.seed + i for i in range(args.n_seeds)]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tallies = pd.read_json(args.tallies_json)
    tallies["path"] = tallies["ImageID"].apply(lambda x: find_image_path(x, args.images_dir))
    tallies = tallies[tallies["path"].notna()].copy()
    tallies["label_int"] = tallies["argmax_label"].map(LABEL_MAP)
    tallies["aid_idx"]   = tallies["MobilityAid"].map({a: i for i, a in enumerate(AIDS)})

    missing = [c for c in COUNT_COLS if c not in tallies.columns]
    if missing:
        raise SystemExit(f"tallies file lacks raw vote counts {missing}; "
                         "the count-based losses need them.")

    print(f"Encoder : {args.encoder}")
    print(f"Rows    : {len(tallies)}  ({tallies['ImageID'].nunique()} panoramas × {len(AIDS)} aids)")
    print(f"Votes   : {tallies[COUNT_COLS].sum(axis=1).min():.0f}"
          f"–{tallies[COUNT_COLS].sum(axis=1).max():.0f} per row")
    print(f"Seeds   : {seeds[0]}..{seeds[-1]}   folds {args.n_folds}   rank {args.rank}\n")

    model, processor, enc_device, enc_type = load_encoder(args.encoder)
    feat_cache = build_feature_cache(model, processor, enc_device,
                                     tallies["path"].tolist(), enc_type)
    print(f"Encoded {len(feat_cache)} images\n")

    panos = np.array(sorted(tallies["ImageID"].unique()))
    records: list[dict] = []

    for seed in seeds:
        # One panorama-level split shared by every variant, so comparisons within
        # a seed are paired rather than confounded by different folds.
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=seed)

        for fold_idx, (tr_i, te_i) in enumerate(kf.split(panos)):
            tr_panos, te_panos = set(panos[tr_i]), set(panos[te_i])
            tr = tallies[tallies["ImageID"].isin(tr_panos)]
            te = tallies[tallies["ImageID"].isin(te_panos)]
            if tr.empty or te.empty:
                continue

            X_tr_raw = _features_for(tr["path"].tolist(), feat_cache)
            X_te_raw = _features_for(te["path"].tolist(), feat_cache)
            scaler   = StandardScaler(with_mean=False).fit(X_tr_raw)
            X_tr, X_te = scaler.transform(X_tr_raw), scaler.transform(X_te_raw)

            y_tr = tr[SOFT_COLS].values.astype(np.float32)
            n_tr = tr[COUNT_COLS].values.astype(np.float32)
            w_tr = tr["sample_weight"].values.astype(np.float32)
            a_tr = tr["aid_idx"].values

            y_te = te[SOFT_COLS].values.astype(np.float32)
            a_te = te["aid_idx"].values
            l_te = te["label_int"].values

            for loss_name, head_name in product(LOSSES, HEADS):
                head, conc = train_variant(
                    X_tr, a_tr, y_tr, n_tr, w_tr,
                    X_tr.shape[1], len(AIDS), loss_name, head_name,
                    device, seed, args.rank, weight_decay=args.weight_decay,
                )
                prob = predict(head, X_te, a_te, device)
                records.append({
                    "seed": seed, "fold": fold_idx,
                    "loss": loss_name, "head": head_name,
                    "brier_soft": brier_score_soft(prob, y_te),
                    "macro_f1":   float(f1_score(l_te, prob.argmax(1),
                                                 average="macro", zero_division=0)),
                    "ece":        expected_calibration_error(prob, l_te),
                    "concentration": conc if loss_name == "dirichlet_mult" else None,
                })

        done = [r for r in records if r["seed"] == seed]
        best = min(done, key=lambda r: r["brier_soft"])
        print(f"  seed {seed}: best this seed = {best['loss']}/{best['head']} "
              f"brier={best['brier_soft']:.4f}")

    # ── Aggregate, paired within seed ────────────────────────────────────────
    df = pd.DataFrame(records)
    summary = {}
    for (loss_name, head_name), grp in df.groupby(["loss", "head"]):
        per_seed = grp.groupby("seed")[["brier_soft", "macro_f1", "ece"]].mean()
        summary[f"{loss_name}/{head_name}"] = {
            "brier_soft_mean": float(per_seed["brier_soft"].mean()),
            "brier_soft_std":  float(per_seed["brier_soft"].std(ddof=0)),
            "macro_f1_mean":   float(per_seed["macro_f1"].mean()),
            "ece_mean":        float(per_seed["ece"].mean()),
            "brier_per_seed":  per_seed["brier_soft"].tolist(),
        }

    base_key = "soft_kl/independent"
    base = np.array(summary[base_key]["brier_per_seed"])
    for k, v in summary.items():
        d = np.array(v["brier_per_seed"]) - base
        v["delta_vs_baseline_mean"] = float(d.mean())
        v["delta_vs_baseline_std"]  = float(d.std(ddof=0))
        v["wins_vs_baseline"]       = int((d < 0).sum())

    with open(out_dir / "coupled_results.json", "w") as f:
        json.dump({"config": vars(args), "summary": summary,
                   "records": records}, f, indent=2)

    print(f"\n── Paired against {base_key}, {len(seeds)} seeds ──")
    print(f"{'variant':<28}{'Brier':>16}{'Δ vs base':>18}{'wins':>7}{'F1':>8}")
    print("-" * 78)
    for k in sorted(summary, key=lambda k: summary[k]["brier_soft_mean"]):
        v = summary[k]
        print(f"{k:<28}{v['brier_soft_mean']:>9.4f} ±{v['brier_soft_std']:.4f}"
              f"{v['delta_vs_baseline_mean']:>+11.4f} ±{v['delta_vs_baseline_std']:.4f}"
              f"{v['wins_vs_baseline']:>5}/{len(seeds)}{v['macro_f1_mean']:>8.3f}")
    print(f"\nSaved → {out_dir / 'coupled_results.json'}")


if __name__ == "__main__":
    main()
