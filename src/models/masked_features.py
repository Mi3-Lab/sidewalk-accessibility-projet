#!/usr/bin/env python3
"""
Mask-pooled features: give the probe the sidewalk, not the whole postcard.

**Why.** The probe is fed one globally pooled vector per image, so it sees sky,
buildings, parked cars and vegetation along with the walking surface. Two results
point at that being a problem:

  - Geographic transfer collapses under an encoder swap (all five cities at
    chance), which is what you would expect if the model keys on city-level scene
    appearance rather than surface condition — Zurich does not look like Taipei.
  - Entropy correlation with human disagreement is ~0.05, i.e. the model does not
    know which scenes people argue about, even though the targets say so.

Accessibility is a property of the walking surface. This module pools encoder
patch tokens over the sidewalk region only, using an off-the-shelf Cityscapes
segmenter.

**Why not the repo's own YOLO checkpoint.** `checkpoints/yolo/bestv12.pt` is a
134-byte Git LFS pointer, not the 380 MB model, and git-lfs is not installed —
the segmentation step described in the paper cannot be reproduced from a clone.
A public SegFormer with a native `sidewalk` class is both available and more
defensible in review.

Writes an .npz keyed by image path so cross-validation can consume the features
directly, and so whole-image and mask-pooled variants are compared on exactly the
same images and folds.

Usage:
    python src/models/masked_features.py \
        --images_dir data/images/sidewalk-images \
        --encoder    dinov2-large \
        --output     results/features/dinov2-large_masked.npz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from train import ENCODERS, load_encoder

SEGFORMER = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
# Cityscapes ids for the walkable surface. Road is included because these are
# kerbside views where the crossing and the ramp often fall on the road side of
# the boundary, and passability depends on both.
WALKABLE_IDS = (0, 1)          # road, sidewalk


def load_segmenter(device):
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    proc = SegformerImageProcessor.from_pretrained(SEGFORMER)
    seg = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER).to(device).eval()
    return seg, proc


@torch.no_grad()
def walkable_masks(seg, proc, images: list, device, grid: int) -> torch.Tensor:
    """Fraction of each patch cell covered by road/sidewalk, shape (B, grid, grid)."""
    inputs = proc(images=images, return_tensors="pt").to(device)
    logits = seg(**inputs).logits                       # (B, C, h, w)
    pred = logits.argmax(dim=1)                         # (B, h, w)
    walkable = torch.zeros_like(pred, dtype=torch.float32)
    for cid in WALKABLE_IDS:
        walkable += (pred == cid).float()
    walkable = walkable.clamp(0, 1).unsqueeze(1)        # (B, 1, h, w)
    return F.interpolate(walkable, size=(grid, grid), mode="area").squeeze(1)


@torch.no_grad()
def patch_tokens(model, processor, images: list, enc_type: str, device):
    """Return (tokens, grid) with tokens shaped (B, grid*grid, D)."""
    if enc_type == "dinov2":
        px = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        out = model(pixel_values=px).last_hidden_state[:, 1:]      # drop CLS
    elif enc_type in ("clip", "siglip"):
        px = processor(images=images, return_tensors="pt", padding=True)["pixel_values"].to(device)
        vision = getattr(model, "vision_model", None)
        if vision is None:
            raise RuntimeError(f"{enc_type}: no vision_model to take patch tokens from")
        hs = vision(pixel_values=px).last_hidden_state
        # CLIP prepends a CLS token; SigLIP does not.
        n = hs.shape[1]
        out = hs[:, 1:] if int(round(n ** 0.5)) ** 2 != n else hs
    else:
        raise RuntimeError(f"patch tokens not supported for enc_type={enc_type}")

    grid = int(round(out.shape[1] ** 0.5))
    if grid * grid != out.shape[1]:
        raise RuntimeError(f"token count {out.shape[1]} is not a square grid")
    return out, grid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--encoder", default="dinov2-large", choices=list(ENCODERS))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--min_mask", type=float, default=0.02,
                   help="If the walkable region covers less than this fraction of "
                        "the image, fall back to mean-pooling every patch rather "
                        "than pooling over almost nothing.")
    p.add_argument("--extra_dirs", default="",
                   help="Comma-separated extra image directories (e.g. the "
                        "generalization cities), so the transfer test uses the "
                        "same feature definition.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dirs = [Path(args.images_dir)]
    dirs += [Path(d) for d in args.extra_dirs.split(",") if d.strip()]
    paths: list[Path] = []
    for d in dirs:
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            paths += sorted(d.rglob(ext))
    paths = sorted({p.resolve() for p in paths})
    print(f"Encoder : {args.encoder}")
    print(f"Images  : {len(paths)} from {len(dirs)} director(ies)\n")

    model, processor, enc_device, enc_type = load_encoder(args.encoder)
    seg, seg_proc = load_segmenter(device)

    keys, masked, whole, coverage = [], [], [], []

    for i in tqdm(range(0, len(paths), args.batch_size), desc="mask-pool"):
        batch = paths[i : i + args.batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]

        tokens, grid = patch_tokens(model, processor, images, enc_type, enc_device)
        m = walkable_masks(seg, seg_proc, images, device, grid).to(tokens.device)
        w = m.reshape(m.shape[0], -1)                     # (B, grid*grid)

        frac = w.mean(dim=1)
        # Where the segmenter finds essentially no walkable surface, fall back to
        # plain mean pooling: a near-empty mask would otherwise amplify one or two
        # arbitrary patches.
        fallback = frac < args.min_mask
        w = torch.where(fallback.unsqueeze(1), torch.ones_like(w), w)

        wn = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (tokens * wn.unsqueeze(-1)).sum(dim=1)   # (B, D)
        plain  = tokens.mean(dim=1)

        pooled = F.normalize(pooled, dim=-1)
        plain  = F.normalize(plain,  dim=-1)

        keys     += [str(p) for p in batch]
        masked   += list(pooled.cpu().numpy())
        whole    += list(plain.cpu().numpy())
        coverage += list(frac.cpu().numpy())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        keys=np.array(keys),
        masked=np.stack(masked).astype(np.float32),
        whole=np.stack(whole).astype(np.float32),
        coverage=np.array(coverage, dtype=np.float32),
        encoder=args.encoder,
    )

    cov = np.array(coverage)
    print(f"\nWalkable coverage: mean {cov.mean():.3f}, median {np.median(cov):.3f}, "
          f"min {cov.min():.3f}, max {cov.max():.3f}")
    print(f"Images falling back to mean pooling: {(cov < args.min_mask).sum()}/{len(cov)}")
    print(f"\nSaved → {out}  ({len(keys)} images, dim {masked[0].shape[0]})")


if __name__ == "__main__":
    main()
