# Human-Aligned Sidewalk Accessibility Perception

**Soft-label learning from 829 mobility aid users for accessibility-aware routing**

> Paper under review at CoRL 2026.

---

## Overview

Sidewalk accessibility is not binary. Two wheelchair users examining the same cracked sidewalk may reach opposite conclusions — not from labeling error, but from real differences in equipment and risk tolerance. We treat this disagreement as signal.

From **49,490 ternary votes (yes / unsure / no)** collected from **829 mobility aid users** via [Project Sidewalk](https://projectsidewalk.io/), we build per-aid human probability distributions for 5 mobility aid types. A frozen vision encoder with a per-aid linear probe is trained via **KL divergence against these distributions**, reproducing community-level uncertainty rather than collapsing it to majority votes.

### Key results

| Method | Overall F1 | Brier ↓ | Latency (ms/img) |
|--------|-----------|---------|-----------------|
| DINOv2-large + Soft-KL (ours) | 0.613 | **0.068** | 78.5 |
| CLIP ViT-B/32 + Hard-CE | **0.617** | 0.231 | 61.6 |
| Qwen3-VL-8B (zero-shot) | 0.576 | 0.468 | 6,182 |
| LLaVA-1.5-7B (zero-shot) | 0.418 | 0.423 | 135 |

- Soft-KL achieves **3× better calibration** (Brier 0.068 vs 0.231) than standard cross-entropy with no loss in F1
- DINOv2-large (no language supervision) ties the best CLIP variant on F1 while achieving the lowest Brier score
- Best zero-shot VLM (Qwen3) comes within 0.041 F1 of the trained probe at **102× higher latency**
- Routing demo on Pittsburgh PA: accessibility-aware routes achieve up to **+0.102 Δp_yes** at +15% distance overhead for wheelchair users

---

## Repository structure

```
sidewalk-accessibility-project/
│
├── src/
│   ├── data/
│   │   ├── firebase.py               # Parse Firebase JSONL → votes CSV
│   │   └── preprocess.py             # Tally votes → per-(image×aid) distributions
│   ├── segmentation/
│   │   ├── deproject.py              # GSV panorama → rectified 90° crops
│   │   ├── segment.py                # YOLOv12-seg mask extraction
│   │   └── verify.py                 # Area-ratio filter (A_pred/A_gt ∈ [0.7,1.3])
│   ├── models/
│   │   ├── train.py                  # Linear probe training (Soft-KL + Hard-CE)
│   │   ├── crossval.py               # 5-fold panorama-level cross-validation
│   │   ├── train_final_model.py      # Train on full dataset (for routing)
│   │   ├── infer.py                  # Single-image inference
│   │   ├── zero_shot.py              # VLM zero-shot evaluation
│   │   ├── latency_benchmark.py      # Encoder + VLM latency measurement
│   │   ├── error_analysis.py         # Per-aid error analysis (walking cane study)
│   │   ├── compute_vlm_brier.py      # Soft Brier scores for VLM predictions
│   │   ├── plot_results.py           # Publication figures (Fig 1–5)
│   │   └── summarize_cv.py           # CV results summary table
│   ├── routing/
│   │   ├── fetch_ps_labels.py        # Download Project Sidewalk GPS labels
│   │   ├── score_osm_edges.py        # Snap PS labels → OSM graph edges
│   │   ├── demo.py                   # Accessibility-aware routing (Dijkstra)
│   │   └── plot_barrier_cost_pareto.py # Pareto curve: accessibility vs distance
│   └── generalization/
│       ├── download_city_images.py   # Download GSV thumbnails for new cities
│       └── evaluate_generalization.py # Out-of-distribution evaluation
│
├── data/
│   └── processed/
│       ├── tallies_firebase.json     # 260 (image×aid) soft-label distributions
│       └── image_selection_firebase.csv  # 49,490 raw votes, 829 participants
│
├── models/
│   └── yolo/bestv12.pt               # YOLOv12-seg checkpoint (sidewalk segmentation)
│
├── results/
│   ├── cv/{soft_kl,hard_ce}/         # CV results per encoder (JSON + logs)
│   ├── zero_shot/                    # VLM zero-shot results + Brier comparison
│   ├── figures/                      # Publication figures (PDF + PNG)
│   ├── generalization/               # Out-of-distribution results
│   ├── routing/                      # Routing figures + Pareto curve
│   ├── error_analysis/               # Walking cane analysis
│   └── latency/                      # Latency benchmark results
│
├── paper/                            # LaTeX manuscript (excluded from git)
├── notebooks/                        # Exploratory notebooks
│
├── run_cv.sh                         # Run 5-fold CV for all 8 encoders
├── run_zero_shot.sh                  # Run VLM zero-shot evaluation
├── run_latency.sh                    # Run latency benchmark
├── run_prompt_sensitivity.sh         # Run prompt sensitivity analysis (3×3)
├── run_generalization.sh             # Run generalization evaluation
└── run_routing_demo.sh               # Full routing pipeline
```

---

## Setup

**Requirements:** Python 3.11+, CUDA GPU recommended (A100 40GB used in experiments).

```bash
git clone https://github.com/wesleymaia999/sidewalk-accessibility-project.git
cd sidewalk-accessibility-project
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Data:** Place Project Sidewalk data in `data/processed/`:
- `tallies_firebase.json` — per-(image×aid) vote distributions (260 rows)
- `image_selection_firebase.csv` — raw votes (49,490 rows)
- Sidewalk crop images in `data/images/sidewalk-images/`

---

## Reproducing experiments

All scripts below write results to `results/` and accept a `WANDB_PROJECT` env var for logging.

### Cross-validation (main results)

```bash
# Soft-KL — paper method (all 8 encoders × 5 folds)
./run_cv.sh

# Hard-CE ablation baseline
LOSS_TYPE=hard_ce ./run_cv.sh
```

Results → `results/cv/{soft_kl,hard_ce}/<encoder>/cv_results.json`

### Zero-shot VLM evaluation

```bash
./run_zero_shot.sh
```

Results → `results/zero_shot/<model>/zero_shot_results.json`

### Brier score comparison (probe vs VLMs)

```bash
python src/models/compute_vlm_brier.py \
    --results results/zero_shot/qwen3-vl-8b/zero_shot_results.json \
              results/zero_shot/qwen2.5-vl-7b/zero_shot_results.json \
              results/zero_shot/llava-1.5-7b/zero_shot_results.json \
    --output  results/zero_shot/vlm_brier_comparison.json
```

### Latency benchmark

```bash
./run_latency.sh
```

Results → `results/latency/latency_results.json`

### Generalization to unseen cities

```bash
# 1. Download images (Pittsburgh + Zurich, 250 images each city)
python src/generalization/download_city_images.py \
    --city pittsburgh --n_per_class 100 \
    --output_dir data/generalization/images/pittsburgh \
    --append_csv data/generalization/test_images.csv

# 2. Evaluate
./run_generalization.sh dinov2-large results/models/dinov2-large
```

Results → `results/generalization/dinov2-large/agreement_summary.json`

### Accessibility-aware routing

```bash
./run_routing_demo.sh
```

Runs the full pipeline: fetch Project Sidewalk Pittsburgh labels → score OSM edges → Dijkstra routing → barrier_cost ablation.

Results → `results/routing/`

### Publication figures

```bash
python src/models/plot_results.py
```

Generates Fig 1–5 in `results/figures/`.

---

## Results summary

### Table 1 — Hard-CE macro-F1 per encoder (5-fold CV)

| Encoder | Cane | Walker | Scooter | Man.WC | Mot.WC | **Overall** |
|---------|------|--------|---------|--------|--------|------------|
| CLIP B/32 | 0.453 | **0.761** | 0.558 | **0.675** | 0.639 | **0.617** |
| DINOv2-base | 0.446 | 0.720 | 0.668 | 0.574 | **0.664** | 0.614 |
| DINOv2-large | 0.489 | 0.675 | **0.746** | 0.502 | 0.652 | 0.613 |
| SigLIP2-base | 0.458 | 0.700 | 0.587 | 0.572 | 0.664 | 0.596 |
| CLIP L/14 | 0.453 | 0.707 | 0.698 | 0.523 | 0.597 | 0.595 |
| ViT-B/16-sup | **0.513** | 0.682 | 0.606 | 0.553 | 0.509 | 0.572 |
| SigLIP2-SO400M | 0.453 | 0.669 | 0.519 | 0.483 | 0.652 | 0.555 |
| CLIP B/16 | 0.434 | 0.536 | 0.569 | 0.562 | 0.618 | 0.544 |

### Routing — Pittsburgh PA (Forbes/Murray corridor)

| Mobility Aid | Std. p_yes | Acc. p_yes | Δ access | Δ dist |
|-------------|-----------|-----------|----------|--------|
| Walking cane | 0.753 | 0.763 | +0.010 | +58m (+3%) |
| Walker | 0.658 | 0.703 | +0.045 | +187m (+9%) |
| Mobility scooter | 0.561 | 0.643 | +0.082 | +267m (+13%) |
| Manual wheelchair | 0.485 | 0.573 | +0.088 | +301m (+15%) |
| Motorized wheelchair | 0.516 | 0.618 | +0.102 | +313m (+15%) |

Each aid produces a **genuinely different optimal route** when real per-edge accessibility scores are used.

---

## Encoders evaluated

| # | Model | HuggingFace ID | Dim |
|---|-------|----------------|-----|
| 1 | CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | 512 |
| 2 | CLIP ViT-B/16 | `openai/clip-vit-base-patch16` | 512 |
| 3 | CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | 768 |
| 4 | DINOv2-base | `facebook/dinov2-base` | 768 |
| 5 | DINOv2-large | `facebook/dinov2-large` | 1024 |
| 6 | SigLIP2-base | `google/siglip2-base-patch16-224` | 768 |
| 7 | SigLIP2-SO400M | `google/siglip2-so400m-patch14-384` | 1152 |
| 8 | ViT-B/16-sup | `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` | 768 |

---

## Citation

```bibtex
@inproceedings{maia2026sidewalk,
  title     = {Human-Aligned Sidewalk Accessibility Perception: Soft-Label Learning
               from 829 Mobility Aid Users for Autonomous Mobility Services},
  author    = {Maia, Wesley and {others}},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
}
```

---

## Acknowledgments

- Crowdsourced labels collected via [Project Sidewalk](https://projectsidewalk.io/) (Saha et al., CHI 2019)
- Street View thumbnails via Google Street View API
- OpenStreetMap pedestrian graphs via [OSMnx](https://github.com/gboeing/osmnx)
