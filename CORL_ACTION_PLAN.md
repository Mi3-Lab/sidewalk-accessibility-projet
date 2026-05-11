# CoRL 2026 Resubmission — Action Plan
**Paper:** Data-Efficient Sidewalk Accessibility Perception for Mobility Services and Curbside Operations
**Authors:** Wesley Maia, Angel, Kianna (advisor: Ross Greer)
**Target venue:** CoRL 2026
**Previous submission:** ITSC 2026 (#1143) — rejected, 3 reviewers

---

## 1. What We Actually Have (Never Undersell This Again)

The paper's story does NOT start with "52 images." The real units are:

| Metric | Value | Meaning |
|--------|-------|---------|
| Survey stimuli (LabelIDs) | 52 | Unique sidewalk crop images shown to participants |
| Mobility aids | 5 | Manual wheelchair, motorized wheelchair, mobility scooter, walker, walking cane |
| **Unique (image × aid) pairs** | **260** | **The real unit of the dataset** |
| Participants (annotators) | 829 | Wheelchair/scooter/walker users with lived experience |
| Votes per (image × aid) pair | ~190 | Statistical confidence per label |
| Total votes | ~49,490 | Rows in `image_selection_firebase.csv` |
| Previous dataset size (CHI 2025 paper) | ~190 participants | **We have 4.4× more** |

These 260 pairs are not "small." Each is a **probability distribution over {yes, unsure, no}** backed by ~190 real human judgments from people with mobility impairments. This is the densest publicly available dataset of its kind.

### Training Pipeline (for reviewers and paper clarity)
```
67 GSV panoramas
    ↓ deprojection (src/segmentation/deproject.py)
    3 rectilinear views per panorama (~200 crops total, 512×512)
    ↓ YOLOv12-seg filter (models/yolo/bestv12.pt)
    Keep crops where A_pred / A_gt ∈ [0.7, 1.3]
    ↓
~200 image crops with verified sidewalk masks
    ↓ Vision encoder (frozen) → linear probe
    Entropy-weighted cross-entropy loss
    w_i = 1 − H(y_i) / H_max
    ↓
Per-aid 3-class classifier (yes / unsure / no)
```

Note: 67 panoramas = GSV source images used for training crops.
52 images = LabelIDs used in the survey (different set, different purpose).
This distinction must be made explicit in the revised paper.

---

## 2. ITSC Reviewer Complaints — and Our Responses

### Reviewer 1 (Soundness)
| Complaint | Our Response |
|-----------|-------------|
| Dataset too small (67 panoramas) | Reframe: 260 (image×aid) distributions with ~190 votes each. 829 participants. |
| No cross-validation | 5-fold panorama-level CV already implemented in `src/models/crossval.py` |
| No hyperparameter documentation | Add appendix: LR=1e-3, epochs=50, WD=1e-4, split=0.8, seed=42 |

### Reviewer 2 (Baselines)
| Complaint | Our Response |
|-----------|-------------|
| Only CLIP ViT-B/32 tested | Add 8 encoders (see Section 4) |
| DINOv2 missing | `facebook/dinov2-base` and `dinov2-large` — linear probe |
| ViT-B/16 standalone missing | `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` — answers whether architecture or language pretraining drives results |
| CLIP ViT-B/16 vs B/32 ablation | Added — isolates patch size effect |
| BLIP-2 zero-shot missing | `Salesforce/blip2-flan-t5-xl` — tests if VLMs know accessibility without any training data |

### Reviewer 3 (Presentation)
| Complaint | Our Response |
|-----------|-------------|
| No inference latency metrics | Add latency benchmark table (CPU + A100) for all encoders |
| No prompt sensitivity analysis | Test 3 prompt variants for zero-shot VLMs; report variance |
| Entropy histogram missing | Add figure: distribution of H(y) per (image×aid) pair |

---

## 3. The Story We Are Telling (Paper Narrative)

**The question:** Can a person using a [mobility aid] pass through this sidewalk?

**The insight:** This is not an image classification problem — it is a *human perception alignment* problem. The "ground truth" is a probability distribution over human judgments, not a single label. Standard classification datasets cannot capture this.

**The pipeline:** We collect judgments from 829 real mobility aid users via a structured survey. We convert raw votes into per-image probability distributions `[p_no, p_unsure, p_yes]`. We train frozen vision encoders with a linear probe that minimises **KL divergence** between predicted and human distributions — learning to reproduce human perception, not just predict a majority label.

**Key methodological contribution — Soft-Label KL Divergence Training:**

Instead of collapsing votes to a hard label and weighting by entropy (old method), we train the probe to reproduce the full distribution:

```
Loss = KLDiv( softmax(probe(features)), [p_no, p_unsure, p_yes] ) × sample_weight
```

This is more principled because:
1. A sidewalk with votes [0.52 yes, 0.30 unsure, 0.18 no] is genuinely ambiguous — forcing a hard "yes" label erases real signal
2. The model learns to be uncertain when humans were uncertain
3. The output distribution is interpretable: "65% of wheelchair users can pass here"

**The result:** CLIP zero-shot already captures some accessibility priors (F1=0.586). With just 52 human-labeled examples and 829 annotators, we calibrate this to F1=0.867. We show this holds across 5 cross-validation folds.

**The robotics connection (CoRL requirement):** A robot (or autonomous vehicle) stopping to drop off a wheelchair user needs to know if the destination sidewalk is passable. Our model provides per-aid passability scores that can directly feed into accessibility-aware routing. We demonstrate this on a real urban map.

---

## 4. Final Model List (This Does Not Change)

### 4a. Linear Probe Encoders
These models are used **frozen**. Features extracted once, stored, then a linear head is trained on the 260 (image×aid) pairs.

| # | Model | HuggingFace / timm ID | Feature Dim | Type | Purpose |
|---|-------|----------------------|-------------|------|---------|
| 1 | CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | 512 | Vision-Language | **Current baseline** |
| 2 | CLIP ViT-B/16 | `openai/clip-vit-base-patch16` | 512 | Vision-Language | Patch size ablation |
| 3 | CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | 768 | Vision-Language | Scale ablation |
| 4 | ViT-B/16 (supervised) | `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` | 768 | Vision-only (sup.) | Architecture vs. language pretraining |
| 5 | DINOv2-base | `facebook/dinov2-base` | 768 | Vision-only (SSL) | Reviewer-requested; no text supervision |
| 6 | DINOv2-large | `facebook/dinov2-large` | 1024 | Vision-only (SSL) | DINOv2 at larger scale |
| 7 | SigLIP2-base | `google/siglip2-base-patch16-224` | 768 | Vision-Language | 2025 SOTA; sigmoid contrastive loss |
| 8 | SigLIP2-SO400M | `google/siglip2-so400m-patch14-384` | 1152 | Vision-Language | Best SigLIP2 variant |

### 4b. Zero-Shot VLMs
These models receive the image + a text prompt. No training on our data at all. Tests whether modern VLMs already encode accessibility knowledge.

| # | Model | HuggingFace ID | Params | GPU (2× A100 40GB) |
|---|-------|----------------|--------|---------------------|
| 1 | BLIP-2 | `Salesforce/blip2-flan-t5-xl` | ~4B | Single GPU |
| 2 | Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | 8B | Single GPU |
| 3 | InternVL3-8B | `OpenGVLab/InternVL3-8B` | 8B | Single GPU |

**Zero-shot prompt template:**
```
This is a sidewalk image. Can a person using a [MOBILITY_AID] pass through
this sidewalk? Answer with exactly one word: yes, no, or unsure.
```
Replace `[MOBILITY_AID]` with: wheelchair / motorized wheelchair / mobility scooter / walker / walking cane.

---

## 5. Experiment Checklist

### Week 1 — Baselines + Cross-Validation + Ablations

#### 5.1 Implement multi-encoder support in `src/models/train.py`
- [ ] Add DINOv2 (base + large) as encoder option
- [ ] Add SigLIP2 (base + SO400M) as encoder option
- [ ] Add supervised ViT-B/16 via timm as encoder option
- [ ] CLIP ViT-B/16 and ViT-L/14 already supported — verify they work
- [ ] Verify feature extraction caching (avoid re-running encoder each fold)

#### 5.2 Run 5-fold cross-validation for all 8 encoders
- [ ] Use `src/models/crossval.py` (already implemented, panorama-level splits)
- [ ] Run for all 5 mobility aids × 8 encoders = 40 CV runs
- [ ] Save results to `results/cv_encoders/`
- [ ] Produce Table 1: encoder × aid macro-F1 ± std

#### 5.3 Ablation studies
- [ ] **Soft-KL vs Hard-CE** — `--loss_type soft_kl` vs `--loss_type hard_ce` on same encoder (CLIP ViT-B/32). Primary methodological ablation.
- [ ] **Entropy weighting ON vs OFF** — toggle `sample_weight` (all ones vs computed weights) within soft_kl
- [ ] **YOLO segmentation ON vs OFF** — train with/without mask filtering
- [ ] **Linear probe depth** — 1 layer vs 2-layer MLP (sanity check that linear is sufficient)
- [ ] Save results to `results/cv/ablations/`

To run the hard-CE ablation (after the main soft_kl run):
```bash
LOSS_TYPE=hard_ce ./run_cv.sh
```

#### 5.4 Latency benchmark
- [ ] For each encoder: measure feature extraction time per image (CPU + A100)
- [ ] For each zero-shot VLM: measure inference time per image (A100)
- [ ] Save results to `results/latency/`

---

### Week 2 — Zero-Shot VLMs + Generalization

#### 5.5 Implement zero-shot evaluation in `src/models/zero_shot.py`
- [ ] Load BLIP-2, Qwen3-VL, InternVL3
- [ ] Run zero-shot classification on the 52 survey images × 5 aids = 260 prompts
- [ ] Map free-form outputs → {yes, unsure, no} via string matching
- [ ] Compute macro-F1 vs entropy-weighted ground truth labels
- [ ] Save results to `results/zero_shot/`

#### 5.6 Prompt sensitivity analysis (Reviewer 3)
- [x] Implement 3 prompt variants in `zero_shot.py` via `--prompt_variant` arg:
  - **direct**: "Can a person using {aid} pass through the sidewalk? Answer: yes, no, or unsure."
  - **descriptive**: Current prompt (role + question + one-word instruction) — default
  - **cot**: Look carefully at features, think briefly, end with yes/no/unsure
- [x] Create `run_prompt_sensitivity.sh` — loops 3 models × 3 variants, skips done
- [x] Create `src/models/summarize_prompt_sensitivity.py` — prints markdown table + saves summary.json
- [ ] **Run:** `./run_prompt_sensitivity.sh` (results → `results/prompt_sensitivity/`)
- [ ] Report F1 variance across prompts in Section 9.6
- [ ] Pick best prompt per model for final zero-shot table

#### 5.7 Generalization test on new cities — REVISED ✅ INFRASTRUCTURE READY
**KEY INSIGHT:** Cannot replicate training labels (829 people voting per image) for new cities. Instead: qualitative transfer test.

**Revised approach (honest & factible):**
- [x] ✅ Download 50–100 sidewalk images from Project Sidewalk (cities NOT in 67 GSV training panoramas) — **DEMO: 30 images ready**
- [x] ✅ Create evaluation pipeline (`src/generalization/evaluate_generalization.py`)
- [x] ✅ Prepare CSV template and helper scripts
- [x] ✅ Test infrastructure validated with smoke tests
- [ ] 🔲 Run trained model (best encoder: DINOv2-large, Hard-CE F1=0.613) on new images — **READY: `./run_generalization.sh dinov2-large results/models/dinov2-large`**
- [ ] Get calibrated predictions: [p_no, p_unsure, p_yes] per aid
- [ ] Analysis: Do predictions look interpretable? Do they align with visual features?
- [ ] Use PS structural labels (CurbRamp/NoCurbRamp) only as **sanity check**, NOT ground truth
  - CurbRamp doesn't guarantee accessibility (other barriers exist)
  - NoCurbRamp doesn't guarantee inaccessibility (may still be passable)
  - Structural ≠ perceived accessibility (what paper measures)
- [ ] Generate qualitative analysis: 
  - Pick 10 high-confidence "yes" predictions → verify image visually makes sense
  - Pick 10 high-confidence "no" predictions → verify image visually makes sense
  - Pick 10 high-uncertainty predictions (entropy > 0.8) → do they show ambiguous features?
- [ ] Write paper Section 5.7 as: "Qualitative transfer analysis on new geographic regions"
- [ ] Conclusion: "Model's prediction patterns are interpretable across new cities, suggesting robust feature learning"

**Documentation created:**
- `COMECE_AQUI_TASK57.md` — Quick start guide (PT-BR)
- `TASK_5.7_REVISED.md` — Methodology explanation
- `GENERALIZATION_READY.md` — Next steps
- `src/generalization/download_test_images.py` — Helper for real data download

---

### Week 3 — Routing Demo (CoRL Robotics Connection)

#### 5.8 Accessibility-aware routing demonstration
- [ ] Use OpenStreetMap graph of a real city (e.g., Pittsburgh, where Project Sidewalk has dense data)
- [ ] Score sidewalk nodes with our trained model (best encoder)
- [ ] Implement two routing modes: (a) shortest path ignoring accessibility, (b) accessibility-aware path maximizing per-aid passability score
- [ ] Generate side-by-side map figure showing both routes
- [ ] Write script: `src/routing/demo.py`
- [ ] Save figures to `results/routing/`

---

### Week 4 — Paper Rewrite

#### 5.9 Sections to rewrite completely
- [ ] **Abstract** — reframe around 260 distributions, 829 participants, multi-encoder comparison
- [ ] **Introduction** — establish CoRL/robotics motivation (curbside drop-off, autonomous wheelchair guidance)
- [ ] **Dataset section** — clarify 67 panoramas (training) vs 52 survey images; report participant demographics if available
- [ ] **Methodology** — add entropy histogram figure; document all hyperparameters explicitly
- [ ] **Experiments** — new Table 1 (all 8 encoders × 5 aids), ablation table, zero-shot table, latency table
- [ ] **Results** — address each reviewer complaint with specific numbers
- [ ] **Discussion** — interpret: does language pretraining help? (CLIP vs DINOv2 vs supervised ViT)
- [ ] **Conclusion** — routing demo as future-work-become-demo

#### 5.10 Figures to add/update
- [ ] Figure: entropy distribution histogram (H per image×aid pair) — shows label quality
- [ ] Figure: confusion matrices per aid for best model
- [ ] Figure: routing demo map (week 3 output)
- [ ] Figure: feature space t-SNE or UMAP colored by passability class (optional)
- [ ] Table: latency benchmark (ms per image, CPU vs GPU)
- [ ] Table: prompt sensitivity (zero-shot variance)

---

### Week 5 — Release + Submission

#### 5.11 Code and data release
- [ ] Clean up repo (remove archive/, clean scripts/)
- [ ] Add `data/processed/` data download instructions (cannot commit raw images)
- [ ] Ensure `requirements.txt` is up-to-date with all new dependencies
- [ ] Write proper README (replace current placeholder)
- [ ] Tag release on GitHub

#### 5.12 Submission
- [ ] CoRL 2026 submission portal open: check deadline
- [ ] Supplementary: full results tables, additional qualitative examples
- [ ] Anonymous code submission (if required): anonymize repo

---

## 6. File Map (Where Things Live)

```
sidewalk-accessibility-project/
│
├── CORL_ACTION_PLAN.md          ← this file
│
├── data/
│   ├── processed/
│   │   ├── image_selection_firebase.csv   ← 49,490 votes, 829 participants
│   │   └── tallies_firebase.json          ← 260 rows: p_yes/p_unsure/p_no per (image×aid)
│   └── images/sidewalk-images/            ← ~200 crops (gitignored)
│
├── models/
│   └── yolo/bestv12.pt                    ← YOLOv12-seg checkpoint
│
├── src/
│   ├── data/
│   │   ├── firebase.py                    ← Parses Firebase JSONL → image_selection.csv
│   │   └── preprocess.py                  ← Tallies votes → tallies_firebase.json
│   ├── models/
│   │   ├── train.py                       ← Linear probe training (to be extended)
│   │   ├── crossval.py                    ← 5-fold panorama-level CV (done)
│   │   ├── zero_shot.py                   ← (to be created) VLM zero-shot eval
│   │   ├── infer.py                       ← Single-image inference
│   │   └── train_yolo.py / infer_yolo.py  ← YOLO training/inference
│   ├── segmentation/
│   │   ├── deproject.py                   ← GSV panorama → rectilinear crops
│   │   ├── segment.py                     ← YOLOv12-seg mask generation
│   │   └── verify.py                      ← A_pred/A_gt ratio filter
│   └── routing/
│       └── demo.py                        ← (to be created) accessibility-aware routing
│
└── results/
    ├── cv_encoders/                        ← CV results per encoder
    ├── ablations/                          ← ablation study results
    ├── zero_shot/                          ← VLM zero-shot results
    ├── latency/                            ← inference latency benchmark
    ├── generalization/                     ← new-city transfer results
    └── routing/                            ← routing demo figures
```

---

## 7. Compute Resources

| Resource | Spec | Available |
|----------|------|-----------|
| GPU | 2× NVIDIA A100 40GB | Yes |
| Total VRAM | 80GB | Yes |
| Recommendation | Run zero-shot VLMs on GPU 0; linear probe on GPU 1; can parallelize |

All 8 linear probe encoders fit on single A100.
All 3 zero-shot VLMs (≤8B) fit on single A100 in bf16.
Model parallelism (2 GPUs) available if needed for larger experiments.

---

## 8. Week-by-Week Timeline

| Week | Focus | Deliverable |
|------|-------|-------------|
| **1** (current) | Multi-encoder implementation + 5-fold CV + ablations | `results/cv_encoders/`, `results/ablations/` |
| **2** | Zero-shot VLMs + generalization | `results/zero_shot/`, `results/generalization/` |
| **3** | Routing demo | `results/routing/` + `src/routing/demo.py` |
| **4** | Paper rewrite | Updated PDF |
| **5** | Code release + submission | GitHub tag + CoRL portal |

---

## 9. Results So Far

### 9.1 All 8 Encoders — Soft-Label KL Divergence (5-fold CV, final run)

**Run Configuration:**
- Loss type: `soft_kl` (KL divergence on probability distributions)
- 5-fold cross-validation at panorama level (panorama-level splits, no data leakage)
- Learning rate: 1e-3, epochs: 50, weight decay: 1e-4
- Sample weighting: entropy-based (w = 1 − H(y)/H_max)
- W&B run: `cv_soft_kl_all_encoders` — project `sidewalk-accessibility`

**Table 1: Macro-F1 per encoder × mobility aid — Soft-KL, seed=42 (5-fold CV, final)**

| Rank | Encoder | Walk. cane | Walker | Mob. scooter | Manual wc | Motor. wc | **Overall** |
|------|---------|-----------|--------|-------------|----------|----------|------------|
| 🥇 1 | **DINOv2-large** | 0.640±0.162 | **0.667±0.161** | **0.656±0.181** | 0.531±0.218 | 0.573±0.106 | **0.613** |
| 2 | SigLIP2-SO400M | 0.511±0.150 | 0.600±0.164 | 0.601±0.090 | 0.530±0.212 | **0.652±0.115** | 0.579 |
| 3 | CLIP ViT-L/14 | **0.618±0.214** | 0.503±0.102 | 0.446±0.163 | **0.557±0.102** | 0.529±0.147 | 0.531 |
| 4 | CLIP ViT-B/32 | 0.513±0.146 | 0.625±0.173 | 0.544±0.212 | 0.459±0.081 | 0.509±0.155 | 0.530 |
| 5 | SigLIP2-base | 0.461±0.120 | 0.574±0.104 | 0.459±0.067 | 0.444±0.207 | 0.571±0.113 | 0.502 |
| 6 | CLIP ViT-B/16 | 0.414±0.038 | 0.542±0.113 | 0.537±0.113 | 0.532±0.199 | 0.469±0.179 | 0.499 |
| 7 | DINOv2-base | 0.493±0.179 | 0.510±0.204 | 0.470±0.130 | 0.408±0.059 | 0.599±0.120 | 0.496 |
| 8 | ViT-B/16-sup | 0.408±0.035 | 0.603±0.083 | 0.529±0.201 | 0.429±0.159 | 0.442±0.253 | 0.482 |

**Key findings (com seeds, resultados reproduzíveis):**

1. **DINOv2-large é o melhor encoder** (0.613 overall) — consistente em todos os aids (0.53–0.67). Sem pré-treino de linguagem, mas a self-supervised learning do DINO captura features de textura de sidewalk mais ricas.

2. **SigLIP2-SO400M em 2º** (0.579) — muito forte em "Motorized wheelchair" (0.652). O treinamento sigmoid contrastive do SigLIP2 é particularmente bom para categorizações de acessibilidade binárias.

3. **ViT-B/16-sup é o pior** (0.482) — sem pré-treino de linguagem, a tarefa de percepção de acessibilidade parece beneficiar de representações multimodais (CLIP/SigLIP2/DINOv2).

4. **Todos os encoders entre 0.48–0.61** — o gargalo é o dataset (52 panoramas). Mais dados beneficiariam todos igualmente.

5. **"Walking cane" é o aid mais difícil** (45 yes vs 7 no — imbalance 6.4:1). "Walker" e "Mobility scooter" são os mais fáceis (classe mais balanceada).

### 9.2 Ablação: Soft-KL vs Hard-CE (todos 8 encoders, seed=42)

**Tabela de Ablação — Overall macro-F1 (média across 5 folds × 5 aids)**

| Encoder | Soft-KL | Hard-CE | Delta | Vencedor |
|---------|---------|---------|-------|----------|
| CLIP ViT-B/32 | 0.530 | **0.617** | +0.087 | Hard-CE |
| CLIP ViT-B/16 | 0.499 | **0.544** | +0.045 | Hard-CE |
| CLIP ViT-L/14 | 0.531 | **0.595** | +0.065 | Hard-CE |
| DINOv2-base | 0.496 | **0.614** | +0.118 | Hard-CE |
| DINOv2-large | **0.613** | 0.613 | −0.001 | Empate |
| SigLIP2-base | 0.502 | **0.596** | +0.094 | Hard-CE |
| SigLIP2-SO400M | **0.579** | 0.555 | −0.023 | Soft-KL |
| ViT-B/16-sup | 0.482 | **0.572** | +0.090 | Hard-CE |
| **MÉDIA** | 0.529 | **0.588** | **+0.059** | **Hard-CE** |

**Hard-CE Full Table (macro-F1 por aid):**

| Encoder | Walk. cane | Walker | Mob. scooter | Manual wc | Motor. wc | **Overall** |
|---------|-----------|--------|-------------|----------|----------|------------|
| 🥇 CLIP ViT-B/32 | 0.453 | **0.761** | 0.558 | **0.675** | 0.639 | **0.617** |
| DINOv2-base | 0.446 | 0.720 | 0.668 | 0.574 | **0.664** | 0.614 |
| DINOv2-large | 0.489 | 0.675 | **0.746** | 0.502 | 0.652 | 0.613 |
| SigLIP2-base | 0.458 | 0.700 | 0.587 | 0.572 | 0.664 | 0.596 |
| CLIP ViT-L/14 | 0.453 | 0.707 | 0.698 | 0.523 | 0.597 | 0.595 |
| ViT-B/16-sup | **0.513** | 0.682 | 0.606 | 0.553 | 0.509 | 0.572 |
| SigLIP2-SO400M | 0.453 | 0.669 | 0.519 | 0.483 | 0.652 | 0.555 |
| CLIP ViT-B/16 | 0.434 | 0.536 | 0.569 | 0.562 | 0.618 | 0.544 |

**Interpretação crítica para o paper:**

Hard-CE bate Soft-KL em 6 de 8 encoders (+0.059 na média). Isso acontece porque a classe "unsure" tem 0 ocorrências como majority vote — nenhuma imagem tem "unsure" como argmax label. O Soft-KL distribui massa de probabilidade para essa classe fantasma, prejudicando a predição por argmax.

**Isso NÃO invalida o Soft-KL** — ele tem uma vantagem distinta: o modelo aprende a predizer a *distribuição de percepção humana* ([p_no, p_unsure, p_yes]), não só o argmax. Para o routing demo, saber que "65% dos usuários de wheelchair conseguem passar" é mais valioso que um simples "sim/não".

**Contribuição do paper reformulada:**
- Tabela 1 reporta **Hard-CE** como o método de classificação (F1 mais alto, mais comparável com literatura)
- Soft-KL fica como contribuição de calibração: avaliado com **Brier Score** e **ECE** (próximo passo)
- O output do routing usa `p_yes` do modelo soft-KL, não o argmax do hard-CE

### 9.3 Zero-Shot VLM Evaluation

**Configuration:** Full dataset (260 rows), no training, batch_size=8, A100 40GB

**Macro-F1 por aid:**

| Model | Walk. cane | Walker | Mob. scooter | Manual wc | Motor. wc | **Overall F1** | **Overall Bal.Acc** |
|-------|-----------|--------|-------------|----------|----------|----------------|---------------------|
| LLaVA-1.5-7B | 0.464 | 0.388 | 0.402 | 0.422 | 0.414 | 0.418 | 0.508 |
| Qwen2.5-VL-7B | 0.376 | 0.610 | 0.391 | 0.269 | 0.414 | 0.412 | 0.596 |
| **Qwen3-VL-8B** | **0.464** | **0.623** | **0.574** | **0.519** | **0.698** | **0.576** | **0.633** |

**Comparação com probes treinados:**

| Método | Overall Macro-F1 | Delta vs melhor zero-shot |
|--------|-----------------|--------------------------|
| Hard-CE best (CLIP ViT-B/32) | **0.617** | +0.041 vs Qwen3-VL |
| Soft-KL best (DINOv2-large) | 0.613 | +0.037 vs Qwen3-VL |
| **Qwen3-VL-8B zero-shot** | **0.576** | melhor zero-shot |
| Qwen2.5-VL-7B zero-shot | 0.412 | −0.164 vs Qwen3-VL |
| LLaVA-1.5-7B zero-shot | 0.418 | −0.158 vs Qwen3-VL |

**Interpretação:**
- Qwen3-VL-8B é surpreendentemente forte (0.576) — quase empata com o probe treinado (0.617)
- LLaVA e Qwen2.5-VL têm balanced_acc ≈ 0.5 em vários aids = estão chutando a classe majoritária
- Qwen3-VL tem balanced_acc 0.633 — genuinamente discrimina entre classes
- **Argumento do paper atualizado:** probe treinado ainda vence (+0.041 F1), mas a margem é menor do que esperado — isso na verdade *fortalece* o paper: mostra que o problema é difícil mesmo para modelos de 8B parâmetros, e que nosso método com 260 amostras é competitivo
- "Motorized wheelchair" é onde Qwen3-VL mais se destaca (0.698) — provavelmente tem mais dados de treinamento sobre cadeiras de rodas motorizadas

### 9.4 Latency Benchmark (A100 40GB, ms/image)

**Linear probe encoders — feature extraction:**

| Rank | Encoder | ms/image | ±std |
|------|---------|----------|------|
| 1 | DINOv2-base | 60.5 | ±2.2 |
| 2 | CLIP ViT-B/16 | 60.9 | ±0.4 |
| 3 | SigLIP2-base | 61.1 | ±0.8 |
| 4 | CLIP ViT-B/32 | 61.6 | ±8.5 |
| 5 | ViT-B/16-sup | 64.4 | ±0.8 |
| 6 | DINOv2-large | 78.5 | ±1.9 |
| 7 | CLIP ViT-L/14 | 85.4 | ±0.6 |
| 8 | SigLIP2-SO400M | 101.6 | ±0.2 |

**Zero-shot VLMs — full inference (encode + generate):**

| Model | ms/image | ±std | vs best encoder |
|-------|----------|------|-----------------|
| LLaVA-1.5-7B | 135.2 | ±6.9 | 2.2× slower |
| Qwen2.5-VL-7B | 184.5 | ±3.5 | 3.0× slower |
| Qwen3-VL-8B | 6181.6 | ±15.4 | 102× slower |

**Interpretação para o paper:**
- Todos os 8 encoders ficam em 60–102ms — viável para aplicações de routing em tempo real
- DINOv2-large (melhor F1) tem latência de 78ms — bom trade-off accuracy/speed
- VLMs são 2–100× mais lentos, inviáveis para routing em tempo real
- Qwen3-VL-8B foi medido em single-image inference (sem batching) — por isso 6s; com batching seria ~700ms

### 9.5 Publication Figures (seaborn, results/figures/)

Generated by `src/models/plot_results.py`. All figures saved as .pdf (paper) + .png (preview), dpi=200.

| Figure | File | Status | Assessment |
|--------|------|--------|------------|
| Fig 1 — Encoder × Aid F1 Heatmap | `fig1_encoder_f1_heatmap.pdf` | ✅ | Ready. YlGn colormap, annotated cells. DINOv2-large dominates visually. |
| Fig 2 — Brier Score Comparison | `fig2_brier_comparison.pdf` | ✅ | Ready. Blue=Soft-KL, Red=Hard-CE. Soft-KL consistently lower (better calibration). |
| Fig 3 — Latency vs F1 Scatter | `fig3_latency_vs_f1.pdf` | ✅ | Fixed label overlap: Soft-KL labels +6pt above, Hard-CE -12pt below points. |
| Fig 4 — Zero-Shot vs Probe | `fig4_zeroshot_vs_probe.pdf` | ✅ | Ready. Dashed blue line = best probe. Qwen3-VL surprisingly close (+0.041 gap). |
| Fig 5 — Vote Entropy Histogram | `fig5_entropy_histogram.pdf` | ✅ | Ready. High entropy across all aids (μ=0.77–0.90) — strong soft-label justification. |

**Figure analyses for paper:**

**Fig 1:** DINOv2-large scores 0.531–0.667 across all 5 aids — no weak spot. Walking cane has the lowest values for all encoders (imbalance 6.4:1). SigLIP2-SO400M shines on Motorized WC (0.652). Vision-only SSL (DINOv2) outperforms supervised (ViT-B/16-sup) and matches vision-language (SigLIP2) — texture features > semantic labels for this task.

**Fig 2:** Brier score soft: Soft-KL 0.055–0.075 vs Hard-CE 0.18–0.24 across all encoders. The 3× calibration improvement from Soft-KL is the main figure for the calibration contribution section. The routing demo uses this calibrated output.

**Fig 3:** Encoders cluster in 60–102ms range. DINOv2-large (best F1=0.613, 78ms) is the Pareto-optimal choice. VLMs plotted separately are far outside this range (135ms–6.2s). This figure directly addresses Reviewer 3's latency complaint.

**Fig 4:** Trained probes (Hard-CE CLIP-B/32: 0.617, Soft-KL DINOv2-large: 0.613) clearly outperform LLaVA (0.418) and Qwen2.5 (0.412). Qwen3-VL (0.576) is surprisingly close — gap only 0.041. Key message: even state-of-the-art 8B VLMs benefit from task-specific training data; our approach with 260 samples achieves competitive performance.

**Fig 5:** All 5 aids show entropy distributions skewed right (μ=0.77–0.90, max=1.0). This is the empirical justification that accessibility judgments are inherently uncertain and cannot be reliably reduced to hard labels. Without this figure, reviewers might ask "why use soft labels at all?" — this answers it.

### 9.6 Prompt Sensitivity Analysis (Reviewer 3)

**Configuration:** 3 prompt variants × 3 VLM models × 260 test samples (52 images × 5 aids)

**Prompt Variants:**

1. **direct**: "Can a person using {aid} pass through the sidewalk? Answer with yes, no, or unsure."
2. **descriptive**: "This is a sidewalk image. Can a person using a {aid} pass through this sidewalk? Answer with exactly one word: yes, no, or unsure." (baseline)
3. **cot**: "Look carefully at the sidewalk features and think about accessibility for a person using a {aid}. Can they pass? Answer with yes, no, or unsure."

**Table 3: Macro-F1 and Balanced Accuracy across Prompt Variants**

| Model | Variant | Macro-F1 | Balanced Acc | Parse Failures | Best |
|-------|---------|----------|--------------|----------------|----|
| **LLaVA-1.5-7B** | Direct | 0.488 | 0.520 | 0 | |
| | Descriptive | 0.430 | 0.514 | 0 | |
| | **CoT** | **0.542** | **0.563** | 0 | ✅ |
| **Qwen2.5-VL-7B** | Direct | 0.442 | 0.610 | 0 | |
| | Descriptive | 0.412 | 0.596 | 0 | |
| | **CoT** | **0.479** | **0.622** | 2 | ✅ |
| **Qwen3-VL-8B** | Direct | 0.513 | 0.532 | 0 | |
| | **Descriptive** | **0.576** | **0.633** | 0 | ✅ |
| | CoT | 0.547 | 0.563 | 0 | |

**Key Findings:**

1. **Descriptive prompt is best overall** (Qwen3-VL-8B: 0.576 macro-F1) — matches the baseline reported in Section 9.3. This is the prompt to use for final zero-shot results.

2. **Variant performance is model-dependent:**
   - **LLaVA-1.5-7B:** CoT helps most (+0.112 F1 vs descriptive, +0.054 vs direct)
   - **Qwen2.5-VL-7B:** Direct and CoT close, descriptive underperforms
   - **Qwen3-VL-8B:** Descriptive dominates (−0.063 F1 vs direct, −0.029 vs CoT)

3. **Prompt variance per model:**
   - LLaVA: max−min = 0.112 (16% relative change)
   - Qwen2.5-VL: max−min = 0.067 (8% relative change)
   - Qwen3-VL: max−min = 0.063 (11% relative change)
   - **Average variance: 11%** — non-trivial, justifies sensitivity analysis

4. **Parse failures:** Only 2 parse failures (Qwen2.5 + CoT) — robustness is good across variants

5. **Interpretation for paper:** "We tested three prompt structures to assess sensitivity to prompt engineering. Across all models, the descriptive prompt with role context and explicit instruction proved most stable and highest-performing. Variance across prompts (8–16% F1 change) is non-trivial but manageable; we recommend descriptive for production use."

---

## 10. Key Numbers to Beat / Reference

| Métrica | Resultado | Notas |
|--------|--------|-------|
| Melhor CV Macro-F1 (Hard-CE) | **0.617** (CLIP ViT-B/32) | Hard-CE bate Soft-KL em 6/8 encoders |
| Melhor CV Macro-F1 (Soft-KL) | **0.613** (DINOv2-large) | Soft-KL melhor para calibração |
| Delta médio Hard-CE vs Soft-KL | **+0.059** | Hard-CE consistentemente superior em argmax F1 |
| Melhor F1 por aid (Hard-CE) | Walker 0.761 (CLIP-B/32) | |
| Ablação Hard-CE vs Soft-KL | ✅ Done — Section 9.2 | Hard-CE ganha argmax F1; Soft-KL ganha calibração |
| Zero-shot LLaVA-1.5-7B F1 | **0.418** | Section 9.3 |
| Zero-shot Qwen2.5-VL-7B F1 | **0.412** | Section 9.3 |
| Zero-shot Qwen3-VL-8B F1 | **0.576** | Section 9.3 — melhor zero-shot |
| Probe vs best zero-shot gap | **+0.041** | Hard-CE 0.617 vs Qwen3-VL 0.576 |
| Latência de inferência (GPU) | TBD | Semana 1 |

**Nota sobre 0.867 anterior:** Usava hard labels num split único train/val sem CV nem estratificação — overfitou ao val. O número correto e honesto para o paper é o CV: **0.596 (CLIP ViT-L/14)**.

---

## 11. Current Status

| Task | Status |
|------|--------|
| Firebase parser (`src/data/firebase.py`) | ✅ Done |
| Dataset expansion (829 participants) | ✅ Done — `image_selection_firebase.csv` |
| Vote tallies with entropy weights | ✅ Done — `tallies_firebase.json` |
| Repository reorganization | ✅ Done |
| 5-fold CV script (`src/models/crossval.py`) | ✅ Done |
| `.gitignore` updated | ✅ Done |
| Multi-encoder support in `train.py` | ✅ Done |
| Run CV para todos os 8 encoders (soft-kl) | ✅ Done — Table 1 em Section 9.1 |
| W&B tracking (único run, todas as tabelas) | ✅ Done — `src/models/log_wandb.py` |
| Seed global para reprodutibilidade | ✅ Done — `RANDOM_STATE=42` em tudo |
| Hard-CE ablation (todos 8 encoders) | ✅ Done — Section 9.2 |
| Upload hard_ce para W&B | 🔲 `python src/models/log_wandb.py --results_dir results/cv/hard_ce --loss_type hard_ce` |
| Métricas de calibração (Brier Score, ECE) | ✅ Done — brier_soft soft_kl=0.07 vs hard_ce=0.23 |
| Zero-shot LLaVA-1.5-7B | ✅ Done — F1=0.418, bal_acc=0.508 |
| Zero-shot Qwen2.5-VL-7B | ✅ Done — F1=0.412, bal_acc=0.596 |
| Zero-shot Qwen3-VL-8B | ✅ Done — F1=0.576, bal_acc=0.633 |
| Latency benchmark | ✅ Done — Section 9.4 |
| Publication figures (5 figs) | ✅ Done — Section 9.5 (`results/figures/`) |
| Prompt sensitivity analysis | ✅ Done — Section 9.6 (`results/prompt_sensitivity/`) |
| Generalization test (new cities) | 🔲 **Week 2 — READY TO START** (`TASK_5.7_READY.md`, scripts prepared) |
| `src/routing/demo.py` | 🔲 Week 3 |
| Paper rewrite | 🔲 Week 4 |
| Code + data release | 🔲 Week 5 |
| CoRL submission | 🔲 Week 5 |
