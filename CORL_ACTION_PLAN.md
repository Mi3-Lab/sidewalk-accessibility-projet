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
- [ ] Test 3 prompt variants per aid (direct, descriptive, chain-of-thought)
- [ ] Report F1 variance across prompts
- [ ] Pick best prompt per model for final table

#### 5.7 Generalization test on new cities
- [ ] Download 50–100 images from Project Sidewalk API (cities NOT in training set — e.g., Washington DC, Columbus OH)
- [ ] Run best linear probe model (expected: CLIP ViT-L/14 or DINOv2-large) on new images
- [ ] Compare predictions to Project Sidewalk structural labels (CurbRamp→accessible, NoCurbRamp→inaccessible) as proxy
- [ ] Report agreement rate as geographic transfer evidence
- [ ] Save results to `results/generalization/`

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
| LLaVA-1.5-7B | 0.464 | 0.388 | 0.402 | 0.422 | 0.414 | **0.418** | 0.508 |
| Qwen2.5-VL-7B | 0.376 | 0.610 | 0.391 | 0.269 | 0.414 | **0.412** | **0.596** |
| Qwen3-VL-8B | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Comparação com probes treinados:**

| Método | Overall Macro-F1 | Delta vs zero-shot |
|--------|-----------------|-------------------|
| Hard-CE best (CLIP ViT-B/32) | **0.617** | +0.199 vs LLaVA |
| Soft-KL best (DINOv2-large) | 0.613 | +0.195 vs LLaVA |
| LLaVA-1.5-7B zero-shot | 0.418 | baseline |
| Qwen2.5-VL-7B zero-shot | 0.412 | −0.006 vs LLaVA |

**Interpretação:**
- Probes treinados (+0.20 F1) justificam claramente o custo de coleta de dados e treinamento
- Zero-shot VLMs têm F1 baixo mas Qwen2.5 tem balanced_acc maior (0.596 vs 0.508) — menos biased para classe majoritária
- LLaVA tende a prever sempre a mesma classe (balanced_acc ≈ 0.5 = chance level por aid)
- Argumento do paper: **mesmo com 7B parâmetros e zero training, VLMs são inferiores ao nosso probe treinado com 260 amostras**

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
| Zero-shot Qwen3-VL-8B F1 | TBD | Rodar ainda |
| Probe vs zero-shot gap | **+0.199** | Justifica treinamento |
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
| Zero-shot Qwen3-VL-8B | 🔲 **PRÓXIMO** — `MODEL=qwen3-vl-8b ./run_zero_shot.sh` |
| Latency benchmark | 🔲 Semana 2 |
| Prompt sensitivity analysis | 🔲 Week 2 |
| Generalization test (new cities) | 🔲 Week 2 |
| `src/routing/demo.py` | 🔲 Week 3 |
| Paper rewrite | 🔲 Week 4 |
| Code + data release | 🔲 Week 5 |
| CoRL submission | 🔲 Week 5 |
