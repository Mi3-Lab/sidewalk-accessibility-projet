# CVPR 2027 — plan and status

Working document for the resubmission of *Human-Aligned Sidewalk Accessibility Perception*
after the CoRL 2026 rejection. Tracks what the reviewers asked for, what we decided to do
about it, and where each piece stands.

**Target:** CVPR 2027, Seattle. Estimated deadline **12–13 Nov 2026** (abstract ~7 Nov).
Dates are still unofficial — confirm on the CVF site once the CFP is posted.

Legend: **DONE** · **RUNNING** · **NEXT** · **HELD** · **OPEN**

---

## 1. Where we stand

CoRL 2026 submission #1080 was rejected (scores 2, 3, 3). Three reviewers, four
recurring objections:

| # | Objection | Raised by | Our response |
|---|-----------|-----------|--------------|
| R1 | Routing evaluation is circular: edge cost and success metric are both the model's own `p_yes` | all three | De-emphasise routing to a downstream demonstration; seek an independent validation signal |
| R2 | Only 52 unique images; single seed=42 CV is statistically unstable | all three | Multi-seed CV + semi-supervised expansion (§3) |
| R3 | Weak fit for CoRL as a robot-learning venue | DaKv, MfuG | Resolved for free by moving to CVPR |
| R4 | "Structural failure" framing of Hard-CE conflicts with Hard-CE winning macro-F1 on 6/8 encoders | M7xg | Reposition the qualifier, do not weaken the claim (§4) |
| R5 | Claimed 829 users vs ~163 contributors visible on Project Sidewalk | M7xg | Factually resolved (§2) — reviewer compared two different systems |

Reviewer M7xg additionally asked for repeated cross-validation with different seeds,
which is exactly what §3.1 delivers.

## 1b. Verdict on every claim in the paper

Everything below was re-measured over 10 seeds. This table is the short version; the
sections named in the last column carry the numbers.

| claim as published | verdict | where |
|---|---|---|
| 3.2× calibration improvement from Soft-KL | **artefact of an untuned hyperparameter.** Reproduces at the paper's `weight_decay=1e-4` (3.09×), but the ratio is a function of that free parameter: 3.92× down to 0.86× — Hard-CE winning — across the grid. **Tuned per method it is 1.50×** | §3.9 |
| soft Brier 0.076 | under-reports the method by half: **0.0376 achievable** with tuned regularisation | §3.9 |
| DINOv2-large leads Soft-KL (F1 0.613) | **does not replicate.** 0.475 ± 0.054 over 10 seeds, the *worst* of eight encoders; seed 42 was the best of ten draws | §3.1 |
| encoder ranking inverts between objectives | **does not replicate.** Top encoders are statistically tied (paired difference 0.0005 ± 0.0150) | §3.1 |
| walking cane +42% relative F1 | **does not replicate.** Rests on the same seed-42 draw | §3.1 |
| Hard-CE below trivial baseline on 6/8 encoders (cane) | **5/8** over 10 seeds | §3.1 |
| temperature scaling cuts Hard-CE Brier ~34%, gap stays >2× | **replicates** — 34.7% and 2.12× | §3.6 |
| routing improves 87.8–91.0% of trips | **reproduces exactly**, and survives an encoder swap (89.3–90.7%) | §3.5b |
| 91.7% edge coverage from real predictions | **confirmed**; the headline routing numbers do come from that file | §3.5b |
| geographic transfer, Zurich +11.6pp over chance | **does not survive an encoder swap** — all five cities fall to chance | §3.11 |
| model learns human uncertainty structure | entropy correlation **0.051**, and Hard-CE scores higher (0.133) | §3.10 |
| 829 mobility-aid users | **correct and verifiable**, though it is two collection waves (135 + 694), which the paper should say | §2, §2b |
| YOLOv12-seg area-ratio filter | **not reproducible from a clone** — the checkpoint is a 134-byte LFS pointer | §3.12 |

Two claims survive intact (temperature scaling, routing), one survives with a corrected
number (829 users), and the rest need rewriting or removal.

---

## 2. Data provenance — the 829 vs 163 question · **DONE**

Investigated in full. The reviewer's objection does not hold, and we now have the
evidence to say so in the paper.

- The 829 users / 49,490 votes are **exact and reproducible**: `data/processed/image_selection_firebase.csv`
  contains precisely 49,490 rows and 829 unique `PId`.
- The votes come from a **dedicated survey instrument** backed by Firebase/Firestore
  (project `sidewalk-survey-f7904`, collection `surveyAnswers0727`), *not* from Project
  Sidewalk's core labelling platform (`ProjectSidewalk/SidewalkWebpage`). The two have
  different participant populations and their counts are not comparable — this is the
  comparison the reviewer made.
- Independent corroboration: Li et al., CHI 2025 (arXiv:2502.19888) used the **same 52
  images** with **N=190** respondents. 829 / 190 = 4.36, matching the "4.4×" the paper
  already claims. The number is anchored to a peer-reviewed baseline.
- No sign of bot contamination: pool includes `wisc.edu`, `uw.edu`, `mail.utoronto.ca`
  alongside consumer domains; the "name+digits" email pattern is ~18–20%, normal for Gmail.

### 2b. The dataset is two collection waves, not one · **DONE**

Re-parsing the raw Firestore dump yields **40,252 votes / 694 participants**, not the
published 49,490 / 829. The gap is exact and explained: `firebase.py --merge` folds in
`data/processed/image_selection.csv` (dated Nov 2025), the pre-Firebase collection wave.

| wave | votes | participants |
|------|------:|-------------:|
| 1 — pre-Firebase (≤ Nov 2025) | 9,238 | 135 |
| 2 — Firebase survey | 40,252 | 694 |
| **published total** | **49,490** | **829** |

Both columns add up exactly. The provenance paragraph must therefore describe **two waves**,
not a single instrument — an earlier draft of that paragraph would have been subtly wrong.
Incidentally, wave 1's 135 participants is close to the ~163 the reviewer cited, which may
be where their number came from.

### 2c. Incomplete sessions: 829 vs 581 · **DONE** (decision still **OPEN**, but low-stakes)

`src/data/firebase.py::parse_doc()` read `logType` but never filtered on it, although its
docstring said incomplete "temp" sessions should be skipped. Of 96,876 raw documents,
**95,979 (99.1%) are `logType: "temp"`** — started and abandoned. `parse_doc` now takes
`keep_log_types`, defaulting to the historical behaviour so nothing changes silently, with
`--keep_log_types` exposed on the CLI.

Measured effect of restricting to completed sessions (`final` + `CompletedOneMobilityAid`),
Firebase wave only:

| | all sessions | completed only |
|---|---:|---:|
| votes | 40,252 | 35,674 |
| participants | 694 | 581 |
| median votes per (image, aid) | 149 | 134 |

**The soft labels barely move**: mean `|Δp_yes|` = 0.0138, max 0.0515, Pearson r = 0.9931,
and only 7 of 260 pairs change argmax. So the choice is presentational, not scientific —
but it has to be stated either way. Recommendation: keep the larger count and define a
participant explicitly as someone who completed at least one aid group, since the labels
are demonstrably unaffected.

The 99.1% abandonment rate is also hard evidence for §5: people do not finish long unpaid
rating sessions.

---

## 3. Working beyond the 52 images

The core scientific problem. Two independent attacks, neither of which needs funding or
a large recruitment campaign.

### 3.1 Repeated-seed cross-validation · **DONE** (16/16 configs, 10 seeds)

> **Read this before touching the paper.** The headline calibration claim survives and is
> now much better supported. Three secondary claims do not survive, including one the
> routing pipeline depends on.

**Survives — and is now robust rather than anecdotal:**

| | seed 42 (paper) | 10 seeds (42–51) |
|---|---|---|
| Brier Soft-KL (mean over encoders) | 0.076 | **0.0808 ± 0.0098** |
| Brier Hard-CE | 0.244 | **0.2497 ± 0.0116** |
| **ratio** | **3.2×** | **3.09×** |

The ratio holds on every one of the 8 encoders (2.83×–3.36×), and the across-seed spread
(±0.008–0.013) is an order of magnitude smaller than the gap itself. This is the paper's
central claim and it is no longer resting on one draw.

**Does not survive:**

1. **DINOv2-large's Soft-KL result was a lucky seed.** Walking-cane F1 per seed:
   `[0.640, 0.399, 0.536, 0.475, 0.534, 0.481, 0.500, 0.460, 0.366, 0.376]` — mean
   **0.477 ± 0.079**. Seed 42 produced the single highest value of the ten. The published
   `0.640` and the "+42% relative" headline rest entirely on that draw.
2. **Overall Soft-KL F1 for DINOv2-large:** 0.613 published → **0.475 ± 0.054**, now the
   *worst* of the eight encoders.
3. **"The ranking inverts between objectives"** (DINOv2-large leads Soft-KL, CLIP-B/32
   leads Hard-CE) does not replicate. Over 10 seeds the best Soft-KL encoder is
   **SigLIP2-SO400M** on both Brier (0.0761) and F1 (0.521).
4. **"6/8 encoders below the trivial baseline under Hard-CE"** on walking cane → **5/8**.

**Consequence for the routing pipeline.** DINOv2-large was chosen for every downstream
routing experiment *because* it led Soft-KL. That justification is gone, so the routing
numbers have to be recomputed under a defensible encoder.

**But do not simply crown a new winner — that is the same mistake.** Per-seed paired
comparison of Soft-KL Brier:

| | |
|---|---|
| best encoder per seed | CLIP-L/14 wins 4/10, SigLIP2-SO400M 3/10, three others 1 each |
| SigLIP2-SO400M vs CLIP-L/14, paired | +0.0005 ± 0.0150 |

They are indistinguishable. Encoder-level Brier differences are ~0.005, while the binomial
standard error on the training target itself is 0.035 (95% CI **±0.068**) — the whole
encoder table compares differences an order of magnitude below label noise.

The honest treatment is a **statistically tied top group**, with the downstream encoder
picked on stated grounds (latency, or simply declared) and, ideally, a demonstration that
the routing result is *insensitive* to which member of the tied group is used. That is a
stronger claim than "we used the best one".

Note the latency knock-on: SigLIP2-SO400M runs at 101.6 ms/image vs DINOv2-large's 78.5 ms,
so the abstract's "78.5 ms, ~75× faster than the closest zero-shot VLM" changes if the
encoder changes.

**What replaces the C2 claim.** Soft-KL still beats Hard-CE on walking-cane F1 on **7 of 8
encoders** (only DINOv2-base flips), consistently across seeds. That is a real, replicable
effect and a fair statement of the contribution. The "+42%" number has to go.

The reviewer's instability objection was, on this specific point, correct.

### 3.1b How the repeated-seed CV was built

Directly answers M7xg's request for multiple seeds, using no new data.

Changes to `src/models/crossval.py`:
- **Feature cache.** The encoder is frozen, so an image's features never change, yet the
  code re-extracted them for every fold (5 aids × 5 folds × 2 splits = 50 extractions per
  encoder). Each unique image is now encoded once. This is what makes 10 seeds affordable.
- **`--seed` / `--n_seeds`.** Re-runs the whole CV over independent fold partitions and
  writes `cv_results_multiseed.json` with mean±std across seeds.

**Regression check passed.** At seed=42 the `macro_f1` reproduces the stored results
*exactly* (difference 0.000000 on all five aids, CLIP-B/32); Brier matches to four decimals
(0.0644 vs 0.0645 stored). Residual differences (~1e-5, and 0.016 on Walker ECE) come from
batching during feature extraction — ECE is binned, so a tiny shift moves a sample across a
bin edge. No number changes at the precision the paper reports. `run_cv.sh` behaviour is
unchanged (defaults are seed=42, n_seeds=1).

Results land in `results/cv_multiseed/` — the published `results/cv/` numbers are untouched.

Gather with:
```bash
python src/models/summarize_multiseed.py --results_dir results/cv_multiseed --latex
```

### 3.2 Semi-supervised expansion · **RUNNING**

`src/models/semisup_expand.py`. Trains on the real votes, pseudo-labels the 484-image
Project Sidewalk pool (`data/generalization/`, 5 cities), filters, retrains, and scores on
**held-out real human votes**. Evaluation never touches the pseudo-labels, so this does not
repeat the circularity the reviewers flagged in the routing evaluation.

Two filters gate a pseudo-label:
1. **Entropy** — the predicted distribution must be confident enough.
2. **Project Sidewalk agreement** — the prediction must not contradict the PS annotator's
   CurbRamp/NoCurbRamp label for that image. This is a *human* signal independent of our
   model, and it is what separates this from plain self-training.

**Result: it does not work, and the reason is the paper's own contribution.**

| config | ΔBrier | ΔF1 | pseudo-labels kept (of 484) |
|--------|-------:|----:|----:|
| DINOv2-large, entropy ≤ 0.60 | +0.0000 | +0.0042 | 11 |
| CLIP-B/32, entropy ≤ 0.60 | −0.0023 | +0.0044 | 38 |
| SigLIP2-SO400M, entropy ≤ 0.60 | −0.0004 | −0.0061 | 22 |
| entropy ≤ 0.30 | +0.0000 | +0.0000 | **0** |
| entropy ≤ 0.75 | +0.0002 | −0.0008 | 47 |
| entropy ≤ 0.90 | +0.0001 | +0.0013 | **166** |
| no PS-consistency filter | −0.0000 | +0.0044 | 18 |

Every delta is within noise, on all nine configurations.

**Why — and it is not what it first looks like.** The obvious reading is that the filters
are too strict: 2–10% acceptance, zero at a strict threshold. The entropy ≤ 0.90 row rules
that out. With **15× more pseudo-labels admitted (166)** the result is still unchanged. And
dropping the PS-consistency filter adds only 7 labels, so the entropy threshold — not the
PS agreement check — was doing the limiting all along.

The actual mechanism is more basic: **self-training with a linear probe on frozen features
is close to a no-op by construction.** The pseudo-labels are generated by the probe's own
decision function, so each added pair (x, f(x)) is a point where the model is already
optimal and contributes almost nothing to the gradient. Unlabelled data helps when the model
has capacity to be *constrained* by it — consistency regularisation, entropy minimisation
during encoder fine-tuning. A linear head over fixed features has no such capacity.

Making unlabelled data pay off here would mean unfreezing the encoder, which changes the
paper's whole setup (and its efficiency and latency claims). Out of scope for November.

The finding is worth a short paragraph in the paper — it is a genuine negative result about
a method reviewers might otherwise ask for. But **semi-supervised expansion does not answer
R2.** The response to the dataset-size objection rests on §3.1 (repeated-seed CV), honest
repositioning (§3.4), and active learning as forward-looking work (§3.3), not on this.

### 3.3 Active learning · **DONE** (implemented and smoke-tested)

`src/models/active_select.py`. Instead of asking for hundreds of new votes, rank which
few images are worth a real person's time. Four strategies: `coverage` (greedy k-centre
over frozen-encoder feature space, seeded with the 52 labelled scenes), `entropy`,
`aid_spread` (disagreement between the five aid probes), and `hybrid`. Outputs a ranked
CSV plus a `trials.json` fragment in the schema the study app already consumes, so a
shortlist can go straight in front of participants.

**The coverage criterion independently rediscovers the paper's own weak spot.** Asked for
12 images (CLIP-B/32 smoke test), it picked Zurich 4× and Taipei 3× — while Pittsburgh,
which is 33% of the pool, got only 2. The 52 labelled scenes are mostly Western/US, so the
feature-space frontier sits exactly where geographic transfer already failed: Taipei scored
0.518 balanced accuracy, barely above chance. Coverage-based acquisition and the
generalization experiment point at the same gap, which is worth saying in the paper.

Should be re-run under whichever encoder wins the §3.1 re-evaluation.

### 3.5b What the routing re-run turned up · **IN PROGRESS**

The SigLIP2 re-run finished. Three things came out of it, one of which is a mistake I made
and corrected, and two of which are real problems with the published routing result.

**My error, for the record.** I first compared the SigLIP2 Monte Carlo (58.5% of trips
improved for walking cane) against the paper's 87.8% and concluded the routing claim
collapses under an encoder change. That was the wrong baseline. The repo's own DINOv2 run at
`results/routing/monte_carlo/` reports **60.6%**, not 87.8%. Matched against that, changing
the encoder costs about two points, not a collapse. The GSV coverage was also slightly
*better* in my run (1387 inference edges, 53.0%, vs 1344 / 51.3%), so the comparison is fair
on that axis.

**Problem 1: the abstract's number is a point on a tunable curve.** The barrier-cost sweep
already in the repo:

| β_c | 1 | 2 | 4 | **8** | 16 |
|---|---|---|---|---|---|
| walking cane % improved | 68.9 | 78.1 | 83.8 | **87.8** | 90.4 |
| distance overhead | +1.4% | +2.9% | +5.0% | **+7.4%** | +10.5% |

Anything between 69% and 90% is purchasable by moving one hand-set parameter. The paper
justifies β_c=8 as the Pareto inflection, but a reviewer will read the sweep and see a knob.

**Problem 2 — resolved, and the paper comes out clean on it.** Re-running both candidate
edge-score inputs under the paper's graph and β_c settled which produced the published
numbers:

| edge scores | cane % | manual % | Δp cane | Δdist |
|---|---:|---:|---:|---:|
| `edge_scores_ps_956.json` (1057 PS + 1561 priors) | 84.8 | 87.3 | 0.0406 | +4.94% |
| **`edge_scores_full.json`** (1344 GSV inference) | **87.8** | **90.9** | **0.0647** | **+7.43%** |
| paper / `monte_carlo_956_bc8` | 87.8 | 90.9 | 0.0647 | +7.43% |

Exact reproduction from the dense GSV-inference scoring, consistent with the README's 91.7%
coverage claim. My earlier suspicion that the coverage claim and the headline number came
from different configurations was **wrong, and is withdrawn**.

**Problem 3 (the real one): there is a broken run sitting in `results/routing/`.**
The two graphs are different areas:

| graph | nodes | edges | centre |
|---|---:|---:|---|
| `pittsburgh_graph.graphml` | 1524 | 1920 | (40.4444, −79.9457) Oakland |
| `pittsburgh_graph_956.graphml` | 2046 | **2618** | (40.4437, −79.9551) |

Every `edge_scores*.json` has 2618 entries, keyed to the **956** graph. So
`results/routing/monte_carlo/` (60.6%) paired those scores with the 1920-edge Oakland graph:
most edges never matched and silently became constant per-aid priors. Those numbers carry
almost no model signal and should not be used or cited.

**My SigLIP2 re-run inherited the same bug**, because I copied the graph path from that
directory. So the "changing the encoder costs about two points" conclusion above compared two
equally broken runs and is void.

**Redone correctly on the 956 graph (100% edge match) — and the headline holds:**

| aid | DINOv2-large | SigLIP2-SO400M | Δ |
|---|---:|---:|---:|
| walking cane | 87.8% | 89.9% | +2.1 |
| walker | 88.2% | 89.3% | +1.1 |
| mobility scooter | 91.0% | 89.3% | −1.7 |
| manual wheelchair | 90.9% | 89.3% | −1.6 |
| motorized wheelchair | 90.8% | 90.7% | −0.1 |

SigLIP2 lands at 89.3–90.7%, inside the paper's published 87.8–91.0% band. **The routing
conclusion does not depend on the encoder choice that turned out to be a seed-42 fluke.**

This is the stronger claim flagged in §3.1: rather than "we used the best encoder" (which the
data does not support, since the top encoders are statistically tied), the paper can say the
routing result is *insensitive* to which member of the tied group is used, and show this
table. That converts a weakness into a robustness result.

Honest caveat: the *magnitude* of improvement is smaller under SigLIP2 (mean Δp_yes
0.057–0.095 vs 0.065–0.120). What is stable is the fraction of trips improved, which is the
number the abstract quotes.

**Fixed along the way**, both worth keeping regardless of the paper:
- `monte_carlo.py` records its inputs (`edge_scores`, `graph_cache`, edge source counts) in
  `summary.json`, so a published number can be traced back to its configuration.
- It now **hard-fails below 50% edge match** and warns below 90%, instead of quietly
  substituting priors. That is exactly the failure that produced the bad directory.
- `run_routing_encoder.sbatch` points at the 956 graph.

### 3.5 Routing re-run under a defensible encoder · **RUNNING**

`run_routing_encoder.sbatch`, parameterised by `ENCODER`, writing to
`results/routing_<encoder>/` so the published DINOv2-large routing results stay intact for
comparison. Steps: final probes → GSV edge scoring → per-aid routes → Monte Carlo.
The PS-label edge scores are encoder-independent and are reused.

While wiring this up, found that `score_gsv_edges.py` cached GSV thumbnails **in memory
only**. Re-running under a second encoder would therefore re-download every thumbnail and,
worse, could get a *different* set of images as panoramas expire — which would confound the
encoder comparison. Added a persistent disk cache (`--pano_cache_dir`, default
`cache/panos`), so both encoders score identical pixels. Expired panos get a marker file so
they are not retried.

### 3.6 Does the temperature-scaling claim replicate? · **DONE — yes**

| | paper (seed 42) | 10 seeds × 8 encoders |
|---|---|---|
| Hard-CE Brier reduction from T-scaling | ~34% | **34.7%** |
| remaining gap to Soft-KL | >2× | **2.12×** |

Both hold. This matters defensively: it is the evidence that the Soft-KL advantage is
**structural** (how the target is encoded) rather than mere overconfidence that post-hoc
calibration could fix. It pre-empts the obvious reviewer question, "why not just calibrate
your Hard-CE model?"

Fitted temperatures run 1.9–2.65, rising with encoder capacity.

### 3.6b Original notes on the temperature-scaling check

The paper's other single-seed claim: post-hoc temperature scaling cuts Hard-CE Brier by
~34% but leaves a >2× gap to Soft-KL. Given that one single-seed claim already failed
(§3.1), this one needed checking too. `temperature_scaling.py` now takes `--seed`/
`--n_seeds` and uses the same feature cache.

Early signal is **good**: a 2-seed CLIP-B/32 smoke test gives a **33.3%** reduction against
the claimed ~34%. Full 8-encoder × 10-seed run is queued.

### 3.4 Positioning · **HELD** (until experiments finish)

Frame the dataset in the tradition of small, densely-rated stimulus sets (psychophysics,
affective computing), not as a competitor to ImageNet-scale vision benchmarks. Costs
nothing, but should be written once we know what §3.1–3.3 actually produced.

---

## 3.7 A SOTA framing for CVPR · **PROPOSED** (awaiting go-ahead)

Deep literature search turned up work that changes how this paper should be positioned, and
the data supports three technical contributions that follow from it.

### The literature

- **[Metric-Dependent Annotation Saturation](https://arxiv.org/abs/2605.29797)** — how many
  annotators you need depends on the metric: KL saturates at N≈10, entropy correlation needs
  N≈20–50. Soft labels beat label smoothing decisively (entropy correlation r=0.643 vs
  0.45–0.49).
- **[Soft-Label Training Preserves Epistemic Uncertainty](https://arxiv.org/html/2511.14117v1)**
  — 32% lower KL, 61% stronger entropy correlation, and explicitly **"matching hard-label
  accuracy"**. Benchmarks: ChaosNLI 100 annotations/item, CIFAR-10H 50, POPQUORN 6.7.
- **[SNEFY-LDL](https://arxiv.org/abs/2412.07324)** — distributions over label distributions
  on the simplex (partial prior art for the count-likelihood idea below).
- **[Confidence Calibration under Ambiguous Ground Truth](https://arxiv.org/pdf/2603.22879)**,
  **[Dirichlet-Based Prediction Calibration](https://arxiv.org/abs/2401.07062)**,
  Uma et al., *Learning from Disagreement: A Survey* (JAIR 2021).

Two immediate consequences:

1. **This is not a small dataset — it is the densest one.** ~190 votes per (image, aid),
   ~950 per image across the five aids, against ChaosNLI's 100 and CIFAR-10H's 50. Far past
   the saturation point on every metric in the literature. The 52 scenes are the *cost* of
   that density, not an oversight.
2. **Reviewer M7xg's F1 objection is a published expectation.** "Matching hard-label
   accuracy" is what soft-label training is supposed to do. The paper was reproducing a known
   result without citing it. That reframes the objection from "your claim is wrong" to "this
   is the established behaviour".

### What the data supports (measured)

| | |
|---|---|
| cross-aid correlation of p_yes | **0.886** (0.824–0.946) |
| binomial SE of the training target | 0.035 → **95% CI ±0.068** |
| Brier differences between encoders | **~0.005** |
| votes per (image, aid) | 141–240, median 188 |
| images respecting a full aid ordering | 4/52 (cane dominates; the rest interleave) |

### Three contributions that follow

1. **Count-based likelihood instead of KL to a point estimate.** The target is an empirical
   distribution treated as exact, but it carries ±0.068 and the vote count varies 141–240 per
   item. A Dirichlet-multinomial likelihood over the raw counts weights each target by its
   actual precision and replaces the ad-hoc entropy `sample_weight` with a statistically
   correct one. Partial prior art exists; the novelty here is real per-item count variation.
2. **Cross-condition coupling — the genuinely novel piece.** Five independent 52-point probes
   discard a 0.886 correlation. A low-rank / shared-trunk model with aid embeddings pools the
   data. No other human-uncertainty benchmark has multiple coupled conditions per item, so
   this comes from the dataset rather than borrowed method.
3. **Noise-aware evaluation.** Report which differences exceed label uncertainty. This is
   what would have caught the DINOv2-large error, and it is exactly the rigour the reviewers
   were reaching for.

Plus a **missing metric**: entropy correlation (model uncertainty vs human disagreement),
treated as central in the literature and uniquely well-estimated at this annotation density.

### 3.8 Coupled probe — first numbers · **RUNNING** (full run queued)

`src/models/coupled_probe.py` implements §3.7's first two ideas and evaluates all
combinations under one harness with a shared fold partition per seed, so comparisons are
paired.

Smoke test (CLIP-B/32, 3 seeds, paired against `soft_kl/independent`):

| variant | Brier | Δ vs baseline | wins | F1 |
|---|---:|---:|---:|---:|
| multinomial / lowrank | 0.0566 | **−0.0297** | 3/3 | 0.561 |
| dirichlet_mult / lowrank | 0.0578 | −0.0286 | 3/3 | 0.542 |
| soft_kl / lowrank | 0.0608 | −0.0256 | 3/3 | 0.539 |
| multinomial / independent | 0.0852 | −0.0012 | 2/3 | 0.427 |
| soft_kl / independent | 0.0864 | — | — | 0.465 |

Almost all of the gain comes from the **head**, not the loss: cross-aid coupling moves Brier
by ~0.026–0.030 while the count-based losses add ~0.001–0.004 on top. F1 improves too, so
this is not calibration bought at the cost of accuracy.

### 3.8b Full results — both of §3.7's ideas are null, and a third thing is real

**Cross-aid sharing is dead.** The `lowrank_peraid` control (same bottleneck, no sharing)
settles it across all four encoders:

| encoder | independent | lowrank (shared) | peraid (no sharing) | gain from capacity | gain from sharing |
|---|---:|---:|---:|---:|---:|
| dinov2-large | 0.0828 | 0.0552 | 0.0553 | 0.0275 | 0.0001 |
| clip-vit-b32 | 0.0827 | 0.0586 | 0.0608 | 0.0220 | 0.0022 |
| siglip2-so400m | 0.0756 | 0.0549 | 0.0540 | 0.0217 | −0.0009 |
| clip-vit-l14 | 0.0744 | 0.0530 | 0.0511 | 0.0233 | −0.0018 |

Sharing contributes −0.0018 to +0.0022 — noise, negative on two of four encoders. The 0.886
correlation is real but a shared subspace extracts nothing a per-aid bottleneck does not
already get. **Capacity control is the entire effect.** Without the control this would have
been written up as "cross-condition coupling", and it would have been false.

**The count-based losses are null too.** Within the same head they differ by ≤0.003 with
inconsistent sign across encoders. So §3.7's idea (1) also fails.

**What is real: the probe is badly under-regularised.** `PROBE_WD = 1e-4` on a d→3 map fit to
52 points is close to no regularisation. A rank-16–64 bottleneck cuts soft Brier ~26–31% and
*raises* macro F1 on every encoder — a Pareto improvement, not a trade. The rank sweep shows a
broad plateau (r=16: 0.0552, r=32: 0.0542, r=64: 0.0547; r=4 too tight at 0.0803), so it is
not a knife-edge hyperparameter.

**This is not novel.** Regularised probes on frozen features are well-covered
([RAPTOR](https://arxiv.org/pdf/2602.00158), [ridge logistic on small data](https://arxiv.org/pdf/2101.11230)).
It is a fix to the pipeline, not a contribution to claim. A weight-decay sweep is queued to
check whether plain L2 gets the same thing, in which case the fix is one hyperparameter.

### 3.8c The consequence that matters: the headline ratio gets *better*

Re-measuring Soft-KL vs Hard-CE under the bottleneck:

| encoder | ratio, independent | ratio, lowrank |
|---|---:|---:|
| dinov2-large | 3.16× | **4.59×** |
| clip-vit-b32 | 2.80× | **3.55×** |
| siglip2-so400m | 3.40× | **4.57×** |
| clip-vit-l14 | 3.44× | **4.45×** |
| **mean** | **3.20×** | **4.29×** |

The bottleneck helps Soft-KL (−30% Brier) far more than Hard-CE (−6%), because Hard-CE's
error is dominated by systematic overconfidence that regularisation cannot fix, while
Soft-KL's is variance from over-parameterisation, which it can. **The paper's central claim
strengthens from 3.2× to 4.29×.**

The F1 gap also narrows, which speaks to reviewer M7xg's objection: from −0.162 to −0.063 on
average, and to −0.016 on CLIP-L/14.

**Harness validation.** The independent-head ratio came out at 3.20×, matching the paper's
published 3.2× exactly. That substantially resolves the baseline-fairness worry below for the
*ratio*. It does not resolve it for absolute F1: the harness's soft-KL F1 is weaker than the
real pipeline's (gap −0.162 vs −0.067 measured by `crossval.py`), so absolute F1 numbers from
this harness must not be quoted.

**Therefore:** `--rank` is now wired into the real `crossval.py` / `train.py` path (default 0,
preserving every published number), and a full 8-encoder × 2-loss × 10-seed run at rank 32 is
in flight. Those are the numbers that can go in the paper.

**Original caveat, kept for the record.** The in-harness `soft_kl/independent` is a
reimplementation: one optimiser over all five probes, plain KFold, loss averaged over 260 rows
rather than five separate 52-row fits with StratifiedKFold. Deltas are trustworthy *within*
the harness; anything quoted externally has to come from the real pipeline.

## 3.9 The 3.2× is an artefact of an untuned hyperparameter · **DONE**

This is the most consequential result of the whole review, and it supersedes §3.8c.

`PROBE_WD = 1e-4` on a d→3 map fit to 52 points is close to no regularisation, and it was
never tuned. Sweeping it over 12 values, 10 seeds each:

| weight decay | Soft-KL | Hard-CE | ratio |
|---|---:|---:|---:|
| **1e-4 (the paper's)** | 0.0828 | 0.2616 | **3.16×** |
| 1e-2 | 0.0536 | 0.2350 | 4.38× |
| 3e-2 | 0.0425 | 0.1896 | 4.46× |
| **3e-1 (best for Soft-KL)** | **0.0372** | 0.1074 | 2.88× |
| 1e0 | 0.0392 | 0.0732 | 1.87× |
| **1e1 (best for Hard-CE)** | 0.0659 | **0.0515** | 0.78× |
| 1e2 | 0.1032 | 0.0976 | 0.95× |

**The ratio is a function of the regularisation strength, which is a free parameter.**
Anywhere from 4.46× to 0.78× — Hard-CE winning — is reachable by moving one number the paper
never tuned. A reviewer with an afternoon and a sweep can produce whichever answer they like.

**Confirmed in the real pipeline** (`crossval.py`, all 8 encoders, 10 seeds, full grid) — these
are the numbers to quote, not the harness's:

| weight decay | Soft-KL | Hard-CE | ratio |
|---|---:|---:|---:|
| **1e-4 (the paper's)** | 0.0808 | 0.2497 | **3.09×** |
| 3e-1 | 0.0412 | 0.1617 | 3.92× |
| **3e0 (best Soft-KL)** | **0.0376** | 0.0973 | 2.59× |
| 1e1 | 0.0412 | 0.0766 | 1.86× |
| 3e1 | 0.0488 | 0.0596 | 1.22× |
| **1e2 (best Hard-CE)** | 0.0657 | **0.0564** | **0.86×** |

> **Fair ratio in the real pipeline: 1.50×** (0.0564 / 0.0376), against 3.2× published.

Caveat: Hard-CE's optimum sits at the grid edge (1e2), so the fair ratio could be marginally
lower still; the 3e1→1e2 gain is already small (0.0596→0.0564), so 1.50× is likely close to
final.

The same experiment in the coupled-probe harness gave fair ratios of 1.38×, 1.40× and 1.28×
on three encoders — consistent with the real pipeline despite a different training loop,
which is reassurance that the effect is not an artefact of either implementation.

Three consequences:

1. **Soft-KL's advantage is real but ~1.4×, not 3.2×.** It survives proper tuning, which the
   3.2× does not. A defensible 1.4× is worth more than a 3.2× that collapses under scrutiny.
2. **The two optima sit 30× apart** (3e-1 vs 1e1), so any single shared setting is unfair to
   one method, and 1e-4 is close to the worst case for Hard-CE — whose error at that setting
   is dominated by overconfidence that L2 shrinkage would have fixed.
3. **The paper under-reports its own method by half.** Best achievable Soft-KL Brier is 0.0372
   against 0.076 published.

The low-rank bottleneck (§3.8b) is subsumed: plain L2 at 3e-1 (0.0372) beats the best
bottleneck result (0.0398), so no architectural change is warranted.

## 3.10 Entropy correlation: the model does not track disagreement · **DONE**

Added `entropy_correlation` to `crossval.py` (Pearson r between predicted and human vote
entropy, the metric the current soft-label literature leads with). Across 8 encoders,
10 seeds:

| | Soft-KL | Hard-CE |
|---|---:|---:|
| rank 0 | **0.051** | 0.133 |
| rank 32 | 0.094 | 0.119 |

Literature reference for soft-label training: r ≈ 0.643.

Two problems. The correlation is near zero in absolute terms, and **Hard-CE tracks item-level
disagreement better than Soft-KL** — the opposite of the paper's thesis.

Checked whether the dataset simply has no headroom: human entropy is uniformly high and
narrow (mean 0.847 normalised, IQR 0.805–0.923, no easy items, unlike CIFAR-10H), but
sampling noise is small (0.0335 vs between-item spread 0.103, reliability 0.90). So there
*was* signal to track and the model does not track it. This should be reported as a limitation
rather than left for a reviewer to find.

## 3.11 Geographic transfer does not survive an encoder swap · **DONE**

Re-running the 5-city transfer evaluation under SigLIP2-SO400M:

| city | DINOv2-large (published) | SigLIP2-SO400M |
|---|---:|---:|
| Zurich | 0.616 | 0.508 |
| Detroit | 0.570 | 0.469 |
| Pittsburgh | 0.566 | 0.517 |
| Taipei | 0.518 | 0.500 |
| LA | 0.507 | 0.531 |

Every city drops to chance. Unlike the routing result, which survived the same swap, this
claim is encoder-dependent and does not hold.

## 3.12 A hypothesis that explains both failures, and a fix under test · **RUNNING**

The probe receives one globally pooled vector per image. Measured on the training set, the
walkable surface covers only **0.45** of each image, so more than half the probe's input is
sky, buildings, vehicles and vegetation. That would explain both §3.10 and §3.11: a model
keying on scene appearance transfers badly across cities (Zurich does not look like Taipei)
and has no reason to know which surfaces are contested.

`src/models/masked_features.py` pools encoder patch tokens over the walkable region only,
using an off-the-shelf Cityscapes SegFormer. Both variants are written to one file and
`crossval.py` takes `--features_npz/--feature_key`, so whole-image and mask-pooled runs
differ in nothing but what the probe sees.

Note: the repo's own segmentation checkpoint cannot be used — `checkpoints/yolo/bestv12.pt`
is a 134-byte Git LFS pointer, not the 380 MB model, and git-lfs is not installed. The
segmentation filter described in the paper's method is **not reproducible from a clone**.
A public segmenter is the better dependency anyway.

**Result: null in-domain, and inconsistent across encoders.**

| encoder | features | Brier | F1 | H-corr |
|---|---|---:|---:|---:|
| dinov2-large | whole | 0.0766 | 0.514 | 0.064 |
| dinov2-large | masked | 0.0776 | **0.532** | **0.106** |
| clip-vit-l14 | whole | 0.0792 | **0.526** | 0.085 |
| clip-vit-l14 | masked | 0.0859 | 0.501 | 0.085 |

Masking helps DINOv2 slightly on F1 and entropy correlation and hurts CLIP-L/14 on both.
Brier moves within noise either way. So restricting the probe to the walkable surface does
not help when train and test come from the same cities — which is defensible: scene context
is legitimately informative in-distribution.

The hypothesis was about *transfer*, so the honest test is leave-one-city-out
(`src/models/leave_one_city.py`): the 52 panoramas span nine cities, each fold holds one out
entirely, scored on the real soft labels rather than the binary CurbRamp proxy. Running.
Expectations should be low given the in-domain inconsistency.

## 3.12b The pattern across every idea tried

| idea | verdict |
|---|---|
| cross-aid coupling (r = 0.886) | null — the control showed it was capacity, not sharing |
| count-based losses (multinomial, Dirichlet-multinomial) | null — ≤0.003, sign inconsistent |
| low-rank bottleneck | subsumed — plain L2 does better |
| semi-supervised pseudo-labelling | null — structural, self-training on a linear head is a no-op |
| sidewalk mask pooling | null in-domain, inconsistent across encoders |
| **tuned weight decay** | **real and large** — halves Brier, but dismantles the 3.2× |

Five of six are null. That is not bad luck: with 52 items and a binomial CI of ±0.068 on the
training target itself, almost any apparent gain is within noise until a control rules the
alternatives out. **Every time an apparent gain was tested against a proper control, it
shrank or disappeared** — including the paper's own headline.

This is the argument for the reality-check framing in §3.13. The defensible contribution is
not a better method; it is a rigorous demonstration that in this regime most reported gains
do not survive controls, with a dataset dense enough to measure that properly.

## 3.13 A framing that turns the bad news into the paper · **PROPOSED**

Most of the paper's claims did not survive (§3.9–3.11). Rather than patch them, there is a
framing in which those findings *are* the contribution, with a proven precedent at a top
venue.

**The precedent.** *A Metric Learning Reality Check* (Musgrave, Belongie, Lim, ECCV 2020)
took a field that had "consistently claimed great advances, often more than doubling the
performance of decade-old methods", found the improvements were "marginal at best" once
evaluation was made fair, and was accepted on exactly that. Our §3.9 is the same shape: a
claimed 3.2× that becomes 1.4× when each method is tuned at its own optimum.

**Why it is not enough on its own.** Musgrave et al. audited a whole field across many
papers. We would have one paper — our own — and one task. Too narrow for CVPR.

**What closes the gap.** Test whether the finding holds on the benchmark the soft-label
literature is actually built on. CIFAR-10H is public and directly downloadable
(`cifar10h-counts.npy`, 10000×10 raw counts). If the soft-label advantage there also shrinks
under per-method regularisation tuning, this stops being a self-correction and becomes a
field-level result: *how much of the soft-label advantage survives fair tuning?*

`src/models/cifar10h_check.py` runs exactly the sidewalk protocol on CIFAR-10H: frozen
encoder, linear probe, Soft-KL vs Hard-CE, weight-decay sweep, repeated seeds, soft Brier and
entropy correlation.

**And the dataset turns out to occupy an unexplored regime**, which is a second contribution
and explains our own negative entropy-correlation result:

| | CIFAR-10H | this dataset |
|---|---:|---:|
| annotations per item | ~51 | **~190** |
| mean normalised entropy | **0.067** | **0.847** |
| IQR | 0.000–0.084 | 0.805–0.923 |
| items at total consensus | **4393 / 10000** | **none** |

CIFAR-10H is mostly easy with a minority of ambiguous items, so entropy has enormous dynamic
range — which is why entropy correlation reaches 0.64 there and 0.05 here. Existing
soft-label results are substantially about learning an easy/hard axis. This dataset removes
that axis: everything is contested. The scientific question becomes *do soft labels still
help when nothing is easy?* — which the literature has not asked.

**The paper this supports:**
1. A benchmark in a regime no existing human-uncertainty dataset covers (densest annotation,
   uniformly high disagreement, five coupled conditions per item).
2. A reality check: the soft-label calibration advantage is substantially a hyperparameter
   artefact, replicated on a public benchmark, with the fair number reported.
3. A characterisation of *when* soft labels help, with the routing result as the case where
   the calibrated output demonstrably matters downstream.

Every part is built from work already done. Nothing here requires new annotation.

## 3.14 Calibration does not reach the planner · **DONE** — and this one is positive

The paper argues calibration is the operative metric *because* the routing objective is
linear in p_yes. Tested directly: score the same 2000 OD pairs twice, once with Soft-KL edge
scores and once with Hard-CE, and measure how much the resulting routes differ.

Edge scoring was first extended to keep the full `[p_no, p_unsure, p_yes]` (it previously
stored p_yes alone, discarding two thirds of the model output before the planner saw it).
The two score files differ on exactly the 1387 inference edges — e.g. walking cane
`[0.264, 0.090, 0.646]` under Soft-KL against `[0.019, 0.000, 0.981]` under Hard-CE — while
the 1231 PS-label and prior edges are shared, so any divergence is attributable to the model.

| objective | Jaccard, Soft-KL vs Hard-CE routes | identical routes |
|---|---:|---:|
| **linear (the paper's)** | **1.0000** | **100%** |
| bottleneck (tail-sensitive) | 0.5688 | 29.1% |

**Swapping a well-calibrated model (Brier 0.08) for one 3× worse (0.25) produces literally
identical routes** under the paper's objective, across 2000 OD pairs × 5 aids.

Controls confirm the harness is sound, and expose a flaw in two of the four objectives:

| control, same score file | Jaccard | identical |
|---|---:|---:|
| linear vs distance-only | 0.30 | 5.5% |
| linear vs bottleneck | 0.21 | 5.2% |
| linear vs logbarrier | 0.99 | 97.5% |
| linear vs risk_split | 1.00 | 100% |

Accessibility scores clearly do change routes (0.30 against plain shortest path), so the
setup measures something. But `logbarrier` is a monotone transform of p_yes, and `risk_split`
collapses to the linear form on the 47% of edges without a real distribution, because the
fallback derives p_no and p_unsure from p_yes — reproducing `length·(1 + 9.75(1−p))`. Those
two are not valid tests as parameterised; that is a design error, not a result. Only
`bottleneck`, which depends on the worst edge's p_no, is a genuinely different objective.

**The conclusion, and why it is good news.** The paper's causal story is backwards: the
objective is linear in p_yes, and *that is precisely why* calibration quality cannot affect
the route. A tail-sensitive objective does distinguish the two models (29% identical), so
calibration can matter — but only under an objective that consumes the distribution rather
than its mean. This is a concrete, actionable fix rather than another null, and it answers
reviewer M7xg's complaint that the edge cost was hand-designed and never compared against a
risk-sensitive alternative.

## 3.15 CIFAR-10H: the soft-label advantage scales with actual disagreement · **DONE**

Ran the identical protocol on CIFAR-10H (frozen CLIP-B/32, linear probe, weight-decay sweep,
5 seeds, 4000 images).

| weight decay | Soft-KL | Hard-CE | ratio |
|---|---:|---:|---:|
| 1e-4 | 0.0932 | **0.0916** | 0.98× |
| 1e-2 | 0.0934 | 0.0917 | 0.98× |
| 1e0 | 0.1790 | 0.1699 | 0.95× |

No soft-label advantage at any setting.

**This must not be reported as a field-level refutation.** The setup does not replicate the
literature: Peterson et al. trained full networks and measured generalisation and adversarial
robustness; this is a linear probe on frozen features scored by soft Brier. Different
question.

What it does establish is a clean relationship worth reporting:

| dataset | mean human entropy | fair ratio |
|---|---:|---:|
| CIFAR-10H | 0.067 | 0.98× (none) |
| this dataset | 0.847 | 1.50× |

On CIFAR-10H, 4393 of 10000 items sit at total consensus and the soft labels are effectively
one-hot, so the two losses receive near-identical targets and produce near-identical models.
There was no advantage to shrink. **The soft-label advantage scales with how much genuine
disagreement the data contains** — which reframes 1.50× on a maximally-contested dataset as
plausibly near the ceiling of what soft labels deliver, not as a weak result.

It also sharpens the novelty claim: this dataset is the regime where the question is even
well-posed.

## 4. Paper changes · **HELD** (deliberately, until experiments finish)

Two rewrites are drafted and agreed in principle, held so the new numbers can go in at the
same time.

**4.1 Data provenance (§Dataset).** Replace "via Project Sidewalk's open-source public
platform" with an explicit statement that the survey is a separate instrument from the PS
labelling platform, and anchor 829 to the CHI 2025 N=190 baseline. Kills R5 pre-emptively.

**4.2 Contribution C2.** The paper already concedes at line 289 that Hard-CE wins argmax F1
on 6/8 encoders inside the 95% CI — but the bold claim sits in the contributions list 190
lines earlier *without* the qualifier. Worse, there are two different "6/8" statistics in
the paper:

- 6/8 encoders below the majority-class baseline **on the walking-cane aid** (our claim)
- Hard-CE ahead on overall macro-F1 in 6/8 encoders (the reviewer's counter)

Both are true and compatible, but the coincidence makes the confusion nearly unavoidable.
Fix by moving the qualifier next to the claim. The science does not change.

**4.3 Reformatting.** CoRL single-column → CVPR IEEE two-column, 8 pages + references.
Content will have to move to supplementary; the routing section is the natural candidate,
which aligns with de-emphasising it per R1.

---

## 5. Independent route validation (R1) · **OPEN**

The one objection all three reviewers raised, and the hardest to close before November.

- `apps/accessibility-study` is a **prototype we want to pitch to the Project Sidewalk
  team**, not a live study we run. Its only "real" session is Wesley testing his own tool.
- Advisor contact with Jon Froehlich (Project Sidewalk PI, UW) is live. A relationship-
  keeping reply is drafted; the ask is framed as an idea, and asks how the CHI 2025 team
  reached N=190 rather than requesting hosting.
- Paid recruitment at scale (Prolific, Fable) was considered and **rejected**: it needs a
  budget and IRB work that do not fit a 10-week window.
- Realistic near-term plan: active learning (§3.3) shrinks the number of new votes needed
  to something a handful of informal participants can supply.

If no independent signal lands before the deadline, the fallback is §4.3 — present routing
as a demonstration rather than a validated claim, and say so plainly.

---

## 6. Cluster jobs

QOS caps submitted jobs per user (`gpu`=4, `cenvalarc.gpu`=2–4, `test`=1), so the work is
split into chunked jobs rather than large arrays. Each chunk runs its share of the configs
sequentially; feature caching keeps each config to a few minutes.

Two scheduling lessons, both worth remembering:

- **Ask for short walltime.** The first attempt requested 2 hours per job and SLURM
  estimated a start two days out. The actual work is minutes. Re-submitting with
  `--time=00:55:00` got a job running in ten seconds.
- **Check `test`.** It caps at 1 job and 1 hour, but it was completely empty (0 running,
  0 pending) with three idle A100 nodes while `gpu` was saturated. The whole 16-config CV
  fits inside its hour.

| Job | Partition | Work | Status |
|-----|-----------|------|--------|
| 331335 | `test` | multi-seed CV, 8 encoders × 2 losses × 10 seeds | **done** — 16/16 in ~10 min |
| 331340 | `test` | semi-supervised: 3 encoders + entropy sweep + PS-filter ablation | **done** — 9/9 |
| 331344 | `test` | temperature scaling, 8 encoders × 10 seeds | **done** — 8/8 |
| 331348 | `test` | routing re-run under SigLIP2-SO400M | **done** (graph path was wrong; Monte Carlo redone separately) |
| 331351/331353 | `test` | Monte Carlo probes: which edge scores / which graph | **done** — settled §3.5b |
| 331354 | `test` | coupled probe: count losses × head types × rank sweep | running |
| 331349 | `gpu` | downstream: generalization + walking-cane analysis under SigLIP2 | queued |

Note the routing job wrote `results/routing_siglip2-so400m/monte_carlo/` with the wrong
graph. The trustworthy one is `monte_carlo_956/`, produced by the separate corrected run.
`run_routing_encoder.sbatch` has since been fixed, so a fresh submit does the right thing.

```bash
sbatch --partition=test --time=00:55:00 --export=ALL,CHUNK=0,NCHUNKS=1 run_cv_multiseed.sbatch
sbatch --partition=test --time=00:55:00 --export=ALL,CHUNK=0,NCHUNKS=1 run_semisup.sbatch
sbatch --partition=test --time=00:55:00 --export=ALL,CHUNK=0,NCHUNKS=1 run_temp_scaling.sbatch
sbatch --partition=test --time=00:55:00 --export=ALL,CHUNK=0,NCHUNKS=1 run_coupled.sbatch
sbatch --partition=gpu  --export=ALL,ENCODER=siglip2-so400m          run_routing_encoder.sbatch
sbatch --partition=gpu  --export=ALL,ENCODER=siglip2-so400m          run_downstream_encoder.sbatch
squeue -u $USER
```

Gathering results:

```bash
python src/models/summarize_multiseed.py --results_dir results/cv_multiseed --latex
python src/models/summarize_multiseed.py --results_dir results/cv_multiseed_ts
```

---

## 7. Datasets surveyed (English, German, Chinese, Japanese)

Searched for anything that could supply more images, or more importantly more *per-aid
subjective votes*. Conclusion: **no one else appears to have collected per-mobility-aid
vote distributions on street-level imagery at any scale.** Project Sidewalk / CHI 2025 is
the only direct precedent found in any language. That cuts both ways — no shortcut exists,
but the novelty claim is genuine and worth stating explicitly against R4's "not a new
contribution".

| Source | What it is | Usable? |
|--------|-----------|---------|
| [SANPO](https://github.com/google-research-datasets/sanpo_dataset) (WACV 2025, Google) | 701 stereo videos, 112k annotated frames, outdoor human navigation. CC 4.0 | Candidate image pool + strong related-work citation. No per-aid votes |
| [Mapillary Vistas](https://www.mapillary.com/dataset/vistas) | 25k densely annotated street-level images, ~90% road/sidewalk | Candidate image pool. Object labels, not votes |
| [StreetSurfaceVis](https://arxiv.org/abs/2407.21454) | 9,122 crowdsourced images, mostly Germany, surface type/quality graded explicitly against wheelchair suitability | Auxiliary signal, worth citing |
| [SurfaceAI](https://arxiv.org/abs/2409.18922) | Pipeline generating surface-quality datasets from Mapillary | Method reference |
| Chinese facility-detection sets (CSDN) | ~5–8k street images, accessibility infrastructure detection | Provenance and licence unclear — do not use without verifying the primary source |
| [Wheelmap / Sozialhelden](https://www.sozialhelden.de/project/wheelmap) | 2M+ venue accessibility ratings, ODbL | POI-level, wrong granularity for this paper. Good fit for the separate [MAP benchmark](https://arxiv.org/abs/2608.28384) |

---

## 8. Method references for the small-data attack

- [CIFAR-10H](https://github.com/jcpeterson/cifar-10h) (ICCV 2019) — precedent for soft
  labels from human uncertainty. **Not** a precedent for few images: it has 10,000. Do not
  cite it as cover for N=52.
- [DiCaP](https://arxiv.org/html/2511.20225v1) — distribution-calibrated pseudo-labelling.
- [SemiPrune](https://arxiv.org/abs/2605.23198) — label-efficient semi-supervised pruning.
- [Label What Matters](https://openaccess.thecvf.com/content/CVPR2026/html/Zeng_Label_What_Matters_Modality-Balanced_and_Difficulty-Aware_Multimodal_Active_Learning_CVPR_2026_paper.html) (CVPR 2026) — difficulty-aware active learning.

---

## 9. Immediate next actions

1. Wait for jobs 331328–331333; gather with `summarize_multiseed.py`. · **RUNNING**
2. Read the semi-supervised deltas honestly. If the pseudo-label expansion does not improve
   calibration on real held-out votes, report that — a negative result is still an answer to
   R2, and inventing a positive one is how papers get rejected twice. · **NEXT**
3. Decide the 829 vs 595 participant-count question (§2). · **OPEN**
4. Apply the two paper rewrites (§4.1, §4.2) once the numbers are in. · **HELD**
5. Active learning (§3.3). · **NEXT**
6. CVPR reformatting (§4.3). · **NEXT**
