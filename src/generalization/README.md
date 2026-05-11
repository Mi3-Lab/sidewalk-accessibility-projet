# ✅ TASK 5.7 — GENERALIZATION TEST (Week 2) — SETUP COMPLETE

---

## 📊 What Just Got Prepared

```
┌──────────────────────────────────────────────────────────────────┐
│ TASK 5.7: GENERALIZATION TEST — Week 2                          │
│ Goal: Prove model works on cities outside training set           │
└──────────────────────────────────────────────────────────────────┘
                                
    YOUR INPUT                SCRIPTS                    OUTPUT
    ┌──────────────────┐    ┌─────────────┐         ┌──────────────┐
    │ New city images  │───→│  Evaluate   │────────→│  Agreement % │
    │ (DC, Columbus)   │    │  Gen.py     │         │  + CSV file  │
    └──────────────────┘    └─────────────┘         └──────────────┘
    
    + Project Sidewalk       + Prepare test  
      labels (PS CSV)        + Smoke test
                             + Run bash script
```

---

## 🎯 What Was Created (7 files)

| File | Purpose | Status |
|------|---------|--------|
| `src/generalization/evaluate_generalization.py` | Main evaluation engine | ✅ Ready |
| `src/generalization/prepare_test_data.py` | CSV template + validation | ✅ Ready |
| `src/generalization/create_example_csv.py` | Example CSV generator | ✅ Ready |
| `src/generalization/smoke_test.py` | Pre-flight checks | ✅ Ready |
| `run_generalization.sh` | Orchestrator bash script | ✅ Ready |
| `init_generalization.sh` | Quick setup script | ✅ Ready |
| `GENERALIZATION_SETUP.md` | Full documentation | ✅ Ready |
| `TASK_5.7_READY.md` | Quick reference (this folder) | ✅ Ready |

---

## 🚀 Your 3-Step Action Plan

### Step 1️⃣ Initialize (30 seconds)
```bash
bash init_generalization.sh
```
Creates:
- `data/generalization/test_images.csv` (empty template)
- `data/generalization/images/` (folder for images)
- `results/generalization/` (results folder)

### Step 2️⃣ Populate Data (1–2 hours, manual)
```
1. Go to https://projectsidewalk.org/map
2. Select cities: Washington DC, Columbus OH, etc.
3. Download ~50 images marked "CurbRamp" (accessible)
4. Download ~50 images marked "NoCurbRamp" (inaccessible)
5. Save to: data/generalization/images/
6. Edit CSV with image paths, labels, cities
```

Example filled CSV:
```
image_id,image_path,ps_label,city
dc_001,data/generalization/images/dc_001.jpg,CurbRamp,Washington DC
dc_002,data/generalization/images/dc_002.jpg,NoCurbRamp,Washington DC
col_001,data/generalization/images/col_001.jpg,CurbRamp,Columbus OH
```

### Step 3️⃣ Run Evaluation (15 minutes)
```bash
# Validate data (safety check)
python src/generalization/prepare_test_data.py \
    --validate \
    --images_csv data/generalization/test_images.csv

# Run generalization test (best encoder: DINOv2-large, F1=0.613)
./run_generalization.sh dinov2-large results/models/dinov2-large
```

**Output:** 
```
results/generalization/dinov2-large/
├── predictions.csv                # [image_id, pred, agree_with_ps]
├── agreement_summary.json         # {"accuracy": 0.847, ...}
└── confusion_matrix_per_aid.json  # per-aid breakdown
```

---

## 📈 What Success Looks Like

```json
{
  "encoder": "dinov2-large",
  "n_test_samples": 100,
  "overall_accuracy": 0.847,
  "per_city_accuracy": {
    "Washington DC": {"accuracy": 0.860, "n_samples": 50},
    "Columbus OH": {"accuracy": 0.820, "n_samples": 50}
  }
}
```

**Paper text (Section 5.7):**
> Geographic Transfer: On 100 out-of-distribution sidewalk
> images from two new US cities (Washington DC, Columbus OH),
> the model achieved 84.7% agreement with Project Sidewalk
> structural labels, demonstrating robust geographic transfer.

---

## ⏱️ Timeline

| Task | Time | Owner |
|------|------|-------|
| Init + download images | 1–2 hr | You |
| Populate CSV | 30 min | You |
| Run evaluation | 15 min | Script (auto) |
| Analyze results | 30 min | You |
| Write paper section | 30 min | You |
| **Total** | **3–4 hours** | |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Image not found" | Check `data/generalization/images/` folder |
| "Checkpoint not found" | Ensure `results/models/dinov2-large/*/probe.pth` exists |
| Accuracy < 70% | Verify PS labels are correct; may indicate legitimate domain shift |
| CUDA out of memory | Script auto-switches to CPU or reduce batch size in code |

---

## 📚 Documentation Files

- **Quick setup**: `TASK_5.7_READY.md` ← You're reading this
- **Full guide**: `GENERALIZATION_SETUP.md`
- **Action plan**: `CORL_ACTION_PLAN.md` (updated Section 5.7)

---

## ✨ Key Insight

You're testing a **crucial question for CoRL**: "Does my model work beyond Pittsburgh?"

- Model trained on: **67 GSV panoramas** (1 city)
- Test on: **100+ images** from 2+ new cities
- Metric: **Agreement with Project Sidewalk labels**

If you get 80%+ accuracy → **Strong geographic transfer** → Include in paper  
If you get 70–80% → **Reasonable, discuss domain shift** → Note in limitations  
If < 70% → **Investigate failure cases** → May indicate overfitting

---

## 🎬 Action Required From You

**Today:**
1. Read `GENERALIZATION_SETUP.md`
2. Run `bash init_generalization.sh`
3. Start downloading test images

**Next session:**
1. Populate CSV with image paths + labels
2. Run: `./run_generalization.sh dinov2-large results/models/dinov2-large`
3. Share results JSON

Then we integrate into paper Section 5.7 and move to **Week 3: Routing Demo** 🗺️

---

**You're ready. 🚀 Go get those test images!**
