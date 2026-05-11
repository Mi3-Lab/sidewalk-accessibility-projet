#!/bin/bash
# Pre-flight check before running Task 5.7 on cluster
# Verifica: GPU, Python, dependências, imagens, modelos, W&B

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   PRE-FLIGHT CHECK — Task 5.7 Generalization Test             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

FAILED=0

# 1. GPU Check
echo "1️⃣  GPU Availability"
echo "   Checking: nvidia-smi"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "   ✅ GPUs found: $GPU_COUNT"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/      /'
else
    echo "   ⚠️  nvidia-smi not found (CPU mode will be slow)"
fi
echo ""

# 2. Python & dependencies
echo "2️⃣  Python Dependencies"
python_check() {
    python -c "$1" 2>/dev/null && return 0 || return 1
}

DEPS=("torch" "pandas" "numpy" "sklearn" "PIL" "joblib" "tqdm")
for dep in "${DEPS[@]}"; do
    if python_check "import $dep"; then
        echo "   ✅ $dep"
    else
        echo "   ❌ $dep (missing)"
        FAILED=1
    fi
done
echo ""

# 3. W&B
echo "3️⃣  Weights & Biases (W&B)"
if python_check "import wandb"; then
    echo "   ✅ wandb installed"
    if [ -f ~/.wandb/settings ]; then
        echo "   ✅ W&B login found (~/.wandb/settings)"
    else
        echo "   ⚠️  W&B not logged in. Run: wandb login"
    fi
else
    echo "   ⚠️  wandb not installed. Run: pip install wandb"
fi
echo ""

# 4. Cluster tools
echo "4️⃣  Cluster Tools"
echo "   SLURM:"
if command -v sbatch &> /dev/null; then
    echo "   ✅ sbatch found (SLURM available)"
else
    echo "   ℹ️  sbatch not found (use tmux instead)"
fi

echo "   tmux:"
if command -v tmux &> /dev/null; then
    echo "   ✅ tmux found (persistent sessions)"
else
    echo "   ⚠️  tmux not found. Install with: apt-get install tmux"
fi
echo ""

# 5. Data files
echo "5️⃣  Data Files"
if [ -d "data/generalization/images" ]; then
    IMG_COUNT=$(ls data/generalization/images/*.jpg 2>/dev/null | wc -l)
    echo "   ✅ Images directory found ($IMG_COUNT images)"
else
    echo "   ❌ data/generalization/images/ not found"
    FAILED=1
fi

if [ -f "data/generalization/test_images.csv" ]; then
    ROWS=$(wc -l < data/generalization/test_images.csv)
    echo "   ✅ test_images.csv found ($ROWS rows)"
else
    echo "   ❌ data/generalization/test_images.csv not found"
    FAILED=1
fi
echo ""

# 6. Models
echo "6️⃣  Trained Models"
ENCODERS=("dinov2-large" "clip-vit-b32" "dinov2-base")
for enc in "${ENCODERS[@]}"; do
    if [ -d "results/models/$enc" ]; then
        PROBES=$(find "results/models/$enc" -name "probe.pth" 2>/dev/null | wc -l)
        echo "   ✅ $enc ($PROBES probes)"
    else
        echo "   ⚠️  results/models/$enc/ not found"
    fi
done
echo ""

# 7. Scripts
echo "7️⃣  Execution Scripts"
SCRIPTS=("run_generalization.sh" "src/generalization/run_generalization_slurm.sh" "src/generalization/run_generalization_tmux.sh" "src/generalization/evaluate_generalization.py")
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "   ✅ $script"
    else
        echo "   ❌ $script not found"
        FAILED=1
    fi
done
echo ""

# 8. Output directories
echo "8️⃣  Output Directories"
mkdir -p results/generalization logs
echo "   ✅ results/generalization/ (created)"
echo "   ✅ logs/ (created)"
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════════╗"
if [ $FAILED -eq 0 ]; then
    echo "║  ✅ ALL CHECKS PASSED — Ready to run Task 5.7!              ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Next steps:"
    echo "  1. Choose method (SLURM, tmux, or nohup)"
    echo "  2. Read: RODAR_EM_CLUSTER.md"
    echo "  3. Run your chosen command"
    exit 0
else
    echo "║  ❌ SOME CHECKS FAILED — Fix issues above first              ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    exit 1
fi
