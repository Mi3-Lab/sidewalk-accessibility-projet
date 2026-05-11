#!/bin/bash
# Task 5.7 Generalization Test — tmux Session (cluster-agnostic)
#
# Usage:
#   bash src/generalization/run_generalization_tmux.sh
#
# This creates a detached tmux session that persists even if you close
# your SSH connection. You can reconnect with:
#   tmux attach-session -t generalization
#
# Kill the session later with:
#   tmux kill-session -t generalization

set -e

# Configuration
SESSION_NAME="generalization"
ENCODER="${1:-dinov2-large}"
CHECKPOINT_DIR="${2:-results/models/dinov2-large}"
PROJECT_DIR="/home/wesleyferreiramaia/data/sidewalk-accessibility-project"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux not found. Install it with: sudo apt-get install tmux"
    exit 1
fi

# Create logs directory
mkdir -p "${PROJECT_DIR}/logs"

# Kill existing session if it exists
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

echo "📌 Creating tmux session: ${SESSION_NAME}"
echo "   To attach: tmux attach-session -t ${SESSION_NAME}"
echo "   To detach: Press Ctrl+B, then D"
echo "   To kill: tmux kill-session -t ${SESSION_NAME}"
echo ""

# Create new detached session
tmux new-session -d -s "${SESSION_NAME}" -c "${PROJECT_DIR}"

# Send commands to the session
tmux send-keys -t "${SESSION_NAME}" "
set -e
echo '════════════════════════════════════════════════════════════════'
echo 'TASK 5.7 GENERALIZATION TEST — tmux Session'
echo '════════════════════════════════════════════════════════════════'
echo 'Session: ${SESSION_NAME}'
echo 'Encoder: ${ENCODER}'
echo 'Checkpoint: ${CHECKPOINT_DIR}'
echo 'Start: '(date)
echo ''

# Activate virtualenv if it exists
[ -d venv ] && source venv/bin/activate

# Install W&B
pip install -q wandb 2>/dev/null || true

# Run evaluation
python -u src/generalization/evaluate_generalization.py \\
    --encoder '${ENCODER}' \\
    --checkpoint_dir '${CHECKPOINT_DIR}' \\
    --use_wandb \\
    --wandb_project 'sidewalk-generalization' \\
    --wandb_run_name 'gen_${ENCODER}_'(date +%Y%m%d_%H%M%S)

echo ''
echo '════════════════════════════════════════════════════════════════'
echo 'Job completed: '(date)
echo '════════════════════════════════════════════════════════════════'
echo ''
echo 'Session still running. Press Ctrl+B then D to detach.'
" C-m

# Attach to the session (user can detach with Ctrl+B then D)
echo ""
echo "✅ Session created and running!"
echo ""
echo "Monitor in real-time:"
echo "  tmux attach-session -t ${SESSION_NAME}"
echo ""
echo "From another terminal:"
echo "  tmux list-sessions"
echo "  tail -f logs/generalization_*.log"
echo ""
