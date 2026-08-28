#!/bin/bash
#SBATCH --job-name=wseq_kcomb
#SBATCH --output=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/wseq_kcomb_%j.out
#SBATCH --error=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/wseq_kcomb_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu2023

set -euo pipefail

PROJECT_DIR="/home/onyero.ofuzim/projects/battery-degradation-spme-sysid"
VENV_DIR="/home/onyero.ofuzim/venvs/jaxsys"

cd "$PROJECT_DIR"
mkdir -p results/logs

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

: "${UN_MODEL_ID:?Need UN_MODEL_ID=S7_C4K or S17_C4K}"
: "${UN_REAL_CYCLE_INDEX:?Need UN_REAL_CYCLE_INDEX}"

echo "=================================================="
echo "COMBINE DIRECT-K/B CYCLE"
echo "Model: ${UN_MODEL_ID}"
echo "Cycle: ${UN_REAL_CYCLE_INDEX}"
echo "=================================================="

python -u scripts/combine_real_cycle_warm_continuation_ctid_kparam.py