#!/bin/bash
#SBATCH --job-name=wseq_s17_comb
#SBATCH --output=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/wseq_s17_comb_%A.out
#SBATCH --error=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/wseq_s17_comb_%A.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu2023

set -euo pipefail

PROJECT_DIR="/home/onyero.ofuzim/projects/battery-degradation-spme-sysid"
VENV_DIR="/home/onyero.ofuzim/venvs/jaxsys"

cd "$PROJECT_DIR"
mkdir -p results/logs

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg

: "${UN_REAL_CYCLE_INDEX:?Need UN_REAL_CYCLE_INDEX}"

export UN_MODEL_ID="S17_C4"

echo "Combining S17_C4 cycle ${UN_REAL_CYCLE_INDEX}"

python -u scripts/combine_real_cycle_warm_continuation_ctid.py