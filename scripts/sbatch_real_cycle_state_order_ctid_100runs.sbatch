#!/bin/bash
#SBATCH --job-name=real_ctid_grid
#SBATCH --output=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/real_ctid_grid_%A_%a.out
#SBATCH --error=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/real_ctid_grid_%A_%a.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu2023
#SBATCH --array=0-15

set -euo pipefail

PROJECT_DIR="/home/onyero.ofuzim/projects/battery-degradation-spme-sysid"
VENV_DIR="/home/onyero.ofuzim/venvs/jaxsys"

cd "$PROJECT_DIR"

mkdir -p results/logs
mkdir -p results/real_cycle_ctid_state_order_grid
mkdir -p results/figures/real_cycle_ctid_state_order_grid
mkdir -p results/metrics
mkdir -p results/tables
mkdir -p results/outputs

source "$VENV_DIR/bin/activate"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg

# ---------------------------------------------------------
# Grid:
#   4 state variants x 4 output candidates = 16 tasks
# ---------------------------------------------------------
STATE_VARIANTS=("S7" "S12" "S14" "S17")
OUTPUT_CANDIDATES=("C1" "C2" "C3" "C4")

STATE_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
OUTPUT_INDEX=$((SLURM_ARRAY_TASK_ID % 4))

export UN_STATE_VARIANTS="${STATE_VARIANTS[$STATE_INDEX]}"
export UN_OUTPUT_CANDIDATES="${OUTPUT_CANDIDATES[$OUTPUT_INDEX]}"

# ---------------------------------------------------------
# Real cycle selection
# ---------------------------------------------------------
export UN_REAL_CYCLE_INDEX=0
export UN_ID_DOWNSAMPLE_DT=1.0
export UN_CURRENT_SIGN_MODE=auto

# Optional:
# If you want to manually force a specific .mpr path, uncomment:
# export UN_MPR_PATH="/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/data/12to1-25%CNC-3%GQDs _C01.mpr"

# ---------------------------------------------------------
# Multistart controls
# ---------------------------------------------------------
export UN_SEED0=200
export UN_N_MULTISTART=100

# ---------------------------------------------------------
# Histogram / plotting
# ---------------------------------------------------------
export UN_HIST_BINS=100
export UN_MAKE_PLOTS=True
export UN_SHOW_PLOTS=False
export UN_SAVE_PLOTS=True

# ---------------------------------------------------------
# Optimization controls
# ---------------------------------------------------------
export UN_IPRINT=0
export UN_ADAM_EPOCHS=500
export UN_ADAM_ETA=2e-3
export UN_LBFGS_EPOCHS=5000
export UN_LBFGS_TOL=1e-12
export UN_LBFGS_MEMORY=30

# ---------------------------------------------------------
# Initialization controls
# ---------------------------------------------------------
export UN_INIT_DYN_JITTER=0.20
export UN_INIT_GAIN_JITTER=0.20
export UN_INIT_THETA_JITTER=0.005

export UN_INIT_C_JITTER=0.05
export UN_INIT_BETA_SCALE=1e-2
export UN_INIT_D1_CENTER=-0.004
export UN_INIT_D1_JITTER=0.002
export UN_INIT_E_SCALE=1e-2

# ---------------------------------------------------------
# Unique run tag
# ---------------------------------------------------------
export UN_RUN_TAG="real_cycle${UN_REAL_CYCLE_INDEX}_${UN_STATE_VARIANTS}_${UN_OUTPUT_CANDIDATES}_${UN_N_MULTISTART}seeds_${UN_SEED0}_to_$((UN_SEED0 + UN_N_MULTISTART - 1))_dt_${UN_ID_DOWNSAMPLE_DT}_bins_${UN_HIST_BINS}"

echo "=================================================="
echo "REAL FIRST-CYCLE CT-ID STATE/ORDER GRID"
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "State index: ${STATE_INDEX}"
echo "Output index: ${OUTPUT_INDEX}"
echo "State variant: ${UN_STATE_VARIANTS}"
echo "Output candidate: ${UN_OUTPUT_CANDIDATES}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Memory: 64G"
echo "Script: scripts/run_real_cycle_state_order_ctid_grid.py"
echo "UN_REAL_CYCLE_INDEX: ${UN_REAL_CYCLE_INDEX}"
echo "UN_SEED0: ${UN_SEED0}"
echo "UN_N_MULTISTART: ${UN_N_MULTISTART}"
echo "Seed range: ${UN_SEED0} to $((UN_SEED0 + UN_N_MULTISTART - 1))"
echo "UN_ID_DOWNSAMPLE_DT: ${UN_ID_DOWNSAMPLE_DT}"
echo "UN_HIST_BINS: ${UN_HIST_BINS}"
echo "UN_ADAM_EPOCHS: ${UN_ADAM_EPOCHS}"
echo "UN_LBFGS_EPOCHS: ${UN_LBFGS_EPOCHS}"
echo "UN_RUN_TAG: ${UN_RUN_TAG}"
echo "=================================================="

python -u scripts/run_real_cycle_state_order_ctid_grid.py

echo "=================================================="
echo "Finished at: $(date)"
echo "=================================================="