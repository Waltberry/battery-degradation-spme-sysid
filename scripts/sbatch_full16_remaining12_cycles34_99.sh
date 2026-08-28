#!/bin/bash
#SBATCH --job-name=full16_rem
#SBATCH --output=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/full16_remaining_%A_%a.out
#SBATCH --error=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/full16_remaining_%A_%a.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu2023
#SBATCH --array=0-791%80

set -euo pipefail

PROJECT_DIR="/home/onyero.ofuzim/projects/battery-degradation-spme-sysid"
VENV_DIR="/home/onyero.ofuzim/venvs/jaxsys"

cd "$PROJECT_DIR"

mkdir -p results/logs
mkdir -p results/real_cycle_ctid_state_order_grid
mkdir -p results/figures/real_cycle_ctid_state_order_grid
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
export JAX_ENABLE_X64=True

# ---------------------------------------------------------
# Remaining models for final 16-model heatmap:
#
# Already have from anchor:
#   S7_C1, S7_C4K
#   S17_C1, S17_C4K
#
# Run now:
#   S7_C2, S7_C3
#   S12_C1, S12_C2, S12_C3, S12_C4
#   S14_C1, S14_C2, S14_C3, S14_C4
#   S17_C2, S17_C3
#
# 66 cycles x 12 models = 792 tasks
# cycle 34--99 inclusive
# ---------------------------------------------------------

MODEL_SPECS=(
  "S7:C2"
  "S7:C3"

  "S12:C1"
  "S12:C2"
  "S12:C3"
  "S12:C4"

  "S14:C1"
  "S14:C2"
  "S14:C3"
  "S14:C4"

  "S17:C2"
  "S17:C3"
)

N_MODELS=${#MODEL_SPECS[@]}

CYCLE_OFFSET=$((SLURM_ARRAY_TASK_ID / N_MODELS))
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % N_MODELS))

export UN_REAL_CYCLE_INDEX=$((34 + CYCLE_OFFSET))

IFS=":" read -r STATE_ID CANDIDATE_ID <<< "${MODEL_SPECS[$MODEL_INDEX]}"

# ---------------------------------------------------------
# Real-data settings
# ---------------------------------------------------------
export UN_STATE_VARIANTS="${STATE_ID}"
export UN_OUTPUT_CANDIDATES="${CANDIDATE_ID}"

export UN_ID_DOWNSAMPLE_DT=1.0
export UN_CURRENT_SIGN_MODE=auto
export UN_SMOOTH_VOLTAGE_WINDOW=1
export UN_MAX_CYCLE_TIME=0.0

# Same seed settings as anchor6.
export UN_SEED0=${UN_SEED0:-200}
export UN_N_MULTISTART=${UN_N_MULTISTART:-20}

# Keep plotting off for speed.
export UN_HIST_BINS=100
export UN_MAKE_PLOTS=False
export UN_SHOW_PLOTS=False
export UN_SAVE_PLOTS=False

# Optimization controls
export UN_IPRINT=0
export UN_ADAM_EPOCHS=${UN_ADAM_EPOCHS:-500}
export UN_ADAM_ETA=${UN_ADAM_ETA:-2e-3}
export UN_LBFGS_EPOCHS=${UN_LBFGS_EPOCHS:-5000}
export UN_LBFGS_TOL=${UN_LBFGS_TOL:-1e-12}
export UN_LBFGS_MEMORY=${UN_LBFGS_MEMORY:-30}

# Initialization controls
export UN_INIT_DYN_JITTER=0.20
export UN_INIT_GAIN_JITTER=0.20
export UN_INIT_THETA_JITTER=0.005

export UN_INIT_C_JITTER=0.05
export UN_INIT_BETA_SCALE=1e-2
export UN_INIT_D1_CENTER=-0.004
export UN_INIT_D1_JITTER=0.002
export UN_INIT_E_SCALE=1e-2

# Unique run tag
export UN_RUN_TAG="full16rem_${STATE_ID}_${CANDIDATE_ID}_${UN_N_MULTISTART}seeds_cycle_${UN_REAL_CYCLE_INDEX}_seed_${UN_SEED0}_dt_${UN_ID_DOWNSAMPLE_DT}"

echo "=================================================="
echo "FULL 16 HEATMAP — REMAINING 12 MODEL RUN"
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Cycle offset: ${CYCLE_OFFSET}"
echo "Real cycle index: ${UN_REAL_CYCLE_INDEX}"
echo "Model index: ${MODEL_INDEX}"
echo "State ID: ${STATE_ID}"
echo "Candidate ID: ${CANDIDATE_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "UN_SEED0: ${UN_SEED0}"
echo "UN_N_MULTISTART: ${UN_N_MULTISTART}"
echo "UN_ID_DOWNSAMPLE_DT: ${UN_ID_DOWNSAMPLE_DT}"
echo "UN_RUN_TAG: ${UN_RUN_TAG}"
echo "=================================================="

python -u scripts/run_real_cycle_state_order_ctid_grid.py

echo "=================================================="
echo "Finished at: $(date)"
echo "=================================================="
