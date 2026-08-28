#!/bin/bash
#SBATCH --job-name=anchor6_ctid
#SBATCH --output=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/anchor6_ctid_%A_%a.out
#SBATCH --error=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/anchor6_ctid_%A_%a.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu2023
#SBATCH --array=0-395%80

set -euo pipefail

PROJECT_DIR="/home/onyero.ofuzim/projects/battery-degradation-spme-sysid"
VENV_DIR="/home/onyero.ofuzim/venvs/jaxsys"

cd "$PROJECT_DIR"

mkdir -p results/logs
mkdir -p results/real_cycle_ctid_state_order_grid
mkdir -p results/real_warm_continuation_ctid
mkdir -p results/figures/real_cycle_ctid_state_order_grid
mkdir -p results/figures/real_warm_continuation_ctid
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
# Anchor model set:
#   66 cycles x 6 models = 396 tasks
#   cycle 34--99 inclusive
# ---------------------------------------------------------
ANCHOR_MODELS=(
  "S7:C1:grid"
  "S7:C4:grid"
  "S7:C4K:kparam"
  "S17:C1:grid"
  "S17:C4:grid"
  "S17:C4K:kparam"
)

N_MODELS=${#ANCHOR_MODELS[@]}

CYCLE_OFFSET=$((SLURM_ARRAY_TASK_ID / N_MODELS))
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % N_MODELS))

export UN_REAL_CYCLE_INDEX=$((34 + CYCLE_OFFSET))

IFS=":" read -r STATE_ID CANDIDATE_ID SCRIPT_MODE <<< "${ANCHOR_MODELS[$MODEL_INDEX]}"

# ---------------------------------------------------------
# Shared real-data settings
# ---------------------------------------------------------
export UN_ID_DOWNSAMPLE_DT=1.0
export UN_CURRENT_SIGN_MODE=auto
export UN_SMOOTH_VOLTAGE_WINDOW=1
export UN_MAX_CYCLE_TIME=0.0

# Multistart controls.
# For speed, start with 20 seeds per cycle/model.
# You can submit again with UN_N_MULTISTART=50 or 100 if needed.
export UN_SEED0=${UN_SEED0:-200}
export UN_N_MULTISTART=${UN_N_MULTISTART:-20}

# Plotting: keep off for speed. The summarizer will make thesis visuals.
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

# Direct-k script controls
export UN_INIT_MODE=cold
export UN_WARM_START_FILE=""

echo "=================================================="
echo "ANCHOR SIX-MODEL REAL CT-ID SCREENING"
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Cycle offset: ${CYCLE_OFFSET}"
echo "Real cycle index: ${UN_REAL_CYCLE_INDEX}"
echo "Model index: ${MODEL_INDEX}"
echo "State ID: ${STATE_ID}"
echo "Candidate ID: ${CANDIDATE_ID}"
echo "Script mode: ${SCRIPT_MODE}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "UN_SEED0: ${UN_SEED0}"
echo "UN_N_MULTISTART: ${UN_N_MULTISTART}"
echo "UN_ID_DOWNSAMPLE_DT: ${UN_ID_DOWNSAMPLE_DT}"
echo "=================================================="

if [[ "${SCRIPT_MODE}" == "grid" ]]; then
    export UN_STATE_VARIANTS="${STATE_ID}"
    export UN_OUTPUT_CANDIDATES="${CANDIDATE_ID}"

    export UN_RUN_TAG="anchor6_${STATE_ID}_${CANDIDATE_ID}_${UN_N_MULTISTART}seeds_cycle_${UN_REAL_CYCLE_INDEX}_seed_${UN_SEED0}_dt_${UN_ID_DOWNSAMPLE_DT}"

    echo "Running grid script:"
    echo "  scripts/run_real_cycle_state_order_ctid_grid.py"
    echo "UN_STATE_VARIANTS: ${UN_STATE_VARIANTS}"
    echo "UN_OUTPUT_CANDIDATES: ${UN_OUTPUT_CANDIDATES}"
    echo "UN_RUN_TAG: ${UN_RUN_TAG}"

    python -u scripts/run_real_cycle_state_order_ctid_grid.py

elif [[ "${SCRIPT_MODE}" == "kparam" ]]; then
    export UN_STATE_VARIANT="${STATE_ID}"
    export UN_OUTPUT_CANDIDATE="C4"

    MODEL_ID="${STATE_ID}_C4K"
    export UN_RUN_TAG="anchor6_${MODEL_ID}_${UN_N_MULTISTART}seeds_cycle_${UN_REAL_CYCLE_INDEX}_seed_${UN_SEED0}_dt_${UN_ID_DOWNSAMPLE_DT}"

    echo "Running direct-k warm-continuation script in cold mode:"
    echo "  scripts/run_real_cycle_warm_continuation_ctid_kparam.py"
    echo "UN_STATE_VARIANT: ${UN_STATE_VARIANT}"
    echo "UN_OUTPUT_CANDIDATE: ${UN_OUTPUT_CANDIDATE}"
    echo "MODEL_ID: ${MODEL_ID}"
    echo "UN_RUN_TAG: ${UN_RUN_TAG}"

    python -u scripts/run_real_cycle_warm_continuation_ctid_kparam.py

else
    echo "Unknown SCRIPT_MODE: ${SCRIPT_MODE}"
    exit 1
fi

echo "=================================================="
echo "Finished at: $(date)"
echo "=================================================="
