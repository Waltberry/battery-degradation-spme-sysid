#!/bin/bash
#SBATCH --job-name=wseq_c4k
#SBATCH --output=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/wseq_c4k_%x_%A_%a.out
#SBATCH --error=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/wseq_c4k_%x_%A_%a.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu2023
#SBATCH --array=0-9

set -euo pipefail

PROJECT_DIR="/home/onyero.ofuzim/projects/battery-degradation-spme-sysid"
VENV_DIR="/home/onyero.ofuzim/venvs/jaxsys"

cd "$PROJECT_DIR"
mkdir -p results/logs

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg

# ============================================================
# Required external variables
# ============================================================
: "${UN_STATE_VARIANT:?Need UN_STATE_VARIANT=S7 or S17}"
: "${UN_REAL_CYCLE_INDEX:?Need UN_REAL_CYCLE_INDEX}"
: "${UN_INIT_MODE:?Need UN_INIT_MODE=cold or warm}"

if [[ "${UN_STATE_VARIANT}" != "S7" && "${UN_STATE_VARIANT}" != "S17" ]]; then
    echo "ERROR: UN_STATE_VARIANT must be S7 or S17"
    exit 1
fi

if [[ "${UN_INIT_MODE}" != "cold" && "${UN_INIT_MODE}" != "warm" ]]; then
    echo "ERROR: UN_INIT_MODE must be cold or warm"
    exit 1
fi

# ============================================================
# Model setup
# ============================================================
export UN_OUTPUT_CANDIDATE="C4"

MODEL_ID="${UN_STATE_VARIANT}_C4K"
CHUNK_INDEX=${SLURM_ARRAY_TASK_ID}
SEEDS_PER_CHUNK=10

# Seed base by model, so S7 and S17 do not use identical seed families.
if [[ "${UN_STATE_VARIANT}" == "S7" ]]; then
    BASE_SEED0=2000
else
    BASE_SEED0=5000
fi

export UN_SEED0=$((BASE_SEED0 + UN_REAL_CYCLE_INDEX * 100 + CHUNK_INDEX * SEEDS_PER_CHUNK))
export UN_N_MULTISTART="${SEEDS_PER_CHUNK}"

# ============================================================
# Warm-start file
# ============================================================
if [[ "${UN_INIT_MODE}" == "warm" ]]; then
    PREV_CYCLE=$((UN_REAL_CYCLE_INDEX - 1))
    export UN_WARM_START_FILE="results/real_warm_continuation_ctid/${MODEL_ID}/anchors/cycle_$(printf "%04d" ${PREV_CYCLE})_best_params_raw.npz"
    test -f "${UN_WARM_START_FILE}"
else
    PREV_CYCLE=-1
    export UN_WARM_START_FILE=""
fi

# ============================================================
# Data and plotting
# ============================================================
export UN_ID_DOWNSAMPLE_DT=1.0
export UN_CURRENT_SIGN_MODE=auto

export UN_MAKE_PLOTS=True
export UN_SHOW_PLOTS=False
export UN_SAVE_PLOTS=True

export UN_IPRINT=0

# ============================================================
# Fixed theta values
# ============================================================
# These are not estimated. They are baseline values used in:
#   x_n = theta_n0 + x_n_surface
#   x_p = theta_p0 + x_p_surface
export UN_THETA_N0_INIT=0.50
export UN_THETA_P0_INIT=0.60

# ============================================================
# Nominal initial guesses
# ============================================================
export UN_ALPHA_N_INIT=0.0064
export UN_ALPHA_P_INIT=0.0048

# Direct electrolyte A couplings.
# This replaces old K_e/d^2.
export UN_K_DIRECT_INIT=0.05

# Solid input gains.
export UN_G_N_INIT=1.20e-4
export UN_G_P_INIT=1.00e-4

# Direct electrolyte B gains.
# This replaces old geometry/distribution ge idea.
export UN_B_EN_INIT=0.03
export UN_B_EP_INIT=0.03

# ============================================================
# Regularization
# ============================================================
export UN_RHO_X0=1e-8
export UN_RHO_TH=1e-8
export UN_TAU_TH=0.0

# ============================================================
# Optimization and jitter regime
# ============================================================
if [[ "${UN_INIT_MODE}" == "cold" ]]; then
    # Cycle 0 broad search.
    export UN_ADAM_EPOCHS=600
    export UN_ADAM_ETA=2e-3
    export UN_LBFGS_EPOCHS=7000
    export UN_LBFGS_TOL=1e-12
    export UN_LBFGS_MEMORY=30

    export UN_INIT_DYN_JITTER=0.45
    export UN_INIT_GAIN_JITTER=0.45
    export UN_INIT_C_JITTER=0.08
    export UN_INIT_BETA_SCALE=2e-2
    export UN_INIT_D1_CENTER=-0.004
    export UN_INIT_D1_JITTER=0.004

elif [[ "${UN_REAL_CYCLE_INDEX}" -ge 6 && "${UN_REAL_CYCLE_INDEX}" -le 33 ]]; then
    # Rescue region: old run struggled here.
    # Use larger local search radius and more optimizer effort.
    export UN_ADAM_EPOCHS=600
    export UN_ADAM_ETA=1.5e-3
    export UN_LBFGS_EPOCHS=6500
    export UN_LBFGS_TOL=1e-12
    export UN_LBFGS_MEMORY=35

    export UN_WARM_RAW_DYN_JITTER=0.12
    export UN_WARM_RAW_GAIN_JITTER=0.10
    export UN_WARM_BETA_REL_JITTER=0.08
    export UN_WARM_BETA_ABS_JITTER=8e-4
    export UN_WARM_X0_ABS_JITTER=8e-5

elif [[ "${UN_REAL_CYCLE_INDEX}" -ge 1 && "${UN_REAL_CYCLE_INDEX}" -le 5 ]]; then
    # Early transition region.
    export UN_ADAM_EPOCHS=500
    export UN_ADAM_ETA=1.5e-3
    export UN_LBFGS_EPOCHS=5500
    export UN_LBFGS_TOL=1e-12
    export UN_LBFGS_MEMORY=30

    export UN_WARM_RAW_DYN_JITTER=0.07
    export UN_WARM_RAW_GAIN_JITTER=0.06
    export UN_WARM_BETA_REL_JITTER=0.05
    export UN_WARM_BETA_ABS_JITTER=5e-4
    export UN_WARM_X0_ABS_JITTER=5e-5

else
    # Stable continuation region.
    export UN_ADAM_EPOCHS=400
    export UN_ADAM_ETA=1e-3
    export UN_LBFGS_EPOCHS=4500
    export UN_LBFGS_TOL=1e-12
    export UN_LBFGS_MEMORY=30

    export UN_WARM_RAW_DYN_JITTER=0.035
    export UN_WARM_RAW_GAIN_JITTER=0.035
    export UN_WARM_BETA_REL_JITTER=0.025
    export UN_WARM_BETA_ABS_JITTER=2e-4
    export UN_WARM_X0_ABS_JITTER=2e-5
fi

# ============================================================
# Run tag
# ============================================================
export UN_RUN_TAG="warmseq_${MODEL_ID}_cycle_${UN_REAL_CYCLE_INDEX}_${UN_INIT_MODE}_chunk_${CHUNK_INDEX}_${UN_N_MULTISTART}seeds_${UN_SEED0}_to_$((UN_SEED0 + UN_N_MULTISTART - 1))_dt_${UN_ID_DOWNSAMPLE_DT}"

echo "=================================================="
echo "DIRECT K/B WARM-SEQUENCE CT-ID"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL_ID}"
echo "State variant: ${UN_STATE_VARIANT}"
echo "Cycle: ${UN_REAL_CYCLE_INDEX}"
echo "Previous cycle: ${PREV_CYCLE}"
echo "Init mode: ${UN_INIT_MODE}"
echo "Chunk: ${CHUNK_INDEX}"
echo "Seed range: ${UN_SEED0} to $((UN_SEED0 + UN_N_MULTISTART - 1))"
echo "Warm file: ${UN_WARM_START_FILE}"
echo "Run tag: ${UN_RUN_TAG}"
echo "Script: scripts/run_real_cycle_warm_continuation_ctid_kparam.py"
echo "=================================================="

python -u scripts/run_real_cycle_warm_continuation_ctid_kparam.py