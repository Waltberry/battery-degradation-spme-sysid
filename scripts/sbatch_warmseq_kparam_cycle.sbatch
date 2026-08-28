#!/bin/bash
#SBATCH --job-name=wseq_kparam
#SBATCH --output=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/wseq_kparam_%A_%a.out
#SBATCH --error=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/wseq_kparam_%A_%a.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu2023

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

: "${UN_STATE_VARIANT:?Need UN_STATE_VARIANT=S7 or S17}"
: "${UN_REAL_CYCLE_INDEX:?Need UN_REAL_CYCLE_INDEX}"
: "${UN_INIT_MODE:?Need UN_INIT_MODE=cold or warm}"

export UN_OUTPUT_CANDIDATE="C4"

MODEL_ID="${UN_STATE_VARIANT}_C4K"
CHUNK_INDEX=${SLURM_ARRAY_TASK_ID}

# ------------------------------------------------------------
# Cold cycle 0:
#   50 chunks x 20 seeds = 1000 fits
#
# Warm cycles:
#   10 chunks x 10 seeds = 100 fits
# ------------------------------------------------------------
if [ "${UN_INIT_MODE}" = "cold" ]; then
    SEEDS_PER_CHUNK=20
    BASE_SEED0=1000

    export UN_SEED0=$((BASE_SEED0 + CHUNK_INDEX * SEEDS_PER_CHUNK))
    export UN_N_MULTISTART="${SEEDS_PER_CHUNK}"

    export UN_WARM_START_FILE=""

    export UN_ADAM_EPOCHS=500
    export UN_ADAM_ETA=2e-3
    export UN_LBFGS_EPOCHS=5000
    export UN_LBFGS_TOL=1e-12
    export UN_LBFGS_MEMORY=30

    export UN_INIT_DYN_JITTER=0.35
    export UN_INIT_GAIN_JITTER=0.35
    export UN_INIT_C_JITTER=0.05
    export UN_INIT_BETA_SCALE=1e-2
    export UN_INIT_D1_CENTER=-0.004
    export UN_INIT_D1_JITTER=0.002

elif [ "${UN_INIT_MODE}" = "warm" ]; then
    SEEDS_PER_CHUNK=10
    BASE_SEED0=3000
    PREV_CYCLE=$((UN_REAL_CYCLE_INDEX - 1))

    export UN_SEED0=$((BASE_SEED0 + UN_REAL_CYCLE_INDEX * 100 + CHUNK_INDEX * SEEDS_PER_CHUNK))
    export UN_N_MULTISTART="${SEEDS_PER_CHUNK}"

    export UN_WARM_START_FILE="results/real_warm_continuation_ctid/${MODEL_ID}/anchors/cycle_$(printf "%04d" ${PREV_CYCLE})_best_params_raw.npz"

    test -f "${UN_WARM_START_FILE}"

    export UN_ADAM_EPOCHS=450
    export UN_ADAM_ETA=1e-3
    export UN_LBFGS_EPOCHS=4500
    export UN_LBFGS_TOL=1e-12
    export UN_LBFGS_MEMORY=30

    # Cycles 1--40 get wider warm exploration.
    # This is meant to help the bad-basin region we saw earlier around 6--33.
    if [ "${UN_REAL_CYCLE_INDEX}" -le 40 ]; then
        export UN_WARM_RAW_DYN_JITTER=0.08
        export UN_WARM_RAW_GAIN_JITTER=0.06
        export UN_WARM_BETA_REL_JITTER=0.05
        export UN_WARM_BETA_ABS_JITTER=5e-4
        export UN_WARM_X0_ABS_JITTER=5e-5
    else
        export UN_WARM_RAW_DYN_JITTER=0.03
        export UN_WARM_RAW_GAIN_JITTER=0.03
        export UN_WARM_BETA_REL_JITTER=0.02
        export UN_WARM_BETA_ABS_JITTER=1e-4
        export UN_WARM_X0_ABS_JITTER=1e-5
    fi

else
    echo "UN_INIT_MODE must be cold or warm."
    exit 1
fi

export UN_ID_DOWNSAMPLE_DT=1.0
export UN_CURRENT_SIGN_MODE=auto

export UN_MAKE_PLOTS=True
export UN_SHOW_PLOTS=False
export UN_SAVE_PLOTS=True

export UN_IPRINT=0

export UN_RUN_TAG="warmseq_${MODEL_ID}_cycle_${UN_REAL_CYCLE_INDEX}_${UN_INIT_MODE}_chunk_${CHUNK_INDEX}_${UN_N_MULTISTART}seeds_${UN_SEED0}_to_$((UN_SEED0 + UN_N_MULTISTART - 1))_dt_${UN_ID_DOWNSAMPLE_DT}"

echo "=================================================="
echo "DIRECT-K/B WARM-CONTINUATION JOB"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL_ID}"
echo "State variant: ${UN_STATE_VARIANT}"
echo "Cycle: ${UN_REAL_CYCLE_INDEX}"
echo "Init mode: ${UN_INIT_MODE}"
echo "Seed range: ${UN_SEED0} to $((UN_SEED0 + UN_N_MULTISTART - 1))"
echo "Warm file: ${UN_WARM_START_FILE:-none}"
echo "RUN_TAG: ${UN_RUN_TAG}"
echo "=================================================="

python -u scripts/run_real_cycle_warm_continuation_ctid_kparam.py