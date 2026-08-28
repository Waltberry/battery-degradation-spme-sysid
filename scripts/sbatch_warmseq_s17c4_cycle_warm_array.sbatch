#!/bin/bash
#SBATCH --job-name=wseq_s17_warm
#SBATCH --output=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/wseq_s17_warm_%A_%a.out
#SBATCH --error=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/wseq_s17_warm_%A_%a.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu2023
#SBATCH --array=0-4

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

: "${UN_REAL_CYCLE_INDEX:?Need UN_REAL_CYCLE_INDEX}"

SEEDS_PER_CHUNK=20
BASE_SEED0=3000
CHUNK_INDEX=${SLURM_ARRAY_TASK_ID}
PREV_CYCLE=$((UN_REAL_CYCLE_INDEX - 1))

export UN_STATE_VARIANT="S17"
export UN_OUTPUT_CANDIDATE="C4"

export UN_INIT_MODE="warm"
export UN_WARM_START_FILE="results/real_warm_continuation_ctid/S17_C4/anchors/cycle_$(printf "%04d" ${PREV_CYCLE})_best_params_raw.npz"

export UN_SEED0=$((BASE_SEED0 + UN_REAL_CYCLE_INDEX * 100 + CHUNK_INDEX * SEEDS_PER_CHUNK))
export UN_N_MULTISTART="${SEEDS_PER_CHUNK}"

export UN_ID_DOWNSAMPLE_DT=1.0
export UN_CURRENT_SIGN_MODE=auto

export UN_MAKE_PLOTS=True
export UN_SHOW_PLOTS=False
export UN_SAVE_PLOTS=True

export UN_IPRINT=0
export UN_ADAM_EPOCHS=400
export UN_ADAM_ETA=1e-3
export UN_LBFGS_EPOCHS=4000
export UN_LBFGS_TOL=1e-12
export UN_LBFGS_MEMORY=30

export UN_WARM_RAW_DYN_JITTER=0.025
export UN_WARM_RAW_GAIN_JITTER=0.025
export UN_WARM_RAW_THETA_JITTER=0.008
export UN_WARM_BETA_REL_JITTER=0.015
export UN_WARM_BETA_ABS_JITTER=1e-4
export UN_WARM_X0_ABS_JITTER=1e-5

export UN_RUN_TAG="warmseq_S17_C4_cycle_${UN_REAL_CYCLE_INDEX}_warm_chunk_${CHUNK_INDEX}_${UN_N_MULTISTART}seeds_${UN_SEED0}_to_$((UN_SEED0 + UN_N_MULTISTART - 1))_dt_${UN_ID_DOWNSAMPLE_DT}"

echo "S17_C4 warm cycle ${UN_REAL_CYCLE_INDEX}, chunk ${CHUNK_INDEX}"
echo "Warm file: ${UN_WARM_START_FILE}"

test -f "${UN_WARM_START_FILE}"

python -u scripts/run_real_cycle_warm_continuation_ctid.py