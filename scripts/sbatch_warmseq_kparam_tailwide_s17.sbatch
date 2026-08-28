#!/bin/bash
#SBATCH --job-name=s17_tailwide
#SBATCH --output=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/s17_tailwide_%A_%a.out
#SBATCH --error=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/s17_tailwide_%A_%a.err
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

# This sbatch is intentionally S17-only for the difficult tail region.
export UN_STATE_VARIANT="S17"
export UN_OUTPUT_CANDIDATE="C4"
export UN_INIT_MODE="warm"

: "${UN_REAL_CYCLE_INDEX:?Need UN_REAL_CYCLE_INDEX}"

MODEL_ID="S17_C4K"
CHUNK_INDEX=${SLURM_ARRAY_TASK_ID}
PREV_CYCLE=$((UN_REAL_CYCLE_INDEX - 1))

# -------------------------------------------------------------------
# Wider search for ugly tail cycles.
#
# Default:
#   20 chunks x 10 seeds = 200 fits per cycle
#
# Override if needed:
#   UN_TAIL_SEEDS_PER_CHUNK=20
#   --array=0-19 gives 400 fits per cycle
# -------------------------------------------------------------------
SEEDS_PER_CHUNK=${UN_TAIL_SEEDS_PER_CHUNK:-10}
BASE_SEED0=${UN_TAIL_BASE_SEED0:-9000}

export UN_SEED0=$((BASE_SEED0 + UN_REAL_CYCLE_INDEX * 1000 + CHUNK_INDEX * SEEDS_PER_CHUNK))
export UN_N_MULTISTART="${SEEDS_PER_CHUNK}"

export UN_WARM_START_FILE="results/real_warm_continuation_ctid/${MODEL_ID}/anchors/cycle_$(printf "%04d" ${PREV_CYCLE})_best_params_raw.npz"

if [ ! -f "${UN_WARM_START_FILE}" ]; then
    echo "ERROR: Missing warm-start anchor:"
    echo "  ${UN_WARM_START_FILE}"
    exit 1
fi

# -------------------------------------------------------------------
# Optimizer settings.
# More exploration than normal warm continuation.
# -------------------------------------------------------------------
export UN_ADAM_EPOCHS=${UN_ADAM_EPOCHS:-700}
export UN_ADAM_ETA=${UN_ADAM_ETA:-8e-4}
export UN_LBFGS_EPOCHS=${UN_LBFGS_EPOCHS:-6500}
export UN_LBFGS_TOL=${UN_LBFGS_TOL:-1e-12}
export UN_LBFGS_MEMORY=${UN_LBFGS_MEMORY:-35}

# -------------------------------------------------------------------
# Wide warm-start jitter for tail cycles.
#
# Old >40-cycle settings were:
#   dyn 0.03, gain 0.03, beta_rel 0.02, beta_abs 1e-4, x0_abs 1e-5
#
# These are intentionally much wider.
# -------------------------------------------------------------------
export UN_WARM_RAW_DYN_JITTER=${UN_WARM_RAW_DYN_JITTER:-0.20}
export UN_WARM_RAW_GAIN_JITTER=${UN_WARM_RAW_GAIN_JITTER:-0.18}
export UN_WARM_BETA_REL_JITTER=${UN_WARM_BETA_REL_JITTER:-0.15}
export UN_WARM_BETA_ABS_JITTER=${UN_WARM_BETA_ABS_JITTER:-2e-3}
export UN_WARM_X0_ABS_JITTER=${UN_WARM_X0_ABS_JITTER:-2e-4}

# Keep data settings consistent.
export UN_ID_DOWNSAMPLE_DT=${UN_ID_DOWNSAMPLE_DT:-1.0}
export UN_CURRENT_SIGN_MODE=${UN_CURRENT_SIGN_MODE:-auto}

export UN_MAKE_PLOTS=True
export UN_SHOW_PLOTS=False
export UN_SAVE_PLOTS=True

export UN_IPRINT=0

export UN_RUN_TAG="warmseq_${MODEL_ID}_cycle_${UN_REAL_CYCLE_INDEX}_tailwide_chunk_${CHUNK_INDEX}_${UN_N_MULTISTART}seeds_${UN_SEED0}_to_$((UN_SEED0 + UN_N_MULTISTART - 1))_dt_${UN_ID_DOWNSAMPLE_DT}"

echo "=================================================="
echo "S17 DIRECT-K/B TAIL-WIDE WARM-CONTINUATION JOB"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL_ID}"
echo "Cycle: ${UN_REAL_CYCLE_INDEX}"
echo "Previous cycle: ${PREV_CYCLE}"
echo "Seed range: ${UN_SEED0} to $((UN_SEED0 + UN_N_MULTISTART - 1))"
echo "Warm file: ${UN_WARM_START_FILE}"
echo "RUN_TAG: ${UN_RUN_TAG}"
echo
echo "Jitter settings:"
echo "  UN_WARM_RAW_DYN_JITTER=${UN_WARM_RAW_DYN_JITTER}"
echo "  UN_WARM_RAW_GAIN_JITTER=${UN_WARM_RAW_GAIN_JITTER}"
echo "  UN_WARM_BETA_REL_JITTER=${UN_WARM_BETA_REL_JITTER}"
echo "  UN_WARM_BETA_ABS_JITTER=${UN_WARM_BETA_ABS_JITTER}"
echo "  UN_WARM_X0_ABS_JITTER=${UN_WARM_X0_ABS_JITTER}"
echo
echo "Optimizer:"
echo "  UN_ADAM_EPOCHS=${UN_ADAM_EPOCHS}"
echo "  UN_ADAM_ETA=${UN_ADAM_ETA}"
echo "  UN_LBFGS_EPOCHS=${UN_LBFGS_EPOCHS}"
echo "=================================================="

python -u scripts/run_real_cycle_warm_continuation_ctid_kparam.py