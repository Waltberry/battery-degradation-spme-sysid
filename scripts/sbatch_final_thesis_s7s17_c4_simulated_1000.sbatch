#!/bin/bash
#SBATCH --job-name=final_s7s17_c4_sim
##SBATCH --account=def-yourPI   # uncomment and replace if your cluster requires an account
#SBATCH --partition=cpu2023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --array=0-19%4
#SBATCH --output=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/final_s7s17_c4_sim_%A_%a.out
#SBATCH --error=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/final_s7s17_c4_sim_%A_%a.err

set -euo pipefail

PROJECT_DIR="/home/onyero.ofuzim/projects/battery-degradation-spme-sysid"
cd "${PROJECT_DIR}"

mkdir -p results/logs

# Activate your JAX / jax-sysid environment.
# Keep these fallbacks so the script works whether your venv is ~/jaxsys or ~/venvs/jaxsys.
if [ -f "${HOME}/jaxsys/bin/activate" ]; then
    source "${HOME}/jaxsys/bin/activate"
elif [ -f "${HOME}/venvs/jaxsys/bin/activate" ]; then
    source "${HOME}/venvs/jaxsys/bin/activate"
elif [ -f "${PROJECT_DIR}/jaxsys/bin/activate" ]; then
    source "${PROJECT_DIR}/jaxsys/bin/activate"
else
    echo "[warn] Could not find jaxsys virtual environment automatically. Using current Python environment."
fi

export JAX_ENABLE_X64=True
export MPLBACKEND=Agg
export N_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"

# ------------------------------------------------------------
# Array layout:
#   tasks 0-9   -> S7_C4 chunks 0-9
#   tasks 10-19 -> S17_C4 chunks 0-9
# Each chunk runs 100 seeds.
# Total: 2 models x 10 chunks x 100 seeds = 2000 fits.
# ------------------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}
CHUNKS_PER_MODEL=10
SEEDS_PER_CHUNK=100
BASE_SEED=200

if [ "${TASK_ID}" -lt "${CHUNKS_PER_MODEL}" ]; then
    STATE_VARIANT="S7"
    CHUNK_INDEX=${TASK_ID}
else
    STATE_VARIANT="S17"
    CHUNK_INDEX=$((TASK_ID - CHUNKS_PER_MODEL))
fi

SEED0=$((BASE_SEED + CHUNK_INDEX * SEEDS_PER_CHUNK))
SEED1=$((SEED0 + SEEDS_PER_CHUNK - 1))

export UN_STATE_VARIANT="${STATE_VARIANT}"
export UN_OUTPUT_CANDIDATE="C4"

# Default: matching synthetic truth for exact verification:
#   S7 chunks use S7-generated data; S17 chunks use S17-generated data.
# If you want BOTH models to fit the same S17-generated synthetic dataset, change this to:
#   export UN_TRUTH_STATE_VARIANT="S17"
export UN_TRUTH_STATE_VARIANT="match"

export UN_CHUNK_INDEX="${CHUNK_INDEX}"
export UN_SEED0="${SEED0}"
export UN_N_MULTISTART="${SEEDS_PER_CHUNK}"

# Synthetic data settings
export UN_SIM_DT="0.1"
export UN_SIM_T_END="500.0"
export UN_ID_DOWNSAMPLE_DT="1.0"
export UN_INPUT_PROFILE="step"
export UN_T_STEP="90.0"
export UN_I_BEFORE="0.0"
export UN_I_AFTER="2.0"
export UN_ADD_NOISE="False"
export UN_NOISE_STD="1e-4"

# Optimizer settings
export UN_ADAM_EPOCHS="500"
export UN_ADAM_ETA="2e-3"
export UN_LBFGS_EPOCHS="8000"
export UN_LBFGS_TOL="1e-12"
export UN_LBFGS_MEMORY="30"
export UN_IPRINT="0"

# Plot settings for chunk-level quick checks.
# Set False to reduce I/O during full production runs.
export UN_MAKE_PLOTS="True"
export UN_SAVE_PLOTS="True"
export UN_SHOW_PLOTS="False"
export UN_HIST_BINS="1000"

MODEL_ID="${STATE_VARIANT}_C4"
export UN_RUN_TAG="final_thesis_${MODEL_ID}_chunk_${CHUNK_INDEX}_${SEEDS_PER_CHUNK}seeds_${SEED0}_to_${SEED1}_truth_${UN_TRUTH_STATE_VARIANT}_tf_${UN_SIM_T_END}_dt_${UN_ID_DOWNSAMPLE_DT}"

echo "============================================================"
echo "Final thesis fully simulated S7/S17 C4 CT-ID"
echo "Task ID:        ${TASK_ID}"
echo "Model:          ${MODEL_ID}"
echo "Truth setting:  ${UN_TRUTH_STATE_VARIANT}"
echo "Chunk index:    ${CHUNK_INDEX}"
echo "Seeds:          ${SEED0} to ${SEED1}"
echo "Run tag:        ${UN_RUN_TAG}"
echo "Python:         $(which python)"
echo "============================================================"

python run_final_thesis_s7s17_c4_simulated_chunk.py
