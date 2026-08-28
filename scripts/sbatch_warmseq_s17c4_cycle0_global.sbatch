#!/bin/bash
#SBATCH --job-name=wseq_s17_c0
#SBATCH --output=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/wseq_s17_c0_%A_%a.out
#SBATCH --error=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/wseq_s17_c0_%A_%a.err
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu2023
#SBATCH --array=0-49

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

SEEDS_PER_CHUNK=20
BASE_SEED0=1000
CHUNK_INDEX=${SLURM_ARRAY_TASK_ID}

export UN_STATE_VARIANT="S17"
export UN_OUTPUT_CANDIDATE="C4"
export UN_REAL_CYCLE_INDEX=0

export UN_INIT_MODE="cold"
export UN_SEED0=$((BASE_SEED0 + CHUNK_INDEX * SEEDS_PER_CHUNK))
export UN_N_MULTISTART="${SEEDS_PER_CHUNK}"

export UN_ID_DOWNSAMPLE_DT=1.0
export UN_CURRENT_SIGN_MODE=auto

export UN_MAKE_PLOTS=True
export UN_SHOW_PLOTS=False
export UN_SAVE_PLOTS=True

export UN_IPRINT=0
export UN_ADAM_EPOCHS=500
export UN_ADAM_ETA=2e-3
export UN_LBFGS_EPOCHS=5000
export UN_LBFGS_TOL=1e-12
export UN_LBFGS_MEMORY=30

export UN_INIT_DYN_JITTER=0.25
export UN_INIT_GAIN_JITTER=0.25
export UN_INIT_THETA_JITTER=0.006
export UN_INIT_C_JITTER=0.05
export UN_INIT_BETA_SCALE=1e-2
export UN_INIT_D1_CENTER=-0.004
export UN_INIT_D1_JITTER=0.002
export UN_INIT_E_SCALE=1e-2

export UN_RUN_TAG="warmseq_S17_C4_cycle_0_cold_chunk_${CHUNK_INDEX}_${UN_N_MULTISTART}seeds_${UN_SEED0}_to_$((UN_SEED0 + UN_N_MULTISTART - 1))_dt_${UN_ID_DOWNSAMPLE_DT}"

echo "S17_C4 cycle 0 global search, chunk ${CHUNK_INDEX}"
python -u scripts/run_real_cycle_warm_continuation_ctid.py