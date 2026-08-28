#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/onyero.ofuzim/projects/battery-degradation-spme-sysid"
VENV_DIR="/home/onyero.ofuzim/venvs/jaxsys"

cd "$PROJECT_DIR"

source "$VENV_DIR/bin/activate"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=16"
export MPLBACKEND=Agg

export UN_REAL_CYCLE_INDEX=0
export UN_ID_DOWNSAMPLE_DT=1.0
export UN_CURRENT_SIGN_MODE=auto

export UN_N_MULTISTART=1
export UN_HIST_BINS=100
export UN_MAKE_PLOTS=True
export UN_SHOW_PLOTS=False
export UN_SAVE_PLOTS=True

export UN_IPRINT=0
export UN_ADAM_EPOCHS=500
export UN_ADAM_ETA=2e-3
export UN_LBFGS_EPOCHS=5000
export UN_LBFGS_TOL=1e-12
export UN_LBFGS_MEMORY=30

# ---------------------------------------------------------
# Top 10 models from combined_real_cycle_best_runs.csv
# Format:
#   STATE CANDIDATE SEED
# ---------------------------------------------------------
TOP_MODELS=(
  "S17 C4 265"
  "S14 C1 237"
  "S17 C2 277"
  "S17 C3 250"
  "S14 C4 242"
  "S17 C1 217"
  "S14 C3 200"
  "S14 C2 260"
  "S7  C4 200"
  "S12 C4 237"
)

for item in "${TOP_MODELS[@]}"; do
    read -r STATE CAND SEED <<< "$item"

    export UN_STATE_VARIANTS="$STATE"
    export UN_OUTPUT_CANDIDATES="$CAND"
    export UN_SEED0="$SEED"

    export UN_RUN_TAG="quick_top10_response_${STATE}_${CAND}_seed_${SEED}"

    echo "=================================================="
    echo "RERUNNING TOP MODEL"
    echo "State: ${UN_STATE_VARIANTS}"
    echo "Candidate: ${UN_OUTPUT_CANDIDATES}"
    echo "Seed: ${UN_SEED0}"
    echo "Run tag: ${UN_RUN_TAG}"
    echo "Start: $(date)"
    echo "=================================================="

    python -u scripts/run_real_cycle_state_order_ctid_grid.py

    echo "Finished ${UN_RUN_TAG} at $(date)"
done

echo "=================================================="
echo "TOP-10 STEP-RESPONSE RERUN COMPLETE"
echo "=================================================="