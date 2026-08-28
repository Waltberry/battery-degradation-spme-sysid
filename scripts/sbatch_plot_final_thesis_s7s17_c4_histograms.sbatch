#!/bin/bash
#SBATCH --job-name=plot_s7s17_c4_hist
##SBATCH --account=def-yourPI   # uncomment and replace if your cluster requires an account
#SBATCH --partition=cpu2023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/plot_s7s17_c4_hist_%j.out
#SBATCH --error=/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/results/logs/plot_s7s17_c4_hist_%j.err

set -euo pipefail

PROJECT_DIR="/home/onyero.ofuzim/projects/battery-degradation-spme-sysid"
cd "${PROJECT_DIR}"
mkdir -p results/logs

if [ -f "${HOME}/jaxsys/bin/activate" ]; then
    source "${HOME}/jaxsys/bin/activate"
elif [ -f "${HOME}/venvs/jaxsys/bin/activate" ]; then
    source "${HOME}/venvs/jaxsys/bin/activate"
elif [ -f "${PROJECT_DIR}/jaxsys/bin/activate" ]; then
    source "${PROJECT_DIR}/jaxsys/bin/activate"
else
    echo "[warn] Could not find jaxsys virtual environment automatically. Using current Python environment."
fi

export MPLBACKEND=Agg
export UN_HIST_BINS="1000"
export UN_EXPECTED_RUNS_PER_MODEL="1000"
export UN_INCLUDE_BETA_PARAMETERS="False"
export UN_INCLUDE_EDGE_K_PARAMETERS="True"

python plot_final_thesis_s7s17_c4_parameter_histograms_1000bins.py
