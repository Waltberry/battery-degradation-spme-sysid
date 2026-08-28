#!/bin/bash
set -euo pipefail

cd /home/onyero.ofuzim/projects/battery-degradation-spme-sysid

echo "Submitting S7_C4K and S17_C4K warm-continuation chains..."
echo "Using direct electrolyte A couplings and direct electrolyte B gains."
echo

submit_model () {
    local STATE_VARIANT="$1"
    local MODEL_ID="${STATE_VARIANT}_C4K"

    echo "=================================================="
    echo "Submitting ${MODEL_ID}"
    echo "=================================================="

    jid0=$(sbatch --parsable \
        --export=ALL,UN_STATE_VARIANT=${STATE_VARIANT},UN_REAL_CYCLE_INDEX=0,UN_INIT_MODE=cold \
        scripts/sbatch_warmseq_c4k_cycle_array.sbatch)

    echo "${MODEL_ID} cycle 0 cold array job: ${jid0}"

    jidc=$(sbatch --parsable \
        --dependency=afterany:${jid0} \
        --export=ALL,UN_MODEL_ID=${MODEL_ID},UN_REAL_CYCLE_INDEX=0 \
        scripts/sbatch_warmseq_s7c4_combine.sbatch)

    echo "${MODEL_ID} cycle 0 combine/anchor job: ${jidc}"

    prev_combine=${jidc}

    for cycle in $(seq 1 99); do
        jid=$(sbatch --parsable \
            --dependency=afterok:${prev_combine} \
            --export=ALL,UN_STATE_VARIANT=${STATE_VARIANT},UN_REAL_CYCLE_INDEX=${cycle},UN_INIT_MODE=warm \
            scripts/sbatch_warmseq_c4k_cycle_array.sbatch)

        echo "${MODEL_ID} cycle ${cycle} warm array job: ${jid}"

        jidc=$(sbatch --parsable \
            --dependency=afterany:${jid} \
            --export=ALL,UN_MODEL_ID=${MODEL_ID},UN_REAL_CYCLE_INDEX=${cycle} \
            scripts/sbatch_warmseq_s7c4_combine.sbatch)

        echo "${MODEL_ID} cycle ${cycle} combine/anchor job: ${jidc}"

        prev_combine=${jidc}
    done

    echo "${MODEL_ID} final combine job: ${prev_combine}"
    echo
}

submit_model "S7"
submit_model "S17"

echo "Submitted both S7_C4K and S17_C4K through cycle 99."