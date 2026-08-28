#!/bin/bash
set -euo pipefail

cd /home/onyero.ofuzim/projects/battery-degradation-spme-sysid

echo "Submitting DIRECT-K/B warm-continuation 5-cycle test for S7_C4K and S17_C4K"
echo "Cycles: 0,1,2,3,4"
echo "Cycle 0: 50 chunks x 20 seeds = 1000 fits"
echo "Warm cycles: 10 chunks x 10 seeds = 100 fits per cycle"
echo

for STATE in S7 S17; do
    MODEL_ID="${STATE}_C4K"

    echo "=================================================="
    echo "Submitting ${MODEL_ID}"
    echo "=================================================="

    # Cycle 0 cold global search.
    jid0=$(sbatch --parsable \
        --array=0-49 \
        --job-name="k_${STATE}_c0" \
        --export=ALL,UN_STATE_VARIANT=${STATE},UN_REAL_CYCLE_INDEX=0,UN_INIT_MODE=cold \
        scripts/sbatch_warmseq_kparam_cycle.sbatch)

    echo "${MODEL_ID} cycle 0 cold array job: ${jid0}"

    # Combine cycle 0 and create anchor.
    jidc=$(sbatch --parsable \
        --dependency=afterany:${jid0} \
        --job-name="kc_${STATE}_c0" \
        --export=ALL,UN_MODEL_ID=${MODEL_ID},UN_REAL_CYCLE_INDEX=0 \
        scripts/sbatch_combine_kparam_cycle.sbatch)

    echo "${MODEL_ID} cycle 0 combine job: ${jidc}"

    prev_combine=${jidc}

    # Warm cycles 1--4.
    for cycle in $(seq 1 4); do
        jid=$(sbatch --parsable \
            --dependency=afterok:${prev_combine} \
            --array=0-9 \
            --job-name="k_${STATE}_c${cycle}" \
            --export=ALL,UN_STATE_VARIANT=${STATE},UN_REAL_CYCLE_INDEX=${cycle},UN_INIT_MODE=warm \
            scripts/sbatch_warmseq_kparam_cycle.sbatch)

        echo "${MODEL_ID} cycle ${cycle} warm array job: ${jid}"

        jidc=$(sbatch --parsable \
            --dependency=afterany:${jid} \
            --job-name="kc_${STATE}_c${cycle}" \
            --export=ALL,UN_MODEL_ID=${MODEL_ID},UN_REAL_CYCLE_INDEX=${cycle} \
            scripts/sbatch_combine_kparam_cycle.sbatch)

        echo "${MODEL_ID} cycle ${cycle} combine job: ${jidc}"

        prev_combine=${jidc}
    done

    echo "${MODEL_ID} final 5-cycle combine job: ${prev_combine}"
    echo
done

echo "Submitted S7_C4K and S17_C4K 5-cycle test."