#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/onyero.ofuzim/projects/battery-degradation-spme-sysid"

cd "${PROJECT_DIR}"

START_CYCLE=${START_CYCLE:-5}
END_CYCLE=${END_CYCLE:-99}

echo "=================================================="
echo "Submitting DIRECT-K/B warm-continuation continuation"
echo "Models: S7_C4K and S17_C4K"
echo "Cycles: ${START_CYCLE} to ${END_CYCLE}"
echo "Warm cycles: 10 chunks x 10 seeds = 100 fits per cycle"
echo "=================================================="
echo

# Basic script checks.
test -f scripts/sbatch_warmseq_kparam_cycle.sbatch
test -f scripts/sbatch_combine_kparam_cycle.sbatch
test -f scripts/run_real_cycle_warm_continuation_ctid_kparam.py
test -f scripts/combine_real_cycle_warm_continuation_ctid_kparam.py

for STATE in S7 S17; do
    MODEL_ID="${STATE}_C4K"

    echo
    echo "=================================================="
    echo "Preparing ${MODEL_ID}"
    echo "=================================================="

    # The first cycle in this continuation must warm-start from previous cycle anchor.
    PREV_START=$((START_CYCLE - 1))
    START_ANCHOR="results/real_warm_continuation_ctid/${MODEL_ID}/anchors/cycle_$(printf "%04d" ${PREV_START})_best_params_raw.npz"

    if [ ! -f "${START_ANCHOR}" ]; then
        echo "ERROR: Missing start anchor for ${MODEL_ID}:"
        echo "  ${START_ANCHOR}"
        echo
        echo "You need this before submitting cycle ${START_CYCLE}."
        exit 1
    fi

    echo "Found start anchor:"
    echo "  ${START_ANCHOR}"

    prev_combine=""

    for cycle in $(seq "${START_CYCLE}" "${END_CYCLE}"); do
        PREV_CYCLE=$((cycle - 1))
        REQUIRED_ANCHOR="results/real_warm_continuation_ctid/${MODEL_ID}/anchors/cycle_$(printf "%04d" ${PREV_CYCLE})_best_params_raw.npz"

        if [ "${cycle}" -eq "${START_CYCLE}" ]; then
            # For the first continuation cycle, the previous anchor already exists.
            test -f "${REQUIRED_ANCHOR}"

            jid=$(sbatch --parsable \
                --array=0-9 \
                --job-name="k_${STATE}_c${cycle}" \
                --export=ALL,UN_STATE_VARIANT=${STATE},UN_REAL_CYCLE_INDEX=${cycle},UN_INIT_MODE=warm \
                scripts/sbatch_warmseq_kparam_cycle.sbatch)
        else
            # Later cycles depend on the previous cycle's combine job.
            jid=$(sbatch --parsable \
                --dependency=afterok:${prev_combine} \
                --array=0-9 \
                --job-name="k_${STATE}_c${cycle}" \
                --export=ALL,UN_STATE_VARIANT=${STATE},UN_REAL_CYCLE_INDEX=${cycle},UN_INIT_MODE=warm \
                scripts/sbatch_warmseq_kparam_cycle.sbatch)
        fi

        echo "${MODEL_ID} cycle ${cycle} warm array job: ${jid}"

        jidc=$(sbatch --parsable \
            --dependency=afterany:${jid} \
            --job-name="kc_${STATE}_c${cycle}" \
            --export=ALL,UN_MODEL_ID=${MODEL_ID},UN_REAL_CYCLE_INDEX=${cycle} \
            scripts/sbatch_combine_kparam_cycle.sbatch)

        echo "${MODEL_ID} cycle ${cycle} combine job: ${jidc}"

        prev_combine="${jidc}"
    done

    echo
    echo "${MODEL_ID} final submitted combine job: ${prev_combine}"
done

echo
echo "=================================================="
echo "Submitted continuation for S7_C4K and S17_C4K."
echo "Cycles submitted: ${START_CYCLE} to ${END_CYCLE}"
echo "=================================================="