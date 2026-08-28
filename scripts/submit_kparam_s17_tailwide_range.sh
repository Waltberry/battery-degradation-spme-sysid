#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/onyero.ofuzim/projects/battery-degradation-spme-sysid"

cd "${PROJECT_DIR}"

MODEL_ID="S17_C4K"

START_CYCLE=${START_CYCLE:-100}
END_CYCLE=${END_CYCLE:?Need END_CYCLE. Example: END_CYCLE=105 ./scripts/submit_kparam_s17_tailwide_range.sh}

# Array size.
# Default 20 chunks x 10 seeds = 200 fits per cycle.
ARRAY_START=${ARRAY_START:-0}
ARRAY_END=${ARRAY_END:-19}

echo "=================================================="
echo "Submitting S17_C4K tail-wide continuation"
echo "Cycles: ${START_CYCLE} to ${END_CYCLE}"
echo "Array: ${ARRAY_START}-${ARRAY_END}"
echo "Default per cycle: 20 chunks x 10 seeds = 200 fits"
echo "=================================================="
echo

test -f scripts/sbatch_warmseq_kparam_tailwide_s17.sbatch
test -f scripts/sbatch_combine_kparam_cycle.sbatch
test -f scripts/run_real_cycle_warm_continuation_ctid_kparam.py
test -f scripts/combine_real_cycle_warm_continuation_ctid_kparam.py

if [ "${END_CYCLE}" -lt "${START_CYCLE}" ]; then
    echo "ERROR: END_CYCLE=${END_CYCLE} is less than START_CYCLE=${START_CYCLE}"
    exit 1
fi

PREV_START=$((START_CYCLE - 1))
START_ANCHOR="results/real_warm_continuation_ctid/${MODEL_ID}/anchors/cycle_$(printf "%04d" ${PREV_START})_best_params_raw.npz"

if [ ! -f "${START_ANCHOR}" ]; then
    echo "ERROR: Missing start anchor:"
    echo "  ${START_ANCHOR}"
    exit 1
fi

echo "Found start anchor:"
echo "  ${START_ANCHOR}"
echo

prev_combine=""

for cycle in $(seq "${START_CYCLE}" "${END_CYCLE}"); do
    PREV_CYCLE=$((cycle - 1))
    REQUIRED_ANCHOR="results/real_warm_continuation_ctid/${MODEL_ID}/anchors/cycle_$(printf "%04d" ${PREV_CYCLE})_best_params_raw.npz"

    if [ "${cycle}" -eq "${START_CYCLE}" ]; then
        test -f "${REQUIRED_ANCHOR}"

        jid=$(sbatch --parsable \
            --array=${ARRAY_START}-${ARRAY_END} \
            --job-name="tw_S17_c${cycle}" \
            --export=ALL,UN_REAL_CYCLE_INDEX=${cycle} \
            scripts/sbatch_warmseq_kparam_tailwide_s17.sbatch)
    else
        jid=$(sbatch --parsable \
            --dependency=afterok:${prev_combine} \
            --array=${ARRAY_START}-${ARRAY_END} \
            --job-name="tw_S17_c${cycle}" \
            --export=ALL,UN_REAL_CYCLE_INDEX=${cycle} \
            scripts/sbatch_warmseq_kparam_tailwide_s17.sbatch)
    fi

    echo "S17_C4K cycle ${cycle} tail-wide array job: ${jid}"

    jidc=$(sbatch --parsable \
        --dependency=afterany:${jid} \
        --job-name="twc_S17_c${cycle}" \
        --export=ALL,UN_MODEL_ID=${MODEL_ID},UN_REAL_CYCLE_INDEX=${cycle} \
        scripts/sbatch_combine_kparam_cycle.sbatch)

    echo "S17_C4K cycle ${cycle} combine job: ${jidc}"

    prev_combine="${jidc}"
done

echo
echo "=================================================="
echo "Submitted S17_C4K tail-wide continuation."
echo "Cycles submitted: ${START_CYCLE} to ${END_CYCLE}"
echo "Final combine job: ${prev_combine}"
echo "=================================================="