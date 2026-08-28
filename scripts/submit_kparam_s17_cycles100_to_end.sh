#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/onyero.ofuzim/projects/battery-degradation-spme-sysid"
VENV_PY="/home/onyero.ofuzim/venvs/jaxsys/bin/python"

cd "${PROJECT_DIR}"

STATE="S17"
MODEL_ID="S17_C4K"

START_CYCLE=${START_CYCLE:-100}

echo "=================================================="
echo "Submitting DIRECT-K/B warm-continuation tail run"
echo "Model: ${MODEL_ID}"
echo "Default start cycle: ${START_CYCLE}"
echo "This uses cycle $((START_CYCLE - 1)) as the warm-start anchor."
echo "=================================================="
echo

test -f scripts/sbatch_warmseq_kparam_cycle.sbatch
test -f scripts/sbatch_combine_kparam_cycle.sbatch
test -f scripts/run_real_cycle_warm_continuation_ctid_kparam.py
test -f scripts/combine_real_cycle_warm_continuation_ctid_kparam.py
test -f scripts/detect_last_real_discharge_cycle.py

if [ -z "${END_CYCLE:-}" ]; then
    echo "END_CYCLE not provided. Detecting last discharge cycle from raw data..."

    DETECT_LOG=$(mktemp)
    "${VENV_PY}" scripts/detect_last_real_discharge_cycle.py | tee "${DETECT_LOG}"

    END_CYCLE=$(tail -n 1 "${DETECT_LOG}" | tr -d '[:space:]')
    rm -f "${DETECT_LOG}"

    if ! [[ "${END_CYCLE}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Could not parse END_CYCLE from detector output."
        echo "Parsed value: ${END_CYCLE}"
        echo
        echo "Run manually instead, for example:"
        echo "  START_CYCLE=100 END_CYCLE=143 ./scripts/submit_kparam_s17_cycles100_to_end.sh"
        exit 1
    fi
fi

echo
echo "Cycle range:"
echo "  START_CYCLE=${START_CYCLE}"
echo "  END_CYCLE=${END_CYCLE}"
echo

if [ "${END_CYCLE}" -lt "${START_CYCLE}" ]; then
    echo "Nothing to submit."
    echo "END_CYCLE=${END_CYCLE} is less than START_CYCLE=${START_CYCLE}."
    exit 0
fi

PREV_START=$((START_CYCLE - 1))
START_ANCHOR="results/real_warm_continuation_ctid/${MODEL_ID}/anchors/cycle_$(printf "%04d" ${PREV_START})_best_params_raw.npz"

if [ ! -f "${START_ANCHOR}" ]; then
    echo "ERROR: Missing start anchor:"
    echo "  ${START_ANCHOR}"
    echo
    echo "You need this before submitting cycle ${START_CYCLE}."
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
            --array=0-9 \
            --job-name="k_S17_c${cycle}" \
            --export=ALL,UN_STATE_VARIANT=${STATE},UN_REAL_CYCLE_INDEX=${cycle},UN_INIT_MODE=warm \
            scripts/sbatch_warmseq_kparam_cycle.sbatch)
    else
        jid=$(sbatch --parsable \
            --dependency=afterok:${prev_combine} \
            --array=0-9 \
            --job-name="k_S17_c${cycle}" \
            --export=ALL,UN_STATE_VARIANT=${STATE},UN_REAL_CYCLE_INDEX=${cycle},UN_INIT_MODE=warm \
            scripts/sbatch_warmseq_kparam_cycle.sbatch)
    fi

    echo "${MODEL_ID} cycle ${cycle} warm array job: ${jid}"

    jidc=$(sbatch --parsable \
        --dependency=afterany:${jid} \
        --job-name="kc_S17_c${cycle}" \
        --export=ALL,UN_MODEL_ID=${MODEL_ID},UN_REAL_CYCLE_INDEX=${cycle} \
        scripts/sbatch_combine_kparam_cycle.sbatch)

    echo "${MODEL_ID} cycle ${cycle} combine job: ${jidc}"

    prev_combine="${jidc}"
done

echo
echo "=================================================="
echo "Submitted S17_C4K tail continuation."
echo "Cycles submitted: ${START_CYCLE} to ${END_CYCLE}"
echo "Final combine job: ${prev_combine}"
echo "=================================================="