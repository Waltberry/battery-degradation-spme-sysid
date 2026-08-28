#!/bin/bash
set -euo pipefail

cd /home/onyero.ofuzim/projects/battery-degradation-spme-sysid

START_CYCLE="${1:-1}"
END_CYCLE="${2:-99}"

echo "Submitting S7_C4 warm-continuation from cycle ${START_CYCLE} to ${END_CYCLE}"

prev_combine=""

for cycle in $(seq "${START_CYCLE}" "${END_CYCLE}"); do
    if [ "${cycle}" -eq "${START_CYCLE}" ]; then
        prev_anchor="results/real_warm_continuation_ctid/S7_C4/anchors/cycle_$(printf "%04d" $((cycle - 1)))_best_params_raw.npz"
        test -f "${prev_anchor}"

        jid=$(sbatch --parsable --export=ALL,UN_REAL_CYCLE_INDEX=${cycle} scripts/sbatch_warmseq_s7c4_cycle_warm_array.sbatch)
    else
        jid=$(sbatch --parsable --dependency=afterok:${prev_combine} --export=ALL,UN_REAL_CYCLE_INDEX=${cycle} scripts/sbatch_warmseq_s7c4_cycle_warm_array.sbatch)
    fi

    echo "Cycle ${cycle} warm array job: ${jid}"

    jidc=$(sbatch --parsable --dependency=afterany:${jid} --export=ALL,UN_REAL_CYCLE_INDEX=${cycle} scripts/sbatch_warmseq_s7c4_combine.sbatch)
    echo "Cycle ${cycle} combine job: ${jidc}"

    prev_combine=${jidc}
done

echo "Submitted S7_C4 chain from cycle ${START_CYCLE} to ${END_CYCLE}."
echo "Final combine job: ${prev_combine}"