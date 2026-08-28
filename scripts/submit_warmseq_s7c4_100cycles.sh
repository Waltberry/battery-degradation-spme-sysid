#!/bin/bash
set -euo pipefail

cd /home/onyero.ofuzim/projects/battery-degradation-spme-sysid

echo "Submitting S7_C4 warm-continuation pipeline..."

jid0=$(sbatch --parsable scripts/sbatch_warmseq_s7c4_cycle0_global.sbatch)
echo "Cycle 0 global array job: ${jid0}"

jidc=$(sbatch --parsable --dependency=afterany:${jid0} --export=ALL,UN_REAL_CYCLE_INDEX=0 scripts/sbatch_warmseq_s7c4_combine.sbatch)
echo "Cycle 0 combine job: ${jidc}"

prev_combine=${jidc}

for cycle in $(seq 1 99); do
    jid=$(sbatch --parsable --dependency=afterok:${prev_combine} --export=ALL,UN_REAL_CYCLE_INDEX=${cycle} scripts/sbatch_warmseq_s7c4_cycle_warm_array.sbatch)
    echo "Cycle ${cycle} warm array job: ${jid}"

    jidc=$(sbatch --parsable --dependency=afterany:${jid} --export=ALL,UN_REAL_CYCLE_INDEX=${cycle} scripts/sbatch_warmseq_s7c4_combine.sbatch)
    echo "Cycle ${cycle} combine job: ${jidc}"

    prev_combine=${jidc}
done

echo "Submitted S7_C4 warm-continuation chain through cycle 99."
echo "Final combine job: ${prev_combine}"