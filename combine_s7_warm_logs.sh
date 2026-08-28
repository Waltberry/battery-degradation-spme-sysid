#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/onyero.ofuzim/projects/battery-degradation-spme-sysid"
LOG_DIR="${PROJECT_DIR}/results/logs"
OUT_TXT="${PROJECT_DIR}/s7_warm_all_out_logs_combined.txt"
ERR_TXT="${PROJECT_DIR}/s7_warm_all_err_logs_combined.txt"
BOTH_TXT="${PROJECT_DIR}/s7_warm_all_logs_combined_out_and_err.txt"

cd "$PROJECT_DIR"

echo "============================================================"
echo "Combining S7 warm-continuation logs"
echo "Project: $PROJECT_DIR"
echo "Log dir: $LOG_DIR"
echo "============================================================"

# ------------------------------------------------------------
# Combine all S7 warm .out files
# Pattern example:
# results/logs/wseq_s7_warm_43487570_0.out
# ------------------------------------------------------------
{
    echo "============================================================"
    echo "COMBINED S7_C4 WARM-CONTINUATION .OUT LOGS"
    echo "Generated: $(date)"
    echo "Pattern: ${LOG_DIR}/wseq_s7_warm_*.out"
    echo "============================================================"
    echo

    mapfile -t out_files < <(find "$LOG_DIR" -maxdepth 1 -type f -name "wseq_s7_warm_*.out" | sort -V)

    echo "Number of .out files found: ${#out_files[@]}"
    echo

    if [ "${#out_files[@]}" -eq 0 ]; then
        echo "No S7 warm .out files found."
    else
        for f in "${out_files[@]}"; do
            echo
            echo "################################################################################"
            echo "FILE: $f"
            echo "################################################################################"
            echo
            cat "$f"
            echo
            echo
        done
    fi
} > "$OUT_TXT"

echo "Wrote .out combined log:"
echo "  $OUT_TXT"

# ------------------------------------------------------------
# Combine all S7 warm .err files
# ------------------------------------------------------------
{
    echo "============================================================"
    echo "COMBINED S7_C4 WARM-CONTINUATION .ERR LOGS"
    echo "Generated: $(date)"
    echo "Pattern: ${LOG_DIR}/wseq_s7_warm_*.err"
    echo "============================================================"
    echo

    mapfile -t err_files < <(find "$LOG_DIR" -maxdepth 1 -type f -name "wseq_s7_warm_*.err" | sort -V)

    echo "Number of .err files found: ${#err_files[@]}"
    echo

    if [ "${#err_files[@]}" -eq 0 ]; then
        echo "No S7 warm .err files found."
    else
        for f in "${err_files[@]}"; do
            echo
            echo "################################################################################"
            echo "FILE: $f"
            echo "################################################################################"
            echo
            cat "$f"
            echo
            echo
        done
    fi
} > "$ERR_TXT"

echo "Wrote .err combined log:"
echo "  $ERR_TXT"

# ------------------------------------------------------------
# Combine both .out and .err into one file
# ------------------------------------------------------------
{
    echo "============================================================"
    echo "COMBINED S7_C4 WARM-CONTINUATION ALL LOGS"
    echo "Generated: $(date)"
    echo "Includes .out and .err files"
    echo "============================================================"
    echo

    echo
    echo "############################################################"
    echo "SECTION 1: ALL .OUT FILES"
    echo "############################################################"
    echo
    cat "$OUT_TXT"

    echo
    echo
    echo "############################################################"
    echo "SECTION 2: ALL .ERR FILES"
    echo "############################################################"
    echo
    cat "$ERR_TXT"

} > "$BOTH_TXT"

echo "Wrote combined out+err log:"
echo "  $BOTH_TXT"

echo
echo "Line counts:"
wc -l "$OUT_TXT" "$ERR_TXT" "$BOTH_TXT"

echo
echo "File sizes:"
ls -lh "$OUT_TXT" "$ERR_TXT" "$BOTH_TXT"
