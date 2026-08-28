#!/bin/bash
set -euo pipefail

echo "Recent S17 tail-wide accounting:"
sacct -S now-2days -u "$USER" \
  --format=JobID,JobName%25,State,ExitCode,Elapsed,MaxRSS,ReqMem,NodeList%20 \
  | grep -E "tw_S17|twc_S17|JobID" || true

echo
echo "Recent tail-wide error logs:"
ls -lt results/logs/s17_tailwide_*.err 2>/dev/null | head -20 || true

echo
echo "Recent combine error logs:"
ls -lt results/logs/wseq_kcomb_*.err 2>/dev/null | head -20 || true
