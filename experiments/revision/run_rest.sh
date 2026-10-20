#!/bin/bash
cd "$(dirname "$0")"
export SSL_CERT_FILE=/root/.ccr/ca-bundle.crt
while pgrep -f run_extract.sh > /dev/null; do sleep 20; done
python3 inference_timing.py > timing.log 2>&1
python3 csiro_cv.py > cv.log 2>&1
( cd ../browse_eval && TARS_BACKBONE=b0 python3 extract_features.py > ../revision/bf_b0.log 2>&1 && TARS_BACKBONE=b0 python3 extract_background.py > ../revision/bg_b0.log 2>&1 )
python3 browse_b0_vs_b4.py > browse_cmp.log 2>&1
echo ALLDONE > rest.done
