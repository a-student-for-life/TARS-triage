#!/bin/bash
cd "$(dirname "$0")"
export SSL_CERT_FILE=/root/.ccr/ca-bundle.crt
python3 extract_csiro_features.py b0 > b0.log 2>&1
python3 extract_csiro_features.py b4 > b4.log 2>&1
