#!/usr/bin/env bash

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-data/data_neuro_like_90_100}"
MIN_VARS="${MIN_VARS:-90}"
MAX_VARS="${MAX_VARS:-100}"
CV_RATIO="${CV_RATIO:-6.0}"
MAX_CLAUSES="${MAX_CLAUSES:-600}"
K="${K:-3}"
CLAUSE_LEN_MODE="${CLAUSE_LEN_MODE:-neurosat}"
MIN_CLAUSE_LEN="${MIN_CLAUSE_LEN:-3}"
MAX_CLAUSE_LEN="${MAX_CLAUSE_LEN:-30}"
CLAUSE_LEN_PROFILE_DIR="${CLAUSE_LEN_PROFILE_DIR:-data/neuro_data/train}"

export OUTPUT_DIR
export MIN_VARS
export MAX_VARS
export CV_RATIO
export MAX_CLAUSES
export K
export CLAUSE_LEN_MODE
export MIN_CLAUSE_LEN
export MAX_CLAUSE_LEN
export CLAUSE_LEN_PROFILE_DIR

bash gen_data.sh
