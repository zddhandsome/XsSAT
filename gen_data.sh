#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-}"
OUTPUT_DIR="${OUTPUT_DIR:-data_200}"
NUM_TRAIN="${NUM_TRAIN:-1000}"
NUM_VAL="${NUM_VAL:-1000}"
NUM_TEST="${NUM_TEST:-1000}"
MIN_VARS="${MIN_VARS:-200}"
MAX_VARS="${MAX_VARS:-200}"
CV_RATIO="${CV_RATIO:-4.26}"
MAX_CLAUSES="${MAX_CLAUSES:-1000}"
NUM_WORKERS="${NUM_WORKERS:-24}"
K="${K:-3}"
CLAUSE_LEN_MODE="${CLAUSE_LEN_MODE:-fixed}"
MIN_CLAUSE_LEN="${MIN_CLAUSE_LEN:-2}"
MAX_CLAUSE_LEN="${MAX_CLAUSE_LEN:-8}"
CLAUSE_LEN_PROFILE_DIR="${CLAUSE_LEN_PROFILE_DIR:-neuro_data/train}"

if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "ERROR: python3 not found. Activate your conda/module environment or set PYTHON_BIN=/path/to/python3" >&2
        exit 127
    fi
fi

"${PYTHON_BIN}" --version

EXTRA_ARGS=()
if [[ "${CLAUSE_LEN_MODE}" == "uniform" ]]; then
    EXTRA_ARGS+=(
        --clause_len_mode uniform
        --min_clause_len "${MIN_CLAUSE_LEN}"
        --max_clause_len "${MAX_CLAUSE_LEN}"
    )
elif [[ "${CLAUSE_LEN_MODE}" == "neurosat" ]]; then
    EXTRA_ARGS+=(
        --clause_len_mode neurosat
        --clause_len_profile_dir "${CLAUSE_LEN_PROFILE_DIR}"
    )
elif [[ "${CLAUSE_LEN_MODE}" != "fixed" ]]; then
    echo "ERROR: CLAUSE_LEN_MODE must be one of: fixed, uniform, neurosat" >&2
    exit 2
fi

"${PYTHON_BIN}" gen_data_packed_fast.py generate-mixed-vars \
    --output_dir "${OUTPUT_DIR}" \
    --num_train "${NUM_TRAIN}" \
    --num_val "${NUM_VAL}" \
    --num_test "${NUM_TEST}" \
    --min_vars "${MIN_VARS}" \
    --max_vars_gen "${MAX_VARS}" \
    --max_vars "${MAX_VARS}" \
    --cv_ratio "${CV_RATIO}" \
    --max_clauses "${MAX_CLAUSES}" \
    --k "${K}" \
    --balanced \
    --num_workers "${NUM_WORKERS}" \
    "${EXTRA_ARGS[@]}"
