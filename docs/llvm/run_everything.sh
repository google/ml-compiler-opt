#!/bin/bash

SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"

SCRIPT_DIR="$(cd -- "$SCRIPT_DIR" && pwd)"
if [[ -z "$SCRIPT_DIR" ]] ; then
    exit 1
fi

"${SCRIPT_DIR}/init.sh"
"${SCRIPT_DIR}/build_clang_for_training.sh"
"${SCRIPT_DIR}/build_clang_for_corpus.sh"
"${SCRIPT_DIR}/extract_corpus.sh"
"${SCRIPT_DIR}/generate_default_trace.sh"
"${SCRIPT_DIR}/generate_vocab.sh"
"${SCRIPT_DIR}/train_with_es.sh"