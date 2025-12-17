#!/bin/bash

./init.sh
./build_clang_for_training.sh
./build_clang_for_corpus.sh
./extract_corpus.sh
./generate_default_trace.sh
./generate_vocab.sh
./train_with_es.sh