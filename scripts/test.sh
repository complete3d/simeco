#!/usr/bin/env bash

if [ "$DEBUG" = "1" ]; then
  set -x
fi

export LOCAL_RANK=0

python main.py "$@"

# Usage:
# CUDA_VISIBLE_DEVICES=0 ./scripts/test.sh